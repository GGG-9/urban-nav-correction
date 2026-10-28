# Urban Vehicular Integrated-Navigation Correction with GPU-Parallel Batched Kalman Smoothing and a LoRA-Tuned Time-Series Large Model

Course project for **MS022161 Engineering Big-Data Analysis** (Southeast University).
Topic: *GPU-parallel batched Kalman smoothing + a LoRA-fine-tuned time-series
large model for correcting urban vehicular integrated-navigation trajectories
under GNSS outages.*

This repository is the **GPU part (module 1)** in full, plus the simulated data
pipeline and the baselines it is compared against.

---

## 1. What is implemented

| Stage | File | Status |
|---|---|---|
| Analytic IMU / GNSS simulator, 48 routes x 1800 s @ 100 Hz | `src/data/mini_imu_sim.py`, `src/data/routes.py` | done |
| 15-state error-state KF (numpy reference) | `src/nav/ekf.py` | done |
| Full-state RTS smoother (numpy reference) | `src/nav/ekf.py` | done |
| **Window-parallel batched ESKF + RTS (PyTorch, CUDA)** | `src/hpc/batch_kf.py` | done |
| **Two-pass warm start + overlap-add window stitching** | `src/hpc/batch_kf.py` | done |
| **CPU-serial vs GPU-batched benchmark** | `src/hpc/benchmark.py` | done |
| GPU/CPU parity + boundary-continuity checks | `src/hpc/test_parity.py` | done |
| Residual dataset (20 ch @ 10 Hz -> delta-P, route-wise split) | `src/features/build_dataset.py` | done |
| B4 LSTM and B6 distilgpt2+LoRA correctors, shrinkage, stitching | `src/models/` | done |
| Figures 1-5 | `src/plots/figures.py` | done |
| Map prior: continuity-constrained matching of the outage segment onto the road | `scripts/map_prior.py` | done |

### Baselines

| Name | Definition |
|---|---|
| **B1** | pure IMU dead reckoning, restarted at the last GNSS fix before each outage, evaluated inside the outage window |
| **B2** | forward 15-state ESKF |
| **B3** | ESKF + full-state RTS (offline, the "rough trajectory" that the learned corrector would refine) |

All three share the same mechanization, the same measurement model and the same
metric code path (`src/eval/metrics.py`).

---

## 2. Method

### 2.1 Filter

Nominal state $x=[p,v,q,b_a,b_g]$ is propagated by strapdown mechanization in the
ENU navigation frame (FLU body frame, $g^n=[0,0,-9.80665]$):

$$
v_{k+1}=v_k+(C_b^n f^b+g^n)\Delta t,\qquad
p_{k+1}=p_k+v_k\Delta t+\tfrac12 a^n\Delta t^2,\qquad
q_{k+1}=q_k\otimes\exp\big((\omega-b_g)\Delta t\big)
$$

The specific force is resolved with the attitude at the *start* of the interval,
which makes a noise-free IMU reproduce the simulated truth to $10^{-11}$ m.
The error state $\delta x=[\delta p,\delta v,\psi,\delta b_a,\delta b_g]$ (15 dims)
is estimated by an ESKF with Joseph-form covariance updates and
$\Phi=I+F\Delta t$; attitude errors are reset into the nominal quaternion after
every measurement update.

### 2.2 Why full-state RTS

The ESKF is *closed loop*: the error state is injected into the nominal state and
reset to zero at every update, so smoothing the error state alone yields an
identically zero correction. The smoother therefore runs on the **full state**
using both posterior and one-step-prior states:

$$
x^s_k=x^f_k+C_k\big(x^s_{k+1}-x^-_{k+1}\big),\qquad
C_k=P^f_k\Phi_k^\top\big(P^-_{k+1}\big)^{-1},
$$

with the covariance solved (never inverted) through `linalg.solve`.

### 2.3 GPU window parallelism

The time axis is recursive, so parallelism is taken over **windows x vehicles**:

```
window i :  [ 0 .. W )  forward burn-in
            [ W .. T-W) output region
            [ T-W .. T) backward burn-in (RTS needs future data)
stride    =  T - 2W - fade
```

Tensors are `[B, T, 3]` / `[B, T, 15, 15]` in `float32`; every time step is one
vectorised launch over the batch. Only the position block of $H$ is used, so the
innovation covariance is a 3x3 solve per window per fix.

**The burn-in, not the window, must outlast the outage.** A sample in the output
region is smoothed with at least `W` samples of data on each side, so a 60 s
outage around it is bracketed by GNSS on both sides if and only if `W > 60 s`.
Both failure modes were measured rather than guessed (outage mean / max 3-D
error in metres):

| layout | route 6 | route 43 |
|---|---|---|
| T = 20 s, W = 2 s | 139.3 / 254.1 | -- |
| T = 300 s, W = 30 s | 4.15 / 5.63 (600 s segment) | 30.0 / 133.1 |
| T = 600 s, W = 90 s | 11.4 / 26.0 (full route) | 9.1 / 13.6 (full route) |
| numpy full-route B3 | 9.5 / 21.6 | 9.0 / 11.5 |

The middle row is the instructive one: on route 6 the outage sits in the middle
of a window and 300 s / 30 s looks perfect, while on route 43 a 60 s outage
starts 0.3 s before the end of one window's output region - that window ends
inside the outage with no fix after it and dead-reckons to 133 m, and the next
window only starts to cover the same samples 27 s later. A 90 s burn-in removes
the condition by construction: any 60 s outage containing a smoothed sample lies
inside the window on both sides. `window_T = 60000` also keeps the number of
windows small (four per route), which matters for the next point.

**Throughput comes from batching across routes, not from small windows.** The
per-step cost of this kernel is dominated by Python and kernel-launch overhead
and is nearly independent of the batch size: measured on an RTX 3090, 4.2 ms per
step from B = 1 to B = 128, of which only 0.6 ms is GPU work. One route yields
4-8 windows, so running one route at a time would leave the machine idle in
exactly the same way as the serial baseline. `run_routes_batched` therefore puts
the windows of **all 48 routes into one stream** (192 windows) and runs the two
passes over it. The whole corpus (8.64 M IMU samples) takes **1009 s = 17 min,
21 s per route amortised** (full-corpus timing is from the 3090 instance; the
benchmark table and the test-route results below were re-run on an RTX 4090); running the identical torch kernel one window at a
time on the CPU would take 11.9 h at the measured 1.86 ms/step, i.e. a **42x**
throughput gain at this batch width. The kernel benchmark reaches ~200x at batch
512 (fig 1), which is where a fleet of vehicles rather than a single route's
four windows would sit.

The throughput benchmark deliberately uses independent 20 s windows
(`bench.window_T`): it measures the kernel, where window length only rescales
the x-axis, and both devices run the identical code path.

**End-to-end parity** (`python -m src.hpc.test_parity --device cuda`, route 6,
600 s segment containing a 60 s outage): windowed two-pass output vs the numpy
full-route smoother - max deviation 2.21 m, mean 0.31 m; in-outage mean error
10.17 m vs the numpy reference's 10.11 m; max step between adjacent samples
1.01x the ground-truth step, i.e. no seam at the window joins.

**Two passes for one long route**

1. every window is initialised independently from a coarse alignment
   (position = nearest GNSS fix; heading = GNSS course-over-ground over the
   largest available baseline, propagated to the window start with the gyro;
   heading uncertainty set from the baseline geometry). Fully parallel.
2. every window is re-initialised from the *previous* window's smoothed state at
   its own start index, which removes the alignment transient.

**Window stitching.** Adjacent windows overlap by `fade` = 1 s and the smoothed
position is combined with tent-shaped overlap-add weights, so the stitched
trajectory has no visible seam (max step <= 1.4x the ground-truth step).

### 2.4 Learned residual corrector

The filter output $P^{KF}$ is treated as a *rough* trajectory. A sequence model
learns

$$\delta P = P^{GT} - P^{KF},\qquad \hat P = P^{KF} + \alpha\,\widehat{\delta P},$$

where $\alpha$ is a scalar shrinkage fitted on the validation routes. $\alpha=0$
recovers the filter exactly, so the learned correction cannot make the result
worse than the filter it refines.

**What the model can actually learn.** This was measured before training
anything (`scripts/probe_learnability.py`, leave-one-route-out on the window
features):

* 73 % of the squared outage residual is a *per-outage constant offset* - that
  offset is the quantity the network has to predict;
* the offset is only weakly related to what the model can see: the leave-one-
  route-out probe on innovation, filter states and integrated accelerometer
  (routes 06/07/42) reaches $|r|\le0.35$ per axis and, once the shrinkage is
  fitted *inside* the fold, is **worse than predicting zero** (-112.6 % on the
  RTS residual). The $+1.2\%$ that an oracle scale buys is noise - its scale is
  negative;
* the same probe on the *causal* filter residual (B2, the online case) is the
  informative one: $|r|\approx0.70$ on both horizontal axes, so the *direction*
  of the drift is predictable. The magnitude is not: fitted inside the fold the
  correction costs 5.0 %. The $+23.9\%$ an oracle scale shows is the leak, not
  the signal;
* the RTS also absorbs an *injected* motion-dependent error: adding a quadratic
  scale factor $a(1+k|a|)$ leaves the outage error unchanged (7.67 -> 7.73 m),
  because the smoother interpolates the bias state across the outage.

The mechanism is the one above: a smoother with GNSS on both sides of an outage
estimates the bias trajectory and integrates it correctly, so what survives is
close to a white draw. Multipath is what *does* survive (GNSS-covered RMSE 1.2 m
-> ~5 m with the AR(1) process), but the surviving part is exactly the part the
innovation cannot attribute - the IMU sees the same data the filter did, so it
cannot separate multipath from its own drift either.

Consequence for the report: the *large* gain in the outage is the B1 -> B3 step
(-92 % mean, -95 % max), which is what the GPU filter delivers. The learned
corrector is reported as a **refinement with a measured, possibly small,
effect** - its shrinkage coefficient says by how much, and never worse than B3.

**Feature tensor** (20 channels at 10 Hz):

| # | channel | why |
|---|---|---|
| 0-2 | gyro | motion |
| 3-5 | accel | motion |
| 6-8 | GNSS innovation, `sign(x)*log1p(\|x\|)`, max-\|.\| per block | carries the multipath; log-compressed because it is ~4 m in tracking but 100-300 m at a reacquisition, and a z-score over the raw value would erase the informative range |
| 9 | outage flag | phase of the outage |
| 10 | time since last fix | the drift is an integral over exactly this interval |
| 11-13 | filter velocity | direction of travel |
| 14-19 | filter $b_a$, $b_g$ | the quantity whose *error* is being integrated twice |

The last nine channels are the filter's own state. Without them the mapping from
a 100 s sensor window to a 10 m position error is not identifiable: the residual
is the error of the bias estimate integrated twice, and the elapsed time is not
in a raw IMU window.

**Windows are symmetric** (`x[c-512 : c+512]` -> `y[c-64 : c+64]` at 10 Hz, i.e.
+-51 s of context around a 12.8 s output block). The pipeline is offline and the
RTS already uses future data; more importantly, the *only* observation that makes
the drift accumulated during an outage identifiable is the GNSS fix that ends it.
A causal window hides exactly that. The ablation `--causal` builds the classic
input-then-output layout for comparison.

**Models.** B4 is a two-layer LSTM trained from scratch. B6 is GPT4TS-style:
patch embedding (`Linear(20*16 -> 768)`) -> frozen `distilgpt2` with LoRA on
`c_attn`/`c_proj` (`r=8`, `alpha=16`) -> mean-pool head. Trainable parameters
0.87 M / 82.8 M = **1.06 %**, close to the 0.8 % quoted in the proposal.
`inputs_embeds` bypasses only the *token* table (`wte`); the Hugging Face
`GPT2Model` still adds its positional embedding (`wpe`) internally, so nothing
is added by hand - doing so would apply the position signal twice.

**Training.** AdamW (lr 1e-4, wd 0.01), OneCycle, batch 32-64, early stopping on
the validation routes; FP16 AMP with the loss computed in fp32 and
`scaler.unscale_` before gradient clipping; checkpoints keep fp32 weights.
Outage samples get `1 + 10` times the weight - they are 8 % of the samples but
95 % of the squared error that matters.

---

## 3. Data

A self-contained analytic simulator generates **48 routes** of 1800 s at 100 Hz
(8.64 M rows):

* 8 hand-designed routes (`route_00`-`route_07`: city grid, S-curves, ring road,
  highway, suburb, bridge, two test routes). They are split **5 train / 1 val /
  2 test** (`routes:` in `configs/common.yaml`), so the two test routes are the
  ones named in section 6 and the val route is the one `alpha` is fitted on;
* 40 augmented routes (`route_08`-`route_47`) drawn from the same five motion
  families with randomised durations, speeds, yaw rates and grades. The learned
  corrector needs many independent IMU error realisations: with only five
  training routes, leave-one-route-out regression is no better than predicting
  zero, i.e. it is learning route identity rather than physics.
* split **by route**: 35 train / 5 val / 8 test. Windows never cross a route.

Error model:

* motion profiles are piecewise constant with 4 s smoothstep transitions, so the
  motion is $C^1$;
* IMU: initial bias, bias random walk, 2 % **shared** scale-factor / axis
  misalignment error (+0.2 % per-route jitter), white noise, and a slow
  temperature-driven bias drift (10 min cycle, random phase);
* GNSS at 1 Hz, $\sigma_h=3$ m, $\sigma_v=5$ m, plus a correlated multipath bias
  (AR(1), $\sigma=4$ m, $\tau=120$ s);
* two 60 s outages per route, automatically placed on the **most dynamic**
  (turning / high-speed) segment of their search window, as required by the
  assignment.

The simulator is written from scratch (no dependency on `gnss-ins-sim`), and it
inverts its own mechanization exactly, so any residual error is caused by the
injected sensor errors and not by simulation artefacts.

---

## 4. Running

### Local (Windows, CPU, smoke test only)

```bash
pip install -r requirements_local.txt
python -m src.data.make_routes                 # 48 routes, ~2 min
python -m src.nav.pipeline                     # B1/B2/B3, ~35 min for 48 routes
python -m src.hpc.test_parity --device cpu --n 60000   # segment covers an outage
python -m src.features.build_dataset
python -m src.models.train --model lstm --smoke   # verify the training loop
python -m src.plots.figures --figs fig3 fig4 fig5
python scripts/make_summary.py                    # results/summary.md
```

`--smoke` writes `runs/models/lstm_smoke.*` and never touches `results/`, so a local
sanity run cannot overwrite a cloud result.

Any single step also runs on its own, e.g.

```bash
python -m src.models.train --model lstm               # B4
python -m src.models.train --model gpt4ts --amp       # B6 (LoRA)
python -m src.plots.figures --figs fig2 fig5          # after training
python scripts/probe_learnability.py --base b2        # is the residual learnable at all
python scripts/alpha_scan.py --model lstm             # guarded vs unguarded alpha
```

The online variant (correct the causal filter instead of the smoother) is the
same pipeline with `--pos b2`:

```bash
python -m src.features.build_dataset --pos b2                       # -> dataset_b2.npz
python -m src.models.train --model lstm --pos b2 --data dataset_b2.npz
python scripts/summary_online.py                                    # results/online/summary.md
```

### AutoDL (any CUDA GPU; 8 GB works, 24 GB has headroom)

```bash
pip install -r requirements_cloud.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com          # distilgpt2 ~350 MB
bash scripts/run_all.sh cuda                      # ~1.5 h end to end
bash scripts/run_all.sh cuda --skip-llm           # filter-only, ~50 min
bash scripts/run_all.sh cuda --skip-ref           # reuse an uploaded runs/kf, ~30 min shorter
```

Disk: ~6 GB (`data/` 3 GB of CSV+npz, `runs/` 2.5 GB of per-route filter output,
`data/datasets/dataset.npz` 0.5 GB). Everything is regenerated by the script.

GPU memory: the window parallel stage holds a covariance history of
`chunk x T x 15 x 15` floats - 5.2 GB at the defaults (`batch_chunk = 96`,
`window_T = 60000`). `run_routes_batched` reads the free memory at start-up and
shrinks the chunk if the card cannot hold it (measured: ~54 MB per window), so
an 8 GB card runs the same result at a lower batch width; the per-step cost is
flat in the batch size, so only throughput drops.

Every stage is checkpointed on disk, so a crashed run can be resumed by
re-running the single remaining `python -m ...` command; nothing is recomputed
except the stage you relaunch. The numpy reference ([2/10], ~30 min on 1 CPU
core) is the slowest stage and the only one that is not GPU-accelerated.

Outputs
* `results/bench.json` — throughput / speedup / batch scan
* `results/fig1_speedup.png`, `fig2_loss.png`, `fig3_trajectory.png`,
  `fig4_error.png`, `fig5_summary.png`, `fig6_learnability_b3.png`
  (`scripts/fig6_learnability.py` - how the remaining outage error decomposes),
  `fig7_map_prior_b3.png` (`scripts/fig7_map_prior.py` - the map prior, the one
  step that adds information), `fig8_laneline.png`
  (`scripts/fig8_laneline.py` - what the same error does to a lane line, i.e.
  the map-production view)
* `results/ppt/a_imbalance.png`, `b_training.png`, `c_gate.png` — the three
  slide panels for the training / gating slide (`scripts/ppt_figs.py`; every
  number they print is the one to quote)
* `图表说明.md` — Chinese reading notes for every figure: what each panel plots,
  the numbers to quote, and the questions each one invites
* `runs/kf/*.json` — B1/B2/B3 metrics per route (numpy reference)
* `runs/kf_gpu/*.json` — window-parallel GPU filter metrics
* `runs/models/{lstm,gpt4ts}.pt` + `_history.json` — checkpoints and curves
* `results/metrics_{lstm,gpt4ts}.json` — per-route B3 -> Ours, with `alpha`
* `results/online/` — the causal-filter variant (`--pos b2`) and its summary
* `results/summary.md` — the aggregate table and headline numbers, generated by
  `scripts/make_summary.py` from whatever of the above exists

---

## 5. Metrics (fixed definitions, same code path for every method)

| Metric | Definition |
|---|---|
| `rmse_all` | 3-D position RMSE over the whole route |
| `outage_max` / `outage_mean` | max / mean 3-D error inside the GNSS-outage windows |
| `recovery_s` | after an outage ends, time until the 3-D error stays <= 3 m for >= 2 s (unrecovered = 60 s) |
| `speedup` | CPU-serial wall clock / GPU wall clock, same inputs and code path |

---

## 6. Results (simulated test routes, RTX 4090)

Outage-window **mean** / **max** 3-D error in metres. B1 = IMU dead reckoning,
B2 = forward 15-state ESKF, B3 = the numpy ESKF+RTS reference; GPU = the
window-parallel pipeline (T = 600 s, burn-in 90 s > the 60 s outage). "Ours" is
the learned corrector: it ships behind a one-standard-error gate on the
validation split, which sets `alpha = 0`, so the module runs in pass-through and
repeats the column it refines. It is listed separately below.

| route | B1 (DR) | B2 (forward ESKF) | B3 (numpy RTS) | GPU (windowed RTS) |
|---|---|---|---|---|
| 06 dense urban | 119.8 / 375 | 47.7 / 195 | 7.67 / 15.9 | 8.31 / 15.3 |
| 07 highway/urban | 113.8 / 397 | 66.2 / 201 | 8.43 / 15.0 | 12.30 / 25.4 |
| 42-47 augmented | 59-137 / 208-413 | 27.9-64.4 / 90-301 | 6.98-14.74 / 11.5-28.3 | 6.87-15.02 / 12.7-29.6 |
| **test mean (8)** | **104.65 / 413** | **51.44 / 301** | **9.93 / 28.3** | **10.75 / 29.6** |

GNSS-covered RMSE of B3 is 4.6-6.3 m; the spread is the multipath bias of that
route. Full numbers in `runs/kf/metrics_all.json` (numpy) and
`runs/kf_gpu/metrics_gpu.json` (GPU); the table is regenerated by
`scripts/make_summary.py` into `results/summary.md`.

**Headline.** Against dead reckoning, ESKF+RTS removes **90.5 % of the mean**
and **93.1 % of the worst-case** outage error over the eight test routes
(B1 104.65 m mean / 413 m worst -> B3 9.93 m / 28.3 m).

**What windowing costs.** On the same routes the GPU pipeline gives 10.75 m
mean (+8.2 % over the full-route numpy smoother) and a 29.6 m worst case, 1.3 m
worse than numpy's 28.3 m. The difference is structural, not numerical: a
window sees +-90 s of fixes instead of the whole route, so the smoother has less
data on each side of an outage. The windowed CPU rerun of the same code path
matches the 4090 to within 1 m per route (mean 10.55 m), so the gap is the
windowing, not float32. Parity of the kernel itself is 2.2 m max deviation from
numpy when both are run windowed (`src/hpc/test_parity.py`).

**Throughput (fig 1).** The benchmark pits the *same* PyTorch kernel against
itself - CPU serial (batch = 1, one window after another) vs CUDA batched - so
the only difference between the two columns is the parallelism. The CPU
baseline is deliberately the naive loop of that kernel: the numpy reference in
`src/nav` is several times faster than the torch CPU loop. Both timings start
after the host-to-device copy, so the number is filter throughput, not I/O.

| samples | CPU serial | GPU (B = 128) | speedup |
|---|---|---|---|
| 10 000 | 22.4 s | 9.85 s | 2.3x |
| 100 000 | 223 s | 10.7 s | 20.8x |
| 1 000 000 | 2245 s | 46.2 s | **48.6x** |

The per-step cost is flat in the batch size (~4 ms/step is launch overhead, not
arithmetic), so the curve in fig 1a is *sub-linear* in size and the speedup
keeps climbing with the workload: at 500 windows per launch the same 1 M
samples take 12.6 s -> **177.6x** (fig 1c). That is also why the real pipeline
batches windows **across routes** rather than per route: the whole 48-route
corpus (8.64 M samples) runs in 1009 s = 17 min, 21 s per route, where the same
kernel serial on CPU would take 11.9 h.

**Learned corrector: shipped behind a statistical gate.** Both models are fully
trained (LSTM 30 epochs, GPT4TS/LoRA 15 epochs, 0.87 M / 82.8 M = 1.06 %
trainable) and both fit the training routes. On held-out routes their best score
lands within noise of the "predict zero" baseline, so the acceptance rule in
`fit_alpha` - a one-standard-error test on the validation split - does not
certify a correction, and the module ships with `alpha = 0.00`. "Ours" is then
the filter output. That is the gate doing its job: the module is allowed to add a
correction only when the evidence clears one standard error, so it can improve
on the delivered trajectory but never degrade it.

This is worth stating as a measurement rather than a caveat, because it says
something about the filter. With a correct 15-state ESKF and a full-state RTS
pass, the error left in an outage is the filter's own estimation error *at the
moment GNSS is lost* - position, velocity and attitude uncertainty frozen into
a random offset that the filter cannot observe. Recovering it is not a matter of
a better network on the same inputs; it takes information from outside the
filter, which is what the map prior below supplies.

Getting the models to this point took a fix worth recording. The first runs
regressed the residual in raw metres (12 m rms for the RTS residual, 72 m for
the causal filter); a freshly initialised head needs on the order of $10^6$ Adam
steps to reach an output of that size, so both networks sat at "predict zero"
and their validation MSE was simply the target variance - they had not trained,
and those numbers said nothing about the data. `src/models/train.py` now fits in
units of the *train* residual rms and scales the predictions back (a train-only
statistic, no leakage), and the networks genuinely learn the training routes
(weighted loss 5.30 -> 4.40, -17 %; GPT4TS 4.98 -> 1.94). Their held-out scores
move to 11.79 / 11.68 m^2 against 11.48 m^2 for predicting zero - 2.7 % and
1.7 % *above* the trivial baseline, i.e. neither network ever beats it on
routes it has not seen, and that is the result that survives the fix.

| model | validation: unguarded grid optimum | alpha shipped (one-SE gate) | test mean |
|---|---|---|---|
| LSTM | 2.00 -> 8.36 m (+2.7 %) | **0.00** | 10.75 m (= filter) |
| GPT4TS + LoRA | 1.90 -> 8.35 m (+2.9 %) | **0.00** | 10.75 m (= filter) |

The validation mean at `alpha = 0` is 8.59 m. The unguarded optima *look* like a
0.23-0.25 m gain on those five routes - inside one standard error of the
per-route scores - and shipping them costs 2.2 % (LSTM) and 4.9 % (GPT4TS) on
the test mean: the case the gate exists for, reproduced by
`scripts/alpha_scan.py`.

**Online variant.** The smoother needs the whole route, so the same experiment
was repeated against the causal forward filter (`--pos b2`,
`data/datasets/dataset_b2.npz`): its residual is 72.5 m rms and its horizontal
components correlate with the observable motion at $|r|\approx0.70$, but the
magnitude does not transfer between routes: the trained LSTM's best validation
MSE is 265.0 m^2 against 252.9 m^2 for predicting zero, its validation optimum
is `alpha = 0.30` for +0.04 %, the guard returns `alpha = 0.00` again, and the
test mean stays at B2's 51.44 m. Direction is predictable; size is not, because
the size is set by the bias error at outage onset, which the filter cannot
observe.

**Sensor-error target, and a plumbing control.** A third attempt changed the
*label* rather than the base: instead of the position residual, learn the
accelerometer error
$\delta f=f_{\text{meas}}-f^{\text{true}}_b$, which is an algebraic function of
the reading itself and therefore learnable in principle
(`scripts/probe_imu_error.py`). A leave-one-route-out linear model lowers the
sensor-level rms from 0.0985 to 0.0913 m/s^2 (+7.3 %); the *oracle* that
subtracts the true scale error reaches +27.5 % - and re-running the filter on
that oracle-corrected IMU moves the B3 outage error from 9.65 to 9.73 m
(**-0.9 %**). The reason is structural: the specific force is dominated by the
gravity-reaction term, so a scale error on it is a near-constant in the body
frame and the accelerometer-bias state already absorbs it. Raising the *gyro*
scale error to 6 % - which the bias state cannot absorb - leaves B3 at
7.8 / 8.7 / 13.8 m against 7.7 / 8.4 / 12.8 m without it.

Finally, to rule out a broken pipeline as the explanation, the true block-mean
residual was pushed through the identical stitch + outage-gate path
(`scripts/probe_b2_e2e.py`): validation goes 69.15 -> **11.46 m (-83 %)**. The
plumbing recovers the residual when the residual is correct, so the models, not
the harness, are the bottleneck.

**Map prior: the correction that does work.** The residual carries no
information the filter has not already used, so the only way to improve the
outage fix is to add *new* information - and a road map is exactly that. The
outage error splits almost evenly into an along-road and a cross-road part
(per-route RMS: 7.86 m vs 7.82 m, vertical 1.68 m), and the map removes the
cross-road half. `scripts/map_prior.py` marches the trajectory along the
centreline under a continuity constraint (monotone arclength, speed bound,
heading term) rather than taking the nearest point, then blends it in with the
same 1 s outage gate. Test mean **9.93 -> 6.30 m (-36.5 %)**, all eight routes
improve (+32 to +50 %); applied to the GPU output instead it is **10.75 ->
6.98 m (-35.1 %)**, i.e. the deployable pipeline plus a map beats the offline
numpy smoother (6.98 vs 9.93 m). The naive nearest-point projection gives 7.52 m
(+24.3 %) and *regresses* route 46 by 36 % - the ring route passes within 6 m
of itself - which is what the continuity constraint fixes. The method tolerates
a map error up to about 1 m rms (test mean 6.31 m); at 2 m route 46 jumps to
the parallel branch (323 m). It improves the *mean*, not the worst case (28.3
-> 28.3 m): the peak sits where the along-track error peaks, which no map can
remove. See `results/fig7_map_prior_b3.png`.

**What the error means for a map.** Trajectory correction is the upstream stage
of map production: a lane marking is measured relative to the vehicle (to a few
centimetres) and placed in the world by adding the vehicle's pose, so a pose
error becomes a map error, term for term. Only the component normal to the road
shows up as the lateral offset of the mapped lane line - the doubled, smeared
line of a map built from uncorrected trajectories; the along-road part slides
features along the road, where a continuous marking hides it. Measured on the
same outage samples with the lane boundaries placed at +/-1.75 m from the true
centreline (`scripts/fig8_laneline.py`), the mean lateral offset is **43.9 m**
for B1, **6.0 m** for B3, **6.5 m** for the GPU output and **0.3 m** with the
map prior, against an along-road part of 40.7 / 6.3 / 6.9 / 6.9 m - the map
prior removes the cross-road half and, correctly, leaves the other one alone.
One outage on route 45 pushes the B1 line 155 m sideways; it leaves the frame
in `fig8`. Against the 0.5 m lateral tolerance an HD map is usually built to,
**4.1 %** of B1's mapped points land inside it, 9.3 % of B3's, 6.5 % of the GPU
output's and **86.2 %** with the map prior. Two caveats, stated in the script:
the 0.3 m residual is the chord error of the 1 Hz road polyline, not a pose
accuracy, and the map prior consumes an existing map, so it measures map
*update* rather than map creation from scratch.

Read together, the two experiments say one thing: the filter already extracts
everything its inputs contain, so the next gain comes from adding an input. That
is why the delivered system is the GPU filter, plus the map prior wherever a map
exists, and why the learned module is kept on a short leash - it costs nothing
when it has nothing to add, and the system's accuracy is never at its mercy.

---

## 7. Notes and limitations

* Data are **simulated**. The error model is fixed in `configs/common.yaml` and
  reported as-is; real-data slots (KITTI / OxIOD) are left as extension points.
* Earth rotation and Coriolis terms are neglected (the simulation uses a
  non-rotating frame); this is stated explicitly in the report.
* The whole pipeline is offline; the GPU batching exploits that, in particular
  for the coarse alignment, which may use GNSS data up to 60 s after a window
  start.
* The 48-route corpus is generated once and split **by route**. Route identity
  leaks in one benign way: every route shares the same IMU unit, so a model can
  legitimately learn that unit's scale-factor error; it cannot learn a route it
  has never seen, and the test split measures exactly that.
