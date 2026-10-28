"""Train a sequence corrector on the KF/RTS residual and evaluate it end-to-end.

    python -m src.models.train --model lstm      --epochs 20
    python -m src.models.train --model gpt4ts    --epochs 12 --batch 64 --amp
    python -m src.models.train --model lstm --smoke            # 30 s sanity run

Artifacts
    runs/models/<model>.pt            best checkpoint (fp32 weights)
    runs/models/<model>_history.json  train/val curves
    results/metrics_learned.json      per-route metrics with the correction applied
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..common import DATASET_DIR, RESULTS_DIR, RUNS_DIR, load_config
from ..eval.metrics import outage_stats, summarize, traj_error
from ..features.build_dataset import route_position
from .correctors import build_model


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_dataset(name="dataset.npz"):
    path = DATASET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run `python -m src.features.build_dataset`")
    with np.load(path) as d:
        out = {k: d[k] for k in d.files if k not in ("mu", "sd", "in_len", "out_len",
                                                     "stride", "mode", "rate")}
        out["in_len"] = int(d["in_len"])
        out["out_len"] = int(d["out_len"])
        out["stride"] = int(d["stride"]) if "stride" in d.files else out["out_len"]
        out["mode"] = str(d["mode"]) if "mode" in d.files else "centered"
        # windows live on the decimated grid, the route arrays are at fs_imu:
        # route_predictions needs the ratio to put the blocks back
        out["rate"] = float(d["rate"]) if "rate" in d.files else float(load_config()["fs_imu"])
    return out


def _batches(n, batch, shuffle, rng=None):
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, batch):
        yield idx[i: i + batch]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train(model, ds, args, device):
    ys = ds.get("y_scale", 1.0)
    x_tr = torch.as_tensor(ds["train_x"])
    y_tr = torch.as_tensor(ds["train_y"] / ys)
    w_tr = torch.as_tensor(ds["train_w"])
    g_tr = torch.as_tensor(ds["train_og"])          # per-sample outage mask
    x_va = torch.as_tensor(ds["val_x"]).to(device)
    y_va = torch.as_tensor(ds["val_y"] / ys).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train:,} / {n_all:,} = {100.0 * n_train / n_all:.2f}%")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    steps = math.ceil(len(x_tr) / args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(steps, 1), pct_start=0.15, anneal_strategy="cos")

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    rng = np.random.default_rng(args.seed)

    best, best_ep, history = float("inf"), -1, []
    for ep in range(args.epochs):
        model.train()
        tot, n_seen = 0.0, 0
        t0 = time.time()
        for idx in _batches(len(x_tr), args.batch, True, rng):
            xb = x_tr[idx].to(device, non_blocking=True)
            yb = y_tr[idx].to(device, non_blocking=True)
            wb = w_tr[idx].to(device, non_blocking=True).view(-1, 1, 1)
            gb = g_tr[idx].to(device, non_blocking=True).view(-1, yb.shape[1], 1)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(xb)
            pred = pred.float()                     # loss always in fp32
            # outage samples carry 1 + outage_weight times the weight: they are
            # 8 % of the data but 95 % of the squared error that matters
            loss = (wb * (1.0 + args.outage_weight * gb) * (pred - yb) ** 2).mean()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
            n_seen += len(idx)

        model.eval()
        with torch.no_grad():
            vals = []
            for i in range(0, len(x_va), args.batch * 4):
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    pv = model(x_va[i:i + args.batch * 4])
                vals.append(F.mse_loss(pv.float(), y_va[i:i + args.batch * 4], reduction="none")
                            .mean(dim=(1, 2)))
            val = float(torch.cat(vals).mean())
        tr = tot / max(n_seen, 1)
        # history and console are in m^2; ``best`` stays in the normalised units
        # the comparison above used
        tr_m, val_m = tr * ys * ys, val * ys * ys
        history.append({"epoch": ep + 1, "train_mse": tr_m, "val_mse": val_m,
                        "lr": sched.get_last_lr()[0], "sec": time.time() - t0})
        flag = ""
        if val < best:
            best, best_ep, flag = val, ep, " *"
            torch.save({"model": args.model, "state": model.state_dict(),
                        "args": vars(args), "epoch": ep + 1}, RUNS_DIR / "models" / f"{args.stem}.pt")
        print(f"ep {ep + 1:02d}/{args.epochs}  train {tr_m:.5f}  val {val_m:.5f}  "
              f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:5.1f}s{flag}")
        if ep - best_ep >= args.patience:
            print(f"early stop (patience {args.patience})")
            break
    return history, best


# --------------------------------------------------------------------------- #
# route-level evaluation: stitch window predictions back onto the route
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, x, device, batch=256, amp=False, amp_dtype=torch.float16):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(x[i:i + batch]).to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 64, 3), np.float32)


def stitch(n: int, pred: np.ndarray, in_len: int, out_len: int,
           stride: int, mode: str = "centered") -> np.ndarray:
    """Overlap-add the per-window delta-P back onto the route.

    The first output block of a *centered* window starts at
    ``in_len/2 - out_len/2`` (the window is x[c-half:c+half] -> y[c-oh:c+oh]);
    for the causal layout it starts at ``in_len``.  Consecutive windows are
    ``stride`` apart, which is *not* the same as ``out_len`` when the windows
    overlap, so the stride has to come from the dataset.
    """
    dp = np.zeros((n, 3), np.float64)
    acc = np.zeros((n, 1), np.float64)
    start0 = in_len // 2 - out_len // 2 if mode == "centered" else in_len
    for i, p in enumerate(pred):
        s = start0 + i * stride
        if s >= n:
            break
        e = min(s + out_len, n)
        if e <= s:
            break
        dp[s:e] += p[: e - s]
        acc[s:e] += 1.0
    acc[acc == 0] = 1.0
    return dp / acc


def outage_gate(outage: np.ndarray, fs: float, taper_s: float = 1.0) -> np.ndarray:
    """1 inside an outage, smoothly tapered to 0 over ``taper_s`` at both edges.

    The corrector is only trusted where the filter is degraded; outside that the
    residual is sub-metre noise that no model can predict, and applying a
    correction there can only add variance.
    """
    m = outage.astype(np.float64)
    k = max(int(round(taper_s * fs)), 1)
    kern = np.hanning(2 * k + 1)
    kern /= kern.sum()
    return np.convolve(m, kern, mode="same")


def route_predictions(model, ds, split, cfg, args, device, pos_source="auto"):
    """Per-route (base position, predicted delta-P, ground truth, outage mask)."""
    out = {}
    for rid in sorted(set(int(r) for r in ds[f"{split}_route"])):
        sel = ds[f"{split}_route"] == rid
        pred = predict(model, ds[f"{split}_x"][sel], device,
                       amp=args.amp and device.type == "cuda")
        stems = sorted((RUNS_DIR / "kf").glob(f"route_{rid:02d}_*.npz"))
        with np.load(stems[0], allow_pickle=True) as d:
            route = {k: d[k] for k in d.files}
        base = route_position(route, stems[0].stem, pos_source)
        # The dataset lives on the decimated grid (10 Hz), the route arrays are
        # at fs_imu (100 Hz).  Stitch on the decimated grid, then hold each
        # sample for one full-rate tick -- otherwise the correction is squeezed
        # into the first 1/10 of the route and never reaches an outage.
        fs = float(cfg["fs_imu"])
        up = max(1, int(round(fs / float(ds["rate"]))))
        n_low = -(-len(route["p_gt"]) // up)
        dp = stitch(n_low, pred, ds["in_len"], ds["out_len"], ds["stride"], ds["mode"])
        dp = dp * ds.get("y_scale", 1.0)                     # back to metres
        dp = np.repeat(dp, up, axis=0)[: len(route["p_gt"])]
        out[rid] = {"route": stems[0].stem, "base": base, "dp": dp,
                    "gt": route["p_gt"], "outage": route["outage"].astype(bool)}
    return out


def fit_alpha(preds, cfg, gate=False, metric="outage_mean", grid=None,
              k_se=1.0):
    """Scalar shrinkage on the validation routes, with a one-SE guard.

    ``alpha = 0`` recovers the filter exactly, so the corrector can never be
    reported as helping when it does not.  The grid is scanned on the headline
    metric, but an alpha is only accepted if its mean validation score beats
    alpha = 0 by more than ``k_se`` standard errors of the per-route scores:
    with five validation routes a few-tenths-of-a-metre "gain" is inside the
    noise, and picking it anyway is exactly how a corrector ends up worse than
    the filter it refines on held-out routes (measured: the unguarded grid
    optimum was alpha = 2.0 for the LSTM and 1.9 for GPT4TS, worth +2.7 % and
    +2.9 % on validation and -2.2 % / -4.9 % on test; `scripts/alpha_scan.py`
    reproduces it).
    """
    grid = grid if grid is not None else np.linspace(0.0, 2.0, 41)
    fs = float(cfg["fs_imu"])
    scores = {}
    for a in grid:
        vals = []
        for r in preds.values():
            g = outage_gate(r["outage"], fs) if gate else np.ones(len(r["gt"]))
            e = traj_error(r["base"] + a * g[:, None] * r["dp"], r["gt"])
            vals.append(summarize(e, r["outage"], fs)[metric])
        scores[float(a)] = np.asarray(vals, float)
    s0 = scores[0.0]
    se = float(np.std(s0, ddof=1) / np.sqrt(max(len(s0), 1)))
    for a in grid:
        if np.nanmean(scores[float(a)]) < np.nanmean(s0) - k_se * se:
            return float(a), float(np.nanmean(scores[float(a)]))
    return 0.0, float(np.nanmean(s0))


def evaluate(preds, cfg, alpha=1.0, gate=False, tag="", base_label="B3"):
    fs = float(cfg["fs_imu"])
    per_route = {}
    for rid, r in preds.items():
        g = outage_gate(r["outage"], fs) if gate else np.ones(len(r["gt"]))
        p_corr = r["base"] + alpha * g[:, None] * r["dp"]
        m = {"route": r["route"], "n": len(r["gt"])}
        for name, p in [("b3", r["base"]), ("ours", p_corr)]:
            m[name] = summarize(traj_error(p, r["gt"]), r["outage"], fs)
        m["outage_mean_reduction_pct"] = 100.0 * (1 - m["ours"]["outage_mean"] / m["b3"]["outage_mean"])
        m["outage_max_reduction_pct"] = 100.0 * (1 - m["ours"]["outage_max"] / m["b3"]["outage_max"])
        per_route[rid] = m
        print(f"  {tag}route {rid:2d}: {base_label} {m['b3']['outage_mean']:6.2f} -> ours "
              f"{m['ours']['outage_mean']:6.2f} m ({-m['outage_mean_reduction_pct']:+5.1f}%)  "
              f"max {m['b3']['outage_max']:6.2f} -> {m['ours']['outage_max']:6.2f}")
    return per_route


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "gpt4ts"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outage-weight", type=float, default=10.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pos", choices=["auto", "kf", "kf_gpu", "b2"], default="auto",
                    help="base trajectory the corrector refines: the offline (B3 / GPU) "
                         "one, or b2 = the causal forward filter (online case)")
    ap.add_argument("--data", default="dataset.npz",
                    help="dataset file under data/datasets/ (use dataset_b2.npz with --pos b2)")
    ap.add_argument("--gate", action="store_true",
                    help="apply the correction only around GNSS outages")
    ap.add_argument("--alpha-metric", default="outage_mean",
                    choices=["outage_mean", "outage_max", "rmse_all", "rmse_gnss"],
                    help="validation metric the shrinkage coefficient minimises")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 1 epoch, small subsets")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    (RUNS_DIR / "models").mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.data)
    # Fit in units of the train residual rms.  On raw metres a freshly
    # initialised head needs ~1e6 Adam steps to reach a 70 m output, so the
    # network stalls at "predict zero" (val MSE = target variance) -- which is
    # what the first round of training did.  Train-only statistic: no leakage.
    ds["y_scale"] = float(np.sqrt(np.mean(np.square(
        ds["train_y"].astype(np.float64))))) or 1.0
    print(f"residual rms on the train windows: {ds['y_scale']:.2f} m "
          f"(targets are fitted in these units)")
    if args.smoke:
        args.epochs, args.batch = 1, 32
        for k in ("train", "val", "test"):
            ds[f"{k}_x"] = ds[f"{k}_x"][:640]
            ds[f"{k}_y"] = ds[f"{k}_y"][:640]
            ds[f"{k}_w"] = ds[f"{k}_w"][:640]
            ds[f"{k}_og"] = ds[f"{k}_og"][:640]
            ds[f"{k}_route"] = ds[f"{k}_route"][:640]
    # a smoke run must not overwrite the real checkpoint / metrics: the README
    # tells people to smoke-test locally and then copy the tree to the GPU host
    args.stem = f"{args.model}_smoke" if args.smoke else args.model
    if args.pos == "b2" and not args.smoke:
        # the online variant keeps its own checkpoints and metrics; the offline
        # numbers that the figures and the README quote must stay untouched
        args.stem += "_b2"

    model = build_model(args.model, cfg, in_len=ds["in_len"], out_len=ds["out_len"]).to(device)
    history, best = train(model, ds, args, device)

    with open(RUNS_DIR / "models" / f"{args.stem}_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    if not args.no_eval:
        ckpt = torch.load(RUNS_DIR / "models" / f"{args.stem}.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state"])
        print("\nvalidation routes: shrinkage fit")
        val_pred = route_predictions(model, ds, "val", cfg, args, device, args.pos)
        alpha, val_score = fit_alpha(val_pred, cfg, gate=args.gate, metric=args.alpha_metric)
        print(f"  alpha* = {alpha:.2f}  (val {args.alpha_metric} {val_score:.2f} m)")
        print("\ntest routes:")
        test_pred = route_predictions(model, ds, "test", cfg, args, device, args.pos)
        per_route = evaluate(test_pred, cfg, alpha=alpha, gate=args.gate,
                             base_label="B2" if args.pos == "b2" else "B3")
        # store the corrected trajectories so the figures can draw "Ours"
        fs = float(cfg["fs_imu"])
        save = {}
        for rid, r in test_pred.items():
            g = outage_gate(r["outage"], fs) if args.gate else np.ones(len(r["gt"]))
            save[f"p_ours_{rid}"] = (r["base"] + alpha * g[:, None] * r["dp"]).astype(np.float32)
        np.savez_compressed(RUNS_DIR / "models" / f"{args.stem}_test.npz", **save)
        # metrics_{model}.json is what the figures and scripts/make_summary.py pick
        # up, so a smoke run writes next to its checkpoint instead
        if args.smoke:
            out = RUNS_DIR / "models" / f"{args.stem}_metrics.json"
        else:
            # the online variant writes to a subdirectory so the glob in
            # figures.py / make_summary.py (results/metrics_*.json) stays clean
            res_dir = RESULTS_DIR / "online" if args.pos == "b2" else RESULTS_DIR
            res_dir.mkdir(parents=True, exist_ok=True)
            out = res_dir / f"metrics_{args.model}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model, "base": args.pos,
                       "val_mse": best * ds["y_scale"] ** 2,   # m^2, comparable
                       "alpha": alpha, "routes": per_route}, fh, indent=2)
        print("-> " + str(out))


if __name__ == "__main__":
    main()
