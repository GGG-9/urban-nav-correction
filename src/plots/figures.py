"""Report figures (English labels, 300 dpi, saved to results/).

    fig1_speedup      CPU-serial vs GPU-batched throughput / speedup
    fig3_trajectory   3-D trajectories, ground truth vs B1 / B2 / B3, plus the
                      window-parallel GPU output when ``runs/kf_gpu`` exists
    fig4_error        3-D position error vs time with GNSS outages shaded

Usage:
    python -m src.plots.figures [--route 6] [--figs fig1 fig3 fig4]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..common import NPZ_DIR, RESULTS_DIR, RUNS_DIR, load_config

# Two palettes: the working deck (default) and the green course-report template
# (E:\大作业\课程汇报绿色模板.pptx, accents 31481D / 587558 / 9DC48C / 34765E).
# Select with FIG_PALETTE=green; redirect output with FIG_OUT_DIR=<dir> so the
# green variants never overwrite the figures the main deck uses.
_PALETTES = {
    "default": {"gt": "#111111", "b1": "#d62728", "b2": "#ff7f0e",
                "b3": "#1f77b4", "ours": "#2ca02c", "gpu": "#9467bd"},
    # a light -> dark green ramp: the darker the line, the better the tier
    "green": {"gt": "#1F2328", "b1": "#A9AFA6", "b2": "#C9DCC0",
              "b3": "#9DC48C", "ours": "#31481D", "gpu": "#587558"},
}
COL = _PALETTES[os.environ.get("FIG_PALETTE", "default")]


def _style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 120,
        "savefig.dpi": load_config()["figures"]["dpi"],
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
    })


def _out(name: str):
    d = pathlib.Path(os.environ.get("FIG_OUT_DIR") or RESULTS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def fig1_speedup(path=None, out=None):
    path = path or (RESULTS_DIR / "bench.json")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    sizes = d["sizes"]
    rows = np.array([s["rows"] for s in sizes], float)
    sp = np.array([s["speedup"] for s in sizes], float)
    cpu = np.array([s["cpu_s"] for s in sizes], float)
    gpu = np.array([s["gpu_s"] for s in sizes], float)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    a = ax[0]
    a.loglog(rows, cpu, "o-", color=COL["b2"], label="CPU serial (batch = 1)")
    a.loglog(rows, gpu, "s-", color=COL["b3"], label=f"GPU batched (B = {d['chunk']})")
    a.set_xlabel("Samples processed")
    a.set_ylabel("Wall-clock time [s]")
    a.set_title("(a) Runtime")
    a.legend(frameon=False, fontsize=8)

    a = ax[1]
    a.loglog(rows, sp, "^-", color=COL["ours"])
    for r, s in zip(rows, sp):
        a.annotate(f"{s:.0f}x", (r, s), textcoords="offset points", xytext=(0, 6),
                   ha="center", fontsize=7)
    a.set_xlabel("Samples processed")
    a.set_ylabel("Speedup (CPU / GPU)")
    a.set_title("(b) Speedup")
    a.axhline(100, color="grey", ls=":", lw=1)

    a = ax[2]
    scan = d["batch_scan"]
    b = [s["batch"] for s in scan]
    thr = [s["rows_per_s"] / 1e3 for s in scan]
    a.semilogy(b, thr, "o-", color=COL["b3"])
    a.set_xlabel("Batch size (windows per launch)")
    a.set_ylabel("Throughput [k samples/s]")
    a.set_title("(c) Batch-size scan")
    fig.suptitle(f"Window-parallel KF+RTS - {d['gpu_name']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out or _out("fig1_speedup.png"))
    plt.close(fig)
    return out or _out("fig1_speedup.png")


def fig2_loss(model_names=("lstm", "gpt4ts"), out=None):
    """Training / validation loss curves (log scale) for the learned correctors."""
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.4))
    plotted = False
    for i, name in enumerate(model_names):
        path = RUNS_DIR / "models" / f"{name}_history.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            h = json.load(fh)
        ep = [r["epoch"] for r in h]
        ax.semilogy(ep, [r["train_mse"] for r in h], "-o", ms=3,
                    color=["#1f77b4", "#2ca02c"][i % 2], label=f"{name} train")
        ax.semilogy(ep, [r["val_mse"] for r in h], "--s", ms=3, alpha=0.75,
                    color=["#1f77b4", "#2ca02c"][i % 2], label=f"{name} val")
        plotted = True
    if not plotted:
        return None
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted MSE of $\\delta P$ [m$^2$]")
    ax.set_title("Residual corrector: training convergence")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out or _out("fig2_loss.png"))
    plt.close(fig)
    return out or _out("fig2_loss.png")


def _kf_metrics(rid: int):
    p = sorted((RUNS_DIR / "kf").glob(f"route_{rid:02d}_*.json"))
    if not p:
        return None
    with open(p[0], encoding="utf-8") as fh:
        return json.load(fh)


def _learned_metrics() -> dict:
    """route id -> its 'ours' entry, from every results/metrics_*.json."""
    ours = {}
    for p in RESULTS_DIR.glob("metrics_*.json"):
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        for rid, r in m.get("routes", {}).items():
            if "ours" in r:
                ours[int(rid)] = r
    return ours


def _gpu_metrics() -> dict:
    """route id -> window-parallel GPU metrics, from runs/kf_gpu/metrics_gpu.json."""
    path = RUNS_DIR / "kf_gpu" / "metrics_gpu.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    out = {}
    for stem, r in m.items():
        parts = stem.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            out[int(parts[1])] = r
    return out


_F5_LABEL = {"B1": "B1\nDR", "B2": "B2\nESKF", "B3": "B3\nESKF+RTS",
             "gpu": "GPU\nESKF+RTS", "ours": "Ours"}
_F5_COLOR = {"B1": COL["b1"], "B2": COL["b2"], "B3": COL["b3"],
             "gpu": COL["gpu"], "ours": COL["ours"]}


def fig5_summary(route_ids=(6, 7), out=None):
    """Bar chart: outage mean / max error for B1, B2, B3, GPU, Ours, + the test mean."""
    ours = _learned_metrics()
    gpu = _gpu_metrics()
    alpha = _best_model()[1]
    use_ours = bool(ours) and alpha > 0   # at alpha = 0 "Ours" *is* the base
    panels = []
    for rid in route_ids:
        m = _kf_metrics(rid)
        if m is None:
            continue
        keys = ["B1", "B2", "B3"] + (["gpu"] if rid in gpu else []) \
               + (["ours"] if use_ours and rid in ours else [])
        src = {"gpu": gpu.get(rid), "ours": ours.get(rid, {}).get("ours")}
        mean = [src[k]["outage_mean"] if k in src else m[k]["outage_mean"] for k in keys]
        mx = [src[k]["outage_max"] if k in src else m[k]["outage_max"] for k in keys]
        panels.append((f"Route {rid}", keys, mean, mx))
    # test-set mean: mean of the per-route outage means / maxima, i.e. the same
    # number the README table quotes
    test_ids = [int(i) for i in load_config()["routes"]["test"] if _kf_metrics(int(i))]
    if test_ids:
        keys = ["B1", "B2", "B3"] \
            + (["gpu"] if all(r in gpu for r in test_ids) else []) \
            + (["ours"] if use_ours and all(r in ours for r in test_ids) else [])
        mean, mx = [], []
        for k in keys:
            if k == "gpu":
                vals = [gpu[r] for r in test_ids]
            elif k == "ours":
                vals = [ours[r]["ours"] for r in test_ids]
            else:
                vals = [_kf_metrics(r)[k] for r in test_ids]
            mean.append(float(np.mean([v["outage_mean"] for v in vals])))
            mx.append(float(np.max([v["outage_max"] for v in vals])))
        panels.append((f"Test mean ({len(test_ids)} routes)", keys, mean, mx))
    if not panels:
        return None

    n = len(panels)
    fig, ax = plt.subplots(1, n, figsize=(4.3 * n, 3.4), squeeze=False)
    for j, (title, keys, mean, mx) in enumerate(panels):
        a = ax[0, j]
        x = np.arange(len(mean))
        colors = [_F5_COLOR[k] for k in keys]
        a.bar(x - 0.2, mean, 0.4, color=colors, label="mean")
        a.bar(x + 0.2, mx, 0.4, color=colors, alpha=0.45, hatch="//", label="max")
        a.set_xticks(x)
        a.set_xticklabels([_F5_LABEL[k] for k in keys], fontsize=8)
        a.set_yscale("log")
        a.set_ylabel("3-D error in outage [m]")
        a.set_title(title)
        for xi, (mv, xv) in enumerate(zip(mean, mx)):
            a.text(xi - 0.2, mv * 1.08, f"{mv:.1f}", ha="center", fontsize=7)
            a.text(xi + 0.2, xv * 1.08, f"{xv:.0f}", ha="center", fontsize=7)
    ax[0, 0].legend(frameon=False, fontsize=8)
    if ours and not use_ours:
        _alpha_note(fig, alpha)
    fig.suptitle("GNSS-outage accuracy: dead reckoning -> GPU ESKF+RTS, "
                 "with the learned corrector on its validation gate", fontsize=10)
    fig.tight_layout()
    fig.savefig(out or _out("fig5_summary.png"))
    plt.close(fig)
    return out or _out("fig5_summary.png")


def _load_run(route_id: int) -> dict:
    path = sorted((RUNS_DIR / "kf").glob(f"route_{route_id:02d}_*.npz"))[0]
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _spans(outage):
    d = np.diff(outage.astype(np.int8))
    s = list(np.flatnonzero(d == 1) + 1)
    e = list(np.flatnonzero(d == -1) + 1)
    if outage[0]:
        s = [0] + s
    if outage[-1]:
        e = e + [len(outage)]
    return list(zip(s, e))


def _finite_runs(x: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous index runs where ``x`` is finite (so lines never bridge gaps)."""
    m = np.isfinite(x)
    if not m.any():
        return []
    edges = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _load_gpu(route_id: int):
    """Window-parallel GPU trajectory, if the HPC stage has been run."""
    paths = sorted((RUNS_DIR / "kf_gpu").glob(f"route_{route_id:02d}_*.npz"))
    if not paths:
        return None
    with np.load(paths[0]) as d:
        return d["p_smooth"] if "p_smooth" in d.files else None


def _best_model():
    """([name], alpha) of the evaluated model with the lowest mean test outage error.

    ``alpha`` is 0 when no metrics file exists, i.e. the corrector was never run.
    """
    names, best, alpha = ["gpt4ts", "lstm"], np.inf, 0.0
    for path in sorted(RESULTS_DIR.glob("metrics_*.json")):
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        vals = [r["ours"]["outage_mean"] for r in m.get("routes", {}).values() if "ours" in r]
        if vals and float(np.mean(vals)) < best:
            best = float(np.mean(vals))
            names = [m.get("model", "?")]
            alpha = float(m.get("alpha", 0.0))
    return names, alpha


def _load_ours(route_id: int):
    """(corrected trajectory, alpha) of the best evaluated model, else (None, 0).

    The model shown is the one with the lowest mean outage error over the test
    routes -- the same rule ``scripts/make_summary.py`` uses, so the figures and
    the table never quote different models.  ``alpha`` is returned so the plot
    can skip the curve when the shrinkage rejected the correction: at
    ``alpha = 0`` the "Ours" trajectory *is* the base trajectory, and drawing it
    twice only hides the curves underneath.
    """
    names, alpha = _best_model()
    for name in names + [n for n in ("gpt4ts", "lstm") if n not in names]:
        p = RUNS_DIR / "models" / f"{name}_test.npz"
        if p.exists():
            with np.load(p) as d:
                key = f"p_ours_{route_id}"
                if key in d.files:
                    return d[key], alpha
    return None, 0.0


def _alpha_note(fig, alpha: float) -> None:
    """Footnote explaining why the learned corrector is absent from the plot."""
    fig.text(0.5, -0.01,
             f"Learned corrector: $\\alpha$ = {alpha:.2f} from the one-standard-error "
             f"gate on the validation routes -- the correction does not clear the bar, "
             f"so it runs in pass-through and the output is the filter trajectory.",
             ha="center", fontsize=7, color="0.35")


def fig3_trajectory(route_id=6, out=None):
    """Top-down trajectory: overview + zoom on the first GNSS outage."""
    d = _load_run(route_id)
    p_gt, out_mask = d["p_gt"], d["outage"].astype(bool)
    fs = load_config()["fs_imu"]
    spans = _spans(out_mask)
    ours, alpha = _load_ours(route_id)
    gpu = _load_gpu(route_id)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    def draw(ax):
        ax.plot(p_gt[:, 0], p_gt[:, 1], color=COL["gt"], lw=1.6, label="Ground truth")
        tracks = [("p_b1", "B1  IMU dead reckoning", ":", 1.4),
                  ("p_b2", "B2  forward ESKF", "--", 1.2),
                  ("p_b3", "B3  ESKF + RTS (numpy)", "-", 1.2)]
        if gpu is not None:
            tracks.append(("p_gpu", "GPU  windowed ESKF + RTS", "-.", 1.1))
        if ours is not None and alpha > 0:
            tracks.append(("p_ours", "Ours  learned correction", "-", 1.6))
        for k, lab, ls, lw in tracks:
            p = ours if k == "p_ours" else (gpu if k == "p_gpu" else d[k])
            color = COL[k[2:]] if k[2:] in COL else COL["b3"]
            for j, (r0, r1) in enumerate(_finite_runs(p[:, 0])):
                ax.plot(p[r0:r1, 0], p[r0:r1, 1], color=color, ls=ls, lw=lw,
                        label=lab if j == 0 else None)
        for i, (i0, i1) in enumerate(spans):
            seg = p_gt[i0:i1: max(1, int(fs))]
            ax.plot(seg[:, 0], seg[:, 1], color="0.55", lw=7.0, alpha=0.55,
                    solid_capstyle="round", zorder=0,
                    label="GNSS outage" if i == 0 else None)

    draw(a)
    a.set_xlabel("East [m]")
    a.set_ylabel("North [m]")
    a.set_title(f"(a) Route {route_id}, full trajectory")
    a.legend(frameon=False, fontsize=8, loc="best")
    a.set_aspect("equal", adjustable="datalim")

    draw(b)
    i0, i1 = spans[0]
    pad = int(20 * fs)                       # +-20 s around the first outage
    lo, hi = max(0, i0 - pad), min(len(p_gt), i1 + pad)
    b.set_xlim(p_gt[lo:hi, 0].min() - 50, p_gt[lo:hi, 0].max() + 50)
    b.set_ylim(p_gt[lo:hi, 1].min() - 50, p_gt[lo:hi, 1].max() + 50)
    b.set_xlabel("East [m]")
    b.set_title("(b) Zoom: first GNSS outage")
    if ours is not None and alpha <= 0:
        _alpha_note(fig, alpha)
    fig.tight_layout()
    fig.savefig(out or _out("fig3_trajectory.png"))
    plt.close(fig)
    return out or _out("fig3_trajectory.png")


def fig4_error(route_id=6, out=None):
    d = _load_run(route_id)
    p_gt, out_mask = d["p_gt"], d["outage"].astype(bool)
    fs = load_config()["fs_imu"]
    t = np.arange(len(p_gt)) / fs
    fig, ax = plt.subplots(figsize=(9, 3.2))
    series = [("p_b1", "B1  dead reckoning", COL["b1"]),
              ("p_b2", "B2  forward ESKF", COL["b2"]),
              ("p_b3", "B3  ESKF + RTS (numpy)", COL["b3"])]
    ours, alpha = _load_ours(route_id)
    gpu = _load_gpu(route_id)
    if gpu is not None:
        series.append(("p_gpu", "GPU  windowed ESKF + RTS", COL["gpu"]))
    if ours is not None and alpha > 0:
        series.append(("p_ours", "Ours  learned correction", COL["ours"]))
    for k, lab, c in series:
        p = ours if k == "p_ours" else (gpu if k == "p_gpu" else d[k])
        e = np.linalg.norm(p - p_gt, axis=1)
        if k == "p_b1":
            # B1 is only defined inside the outage; past the window it keeps
            # diverging without re-anchoring (it never recovers), which would
            # otherwise dominate the panel and hide the other curves
            e = np.where(out_mask, e, np.nan)
        ax.plot(t, np.where(np.isfinite(p[:, 0]), e, np.nan), color=c, lw=1.0, label=lab)
    for i0, i1 in _spans(out_mask):
        ax.axvspan(i0 / fs, i1 / fs, color="grey", alpha=0.18, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("3-D position error [m]")
    ax.set_title(f"Route {route_id}: error during GNSS outages (grey bands)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.set_xlim(0, t[-1])
    if ours is not None and alpha <= 0:
        _alpha_note(fig, alpha)
    fig.tight_layout()
    fig.savefig(out or _out("fig4_error.png"))
    plt.close(fig)
    return out or _out("fig4_error.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", type=int, default=6)
    ap.add_argument("--figs", nargs="*", default=["fig1", "fig2", "fig3", "fig4", "fig5"])
    args = ap.parse_args()
    _style()
    made = []
    if "fig1" in args.figs:
        made.append(fig1_speedup())
    if "fig2" in args.figs:
        made.append(fig2_loss())
    if "fig3" in args.figs:
        made.append(fig3_trajectory(args.route))
    if "fig4" in args.figs:
        made.append(fig4_error(args.route))
    if "fig5" in args.figs:
        made.append(fig5_summary())
    for m in made:
        if m:
            print("saved", m)


if __name__ == "__main__":
    main()
