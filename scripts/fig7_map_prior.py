"""Figure 7: what a road map adds to the outage fix.

The learned corrector cannot help because the residual carries no information
the filter has not already used (fig. 6).  A map is different: it is *new*
information.  The filter's outage error grows both along the road and across
it; the map removes the across-track part, because the vehicle is on the
carriageway.

Panels:

    (a) one outage: ground truth, the filter, and the map-projected fix;
    (b) every test route: B3, B3 + map, and (when available) the
        window-parallel GPU output + map;
    (c) robustness to survey error -- the centreline is perturbed by a
        smooth random offset of `noise` metres rms.

    python scripts/fig7_map_prior.py [--base b3]
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import map_prior                                       # noqa: E402
from src.common import RUNS_DIR, load_config           # noqa: E402
from src.eval.metrics import summarize, traj_error     # noqa: E402
from src.models.train import outage_gate               # noqa: E402
from src.plots.figures import COL, _out, _style        # noqa: E402

MAP_COL = COL["ours"]          # follows FIG_PALETTE (see src/plots/figures.py)


def compute(rid, base, fs, noise=0.0, vwin=20.0, head_w=10.0):
    """Return (corrected trajectory, base trajectory, gt, outage mask)."""
    d, p = map_prior.load(rid, base)
    out = d["outage"].astype(bool)
    _, xy, z = map_prior.build_road(d["p_gt"], noise, seed=rid)
    road = map_prior.track(xy, z, p, out, 1.0 / fs, v=d["v_b2"],
                           vwin=vwin, head_w=head_w)
    road = np.where(np.isnan(road), p, road)
    g = outage_gate(out, fs)
    return (1.0 - g[:, None]) * p + g[:, None] * road, p, d["p_gt"], out


def outage_mean(p, gt, out, fs):
    return summarize(traj_error(p, gt), out, fs)["outage_mean"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="b3", choices=["b3", "b2", "gpu"])
    ap.add_argument("--vwin", type=float, default=20.0)
    ap.add_argument("--head-w", type=float, default=10.0)
    args = ap.parse_args()
    _style()
    cfg = load_config()
    fs = float(cfg["fs_imu"])
    test_ids = [int(r) for r in cfg["routes"]["test"]]

    has_gpu = all(glob.glob(str(RUNS_DIR / "kf_gpu" / f"route_{r:02d}_*.npz"))
                  for r in test_ids)

    # ---------------------------------------------------------------- numbers
    rows = {}
    for rid in test_ids:
        p, base_p, gt, out = compute(rid, args.base, fs, 0.0, args.vwin, args.head_w)
        r = {"base": outage_mean(base_p, gt, out, fs),
             "map": outage_mean(p, gt, out, fs)}
        if has_gpu:
            pg, pb, _, _ = compute(rid, "gpu", fs, 0.0, args.vwin, args.head_w)
            r["gpu_base"] = outage_mean(pb, gt, out, fs)
            r["gpu_map"] = outage_mean(pg, gt, out, fs)
        rows[rid] = r

    # robustness: perturb the map, keep everything else identical
    noises = [0.0, 0.25, 0.5, 1.0, 2.0]
    sweep = {n: [] for n in noises}
    for rid in test_ids:
        for n in noises:
            p, _, gt, out = compute(rid, args.base, fs, n, args.vwin, args.head_w)
            sweep[n].append(outage_mean(p, gt, out, fs))

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(12.6, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0])

    # (a) one outage, error vs time: the map removes the across-track part
    a = fig.add_subplot(gs[0, 0])
    rid = max(rows, key=lambda k: rows[k]["base"])
    d, base_p = map_prior.load(rid, args.base)
    out = d["outage"].astype(bool)
    gt = d["p_gt"]
    p_map, _, _, _ = compute(rid, args.base, fs, 0.0, args.vwin, args.head_w)
    st = np.flatnonzero(np.diff(np.r_[0, out.astype(np.int8), 0]))
    segs = list(zip(st[::2], st[1::2]))
    a0, a1 = max(segs, key=lambda s: np.linalg.norm(
        (base_p[s[0]:s[1]] - gt[s[0]:s[1]]), axis=1).mean())
    t = (np.arange(a1 - a0)) / fs
    e_base = np.linalg.norm(base_p[a0:a1] - gt[a0:a1], axis=1)
    e_map = np.linalg.norm(p_map[a0:a1] - gt[a0:a1], axis=1)
    a.fill_between(t, 0, e_base.max() * 1.08, color="#dddddd", alpha=0.5, lw=0)
    a.plot(t, e_base, color=COL["b3"], lw=1.5,
           label=f"{args.base.upper()} ({e_base.mean():.1f} m mean)")
    a.plot(t, e_map, color=MAP_COL, lw=1.6,
           label=f"+ map prior ({e_map.mean():.1f} m mean)")
    a.set_xlabel("Time since outage onset [s]")
    a.set_ylabel("3-D position error [m]")
    a.set_title(f"(a) Route {rid:02d}: one GNSS outage")
    a.legend(frameon=False, fontsize=8, loc="lower right")
    a.set_xlim(0, t[-1])
    a.set_ylim(0, e_base.max() * 1.08)

    # (b) per-route bars
    b = fig.add_subplot(gs[0, 1])
    x = np.arange(len(test_ids))
    w = 0.38
    b.bar(x - w / 2, [rows[r]["base"] for r in test_ids], w,
          color=COL["b3"], label=f"{args.base.upper()}")
    b.bar(x + w / 2, [rows[r]["map"] for r in test_ids], w,
          color=MAP_COL, label="+ map prior")
    mb = np.mean([rows[r]["base"] for r in test_ids])
    mm = np.mean([rows[r]["map"] for r in test_ids])
    b.axhline(mb, color=COL["b3"], ls="--", lw=1.0)
    b.axhline(mm, color=MAP_COL, ls="--", lw=1.0)
    b.annotate(f"mean {mb:.1f} -> {mm:.1f} m ({100 * (1 - mm / mb):+.0f} %)",
               xy=(0.02, 0.95), xycoords="axes fraction", fontsize=8, va="top")
    b.set_xticks(x, [f"{r:02d}" for r in test_ids])
    b.set_xlabel("Test route")
    b.set_ylabel("Outage mean error [m]")
    b.set_title("(b) Every route improves")
    b.legend(frameon=False, fontsize=8, loc="upper right")

    # (c) robustness to survey error
    c = fig.add_subplot(gs[0, 2])
    mean = [np.mean(sweep[n]) for n in noises]
    for rid in test_ids:
        c.plot(noises, [sweep[n][test_ids.index(rid)] for n in noises],
               color=COL["b3"], lw=0.7, alpha=0.35)
    c.plot(noises, mean, "o-", color=MAP_COL, lw=1.8, label="test-route mean")
    c.axhline(mb, color=COL["b3"], ls="--", lw=1.0, label=f"{args.base.upper()} baseline")
    c.set_xlabel("Map centreline error, rms [m]")
    c.set_ylabel("Outage mean error [m]")
    c.set_yscale("log")
    c.set_title("(c) Tolerates a ~1 m survey error")
    c.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    out_path = _out(f"fig7_map_prior_{args.base}.png")
    fig.savefig(out_path)
    print(f"-> {out_path}")
    print(f"   (b) mean {mb:.2f} -> {mm:.2f} m ({100 * (1 - mm / mb):+.1f} %)")
    for n in noises:
        print(f"   (c) map noise {n:4.2f} m -> mean {np.mean(sweep[n]):6.2f} m")
    if has_gpu:
        gb = np.mean([rows[r]["gpu_base"] for r in test_ids])
        gm = np.mean([rows[r]["gpu_map"] for r in test_ids])
        print(f"   GPU: {gb:.2f} -> {gm:.2f} m ({100 * (1 - gm / gb):+.1f} %)")


if __name__ == "__main__":
    main()
