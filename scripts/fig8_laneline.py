"""Figure 8: what the trajectory error does to a map.

Trajectory correction is the upstream stage of map production.  A lane marking
is observed *relative to the vehicle* (a camera measures it to a few
centimetres) and placed in the world by adding the vehicle's pose.  A pose
error ``e`` therefore becomes a map error ``e``: the component normal to the
road shows up as the lateral offset of the mapped lane line -- the doubled,
smeared lane line you see in a map built from uncorrected trajectories -- while
the tangential component only slides features along the road.

Panels:

    (a) one outage, road frame: lateral offset of the mapped lane-marking
        points against distance along the road, with the true lane boundaries;
    (b) every test route: mean |lateral offset| for B1, B3, GPU, GPU + map;
    (c) the same samples as a distribution: what fraction of mapped points
        land within a lateral tolerance.

The lateral offset is measured as ``(p_est - p_gt) . n``, ``n`` the horizontal
normal of the *true* road at that instant.  This is the perpendicular distance
from the mapped point to the true lane line, and it is exactly what the map
consumer sees; the along-track error drops out of it, which is why it needs to
be reported separately.

    python scripts/fig8_laneline.py
"""
from __future__ import annotations

import argparse
import json
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
from src.common import RESULTS_DIR, load_config        # noqa: E402
from src.models.train import outage_gate               # noqa: E402
from src.plots.figures import COL, _out, _style        # noqa: E402

LANE_HALF = 1.75          # half of a 3.5 m lane, in metres
HD_TOL = 0.5              # lateral tolerance an HD map is usually built to
MAP_COL = COL["ours"]          # follows FIG_PALETTE (see src/plots/figures.py)
SERIES = [("b1", "B1  IMU dead reckoning", COL["b1"]),
          ("b3", "B3  ESKF + RTS (numpy)", COL["b3"]),
          ("gpu", "GPU  windowed ESKF + RTS", COL["gpu"]),
          ("map", "GPU + map prior", MAP_COL)]


def road_frame(p_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arclength and horizontal left-normal of the true road, per sample."""
    dxy = np.gradient(p_gt[:, :2], axis=0)
    u = dxy / np.maximum(np.linalg.norm(dxy, axis=1, keepdims=True), 1e-9)
    n = np.stack([-u[:, 1], u[:, 0]], axis=1)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p_gt[:, :2], axis=0), axis=1))]
    return s, n


def lateral(est: np.ndarray, gt: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Signed lateral offset of the estimated pose, + is left of the road."""
    return np.sum((est[:, :2] - gt[:, :2]) * n, axis=1)


def load_route(rid: int, fs: float, vwin: float, head_w: float):
    """Every trajectory for one route, the map-prior output included."""
    d, p_gpu = map_prior.load(rid, "gpu")
    out = d["outage"].astype(bool)
    _, xy, z = map_prior.build_road(d["p_gt"], 0.0, seed=rid)
    road = map_prior.track(xy, z, p_gpu, out, 1.0 / fs, v=d["v_b2"],
                           vwin=vwin, head_w=head_w)
    road = np.where(np.isnan(road), p_gpu, road)
    g = outage_gate(out, fs)
    p_map = (1.0 - g[:, None]) * p_gpu + g[:, None] * road
    return {"b1": d["p_b1"], "b3": d["p_b3"], "gpu": p_gpu, "map": p_map}, d["p_gt"], out


def longest_outage(out: np.ndarray) -> tuple[int, int]:
    st = np.flatnonzero(np.diff(np.r_[0, out.astype(np.int8), 0]))
    return max(zip(st[::2], st[1::2]), key=lambda s: s[1] - s[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vwin", type=float, default=20.0)
    ap.add_argument("--head-w", type=float, default=10.0)
    args = ap.parse_args()
    _style()
    cfg = load_config()
    fs = float(cfg["fs_imu"])
    test_ids = [int(r) for r in cfg["routes"]["test"]]

    # ------------------------------------------------------------- numbers
    lat = {k: [] for k, _, _ in SERIES}      # per-route mean |lateral|
    long = {k: [] for k, _, _ in SERIES}     # per-route mean |along-track|
    pooled = {k: [] for k, _, _ in SERIES}   # every outage sample, pooled
    tracks = {}
    for rid in test_ids:
        p, gt, out = load_route(rid, fs, args.vwin, args.head_w)
        _, n = road_frame(gt)
        dxy = np.gradient(gt[:, :2], axis=0)
        u = dxy / np.maximum(np.linalg.norm(dxy, axis=1, keepdims=True), 1e-9)
        for k, _, _ in SERIES:
            el = np.abs(lateral(p[k], gt, n)[out])
            ea = np.abs(np.sum((p[k][:, :2] - gt[:, :2]) * u, axis=1)[out])
            lat[k].append(float(el.mean()))
            long[k].append(float(ea.mean()))
            pooled[k].append(el)
        tracks[rid] = (p, gt, out)
        print(f"  route {rid:2d}: " + "  ".join(
            f"{k} {lat[k][-1]:6.2f}" for k, _, _ in SERIES) + "  m lateral")

    print("\n  mean |lateral| [m]:")
    for k, name, _ in SERIES:
        print(f"    {name:30s} {np.mean(lat[k]):7.2f}")
    print("  mean |along| [m]:")
    for k, name, _ in SERIES:
        print(f"    {name:30s} {np.mean(long[k]):7.2f}")

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(12.6, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.05, 1.0])

    # (a) one outage in the road frame
    rid_a = max(test_ids, key=lambda r: lat["gpu"][test_ids.index(r)])
    p, gt, out = tracks[rid_a]
    s, n = road_frame(gt)
    o0, o1 = longest_outage(out)
    pad = int(5 * fs)
    a0, a1 = max(0, o0 - pad), min(len(out), o1 + pad)
    a = fig.add_subplot(gs[0, 0])
    a.axhspan(-LANE_HALF, LANE_HALF, color="#e8e8e8", lw=0, zorder=0)
    for y in (LANE_HALF, -LANE_HALF):
        a.axhline(y, color="0.45", ls="--", lw=1.0)
    a.text(s[a0] + 8, 20.4, "true lane boundaries", fontsize=7, color="0.35")
    for xv in (s[o0], s[o1 - 1]):
        a.axvline(xv, color="0.3", ls=":", lw=1.0, zorder=1)
    a.text(0.5 * (s[o0] + s[o1]), 23.2, "GNSS outage", fontsize=7,
           color="0.3", ha="center")
    for k, name, col in SERIES:
        e = lateral(p[k][a0:a1], gt[a0:a1], n[a0:a1])
        a.plot(s[a0:a1], e, color=col, lw=1.0 if k == "b1" else 1.3,
               alpha=0.9 if k == "b1" else 1.0,
               label=f"{name.split('  ')[0]}  "
                     f"({lat[k][test_ids.index(rid_a)]:.1f} m mean)")
    a.annotate(f"B1 leaves the frame:\n{np.abs(lateral(p['b1'][o0:o1], gt[o0:o1], n[o0:o1])).max():.0f} m peak",
               xy=(0.985, 0.05), xycoords="axes fraction", fontsize=7,
               color=COL["b1"], ha="right")
    a.set_ylim(-25, 25)
    a.set_xlabel("Distance along the road [m]")
    a.set_ylabel("Lateral offset of the mapped lane line [m]")
    a.set_title(f"(a) Route {rid_a:02d}: one GNSS outage, road frame")
    a.legend(frameon=False, fontsize=7, loc="lower left", ncol=1)
    a.set_xlim(s[a0], s[a1 - 1])

    # (b) per-route dots on a log axis: the range spans three decades
    b = fig.add_subplot(gs[0, 1])
    x = np.arange(len(test_ids))
    w = 0.21
    for i, (k, name, col) in enumerate(SERIES):
        b.scatter(x + (i - 1.5) * w, lat[k], s=20, color=col,
                  label=name.split("  ")[0], zorder=3,
                  edgecolors="white", linewidths=0.4)
    b.axhline(HD_TOL, color="0.3", ls=":", lw=1.0)
    b.set_yscale("log")
    b.set_ylim(0.03, 200)
    b.set_xticks(x, [f"{r:02d}" for r in test_ids])
    b.set_xlim(-0.5, len(test_ids) - 0.5)
    b.set_xlabel("Test route")
    b.set_ylabel("Mean |lateral offset| [m]")
    b.set_title("(b) Every route improves")
    b.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center")
    b.annotate(f"route mean: {np.mean(lat['b1']):.1f} $\\rightarrow$ "
               f"{np.mean(lat['gpu']):.1f} $\\rightarrow$ "
               f"{np.mean(lat['map']):.2f} m",
               xy=(0.02, 0.97), xycoords="axes fraction", fontsize=8, va="top")
    b.text(-0.42, HD_TOL * 1.25, f"{HD_TOL} m", fontsize=7, color="0.3")

    # (c) distribution of the lateral offset
    c = fig.add_subplot(gs[0, 2])
    for k, name, col in SERIES:
        e = np.sort(np.abs(np.concatenate(pooled[k])))
        frac = np.arange(1, len(e) + 1) / len(e)
        c.plot(e, frac, color=col, lw=1.4, label=name.split("  ")[0])
    c.axvline(HD_TOL, color="0.3", ls=":", lw=1.2,
              label=f"HD-map tolerance ({HD_TOL} m)")
    c.set_xscale("log")
    c.set_xlabel("|Lateral offset| [m]")
    c.set_ylabel("Fraction of mapped points")
    c.set_title("(c) What lands within tolerance")
    c.set_xlim(0.05, 200)
    c.set_ylim(0, 1.02)
    c.legend(frameon=False, fontsize=7, loc="lower right")

    fig.tight_layout()
    out_path = _out("fig8_laneline.png")
    fig.savefig(out_path)
    print(f"-> {out_path}")

    rows = {str(rid): {k: lat[k][i] for k, _, _ in SERIES} for i, rid in enumerate(test_ids)}
    dst = RESULTS_DIR / "laneline.json"
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump({"lane_half_m": LANE_HALF, "hd_tolerance_m": HD_TOL,
                   "routes": rows,
                   "mean": {k: float(np.mean(lat[k])) for k, _, _ in SERIES},
                   "mean_along": {k: float(np.mean(long[k])) for k, _, _ in SERIES}},
                  fh, indent=2)
    print(f"-> {dst}")
    for k, name, _ in SERIES:
        e = np.concatenate(pooled[k])
        print(f"   within {HD_TOL} m: {name:30s} {100 * np.mean(e <= HD_TOL):5.1f} %")


if __name__ == "__main__":
    main()
