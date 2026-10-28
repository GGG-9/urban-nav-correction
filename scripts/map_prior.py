"""Map prior: snap the GNSS-outage trajectory onto the road centreline.

The filter has no idea where the road is, so during an outage its error grows in
*both* directions: along the road and across it.  A map removes the second one,
because the vehicle is on the carriageway.  It cannot remove the first - nor
should it, since where the vehicle is *along* the road is genuinely unknown.

In this simulation the road centreline is the route's ground-truth path; the
method uses only its geometry, never the timing.  `--map-noise` perturbs the
centreline to emulate a survey map that is not exact.

    python scripts/map_prior.py --base b3
    python scripts/map_prior.py --base gpu --map-noise 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common import RUNS_DIR, load_config               # noqa: E402
from src.eval.metrics import summarize, traj_error         # noqa: E402
from src.models.train import outage_gate                   # noqa: E402


def load(rid: int, base: str):
    path = sorted(glob.glob(str(RUNS_DIR / "kf" / f"route_{rid:02d}_*.npz")))[0]
    with np.load(path, allow_pickle=True) as d:
        d = {k: d[k] for k in d.files}
    if base == "gpu":
        g = sorted(glob.glob(str(RUNS_DIR / "kf_gpu" / f"route_{rid:02d}_*.npz")))[0]
        with np.load(g) as z:
            p = z["p_smooth"]
        if len(p) != len(d["p_gt"]):
            raise ValueError(f"gpu route {rid}: {len(p)} vs {len(d['p_gt'])}")
    else:
        p = d[f"p_{base}"]
    return d, p


def build_road(p_gt: np.ndarray, noise: float = 0.0, seed: int = 0):
    """The map: a 2-D polyline plus its elevation.  Geometry only, no timing."""
    xy = p_gt[:, :2].astype(np.float64)
    if noise > 0:
        rng = np.random.default_rng(seed)
        # a survey error is smooth, not white: low-pass the perturbation
        e = rng.normal(0.0, 1.0, xy.shape)
        k = np.ones(101) / 101.0
        e = np.stack([np.convolve(e[:, 0], k, "same"), np.convolve(e[:, 1], k, "same")], 1)
        xy = xy + e * noise
    return cKDTree(xy), xy, p_gt[:, 2].astype(np.float64)


def project(tree, xy, z, p):
    """Nearest point on the centreline, no continuity - the naive baseline."""
    _, idx = tree.query(np.asarray(p, dtype=np.float64)[:, :2])
    return np.c_[xy[idx], z[idx]]


def track(xy, z, p, mask, dt, vmax=45.0, vwin=3.0, v=None, head_w=10.0):
    """March along the centreline instead of jumping to the nearest point.

    A vehicle cannot teleport: between two samples its arclength advances by
    ``v*dt`` with ``v`` bounded by the vehicle's speed.  Restricting the search
    to that interval removes the ambiguity at self-intersections, which is what
    makes route 46 (which retraces its own path) unusable for a nearest-point
    lookup.

    ``v`` is the filter's own speed estimate, used only to tighten the window.
    """
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    # road tangent at every vertex, used as a second matching cue: a branch
    # running the other way is rejected even when it is the nearest point
    tan = np.gradient(xy, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    out = np.full((len(p), 3), np.nan)
    step = max(int(round(1.0 / dt)), 1)          # track at 1 Hz, interpolate back
    # The track runs over the *whole* route, not just the outages.  A route can
    # pass within a few metres of itself; seeding the search at the outage edge
    # then picks the other branch and the whole segment follows the wrong road
    # (measured: route 47, 2.3 m from the truth but 887 m off along the road).
    # Matching continuously keeps the arclength on the right branch, which is
    # also how a real map matcher runs.
    ks = list(range(0, len(p), step))
    if ks[-1] != len(p) - 1:
        ks.append(len(p) - 1)
    # The drive starts where the route starts: seeding by a global nearest
    # lookup is not safe because a loop can pass within a few metres of its own
    # origin (route 46 does, 6 m away at s = 1007 m).
    cur = 0.0
    pos = []
    for k in ks:
        span = step * dt
        sp = float(np.linalg.norm(v[k])) if v is not None else vmax * 0.5
        lo = cur + max(0.0, sp - vwin) * span
        hi = cur + min(vmax, sp + vwin) * span
        i0, i1 = np.searchsorted(s, [lo, hi])
        seg = np.arange(i0, max(i1, i0 + 1)).clip(0, len(xy) - 1)
        d = np.linalg.norm(xy[seg] - p[k, :2], axis=1)
        if v is not None and np.linalg.norm(v[k, :2]) > 1.0 and head_w > 0:
            hd = v[k, :2] / np.linalg.norm(v[k, :2])
            cosang = tan[seg] @ hd
            d = np.sqrt(d * d + head_w * head_w * (1.0 - cosang) ** 2)
        j = int(seg[d.argmin()])
        cur = s[j]
        pos.append((xy[j, 0], xy[j, 1], z[j]))
    pos = np.asarray(pos)
    for c in range(3):
        out[:, c] = np.interp(np.arange(len(p)), np.asarray(ks), pos[:, c])
    out[~mask] = np.nan          # only the outage is replaced by the map
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="b3", choices=["b3", "b2", "gpu"])
    ap.add_argument("--map-noise", type=float, default=0.0,
                    help="rms perturbation of the centreline [m]")
    ap.add_argument("--policy", default="track", choices=["track", "nearest"],
                    help="continuity-constrained march along the road, or the "
                         "naive nearest-point projection it is compared against")
    ap.add_argument("--vwin", type=float, default=20.0,
                    help="speed uncertainty around the filter's own speed [m/s]; "
                         "large = monotone nearest-point, small = hard speed prior")
    ap.add_argument("--head-w", type=float, default=10.0,
                    help="weight of the heading mismatch in the matching cost [m]")
    args = ap.parse_args()
    cfg = load_config()
    fs = float(cfg["fs_imu"])
    test_ids = [int(r) for r in cfg["routes"]["test"]]

    print(f"base = {args.base}   map noise = {args.map_noise} m")
    b, o, rows = [], [], {}
    for rid in test_ids:
        d, base = load(rid, args.base)
        out = d["outage"].astype(bool)
        tree, xy, z = build_road(d["p_gt"], args.map_noise, seed=rid)
        if args.policy == "nearest":
            road = project(tree, xy, z, base)
        else:
            road = track(xy, z, base, out, 1.0 / fs, v=d["v_b2"],
                         vwin=args.vwin, head_w=args.head_w)
            road = np.where(np.isnan(road), base, road)
        g = outage_gate(out, fs)
        p = (1.0 - g[:, None]) * base + g[:, None] * road
        m0 = summarize(traj_error(base, d["p_gt"]), out, fs)
        m1 = summarize(traj_error(p, d["p_gt"]), out, fs)
        e0, e1 = m0["outage_mean"], m1["outage_mean"]
        b.append(e0)
        o.append(e1)
        rows[str(rid)] = {"base": float(e0), "map": float(e1),
                          "base_max": float(m0["outage_max"]),
                          "map_max": float(m1["outage_max"])}
        print(f"  route {rid:2d}: {e0:7.2f} -> {e1:7.2f} m  ({100 * (1 - e1 / e0):+5.1f} %)")
    print(f"  mean        {np.mean(b):7.2f} -> {np.mean(o):7.2f} m"
          f"  ({100 * (1 - np.mean(o) / np.mean(b)):+.1f} %)")
    if args.map_noise == 0.0 and args.policy == "track":
        from src.common import RESULTS_DIR
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        dst = RESULTS_DIR / f"map_prior_{args.base}.json"
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump({"base": args.base, "vwin": args.vwin,
                       "head_w": args.head_w, "routes": rows,
                       "mean": {"base": float(np.mean(b)), "map": float(np.mean(o))}},
                      fh, indent=2)
        print(f"  -> {dst}")


if __name__ == "__main__":
    main()
