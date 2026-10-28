"""Is the causal-filter residual a fixed fraction of the displacement?

A scale-factor miscalibration makes the mechanisation integrate too little or
too much acceleration, so the position error over an outage should be

    delta P = kappa * (p(t) - p(last fix))

with the *same* kappa on every route (one physical IMU unit).  This script
measures kappa per outage segment: if it is stable across routes, a learned
corrector has something real to identify, and the regressor it needs is the
displacement, not the raw accelerometer.

    python scripts/check_scale.py [--routes 6 7 42 ...]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common import load_config                        # noqa: E402
from src.data.mini_imu_sim import simulate_route          # noqa: E402
from src.data.routes import route_specs                   # noqa: E402
from src.nav.ekf import initial_covariance, init_from_route, run_filter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", type=int, nargs="*", default=[6, 7, 42, 43, 44, 45])
    ap.add_argument("--base", choices=["b2", "b3"], default="b2")
    args = ap.parse_args()

    cfg = load_config()
    specs = {s["id"]: s for s in route_specs(int(cfg["routes"].get("n_aug", 0)))}
    ks, rs, k_axis = [], [], []
    for rid in args.routes:
        r = simulate_route(specs[rid], cfg)
        routed = {"gyro": r.gyro, "accel": r.accel, "gnss": r.gnss, "p": r.p, "q": r.q}
        p0, v0, q0, ba0, bg0 = init_from_route(routed, cfg)
        res = run_filter(r.gyro, r.accel, r.gnss, p0, v0, q0, ba0, bg0,
                         initial_covariance(cfg), cfg, smooth=(args.base == "b3"))
        base = res["p_smooth"] if args.base == "b3" else res["p"]
        gt, out = r.p, r.outage.astype(bool)
        fix = ~out
        last = np.maximum.accumulate(np.where(fix, np.arange(len(fix)), -1))
        st = np.flatnonzero(np.diff(np.r_[0, out.view(np.int8), 0]))
        line = []
        v = res["v"]
        dt = 1.0 / 100.0
        for a, b in zip(st[::2], st[1::2]):
            # Increment from the first outage sample, with the constant-velocity
            # part removed: a scale error integrates the *acceleration*, i.e.
            # kappa * (p(t) - p(a) - v(a) * (t - a)), not the raw displacement.
            t = (np.arange(a, b) - a) * dt
            disp = (base[a:b] - base[a]) - np.outer(t, v[a])
            resid = (gt[a:b] - base[a:b]) - (gt[a] - base[a])
            k = float((disp * resid).sum() / max((disp * disp).sum(), 1e-12))
            rr = np.corrcoef(disp.ravel(), resid.ravel())[0, 1]
            ks.append(k)
            rs.append(rr)
            # Per-axis: a unit-to-unit scale miscalibration is three numbers
            # [kx, ky, kz], not one.  Collapsing them into a single scalar
            # weights each axis by its share of the displacement, so axes with
            # opposite-signed scale errors average to ~0 and the sign of the
            # scalar fit flips with the route geometry -- which is exactly the
            # pattern that made the earlier scalar test look unstable.
            kax = (disp * resid).sum(0) / np.maximum((disp * disp).sum(0), 1e-12)
            k_axis.append(kax)
            line.append(f"{k:+.4f} (r={rr:+.2f}) axis[{kax[0]:+.3f} {kax[1]:+.3f} {kax[2]:+.3f}]")
        print(f"route {rid:2d}: " + "  ".join(line))
    print(f"\nkappa  mean {np.mean(ks):+.4f}  std {np.std(ks):.4f}  "
          f"min {np.min(ks):+.4f}  max {np.max(ks):+.4f}")
    print(f"corr   mean {np.mean(rs):+.3f}")
    ka = np.array(k_axis)
    print(f"kappa per axis  mean [{ka.mean(0)[0]:+.4f} {ka.mean(0)[1]:+.4f} {ka.mean(0)[2]:+.4f}]"
          f"  std [{ka.std(0)[0]:.4f} {ka.std(0)[1]:.4f} {ka.std(0)[2]:.4f}]")


if __name__ == "__main__":
    main()
