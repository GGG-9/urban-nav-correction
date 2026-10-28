"""Headroom check for an NLOS-spike GNSS error model.

Urban-canyon GNSS fails in bursts: for a few seconds the fix is off by tens of
metres (reflection / blockage), then it snaps back.  A Kalman filter with a
Gaussian R swallows the burst whole -- and if the burst lands just before an
outage it poisons exactly the state the dead-reckoning starts from.

This script injects that error on top of the clean fixes *after* simulation
(no simulator change), then measures three filters on the same truth:

    clean     the existing fixes (the ceiling a perfect detector could reach),
    ungated   the fixes with the injected bursts, ordinary filter,
    gated     the same, plus a classical innovation gate.

ungated -> gated is what a learned detector can hope to reach on its own;
gated -> clean is what is left for a model that estimates the burst magnitude
instead of discarding the fix.

    python scripts/probe_nlos.py
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
from src.eval.metrics import summarize                    # noqa: E402
from src.nav.ekf import initial_covariance, init_from_route, run_filter  # noqa: E402


def inject_nlos(gnss, fs_gnss, rng, n_burst=8, amp=(10.0, 30.0), dur=(1.0, 4.0),
                outage=None):
    """Smooth correlated bursts added to the fixes; half of them near an
    outage edge, where they hurt most."""
    g = gnss.copy()
    step = int(round(100.0 / fs_gnss))
    idx = np.flatnonzero(np.isfinite(g[:, 0]))
    edges = np.flatnonzero(np.diff(np.r_[0, outage.view(np.int8), 0]))
    edges = edges[::2]
    mask = np.zeros(len(idx), dtype=bool)
    for b in range(n_burst):
        if outage is not None and b % 2 == 0 and len(edges):
            e = int(edges[b // 2 % len(edges)])
            k0 = int(np.searchsorted(idx, max(e - 15 * step, idx[0])))
        else:
            k0 = int(rng.integers(0, max(len(idx) - 10, 1)))
        n = int(rng.uniform(*dur) * fs_gnss)
        a = rng.normal(0.0, 1.0, 3)
        a[2] *= 0.3
        a = a / np.linalg.norm(a) * rng.uniform(*amp)
        ramp = np.hanning(max(n, 3))[:, None]
        seg = slice(k0, min(k0 + n, len(idx)))
        g[idx[seg]] += a * ramp[: seg.stop - seg.start]
        mask[seg] = True
    return g, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", type=int, nargs="*", default=[6, 7, 42, 43, 44, 45])
    ap.add_argument("--thr", type=float, nargs="*", default=[10.0, 15.0, 20.0],
                    help="innovation gate threshold [m] (3D norm)")
    args = ap.parse_args()

    cfg = load_config()
    specs = {s["id"]: s for s in route_specs(int(cfg["routes"].get("n_aug", 0)))}
    fs = float(cfg["fs_imu"])
    agg = {k: [] for k in ["clean", "ungated"] + [f"gate {t:g}" for t in args.thr]}
    for rid in args.routes:
        r = simulate_route(specs[rid], cfg)
        rng = np.random.default_rng(1000 + rid)
        out = r.outage.astype(bool)
        gnss_bad, mask = inject_nlos(r.gnss, cfg["fs_gnss"], rng, outage=out)
        routed = {"gyro": r.gyro, "accel": r.accel, "gnss": r.gnss, "p": r.p, "q": r.q}
        p0, v0, q0, ba0, bg0 = init_from_route(routed, cfg)
        P0 = initial_covariance(cfg)
        run = lambda g: run_filter(r.gyro, r.accel, g, p0, v0, q0, ba0, bg0, P0, cfg)
        err = lambda res: summarize(np.linalg.norm(res["p"] - r.p, axis=1), out, fs)

        res_c = run(r.gnss)
        res_b = run(gnss_bad)
        inn = res_b["innovation"]
        fix = np.isfinite(gnss_bad[:, 0])
        agg["clean"].append(err(res_c)["outage_mean"])
        agg["ungated"].append(err(res_b)["outage_mean"])
        line = (f"route {rid:2d}: clean {agg['clean'][-1]:6.2f}  "
                f"ungated {agg['ungated'][-1]:7.2f}")
        for t in args.thr:
            bad = fix & (np.linalg.norm(inn, axis=1) > t)
            g = gnss_bad.copy()
            g[bad] = np.nan
            e = err(run(g))["outage_mean"]
            agg[f"gate {t:g}"].append(e)
            line += f"  gate{t:g} {e:7.2f} ({100 * bad.sum() / max(fix.sum(), 1):4.1f}% flagged)"
        print(line)
        print(f"           injected on {100 * mask.sum() / len(mask):4.1f}% of the fixes")
    print()
    for k, v in agg.items():
        print(f"{k:10s} mean outage error {np.mean(v):6.2f} m")


if __name__ == "__main__":
    main()
