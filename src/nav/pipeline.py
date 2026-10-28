"""Run the numpy reference baselines (B1 / B2 / B3) on simulated routes.

B1  pure IMU dead reckoning evaluated inside each GNSS-outage window: the
    integrator is restarted from the last GNSS fix (position) plus the filter's
    velocity / attitude at that instant, with zero bias compensation.
B2  forward 15-state ESKF.
B3  ESKF + full-state RTS smoother (offline).

Usage:
    python -m src.nav.pipeline [--routes 0 1 2 ...]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..common import NPZ_DIR, RUNS_DIR, ensure_dirs, load_config
from ..eval.metrics import summarize, traj_error
from .ekf import initial_covariance, init_from_route, run_dead_reckoning, run_filter

RECOVERY_HORIZON_S = 30.0


def load_route(path):
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def route_paths() -> list:
    return sorted(NPZ_DIR.glob("route_*.npz"))


def outage_spans(outage: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(outage.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if outage[0]:
        starts = [0] + starts
    if outage[-1]:
        ends = ends + [len(outage)]
    return list(zip(starts, ends))


def b1_outage_dr(gyro, accel, gnss, outage, v_ref, q_ref, fs: float) -> np.ndarray:
    """Pure DR inside each outage; NaN elsewhere."""
    n = len(gyro)
    p_b1 = np.full((n, 3), np.nan)
    fixes = np.flatnonzero(np.isfinite(gnss[:, 0]))
    horizon = int(round(RECOVERY_HORIZON_S * fs))
    for i0, i1 in outage_spans(outage):
        prior = fixes[fixes < i0]
        if len(prior) == 0:
            continue
        k0 = int(prior[-1])
        end = min(i1 + horizon, n)
        p0 = gnss[k0].copy()
        p_hist, _ = run_dead_reckoning(gyro[k0:end], accel[k0:end], p0, v_ref[k0].copy(), q_ref[k0].copy(), fs)
        p_b1[k0:end] = p_hist
    return p_b1


def run_route(npz_path, cfg: dict, save: bool = True) -> dict:
    d = load_route(npz_path)
    fs = float(cfg["fs_imu"])
    gyro, accel, gnss, p_gt, outage = d["gyro"], d["accel"], d["gnss"], d["p"], d["outage"].astype(bool)
    p0, v0, q0, ba0, bg0 = init_from_route(d, cfg)
    P0 = initial_covariance(cfg)

    t0 = time.time()
    res = run_filter(gyro, accel, gnss, p0, v0, q0, ba0, bg0, P0, cfg, smooth=True)
    t_kf = time.time() - t0

    p_b2, p_b3 = res["p"], res["p_smooth"]
    p_b1 = b1_outage_dr(gyro, accel, gnss, outage, res["v"], res["q"], fs)

    e1 = traj_error(p_b1, p_gt)
    e2 = traj_error(p_b2, p_gt)
    e3 = traj_error(p_b3, p_gt)

    metrics = {
        "B1": {
            "rmse_all": float("nan"),
            "rmse_gnss": float("nan"),
            "outage_max": float(np.nanmax(e1[outage])),
            "outage_mean": float(np.nanmean(e1[outage])),
            "recovery_s": float("nan"),
        },
        "B2": summarize(e2, outage, fs),
        "B3": summarize(e3, outage, fs),
    }

    name = npz_path.stem
    print(
        f"{name:<28s} ({t_kf:5.1f}s)  "
        f"B1 out max={metrics['B1']['outage_max']:8.1f} | "
        f"B2 max={metrics['B2']['outage_max']:8.1f} mean={metrics['B2']['outage_mean']:6.1f} | "
        f"B3 max={metrics['B3']['outage_max']:8.1f} mean={metrics['B3']['outage_mean']:6.1f}"
    )

    out = {
        "p_gt": p_gt,
        "outage": outage,
        "p_b1": p_b1,
        "p_b2": p_b2,
        "p_b3": p_b3,
        "v_b2": res["v"],
        "q_b2": res["q"],
        "ba_b2": res["ba"],
        "bg_b2": res["bg"],
        "innovation": res["innovation"],
        "gnss": gnss,
        "gyro": gyro,
        "accel": accel,
        "metrics": metrics,
    }
    if save:
        out_dir = RUNS_DIR / "kf"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{name}.npz", **{k: v for k, v in out.items() if k != "metrics"})
        with open(out_dir / f"{name}.json", "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs()
    paths = route_paths()
    if args.routes is not None:
        paths = [p for p in paths if int(p.stem.split("_")[1]) in args.routes]

    all_metrics = {}
    for path in paths:
        out = run_route(path, cfg)
        all_metrics[path.stem] = out["metrics"]

    with open(RUNS_DIR / "kf" / "metrics_all.json", "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"\nmetrics -> {RUNS_DIR / 'kf' / 'metrics_all.json'}")


if __name__ == "__main__":
    main()
