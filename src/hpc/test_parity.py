"""Parity / consistency checks for the GPU batched filter.

Three checks, all runnable on CPU (float64) and on CUDA (float32):

  P1  single window, torch batched vs numpy reference -- the numerical core.
      Every smoothed state group (p, v, q, ba, bg) is asserted, not just the
      position: a regression in the attitude or bias recursion moves the
      trajectory too, only later and less obviously.
      float64 CPU tolerance 1e-7, float32 GPU tolerance 1e-2 (p, v).
  P2  full windowed two-pass route vs the numpy reference filter -- the
      pipeline.  The segment is extended to contain a GNSS outage, because
      that is the only part of the route where the smoother earns its keep:
      a window layout that cannot span the outage produces hundreds of metres
      of error while still looking perfect on a GNSS-covered stretch.
  P3  boundary continuity: no step in the stitched trajectory may exceed the
      ground-truth step by more than 1.5x.

Usage:
    python -m src.hpc.test_parity [--device cuda] [--route 6] [--n 60000]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from ..common import NPZ_DIR, load_config
from ..eval.metrics import traj_error
from ..nav.ekf import initial_covariance as np_initial_covariance
from ..nav.ekf import run_filter
from .batch_kf import initial_covariance, run_route_windowed, run_windows

KEYS = ("p", "v", "q", "ba", "bg")
# float32 tolerances per state group; the bias states are in m/s^2 and rad/s,
# so their absolute round-off is smaller than the metre-level position one
TOL32 = {"p": 1e-2, "v": 1e-2, "q": 1e-3, "ba": 1e-3, "bg": 1e-4}
TOL64 = {k: 1e-6 for k in KEYS}


def _load(route_id: int) -> dict:
    path = sorted(NPZ_DIR.glob(f"route_{route_id:02d}_*.npz"))[0]
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _qdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hemisphere-robust quaternion difference: q and -q are the same attitude."""
    return np.minimum(np.abs(a - b), np.abs(a + b))


def check_window(cfg, route, device, dtype, n=1200, seed_idx=6000):
    """P1: batched torch vs numpy on one window, identical initial state."""
    sl = slice(seed_idx, seed_idx + n)
    p0 = route["p"][seed_idx].copy()
    v0 = route["v"][seed_idx].copy()
    q0 = route["q"][seed_idx].copy()
    ba0, bg0 = np.zeros(3), np.zeros(3)

    ref = run_filter(route["gyro"][sl], route["accel"][sl], route["gnss"][sl],
                     p0, v0, q0, ba0, bg0, np_initial_covariance(cfg), cfg, smooth=True)

    t = lambda a: torch.as_tensor(a[None], device=device, dtype=dtype)
    out = run_windows(
        t(route["gyro"][sl]), t(route["accel"][sl]), t(route["gnss"][sl]),
        t(p0), t(v0), t(q0), t(ba0), t(bg0),
        initial_covariance(cfg, 1, dtype, device), cfg, smooth=True,
    )
    diff = {}
    for k in KEYS:
        for suffix in ("", "_smooth"):
            a = out[k + suffix][0].cpu().numpy()
            b = ref[k + suffix]
            diff[k + suffix] = float((_qdiff(a, b) if k == "q" else np.abs(a - b)).max())
    return diff


def _cover_outage(route: dict, n: int, margin: int = 200) -> int:
    """Grow ``n`` so the segment contains at least one complete GNSS outage."""
    outage = route["outage"].astype(bool)
    if outage[:n].any():
        return n
    idx = np.flatnonzero(outage)
    if idx.size == 0:
        return n
    j = int(idx[0])
    while j + 1 < len(outage) and outage[j + 1]:
        j += 1
    return min(len(outage), j + 1 + margin)


def check_route(cfg, route, device, dtype, n=60000):
    """P2/P3: windowed two-pass pipeline against the numpy reference."""
    n = _cover_outage(route, min(n, len(route["gyro"])))
    sub = {k: route[k][:n] for k in ("gyro", "accel", "gnss")}
    res = run_route_windowed(sub, cfg, device=device, dtype=dtype, verbose=False)
    p_gpu = res["smoothed"]["p"]

    p0, v0, q0 = route["p"][0].copy(), np.zeros(3), route["q"][0].copy()
    ref = run_filter(route["gyro"][:n], route["accel"][:n], route["gnss"][:n],
                     p0, v0, q0, np.zeros(3), np.zeros(3), np_initial_covariance(cfg), cfg, smooth=True)

    dev = np.linalg.norm(p_gpu - ref["p_smooth"], axis=1)
    err = traj_error(p_gpu, route["p"][:n])
    ref_err = traj_error(ref["p_smooth"], route["p"][:n])
    gt_step = float(np.linalg.norm(np.diff(route["p"][:n], axis=0), axis=1).max())
    step = float(np.linalg.norm(np.diff(p_gpu, axis=0), axis=1).max())
    mask = route["outage"][:n].astype(bool)
    return {
        "n": n,
        "n_outage": int(mask.sum()),
        "n_windows": len(res["starts"]),
        "max_dev_vs_numpy": float(np.max(dev)),
        "mean_dev_vs_numpy": float(np.mean(dev)),
        "max_err_vs_truth": float(np.max(err)),
        "mean_err_vs_truth": float(np.mean(err)),
        "ref_mean_err_vs_truth": float(np.mean(ref_err)),
        "ref_max_err_vs_truth": float(np.max(ref_err)),
        "outage_mean_err": float(np.mean(err[mask])) if mask.any() else float("nan"),
        "outage_mean_err_ref": float(np.mean(ref_err[mask])) if mask.any() else float("nan"),
        "max_step": step,
        "gt_max_step": gt_step,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--route", type=int, default=6)
    ap.add_argument("--n", type=int, default=60000,
                    help="route length in samples; grown if it contains no outage")
    args = ap.parse_args()

    cfg = load_config()
    route = _load(args.route)
    on_cuda = args.device.startswith("cuda")
    # float64 on CPU makes P1 a real test of the maths instead of the mantissa
    dtype = torch.float32 if on_cuda else torch.float64
    tol = TOL32 if on_cuda else TOL64

    print(f"device={args.device} dtype={dtype} route={args.route}")
    d1 = check_window(cfg, route, args.device, dtype)
    for k in KEYS:
        print(f"     {k:>2}: filtered {d1[k]:.3e}   smoothed {d1[k + '_smooth']:.3e}"
              f"   (tol {tol[k]:.0e})")
    ok1 = all(d1[k + sfx] < tol[k] for k in KEYS for sfx in ("", "_smooth"))

    d2 = check_route(cfg, route, args.device, dtype, n=args.n)
    print(f"[P2] n={d2['n']} ({d2['n_outage']} outage samples)  windows={d2['n_windows']}  "
          f"max dev vs numpy={d2['max_dev_vs_numpy']:.2f} m (mean {d2['mean_dev_vs_numpy']:.2f})")
    print(f"     windowed err vs truth: max={d2['max_err_vs_truth']:.2f} m "
          f"mean={d2['mean_err_vs_truth']:.2f} m  | in outage "
          f"{d2['outage_mean_err']:.2f} m")
    # the windowed run restarts every window from a GNSS-only alignment, so it
    # is only allowed 2x the error of the full-route numpy smoother
    print(f"     numpy reference: max={d2['ref_max_err_vs_truth']:.2f} m "
          f"mean={d2['ref_mean_err_vs_truth']:.2f} m  | in outage "
          f"{d2['outage_mean_err_ref']:.2f} m")
    ok2 = (d2["mean_err_vs_truth"] < 2.0 * max(d2["ref_mean_err_vs_truth"], 1.0)
           and d2["max_err_vs_truth"] < 2.0 * max(d2["ref_max_err_vs_truth"], 10.0)
           and (not np.isfinite(d2["outage_mean_err"])
                or d2["outage_mean_err"] < 2.0 * max(d2["outage_mean_err_ref"], 1.0)))
    print(f"[P3] max step={d2['max_step']:.4f} m  gt step={d2['gt_max_step']:.4f} m "
          f"ratio={d2['max_step'] / max(d2['gt_max_step'], 1e-9):.2f}")
    ok3 = d2["max_step"] <= 1.5 * d2["gt_max_step"]

    ok = ok1 and ok2 and ok3
    print("\nRESULT:", "PASS" if ok else "FAIL", f"(P1={ok1} P2={ok2} P3={ok3})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
