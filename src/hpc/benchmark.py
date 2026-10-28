"""CPU-serial vs GPU-batched throughput benchmark for the window-parallel KF+RTS.

Protocol (fixed, reported in the paper):
  * identical inputs and identical code path on both devices (``run_windows``);
  * each *window* is one independent 20 s batch element (``bench.window_T``,
    deliberately shorter than the pipeline's outage-spanning window: this
    measures kernel throughput, not outage recovery);
  * the timed region excludes the host-to-device copy on *both* devices --
    the tensors are transferred once before the timer starts, so the number
    reported is pure filter throughput;
  * CPU baseline = the same function with batch size 1, windows processed one
    after another (the naive implementation);
  * GPU = windows grouped into batches of ``bench.chunk``;
  * ``n_repeat`` repeats, median reported;
  * sizes with more than ``CPU_WINDOWS_FULL`` windows are extrapolated
    linearly from the first eight timed windows (the serial per-window cost is
    constant to a few percent); the smaller sizes are measured in full, so the
    linearity is visible in the same figure.

Usage:
    python -m src.hpc.benchmark [--device cuda] [--sizes 10000 100000] [--quick]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..common import NPZ_DIR, RESULTS_DIR, load_config
from .batch_kf import initial_covariance, run_windows

CPU_WINDOWS_FULL = 8       # sizes with <= this many windows are timed in full


def load_route(route_id: int = 6) -> dict:
    path = sorted(NPZ_DIR.glob(f"route_{route_id:02d}_*.npz"))[0]
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def make_batch(route: dict, n_win: int, T: int, seed: int = 0) -> dict:
    """Slice ``n_win`` independent windows of length T out of the route."""
    n = len(route["gyro"])
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - T, size=n_win)
    idx = starts[:, None] + np.arange(T)[None, :]
    return {
        "gyro": route["gyro"][idx].astype(np.float32),
        "accel": route["accel"][idx].astype(np.float32),
        "gnss": route["gnss"][idx].astype(np.float32),
    }


def _initial_states(route: dict, n_win: int, T: int, dtype, device):
    p = route["p"][0].astype(np.float32)
    q = route["q"][0].astype(np.float32)
    z3 = torch.zeros(n_win, 3, dtype=dtype, device=device)
    return (
        torch.as_tensor(np.tile(p, (n_win, 1)), dtype=dtype, device=device),
        z3.clone(),
        torch.as_tensor(np.tile(q, (n_win, 1)), dtype=dtype, device=device),
        z3.clone(),
        z3.clone(),
    )


def time_cpu(batch: dict, states, cfg, n_win, T, repeats=3, max_windows=None):
    """Serial CPU baseline: one window at a time, same code path."""
    p0, v0, q0, ba0, bg0 = states
    P0 = initial_covariance(cfg, 1, torch.float32, "cpu")
    timed = n_win if (max_windows is None or n_win <= max_windows) else max_windows
    # convert once, exactly like the GPU path does, so the timed region contains
    # only the filter itself on both devices
    g = torch.as_tensor(batch["gyro"])
    a = torch.as_tensor(batch["accel"])
    z = torch.as_tensor(batch["gnss"])
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(timed):
            run_windows(
                g[i][None], a[i][None], z[i][None],
                p0[i:i + 1].cpu(), v0[i:i + 1].cpu(), q0[i:i + 1].cpu(),
                ba0[i:i + 1].cpu(), bg0[i:i + 1].cpu(), P0, cfg, smooth=True,
            )
        samples.append(time.perf_counter() - t0)
    t_med = float(np.median(samples)) * (n_win / timed)
    return t_med, timed < n_win


def time_gpu(batch: dict, states, cfg, device, chunk, repeats=3):
    n_win = batch["gyro"].shape[0]
    p0, v0, q0, ba0, bg0 = states
    P0 = initial_covariance(cfg, chunk, torch.float32, device)
    g_all = torch.as_tensor(batch["gyro"], device=device)
    a_all = torch.as_tensor(batch["accel"], device=device)
    z_all = torch.as_tensor(batch["gnss"], device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for c0 in range(0, n_win, chunk):
            c1 = min(c0 + chunk, n_win)
            b = c1 - c0
            P = P0 if b == chunk else initial_covariance(cfg, b, torch.float32, device)
            run_windows(
                g_all[c0:c1], a_all[c0:c1], z_all[c0:c1],
                p0[c0:c1], v0[c0:c1], q0[c0:c1], ba0[c0:c1], bg0[c0:c1], P, cfg, smooth=True,
            )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    T = int(cfg["bench"].get("window_T") or cfg["hpc"]["window_T"])
    chunk = int(cfg["bench"].get("chunk") or cfg["hpc"]["batch_chunk"])
    repeats = int(cfg["bench"]["n_repeat"])
    sizes = args.sizes or list(cfg["bench"]["sizes"])
    if args.quick:
        sizes = [s for s in sizes if s <= 100000]

    route = load_route(6)
    device = args.device
    print(f"device={device}  T={T}  chunk={chunk}  repeats={repeats}")

    results = []
    for rows in sizes:
        n_win = max(1, int(np.ceil(rows / T)))
        batch = make_batch(route, n_win, T)
        states = _initial_states(route, n_win, T, torch.float32, device)
        max_win = None if (args.quick or n_win <= CPU_WINDOWS_FULL) else CPU_WINDOWS_FULL
        cpu_rep = repeats if (max_win is None) else 1
        t_cpu, extrap = time_cpu(batch, states, cfg, n_win, T, repeats=cpu_rep, max_windows=max_win)
        t_gpu = time_gpu(batch, states, cfg, device, chunk, repeats=repeats)
        rec = {
            "rows": int(n_win * T),
            "n_windows": n_win,
            "cpu_s": t_cpu,
            "cpu_extrapolated": bool(extrap),
            "gpu_s": t_gpu,
            "speedup": t_cpu / t_gpu if t_gpu > 0 else float("nan"),
            "cpu_rows_per_s": n_win * T / t_cpu if t_cpu > 0 else float("nan"),
            "gpu_rows_per_s": n_win * T / t_gpu if t_gpu > 0 else float("nan"),
        }
        results.append(rec)
        print(f"rows={rec['rows']:>8d}  n_win={n_win:>4d}  CPU={t_cpu:8.2f}s  "
              f"GPU={t_gpu:6.3f}s  speedup={rec['speedup']:7.1f}x  "
              f"{'extrapolated ' if extrap else ''}GPU {rec['gpu_rows_per_s']/1e3:8.1f}k rows/s")

    # ---- batch-size scan -------------------------------------------------
    # the scan runs at the largest size so that the top of the batch sweep is
    # actually reachable: the GPU step costs the same at every batch size, so
    # the curve only turns over once the batch covers every window.
    scan_rows = max(sizes[0], sizes[-1])
    n_win = max(1, int(np.ceil(scan_rows / T)))
    batch = make_batch(route, n_win, T, seed=1)
    states = _initial_states(route, n_win, T, torch.float32, device)
    scan = []
    for b in cfg["bench"]["batch_scan"]:
        b = int(min(b, n_win))
        # one repeat per point: the GPU timing varies by <2 % between repeats,
        # while a small batch at this size already costs minutes per repeat
        t = time_gpu(batch, states, cfg, device, b, repeats=1)
        scan.append({"batch": b, "gpu_s": t, "rows_per_s": n_win * T / t,
                     "cpu_s": results[-1]["cpu_s"]})
        print(f"batch={b:>4d}  GPU={t:7.3f}s  {n_win * T / t / 1e3:9.1f}k rows/s  "
              f"speedup={results[-1]['cpu_s'] / t:7.1f}x", flush=True)

    payload = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        "window_T": T,
        "warm_W": 0,          # independent windows: nothing to burn in
        "chunk": chunk,
        "repeats": repeats,
        "sizes": results,
        "batch_scan": scan,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "bench.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n-> {RESULTS_DIR / 'bench.json'}")


if __name__ == "__main__":
    main()
