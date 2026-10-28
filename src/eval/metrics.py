"""Evaluation metrics with fixed, documented definitions.

Definitions (all methods use the same code path):
  * trajectory RMSE      : root mean squared 3-D position error over a span
  * outage max / mean    : max / mean 3-D error inside GNSS-outage windows
  * recovery time        : after an outage ends, the time until the 3-D error
                           stays <= 3 m for at least 2 s
"""
from __future__ import annotations

import numpy as np


def traj_error(p_est: np.ndarray, p_gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(p_est) - np.asarray(p_gt), axis=1)


def rmse(err: np.ndarray) -> float:
    err = np.asarray(err, dtype=float)
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err**2))) if err.size else float("nan")


def outage_stats(err: np.ndarray, outage: np.ndarray) -> dict:
    e = np.asarray(err)[outage]
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"max": float("nan"), "mean": float("nan")}
    return {"max": float(e.max()), "mean": float(e.mean())}


def recovery_time(err: np.ndarray, outage: np.ndarray, fs: float,
                  threshold_m: float = 3.0, hold_s: float = 2.0) -> float:
    """Mean recovery time over all outage windows (NaN if never recovered).

    A window that is not recovered within 60 s after the outage ends counts as
    the full 60 s.
    """
    err = np.asarray(err)
    times = []
    n = len(err)
    hold = int(round(hold_s * fs))
    horizon = int(round(60.0 * fs))
    edges = np.flatnonzero(np.diff(outage.astype(np.int8)) == 1) + 1
    ends = np.flatnonzero(np.diff(outage.astype(np.int8)) == -1) + 1
    if outage[0]:
        edges = np.concatenate([[0], edges])
    if outage[-1]:
        ends = np.concatenate([ends, [n]])
    for i1 in ends:
        span = err[i1:min(i1 + horizon, n)]
        ok = span <= threshold_m
        rec = np.nan
        run = 0
        for j, flag in enumerate(ok):
            run = run + 1 if flag else 0
            if run >= hold:
                rec = (j - hold + 1) / fs
                break
        times.append(rec if np.isfinite(rec) else 60.0)
    return float(np.mean(times)) if times else float("nan")


def summarize(err: np.ndarray, outage: np.ndarray, fs: float) -> dict:
    return {
        "rmse_all": rmse(err),
        "rmse_gnss": rmse(err[~outage]),
        "outage_max": outage_stats(err, outage)["max"],
        "outage_mean": outage_stats(err, outage)["mean"],
        "recovery_s": recovery_time(err, outage, fs),
    }
