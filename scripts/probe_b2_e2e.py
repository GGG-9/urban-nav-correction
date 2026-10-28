"""Does the causal-filter residual still look learnable after stitching?

The window-level LOO probe said +11 % against B2, but a probe scores the
*residual*; the report scores the *trajectory*.  This runs the exact pipeline
the neural corrector uses -- 102.4 s centered window -> 12.8 s residual block ->
overlap-add -> outage-gated shrinkage -- with a linear model in place of the
network, and evaluates it on the held-out routes with the same alpha guard.

A linear model getting a real gain here means the signal survives stitching and
the sequence model is the bottleneck.  A gain that vanishes means the probe was
scoring a route-level scaling that never reaches the trajectory.

    python scripts/probe_b2_e2e.py --base b2
    python scripts/probe_b2_e2e.py --base b3
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common import RUNS_DIR, load_config                       # noqa: E402
from src.eval.metrics import summarize, traj_error                 # noqa: E402
from src.features.build_dataset import (IN_LEN, OUT_LEN, RATE,     # noqa: E402
                                        cut_centered, downsample_mean,
                                        route_features)
from src.models.train import outage_gate, stitch                   # noqa: E402


def route_arrays(rid: int, cfg: dict, base: str):
    path = sorted(glob.glob(str(RUNS_DIR / "kf" / f"route_{rid:02d}_*.npz")))[0]
    with np.load(path, allow_pickle=True) as d:
        d = {k: d[k] for k in d.files}
    rate = float(RATE)
    x = route_features(d, rate)
    p = d["p_b2"] if base == "b2" else d["p_b3"]
    y = downsample_mean(d["p_gt"] - p, rate)
    return d, x, y


def windows(x, y, in_len, out_len, stride):
    xw, yw, cs = cut_centered(x, y, in_len, out_len, stride)
    # a block-constant predictor: the window's whole context collapsed to its
    # mean and spread, the block's residual collapsed to its mean.  Cheap, and
    # it is the strongest thing a linear map can do on this geometry.
    f = np.c_[xw.mean(1), xw.std(1)]
    t = yw.mean(1)
    return f, t, cs, xw, yw


def fit(f, t, lam=1.0):
    mu, sd = f.mean(0), f.std(0) + 1e-9
    Z = (f - mu) / sd
    A = Z.T @ Z
    A += lam * np.trace(A) / len(A) * np.eye(len(A))
    W = np.linalg.solve(A, Z.T @ t)
    return lambda X: ((X - mu) / sd) @ W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["b2", "b3"], default="b2")
    args = ap.parse_args()
    cfg = load_config()
    fs = float(cfg["fs_imu"])
    up = int(round(fs / RATE))

    tr = [int(r) for r in cfg["routes"]["train"]]
    va = [int(r) for r in cfg["routes"]["val"]]
    te = [int(r) for r in cfg["routes"]["test"]]

    F, T = [], []
    for rid in tr:
        _, x, y = route_arrays(rid, cfg, args.base)
        f, t, *_ = windows(x, y, IN_LEN, OUT_LEN, OUT_LEN // 2)
        F.append(f)
        T.append(t)
    model = fit(np.concatenate(F), np.concatenate(T))

    preds = {}
    for rid in va + te:
        d, x, y = route_arrays(rid, cfg, args.base)
        f, t, cs, xw, yw = windows(x, y, IN_LEN, OUT_LEN, OUT_LEN // 2)
        p = model(f)                                     # [N,3] block means
        p = np.repeat(p[:, None, :], OUT_LEN, axis=1).astype(np.float32)
        n_low = len(y)
        dp = stitch(n_low, p, IN_LEN, OUT_LEN, OUT_LEN // 2, "centered")
        dp = np.repeat(dp, up, axis=0)[: len(d["p_gt"])]
        base_p = d["p_b2"] if args.base == "b2" else d["p_b3"]
        preds[rid] = {"base": base_p, "dp": dp, "gt": d["p_gt"],
                      "outage": d["outage"].astype(bool)}

    # same one-standard-error guard as src/models/train.py:fit_alpha
    grid = np.linspace(0.0, 2.0, 41)
    scores = {}
    for a in grid:
        vals = []
        for r in (preds[i] for i in va):
            g = outage_gate(r["outage"], fs)
            vals.append(summarize(traj_error(r["base"] + a * g[:, None] * r["dp"], r["gt"]),
                                  r["outage"], fs)["outage_mean"])
        scores[float(a)] = np.asarray(vals, float)
    s0 = scores[0.0]
    se = float(np.std(s0, ddof=1) / np.sqrt(len(s0)))
    alpha = 0.0
    for a in grid:
        if scores[float(a)].mean() < s0.mean() - 1.0 * se:
            alpha = float(a)
            break

    print(f"[{args.base}] ridge, block-constant over a {IN_LEN / RATE:.1f} s window")
    print(f"  val ({len(va)} routes): alpha* = {alpha:.2f}   "
          f"(val outage_mean {scores[0.0].mean():.2f} -> "
          f"{np.nanmean([summarize(traj_error(r['base'] + alpha * outage_gate(r['outage'], fs)[:, None] * r['dp'], r['gt']), r['outage'], fs)['outage_mean'] for r in (preds[i] for i in va)]):.2f} m)")
    print(f"  test ({len(te)} routes):")
    b, o = [], []
    for rid in te:
        r = preds[rid]
        g = outage_gate(r["outage"], fs)
        e0 = summarize(traj_error(r["base"], r["gt"]), r["outage"], fs)["outage_mean"]
        e1 = summarize(traj_error(r["base"] + alpha * g[:, None] * r["dp"], r["gt"]),
                       r["outage"], fs)["outage_mean"]
        b.append(e0)
        o.append(e1)
        print(f"    route {rid:2d}: {e0:7.2f} -> {e1:7.2f} m")
    print(f"    mean        {np.mean(b):7.2f} -> {np.mean(o):7.2f} m"
          f"   ({100 * (1 - np.mean(o) / np.mean(b)):+.1f} %)")


if __name__ == "__main__":
    main()
