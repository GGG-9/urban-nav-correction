"""Is the GNSS multipath observable from the innovation sequence?

The corrector for plan B predicts the *fix error* (multipath + white noise)
from the observation stream instead of the outage residual.  Unlike the outage
residual, this quantity is produced by the measurement process itself, so it
may be recoverable: the innovations are the independent information increments
about the AR(1) multipath state.

    python scripts/probe_multipath.py [--routes 6 7 42 43 44 45] [--lags 64]

Reports, per route, the true fix-error rms and its autocorrelation time; then
leave-one-route-out predictive scores for
  * a causal FIR on the last `lags` innovations (3 axes x lags parameters),
  * the best exponential moving average (one pole, fitted on the training
    routes) -- the simplest estimator a deployed system could run.
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


def route_series(cfg, rid, specs):
    r = simulate_route(specs[rid], cfg)
    routed = {"gyro": r.gyro, "accel": r.accel, "gnss": r.gnss, "p": r.p, "q": r.q}
    p0, v0, q0, ba0, bg0 = init_from_route(routed, cfg)
    res = run_filter(r.gyro, r.accel, r.gnss, p0, v0, q0, ba0, bg0,
                     initial_covariance(cfg), cfg, smooth=False)
    fix = np.isfinite(r.gnss[:, 0])
    idx = np.flatnonzero(fix)
    y = r.gnss[idx] - r.p[idx]                 # true fix error (label)
    inn = res["innovation"][idx]               # what the filter saw
    return y, inn


def lag_matrix(x: np.ndarray, L: int) -> np.ndarray:
    """Rows = [x[k], x[k-1], ..., x[k-L+1]] (zero before the start)."""
    T = len(x)
    out = np.zeros((T, L, x.shape[1]))
    for j in range(L):
        out[j:, j] = x[:T - j] if j else x
    return out.reshape(T, L * x.shape[1])


def rms(y):
    return float(np.sqrt((y ** 2).sum(1).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", type=int, nargs="*", default=[6, 7, 42, 43, 44, 45])
    ap.add_argument("--lags", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config()
    specs = {s["id"]: s for s in route_specs(int(cfg["routes"].get("n_aug", 0)))}
    data = {}
    for rid in args.routes:
        y, inn = route_series(cfg, rid, specs)
        data[rid] = (y, inn)
        ac = [np.corrcoef(y[:-l, 0], y[l:, 0])[0, 1] for l in (10, 30, 60, 120)]
        print(f"route {rid:2d}: fix-error rms {rms(y):5.2f} m   "
              f"acf h {np.round(ac, 2)}   innov rms {rms(inn):5.2f} m")

    # causal FIR, leave-one-route-out
    print(f"\nLOO causal FIR, last {args.lags} innovations:")
    errs, raw = [], []
    for held in args.routes:
        Xs, ys = [], []
        for rid in args.routes:
            if rid == held:
                continue
            y, inn = data[rid]
            Xs.append(lag_matrix(inn, args.lags))
            ys.append(y)
        X = np.concatenate(Xs)
        Y = np.concatenate(ys)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        W, *_ = np.linalg.lstsq(Z.T @ Z + 10.0 * np.eye(Z.shape[1]), Z.T @ Y, rcond=None)
        y, inn = data[held]
        pred = ((lag_matrix(inn, args.lags) - mu) / sd) @ W
        errs.append(rms(y - pred))
        raw.append(rms(y))
        print(f"  held {held:2d}: fix error {rms(y):5.2f} -> {rms(y - pred):5.2f} m")
    print(f"  mean {np.mean(raw):5.2f} -> {np.mean(errs):5.2f} m"
          f"  ({100 * (1 - np.mean(errs) / np.mean(raw)):+.1f} %)")

    # single-pole EMA, one (beta, gain) per axis fitted on the training routes
    print("\nLOO exponential moving average on the innovations:")
    raw, errs = [], []
    for held in args.routes:
        best = None
        for beta in np.linspace(0.90, 0.999, 40):
            num = np.zeros(3)
            den = np.zeros(3)
            for rid in args.routes:
                if rid == held:
                    continue
                y, inn = data[rid]
                e = np.zeros_like(inn)
                acc = np.zeros(3)
                for k in range(len(inn)):
                    acc = beta * acc + (1 - beta) * inn[k]
                    e[k] = acc
                num += (e * y).sum(0)
                den += (e * e).sum(0)
            g = num / np.maximum(den, 1e-12)
            sse = 0.0
            for rid in args.routes:
                if rid == held:
                    continue
                y, inn = data[rid]
                acc = np.zeros(3)
                e = np.empty_like(inn)
                for k in range(len(inn)):
                    acc = beta * acc + (1 - beta) * inn[k]
                    e[k] = acc
                sse += ((y - g * e) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, beta, g)
        _, beta, g = best
        y, inn = data[held]
        acc = np.zeros(3)
        e = np.empty_like(inn)
        for k in range(len(inn)):
            acc = beta * acc + (1 - beta) * inn[k]
            e[k] = acc
        pred = g * e
        raw.append(rms(y))
        errs.append(rms(y - pred))
        print(f"  held {held:2d}: beta={beta:.3f} g={np.round(g, 2)}  "
              f"fix error {rms(y):5.2f} -> {rms(y - pred):5.2f} m")
    print(f"  mean {np.mean(raw):5.2f} -> {np.mean(errs):5.2f} m"
          f"  ({100 * (1 - np.mean(errs) / np.mean(raw)):+.1f} %)")


if __name__ == "__main__":
    main()
