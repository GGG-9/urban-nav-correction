"""Feasibility check: can a model learn the *sensor* error instead of the residual?

The outage residual is a double integral, and its dominant part is the random
state error at the outage instant -- nothing to learn.  The sensor error is a
different object: it is an algebraic function of the reading itself
(``df_i = (k_i - 1) f_i`` for a scale factor, ``df = k2 a|a|`` for the quadratic
term), so a model that sees the IMU can in principle remove it, and *then* the
filter integrates a cleaner signal.

This script measures both layers, leave-one-route-out:

    sensor      rms of  accel_error - predicted      [m/s^2]
    trajectory  B3 outage error with the corrected IMU [m]

    python scripts/probe_imu_error.py            # as-shipped config
    python scripts/probe_imu_error.py --profile learnable

``learnable`` raises the deterministic part (quadratic scale factor), lowers the
random part (bias random walk), and is the profile the learned model would be
trained on if the sensor-level result justifies it.
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common import G_N, load_config                     # noqa: E402
from src.data.mini_imu_sim import simulate_route            # noqa: E402
from src.data.routes import route_specs                     # noqa: E402
from src.eval.metrics import summarize                      # noqa: E402
from src.nav.ekf import initial_covariance, init_from_route, run_filter  # noqa: E402
from src.nav.quaternion import quat_to_dcm                  # noqa: E402

_G = np.asarray(G_N, dtype=float)


def learnable(cfg: dict) -> dict:
    """Push the deterministic error up, the random error down."""
    cfg = copy.deepcopy(cfg)
    imu = cfg["sim"]["imu"]
    imu.update({
        "accel_nl_scale": 0.015,          # quadratic scale: nonlinear but exact
        "accel_bias_init": 0.01, "accel_bias_rw": 3e-5,
        "gyro_bias_init_deg": 10.0, "gyro_bias_rw_deg": 0.5,
        "accel_scale_shared": [1.03, 0.97, 1.015],
        "scale_route_jitter": 0.001,
        "temp_bias_accel": [0.02, -0.015, 0.01],
    })
    return cfg


def route_series(cfg: dict, spec: dict, fs: float):
    """Return the route, the model features, and the true accelerometer error."""
    r = simulate_route(spec, cfg)
    dt = 1.0 / fs
    a = np.empty_like(r.v)
    a[:-1] = np.diff(r.v, axis=0) / dt
    a[-1] = a[-2]
    C = quat_to_dcm(r.q)
    f_b = np.einsum("nji,nj->ni", C, a - _G)      # the ideal specific force
    df = r.accel - f_b                            # what a perfect model would remove
    s = np.abs(r.accel)
    x = np.c_[r.accel, r.accel * s, s.sum(1, keepdims=True), r.gyro]
    return r, x, df, f_b


def loo_ridge(X, Y, G, lam=1e-3):
    pred = np.zeros_like(Y)
    for g in np.unique(G):
        tr = G != g
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X[tr] - mu) / sd
        A = Z.T @ Z
        A += lam * np.trace(A) / len(A) * np.eye(len(A))
        W = np.linalg.solve(A, Z.T @ Y[tr])
        pred[~tr] = ((X[~tr] - mu) / sd) @ W
    return pred


def run(cfg, specs, routes, tag, oracle_scale=False):
    fs = float(cfg["fs_imu"])
    data = {}
    for rid in routes:
        data[rid] = route_series(cfg, specs[rid], fs)
    X = np.concatenate([data[r][1] for r in routes])
    Y = np.concatenate([data[r][2] for r in routes])
    G = np.concatenate([np.full(len(data[r][1]), r) for r in routes])
    if oracle_scale:
        # the part a model *could* learn from other routes: the shared scale
        # factor.  Quoted as an upper bound, not as a result.
        k = np.asarray(cfg["sim"]["imu"]["accel_scale_shared"], float) - 1.0
        pred = np.concatenate([data[r][3] * k for r in routes])
    else:
        pred = loo_ridge(X, Y, G)
    print(f"[{tag}{' (oracle scale)' if oracle_scale else ''}] sensor layer")
    e0 = Y
    e1 = Y - pred
    rms = lambda z: float(np.sqrt((z ** 2).mean()))
    print(f"  accelerometer error   {rms(e0):.4f} -> {rms(e1):.4f} m/s^2"
          f"   ({100 * (1 - rms(e1) / rms(e0)):+.1f} %)")

    print(f"[{tag}] trajectory layer (B3, leave-one-route-out)")
    base, ours = [], []
    k = 0
    for r in routes:
        n = len(data[r][1])
        rt = data[r][0]
        routed = {"gyro": rt.gyro, "accel": rt.accel, "gnss": rt.gnss, "p": rt.p, "q": rt.q}
        p0, v0, q0, ba0, bg0 = init_from_route(routed, cfg)
        P0 = initial_covariance(cfg)
        acc_fix = rt.accel - pred[k:k + n]
        k += n
        out = rt.outage.astype(bool)
        e0_ = summarize(np.linalg.norm(
            run_filter(rt.gyro, rt.accel, rt.gnss, p0, v0, q0, ba0, bg0, P0, cfg)["p_smooth"] - rt.p, axis=1),
            out, fs)["outage_mean"]
        e1_ = summarize(np.linalg.norm(
            run_filter(rt.gyro, acc_fix, rt.gnss, p0, v0, q0, ba0, bg0, P0, cfg)["p_smooth"] - rt.p, axis=1),
            out, fs)["outage_mean"]
        base.append(e0_)
        ours.append(e1_)
        print(f"  route {r:2d}: {e0_:7.2f} -> {e1_:7.2f} m")
    print(f"  mean        {np.mean(base):7.2f} -> {np.mean(ours):7.2f} m"
          f"   ({100 * (1 - np.mean(ours) / np.mean(base)):+.1f} %)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", type=int, nargs="*", default=[6, 7, 42, 43, 44, 45])
    ap.add_argument("--profile", choices=["default", "learnable"], default="default")
    ap.add_argument("--oracle-scale", action="store_true",
                    help="subtract the true scale error instead of a learned "
                         "prediction: the ceiling of what path (1) could buy")
    args = ap.parse_args()
    cfg = load_config()
    if args.profile == "learnable":
        cfg = learnable(cfg)
    specs = {s["id"]: s for s in route_specs(int(cfg["routes"].get("n_aug", 0)))}
    run(cfg, specs, args.routes, args.profile, args.oracle_scale)


if __name__ == "__main__":
    main()
