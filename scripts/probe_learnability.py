"""Is the RTS residual learnable at all?

Before training a sequence model, check whether the residual the model is
supposed to predict can be predicted *from the observations it is given*.  The
probe is a leave-one-route-out linear model on window-level statistics; if that
cannot beat predicting zero, no network will.

    python scripts/probe_learnability.py                 # current config
    python scripts/probe_learnability.py --base b2       # residual of the causal
                                                         # forward filter instead
    python scripts/probe_learnability.py --nl 0.0025     # with a quadratic
                                                         # accelerometer error

Reports, per route: outage error of B3 and the residual rms; then the LOO
reduction (%) obtained by regressing the residual on features that the model
actually sees (integrated accelerometer / gyro, filter velocity, innovations).
"""
from __future__ import annotations

import argparse
import copy
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common import load_config                        # noqa: E402
from src.data.mini_imu_sim import simulate_route          # noqa: E402
from src.data.routes import route_specs                   # noqa: E402
from src.eval.metrics import traj_error, summarize        # noqa: E402
from src.nav.ekf import initial_covariance, init_from_route, run_filter  # noqa: E402
from src.nav.quaternion import quat_to_dcm                # noqa: E402


def features(acc, gyr, v, inn, q, fs=100.0):
    """Per-sample feature rows, normalised so that all of them are O(1).

    The double integrals are divided by the segment duration squared; without
    that the regression is dominated by the longest outage and the LOO score is
    meaningless.
    """
    dt = 1.0 / fs
    d = max(len(acc) * dt, 1e-6)
    C = quat_to_dcm(q)
    acc_n = np.einsum("nij,nj->ni", C, acc)
    a2 = np.cumsum(np.cumsum(acc_n, 0) * dt, 0) * dt / d**2
    n = acc * np.abs(acc)
    # the quadratic scale factor, in the navigation frame: a body-frame error
    # integrated over a turn is a rotating combination, so the un-rotated
    # version below cannot be matched by a fixed linear map
    n_nav = np.einsum("nij,nj->ni", C, n)
    n2_nav = np.cumsum(np.cumsum(n_nav, 0) * dt, 0) * dt / d**2
    n2 = np.cumsum(np.cumsum(n, 0) * dt, 0) * dt / d**2
    g2 = np.cumsum(np.cumsum(gyr, 0) * dt, 0) * dt / d**2
    # The *un*-normalised double integrals.  Dividing by d^2 (d = segment
    # length) makes the regression coefficient a per-segment k*d^2, which hides
    # the one error term the filter cannot model: a shared scale-factor
    # miscalibration gives delta-P = kappa * int(int(a)), i.e. a constant
    # coefficient on exactly this feature.  Both variants are kept so the two
    # hypotheses can be compared in one run.
    a2_raw = a2 * d * d
    g2_raw = g2 * d * d
    # innovation is zero-order held between fixes; compress its magnitude
    i_log = np.sign(inn) * np.log1p(np.abs(inn))
    return np.c_[a2, n2, n2_nav, g2, a2_raw, g2_raw, v, i_log]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", type=int, nargs="*", default=[6, 7, 42])
    ap.add_argument("--nl", type=float, default=None,
                    help="override sim.imu.accel_nl_scale [1/(m/s^2)]")
    ap.add_argument("--gscale", type=float, default=None,
                    help="override the shared gyro scale factor, e.g. 0.05 -> "
                         "[1.05, 0.95, 1.05]; additive errors are absorbed by the "
                         "bias state, a multiplicative one is not")
    ap.add_argument("--base", choices=["b3", "b2"], default="b3",
                    help="residual of which base trajectory: b3 = RTS smoother "
                         "(offline), b2 = forward filter (causal, the online case)")
    ap.add_argument("--sub", type=int, default=100, help="subsample every N samples")
    args = ap.parse_args()

    cfg = copy.deepcopy(load_config())
    if args.nl is not None:
        cfg["sim"]["imu"]["accel_nl_scale"] = args.nl
    if args.gscale is not None:
        g = float(args.gscale)
        cfg["sim"]["imu"]["gyro_scale_shared"] = [1.0 + g, 1.0 - g, 1.0 + g]
    nl = float(cfg["sim"]["imu"].get("accel_nl_scale", 0.0))
    gs = cfg["sim"]["imu"]["gyro_scale_shared"]
    fs = float(cfg["fs_imu"])
    print(f"accel_nl_scale = {nl}  gyro_scale = {gs}  base = {args.base}")

    specs = {s["id"]: s for s in route_specs(int(cfg["routes"].get("n_aug", 0)))}
    X, Y, G = [], [], []
    for rid in args.routes:
        r = simulate_route(specs[rid], cfg)
        routed = {"gyro": r.gyro, "accel": r.accel, "gnss": r.gnss, "p": r.p, "q": r.q}
        p0, v0, q0, ba0, bg0 = init_from_route(routed, cfg)
        res = run_filter(r.gyro, r.accel, r.gnss, p0, v0, q0, ba0, bg0,
                         initial_covariance(cfg), cfg, smooth=(args.base == "b3"))
        gt, out = r.p, r.outage.astype(bool)
        base = res["p_smooth"] if args.base == "b3" else res["p"]
        e3 = traj_error(base, gt)
        m = summarize(e3, out, fs)
        resid = gt - base
        print(f"route {rid:2d}: {args.base.upper()} outage mean {m['outage_mean']:6.2f} "
              f"max {m['outage_max']:7.2f} m"
              f" | residual rms {np.sqrt((resid[out] ** 2).sum(1).mean()):6.2f}")

        st = np.flatnonzero(np.diff(np.r_[0, out.view(np.int8), 0]))
        for a, b in zip(st[::2], st[1::2]):
            idx = np.arange(a, b, args.sub)
            X.append(features(r.accel[a:b], r.gyro[a:b], res["v"][a:b],
                              res["innovation"][a:b], res["q"][a:b], fs)[::args.sub])
            Y.append(resid[a:b][::args.sub])
            G.append(np.full(len(idx), rid))

    if not X:
        return
    X, Y, G = np.concatenate(X), np.concatenate(Y), np.concatenate(G)
    pred = np.zeros_like(Y)
    pred_in = np.zeros_like(Y)          # scale fitted inside the fold only
    for rid in np.unique(G):
        tr = G != rid
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X[tr] - mu) / sd
        W = np.linalg.solve(Z.T @ Z + 10.0 * np.eye(Z.shape[1]), Z.T @ Y[tr])
        p_tr = Z @ W
        pred[~tr] = ((X[~tr] - mu) / sd) @ W
        # the honest variant: the shrinkage a deployer could actually fit,
        # from the training folds alone (no held-out information)
        c_fold = float((Y[tr] * p_tr).sum() / max((p_tr * p_tr).sum(), 1e-12))
        pred_in[~tr] = c_fold * ((X[~tr] - mu) / sd) @ W
    rms = lambda z: float(np.sqrt((z ** 2).sum(1).mean()))
    # least-squares rescale of the prediction: separates "is there any signal"
    # (correlation) from "is the regression calibrated", which a raw LOO score
    # conflates on this few routes.  c is fitted on ALL routes including the
    # held-out one -- it is an oracle, quoted only to bound what scaling could buy.
    c = float((Y * pred).sum() / max((pred * pred).sum(), 1e-12))
    print(f"\nLOO probe: residual rms {rms(Y):.2f} m")
    print(f"  raw LOO        {rms(Y - pred):.2f} m  ({100 * (1 - rms(Y - pred) / rms(Y)):+.1f} %)")
    print(f"  in-fold scale  {rms(Y - pred_in):.2f} m  ({100 * (1 - rms(Y - pred_in) / rms(Y)):+.1f} %)"
          f"   <- the honest number")
    print(f"  oracle scale   {rms(Y - c * pred):.2f} m  ({100 * (1 - rms(Y - c * pred) / rms(Y)):+.1f} %)"
          f"   scale c = {c:.2f}")
    for k in range(3):
        c = np.corrcoef(Y[:, k], pred[:, k])[0, 1]
        print(f"  axis {k}: corr {c:+.3f}")


if __name__ == "__main__":
    main()
