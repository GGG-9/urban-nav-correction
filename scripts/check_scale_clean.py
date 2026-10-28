"""Does a scale-factor-only simulator produce a transferable outage residual?

Exploratory companion to ``check_scale.py``.  It rewrites the IMU profile in
memory to leave *one* model error standing -- a unit-to-unit accelerometer
scale miscalibration -- and then asks the question the learned corrector needs
answered: can the outage residual be explained by *one set of coefficients*
shared across routes?

The residual model is the physically complete one for a dead-reckoning outage
that starts with a state error:

    resid(t) = e_p(a) + e_v(a) * (t - a) + K * (p(t) - p(a) - v(a) * (t - a))

a constant offset, a velocity-error ramp, and the scale term.  The first two are
the unobservable state error at the outage instant and no corrector can know
them; only the last is a function of the data.  So the test projects [1, t] out
of both the residual and the displacement, fits K on the training routes, and
applies it to the held-out route.

    python scripts/check_scale_clean.py
"""
from __future__ import annotations

import copy
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.common import load_config                        # noqa: E402
from src.data.mini_imu_sim import simulate_route          # noqa: E402
from src.data.routes import route_specs                   # noqa: E402
from src.nav.ekf import initial_covariance, init_from_route, run_filter  # noqa: E402
from src.nav.quaternion import quat_to_dcm                # noqa: E402

ROUTES = [6, 7, 42, 43, 44, 45]
SCALE = [1.05, 0.95, 1.025]


def profile(cfg: dict) -> dict:
    """One model error only: the accelerometer scale factor."""
    cfg = copy.deepcopy(cfg)
    imu = cfg["sim"]["imu"]
    imu.update({
        "accel_bias_init": 0.0, "accel_bias_rw": 0.0,
        "gyro_bias_init_deg": 0.0, "gyro_bias_rw_deg": 0.0,
        "accel_noise": 0.005, "gyro_noise_deg": 0.01,
        "accel_scale_shared": SCALE,
        "gyro_scale_shared": [1.0, 1.0, 1.0],
        "scale_route_jitter": 0.0,
        "accel_nl_scale": 0.0,
        "temp_bias_accel": [0.0, 0.0, 0.0],
        "temp_bias_gyro_deg": [0.0, 0.0, 0.0],
    })
    cfg["filter"]["P0"]["att_deg"] = 0.02
    cfg["filter"]["P0"]["ba"] = 1e-8
    # A good receiver: the onset state error is set by the GNSS noise, and at
    # 3 m it dominates the outage residual (a 0.5 m/s velocity error becomes
    # 30 m of drift over 60 s), burying the model error under it.
    cfg["sim"]["gnss_sigma_h"] = 1.0
    cfg["sim"]["gnss_sigma_v"] = 1.5
    cfg["filter"]["R"]["gnss_h"] = 1.0
    cfg["filter"]["R"]["gnss_v"] = 2.25
    return cfg


def segments(cfg: dict, rid: int):
    specs = {s["id"]: s for s in route_specs(int(cfg["routes"].get("n_aug", 0)))}
    r = simulate_route(specs[rid], cfg)
    routed = {"gyro": r.gyro, "accel": r.accel, "gnss": r.gnss, "p": r.p, "q": r.q}
    p0, v0, q0, ba0, bg0 = init_from_route(routed, cfg)
    res = run_filter(r.gyro, r.accel, r.gnss, p0, v0, q0, ba0, bg0,
                     initial_covariance(cfg), cfg, smooth=False)
    base, v = res["p"], res["v"]
    gt, out = r.p, r.outage.astype(bool)
    st = np.flatnonzero(np.diff(np.r_[0, out.view(np.int8), 0]))
    dt = 1.0 / 100.0
    # specific force rotated into the navigation frame by the *estimated*
    # attitude, integrated twice: the regressor a corrector would build if it
    # knew the mechanisation.  A scale/misalignment error makes the residual a
    # fixed 3x3 map of this vector, so unlike the position displacement it does
    # not get scrambled by heading changes.
    C = quat_to_dcm(res["q"])
    f_n = np.einsum("nij,nj->ni", C, r.accel)
    out_seg = []
    for a, b in zip(st[::2], st[1::2]):
        t = (np.arange(a, b) - a) * dt
        disp = (base[a:b] - base[a]) - np.outer(t, v[a])
        u = np.cumsum(np.cumsum(f_n[a:b], 0) * dt, 0) * dt
        u -= u[0]
        # The body-frame basis: a scale/misalignment error M acts per sample as
        # C(t) M f_b(t), so the *linear* model of the residual is
        #   resid(t) = sum_jk M_jk * z_jk(t),  z_jk = double-integral of C[:,j] f_k
        # Nine 3-vector features.  Unlike a single 3x3 map on the displacement,
        # this basis is invariant to the route heading.
        z = np.zeros((b - a, 9, 3))
        for j in range(3):
            for k in range(3):
                w = C[a:b, :, j] * r.accel[a:b, k:k + 1]
                zz = np.cumsum(np.cumsum(w, 0) * dt, 0) * dt
                z[:, 3 * j + k, :] = zz - zz[0]
        resid = (gt[a:b] - base[a:b]) - (gt[a] - base[a])
        out_seg.append((t, disp, u, resid, z))
    return out_seg


def perp(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Remove the [1, t] subspace from x (column-wise)."""
    A = np.c_[np.ones_like(t), t]
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def main() -> None:
    cfg = profile(load_config())
    seg = {rid: segments(cfg, rid) for rid in ROUTES}
    all_seg = [s for rid in ROUTES for s in seg[rid]]
    stat = lambda e: float(np.sqrt((e ** 2).sum(1).mean()))   # noqa: E731

    raw = float(np.mean([stat(s[3]) for s in all_seg]))
    print(f"residual rms, all segments        {raw:6.1f} m")

    lin = []
    for s in all_seg:
        t = s[0]
        L = np.c_[np.ones_like(t), t]
        c, *_ = np.linalg.lstsq(L, s[3], rcond=None)
        lin.append(stat(s[3] - L @ c))
    print(f"per-segment oracle [1,t]          {np.mean(lin):6.1f} m"
          f"  ({100 * (1 - np.mean(lin) / raw):+.1f} %)   <- onset state error")
    for j, name in ((1, "disp"), (2, "u = C(q) f dt^2")):
        fit = []
        for t, _d, _u, resid, _z in all_seg:
            x = (_d, _u)[j - 1]
            A = np.c_[np.ones_like(t), t, x]
            c, *_ = np.linalg.lstsq(A, resid, rcond=None)
            fit.append(stat(resid - A @ c))
        print(f"per-segment oracle [1,t,{name:16s}] {np.mean(fit):6.1f} m"
              f"  ({100 * (1 - np.mean(fit) / raw):+.1f} %)")

    for j, name in ((1, "displacement"), (2, "integrated specific force")):
        print(f"\nleave-one-route-out, shared 3x3 map on {name}:")
        errs = []
        for held in ROUTES:
            A_list, Y_list = [], []
            for rid in ROUTES:
                if rid == held:
                    continue
                for s in seg[rid]:
                    A_list.append(perp(s[0], s[j]))
                    Y_list.append(perp(s[0], s[3]))
            M, *_ = np.linalg.lstsq(np.concatenate(A_list),
                                    np.concatenate(Y_list), rcond=None)
            e = [stat(s[3] - s[j] @ M) for s in seg[held]]
            errs += e
            print(f"  held {held:2d}: {np.mean([stat(s[3]) for s in seg[held]]):5.1f} -> "
                  f"{np.mean(e):5.1f} m")
        print(f"held-out rms  {raw:6.1f} -> {np.mean(errs):6.1f} m"
              f"  ({100 * (1 - np.mean(errs) / raw):+.1f} %)")

    # shared 27-parameter model in the body-frame basis
    for lam in (0.0, 1e-4, 1e-2):
        print(f"\nleave-one-route-out, shared body-frame basis, ridge {lam:g}:")
        errs = []
        for held in ROUTES:
            W = np.zeros((9, 3))
            for i in range(3):
                A_list, y_list = [], []
                for rid in ROUTES:
                    if rid == held:
                        continue
                    for s in seg[rid]:
                        A_list.append(perp(s[0], s[4][:, :, i]))
                        y_list.append(perp(s[0], s[3][:, i]))
                A = np.concatenate(A_list)
                y = np.concatenate(y_list)
                G = A.T @ A
                W[:, i] = np.linalg.solve(G + lam * np.trace(G) / 9 * np.eye(9),
                                          A.T @ y)
            e = []
            for s in seg[held]:
                pred = np.stack([s[4][:, :, i] @ W[:, i] for i in range(3)], 1)
                e.append(stat(s[3] - pred))
            errs += e
            print(f"  held {held:2d}: {np.mean([stat(s[3]) for s in seg[held]]):5.1f} -> "
                  f"{np.mean(e):5.1f} m")
        print(f"held-out rms  {raw:6.1f} -> {np.mean(errs):6.1f} m"
              f"  ({100 * (1 - np.mean(errs) / raw):+.1f} %)")
    print(f"\ntarget scale error        [{SCALE[0]-1:+.3f} {SCALE[1]-1:+.3f} {SCALE[2]-1:+.3f}]"
          "   (body frame)")


if __name__ == "__main__":
    main()
