"""15-state error-state Kalman filter (ESKF) and RTS smoother, numpy reference.

Nominal state propagates through the strapdown mechanization:
    p (ENU), v (ENU), q (C_b^n), b_a, b_g
Error state estimated by the EKF:
    dx = [dp, dv, psi, db_a, db_g]   (15 x 1)
Attitude error ``psi`` is a small rotation in the navigation frame.

GNSS position fixes drive the measurement update; during an outage only the
time update runs (biases become unobservable and the covariance grows).
"""
from __future__ import annotations

import numpy as np

from ..common import G_N, load_config
from .quaternion import (
    quat_conj,
    quat_exp,
    quat_log,
    quat_mul,
    quat_normalize,
    quat_to_dcm,
    skew,
)

_I3 = np.eye(3)
_G = np.asarray(G_N, dtype=float)
_G15 = np.zeros((15, 3))
_G15[0:3, :] = _I3


def initial_covariance(cfg: dict | None = None) -> np.ndarray:
    cfg = cfg or load_config()
    p0 = cfg["filter"]["P0"]
    P = np.zeros((15, 15))
    P[0:3, 0:3] = np.eye(3) * p0["pos"]
    P[3:6, 3:6] = np.eye(3) * p0["vel"]
    P[6:9, 6:9] = np.eye(3) * np.deg2rad(p0["att_deg"]) ** 2
    P[9:12, 9:12] = np.eye(3) * p0["ba"]
    P[12:15, 12:15] = np.eye(3) * np.deg2rad(p0["bg_deg"]) ** 2
    return P


def process_noise(cfg: dict, dt: float) -> np.ndarray:
    q = cfg["filter"]["Qd"]
    Qd = np.zeros((15, 15))
    Qd[3:6, 3:6] = np.eye(3) * q["vrw"] ** 2 * dt
    Qd[6:9, 6:9] = np.eye(3) * q["arw"] ** 2 * dt
    Qd[9:12, 9:12] = np.eye(3) * q["ba_drive"] ** 2 * dt
    Qd[12:15, 12:15] = np.eye(3) * q["bg_drive"] ** 2 * dt
    return Qd


def measurement_noise(cfg: dict) -> np.ndarray:
    R = np.diag([cfg["filter"]["R"]["gnss_h"], cfg["filter"]["R"]["gnss_h"], cfg["filter"]["R"]["gnss_v"]])
    return R


def error_transition(C: np.ndarray, f_n: np.ndarray, dt: float) -> np.ndarray:
    """Phi = I + F dt for the 15-state error dynamics."""
    F = np.zeros((15, 15))
    F[0:3, 3:6] = _I3
    F[3:6, 6:9] = -skew(f_n)
    F[3:6, 9:12] = -C
    F[6:9, 12:15] = -C
    return np.eye(15) + F * dt


def propagate(p, v, q, ba, bg, gyro, accel, dt):
    """One strapdown mechanization step (identical recursion to the simulator).

    The specific force is resolved with the attitude at the *start* of the
    interval (standard first-order Euler mechanization); the attitude is then
    advanced by the gyro.  Using the updated attitude here would introduce a
    systematic O(omega*dt) error during turns.
    """
    C = quat_to_dcm(q)
    f_n = C @ (accel - ba)
    a_n = f_n + _G
    v_new = v + a_n * dt
    p_new = p + v * dt + 0.5 * a_n * dt * dt
    q_new = quat_normalize(quat_mul(q, quat_exp((gyro - bg) * dt)))
    return p_new, v_new, q_new, C, f_n, a_n


def _update(p, v, q, ba, bg, P, z, R):
    y = z - p
    S = P[0:3, 0:3] + R
    K = np.linalg.solve(S, P[0:3, :]).T          # 15 x 3
    dx = K @ y
    p = p + dx[0:3]
    v = v + dx[3:6]
    q = quat_normalize(quat_mul(quat_exp(dx[6:9]), q))
    ba = ba + dx[9:12]
    bg = bg + dx[12:15]
    IKH = np.eye(15) - K @ _G15.T
    P = IKH @ P @ IKH.T + K @ R @ K.T
    P = 0.5 * (P + P.T)
    return p, v, q, ba, bg, P, y


def run_filter(gyro, accel, gnss, p0, v0, q0, ba0, bg0, P0, cfg, smooth=True, store_predictions=False):
    """Run the ESKF over a whole trajectory (or a window).

    Parameters
    ----------
    gyro, accel : (n, 3) IMU samples
    gnss        : (n, 3) fixes, NaN where unavailable (outage / between fixes)
    Returns dict with filtered (and optionally RTS-smoothed) states.

    Both the posterior and the one-step *prior* states are stored so that the
    RTS smoother can be applied to the full state (in a closed-loop ESKF the
    error state is reset to zero at every update, so smoothing the error state
    itself would give an identically-zero correction).
    """
    n = len(gyro)
    dt = 1.0 / float(cfg["fs_imu"])
    Qd = process_noise(cfg, dt)
    R = measurement_noise(cfg)

    p, v, q, ba, bg = p0.copy(), v0.copy(), q0.copy(), ba0.copy(), bg0.copy()
    P = P0.copy()

    p_hist = np.empty((n, 3))
    v_hist = np.empty((n, 3))
    q_hist = np.empty((n, 4))
    ba_hist = np.empty((n, 3))
    bg_hist = np.empty((n, 3))
    p_prior = np.empty((n, 3))
    v_prior = np.empty((n, 3))
    q_prior = np.empty((n, 4))
    ba_prior = np.empty((n, 3))
    bg_prior = np.empty((n, 3))
    phi_hist = np.empty((n, 15, 15), dtype=np.float32)
    P_hist = np.empty((n, 15, 15), dtype=np.float32)
    innov = np.zeros((n, 3))
    has_fix = np.isfinite(gnss[:, 0])

    p_hist[0], v_hist[0], q_hist[0] = p, v, q
    ba_hist[0], bg_hist[0] = ba, bg
    p_prior[0], v_prior[0], q_prior[0] = p, v, q
    ba_prior[0], bg_prior[0] = ba, bg
    P_hist[0] = P
    phi_hist[0] = np.eye(15, dtype=np.float32)

    for k in range(n - 1):
        p_pr, v_pr, q_pr, C, f_n, _ = propagate(p, v, q, ba, bg, gyro[k], accel[k], dt)
        Phi = error_transition(C, f_n, dt)
        P = Phi @ P @ Phi.T + Qd
        p_prior[k + 1], v_prior[k + 1], q_prior[k + 1] = p_pr, v_pr, q_pr
        ba_prior[k + 1], bg_prior[k + 1] = ba, bg
        p, v, q = p_pr, v_pr, q_pr
        if has_fix[k + 1]:
            p, v, q, ba, bg, P, y = _update(p, v, q, ba, bg, P, gnss[k + 1], R)
            innov[k + 1] = y
        phi_hist[k] = Phi.astype(np.float32)
        P_hist[k + 1] = P.astype(np.float32)
        p_hist[k + 1], v_hist[k + 1], q_hist[k + 1] = p, v, q
        ba_hist[k + 1], bg_hist[k + 1] = ba, bg

    out = {"p": p_hist, "v": v_hist, "q": q_hist, "ba": ba_hist, "bg": bg_hist, "innovation": innov}
    if store_predictions:
        out["P"] = P_hist
    if smooth:
        out.update(
            rts_smooth(
                p_hist, v_hist, q_hist, ba_hist, bg_hist,
                p_prior, v_prior, q_prior, ba_prior, bg_prior,
                P_hist, phi_hist, Qd,
            )
        )
    return out


def rts_smooth(p_f, v_f, q_f, ba_f, bg_f, p_pr, v_pr, q_pr, ba_pr, bg_pr, P_kk, phi, Qd):
    """Full-state RTS smoother, run backwards with P_{k+1|k} rebuilt from Phi.

    x_s[k] = x_f[k] + C_k (x_s[k+1] - x_pred[k+1]),
    C_k    = P_f[k] Phi_k^T P_pred[k+1]^-1.
    """
    n = len(p_f)
    p_s, v_s = p_f.copy(), v_f.copy()
    q_s, ba_s, bg_s = q_f.copy(), ba_f.copy(), bg_f.copy()
    delta = np.zeros(15)
    for k in range(n - 2, -1, -1):
        Phi = phi[k].astype(np.float64)
        Pk = P_kk[k].astype(np.float64)
        P_pred_next = Phi @ Pk @ Phi.T + Qd
        B = Pk @ Phi.T
        C = np.linalg.solve(P_pred_next, B.T).T
        delta[0:3] = p_s[k + 1] - p_pr[k + 1]
        delta[3:6] = v_s[k + 1] - v_pr[k + 1]
        delta[6:9] = quat_log(quat_mul(q_s[k + 1], quat_conj(q_pr[k + 1])))
        delta[9:12] = ba_s[k + 1] - ba_pr[k + 1]
        delta[12:15] = bg_s[k + 1] - bg_pr[k + 1]
        corr = C @ delta
        p_s[k] = p_f[k] + corr[0:3]
        v_s[k] = v_f[k] + corr[3:6]
        q_s[k] = quat_normalize(quat_mul(quat_exp(corr[6:9]), q_f[k]))
        ba_s[k] = ba_f[k] + corr[9:12]
        bg_s[k] = bg_f[k] + corr[12:15]
    return {"p_smooth": p_s, "v_smooth": v_s, "q_smooth": q_s, "ba_smooth": ba_s, "bg_smooth": bg_s}


def run_dead_reckoning(gyro, accel, p0, v0, q0, fs: float):
    """B1: pure IMU strapdown dead reckoning (no filtering, no bias feedback)."""
    n = len(gyro)
    dt = 1.0 / fs
    zero = np.zeros(3)
    p_hist = np.empty((n, 3))
    v_hist = np.empty((n, 3))
    p, v, q = p0.copy(), v0.copy(), q0.copy()
    p_hist[0], v_hist[0] = p, v
    for k in range(n - 1):
        p, v, q, _, _, _ = propagate(p, v, q, zero, zero, gyro[k], accel[k], dt)
        p_hist[k + 1], v_hist[k + 1] = p, v
    return p_hist, v_hist


def init_from_route(route_npz: dict, cfg: dict):
    """Initial state: first GNSS fix for position, rest at rest, ideal alignment."""
    gnss = np.asarray(route_npz["gnss"])
    has_fix = np.isfinite(gnss[:, 0])
    first = int(np.argmax(has_fix)) if has_fix.any() else 0
    p0 = gnss[first].copy() if has_fix.any() else np.asarray(route_npz["p"])[0].copy()
    v0 = np.zeros(3)
    q0 = np.asarray(route_npz["q"])[0].copy()   # ideal initial alignment
    ba0 = np.zeros(3)
    bg0 = np.zeros(3)
    return p0, v0, q0, ba0, bg0
