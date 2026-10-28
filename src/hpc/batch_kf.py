"""GPU-parallel window-batched ESKF + full-state RTS smoother (PyTorch).

Parallelism
-----------
The time axis is inherently recursive, so we split a long trajectory into
overlapping windows of ``T`` samples and run all windows of a batch at once:
tensors are ``[B, T, ...]`` and every time step is one vectorised kernel launch
over the batch.  The parallel axis is therefore *windows x vehicles*, not time
inside a window.

Each window carries ``W`` burn-in samples at both ends:
    [0, W)      forward warm-up   (filter transient decays)
    [W, T-W)    output region
    [T-W, T)    backward burn-in  (RTS needs future data)
Consecutive windows are placed with stride ``T - 2W - fade`` so the output
regions tile the route with a ``fade``-sample overlap (the two estimates are
combined there with tent weights).

Two-pass warm start for a single long route
-------------------------------------------
pass 1 : every window starts from a GNSS-derived alignment (position from the
         nearest fix, heading from the local GNSS track, zero biases) -- fully
         parallel, but the first samples of each window are transient.
pass 2 : every window is re-initialised from the pass-1 estimate at its own
         start index, so the transient shrinks to the burn-in length.
Only pass-2 output regions are kept.
"""
from __future__ import annotations

import numpy as np
import torch

from ..common import G_N, load_config

# The parity check compares against a float64 numpy reference; TF32 matmuls
# would cut the mantissa to 10 bits and show up as ~1e-3 m "parity" errors that
# are not bugs.  PyTorch already defaults to full fp32 matmul on CUDA, so this
# only guards against a global setting elsewhere in the process.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False

_I3 = np.eye(3, dtype=np.float64)
_G = np.asarray(G_N, dtype=np.float64)


# --------------------------------------------------------------------------
# torch quaternion helpers (same conventions as src/nav/quaternion.py)
# --------------------------------------------------------------------------
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def quat_exp(v: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(v, dim=-1, keepdim=True)
    small = angle < 1e-10
    half = 0.5 * angle
    safe = torch.where(small, torch.ones_like(angle), angle)
    w = torch.cos(half)
    xyz = torch.sin(half) / safe * v
    xyz = torch.where(small, 0.5 * v, xyz)
    return torch.cat([w, xyz], dim=-1)


def quat_log(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w = q[..., :1]
    v = q[..., 1:]
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(n, w)
    angle = torch.where(angle > np.pi, angle - 2.0 * np.pi, angle)
    small = n < 1e-10
    safe = torch.where(small, torch.ones_like(n), n)
    out = angle * (v / safe)
    return torch.where(small, 2.0 * v, out)


def quat_to_dcm(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    C = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    C[..., 0, 0] = ww + xx - yy - zz
    C[..., 0, 1] = 2.0 * (x * y - w * z)
    C[..., 0, 2] = 2.0 * (x * z + w * y)
    C[..., 1, 0] = 2.0 * (x * y + w * z)
    C[..., 1, 1] = ww - xx + yy - zz
    C[..., 1, 2] = 2.0 * (y * z - w * x)
    C[..., 2, 0] = 2.0 * (x * z - w * y)
    C[..., 2, 1] = 2.0 * (y * z + w * x)
    C[..., 2, 2] = ww - xx - yy + zz
    return C


def skew(v: torch.Tensor) -> torch.Tensor:
    S = torch.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype, device=v.device)
    S[..., 0, 1] = -v[..., 2]
    S[..., 0, 2] = v[..., 1]
    S[..., 1, 0] = v[..., 2]
    S[..., 1, 2] = -v[..., 0]
    S[..., 2, 0] = -v[..., 1]
    S[..., 2, 1] = v[..., 0]
    return S


# --------------------------------------------------------------------------
# filter matrices
# --------------------------------------------------------------------------
def initial_covariance(cfg: dict, B: int, dtype, device, att_sigma=None) -> torch.Tensor:
    """P0 for a batch; ``att_sigma`` (rad, per window) overrides the attitude std."""
    p0 = cfg["filter"]["P0"]
    diag = np.concatenate(
        [
            np.full(3, p0["pos"]),
            np.full(3, p0["vel"]),
            np.full(3, np.deg2rad(p0["att_deg"]) ** 2),
            np.full(3, p0["ba"]),
            np.full(3, np.deg2rad(p0["bg_deg"]) ** 2),
        ]
    )
    P = torch.diag(torch.as_tensor(diag, dtype=dtype, device=device))
    P = P.unsqueeze(0).repeat(B, 1, 1)
    if att_sigma is not None:
        s = torch.as_tensor(att_sigma, dtype=dtype, device=device).reshape(B, 1) ** 2
        P[:, 6:9, 6:9] = torch.diag_embed(s.repeat(1, 3))
    return P


def process_noise(cfg: dict, dt: float, dtype, device) -> torch.Tensor:
    q = cfg["filter"]["Qd"]
    vals = np.concatenate(
        [
            np.zeros(3),
            np.full(3, q["vrw"] ** 2 * dt),
            np.full(3, q["arw"] ** 2 * dt),
            np.full(3, q["ba_drive"] ** 2 * dt),
            np.full(3, q["bg_drive"] ** 2 * dt),
        ]
    )
    return torch.diag(torch.as_tensor(vals, dtype=dtype, device=device))


def measurement_noise(cfg: dict, dtype, device) -> torch.Tensor:
    R = cfg["filter"]["R"]
    return torch.diag(torch.as_tensor([R["gnss_h"], R["gnss_h"], R["gnss_v"]], dtype=dtype, device=device))


# --------------------------------------------------------------------------
# one batched window step
# --------------------------------------------------------------------------
def _propagate(p, v, q, ba, bg, gyro, accel, dt):
    C = quat_to_dcm(q)
    f_n = torch.bmm(C, (accel - ba).unsqueeze(-1)).squeeze(-1)
    a_n = f_n + torch.as_tensor(_G, dtype=q.dtype, device=q.device)
    v_new = v + a_n * dt
    p_new = p + v * dt + 0.5 * a_n * dt * dt
    q_new = quat_normalize(quat_mul(q, quat_exp((gyro - bg) * dt)))
    return p_new, v_new, q_new, C, f_n


def _transition(C, f_n, dt):
    B = C.shape[0]
    dev, dtp = C.device, C.dtype
    Phi = torch.zeros(B, 15, 15, dtype=dtp, device=dev)
    eye = torch.eye(3, dtype=dtp, device=dev)
    Phi[:, 0:3, 3:6] = eye * dt
    Phi[:, 3:6, 6:9] = -skew(f_n) * dt
    Phi[:, 3:6, 9:12] = -C * dt
    Phi[:, 6:9, 12:15] = -C * dt
    Phi[:, 0:15, 0:15] += torch.eye(15, dtype=dtp, device=dev)
    return Phi


def _update(p, v, q, ba, bg, P, z, R):
    y = z - p
    S = P[:, 0:3, 0:3] + R
    PHt = P[:, :, 0:3]                                     # [B,15,3]
    K = torch.linalg.solve(S, PHt.transpose(1, 2)).transpose(1, 2)   # [B,15,3]
    dx = torch.bmm(K, y.unsqueeze(-1)).squeeze(-1)
    p = p + dx[:, 0:3]
    v = v + dx[:, 3:6]
    q = quat_normalize(quat_mul(quat_exp(dx[:, 6:9]), q))
    ba = ba + dx[:, 9:12]
    bg = bg + dx[:, 12:15]
    B = P.shape[0]
    IKH = torch.eye(15, dtype=P.dtype, device=P.device).expand(B, 15, 15).clone()
    IKH[:, :, 0:3] -= K
    P = IKH @ P @ IKH.transpose(1, 2) + K @ R @ K.transpose(1, 2)
    P = 0.5 * (P + P.transpose(1, 2))
    return p, v, q, ba, bg, P, y


# --------------------------------------------------------------------------
# batched forward filter + RTS
# --------------------------------------------------------------------------
@torch.no_grad()
def run_windows(gyro, accel, gnss, p0, v0, q0, ba0, bg0, P0, cfg, smooth=True):
    """Run the ESKF on a batch of windows.

    Parameters
    ----------
    gyro, accel, gnss : [B, T, 3] tensors (gnss may contain NaN)
    p0, v0, ba0, bg0  : [B, 3]     q0 [B, 4]      P0 [B, 15, 15]
    """
    B, T, _ = gyro.shape
    dtype = gyro.dtype
    device = gyro.device
    dt = 1.0 / float(cfg["fs_imu"])
    Qd = process_noise(cfg, dt, dtype, device)
    R = measurement_noise(cfg, dtype, device)
    eye15 = torch.eye(15, dtype=dtype, device=device)

    has_fix = torch.isfinite(gnss).all(dim=-1)             # [B,T]
    gnss = torch.nan_to_num(gnss, nan=0.0)

    p, v, q, ba, bg = p0.clone(), v0.clone(), q0.clone(), ba0.clone(), bg0.clone()
    P = P0.clone()

    p_hist = torch.empty(B, T, 3, dtype=dtype, device=device)
    v_hist = torch.empty(B, T, 3, dtype=dtype, device=device)
    q_hist = torch.empty(B, T, 4, dtype=dtype, device=device)
    ba_hist = torch.empty(B, T, 3, dtype=dtype, device=device)
    bg_hist = torch.empty(B, T, 3, dtype=dtype, device=device)
    p_pr = torch.empty(B, T, 3, dtype=dtype, device=device)
    v_pr = torch.empty(B, T, 3, dtype=dtype, device=device)
    q_pr = torch.empty(B, T, 4, dtype=dtype, device=device)
    ba_pr = torch.empty(B, T, 3, dtype=dtype, device=device)
    bg_pr = torch.empty(B, T, 3, dtype=dtype, device=device)
    innov = torch.zeros(B, T, 3, dtype=dtype, device=device)
    # Only the posterior covariance is stored: Phi_k is a function of the
    # attitude and the specific force, both of which are already in q_hist and
    # ba_hist, so keeping a [B,T,15,15] copy would double the dominant footprint
    # for nothing.  _rts rebuilds it step by step.
    P_hist = torch.zeros(B, T, 15, 15, dtype=dtype, device=device)

    p_hist[:, 0], v_hist[:, 0], q_hist[:, 0] = p, v, q
    ba_hist[:, 0], bg_hist[:, 0] = ba, bg
    p_pr[:, 0], v_pr[:, 0], q_pr[:, 0] = p, v, q
    ba_pr[:, 0], bg_pr[:, 0] = ba, bg
    P_hist[:, 0] = P

    for k in range(T - 1):
        p_new, v_new, q_new, C, f_n = _propagate(p, v, q, ba, bg, gyro[:, k], accel[:, k], dt)
        Phi = _transition(C, f_n, dt)
        P = Phi @ P @ Phi.transpose(1, 2) + Qd
        p_pr[:, k + 1], v_pr[:, k + 1], q_pr[:, k + 1] = p_new, v_new, q_new
        ba_pr[:, k + 1], bg_pr[:, k + 1] = ba, bg
        p, v, q = p_new, v_new, q_new
        fix = has_fix[:, k + 1]
        if bool(fix.any()):
            p_u, v_u, q_u, ba_u, bg_u, P_u, y = _update(p, v, q, ba, bg, P, gnss[:, k + 1], R)
            m = fix.unsqueeze(-1)
            p = torch.where(m, p_u, p)
            v = torch.where(m, v_u, v)
            q = torch.where(m, q_u, q)
            ba = torch.where(m, ba_u, ba)
            bg = torch.where(m, bg_u, bg)
            mm = fix.view(-1, 1, 1)
            P = torch.where(mm, P_u, P)
            innov[:, k + 1] = torch.where(m, y, torch.zeros_like(y))
        P_hist[:, k + 1] = P
        p_hist[:, k + 1], v_hist[:, k + 1], q_hist[:, k + 1] = p, v, q
        ba_hist[:, k + 1], bg_hist[:, k + 1] = ba, bg

    out = {"p": p_hist, "v": v_hist, "q": q_hist, "ba": ba_hist, "bg": bg_hist, "innovation": innov}
    if smooth:
        out.update(
            _rts(p_hist, v_hist, q_hist, ba_hist, bg_hist,
                 p_pr, v_pr, q_pr, ba_pr, bg_pr, P_hist, accel, Qd, dt, dtype)
        )
    return out


@torch.no_grad()
def _rts(p_f, v_f, q_f, ba_f, bg_f, p_pr, v_pr, q_pr, ba_pr, bg_pr, P_hist, accel, Qd, dt, dtype):
    B, T = p_f.shape[0], p_f.shape[1]
    p_s, v_s, q_s = p_f.clone(), v_f.clone(), q_f.clone()
    ba_s, bg_s = ba_f.clone(), bg_f.clone()
    for k in range(T - 2, -1, -1):
        # rebuild Phi_k exactly as the forward pass did (same C, same f_n)
        C = quat_to_dcm(q_f[:, k])
        f_n = torch.bmm(C, (accel[:, k] - ba_f[:, k]).unsqueeze(-1)).squeeze(-1)
        Phi = _transition(C, f_n, dt)
        Pk = P_hist[:, k]
        P_pred = Phi @ Pk @ Phi.transpose(1, 2) + Qd
        Bm = Pk @ Phi.transpose(1, 2)
        C = torch.linalg.solve(P_pred, Bm.transpose(1, 2)).transpose(1, 2)
        delta = torch.cat(
            [
                p_s[:, k + 1] - p_pr[:, k + 1],
                v_s[:, k + 1] - v_pr[:, k + 1],
                quat_log(quat_mul(q_s[:, k + 1], quat_conj(q_pr[:, k + 1]))),
                ba_s[:, k + 1] - ba_pr[:, k + 1],
                bg_s[:, k + 1] - bg_pr[:, k + 1],
            ],
            dim=-1,
        )
        corr = torch.bmm(C, delta.unsqueeze(-1)).squeeze(-1)
        p_s[:, k] = p_f[:, k] + corr[:, 0:3]
        v_s[:, k] = v_f[:, k] + corr[:, 3:6]
        q_s[:, k] = quat_normalize(quat_mul(quat_exp(corr[:, 6:9]), q_f[:, k]))
        ba_s[:, k] = ba_f[:, k] + corr[:, 9:12]
        bg_s[:, k] = bg_f[:, k] + corr[:, 12:15]
    return {"p_smooth": p_s, "v_smooth": v_s, "q_smooth": q_s,
            "ba_smooth": ba_s, "bg_smooth": bg_s}


# --------------------------------------------------------------------------
# window bookkeeping for one long route
# --------------------------------------------------------------------------
def window_starts(n: int, T: int, W: int, fade: int = 0) -> list[int]:
    stride = max(1, T - 2 * W - fade)
    starts, s = [], 0
    while s + T < n:
        starts.append(s)
        s += stride
    if not starts or starts[-1] + T < n:
        starts.append(max(n - T, 0))           # tail window, overlapped
    return starts


def gnss_alignment(gnss: np.ndarray, gyro: np.ndarray, fs: float, i0: int,
                   sigma_gnss: float = 3.0, search_s: float = 60.0,
                   base_s: float = 5.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Coarse alignment for a window starting at index ``i0``.

    A GNSS course-over-ground is only meaningful while the vehicle is moving,
    so we look for the *largest* displacement baseline within +-``search_s`` of
    the window start, take its COG, and propagate that heading back/forward to
    ``i0`` with the gyro.  Returns ``(p0, q0, yaw_sigma_rad)``.
    """
    from ..nav.quaternion import quat_from_euler

    n = len(gnss)
    valid = np.flatnonzero(np.isfinite(gnss[:, 0]))
    if valid.size == 0:
        return np.zeros(3), quat_from_euler(0.0, 0.0, 0.0), float(np.pi)

    after = valid[valid >= i0]
    k = int(after[0]) if after.size else int(valid[-1])
    p0 = gnss[k].copy()

    b = max(1, int(base_s * fs))
    lo = max(0, i0 - int(search_s * fs))
    hi = min(n - 1, i0 + int(search_s * fs))
    cands = valid[(valid >= lo) & (valid <= hi - b)]
    best, best_score, best_disp = None, -np.inf, 0.0
    for c in cands:
        c = int(c)
        dx = float(gnss[c + b, 0] - gnss[c, 0])
        dy = float(gnss[c + b, 1] - gnss[c, 1])
        disp = float(np.hypot(dx, dy))
        score = disp - 1e-4 * abs(c - i0)      # prefer baselines close to i0
        if score > best_score:
            best, best_score, best_disp = c, score, disp
            best_d = (dx, dy)
    if best is None or best_disp < 10.0:
        # nothing moved near this window: heading unobservable
        return p0, quat_from_euler(0.0, 0.0, 0.0), float(np.deg2rad(60.0))

    yaw = float(np.arctan2(best_d[1], best_d[0]))
    if best > i0:                              # baseline lies after i0
        yaw -= float(gyro[i0:best + 1, 2].sum()) / fs
    elif best + b < i0:                        # baseline lies before i0
        yaw += float(gyro[best + b:i0 + 1, 2].sum()) / fs
    sigma = max(np.deg2rad(3.0), np.arctan(sigma_gnss * 1.5 / best_disp))
    return p0, quat_from_euler(yaw, 0.0, 0.0), float(sigma)


@torch.no_grad()
def run_route_windowed(route: dict, cfg: dict, device: str = "cuda",
                       dtype=torch.float32, two_pass: bool = True,
                       fade_s: float = 1.0, verbose: bool = True) -> dict:
    """Window-parallel ESKF+RTS over one long route; returns numpy arrays."""
    T = int(cfg["hpc"]["window_T"])
    W = int(cfg["hpc"]["warm_W"])
    chunk = int(cfg["hpc"]["batch_chunk"])
    fs = float(cfg["fs_imu"])
    n = len(route["gyro"])
    fade = int(round(fade_s * fs))
    if n < T:
        T, W, fade = n, max(1, n // 8), 0

    starts = window_starts(n, T, W, fade)
    n_win = len(starts)
    gyro_np, accel_np, gnss_np = route["gyro"], route["accel"], route["gnss"]

    # padded window tensors [n_win, T, 3]
    g_in = np.zeros((n_win, T, 3), dtype=np.float32)
    a_in = np.zeros((n_win, T, 3), dtype=np.float32)
    z_in = np.full((n_win, T, 3), np.nan, dtype=np.float32)
    for i, s in enumerate(starts):
        e = min(n, s + T)
        g_in[i, : e - s] = gyro_np[s:e]
        a_in[i, : e - s] = accel_np[s:e]
        z_in[i, : e - s] = gnss_np[s:e]
        if e - s < T:                       # edge-replicate the tail
            g_in[i, e - s:] = gyro_np[e - 1]
            a_in[i, e - s:] = accel_np[e - 1]

    # -------- pass 1: GNSS + gyro coarse alignment for every window -------
    p0 = np.zeros((n_win, 3), np.float32)
    v0 = np.zeros((n_win, 3), np.float32)
    q0 = np.zeros((n_win, 4), np.float32)
    att_sig = np.zeros(n_win, np.float32)
    for i, s in enumerate(starts):
        pi, qi, si = gnss_alignment(gnss_np, gyro_np, fs, s,
                                    sigma_gnss=float(cfg["sim"]["gnss_sigma_h"]))
        p0[i], q0[i], att_sig[i] = pi, qi, si
    ba0 = np.zeros((n_win, 3), np.float32)
    bg0 = np.zeros((n_win, 3), np.float32)

    def _run_chunk(idx, p0_, v0_, q0_, ba0_, bg0_, sig=None):
        g = torch.as_tensor(g_in[idx], device=device).to(dtype)
        a = torch.as_tensor(a_in[idx], device=device).to(dtype)
        z = torch.as_tensor(z_in[idx], device=device).to(dtype)
        return run_windows(
            g, a, z,
            torch.as_tensor(p0_, device=device, dtype=dtype),
            torch.as_tensor(v0_, device=device, dtype=dtype),
            torch.as_tensor(q0_, device=device, dtype=dtype),
            torch.as_tensor(ba0_, device=device, dtype=dtype),
            torch.as_tensor(bg0_, device=device, dtype=dtype),
            initial_covariance(cfg, len(idx), dtype, device,
                               None if sig is None else np.asarray(sig)),
            cfg, smooth=True,
        )

    keys = ["p", "v", "q", "ba", "bg"]
    dims = {"p": 3, "v": 3, "q": 4, "ba": 3, "bg": 3}
    out_f = {k: np.full((n, dims[k]), np.nan, np.float32) for k in keys}
    out_s = {k: np.full((n, dims[k]), np.nan, np.float32) for k in keys}
    acc = {k: np.zeros((n, dims[k]), np.float32) for k in keys}
    wacc = np.zeros(n, np.float32)

    def _region(s, e):
        lo = 0 if s == 0 else W
        hi = T if e == n else T - W
        hi = min(min(hi, T), e - s)
        return lo, hi

    def _weights(lo, hi, s, e):
        # tent-shaped overlap-add weights; eps keeps every sample covered so the
        # normalisation never divides by zero at a non-overlapping seam
        eps = 1e-3
        w = np.ones(hi - lo, np.float32)
        if s > 0 and fade > 0:
            k = min(fade, len(w))
            w[:k] = eps + (1.0 - eps) * np.linspace(0.0, 1.0, k, endpoint=False, dtype=np.float32)
        if e < n and fade > 0:
            k = min(fade, len(w))
            ramp = eps + (1.0 - eps) * np.linspace(1.0, 0.0, k, endpoint=False, dtype=np.float32)
            w[-k:] = np.minimum(w[-k:], ramp)
        return w

    def _store(res, s, e, tgt, suf=""):
        lo, hi = _region(s, e)
        if hi <= lo:
            return
        for k in keys:
            tgt[k][s + lo: s + hi] = res[k + suf][:, lo:hi].cpu().numpy()

    def _accumulate(res, s, e):
        lo, hi = _region(s, e)
        if hi <= lo:
            return
        w = _weights(lo, hi, s, e)
        wacc[s + lo: s + hi] += w
        for k in keys:
            v = res[k + "_smooth"][:, lo:hi].cpu().numpy()[0]
            if k == "q":                      # hemisphere-consistent average
                ref = acc["q"][s + lo: s + hi]
                flip = np.sum(v * ref, axis=1) < 0.0
                v = np.where(flip[:, None], -v, v)
            acc[k][s + lo: s + hi] += w[:, None] * v

    stride = T - 2 * W - fade
    seeds = {k: np.zeros((n_win + 1, dims[k]), np.float32) for k in keys}
    seeds["p"][0], seeds["q"][0] = p0[0], q0[0]

    for c0 in range(0, n_win, chunk):
        idx = list(range(c0, min(c0 + chunk, n_win)))
        res = _run_chunk(idx, p0[idx], v0[idx], q0[idx], ba0[idx], bg0[idx], att_sig[idx])
        for j, i in enumerate(idx):
            s = starts[i]
            e = min(n, s + T)
            single = {k: res[k][j:j + 1] for k in res}
            _store(single, s, e, out_f)
            _store(single, s, e, out_s, "_smooth")
            if i + 1 < n_win:
                # seed the next window at *its own* start index, which equals
                # stride for every window except the tail one (that one is
                # placed at n-T, closer than stride, so a fixed stride would
                # seed it with a state from the future)
                jj = min(starts[i + 1] - s, res["p_smooth"].shape[1] - 1)
                for k in keys:
                    seeds[k][i + 1] = res[k + "_smooth"][j, jj].cpu().numpy()
        if verbose:
            print(f"  pass1 chunk {c0 // chunk + 1}/{(n_win + chunk - 1) // chunk}")

    if not two_pass:
        return {"starts": starts, "out_f": out_f, "out_s": out_s}

    # -------- pass 2: warm start from the previous window's estimate -------
    for i in range(1, n_win):
        p0[i], v0[i], q0[i], ba0[i], bg0[i] = (seeds[k][i] for k in keys)

    for c0 in range(0, n_win, chunk):
        idx = list(range(c0, min(c0 + chunk, n_win)))
        res = _run_chunk(idx, p0[idx], v0[idx], q0[idx], ba0[idx], bg0[idx],
                         np.full(len(idx), np.deg2rad(2.0), np.float32))
        for j, i in enumerate(idx):
            s = starts[i]
            e = min(n, s + T)
            single = {k: res[k][j:j + 1] for k in res}
            _store(single, s, e, out_f)      # pass-2 filtered (latest wins)
            _accumulate(single, s, e)        # overlap-add of the smoothed states
        if verbose:
            print(f"  pass2 chunk {c0 // chunk + 1}/{(n_win + chunk - 1) // chunk}")

    smoothed = {}
    w = np.maximum(wacc, 1e-6)[:, None]
    for k in keys:
        v = acc[k] / w
        if k == "q":
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
        smoothed[k] = v.astype(np.float32)
    return {"starts": starts, "filtered": out_f, "smoothed": smoothed,
            "weight": wacc}


def run_routes_batched(routes: dict, cfg: dict, device: str = "cuda",
                       dtype=torch.float32, two_pass: bool = True,
                       fade_s: float = 1.0, chunk: int | None = None,
                       verbose: bool = True) -> dict:
    """Window-parallel ESKF+RTS over **many routes at once**; numpy out.

    The per-step cost of this kernel is dominated by kernel-launch overhead and
    is essentially independent of the batch size (measured on the RTX 3090:
    ~4.2 ms/step from B=1 to B=64).  One route at 300 s windows only yields
    ~8 windows, i.e. a batch of 8 -- enough to fill the arithmetic units but
    not to amortise the launches.  Batching the windows of *all* routes into a
    single stream turns that fixed cost into throughput: 48 routes x 8 windows
    = 384 windows per step instead of 8.

    Parameters
    ----------
    routes : {route_id: {"gyro", "accel", "gnss", "outage", "p", ...}} numpy
    chunk  : windows per GPU chunk (default ``cfg["hpc"]["batch_chunk"]``);
             the covariance history is [chunk, T, 15, 15] float32.

    Returns ``{route_id: {"starts", "filtered", "smoothed", "weight"}}``, i.e.
    exactly what ``run_route_windowed`` returns for a single route.
    """
    fs = float(cfg["fs_imu"])
    T0 = int(cfg["hpc"]["window_T"])
    W0 = int(cfg["hpc"]["warm_W"])
    fade0 = int(round(fade_s * fs))
    if chunk is None:
        chunk = int(cfg["hpc"]["batch_chunk"])
    keys = ["p", "v", "q", "ba", "bg"]
    dims = {"p": 3, "v": 3, "q": 4, "ba": 3, "bg": 3}

    # -------- window bookkeeping, route by route ---------------------------
    wins = []
    per_route = {}
    for rid in sorted(routes):
        route = routes[rid]
        n = len(route["gyro"])
        T, W, fade = T0, W0, fade0
        if n < T:
            T, W, fade = n, max(1, n // 8), 0
        starts = window_starts(n, T, W, fade)
        idxs = []
        for i, s in enumerate(starts):
            e = min(n, s + T)
            lo = 0 if s == 0 else W
            hi = T if e == n else T - W
            hi = min(min(hi, T), e - s)
            idxs.append(len(wins))
            wins.append({"rid": rid, "s": s, "e": e, "T": T, "W": W,
                         "fade": fade, "lo": lo, "hi": hi})
        per_route[rid] = {"n": n, "starts": starts, "T": T, "W": W,
                          "fade": fade, "win_idx": idxs}

    n_win = len(wins)
    T = max(w["T"] for w in wins)
    # Keep the covariance history inside the card: one window's P history is
    # T * 15 * 15 * 4 bytes (~54 MB at T = 60000), so a 24 GB card holds ~180
    # of them while an 8 GB card holds ~55.  The per-step cost is flat in the
    # batch size, so shrinking the chunk costs throughput only in proportion --
    # it does not change the result.  Auto-capping keeps the default config
    # runnable on any card instead of OOM-ing on the first chunk.
    if device.startswith("cuda") and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        per_win = T * 225 * 4 * 1.35
        cap = max(4, int(0.6 * free / per_win))
        if chunk > cap:
            if verbose:
                print(f"  chunk {chunk} -> {cap} (free {free / 1e9:.1f} GB, "
                      f"~{per_win / 1e6:.0f} MB per window)")
            chunk = cap
    if verbose:
        print(f"  {len(routes)} routes -> {n_win} windows of {T} samples "
              f"({n_win / max(len(routes), 1):.1f} per route), chunk={chunk}")

    # -------- one padded input tensor for every window ---------------------
    g_in = np.zeros((n_win, T, 3), np.float32)
    a_in = np.zeros((n_win, T, 3), np.float32)
    z_in = np.full((n_win, T, 3), np.nan, np.float32)
    for j, w in enumerate(wins):
        r = routes[w["rid"]]
        s, e = w["s"], w["e"]
        L = e - s
        g_in[j, :L] = r["gyro"][s:e]
        a_in[j, :L] = r["accel"][s:e]
        z_in[j, :L] = r["gnss"][s:e]
        if L < T:                            # edge-replicate the tail
            g_in[j, L:] = r["gyro"][e - 1]
            a_in[j, L:] = r["accel"][e - 1]

    # -------- pass 1 alignment: GNSS + gyro, every window independently ----
    p0 = np.zeros((n_win, 3), np.float32)
    v0 = np.zeros((n_win, 3), np.float32)
    q0 = np.zeros((n_win, 4), np.float32)
    att_sig = np.zeros(n_win, np.float32)
    for j, w in enumerate(wins):
        r = routes[w["rid"]]
        pi, qi, si = gnss_alignment(r["gnss"], r["gyro"], fs, w["s"],
                                    sigma_gnss=float(cfg["sim"]["gnss_sigma_h"]))
        p0[j], q0[j], att_sig[j] = pi, qi, si
    ba0 = np.zeros((n_win, 3), np.float32)
    bg0 = np.zeros((n_win, 3), np.float32)

    def _run_chunk(idx, p0_, v0_, q0_, ba0_, bg0_, sig=None):
        g = torch.as_tensor(g_in[idx], device=device).to(dtype)
        a = torch.as_tensor(a_in[idx], device=device).to(dtype)
        z = torch.as_tensor(z_in[idx], device=device).to(dtype)
        return run_windows(
            g, a, z,
            torch.as_tensor(p0_, device=device, dtype=dtype),
            torch.as_tensor(v0_, device=device, dtype=dtype),
            torch.as_tensor(q0_, device=device, dtype=dtype),
            torch.as_tensor(ba0_, device=device, dtype=dtype),
            torch.as_tensor(bg0_, device=device, dtype=dtype),
            initial_covariance(cfg, len(idx), dtype, device,
                               None if sig is None else np.asarray(sig)),
            cfg, smooth=True,
        )

    seeds = {k: np.zeros((n_win, dims[k]), np.float32) for k in keys}
    for rid, pr in per_route.items():
        first = pr["win_idx"][0]
        seeds["p"][first], seeds["q"][first] = p0[first], q0[first]

    # -------- pass 1: seeds for the next window of the same route ----------
    starts = [w["s"] for w in wins]
    for c0 in range(0, n_win, chunk):
        idx = list(range(c0, min(c0 + chunk, n_win)))
        res = _run_chunk(idx, p0[idx], v0[idx], q0[idx], ba0[idx], bg0[idx],
                         att_sig[idx])
        for j, i in enumerate(idx):
            nxt = i + 1
            if nxt < n_win and wins[nxt]["rid"] == wins[i]["rid"]:
                jj = min(starts[nxt] - starts[i], res["p_smooth"].shape[1] - 1)
                for k in keys:
                    seeds[k][nxt] = res[k + "_smooth"][j, jj].cpu().numpy()
        if verbose:
            print(f"  pass1 chunk {c0 // chunk + 1}/{(n_win + chunk - 1) // chunk}",
                  flush=True)
    del res

    # -------- per-route output buffers -------------------------------------
    out_f, acc, wacc = {}, {}, {}
    for rid, pr in per_route.items():
        n = pr["n"]
        out_f[rid] = {k: np.full((n, dims[k]), np.nan, np.float32) for k in keys}
        acc[rid] = {k: np.zeros((n, dims[k]), np.float32) for k in keys}
        wacc[rid] = np.zeros(n, np.float32)

    def _weights(lo, hi, s, e, fade, n):
        eps = 1e-3
        w = np.ones(hi - lo, np.float32)
        if s > 0 and fade > 0:
            k = min(fade, len(w))
            w[:k] = eps + (1.0 - eps) * np.linspace(0.0, 1.0, k, endpoint=False,
                                                    dtype=np.float32)
        if e < n and fade > 0:
            k = min(fade, len(w))
            ramp = eps + (1.0 - eps) * np.linspace(1.0, 0.0, k, endpoint=False,
                                                  dtype=np.float32)
            w[-k:] = np.minimum(w[-k:], ramp)
        return w

    def _place(res, j, i):
        w = wins[i]
        rid, s, lo, hi = w["rid"], w["s"], w["lo"], w["hi"]
        if hi <= lo:
            return
        tgt_f, tgt_a = out_f[rid], acc[rid]
        for k in keys:
            tgt_f[k][s + lo: s + hi] = res[k][j, lo:hi].cpu().numpy()
        ww = _weights(lo, hi, s, w["e"], w["fade"], per_route[rid]["n"])
        wacc[rid][s + lo: s + hi] += ww
        for k in keys:
            v = res[k + "_smooth"][j, lo:hi].cpu().numpy()
            if k == "q":                     # hemisphere-consistent average
                ref = tgt_a["q"][s + lo: s + hi]
                flip = np.sum(v * ref, axis=1) < 0.0
                v = np.where(flip[:, None], -v, v)
            tgt_a[k][s + lo: s + hi] += ww[:, None] * v

    # -------- pass 2: warm start from the previous window's estimate -------
    for i in range(n_win):
        p0[i], v0[i], q0[i], ba0[i], bg0[i] = (seeds[k][i] for k in keys)

    for c0 in range(0, n_win, chunk):
        idx = list(range(c0, min(c0 + chunk, n_win)))
        res = _run_chunk(idx, p0[idx], v0[idx], q0[idx], ba0[idx], bg0[idx],
                         np.full(len(idx), np.deg2rad(2.0), np.float32))
        for j, i in enumerate(idx):
            _place(res, j, i)
        if verbose:
            print(f"  pass2 chunk {c0 // chunk + 1}/{(n_win + chunk - 1) // chunk}",
                  flush=True)

    out = {}
    for rid, pr in per_route.items():
        w = np.maximum(wacc[rid], 1e-6)[:, None]
        smoothed = {}
        for k in keys:
            v = acc[rid][k] / w
            if k == "q":
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
            smoothed[k] = v.astype(np.float32)
        out[rid] = {"starts": pr["starts"], "filtered": out_f[rid],
                    "smoothed": smoothed, "weight": wacc[rid]}
    return out


def _main() -> None:
    import argparse
    import json
    import time

    from ..common import NPZ_DIR, RUNS_DIR
    from ..eval.metrics import traj_error, summarize

    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=int, nargs="*", default=None,
                        help="route ids (default: every route under data/npz)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--n", type=int, default=None, help="cap samples (smoke test)")
    parser.add_argument("--fade", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config()
    if args.chunk:
        cfg["hpc"]["batch_chunk"] = args.chunk
    out_dir = RUNS_DIR / "kf_gpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    routes, paths = {}, {}
    for path in sorted(NPZ_DIR.glob("route_*.npz")):
        rid = int(path.stem.split("_")[1])
        if args.routes is not None and rid not in args.routes:
            continue
        with np.load(path, allow_pickle=True) as d:
            route = {k: d[k] for k in d.files}
        if args.n:
            route = {k: (v[: args.n] if getattr(v, "ndim", 0) > 0 else v)
                     for k, v in route.items()}
        routes[rid] = route
        paths[rid] = path

    # every route's windows go through the GPU in one stream: the per-step cost
    # is launch-bound and flat in the batch size (see run_routes_batched)
    t0 = time.time()
    res_all = run_routes_batched(routes, cfg, device=args.device, fade_s=args.fade)
    dt_total = time.time() - t0
    dt_s = dt_total / max(len(routes), 1)

    for rid, route in routes.items():
        stem = paths[rid].stem
        res = res_all[rid]
        outage = route["outage"].astype(bool)
        fs = float(cfg["fs_imu"])
        p_s = res["smoothed"]["p"]
        err = traj_error(p_s, route["p"])
        m = summarize(err, outage, fs)
        m["seconds"] = dt_s              # amortised over the shared batch
        m["seconds_total"] = dt_total
        m["n_windows"] = len(res["starts"])
        m["device"] = args.device
        all_metrics[stem] = m
        print(f"{stem:<28s} win={len(res['starts']):3d}  "
              f"outage max={m['outage_max']:6.2f} mean={m['outage_mean']:5.2f}  "
              f"rmse={m['rmse_all']:5.2f}", flush=True)
        np.savez_compressed(out_dir / f"{stem}.npz",
                            p_smooth=p_s,
                            p_filtered=res["filtered"]["p"],
                            weight=res["weight"],
                            starts=np.asarray(res["starts"]))
        with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=2)

    with open(out_dir / "metrics_gpu.json", "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"total {dt_total:.1f}s for {len(routes)} routes "
          f"({dt_s:.2f}s/route amortised)")
    print("-> " + str(out_dir / "metrics_gpu.json"))


if __name__ == "__main__":
    _main()
