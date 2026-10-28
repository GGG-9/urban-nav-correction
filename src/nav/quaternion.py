"""Quaternion utilities, scalar-first [w, x, y, z], Hamilton product.

Frames
------
b : body frame, FLU  (x forward, y left, z up)
n : navigation frame, ENU (x east, y north, z up)

``quat_to_dcm(q)`` returns :math:`C_b^n`, i.e. it maps a vector expressed in
the body frame to the navigation frame:  ``v_n = C_b^n @ v_b``.
"""
from __future__ import annotations

import numpy as np


def quat_mul(a, b):
    """Hamilton product of quaternions with a leading batch axis (any shape)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj(q):
    out = np.array(q, dtype=float, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_normalize(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_exp(v):
    """Exp map: rotation vector (axis * angle) -> unit quaternion."""
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    small = angle < 1e-10
    half = 0.5 * angle
    angle_safe = np.where(small, 1.0, angle)
    w = np.cos(half)
    xyz = np.sin(half) / angle_safe * v
    xyz = np.where(small, 0.5 * v, xyz)
    return np.concatenate([w, xyz], axis=-1)


def quat_log(q):
    """Log map: unit quaternion -> rotation vector (angle in [0, pi])."""
    q = quat_normalize(np.asarray(q, dtype=float))
    w = q[..., :1]
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(n, w)
    angle = np.where(angle > np.pi, angle - 2.0 * np.pi, angle)
    small = n < 1e-10
    n_safe = np.where(small, 1.0, n)
    out = angle * (v / n_safe)
    return np.where(small, 2.0 * v, out)


def quat_from_euler(yaw, pitch, roll=0.0):
    """Attitude from yaw / pitch / roll for C_b^n (ENU nav frame, FLU body).

    Convention: ``C_b^n = Rz(yaw) @ Ry(-pitch) @ Rx(roll)``.
    Positive yaw is counter-clockwise from East, **positive pitch is nose up**
    (body x-axis gains +z in the nav frame), positive roll is right wing down.
    """
    yaw = np.asarray(yaw, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    roll = np.asarray(roll, dtype=float)
    yaw, pitch, roll = np.broadcast_arrays(yaw, pitch, roll)
    zeros = np.zeros_like(yaw)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    qz = np.stack([cy, zeros, zeros, sy], axis=-1)
    qy = np.stack([cp, zeros, -sp, zeros], axis=-1)  # Ry(-pitch)
    qx = np.stack([cr, sr, zeros, zeros], axis=-1)
    return quat_mul(quat_mul(qz, qy), qx)


def quat_to_dcm(q):
    """Quaternion -> direction cosine matrix C_b^n (3x3 or batch [...,3,3])."""
    q = quat_normalize(np.asarray(q, dtype=float))
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    C = np.empty(q.shape[:-1] + (3, 3), dtype=float)
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


def skew(v):
    """Skew-symmetric matrix of a 3-vector (batch [...,3] -> [...,3,3])."""
    v = np.asarray(v, dtype=float)
    S = np.zeros(v.shape[:-1] + (3, 3), dtype=float)
    S[..., 0, 1] = -v[..., 2]
    S[..., 0, 2] = v[..., 1]
    S[..., 1, 0] = v[..., 2]
    S[..., 1, 2] = -v[..., 0]
    S[..., 2, 0] = -v[..., 1]
    S[..., 2, 1] = v[..., 0]
    return S
