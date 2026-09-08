# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Six-generator SE(3) response-profile identification (Sprint F).

Faithful 6D port of ``maniskill_spectrum``'s ``SE3SmoothFinitePDiagModel``
(commit ``e5ce8a3``, blob ``35cf06a9``) plus the threshold-free selectivity
metrics. The generator diagonal is ``[du, dv, dw, d_roll, d_pitch, d_yaw]``;
rotation channels use the local convention ``R_local = Rx(roll) Ry(pitch) Rz(yaw)``
(column-vector, yaw innermost). This module neither loads ManiSkill / HDF5 data
(see test-side ``tests.gtsr._maniskill_data_se3``) nor touches the planar
``profile.py``.
"""

from __future__ import annotations

import numpy as np

SE3_DIM = 6
# 3 translations in metres + 3 rotations at 0.03 m/rad (same axial scale as SE(2)).
SE3_METRIC_SCALE = np.array([1.0, 1.0, 1.0, 0.03, 0.03, 0.03], dtype=np.float64)


def se3_to_metric(values):
    return np.asarray(values, dtype=np.float64) * SE3_METRIC_SCALE


def se3_from_metric(values):
    return np.asarray(values, dtype=np.float64) / SE3_METRIC_SCALE


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


# --- SO(3)/SE(3) Lie-group math -------------------------------------------------


def _skew(w):
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=np.float64,
    )


def so3_exp(w):
    w = np.asarray(w, dtype=np.float64)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + _skew(w)
    W = _skew(w)
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * W
        + ((1.0 - np.cos(theta)) / (theta * theta)) * (W @ W)
    )


def so3_log(R):
    R = np.asarray(R, dtype=np.float64)
    theta = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if theta < 1e-12:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def se3_exp(xi):
    xi = np.asarray(xi, dtype=np.float64)
    v = xi[:3]
    w = xi[3:]
    theta = float(np.linalg.norm(w))
    R = so3_exp(w)
    if theta < 1e-12:
        V = np.eye(3) + 0.5 * _skew(w)
    else:
        W = _skew(w)
        V = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / (theta * theta)) * W
            + ((theta - np.sin(theta)) / (theta ** 3)) * (W @ W)
        )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    w = so3_log(R)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        Vinv = np.eye(3) - 0.5 * _skew(w)
    else:
        W = _skew(w)
        Vinv = (
            np.eye(3)
            - 0.5 * W
            + (1.0 / (theta * theta))
            * (1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta))))
            * (W @ W)
        )
    v = Vinv @ t
    return np.concatenate([v, w])


def se3_exp_batched(twists):
    """Vectorized SE(3) exponential: (..., 6) -> (..., 4, 4)."""
    twists = np.asarray(twists, dtype=np.float64)
    v = twists[..., :3]
    w = twists[..., 3:]
    theta = np.linalg.norm(w, axis=-1)
    theta_safe = np.where(theta < 1e-12, 1.0, theta)
    small = theta < 1e-12

    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    W = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    W[..., 0, 1] = -wz
    W[..., 0, 2] = wy
    W[..., 1, 0] = wz
    W[..., 1, 2] = -wx
    W[..., 2, 0] = -wy
    W[..., 2, 1] = wx
    W2 = np.einsum("...ij,...jk->...ik", W, W)

    a = np.where(small, 1.0, np.sin(theta_safe) / theta_safe)
    b = np.where(small, 0.5, (1.0 - np.cos(theta_safe)) / (theta_safe ** 2))
    c = np.where(small, 1.0 / 6.0, (theta_safe - np.sin(theta_safe)) / (theta_safe ** 3))

    eye = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    eye[..., 0, 0] = eye[..., 1, 1] = eye[..., 2, 2] = 1.0
    R = eye + a[..., None, None] * W + b[..., None, None] * W2
    V = eye + b[..., None, None] * W + c[..., None, None] * W2
    t = np.einsum("...ij,...j->...i", V, v)

    T = np.zeros(w.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_inverse(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def se3_inverse_batched(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


# --- Local euler convention R = Rx(roll) Ry(pitch) Rz(yaw) ----------------------


def matrix_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rx @ Ry @ Rz


def euler_from_matrix(R):
    R = np.asarray(R, dtype=np.float64)
    pitch = float(np.arcsin(np.clip(R[0, 2], -1.0, 1.0)))
    roll = float(np.arctan2(-R[1, 2], R[2, 2]))
    yaw = float(np.arctan2(-R[0, 1], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def matrix_from_euler_batched(rpy):
    rpy = np.asarray(rpy, dtype=np.float64)
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.zeros(roll.shape + (3, 3), dtype=np.float64)
    Rx[..., 0, 0] = 1.0
    Rx[..., 1, 1] = cr
    Rx[..., 1, 2] = -sr
    Rx[..., 2, 1] = sr
    Rx[..., 2, 2] = cr
    Ry = np.zeros(roll.shape + (3, 3), dtype=np.float64)
    Ry[..., 0, 0] = cp
    Ry[..., 0, 2] = sp
    Ry[..., 1, 1] = 1.0
    Ry[..., 2, 0] = -sp
    Ry[..., 2, 2] = cp
    Rz = np.zeros(roll.shape + (3, 3), dtype=np.float64)
    Rz[..., 0, 0] = cy
    Rz[..., 0, 1] = -sy
    Rz[..., 1, 0] = sy
    Rz[..., 1, 1] = cy
    Rz[..., 2, 2] = 1.0
    return np.einsum("...ij,...jk,...kl->...il", Rx, Ry, Rz)


def se3_from_pose6(pose6):
    pose6 = np.asarray(pose6, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = matrix_from_euler(pose6[3], pose6[4], pose6[5])
    T[:3, 3] = pose6[:3]
    return T


def se3_from_pose6_batched(pose6):
    pose6 = np.asarray(pose6, dtype=np.float64)
    T = np.zeros(pose6.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = matrix_from_euler_batched(pose6[..., 3:])
    T[..., :3, 3] = pose6[..., :3]
    T[..., 3, 3] = 1.0
    return T


def pose6_from_se3_batched(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    roll = np.arctan2(-R[..., 1, 2], R[..., 2, 2])
    pitch = np.arcsin(np.clip(R[..., 0, 2], -1.0, 1.0))
    yaw = np.arctan2(-R[..., 0, 1], R[..., 0, 0])
    out = np.empty(T.shape[:-2] + (6,), dtype=np.float64)
    out[..., :3] = T[..., :3, 3]
    out[..., 3] = roll
    out[..., 4] = pitch
    out[..., 5] = yaw
    return out


def pose6_from_se3(T):
    return pose6_from_se3_batched(np.asarray(T, dtype=np.float64))


def socket_frame6(context, nominal_frame):
    """SE(3) relation pose for a 6-vector intervention: ``C = C0 Exp(c)``.

    ``nominal_frame`` is the 4x4 nominal socket frame; ``context`` is the
    length-6 intervention ``[du, dv, dw, d_roll, d_pitch, d_yaw]``. This is the
    6D analogue of the planar :func:`~tsr.gtsr.profile.socket_frame`, and is the
    relation pose ``ResponseOperator`` consumes (spec §2, F-02).
    """
    return np.asarray(nominal_frame, dtype=np.float64) @ se3_from_pose6(context)
