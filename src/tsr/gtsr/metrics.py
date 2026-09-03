# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Explicit SE(3) metric and Lie-group primitives for the G-TSR kernel.

Twist ordering is ``xi = [v, w]`` (linear ``v`` first, angular ``w`` last).
"""

import numpy as np

from ._validation import finite_vector, rotation_matrix, se3_transform

_TINY = 1e-12


def skew(w):
    w = finite_vector(w, 3, "w")
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def se3_hat(xi):
    xi = finite_vector(xi, 6, "xi")
    H = np.zeros((4, 4))
    H[:3, :3] = skew(xi[3:6])
    H[:3, 3] = xi[0:3]
    return H


def se3_vee(H):
    H = np.asarray(H, dtype=float)
    if H.shape != (4, 4):
        raise ValueError("se3_vee expects a 4x4 matrix")
    W = H[:3, :3]
    return np.array([H[0, 3], H[1, 3], H[2, 3], W[2, 1], W[0, 2], W[1, 0]])


def so3_log(R):
    R = rotation_matrix(R)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sin_theta = 0.5 * np.linalg.norm(vee)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arctan2(sin_theta, cos_theta)
    if theta < _TINY:
        return vee / 2.0
    if cos_theta < 0.0:
        # For obtuse rotations, extracting the axis from the skew part becomes
        # ill-conditioned as theta approaches pi.  The principal eigenvector of
        # sym(R) remains well-conditioned here: its eigengap is 1-cos(theta) >= 1.
        _, eigenvectors = np.linalg.eigh((R + R.T) / 2.0)
        axis = eigenvectors[:, -1]
        if np.dot(axis, vee) < 0.0:
            axis = -axis
        return theta * axis
    return theta / (2.0 * sin_theta) * vee


def se3_exp(xi):
    xi = finite_vector(xi, 6, "xi")
    v, w = xi[0:3], xi[3:6]
    theta = np.linalg.norm(w)
    T = np.eye(4)
    if theta < _TINY:
        T[:3, :3] = np.eye(3) + skew(w)
        T[:3, 3] = v
        return T
    W = skew(w)
    W2 = W @ W
    R = np.eye(3) + np.sin(theta) / theta * W + (1.0 - np.cos(theta)) / theta**2 * W2
    V = np.eye(3) + (1.0 - np.cos(theta)) / theta**2 * W + (theta - np.sin(theta)) / theta**3 * W2
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T):
    T = se3_transform(T, "T")
    R, p = T[:3, :3], T[:3, 3]
    w = so3_log(R)
    theta = np.linalg.norm(w)
    if theta < _TINY:
        return np.hstack([p, w])
    W = skew(w)
    W2 = W @ W
    if theta < 1e-4:
        theta2 = theta * theta
        coef = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    else:
        half_theta = theta / 2.0
        coef = (1.0 - half_theta / np.tan(half_theta)) / theta**2
    V_inv = np.eye(3) - 0.5 * W + coef * W2
    return np.hstack([V_inv @ p, w])


def se3_adjoint(T):
    T = se3_transform(T, "T")
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def se3_left_jacobian_inverse(xi):
    xi = finite_vector(xi, 6, "xi")
    v, w = xi[0:3], xi[3:6]
    # ad_xi for [v, w] ordering: [[skew(w), skew(v)], [0, skew(w)]]
    ad = np.zeros((6, 6))
    W = skew(w)
    ad[:3, :3] = W
    ad[:3, 3:] = skew(v)
    ad[3:, 3:] = W
    # J_l(xi) = sum_n (-1)^n/(n+1)! ad^n = (I - exp(-ad)) ad^{-1}.
    # The se(3) inverse is NOT the truncated SO(3) form ``I - 1/2 ad + gamma ad^2``
    # (that drops the translation/rotation coupling in ad^3 and above); the series
    # converges rapidly (theta <= pi for a principal log) and the 6x6 inverse is
    # exact to machine precision.
    Jl = np.eye(6)
    term = np.eye(6)
    for k in range(1, 60):
        term = term @ (-ad) / (k + 1)
        Jl = Jl + term
    return np.linalg.inv(Jl)


class SE3Metric:
    """Weighted SE(3) metric ``d_W^2 = ||rho||^2 + w^2 ||theta||^2``."""

    def __init__(self, rotation_weight=1.0):
        try:
            self.rotation_weight = float(rotation_weight)
        except (TypeError, ValueError) as exc:
            raise TypeError("rotation_weight must be a real scalar") from exc
        if not np.isfinite(self.rotation_weight) or self.rotation_weight <= 0.0:
            raise ValueError("rotation_weight must be finite and strictly positive")

    def error(self, T_from, T_to):
        T_from = se3_transform(T_from, "T_from")
        T_to = se3_transform(T_to, "T_to")
        return se3_log(np.linalg.inv(T_from) @ T_to)

    def distance(self, T_from, T_to):
        r = self.error(T_from, T_to)
        return np.sqrt(self.distance_sq(r))

    def distance_sq(self, r):
        r = finite_vector(r, 6, "r")
        w = self.rotation_weight
        return float(r[0:3] @ r[0:3] + w * w * (r[3:6] @ r[3:6]))

    def weighted_error(self, r):
        r = finite_vector(r, 6, "r")
        w = self.rotation_weight
        out = r.copy()
        out[3:6] *= w * w
        return out

    def error_body_jacobian(self, T_from, T_to):
        """E = d r / d(body perturbation of T_from), r = Log(T_from^-1 T_to)."""
        T_from = se3_transform(T_from, "T_from")
        T_to = se3_transform(T_to, "T_to")
        g = np.linalg.inv(T_from) @ T_to
        r = se3_log(g)
        return -se3_left_jacobian_inverse(r) @ se3_adjoint(np.linalg.inv(g))
