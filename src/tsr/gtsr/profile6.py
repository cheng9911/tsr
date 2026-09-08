# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Six-generator SE(3) response-profile identification (Sprint F).

Faithful 6D port of ``maniskill_spectrum``'s ``SE3SmoothFinitePDiagModel``
(commit ``e5ce8a3``, blob ``35cf06a9``) plus the structural-selectivity selectivity
metrics. The generator diagonal is ``[du, dv, dw, d_roll, d_pitch, d_yaw]``;
rotation channels use the local convention ``R_local = Rx(roll) Ry(pitch) Rz(yaw)``
(column-vector, yaw innermost). This module neither loads ManiSkill / HDF5 data
(see test-side ``tests.gtsr._maniskill_data_se3``) nor touches the planar
``profile.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import least_squares

from .profile import adjudicate_cell, rbf_basis

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


class ResponseProfile6D:
    """Serializable, read-only ``q -> alpha`` finite diagonal profile (6 generators).

    ``alpha(q, rho)`` is the length-6 relevance vector ``[a_du, a_dv, a_dw,
    a_roll, a_pitch, a_yaw]`` under the frozen RBF-sigmoid parameterization, with
    ``g = (q + rho) / n_phases`` and ``q`` the fitting index (0..3). Unlike the
    planar profile, the 6D pipeline works in q-indices directly (the loader maps
    solver phase ``s -> q = s - 3``), so there is no solver-code map here.
    """

    __slots__ = ("_theta", "_n_basis", "_n_phases", "_alpha_max", "_basis_width", "_optimization", "_frozen")

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("ResponseProfile6D is read-only")
        object.__setattr__(self, name, value)

    def __init__(self, theta, alpha_max=1.25, basis_width=0.065, *, optimization=None, n_phases=4):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 2 or theta.shape[0] != SE3_DIM:
            raise ValueError(f"theta must have shape (6, n_basis), got {theta.shape}")
        self._theta = tuple(tuple(float(value) for value in row) for row in theta)
        self._n_basis = theta.shape[1]
        self._n_phases = int(n_phases)
        self._alpha_max = float(alpha_max)
        self._basis_width = float(basis_width)
        self._optimization = dict(optimization) if optimization else {}
        self._frozen = True

    @property
    def theta(self):
        return np.asarray(self._theta, dtype=np.float64).copy()

    @property
    def n_basis(self):
        return self._n_basis

    @property
    def alpha_max(self):
        return self._alpha_max

    @property
    def basis_width(self):
        return self._basis_width

    @property
    def optimization(self):
        return dict(self._optimization)

    def alpha_from_g(self, g):
        g = np.asarray(g, dtype=np.float64)
        basis = rbf_basis(np.atleast_1d(g), self._n_basis, self._basis_width)
        logits = basis @ np.asarray(self._theta, dtype=np.float64).T
        return self._alpha_max / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def alpha(self, q, rho):
        """Length-6 relevance at fitting phase ``q`` and within-phase progress ``rho``."""
        g = (int(q) + float(rho)) / self._n_phases
        return np.asarray(self.alpha_from_g(g)).reshape(-1)

    def to_dict(self):
        return {
            "alpha_max": self._alpha_max,
            "basis_width": self._basis_width,
            "n_basis": self._n_basis,
            "n_phases": self._n_phases,
            "theta": np.asarray(self._theta, dtype=np.float64).tolist(),
            "optimization": dict(self._optimization),
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        return cls(
            np.asarray(data["theta"], dtype=np.float64),
            alpha_max=float(data["alpha_max"]),
            basis_width=float(data["basis_width"]),
            optimization=data.get("optimization"),
            n_phases=int(data.get("n_phases", 4)),
        )


class ProfileIdentifier6D:
    """Frozen six-generator Pdiag-finite fitter: ``D_E -> theta`` (spec §3).

    Faithful 6D port of ``SE3SmoothFinitePDiagModel``: ``alpha_max``-constrained
    RBF-sigmoid generator profiles with second-difference smoothness, realized by
    ``C0 Exp(P(s) Log(C0^-1 C)) C0^-1 X0`` in SE(3), fitted with
    ``scipy.optimize.least_squares(method="trf")`` and three alternating
    nominal/profile updates. The nominal frame pose is supplied at fit time (like
    the planar ``ProfileIdentifier``); ``phase_codes`` (q-indices) is accepted but
    unused by the fit, mirroring the frozen model.
    """

    def __init__(self, alpha_max=1.25, n_basis=24, basis_width=0.065, smoothness_weight=0.1, nominal_iterations=3):
        self.alpha_max = float(alpha_max)
        self.n_basis = int(n_basis)
        self.basis_width = float(basis_width)
        self.smoothness_weight = float(smoothness_weight)
        self.nominal_iterations = int(nominal_iterations)

    def _basis(self, progress):
        return rbf_basis(progress, self.n_basis, self.basis_width)

    def _alpha(self, parameters, basis):
        logits = basis @ parameters.reshape(SE3_DIM, self.n_basis).T
        return self.alpha_max / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def _context_twist(self, context):
        intervention = se3_from_pose6(context)
        socket_world = self.nominal_frame @ intervention
        relative = self.nominal_frame_inverse @ socket_world
        return se3_log(relative)

    def _twists(self, contexts):
        return np.asarray([self._context_twist(context) for context in contexts])

    def _local_actions(self, contexts, alpha):
        twists = self._twists(contexts)
        scaled = twists[:, None, :] * alpha[None, :, :]
        return se3_exp_batched(scaled.reshape(-1, 6)).reshape(len(contexts), alpha.shape[0], 4, 4)

    def _predict_with(self, contexts, nominal_curve, alpha):
        contexts = np.asarray(contexts, dtype=np.float64)
        actions = self._local_actions(contexts, alpha)
        local_nominal = self.nominal_frame_inverse @ se3_from_pose6_batched(
            np.asarray(nominal_curve, dtype=np.float64)
        )
        pred_local = np.einsum("csij,sjk->csik", actions, local_nominal)
        pred_world = np.einsum("ij,csjk->csik", self.nominal_frame, pred_local)
        return pose6_from_se3_batched(pred_world)

    def _update_nominal(self, contexts, curves, alpha):
        contexts = np.asarray(contexts, dtype=np.float64)
        curves = np.asarray(curves, dtype=np.float64)
        actions = self._local_actions(contexts, alpha)
        curves_T = se3_from_pose6_batched(curves.reshape(-1, 6)).reshape(
            len(contexts), curves.shape[1], 4, 4
        )
        obs_local = np.einsum("ij,csjk->csik", self.nominal_frame_inverse, curves_T)
        actions_inv = se3_inverse_batched(actions)
        implied_local = np.einsum("csij,csjk->csik", actions_inv, obs_local)
        implied_world = np.einsum("ij,csjk->csik", self.nominal_frame, implied_local)
        implied = pose6_from_se3_batched(implied_world)
        nominal = np.empty((curves.shape[1], SE3_DIM), dtype=np.float64)
        nominal[:, :3] = implied[:, :, :3].mean(axis=0)
        for generator in range(3):
            angle = implied[:, :, 3 + generator]
            nominal[:, 3 + generator] = np.arctan2(
                np.sin(angle).mean(axis=0), np.cos(angle).mean(axis=0)
            )
        return nominal

    def _initial_parameters(self, pointwise_diagonal, basis):
        clipped = np.clip(np.asarray(pointwise_diagonal), 1e-4, self.alpha_max - 1e-4)
        logits = np.log(clipped / (self.alpha_max - clipped))
        regularizer = 1e-6 * np.eye(self.n_basis)
        inverse = np.linalg.solve(basis.T @ basis + regularizer, basis.T)
        parameters = [inverse @ logits[:, generator] for generator in range(SE3_DIM)]
        return np.asarray(parameters).reshape(-1)

    def _pointwise_diagonal(self, contexts, curves):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        n_steps = curves.shape[1]
        intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        diagonal = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        for step in range(n_steps):
            for generator in range(SE3_DIM):
                design = np.column_stack([np.ones(len(contexts_metric)), contexts_metric[:, generator]])
                coefficients = np.linalg.lstsq(design, curves_metric[:, step, generator], rcond=None)[0]
                intercept_metric[step, generator] = coefficients[0]
                diagonal[step, generator] = coefficients[1]
        return intercept_metric, diagonal

    def fit(self, contexts, curves, progress, phase_codes=None, nominal_frame_pose=None):
        contexts = np.asarray(contexts, dtype=np.float64)
        curves = np.asarray(curves, dtype=np.float64)
        progress = np.asarray(progress, dtype=np.float64)
        if contexts.ndim != 2 or contexts.shape[1] != 6:
            raise ValueError(f"contexts must have shape (n, 6), got {contexts.shape}")
        if curves.ndim != 3 or curves.shape[2] != 6 or curves.shape[0] != contexts.shape[0]:
            raise ValueError(f"curves must have shape (n, steps, 6), got {curves.shape}")
        if progress.ndim != 1 or progress.shape[0] != curves.shape[1]:
            raise ValueError(f"progress must have shape (steps,), got {progress.shape}")
        if not np.all(np.isfinite(contexts)) or not np.all(np.isfinite(curves)) or not np.all(np.isfinite(progress)):
            raise ValueError("contexts, curves and progress must be finite")
        if np.linalg.matrix_rank(se3_to_metric(contexts)) < 6:
            raise ValueError("intervention contexts are rank-deficient; all six generators must be excited")
        if not np.all(excitation_mask6(contexts)):
            raise ValueError("one or more generators lack signed (+/-) intervention coverage (UNEXCITED)")
        if nominal_frame_pose is None:
            raise ValueError("nominal_frame_pose is required")
        self.nominal_frame_pose = np.asarray(nominal_frame_pose, dtype=np.float64)
        if self.nominal_frame_pose.shape != (6,):
            raise ValueError("nominal_frame_pose must have shape (6,)")
        self.nominal_frame = se3_from_pose6(self.nominal_frame_pose)
        self.nominal_frame_inverse = se3_inverse(self.nominal_frame)
        if phase_codes is not None:
            phase_codes = np.asarray(phase_codes)
            if phase_codes.shape != progress.shape:
                raise ValueError("phase_codes must have shape (steps,)")
            if not np.issubdtype(phase_codes.dtype, np.integer):
                raise ValueError("phase_codes must contain integer q indices")
            if not set(phase_codes.tolist()).issubset({0, 1, 2, 3}):
                raise ValueError("phase_codes must be q indices in {0, 1, 2, 3}")
            self.phase_codes = phase_codes.astype(np.int64)

        basis = self._basis(progress)
        intercept_metric, pointwise_diagonal = self._pointwise_diagonal(contexts, curves)
        nominal_curve = se3_from_metric(intercept_metric)
        parameters = self._initial_parameters(pointwise_diagonal, basis)

        optimization = None
        for iteration in range(self.nominal_iterations):

            def residual(candidate, contexts=contexts, curves=curves, nominal_curve=nominal_curve):
                alpha = self._alpha(candidate, basis)
                prediction = self._predict_with(contexts, nominal_curve, alpha)
                data_residual = prediction - curves
                data_residual[..., 3:] = wrap_angle(data_residual[..., 3:])
                data_residual = se3_to_metric(data_residual).reshape(-1)
                smooth = np.diff(alpha, n=2, axis=0)
                scale = np.sqrt(self.smoothness_weight * data_residual.size / max(smooth.size, 1)) * np.mean(
                    np.abs(se3_to_metric(contexts)), axis=0
                )
                smooth_residual = (smooth * scale[None, :]).reshape(-1)
                return np.concatenate([data_residual, smooth_residual])

            optimization = least_squares(
                residual,
                parameters,
                method="trf",
                max_nfev=500,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            parameters = optimization.x
            alpha = self._alpha(parameters, basis)
            if iteration + 1 < self.nominal_iterations:
                nominal_curve = self._update_nominal(contexts, curves, alpha)

        self.parameters = parameters.reshape(SE3_DIM, self.n_basis)
        self.diagonal = self._alpha(parameters, basis)
        self.nominal_curve = nominal_curve
        self.optimization_success = bool(optimization.success)
        self.optimization_cost = float(optimization.cost)
        self.optimization_nfev = int(optimization.nfev)
        self.optimization_optimality = float(optimization.optimality)
        return ResponseProfile6D(
            self.parameters,
            alpha_max=self.alpha_max,
            basis_width=self.basis_width,
            optimization={
                "success": self.optimization_success,
                "cost": self.optimization_cost,
                "nfev": self.optimization_nfev,
                "optimality": self.optimization_optimality,
            },
        )

    def predict(self, contexts):
        return self._predict_with(contexts, self.nominal_curve, self.diagonal)

    def jacobian_diag(self):
        return self.diagonal.copy()

    def action(self, context, alpha):
        """Return the frozen SE(3) left action ``C0 Exp(alpha . c) C0^-1``."""
        twist = self._context_twist(context)
        return self.nominal_frame @ se3_exp(np.asarray(alpha, dtype=np.float64) * twist) @ self.nominal_frame_inverse


def excitation_mask6(contexts, *, tol=1e-12):
    """Per-generator excitation booleans for 6-vector contexts (spec §7)."""
    contexts = np.asarray(contexts, dtype=np.float64)
    if contexts.ndim != 2:
        raise ValueError(f"contexts must have shape (n, 6), got {contexts.shape}")
    return (np.min(contexts, axis=0) < -tol) & (np.max(contexts, axis=0) > tol)


@dataclass(frozen=True)
class CellSelectivity:
    """Per-cell selectivity quantities for one fitted profile (keyed or circular)."""

    alpha_pre: np.ndarray      # (6,) mean alpha over q<2
    alpha_post: np.ndarray     # (6,) mean alpha over q>=2
    delta: np.ndarray          # (6,) alpha_pre - alpha_post
    delta_yaw: float
    delta_others_max: float
    rank1_yaw: bool
    transition: bool


class SelectivityStatus(str, Enum):
    """Per-generator selectivity verdict (spec §7)."""

    SELECTIVE = "selective"
    NON_SELECTIVE = "non_selective"
    ABSTAIN = "abstain"


class SelectivityMetrics6D:
    """M1 / purity / M2 rank / M3 transition, per cell and 18-pair aggregate (spec §6)."""

    YAW_INDEX = 5
    TRANSITION_TAU = 0.2

    def __init__(self, phase_codes):
        self.phase_codes = np.asarray(phase_codes, dtype=np.int64)
        self.pre_mask = self.phase_codes < 2

    def per_cell(self, profile):
        profile = np.asarray(profile, dtype=np.float64)
        alpha_pre = profile[self.pre_mask].mean(axis=0)
        alpha_post = profile[~self.pre_mask].mean(axis=0)
        delta = alpha_pre - alpha_post
        delta_yaw = float(delta[self.YAW_INDEX])
        delta_others_max = float(np.max(np.delete(delta, self.YAW_INDEX)))
        rank1_yaw = bool(int(np.argmax(delta)) == self.YAW_INDEX and delta_yaw > 0.0)
        transition = bool(delta_yaw > self.TRANSITION_TAU)
        return CellSelectivity(alpha_pre, alpha_post, delta, delta_yaw, delta_others_max, rank1_yaw, transition)

    def cross_arm(self, keyed_cell, circular_cell):
        """S_j = |alpha_pre_j(keyed) - alpha_pre_j(circular)| and the per-cell purity."""
        S = np.abs(keyed_cell.alpha_pre - circular_cell.alpha_pre)
        s_yaw = float(S[self.YAW_INDEX])
        total = float(S.sum())
        purity = (s_yaw / total) if total > 0 else np.nan
        return S, purity

    def aggregate(self, keyed_cells, circular_cells):
        """Aggregate 18 ``(keyed, circular)`` pairs into the frozen headline dict."""
        if len(keyed_cells) != len(circular_cells):
            raise ValueError("keyed_cells and circular_cells must have equal length")
        s_yaw_vals = []
        s_others_max_vals = []
        purity_vals = []
        S_mean = np.zeros(6)
        for keyed, circular in zip(keyed_cells, circular_cells):
            S, purity = self.cross_arm(keyed, circular)
            s_yaw_vals.append(S[self.YAW_INDEX])
            s_others_max_vals.append(np.max(np.delete(S, self.YAW_INDEX)))
            purity_vals.append(purity)
            S_mean += S
        S_mean /= len(keyed_cells)
        return {
            "S_yaw": float(np.mean(s_yaw_vals)),
            "S_others_max": float(np.mean(s_others_max_vals)),
            "purity": float(np.mean(purity_vals)),
            "rank": float(np.mean([cell.rank1_yaw for cell in keyed_cells])),
            "transition": float(np.mean([cell.transition for cell in keyed_cells])),
            "S_mean": S_mean,
        }


class SelectivityVerdict6D:
    """Two-layer verdict: identification (``CellStatus``) + selectivity (spec §7)."""

    YAW_INDEX = 5

    def __init__(self, headline):
        self.headline = headline

    def identify(self, contexts, *, observable, fit_succeeded, meets_gate):
        """Per-generator ``CellStatus`` from data (6-generator excitation)."""
        excited = excitation_mask6(contexts)
        observable = np.broadcast_to(np.asarray(observable), (6,))
        fit_succeeded = np.broadcast_to(np.asarray(fit_succeeded), (6,))
        meets_gate = np.broadcast_to(np.asarray(meets_gate), (6,))
        return [
            adjudicate_cell(
                excited=bool(exc),
                observable=bool(obs),
                fit_succeeded=bool(fit),
                meets_heldout_gate=bool(gate),
            )
            for exc, obs, fit, gate in zip(excited, observable, fit_succeeded, meets_gate)
        ]

    def selectivity(self, identification, *, fit_succeeded):
        """Per-generator ``SelectivityStatus``: yaw SELECTIVE iff all §6 gates pass."""
        headline = self.headline
        yaw_selective = bool(
            headline["S_yaw"] > 4.0 * headline["S_others_max"]
            and headline["rank"] == 1.0
            and headline["transition"] == 1.0
        )
        statuses = []
        for j, (_, ok) in enumerate(zip(identification, np.broadcast_to(np.asarray(fit_succeeded), (6,)))):
            if not bool(ok):
                statuses.append(SelectivityStatus.ABSTAIN)
            elif j == self.YAW_INDEX:
                statuses.append(SelectivityStatus.SELECTIVE if yaw_selective else SelectivityStatus.NON_SELECTIVE)
            else:
                statuses.append(SelectivityStatus.NON_SELECTIVE)
        return statuses
