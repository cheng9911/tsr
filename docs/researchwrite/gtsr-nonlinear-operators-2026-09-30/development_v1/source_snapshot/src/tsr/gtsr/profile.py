# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Response profile identification (Sprint E): finite diagonal profile ``zeta -> alpha``.

The first profile is the frozen planar ``[du, dv, d_axial]`` Pdiag-finite
parameterization, ported faithfully from ``maniskill_spectrum``'s
``SmoothFinitePDiagModel``. It is identified in the planar relation space and then
embedded to the 6-D twist / coefficient space consumed by
:class:`~tsr.gtsr.response.ResponseOperator`. This module neither fits a
six-generator profile (a later, separately-manifested result) nor loads
ManiSkill / HDF5 data.
"""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType

import numpy as np
from scipy.optimize import least_squares

# Frozen planar relation basis scale: [du, dv, d_axial] -> meters/meters/rad.
METRIC_SCALE = np.array([1.0, 1.0, 0.03])


def to_metric(values):
    return np.asarray(values, dtype=np.float64) * METRIC_SCALE


def from_metric(values):
    return np.asarray(values, dtype=np.float64) / METRIC_SCALE


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


def pose2_matrix(pose):
    x, y, yaw = np.asarray(pose, dtype=np.float64)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.array(
        [[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def matrix_pose2(transform):
    transform = np.asarray(transform, dtype=np.float64)
    return np.array(
        [
            transform[0, 2],
            transform[1, 2],
            np.arctan2(transform[1, 0], transform[0, 0]),
        ]
    )


def se2_exp(twist):
    vx, vy, yaw = np.asarray(twist, dtype=np.float64)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    if abs(yaw) < 1e-8:
        a = 1.0 - yaw**2 / 6.0
        b = 0.5 * yaw - yaw**3 / 24.0
    else:
        a = sine / yaw
        b = (1.0 - cosine) / yaw
    translation = np.array([[a, -b], [b, a]]) @ np.array([vx, vy])
    return np.array(
        [
            [cosine, -sine, translation[0]],
            [sine, cosine, translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def se2_log(transform):
    transform = np.asarray(transform, dtype=np.float64)
    yaw = np.arctan2(transform[1, 0], transform[0, 0])
    if abs(yaw) < 1e-8:
        a = 1.0 - yaw**2 / 6.0
        b = 0.5 * yaw - yaw**3 / 24.0
    else:
        a = np.sin(yaw) / yaw
        b = (1.0 - np.cos(yaw)) / yaw
    velocity = np.linalg.solve(np.array([[a, -b], [b, a]]), transform[:2, 2])
    return np.array([velocity[0], velocity[1], yaw], dtype=np.float64)


def socket_frame(context, nominal_xy, nominal_yaw=0.0):
    context = np.asarray(context, dtype=np.float64)
    return pose2_matrix(
        [
            nominal_xy[0] + context[0],
            nominal_xy[1] + context[1],
            nominal_yaw + context[2],
        ]
    )


# The only permitted planar -> 6-D embeddings (scope §2). Coordinates use
# ``[du, dv, 0, 0, 0, d_axial]``; coefficients use ``[a_du, a_dv, 1, 1, 1, a_yaw]``.
def planar_to_se3_twist(c):
    c = np.asarray(c, dtype=np.float64)
    return np.array([c[0], c[1], 0.0, 0.0, 0.0, c[2]])


def planar_to_se3_alpha(a):
    a = np.asarray(a, dtype=np.float64)
    return np.array([a[0], a[1], 1.0, 1.0, 1.0, a[2]])


def se2_to_se3(se2):
    se2 = np.asarray(se2, dtype=np.float64)
    out = np.eye(4)
    out[:2, :2] = se2[:2, :2]
    out[:2, 3] = se2[:2, 2]
    return out


class CellStatus(str, Enum):
    """Per-``(generator, phase)`` identification verdict (scope §1/§7)."""

    UNEXCITED = "unexcited"
    UNOBSERVABLE = "unobservable"
    FIT_FAILED = "fit_failed"
    MODEL_MISMATCH = "model_mismatch"
    IDENTIFIED = "identified"


def adjudicate_cell(*, excited, observable, fit_succeeded, meets_heldout_gate):
    """Return the cell status following the frozen §7 priority order."""
    if not excited:
        return CellStatus.UNEXCITED
    if not observable:
        return CellStatus.UNOBSERVABLE
    if not fit_succeeded:
        return CellStatus.FIT_FAILED
    if not meets_heldout_gate:
        return CellStatus.MODEL_MISMATCH
    return CellStatus.IDENTIFIED


def excitation_mask(contexts, *, tol=1e-12):
    """Per-generator excitation booleans from intervention contexts (scope §7 step 1).

    A generator is *excited* only when the training interventions cover both signs of
    that relation coordinate (a positive and a negative control); a column that never
    crosses zero cannot identify a signed response law and yields ``UNEXCITED``.
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    if contexts.ndim != 2:
        raise ValueError(f"contexts must have shape (n, 3), got {contexts.shape}")
    return (np.min(contexts, axis=0) < -tol) & (np.max(contexts, axis=0) > tol)


def cell_statuses(contexts, *, observable, fit_succeeded, meets_heldout_gate):
    """Produce per-generator ``CellStatus`` from data (scope §7 priority order).

    ``excited`` is derived from ``contexts`` (signed intervention coverage); the
    remaining inputs are per-generator booleans (scalars broadcast across the three
    planar generators). This is the data -> status adjudication path;
    :func:`adjudicate_cell` is only its underlying truth table.
    """
    excited = excitation_mask(contexts)
    shape = excited.shape
    observable = np.broadcast_to(np.asarray(observable), shape)
    fit_succeeded = np.broadcast_to(np.asarray(fit_succeeded), shape)
    meets_heldout_gate = np.broadcast_to(np.asarray(meets_heldout_gate), shape)
    return [
        adjudicate_cell(
            excited=bool(exc),
            observable=bool(obs),
            fit_succeeded=bool(fit),
            meets_heldout_gate=bool(gate),
        )
        for exc, obs, fit, gate in zip(excited, observable, fit_succeeded, meets_heldout_gate)
    ]


def rbf_basis(progress, n_basis=24, basis_width=0.065):
    """Normalized Gaussian RBF basis on ``[0, 1]`` (frozen parameterization)."""
    progress = np.asarray(progress, dtype=np.float64)
    centers = np.linspace(0.0, 1.0, n_basis)
    basis = np.exp(-0.5 * ((progress[:, None] - centers[None, :]) / basis_width) ** 2)
    return basis / basis.sum(axis=1, keepdims=True)


def _normalize_phase_to_q(phase_to_q):
    """Coerce and validate the oracle-phase -> q-index map.

    The frozen mapping is ``g(k, rho) = (q(k) + rho) / K`` with ``K`` the number of
    registered fitting phases; the ``q`` values must therefore be the contiguous
    range ``0 .. K-1`` (scope §3).
    """
    mapping = {int(k): int(q) for k, q in dict(phase_to_q).items()}
    if not mapping:
        raise ValueError("phase_to_q must map at least one phase code to a q index")
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("phase_to_q values must be the contiguous range 0..K-1")
    return mapping


class ResponseProfile:
    """Serializable, read-only ``zeta -> alpha`` finite diagonal profile.

    ``zeta = (k, rho)`` is the hybrid oracle phase; ``alpha(k, rho)`` is the
    length-3 planar relevance vector ``[a_du, a_dv, a_yaw]`` under the frozen
    RBF-sigmoid parameterization (scope §4). ``alpha6`` lifts it to the 6-D
    coefficient vector consumed by ``ResponseOperator`` (scope §2).
    """

    __slots__ = (
        "_theta",
        "_n_basis",
        "_phase_to_q",
        "_alpha_max",
        "_basis_width",
        "_optimization",
        "_frozen",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("ResponseProfile is read-only")
        object.__setattr__(self, name, value)

    def __init__(self, theta, phase_to_q, alpha_max=1.25, basis_width=0.065, *, optimization=None):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 2 or theta.shape[0] != 3:
            raise ValueError(f"theta must have shape (3, n_basis), got {theta.shape}")
        # Store coefficients as immutable Python values rather than an ndarray:
        # ``ndarray.setflags(write=True)`` can re-enable an array's own buffer.
        self._theta = tuple(tuple(float(value) for value in row) for row in theta)
        self._n_basis = theta.shape[1]
        self._phase_to_q = MappingProxyType(_normalize_phase_to_q(phase_to_q))
        self._alpha_max = float(alpha_max)
        self._basis_width = float(basis_width)
        self._optimization = MappingProxyType(dict(optimization) if optimization else {})
        self._frozen = True

    @property
    def theta(self):
        return np.asarray(self._theta, dtype=np.float64).copy()

    @property
    def n_basis(self):
        return self._n_basis

    @property
    def phase_to_q(self):
        return dict(self._phase_to_q)

    @property
    def alpha_max(self):
        return self._alpha_max

    @property
    def basis_width(self):
        return self._basis_width

    @property
    def optimization(self):
        return dict(self._optimization)

    def _alpha_from_g(self, g):
        g = np.asarray(g, dtype=np.float64)
        basis = rbf_basis(np.atleast_1d(g), self._n_basis, self._basis_width)
        logits = basis @ np.asarray(self._theta, dtype=np.float64).T
        return self._alpha_max / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def alpha(self, phase_code, rho):
        """Planar relevance ``[a_du, a_dv, a_yaw]`` at hybrid phase ``(k, rho)``."""
        q = self._phase_to_q[int(phase_code)]
        g = (q + float(rho)) / len(self._phase_to_q)
        return np.asarray(self._alpha_from_g(g)).reshape(-1)

    def alpha6(self, phase_code, rho):
        """6-D coefficient vector ``[a_du, a_dv, 1, 1, 1, a_yaw]`` (scope §2)."""
        return planar_to_se3_alpha(self.alpha(phase_code, rho))

    def to_dict(self):
        return {
            "alpha_max": self._alpha_max,
            "basis_width": self._basis_width,
            "n_basis": self._n_basis,
            "phase_to_q": {str(k): v for k, v in self._phase_to_q.items()},
            "theta": np.asarray(self._theta, dtype=np.float64).tolist(),
            "optimization": dict(self._optimization),
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        phase_to_q = {int(k): int(v) for k, v in data["phase_to_q"].items()}
        return cls(
            np.asarray(data["theta"], dtype=np.float64),
            phase_to_q,
            alpha_max=float(data["alpha_max"]),
            basis_width=float(data["basis_width"]),
            optimization=data.get("optimization"),
        )


class ProfileIdentifier:
    """Frozen planar Pdiag-finite fitter: ``D_E -> theta`` (scope §4).

    Faithful port of ``SmoothFinitePDiagModel`` (3 generators: du, dv, yaw):
    ``alpha_max``-constrained RBF-sigmoid generator profiles with second-difference
    smoothness, realized by ``C0 Exp(P(s) Log(C0^-1 C)) C0^-1 X0`` in SE(2), fitted
    with ``scipy.optimize.least_squares(method="trf")`` and three alternating
    nominal/profile updates.
    """

    def __init__(
        self,
        alpha_max=1.25,
        n_basis=24,
        basis_width=0.065,
        smoothness_weight=0.1,
        nominal_iterations=3,
    ):
        self.alpha_max = float(alpha_max)
        self.n_basis = int(n_basis)
        self.basis_width = float(basis_width)
        self.smoothness_weight = float(smoothness_weight)
        self.nominal_iterations = int(nominal_iterations)

    # -- frozen SE(2) fit machinery -------------------------------------------
    def _basis(self, progress):
        return rbf_basis(progress, self.n_basis, self.basis_width)

    def _alpha(self, parameters, basis):
        logits = basis @ parameters.reshape(3, self.n_basis).T
        return self.alpha_max / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def _context_twist(self, context):
        relative = self.nominal_frame_inverse @ socket_frame(context, self.nominal_frame_xy, self.nominal_frame_yaw)
        return se2_log(relative)

    def _batched_local_actions(self, contexts, alpha):
        twists = np.asarray([self._context_twist(context) for context in contexts])
        selected = twists[:, None, :] * alpha[None, :, :]
        yaw = selected[:, :, 2]
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        small = np.abs(yaw) < 1e-8
        a = np.empty_like(yaw)
        b = np.empty_like(yaw)
        a[small] = 1.0 - yaw[small] ** 2 / 6.0
        b[small] = 0.5 * yaw[small] - yaw[small] ** 3 / 24.0
        a[~small] = sine[~small] / yaw[~small]
        b[~small] = (1.0 - cosine[~small]) / yaw[~small]
        vx = selected[:, :, 0]
        vy = selected[:, :, 1]
        translation = np.stack([a * vx - b * vy, b * vx + a * vy], axis=-1)
        return yaw, cosine, sine, translation

    def _predict_with(self, contexts, nominal_curve, alpha):
        contexts = np.asarray(contexts, dtype=np.float64)
        yaw, cosine, sine, translation = self._batched_local_actions(contexts, alpha)
        frame_rotation = self.nominal_frame[:2, :2]
        nominal_local = (np.asarray(nominal_curve)[:, :2] - self.nominal_frame_xy) @ frame_rotation
        local_x = nominal_local[None, :, 0]
        local_y = nominal_local[None, :, 1]
        transformed_local = np.stack(
            [
                cosine * local_x - sine * local_y + translation[:, :, 0],
                sine * local_x + cosine * local_y + translation[:, :, 1],
            ],
            axis=-1,
        )
        prediction = np.empty((len(contexts), len(nominal_curve), 3), dtype=np.float64)
        prediction[:, :, :2] = transformed_local @ frame_rotation.T + self.nominal_frame_xy[None, None, :]
        prediction[:, :, 2] = wrap_angle(np.asarray(nominal_curve)[None, :, 2] + yaw)
        return prediction

    def _initial_parameters(self, pointwise_diagonal, basis):
        clipped = np.clip(np.asarray(pointwise_diagonal), 1e-4, self.alpha_max - 1e-4)
        logits = np.log(clipped / (self.alpha_max - clipped))
        regularizer = 1e-6 * np.eye(self.n_basis)
        inverse = np.linalg.solve(basis.T @ basis + regularizer, basis.T)
        parameters = [inverse @ logits[:, generator] for generator in range(3)]
        return np.asarray(parameters).reshape(-1)

    def _update_nominal(self, contexts, curves, alpha):
        yaw, cosine, sine, translation = self._batched_local_actions(contexts, alpha)
        frame_rotation = self.nominal_frame[:2, :2]
        observed_local = (np.asarray(curves)[:, :, :2] - self.nominal_frame_xy[None, None, :]) @ frame_rotation
        shifted = observed_local - translation
        implied_local = np.stack(
            [
                cosine * shifted[:, :, 0] + sine * shifted[:, :, 1],
                -sine * shifted[:, :, 0] + cosine * shifted[:, :, 1],
            ],
            axis=-1,
        )
        implied = np.empty_like(curves, dtype=np.float64)
        implied[:, :, :2] = implied_local @ frame_rotation.T + self.nominal_frame_xy[None, None, :]
        implied[:, :, 2] = wrap_angle(curves[:, :, 2] - yaw)
        nominal = np.empty((curves.shape[1], 3), dtype=np.float64)
        nominal[:, :2] = implied[:, :, :2].mean(axis=0)
        nominal[:, 2] = np.arctan2(
            np.sin(implied[:, :, 2]).mean(axis=0),
            np.cos(implied[:, :, 2]).mean(axis=0),
        )
        return nominal

    # -- public interface -----------------------------------------------------
    def fit(
        self,
        contexts,
        curves,
        progress,
        phase_codes=None,
        nominal_frame_xy=None,
        nominal_frame_yaw=0.0,
        phase_to_q=None,
    ):
        contexts = np.asarray(contexts, dtype=np.float64)
        curves = np.asarray(curves, dtype=np.float64)
        progress = np.asarray(progress, dtype=np.float64)
        if contexts.ndim != 2 or contexts.shape[1] != 3:
            raise ValueError(f"contexts must have shape (n, 3), got {contexts.shape}")
        if curves.ndim != 3 or curves.shape[2] != 3 or curves.shape[0] != contexts.shape[0]:
            raise ValueError(f"curves must have shape (n, steps, 3), got {curves.shape}")
        if progress.ndim != 1 or progress.shape[0] != curves.shape[1]:
            raise ValueError(f"progress must have shape (steps,), got {progress.shape}")
        if not np.all(np.isfinite(contexts)) or not np.all(np.isfinite(curves)):
            raise ValueError("contexts and curves must be finite")
        if not np.all(np.isfinite(progress)):
            raise ValueError("progress must be finite")
        if np.linalg.matrix_rank(to_metric(contexts)) < 3:
            raise ValueError("intervention contexts are rank-deficient; all three generators must be excited")
        if not np.all(excitation_mask(contexts)):
            raise ValueError("one or more generators lack signed (+/-) intervention coverage (UNEXCITED)")
        if nominal_frame_xy is None:
            raise ValueError("nominal_frame_xy is required")
        self.nominal_frame_xy = np.asarray(nominal_frame_xy, dtype=np.float64)
        if self.nominal_frame_xy.shape != (2,):
            raise ValueError("nominal_frame_xy must have shape (2,)")
        self.nominal_frame_yaw = float(nominal_frame_yaw)
        self.nominal_frame = pose2_matrix([*self.nominal_frame_xy, self.nominal_frame_yaw])
        self.nominal_frame_inverse = np.linalg.inv(self.nominal_frame)
        phase_to_q = _normalize_phase_to_q({3: 0, 4: 1, 5: 2, 6: 3} if phase_to_q is None else phase_to_q)
        if phase_codes is not None:
            phase_codes = np.asarray(phase_codes)
            if phase_codes.shape != progress.shape:
                raise ValueError("phase_codes must have shape (steps,)")
            if not np.issubdtype(phase_codes.dtype, np.integer):
                raise ValueError("phase_codes must contain integer phase codes")
            unknown_codes = set(phase_codes.tolist()).difference(phase_to_q)
            if unknown_codes:
                raise ValueError(f"phase_codes contain values outside the registered phase_to_q map: {unknown_codes}")
            phase_indices = np.asarray([phase_to_q[int(code)] for code in phase_codes], dtype=np.float64)
            phase_count = len(phase_to_q)
            lower = phase_indices / phase_count
            upper = (phase_indices + 1.0) / phase_count
            if np.any(progress < lower - 1e-12) or np.any(progress > upper + 1e-12):
                raise ValueError("progress is inconsistent with phase_codes and phase_to_q")
            self.phase_codes = phase_codes.astype(np.int64)

        basis = self._basis(progress)
        intercept_metric, pointwise_diagonal = self._pointwise_diagonal(contexts, curves)
        nominal_curve = from_metric(intercept_metric)
        parameters = self._initial_parameters(pointwise_diagonal, basis)

        optimization = None
        for iteration in range(self.nominal_iterations):

            def residual(candidate, contexts=contexts, curves=curves, nominal_curve=nominal_curve):
                alpha = self._alpha(candidate, basis)
                prediction = self._predict_with(contexts, nominal_curve, alpha)
                data_residual = prediction - curves
                data_residual[:, :, 2] = wrap_angle(data_residual[:, :, 2])
                data_residual = to_metric(data_residual).reshape(-1)
                smooth = np.diff(alpha, n=2, axis=0)
                scale = np.sqrt(self.smoothness_weight * data_residual.size / max(smooth.size, 1)) * np.mean(
                    np.abs(to_metric(contexts)), axis=0
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

        self.parameters = parameters.reshape(3, self.n_basis)
        self.diagonal = self._alpha(parameters, basis)
        self.nominal_curve = nominal_curve
        self.optimization_success = bool(optimization.success)
        self.optimization_cost = float(optimization.cost)
        self.optimization_nfev = int(optimization.nfev)
        self.optimization_optimality = float(optimization.optimality)
        return ResponseProfile(
            self.parameters,
            phase_to_q,
            alpha_max=self.alpha_max,
            basis_width=self.basis_width,
            optimization={
                "success": self.optimization_success,
                "cost": self.optimization_cost,
                "nfev": self.optimization_nfev,
                "optimality": self.optimization_optimality,
            },
        )

    def _pointwise_diagonal(self, contexts, curves):
        contexts_metric = to_metric(contexts)
        curves_metric = to_metric(curves)
        n_steps = curves.shape[1]
        intercept_metric = np.empty((n_steps, 3), dtype=np.float64)
        diagonal = np.empty((n_steps, 3), dtype=np.float64)
        for step in range(n_steps):
            for generator in range(3):
                design = np.column_stack([np.ones(len(contexts_metric)), contexts_metric[:, generator]])
                coefficients = np.linalg.lstsq(design, curves_metric[:, step, generator], rcond=None)[0]
                intercept_metric[step, generator] = coefficients[0]
                diagonal[step, generator] = coefficients[1]
        return intercept_metric, diagonal

    def predict(self, contexts):
        return self._predict_with(contexts, self.nominal_curve, self.diagonal)

    def jacobian_diag(self):
        return self.diagonal.copy()

    def action(self, context, alpha):
        """Return the frozen SE(2) left action ``C0 Exp(alpha . c) C0^-1``."""
        twist = self._context_twist(context)
        return self.nominal_frame @ se2_exp(np.asarray(alpha, dtype=np.float64) * twist) @ self.nominal_frame_inverse
