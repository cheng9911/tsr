# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Test-side E-09 baseline adapters with explicit planar left-action semantics.

This module is deliberately under ``tests``: it ports frozen experimental controls
without making ManiSkill or learning-baseline dependencies part of the ``tsr``
runtime package. Direct predictors added in later E-09 tasks are adapted through
``nominal_relative_actions``; they are not response-law classes.
"""

from __future__ import annotations

import numpy as np

from tsr.gtsr.profile import (
    ProfileIdentifier,
    from_metric,
    matrix_pose2,
    pose2_matrix,
    se2_exp,
    se2_log,
    socket_frame,
    to_metric,
)


def _sklearn_components():
    """Load the E-09-only learning dependency with an actionable failure."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import GroupKFold
    except ImportError as error:  # pragma: no cover - integration suite skip-gates this extra.
        raise ImportError("E-09 learning baselines require the 'test' extra (scikit-learn)") from error
    return GaussianProcessRegressor, ConstantKernel, RBF, WhiteKernel, RidgeCV, GroupKFold


def _curve(curve, name):
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 3 or not np.all(np.isfinite(curve)):
        raise ValueError(f"{name} must be a finite array shaped (steps, 3)")
    return curve


def _contexts(contexts):
    contexts = np.asarray(contexts, dtype=np.float64)
    if contexts.ndim != 2 or contexts.shape[1] != 3 or not np.all(np.isfinite(contexts)):
        raise ValueError("contexts must be a finite array shaped (n, 3)")
    return contexts


def nominal_relative_actions(prediction, nominal_curve):
    """Return ``A=X_hat X_0^-1`` for direct pose/update comparators."""
    prediction = np.asarray(prediction, dtype=np.float64)
    nominal_curve = _curve(nominal_curve, "nominal_curve")
    if prediction.ndim != 3 or prediction.shape[1:] != nominal_curve.shape:
        raise ValueError("prediction must have shape (n, steps, 3) matching nominal_curve")
    nominal_inverse = [np.linalg.inv(pose2_matrix(pose)) for pose in nominal_curve]
    return np.asarray(
        [
            [pose2_matrix(prediction[episode, step]) @ nominal_inverse[step] for step in range(len(nominal_curve))]
            for episode in range(len(prediction))
        ],
        dtype=np.float64,
    )


def apply_actions(actions, nominal_curve):
    """Apply planar left actions to the common nominal curve."""
    actions = np.asarray(actions, dtype=np.float64)
    nominal_curve = _curve(nominal_curve, "nominal_curve")
    if actions.ndim != 4 or actions.shape[1:] != (len(nominal_curve), 3, 3):
        raise ValueError("actions must have shape (n, steps, 3, 3) matching nominal_curve")
    return np.asarray(
        [
            [
                matrix_pose2(actions[episode, step] @ pose2_matrix(nominal_curve[step]))
                for step in range(len(nominal_curve))
            ]
            for episode in range(len(actions))
        ],
        dtype=np.float64,
    )


class FixedActionModel:
    """TSR-Fixed control: ``P=0`` and hence the nominal trajectory unchanged."""

    name = "TSR-Fixed"

    def __init__(self, nominal_curve):
        self.nominal_curve = _curve(nominal_curve, "nominal_curve").copy()

    def fit(self, contexts, curves, progress, phase_codes):
        _contexts(contexts)
        _curve(np.asarray(curves)[0], "curves[0]")
        return self

    def predict(self, contexts):
        contexts = _contexts(contexts)
        return np.repeat(self.nominal_curve[None, :, :], len(contexts), axis=0)

    def actions(self, contexts):
        contexts = _contexts(contexts)
        return np.tile(np.eye(3), (len(contexts), len(self.nominal_curve), 1, 1))


class FrameActionModel(FixedActionModel):
    """TSR-Frame control: identity propagation ``P=I`` in the nominal relation frame."""

    name = "TSR-Frame"

    def __init__(self, nominal_curve, nominal_frame_xy, nominal_frame_yaw=0.0):
        super().__init__(nominal_curve)
        self.nominal_frame_xy = np.asarray(nominal_frame_xy, dtype=np.float64)
        if self.nominal_frame_xy.shape != (2,) or not np.all(np.isfinite(self.nominal_frame_xy)):
            raise ValueError("nominal_frame_xy must be finite with shape (2,)")
        self.nominal_frame_yaw = float(nominal_frame_yaw)
        if not np.isfinite(self.nominal_frame_yaw):
            raise ValueError("nominal_frame_yaw must be finite")
        self.nominal_frame = pose2_matrix([*self.nominal_frame_xy, self.nominal_frame_yaw])
        self.nominal_frame_inverse = np.linalg.inv(self.nominal_frame)

    def action(self, context):
        context = _contexts(np.asarray(context, dtype=np.float64).reshape(1, -1))[0]
        relation = socket_frame(context, self.nominal_frame_xy, self.nominal_frame_yaw)
        twist = se2_log(self.nominal_frame_inverse @ relation)
        return self.nominal_frame @ se2_exp(twist) @ self.nominal_frame_inverse

    def actions(self, contexts):
        contexts = _contexts(contexts)
        per_context = np.asarray([self.action(context) for context in contexts])
        return np.repeat(per_context[:, None, :, :], len(self.nominal_curve), axis=1)

    def predict(self, contexts):
        return apply_actions(self.actions(contexts), self.nominal_curve)


class PhaseScalarActionModel(FrameActionModel):
    """Frozen phase-scalar GP profile, realized as ``P=w(zeta) I``.

    The pointwise scalar and GP hyperparameters are the sister implementation's
    frozen estimator. Unlike that source's direct affine predictor, E-09 realizes
    the learned scalar through the common nominal-frame left action so the only
    difference from Pdiag is scalar versus generator-wise selectivity.
    """

    name = "Phase-Scalar GP"
    kernel_amplitude = 1.0
    kernel_amplitude_bounds = (1e-2, 1e2)
    length_scale = 0.12
    length_scale_bounds = (0.015, 1.0)
    noise_level = 1e-4
    noise_level_bounds = (1e-8, 1e-1)
    optimizer_restarts = 5
    random_state = 20260820

    def fit(self, contexts, curves, progress, phase_codes):
        contexts = _contexts(contexts)
        curves = np.asarray(curves, dtype=np.float64)
        progress = np.asarray(progress, dtype=np.float64)
        phase_codes = np.asarray(phase_codes)
        expected = (len(contexts), len(self.nominal_curve), 3)
        if curves.shape != expected or not np.all(np.isfinite(curves)):
            raise ValueError(f"curves must be finite with shape {expected}")
        if progress.shape != (len(self.nominal_curve),) or not np.all(np.isfinite(progress)):
            raise ValueError("progress must be finite with shape (steps,)")
        if phase_codes.shape != progress.shape or not np.issubdtype(phase_codes.dtype, np.integer):
            raise ValueError("phase_codes must be integer-valued with shape (steps,)")

        contexts_metric = to_metric(contexts)
        curves_metric = to_metric(curves)
        centered = contexts_metric - contexts_metric.mean(axis=0, keepdims=True)
        denominator = np.sum(centered * centered)
        if denominator <= 0.0:
            raise ValueError("contexts do not excite the scalar response")
        weights = np.empty(len(progress), dtype=np.float64)
        for step in range(len(progress)):
            response_centered = curves_metric[:, step] - curves_metric[:, step].mean(axis=0)
            weights[step] = np.sum(centered * response_centered) / denominator

        GaussianProcessRegressor, ConstantKernel, RBF, WhiteKernel, _, _ = _sklearn_components()
        kernel = ConstantKernel(self.kernel_amplitude, self.kernel_amplitude_bounds) * RBF(
            length_scale=self.length_scale,
            length_scale_bounds=self.length_scale_bounds,
        ) + WhiteKernel(noise_level=self.noise_level, noise_level_bounds=self.noise_level_bounds)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self.optimizer_restarts,
            random_state=self.random_state,
        )
        self.gp.fit(progress[:, None], weights)
        self.weight = self.gp.predict(progress[:, None])
        self.progress = progress.copy()
        self.phase_codes = phase_codes.astype(np.int64, copy=True)
        return self

    def action(self, context, weight):
        context = _contexts(np.asarray(context, dtype=np.float64).reshape(1, -1))[0]
        relation = socket_frame(context, self.nominal_frame_xy, self.nominal_frame_yaw)
        twist = se2_log(self.nominal_frame_inverse @ relation)
        return self.nominal_frame @ se2_exp(float(weight) * twist) @ self.nominal_frame_inverse

    def actions(self, contexts):
        contexts = _contexts(contexts)
        return np.asarray([[self.action(context, weight) for weight in self.weight] for context in contexts])

    def predict(self, contexts):
        return apply_actions(self.actions(contexts), self.nominal_curve)

    def generator_profile(self):
        return np.repeat(self.weight[:, None], 3, axis=1)


class PDiagActionModel(FrameActionModel):
    """Adapter for the frozen finite diagonal identifier on the common nominal."""

    name = "G-TSR Pdiag finite"

    def __init__(self, nominal_curve, nominal_frame_xy, nominal_frame_yaw=0.0, *, identifier=None):
        super().__init__(nominal_curve, nominal_frame_xy, nominal_frame_yaw)
        self.identifier = ProfileIdentifier() if identifier is None else identifier

    def fit(self, contexts, curves, progress, phase_codes):
        self.profile = self.identifier.fit(
            contexts,
            curves,
            progress,
            phase_codes,
            nominal_frame_xy=self.nominal_frame_xy,
            nominal_frame_yaw=self.nominal_frame_yaw,
        )
        self.diagonal = self.identifier.jacobian_diag()
        self.fit_status = "ok" if self.profile.optimization["success"] else "fit_failed"
        return self

    def actions(self, contexts):
        contexts = _contexts(contexts)
        return np.asarray([[self.identifier.action(context, alpha) for alpha in self.diagonal] for context in contexts])

    def predict(self, contexts):
        return apply_actions(self.actions(contexts), self.nominal_curve)

    def generator_profile(self):
        return self.diagonal.copy()


class FullOperatorModel:
    """Frozen sister dense affine 3x3 direct-update comparator (not a response law)."""

    name = "Full Operator"

    def __init__(self, nominal_curve, ridge=1e-8):
        self.nominal_curve = _curve(nominal_curve, "nominal_curve").copy()
        self.ridge = float(ridge)
        if not np.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")

    def fit(self, contexts, curves, progress, phase_codes):
        contexts = _contexts(contexts)
        curves = np.asarray(curves, dtype=np.float64)
        if curves.ndim != 3 or curves.shape != (len(contexts), len(self.nominal_curve), 3):
            raise ValueError("curves must have shape (n, steps, 3) matching nominal_curve")
        if not np.all(np.isfinite(curves)):
            raise ValueError("curves must be finite")
        contexts_metric = to_metric(contexts)
        curves_metric = to_metric(curves)
        design = np.column_stack([np.ones(len(contexts_metric)), contexts_metric])
        regularizer = self.ridge * np.eye(design.shape[1])
        regularizer[0, 0] = 0.0
        inverse = np.linalg.solve(design.T @ design + regularizer, design.T)
        self.intercept_metric = np.empty((len(self.nominal_curve), 3), dtype=np.float64)
        self.operator_metric = np.empty((len(self.nominal_curve), 3, 3), dtype=np.float64)
        for step in range(len(self.nominal_curve)):
            coefficients = inverse @ curves_metric[:, step]
            self.intercept_metric[step] = coefficients[0]
            self.operator_metric[step] = coefficients[1:]
        return self

    def predict(self, contexts):
        contexts_metric = _contexts(contexts)
        contexts_metric = to_metric(contexts_metric)
        prediction = np.empty((len(contexts_metric), len(self.nominal_curve), 3), dtype=np.float64)
        for step in range(len(self.nominal_curve)):
            prediction[:, step] = self.intercept_metric[step] + contexts_metric @ self.operator_metric[step]
        return from_metric(prediction)

    def actions(self, contexts):
        return nominal_relative_actions(self.predict(contexts), self.nominal_curve)


class GenericConditionalRBFModel:
    """Frozen phase-local RBF direct predictor with episode-grouped ridge CV."""

    name = "Context Regression"
    default_centers_per_phase = 10
    default_width = 0.16
    ridge_alphas = np.logspace(-10, 0, 16)

    def __init__(self, nominal_curve, centers_per_phase=default_centers_per_phase, width=default_width):
        self.nominal_curve = _curve(nominal_curve, "nominal_curve").copy()
        self.centers_per_phase = int(centers_per_phase)
        self.width = float(width)
        if self.centers_per_phase <= 0 or not np.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("centers_per_phase and width must be positive")
        self.context_scale_metric = np.array([0.012, 0.012, 0.03 * np.deg2rad(30.0)])

    def _basis(self, progress, phase_codes):
        progress = np.asarray(progress, dtype=np.float64)
        phase_codes = np.asarray(phase_codes)
        if progress.shape != phase_codes.shape or not np.issubdtype(phase_codes.dtype, np.integer):
            raise ValueError("phase_codes must be integer-valued and match progress")
        registered = np.unique(phase_codes)
        if len(registered) != 4:
            raise ValueError("Generic RBF requires exactly four registered phases")
        phase_to_q = {int(code): q for q, code in enumerate(registered)}
        phase_indices = np.asarray([phase_to_q[int(code)] for code in phase_codes])
        local_progress = progress * 4.0 - phase_indices
        centers = np.linspace(0.0, 1.0, self.centers_per_phase)
        local_basis = np.exp(-0.5 * ((local_progress[:, None] - centers[None, :]) / self.width) ** 2)
        local_basis /= local_basis.sum(axis=1, keepdims=True)
        basis = np.zeros((len(progress), 4 * self.centers_per_phase), dtype=np.float64)
        for row, phase_index in enumerate(phase_indices):
            start = phase_index * self.centers_per_phase
            basis[row, start : start + self.centers_per_phase] = local_basis[row]
        return basis

    def _design(self, contexts, basis):
        contexts = _contexts(contexts)
        contexts_normalized = to_metric(contexts) / self.context_scale_metric
        repeated_basis = np.tile(basis, (len(contexts), 1))
        repeated_context = np.repeat(contexts_normalized, len(basis), axis=0)
        interactions = [repeated_basis * repeated_context[:, generator, None] for generator in range(3)]
        return np.column_stack([repeated_basis, *interactions])

    def fit(self, contexts, curves, progress, phase_codes):
        contexts = _contexts(contexts)
        curves = np.asarray(curves, dtype=np.float64)
        self.progress = np.asarray(progress, dtype=np.float64)
        if curves.shape != (len(contexts), len(self.progress), 3) or not np.all(np.isfinite(curves)):
            raise ValueError("curves must be finite with shape (n, steps, 3)")
        self.phase_codes = np.asarray(phase_codes)
        self.basis = self._basis(self.progress, self.phase_codes)
        design = self._design(contexts, self.basis)
        outputs = to_metric(curves).reshape(-1, 3)
        groups = np.repeat(np.arange(len(contexts)), len(self.progress))
        _, _, _, _, RidgeCV, GroupKFold = _sklearn_components()
        if len(contexts) < 5:
            raise ValueError("Generic RBF grouped CV requires at least five training episodes")
        self.cv_groups = groups.copy()
        grouped_cv = list(GroupKFold(n_splits=5).split(design, outputs, groups))
        self.model = RidgeCV(
            alphas=self.ridge_alphas,
            fit_intercept=False,
            cv=grouped_cv,
        ).fit(design, outputs)
        self.alpha = float(self.model.alpha_)
        return self

    def predict(self, contexts):
        contexts = _contexts(contexts)
        prediction = self.model.predict(self._design(contexts, self.basis)).reshape(
            len(contexts), len(self.progress), 3
        )
        return from_metric(prediction)

    def actions(self, contexts):
        return nominal_relative_actions(self.predict(contexts), self.nominal_curve)
