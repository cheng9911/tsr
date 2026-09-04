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

from tsr.gtsr.profile import matrix_pose2, pose2_matrix, se2_exp, se2_log, socket_frame


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
