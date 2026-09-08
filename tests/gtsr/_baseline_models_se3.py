# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""SE(3) baseline models (Frame-weighted / Full operator / Pdiag pointwise).

Vendored from ``maniskill_spectrum``'s ``phase_switch_se3_baselines.py`` at commit
``e5ce8a3`` (blob ``35cf06a9``), minus the Pdiag-finite model (which is
:class:`tsr.gtsr.profile6.ProfileIdentifier6D`). These drive the F-05 negative
control and F-03's pointwise initialization reference.
"""

from __future__ import annotations

import numpy as np

from tsr.gtsr.profile6 import SE3_DIM, se3_from_metric, se3_to_metric


class SE3DiagonalOperatorModel:
    name = "Pdiag pointwise (SE(3))"

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        self.diagonal = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        for step in range(n_steps):
            for generator in range(SE3_DIM):
                design = np.column_stack([np.ones(len(contexts_metric)), contexts_metric[:, generator]])
                coefficients = np.linalg.lstsq(design, curves_metric[:, step, generator], rcond=None)[0]
                self.intercept_metric[step, generator] = coefficients[0]
                self.diagonal[step, generator] = coefficients[1]
        return self

    def predict(self, contexts):
        contexts_metric = se3_to_metric(contexts)
        prediction = self.intercept_metric[None, :, :] + self.diagonal[None, :, :] * contexts_metric[:, None, :]
        return se3_from_metric(prediction)

    def jacobian_diag(self):
        return self.diagonal.copy()


class SE3FrameWeightedModel:
    name = "Frame-weighted (SE(3))"

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        self.weight = np.empty(n_steps, dtype=np.float64)
        for step in range(n_steps):
            c_centered = contexts_metric - contexts_metric.mean(axis=0, keepdims=True)
            y = curves_metric[:, step]
            y_centered = y - y.mean(axis=0, keepdims=True)
            denominator = np.sum(c_centered * c_centered)
            self.weight[step] = np.sum(c_centered * y_centered) / denominator
            self.intercept_metric[step] = y.mean(axis=0) - self.weight[step] * contexts_metric.mean(axis=0)
        return self

    def predict(self, contexts):
        contexts_metric = se3_to_metric(contexts)
        prediction = self.intercept_metric[None, :, :] + self.weight[None, :, None] * contexts_metric[:, None, :]
        return se3_from_metric(prediction)

    def jacobian_diag(self):
        return np.repeat(self.weight[:, None], SE3_DIM, axis=1)


class SE3FullOperatorModel:
    name = "Full operator (SE(3))"

    def __init__(self, ridge=1e-8):
        self.ridge = ridge

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        design = np.column_stack([np.ones(len(contexts_metric)), contexts_metric])
        regularizer = self.ridge * np.eye(design.shape[1])
        regularizer[0, 0] = 0.0
        inverse = np.linalg.inv(design.T @ design + regularizer) @ design.T
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        self.operator_metric = np.empty((n_steps, SE3_DIM, SE3_DIM), dtype=np.float64)
        for step in range(n_steps):
            coefficients = inverse @ curves_metric[:, step]
            self.intercept_metric[step] = coefficients[0]
            self.operator_metric[step] = coefficients[1:]
        return self

    def predict(self, contexts):
        contexts_metric = se3_to_metric(contexts)
        prediction = np.empty((len(contexts_metric), len(self.intercept_metric), SE3_DIM), dtype=np.float64)
        for step in range(len(self.intercept_metric)):
            prediction[:, step] = self.intercept_metric[step] + contexts_metric @ self.operator_metric[step]
        return se3_from_metric(prediction)

    def jacobian_diag(self):
        return np.diagonal(self.operator_metric, axis1=1, axis2=2).copy()
