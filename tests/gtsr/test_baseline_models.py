# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Unit contracts for E-09 test-side baseline adapters."""

import numpy as np
import pytest

from tests.gtsr._baseline_models import (
    FixedActionModel,
    FrameActionModel,
    FullOperatorModel,
    apply_actions,
    nominal_relative_actions,
)
from tsr.gtsr import ResponseOperator
from tsr.gtsr.profile import planar_to_se3_alpha, pose2_matrix, se2_to_se3, socket_frame


def _nominal_curve():
    return np.array([[0.0, 0.0, 0.0], [0.1, -0.05, 0.2], [0.2, 0.1, -0.1]])


def test_B01_fixed_returns_the_shared_nominal_curve():
    nominal = _nominal_curve()
    contexts = np.array([[0.01, -0.02, 0.3], [-0.01, 0.02, -0.3]])
    model = FixedActionModel(nominal).fit(contexts, np.repeat(nominal[None], 2, axis=0), None, None)
    np.testing.assert_allclose(model.predict(contexts), np.repeat(nominal[None], 2, axis=0))
    np.testing.assert_allclose(model.actions(contexts), np.tile(np.eye(3), (2, len(nominal), 1, 1)))


def test_B02_frame_identity_propagation_matches_response_operator():
    nominal = _nominal_curve()
    frame_xy = np.array([0.03, -0.04])
    frame_yaw = 0.3
    context = np.array([0.01, -0.02, 0.25])
    model = FrameActionModel(nominal, frame_xy, frame_yaw)

    reference = se2_to_se3(pose2_matrix([*frame_xy, frame_yaw]))
    relation = se2_to_se3(socket_frame(context, frame_xy, frame_yaw))
    expected = ResponseOperator(reference, np.eye(6)).action(relation, planar_to_se3_alpha(np.ones(3)))
    np.testing.assert_allclose(se2_to_se3(model.action(context)), expected, atol=1e-12)


def test_B03_nominal_relative_adapter_reconstructs_direct_prediction():
    nominal = _nominal_curve()
    actions = np.tile(pose2_matrix([0.02, -0.01, 0.1]), (2, len(nominal), 1, 1))
    prediction = apply_actions(actions, nominal)
    recovered = nominal_relative_actions(prediction, nominal)
    np.testing.assert_allclose(recovered, actions, atol=1e-12)


def test_B04_baseline_adapter_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        FixedActionModel(np.zeros((3, 2)))
    with pytest.raises(ValueError):
        nominal_relative_actions(np.zeros((2, 3, 3)), _nominal_curve()[:2])


def test_B05_full_operator_recovers_a_planted_dense_direct_update():
    nominal = _nominal_curve()
    contexts = np.array([[-0.01, 0.02, -0.2], [0.01, -0.02, 0.2], [0.015, 0.01, -0.1], [-0.015, -0.01, 0.1]])
    transform = np.array([[1.0, 0.3, 0.1], [-0.2, 0.8, 0.2], [0.1, 0.1, 0.7]])
    offset = np.array([0.02, -0.01, 0.1])
    curves = np.empty((len(contexts), len(nominal), 3))
    for step in range(len(nominal)):
        curves[:, step] = offset + contexts @ transform
    model = FullOperatorModel(nominal, ridge=0.0).fit(contexts, curves, None, None)
    np.testing.assert_allclose(model.predict(contexts), curves, atol=1e-12)
    np.testing.assert_allclose(apply_actions(model.actions(contexts), nominal), curves, atol=1e-12)
