# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Unit contracts for E-09 test-side baseline adapters."""

import numpy as np
import pytest

from tests.gtsr._baseline_models import (
    FixedActionModel,
    FrameActionModel,
    FullOperatorModel,
    GenericConditionalRBFModel,
    PDiagActionModel,
    PhaseScalarActionModel,
    apply_actions,
    nominal_relative_actions,
)
from tsr.gtsr import ProfileIdentifier, ResponseOperator
from tsr.gtsr.profile import planar_to_se3_alpha, pose2_matrix, se2_exp, se2_log, se2_to_se3, socket_frame


def _nominal_curve():
    return np.array([[0.0, 0.0, 0.0], [0.1, -0.05, 0.2], [0.2, 0.1, -0.1]])


def _structured_fixture(steps=8):
    nominal = np.column_stack(
        [
            np.linspace(0.0, 0.12, steps),
            np.linspace(0.0, -0.04, steps),
            np.linspace(0.0, 0.2, steps),
        ]
    )
    contexts = np.array(
        [
            [-0.015, -0.010, -0.30],
            [-0.010, 0.012, 0.25],
            [0.012, -0.015, 0.20],
            [0.015, 0.010, -0.25],
            [-0.012, 0.015, 0.15],
            [0.010, -0.012, -0.15],
        ]
    )
    progress = np.concatenate([(q + np.linspace(0.0, 1.0, steps // 4)) / 4 for q in range(4)])
    phase_codes = np.repeat([3, 4, 5, 6], steps // 4)
    return nominal, contexts, progress, phase_codes


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


def test_B06_phase_scalar_has_one_shared_coefficient_and_frozen_left_action():
    pytest.importorskip("sklearn")
    nominal, contexts, progress, phase_codes = _structured_fixture()
    frame_xy = np.array([0.03, -0.02])
    generator = FrameActionModel(nominal, frame_xy, 0.1)
    curves = generator.predict(0.55 * contexts)
    model = PhaseScalarActionModel(nominal, frame_xy, 0.1).fit(contexts, curves, progress, phase_codes)

    profile = model.generator_profile()
    assert model.gp.n_restarts_optimizer == 5
    assert model.gp.random_state == 20260820
    assert model.length_scale == 0.12
    assert model.length_scale_bounds == (0.015, 1.0)
    assert model.noise_level == 1e-4
    assert model.noise_level_bounds == (1e-8, 1e-1)
    np.testing.assert_allclose(profile[:, 0], profile[:, 1])
    np.testing.assert_allclose(profile[:, 1], profile[:, 2])
    np.testing.assert_allclose(model.predict(np.zeros((1, 3)))[0], nominal, atol=1e-12)

    step = 3
    reference = se2_to_se3(pose2_matrix([*frame_xy, 0.1]))
    relation = se2_to_se3(socket_frame(contexts[0], frame_xy, 0.1))
    expected = ResponseOperator(reference, np.eye(6)).action(
        relation,
        planar_to_se3_alpha(np.repeat(profile[step, 0], 3)),
    )
    np.testing.assert_allclose(se2_to_se3(model.actions(contexts[:1])[0, step]), expected, atol=1e-9)


def test_B07_pdiag_adapter_has_three_coefficients_and_common_nominal():
    nominal, contexts, progress, phase_codes = _structured_fixture()
    frame_xy = np.array([0.03, -0.02])
    source = FrameActionModel(nominal, frame_xy, 0.1)
    scales = np.array([0.8, 0.4, 0.2])
    actions = np.asarray(
        [
            [
                source.nominal_frame
                @ se2_exp(scales * se2_log(source.nominal_frame_inverse @ socket_frame(context, frame_xy, 0.1)))
                @ source.nominal_frame_inverse
                for _ in progress
            ]
            for context in contexts
        ]
    )
    curves = apply_actions(actions, nominal)
    identifier = ProfileIdentifier(n_basis=4, nominal_iterations=1)
    model = PDiagActionModel(nominal, frame_xy, 0.1, identifier=identifier).fit(contexts, curves, progress, phase_codes)

    assert model.generator_profile().shape == (len(progress), 3)
    np.testing.assert_allclose(model.predict(np.zeros((1, 3)))[0], nominal, atol=1e-12)
    np.testing.assert_allclose(
        apply_actions(model.actions(contexts[:1]), nominal), model.predict(contexts[:1]), atol=1e-12
    )


def test_B08_generic_rbf_uses_episode_groups_and_reconstructs_direct_prediction():
    pytest.importorskip("sklearn")
    nominal, contexts, progress, phase_codes = _structured_fixture()
    transform = np.array([[0.8, 0.1, 0.2], [-0.2, 0.6, 0.1], [0.1, -0.1, 0.5]])
    curves = np.asarray([nominal + (context @ transform)[None, :] * (0.5 + progress[:, None]) for context in contexts])
    model = GenericConditionalRBFModel(nominal, centers_per_phase=3).fit(contexts, curves, progress, phase_codes)

    np.testing.assert_array_equal(model.cv_groups, np.repeat(np.arange(len(contexts)), len(progress)))
    np.testing.assert_allclose(model.ridge_alphas, np.logspace(-10, 0, 16))
    assert model.alpha in model.ridge_alphas
    prediction = model.predict(contexts[:2])
    np.testing.assert_allclose(apply_actions(model.actions(contexts[:2]), nominal), prediction, atol=1e-12)
    assert prediction.shape == (2, len(progress), 3)


def test_B09_generic_rbf_maps_registered_phase_codes_to_four_local_blocks():
    pytest.importorskip("sklearn")
    nominal, _, _, _ = _structured_fixture()
    rho = np.array([0.0, 0.5, 1.0])
    progress = np.concatenate([(q + rho) / 4 for q in range(4)])
    phase_codes = np.repeat([3, 4, 5, 6], len(rho))
    model = GenericConditionalRBFModel(nominal, centers_per_phase=2)
    basis = model._basis(progress, phase_codes)
    assert model.default_centers_per_phase == 10
    assert model.default_width == 0.16
    np.testing.assert_allclose(model.context_scale_metric, [0.012, 0.012, 0.03 * np.deg2rad(30.0)])
    assert basis.shape == (12, 8)
    for row, phase in enumerate(phase_codes):
        block = (phase - 3) * 2
        assert np.count_nonzero(basis[row, block : block + 2]) == 2
        assert np.count_nonzero(basis[row]) == 2
    tail = np.exp(-0.5 * (1.0 / model.width) ** 2)
    expected_local = np.array([[1.0, tail], [0.5 * (1.0 + tail), 0.5 * (1.0 + tail)], [tail, 1.0]])
    expected_local /= 1.0 + tail
    for phase_index in range(4):
        block = phase_index * 2
        np.testing.assert_allclose(basis[phase_index * 3 : (phase_index + 1) * 3, block : block + 2], expected_local)
