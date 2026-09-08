# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Unit tests for the 6D SE(3)/pose6 math primitives (Sprint F, Task 1)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.gtsr._baseline_models_se3 import (
    SE3DiagonalOperatorModel,
    SE3FrameWeightedModel,
    SE3FullOperatorModel,
)
from tsr.gtsr.profile import CellStatus
from tsr.gtsr.profile6 import (
    SE3_METRIC_SCALE,
    ProfileIdentifier6D,
    ResponseProfile6D,
    SelectivityMetrics6D,
    SelectivityStatus,
    SelectivityVerdict6D,
    euler_from_matrix,
    matrix_from_euler,
    matrix_from_euler_batched,
    pose6_from_se3,
    pose6_from_se3_batched,
    se3_exp,
    se3_exp_batched,
    se3_from_pose6,
    se3_from_pose6_batched,
    se3_inverse,
    se3_inverse_batched,
    se3_log,
    socket_frame6,
)
from tsr.gtsr.response import ResponseOperator


def test_metric_scale_frozen():
    np.testing.assert_array_equal(
        SE3_METRIC_SCALE, np.array([1.0, 1.0, 1.0, 0.03, 0.03, 0.03])
    )


def test_so3_exp_log_roundtrip():
    from tsr.gtsr.profile6 import so3_exp, so3_log

    rng = np.random.default_rng(0)
    for _ in range(10):
        w = rng.uniform(-1.0, 1.0, size=3)
        R = so3_exp(w)
        np.testing.assert_allclose(so3_log(R), w, atol=1e-10)


def test_se3_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(10):
        xi = rng.uniform(-0.5, 0.5, size=6)
        T = se3_exp(xi)
        np.testing.assert_allclose(se3_log(T), xi, atol=1e-9)


def test_se3_exp_batched_matches_scalar():
    rng = np.random.default_rng(2)
    twists = rng.uniform(-0.5, 0.5, size=(5, 6))
    batched = se3_exp_batched(twists)
    for i in range(5):
        np.testing.assert_allclose(batched[i], se3_exp(twists[i]), atol=1e-12)


def test_se3_inverse_batched_matches_scalar():
    rng = np.random.default_rng(3)
    twists = rng.uniform(-0.5, 0.5, size=(4, 6))
    T = se3_exp_batched(twists)
    batched = se3_inverse_batched(T)
    for i in range(4):
        np.testing.assert_allclose(batched[i], se3_inverse(T[i]), atol=1e-12)
        np.testing.assert_allclose(batched[i] @ T[i], np.eye(4), atol=1e-12)


def test_pose6_euler_convention_yaw_only():
    # A pure yaw rotation must be Rx(0) Ry(0) Rz(yaw) = Rz(yaw).
    yaw = 0.3
    R = matrix_from_euler(0.0, 0.0, yaw)
    np.testing.assert_allclose(R, np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0],
         [np.sin(yaw), np.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]]
    ), atol=1e-12)


def test_pose6_euler_roundtrip():
    rng = np.random.default_rng(4)
    for _ in range(10):
        rpy = rng.uniform(-0.5, 0.5, size=3)
        np.testing.assert_allclose(
            euler_from_matrix(matrix_from_euler(*rpy)), rpy, atol=1e-10
        )


def test_pose6_se3_roundtrip():
    rng = np.random.default_rng(5)
    for _ in range(10):
        pose6 = rng.uniform(-0.5, 0.5, size=6)
        np.testing.assert_allclose(
            pose6_from_se3(se3_from_pose6(pose6)), pose6, atol=1e-10
        )


def test_pose6_se3_batched_roundtrip():
    rng = np.random.default_rng(6)
    pose6 = rng.uniform(-0.5, 0.5, size=(7, 6))
    np.testing.assert_allclose(
        pose6_from_se3_batched(se3_from_pose6_batched(pose6)), pose6, atol=1e-10
    )


def test_matrix_from_euler_batched_matches_scalar():
    rng = np.random.default_rng(7)
    rpy = rng.uniform(-0.5, 0.5, size=(6, 3))
    batched = matrix_from_euler_batched(rpy)
    for i in range(6):
        np.testing.assert_allclose(
            batched[i], matrix_from_euler(*rpy[i]), atol=1e-12
        )


def test_socket_frame6():
    nominal = se3_from_pose6(np.array([0.1, 0.2, 0.3, 0.1, -0.1, 0.2]))
    context = np.array([0.01, -0.02, 0.03, 0.05, -0.05, 0.1])
    expected = nominal @ se3_from_pose6(context)
    np.testing.assert_allclose(socket_frame6(context, nominal), expected, atol=1e-12)


def test_F02_action_matches_response_operator():
    """ProfileIdentifier6D.action == ResponseOperator(reference, I).action(socket_frame6)."""
    nominal_pose6 = np.array([0.1, -0.2, 0.15, 0.05, -0.04, 0.3])
    nominal_frame = se3_from_pose6(nominal_pose6)
    identifier = ProfileIdentifier6D()
    identifier.nominal_frame = nominal_frame
    identifier.nominal_frame_inverse = np.linalg.inv(nominal_frame)

    operator = ResponseOperator(nominal_frame, np.eye(6))
    contexts = np.array(
        [
            [0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.015, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.261799, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.261799, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.523599],
            [0.012, -0.012, 0.012, 0.2, -0.2, 0.4],
        ]
    )
    alphas = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.5, 0.7, 0.9, 1.1, 1.25],
            [1.25, 0.2, 0.4, 0.6, 0.8, 0.1],
        ]
    )
    for context in contexts:
        relation = socket_frame6(context, nominal_frame)
        for alpha in alphas:
            action_identifier = identifier.action(context, alpha)
            action_operator = operator.action(relation, alpha)
            assert np.linalg.norm(action_identifier - action_operator, ord="fro") < 1e-9


def _planted_recovery_data():
    """Synthetic 60-mixed contexts/curves with a planted 6D profile (spec F-03)."""
    n_steps = 100
    scale = np.array([0.012, 0.012, 0.012, np.deg2rad(15), np.deg2rad(15), np.deg2rad(30)])
    rng = np.random.default_rng(20260908)
    contexts = rng.uniform(-1.0, 1.0, size=(60, 6)) * scale
    assert np.linalg.matrix_rank(contexts / scale) == 6
    assert (np.min(contexts, axis=0) < 0).all() and (np.max(contexts, axis=0) > 0).all()

    progress = np.concatenate([(i + np.linspace(0.0, 1.0, 25)) / 4 for i in range(4)])
    phase_codes = np.repeat(np.arange(4), 25)

    nominal_frame = np.eye(4)
    # Nominal curve: peg descends along -z from 0 to -0.05, fixed orientation.
    nominal_curve = np.zeros((n_steps, 6))
    nominal_curve[:, 2] = -0.05 * (1.0 - progress)

    def alpha_star(g):
        return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.1 + 0.9 * (1.0 - g)])

    operator = ResponseOperator(nominal_frame, np.eye(6))
    curves = np.empty((60, n_steps, 6))
    for c, context in enumerate(contexts):
        relation = socket_frame6(context, nominal_frame)
        for s in range(n_steps):
            action = operator.action(relation, alpha_star(progress[s]))
            curves[c, s] = pose6_from_se3(action @ se3_from_pose6(nominal_curve[s]))
    return contexts, curves, progress, phase_codes, alpha_star, nominal_curve


def test_F03_planted_recovery():
    contexts, curves, progress, phase_codes, alpha_star, _ = _planted_recovery_data()
    identifier = ProfileIdentifier6D()
    profile = identifier.fit(
        contexts, curves, progress, phase_codes, nominal_frame_pose=np.zeros(6)
    )
    assert isinstance(profile, ResponseProfile6D)
    assert profile.optimization["success"]
    assert identifier.jacobian_diag().shape == (100, 6)
    assert profile.theta.shape == (6, 24)
    for q in range(4):
        for rho in (0.0, 0.5, 1.0):
            g = (q + rho) / 4
            recovered = identifier.jacobian_diag()[q * 25 + int(round(rho * 24))]
            target = alpha_star(g)
            assert np.max(np.abs(recovered - target)) <= 5e-3


def test_baselines_jacobian_diag_shapes_and_frame_weighted_shared_weight():
    contexts, curves, progress, phase_codes, _, _ = _planted_recovery_data()
    for model in (
        SE3DiagonalOperatorModel(),
        SE3FrameWeightedModel(),
        SE3FullOperatorModel(),
    ):
        model.fit(contexts, curves, progress, phase_codes)
        assert model.jacobian_diag().shape == (100, 6)
    # Frame-weighted collapses every generator to one shared scalar w(s).
    fw = SE3FrameWeightedModel().fit(contexts, curves, progress, phase_codes)
    diagonal = fw.jacobian_diag()
    for generator in range(6):
        np.testing.assert_allclose(diagonal[:, generator], fw.weight, atol=1e-12)


def _synthetic_cell(phase_codes, yaw_pre=0.9, yaw_post=0.3, other_pre=0.95, other_post=0.95):
    profile = np.empty((len(phase_codes), 6))
    profile[:, 5] = np.where(phase_codes < 2, yaw_pre, yaw_post)
    profile[:, :5] = np.where(phase_codes < 2, other_pre, other_post)[:, None]
    return profile


def test_selectivity_metrics_per_cell_and_cross_arm():
    phase_codes = np.repeat(np.arange(4), 10)
    metrics = SelectivityMetrics6D(phase_codes)
    keyed = metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.9, yaw_post=0.3))
    circular = metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.1, yaw_post=0.1))
    assert keyed.delta_yaw == pytest.approx(0.6)
    assert keyed.delta_others_max == pytest.approx(0.0)
    assert keyed.rank1_yaw is True
    assert keyed.transition is True

    S, purity = metrics.cross_arm(keyed, circular)
    assert S[5] == pytest.approx(0.8)  # |0.9 - 0.1|
    assert purity == pytest.approx(S[5] / S.sum())
    assert S[0] == pytest.approx(0.0)


def test_selectivity_metrics_aggregate_matches_frozen_semantics():
    phase_codes = np.repeat(np.arange(4), 25)
    metrics = SelectivityMetrics6D(phase_codes)
    keyed_cells = []
    circular_cells = []
    for r in range(18):
        # Fixed 0.8 yaw gap; the five "other" generators get a per-cell
        # varying gap so mean(per-cell ratio) != ratio of means.
        keyed_cells.append(
            metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.9, yaw_post=0.3,
                                             other_pre=0.5 + 0.02 * r, other_post=0.5 + 0.02 * r))
        )
        circular_cells.append(
            metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.1, yaw_post=0.1,
                                             other_pre=0.3, other_post=0.3))
        )
    headline = metrics.aggregate(keyed_cells, circular_cells)

    S_yaw = 0.8
    # S_other,r = |(0.5+0.02r) - 0.3| = 0.2 + 0.02r for each of 5 generators.
    purity_expected = float(np.mean([S_yaw / (S_yaw + 5 * (0.2 + 0.02 * r)) for r in range(18)]))
    ratio_of_means = float((18 * S_yaw) / sum(S_yaw + 5 * (0.2 + 0.02 * r) for r in range(18)))
    assert headline["S_yaw"] == pytest.approx(S_yaw)
    assert headline["S_others_max"] == pytest.approx(float(np.mean([0.2 + 0.02 * r for r in range(18)])))
    assert headline["purity"] == pytest.approx(purity_expected)
    assert headline["rank"] == pytest.approx(1.0)
    assert headline["transition"] == pytest.approx(1.0)
    # The forbidden ratio-of-means convention must NOT be what aggregate computed.
    assert abs(headline["purity"] - ratio_of_means) > 1e-6


def test_selectivity_verdict_two_layer():
    phase_codes = np.repeat(np.arange(4), 25)
    metrics = SelectivityMetrics6D(phase_codes)
    keyed_cells = [metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.9, yaw_post=0.3)) for _ in range(18)]
    circular_cells = [metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.1, yaw_post=0.1)) for _ in range(18)]
    headline = metrics.aggregate(keyed_cells, circular_cells)

    contexts = np.random.default_rng(0).uniform(-0.012, 0.012, size=(60, 6))
    verdict = SelectivityVerdict6D(headline)
    identification = verdict.identify(
        contexts, observable=np.ones(6, dtype=bool), fit_succeeded=True, meets_gate=np.ones(6, dtype=bool)
    )
    selectivity = verdict.selectivity(identification, fit_succeeded=np.ones(6, dtype=bool))
    assert selectivity[5] is SelectivityStatus.SELECTIVE
    assert all(s is SelectivityStatus.NON_SELECTIVE for s in selectivity[:5])


def test_selectivity_verdict_abstain_on_data_insufficient():
    phase_codes = np.repeat(np.arange(4), 25)
    metrics = SelectivityMetrics6D(phase_codes)
    keyed_cells = [metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.9, yaw_post=0.3)) for _ in range(18)]
    circular_cells = [metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.1, yaw_post=0.1)) for _ in range(18)]
    headline = metrics.aggregate(keyed_cells, circular_cells)
    verdict = SelectivityVerdict6D(headline)
    identification = [CellStatus.UNOBSERVABLE] * 6
    selectivity = verdict.selectivity(identification, fit_succeeded=np.ones(6, dtype=bool))
    assert all(status is SelectivityStatus.ABSTAIN for status in selectivity)
