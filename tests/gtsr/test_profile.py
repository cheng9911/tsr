# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""P-01 ~ P-07: ResponseProfile / ProfileIdentifier unit tests (Sprint E)."""

import numpy as np
import pytest

from tsr.gtsr import ResponseOperator
from tsr.gtsr.profile import (
    CellStatus,
    ProfileIdentifier,
    ResponseProfile,
    adjudicate_cell,
    cell_statuses,
    excitation_mask,
    matrix_pose2,
    planar_to_se3_alpha,
    planar_to_se3_twist,
    pose2_matrix,
    rbf_basis,
    se2_exp,
    se2_log,
    se2_to_se3,
    socket_frame,
)


def _progress_grid(bins):
    phase_progress = np.linspace(0.0, 1.0, bins)
    progress = np.concatenate([(i + phase_progress) / 4 for i in range(4)])
    phase_codes = np.repeat(np.arange(3, 7), bins)
    return progress, phase_codes


def _nominal_curve(steps):
    s = np.linspace(0.0, 1.0, steps)
    return np.column_stack([0.1 * s, 0.05 * np.sin(np.pi * s), 0.2 * s])


def _full_rank_contexts(n, rng):
    return np.column_stack(
        [
            rng.uniform(-0.015, 0.015, n),
            rng.uniform(-0.015, 0.015, n),
            rng.uniform(-0.3, 0.3, n),
        ]
    )


def _response_curves(nominal_curve, contexts, alpha_true, nominal_xy=(0.0, 0.0), nominal_yaw=0.0):
    c0 = pose2_matrix([*nominal_xy, nominal_yaw])
    c0_inv = np.linalg.inv(c0)
    curves = []
    for c in contexts:
        twist = se2_log(c0_inv @ socket_frame(c, nominal_xy, nominal_yaw))
        curve = []
        for s in range(len(nominal_curve)):
            action = c0 @ se2_exp(alpha_true[s] * twist) @ c0_inv
            curve.append(matrix_pose2(action @ pose2_matrix(nominal_curve[s])))
        curves.append(curve)
    return np.asarray(curves)


def test_P01_adjudicate_cell_priority():
    assert (
        adjudicate_cell(excited=False, observable=True, fit_succeeded=True, meets_heldout_gate=True)
        is CellStatus.UNEXCITED
    )
    assert (
        adjudicate_cell(excited=True, observable=False, fit_succeeded=True, meets_heldout_gate=True)
        is CellStatus.UNOBSERVABLE
    )
    assert (
        adjudicate_cell(excited=True, observable=True, fit_succeeded=False, meets_heldout_gate=True)
        is CellStatus.FIT_FAILED
    )
    assert (
        adjudicate_cell(excited=True, observable=True, fit_succeeded=True, meets_heldout_gate=False)
        is CellStatus.MODEL_MISMATCH
    )
    assert (
        adjudicate_cell(excited=True, observable=True, fit_succeeded=True, meets_heldout_gate=True)
        is CellStatus.IDENTIFIED
    )


def test_P02_embedding_contract():
    np.testing.assert_allclose(planar_to_se3_alpha([0.4, 0.7, -0.2]), [0.4, 0.7, 1.0, 1.0, 1.0, -0.2])
    np.testing.assert_allclose(planar_to_se3_twist([0.01, -0.02, 0.3]), [0.01, -0.02, 0.0, 0.0, 0.0, 0.3])
    se2 = pose2_matrix([0.1, -0.2, 0.5])
    se3 = se2_to_se3(se2)
    assert se3.shape == (4, 4)
    np.testing.assert_allclose(se3[:2, :2], se2[:2, :2], atol=1e-15)
    np.testing.assert_allclose(se3[:2, 3], se2[:2, 2], atol=1e-15)
    np.testing.assert_allclose(se3[2, 2], 1.0)


def test_P03_response_profile_roundtrip_and_range():
    theta = np.linspace(-1.0, 1.0, 3 * 24).reshape(3, 24)
    profile = ResponseProfile(theta, {3: 0, 4: 1, 5: 2, 6: 3})
    for phase_code in (3, 4, 5, 6):
        a = profile.alpha(phase_code, 0.5)
        assert a.shape == (3,)
        assert np.all(a > 0.0) and np.all(a < profile.alpha_max)
        a6 = profile.alpha6(phase_code, 0.5)
        assert a6.shape == (6,)
        np.testing.assert_allclose(a6[2:5], [1.0, 1.0, 1.0])

    restored = ResponseProfile.from_dict(profile.to_dict())
    np.testing.assert_allclose(restored.theta, profile.theta)
    np.testing.assert_allclose(restored.alpha(3, 0.25), profile.alpha(3, 0.25))
    # theta is read-only copy.
    theta_before = profile.theta.copy()
    profile.theta[0, 0] = 999.0
    np.testing.assert_allclose(profile.theta, theta_before)


def test_P04_rbf_basis_normalized():
    basis = rbf_basis(np.linspace(0.0, 1.0, 25), n_basis=24, basis_width=0.065)
    assert basis.shape == (25, 24)
    np.testing.assert_allclose(basis.sum(axis=1), np.ones(25), atol=1e-12)
    assert np.all(basis >= 0.0)


def test_P05_planted_profile_recovery():
    identifier = ProfileIdentifier()
    progress, phase_codes = _progress_grid(10)  # 40 steps
    nominal_curve = _nominal_curve(40)
    rng = np.random.default_rng(7)
    contexts = _full_rank_contexts(30, rng)

    s = np.linspace(0.0, 1.0, 40)
    alpha_true = np.ones((40, 3))
    alpha_true[:, 2] = 1.25 / (1.0 + np.exp(40.0 * (s - 0.5)))  # yaw: 1.25 -> 0 ramp
    curves = _response_curves(nominal_curve, contexts, alpha_true)

    profile = identifier.fit(
        contexts,
        curves,
        progress,
        phase_codes,
        nominal_frame_xy=(0.0, 0.0),
        nominal_frame_yaw=0.0,
    )
    assert isinstance(profile, ResponseProfile)
    diag = identifier.jacobian_diag()
    # translation stays ~1 everywhere.
    assert np.all(np.abs(diag[:, 0] - 1.0) < 0.15)
    assert np.all(np.abs(diag[:, 1] - 1.0) < 0.15)
    # yaw near 1.25 pre-ramp, near 0 post-ramp.
    assert np.all(np.abs(diag[:12, 2] - 1.25) < 0.25)
    assert np.all(np.abs(diag[28:, 2] - 0.0) < 0.15)


def test_P06_profile_to_operator_differential():
    nominal_xy = (0.05, -0.07)
    nominal_yaw = 0.4
    c0 = pose2_matrix([*nominal_xy, nominal_yaw])
    t_ref = se2_to_se3(c0)
    op = ResponseOperator(t_ref, np.eye(6))

    rng = np.random.default_rng(42)
    for _ in range(20):
        c = np.concatenate([rng.uniform(-0.015, 0.015, 2), rng.uniform(-0.3, 0.3, 1)])
        a3 = np.concatenate([rng.uniform(0.0, 1.25, 2), rng.uniform(0.0, 1.25, 1)])
        twist = se2_log(np.linalg.inv(c0) @ socket_frame(c, nominal_xy, nominal_yaw))
        a2 = c0 @ se2_exp(a3 * twist) @ np.linalg.inv(c0)

        a6 = planar_to_se3_alpha(a3)
        relation = se2_to_se3(socket_frame(c, nominal_xy, nominal_yaw))
        a3_matrix = op.action(relation, a6)

        np.testing.assert_allclose(se2_to_se3(a2), a3_matrix, atol=1e-12)


def test_P07_input_contract():
    identifier = ProfileIdentifier()
    progress, phase_codes = _progress_grid(10)
    contexts = _full_rank_contexts(30, np.random.default_rng(0))
    curves = _response_curves(_nominal_curve(40), contexts, np.ones((40, 3)))

    with pytest.raises(ValueError):
        identifier.fit(contexts[:, :2], curves, progress, phase_codes, nominal_frame_xy=(0.0, 0.0))
    with pytest.raises(ValueError):
        identifier.fit(contexts, curves[:, :, :2], progress, phase_codes, nominal_frame_xy=(0.0, 0.0))
    with pytest.raises(ValueError):
        identifier.fit(contexts, curves, progress[:-1], phase_codes, nominal_frame_xy=(0.0, 0.0))
    with pytest.raises(ValueError):
        identifier.fit(contexts, curves, progress, phase_codes, nominal_frame_xy=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        identifier.fit(contexts, curves, progress, phase_codes)


def test_P08_rank_deficient_and_unexcited_rejected():
    identifier = ProfileIdentifier()
    progress, phase_codes = _progress_grid(10)
    rng = np.random.default_rng(1)
    contexts = _full_rank_contexts(30, rng)
    curves = _response_curves(_nominal_curve(40), contexts, np.ones((40, 3)))

    # Rank-deficient: one generator never varies.
    degenerate = contexts.copy()
    degenerate[:, 2] = 0.0
    with pytest.raises(ValueError):
        identifier.fit(degenerate, curves, progress, phase_codes, nominal_frame_xy=(0.0, 0.0))

    # Full rank but no signed (negative) coverage on one generator.
    one_sided = contexts.copy()
    one_sided[:, 0] = np.abs(one_sided[:, 0])
    with pytest.raises(ValueError):
        identifier.fit(one_sided, curves, progress, phase_codes, nominal_frame_xy=(0.0, 0.0))


def test_P08b_phase_codes_must_match_the_registered_oracle_vocabulary():
    identifier = ProfileIdentifier()
    progress, phase_codes = _progress_grid(10)
    contexts = _full_rank_contexts(30, np.random.default_rng(3))
    curves = _response_curves(_nominal_curve(40), contexts, np.ones((40, 3)))

    with pytest.raises(ValueError, match="registered phase_to_q"):
        identifier.fit(
            contexts,
            curves,
            progress,
            phase_codes - 3,
            nominal_frame_xy=(0.0, 0.0),
        )


def test_P09_excitation_and_cell_statuses_from_data():
    contexts = _full_rank_contexts(30, np.random.default_rng(2))
    assert excitation_mask(contexts).tolist() == [True, True, True]

    # A generator with no negative control is unexcited -> UNEXCITED (abstain).
    one_sided = contexts.copy()
    one_sided[:, 2] = np.abs(one_sided[:, 2])
    assert excitation_mask(one_sided).tolist() == [True, True, False]
    statuses = cell_statuses(one_sided, observable=True, fit_succeeded=True, meets_heldout_gate=True)
    assert statuses[0] is CellStatus.IDENTIFIED
    assert statuses[1] is CellStatus.IDENTIFIED
    assert statuses[2] is CellStatus.UNEXCITED

    # Unobservable (even when excited and gated) -> UNOBSERVABLE, never IDENTIFIED.
    statuses = cell_statuses(contexts, observable=False, fit_succeeded=True, meets_heldout_gate=True)
    assert all(status is CellStatus.UNOBSERVABLE for status in statuses)


def test_P10_response_profile_strictly_read_only():
    profile = ResponseProfile(np.linspace(-1.0, 1.0, 3 * 24).reshape(3, 24), {3: 0, 4: 1, 5: 2, 6: 3})

    # Mutating a returned copy never mutates the profile state.
    phase_to_q = profile.phase_to_q
    phase_to_q[99] = 0
    assert 99 not in profile.phase_to_q

    optimization = profile.optimization
    optimization["tampered"] = True
    assert "tampered" not in profile.optimization

    # Scalar metadata is read-only (no setter).
    with pytest.raises(AttributeError):
        profile.alpha_max = 2.0
    with pytest.raises(AttributeError):
        profile.phase_to_q = {3: 0}

    # Internal storage cannot be mutated or replaced through ordinary Python access.
    with pytest.raises(TypeError):
        profile._theta[0, 0] = 999.0
    with pytest.raises(AttributeError):
        profile._theta.setflags(write=True)
    with pytest.raises(TypeError):
        profile._phase_to_q[99] = 0
    with pytest.raises(AttributeError):
        profile._alpha_max = 2.0
    with pytest.raises(AttributeError):
        profile.new_attribute = "not allowed"
