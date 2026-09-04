# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""R-01 ~ R-08: ResponseOperator oracle-phase finite-action assembly.

These tests verify the Sprint D response-region adapter (``docs/superpowers/specs/
2026-09-04-gtsr-response-law-design.md``). They are pure NumPy and do not load
ManiSkill / HDF5 / learned models.
"""

import numpy as np
import pytest

from tsr.gtsr import GTSR, ResponseOperator
from tsr.gtsr.metrics import se3_exp, se3_log

# Non-unit reference transform and region outer frames (design §4): these rule out
# accidentally applying the task-local action in the world frame.
T_REF = se3_exp(np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.15]))
T0_W = se3_exp(np.array([0.4, 0.1, -0.2, -0.3, 0.2, 0.1]))
TW_E = se3_exp(np.array([-0.2, 0.3, 0.1, 0.1, -0.2, -0.15]))

# A non-zero six-axis relation twist (du, dv, dw, roll, pitch, yaw).
C0 = np.array([0.02, -0.03, 0.04, 0.05, -0.06, 0.07])

I6 = np.eye(6)


def _e(j):
    e = np.zeros(6)
    e[j] = 1.0
    return e


def _relation(c):
    return T_REF @ se3_exp(c)


def test_R01_input_contract():
    # Non-SE(3) transforms are rejected.
    with pytest.raises(ValueError):
        ResponseOperator(np.eye(3), I6)  # wrong shape
    with pytest.raises(ValueError):
        ResponseOperator(np.ones((4, 4)), I6)  # bad homogeneous last row
    bad_rot = np.eye(4)
    bad_rot[0, 1] = 0.5  # non-orthonormal rotation
    with pytest.raises(ValueError):
        ResponseOperator(bad_rot, I6)

    # Non-6x6 / singular basis rejected.
    with pytest.raises(ValueError):
        ResponseOperator(T_REF, np.eye(5))
    with pytest.raises(ValueError):
        ResponseOperator(T_REF, np.zeros((6, 6)))

    op = ResponseOperator(T_REF, I6)
    T = _relation(C0)

    # Relation transform contract.
    with pytest.raises(ValueError):
        op.coordinates(np.eye(3))

    # Wrong-shape / non-finite alpha rejected.
    with pytest.raises(ValueError):
        op.action(T, np.ones(5))
    with pytest.raises(ValueError):
        op.action(T, np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.inf]))

    # condition() requires a GTSR.
    with pytest.raises(TypeError):
        op.condition(object(), T, np.ones(6))


def test_R02_local_coordinate_and_action():
    op = ResponseOperator(T_REF, I6)
    assert op.dim == 6
    T = _relation(C0)

    np.testing.assert_allclose(op.coordinates(T), C0, atol=1e-12)

    alpha = np.array([1.0, 0.7, 0.3, 1.0, -0.5, 0.0])
    expected = T_REF @ se3_exp(alpha * se3_log(np.linalg.inv(T_REF) @ T)) @ np.linalg.inv(T_REF)
    np.testing.assert_allclose(op.action(T, alpha), expected, atol=1e-12)


def test_R03_full_follow_equals_whole_frame():
    op = ResponseOperator(T_REF, I6)
    ones = np.ones(6)
    for j in range(6):
        T = _relation(0.1 * _e(j))
        # Whole-frame response: A = T @ inv(T_ref) (design §3.5, alpha=1, B=I).
        np.testing.assert_allclose(op.action(T, ones), T @ np.linalg.inv(T_REF), atol=1e-12)


def test_R04_yaw_release_suppresses_yaw_only():
    op = ResponseOperator(T_REF, I6)
    alpha_postclear = np.array([1, 1, 1, 1, 1, 0])  # post-clear: yaw released

    # yaw-only intervention -> identity action.
    np.testing.assert_allclose(op.action(_relation(0.1 * _e(5)), alpha_postclear), np.eye(4), atol=1e-12)

    # du / roll / pitch interventions still propagate.
    for j in (0, 3, 4):
        T = _relation(0.1 * _e(j))
        np.testing.assert_allclose(op.action(T, alpha_postclear), T @ np.linalg.inv(T_REF), atol=1e-12)


def test_R05_simultaneous_release_du_and_yaw():
    op = ResponseOperator(T_REF, I6)
    alpha_multigen = np.array([0, 1, 1, 1, 1, 0])  # du (j=0) and yaw (j=5) released

    # du-only, yaw-only, and combined -> identity.
    for c in (_e(0), _e(5), 0.1 * _e(0) + 0.2 * _e(5)):
        np.testing.assert_allclose(op.action(_relation(c), alpha_multigen), np.eye(4), atol=1e-12)

    # dv / dw / roll / pitch still propagate.
    for j in (1, 2, 3, 4):
        T = _relation(0.1 * _e(j))
        np.testing.assert_allclose(op.action(T, alpha_multigen), T @ np.linalg.inv(T_REF), atol=1e-12)


def test_R06_region_assembly():
    op = ResponseOperator(T_REF, I6)
    Bw = np.array(
        [
            [-0.3, 0.3],
            [-0.3, 0.3],
            [-0.3, 0.3],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    region = GTSR.canonical(T0_w=T0_W, Tw_e=TW_E, Bw=Bw)
    T = _relation(C0)
    alpha = np.ones(6)
    A = op.action(T, alpha)
    cond = op.condition(region, T, alpha)

    rng = np.random.default_rng(20260902)
    for _ in range(20):
        z = region.domain.sample(rng)
        np.testing.assert_allclose(cond.to_transform(z), A @ region.to_transform(z), atol=1e-12)


def test_R07_non_mutation_and_domain_separation():
    op = ResponseOperator(T_REF, I6)
    Bw = np.array(
        [
            [-0.3, 0.3],
            [-0.3, 0.3],
            [-0.3, 0.3],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    region = GTSR.canonical(T0_w=T0_W, Tw_e=TW_E, Bw=Bw)
    t0_before = region.T0_w.copy()
    tw_before = region.Tw_e.copy()
    chart_before = region.chart
    domain_before = region.domain

    T = _relation(C0)
    alpha = np.ones(6)
    cond = op.condition(region, T, alpha)

    # Non-mutation: input region's frames/chart/domain are unchanged.
    np.testing.assert_allclose(region.T0_w, t0_before)
    np.testing.assert_allclose(region.Tw_e, tw_before)
    assert region.chart is chart_before
    assert region.domain is domain_before

    # Same alpha, different domain -> same action (same T0_w update), domain preserved.
    Bw2 = Bw.copy()
    Bw2[0] = [0.0, 0.0]
    region2 = GTSR.canonical(T0_w=T0_W, Tw_e=TW_E, Bw=Bw2)
    cond2 = op.condition(region2, T, alpha)
    np.testing.assert_allclose(cond.T0_w, cond2.T0_w, atol=1e-15)
    assert cond2.domain is region2.domain

    # Change alpha -> different T0_w, same domain.
    alpha_yaw0 = np.array([1, 1, 1, 1, 1, 0])
    cond3 = op.condition(region, T, alpha_yaw0)
    assert cond3.domain is region.domain
    assert not np.allclose(cond3.T0_w, cond.T0_w)


def test_R08_unit_consistent_non_identity_basis():
    R_t = se3_exp(np.array([0.0, 0.0, 0.0, 0.2, -0.1, 0.15]))[:3, :3]
    R_r = se3_exp(np.array([0.0, 0.0, 0.0, -0.3, 0.25, 0.1]))[:3, :3]
    B = np.block([[R_t, np.zeros((3, 3))], [np.zeros((3, 3)), R_r]])
    op = ResponseOperator(T_REF, B)

    cstar = np.array([0.02, -0.03, 0.04, 0.05, -0.06, 0.07])
    alpha = np.array([1.0, 0.5, 0.0, 1.0, -0.4, 0.0])

    T = _relation(B @ cstar)
    expected = T_REF @ se3_exp(B @ (alpha * cstar)) @ np.linalg.inv(T_REF)
    np.testing.assert_allclose(op.action(T, alpha), expected, atol=1e-12)
