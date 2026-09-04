# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""A-01 ~ A-05: region-level observability (Sprint E, E-A evidence layer)."""

import numpy as np

from tsr.gtsr import GTSR, ResponseOperator
from tsr.gtsr.metrics import SE3Metric, se3_exp, se3_log
from tsr.gtsr.observability import (
    calibrate_epsilon_obs,
    is_region_observable,
    region_discrepancy,
    region_observability_residual,
    symmetric_residual_set_discrepancy,
)
from tsr.gtsr.profile import CellStatus, adjudicate_cell

I6 = np.eye(6)
DELTA = 0.3  # pure-yaw intervention amplitude (rad)

# x = y = 0 makes the pure-yaw action map Phi(z) -> Phi(z + delta) exactly, so the
# follow/ignore and symmetry-control discrepancies are analytic (scope §6).
NARROW_BW = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.1, 0.1],
        [-0.1, 0.1],
        [-0.1, 0.1],
        [-0.05, 0.05],
    ]
)
FULL_YAW_BW = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.1, 0.1],
        [-0.1, 0.1],
        [-0.1, 0.1],
        [-np.pi, np.pi],
    ]
)


def _yaw_relation():
    return se3_exp(np.array([0.0, 0.0, 0.0, 0.0, 0.0, DELTA]))


def test_A01_narrow_follow_ignore_is_observable():
    op = ResponseOperator(np.eye(4), I6)
    base = GTSR.canonical(Bw=NARROW_BW)
    relation = _yaw_relation()
    follow = op.condition(base, relation, np.ones(6))  # yaw propagated
    ignore = op.condition(base, relation, np.array([1, 1, 1, 1, 1, 0]))  # yaw released

    # Deterministic identity: the follow region is the ignore region left-multiplied
    # by a pure yaw rotation of DELTA (exact for every chart coordinate).
    rng = np.random.default_rng(0)
    for _ in range(8):
        z = base.domain.sample(rng)
        np.testing.assert_allclose(follow.to_transform(z), relation @ ignore.to_transform(z), atol=1e-12)

    epsilon_obs = calibrate_epsilon_obs([], epsilon_num=1e-3)
    # The analytic observability check (exact canonical bounds, not sampled points)
    # reports the yaw response observable: the follow action carries poses out of the
    # narrow region, and the follow/ignore pair is strictly above the calibration floor.
    observable = is_region_observable(base, op.action(relation, np.ones(6)), epsilon_obs, rng=rng)
    assert observable
    assert region_discrepancy(follow, ignore, n_samples=128, rng=rng) > epsilon_obs

    status = adjudicate_cell(excited=True, observable=observable, fit_succeeded=True, meets_heldout_gate=True)
    assert status is CellStatus.IDENTIFIED


def test_A02_wide_domain_symmetry_control_unobservable():
    op = ResponseOperator(np.eye(4), I6)
    base = GTSR.canonical(Bw=FULL_YAW_BW)
    relation = _yaw_relation()
    action = op.action(relation, np.ones(6))
    cond = op.condition(base, relation, np.ones(6))

    # The action is a pure z-rotation and the yaw domain is a full circle, so the
    # action maps the region set onto itself.
    log = se3_log(action)
    np.testing.assert_allclose(log[:3], 0.0, atol=1e-12)
    np.testing.assert_allclose(log[3:], [0.0, 0.0, DELTA], atol=1e-12)
    yaw_span = base.domain.cont_upper[5] - base.domain.cont_lower[5]
    assert np.isclose(yaw_span, 2.0 * np.pi)
    assert cond.domain is base.domain and cond.chart is base.chart

    # The analytic observability check is *exactly* zero for the region-set symmetry
    # (a sampled point-set discrepancy would return a positive value here), so the
    # yaw cell is UNOBSERVABLE — never IDENTIFIED.
    rng = np.random.default_rng(0)
    epsilon_obs = calibrate_epsilon_obs([], epsilon_num=1e-3)
    assert region_observability_residual(base, action, n_samples=128, rng=rng) == 0.0
    assert not is_region_observable(base, action, epsilon_obs, rng=rng)

    status = adjudicate_cell(excited=True, observable=False, fit_succeeded=True, meets_heldout_gate=True)
    assert status is CellStatus.UNOBSERVABLE


def test_A03_epsilon_obs_calibration():
    # Deterministic control -> only the numeric floor survives.
    assert calibrate_epsilon_obs([], epsilon_num=1e-6) == 1e-6
    assert calibrate_epsilon_obs([1e-15], epsilon_num=1e-6) == 1e-6
    # Non-deterministic control -> 0.99 quantile dominates when it exceeds the floor.
    control = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert calibrate_epsilon_obs(control, epsilon_num=1e-6) == np.quantile(control, 0.99)


def test_A04_symmetric_residual_set_discrepancy_basic():
    metric = SE3Metric()
    a = np.stack([np.eye(4), se3_exp(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))])
    b = np.stack([se3_exp(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2]))])
    d = symmetric_residual_set_discrepancy(a, b, metric)
    assert d > 0.0 and np.isfinite(d)


def test_A05_identical_region_discrepancy_is_zero():
    # Shared chart coordinates make an identical-region comparison exactly zero;
    # independent sampling would return a positive value and is unusable here.
    base = GTSR.canonical(Bw=NARROW_BW)
    rng = np.random.default_rng(0)
    assert region_discrepancy(base, base, n_samples=64, rng=rng) < 1e-12
