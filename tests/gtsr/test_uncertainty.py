"""Check uncertainty guarantees, their premises, and estimated-only geometry."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from gtsr_uncertainty import (
    EPSILON,
    METHODS,
    RHO,
    TRUE_B,
    TRUE_H,
    EstimatedSampler,
    Q,
    estimated_route,
    estimated_walls,
    exact_fixed_error_bound,
    uncertainty_bound,
)


def test_uniform_bound_is_attained_by_uncertainty_vertices():
    for db, dh, rho in itertools.product((0, 0.0005, 0.002), repeat=3):
        errors = [
            eb + eh * theta + eta for eb, eh, theta, eta in itertools.product((-db, db), (-dh, dh), (0, Q), (-rho, rho))
        ]
        assert max(abs(e) for e in errors) == pytest.approx(uncertainty_bound(db, dh, rho), abs=1e-15)


def test_canceling_fixed_errors_can_be_safe_outside_uniform_bound():
    assert uncertainty_bound(0.001, 0.001, RHO) > EPSILON
    assert exact_fixed_error_bound(0.001, -0.001, RHO) <= EPSILON + 1e-15


def test_tracking_terms_are_necessary_and_attainable():
    h, db, dh, eta, dz, dtheta = -0.05, 0.0004, 0.0006, 0.0005, 0.0002, 0.004
    z_command = db + (h + dh) * Q + eta
    residual = z_command + dz - h * (Q + dtheta)
    assert residual == pytest.approx(uncertainty_bound(db, dh, eta, Q, dz, dtheta, abs(h)))
    assert db + Q * dh + eta <= EPSILON + 1e-15 and residual > EPSILON


@pytest.mark.parametrize("method", METHODS)
def test_filtered_production_samples_obey_guarantee(method):
    db, dh = 0.0004, 0.0006
    sampler = EstimatedSampler(method, TRUE_B + db, TRUE_H + dh)
    accepted = []
    for u in np.random.default_rng(81).random((1000, 2)):
        pose = sampler.sample(u)
        if abs(sampler.residual(pose)) <= RHO + 1e-12:
            theta = np.arctan2(pose[1, 0], pose[0, 0])
            accepted.append(abs(pose[2, 3] - TRUE_B - TRUE_H * theta))
    assert len(accepted) > 100
    assert max(accepted) <= EPSILON + 1e-12


def test_raw_tsr_union_requires_estimated_band_filter():
    sampler = EstimatedSampler("tsr_union_64", TRUE_B + 0.001, TRUE_H)
    pose = sampler.sample([0.99, 0.999999])
    assert sampler.residual(pose) > RHO
    theta = np.arctan2(pose[1, 0], pose[0, 0])
    assert pose[2, 3] - TRUE_B - TRUE_H * theta > EPSILON


def test_estimated_walls_match_affine_relation_without_true_pitch():
    b, h = 0.123, -0.049
    for index, (_, position, yaw) in enumerate(estimated_walls(b, h)):
        if index % 4 == 0:
            expected_peg_depth = position[2] + 0.037
            assert b + h * yaw == pytest.approx(expected_peg_depth)


def test_route_checks_estimate_and_can_violate_unknown_truth():
    b, h = TRUE_B + 0.003, TRUE_H + 0.003
    points = [[z, (z - b) / h] for z in np.linspace(0.121, 0.083, 80)]
    route = estimated_route(points, b, h)
    assert route is not None
    assert max(abs(z - b - h * theta) for z, theta in route) < 1e-12
    assert max(abs(z - TRUE_B - TRUE_H * theta) for z, theta in route) > EPSILON


def test_sampling_support_matches_analytic_parameterization():
    sampler = EstimatedSampler("gtsr", 0.13, -0.06)
    pose = sampler.sample([0.4, 0.75])
    assert pose[2, 3] == pytest.approx(0.13 - 0.06 * 0.4 + RHO / 2)
    assert np.arctan2(pose[1, 0], pose[0, 0]) == pytest.approx(0.4)


def test_unsupported_wrapped_command_interval_is_rejected():
    with pytest.raises(ValueError):
        EstimatedSampler("gtsr", TRUE_B, TRUE_H, q=4)
