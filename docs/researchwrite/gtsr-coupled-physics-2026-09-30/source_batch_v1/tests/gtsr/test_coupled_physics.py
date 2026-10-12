"""Geometry and accounting checks independent of the physical success outcome."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from benchmark_gtsr_coupled import METHODS, geometric_trial, make_route
from gtsr_coupled_geometry import Gate, RegionSampler, coordinates, in_band


def test_straight_control_identical_production_poses():
    uniforms = np.random.default_rng(33).random((100, 2))
    reference = [RegionSampler(Gate(angle=0), "tsr").sample(u) for u in uniforms]
    for method in METHODS:
        sampler = RegionSampler(Gate(angle=0), method)
        np.testing.assert_allclose([sampler.sample(u) for u in uniforms], reference, atol=1e-14)


@pytest.mark.parametrize("method", METHODS)
def test_expected_acceptance(method):
    gate, rng = Gate(), np.random.default_rng(77)
    sampler = RegionSampler(gate, method)
    valid = [in_band(gate, sampler.sample(u)) for u in rng.random((4000, 2))]
    assert abs(np.mean(valid) - sampler.expected_band_acceptance()) < 0.035


def test_conservative_band_fits_but_straight_twisted_path_does_not():
    for angle in (0, 0.8):
        gate = Gate(angle=angle)
        assert (
            min(gate.pin_clearance(s, angle * s + eta) for s in np.linspace(0, 1, 1001) for eta in (-0.03, 0, 0.03))
            > 0.0007
        )
    assert Gate().pin_clearance(1, 0) < 0


def test_connector_rejects_missing_coverage_and_preserves_constraints():
    gate = Gate()
    assert make_route(gate, [[0.2, 0.16], [0.8, 0.64]]) is None
    points = [[s, 0.8 * s + 0.02 * np.sin(20 * s)] for s in np.linspace(0.02, 0.98, 80)]
    route = make_route(gate, points)
    assert route is not None
    array = np.asarray([[0, 0], *route])
    assert np.all(np.diff(array[:, 0]) > 0)
    assert np.max(np.diff(array[:, 0])) <= 0.025 + 1e-12
    np.testing.assert_allclose(array[-1], [1, 0.8])
    assert all(abs(y - 0.8 * s) <= 0.03 and gate.pin_clearance(s, y) > 0 for s, y in route)


def test_trial_failed_cells_are_explicit_and_paired_control_routes_match():
    failure = geometric_trial(0.8, "tsr", 12, "proposals", 1)
    assert not failure["path_found"] and failure["route"] is None
    trials = [geometric_trial(0, method, 930201, "proposals", 128) for method in METHODS]
    assert all(t["path_found"] for t in trials)
    for trial in trials[1:]:
        np.testing.assert_allclose(trial["route"], trials[0]["route"], atol=1e-13)


def test_sampler_coordinates_and_budget_deadline():
    gate = Gate()
    sampler = RegionSampler(gate, "gtsr")
    np.testing.assert_allclose(coordinates(gate, sampler.sample([0.2, 0.75])), [0.2, 0.175])
    trial = geometric_trial(0.8, "gtsr", 2, "milliseconds", 0)
    assert not trial["path_found"] and not trial["within_budget"]
