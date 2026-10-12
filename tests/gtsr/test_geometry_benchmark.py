"""Independent checks for the synthetic TSR comparison, not physical validation."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from scripts.benchmark_gtsr_geometry import (
    Case,
    edge_valid,
    expected_acceptance,
    known_path,
    production_check,
    representation_contains,
    run_rrt,
    sample_states,
    segment_hits_rectangle,
    task_valid,
)


@pytest.mark.parametrize("pitch", [-0.05, 0, 0.05])
def test_analytic_acceptance_matches_independent_vertical_integration(pitch):
    case = Case(pitch, 0.002)
    for k in [1, 4, 16, 64]:
        lo, hi = case.qlo, case.qlo + (case.qhi - case.qlo) / k
        zlo, zhi = case.bounds(lo, hi)
        area = quad(
            lambda q: max(0, min(zhi, pitch * q + case.tolerance) - max(zlo, pitch * q - case.tolerance)), lo, hi
        )[0]
        np.testing.assert_allclose(
            area / ((hi - lo) * (zhi - zlo)), expected_acceptance(case, f"tsr_union_{k}"), atol=1e-12
        )


def test_tight_boxes_cover_entire_band_including_boundaries():
    case = Case(0.05, 0.0005)
    q = np.linspace(case.qlo, case.qhi, 257)
    for x in [case.xlo, case.xhi]:
        for e in [-case.tolerance, 0, case.tolerance]:
            states = np.column_stack([np.full(len(q), x), q, case.pitch * q + e])
            for method in ["tsr_box", "tsr_union_4", "tsr_union_16", "tsr_union_64", "gtsr_poe"]:
                assert representation_contains(case, method, states).all()
    # This point is inside the tight box but outside the true band.
    query = np.array([[0, case.qlo, case.bounds()[1]]])
    assert representation_contains(case, "tsr_box", query).all()
    assert not task_valid(case, query).any()


def test_fast_maps_match_actual_tsr_chain_and_poe_apis():
    for pitch in [-0.05, 0, 0.05]:
        assert production_check(Case(pitch, 0.002), np.random.default_rng(72).random((12, 3))) < 1e-12


def test_continuous_collision_finds_obstacle_between_clear_endpoints():
    rectangle = [[-0.004, 0.004], [-0.1, 0.1]]
    assert segment_hits_rectangle(np.array([-0.01, 0, 0]), np.array([0.01, 0, 0]), rectangle)
    assert not segment_hits_rectangle(np.array([-0.01, 0.2, 0]), np.array([0.01, 0.2, 0]), rectangle)
    assert segment_hits_rectangle(np.array([0.004, -0.2, 0]), np.array([0.004, 0.2, 0]), rectangle)


def test_repair_uses_coupling_without_dropping_task_tolerance():
    case = Case(0.05, 0.0005)
    u = np.random.default_rng(212).random((1000, 3))
    raw, repaired = sample_states(case, "tsr_box", u), sample_states(case, "tsr_box_repaired", u)
    assert task_valid(case, repaired).all() and not task_valid(case, raw).all()
    np.testing.assert_array_equal(raw[:, :2], repaired[:, :2])
    assert np.count_nonzero(np.isclose(abs(repaired[:, 2] - case.pitch * repaired[:, 1]), case.tolerance)) > 900


def test_common_rrt_canonical_limit_and_known_feasible_path():
    path = Path(__file__).resolve().parents[2] / "docs/researchwrite/gtsr-tsr-geometry-2026-09-30/protocol.json"
    config = json.loads(path.read_text())
    case = Case(0, 0.002)
    random = np.random.default_rng(1100).random((max(config["planning_proposal_budgets"]), 4))
    box = sample_states(case, "tsr_box", random[:, :3])
    exact = sample_states(case, "gtsr_poe", random[:, :3])
    np.testing.assert_allclose(box, exact, atol=1e-18)
    a, pa = run_rrt(case, box, random[:, 3] < config["goal_bias"], config)
    b, pb = run_rrt(case, exact, random[:, 3] < config["goal_bias"], config)
    assert [r["success"] for r in a] == [r["success"] for r in b]
    assert [r["proposals_used"] for r in a] == [r["proposals_used"] for r in b]
    if pa is not None:
        np.testing.assert_allclose(pa, pb, atol=1e-16)
    for pitch in [0, 0.02, 0.05, 0.1]:
        c = Case(pitch, 0.0005)
        route = known_path(c)
        assert all(edge_valid(c, p, q, config["configuration_obstacles_x_q"]) for p, q in zip(route[:-1], route[1:]))


def test_every_rejected_proposal_consumes_budget_and_failure_is_retained():
    path = Path(__file__).resolve().parents[2] / "docs/researchwrite/gtsr-tsr-geometry-2026-09-30/protocol.json"
    config = json.loads(path.read_text())
    n = max(config["planning_proposal_budgets"])
    invalid = np.tile([0, 0, 0.2], (n, 1))
    rows, solution = run_rrt(Case(0.05, 0.0005), invalid, np.zeros(n, dtype=bool), config)
    assert solution is None
    for row in rows:
        assert row["success"] is False
        assert row["proposals_used"] == row["budget"]
        assert row["accepted_proposals"] == 0 and row["vertices"] == 1
        assert row["path_length_mm_equivalent"] is None
