"""Scientific contracts for the physical execution bridge, without a simulator."""

import numpy as np
import pytest

from scripts.gtsr_robot_execution import (
    cell_key,
    interpolate_action,
    preserve_interrupted_attempt,
    region_pose,
    schedule,
    validate_preparation,
)
from tsr.gtsr.metrics import se3_exp


def test_production_regions_are_identical_with_free_yaw():
    rng = np.random.default_rng(9381)
    for free in (False, True):
        for _ in range(40):
            center = se3_exp(rng.normal(size=6) * [0.2, 0.2, 0.2, 0.5, 0.5, 0.5])
            coordinate = rng.uniform(-1, 1, 4) * [0.0005, 0.0005, 0.0005, np.pi / 6 if free else 0]
            tsr = region_pose(center, "tsr", coordinate, free)
            generalized = region_pose(center, "gtsr", coordinate, free)
            np.testing.assert_allclose(generalized, tsr, atol=2e-15, rtol=0)


def test_locked_yaw_cannot_be_sampled_as_free():
    with pytest.raises(ValueError, match="outside bounds"):
        region_pose(np.eye(4), "gtsr", [0, 0, 0, 0.1], False)


def test_action_interpolation_never_crosses_phase_boundary():
    values = np.repeat(np.eye(4)[None], 100, axis=0)
    values[:, 0, 3] = np.repeat([0, 1, 3, 7], 25)
    for q, expected in enumerate((0, 1, 3, 7)):
        for rho in (0, 0.4, 1):
            assert interpolate_action(values, q, rho)[0, 3] == expected


def test_nominal_schedule_reaches_clearance_before_unlock():
    rows = schedule()
    assert sorted({q for q, *_ in rows}) == [0, 1, 2, 3]
    enter = np.array([depth for q, _, depth, _ in rows if q == 1])
    assert np.all(np.diff(enter) < 0)
    assert np.max(-np.diff(np.r_[0.125, enter])) <= 0.003 + 1e-12
    assert enter[-1] == 0.078
    assert all(depth == 0.078 for q, _, depth, _ in rows if q == 2)
    assert rows[-1][2] == 0.060


def test_interrupted_attempt_is_retained_without_retry(tmp_path):
    import json

    cell = dict(arm="keyed", context="nominal", seed=123, method="tsr_frame")
    key = cell_key(cell)
    assert not preserve_interrupted_attempt(tmp_path, cell)
    (tmp_path / f"{key}.started.json").write_text("{}")
    (tmp_path / f"{key}.log").write_text("preserve original attempt")
    assert preserve_interrupted_attempt(tmp_path, cell)
    record = json.loads((tmp_path / f"{key}.json").read_text())
    assert not record["success"]
    assert record["stop_reason"] == "interrupted_attempt_not_retried"
    assert (tmp_path / f"{key}.log").read_text() == "preserve original attempt"


def test_prepared_config_cannot_drift(tmp_path):
    import json

    (tmp_path / "protocol.json").write_text("changed")
    (tmp_path / "preparation.json").write_text(json.dumps({"config_sha256": "not-the-same-hash"}))
    with pytest.raises(ValueError, match="prepared input drift"):
        validate_preparation(tmp_path)


def test_execution_summary_keeps_timeout_in_denominator():
    from scripts.report_gtsr_robot_execution import summarize

    rows = [
        {
            "arm": "keyed",
            "method": "tsr_frame",
            "context": "nominal",
            "success": True,
            "steps": 4,
            "max_net_contact_force_N": 3,
            "planning_seconds": 2,
            "stop_reason": "terminated",
        },
        {
            "arm": "keyed",
            "method": "tsr_frame",
            "context": "nominal",
            "success": False,
            "stop_reason": "process_timeout",
        },
        {
            "arm": "keyed",
            "method": "tsr_frame",
            "context": "nominal",
            "success": False,
            "stop_reason": "interrupted_attempt_not_retried",
        },
    ]
    summary = summarize(rows, ["keyed"], ["tsr_frame"])[0]
    assert summary["attempts"] == 3
    assert summary["successes"] == 1
    assert summary["traces"] == 1
    assert summary["planning_time_measurement_count"] == 1
