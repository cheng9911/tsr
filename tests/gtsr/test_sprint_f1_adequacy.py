# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""F.1: held-out adequacy for the frozen six-generator SE(3) profile (Sprint F.1).

Task 1/2 data-dependent tests (isolated split + full read-only artifacts), Task 3/4/5
pure-numpy unit tests for the SE(3) transform-level residual and per-generator adequacy,
and Task 6 the development-level integration that reports primary-direction paired
differences without freezing any margin (margins are G-00 / B1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.gtsr._fit_se3_adequacy import (
    adequacy_summary,
    aggregate_cells,
    held_out_response_error,
    primary_oracle,
    se3_response_action,
    se3_transform_error,
)
from tests.gtsr._fit_se3_cells import _artifact_sha256, fit_all_pdiag_artifacts
from tests.gtsr._maniskill_data import multiseed_root
from tests.gtsr._maniskill_data_se3 import (
    ARMS,
    EXPECTED_ISOLATED_CONTEXTS_6D,
    GENERATOR_NAMES,
    SE3_SEEDS,
    load_se3_isolated,
    progress_grid_se3,
    se3_rollouts_root,
)
from tsr.gtsr.profile6 import (
    se3_exp,
    se3_from_pose6,
    se3_inverse,
    se3_to_metric,
)

pytest.importorskip("h5py")

BINS = 25


def _require_spectrum() -> Path:
    root = multiseed_root()
    if not (root / "se3_fixed_contexts.json").exists():
        pytest.skip(f"SE(3) multiseed manifests not found at {root}")
    if not (se3_rollouts_root() / "keyed_seed_20260818.h5").exists():
        pytest.skip(f"SE(3) rollouts not found at {se3_rollouts_root()}")
    return root


# --- Task 1: isolated split -------------------------------------------------------


def test_load_se3_isolated_split_shape_and_generators():
    _require_spectrum()
    path = se3_rollouts_root() / "keyed_seed_20260818.h5"
    data = load_se3_isolated(path, BINS)
    assert len(data["isolated_keys"]) == 14
    assert data["isolated_contexts"].shape == (14, 6)
    assert data["isolated_curves"].shape == (14, 4 * BINS, 6)
    assert data["baseline_curve"].shape == (4 * BINS, 6)
    counts = {name: data["isolated_generators"].count(name) for name in GENERATOR_NAMES}
    assert counts == {"du": 2, "dv": 2, "dw": 2, "roll": 2, "pitch": 2, "yaw": 4}

    isolated_sorted = data["isolated_contexts"][
        np.lexsort(tuple(data["isolated_contexts"][:, i] for i in range(5, -1, -1)))
    ]
    expected_sorted = EXPECTED_ISOLATED_CONTEXTS_6D[
        np.lexsort(tuple(EXPECTED_ISOLATED_CONTEXTS_6D[:, i] for i in range(5, -1, -1)))
    ]
    np.testing.assert_allclose(isolated_sorted, expected_sorted, atol=1e-12)


# --- Task 2: full read-only profile artifacts -------------------------------------


@pytest.mark.slow
def test_fit_all_pdiag_artifacts_are_complete_and_hashed():
    root = _require_spectrum()
    artifacts = fit_all_pdiag_artifacts(root)
    assert len(artifacts) == 36
    for (arm, seed, subset_id), art in artifacts.items():
        assert arm in ("keyed", "circular_honest")
        assert art["theta"].shape == (6, 24)
        assert art["nominal_curve"].shape == (4 * BINS, 6)
        assert art["nominal_frame_pose"].shape == (6,)
        assert art["optimization_success"] is True
        assert len(art["source_condition_ids"]) == 30
        assert art["manifest_sha256"]
        assert len(art["artifact_sha256"]) == 64
        assert art["artifact_sha256"] == _artifact_sha256(art)


# --- Task 3: SE(3) transform-level residual core ----------------------------------


def test_se3_response_action_fixed_alpha_is_identity():
    nominal = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    context = np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0])
    T = se3_response_action(nominal, context, np.zeros(6))
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_se3_response_action_frame_alpha_recovers_full_translation():
    nominal = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    context = np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0])
    T = se3_response_action(nominal, context, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    expected = se3_from_pose6(np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(T, expected, atol=1e-12)


def test_se3_transform_error_identity_zero():
    assert se3_transform_error(np.eye(4), np.eye(4)) == pytest.approx(0.0, abs=1e-12)


def test_se3_transform_error_translation_magnitude():
    T_pred = np.eye(4)
    T_pred[:3, 3] = [0.03, 0.0, 0.0]
    assert se3_transform_error(T_pred, np.eye(4)) == pytest.approx(0.03, abs=1e-12)


def test_se3_transform_error_left_invariant():
    # Non-unit C0 with rotation: the transform-level residual must be left-invariant.
    rng = np.random.default_rng(0)
    C0 = se3_from_pose6(rng.uniform(-1.0, 1.0, 6))
    A = se3_exp(rng.uniform(-0.5, 0.5, 6))
    B = se3_exp(rng.uniform(-0.5, 0.5, 6))
    assert se3_transform_error(C0 @ A, C0 @ B) == pytest.approx(se3_transform_error(A, B), rel=1e-9)


def test_se3_transform_error_coupled_twist_is_log_norm():
    # A screw motion (coupled translation + non-commutative rotation) residual must
    # equal the metric norm of the relative log, not the pose6 coordinate difference.
    xi = np.array([0.01, 0.02, 0.03, 0.2, 0.3, 0.4])
    expected = float(np.linalg.norm(se3_to_metric(xi)))
    assert se3_transform_error(se3_exp(xi), np.eye(4)) == pytest.approx(expected, rel=1e-12)


def test_se3_response_action_conjugates_left_with_rotated_nominal():
    # Left-multiplication direction is observable: a rotated C0 must reveal a wrong
    # conjugation side (right action) as a different transform.
    nominal = np.array([0.1, 0.2, 0.3, 0.4, -0.3, 0.2])
    context = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    T = se3_response_action(nominal, context, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    C0 = se3_from_pose6(nominal)
    expected = C0 @ se3_exp(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])) @ se3_inverse(C0)
    np.testing.assert_allclose(T, expected, atol=1e-12)
    wrong = se3_inverse(C0) @ se3_exp(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])) @ C0
    assert not np.allclose(T, wrong, atol=1e-12)


# --- Task 4: per-generator held-out metrics + baseline oracles --------------------


def _synthetic_isolated():
    # nominal = identity; baseline = identity poses; one "du" isolated row, alpha_du = 0.5.
    nominal = np.zeros(6)
    baseline = np.zeros((4, 6))
    contexts = np.array([[0.015, 0.0, 0.0, 0.0, 0.0, 0.0]])
    observed = np.tile(np.array([0.0075, 0.0, 0.0, 0.0, 0.0, 0.0]), (1, 4, 1))  # (1 row, 4 steps, 6)
    generators = ["du"]
    return nominal, contexts, observed, baseline, generators


def test_held_out_response_error_zero_when_fit_matches():
    nominal, contexts, observed, baseline, generators = _synthetic_isolated()
    err = held_out_response_error(nominal, contexts, observed, baseline, generators, "du",
                                  lambda step: np.array([0.5, 0, 0, 0, 0, 0], dtype=np.float64))
    assert err == pytest.approx(0.0, abs=1e-12)


def test_held_out_response_error_positive_when_fit_wrong():
    nominal, contexts, observed, baseline, generators = _synthetic_isolated()
    err = held_out_response_error(nominal, contexts, observed, baseline, generators, "du",
                                  lambda step: np.array([0.25, 0, 0, 0, 0, 0], dtype=np.float64))
    # predicted translation = 0.25 * 0.015 = 0.00375 vs observed 0.0075 -> 0.00375.
    assert err == pytest.approx(0.00375, abs=1e-12)


# --- Task 5: per-arm/seed adequacy summary (paired differences) -------------------


def _synthetic_artifact_and_isolated():
    nominal = np.zeros(6)
    baseline = np.zeros((4, 6))
    # One isolated row per generator, all at true alpha = 0.5 (constant-theta profile).
    contexts = np.array([
        [0.015, 0.0, 0.0, 0.0, 0.0, 0.0],    # du
        [0.0, 0.015, 0.0, 0.0, 0.0, 0.0],    # dv
        [0.0, 0.0, 0.015, 0.0, 0.0, 0.0],    # dw
        [0.0, 0.0, 0.0, 0.2618, 0.0, 0.0],   # roll
        [0.0, 0.0, 0.0, 0.0, 0.2618, 0.0],   # pitch
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5236],   # yaw
    ])
    observed = np.tile(0.5 * contexts[:, None, :], (1, 4, 1))  # (6, 4, 6), true alpha = 0.5
    generators = ["du", "dv", "dw", "roll", "pitch", "yaw"]
    isolated = {"isolated_contexts": contexts, "isolated_curves": observed,
                "baseline_curve": baseline, "isolated_generators": generators}
    # Constant logit = log(0.5/0.75) -> sigmoid = 0.5/1.25 = 0.4 -> alpha_j = 0.5 for all six,
    # so fitted matches observed -> zero response error; Frame(1.0) overshoots and Fixed(0)
    # undershoots, so both paired diffs are negative for every generator.
    artifact = {"theta": np.full((6, 24), np.log(0.5 / 0.75)), "nominal_curve": baseline,
                "nominal_frame_pose": nominal, "alpha_max": 1.25, "basis_width": 0.065,
                "n_phases": 4}
    return artifact, isolated


def test_adequacy_summary_du_diff_signs():
    artifact, isolated = _synthetic_artifact_and_isolated()
    progress = np.linspace(0.0, 0.99, 4)
    summary = adequacy_summary(artifact, progress, isolated)
    assert summary["du"]["response_error"] == pytest.approx(0.0, abs=1e-12)
    assert summary["du"]["diff_vs_frame"] < 0.0
    assert summary["du"]["diff_vs_fixed"] < 0.0


def test_primary_oracle_frozen_direction():
    assert primary_oracle("keyed", "yaw") == "frame"
    assert primary_oracle("circular_honest", "yaw") == "fixed"
    assert primary_oracle("keyed", "du") == "frame"
    assert primary_oracle("circular_honest", "pitch") == "frame"


def test_aggregate_cells_means_over_subsets():
    cells = [
        {"du": {"response_error": 1.0, "task_error": 2.0, "endpoint_error": 3.0,
                "diff_vs_frame": -0.1, "diff_vs_fixed": 0.2}},
        {"du": {"response_error": 3.0, "task_error": 4.0, "endpoint_error": 5.0,
                "diff_vs_frame": -0.3, "diff_vs_fixed": 0.4}},
    ]
    agg = aggregate_cells(cells)
    assert agg["du"]["response_error"] == pytest.approx(2.0)
    assert agg["du"]["diff_vs_frame"] == pytest.approx(-0.2)


# --- Task 6: development-level integration (no constant status) -------------------


@pytest.mark.slow
def test_F1_held_out_adequacy_no_constant_status():
    root = _require_spectrum()
    artifacts = fit_all_pdiag_artifacts(root)
    progress, _ = progress_grid_se3(BINS)

    # Development-level report: one primary paired difference per arm×seed×generator,
    # selected by the frozen primary_oracle direction (never "pick the better diff").
    primary_diff = {}
    for arm in ARMS:
        for seed in SE3_SEEDS:
            subset_ids = sorted({k[2] for k in artifacts if k[:2] == (arm, seed)})
            assert len(subset_ids) == 6  # all 6 N=30 subsets participate
            isolated = load_se3_isolated(se3_rollouts_root() / f"{arm}_seed_{seed}.h5", BINS)
            cell_summaries = [
                adequacy_summary(artifacts[(arm, seed, subset_id)], progress, isolated)
                for subset_id in subset_ids
            ]
            per_seed = aggregate_cells(cell_summaries)
            for generator, values in per_seed.items():
                # No constant status: every generator produces finite real quantities.
                assert np.isfinite(values["response_error"])
                assert np.isfinite(values["task_error"])
                assert np.isfinite(values["endpoint_error"])
                assert np.isfinite(values["diff_vs_frame"])
                assert np.isfinite(values["diff_vs_fixed"])
                direction = primary_oracle(arm, generator)
                primary_diff[(arm, seed, generator)] = values[
                    "diff_vs_frame" if direction == "frame" else "diff_vs_fixed"
                ]

    # Six generators produce real quantities; the superiority/equivalence decision
    # with delta margins is frozen by G-00 (B1) and asserted there, not here.
