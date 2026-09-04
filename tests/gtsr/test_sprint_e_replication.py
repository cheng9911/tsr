# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""E-01/E-02/E-05/E-06/E-07/E-08: frozen Pdiag-finite replication (Sprint E, E-B layer).

These tests read the pre-registered ManiSkill phase-switch multiseed rollouts from
the sister repository ``../maniskill_spectrum`` and verify that
``ProfileIdentifier`` reproduces the frozen ``SmoothFinitePDiagModel`` gates, that
the planar -> 6-D embedding is exactly realized by ``ResponseOperator``'s SE(3) left
action, and that held-out task error beats the frozen scalar (Frame-weighted)
baseline across every seed. The per-generator cell status is produced from the data
via :func:`cell_statuses` (excitation from the intervention contexts, observability
from the analytic E-A check, fit from the optimizer, adequacy from the frozen gate)
rather than hardcoded verdicts. Everything here is skip-gated on ``h5py`` and
``threadpoolctl`` and the data being present.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tests.gtsr._maniskill_data import (
    CONTEXT_MANIFEST_SHA256,
    SEEDS,
    load_collection_status,
    load_experiment,
    load_seed_dataset,
    metric_errors,
    multiseed_root,
    progress_grid,
    sha256,
    switch_diagnostics,
)
from tsr.gtsr import GTSR, ResponseOperator
from tsr.gtsr.metrics import se3_exp
from tsr.gtsr.observability import (
    calibrate_epsilon_obs,
    is_region_observable,
    region_observability_residual,
)
from tsr.gtsr.profile import (
    CellStatus,
    ProfileIdentifier,
    cell_statuses,
    from_metric,
    planar_to_se3_alpha,
    pose2_matrix,
    se2_to_se3,
    socket_frame,
    to_metric,
)

pytest.importorskip("h5py")
threadpoolctl = pytest.importorskip("threadpoolctl")


@pytest.fixture(autouse=True)
def _single_threaded_blas():
    """Pin OpenBLAS to one thread for the deterministic least-squares fits.

    The frozen ``SmoothFinitePDiagModel`` fits with a dense finite-difference
    Jacobian; multi-threaded OpenBLAS thrashes on those small GEMMs (~34s/fit vs
    ~5s single-threaded). ``threadpool_limits`` restores the prior thread count on
    exit, so the import-time global mutation the previous helper performed is gone
    and nothing leaks across the rest of the suite.
    """
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        yield


# The per-seed fits are deterministic but heavy (~5s each single-threaded); the
# remaining suite is fast, so only the replication module opts into the slow mark.
pytestmark = pytest.mark.slow

BINS = 25
PRECLEAR_INDEX = 2 * BINS - 1
FINAL_INDEX = 4 * BINS - 1
UNLOCK_START = 2 * BINS

# Loose numeric agreement with the frozen ``multiseed_seed_summary.csv``. The fit is
# deterministic, so these are wide margins that only catch gross divergence.
GATE_ATOL = 5e-3
TASK_ERROR_ATOL = 0.5

# E-A observability geometry (scope §6). A narrow domain makes every generator's
# pure response action carry poses out of the region (observable); a full-circle
# yaw domain makes the pure-yaw action a region-set symmetry (unobservable).
NARROW_BW = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.01, 0.01],
        [-0.01, 0.01],
        [-0.01, 0.01],
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
GENERATOR_TWISTS = (
    np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0]),  # du
    np.array([0.0, 0.015, 0.0, 0.0, 0.0, 0.0]),  # dv
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3]),  # yaw
)


def _epsilon_obs():
    # Deterministic synthetic control -> only the numeric floor survives (scope §6).
    return calibrate_epsilon_obs([], epsilon_num=1e-3)


def _generator_observabilities(bw):
    """Per-generator E-A observability verdict (analytic, not a hardcoded flag).

    Each planar generator is probed with its pure response action on the given
    region; the verdict comes from :func:`is_region_observable` so that E-B cells
    carry the E-A layer's conclusion into :func:`cell_statuses` rather than a
    literal ``observable=True``.
    """
    op = ResponseOperator(np.eye(4), np.eye(6))
    base = GTSR.canonical(Bw=bw)
    epsilon_obs = _epsilon_obs()
    return [
        is_region_observable(base, op.action(se3_exp(twist), np.ones(6)), epsilon_obs) for twist in GENERATOR_TWISTS
    ]


def _require_root() -> Path:
    root = multiseed_root()
    if not (root / "collection_status.json").exists():
        pytest.skip(f"Maniskill spectrum multiseed data not found at {root}")
    return root


def _frozen_seed_summary(root: Path) -> dict:
    rows = {}
    path = root / "results" / "multiseed_seed_summary.csv"
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed = int(row["seed"])
            numeric = {}
            for key, value in row.items():
                if key in {
                    "seed",
                    "translation_pass",
                    "pre_yaw_pass",
                    "final_yaw_pass",
                    "switch_pass",
                    "improvement_pass",
                    "all_criteria_pass",
                }:
                    continue
                numeric[key] = float(value)
            rows[seed] = numeric
    return rows


def _frame_weighted_predict(train_contexts, train_curves, test_contexts):
    """Frozen Frame-weighted scalar baseline (per-progress shared scalar w(s))."""
    contexts_metric = to_metric(train_contexts)
    curves_metric = to_metric(train_curves)
    n_steps = curves_metric.shape[1]
    intercept_metric = np.empty((n_steps, 3), dtype=np.float64)
    weight = np.empty(n_steps, dtype=np.float64)
    for step in range(n_steps):
        centered = contexts_metric - contexts_metric.mean(axis=0, keepdims=True)
        y = curves_metric[:, step]
        y_centered = y - y.mean(axis=0, keepdims=True)
        w = np.sum(centered * y_centered) / np.sum(centered * centered)
        weight[step] = w
        intercept_metric[step] = y.mean(axis=0) - w * contexts_metric.mean(axis=0)
    test_metric = to_metric(test_contexts)
    prediction_metric = intercept_metric[None, :, :] + weight[None, :, None] * test_metric[:, None, :]
    return from_metric(prediction_metric)


def test_E01_manifest_and_dataset_hashes():
    root = _require_root()
    status = load_collection_status(root)
    # The context manifest is pinned to an independent constant (scope §5) so a
    # tampered ``fixed_contexts.json`` that agrees with ``collection_status.json``
    # is still caught.
    assert sha256(root / "fixed_contexts.json") == CONTEXT_MANIFEST_SHA256
    assert status["context_manifest_sha256"] == CONTEXT_MANIFEST_SHA256
    assert status["experiment_sha256"] == sha256(root / "experiment.json")
    for seed in SEEDS:
        entry = status["datasets"][str(seed)]
        assert entry["complete_container"] is True
        assert sha256(Path(entry["path"])) == entry["sha256"]
        # Loading is part of E-01: it fail-fast validates the registered 30/8
        # split, mixed-context rank/uniqueness/signed coverage, and every isolated
        # context amplitude rather than merely checking the container hash.
        data = load_seed_dataset(Path(entry["path"]), BINS)
        assert data.seed == seed


def test_E02_progress_grid_matches_phase_mapping():
    progress, phase_codes = progress_grid(BINS)
    assert progress.shape == (4 * BINS,)
    assert phase_codes.shape == (4 * BINS,)
    phase_progress = np.linspace(0.0, 1.0, BINS)
    expected = np.concatenate([(i + phase_progress) / 4 for i in range(4)])
    np.testing.assert_allclose(progress, expected)
    np.testing.assert_array_equal(phase_codes, np.repeat([3, 4, 5, 6], BINS))


def test_E05_per_seed_gates_match_frozen_summary():
    root = _require_root()
    experiment = load_experiment(root)
    criteria = experiment["success_criteria"]
    frozen = _frozen_seed_summary(root)
    identifier = ProfileIdentifier()

    for seed in SEEDS:
        data = load_seed_dataset(root / "rollouts" / f"seed_{seed}.h5", BINS)
        profile = identifier.fit(
            data.train_contexts,
            data.train_curves,
            data.progress,
            data.phase_codes,
            nominal_frame_xy=data.nominal_frame_xy,
            nominal_frame_yaw=data.nominal_frame_yaw,
        )
        diagonal = identifier.jacobian_diag()

        g_translation_final = 0.5 * (diagonal[FINAL_INDEX, 0] + diagonal[FINAL_INDEX, 1])
        g_yaw_preclear = diagonal[PRECLEAR_INDEX, 2]
        g_yaw_final = diagonal[FINAL_INDEX, 2]

        target_switch = switch_diagnostics(data.progress, data.target_profile[:, 2], unlock_start=UNLOCK_START)
        model_switch = switch_diagnostics(data.progress, diagonal[:, 2], unlock_start=UNLOCK_START)
        assert target_switch["detected"], "empirical isolated yaw profile has no 0.5 crossing"
        switch_error = abs(model_switch["location"] - target_switch["location"])

        ref = frozen[seed]
        # The pre-registered gates (pass/fail), matching the frozen summary verdicts.
        assert g_translation_final > criteria["g_translation_final_min"]
        assert g_yaw_preclear > criteria["g_yaw_preclear_min"]
        assert abs(g_yaw_final) < criteria["abs_g_yaw_final_max"]
        assert switch_error < criteria["switch_absolute_error_max"]
        assert profile.optimization["success"]

        # Loose numeric agreement with the frozen values (deterministic fit).
        assert abs(g_translation_final - ref["g_translation_final"]) < GATE_ATOL
        assert abs(g_yaw_preclear - ref["g_yaw_preclear"]) < GATE_ATOL
        assert abs(g_yaw_final - ref["g_yaw_final"]) < GATE_ATOL
        assert abs(switch_error - ref["switch_absolute_error"]) < GATE_ATOL


def test_E06_planar_to_se3_action_matches_frozen_se2():
    root = _require_root()
    data = load_seed_dataset(root / "rollouts" / "seed_20260818.h5", BINS)
    identifier = ProfileIdentifier()
    identifier.fit(
        data.train_contexts,
        data.train_curves,
        data.progress,
        data.phase_codes,
        nominal_frame_xy=data.nominal_frame_xy,
        nominal_frame_yaw=data.nominal_frame_yaw,
    )
    nominal_frame = pose2_matrix([*data.nominal_frame_xy, data.nominal_frame_yaw])
    operator = ResponseOperator(se2_to_se3(nominal_frame), np.eye(6))
    diagonal = identifier.jacobian_diag()

    for context in data.test_contexts:
        relation = se2_to_se3(socket_frame(context, data.nominal_frame_xy, data.nominal_frame_yaw))
        for step in range(0, len(data.progress), 10):
            alpha3 = diagonal[step]
            se3_action = operator.action(relation, planar_to_se3_alpha(alpha3))
            se2_action = identifier.action(context, alpha3)
            np.testing.assert_allclose(se2_to_se3(se2_action), se3_action, atol=1e-9)


def test_E07_heldout_task_error_beats_scalar_and_cells_identified():
    root = _require_root()
    experiment = load_experiment(root)
    criteria = experiment["success_criteria"]
    frozen = _frozen_seed_summary(root)
    identifier = ProfileIdentifier()
    observabilities = _generator_observabilities(NARROW_BW)
    assert observabilities == [True, True, True], "narrow-domain E-A check must report all generators observable"

    for seed in SEEDS:
        data = load_seed_dataset(root / "rollouts" / f"seed_{seed}.h5", BINS)
        profile = identifier.fit(
            data.train_contexts,
            data.train_curves,
            data.progress,
            data.phase_codes,
            nominal_frame_xy=data.nominal_frame_xy,
            nominal_frame_yaw=data.nominal_frame_yaw,
        )
        prediction = identifier.predict(data.test_contexts)
        yaw_weights = (data.phase_codes < 5).astype(np.float64)

        pdiag_task_error, _ = metric_errors(prediction, data.test_curves, yaw_weights=yaw_weights)
        scalar_prediction = _frame_weighted_predict(data.train_contexts, data.train_curves, data.test_contexts)
        scalar_task_error, _ = metric_errors(scalar_prediction, data.test_curves, yaw_weights=yaw_weights)

        # Every seed must beat the frozen scalar baseline, not just seed 20260818.
        assert pdiag_task_error.mean() < scalar_task_error.mean()

        # Loose agreement with the frozen per-seed held-out task errors.
        ref = frozen[seed]
        assert abs(pdiag_task_error.mean() - ref["pdiag_task_error_mm_equiv"]) < TASK_ERROR_ATOL
        assert abs(scalar_task_error.mean() - ref["scalar_task_error_mm_equiv"]) < TASK_ERROR_ATOL

        # Data -> status: excitation from contexts, fit from the optimizer, adequacy
        # from the frozen gate, observability from the E-A analytic check. The frozen
        # translation gate is joint over du/dv, so both share it.
        diagonal = identifier.jacobian_diag()
        g_translation = 0.5 * (diagonal[FINAL_INDEX, 0] + diagonal[FINAL_INDEX, 1])
        g_yaw_preclear = diagonal[PRECLEAR_INDEX, 2]
        g_yaw_final = diagonal[FINAL_INDEX, 2]
        meets_gate = np.array(
            [
                g_translation > criteria["g_translation_final_min"],
                g_translation > criteria["g_translation_final_min"],
                g_yaw_preclear > criteria["g_yaw_preclear_min"] and abs(g_yaw_final) < criteria["abs_g_yaw_final_max"],
            ]
        )
        statuses = cell_statuses(
            data.train_contexts,
            observable=observabilities,
            fit_succeeded=profile.optimization["success"],
            meets_heldout_gate=meets_gate,
        )
        assert statuses == [CellStatus.IDENTIFIED, CellStatus.IDENTIFIED, CellStatus.IDENTIFIED]


def test_E08_wide_domain_negative_control_abstains():
    op = ResponseOperator(np.eye(4), np.eye(6))
    base = GTSR.canonical(Bw=FULL_YAW_BW)
    yaw_action = op.action(se3_exp(GENERATOR_TWISTS[2]), np.ones(6))
    epsilon_obs = _epsilon_obs()

    # Analytic region-set symmetry: the residual is exactly zero, so the wide-domain
    # yaw response is UNOBSERVABLE — never IDENTIFIED (scope §6 E-A, gate E-08).
    assert region_observability_residual(base, yaw_action) == 0.0
    assert not is_region_observable(base, yaw_action, epsilon_obs)

    # Derive every generator's verdict from the analytic check and wire it into the
    # data -> status path. The yaw generator abstains to UNOBSERVABLE; no manual
    # ``observable=False`` literal is involved.
    observabilities = _generator_observabilities(FULL_YAW_BW)
    assert observabilities[2] is False
    contexts = np.array(
        [
            [-0.015, -0.015, -0.3],
            [0.015, 0.015, 0.3],
            [-0.015, 0.015, -0.3],
            [0.015, -0.015, 0.3],
        ]
    )
    statuses = cell_statuses(
        contexts,
        observable=observabilities,
        fit_succeeded=True,
        meets_heldout_gate=True,
    )
    assert statuses[0] is CellStatus.IDENTIFIED
    assert statuses[1] is CellStatus.IDENTIFIED
    assert statuses[2] is CellStatus.UNOBSERVABLE
