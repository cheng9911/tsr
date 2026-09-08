# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""F-01/F-04/F-05/F-06: frozen 6D six-generator replication (Sprint F).

Reads the pre-registered SE(3) phase-switch rollouts from the sister repository
``../maniskill_spectrum/phase_switch_symmetry_rollouts_se3`` and the manifests at
``../maniskill_spectrum/phase_switch_symmetry_multiseed``. F-01 verifies manifest /
rollout integrity; F-04 reproduces the frozen 18-pair Pdiag-finite selectivity
headline; F-05 is the Frame-weighted negative control; F-06 runs the full suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.gtsr._maniskill_data import multiseed_root, sha256
from tests.gtsr._maniskill_data_se3 import (
    ARMS,
    CONTEXT_SCALE,
    EXPECTED_ISOLATED_CONTEXTS_6D,
    SE3_CONTEXT_MANIFEST_SHA256,
    SE3_EXPERIMENT_SHA256,
    SE3_SEEDS,
    SE3_SUBSET_MANIFEST_SHA256,
    load_se3_manifests,
    load_se3_rollout,
    load_se3_subsets,
    n30_subsets,
    progress_grid_se3,
    se3_rollouts_root,
)

pytest.importorskip("h5py")
threadpoolctl = pytest.importorskip("threadpoolctl")

BINS = 25


@pytest.fixture(autouse=True)
def _single_threaded_blas():
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        yield


def _require_spectrum() -> Path:
    root = multiseed_root()
    if not (root / "se3_fixed_contexts.json").exists():
        pytest.skip(f"SE(3) multiseed manifests not found at {root}")
    if not (se3_rollouts_root() / "keyed_seed_20260818.h5").exists():
        pytest.skip(f"SE(3) rollouts not found at {se3_rollouts_root()}")
    return root


def test_F01_manifests_and_rollouts_integrity():
    root = _require_spectrum()
    assert sha256(root / "se3_fixed_contexts.json") == SE3_CONTEXT_MANIFEST_SHA256
    assert sha256(root / "se3_experiment.json") == SE3_EXPERIMENT_SHA256
    assert sha256(root / "se3_subsets.json") == SE3_SUBSET_MANIFEST_SHA256

    fixed, experiment, _ = load_se3_manifests(root)
    subsets = load_se3_subsets(root)
    assert subsets["experiment_sha256"] == SE3_EXPERIMENT_SHA256
    assert subsets["context_manifest_sha256"] == SE3_CONTEXT_MANIFEST_SHA256
    assert experiment["context_manifest_sha256"] == SE3_CONTEXT_MANIFEST_SHA256

    conditions = fixed["conditions"]
    assert len(conditions) == 75
    mixed = [c for c in conditions if c["generator"] == "mixed"]
    isolated = [c for c in conditions if c["generator"] not in ("mixed", "baseline")]
    baseline = [c for c in conditions if c["generator"] == "baseline"]
    assert len(mixed) == 60 and len(isolated) == 14 and len(baseline) == 1

    isolated_sorted = np.asarray(
        sorted((c["causal_delta"] for c in isolated), key=lambda v: tuple(v)),
        dtype=np.float64,
    )
    expected_sorted = EXPECTED_ISOLATED_CONTEXTS_6D[
        np.lexsort(tuple(EXPECTED_ISOLATED_CONTEXTS_6D[:, i] for i in range(5, -1, -1)))
    ]
    np.testing.assert_allclose(isolated_sorted, expected_sorted, atol=1e-12)

    # 6 N=30 subsets, rank-6, and normalized condition number < 10.
    n30 = n30_subsets(subsets)
    assert len(n30) == 6
    for subset in n30:
        assert subset["sample_size"] == 30
        assert subset["rank"] == 6
        assert subset["condition_number"] < 10.0

    # Rollouts: every (arm, seed) has 75 usable episodes, 60 mixed, 14 isolated, 1 baseline;
    # the 60-mixed normalized context matrix is rank-6 with cond < 10.
    for arm in ARMS:
        for seed in SE3_SEEDS:
            path = se3_rollouts_root() / f"{arm}_seed_{seed}.h5"
            data = load_se3_rollout(path, BINS)
            assert len(data["usable_keys"]) == 75
            assert len(data["mixed_keys"]) == 60
            assert len(data["isolated_keys"]) == 14
            assert len(data["baseline_keys"]) == 1
            normalized = data["mixed_contexts"] / CONTEXT_SCALE
            assert np.linalg.matrix_rank(normalized) == 6
            assert np.linalg.cond(normalized) < 10.0


def test_F01_progress_grid_q_indices():
    progress, phase_codes = progress_grid_se3(BINS)
    assert progress.shape == (4 * BINS,)
    assert phase_codes.shape == (4 * BINS,)
    np.testing.assert_allclose(
        progress,
        np.concatenate([(i + np.linspace(0.0, 1.0, BINS)) / 4 for i in range(4)]),
    )
    np.testing.assert_array_equal(phase_codes, np.repeat([0, 1, 2, 3], BINS))
