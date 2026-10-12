# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Process-parallel, resumable Pdiag-finite cell fitting for Sprint F F-04/F-05.

36 cells = 2 arms x 3 seeds x 6 N=30 subsets. Each cell recovers its nominal frame
and fits :class:`tsr.gtsr.profile6.ProfileIdentifier6D`; per-cell profiles are
cached as ``<task>_<seed>_<subset_id>.npz`` so a mid-run failure resumes without
refitting already-completed cells. Fork-based parallelism inherits the parent's
single-thread BLAS pin.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from tests.gtsr._maniskill_data_se3 import (
    ARMS,
    SE3_CONTEXT_MANIFEST_SHA256,
    SE3_EXPERIMENT_SHA256,
    SE3_SEEDS,
    SE3_SUBSET_MANIFEST_SHA256,
    load_se3_rollout,
    n30_subsets,
    nominal_frame_se3,
    progress_grid_se3,
    se3_rollouts_root,
)
from tsr.gtsr.profile6 import ProfileIdentifier6D

BINS = 25

# The resumable cache is only valid for the hash-pinned frozen dataset (F-01 is
# the authoritative integrity gate). Fold the manifest pins into the cache
# directory so re-pinning the codebase to a new dataset version invalidates
# stale cells instead of silently returning them.
_CACHE_TAG = f"{SE3_CONTEXT_MANIFEST_SHA256[:8]}_{SE3_EXPERIMENT_SHA256[:8]}_{SE3_SUBSET_MANIFEST_SHA256[:8]}"

# Module globals populated before forking (fork copies them into each worker).
_DATASETS = None
_PROGRESS = None
_PHASE_CODES = None


def _load_cell_datasets():
    global _DATASETS, _PROGRESS, _PHASE_CODES
    _PROGRESS, _PHASE_CODES = progress_grid_se3(BINS)
    _DATASETS = {
        arm: {seed: load_se3_rollout(se3_rollouts_root() / f"{arm}_seed_{seed}.h5", BINS) for seed in SE3_SEEDS}
        for arm in ARMS
    }


def _fit_one(task, seed, subset):
    import h5py
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        ds = _DATASETS[task][seed]
        cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
        indices = [cid_to_idx[cid] for cid in subset["source_condition_ids"]]
        contexts = ds["mixed_contexts"][indices]
        curves = ds["mixed_curves"][indices]
        keys = [ds["mixed_keys"][i] for i in indices]
        with h5py.File(se3_rollouts_root() / f"{task}_seed_{seed}.h5", "r") as data_file:
            nominal_frame_pose = nominal_frame_se3(data_file, keys, contexts)
        identifier = ProfileIdentifier6D()
        profile = identifier.fit(contexts, curves, _PROGRESS, _PHASE_CODES, nominal_frame_pose=nominal_frame_pose)
        return identifier, profile, nominal_frame_pose


def _fit_cell(job):
    task, seed, subset = job
    identifier, profile, _ = _fit_one(task, seed, subset)
    return task, seed, subset["subset_id"], identifier.jacobian_diag(), bool(profile.optimization["success"])


def fit_all_pdiag_cells(root, processes=None, cache_dir=None):
    """Fit all 36 Pdiag-finite cells, caching per-cell profiles. Returns dict + success map."""
    from tests.gtsr._maniskill_data_se3 import load_se3_subsets

    subsets = n30_subsets(load_se3_subsets(root))
    cache_dir = Path(
        cache_dir or os.environ.get("TSR_SE3_CACHE") or Path(tempfile.gettempdir()) / f"tsr_se3_fit_cache_{_CACHE_TAG}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    _load_cell_datasets()
    jobs = [(task, seed, subset) for task in ARMS for seed in SE3_SEEDS for subset in subsets]

    profiles = {}
    success = {}
    pending = []
    for job in jobs:
        task, seed, subset = job
        cache_path = cache_dir / f"{task}_{seed}_{subset['subset_id']}.npz"
        if cache_path.exists():
            with np.load(cache_path) as loaded:
                profiles[(task, seed, subset["subset_id"])] = loaded["profile"]
                success[(task, seed, subset["subset_id"])] = bool(loaded["success"])
        else:
            pending.append(job)

    if pending:
        import multiprocessing

        n_workers = processes or min(8, os.cpu_count() or 1)
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for task, seed, subset_id, profile, ok in pool.imap_unordered(_fit_cell, pending):
                profiles[(task, seed, subset_id)] = profile
                success[(task, seed, subset_id)] = ok
                np.savez(cache_dir / f"{task}_{seed}_{subset_id}.npz", profile=profile, success=ok)
    return profiles, success


def _artifact_sha256(artifact):
    """Deterministic SHA-256 over the canonical artifact content (sorted keys, tight JSON)."""
    canonical = {
        "arm": str(artifact["arm"]),
        "seed": int(artifact["seed"]),
        "subset_id": str(artifact["subset_id"]),
        "source_condition_ids": [int(cid) for cid in artifact["source_condition_ids"]],
        "theta": np.asarray(artifact["theta"], dtype=np.float64).tolist(),
        "nominal_curve": np.asarray(artifact["nominal_curve"], dtype=np.float64).tolist(),
        "nominal_frame_pose": np.asarray(artifact["nominal_frame_pose"], dtype=np.float64).tolist(),
        "alpha_max": float(artifact["alpha_max"]),
        "basis_width": float(artifact["basis_width"]),
        "n_basis": int(artifact["n_basis"]),
        "n_phases": int(artifact["n_phases"]),
        "optimization_success": bool(artifact["optimization_success"]),
        "optimization_cost": float(artifact["optimization_cost"]),
        "optimization_nfev": int(artifact["optimization_nfev"]),
        "optimization_optimality": float(artifact["optimization_optimality"]),
        "manifest_sha256": str(artifact["manifest_sha256"]),
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fit_artifact_cell(job):
    task, seed, subset = job
    identifier, profile, nominal_frame_pose = _fit_one(task, seed, subset)
    optimization = profile.optimization
    artifact = {
        "arm": task,
        "seed": int(seed),
        "subset_id": subset["subset_id"],
        "source_condition_ids": list(subset["source_condition_ids"]),
        "theta": np.asarray(profile.theta, dtype=np.float64),
        "nominal_curve": np.asarray(identifier.nominal_curve, dtype=np.float64),
        "nominal_frame_pose": np.asarray(nominal_frame_pose, dtype=np.float64),
        "alpha_max": profile.alpha_max,
        "basis_width": profile.basis_width,
        "n_basis": profile.n_basis,
        "n_phases": 4,
        "optimization_success": bool(optimization["success"]),
        "optimization_cost": float(optimization["cost"]),
        "optimization_nfev": int(optimization["nfev"]),
        "optimization_optimality": float(optimization["optimality"]),
        "manifest_sha256": SE3_CONTEXT_MANIFEST_SHA256,
    }
    artifact["artifact_sha256"] = _artifact_sha256(artifact)
    return artifact


def _save_artifact(path, artifact):
    np.savez(
        path,
        arm=artifact["arm"],
        seed=artifact["seed"],
        subset_id=artifact["subset_id"],
        source_condition_ids=np.asarray(artifact["source_condition_ids"]),
        theta=artifact["theta"],
        nominal_curve=artifact["nominal_curve"],
        nominal_frame_pose=artifact["nominal_frame_pose"],
        alpha_max=artifact["alpha_max"],
        basis_width=artifact["basis_width"],
        n_basis=artifact["n_basis"],
        n_phases=artifact["n_phases"],
        optimization_success=artifact["optimization_success"],
        optimization_cost=artifact["optimization_cost"],
        optimization_nfev=artifact["optimization_nfev"],
        optimization_optimality=artifact["optimization_optimality"],
        manifest_sha256=artifact["manifest_sha256"],
    )


def _load_artifact(loaded):
    artifact = {
        "arm": str(loaded["arm"]),
        "seed": int(loaded["seed"]),
        "subset_id": str(loaded["subset_id"]),
        "source_condition_ids": [int(cid) for cid in loaded["source_condition_ids"]],
        "theta": loaded["theta"],
        "nominal_curve": loaded["nominal_curve"],
        "nominal_frame_pose": loaded["nominal_frame_pose"],
        "alpha_max": float(loaded["alpha_max"]),
        "basis_width": float(loaded["basis_width"]),
        "n_basis": int(loaded["n_basis"]),
        "n_phases": int(loaded["n_phases"]),
        "optimization_success": bool(loaded["optimization_success"]),
        "optimization_cost": float(loaded["optimization_cost"]),
        "optimization_nfev": int(loaded["optimization_nfev"]),
        "optimization_optimality": float(loaded["optimization_optimality"]),
        "manifest_sha256": str(loaded["manifest_sha256"]),
    }
    artifact["artifact_sha256"] = _artifact_sha256(artifact)
    return artifact


def fit_all_pdiag_artifacts(root, processes=None, cache_dir=None):
    """Fit all 36 cells and return full provenance-pinned artifacts (theta + nominal + provenance).

    The artifact carries ``artifact_sha256``, a canonical SHA-256 over its content, so any
    mutation is detectable (tamper-evident) rather than blocked at the type level.
    """
    from tests.gtsr._maniskill_data_se3 import load_se3_subsets

    subsets = n30_subsets(load_se3_subsets(root))
    cache_dir = Path(
        cache_dir
        or os.environ.get("TSR_SE3_ARTIFACT_CACHE")
        or Path(tempfile.gettempdir()) / f"tsr_se3_artifact_cache_{_CACHE_TAG}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    _load_cell_datasets()
    jobs = [(task, seed, subset) for task in ARMS for seed in SE3_SEEDS for subset in subsets]

    artifacts = {}
    pending = []
    for job in jobs:
        task, seed, subset = job
        cache_path = cache_dir / f"{task}_{seed}_{subset['subset_id']}.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as loaded:
                artifacts[(task, seed, subset["subset_id"])] = _load_artifact(loaded)
        else:
            pending.append(job)

    if pending:
        import multiprocessing

        n_workers = processes or min(8, os.cpu_count() or 1)
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for artifact in pool.imap_unordered(_fit_artifact_cell, pending):
                task, seed, subset_id = artifact["arm"], artifact["seed"], artifact["subset_id"]
                artifacts[(task, seed, subset_id)] = artifact
                _save_artifact(cache_dir / f"{task}_{seed}_{subset_id}.npz", artifact)
    return artifacts
