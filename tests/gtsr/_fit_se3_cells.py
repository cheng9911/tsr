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

import os
import tempfile
from pathlib import Path

import numpy as np

from tests.gtsr._maniskill_data_se3 import (
    ARMS,
    SE3_SEEDS,
    load_se3_rollout,
    n30_subsets,
    nominal_frame_se3,
    progress_grid_se3,
    se3_rollouts_root,
)
from tsr.gtsr.profile6 import ProfileIdentifier6D

BINS = 25

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


def _fit_cell(job):
    import h5py
    import threadpoolctl

    task, seed, subset = job
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
        profile = identifier.fit(
            contexts, curves, _PROGRESS, _PHASE_CODES, nominal_frame_pose=nominal_frame_pose
        )
        return task, seed, subset["subset_id"], identifier.jacobian_diag(), bool(profile.optimization["success"])


def fit_all_pdiag_cells(root, processes=None, cache_dir=None):
    """Fit all 36 Pdiag-finite cells, caching per-cell profiles. Returns dict + success map."""
    from tests.gtsr._maniskill_data_se3 import load_se3_subsets

    subsets = n30_subsets(load_se3_subsets(root))
    cache_dir = Path(cache_dir or os.environ.get("TSR_SE3_CACHE", Path(tempfile.gettempdir()) / "tsr_se3_fit_cache"))
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
