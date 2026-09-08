# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Test-side HDF5/JSON loader for the frozen SE(3) six-generator dataset.

Mirrors ``maniskill_spectrum``'s ``benchmark_se3_transfer.py`` extraction logic
(pose6 I/O via the local ``R = Rx Ry Rz`` convention, q-index ``progress_grid``,
nominal-frame recovery, mixed/isolated/baseline split) so Sprint F's F-01/F-04
can consume the pre-registered rollouts without pandas / transforms3d.

CRITICAL: unlike the planar loader, the frozen SE(3) ``progress_grid`` emits
q-indices ``[0, 1, 2, 3]`` (NOT solver codes ``[3, 4, 5, 6]``); ``pre = q<2``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.gtsr._maniskill_data import (
    complete,
    episode_keys,
    multiseed_root,
    resample,
    usable,
)
from tsr.gtsr.profile6 import (
    euler_from_matrix,
    pose6_from_se3,
    se3_from_pose6,
    se3_inverse,
)

SE3_SEEDS = (20260818, 20270818, 20280818)
ARMS = ("keyed", "circular_honest")
GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")
YAW_INDEX = 5
PHASE_CODES = (3, 4, 5, 6)

CONTEXT_SCALE = np.array(
    [0.012, 0.012, 0.012, 0.2617993877991494, 0.2617993877991494, 0.5235987755982988],
    dtype=np.float64,
)

SE3_CONTEXT_MANIFEST_SHA256 = "30d7f46f8ce73fc94704e0cdf433087de2f129156a45c90dd3c1a516ed5f64d5"
SE3_EXPERIMENT_SHA256 = "44e219fe7c00a1ad2a7761d61618655420cbbd29f542ee2b897a4e1ec4244e4a"
SE3_SUBSET_MANIFEST_SHA256 = "b14b6d26b52daa7d26fd832666d6c195731f2180726e499dd13edc57773664f0"

# The 14 isolated interventions, ordered as in se3_fixed_contexts.json.
EXPECTED_ISOLATED_CONTEXTS_6D = np.array(
    [
        [-0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.015, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.015, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.015, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.015, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.2617993877991494, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.2617993877991494, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.2617993877991494, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.2617993877991494, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.5235987755982988],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.2617993877991494],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2617993877991494],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5235987755982988],
    ],
    dtype=np.float64,
)


def quat2mat(q):
    """Reimplement ``transforms3d.quaternions.quat2mat`` for a ``[w, x, y, z]`` quaternion."""
    w, x, y, z = (float(v) for v in q)
    nq = w * w + x * x + y * y + z * z
    if nq < 1e-8:
        return np.eye(3)
    s = 2.0 / nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ = y * Y, y * Z
    zZ = z * Z
    return np.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ],
        dtype=np.float64,
    )


def pose_to_pose6(pose):
    """SAPIEN raw pose ``(..., 7) = [x, y, z, w, x, y, z]`` -> ``[x, y, z, roll, pitch, yaw]``."""
    pose = np.asarray(pose, dtype=np.float64)
    single = pose.ndim == 1
    if single:
        pose = pose[None]
    rpy = np.stack([euler_from_matrix(quat2mat(q)) for q in pose[..., 3:]])
    out = np.concatenate([pose[..., :3], rpy], axis=-1)
    return out[0] if single else out


def phase_task_curve_se3(group, phase, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    peg = np.asarray(group["peg_pose"])[indices]
    rpy = np.stack([euler_from_matrix(quat2mat(q)) for q in peg[:, 3:]])
    rpy = np.unwrap(rpy, axis=0)
    pose6 = np.column_stack([peg[:, 0], peg[:, 1], peg[:, 2], rpy])
    return resample(pose6, bins)


def task_curve_se3(group, bins):
    return np.concatenate([phase_task_curve_se3(group, phase, bins) for phase in PHASE_CODES], axis=0)


def progress_grid_se3(bins):
    phase_progress = np.linspace(0.0, 1.0, bins)
    progress = np.concatenate([(phase_index + phase_progress) / len(PHASE_CODES) for phase_index in range(4)])
    phase_codes = np.repeat(np.arange(4), bins)  # q-indices, NOT solver codes
    return progress, phase_codes


def nominal_frame_se3(data_file, keys, contexts):
    """Recover the nominal socket frame pose6: ``C0 = socket_world o intervention^-1``."""
    pose6s = []
    for key, context in zip(keys, contexts):
        socket_pose = np.asarray(data_file[key]["socket_pose"])[0]
        socket_pose6 = pose_to_pose6(socket_pose)
        socket_T = se3_from_pose6(socket_pose6)
        intervention_T = se3_from_pose6(np.asarray(context, dtype=np.float64))
        nominal_T = socket_T @ se3_inverse(intervention_T)
        pose6s.append(pose6_from_se3(nominal_T))
    pose6s = np.asarray(pose6s)
    nominal = np.empty(6, dtype=np.float64)
    nominal[:3] = pose6s[:, :3].mean(axis=0)
    for generator in range(3):
        angle = pose6s[:, 3 + generator]
        nominal[3 + generator] = float(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()))
    return nominal


def se3_rollouts_root() -> Path:
    return multiseed_root().parent / "phase_switch_symmetry_rollouts_se3"


def load_se3_manifests(root):
    with (root / "se3_fixed_contexts.json").open(encoding="utf-8") as handle:
        fixed = json.load(handle)
    with (root / "se3_experiment.json").open(encoding="utf-8") as handle:
        experiment = json.load(handle)
    with (root / "se3_subsets.json").open(encoding="utf-8") as handle:
        subsets = json.load(handle)
    return fixed, experiment, subsets


def load_se3_subsets(root):
    with (root / "se3_subsets.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def n30_subsets(subsets):
    return [s for s in subsets["subsets"] if s["sample_size"] == 30]


def load_se3_rollout(path, bins):
    """Extract the usable mixed/isolated/baseline split and mixed curves from one arm/seed .h5."""
    import h5py

    with h5py.File(path, "r") as data_file:
        usable_groups = {
            int(data_file[key].attrs["condition_id"]): key
            for key in episode_keys(data_file)
            if usable(data_file[key]) and complete(data_file[key])
        }
        mixed_cond = {
            cid: key
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) == "mixed"
        }
        isolated_cond = {
            cid: key
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) in {"du", "dv", "dw", "roll", "pitch", "yaw"}
        }
        baseline_cond = {
            cid: key
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) == "baseline"
        }
        mixed_cids = sorted(mixed_cond)
        mixed_keys = [mixed_cond[cid] for cid in mixed_cids]
        mixed_contexts = np.asarray([np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys])
        mixed_curves = np.asarray([task_curve_se3(data_file[key], bins) for key in mixed_keys])
    return {
        "mixed_cids": mixed_cids,
        "mixed_keys": mixed_keys,
        "mixed_contexts": mixed_contexts,
        "mixed_curves": mixed_curves,
        "isolated_keys": sorted(isolated_cond.values()),
        "baseline_keys": sorted(baseline_cond.values()),
        "usable_keys": sorted(usable_groups.values()),
    }
