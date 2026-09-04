# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Test-side HDF5 loader for the frozen ManiSkill phase-switch multiseed dataset.

This mirrors the frozen extraction logic in ``maniskill_spectrum``'s
``analyze_phase_switch_rollouts.py`` and ``benchmark_phase_switch_baselines.py``
(phase -> task-curve resampling, nominal-frame recovery, empirical isolated
profile, switch diagnostics) so that Sprint E's E-B replication can consume the
pre-registered rollouts without pulling in matplotlib / pandas / transforms3d.

``pose_yaw`` reimplements ``transforms3d.euler.quat2euler(q)[2]`` for a
``[w, x, y, z]`` quaternion (yaw only) to avoid that dependency; the formula is
``atan2(2 (w z + x y), 1 - 2 (y^2 + z^2))``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from tsr.gtsr.profile import wrap_angle

ROTATION_SCALE_M_PER_RAD = 0.03
METRIC_SCALE = np.array([1.0, 1.0, ROTATION_SCALE_M_PER_RAD])
PHASES = {3: "Align keyed", 4: "Enter key", 5: "Unlock yaw", 6: "Circular insert"}
REQUIRED_PHASES = tuple(PHASES)
PHASE_CODES = (3, 4, 5, 6)
SEEDS = (20260818, 20270818, 20280818, 20290818, 20300818)

# Pre-registered context-manifest SHA-256 (scope §5). Kept as an independent
# constant so a tampered ``fixed_contexts.json`` or ``collection_status.json`` is
# caught even when the two would otherwise agree with each other.
CONTEXT_MANIFEST_SHA256 = "3751db801cf49ce698c9b5a74ce36cf0b39897d967ddf2959846a3965b8bd653"

EXPECTED_ISOLATED_CONTEXTS = np.array(
    [
        [0.0, 0.0, np.deg2rad(-30.0)],
        [0.0, 0.0, np.deg2rad(-15.0)],
        [0.0, 0.0, np.deg2rad(15.0)],
        [0.0, 0.0, np.deg2rad(30.0)],
        [-0.015, 0.0, 0.0],
        [0.015, 0.0, 0.0],
        [0.0, -0.015, 0.0],
        [0.0, 0.015, 0.0],
    ],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_yaw(pose):
    pose = np.asarray(pose, dtype=np.float64)
    w, x, y, z = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def episode_keys(data_file):
    return sorted(
        [key for key in data_file if key.startswith("episode_")],
        key=lambda key: int(key.split("_")[-1]),
    )


def complete(group):
    phases = set(int(value) for value in np.asarray(group["solver_phase"]))
    return all(phase in phases for phase in REQUIRED_PHASES)


def usable(group):
    return bool(np.asarray(group["success"])[-1]) and complete(group) and not bool(np.asarray(group["truncated"]).any())


def resample(values, bins):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, bins)
    return np.column_stack([np.interp(target, source, values[:, column]) for column in range(values.shape[1])])


def phase_task_curve(group, phase, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    peg = np.asarray(group["peg_pose"])[indices]
    yaw = np.unwrap(pose_yaw(peg))
    return resample(np.column_stack([peg[:, 0], peg[:, 1], yaw]), bins)


def task_curve(group, bins):
    return np.concatenate([phase_task_curve(group, phase, bins) for phase in PHASE_CODES], axis=0)


def progress_grid(bins):
    phase_progress = np.linspace(0.0, 1.0, bins)
    progress = np.concatenate([(phase_index + phase_progress) / len(PHASE_CODES) for phase_index in range(4)])
    phase_codes = np.repeat(np.arange(4), bins)
    return progress, phase_codes


def fit_scalar_slope(context_values, output_values):
    context_values = np.asarray(context_values, dtype=np.float64)
    design = np.column_stack([np.ones(len(context_values)), context_values])
    return np.linalg.lstsq(design, output_values, rcond=None)[0][1]


def empirical_isolated_profile(data_file, baseline_key, isolated_keys, bins):
    baseline_curve = task_curve(data_file[baseline_key], bins)
    diagonal = np.empty((len(baseline_curve), 3), dtype=np.float64)
    for generator in range(3):
        keys = [key for key in isolated_keys if abs(float(np.asarray(data_file[key]["causal_delta"])[generator])) > 0]
        contexts = [0.0]
        values = [baseline_curve[:, generator]]
        for key in keys:
            contexts.append(float(np.asarray(data_file[key]["causal_delta"])[generator]))
            values.append(task_curve(data_file[key], bins)[:, generator])
        diagonal[:, generator] = fit_scalar_slope(contexts, np.asarray(values))
    return diagonal


def switch_diagnostics(progress, yaw_profile, unlock_start):
    progress = np.asarray(progress, dtype=np.float64)
    yaw_profile = np.asarray(yaw_profile, dtype=np.float64)
    for index in range(max(unlock_start, 1), len(progress)):
        y0, y1 = yaw_profile[index - 1], yaw_profile[index]
        if y0 >= 0.5 and y1 < 0.5:
            x0, x1 = progress[index - 1], progress[index]
            if abs(y1 - y0) < 1e-12 or abs(x1 - x0) < 1e-12:
                location = float(x1)
            else:
                fraction = (0.5 - y0) / (y1 - y0)
                location = float(x0 + fraction * (x1 - x0))
            return {"detected": True, "status": "detected", "location": location}
    if yaw_profile[unlock_start] < 0.5:
        status = "left_censored"
    elif yaw_profile[-1] >= 0.5:
        status = "right_censored"
    else:
        status = "no_downward_crossing"
    return {"detected": False, "status": status, "location": float("nan")}


def metric_errors(prediction, target, yaw_weights=None):
    residual = np.asarray(prediction) - np.asarray(target)
    residual = residual.copy()
    residual[..., 2] = np.vectorize(wrap_angle)(residual[..., 2])
    residual *= METRIC_SCALE
    if yaw_weights is not None:
        residual[..., 2] *= np.asarray(yaw_weights)[None, :]
    point_error = np.linalg.norm(residual, axis=-1) * 1000.0
    return point_error.mean(axis=1), point_error[:, -1]


def multiseed_root() -> Path:
    env = os.environ.get("MANISKILL_SPECTRUM_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "maniskill_spectrum" / "phase_switch_symmetry_multiseed"


@dataclass
class SeedDataset:
    seed: int
    h5_path: Path
    progress: np.ndarray
    phase_codes: np.ndarray
    train_contexts: np.ndarray
    train_curves: np.ndarray
    test_contexts: np.ndarray
    test_curves: np.ndarray
    test_generators: list
    test_condition_ids: list
    nominal_frame_xy: np.ndarray
    nominal_frame_yaw: float
    target_profile: np.ndarray
    mixed_keys: list
    isolated_keys: list
    baseline_key: str


def load_seed_dataset(h5_path, bins=25):
    """Extract the frozen 30/8/1 split and reference profile from one seed .h5."""
    progress, phase_codes = progress_grid(bins)
    with h5py.File(h5_path, "r") as data_file:
        seed = int(data_file.attrs["seed"])
        keys = episode_keys(data_file)
        usable_keys = [key for key in keys if usable(data_file[key])]
        mixed_keys = [key for key in usable_keys if str(data_file[key].attrs.get("generator", "")) == "mixed"]
        isolated_all = [
            key for key in usable_keys if str(data_file[key].attrs.get("generator", "")) in {"yaw", "translation"}
        ]
        baseline_candidates = [
            key for key in isolated_all if np.linalg.norm(np.asarray(data_file[key]["causal_delta"])) < 1e-12
        ]
        if len(mixed_keys) != 30:
            raise RuntimeError(f"expected 30 successful mixed episodes, got {len(mixed_keys)}")
        if len(baseline_candidates) != 1:
            raise RuntimeError("expected exactly one zero-intervention baseline")
        baseline_key = baseline_candidates[0]
        isolated_keys = [key for key in isolated_all if key != baseline_key]
        if len(isolated_keys) != 8:
            raise RuntimeError(f"expected 8 held-out isolated episodes, got {len(isolated_keys)}")
        if sum(data_file[key].attrs.get("generator", "") == "yaw" for key in isolated_keys) != 4:
            raise RuntimeError("expected four nonzero isolated yaw interventions")
        if sum(data_file[key].attrs.get("generator", "") == "translation" for key in isolated_keys) != 4:
            raise RuntimeError("expected four nonzero isolated translation interventions")

        train_contexts = np.asarray([np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys])
        train_curves = np.asarray([task_curve(data_file[key], bins) for key in mixed_keys])
        test_contexts = np.asarray([np.asarray(data_file[key]["causal_delta"]) for key in isolated_keys])
        test_curves = np.asarray([task_curve(data_file[key], bins) for key in isolated_keys])
        test_generators = [str(data_file[key].attrs.get("generator", "")) for key in isolated_keys]
        test_condition_ids = [int(data_file[key].attrs.get("condition_id", -1)) for key in isolated_keys]

        # Frozen intervention integrity (scope §5): the mixed training contexts must
        # be full-rank, unique, and signed along every generator; the held-out
        # isolated contexts must reproduce the pre-registered amplitudes exactly.
        if np.linalg.matrix_rank(train_contexts) != 3:
            raise RuntimeError("mixed training contexts are not full-rank")
        if len(np.unique(train_contexts, axis=0)) != len(train_contexts):
            raise RuntimeError("mixed training contexts are not unique")
        if (np.min(train_contexts, axis=0) >= 0).any() or (np.max(train_contexts, axis=0) <= 0).any():
            raise RuntimeError("mixed training contexts lack signed coverage on some generator")
        isolated_sorted = test_contexts[np.lexsort((test_contexts[:, 0], test_contexts[:, 1], test_contexts[:, 2]))]
        expected_sorted = EXPECTED_ISOLATED_CONTEXTS[
            np.lexsort(
                (EXPECTED_ISOLATED_CONTEXTS[:, 0], EXPECTED_ISOLATED_CONTEXTS[:, 1], EXPECTED_ISOLATED_CONTEXTS[:, 2])
            )
        ]
        if not np.allclose(isolated_sorted, expected_sorted, atol=1e-12):
            raise RuntimeError("held-out isolated amplitudes do not match the frozen manifest")

        target_profile = empirical_isolated_profile(data_file, baseline_key, isolated_keys, bins)

        # Recover the nominal socket frame from mixed conditions only.
        nominal_xy_candidates = []
        nominal_yaw_candidates = []
        for key, context in zip(mixed_keys, train_contexts):
            socket_pose = np.asarray(data_file[key]["socket_pose"])[0]
            nominal_xy_candidates.append(socket_pose[:2] - context[:2])
            nominal_yaw_candidates.append(wrap_angle(float(pose_yaw(socket_pose[None, :])[0]) - context[2]))
        nominal_xy_candidates = np.asarray(nominal_xy_candidates, dtype=np.float64)
        nominal_yaw_candidates = np.asarray(nominal_yaw_candidates, dtype=np.float64)
        nominal_frame_xy = nominal_xy_candidates.mean(axis=0)
        nominal_frame_yaw = float(
            np.arctan2(
                np.sin(nominal_yaw_candidates).mean(),
                np.cos(nominal_yaw_candidates).mean(),
            )
        )
        nominal_xy_spread = float(np.max(np.linalg.norm(nominal_xy_candidates - nominal_frame_xy, axis=1)))
        nominal_yaw_spread = float(
            np.max(np.abs([wrap_angle(yaw - nominal_frame_yaw) for yaw in nominal_yaw_candidates]))
        )
        if nominal_xy_spread > 1e-6 or nominal_yaw_spread > 1e-6:
            raise RuntimeError(
                "mixed conditions do not imply a consistent nominal socket frame: "
                f"xy spread={nominal_xy_spread}, yaw spread={nominal_yaw_spread}"
            )

    return SeedDataset(
        seed=seed,
        h5_path=Path(h5_path),
        progress=progress,
        phase_codes=phase_codes,
        train_contexts=train_contexts,
        train_curves=train_curves,
        test_contexts=test_contexts,
        test_curves=test_curves,
        test_generators=test_generators,
        test_condition_ids=test_condition_ids,
        nominal_frame_xy=nominal_frame_xy,
        nominal_frame_yaw=nominal_frame_yaw,
        target_profile=target_profile,
        mixed_keys=mixed_keys,
        isolated_keys=isolated_keys,
        baseline_key=baseline_key,
    )


def load_collection_status(root: Path) -> dict:
    with (root / "collection_status.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def load_experiment(root: Path) -> dict:
    with (root / "experiment.json").open(encoding="utf-8") as handle:
        return json.load(handle)
