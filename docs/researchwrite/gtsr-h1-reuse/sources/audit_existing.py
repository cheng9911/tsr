"""Read-only H1 inventory: no simulation, fitting, or writes to either repo.

Run with tsr-dev; JSON is emitted on stdout for an explicitly reviewed snapshot.
"""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from tsr.gtsr.metrics import se3_log

ROOT = Path("/home/rocos/sia/maniskill_spectrum")
DATA = ROOT / "phase_switch_symmetry_multiseed"
REL = DATA / "libero_relation_suite"
SEEDS = (20260818, 20270818, 20280818)


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def historical_pose6_transform(pose):
    """Historical relation-suite convention, not canonical TSR Euler order."""
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("XYZ", pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def audit():
    validation = read_json(DATA / "geometry_transfer/geometry_transfer_validation.json")
    experiment_path = REL / "libero_relation_suite_experiment.json"
    profile_path = REL / "libero_relation_suite_profiles.npz"
    contexts_path = REL / "libero_relation_suite_fixed_contexts.json"
    subsets_path = REL / "libero_relation_suite_subsets.json"
    experiment, manifest, subsets = map(read_json, (experiment_path, contexts_path, subsets_path))
    assert digest(experiment_path) == validation["experiment_sha256"]
    assert digest(profile_path) == validation["profile_sha256"]
    assert digest(contexts_path) == experiment["context_manifest_sha256"] == subsets["context_manifest_sha256"]
    pairs = validation["pairs"]
    tasks = sorted({p[k] for p in pairs for k in ("source_task", "target_task")})
    task_specs = [s for s in experiment["task_specs"] if s["task_key"] in tasks]
    files = [experiment_path, profile_path, contexts_path, subsets_path]
    files += [
        ROOT / "phase_switch_symmetry" / name
        for name in (
            "benchmark_geometry_transfer.py",
            "collect_libero_relation_suite.py",
            "benchmark_libero_relation_suite.py",
            "libero_relation_suite_specs.py",
            "benchmark_phase_switch_baselines.py",
            "phase_switch_se3_baselines.py",
        )
    ]
    reports = {}
    for folder in ("geometry_transfer", "geometry_transfer_tpgmm_n8"):
        base = DATA / folder
        val = read_json(base / "geometry_transfer_validation.json")
        with (base / "geometry_transfer_fits.csv").open() as f:
            rows = list(csv.DictReader(f))
        with (base / "geometry_transfer_summary.csv").open() as f:
            summaries = list(csv.DictReader(f))
        assert len(rows) == val["fit_count"]
        failures = sum(r["fit_success"] != "True" for r in rows)
        assert failures == val["fit_failure_count"]
        for s in summaries:
            selected = [
                r
                for r in rows
                if r["method"] == s["method"] and r["sample_size"] == s["sample_size"] and r["fit_success"] == "True"
            ]
            assert len(selected) == int(s["count"])
            assert np.isclose(
                np.mean([r["m_transfer_correct"] == "True" for r in selected]),
                float(s["m_transfer_accuracy"]),
            )
            for field in ("e_alpha", "e_task"):
                assert np.isclose(
                    np.mean([float(r[field]) for r in selected]), float(s[field + "_mean"]), atol=1e-15, rtol=1e-12
                )

        def cell_key(row):
            return tuple(row[k] for k in ("pair_key", "seed", "sample_size", "repeat"))

        transfer_rows = {cell_key(r): r for r in rows if r["method"].startswith("Ours transfer:")}
        comparison = []
        for row in rows:
            if row["method"] == "Pdiag finite target scratch":
                paired = transfer_rows[cell_key(row)]
                comparison.append(
                    dict(
                        cell=list(cell_key(row)),
                        difference_transfer_minus_scratch=float(paired["e_task"]) - float(row["e_task"]),
                    )
                )
        reports[folder] = dict(
            rows=len(rows),
            failures=failures,
            repeats=val["repeats"],
            sample_sizes=val["sample_sizes"],
            split_seed=val["split_seed"],
            summary=summaries,
            paired_pdiag_differences=comparison,
        )
        files += [
            base / name
            for name in (
                "geometry_transfer_fits.csv",
                "geometry_transfer_summary.csv",
                "geometry_transfer_by_pair.csv",
                "geometry_transfer_validation.json",
            )
        ]
    source_profiles = {}
    with np.load(profile_path, allow_pickle=False) as data:
        for task in sorted({p["source_task"] for p in pairs}):
            mask = (data["task_key"] == task) & (data["sample_size"] == 30)
            assert int(mask.sum()) == 18
            source_profiles[task] = dict(
                profile_count=int(mask.sum()),
                seed_ids=sorted(set(data["seed"][mask].tolist())),
                subset_ids=sorted(set(data["subset_id"][mask].tolist())),
            )
        grid = dict(
            shape=list(data["profile"].shape),
            fields=data.files,
            phase_codes=sorted(set(data["phase_codes"].tolist())),
            bins_per_phase=25,
        )
    source_ids = sorted(
        {cid for s in subsets["subsets"] if s["sample_size"] == 30 for cid in s["source_condition_ids"]}
    )
    coords = {}
    for row in manifest["conditions"]:
        pose = row["causal_delta"]
        transform = historical_pose6_transform(pose)
        coords[row["condition_id"]] = se3_log(transform)
    training = np.array([coords[cid] for cid in source_ids])
    lower, upper = training.min(axis=0), training.max(axis=0)
    outside = [
        r["condition_id"]
        for r in manifest["conditions"]
        if not (
            np.all(coords[r["condition_id"]] >= lower - 1e-12) and np.all(coords[r["condition_id"]] <= upper + 1e-12)
        )
    ]
    raw = {}
    for task in tasks:
        entries = []
        for seed in SEEDS:
            path = ROOT / "phase_switch_symmetry_rollouts_libero_relation_suite" / task / f"{task}_seed_{seed}.h5"
            files.append(path)
            trajectory_hash = hashlib.sha256()
            with h5py.File(path, "r") as data:
                groups = sorted(k for k in data if k.startswith("episode_"))
                ids = []
                usable = 0
                for key in groups:
                    group = data[key]
                    cid = int(group.attrs["condition_id"])
                    ids.append(cid)
                    row = next(r for r in manifest["conditions"] if r["condition_id"] == cid)
                    np.testing.assert_array_equal(group["causal_delta"][:], row["causal_delta"])
                    trajectory_hash.update(group["object_pose"][:].tobytes())
                    usable += bool(group["success"][-1]) and not bool(np.any(group["truncated"][:]))
                assert sorted(ids) == list(range(75))
                entries.append(
                    dict(
                        seed=seed,
                        episodes=len(groups),
                        usable_flag_count=usable,
                        object_pose_sha256=trajectory_hash.hexdigest(),
                    )
                )
        raw[task] = dict(files=entries, unique_scored_trajectory_hashes=len({e["object_pose_sha256"] for e in entries}))
    return dict(
        schema_version=1,
        mode="read_only_inventory_no_fit_no_rollout",
        root=str(ROOT),
        pairs=pairs,
        task_specs=task_specs,
        reports=reports,
        source_profiles=source_profiles,
        profile_grid=grid,
        raw_data=raw,
        proposed_source_range=dict(
            coordinate_convention="Log(Trans(xyz) Rx(roll) Ry(pitch) Rz(yaw))",
            source_condition_ids=source_ids,
            lower=lower.tolist(),
            upper=upper.tolist(),
            coordinate_scales=np.maximum(abs(lower), abs(upper)).tolist(),
            tolerance=1e-12,
            out_of_support_condition_ids=outside,
        ),
        sha256={str(p.relative_to(ROOT)): digest(p) for p in files},
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
