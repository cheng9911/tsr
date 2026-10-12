"""Read every geometry-probe attempt and test the declared axial-symmetry proxy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gtsr_unified_eval import (  # noqa: E402
    ARMS,
    SE3_SEEDS,
    SISTER,
    load_se3_isolated,
    load_se3_rollout,
    progress_grid_se3,
    residual,
    se3_from_pose6_batched,
    se3_rollouts_root,
    sha,
    task_proxy,
)


def read_probes(path):
    """Keep all attempts, including deliberate early stops and planning failures."""
    rows = []
    with h5py.File(path, "r") as handle:
        root = {k: v.item() if isinstance(v, np.generic) else v for k, v in handle.attrs.items()}
        for name in sorted(handle, key=lambda key: int(key.split("_")[-1])):
            ep = handle[name]
            row = {k: v.item() if isinstance(v, np.generic) else v for k, v in ep.attrs.items()}
            pose = ep["peg_pose"][-1]
            quat_xyzw = pose[[4, 5, 6, 3]]
            force = np.linalg.norm(ep["contact_force"][1:], axis=-1)
            done = ep["terminated"][:] | ep["truncated"][:]
            row.update(
                episode=name,
                steps=len(ep["actions"]),
                final_success=bool(ep["success"][-1]),
                final_clearance_mm=float(ep["key_clearance_margin"][-1] * 1000),
                maximum_force_N=float(force.max()),
                contact_frames=int((force > 1e-3).sum()),
                final_yaw_deg=float(Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)[2]),
                final_position_error_mm=float(ep["obj_to_goal_dist"][-1] * 1000),
                final_axis_error_deg=float(np.rad2deg(ep["axis_angle_err"][-1])),
                phases=np.unique(ep["solver_phase"][:]).tolist(),
                post_done_actions=0 if not done.any() else len(done) - 1 - int(np.flatnonzero(done)[0]),
            )
            rows.append(row)
    assert len(rows) == root["episode_count"] == 7
    assert all(r["post_done_actions"] == 0 for r in rows)
    return {"source_attributes": root, "attempts": rows}


def symmetry_counterfactual():
    """Numerical change in body yaw, not a simulation of a modified controller."""
    _, phases = progress_grid_se3(25)
    rows = []
    for arm in ARMS:
        free = phases >= 2 if arm == "keyed" else np.ones(len(phases), dtype=bool)
        for seed in SE3_SEEDS:
            path = se3_rollouts_root() / f"{arm}_seed_{seed}.h5"
            ds, iso = load_se3_rollout(path, 25), load_se3_isolated(path, 25)
            for split, contexts, curves, labels in (
                ("isolated_yaw", iso["isolated_contexts"], iso["isolated_curves"], iso["isolated_generators"]),
                ("mixed", ds["mixed_contexts"], ds["mixed_curves"], ["yaw"] * 60),
            ):
                keep = np.asarray(labels) == "yaw"
                contexts, curves = contexts[keep], curves[keep]
                observed = se3_from_pose6_batched(curves)
                angle = contexts[:, 5, None] * free[None, :]
                added = np.tile(np.eye(4), (*angle.shape, 1, 1))
                rotvec = np.zeros((angle.size, 3))
                rotvec[:, 2] = angle.ravel()
                added[..., :3, :3] = Rotation.from_rotvec(rotvec).as_matrix().reshape(*angle.shape, 3, 3)
                alternative = observed @ added
                proxy = task_proxy(alternative, observed, arm, phases)
                complete = residual(alternative, observed)
                assert proxy.max() < 1e-8
                rows.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        split=split,
                        n_conditions=len(contexts),
                        task_proxy_max_mm=float(proxy.max()),
                        complete_pose_all_phases_mm=float(complete.mean()),
                        complete_pose_free_phases_mm=float(complete[:, free].mean()),
                    )
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    probe = SISTER / "phase_switch_symmetry_rollouts/keyed_circular_geometry_probes_v2.h5"
    sources = [
        Path(__file__),
        Path(__file__).with_name("gtsr_unified_eval.py"),
        probe,
        *[se3_rollouts_root() / f"{a}_seed_{s}.h5" for a in ARMS for s in SE3_SEEDS],
        *[
            SISTER / "phase_switch_symmetry" / f
            for f in (
                "collect_phase_switch_geometry_probes.py",
                "collect_phase_switch_rollouts.py",
                "phase_switch_symmetry_env.py",
                "validate_phase_switch_geometry_probes.py",
            )
        ],
    ]
    result = {
        "source_sha256": {str(p): sha(p) for p in sources},
        "physical_probes": read_probes(probe),
        "kinematic_counterfactual": symmetry_counterfactual(),
        "limitations": [
            "Existing simulator geometry probes; no new physical rollout was executed.",
            "Adaptive retries: retain every attempt, do not estimate an unbiased success rate.",
            "Four preclear probes intentionally stop after enter_key; final_success is not their endpoint.",
            "Positive clearance alone does not certify success; failed attempt 4 never reached enter_key.",
            "The kinematic counterfactual is evaluated under declared phase symmetry, not collision dynamics.",
            "The learned yaw profile is demonstrator response, not uniquely identified task necessity.",
        ],
    }
    (args.out / "task_preference.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"attempts": len(result["physical_probes"]["attempts"]), "counterfactual_cells": 12}))


if __name__ == "__main__":
    main()
