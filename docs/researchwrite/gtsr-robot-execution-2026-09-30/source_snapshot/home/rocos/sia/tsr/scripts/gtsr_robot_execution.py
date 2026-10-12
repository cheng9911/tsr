"""Frozen-response Panda/PhysX execution bridge, with every attempt retained.

Run prepare with tsr-dev; run batch/worker with maniskill_download. Simulation
imports are deliberately lazy so the protocol and geometry can be unit tested.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sqlite3  # noqa: F401 - load conda libstdc++ before torch/sapien
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SISTER = ROOT.parent / "maniskill_spectrum" / "phase_switch_symmetry"
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(SISTER)]

from tsr.gtsr.charts import PoEChart  # noqa: E402
from tsr.gtsr.domains import CoordinateDomain  # noqa: E402
from tsr.gtsr.metrics import se3_exp, se3_log  # noqa: E402
from tsr.gtsr.profile6 import se3_from_pose6  # noqa: E402
from tsr.gtsr.region import GTSR  # noqa: E402
from tsr.tsr import TSR  # noqa: E402

RESPONSES = ("fixed", "frame", "pdiag", "full_affine", "tpgmm_se3")
PHASE_NAMES = ("align_keyed", "enter_key", "unlock_yaw", "circular_insert")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def schedule(step=0.003):
    """(phase, within-phase progress, nominal peg depth, refinement steps)."""
    pre, clear, final = 0.125, 0.078, 0.060
    rows = [(0, 1.0, pre, 5), (0, 1.0, pre, 5)]
    n = int(np.ceil((pre - clear) / step))
    rows += [(1, i / n, max(pre - i * step, clear), 3) for i in range(1, n + 1)]
    rows += [(2, rho, clear, 4) for rho in (0.25, 0.5, 0.75, 1.0)]
    rows += [(3, rho, clear + rho * (final - clear), 5) for rho in (0.25, 0.5, 0.75, 1.0)]
    rows += [(3, 1.0, final, 10)]
    return rows


def interpolate_action(actions, phase, rho):
    index = float(rho) * 24
    lo, hi = int(np.floor(index)), int(np.ceil(index))
    a, b = actions[phase * 25 + lo], actions[phase * 25 + hi]
    return a @ se3_exp((index - lo) * se3_log(np.linalg.inv(a) @ b))


def region_pose(center, geometry, coordinate, free_yaw, width=0.0005, yaw_width=np.pi / 6):
    """Equivalent production TSR and PoE regions: local xyz plus axial yaw."""
    bounds = np.array([[-width, width]] * 3 + [[-yaw_width, yaw_width] if free_yaw else [0.0, 0.0]])
    coordinate = np.asarray(coordinate, dtype=float)
    if coordinate.shape != (4,) or not np.all(np.isfinite(coordinate)):
        raise ValueError("region coordinates must be a finite four-vector")
    if np.any(coordinate < bounds[:, 0]) or np.any(coordinate > bounds[:, 1]):
        raise ValueError("region coordinates outside bounds")
    if geometry == "tsr":
        bw = np.zeros((6, 2))
        bw[:3], bw[5] = bounds[:3], bounds[3]
        return TSR(T0_w=center, Bw=bw).to_transform(np.r_[coordinate[:3], 0, 0, coordinate[3]])
    if geometry != "gtsr":
        raise ValueError(f"unknown geometry {geometry}")
    axes = np.zeros((4, 6))
    axes[:3, :3] = np.eye(3)
    axes[3, 5] = 1
    chart = PoEChart(axes, representation="space")
    domain = CoordinateDomain.from_bounds(bounds, chart.periodic, chart.periods)
    return GTSR(T0_w=center, chart=chart, domain=domain).to_transform(coordinate)


def input_sources(directory):
    sources = [Path(__file__), directory / "protocol.json", directory / "targets.npz"]
    sources += sorted((ROOT / "src" / "tsr").rglob("*.py"))
    sources += [
        SISTER / name
        for name in (
            "phase_switch_symmetry_env.py",
            "collect_phase_switch_rollouts.py",
            "collect_phase_switch_rotated.py",
        )
    ]
    return {str(p.resolve()): sha(p) for p in sources}


def validate_preparation(directory):
    prepared = json.loads((directory / "preparation.json").read_text())
    for name, field in (("protocol.json", "config_sha256"), ("targets.npz", "targets_sha256")):
        if sha(directory / name) != prepared[field]:
            raise ValueError(f"prepared input drift: {name}")


def check_sources(expected):
    changed = [path for path, value in expected.items() if sha(path) != value]
    if changed:
        raise ValueError(f"source/input changed during execution: {changed}")


def preserve_interrupted_attempt(out, cell):
    """A started attempt is never implicitly retried, including coordinator crashes."""
    key = cell_key(cell)
    target = out / f"{key}.json"
    if target.exists():
        return True
    if (out / f"{key}.started.json").exists():
        save_json(
            target,
            {
                **cell,
                "status": "failed",
                "success": False,
                "stop_reason": "interrupted_attempt_not_retried",
                "unverified_partial_trace": (out / f"{key}.h5").exists(),
            },
        )
        return True
    return False


def prepare(directory):
    from scripts.gtsr_unified_eval import predictions
    from scripts.gtsr_unified_eval import prepare as prepare_training
    from tests.gtsr._maniskill_data import multiseed_root
    from tests.gtsr._maniskill_data_se3 import load_se3_subsets, n30_subsets, progress_grid_se3, se3_rollouts_root

    config = json.loads((directory / "protocol.json").read_text())
    if (directory / "targets.npz").exists():
        raise FileExistsError("prepared inputs are immutable; use a new directory")
    subset = next(
        s for s in n30_subsets(load_se3_subsets(multiseed_root())) if int(s["subset_id"]) == config["source_subset"]
    )
    progress, phases = progress_grid_se3(25)
    artifacts = ROOT / "docs/researchwrite/gtsr-unified-eval-2026-09-30/artifacts/pdiag"
    output, references, metadata = [], [], {}
    for arm in config["arms"]:
        data = prepare_training(arm, config["source_seed"], subset)
        data["eval_contexts"] = np.array([c["pose6"] for c in config["contexts"]])
        arm_actions = []
        for method in RESPONSES:
            _, actions, model_meta = predictions(
                method,
                data,
                progress,
                phases,
                artifacts,
                arm,
                config["source_seed"],
                subset["subset_id"],
            )
            arm_actions.append(actions)
            metadata[f"{arm}/{method}"] = model_meta
        output.append(arm_actions)
        references.append(data["reference"])
        path = se3_rollouts_root() / f"{arm}_seed_{config['source_seed']}.h5"
        metadata[f"{arm}/source"] = {"path": str(path), "sha256": sha(path), "train_ids": data["train_ids"]}
    np.savez_compressed(
        directory / "targets.npz",
        actions=np.array(output),
        reference=np.array(references),
        responses=RESPONSES,
        contexts=np.array([c["pose6"] for c in config["contexts"]]),
    )
    save_json(
        directory / "preparation.json",
        {
            "source_metadata": metadata,
            "targets_sha256": sha(directory / "targets.npz"),
            "config_sha256": sha(directory / "protocol.json"),
            "source_script_sha256": sha(__file__),
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn", "h5py", "threadpoolctl")
            },
            "loaded_project_sources": {
                str(Path(module.__file__).resolve()): sha(module.__file__)
                for module in list(sys.modules.values())
                if getattr(module, "__file__", None)
                and str(module.__file__).endswith(".py")
                and (
                    Path(module.__file__).resolve().is_relative_to(ROOT)
                    or Path(module.__file__).resolve().is_relative_to(SISTER)
                )
            },
        },
    )
    print("Prepared frozen response actions", np.array(output).shape, flush=True)


def cell_key(cell):
    return f"{cell['arm']}_{cell['context']}_{cell['seed']}_{cell['method']}"


def worker(directory, out, cell):
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401
    import phase_switch_symmetry_env as task
    import sapien
    import torch
    from collect_phase_switch_rollouts import EpisodeFinished, PhaseSwitchTraceWrapper, write_episode
    from mani_skill.agents.robots.panda import Panda
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

    config = json.loads((directory / "protocol.json").read_text())
    config_context = next(c for c in config["contexts"] if c["name"] == cell["context"])
    arm_index = config["arms"].index(cell["arm"])
    context_index = config["contexts"].index(config_context)
    geometry, response = cell["method"].split("_", 1)
    with np.load(directory / "targets.npz", allow_pickle=False) as loaded:
        actions = loaded["actions"][arm_index, RESPONSES.index(response), context_index]
        reference = loaded["reference"][arm_index]
    key = cell_key(cell)
    np.random.seed(cell["seed"])
    torch.manual_seed(cell["seed"])
    rng = np.random.default_rng(cell["seed"] + 17)
    Panda.gripper_stiffness, Panda.gripper_force_limit = 2.5e3, 150.0
    env_id = "KeyedCircularPhaseSwitchSE3-v1" if cell["arm"] == "keyed" else "CircularPhaseSwitchSE3-v1"
    started = time.monotonic()
    env = planner = None
    planning_log, target_log = [], []
    record = {**cell, "status": "failed", "success": False, "stop_reason": "initializing"}
    try:
        env = PhaseSwitchTraceWrapper(
            gym.make(
                env_id,
                num_envs=1,
                obs_mode="state_dict",
                control_mode="pd_joint_pos",
                reward_mode="sparse",
                sim_backend="physx_cpu",
                render_mode=None,
                robot_init_qpos_noise=config["robot_init_qpos_noise"],
                orientation=config["orientation_wxyz"],
                task_anchor=config["task_anchor"],
                max_episode_steps=config["maximum_steps"],
            )
        )
        env.reset(seed=cell["seed"], options={"causal_delta": config_context["pose6"]})
        env.start_trace()
        base = env.unwrapped
        # Targets are precomputed for prescribed contexts, not from noisy perception.
        # Reject a reset that did not instantiate the same physical relation.
        actual_socket = base.socket.pose.sp.to_transformation_matrix()
        expected_socket = reference @ se3_from_pose6(config_context["pose6"])
        np.testing.assert_allclose(actual_socket, expected_socket, atol=2e-7, rtol=0)
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=False,
            base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_vel_limits=config["planner_joint_velocity_scale"],
            joint_acc_limits=config["planner_joint_acceleration_scale"],
        )
        # mplib uses its own random stream; seed the planner when supported.
        if hasattr(planner.planner, "set_seed"):
            planner.planner.set_seed(cell["seed"])
            record["mplib_seeded"] = True
        else:
            record["mplib_seeded"] = False

        original_step = env.step

        def monitored_step(action):
            try:
                value = original_step(action)
            except EpisodeFinished:
                if np.linalg.norm(env.trace["states"][-1]["contact_force"]) > config["force_abort_N"]:
                    raise RuntimeError("force_abort") from None
                raise
            if np.linalg.norm(env.trace["states"][-1]["contact_force"]) > config["force_abort_N"]:
                raise RuntimeError("force_abort")
            return value

        env.step = monitored_step

        def plan_pose(pose, deadline):
            qpos = base.agent.robot.get_qpos().cpu().numpy()[0]
            pose7 = np.r_[pose.p, pose.q]
            for algorithm in ("screw", "rrt_connect"):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                tick = time.monotonic()
                if algorithm == "screw":
                    result = planner.planner.plan_screw(
                        pose7,
                        qpos,
                        time_step=base.control_timestep,
                        use_point_cloud=False,
                    )
                else:
                    result = planner.planner.plan_qpos_to_pose(
                        pose7,
                        qpos,
                        time_step=base.control_timestep,
                        use_point_cloud=False,
                        planning_time=remaining,
                        wrt_world=True,
                    )
                ended = time.monotonic()
                planning_log.append(
                    {
                        "phase": env.phase,
                        "algorithm": algorithm,
                        "seconds": ended - tick,
                        "result": result["status"],
                        "within_budget": ended <= deadline,
                    }
                )
                if result["status"] == "Success" and ended <= deadline:
                    return result
            return None

        def move(pose, refine=0):
            result = plan_pose(pose, time.monotonic() + config["planning_seconds_per_waypoint"])
            if result is None:
                raise RuntimeError("planning_failed_or_budget_exhausted")
            planner.follow_path(result, refine_steps=refine)

        axis = base.insertion_axis
        grasp = base.agent.build_grasp_pose(
            approaching=-axis,
            closing=base.Q_mat @ np.array([0, 1, 0]),
            center=base.peg.pose.sp.p + 0.020 * axis,
        )
        env.set_phase("reach")
        move(grasp * sapien.Pose([0, 0, -0.070]))
        env.set_phase("grasp")
        move(grasp)
        planner.close_gripper()
        env.set_phase("lift")
        move(grasp * sapien.Pose([0, 0, -0.095]))

        assert np.isclose(task.PRE_ENTRY_PEG_Z, 0.125)
        assert np.isclose(task.KEY_CLEAR_PEG_Z, 0.078)
        assert np.isclose(task.FINAL_PEG_Z, 0.060)
        for waypoint, (phase, rho, depth, refine) in enumerate(schedule(config["insertion_step_m"])):
            env.set_phase(PHASE_NAMES[phase])
            nominal = reference.copy()
            nominal[:3, 3] += reference[:3, 2] * depth
            center = interpolate_action(actions, phase, rho) @ nominal
            free_yaw = cell["arm"] == "circular_honest" or phase >= 2
            width, yaw_width = config["region_translation_halfwidth_m"], config["region_free_yaw_halfwidth_rad"]
            coordinates = np.zeros((config["region_candidates"], 4))
            coordinates[1:, :3] = rng.uniform(-width, width, (len(coordinates) - 1, 3))
            if free_yaw:
                coordinates[1:, 3] = rng.uniform(-yaw_width, yaw_width, len(coordinates) - 1)
            deadline = time.monotonic() + config["planning_seconds_per_waypoint"]
            result = None
            for candidate, coordinate in enumerate(coordinates):
                target = region_pose(center, geometry, coordinate, free_yaw, width, yaw_width)
                quat = Rotation.from_matrix(target[:3, :3]).as_quat()
                peg_target = sapien.Pose(target[:3, 3], quat[[3, 0, 1, 2]])
                tcp_target = peg_target * base.peg.pose.sp.inv() * base.agent.tcp.pose.sp
                target_log.append(
                    {
                        "waypoint": waypoint,
                        "phase": phase,
                        "rho": rho,
                        "candidate": candidate,
                        "coordinate": coordinate.tolist(),
                        "peg_target": target.tolist(),
                    }
                )
                result = plan_pose(tcp_target, deadline)
                if result is not None:
                    break
            if result is None:
                raise RuntimeError("planning_failed_or_budget_exhausted")
            planner.follow_path(result, refine_steps=refine)
        record.update(status="executed", stop_reason="schedule_finished")
    except EpisodeFinished as exc:
        record.update(status="executed", stop_reason=str(exc))
    except Exception as exc:
        record.update(stop_reason=str(exc), exception=repr(exc), traceback=traceback.format_exc())
    finally:
        if env is not None and env.trace is not None:
            states = env.trace["states"]
            last = states[-1]
            forces = np.linalg.norm(np.array([s["contact_force"] for s in states]), axis=1)
            record.update(
                env_success=bool(last["success"]),
                success=bool(last["success"]) and record["status"] == "executed",
                steps=len(env.trace["actions"]),
                final_position_error_mm=last["obj_to_goal_dist"] * 1000,
                final_axis_error_rad=last["axis_angle_err"],
                max_net_contact_force_N=float(forces.max()),
                reached_phases=sorted({s["solver_phase"] for s in states}),
            )
            with h5py.File(out / f"{key}.h5", "w") as handle:
                write_episode(
                    handle.create_group("episode_0"),
                    {"generator": cell["context"], "causal_delta": config_context["pose6"]},
                    env.trace,
                    record.get("exception"),
                    record["stop_reason"],
                )
                handle.attrs["cell_json"] = json.dumps(cell)
        if planner is not None:
            planner.close()
        if env is not None:
            env.close()
    record.update(
        wall_seconds=time.monotonic() - started,
        planning_seconds=sum(x["seconds"] for x in planning_log),
        planning_calls=planning_log,
        targets=target_log,
    )
    save_json(out / f"{key}.json", record)
    print(
        json.dumps({k: v for k, v in record.items() if k not in ("planning_calls", "targets", "traceback")}), flush=True
    )


def batch(directory, smoke=False):
    validate_preparation(directory)
    config = json.loads((directory / "protocol.json").read_text())
    out = directory / ("smoke" if smoke else "execution")
    out.mkdir(exist_ok=True)
    sources = input_sources(directory)
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "mani_skill", "sapien", "mplib")}
    manifest = {
        "sources": sources,
        "packages": packages,
        "python": sys.version,
        "preparation_sha256": sha(directory / "preparation.json"),
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("input/source/environment drift; use a new output directory")
    else:
        save_json(manifest_path, manifest)
    seeds = [config["smoke_seed"]] if smoke else config["execution_seeds"]
    contexts = ["nominal"] if smoke else [c["name"] for c in config["contexts"]]
    methods = ["tsr_frame", "gtsr_pdiag"] if smoke else config["methods"]
    cells = [
        dict(arm=a, context=c, seed=s, method=m)
        for a in config["arms"]
        for c in contexts
        for s in seeds
        for m in methods
    ]
    np.random.default_rng(config["method_order_seed"]).shuffle(cells)
    save_json(out / "cells.json", cells)
    for index, cell in enumerate(cells):
        check_sources(sources)
        key = cell_key(cell)
        target = out / f"{key}.json"
        if preserve_interrupted_attempt(out, cell):
            continue
        save_json(out / f"{key}.started.json", {"cell": cell, "time_unix": time.time()})
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "worker",
            "--directory",
            str(directory),
            "--out",
            str(out),
            "--cell",
            json.dumps(cell),
        ]
        print(f"[{index + 1}/{len(cells)}] {key}", flush=True)
        started = time.monotonic()
        with (out / f"{key}.log").open("w") as log:
            try:
                completed = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=config["episode_timeout_seconds"],
                    env=dict(os.environ),
                )
                reason = f"worker_exit_{completed.returncode}"
            except subprocess.TimeoutExpired:
                reason = "process_timeout"
        if not target.exists():
            save_json(
                target,
                {
                    **cell,
                    "status": "failed",
                    "success": False,
                    "stop_reason": reason,
                    "wall_seconds": time.monotonic() - started,
                },
            )
        check_sources(sources)
        result = json.loads(target.read_text())
        print(
            f"  success={result['success']} stop={result['stop_reason']} wall={result['wall_seconds']:.1f}s", flush=True
        )
    save_json(
        out / "completion.json",
        {"planned": len(cells), "recorded": sum((out / f"{cell_key(c)}.json").exists() for c in cells)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "batch", "worker"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cell")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.command == "prepare":
        prepare(directory)
    elif args.command in ("smoke", "batch"):
        batch(directory, smoke=args.command == "smoke")
    else:
        worker(directory, args.out, json.loads(args.cell))


if __name__ == "__main__":
    main()
