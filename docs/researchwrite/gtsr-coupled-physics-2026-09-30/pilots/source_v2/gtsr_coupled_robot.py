"""Physical validation and robot execution of a fixed known helical task region."""

from __future__ import annotations

import argparse
import json
import sqlite3  # noqa: F401
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT.parent / "maniskill_spectrum/phase_switch_symmetry")]
from gtsr_coupled_geometry import Gate  # noqa: E402


def run_pilot(output, angle, mode, seed, scene_collision=False):
    import gymnasium as gym
    import h5py
    import mani_skill.envs  # noqa: F401
    import mplib
    import sapien
    import torch
    from collect_phase_switch_rollouts import EpisodeFinished, PhaseSwitchTraceWrapper, write_episode
    from gtsr_coupled_env import HelicalGateEnv  # noqa: F401
    from mani_skill.agents.robots.panda import Panda
    from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
    from transforms3d.euler import euler2quat

    if output.with_suffix(".json").exists() or output.with_suffix(".h5").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Panda.gripper_stiffness, Panda.gripper_force_limit = 2.5e3, 150.0
    env = PhaseSwitchTraceWrapper(
        gym.make(
            "GTSRHelicalGate-v1",
            gate_angle=angle,
            num_envs=1,
            obs_mode="state_dict",
            control_mode="pd_joint_pos",
            reward_mode="sparse",
            sim_backend="physx_cpu",
            render_mode=None,
            robot_init_qpos_noise=0.0,
            orientation=[1, 0, 0, 0],
            task_anchor=[-0.15, 0, 0.08],
        )
    )
    env.reset(seed=seed, options={"causal_delta": [0] * 6})
    env.start_trace()
    base, gate = env.unwrapped, Gate(angle=angle)
    planner = PandaArmMotionPlanningSolver(
        env,
        vis=False,
        debug=False,
        base_pose=base.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=0.3,
        joint_acc_limits=0.3,
    )
    plans, targets = [], []
    attached = False
    collision_count = 0
    record = {
        "angle": angle,
        "mode": mode,
        "seed": seed,
        "scene_collision": scene_collision,
        "success": False,
        "status": "failed",
    }
    original_step = env.step

    def guarded_step(action):
        try:
            result = original_step(action)
        except EpisodeFinished:
            if np.linalg.norm(env.trace["states"][-1]["contact_force"]) > 80:
                raise RuntimeError("force_limit") from None
            raise
        if np.linalg.norm(env.trace["states"][-1]["contact_force"]) > 80:
            raise RuntimeError("force_limit")
        return result

    env.step = guarded_step

    def move(pose, refine=5):
        tick = time.monotonic()
        qpos = base.agent.robot.get_qpos().cpu().numpy()[0]
        result = planner.planner.plan_screw(
            np.r_[pose.p, pose.q], qpos, time_step=base.control_timestep, use_attach=attached, use_point_cloud=False
        )
        if result["status"] != "Success":
            result = planner.planner.plan_qpos_to_pose(
                np.r_[pose.p, pose.q],
                qpos,
                time_step=base.control_timestep,
                planning_time=2,
                use_attach=attached,
                use_point_cloud=False,
            )
        plans.append({"phase": env.phase, "status": result["status"], "seconds": time.monotonic() - tick})
        if result["status"] != "Success":
            raise RuntimeError("planning_failed: " + result["status"])
        planner.follow_path(result, refine_steps=refine)

    def peg_move(depth, yaw, refine=5):
        pose = base.target_pose_at(depth, 0, 0, yaw).sp
        targets.append({"depth": depth, "yaw": yaw})
        move(pose * base.peg.pose.sp.inv() * base.agent.tcp.pose.sp, refine)
        error = np.linalg.norm(base.peg.pose.sp.p - pose.p)
        if error > 0.002:
            move(pose * base.peg.pose.sp.inv() * base.agent.tcp.pose.sp, refine)
            error = np.linalg.norm(base.peg.pose.sp.p - pose.p)
        if error > 0.003:
            raise RuntimeError(f"physical_progress_not_reached: {error:.6f}")

    started = time.monotonic()
    try:
        if scene_collision:
            fcl = mplib.collision_detection.fcl
            for index, (half, position, yaw) in enumerate(gate.walls()):
                pose = base.socket.pose.sp * sapien.Pose(position, euler2quat(0, 0, yaw))
                planner.planner.set_normal_object(
                    f"wall_{index}", fcl.CollisionObject(fcl.Box(2 * half), pose.p, pose.q)
                )
            # Same primitives and transforms as the imported physical table/pedestal.
            table_pose = base.table_scene.table.pose.sp * sapien.Pose([0, 0, 0.9196429 / 2])
            planner.planner.set_normal_object(
                "table", fcl.CollisionObject(fcl.Box(2.418, 1.209, 0.9196429), table_pose.p, table_pose.q)
            )
            pedestal = base.pedestal.pose.sp * sapien.Pose([-0.25, -0.18, 0.14])
            planner.planner.set_normal_object(
                "pedestal", fcl.CollisionObject(fcl.Box(0.06, 0.06, 0.28), pedestal.p, pedestal.q)
            )
            record["initial_scene_collisions"] = [
                {key: str(getattr(c, key)) for key in ("object_name1", "object_name2", "link_name1", "link_name2")}
                for c in planner.planner.check_for_env_collision(qpos=base.agent.robot.get_qpos().cpu().numpy()[0])
            ]
            collision_count = 162
        grasp = base.agent.build_grasp_pose(
            approaching=np.array([0, 0, -1]),
            closing=np.array([0, -1, 0]),
            center=base.peg.pose.sp.p + np.array([0, 0, 0.020]),
        )
        env.set_phase("reach")
        move(grasp * sapien.Pose([0, 0, -0.070]))
        env.set_phase("grasp")
        move(grasp)
        planner.close_gripper()
        env.set_phase("lift")
        move(grasp * sapien.Pose([0, 0, -0.095]))
        if scene_collision:
            # Collision hull contains both the thin rectangular pin and the shaft.
            import trimesh

            shaft = trimesh.creation.cylinder(radius=0.013, height=0.090, sections=64)
            pin = trimesh.creation.box(extents=[2 * gate.pin_halfx, 2 * gate.pin_halfy, 2 * gate.pin_halfheight])
            pin.apply_translation([0, 0, gate.pin_center])
            mesh = trimesh.util.concatenate([shaft, pin])
            geom = fcl.BVHModel()
            geom.beginModel(len(mesh.faces), len(mesh.vertices))
            geom.addSubModel(mesh.vertices.tolist(), [np.asarray(t, dtype=np.int32) for t in mesh.faces])
            geom.endModel()
            offset = base.agent.tcp.pose.sp.inv() * base.peg.pose.sp
            planner.planner.update_attached_tool(geom, np.r_[offset.p, offset.q])
            attached = True
        env.set_phase("align_keyed")
        peg_move(gate.start_depth + 0.010, 0)
        peg_move(gate.start_depth + 0.010, 0)
        for depth in np.linspace(gate.start_depth + 0.010, gate.start_depth, 11)[1:]:
            peg_move(depth, 0)
        env.set_phase("enter_key")
        for progress in np.linspace(0, 1, 41)[1:]:
            peg_move(gate.start_depth - gate.travel * progress, gate.angle * progress if mode == "follow" else 0)
        env.set_phase("circular_insert")
        for depth in np.linspace(gate.end_depth, gate.final_depth, 7)[1:]:
            peg_move(depth, gate.angle if mode == "follow" else 0)
        record.update(status="executed", stop_reason="schedule_finished")
    except EpisodeFinished as exc:
        record.update(status="executed", stop_reason=str(exc))
    except Exception as exc:
        record.update(stop_reason=str(exc), exception=repr(exc), traceback=traceback.format_exc())
    finally:
        states = env.trace["states"]
        last = states[-1]
        record.update(
            success=bool(last["success"]) and record["status"] == "executed",
            steps=len(env.trace["actions"]),
            max_force_N=float(np.linalg.norm(np.array([s["contact_force"] for s in states]), axis=1).max()),
            final_error_mm=last["obj_to_goal_dist"] * 1000,
            final_clearance_mm=last["key_clearance_margin"] * 1000,
            plan_seconds=sum(x["seconds"] for x in plans),
            wall_seconds=time.monotonic() - started,
            plans=plans,
            targets=targets,
            scene_collision_objects=collision_count,
        )
        with h5py.File(output.with_suffix(".h5"), "w") as handle:
            write_episode(
                handle.create_group("episode_0"),
                {"generator": "fixed_frame", "causal_delta": [0] * 6},
                env.trace,
                record.get("exception"),
                record["stop_reason"],
            )
        output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
        planner.close()
        env.close()
    print(json.dumps({k: v for k, v in record.items() if k not in ("plans", "targets", "traceback")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--angle", type=float, default=0.8)
    parser.add_argument("--mode", choices=("follow", "straight"), default="follow")
    parser.add_argument("--seed", type=int, default=930100)
    parser.add_argument("--scene-collision", action="store_true")
    args = parser.parse_args()
    run_pilot(args.output, args.angle, args.mode, args.seed, args.scene_collision)


if __name__ == "__main__":
    main()
