"""Panda gate with a thin rectangular pin and a physically twisted opening."""

from __future__ import annotations

import sqlite3  # noqa: F401 - conda runtime ordering
import sys
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parents[2] / "maniskill_spectrum/phase_switch_symmetry"),
]
from gtsr_coupled_geometry import Gate  # noqa: E402
from mani_skill.utils.registration import register_env  # noqa: E402
from phase_switch_symmetry_env import KeyedCircularPhaseSwitchSE3Env, _material  # noqa: E402


@register_env("GTSRHelicalGate-v1", max_episode_steps=2000)
class HelicalGateEnv(KeyedCircularPhaseSwitchSE3Env):
    def __init__(self, *args, gate_angle=0.8, **kwargs):
        self.gate = Gate(angle=float(gate_angle))
        super().__init__(*args, **kwargs)

    def _build_peg(self):
        builder = self.scene.create_actor_builder()
        pose = sapien.Pose(q=euler2quat(0, -np.pi / 2, 0))
        builder.add_cylinder_collision(pose, radius=0.013, half_length=0.045)
        builder.add_cylinder_visual(pose, radius=0.013, half_length=0.045, material=_material([0.95, 0.42, 0.24, 1]))
        pin = sapien.Pose([0, 0, self.gate.pin_center])
        half = [self.gate.pin_halfx, self.gate.pin_halfy, self.gate.pin_halfheight]
        builder.add_box_collision(pin, half)
        builder.add_box_visual(pin, half, material=_material([0.8, 0.2, 0.1, 1]))
        builder.initial_pose = sapien.Pose([-0.25, -0.18, 0.325])
        return builder.build("helical_pin")

    def _build_socket(self):
        builder = self.scene.create_actor_builder()
        for half, position, yaw in self.gate.walls():
            pose = sapien.Pose(position, euler2quat(0, 0, yaw))
            builder.add_box_collision(pose, half)
            builder.add_box_visual(pose, half, material=_material([0.28, 0.45, 0.58, 1]))
        builder.initial_pose = sapien.Pose([-0.15, 0, 0.08])
        return builder.build_kinematic("helical_gate")

    @property
    def goal_pose(self):
        return self.target_pose_at(self.gate.final_depth, 0, 0, self.gate.angle)

    @property
    def key_clearance_margin(self):
        axis = torch.tensor(self.socket_axis, dtype=torch.float32, device=self.device)
        z = torch.sum((self.peg.pose.p - self.socket.pose.p) * axis, dim=1)
        return self.gate.lower - (z + self.gate.pin_center + self.gate.pin_halfheight)

    def evaluate(self):
        info = super().evaluate()
        info["success"] = (
            (info["obj_to_goal_dist"] < 0.003)
            & (info["axis_angle_err"] < 0.05)
            & (info["key_clearance_margin"] > 0.001)
        )
        return info
