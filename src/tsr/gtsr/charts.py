# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generator charts for the G-TSR geometry kernel."""

from abc import ABC, abstractmethod

import numpy as np
from numpy import pi

from ..utils import EPSILON, wrap_to_interval


class GeneratorChart(ABC):
    """Base protocol: owns coordinate topology, maps coordinates to local poses."""

    coord_names: tuple = ()
    periodic: tuple = ()
    periods: tuple = ()

    @property
    def dim(self):
        return len(self.coord_names)

    @abstractmethod
    def forward(self, z):
        """Map a coordinate vector ``z`` to a 4x4 local transform."""


class CanonicalTSRChart(GeneratorChart):
    """The canonical chart: ``Trans(x,y,z) Rz(yaw) Ry(pitch) Rx(roll)``."""

    coord_names = ("x", "y", "z", "roll", "pitch", "yaw")
    periodic = (False, False, False, True, True, True)
    periods = (None, None, None, 2 * pi, 2 * pi, 2 * pi)

    def forward(self, z):
        return self.xyzrpy_to_trans(z)

    @staticmethod
    def xyzrpy_to_trans(xyzrpy):
        xyzrpy = np.asarray(xyzrpy, dtype=float)
        trans = np.zeros((4, 4))
        trans[3, 3] = 1.0
        trans[0:3, 3] = xyzrpy[0:3]
        trans[0:3, 0:3] = CanonicalTSRChart.rpy_to_rot(xyzrpy[3:6])
        return trans

    @staticmethod
    def rpy_to_rot(rpy):
        rpy = np.asarray(rpy, dtype=float)
        rot = np.zeros((3, 3))
        r, p, y = rpy[0], rpy[1], rpy[2]
        rot[0, 0] = np.cos(p) * np.cos(y)
        rot[1, 0] = np.cos(p) * np.sin(y)
        rot[2, 0] = -np.sin(p)
        rot[0, 1] = np.sin(r) * np.sin(p) * np.cos(y) - np.cos(r) * np.sin(y)
        rot[1, 1] = np.sin(r) * np.sin(p) * np.sin(y) + np.cos(r) * np.cos(y)
        rot[2, 1] = np.sin(r) * np.cos(p)
        rot[0, 2] = np.cos(r) * np.sin(p) * np.cos(y) + np.sin(r) * np.sin(y)
        rot[1, 2] = np.cos(r) * np.sin(p) * np.sin(y) - np.sin(r) * np.cos(y)
        rot[2, 2] = np.cos(r) * np.cos(p)
        return rot

    def coordinates(self, T_local, domain):
        """Inverse map: local transform -> coordinate vector (bounds-consistent rpy)."""
        xyz = T_local[0:3, 3]
        rot = T_local[0:3, 0:3]
        rpy_bounds = np.column_stack((domain.cont_lower[3:6], domain.cont_upper[3:6]))
        rpycheck, rpy = self.rot_within_rpy_bounds(rot, rpy_bounds)
        if not all(rpycheck):
            rpy = self.rot_to_rpy(rot)
        return np.hstack((xyz, rpy))

    @staticmethod
    def rot_to_rpy(rot):
        rot = np.asarray(rot, dtype=float)
        rpy = np.zeros(3)
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            p = -np.arcsin(rot[2, 0])
            rpy[0] = np.arctan2((rot[2, 1] / np.cos(p)), (rot[2, 2] / np.cos(p)))
            rpy[1] = p
            rpy[2] = np.arctan2((rot[1, 0] / np.cos(p)), (rot[0, 0] / np.cos(p)))
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = np.arctan2(rot[0, 1], rot[0, 2])
                rpy[0] = r_offset
                rpy[1] = pi / 2
                rpy[2] = 0.0
            else:
                r_offset = np.arctan2(-rot[0, 1], -rot[0, 2])
                rpy[0] = r_offset
                rpy[1] = -pi / 2
                rpy[2] = 0.0
        return rpy

    @staticmethod
    def xyz_within_bounds(xyz, Bw):
        xyz = np.asarray(xyz, dtype=float)
        Bw = np.asarray(Bw, dtype=float)
        xyzcheck = []
        for i, x in enumerate(xyz):
            x_val = x.item() if hasattr(x, "item") else float(x)
            xyzcheck.append(((x_val + EPSILON) >= Bw[i, 0]) and ((x_val - EPSILON) <= Bw[i, 1]))
        return xyzcheck

    @staticmethod
    def rpy_within_bounds(rpy, Bw):
        rpy = np.asarray(rpy, dtype=float)
        Bw = np.asarray(Bw, dtype=float)
        rpy = wrap_to_interval(rpy, lower=Bw[:3, 0])
        rpycheck = [False] * 3
        for i in range(3):
            if Bw[i, 0] > Bw[i, 1] + EPSILON:
                rpycheck[i] = ((rpy[i] + EPSILON) >= Bw[i, 0]) or ((rpy[i] - EPSILON) <= Bw[i, 1])
            else:
                rpycheck[i] = ((rpy[i] + EPSILON) >= Bw[i, 0]) and ((rpy[i] - EPSILON) <= Bw[i, 1])
        return rpycheck

    @staticmethod
    def rot_within_rpy_bounds(rot, Bw):
        rot = np.asarray(rot, dtype=float)
        Bw = np.asarray(Bw, dtype=float)
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            psol = -np.arcsin(rot[2, 0])
            for p in [psol, (pi - psol)]:
                rpy = np.zeros(3)
                rpy[0] = np.arctan2((rot[2, 1] / np.cos(p)), (rot[2, 2] / np.cos(p)))
                rpy[1] = p
                rpy[2] = np.arctan2((rot[1, 0] / np.cos(p)), (rot[0, 0] / np.cos(p)))
                rpycheck = CanonicalTSRChart.rpy_within_bounds(rpy, Bw)
                if all(rpycheck):
                    return rpycheck, rpy
            return rpycheck, None
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = np.arctan2(rot[0, 1], rot[0, 2])
                rpy_list = [
                    np.array([Bw[2, 0] + r_offset, pi / 2, Bw[2, 0]]),
                    np.array([Bw[2, 1] + r_offset, pi / 2, Bw[2, 1]]),
                    np.array([Bw[0, 0], pi / 2, Bw[0, 0] - r_offset]),
                    np.array([Bw[0, 1], pi / 2, Bw[0, 1] - r_offset]),
                ]
                for rpy in rpy_list:
                    rpycheck = CanonicalTSRChart.rpy_within_bounds(rpy, Bw)
                    if rpycheck[1] is False:
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
            else:
                r_offset = np.arctan2(-rot[0, 1], -rot[0, 2])
                rpy_list = [
                    np.array([-Bw[2, 0] + r_offset, -pi / 2, Bw[2, 0]]),
                    np.array([-Bw[2, 1] + r_offset, -pi / 2, Bw[2, 1]]),
                    np.array([Bw[0, 0], -pi / 2, -Bw[0, 0] + r_offset]),
                    np.array([Bw[0, 1], -pi / 2, -Bw[0, 1] + r_offset]),
                ]
                for rpy in rpy_list:
                    rpycheck = CanonicalTSRChart.rpy_within_bounds(rpy, Bw)
                    if rpycheck[1] is False:
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
        return rpycheck, None
