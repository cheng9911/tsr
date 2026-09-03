# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generator charts for the G-TSR geometry kernel."""

from abc import ABC, abstractmethod

import numpy as np
from numpy import pi

from ..utils import wrap_to_interval


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
