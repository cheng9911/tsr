# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Region-level composition for the G-TSR geometry kernel."""

from functools import reduce

import numpy as np

from .charts import CanonicalTSRChart
from .domains import CoordinateDomain


class GTSR:
    """A generalized TSR: ``{T0_w @ chart.forward(z) @ Tw_e | z in domain}``."""

    def __init__(self, T0_w=None, Tw_e=None, chart=None, domain=None):
        if T0_w is None:
            T0_w = np.eye(4)
        if Tw_e is None:
            Tw_e = np.eye(4)
        self.T0_w = np.asarray(T0_w, dtype=float)
        self.Tw_e = np.asarray(Tw_e, dtype=float)
        self.chart = chart
        self.domain = domain

    @classmethod
    def canonical(cls, T0_w=None, Tw_e=None, Bw=None):
        """The §6.2 construction map E(T0_w,Tw_e,Bw) = (T0_w,Tw_e,C_TSR,Z(Bw))."""
        if Bw is None:
            Bw = np.zeros((6, 2))
        chart = CanonicalTSRChart()
        domain = CoordinateDomain.from_bounds(Bw, chart.periodic, chart.periods)
        return cls(T0_w, Tw_e, chart, domain)

    def is_valid(self, z, ignoreNAN=False):
        z = np.asarray(z, dtype=float)
        xyz, rpy = z[0:3], z[3:6]
        xyz_bounds = np.column_stack((self.domain.cont_lower[0:3], self.domain.cont_upper[0:3]))
        rpy_bounds = np.column_stack((self.domain.cont_lower[3:6], self.domain.cont_upper[3:6]))
        xyzcheck = self.chart.xyz_within_bounds(xyz, xyz_bounds)
        rpycheck = self.chart.rpy_within_bounds(rpy, rpy_bounds)
        check = np.hstack((xyzcheck, rpycheck))
        if ignoreNAN:
            check = check | np.isnan(z)
        return check

    def to_transform(self, z):
        z = np.asarray(z, dtype=float)
        if len(z) != 6:
            raise ValueError("z must be of length 6")
        validity = self.is_valid(z)
        if not all(validity):
            violated = [i for i, v in enumerate(validity) if not v]
            raise ValueError(f"z violates bounds at dimensions {violated}: z={z}")
        T_local = self.chart.forward(z)
        return reduce(np.dot, [self.T0_w, T_local, self.Tw_e])
