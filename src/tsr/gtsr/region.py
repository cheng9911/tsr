# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Region-level composition for the G-TSR geometry kernel."""

from functools import reduce

import numpy as np
from numpy import pi

from ..utils import wrap_to_interval
from .charts import CanonicalTSRChart
from .domains import CoordinateDomain

NANBW = np.ones(6) * float("nan")


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

    def to_xyzrpy(self, T):
        T_local = reduce(np.dot, [np.linalg.inv(self.T0_w), T, np.linalg.inv(self.Tw_e)])
        return self.chart.coordinates(T_local, self.domain)

    def contains(self, T):
        T0_s_prime = np.dot(T, np.linalg.inv(self.Tw_e))
        Tw_s_prime = np.dot(np.linalg.inv(self.T0_w), T0_s_prime)
        xyz, rot = Tw_s_prime[0:3, 3], Tw_s_prime[0:3, 0:3]
        xyz_bounds = np.column_stack((self.domain.cont_lower[0:3], self.domain.cont_upper[0:3]))
        rpy_bounds = np.column_stack((self.domain.cont_lower[3:6], self.domain.cont_upper[3:6]))
        xyzcheck = self.chart.xyz_within_bounds(xyz, xyz_bounds)
        rotcheck, _ = self.chart.rot_within_rpy_bounds(rot, rpy_bounds)
        return all(np.hstack((xyzcheck, rotcheck)))

    def _displacement_to_tsr(self, T):
        T0_s_prime = np.dot(T, np.linalg.inv(self.Tw_e))
        Tw_s_prime = np.dot(np.linalg.inv(self.T0_w), T0_s_prime)
        xyz = Tw_s_prime[0:3, 3]
        rot = Tw_s_prime[0:3, 0:3]
        rpy = self.chart.rot_to_rpy(rot)

        best_dx = None
        best_dw = None
        best_norm = float("inf")

        rpy_candidates = [rpy]
        r, p, y = rpy
        rpy_candidates.append(np.array([r + pi, pi - p, y + pi]))
        rpy_candidates.append(np.array([r + pi, pi - p, y - pi]))
        rpy_candidates.append(np.array([r - pi, pi - p, y + pi]))
        rpy_candidates.append(np.array([r - pi, pi - p, y - pi]))
        rpy_candidates.append(np.array([r + pi, -pi - p, y + pi]))
        rpy_candidates.append(np.array([r + pi, -pi - p, y - pi]))
        rpy_candidates.append(np.array([r - pi, -pi - p, y + pi]))
        rpy_candidates.append(np.array([r - pi, -pi - p, y - pi]))

        for rpy_cand in rpy_candidates:
            rpy_wrapped = wrap_to_interval(rpy_cand, lower=self.domain.cont_lower[3:6])
            dw_cand = np.hstack((xyz, rpy_wrapped))
            dx = np.zeros(6)
            for i in range(6):
                if dw_cand[i] < self.domain.cont_lower[i]:
                    dx[i] = dw_cand[i] - self.domain.cont_lower[i]
                elif dw_cand[i] > self.domain.cont_upper[i]:
                    dx[i] = dw_cand[i] - self.domain.cont_upper[i]
            norm = np.linalg.norm(dx)
            if norm < best_norm:
                best_norm = norm
                best_dx = dx
                best_dw = dw_cand
        return best_dx, best_dw

    def distance(self, T, rotation_weight=1.0):
        if self.contains(T):
            return 0.0, self.to_xyzrpy(T)
        dx, dw = self._displacement_to_tsr(T)
        dx_weighted = dx.copy()
        dx_weighted[3:6] *= rotation_weight
        dist = np.linalg.norm(dx_weighted)
        bwopt = np.clip(dw, self.domain.cont_lower, self.domain.cont_upper)
        bwopt[3:6] = wrap_to_interval(bwopt[3:6])
        return dist, bwopt

    def sample_xyzrpy(self, xyzrpy=NANBW, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        check = self.is_valid(xyzrpy, ignoreNAN=True)
        if not all(check):
            raise ValueError("xyzrpy must be within bounds", check)
        u = rng.random(6)
        Bw_sample = np.array([
            self.domain.cont_lower[i] + (self.domain.cont_upper[i] - self.domain.cont_lower[i]) * u[i]
            if np.isnan(x)
            else x
            for i, x in enumerate(xyzrpy)
        ])
        Bw_sample[3:6] = wrap_to_interval(Bw_sample[3:6])
        return Bw_sample

    def sample(self, xyzrpy=NANBW, rng=None):
        return self.to_transform(self.sample_xyzrpy(xyzrpy, rng=rng))
