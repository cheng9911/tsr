# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Region-level composition for the G-TSR geometry kernel."""

import numpy as np
from numpy import pi

from ..utils import wrap_to_interval
from ._validation import finite_vector, se3_transform
from .charts import CanonicalTSRChart, GeneratorChart
from .domains import CoordinateDomain

NANBW = np.ones(6) * float("nan")


class GTSR:
    """A bounded chart region ``{T0_w @ chart.forward(z) @ Tw_e}``.

    Generic charts support composition, projection, metric membership, and
    coordinate-domain sampling. The legacy ``xyzrpy`` membership, distance,
    and coordinate sampling methods remain restricted to
    :class:`CanonicalTSRChart`.
    """

    def __init__(self, T0_w=None, Tw_e=None, chart=None, domain=None):
        if T0_w is None:
            T0_w = np.eye(4)
        if Tw_e is None:
            Tw_e = np.eye(4)
        if not isinstance(chart, GeneratorChart):
            raise TypeError("chart must be a non-null GeneratorChart")
        if not isinstance(domain, CoordinateDomain):
            raise TypeError("domain must be a non-null CoordinateDomain")
        if chart.dim != domain.dim:
            raise ValueError(f"chart/domain dimension mismatch: {chart.dim} != {domain.dim}")
        try:
            periodic = tuple(chart.periodic)
            periods = tuple(chart.periods)
        except TypeError as exc:
            raise TypeError("chart.periodic and chart.periods must be iterable") from exc
        if len(periodic) != chart.dim:
            raise ValueError(f"chart.periodic must have length {chart.dim}, got {len(periodic)}")
        if len(periods) != chart.dim:
            raise ValueError(f"chart.periods must have length {chart.dim}, got {len(periods)}")
        if not all(isinstance(value, (bool, np.bool_)) for value in periodic):
            raise TypeError("chart.periodic must contain only booleans")
        for i, interval in enumerate(domain.intervals):
            if periodic[i] != interval.periodic:
                raise ValueError(f"chart/domain periodicity mismatch at dimension {i}")
            if interval.periodic:
                try:
                    period = float(periods[i])
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"chart.periods[{i}] must be a real scalar") from exc
                if not np.isfinite(period) or period <= 0.0:
                    raise ValueError(f"chart.periods[{i}] must be positive and finite")
                if not np.isclose(period, interval.period, rtol=0.0, atol=0.0):
                    raise ValueError(f"chart/domain period mismatch at dimension {i}")
            elif periods[i] is not None:
                raise ValueError(f"chart.periods[{i}] must be None for a non-periodic coordinate")
        self.T0_w = se3_transform(T0_w, "T0_w")
        self.Tw_e = se3_transform(Tw_e, "Tw_e")
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
        self._require_canonical("is_valid")
        z = finite_vector(z, 6, "z", allow_nan=ignoreNAN)
        z_for_check = np.nan_to_num(z, nan=0.0) if ignoreNAN else z
        xyz, rpy = z_for_check[0:3], z_for_check[3:6]
        xyz_bounds = np.column_stack((self.domain.cont_lower[0:3], self.domain.cont_upper[0:3]))
        rpy_bounds = np.column_stack((self.domain.cont_lower[3:6], self.domain.cont_upper[3:6]))
        xyzcheck = self.chart.xyz_within_bounds(xyz, xyz_bounds)
        rpycheck = self.chart.rpy_within_bounds(rpy, rpy_bounds)
        check = np.hstack((xyzcheck, rpycheck))
        if ignoreNAN:
            check = check | np.isnan(z)
        return check

    def to_transform(self, z):
        z = finite_vector(z, self.chart.dim, "z")
        validity = self.is_valid(z) if isinstance(self.chart, CanonicalTSRChart) else self.domain.contains(z)
        if not all(validity):
            violated = [i for i, v in enumerate(validity) if not v]
            raise ValueError(f"z violates bounds at dimensions {violated}: z={z}")
        T_local = se3_transform(self.chart.forward(z), "chart.forward(z)")
        return self.T0_w @ T_local @ self.Tw_e

    def to_xyzrpy(self, T):
        self._require_canonical("to_xyzrpy")
        T = se3_transform(T, "T")
        T_local = np.linalg.inv(self.T0_w) @ T @ np.linalg.inv(self.Tw_e)
        return self.chart.coordinates(T_local, self.domain)

    def project(self, T, metric):
        from .projection import project as _project

        return _project(self, T, metric)

    def contains(self, T, metric=None, tol=1e-8):
        if metric is None:
            self._require_canonical("contains")
            T = se3_transform(T, "T")
            T0_s_prime = np.dot(T, np.linalg.inv(self.Tw_e))
            Tw_s_prime = np.dot(np.linalg.inv(self.T0_w), T0_s_prime)
            xyz, rot = Tw_s_prime[0:3, 3], Tw_s_prime[0:3, 0:3]
            xyz_bounds = np.column_stack((self.domain.cont_lower[0:3], self.domain.cont_upper[0:3]))
            rpy_bounds = np.column_stack((self.domain.cont_lower[3:6], self.domain.cont_upper[3:6]))
            xyzcheck = self.chart.xyz_within_bounds(xyz, xyz_bounds)
            rotcheck, _ = self.chart.rot_within_rpy_bounds(rot, rpy_bounds)
            return all(np.hstack((xyzcheck, rotcheck)))
        from .projection import project as _project

        return _project(self, T, metric).distance <= tol

    def _displacement_to_tsr(self, T):
        self._require_canonical("distance")
        T = se3_transform(T, "T")
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
        self._require_canonical("distance")
        T = se3_transform(T, "T")
        try:
            rotation_weight = float(rotation_weight)
        except (TypeError, ValueError) as exc:
            raise TypeError("rotation_weight must be a real scalar") from exc
        if not np.isfinite(rotation_weight) or rotation_weight < 0.0:
            raise ValueError("rotation_weight must be finite and non-negative")
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
        self._require_canonical("sample_xyzrpy")
        if rng is None:
            rng = np.random.default_rng()
        if not callable(getattr(rng, "random", None)):
            raise TypeError("rng must provide a callable random method")
        xyzrpy = finite_vector(xyzrpy, 6, "xyzrpy", allow_nan=True)
        check = self.is_valid(xyzrpy, ignoreNAN=True)
        if not all(check):
            raise ValueError("xyzrpy must be within bounds", check)
        Bw_sample = xyzrpy.copy()
        for i, x in enumerate(xyzrpy):
            if np.isnan(x):
                Bw_sample[i] = self.domain.intervals[i].map_from_unit(rng.random())
        Bw_sample[3:6] = wrap_to_interval(Bw_sample[3:6])
        return Bw_sample

    def sample(self, partial=None, rng=None):
        if isinstance(self.chart, CanonicalTSRChart):
            xyzrpy = NANBW if partial is None else partial
            return self.to_transform(self.sample_xyzrpy(xyzrpy, rng=rng))

        if rng is None:
            rng = np.random.default_rng()
        if not callable(getattr(rng, "random", None)):
            raise TypeError("rng must provide a callable random method")
        if partial is None:
            partial = np.full(self.chart.dim, np.nan)
        z = finite_vector(partial, self.chart.dim, "partial", allow_nan=True)
        valid = np.array([np.isnan(value) or self.domain.intervals[i].contains(value) for i, value in enumerate(z)])
        if not all(valid):
            violated = np.flatnonzero(~valid).tolist()
            raise ValueError(f"partial coordinates violate bounds at dimensions {violated}")
        sampled = z.copy()
        for i in np.flatnonzero(np.isnan(sampled)):
            sampled[i] = self.domain.intervals[i].map_from_unit(rng.random())
        return self.to_transform(sampled)

    def _require_canonical(self, method):
        if not isinstance(self.chart, CanonicalTSRChart):
            raise TypeError(f"{method} is only supported for CanonicalTSRChart in Sprint A")
