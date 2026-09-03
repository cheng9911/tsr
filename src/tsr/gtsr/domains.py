# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Coordinate domains for the G-TSR geometry kernel.

A :class:`CoordinateDomain` is the single source of truth for the feasible
coordinate set ``Z`` of a region: an ordered product of per-axis
:class:`Interval` objects. The chart owns the *topology* (names, generators,
order, frame, periodicity metadata); the domain owns the *bounds*.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy import pi

from ..utils import wrap_to_interval


class Interval(ABC):
    """Base class for a single coordinate interval. All intervals are closed."""

    periodic: bool = False
    period = None

    @abstractmethod
    def contains(self, x):
        """Return True iff scalar ``x`` is in the interval."""

    @abstractmethod
    def map_from_unit(self, u):
        """Map ``u`` in ``[0, 1)`` to a coordinate value in the interval."""


class LinearInterval(Interval):
    """A closed real interval ``[lower, upper]`` for a non-periodic coordinate."""

    periodic = False
    period = None

    def __init__(self, lower, upper):
        lower = float(lower)
        upper = float(upper)
        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise ValueError(f"LinearInterval bounds must be finite, got ({lower}, {upper})")
        if lower > upper:
            raise ValueError(f"LinearInterval lower ({lower}) must be <= upper ({upper})")
        self.lower = lower
        self.upper = upper

    def contains(self, x):
        x = float(x)
        return self.lower <= x <= self.upper

    def map_from_unit(self, u):
        return self.lower + (self.upper - self.lower) * float(u)


class CircularInterval(Interval):
    """A circular interval of ``period`` (default 2π), closed.

    The raw ``(lower, upper)`` bounds may describe an inner interval, an outer
    interval (``lower > upper``) or a full circle. All are normalized once at
    construction into a continuous interval ``[lo_cont, hi_cont]``, matching the
    legacy ``TSR._Bw_cont`` semantics.
    """

    periodic = True

    def __init__(self, lower, upper, period=2 * pi):
        self.period = float(period)
        if not np.isfinite(self.period) or self.period <= 0:
            raise ValueError(f"CircularInterval period must be positive finite, got {self.period}")
        self.raw_lower = float(lower)
        self.raw_upper = float(upper)
        if not (np.isfinite(self.raw_lower) and np.isfinite(self.raw_upper)):
            raise ValueError(f"CircularInterval bounds must be finite, got ({lower}, {upper})")
        self.lo_cont = float(wrap_to_interval(np.array([self.raw_lower]))[0])
        interval = self.raw_upper - self.raw_lower
        if interval < 0:
            interval += self.period
        interval = min(interval, self.period)
        self.hi_cont = self.lo_cont + interval

    def contains(self, x):
        x = float(x)
        w = float(wrap_to_interval(np.array([x]), lower=np.array([self.lo_cont]))[0])
        return self.lo_cont <= w <= self.hi_cont

    def map_from_unit(self, u):
        z = self.lo_cont + (self.hi_cont - self.lo_cont) * float(u)
        return float(wrap_to_interval(np.array([z]))[0])


class CoordinateDomain:
    """An ordered product of per-axis :class:`Interval` objects."""

    def __init__(self, intervals):
        self.intervals = tuple(intervals)

    @property
    def dim(self):
        return len(self.intervals)

    def contains(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (self.dim,):
            raise ValueError(f"z must have shape ({self.dim},), got {z.shape}")
        return np.array([iv.contains(z[i]) for i, iv in enumerate(self.intervals)], dtype=bool)

    def map_from_unit_cube(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape != (self.dim,):
            raise ValueError(f"u must have shape ({self.dim},), got {u.shape}")
        return np.array([iv.map_from_unit(u[i]) for i, iv in enumerate(self.intervals)], dtype=float)

    def sample(self, rng):
        return self.map_from_unit_cube(rng.random(self.dim))

    @property
    def cont_lower(self):
        return np.array([iv.lo_cont if iv.periodic else iv.lower for iv in self.intervals])

    @property
    def cont_upper(self):
        return np.array([iv.hi_cont if iv.periodic else iv.upper for iv in self.intervals])

    @classmethod
    def from_bounds(cls, Bw, periodic, periods=None):
        Bw = np.asarray(Bw, dtype=float)
        if Bw.ndim != 2 or Bw.shape[1] != 2:
            raise ValueError(f"Bw must have shape (dim, 2), got {Bw.shape}")
        intervals = []
        for i in range(Bw.shape[0]):
            lo, hi = Bw[i, 0], Bw[i, 1]
            if periodic[i]:
                period = periods[i] if periods is not None else 2 * pi
                intervals.append(CircularInterval(lo, hi, period=period))
            else:
                intervals.append(LinearInterval(lo, hi))
        return cls(intervals)
