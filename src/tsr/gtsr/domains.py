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

from ._validation import finite_matrix, finite_vector


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
        lower = _finite_scalar(lower, "lower")
        upper = _finite_scalar(upper, "upper")
        if lower > upper:
            raise ValueError(f"LinearInterval lower ({lower}) must be <= upper ({upper})")
        self.lower = lower
        self.upper = upper

    def contains(self, x):
        x = _finite_scalar(x, "x")
        return self.lower <= x <= self.upper

    def map_from_unit(self, u):
        u = _unit_scalar(u)
        return self.lower + (self.upper - self.lower) * u


class CircularInterval(Interval):
    """A circular interval of ``period`` (default 2π), closed.

    The raw ``(lower, upper)`` bounds may describe an inner interval, an outer
    interval (``lower > upper``) or a full circle. All are normalized once at
    construction into a continuous interval ``[lo_cont, hi_cont]``, matching the
    legacy ``TSR._Bw_cont`` semantics.
    """

    periodic = True

    def __init__(self, lower, upper, period=2 * pi):
        self.period = _finite_scalar(period, "period")
        if self.period <= 0:
            raise ValueError(f"CircularInterval period must be positive finite, got {self.period}")
        self.raw_lower = _finite_scalar(lower, "lower")
        self.raw_upper = _finite_scalar(upper, "upper")
        raw_span = self.raw_upper - self.raw_lower
        span_tolerance = 8.0 * np.spacing(self.period)
        if raw_span < -self.period - span_tolerance or raw_span > self.period + span_tolerance:
            raise ValueError(
                f"CircularInterval raw span must be within one period [-{self.period}, {self.period}], got {raw_span}"
            )
        raw_span = float(np.clip(raw_span, -self.period, self.period))
        interval = raw_span + self.period if raw_span < 0 else raw_span
        self.lo_cont = self._wrap(self.raw_lower, lower=-self.period / 2.0)
        self.hi_cont = self.lo_cont + interval

    def contains(self, x):
        x = _finite_scalar(x, "x")
        w = self._wrap(x, lower=self.lo_cont)
        return self.lo_cont <= w <= self.hi_cont

    def map_from_unit(self, u):
        u = _unit_scalar(u)
        z = self.lo_cont + (self.hi_cont - self.lo_cont) * u
        return self._wrap(z, lower=-self.period / 2.0)

    def _wrap(self, value, *, lower):
        return float((value - lower) % self.period + lower)


class CoordinateDomain:
    """An ordered product of per-axis :class:`Interval` objects."""

    def __init__(self, intervals):
        if intervals is None:
            raise TypeError("intervals must not be None")
        try:
            self.intervals = tuple(intervals)
        except TypeError as exc:
            raise TypeError("intervals must be an iterable of Interval objects") from exc
        if not all(isinstance(interval, Interval) for interval in self.intervals):
            raise TypeError("intervals must contain only Interval objects")

    @property
    def dim(self):
        return len(self.intervals)

    def contains(self, z):
        z = finite_vector(z, self.dim, "z")
        return np.array([iv.contains(z[i]) for i, iv in enumerate(self.intervals)], dtype=bool)

    def map_from_unit_cube(self, u):
        u = finite_vector(u, self.dim, "u")
        if np.any((u < 0.0) | (u >= 1.0)):
            raise ValueError("u values must lie in [0, 1)")
        return np.array([iv.map_from_unit(u[i]) for i, iv in enumerate(self.intervals)], dtype=float)

    def sample(self, rng):
        if rng is None or not callable(getattr(rng, "random", None)):
            raise TypeError("rng must provide a callable random method")
        return self.map_from_unit_cube(rng.random(self.dim))

    @property
    def cont_lower(self):
        return np.array([iv.lo_cont if iv.periodic else iv.lower for iv in self.intervals])

    @property
    def cont_upper(self):
        return np.array([iv.hi_cont if iv.periodic else iv.upper for iv in self.intervals])

    @classmethod
    def from_bounds(cls, Bw, periodic, periods=None):
        try:
            periodic = tuple(periodic)
        except TypeError as exc:
            raise TypeError("periodic must be an iterable of booleans") from exc
        if not all(isinstance(value, (bool, np.bool_)) for value in periodic):
            raise TypeError("periodic must contain only booleans")
        Bw = finite_matrix(Bw, (len(periodic), 2), "Bw")
        if periods is None:
            periods = tuple(2 * pi if is_periodic else None for is_periodic in periodic)
        else:
            try:
                periods = tuple(periods)
            except TypeError as exc:
                raise TypeError("periods must be an iterable") from exc
            if len(periods) != len(periodic):
                raise ValueError(f"periods must have length {len(periodic)}, got {len(periods)}")
        intervals = []
        for i in range(Bw.shape[0]):
            lo, hi = Bw[i, 0], Bw[i, 1]
            if periodic[i]:
                intervals.append(CircularInterval(lo, hi, period=periods[i]))
            else:
                if periods[i] is not None:
                    raise ValueError(f"periods[{i}] must be None for a non-periodic coordinate")
                intervals.append(LinearInterval(lo, hi))
        return cls(intervals)


def _finite_scalar(value, name):
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _unit_scalar(value):
    scalar = _finite_scalar(value, "u")
    if not 0.0 <= scalar < 1.0:
        raise ValueError("u must lie in [0, 1)")
    return scalar
