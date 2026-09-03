# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest
from numpy import pi

from tsr.gtsr.domains import CircularInterval, CoordinateDomain, LinearInterval


# --- D-01 ---------------------------------------------------------------
class TestLinearInterval:
    def test_interior_endpoint_outside(self):
        iv = LinearInterval(-0.1, 0.1)
        assert iv.contains(-0.1) is True
        assert iv.contains(0.0) is True
        assert iv.contains(0.1) is True
        assert iv.contains(0.1000001) is False
        assert iv.contains(-0.1000001) is False

    def test_zero_width(self):
        iv = LinearInterval(1.0, 1.0)
        assert iv.contains(1.0) is True
        assert iv.contains(1.0 + 1e-12) is False

    def test_rejects_inverted(self):
        with pytest.raises(ValueError):
            LinearInterval(1.0, 0.0)

    def test_rejects_nonfinite(self):
        with pytest.raises(ValueError):
            LinearInterval(np.nan, 0.0)
        with pytest.raises(ValueError):
            LinearInterval(0.0, np.inf)


# --- D-02 ---------------------------------------------------------------
def test_circular_interval_inner():
    iv = CircularInterval(-pi / 4, pi / 4)
    assert iv.contains(0.0) is True
    assert iv.contains(pi / 4) is True
    assert iv.contains(-pi / 4) is True
    assert iv.contains(pi / 2) is False
    # ±2kπ equivalence class
    assert iv.contains(pi / 4 + 2 * pi) is True
    assert iv.contains(-pi / 4 - 2 * pi) is True


# --- D-03 ---------------------------------------------------------------
def test_circular_interval_outer():
    iv = CircularInterval(3 * pi / 4, -3 * pi / 4)
    assert iv.contains(pi) is True
    assert iv.contains(-pi) is True
    assert iv.contains(0.0) is False
    assert iv.contains(pi / 2) is False


# --- D-04 ---------------------------------------------------------------
def test_circular_interval_full():
    iv = CircularInterval(-pi, pi)
    for x in [0.0, pi / 2, pi, -pi, 3 * pi / 2, 100.0, -100.0]:
        assert iv.contains(x) is True


@pytest.mark.parametrize(
    ("period", "lower", "upper", "inside", "outside"),
    [
        (360.0, 300.0, 60.0, [350.0, -10.0, 30.0, 710.0], [180.0, -180.0]),
        (1.0, 0.4, -0.4, [0.45, -0.45, 1.45], [0.0, 1.0]),
    ],
)
def test_circular_interval_custom_period(period, lower, upper, inside, outside):
    iv = CircularInterval(lower, upper, period=period)
    for value in inside:
        assert iv.contains(value) is True
    for value in outside:
        assert iv.contains(value) is False
    mapped = [iv.map_from_unit(u) for u in [0.0, 0.25, 0.75]]
    assert all(iv.contains(value) for value in mapped)


@pytest.mark.parametrize(
    ("lower", "upper", "period"),
    [(0.0, 361.0, 360.0), (0.0, -361.0, 360.0), (0.0, 1.01, 1.0), (0.0, -1.01, 1.0)],
)
def test_circular_interval_rejects_more_than_one_raw_period(lower, upper, period):
    with pytest.raises(ValueError, match="raw span must be within one period"):
        CircularInterval(lower, upper, period=period)


# --- D-05 ---------------------------------------------------------------
def test_domain_dimension_error():
    dom = CoordinateDomain([LinearInterval(0, 1), LinearInterval(0, 1)])
    with pytest.raises(ValueError):
        dom.contains(np.array([0.5]))
    with pytest.raises(ValueError):
        dom.map_from_unit_cube(np.array([0.5]))


def test_domain_rejects_nonfinite_and_out_of_unit_inputs():
    dom = CoordinateDomain([LinearInterval(0, 1)])
    with pytest.raises(ValueError, match="finite"):
        dom.contains([np.nan])
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        dom.map_from_unit_cube([1.1])
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        dom.map_from_unit_cube([1.0])


def test_from_bounds_validates_metadata_lengths_and_types():
    with pytest.raises(ValueError, match="Bw must have shape"):
        CoordinateDomain.from_bounds(np.zeros((2, 2)), (False, False, False))
    with pytest.raises(ValueError, match="periods must have length"):
        CoordinateDomain.from_bounds(np.zeros((2, 2)), (False, True), (None,))
    with pytest.raises(ValueError, match="must be None"):
        CoordinateDomain.from_bounds(np.zeros((1, 2)), (False,), (2 * pi,))
    with pytest.raises(TypeError, match="booleans"):
        CoordinateDomain.from_bounds(np.zeros((2, 2)), (False, 1))
    with pytest.raises(ValueError, match="finite"):
        CoordinateDomain.from_bounds([[0.0, np.inf]], (False,))


# --- D-06 ---------------------------------------------------------------
def test_linear_interval_aperiodic():
    # A screw coordinate q is linear: q and q+2π are DISTINCT points.
    iv = LinearInterval(0.0, 1.0)
    assert iv.contains(0.5) is True
    assert iv.contains(0.5 + 2 * pi) is False


# --- D-07 ---------------------------------------------------------------
def test_sampling_reproducible():
    dom = CoordinateDomain(
        [
            LinearInterval(-0.1, 0.1),
            LinearInterval(-0.1, 0.1),
            LinearInterval(-0.1, 0.1),
            CircularInterval(-pi, pi),
            CircularInterval(-pi / 2, pi / 2),
            CircularInterval(3 * pi / 4, -3 * pi / 4),
        ]
    )
    s1 = dom.sample(np.random.default_rng(42))
    s2 = dom.sample(np.random.default_rng(42))
    np.testing.assert_array_equal(s1, s2)


def test_from_bounds_and_contains():
    Bw = np.array(
        [
            [-0.1, 0.1],
            [-0.1, 0.1],
            [0.0, 0.02],
            [-pi / 4, pi / 4],
            [-pi / 4, pi / 4],
            [3 * pi / 4, -3 * pi / 4],
        ]
    )
    periodic = (False, False, False, True, True, True)
    dom = CoordinateDomain.from_bounds(Bw, periodic)
    assert dom.dim == 6
    assert all(dom.contains(np.array([0.0, 0.0, 0.01, 0.0, 0.0, pi])))
    assert not dom.contains(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))[5]
