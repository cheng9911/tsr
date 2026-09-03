# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest
from numpy import pi

from tests.gtsr.helpers import (
    assert_transforms_close,
    make_pair,
    random_bounds,
    random_se3,
)
from tsr.gtsr.region import GTSR
from tsr.tsr import TSR
from tsr.utils import EPSILON, wrap_to_interval


# --- C-01 ---------------------------------------------------------------
def test_C01_same_coordinate_same_pose(rng):
    for iteration in range(50):
        T0_w, Tw_e, Bw = random_se3(rng), random_se3(rng), random_bounds(rng)
        old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        z = new.domain.sample(rng)
        assert_transforms_close(
            old.to_transform(z),
            new.to_transform(z),
            context=f"iteration={iteration}, z={z!r}, Bw={Bw!r}",
        )


# --- C-02 ---------------------------------------------------------------
def test_C02_contains_boundary_corpus(rng):
    T0_w, Tw_e = random_se3(rng), random_se3(rng)
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
    old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
    # 直接由局部坐标构造世界变换，绕过 to_transform 的边界校验（contains 的职责正是判断越界）
    cases = [
        ([0.0, 0.0, 0.01, 0.0, 0.0, pi], True, "outer yaw interior"),
        ([0.1, 0.1, 0.02, pi / 4, pi / 4, -3 * pi / 4], True, "endpoints"),
        ([0.1 + 0.5 * EPSILON, 0.0, 0.01, 0.0, 0.0, pi], True, "translation tolerance inside"),
        ([0.1 + 1.5 * EPSILON, 0.0, 0.01, 0.0, 0.0, pi], False, "translation tolerance outside"),
        ([0.0, 0.0, 0.01, pi / 4 + 0.5 * EPSILON, 0.0, pi], True, "wrapped rotation tolerance inside"),
        ([0.0, 0.0, 0.01, pi / 4 + 1.5 * EPSILON, 0.0, pi], False, "wrapped rotation tolerance outside"),
        ([0.0, 0.0, 0.01, 0.0, 0.0, -3 * pi / 4 + 0.5 * EPSILON], True, "outer tolerance inside"),
        ([0.0, 0.0, 0.01, 0.0, 0.0, -3 * pi / 4 + 1.5 * EPSILON], False, "outer tolerance outside"),
        ([0.0, 0.0, 0.01, 0.0, 0.0, 0.0], False, "outer yaw excluded arc"),
        ([5.0, 0.0, 0.0, 0.0, 0.0, pi], False, "far outside"),
    ]
    for z, expected, label in cases:
        T = new.T0_w @ new.chart.forward(np.array(z)) @ new.Tw_e
        assert old.contains(T) == expected, f"legacy oracle disagrees for {label}: z={z}"
        assert new.contains(T) == expected, f"GTSR disagrees for {label}: z={z}"

    raw_outer = np.array([[3 * pi / 4, -3 * pi / 4], [-pi, pi], [-pi / 4, pi / 4]])
    for yaw in [
        3 * pi / 4 - 0.5 * EPSILON,
        3 * pi / 4 - 1.5 * EPSILON,
        -3 * pi / 4 + 0.5 * EPSILON,
        -3 * pi / 4 + 1.5 * EPSILON,
    ]:
        rpy = np.array([yaw, 0.0, 0.0])
        legacy_result = TSR.rpy_within_bounds(rpy, raw_outer)
        assert bool(legacy_result[0]) is True
        assert new.chart.rpy_within_bounds(rpy, raw_outer) == legacy_result


# --- C-03 ---------------------------------------------------------------
def test_C03_coordinates_forward_roundtrip(rng):
    for iteration in range(50):
        T0_w, Tw_e, Bw = random_se3(rng), random_se3(rng), random_bounds(rng)
        old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        z = new.domain.sample(rng)
        T = new.to_transform(z)
        T_local = np.linalg.inv(new.T0_w) @ T @ np.linalg.inv(new.Tw_e)
        recovered = new.chart.coordinates(T_local, new.domain)
        assert all(new.domain.contains(recovered)), (
            f"coordinates escaped domain: iteration={iteration}, z={z!r}, Bw={Bw!r}, recovered={recovered!r}"
        )
        assert_transforms_close(
            T_local,
            new.chart.forward(recovered),
            context=f"iteration={iteration}, z={z!r}, Bw={Bw!r}, recovered={recovered!r}",
        )


# --- C-04 ---------------------------------------------------------------
@pytest.mark.parametrize("rotation_weight", [0.25, 1.0, 3.0])
def test_C04_legacy_distance_guaranteed_outside(rng, rotation_weight):
    for iteration in range(50):
        T0_w, Tw_e, Bw = random_se3(rng), random_se3(rng), random_bounds(rng)
        old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        z = new.domain.sample(rng)
        T_in = new.to_transform(z)
        d_old, _ = old.distance(T_in, rotation_weight=rotation_weight)
        d_new, _ = new.distance(T_in, rotation_weight=rotation_weight)
        assert d_old == 0.0 and d_new == 0.0

        # Force an outside query in local translation, independent of the frames and rotation bounds.
        z_out = z.copy()
        z_out[0] = new.domain.cont_upper[0] + 2.0 * EPSILON
        T_out = new.T0_w @ new.chart.forward(z_out) @ new.Tw_e
        assert not old.contains(T_out) and not new.contains(T_out)
        d_old, bw_old = old.distance(T_out, rotation_weight=rotation_weight)
        d_new, bw_new = new.distance(T_out, rotation_weight=rotation_weight)
        assert d_new == pytest.approx(d_old, abs=1e-12), (
            f"iteration={iteration}, rotation_weight={rotation_weight}, z={z!r}, Bw={Bw!r}"
        )
        assert_transforms_close(
            old.to_transform(bw_old),
            new.to_transform(bw_new),
            context=(
                f"outside seed fixture iteration={iteration}, rotation_weight={rotation_weight}, "
                f"z={z!r}, Bw={Bw!r}, bw_old={bw_old!r}, bw_new={bw_new!r}"
            ),
        )


def test_C04_inside_bwopt_reconstructs_query():
    Bw = np.array(
        [
            [-0.2, 0.2],
            [-0.2, 0.2],
            [-0.2, 0.2],
            [3 * pi / 4, -3 * pi / 4],
            [-pi, pi],
            [-0.4, 0.4],
        ]
    )
    old, new = make_pair(Bw=Bw)
    z = np.array([0.1, -0.1, 0.05, pi, 0.3, 0.2])
    T_in = new.to_transform(z)
    for region in [old, new]:
        distance, bwopt = region.distance(T_in, rotation_weight=2.5)
        assert distance == 0.0
        assert_transforms_close(T_in, region.to_transform(bwopt))


def test_C04_distance_outer_full_and_near_singular():
    Bw = np.array(
        [
            [-0.1, 0.1],
            [-0.1, 0.1],
            [-0.1, 0.1],
            [3 * pi / 4, -3 * pi / 4],
            [-pi, pi],
            [-0.4, 0.4],
        ]
    )
    old, new = make_pair(Bw=Bw)
    for pitch in [pi / 2 - 1e-6, pi / 2, -pi / 2 + 1e-6, -pi / 2]:
        z_out = np.array([0.1 + 2 * EPSILON, 0.0, 0.0, pi, pitch, 0.2])
        T_out = new.chart.forward(z_out)
        assert not old.contains(T_out) and not new.contains(T_out)
        for rotation_weight in [0.2, 2.5]:
            d_old, bw_old = old.distance(T_out, rotation_weight=rotation_weight)
            d_new, bw_new = new.distance(T_out, rotation_weight=rotation_weight)
            assert d_new == pytest.approx(d_old, abs=1e-12)
            assert_transforms_close(old.to_transform(bw_old), new.to_transform(bw_new))


# --- C-05 ---------------------------------------------------------------
def test_C05_sampling_mapping_compat(rng):
    T0_w, Tw_e = random_se3(rng), random_se3(rng)
    Bw = np.array(
        [
            [-0.2, 0.2],
            [-0.3, 0.3],
            [-0.4, 0.4],
            [3 * pi / 4, -3 * pi / 4],
            [-pi, pi],
            [-0.5, 0.5],
        ]
    )
    old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
    u = rng.random(6)
    new_z = new.domain.map_from_unit_cube(u)
    old_z = old._Bw_cont[:, 0] + (old._Bw_cont[:, 1] - old._Bw_cont[:, 0]) * u
    old_z[3:6] = wrap_to_interval(old_z[3:6])
    np.testing.assert_allclose(new_z, old_z, atol=1e-12)

    # Partial legacy sampling consumes one scalar draw per NaN, in coordinate order.
    z0 = new.domain.map_from_unit_cube(np.full(6, 0.25))
    partial = np.full(6, np.nan)
    partial[0] = z0[0]
    partial[4] = z0[4]
    expected_stream = RecordingStream(1234)
    expected = partial.copy()
    for i in np.flatnonzero(np.isnan(expected)):
        expected[i] = new.domain.intervals[i].map_from_unit(expected_stream.random())
    expected[3:6] = wrap_to_interval(expected[3:6])
    expected_next = expected_stream.random()

    actual_stream = RecordingStream(1234)
    actual = new.sample_xyzrpy(partial, rng=actual_stream)
    np.testing.assert_array_equal(actual, expected)
    assert actual_stream.scalar_draws == 4
    assert actual_stream.random() == expected_next


def test_C05_shared_stream_matches_legacy(monkeypatch):
    Bw = np.array(
        [
            [-0.2, 0.2],
            [-0.3, 0.3],
            [-0.4, 0.4],
            [3 * pi / 4, -3 * pi / 4],
            [-pi, pi],
            [-0.5, 0.5],
        ]
    )
    old, new = make_pair(Bw=Bw)
    partial = np.array([np.nan, 0.0, np.nan, pi, np.nan, 0.25])
    old_stream = RecordingStream(77)
    new_stream = RecordingStream(77)
    monkeypatch.setattr(np.random, "random_sample", old_stream.random)
    old_sample = old.sample_xyzrpy(partial)
    new_sample = new.sample_xyzrpy(partial, rng=new_stream)
    np.testing.assert_array_equal(new_sample, old_sample)
    assert old_stream.scalar_draws == new_stream.scalar_draws == 3
    assert old_stream.random() == new_stream.random()


# --- 域连续边界与旧 _Bw_cont 逐位一致（强不变量） --------------------------
def test_domain_cont_bounds_match_oracle(rng):
    for _ in range(20):
        Bw = random_bounds(rng)
        old = TSR(Bw=Bw)
        new = GTSR.canonical(Bw=Bw)
        np.testing.assert_allclose(new.domain.cont_lower, old._Bw_cont[:, 0], atol=1e-15)
        np.testing.assert_allclose(new.domain.cont_upper, old._Bw_cont[:, 1], atol=1e-15)


class RecordingStream:
    """Small controllable stream shared by legacy and injectable RNG APIs."""

    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)
        self.scalar_draws = 0

    def random(self, size=None):
        if size is None:
            self.scalar_draws += 1
        return self.generator.random(size)
