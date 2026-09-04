# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest
from numpy import pi

from tests.gtsr.helpers import (
    DIFF_SAMPLES_PER_SEED,
    SEEDS,
    assert_transforms_close,
    make_pair,
    random_bounds,
    random_se3,
)
from tsr.gtsr.charts import CanonicalTSRChart
from tsr.gtsr.region import GTSR
from tsr.tsr import TSR


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_forward_matches_oracle():
    chart = CanonicalTSRChart()
    for z in [
        [0, 0, 0, 0, 0, 0],
        [0.1, -0.2, 0.3, 0.0, 0.0, 0.0],
        [0, 0, 0, 0.3, -0.4, 0.5],
        [0.1, 0.2, 0.3, 0.4, -0.5, 0.6],
    ]:
        np.testing.assert_allclose(chart.forward(z), TSR.xyzrpy_to_trans(np.array(z)), atol=1e-15)


def test_forward_rotation_order_is_rz_ry_rx():
    # 三个角均非零：直接与 Rz(y) @ Ry(p) @ Rx(r) 比较，区分错误次序（E-04）
    r, p, y = 0.3, -0.4, 0.5
    chart = CanonicalTSRChart()
    T = chart.forward([0, 0, 0, r, p, y])
    expected_R = _rotz(y) @ _roty(p) @ _rotx(r)
    np.testing.assert_allclose(T[0:3, 0:3], expected_R, atol=1e-15)
    np.testing.assert_allclose(T[0:3, 3], [0, 0, 0], atol=1e-15)


def test_rpy_to_rot_matches_oracle():
    chart = CanonicalTSRChart()
    for rpy in [[0.3, -0.4, 0.5], [pi, 0, pi], [0.1, -0.2, 0.3]]:
        np.testing.assert_allclose(chart.rpy_to_rot(np.array(rpy)), TSR.rpy_to_rot(np.array(rpy)), atol=1e-15)


def test_legacy_rot_to_rpy_matches_oracle():
    # Legacy helper stays byte-for-byte compatible for contains/distance.
    chart = CanonicalTSRChart()
    for rpy in [[0.3, 0.4, 0.5], [0.3, pi / 2, 0.5], [0.3, pi / 2 - 1e-6, 0.5], [0.1, -0.2, 0.3]]:
        R = TSR.rpy_to_rot(np.array(rpy))
        np.testing.assert_allclose(chart.rot_to_rpy(R), TSR.rot_to_rpy(R), atol=1e-12)


def test_rot_within_rpy_bounds_matches_oracle():
    chart = CanonicalTSRChart()
    Bw = np.array([[-pi / 4, pi / 4], [-pi / 4, pi / 4], [-pi / 4, pi / 4]])
    for rpy in [[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [pi, 0, pi], [0.0, 0.0, 0.0]]:
        R = TSR.rpy_to_rot(np.array(rpy))
        c_new, rpy_new = chart.rot_within_rpy_bounds(R, Bw)
        c_old, rpy_old = TSR.rot_within_rpy_bounds(R, Bw)
        assert list(c_new) == list(c_old)
        if rpy_old is None:
            assert rpy_new is None
        else:
            np.testing.assert_allclose(rpy_new, rpy_old, atol=1e-12)


# --- E-series: canonical embedding differential against the oracle ---


def _trans(xyz, rpy):
    return TSR.xyzrpy_to_trans(np.hstack([xyz, rpy]))


def test_E01_fixed_point():
    old, new = make_pair(Bw=np.zeros((6, 2)))
    assert_transforms_close(old.to_transform(np.zeros(6)), new.to_transform(np.zeros(6)))


def test_E02_translation_only():
    Bw = np.zeros((6, 2))
    Bw[0:3] = [-0.2, 0.2]
    old, new = make_pair(Bw=Bw)
    z = np.array([0.1, -0.05, 0.15, 0.0, 0.0, 0.0])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E03_rotation_axes():
    Bw = np.zeros((6, 2))
    Bw[3] = [-pi / 4, pi / 4]
    old, new = make_pair(Bw=Bw)
    assert_transforms_close(old.to_transform([0, 0, 0, 0.3, 0, 0]), new.to_transform([0, 0, 0, 0.3, 0, 0]))
    Bw[4] = [-pi / 4, pi / 4]
    old, new = make_pair(Bw=Bw)
    assert_transforms_close(old.to_transform([0, 0, 0, 0.3, 0.2, 0]), new.to_transform([0, 0, 0, 0.3, 0.2, 0]))
    Bw[5] = [-pi / 4, pi / 4]
    old, new = make_pair(Bw=Bw)
    assert_transforms_close(old.to_transform([0, 0, 0, 0.3, 0.2, -0.1]), new.to_transform([0, 0, 0, 0.3, 0.2, -0.1]))


def test_E04_all_angles_nonzero():
    Bw = np.zeros((6, 2))
    Bw[3:6] = [-pi / 2, pi / 2]
    old, new = make_pair(Bw=Bw)
    z = np.array([0, 0, 0, 0.3, -0.4, 0.5])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E05_nonunit_T0_w():
    Bw = np.zeros((6, 2))
    Bw[0:3] = [-0.1, 0.1]
    Bw[3:6] = [-pi / 4, pi / 4]
    T0_w = _trans([0.5, -0.3, 1.2], [0.4, -0.5, 1.1])
    old, new = make_pair(T0_w=T0_w, Bw=Bw)
    z = np.array([0.05, -0.02, 0.03, 0.1, -0.1, 0.2])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E06_nonunit_Tw_e():
    Bw = np.zeros((6, 2))
    Bw[0:3] = [-0.1, 0.1]
    Bw[3:6] = [-pi / 4, pi / 4]
    Tw_e = _trans([0.0, 0.0, 0.15], [0.0, 1.2, -0.7])
    old, new = make_pair(Tw_e=Tw_e, Bw=Bw)
    z = np.array([0.05, -0.02, 0.03, 0.1, -0.1, 0.2])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E07_both_frames_nonunit():
    Bw = np.zeros((6, 2))
    Bw[0:3] = [-0.1, 0.1]
    Bw[3:6] = [-pi / 4, pi / 4]
    T0_w = _trans([0.5, -0.3, 1.2], [0.4, -0.5, 1.1])
    Tw_e = _trans([0.0, 0.0, 0.15], [0.0, 1.2, -0.7])
    old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
    z = np.array([0.05, -0.02, 0.03, 0.1, -0.1, 0.2])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E08_wrap_and_outer_and_full():
    Bw = np.zeros((6, 2))
    Bw[0:3] = [-0.1, 0.1]
    Bw[3] = [3 * pi / 4, -3 * pi / 4]  # outer
    Bw[4] = [-pi, pi]  # full
    Bw[5] = [pi / 2 - 1e-6, -pi / 2 + 1e-6]
    old, new = make_pair(Bw=Bw)
    for z in [[0, 0, 0, pi, 0, pi], [0, 0, 0, -pi, 0.3, -2.0], [0.05, 0, 0, 0.9 * pi, -0.4, 2.0]]:
        assert_transforms_close(old.to_transform(np.array(z)), new.to_transform(np.array(z)))


def test_E08_legacy_positive_angular_span_over_one_turn_clamps_to_full_circle():
    Bw = np.zeros((6, 2))
    Bw[5] = [0.2, 0.2 + 4 * pi]
    old, new = make_pair(Bw=Bw)
    np.testing.assert_allclose(new.domain.cont_upper[5] - new.domain.cont_lower[5], 2 * pi, atol=1e-15)
    for yaw in [-pi, -0.4, 0.2, pi]:
        z = np.array([0, 0, 0, 0, 0, yaw])
        assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E08_negative_angular_span_beyond_one_turn_is_outside_canonical_contract():
    Bw = np.zeros((6, 2))
    Bw[5] = [2 * pi, -2 * pi - 0.1]
    with pytest.raises(ValueError, match="at least -2\\*pi wide"):
        GTSR.canonical(Bw=Bw)


def test_E08_negative_full_turn_accepts_ulp_roundoff_at_large_lower_bound():
    Bw = np.zeros((6, 2))
    lower = 128.80518381
    Bw[5] = [lower, lower - 2 * pi]
    old, new = make_pair(Bw=Bw)
    np.testing.assert_allclose(new.domain.cont_upper[5], new.domain.cont_lower[5], atol=1e-15)
    for yaw in [new.domain.cont_lower[5], -pi, pi]:
        z = np.array([0, 0, 0, 0, 0, yaw])
        assert old.is_valid(z)[5]
        assert new.is_valid(z)[5]
        assert_transforms_close(old.to_transform(z), new.to_transform(z))
    zero = np.zeros(6)
    assert not old.is_valid(zero)[5]
    assert not new.is_valid(zero)[5]


def test_E09_pitch_singularity():
    Bw = np.zeros((6, 2))
    Bw[3:6] = [-pi, pi]
    old, new = make_pair(Bw=Bw)
    for pitch in [pi / 2, -pi / 2, pi / 2 - 1e-6, pi / 2 + 1e-6, -pi / 2 + 1e-6]:
        z = np.array([0, 0, 0, 0.3, pitch, 0.5])
        assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E10_degenerate_and_mixed():
    Bw = np.zeros((6, 2))
    Bw[0] = [0.1, 0.1]  # fixed x
    Bw[1:3] = [-0.2, 0.2]  # free y,z
    Bw[3] = [0.5, 0.5]  # fixed roll
    Bw[4:6] = [-pi / 4, pi / 4]
    old, new = make_pair(Bw=Bw)
    z = np.array([0.1, 0.05, -0.05, 0.5, 0.1, -0.1])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


@pytest.mark.parametrize("seed", SEEDS)
def test_diff_5000_samples(seed):
    rng = np.random.default_rng(seed)
    for iteration in range(DIFF_SAMPLES_PER_SEED):
        T0_w = random_se3(rng)
        Tw_e = random_se3(rng)
        Bw = random_bounds(rng)
        old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        z = new.domain.sample(rng)
        assert_transforms_close(
            old.to_transform(z),
            new.to_transform(z),
            context=f"seed={seed}, iteration={iteration}, z={z!r}, Bw={Bw!r}",
        )


@pytest.mark.parametrize("pitch_sign", [-1.0, 1.0])
@pytest.mark.parametrize("delta", [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2])
@pytest.mark.parametrize("delta_sign", [-1.0, 1.0])
def test_coordinates_reconstructs_near_pitch_singularities(pitch_sign, delta, delta_sign):
    chart = CanonicalTSRChart()
    domain = GTSR.canonical(Bw=np.column_stack((-pi * np.ones(6), pi * np.ones(6)))).domain
    pitch = pitch_sign * pi / 2 + delta_sign * delta
    z = np.array([0.1, -0.2, 0.3, 0.7, pitch, -1.1])
    T_local = chart.forward(z)
    recovered = chart.coordinates(T_local, domain)
    assert_transforms_close(
        T_local,
        chart.forward(recovered),
        context=f"pitch_sign={pitch_sign}, delta={delta}, delta_sign={delta_sign}, recovered={recovered!r}",
    )


@pytest.mark.parametrize("pitch", [2.0, pi / 2, -pi / 2])
def test_coordinates_selects_domain_feasible_nonprincipal_solution(pitch):
    Bw = np.zeros((6, 2))
    Bw[3] = [0.6, 0.8]
    Bw[4] = [pitch - 0.1, pitch + 0.1] if abs(pitch) != pi / 2 else [pitch, pitch]
    Bw[5] = [-1.2, -1.0]
    region = GTSR.canonical(Bw=Bw)
    z = np.array([0.0, 0.0, 0.0, 0.7, pitch, -1.1])
    T_local = region.chart.forward(z)
    recovered = region.chart.coordinates(T_local, region.domain)
    assert all(region.domain.contains(recovered)), f"recovered={recovered!r}, Bw={Bw!r}"
    assert_transforms_close(T_local, region.chart.forward(recovered), context=f"recovered={recovered!r}, Bw={Bw!r}")


def test_coordinates_exact_singularity_preserves_wrapped_domain_endpoint():
    Bw = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.71970663, 1.92542320],
            [-pi / 2, -pi / 2],
            [2.32322348, 2.85843471],
        ]
    )
    region = GTSR.canonical(Bw=Bw)
    z = np.array([0.0, 0.0, 0.0, 1.82256492, -pi / 2, 2.59082909])
    T_local = region.chart.forward(z)
    recovered = region.chart.coordinates(T_local, region.domain)
    assert all(region.domain.contains(recovered)), f"recovered={recovered!r}, Bw={Bw!r}"
    assert_transforms_close(T_local, region.chart.forward(recovered), context=f"recovered={recovered!r}, Bw={Bw!r}")
