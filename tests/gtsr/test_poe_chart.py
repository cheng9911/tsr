# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest

from tests.gtsr.helpers import make_seed9917_region
from tsr.gtsr.charts import PoEChart
from tsr.gtsr.domains import CoordinateDomain, LinearInterval
from tsr.gtsr.metrics import SE3Metric, se3_adjoint, se3_exp, se3_vee, so3_log
from tsr.gtsr.region import GTSR


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


@pytest.mark.parametrize("axis_index", [0, 1, 2])
def test_G01_single_translation_axis(axis_index):
    axis = np.zeros(6)
    axis[axis_index] = 1.0
    chart = PoEChart([axis], representation="space")
    T = chart.forward([0.5])
    expected = np.eye(4)
    expected[axis_index, 3] = 0.5
    np.testing.assert_allclose(T, expected, atol=1e-15)


def test_G02_pure_rotation_rodrigues():
    w = np.array([1.0, 2.0, -1.0])
    w = w / np.linalg.norm(w)
    chart = PoEChart([np.hstack([0, 0, 0, w])], representation="space")
    th = 0.7
    T = chart.forward([th])
    W = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + np.sin(th) * W + (1 - np.cos(th)) * (W @ W)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-15)
    np.testing.assert_allclose(T[:3, 3], [0, 0, 0], atol=1e-15)


def test_G03_off_origin_zero_pitch_rotation_about_point():
    # rotation about the z-axis line through q = (1, 2, 0): S = [v, w] with v = -w x q
    w = np.array([0.0, 0.0, 1.0])
    q = np.array([1.0, 2.0, 0.0])
    v = -np.cross(w, q)
    chart = PoEChart([np.hstack([v, w])], representation="space")
    th = 0.4
    T = chart.forward([th])
    R = _rotz(th)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-15)
    np.testing.assert_allclose(T[:3, 3], (np.eye(3) - R) @ q, atol=1e-15)


def test_G04_nonzero_pitch_screw_linear():
    # pitch h: screw axis S = [v, w], w unit, h = w . v. q + 2pi shifts along axis by 2*pi*h.
    w = np.array([0.0, 0.0, 1.0])
    h = 0.15
    v = h * w
    chart = PoEChart([np.hstack([v, w])], representation="space")
    assert chart.periodic == (False,)
    q = 0.3
    T = chart.forward([q])
    R = _rotz(q)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-15)
    np.testing.assert_allclose(T[:3, 3], [0, 0, h * q], atol=1e-15)
    T2 = chart.forward([q + 2 * np.pi])
    np.testing.assert_allclose(T2[:3, :3], T[:3, :3], atol=1e-15)  # same rotation
    np.testing.assert_allclose(T2[:3, 3], [0, 0, h * (q + 2 * np.pi)], atol=1e-15)  # NOT wrapped


def test_G05_product_order_two_noncommuting_generators():
    S_z = np.array([0, 0, 0, 0, 0, 1])
    S_x = np.array([0, 0, 0, 1, 0, 0])
    th1, th2 = 0.4, 0.6
    chart = PoEChart([S_z, S_x], representation="space")
    T = chart.forward([th1, th2])
    expected_R = _rotz(th1) @ _rotx(th2)  # Exp(S1 z1) Exp(S2 z2) M
    np.testing.assert_allclose(T[:3, :3], expected_R, atol=1e-15)
    # swapped axes => different result (non-commuting), still matches analytic order
    chart2 = PoEChart([S_x, S_z], representation="space")
    T2 = chart2.forward([th2, th1])
    expected_R2 = _rotx(th2) @ _rotz(th1)
    np.testing.assert_allclose(T2[:3, :3], expected_R2, atol=1e-15)
    assert not np.allclose(T[:3, :3], T2[:3, :3])


@pytest.mark.parametrize("order", [[0.0, 1.0], [False, 1], ["0", 1]])
def test_G05_order_rejects_non_integer_indices_instead_of_coercing(order):
    axes = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]]
    with pytest.raises(TypeError, match="integer indices"):
        PoEChart(axes, representation="space", order=order)


def test_G05_order_accepts_numpy_integer_indices():
    axes = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]]
    chart = PoEChart(axes, representation="space", order=np.array([1, 0], dtype=np.int64))
    assert chart.dim == 2


def test_G05_explicit_order_changes_the_product_order():
    axes = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0]]
    z = [0.4, 0.6]
    chart = PoEChart(axes, representation="space", order=[1, 0])
    expected = se3_exp(np.asarray(axes[1]) * z[1]) @ se3_exp(np.asarray(axes[0]) * z[0])
    np.testing.assert_allclose(chart.forward(z), expected, atol=1e-15)


def test_G06_space_body_adjoint_consistency():
    M = se3_exp([0.1, -0.2, 0.3, 0.3, -0.4, 0.5])
    B = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    S = se3_adjoint(M) @ B  # space screw = Ad_M * body screw
    body = PoEChart([B], representation="body", M=M)
    space = PoEChart([S], representation="space", M=M)
    z = [0.7]
    np.testing.assert_allclose(body.forward(z), space.forward(z), atol=1e-12)


def test_G07_external_frames_are_applied_by_region():
    chart = PoEChart([[0, 0, 0, 0, 0, 1]], representation="space")
    domain = CoordinateDomain.from_bounds([[-0.8, 0.8]], chart.periodic, chart.periods)
    T0_w = se3_exp([0.3, -0.2, 0.1, 0.2, 0.0, -0.1])
    Tw_e = se3_exp([-0.1, 0.4, 0.2, 0.0, -0.3, 0.1])
    region = GTSR(T0_w=T0_w, Tw_e=Tw_e, chart=chart, domain=domain)
    z = np.array([0.37])

    np.testing.assert_allclose(
        region.to_transform(z),
        T0_w @ chart.forward(z) @ Tw_e,
        atol=1e-14,
    )


def test_poe_rotational_axes_require_unit_angular_part():
    periodic_chart = PoEChart([[0, 0, 0, 0, 0, 1]], representation="space")
    screw_chart = PoEChart([[0.2, -0.1, 0.0, 1, 0, 0]], representation="space")
    assert periodic_chart.periods == (2 * np.pi,)
    assert screw_chart.periods == (None,)

    eps = np.finfo(float).eps
    for angular_norm in (np.nextafter(0.0, 1.0), 1e-13, 0.5, 1.0 + 5e-11, 2.0):
        with pytest.raises(ValueError, match="unit angular part"):
            PoEChart([[0, 0, 0, 0, 0, angular_norm]], representation="space")

    near_unit = PoEChart([[0, 0, 32 * eps, 0, 0, 1 + 32 * eps]], representation="space")
    assert near_unit.periodic == (True,)
    # 2π-periodicity is verified with translation and rotation split (not a mixed-dimension
    # matrix max): the rotation-angle error e_R and the translation 2-norm e_t scale with ‖v‖.
    v = np.array([0.0, 0.0, 32 * eps])
    T_ref = near_unit.forward([0.3])
    T_wrap = near_unit.forward([0.3 + 2 * np.pi])
    e_R = float(np.linalg.norm(so3_log(T_wrap[:3, :3].T @ T_ref[:3, :3])))
    e_t = float(np.linalg.norm(T_wrap[:3, 3] - T_ref[:3, 3]))
    assert e_R <= 1e-12
    assert e_t <= 1e-12 * max(1.0, float(np.linalg.norm(v)))

    # A zero-pitch rotation about a distant axis makes the scale term active.
    large_v = np.array([1000.0, 0.0, 0.0])
    large_lever_arm = PoEChart([np.hstack((large_v, [0.0, 0.0, 1.0]))], representation="space")
    T_ref = large_lever_arm.forward([0.3])
    T_wrap = large_lever_arm.forward([0.3 + 2 * np.pi])
    e_R = float(np.linalg.norm(so3_log(T_wrap[:3, :3].T @ T_ref[:3, :3])))
    e_t = float(np.linalg.norm(T_wrap[:3, 3] - T_ref[:3, 3]))
    assert np.linalg.norm(large_v) > 1.0
    assert e_R <= 1e-12
    assert e_t <= 1e-12 * max(1.0, float(np.linalg.norm(large_v)))


def test_generalized_sample_to_contains_and_partial_coordinates():
    chart = PoEChart(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
        representation="space",
    )
    domain = CoordinateDomain([LinearInterval(-0.5, 0.5), LinearInterval(1.0, 2.0)])
    region = GTSR(chart=chart, domain=domain)

    default_sample = region.sample(rng=np.random.default_rng(7))
    np.testing.assert_array_equal(default_sample, region.sample(rng=np.random.default_rng(7)))
    assert region.contains(default_sample, metric=SE3Metric())

    expected_rng = np.random.default_rng(20260903)
    expected_z = np.array([0.25, 1.0 + expected_rng.random()])
    sampled = region.sample(partial=[0.25, np.nan], rng=np.random.default_rng(20260903))
    positional_sample = region.sample([0.25, np.nan], rng=np.random.default_rng(20260903))

    np.testing.assert_allclose(sampled, region.to_transform(expected_z), atol=1e-15)
    np.testing.assert_array_equal(positional_sample, sampled)
    assert region.contains(sampled, metric=SE3Metric())


def test_high_dimensional_poe_sample_to_contains():
    region, metric, rng = make_seed9917_region()

    for _ in range(10):
        sampled = region.sample(rng=rng)
        result = region.project(sampled, metric)
        assert result.distance <= 1e-8
        assert region.contains(sampled, metric=metric)


def test_G08_se3_sanity_all_outputs():
    axes = [[0.2, 0.1, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
    chart = PoEChart(axes, representation="space")
    rng = np.random.default_rng(0)
    for _ in range(20):
        z = rng.normal(size=2)
        T = chart.forward(z)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
        assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-12)


def test_G09_jacobian_matches_central_finite_difference():
    axes = [
        [0.2, 0.1, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.15, 1.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0, 1.0, 0.0],
    ]
    for representation in ("space", "body"):
        chart = PoEChart(axes, representation=representation)
        rng = np.random.default_rng(42)
        for _ in range(10):
            z = rng.normal(size=3)
            J = chart.jacobian(z)
            assert J.shape == (6, 3)
            J_fd = np.zeros((6, 3))
            h = 1e-6
            for i in range(3):
                d = np.zeros(3)
                d[i] = h
                Tp = chart.forward(z + d)
                Tm = chart.forward(z - d)
                J_fd[:, i] = se3_vee(np.linalg.inv(chart.forward(z)) @ (Tp - Tm)) / (2 * h)
            np.testing.assert_allclose(J, J_fd, atol=1e-6)


def test_G09_canonical_chart_jacobian_matches_central_finite_difference():
    from tsr.gtsr.charts import CanonicalTSRChart

    chart = CanonicalTSRChart()
    rng = np.random.default_rng(20260902)
    for _ in range(10):
        z = rng.normal(size=6)
        J = chart.jacobian(z)
        assert J.shape == (6, 6)
        J_fd = np.zeros((6, 6))
        h = 1e-6
        for i in range(6):
            d = np.zeros(6)
            d[i] = h
            J_fd[:, i] = se3_vee(np.linalg.inv(chart.forward(z)) @ (chart.forward(z + d) - chart.forward(z - d))) / (
                2 * h
            )
        np.testing.assert_allclose(J, J_fd, atol=1e-6)
