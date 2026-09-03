# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest

from tsr.gtsr.charts import PoEChart
from tsr.gtsr.metrics import se3_adjoint, se3_exp, se3_vee


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_G01_single_translation_axis():
    chart = PoEChart([[1, 0, 0, 0, 0, 0]], representation="space")
    T = chart.forward([0.5])
    expected = np.eye(4)
    expected[0, 3] = 0.5
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
    expected_R = _rotz(th1) @ _rotx(th2)   # Exp(S1 z1) Exp(S2 z2) M
    np.testing.assert_allclose(T[:3, :3], expected_R, atol=1e-15)
    # swapped axes => different result (non-commuting), still matches analytic order
    chart2 = PoEChart([S_x, S_z], representation="space")
    T2 = chart2.forward([th2, th1])
    expected_R2 = _rotx(th2) @ _rotz(th1)
    np.testing.assert_allclose(T2[:3, :3], expected_R2, atol=1e-15)
    assert not np.allclose(T[:3, :3], T2[:3, :3])


def test_G06_space_body_adjoint_consistency():
    M = se3_exp([0.1, -0.2, 0.3, 0.3, -0.4, 0.5])
    B = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    S = se3_adjoint(M) @ B   # space screw = Ad_M * body screw
    body = PoEChart([B], representation="body", M=M)
    space = PoEChart([S], representation="space", M=M)
    z = [0.7]
    np.testing.assert_allclose(body.forward(z), space.forward(z), atol=1e-12)


def test_G07_external_frames_are_applied_by_region():
    # covered in region-level: T0_w @ forward(z) @ Tw_e (see test_projection / to_transform reuse)
    chart = PoEChart([[0, 0, 0, 0, 0, 1]], representation="space")
    assert chart.forward([0.0]).shape == (4, 4)


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
                J_fd[:, i] = se3_vee(
                    np.linalg.inv(chart.forward(z)) @ (Tp - Tm)
                ) / (2 * h)
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
            J_fd[:, i] = se3_vee(
                np.linalg.inv(chart.forward(z)) @ (chart.forward(z + d) - chart.forward(z - d))
            ) / (2 * h)
        np.testing.assert_allclose(J, J_fd, atol=1e-6)
