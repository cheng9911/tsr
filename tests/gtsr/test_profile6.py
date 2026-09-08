# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Unit tests for the 6D SE(3)/pose6 math primitives (Sprint F, Task 1)."""

from __future__ import annotations

import numpy as np

from tsr.gtsr.profile6 import (
    SE3_METRIC_SCALE,
    euler_from_matrix,
    matrix_from_euler,
    matrix_from_euler_batched,
    pose6_from_se3,
    pose6_from_se3_batched,
    se3_exp,
    se3_exp_batched,
    se3_from_pose6,
    se3_from_pose6_batched,
    se3_inverse,
    se3_inverse_batched,
    se3_log,
    socket_frame6,
)


def test_metric_scale_frozen():
    np.testing.assert_array_equal(
        SE3_METRIC_SCALE, np.array([1.0, 1.0, 1.0, 0.03, 0.03, 0.03])
    )


def test_so3_exp_log_roundtrip():
    from tsr.gtsr.profile6 import so3_exp, so3_log

    rng = np.random.default_rng(0)
    for _ in range(10):
        w = rng.uniform(-1.0, 1.0, size=3)
        R = so3_exp(w)
        np.testing.assert_allclose(so3_log(R), w, atol=1e-10)


def test_se3_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(10):
        xi = rng.uniform(-0.5, 0.5, size=6)
        T = se3_exp(xi)
        np.testing.assert_allclose(se3_log(T), xi, atol=1e-9)


def test_se3_exp_batched_matches_scalar():
    rng = np.random.default_rng(2)
    twists = rng.uniform(-0.5, 0.5, size=(5, 6))
    batched = se3_exp_batched(twists)
    for i in range(5):
        np.testing.assert_allclose(batched[i], se3_exp(twists[i]), atol=1e-12)


def test_se3_inverse_batched_matches_scalar():
    rng = np.random.default_rng(3)
    twists = rng.uniform(-0.5, 0.5, size=(4, 6))
    T = se3_exp_batched(twists)
    batched = se3_inverse_batched(T)
    for i in range(4):
        np.testing.assert_allclose(batched[i], se3_inverse(T[i]), atol=1e-12)
        np.testing.assert_allclose(batched[i] @ T[i], np.eye(4), atol=1e-12)


def test_pose6_euler_convention_yaw_only():
    # A pure yaw rotation must be Rx(0) Ry(0) Rz(yaw) = Rz(yaw).
    yaw = 0.3
    R = matrix_from_euler(0.0, 0.0, yaw)
    np.testing.assert_allclose(R, np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0],
         [np.sin(yaw), np.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]]
    ), atol=1e-12)


def test_pose6_euler_roundtrip():
    rng = np.random.default_rng(4)
    for _ in range(10):
        rpy = rng.uniform(-0.5, 0.5, size=3)
        np.testing.assert_allclose(
            euler_from_matrix(matrix_from_euler(*rpy)), rpy, atol=1e-10
        )


def test_pose6_se3_roundtrip():
    rng = np.random.default_rng(5)
    for _ in range(10):
        pose6 = rng.uniform(-0.5, 0.5, size=6)
        np.testing.assert_allclose(
            pose6_from_se3(se3_from_pose6(pose6)), pose6, atol=1e-10
        )


def test_pose6_se3_batched_roundtrip():
    rng = np.random.default_rng(6)
    pose6 = rng.uniform(-0.5, 0.5, size=(7, 6))
    np.testing.assert_allclose(
        pose6_from_se3_batched(se3_from_pose6_batched(pose6)), pose6, atol=1e-10
    )


def test_matrix_from_euler_batched_matches_scalar():
    rng = np.random.default_rng(7)
    rpy = rng.uniform(-0.5, 0.5, size=(6, 3))
    batched = matrix_from_euler_batched(rpy)
    for i in range(6):
        np.testing.assert_allclose(
            batched[i], matrix_from_euler(*rpy[i]), atol=1e-12
        )


def test_socket_frame6():
    nominal = se3_from_pose6(np.array([0.1, 0.2, 0.3, 0.1, -0.1, 0.2]))
    context = np.array([0.01, -0.02, 0.03, 0.05, -0.05, 0.1])
    expected = nominal @ se3_from_pose6(context)
    np.testing.assert_allclose(socket_frame6(context, nominal), expected, atol=1e-12)
