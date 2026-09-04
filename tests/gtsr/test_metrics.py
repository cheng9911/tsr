# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest

from tsr.gtsr.metrics import (
    SE3Metric,
    se3_adjoint,
    se3_exp,
    se3_hat,
    se3_left_jacobian_inverse,
    se3_log,
    se3_vee,
    skew,
    so3_log,
)


def _random_se3(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    T = np.eye(4)
    T[:3, :3] = q
    T[:3, 3] = rng.normal(size=3)
    return T


def test_M01_hat_vee_roundtrip():
    xi = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    np.testing.assert_allclose(se3_vee(se3_hat(xi)), xi, atol=1e-15)
    H = se3_hat(xi)
    assert H.shape == (4, 4)
    np.testing.assert_allclose(H[3, :], [0, 0, 0, 0], atol=1e-15)


def test_M01_vee_rejects_nonfinite_matrix():
    H = np.zeros((4, 4))
    H[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        se3_vee(H)


def test_M02_skew_is_antisymmetric():
    w = np.array([0.1, -0.2, 0.3])
    np.testing.assert_allclose(skew(w), -skew(w).T, atol=1e-15)


def test_M03_so3_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        w = so3_log(R)
        R2 = np.eye(3)
        th = np.linalg.norm(w)
        if th > 1e-12:
            W = skew(w)
            R2 = np.eye(3) + np.sin(th) / th * W + (1 - np.cos(th)) / th**2 * (W @ W)
        np.testing.assert_allclose(R2, R, atol=1e-10)


def test_M04_se3_exp_log_roundtrip():
    rng = np.random.default_rng(42)
    for _ in range(50):
        xi = rng.normal(size=6) * 0.5  # keep |w| < pi
        T = se3_exp(xi)
        xi2 = se3_log(T)
        np.testing.assert_allclose(xi2, xi, atol=1e-10)
        # and exp(log(T)) for a random SE(3)
        T3 = _random_se3(rng)
        np.testing.assert_allclose(se3_exp(se3_log(T3)), T3, atol=1e-10)


def test_M05_adjoint_consistency():
    rng = np.random.default_rng(7)
    T = _random_se3(rng)
    xi = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    lhs = se3_adjoint(T) @ xi
    rhs = se3_vee(T @ se3_hat(xi) @ np.linalg.inv(T))
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_M06_left_jacobian_inverse_matches_finite_difference():
    xi = np.array([0.1, -0.2, 0.3, 0.3, -0.4, 0.5])
    Jl_inv = se3_left_jacobian_inverse(xi)
    # central finite-difference of the LEFT Jacobian: (exp(xi)^-1 d exp(xi))^vee
    Jl = np.zeros((6, 6))
    h = 1e-6
    for i in range(6):
        d = np.zeros(6)
        d[i] = h
        Jl[:, i] = se3_vee(np.linalg.inv(se3_exp(xi)) @ (se3_exp(xi + d) - se3_exp(xi - d))) / (2 * h)
    np.testing.assert_allclose(Jl @ Jl_inv, np.eye(6), atol=1e-8)


def test_M07_error_body_jacobian_and_metric_basics():
    rng = np.random.default_rng(20260902)
    metric = SE3Metric(rotation_weight=0.3)
    A, B = _random_se3(rng), _random_se3(rng)
    # symmetric distance
    np.testing.assert_allclose(metric.distance(A, B), metric.distance(B, A), atol=1e-12)
    assert metric.distance(A, A) < 1e-12
    # error_body_jacobian against finite differences (body-frame perturbation)
    E = metric.error_body_jacobian(A, B)
    E_fd = np.zeros((6, 6))
    h = 1e-7
    for i in range(6):
        d = np.zeros(6)
        d[i] = h
        rp = metric.error(A @ se3_exp(d), B)
        rm = metric.error(A @ se3_exp(-d), B)
        E_fd[:, i] = (rp - rm) / (2 * h)
    np.testing.assert_allclose(E, E_fd, atol=1e-6)
    # rotation_weight scaling on a pure rotation error
    R = np.eye(4)
    R[:3, :3] = se3_exp([0, 0, 0, 0, 0, 0.5])[:3, :3]
    assert metric.distance(np.eye(4), R) == pytest.approx(0.3 * 0.5, abs=1e-12)


def test_M08_so3_log_pi_rotation():
    # At exactly θ=π the antisymmetric part of R vanishes, so the θ/(2 sinθ)
    # form in so3_log degenerates to a spurious zero axis (identity). The
    # rotation axis must instead be recovered from R + I = 2 a aᵀ.
    a = np.array([1.0, -2.0, 0.5])
    a = a / np.linalg.norm(a)
    R = se3_exp(np.hstack([[0.0, 0.0, 0.0], np.pi * a]))[:3, :3]
    w = so3_log(R)
    assert np.linalg.norm(w) == pytest.approx(np.pi, abs=1e-10)
    assert abs(np.dot(w, a)) == pytest.approx(np.linalg.norm(w), abs=1e-10)
    # round-trip: exp(log(R)) == R
    np.testing.assert_allclose(se3_exp(np.hstack([[0.0, 0.0, 0.0], w]))[:3, :3], R, atol=1e-10)
    # the metric must NOT report a 180° rotation as identity
    metric = SE3Metric()
    r = metric.error(np.eye(4), se3_exp(np.hstack([[0.0, 0.0, 0.0], np.pi * a])))
    assert np.linalg.norm(r[3:6]) == pytest.approx(np.pi, abs=1e-10)


@pytest.mark.parametrize("delta", [0.0, 1e-10, 1e-8, 1e-6])
def test_M09_so3_and_se3_log_random_axes_near_pi_roundtrip(delta):
    rng = np.random.default_rng(20260903)
    for _ in range(250):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        xi = np.hstack((rng.normal(size=3), (np.pi - delta) * axis))
        T = se3_exp(xi)

        recovered_R = se3_exp(np.hstack((np.zeros(3), so3_log(T[:3, :3]))))[:3, :3]
        np.testing.assert_allclose(recovered_R, T[:3, :3], rtol=0.0, atol=1e-10)
        np.testing.assert_allclose(se3_exp(se3_log(T)), T, rtol=0.0, atol=1e-10)


def test_M10_principal_log_residual_is_not_a_mathematical_metric():
    """Limitation canary: the compatibility name must not imply triangle inequality."""
    xi_a = np.array(
        [
            -2.846905572150202,
            0.4838265621800239,
            2.6171445119208743,
            1.5205491102877773,
            -0.6905661330922319,
            1.2585542302394472,
        ]
    )
    xi_b = np.array(
        [
            -0.07022676080177756,
            1.7362252912372718,
            -1.746860569184871,
            1.268868688332276,
            0.34793094906799144,
            0.1339835733116889,
        ]
    )
    xi_c = np.array(
        [
            2.421395130178081,
            3.3820542492183465,
            -0.9516262667279863,
            1.7292616414824393,
            1.7389551724334569,
            0.4400481754030808,
        ]
    )
    metric = SE3Metric(rotation_weight=1.0)
    A, B, C = se3_exp(xi_a), se3_exp(xi_b), se3_exp(xi_c)

    ab = metric.distance(A, B)
    bc = metric.distance(B, C)
    ac = metric.distance(A, C)

    assert ac > ab + bc + 0.1


@pytest.mark.parametrize("weight", [0.0, -1.0, np.nan, np.inf])
def test_se3_metric_requires_strictly_positive_rotation_weight(weight):
    with pytest.raises(ValueError, match="strictly positive"):
        SE3Metric(rotation_weight=weight)
