# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
from numpy import pi

from tsr.gtsr.charts import CanonicalTSRChart
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
