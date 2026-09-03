# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Numerical demonstrations for G-TSR expressivity (X series).

These tests *demonstrate* the constructions behind the P1/P2 proofs in
``docs/gtsr_expressivity_proofs.md``; they do not carry proof burden
(upstream §6.6). See ``docs/gtsr_geometry_validation_plan.md`` §6.6.

- X-01: a nonzero-finite-pitch screw ``S_h`` is represented by ``PoEChart([xi_h])``;
  ``q + 2*pi`` produces an axial shift of ``2*pi*h`` (not a wrap).
- X-02: pitch is recovered from ``se3_log`` of adjacent relative transforms.
- X-03: a traditional z-yaw box is strictly larger than the 1D screw curve
  (positive transverse deviation). Evidence only, not proof.
"""

import numpy as np

from tsr.gtsr.charts import PoEChart
from tsr.gtsr.domains import CoordinateDomain, LinearInterval
from tsr.gtsr.metrics import SE3Metric, se3_exp, se3_log
from tsr.gtsr.region import GTSR


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _screw_region(h=0.05, lo=-0.5, hi=0.5):
    """1D witness region S_h = {Exp(xi_h q) | q in [lo, hi]}, xi_h = [0,0,h,0,0,1]."""
    w = np.array([0.0, 0.0, 1.0])
    xi_h = np.hstack([h * w, w])
    chart = PoEChart([xi_h], representation="space")
    domain = CoordinateDomain([LinearInterval(lo, hi)])
    return GTSR(chart=chart, domain=domain)


def test_X01_nonzero_pitch_screw_is_a_gtsr_region():
    h = 0.05
    w = np.array([0.0, 0.0, 1.0])
    xi_h = np.hstack([h * w, w])
    chart = PoEChart([xi_h], representation="space")
    assert chart.periodic == (False,)

    region = GTSR(chart=chart, domain=CoordinateDomain([LinearInterval(-0.5, 0.5)]))

    q = 0.3
    T = region.to_transform([q])
    np.testing.assert_allclose(T[:3, :3], _rotz(q), atol=1e-15)
    np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, h * q], atol=1e-15)
    # forward == Exp(xi_h q).
    np.testing.assert_allclose(T, se3_exp(xi_h * q), atol=1e-14)

    # q + 2*pi (chart-level, outside the region's coordinate domain) advances
    # axially by 2*pi*h: a nonzero-finite-pitch screw is NOT 2*pi-periodic.
    T_wrap = chart.forward([q + 2 * np.pi])
    np.testing.assert_allclose(T_wrap[:3, :3], T[:3, :3], atol=1e-15)
    np.testing.assert_allclose(T_wrap[2, 3] - T[2, 3], 2 * np.pi * h, atol=1e-8)


def test_X02_pitch_estimate_from_relative_log():
    h = 0.05
    region = _screw_region(h=h)
    q = np.linspace(-0.5, 0.5, 21)
    for qi, qj in zip(q[:-1], q[1:]):
        rel = np.linalg.inv(region.to_transform([qi])) @ region.to_transform([qj])
        xi = se3_log(rel)  # [v, w]
        v_est, w_est = xi[0:3], xi[3:6]
        h_est = float(np.dot(w_est, v_est) / np.dot(w_est, w_est))
        np.testing.assert_allclose(h_est, h, atol=1e-8, rtol=0.0)


def test_X03_z_yaw_box_strictly_larger_than_screw_curve():
    """传统 z-yaw box 采样云相对 screw 曲线存在正横向偏离（box 严格大于 1 维 screw）。

    佐证，非证明：只展示一个传统 TSR 候选（z-yaw box）相对 screw 曲线有多余自由体积，
    不遍历所有传统 TSR（上游 §6.6）。
    """
    h = 0.05
    screw = _screw_region(h=h)
    metric = SE3Metric()

    # 最接近 screw 曲线的传统 TSR：z 平移 + yaw 旋转，其余坐标零宽。
    Bw = np.zeros((6, 2))
    Bw[2] = [-0.5, 0.5]  # z
    Bw[5] = [-0.5, 0.5]  # yaw
    box = GTSR.canonical(Bw=Bw)

    rng = np.random.default_rng(20260902)
    distances = []
    for _ in range(200):
        z = rng.uniform(-0.5, 0.5)
        yaw = rng.uniform(-0.5, 0.5)
        box_point = box.to_transform(np.array([0.0, 0.0, z, 0.0, 0.0, yaw]))
        distances.append(screw.project(box_point, metric).distance)

    # box 含远离 screw 曲线的点（多余自由体积，横向偏离显著大于 0）。
    assert max(distances) > 0.1

    # on-curve 点（z = h * yaw）可被精确恢复到 screw 曲线上（残差 ~ 0）。
    on_curve = box.to_transform(np.array([0.0, 0.0, h * 0.4, 0.0, 0.0, 0.4]))
    assert screw.project(on_curve, metric).distance <= 1e-8
