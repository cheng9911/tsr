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
from tsr.utils import wrap_to_interval


# --- C-01 ---------------------------------------------------------------
def test_C01_same_coordinate_same_pose(rng):
    for _ in range(50):
        T0_w, Tw_e, Bw = random_se3(rng), random_se3(rng), random_bounds(rng)
        old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        z = new.domain.sample(rng)
        assert_transforms_close(old.to_transform(z), new.to_transform(z))


# --- C-02 ---------------------------------------------------------------
def test_C02_contains_boundary_corpus(rng):
    T0_w, Tw_e = random_se3(rng), random_se3(rng)
    Bw = np.array([
        [-0.1, 0.1], [-0.1, 0.1], [0.0, 0.02],
        [-pi / 4, pi / 4], [-pi / 4, pi / 4], [3 * pi / 4, -3 * pi / 4],
    ])
    old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
    # 直接由局部坐标构造世界变换，绕过 to_transform 的边界校验（contains 的职责正是判断越界）
    zs = [
        [0.0, 0.0, 0.01, 0.0, 0.0, pi],                # interior (outer yaw)
        [0.1, 0.1, 0.02, pi / 4, pi / 4, -3 * pi / 4],  # endpoints
        [0.0, 0.0, 0.01, 0.0, 0.0, 0.0],               # yaw=0 -> outside outer
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],                # far outside
    ]
    transforms = [new.T0_w @ new.chart.forward(np.array(z)) @ new.Tw_e for z in zs]
    transforms.append(np.eye(4))                        # world origin
    for T in transforms:
        assert old.contains(T) == new.contains(T), f"contains mismatch for T={T}"


# --- C-03 ---------------------------------------------------------------
def test_C03_coordinates_forward_roundtrip(rng):
    for _ in range(50):
        T0_w, Tw_e, Bw = random_se3(rng), random_se3(rng), random_bounds(rng)
        old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        z = new.domain.sample(rng)
        T = new.to_transform(z)
        # 逆映射位姿级一致：old/new 的 to_xyzrpy 返回同一 SE(3) 位姿。
        # 不比较欧拉角数组（old 有 _Bw_cont 切片 bug，恒 fallback 主值 rpy）；
        # 也不断言精确 roundtrip（near-singular 邻域内 old/new 都按 EPSILON=1e-3
        # 阈值 snap 到 ±π/2，roundtrip 仅在该邻域外精确——这是 oracle 的既有行为）。
        assert_transforms_close(
            TSR.xyzrpy_to_trans(old.to_xyzrpy(T)),
            TSR.xyzrpy_to_trans(new.to_xyzrpy(T)),
        )


# --- C-04 ---------------------------------------------------------------
def test_C04_legacy_distance(rng):
    for _ in range(50):
        T0_w, Tw_e, Bw = random_se3(rng), random_se3(rng), random_bounds(rng)
        old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        z = new.domain.sample(rng)
        T_in = new.to_transform(z)
        d_old, _ = old.distance(T_in)
        d_new, _ = new.distance(T_in)
        assert d_old == 0.0 and d_new == 0.0
        # 非包含点：距离与 bwopt 重构位姿一致（_displacement_to_tsr 路径无 bug）
        T_out = np.eye(4)
        d_old, bw_old = old.distance(T_out)
        d_new, bw_new = new.distance(T_out)
        assert d_new == pytest.approx(d_old, abs=1e-12)
        assert_transforms_close(old.to_transform(bw_old), new.to_transform(bw_new))


# --- C-05 ---------------------------------------------------------------
def test_C05_sampling_mapping_compat(rng):
    T0_w, Tw_e, Bw = random_se3(rng), random_se3(rng), random_bounds(rng)
    old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
    u = rng.random(6)
    new_z = new.domain.map_from_unit_cube(u)
    old_z = old._Bw_cont[:, 0] + (old._Bw_cont[:, 1] - old._Bw_cont[:, 0]) * u
    old_z[3:6] = wrap_to_interval(old_z[3:6])
    np.testing.assert_allclose(new_z, old_z, atol=1e-12)

    # 固定分量保留：从合法采样中取定值，避免随机 Bw 下固定值越界。
    z0 = new.domain.sample(np.random.default_rng(1))
    partial = np.full(6, np.nan)
    partial[0] = z0[0]
    partial[4] = z0[4]
    s_new = new.sample_xyzrpy(partial, rng=np.random.default_rng(0))
    assert s_new[0] == pytest.approx(z0[0])
    assert s_new[4] == pytest.approx(z0[4])


# --- 域连续边界与旧 _Bw_cont 逐位一致（强不变量） --------------------------
def test_domain_cont_bounds_match_oracle(rng):
    for _ in range(20):
        Bw = random_bounds(rng)
        old = TSR(Bw=Bw)
        new = GTSR.canonical(Bw=Bw)
        np.testing.assert_allclose(new.domain.cont_lower, old._Bw_cont[:, 0], atol=1e-15)
        np.testing.assert_allclose(new.domain.cont_upper, old._Bw_cont[:, 1], atol=1e-15)
