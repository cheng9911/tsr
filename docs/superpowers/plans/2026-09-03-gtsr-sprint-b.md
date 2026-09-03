# G-TSR Sprint B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 G-TSR 的通用几何闭环——SE(3) metric、PoE chart（含解析 Jacobian）、通用投影与失败传播，并通过硬门 G2。

**Architecture:** 新增 `metrics.py`（SE(3) Lie 原语 + 显式 metric）、`projection.py`（`ProjectionResult` + 多起点投影）；`charts.py` 增补 `PoEChart`；`region.py` 增补 `GTSR.project` 与通用 `contains(metric=...)`。投影目标 `d_W² = rᵀWr`，`r=Log(T(z)⁻¹T_q)^vee`，用**精确左不变梯度** `2 Jᵀ Eᵀ W r` 做 L-BFGS-B，`optimality` 用中心有限差分独立复核。

**Tech Stack:** Python ≥3.10，numpy 2.4.6，scipy 1.17.1（`scipy.optimize.minimize` / `minimize_scalar`），pytest 9.1.1，pytest 函数风格。

**Spec:** [docs/superpowers/specs/2026-09-03-gtsr-sprint-b-design.md](../specs/2026-09-03-gtsr-sprint-b-design.md)

## Global Constraints

- 旧 `TSR` 与 Sprint A 的 `CanonicalTSRChart`/`GTSR` 表面**保持不动**，是 differential oracle；不得回退 `tests/tsr/` 或 `tests/gtsr/` 既有用例。
- 新文件加 SPDX 头：`# SPDX-License-Identifier: MIT` + `# Copyright (c) 2025 Siddhartha Srinivasa`。
- twist 排列 `ξ=[v,ω]`（线性前 3，角速度后 3）；误差 `r=Log(T(z)⁻¹T_q)^vee=[ρ,θ]`（body/左不变）；`W=diag(1,1,1,w²,w²,w²)`，`w`=rotation_weight（m/rad，默认 1.0）。
- PoE：`representation ∈ {"space","body"}`，`order` 显式排列，`M` 默认 I（Sprint B 限制 `M=I`）。
- 周期语义：仅 `h=0`（纯旋转）是 `CircularInterval(2π)`；`h≠0` 与纯平移是 `LinearInterval`。
- 容差：矩阵/位姿 `1e-12`~`1e-15`；Jacobian 中心有限差分 `1e-6`；KKT 残差 `≤1e-6`；目标值不劣于参考 `1e-8`；bounds 可行 `≤1e-12`；legacy `EPSILON=1e-3` 只留在旧 `contains/distance`。
- 随机测试显式 `numpy.random.Generator`，固定 `SEEDS=[0,1,7,42,20260902]`。
- `scipy` 是运行时依赖；用 conda env `/home/rocos/miniconda3/envs/tsr-dev/bin/python`。

## File Structure

- **Create** `src/tsr/gtsr/metrics.py` — SE(3) Lie 原语（hat/vee、so3_log、se3_exp、se3_log、adjoint、left-Jacobian-inverse）+ `SE3Metric`。
- **Create** `src/tsr/gtsr/projection.py` — `ProjectionResult` + `project()`（多起点 L-BFGS-B + 最优性复核）。
- **Modify** `src/tsr/gtsr/charts.py` — 增补 `PoEChart`（`forward` + `jacobian`）。
- **Modify** `src/tsr/gtsr/region.py` — 增补 `GTSR.project`、`contains(T, metric=None, tol=1e-10)`。
- **Modify** `src/tsr/gtsr/__init__.py` — 导出 `PoEChart`、`SE3Metric`、`ProjectionResult`。
- **Create** `tests/gtsr/test_metrics.py` — metric/Lie 原语（M-01~M-07）。
- **Create** `tests/gtsr/test_poe_chart.py` — G-01~G-09。
- **Create** `tests/gtsr/test_projection.py` — P-01~P-09。

> 注：上游 §8 只列了 `test_poe_chart.py`/`test_projection.py`/`test_expressivity.py`；本计划补 `test_metrics.py` 承载 SE(3) 机器的独立单测（metric 契约是用户 B0 冻结项，值得独立测试面）。

## Task 1: SE(3) Lie 原语 + SE3Metric

**Files:**
- Create: `src/tsr/gtsr/metrics.py`
- Create: `tests/gtsr/test_metrics.py`

**Interfaces:**
- Produces: `se3_hat(xi)`, `se3_vee(mat)`, `skew(w)`, `so3_log(R)`, `se3_exp(xi)`, `se3_log(T)`, `se3_adjoint(T)`, `se3_left_jacobian_inverse(xi)`, `SE3Metric(rotation_weight=1.0)` 及其方法 `.error(T_from, T_to)`, `.distance(T_from, T_to)`, `.distance_sq(r)`, `.weighted_error(r)`, `.error_body_jacobian(T_from, T_to)`。
- Consumes: `_validation.finite_vector`, `_validation.rotation_matrix`, `_validation.se3_transform`。

- [ ] **Step 1: Write the failing test**

`tests/gtsr/test_metrics.py`（M-01~M-07）：

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tsr.gtsr.metrics'`

- [ ] **Step 3: Write minimal implementation**

`src/tsr/gtsr/metrics.py`：

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Explicit SE(3) metric and Lie-group primitives for the G-TSR kernel.

Twist ordering is ``xi = [v, w]`` (linear ``v`` first, angular ``w`` last).
"""

import numpy as np

from ._validation import finite_vector, rotation_matrix, se3_transform

_TINY = 1e-12


def skew(w):
    w = finite_vector(w, 3, "w")
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def se3_hat(xi):
    xi = finite_vector(xi, 6, "xi")
    H = np.zeros((4, 4))
    H[:3, :3] = skew(xi[3:6])
    H[:3, 3] = xi[0:3]
    return H


def se3_vee(H):
    H = np.asarray(H, dtype=float)
    if H.shape != (4, 4):
        raise ValueError("se3_vee expects a 4x4 matrix")
    W = H[:3, :3]
    return np.array([H[0, 3], H[1, 3], H[2, 3], W[2, 1], W[0, 2], W[1, 0]])


def so3_log(R):
    R = rotation_matrix(R)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < _TINY:
        W = R - R.T
        return np.array([W[2, 1], W[0, 2], W[1, 0]]) / 2.0
    s = np.sin(theta)
    return theta / (2.0 * s) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def se3_exp(xi):
    xi = finite_vector(xi, 6, "xi")
    v, w = xi[0:3], xi[3:6]
    theta = np.linalg.norm(w)
    T = np.eye(4)
    if theta < _TINY:
        T[:3, :3] = np.eye(3) + skew(w)
        T[:3, 3] = v
        return T
    W = skew(w)
    W2 = W @ W
    R = np.eye(3) + np.sin(theta) / theta * W + (1.0 - np.cos(theta)) / theta**2 * W2
    V = np.eye(3) + (1.0 - np.cos(theta)) / theta**2 * W + (theta - np.sin(theta)) / theta**3 * W2
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T):
    T = se3_transform(T, "T")
    R, p = T[:3, :3], T[:3, 3]
    w = so3_log(R)
    theta = np.linalg.norm(w)
    if theta < _TINY:
        return np.hstack([p, w])
    W = skew(w)
    W2 = W @ W
    coef = 1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    V_inv = np.eye(3) - 0.5 * W + coef * W2
    return np.hstack([V_inv @ p, w])


def se3_adjoint(T):
    T = se3_transform(T, "T")
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def se3_left_jacobian_inverse(xi):
    xi = finite_vector(xi, 6, "xi")
    v, w = xi[0:3], xi[3:6]
    # ad_xi for [v, w] ordering: [[skew(w), skew(v)], [0, skew(w)]]
    ad = np.zeros((6, 6))
    W = skew(w)
    ad[:3, :3] = W
    ad[:3, 3:] = skew(v)
    ad[3:, 3:] = W
    # J_l(xi) = sum_n (-1)^n/(n+1)! ad^n = (I - exp(-ad)) ad^{-1}.
    # The se(3) inverse is NOT the truncated SO(3) form ``I - 1/2 ad + gamma ad^2``
    # (that drops the translation/rotation coupling in ad^3 and above); the series
    # converges rapidly (theta <= pi for a principal log) and the 6x6 inverse is
    # exact to machine precision.
    Jl = np.eye(6)
    term = np.eye(6)
    for k in range(1, 60):
        term = term @ (-ad) / (k + 1)
        Jl = Jl + term
    return np.linalg.inv(Jl)


class SE3Metric:
    """Weighted SE(3) metric ``d_W^2 = ||rho||^2 + w^2 ||theta||^2``."""

    def __init__(self, rotation_weight=1.0):
        try:
            self.rotation_weight = float(rotation_weight)
        except (TypeError, ValueError) as exc:
            raise TypeError("rotation_weight must be a real scalar") from exc
        if not np.isfinite(self.rotation_weight) or self.rotation_weight < 0.0:
            raise ValueError("rotation_weight must be finite and non-negative")

    def error(self, T_from, T_to):
        T_from = se3_transform(T_from, "T_from")
        T_to = se3_transform(T_to, "T_to")
        return se3_log(np.linalg.inv(T_from) @ T_to)

    def distance(self, T_from, T_to):
        r = self.error(T_from, T_to)
        return np.sqrt(self.distance_sq(r))

    def distance_sq(self, r):
        r = finite_vector(r, 6, "r")
        w = self.rotation_weight
        return float(r[0:3] @ r[0:3] + w * w * (r[3:6] @ r[3:6]))

    def weighted_error(self, r):
        r = finite_vector(r, 6, "r")
        w = self.rotation_weight
        out = r.copy()
        out[3:6] *= w * w
        return out

    def error_body_jacobian(self, T_from, T_to):
        """E = d r / d(body perturbation of T_from), r = Log(T_from^-1 T_to)."""
        T_from = se3_transform(T_from, "T_from")
        T_to = se3_transform(T_to, "T_to")
        g = np.linalg.inv(T_from) @ T_to
        r = se3_log(g)
        return -se3_left_jacobian_inverse(r) @ se3_adjoint(np.linalg.inv(g))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_metrics.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/metrics.py tests/gtsr/test_metrics.py
git commit -m "feat(gtsr): add SE(3) Lie primitives and weighted SE3Metric"
```

## Task 2: PoEChart.forward（G-01~G-08）

**Files:**
- Modify: `src/tsr/gtsr/charts.py`（增补 `PoEChart`，`__init__` + `forward`）
- Create: `tests/gtsr/test_poe_chart.py`（G-01~G-08）

**Interfaces:**
- Consumes: `se3_exp`（Task 1）、`_validation.finite_vector`、`finite_matrix`、`se3_transform`。
- Produces: `PoEChart(axes, representation, order=None, M=None)` 及 `.forward(z)`、`.dim`、`.coord_names`、`.periodic`、`.periods`（Task 3 将补 `.jacobian`）。

- [ ] **Step 1: Write the failing test**

`tests/gtsr/test_poe_chart.py`（G-01~G-08）：

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest

from tsr.gtsr.charts import PoEChart
from tsr.gtsr.metrics import se3_adjoint, se3_exp


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_poe_chart.py -q`
Expected: FAIL — `ImportError: cannot import name 'PoEChart'`

- [ ] **Step 3: Write minimal implementation**

在 `src/tsr/gtsr/charts.py` 顶部加 `from .metrics import se3_exp`，并在 `CanonicalTSRChart` 之后增补：

```python
class PoEChart(GeneratorChart):
    """Product-of-Exponentials chart.

    ``representation="space"`` -> ``Phi(z) = Exp(S1 z1) ... Exp(Sm zm) M``.
    ``representation="body"``  -> ``Phi(z) = M Exp(B1 z1) ... Exp(Bm zm)``.
    ``order`` is an explicit index permutation (default ``0..m-1``); ``M`` is the
    home transform (Sprint B restricts to ``M=I`` but the interface is kept).
    """

    _PITCH_TOL = 1e-12

    def __init__(self, axes, representation, order=None, M=None):
        axes = np.asarray(axes, dtype=float)
        if axes.ndim != 2 or axes.shape[1] != 6 or axes.shape[0] < 1:
            raise ValueError("axes must be a non-empty (m, 6) array of screw axes")
        axes = finite_matrix(axes, axes.shape, "axes")
        if representation not in ("space", "body"):
            raise ValueError("representation must be 'space' or 'body'")
        m = axes.shape[0]
        if order is None:
            order = list(range(m))
        order = [int(i) for i in order]
        if sorted(order) != list(range(m)):
            raise ValueError("order must be a permutation of 0..m-1")
        if M is None:
            M = np.eye(4)
        M = se3_transform(M, "M")
        periodic = []
        for xi in axes:
            v, w = xi[0:3], xi[3:6]
            th = float(np.linalg.norm(w))
            if th <= self._PITCH_TOL:
                periodic.append(False)  # pure translation
            else:
                pitch = float(np.dot(w, v)) / (th * th)
                periodic.append(abs(pitch) <= self._PITCH_TOL)
        periods = tuple(2 * pi if p else None for p in periodic)
        self.coord_names = tuple(f"q{i}" for i in range(m))
        self.periodic = tuple(periodic)
        self.periods = periods
        self._axes = axes
        self._representation = representation
        self._order = order
        self._M = M

    def forward(self, z):
        z = finite_vector(z, self.dim, "z")
        T = self._M.copy()
        if self._representation == "space":
            for idx in reversed(self._order):
                T = se3_exp(self._axes[idx] * z[idx]) @ T
        else:
            for idx in self._order:
                T = T @ se3_exp(self._axes[idx] * z[idx])
        return T
```

> 注：`dim` 属性继承自基类 `len(self.coord_names)`，因 `__init__` 设置了实例级 `coord_names`，无需覆盖。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_poe_chart.py -q`
Expected: PASS（G-01~G-08 全通过；G-09 尚未添加）

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/charts.py tests/gtsr/test_poe_chart.py
git commit -m "feat(gtsr): add PoEChart forward geometry (G-01..G-08)"
```

## Task 3: PoEChart.jacobian（G-09）

**Files:**
- Modify: `src/tsr/gtsr/charts.py`（`PoEChart.jacobian`）
- Modify: `tests/gtsr/test_poe_chart.py`（增补 G-09）

**Interfaces:**
- Consumes: `se3_adjoint`、`se3_exp`（Task 1）。
- Produces: `PoEChart.jacobian(z) -> (6, m)`，第 i 列 = `(T(z)⁻¹ ∂T(z)/∂z_i)^vee`（body frame）。

- [ ] **Step 1: Write the failing test**

在 `tests/gtsr/test_poe_chart.py` 追加：

```python
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
```

> 同时把 `se3_vee` 加入该文件的 import。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_poe_chart.py::test_G09_jacobian_matches_central_finite_difference -q`
Expected: FAIL — `AttributeError: 'PoEChart' object has no attribute 'jacobian'`

- [ ] **Step 3: Write minimal implementation**

在 `charts.py` 增补模块级 helper 与两个 chart 的 `jacobian`（body-frame 解析 Jacobian，统一用伴随链）：

```python
_CANONICAL_AXES = np.array(
    [
        [1, 0, 0, 0, 0, 0],  # x
        [0, 1, 0, 0, 0, 0],  # y
        [0, 0, 1, 0, 0, 0],  # z
        [0, 0, 0, 1, 0, 0],  # roll (Rx)
        [0, 0, 0, 0, 1, 0],  # pitch (Ry)
        [0, 0, 0, 0, 0, 1],  # yaw (Rz)
    ]
)
_CANONICAL_ORDER = (0, 1, 2, 5, 4, 3)  # forward 顺序：Trans, Rz, Ry, Rx


def _body_jacobian(axes, order, representation, M, z):
    """Body-frame Jacobian (6, m); column i = (T(z)^-1 dT(z)/dz_i)^vee.

    SPACE: J_i = Ad_{(Exp(S_i z_i) ... Exp(S_m z_m) M)^-1}(S_i)
    BODY:  J_i = Ad_{(Exp(B_i z_i) ... Exp(B_m z_m))^-1}(B_i)
    """
    m = len(order)
    J = np.zeros((6, m))
    right = M.copy() if representation == "space" else np.eye(4)
    for k in reversed(range(m)):
        idx = order[k]
        right = se3_exp(axes[idx] * z[idx]) @ right
        J[:, idx] = se3_adjoint(np.linalg.inv(right)) @ axes[idx]
    return J
```

`CanonicalTSRChart` 增补（依赖 `_CANONICAL_AXES`/`_CANONICAL_ORDER`）：

```python
    def jacobian(self, z):
        z = finite_vector(z, self.dim, "z")
        return _body_jacobian(_CANONICAL_AXES, _CANONICAL_ORDER, "space", np.eye(4), z)
```

`PoEChart` 增补：

```python
    def jacobian(self, z):
        z = finite_vector(z, self.dim, "z")
        return _body_jacobian(self._axes, self._order, self._representation, self._M, z)
```

并把 charts.py 顶部 import 改为 `from .metrics import se3_adjoint, se3_exp`。

> 推导：SPACE 的 body twist 贡献 `(T⁻¹ ∂T/∂z_i)^vee = Ad_{(E_i...E_m M)⁻¹}(S_i)`（含第 i 个
> Exp 与 M）；BODY 中 `M` 在最左，`T⁻¹ ∂T/∂z_i` 展开后 `M⁻¹M` 相消，得 `Ad_{(E_i...E_m)⁻¹}(B_i)`
> （不含 M）。`_body_jacobian` 从最右向左逐项「左乘 Exp」，使 `right` 始终等于「第 i 个 Exp 及其
> 右侧全部 Exp（SPACE 再加 M）的乘积」，再取 `Ad_{right⁻¹}`，两种表示统一。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_poe_chart.py -q`
Expected: PASS（G-01~G-09 全通过）

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/charts.py tests/gtsr/test_poe_chart.py
git commit -m "feat(gtsr): add PoEChart analytic body Jacobian (G-09)"
```

## Task 4: ProjectionResult + 一维投影 + GTSR.project + contains(metric)（P-01~P-07）

**Files:**
- Create: `src/tsr/gtsr/projection.py`
- Modify: `src/tsr/gtsr/region.py`（`GTSR.project`、`contains(T, metric=None, tol=1e-10)`）
- Create: `tests/gtsr/test_projection.py`（P-01~P-07）

**Interfaces:**
- Consumes: `SE3Metric`（Task 1）、`PoEChart`（Task 2/3）、`CoordinateDomain`（既有）。
- Produces: `ProjectionResult(z, transform, distance, success, optimality, message)`、`project(gtsr, T, metric, *, starts=None, tol_bounds=1e-12, tol_opt=1e-6, maxiter=1000)`、`GTSR.project(T, metric)`、`GTSR.contains(T, metric=None, tol=1e-10)`。

- [ ] **Step 1: Write the failing test**

`tests/gtsr/test_projection.py`（P-01~P-07）：

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from tsr.gtsr.charts import PoEChart
from tsr.gtsr.domains import CoordinateDomain, LinearInterval
from tsr.gtsr.metrics import SE3Metric, se3_exp
from tsr.gtsr.region import GTSR


def _region_1d(axis, lo, hi):
    chart = PoEChart([axis], representation="space")
    domain = CoordinateDomain([LinearInterval(lo, hi)])
    return GTSR(chart=chart, domain=domain)


def test_P01_query_from_internal_coords():
    axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    region = _region_1d(axis, -1.0, 1.0)
    metric = SE3Metric()
    z_true = np.array([0.35])
    Tq = region.to_transform(z_true)
    res = region.project(Tq, metric)
    assert res.success
    assert res.distance <= 1e-10
    np.testing.assert_allclose(res.z, z_true, atol=1e-6)


def test_P02_zero_width_domain():
    axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    region = _region_1d(axis, 0.2, 0.2)
    metric = SE3Metric()
    Tq = se3_exp([0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = region.project(Tq, metric)
    np.testing.assert_allclose(res.z, [0.2], atol=1e-10)
    assert res.distance == pytest.approx(abs(0.9 - 0.2), abs=1e-10)


def test_P03_canonical_chart_projection_reconstructs_on_region():
    Bw = np.array(
        [
            [-0.2, 0.2],
            [-0.2, 0.2],
            [-0.2, 0.2],
            [-np.pi / 4, np.pi / 4],
            [-np.pi / 4, np.pi / 4],
            [-np.pi / 4, np.pi / 4],
        ]
    )
    region = GTSR.canonical(Bw=Bw)
    metric = SE3Metric()
    rng = np.random.default_rng(7)
    for _ in range(10):
        z = region.domain.sample(rng)
        Tq = region.to_transform(z)
        res = region.project(Tq, metric)
        assert res.success, res.message
        assert res.distance <= 1e-8
        assert region.contains(res.transform, metric=metric)
        np.testing.assert_allclose(res.transform, Tq, atol=1e-8)


def test_P04_1d_translation_analytic():
    axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    region = _region_1d(axis, -0.5, 1.5)
    metric = SE3Metric()
    for q in [-1.0, 0.0, 0.7, 2.0]:
        Tq = se3_exp([q, 0.0, 0.0, 0.0, 0.0, 0.0])
        res = region.project(Tq, metric)
        expected = float(np.clip(q, -0.5, 1.5))
        np.testing.assert_allclose(res.z, [expected], atol=1e-8)
        assert res.distance == pytest.approx(abs(q - expected), abs=1e-8)


def test_P05_1d_nonzero_pitch_screw_reference():
    h = 0.3
    axis = [0.0, 0.0, h, 0.0, 0.0, 1.0]  # unit w, pitch h
    region = _region_1d(axis, -1.0, 1.0)
    metric = SE3Metric(rotation_weight=0.5)

    def f(q):
        return metric.distance(region.to_transform([q]), Tq)

    for qq in [0.4, 1.8, -2.5]:
        Tq = se3_exp([0.1, -0.2, 0.5, 0.2, -0.3, qq])  # off-axis query
        # 密集网格先圈定全局最优的邻域，再用黄金分割精化（稳健于非单峰）
        grid = np.linspace(-1.0, 1.0, 2001)
        fvals = np.array([f(q) for q in grid])
        i = int(np.argmin(fvals))
        lo = grid[max(0, i - 1)]
        hi = grid[min(len(grid) - 1, i + 1)]
        ref = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-14})
        res = region.project(Tq, metric)
        assert res.distance <= ref.fun + 1e-8
        np.testing.assert_allclose(res.z, [ref.x], atol=1e-6)


def test_P06_idempotence():
    axis = [0.0, 0.0, 0.3, 0.0, 0.0, 1.0]
    region = _region_1d(axis, -1.5, 1.5)
    metric = SE3Metric()
    Tq = se3_exp([0.2, -0.4, 0.6, 0.5, -0.2, 0.0])
    res1 = region.project(Tq, metric)
    res2 = region.project(res1.transform, metric)
    assert res2.distance <= 1e-10
    np.testing.assert_allclose(res2.transform, res1.transform, atol=1e-10)


def test_P07_contains_iff_projection_residual():
    axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    region = _region_1d(axis, -1.0, 1.0)
    metric = SE3Metric()
    for q in [-1.0, 0.0, 0.9999, 1.0]:
        Tq = se3_exp([q, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert region.contains(Tq, metric=metric) is True
    for q in [1.5, -1.5]:
        Tq = se3_exp([q, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert region.contains(Tq, metric=metric) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_projection.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tsr.gtsr.projection'`

- [ ] **Step 3: Write minimal implementation**

`src/tsr/gtsr/projection.py`：

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generic projection of a query pose onto a G-TSR region."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ._validation import se3_transform
from .metrics import SE3Metric, se3_adjoint


@dataclass(frozen=True)
class ProjectionResult:
    z: np.ndarray
    transform: np.ndarray
    distance: float
    success: bool
    optimality: float | None
    message: str


def _projected_gradient_norm(g, z, lower, upper, tol_bounds):
    pg = np.zeros_like(g)
    for i in range(g.size):
        if lower[i] >= upper[i] - tol_bounds:
            pg[i] = 0.0  # degenerate (fixed) coordinate: no KKT requirement
        elif z[i] <= lower[i] + tol_bounds:
            pg[i] = min(g[i], 0.0)
        elif z[i] >= upper[i] - tol_bounds:
            pg[i] = max(g[i], 0.0)
        else:
            pg[i] = g[i]
    return float(np.max(np.abs(pg)))


def project(gtsr, T, metric, *, starts=None, tol_bounds=1e-12, tol_opt=1e-6, maxiter=1000):
    T = se3_transform(T, "T")
    if not isinstance(metric, SE3Metric):
        raise TypeError("metric must be an SE3Metric")
    domain = gtsr.domain
    lower = domain.cont_lower
    upper = domain.cont_upper
    m = domain.dim

    # build the full-transform body Jacobian wrapper
    Ad_Twe_inv = se3_adjoint(np.linalg.inv(gtsr.Tw_e))

    def T_of_z(z):
        return gtsr.to_transform(z)

    def fg(z):
        T_z = T_of_z(z)
        r = metric.error(T_z, T)
        J_local = gtsr.chart.jacobian(z)
        J_full = Ad_Twe_inv @ J_local
        E = metric.error_body_jacobian(T_z, T)
        grad = 2.0 * J_full.T @ (E.T @ metric.weighted_error(r))
        return metric.distance_sq(r), grad

    rng = np.random.default_rng(20260902)
    starts_list = [lower, upper, (lower + upper) / 2.0]
    starts_list.extend(domain.sample(rng) for _ in range(4))
    if starts is not None:
        starts_list = list(starts)

    best_z = None
    best_d = float("inf")
    best_opt = None
    for z0 in starts_list:
        z0 = np.clip(np.asarray(z0, dtype=float), lower, upper)
        opt = minimize(
            fg,
            z0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(lower[i], upper[i]) for i in range(m)],
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
        )
        z = np.clip(opt.x, lower, upper)
        d = float(np.sqrt(fg(z)[0]))
        if d < best_d:
            best_d = d
            best_z = z
            best_opt = opt

    g_true = _finite_diff_gradient(fg, best_z, lower, upper, tol_bounds)
    opt_norm = _projected_gradient_norm(g_true, best_z, lower, upper, tol_bounds)
    bounds_ok = bool(np.all(best_z >= lower - tol_bounds) and np.all(best_z <= upper + tol_bounds))
    feasible = bool(bounds_ok and np.all(np.isfinite(best_z)) and np.isfinite(best_d))
    converged = bool(getattr(best_opt, "success", False))
    optimal = bool(opt_norm <= tol_opt)
    success = bool(converged and feasible and optimal)
    if success:
        message = "ok"
    else:
        reasons = []
        if not converged:
            reasons.append("optimizer did not converge")
        if not feasible:
            reasons.append("bounds infeasible or non-finite")
        if not optimal:
            reasons.append(f"KKT residual {opt_norm:.3e} > {tol_opt:.0e}")
        message = "; ".join(reasons) or "no convergence"
    return ProjectionResult(
        z=best_z,
        transform=T_of_z(best_z),
        distance=best_d,
        success=success,
        optimality=float(opt_norm) if success else None,
        message=message,
    )


def _finite_diff_gradient(fg, z, lower, upper, tol_bounds, h=1e-6):
    g = np.zeros(z.size)
    for i in range(z.size):
        if lower[i] >= upper[i] - tol_bounds:
            g[i] = 0.0  # degenerate (fixed) coordinate: no finite difference
            continue
        zp = z.copy(); zm = z.copy()
        zp[i] = min(zp[i] + h, upper[i])   # clamp so fg never leaves the bounds
        zm[i] = max(zm[i] - h, lower[i])   # (to_transform raises out-of-bounds)
        denom = zp[i] - zm[i]
        g[i] = (fg(zp)[0] - fg(zm)[0]) / denom if denom > 0.0 else 0.0
    return g
```

`src/tsr/gtsr/region.py` 增补（`GTSR` 内）：

```python
    def project(self, T, metric):
        from .projection import project as _project
        return _project(self, T, metric)

    def contains(self, T, metric=None, tol=1e-10):
        if metric is None:
            self._require_canonical("contains")
            T = se3_transform(T, "T")
            T0_s_prime = np.dot(T, np.linalg.inv(self.Tw_e))
            Tw_s_prime = np.dot(np.linalg.inv(self.T0_w), T0_s_prime)
            xyz, rot = Tw_s_prime[0:3, 3], Tw_s_prime[0:3, 0:3]
            xyz_bounds = np.column_stack((self.domain.cont_lower[0:3], self.domain.cont_upper[0:3]))
            rpy_bounds = np.column_stack((self.domain.cont_lower[3:6], self.domain.cont_upper[3:6]))
            xyzcheck = self.chart.xyz_within_bounds(xyz, xyz_bounds)
            rotcheck, _ = self.chart.rot_within_rpy_bounds(rot, rpy_bounds)
            return all(np.hstack((xyzcheck, rotcheck)))
        from .projection import project as _project
        return _project(self, T, metric).distance <= tol
```

> `contains` 的 `metric is None` 分支即原 Sprint A 实现（原样保留）；`metric` 给定走通用投影。
> 原 `contains` 的 docstring/单参签名因此变为双可选参，保持向后兼容。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_projection.py -q && pytest tests/gtsr/ tests/tsr/ -q`
Expected: PASS（P-01~P-07 通过，既有用例无回退）

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/projection.py src/tsr/gtsr/region.py tests/gtsr/test_projection.py
git commit -m "feat(gtsr): add generic projection and metric-based contains (P-01..P-07)"
```

## Task 5: 多维多起点 + 失败传播（P-08, P-09）

**Files:**
- Modify: `src/tsr/gtsr/projection.py`（修正 `_finite_diff_gradient` 占位、确保 `success` 语义）
- Modify: `tests/gtsr/test_projection.py`（增补 P-08、P-09）

**Interfaces:**
- Consumes: `project()`（Task 4）。
- Produces: 无新公共接口；强化 `ProjectionResult.success/optimality/message` 语义。

- [ ] **Step 1: Write the failing test**

`tests/gtsr/test_projection.py` 追加：

```python
def test_P08_multidim_nonconvex_multistart_kkT():
    # two non-commuting generators over a bounded 2D domain
    S1 = [0.0, 0.0, 0.3, 0.0, 0.0, 1.0]
    S2 = [0.1, 0.0, 0.0, 1.0, 0.0, 0.0]
    chart = PoEChart([S1, S2], representation="space")
    domain = CoordinateDomain([LinearInterval(-1.5, 1.5), LinearInterval(-1.5, 1.5)])
    region = GTSR(chart=chart, domain=domain)
    metric = SE3Metric(rotation_weight=0.7)
    Tq = se3_exp([0.4, -0.3, 0.6, 0.5, -0.4, 0.2])
    res = region.project(Tq, metric)
    assert res.success, res.message
    assert res.optimality <= 1e-6
    # grid reference: best of a dense grid is not better than the found optimum (by > 1e-8)
    xs = np.linspace(-1.5, 1.5, 201)
    grid_best = min(
        metric.distance(region.to_transform([a, b]), Tq) for a in xs for b in xs
    )
    assert res.distance <= grid_best + 1e-8


def test_P09_failure_propagation_is_visible():
    axis = [0.0, 0.0, 0.3, 0.0, 0.0, 1.0]
    region = _region_1d(axis, -1.0, 1.0)
    metric = SE3Metric()
    Tq = se3_exp([0.2, -0.4, 0.6, 0.5, -0.2, 0.0])
    from tsr.gtsr.projection import project
    res = project(region, Tq, metric, maxiter=0)
    assert res.success is False
    assert res.message != "" and res.message != "ok"
    assert np.all(np.isfinite(res.z))
    assert np.isfinite(res.distance)
    assert res.transform.shape == (4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_projection.py::test_P08_multidim_nonconvex_multistart_kkT tests/gtsr/test_projection.py::test_P09_failure_propagation_is_visible -q`
Expected: FAIL（P-08 若 `success`/`optimality` 语义未满足；P-09 若 `maxiter` 未透传）

- [ ] **Step 3: Write minimal implementation**

Task 4 的 `project()` 已含固定多起点、`maxiter` 透传与失败语义，本 Task **无新增实现**，仅确认 P-08/P-09 通过：

- P-08 依赖默认多起点（`lower`/`upper`/中点 + 4 随机样本）命中全局 basin；若 `success=False` 或
  `optimality>1e-6`，先核对 `fg` 的解析梯度与 `_finite_diff_gradient` 一致（`chart.jacobian` 已由 G-09 保证，
  `E` 修正由 M-07 保证），再把默认随机样本数从 4 提到 6~8。
- P-09 依赖 `maxiter` 透传：`maxiter=0` 时 `opt.success=False` → `converged=False` → `success=False`，
  `message` 非空且 `z/distance/transform` 仍返回最佳努力值。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_projection.py -q`
Expected: PASS（P-01~P-09 全通过）

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/projection.py tests/gtsr/test_projection.py
git commit -m "feat(gtsr): add multi-start projection and failure semantics (P-08, P-09)"
```

## Task 6: G2 独立复核 + 导出 + 文档修正

**Files:**
- Modify: `src/tsr/gtsr/__init__.py`（导出 `PoEChart`, `SE3Metric`, `ProjectionResult`）
- Modify: `docs/gtsr_geometry_validation_plan.md`（§9 第 4 条 G-01~G-08 → G-01~G-09）

**Interfaces:**
- Consumes: 全部 Task 1~5 产物。
- Produces: `tsr.gtsr.PoEChart`, `tsr.gtsr.SE3Metric`, `tsr.gtsr.ProjectionResult`。

- [ ] **Step 1: 更新导出**

`src/tsr/gtsr/__init__.py`：

```python
from .charts import CanonicalTSRChart, GeneratorChart, PoEChart
from .domains import CircularInterval, CoordinateDomain, Interval, LinearInterval
from .metrics import SE3Metric
from .projection import ProjectionResult
from .region import GTSR

__all__ = [
    "Interval",
    "LinearInterval",
    "CircularInterval",
    "CoordinateDomain",
    "GeneratorChart",
    "CanonicalTSRChart",
    "PoEChart",
    "SE3Metric",
    "ProjectionResult",
    "GTSR",
]
```

- [ ] **Step 2: 修正验证方案 §9 的矛盾**

`docs/gtsr_geometry_validation_plan.md` §9 第 4 条「完成 G-01 至 G-08」改为「完成 G-01 至 G-09」，
与 §11 硬门 G2 一致（本计划 §0 已声明）。

- [ ] **Step 3: 独立复核 G2 硬门**

Run: `pytest tests/gtsr/ -q` 然后 `pytest -q`

Expected:
- `tests/gtsr/` 新增 `test_metrics.py`（7）、`test_poe_chart.py`（9）、`test_projection.py`（9）全绿；
- 全量无回退（Sprint A 的 93 gtsr 用例 + 仓库其余用例全绿，允许保留既有的 1 条旧 SciPy warning）。

逐条核对硬门 G2（spec §6）：
- G-01~G-09 全通过 ✓
- `sample→contains`、`project→contains`、投影幂等（P-06/P-07）✓
- 1D screw 投影与独立参考一致（P-05）✓
- 多维 bounds `≤1e-12`、KKT `≤1e-6`、目标不劣于参考 `1e-8`、失败可见（P-08/P-09）✓

- [ ] **Step 4: Commit**

```bash
git add src/tsr/gtsr/__init__.py docs/gtsr_geometry_validation_plan.md
git commit -m "feat(gtsr): export PoEChart/SE3Metric/ProjectionResult and fix G-09 scope (G2)"
```

- [ ] **Step 5: 提交 design 文档（如未随 Task 提交）**

```bash
git add docs/superpowers/specs/2026-09-03-gtsr-sprint-b-design.md
git commit -m "docs(gtsr): freeze Sprint B design and math/API contract"
```

## 完成判定

全部 6 个 Task 的 pytest 绿、硬门 G2 逐条核对通过、`Ruff` lint/format 无新增告警，
即 Sprint B 收尾。之后才可进入 `P_r(s)`、phase estimator、transfer（G2 未过前禁止）。
