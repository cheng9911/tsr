# G-TSR Sprint A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 G-TSR 几何内核的 canonical 嵌入，使 `GTSR.canonical(T0_w, Tw_e, Bw)` 与传统 `TSR` 在集合/位姿/contains/distance/sampling 上逐位兼容，跑通硬门 G1。

**Architecture:** 三个新模块：`domains.py`（坐标域/区间原语，bounds 单一事实源）、`charts.py`（`GeneratorChart` 协议 + `CanonicalTSRChart`，前向/逆映射与欧拉角数学）、`region.py`（`GTSR` 组合外变换+chart+domain，复刻 legacy 兼容表面）。旧 `TSR` 保持不动作 differential oracle。

**Tech Stack:** Python 3.11（`tsr-dev` conda env），numpy，pytest 函数风格。测试运行器：`/home/rocos/miniconda3/envs/tsr-dev/bin/python`（下文记作 `$PY`）。

**Spec:** `docs/superpowers/specs/2026-09-02-gtsr-sprint-a-design.md`

## Global Constraints

- 坐标数组 `[x,y,z,roll,pitch,yaw]`，长度 6，float64。
- 旋转组合 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`；局部位姿 `Trans(x,y,z) @ Rz @ Ry @ Rx`。
- 外部组合 `T0_w @ T_local @ Tw_e`。
- 周期轴索引 3,4,5（period 2π）；线性轴 0,1,2。
- 位姿误差硬阈值 `max(e_t, e_R) <= 1e-10`（`e_R` 用 §6.2 的 atan2 形式）。
- 随机测试用显式 `numpy.random.Generator`，固定 seeds `[0,1,7,42,20260902]`。
- legacy `contains`/`distance`/`is_valid` 保留 `EPSILON=1e-3`；域 `contains` 用干净闭区间语义。
- 新文件 SPDX 头：`# SPDX-License-Identifier: MIT` + `# Copyright (c) 2025 Siddhartha Srinivasa`。
- 旧 `TSR` 与 `tests/tsr/` 一律不改动。
- ruff line-length 120；`select = ["F","E","W","I"]`。

---

### Task 1: 包骨架 + domains.py + 域测试（D-01~D-07）

**Files:**
- Create: `src/tsr/gtsr/__init__.py`
- Create: `src/tsr/gtsr/domains.py`
- Create: `tests/gtsr/__init__.py`
- Create: `tests/gtsr/conftest.py`
- Create: `tests/gtsr/helpers.py`
- Create: `tests/gtsr/test_domains.py`

**Interfaces:**
- Produces: `LinearInterval(lower, upper)` / `CircularInterval(lower, upper, period=2π)` / `CoordinateDomain(intervals)`，方法 `contains`、`map_from_unit`、`map_from_unit_cube`、`sample`、`cont_lower`、`cont_upper`、`dim`，类方法 `CoordinateDomain.from_bounds(Bw, periodic, periods)`。

- [ ] **Step 1: 写 `src/tsr/gtsr/__init__.py`**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generalized Task Space Region (G-TSR) geometry kernel."""

from .domains import CircularInterval, CoordinateDomain, Interval, LinearInterval

__all__ = ["Interval", "LinearInterval", "CircularInterval", "CoordinateDomain"]
```

（`__init__.py` 在 Task 2/4 会扩充，此处先只导出 domains 符号。）

- [ ] **Step 2: 写 `src/tsr/gtsr/domains.py`**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Coordinate domains for the G-TSR geometry kernel.

A :class:`CoordinateDomain` is the single source of truth for the feasible
coordinate set ``Z`` of a region: an ordered product of per-axis
:class:`Interval` objects. The chart owns the *topology* (names, generators,
order, frame, periodicity metadata); the domain owns the *bounds*.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy import pi

from ..utils import wrap_to_interval


class Interval(ABC):
    """Base class for a single coordinate interval. All intervals are closed."""

    periodic: bool = False
    period = None

    @abstractmethod
    def contains(self, x):
        """Return True iff scalar ``x`` is in the interval."""

    @abstractmethod
    def map_from_unit(self, u):
        """Map ``u`` in ``[0, 1)`` to a coordinate value in the interval."""


class LinearInterval(Interval):
    """A closed real interval ``[lower, upper]`` for a non-periodic coordinate."""

    periodic = False
    period = None

    def __init__(self, lower, upper):
        lower = float(lower)
        upper = float(upper)
        if not (np.isfinite(lower) and np.isfinite(upper)):
            raise ValueError(f"LinearInterval bounds must be finite, got ({lower}, {upper})")
        if lower > upper:
            raise ValueError(f"LinearInterval lower ({lower}) must be <= upper ({upper})")
        self.lower = lower
        self.upper = upper

    def contains(self, x):
        x = float(x)
        return self.lower <= x <= self.upper

    def map_from_unit(self, u):
        return self.lower + (self.upper - self.lower) * float(u)


class CircularInterval(Interval):
    """A circular interval of ``period`` (default 2π), closed.

    The raw ``(lower, upper)`` bounds may describe an inner interval, an outer
    interval (``lower > upper``) or a full circle. All are normalized once at
    construction into a continuous interval ``[lo_cont, hi_cont]``, matching the
    legacy ``TSR._Bw_cont`` semantics.
    """

    periodic = True

    def __init__(self, lower, upper, period=2 * pi):
        self.period = float(period)
        if not np.isfinite(self.period) or self.period <= 0:
            raise ValueError(f"CircularInterval period must be positive finite, got {self.period}")
        self.raw_lower = float(lower)
        self.raw_upper = float(upper)
        if not (np.isfinite(self.raw_lower) and np.isfinite(self.raw_upper)):
            raise ValueError(f"CircularInterval bounds must be finite, got ({lower}, {upper})")
        self.lo_cont = float(wrap_to_interval(np.array([self.raw_lower]))[0])
        interval = self.raw_upper - self.raw_lower
        if interval < 0:
            interval += self.period
        interval = min(interval, self.period)
        self.hi_cont = self.lo_cont + interval

    def contains(self, x):
        x = float(x)
        w = float(wrap_to_interval(np.array([x]), lower=np.array([self.lo_cont]))[0])
        return self.lo_cont <= w <= self.hi_cont

    def map_from_unit(self, u):
        z = self.lo_cont + (self.hi_cont - self.lo_cont) * float(u)
        return float(wrap_to_interval(np.array([z]))[0])


class CoordinateDomain:
    """An ordered product of per-axis :class:`Interval` objects."""

    def __init__(self, intervals):
        self.intervals = tuple(intervals)

    @property
    def dim(self):
        return len(self.intervals)

    def contains(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (self.dim,):
            raise ValueError(f"z must have shape ({self.dim},), got {z.shape}")
        return np.array([iv.contains(z[i]) for i, iv in enumerate(self.intervals)], dtype=bool)

    def map_from_unit_cube(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape != (self.dim,):
            raise ValueError(f"u must have shape ({self.dim},), got {u.shape}")
        return np.array([iv.map_from_unit(u[i]) for i, iv in enumerate(self.intervals)], dtype=float)

    def sample(self, rng):
        return self.map_from_unit_cube(rng.random(self.dim))

    @property
    def cont_lower(self):
        return np.array([iv.lo_cont if iv.periodic else iv.lower for iv in self.intervals])

    @property
    def cont_upper(self):
        return np.array([iv.hi_cont if iv.periodic else iv.upper for iv in self.intervals])

    @classmethod
    def from_bounds(cls, Bw, periodic, periods=None):
        Bw = np.asarray(Bw, dtype=float)
        if Bw.ndim != 2 or Bw.shape[1] != 2:
            raise ValueError(f"Bw must have shape (dim, 2), got {Bw.shape}")
        intervals = []
        for i in range(Bw.shape[0]):
            lo, hi = Bw[i, 0], Bw[i, 1]
            if periodic[i]:
                period = periods[i] if periods is not None else 2 * pi
                intervals.append(CircularInterval(lo, hi, period=period))
            else:
                intervals.append(LinearInterval(lo, hi))
        return cls(intervals)
```

- [ ] **Step 3: 写 `tests/gtsr/__init__.py`（空包标记）**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa
```

- [ ] **Step 4: 写 `tests/gtsr/conftest.py`（fixtures）**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)
```

- [ ] **Step 5: 写 `tests/gtsr/helpers.py`（共享纯函数与常量）**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
from numpy import pi

from tsr.tsr import TSR

SEEDS = [0, 1, 7, 42, 20260902]
DIFF_SAMPLES_PER_SEED = 1000


def rotation_error(R_old, R_new):
    """Stable rotation error near zero (atan2 form), per §6.2."""
    R_delta = R_old.T @ R_new
    S = R_delta - R_delta.T
    vee = np.array([S[2, 1], S[0, 2], S[1, 0]])
    return np.arctan2(0.5 * np.linalg.norm(vee), 0.5 * (np.trace(R_delta) - 1.0))


def transform_error(T_old, T_new):
    e_t = np.linalg.norm(T_old[0:3, 3] - T_new[0:3, 3])
    e_R = rotation_error(T_old[0:3, 0:3], T_new[0:3, 0:3])
    return e_t, e_R


def assert_transforms_close(T_old, T_new, tol=1e-10):
    e_t, e_R = transform_error(T_old, T_new)
    assert max(e_t, e_R) <= tol, f"pose mismatch: e_t={e_t:.3e}, e_R={e_R:.3e}"


def random_rotation(rng):
    """Uniformly random SO(3) via SVD (never an arbitrary 4x4)."""
    A = rng.standard_normal((3, 3))
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def random_se3(rng):
    T = np.eye(4)
    T[0:3, 0:3] = random_rotation(rng)
    T[0:3, 3] = rng.uniform(-2.0, 2.0, size=3)
    return T


def random_bounds(rng):
    """Valid (6,2) bounds: translation lo<=hi; rotation may be inner/outer/full."""
    Bw = np.zeros((6, 2))
    Bw[0:3, 0] = rng.uniform(-1.0, 1.0, size=3)
    Bw[0:3, 1] = Bw[0:3, 0] + rng.uniform(0.0, 0.5, size=3)
    for i in range(3, 6):
        lo = rng.uniform(-pi, pi)
        width = rng.uniform(0.0, 2.0 * pi)
        Bw[i, 0] = lo
        Bw[i, 1] = lo + width
    return Bw
```

- [ ] **Step 6: 写 `tests/gtsr/test_domains.py`**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest
from numpy import pi

from tsr.gtsr.domains import CircularInterval, CoordinateDomain, LinearInterval


# --- D-01 ---------------------------------------------------------------
class TestLinearInterval:
    def test_interior_endpoint_outside(self):
        iv = LinearInterval(-0.1, 0.1)
        assert iv.contains(-0.1) is True
        assert iv.contains(0.0) is True
        assert iv.contains(0.1) is True
        assert iv.contains(0.1000001) is False
        assert iv.contains(-0.1000001) is False

    def test_zero_width(self):
        iv = LinearInterval(1.0, 1.0)
        assert iv.contains(1.0) is True
        assert iv.contains(1.0 + 1e-12) is False

    def test_rejects_inverted(self):
        with pytest.raises(ValueError):
            LinearInterval(1.0, 0.0)

    def test_rejects_nonfinite(self):
        with pytest.raises(ValueError):
            LinearInterval(np.nan, 0.0)
        with pytest.raises(ValueError):
            LinearInterval(0.0, np.inf)


# --- D-02 ---------------------------------------------------------------
def test_circular_interval_inner():
    iv = CircularInterval(-pi / 4, pi / 4)
    assert iv.contains(0.0) is True
    assert iv.contains(pi / 4) is True
    assert iv.contains(-pi / 4) is True
    assert iv.contains(pi / 2) is False
    # ±2kπ equivalence class
    assert iv.contains(pi / 4 + 2 * pi) is True
    assert iv.contains(-pi / 4 - 2 * pi) is True


# --- D-03 ---------------------------------------------------------------
def test_circular_interval_outer():
    iv = CircularInterval(3 * pi / 4, -3 * pi / 4)
    assert iv.contains(pi) is True
    assert iv.contains(-pi) is True
    assert iv.contains(0.0) is False
    assert iv.contains(pi / 2) is False


# --- D-04 ---------------------------------------------------------------
def test_circular_interval_full():
    iv = CircularInterval(-pi, pi)
    for x in [0.0, pi / 2, pi, -pi, 3 * pi / 2, 100.0, -100.0]:
        assert iv.contains(x) is True


# --- D-05 ---------------------------------------------------------------
def test_domain_dimension_error():
    dom = CoordinateDomain([LinearInterval(0, 1), LinearInterval(0, 1)])
    with pytest.raises(ValueError):
        dom.contains(np.array([0.5]))
    with pytest.raises(ValueError):
        dom.map_from_unit_cube(np.array([0.5]))


# --- D-06 ---------------------------------------------------------------
def test_linear_interval_aperiodic():
    # A screw coordinate q is linear: q and q+2π are DISTINCT points.
    iv = LinearInterval(0.0, 1.0)
    assert iv.contains(0.5) is True
    assert iv.contains(0.5 + 2 * pi) is False


# --- D-07 ---------------------------------------------------------------
def test_sampling_reproducible():
    dom = CoordinateDomain([
        LinearInterval(-0.1, 0.1),
        LinearInterval(-0.1, 0.1),
        LinearInterval(-0.1, 0.1),
        CircularInterval(-pi, pi),
        CircularInterval(-pi / 2, pi / 2),
        CircularInterval(3 * pi / 4, -3 * pi / 4),
    ])
    s1 = dom.sample(np.random.default_rng(42))
    s2 = dom.sample(np.random.default_rng(42))
    np.testing.assert_array_equal(s1, s2)


def test_from_bounds_and_contains():
    Bw = np.array([
        [-0.1, 0.1],
        [-0.1, 0.1],
        [0.0, 0.02],
        [-pi / 4, pi / 4],
        [-pi / 4, pi / 4],
        [3 * pi / 4, -3 * pi / 4],
    ])
    periodic = (False, False, False, True, True, True)
    dom = CoordinateDomain.from_bounds(Bw, periodic)
    assert dom.dim == 6
    assert all(dom.contains(np.array([0.0, 0.0, 0.01, 0.0, 0.0, pi])))
    assert not dom.contains(np.array([0.0, 0.0, 0.01, 0.0, 0.0, 0.0]))[5]
```

- [ ] **Step 7: 运行测试确认通过**

Run: `$PY -m pytest tests/gtsr/test_domains.py -v`
Expected: 全部 PASS（`test_domains.py` 不 import helpers，`helpers.py` 目前只依赖 `tsr.tsr.TSR`，无循环依赖）。

- [ ] **Step 8: 提交**

```bash
git add src/tsr/gtsr/__init__.py src/tsr/gtsr/domains.py tests/gtsr/
git commit -m "feat(gtsr): add coordinate domains (Linear/Circular/CoordinateDomain)"
```

---

### Task 2: charts.py 前向 + 旋转数学（rpy_to_rot / xyzrpy_to_trans）

**Files:**
- Create: `src/tsr/gtsr/charts.py`
- Modify: `src/tsr/gtsr/__init__.py`（补导出）
- Test: 在 `tests/gtsr/test_canonical_embedding.py` 里先落 chart 前向差分测试

**Interfaces:**
- Produces: `GeneratorChart`（`coord_names`/`periodic`/`periods`/`dim`/`forward`）、`CanonicalTSRChart`（`forward`、`xyzrpy_to_trans`、`rpy_to_rot`、`rot_to_rpy`、`xyz_within_bounds`、`rpy_within_bounds`、`rot_within_rpy_bounds`、`coordinates`）。

- [ ] **Step 1: 写 `src/tsr/gtsr/charts.py`（本任务先写前向部分，逆映射留 Task 3 补全）**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generator charts for the G-TSR geometry kernel."""

from abc import ABC, abstractmethod

import numpy as np
from numpy import pi

from ..utils import EPSILON, wrap_to_interval


class GeneratorChart(ABC):
    """Base protocol: owns coordinate topology, maps coordinates to local poses."""

    coord_names: tuple = ()
    periodic: tuple = ()
    periods: tuple = ()

    @property
    def dim(self):
        return len(self.coord_names)

    @abstractmethod
    def forward(self, z):
        """Map a coordinate vector ``z`` to a 4x4 local transform."""


class CanonicalTSRChart(GeneratorChart):
    """The canonical chart: ``Trans(x,y,z) Rz(yaw) Ry(pitch) Rx(roll)``."""

    coord_names = ("x", "y", "z", "roll", "pitch", "yaw")
    periodic = (False, False, False, True, True, True)
    periods = (None, None, None, 2 * pi, 2 * pi, 2 * pi)

    def forward(self, z):
        return self.xyzrpy_to_trans(z)

    @staticmethod
    def xyzrpy_to_trans(xyzrpy):
        xyzrpy = np.asarray(xyzrpy, dtype=float)
        trans = np.zeros((4, 4))
        trans[3, 3] = 1.0
        trans[0:3, 3] = xyzrpy[0:3]
        trans[0:3, 0:3] = CanonicalTSRChart.rpy_to_rot(xyzrpy[3:6])
        return trans

    @staticmethod
    def rpy_to_rot(rpy):
        rpy = np.asarray(rpy, dtype=float)
        rot = np.zeros((3, 3))
        r, p, y = rpy[0], rpy[1], rpy[2]
        rot[0, 0] = np.cos(p) * np.cos(y)
        rot[1, 0] = np.cos(p) * np.sin(y)
        rot[2, 0] = -np.sin(p)
        rot[0, 1] = np.sin(r) * np.sin(p) * np.cos(y) - np.cos(r) * np.sin(y)
        rot[1, 1] = np.sin(r) * np.sin(p) * np.sin(y) + np.cos(r) * np.cos(y)
        rot[2, 1] = np.sin(r) * np.cos(p)
        rot[0, 2] = np.cos(r) * np.sin(p) * np.cos(y) + np.sin(r) * np.sin(y)
        rot[1, 2] = np.cos(r) * np.sin(p) * np.sin(y) - np.sin(r) * np.cos(y)
        rot[2, 2] = np.cos(r) * np.cos(p)
        return rot
```

- [ ] **Step 2: 写 `tests/gtsr/test_canonical_embedding.py` 的 chart 前向差分测试**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest
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
```

- [ ] **Step 3: 运行测试确认通过**

Run: `$PY -m pytest tests/gtsr/test_canonical_embedding.py -v`
Expected: 3 passed。

- [ ] **Step 4: 提交**

```bash
git add src/tsr/gtsr/charts.py src/tsr/gtsr/__init__.py tests/gtsr/test_canonical_embedding.py
git commit -m "feat(gtsr): add CanonicalTSRChart forward map (Rz·Ry·Rx)"
```

---

### Task 3: charts.py 逆映射 + contains 数学

**Files:**
- Modify: `src/tsr/gtsr/charts.py`（补 `rot_to_rpy`、`xyz_within_bounds`、`rpy_within_bounds`、`rot_within_rpy_bounds`、`coordinates`）

**Interfaces:**
- Consumes: `CoordinateDomain.cont_lower` / `cont_upper`（Task 1）。
- Produces: `CanonicalTSRChart.coordinates(T_local, domain)`、`rot_to_rpy`、三个 `*_within_bounds`。

- [ ] **Step 1: 在 charts.py 的 `CanonicalTSRChart` 内追加方法（在 `rpy_to_rot` 之后）**

```python
    def coordinates(self, T_local, domain):
        """Inverse map: local transform -> coordinate vector (bounds-consistent rpy)."""
        xyz = T_local[0:3, 3]
        rot = T_local[0:3, 0:3]
        rpy_bounds = np.column_stack((domain.cont_lower[3:6], domain.cont_upper[3:6]))
        rpycheck, rpy = self.rot_within_rpy_bounds(rot, rpy_bounds)
        if not all(rpycheck):
            rpy = self.rot_to_rpy(rot)
        return np.hstack((xyz, rpy))

    @staticmethod
    def rot_to_rpy(rot):
        rot = np.asarray(rot, dtype=float)
        rpy = np.zeros(3)
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            p = -np.arcsin(rot[2, 0])
            rpy[0] = np.arctan2((rot[2, 1] / np.cos(p)), (rot[2, 2] / np.cos(p)))
            rpy[1] = p
            rpy[2] = np.arctan2((rot[1, 0] / np.cos(p)), (rot[0, 0] / np.cos(p)))
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = np.arctan2(rot[0, 1], rot[0, 2])
                rpy[0] = r_offset
                rpy[1] = pi / 2
                rpy[2] = 0.0
            else:
                r_offset = np.arctan2(-rot[0, 1], -rot[0, 2])
                rpy[0] = r_offset
                rpy[1] = -pi / 2
                rpy[2] = 0.0
        return rpy

    @staticmethod
    def xyz_within_bounds(xyz, Bw):
        xyz = np.asarray(xyz, dtype=float)
        Bw = np.asarray(Bw, dtype=float)
        xyzcheck = []
        for i, x in enumerate(xyz):
            x_val = x.item() if hasattr(x, "item") else float(x)
            xyzcheck.append(((x_val + EPSILON) >= Bw[i, 0]) and ((x_val - EPSILON) <= Bw[i, 1]))
        return xyzcheck

    @staticmethod
    def rpy_within_bounds(rpy, Bw):
        rpy = np.asarray(rpy, dtype=float)
        Bw = np.asarray(Bw, dtype=float)
        rpy = wrap_to_interval(rpy, lower=Bw[:3, 0])
        rpycheck = [False] * 3
        for i in range(3):
            if Bw[i, 0] > Bw[i, 1] + EPSILON:
                rpycheck[i] = ((rpy[i] + EPSILON) >= Bw[i, 0]) or ((rpy[i] - EPSILON) <= Bw[i, 1])
            else:
                rpycheck[i] = ((rpy[i] + EPSILON) >= Bw[i, 0]) and ((rpy[i] - EPSILON) <= Bw[i, 1])
        return rpycheck

    @staticmethod
    def rot_within_rpy_bounds(rot, Bw):
        rot = np.asarray(rot, dtype=float)
        Bw = np.asarray(Bw, dtype=float)
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            psol = -np.arcsin(rot[2, 0])
            for p in [psol, (pi - psol)]:
                rpy = np.zeros(3)
                rpy[0] = np.arctan2((rot[2, 1] / np.cos(p)), (rot[2, 2] / np.cos(p)))
                rpy[1] = p
                rpy[2] = np.arctan2((rot[1, 0] / np.cos(p)), (rot[0, 0] / np.cos(p)))
                rpycheck = CanonicalTSRChart.rpy_within_bounds(rpy, Bw)
                if all(rpycheck):
                    return rpycheck, rpy
            return rpycheck, None
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = np.arctan2(rot[0, 1], rot[0, 2])
                rpy_list = [
                    np.array([Bw[2, 0] + r_offset, pi / 2, Bw[2, 0]]),
                    np.array([Bw[2, 1] + r_offset, pi / 2, Bw[2, 1]]),
                    np.array([Bw[0, 0], pi / 2, Bw[0, 0] - r_offset]),
                    np.array([Bw[0, 1], pi / 2, Bw[0, 1] - r_offset]),
                ]
                for rpy in rpy_list:
                    rpycheck = CanonicalTSRChart.rpy_within_bounds(rpy, Bw)
                    if rpycheck[1] is False:
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
            else:
                r_offset = np.arctan2(-rot[0, 1], -rot[0, 2])
                rpy_list = [
                    np.array([-Bw[2, 0] + r_offset, -pi / 2, Bw[2, 0]]),
                    np.array([-Bw[2, 1] + r_offset, -pi / 2, Bw[2, 1]]),
                    np.array([Bw[0, 0], -pi / 2, -Bw[0, 0] + r_offset]),
                    np.array([Bw[0, 1], -pi / 2, -Bw[0, 1] + r_offset]),
                ]
                for rpy in rpy_list:
                    rpycheck = CanonicalTSRChart.rpy_within_bounds(rpy, Bw)
                    if rpycheck[1] is False:
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
        return rpycheck, None
```

- [ ] **Step 2: 在 `test_canonical_embedding.py` 追加逆映射 roundtrip 差分测试**

```python
def test_rot_to_rpy_roundtrip_matches_oracle():
    # 逆映射与旧实现逐位一致（含奇异点附近）
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
```

- [ ] **Step 3: 运行测试确认通过**

Run: `$PY -m pytest tests/gtsr/test_canonical_embedding.py -v`
Expected: 5 passed。

- [ ] **Step 4: 提交**

```bash
git add src/tsr/gtsr/charts.py tests/gtsr/test_canonical_embedding.py
git commit -m "feat(gtsr): add CanonicalTSRChart inverse + within-bounds math"
```

---

### Task 4: region.py GTSR.canonical + to_transform + 嵌入差分测试（E + 5000 样本）

**Files:**
- Create: `src/tsr/gtsr/region.py`
- Modify: `src/tsr/gtsr/__init__.py`（补导出 charts/region）
- Modify: `tests/gtsr/helpers.py`（新增 `from tsr.gtsr.region import GTSR` 与 `make_pair`）
- Modify: `tests/gtsr/test_canonical_embedding.py`（加 E-01~E-10 + 5000 样本差分）

**Interfaces:**
- Consumes: `CanonicalTSRChart`、`CoordinateDomain`（Task 1/3）。
- Produces: `GTSR(T0_w, Tw_e, chart, domain)`、`GTSR.canonical(T0_w, Tw_e, Bw)`、`GTSR.to_transform(z)`（本任务仅前向；contains/distance/sample 留 Task 5）。

- [ ] **Step 1: 写 `src/tsr/gtsr/region.py`（本任务先写构造与前向）**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Region-level composition for the G-TSR geometry kernel."""

from functools import reduce

import numpy as np

from ..utils import wrap_to_interval
from .charts import CanonicalTSRChart
from .domains import CoordinateDomain

NANBW = np.ones(6) * float("nan")


class GTSR:
    """A generalized TSR: ``{T0_w @ chart.forward(z) @ Tw_e | z in domain}``."""

    def __init__(self, T0_w=None, Tw_e=None, chart=None, domain=None):
        if T0_w is None:
            T0_w = np.eye(4)
        if Tw_e is None:
            Tw_e = np.eye(4)
        self.T0_w = np.asarray(T0_w, dtype=float)
        self.Tw_e = np.asarray(Tw_e, dtype=float)
        self.chart = chart
        self.domain = domain

    @classmethod
    def canonical(cls, T0_w=None, Tw_e=None, Bw=None):
        """The §6.2 construction map E(T0_w,Tw_e,Bw) = (T0_w,Tw_e,C_TSR,Z(Bw))."""
        if Bw is None:
            Bw = np.zeros((6, 2))
        chart = CanonicalTSRChart()
        domain = CoordinateDomain.from_bounds(Bw, chart.periodic, chart.periods)
        return cls(T0_w, Tw_e, chart, domain)

    def is_valid(self, z, ignoreNAN=False):
        z = np.asarray(z, dtype=float)
        xyz, rpy = z[0:3], z[3:6]
        xyz_bounds = np.column_stack((self.domain.cont_lower[0:3], self.domain.cont_upper[0:3]))
        rpy_bounds = np.column_stack((self.domain.cont_lower[3:6], self.domain.cont_upper[3:6]))
        xyzcheck = self.chart.xyz_within_bounds(xyz, xyz_bounds)
        rpycheck = self.chart.rpy_within_bounds(rpy, rpy_bounds)
        check = np.hstack((xyzcheck, rpycheck))
        if ignoreNAN:
            check = check | np.isnan(z)
        return check

    def to_transform(self, z):
        z = np.asarray(z, dtype=float)
        if len(z) != 6:
            raise ValueError("z must be of length 6")
        validity = self.is_valid(z)
        if not all(validity):
            violated = [i for i, v in enumerate(validity) if not v]
            raise ValueError(f"z violates bounds at dimensions {violated}: z={z}")
        T_local = self.chart.forward(z)
        return reduce(np.dot, [self.T0_w, T_local, self.Tw_e])
```

- [ ] **Step 2: 更新 `__init__.py` 导出**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generalized Task Space Region (G-TSR) geometry kernel."""

from .charts import CanonicalTSRChart, GeneratorChart
from .domains import CircularInterval, CoordinateDomain, Interval, LinearInterval
from .region import GTSR

__all__ = [
    "Interval",
    "LinearInterval",
    "CircularInterval",
    "CoordinateDomain",
    "GeneratorChart",
    "CanonicalTSRChart",
    "GTSR",
]
```

- [ ] **Step 3: 在 `test_canonical_embedding.py` 追加 E-01~E-10 + 5000 样本差分**

```python
from tsr.gtsr.region import GTSR
from tests.gtsr.helpers import DIFF_SAMPLES_PER_SEED, SEEDS, assert_transforms_close, make_pair, random_bounds, random_se3


def _trans(xyz, rpy):
    return TSR.xyzrpy_to_trans(np.hstack([xyz, rpy]))


def test_E01_fixed_point():
    old, new = make_pair(Bw=np.zeros((6, 2)))
    assert_transforms_close(old.to_transform(np.zeros(6)), new.to_transform(np.zeros(6)))


def test_E02_translation_only():
    Bw = np.zeros((6, 2)); Bw[0:3] = [-0.2, 0.2]
    old, new = make_pair(Bw=Bw)
    z = np.array([0.1, -0.05, 0.15, 0.0, 0.0, 0.0])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E03_rotation_axes():
    Bw = np.zeros((6, 2)); Bw[3] = [-pi / 4, pi / 4]
    old, new = make_pair(Bw=Bw)
    assert_transforms_close(old.to_transform([0, 0, 0, 0.3, 0, 0]), new.to_transform([0, 0, 0, 0.3, 0, 0]))
    Bw[4] = [-pi / 4, pi / 4]
    old, new = make_pair(Bw=Bw)
    assert_transforms_close(old.to_transform([0, 0, 0, 0.3, 0.2, 0]), new.to_transform([0, 0, 0, 0.3, 0.2, 0]))
    Bw[5] = [-pi / 4, pi / 4]
    old, new = make_pair(Bw=Bw)
    assert_transforms_close(old.to_transform([0, 0, 0, 0.3, 0.2, -0.1]), new.to_transform([0, 0, 0, 0.3, 0.2, -0.1]))


def test_E04_all_angles_nonzero():
    Bw = np.zeros((6, 2)); Bw[3:6] = [-pi / 2, pi / 2]
    old, new = make_pair(Bw=Bw)
    z = np.array([0, 0, 0, 0.3, -0.4, 0.5])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E05_nonunit_T0_w():
    Bw = np.zeros((6, 2)); Bw[0:3] = [-0.1, 0.1]; Bw[3:6] = [-pi / 4, pi / 4]
    T0_w = _trans([0.5, -0.3, 1.2], [0.4, -0.5, 1.1])
    old, new = make_pair(T0_w=T0_w, Bw=Bw)
    z = np.array([0.05, -0.02, 0.03, 0.1, -0.1, 0.2])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E06_nonunit_Tw_e():
    Bw = np.zeros((6, 2)); Bw[0:3] = [-0.1, 0.1]; Bw[3:6] = [-pi / 4, pi / 4]
    Tw_e = _trans([0.0, 0.0, 0.15], [0.0, 1.2, -0.7])
    old, new = make_pair(Tw_e=Tw_e, Bw=Bw)
    z = np.array([0.05, -0.02, 0.03, 0.1, -0.1, 0.2])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E07_both_frames_nonunit():
    Bw = np.zeros((6, 2)); Bw[0:3] = [-0.1, 0.1]; Bw[3:6] = [-pi / 4, pi / 4]
    T0_w = _trans([0.5, -0.3, 1.2], [0.4, -0.5, 1.1])
    Tw_e = _trans([0.0, 0.0, 0.15], [0.0, 1.2, -0.7])
    old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
    z = np.array([0.05, -0.02, 0.03, 0.1, -0.1, 0.2])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E08_wrap_and_outer_and_full():
    Bw = np.zeros((6, 2))
    Bw[0:3] = [-0.1, 0.1]
    Bw[3] = [3 * pi / 4, -3 * pi / 4]  # outer
    Bw[4] = [-pi, pi]                  # full
    Bw[5] = [pi / 2 - 1e-6, -pi / 2 + 1e-6]
    old, new = make_pair(Bw=Bw)
    for z in [[0, 0, 0, pi, 0, 0], [0, 0, 0, -pi, 0.3, 0], [0.05, 0, 0, 0.9 * pi, -0.4, 1.5]]:
        assert_transforms_close(old.to_transform(np.array(z)), new.to_transform(np.array(z)))


def test_E09_pitch_singularity():
    Bw = np.zeros((6, 2)); Bw[3:6] = [-pi, pi]
    old, new = make_pair(Bw=Bw)
    for pitch in [pi / 2, -pi / 2, pi / 2 - 1e-6, pi / 2 + 1e-6, -pi / 2 + 1e-6]:
        z = np.array([0, 0, 0, 0.3, pitch, 0.5])
        assert_transforms_close(old.to_transform(z), new.to_transform(z))


def test_E10_degenerate_and_mixed():
    Bw = np.zeros((6, 2))
    Bw[0] = [0.1, 0.1]     # fixed x
    Bw[1:3] = [-0.2, 0.2]  # free y,z
    Bw[3] = [0.5, 0.5]     # fixed roll
    Bw[4:6] = [-pi / 4, pi / 4]
    old, new = make_pair(Bw=Bw)
    z = np.array([0.1, 0.05, -0.05, 0.5, 0.1, -0.1])
    assert_transforms_close(old.to_transform(z), new.to_transform(z))


@pytest.mark.parametrize("seed", SEEDS)
def test_diff_5000_samples(seed):
    rng = np.random.default_rng(seed)
    for _ in range(DIFF_SAMPLES_PER_SEED):
        T0_w = random_se3(rng)
        Tw_e = random_se3(rng)
        Bw = random_bounds(rng)
        old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        z = new.domain.sample(rng)
        assert_transforms_close(old.to_transform(z), new.to_transform(z))
```

- [ ] **Step 4: 在 `helpers.py` 追加 `GTSR` import 与 `make_pair`**

在 `helpers.py` 顶部 `from tsr.tsr import TSR` 之后加 `from tsr.gtsr.region import GTSR`；在文件末尾追加：

```python
def make_pair(T0_w=None, Tw_e=None, Bw=None):
    old = TSR(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
    new = GTSR.canonical(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
    return old, new
```

- [ ] **Step 5: 运行测试确认通过**

Run: `$PY -m pytest tests/gtsr/test_canonical_embedding.py -v`
Expected: E-01~E-10（10）+ 3 chart 前向 + 2 chart 逆映射 + 5 seed 参数化（5）= 20 passed。

- [ ] **Step 6: 提交**

```bash
git add src/tsr/gtsr/region.py src/tsr/gtsr/__init__.py tests/gtsr/helpers.py tests/gtsr/test_canonical_embedding.py
git commit -m "feat(gtsr): add GTSR region + canonical forward embedding (E, 5000-sample diff)"
```

---

### Task 5: region.py 完整兼容表面（contains / to_xyzrpy / distance / sample）

**Files:**
- Modify: `src/tsr/gtsr/region.py`（补 `to_xyzrpy`、`contains`、`_displacement_to_tsr`、`distance`、`sample_xyzrpy`、`sample`）
- Create: `tests/gtsr/test_canonical_compatibility.py`

**Interfaces:**
- Produces: `GTSR.to_xyzrpy(T)`、`contains(T)`、`distance(T, rotation_weight)`、`sample_xyzrpy(partial, rng)`、`sample(partial, rng)`。

- [ ] **Step 1: 在 region.py 的 `GTSR` 类内追加方法（在 `to_transform` 之后）**

```python
    def to_xyzrpy(self, T):
        T_local = reduce(np.dot, [np.linalg.inv(self.T0_w), T, np.linalg.inv(self.Tw_e)])
        return self.chart.coordinates(T_local, self.domain)

    def contains(self, T):
        T0_s_prime = np.dot(T, np.linalg.inv(self.Tw_e))
        Tw_s_prime = np.dot(np.linalg.inv(self.T0_w), T0_s_prime)
        xyz, rot = Tw_s_prime[0:3, 3], Tw_s_prime[0:3, 0:3]
        xyz_bounds = np.column_stack((self.domain.cont_lower[0:3], self.domain.cont_upper[0:3]))
        rpy_bounds = np.column_stack((self.domain.cont_lower[3:6], self.domain.cont_upper[3:6]))
        xyzcheck = self.chart.xyz_within_bounds(xyz, xyz_bounds)
        rotcheck, _ = self.chart.rot_within_rpy_bounds(rot, rpy_bounds)
        return all(np.hstack((xyzcheck, rotcheck)))

    def _displacement_to_tsr(self, T):
        T0_s_prime = np.dot(T, np.linalg.inv(self.Tw_e))
        Tw_s_prime = np.dot(np.linalg.inv(self.T0_w), T0_s_prime)
        xyz = Tw_s_prime[0:3, 3]
        rot = Tw_s_prime[0:3, 0:3]
        rpy = self.chart.rot_to_rpy(rot)

        best_dx = None
        best_dw = None
        best_norm = float("inf")

        rpy_candidates = [rpy]
        r, p, y = rpy
        rpy_candidates.append(np.array([r + pi, pi - p, y + pi]))
        rpy_candidates.append(np.array([r + pi, pi - p, y - pi]))
        rpy_candidates.append(np.array([r - pi, pi - p, y + pi]))
        rpy_candidates.append(np.array([r - pi, pi - p, y - pi]))
        rpy_candidates.append(np.array([r + pi, -pi - p, y + pi]))
        rpy_candidates.append(np.array([r + pi, -pi - p, y - pi]))
        rpy_candidates.append(np.array([r - pi, -pi - p, y + pi]))
        rpy_candidates.append(np.array([r - pi, -pi - p, y - pi]))

        for rpy_cand in rpy_candidates:
            rpy_wrapped = wrap_to_interval(rpy_cand, lower=self.domain.cont_lower[3:6])
            dw_cand = np.hstack((xyz, rpy_wrapped))
            dx = np.zeros(6)
            for i in range(6):
                if dw_cand[i] < self.domain.cont_lower[i]:
                    dx[i] = dw_cand[i] - self.domain.cont_lower[i]
                elif dw_cand[i] > self.domain.cont_upper[i]:
                    dx[i] = dw_cand[i] - self.domain.cont_upper[i]
            norm = np.linalg.norm(dx)
            if norm < best_norm:
                best_norm = norm
                best_dx = dx
                best_dw = dw_cand
        return best_dx, best_dw

    def distance(self, T, rotation_weight=1.0):
        if self.contains(T):
            return 0.0, self.to_xyzrpy(T)
        dx, dw = self._displacement_to_tsr(T)
        dx_weighted = dx.copy()
        dx_weighted[3:6] *= rotation_weight
        dist = np.linalg.norm(dx_weighted)
        bwopt = np.clip(dw, self.domain.cont_lower, self.domain.cont_upper)
        bwopt[3:6] = wrap_to_interval(bwopt[3:6])
        return dist, bwopt

    def sample_xyzrpy(self, xyzrpy=NANBW, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        check = self.is_valid(xyzrpy, ignoreNAN=True)
        if not all(check):
            raise ValueError("xyzrpy must be within bounds", check)
        u = rng.random(6)
        Bw_sample = np.array([
            self.domain.cont_lower[i] + (self.domain.cont_upper[i] - self.domain.cont_lower[i]) * u[i]
            if np.isnan(x)
            else x
            for i, x in enumerate(xyzrpy)
        ])
        Bw_sample[3:6] = wrap_to_interval(Bw_sample[3:6])
        return Bw_sample

    def sample(self, xyzrpy=NANBW, rng=None):
        return self.to_transform(self.sample_xyzrpy(xyzrpy, rng=rng))
```

- [ ] **Step 2: 写 `tests/gtsr/test_canonical_compatibility.py`**

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest
from numpy import pi

from tsr.gtsr.region import GTSR
from tsr.tsr import TSR
from tsr.utils import wrap_to_interval
from tests.gtsr.helpers import assert_transforms_close, make_pair, random_bounds, random_se3


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
    # 内点、端点、容差带内、容差带外、远外
    transforms = []
    for z in [
        [0.0, 0.0, 0.01, 0.0, 0.0, pi],       # interior (outer yaw)
        [0.1, 0.1, 0.02, pi / 4, pi / 4, -3 * pi / 4],  # endpoints
        [0.0, 0.0, 0.01, 0.0, 0.0, 0.0],      # yaw=0 -> outside outer
        [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # far outside
    ]:
        transforms.append(new.to_transform(np.array(z)))
    transforms.append(np.eye(4))               # world origin
    for T in transforms:
        assert old.contains(T) == new.contains(T), f"contains mismatch for T={T}"


# --- C-03 ---------------------------------------------------------------
def test_C03_coordinates_forward_roundtrip(rng):
    for _ in range(50):
        T0_w, Tw_e, Bw = random_se3(rng), random_se3(rng), random_bounds(rng)
        old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        z = new.domain.sample(rng)
        T = new.to_transform(z)
        z_rec = new.to_xyzrpy(T)
        assert_transforms_close(T, new.to_transform(z_rec))
        # 与旧 to_xyzrpy 位姿级一致（不比较欧拉角数组）
        assert_transforms_close(T, old.to_transform(old.to_xyzrpy(T)))


# --- C-04 ---------------------------------------------------------------
def test_C04_legacy_distance(rng):
    for _ in range(50):
        T0_w, Tw_e, Bw = random_se3(rng), random_se3(rng), random_bounds(rng)
        old, new = make_pair(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
        z = new.domain.sample(rng)
        T_in = new.to_transform(z)
        d_old, bw_old = old.distance(T_in)
        d_new, bw_new = new.distance(T_in)
        assert d_old == 0.0 and d_new == 0.0
        assert_transforms_close(old.to_transform(bw_old), new.to_transform(bw_new))
        # 非包含点：距离与 bwopt 重构位姿一致
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

    # 固定分量保留
    partial = np.array([0.05, np.nan, np.nan, np.nan, 0.1, np.nan])
    s_new = new.sample_xyzrpy(partial, rng=np.random.default_rng(0))
    assert s_new[0] == pytest.approx(0.05)
    assert s_new[4] == pytest.approx(0.1)


# --- 域连续边界与旧 _Bw_cont 逐位一致（强不变量） --------------------------
def test_domain_cont_bounds_match_oracle(rng):
    for _ in range(20):
        Bw = random_bounds(rng)
        old = TSR(Bw=Bw)
        new = GTSR.canonical(Bw=Bw)
        np.testing.assert_allclose(new.domain.cont_lower, old._Bw_cont[:, 0], atol=1e-15)
        np.testing.assert_allclose(new.domain.cont_upper, old._Bw_cont[:, 1], atol=1e-15)
```

- [ ] **Step 3: 运行测试确认通过**

Run: `$PY -m pytest tests/gtsr/test_canonical_compatibility.py -v`
Expected: 6 passed。

- [ ] **Step 4: 提交**

```bash
git add src/tsr/gtsr/region.py tests/gtsr/test_canonical_compatibility.py
git commit -m "feat(gtsr): add legacy-compat surface (contains/distance/sample)"
```

---

### Task 6: 硬门 G1 验证 + 全量回归

**Files:**
- 无新代码；可能修改 `docs/superpowers/specs/2026-09-02-gtsr-sprint-a-design.md`（补 convention 表确认）。

- [ ] **Step 1: 运行 gtsr 全量测试**

Run: `$PY -m pytest tests/gtsr/ -v`
Expected: 全 PASS（D 8 + chart 5 + E 20 + C 6 = 39 项，含参数化展开）。

- [ ] **Step 2: 运行旧 TSR 回归确认 oracle 未被破坏**

Run: `$PY -m pytest tests/tsr/ -q`
Expected: `13 passed`（与基线一致）。

- [ ] **Step 3: 全量回归**

Run: `$PY -m pytest -q`
Expected: 全部通过，无回归。

- [ ] **Step 4: 对照硬门 G1 清单逐项确认**

- [ ] E-01~E-10 全通过；
- [ ] 5000 差分样本 `max(e_t,e_R) <= 1e-10`（`test_diff_5000_samples`）；
- [ ] contains 边界语料与旧实现逐例一致（`test_C02_contains_boundary_corpus`）；
- [ ] legacy distance 兼容路径通过（`test_C04_legacy_distance`）；
- [ ] P1 构造映射 `GTSR.canonical` 代码对象逐项对应 `T0↔T0_w, Te↔Tw_e, C_TSR↔CanonicalTSRChart, Z(Bw)↔CoordinateDomain`（设计文档已声明）。

- [ ] **Step 5: 提交（如有文档更新）**

```bash
git add -A && git commit -m "test(gtsr): Sprint A hard gate G1 verification"
```

---

## Self-Review Notes

- **Spec coverage:** D-01~D-07（Task 1）、E-01~E-10 + 5000 样本（Task 4）、C-01~C-05（Task 5）、G1 硬门（Task 6）。P1 构造映射由 `GTSR.canonical` 提供并在设计文档声明对应关系。
- **Type consistency:** `CoordinateDomain.cont_lower/cont_upper/map_from_unit_cube/sample/from_bounds` 在 Task 1 定义，Task 4/5 消费，签名一致；`CanonicalTSRChart.coordinates(T_local, domain)` 在 Task 3 定义，Task 5 消费，签名一致。
- **已知偏差（有意为之）：** 旧 `TSR.to_xyzrpy` 向 `rot_within_rpy_bounds` 误传全量 `_Bw_cont`（而非 `_Bw_cont[3:6]`），新实现 `coordinates` 传正确的 rpy 边界。因两者均返回合法逆映射（位姿级等价），且 C-03 只比较位姿不比较欧拉角数组、C-04 只比较位姿，此修正不影响任何 G1 断言。
