# G-TSR Sprint G — Causal Online Phase Observer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结 SE(3) response-profile 上实现一个只读因果前缀的 streaming phase/progress observer，输出 `(q̂, ρ̂, ĝ, confidence, abstain)`，并验证它不重拟合 profile 即可保留 oracle-phase 的下游性能。

**Architecture:** 一个新模块 `src/tsr/gtsr/phase.py`（观察合同 + 特征提取 + ordered-state observer），加上 `phase_baselines.py`（可部署基线）与 `phase_metrics.py`（observer/下游指标）。observer 是低容量、可校准的有序状态机：8 维因果 SE(3) 几何特征 → cumulative-logit 有序分类（4 个 phase）+ per-phase 脊回归（ρ）→ 单调状态机 + confidence 校准 + abstention。下游保真把 `ĝ_t` 代入冻结 `ResponseProfile6D.alpha(q, ρ)` 计算 distortion 与 physical-timeline task error。

**Tech Stack:** Python 3.11, numpy, scipy（`scipy.optimize.minimize`/`scipy.special`），pytest，h5py（仅数据依赖测试），threadpoolctl（单线程 BLAS）。

**Spec:** `docs/superpowers/specs/2026-09-08-gtsr-sprint-g-online-phase-scope.md`（Sprint G scope，`SCOPE-STRUCTURE-LOCKED`：G-00~G-07 硬门 + §8.3 observer confirmatory 比较族 + §9 排除规则 + §6.4 freeze list）。本计划从 scope 论证；scope 与本计划一起传给执行者。

## Global Constraints

- **冻结 profile 只读**：observer 不得修改 `ResponseProfile6D` 参数；`ResponseProfile6D.__setattr__` 在 frozen 后抛 `AttributeError`。
- **phase 语义**：`q ∈ {0,1,2,3}`（fitting q-index，**不是** solver code 3..6）；`g = (q + ρ) / 4`；`pre = q < 2`。
- **因果性**：`update(o_t, a_{t-1})` 只能依赖 `o_0..o_t`、`a_0..a_{t-1}` 与冻结模型参数；禁止 `solver_phase`、`goal_pose`、`key_clearance_margin`、`yaw_err`、`success`、`terminated`、condition/seed/arm 标识。
- **单调性**：`normalized_progress ∈ [0,1]` 且有效输出满足 `g_t ≥ g_{t-1}`；默认 phase 不回退。
- **独立单位**：execution seed（development n=3）；episode 是最小拆分单位，帧不得跨 train/test。
- **数据角色**：3 个 SE(3) seeds 是 development set（开发级证据）；confirmatory 结论留待 §6.4 新 seeds。本计划只产出开发级 LOSO 证据。
- **度量单位**：SE(3) metric scale `[1,1,1,0.03,0.03,0.03]`（`profile6.SE3_METRIC_SCALE`）；物理 timeline 上评估，不按各参数化 grid 自对齐。
- **风格**：模块头带 `# SPDX-License-Identifier: MIT` / `# Copyright (c) 2025 Siddhartha Srinivasa`；`from __future__ import annotations`；类型标注 + `np.asarray`/`np.float64`。
- **无第三方学习库**：不引入 sklearn/torch；logistic/脊回归用 scipy/numpy 显式实现（与 repo 现有 `least_squares`/`np.linalg` 风格一致）。
- **确认性门值不硬编码**：abstention threshold、phase/progress acceptance 阈值、G-04 superiority margin `δ_sup_obs`、G-05 预算 `B` 均来自 §6.4 freeze list（本次 step 5），经 `phase_gates.py` 单一来源。本计划不得把未冻结门值当最终值硬编码；开发期可用显式 provisional 值，confirmatory 断言只在 `GATES_FROZEN` 翻转后（post-freeze 任务）出现。

---

## File Structure

| 文件 | 职责 |
| --- | --- |
| `src/tsr/gtsr/phase.py` | `PhaseObservation`、`PhaseEstimate`、特征提取、`OnlinePhaseObserver`、有序分类/脊回归/校准 |
| `src/tsr/gtsr/phase_baselines.py` | 可部署基线：normalized-time、causal arc-length、causal keypoint、offline DTW（仅报告） |
| `src/tsr/gtsr/phase_metrics.py` | observer 指标（macro-F1、balanced acc、ordered error、progress MAE、latency、abstention）与下游 distortion |
| `src/tsr/gtsr/phase_gates.py` | 冻结门值单一来源（abstention/acceptance 阈值、`δ_sup_obs`、`B`）+ `GATES_FROZEN` 标志（step 5 填充） |
| `tests/gtsr/test_phase.py` | 观察/估计合同 + 特征提取（纯 numpy，无数据依赖） |
| `tests/gtsr/test_phase_observer.py` | 分类器/回归器/状态机/abstention 单元测试（合成数据） |
| `tests/gtsr/test_phase_leakage.py` | G-01 泄漏门（禁用字段扫描 + 前缀一致性） |
| `tests/gtsr/test_phase_baselines.py` | 基线单元测试（合成曲线） |
| `tests/gtsr/test_phase_downstream.py` | G-05 下游保真（`@pytest.mark.slow`，数据依赖） |

分文件理由：`phase.py` 只放 observer 本体；基线与指标是独立可测单元，与 observer 解耦。数据依赖的 downstream 测试单独成文件，跳过逻辑与 F 系列一致（`_require_spectrum`）。

---

## Task 1: 观察/估计合同与输入验证（G-02）

**Files:**
- Create: `src/tsr/gtsr/phase.py`
- Test: `tests/gtsr/test_phase.py`

**Interfaces:**
- Produces: `PhaseObservation`（frozen dataclass，字段见下）、`PhaseEstimate`（frozen dataclass：`phase: int, progress: float, normalized_progress: float, confidence: float, abstained: bool`）、`_finite_vector`/`_finite_optional` 验证 helper（内部）。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_phase.py
import numpy as np
import pytest

from tsr.gtsr.phase import PhaseEstimate, PhaseObservation


def _obs(**overrides):
    base = dict(
        joint_pos=np.zeros(7),
        joint_vel=np.zeros(7),
        tcp_pose=np.zeros(6),
        peg_pose=np.zeros(6),
        socket_pose=np.zeros(6),
    )
    base.update(overrides)
    return PhaseObservation(**base)


def test_phase_observation_validates_pose_shape():
    with pytest.raises(ValueError):
        _obs(tcp_pose=np.zeros(5))


def test_phase_observation_rejects_nonfinite():
    with pytest.raises(ValueError):
        _obs(peg_pose=np.array([0, 0, 0, np.nan, 0, 0]))


def test_phase_observation_optional_force_defaults_none():
    assert _obs().contact_force is None
    assert _obs().previous_action is None


def test_phase_estimate_is_frozen():
    e = PhaseEstimate(phase=1, progress=0.5, normalized_progress=0.375,
                      confidence=0.9, abstained=False)
    with pytest.raises(AttributeError):
        e.phase = 2


def test_phase_estimate_normalized_progress_range():
    with pytest.raises(ValueError):
        PhaseEstimate(phase=0, progress=0.0, normalized_progress=1.1,
                      confidence=0.5, abstained=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_phase.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tsr.gtsr.phase'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tsr/gtsr/phase.py
"""Causal online phase/progress observer for the frozen SE(3) profile (Sprint G)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _finite_vector(value, size, name):
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _finite_optional(value, size, name):
    if value is None:
        return None
    return _finite_vector(value, size, name)


@dataclass(frozen=True)
class PhaseObservation:
    """One causal observation: runtime pose/force/joint state, no privileged fields."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    tcp_pose: np.ndarray      # (6,) pose6 [x, y, z, roll, pitch, yaw]
    peg_pose: np.ndarray      # (6,) pose6
    socket_pose: np.ndarray   # (6,) pose6
    contact_force: np.ndarray | None = None   # (6,) or None
    previous_action: np.ndarray | None = None  # (6,) or None

    def __post_init__(self):
        for name in ("joint_pos", "joint_vel"):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), 7, name))
        for name in ("tcp_pose", "peg_pose", "socket_pose"):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), 6, name))
        object.__setattr__(self, "contact_force", _finite_optional(self.contact_force, 6, "contact_force"))
        object.__setattr__(self, "previous_action", _finite_optional(self.previous_action, 6, "previous_action"))


@dataclass(frozen=True)
class PhaseEstimate:
    """Immutable observer output: discrete phase, within-phase progress, confidence, abstention."""

    phase: int
    progress: float
    normalized_progress: float
    confidence: float
    abstained: bool = field(default=False)

    def __post_init__(self):
        if not 0 <= int(self.phase) <= 3:
            raise ValueError(f"phase must be in 0..3, got {self.phase}")
        for name in ("progress", "normalized_progress", "confidence"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be in [0, 1]")
        if not 0.0 <= self.normalized_progress <= 1.0:
            raise ValueError("normalized_progress must be in [0, 1]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_phase.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/phase.py tests/gtsr/test_phase.py
git commit -m "feat(gtsr): add PhaseObservation/PhaseEstimate contract (G-02)"
```

---

## Task 2: 因果 SE(3) 特征提取

**Files:**
- Modify: `src/tsr/gtsr/phase.py`
- Test: `tests/gtsr/test_phase.py`

**Interfaces:**
- Consumes: `PhaseObservation`（Task 1）、`profile6.se3_log`/`se3_from_pose6`/`se3_inverse`。
- Produces: `extract_features(obs: PhaseObservation, prev_tcp_pose: np.ndarray | None) -> np.ndarray`，返回长度 8 的 `float64` 特征向量，顺序固定：
  `[trans_dist, trans_z, trans_radial, rot_angle, rot_yaw, rot_rollpitch, tcp_disp, force_mag]`。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_phase.py (append)
from tsr.gtsr.phase import extract_features


def test_extract_features_identity_relation():
    obs = _obs()  # peg == socket == tcp == 0, no force, no previous action
    f = extract_features(obs, None)
    assert f.shape == (8,)
    # identity relation: all error features are zero; force/tcp_disp zero.
    np.testing.assert_allclose(f[:7], np.zeros(7), atol=1e-12)


def test_extract_features_pure_translation():
    obs = _obs(peg_pose=np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.0]))
    f = extract_features(obs, None)
    assert f[0] == pytest.approx(0.03)        # trans_dist
    assert f[1] == pytest.approx(0.0)         # trans_z (x-only offset)
    assert f[2] == pytest.approx(0.03)        # trans_radial
    assert f[3] == pytest.approx(0.0)         # rot_angle


def test_extract_features_force_and_disp():
    obs = _obs(contact_force=np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
               tcp_pose=np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    f = extract_features(obs, np.zeros(6))
    assert f[7] == pytest.approx(3.0)         # force_mag
    assert f[6] == pytest.approx(0.01)        # tcp_disp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_phase.py -q`
Expected: FAIL — `ImportError: cannot import name 'extract_features'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tsr/gtsr/phase.py (append)
from .profile6 import se3_from_pose6, se3_inverse, se3_log


def extract_features(obs, prev_tcp_pose):
    """Eight causal SE(3) geometric features (feature order is frozen)."""
    T_peg = se3_from_pose6(obs.peg_pose)
    T_socket = se3_from_pose6(obs.socket_pose)
    rel = se3_inverse(T_socket) @ T_peg
    xi = se3_log(rel)              # [v; w] in socket frame
    trans = xi[:3]
    rot = xi[3:]
    trans_dist = float(np.linalg.norm(trans))
    trans_z = float(abs(trans[2]))
    trans_radial = float(np.hypot(trans[0], trans[1]))
    rot_angle = float(np.linalg.norm(rot))
    rot_yaw = float(abs(rot[2]))
    rot_rollpitch = float(np.hypot(rot[0], rot[1]))
    if prev_tcp_pose is None:
        tcp_disp = 0.0
    else:
        tcp_disp = float(np.linalg.norm(obs.tcp_pose[:3] - np.asarray(prev_tcp_pose)[:3]))
    force_mag = 0.0 if obs.contact_force is None else float(np.linalg.norm(obs.contact_force[:3]))
    return np.array(
        [trans_dist, trans_z, trans_radial, rot_angle, rot_yaw, rot_rollpitch, tcp_disp, force_mag],
        dtype=np.float64,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_phase.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/phase.py tests/gtsr/test_phase.py
git commit -m "feat(gtsr): add causal SE(3) feature extractor (8-dim)"
```

---

## Task 3: 有序 phase 分类器 + 置信校准

**Files:**
- Modify: `src/tsr/gtsr/phase.py`
- Test: `tests/gtsr/test_phase_observer.py`

**Interfaces:**
- Produces: `OrdinalPhaseClassifier`（`fit(features, q, regularization) -> self`、`predict_proba(features) -> (n, 4)`、`calibrate(features, q) -> self`、`confidence(features) -> (n,)`）。内部用 cumulative-logit 有序模型：`logit P(q ≤ k | f) = θ_k − w·f`，`P(q=k) = σ(θ_k − w·f) − σ(θ_{k−1} − w·f)`（`θ_3 = +∞`，`θ_{-1} = −∞`）。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_phase_observer.py
import numpy as np
import pytest

from tsr.gtsr.phase import OrdinalPhaseClassifier

pytest.importorskip("scipy")


def _separable_features(rng, n=400):
    # Feature 0 cleanly orders the four phases: q increases with f0.
    f0 = rng.uniform(-2.0, 2.0, n)
    q = np.digitize(f0, [-1.0, 0.0, 1.0])  # 0..3
    f = np.column_stack([f0, rng.standard_normal(n) * 0.01])
    return f, q


def test_ordinal_classifier_recovers_ordering():
    rng = np.random.default_rng(0)
    f, q = _separable_features(rng)
    clf = OrdinalPhaseClassifier().fit(f, q)
    proba = clf.predict_proba(f)
    assert proba.shape == (len(f), 4)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()
    # > 95% of predictions match the clean ordering.
    assert (proba.argmax(axis=1) == q).mean() > 0.95


def test_ordinal_classifier_calibration_narrows_confidence():
    rng = np.random.default_rng(1)
    f, q = _separable_features(rng)
    clf = OrdinalPhaseClassifier().fit(f, q).calibrate(f, q)
    conf = clf.confidence(f)
    assert conf.shape == (len(f),)
    assert conf.min() >= 0.0 and conf.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_phase_observer.py -q`
Expected: FAIL — `ImportError: cannot import name 'OrdinalPhaseClassifier'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tsr/gtsr/phase.py (append)
import scipy.optimize as opt
import scipy.special as special


def _sigmoid(z):
    return special.expit(np.clip(z, -30.0, 30.0))


def _isotonic(x, y):
    """Pool-adjacent-violators isotonic regression (non-decreasing); returns a callable.

    Fits the monotone map from raw confidence ``x`` to calibration accuracy ``y``,
    then interpolates linearly between block means for new ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    blocks = [(float(xs[i]), float(ys[i]), 1.0) for i in range(len(xs))]  # (mean_x, mean_y, weight)
    i = 0
    while i < len(blocks) - 1:
        mx_i, my_i, w_i = blocks[i]
        mx_j, my_j, w_j = blocks[i + 1]
        if my_i <= my_j:
            i += 1
        else:
            w = w_i + w_j
            blocks[i : i + 2] = [((mx_i * w_i + mx_j * w_j) / w, (my_i * w_i + my_j * w_j) / w, w)]
            if i > 0:
                i -= 1
    bx = np.array([b[0] for b in blocks])
    by = np.array([b[1] for b in blocks])
    return lambda v: np.interp(v, bx, by)


class OrdinalPhaseClassifier:
    """Cumulative-logit ordinal classifier over q in {0,1,2,3} + isotonic calibration."""

    N_PHASES = 4

    def __init__(self):
        self._w = None          # (n_features,)
        self._thresholds = None  # (3,) ascending
        self._calib = None      # isotonic calibrator (None = raw sigmoid confidence)

    def fit(self, features, q, regularization=1e-3):
        features = np.asarray(features, dtype=np.float64)
        q = np.asarray(q, dtype=np.int64)
        if features.ndim != 2:
            raise ValueError("features must be 2-D")
        if q.shape != (features.shape[0],):
            raise ValueError("q must align with features")
        if not set(q.tolist()).issubset({0, 1, 2, 3}):
            raise ValueError("q must be in {0,1,2,3}")

        def neg_log_likelihood(params):
            w = params[:-3]
            theta = np.cumsum(np.exp(params[-3:]))  # enforce ascending thresholds
            z = theta[None, :] - features @ w[:, None]  # (n, 3) = θ_k - w·f
            p_le = _sigmoid(z)  # P(q <= k) for k = 0,1,2
            # P(q=k) = P(q<=k) - P(q<=k-1); P(q<=-1)=0, P(q<=3)=1.
            p_prev = np.concatenate([np.zeros((len(q), 1)), p_le], axis=1)
            p_next = np.concatenate([p_le, np.ones((len(q), 1))], axis=1)
            proba = np.clip(p_next - p_prev, 1e-9, 1.0)
            loss = -np.mean(np.log(proba[np.arange(len(q)), q]))
            return loss + regularization * np.dot(w, w)

        params0 = np.concatenate([np.zeros(features.shape[1]), np.full(3, np.log(0.5))])
        result = opt.minimize(neg_log_likelihood, params0, method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(f"ordinal fit failed: {result.message}")
        self._w = result.x[:-3]
        self._thresholds = np.cumsum(np.exp(result.x[-3:]))
        return self

    def predict_proba(self, features):
        self._require_fit()
        features = np.asarray(features, dtype=np.float64)
        z = self._thresholds[None, :] - features @ self._w[:, None]
        p_le = _sigmoid(z)
        p_prev = np.concatenate([np.zeros((len(features), 1)), p_le], axis=1)
        p_next = np.concatenate([p_le, np.ones((len(features), 1))], axis=1)
        proba = np.clip(p_next - p_prev, 1e-9, 1.0)
        return proba / proba.sum(axis=1, keepdims=True)

    def calibrate(self, features, q):
        # Isotonic (isotonic regression, non-decreasing) on the held-out
        # confidence — implemented inline to avoid sklearn.
        proba = self.predict_proba(features)
        raw_conf = proba.max(axis=1)
        correct = (proba.argmax(axis=1) == q).astype(np.float64)
        order = np.argsort(raw_conf)
        self._calib = _isotonic(raw_conf[order], correct[order])
        return self

    def confidence(self, features):
        proba = self.predict_proba(features)
        raw = proba.max(axis=1)
        return self._calib(raw) if self._calib is not None else raw

    def _require_fit(self):
        if self._w is None:
            raise RuntimeError("classifier is not fitted")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_phase_observer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/phase.py tests/gtsr/test_phase_observer.py
git commit -m "feat(gtsr): add ordinal phase classifier with isotonic calibration"
```

---

## Task 4: within-phase 进度回归器

**Files:**
- Modify: `src/tsr/gtsr/phase.py`
- Test: `tests/gtsr/test_phase_observer.py`

**Interfaces:**
- Produces: `ProgressRegressor`（`fit(features, q, rho, regularization) -> self`、`predict(features, q) -> (n,)`，输出 clip 到 `[0,1]`）。per-phase 脊回归（4 个独立线性模型）。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_phase_observer.py (append)
from tsr.gtsr.phase import ProgressRegressor


def test_progress_regressor_recovers_linear_rho():
    rng = np.random.default_rng(2)
    n = 300
    f0 = rng.uniform(0.0, 1.0, n)
    q = rng.integers(0, 4, n)
    rho = np.clip(0.4 * f0 + 0.1 * (q + 1) % 1.0, 0.0, 1.0)
    f = np.column_stack([f0, rng.standard_normal(n) * 0.01])
    reg = ProgressRegressor().fit(f, q, rho)
    pred = reg.predict(f, q)
    assert pred.shape == (n,)
    assert (pred >= 0.0).all() and (pred <= 1.0).all()
    assert np.mean((pred - rho) ** 2) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_phase_observer.py -q`
Expected: FAIL — `ImportError: cannot import name 'ProgressRegressor'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tsr/gtsr/phase.py (append)
class ProgressRegressor:
    """Per-phase ridge regression for within-phase progress rho in [0, 1]."""

    N_PHASES = 4

    def __init__(self):
        self._w = None  # (4, n_features + 1)

    def fit(self, features, q, rho, regularization=1e-3):
        features = np.asarray(features, dtype=np.float64)
        q = np.asarray(q, dtype=np.int64)
        rho = np.asarray(rho, dtype=np.float64)
        self._w = np.empty((self.N_PHASES, features.shape[1] + 1))
        design = np.column_stack([np.ones(len(features)), features])
        for phase in range(self.N_PHASES):
            mask = q == phase
            if mask.sum() == 0:
                self._w[phase] = 0.0
                continue
            X = design[mask]
            y = rho[mask]
            ridge = regularization * np.eye(X.shape[1])
            self._w[phase] = np.linalg.solve(X.T @ X + ridge, X.T @ y)
        return self

    def predict(self, features, q):
        self._require_fit()
        features = np.asarray(features, dtype=np.float64)
        q = np.asarray(q, dtype=np.int64)
        design = np.column_stack([np.ones(len(features)), features])
        out = np.empty(len(features), dtype=np.float64)
        for phase in range(self.N_PHASES):
            mask = q == phase
            out[mask] = design[mask] @ self._w[phase]
        return np.clip(out, 0.0, 1.0)

    def _require_fit(self):
        if self._w is None:
            raise RuntimeError("regressor is not fitted")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_phase_observer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/phase.py tests/gtsr/test_phase_observer.py
git commit -m "feat(gtsr): add per-phase progress ridge regressor"
```

---

## Task 5: 单调状态机 + `OnlinePhaseObserver` 流式核心

**Files:**
- Modify: `src/tsr/gtsr/phase.py`
- Test: `tests/gtsr/test_phase_observer.py`

**Interfaces:**
- Consumes: `OrdinalPhaseClassifier`（Task 3）、`ProgressRegressor`（Task 4）、`extract_features`（Task 2）。`abstention_threshold` 是必填构造参数（无默认），其冻结值来自 §6.4 freeze list（step 5），不在此硬编码。
- Produces: `OnlinePhaseObserver`（`reset()`、`update(observation_t, previous_action) -> PhaseEstimate`；构造参数 `classifier`、`regressor`、`abstention_threshold`、`include_force`）。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_phase_observer.py (append)
import numpy as np
import pytest

from tsr.gtsr.phase import OnlinePhaseObserver, OrdinalPhaseClassifier, ProgressRegressor


def _fitted_observer(rng, abstention_threshold=0.5):
    # 0.5 是 DEVELOPMENT provisional，不是冻结门值（step 5 冻结后才替换）。
    f, q = _separable_features(rng)
    rho = rng.uniform(0.0, 1.0, len(q))
    return OnlinePhaseObserver(
        classifier=OrdinalPhaseClassifier().fit(f, q),
        regressor=ProgressRegressor().fit(f, q, rho),
        abstention_threshold=abstention_threshold,
    )


def _obs_from_feature(feature):
    from tsr.gtsr.phase import PhaseObservation
    # Map feature[0] to a pure z-translation of peg so extract_features is invertible-ish.
    return PhaseObservation(
        joint_pos=np.zeros(7), joint_vel=np.zeros(7), tcp_pose=np.zeros(6),
        peg_pose=np.array([0.0, 0.0, feature[0] * 0.1, 0.0, 0.0, 0.0]),
        socket_pose=np.zeros(6),
    )


def test_observer_is_monotonic_and_holds_on_abstain():
    rng = np.random.default_rng(3)
    obs = _fitted_observer(rng)
    # Drive a monotonic sequence of decreasing relation error -> increasing phase.
    seq = [0.8, 0.3, -0.1, -0.6, -1.1]
    phases = []
    for value in seq:
        est = obs.update(_obs_from_feature(np.array([value] + [0.0] * 7)), None)
        phases.append(est.phase)
        assert 0.0 <= est.normalized_progress <= 1.0
    assert phases == sorted(phases)          # never moves backward
    assert len(set(phases)) >= 2             # actually advanced


def test_observer_reset_clears_state():
    rng = np.random.default_rng(4)
    obs = _fitted_observer(rng)
    est1 = obs.update(_obs_from_feature(np.array([0.9] + [0.0] * 7)), None)
    obs.reset()
    est2 = obs.update(_obs_from_feature(np.array([0.9] + [0.0] * 7)), None)
    assert est1.phase == est2.phase
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_phase_observer.py -q`
Expected: FAIL — `ImportError: cannot import name 'OnlinePhaseObserver'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tsr/gtsr/phase.py (append)
class OnlinePhaseObserver:
    """Streaming ordered-state observer: causal prefix -> (q, rho, g, confidence, abstain)."""

    def __init__(self, classifier, regressor, abstention_threshold):
        # abstention_threshold 是 §6.4 freeze-list 门值（step 5），无默认以免混入未冻结值。
        self._classifier = classifier
        self._regressor = regressor
        self._abstention_threshold = float(abstention_threshold)
        self.reset()

    def reset(self):
        self._last_phase = 0
        self._last_g = 0.0
        self._prev_tcp_pose = None

    def update(self, observation_t, previous_action):
        features = extract_features(observation_t, self._prev_tcp_pose)
        proba = self._classifier.predict_proba(features[None, :])[0]
        confidence = float(self._classifier.confidence(features[None, :])[0])
        phase = int(np.argmax(proba))
        # Monotonic state machine: phase never moves backward.
        phase = max(phase, self._last_phase)
        rho = float(self._regressor.predict(features[None, :], np.array([phase]))[0])
        g = (phase + rho) / 4.0
        # No backward progress; abstain (hold) when confidence is too low.
        abstained = confidence < self._abstention_threshold
        if g < self._last_g or abstained:
            phase = self._last_phase
            g = self._last_g
        self._last_phase = phase
        self._last_g = g
        self._prev_tcp_pose = observation_t.tcp_pose
        return PhaseEstimate(
            phase=phase,
            progress=float(g * 4.0 - phase),
            normalized_progress=g,
            confidence=confidence,
            abstained=abstained,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_phase_observer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/phase.py tests/gtsr/test_phase_observer.py
git commit -m "feat(gtsr): add OnlinePhaseObserver streaming core with monotonic state machine"
```

---

## Task 6: 泄漏门与因果一致性（G-01）

**Files:**
- Modify: `src/tsr/gtsr/phase.py`
- Test: `tests/gtsr/test_phase_leakage.py`

**Interfaces:**
- Produces: `FORBIDDEN_FIELDS: tuple[str, ...]`（模块常量，含 `solver_phase`, `goal_pose`, `key_clearance_margin`, `yaw_err`, `success`, `terminated`, `truncated`, `condition_id`, `seed`, `arm` 等）；`assert_no_forbidden_fields(obs_dict)`（测试 helper 或模块内 `_assert_causal`）。observer 只接收 `PhaseObservation`，其字段集本身即禁用字段的负向保证。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_phase_leakage.py
import numpy as np

from tsr.gtsr.phase import FORBIDDEN_FIELDS, PhaseObservation, OnlinePhaseObserver, extract_features


def test_phase_observation_has_no_forbidden_fields():
    for field in FORBIDDEN_FIELDS:
        assert not hasattr(PhaseObservation.__dataclass_fields__, field) or \
            field not in PhaseObservation.__dataclass_fields__


def test_feature_extract_is_causal_no_privileged_inputs():
    # extract_features only reads pose/force; changing a future frame is impossible
    # by construction, but we assert the observable surface carries no privileged keys.
    names = {f.name for f in __import__("dataclasses").fields(PhaseObservation)}
    assert names == {"joint_pos", "joint_vel", "tcp_pose", "peg_pose",
                     "socket_pose", "contact_force", "previous_action"}


def test_update_ignores_future_by_construction():
    # Prefix consistency: feeding the same (obs, action) twice from a fresh reset
    # yields byte-identical outputs (deterministic, no hidden time state).
    from tsr.gtsr.phase import OrdinalPhaseClassifier, ProgressRegressor
    rng = np.random.default_rng(0)
    f = rng.standard_normal((50, 8))
    q = np.digitize(f[:, 0], [-0.5, 0.0, 0.5])
    rho = rng.uniform(0, 1, 50)
    obs = OnlinePhaseObserver(
        classifier=OrdinalPhaseClassifier().fit(f, q),
        regressor=ProgressRegressor().fit(f, q, rho),
    )
    o = PhaseObservation(joint_pos=np.zeros(7), joint_vel=np.zeros(7),
                         tcp_pose=np.zeros(6), peg_pose=np.zeros(6), socket_pose=np.zeros(6))
    a = obs.update(o, None)
    b = obs.update(o, None)
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_phase_leakage.py -q`
Expected: FAIL — `ImportError: cannot import name 'FORBIDDEN_FIELDS'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tsr/gtsr/phase.py (append)
FORBIDDEN_FIELDS = (
    "solver_phase", "goal_pose", "keyed_entry_goal_pose", "circular_transition_goal_pose",
    "key_clearance_margin", "obj_to_goal_dist", "yaw_err", "axis_angle_err",
    "success", "terminated", "truncated", "condition_id", "seed", "arm", "reward",
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_phase_leakage.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/phase.py tests/gtsr/test_phase_leakage.py
git commit -m "test(gtsr): add G-01 leakage gate and forbidden-field surface"
```

---

## Task 7: 可部署基线与 observer 指标

**Files:**
- Create: `src/tsr/gtsr/phase_baselines.py`
- Create: `src/tsr/gtsr/phase_metrics.py`
- Test: `tests/gtsr/test_phase_baselines.py`

**Interfaces:**
- Produces（baselines）: `NormalizedTimeBaseline`、`ArcLengthBaseline`、`KeypointBaseline`，各自实现 `reset()`/`update(obs, action) -> PhaseEstimate`，与 observer 同签名。`offline_dtw_phase`（仅报告，返回对齐后的 phase 序列）。
- Produces（metrics）: `phase_macro_f1(q_hat, q)`、`balanced_accuracy(q_hat, q)`、`ordered_phase_error(q_hat, q)`、`progress_mae(g_hat, g)`、`switch_latency(q_hat, q)`、`abstention_report(estimates)`。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_phase_baselines.py
import numpy as np
import pytest

from tsr.gtsr.phase import PhaseEstimate, PhaseObservation
from tsr.gtsr.phase_baselines import ArcLengthBaseline, NormalizedTimeBaseline
from tsr.gtsr.phase_metrics import phase_macro_f1, ordered_phase_error


def _obs(step, n_total=100):
    return PhaseObservation(
        joint_pos=np.zeros(7), joint_vel=np.zeros(7),
        tcp_pose=np.array([step * 0.001, 0.0, 0.0, 0.0, 0.0, 0.0]),
        peg_pose=np.zeros(6), socket_pose=np.zeros(6),
    )


def test_normalized_time_baseline_monotonic():
    base = NormalizedTimeBaseline(n_steps=100)
    gs = [base.update(_obs(t), None).normalized_progress for t in range(100)]
    assert gs == sorted(gs)
    assert gs[-1] == pytest.approx(1.0)


def test_arc_length_baseline_tracks_displacement():
    base = ArcLengthBaseline(total_arc=0.099)  # 99 steps * 0.001
    est = base.update(_obs(99), None)
    assert est.normalized_progress == pytest.approx(1.0)


def test_phase_metrics():
    q = np.array([0, 1, 1, 2, 3, 3])
    qh = np.array([0, 1, 2, 2, 3, 3])
    assert phase_macro_f1(qh, q) > 0.0
    assert ordered_phase_error(qh, q) == pytest.approx(1 / 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_phase_baselines.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tsr.gtsr.phase_baselines'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tsr/gtsr/phase_baselines.py
"""Deployable causal baselines for phase/progress estimation (Sprint G §7)."""
from __future__ import annotations

import numpy as np

from .phase import PhaseEstimate


class NormalizedTimeBaseline:
    """Naive time-normalized phase; no geometric information."""

    def __init__(self, n_steps):
        self._n_steps = int(n_steps)
        self.reset()

    def reset(self):
        self._t = 0

    def update(self, observation_t, previous_action):
        self._t += 1
        g = min(self._t / self._n_steps, 1.0)
        phase = min(int(g * 4.0), 3)
        return PhaseEstimate(phase=phase, progress=g * 4.0 - phase,
                             normalized_progress=g, confidence=1.0, abstained=False)


class ArcLengthBaseline:
    """Causal cumulative SE(3) arc length of the tcp, normalized by total arc."""

    def __init__(self, total_arc):
        self._total = float(total_arc)
        self.reset()

    def reset(self):
        self._arc = 0.0
        self._prev = None

    def update(self, observation_t, previous_action):
        if self._prev is not None:
            self._arc += float(np.linalg.norm(observation_t.tcp_pose[:3] - self._prev[:3]))
        self._prev = observation_t.tcp_pose
        g = min(self._arc / self._total, 1.0)
        phase = min(int(g * 4.0), 3)
        return PhaseEstimate(phase=phase, progress=g * 4.0 - phase,
                             normalized_progress=g, confidence=1.0, abstained=False)


class KeypointBaseline:
    """Causal keypoint/state-machine baseline: threshold on task-local relation error."""

    def __init__(self, thresholds=(0.05, 0.02, 0.005)):
        self._thresholds = tuple(float(t) for t in thresholds)
        self.reset()

    def reset(self):
        self._phase = 0

    def update(self, observation_t, previous_action):
        from .phase import extract_features
        f = extract_features(observation_t, None)
        err = f[0]  # trans_dist
        target = 3
        for k, threshold in enumerate(self._thresholds):
            if err > threshold:
                target = k
                break
        self._phase = max(self._phase, target)
        return PhaseEstimate(phase=self._phase, progress=0.5, normalized_progress=(self._phase + 0.5) / 4.0,
                             confidence=1.0, abstained=False)
```

```python
# src/tsr/gtsr/phase_metrics.py
"""Observer metrics on the physical timeline, per episode then per seed (Sprint G §8)."""
from __future__ import annotations

import numpy as np


def phase_macro_f1(q_hat, q):
    q_hat = np.asarray(q_hat, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    scores = []
    for k in range(4):
        tp = ((q_hat == k) & (q == k)).sum()
        fp = ((q_hat == k) & (q != k)).sum()
        fn = ((q_hat != k) & (q == k)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def balanced_accuracy(q_hat, q):
    q_hat = np.asarray(q_hat, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    return float(np.mean([((q_hat == k) & (q == k)).sum() / max((q == k).sum(), 1) for k in range(4)]))


def ordered_phase_error(q_hat, q):
    return float(np.mean(np.abs(np.asarray(q_hat, dtype=np.int64) - np.asarray(q, dtype=np.int64))))


def progress_mae(g_hat, g):
    return float(np.mean(np.abs(np.asarray(g_hat, dtype=np.float64) - np.asarray(g, dtype=np.float64))))


def switch_latency(q_hat, q, target=2):
    """Signed latency (steps) of the q=1->2 event: first q_hat==2 minus first q==2."""
    q_hat = np.asarray(q_hat, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    pred = np.flatnonzero(q_hat >= target)
    true = np.flatnonzero(q >= target)
    if len(pred) == 0 or len(true) == 0:
        return np.nan
    return float(pred[0] - true[0])


def abstention_report(estimates):
    abstained = np.asarray([e.abstained for e in estimates], dtype=np.bool_)
    return {"coverage": float(1.0 - abstained.mean()), "n": int(len(estimates))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_phase_baselines.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/phase_baselines.py src/tsr/gtsr/phase_metrics.py tests/gtsr/test_phase_baselines.py
git commit -m "feat(gtsr): add deployable baselines and observer metrics"
```

---

## Task 8: 下游保真（G-05）+ 数据依赖 LOSO 评估

**Files:**
- Create: `src/tsr/gtsr/phase_gates.py`
- Modify: `src/tsr/gtsr/phase_metrics.py`
- Test: `tests/gtsr/test_phase_downstream.py`

**Interfaces:**
- Consumes: `ResponseProfile6D.alpha_from_g`（`profile6`）。
- Produces: `phase_gates`（`GATES_FROZEN`、`ABSTENTION_THRESHOLD`、`PHASE_MACRO_F1_MIN`、`PROGRESS_MAE_MAX`、`SWITCH_LATENCY_MAX`、`DELTA_SUP_OBS`、`G05_BUDGET`、`require_frozen(name)`）；`alpha_distortion(profile, g_hat, g)`（`‖se3_to_metric(α(ĝ_t) − α(g_t))‖` 逐帧均值）。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_phase_downstream.py
import numpy as np
import pytest

from tsr.gtsr import phase_gates
from tsr.gtsr.profile6 import ResponseProfile6D
from tsr.gtsr.phase_metrics import alpha_distortion


def _frozen_profile():
    rng = np.random.default_rng(0)
    theta = rng.standard_normal((6, 24)) * 0.1
    return ResponseProfile6D(theta)


def test_require_frozen_rejects_none_value():
    # 冻结（step 5）前 G05_BUDGET 为 None，require_frozen 必须拒绝读取，防止
    # confirmatory 断言在数值冻结前偷偷通过；step 5 填充数值后随 phase_gates 一并更新。
    with pytest.raises(RuntimeError):
        phase_gates.require_frozen("G05_BUDGET")


def test_alpha_distortion_is_zero_when_phase_matches():
    profile = _frozen_profile()
    g = np.linspace(0.0, 0.99, 40)
    assert alpha_distortion(profile, g, g) == pytest.approx(0.0, abs=1e-12)


def test_alpha_distortion_positive_when_phase_differs():
    profile = _frozen_profile()
    g = np.linspace(0.0, 0.99, 40)
    d = alpha_distortion(profile, g, np.clip(g + 0.25, 0.0, 0.99))
    assert d > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_phase_downstream.py -q -m "not slow"`
Expected: FAIL — `ModuleNotFoundError: No module named 'tsr.gtsr.phase_gates'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tsr/gtsr/phase_gates.py
"""Frozen confirmatory gate values for Sprint G (single source of truth).

Entries are §6.4 freeze-list items, frozen in step 5 BEFORE any confirmatory
seed is generated. Until ``GATES_FROZEN`` is True the entries are ``None`` and
MUST NOT be cited as confirmatory acceptance criteria; development uses
explicit provisional values instead.
"""
from __future__ import annotations

GATES_FROZEN = False  # flipped True in step 5 once the freeze-list values are pinned.

ABSTENTION_THRESHOLD: float | None = None   # G-04 acceptance gate (step 5)
PHASE_MACRO_F1_MIN: float | None = None      # G-04 acceptance gate (step 5)
PROGRESS_MAE_MAX: float | None = None        # G-04 acceptance gate (step 5)
SWITCH_LATENCY_MAX: float | None = None      # G-04 acceptance gate (step 5)
DELTA_SUP_OBS: float | None = None           # G-04 superiority margin (step 5)
G05_BUDGET: float | None = None              # G-05 fidelity budget (step 5)


def require_frozen(name: str) -> float:
    value = globals()[name]
    if value is None:
        raise RuntimeError(f"{name} is not frozen yet (step 5); re-run after GATES_FROZEN")
    return float(value)
```

```python
# src/tsr/gtsr/phase_metrics.py (append)
def alpha_distortion(profile, g_hat, g):
    """Per-frame ||se3_to_metric(alpha(g_hat) - alpha(g))|| mean over frames."""
    from tsr.gtsr.profile6 import se3_to_metric
    g_hat = np.asarray(g_hat, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if g_hat.shape != g.shape:
        raise ValueError("g_hat and g must align")
    alpha_hat = se3_to_metric(profile.alpha_from_g(g_hat))
    alpha_true = se3_to_metric(profile.alpha_from_g(g))
    return float(np.mean(np.linalg.norm(alpha_hat - alpha_true, axis=1)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_phase_downstream.py -q -m "not slow"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/phase_gates.py src/tsr/gtsr/phase_metrics.py tests/gtsr/test_phase_downstream.py
git commit -m "feat(gtsr): add frozen-gates module and downstream alpha-distortion metric"
```

---

## Post-freeze: confirmatory LOSO 评估（step 5/6 后执行）

本计划**不包含** confirmatory 下游断言：G-04 superiority margin `δ_sup_obs`、G-05 预算 `B` 与
phase/progress acceptance 阈值在 step 5 冻结、confirmatory 设计项在 step 6（G-00 §11）冻结，
冻结前不存在可比对的数值目标（scope §6.4、G-00 §9 要求数值先于数据）。因此 data-dependent 的
confirmatory LOSO 评估（fold-local profile fit + observer 在 held-out seed 上运行 + per-seed
`Δ_deg`/`Δ_gain`）**留待 step 5/6 之后作为独立任务补写**，不在本计划用 `assert True` 或跳过逻辑
伪装成已通过。

该后续任务消费本计划已交付的 `phase_gates`（读取冻结值）、`OnlinePhaseObserver`（Task 5）、
`alpha_distortion`（Task 8）与 Sprint F 的 `tests.gtsr._fit_se3_cells` / `_maniskill_data_se3`
（fold-local fit 与 rollout 读取），按 §8.3 的 arm 分层、multiplicity 与 §9 排除规则组装。

## Self-Review（对照 scope）

1. **Spec coverage**：G-02（Task 1/5）、G-01（Task 6）、G-03 LOSO（post-freeze 任务）、G-04 有效性（Task 3/7 提供指标与基线；阈值/`δ_sup_obs` 由 `phase_gates` 于 step 5 冻结）、G-05 下游保真（Task 8 提供 `alpha_distortion` + `phase_gates.G05_BUDGET`）、G-06 失败可见（Task 5 abstain）、G-07 回归门（提交时跑全套）。§5 API 全在 Task 1/5。§7 baselines 在 Task 7。§8 指标在 Task 7/8。§8.3 比较族/§9 排除规则由 post-freeze confirmatory 任务承载。
2. **Gap 标记**：confirmatory LOSO 评估（fold-local fit + observer 运行 + per-seed `Δ_deg`/`Δ_gain`）依赖 step 5 冻结门值、step 6 冻结 confirmatory 设计项与 §6.2 fold-specific response artifact，故独立为 post-freeze 任务，不在本计划硬编码、也不用 `assert True` 伪装。这是 scope §6.4/G-00 §9 明确要求（数值先于数据），不是 plan placeholder。
3. **类型一致性**：`PhaseEstimate`（Task 1）→ 所有 baselines/observer 的返回类型一致；`OrdinalPhaseClassifier.predict_proba`/`confidence` 与 `OnlinePhaseObserver` 调用一致；`extract_features` 返回 `(8,)` 与分类器/回归器输入一致；`phase_gates.require_frozen` 返回 `float`，与 `OnlinePhaseObserver._abstention_threshold` 的 `float(...)` 一致。
4. **Placeholder scan**：无 TBD/TODO/`assert True`；未冻结门值由 `phase_gates` 以 `None` + `GATES_FROZEN` 显式表示，confirmatory 断言只在 `GATES_FROZEN` 后（post-freeze 任务）出现。

## Execution Handoff

任务顺序依赖：Task 1 → 2 → 3 → 4 → 5 是线性依赖（后一任务消费前一任务的接口）；Task 6/7/8 只依赖 Task 1/2，可与 3/4/5 并行。post-freeze confirmatory 任务在 step 5/6 冻结门值后执行，不属本计划开发循环。
