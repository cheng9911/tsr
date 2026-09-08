# G-TSR Sprint F.1 — 6D Held-Out Adequacy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在冻结 SE(3) 数据上，为六个 generator 分别建立真实 held-out adequacy 证据层：确定性重生成 36 个完整 provenance-pinned profile artifact，并用 `c_j≠0` 的 isolated 干预计算逐 generator 的变换级响应误差、task error、endpoint error 与 paired difference（vs Frame/Fixed oracle），供 G-00 冻结 `δ_sup`/`δ_eq` 后落地 superiority/equivalence 判定。

**Architecture:** 两个测试侧模块 + 一个测试文件。`_maniskill_data_se3.py` 增加 isolated split 提取（contexts/curves/generators/baseline）；`_fit_se3_cells.py` 增加完整 artifact 持久化（theta + nominal_curve + nominal_frame_pose + provenance + SHA-256）；新模块 `_fit_se3_adequacy.py` 承载 SE(3) 变换级残差与逐 generator adequacy 计算（action 复用 `ResponseProfile6D.action` 的 left-action 语义，残差用 `profile6` 的 `SE3_METRIC_SCALE` + `wrap_angle` 约定）。判据 margin 不在本计划冻结。

**Tech Stack:** Python 3.11, numpy, scipy, pytest, h5py（数据依赖测试），threadpoolctl（单线程 BLAS）。

**Spec:** `docs/superpowers/specs/2026-09-08-gtsr-sprint-f1-adequacy-annex.md`（F.1 held-out adequacy 附录）与 `docs/superpowers/specs/2026-09-08-gtsr-sprint-g00-preregistration-structure.md`（G-00 统计结构预注册 B0）。本计划从前者论证 adequacy 计算，margin 与主比较方向/聚合/单位约定以后者为准；两份 spec 与本计划一起传给执行者。

## Global Constraints

- **不冻结 margin**：`δ_sup`（keyed yaw superiority）与 `δ_eq`（equivalence）的数值由 G-00 预注册记录冻结，本计划只产出 development 级的原始量与 paired difference，不硬编码任何 margin。
- **无常量状态**：`held_out_adequacy` 必须来自真实计算，禁止 `observable=True`/`meets_gate=True` 等常量构造 `IDENTIFIED`。
- **独立单位是 execution seed（n=3 development）**：isolated 干预是 seed 内重复观测，非独立样本；`keyed` 与 `circular_honest` 必须分开。
- **generator 计数**：非 yaw 五轴各 2 条 isolated（`c_j≠0`），yaw 4 条；只 `c_j≠0` 的行算该 generator 的 adequacy 样本。
- **度量约定**：SE(3) metric scale `[1,1,1,0.03,0.03,0.03]`（`profile6.SE3_METRIC_SCALE`）；旋转分量按 `wrap_angle` 卷绕（与 `ProfileIdentifier6D` 的 data-residual 一致）。
- **风格**：模块头带 `# SPDX-License-Identifier: MIT` / `# Copyright (c) 2025 Siddhartha Srinivasa`；`from __future__ import annotations`；类型标注 + `np.asarray`/`np.float64`。
- **确定性**：artifact 重生成固定 RNG、数据顺序、优化器配置；artifact 内容决定其 SHA-256。
- **不回归**：不改动 Sprint F 的 F-01/F-04/F-05/F-06 冻结合同与 `fit_all_pdiag_cells` 的返回签名。
- **统计独立单位与聚合**：36 cells（2 arms × 3 seeds × 6 subsets）中，同一 seed 的 6 个 N=30 subset 是**拟合重复**（来自同一 rollout，可能重叠），不是独立实验重复。聚合顺序必须先 6 subsets → 每个 `arm×seed×generator`，再跨 seed 报告；最终独立样本量 `n=3 seeds`（**不得**写成 n=18 或 n=36）。
- **主比较方向冻结（spec §3）**：`keyed×yaw` → fitted 优于 Frame；`circular×yaw` → fitted 等价/不劣于 Fixed；两个 arm × 非 yaw 五轴 → fitted 等价/不劣于 Frame。`primary_oracle(arm, generator)` 冻结此映射；报告只用主比较的 paired difference，禁止产出 `diff_vs_frame` 与 `diff_vs_fixed` 双方差后挑有利者。
- **单位**：所有内部误差单位为 metric-scale；mm-equivalent 仅在报告输出边界 `×1000`，不在内部计算里提前缩放。
- **endpoint 语义**：保留完整 SE(3) endpoint error（response identification 用）；task-quotient endpoint error 是独立指标（task-success 报告用，其 frame 投影约定由 B0 冻结）；**禁止用忽略 yaw 的 quotient endpoint 证明 yaw response adequacy**。
- **状态命名**：F.1 在 development set 通过后只升级为 `ADEQUACY-DEVELOPMENT-LOCKED`；正式 `IDENTIFICATION-LOCKED` 留待 prospective confirmatory seeds（spec §5）。

---

## File Structure

| 文件 | 职责 |
| --- | --- |
| `tests/gtsr/_maniskill_data_se3.py`（改） | 增加 `load_se3_isolated`：提取 isolated contexts/curves/generators + baseline curve/key |
| `tests/gtsr/_fit_se3_cells.py`（改） | 抽出共享 `_fit_one`；新增 `fit_all_pdiag_artifacts` + `_artifact_sha256` |
| `tests/gtsr/_fit_se3_adequacy.py`（新） | `se3_response_action`、`se3_transform_error`、`held_out_response_error`、`held_out_task_error`、`held_out_endpoint_error`、`ALPHA_FRAME`/`ALPHA_FIXED`、`adequacy_summary` |
| `tests/gtsr/test_sprint_f1_adequacy.py`（新） | Task 1/2 数据依赖测试 + Task 3/4 合成单测 + Task 6 integration |

分文件理由：loader 与 fit 基础设施各自只加一个函数；adequacy 数学集中在独立模块，纯 numpy 可单测，与 F 系列的数据依赖测试隔离。

---

## Task 1: isolated split 提取（`load_se3_isolated`）

**Files:**
- Modify: `tests/gtsr/_maniskill_data_se3.py`
- Test: `tests/gtsr/test_sprint_f1_adequacy.py`

**Interfaces:**
- Consumes: `episode_keys`/`usable`/`complete`（`_maniskill_data.py`）、`task_curve_se3`、`EXPECTED_ISOLATED_CONTEXTS_6D`。
- Produces: `load_se3_isolated(path: Path, bins: int = 25) -> dict`，返回 `isolated_keys`/`isolated_contexts`（`(14,6)`）/`isolated_curves`（`(14,100,6)`）/`isolated_generators`/`isolated_condition_ids`/`baseline_key`/`baseline_curve`（`(100,6)`）。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_sprint_f1_adequacy.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.gtsr._maniskill_data import multiseed_root
from tests.gtsr._maniskill_data_se3 import (
    EXPECTED_ISOLATED_CONTEXTS_6D,
    GENERATOR_NAMES,
    load_se3_isolated,
    se3_rollouts_root,
)
from tests.gtsr._fit_se3_adequacy import (
    ALPHA_FIXED,
    ALPHA_FRAME,
    adequacy_summary,
    held_out_response_error,
    se3_response_action,
    se3_transform_error,
)

pytest.importorskip("h5py")


def _require_spectrum() -> Path:
    root = multiseed_root()
    if not (root / "se3_fixed_contexts.json").exists():
        pytest.skip(f"SE(3) multiseed manifests not found at {root}")
    if not (se3_rollouts_root() / "keyed_seed_20260818.h5").exists():
        pytest.skip(f"SE(3) rollouts not found at {se3_rollouts_root()}")
    return root


BINS = 25


def test_load_se3_isolated_split_shape_and_generators():
    _require_spectrum()
    path = se3_rollouts_root() / "keyed_seed_20260818.h5"
    data = load_se3_isolated(path, BINS)
    assert len(data["isolated_keys"]) == 14
    assert data["isolated_contexts"].shape == (14, 6)
    assert data["isolated_curves"].shape == (14, 4 * BINS, 6)
    assert data["baseline_curve"].shape == (4 * BINS, 6)
    counts = {name: data["isolated_generators"].count(name) for name in GENERATOR_NAMES}
    assert counts == {"du": 2, "dv": 2, "dw": 2, "roll": 2, "pitch": 2, "yaw": 4}

    isolated_sorted = data["isolated_contexts"][
        np.lexsort(tuple(data["isolated_contexts"][:, i] for i in range(5, -1, -1)))
    ]
    expected_sorted = EXPECTED_ISOLATED_CONTEXTS_6D[
        np.lexsort(tuple(EXPECTED_ISOLATED_CONTEXTS_6D[:, i] for i in range(5, -1, -1)))
    ]
    np.testing.assert_allclose(isolated_sorted, expected_sorted, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py::test_load_se3_isolated_split_shape_and_generators -q`
Expected: FAIL — `ImportError: cannot import name 'load_se3_isolated'`（数据缺失时 SKIP，同样视为"待实现"）。

- [ ] **Step 3: Write minimal implementation**

```python
# tests/gtsr/_maniskill_data_se3.py (append)
def load_se3_isolated(path, bins):
    """Extract the isolated + baseline split (contexts, curves, generators) from one arm/seed .h5."""
    import h5py

    with h5py.File(path, "r") as data_file:
        usable_groups = {
            int(data_file[key].attrs["condition_id"]): key
            for key in episode_keys(data_file)
            if usable(data_file[key]) and complete(data_file[key])
        }
        isolated_cond = {
            cid: key
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) in {"du", "dv", "dw", "roll", "pitch", "yaw"}
        }
        baseline_cond = {
            cid: key for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) == "baseline"
        }
        isolated_cids = sorted(isolated_cond)
        isolated_keys = [isolated_cond[cid] for cid in isolated_cids]
        baseline_key = sorted(baseline_cond.values())[0]
        isolated_contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in isolated_keys], dtype=np.float64
        )
        isolated_curves = np.asarray(
            [task_curve_se3(data_file[key], bins) for key in isolated_keys], dtype=np.float64
        )
        isolated_generators = [str(data_file[key].attrs["generator"]) for key in isolated_keys]
        isolated_condition_ids = [int(data_file[key].attrs["condition_id"]) for key in isolated_keys]
        baseline_curve = task_curve_se3(data_file[baseline_key], bins)
    return {
        "isolated_keys": isolated_keys,
        "isolated_contexts": isolated_contexts,
        "isolated_curves": isolated_curves,
        "isolated_generators": isolated_generators,
        "isolated_condition_ids": isolated_condition_ids,
        "baseline_key": baseline_key,
        "baseline_curve": baseline_curve,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py::test_load_se3_isolated_split_shape_and_generators -q`
Expected: PASS（数据缺失时 SKIP）。

- [ ] **Step 5: Commit**

```bash
git add tests/gtsr/_maniskill_data_se3.py tests/gtsr/test_sprint_f1_adequacy.py
git commit -m "feat(gtsr): add isolated split loader for F.1 held-out adequacy"
```

---

## Task 2: 完整 provenance-pinned profile artifact 持久化

**Files:**
- Modify: `tests/gtsr/_fit_se3_cells.py`
- Test: `tests/gtsr/test_sprint_f1_adequacy.py`

**Interfaces:**
- Consumes: `_load_cell_datasets`、`nominal_frame_se3`、`progress_grid_se3`、`ProfileIdentifier6D`、`SE3_CONTEXT_MANIFEST_SHA256`。
- Produces: `fit_all_pdiag_artifacts(root, processes=None, cache_dir=None) -> dict[(task, seed, subset_id)] -> artifact`；artifact 含 `arm/seed/subset_id/source_condition_ids/theta/nominal_curve/nominal_frame_pose/alpha_max/basis_width/n_basis/n_phases/optimization_*/manifest_sha256/artifact_sha256`。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_sprint_f1_adequacy.py (append)
from tests.gtsr._fit_se3_cells import fit_all_pdiag_artifacts


def test_fit_all_pdiag_artifacts_are_complete_and_hashed():
    root = _require_spectrum()
    artifacts = fit_all_pdiag_artifacts(root)
    assert len(artifacts) == 36
    for (arm, seed, subset_id), art in artifacts.items():
        assert arm in ("keyed", "circular_honest")
        assert art["theta"].shape == (6, 24)
        assert art["nominal_curve"].shape == (4 * BINS, 6)
        assert art["nominal_frame_pose"].shape == (6,)
        assert art["optimization_success"] is True
        assert len(art["source_condition_ids"]) == 30
        assert art["manifest_sha256"]
        assert len(art["artifact_sha256"]) == 64
        assert art["artifact_sha256"] == _artifact_sha256(art)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py::test_fit_all_pdiag_artifacts_are_complete_and_hashed -q`
Expected: FAIL — `ImportError: cannot import name 'fit_all_pdiag_artifacts'`。

- [ ] **Step 3: Write minimal implementation**

先抽出共享拟合核心，再新增 artifact 版本（保持 `fit_all_pdiag_cells` 返回签名不变）：

```python
# tests/gtsr/_fit_se3_cells.py — 抽出 _fit_one（替换原 _fit_cell 主体）
import hashlib
import json

from tests.gtsr._maniskill_data_se3 import SE3_CONTEXT_MANIFEST_SHA256  # 已 import，确认存在


def _fit_one(task, seed, subset):
    import h5py
    import threadpoolctl

    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        ds = _DATASETS[task][seed]
        cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
        indices = [cid_to_idx[cid] for cid in subset["source_condition_ids"]]
        contexts = ds["mixed_contexts"][indices]
        curves = ds["mixed_curves"][indices]
        keys = [ds["mixed_keys"][i] for i in indices]
        with h5py.File(se3_rollouts_root() / f"{task}_seed_{seed}.h5", "r") as data_file:
            nominal_frame_pose = nominal_frame_se3(data_file, keys, contexts)
        identifier = ProfileIdentifier6D()
        profile = identifier.fit(contexts, curves, _PROGRESS, _PHASE_CODES, nominal_frame_pose=nominal_frame_pose)
        return identifier, profile, nominal_frame_pose


def _fit_cell(job):
    task, seed, subset = job
    identifier, profile, _ = _fit_one(task, seed, subset)
    return task, seed, subset["subset_id"], identifier.jacobian_diag(), bool(profile.optimization["success"])


def _fit_artifact_cell(job):
    task, seed, subset = job
    identifier, profile, nominal_frame_pose = _fit_one(task, seed, subset)
    optimization = profile.optimization
    artifact = {
        "arm": task,
        "seed": int(seed),
        "subset_id": subset["subset_id"],
        "source_condition_ids": list(subset["source_condition_ids"]),
        "theta": np.asarray(profile.theta, dtype=np.float64),
        "nominal_curve": np.asarray(identifier.nominal_curve, dtype=np.float64),
        "nominal_frame_pose": np.asarray(nominal_frame_pose, dtype=np.float64),
        "alpha_max": profile.alpha_max,
        "basis_width": profile.basis_width,
        "n_basis": profile.n_basis,
        "n_phases": 4,
        "optimization_success": bool(optimization["success"]),
        "optimization_cost": float(optimization["cost"]),
        "optimization_nfev": int(optimization["nfev"]),
        "optimization_optimality": float(optimization["optimality"]),
        "manifest_sha256": SE3_CONTEXT_MANIFEST_SHA256,
    }
    artifact["artifact_sha256"] = _artifact_sha256(artifact)
    return artifact


def _artifact_sha256(artifact):
    """Deterministic SHA-256 over the canonical artifact content (sorted keys, tight JSON)."""
    canonical = {
        "arm": artifact["arm"],
        "seed": int(artifact["seed"]),
        "subset_id": artifact["subset_id"],
        "source_condition_ids": list(artifact["source_condition_ids"]),
        "theta": np.asarray(artifact["theta"], dtype=np.float64).tolist(),
        "nominal_curve": np.asarray(artifact["nominal_curve"], dtype=np.float64).tolist(),
        "nominal_frame_pose": np.asarray(artifact["nominal_frame_pose"], dtype=np.float64).tolist(),
        "alpha_max": float(artifact["alpha_max"]),
        "basis_width": float(artifact["basis_width"]),
        "n_basis": int(artifact["n_basis"]),
        "n_phases": int(artifact["n_phases"]),
        "optimization_success": bool(artifact["optimization_success"]),
        "optimization_cost": float(artifact["optimization_cost"]),
        "optimization_nfev": int(artifact["optimization_nfev"]),
        "optimization_optimality": float(artifact["optimization_optimality"]),
        "manifest_sha256": artifact["manifest_sha256"],
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fit_all_pdiag_artifacts(root, processes=None, cache_dir=None):
    """Fit all 36 cells and return full provenance-pinned artifacts (theta + nominal + provenance)."""
    from tests.gtsr._maniskill_data_se3 import load_se3_subsets

    subsets = n30_subsets(load_se3_subsets(root))
    cache_dir = Path(
        cache_dir
        or os.environ.get("TSR_SE3_ARTIFACT_CACHE")
        or Path(tempfile.gettempdir()) / f"tsr_se3_artifact_cache_{_CACHE_TAG}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    _load_cell_datasets()
    jobs = [(task, seed, subset) for task in ARMS for seed in SE3_SEEDS for subset in subsets]

    artifacts = {}
    pending = []
    for job in jobs:
        task, seed, subset = job
        cache_path = cache_dir / f"{task}_{seed}_{subset['subset_id']}.npz"
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as loaded:
                artifacts[(task, seed, subset["subset_id"])] = {
                    "arm": str(loaded["arm"]),
                    "seed": int(loaded["seed"]),
                    "subset_id": str(loaded["subset_id"]),
                    "source_condition_ids": list(loaded["source_condition_ids"]),
                    "theta": loaded["theta"],
                    "nominal_curve": loaded["nominal_curve"],
                    "nominal_frame_pose": loaded["nominal_frame_pose"],
                    "alpha_max": float(loaded["alpha_max"]),
                    "basis_width": float(loaded["basis_width"]),
                    "n_basis": int(loaded["n_basis"]),
                    "n_phases": int(loaded["n_phases"]),
                    "optimization_success": bool(loaded["optimization_success"]),
                    "optimization_cost": float(loaded["optimization_cost"]),
                    "optimization_nfev": int(loaded["optimization_nfev"]),
                    "optimization_optimality": float(loaded["optimization_optimality"]),
                    "manifest_sha256": str(loaded["manifest_sha256"]),
                    "artifact_sha256": _artifact_sha256({"arm": str(loaded["arm"]), "seed": int(loaded["seed"]),
                                                         "subset_id": str(loaded["subset_id"]),
                                                         "source_condition_ids": list(loaded["source_condition_ids"]),
                                                         "theta": loaded["theta"], "nominal_curve": loaded["nominal_curve"],
                                                         "nominal_frame_pose": loaded["nominal_frame_pose"],
                                                         "alpha_max": float(loaded["alpha_max"]),
                                                         "basis_width": float(loaded["basis_width"]),
                                                         "n_basis": int(loaded["n_basis"]), "n_phases": int(loaded["n_phases"]),
                                                         "optimization_success": bool(loaded["optimization_success"]),
                                                         "optimization_cost": float(loaded["optimization_cost"]),
                                                         "optimization_nfev": int(loaded["optimization_nfev"]),
                                                         "optimization_optimality": float(loaded["optimization_optimality"]),
                                                         "manifest_sha256": str(loaded["manifest_sha256"])}),
                }
        else:
            pending.append(job)

    if pending:
        import multiprocessing

        n_workers = processes or min(8, os.cpu_count() or 1)
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for artifact in pool.imap_unordered(_fit_artifact_cell, pending):
                task, seed, subset_id = artifact["arm"], artifact["seed"], artifact["subset_id"]
                artifacts[(task, seed, subset_id)] = artifact
                np.savez(
                    cache_dir / f"{task}_{seed}_{subset_id}.npz",
                    arm=artifact["arm"],
                    seed=artifact["seed"],
                    subset_id=artifact["subset_id"],
                    source_condition_ids=np.asarray(artifact["source_condition_ids"]),
                    theta=artifact["theta"],
                    nominal_curve=artifact["nominal_curve"],
                    nominal_frame_pose=artifact["nominal_frame_pose"],
                    alpha_max=artifact["alpha_max"],
                    basis_width=artifact["basis_width"],
                    n_basis=artifact["n_basis"],
                    n_phases=artifact["n_phases"],
                    optimization_success=artifact["optimization_success"],
                    optimization_cost=artifact["optimization_cost"],
                    optimization_nfev=artifact["optimization_nfev"],
                    optimization_optimality=artifact["optimization_optimality"],
                    manifest_sha256=artifact["manifest_sha256"],
                )
    return artifacts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py::test_fit_all_pdiag_artifacts_are_complete_and_hashed -q`
Expected: PASS（数据缺失时 SKIP）。同时跑 `pytest tests/gtsr/test_sprint_f_replication.py -m "not slow" -q` 确认 F-01/F-06 无回归。

- [ ] **Step 5: Commit**

```bash
git add tests/gtsr/_fit_se3_cells.py tests/gtsr/test_sprint_f1_adequacy.py
git commit -m "feat(gtsr): persist full provenance-pinned profile artifacts for F.1 adequacy"
```

---

## Task 3: SE(3) 变换级残差核心（`se3_response_action` / `se3_transform_error`）

**Files:**
- Create: `tests/gtsr/_fit_se3_adequacy.py`
- Test: `tests/gtsr/test_sprint_f1_adequacy.py`

**Interfaces:**
- Consumes: `profile6.se3_from_pose6/se3_inverse/se3_log/se3_exp/pose6_from_se3/wrap_angle/se3_to_metric`。
- Produces:
  - `se3_response_action(nominal_frame_pose, context, alpha) -> np.ndarray (4,4)`：`C0 Exp(α ⊙ c) C0^{-1}`，复用 `ProfileIdentifier6D.action` 语义。
  - `se3_transform_error(T_pred, T_obs) -> float`：`‖se3_to_metric(pose6_from_se3(T_pred) − pose6_from_se3(T_obs))‖`，旋转卷绕。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_sprint_f1_adequacy.py (append)
from tsr.gtsr.profile6 import se3_exp, se3_from_pose6, se3_inverse, se3_to_metric


def test_se3_response_action_fixed_alpha_is_identity():
    nominal = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    context = np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0])
    T = se3_response_action(nominal, context, np.zeros(6))
    np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


def test_se3_response_action_frame_alpha_recovers_full_translation():
    nominal = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    context = np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0])
    T = se3_response_action(nominal, context, np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    expected = se3_from_pose6(np.array([0.015, 0.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(T, expected, atol=1e-12)


def test_se3_transform_error_identity_zero():
    assert se3_transform_error(np.eye(4), np.eye(4)) == pytest.approx(0.0, abs=1e-12)


def test_se3_transform_error_translation_magnitude():
    T_pred = np.eye(4)
    T_pred[:3, 3] = [0.03, 0.0, 0.0]
    assert se3_transform_error(T_pred, np.eye(4)) == pytest.approx(0.03, abs=1e-12)


def test_se3_transform_error_left_invariant():
    # Non-unit C0 with rotation: the transform-level residual must be left-invariant.
    rng = np.random.default_rng(0)
    C0 = se3_from_pose6(rng.uniform(-1.0, 1.0, 6))
    A = se3_exp(rng.uniform(-0.5, 0.5, 6))
    B = se3_exp(rng.uniform(-0.5, 0.5, 6))
    assert se3_transform_error(C0 @ A, C0 @ B) == pytest.approx(se3_transform_error(A, B), rel=1e-9)


def test_se3_transform_error_coupled_twist_is_log_norm():
    # A screw motion (coupled translation + non-commutative rotation) residual must
    # equal the metric norm of the relative log, not the pose6 coordinate difference.
    xi = np.array([0.01, 0.02, 0.03, 0.2, 0.3, 0.4])
    expected = float(np.linalg.norm(se3_to_metric(xi)))
    assert se3_transform_error(se3_exp(xi), np.eye(4)) == pytest.approx(expected, rel=1e-12)


def test_se3_response_action_conjugates_left_with_rotated_nominal():
    # Left-multiplication direction is observable: a rotated C0 must reveal a wrong
    # conjugation side (right action) as a different transform.
    nominal = np.array([0.1, 0.2, 0.3, 0.4, -0.3, 0.2])
    context = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    T = se3_response_action(nominal, context, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    C0 = se3_from_pose6(nominal)
    expected = C0 @ se3_exp(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])) @ se3_inverse(C0)
    np.testing.assert_allclose(T, expected, atol=1e-12)
    wrong = se3_inverse(C0) @ se3_exp(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])) @ C0
    assert not np.allclose(T, wrong, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py -k "response_action or transform_error" -q`
Expected: FAIL — `ImportError: cannot import name 'se3_response_action'`。

- [ ] **Step 3: Write minimal implementation**

```python
# tests/gtsr/_fit_se3_adequacy.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""SE(3) held-out adequacy computation for the frozen six-generator profile (Sprint F.1).

Transform-level residuals in the ``profile6`` convention: the response action is the
frozen left action ``C0 Exp(alpha . c) C0^{-1}`` (matching
``ProfileIdentifier6D.action``); the residual ``delta_W`` uses
``SE3_METRIC_SCALE = [1,1,1,0.03,0.03,0.03]`` with rotation components angle-wrapped.
"""
from __future__ import annotations

import numpy as np

from tsr.gtsr.profile6 import (
    pose6_from_se3,
    se3_exp,
    se3_from_pose6,
    se3_inverse,
    se3_log,
    se3_to_metric,
)

GENERATOR_INDEX = {"du": 0, "dv": 1, "dw": 2, "roll": 3, "pitch": 4, "yaw": 5}
ALPHA_FRAME = np.ones(6, dtype=np.float64)  # alpha_j = 1 (always-on)
ALPHA_FIXED = np.zeros(6, dtype=np.float64)  # alpha_j = 0 (no response)


def se3_response_action(nominal_frame_pose, context, alpha):
    """Return ``C0 Exp(alpha . c) C0^{-1}`` for a 6-vector intervention ``context``."""
    C0 = se3_from_pose6(np.asarray(nominal_frame_pose, dtype=np.float64))
    twist = se3_log(se3_from_pose6(np.asarray(context, dtype=np.float64)))
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.shape != twist.shape:
        raise ValueError(f"alpha must have shape {twist.shape}, got {alpha.shape}")
    return C0 @ se3_exp(alpha * twist) @ se3_inverse(C0)


def se3_transform_error(T_pred, T_obs):
    """``delta_W = ||se3_to_metric(log(T_pred^{-1} T_obs))||_2``.

    Left-invariant, coordinate-free transform residual; correct under non-commutative
    rotations and Euler singularities (unlike a pose6 coordinate difference).
    """
    T_pred = np.asarray(T_pred, dtype=np.float64)
    T_obs = np.asarray(T_obs, dtype=np.float64)
    r = se3_log(se3_inverse(T_pred) @ T_obs)
    return float(np.linalg.norm(se3_to_metric(r)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py -k "response_action or transform_error" -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/gtsr/_fit_se3_adequacy.py tests/gtsr/test_sprint_f1_adequacy.py
git commit -m "feat(gtsr): add SE(3) response action and transform-level residual (F.1)"
```

---

## Task 4: 逐 generator held-out 指标与 baseline oracle

**Files:**
- Modify: `tests/gtsr/_fit_se3_adequacy.py`
- Test: `tests/gtsr/test_sprint_f1_adequacy.py`

**Interfaces:**
- Consumes: `se3_response_action`、`se3_transform_error`（Task 3）。
- Produces（均返回 `float`）：
  - `held_out_response_error(nominal_frame_pose, isolated_contexts, isolated_curves, baseline_curve, generators, generator, alpha_fn)`：metric #1，action 级残差。
  - `held_out_task_error(nominal_frame_pose, nominal_curve, isolated_contexts, isolated_curves, generators, generator, alpha_fn)`：metric #2，全曲线预测误差。
  - `held_out_endpoint_error(...)`：metric #3，最终步误差。
  - `alpha_fn(step) -> (6,)` 由调用方给出（fitted 用 `profile.alpha_from_g(progress)`；oracle 用 `lambda step: ALPHA_FRAME` / `ALPHA_FIXED`）。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_sprint_f1_adequacy.py (append)
def _synthetic_isolated():
    # nominal = identity; baseline = identity poses; one "du" isolated row, alpha_du = 0.5.
    nominal = np.zeros(6)
    baseline = np.zeros((4, 6))
    contexts = np.array([[0.015, 0.0, 0.0, 0.0, 0.0, 0.0]])
    observed = np.tile(np.array([0.0075, 0.0, 0.0, 0.0, 0.0, 0.0]), (4, 1))
    generators = ["du"]
    return nominal, contexts, observed, baseline, generators


def test_held_out_response_error_zero_when_fit_matches():
    nominal, contexts, observed, baseline, generators = _synthetic_isolated()
    err = held_out_response_error(nominal, contexts, observed, baseline, generators, "du",
                                  lambda step: np.array([0.5, 0, 0, 0, 0, 0], dtype=np.float64))
    assert err == pytest.approx(0.0, abs=1e-12)


def test_held_out_response_error_positive_when_fit_wrong():
    nominal, contexts, observed, baseline, generators = _synthetic_isolated()
    err = held_out_response_error(nominal, contexts, observed, baseline, generators, "du",
                                  lambda step: np.array([0.25, 0, 0, 0, 0, 0], dtype=np.float64))
    # predicted translation = 0.25 * 0.015 = 0.00375 vs observed 0.0075 -> 0.00375.
    assert err == pytest.approx(0.00375, abs=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py -k held_out -q`
Expected: FAIL — `ImportError: cannot import name 'held_out_response_error'`。

- [ ] **Step 3: Write minimal implementation**

```python
# tests/gtsr/_fit_se3_adequacy.py (append)
def _generator_rows(generators, generator):
    return [i for i, name in enumerate(generators) if name == generator]


def _predicted_pose6(nominal_frame_pose, nominal_curve, context, alpha):
    action = se3_response_action(nominal_frame_pose, context, alpha)
    return pose6_from_se3(action @ se3_from_pose6(np.asarray(nominal_curve, dtype=np.float64)))


def held_out_response_error(nominal_frame_pose, isolated_contexts, isolated_curves,
                            baseline_curve, generators, generator, alpha_fn):
    """Metric #1: action-level transform residual, mean over c_j!=0 rows and steps."""
    rows = _generator_rows(generators, generator)
    if not rows:
        raise ValueError(f"no isolated rows for generator {generator}")
    errors = []
    for row in rows:
        context = isolated_contexts[row]
        observed = isolated_curves[row]
        for step in range(len(baseline_curve)):
            T_pred = se3_response_action(nominal_frame_pose, context, alpha_fn(step))
            T_obs = se3_from_pose6(observed[step]) @ se3_inverse(se3_from_pose6(baseline_curve[step]))
            errors.append(se3_transform_error(T_pred, T_obs))
    return float(np.mean(errors))


def held_out_task_error(nominal_frame_pose, nominal_curve, isolated_contexts, isolated_curves,
                        generators, generator, alpha_fn):
    """Metric #2: full-curve prediction error (action . nominal_curve) vs observed curve."""
    rows = _generator_rows(generators, generator)
    if not rows:
        raise ValueError(f"no isolated rows for generator {generator}")
    errors = []
    for row in rows:
        context = isolated_contexts[row]
        observed = isolated_curves[row]
        for step in range(len(nominal_curve)):
            pred = _predicted_pose6(nominal_frame_pose, nominal_curve[step], context, alpha_fn(step))
            errors.append(se3_transform_error(se3_from_pose6(pred), se3_from_pose6(observed[step])))
    return float(np.mean(errors))


def held_out_endpoint_error(nominal_frame_pose, nominal_curve, isolated_contexts, isolated_curves,
                            generators, generator, alpha_fn):
    """Metric #3: final-step component of metric #2."""
    rows = _generator_rows(generators, generator)
    if not rows:
        raise ValueError(f"no isolated rows for generator {generator}")
    errors = []
    last = len(nominal_curve) - 1
    for row in rows:
        context = isolated_contexts[row]
        observed = isolated_curves[row]
        pred = _predicted_pose6(nominal_frame_pose, nominal_curve[last], context, alpha_fn(last))
        errors.append(se3_transform_error(se3_from_pose6(pred), se3_from_pose6(observed[last])))
    return float(np.mean(errors))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py -k held_out -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/gtsr/_fit_se3_adequacy.py tests/gtsr/test_sprint_f1_adequacy.py
git commit -m "feat(gtsr): add per-generator held-out adequacy metrics and oracles (F.1)"
```

---

## Task 5: 逐 arm×seed adequacy 汇总（paired difference）

**Files:**
- Modify: `tests/gtsr/_fit_se3_adequacy.py`
- Test: `tests/gtsr/test_sprint_f1_adequacy.py`

**Interfaces:**
- Consumes: `held_out_response_error`/`held_out_task_error`/`held_out_endpoint_error`（Task 4）、`ResponseProfile6D`。
- Produces:
  - `adequacy_summary(artifact, progress, isolated) -> dict[generator]`：单 cell（单 subset）的逐 generator 指标，含 `response_error`/`task_error`/`endpoint_error`/`diff_vs_frame`/`diff_vs_fixed`。`diff_vs_frame = err_fitted − err_frame`，`diff_vs_fixed = err_fitted − err_fixed`。
  - `aggregate_cells(cell_summaries) -> dict[generator]`：把同一 `arm×seed` 的 6 个 subset cell summary 聚合为 per-seed summary（逐字段取均值；6 subsets 是拟合重复，非独立样本）。
  - `primary_oracle(arm, generator) -> str`：冻结的主比较方向（`"frame"` 或 `"fixed"`，见 Global Constraints）。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_sprint_f1_adequacy.py (append)
from tsr.gtsr.profile6 import ResponseProfile6D


def _synthetic_artifact_and_isolated():
    nominal = np.zeros(6)
    baseline = np.zeros((4, 6))
    contexts = np.array([[0.015, 0.0, 0.0, 0.0, 0.0, 0.0]])
    observed = np.tile(np.array([0.0075, 0.0, 0.0, 0.0, 0.0, 0.0]), (4, 1))
    generators = ["du"]
    isolated = {"isolated_contexts": contexts, "isolated_curves": observed,
                "baseline_curve": baseline, "isolated_generators": generators}
    # Constant logit = log(0.5/0.75) -> sigmoid = 0.5/1.25 = 0.4 -> alpha_du = 0.5 exactly,
    # so fitted matches observed (0.5 * 0.015 = 0.0075) -> zero response error; Frame(1.0)
    # overshoots (0.015) and Fixed(0) undershoots (0.0), so both paired diffs are negative.
    artifact = {"theta": np.full((6, 24), np.log(0.5 / 0.75)), "nominal_curve": baseline,
                "nominal_frame_pose": nominal, "alpha_max": 1.25, "basis_width": 0.065,
                "n_phases": 4}
    return artifact, isolated


def test_adequacy_summary_du_diff_signs():
    artifact, isolated = _synthetic_artifact_and_isolated()
    progress = np.linspace(0.0, 0.99, 4)
    summary = adequacy_summary(artifact, progress, isolated)
    assert summary["du"]["response_error"] == pytest.approx(0.0, abs=1e-12)
    assert summary["du"]["diff_vs_frame"] < 0.0
    assert summary["du"]["diff_vs_fixed"] < 0.0


def test_primary_oracle_frozen_direction():
    from tests.gtsr._fit_se3_adequacy import primary_oracle
    assert primary_oracle("keyed", "yaw") == "frame"
    assert primary_oracle("circular_honest", "yaw") == "fixed"
    assert primary_oracle("keyed", "du") == "frame"
    assert primary_oracle("circular_honest", "pitch") == "frame"


def test_aggregate_cells_means_over_subsets():
    from tests.gtsr._fit_se3_adequacy import aggregate_cells
    cells = [
        {"du": {"response_error": 1.0, "task_error": 2.0, "endpoint_error": 3.0,
                "diff_vs_frame": -0.1, "diff_vs_fixed": 0.2}},
        {"du": {"response_error": 3.0, "task_error": 4.0, "endpoint_error": 5.0,
                "diff_vs_frame": -0.3, "diff_vs_fixed": 0.4}},
    ]
    agg = aggregate_cells(cells)
    assert agg["du"]["response_error"] == pytest.approx(2.0)
    assert agg["du"]["diff_vs_frame"] == pytest.approx(-0.2)
```

注意：`theta = np.full((6,24), np.log(0.5/0.75))` 使 logit 恒定 = `log(0.5/0.75)`，sigmoid = `0.5/1.25 = 0.4`，故 fitted `alpha = 1.25 × 0.4 = 0.5` 精确成立（RBF basis 对 constant theta 退化为常数，已验证）。此构造保证 `response_error == 0` 的精确断言，无需执行者再校准数值。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py -k adequacy_summary -q`
Expected: FAIL — `ImportError: cannot import name 'adequacy_summary'`。

- [ ] **Step 3: Write minimal implementation**

```python
# tests/gtsr/_fit_se3_adequacy.py (append)
from tsr.gtsr.profile6 import ResponseProfile6D


def adequacy_summary(artifact, progress, isolated):
    """Per-generator held-out adequacy quantities for one arm/seed artifact.

    ``progress`` is the (steps,) progress grid (``progress_grid_se3(BINS)[0]``).
    Returns a dict keyed by generator name with response/task/endpoint error and the
    paired differences vs the Frame (alpha=1) and Fixed (alpha=0) oracles.
    """
    nominal_frame_pose = artifact["nominal_frame_pose"]
    nominal_curve = artifact["nominal_curve"]
    profile = ResponseProfile6D(
        artifact["theta"], alpha_max=artifact["alpha_max"],
        basis_width=artifact["basis_width"], n_phases=artifact["n_phases"],
    )
    alpha_all = profile.alpha_from_g(np.asarray(progress, dtype=np.float64))  # (steps, 6)

    def fitted(step):
        return alpha_all[step]

    def frame(step):
        return ALPHA_FRAME

    def fixed(step):
        return ALPHA_FIXED

    isolated_contexts = isolated["isolated_contexts"]
    isolated_curves = isolated["isolated_curves"]
    baseline_curve = isolated["baseline_curve"]
    generators = isolated["isolated_generators"]

    summary = {}
    for generator in GENERATOR_INDEX:
        err_fitted = held_out_response_error(nominal_frame_pose, isolated_contexts, isolated_curves,
                                             baseline_curve, generators, generator, fitted)
        err_frame = held_out_response_error(nominal_frame_pose, isolated_contexts, isolated_curves,
                                            baseline_curve, generators, generator, frame)
        err_fixed = held_out_response_error(nominal_frame_pose, isolated_contexts, isolated_curves,
                                            baseline_curve, generators, generator, fixed)
        task_err = held_out_task_error(nominal_frame_pose, nominal_curve, isolated_contexts,
                                       isolated_curves, generators, generator, fitted)
        endpoint_err = held_out_endpoint_error(nominal_frame_pose, nominal_curve, isolated_contexts,
                                               isolated_curves, generators, generator, fitted)
        summary[generator] = {
            "response_error": err_fitted,
            "task_error": task_err,
            "endpoint_error": endpoint_err,
            "diff_vs_frame": err_fitted - err_frame,
            "diff_vs_fixed": err_fitted - err_fixed,
        }
    return summary


def primary_oracle(arm, generator):
    """Frozen primary comparison direction (spec §3 / Global Constraints).

    keyed × yaw          -> "frame" (superiority: fitted must beat non-phase-selective Frame)
    circular × yaw       -> "fixed" (equivalence: fitted must match Fixed, alpha_yaw ≈ 0)
    any arm × non-yaw 5  -> "frame" (equivalence: fitted must match Frame, alpha ≈ 1)
    """
    if generator == "yaw" and arm == "circular_honest":
        return "fixed"
    return "frame"


def aggregate_cells(cell_summaries):
    """Aggregate the 6 per-subset cell summaries of one arm × seed into a per-seed summary.

    The 6 N=30 subsets are fitting replicates of the same seed, NOT independent
    samples (they draw from one rollout and may overlap), so aggregation is a
    field-wise mean across subsets. The independent unit remains the seed (n=1 here).
    """
    generators = sorted(cell_summaries[0])
    fields = sorted(cell_summaries[0][generators[0]])
    out = {}
    for generator in generators:
        out[generator] = {
            field: float(np.mean([cell[generator][field] for cell in cell_summaries]))
            for field in fields
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py -k adequacy_summary -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/gtsr/_fit_se3_adequacy.py tests/gtsr/test_sprint_f1_adequacy.py
git commit -m "feat(gtsr): add per-arm/seed adequacy summary with paired differences (F.1)"
```

---

## Task 6: 数据依赖 F.1 integration（无常量状态 + development 级报告）

**Files:**
- Modify: `tests/gtsr/test_sprint_f1_adequacy.py`

**Interfaces:**
- Consumes: `load_se3_isolated`（Task 1）、`fit_all_pdiag_artifacts`（Task 2）、`adequacy_summary`（Task 5）、`progress_grid_se3`。
- Produces: 无新 API；本任务把前五任务串成 integration 测试，断言六个 generator 全部产出真实 adequacy 量（无常量状态），并报告 development 级 paired difference（供 G-00 冻结 margin）。

- [ ] **Step 1: Write the failing test**

```python
# tests/gtsr/test_sprint_f1_adequacy.py (append)
from tests.gtsr._fit_se3_adequacy import aggregate_cells, primary_oracle
from tests.gtsr._maniskill_data_se3 import ARMS, SE3_SEEDS, progress_grid_se3


@pytest.mark.slow
def test_F1_held_out_adequacy_no_constant_status():
    root = _require_spectrum()
    artifacts = fit_all_pdiag_artifacts(root)
    progress, _ = progress_grid_se3(BINS)

    # Development-level report: one primary paired difference per arm×seed×generator,
    # selected by the frozen primary_oracle direction (never "pick the better diff").
    primary_diff = {}
    for arm in ARMS:
        for seed in SE3_SEEDS:
            subset_ids = sorted({k[2] for k in artifacts if k[:2] == (arm, seed)})
            assert len(subset_ids) == 6  # all 6 N=30 subsets participate
            isolated = load_se3_isolated(se3_rollouts_root() / f"{arm}_seed_{seed}.h5", BINS)
            cell_summaries = [
                adequacy_summary(artifacts[(arm, seed, subset_id)], progress, isolated)
                for subset_id in subset_ids
            ]
            per_seed = aggregate_cells(cell_summaries)
            for generator, values in per_seed.items():
                # No constant status: every generator produces finite real quantities.
                assert np.isfinite(values["response_error"])
                assert np.isfinite(values["task_error"])
                assert np.isfinite(values["endpoint_error"])
                assert np.isfinite(values["diff_vs_frame"])
                assert np.isfinite(values["diff_vs_fixed"])
                direction = primary_oracle(arm, generator)
                primary_diff[(arm, seed, generator)] = values[
                    "diff_vs_frame" if direction == "frame" else "diff_vs_fixed"
                ]

    # Six generators produce real quantities; the superiority/equivalence decision
    # with delta margins is frozen by G-00 (B1) and asserted there, not here.
```

**注：** 本任务是**集成测试**，无新 API（`aggregate_cells`/`primary_oracle` 已在 Task 5 定义）。`subset_ids = sorted({k[2] for k in artifacts if k[:2] == (arm, seed)})` 从 Task 2 的真实 artifact dict 取全部 6 个 N=30 子集，**不再使用占位 `"s0"`**；36 cells 全部参与。每个 `arm×seed` 先 `aggregate_cells`（6 subsets → 1 per-seed），再按 `primary_oracle(arm, generator)` 选择**唯一**主比较方向，只报告该方向的 paired difference，禁止同时保留双方差后挑有利者。margin 判定（superiority/equivalence）**不在本任务**，留待 G-00（B1）冻结 `δ_sup`/`δ_eq` 后另写 verdict 断言。

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py::test_F1_held_out_adequacy_no_constant_status -q`
Expected: 首次运行**可能直接 PASS**（数据在场时）——因为 Task 1–5 已交付全部 API，这是集成测试而非 red-green 单测。若 FAIL 为 `ImportError`/`KeyError`/`AttributeError`，说明前序任务未完成，需回退补齐。确认无数据时 SKIP。

- [ ] **Step 3: Verify all six generators are covered**

本任务无新 API。Step 3 为：确认 `per_seed` 恰好含六个 generator key（`du,dv,dw,roll,pitch,yaw`），`primary_diff` 含 `2 arm × 3 seed × 6 generator = 36` 个条目，且每个条目经 `primary_oracle` 选择了唯一主方向。integration 只做"真实量已产出、无常量状态、主方向唯一"的收口，margin 判定留待 G-00（B1）。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_sprint_f1_adequacy.py::test_F1_held_out_adequacy_no_constant_status -q`
Expected: PASS（数据缺失时 SKIP）。

- [ ] **Step 5: Commit**

```bash
git add tests/gtsr/test_sprint_f1_adequacy.py
git commit -m "test(gtsr): add F.1 held-out adequacy integration (no constant status)"
```

---

## Self-Review（对照 spec）

1. **Spec coverage**：§1 训练/验证分离 → Task 1（isolated split 提取 + generator 计数）+ Task 2（完整 artifact + provenance + SHA）；§1.1 完整 provenance-pinned artifact → Task 2；§2 逐 generator 指标（响应误差/ task error / endpoint error / paired difference / 五项状态）→ Task 4 + Task 5；§3 arm×generator 基线（Frame α=1、Fixed α=0）→ Task 4 的 `ALPHA_FRAME`/`ALPHA_FIXED` + `diff_vs_frame`/`diff_vs_fixed`；§3 主比较方向冻结 → Task 5 的 `primary_oracle`，§3 的 36≠36 聚合规则 → Task 5 的 `aggregate_cells`，两者在 Task 6 汇合为"每 arm×seed 先聚合、再按唯一主方向报告"；§4 数据角色 → 无新任务（沿用 3 个 development seeds）；§5 退出条件 → Task 6 只产出 development 级量，margin 判定留 G-00。
2. **Gap 标记**：`δ_sup`/`δ_eq` margin 与 superiority/equivalence 判定**明确不在本计划**，由 G-00（B1）预注册记录冻结（spec §3/§5 要求"margin 在查看 confirmatory 数据前确定"）。Task 6 只产出 development 级 paired difference 与 finite 收口，**不写** `assert True`、不写 verdict 断言、不宣称 superiority/equivalence；这些判定在 margin 冻结后另立（不在此计划内）。
3. **类型一致性**：`alpha_fn(step) -> (6,)` 贯穿 Task 4/5；`se3_response_action(nominal, context, alpha)` 的 `alpha` 为 `(6,)`；`adequacy_summary` 的 `progress` 为 `(steps,)`，与 `progress_grid_se3(BINS)[0]` 一致；artifact 的 `theta`(6,24)/`nominal_curve`(steps,6)/`nominal_frame_pose`(6,) 与 `ResponseProfile6D`/`se3_from_pose6` 消费一致。`aggregate_cells(list[dict[gen]dict[field]]) -> dict[gen]dict[field]` 的输入恰为 `adequacy_summary` 的输出（dict of dict），字段名（`response_error/task_error/endpoint_error/diff_vs_frame/diff_vs_fixed`）在 Task 5 两处一致；`primary_oracle(arm, generator) -> "frame"|"fixed"` 与 Task 6 的字段选择映射一致。
4. **Placeholder scan**：全计划无 `"s0"`、无 `assert True`、无 `TBD`/`TODO`/`待实现`。Task 6 的 `subset_ids` 从真实 artifact dict 取全部 6 个 N=30 子集；margin 未冻结是**显式范围界定**（`δ_sup`/`δ_eq` 由 G-00 B1 填），不是隐藏 TODO。

## Execution Handoff

任务顺序依赖：Task 1 →（并行）Task 3/4 → Task 5 → Task 6；Task 2 独立于 1/3/4，Task 6 依赖 1/2/5。建议顺序 1 → 2 → 3 → 4 → 5 → 6。
