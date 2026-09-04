# G-TSR Sprint D — Response Operator 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把 `maniskill_spectrum` 已验证的 finite relation action 接到 G-TSR，提供 oracle-phase 的
response-region adapter（`ResponseOperator`），并通过 R-01~R-08 硬门。

**Architecture:** 无几何内核改动。新增一个纯几何模块 `src/tsr/gtsr/response.py`，组合已冻结的
`se3_log`/`se3_exp` 与 `GTSR`。`ResponseOperator` 只做 finite-action assembly，不拟合 `alpha`
（Sprint E），不加载 ManiSkill/HDF5。

**Tech Stack:** numpy（仅用已冻结的 `metrics.py` 原语 + `GTSR`）。

**Spec:** `docs/superpowers/specs/2026-09-04-gtsr-response-law-design.md`（binding authority，DESIGN-LOCKED）

## Global Constraints

- 不改 `GTSR`、`GeneratorChart`、`CoordinateDomain`、legacy `TSR`，不改 `metrics.py`/`charts.py`/
  `region.py`/`projection.py`/`domains.py`（几何内核已冻结，见 `docs/gtsr_freeze_record.md`）。
- 只接受 6×6 满秩 square basis；rectangular/weighted pseudo-inverse 明确排除（design §2/§3）。
- `alpha` 长度 6 有限向量；不假定 `alpha∈[0,1]`（`alpha_max` 由 experiment manifest 决定，design §3）。
- twist 顺序 `[v,w]`，与 `metrics.py` 一致。
- 随机测试显式 `numpy.random.Generator`，固定 seeds `[0,1,7,42,20260902]`；R-06 用 `20260902`。
- 阈值不得为通过而放宽；R-02/R-06/R-08 pose-level 差 `≤1e-12`。
- 新文件带 SPDX 头（MIT）。

---

### Task 1: `ResponseOperator` 核心（ctor / coordinates / action / dim）

**Files:**
- Create: `src/tsr/gtsr/response.py`
- Test: `tests/gtsr/test_response.py`（R-01、R-02）

**Interfaces:**
- Consumes: `metrics.py::se3_log`、`se3_exp`；`_validation.py::se3_transform`、`finite_vector`；
  `region.py::GTSR`。
- Produces: `ResponseOperator(reference, basis)`，`.dim`，`.coordinates(relation)`，`.action(relation, alpha)`。

- [ ] 实现 `__init__`：`reference` 经 `se3_transform` 校验；`basis` 校验 (6,6) 有限且满秩
      （`np.linalg.matrix_rank==6`），否则 `ValueError`。
- [ ] 实现 `coordinates(T) = solve(basis, se3_log(inv(T_ref) @ T))`。
- [ ] 实现 `action(T, alpha) = T_ref @ se3_exp(basis @ (alpha * coordinates(T))) @ inv(T_ref)`；
      `alpha` 经 `finite_vector(alpha, 6, "alpha")` 校验。
- [ ] R-01：非 SE(3) transform、非 6×6/奇异 basis、错误形状/非有限 alpha 全部明确拒绝。
- [ ] R-02：`B=I` 下 `action` 与 `T_ref @ Exp(alpha * Log(inv(T_ref) @ T)) @ inv(T_ref)` 差 `≤1e-12`；
      并断言 `dim == 6`。

### Task 2: `condition()` + 区域级测试

**Files:**
- Modify: `src/tsr/gtsr/response.py`（`condition`）
- Modify: `tests/gtsr/test_response.py`（R-03~R-07）

**Interfaces:**
- Consumes: `region.py::GTSR`（构造新 region：`GTSR(T0_w=..., Tw_e=..., chart=..., domain=...)`）。
- Produces: `ResponseOperator.condition(region, relation, alpha) -> GTSR`。

- [ ] 实现 `condition`：返回 `GTSR(T0_w=action(T,alpha) @ region.T0_w, Tw_e=region.Tw_e,
      chart=region.chart, domain=region.domain)`；`region` 非 `GTSR` 抛 `TypeError`；不 mutate 输入。
- [ ] R-03：pre-clear keyed `alpha=1` 对每个 generator 的 action 逐点等于 whole-frame response
      `T @ inv(T_ref)`。
- [ ] R-04：post-clear `alpha=[1,1,1,1,1,0]`；yaw-only intervention 给 identity action；
      du/roll/pitch intervention 仍传播（`= T @ inv(T_ref)`）。
- [ ] R-05：multigen `alpha=[0,1,1,1,1,0]`；du 与 yaw 单轴及组合均 identity；dv/dw/roll/pitch 传播。
- [ ] R-06：非单位 `T_ref` 与非单位 region `T0_w,Tw_e`；20 个固定 seed `20260902` domain samples
      满足 `condition(...).to_transform(z) == A @ region.to_transform(z)`，差 `≤1e-12`。
- [ ] R-07：原 region 的 `T0_w/Tw_e/chart/domain` 均不变；相同 alpha 改 domain 不改 action；
      改 alpha 不改 domain。

### Task 3: R-08 非单位 basis + 导出

**Files:**
- Modify: `src/tsr/gtsr/__init__.py`（导出 `ResponseOperator`）
- Modify: `tests/gtsr/test_response.py`（R-08）

- [ ] R-08：`B=blockdiag(R_t,R_r)`（`R_t,R_r∈SO(3)`）；由独立 `c*` 构造 `T=T_ref Exp(B c*)`，
      验证 `action(T,alpha) == T_ref Exp(B (alpha⊙c*)) inv(T_ref)`，差 `≤1e-12`。
- [ ] `__init__.py` 增 `ResponseOperator` 到 import 与 `__all__`。

### Task 4: 全量回归 + 冻结

- [ ] `pytest tests/gtsr/test_response.py -q` 全绿（R-01~R-08）。
- [ ] 全量 `pytest -q` 无回退（`tests/tsr` 309、`tests/gtsr` 原 157+1 xfail、`tests/benchmarks` 5）。
- [ ] `.venv/bin/ruff check` + `format --check` 干净（改动范围）。
- [ ] `git diff --check` 干净；`src/tsr/tsr.py` 未动；提交。

## 完成判定

R-01~R-08 全绿、全量回归无回退、ruff 干净、几何内核未改 → Sprint D 收尾。
通过只证明 oracle-profile response-region assembly；不据此声称在 ManiSkill 重现了 profile fitting
或 physical rollout 成功率（design §5）。
