# G-TSR Sprint E — Response Profile Identification 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 oracle-phase 有限维对角 response profile 的识别（`ResponseProfile` /
`ProfileIdentifier` / cell-status），通过 E-A（区域级可观测性）与 E-B（冻结 Pdiag 复现）两层证据，
不改冻结几何内核。

**Architecture:** 新增两个纯几何/数值模块 `src/tsr/gtsr/profile.py` 与
`src/tsr/gtsr/observability.py`，组合已冻结的 `ResponseOperator` 与 `GTSR`。E-B 复现读取 sister
repository `../maniskill_spectrum` 的冻结数据（h5py，测试级依赖），逐 seed 复现
`SmoothFinitePDiagModel` 并核对预注册门。不改 `response.py`、geometry kernel、legacy TSR。

**Tech Stack:** numpy + scipy（`least_squares`）；E-B 测试用 h5py（可选，skip-gated）。

**Spec:** `docs/superpowers/specs/2026-09-04-gtsr-sprint-e-scope.md`（binding authority，SCOPE-DRAFT）

## Global Constraints

- 不改 `GTSR`/`GeneratorChart`/`CoordinateDomain`/legacy `TSR`，不改 `metrics.py`/`charts.py`/
  `region.py`/`projection.py`/`domains.py`/`response.py`（几何内核与 Sprint-D 已冻结）。
- `ResponseProfile`（ζ→α 只读可序列化）与 `ProfileIdentifier`（D_E→Θ̂）必须与 `ResponseOperator`
  分离模块；三者都不得加载 ManiSkill/HDF5（E-B 的 HDF5 读取只在测试侧）。
- 冻结 Pdiag-finite 参数化（不许换 spline/RBF）：`alpha_max=1.25`、`n_basis=24`、`basis_width=0.065`、
  `smoothness_weight=0.1`、`nominal_iterations=3`；`least_squares(method="trf", max_nfev=500,
  ftol=xtol=gtol=1e-9)`。
- 唯一允许的 planar→6D embedding：坐标 `[du,dv,d_axial]→[du,dv,0,0,0,d_axial]`；
  系数 `[α_du,α_dv,α_yaw]→[α_du,α_dv,1,1,1,α_yaw]`。绝不把 E-B 误报为 six-generator profile identification。
- 可观测性量命名为 **symmetric residual-set discrepancy**（非 Hausdorff metric/distance），`SE3Metric`
  是 principal-log residual objective（无 triangle inequality）。
- cell status ∈ {`UNEXCITED`,`UNOBSERVABLE`,`FIT_FAILED`,`MODEL_MISMATCH`,`IDENTIFIED`}，判定顺序固定。
- 冻结 seeds `(20260818,20270818,20280818,20290818,20300818)`；context manifest SHA-256
  `3751db801cf49ce698c9b5a74ce36cf0b39897d967ddf2959846a3965b8bd653`；dataset SHA-256 见
  `collection_status.json`。禁止重采样/重选 intervention。
- E-B 预注册门（每 seed）：`g_translation_final>0.90`、`g_yaw_preclear>0.90`、`|g_yaw_final|<0.10`、
  `switch_absolute_error<0.03`、Pdiag task error < phase-scalar baseline。
- 新文件带 SPDX 头（MIT）；随机测试用显式 `numpy.random.Generator`。

---

### Task 1: `ResponseProfile` + `CellStatus` + SE(2) 原语

**Files:**
- Create: `src/tsr/gtsr/profile.py`
- Test: `tests/gtsr/test_profile.py`（P-01~P-04）

**Interfaces:**
- Consumes: `numpy`、`scipy.optimize.least_squares`；无几何内核依赖。
- Produces: `CellStatus`（Enum）、`adjudicate_cell(...)`、`rbf_basis(progress,n_basis,width)`、
  `ResponseProfile(theta, phase_to_q, alpha_max, basis_width)`（`.alpha(phase_code,rho)`→(3,)、
  `.alpha6(phase_code,rho)`→(6,) embedded、`.to_dict`/`.from_dict`、只读）、
  `planar_to_se3_alpha(alpha3)`→6D。

- [ ] 实现 `CellStatus`（5 值）与 `adjudicate_cell(excited,observable,fit_succeeded,meets_gate)`，判定顺序
      严格按 §7（excitation→observability→fit→adequacy→identified）。
- [ ] 实现 `rbf_basis`：centers=linspace(0,1,n_basis)，normalized Gaussian，width 0.065。
- [ ] 实现 SE(2) 私有原语（与冻结 `phase_switch_baselines.py` 逐字一致）：`pose2_matrix`、`se2_exp`、
      `se2_log`、`wrap_angle`、`METRIC_SCALE=[1,1,0.03]`、`to_metric`/`from_metric`、`socket_frame`。
- [ ] 实现 `ResponseProfile`：`alpha(phase_code,rho)=alpha_max*σ(clip(basis(g)@θᵀ,-30,30))`，其中
      `g=(q(k)+rho)/4`；`alpha6` 经 `planar_to_se3_alpha`（后三维补 `1,1,1`）；`to_dict`/`from_dict`
      roundtrip；`theta` 暴露为只读 copy。
- [ ] P-01：`adjudicate_cell` 的 5 个分支按优先级正确；P-02：`planar_to_se3_alpha([a,b,c])==[a,b,1,1,1,c]`；
      P-03：`ResponseProfile.alpha` 在 g=0/1 端点的 sigmoid 值域 `(0,alpha_max)`，`to_dict`/`from_dict`
      roundtrip 一致；P-04：`rbf_basis` 每行归一（sum=1）、中心对称。

### Task 2: `ProfileIdentifier`（冻结 Pdiag finite fit 复现）

**Files:**
- Modify: `src/tsr/gtsr/profile.py`
- Test: `tests/gtsr/test_profile.py`（P-05~P-07，合成数据）

**Interfaces:**
- Consumes: `ResponseProfile`、SE(2) 原语、`least_squares`。
- Produces: `ProfileIdentifier(alpha_max,n_basis,basis_width,smoothness_weight,nominal_iterations)`
  的 `.fit(contexts,curves,progress,phase_codes,nominal_frame_xy,nominal_frame_yaw)` →
  `ResponseProfile`；`.predict(contexts)`→(N,steps,3)；`.jacobian_diag()`→(steps,3)。

- [ ] 移植 `SmoothFinitePDiagModel` 核心（3-generator planar）：
      `_basis`、`_alpha`、`_context_twist`、`_selected_action`、`_batched_local_actions`、
      `_predict_with`、`_initial_parameters`（pointwise diagonal → clip[1e-4,αmax-1e-4] → logit →
      ridge `(BᵀB+1e-6 I)⁻¹Bᵀ`）、`_update_nominal`、`fit`（3 次 alternating nominal/profile update，
      `least_squares trf max_nfev=500 ftol=xtol=gtol=1e-9`，residual = [metric data residual (yaw wrapped),
      smoothness second-difference]）。
- [ ] `fit` 返回 `ResponseProfile(theta=parameters.reshape(3,n_basis), ...)`，记录 `optimization_success/cost/
      nfev/optimality` 到 profile 元数据。
- [ ] P-05：零 relation 合成数据下 `jacobian_diag` 趋于常量、`predict` 复现 nominal curve（atol 1e-6）。
- [ ] P-06：纯 yaw 干预合成数据（yaw 响应 = c·yaw），拟合出 `α_yaw≈1` 于 pre-clear、`α_yaw≈0` 于 final
      之前（阈值 0.05），`α_du/α_dv` 稳定。
- [ ] P-07：输入合同——非满秩 contexts、形状不匹配 curves、非有限 progress 均显式拒绝。

### Task 3: E-A 可观测性模块

**Files:**
- Create: `src/tsr/gtsr/observability.py`
- Test: `tests/gtsr/test_observability.py`（A-01~A-03）

**Interfaces:**
- Consumes: `SE3Metric`、`GTSR`、`ResponseOperator`。
- Produces: `symmetric_residual_set_discrepancy(points_a,points_b,metric)`（诚实命名）、
  `calibrate_epsilon_obs(control_discrepancies, epsilon_num, quantile=0.99)`。

- [ ] 实现 `symmetric_residual_set_discrepancy`：`max{ sup_x inf_y δ_W(x,y), sup_y inf_x δ_W(y,x) }`
      over 两组 pose samples（`to_transform` 精确采样，非 `project()`）。docstring 明示非 Hausdorff。
- [ ] 实现 `calibrate_epsilon_obs = max(ε_num, Q_0.99(control D_W))`；确定性 control → 仅 ε_num。
- [ ] A-01：full-circle yaw domain region + pure-yaw response action → `D_W ≈ 0`（< ε_obs），
      cell `UNOBSERVABLE`；narrow yaw domain 同 action → `D_W > ε_obs`（`IDENTIFIED` 前置 observable）。
- [ ] A-02：follow vs ignore（α_yaw=1 vs 0）narrow region → `D_W` 严格大于校准 floor。
- [ ] A-03：ε_obs 由 calibration/control split 冻结，held-out 前不随 held-out 改变。

### Task 4: E-B 冻结数据读取器 + 复现集成测试

**Files:**
- Create: `tests/gtsr/_maniskill_data.py`（测试侧 HDF5 读取器）
- Create: `tests/gtsr/test_sprint_e_replication.py`（E-01/E-02/E-05/E-06/E-07）

**Interfaces:**
- Consumes: `h5py`（测试依赖）、`ProfileIdentifier`、`ResponseOperator`。
- Produces: `load_seed_dataset(h5_path, bins)` → `(train_contexts,train_curves,test_contexts,test_curves,
  progress,phase_codes,nominal_frame_xy,nominal_frame_yaw,target_profile,...)`。

- [ ] 移植冻结 HDF5 提取（与 `benchmark_phase_switch_baselines.py`/`analyze_phase_switch_rollouts.py`
      逐字一致）：`episode_keys`、`complete`、`usable`、`phase_task_curve`（peg_pose per-phase →
      `resample` → unwrap yaw）、`task_curve`（concat 4 phase × 25 bins）、`progress_grid`（=(q(k)+ρ)/4，
      25 bins）、`empirical_isolated_profile`、nominal frame 恢复（mixed socket_pose 一致）。
- [ ] 复现校验：30 mixed / 8 isolated / 1 zero baseline；4 yaw + 4 translation isolated；isolated 覆盖
      signed yaw ±15°/±30° 与 signed x/y ±0.015；mixed 满秩且唯一。
- [ ] E-01：context_manifest 与 5 个 dataset SHA-256 与 `collection_status.json` 一致，不符即 fail。
- [ ] E-02：`progress_grid(bins=25)` 精确等于 `(q(k)+ρ)/4`。
- [ ] E-05：逐 seed `ProfileIdentifier.fit`，核对 `g_translation_final>0.90`、`g_yaw_preclear>0.90`、
      `|g_yaw_final|<0.10`、`switch_absolute_error<0.03`、Pdiag task error < phase-scalar baseline；
      与冻结 `multiseed_seed_summary.csv` 的阈值判断一致（不要求逐位相等）。
- [ ] E-06：对每个 held-out 干预，`planar_to_se3_alpha(α̂)` 经 `ResponseOperator.action(T_i,α6)` 的 SE(3)
      left action 与冻结 SE(2) `_selected_action` 一致（pose 差 ≤ 数值 tolerance 1e-9）。
- [ ] E-07：held-out finite response——8 isolated 的 task error 均值 < phase-scalar baseline（复现
      E-B 原始门），并对无法满足的 cell 正确输出 `MODEL_MISMATCH`/`UNEXCITED` 而非伪造 `IDENTIFIED`。
- [ ] 测试 skip-gated：`h5py` 缺失或数据路径缺失时 `pytest.skip`，不污染 CI。

### Task 5: 导出 + 全量回归 + 冻结

- [ ] `src/tsr/gtsr/__init__.py` 增 `ResponseProfile`、`ProfileIdentifier`、`CellStatus`、
      `symmetric_residual_set_discrepancy` 到 import 与 `__all__`。
- [ ] `pytest tests/gtsr/test_profile.py tests/gtsr/test_observability.py tests/gtsr/test_sprint_e_replication.py -q`
      全绿。
- [ ] 全量 `pytest -q` 无回退（基线 479 passed, 1 xfailed, 1 warning）。
- [ ] `.venv/bin/ruff check` + `format --check` 干净（改动范围）；`git diff --check` 干净；几何内核未改；提交。

## 完成判定

P-01~P-07、A-01~A-03、E-01/E-02/E-05/E-06/E-07 全绿、全量回归无回退、ruff 干净、冻结几何内核未改
→ Sprint E 收尾。通过只允许宣称：在冻结 oracle solver policy、冻结 intervention distribution 与
cell-status 合同内，有限维对角 response profile 可被识别或被正确 abstain；不宣称 online phase、
real-demonstration identifiability、全局 region observability 或 transfer（scope §9）。
