# G-TSR Sprint C — 实施计划（严格表示能力与冻结）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 完成 P1/P2 严格证明、screw 数值佐证、示意图与冻结记录，通过硬门 G3，几何内核冻结。

**Architecture:** 无新几何模块。新增一份证明文档、一个数值测试文件、一张图、一份冻结记录；
全部建立在 Sprint A/B 已冻结的 `charts.py`/`metrics.py`/`region.py` 原语之上。

**Tech Stack:** numpy / scipy（测试）；matplotlib（`viz` extra，仅图）。

**Spec:** `docs/superpowers/specs/2026-09-03-gtsr-sprint-c-design.md`（binding authority）

## Global Constraints

- 旧 `TSR` 与 Sprint A/B 的 `src/tsr/gtsr/`、`tests/gtsr/` 一律不动（differential oracle）。
- 数值图/拟合只作佐证，不承担证明责任（spec §1、上游 §6.6）。
- 随机测试显式 `numpy.random.Generator`，固定 seeds `[0, 1, 7, 42, 20260902]`；失败样本固化为回归。
- 阈值不得为通过而放宽；任何阈值变更须记录根因/量纲/对命题的影响（上游 §10）。
- 新文件带 SPDX 头（MIT）。
- 证明用 Markdown + LaTeX 数学，与上游/设计文档一致。

---

### Task 1: P1 构造性嵌入证明

**Files:**
- Create: `docs/gtsr_expressivity_proofs.md`（notation/定义节 + P1 节）

**Interfaces:**
- Consumes: `src/tsr/gtsr/region.py::GTSR.canonical`、`src/tsr/gtsr/charts.py::CanonicalTSRChart`
  （Sprint A 已交付）；旧 `src/tsr/tsr.py::TSR` 语义（§2 baseline）。
- Produces: P1 断言 `𝔉_TSR ⊆ 𝔉_GTSR` 的构造性证明。

- [ ] 写出 notation/定义节：`𝔉_GTSR`、`𝔉_TSR`、螺距 `h(ξ)`、见证曲线 `S_h`（按 spec §2 逐字）。
- [ ] 写出 P1：构造映射 `E(T0,Te,Bw)=(T0,Te,C_TSR,Z(Bw))`；断言 `R(E)=R(TSR)` 集合相等，双向包含。
- [ ] 说明 `GTSR.canonical`/`CanonicalTSRChart.forward` 是 `E`/`Φ_TSR` 的代码对应，逐项对应数学符号；
      E-01~E-10 + 5000 差分样本为数值佐证（Sprint A 已交付）。

### Task 2: P2 严格包含证明

**Files:**
- Modify: `docs/gtsr_expressivity_proofs.md`（P2 节）

**Interfaces:**
- Consumes: Task 1 的 notation；`PoEChart`（Sprint B）。
- Produces: P2 断言 `𝔉_TSR ⊊ 𝔉_GTSR` 的严格证明（L1/L2/L3 + 奇异面 + 结论）。

- [ ] 引理 L1（螺距共轭不变）：`h(Ad_T ξ)=h(ξ)`，用 `Ad_T:[v,ω]→[Rv+p×Rω, Rω]`、
      `(Rω)ᵀ(p×Rω)=0`、`(Rω)ᵀ(Rv)=ωᵀv`。
- [ ] 引理 L2（单坐标切向螺距 ∈ {0,∞}）：单自由坐标 → 切 screw 是纯平移/纯旋转经固定 `T0,Te` 共轭。
- [ ] 引理 L3（维数/秩）：`Φ_TSR` 在 pitch≠±π/2 处 rank 6，在 pitch=±π/2 处 rank 5（roll/yaw 退化）；
      像维数 = 自由坐标数 − 奇异性亏缺。
- [ ] 情形分析：d=1 → 切向螺距 ∈{0,∞}（悖 `h≠0`）；d≥2 → 正则点 ≥2 维，或纯 roll+yaw 奇异面 1 维
      但零螺距（均悖 `S_h`）。
- [ ] 结论：`S_h∈𝔉_GTSR`（`PoEChart([ξ_h])`+1D 线性 domain）且 `S_h∉𝔉_TSR` → 严格包含。

### Task 3: screw 数值验证测试

**Files:**
- Create: `tests/gtsr/test_expressivity.py`

**Interfaces:**
- Consumes: `PoEChart`、`GTSR.canonical`、`SE3Metric`、`se3_log`、`se3_exp`（Sprint A/B）。

- [ ] X-01 非零有限螺距构造：`ξ_h=[0,0,h,0,0,1]`（轴 z，`h=0.05`），`PoEChart([ξ_h])` 表示 `S_h`；
      `q+2π` 相对 `q` 的轴向位移为 `2πh`（≤1e-8 m）。
- [ ] X-02 pitch 估计：相邻 `se3_log(T(q_i)⁻¹T(q_{i+1}))` → `ĥ`；`|ĥ−h|≤1e-8 m/rad`。
- [ ] X-03 box 多余自由体积（佐证）：z-yaw box 采样云存在到 screw 曲线的正横向偏离（box 严格 > 1 维）。
      显式注释「佐证，非证明」。
- [ ] `pytest tests/gtsr/test_expressivity.py -q` 全绿；全量回归无回退。

### Task 4: 示意图 + 冻结记录

**Files:**
- Modify: `docs/figures/generate_figures.py`（增补 `fig7_screw_vs_tsr()`，注册进 `main()`）
- Create: `docs/figures/fig7_screw_vs_tsr.png`
- Create: `docs/gtsr_freeze_record.md`

- [ ] `fig7_screw_vs_tsr()`：3D 并置 screw 曲线（轴 z，`h=0.05`，`q∈[-0.5,0.5]`）与 z-yaw box 采样云；
      标注 pitch 与横向偏离；输出 PNG。
- [ ] 安装 matplotlib（`viz` extra）并生成 `fig7_screw_vs_tsr.png`，提交。
- [ ] `docs/gtsr_freeze_record.md`：最终 API/度量/容差表 + 固定 seeds + 环境版本 + G1/G2/G3 结果表。
- [ ] 全量 `pytest -q` + `.venv/bin/ruff check/format` 全绿；`git diff --check` 干净；提交。

## 完成判定

四件交付物齐备、`pytest -q` 全绿、ruff 干净、G3 逐条核对通过 → 几何内核冻结，Sprint C 收尾。
之后才可进入 `P_r(s)` / phase estimator / transfer（G3 未过前禁止）。
