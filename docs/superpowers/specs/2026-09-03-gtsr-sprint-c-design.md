# G-TSR Sprint C — 实现级设计（严格表示能力与冻结）

状态：`DESIGN-FREEZE`
日期：`2026-09-03`
上游：`docs/gtsr_geometry_validation_plan.md`（§3.5、§6.6、§9 Sprint C、§11 硬门 G3、§12）

本文档把上游方案里 Sprint C 的数学契约落到交付物级。范围严格限定在 **Sprint C**：
P1/P2 严格证明 + screw 数值佐证 + 冻结。不实现 `P_r(s)`、phase estimator、transfer（硬门 G3
通过前禁止启动，见 §6）。

## 0. 与 Sprint A/B 的关系

Sprint A（G1）已交付 canonical 嵌入 `GTSR.canonical` 与 E/C 系列差分验证；Sprint B（G2）已交付
PoE chart + 显式 metric + 通用投影，并把 G2 收口为「冻结语料上的工程硬门」。Sprint C **不新增几何
原语、不改动任何已冻结 API**；只新增：(a) 一份证明文档；(b) 一个数值验证测试文件；(c) 一张
screw-vs-box 示意图；(d) 冻结记录。Sprint C 是「证明 + 佐证 + 归档」Sprint，不是代码 Sprint。

## 1. 范围与不变量

**交付：**

- `docs/gtsr_expressivity_proofs.md`：P1 构造性嵌入证明 + P2 严格包含证明；
- `tests/gtsr/test_expressivity.py`：screw 数值验证（X 系列）；
- `docs/figures/generate_figures.py` 增补 `fig7_screw_vs_tsr()`，输出 `docs/figures/fig7_screw_vs_tsr.png`；
- `docs/gtsr_freeze_record.md`：冻结记录（最终 API/度量/容差 + seeds + 环境 + 结果表）。

**不变量（贯穿全部）：**

- 旧 `TSR` 与 Sprint A/B 的 `src/tsr/gtsr/`、`tests/gtsr/` 一律不动，作为 differential oracle。
- **数值图/拟合只是佐证，不承担证明责任**（§6.6：数值部分「不承担遍历所有传统 TSR 的证明责任」）。
- 随机测试显式 `numpy.random.Generator`，固定 seeds `[0, 1, 7, 42, 20260902]`；失败样本固化为回归。
- 阈值不得为通过而放宽（§10：阈值变更必须记录根因、量纲、对命题的影响）。

**明确不做：** `P_r(s)`、phase estimator、transfer、sim-to-real、任何系统级结论。

## 2. 冻结的定义与记号

- 表示族 `𝔉_GTSR`：所有可由 `(T0, Te, C, Z)` 生成的位姿子集
  `R(G) = { T0 Φ_C(z) Te | z ∈ Z }`，其中 `C` 是任意 generator chart（PoE，任意 screw 生成元），
  `Z` 是有界坐标域。
- 表示族 `𝔉_TSR`：所有传统 TSR 的位姿子集，
  `Φ_TSR(z) = Trans(x,y,z) Rz(yaw) Ry(pitch) Rx(roll)`，`z` 遍历 box bounds `Bw`。
- 螺距：对 `ξ = [v, ω]`、`ω ≠ 0`，`h(ξ) = ωᵀv / ‖ω‖²`；`ω = 0`（纯平移）记 `h = ∞`；纯旋转 `h = 0`。
- 见证曲线（witness）：`S_h = { Exp(ξ_h q) | q ∈ [a,b] }`，`ξ_h = [v, ω]`，`ω ≠ 0`，
  `h = ωᵀv/‖ω‖² ≠ 0`，`a < b`。

P1 断言 `𝔉_TSR ⊆ 𝔉_GTSR`；P2 断言 `𝔉_TSR ⊊ 𝔉_GTSR`（严格），以 `S_h` 为反例。

## 3. 交付物设计

### 3.1 `docs/gtsr_expressivity_proofs.md` — 证明文档

Markdown + LaTeX 数学（`\[...\]`，与上游/设计文档一致），分两节：

**P1（构造性嵌入，`𝔉_TSR ⊆ 𝔉_GTSR`）：**

- 定义构造映射 `E(T0, Te, Bw) = (T0, Te, C_TSR, Z(Bw))`，其中 `C_TSR` 即 canonical chart、
  `Z(Bw)` 是与 `Bw` 同 bounds 的坐标域。
- 断言 `R(E(T0,Te,Bw)) = R(TSR(T0,Te,Bw))` 为集合相等，双向包含：
  - `⊇`：任一传统 TSR 位姿由某 `z ∈ Bw` 生成，其局部 `Φ_TSR(z)` 正是 canonical chart 前向，故属于 `R(E)`；
  - `⊆`：任一 `R(E)` 位姿形如 `T0 Φ_TSR(z) Te`、`z ∈ Z(Bw)`，即传统 TSR 位姿。
- 说明 `GTSR.canonical` 与 `CanonicalTSRChart.forward` 是 `E` 与 `Φ_TSR` 的代码对应
  （逐项对应数学符号）；数值佐证为 E-01~E-10 + 5000 差分样本（Sprint A 已交付）。

**P2（严格包含，`𝔉_TSR ⊊ 𝔉_GTSR`）：**

- **引理 L1（螺距共轭不变）**：对任意 `T ∈ SE(3)`、`ξ ∈ se(3)`，`h(Ad_T ξ) = h(ξ)`。证明用
  `Ad_T : [v, ω] ↦ [p×Rω + Rv, Rω]`，其中 `(Rω)ᵀ(p×Rω) = 0`（叉积垂直于自身）且
  `(Rω)ᵀ(Rv) = ωᵀv`（`R` 正交）。
- **引理 L2（单坐标切向螺距 ∈ {0, ∞}）**：传统 TSR 中仅一个坐标自由（其余零宽）时，其 1 参数族的
  切 screw 是「纯平移或纯旋转」经固定 `T0, Te` 共轭，故螺距 ∈ {0, ∞}（由 L1）。
- **引理 L3（维数）**：若传统 TSR 的像恰为 1 维嵌入子流形，则在正则点的切空间由单个非退化坐标的
  微分张成 → 螺距 ∈ {0, ∞}；≥2 个非退化坐标在正则点给 ≥2 维局部像，不能等于 1 维曲线。
- **奇异面情形（Euler，pitch = ±π/2）**：roll/yaw 退化合并为单一纯旋转（仍零螺距）；再加任一独立
  平移则局部像 ≥2 维。故奇异参数化也不能产生非零有限螺距的一维曲线。
- **结论**：`S_h`（`h ≠ 0` 有限）∈ `𝔉_GTSR`（`PoEChart([ξ_h])` + 1D 线性 domain），但其每点切 screw
  螺距为 `h ∉ {0, ∞}`，故 ∉ `𝔉_TSR`；得严格包含 `𝔉_TSR ⊊ 𝔉_GTSR`。

### 3.2 `tests/gtsr/test_expressivity.py` — 数值验证（X 系列）

| ID | 断言 | 通过条件 |
|---|---|---|
| X-01 | 非零有限螺距 screw 构造 | `PoEChart([ξ_h])` 表示 `S_h`；`q+2π` 相对 `q` 的轴向位移为 `2πh`（误差 ≤ 1e-8 m） |
| X-02 | pitch 估计 | 相邻相对变换 `se3_log(T(q_i)⁻¹ T(q_{i+1}))` → pitch `ĥ`；`|ĥ − h| ≤ 1e-8 m/rad` |
| X-03 | box 多余自由体积（佐证） | 传统 z-yaw box 采样云相对 screw 曲线存在正横向偏离（box 严格大于 1 维 screw） |

数值只用现有原语 `se3_log`/`se3_exp`/`PoEChart`/`GTSR.canonical`，不新增几何代码。X-03 显式注释
「佐证，非证明」；不把采样拟合误差写成严格不可表示的唯一依据（§6.6）。

### 3.3 `docs/figures/generate_figures.py` — 增补 `fig7_screw_vs_tsr()`

3D 图：screw 曲线 `S_h`（轴 z，`h = 0.05 m/rad`，`q ∈ [-0.5, 0.5]`）与 z-yaw box 采样云并置；标注
pitch 与横向偏离。matplotlib（`viz` extra），输出 `docs/figures/fig7_screw_vs_tsr.png`，注册进 `main()`。

### 3.4 `docs/gtsr_freeze_record.md` — 冻结记录

最终 API/度量/容差表 + 固定 seeds + 环境版本（numpy/scipy/python）+ G1/G2/G3 结果表。

## 4. 数据流（无新模块）

```
ξ_h ──PoEChart──> S_h（forward）              # X-01/X-02 构造
T(q_i)⁻¹ T(q_{i+1}) ──se3_log──> ĥ           # X-02 pitch 估计
z-yaw box ──GTSR.canonical──> 采样云          # X-03 佐证 + fig7
```

## 5. 测试矩阵（§6.6 数值部分 + G3）

- X-01/X-02/X-03 如上；全部 matplotlib-free（纯 numpy/scipy），图由 fig7 脚本独立产出。
- 失败样本固化为确定性回归；固定 seeds `[0, 1, 7, 42, 20260902]`。

## 6. 退出条件（硬门 G3：严格表示能力）

- 明确区分单个区域 `R` 与表示族 `𝔉`（§2 记号已冻结）；
- `docs/gtsr_expressivity_proofs.md` 完成 P1、P2 严格证明，L1/L2/L3 + 奇异面情形完整；
- `tests/gtsr/test_expressivity.py` X-01/X-02/X-03 全通过；`fig7_screw_vs_tsr.png` 生成并提交；
- `docs/gtsr_freeze_record.md` 归档 seeds/环境/结果表；
- 全量回归无回退（`tests/tsr/` 与 `tests/gtsr/` 全绿）。

G3 通过前，**禁止**启动 `P_r(s)`、phase estimator 或 transfer（§11 硬门规则）。
