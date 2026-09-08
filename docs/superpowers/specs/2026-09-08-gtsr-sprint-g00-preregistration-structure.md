# G-TSR Sprint G-00 — 统计结构预注册（B0 + B1）

状态：`PREREGISTRATION-LOCKED`（2026-09-08，B1 数值已冻结）

## 0. 本文档的位置

本文档是 G-00 的**统计预注册**：冻结"比较什么、怎么聚合、用什么指标、什么方向、什么
家庭、什么规则"，并（经 B1）冻结 `δ_sup`/`δ_eq`/`α`/多重比较校正因子等数值。

执行顺序（已按序完成，留作 provenance）：

1. **B0**：冻结统计结构（§1–§9），数值留空；
2. **A**：执行 F.1 implementation plan（Tasks 1–6），产出 development 级 paired difference；
3. **B1**：用 development 级证据填 `δ_sup`/`δ_eq`/`α`/校正因子并锁定（§10）。

B0 与 B1 均已完成；§9 prospective confirmatory seeds 仍**未生成**，须在 §9 列出的全部冻结项
（含 observer 结构、feature manifest、阈值、下游 budget）完成并一次性解封后才生成。

## 1. 比较族（comparison family）

主比较族由 **arm × generator** 定义，共 12 个 primary comparison（2 arm × 6 generator），全部
预注册，不得在看完结果后增删：

| arm | generator | 主张类型 | 主控制/oracle | 方向 |
| --- | --- | --- | --- | --- |
| keyed | yaw | superiority | Frame（`α≈1`） | fitted 显著优于 Frame |
| circular | yaw | equivalence | Fixed（`α≈0`） | fitted 与 Fixed 等价 |
| keyed | du,dv,dw,roll,pitch | equivalence | Frame（`α≈1`） | fitted 与 Frame 等价 |
| circular | du,dv,dw,roll,pitch | equivalence | Frame（`α≈1`） | fitted 与 Frame 等价 |

- superiority：`Δ = err_fitted − err_frame`，单侧置信区间上界 `< −δ_sup`；
- equivalence（TOST）：`Δ = err_fitted − err_oracle`，双侧置信区间完全落在 `[−δ_eq, +δ_eq]`；
- **本预注册不使用 non-inferiority**（见 F.1 附录 §3）。若将来另立"不比某部署基线差"的主张，
  须在独立 scope 中另立预注册。

**Frame 是主控制，shared-scalar 是可选次级控制**（F.1 附录 §3 注）。`SE3FrameWeightedModel`
（单一共享标量权重）是**拟合**控制模型，与结构 oracle `Frame` 不是同一对象；它只作可选次级
控制，不进入主比较方向、不参与主判定，其报告与主比较分开。

## 2. 统计单位与聚合顺序

- **独立单位是 execution seed**。development 集为 `{20260818, 20270818, 20280818}`（n=3）。
- 每个 seed 的 **6 个 N=30 subset 是拟合重复**（来自同一 rollout，可能重叠），不是独立实验
  重复。
- 聚合顺序（冻结，不可改）：
  1. 每个 cell（arm×seed×subset）产出一个逐 generator summary（F.1 Task 5 `adequacy_summary`）；
  2. **先**对同一 `arm×seed` 的 6 个 subset summary 逐字段取均值（F.1 Task 5
     `aggregate_cells`），得到 per-seed summary；
  3. **再**跨 seed 报告（mean/CI 等，方法由 B1 填）。
- 独立样本量只能写 `n=3 seeds`，**不得**写 n=18（cell 数）或 n=36（cell×subset 数）。
- `keyed` 与 `circular_honest` 两个 arm **分开**拟合、**分开**评价、**分开**报告，不得合并后
  只报容易的 arm。

## 3. 指标方向与单位

- **主指标**：held-out transform-level response error
  `δ_W = ‖se3_to_metric(log(T_pred⁻¹ T_obs))‖₂`，其中
  `T_pred = C0 Exp(α̂_j c_j e_j) C0⁻¹`（F.1 附录 §2）。这是 left-invariant、坐标无关的变换级
  残差，**禁止**用 pose6 坐标差替代。
- **度量约定**：`SE3_METRIC_SCALE = [1,1,1,0.03,0.03,0.03]`（`profile6.SE3_METRIC_SCALE`），
  旋转分量按 `wrap_angle` 卷绕。
- **单位**：所有内部误差用 metric-scale 计算；mm-equivalent 只在**报告输出边界 `×1000`**，
  不在内部提前缩放。所有预注册阈值以 metric-scale 为单位（报告时注明 mm-equivalent）。
- **endpoint 语义**：
  - 完整 SE(3) endpoint error（6 DOF 全保留）用于 **response identification adequacy**，是所有
    六个 generator（含 yaw）的 adequacy 指标；
  - task-quotient endpoint error 是**独立**指标，仅用于 task-success 报告：其 frame 投影约定
    为"从 metric-scale 残差中**去掉 yaw 角分量（绕插入/键轴的分量，index 5）**再取模"，理由是
    键插入任务绕自身轴的旋转不改变插入状态；
  - **禁止用忽略 yaw 的 quotient endpoint 证明 yaw 的 response adequacy**（yaw 的 adequacy 只能
    由完整 SE(3) endpoint / response error 证明）。

## 4. 主比较方向冻结

`primary_oracle(arm, generator) -> "frame" | "fixed"` 冻结 §1 的映射：

```text
circular × yaw        -> "fixed"   （equivalence to Fixed）
其余 11 个 primary    -> "frame"   （keyed×yaw superiority；两个 arm × 非 yaw 五轴 equivalence）
```

报告**只**用 `primary_oracle` 选出的**唯一**主方向的 paired difference；禁止同时产出
`diff_vs_frame` 与 `diff_vs_fixed` 双方差后挑有利者。

## 5. 多重比较处理（结构）

- **家庭定义**：§1 的 12 个 primary comparison 构成一个 family；探索性/事后比较不得获得
  confirmatory 地位。
- **层级（gate-keeping）**：keyed×yaw 的 superiority 是**首层主张**，11 个 equivalence 是
  **次层主张**。此层级顺序冻结；首层未通过时不宣称首层结论，次层结论的解释必须受此层级约束。
- **控制原则（B1 已填）**：全局 α = 0.05，层级 gate-keeping：
  1. **首层** keyed×yaw superiority：单侧检验，α = 0.05（唯一 superiority 主张，不校正）；
  2. **次层** 11 个 equivalence：Bonferroni 校正，每个 TOST 的 per-claim 水平
     `α_eq = 0.05 / 11 ≈ 0.004545`，对应每个 TOST 用 `(1 − 2α_eq)` 置信区间落在
     `[−δ_eq, +δ_eq]` 内。

## 6. margin 确定原则与冻结值（B1 已填）

margin 必须在查看任何 confirmatory 数据之前确定，且只能来自**冻结 planar evidence**或
**development 集**，不得查看任一 confirmatory seed 后回填。以下数值已按此原则冻结（单位
metric-scale，mm-equivalent = ×1000）：

- `δ_eq`（equivalence margin，分层）：由 development 集测得的噪声尺度 + 部署容忍度确定。
  - 非 yaw 五轴（du,dv,dw,roll,pitch，两个 arm，equivalence to Frame）：**δ_eq = 0.003**；
  - circular × yaw（equivalence to Fixed）：**δ_eq = 0.008**。
- `δ_sup`（superiority margin，keyed × yaw）：**δ_sup = 0.005**。
- 依据（development 集实测，见 F.1 执行记录）：非 yaw 五轴 |diff| ≤ 0.0018、seed std ≤ 0.0007；
  circular yaw diff = +0.0054 ± 0.0004；keyed yaw diff = −0.0232 ± 0.0011。margin 由这些
  development 统计量确定，**非**由 confirmatory 数据回填；冻结时点 2026-09-08（在 §9 confirmatory
  seeds 生成之前）。

## 7. 排除规则（结构）

排除只能在**预注册的客观理由**下发生，且排除的 seed/generator 必须**报告**，不得静默丢弃：

- **manifest/完整性**：数据 manifest SHA-256 与冻结值不符、rollout 损坏；
- **拟合失败**：`optimization_success == False`（F.1 Task 2 artifact 中该字段为 `False` 的 cell）；
- **不可观测**：isolated 干预数据缺失（observability 不满足）；
- **不可评价**：某 generator 的 `c_j≠0` isolated 条数为 0（当前冻结数据不会发生）时，该
  generator 报"不可评价"，**不得**标记 `IDENTIFIED`。

排除的**判据本身**冻结于此；判据中若含数值阈值（如"拟合失败的判定"）由 B1 填。任何排除都要
成对报告（含/不含被排除样本），不得只报排后有利结果。

## 8. 状态命名

- F.1 在 development 集通过后，状态只升级为 `ADEQUACY-DEVELOPMENT-LOCKED`；
- 对应 generator 由 `FIT_SUCCEEDED + NON_SELECTIVE` 升级为 development 级
  `IDENTIFIED (development) + NON_SELECTIVE`；
- 正式 `IDENTIFICATION-LOCKED` / `ADEQUACY-LOCKED` **不能**由 3 个 development seeds 授予，须
  等 §9 confirmatory seeds 通过。

## 9. confirmatory 数据不早生成

prospective confirmatory seeds 在以下全部冻结并**一次性解封**之前**不生成**（Sprint G §6.4）：

- 本预注册的 B0 结构 + B1 数值；
- observer 结构与算法/超参数、输入 feature manifest、归一化、confidence calibration、
  abstention threshold、phase/progress acceptance gates、下游 non-inferiority budget（见 Sprint
  G scope §9 G-00 门）。

统计独立单位是 seed。要作形式化双侧 exact sign-flip 推断，独立 seed 数须满足 `n≥6`（`n=5` 时
最小双侧 exact p=0.0625，达不到 0.05）；最终 seed 数由采集成本与 development 集上测得的 effect
size 共同决定，并在解封前预注册。**development seeds `{20260818,20270818,20280818}` 已用于
Sprint F 开发，不能重新标记为 confirmatory。**

## 10. B1 冻结数值（已完成，2026-09-08）

B1 已把下列数值填入并冻结；本文档自此升级为 `PREREGISTRATION-LOCKED`：

| 项 | 冻结值 |
| --- | --- |
| `δ_sup`（keyed × yaw superiority） | `0.005` |
| `δ_eq`（非 yaw 五轴 × 两个 arm，equivalence to Frame） | `0.003` |
| `δ_eq`（circular × yaw，equivalence to Fixed） | `0.008` |
| 全局 α | `0.05` |
| 多重比较 | 层级 gate-keeping：superiority 单侧 α=0.05；11 个 equivalence 用 Bonferroni `α_eq = 0.05/11 ≈ 0.004545` |
| 置信区间方法 | development（n=3）：paired t-interval over seeds（df=2）；confirmatory（n≥6）：exact 双侧 sign-flip |
| 排除规则数值阈值 | 无额外数值阈值（`optimization_success == False` 为布尔判据，见 §7） |

之后任何"通过/不通过"判定必须引用本节冻结值；在 confirmatory seeds（§9）解封前不得改动。
