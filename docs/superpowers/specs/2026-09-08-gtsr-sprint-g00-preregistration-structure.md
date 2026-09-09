# G-TSR Sprint G-00 — 统计结构预注册（B0 + B1）

状态：`ADEQUACY-MARGINS-LOCKED` / `CONFIRMATORY-DESIGN-PENDING`（2026-09-09，B1 数值已冻结并
修正；confirmatory 设计项仍待冻结，见 §11，届时才升级为 `PREREGISTRATION-LOCKED`）

## 0. 本文档的位置

本文档是 G-00 的**统计预注册**：冻结"比较什么、怎么聚合、用什么指标、什么方向、什么
家庭、什么规则"，并（经 B1）冻结 `δ_sup`/`δ_eq`/`α`/多重比较校正因子等数值。

执行顺序（已按序完成，留作 provenance）：

1. **B0**：冻结统计结构（§1–§9），数值留空；
2. **A**：执行 F.1 implementation plan（Tasks 1–6），产出 development 级 paired difference；
3. **B1**：用 development 级证据填 `δ_sup`/`δ_eq`/`α`/校正因子并锁定（§10）。

B0 已完成；B1 已冻结 margin（§10），并在此基础上做了 2026-09-09 correction（修正 exact-test
样本量下限、optimizer failure 规则、强 gate-keeping、数值追溯措辞，见 §5/§6/§7）。完整
`PREREGISTRATION-LOCKED` **尚未**达成：confirmatory 设计项（确切 seed 数、固定/序贯、补采、
缺失替换、ties/零差值处理，见 §11）仍待冻结。§9 prospective confirmatory seeds 仍**未生成**，
须在 §9 + §11 列出的全部冻结项完成后一次性解封。

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
- **强 gate-keeping（层级顺序冻结）**：keyed×yaw 的 superiority 是**首层主张**，11 个
  equivalence 是**次层主张**。**仅当首层 superiority 通过时**，11 项 equivalence 才获得
  confirmatory 检验资格；否则 equivalence 只作描述性报告，**不得宣称确认性通过**。
- **控制原则（B1 已填）**：全局 α = 0.05，强 gate-keeping：
  1. **首层** keyed×yaw superiority：单侧检验，α = 0.05（唯一 superiority 主张，不校正）；
  2. **次层** 11 个 equivalence：Bonferroni 校正，每个 TOST 的 per-claim 水平
     `α_eq = 0.05 / 11 ≈ 0.004545`，对应每个 TOST 用两个 one-sided exact sign-flip 检验
     （上侧 `Δ < +δ_eq` 与下侧 `Δ > −δ_eq` **同时**拒绝），等价于 `(1 − 2α_eq)` 置信区间完全
     落在 `[−δ_eq, +δ_eq]` 内。
- **exact sign-flip 分辨率与样本量下限（2026-09-09 correction）**：exact sign-flip 的离散
  分辨率为单侧最小 p = `2^{-n}`、双侧最小 p = `2^{1-n}`。因此：
  - TOST 的**两个 one-sided** exact sign-flip 检验在 `α_eq = 0.004545` 下，数学最低样本量为
    `n ≥ 8`（`2^{-8} = 0.003906 < 0.004545`，而 `2^{-7} = 0.007813` 不达标）；
  - 若保留"exact 双侧 sign-flip"措辞，则至少 `n ≥ 9`（`2^{1-9} = 0.003906` 达标，`2^{1-8} = 0.007813` 不达标）；
  - **本文档采纳 TOST = 两个 one-sided exact sign-flip**（每个 one-sided 水平 `α_eq`）；
  - 为避免把"最低可计算样本量"误写成"充分样本量"，最终 confirmatory seed 数**不得少于 9**，
    并由预先冻结的功效/精度模拟确定（§11）。

## 6. margin 确定原则与冻结值（B1 已填，2026-09-09 修正措辞）

margin 必须在查看任何 confirmatory 数据之前确定，且只能来自**冻结 planar evidence**或
**development 集**，不得查看任一 confirmatory seed 后回填。以下数值已按此原则冻结（单位
metric-scale；平移分量采用米，`0.001 metric-scale = 1 mm-equivalent`（故 `0.003 = 3 mm-equivalent`），
旋转分量 1 单位按 `SE3_METRIC_SCALE` 的 `0.03 m/rad` 力臂折算，1 mm-equivalent = 0.0333 rad ≈ 1.91°）：

- `δ_eq`（equivalence margin，分层）：由 development 集测得的噪声尺度 + 部署容忍度确定。
  - 非 yaw 五轴（du,dv,dw,roll,pitch，两个 arm，equivalence to Frame）：**δ_eq = 0.003**；
  - circular × yaw（equivalence to Fixed）：**δ_eq = 0.008**。
- `δ_sup`（superiority margin，keyed × yaw）：**δ_sup = 0.005**。
- 依据（development 集实测，机器可读源数据见
  `docs/superpowers/specs/2026-09-08-gtsr-sprint-f1-adequacy-source-data.json`）：
  非 yaw 五轴 comparison mean |diff| ≤ 0.0018（单 seed 最大 0.0018786）、max seed SD = 0.000734
  （约 0.0007）；circular yaw diff = +0.0054 ± 0.0004；keyed yaw diff = −0.0232 ± 0.0011。
  margin 由这些 development 统计量确定，**非**由 confirmatory 数据回填；冻结时点 2026-09-08
  （在 §9 confirmatory seeds 生成之前）。

### 6.1 margin 的工程意义（2026-09-09 补充）

以下说明各 margin 在键插入任务里的**暂定 metric-scale 工程解释**（为何可视为"可忽略误差"或
"最小有意义收益"），以免审稿人把 margin 读成"看完结果后包住观测值"。这些解释在任务级锚点
钉死前是**暂定**的（见本节末），是否等价/优越仍以 prospective confirmatory TOST 为准。

- **非 yaw 五轴 `δ_eq = 0.003`**：du/dv/dw 是平移（3 mm），roll/pitch 是绕非插入轴的旋转
  （0.003/0.03 = 0.1 rad ≈ 5.7°）。键插入的主约束是插入轴（z）对齐，非 yaw 五轴是"标称帧
  本身的对齐残差"，3 mm / 5.7° 属**插入可忽略误差**。它同时**严格大于** development 噪声
  （comparison mean ≤ 1.8 mm、max seed SD ≈ 0.7 mm），使后续 confirmatory equivalence 判定具有
  可分辨性（margin 高于噪声尺度，不会因噪声太粗而虚等价）；是否达到等价仍以 prospective
  confirmatory TOST 为准。
- **circular × yaw `δ_eq = 0.008`**（0.008/0.03 = 0.267 rad ≈ 15.3°）：circular arm 是圆键/
  无键，绕插入轴自转不改变插入状态（§3 endpoint 语义），故该臂对 yaw **应当无响应**，主控制是
  Fixed（`α_yaw≈0`）。≈15° 是"yaw 不敏感"的宽容差：development 上 fitted 相对 Fixed 的 yaw
  多余误差仅 +0.0054（≈0.18 rad ≈ +10.3°，即 fitted 的 yaw 响应只引入 ~10° 的微小额外误差），
  δ_eq=15° 把"与 Fixed 等价"判得诚实——它承认"无响应"不必是 0，而是"剩余 yaw 响应小到可
  忽略"，而非把拟合噪声当成有意义的 yaw 响应。
- **keyed × yaw `δ_sup = 0.005`**（0.005/0.03 = 0.167 rad ≈ 9.5°）：keyed arm 有键，必须对齐
  yaw 才能插入，故 yaw 是**关键自由度**，主控制是 Frame（`α_yaw≈1`，无相位选择性）。"最小
  有意义收益"定义为 fitted 相对 Frame 至少改善 9.5°：它严格大于 development 噪声，又远小于
  观测到的 −0.0232（≈0.77 rad ≈ 44°）效果，使 superiority 是"确有收益"而非"数值上略优"。

**任务级锚点与"暂定"标注**：键/槽实际装配间隙、成功判据的位置/角度容差、仿真/传感器分辨率、
控制器允许误差——这些是 margin 工程意义的最终依据，目前在仓库中**尚未以数值形式固定**，列为
§11 的 confirmatory 设计项。因此，上文的"可忽略误差""最小有意义收益"是**暂定的 metric-scale
operational interpretation**，以 metric-scale 约定 + development 噪声地板为据，**不**以
development 观测值反推 margin；是否达到等价/优越仍以 prospective confirmatory TOST 为准。

## 7. 排除规则（结构，2026-09-09 修正 optimizer failure 规则）

排除只能在**预注册的客观理由**下发生，且排除的 seed/generator 必须**报告**，不得静默丢弃。
两类原因区别对待：

- **外部数据损坏（允许按冻结规则重新生成/替换）**：manifest SHA-256 与冻结值不符、rollout
  损坏、文件不可读。此类排除有客观、独立于算法表现的判据，允许按冻结规则重新生成或替换，
  并成对报告（含/不含替换样本）。
- **方法失败（不得当作普通排除）**：`optimization_success == False`（或 profile 拟合失败）。
  这是**方法失败**而非外部数据损坏，直接排除会系统性美化算法表现。它**计为 confirmatory gate
  failure**：该 seed/generator 的 confirmatory 判定为"失败"，不进入主分析的成功集；同时
  **报告 failure rate**，绝不从主分析中静默删除。
- **不可观测**：isolated 干预数据缺失（observability 不满足）；
- **不可评价**：某 generator 的 `c_j≠0` isolated 条数为 0（当前冻结数据不会发生）时，该
  generator 报"不可评价"，**不得**标记 `IDENTIFIED`。

排除的**判据本身**冻结于此；任何排除都要成对报告（含/不含被排除样本），不得只报排后有利结果。

## 8. 状态命名

- F.1 在 development 集通过后，状态只升级为 `ADEQUACY-DEVELOPMENT-LOCKED`；
- 对应 generator 由 `FIT_SUCCEEDED + NON_SELECTIVE` 升级为 development 级
  `IDENTIFIED (development) + NON_SELECTIVE`；
- 正式 `IDENTIFICATION-LOCKED` / `ADEQUACY-LOCKED` **不能**由 3 个 development seeds 授予，须
  等 §9 confirmatory seeds 通过。

## 9. confirmatory 数据不早生成

prospective confirmatory seeds 在以下全部冻结并**一次性解封**之前**不生成**（Sprint G §6.4）：

- 本预注册的 B0 结构 + B1 数值；
- §11 的 confirmatory 设计项（确切 seed 数、固定/序贯、补采、缺失替换、exact test 统计量与
  ties/零差值处理、margin 任务级锚点）；
- observer 结构与算法/超参数、输入 feature manifest、归一化、confidence calibration、
  abstention threshold、phase/progress acceptance gates、下游 non-inferiority budget（见 Sprint
  G scope §9 G-00 门）。

统计独立单位是 seed。exact sign-flip 的离散分辨率（§5）把最低样本量抬到 `n≥9`（双侧）或
`n≥8`（两个 one-sided 的 TOST）；最终 seed 数在解封前预注册（§11）。**development seeds
`{20260818,20270818,20280818}` 已用于 Sprint F 开发，不能重新标记为 confirmatory。**

## 10. B1 冻结数值（已完成，2026-09-08；2026-09-09 修正）

B1 已把下列数值填入并冻结。本文档状态为 `ADEQUACY-MARGINS-LOCKED`，在 §11 confirmatory 设计项
冻结后才升级为 `PREREGISTRATION-LOCKED`（不再以"B1 完成"即锁定为准）：

| 项 | 冻结值 |
| --- | --- |
| `δ_sup`（keyed × yaw superiority） | `0.005` |
| `δ_eq`（非 yaw 五轴 × 两个 arm，equivalence to Frame） | `0.003` |
| `δ_eq`（circular × yaw，equivalence to Fixed） | `0.008` |
| 全局 α | `0.05` |
| 多重比较 | 强 gate-keeping：superiority 单侧 α=0.05（首层）；11 个 equivalence（次层，仅首层通过后获 confirmatory 资格）用 Bonferroni `α_eq = 0.05/11 ≈ 0.004545` |
| 假设检验方法 | development（n=3）：paired t-interval over seeds（df=2，仅描述性）；confirmatory：TOST = 两个 one-sided exact sign-flip（每侧 `α_eq`）；数学可计算下限 `n=8`，本预注册设计下限冻结为 `n≥9`，最终确切 n 由 §11 功效/精度模拟确定 |
| 排除规则 | 外部数据损坏可替换；`optimization_success == False` 计为 confirmatory gate failure 并报告 failure rate（§7） |

之后任何"通过/不通过"判定必须引用本节冻结值；在 confirmatory seeds（§9）解封前不得改动。

## 11. confirmatory 设计待冻结项（CONFIRMATORY-DESIGN-PENDING）

以下项在本文档达成完整 `PREREGISTRATION-LOCKED` 之前**必须逐项冻结**，缺一不可：

1. **确切 confirmatory seed 数**：由预先冻结的功效/精度模拟确定，且 `n ≥ 9`（§5）。不得在看
   结果后回填。
2. **固定样本量 vs 序贯设计**：显式选择其一；若序贯，预注册 stopping rule 与 α-spending。
3. **是否允许补采**：允许与否、补采的条件与上限，均需预注册。
4. **缺失/损坏 seed 的替换规则**：区分外部数据损坏（可替换，§7）与方法失败（不可替换、计
   gate failure）的替换语义。
5. **exact sign-flip 的统计量与 ties/零差值处理**：预注册所用统计量（paired difference 的符号
   置换）与零差值（diff 恰为 0）时如何处理 ties。
6. **margin 的任务级工程锚点（§6.1）**：键/槽实际装配间隙、成功判据位置/角度容差、仿真/传感
   器分辨率、控制器允许误差——以仓库中可追溯的数值固定，而非以 development 观测反推。

在 §9 的 confirmatory seeds 解封前，本节不得改动；解封后本节连同 §5–§10 一并冻结。
