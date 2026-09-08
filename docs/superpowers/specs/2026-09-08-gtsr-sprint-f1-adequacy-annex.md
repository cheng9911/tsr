# G-TSR Sprint F.1 — 6D Held-Out Adequacy 附录

状态：`SCOPE-DRAFT`（2026-09-08）

前置：Sprint F 已 `REPLICATION-LOCKED`（`d5d3d72`）。本附录不是重新做 F-04，而是修复独立
审查发现的证据缺口：F-04 证明了 36 个 optimizer fit 全部成功、以及 yaw 与非 yaw 五轴在
selectivity 上的结构区分，但**没有为六个 generator 分别建立真实 held-out adequacy gate**。
因此 Sprint F §7 明确把非 yaw 五轴的可证状态限定为 `FIT_SUCCEEDED + NON_SELECTIVE`，正式
`CellStatus.IDENTIFIED` 留待本附录。

本附录把这一缺口补齐为可执行的 adequacy 收口：在**同一冻结 SE(3) 数据**上，用 mixed
interventions 拟合 profile、用 isolated interventions 做严格 held-out 评价，并规定每个
generator 的判定结构与预注册要求。通过后，Sprint F 才可在 development set 上从
`REPLICATION-LOCKED` 升级为 `ADEQUACY-DEVELOPMENT-LOCKED`；正式的 `IDENTIFICATION-LOCKED`
仍留待 prospective confirmatory seeds（见 §5）。

## 1. 训练 / 验证严格分离

profile 拟合与 adequacy 评价使用互不重叠的数据切片，且所有数据均来自 Sprint F 已冻结的
`se3_fixed_contexts.json`（75 conditions = **60 mixed + 14 isolated + 1 baseline**）：

| 切片 | 数量 | 用途 | 能否用于拟合 |
| --- | ---: | --- | --- |
| mixed interventions | 60/arm（fit 用 6 个 frozen N=30 subsets） | 拟合 `θ∈ℝ^{6×24}` | 是 |
| isolated interventions | 14/arm（single-axis，见下） | **仅** held-out adequacy | 否 |
| baseline trajectory | 1/arm | nominal reference（zero-intervention） | 否 |

**isolated 干预不是每个 generator 各 14 条。** 14 是单个 arm 内所有 single-axis 干预的**总数**，
按 generator 拆分如下：

| generator `j` | isolated 条数 | 其中 `c_j≠0` 的 adequacy 样本数 |
| --- | ---: | ---: |
| du | 2 | 2 |
| dv | 2 | 2 |
| dw | 2 | 2 |
| roll | 2 | 2 |
| pitch | 2 | 2 |
| yaw | 4 | 4 |
| **合计** | **14** | 14 |

generator `j` 的 adequacy 评价**只**使用 `c_j≠0` 的 isolated 干预（即 single-axis 干预命中第
`j` 维的样本）；其他 generator 的 isolated 干预里 `c_j=0`，该维未被激励，不能作为 generator
`j` 的 adequacy 样本。因此非 yaw 五轴的 held-out 样本量各为 n=2、yaw 为 n=4，样本量以
**seed 内重复观测**计，独立单位仍是 execution seed（n=3）。

- `keyed` 与 `circular_honest` 两个 arm 必须**分开**拟合与**分开**评价，不得合并后只报容易的
  arm。
- 拟合复用 Sprint F 的 36 个 cell（2 arm × 3 seeds × 6 N=30 subsets）；held-out 评价在这些
  已冻结 profile 上运行，profile 参数在评价期间只读。
- **独立单位是 execution seed（n=3）**，isolated 干预是 seed 内的重复观测，不是独立样本；
  跨 18 cells 的 spread 仍只能称为 computational-cell spread。

### 1.1 完整只读 profile artifact（前置）

Sprint F 现有的缓存（`tests/gtsr/_fit_se3_cells.py`）主要保存**采样后的 profile/Jacobian**，
不是完整的部署 artifact（`theta` + `nominal_curve` + provenance）。因此"不重新拟合、直接复用
36 profiles"目前**不可执行**。本附录执行前必须：

- **确定性重新生成**：固定 RNG、数据顺序、优化器配置，逐 cell 复现 36 个 profile；
- **保存完整只读 artifact**：每个 cell 至少含 `theta`（`ℝ^{6×24}`）、`nominal_curve`（`steps×6`）、
  `alpha_max/basis_width/n_basis/n_phases`、`optimization`（success/cost/nfev/optimality），以及
  provenance：arm、train-seed IDs、subset IDs、数据 manifest SHA-256、artifact SHA-256；
- **fit/评价分离**：所有 36 个 artifact 在触碰任何 held-out isolated 数据前拟合并冻结为只读。

这些 cell-specific artifact 与 Sprint G §6.2 的 fold-specific response artifact 是同一要求的
两种表述：一个用于本附录的 held-out adequacy，一个用于 Sprint G 的下游保真。

## 2. 逐 generator held-out 指标

对每个 generator `j∈{du,dv,dw,roll,pitch,yaw}`，在其 arm 的 `c_j≠0` 条 held-out isolated
interventions 上计算：

1. **held-out transform-level response error**（见下，取代旧的坐标差公式）；
2. **physical-timeline task error**：将 profile 代入 task curve 后，在 physical timeline 上的
   端到端误差（mm-equivalent）；
3. **endpoint error**：任务终点 pose 误差；
4. **paired difference vs correct-structure baseline**：与 §3 中该 generator 的"正确结构基线"
   的逐 intervention 配对差；
5. 每个 cell 仍记录五项状态：`excitation`（60-mixed 覆盖）、`observability`（数据可用）、
   `fit_success`（optimizer 收敛）、`held_out_adequacy`（本附录新增）、以及最终两层
   verdict（identification + selectivity）。

**响应误差必须用变换级 SE(3) 残差，不是欧氏坐标差。** 旧公式
`‖α̂_j(ζ)·c_j − Δx_obs(ζ)‖` 忽略了 nominal frame、名义轨迹与 SE(3) 非交换作用。正确形式是
比较预测与观测的**变换级残差**：

\[
\delta_W\!\big(T_{\rm pred}(\zeta,c_j),\, T_{\rm obs}(\zeta,c_j)\big),
\qquad
T_{\rm pred}(\zeta,c_j)=C_0\,\mathrm{Exp}\!\big(\hat\alpha_j(\zeta)\,c_j\,\mathbf e_j\big)\,C_0^{-1}
\]

其中 `T_pred` 是 frozen profile 对 single-axis 干预 `c_j` 的预测世界系动作（复用
`ResponseProfile6D.action` 的 left-action 语义），`T_obs` 是该干预的观测变换，`δ_W` 用冻结的
`SE3_METRIC_SCALE=[1,1,1,0.03,0.03,0.03]`（`profile6.SE3_METRIC_SCALE`）度量，旋转分量按
`wrap_angle` 卷绕，与 `ProfileIdentifier6D` 的 data-residual 约定一致。

**禁用常量构造状态：** 不再使用 `observable=True`、`meets_gate=True` 等常量直接生成
`IDENTIFIED`。`held_out_adequacy` 必须来自第 1 项到第 4 项在 `c_j≠0` 条 isolated 干预上的真实
计算，`IDENTIFIED`（或其 development-set 等价物）只能由该证据导出。

## 3. 正确结构基线（arm × generator）与判据类型

ground truth 本身是非对称的，且**正确的结构基线必须按 arm × generator 定义**，不能把两个
arm 的 yaw 统一写成"优于 Frame/Scalar"：

| arm | generator | 正确结构基线（oracle） | 主张类型 |
| --- | --- | --- | --- |
| keyed | yaw | phase-selective：pre-clear `α≈1`、post-clear `α≈0` | **superiority**（优于无 phase-selectivity 的 Frame 控制） |
| circular | yaw | **Fixed**：`α_yaw≈0`（circular 阶段 yaw 不被激励） | **equivalence**（与 Fixed oracle 等价） |
| keyed / circular | du,dv,dw,roll,pitch | **Frame**：`α≈1`（always-on） | **equivalence**（与 Frame oracle 等价） |

判据类型必须各自明确，不得混用：

- **superiority**（仅 keyed yaw）：拟合 profile 的 yaw 响应误差必须**显著优于**无法表达
  phase-selective yaw 的 Frame 控制。判据：配对差
  `Δ = err_fitted − err_control` 的单侧置信区间上界 `< −δ_sup`（`δ_sup` 为预注册 superiority
  margin），否则不宣称 superiority。
- **equivalence**（circular yaw 与两个 arm 的非 yaw 五轴）：拟合 profile 必须与正确 oracle
  **等价**（TOST，two one-sided tests）：配对差 `Δ = err_fitted − err_oracle` 的双侧置信区间
  完全落在 `[−δ_eq, +δ_eq]` 内（`δ_eq` 为预注册 equivalence margin）。

> 注：`non-inferiority`（"不比基线差超过 margin"）是**另一种**主张，与本附录的"与 oracle
> 一致"目标不同；本附录不使用 non-inferiority，避免与 equivalence 混用。若某 generator
> 的实际目标是"不比某部署基线差"，须另立主张与对应区间判据。

**Frame 是主控制，shared-scalar 是可选次级控制。** 上表每个 `arm × generator` 的**主比较**
只针对 `Frame`（α≈1 always-on）或 `Fixed`（α≈0）这两个结构 oracle 之一。Sprint F 的
`SE3FrameWeightedModel`（单一共享标量权重）是**拟合**控制模型，与结构 oracle `Frame` 不是同一
对象；它只作为**可选次级控制**用于检查"一个标量是否就够"，不进入主比较方向、不参与
`δ_sup`/`δ_eq` 的主判定，其报告必须与主比较明确分开，且不得影响主方向的冻结。

判据的**数值 margin（`δ_sup` 与 `δ_eq`）必须在查看任何 confirmatory 数据之前确定**，并写死进
本附录或 G-00 预注册记录；不得在看完结果后回填。

## 4. 数据角色与确认集

本附录沿用 Sprint G 的数据角色定义（见 Sprint G scope §6）：

- 现有 3 个 SE(3) seeds `{20260818, 20270818, 20280818}` 已用于 Sprint F 开发、复现与审查，
  因此是 **development set**，不能称为未见过的 outer test；
- 本附录在 development set 上建立的 adequacy 是**开发级证据**，用于冻结 observer、特征、
  阈值与统计方案；
- 新的 execution seeds 是 **prospective confirmatory set**，必须在 observer 结构、特征、
  normalization、calibration、abstention threshold、phase/progress acceptance gates、下游
  non-inferiority budget、排除与失败规则全部冻结之后才生成，并一次性解封。

## 5. 退出条件与状态命名

F.1 在 **development set** 上通过的条件：

- 六个 generator 全部产出真实 held-out adequacy 指标（§2），无常量状态；
- keyed yaw 满足 superiority、circular yaw 满足 equivalence、两个 arm 的非 yaw 五轴满足
  equivalence（§3），`δ_sup`/`δ_eq` 已预注册；
- `keyed` 与 `circular_honest` 两个 arm 分别报告，n=3 seed 为独立单位。

**development set 通过后，状态只升级为：**

```text
ADEQUACY-DEVELOPMENT-LOCKED
```

对应的 generator 状态由 `FIT_SUCCEEDED + NON_SELECTIVE` 升级为 development 级的
`IDENTIFIED (development) + NON_SELECTIVE`。**正式的 `IDENTIFICATION-LOCKED` 不能由这 3 个
development seeds 授予**，必须等待 §4 的 prospective confirmatory seeds 通过后才能解封；届时
本附录状态才升级为 `ADEQUACY-LOCKED`。

**与 Sprint G 的关系：** Sprint G 把 frozen response profile 当作下游真值之前，必须先完成本
附录在 development set 上的收口；否则在线 observer 会把未经 held-out 验证的 profile 当作
ground truth 传播误差。

## 6. 与既有证据的关系

本附录只新增 held-out adequacy 证据层，不改变 Sprint F 的任何冻结合同（oracle-phase 拟合、
selectivity 指标、欧拉约定、manifest hashes、18 对聚合验收）。它与
`docs/superpowers/specs/2026-09-04-gtsr-response-law-evidence-annex.md` 互补：后者登记 sister
repo 的 task-specific 先验证据，本附录登记 tsr 侧 6D profile 的 held-out 验证。
