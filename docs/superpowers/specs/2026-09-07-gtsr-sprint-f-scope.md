# G-TSR Sprint F — 6D Six-Generator Response-Profile Replication 范围契约

状态：`REPLICATION-LOCKED`（2026-09-08）

前置：Sprint E 已 `IDENTIFICATION-LOCKED`（仅 planar oracle-phase `[du,dv,yaw]`）。Sprint D
`ResponseOperator`（`15af5c7`）已是满秩 6D operator；Sprint E 的 `ProfileIdentifier` /
`ResponseProfile` 是 planar 3-generator。本 Sprint 把 response-profile identification 从
3-generator 提升到 **6-generator SE(3)**，并复现 sister repository 已冻结的 six-generator 结果。

Sprint F 不修改 `response.py`、G-TSR geometry kernel、legacy TSR、Sprint E 的 planar
`profile.py`，也不修改任何既有 `tests/tsr/`、`tests/gtsr/` 文件。

## 1. 唯一研究问题与交付

给定 frozen oracle phase `\zeta` 与 6D relation intervention `c\in\mathbb R^6`，能否从训练
interventions 识别六维有限对角 profile

\[
\hat P_r(\zeta)=\operatorname{diag}(\hat\alpha_1(\zeta),\ldots,\hat\alpha_6(\zeta)),
\qquad
[\alpha_{du},\alpha_{dv},\alpha_{dw},\alpha_{roll},\alpha_{pitch},\alpha_{yaw}]
\]

并在 **six generators 中把 yaw 识别为唯一 selective generator**？

本 Sprint 的唯一输出是：

\[
\boxed{\hat\Theta^{(6)},\quad \hat P_r^{(6)}(\zeta),\quad\text{fit + selectivity verdicts}.}
\]

verdict 分两层（§7）：**identification**（能否正确拟合该 generator）与 **selectivity**（该
generator 是否具有跨 arm 的选择性）。选择性判定不用被否决的 0.5 recovered-magnitude 阈值
（见 §6.3），而是结构排序判据加预注册 transition floor `\tau=0.2`（§6）。除 yaw 外的五个
generator 是 oracle always-on（`\alpha\approx1`）——它们应当拟合成功并判定为
**NON_SELECTIVE**（无跨 arm gap）。要进一步称为 `IDENTIFIED`，还需真实 held-out
adequacy 证据（见 §7）。

**不做：** learned phase、matched-relation transfer、online phase observer、planner/IK 成功率、
6D 数据收集（manifest 与 rollouts 均已冻结）、对 `ResponseOperator` / planar `ProfileIdentifier`
的修改，以及“任意多个 generator 可同时被选择性识别”的泛化主张（该主张属于 sister repo 的
multigen 实验，不在本 Sprint 的证据范围内）。

## 2. 观察合同（冻结 SE(3) 数据）

复用 sister repository `maniskill_spectrum` 已冻结的 six-generator supplement：

- raw rollouts：`phase_switch_symmetry_rollouts_se3/{arm}_seed_{seed}.h5`，
  `arm∈{keyed,circular_honest}`，`seed∈{20260818,20270818,20280818}`；
- 每条 episode 的 `causal_delta` 是 6-vector（`[du,dv,dw,d_roll,d_pitch,d_yaw]`）；
- task pose 取 `peg_pose`（7 列 quaternion），task curve 为 6-channel
  `[x,y,z,roll,pitch,yaw]`；
- solver phase code `s∈{-1..6}`，拟合只用 `align_keyed=3`、`enter_key=4`、`unlock_yaw=5`、
  `circular_insert=6` 四个 phase，每 phase 25-bin oracle-resampled（`bins_per_phase=25`）；
- 拟合索引：solver phase `s∈{3,4,5,6}` 映射为拟合索引 `q=s-3∈{0,1,2,3}`，progress
  `g=(q+\rho)/4`、`\rho∈[0,1]` 为 phase 内 progress。指标中的 pre/post 一律写 `q`：**pre =
  `q<2`（align + enter），post = `q≥2`（unlock + insert）**。

### 2.1 欧拉角约定（必须显式，否则重现旋转顺序错误）

冻结数据的 pose6 序列化采用 **column-vector、yaw innermost** 的局部欧拉约定：

\[
R_{\rm local}=R_x(\mathrm{roll})\,R_y(\mathrm{pitch})\,R_z(\mathrm{yaw}),
\]

与冻结 `phase_switch_se3_baselines.py` 的 `matrix_from_euler` 一致。而 TSR canonical chart 使用
`R=R_z(\mathrm{yaw})R_y(\mathrm{pitch})R_x(\mathrm{roll})`。约定分工：

- **前者**仅用于冻结 HDF5 数据的 pose6 I/O（quaternion → `[x,y,z,roll,pitch,yaw]` 与逆向）；
- **拟合与 response action** 一律走 SE(3) log/exp（`ResponseOperator` 的 `C0 Exp(P Log(C0^-1 C)) C0^-1 X0`）；
- **不得**用 `CanonicalTSRChart` 的 RPY 转换读取 sister-repo 数据。

## 3. 冻结参数化合同（6D Pdiag finite）

逐项复用 frozen Pdiag-finite parameterization，仅把生成器数从 3 提升到 6：

\[
\hat\alpha_j(\zeta)=\alpha_{\max}\,\sigma\!\left(\Phi(g(\zeta))^\top\theta_j\right),
\qquad \alpha_{\max}=1.25,\quad j=1..6,\quad \theta_j\in\mathbb R^{24}.
\]

`\Phi` 为 24 个中心均匀分布在 `[0,1]` 的 normalized Gaussian RBF，width `0.065`；训练有
second-difference smoothness weight `0.1`、three alternating nominal/profile updates；
`scipy.optimize.least_squares(method="trf", max_nfev=500, ftol=xtol=gtol=1e-9)`。

- `\Theta^{(6)}\in\mathbb R^{6\times 24}`；
- 6D metric scale 冻结为 `[1,1,1,0.03,0.03,0.03]`（translation 单位为 m，rotation 单位为
  m/rad，与 planar `ROTATION_SCALE_M_PER_RAD=0.03` 一致）；
- finite action 仍由 Sprint D `ResponseOperator` 在 SE(3) 上实现，tsr 侧不重新实现该 operator。

### 3.1 权威代码冻结（commit-pinned）

sister repo 当前 HEAD 的 `phase_switch_se3_baselines.py` 已因后续 GP/TPGMM 扩展而漂移，不再等于
冻结时内容。Sprint F 固定到 **git commit `e5ce8a3`**，并引用该 commit 下的具体 blob：

| 文件 | git blob SHA | content SHA-256 |
| --- | --- | --- |
| `phase_switch_se3_baselines.py` | `35cf06a9aeaade16662d6020f3c91e561a6c440c` | `820ac06d795f51e0332444796b37f8ac993262b5903a4c916722e04ef0e24080` |
| `benchmark_se3_transfer.py` | （该 commit 下） | `6f71d22b7b3e137dba23208227ff81c1816b8d0cbc666b86e7e2b2b09df071ec` |

两处 content SHA-256 与冻结 `se3_transfer_summary.json` 的 `source_sha256` 一致（已核对）。测试侧
须把最小参考实现 vendoring 到测试夹具中，或按上述 blob 读取，避免 sister repo 继续演化导致漂移。

## 4. 干预 manifest 与 split

冻结 manifest：`se3_fixed_contexts.json`（75 conditions = **60 mixed + 14 isolated + 1 baseline**），

- context manifest SHA-256：`30d7f46f8ce73fc94704e0cdf433087de2f129156a45c90dd3c1a516ed5f64d5`；
- experiment SHA-256：`44e219fe7c00a1ad2a7761d61618655420cbbd29f542ee2b897a4e1ec4244e4a`
  （`se3_experiment.json`）；
- subset manifest SHA-256：`b14b6d26b52daa7d26fd832666d6c195731f2180726e499dd13edc57773664f0`
  （`se3_subsets.json`）；
- generator basis：`["du","dv","dw","d_roll","d_pitch","d_yaw"]`；
- mixed scale（`generator_scale`）：translation `±0.012 m`、roll/pitch `±15°`、yaw `±30°`；
- 14 isolated = du/dv/dw `±0.015`、roll/pitch `±15°`、yaw `±15°` 与 `±30°`（各 2/2/2/2/2/4 条，
  yaw isolated 四幅度为 `-30°,-15°,+15°,+30°`）；
- mixed amplitudes 必须从 `se3_fixed_contexts.json` 原样读取，禁止重采样/重选。

**满秩判定（归一化）：** 60-mixed 的 rank-6 与 condition number 都作用在 **归一化 context
矩阵** 上：

\[
\operatorname{cond}\!\left(X\oslash[0.012,0.012,0.012,\operatorname{rad}(15),\operatorname{rad}(15),\operatorname{rad}(30)]\right)<10,
\qquad \operatorname{rank}(X_{\rm norm})=6.
\]

（原始 context matrix 的 condition number 约 51.5；归一化后约 1.83。冻结
`prepare_se3_subsets.py` 的 `diagnostics` 即为 `contexts/scale` 后取 `np.linalg.cond`。）

本 Sprint 的 primary 拟合用 **N=30 的 6 个 frozen subsets**（`se3_subsets.json` 中
`sample_size=30` 的 `protocol∈{random,qualified}` × `repeat∈{0,1,2}`，共 6 个），每个
subset × seed × arm 独立拟合，即每 arm 18 fits（6 subsets × 3 seeds），跨 arm 按相同
`(seed, subset_id)` 配成 18 对；不复制 N∈{8,15} few-shot 主张或 18-subset 全扫描，那属于后续
ablation。

## 5. 对称性 / 选择性控制

- `keyed` 与 `circular_honest` 是**两个实验 arm**，共享匹配的 intervention manifest、task anchor
  与 solver family，而非“同一 environment”的两种状态。
- `keyed` arm：yaw 在 align/enter engaged（`alpha_yaw=1`）、unlock/insert 释放（`alpha_yaw=0`）；
  du/dv/dw/roll/pitch 在 insertion 全程 always-on（`alpha=1`）。
- `circular_honest` arm：SO(2) yaw gauge symmetry，`alpha_yaw=0` everywhere；其余五轴 always-on。

跨 arm 的 relevance gap 证明的是 **response selectivity**（yaw 在 keyed 与 circular 间有可辨识的
响应差异），**不是** formal region-level observability（本 Sprint 不重建 6D region-level
observability 证据层）。recovered magnitude 不匹配理想 oracle `[1,1,0,0]` 是已知的 align-phase
transient（§6.3），不构成失败。

## 6. 选择性指标与硬门（结构排序 + 预注册 transition floor）

对每个 generator `j`，在 **keyed arm 内** 定义 clearance drop，并在 **两 arm 间** 定义 relevance
gap（`pre`/`post` 按拟合索引 `q`）：

\[
\Delta_j=\alpha_j^{\rm pre}(\text{keyed})-\alpha_j^{\rm post}(\text{keyed}),
\qquad
S_j=\left|\alpha_j^{\rm pre}(\text{keyed})-\alpha_j^{\rm pre}(\text{circular})\right|,
\]

其中 `pre = mean(α_j | q<2)`、`post = mean(α_j | q≥2)`。指标为：

- **M1**（generator selectivity）：`S_j`（keyed-vs-circular gap，跨 arm）；
- **purity**（逐 cell 比率后取均值）：`purity_r = S_{yaw,r} / Σ_j S_{j,r}`；
- **M2**（rank）：keyed arm 内 `Δ_j` 的 yaw 为 top-1 且 `Δ_yaw > 0`；
- **M3**（transition）：keyed arm 内 `Δ_yaw > 0.2`（预注册 floor `\tau=0.2`）；
- **M4**（rejected trap）：`Δ_yaw > 0.5 ∧ max_{j≠yaw} Δ_j < 0.5` 的 recovered-magnitude 阈值，仅
  保留为显式反例，不作为判据。

**聚合约定（F-04 验收层级）：** 每 arm 18 fits，按 `(seed, subset_id)` 配成 18 对 `r`；对每对
计算 `S_{j,r}`、`purity_r`（keyed 单独计算 `Δ_{j,r}`、`rank_r`、`transition_r`），headline 只在
**18 对的聚合结果**上验收：

\[
S_{yaw}=\operatorname{mean}_r S_{yaw,r},
\qquad
S_{\rm others,max}=\operatorname{mean}_r\!\left[\max_{j\ne yaw}S_{j,r}\right],
\qquad
\mathrm{purity}=\operatorname{mean}_r\!\frac{S_{yaw,r}}{\sum_j S_{j,r}},
\]

**不是** `max_j mean_r S_{j,r}`，也**不是**聚合均值之比。frozen headline（N=30，18 对）为：

| 指标 | Pdiag finite (SE(3)) | Frame-weighted (SE(3)) |
| --- | --- | --- |
| `S_yaw` | 0.6495 | 0.2302 |
| `S_others_max`（mean of per-cell max） | 0.1273 | 0.2302 |
| purity（mean of per-cell ratio） | 0.7289 | 0.1667 |
| rank（yaw top-1 clearance drop） | 1.000 | 0.000 |
| transition（`Δ_yaw > 0.2`） | 1.000 | 0.000 |

**统计独立性：** 六个 subset 在同一 execution seed 内相互重叠，因此 **18 只是每 arm 的计算单元
数，不是独立实验样本**；真正独立的 rollout seed 是 **n=3**，subset/protocol/repeat 是 seed 内的
重采样与敏感性分析。跨 18 cells 的 std 只能称为 computational-cell spread，不得按独立 `n=18`
报告置信区间或显著性。

### 硬门

- **F-01** manifest/rollout integrity：hash 命中、75/75 usable、60/14/1 split、60-mixed 归一化后
  rank-6 且 `cond<10`、14 isolated amplitude 与原 manifest 一致；
- **F-02** 6D embedding/roundtrip：对任意 `α∈[0,\alpha_{\max}]^6` 与 relation `c`，
  `identifier.action(c, α)` 与 `ResponseOperator(reference, basis).action(socket_frame6(c), α)` 的
  metric 距离 `<10^{-9}`；`jacobian_diag()` shape `== (4·bins, 6)`、`Θ^{(6)}` shape `== (6,24)`、
  metric scale `[1,1,1,0.03,0.03,0.03]`；
- **F-03** planted 6D profile recovery（无噪合成）：plant `α*_{yaw}(ζ)=1.25·(1-g(ζ))`、
  `α*_{j≠yaw}(ζ)=1.0`，用 `ResponseOperator` 生成 60-mixed 的合成 task curve 后拟合，
  `max_{j,ζ}|α̂_j(ζ)-α*_j(ζ)| ≤ 5e-3`；
- **F-04** frozen Pdiag-finite selectivity（18 对聚合）：每 arm 18 fits 全部 `fit_success`；对
  聚合结果，`|S_yaw-0.6495| ≤ 5e-3`、`S_yaw > 4·S_others_max`、`rank=1.0`、`transition=1.0`、
  `|purity-0.7289| ≤ 5e-3`；
- **F-05** negative control：Frame-weighted (SE(3)) 的聚合 rank=0.0、transition=0.0、
  `purity≈1/6`（无法表达 per-generator selectivity）；
- **F-06** 全量回归：planar E-B、`tests/tsr/`、legacy 语义无回退。

**数值容差（spec 阶段冻结）：** `±0.05` 是 gross-divergence diagnostic（仅用于快速报警，一旦超
出即说明复现方向错误）；`≤5e-3` 是正式 frozen-headline reproduction gate（对照 frozen
`S_yaw` std=0.0256、purity std=0.0765，二者是跨 subset/seed 的 computational-cell spread，不是
复现误差）。

### 6.3 已知陷阱（必须冻结，不得误判为失败）

align phase（`q=0`）含每个 generator 的真实 within-phase `0→1` rise（peg 从 lift pose 运动到
socket 的 SE(3) pose），其 per-phase 均值 ≈0.5，使 `q<2` 的 pre-mean 落到 ≈0.7 而非 oracle 1.0。
这是真实 trajectory transient，不是 smoothness prior 的衰减（`smoothness_weight` 不改变 recovered
profile）。因此 recovered keyed yaw 的 within-keyed drop `Δ_yaw≈0.48`（frozen Pdiag finite N=30
`delta_yaw_mean=0.4804`），低于 M4 的 0.5 magnitude 阈值（frozen `m4_accuracy=0.111`），但结构
判据全部通过：`S_yaw=0.6495`（M1）、rank=1.0（M2）、transition=1.0（M3，`0.48>0.2`）、
purity=0.7289。故选择性只用 clearance drop（M2/M3）与 keyed-vs-circular gap（M1/purity）评分，
**绝不用 0.5 recovered-magnitude 阈值（M4 / M2-discrimination）**。

## 7. 两层 verdict（identification 与 selectivity 分离）

拟合成功与“该 generator 具有选择性”是两个正交状态，不得合并：

1. **Identification 层**（复用 Sprint E `CellStatus` 语义，6-generator 版本）：
   `IDENTIFIED / UNOBSERVABLE / INADEQUATE / FIT_FAILED / UNEXCITED`。判定依据：excitation 来自
   60-mixed 干预覆盖，observability 来自数据可用性，fit 来自 optimizer，adequacy 来自 §6 门。
2. **Selectivity 层**（新，`SelectivityVerdict6D`）：
   `SELECTIVE / NON_SELECTIVE / ABSTAIN`。yaw 须同时满足 `S_yaw` 显著（§6 门）、rank=1、
   transition=1 才判 `SELECTIVE`；其余五轴 oracle always-on，`S_j` 低于门 → `NON_SELECTIVE`
   （它们被正确识别，只是无跨 arm 选择性）；fit 失败或数据不足 → `ABSTAIN`。

本 Sprint 实际硬门证明了 36 个 optimizer fit 全部成功与 yaw/non-yaw 的 selectivity
区分，但没有为六个 generator 分别建立 real held-out adequacy gate。因此非 yaw 五轴在本
Sprint 的可证状态是 **FIT_SUCCEEDED + NON_SELECTIVE**；正式 `CellStatus.IDENTIFIED` 留待独立
held-out adequacy 证据，不由常量 `observable/meets_gate=True` 代替。

## 8. 新模块与接口

- `src/tsr/gtsr/profile6.py`：`ResponseProfile6D`（只读、`θ∈ℝ^{6×24}`）、`ProfileIdentifier6D`
  （`fit`/`predict`/`jacobian_diag`/`action`）、`SelectivityMetrics6D`（M1/purity/M2/M3 的逐 cell
  与 18 对聚合）、`SelectivityVerdict6D`（两层 verdict）。复用 `profile.py` 的 `rbf_basis`、
  `_normalize_phase_to_q`；不修改 `profile.py`。
- 测试侧新增 `tests/gtsr/_maniskill_data_se3.py`（6D loader，含 §2.1 的欧拉 I/O 与归一化满秩
  判定）与 `tests/gtsr/_baseline_models_se3.py`（Pdiag pointwise / Full operator /
  Frame-weighted 的 SE(3) 版本，按 §3.1 commit-pinned 参考实现 vendoring），复用
  `_maniskill_data.py` 的 episode iteration/resample/sha256 辅助。
- 从 `tsr.gtsr` 导出 `ResponseProfile6D`、`ProfileIdentifier6D`、`SelectivityMetrics6D`、
  `SelectivityVerdict6D`。

**运行约束（计划须落实）：** 36 个 Pdiag fit（2 arm × 6 subsets × 3 seeds）单线程约 39 CPU
分钟。计划须包含进程级并行、每进程 BLAS 单线程（`threadpool_limits`），以及按 fit 保存可恢复
中间结果（避免中途失败后重跑全部）。

## 9. 退出条件与后续

Sprint F 通过只允许宣称：在冻结 SE(3) intervention distribution、oracle phase 与 frozen
Pdiag-finite parameterization 内，yaw 在 **six generators 中是唯一 selective generator**（其余五轴
FIT_SUCCEEDED + NON_SELECTIVE），negative control（Frame-weighted）如预测般失败。它**不**允许宣称：
learned phase、matched-relation transfer、formal region-level observability、6D few-shot
（N=8/15）、或“任意多个 generator 可同时被选择性识别”的泛化（后者属于 multigen 实验）。

F-01–F-06 已在实现基线 `d5d3d72` 通过，本文件以 `REPLICATION-LOCKED` 冻结；
完整验收事实和 identification-adequacy 边界见 `docs/gtsr_sprint_f_validation_results.md`。后续分别
scope-lock learned phase、matched-relation transfer 与 6D few-shot ablation，不得回写或放宽本 Sprint
的 oracle-phase 拟合/选择性合同。
