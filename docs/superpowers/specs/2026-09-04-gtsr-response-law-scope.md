# G-TSR 响应规律阶段 — 范围与数学契约（Scope Draft）

状态：`SCOPE-DRAFT / evidence-backed`（2026-09-04）

前置：`docs/gtsr_freeze_record.md`（G1/G2/G3 已冻结）
后续：本文件经任务实例和观测合同确认后，才可升级为实现级 design 与 plan。

第一项任务实例不再待定：采用相邻仓库 `../maniskill_spectrum` 已冻结的
`KeyedCircularPhaseSwitchSE3-v1` relation family。其证据、适用边界与复现入口见
`docs/superpowers/specs/2026-09-04-gtsr-response-law-evidence-annex.md`。

## 0. 本阶段唯一问题

在给定**oracle phase** 与已知任务 relation 的条件下，是否能从受控 relation
interventions 中识别一个 generator-wise response law，使其对任务区域产生选择性、可验证的
有限响应？

本阶段验证的是 **representation / identification**，不是闭环机器人系统性能。

## 1. 范围锁定

**本阶段交付：**

- `P_r(\zeta)` 的有限维、可解释定义；
- oracle phase 标注与 relation intervention 数据合同；
- 将已验证的 phase-switch intervention contract 映射到 G-TSR 的 SE(3) task-region interface；
- 与该问题输入/输出对齐的 baseline、指标和失败判据。

**明确不做：**

- 在线相位估计器、视觉/力觉 observer、闭环 planner 或真实机器人整机结论；
- 从单条自然示教识别 `P_r`；
- 无已知 correspondence 的跨任务迁移；
- 修改已冻结的 `src/tsr/gtsr/` 几何 API、RPY convention 或 P1/P2 证明；
- 将 intervention-based identification 称为无条件的 causal identification。

`P_r` 的辨识是离线步骤；在线阶段（未来）只查询已辨识的 law。

## 2. 已冻结的几何接口

静态 G-TSR 的区域仍是

\[
\mathcal R(\mathcal G)=
\{T_0^w\,\Phi_{\mathcal C}(z)\,T_w^e:z\in\mathcal Z\}.
\]

其中 canonical TSR chart 始终使用

\[
\Phi_{\rm TSR}(z)=\operatorname{Trans}(x,y,z)R_z(\mathrm{yaw})
R_y(\mathrm{pitch})R_x(\mathrm{roll}).
\]

响应阶段只能向该区域提供 phase/relation-conditioned 外部位姿或域参数；不得改写 chart
的 generator order、frame/action convention 或 circular-domain 语义。几何层的权威记录为
`docs/gtsr_geometry_validation_plan.md` §3 与 `docs/gtsr_expressivity_proofs.md`。

## 3. 术语与对象

### 3.1 Relation family

一个 relation family 是任务设计者事先给定的

\[
\mathfrak F=(B_r,\;\text{generator semantics},\;\text{relation frame},\;
\text{phase semantics},\;\text{admissibility semantics}).
\]

`B_r=[b_1,\ldots,b_m]\in\mathbb R^{6\times m}` 是 relation generator basis；每一列的物理
语义、单位、表达 frame 与有限扰动方向必须写入具体任务的 intervention manifest。不能仅因两个
任务看起来相似而称其 matched。

**Sprint-D specialization：** 本 scope 的一般 relation-family 合同允许 full-column-rank
`B_r\in\mathbb R^{6\times m}`；但首个 response-region implementation 限定
`B_r\in GL(6)`，只接受 6×6 可逆 basis，并使用 `B_r^{-1}`。rectangular basis、weighted
pseudo-inverse 与其单位归一化不属于 Sprint D。

给定名义 relation `T_ref` 与当前 relation `T`，定义其选择的局部坐标为

\[
c(T;T_{\rm ref})=B_r^\dagger\,\operatorname{Log}(T_{\rm ref}^{-1}T).
\]

这里的 `Log` 是已明确 frame 的 `SE(3)` principal log；`B_r^\dagger` 的加权/单位归一化必须在
实现级 design 中固定。若 `B_r` 不满列秩或 relation 不在 chart 可表达邻域内，系统必须显式拒绝
该样本，不能静默产生坐标。

### 3.2 Hybrid oracle phase

\[
\zeta=(k,\rho),\qquad k\in\{1,\ldots,K\},\quad \rho\in[0,1].
\]

`k` 是具名 interaction phase，`\rho` 是该 phase 内进度。第一版的 `\zeta` 来自 oracle
annotation，而非时间戳替代品：每个 phase 必须有可复核的进入/退出事件、`\rho` 的起点/终点、
单调性规则和 reset 规则。首个任务的 seven-phase 标签及 physical key-clear event 已在证据附录
登记；`\rho` 的逐 phase 定义将在实现级 design 中从其 frozen rollout contract 逐项移植。

### 3.3 Response law 与 admissibility law

第一版只允许对角的 generator relevance law：

\[
P_r(\zeta)=\operatorname{diag}(\alpha_1(\zeta),\ldots,\alpha_m(\zeta)),
\qquad \alpha_j(\zeta)\in[0,\alpha_{\max}].
\]

`\alpha_{\max}` 是实验 manifest 的显式超参数，而不是概率上界。现有可复现的
`maniskill_spectrum` Pdiag-finite benchmark 使用 `\alpha_{\max}=1.25`；因此不能把本合同
错误地锁死为 `[0,1]`。若具体任务要求严格“不放大”的 response，必须由该任务 manifest 明确设
`\alpha_{\max}=1`，并重做相应验证。

它将 relation perturbation 选择性映射为

\[
u_r(\zeta,T)=P_r(\zeta)c(T;T_{\rm ref}),\qquad
C_r(\zeta,T)=\operatorname{Exp}([B_ru_r(\zeta,T)]^\wedge).
\]

`P_r` 回答“哪些 relation generators 在该 phase 应传播”；它不是成功概率、不是 region bounds、
也不是允许位姿集合。执行自由度由独立对象 `\mathcal Z(\zeta)` 描述。二者是独立参数化的
component，其坐标空间可以有不同维度和不同 generator semantics：

\[
\mathcal L(\zeta)=
\bigl(\mathcal C_r,P_r(\zeta);\mathcal C_a,\mathcal Z(\zeta)\bigr).
\]

首个实现级 design 必须明确 `C_r` 如何与名义区域相乘（左/右作用及 relation frame）。在该决定
完成前，本 scope 不声称一个唯一的 response-conditioned region assembly。

## 4. 数据与辨识合同

给定名义 relation `T_ref`、oracle-aligned nominal pose trajectory `X_0(\zeta)`，采集受控样本

\[
\mathcal D=\{(T_i,X_i(\zeta_n),\zeta_n)\}_{i,n},
\qquad
T_i=T_{\rm ref}\operatorname{Exp}([B_rc_i]^\wedge).
\]

每个 intervention 必须记录：generator coordinate `c_i`、幅度/单位、relation frame、操作者或
控制器、成功标签、phase alignment 与随机 seed。每个待辨识 generator 在每个报告 phase 必须有
非零且双向（正/负或等价对照）excitation；否则该 phase/generator cell 标记为 **unidentified**，
不得被报告为 `\alpha_j=0`。

Sprint D 的冻结 evidence 中，`X_i` 是固定 response-generation policy 下的 canonical solver
response，而不是任意人类示教。拟合可以最小化对齐后的 held-out pose residual 或由 admissible
freedom quotient 后的 residual，但训练/测试 intervention 必须分离。单一 nominal trajectory 只能
给出 `X_0`，不足以辨识 `P_r`。

未来的自然示教必须另立 observation model，例如
`X_i=A(P_r,c_i)\Phi(z_i)T_w^e`，其中 `z_i` 是可能随示教者 redundancy choice 改变的 latent
within-region coordinate；不得将该问题并入 Sprint D。

## 5. 可观测性与主张边界

对 generator `j` 的有限 intervention `T_j(\delta)`，令 `\mathcal R_j^{(1)}` 与
`\mathcal R_j^{(0)}` 分别是 `\alpha_j=1` 与 `\alpha_j=0` 的候选区域。只有在任务指定的、已
quotient 掉 `\mathcal Z(\zeta)` 自由度和已声明 symmetry 的集合距离下

\[
d_{\rm obs}(\mathcal R_j^{(1)},\mathcal R_j^{(0)})>\varepsilon_{\rm obs}
\]

时，才称该 cell 在该实验中 operationally distinguishable。`c_j\ne0` 或局部投影非零只能作为
设计 intervention 的直觉条件，不是充分 identifiability theorem。

允许的结论是：在定义的 intervention distribution、观测模型和 relation family 内，law 被
**intervention-identified**。不允许推广为对所有 task/environment 的因果定律或全局唯一解。

## 6. 第一项实验：最小必要性证据

第一项任务固定为 keyed-to-circular phase-switch insertion；其 relation family 与已完成证据为：

- SE(2) basis：`[du,dv,d_axial]`；
- SE(3) basis：`[du,dv,dw,d_roll,d_pitch,d_yaw]`，其中 keyed arm 的 yaw law 是
  `[0,0,0,1,1,0,0]`（完整七 phase），其余五个 generator 为 `[0,0,0,1,1,1,1]`；
- 已有 multi-generator 变体另令 `du` 与 `yaw` 在 clearance 后共同释放，检验 response 是否是
  relevance vector 而不是“单一最相关轴”。

该任务满足以下必要性条件：

1. 同一 task 的两个 preregistered response conditions 共享 nominal region、`T_ref` 与 `B_r`；
2. 对同一非零 `c_j`，其正确 response 不同（一个 phase follow，另一个 phase ignore，或有不同的
   phase-dependent weights）；
3. 固定 TSR 与 whole-frame update 不能同时表达两种正确 response；
4. 对角 `P_r(\zeta)` 能在 held-out intervention 上恢复预定义 response；
5. 将 `\mathcal Z` 放宽到足以吸收差异的消融必须导致该 generator cell 不可观测，而不是伪造
   “已识别”的胜利。

这是“选择性 response 需要 `P_r`”的必要性证据；它不证明真实世界泛化，也不自动证明本仓库
的 G-TSR adapter 正确。后者必须以相同 frozen intervention manifest 的独立集成测试证明。

## 7. 预注册比较与指标

第一版只比较输入/输出可对齐的五类方法：

| 方法 | relation response | admissibility | 用途 |
| --- | --- | --- | --- |
| TSR-Fixed | 无 | 静态 | 下界 |
| TSR-Frame (`P_r=I`) | 全 generator 传播 | 静态/指定 | 检验 whole-frame equivariance 是否足够 |
| Oracle Engineered / Oracle Phase-Adaptive TSR | 人工完整 task law | 人工指定 | 性能上界，不是可学习 baseline |
| Context Regression | 非结构化 context-to-update | 同一观测预算 | 检验结构化低维 law 的价值 |
| G-TSR response law | `P_r(\zeta)` | 独立 `\mathcal Z(\zeta)` | 待验证方法 |

每项实验至少报告：held-out response pose error（或预注册 quotient residual）、region feasibility/
coverage、generator-wise recovery error（仅合成真值可用）、成功率（若有执行器）、训练 interventions
数，以及 fail/abstain cell 数。所有方法共享同一 nominal data、oracle phase 与 relation observations；
Oracle baseline 的人工信息必须单独计入，不可称为公平学习比较。

## 8. 退出门与后续顺序

在 Sprint E（Response Profile Identification）的任何实现或辨识主张前，必须补全一个
task-specific annex，含：

- relation family、`B_r`、单位和 `T_ref^{-1}T` 的 frame convention；
- phase table（事件、`\rho`、annotation source）；
- intervention manifest（幅度、双向对照、随机化、排除与成功判据）；
- `d_obs`、`\varepsilon_obs` 与 quotient/symmetry 定义；
- 正确 response 的独立来源或合成 ground truth；
- 完整的 Claim → Experiment → Baseline → Metric → Failure Criterion 矩阵。

首个 annex 的多数内容已由 `maniskill_spectrum` 冻结。Sprint D 仅需要已锁定的 finite-action
assembly 与其 R-01~R-08 单测；G-TSR response profile fitting 才需要补齐物理 key-clear event 到
`\mathcal Z(\zeta)` 的 yaw quotient/region-domain 语义，以及下列完整辨识合同。

`P_r(\zeta)` 的有限维 profile parameterization、`d_{\rm obs}`、`\varepsilon_{\rm obs}` 与
abstain criterion 不由 Sprint D 冻结。这些属于随后独立的 Sprint E（Response Profile
Identification）进入条件。

满足该 annex 后，顺序固定为：

1. 实现 oracle-phase response-law 的最小离线验证；
2. 通过预注册的表示/辨识门；
3. 再单独 scope-lock online phase observer；
4. 最后才定义 matched relation、`\Gamma` generator correspondence 与 `\varphi` phase
   correspondence，并验证 source-to-target law transfer。

若第一步失败，应报告该 relation family/cell 不可辨识或表示不足；不得用更大的 black-box model、
在线 phase estimator 或 target-side refit 掩盖失败。
