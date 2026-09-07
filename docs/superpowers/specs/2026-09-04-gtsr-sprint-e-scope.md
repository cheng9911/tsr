# G-TSR Sprint E — Response Profile Identification 范围契约

状态：`IDENTIFICATION-LOCKED`（2026-09-07）

前置：Sprint D `ResponseOperator` 已在 `15af5c7` 交付。其职责冻结为

\[
(T,\alpha)\longmapsto A(T,\alpha)\longmapsto\mathcal R_{\rm resp}.
\]

Sprint E 不修改 `response.py`、G-TSR geometry kernel、legacy TSR 或 Sprint D 的 action/assembly
语义。

## 1. 唯一研究问题与交付

给定 frozen oracle phase `\zeta`、已知 relation pose `T`、及受控 relation interventions，能否从
训练 interventions 识别有限维对角 profile

\[
\hat P_r(\zeta)=\operatorname{diag}(\hat\alpha_1(\zeta),\ldots,
\hat\alpha_m(\zeta))
\]

并预测 held-out finite responses？

本 Sprint 的唯一输出是：

\[
\boxed{\hat\Theta,\quad \hat P_r(\zeta),\quad\text{cell status}.}
\]

cell status 必须是下列之一：`UNEXCITED`、`UNOBSERVABLE`、`MODEL_MISMATCH`、`FIT_FAILED`、
`IDENTIFIED`。除 `IDENTIFIED` 外，系统不得将对应 `\hat\alpha_j` 报告为被辨识的 task law。

**不做：** planner/IK 成功率、online phase observer、视觉/力觉输入、跨 task transfer、自然人类
demonstration 的 latent-within-region-coordinate model，或对 `ResponseOperator` 的修改。

## 2. 观察合同

第一版仅使用 sister repository `maniskill_spectrum` 中固定 oracle solver / response-generation
policy 产生的 canonical pose response：

\[
\mathcal D_E=
\{c_i,\zeta_n,X_0(\zeta_n),X_i(\zeta_n),\text{seed}_i,\text{split}_i\}_{i,n},
\]

其中

\[
T_i=T_{\rm ref}\operatorname{Exp}([B_rc_i]^\wedge).
\]

`X_i` 不是任意成功 demonstration。自然示教的
`X_i=A(P_r,c_i)\Phi(z_i)T_w^e` 含有 latent redundancy choice `z_i`；其观测模型不属于 Sprint E。

对拟合 profile 的 pose-level prediction 必须经由冻结 operator：

\[
\hat X_{i,n}=\operatorname{ResponseOperator.action}(T_i,
\hat\alpha(\zeta_n))X_0(\zeta_n).
\]

对应 region prediction 为

\[
\hat{\mathcal R}_{i,n}=
\operatorname{ResponseOperator.condition}(\mathcal R_0(\zeta_n),T_i,
\hat\alpha(\zeta_n)).
\]

E-B 的冻结 fit 本身是 planar `[du,dv,d_{\rm axial}]` profile。为检验它与 Sprint-D 的 6D operator
接口一致，唯一允许的 embedding 是

\[
[du,dv,d_{\rm axial}]\mapsto[du,dv,0,0,0,d_{\rm axial}],
\]

以及

\[
[\alpha_{du},\alpha_{dv},\alpha_{yaw}]
\mapsto[\alpha_{du},\alpha_{dv},1,1,1,\alpha_{yaw}].
\]

后三个 coefficient 对零 relation coordinates 无影响。该 embedding 只验证 planar profile 到 6D
assembly 的接口，绝不把 E-B 误报为 six-generator profile identification。

## 3. Oracle phase 合同

登记完整 phase vocabulary：

\[
(\mathrm{reach},\mathrm{grasp},\mathrm{lift},\mathrm{align\_keyed},
\mathrm{enter\_key},\mathrm{unlock\_yaw},\mathrm{circular\_insert}).
\]

首轮拟合只使用 codes 3--6：`align_keyed`、`enter_key`、`unlock_yaw`、`circular_insert`；前三个
phase 只登记而不参与 fitting。原始 source 是
`../maniskill_spectrum/phase_switch_symmetry/collect_phase_switch_rollouts.py::PHASES`。

每个拟合 phase 有 25 个 oracle-resampled samples。其 frozen mapping 是

\[
g(k,\rho)=\frac{q(k)+\rho}{4},\qquad
q(\mathrm{align\_keyed},\mathrm{enter\_key},\mathrm{unlock\_yaw},
\mathrm{circular\_insert})=(0,1,2,3),
\]

其中 `\rho\in[0,1]` 与 25-bin interpolation 复用
`benchmark_phase_switch_baselines.py::progress_grid` / `phase_task_curve`；不得重新以 wall-clock time
定义 phase progress。physical key-clear event 保留为 post-clear quotient 与后续 non-oracle phase
evaluation 的事件 source，不能替代上述 frozen fitting alignment。

## 4. 有限维 profile 合同

第一版必须逐项复用 frozen Pdiag-finite parameterization，而不是另换 spline/RBF：

\[
\hat\alpha_j(\zeta)=\alpha_{\max}\,
\sigma\!\left(\Phi(g(\zeta))^\top\theta_j\right),
\qquad \alpha_{\max}=1.25.
\]

`\Phi` 为 24 个中心均匀分布在 `[0,1]` 的 normalized Gaussian RBF，width `0.065`；训练有
second-difference profile smoothness weight `0.1`、three alternating nominal/profile updates。优化器是
`scipy.optimize.least_squares(method="trf", max_nfev=500, ftol=xtol=gtol=1e-9)`；初始化来自
pointwise diagonal fit 并经 logit transform。权威代码为
`../maniskill_spectrum/phase_switch_symmetry/phase_switch_baselines.py::SmoothFinitePDiagModel`，
configuration 为 `../maniskill_spectrum/phase_switch_symmetry_multiseed/experiment.json`。

`ResponseProfile` 是 `\zeta\mapsto\alpha` 的可序列化只读对象；`ProfileIdentifier` 是
`\mathcal D_E\mapsto\hat\Theta` 的拟合对象。两者与 Sprint-D `ResponseOperator` 必须是分离模块。

## 5. 干预 manifest 与 split

E-B（冻结 profile replication）复用 planar phase-switch evidence 的完整冻结 split：每个 execution
seed 有 30 个 successful mixed training interventions、8 个 nonzero isolated held-out interventions，
zero intervention 只用于 empirical response target / nominal reference，绝不用于 fitting。固定 seeds：

\[
(20260818,20270818,20280818,20290818,20300818).
\]

context manifest SHA-256 为
`3751db801cf49ce698c9b5a74ce36cf0b39897d967ddf2959846a3965b8bd653`。isolated test amplitudes固定为
axial yaw `±15°/±30°` 与 task-local `du,dv=±0.015 m`；mixed amplitudes 必须从
`phase_switch_symmetry_multiseed/fixed_contexts.json` 原样读取，禁止重采样或重选。每条记录必须保存：
intervention ID、split、seed、`c`、active generator、sign、magnitude、success、exclude reason 与 source
record。

该五-seed 30/8 contract 是 planar `[du,dv,d_{\rm axial}]` profile evidence，**不是** six-generator
SE(3) dataset 的替代物。6D profile replication 必须另行采用其自身已冻结的 60-mixed/14-isolated、
three-seed manifest，不能混写为本 Sprint 的 primary E-B result。

## 6. 可观测性合同与两个证据层

### E-A：region-level observable / unobservable control

E-A 只使用具有解析可观测性证书的合成 G-TSR regions：

- narrow-domain follow/ignore pair：region discrepancy 严格大于 calibration floor；
- wide-domain symmetry control：response action 保持 region set 不变，严格等于零；
- 同一 relation intervention、nominal region 与 basis 下，只改变 response condition。

仓库的 `SE3Metric` 是 principal-log residual objective，不满足严格 triangle inequality。因此由

\[
D_W(A,B)=\max\left\{
\sup_{x\in A}\inf_{y\in B}\delta_W(x,y),
\sup_{y\in B}\inf_{x\in A}\delta_W(y,x)
\right\}
\]

得到的量在 Sprint E 中命名为 **symmetric residual-set discrepancy**，不得称为 Hausdorff metric/distance。
非凸通用 `GTSR.project()` 只给 local candidate，不能承担 `\inf` 的全局证书；故 E-A 必须选择可解析
求值的 region/domain，不允许以 multi-start projection 结果伪造全局 observability proof。

`\varepsilon_{\rm obs}` 在任何 held-out evaluation 前由 calibration/control split 冻结：

\[
\varepsilon_{\rm obs}=\max\{\varepsilon_{\rm num},
Q_{0.99}(D_W(\mathcal R^{\rm control,a},\mathcal R^{\rm control,b}))\}.
\]

若仿真 control 是确定的，第二项退化为零，仅保留数值 floor。后 clear yaw 若被 `\mathcal Z_{\rm yaw}`
完全吸收，应得到 `UNOBSERVABLE`；trajectory-policy 拟合出 yaw coefficient 接近零不能取代该 region-set
结论。

### E-B：frozen solver-trajectory-policy profile replication

E-B 复现既有 Pdiag finite fitting 与预注册结果，证明 profile 对 canonical solver response 的 held-out
prediction，不把该结果升级为 task-region observability theorem。其原始门为：final translation relevance
`>0.90`、pre-clear yaw relevance `>0.90`、`|final yaw relevance|<0.10`、half-decay error `<0.03`，以及
每个 seed 的 Pdiag task error 小于 phase-scalar baseline。

## 7. Cell-status 判定顺序

对 `(j,k)`（或预注册的 `(j,k,\rho\text{-bin})`）按以下顺序判定：

1. **excitation：** 未满足正/负或等价对照覆盖，标 `UNEXCITED`；
2. **observability：** E-A 的解析 discrepancy 不超过 `\varepsilon_{\rm obs}`，标 `UNOBSERVABLE`；
3. **fit：** 数值求解失败，标 `FIT_FAILED`；
4. **model adequacy：** 可观测、已激发但 diagonal finite profile 未达预注册 held-out 门，标
   `MODEL_MISMATCH`；
5. 仅其余 cell 标 `IDENTIFIED`，允许报告 `\hat\alpha_j`。

E-B 的 solver-trajectory cells 与 E-A 的 region-set cells 必须在报告中分别标注，不能用前者覆盖后者
的 `UNOBSERVABLE` verdict。

## 8. 基线、指标与硬门

学习基线：TSR-Fixed (`P=0`)、TSR-Frame (`P=I`)、Phase-Scalar (`P=w(\zeta)I`)、Context Regression
（`(c,\zeta)\mapsto\xi_{\rm update}`）、G-TSR Pdiag 与 Full Operator (`u=M(\zeta)c`)。Pdiag/scalar/fixed/
frame 通过 `ResponseOperator` 得到 action；Context/Full 直接给出 task-local update `\xi_{\rm update}`，再
以同一冻结 left-action rule

\[
A=T_{\rm ref}\operatorname{Exp}([\xi_{\rm update}]^\wedge)T_{\rm ref}^{-1},
\qquad \mathcal R_{\rm resp}=A\mathcal R_0
\]

组装 region。这样所有方法共享 action-to-region 语义，却不把非对角 baseline 错标为
`ResponseOperator` 的对角 law。Oracle Engineered TSR 只作人工上界，不作为公平学习 baseline。

| Gate | 内容 | 失败语义 |
| --- | --- | --- |
| E-01 | dataset / seed / split integrity | 数据 hash 或 manifest 不符，停止 |
| E-02 | oracle phase mapping `g(k,\rho)` | 与 frozen 25-bin contract 不符，停止 |
| E-03 | intervention excitation | cell `UNEXCITED` |
| E-04 | E-A analytic observability | cell `UNOBSERVABLE` |
| E-05 | frozen Pdiag fitting replication | 原预注册门未通过 |
| E-06 | profile → Sprint-D action differential | pose mismatch 超数值 tolerance |
| E-07 | held-out finite response | 未满足预注册 baseline criterion |
| E-08 | wide-`\mathcal Z` negative control | 错误输出 identified 则失败 |
| E-09 | Pdiag / scalar / context / full 对比 | 如实报告结构 claim 是否成立 |
| E-10 | full regression | 冻结 geometry/legacy 语义回退 |

## 9. 退出条件与后续

Sprint E 通过只允许宣称：在所定义的 oracle solver policy、冻结 intervention distribution 与 cell-status
合同内，有限维 diagonal response profile 可被识别或被正确 abstain。它不允许宣称 online phase、
real-demonstration identifiability、全局 region observability 或 transfer。

完成 E-A/E-B 并通过 E-01--E-10 后，才可将本文件升级为 `IDENTIFICATION-LOCKED`；随后分别 scope-lock
6D profile replication、online phase observer 与 matched-relation transfer。

### 2026-09-07 冻结记录

- E-01--E-08：`tests/gtsr/test_sprint_e_replication.py` 与既有 profile/observability contracts；
- E-09：`tests/gtsr/test_sprint_e09_baselines.py`，六方法 × 五 execution seeds，无 skip；
- Source Data：`docs/source_data/gtsr_e09_results.json`，保存逐 condition errors/status/budget 与
  structured profiles，并由 E-09 gate 反向校验；
- E-10：全量 `pytest -q` 为 512 passed、1 expected xfail、1 个既有 SciPy RuntimeWarning；
- 独立 final review：APPROVE，无 Critical/Important/Minor 遗留。

该锁定只覆盖本文定义的 planar oracle-phase identification contract；后续 6D、online phase 与 transfer
仍须分别 scope-lock，不因本状态自动解锁论文主张。
