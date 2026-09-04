# G-TSR 响应规律阶段 — ManiSkill Spectrum 证据附录

状态：`EVIDENCE-LOCKED`（2026-09-04）

本附录登记相邻仓库 `../maniskill_spectrum` 的既有结果，作为本仓库 response-law 阶段的
task-specific evidence source。它不是 `tsr` 的运行时依赖，也不替代在 G-TSR assembly 上应做的
独立集成验证。

## 1. 任务与已知真值

主任务是 `KeyedCircularPhaseSwitchSE3-v1`：矩形 key 首先穿过 keyed gate，随后进入允许轴向
yaw 的 circular bore。relation intervention 在 socket-local frame 中给出：

\[
c=[du,dv,dw,d_{\rm roll},d_{\rm pitch},d_{\rm yaw}]^\top.
\]

完整 solver phase 为

\[
(\mathrm{reach},\mathrm{grasp},\mathrm{lift},\mathrm{align\_keyed},
\mathrm{enter\_key},\mathrm{unlock\_yaw},\mathrm{circular\_insert}).
\]

已有 SE(3) keyed arm 的 model-independent oracle 为：

\[
\alpha_{du,dv,dw,roll,pitch}=[0,0,0,1,1,1,1],\qquad
\alpha_{yaw}=[0,0,0,1,1,0,0].
\]

它提供了同一 nominal insertion、同一 intervention basis 下“所有轴 follow”与“仅 yaw 在 clearance
后 release”并存的选择性 response。对应任务几何、frame 与 intervention 代码记录在
`../maniskill_spectrum/phase_switch_symmetry/phase_switch_symmetry_env.py`；冻结说明见
`../maniskill_spectrum/phase_switch_symmetry_multiseed/VALIDATION_se3_six_generator.md`。

## 2. 已有辨识证据

平面 keyed-to-circular task 的 Pdiag-finite law 使用

\[
\alpha(s)=1.25\,\sigma(\Phi(s)^\top\theta),\qquad
X_{\rm pred}(s)=C_0\operatorname{Exp}(P(s)\operatorname{Log}(C_0^{-1}C))
C_0^{-1}X_0(s).
\]

其训练使用 30 个 mixed interventions；测试保留 8 个非零 isolated interventions；oracle solver
phase 在四个 insertion phases 内每 phase 取 25 个样本。固定五 execution seeds
`20260818, 20270818, 20280818, 20290818, 20300818` 的 preregistered 条件全部通过：

- final translation relevance `>0.90`；
- pre-clear yaw relevance `>0.90`；
- final yaw relevance 的绝对值 `<0.10`；
- response half-decay timing error `<0.03` progress；
- Pdiag task error 在每个 seed 均低于 shared-scalar frame baseline。

记录的 scalar-minus-Pdiag mean improvement 为 `1.7576 mm-equiv`，hierarchical bootstrap interval
为 `[1.2964, 2.2295] mm-equiv`。这证明 shared scalar `w(s)I` 对该 task 的 generator selectivity
结构过粗；它不声称 Pdiag 在所有指标上压倒 Generic RBF 或 Full operator。

权威冻结记录：

- `../maniskill_spectrum/phase_switch_symmetry_multiseed/VALIDATION.md`；
- `../maniskill_spectrum/phase_switch_symmetry_multiseed/results/multiseed_summary.json`；
- `../maniskill_spectrum/phase_switch_symmetry_baselines/BASELINE_VALIDATION.md`。

## 3. 可识别性与 phase 的边界

已完成的 few-shot sweep 表明 Pdiag 在该冻结 distribution 内有低样本优势，但 simple context
condition number 与所测试的 parameter-space Gauss--Newton spectrum 都不能可靠排序 subset quality。
因此它支持“存在可识别的 structured law”，不支持以一个通用 spectral threshold 判定所有
intervention designs 的可识别性。

同一仓库也测试了非 oracle progress parameterizations。time 明显退化；keypoint、arc 与
DTW-position 各有不同的准确性/部署性权衡。DTW-position 是离线对齐上界，不可报告为在线 phase
estimator。故 G-TSR 的第一实现保持 oracle phase，在线 observer 另立 scope。

权威冻结记录：

- `../maniskill_spectrum/phase_switch_symmetry_multiseed/VALIDATION_identifiability_tpgmm_nonoracle.md`；
- `../maniskill_spectrum/phase_switch_symmetry_multiseed/nonoracle_progress/nonoracle_progress_summary.json`。

## 4. 本仓库的集成要求

进入实现级 design 前，必须将上述二维/三维 finite action 翻译为明确的 G-TSR region assembly，
并新增独立测试：

1. 用一致的 local generator convention，把 frozen relation `c` 映射到 `SE(3)` transform；
2. 对 pre-clear phase 验证 yaw intervention 导致 keyed G-TSR center/region 随 yaw response；
3. 对 post-clear phase 验证 yaw response 被抑制，而 translation response 保留；
4. 验证 `\mathcal Z(\zeta)` 的 yaw freedom 与 `P_r(\zeta)` 的 yaw response 不被混为同一对象；
5. 对 multi-generator case 验证 `du`、yaw 可共同 release，其他四轴不被错误释放；
6. 与 frozen `maniskill_spectrum` oracle action 做 pose-level differential comparison，且明确
   两边的 frame/action convention。

这些是 G-TSR integration 的新证据；不能仅引用相邻仓库的已通过结果来代替。
