# G-TSR Sprint G — Causal Online Phase Observer 范围契约

状态：`SCOPE-DRAFT`（2026-09-08）

前置：Sprint F 已 `REPLICATION-LOCKED`，在 oracle phase 下冻结 6D response-profile 拟合与
selectivity 复现。本 Sprint 只解除这一个 oracle 假设：从运行时可观测、因果前缀信号中输出
与 frozen profile 兼容的 phase/progress 估计。它不修改 G-TSR 几何内核、`ResponseOperator`、
`ProfileIdentifier6D` 或 Sprint F 的 oracle-profile 参数。

## 1. 唯一研究问题

对时刻 `t` 的因果观测前缀 `o_{0:t}`，是否能在不读取 solver phase、未来帧或任务结果的
情况下估计

\[
\hat\zeta_t=(\hat q_t,\hat\rho_t),\qquad
\hat q_t\in\{0,1,2,3\},\quad \hat\rho_t\in[0,1],
\qquad \hat g_t=\frac{\hat q_t+\hat\rho_t}{4},
\]

并且将 `\hat g_t` 代入冻结 response profile 后，仍保留 oracle-phase 方法的主要下游性能？

本 Sprint 的交付是：

\[
\boxed{
o_{0:t}\mapsto
(\hat q_t,\hat\rho_t,\hat g_t,\text{confidence}_t,\text{abstain}_t),
\quad
\hat P_r(\hat g_t)
}
\]

这是 **phase/progress observer**，不是新的 response-law fitter。本 Sprint 首先验证 replay-time
causal deployment contract；它不宣称已经在反馈控制中改变机器人动作。

## 2. 范围边界

### 必须做

- 显式、有状态的 streaming API，每次只接收当前观测并更新估计；
- discrete phase `q` 和 within-phase progress `rho` 的联合输出，以及标准化进度 `g`；
- confidence 与 abstention，不得在不确定时伪造高置信 phase；
- episode/seed-grouped 评估，禁止帧级随机分割；
- 与 frozen oracle phase、时间基线及非因果离线上界的分层对照；
- 在物理 timeline 上评估，不在各自参数化后的 grid 上自我对齐评分；
- 将 `\hat g_t` 代入已冻结 profile，评估下游 pose/action 误差和 switch timing。

### 明确不做

- matched-relation transfer、`\Gamma` generator correspondence 或 `\varphi` phase correspondence；
- target-side profile refit、新的 6D response law 训练或 Sprint F 阈值调整；
- planner/IK 重规划、policy learning、真机或 sim-to-real 主张；
- 使用完整轨迹的 DTW 作为可部署 observer；
- 把回放评估写成 closed-loop task-success 结果。

## 3. Phase 语义与标签

oracle 只用作监督/评估标签，不得出现在部署输入中：

| `q` | solver label（仅监督） | 任务语义 |
| ---: | ---: | --- |
| 0 | 3 | align keyed relation |
| 1 | 4 | enter key |
| 2 | 5 | unlock/release yaw |
| 3 | 6 | circular insertion |

`rho` 是同一 oracle phase 内的归一化进度，只用于训练监督和评估。物理 key-clear
event 独立由真值几何事件定义，用于对齐评分；不能用 observer 自己预测的 switch
来移动评分边界。

## 4. 在线观测合同

默认输入只由运行时可获得量构成：

- robot joint position/velocity：`qpos_t`、`qvel_t`；
- end-effector pose：`tcp_pose_t`；
- task relation poses：`peg_pose_t`、`socket_pose_t`，并在 task-local frame 中构造
  `T_socket^{-1}T_peg`；
- 上一控制周期已经执行的 action `a_{t-1}` 及更早历史（可选）；
- 实际系统可测的 force/torque 通道；当前冻结数据只有 `contact_force`，因此使用时必须
  单独做“含力/不含力”消融。

禁止输入：

- `solver_phase`、oracle `q/rho/g`、未来 observation 或 bidirectional smoothing；
- `goal_pose`、`keyed_entry_goal_pose`、`circular_transition_goal_pose`；
- `key_clearance_margin`、`obj_to_goal_dist`、`yaw_err`、`axis_angle_err`；
- `success`、`terminated`、`truncated`、reward 或 episode 最终长度；
- condition ID、execution seed、arm 名称或可反推标签的元数据。

`causal_delta` 是已知任务 context，但 primary observer **不输入**它；它只可作为消融项，
检查 phase 估计是否偶然依赖 intervention 标识。

## 5. Causal streaming API

建议新模块 `src/tsr/gtsr/phase.py`：

```python
@dataclass(frozen=True)
class PhaseEstimate:
    phase: int
    progress: float
    normalized_progress: float
    confidence: float
    abstained: bool

class OnlinePhaseObserver:
    def reset(self) -> None: ...
    def update(
        self,
        observation_t: PhaseObservation,
        previous_action: np.ndarray | None,
    ) -> PhaseEstimate: ...
```

契约：

- 时间轴固定为：执行 `a_{t-1}` 后获得 state observation `o_t`，再调用
  `update(o_t, a_{t-1})` 输出 `\hat\zeta_t`，该输出只能影响随后的 `a_t`；
- `update(o_t, a_{t-1})` 的输出只取决于 `o_0,...,o_t`、`a_0,...,a_{t-1}` 和已冻结
  模型参数；禁止输入与 `o_{t+1}` 对齐的 `a_t`；
- `normalized_progress∈[0,1]`，有效输出须满足 `g_t≥g_{t-1}`；
- 默认 phase 不回退；若任务允许回退，必须在新 scope 中改变状态机；
- 低置信时输出 `abstained=True`，下游保持上一个已确认 phase/progress，不越级切换；
- 数组输入执行 shape、finite 与单位验证；输出是不可变快照。

## 6. 数据、拆分与独立性

### 6.0 数据角色

Sprint G 明确区分三个数据层级，避免把开发证据冒充 confirmatory 证据：

| 层级 | 数据 | 独立单位 | 角色 |
| --- | --- | --- | --- |
| development set | 冻结 6D SE(3) seeds `{20260818, 20270818, 20280818}` | seed（n=3） | 开发、复现、审查与 LOSO 内折评估；不得称为未见过的 outer test |
| external prior | planar 5-seed non-oracle progress（§6.3） | seed（n=5） | 范围先验与基线边界，冻结于 scope-lock |
| prospective confirmatory set | 尚未收集的新 execution seeds | seed | 一次性解封的外部确认；见 §6.4 |

三个层级不可互相替代。LOSO 在 development set 上得到的是**开发级证据**，用于冻结 observer、
特征、归一化、阈值与统计方案；confirmatory 结论只能来自 §6.4 的新 seeds，不能用 episode/frame
数量伪装独立 seed 数量。

### 6.1 Primary：冻结 6D SE(3) 轨迹（development set）

使用 Sprint F 已校验的 keyed/circular_honest rollouts，execution seeds
`{20260818, 20270818, 20280818}`。数据完整性、manifest hashes 和 usable 语义直接复用
F-01，不重新选 episode。

评估必须使用 leave-one-execution-seed-out（LOSO）：

- observer 的所有 learned parameters、normalization、阈值和 calibration 只从两个训练 seed 得到；
- 整条 episode 是最小拆分单位，同一 episode 的帧不得跨 train/test；
- 三个折各自保留一个完整 seed，最终以 seed 为独立单位报告 `n=3`；
- keyed/circular_honest 均必须报告，不得只选更容易的 arm。

此 LOSO 是 development-set 内的**开发级评估**：它的输出用于冻结 observer、阈值与统计方案，
本身不构成 confirmatory 结论（见 §6.4）。

### 6.2 Fold-specific frozen response artifacts

Sprint F 生成的 36 个 cell profile 是复现计算单元，不是唯一的部署 artifact。Sprint G
为每个 outer LOSO fold 建立一个明确的 training-only response artifact：

- 只使用该 fold 的两个训练 execution seeds，按 Sprint F 冻结的 N=30、Pdiag-finite、
  optimizer 和单位合同拟合；
- keyed/circular_honest 分开保存，并固定 train-seed IDs、subset IDs、数据 hashes、
  parameters 和 artifact SHA-256；
- 在解封该 fold 的 held-out seed 前完成拟合并只读冻结；
- G-05 中“不重拟合”的精确含义是：评估 held-out seed 时不使用其轨迹、标签或误差
  更新 observer 或 response artifact。

这些 fold-specific artifacts 还需要真实 held-out adequacy 检查；该检查由 Sprint F.1
held-out adequacy 附录（`2026-09-08-gtsr-sprint-f1-adequacy-annex.md`）在 development set 上
收口，Sprint F 中由常量 true 生成的 `CellStatus.IDENTIFIED` 不能作为该检查的证据。

### 6.3 Supporting evidence：planar non-oracle progress

sister repository 已有五 seed 的 non-oracle progress 实验，用作 scope 先验和基线边界：

| 产物 | SHA-256 |
| --- | --- |
| `phase_switch_symmetry/analyze_nonoracle_progress.py` | `f722116de2603b839034b1113de906e06541ae4ee2a1bd0eced026a6ad0d1e6b` |
| `nonoracle_progress_summary.json` | `7c3878f650fd330e5644b034cf4a34a6d8a27e400503f8f714544404d6a3a69f` |

已冻结的五 seed 均值（mm-equivalent task error）：oracle `4.603`，time `11.108`，
arc-length `5.500`，keypoint `8.894`，DTW-position `3.188`。DTW-position 使用整条轨迹，
只是离线 optimistic upper bound；它不是 OnlinePhaseObserver 的候选实现。

这份证据只说明 response law 不是 solver 语义标签的人工产物，不等于已经完成
6D causal online observer。

### 6.4 Prospective confirmatory set（预注册确认集）

Sprint G 的 confirmatory 结论需要新的 execution seeds，它们在以下事项全部冻结并一次性解封
之前**不生成**：

- observer 结构与算法/超参数；
- 输入 feature manifest 与禁用字段；
- 数据归一化方案；
- confidence calibration 规则；
- abstention threshold；
- phase/progress acceptance gates 的全部阈值；
- 下游 non-inferiority budget；
- 排除与失败规则。

统计独立单位是 execution seed。exact sign-flip 的离散分辨率（G-00 §5）把最低样本量抬到
`n≥9`（双侧）或 `n≥8`（两个 one-sided 的 TOST）；最终 confirmatory seed 数由 G-00 §11 预先
冻结的功效/精度模拟确定，并在解封前预注册，且设计下限冻结为 `n≥9`。不得用 episode 或 frame
数量替代独立 seed 数。

## 7. 对照组

| 方法 | 因果/可部署 | 角色 |
| --- | --- | --- |
| Oracle phase | 否（privileged） | 参考上界，不与 learned observer 混为同类方法 |
| Normalized time | 是 | 无几何信息的 naive baseline |
| Causal SE(3) arc length | 是 | 可观测几何进度 baseline |
| Causal task keypoint/state machine | 是 | 工程规则 baseline |
| Offline DTW-position | 否（未来泄漏） | optimistic alignment upper bound，只报告不部署 |
| Proposed observer | 是 | 需评估的 streaming method |

不强制本 Sprint 使用深度网络。在只有 3 个独立 6D seeds 的条件下，优先选择低容量、
可校准的 ordered-state observer；如果高容量模型没有额外独立训练数据，不得用 frame count
伪装样本量。

## 8. 评估指标

所有 phase 指标都在原始 physical timeline 上计算，并先按 episode，再按 execution seed
聚合。

### 8.1 Observer 指标

- phase macro-F1 和 balanced accuracy；
- ordered phase error `|q_hat-q|`，以及 `|q_hat-q|≤1` 的覆盖率；
- normalized-progress MAE `|g_hat-g|`；
- q=1→2 事件的 signed latency（control steps 及相对 physical key-clear 的 mm offset）；
- backward-transition count、skipped-phase count、abstention coverage 和 selective risk；
- per-arm/per-seed 结果，不只报告 pooled-frame 平均。

对 abstention 的计分固定为：

- phase macro-F1/balanced accuracy 中，abstained frame 按错误预测计，不从分母删除；
- progress MAE 和所有下游指标使用真正部署语义：abstain 时持有上一个已确认估计，
  因此每一帧都进入误差；
- selective risk 可在 non-abstained frames 上条件报告，但必须与 coverage 成对展示，
  不作为唯一通过门。

### 8.2 下游保真指标

保持 frozen response profile 参数不变，只把 `g_t` 从 oracle 换成 `g_hat_t`：

- `alpha` distortion：`||P_r(g_hat_t)-P_r(g_t)||`；
- physical-timeline response/task error，同 Sprint E 的 mm-equivalent metric；
- physical key-clear 之前和之后分层的 yaw-action error；
- learned-phase 相对 oracle-phase 的 paired degradation，以及相对 time baseline 的 paired gain。

任务 quotient 仍绑定于真实 physical key-clear event，不绑定于 predicted q=2，否则 observer 可以
通过移动评分边界虚假降低误差。

## 9. 硬门

- **G-00 证据入场门**：实施计划前冻结 primary observer 的 feature manifest、标签映射、
  算法/超参数、fold-specific response-artifact 生成规则、外部 acceptance thresholds 和 sister
  evidence hashes。外部阈值只能来自预先冻结的 planar evidence 或独立 development data，不得
  查看任一 6D outer-test seed 后调整。
- **G-01 无泄漏/因果性**：禁止字段扫描全通过；对任意两条共享前缀，输出在该前缀
  的每个时刻逐位一致；改变未来帧不得改变当前输出。
- **G-02 状态机合同**：所有非 abstain 输出均满足 phase/range/finite 合同；
  `g_hat_t` 单调，无 backward 和 skipped transition；`reset()` 不携带上一 episode 状态。
- **G-03 分割与复现**：LOSO 三折的 preprocessing/model/calibration 仅用训练 seeds；
  confidence calibration 使用预先固定的 fold-local 规则（在两个训练 seed 之间做 inner
  leave-one-seed-out，然后用全部两个训练 seed 重拟合）；相同 seed 与配置重跑逐 episode
  输出一致；帧级随机 split 必须由测试显式拒绝。
- **G-04 observer 有效性**：primary observer 的预注册 phase/progress/latency/abstention 阈值全部
  通过，且在每个独立 seed 上的 physical-timeline task error 均优于 normalized-time
  baseline。具体阈值由 G-00 冻结，不在 implementation plan 中改动。
- **G-05 下游保真**：评估时不重拟合 §6.2 的 fold-specific frozen profile；learned-phase 的
  response/task error 相对
  oracle-phase 的 paired degradation 不超过 G-00 预注册预算，并保持 key-clear 前后 yaw
  action 的正确切换方向。
- **G-06 失败可见**：低置信、越分布或观测缺失时必须 abstain；禁止静默回退到
  oracle label、episode length 或未来对齐。
- **G-07 回归门**：Sprint F F-01–F-06、planar identification/baselines、`tests/tsr/` 和几何
  内核全部无回退；Ruff 与 `git diff --check` 干净。

G-04/G-05 暂不写入未经证据支持的绝对数字。这不是放宽验收：G-00 要求在编写
implementation plan 之前，只使用冻结 planar evidence 或额外独立 development data 预注册数字。
现有三个 6D seeds `{20260818, 20270818, 20280818}` 已用于 Sprint F 开发、复现与审查，因此是
**development set**（§6.0），其 LOSO 内折评估是开发级证据，不能称为未见过的 outer test；
confirmatory 结论必须来自 §6.4 的新 seeds。

## 10. 论文主张合同

Sprint G 通过后只允许声称：

> 在冻结 task family 的回放轨迹上，一个只使用因果可观测前缀的 streaming observer
> 能在独立 execution seeds 上估计 phase/progress，并在不重拟合 frozen response law
> 的情况下保留预注册的下游性能。

禁止升级为：

- 真正 closed-loop controller 已被验证；
- observer 对新 task/object/environment 普遍泛化；
- phase 与 response law 已经完成跨 relation transfer；
- 回放成功率等同于物理任务成功率。

## 11. 退出条件与后续

G-00–G-07 全部通过后，本文件才可升级为 `PHASE-OBSERVER-LOCKED`。之后进入
matched-relation transfer 的 scope-lock，显式定义 generator correspondence `\Gamma` 和 phase
correspondence `\varphi`。

如果 observer 直接指标达标但 G-05 失败，应报告 phase 误差对 response law 过于敏感；
不得通过 target-side refit 或改变 Sprint F profile 来掩盖。如果只有 offline DTW 通过，则结论仍是
“存在可观测进度参数化”，不能声称“online phase observer 成立”。
