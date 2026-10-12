# G-TSR Sprint H0 — Matched-Relation Transfer Contract

状态：`TRANSFER-CONTRACT-LOCKED`（2026-09-09；数学/API 合同已验收，不等于经验迁移已验证）

验收版本、内容哈希与完整结果见 [H0 acceptance record](../../gtsr_transfer_contract_record.md)。
H0 定向 59 passed；全量 610 passed、1 xfailed、1 warning；独立复审无剩余阻断项。

前置：Sprint D–F.1 已分别建立 6D response-region assembly、oracle-phase response-profile
identification、六生成元复现和 development 级 held-out adequacy。Sprint G 的 causal phase observer
scope/plan 已存在，但论文证据顺序改为先验证 oracle-phase transfer，再用 learned phase 替换 oracle。

本文档只定义第一版可实现、可拒绝的 matched-relation transfer 合同。它不声称 task law 已经跨
任务或跨域稳定，也不授权生成 prospective confirmatory seeds。

## 0. 论文主张与术语账本

唯一候选主张是：

> G-TSR 将任务描述分解为可迁移的 relation-response component 与目标域重新实例化的
> admissibility/execution component。在 generator 和 phase semantics 满足显式匹配合同时，
> source 的对角 response law 可以被搬运到 target，而不在 target 上重新拟合 response law。

本文固定以下术语，后续文档不得混用：

| 术语 | 本文含义 | 明确不是 |
| --- | --- | --- |
| `P_r(zeta)` | generator-coordinate response operator `diag(alpha(zeta))` | 成功概率、region bounds |
| `alpha(zeta)` | `P_r` 的六个对角系数 | policy action |
| `Gamma` | source→target generator-coordinate semantic correspondence | physical frame transform |
| `H` | source→target physical relation-frame transform in `SE(3)` | learned correspondence |
| `phi_{t->s}` | target phase query→source phase query | normalized wall-clock time |
| `Z_t(zeta)` | target-instantiated admissibility domain | 从 source 直接迁移的对象 |
| zero target-side law refit | target 不使用 response interventions 更新 `P_r` | zero calibration / zero target data |

## 1. 唯一研究问题

给定 source response profile、目标任务几何以及由任务规范/CAD/标定提供的 `H`、`Gamma` 和
`phi_{t->s}`，能否在 transfer 前用确定性 gate 验证 matched-family 条件，并在通过时构造：

\[
P_t(\zeta_t)
=
\Gamma P_s\!\left(\varphi_{t\to s}(\zeta_t)\right)\Gamma^{-1},
\]

使该 law 仍属于当前 diagonal response-profile class，并与 `SE(3)` frame transport 一致？

H0 只回答“合同是否数学一致且可实现”。law 是否真的跨 task/domain 稳定由 H1/H2/H3 实验回答。

## 2. 范围

### 必须做

- 满秩六维 generator basis：`B_s,B_t in GL(6)`；
- provided monomial correspondence：`Gamma = D Pi`；
- `SE(3)` Adjoint 与 generator-basis compatibility；
- diagonal-law closure 与 physical-operator equivariance 的 Proposition；
- oracle phase correspondence（有序 phase bijection + phase 内单调 warp）；
- source coordinate range support 与 target-query support gate；
- 结构化拒绝状态和可审计 diagnostic；
- target `Z_t` 与 target geometry 由调用者提供，transfer API 不复制 source `Z_s`。

### 明确不做

- rectangular/rank-deficient `B`、pseudo-inverse 或 subspace operator；
- full `6x6` non-diagonal learned response law；
- automatic `Gamma` learning、learned match classifier 或 black-box compatibility score；
- learned phase observer（Sprint G）或自动 phase-graph discovery；
- `Z_s -> Z_t` transfer、probabilistic G-TSR、uncertainty-driven region inflation；
- policy/trajectory transfer、IK/planner/controller、physical rollout success；
- Sim-to-Real invariance 结论或 confirmatory 数据采集。

## 3. 坐标、frame 与维度合同

### 3.1 Generator coordinates

第一版严格限定：

\[
B_s,B_t\in\mathbb R^{6\times6},\qquad
\det B_s\ne0,\quad\det B_t\ne0,
\]

并沿用仓库 twist 顺序 `[v_x,v_y,v_z,w_x,w_y,w_z]`。generator coordinates 满足：

\[
\xi_s=B_sc_s,\qquad \xi_t=B_tc_t.
\]

该限制与当前 `ResponseOperator` 的 `basis must be a full-rank (invertible) 6x6 matrix` 合同一致。
低维 generator family 不能把 `B^{-1}` 静默替换为 `B^+`；它需要另立 scope。

接入冻结 `ResponseOperator` 时，`c = solve(B, Log(T_ref^-1 T))`；不能把数据文件里的
XYZ/RPY pose6 直接当作 `c`。实现要求数值满秩、运算可表示；不承诺任意病态 GL(6) 的浮点精度。

### 3.2 Physical frame transport

`H in SE(3)` 是 provided source→target relation-frame map。对 twist order `[v,w]`，若

\[
H=\begin{bmatrix}R&p\\0&1\end{bmatrix},
\]

则本文固定：

\[
\operatorname{Ad}_H=
\begin{bmatrix}
R &[p]_\times R\\
0 &R
\end{bmatrix},
\qquad
\xi_t=\operatorname{Ad}_H\xi_s.
\]

group relation/reference 使用同一方向：

\[
T_t=HT_sH^{-1},\qquad T_{\mathrm{ref},t}=HT_{\mathrm{ref},s}H^{-1}.
\]

上式是 H0 明确构造的共轭 relation/reference 配对，不是任意两个物体绝对 world pose 的关系。
`H` 搬运局部位移/响应动作，不自动证明跨对象的语义对应、尺寸对应或 nominal trajectory 对应。
H1 必须另行声明任务局部坐标和 reference 的实例化方式；不得仅凭设置 `H` 宣称 matched target。

任何实现若采用相反的 passive-frame convention，必须另立类型/函数名，不得在本 API 内隐式转置。

### 3.3 Semantic coordinate correspondence

`Gamma` 只允许由 permutation 和非零有限 scales 构造：

\[
c_t=\Gamma c_s,\qquad \Gamma=D\Pi\in\mathcal M_6.
\]

API 约定 `permutation[i]=j`、`scales[i]=d_i` 表示：

\[
(c_t)_i=d_i(c_s)_j.
\]

`D` 的符号/尺度必须来自 task specification、单位换算或 CAD correspondence，禁止通过最小化
compatibility residual 现场学习。对应 target query 必须映回：

\[
c_s=\Gamma^{-1}c_t.
\]

## 4. Matched generator-family gate

### 4.1 Exact compatibility

matched generator family 的核心条件是：

\[
\boxed{B_t\Gamma=\operatorname{Ad}_HB_s.}
\]

它同时约束 generator 的 frame、方向、符号、顺序和尺度。

### 4.2 Dimensionless approximate residual

浮点/标定条件下使用无量纲 residual。令：

\[
S_\xi=\operatorname{diag}(1,1,1,0.03,0.03,0.03),
\]

并由 source intervention manifest 提供严格正的代表幅度：

\[
S_c=\operatorname{diag}(s_1,\ldots,s_6),\qquad s_j>0.
\]

定义：

\[
e_\Gamma=
\frac{
\left\|S_\xi(B_t\Gamma-\operatorname{Ad}_HB_s)S_c\right\|_F
}{
\left\|S_\xi\operatorname{Ad}_HB_sS_c\right\|_F
}.
\]

逐 generator diagnostic 为：

\[
e_{\Gamma,j}=
\frac{
\left\|S_\xi(B_t\Gamma-\operatorname{Ad}_HB_s)S_ce_j\right\|_2
}{
\left\|S_\xi\operatorname{Ad}_HB_sS_ce_j\right\|_2
}.
\]

由于 `B_s`/`Ad_H` 可逆且 `S_c` 严格正，total 与逐列 denominator 均严格非零，不加入会改变尺度
语义的人工 `epsilon`。任务 manifest 必须在查看 transfer 结果前冻结 `epsilon_Gamma_total` 与
`epsilon_Gamma_axis[6]`；只有 total 和六个 axis residual 全部通过才可标记 matched。H0 单元测试
使用显式测试 tolerance，不替真实任务选择阈值。

axis diagnostic 按 **source generator** 索引。浮点 denominator 若为零或非有限，或尺度运算/
最终归一化发生可检测的 underflow/overflow，则抛 `ValueError`，不得通过 epsilon 或无限大
归一化分母制造零 residual。普通舍入误差仍受显式 tolerance 管理，不宣称任意精度算术。
上述单位不变性只指 generator
coordinates 的一致重参数化；物理 `[v,w]` 及 `H` 仍使用固定米/弧度约定。

## 5. Diagonal closure 与 frame equivariance

source law：

\[
P_s(\zeta_s)=\operatorname{diag}(\alpha_s(\zeta_s)).
\]

target law 定义为：

\[
P_t(\zeta_t)=
\Gamma P_s(\varphi_{t\to s}(\zeta_t))\Gamma^{-1}.
\]

因为 `Gamma=D Pi`，diagonal scale 消去，所以：

\[
P_t(\zeta_t)=
\operatorname{diag}
\bigl(
\alpha_{s,\mathrm{permutation}[0]},\ldots,
\alpha_{s,\mathrm{permutation}[5]}
\bigr).
\]

因此第一版 transfer 不需要把 diagonal profile 扩展成 full operator。注意：`D` 虽从 `P_t` 中
消去，但仍作用于 coordinate/support mapping，不能被删除。

仅在 `B_s,B_t in GL(6)` 下定义物理 twist operator：

\[
K_s=B_sP_sB_s^{-1},\qquad K_t=B_tP_tB_t^{-1}.
\]

若 exact compatibility 成立，则：

\[
\boxed{K_t=\operatorname{Ad}_HK_s\operatorname{Ad}_H^{-1}.}
\]

证明：设 `A=Ad_H`，由兼容条件 `B_t=A B_s Gamma^-1`，代入
`B_t (Gamma P_s Gamma^-1) B_t^-1`，相邻的 `Gamma^-1 Gamma` 消去，得到
`A B_s P_s B_s^-1 A^-1`。又由 `Gamma=D Pi` 且两个 diagonal 矩阵可交换，
`Gamma P_s Gamma^-1 = Pi P_s Pi^-1`，所以 target law 仍 diagonal。
该结论针对单个 twist exponential 的 response action，不是任意 ordered PoE chart 的等价定理。
利用 `Exp(Ad_H xi)=H Exp(xi) H^-1`，在同一 Log 分支内及 §3.2 的 reference 配对下，
冻结 `ResponseOperator.action` 满足相同共轭关系；性质测试避开旋转角 π 的分支边界。
approximate residual 通过不等于这个 exact identity 已获证明，也不直接给出 rollout 误差界。

该结论是 frame equivariance，不是 task-law domain invariance；近似 compatibility 时实现只报告
residual，不把等号升级为 theorem。

## 6. Oracle phase correspondence

phase query 写作 `zeta=(q,rho)`。第一版 source/target phase graph 必须是相同节点数的有向链，并
提供 target→source 的语义双射 `phase_map[q_t]=q_s`。该映射必须保持拓扑顺序；禁止任意重排后仍
交给单调 observer。

每个 target phase 提供严格单调的 piecewise-linear warp：

\[
f_q:[0,1]\to[0,1],\qquad f_q(0)=0,\quad f_q(1)=1,
\]

其 knots 在构造时给定并冻结，target/source knot coordinates 均严格递增。于是：

\[
\varphi_{t\to s}(q_t,\rho_t)=
\left(\mathrm{phase\_map}[q_t],f_{q_t}(\rho_t)\right).
\]

H0/H1 的 `zeta_t` 来自 oracle annotation 或独立 instrumentation；它只用于隔离 law transfer，不能
写成可部署 observer。identity warp 是第一候选；event-aligned warp 只能由预先声明的 physical
events 构造，不得根据 transferred-law error 做 DTW 对齐。

## 7. Coordinate support contract

即使 generator semantics matched，也只允许在 source evidence support 内查询。第一版实现一个
manifest-pinned closed box：

\[
\mathcal C_s^{\mathrm{range}}=
\{c_s:\ell_s\le c_s\le u_s\}.
\]

target query `c_t` 先映射为 `c_s=Gamma^{-1}c_t`，再检查 closed bounds。边界 tolerance 只用于
浮点比较，必须是 API 显式参数。

该 box 仅证明 coordinate-range support，**不证明训练分布密度或任意组合的插值可靠性**。H1 若
需要更强的 empirical-support claim，必须另加 convex-hull/kNN density gate；不得把 range support
改名为 distribution support。

## 8. Deterministic match/abstention states

H0 返回结构化状态，而不是布尔值或 silent fallback：

| 状态 | 含义 |
| --- | --- |
| `MATCHED` | 静态检查：语义/basis/phase 通过；逐查询检查另含当前 support 通过 |
| `UNMATCHED_FAMILY` | 格式合法，但 provided generator 或 phase semantic IDs 不匹配 |
| `INCOMPATIBLE_BASIS` | total 或任一 axis compatibility residual 超门值 |
| `OUT_OF_SUPPORT` | `Gamma^{-1}c_t` 超出 source range support |

输入合同错误（shape、非有限值、非双射 permutation、phase graph 非保序等）抛
`TypeError/ValueError`，不伪装成拒绝状态。状态优先级为 semantics → basis → query support。
静态 `MATCHED` 的 `source_coordinates=None`，reason 明示尚未检查 query support。

`PHASE_UNCERTAIN` 属于 Sprint G；`CALIBRATION_UNSAFE` 需要 uncertainty contract，均不在 H0 伪造。
provided semantic labels 触发的拒绝只能称为 **contract-based abstention**，不能称为 learned
negative-transfer detection。

## 9. Target instantiation boundary

H0 transfer API 只产生 target law query `alpha_t(zeta_t)` 和 diagnostic，不接收或复制 source
`GTSR.domain`。后续 target region 必须由调用者提供：

\[
\mathcal Z_t=
f_{\mathrm{fixed}}
(\mathrm{target\ CAD},\mathrm{tolerance},\mathrm{safety\ margin}).
\]

在 H1–H3 比较 transferred/static/target-refit methods 时，所有方法必须共享同一冻结 `Z_t`、target
chart、planner/controller；`Z_t` 不得根据 transfer performance 或 target response interventions
调参。这样才能把差异归因到 response law。

## 10. 实现 API

新模块为 `src/tsr/gtsr/transfer.py`：

```python
class MatchStatus(Enum):
    MATCHED = "matched"
    UNMATCHED_FAMILY = "unmatched_family"
    INCOMPATIBLE_BASIS = "incompatible_basis"
    OUT_OF_SUPPORT = "out_of_support"

@dataclass(frozen=True)
class MonomialCorrespondence:
    permutation: tuple[int, ...]
    scales: tuple[float, ...]
    # matrix, inverse, source_to_target(), target_to_source()

@dataclass(frozen=True)
class GeneratorSemantics:
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    # target_ids[i] must equal source_ids[permutation[i]]

@dataclass(frozen=True)
class PiecewiseLinearPhaseMap:
    source_semantic_ids: tuple[str, ...]
    target_semantic_ids: tuple[str, ...]
    source_phase_for_target: tuple[int, ...]
    target_knots: tuple[tuple[float, ...], ...]
    source_knots: tuple[tuple[float, ...], ...]
    # map(q_target, rho_target) -> (q_source, rho_source)

@dataclass(frozen=True)
class CoordinateRangeSupport:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    tolerance: float

@dataclass(frozen=True)
class MatchResult:
    status: MatchStatus
    total_residual: float
    axis_residuals: tuple[float, ...]
    source_coordinates: tuple[float, ...] | None
    reason: str

@dataclass(frozen=True)
class TransferQueryResult:
    status: MatchStatus
    alpha_target: tuple[float, ...] | None
    source_phase: int | None
    source_progress: float | None
    source_coordinates: tuple[float, ...] | None
    match: MatchResult

@dataclass(frozen=True)
class MatchedRelationTransfer:
    source_basis: tuple[tuple[float, ...], ...]
    target_basis: tuple[tuple[float, ...], ...]
    frame_transform: tuple[tuple[float, ...], ...]
    correspondence: MonomialCorrespondence
    semantics: GeneratorSemantics
    phase_map: PiecewiseLinearPhaseMap
    support: CoordinateRangeSupport
    coordinate_scales: tuple[float, ...]
    total_tolerance: float
    axis_tolerances: tuple[float, ...]
    # validates B_s/B_t/H/Gamma/semantics/phase/support and freezes defensive copies
    def evaluate_contract(self) -> MatchResult: ...
    def evaluate_query(self, target_coordinates) -> MatchResult: ...
    def source_phase(self, target_phase, target_progress): ...
    def query(self, source_profile, target_phase, target_progress,
              target_coordinates) -> TransferQueryResult: ...
```

辅助函数：

```python
def se3_adjoint(H) -> np.ndarray: ...
def compatibility_residual(source_basis, target_basis, H, correspondence,
                           coordinate_scales) -> tuple[float, np.ndarray]: ...
def transport_diagonal(alpha_source, correspondence) -> np.ndarray: ...
def physical_response_operator(basis, alpha) -> np.ndarray: ...
```

所有输入必须检查 shape、finite、`SE(3)`、满秩、permutation bijection、nonzero scales、正
coordinate scales、phase knots 严格单调及 bounds。对象保存 defensive copy；结果不得暴露可写
内部数组。

矩阵构造参数接受 ndarray，内部以嵌套 tuple 保存；matrix/inverse 属性返回新数组。
`query` 仅接受 `ResponseProfile6D`，核对 profile 的 phase count、非空有限 theta、正有限
alpha_max/basis_width；phase index 必须为整数且 progress∈[0,1]，不得隐式取整或外插。
两侧均为同长度有序链时，保序双射在 phase 序号上只能是 identity；允许非线性的 phase 内时间
warp，semantic ID 必须先由调用者规范化为共同任务标签，而非靠名称猜测。

`GeneratorSemantics` 中的 ID 是 task-function semantic ID，不是必须同名的 world axis。例如
source `x` 与 target `y` 可以共享 `lateral_alignment_1`；只有映射后 semantic ID 相同才通过。
phase map 同样保存 source/target phase semantic IDs，并按 `source_phase_for_target` 逐项核对。
`evaluate_contract()` 只判断静态 structure/basis/phase compatibility；`evaluate_query()` 在前者通过
后再检查当前 `Gamma^-1 c_t` 的 support。不得用缺省 `None` 绕过 query support 后仍返回 query-level
`MATCHED`。`query()` 是经过完整 gate 后产生 `TransferQueryResult.alpha_target` 的路径；底层
`transport_diagonal` 仅做代数重排，不能用作已匹配的证书。任一 gate 失败时 `alpha_target=None`，
不得返回退化的 Frame/Fixed law 或继续执行。

## 11. H0 硬门

| Gate | 验证内容 |
| --- | --- |
| H0-01 | monomial 构造、bijection、sign/scale、双向 coordinate roundtrip |
| H0-02 | 随机 diagonal `P` 经 `Gamma P Gamma^-1` 后仍 diagonal，且等于 permutation 后 alpha |
| H0-03 | source/target generator semantic IDs 按 permutation 一一对应；错配返回 `UNMATCHED_FAMILY` |
| H0-04 | canonical identity 与 permutation/sign/scale exact-match 的 `e_Gamma` 近机器零 |
| H0-05 | 含平移 cross-term 的 `SE(3)` Adjoint 与 basis compatibility 方向测试 |
| H0-06 | `K_t = Ad_H K_s Ad_H^-1` 数值误差 `<=1e-10`（仅 GL(6)） |
| H0-07 | normalized total/axis residual 能拒绝单轴错误；一致的 coordinate 单位重参数化不改变 verdict |
| H0-08 | identity 与 piecewise-linear oracle phase map；语义、端点、严格单调、拓扑错误拒绝 |
| H0-09 | `Gamma^-1 c_t` support 边界、符号/尺度映射和 `OUT_OF_SUPPORT` |
| H0-10 | transfer API 不接收/复制 `Z_s`；target law query 不 mutate source profile/contract |
| H0-11 | 全量回归，无 Sprint D–G、geometry 或 legacy TSR 语义回退 |

H0 通过后状态只能是 `TRANSFER-CONTRACT-LOCKED`。它不等于 `TRANSFER-VALIDATED`。

## 12. 后续证据顺序

1. **H1 — Oracle-phase Sim→Sim matched-target transfer**：验证 task-instance transfer；
2. **H2 — Oracle-phase Sim→Real same-instance transfer**：隔离 domain shift；
3. **H3 — Oracle-phase Sim→Real matched-target transfer**：同时验证 task + domain shift；
4. **G — Learned causal phase replacement**：固定 `P/Gamma/phi`，只替换 oracle phase；
5. **H4 — Learned-phase transfer degradation**：报告相对 oracle 的增量损失；
6. **I — Closed-loop execution**：相同 target `Z_t`、IK/planner/controller 下验证执行与成本。

H1–H3 必须严格分离 target geometry/calibration data、target-refit evaluation interventions 和
execution trials。target-refit 数据可构造研究评价上界，但不得进入 transferred law、match gate、
phase map、support threshold 或 `Z_t`。

## 13. 允许与禁止的论文措辞

H0 后允许：

> We define a frame-consistent transfer contract for diagonal response laws under provided one-to-one
> generator and oracle-phase correspondences.

H0 后禁止：

- “G-TSR transfers across domains”；
- “task law is domain invariant”；
- “zero-shot real deployment”；
- “the system automatically detects unmatched tasks”；
- “the entire G-TSR, including `Z`, transfers unchanged”。

只有 H1/H2/H3 的对应证据通过后，才能逐级升级为 task-instance transfer、tested-domain portability
和 combined task/domain transfer。
