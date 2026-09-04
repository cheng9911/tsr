# G-TSR Sprint D — Oracle-Phase Response Operator 实现级设计

状态：`DESIGN-LOCKED`（2026-09-04）

上游：`2026-09-04-gtsr-response-law-scope.md`、
`2026-09-04-gtsr-response-law-evidence-annex.md`

几何前置：`docs/gtsr_freeze_record.md`（冻结，不修改）

## 1. 目标与非目标

Sprint D 将 `maniskill_spectrum` 已验证的 finite relation action 接到 G-TSR，提供一个**纯几何、
oracle-phase** 的 response-region adapter。输入 relation pose 与显式 profile value `alpha`，输出新的
`GTSR`；本 Sprint 不拟合 `alpha`，也不加载 ManiSkill、HDF5 或学习模型。因此它实现的是 response
**operator**，不是 `P_r(\zeta)` 本身；有限维 profile fitting 属于 Sprint E。

这使下面的有限 action 成为本仓库可测试的对象：

\[
A(\zeta,T)=T_{\rm ref}\,
\operatorname{Exp}\!\left(B_rP_r(\zeta)B_r^{-1}
\operatorname{Log}(T_{\rm ref}^{-1}T)\right)T_{\rm ref}^{-1}.
\]

对任意静态 region

\[
\mathcal R_0=\{T_0^w\Phi_{\mathcal C}(z)T_w^e:z\in\mathcal Z\},
\]

response-conditioned region 定义为

\[
\mathcal R_{\rm resp}(\zeta,T)=A(\zeta,T)\mathcal R_0
=\{(A(\zeta,T)T_0^w)\Phi_{\mathcal C}(z)T_w^e:z\in\mathcal Z\}.
\]

因此 action side 固定为**world-frame left action**；local relation coordinates 固定为
`Log(inv(T_ref) @ T)`。这正是 sister repository 的
`C0 @ Exp(P(s) @ Log(inv(C0) @ C)) @ inv(C0) @ X0` 的 `SE(3)` region 版本。

**非目标：** 不改 `GTSR`、`GeneratorChart`、`CoordinateDomain` 或 legacy TSR；不把 `P_r` 训练器、
phase observer、planner、transfer 做进这个 Sprint。

## 2. 新模块与 API

新增 `src/tsr/gtsr/response.py`，并从 `tsr.gtsr` 导出：

```python
class ResponseOperator:
    def __init__(self, reference: ArrayLike, basis: ArrayLike): ...
    @property
    def dim(self) -> int: ...
    def coordinates(self, relation: ArrayLike) -> NDArray[np.float64]: ...
    def action(self, relation: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]: ...
    def condition(self, region: GTSR, relation: ArrayLike, alpha: ArrayLike) -> GTSR: ...
```

`reference=T_ref` 与 `relation=T` 是合法 `SE(3)` transforms。`basis=B_r` 是一个 6×6、可逆的
local twist-coordinate basis（twist order `[v,w]`，与 `metrics.py` 一致）。本 Sprint 故意只接受满秩
square basis；rectangular/weighted pseudo-inverse basis 会受单位选择影响，作为后续扩展，不能悄悄以
`pinv` 进入 v1 合同。

`alpha` 是长度 6 的有限向量。它可由任何上游 oracle profile 或已训练 profile 提供；本类不假定
`alpha∈[0,1]`，但调用方必须遵守其 experiment manifest 的 `alpha_max`。未来的分层是
`alpha = ResponseProfile(\zeta)`，再由 `ResponseOperator.condition(...)` 组装 region。

## 3. 精确行为

1. `coordinates(T)` 返回 `solve(B_r, se3_log(inv(T_ref) @ T))`。
2. `action(T, alpha)` 计算 `xi=B_r @ (alpha * coordinates(T))`，返回
   `T_ref @ se3_exp(xi) @ inv(T_ref)`。
3. `condition(region, T, alpha)` 返回新的
   `GTSR(T0_w=action(T, alpha) @ region.T0_w, Tw_e=region.Tw_e,
   chart=region.chart, domain=region.domain)`。
4. 该方法不 mutate 输入 region。返回的新 region 有更新后的 `T0_w`；`Tw_e`、chart 与 domain
   均保留。对于所有 `z∈Z`，结果必须满足
   `condition(...).to_transform(z) == action(...) @ region.to_transform(z)`。
5. `alpha=0` 时 action 为 identity，conditioned region 的位姿逐点等于原 region；
   `alpha=1` 且 `B_r=I` 时 action 是 `T_ref @ (inv(T_ref) @ T) @ inv(T_ref)`，即 `T @ inv(T_ref)`。

`condition` 不修改 `Z`。特别地，post-clear yaw 的 `alpha_yaw≈0` 与“执行端 yaw 是否可自由”是两个
不同问题：前者来自 response law，后者只能由 caller 提供的 `CoordinateDomain` 表示。

## 4. ManiSkill Spectrum oracle fixture

测试只用 NumPy 构造，不读取相邻仓库数据文件。fixture 固定 identity local basis

\[
B_r=I_6,
\quad \alpha_{\rm keyed}=[1,1,1,1,1,1],
\quad
\alpha_{\rm postclear}=[1,1,1,1,1,0].
\]

relation 取非零 `[du,dv,dw,roll,pitch,yaw]` twist 的 finite `SE(3)` pose；另采用非单位
`T_ref` 与非单位 `region.T0_w,T_w^e`，以排除把 task-local action 错施到 world frame 的假阳性。
这对应 evidence annex 中 six-generator keyed oracle 的 insertion segment；完整 oracle phase profile
仍由上游实验保存。

多 generator fixture 还应检验

\[
\alpha_{\rm multigen\ postclear}=[0,1,1,1,1,0],
\]

即 `du` 与 yaw 同时 release，其他轴继续 follow。

## 5. 测试矩阵与 Sprint D 硬门

| ID | 测试 | 通过条件 |
| --- | --- | --- |
| R-01 | 输入合同 | 非 `SE(3)` transform、非 6×6/奇异 basis、错误形状/非有限 alpha 全部明确拒绝 |
| R-02 | local coordinate/action | 与 `T_ref @ Exp(alpha * Log(inv(T_ref) @ T)) @ inv(T_ref)` 的 pose-level 差 `≤1e-12`（`B=I`） |
| R-03 | full follow | keyed pre-clear `alpha=1` 对每个 generator 逐点等于 whole-frame response |
| R-04 | yaw release | post-clear yaw-only intervention 给 identity action；translation/roll/pitch intervention 仍传播 |
| R-05 | simultaneous release | multigen `du`、yaw 均不传播，其余四轴传播 |
| R-06 | region assembly | 20 个固定 domain samples 满足 `R_resp(z)=A R_0(z)`，误差 `≤1e-12` |
| R-07 | non-mutation/domain separation | 原 region frame/domain 不变；相同 alpha 下改变 domain 不改变 action；改变 alpha 不改变 domain |
| R-08 | unit-consistent non-identity basis | 令 `B=blockdiag(R_t,R_r)`，`R_t,R_r∈SO(3)`；由独立给定的 `c*` 构造 `T=T_ref Exp(Bc*)`，验证 action 等于以 `B(alpha⊙c*)` 显式构造的 expected action，误差 `≤1e-12` |

Sprint D 通过条件：R-01~R-08 全绿，`pytest -q` 全量无回退，且原有 `tests/gtsr/`、
`tests/tsr/` 不修改语义。通过只证明 oracle-profile response-region assembly；不据此声称在
ManiSkill 中重现了 profile fitting 或 physical rollout 成功率。

## 6. 实施顺序

1. 新增 `response.py` 与 R-01/R-02；
2. 实现 `condition()`，完成 R-03~R-07；
3. 加入 non-identity-basis R-08；
4. 全量回归、review 后再冻结 Sprint D；
5. 单独启动 profile fitting adapter（读取已冻结 profile）与 phase observer scope，不与本 Sprint 混合。
