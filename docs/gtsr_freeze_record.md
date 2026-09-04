<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2025 Siddhartha Srinivasa -->

# G-TSR 几何内核冻结记录

状态：`FROZEN`（2026-09-04 审查修订已复核）
日期：`2026-09-04`
上游：`docs/gtsr_geometry_validation_plan.md`（§9~§12 硬门 G1/G2/G3）

- 被审 revision：`45786b9`（含前置 G2 收口 `eae1ace`）
- 修订 revision：`3c88724`（硬门审查修补实现基线）

本文件归档 G-TSR 几何内核（Sprint A/B/C）的原冻结基线与 2026-09-04 合同修订：
API、约定、残差与容差、固定 seeds、环境版本和三道硬门结果。重新冻结后，这些契约**不得**在不记录
根因/量纲/对命题影响的前提下放宽（上游 §10）；旧 `TSR` 与 `tests/tsr/` 作为 differential oracle
保持不动。

## 1. 冻结模块与 API

| 模块 | 公共接口 | 冻结语义 |
|---|---|---|
| `gtsr/metrics.py` | `skew`、`se3_hat`、`se3_vee`、`so3_log`、`se3_exp`、`se3_log`、`se3_adjoint`、`se3_left_jacobian_inverse` | 纯函数，twist 排列 `ξ=[v,ω]`（`v` 在前） |
| `gtsr/metrics.py` | `SE3Metric(rotation_weight=1.0)`；`.error(T_from,T_to)->[ρ,θ]`、`.distance`、`.distance_sq`、`.weighted_error`、`.error_body_jacobian` | 兼容类/方法名；实际为 principal-log 残差目标 `δ_W²=‖ρ‖²+w²‖θ‖²`，一般不满足三角不等式、非严格数学 metric；`r=Log(T_from⁻¹T_to)^vee`；`w` 严格正且有限 |
| `gtsr/charts.py` | `GeneratorChart`（`coord_names/periodic/periods/dim/forward/jacobian`） | 坐标拓扑与周期语义的所有者 |
| `gtsr/charts.py` | `CanonicalTSRChart` | `Φ_TSR=Trans(x,y,z)Rz(yaw)Ry(pitch)Rx(roll)`；`periodic=(F,F,F,T,T,T)` |
| `gtsr/charts.py` | `PoEChart(axes, representation, order=None, M=None)` | SPACE `Exp(S₁z₁)⋯Exp(Sₘzₘ)M`；BODY `M Exp(B₁z₁)⋯Exp(Bₘzₘ)`；`order` 显式排列 |
| `gtsr/domains.py` | `LinearInterval`、`CircularInterval`、`CoordinateDomain`（`.from_bounds`、`.contains`、`.sample`、`.cont_lower/upper`） | 有界坐标域；周期维 `CircularInterval(2π)` |
| `gtsr/projection.py` | `ProjectionResult`（frozen dataclass，数组 defensive-copy 且默认只读：`z/transform/distance/success/optimality/message`）、`project(gtsr,T,metric,*,starts=None,tol_bounds=1e-12,tol_opt=1e-6,maxiter=1000)` | 多起点局部优化；`success=True` 仅 stationary/KKT candidate，非局部极小或全局 membership 证明 |
| `gtsr/region.py` | `GTSR(T0_w,Tw_e,chart,domain)`、`.canonical(T0_w,Tw_e,Bw)`、`.to_transform`、`.project(T,metric,*,starts=None)`、`.contains(T,metric=None,tol=1e-8)`、`.sample(...,*,partial=...)` | `R(G)={T0_w Φ_C(z) Tw_e | z∈Z}`；canonical 入口对 legacy 角域执行 `κ(Bw)` |

## 2. 冻结约定

| 项 | 约定 |
|---|---|
| twist 排列 | `ξ=[v,ω]`；`hat=[[skew(ω),v],[0,0]]`；`se3_vee` 返回 `[H[0,3],H[1,3],H[2,3],W[2,1],W[0,2],W[1,0]]`（`v` 前 `ω` 后） |
| 螺距 | `h=ωᵀv/‖ω‖²`；`ω=0` 记 `h=∞`（纯平移）；`h=0` 纯旋转；`h≠0` 有限螺距 |
| screw 轴归一化 | 仅逐分量精确为零的 `ω` 是平移轴；任何非零 `ω` 须 `\|‖ω‖-1\|≤64ε_machine`，不静默归一化 |
| 周期语义 | 仅 `h=0` 坐标为 `CircularInterval(2π)`；`h≠0` 与纯平移为 `LinearInterval`（`q+2π` 轴向位移 `2πh`，不 wrap） |
| PoE 生成元顺序 | `order` 显式索引排列（默认 `0..m-1`），不依赖容器偶然顺序 |
| 残差误差 | `r=Log(T(z)⁻¹T_q)^vee=[ρ,θ]`（左不变/body frame）；`δ_W=√(rᵀWr)` 是投影 objective 而非严格数学 metric |
| Jacobian | body frame `J_i=(T(z)⁻¹ ∂T/∂z_i)^vee`，解析 `Ad` 累乘，对中心有限差分 ≤1e-6 |

## 3. 残差目标与容差（冻结）

| 阈值 | 值 | 用途 |
|---|---|---|
| 位姿旋转误差 `e_R` | ≤ 1e-12 | 周期 wrap 验证（旋转角误差，`atan2` 形式） |
| 位姿平移误差 `e_t` | ≤ 1e-12·max(1,‖v‖) | 周期 wrap 验证（平移二范数，按 ‖v‖ 缩放，不与混合量纲比较） |
| 单位轴判定 | `|‖ω‖-1| ≤ 64ε_machine` | PoE 旋转生成元拒绝非单位轴 |
| 螺距零判定 | `|h| ≤ 64ε_machine` | 周期轴判定 |
| 差分样本误差 | `max(e_t,e_R) ≤ 1e-10` | Sprint A E/C 系列、5000 差分样本 |
| 投影 bounds 可行 | ≤ 1e-12 | `project` 成功条件 |
| 投影 KKT/最优性 | ≤ 1e-6 | `project` 成功条件（`optimality` 中心有限差分复核） |
| 投影目标值 | 不劣于参考最优 −1e-8 | 多维非凸（P-08） |
| 通用 membership `tol` | 1e-8 | `contains(...,metric=...,tol=)` 的 witness 残差阈值；不宣称统一的全局投影残差 floor |
| legacy membership | `EPSILON=1e-3` | 仅旧 `TSR`/canonical `contains`/`distance` 路径 |
| 测试数值比较 | 1e-10~1e-15（矩阵/位姿），1e-6（Jacobian 中心差分） | 全测试矩阵 |

## 4. 固定 seeds（随机测试）

`[0, 1, 7, 42, 20260902]`，另加 seed `9917`（高维随机 PoE 采样）、`20260903`（sample 部分坐标）、
seed `441`（已知冷启动假阴性 canary，动态 `xfail`）。所有随机测试显式 `numpy.random.Generator`，
失败样本固化为确定性回归。fig7 采样用 `20260902`。

### 4.1 Canonical differential 结果

| seed | 样本数 | 门槛 | 结果 |
|---:|---:|---:|---|
| 0 | 1000 | `max(e_t,e_R)≤1e-10` | 通过 |
| 1 | 1000 | `max(e_t,e_R)≤1e-10` | 通过 |
| 7 | 1000 | `max(e_t,e_R)≤1e-10` | 通过 |
| 42 | 1000 | `max(e_t,e_R)≤1e-10` | 通过 |
| 20260902 | 1000 | `max(e_t,e_R)≤1e-10` | 通过 |

现有测试只逐样本断言硬阈值，未持久化每 seed 的精确最大误差；因此本记录不编造该统计量。

### 4.2 已知失败/对抗样本清单

- seed `441`：on-region 6D PoE 冷启动落入稳定错误盆地，`δ_W≈0.12828521902191672`、末坐标命中
  下界 `-1.2`；动态 `xfail` 在退出前锁定 success/有限性/bounds/message/盆地数值，warm-start 恢复。
- seed `9917`：显式 bad start 的 6D PoE 投影残差 `>1`；显式 bad+good / good+bad 两种排列均
  `<1e-10`，用于证明实现确实遍历全部 starts，而非只取首项或末项。

## 5. 环境版本（冻结语料）

| 组件 | 版本 |
|---|---|
| Python | 3.11.16（conda-forge，GCC 14.4.0） |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pytest | 9.1.1 |
| matplotlib | 3.11.1（`viz` extra，仅图） |
| ruff | repo-root `.venv/bin/ruff` |

## 6. 硬门结果表

| 硬门 | 内容 | 状态 | 证据 |
|---|---|---|---|
| G1 | canonical 精确嵌入（`GTSR.canonical` + E/C 差分） | ✅ 通过 | `tests/gtsr/test_canonical_embedding.py`、`test_canonical_compatibility.py`；E-01~E-10 + 5000 差分样本 |
| G2 | 通用几何闭环（PoE + 显式残差目标 + 通用投影，冻结语料工程硬门） | ✅ 通过 | `test_poe_chart.py`（G-01~G-09）、`test_projection.py`（P-01~P-10）、`test_metrics.py`（M-01~M-10）；seed-441 canary 动态 `xfail` |
| G3 | 严格表示能力（`𝔉_TSR ⊊ 𝔉_GTSR`） | ✅ 通过 | `docs/gtsr_expressivity_proofs.md`（P1/P2 + L1/L2/L3）；`test_expressivity.py`（X-01~X-03）；`fig7_screw_vs_tsr.png` |

## 7. 冻结交付物清单

- `docs/gtsr_expressivity_proofs.md`：P1 构造性嵌入证明 + P2 严格包含证明（L1/L2/L3 + 奇异面）。
- `tests/gtsr/test_expressivity.py`：X-01（非零螺距 screw 构造 + `2πh` 轴向位移）、
  X-02（`se3_log` pitch 估计 `≤1e-8`）、X-03（z-yaw box 正横向偏离，佐证非证明）。
- `docs/figures/fig7_screw_vs_tsr.png` + `generate_figures.py::fig7_screw_vs_tsr()`。
- 本文档（冻结记录）。

## 8. 全量回归（2026-09-04 修订验证）

- `tests/tsr`：309 passed（legacy differential oracle 未动）。
- `tests/gtsr`：157 passed，1 xfailed（seed-441 canary）。
- `tests/benchmarks`：5 passed。
- `tsr-dev` 全量 `pytest -q`：**471 passed，1 xfailed，1 pre-existing SciPy warning**
  （`test_tsr_chain.py` 的 `_numdiff` 除零警告，与本次改动无关）。
- `.venv/bin/ruff check` + `format --check`：仅对 G-TSR 本次改动范围检查干净；不据此声称 repo-wide format clean。
- 改动文件不含 `src/tsr/tsr.py`（legacy 未动）。

修订复核已通过，几何内核重新冻结；现在可以按上游方案启动 `P_r(s)` / phase estimator / transfer 的 scope 锁定。
