<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (c) 2025 Siddhartha Srinivasa -->

# G-TSR 表示能力：严格证明（P1 与 P2）

状态：`FROZEN`（2026-09-04 独立复核通过）
日期：`2026-09-04`
上游：`docs/gtsr_geometry_validation_plan.md`（§3.5、§6.6、§11 硬门 G3）

本文给出 G-TSR 表示能力的两个核心命题的严格证明：

- **P1（精确嵌入）**：传统 TSR 表示族 `𝔉_TSR` 精确嵌入 G-TSR 表示族 `𝔉_GTSR`，即 `𝔉_TSR ⊆ 𝔉_GTSR`。
- **P2（严格包含）**：`𝔉_TSR ⊊ 𝔉_GTSR`，G-TSR 表示族严格更大，以非零有限螺距 screw 曲线为反例。

记号与定义沿用 `docs/superpowers/specs/2026-09-03-gtsr-sprint-c-design.md` §2。

## 1. 记号与定义

- `SE(3)`：刚体变换群；`se(3)`：其李代数，twist 记为 `ξ = [v, ω] ∈ ℝ⁶`（`v` 线性在前，`ω` 角速度在后）。
- 指数映射 `Exp : se(3) → SE(3)`；`Ad_T : se(3) → se(3)` 为伴随作用。
- 螺距：对 `ξ = [v, ω]`、`ω ≠ 0`，`h(ξ) = ωᵀv / ‖ω‖²`；`ω = 0` 记 `h = ∞`（纯平移）。
- 表示族（两者均指**单个区域**，不包含 `TSRChain`、有限/无限 union 或轨迹级组合）：
  - `𝔉_GTSR`：所有 `R(G) = { T0 Φ_C(z) Te | z ∈ Z }`，`C` 为任意 generator chart（PoE，任意 screw 生成元），`Z` 有界。
  - `𝔉_TSR`：所有**语义合法**传统单 TSR 的
    `R(TSR) = { T0_w Φ_TSR(z) Tw_e | z ∈ κ(B_w) }`。这里 `T0_w,Tw_e` 是有限合法的 `SE(3)`
    变换，`B_w` 是有限 `6×2` bounds，平移下界不大于上界，且每个 raw 角 span `Δ≥-2π`。
    `Δ<-2π` 会使旧实现的 `EPSILON` 分支在近阈值处改变集合语义，因而不是本数学表示族的合法角域。
- legacy 角 bounds 规范化 `κ`：平移三维不变。对角维 raw span `Δ=u-l`，令

  \[
  \Delta_\kappa=
  \begin{cases}
  \min\{2\pi,\Delta\}, & \Delta\ge0,\\
  \Delta+2\pi, & -2\pi\le\Delta<0.
  \end{cases}
  \]

  连续下界是 `wrap(l)` 到 `[-π,π)`，连续上界为 `wrap(l)+Δκ`。因此负 span 只 wrap 一次，正 span
  超过 `2π` clamp 为整圆。这正是具有合法连续角域的 legacy TSR 的位姿集合语义。
- canonical map 为
  `Φ_TSR(z) = Trans(x,y,z) Rz(yaw) Ry(pitch) Rx(roll)`。
- 见证曲线（witness）：`S_h = { Exp(ξ_h q) | q ∈ [a,b] }`，`ξ_h = [v,ω]`，`ω ≠ 0`，`h = ωᵀv/‖ω‖² ≠ 0`，`a < b`。

## 2. P1：构造性精确嵌入

**命题（P1）** `𝔉_TSR ⊆ 𝔉_GTSR`。

**证明。** 定义构造映射

\[
E(T_0, T_e, B_w) = (T_0, T_e, \mathcal C_{\mathrm{TSR}}, \mathcal Z(\kappa(B_w))),
\]

其中 `C_TSR` 是 canonical chart，其前向

\[
\Phi_{\mathrm{TSR}}(z)
= \operatorname{Trans}(x,y,z) R_z(\mathrm{yaw}) R_y(\mathrm{pitch}) R_x(\mathrm{roll}),
\]

`Z(κ(B_w))` 使用上述规范化后的集合 bounds（周期维按 `CircularInterval(2π)`，其余
`LinearInterval`）。不同但相差整数个 `2π` 的角坐标代表同一旋转，因此规范化不改变 legacy 位姿集合。

对任意上述定义的语义合法传统 TSR（外 frame `T0_w, Tw_e`，bounds `B_w`），令
`G = E(T0_w, Tw_e, B_w)`。断言两个位姿集合相等：

\[
R(\mathrm{TSR}) = R(G).
\]

- **`R(TSR) ⊆ R(G)`**：取任意 `T ∈ R(TSR)`，由规范化后的集合定义存在
  `z ∈ κ(B_w)` 使 `T = T0_w Φ_TSR(z) Tw_e`。而 `Φ_TSR(z)` 恰为 `C_TSR.forward(z)`，且
  `z ∈ κ(B_w) ⟺ z ∈ Z(κ(B_w))`。
  故 `T = T0_w C_TSR.forward(z) Tw_e ∈ R(G)`。
- **`R(G) ⊆ R(TSR)`**：取任意 `T ∈ R(G)`，形如 `T = T0_w C_TSR.forward(z) Tw_e`、
  `z ∈ Z(κ(B_w))`。由 `C_TSR.forward ≡ Φ_TSR` 与规范域定义，`T` 正是传统 TSR 的位姿，故
  `T ∈ R(TSR)`。

故 `R(E(...)) = R(TSR)`，嵌入是精确的。∎

**代码对应**：`E` 由 `GTSR.canonical(T0_w, Tw_e, Bw)` 实现（入口先执行 `κ`），`Φ_TSR` 由
`CanonicalTSRChart.forward` 实现，二者逐项对应上述数学符号。数值佐证为 E-01~E-10（确定性场景）与 5000 差分样本
（`max(e_t, e_R) ≤ 1e-10`，Sprint A 已交付）。

## 3. P2：严格包含

**命题（P2）** `𝔉_TSR ⊊ 𝔉_GTSR`。

**证明分两步**：先证 `S_h ∈ 𝔉_GTSR`（构造），再证 `S_h ∉ 𝔉_TSR`（反证）。

### 3.1 构造：`S_h ∈ 𝔉_GTSR`

取 chart `C = PoEChart([ξ_h])`（单个生成元，`representation="space"`）、1 维线性 domain `Z = [a,b]`、
外 frame `T0 = Te = I`。则

\[
R(G) = \{ \Phi_C(q) \mid q \in [a,b] \} = \{ \operatorname{Exp}(\xi_h q) \mid q \in [a,b] \} = S_h.
\]

故 `S_h ∈ 𝔉_GTSR`。∎

### 3.2 引理 L1：螺距共轭不变

**引理** 对任意 `T = (R, p) ∈ SE(3)` 与 `ξ = [v, ω] ∈ se(3)`，`h(Ad_T ξ) = h(ξ)`。

**证明。** `Ad_T` 在 `[v, ω]` 排列下为 `Ad_T = [[R, [p]× R], [0, R]]`，故

\[
Ad_T \xi = \big[ R v + p \times (R \omega),\; R \omega \big].
\]

若 `ω=0`，则变换后 `Rω=0`，纯平移仍为纯平移，故约定意义下 `∞` 保持不变。以下设 `ω≠0`，于是

\[
h(Ad_T \xi)
= \frac{(R\omega)^{\top} (Rv + p \times (R\omega))}{\|R\omega\|^2}
= \frac{(R\omega)^{\top} Rv + (R\omega)^{\top} (p \times R\omega)}{\|\omega\|^2}.
\]

由 `R` 正交，`(Rω)ᵀ Rv = ωᵀ v`；由叉积垂直性，`(Rω)ᵀ (p × Rω) = 0`。故 `h(Ad_T ξ) = ωᵀv/‖ω‖² = h(ξ)`。∎

### 3.3 引理 L2：单坐标切向螺距 ∈ {0, ∞}

**引理** 传统 TSR 中若仅一个坐标自由（其余零宽），其 1 参数族在每点的切 screw 螺距属于 `{0, ∞}`。

**证明。** 该 1 参数族形如 `g(q) = T0_w Φ_TSR(z_1,…,q,…,z_6) Tw_e`（仅第 `i` 坐标 `q` 自由）。
以 body 或 spatial trivialization 将矩阵切向量映射到 `se(3)`。具体地，左外 frame 在 spatial twist
中给出 `Ad_{T0_w}`，右外 frame 在 body twist 中给出 `Ad_{Tw_e^{-1}}`；canonical 乘积中该坐标前后
已固定的因子也只产生固定 `Ad` 变换。因此无需把裸矩阵偏导误称为 screw。该坐标的 trivialized
生成方向为：

- 平移坐标 `x/y/z`：对应的 trivialized direction 是纯平移（`ω = 0`，螺距 `∞`）；
- 旋转坐标 `roll/pitch/yaw`：对应的 trivialized direction 是绕相应轴的纯旋转（`ω ≠ 0` 且 `h = 0`）。

由 L1，螺距在这些 `Ad` 变换下不变，故曲线的 trivialized tangent screw 螺距 ∈ `{0, ∞}`。∎

### 3.4 引理 L3：维数与奇异面

**引理** 设 `g = T0_w ∘ Φ_TSR ∘ Tw_e`（`Tw_e` 右乘、`T0_w` 左乘均为 `SE(3)` 微分同胚，不改变 rank 与
维数）。则

1. 在 `cos(pitch)≠0` 处，`dΦ_TSR` 满秩（rank = 6）：六个坐标方向线性无关；
2. 在 `cos(pitch)=0`（即 `pitch=π/2 mod π`）处，roll 与 yaw 的旋转轴共线，`dΦ_TSR` 秩降为 5，此时 roll/yaw
   合并为单一有效旋转自由度，其余坐标（`x,y,z` 平移）仍独立。

**证明（要点）** `Φ_TSR(z) = Trans · Rz · Ry · Rx`。其六个生成方向为 `x,y,z` 平移与 `x,y,z` 轴旋转；
平移三方向恒线性无关。旋转部分 `Rz(yaw) Ry(pitch) Rx(roll)` 的 Jacobian（body 速度）在
`cos(pitch)≠0` 时为满秩 3；在 `cos(pitch)=0` 时 roll 与 yaw 轴对齐，旋转 Jacobian 秩降为 2：
一个 pitch 方向加一个合并后的 roll/yaw 方向。后文若把 pitch 固定在奇异面上，则只余后者。故结论成立。∎

### 3.5 反证：`S_h ∉ 𝔉_TSR`

**命题** 非零有限螺距的见证曲线 `S_h` 不在 `𝔉_TSR` 中。

**证明。** 反设存在传统 TSR 使 `R(TSR) = S_h`，记 `g = T0_w ∘ Φ_TSR ∘ Tw_e`、`Z = κ(B_w)`，令 `d` 为
`κ(B_w)` 中非退化（非零宽）坐标的个数。因非零螺距使 `S_h` 为 1 维嵌入光滑曲线（含端点），且
`g(Z) = S_h`，必有 `d ≥ 1`
（否则像为单点）。

**情形 1：`d = 1`。** 由 L2，`g` 的 1 参数族切 screw 螺距 ∈ `{0, ∞}`。但 `S_h = Exp(ξ_h q)` 在每点的
切 screw 螺距为 `h(ξ_h) = h ∉ {0, ∞}`。矛盾。

**情形 2：`d ≥ 2`。**

- 若 active sub-box 的相对内部存在正则点 `z`（`cos(pitch)≠0`），则由 L3(1)，完整 Jacobian 的任意
  active 列子集线性无关，restricted Jacobian 的秩为 `d≥2`。常秩/逆函数定理给出局部像维数至少 2，
  不能等于 1 维 `S_h`。矛盾。特别地，pitch 若非退化，其区间相对内部必含正则点，因为奇异值是离散的。
- 否则 active sub-box 完全落在 `cos(pitch)=0` 奇异面上，即 pitch 固定为 `π/2 mod π`（零宽），自由坐标
  属于 `{x, y, z, roll, yaw}`。由 L3(2)，奇异面上 roll 与 yaw 合并为一个有效旋转（零螺距）。于是：
  - 若至少两个平移坐标自由，其平移列彼此独立，restricted rank 至少 2，矛盾；
  - 若至少一个平移坐标与至少一个旋转坐标自由，平移方向与非零有效旋转方向独立，restricted rank
    至少 2，矛盾；
  - 若仅 `roll, yaw` 自由（`d = 2`），奇异面上二者合并为单一有效旋转，像为 1 维纯旋转（零螺距），
    不可能是非零有限螺距的 `S_h`。矛盾。

综上，反设不成立，`S_h ∉ 𝔉_TSR`。∎

### 3.6 结论

由 §3.1（`S_h ∈ 𝔉_GTSR`）与 §3.5（`S_h ∉ 𝔉_TSR`），且 §2 P1 已证 `𝔉_TSR ⊆ 𝔉_GTSR`，故

\[
\mathfrak F_{\mathrm{TSR}} \;\subsetneq\; \mathfrak F_{\mathrm{GTSR}},
\]

即 G-TSR 表示族严格包含传统 TSR 表示族。∎

## 4. 数值佐证（非证明）

`tests/gtsr/test_expressivity.py`（X-01~X-03）对上述构造与差异给出数值展示：

- **X-01**：`ξ_h = [0,0,h,0,0,1]`（轴 z，`h = 0.05`）构造 `S_h`；`q+2π` 产生 `2πh` 轴向位移（不 wrap）。
- **X-02**：由相邻相对变换的 `se3_log` 估计 pitch，`|ĥ − h| ≤ 1e-8`。
- **X-03**：z-yaw box（最接近的传统 TSR 候选）相对 screw 曲线有正横向偏离（box 严格大于 1 维曲线）。

这些数值只展示构造与差异，**不承担** §3 的证明责任（上游 §6.6：数值部分「不承担遍历所有传统 TSR 的
证明责任」）。
