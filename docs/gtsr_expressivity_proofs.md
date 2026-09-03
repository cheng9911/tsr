# G-TSR 表示能力：严格证明（P1 与 P2）

状态：`FROZEN`
日期：`2026-09-03`
上游：`docs/gtsr_geometry_validation_plan.md`（§3.5、§6.6、§11 硬门 G3）

本文给出 G-TSR 表示能力的两个核心命题的严格证明：

- **P1（精确嵌入）**：传统 TSR 表示族 `𝔉_TSR` 精确嵌入 G-TSR 表示族 `𝔉_GTSR`，即 `𝔉_TSR ⊆ 𝔉_GTSR`。
- **P2（严格包含）**：`𝔉_TSR ⊊ 𝔉_GTSR`，G-TSR 表示族严格更大，以非零有限螺距 screw 曲线为反例。

记号与定义沿用 `docs/superpowers/specs/2026-09-03-gtsr-sprint-c-design.md` §2。

## 1. 记号与定义

- `SE(3)`：刚体变换群；`se(3)`：其李代数，twist 记为 `ξ = [v, ω] ∈ ℝ⁶`（`v` 线性在前，`ω` 角速度在后）。
- 指数映射 `Exp : se(3) → SE(3)`；`Ad_T : se(3) → se(3)` 为伴随作用。
- 螺距：对 `ξ = [v, ω]`、`ω ≠ 0`，`h(ξ) = ωᵀv / ‖ω‖²`；`ω = 0` 记 `h = ∞`（纯平移）。
- 表示族：
  - `𝔉_GTSR`：所有 `R(G) = { T0 Φ_C(z) Te | z ∈ Z }`，`C` 为任意 generator chart（PoE，任意 screw 生成元），`Z` 有界。
  - `𝔉_TSR`：所有传统 TSR 的 `R(TSR) = { T0_w Φ_TSR(z) Tw_e | z ∈ B_w }`，其中
    `Φ_TSR(z) = Trans(x,y,z) Rz(yaw) Ry(pitch) Rx(roll)`，`B_w` 为 6×2 box bounds。
- 见证曲线（witness）：`S_h = { Exp(ξ_h q) | q ∈ [a,b] }`，`ξ_h = [v,ω]`，`ω ≠ 0`，`h = ωᵀv/‖ω‖² ≠ 0`，`a < b`。

## 2. P1：构造性精确嵌入

**命题（P1）** `𝔉_TSR ⊆ 𝔉_GTSR`。

**证明。** 定义构造映射

\[
E(T_0, T_e, B_w) = (T_0, T_e, \mathcal C_{\mathrm{TSR}}, \mathcal Z(B_w)),
\]

其中 `C_TSR` 是 canonical chart，其前向

\[
\Phi_{\mathrm{TSR}}(z)
= \operatorname{Trans}(x,y,z) R_z(\mathrm{yaw}) R_y(\mathrm{pitch}) R_x(\mathrm{roll}),
\]

`Z(B_w)` 是与 `B_w` 相同 bounds 的坐标域（周期维按 `CircularInterval(2π)`，其余 `LinearInterval`）。

对任意传统 TSR（外 frame `T0_w, Tw_e`，bounds `B_w`），令 `G = E(T0_w, Tw_e, B_w)`。断言两个位姿集合相等：

\[
R(\mathrm{TSR}) = R(G).
\]

- **`R(TSR) ⊆ R(G)`**：取任意 `T ∈ R(TSR)`，由定义存在 `z ∈ B_w` 使 `T = T0_w Φ_TSR(z) Tw_e`。
  而 `Φ_TSR(z)` 恰为 `C_TSR.forward(z)`，且 `z ∈ B_w ⟺ z ∈ Z(B_w)`（`Z` 由 `B_w` 构造）。
  故 `T = T0_w C_TSR.forward(z) Tw_e ∈ R(G)`。
- **`R(G) ⊆ R(TSR)`**：取任意 `T ∈ R(G)`，形如 `T = T0_w C_TSR.forward(z) Tw_e`、`z ∈ Z(B_w)`。
  由 `C_TSR.forward ≡ Φ_TSR` 与 `Z(B_w)` 的定义，`T` 正是传统 TSR 在 `z` 处的位姿，故 `T ∈ R(TSR)`。

故 `R(E(...)) = R(TSR)`，嵌入是精确的。∎

**代码对应**：`E` 由 `GTSR.canonical(T0_w, Tw_e, Bw)` 实现，`Φ_TSR` 由 `CanonicalTSRChart.forward`
实现，二者逐项对应上述数学符号。数值佐证为 E-01~E-10（确定性场景）与 5000 差分样本
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

于是

\[
h(Ad_T \xi)
= \frac{(R\omega)^{\top} (Rv + p \times (R\omega))}{\|R\omega\|^2}
= \frac{(R\omega)^{\top} Rv + (R\omega)^{\top} (p \times R\omega)}{\|\omega\|^2}.
\]

由 `R` 正交，`(Rω)ᵀ Rv = ωᵀ v`；由叉积垂直性，`(Rω)ᵀ (p × Rω) = 0`。故 `h(Ad_T ξ) = ωᵀv/‖ω‖² = h(ξ)`。∎

### 3.3 引理 L2：单坐标切向螺距 ∈ {0, ∞}

**引理** 传统 TSR 中若仅一个坐标自由（其余零宽），其 1 参数族在每点的切 screw 螺距属于 `{0, ∞}`。

**证明。** 该 1 参数族形如 `g(q) = T0_w Φ_TSR(z_1,…,q,…,z_6) Tw_e`（仅第 `i` 坐标 `q` 自由）。
其切 screw 在任意 frame（body 或 spatial）下都是 `∂Φ_TSR/∂z_i` 经固定 `T0_w, Tw_e` 的 `Ad` 共轭；
而 pitch 是 frame 无关标量。`Φ_TSR` 的坐标偏导：

- 平移坐标 `x/y/z`：`∂Φ_TSR/∂z_i` 是纯平移（`ω = 0`，螺距 `∞`）；
- 旋转坐标 `roll/pitch/yaw`：`∂Φ_TSR/∂z_i` 是绕对应轴的纯旋转（`ω ≠ 0` 且 `h = 0`）。

由 L1，螺距在 `Ad` 共轭下不变，故 `dg/dq` 的螺距 ∈ `{0, ∞}`。∎

### 3.4 引理 L3：维数与奇异面

**引理** 设 `g = T0_w ∘ Φ_TSR ∘ Tw_e`（`Tw_e` 右乘、`T0_w` 左乘均为 `SE(3)` 微分同胚，不改变 rank 与
维数）。则

1. 在 `pitch ≠ ±π/2` 处，`dΦ_TSR` 满秩（rank = 6）：六个坐标方向线性无关；
2. 在 `pitch = ±π/2` 处（Euler 奇异面），roll 与 yaw 的旋转轴共线，`dΦ_TSR` 秩降为 5，此时 roll/yaw
   合并为单一有效旋转自由度，其余坐标（`x,y,z` 平移）仍独立。

**证明（要点）** `Φ_TSR(z) = Trans · Rz · Ry · Rx`。其六个生成方向为 `x,y,z` 平移与 `x,y,z` 轴旋转；
平移三方向恒线性无关。旋转部分 `Rz(yaw) Ry(pitch) Rx(roll)` 的 Jacobian（body 速度）在
`pitch ≠ ±π/2` 时为满秩 3；在 `pitch = ±π/2` 时 roll 与 yaw 轴对齐，旋转 Jacobian 秩降为 2，仅剩
一个有效旋转方向（`pitch` 此时固定）。故结论成立。∎

### 3.5 反证：`S_h ∉ 𝔉_TSR`

**命题** 非零有限螺距的见证曲线 `S_h` 不在 `𝔉_TSR` 中。

**证明。** 反设存在传统 TSR 使 `R(TSR) = S_h`，记 `g = T0_w ∘ Φ_TSR ∘ Tw_e`、`Z = B_w`，令 `d` 为
`B_w` 中非退化（非零宽）坐标的个数。因 `S_h` 是 1 维光滑子流形且 `g(Z) = S_h`，必有 `d ≥ 1`
（否则像为单点）。

**情形 1：`d = 1`。** 由 L2，`g` 的 1 参数族切 screw 螺距 ∈ `{0, ∞}`。但 `S_h = Exp(ξ_h q)` 在每点的
切 screw 螺距为 `h(ξ_h) = h ∉ {0, ∞}`。矛盾。

**情形 2：`d ≥ 2`。**

- 若存在正则点 `z ∈ int(Z)`（`pitch ≠ ±π/2`），则由 L3(1)，`dg_z` 满秩 `d ≥ 2`，局部像为 ≥2 维
  子流形，不能等于 1 维 `S_h`。矛盾。
- 否则 `g(Z) = S_h` 完全落在 `pitch = ±π/2` 奇异面上，即 `pitch` 固定在 `±π/2`（零宽），自由坐标
  属于 `{x, y, z, roll, yaw}`。由 L3(2)，奇异面上 roll 与 yaw 合并为一个有效旋转（零螺距）。于是：
  - 若任一平移坐标 `{x,y,z}` 自由，该平移方向（螺距 `∞`）与有效旋转方向独立，局部像 ≥2 维，矛盾；
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
