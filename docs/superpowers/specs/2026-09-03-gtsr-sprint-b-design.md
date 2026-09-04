# G-TSR Sprint B — 实现级设计（通用几何闭环）

状态：`DESIGN-FROZEN`（2026-09-04 审查修订已复核）
日期：`2026-09-04`
上游：`docs/gtsr_geometry_validation_plan.md`（§3.4~§3.7、§6.4、§6.5、§9 Sprint B、§11 硬门 G2）

本文档把上游验证方案里 Sprint B 的数学契约落到代码级 API。范围严格限定在 **Sprint B**：
PoE chart + 显式 principal-log residual objective + 通用 `project`。不实现 `P_r(s)`、phase estimator、transfer（硬门 G2 通过前
禁止启动，见 §6）。

## 0. 对上游文档的一处修正（G-09 归属）

上游 §9「Sprint B」第 4 条写作「完成 G-01 至 G-08」，但 §11 硬门 G2 要求「G-01 至 G-09」。
**以硬门为准**：Sprint B 的范围是 **G-01 ~ G-09**（G-09 是 chart Jacobian 的有限差分验证，属于
Sprint B 第 3 步）。本设计文档与后续计划均按 G-01~G-09 执行；§9 第 4 条的上游文本视为笔误，
在 Sprint B 落地提交中一并修正。

## 1. 范围与不变量

**本 Sprint 交付：**

- `src/tsr/gtsr/metrics.py`、`projection.py`，`charts.py` 增补 `PoEChart`，
  `region.py` 增补 `GTSR.project` 与通用 membership；
- `tests/gtsr/test_poe_chart.py`（G-01~G-09）、`test_projection.py`（P-01~P-09）；
- 硬门 G2 全绿。

**不变量（贯穿全部测试）：**

- 旧 `TSR` 保持不动，作为 differential oracle；Sprint A 的 canonical 表面（E/C/D 系列）不回退。
- 位姿误差沿用 §6.2 的 `e_t` + `atan2` 形式 `e_R`，矩阵/优化数值比较用 `1e-10`~`1e-12`，
  Jacobian 用中心有限差分 `1e-6`，legacy `EPSILON=1e-3` 只留在旧 `contains/distance`。
- 随机测试一律显式 `numpy.random.Generator`，固定 seeds `[0, 1, 7, 42, 20260902]`。
- 多维非凸投影是**多起点局部优化**，不宣称总能找到全局最优；失败必须显式暴露（`success=False`），
  不得伪装。这里 `SE3Metric` 是兼容类名；实际目标 `δ_W(A,B)=‖Log(A⁻¹B)^vee‖_W` 是非负
  principal-log residual，一般不满足三角不等式，并非严格数学 metric。设 `ẑ ∈ Z`、
  `δ̂ = δ_W(T(ẑ), T_q)`、`δ* = inf_{z∈Z} δ_W(T(z), T_q)`；因 `ẑ` 可行有
  `δ* ≤ δ̂`，故 `δ̂ ≤ tol ⟹ δ* ≤ tol`，但反方向不成立。此单向健全性只用非负可行 witness，
  不依赖三角不等式。据此：
  - `contains=True` 是对残差膨胀集合 `R_tol = {T | inf_{X∈R} δ_W(X,T) ≤ tol}` 的声音证书（无假阳性），
    不代表严格属于原集合 `R`；
  - `contains=False` 解释为「未找到区域内证据」（局部优化可能落入错误局部极小，存在假阴性），
    不得写成「不可达」；
  - `success=True` 只表示「返回了满足 bounds 与数值 KKT 的局部候选」，不是「局部极小证明」，
    更不是全局 membership 证书。

**明确不做：** P1/P2 严格证明（Sprint C）、`test_expressivity.py`、序列化、自适应/相位/迁移。

## 2. 约定表（convention，Sprint B 新增）

| 项 | 约定 |
|---|---|
| twist 排列 | `ξ = [v, ω] ∈ ℝ⁶`，`v`（线性）在前，`ω`（角速度）在后 |
| `hat` | `hat(ξ) = [[skew(ω), v], [0, 0]] ∈ se(3)`；`vee` 为其逆 |
| screw 螺距 | `h = ωᵀv / ‖ω‖²`；`ω=0` 纯平移（`h=∞`），`h=0` 纯旋转，`h≠0` 有限螺距 |
| screw 轴归一化 | 仅逐分量精确为零的 `ω` 表示纯平移；任何非零 `ω` 均要求 `|‖ω‖-1|≤64ε_machine`，不静默归一化；因此坐标量 `q` 就是旋转角 |
| 周期语义 | 在上述单位轴契约下，仅 `h=0`（纯旋转）坐标是周期轴 `CircularInterval(2π)`；`h≠0` 与纯平移坐标是 `LinearInterval`（`q` 与 `q+2π` 轴向位移相差 `2πh`，不得 wrap）|
| PoE SPACE | `Φ(z) = Exp(S₁z₁) ⋯ Exp(Sₘzₘ) M`（screw 轴在空间/世界 frame）|
| PoE BODY | `Φ(z) = M Exp(B₁z₁) ⋯ Exp(Bₘzₘ)`（screw 轴在 body/home frame）|
| 生成元顺序 | `order` 为显式索引排列（默认 `0..m-1`），不依赖容器偶然顺序 |
| home 变换 | chart 内部固定 `M`（Sprint B 限制 `M=I`，接口保留）|
| 残差 | `r(z,T) = Log(T(z)⁻¹ T)^vee = [ρ, θ]`（body/左不变 frame）|
| 残差权重 | `W = diag(1,1,1, w²,w²,w²)`，`δ_W² = ‖ρ‖² + w²‖θ‖²`；`w` 为 `rotation_weight`（m/rad，默认 1.0）；`δ_W` 一般不满足三角不等式 |
| Jacobian | `J(z) ∈ ℝ^{6×m}`，第 `i` 列 = `(T(z)⁻¹ ∂T(z)/∂z_i)^vee`（body frame）|
| 投影问题 | `z* = argmin_{z∈Z} δ_W(T(z), T_q)²` |

## 3. 模块设计

### 3.1 metrics.py — 显式 SE(3) principal-log residual objective

```python
# Lie 群纯函数（se(3)，[v,ω] 排列）
se3_hat(xi) / se3_vee(mat)                  # se(3) ↔ ℝ⁶
so3_log(R) -> omega                          # axis-angle，θ≈0 用极限
se3_log(T) -> xi                             # [v, ω]
se3_adjoint(T) -> Ad                         # 6×6，作用于 [v,ω]
se3_left_jacobian_inverse(xi) -> J_l⁻¹       # 6×6（左 Jacobian 逆，精确级数 + 6×6 求逆）

class SE3Metric:
    def __init__(self, rotation_weight=1.0): ...   # w（m/rad），严格正且有限
    def error(self, T_from, T_to) -> np.ndarray    # Log(T_from⁻¹ T_to)^vee -> [ρ, θ] (6,)
    def distance(self, T_from, T_to) -> float      # δ_W=sqrt(‖ρ‖²+w²‖θ‖²)；兼容方法名
    def distance_sq(self, r) -> float              # rᵀ W r
    def error_body_jacobian(self, T_from, T_to) -> E  # E = ∂r/∂(body 扰动 T_from) ∈ ℝ^{6×6}
```

`r` 语义严格等于 §3.7 的 `Log(T(z)⁻¹T)^vee`。**投影的解析梯度必须用精确左不变梯度**，
不得用一阶近似（否则对残差非零的不可达查询，KKT 残差会系统性偏离，G2 的 `≤1e-6` 无法保证）：

```
g = T(z)⁻¹ T_q ;  r = se3_log(g)^vee ;  E = -J_l(r)⁻¹ Ad_{g⁻¹}
∂r/∂z_i = E · J_i(z)        (J_i = chart.jacobian 第 i 列)
∇_z f  = 2 J(z)ᵀ Eᵀ W r      (f = rᵀ W r)
r=0 时 E=-I，退化为 -2 Jᵀ W r。
```

`so3_log` 用 `atan2(sinθ, cosθ)` 求主值角；对钝角旋转从 `sym(R)` 的最大特征值特征向量恢复
旋转轴，避免 `θ→π` 时反对称部分退化。`se3_log` 的 `V⁻¹` 系数在小角处用级数、其余使用稳定的
半角式。`J_l⁻¹` 在 `θ→0` 处用极限展开，数值稳定。

### 3.2 charts.py — PoEChart

```python
class PoEChart(GeneratorChart):
    def __init__(self, axes, representation, order=None, M=None):
        # axes: Sequence[6-vec [v,ω]]，长度 m；若 ω!=0，必须 |‖ω‖-1|≤64ε_machine
        # representation ∈ {"space","body"}
        # order: 显式索引排列（默认 range(m)）；M: 4x4 home（默认 I）
        # coord_names 自动生成 ("q0","q1",...)，periodic 由每轴 pitch 决定：
        #   ‖ω‖>0 且 h≈0 -> True/2π；否则 False/None
    def forward(self, z) -> np.ndarray              # 依 representation 连乘 Exp
    def jacobian(self, z) -> np.ndarray             # body-frame (6, m)，解析 Ad 累乘
```

**Jacobian 公式**（body frame，解析）：

- SPACE：`J_i = Ad_{(Exp(S_i z_i)⋯Exp(S_m z_m) M)⁻¹}(S_i)`；
- BODY：`J_i = Ad_{(Exp(B_i z_i)⋯Exp(B_m z_m))⁻¹}(B_i)`。

`periodic` 由满足单位轴契约的 `axes` 的 pitch 自动推断（§3.4：周期是显式属性，此处按纯旋转唯一判定），
pitch 的零判定同样使用 `64ε_machine`。非零且不满足单位契约的 `ω` 直接拒绝；微小但非零的
`ω` 不得降级为平移轴。这保证 `q` 的角度语义。被声明为周期坐标时，`q+2π` 的 pose 验证容差按
旋转角误差 `e_R ≤ 1e-12` 与平移二范数 `e_t ≤ 1e-12·max(1, ‖v‖)` 分开计（不比较混合量纲的矩阵
最大元素）；测试坐标 `q` 限定在单个规范周期内；数学上的零螺距 screw 仍严格满足 `2π` 周期。
`representation` 仅允许 `{"space","body"}`（§3.5）。

### 3.3 projection.py — ProjectionResult 与通用投影

```python
@dataclass(frozen=True)
class ProjectionResult:
    z: np.ndarray            # defensive-copy + read-only；失败时仍返回最佳努力值
    transform: np.ndarray    # defensive-copy + read-only；T0_w @ forward(z) @ Tw_e
    distance: float          # δ_W(T(z), T_q)；兼容字段名
    success: bool            # 是否返回满足 bounds + 数值 KKT 的 stationary candidate（非最小值/全局证书）
    optimality: float | None # KKT/梯度残差（success=True 时 <= 1e-6；失败可为 None）
    message: str             # 诊断信息（success=False 时说明原因）

def project(gtsr, T, metric, *, starts=None, tol_bounds=1e-12, tol_opt=1e-6) -> ProjectionResult
```

投影流程：

1. **固定多起点**：`domain.cont_lower`、`cont_upper`、中点，外加固定种子生成的若干合法坐标
   （默认 4 个，`numpy.random.default_rng(20260902)`）；canonical 查询若可逆到 domain 内，则追加
   精确 inverse start；显式 `starts` 不得为空；
2. 每个起点做 **L-BFGS-B**（`bounds=domain`，精确左不变解析梯度 `2 Jᵀ Eᵀ W r`），保留 `distance` 最小者；
3. 对最佳点的自由坐标执行 bounded nonlinear least-squares，残差为 `[ρ, wθ]`、Jacobian 为
   `sqrt(W) E J`；该阶段把 L-BFGS 的一般优化收敛与 on-region 的零残差可行性区分开；
4. 最优性独立复核：对最优 `z` 用**中心有限差分**求真实梯度，投影到非活跃坐标后取范数
   （`optimality`）；
5. `success = 收敛 且 bounds 可行 ≤ 1e-12 且 optimality ≤ 1e-6 且 无 NaN`；
   任一失败即 `success=False`，`message` 写明失败维度，`z/transform/distance` 仍返回最佳努力值。
   `success=True` 时 `message` 固定为 `"stationary/KKT candidate; global optimality not certified"`。

一维投影（P-04/P-05）走独立的高精度标量参考（`scipy.optimize.minimize_scalar` 或密集网格 + 黄金分割），
**不用被测的多维优化器自身作为 oracle**（§6.5）。

### 3.4 region.py — 通用区域运算

```python
class GTSR:
    # --- Sprint A（不动）---
    def to_transform(self, z) -> np.ndarray
    def to_xyzrpy(self, T) -> np.ndarray            # canonical only
    def is_valid(self, z, ignoreNAN=False) -> bool[6]   # canonical only
    def contains(self, T, metric=None, tol=1e-8) -> bool
    def distance(self, T, rotation_weight=1.0) -> (float, np.ndarray)  # legacy, canonical only
    def sample_xyzrpy(self, partial=NANBW, rng=None) -> np.ndarray      # canonical only
    def sample(self, xyzrpy=_UNSET, rng=None, *, partial=_UNSET) -> np.ndarray
    # --- Sprint B 新增 ---
    def project(self, T, metric, *, starts=None) -> ProjectionResult
```

`contains(T, metric=None, tol=1e-8)`：`metric is None` 走 legacy 路径（canonical 专属，
`EPSILON=1e-3`）；`metric` 给定走通用路径（`project(T, metric).distance <= tol`，任意 chart）。
通用路径的 `contains=True` 是对残差膨胀集合
`R_tol = {T | inf_{X∈R} δ_W(X,T) ≤ tol}` 的声音证书（无假阳性，
不代表严格属于原集合 `R`）；`contains=False` 表示「未找到区域内证据」（局部优化可能落入错误局部
极小，存在假阴性，不得读作「不可达」）。已知 on-region 查询的投影残差在 canonical inverse start 与
bounded least-squares 精化下不超过默认容差；距离边界大于 `tol` 的 pose 仍判为区域外。

`project(..., starts=None)`：`starts` 透传给通用投影器，供轨迹/相位应用以「上一时刻坐标」warm-start
（`starts=[z_prev]`）绕过冷启动局部极小；显式空 `starts` 抛 `ValueError`。
通用 `distance` 由 `project(T, metric).distance` 提供，不再另设一个与 legacy `distance` 同名的
第二语义方法（避免混淆，§3.7「互不混淆」）。

`sample(xyzrpy=..., rng=...)` 保留 canonical legacy 关键字与位置参数；chart-neutral 的新入口是
keyword-only `partial=...`。两者同时传入时明确抛 `TypeError`。对 generalized chart，坐标向量长度为
`chart.dim`，`NaN` 维按坐标顺序各消费一次标量随机抽样，非 `NaN` 维必须已在 domain 内；省略坐标时
全维抽样。返回值统一为外 frame 组合后的变换。

## 4. 数据流

```
z ──forward──> T_local ──> T0_w @ . @ Tw_e ──> T(z)          # to_transform
z ──jacobian──> J(z)（body frame）                            # 解析，G-09 对有限差分
T_q ──project──> multi-start L-BFGS-B ──> bounded LS refine ──> ProjectionResult
T_q ──metric.error──> r = Log(T(z)⁻¹T_q)^vee ──> δ_W         # 残差目标，P 系列
```

## 5. 测试矩阵（对应文档 §6.4 / §6.5）

- `test_poe_chart.py`：G-01~G-09。
  - G-01 x/y/z 单轴平移；G-02 任意单位轴纯旋转（Rodrigues）；G-03 轴不过原点的零螺距 screw；
    G-04 非零有限螺距 screw（`θ=q`、轴向 `hq`）；非单位 `ω` 拒绝；G-05 两个不对易生成元（显式解析矩阵 oracle，
    区分度 > 数值噪声 6 个数量级）；G-06 space/body + Adjoint 一致；G-07 外 `T0/Te` 组合；
    G-08 `RᵀR=I`、`det=1`、末行 `[0,0,0,1]`（残差 ≤ 1e-12）；G-09 解析 Jacobian 对中心有限差分（≤ 1e-6）。
  - M-09 随机轴精确/近 `π` 的 `Exp(Log(T))`（含平移）矩阵 round-trip。
  - 高维随机 PoE 与非单位 outer frames 的 `sample→project/contains`，以及 reviewer 4D 对抗复现。
- `test_projection.py`：P-01~P-09。
  - P-01 内部坐标查询（`distance ≤ 1e-10`）；P-02 零宽 domain；P-03 canonical 对 legacy/解析裁剪
    （各自残差/距离定义）；P-04 1D 平移/纯旋转对解析投影；P-05 1D 非零螺距 screw 对高精度标量参考；
    P-06 幂等 `project(project(T).transform)`；P-07 `contains(T)` ⟺ 投影残差 ≤ tol；
    P-08 多生成元非凸（固定多起点 + KKT 残差 + 目标值 ≥ 参考最优 −1e-8）；P-09 失败传播（`success=False`）。

一维高精度参考与多维最优性门槛严格按 §6.5 的 `1e-12`（bounds）/`1e-6`（KKT）/`1e-8`（目标值）执行。

## 6. 退出条件（硬门 G2：冻结语料上的工程硬门）

G2 是**冻结语料上的工程硬门**：`project` 返回多起点优化所得 stationary/KKT candidate（非全局证书），`contains` 是近似
membership（无假阳性、可能假阴性）。本硬门只在冻结 seeds 与公开验证语料上断言，不宣称任意非凸
PoE 的严格全局 membership（那需要全局优化/认证求解，超出本 Sprint 范围）。

- G-01~G-09 全通过；
- `sample → contains`、`project → contains`、投影幂等性通过（限定冻结 seeds 与公开验证语料）；
- 1D screw 投影与独立高精度参考一致；
- 多维优化满足 bounds `≤ 1e-12`、KKT `≤ 1e-6`、目标值不劣于参考最优 `1e-8`，失败状态可见；
- 周期坐标 `q+2π` 的 pose 验证按 `e_R ≤ 1e-12` 与 `e_t ≤ 1e-12·max(1, ‖v‖)` 分开计；
- seed 441 的已知冷启动假阴性以动态 `xfail` canary 显式保留（warm-start `starts=[z]` 复现
  `distance ≤ 1e-10`），不得静默删除；
- 全量回归无回退（`tests/tsr/` 与 `tests/gtsr/` 全绿）。

G2 通过前，**禁止**启动 `P_r(s)`、phase estimator 或 transfer（§11 硬门规则）。
