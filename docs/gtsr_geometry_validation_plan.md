# G-TSR 几何内核设计与验证方案

状态：`ACTIVE / GEOMETRY-FREEZE`

版本：`v0.1`

日期：`2026-09-02`

## 1. 目标与范围锁定

本阶段只回答三个问题：

1. G-TSR 的几何定义是否明确、无歧义且可实现；
2. 传统 TSR 是否能够无损嵌入 G-TSR；
3. G-TSR 是否至少能表示一个传统 TSR 不可表示的几何集合。

本阶段明确不实现、不验证：

- 任务响应规律 `P_r(s)`；
- 在线相位估计；
- 跨对象、跨任务或跨场景迁移；
- 任务成功率、闭环控制性能和 sim-to-real；
- 任何“优于全部相关方法”的系统级结论。

只有本文第 11 节的三个硬门全部通过，才允许进入自适应、相位和迁移阶段。

## 2. 当前仓库基线

现有实现以 `src/tsr/tsr.py::TSR` 为参考 oracle。已经确认的语义如下：

| 项目 | 当前实现语义 |
|---|---|
| 坐标存储顺序 | `[x, y, z, roll, pitch, yaw]` |
| 旋转约定 | `Rz(yaw) Ry(pitch) Rx(roll)` |
| 局部位姿 | `Trans(x,y,z) Rz(yaw) Ry(pitch) Rx(roll)` |
| 外部变换 | `T0_w @ Tw @ Tw_e` |
| 角度范围 | 支持普通区间、跨 `±π` 的 outer interval 和最长 `2π` 的区间 |
| `contains()` | 解析式坐标/欧拉角区间判断，容差来自 `utils.EPSILON=1e-3` |
| `distance()` | 在 `[xyz,rpy]` 坐标中裁剪并求加权二范数 |
| `distance_optimize()` | 用现有 `geodesic_distance()` 数值优化，度量与 `distance()` 不同 |
| `sample_xyzrpy()` | 各自由坐标独立均匀采样，并支持用 `NaN` 指定待采样维度 |

基线检查：`tests/tsr/test_tsr.py` 当前为 `13 passed`。该结果只证明旧实现当前测试通过，不证明 G-TSR 命题。

## 3. 冻结的数学契约

### 3.1 区域定义

定义

\[
\mathcal G=(T_0, T_e, \mathcal C, \mathcal Z),
\qquad
\mathcal R(\mathcal G)
=\{T_0\,\Phi_{\mathcal C}(z)\,T_e\mid z\in\mathcal Z\}\subset SE(3).
\]

其中：

- `T0`、`Te` 是合法的 `SE(3)` 齐次变换；
- `GeneratorChart` 定义坐标的几何意义、组合顺序、表达坐标系和拓扑；
- `CoordinateDomain` 是有界的可行坐标集合 `Z`；
- `GTSR` 组合外部变换、chart 与 domain，提供区域级操作。

### 3.2 单一事实源

坐标拓扑与任务边界必须分离：

- chart 拥有坐标名称、生成元、组合顺序、表达 frame，以及坐标是线性还是周期性的元数据；
- `GTSR.domain` 唯一拥有本区域的上下界、端点闭合性和跨周期边界表示；
- chart 中不得再保存一份可变 bounds，防止 chart 和 GTSR 的 domain 不一致。

### 3.3 Canonical TSR chart

传统 TSR 的 canonical chart 冻结为

\[
\Phi_{\mathrm{TSR}}(z)
=\operatorname{Trans}(x,y,z)
R_z(\mathrm{yaw})R_y(\mathrm{pitch})R_x(\mathrm{roll}),
\]

其中数组仍按

\[
z=[x,y,z,\mathrm{roll},\mathrm{pitch},\mathrm{yaw}]^\top
\]

存储。必须区分“数组存储顺序”和“矩阵乘法顺序”。

### 3.4 周期性

周期性是坐标域的显式属性，不能由“生成元包含旋转”自动推断。

- 纯角坐标可声明为 `CircularInterval(period=2π)`；
- 线性位移使用 `LinearInterval`；
- 非零螺距的 screw 坐标默认是线性的，因为 `q` 增加 `2π` 后仍产生轴向位移；
- outer interval 只对明确声明为周期坐标的维度有效。

### 3.5 PoE 表达与作用约定

PoE chart 不接受含义模糊的自由字符串。其表达约定冻结为枚举与明确公式：

- `representation=SPACE`：`Φ(z)=Exp(S1 z1)...Exp(Sm zm) M`；
- `representation=BODY`：`Φ(z)=M Exp(B1 z1)...Exp(Bm zm)`；
- `order` 是显式索引排列，不能依赖容器偶然顺序；
- `M` 是 chart 内部的固定 home transform，`T0` 和 `Te` 仍只属于区域外层。

若第一版不需要非单位 `M`，实现可以限制 `M=I`，但接口和测试仍须保留 space/body 的数学含义，不得用 `"left"`、`"local"` 等未定义字符串代替。

### 3.6 逆映射与投影

不把全局 `inverse(T)` 作为所有 chart 的强制协议。原因是欧拉角在奇异点多值，任意生成元 chart 也可能非单射。

冻结接口如下：

- `chart.forward(z) -> T_local`：所有 chart 必须实现；
- `canonical_chart.coordinates(T_local, domain) -> z`：canonical chart 的兼容能力；
- `gtsr.project(T, metric) -> ProjectionResult`：通用 chart 的区域级求解接口；
- `ProjectionResult` 至少包含 `z`, `transform`, `distance`, `success`, `optimality`, `message`。

逆向验证比较重构后的位姿，而不是强行比较唯一的欧拉角数组。

### 3.7 距离契约

必须同时保留两条互不混淆的距离路径：

1. **legacy compatibility path**：canonical G-TSR 调用或严格复现现有 `TSR.distance()`，仅用于旧 API 回归；
2. **generic geometry path**：定义

   \[
   r(z,T)=\operatorname{Log}\!\left(T(z)^{-1}T\right)^\vee,
   \qquad
   d_W(T(z),T)=\sqrt{r^\top W r},
   \]

   其中 `W` 必须显式给出、正定，并说明平移与旋转的单位权重。

“集合嵌入成立”不推出这两种距离的数值相等。论文和测试都不得用旧 `distance()` 的结果证明新度量正确。

## 4. 待验证命题与证据链

| ID | 命题 | 主要证据 | 结论边界 |
|---|---|---|---|
| P1 | 任意传统 TSR 都可构造一个集合完全相同的 canonical G-TSR | 构造性证明 + differential tests | 只证明集合与指定兼容 API，不自动证明所有新 API 数值相同 |
| P2 | G-TSR 表示族严格大于传统 TSR 表示族 | 非零有限螺距的一维 screw 反例 + 局部维数/螺距不变量证明 | 数值图只是佐证，不能代替严格证明 |
| P3 | PoE chart 能按声明的生成元、顺序和 frame 产生合法 `SE(3)` 位姿 | 解析特例 + 非交换次序测试 + 有限差分 | 不涉及学习或任务性能 |
| P4 | 通用 `project/distance/contains` 在声明度量下内部一致 | 已知解、1D 全局参考、多起点与 KKT/最优性检查 | 多维非凸投影不宣称总能找到全局最优 |

## 5. 验证总体结构

验证分为五层，后一层不能替代前一层。

| 层级 | 内容 | 主要失败含义 |
|---|---|---|
| L0 契约检查 | shape、单位、frame、action、顺序、周期元数据 | 定义仍有歧义 |
| L1 确定性单元测试 | 手算特例、边界、奇异点、非法输入 | 基础实现错误 |
| L2 差分/性质测试 | 旧 TSR oracle、固定随机种子、变换不变量 | 兼容性或组合语义错误 |
| L3 优化验证 | 已知投影、1D 高精度参考、多起点、最优性 | 投影器或度量实现错误 |
| L4 命题验证 | 嵌入构造、严格包含证明、数值示例 | 论文核心几何主张不成立 |

所有随机测试都必须：

- 使用显式 `numpy.random.Generator`；
- 记录 seed 和失败坐标；
- 一旦发现失败，将该样本固化为确定性回归样本；
- 不允许只靠增加随机样本掩盖确定性边界缺陷。

第一版固定 seeds 为 `[0, 1, 7, 42, 20260902]`。

## 6. 详细测试矩阵

### 6.1 Domain 合同

| ID | 测试 | Oracle/不变量 | 通过条件 |
|---|---|---|---|
| D-01 | `LinearInterval` 内点、端点、外点、零宽区间 | 实数闭区间定义 | 布尔结果完全一致 |
| D-02 | 普通 `CircularInterval` | 显式 wrap 公式 | 端点及 `±2kπ` 等价类一致 |
| D-03 | outer interval `[3π/4,-3π/4]` | 现有 TSR 行为 | `±π` 邻域在内，0 在外 |
| D-04 | 完整 `2π` 区间 | 圆周全域 | 任意有限角都在内 |
| D-05 | 非法区间、NaN、无穷和维数错误 | 输入契约 | 抛出稳定、可诊断的异常 |
| D-06 | screw 坐标的 `q` 与 `q+2π` | 非零螺距 screw 解析式 | 两位姿轴向位移相差 `2πh`，不得被 wrap 为同一点 |
| D-07 | 采样可复现性 | 同一 Generator 状态 | 坐标样本逐元素相同 |

### 6.2 Canonical 前向映射与精确嵌入

构造映射：

\[
E(T_{0,w},T_{w,e},B_w)
=(T_{0,w},T_{w,e},\mathcal C_{\mathrm{TSR}},\mathcal Z(B_w)).
\]

对同一个合法坐标 `z`，比较：

\[
T_{\mathrm{old}}=\operatorname{TSR.to\_transform}(z),
\qquad
T_{\mathrm{new}}=\operatorname{GTSR.to\_transform}(z).
\]

| ID | 场景 | 设计目的 |
|---|---|---|
| E-01 | 全零 bounds、单位外变换 | 固定点最小例 |
| E-02 | 仅 x/y/z 自由 | 平移坐标与单位检查 |
| E-03 | 仅 roll/pitch/yaw 分别自由 | 单轴方向和符号 |
| E-04 | 三个角均非零 | 明确区分 `RzRyRx` 与错误次序 |
| E-05 | 非单位 `T0_w`，包含旋转和平移 | 左侧 frame 语义 |
| E-06 | 非单位 `Tw_e`，包含旋转和平移 | 右侧工具偏置语义 |
| E-07 | 两个外变换同时非单位 | 完整组合链 |
| E-08 | `±π` 邻域、outer interval、完整圆周 | wrap 与端点语义 |
| E-09 | pitch 接近及等于 `±π/2` | 欧拉奇异点；只比较位姿重构 |
| E-10 | 退化与混合自由度 bounds | 零宽和自由坐标共存 |

误差分解为

令 `RΔ=R_oldᵀR_new`，使用近零处更稳定的旋转误差：

\[
e_t=\|p_{\mathrm{old}}-p_{\mathrm{new}}\|_2,
\qquad
e_R=\operatorname{atan2}\!\left(
\tfrac12\|\operatorname{vee}(R_\Delta-R_\Delta^T)\|_2,
\tfrac12(\operatorname{tr}(R_\Delta)-1)
\right).
\]

硬阈值：`max(e_t, e_R) <= 1e-10`。每个确定性场景必须通过；每个 seed 再生成 1000 个合法坐标与外部位姿，共 5000 个差分样本。

### 6.3 Canonical 集合与旧 API 兼容

| ID | 测试 | 通过条件 |
|---|---|---|
| C-01 | 同一坐标注入后得到的 sample pose | 位姿误差 `<=1e-10` |
| C-02 | `contains()` 对内点、端点、容差带内外点 | old/new 布尔结果逐例相同 |
| C-03 | `coordinates -> forward` 往返 | 位姿误差 `<=1e-10`；不要求欧拉角数组唯一 |
| C-04 | legacy `distance()` | 距离和 `bwopt` 重构位姿分别与旧实现一致 |
| C-05 | 部分坐标固定、其余为 NaN 的兼容采样 | 合法性、固定分量及相同底层均匀变量映射一致 |
| C-06 | 序列化往返（若本阶段公开该 API） | 数学区域和枚举元数据不变 |

注意：旧 `contains()` 使用 `EPSILON=1e-3`。兼容测试保留该语义，但矩阵相等、优化收敛和新几何 membership 不得复用这个过宽容差。

不要直接比较两个独立 RNG 调用产生的 sample。应先生成同一组单位超立方体样本 `u`，再比较 old/new 从 `u` 到坐标和位姿的映射。

### 6.4 PoE chart

先实现并验证下列最小特例：

| ID | 场景 | Oracle/不变量 |
|---|---|---|
| G-01 | 单一 x/y/z 平移生成元 | 与解析平移矩阵一致 |
| G-02 | 任意单位轴纯旋转 | Rodrigues 公式 |
| G-03 | 坐标原点不在世界原点的旋转轴 | 零螺距 screw 解析式 |
| G-04 | 非零有限螺距 screw | 旋转角为 `q`，轴向位移为 `hq` |
| G-05 | 两个不对易生成元 | `exp(ξ1a)exp(ξ2b)` 与反序结果显著不同 |
| G-06 | space/body 表达约定 | 与显式矩阵链和 Adjoint 变换结果一致 |
| G-07 | 外部 `T0/Te` 组合 | `T0 @ chart.forward(z) @ Te` 精确一致 |
| G-08 | 输出合法性 | `RᵀR=I`、`det(R)=1`、末行 `[0,0,0,1]` |
| G-09 | chart Jacobian | 解析/自动 Jacobian 与中心有限差分一致 |

合法性残差阈值为 `1e-12`。非交换测试不能只检查“不相等”，还必须用显式解析矩阵作 oracle；区分度应大于数值噪声至少六个数量级。

### 6.5 `project / distance / contains`

通用投影问题为

\[
z^*=\arg\min_{z\in\mathcal Z} d_W(T(z),T_q)^2.
\]

| ID | 测试 | 通过条件 |
|---|---|---|
| P-01 | 查询位姿由内部坐标生成 | `distance<=1e-10` 且重构位姿一致 |
| P-02 | 零宽 domain | 投影恒为唯一固定位姿 |
| P-03 | canonical chart | 与解析裁剪或 legacy 路径按各自度量分别对照 |
| P-04 | 1D 平移/纯旋转 | 与解析投影一致 |
| P-05 | 1D 非零螺距 screw | 与高精度有界标量优化参考一致 |
| P-06 | 投影幂等性 | `project(project(T))` 的位姿变化 `<=1e-10` |
| P-07 | distance/contains 一致性 | 新语义下 `contains(T)` 当且仅当投影残差在声明容差内 |
| P-08 | 多生成元非凸例 | 固定多起点；检查最优性和相对参考目标值 |
| P-09 | 优化失败传播 | 返回 `success=False` 与诊断信息，不伪装为有效投影 |

一维参考使用高精度标量方法，不以被测优化器自身作为 oracle。多维情形至少满足：

- bounds 可行性误差 `<=1e-12`；
- 投影梯度/KKT 残差 `<=1e-6`；
- 目标值不劣于固定多起点参考最优值 `1e-8` 以上；
- 所有失败必须显式暴露，不能仅凭优化器返回“converged”判定正确。

### 6.6 严格表示能力

#### 理论对象

取一维非零有限螺距 screw：

\[
\mathcal S_h=\{\exp(\hat\xi_h q)\mid q\in[a,b]\},
\qquad
\omega\ne0,\quad
h=\frac{\omega^Tv}{\|\omega\|^2}\ne0,
\quad a<b.
\]

证明分两步：

1. `PoEChart([ξ_h])` 和一维线性 domain 可直接表示 `S_h`，所以 `S_h` 属于 G-TSR 表示族；
2. 若传统 TSR 表示同一一维集合，则在欧拉 chart 的正则邻域中只能有一个非退化坐标：
   - 单一平移坐标产生纯平移，即无限螺距；
   - 单一旋转坐标经固定外变换后仍是零螺距旋转，因为 screw pitch 在刚体共轭下不变；
   - 两个及以上独立非退化坐标在正则点具有至少二维局部像，不能等于一维曲线。

还必须单独排除“把传统 TSR 永久放在欧拉奇异面上”的规避方式：若 pitch 区间非退化，则区域必然包含正则点；若 pitch 固定为 `±π/2`，roll/yaw 的退化组合最多产生一维纯旋转像，仍为零螺距，而加入任一独立平移区间又会把局部像提升到至少二维。因此奇异参数化也不能生成非零有限螺距的一维曲线。

因此非零有限螺距的一维曲线不能由传统 axis-aligned TSR 精确表示，得到表示族的严格包含。

#### 数值验证

数值部分只验证构造和展示差异，不承担“遍历所有传统 TSR”的证明责任：

- 设轴为 z，`h=0.05 m/rad`，`q∈[-0.5,0.5] rad`；
- 检查相邻相对变换估计出的 pitch 为常数，误差 `<=1e-8 m/rad`；
- 检查 `q+2π` 相对 `q` 产生 `2πh` 的轴向位移；
- 生成传统独立 `z-yaw` box 的采样云，与 screw 曲线画在同一图中；
- 报告曲线拟合误差、box 的多余自由体积或横向偏离，但不得将有限采样的拟合失败写成严格不可表示的唯一依据。

## 7. 固定测试数据集

确定性 fixture 至少包括：

1. 单位变换；
2. 纯平移外 frame；
3. 纯旋转外 frame；
4. 同时包含平移和旋转的 `T0_w`、`Tw_e`；
5. `roll/pitch/yaw` 均非零且次序敏感的坐标；
6. `π-δ`、`-π+δ`、`±π/2` 和 `±π/2±δ`，其中 `δ∈{1e-12,1e-9,1e-6}`；
7. 零宽、内区间、outer interval 和完整圆周 domain；
8. 平移、纯旋转、零螺距旋转、非零螺距 screw、两个不对易生成元。

随机 SE(3) fixture 必须通过合法旋转参数化生成，不得用任意 `4×4` 随机矩阵冒充刚体变换。

## 8. 代码与测试落点

建议保持几何内核独立，第一版文件布局如下：

```text
src/tsr/gtsr/
  __init__.py
  domains.py
  charts.py
  region.py
  metrics.py
  projection.py

tests/gtsr/
  conftest.py
  test_domains.py
  test_canonical_embedding.py
  test_canonical_compatibility.py
  test_poe_chart.py
  test_projection.py
  test_expressivity.py
```

旧 `TSR` 在几何冻结前保持不变，作为 differential oracle。不要先重构旧实现再验证新实现，否则 oracle 会同时移动。

## 9. 实施顺序

### Sprint A：canonical 嵌入

1. 实现 domain 基类及线性/周期区间；
2. 实现 `GeneratorChart` 最小协议；
3. 实现 `CanonicalTSRChart.forward()`；
4. 实现 `GTSR.to_transform()`；
5. 完成 D-01 至 D-07、E-01 至 E-10；
6. 完成 C-01 至 C-05。

退出条件：硬门 G1 通过。

### Sprint B：通用几何

1. 实现 `PoEChart`；
2. 实现显式 metric；
3. 实现通用 `project/distance/contains`；
4. 完成 G-01 至 G-09、P-01 至 P-09。

退出条件：硬门 G2 通过。

### Sprint C：严格能力与冻结

1. 写出 P1 构造性证明；
2. 写出 P2 的正则邻域、维数和 pitch 不变量证明；
3. 实现 screw 数值验证与论文示意图；
4. 运行完整回归、归档 seeds、环境和结果表。

退出条件：硬门 G3 通过，几何内核冻结。

## 10. 失败诊断规则

| 失败模式 | 优先检查 |
|---|---|
| 前向映射不一致 | RPY 乘法顺序、数组索引、左右乘、frame 定义 |
| 只有 `Tw_e` 非单位时失败 | 右侧工具偏置是否被错误地前乘或求逆 |
| `±π` 附近失败 | 周期域与线性域混用、outer interval 展开 |
| 集合测试通过但 distance 不同 | 是否错误地把 legacy 坐标距离和新 Log 度量视为同一量 |
| 欧拉角数组不同但位姿相同 | 多值表示，不应判失败 |
| 投影偶发失败 | 初始化、Jacobian、局部极小和 bounds 活跃约束 |
| screw 被错误 wrap | 仅看到旋转生成元，忽略了非零 pitch 的平移 |
| 数值拟合看似能逼近 screw | 逼近不等于集合精确相等；回到 pitch 不变量与局部维数证明 |

禁止通过单纯放宽阈值处理失败。阈值变更必须记录根因、量纲和对论文命题的影响。

## 11. 三个硬门

### G1：传统 TSR 精确嵌入

必须全部满足：

- E-01 至 E-10 全通过；
- 5000 个差分样本的 `max(e_t,e_R)<=1e-10`；
- contains 边界语料与旧实现逐例一致；
- legacy distance 兼容路径通过；
- P1 有完整构造映射，且代码对象逐项对应数学符号。

### G2：任意 chart 的几何闭环

必须全部满足：

- G-01 至 G-09 的 PoE 特例、非交换顺序、frame/action 和 Jacobian 测试通过；
- `sample -> contains`、`project -> contains`、投影幂等性通过；
- 1D screw 投影与独立高精度参考一致；
- 多维优化满足可行性和最优性门槛，失败状态可见。

### G3：严格表示能力

必须全部满足：

- 明确区分单个区域 `R` 与表示族 `𝔉`；
- 完成 `𝔉_TSR ⊂ 𝔉_GTSR` 的严格证明；
- 非零有限螺距 screw 构造、周期性和数值曲线测试通过；
- 数值图不承担超出其证据能力的理论结论。

任一硬门失败，均不得开始 `P_r(s)`、phase estimator 或 transfer 实验。

## 12. 本阶段交付物

- 可运行的 `tests/gtsr/` 测试集；
- 一页 convention 表：坐标顺序、组合顺序、frame、action、periodicity、单位；
- P1 精确嵌入证明；
- P2 严格包含证明；
- canonical differential test 结果表；
- screw case 的曲线/box 对比图及生成脚本；
- 失败样本回归清单和固定 seeds；
- 几何内核冻结记录，列出最终 API、度量和容差。

## 13. 当前立即执行项

下一步只启动 Sprint A。首批提交应包含 `domains.py`、`charts.py`、`region.py` 和前三个测试文件，不包含 PoE、优化器或自适应模块。首要验收不是“代码能跑”，而是 canonical G-TSR 在非单位外 frame、旋转次序、wrap 边界和欧拉奇异点上都与当前 TSR oracle 一致。
