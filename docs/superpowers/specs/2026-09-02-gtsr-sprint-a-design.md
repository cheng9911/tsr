# G-TSR Sprint A — 实现级设计（canonical 嵌入）

状态：`APPROVED`
日期：`2026-09-02`
上游：`docs/gtsr_geometry_validation_plan.md`（§8、§9 Sprint A、§11 硬门 G1）

本文档把上游验证方案的"数学契约"落到"代码级 API"。范围严格限定在 **Sprint A**：
canonical 嵌入 + 旧 TSR 精确兼容。不实现 PoE、metric、project、严格证明。

## 1. 范围与不变量

**本 Sprint 交付：**

- `src/tsr/gtsr/domains.py`、`charts.py`、`region.py`、`__init__.py`
- `tests/gtsr/conftest.py`、`test_domains.py`、`test_canonical_embedding.py`、`test_canonical_compatibility.py`
- 硬门 G1 全绿。

**不变量（贯穿全部测试）：**

- 旧 `TSR` 保持不动，作为 differential oracle。
- 位姿误差用 §6.2 的 `e_t` + `atan2` 形式 `e_R`，硬阈值 `max(e_t, e_R) <= 1e-10`。
- 随机测试一律显式 `numpy.random.Generator`，固定 seeds `[0, 1, 7, 42, 20260902]`。

**明确不做：** PoEChart、显式 metric、通用 `project`、P1/P2 严格证明、序列化（C-06）、
把 `GTSR` 挂到顶层 `tsr/__init__.py`。

## 2. 约定表（convention）

| 项 | 约定 |
|---|---|
| 坐标数组 | `[x, y, z, roll, pitch, yaw]`，长度 6，float64 |
| 旋转组合 | `Rz(yaw) @ Ry(pitch) @ Rx(roll)`（矩阵乘法顺序）|
| 局部位姿 | `Trans(x,y,z) @ Rz(yaw) @ Ry(pitch) @ Rx(roll)` |
| 外部组合 | `T0_w @ T_local @ Tw_e` |
| 变换类型 | 4×4 numpy 齐次矩阵，末行 `[0,0,0,1]` |
| 周期轴 | 索引 3,4,5（roll/pitch/yaw），period `2π` |
| 线性轴 | 索引 0,1,2（x/y/z）|
| 容差 | legacy `contains/distance` 用 `utils.EPSILON=1e-3`；域 membership、矩阵相等用干净语义 |
| 随机 | `numpy.random.Generator`，seeds `[0,1,7,42,20260902]` |

## 3. 模块设计

### 3.1 domains.py — 单一事实源（bounds）

chart 拥有"拓扑"（名字、生成元、顺序、frame、周期元数据），**domain 唯一拥有 bounds**。
domain 不保存一份可变 bounds 于 chart 中（§3.2）。

本 Sprint 所有区间均为**闭区间**（D-01 明确"实数闭区间定义"），不引入开区间/端点闭合参数（YAGNI）。

```python
class Interval(ABC):
    periodic: bool
    period: float | None
    def contains(self, x: float) -> bool: ...
    def map_from_unit(self, u: float) -> float: ...

class LinearInterval(Interval):
    # periodic=False；clean 闭区间 lower <= x <= upper（无 EPSILON）
    def __init__(self, lower, upper): ...

class CircularInterval(Interval):
    # periodic=True, period=2π
    # 构造时把 raw(lower,upper) 一次性转连续区间：
    #   lo_cont = wrap_to_interval(lower)                  # → [-π, π)
    #   interval = upper - lower; if interval<0: += 2π; interval = min(interval, 2π)
    #   hi_cont = lo_cont + interval
    # contains(x) = lo_cont <= wrap_to_interval(x, lower=lo_cont) <= hi_cont
    def __init__(self, lower, upper, period=2*pi): ...

class CoordinateDomain:
    def __init__(self, intervals: Sequence[Interval]): ...
    @property dim -> int
    def contains(self, z) -> np.ndarray[bool]         # 逐轴 clean
    def map_from_unit_cube(self, u) -> np.ndarray     # 确定性 lo + (hi-lo)*u，周期轴再 wrap
    def sample(self, rng: np.random.Generator) -> np.ndarray
    @property cont_lower / cont_upper                 # 连续边界（legacy 路径复用）
    @classmethod from_bounds(cls, Bw, periodic, periods) -> CoordinateDomain
```

`from_bounds(Bw, periodic, periods)`：对 `periodic[i]==False` 构造 `LinearInterval`，
否则 `CircularInterval`；平移轴若 `lo > hi` 抛 `ValueError`（复刻旧 TSR 行为）。

### 3.2 charts.py — 前向与逆映射

```python
class GeneratorChart(ABC):
    coord_names: tuple[str, ...]
    periodic: tuple[bool, ...]
    periods: tuple[float, ...]
    @property dim -> int
    @abstractmethod forward(self, z) -> np.ndarray   # (4,4)

class CanonicalTSRChart(GeneratorChart):
    coord_names = ("x","y","z","roll","pitch","yaw")
    periodic = (False, False, False, True, True, True)
    periods = (None, None, None, 2*pi, 2*pi, 2*pi)
    def forward(self, z) -> np.ndarray                # xyzrpy_to_trans
    def coordinates(self, T_local, domain) -> np.ndarray  # 逆映射
    # 旧 TSR 的欧拉角数学整体迁入（canonical 专属）：
    @staticmethod rot_to_rpy(rot) -> rpy
    @staticmethod rpy_to_rot(rpy) -> rot
    @staticmethod xyz_within_bounds(xyz, bounds) -> bool[3]      # EPSILON
    @staticmethod rpy_within_bounds(rpy, bounds) -> bool[3]      # EPSILON
    @staticmethod rot_within_rpy_bounds(rot, rpy_bounds) -> (bool[3], rpy|None)  # EPSILON
```

`coordinates` 复刻旧 `to_xyzrpy`：`xyz` 直接取；`rpy` 走 `rot_within_rpy_bounds`
（找到 bounds 一致解则用之，否则 fallback `rot_to_rpy`）。

### 3.3 region.py — 区域级组合

```python
class GTSR:
    def __init__(self, T0_w, Tw_e, chart, domain): ...
    @classmethod
    def canonical(cls, T0_w=None, Tw_e=None, Bw=None) -> GTSR
        # 即 §6.2 构造映射 E(T0_w,Tw_e,Bw) = (T0_w,Tw_e,C_TSR,Z(Bw))
        # chart=CanonicalTSRChart(); domain=CoordinateDomain.from_bounds(Bw, chart.periodic, chart.periods)

    def to_transform(self, z) -> np.ndarray      # 校验 z∈domain → T0_w @ forward(z) @ Tw_e
    def to_xyzrpy(self, T) -> np.ndarray         # inv(T0_w)@T@inv(Tw_e) → chart.coordinates
    def is_valid(self, z, ignoreNAN=False) -> bool[6]
    def contains(self, T) -> bool                # legacy：xyz_within_bounds + rot_within_rpy_bounds（EPSILON）
    def distance(self, T, rotation_weight=1.0) -> (float, np.ndarray)  # legacy 解析裁剪
    def sample_xyzrpy(self, partial=NANBW) -> np.ndarray
    def sample(self, partial=NANBW) -> np.ndarray
```

**兼容策略（关键）：**

- `GTSR.contains` **不复用** clean `domain.contains`，而是走 chart 的 `*_within_bounds`
  （带 `EPSILON=1e-3`）+ domain 连续边界，逐位复刻旧 `TSR.contains`。
- clean `domain.contains` 留给 D-01~D-07 与后续 Sprint B 的通用 membership。
- `distance` 完整重写 `_displacement_to_tsr`（9 个 rpy 冗余候选）+ `bwopt` clip/wrap。
- `sample_xyzrpy` 走 `domain.map_from_unit_cube(u)`，不各自调 RNG。

## 4. 数据流

```
z ──to_transform──> forward(z) ──> T0_w @ . @ Tw_e ──> T
T ──contains/to_xyzrpy──> inv(T0_w)@T@inv(Tw_e) ──> coordinates ──> z / 布尔
u ──map_from_unit_cube──> z（周期轴 wrap）
```

## 5. 测试矩阵（对应文档 §6）

- `test_domains.py`：D-01~D-07。
- `test_canonical_embedding.py`：E-01~E-10（前向映射差分）+ 每 seed 1000、共 5000 差分样本。
- `test_canonical_compatibility.py`：C-01~C-05（contains 边界语料、distance、采样映射、往返）。

`conftest.py` 提供：固定 `SEEDS`、合法 `SE(3)` 生成器（QR/Gram-Schmidt，非任意 4×4）、
位姿误差助手（`e_t` + `atan2 e_R`）、`rng` fixture、old/new 构造助手。

## 6. 退出条件（硬门 G1）

- E-01~E-10 全通过；
- 5000 差分样本 `max(e_t, e_R) <= 1e-10`；
- contains 边界语料与旧实现逐例一致；
- legacy distance 兼容路径通过；
- P1 构造映射（`GTSR.canonical`）代码对象逐项对应数学符号 `T0↔T0_w, Te↔Tw_e, C_TSR↔CanonicalTSRChart, Z(Bw)↔CoordinateDomain`。
