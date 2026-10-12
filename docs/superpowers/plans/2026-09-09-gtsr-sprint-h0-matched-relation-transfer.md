# G-TSR Sprint H0 — Matched-Relation Transfer Contract Implementation Plan

状态：`IMPLEMENTATION-COMPLETE`（H0-01–H0-11 验收通过；未生成 transfer 实验数据）

执行结果：H0 59 passed；全量 610 passed、1 xfailed、1 warning；独立复审无剩余阻断项。
验收版本与偏离记录见 [H0 acceptance record](../../gtsr_transfer_contract_record.md)。

**Goal:** 实现一个纯数学、无 ManiSkill 依赖的 matched-relation transfer layer：验证 provided
`H/Gamma/phase correspondence`，拒绝不匹配或超 support 的 query，并在通过时把冻结的 diagonal
`ResponseProfile6D` 搬运到 target generator coordinates。H0 只锁合同，不声称 transfer 有效。

**Spec:** `docs/superpowers/specs/2026-09-09-gtsr-sprint-h0-matched-relation-transfer-contract.md`

**Architecture:** 新增 `src/tsr/gtsr/transfer.py` 和一个集中测试文件
`tests/gtsr/test_transfer_contract.py`。底层 primitive（Adjoint、monomial、phase map、support、residual）
先独立 TDD，再组装 `MatchedRelationTransfer.query()`。不修改 `ResponseOperator`、`ResponseProfile6D`、
几何内核、Sprint G observer 或 legacy TSR。

## Global constraints

- 只支持 `B_s,B_t in GL(6)`；rectangular/pseudo-inverse 必须拒绝。
- twist order 固定 `[v_x,v_y,v_z,w_x,w_y,w_z]`。
- `Gamma` 只能从 permutation + nonzero scales 构造，不接受任意 matrix。
- `H` 使用 active conjugation：`T_t=H T_s H^-1`，Adjoint 为
  `[[R, skew(p)R],[0,R]]`；这里只是共轭 relation/reference 配对，不泛指两物体的绝对 pose。
- 接入 query 的 coordinates 必须来自 `solve(B, Log(T_ref^-1 T))`，不是原始 XYZ/RPY。
- `P_r` 是 response operator，不是概率；`alpha_max` 不擅自锁为 1。
- query 必须先通过静态合同，再通过 `Gamma^-1 c_t` range support；失败返回结构化状态且
  `alpha_target=None`，禁止 silent fallback。
- target `Z_t` 不属于本模块输入或输出。
- source profile 只读查询，不改参数；所有内部数组 defensive copy。
- 浮点测试固定 RNG seed `20260909`。
- H0 不加载 HDF5、不访问 sister repo、不产生 H1/H2/H3 结果。

## Deliverables

| 文件 | 交付 |
| --- | --- |
| `src/tsr/gtsr/transfer.py` | H0 全部 public types/primitives/high-level query |
| `src/tsr/gtsr/__init__.py` | 只导出审定后的 H0 public API |
| `tests/gtsr/test_transfer_contract.py` | H0-01–H0-11 单元/性质测试 |
| H0 spec | 对齐实际 API 和声明边界，复审/验证后改为 `TRANSFER-CONTRACT-LOCKED` |

## Task 1 — SE(3) Adjoint 与物理 operator primitives（H0-05/H0-06）

**Red tests**

- `se3_adjoint(I) == I6`；
- 非零平移 `H` 的 upper-right block 等于 `[p]_x R`；
- 固定随机 `xi`：
  `se3_exp(Ad_H @ xi) == H @ se3_exp(xi) @ inv(H)`，误差 `<=1e-10`；
- `physical_response_operator(B, alpha)` 拒绝非 `(6,6)`、非有限、rank-deficient `B` 和非 `(6,)`
  `alpha`；
- 返回 `B @ diag(alpha) @ inv(B)`，且输入不被修改。

**Implementation**

```python
def se3_adjoint(H): ...
def physical_response_operator(basis, alpha): ...
```

复用 `._validation.se3_transform/finite_matrix/finite_vector`，不要在 transfer 模块复制另一套 SE(3)
验证。`se3_adjoint` 必须测试 translation–rotation cross term，不能只测纯旋转。

**Verify**

```bash
pytest tests/gtsr/test_transfer_contract.py -k "adjoint or physical_response" -q
```

## Task 2 — Monomial correspondence 与 generator semantics（H0-01/H0-02/H0-03）

**Red tests**

- permutation 必须恰为 `0..5` 的双射；duplicate/out-of-range/长度错误均拒绝；
- scales 长度 6、finite、nonzero；允许负号；
- `matrix[i, permutation[i]] == scales[i]`，每行/列恰一非零；
- 100 个固定随机 `c_s` 做 source→target→source roundtrip `<=1e-12`；
- 100 个固定随机 diagonal `alpha`：`Gamma diag(alpha) Gamma^-1` off-diagonal
  `<=1e-12`，target diagonal 等于 `alpha[permutation]`；
- `GeneratorSemantics` 长度必须为 6；映射后的 task-function IDs 不同不抛构造异常，而由
  `matches(correspondence)==False` 供 match gate 返回 `UNMATCHED_FAMILY`；
- 暴露的 matrix/arrays 是 copy，外部修改不改变对象。

**Implementation**

```python
@dataclass(frozen=True)
class MonomialCorrespondence: ...

@dataclass(frozen=True)
class GeneratorSemantics: ...

def transport_diagonal(alpha_source, correspondence): ...
```

`transport_diagonal` 直接按 permutation 重排，不通过 dense inverse 计算生产结果；dense similarity
只用于性质测试。

**Verify**

```bash
pytest tests/gtsr/test_transfer_contract.py -k "monomial or semantics or diagonal" -q
```

## Task 3 — Oracle phase correspondence（H0-08）

**Red tests**

- 四 phase identity map 的所有端点/中点保持不变；
- piecewise-linear knots，例如 target `[0,.25,1]` → source `[0,.5,1]`，插值正确；
- source/target knots 必须从 0 开始、到 1 结束、长度相同且严格递增；
- phase map 必须是保持有向链顺序的 bijection；duplicate、逆序、缺 phase 拒绝；
- source/target phase semantic IDs 映射后不一致时 `semantics_match=False`；
- `rho` 超 `[0,1]`、非 finite、phase 越界为调用错误并抛 `ValueError`；
- map 不读取 wall-clock time、未来帧或 response error。

**Implementation**

```python
@dataclass(frozen=True)
class PiecewiseLinearPhaseMap:
    ...
    def map(self, target_phase, target_progress): ...
    def semantics_match(self): ...
```

使用 `np.interp` 前完成严格 knot 验证；内部保存 tuple，不保留调用者的可写数组。

**Verify**

```bash
pytest tests/gtsr/test_transfer_contract.py -k phase -q
```

## Task 4 — Coordinate range support（H0-09）

**Red tests**

- lower/upper 长度 6、finite 且 `lower<=upper`；tolerance finite 且 `>=0`；
- closed bounds 内和恰在边界返回 true；超过 tolerance 返回 false；
- negative correspondence scale 和 axis permutation 后，必须检查
  `c_s=Gamma^-1 c_t`，不能直接检查 `c_t`；
- 返回 mapped source coordinates，便于审计；
- 文档/类型名保持 `RangeSupport`，不称作 density/distribution support。

**Implementation**

```python
@dataclass(frozen=True)
class CoordinateRangeSupport:
    ...
    def contains_source(self, coordinates): ...
```

**Verify**

```bash
pytest tests/gtsr/test_transfer_contract.py -k support -q
```

## Task 5 — Compatibility residual 与静态 match gate（H0-04/H0-05/H0-07）

**Red tests**

- canonical `B_s=B_t=I,H=I,Gamma=I` 的 total/axis residual 近机器零；
- 构造 target basis：
  `B_t = Ad_H @ B_s @ inv(Gamma)`，含 permutation/sign/scale 和带平移 `H`，residual
  `<=1e-12`；
- 单独扰动 target basis 一列时，对应 axis residual 超门，返回 `INCOMPATIBLE_BASIS`；
- total 通过但任一 axis 不通过仍拒绝；
- 用正 diagonal `Q_s,Q_t` 做一致 coordinate 单位重参数化
  `B_s'=B_s Q_s^-1, B_t'=B_t Q_t^-1, Gamma'=Q_t Gamma Q_s^-1,
  S_c'=Q_s S_c` 不改变 residual/verdict；
- generator 或 phase semantics 错配优先返回 `UNMATCHED_FAMILY`；
- thresholds 必须 finite/nonnegative，axis thresholds shape `(6,)`。

**Implementation**

```python
class MatchStatus(Enum): ...
@dataclass(frozen=True)
class MatchResult: ...

def compatibility_residual(...): ...

class MatchedRelationTransfer:
    def evaluate_contract(self): ...
```

冻结判定优先级：输入合同错误抛 `TypeError/ValueError`；合法但语义不匹配
`UNMATCHED_FAMILY`；语义匹配但 basis residual 失败 `INCOMPATIBLE_BASIS`；全部通过才
`MATCHED`。不得通过异常捕获把程序错误改成 abstention。

**Verify**

```bash
pytest tests/gtsr/test_transfer_contract.py -k "compatibility or contract_match" -q
```

## Task 6 — Query gate 与 transferred alpha（H0-08/H0-09/H0-10）

**Red tests**

- 静态合同失败时 query 返回相同 failure status，`alpha_target=None`；
- 静态合同通过但 target coordinates 超 support 时返回 `OUT_OF_SUPPORT`；
- matched identity phase/correspondence 对 `ResponseProfile6D.alpha(q,rho)` 完全一致；
- permutation/sign/scale 下 target alpha 只按 permutation 重排，scale 不改变 alpha；
- piecewise phase warp 先映射 `(q_t,rho_t)`，再查询 source profile；
- `TransferQueryResult` 同时保存 source phase/progress/coordinates 和完整 `MatchResult`；
- source profile、theta、contract、输入 coordinates 均不 mutate；
- 失败时不返回 Frame/Fixed 或上一次 alpha；
- public query signature 不含 `GTSR`/`CoordinateDomain`/`Z_s` 参数。

**Implementation**

```python
@dataclass(frozen=True)
class TransferQueryResult: ...

class MatchedRelationTransfer:
    def evaluate_query(self, target_coordinates): ...
    def source_phase(self, target_phase, target_progress): ...
    def query(self, source_profile, target_phase, target_progress,
              target_coordinates): ...
```

首版 `source_profile` 明确接受 `ResponseProfile6D`，避免用未受约束的 duck typing 扩大合同。

**Verify**

```bash
pytest tests/gtsr/test_transfer_contract.py -q
```

## Task 7 — Public exports、Proposition differential tests 与收口（H0-06/H0-11）

**Red tests / checks**

- 500 个固定随机、condition number 受控的 exact-compatible tuple
  `(B_s,H,Gamma,alpha)`，构造 `B_t=Ad_H B_s Gamma^-1`，验证：

  ```text
  K_t == Ad_H K_s Ad_H^-1
  max matrix error <= 1e-10
  ```

- 同批样本验证 `Exp(Ad_H xi)==H Exp(xi) H^-1`；
- `tsr.gtsr` 导出：`MatchStatus`、`MonomialCorrespondence`、`GeneratorSemantics`、
  `PiecewiseLinearPhaseMap`、`CoordinateRangeSupport`、`MatchResult`、`TransferQueryResult`、
  `MatchedRelationTransfer`；
- 检查 `src/tsr/gtsr/response.py`、`profile6.py`、geometry kernel 和 legacy `src/tsr/tsr.py`
  未被修改；
- 全量回归。

**Verify**

```bash
pytest tests/gtsr/test_transfer_contract.py -q
pytest -q
ruff check src/tsr/gtsr/transfer.py src/tsr/gtsr/__init__.py tests/gtsr/test_transfer_contract.py
ruff format --check src/tsr/gtsr/transfer.py src/tsr/gtsr/__init__.py tests/gtsr/test_transfer_contract.py
git diff --check
```

若 `ruff` 不在 `tsr-dev` 环境，交付记录必须明确写“未运行”，不得写 clean。

## Hard-gate traceability

| Spec gate | Plan task |
| --- | --- |
| H0-01 | Task 2 |
| H0-02 | Task 2 |
| H0-03 | Task 2 + Task 5 |
| H0-04 | Task 5 |
| H0-05 | Task 1 + Task 5 |
| H0-06 | Task 1 + Task 7 |
| H0-07 | Task 5 |
| H0-08 | Task 3 |
| H0-09 | Task 4 + Task 6 |
| H0-10 | Task 6 |
| H0-11 | Task 7 |

## Commit sequence

每个 Task 独立提交，建议消息：

1. `feat(gtsr): add SE(3) transfer adjoint primitives`
2. `feat(gtsr): add monomial generator correspondence`
3. `feat(gtsr): add oracle phase correspondence`
4. `feat(gtsr): add transfer coordinate support gate`
5. `feat(gtsr): add matched-family compatibility diagnostics`
6. `feat(gtsr): add guarded diagonal-law transfer query`
7. `test(gtsr): close H0 matched-transfer contract`

不要在 Task 7 自动提交/推送 H1 数据或 Sprint G 实现。H0 review 通过后，单独将 spec 状态改为
`TRANSFER-CONTRACT-LOCKED` 并记录实现 commit/hash。

执行偏离记录（2026-09-09）：本轮在一个工作区按完整测试先 red（缺少 transfer 模块）、
实现后 green 的方式覆盖七项任务，未逐 Task 制造独立 red/green 或提交记录。
当前交付保留为未提交改动；最终验收记录以 base HEAD 与文件 SHA-256 标识，不虚构实现 commit。
补充 profile phase-count/参数检查、Log-coordinate 约定和归一化 overflow 拒绝测试，
不改既有 profile/response 的合同或实现。

## Completion boundary

H0 完成只允许报告：

> The implementation satisfies the provided matched-family transfer contract and its frame-equivariance
> tests.

不得报告 transfer accuracy、Sim-to-Real gain、real intervention reduction 或 task success；这些均需
H1–H3/I 的独立数据。
