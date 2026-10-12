# G-TSR H1 controlled-reuse bridge record

日期：2026-09-10
状态：`H1-CONTROLLED-REUSE-LOCKED`

本记录是 H1-B（`docs/superpowers/plans/2026-09-10-gtsr-sprint-h1b-controlled-reuse-bridge.md`）
执行完成后的实现与验证快照。它把 sister repo 已有5对 controlled-pose transfer artifact 接入 H0 的
显式 family/basis/phase/support gate，生成只读结果索引与75条件账本；**没有**重跑 benchmark、拟合、
构造环境、写 HDF5 或启动 collector，也没有新增经验迁移证据。

本状态的上限就是 `H1-CONTROLLED-REUSE-LOCKED`，不是 `TRANSFER-VALIDATED`。独立 physical Sim→Sim
验证仍需另立 annex。

## 1. 冻结的5对 target

| pair_key | source → target | target nominal_pose6 [m,rad] |
| --- | --- | --- |
| `drawer_to_plate_push` | drawer_middle_open → plate_front_push | [0.10, 0, 0, 0, 0, 0] |
| `knob_to_microwave_door` | stove_knob_turn → microwave_door_revolute | [0, 0, 0, 0, 0, −1] |
| `bowl_stove_to_bowl_plate` | bowl_on_stove → bowl_on_plate | [0.07, −0.02, 0.055, 0, 0, 0] |
| `bowl_stove_to_cream_cheese_bowl` | bowl_on_stove → cream_cheese_in_bowl | [0.07, 0.02, 0.045, 0, 0, 0] |
| `bowl_stove_to_wine_cabinet` | bowl_on_stove → wine_bottle_on_cabinet | [0.08, 0, 0.10, 0, 0, 0] |

allowlist 先于 H0，且是**结构性强制**而非旁路检查：`query_grid`/`bridge_action` 只接受经 allowlist
验证的 `PairBridge`（由 `build_pair_bridge` 产出），不接受裸 `MatchedRelationTransfer`；不提供 pair 身份
的裸 contract 无法搬运 alpha。`build_pair_bridge` 对不在清单内的 `(source,target)` 返回
`UNMATCHED_FAMILY` 型实验层拒绝，不因六轴同名而放行；self-pair 与 reversed pair 都被拒绝（负例测试覆盖）。

## 2. 两个冻结比较 run 与预算

| run | 行数 | 结构 | N |
| --- | ---: | --- | --- |
| `geometry_transfer` | 180 | 5 pairs × 3 seeds × 3 N × 1 repeat × 4 methods | 3/5/8 |
| `geometry_transfer_tpgmm_n8` | 75 | 5 pairs × 3 seeds × 1 N × 1 repeat × 5 methods | 8 |

split 由冻结算法物化（`np.random.default_rng(20260831)`，pair→seed→N→repeat=0 推进 RNG；
`train=sort(choice(60,N,replace=False))`），不运行拟合。物化结果：主run 45 个 cell、补充run 15 个
cell，键唯一，跨run不配对（N=8 在两个 run 的 train 集不同，RNG 消耗不同）。

- `derived_n_eval = 60 − N`，即 57/55/52；这是正确的评价数。
- 历史 transfer CSV 的 `target_interventions_for_validation = N` 与代码不符；新索引把该字段原样保留
  为 `historical_csv_target_interventions_for_validation`，并在旁边列正确派生值 `derived_n_eval`。
- 索引同时保留 `historical_csv_target_interventions_for_fit` 与 `historical_csv_source_sample_size`：
  transfer 行为 `0` / `30`，scratch 行为 `N` / `0`。原 CSV 未改写。
- 索引在读取时强校验 CSV SHA（audit 快照）、source/target 元数据与 manifest allowlist 一致、
  每 cell 恰好包含冻结方法集、无重复行、无缺失/多余 cell；任一漂移即停止。

## 3. H0 合同工厂与查询 gate

`build_contract` 构造 `MatchedRelationTransfer`：`Bs = Bt = I6`、`H = I4`、`Γ = I6`（identity
correspondence）、六个 relation semantic ID、四 phase identity map、manifest 冻结 support 盒与
total/axis tolerance 1e-10。静态 `evaluate_contract()` 返回 `MATCHED`；basis 篡改 → `INCOMPATIBLE_BASIS`，
phase 语义篡改 → `UNMATCHED_FAMILY`（测试覆盖）。

`query_grid` 先以历史 `Trans(xyz) Rx Ry Rz` 转换并 `Log`，再依次 `evaluate_contract()` →
`source_phase()` → `evaluate_query()`；只有 `MATCHED` 才搬运已审计 grid alpha。拒绝返回 `alpha=None`。
`transport_diagonal` 仅在完整 gate 通过后被调用（identity permutation 下的代数搬运），不构成绕过 gate 的
单独入口。`PairBridge.source_grid` 存为私有只读副本，调用者改动原数组不会泄漏进 bridge 状态。

## 4. 75-condition 账本

对每个 pair 的 75 个 condition（60 mixed + baseline + 14 isolated）：

- 65 `MATCHED`（60 mixed + baseline + 4 isolated）
- 10 `OUT_OF_SUPPORT`（condition_id `[1,2,3,4,5,6,8,9,11,14]`）

`OUT_OF_SUPPORT` 是合同拒绝，单列 coverage，不把误差置零、不从分母删除。5 个 pair 的账本逐 pair
验证一致。

## 5. 结果索引（描述性比较，非新实验）

transfer `e_task_mean`（主run按scratch N分组）：

| scratch N | frozen transfer mean | count |
| --- | ---: | ---: |
| 3 | 1.8247556345547286e-7 | 15 rows |
| 5 | 1.878916112704118e-7 | 15 rows |
| 8 | 1.8844053700248234e-7 | 15 rows |

逐 cell 只读比较 transfer 与 Pdiag finite scratch：**41/45 较低、4/45 较高**；15 个 pair×N 宏均值
全部较低。不是“所有 cell 都赢”，反向 cell 保留：

| pair_key | transfer 较低 | transfer 较高 |
| --- | ---: | ---: |
| drawer_to_plate_push | 7 | 2 |
| knob_to_microwave_door | 8 | 1 |
| bowl_stove_to_bowl_plate | 9 | 0 |
| bowl_stove_to_cream_cheese_bowl | 8 | 1 |
| bowl_stove_to_wine_cabinet | 9 | 0 |

conditional `M_transfer`（`m_transfer_correct`）：

| method | main | tpgmm |
| --- | ---: | ---: |
| Ours transfer | 45/45 True | 15/15 True |
| Pdiag finite target scratch | 45/45 True | 15/15 True |
| TP-GMM SE(3) target scratch | — | 15/15 True |
| Frame-weighted target scratch | 0/45 | 0/15 |
| Phase scalar GP target scratch | 0/45 | 0/15 |

transfer 方法的 `m_transfer_correct = True` 是构造结果（benchmark 先
`assert source_selector == target_selector` 才输出该行），不能当作独立的匹配识别准确率或新的 H1 验收门。
`fit_success` 全部 True，无 failure 行；索引保留 failure 字段与失败率语义。

## 6. 有限 action 桥接差分

在 translation-only、yaw-only、mixed 三类非交换条件上，用冻结 grid、target `C0` 与 synthetic nominal
curve 比较桥接 action 与独立矩阵实现 `_transfer_predict`：

- alpha（identity permutation）误差 ≤ 1e-12；
- finite SE(3) action 矩阵误差 ≤ 1e-10。

未遍历 heldout curves 重生成 `e_task`；历史性能数值直接引用 CSV 索引。任一 gate 失败（如
`OUT_OF_SUPPORT` 的 [0.05,0,…]）返回 `None`，不产出 action。

## 7. 历史 Euler 顺序

`historical_pose6_transform` 显式 `Rx(roll) Ry(pitch) Rz(yaw)`（`Rx @ Ry @ Rz`），与 scipy
`Rotation.from_euler("XYZ")`（即 sister 采用的 extrinsic XYZ 约定）一致；与 TSR canonical `Rz Ry Rx`
在非交换角度上差分明显。测试覆盖显式矩阵、scipy `XYZ` 一致性与 canonical `Rz Ry Rx` 差异三个断言。
（测试比较的是本地 `Rx@Ry@Rz` 复制公式与 scipy 约定，未直接调用 sister 的 `_transfer_predict` 或其
`matrix_from_euler`。）

## 8. 伪重复与 physical gap

- 继承证据是 retrospective controlled-pose development evidence，不是独立 physical Sim→Sim 验证。
- 3 个 seed 是同一 scene/body/joint 的程序化重放，其评分轨迹 hash 相同；不能当作独立物理重复，
  场景数也不能当作独立机制数。
- `repeats` 实际为 1，而非 CLI 默认 3。
- `M_transfer` 1.0 由 `assert source_selector == target_selector` 构造而来，非独立识别准确率。
- 本状态上限 `H1-CONTROLLED-REUSE-LOCKED`；论文若需独立 physical 证据，须另立 annex（实际物理测量
  目标、独立 success/adequacy 判据、非程序化 selector 生成的扰动响应、统一 target 几何/执行预算、
  真正独立的数据单位）。

## 9. Provenance

42 个固定 source/artifact SHA-256 全部匹配（硬门）；observed sister HEAD 为
`7592abb8a37dfbdd60834ce03aabbc6a64f814f3`，仅作 provenance。sister repo 未修改、未提交、未推送。

冻结 profile NPZ：`1be02ba3f905c756b40b1aa8fd70a7582904c032cd1f6f2d147712a8aeef1aa5`。

实现文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `tests/gtsr/_h1_reuse.py` | `5fd08791c4d4c3b85b8b3c418a5a6f9728d2e8d9ac94ba6a41a68d0f8aeda032` |
| `tests/gtsr/test_h1_controlled_reuse.py` | `7a707a7ac55fb3e25b1e302d0f633b9ea43f6c782097e78848a503544a26c4f8` |

## 10. Gate traceability

| Gate | Exit evidence |
| --- | --- |
| H1B-01 provenance | 42项SHA、audited/observed HEAD、空SHA映射拒绝、schema/pair/duplicate-(source,target)负例、`load_manifest`原子校验targets |
| H1B-02 grid | 3 source × 18 profile（精确3×6笛卡尔积）、100×6、phase/finite/read-only |
| H1B-03 split/index | 180/75完整cell、60−N、run内配对、跨run不配对、CSV SHA校验、方法集/source-target一致性、重复行/缺失cell拒绝 |
| H1B-04 match | 5 allowlist MATCHED、unlisted/self/reversed/basis/phase篡改拒绝、裸contract被`query_grid`结构拒绝 |
| H1B-05 support | 每pair 65 MATCHED / 10 OUT_OF_SUPPORT、alpha=None拒绝 |
| H1B-06 bridge | 历史Euler与有限action差分（≤1e-12 / ≤1e-10），不重算e_task |
| H1B-07 boundary | 全量回归、上游只读、无fit/collector调用 |

## 11. 验证

```text
pytest tests/gtsr/test_h1_controlled_reuse.py -q   -> 28 passed
pytest -q                                          -> 638 passed, 1 xfailed
ruff check     tests/gtsr/_h1_reuse.py tests/gtsr/test_h1_controlled_reuse.py   -> All checks passed
ruff format --check tests/gtsr/_h1_reuse.py tests/gtsr/test_h1_controlled_reuse.py -> already formatted
git diff --check                                    -> clean
```

集成测试在 `maniskill_spectrum` 存在时执行 42-SHA 校验、grid 加载、180/75 索引、75 条件账本与 action
差分，并覆盖索引读取期的 CSV SHA 漂移/重复行/source 错配/缺失方法负例；单元测试（Euler、合同 factory、
结构性 allowlist gate、裸 contract 拒绝、PairBridge 只读别名、空 SHA 映射、copy-immunity）不依赖 sister repo。
