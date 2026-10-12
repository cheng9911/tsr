# Sprint H1-B — Controlled-Reuse H0 Bridge Implementation Plan

状态：`READY-TO-EXECUTE`（依赖已锁定H1目标/比较manifest；禁止仿真、拟合和上游写入）。

**Goal:** 将 sister repo 已有5对controlled-pose transfer artifact接入H0的显式family/basis/phase/support
gate，生成可审计结果索引；不重复已有benchmark，也不增加经验迁移证据。

**Spec:** `docs/superpowers/specs/2026-09-09-gtsr-sprint-h1-oracle-transfer-reuse-scope.md`

## Global constraints

- sister root只读；42个source/artifact SHA必须匹配reuse audit。审计基线HEAD只作provenance；允许
  后续descendant HEAD，但必须记录observed HEAD，且任一固定依赖SHA漂移即停止。图像dirty changes不触碰。
- 禁止调用`benchmark_geometry_transfer.main()`、任何`.fit()`、环境构造器、HDF5写模式或collector。
- 历史pose6导入固定`Trans(xyz) Rx Ry Rz`；TSR canonical `Rz Ry Rx`不得混用。
- NPZ是100×6 grid，不含theta；不得伪造`ResponseProfile6D`或放宽H0 `query()`类型合同。
- family allowlist先于H0；H0 primitive只有完整gate之后才能用于alpha搬运。
- 已有CSV保持原样；新文件只索引、校验和明确纠正metadata，不改写结果。
- 结果状态最高为`H1-CONTROLLED-REUSE-LOCKED`，不写physical transfer validated。

## Task 1 — 冻结manifest loader与来源完整性

新增`tests/gtsr/_h1_reuse.py`，读取H1 JSON manifest，验证schema/status、记录sister observed HEAD、
校验所有列明SHA，
并要求5个pair无重复、source/target/task instance字段完整。缺失或漂移立即抛出，不下载或替换。

测试：临时复制manifest做missing/hash-drift/duplicate-pair负例；真实sister存在时验证42项。

## Task 2 — 历史pose6与phase-tagged grid loader

实现独立`historical_pose6_transform()`，显式矩阵乘法`Rx @ Ry @ Rz`；用非交换角度与sister函数
差分。加载NPZ时验证字段、SHA、shape、finite、100点、每phase25点、phase0..3、每source
N30正好18条（3seed×6subset），并只读算术平均。暴露数组为copy/read-only view，不保留可写别名。

不得从100点grid回归theta。grid query只接受`(q,k)`，`q∈0..3,k∈0..24`；无插值。

## Task 3 — 无拟合物化历史split与结果索引

按manifest冻结loop/RNG算法生成两个run的train/eval condition IDs；验证主run45个cell各4methods、
补充run15个cell各5methods，键唯一，失败行保留。同run方法必须拥有同一派生split；跨run禁止配对。

新索引字段含`derived_n_eval=60-N`和`historical_csv_target_interventions_for_validation=N`，明确后者
为错误metadata。测试主run180、补充75，无调用模型fit。

## Task 4 — Allowlist与H0合同factory

为精确pair key构造`MatchedRelationTransfer`：`Bs=Bt=I6,H=I4,Gamma=I6`、六个relation semantic IDs、
四phase identity map、manifest source range和固定tolerances。不在allowlist的pair返回
`UNMATCHED_FAMILY`型实验层拒绝，不因六轴同名通过。

测试每个allowlisted pair静态MATCHED、假pair拒绝、basis/phase篡改拒绝；H只解释local increment，
不得比较或共轭source/target world nominal pose。

## Task 5 — Grid query gate与75-condition ledger

新增实验层`query_grid(contract, grid, q, k, target_pose6)`：先以历史Euler转换并Log，调用
`evaluate_contract()`、`source_phase()`、`evaluate_query()`；只有MATCHED才用已审计grid alpha。
返回冻结diagnostic，失败alpha=None。不能直接调用`transport_diagonal`绕过gate。

对每个pair及75个condition建立状态：60 mixed+baseline+4 isolated MATCHED，10 isolated
OUT_OF_SUPPORT（IDs 1,2,3,4,5,6,8,9,11,14）。不得把拒绝误差置零或删出分母。

## Task 6 — 有限action桥接差分（不重算历史性能）

在代表性的translation-only、yaw-only和mixed非交换条件上，用冻结grid、target `C0`与nominal curve
比较桥接action和sister `_transfer_predict`相同公式的独立矩阵实现：alpha≤1e-12，transform≤1e-10。
不要遍历heldout curves重新生成`e_task`，已有性能数值直接引用CSV索引。

测试同时确认修改输入array不改变contract/grid结果；任一gate失败不产生action。

## Task 7 — 报告、审查与回归

生成`docs/gtsr_h1_controlled_reuse_record.md`：列5pairs、两个run、预算、41/45与4/45、15/15宏均值、
conditional M_transfer、伪重复和physical gap。记录实现文件SHA与sister artifact SHA，不复制原CSV。

验证：

```bash
pytest tests/gtsr/test_h1_controlled_reuse.py -q
pytest -q
ruff check tests/gtsr/_h1_reuse.py tests/gtsr/test_h1_controlled_reuse.py
ruff format --check tests/gtsr/_h1_reuse.py tests/gtsr/test_h1_controlled_reuse.py
git diff --check
```

独立review关注Euler顺序、family gate、无fit/无上游写、split pairing、support coverage和声明边界。
通过后只将H1-B标为`H1-CONTROLLED-REUSE-LOCKED`；physical H1另立annex。

## Gate traceability

| Gate | Exit evidence |
| --- | --- |
| H1B-01 provenance | 42项SHA、audited/observed HEAD记录、schema/pair负例 |
| H1B-02 grid | 3source×18profile、100×6、phase/finite/read-only |
| H1B-03 split/index | 180/75完整cell；60−N；run内配对 |
| H1B-04 match | 5 allowlist MATCHED、unlisted/mutated拒绝 |
| H1B-05 support | 每pair65 MATCHED/10 OUT_OF_SUPPORT、alpha=None拒绝 |
| H1B-06 bridge | 历史Euler与有限action差分通过，不重算e_task |
| H1B-07 boundary | 全量回归、上游只读、无fit/collector调用、审查通过 |
