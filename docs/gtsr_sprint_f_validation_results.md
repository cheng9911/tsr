# G-TSR Sprint F：6D 六生成元响应轮廓复现结果

状态：`IDENTIFICATION-LOCKED`（2026-09-08）

实现基线：`d5d3d72`

本 Sprint 验证一个受限命题：在冻结的 SE(3) intervention distribution、oracle phase 与
Pdiag-finite parameterization 下，是否能识别六生成元响应轮廓，并将 yaw 判定为唯一
selective generator。它不验证 learned phase、matched-relation transfer、闭环任务成功率或
formal region-level observability。

## 冻结合同

- 数据：sister repository `maniskill_spectrum` 的 keyed/circular_honest 两个 arm，3 个独立
  simulator execution seeds。
- 拟合：每个 arm 使用 6 个 N=30 frozen subsets × 3 seeds，共 18 fit cells；两 arm
  按 `(seed, subset_id)` 形成 18 个配对单元。
- 统计单位：独立单位是 3 个 execution seeds；18 cells 仅是 seed 内重采样/敏感性
  计算单元，不解释为独立 `n=18`。
- 实现：`ResponseProfile6D`、`ProfileIdentifier6D`、`SelectivityMetrics6D` 和
  `SelectivityVerdict6D`；拟合成功与生成元选择性是两层正交 verdict。
- 运行：36 个 Pdiag fit 使用进程级并行、每进程 BLAS 单线程与 manifest-pinned
  resumable cache。

## 硬门结果

| 硬门 | 内容 | 结果 |
| --- | --- | --- |
| F-01 | manifest/rollout integrity；75/75 usable；60/14/1 split；rank-6 | 通过 |
| F-02 | 6D embedding/action 与 `ResponseOperator` 一致；shape/metric 合同 | 通过 |
| F-03 | 无噪 planted 6D profile recovery，最大误差 `≤5e-3` | 通过 |
| F-04 | 36 fits 成功；18-pair Pdiag aggregate 复现 | 通过 |
| F-05 | Frame-weighted negative control 按预测失去 per-generator selectivity | 通过 |
| F-06 | 公共导出及全量回归无回退 | 通过 |

18-pair aggregate 冻结 headline：

| 指标 | Pdiag finite (SE(3)) | Frame-weighted (SE(3)) |
| --- | ---: | ---: |
| `S_yaw` | 0.6495 | 0.2302 |
| `S_others_max` | 0.1273 | 0.2302 |
| purity | 0.7289 | 0.1667 |
| rank | 1.000 | 0.000 |
| transition | 1.000 | 0.000 |

Pdiag 满足 `S_yaw > 4·S_others_max`。非 yaw 五轴的最终状态是
`IDENTIFIED + NON_SELECTIVE`，yaw 为 `IDENTIFIED + SELECTIVE`。

## 实现与验证证据

- 实现：`src/tsr/gtsr/profile6.py`。
- 数学/合成门：`tests/gtsr/test_profile6.py`。
- 数据、复现与 negative-control 门：`tests/gtsr/test_sprint_f_replication.py`。
- 可恢复拟合：`tests/gtsr/_fit_se3_cells.py`。
- 独立整体复审：Ready to merge；唯一 Important 问题“cache 未被冻结 manifest 版本化”已在
  `d5d3d72` 修复。

收口验证：

- `pytest -q`：**535 passed, 1 xfailed**；`xfail` 为既有 seed-441 projection canary。
- `ruff check`、`ruff format --check`、`git diff --check`：干净。
- 冻结几何内核、Sprint D `response.py` 与 Sprint E planar `profile.py` 未改变。

## 论文可用结论与边界

本结果支持：在冻结的 6D SE(3) intervention distribution 和 oracle phase 下，有限对角
response law 能从六个生成元中分离出 keyed relation 的 yaw-selective response，而共享
Frame-weighted control 无法表达该 per-generator selectivity。

本结果不支持 learned/online phase、target-side zero-refit transfer、任意多生成元选择性、物理
rollout 成功率或跨环境总体推断。下一阶段必须在不改变本冻结 oracle-profile 的条件下，
单独验证 online phase observer。
