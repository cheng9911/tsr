# 全部物理执行尝试

分母包含预定条件中的全部失败。每个任务/方法12次尝试，3个执行seed×4个固定context。
不是12个独立任务实例，不能据此进行总体显著性声明。

| 任务 | 方法 | 总成功 | nominal | translation | yaw25 | mixed | 最大净接触力 N |
|---|---|---:|---:|---:|---:|---:|---:|
| keyed | tsr_frame | 10/12 | 3/3 | 3/3 | 3/3 | 1/3 | 100.933 |
| keyed | gtsr_frame | 10/12 | 3/3 | 3/3 | 3/3 | 1/3 | 100.933 |
| keyed | tsr_pdiag | 11/12 | 3/3 | 3/3 | 3/3 | 2/3 | 102.848 |
| keyed | gtsr_pdiag | 11/12 | 3/3 | 3/3 | 3/3 | 2/3 | 102.848 |
| keyed | tsr_fixed | 3/12 | 3/3 | 0/3 | 0/3 | 0/3 | 107.614 |
| keyed | tsr_full_affine | 12/12 | 3/3 | 3/3 | 3/3 | 3/3 | 1.730 |
| keyed | tsr_tpgmm_se3 | 10/12 | 3/3 | 2/3 | 3/3 | 2/3 | 104.528 |
| circular_honest | tsr_frame | 10/12 | 3/3 | 3/3 | 3/3 | 1/3 | 110.095 |
| circular_honest | gtsr_frame | 10/12 | 3/3 | 3/3 | 3/3 | 1/3 | 110.095 |
| circular_honest | tsr_pdiag | 10/12 | 3/3 | 3/3 | 3/3 | 1/3 | 102.461 |
| circular_honest | gtsr_pdiag | 10/12 | 3/3 | 3/3 | 3/3 | 1/3 | 102.461 |
| circular_honest | tsr_fixed | 6/12 | 3/3 | 0/3 | 3/3 | 0/3 | 108.678 |
| circular_honest | tsr_full_affine | 12/12 | 3/3 | 3/3 | 3/3 | 3/3 | 0.813 |
| circular_honest | tsr_tpgmm_se3 | 12/12 | 3/3 | 3/3 | 3/3 | 3/3 | 0.000 |

## 失败与停止原因

| 任务 | 方法 | 原因及次数 |
|---|---|---|
| keyed | tsr_frame | {"terminated": 10, "force_abort": 1, "schedule_finished": 1} |
| keyed | gtsr_frame | {"terminated": 10, "force_abort": 1, "schedule_finished": 1} |
| keyed | tsr_pdiag | {"terminated": 11, "force_abort": 1} |
| keyed | gtsr_pdiag | {"terminated": 11, "force_abort": 1} |
| keyed | tsr_fixed | {"force_abort": 7, "schedule_finished": 2, "terminated": 3} |
| keyed | tsr_full_affine | {"terminated": 12} |
| keyed | tsr_tpgmm_se3 | {"terminated": 10, "Fail to parameterize path": 1, "force_abort": 1} |
| circular_honest | tsr_frame | {"terminated": 10, "force_abort": 2} |
| circular_honest | gtsr_frame | {"terminated": 10, "force_abort": 2} |
| circular_honest | tsr_pdiag | {"terminated": 10, "force_abort": 2} |
| circular_honest | gtsr_pdiag | {"terminated": 10, "force_abort": 2} |
| circular_honest | tsr_fixed | {"force_abort": 5, "terminated": 6, "schedule_finished": 1} |
| circular_honest | tsr_full_affine | {"terminated": 12} |
| circular_honest | tsr_tpgmm_se3 | {"terminated": 12} |
