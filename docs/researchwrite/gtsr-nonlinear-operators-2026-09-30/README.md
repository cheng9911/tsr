# 最小显式基线＋变螺距验证

## 结论

第一决策节点已完成。本批证明了算子一致性，尚未发现 G-TSR 相对同映射直接参数化的新增求解能力。
暂缓扩大曲面和机器人实验，先定义可区分的技术机制。

| 实现 | 参考正确投影 | 求解器报告成功 | 中位时间 ms |
|---|---:|---:|---:|
| G-TSR＋解析导数 | 320/320 | 310/320 | 21.47 |
| 直接映射＋解析导数 | 320/320 | 310/320 | 21.39 |
| 直接映射＋自动微分 | 320/320 | 310/320 | 68.22 |

三个实现共享生产投影器。解析对照的结果与映射/导数调用次数完全一致。
自动微分的时间差是本次 CPU eager 后端的实现成本，不能作方法固有效率结论；
本批也不是隔离整机负载后的微基准，不能解释不足1%的时间差。
320是每方法配对查询次数，包含跨种子重复的固定边界查询；不同参数/位姿/权重组合为200个。
没有统计独立性或跨任务泛化结论。

- 总记录：15,360个样本、960个导数检查、960次投影、3,840条连接记录、40个分段近似检查。
- 每方法采样合法5,120/5,120，最大跨实现位姿差约1.11e-16。
- 最大投影距离差约4.81e-10 m；最大导数误差约6.11e-11。
- 每方法坐标映射连接640/640带内，位姿线性连接228/640带内；该差异三种实现共享。
- 所有独立参考数值复核通过；参考具有60位根细化，没有区间算术认证。
- 每方法10条求解器失败标志保留：边界附近的裁剪差分带来约1e-6梯度偏差。
  独立解析KKT残差最大1.46e-10，而裁剪差分残差约1.00002e-6至1.00005e-6。
  这是共同终止诊断问题，不是某方法找不到可行解，也不是论文创新。

## 文件

- [冻结开发协议](protocol.md)
- [独立参考与连接界推导](theory.md)
- [当前结果报告](report_v2/results.md)、[汇总](report_v2/summary.json)
- [结果图](report_v2/decision_gate.png)：另有SVG和PDF
- [原始开发批次](development_v1/manifest.json)：JSONL记录、源码快照、预热时间
- [原始异常标志](report_v2/projection_flags.json)、[边界诊断](boundary_diagnosis.json)
- [研究决策与方法缺口](decision.md)
- [旧数据校验](preservation_check.json)：4,174个已有文件哈希未变

`pilot_v1` 在快照阶段因相对/绝对路径混用退出，没有实验结果；目录保留。
修复后 `pilot_v2` 完成48次投影，见 `pilot_report_v1`。
`report_v1` 保留；`report_v2` 将“unique queries”改为配对次数，并单列200个不同查询设置。
没有因结果好坏删除或重选参数。`development_v1` 源码快照是生成该批原始数据的版本。
其后只加固异常标志保存、修正参考未知时的图表分母、增加可选依赖跳过和边界诊断；
没有修改生产内核、任务参数或本批原始记录。最终交付脚本另存于 `delivery/source_snapshot`。

## 复现

在仓库根目录执行，输出目录必须不存在。环境为 Python3.10.20、NumPy1.26.4、SciPy1.11.4、
PyTorch2.5.1（本批仅CPU）、mpmath1.3.0、matplotlib3.10.9。没有安装新依赖。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/maniskill_download/bin/python \
  tests/gtsr/test_nonlinear_operators.py -v

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/maniskill_download/bin/python \
  scripts/benchmark_gtsr_nonlinear.py \
  --out docs/researchwrite/gtsr-nonlinear-operators-2026-09-30/development_reproduction

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/maniskill_download/bin/python \
  scripts/report_gtsr_nonlinear.py \
  --input docs/researchwrite/gtsr-nonlinear-operators-2026-09-30/development_reproduction \
  --out docs/researchwrite/gtsr-nonlinear-operators-2026-09-30/report_reproduction
```

新增15项unittest通过；相关生产回归58项通过、1项既有预期失败。
未安装 torch/mpmath 的轻量环境会跳过整个实验测试模块；上述专用环境已实际运行全部15项。
未运行DSS/SPS、CCA或机器人。未接入的方法不出现在成功率分母中。
