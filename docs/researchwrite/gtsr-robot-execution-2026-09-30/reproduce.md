# 复现与审计

## 环境

本轮使用两个已有环境。拟合/核验在 `tsr-dev`（Python 3.11）中，物理执行在
`maniskill_download`（Python 3.10）中。精确包版本见 `preparation.json` 和
`execution/manifest.json`。没有安装新依赖。

兄弟仓库 `../maniskill_spectrum/phase_switch_symmetry` 提供 Panda/PhysX 环境和历史基线，
它是只读依赖。本仓库的生产内核和 Sprint G 文件没有因本轮实验而修改。
执行目标只由旧训练集和冻结配置生成，没有使用新增执行结果调参。

## 命令

在仓库根目录运行。已有输入文件不可覆盖；从头复现请另建目录并复制协议文件。

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/tsr-dev/bin/python \
  scripts/gtsr_robot_execution.py prepare \
  --directory docs/researchwrite/gtsr-robot-execution-2026-09-30

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/maniskill_download/bin/python -u \
  scripts/gtsr_robot_execution.py batch \
  --directory docs/researchwrite/gtsr-robot-execution-2026-09-30

env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/envs/tsr-dev/bin/python \
  scripts/report_gtsr_robot_execution.py \
  --directory docs/researchwrite/gtsr-robot-execution-2026-09-30
```

`prepare` 使用单个 N=30 子集，不是18个模型集成。Pdiag 读取冻结 artifact，
Full affine 和 TP-GMM 使用同一训练集重拟合。所有方法使用共同解析名义路径。
三个新执行 seed 与旧训练 seed 不重合，但本轮只作开发证据。

## 尝试、失败和恢复

- 每单元先写 `.started.json`，结束后写 JSON。动作与真实状态保存在同名 HDF5。
- 批次恢复时跳过已有结果；只有开始标记而没有结果的单元记为中断失败，不自动重试。
- 进程超时/异常退出均产生失败 JSON，保留原始日志与可能存在的部分 HDF5。
- 每单元执行前后核对源码和输入哈希；漂移会停止批次。
- `execution/cells.json` 是预定尝试清单，成功率始终以该清单为分母。
- 规划耗时均值明确限定在有测量值的尝试，不能把缺失耗时的崩溃记为零。
- 正常 worker 的 `wall_seconds` 从模拟器创建前开始，包含环境初始化和执行，但不含 Python
  导入时间；进程级失败由外层记录耗时。这两种口径不合并成端到端速度排名。
  本轮公平预算是每目标最多2秒规划及每进程120秒上限，尚未提供完整端到端效率比较。
- mplib 当前安装没有公开的随机种子设置接口。环境初始条件配对可核验，
  不声称 RRT 随机路径逐位配对或跨机器逐位复现。

## 冒烟和输入修订记录

`smoke/` 保存正式批次之前的四次名义环境冒烟执行，使用单独 seed 202609300。
它们来自早期桥接脚本，只用于确认仿真与规划接口可运行，不并入168次开发比较。
代码审查后补充了中断保留、逐单元源码检查、力阈值优先处理、环境context核对和来源记录。

`preparation_v1/` 保留冒烟时使用的输入。正式准备阶段重新生成目标；两次准备的 action
数组逐元素一致（最大绝对差0）。正式执行的来源以 `execution/manifest.json` 为准，
不能用当前源码声称重新验证了早期冒烟源码。`source_snapshot/` 保存正式批次的源码副本。

## 解释边界

- 环境成功定义是位置<12 mm、插入轴误差<0.15 rad，触发后立即结束。
  因此不能将成功写成“达到最终名义深度”“全姿态精确跟踪”或“严格装配完成”。
- 程序阶段是预定动作阶段，不是由物理事件确认的 oracle phase；标签推进不保证物体通过入口。
- 目标区域是控制容差，未学得任务允许集合 Z；本轮也不是耦合几何表达能力比较。
- PhysX 启用碰撞和接触；底层规划器未导入完整场景碰撞模型，不具有连续无碰撞证书。
- 阈值、阶段或控制方式若需修改，必须另建实验版本，保留这一批全部结果。
