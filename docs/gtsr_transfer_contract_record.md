# G-TSR H0 — Transfer Contract Acceptance Record

日期：2026-09-09。状态：`TRANSFER-CONTRACT-LOCKED`（工作区内容哈希冻结，非经验迁移验收）。

## 交付边界

本记录只验收 provided matched-family 条件下的 diagonal-law transport 数学/API 合同。
不验收经验迁移性能、Sim→Real、任务成功率、真实干预次数减少或 learned phase。

- Spec：[H0 contract](superpowers/specs/2026-09-09-gtsr-sprint-h0-matched-relation-transfer-contract.md)
- Plan：[H0 implementation](superpowers/plans/2026-09-09-gtsr-sprint-h0-matched-relation-transfer.md)
- 实现：`src/tsr/gtsr/transfer.py`；测试：`tests/gtsr/test_transfer_contract.py`。
- 基线 HEAD：`c28dffa3e5d1f51cdfc1ae9a6bc29ba3b9f8b07c`。
- 本次改动尚未提交、合并或推送；以下内容哈希标识工作区验收版本，不伪称实现 commit。

## 内容标识

| 文件 | SHA-256 |
| --- | --- |
| `src/tsr/gtsr/transfer.py` | `4cb7e33e28308737b35f560c044754ec0cda470683e55c47d41b7356a1a02db1` |
| `src/tsr/gtsr/__init__.py` | `0467a60f9e6f20af8b848092dfbf4f0f073bfa2e061a6e918cf8f4a660b7f034` |
| `tests/gtsr/test_transfer_contract.py` | `6f525ac9964e3f68ba422bb24983ae64d986bb361f74a6a576f413eb7d90eb2e` |

## 硬门映射

| Gate | 证据 |
| --- | --- |
| H0-01/02 | 100 组带符号/尺度 monomial roundtrip、diagonal closure、copy 与非法输入 |
| H0-03 | generator semantic 匹配、合法错配优先拒绝、非法 ID 拒绝 |
| H0-04/05 | identity residual、含平移 cross-term 的 Adjoint、500 组 exact-compatible basis |
| H0-06 | 500 组 `K_t=Ad_H K_s Ad_H^-1`，绝对误差门 1e-10；独立 `scipy.linalg.expm` 与冻结 `ResponseOperator` 有限动作交叉检查 |
| H0-07 | 单轴 veto、总量/逐轴尺度不变性、overflow 与两个 underflow 回归 |
| H0-08 | identity/分段线性 phase warp、端点、拓扑/语义/非法查询拒绝 |
| H0-09 | 带符号/尺度映射后的 source range 边界和 tolerance |
| H0-10 | profile identity/permutation 查询、三条拒绝路径 alpha=None、phase-count/参数验证与非突变 |
| H0-11 | 公共导出、全量回归、冻结模块无 diff（最终结果见下） |

## 执行与审查

实现前集中测试因缺少 `tsr.gtsr.transfer` 出现预期 red；首轮实现后 39 passed。
未按草案逐 Task 创建七个独立提交，实际按一个工作区的 red→green 交付，并如实记录该偏离。

依 `requesting-code-review` 技能执行独立只读审查。初审发现一个 Important：兼容 residual
在极端尺度下下溢，错误返回零；复验又定位同一问题的最终归一化路径。两个复现分别为：

1. `B_s=I, B_t=diag(1,1,1,1.1,1,1), S_c=1e-322 I`。
2. `B_s=1e300 I, B_t=B_s; B_t[0,1]=1e-30, S_c=I`。

两者均先独立运行新增回归测试确认 `DID NOT RAISE`，再修复为捕获可检测浮点异常并抛
`ValueError`；不返回 `MATCHED`。正常可表示尺度仍计算原定义 residual。
最终独立复验：**Ready to merge（以主 agent 最终全量回归为前提）**；59 项测试独立通过，
无剩余 Critical/Important，原问题关闭。复验还确认正常尺度 `1` 和 `1e-300` 的 axis residual
仍约为 `0.1`。此结论不扩展为经验迁移验证。

## 验证结果

环境：`tsr-dev`，Python 3.11.16、NumPy 2.4.6、pytest 9.1.1。
本轮安装仓库已声明的开发工具 Ruff 0.16.6，未改依赖文件。

- 最终 H0 定向测试：**59 passed**。
- 最终全量测试：**610 passed、1 xfailed、1 warning，143.94s**，611 项完整收集，退出码 0。
  xfail 为既有投影 canary；warning 来自既有 `test_tsr_chain` 的 SciPy 数值差分，与 H0 无关。
- Ruff check / format --check：改动的三个 Python 文件通过。
- `git diff --check` 及冻结 response/profile/geometry/legacy 文件无 diff：通过。

最终全量命令（仅在该进程移除 ROS PYTHONPATH、限制 BLAS 单线程，不持久修改环境）：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/rocos/miniconda3/bin/conda run --no-capture-output -n tsr-dev pytest -q
```

可选覆盖率诊断曾用 `--cov=tsr.gtsr.transfer` 在 conftest 加载时触发 NumPy 重复导入错误。
普通 NumPy 导入及普通 pytest 正常；保持同一 conda 命令、仅改为路径
`--cov=src/tsr/gtsr` 后成功。未因此改动业务代码或冻结依赖。覆盖率不是 H0 硬门。

续接会话后，原测试 session 不再可取回退出状态，因此最终验证重新完整执行，不推断其已通过。
终端当时的 `PYTHONPATH` 带入 `/opt/ros/humble/...python3.10...`，Python 3.11 的 pytest
自动加载外部 `launch_testing` 插件，在收集前因缺少 `lark` 退出。仅在测试进程移除该路径后
恢复正常收集 611 项；未为此安装 ROS 依赖或修改系统配置。

## 进入 H1 前必须另行冻结的输入

1. 具体 source/target relation family 与任务语义；不能把 H0 共轭构造直接当成对象迁移证据。
2. 独立给定的 `H / B_s / B_t / Gamma`、source Log-coordinate support、compatibility thresholds。
3. Oracle phase 来源和 phase correspondence；禁止按 transfer error 对齐。
4. 只读 source law artifact 与 provenance；target 数据不参与 source law 重新拟合。
5. 统一的 target `Z_t`、chart、planner/controller 与 transferred/static/target-refit 比较协议。
6. Development/confirmatory 数据角色、指标与失败规则；不得把 H0 的 MATCHED 当作迁移有效结论。

H0 不重开几何冻结，不自动启动 Sprint G、目标域采样或 confirmatory 数据生成。
