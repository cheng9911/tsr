# G-TSR E-09：公平基线验证结果

状态：`IDENTIFICATION-LOCKED`（2026-09-07）

本实验验证一个受限命题：在 planar、oracle-phase 的 canonical solver trajectory 上，有限对角响应律
是否比共享标量响应律更好地表达 generator-selective response。它不是六生成元 SE(3) 识别、在线相位、
跨任务迁移或物理成功率实验。

## 实验合同

- 数据：5 个独立 simulator execution seeds；每个 seed 固定使用 30 条 mixed-intervention trajectory
  拟合，并在同一组 8 条 isolated intervention trajectory 上配对评估。
- 相位：oracle phases 3--6，每相位重采样 25 点，共 100 点；没有在线 phase estimator。
- 名义量：六个方法共享同一 zero-intervention nominal trajectory 和 nominal socket frame。
- 信息边界：所有拟合与超参数选择只使用 30 条训练 trajectory。Generic RBF 的 5-fold CV 以 episode
  为 group；held-out trajectory、empirical isolated profile 和 success label 均不参与训练。
- 作用语义：Fixed、Frame、Scalar 和 Pdiag 直接构造 nominal-frame left action。Context Regression 和
  Full Operator 是 direct-pose predictors，仅通过
  \(A_m=\hat X_mX_0^{-1}\) 转成可比较的区域作用；不把它们解释成 generator-wise response law。
- 误差：平移以米计、yaw 乘以 0.03 m rad\(^{-1}\)，再以 mm-equivalent 报告。task error 在 phases
  5--6 将 yaw residual 权重置零，匹配任务 clear 后的 quotient convention。

分析时没有删除 seed 或 held-out condition。冻结 loader 只接受 collection protocol 中 successful、完整且
非 truncated 的 episode，并逐项复核 30/8 split 与 intervention manifest。

## 方法

| 方法 | 训练轨迹数 | 输出/约束 |
|---|---:|---|
| TSR-Fixed | 0 | \(P=0\) |
| TSR-Frame | 0 | \(P=I\) |
| Phase-Scalar GP | 30 | frozen GP estimator；\(P=w(\zeta)I\) |
| G-TSR Pdiag finite | 30 | frozen finite profile；\(P=\operatorname{diag}(\alpha(\zeta))\) |
| Context Regression | 30 | phase-local Generic RBF direct predictor；episode-grouped ridge CV |
| Full Operator | 30 | per-progress dense affine \(3\times3\) direct predictor |

Scalar GP 复用了 sister implementation 的 pointwise shared-scalar estimator、kernel、bounds、5 次 optimizer
restart 和固定 random state；为完成本实验的结构性对照，其 learned scalar 通过共同 nominal left action
实现，而不是沿用 sister code 的自由 affine intercept。因此这里检验的是 shared-scalar 与 diagonal
selectivity，不应把数值与旧 E-07 的 direct affine scalar 数值混为一谈。

## 结果

下表是每个独立 seed 上 8 条 held-out interventions 的 task-error 均值（mm-equivalent）。

| Seed | Fixed | Frame | Scalar GP | Pdiag | Context RBF | Full |
|---:|---:|---:|---:|---:|---:|---:|
| 20260818 | 22.104 | 17.734 | 15.983 | 14.886 | 4.509 | 3.647 |
| 20270818 | 10.252 | 5.623 | 3.017 | 1.938 | 1.943 | 1.932 |
| 20280818 | 12.024 | 7.352 | 5.364 | 4.016 | 5.169 | 4.229 |
| 20290818 | 10.852 | 5.788 | 4.047 | 2.538 | 4.957 | 4.050 |
| 20300818 | 10.641 | 6.710 | 4.248 | 3.033 | 1.570 | 1.517 |

跨 seed 汇总使用 mean ± s.d.，其中 \(n=5\) 指独立 simulator execution seeds；每个 seed 内的 8 条
interventions 是配对重复观测，未被当成 \(n=40\)。

| 方法 | Full response error | Response endpoint | Task error | Task endpoint |
|---|---:|---:|---:|---:|
| TSR-Fixed | 13.741 ± 5.011 | 7.615 ± 0.089 | 13.175 ± 5.035 | 7.613 ± 0.088 |
| TSR-Frame | 10.913 ± 5.095 | 6.067 ± 0.156 | 8.642 ± 5.131 | 0.326 ± 0.238 |
| Phase-Scalar GP | 7.803 ± 5.313 | 6.559 ± 0.050 | 6.532 ± 5.349 | 3.343 ± 0.108 |
| G-TSR Pdiag finite | 5.340 ± 5.416 | 0.500 ± 0.178 | 5.282 ± 5.422 | 0.421 ± 0.199 |
| Context Regression | 3.653 ± 1.726 | 1.764 ± 1.303 | 3.630 ± 1.732 | 1.762 ± 1.305 |
| Full Operator | 3.109 ± 1.262 | 0.231 ± 0.041 | 3.075 ± 1.259 | 0.228 ± 0.041 |

唯一预注册的 superiority contrast 是 Pdiag minus Scalar task error。五个 seed 的差值分别为
\(-1.098,-1.078,-1.348,-1.509,-1.215\) mm-equivalent，全部支持 Pdiag；seed-level mean difference
为 \(-1.250\) mm-equivalent，描述性 95% bootstrap interval 为 \([-1.394,-1.113]\)。该区间枚举
5 个 seed 的全部 \(5^5=3125\) 个有放回均值，不作为大样本总体推断。

其余 comparison 是预先声明的描述性对照，不给出 p value，也不做胜负硬门：

| Pdiag task error minus | Mean difference | Descriptive 95% seed-bootstrap interval |
|---|---:|---:|
| TSR-Fixed | -7.892 | [-8.253, -7.515] |
| TSR-Frame | -3.359 | [-3.612, -3.094] |
| Context Regression | +1.653 | [-1.430, +6.224] |
| Full Operator | +2.207 | [-0.688, +6.745] |

## 论文可用结论与边界

结果支持有限且清晰的结构性结论：在五个冻结 seed 上，generator-wise Pdiag response law 一致优于
共享 scalar law，说明任务中平移与 yaw 的 phase response 不能由一个共同权重充分表示。相对 Fixed 和
Frame controls 的改善也方向一致。

结果不支持“Pdiag 是最佳 predictor”。两个更灵活的 direct predictors 在跨-seed平均 task error 上更低，
且与 Pdiag 的差值区间跨零。它们不能输出已识别的 generator-wise law，因此此结果刻画的是
**结构可解释性/可识别性与纯预测容量之间的取舍**，而不是证明对角结构具有普遍预测优势。各方法均
呈现明显 seed-level heterogeneity，尤其是 seed 20260818；后续论文实验应增加
task/object/environment 层级的独立重复，而不应只增加同一 seed 内 intervention 数量。

## 可复现性

- 可执行 gate：`tests/gtsr/test_sprint_e09_baselines.py`
- 逐 condition/profile Source Data：`docs/source_data/gtsr_e09_results.json`；gate 会从当前模型重新生成
  内存结果并与该 artifact 比较，防止正文数值与实现静默漂移。结构字段严格相等；为覆盖项目支持的
  SciPy 1.10--1.17 least-squares termination 差异，误差允许 0.01 mm-equivalent、profile coefficient
  允许 0.005 的数值容差，均小于既有 replication gate 的容差尺度。
- baseline contracts：`tests/gtsr/test_baseline_models.py`
- 数据来源：`../maniskill_spectrum/phase_switch_symmetry_multiseed`
- 本次环境：Python 3.11.16、NumPy 2.4.6、SciPy 1.17.1、scikit-learn 1.9.0、h5py 3.16.0、
  threadpoolctl 3.6.0。
- Scalar GP 和 Generic RBF 依赖仅位于项目的 `test` extra；BLAS 在比较测试中限制为单线程以避免
  finite-difference least-squares 的线程抖动，退出 fixture 后自动恢复。
- 重新生成 artifact：
  `python -m tests.gtsr.test_sprint_e09_baselines --write-artifact`，随后必须重新运行 E-09 gate 并审查 diff。
