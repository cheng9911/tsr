# G-TSR 论文论证与实验审计

日期：2026-09-30。状态：探索性论证与开发数据审计；没有授予新的确认性实验通过状态。

本轮把论文主线整理为：在已知任务关系和阶段的条件下，将可行区域与扰动响应分别参数化，
学习各生成元随阶段变化的响应强度，并检验这种结构何时有用、何时失效。

当前判断：表示与选择性响应有明确依据；“普遍最优预测”“任意对称群等变”“独立物理迁移已成立”
均超出现有证据。新增实验发现，F1 使用的世界系动作残差受坐标原点影响，必须在论文中明确。

## 文件

- [论文主线与方法初稿](01_paper_story.md)：一句话主张、中文论述、方法定义、贡献与术语。
- [理论审计与命题](02_theory_audit.md)：可证明性质、假设、证明及反例。
- [实验结果与判断](03_evidence_report.md)：已有数据复算、新实验、验证记录和局限。
- [后续实验设计](04_experiment_plan.md)：下一阶段的具体问题、对照、指标与失败规则。
- [完整机器可读结果](results/audit_results.json)；[汇总图 PNG](results/evidence_audit.png)、
  [SVG](results/evidence_audit.svg)、[PDF](results/evidence_audit.pdf)。
- [复现脚本](../../../scripts/audit_gtsr_paper.py)。

## 复现

在仓库根目录运行，输出使用新目录，避免覆盖已保存结果：

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/gtsr-paper-audit-mpl \
  /home/rocos/miniconda3/envs/tsr-dev/bin/python \
  scripts/audit_gtsr_paper.py --out /tmp/gtsr-paper-audit-reproduction --raw-controls
```

`--raw-controls` 需要相邻 `maniskill_spectrum` 仓库的既有六维 HDF5。不传此参数仍可复算
仓库中的 JSON/CSV 并运行合成实验。脚本不拟合原有 profile，不生成物理 rollout。

原有未提交修改、冻结文档、模型与源数据保持原样。诊断使用新增文件，不将探索性结果回写为预注册结果。
