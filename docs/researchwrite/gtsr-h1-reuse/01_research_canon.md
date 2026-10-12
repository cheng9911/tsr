# Research canon

## 来源（代码/模型事实，不冒充文献事实）

- S1: sister `phase_switch_symmetry/benchmark_geometry_transfer.py`：5 pairs、18-profile source mean、finite action、split 与 metric 定义。
- S2: sister `phase_switch_symmetry/collect_libero_relation_suite.py`：selector 生成评分 pose；直接置 qpos+forward；末帧 success=True。
- S3: sister `phase_switch_symmetry/benchmark_libero_relation_suite.py`：object_pose 评分、phase重采样、nominal_frame 与 oracle_alpha。
- S4: [只读来源快照](sources/reuse_audit.json)：42 文件 SHA、24 个 H5、CSV 聚合复核、source range。
- S5: sister `phase_switch_symmetry_multiseed/libero_relation_suite/libero_relation_suite_experiment.json`：具体任务、scene/body/joint、nominal_pose6。
- S6: [H0 冻结记录](../../gtsr_transfer_contract_record.md)。

## 实测事实（本轮仅只读审计）

1. 主报告 180 行，TP-GMM 补充 75 行，均无 fit_failure；repeats 实际为1，不是 CLI 默认3。
2. 所涉及8任务各3个H5，每文件75 conditions；同任务各seed的评分 object_pose 哈希完全相同。
3. drawer source 与 plate target 的评分轨迹哈希也相同；对象/scene不同不能自动提供新的评分轨迹证据。
4. 源 artifact 保存 profile grid，不含 theta；每个 source N30 是6 subsets×3 seeds共18条 profile的均值。
5. 59个源训练condition的Log-coordinate包围盒包含全部60个mixed；按历史`Rx Ry Rz`约定，
   10个isolated越界、4个isolated与baseline在界内。
6. Markdown旧轨迹误差与当前CSV不一致；本轮只使用逐行CSV可复核的数值。
7. S1在benchmark循环前assert learned source selector等于target oracle selector；故M_transfer=1
   是条件性入组一致结果，不能作为独立的matched-family识别门或准确率验证。
8. 原CSV的target_interventions_for_validation=N字段填错；实际e_task评分60−N个heldout
   mixed contexts（N=3/5/8分别57/55/52），scratch与transfer在各自run内共享该集合。
9. 两份run都从split_seed=20260831起步，但TP-GMM仅遍历N=8，RNG消耗不同；不能把其行拼入主run。
10. target 60个mixed中59个干预值属于source ensemble训练union；这测试同干预设计上的跨task
    controlled reuse，不是unseen intervention-value generalization。

## 术语与禁用主张

H1-R = retrospective controlled-pose transfer reuse；H1-B = 待实施的H0合同桥接；physical H1 evidence仍未闭合。
禁止把程序生成 success 叫机器人成功率，把3个同轨迹seed叫独立物理重复，把180行叫180次训练。
禁止把target nominal calibration叫zero target data，或把自动通过selector标签叫学会物理对称性。
禁止把mixed上的in-range叫joint-distribution interpolation保证；禁止补采/重拟合来悄悄修复本轮审计。

## 未决项

已存baseline与transfer的nominal预算不同，无法据此隔离纯law贡献；无统一target Z_t及闭环验证。
历史pose6为`Rx Ry Rz`而TSR canonical chart为`Rz Ry Rx`；adapter必须显式处理该导入边界。
grid artifact不能直接构造成ResponseProfile6D；新adapter必须另写有拒绝语义的实施合同，不能伪造theta。
旧输出没有逐cell保存split IDs；仅有确定性RNG、loop order与split_seed，可无拟合重建但尚未物化。
