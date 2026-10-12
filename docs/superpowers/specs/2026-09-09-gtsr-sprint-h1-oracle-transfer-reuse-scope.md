# Sprint H1 — Oracle-phase Sim→Sim：已有证据复用与目标/比较协议

日期：2026-09-09。状态：`TARGETS-COMPARISON-LOCKED_RETROSPECTIVE` / `H1-CONTROLLED-REUSE-LOCKED`。
本协议是对已看过结果的 retrospective development evidence 进行冻结，**不是预注册确认实验**。
H1的独立物理迁移证据仍未闭合；不得以本文件状态宣称 `TRANSFER-VALIDATED`。

机器合同：[reuse manifest](2026-09-09-gtsr-sprint-h1-reuse-manifest.json)。
证据与哈希：[reuse audit](../../researchwrite/gtsr-h1-reuse/sources/reuse_audit.json)。
前置：[H0 acceptance](../../gtsr_transfer_contract_record.md)。

## 1. 本轮决定：继承结果，不重跑实验

用户要求复用 `/home/rocos/sia/maniskill_spectrum` 已有验证。经源码、CSV、NPZ、HDF5只读核对：

| 已有工作 | 本轮处理 | 可支持的主张 |
| --- | --- | --- |
| `geometry_transfer/` 180行 | 继承全部5pairs、4methods、N=3/5/8 | 既定受控位姿关系下的zero target-side law refit |
| `geometry_transfer_tpgmm_n8/` 75行 | 作为独立run继承5methods、N=8 | 同run内TP-GMM比较，不拼主run |
| `geometry_transfer_two_scene*` | 两个目标的展示摘要，非新数据 | 不新增样本、不重复计数 |
| `se3_transfer/` | 由Sprint F接管，不重跑 | 六生成元oracle识别；文件名含transfer不代表本H1匹配目标实验 |
| `symmetry_transfer/` | 保留既有对称性/行为错配证据 | 不能替代本H1 physical matched-target验证 |

H1-R为本轮已有证据继承；H1-B为下一步无拟合的H0合同桥接。H1物理证据需要单独设计，
不能通过给同一个程序化collector再加seed补齐。

## 2. 已有证据的真实生成机制

`collect_libero_relation_suite.py`先用`selected_delta = oracle_selector * causal_delta`，
再以`pose6 = alpha * (nominal_pose6 + selected_delta)`生成评分用`object_pose`。
它直接设置MuJoCo qpos并调用`sim.forward()`，同时保存`libero_body_pose_world`，但benchmark
实际评分的是前述task-local `object_pose`。末帧`success=True`由采集器写入，并非任务成功检测。

因此这是有真实scene/body/joint来源的**程序化受控位姿数据**，不是机器人策略产生的独立物理响应。
共享selector并按该selector评分，能够检查表示/复现的一致性，不能独立发现或证明物理任务规律。

只读核对的8任务×3seed共24个H5，每个含75个条件；同一任务三个seed的评分轨迹hash完全相同。
drawer与plate两任务的评分轨迹hash也相同。不得把这些seed当独立物理重复，或把场景数量当独立机制数。
原文件的usable/success计数只作为文件完整性，不作为执行成功率。

## 3. 冻结目标实例：全部5对，不按结果挑选

| pair_key | source → target | target scene | target nominal_pose6 [m,rad] |
| --- | --- | --- | --- |
| `drawer_to_plate_push` | drawer_middle_open → plate_front_push | libero_goal / task5 | [0.10,0,0,0,0,0] |
| `knob_to_microwave_door` | stove_knob_turn → microwave_door_revolute | libero_10 / task9 | [0,0,0,0,0,-1] |
| `bowl_stove_to_bowl_plate` | bowl_on_stove → bowl_on_plate | libero_goal / task8 | [0.07,-0.02,0.055,0,0,0] |
| `bowl_stove_to_cream_cheese_bowl` | bowl_on_stove → cream_cheese_in_bowl | libero_goal / task6 | [0.07,0.02,0.045,0,0,0] |
| `bowl_stove_to_wine_cabinet` | bowl_on_stove → wine_bottle_on_cabinet | libero_goal / task2 | [0.08,0,0.10,0,0,0] |

body/joint/free_joint的精确标识由manifest的`targets[].target_instance`与audit中`task_specs`固定。
任务语言可能包含复合任务，但本表只评价给定关系子任务，不能声称完成完整LIBERO任务。
可展示既有two-scene的knob→door与bowl→container，但数值必须报告全部5对。

匹配family由这份pair allowlist与原任务定义给定，不通过source预测alpha或target误差选择新pair。
H0的六轴名称相同不自动证明family匹配；不在allowlist的配对应拒绝，不现场拟合Gamma。

## 4. Source law、目标校准与数据预算

Source固定NPZ `libero_relation_suite_profiles.npz`，SHA见manifest。对每个source task，选全部
N=30、subset12–17、三个seed的18条profile取算术均值；3个不同source共54条已有profile。
这不是“单次N=30拟合”，也不是“18次独立物理采样”。六子集的训练condition并集为59。
target的60个mixed中59个也在该source训练干预值并集中：这是跨task复用，不是新干预值泛化。

| pipeline | target nominal trajectory | target law-fit interventions | 实际消费的reference/frame |
| --- | ---: | ---: | --- |
| Existing transfer | 每target/seed的condition0曲线一条 | 0 | task spec `nominal_pose6`作为action frame；condition0作为nominal curve |
| Frame-weighted scratch | 0条专用nominal | N | 直接由训练contexts/curves拟合metric-space intercept与共享weight |
| Phase scalar GP scratch | 0条专用nominal | N | 先拟合Frame-weighted intercept，再平滑共享phase scalar |
| Pdiag finite scratch | 0条专用nominal | N | 从训练`target_pose`去除intervention估计action frame；pointwise intercept初始化并更新nominal curve |
| TP-GMM SE(3) scratch | 0条专用nominal | 8 | 从训练`target_pose`去除intervention估计SE(3) frame；仅补充run |

所有历史行保持原预算。不能称zero target data，也不能称common-nominal law-only ablation。
N是scratch训练预算；transfer在各N下重复评价同一source law，不是新增transfer fit。
target `Z_t`、统一chart、IK/controller尚未在这批数据中装配；不补造domain或成功容差。

## 5. H0接入的局部几何与phase合同

本数据的关系增量在各自规范化task-local坐标中编码。历史sister实现明确使用
`U(c_pose)=Trans(xyz) Rx(roll) Ry(pitch) Rz(yaw)`。它不同于TSR canonical chart的
`Rz Ry Rx`；H1-B必须在import boundary显式转换，禁止把两种pose6当成同一Euler chart：

\[
c_{\log}=\operatorname{Log}U(c_{\mathrm{pose}}),\quad
B_s=B_t=I_6,\quad H_{\mathrm{local}}=I_4,\quad\Gamma=I_6.
\]

这里`H_local=I`只对已经声明的局部增量成立，**不是两物体world frame相同**；target nominal
`C0_t`独立实例化，不能令`C0_t=C0_s`或用H0共轭reference theorem声称两条nominal trajectory相等。
Source/target family先由§3 allowlist检查，之后H0检查basis、phase、support。

冻结数值检查：total与六个axis tolerance均1e-10；这是identity synthetic接口检查容差，
不是由历史性能反推的物理标定预算。不可表示浮点输入沿用H0拒绝语义。

Raw phase `[3,4,5,6]`映射为`[0,1,2,3]` = reach/move/hold/retract。
每phase25点，`rho=k/24`，phase correspondence与内部warp均identity；只查询已有phase-tagged grid。
phase交界的progress数值重复，必须保留`(q,rho)`，不得先去重为一维时间或跨边界插值。
这些phase来自solver annotation，不是在线识别或可部署观测器。

Target finite prediction保持原benchmark定义：

\[
\hat X_t(q,k;c)=C0_t\operatorname{Exp}
\left(\alpha_s[q,k]\odot c_{\log}\right)C0_t^{-1}X_{0,t}(q,k).
\]

NPZ只有grid，没有theta。H1-B须增加**实验层**只读grid适配器，不能重拟合source来制造
`ResponseProfile6D`，也不能修改冻结`query()`使任意duck type绕过验证。
适配器须验证形状/有限性/phase-grid与来源，调用`evaluate_contract/evaluate_query/source_phase`，
只有MATCHED后才代数搬运grid alpha并装配action；拒绝返回alpha=None。原始primitive单独调用不是证书。

## 6. Source support与拒绝条件

范围仅取上述59个source训练conditions的`c_log`逐维min/max；精确数组冻结在manifest的`support`。
`coordinate_scales=max(abs(lower),abs(upper))`；closed-bound tolerance=1e-12。
不使用target拟合结果、isolated oracle或历史error调整边界。

按修正的历史`Rx Ry Rz`约定只读计算确认60个mixed和baseline均在该box内，但14个isolated中的
condition `[1,2,3,4,5,6,8,9,11,14]`越界。H1-B必须保留这10个OUT_OF_SUPPORT，
不得改盒子来让它们通过；其余4个isolated通过。因此75个条件的预期账本为
65 `MATCHED`（60 mixed+baseline+4 isolated）与10 `OUT_OF_SUPPORT`。
这只证明coordinate range，不证明联合分布覆盖；即使nominal位于box，也不替代任务adequacy。

## 7. 两个冻结比较run及精确split规则

主run：5pairs×3seed×3个N×1repeat×4methods=180行。
TP-GMM补充：5pairs×3seed×N8×1repeat×5methods=75行。
seed标签固定20260818、20270818、20280818；repeats实际为1，而非CLI默认3。

每run独立初始化`np.random.default_rng(20260831)`。按§3 pair顺序→seed列表顺序→本run的N顺序
→repeat0推进RNG。mixed condition_ids为升序15..74：

```text
train_indices = sort(rng.choice(60, size=N, replace=False))
eval_indices = setdiff(arange(60), train_indices)
condition_id = mixed_condition_ids[index]
```

原输出未保存逐cell split；H1-B只用以上冻结算法物化split IDs，不运行拟合。
每个paired比较键是`(run_id,pair_key,seed,N,repeat)`，不跨run拼行。
两run虽然seed相同，但N循环不同、RNG消耗不同，TP-GMM结果须和其所在补充run的其它4方法配对。

实际评价数`derived_n_eval=60−N`，即57/55/52。历史transfer CSV的
`target_interventions_for_validation=N`与代码不符；不修改原CSV，在新索引旁列正确派生值。
不得将N称为transfer的评价样本数。

## 8. 指标、结果继承与失败报告

Primary沿用`e_task`：预测与评分`object_pose`转换为pose6后的分量差，RPY分量分别wrap，
乘`[1,1,1,0.03,0.03,0.03]`，再对heldout contexts×100bins×6分量平方取均值。
这是pose6 metric-scaled MSE（m-equivalent²），**不是intrinsic SE(3) geodesic**，不能直接当mm误差。
新增桥接检查可比较群矩阵，不能静默替换历史性能metric。

Secondary `e_alpha`为grid与collector预设selector/ramp oracle的六维MSE；不当成独立物理ground truth。
`M_transfer`阈值0.5仅为历史条件性selector agreement：benchmark先assert source_selector==target_selector
才输出结果，故其1.0不能作为独立匹配识别准确率或新的H1验收门。

主run当前CSV的transfer `e_task_mean`：

| scratch N | frozen transfer mean | count |
| --- | ---: | ---: |
| 3 | 1.8247556345547286e-7 | 15 rows |
| 5 | 1.878916112704118e-7 | 15 rows |
| 8 | 1.8844053700248234e-7 | 15 rows |

旧MD中约9e-9的数值及two-scene摘要不再作为当前数值来源；保留原文件，不偷偷覆盖历史。
逐cell只读比较transfer与Pdiag scratch：41/45较低、4/45较高；15个pair×N均值均较低。
这不是“所有cell都赢”。所有图表必须保留4个反向cell及各pair结果，不只展示总均值。

这里只做描述性比较，不用重复seed/共享source ensemble/相关subset行做独立样本p值、CI或equivalence。
不沿用F.1/G-00 margin；本阶段不设由既有结果调出来的transfer superiority门。

原文件无fit_failure；后续读取仍保留失败行，报告分母和failure rate。
缺文件、SHA漂移、重复或缺失cell导致受影响继承项停止，不能默默排除/补采/重训。
OUT_OF_SUPPORT是合同拒绝，单列coverage，不把其误差设0或从总查询数静默删除。

## 9. 仅允许的下一步增量与退出条件

本轮交付：目标/比较协议、机器manifest、来源快照、只读审计脚本、foundation/QA记录。
**已经完成的实验不重跑**。H1-B后续实施计划只包含：

1. 读取冻结grid artifact与phase标签、来源校验；不用theta、不拟合。
2. 按§7物化两run的split及正确的评价计数，核对完整method pairing。
3. 实现实验层grid→H0合同桥接及target nominal action装配；不改H0/D/E/F。
4. 对已有grid/context做接口差分：alpha permutation identity误差≤1e-12、finite action矩阵误差≤1e-10；
   禁止调用benchmark main()、重新训练或生成仿真episodes。
5. 所有75个manifest条件都输出合同状态；10个isolated拒绝、4个isolated/60个mixed/baseline通过。
6. 生成引用原CSV的结果索引、预算与失败/拒绝账本，不将继承结果改名为新实验。

H1-B通过只能标为`H1-CONTROLLED-REUSE-LOCKED`，不能标为物理迁移已验证。
若论文需要独立physical Sim→Sim证据，下一份annex必须锁定：实际物理测量目标、独立success/adequacy
判据、非程序化selector生成的扰动响应、统一target几何/执行预算、真正独立的数据单位。
那是新证据缺口，不是要求重复现有collector。

## 10. 审查与来源

完整审计和论证见 [researchwrite foundation](../../researchwrite/gtsr-h1-reuse/00_scope.md)。
foundation独立审查8.44/10，允许起草；关键审查项已直接落实§4/7/8。
代码/数据来源以audit快照42个SHA为准；审计基线HEAD仅作provenance。最终复核时sister HEAD已前进，
但相对基线的已提交变化仅在`paper_figures/`且42个依赖SHA仍匹配，因此不构成实验输入漂移。
后续HEAD可继续前进，只要全部固定依赖SHA匹配并记录observed HEAD；这不是完整依赖环境复现证明。
本轮不修改sister repo（含正在编辑的图像），不提交、不推送；H0既有未提交内容保持不动。
