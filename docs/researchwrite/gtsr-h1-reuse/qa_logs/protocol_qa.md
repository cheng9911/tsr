# H1 target/comparison protocol QA — 2026-09-10

状态：`QA-PASSED-FOR-TARGETS-COMPARISON-LOCK`。不是H1经验实验验收。

独立review先给foundation 8.44/10并允许起草；正文review随后发现并修正：

1. P0：历史sister为`Rx Ry Rz`，首版audit误用TSR canonical `Rz Ry Rx`。合成非交换角度测试
   先3 failed，修正后4 passed；support只读重算为65 MATCHED/10 OUT_OF_SUPPORT。
2. P1：Frame/GP不消费nominal frame；Pdiag/TP-GMM才使用从target_pose恢复的frame，现已分项记录。
3. P1：M_transfer在source/target selector相等assert之后计算，只能称条件性一致。
4. P1：主/TP-GMM run的RNG路径不同，禁止跨run配对；评价数是60−N，不是原CSV字段N。
5. P1：三个seed的评分object_pose相同，禁止物理独立样本推断。

修订后自检分数：问题9.0、张力9.0、证据9.0、逻辑9.0、可行性8.5、贡献7.5、风险9.5、
语言8.5，均值8.75。贡献分较低是刻意边界：本协议是证据复用治理，不冒充算法创新。

自动检查：目标5对、42项source/artifact SHA、24个H5、180/75行聚合、45/15完整paired cells、
support数组和6项审计测试。事实claim均追溯仓库文件；不需要外部文献来证明本地实现事实。

剩余研究缺口：独立physical response ground truth、统一target Z_t/执行预算、closed-loop success。
这些缺口阻止`TRANSFER-VALIDATED`，但不阻止本次目标与历史比较合同冻结。

独立reviewer在最终回复阶段触发服务usage limit；此前已返回全部P0/P1证据，并确认除上述修订外
无其他阻断。最终修订由主agent按同一检查项逐条复核，未虚构第二个最终批准结论。
