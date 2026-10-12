# G-TSR H1 target/comparison freeze record

日期：2026-09-10
状态：`TARGETS-COMPARISON-LOCKED_RETROSPECTIVE` / `H1-CONTROLLED-REUSE-LOCKED`

## 冻结结论

- H1继承`maniskill_spectrum`的5个匹配关系pair，不按结果删选。
- 主run冻结180行；TP-GMM补充run冻结75行，两个run不跨run配对。
- source law固定为每个source task的18条N=30 profile算术均值；不重拟合。
- 历史pose6导入采用`Trans(xyz) Rx(roll) Ry(pitch) Rz(yaw)`，不与TSR canonical
  `Rz Ry Rx`混用。
- 75个condition的合同覆盖为65 `MATCHED`与10 `OUT_OF_SUPPORT`；后者不得删去或记零误差。
- 继承结果是retrospective controlled-pose development evidence，不是独立physical Sim→Sim验证。

## 不重复工作的边界

本轮没有启动模拟器、collector、benchmark main或任何模型fit，也没有重算历史`e_task`。已有CSV只读
继承；H1-B只允许做artifact加载、split物化、H0 gate桥接、有限代数差分和结果索引。

## Provenance

审计基线sister HEAD为`aaf29d6e15897522ffdad534d808cf089a562976`。冻结复核时observed HEAD为
`7592abb8a37dfbdd60834ce03aabbc6a64f814f3`；两者间已提交差异仅位于`paper_figures/`，且42个
固定source/artifact SHA-256均仍匹配。执行硬门是固定依赖哈希；HEAD仅记录来源历史。

冻结文件SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| H1 scope | `3e4f8c15d9ea6729831594b2fa12e458dd335eec9e5c0d4067eb8e471bf2c3a4` |
| reuse manifest | `fb574f0066fe8a074959dee1714f3d8887807908b26b05bafce5acc7677295df` |
| audit script | `5448121c123f2f819a43a78fba16b4f273b5edfbc63dffb92849e4ac56e0e5d3` |
| audit regression tests | `62f2898e723758faf7f0980b402d05d5560e31b7c7678dd28c5bd0197c204e8a` |
| reuse audit snapshot | `93a4db3df715b27d155de4ae11ccf8d6d4bf221aa1b9203397e73e5a242f7b5c` |
| H1-B plan | `364f89573f953585df2f353fbb87978405db8d8567a128395fafbcebaa6d7e76` |

## 验证

- audit回归：6 passed；包含历史Euler非交换测试、audit语义重生成及manifest对账。
- manifest：5 targets；65/10 coverage；180+75 rows；retrospective状态有效。
- audit脚本：Ruff check与format check通过。
- `git diff --check`通过。

## 下一硬门

H1-B已按`docs/superpowers/plans/2026-09-10-gtsr-sprint-h1b-controlled-reuse-bridge.md`执行完成，
状态已推进为`H1-CONTROLLED-REUSE-LOCKED`；实现与验证见
`docs/gtsr_h1_controlled_reuse_record.md`。独立physical Sim→Sim证据必须另立annex，
不得由本冻结记录推导。
