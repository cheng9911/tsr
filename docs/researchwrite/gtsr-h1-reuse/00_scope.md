# Scope

- Project: gtsr-h1-reuse；mode: compose；text type: 单阶段实验协议。
- Reader/language: 项目研究者与论文审阅者；中文，保留代码标识。
- 本轮：先审计 sister repo 已有证据，再冻结 H1 的已有目标集、比较协议和增量工作边界。
- 不执行：新仿真、新采集、重新拟合、旧 benchmark 重跑、物理 rollout、G/H2/H3、图像修订。
- 输出：仓库原生 Markdown 协议 + JSON 来源快照；不是完整开题本，不导出 DOCX，不写外部归档。
- 用户硬约束：`/home/rocos/sia/maniskill_spectrum` 已有验证，不能重复。
- 数据角色：全部已有数据为 retrospective development；冻结本协议不将其变为 prospective test。
- sister repo审计基线HEAD：`aaf29d6e15897522ffdad534d808cf089a562976`。最终复核时HEAD已前进到
  `7592abb8a37dfbdd60834ce03aabbc6a64f814f3`，其相对基线的已提交差异仅涉及`paper_figures/`；
  42个固定依赖SHA仍全部匹配。HEAD只作provenance，依赖文件SHA才是执行完整性硬门；用户图像改动只读保留。
- H0: 已冻结数学/API；本轮不改 src/ 或测试、也不把 H0 等变性当成迁移有效证据。

采用 researchwrite 的证据表→论证→章节合同→审查流程；文献综述、设备采购、全文论文均不在范围内。
