# G-TSR 论文主线与方法初稿

## 一句话讲清楚

**我们研究机器人操作中随阶段变化的选择性适应：把任务允许的位姿集合与环境扰动的传递规律分别建模，
从受控干预中学习哪些运动方向应当跟随、何时跟随，并在明确对应关系下研究规律复用。**

建议工作标题：**G-TSR：面向操作任务的阶段条件生成元响应建模**。
英文工作标题：**G-TSR: Phase-Conditioned Generator Responses for Manipulation**。
跨任务迁移可作为待验证扩展；在独立物理证据完成前，不放入标题的已实现承诺。

## 英文核心段落草稿

Manipulation tasks require different responses to changes in object relations at different stages of execution.
We represent these responses using a phase-conditioned diagonal operator in a specified generator basis,
while parameterizing the admissible pose region separately. Controlled interventions provide observations
for fitting the response profile, and a finite SE(3) transformation applies the learned response to a nominal
region. Existing oracle-phase experiments support generator-specific responses over a shared scalar control
on a keyed-to-circular insertion task. More flexible predictors achieve lower average errors in the same
planar benchmark, defining the limits of the diagonal model. Transfer under supplied correspondences remains
supported by controlled-pose evidence; independent physical transfer requires further evaluation.

这是论证段落，不是完整投稿摘要。最终摘要还需统一评价指标并补充确定的核心实验结果。

## 可直接用于引言的中文草稿

操作任务对环境变化的响应取决于当前阶段和变化方向。例如，在带键入口的插入任务中，入口绕轴
旋转要求机器人相应调整朝向；通过入口进入圆形孔段后，同一朝向变化可能不再影响任务可行性。
位置变化仍可能需要继续传递。这个例子提出了一个具体问题：如何在统一表示中描述各方向的响应，
并识别其随任务阶段发生的变化？

任务空间区域描述满足位姿约束的一组构型。为了表达任务适应，还需要指定环境关系变化如何作用于
这个区域。本文分别参数化两个对象：可行域描述执行允许的自由度，响应算子描述扰动的传递强度。
二者可以共享任务阶段，但不必具有相同的坐标维数或物理语义。

我们以 G-TSR 为几何载体，在预先给定的关系生成元基中引入阶段条件对角响应算子。受控干预用于
估计各方向的响应轮廓，SE(3) 指数映射将估计结果转换为有限位姿变换。方法的结构假设是：在所选
基、阶段和干预范围内，响应近似按生成元分离。该假设使响应具有可解释的方向语义，也带来必须
验证的适用边界，尤其是跨轴耦合与任务自由度引起的不可辨识性。

现有实验支持逐生成元结构相对共享标量结构的价值，同时显示更灵活模型具有更低的平均预测误差。
因此本文需要用结构恢复、预测误差与任务可行性三类证据共同评价方法。匹配任务之间的规律复用
进一步检验这种结构的可重用性，其结论必须建立在独立目标响应和统一目标侧数据预算上。

## 方法定义

给定关系名义位姿 \(C_0\)、当前关系位姿 \(C\)、已知阶段 \(\zeta=(q,\rho)\)、可逆的关系基
\(B_r\in\mathbb R^{6\times6}\) 和名义任务区域，定义

\[
c=B_r^{-1}\operatorname{Log}(C_0^{-1}C)^\vee,\qquad
D(\zeta)=\operatorname{diag}(\alpha_1(\zeta),\ldots,\alpha_6(\zeta)),
\]
\[
A(\zeta,C)=C_0\operatorname{Exp}\big([B_rD(\zeta)c]^\wedge\big)C_0^{-1},
\]
\[
\mathcal R(\zeta,C)=
\{A(\zeta,C)T_0(\zeta)\Phi_{\mathcal C_a}(z)T_e(\zeta):z\in\mathcal Z(\zeta)\}.
\]

\(\alpha_j=0\) 表示该方向不传递，\(\alpha_j=1\) 表示完整传递。当前拟合器使用 24 个 RBF
和 sigmoid，将六轴响应限制在约 \((0,1.25)\)；精确的 0/1 是解释性参考或 oracle 值。
\(g=(q+\rho)/4\)，\(\alpha_j(g)=1.25\,\sigma(\phi(g)^T\theta_j)\)。

当前实现从训练干预拟合 \(\theta\)，并交替估计名义轨迹；输入阶段是 oracle。
一般公式允许外部提供随阶段变化的可行域，但 `ResponseOperator.condition` 本身保持输入区域的
chart 和 domain，没有学习 \(\mathcal Z\)。不要把这一接口写成已经完成的联合约束学习。
六维拟合器固定关系基为局部六轴，其历史 pose6 顺序为 \(R_xR_yR_z\)；几何 TSR chart 为
\(R_zR_yR_x\)。导入边界必须显式转换。

## 三项贡献的准确表述

1. **表示：** 构造生成元 chart 与有界坐标域组合的 G-TSR；原有证明讨论其对合法单 TSR
   表示族的精确嵌入和严格扩展，不推广到 TSRChain 或任意区域组合。
2. **响应建模：** 将阶段条件响应与可行域分别参数化，提供可学习的逐生成元有限响应及
   明确的未激励、不可观测、拟合失败和模型不匹配状态。
3. **验证：** 在冻结任务与 oracle phase 下，检验对角结构相对共享标量的必要性，并报告
   更灵活模型、跨轴耦合、坐标敏感性和复用边界。独立物理迁移暂列待完成实验。

## 术语表

| 术语 | 本文定义 | 写作边界 |
|---|---|---|
| G-TSR | 沿用仓库方法名，生成元 chart 与坐标域组成的任务区域 | 不擅自扩写名称 |
| 生成元 | 基 \(B_r\) 中一个有物理语义的平移或旋转方向 | 不等同于神经网络特征 |
| 响应规律 \(P_r=D\) | 阶段条件的扰动传递权重 | 不等同于成功概率或可行域 |
| 可行域 \(\mathcal Z\) | 执行可接受的坐标集合 | 与响应分别参数化，不声称统计独立 |
| 阶段 \(\zeta\) | 离散阶段与阶段内进度 | 当前主证据使用 oracle phase |
| 可辨识性 | 在指定观测、干预和自由度处理下能区分响应参数 | 优化器收敛不代表已辨识 |
| 受控规律复用 | 在给定对应关系下使用源响应 | 不等同于未知关系发现或零目标数据 |
| mm-equivalent | 按 0.03 m/rad 换算后报告的混合残差 | 不等同于纯位置误差或装配公差 |

## 文献定位

任务参数化运动学习已使用参考坐标系及统计结构适应新情境。TP-GMM 结合多个坐标系中的均值和
协方差，因此不能把它概括成一个共享标量。对本文的新意，应比较显式的生成元响应语义、干预辨识
和区域接口，而不能声称已有方法都不会按方向适应。
[Calinon, 2016，作者全文](https://calinon.ch/papers/Calinon-JIST2015.pdf)

螺旋几何也已用于分割示教并在新任务实例上规划。仅“使用 screw/SE(3)”不足以构成新意；本工作
需要突出阶段条件的扰动传递和证据边界。
[Mahalingam and Chakraborty, 2022，作者摘要](https://arxiv.org/abs/2209.05672)

这只是已核查的近邻工作定位，不是系统综述或首创性证明。传统 TSR 证明以仓库
[既有数学定义](../../gtsr_expressivity_proofs.md) 为依据；原论文全文链接本次未成功取得，
投稿前仍需重新核对原文。

## 正文组织

1. 用一个阶段切换实例提出选择性响应问题。
2. 定义区域与响应，列明基、阶段、观测和支持范围假设。
3. 给出标量表示不足、局部辨识和坐标变换性质；几何严格包含的长证明放附录。
4. 报告共享标量对照和完整预测基线；更灵活模型更优的结果保留在正文。
5. 报告指标坐标依赖、跨轴耦合和不可观测反例。
6. 独立物理迁移完成后，再决定是否升级为第二条核心实验主线。

当前缺口：统一且具任务意义的误差定义、与装配公差连接的门槛、独立目标物理响应。
本稿从已有证据形成论证，尚未构成完整投稿论文。
