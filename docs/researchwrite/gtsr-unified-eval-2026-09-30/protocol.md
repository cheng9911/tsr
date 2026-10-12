# G-TSR 统一评价开发协议

2026-09-30，探索性开发实验。本文件在本轮新比较结果生成前建立；既有数据已经用于开发，
不称为前瞻确认性实验。历史协议与结果保持原样。

## 数据和预算

使用两个 arm（keyed、circular_honest）、三个既有 execution seeds、六个冻结 N=30 子集
（12–17），共36个配对计算单元。独立单位仍是 seed，每个 arm 为 n=3。
每单元训练30 mixed，评价其余30 mixed及14 isolated，baseline只作共同名义校准。
四个 oracle phase各25点，指标明确称为阶段归一化误差，不称为原始时间轴误差。
所有方法共享训练/评价 IDs、名义关系 frame 和一条 zero-intervention baseline。
禁止用 held-out 数据选择方法超参数、重新估计名义量或筛除失败单元。

## 两条比较轨道

1. **共同名义响应轨道：** 直接预测器先生成原生预测 Y(c)及Y(0)，定义
   A(c)=Y(c)Y(0)^-1，再作用于共同baseline X0。Pdiag使用其已拟合alpha的有限动作；
   Fixed/Frame分别使用0/I。此轨道比较响应，不把每个方法自身intercept当成共同名义。
2. **原生预测轨道：** 各模型原始预测直接与观测比较，保留名义轨迹估计误差。
   Pdiag需要完整theta/nominal artifact，不能由grid伪造完整拟合模型。

## 指标

主报告同时列出两种坐标一致的完整6D残差，不按结果选择其中更有利的一种：

- 关系局部动作：delta(C0^-1 Ahat C0, C0^-1 Aobs C0)，Aobs=Xi X0^-1。
- 位姿预测：delta(Xhat,Xi)。

delta(U,V)=||S Log(U^-1 V)||，S=diag(1,1,1,.03,.03,.03)，报告边界乘1000，
单位mm-equivalent。完整pose残差另列平移范数mm和旋转角deg。
历史世界动作残差仅作对照；不移植历史指标的equivalence/superiority门槛。
同一物理预测被世界平移/旋转后，局部动作及pose残差必须保持一致。

另外报告明确的**轴向对称任务代理误差**：位置使用欧氏距离；方向使用两姿态插入轴的夹角，
在keyed的前两个phase保留完整旋转角，在后两个phase及circular全程忽略绕自身轴旋转。
该代理不等于真实碰撞/接触可行性，也不从预测phase推断释放事件。它用于区分示教轨迹模仿与
已声明对称性下的任务相关偏差，不用于证明yaw响应已辨识。

按phase、generator和split保留误差；先条件内平均、再subset平均、最后跨seed报告均值及SD。
mixed与isolated分别汇总；isolated总体同时给condition平均和六generator等权平均。
全部方法失败保留状态与原因；不把缺失指标置零。不作新显著性或确认性通过声明。

## 方法和选择

保留既有六维七方法：Frame-weighted affine、Phase scalar GP affine、TP-GMM additive、
TP-GMM SE3、Full affine、Pointwise diagonal affine、Pdiag finite；补充Fixed、Frame、
有限作用的Scalar GP，以及六维Context RBF，共11方法。

Pdiag使用原生产拟合器配置（24 RBF、宽度.065、alpha_max1.25、平滑.1、名义迭代3）。
六维既有基线使用sister源码当前配置并保存SHA。TP-GMM候选K=(2,4,6,8)，5折episode分组CV，
CV一次初始化、最终3次初始化，不以测试误差选K。
Context RBF沿用E09的phase-local RBF构造（每phase10基、宽度.16、16个ridge候选、5折episode CV），
推广到六轴并采用冻结CONTEXT_SCALE归一化。它是本轮明确新增的六维实现。
有限Scalar GP复用训练得到的标量profile，通过与Pdiag相同的有限SE3作用实现；其原生轨道使用
训练得到的scalar affine零context intercept作为名义轨迹，共同名义轨道才使用baseline。

## 必需响应与控制偏好

只读重析既有geometry-probe HDF5，并核对collector、环境评价和solver的实际语义。
保留全部attempt而不只选满足预期的probe。入口前匹配与错配检查真实clearance/contact，
入口后不同yaw检查实际姿态与环境success。受限证据不升级为跨对象总体结论。
加入同一任务等价类内不同yaw轨迹的受控数值检查：完整轨迹误差应改变，轴向对称任务代理应不变。
这只能证明评价/可辨识性边界，不能冒充新增物理执行。
