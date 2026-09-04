# G-TSR Sprint E — E-09 baseline comparison scope

状态：`SCOPE-DRAFT`（2026-09-04）

本文件只关闭 Sprint E 的 E-09：在既有 planar oracle-phase 30/8、五 seed
evidence 上完成公平 baseline matrix。它不新增 task、不开启 online phase/transfer，且不改变
geometry kernel、`ResponseOperator` 或 Sprint-E 已冻结的 Pdiag fitting contract。

## 1. 共同合同

每个方法使用同一 seed 的 30 mixed training trajectories、相同 nominal zero intervention、同一
oracle phase `(3,4,5,6)×25` grid、同一 8 isolated held-out interventions 和同一 `mm-equiv`
residual。模型选择（Generic RBF ridge CV、Scalar GP hyperparameter fitting）仅在 30 training
episodes 内完成；不得读取 held-out curve、profile 或成功标签。

所有输出都以同一 nominal `X_0(zeta)` 转为 left action 后再作用于名义 G-TSR：

\[
A_m(zeta,c)=\hat X_m(zeta,c)X_0(zeta)^{-1},\qquad
\hat{\mathcal R}_m=A_m\mathcal R_0.
\]

对 Pdiag/Fixed/Frame/Scalar，`A_m` 必须直接由冻结 `ResponseOperator` 给出；对 Context 与
Full，以上等式只是 direct-pose/update adapter，绝不将其标作 diagonal law 或
`ResponseOperator` output。

## 2. 方法矩阵

| Method | Fitting budget | Output semantics | Claim boundary |
|---|---:|---|---|
| TSR-Fixed | none | `P=0` | static lower control |
| TSR-Frame | none | `P=I` | whole-relation propagation control |
| Phase-Scalar GP | 30 mixed | `P=w(zeta)I`; source GP config | scalar-selectivity control |
| G-TSR Pdiag finite | 30 mixed | frozen `diag(alpha)` | proposed identifiable structured law |
| Context Regression | 30 mixed, grouped-CV only | Generic RBF direct pose/update | unstructured capacity comparator |
| Full Operator | 30 mixed | dense per-progress affine 3x3 direct update | non-diagonal capacity comparator |

`Phase-Scalar GP` and `Context Regression` shall faithfully port the corresponding
frozen sister implementations into test-side code and use `scikit-learn` only as a
test extra. Absence of that extra is an explicit integration skip, not a green E-09
gate.

## 3. Metrics and hard gate

For every `(seed, method)` save: eight held-out full response errors, quotient task
errors (yaw weight zero after clearance), endpoint errors, method status and training
budget. For Pdiag/Scalar/Frame/Fixed also save generator profiles; for Full/Context
profile recovery is **not applicable**.

E-09 passes only if:

1. six method rows are produced for all five seeds with no test leakage;
2. all pose/action adapters satisfy a frozen differential test against their direct
   predicted pose;
3. Pdiag remains below Phase-Scalar GP task error in each seed (the existing E-07
   preregistered condition);
4. Fixed/Frame, Context and Full results are reported paired with Pdiag regardless
   of winner; no Pdiag-win assertion is imposed for the two flexible comparators;
5. all outputs identify this as planar oracle-policy evidence, not a six-generator,
   physical-success or transfer result.

## 4. Implementation sequence

1. Freeze a test-side `baseline_models.py` port, including deterministic scalar GP,
   Generic RBF grouped episode CV and full operator.
2. Extend the loader with the nominal curve required by Fixed/Frame/action adapters.
3. Add differential/action tests for every method family and no-leakage tests.
4. Run all five seed comparison and emit a deterministic test artifact/summary.
5. Re-open the Sprint-E scope only after E-09 evidence is reviewed; then promote it
   from `SCOPE-DRAFT` to `IDENTIFICATION-LOCKED` only if every E-01--E-10 condition
   is actually met.

## 5. Explicit deferrals

- 6D 60-mixed/14-isolated replication;
- learned online phase, phase correspondence and transfer;
- planner/IK/physical rollout success;
- a claim that Full or Context is inferior, or that a direct predictor identifies a
  generator-wise response law.
