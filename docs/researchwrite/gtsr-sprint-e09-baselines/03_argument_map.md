# Argument map

## Scientific tension

- Known: finite Pdiag reproduces the frozen profile and beats a historical shared
  scalar trajectory baseline.
- Unknown: whether that comparison uses a single, fair action-to-region interface
  and how Pdiag trades off against non-structured direct predictors.
- Why it matters: a scalar-only win does not by itself establish that the proposed
  representation is preferable to a flexible black-box predictor.

## Central research question

Under one frozen oracle-phase intervention distribution, what held-out response
error and interpretable-law trade-off does finite diagonal G-TSR response modeling
have relative to fixed, frame, scalar, context-regression and dense-operator
baselines?

## Central thesis

E-09 can establish a fair, bounded comparison protocol; it can establish scalar
insufficiency only where the preregistered per-seed gate succeeds, while Full and
Context outcomes remain descriptive capacity comparisons.

## Supporting arguments

### Argument 1
- Claim: Fixed, Frame and Scalar can be stated as nested action laws.
- Evidence: `P=0`, `P=I`, and `P=w(zeta)I` under the frozen ResponseOperator.
- Limitation: the scalar GP needs a test-only scikit-learn implementation.

### Argument 2
- Claim: Direct models can be compared without pretending they are response laws.
- Evidence: convert `X_hat(zeta,c)` to `A_hat=X_hat X_0(zeta)^-1`, then apply
  `A_hat` to the same nominal region.
- Limitation: this evaluates output prediction, not an identifiable generator law.

### Argument 3
- Claim: five paired seed results prevent a single-seed narrative.
- Evidence: frozen five-seed manifest and fixed 30/8 split.
- Limitation: five execution seeds are not a population-level generalization proof.

## Counterarguments / alternative explanations

- A flexible model may have lower error because it has more capacity, not because it
  has a better response representation. Report this, do not relabel it a law.
- Pdiag may win through smoothness regularization on this distribution. This is a
  bounded sample-efficiency observation, not a universal result.

## Final move

Freeze E-09 only when every method follows the same data and evaluation contract,
and the results preserve both negative and non-winning outcomes.
