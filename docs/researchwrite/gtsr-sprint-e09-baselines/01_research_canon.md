# Research canon

## Literature / source facts

- The frozen sister benchmark defines Frame-weighted, Phase scalar GP, Generic RBF,
  Full operator, pointwise Pdiag, and finite Pdiag in
  `../maniskill_spectrum/phase_switch_symmetry/phase_switch_baselines.py`.
- Its five-seed result establishes only that finite Pdiag beats the shared-scalar
  baseline on task error for each registered seed; it does not establish universal
  superiority over Generic RBF or Full operator.

## Experimental facts

- Every seed has 30 successful mixed training interventions, 8 nonzero isolated
  held-out interventions, one zero-intervention nominal reference, and oracle
  phases 3--6 with 25 resampled bins each.
- Sprint E already reproduces finite Pdiag and uses the same data loader in
  `tests/gtsr/_maniskill_data.py`.

## Model facts

- Sprint-D `ResponseOperator` supplies the frozen left action and region assembly
  semantics. Pdiag-finite uses that action exactly under the permitted planar-to-6D
  embedding.
- Direct regressors produce a pose/update rather than a diagonal law. They can be
  composed fairly by converting each predicted pose into the left action relative to
  the same nominal pose at the same oracle phase.

## Terminology

- **Frame** means identity propagation `P=I`, not the sister repository's historical
  class name `FrameWeightedModel`.
- **Scalar** means `P=w(zeta)I`; its fitted `w` follows the frozen Phase scalar GP.
- **Context regression** means the frozen Generic RBF conditional regressor.
- **Full operator** means the frozen dense per-progress affine 3x3 operator.

## Forbidden claims

- No claim that Pdiag must outperform Full operator or Generic RBF on held-out error.
- No comparison that gives an oracle or test-set information unavailable to the other
  learned methods.
- No claim that a direct regressor is a diagonal response law or uses `ResponseOperator`.

## Unresolved claims

- Whether the environment has scikit-learn is not a scientific uncertainty: the
  eventual E-09 integration test will skip only if its explicitly declared test
  extra is absent; the locked benchmark requires it for Scalar GP and Generic RBF.
