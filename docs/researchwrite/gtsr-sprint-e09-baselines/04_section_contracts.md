# Section contracts

## Section: fairness contract

- Purpose: define identical information and action semantics.
- Inputs: Sprint-E scope §§2, 5, 8; frozen sister manifest.
- Allowed claims: every learned method receives the same 30 mixed trajectories,
  oracle phase grid, nominal reference and held-out 8 interventions.
- Forbidden claims: direct models are diagonal laws; test data inform fit or tuning.
- Required evidence: explicit train/test paths and model-specific hyperparameters.
- Validation checklist: no held-out fit; no hidden oracle law; one nominal reference.

## Section: metrics and failure semantics

- Purpose: make E-09 reproducible and interpretation conservative.
- Inputs: frozen `metric_errors`, five-seed summary convention.
- Allowed claims: per-seed response/task error, paired differences, descriptive rank,
  profile recovery where a generator law exists.
- Forbidden claims: significance/generalization beyond five seeds; Pdiag superiority
  over flexible baselines without observed evidence.
- Required evidence: saved per-seed per-method metrics and failed/skip statuses.
- Validation checklist: scalar gate checked per seed; Full/Context reported even when
  they win; unavailable optional dependency is explicit skip, never a silent pass.
