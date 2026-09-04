# G-TSR Sprint E — E-09 fair baseline comparison plan

**Goal:** Close E-09 with an oracle-phase planar baseline matrix that uses one
frozen 30-mixed/8-isolated split, one nominal trajectory, and auditable action
semantics. No geometry kernel or runtime API changes are allowed.

**Spec:** `docs/superpowers/specs/2026-09-04-gtsr-sprint-e09-baseline-scope.md`

## Task 1 — Test-side data and baseline interface

**Files:**

- Modify `tests/gtsr/_maniskill_data.py`
- Create `tests/gtsr/_baseline_models.py`
- Create `tests/gtsr/test_baseline_models.py`

**Contract:** Extend `SeedDataset` with the resampled zero-intervention nominal
curve. Define a test-only common interface `fit(...); predict(contexts)` returning
`(N, 100, 3)` planar poses. Each baseline stores enough information to expose a
nominal-relative left action for differential validation.

**Acceptance:** loader preserves existing extraction; Fixed predicts nominal;
Frame (`P=I`) has exact SE(2)/SE(3) action agreement; malformed dimensions and
optional `scikit-learn` absence are visible.

## Task 2 — Structured action baselines

**Files:**

- Modify `tests/gtsr/_baseline_models.py`
- Modify `tests/gtsr/test_baseline_models.py`

**Contract:** Implement Fixed (`P=0`), Frame (`P=I`), Scalar GP
(`P=w(zeta)I`, frozen GP configuration), and an adapter for the existing finite
Pdiag profile. These models use the identical nominal frame and SE(2) left action;
the planar-to-SE(3) action must agree with `ResponseOperator` to `1e-9`.

**Acceptance:** scalar is visibly one coefficient shared by all three generators;
Pdiag has three coefficients; zero relation is the nominal curve for every
structured baseline; no learned model reads held-out data.

## Task 3 — Direct capacity baselines

**Files:**

- Modify `tests/gtsr/_baseline_models.py`
- Modify `pyproject.toml`, `uv.lock`
- Modify `tests/gtsr/test_baseline_models.py`

**Contract:** Faithfully port sister `FullOperatorModel` (dense affine 3x3,
ridge `1e-8`) and `GenericConditionalRBF` (phase-local RBF, grouped 5-fold
episode CV, frozen feature scales). Both output direct pose predictions and are
adapted by `A=X_hat X_0^-1`; they are never exposed as `ResponseOperator`/law
classes. Add `scikit-learn` under test extras only.

**Acceptance:** direct-output-to-action reconstruction is exact; CV groups are
episodes, not progress rows; test data are absent during all fitting/CV.

## Task 4 — Five-seed E-09 integration gate

**Files:**

- Create `tests/gtsr/test_sprint_e09_baselines.py`

**Contract:** For all five frozen seeds fit the six methods using only the training
split; emit in-memory deterministic rows for response/task/endpoint errors and
method status. Compare Pdiag against each method pairwise per seed. Require the
registered Pdiag < Scalar task-error condition per seed; preserve Full/Context
outcomes descriptively without a forced winner.

**Acceptance:** 30/8 provenance is rechecked; six methods × five seeds are present;
all actions reconstruct their direct pose prediction; no skip is counted as an E-09
pass; planar-only claim is encoded in test/docstrings.

## Task 5 — Verification and freeze decision

**Files:**

- Modify Sprint-E scope only if every E-01--E-10 gate has evidence.

**Commands:**

```bash
pytest -q tests/gtsr/test_baseline_models.py tests/gtsr/test_sprint_e09_baselines.py
pytest -q
ruff check … && ruff format --check …
uv sync --locked --all-extras --dev --dry-run
git diff --check
```

**Acceptance:** full regression stays green, test extras lock cleanly, and a
separate review finds no information leakage or action-semantics conflation. Only
then decide whether Sprint E can become `IDENTIFICATION-LOCKED`.
