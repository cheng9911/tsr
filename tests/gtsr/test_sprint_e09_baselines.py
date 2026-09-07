# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""E-09: planar oracle-phase held-out comparison across five execution seeds.

The independent repeat is the simulator execution seed. Each seed uses only its
30 mixed interventions for fitting/model selection and the same eight isolated
interventions for paired evaluation. This is solver-trajectory-policy evidence;
it is not a six-generator, learned-phase, transfer, or physical-success claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.gtsr._baseline_models import (
    FixedActionModel,
    FrameActionModel,
    FullOperatorModel,
    GenericConditionalRBFModel,
    PDiagActionModel,
    PhaseScalarActionModel,
    apply_actions,
)
from tests.gtsr._maniskill_data import SEEDS, load_seed_dataset, metric_errors, multiseed_root, wrap_angle

pytest.importorskip("h5py")
pytest.importorskip("sklearn")
threadpoolctl = pytest.importorskip("threadpoolctl")

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BINS = 25
RESULTS_ARTIFACT = Path(__file__).resolve().parents[2] / "docs" / "source_data" / "gtsr_e09_results.json"
METHOD_NAMES = (
    "TSR-Fixed",
    "TSR-Frame",
    "Phase-Scalar GP",
    "G-TSR Pdiag finite",
    "Context Regression",
    "Full Operator",
)


@pytest.fixture(autouse=True)
def _single_threaded_blas():
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        yield


def _require_root():
    root = multiseed_root()
    if not (root / "collection_status.json").exists():
        pytest.skip(f"ManiSkill spectrum multiseed data not found at {root}")
    return root


def _models(data):
    common = (data.nominal_curve, data.nominal_frame_xy, data.nominal_frame_yaw)
    return [
        FixedActionModel(data.nominal_curve),
        FrameActionModel(*common),
        PhaseScalarActionModel(*common),
        PDiagActionModel(*common),
        GenericConditionalRBFModel(data.nominal_curve),
        FullOperatorModel(data.nominal_curve),
    ]


def _generator_profile(model, steps):
    if isinstance(model, FixedActionModel) and not isinstance(model, FrameActionModel):
        return np.zeros((steps, 3), dtype=np.float64)
    if type(model) is FrameActionModel:
        return np.ones((steps, 3), dtype=np.float64)
    if hasattr(model, "generator_profile"):
        return model.generator_profile()
    return None


def _action_reconstruction_error(prediction, reconstructed):
    residual = np.asarray(prediction) - np.asarray(reconstructed)
    residual = residual.copy()
    residual[..., 2] = wrap_angle(residual[..., 2])
    return float(np.max(np.abs(residual)))


def _evaluate_seed(data):
    """Fit on 30 mixed episodes and return six held-out result records."""
    if len(data.train_contexts) != 30 or len(data.test_contexts) != 8:
        raise AssertionError("E-09 requires the frozen 30 mixed / 8 isolated split")
    yaw_weights = (data.phase_codes < 5).astype(np.float64)
    rows = []
    for model in _models(data):
        model.fit(data.train_contexts, data.train_curves, data.progress, data.phase_codes)
        prediction = model.predict(data.test_contexts)
        reconstruction = apply_actions(model.actions(data.test_contexts), data.nominal_curve)
        response_errors, response_endpoint_errors = metric_errors(prediction, data.test_curves)
        task_errors, task_endpoint_errors = metric_errors(prediction, data.test_curves, yaw_weights=yaw_weights)
        rows.append(
            {
                "seed": data.seed,
                "method": model.name,
                "status": getattr(model, "fit_status", "ok"),
                "training_episodes": 0 if model.name in {"TSR-Fixed", "TSR-Frame"} else 30,
                "response_errors": response_errors,
                "response_endpoint_errors": response_endpoint_errors,
                "task_errors": task_errors,
                "task_endpoint_errors": task_endpoint_errors,
                "generator_profile": _generator_profile(model, len(data.progress)),
                "action_reconstruction_error": _action_reconstruction_error(prediction, reconstruction),
            }
        )
    return rows


def _rounded(values):
    return np.round(np.asarray(values, dtype=np.float64), decimals=9).tolist()


def _result_artifact(rows, datasets):
    """Build the canonical machine-readable E-09 Source Data object."""
    artifact = {
        "schema_version": 1,
        "claim_scope": "planar oracle-phase solver-trajectory-policy evidence",
        "independent_unit": "simulator execution seed",
        "metric_unit": "mm-equivalent",
        "seeds": {},
    }
    for seed in SEEDS:
        data = datasets[seed]
        per_seed = [row for row in rows if row["seed"] == seed]
        artifact["seeds"][str(seed)] = {
            "training_episodes": len(data.train_contexts),
            "heldout_episodes": len(data.test_contexts),
            "conditions": [
                {
                    "condition_id": int(condition_id),
                    "generator": generator,
                    "context": _rounded(context),
                }
                for condition_id, generator, context in zip(
                    data.test_condition_ids,
                    data.test_generators,
                    data.test_contexts,
                )
            ],
            "methods": {
                row["method"]: {
                    "status": row["status"],
                    "training_budget": int(row["training_episodes"]),
                    "response_errors": _rounded(row["response_errors"]),
                    "response_endpoint_errors": _rounded(row["response_endpoint_errors"]),
                    "task_errors": _rounded(row["task_errors"]),
                    "task_endpoint_errors": _rounded(row["task_endpoint_errors"]),
                    "generator_profile": (
                        None if row["generator_profile"] is None else _rounded(row["generator_profile"])
                    ),
                }
                for row in per_seed
            },
        }

    summary = {}
    for method in METHOD_NAMES:
        per_method = [row for row in rows if row["method"] == method]
        summary[method] = {}
        for field in (
            "response_errors",
            "response_endpoint_errors",
            "task_errors",
            "task_endpoint_errors",
        ):
            seed_means = np.asarray([np.mean(row[field]) for row in per_method])
            summary[method][field] = {
                "seed_means": _rounded(seed_means),
                "mean": round(float(seed_means.mean()), 9),
                "sd": round(float(seed_means.std(ddof=1)), 9),
            }
    artifact["summary"] = summary

    pdiag = np.asarray(summary["G-TSR Pdiag finite"]["task_errors"]["seed_means"])
    bootstrap_indices = np.indices((len(SEEDS),) * len(SEEDS)).reshape(len(SEEDS), -1).T
    contrasts = {}
    for method in METHOD_NAMES:
        if method == "G-TSR Pdiag finite":
            continue
        baseline = np.asarray(summary[method]["task_errors"]["seed_means"])
        differences = pdiag - baseline
        bootstrap = differences[bootstrap_indices].mean(axis=1)
        contrasts[method] = {
            "seed_differences": _rounded(differences),
            "mean": round(float(differences.mean()), 9),
            "bootstrap_95_interval": _rounded(np.quantile(bootstrap, [0.025, 0.975])),
        }
    artifact["pdiag_minus_baseline_task_error"] = contrasts
    return artifact


def _assert_artifact_matches(actual, expected):
    """Check Source Data structure exactly and supported-stack numerics tightly."""
    for field in ("schema_version", "claim_scope", "independent_unit", "metric_unit"):
        assert actual[field] == expected[field]
    assert set(actual["seeds"]) == set(expected["seeds"])
    error_fields = (
        "response_errors",
        "response_endpoint_errors",
        "task_errors",
        "task_endpoint_errors",
    )
    for seed in actual["seeds"]:
        actual_seed = actual["seeds"][seed]
        expected_seed = expected["seeds"][seed]
        assert actual_seed["training_episodes"] == expected_seed["training_episodes"]
        assert actual_seed["heldout_episodes"] == expected_seed["heldout_episodes"]
        assert len(actual_seed["conditions"]) == len(expected_seed["conditions"])
        for actual_condition, expected_condition in zip(actual_seed["conditions"], expected_seed["conditions"]):
            assert actual_condition["condition_id"] == expected_condition["condition_id"]
            assert actual_condition["generator"] == expected_condition["generator"]
            np.testing.assert_allclose(actual_condition["context"], expected_condition["context"], atol=1e-9, rtol=0)
        assert set(actual_seed["methods"]) == set(expected_seed["methods"])
        for method in actual_seed["methods"]:
            actual_method = actual_seed["methods"][method]
            expected_method = expected_seed["methods"][method]
            assert actual_method["status"] == expected_method["status"]
            assert actual_method["training_budget"] == expected_method["training_budget"]
            for field in error_fields:
                # SciPy 1.10--1.17 least-squares termination shifts a few Pdiag
                # errors by at most 0.0064 mm-equiv; 0.01 is the frozen portability bound.
                np.testing.assert_allclose(actual_method[field], expected_method[field], atol=1e-2, rtol=0)
            assert (actual_method["generator_profile"] is None) == (expected_method["generator_profile"] is None)
            if actual_method["generator_profile"] is not None:
                np.testing.assert_allclose(
                    actual_method["generator_profile"],
                    expected_method["generator_profile"],
                    atol=5e-3,
                    rtol=0,
                )

    assert set(actual["summary"]) == set(expected["summary"])
    for method in actual["summary"]:
        for field in error_fields:
            for statistic in ("seed_means", "mean", "sd"):
                np.testing.assert_allclose(
                    actual["summary"][method][field][statistic],
                    expected["summary"][method][field][statistic],
                    atol=1e-2,
                    rtol=0,
                )
    assert set(actual["pdiag_minus_baseline_task_error"]) == set(expected["pdiag_minus_baseline_task_error"])
    for method in actual["pdiag_minus_baseline_task_error"]:
        for statistic in ("seed_differences", "mean", "bootstrap_95_interval"):
            np.testing.assert_allclose(
                actual["pdiag_minus_baseline_task_error"][method][statistic],
                expected["pdiag_minus_baseline_task_error"][method][statistic],
                atol=1e-2,
                rtol=0,
            )


def test_E09_five_seed_fair_baseline_matrix():
    """Close E-09 without treating the 8 within-seed conditions as n=40."""
    root = _require_root()
    rows = []
    datasets = {}
    for seed in SEEDS:
        data = load_seed_dataset(root / "rollouts" / f"seed_{seed}.h5", BINS)
        datasets[seed] = data
        rows.extend(_evaluate_seed(data))

    assert len(rows) == len(SEEDS) * len(METHOD_NAMES)
    assert {(row["seed"], row["method"]) for row in rows} == {
        (seed, method) for seed in SEEDS for method in METHOD_NAMES
    }
    assert all(row["status"] == "ok" for row in rows)
    assert all(len(row["response_errors"]) == 8 for row in rows)
    assert all(len(row["response_endpoint_errors"]) == 8 for row in rows)
    assert all(len(row["task_errors"]) == 8 for row in rows)
    assert all(len(row["task_endpoint_errors"]) == 8 for row in rows)
    assert all(row["action_reconstruction_error"] < 1e-9 for row in rows)

    structured = {"TSR-Fixed", "TSR-Frame", "Phase-Scalar GP", "G-TSR Pdiag finite"}
    assert all((row["generator_profile"] is not None) == (row["method"] in structured) for row in rows)
    assert all(row["generator_profile"] is None or row["generator_profile"].shape == (4 * BINS, 3) for row in rows)
    assert all(row["training_episodes"] == (0 if row["method"] in {"TSR-Fixed", "TSR-Frame"} else 30) for row in rows)

    # The only preregistered superiority gate: Pdiag versus scalar, paired by seed.
    # Flexible direct comparators remain descriptive regardless of which method wins.
    paired_differences = {}
    for seed in SEEDS:
        per_method = {row["method"]: row for row in rows if row["seed"] == seed}
        pdiag_errors = per_method["G-TSR Pdiag finite"]["task_errors"]
        for method in METHOD_NAMES:
            if method != "G-TSR Pdiag finite":
                paired_differences[seed, method] = pdiag_errors - per_method[method]["task_errors"]
        pdiag = np.mean(pdiag_errors)
        scalar = np.mean(per_method["Phase-Scalar GP"]["task_errors"])
        assert pdiag < scalar
    assert set(paired_differences) == {
        (seed, method) for seed in SEEDS for method in METHOD_NAMES if method != "G-TSR Pdiag finite"
    }
    assert all(
        differences.shape == (8,) and np.all(np.isfinite(differences)) for differences in paired_differences.values()
    )

    # Bind the executable experiment to its paper-facing Source Data. A model,
    # metric, split or profile change cannot leave the rounded report silently stale.
    with RESULTS_ARTIFACT.open(encoding="utf-8") as source:
        expected_artifact = json.load(source)
    _assert_artifact_matches(_result_artifact(rows, datasets), expected_artifact)


def _write_result_artifact():
    root = _require_root()
    rows = []
    datasets = {}
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        for seed in SEEDS:
            data = load_seed_dataset(root / "rollouts" / f"seed_{seed}.h5", BINS)
            datasets[seed] = data
            rows.extend(_evaluate_seed(data))
    artifact = _result_artifact(rows, datasets)
    RESULTS_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_ARTIFACT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] != ["--write-artifact"]:
        raise SystemExit("usage: python -m tests.gtsr.test_sprint_e09_baselines --write-artifact")
    _write_result_artifact()
