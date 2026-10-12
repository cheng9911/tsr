"""Reproduce descriptive paper evidence and run bounded falsification experiments.

Run from the repository root with its tsr-dev Python. No production-model fitting, simulator
execution, source-data edits, or confirmatory claims are performed here. The
synthetic hypothesis-class diagnostic fits closed-form least-squares models;
no production response profiles are fitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tsr.gtsr.metrics import se3_exp, se3_hat, se3_log  # noqa: E402
from tsr.gtsr.response import ResponseOperator  # noqa: E402

F1 = ROOT / "docs/superpowers/specs/2026-09-08-gtsr-sprint-f1-adequacy-source-data.json"
E09 = ROOT / "docs/source_data/gtsr_e09_results.json"
SCALE = np.array([1, 1, 1, 0.03, 0.03, 0.03])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def error(a, b):
    return float(np.linalg.norm(SCALE * se3_log(np.linalg.solve(a, b))))


def frozen_results():
    f1, e09 = json.loads(F1.read_text()), json.loads(E09.read_text())
    keys = [(r["arm"], r["seed"], r["generator"]) for r in f1["per_seed"]]
    expected = {
        (a, s, g)
        for a in f1["frozen_inputs"]["arms"]
        for s in f1["frozen_inputs"]["seeds"]
        for g in f1["frozen_inputs"]["generators"]
    }
    assert len(keys) == len(set(keys)) == 36 and set(keys) == expected
    with F1.with_suffix(".csv").open() as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_map = {(r["arm"], int(r["seed"]), r["generator"]): r for r in csv_rows}
    assert len(csv_rows) == 36 and set(csv_map) == expected
    f1_summary = []
    for arm in f1["frozen_inputs"]["arms"]:
        for generator in f1["frozen_inputs"]["generators"]:
            rows = [r for r in f1["per_seed"] if r["arm"] == arm and r["generator"] == generator]
            oracle = "fixed" if arm == "circular_honest" and generator == "yaw" else "frame"
            for row in rows:
                assert row["primary_direction"] == oracle
                assert row["primary_diff"] == row[f"diff_vs_{oracle}"]
                old = csv_map[(arm, row["seed"], generator)]
                assert abs(float(old["primary_diff"]) - row["primary_diff"]) < 1e-14
            f1_summary.append(
                {
                    "arm": arm,
                    "generator": generator,
                    "oracle": oracle,
                    "n_seeds": 3,
                    "response_error_mm": float(np.mean([r["response_error"] for r in rows]) * 1000),
                    "primary_diff_mm": [r["primary_diff"] * 1000 for r in rows],
                    "primary_diff_mean_mm": float(np.mean([r["primary_diff"] for r in rows]) * 1000),
                }
            )
    methods = list(next(iter(e09["seeds"].values()))["methods"])
    e09_summary = {}
    for method in methods:
        per_seed = [float(np.mean(s["methods"][method]["task_errors"])) for s in e09["seeds"].values()]
        e09_summary[method] = {
            "per_seed_mm": per_seed,
            "mean_mm": float(np.mean(per_seed)),
            "sd_mm": float(np.std(per_seed, ddof=1)),
            "n_seeds": len(per_seed),
        }
    ours = np.array(e09_summary["G-TSR Pdiag finite"]["per_seed_mm"])
    for method, contrast in e09["pdiag_minus_baseline_task_error"].items():
        differences = ours - e09_summary[method]["per_seed_mm"]
        np.testing.assert_allclose(differences, contrast["seed_differences"], atol=2e-9, rtol=0)
    return {
        "f1": f1_summary,
        "e09": e09_summary,
        "scope": "Read-only reaggregation of stored development results; no profile refits.",
    }


def finite_operator_experiment():
    rng = np.random.default_rng(20260930)
    oracle_errors, covariance_errors = [], []
    for _ in range(128):
        ref = expm(se3_hat(rng.normal(size=6) * [0.3, 0.3, 0.3, 0.4, 0.4, 0.4]))
        basis = np.eye(6) + rng.normal(size=(6, 6)) * 0.05
        c = rng.normal(size=6) * [0.02, 0.02, 0.02, 0.1, 0.1, 0.1]
        alpha = rng.uniform(0, 1.25, size=6)
        relation = ref @ expm(se3_hat(basis @ c))
        operator = ResponseOperator(ref, basis)
        predicted = operator.action(relation, alpha)
        independent = ref @ expm(se3_hat(basis @ (alpha * c))) @ np.linalg.inv(ref)
        oracle_errors.append(float(np.max(np.abs(predicted - independent))))
        g = expm(se3_hat(rng.normal(size=6) * [1, 1, 1, 0.3, 0.3, 0.3]))
        reframed = ResponseOperator(g @ ref, basis).action(g @ relation, alpha)
        covariance_errors.append(float(np.max(np.abs(reframed - g @ predicted @ np.linalg.inv(g)))))
    assert max(oracle_errors) < 1e-10 and max(covariance_errors) < 1e-10
    # A selective map need not preserve products, even though each output is SE(3).
    alpha = np.array([1, 1, 1, 0, 0, 0])
    op = ResponseOperator(np.eye(4), np.eye(6))
    a = se3_exp([0, 0, 0, 0, 0, 0.4])
    b = se3_exp([0.02, 0, 0, 0, 0, 0])
    composition_error = error(op.action(a @ b, alpha), op.action(a, alpha) @ op.action(b, alpha))
    assert composition_error > 1e-4
    return {
        "n_trials": 128,
        "reference_expm_max_abs": max(oracle_errors),
        "world_frame_covariance_max_abs": max(covariance_errors),
        "group_composition_counterexample_mm": composition_error * 1000,
    }


def hypothesis_class_experiment():
    """Closed-form local-coordinate models; not the production RBF estimator."""
    rows = []
    for seed in range(20):
        rng = np.random.default_rng(20260930 + seed)
        train, test = rng.normal(size=(128, 6)), rng.normal(size=(512, 6))
        for case in ("diagonal", "cross_coupled"):
            truth = np.diag([1, 1, 1, 1, 1, 0])
            if case == "cross_coupled":
                truth = truth.astype(float)
                truth[0, 1] = 0.4
            clean = train @ truth.T
            observed = clean + rng.normal(scale=0.02, size=clean.shape)
            target = test @ truth.T
            diag = np.sum(train * observed, axis=0) / np.sum(train * train, axis=0)
            scalar = np.sum(train * observed) / np.sum(train * train)
            full = np.linalg.lstsq(train, observed, rcond=None)[0]
            for method, pred in (("scalar", scalar * test), ("diagonal", diag * test), ("dense", test @ full)):
                rows.append(
                    {
                        "seed": seed,
                        "case": case,
                        "method": method,
                        "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
                    }
                )
    summary = []
    for case in ("diagonal", "cross_coupled"):
        for method in ("scalar", "diagonal", "dense"):
            vals = [r["rmse"] for r in rows if r["case"] == case and r["method"] == method]
            summary.append(
                {
                    "case": case,
                    "method": method,
                    "mean_rmse": float(np.mean(vals)),
                    "sd_rmse": float(np.std(vals, ddof=1)),
                }
            )
    return {
        "scope": "Synthetic normalized log-coordinate regression, known nominal; not physical evidence.",
        "train_n": 128,
        "test_n": 512,
        "noise_sd": 0.02,
        "n_simulation_seeds": 20,
        "scalar_noiseless_population_rmse_diagonal_case": float(np.sqrt(5 / 36)),
        "diagonal_noiseless_population_rmse_coupled_case": float(np.sqrt(0.16 / 6)),
        "summary": summary,
        "rows": rows,
    }


def origin_experiment():
    a = np.eye(4)
    b = se3_exp([0, 0, 0, 0, 0, 0.1])
    rows = []
    for shift in (0, 0.1, 0.5, 1.0):
        g = np.eye(4)
        g[0, 3] = shift
        inv = np.linalg.inv(g)
        rows.append(
            {
                "origin_shift_m": shift,
                "world_action_error_mm": error(g @ a @ inv, g @ b @ inv) * 1000,
                "pose_error_mm": error(g @ a, g @ b) * 1000,
            }
        )
    assert rows[-1]["world_action_error_mm"] > 30 * rows[0]["world_action_error_mm"]
    assert abs(rows[-1]["pose_error_mm"] - rows[0]["pose_error_mm"]) < 1e-10
    return rows


def raw_yaw_controls():
    """Fresh read-only HDF5 checks, using Frame/Fixed, without learned profiles."""
    import h5py

    from tests.gtsr._maniskill_data import multiseed_root
    from tests.gtsr._maniskill_data_se3 import (
        ARMS,
        SE3_SEEDS,
        load_se3_isolated,
        load_se3_rollout,
        load_se3_subsets,
        n30_subsets,
        nominal_frame_se3,
        se3_rollouts_root,
    )
    from tsr.gtsr.profile6 import se3_from_pose6

    records, hashes, nominal_records = [], {}, []
    source = json.loads(F1.read_text())
    frozen = source["per_seed"]
    for filename, key in (
        ("se3_fixed_contexts.json", "SE3_CONTEXT_MANIFEST_SHA256"),
        ("se3_experiment.json", "SE3_EXPERIMENT_SHA256"),
        ("se3_subsets.json", "SE3_SUBSET_MANIFEST_SHA256"),
    ):
        manifest = multiseed_root() / filename
        hashes[str(manifest)] = sha(manifest)
        assert hashes[str(manifest)] == source["frozen_inputs"][key]
    for arm in ARMS:
        for seed in SE3_SEEDS:
            path = se3_rollouts_root() / f"{arm}_seed_{seed}.h5"
            hashes[str(path)] = sha(path)
            data = load_se3_isolated(path, 25)
            mixed = load_se3_rollout(path, 25)
            references = []
            with h5py.File(path, "r") as handle:
                for subset in n30_subsets(load_se3_subsets(multiseed_root())):
                    indices = [mixed["mixed_cids"].index(cid) for cid in subset["source_condition_ids"]]
                    nominal = nominal_frame_se3(
                        handle, [mixed["mixed_keys"][i] for i in indices], mixed["mixed_contexts"][indices]
                    )
                    references.append(se3_from_pose6(nominal))
                    nominal_records.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "subset_id": subset["subset_id"],
                            "source_condition_ids": subset["source_condition_ids"],
                            "nominal_transform": references[-1].tolist(),
                        }
                    )
            assert len(references) == 6
            baseline = [se3_from_pose6(v) for v in data["baseline_curve"]]
            for method in ("frame", "fixed"):
                values = {"world": [], "shifted": [], "local": [], "pose": [], "shifted_pose": []}
                shift = np.eye(4)
                shift[0, 3] = 1.0
                shift_inv = np.linalg.inv(shift)
                for ref in references:
                    ref_inv = np.linalg.inv(ref)
                    operator = ResponseOperator(ref, np.eye(6))
                    for i, gen in enumerate(data["isolated_generators"]):
                        if gen != "yaw":
                            continue
                        relation = ref @ se3_from_pose6(data["isolated_contexts"][i])
                        predicted = operator.action(relation, np.ones(6) if method == "frame" else np.zeros(6))
                        for base, pose in zip(baseline, data["isolated_curves"][i]):
                            observed_pose = se3_from_pose6(pose)
                            observed = observed_pose @ np.linalg.inv(base)
                            values["world"].append(error(predicted, observed))
                            values["shifted"].append(error(shift @ predicted @ shift_inv, shift @ observed @ shift_inv))
                            values["local"].append(error(ref_inv @ predicted @ ref, ref_inv @ observed @ ref))
                            values["pose"].append(error(predicted @ base, observed_pose))
                            values["shifted_pose"].append(error(shift @ predicted @ base, shift @ observed_pose))
                row = {
                    "arm": arm,
                    "seed": seed,
                    "method": method,
                    **{f"{k}_mm": float(np.mean(v) * 1000) for k, v in values.items()},
                }
                stored = next(r for r in frozen if r["arm"] == arm and r["seed"] == seed and r["generator"] == "yaw")
                expected = 1000 * (stored["response_error"] - stored[f"diff_vs_{method}"])
                row["frozen_control_mm"] = expected
                assert abs(row["world_mm"] - expected) < 1e-7, (arm, seed, method, row["world_mm"], expected)
                assert abs(row["pose_mm"] - row["shifted_pose_mm"]) < 1e-8
                records.append(row)
    return {
        "rows": records,
        "input_sha256": hashes,
        "nominal_reconstructions": nominal_records,
        "scope": "Six existing development rollouts, four yaw conditions each, 100 phase-normalized samples. "
        "Origin shift is a passive coordinate change, not a new physical rollout. "
        "Fitted F1 predictions are not recomputed.",
    }


def plot_results(result, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), layout="constrained")
    labels = ["TSR-Fixed", "TSR-Frame", "Phase-Scalar GP", "G-TSR Pdiag finite", "Context Regression", "Full Operator"]
    ax = axes[0, 0]
    data = result["stored_evidence"]["e09"]
    ax.bar(np.arange(6), [data[m]["mean_mm"] for m in labels], color=["#999999"] * 3 + ["#0072B2"] + ["#999999"] * 2)
    for i, name in enumerate(labels):
        ax.scatter(np.full(5, i), data[name]["per_seed_mm"], s=18, color="black", zorder=3)
    ax.set(
        xticks=np.arange(6),
        xticklabels=["Fixed", "Frame", "Scalar", "Pdiag", "RBF", "Dense"],
        ylabel="Task error (mm-equivalent)",
        title="A  Stored planar results: five execution seeds",
    )
    ax = axes[0, 1]
    f1 = [r for r in result["stored_evidence"]["f1"] if r["generator"] == "yaw"]
    for i, r in enumerate(f1):
        ax.scatter(np.full(3, i), r["primary_diff_mm"], color=["#0072B2", "#D55E00"][i], s=30)
        ax.plot([i - 0.17, i + 0.17], [r["primary_diff_mean_mm"]] * 2, color="black")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set(
        xticks=[0, 1],
        xticklabels=["Keyed vs Frame", "Circular vs Fixed"],
        ylabel="Pdiag minus primary control (mm-equivalent)",
        title="B  Stored 6D yaw: three development seeds",
    )
    ax = axes[1, 0]
    summary = result["synthetic_class_test"]["summary"]
    for i, (method, color) in enumerate(zip(("scalar", "diagonal", "dense"), ("#999999", "#0072B2", "#E69F00"))):
        rows = [r for r in summary if r["method"] == method]
        ax.bar(
            np.arange(2) + (i - 1) * 0.24,
            [r["mean_rmse"] for r in rows],
            width=0.24,
            yerr=[r["sd_rmse"] for r in rows],
            label=method,
            color=color,
            capsize=3,
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Diagonal truth", "Cross-coupled truth"],
        ylabel="Normalized coordinate RMSE",
        title="C  New synthetic test: mean and SD, 20 seeds",
    )
    ax.legend(frameon=False)
    ax = axes[1, 1]
    rows = result["synthetic_origin_test"]
    for key, label in (("world_action_error_mm", "World action residual"), ("pose_error_mm", "Pose residual")):
        ax.plot([r["origin_shift_m"] for r in rows], [r[key] for r in rows], "o-", label=label)
    ax.set(
        xlabel="Passive translation of coordinate origin (m)",
        ylabel="Error (mm-equivalent)",
        title="D  New metric check: identical physical discrepancy",
    )
    ax.legend(frameon=False)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / f"evidence_audit.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-controls", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "audit_results.json").exists():
        raise FileExistsError("Choose a new output directory to preserve existing evidence.")
    result = {
        "schema": "gtsr-paper-audit/1",
        "evidence_role": "exploratory development audit",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                F1,
                F1.with_suffix(".csv"),
                E09,
                Path(__file__),
                ROOT / "src/tsr/gtsr/response.py",
                ROOT / "src/tsr/gtsr/metrics.py",
                ROOT / "src/tsr/gtsr/profile6.py",
                ROOT / "tests/gtsr/_maniskill_data.py",
                ROOT / "tests/gtsr/_maniskill_data_se3.py",
            )
        },
        "stored_evidence": frozen_results(),
        "finite_operator_test": finite_operator_experiment(),
        "synthetic_class_test": hypothesis_class_experiment(),
        "synthetic_origin_test": origin_experiment(),
    }
    if args.raw_controls:
        result["raw_yaw_controls"] = raw_yaw_controls()
    (args.out / "audit_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    plot_results(result, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "finite_operator_test": result["finite_operator_test"],
                "raw_control_rows": len(result.get("raw_yaw_controls", {}).get("rows", [])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
