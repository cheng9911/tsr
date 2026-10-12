"""Unified SE(3) development comparison, with preserved native and anchored predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import scipy
import sklearn
from scipy.spatial.transform import Rotation
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SISTER = ROOT.parent / "maniskill_spectrum"
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(SISTER / "phase_switch_symmetry")]

from phase_switch_se3_baselines import (  # noqa: E402
    SE3DiagonalOperatorModel,
    SE3FrameWeightedModel,
    SE3FullOperatorModel,
    SE3PhaseScalarGPModel,
    SE3TPGMMModel,
)

from tests.gtsr._fit_se3_cells import _load_artifact  # noqa: E402
from tests.gtsr._maniskill_data import multiseed_root  # noqa: E402
from tests.gtsr._maniskill_data_se3 import (  # noqa: E402
    ARMS,
    CONTEXT_SCALE,
    SE3_CONTEXT_MANIFEST_SHA256,
    SE3_EXPERIMENT_SHA256,
    SE3_SEEDS,
    SE3_SUBSET_MANIFEST_SHA256,
    load_se3_isolated,
    load_se3_rollout,
    load_se3_subsets,
    n30_subsets,
    nominal_frame_se3,
    progress_grid_se3,
    se3_rollouts_root,
)
from tsr.gtsr.profile6 import (  # noqa: E402
    ResponseProfile6D,
    se3_exp_batched,
    se3_from_metric,
    se3_from_pose6,
    se3_from_pose6_batched,
    se3_to_metric,
)

SCALE = np.array([1, 1, 1, 0.03, 0.03, 0.03])
METHODS = (
    "fixed",
    "frame",
    "scalar_finite_gp",
    "frame_affine",
    "scalar_affine_gp",
    "full_affine",
    "diagonal_affine",
    "context_rbf",
    "tpgmm_additive",
    "tpgmm_se3",
    "pdiag",
)
METRICS = (
    "local_action_mm",
    "world_action_mm",
    "anchored_pose_mm",
    "native_pose_mm",
    "anchored_task_proxy_mm",
    "native_task_proxy_mm",
    "anchored_position_mm",
    "anchored_rotation_deg",
    "native_position_mm",
    "native_rotation_deg",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inverse(t):
    out = np.zeros_like(t)
    out[..., :3, :3] = t[..., :3, :3].swapaxes(-1, -2)
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", out[..., :3, :3], t[..., :3, 3])
    out[..., 3, 3] = 1
    return out


def log_coordinates(t):
    """Principal SE(3) log, robust rotation vector and analytic inverse left Jacobian."""
    shape = t.shape[:-2]
    w = Rotation.from_matrix(t[..., :3, :3].reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3)
    theta = np.linalg.norm(w, axis=-1)
    small = theta < 1e-4
    safe = np.where(small, 1.0, theta)
    coefficient = np.where(small, 1 / 12 + theta**2 / 720, (1 - 0.5 * safe / np.tan(0.5 * safe)) / safe**2)
    translation = t[..., :3, 3]
    rho = translation - 0.5 * np.cross(w, translation) + coefficient[..., None] * np.cross(w, np.cross(w, translation))
    return np.concatenate([rho, w], axis=-1)


def residual(a, b):
    return np.linalg.norm(log_coordinates(inverse(a) @ b) * SCALE, axis=-1) * 1000


def pose_components(pred, obs):
    position = np.linalg.norm(pred[..., :3, 3] - obs[..., :3, 3], axis=-1)
    relative = pred[..., :3, :3].swapaxes(-1, -2) @ obs[..., :3, :3]
    angle = np.linalg.norm(Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec(), axis=-1).reshape(
        position.shape
    )
    return position, angle


def task_proxy(pred, obs, arm, phases):
    """Declared axial-symmetry proxy, not collision feasibility or physical success."""
    position, angle = pose_components(pred, obs)
    a, b = pred[..., :3, 2], obs[..., :3, 2]
    axis_angle = np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))
    relevant = np.asarray(phases) < 2 if arm == "keyed" else np.zeros(len(phases), dtype=bool)
    orientation = np.where(relevant, angle, axis_angle)
    return np.sqrt(position**2 + (0.03 * orientation) ** 2) * 1000


def evaluate_predictions(native, actions, baseline, observed, reference, arm, phases):
    anchored = actions @ baseline
    observed_actions = observed @ inverse(baseline)
    local = inverse(reference) @ actions @ reference
    observed_local = inverse(reference) @ observed_actions @ reference
    pos_a, rot_a = pose_components(anchored, observed)
    pos_n, rot_n = pose_components(native, observed)
    values = np.stack(
        [
            residual(local, observed_local),
            residual(actions, observed_actions),
            residual(anchored, observed),
            residual(native, observed),
            task_proxy(anchored, observed, arm, phases),
            task_proxy(native, observed, arm, phases),
            pos_a * 1000,
            np.rad2deg(rot_a),
            pos_n * 1000,
            np.rad2deg(rot_n),
        ],
        axis=-1,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite evaluation")
    return values


class ContextRBF6:
    """Six-axis extension of E09's phase-local RBF, episode-grouped training-only CV."""

    def design(self, contexts):
        repeated = np.tile(self.basis, (len(contexts), 1))
        normalized = np.repeat(contexts / CONTEXT_SCALE, len(self.basis), axis=0)
        return np.column_stack([repeated, *[repeated * normalized[:, j, None] for j in range(6)]])

    def fit(self, contexts, curves, progress, phases):
        local = progress * 4 - phases
        values = np.exp(-0.5 * ((local[:, None] - np.linspace(0, 1, 10)) / 0.16) ** 2)
        values /= values.sum(axis=1, keepdims=True)
        self.basis = np.zeros((len(progress), 40))
        for q in range(4):
            self.basis[phases == q, q * 10 : (q + 1) * 10] = values[phases == q]
        design = self.design(contexts)
        groups = np.repeat(np.arange(len(contexts)), len(progress))
        cv = list(GroupKFold(n_splits=5).split(design, groups=groups))
        self.model = RidgeCV(alphas=np.logspace(-10, 0, 16), fit_intercept=False, cv=cv)
        self.model.fit(design, se3_to_metric(curves).reshape(-1, 6))
        return self

    def predict(self, contexts):
        return se3_from_metric(self.model.predict(self.design(contexts)).reshape(len(contexts), len(self.basis), 6))


def prepare(arm, seed, subset):
    path = se3_rollouts_root() / f"{arm}_seed_{seed}.h5"
    ds, iso = load_se3_rollout(path, 25), load_se3_isolated(path, 25)
    ids = subset["source_condition_ids"]
    train = [ds["mixed_cids"].index(cid) for cid in ids]
    test = [i for i, cid in enumerate(ds["mixed_cids"]) if cid not in ids]
    contexts, curves = ds["mixed_contexts"][train], ds["mixed_curves"][train]
    with h5py.File(path, "r") as handle:
        nominal = nominal_frame_se3(handle, [ds["mixed_keys"][i] for i in train], contexts)
    eval_contexts = np.concatenate([iso["isolated_contexts"], ds["mixed_contexts"][test]])
    eval_curves = np.concatenate([iso["isolated_curves"], ds["mixed_curves"][test]])
    eval_ids = iso["isolated_condition_ids"] + [ds["mixed_cids"][i] for i in test]
    assert len(ids) == 30 and len(eval_ids) == 44 and not set(ids).intersection(eval_ids)
    coords_train = log_coordinates(se3_from_pose6_batched(contexts))
    coords_eval = log_coordinates(se3_from_pose6_batched(eval_contexts))
    in_support = np.all(
        (coords_eval >= coords_train.min(0) - 1e-10) & (coords_eval <= coords_train.max(0) + 1e-10), axis=1
    )
    return {
        "contexts": contexts,
        "curves": curves,
        "eval_contexts": eval_contexts,
        "observed": se3_from_pose6_batched(eval_curves),
        "baseline": se3_from_pose6_batched(iso["baseline_curve"]),
        "nominal": nominal,
        "reference": se3_from_pose6(nominal),
        "eval_ids": eval_ids,
        "generators": iso["isolated_generators"] + ["mixed"] * 30,
        "in_support": in_support,
        "train_ids": ids,
    }


def finite_actions(reference, contexts, alpha):
    twists = log_coordinates(se3_from_pose6_batched(contexts))
    local = se3_exp_batched(twists[:, None, :] * alpha[None, :, :])
    return reference @ local @ inverse(reference)


def predictions(method, data, progress, phases, artifact_dir, arm, seed, subset_id):
    meta = {}
    reference, baseline, contexts = data["reference"], data["baseline"], data["eval_contexts"]
    if method in ("fixed", "frame", "pdiag"):
        if method == "pdiag":
            path = artifact_dir / f"{arm}_{seed}_{subset_id}.npz"
            with np.load(path, allow_pickle=False) as loaded:
                art = _load_artifact(loaded)
            assert (art["arm"], art["seed"], art["subset_id"]) == (arm, seed, str(subset_id))
            assert art["manifest_sha256"] == SE3_CONTEXT_MANIFEST_SHA256
            assert art["source_condition_ids"] == data["train_ids"]
            np.testing.assert_allclose(art["nominal_frame_pose"], data["nominal"], atol=1e-12, rtol=0)
            if not art["optimization_success"]:
                raise ValueError("Pdiag optimizer did not converge")
            profile = ResponseProfile6D(art["theta"], alpha_max=art["alpha_max"], basis_width=art["basis_width"])
            alpha = profile.alpha_from_g(progress)
            nominal = se3_from_pose6_batched(art["nominal_curve"])
            meta.update(artifact_sha256=art["artifact_sha256"], production_file_sha256=sha(path))
        else:
            alpha = np.full((len(progress), 6), int(method == "frame"), dtype=float)
            nominal = baseline
        actions = finite_actions(reference, contexts, alpha)
        return actions @ nominal, actions, meta
    models = {
        "frame_affine": lambda: SE3FrameWeightedModel(),
        "scalar_affine_gp": lambda: SE3PhaseScalarGPModel(),
        "scalar_finite_gp": lambda: SE3PhaseScalarGPModel(),
        "full_affine": lambda: SE3FullOperatorModel(),
        "diagonal_affine": lambda: SE3DiagonalOperatorModel(),
        "context_rbf": lambda: ContextRBF6(),
        "tpgmm_additive": lambda: SE3TPGMMModel(frame_mode="additive"),
        "tpgmm_se3": lambda: SE3TPGMMModel(frame_mode="se3", nominal_frame_pose=data["nominal"]),
    }
    model = models[method]().fit(data["contexts"], data["curves"], progress, phases)
    if method.startswith("tpgmm"):
        meta.update(
            n_components=model.n_components,
            em_converged=bool(model.model.converged_),
            em_iterations=int(model.model.n_iter_),
            cv_scores=model.cv_log_likelihood_by_components,
        )
        if not model.model.converged_:
            raise ValueError(f"TP-GMM final EM did not converge in {model.model.n_iter_} iterations")
    if method == "context_rbf":
        meta["ridge_alpha"] = float(model.model.alpha_)
    if method == "scalar_finite_gp":
        actions = finite_actions(reference, contexts, np.repeat(model.weight[:, None], 6, axis=1))
        learned_nominal = se3_from_pose6_batched(model.predict(np.zeros((1, 6)))[0])
        meta["native_nominal"] = "training-only scalar affine zero-context intercept"
        return actions @ learned_nominal, actions, meta
    native_all = se3_from_pose6_batched(model.predict(np.vstack([np.zeros((1, 6)), contexts])))
    native = native_all[1:]
    actions = native @ inverse(native_all[0])
    return native, actions, meta


def summarize(values, generators, phases, in_support):
    groups = {"isolated": np.arange(14), "mixed": np.arange(14, 44)}
    for g in ("du", "dv", "dw", "roll", "pitch", "yaw"):
        groups[g] = np.flatnonzero(np.array(generators) == g)
    groups["in_support"] = np.flatnonzero(in_support)
    groups["out_of_support"] = np.flatnonzero(~in_support)
    out = {}
    for name, indices in groups.items():
        if len(indices):
            out[name] = {
                "n_conditions": len(indices),
                "mean": dict(zip(METRICS, values[indices].mean((0, 1)).tolist())),
                "phase_means": {
                    str(q): dict(zip(METRICS, values[indices][:, phases == q].mean((0, 1)).tolist())) for q in range(4)
                },
            }
    out["isolated_equal_generators"] = {
        "n_generators": 6,
        "mean": {
            metric: float(np.mean([out[g]["mean"][metric] for g in ("du", "dv", "dw", "roll", "pitch", "yaw")]))
            for metric in METRICS
        },
    }
    return out


def run_job(job):
    arm, seed, subset, method, output, artifact_dir, run_hash = job
    folder = Path(output) / "cells"
    name = f"{arm}_{seed}_{subset['subset_id']}_{method}"
    target = folder / f"{name}.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["run_hash"] != run_hash:
            raise ValueError(f"source/configuration drift for cached cell {name}")
        if saved["status"] != "ok" or (folder / f"{name}.npz").exists():
            return saved
        raise ValueError(f"missing prediction artifact {name}")
    started = time.monotonic()
    record = {"arm": arm, "seed": seed, "subset_id": subset["subset_id"], "method": method, "run_hash": run_hash}
    try:
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            data = prepare(arm, seed, subset)
            progress, phases = progress_grid_se3(25)
            native, actions, metadata = predictions(
                method, data, progress, phases, Path(artifact_dir), arm, seed, subset["subset_id"]
            )
            values = evaluate_predictions(
                native, actions, data["baseline"], data["observed"], data["reference"], arm, phases
            )
            # Passive origin shift + rotation on all predicted/observed quantities.
            g = np.eye(4)
            g[:3, :3] = Rotation.from_rotvec([0.2, -0.3, 0.1]).as_matrix()
            g[:3, 3] = [1, -0.4, 0.7]
            shifted = evaluate_predictions(
                g @ native,
                g @ actions @ inverse(g),
                g @ data["baseline"],
                g @ data["observed"],
                g @ data["reference"],
                arm,
                phases,
            )
            invariant = [i for i, name in enumerate(METRICS) if name != "world_action_mm"]
            difference = float(np.max(np.abs(shifted[..., invariant] - values[..., invariant])))
            assert difference < 1e-6, difference
            record.update(
                status="ok",
                metadata=metadata,
                warnings=sorted({str(w.message) for w in captured}),
                invariance_max_abs=difference,
                train_ids=data["train_ids"],
                eval_ids=data["eval_ids"],
                summary=summarize(values, data["generators"], phases, data["in_support"]),
            )
            np.savez_compressed(
                folder / f"{name}.npz",
                native=native,
                actions=actions,
                metrics=values,
                eval_contexts=data["eval_contexts"],
                eval_ids=data["eval_ids"],
                train_ids=data["train_ids"],
                reference=data["reference"],
                baseline=data["baseline"],
                observed=data["observed"],
                generators=data["generators"],
                phases=phases,
                in_support=data["in_support"],
                metric_names=METRICS,
            )
    except Exception as exc:
        record.update(status="failed", exception=repr(exc), traceback=traceback.format_exc())
    record["seconds"] = time.monotonic() - started
    target.write_text(json.dumps(record, indent=2) + "\n")
    return record


def aggregate(records):
    out = []
    for arm in ARMS:
        for method in METHODS:
            selected = [r for r in records if r["arm"] == arm and r["method"] == method]
            if not selected:
                continue
            rows = [r for r in selected if r["status"] == "ok"]
            summary = {
                "arm": arm,
                "method": method,
                "expected_cells": 18,
                "completed_cells": len(selected),
                "successful_cells": len(rows),
                "failed_cells": len(selected) - len(rows),
                "groups": {},
            }
            if len(rows) == 18:
                for group in (
                    "isolated",
                    "isolated_equal_generators",
                    "mixed",
                    "du",
                    "dv",
                    "dw",
                    "roll",
                    "pitch",
                    "yaw",
                ):
                    per_seed = {
                        str(s): {
                            m: float(np.mean([r["summary"][group]["mean"][m] for r in rows if r["seed"] == s]))
                            for m in METRICS
                        }
                        for s in SE3_SEEDS
                    }
                    summary["groups"][group] = {
                        "per_seed": per_seed,
                        "mean": {m: float(np.mean([v[m] for v in per_seed.values()])) for m in METRICS},
                        "sd": {m: float(np.std([v[m] for v in per_seed.values()], ddof=1)) for m in METRICS},
                    }
            out.append(summary)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cells").mkdir(exist_ok=True)
    sources = [
        Path(__file__),
        ROOT / "src/tsr/gtsr/profile6.py",
        ROOT / "tests/gtsr/_fit_se3_cells.py",
        ROOT / "tests/gtsr/_maniskill_data.py",
        ROOT / "tests/gtsr/_maniskill_data_se3.py",
        SISTER / "phase_switch_symmetry/phase_switch_baselines.py",
        SISTER / "phase_switch_symmetry/phase_switch_se3_baselines.py",
        args.out.parent / "protocol.md",
    ]
    sources += [multiseed_root() / f for f in ("se3_fixed_contexts.json", "se3_experiment.json", "se3_subsets.json")]
    for filename, expected in zip(
        ("se3_fixed_contexts.json", "se3_experiment.json", "se3_subsets.json"),
        (SE3_CONTEXT_MANIFEST_SHA256, SE3_EXPERIMENT_SHA256, SE3_SUBSET_MANIFEST_SHA256),
    ):
        assert sha(multiseed_root() / filename) == expected, f"Frozen manifest changed: {filename}"
    sources += [se3_rollouts_root() / f"{a}_seed_{s}.h5" for a in ARMS for s in SE3_SEEDS]
    artifact_paths = [
        args.artifacts / f"{a}_{s}_{sub['subset_id']}.npz"
        for a in ARMS
        for s in SE3_SEEDS
        for sub in n30_subsets(load_se3_subsets(multiseed_root()))
    ]
    if not all(p.exists() for p in artifact_paths):
        raise FileNotFoundError("All 36 Pdiag artifacts must be ready before starting this comparison")
    sources += artifact_paths
    source_hashes = {str(p): sha(p) for p in sources}
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "h5py": h5py.__version__,
    }
    run_hash = hashlib.sha256(
        json.dumps({"sources": source_hashes, "versions": versions}, sort_keys=True).encode()
    ).hexdigest()
    manifest = {
        "run_hash": run_hash,
        "source_sha256": source_hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "versions": versions,
        "methods": list(METHODS),
        "n_seeds_per_arm": 3,
        "evidence_role": "exploratory development comparison",
        "metric_names": METRICS,
    }
    target = args.out / "manifest.json"
    if target.exists():
        assert json.loads(target.read_text())["run_hash"] == run_hash, "source drift; use a new run directory"
    else:
        target.write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.aggregate_only:
        subsets = n30_subsets(load_se3_subsets(multiseed_root()))
        jobs = [
            (a, s, sub, m, str(args.out), str(args.artifacts), run_hash)
            for a in ARMS
            for s in SE3_SEEDS
            for sub in subsets
            for m in args.methods
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                print(
                    f"{i}/{len(jobs)} {result['arm']} {result['seed']} {result['subset_id']} "
                    f"{result['method']} {result['status']} {result['seconds']:.1f}s",
                    flush=True,
                )
    records = [json.loads(p.read_text()) for p in sorted((args.out / "cells").glob("*.json"))]
    assert all(r["run_hash"] == run_hash for r in records)
    result = {
        "manifest": manifest,
        "total_cells": len(records),
        "status_counts": {s: sum(r["status"] == s for r in records) for s in ("ok", "failed")},
        "complete": len(records) == 36 * len(METHODS),
        "summary": aggregate(records),
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("total_cells", "status_counts", "complete")}))


if __name__ == "__main__":
    main()
