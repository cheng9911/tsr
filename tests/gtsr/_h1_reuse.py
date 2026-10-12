# SPDX-License-Identifier: MIT
"""Read-only H1-B controlled-reuse bridge (experiment layer, no fit / rollout / write).

This module wires the frozen ``maniskill_spectrum`` controlled-pose transfer
artifacts into the H0 explicit ``MatchedRelationTransfer`` gate. It performs only
artifact loading, source-integrity verification, split materialization, gated grid
queries, and finite SE(3) action differentials. It never imports or calls the
sister ``benchmark_geometry_transfer.main()``, any ``.fit()``, the collector, an
environment constructor, or HDF5 write mode.

The sister repository is read-only. Execution integrity is enforced by the frozen
SHA-256 snapshot (the hard gate); the observed git HEAD is recorded only as
provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tsr.gtsr.metrics import se3_log as _metric_se3_log
from tsr.gtsr.profile6 import (
    pose6_from_se3_batched,
    se3_exp,
    se3_exp_batched,
    se3_from_pose6,
    se3_from_pose6_batched,
    se3_inverse,
)
from tsr.gtsr.profile6 import (
    se3_log as _profile_se3_log,
)
from tsr.gtsr.transfer import (
    CoordinateRangeSupport,
    GeneratorSemantics,
    MatchedRelationTransfer,
    MatchResult,
    MatchStatus,
    MonomialCorrespondence,
    PiecewiseLinearPhaseMap,
    TransferQueryResult,
    transport_diagonal,
)

TSR_ROOT = Path(__file__).resolve().parents[2]
SISTER_ROOT = Path("/home/rocos/sia/maniskill_spectrum")
MANIFEST_PATH = TSR_ROOT / "docs/superpowers/specs/2026-09-09-gtsr-sprint-h1-reuse-manifest.json"
AUDIT_PATH = TSR_ROOT / "docs/researchwrite/gtsr-h1-reuse/sources/reuse_audit.json"

SEEDS = (20260818, 20270818, 20280818)
SPLIT_SEED = 20260831
N_MIXED = 60
MIXED_CONDITION_IDS = np.arange(15, 75)
BINS_PER_PHASE = 25
N_PHASES = 4
GRID_POINTS = BINS_PER_PHASE * N_PHASES
TRANSFER_SOURCE_SIZE = 30
N_TARGETS = 5

RELATION_SEMANTIC_IDS = (
    "relation_du",
    "relation_dv",
    "relation_dw",
    "relation_roll",
    "relation_pitch",
    "relation_yaw",
)
PHASE_SEMANTIC_IDS = ("reach", "move", "hold", "retract")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observed_sister_head(root: Path = SISTER_ROOT) -> str | None:
    """Record the observed sister HEAD as provenance; the SHA snapshot is the gate."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# --- Task 1: frozen manifest loader and source integrity -----------------------


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if manifest.get("status") != "TARGETS-COMPARISON-LOCKED_RETROSPECTIVE":
        raise ValueError("manifest status is not TARGETS-COMPARISON-LOCKED_RETROSPECTIVE")
    if not manifest["source_repository"].get("read_only"):
        raise ValueError("manifest source_repository must be read_only")
    validate_targets(manifest)
    return manifest


def load_audit(path: Path = AUDIT_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_targets(manifest: dict) -> list[dict]:
    targets = manifest["targets"]
    if len(targets) != N_TARGETS:
        raise ValueError(f"expected {N_TARGETS} frozen targets, got {len(targets)}")
    pair_keys = [t["pair_key"] for t in targets]
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError("duplicate pair_key in manifest targets")
    pair_pairs = [(t["source_task"], t["target_task"]) for t in targets]
    if len(set(pair_pairs)) != len(pair_pairs):
        raise ValueError("duplicate (source_task, target_task) under distinct pair_key in manifest targets")
    for target in targets:
        for field in ("pair_key", "source_task", "target_task", "family", "target_instance"):
            if not target.get(field):
                raise ValueError(f"target missing required field {field}")
        instance = target["target_instance"]
        for field in ("task_key", "suite_name", "task_id", "oracle_selector", "nominal_pose6", "body_name", "control"):
            if field not in instance:
                raise ValueError(f"target_instance missing required field {field}")
        if len(instance["oracle_selector"]) != 6 or len(instance["nominal_pose6"]) != 6:
            raise ValueError("oracle_selector and nominal_pose6 must have length 6")
    return targets


def verify_dependency_hashes(audit: dict, root: Path = SISTER_ROOT) -> dict[str, str]:
    """Halt on any pinned dependency SHA drift; return the observed map.

    An empty SHA map is rejected rather than treated as a trivially clean gate.
    """
    sha_map = audit["sha256"]
    if not sha_map:
        raise ValueError("audit sha256 map is empty; refusing to pass with zero verified dependencies")
    observed = {}
    for relpath, expected in sha_map.items():
        actual = sha256(root / relpath)
        if actual != expected:
            raise RuntimeError(f"SHA-256 drift for {relpath}: expected {expected}, got {actual}")
        observed[relpath] = actual
    return observed


# --- Task 2: historical pose6 and phase-tagged grid loader ----------------------


def historical_pose6_transform(pose) -> np.ndarray:
    """``Trans(xyz) Rx(roll) Ry(pitch) Rz(yaw)``; NOT the canonical TSR ``Rz Ry Rx``."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must be a finite 6-vector")
    x, y, z, roll, pitch, yaw = pose
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    transform = np.eye(4)
    transform[:3, :3] = rx @ ry @ rz
    transform[:3, 3] = pose[:3]
    return transform


def historical_log_coordinates(pose) -> np.ndarray:
    """``c_log = Log(Trans(xyz) Rx Ry Rz)`` in ``[v, w]`` twist order."""
    return _metric_se3_log(historical_pose6_transform(pose))


def _validate_phase_grid(progress, phase_codes) -> None:
    progress = np.asarray(progress, dtype=np.float64)
    phase_codes = np.asarray(phase_codes)
    if progress.shape != (GRID_POINTS,) or phase_codes.shape != (GRID_POINTS,):
        raise ValueError("progress and phase_codes must have length 100")
    if not np.all(np.isfinite(progress)):
        raise ValueError("progress must be finite")
    if sorted(set(phase_codes.tolist())) != [0, 1, 2, 3]:
        raise ValueError("phase_codes must contain exactly 0..3")
    for q in range(N_PHASES):
        block = phase_codes[q * BINS_PER_PHASE : (q + 1) * BINS_PER_PHASE]
        if not np.all(block == q):
            raise ValueError(f"phase {q} must occupy a contiguous {BINS_PER_PHASE}-point block")
        for k in range(BINS_PER_PHASE):
            expected = (q + k / (BINS_PER_PHASE - 1)) / N_PHASES
            if not np.isclose(progress[q * BINS_PER_PHASE + k], expected, atol=1e-12):
                raise ValueError("progress does not match the frozen (q + rho)/4 convention")


def _validate_source_selection(seed_values, subset_values, expected_seeds, expected_subsets, task: str) -> None:
    """Require the exact 3-seed x 6-subset Cartesian product with no duplicates."""
    observed = sorted(zip(seed_values, subset_values))
    expected = sorted((seed, subset) for seed in expected_seeds for subset in expected_subsets)
    if len(observed) != 18 or observed != expected:
        raise ValueError(f"{task}: N=30 selection must be the exact 3-seed x 6-subset Cartesian product")


def load_source_grids(manifest: dict) -> dict[str, np.ndarray]:
    """Load and mean the frozen N=30 source profiles into read-only (100, 6) grids."""
    profile_path = SISTER_ROOT / manifest["source_law"]["artifact"]
    if sha256(profile_path) != manifest["source_law"]["sha256"]:
        raise RuntimeError("profile NPZ SHA-256 drift")
    expected_seeds = manifest["source_law"]["selection"]["seeds"]
    expected_subsets = manifest["source_law"]["selection"]["subset_ids"]
    with np.load(profile_path, allow_pickle=False) as data:
        required = {"profile", "task_key", "seed", "subset_id", "sample_size", "progress", "phase_codes"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"profile NPZ missing fields {sorted(missing)}")
        profile = data["profile"]
        task_keys = data["task_key"].astype(str)
        seeds = data["seed"]
        subset_ids = data["subset_id"]
        sample_sizes = data["sample_size"]
        if profile.shape != (540, GRID_POINTS, 6) or not np.all(np.isfinite(profile)):
            raise ValueError("profile must be a finite (540, 100, 6) array")
        _validate_phase_grid(data["progress"], data["phase_codes"])
        source_tasks = sorted({t["source_task"] for t in manifest["targets"]})
        grids = {}
        for task in source_tasks:
            mask = (task_keys == task) & (sample_sizes == TRANSFER_SOURCE_SIZE)
            if int(mask.sum()) != 18:
                raise ValueError(f"{task}: expected 18 N=30 profiles, got {int(mask.sum())}")
            _validate_source_selection(
                seeds[mask].tolist(), subset_ids[mask].tolist(), expected_seeds, expected_subsets, task
            )
            grid = profile[mask].mean(axis=0)
            grid.setflags(write=False)
            grids[task] = grid
    return grids


def grid_alpha(grid, q: int, k: int) -> np.ndarray:
    """Read-only length-6 alpha at phase ``q`` and bin ``k`` (no interpolation)."""
    grid = np.asarray(grid, dtype=np.float64)
    if grid.shape != (GRID_POINTS, 6) or not np.all(np.isfinite(grid)):
        raise ValueError("grid must be a finite (100, 6) array")
    return grid[_phase_index(q, "q") * BINS_PER_PHASE + _bin_index(k, "k")].copy()


def _phase_index(q, name: str) -> int:
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, int) or not 0 <= q < N_PHASES:
        raise ValueError(f"{name} must be an integer in 0..{N_PHASES - 1}")
    return int(q)


def _bin_index(k, name: str) -> int:
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, int) or not 0 <= k < BINS_PER_PHASE:
        raise ValueError(f"{name} must be an integer in 0..{BINS_PER_PHASE - 1}")
    return int(k)


# --- Task 3: no-fit split materialization and result index ----------------------


def materialize_run_splits(sample_sizes, pairs, seeds=SEEDS, split_seed: int = SPLIT_SEED) -> list[dict]:
    """Reproduce the frozen per-run RNG split; one fresh generator per run."""
    rng = np.random.default_rng(split_seed)
    cells = []
    for pair_key in pairs:
        for seed in seeds:
            for sample_size in sample_sizes:
                train = np.sort(rng.choice(N_MIXED, size=sample_size, replace=False))
                heldout = np.setdiff1d(np.arange(N_MIXED), train)
                cells.append(
                    {
                        "pair_key": pair_key,
                        "seed": seed,
                        "sample_size": sample_size,
                        "repeat": 0,
                        "train_indices": tuple(train.tolist()),
                        "eval_indices": tuple(heldout.tolist()),
                        "train_condition_ids": tuple(MIXED_CONDITION_IDS[train].tolist()),
                        "eval_condition_ids": tuple(MIXED_CONDITION_IDS[heldout].tolist()),
                        "derived_n_eval": int(N_MIXED - sample_size),
                    }
                )
    return cells


def build_result_index(manifest: dict) -> list[dict]:
    """Read-only index over both runs: frozen splits joined with the historical CSVs.

    Integrity is enforced at read time: the CSV SHA must match the audit snapshot,
    every cell must carry exactly the frozen method set, and source/target metadata
    in the CSV must agree with the manifest allowlist.
    """
    rows = []
    audit = load_audit()
    pair_meta = {t["pair_key"]: (t["source_task"], t["target_task"]) for t in manifest["targets"]}
    pairs = [t["pair_key"] for t in manifest["targets"]]
    for run in manifest["comparison_runs"]:
        run_id = run["id"]
        expected_methods = set(run["methods"])
        cells = materialize_run_splits(run["sample_sizes"], pairs)
        cell_index = {(c["pair_key"], c["seed"], c["sample_size"], c["repeat"]): c for c in cells}
        fits_rel = f"phase_switch_symmetry_multiseed/{run_id}/geometry_transfer_fits.csv"
        if fits_rel not in audit["sha256"]:
            raise ValueError(f"CSV {fits_rel} is not pinned in the audit SHA snapshot")
        fits_path = SISTER_ROOT / fits_rel
        if sha256(fits_path) != audit["sha256"][fits_rel]:
            raise RuntimeError(f"CSV SHA-256 drift for {fits_rel}")
        seen = {}
        with fits_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                pair_key = row["pair_key"]
                source_task, target_task = pair_meta[pair_key]
                if row["source_task"] != source_task or row["target_task"] != target_task:
                    raise ValueError(f"CSV source/target metadata for {pair_key} disagrees with the manifest allowlist")
                key = (pair_key, int(row["seed"]), int(row["sample_size"]), int(row["repeat"]))
                if key not in cell_index:
                    raise ValueError(f"CSV cell {key} has no frozen split in run {run_id}")
                if (key, row["method"]) in seen:
                    raise ValueError(f"duplicate CSV row for cell {key} method {row['method']} in run {run_id}")
                seen[(key, row["method"])] = True
                cell = cell_index[key]
                rows.append(
                    {
                        "run_id": run_id,
                        "pair_key": cell["pair_key"],
                        "source_task": source_task,
                        "target_task": target_task,
                        "seed": cell["seed"],
                        "sample_size": cell["sample_size"],
                        "repeat": cell["repeat"],
                        "method": row["method"],
                        "fit_success": row["fit_success"] == "True",
                        "derived_n_eval": cell["derived_n_eval"],
                        "historical_csv_target_interventions_for_validation": int(
                            row["target_interventions_for_validation"]
                        ),
                        "historical_csv_target_interventions_for_fit": int(row["target_interventions_for_fit"]),
                        "historical_csv_source_sample_size": int(row["source_sample_size"]),
                        "train_condition_ids": cell["train_condition_ids"],
                        "eval_condition_ids": cell["eval_condition_ids"],
                        "e_task": float(row["e_task"]),
                        "e_alpha": float(row["e_alpha"]),
                        "m_transfer_correct": row["m_transfer_correct"] == "True",
                    }
                )
        # structural integrity: every cell, exactly the frozen method set, no duplicates
        cell_keys = {(c["pair_key"], c["seed"], c["sample_size"], c["repeat"]) for c in cells}
        seen_keys = {key for key, _ in seen}
        if seen_keys != cell_keys:
            raise ValueError(f"run {run_id}: CSV cells do not match the frozen split cells")
        for key in cell_keys:
            methods = {method for (cell_key, method) in seen if cell_key == key}
            if methods != expected_methods:
                raise ValueError(
                    f"run {run_id} cell {key}: methods {sorted(methods)} != frozen {sorted(expected_methods)}"
                )
    return rows


# --- Task 4: allowlist and H0 contract factory ----------------------------------


def build_contract(manifest: dict) -> MatchedRelationTransfer:
    """Identity local contract for the frozen allowlist: Bs=Bt=I6, H=I4, Gamma=I6."""
    local = manifest["local_contract"]
    support = manifest["support"]
    correspondence = MonomialCorrespondence(
        tuple(local["gamma_permutation"]),
        tuple(local["gamma_scales"]),
    )
    semantics = GeneratorSemantics(RELATION_SEMANTIC_IDS, RELATION_SEMANTIC_IDS)
    phase_map = PiecewiseLinearPhaseMap(
        PHASE_SEMANTIC_IDS,
        PHASE_SEMANTIC_IDS,
        tuple(range(N_PHASES)),
        ((0.0, 1.0),) * N_PHASES,
        ((0.0, 1.0),) * N_PHASES,
    )
    support_region = CoordinateRangeSupport(tuple(support["lower"]), tuple(support["upper"]), support["tolerance"])
    return MatchedRelationTransfer(
        source_basis=np.eye(6),
        target_basis=np.eye(6),
        frame_transform=np.eye(4),
        correspondence=correspondence,
        semantics=semantics,
        phase_map=phase_map,
        support=support_region,
        coordinate_scales=tuple(support["coordinate_scales"]),
        total_tolerance=local["total_tolerance"],
        axis_tolerances=tuple(local["axis_tolerances"]),
    )


def pair_allowlist(manifest: dict) -> dict[tuple[str, str], dict]:
    return {(t["source_task"], t["target_task"]): t for t in manifest["targets"]}


@dataclass(frozen=True)
class PairBridge:
    """Frozen allowlisted pair: H0 contract, read-only source grid, target nominal pose.

    The only way to obtain a ``PairBridge`` is ``build_pair_bridge``, which checks the
    allowlist; ``query_grid``/``bridge_action`` require one, so a bare ``MatchedRelationTransfer``
    can never move alpha without a verified pair identity.
    """

    pair_key: str
    source_task: str
    target_task: str
    contract: MatchedRelationTransfer
    source_grid: np.ndarray
    target_nominal_pose6: tuple[float, ...]


def _freeze_grid(grid) -> np.ndarray:
    """Return a private read-only copy so caller mutations cannot alias bridge state."""
    result = np.array(grid, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def build_pair_bridge(manifest: dict, grids: dict, source_task: str, target_task: str) -> PairBridge | MatchResult:
    """Return the gated bridge for an allowlisted pair, or a UNMATCHED_FAMILY rejection."""
    entry = pair_allowlist(manifest).get((source_task, target_task))
    if entry is None:
        return MatchResult(
            MatchStatus.UNMATCHED_FAMILY,
            float("nan"),
            (float("nan"),) * 6,
            None,
            "pair is not in the frozen allowlist; matching axis labels do not imply a family match",
        )
    contract = build_contract(manifest)
    return PairBridge(
        entry["pair_key"],
        source_task,
        target_task,
        contract,
        _freeze_grid(grids[source_task]),
        tuple(entry["target_instance"]["nominal_pose6"]),
    )


# --- Task 5: gated grid query and 75-condition ledger ---------------------------


def query_grid(bridge: PairBridge, q: int, k: int, target_pose6) -> TransferQueryResult:
    """Gate a single frozen grid alpha through an allowlist-verified pair bridge.

    The bridge (not a bare contract) is the entry point: a ``MatchedRelationTransfer``
    alone carries no pair identity and cannot be used to move alpha here.
    """
    if not isinstance(bridge, PairBridge):
        raise TypeError("query_grid requires a PairBridge from build_pair_bridge (allowlist gate)")
    contract = bridge.contract
    grid = bridge.source_grid
    q = _phase_index(q, "q")
    k = _bin_index(k, "k")
    target_coordinates = historical_log_coordinates(np.asarray(target_pose6, dtype=np.float64))
    contract_result = contract.evaluate_contract()
    if contract_result.status is not MatchStatus.MATCHED:
        return TransferQueryResult(contract_result.status, None, None, None, None, contract_result)
    source_phase, source_progress = contract.source_phase(q, k / (BINS_PER_PHASE - 1))
    query_result = contract.evaluate_query(target_coordinates)
    if query_result.status is not MatchStatus.MATCHED:
        return TransferQueryResult(query_result.status, None, None, None, query_result.source_coordinates, query_result)
    source_bin = int(round(source_progress * (BINS_PER_PHASE - 1)))
    alpha_target = transport_diagonal(grid_alpha(grid, source_phase, source_bin), contract.correspondence)
    return TransferQueryResult(
        MatchStatus.MATCHED,
        tuple(alpha_target),
        source_phase,
        source_progress,
        query_result.source_coordinates,
        query_result,
    )


def load_conditions(manifest: dict) -> list[dict]:
    """Read the frozen 75-condition fixed-contexts manifest (read-only)."""
    contexts_rel = "phase_switch_symmetry_multiseed/libero_relation_suite/libero_relation_suite_fixed_contexts.json"
    audit = load_audit()
    contexts_path = SISTER_ROOT / contexts_rel
    expected = audit["sha256"][contexts_rel]
    if sha256(contexts_path) != expected:
        raise RuntimeError("fixed-contexts manifest SHA-256 drift")
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    conditions = contexts["conditions"]
    if len(conditions) != 75 or [c["condition_id"] for c in conditions] != list(range(75)):
        raise ValueError("expected 75 conditions with condition_id 0..74")
    return conditions


def build_condition_ledger(manifest: dict, grids: dict) -> dict[tuple[str, int], MatchStatus]:
    """Per (pair_key, condition_id) H0 gate status over the frozen 75 conditions."""
    conditions = load_conditions(manifest)
    ledger = {}
    for target in manifest["targets"]:
        bridge = build_pair_bridge(manifest, grids, target["source_task"], target["target_task"])
        if not isinstance(bridge, PairBridge):
            raise RuntimeError(f"allowlisted pair {target['pair_key']} failed to build a bridge")
        for condition in conditions:
            result = query_grid(bridge, 0, 0, np.asarray(condition["causal_delta"], dtype=np.float64))
            ledger[(target["pair_key"], condition["condition_id"])] = result.status
    return ledger


# --- Task 6: finite action bridge differential ----------------------------------


def transfer_predict(nominal_curve, contexts, profile, nominal_frame_pose) -> np.ndarray:
    """Independent matrix realization of the frozen ``_transfer_predict`` formula."""
    nominal_frame = se3_from_pose6(np.asarray(nominal_frame_pose, dtype=np.float64))
    nominal_frame_inverse = se3_inverse(nominal_frame)
    contexts = np.asarray(contexts, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    twists = np.asarray([_profile_se3_log(se3_from_pose6(c)) for c in contexts])
    scaled = twists[:, None, :] * profile[None, :, :]
    actions = se3_exp_batched(scaled.reshape(-1, 6)).reshape(len(contexts), profile.shape[0], 4, 4)
    local_nominal = nominal_frame_inverse @ se3_from_pose6_batched(np.asarray(nominal_curve, dtype=np.float64))
    pred_local = np.einsum("csij,sjk->csik", actions, local_nominal)
    pred_world = np.einsum("ij,csjk->csik", nominal_frame, pred_local)
    return pose6_from_se3_batched(pred_world)


def bridge_action(
    bridge: PairBridge, q: int, k: int, intervention_pose6, nominal_pose6, nominal_curve
) -> np.ndarray | None:
    """Gated finite action ``C0 Exp(alpha odot c_log) C0^-1 X0``; None on gate failure."""
    result = query_grid(bridge, q, k, intervention_pose6)
    if result.status is not MatchStatus.MATCHED:
        return None
    alpha = np.asarray(result.alpha_target, dtype=np.float64)
    nominal_frame = se3_from_pose6(np.asarray(nominal_pose6, dtype=np.float64))
    nominal_frame_inverse = se3_inverse(nominal_frame)
    twist = _profile_se3_log(se3_from_pose6(np.asarray(intervention_pose6, dtype=np.float64)))
    action = nominal_frame @ se3_exp(alpha * twist) @ nominal_frame_inverse
    x0 = se3_from_pose6(np.asarray(nominal_curve[q * BINS_PER_PHASE + k], dtype=np.float64))
    return action @ x0
