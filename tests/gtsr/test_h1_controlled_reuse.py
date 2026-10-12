# SPDX-License-Identifier: MIT
"""H1-B controlled-reuse bridge: read-only artifact -> H0 contract gate.

Integration tests are skip-gated on the ``maniskill_spectrum`` repository being
present; unit tests exercise the frozen manifest, Euler convention, H0 contract
factory, allowlist gate, gated grid queries and the finite action differential
without it. The allowlist is enforced structurally: ``query_grid``/``bridge_action``
only accept an allowlist-verified ``PairBridge``, never a bare contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.gtsr import _h1_reuse
from tests.gtsr._h1_reuse import (
    SISTER_ROOT,
    PairBridge,
    _validate_source_selection,
    bridge_action,
    build_condition_ledger,
    build_contract,
    build_pair_bridge,
    build_result_index,
    grid_alpha,
    historical_log_coordinates,
    historical_pose6_transform,
    load_audit,
    load_manifest,
    load_source_grids,
    materialize_run_splits,
    observed_sister_head,
    query_grid,
    transfer_predict,
    validate_targets,
    verify_dependency_hashes,
)
from tsr.gtsr.profile6 import se3_from_pose6
from tsr.gtsr.transfer import MatchResult, MatchStatus

requires_sister = pytest.mark.skipif(not SISTER_ROOT.exists(), reason="maniskill_spectrum not present")


def representative_nominal_curve(nominal_pose6):
    """Deterministic synthetic nominal curve for the interface differential (not task data)."""
    rng = np.random.default_rng(20260831)
    curve = np.tile(np.asarray(nominal_pose6, dtype=np.float64), (100, 1))
    curve[:, :3] += rng.uniform(-0.002, 0.002, size=(100, 3))
    curve[:, 3:] += rng.uniform(-0.02, 0.02, size=(100, 3))
    return curve


def _bridge(manifest, source_task="drawer_middle_open", target_task="plate_front_push", grid=None):
    """Build an allowlisted bridge over a synthetic (default zeros) source grid."""
    if grid is None:
        grid = np.zeros((100, 6))
    result = build_pair_bridge(manifest, {source_task: grid}, source_task, target_task)
    assert isinstance(result, PairBridge)
    return result


# --- Task 1: manifest loader and source integrity ------------------------------


def test_manifest_schema_and_status():
    manifest = load_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "TARGETS-COMPARISON-LOCKED_RETROSPECTIVE"
    assert manifest["source_repository"]["read_only"] is True
    targets = validate_targets(manifest)
    assert len(targets) == 5
    assert len({t["pair_key"] for t in targets}) == 5


def test_manifest_negative_examples(tmp_path):
    manifest = load_manifest()

    bad_schema = dict(manifest)
    bad_schema["schema_version"] = 2
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(bad_schema))
    with pytest.raises(ValueError):
        load_manifest(path)

    bad_status = dict(manifest)
    bad_status["status"] = "TRANSFER-VALIDATED"
    path = tmp_path / "status.json"
    path.write_text(json.dumps(bad_status))
    with pytest.raises(ValueError):
        load_manifest(path)

    missing_field = dict(manifest)
    del missing_field["targets"][0]["target_instance"]["task_id"]
    with pytest.raises(ValueError):
        validate_targets(missing_field)

    duplicate_pair = dict(manifest)
    duplicate_pair["targets"].append(dict(duplicate_pair["targets"][0]))
    with pytest.raises(ValueError):
        validate_targets(duplicate_pair)


def test_validate_targets_rejects_duplicate_source_target_pair():
    manifest = load_manifest()
    dup = dict(manifest)
    dup["targets"] = [dict(t) for t in manifest["targets"]]
    dup["targets"][1] = dict(dup["targets"][0])
    dup["targets"][1]["pair_key"] = "forged_pair_key"
    with pytest.raises(ValueError, match="source_task"):
        validate_targets(dup)


def test_load_manifest_is_atomic_and_validates_targets(tmp_path):
    manifest = load_manifest()
    bad = dict(manifest)
    bad["targets"] = [dict(t) for t in manifest["targets"]]
    bad["targets"][0]["target_instance"] = dict(bad["targets"][0]["target_instance"])
    del bad["targets"][0]["target_instance"]["task_id"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_verify_dependency_hashes_detects_drift(tmp_path):
    audit = load_audit()
    relpath = next(iter(audit["sha256"]))
    (tmp_path / relpath).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / relpath).write_bytes(b"drift")
    tampered = dict(audit)
    tampered["sha256"] = {relpath: "0" * 64}
    with pytest.raises(RuntimeError):
        verify_dependency_hashes(tampered, root=tmp_path)


def test_verify_dependency_hashes_rejects_empty_map(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        verify_dependency_hashes({"sha256": {}}, root=tmp_path)


# --- Task 2: historical pose6 and phase-tagged grid loader ----------------------


@pytest.mark.parametrize("angles", [(0.2, -0.15, 0.4), (-0.1, 0.25, -0.5), (0.4, 0.3, -0.7)])
def test_historical_euler_is_rx_ry_rz_not_canonical(angles):
    r, p, y = angles
    pose = np.array([0.01, -0.02, 0.03, r, p, y])
    result = historical_pose6_transform(pose)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    np.testing.assert_allclose(result[:3, :3], rx @ ry @ rz, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(result[:3, 3], pose[:3])
    # scipy extrinsic XYZ matches the historical convention; canonical TSR Rz Ry Rx does not.
    np.testing.assert_allclose(result[:3, :3], Rotation.from_euler("XYZ", angles).as_matrix(), atol=1e-14, rtol=0)
    assert np.max(np.abs(result[:3, :3] - rz @ ry @ rx)) > 0.05


def test_source_selection_requires_exact_cartesian_product():
    seeds = [20260818, 20270818, 20280818]
    subsets = [12, 13, 14, 15, 16, 17]
    full = [(seed, subset) for seed in seeds for subset in subsets]
    _validate_source_selection([s for s, _ in full], [sub for _, sub in full], seeds, subsets, "task")  # ok
    # replace the last pair with a duplicate of the first -> missing one, duplicated another
    bad_seeds = [s for s, _ in full[:-1]] + [full[0][0]]
    bad_subsets = [sub for _, sub in full[:-1]] + [full[0][1]]
    with pytest.raises(ValueError):
        _validate_source_selection(bad_seeds, bad_subsets, seeds, subsets, "task")


@requires_sister
def test_source_grid_loader_integrity():
    manifest = load_manifest()
    grids = load_source_grids(manifest)
    assert set(grids) == {"drawer_middle_open", "stove_knob_turn", "bowl_on_stove"}
    for grid in grids.values():
        assert grid.shape == (100, 6)
        assert np.all(np.isfinite(grid))
        assert not grid.flags.writeable
        with pytest.raises(ValueError):
            grid[0, 0] = 1.0


@requires_sister
def test_provenance_42_sha_and_observed_head():
    audit = load_audit()
    observed = verify_dependency_hashes(audit)
    assert len(observed) == 42
    assert observed_sister_head()


# --- Task 3: split materialization and result index ----------------------------


def test_split_materialization_cell_counts_and_cross_run_isolation():
    manifest = load_manifest()
    pairs = [t["pair_key"] for t in manifest["targets"]]
    main = materialize_run_splits([3, 5, 8], pairs)
    tpgmm = materialize_run_splits([8], pairs)
    assert len(main) == 45
    assert len(tpgmm) == 15
    assert len({(c["pair_key"], c["seed"], c["sample_size"]) for c in main}) == 45
    assert all(c["derived_n_eval"] == 60 - c["sample_size"] for c in main + tpgmm)
    assert all(len(c["train_condition_ids"]) == c["sample_size"] for c in main + tpgmm)
    assert all(len(c["eval_condition_ids"]) == c["derived_n_eval"] for c in main + tpgmm)
    # identical seed labels but different RNG consumption -> N=8 splits differ across runs
    main_n8 = {(c["pair_key"], c["seed"]): c["train_indices"] for c in main if c["sample_size"] == 8}
    tpgmm_n8 = {(c["pair_key"], c["seed"]): c["train_indices"] for c in tpgmm}
    assert all(main_n8[key] != tpgmm_n8[key] for key in main_n8)


@requires_sister
def test_result_index_180_and_75_without_fit():
    manifest = load_manifest()
    rows = build_result_index(manifest)
    main = [r for r in rows if r["run_id"] == "geometry_transfer"]
    tpgmm = [r for r in rows if r["run_id"] == "geometry_transfer_tpgmm_n8"]
    assert len(main) == 180
    assert len(tpgmm) == 75
    assert len({(r["pair_key"], r["seed"], r["sample_size"], r["repeat"]) for r in main}) == 45
    assert len({(r["pair_key"], r["seed"], r["sample_size"], r["repeat"]) for r in tpgmm}) == 15
    keys = {(r["run_id"], r["pair_key"], r["seed"], r["sample_size"], r["repeat"], r["method"]) for r in rows}
    assert len(keys) == 255
    assert all(r["fit_success"] for r in rows)
    # transfer rows carry the incorrect CSV metadata alongside the derived eval count and budget
    transfer_rows = [r for r in main if r["method"].startswith("Ours transfer")]
    assert len(transfer_rows) == 45
    for r in transfer_rows:
        assert r["historical_csv_target_interventions_for_validation"] == r["sample_size"]
        assert r["historical_csv_target_interventions_for_fit"] == 0
        assert r["historical_csv_source_sample_size"] == 30
        assert r["derived_n_eval"] == 60 - r["sample_size"]
        assert r["historical_csv_target_interventions_for_validation"] != r["derived_n_eval"]
    # scratch rows consume N interventions for fit and zero source budget
    scratch_rows = [r for r in main if not r["method"].startswith("Ours transfer")]
    assert len(scratch_rows) == 135
    for r in scratch_rows:
        assert r["historical_csv_target_interventions_for_fit"] == r["sample_size"]
        assert r["historical_csv_source_sample_size"] == 0


def _tampered_run(monkeypatch, tmp_path, mutate, run_id="geometry_transfer"):
    """Write a tampered copy of one run's CSV under tmp and re-point the bridge at it."""
    rel = f"phase_switch_symmetry_multiseed/{run_id}/geometry_transfer_fits.csv"
    with (_h1_reuse.SISTER_ROOT / rel).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mutated = mutate([dict(r) for r in rows])
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(mutated)
    audit = {"sha256": {rel: hashlib.sha256(path.read_bytes()).hexdigest()}}
    monkeypatch.setattr(_h1_reuse, "SISTER_ROOT", tmp_path)
    monkeypatch.setattr(_h1_reuse, "load_audit", lambda: audit)
    manifest = load_manifest()
    manifest = {**manifest, "comparison_runs": [r for r in manifest["comparison_runs"] if r["id"] == run_id]}
    return manifest


@requires_sister
def test_result_index_rejects_duplicate_row(monkeypatch, tmp_path):
    manifest = _tampered_run(monkeypatch, tmp_path, lambda rows: rows + [dict(rows[0])])
    with pytest.raises(ValueError, match="duplicate"):
        build_result_index(manifest)


@requires_sister
def test_result_index_rejects_source_metadata_mismatch(monkeypatch, tmp_path):
    def mutate(rows):
        rows[0]["source_task"] = "forged_source"
        return rows

    manifest = _tampered_run(monkeypatch, tmp_path, mutate)
    with pytest.raises(ValueError, match="allowlist"):
        build_result_index(manifest)


@requires_sister
def test_result_index_rejects_missing_method(monkeypatch, tmp_path):
    def mutate(rows):
        for i, row in enumerate(rows):
            if row["method"] == "Pdiag finite target scratch":
                del rows[i]
                break
        return rows

    manifest = _tampered_run(monkeypatch, tmp_path, mutate)
    with pytest.raises(ValueError, match="methods"):
        build_result_index(manifest)


@requires_sister
def test_result_index_rejects_csv_sha_drift(monkeypatch):
    rel = "phase_switch_symmetry_multiseed/geometry_transfer/geometry_transfer_fits.csv"
    monkeypatch.setattr(_h1_reuse, "load_audit", lambda: {"sha256": {rel: "0" * 64}})
    manifest = load_manifest()
    manifest = {
        **manifest,
        "comparison_runs": [r for r in manifest["comparison_runs"] if r["id"] == "geometry_transfer"],
    }
    with pytest.raises(RuntimeError, match="SHA-256 drift"):
        build_result_index(manifest)


# --- Task 4: allowlist and H0 contract factory ---------------------------------


def test_contract_factory_static_match_and_tamper_rejection():
    manifest = load_manifest()
    contract = build_contract(manifest)
    assert contract.evaluate_contract().status is MatchStatus.MATCHED
    tampered_basis = replace(contract, target_basis=2 * np.eye(6))
    assert tampered_basis.evaluate_contract().status is MatchStatus.INCOMPATIBLE_BASIS
    wrong_phase = replace(contract.phase_map, target_semantic_ids=("reach", "move", "hold", "wrong"))
    tampered_phase = replace(contract, phase_map=wrong_phase)
    assert tampered_phase.evaluate_contract().status is MatchStatus.UNMATCHED_FAMILY


def test_allowlist_family_gate_rejects_unlisted_pairs():
    manifest = load_manifest()
    source_tasks = sorted({t["source_task"] for t in manifest["targets"]})
    grids = {task: np.zeros((100, 6)) for task in source_tasks}
    for target in manifest["targets"]:
        bridge = build_pair_bridge(manifest, grids, target["source_task"], target["target_task"])
        assert isinstance(bridge, PairBridge)
        assert bridge.contract.evaluate_contract().status is MatchStatus.MATCHED
    # matching yaw axis labels alone do not pass: self-pair and reversed pair are unlisted
    for source, target in [("stove_knob_turn", "stove_knob_turn"), ("microwave_door_revolute", "stove_knob_turn")]:
        rejected = build_pair_bridge(manifest, grids, source, target)
        assert isinstance(rejected, MatchResult)
        assert rejected.status is MatchStatus.UNMATCHED_FAMILY


# --- Task 5: gated grid query and 75-condition ledger --------------------------


def test_query_grid_rejects_bare_contract():
    """The allowlist is enforced structurally: a bare contract has no pair identity."""
    manifest = load_manifest()
    contract = build_contract(manifest)
    with pytest.raises(TypeError):
        query_grid(contract, 2, 3, np.zeros(6))


def test_pair_bridge_source_grid_is_private_read_only_copy():
    manifest = load_manifest()
    grid = np.zeros((100, 6))
    bridge = _bridge(manifest, grid=grid)
    assert not bridge.source_grid.flags.writeable
    with pytest.raises(ValueError):
        bridge.source_grid[0, 0] = 1.0
    # mutating the caller's grid after construction must not alias the bridge state
    grid[:] = 9.0
    np.testing.assert_array_equal(bridge.source_grid, np.zeros((100, 6)))


@requires_sister
def test_75_condition_ledger_65_matched_10_out_of_support():
    manifest = load_manifest()
    grids = load_source_grids(manifest)
    ledger = build_condition_ledger(manifest, grids)
    assert len(ledger) == 5 * 75
    expected_out = [1, 2, 3, 4, 5, 6, 8, 9, 11, 14]
    for target in manifest["targets"]:
        statuses = [ledger[(target["pair_key"], cid)] for cid in range(75)]
        assert sum(1 for s in statuses if s is MatchStatus.MATCHED) == 65
        assert [
            cid for cid in range(75) if ledger[(target["pair_key"], cid)] is MatchStatus.OUT_OF_SUPPORT
        ] == expected_out


def test_query_grid_identity_transport_and_read_only_grid():
    manifest = load_manifest()
    bridge = _bridge(manifest)
    pose = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = query_grid(bridge, 2, 3, pose)
    assert result.status is MatchStatus.MATCHED
    assert (result.source_phase, result.source_progress) == (2, 3 / 24)
    np.testing.assert_array_equal(result.alpha_target, np.zeros(6))
    with pytest.raises(ValueError):
        query_grid(bridge, 4, 0, pose)
    with pytest.raises(ValueError):
        query_grid(bridge, 0, 25, pose)


# --- Task 6: finite action bridge differential ---------------------------------


@requires_sister
def test_finite_action_bridge_differential():
    manifest = load_manifest()
    grids = load_source_grids(manifest)
    target_for = {}
    for t in manifest["targets"]:
        target_for.setdefault(t["source_task"], t["target_task"])
    cases = [
        ("drawer_middle_open", [0.01, 0.0, 0.0, 0.0, 0.0, 0.0], [0.10, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ("stove_knob_turn", [0.0, 0.0, 0.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        ("bowl_on_stove", [0.005, -0.004, 0.003, 0.1, -0.15, 0.2], [0.07, -0.02, 0.055, 0.0, 0.0, 0.0]),
    ]
    for source_task, intervention, nominal_pose in cases:
        grid = grids[source_task]
        bridge = build_pair_bridge(manifest, grids, source_task, target_for[source_task])
        assert isinstance(bridge, PairBridge)
        curve = representative_nominal_curve(nominal_pose)
        for q, k in [(0, 0), (1, 12), (3, 24)]:
            result = query_grid(bridge, q, k, np.asarray(intervention))
            assert result.status is MatchStatus.MATCHED
            np.testing.assert_allclose(result.alpha_target, grid[q * 25 + k], atol=1e-12, rtol=0)
            prediction = transfer_predict(curve, [np.asarray(intervention)], grid, nominal_pose)
            action = bridge_action(bridge, q, k, np.asarray(intervention), nominal_pose, curve)
            np.testing.assert_allclose(action, se3_from_pose6(prediction[0, q * 25 + k]), atol=1e-10, rtol=0)


@requires_sister
def test_gate_failure_produces_no_action():
    manifest = load_manifest()
    grids = load_source_grids(manifest)
    bridge = build_pair_bridge(manifest, grids, "bowl_on_stove", "bowl_on_plate")
    assert isinstance(bridge, PairBridge)
    nominal_pose = [0.07, -0.02, 0.055, 0.0, 0.0, 0.0]
    curve = representative_nominal_curve(nominal_pose)
    out_of_support = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = query_grid(bridge, 0, 0, out_of_support)
    assert result.status is MatchStatus.OUT_OF_SUPPORT
    assert result.alpha_target is None
    assert bridge_action(bridge, 0, 0, out_of_support, nominal_pose, curve) is None


def test_contract_and_grid_copy_immunity():
    manifest = load_manifest()
    contract = build_contract(manifest)
    assert isinstance(contract.source_basis, tuple)
    with pytest.raises(TypeError):
        contract.source_basis[0][0] = 1.0
    with pytest.raises(FrozenInstanceError):
        contract.total_tolerance = 1.0
    # grid_alpha copies; mutating the caller's grid does not change a prior result
    grid = np.zeros((100, 6))
    alpha = grid_alpha(grid, 2, 3)
    grid[:] = 7.0
    np.testing.assert_array_equal(alpha, np.zeros(6))
    # query_grid snapshots the intervention; mutating the caller's pose does not alias
    pose = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    bridge = _bridge(manifest)
    result = query_grid(bridge, 0, 0, pose)
    pose[:] = 999.0
    assert result.status is MatchStatus.MATCHED


def test_historical_log_coordinates_matches_frozen_support_convention():
    # a pure x translation yields the x coordinate directly (identity rotation)
    coordinates = historical_log_coordinates(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(coordinates, [0.01, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-15, rtol=0)
