"""H1 reuse contract checks only; no rollout or model fitting."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SOURCE_DIR = Path(__file__).parent
AUDIT_MODULE = runpy.run_path(str(SOURCE_DIR / "audit_existing.py"))
ADAPTER = AUDIT_MODULE["historical_pose6_transform"]
AUDIT = AUDIT_MODULE["audit"]
TSR_ROOT = SOURCE_DIR.parents[3]
MANIFEST_PATH = TSR_ROOT / "docs/superpowers/specs/2026-09-09-gtsr-sprint-h1-reuse-manifest.json"


@pytest.mark.parametrize("angles", [(0.2, -0.15, 0.4), (-0.1, 0.25, -0.5), (0, 0, 0)])
def test_historical_rx_ry_rz(angles):
    r, p, y = angles
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    pose = np.array([0.01, -0.02, 0.03, r, p, y])
    result = ADAPTER(pose)
    np.testing.assert_allclose(result[:3, :3], rx @ ry @ rz, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(result[:3, 3], pose[:3])


def test_canonical_order_is_not_the_historical_import_convention():
    pose = [0, 0, 0, 0.2, -0.15, 0.4]
    canonical = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    assert np.max(abs(ADAPTER(pose)[:3, :3] - canonical)) > 0.05


def test_checked_in_audit_is_semantically_reproducible():
    stored = json.loads((SOURCE_DIR / "reuse_audit.json").read_text())
    assert AUDIT() == stored


def test_manifest_tracks_audited_targets_support_and_cell_counts():
    audit = json.loads((SOURCE_DIR / "reuse_audit.json").read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert [p["pair_key"] for p in audit["pairs"]] == [p["pair_key"] for p in manifest["targets"]]
    expected_support = audit["proposed_source_range"]
    for field in (
        "coordinate_convention",
        "source_condition_ids",
        "lower",
        "upper",
        "coordinate_scales",
        "tolerance",
        "out_of_support_condition_ids",
    ):
        assert manifest["support"][field] == expected_support[field]
    assert manifest["support"]["mixed_in_range"] == 60
    assert manifest["support"]["baseline_in_range"] == 1
    assert manifest["support"]["isolated_in_range"] == 4
    assert len(manifest["support"]["out_of_support_condition_ids"]) == 10
    assert [(r["id"], r["rows"]) for r in manifest["comparison_runs"]] == [
        ("geometry_transfer", 180),
        ("geometry_transfer_tpgmm_n8", 75),
    ]
