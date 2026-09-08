# G-TSR Sprint F — 6D Six-Generator Response-Profile Replication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the frozen 6-generator SE(3) Pdiag-finite response-profile identification into `tsr.gtsr.profile6`, and reproduce the frozen 18-pair selectivity headline (yaw is the unique selective generator among six) against the commit-pinned `maniskill_spectrum` source.

**Architecture:** A new production module `src/tsr/gtsr/profile6.py` vendors the frozen SE(3)/pose6 Lie math (blob `35cf06a9`), exposes `ResponseProfile6D` + `ProfileIdentifier6D` (a faithful 6D port of the planar `ProfileIdentifier`), and `SelectivityMetrics6D` + `SelectivityVerdict6D` (the two-layer identification/selectivity verdict). Test-side `_maniskill_data_se3.py` (6D loader emitting **q-index** phase codes) and `_baseline_models_se3.py` (Frame-weighted/Full/pointwise baselines) drive the F-01–F-06 gates; the F-04 integration test runs 36 Pdiag fits with process-level parallelism, per-process single-thread BLAS, and per-fit resumable caching.

**Tech Stack:** Python ≥3.10, NumPy ≥1.24, SciPy ≥1.10 (already deps). Test-only: `h5py`, `threadpoolctl` (already in `[project.optional-dependencies].test`). No new runtime deps; no pandas/transforms3d.

**Spec:** `docs/superpowers/specs/2026-09-07-gtsr-sprint-f-scope.md` (committed `3fa30ac`). The spec is the binding authority; this plan is its argument.

## Global Constraints

- `requires-python = ">=3.10"`; runtime deps stay `numpy>=1.24.0`, `scipy>=1.10.0`, `pyyaml>=6.0.1` (do not add pandas/transforms3d).
- **Do not modify** `src/tsr/gtsr/profile.py`, `response.py`, `metrics.py`, `region.py`, or any existing `tests/tsr/`/`tests/gtsr/` file.
- Frozen generator basis `["du","dv","dw","d_roll","d_pitch","d_yaw"]`; episode `generator` attrs are `du/dv/dw/roll/pitch/yaw/baseline/mixed` (no `d_` prefix on the attr).
- Metric scale frozen `SE3_METRIC_SCALE = [1.0, 1.0, 1.0, 0.03, 0.03, 0.03]`.
- Pdiag-finite params frozen: `alpha_max=1.25`, `n_basis=24`, `basis_width=0.065`, `smoothness_weight=0.1`, `nominal_iterations=3`, `least_squares(method="trf", max_nfev=500, ftol=xtol=gtol=1e-9)`.
- Euler convention (spec §2.1): local pose6 I/O uses `R = Rx(roll) Ry(pitch) Rz(yaw)`; fitting/action uses SE(3) log/exp. Never use `CanonicalTSRChart` RPY to read sister-repo data.
- **q-index phase codes:** the frozen `progress_grid` returns `phase_codes = np.repeat(np.arange(4), bins)` = `[0,1,2,3]` (NOT solver codes `3,4,5,6`). `pre = q<2`, `post = q≥2`, `unlock_start = 2*bins`. The 6D loader must emit q-indices.
- `bins = 25`, so `n_steps = 4*25 = 100`.
- Frozen SHA-256 pins: source `phase_switch_se3_baselines.py` blob `35cf06a9aeaade16662d6020f3c91e561a6c440c` (content `820ac06d…`); `benchmark_se3_transfer.py` content `6f71d22b…`; context manifest `30d7f46f8ce73fc94704e0cdf433087de2f129156a45c90dd3c1a516ed5f64d5`; experiment `44e219fe7c00a1ad2a7761d61618655420cbbd29f542ee2b897a4e1ec4244e4a`; subset manifest `b14b6d26b52daa7d26fd832666d6c195731f2180726e499dd13edc57773664f0`.
- Seeds `(20260818, 20270818, 20280818)`; arms `("keyed", "circular_honest")`; 6 N=30 subsets are `subset_id ∈ {12,13,14}` (random, repeat 0/1/2) and `{15,16,17}` (qualified, repeat 0/1/2).
- `context_scale = [0.012, 0.012, 0.012, 0.2617993877991494, 0.2617993877991494, 0.5235987755982988]` (rad(15°), rad(15°), rad(30°)).
- Tolerance: `≤5e-3` is the formal frozen-headline gate; `0.05` is a gross-divergence diagnostic only.
- Frozen N=30 Pdiag-finite headline: `S_yaw=0.6495259496149173`, `S_others_max=0.12733367562537806`, `purity=0.728870060187765`, `rank=1.0`, `transition=1.0`.
- Frozen N=30 Frame-weighted headline: `S_yaw=S_others_max=0.23015302455697256`, `purity=0.16666666666666666`, `rank=0.0`, `transition=0.0`.
- Every new file carries the SPDX header `# SPDX-License-Identifier: MIT` + `# Copyright (c) 2025 Siddhartha Srinivasa`.
- ruff `line-length = 120`, select `F,E,W,I`.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `src/tsr/gtsr/profile6.py` (create) | Vendored SE(3)/pose6 Lie math, `ResponseProfile6D`, `ProfileIdentifier6D`, `SelectivityMetrics6D`, `SelectivityVerdict6D`, `socket_frame6`, 6-generator `excitation_mask6`/`cell_statuses6` |
| `tests/gtsr/_maniskill_data_se3.py` (create) | 6D HDF5/JSON loader: `pose_to_pose6`, `quat2mat`, `task_curve_se3`, `nominal_frame_se3`, `progress_grid_se3` (q-indices), manifest/subset/rollout loading, integrity checks |
| `tests/gtsr/_baseline_models_se3.py` (create) | Vendored `SE3FrameWeightedModel`, `SE3FullOperatorModel`, `SE3DiagonalOperatorModel` |
| `tests/gtsr/_fit_se3_cells.py` (create) | Process-parallel + resumable Pdiag-finite cell fitting for F-04/F-05 |
| `tests/gtsr/test_profile6.py` (create) | Unit tests: math roundtrip/convention, F-02 embedding, F-03 planted recovery, selectivity-metric unit tests |
| `tests/gtsr/test_sprint_f_replication.py` (create) | Integration gates F-01, F-04, F-05, F-06 |
| `src/tsr/gtsr/__init__.py` (modify) | Export the four new public names |

---

### Task 1: SE(3)/pose6 math primitives in `profile6.py`

**Files:**
- Create: `src/tsr/gtsr/profile6.py`
- Test: `tests/gtsr/test_profile6.py`

**Interfaces:**
- Produces (module-level functions imported by later tasks): `SE3_DIM`, `SE3_METRIC_SCALE`, `se3_to_metric`, `se3_from_metric`, `wrap_angle`, `_skew`, `so3_exp`, `so3_log`, `se3_exp`, `se3_log`, `se3_exp_batched`, `se3_inverse`, `se3_inverse_batched`, `matrix_from_euler`, `euler_from_matrix`, `matrix_from_euler_batched`, `se3_from_pose6`, `se3_from_pose6_batched`, `pose6_from_se3`, `pose6_from_se3_batched`, `socket_frame6`.

- [ ] **Step 1: Write the failing test**

Create `tests/gtsr/test_profile6.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Unit tests for the 6D SE(3)/pose6 math primitives (Sprint F, Task 1)."""

from __future__ import annotations

import numpy as np

from tsr.gtsr.profile6 import (
    SE3_METRIC_SCALE,
    euler_from_matrix,
    matrix_from_euler,
    matrix_from_euler_batched,
    pose6_from_se3,
    pose6_from_se3_batched,
    se3_exp,
    se3_exp_batched,
    se3_from_pose6,
    se3_from_pose6_batched,
    se3_inverse,
    se3_inverse_batched,
    se3_log,
    socket_frame6,
)


def test_metric_scale_frozen():
    np.testing.assert_array_equal(
        SE3_METRIC_SCALE, np.array([1.0, 1.0, 1.0, 0.03, 0.03, 0.03])
    )


def test_so3_exp_log_roundtrip():
    from tsr.gtsr.profile6 import so3_exp, so3_log

    rng = np.random.default_rng(0)
    for _ in range(10):
        w = rng.uniform(-1.0, 1.0, size=3)
        R = so3_exp(w)
        np.testing.assert_allclose(so3_log(R), w, atol=1e-10)


def test_se3_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(10):
        xi = rng.uniform(-0.5, 0.5, size=6)
        T = se3_exp(xi)
        np.testing.assert_allclose(se3_log(T), xi, atol=1e-9)


def test_se3_exp_batched_matches_scalar():
    rng = np.random.default_rng(2)
    twists = rng.uniform(-0.5, 0.5, size=(5, 6))
    batched = se3_exp_batched(twists)
    for i in range(5):
        np.testing.assert_allclose(batched[i], se3_exp(twists[i]), atol=1e-12)


def test_se3_inverse_batched_matches_scalar():
    rng = np.random.default_rng(3)
    twists = rng.uniform(-0.5, 0.5, size=(4, 6))
    T = se3_exp_batched(twists)
    batched = se3_inverse_batched(T)
    for i in range(4):
        np.testing.assert_allclose(batched[i], se3_inverse(T[i]), atol=1e-12)
        np.testing.assert_allclose(batched[i] @ T[i], np.eye(4), atol=1e-12)


def test_pose6_euler_convention_yaw_only():
    # A pure yaw rotation must be Rx(0) Ry(0) Rz(yaw) = Rz(yaw).
    yaw = 0.3
    R = matrix_from_euler(0.0, 0.0, yaw)
    np.testing.assert_allclose(R, np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0],
         [np.sin(yaw), np.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]]
    ), atol=1e-12)


def test_pose6_euler_roundtrip():
    rng = np.random.default_rng(4)
    for _ in range(10):
        rpy = rng.uniform(-0.5, 0.5, size=3)
        np.testing.assert_allclose(
            euler_from_matrix(matrix_from_euler(*rpy)), rpy, atol=1e-10
        )


def test_pose6_se3_roundtrip():
    rng = np.random.default_rng(5)
    for _ in range(10):
        pose6 = rng.uniform(-0.5, 0.5, size=6)
        np.testing.assert_allclose(
            pose6_from_se3(se3_from_pose6(pose6)), pose6, atol=1e-10
        )


def test_pose6_se3_batched_roundtrip():
    rng = np.random.default_rng(6)
    pose6 = rng.uniform(-0.5, 0.5, size=(7, 6))
    np.testing.assert_allclose(
        pose6_from_se3_batched(se3_from_pose6_batched(pose6)), pose6, atol=1e-10
    )


def test_matrix_from_euler_batched_matches_scalar():
    rng = np.random.default_rng(7)
    rpy = rng.uniform(-0.5, 0.5, size=(6, 3))
    batched = matrix_from_euler_batched(rpy)
    for i in range(6):
        np.testing.assert_allclose(
            batched[i], matrix_from_euler(*rpy[i]), atol=1e-12
        )


def test_socket_frame6():
    nominal = se3_from_pose6(np.array([0.1, 0.2, 0.3, 0.1, -0.1, 0.2]))
    context = np.array([0.01, -0.02, 0.03, 0.05, -0.05, 0.1])
    expected = nominal @ se3_from_pose6(context)
    np.testing.assert_allclose(socket_frame6(context, nominal), expected, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_profile6.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tsr.gtsr.profile6'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/tsr/gtsr/profile6.py` with the vendored frozen math block (transcribed verbatim from `phase_switch_se3_baselines.py` at `e5ce8a3`, blob `35cf06a9`):

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Six-generator SE(3) response-profile identification (Sprint F).

Faithful 6D port of ``maniskill_spectrum``'s ``SE3SmoothFinitePDiagModel``
(commit ``e5ce8a3``, blob ``35cf06a9``) plus the threshold-free selectivity
metrics. The generator diagonal is ``[du, dv, dw, d_roll, d_pitch, d_yaw]``;
rotation channels use the local convention ``R_local = Rx(roll) Ry(pitch) Rz(yaw)``
(column-vector, yaw innermost). This module neither loads ManiSkill / HDF5 data
(see test-side ``tests.gtsr._maniskill_data_se3``) nor touches the planar
``profile.py``.
"""

from __future__ import annotations

import numpy as np

SE3_DIM = 6
# 3 translations in metres + 3 rotations at 0.03 m/rad (same axial scale as SE(2)).
SE3_METRIC_SCALE = np.array([1.0, 1.0, 1.0, 0.03, 0.03, 0.03], dtype=np.float64)


def se3_to_metric(values):
    return np.asarray(values, dtype=np.float64) * SE3_METRIC_SCALE


def se3_from_metric(values):
    return np.asarray(values, dtype=np.float64) / SE3_METRIC_SCALE


def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi


# --- SO(3)/SE(3) Lie-group math -------------------------------------------------


def _skew(w):
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=np.float64,
    )


def so3_exp(w):
    w = np.asarray(w, dtype=np.float64)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + _skew(w)
    W = _skew(w)
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * W
        + ((1.0 - np.cos(theta)) / (theta * theta)) * (W @ W)
    )


def so3_log(R):
    R = np.asarray(R, dtype=np.float64)
    theta = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    if theta < 1e-12:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        )
    return (theta / (2.0 * np.sin(theta))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def se3_exp(xi):
    xi = np.asarray(xi, dtype=np.float64)
    v = xi[:3]
    w = xi[3:]
    theta = float(np.linalg.norm(w))
    R = so3_exp(w)
    if theta < 1e-12:
        V = np.eye(3) + 0.5 * _skew(w)
    else:
        W = _skew(w)
        V = (
            np.eye(3)
            + ((1.0 - np.cos(theta)) / (theta * theta)) * W
            + ((theta - np.sin(theta)) / (theta ** 3)) * (W @ W)
        )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def se3_log(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    w = so3_log(R)
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        Vinv = np.eye(3) - 0.5 * _skew(w)
    else:
        W = _skew(w)
        Vinv = (
            np.eye(3)
            - 0.5 * W
            + (1.0 / (theta * theta))
            * (1.0 - (theta * np.sin(theta)) / (2.0 * (1.0 - np.cos(theta))))
            * (W @ W)
        )
    v = Vinv @ t
    return np.concatenate([v, w])


def se3_exp_batched(twists):
    """Vectorized SE(3) exponential: (..., 6) -> (..., 4, 4)."""
    twists = np.asarray(twists, dtype=np.float64)
    v = twists[..., :3]
    w = twists[..., 3:]
    theta = np.linalg.norm(w, axis=-1)
    theta_safe = np.where(theta < 1e-12, 1.0, theta)
    small = theta < 1e-12

    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    W = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    W[..., 0, 1] = -wz
    W[..., 0, 2] = wy
    W[..., 1, 0] = wz
    W[..., 1, 2] = -wx
    W[..., 2, 0] = -wy
    W[..., 2, 1] = wx
    W2 = np.einsum("...ij,...jk->...ik", W, W)

    a = np.where(small, 1.0, np.sin(theta_safe) / theta_safe)
    b = np.where(small, 0.5, (1.0 - np.cos(theta_safe)) / (theta_safe ** 2))
    c = np.where(small, 1.0 / 6.0, (theta_safe - np.sin(theta_safe)) / (theta_safe ** 3))

    eye = np.zeros(w.shape[:-1] + (3, 3), dtype=np.float64)
    eye[..., 0, 0] = eye[..., 1, 1] = eye[..., 2, 2] = 1.0
    R = eye + a[..., None, None] * W + b[..., None, None] * W2
    V = eye + b[..., None, None] * W + c[..., None, None] * W2
    t = np.einsum("...ij,...j->...i", V, v)

    T = np.zeros(w.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def se3_inverse(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def se3_inverse_batched(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


# --- Local euler convention R = Rx(roll) Ry(pitch) Rz(yaw) ----------------------


def matrix_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rx @ Ry @ Rz


def euler_from_matrix(R):
    R = np.asarray(R, dtype=np.float64)
    pitch = float(np.arcsin(np.clip(R[0, 2], -1.0, 1.0)))
    roll = float(np.arctan2(-R[1, 2], R[2, 2]))
    yaw = float(np.arctan2(-R[0, 1], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def matrix_from_euler_batched(rpy):
    rpy = np.asarray(rpy, dtype=np.float64)
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.zeros(roll.shape + (3, 3), dtype=np.float64)
    Rx[..., 0, 0] = 1.0
    Rx[..., 1, 1] = cr
    Rx[..., 1, 2] = -sr
    Rx[..., 2, 1] = sr
    Rx[..., 2, 2] = cr
    Ry = np.zeros(roll.shape + (3, 3), dtype=np.float64)
    Ry[..., 0, 0] = cp
    Ry[..., 0, 2] = sp
    Ry[..., 1, 1] = 1.0
    Ry[..., 2, 0] = -sp
    Ry[..., 2, 2] = cp
    Rz = np.zeros(roll.shape + (3, 3), dtype=np.float64)
    Rz[..., 0, 0] = cy
    Rz[..., 0, 1] = -sy
    Rz[..., 1, 0] = sy
    Rz[..., 1, 1] = cy
    Rz[..., 2, 2] = 1.0
    return np.einsum("...ij,...jk,...kl->...il", Rx, Ry, Rz)


def se3_from_pose6(pose6):
    pose6 = np.asarray(pose6, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = matrix_from_euler(pose6[3], pose6[4], pose6[5])
    T[:3, 3] = pose6[:3]
    return T


def se3_from_pose6_batched(pose6):
    pose6 = np.asarray(pose6, dtype=np.float64)
    T = np.zeros(pose6.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = matrix_from_euler_batched(pose6[..., 3:])
    T[..., :3, 3] = pose6[..., :3]
    T[..., 3, 3] = 1.0
    return T


def pose6_from_se3_batched(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    roll = np.arctan2(-R[..., 1, 2], R[..., 2, 2])
    pitch = np.arcsin(np.clip(R[..., 0, 2], -1.0, 1.0))
    yaw = np.arctan2(-R[..., 0, 1], R[..., 0, 0])
    out = np.empty(T.shape[:-2] + (6,), dtype=np.float64)
    out[..., :3] = T[..., :3, 3]
    out[..., 3] = roll
    out[..., 4] = pitch
    out[..., 5] = yaw
    return out


def pose6_from_se3(T):
    return pose6_from_se3_batched(np.asarray(T, dtype=np.float64))


def socket_frame6(context, nominal_frame):
    """SE(3) relation pose for a 6-vector intervention: ``C = C0 Exp(c)``.

    ``nominal_frame`` is the 4x4 nominal socket frame; ``context`` is the
    length-6 intervention ``[du, dv, dw, d_roll, d_pitch, d_yaw]``. This is the
    6D analogue of the planar :func:`~tsr.gtsr.profile.socket_frame`, and is the
    relation pose ``ResponseOperator`` consumes (spec §2, F-02).
    """
    return np.asarray(nominal_frame, dtype=np.float64) @ se3_from_pose6(context)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_profile6.py -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/profile6.py tests/gtsr/test_profile6.py
git commit -m "feat(gtsr): vendor frozen SE(3)/pose6 math into profile6"
```

---

### Task 2: `ResponseProfile6D` + `ProfileIdentifier6D`

**Files:**
- Modify: `src/tsr/gtsr/profile6.py` (append the classes below)
- Test: `tests/gtsr/test_profile6.py` (append F-02 + F-03 tests)

**Interfaces:**
- Consumes: Task 1's math functions + `rbf_basis` from `tsr.gtsr.profile`, `se3_log`/`se3_exp` (vendored in Task 1).
- Produces: `ResponseProfile6D(theta, alpha_max=1.25, basis_width=0.065, *, optimization=None, n_phases=4)` with `.alpha(q, rho) -> (6,)`, `.alpha_from_g(g) -> (6,)`, `.theta`, `.optimization`; `ProfileIdentifier6D(alpha_max=1.25, n_basis=24, basis_width=0.065, smoothness_weight=0.1, nominal_iterations=3)` with `.fit(contexts, curves, progress, phase_codes=None, nominal_frame_pose=None) -> ResponseProfile6D`, `.predict(contexts)`, `.jacobian_diag() -> (n_steps, 6)`, `.action(context, alpha) -> 4x4`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/gtsr/test_profile6.py`:

```python
# -- imports added to the module header (Task 2) -------------------------------
import numpy as np

from tsr.gtsr.response import ResponseOperator
from tsr.gtsr.profile6 import (
    ProfileIdentifier6D,
    ResponseProfile6D,
    pose6_from_se3,
    se3_from_pose6,
    socket_frame6,
)


def test_F02_action_matches_response_operator():
    """ProfileIdentifier6D.action == ResponseOperator(reference, I).action(socket_frame6)."""
    rng = np.random.default_rng(20)
    nominal_pose6 = np.array([0.1, -0.2, 0.15, 0.05, -0.04, 0.3])
    nominal_frame = se3_from_pose6(nominal_pose6)
    identifier = ProfileIdentifier6D()
    identifier.nominal_frame = nominal_frame
    identifier.nominal_frame_inverse = np.linalg.inv(nominal_frame)

    operator = ResponseOperator(nominal_frame, np.eye(6))
    contexts = np.array(
        [
            [0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.015, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.261799, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.261799, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.523599],
            [0.012, -0.012, 0.012, 0.2, -0.2, 0.4],
        ]
    )
    alphas = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.5, 0.7, 0.9, 1.1, 1.25],
            [1.25, 0.2, 0.4, 0.6, 0.8, 0.1],
        ]
    )
    for context in contexts:
        relation = socket_frame6(context, nominal_frame)
        for alpha in alphas:
            action_identifier = identifier.action(context, alpha)
            action_operator = operator.action(relation, alpha)
            assert np.linalg.norm(action_identifier - action_operator, ord="fro") < 1e-9


def _planted_recovery_data():
    """Synthetic 60-mixed contexts/curves with a planted 6D profile (spec F-03)."""
    n_basis = 24
    n_steps = 100
    scale = np.array([0.012, 0.012, 0.012, np.deg2rad(15), np.deg2rad(15), np.deg2rad(30)])
    rng = np.random.default_rng(20260908)
    contexts = rng.uniform(-1.0, 1.0, size=(60, 6)) * scale
    assert np.linalg.matrix_rank(contexts / scale) == 6
    assert (np.min(contexts, axis=0) < 0).all() and (np.max(contexts, axis=0) > 0).all()

    progress = np.concatenate([(i + np.linspace(0.0, 1.0, 25)) / 4 for i in range(4)])
    phase_codes = np.repeat(np.arange(4), 25)

    nominal_frame = np.eye(4)
    # Nominal curve: peg descends along -z from 0 to -0.05, fixed orientation.
    nominal_curve = np.zeros((n_steps, 6))
    nominal_curve[:, 2] = -0.05 * (1.0 - progress)

    def alpha_star(g):
        return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.25 * (1.0 - g)])

    operator = ResponseOperator(nominal_frame, np.eye(6))
    curves = np.empty((60, n_steps, 6))
    for c, context in enumerate(contexts):
        relation = socket_frame6(context, nominal_frame)
        for s in range(n_steps):
            action = operator.action(relation, alpha_star(progress[s]))
            curves[c, s] = pose6_from_se3(action @ se3_from_pose6(nominal_curve[s]))
    return contexts, curves, progress, phase_codes, alpha_star, nominal_curve


def test_F03_planted_recovery():
    contexts, curves, progress, phase_codes, alpha_star, _ = _planted_recovery_data()
    identifier = ProfileIdentifier6D()
    profile = identifier.fit(
        contexts, curves, progress, phase_codes, nominal_frame_pose=np.zeros(6)
    )
    assert profile.optimization["success"]
    assert identifier.jacobian_diag().shape == (100, 6)
    assert profile.theta.shape == (6, 24)
    for q in range(4):
        for rho in (0.0, 0.5, 1.0):
            g = (q + rho) / 4
            recovered = identifier.jacobian_diag()[q * 25 + int(round(rho * 24))]
            target = alpha_star(g)
            assert np.max(np.abs(recovered - target)) <= 5e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_profile6.py::test_F02_action_matches_response_operator tests/gtsr/test_profile6.py::test_F03_planted_recovery -v`
Expected: FAIL with `ImportError: cannot import name 'ProfileIdentifier6D'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/tsr/gtsr/profile6.py` (below the `socket_frame6` definition). Add the import `from .profile import rbf_basis` at the top of the module.

```python
class ResponseProfile6D:
    """Serializable, read-only ``q -> alpha`` finite diagonal profile (6 generators).

    ``alpha(q, rho)`` is the length-6 relevance vector ``[a_du, a_dv, a_dw,
    a_roll, a_pitch, a_yaw]`` under the frozen RBF-sigmoid parameterization, with
    ``g = (q + rho) / n_phases`` and ``q`` the fitting index (0..3). Unlike the
    planar profile, the 6D pipeline works in q-indices directly (the loader maps
    solver phase ``s -> q = s - 3``), so there is no solver-code map here.
    """

    __slots__ = ("_theta", "_n_basis", "_n_phases", "_alpha_max", "_basis_width", "_optimization", "_frozen")

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("ResponseProfile6D is read-only")
        object.__setattr__(self, name, value)

    def __init__(self, theta, alpha_max=1.25, basis_width=0.065, *, optimization=None, n_phases=4):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 2 or theta.shape[0] != SE3_DIM:
            raise ValueError(f"theta must have shape (6, n_basis), got {theta.shape}")
        self._theta = tuple(tuple(float(value) for value in row) for row in theta)
        self._n_basis = theta.shape[1]
        self._n_phases = int(n_phases)
        self._alpha_max = float(alpha_max)
        self._basis_width = float(basis_width)
        self._optimization = dict(optimization) if optimization else {}
        self._frozen = True

    @property
    def theta(self):
        return np.asarray(self._theta, dtype=np.float64).copy()

    @property
    def n_basis(self):
        return self._n_basis

    @property
    def alpha_max(self):
        return self._alpha_max

    @property
    def basis_width(self):
        return self._basis_width

    @property
    def optimization(self):
        return dict(self._optimization)

    def alpha_from_g(self, g):
        g = np.asarray(g, dtype=np.float64)
        basis = rbf_basis(np.atleast_1d(g), self._n_basis, self._basis_width)
        logits = basis @ np.asarray(self._theta, dtype=np.float64).T
        return self._alpha_max / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def alpha(self, q, rho):
        """Length-6 relevance at fitting phase ``q`` and within-phase progress ``rho``."""
        g = (int(q) + float(rho)) / self._n_phases
        return np.asarray(self.alpha_from_g(g)).reshape(-1)

    def to_dict(self):
        return {
            "alpha_max": self._alpha_max,
            "basis_width": self._basis_width,
            "n_basis": self._n_basis,
            "n_phases": self._n_phases,
            "theta": np.asarray(self._theta, dtype=np.float64).tolist(),
            "optimization": dict(self._optimization),
        }

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        return cls(
            np.asarray(data["theta"], dtype=np.float64),
            alpha_max=float(data["alpha_max"]),
            basis_width=float(data["basis_width"]),
            optimization=data.get("optimization"),
            n_phases=int(data.get("n_phases", 4)),
        )


class ProfileIdentifier6D:
    """Frozen six-generator Pdiag-finite fitter: ``D_E -> theta`` (spec §3).

    Faithful 6D port of ``SE3SmoothFinitePDiagModel``: ``alpha_max``-constrained
    RBF-sigmoid generator profiles with second-difference smoothness, realized by
    ``C0 Exp(P(s) Log(C0^-1 C)) C0^-1 X0`` in SE(3), fitted with
    ``scipy.optimize.least_squares(method="trf")`` and three alternating
    nominal/profile updates. The nominal frame pose is supplied at fit time (like
    the planar ``ProfileIdentifier``); ``phase_codes`` (q-indices) is accepted but
    unused by the fit, mirroring the frozen model.
    """

    def __init__(self, alpha_max=1.25, n_basis=24, basis_width=0.065, smoothness_weight=0.1, nominal_iterations=3):
        self.alpha_max = float(alpha_max)
        self.n_basis = int(n_basis)
        self.basis_width = float(basis_width)
        self.smoothness_weight = float(smoothness_weight)
        self.nominal_iterations = int(nominal_iterations)

    def _basis(self, progress):
        return rbf_basis(progress, self.n_basis, self.basis_width)

    def _alpha(self, parameters, basis):
        logits = basis @ parameters.reshape(SE3_DIM, self.n_basis).T
        return self.alpha_max / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def _context_twist(self, context):
        intervention = se3_from_pose6(context)
        socket_world = self.nominal_frame @ intervention
        relative = self.nominal_frame_inverse @ socket_world
        return se3_log(relative)

    def _twists(self, contexts):
        return np.asarray([self._context_twist(context) for context in contexts])

    def _local_actions(self, contexts, alpha):
        twists = self._twists(contexts)
        scaled = twists[:, None, :] * alpha[None, :, :]
        return se3_exp_batched(scaled.reshape(-1, 6)).reshape(len(contexts), alpha.shape[0], 4, 4)

    def _predict_with(self, contexts, nominal_curve, alpha):
        contexts = np.asarray(contexts, dtype=np.float64)
        actions = self._local_actions(contexts, alpha)
        local_nominal = self.nominal_frame_inverse @ se3_from_pose6_batched(
            np.asarray(nominal_curve, dtype=np.float64)
        )
        pred_local = np.einsum("csij,sjk->csik", actions, local_nominal)
        pred_world = np.einsum("ij,csjk->csik", self.nominal_frame, pred_local)
        return pose6_from_se3_batched(pred_world)

    def _update_nominal(self, contexts, curves, alpha):
        contexts = np.asarray(contexts, dtype=np.float64)
        curves = np.asarray(curves, dtype=np.float64)
        actions = self._local_actions(contexts, alpha)
        curves_T = se3_from_pose6_batched(curves.reshape(-1, 6)).reshape(
            len(contexts), curves.shape[1], 4, 4
        )
        obs_local = np.einsum("ij,csjk->csik", self.nominal_frame_inverse, curves_T)
        actions_inv = se3_inverse_batched(actions)
        implied_local = np.einsum("csij,csjk->csik", actions_inv, obs_local)
        implied_world = np.einsum("ij,csjk->csik", self.nominal_frame, implied_local)
        implied = pose6_from_se3_batched(implied_world)
        nominal = np.empty((curves.shape[1], SE3_DIM), dtype=np.float64)
        nominal[:, :3] = implied[:, :, :3].mean(axis=0)
        for generator in range(3):
            angle = implied[:, :, 3 + generator]
            nominal[:, 3 + generator] = np.arctan2(
                np.sin(angle).mean(axis=0), np.cos(angle).mean(axis=0)
            )
        return nominal

    def _initial_parameters(self, pointwise_diagonal, basis):
        clipped = np.clip(np.asarray(pointwise_diagonal), 1e-4, self.alpha_max - 1e-4)
        logits = np.log(clipped / (self.alpha_max - clipped))
        regularizer = 1e-6 * np.eye(self.n_basis)
        inverse = np.linalg.solve(basis.T @ basis + regularizer, basis.T)
        parameters = [inverse @ logits[:, generator] for generator in range(SE3_DIM)]
        return np.asarray(parameters).reshape(-1)

    def _pointwise_diagonal(self, contexts, curves):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        n_steps = curves.shape[1]
        intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        diagonal = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        for step in range(n_steps):
            for generator in range(SE3_DIM):
                design = np.column_stack([np.ones(len(contexts_metric)), contexts_metric[:, generator]])
                coefficients = np.linalg.lstsq(design, curves_metric[:, step, generator], rcond=None)[0]
                intercept_metric[step, generator] = coefficients[0]
                diagonal[step, generator] = coefficients[1]
        return intercept_metric, diagonal

    def fit(self, contexts, curves, progress, phase_codes=None, nominal_frame_pose=None):
        contexts = np.asarray(contexts, dtype=np.float64)
        curves = np.asarray(curves, dtype=np.float64)
        progress = np.asarray(progress, dtype=np.float64)
        if contexts.ndim != 2 or contexts.shape[1] != 6:
            raise ValueError(f"contexts must have shape (n, 6), got {contexts.shape}")
        if curves.ndim != 3 or curves.shape[2] != 6 or curves.shape[0] != contexts.shape[0]:
            raise ValueError(f"curves must have shape (n, steps, 6), got {curves.shape}")
        if progress.ndim != 1 or progress.shape[0] != curves.shape[1]:
            raise ValueError(f"progress must have shape (steps,), got {progress.shape}")
        if not np.all(np.isfinite(contexts)) or not np.all(np.isfinite(curves)) or not np.all(np.isfinite(progress)):
            raise ValueError("contexts, curves and progress must be finite")
        if np.linalg.matrix_rank(se3_to_metric(contexts)) < 6:
            raise ValueError("intervention contexts are rank-deficient; all six generators must be excited")
        if not np.all(excitation_mask6(contexts)):
            raise ValueError("one or more generators lack signed (+/-) intervention coverage (UNEXCITED)")
        if nominal_frame_pose is None:
            raise ValueError("nominal_frame_pose is required")
        self.nominal_frame_pose = np.asarray(nominal_frame_pose, dtype=np.float64)
        if self.nominal_frame_pose.shape != (6,):
            raise ValueError("nominal_frame_pose must have shape (6,)")
        self.nominal_frame = se3_from_pose6(self.nominal_frame_pose)
        self.nominal_frame_inverse = se3_inverse(self.nominal_frame)
        if phase_codes is not None:
            phase_codes = np.asarray(phase_codes)
            if phase_codes.shape != progress.shape:
                raise ValueError("phase_codes must have shape (steps,)")
            if not np.issubdtype(phase_codes.dtype, np.integer):
                raise ValueError("phase_codes must contain integer q indices")
            if not set(phase_codes.tolist()).issubset({0, 1, 2, 3}):
                raise ValueError("phase_codes must be q indices in {0, 1, 2, 3}")
            self.phase_codes = phase_codes.astype(np.int64)

        basis = self._basis(progress)
        intercept_metric, pointwise_diagonal = self._pointwise_diagonal(contexts, curves)
        nominal_curve = se3_from_metric(intercept_metric)
        parameters = self._initial_parameters(pointwise_diagonal, basis)

        optimization = None
        for iteration in range(self.nominal_iterations):

            def residual(candidate, contexts=contexts, curves=curves, nominal_curve=nominal_curve):
                alpha = self._alpha(candidate, basis)
                prediction = self._predict_with(contexts, nominal_curve, alpha)
                data_residual = prediction - curves
                data_residual[..., 3:] = wrap_angle(data_residual[..., 3:])
                data_residual = se3_to_metric(data_residual).reshape(-1)
                smooth = np.diff(alpha, n=2, axis=0)
                scale = np.sqrt(self.smoothness_weight * data_residual.size / max(smooth.size, 1)) * np.mean(
                    np.abs(se3_to_metric(contexts)), axis=0
                )
                smooth_residual = (smooth * scale[None, :]).reshape(-1)
                return np.concatenate([data_residual, smooth_residual])

            optimization = least_squares(
                residual,
                parameters,
                method="trf",
                max_nfev=500,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-9,
            )
            parameters = optimization.x
            alpha = self._alpha(parameters, basis)
            if iteration + 1 < self.nominal_iterations:
                nominal_curve = self._update_nominal(contexts, curves, alpha)

        self.parameters = parameters.reshape(SE3_DIM, self.n_basis)
        self.diagonal = self._alpha(parameters, basis)
        self.nominal_curve = nominal_curve
        self.optimization_success = bool(optimization.success)
        self.optimization_cost = float(optimization.cost)
        self.optimization_nfev = int(optimization.nfev)
        self.optimization_optimality = float(optimization.optimality)
        return ResponseProfile6D(
            self.parameters,
            alpha_max=self.alpha_max,
            basis_width=self.basis_width,
            optimization={
                "success": self.optimization_success,
                "cost": self.optimization_cost,
                "nfev": self.optimization_nfev,
                "optimality": self.optimization_optimality,
            },
        )

    def predict(self, contexts):
        return self._predict_with(contexts, self.nominal_curve, self.diagonal)

    def jacobian_diag(self):
        return self.diagonal.copy()

    def action(self, context, alpha):
        """Return the frozen SE(3) left action ``C0 Exp(alpha . c) C0^-1``."""
        twist = self._context_twist(context)
        return self.nominal_frame @ se3_exp(np.asarray(alpha, dtype=np.float64) * twist) @ self.nominal_frame_inverse


def excitation_mask6(contexts, *, tol=1e-12):
    """Per-generator excitation booleans for 6-vector contexts (spec §7)."""
    contexts = np.asarray(contexts, dtype=np.float64)
    if contexts.ndim != 2:
        raise ValueError(f"contexts must have shape (n, 6), got {contexts.shape}")
    return (np.min(contexts, axis=0) < -tol) & (np.max(contexts, axis=0) > tol)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gtsr/test_profile6.py -v`
Expected: PASS (14 tests; F-02 and F-03 pass).

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/profile6.py tests/gtsr/test_profile6.py
git commit -m "feat(gtsr): add ResponseProfile6D and ProfileIdentifier6D"
```

---

### Task 3: 6D data loader `_maniskill_data_se3.py`

**Files:**
- Create: `tests/gtsr/_maniskill_data_se3.py`
- Create: `tests/gtsr/test_sprint_f_replication.py` (F-01 only for now)

**Interfaces:**
- Consumes: `sha256`, `episode_keys`, `complete`, `usable`, `resample`, `multiseed_root` from `tests.gtsr._maniskill_data`; `euler_from_matrix`, `se3_from_pose6`, `pose6_from_se3`, `se3_inverse` from `tsr.gtsr.profile6`.
- Produces: `SE3_SEEDS`, `ARMS`, `GENERATOR_NAMES`, `YAW_INDEX`, `CONTEXT_SCALE`, `SE3_CONTEXT_MANIFEST_SHA256`, `SE3_EXPERIMENT_SHA256`, `SE3_SUBSET_MANIFEST_SHA256`, `EXPECTED_ISOLATED_CONTEXTS_6D`, `quat2mat`, `pose_to_pose6`, `phase_task_curve_se3`, `task_curve_se3`, `progress_grid_se3`, `nominal_frame_se3`, `se3_rollouts_root`, `load_se3_rollout`, `load_se3_manifests`, `load_se3_subsets`, `n30_subsets`.

- [ ] **Step 1: Write the failing test (F-01)**

Create `tests/gtsr/test_sprint_f_replication.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""F-01/F-04/F-05/F-06: frozen 6D six-generator replication (Sprint F).

Reads the pre-registered SE(3) phase-switch rollouts from the sister repository
``../maniskill_spectrum/phase_switch_symmetry_rollouts_se3`` and the manifests at
``../maniskill_spectrum/phase_switch_symmetry_multiseed``. F-01 verifies manifest /
rollout integrity; F-04 reproduces the frozen 18-pair Pdiag-finite selectivity
headline; F-05 is the Frame-weighted negative control; F-06 runs the full suite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from tests.gtsr._maniskill_data import multiseed_root, sha256
from tests.gtsr._maniskill_data_se3 import (
    ARMS,
    CONTEXT_SCALE,
    EXPECTED_ISOLATED_CONTEXTS_6D,
    SE3_CONTEXT_MANIFEST_SHA256,
    SE3_EXPERIMENT_SHA256,
    SE3_SEEDS,
    SE3_SUBSET_MANIFEST_SHA256,
    load_se3_manifests,
    load_se3_rollout,
    load_se3_subsets,
    n30_subsets,
    progress_grid_se3,
    se3_rollouts_root,
)

pytest.importorskip("h5py")
threadpoolctl = pytest.importorskip("threadpoolctl")

BINS = 25


@pytest.fixture(autouse=True)
def _single_threaded_blas():
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        yield


def _require_spectrum() -> Path:
    root = multiseed_root()
    if not (root / "se3_fixed_contexts.json").exists():
        pytest.skip(f"SE(3) multiseed manifests not found at {root}")
    if not (se3_rollouts_root() / "keyed_seed_20260818.h5").exists():
        pytest.skip(f"SE(3) rollouts not found at {se3_rollouts_root()}")
    return root


def test_F01_manifests_and_rollouts_integrity():
    root = _require_spectrum()
    assert sha256(root / "se3_fixed_contexts.json") == SE3_CONTEXT_MANIFEST_SHA256
    assert sha256(root / "se3_experiment.json") == SE3_EXPERIMENT_SHA256
    assert sha256(root / "se3_subsets.json") == SE3_SUBSET_MANIFEST_SHA256

    fixed, experiment, _ = load_se3_manifests(root)
    subsets = load_se3_subsets(root)
    assert subsets["experiment_sha256"] == SE3_EXPERIMENT_SHA256
    assert subsets["context_manifest_sha256"] == SE3_CONTEXT_MANIFEST_SHA256
    assert experiment["context_manifest_sha256"] == SE3_CONTEXT_MANIFEST_SHA256

    conditions = fixed["conditions"]
    assert len(conditions) == 75
    mixed = [c for c in conditions if c["generator"] == "mixed"]
    isolated = [c for c in conditions if c["generator"] not in ("mixed", "baseline")]
    baseline = [c for c in conditions if c["generator"] == "baseline"]
    assert len(mixed) == 60 and len(isolated) == 14 and len(baseline) == 1

    isolated_sorted = np.asarray(sorted((c["causal_delta"] for c in isolated), key=lambda v: tuple(v)), dtype=np.float64)
    expected_sorted = EXPECTED_ISOLATED_CONTEXTS_6D[
        np.lexsort(tuple(EXPECTED_ISOLATED_CONTEXTS_6D[:, i] for i in range(5, -1, -1)))
    ]
    np.testing.assert_allclose(isolated_sorted, expected_sorted, atol=1e-12)

    # 6 N=30 subsets, rank-6, and normalized condition number < 10.
    n30 = n30_subsets(subsets)
    assert len(n30) == 6
    for subset in n30:
        assert subset["sample_size"] == 30
        assert subset["rank"] == 6
        assert subset["condition_number"] < 10.0

    # Rollouts: every (arm, seed) has 75 usable episodes, 60 mixed, 14 isolated, 1 baseline;
    # the 60-mixed normalized context matrix is rank-6 with cond < 10.
    for arm in ARMS:
        for seed in SE3_SEEDS:
            path = se3_rollouts_root() / f"{arm}_seed_{seed}.h5"
            data = load_se3_rollout(path, BINS)
            assert len(data["usable_keys"]) == 75
            assert len(data["mixed_keys"]) == 60
            assert len(data["isolated_keys"]) == 14
            assert len(data["baseline_keys"]) == 1
            normalized = data["mixed_contexts"] / CONTEXT_SCALE
            assert np.linalg.matrix_rank(normalized) == 6
            assert np.linalg.cond(normalized) < 10.0


def test_F01_progress_grid_q_indices():
    progress, phase_codes = progress_grid_se3(BINS)
    assert progress.shape == (4 * BINS,)
    assert phase_codes.shape == (4 * BINS,)
    np.testing.assert_allclose(
        progress,
        np.concatenate([(i + np.linspace(0.0, 1.0, BINS)) / 4 for i in range(4)]),
    )
    np.testing.assert_array_equal(phase_codes, np.repeat([0, 1, 2, 3], BINS))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_sprint_f_replication.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.gtsr._maniskill_data_se3'`.

- [ ] **Step 3: Write minimal implementation**

Create `tests/gtsr/_maniskill_data_se3.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Test-side HDF5/JSON loader for the frozen SE(3) six-generator dataset.

Mirrors ``maniskill_spectrum``'s ``benchmark_se3_transfer.py`` extraction logic
(pose6 I/O via the local ``R = Rx Ry Rz`` convention, q-index ``progress_grid``,
nominal-frame recovery, mixed/isolated/baseline split) so Sprint F's F-01/F-04
can consume the pre-registered rollouts without pandas / transforms3d.

CRITICAL: unlike the planar loader, the frozen SE(3) ``progress_grid`` emits
q-indices ``[0, 1, 2, 3]`` (NOT solver codes ``[3, 4, 5, 6]``); ``pre = q<2``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.gtsr._maniskill_data import (
    complete,
    episode_keys,
    multiseed_root,
    resample,
    usable,
)
from tsr.gtsr.profile6 import (
    euler_from_matrix,
    pose6_from_se3,
    se3_from_pose6,
    se3_inverse,
)

SE3_SEEDS = (20260818, 20270818, 20280818)
ARMS = ("keyed", "circular_honest")
GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")
YAW_INDEX = 5
PHASE_CODES = (3, 4, 5, 6)

CONTEXT_SCALE = np.array(
    [0.012, 0.012, 0.012, 0.2617993877991494, 0.2617993877991494, 0.5235987755982988],
    dtype=np.float64,
)

SE3_CONTEXT_MANIFEST_SHA256 = "30d7f46f8ce73fc94704e0cdf433087de2f129156a45c90dd3c1a516ed5f64d5"
SE3_EXPERIMENT_SHA256 = "44e219fe7c00a1ad2a7761d61618655420cbbd29f542ee2b897a4e1ec4244e4a"
SE3_SUBSET_MANIFEST_SHA256 = "b14b6d26b52daa7d26fd832666d6c195731f2180726e499dd13edc57773664f0"

# The 14 isolated interventions, ordered as in se3_fixed_contexts.json.
EXPECTED_ISOLATED_CONTEXTS_6D = np.array(
    [
        [-0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.015, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.015, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.015, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.015, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.015, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -0.2617993877991494, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.2617993877991494, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.2617993877991494, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.2617993877991494, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.5235987755982988],
        [0.0, 0.0, 0.0, 0.0, 0.0, -0.2617993877991494],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2617993877991494],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5235987755982988],
    ],
    dtype=np.float64,
)


def quat2mat(q):
    """Reimplement ``transforms3d.quaternions.quat2mat`` for a ``[w, x, y, z]`` quaternion."""
    w, x, y, z = (float(v) for v in q)
    nq = w * w + x * x + y * y + z * z
    if nq < 1e-8:
        return np.eye(3)
    s = 2.0 / nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ = y * Y, y * Z
    zZ = z * Z
    return np.array(
        [
            [1.0 - (yY + zZ), xY - wZ, xZ + wY],
            [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
            [xZ - wY, yZ + wX, 1.0 - (xX + yY)],
        ],
        dtype=np.float64,
    )


def pose_to_pose6(pose):
    """SAPIEN raw pose ``(..., 7) = [x, y, z, w, x, y, z]`` -> ``[x, y, z, roll, pitch, yaw]``."""
    pose = np.asarray(pose, dtype=np.float64)
    single = pose.ndim == 1
    if single:
        pose = pose[None]
    rpy = np.stack([euler_from_matrix(quat2mat(q)) for q in pose[..., 3:]])
    out = np.concatenate([pose[..., :3], rpy], axis=-1)
    return out[0] if single else out


def phase_task_curve_se3(group, phase, bins):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    peg = np.asarray(group["peg_pose"])[indices]
    rpy = np.stack([euler_from_matrix(quat2mat(q)) for q in peg[:, 3:]])
    rpy = np.unwrap(rpy, axis=0)
    pose6 = np.column_stack([peg[:, 0], peg[:, 1], peg[:, 2], rpy])
    return resample(pose6, bins)


def task_curve_se3(group, bins):
    return np.concatenate([phase_task_curve_se3(group, phase, bins) for phase in PHASE_CODES], axis=0)


def progress_grid_se3(bins):
    phase_progress = np.linspace(0.0, 1.0, bins)
    progress = np.concatenate([(phase_index + phase_progress) / len(PHASE_CODES) for phase_index in range(4)])
    phase_codes = np.repeat(np.arange(4), bins)  # q-indices, NOT solver codes
    return progress, phase_codes


def nominal_frame_se3(data_file, keys, contexts):
    """Recover the nominal socket frame pose6: ``C0 = socket_world o intervention^-1``."""
    pose6s = []
    for key, context in zip(keys, contexts):
        socket_pose = np.asarray(data_file[key]["socket_pose"])[0]
        socket_pose6 = pose_to_pose6(socket_pose)
        socket_T = se3_from_pose6(socket_pose6)
        intervention_T = se3_from_pose6(np.asarray(context, dtype=np.float64))
        nominal_T = socket_T @ se3_inverse(intervention_T)
        pose6s.append(pose6_from_se3(nominal_T))
    pose6s = np.asarray(pose6s)
    nominal = np.empty(6, dtype=np.float64)
    nominal[:3] = pose6s[:, :3].mean(axis=0)
    for generator in range(3):
        angle = pose6s[:, 3 + generator]
        nominal[3 + generator] = float(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean()))
    return nominal


def se3_rollouts_root() -> Path:
    return multiseed_root().parent / "phase_switch_symmetry_rollouts_se3"


def load_se3_manifests(root):
    with (root / "se3_fixed_contexts.json").open(encoding="utf-8") as handle:
        fixed = json.load(handle)
    with (root / "se3_experiment.json").open(encoding="utf-8") as handle:
        experiment = json.load(handle)
    with (root / "se3_subsets.json").open(encoding="utf-8") as handle:
        subsets = json.load(handle)
    return fixed, experiment, subsets


def load_se3_subsets(root):
    with (root / "se3_subsets.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def n30_subsets(subsets):
    return [s for s in subsets["subsets"] if s["sample_size"] == 30]


def load_se3_rollout(path, bins):
    """Extract the usable mixed/isolated/baseline split and mixed curves from one arm/seed .h5."""
    import h5py

    with h5py.File(path, "r") as data_file:
        usable_groups = {
            int(data_file[key].attrs["condition_id"]): key
            for key in episode_keys(data_file)
            if usable(data_file[key]) and complete(data_file[key])
        }
        mixed_cond = {
            cid: key
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) == "mixed"
        }
        isolated_cond = {
            cid: key
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) in {"du", "dv", "dw", "roll", "pitch", "yaw"}
        }
        baseline_cond = {
            cid: key
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) == "baseline"
        }
        mixed_cids = sorted(mixed_cond)
        mixed_keys = [mixed_cond[cid] for cid in mixed_cids]
        mixed_contexts = np.asarray([np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys])
        mixed_curves = np.asarray([task_curve_se3(data_file[key], bins) for key in mixed_keys])
    return {
        "mixed_cids": mixed_cids,
        "mixed_keys": mixed_keys,
        "mixed_contexts": mixed_contexts,
        "mixed_curves": mixed_curves,
        "isolated_keys": sorted(isolated_cond.values()),
        "baseline_keys": sorted(baseline_cond.values()),
        "usable_keys": sorted(usable_groups.values()),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_sprint_f_replication.py -v`
Expected: PASS (2 tests), with the data present.

- [ ] **Step 5: Commit**

```bash
git add tests/gtsr/_maniskill_data_se3.py tests/gtsr/test_sprint_f_replication.py
git commit -m "test(gtsr): add 6D SE(3) loader and F-01 integrity gate"
```

---

### Task 4: SE(3) baseline models `_baseline_models_se3.py`

**Files:**
- Create: `tests/gtsr/_baseline_models_se3.py`
- Test: `tests/gtsr/test_profile6.py` (append baseline unit tests)

**Interfaces:**
- Consumes: `se3_to_metric`, `se3_from_metric` from `tsr.gtsr.profile6`.
- Produces: `SE3DiagonalOperatorModel`, `SE3FrameWeightedModel`, `SE3FullOperatorModel`, each with `.fit(contexts, curves, progress, phase_codes)`, `.predict(contexts)`, `.jacobian_diag()`, and `.name`.

- [ ] **Step 1: Write the failing test**

Append to `tests/gtsr/test_profile6.py`:

```python
# -- Task 4 imports (add to header) --------------------------------------------
from tests.gtsr._baseline_models_se3 import (
    SE3DiagonalOperatorModel,
    SE3FrameWeightedModel,
    SE3FullOperatorModel,
)


def test_baselines_jacobian_diag_shapes_and_frame_weighted_shared_weight():
    contexts, curves, progress, phase_codes, _, _ = _planted_recovery_data()
    for model in (
        SE3DiagonalOperatorModel(),
        SE3FrameWeightedModel(),
        SE3FullOperatorModel(),
    ):
        model.fit(contexts, curves, progress, phase_codes)
        assert model.jacobian_diag().shape == (100, 6)
    # Frame-weighted collapses every generator to one shared scalar w(s).
    fw = SE3FrameWeightedModel().fit(contexts, curves, progress, phase_codes)
    diagonal = fw.jacobian_diag()
    for generator in range(6):
        np.testing.assert_allclose(diagonal[:, generator], fw.weight, atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_profile6.py::test_baselines_jacobian_diag_shapes_and_frame_weighted_shared_weight -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.gtsr._baseline_models_se3'`.

- [ ] **Step 3: Write minimal implementation**

Create `tests/gtsr/_baseline_models_se3.py` (transcribed from `phase_switch_se3_baselines.py` at `e5ce8a3`):

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""SE(3) baseline models (Frame-weighted / Full operator / Pdiag pointwise).

Vendored from ``maniskill_spectrum``'s ``phase_switch_se3_baselines.py`` at commit
``e5ce8a3`` (blob ``35cf06a9``), minus the Pdiag-finite model (which is
:class:`tsr.gtsr.profile6.ProfileIdentifier6D`). These drive the F-05 negative
control and F-03's pointwise initialization reference.
"""

from __future__ import annotations

import numpy as np

from tsr.gtsr.profile6 import SE3_DIM, se3_from_metric, se3_to_metric


class SE3DiagonalOperatorModel:
    name = "Pdiag pointwise (SE(3))"

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        self.diagonal = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        for step in range(n_steps):
            for generator in range(SE3_DIM):
                design = np.column_stack([np.ones(len(contexts_metric)), contexts_metric[:, generator]])
                coefficients = np.linalg.lstsq(design, curves_metric[:, step, generator], rcond=None)[0]
                self.intercept_metric[step, generator] = coefficients[0]
                self.diagonal[step, generator] = coefficients[1]
        return self

    def predict(self, contexts):
        contexts_metric = se3_to_metric(contexts)
        prediction = self.intercept_metric[None, :, :] + self.diagonal[None, :, :] * contexts_metric[:, None, :]
        return se3_from_metric(prediction)

    def jacobian_diag(self):
        return self.diagonal.copy()


class SE3FrameWeightedModel:
    name = "Frame-weighted (SE(3))"

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        self.weight = np.empty(n_steps, dtype=np.float64)
        for step in range(n_steps):
            c_centered = contexts_metric - contexts_metric.mean(axis=0, keepdims=True)
            y = curves_metric[:, step]
            y_centered = y - y.mean(axis=0, keepdims=True)
            denominator = np.sum(c_centered * c_centered)
            self.weight[step] = np.sum(c_centered * y_centered) / denominator
            self.intercept_metric[step] = y.mean(axis=0) - self.weight[step] * contexts_metric.mean(axis=0)
        return self

    def predict(self, contexts):
        contexts_metric = se3_to_metric(contexts)
        prediction = self.intercept_metric[None, :, :] + self.weight[None, :, None] * contexts_metric[:, None, :]
        return se3_from_metric(prediction)

    def jacobian_diag(self):
        return np.repeat(self.weight[:, None], SE3_DIM, axis=1)


class SE3FullOperatorModel:
    name = "Full operator (SE(3))"

    def __init__(self, ridge=1e-8):
        self.ridge = ridge

    def fit(self, contexts, curves, progress, phase_codes):
        contexts_metric = se3_to_metric(contexts)
        curves_metric = se3_to_metric(curves)
        design = np.column_stack([np.ones(len(contexts_metric)), contexts_metric])
        regularizer = self.ridge * np.eye(design.shape[1])
        regularizer[0, 0] = 0.0
        inverse = np.linalg.inv(design.T @ design + regularizer) @ design.T
        n_steps = curves.shape[1]
        self.intercept_metric = np.empty((n_steps, SE3_DIM), dtype=np.float64)
        self.operator_metric = np.empty((n_steps, SE3_DIM, SE3_DIM), dtype=np.float64)
        for step in range(n_steps):
            coefficients = inverse @ curves_metric[:, step]
            self.intercept_metric[step] = coefficients[0]
            self.operator_metric[step] = coefficients[1:]
        return self

    def predict(self, contexts):
        contexts_metric = se3_to_metric(contexts)
        prediction = np.empty((len(contexts_metric), len(self.intercept_metric), SE3_DIM), dtype=np.float64)
        for step in range(len(self.intercept_metric)):
            prediction[:, step] = self.intercept_metric[step] + contexts_metric @ self.operator_metric[step]
        return se3_from_metric(prediction)

    def jacobian_diag(self):
        return np.diagonal(self.operator_metric, axis1=1, axis2=2).copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/gtsr/test_profile6.py -v`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/gtsr/_baseline_models_se3.py tests/gtsr/test_profile6.py
git commit -m "test(gtsr): vendor SE(3) baseline models for F-05 negative control"
```

---

### Task 5: `SelectivityMetrics6D` + `SelectivityVerdict6D`

**Files:**
- Modify: `src/tsr/gtsr/profile6.py` (append the classes)
- Test: `tests/gtsr/test_profile6.py` (append unit tests)

**Interfaces:**
- Consumes: `CellStatus`, `adjudicate_cell` from `tsr.gtsr.profile`; `excitation_mask6` (Task 2).
- Produces: `CellSelectivity` (dataclass: `alpha_pre`, `alpha_post`, `delta`, `delta_yaw`, `delta_others_max`, `rank1_yaw`, `transition`), `SelectivityStatus` enum, `SelectivityMetrics6D(phase_codes)` with `.per_cell(profile)`, `.cross_arm(keyed_cell, circular_cell)`, `.aggregate(keyed_cells, circular_cells)`, and `SelectivityVerdict6D` with `.identify(...)` and `.selectivity(...)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/gtsr/test_profile6.py`:

```python
# -- Task 5 imports (add to header) --------------------------------------------
from tsr.gtsr.profile6 import (
    SelectivityMetrics6D,
    SelectivityStatus,
    SelectivityVerdict6D,
)


def _synthetic_cell(phase_codes, yaw_pre=0.9, yaw_post=0.3, other_pre=0.95, other_post=0.95):
    profile = np.empty((len(phase_codes), 6))
    profile[:, 5] = np.where(phase_codes < 2, yaw_pre, yaw_post)
    profile[:, :5] = np.where(phase_codes < 2, other_pre, other_post)[:, None]
    return profile


def test_selectivity_metrics_per_cell_and_cross_arm():
    phase_codes = np.repeat(np.arange(4), 10)
    metrics = SelectivityMetrics6D(phase_codes)
    keyed = metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.9, yaw_post=0.3))
    circular = metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.1, yaw_post=0.1))
    assert keyed.delta_yaw == pytest.approx(0.6)
    assert keyed.delta_others_max == pytest.approx(0.0)
    assert keyed.rank1_yaw is True
    assert keyed.transition is True

    S, purity = metrics.cross_arm(keyed, circular)
    assert S[5] == pytest.approx(0.8)  # |0.9 - 0.1|
    assert purity == pytest.approx(S[5] / S.sum())
    assert S[0] == pytest.approx(0.0)


def test_selectivity_metrics_aggregate_matches_frozen_semantics():
    phase_codes = np.repeat(np.arange(4), 25)
    metrics = SelectivityMetrics6D(phase_codes)
    keyed_cells = [metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.9, yaw_post=0.3)) for _ in range(18)]
    circular_cells = [metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.1, yaw_post=0.1)) for _ in range(18)]
    headline = metrics.aggregate(keyed_cells, circular_cells)
    # S_yaw = mean_r |0.9-0.1| = 0.8; others = 0 -> purity = 1.0; rank/transition = 1.0.
    assert headline["S_yaw"] == pytest.approx(0.8)
    assert headline["S_others_max"] == pytest.approx(0.0)
    assert headline["purity"] == pytest.approx(1.0)
    assert headline["rank"] == pytest.approx(1.0)
    assert headline["transition"] == pytest.approx(1.0)


def test_selectivity_verdict_two_layer():
    phase_codes = np.repeat(np.arange(4), 25)
    metrics = SelectivityMetrics6D(phase_codes)
    keyed_cells = [metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.9, yaw_post=0.3)) for _ in range(18)]
    circular_cells = [metrics.per_cell(_synthetic_cell(phase_codes, yaw_pre=0.1, yaw_post=0.1)) for _ in range(18)]
    headline = metrics.aggregate(keyed_cells, circular_cells)

    contexts = np.random.default_rng(0).uniform(-0.012, 0.012, size=(60, 6))
    verdict = SelectivityVerdict6D(headline)
    identification = verdict.identify(
        contexts, observable=np.ones(6, dtype=bool), fit_succeeded=True, meets_gate=np.ones(6, dtype=bool)
    )
    selectivity = verdict.selectivity(identification, fit_succeeded=np.ones(6, dtype=bool))
    assert selectivity[5] is SelectivityStatus.SELECTIVE
    assert all(s is SelectivityStatus.NON_SELECTIVE for s in selectivity[:5])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_profile6.py::test_selectivity_metrics_per_cell_and_cross_arm -v`
Expected: FAIL with `ImportError: cannot import name 'SelectivityMetrics6D'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/tsr/gtsr/profile6.py` (add `from dataclasses import dataclass`, `from enum import Enum`, and `from .profile import CellStatus, adjudicate_cell` imports):

```python
@dataclass(frozen=True)
class CellSelectivity:
    """Per-cell selectivity quantities for one fitted profile (keyed or circular)."""

    alpha_pre: np.ndarray      # (6,) mean alpha over q<2
    alpha_post: np.ndarray     # (6,) mean alpha over q>=2
    delta: np.ndarray          # (6,) alpha_pre - alpha_post
    delta_yaw: float
    delta_others_max: float
    rank1_yaw: bool
    transition: bool


class SelectivityStatus(str, Enum):
    """Per-generator selectivity verdict (spec §7)."""

    SELECTIVE = "selective"
    NON_SELECTIVE = "non_selective"
    ABSTAIN = "abstain"


class SelectivityMetrics6D:
    """M1 / purity / M2 rank / M3 transition, per cell and 18-pair aggregate (spec §6)."""

    YAW_INDEX = 5
    TRANSITION_TAU = 0.2

    def __init__(self, phase_codes):
        self.phase_codes = np.asarray(phase_codes, dtype=np.int64)
        self.pre_mask = self.phase_codes < 2

    def per_cell(self, profile):
        profile = np.asarray(profile, dtype=np.float64)
        alpha_pre = profile[self.pre_mask].mean(axis=0)
        alpha_post = profile[~self.pre_mask].mean(axis=0)
        delta = alpha_pre - alpha_post
        delta_yaw = float(delta[self.YAW_INDEX])
        delta_others_max = float(np.max(np.delete(delta, self.YAW_INDEX)))
        rank1_yaw = bool(int(np.argmax(delta)) == self.YAW_INDEX and delta_yaw > 0.0)
        transition = bool(delta_yaw > self.TRANSITION_TAU)
        return CellSelectivity(alpha_pre, alpha_post, delta, delta_yaw, delta_others_max, rank1_yaw, transition)

    def cross_arm(self, keyed_cell, circular_cell):
        """S_j = |alpha_pre_j(keyed) - alpha_pre_j(circular)| and the per-cell purity."""
        S = np.abs(keyed_cell.alpha_pre - circular_cell.alpha_pre)
        s_yaw = float(S[self.YAW_INDEX])
        total = float(S.sum())
        purity = (s_yaw / total) if total > 0 else np.nan
        return S, purity

    def aggregate(self, keyed_cells, circular_cells):
        """Aggregate 18 ``(keyed, circular)`` pairs into the frozen headline dict."""
        if len(keyed_cells) != len(circular_cells):
            raise ValueError("keyed_cells and circular_cells must have equal length")
        s_yaw_vals = []
        s_others_max_vals = []
        purity_vals = []
        S_mean = np.zeros(6)
        for keyed, circular in zip(keyed_cells, circular_cells):
            S, purity = self.cross_arm(keyed, circular)
            s_yaw_vals.append(S[self.YAW_INDEX])
            s_others_max_vals.append(np.max(np.delete(S, self.YAW_INDEX)))
            purity_vals.append(purity)
            S_mean += S
        S_mean /= len(keyed_cells)
        return {
            "S_yaw": float(np.mean(s_yaw_vals)),
            "S_others_max": float(np.mean(s_others_max_vals)),
            "purity": float(np.mean(purity_vals)),
            "rank": float(np.mean([cell.rank1_yaw for cell in keyed_cells])),
            "transition": float(np.mean([cell.transition for cell in keyed_cells])),
            "S_mean": S_mean,
        }


class SelectivityVerdict6D:
    """Two-layer verdict: identification (``CellStatus``) + selectivity (spec §7)."""

    YAW_INDEX = 5

    def __init__(self, headline):
        self.headline = headline

    def identify(self, contexts, *, observable, fit_succeeded, meets_gate):
        """Per-generator ``CellStatus`` from data (6-generator excitation)."""
        excited = excitation_mask6(contexts)
        observable = np.broadcast_to(np.asarray(observable), (6,))
        fit_succeeded = np.broadcast_to(np.asarray(fit_succeeded), (6,))
        meets_gate = np.broadcast_to(np.asarray(meets_gate), (6,))
        return [
            adjudicate_cell(excited=bool(exc), observable=bool(obs), fit_succeeded=bool(fit), meets_heldout_gate=bool(gate))
            for exc, obs, fit, gate in zip(excited, observable, fit_succeeded, meets_gate)
        ]

    def selectivity(self, identification, *, fit_succeeded):
        """Per-generator ``SelectivityStatus``: yaw SELECTIVE iff all §6 gates pass."""
        headline = self.headline
        yaw_selective = bool(
            headline["S_yaw"] > 4.0 * headline["S_others_max"]
            and headline["rank"] == 1.0
            and headline["transition"] == 1.0
        )
        statuses = []
        for j, (cell_status, ok) in enumerate(zip(identification, np.broadcast_to(np.asarray(fit_succeeded), (6,)))):
            if not bool(ok):
                statuses.append(SelectivityStatus.ABSTAIN)
            elif j == self.YAW_INDEX:
                statuses.append(SelectivityStatus.SELECTIVE if yaw_selective else SelectivityStatus.NON_SELECTIVE)
            else:
                statuses.append(SelectivityStatus.NON_SELECTIVE)
        return statuses
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/gtsr/test_profile6.py -v`
Expected: PASS (18 tests).

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/profile6.py tests/gtsr/test_profile6.py
git commit -m "feat(gtsr): add SelectivityMetrics6D and SelectivityVerdict6D"
```

---

### Task 6: F-04 headline + F-05 negative control (integration)

**Files:**
- Create: `tests/gtsr/_fit_se3_cells.py`
- Modify: `tests/gtsr/test_sprint_f_replication.py` (append F-04 + F-05)

**Interfaces:**
- Consumes: loader (Task 3), baselines (Task 4), `ProfileIdentifier6D`/`SelectivityMetrics6D` (Tasks 2/5).
- Produces: `fit_all_pdiag_cells(root, processes=None, cache_dir=None) -> dict[(task, seed, subset_id), profile]`.

- [ ] **Step 1: Write the fitting helper**

Create `tests/gtsr/_fit_se3_cells.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Process-parallel, resumable Pdiag-finite cell fitting for Sprint F F-04/F-05.

36 cells = 2 arms x 3 seeds x 6 N=30 subsets. Each cell recovers its nominal frame
and fits :class:`tsr.gtsr.profile6.ProfileIdentifier6D`; per-cell profiles are
cached as ``<task>_<seed>_<subset_id>.npz`` so a mid-run failure resumes without
refitting already-completed cells. Fork-based parallelism inherits the parent's
single-thread BLAS pin.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from tests.gtsr._maniskill_data_se3 import (
    ARMS,
    SE3_SEEDS,
    load_se3_rollout,
    n30_subsets,
    nominal_frame_se3,
    progress_grid_se3,
    se3_rollouts_root,
)
from tsr.gtsr.profile6 import ProfileIdentifier6D

BINS = 25

# Module globals populated before forking (fork copies them into each worker).
_DATASETS = None
_PROGRESS = None
_PHASE_CODES = None
_SUBSETS = None


def _load_cell_datasets():
    global _DATASETS, _PROGRESS, _PHASE_CODES
    _PROGRESS, _PHASE_CODES = progress_grid_se3(BINS)
    _DATASETS = {
        arm: {seed: load_se3_rollout(se3_rollouts_root() / f"{arm}_seed_{seed}.h5", BINS) for seed in SE3_SEEDS}
        for arm in ARMS
    }


def _fit_cell(job):
    import h5py
    import threadpoolctl

    task, seed, subset = job
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        ds = _DATASETS[task][seed]
        cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
        indices = [cid_to_idx[cid] for cid in subset["source_condition_ids"]]
        contexts = ds["mixed_contexts"][indices]
        curves = ds["mixed_curves"][indices]
        keys = [ds["mixed_keys"][i] for i in indices]
        with h5py.File(se3_rollouts_root() / f"{task}_seed_{seed}.h5", "r") as data_file:
            nominal_frame_pose = nominal_frame_se3(data_file, keys, contexts)
        identifier = ProfileIdentifier6D()
        profile = identifier.fit(
            contexts, curves, _PROGRESS, _PHASE_CODES, nominal_frame_pose=nominal_frame_pose
        )
        return task, seed, subset["subset_id"], identifier.jacobian_diag(), bool(profile.optimization["success"])


def fit_all_pdiag_cells(root, processes=None, cache_dir=None):
    """Fit all 36 Pdiag-finite cells, caching per-cell profiles. Returns dict + success map."""
    from tests.gtsr._maniskill_data_se3 import load_se3_subsets

    subsets = n30_subsets(load_se3_subsets(root))
    cache_dir = Path(cache_dir or os.environ.get("TSR_SE3_CACHE", Path(tempfile.gettempdir()) / "tsr_se3_fit_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    _load_cell_datasets()
    jobs = [(task, seed, subset) for task in ARMS for seed in SE3_SEEDS for subset in subsets]

    profiles = {}
    success = {}
    pending = []
    for job in jobs:
        task, seed, subset = job
        cache_path = cache_dir / f"{task}_{seed}_{subset['subset_id']}.npz"
        if cache_path.exists():
            with np.load(cache_path) as loaded:
                profiles[(task, seed, subset["subset_id"])] = loaded["profile"]
                success[(task, seed, subset["subset_id"])] = bool(loaded["success"])
        else:
            pending.append(job)

    if pending:
        import multiprocessing

        n_workers = processes or min(8, os.cpu_count() or 1)
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            for task, seed, subset_id, profile, ok in pool.imap_unordered(_fit_cell, pending):
                profiles[(task, seed, subset_id)] = profile
                success[(task, seed, subset_id)] = ok
                np.savez(cache_dir / f"{task}_{seed}_{subset_id}.npz", profile=profile, success=ok)
    return profiles, success
```

- [ ] **Step 2: Write the failing F-04 + F-05 tests**

Append to `tests/gtsr/test_sprint_f_replication.py`:

```python
from tests.gtsr._baseline_models_se3 import SE3FrameWeightedModel
from tests.gtsr._fit_se3_cells import fit_all_pdiag_cells
from tests.gtsr._maniskill_data_se3 import (
    load_se3_rollout,
    n30_subsets,
    progress_grid_se3,
    se3_rollouts_root,
)
from tsr.gtsr.profile6 import SelectivityMetrics6D, SelectivityStatus, SelectivityVerdict6D

GATE_ATOL = 5e-3


def _frozen_pdiag_row(root, csv_name, model):
    with (root / "se3_transfer" / csv_name).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == model and row.get("sample_size") == "30":
                return row
    raise AssertionError(f"missing frozen N=30 row for {model} in {csv_name}")


def _fit_frame_weighted_cell(data, subset):
    cid_to_idx = {cid: i for i, cid in enumerate(data["mixed_cids"])}
    indices = [cid_to_idx[cid] for cid in subset["source_condition_ids"]]
    contexts = data["mixed_contexts"][indices]
    curves = data["mixed_curves"][indices]
    model = SE3FrameWeightedModel()
    progress, phase_codes = progress_grid_se3(BINS)
    model.fit(contexts, curves, progress, phase_codes)
    return model.jacobian_diag()


def _aggregate(metrics, profiles, seeds, subsets):
    """Pair keyed/circular by (seed, subset_id) and aggregate (spec §6)."""
    keyed_cells = []
    circular_cells = []
    for seed in seeds:
        for subset in subsets:
            sid = subset["subset_id"]
            keyed_cells.append(metrics.per_cell(profiles[("keyed", seed, sid)]))
            circular_cells.append(metrics.per_cell(profiles[("circular_honest", seed, sid)]))
    return metrics.aggregate(keyed_cells, circular_cells)


@pytest.mark.slow
def test_F04_frozen_pdiag_selectivity_headline():
    root = _require_spectrum()
    subsets = n30_subsets(load_se3_subsets(root))
    profiles, success = fit_all_pdiag_cells(root)

    # Every one of the 36 Pdiag fits must succeed (spec F-04).
    for seed in SE3_SEEDS:
        for subset in subsets:
            sid = subset["subset_id"]
            assert success[("keyed", seed, sid)], f"keyed fit failed for seed={seed} subset={sid}"
            assert success[("circular_honest", seed, sid)], f"circular fit failed for seed={seed} subset={sid}"

    progress, phase_codes = progress_grid_se3(BINS)
    metrics = SelectivityMetrics6D(phase_codes)
    headline = _aggregate(metrics, profiles, SE3_SEEDS, subsets)

    frozen_purity = _frozen_pdiag_row(root, "se3_transfer_purity.csv", "Pdiag finite (SE(3))")
    frozen_rank = _frozen_pdiag_row(root, "se3_transfer_M2_rank.csv", "Pdiag finite (SE(3))")
    frozen_trans = _frozen_pdiag_row(root, "se3_transfer_M3_transition.csv", "Pdiag finite (SE(3))")

    # Formal frozen-headline gate (spec §6.3 / F-04), against the frozen CSV values.
    assert abs(headline["S_yaw"] - float(frozen_purity["S_yaw_mean"])) <= GATE_ATOL
    assert abs(headline["S_others_max"] - float(frozen_purity["S_others_max_mean"])) <= GATE_ATOL
    assert abs(headline["purity"] - float(frozen_purity["purity_mean"])) <= GATE_ATOL
    assert headline["S_yaw"] > 4.0 * headline["S_others_max"]
    assert headline["rank"] == pytest.approx(float(frozen_rank["accuracy"]))
    assert headline["transition"] == pytest.approx(float(frozen_trans["accuracy"]))
    assert headline["rank"] == 1.0 and headline["transition"] == 1.0

    # Two-layer verdict: yaw SELECTIVE; the other five IDENTIFIED + NON_SELECTIVE.
    mixed = np.vstack([load_se3_rollout(se3_rollouts_root() / "keyed_seed_20260818.h5", BINS)["mixed_contexts"]])
    verdict = SelectivityVerdict6D(headline)
    identification = verdict.identify(
        mixed, observable=np.ones(6, dtype=bool), fit_succeeded=True, meets_gate=np.ones(6, dtype=bool)
    )
    selectivity = verdict.selectivity(identification, fit_succeeded=np.ones(6, dtype=bool))
    assert selectivity[5] is SelectivityStatus.SELECTIVE
    assert all(s is SelectivityStatus.NON_SELECTIVE for s in selectivity[:5])


@pytest.mark.slow
def test_F05_frame_weighted_negative_control():
    root = _require_spectrum()
    subsets = n30_subsets(load_se3_subsets(root))
    progress, phase_codes = progress_grid_se3(BINS)
    metrics = SelectivityMetrics6D(phase_codes)

    profiles = {}
    for arm in ARMS:
        for seed in SE3_SEEDS:
            data = load_se3_rollout(se3_rollouts_root() / f"{arm}_seed_{seed}.h5", BINS)
            for subset in subsets:
                profiles[(arm, seed, subset["subset_id"])] = _fit_frame_weighted_cell(data, subset)

    headline = _aggregate(metrics, profiles, SE3_SEEDS, subsets)
    frozen_purity = _frozen_pdiag_row(root, "se3_transfer_purity.csv", "Frame-weighted (SE(3))")
    frozen_rank = _frozen_pdiag_row(root, "se3_transfer_M2_rank.csv", "Frame-weighted (SE(3))")
    frozen_trans = _frozen_pdiag_row(root, "se3_transfer_M3_transition.csv", "Frame-weighted (SE(3))")

    # Frame-weighted collapses all six generators to one shared scalar, so it cannot
    # express per-generator selectivity: rank=0, transition=0, purity ~ 1/6.
    assert abs(headline["purity"] - float(frozen_purity["purity_mean"])) <= GATE_ATOL
    assert headline["rank"] == pytest.approx(float(frozen_rank["accuracy"]))
    assert headline["transition"] == pytest.approx(float(frozen_trans["accuracy"]))
    assert headline["rank"] == 0.0 and headline["transition"] == 0.0
    assert abs(headline["purity"] - 1.0 / 6.0) <= GATE_ATOL
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `pytest tests/gtsr/test_sprint_f_replication.py -v`
Expected: PASS. F-04 runs 36 Pdiag fits (~39 CPU min single-threaded; ~5 min with 8 processes, then instant from cache on re-runs). F-05 runs 36 cheap Frame-weighted fits.

- [ ] **Step 4: Commit**

```bash
git add tests/gtsr/_fit_se3_cells.py tests/gtsr/test_sprint_f_replication.py
git commit -m "test(gtsr): reproduce frozen 18-pair selectivity (F-04) and negative control (F-05)"
```

---

### Task 7: Export from `tsr.gtsr` + F-06 full regression

**Files:**
- Modify: `src/tsr/gtsr/__init__.py`
- Test: `tests/gtsr/test_sprint_f_replication.py` (append F-06)

**Interfaces:**
- Consumes: `ResponseProfile6D`, `ProfileIdentifier6D`, `SelectivityMetrics6D`, `SelectivityVerdict6D` from `tsr.gtsr.profile6`.
- Produces: public exports `ResponseProfile6D`, `ProfileIdentifier6D`, `SelectivityMetrics6D`, `SelectivityVerdict6D`.

- [ ] **Step 1: Write the failing test**

Append to `tests/gtsr/test_sprint_f_replication.py`:

```python
def test_F06_public_exports_and_no_planar_regression():
    import tsr.gtsr as gtsr
    from tsr.gtsr.profile6 import (
        ProfileIdentifier6D,
        ResponseProfile6D,
        SelectivityMetrics6D,
        SelectivityVerdict6D,
    )

    assert gtsr.ResponseProfile6D is ResponseProfile6D
    assert gtsr.ProfileIdentifier6D is ProfileIdentifier6D
    assert gtsr.SelectivityMetrics6D is SelectivityMetrics6D
    assert gtsr.SelectivityVerdict6D is SelectivityVerdict6D
    # Planar identifiers remain exported and untouched.
    assert gtsr.ResponseProfile is not None
    assert gtsr.ProfileIdentifier is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/gtsr/test_sprint_f_replication.py::test_F06_public_exports_and_no_planar_regression -v`
Expected: FAIL with `AttributeError: module 'tsr.gtsr' has no attribute 'ResponseProfile6D'`.

- [ ] **Step 3: Modify the export**

In `src/tsr/gtsr/__init__.py`, add the import and the four `__all__` entries:

```python
from .profile6 import (
    ProfileIdentifier6D,
    ResponseProfile6D,
    SelectivityMetrics6D,
    SelectivityVerdict6D,
)
```

And extend `__all__` with `"ResponseProfile6D", "ProfileIdentifier6D", "SelectivityMetrics6D", "SelectivityVerdict6D"`.

- [ ] **Step 4: Run the full suite (F-06)**

Run: `pytest tests/ -m "not slow" -v` (fast suite) then `pytest tests/gtsr/test_sprint_e_replication.py tests/gtsr/test_sprint_f_replication.py -v` (both replication modules).

Run: `ruff check src/tsr/gtsr/profile6.py tests/gtsr/`
Expected: no lint errors; full suite passes with no regression in `tests/tsr/`, legacy, or planar E-B.

- [ ] **Step 5: Commit**

```bash
git add src/tsr/gtsr/__init__.py tests/gtsr/test_sprint_f_replication.py
git commit -m "feat(gtsr): export 6D profile API (F-06)"
```

---

## Self-Review

**Spec coverage:** F-01 (Task 3) → manifest/rollout hashes, 75/75, 60/14/1, normalized rank-6 + cond<10, isolated amplitudes; F-02 (Task 2) → `action` ≡ `ResponseOperator` <1e-9, `jacobian_diag` shape (100,6), θ (6,24), metric scale; F-03 (Task 2) → planted recovery ≤5e-3; F-04 (Task 6) → 36 fits success + 18-pair aggregate `S_yaw`/`S_others_max`/purity/rank/transition vs frozen, `S_yaw > 4·S_others_max`; F-05 (Task 6) → Frame-weighted rank=0/transition=0/purity≈1/6; F-06 (Task 7) → exports + no regression. §8 runtime (process parallelism, per-process single-thread BLAS, resumable per-fit cache) → Task 6 `_fit_se3_cells.py`.

**Placeholder scan:** none — every step has concrete code.

**Type consistency:** `ProfileIdentifier6D.fit(..., nominal_frame_pose=None)` returns `ResponseProfile6D` with `.optimization["success"]`; `jacobian_diag()` → `(100,6)`; `SelectivityMetrics6D.per_cell` → `CellSelectivity` (fields `alpha_pre`, `delta_yaw`, `rank1_yaw`, `transition`, …); `cross_arm` → `(S, purity)`; `aggregate` → dict keys `S_yaw`, `S_others_max`, `purity`, `rank`, `transition`, `S_mean`; `SelectivityVerdict6D.identify`/`.selectivity` signatures match their call sites. Loader names (`n30_subsets`, `load_se3_rollout`, `se3_rollouts_root`, `progress_grid_se3`) match all imports across Tasks 3/6.
