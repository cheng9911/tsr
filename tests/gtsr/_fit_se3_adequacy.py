# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""SE(3) held-out adequacy computation for the frozen six-generator profile (Sprint F.1).

Transform-level residuals in the ``profile6`` convention: the response action is the
frozen left action ``C0 Exp(alpha . c) C0^{-1}`` (matching ``ProfileIdentifier6D.action``);
the residual ``delta_W`` uses ``SE3_METRIC_SCALE = [1,1,1,0.03,0.03,0.03]`` with rotation
components angle-wrapped.
"""

from __future__ import annotations

import numpy as np

from tsr.gtsr.profile6 import (
    ResponseProfile6D,
    pose6_from_se3,
    se3_exp,
    se3_from_pose6,
    se3_inverse,
    se3_log,
    se3_to_metric,
)

GENERATOR_INDEX = {"du": 0, "dv": 1, "dw": 2, "roll": 3, "pitch": 4, "yaw": 5}
ALPHA_FRAME = np.ones(6, dtype=np.float64)  # alpha_j = 1 (always-on)
ALPHA_FIXED = np.zeros(6, dtype=np.float64)  # alpha_j = 0 (no response)


def se3_response_action(nominal_frame_pose, context, alpha):
    """Return ``C0 Exp(alpha . c) C0^{-1}`` for a 6-vector intervention ``context``."""
    C0 = se3_from_pose6(np.asarray(nominal_frame_pose, dtype=np.float64))
    twist = se3_log(se3_from_pose6(np.asarray(context, dtype=np.float64)))
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.shape != twist.shape:
        raise ValueError(f"alpha must have shape {twist.shape}, got {alpha.shape}")
    return C0 @ se3_exp(alpha * twist) @ se3_inverse(C0)


def se3_transform_error(T_pred, T_obs):
    """``delta_W = ||se3_to_metric(log(T_pred^{-1} T_obs))||_2``.

    Left-invariant, coordinate-free transform residual; correct under non-commutative
    rotations and Euler singularities (unlike a pose6 coordinate difference).
    """
    T_pred = np.asarray(T_pred, dtype=np.float64)
    T_obs = np.asarray(T_obs, dtype=np.float64)
    r = se3_log(se3_inverse(T_pred) @ T_obs)
    return float(np.linalg.norm(se3_to_metric(r)))


def _generator_rows(generators, generator):
    return [i for i, name in enumerate(generators) if name == generator]


def _predicted_pose6(nominal_frame_pose, nominal_curve, context, alpha):
    action = se3_response_action(nominal_frame_pose, context, alpha)
    return pose6_from_se3(action @ se3_from_pose6(np.asarray(nominal_curve, dtype=np.float64)))


def held_out_response_error(nominal_frame_pose, isolated_contexts, isolated_curves,
                            baseline_curve, generators, generator, alpha_fn):
    """Metric #1: action-level transform residual, mean over c_j != 0 rows and steps."""
    rows = _generator_rows(generators, generator)
    if not rows:
        raise ValueError(f"no isolated rows for generator {generator}")
    errors = []
    for row in rows:
        context = isolated_contexts[row]
        observed = isolated_curves[row]
        for step in range(len(baseline_curve)):
            T_pred = se3_response_action(nominal_frame_pose, context, alpha_fn(step))
            T_obs = se3_from_pose6(observed[step]) @ se3_inverse(se3_from_pose6(baseline_curve[step]))
            errors.append(se3_transform_error(T_pred, T_obs))
    return float(np.mean(errors))


def held_out_task_error(nominal_frame_pose, nominal_curve, isolated_contexts, isolated_curves,
                        generators, generator, alpha_fn):
    """Metric #2: full-curve prediction error (action . nominal_curve) vs observed curve."""
    rows = _generator_rows(generators, generator)
    if not rows:
        raise ValueError(f"no isolated rows for generator {generator}")
    errors = []
    for row in rows:
        context = isolated_contexts[row]
        observed = isolated_curves[row]
        for step in range(len(nominal_curve)):
            pred = _predicted_pose6(nominal_frame_pose, nominal_curve[step], context, alpha_fn(step))
            errors.append(se3_transform_error(se3_from_pose6(pred), se3_from_pose6(observed[step])))
    return float(np.mean(errors))


def held_out_endpoint_error(nominal_frame_pose, nominal_curve, isolated_contexts, isolated_curves,
                            generators, generator, alpha_fn):
    """Metric #3: final-step component of metric #2."""
    rows = _generator_rows(generators, generator)
    if not rows:
        raise ValueError(f"no isolated rows for generator {generator}")
    errors = []
    last = len(nominal_curve) - 1
    for row in rows:
        context = isolated_contexts[row]
        observed = isolated_curves[row]
        pred = _predicted_pose6(nominal_frame_pose, nominal_curve[last], context, alpha_fn(last))
        errors.append(se3_transform_error(se3_from_pose6(pred), se3_from_pose6(observed[last])))
    return float(np.mean(errors))


def adequacy_summary(artifact, progress, isolated):
    """Per-generator held-out adequacy quantities for one arm/seed artifact.

    ``progress`` is the (steps,) normalized progress grid (``progress_grid_se3(BINS)[0]``).
    Returns a dict keyed by generator name with response/task/endpoint error and the
    paired differences vs the Frame (alpha=1) and Fixed (alpha=0) oracles.
    """
    nominal_frame_pose = artifact["nominal_frame_pose"]
    nominal_curve = artifact["nominal_curve"]
    profile = ResponseProfile6D(
        artifact["theta"], alpha_max=artifact["alpha_max"],
        basis_width=artifact["basis_width"], n_phases=artifact["n_phases"],
    )
    alpha_all = profile.alpha_from_g(np.asarray(progress, dtype=np.float64))  # (steps, 6)

    def fitted(step):
        return alpha_all[step]

    def frame(step):
        return ALPHA_FRAME

    def fixed(step):
        return ALPHA_FIXED

    isolated_contexts = isolated["isolated_contexts"]
    isolated_curves = isolated["isolated_curves"]
    baseline_curve = isolated["baseline_curve"]
    generators = isolated["isolated_generators"]

    summary = {}
    for generator in GENERATOR_INDEX:
        err_fitted = held_out_response_error(nominal_frame_pose, isolated_contexts, isolated_curves,
                                             baseline_curve, generators, generator, fitted)
        err_frame = held_out_response_error(nominal_frame_pose, isolated_contexts, isolated_curves,
                                            baseline_curve, generators, generator, frame)
        err_fixed = held_out_response_error(nominal_frame_pose, isolated_contexts, isolated_curves,
                                            baseline_curve, generators, generator, fixed)
        task_err = held_out_task_error(nominal_frame_pose, nominal_curve, isolated_contexts,
                                       isolated_curves, generators, generator, fitted)
        endpoint_err = held_out_endpoint_error(nominal_frame_pose, nominal_curve, isolated_contexts,
                                               isolated_curves, generators, generator, fitted)
        summary[generator] = {
            "response_error": err_fitted,
            "task_error": task_err,
            "endpoint_error": endpoint_err,
            "diff_vs_frame": err_fitted - err_frame,
            "diff_vs_fixed": err_fitted - err_fixed,
        }
    return summary


def primary_oracle(arm, generator):
    """Frozen primary comparison direction (spec §3 / Global Constraints).

    keyed × yaw          -> "frame" (superiority: fitted must beat non-phase-selective Frame)
    circular × yaw       -> "fixed" (equivalence: fitted must match Fixed, alpha_yaw ≈ 0)
    any arm × non-yaw 5  -> "frame" (equivalence: fitted must match Frame, alpha ≈ 1)
    """
    if generator == "yaw" and arm == "circular_honest":
        return "fixed"
    return "frame"


def aggregate_cells(cell_summaries):
    """Aggregate the 6 per-subset cell summaries of one arm × seed into a per-seed summary.

    The 6 N=30 subsets are fitting replicates of the same seed, NOT independent
    samples (they draw from one rollout and may overlap), so aggregation is a
    field-wise mean across subsets. The independent unit remains the seed (n=1 here).
    """
    generators = sorted(cell_summaries[0])
    fields = sorted(cell_summaries[0][generators[0]])
    out = {}
    for generator in generators:
        out[generator] = {
            field: float(np.mean([cell[generator][field] for cell in cell_summaries]))
            for field in fields
        }
    return out
