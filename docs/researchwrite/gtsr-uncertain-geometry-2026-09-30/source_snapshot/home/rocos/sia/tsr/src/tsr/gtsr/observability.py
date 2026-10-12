# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Region-level observability primitives for Sprint E (E-A evidence layer).

The response of a generator is *region-observable* when a follow/ignore pair of
conditioned regions is strictly farther apart than a frozen calibration floor,
and *region-unobservable* when a response action leaves the region set invariant
(wide-domain symmetry). The quantity computed here is named the **symmetric
residual-set discrepancy** ``D_W`` — a directed-Hausdorff-like supremum over
principal-log residuals — and is deliberately NOT called a Hausdorff metric or
distance, because ``SE3Metric``'s principal-log residual does not satisfy the
triangle inequality (scope §6).
"""

import numpy as np

from .charts import CanonicalTSRChart
from .metrics import SE3Metric, se3_log


def symmetric_residual_set_discrepancy(points_a, points_b, metric):
    """Return ``D_W(A, B) = max{ sup_x inf_y d(x,y), sup_y inf_x d(y,x) }``.

    ``points_a`` / ``points_b`` are finite sets of SE(3) transforms shaped
    ``(n, 4, 4)``; ``metric`` provides ``.distance(T_from, T_to)``. The result is
    a symmetric residual-set discrepancy on the sampled points, not a global
    certificate for a continuum: global sup/inf requires an analytic region
    (see the E-A tests), never multi-start ``GTSR.project()``.
    """
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    if points_a.ndim != 3 or points_a.shape[1:] != (4, 4):
        raise ValueError(f"points_a must have shape (n, 4, 4), got {points_a.shape}")
    if points_b.ndim != 3 or points_b.shape[1:] != (4, 4):
        raise ValueError(f"points_b must have shape (n, 4, 4), got {points_b.shape}")
    distances = np.empty((len(points_a), len(points_b)), dtype=np.float64)
    for i in range(len(points_a)):
        for j in range(len(points_b)):
            distances[i, j] = metric.distance(points_a[i], points_b[j])
    directed_ab = float(distances.min(axis=1).max())
    directed_ba = float(distances.min(axis=0).max())
    return max(directed_ab, directed_ba)


def calibrate_epsilon_obs(control_discrepancies, epsilon_num=1e-6, quantile=0.99):
    """Freeze ``epsilon_obs = max(epsilon_num, Q_quantile(control D_W))``.

    The second term is the calibration/control split's discrepancy quantile; for
    a deterministic control it collapses to (near) zero and only ``epsilon_num``
    remains (scope §6). Must be computed before any held-out evaluation.
    """
    control = np.asarray(control_discrepancies, dtype=np.float64)
    if control.size == 0:
        return float(epsilon_num)
    floor = float(np.quantile(control, quantile))
    return float(max(epsilon_num, floor))


def region_points(region, rng, n_samples):
    """Sample ``n_samples`` poses from ``region`` via exact ``to_transform``."""
    return np.asarray(
        [region.to_transform(region.domain.sample(rng)) for _ in range(n_samples)],
        dtype=np.float64,
    )


def region_discrepancy(region_a, region_b, metric=None, n_samples=64, rng=None):
    """Sampled symmetric residual-set discrepancy between two regions.

    Both regions are evaluated at the *same* sampled chart coordinates, so two
    identical regions yield exactly zero (independent sampling of the same region
    returns a positive value and is therefore unusable as an observability oracle).
    This is a sampled estimate of ``D_W`` for genuinely distinct regions (e.g. a
    narrow-domain follow/ignore pair), not a global certificate; the two regions
    must share a chart/domain, as they do under ``ResponseOperator.condition``.
    """
    if metric is None:
        metric = SE3Metric()
    if rng is None:
        rng = np.random.default_rng()
    coordinates = [region_a.domain.sample(rng) for _ in range(n_samples)]
    points_a = np.asarray([region_a.to_transform(z) for z in coordinates], dtype=np.float64)
    points_b = np.asarray([region_b.to_transform(z) for z in coordinates], dtype=np.float64)
    return symmetric_residual_set_discrepancy(points_a, points_b, metric)


def region_observability_residual(base_region, action):
    """Return an exact residual for Sprint E's canonical axis controls.

    This intentionally supports only the analytic E-A controls: a pure local
    translation, or a pure local yaw with zero x/y translation bounds.  A bounded
    translation box is invariant only to zero translation; a full-circle yaw domain
    is invariant to pure yaw, while a proper yaw interval is not.  Other charts and
    coupled actions require a separate proof and are rejected rather than being
    silently decided from finite random samples.
    """
    if not isinstance(base_region.chart, CanonicalTSRChart):
        raise TypeError("analytic observability is restricted to CanonicalTSRChart")
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (4, 4) or not np.all(np.isfinite(action)):
        raise ValueError("action must be a finite transform shaped (4, 4)")

    local_action = np.linalg.inv(base_region.T0_w) @ action @ base_region.T0_w
    twist = se3_log(local_action)
    linear, angular = twist[:3], twist[3:]
    atol = 1e-12

    if np.allclose(angular, 0.0, atol=atol, rtol=0.0):
        # Every linear canonical coordinate has bounded support, so a nonzero
        # translation changes the region set. The magnitude is its exact maximum
        # coordinate residual at the translated boundary.
        return float(np.linalg.norm(linear))

    if not np.allclose(linear, 0.0, atol=atol, rtol=0.0) or not np.allclose(angular[:2], 0.0, atol=atol, rtol=0.0):
        raise ValueError("analytic observability supports only pure translation or pure local yaw")

    xy_bounds = np.column_stack((base_region.domain.cont_lower[:2], base_region.domain.cont_upper[:2]))
    if not np.allclose(xy_bounds, 0.0, atol=atol, rtol=0.0):
        raise ValueError("pure-yaw analytic control requires zero x/y translation bounds")
    yaw_span = base_region.domain.cont_upper[5] - base_region.domain.cont_lower[5]
    if np.isclose(yaw_span, 2.0 * np.pi, rtol=0.0, atol=atol):
        return 0.0
    return float(abs(angular[2]))


def is_region_observable(base_region, action, epsilon_obs):
    """Whether a response action measurably moves the region (E-A observability).

    ``False`` when an analytic action certificate establishes a region-set symmetry
    (residual <= ``epsilon_obs``, ``UNOBSERVABLE``); ``True`` otherwise. Unsupported
    geometry raises rather than inventing a status from finite samples.
    """
    epsilon_obs = float(epsilon_obs)
    if not np.isfinite(epsilon_obs) or epsilon_obs < 0.0:
        raise ValueError("epsilon_obs must be finite and non-negative")
    return region_observability_residual(base_region, action) > epsilon_obs
