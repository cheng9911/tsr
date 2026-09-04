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

from .metrics import SE3Metric


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


def region_observability_residual(base_region, action, n_samples=64, rng=None):
    """Max analytic region-fit residual of a response action over the base region.

    For sampled chart coordinates ``z``, the moved pose
    ``action @ base_region.to_transform(z)`` is residualized against ``base_region``
    using its exact canonical bounds (``GTSR.distance``), never a non-convex
    ``project()``. This is zero iff the action leaves the region set invariant
    (wide-domain symmetry -> ``UNOBSERVABLE``) and positive iff it carries poses out
    of the region (narrow-domain -> observable). The residual uses the region's
    coordinate displacement (not ``SE3Metric``'s principal log); only its zero /
    strictly-positive character feeds the verdict.
    """
    action = np.asarray(action, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()
    residual = 0.0
    for _ in range(n_samples):
        z = base_region.domain.sample(rng)
        moved = action @ base_region.to_transform(z)
        residual = max(residual, float(base_region.distance(moved)[0]))
    return residual


def is_region_observable(base_region, action, epsilon_obs, n_samples=64, rng=None):
    """Whether a response action measurably moves the region (E-A observability).

    ``False`` when the action is a region-set symmetry (residual <= ``epsilon_obs``,
    ``UNOBSERVABLE``); ``True`` otherwise. Wires the analytic E-A check into the
    cell-status adjudication path rather than a hardcoded observable flag.
    """
    return region_observability_residual(base_region, action, n_samples=n_samples, rng=rng) > epsilon_obs
