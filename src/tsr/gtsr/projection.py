# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generic projection of a query pose onto a G-TSR region."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ._validation import se3_transform
from .metrics import SE3Metric, se3_adjoint


@dataclass(frozen=True)
class ProjectionResult:
    z: np.ndarray
    transform: np.ndarray
    distance: float
    success: bool
    optimality: float | None
    message: str


def _projected_gradient_norm(g, z, lower, upper, tol_bounds):
    pg = np.zeros_like(g)
    for i in range(g.size):
        if lower[i] >= upper[i] - tol_bounds:
            pg[i] = 0.0  # degenerate (fixed) coordinate: no KKT requirement
        elif z[i] <= lower[i] + tol_bounds:
            pg[i] = min(g[i], 0.0)
        elif z[i] >= upper[i] - tol_bounds:
            pg[i] = max(g[i], 0.0)
        else:
            pg[i] = g[i]
    return float(np.max(np.abs(pg)))


def project(gtsr, T, metric, *, starts=None, tol_bounds=1e-12, tol_opt=1e-6, maxiter=1000):
    T = se3_transform(T, "T")
    if not isinstance(metric, SE3Metric):
        raise TypeError("metric must be an SE3Metric")
    domain = gtsr.domain
    lower = domain.cont_lower
    upper = domain.cont_upper
    m = domain.dim

    # build the full-transform body Jacobian wrapper
    Ad_Twe_inv = se3_adjoint(np.linalg.inv(gtsr.Tw_e))

    def T_of_z(z):
        return gtsr.to_transform(z)

    def fg(z):
        T_z = T_of_z(z)
        r = metric.error(T_z, T)
        J_local = gtsr.chart.jacobian(z)
        J_full = Ad_Twe_inv @ J_local
        E = metric.error_body_jacobian(T_z, T)
        grad = 2.0 * J_full.T @ (E.T @ metric.weighted_error(r))
        return metric.distance_sq(r), grad

    rng = np.random.default_rng(20260902)
    starts_list = [lower, upper, (lower + upper) / 2.0]
    starts_list.extend(domain.sample(rng) for _ in range(4))
    if starts is not None:
        starts_list = list(starts)
        if not starts_list:
            raise ValueError("starts must contain at least one initial coordinate vector")

    best_z = None
    best_d = float("inf")
    best_opt = None
    for z0 in starts_list:
        z0 = np.clip(np.asarray(z0, dtype=float), lower, upper)
        opt = minimize(
            fg,
            z0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(lower[i], upper[i]) for i in range(m)],
            options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
        )
        z = np.clip(opt.x, lower, upper)
        d = float(np.sqrt(fg(z)[0]))
        if d < best_d:
            best_d = d
            best_z = z
            best_opt = opt

    g_true = _finite_diff_gradient(fg, best_z, lower, upper, tol_bounds)
    opt_norm = _projected_gradient_norm(g_true, best_z, lower, upper, tol_bounds)
    bounds_ok = bool(np.all(best_z >= lower - tol_bounds) and np.all(best_z <= upper + tol_bounds))
    feasible = bool(bounds_ok and np.all(np.isfinite(best_z)) and np.isfinite(best_d))
    converged = bool(getattr(best_opt, "success", False))
    optimal = bool(opt_norm <= tol_opt)
    success = bool(converged and feasible and optimal)
    if success:
        message = "ok"
    else:
        reasons = []
        if not converged:
            reasons.append("optimizer did not converge")
        if not feasible:
            reasons.append("bounds infeasible or non-finite")
        if not optimal:
            reasons.append(f"KKT residual {opt_norm:.3e} > {tol_opt:.0e}")
        message = "; ".join(reasons) or "no convergence"
    return ProjectionResult(
        z=best_z,
        transform=T_of_z(best_z),
        distance=best_d,
        success=success,
        optimality=float(opt_norm) if success else None,
        message=message,
    )


def _finite_diff_gradient(fg, z, lower, upper, tol_bounds, h=1e-6):
    g = np.zeros(z.size)
    for i in range(z.size):
        if lower[i] >= upper[i] - tol_bounds:
            g[i] = 0.0  # degenerate (fixed) coordinate: no finite difference
            continue
        zp = z.copy()
        zm = z.copy()
        zp[i] = min(zp[i] + h, upper[i])  # clamp so fg never leaves the bounds
        zm[i] = max(zm[i] - h, lower[i])  # (to_transform raises out-of-bounds)
        denom = zp[i] - zm[i]
        g[i] = (fg(zp)[0] - fg(zm)[0]) / denom if denom > 0.0 else 0.0
    return g
