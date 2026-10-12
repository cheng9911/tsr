"""Independent scalar projection reference for the restricted coaxial query family.

After eliminating eta, F(t)=max(|f(t)-zq|-rho,0)^2+w^2(t-tq)^2.
Every minimum on [0,Q] is an endpoint, a piece switch f(t)=zq +/- rho,
the angular minimizer tq in a zero-axial-error piece, or a stationary point
of one of the outer quartic pieces. No production map/optimizer is used.
"""

from __future__ import annotations

import mpmath as mp
import numpy as np
from numpy.polynomial import Polynomial


def reference_projection(task, theta_query, height_query, weight):
    if not np.all(np.isfinite([theta_query, height_query, weight])) or weight <= 0:
        raise ValueError("invalid query or rotation weight")
    if max(abs(theta_query), abs(theta_query - task.q)) >= np.pi:
        raise ValueError("reference requires all yaw differences strictly below pi")
    failures = []
    with mp.workdps(60):
        a, b, h, rho, q, tq, zq, w = map(
            mp.mpf, map(str, (task.a, task.b, task.h, task.rho, task.q, theta_query, height_query, weight))
        )
        f = Polynomial([task.b, task.h, task.a])
        polynomials = [f - height_query - task.rho, f - height_query + task.rho]
        for sign in (-1, 1):
            polynomials.append(
                (f + sign * task.rho - height_query) * f.deriv() + weight**2 * Polynomial([-theta_query, 1])
            )
        candidates = [mp.mpf(0), q, min(q, max(mp.mpf(0), tq))]
        root_residuals = []
        for index, polynomial in enumerate(polynomials):
            for root in polynomial.roots():
                if abs(root.imag) > 1e-7 or not -1e-8 <= root.real <= task.q + 1e-8:
                    continue
                if index < 2:
                    offset = -rho if index == 0 else rho

                    def equation(t, offset=offset):
                        return b + h * t + a * t * t - zq + offset
                else:
                    offset = -rho if index == 2 else rho

                    def equation(t, offset=offset):
                        return (b + h * t + a * t * t + offset - zq) * (h + 2 * a * t) + w * w * (t - tq)

                try:
                    refined = mp.findroot(equation, mp.mpf(str(root.real)), tol=mp.mpf("1e-45"), maxsteps=100)
                    residual = abs(equation(refined))
                    if not mp.isfinite(refined) or residual > mp.mpf("1e-30"):
                        raise ValueError("root refinement residual too large")
                    if -mp.mpf("1e-12") <= refined <= q + mp.mpf("1e-12"):
                        candidates.append(min(q, max(mp.mpf(0), refined)))
                        root_residuals.append(float(residual))
                except (ValueError, ZeroDivisionError) as exc:
                    failures.append(str(exc))

        def objective(t):
            axial = max(mp.mpf(0), abs(b + h * t + a * t * t - zq) - rho)
            return axial * axial + w * w * (t - tq) ** 2

        best = min(candidates, key=objective)
        eta = min(rho, max(-rho, zq - b - h * best - a * best * best))
        distance = mp.sqrt(objective(best))
        exact_text = mp.nstr(distance, 55)
        coords = [float(best), float(eta)]
    grid = np.linspace(0, task.q, 20001)
    grid_objective = np.maximum(np.abs(task.b + task.h * grid + task.a * grid**2 - height_query) - task.rho, 0) ** 2
    grid_objective += weight**2 * (grid - theta_query) ** 2
    grid_distance = float(np.sqrt(np.min(grid_objective)))
    verified = not failures and np.isfinite(float(distance)) and float(distance) <= grid_distance + 1e-12
    return dict(
        z=coords,
        distance=float(distance),
        distance_decimal=exact_text,
        verified=bool(verified),
        certification="analytic_candidate_enumeration_60_digit_refinement_not_intervals",
        candidate_count=len(candidates),
        max_root_residual=max(root_residuals, default=0),
        dense_grid_distance=grid_distance,
        grid_step=task.q / 20000,
        refinement_failures=failures,
    )
