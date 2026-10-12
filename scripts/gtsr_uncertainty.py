"""Estimated helical bands: production samplers with no access to true b or h."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
from gtsr_coupled_geometry import Gate  # noqa: E402

from tsr.gtsr.charts import PoEChart  # noqa: E402
from tsr.gtsr.domains import CoordinateDomain  # noqa: E402
from tsr.gtsr.region import GTSR  # noqa: E402
from tsr.tsr import TSR  # noqa: E402

METHODS = ("gtsr", "tsr_union_64", "tsr_projected")
TRUE_B, TRUE_H, EPSILON, Q, RHO = 0.122, -0.05, 0.0015, 1.0, 0.0005


def uncertainty_bound(delta_b, delta_h, rho, q=Q, tracking_z=0.0, tracking_theta=0.0, h_max=0.0):
    """All lengths in meters, angles in radians; deterministic bounds, not SDs."""
    values = (delta_b, delta_h, rho, q, tracking_z, tracking_theta, h_max)
    if not all(np.isfinite(v) and v >= 0 for v in values):
        raise ValueError("bounds must be nonnegative and finite")
    return delta_b + q * delta_h + rho + tracking_z + h_max * tracking_theta


def exact_fixed_error_bound(error_b, error_h, rho, lower=0.0, upper=Q):
    return rho + max(abs(error_b + error_h * lower), abs(error_b + error_h * upper))


class EstimatedSampler:
    def __init__(self, method, b, h, rho=RHO, q=Q):
        self.method, self.b, self.h, self.rho, self.q = method, float(b), float(h), float(rho), float(q)
        if method not in METHODS or not np.all(np.isfinite([b, h, rho, q])) or rho < 0 or not 0 < q < np.pi:
            raise ValueError("invalid method or domain")
        anchor = np.eye(4)
        anchor[2, 3] = b
        if method == "gtsr":
            chart = PoEChart([[0, 0, h, 0, 0, 1], [0, 0, 1, 0, 0, 0]], representation="space")
            domain = CoordinateDomain.from_bounds(np.array([[0, q], [-rho, rho]]), chart.periodic, chart.periods)
            self.regions = [GTSR(T0_w=anchor, chart=chart, domain=domain)]
        else:
            self.regions = []
            count = 64 if method == "tsr_union_64" else 1
            for k in range(count):
                lo, hi = q * k / count, q * (k + 1) / count
                bounds = np.zeros((6, 2))
                bounds[2] = [min(h * lo, h * hi) - rho, max(h * lo, h * hi) + rho]
                bounds[5] = [lo, hi]
                self.regions.append(TSR(T0_w=anchor, Bw=bounds))

    def sample(self, uniform):
        u, v = np.asarray(uniform, float)
        if not (0 <= u < 1 and 0 <= v < 1):
            raise ValueError("uniforms outside [0,1)")
        theta = self.q * u
        if self.method == "gtsr":
            return self.regions[0].to_transform([theta, self.rho * (2 * v - 1)])
        count = len(self.regions)
        k = min(int(u * count), count - 1)
        lo, hi = self.q * k / count, self.q * (k + 1) / count
        low = min(self.h * lo, self.h * hi) - self.rho
        high = max(self.h * lo, self.h * hi) + self.rho
        z = low + v * (high - low)
        if self.method == "tsr_projected":
            z = self.h * theta + np.clip(z - self.h * theta, -self.rho, self.rho)
        return self.regions[k].to_transform([0, 0, z, 0, 0, theta])

    def residual(self, pose):
        theta = np.arctan2(pose[1, 0], pose[0, 0])
        return float(pose[2, 3] - self.b - self.h * theta)


def estimated_walls(b, h):
    """Known fixture extents, uncertain twist/phase; no true pitch supplied."""
    if h >= 0:
        raise ValueError("this physical fixture requires negative pitch")
    gate = Gate(angle=-Gate().travel / h)
    phase = (gate.start_depth - b) / h
    c, s = np.cos(phase), np.sin(phase)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for half, position, yaw in gate.walls():
        yield half, rotation @ position, yaw + phase


def estimated_route(accepted, b, h, rho=RHO, q=Q):
    """Shared monotone local connector; known entry/exit heights, no true band."""
    gate = Gate()
    endpoints = [[gate.start_depth, (gate.start_depth - b) / h], [gate.end_depth, (gate.end_depth - b) / h]]
    if not all(0 <= theta <= q for _, theta in endpoints):
        return None
    points = sorted([*endpoints, *accepted], key=lambda x: -x[0])
    if max(-np.diff(np.asarray(points)[:, 0])) > 0.004 + 1e-12:
        return None
    selected, index = [points[0]], 0
    while index < len(points) - 1:
        nxt = index + 1
        while nxt + 1 < len(points) and points[index][0] - points[nxt + 1][0] <= 0.004 + 1e-12:
            nxt += 1
        selected.append(points[nxt])
        index = nxt
    route = []
    for first, last in zip(selected[:-1], selected[1:]):
        n = max(1, int(np.ceil((first[0] - last[0]) / 0.001)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            z, theta = np.array(first) + t * (np.array(last) - first)
            if abs(z - b - h * theta) > rho + 1e-12:
                return None
            route.append([float(z), float(theta)])
    return route
