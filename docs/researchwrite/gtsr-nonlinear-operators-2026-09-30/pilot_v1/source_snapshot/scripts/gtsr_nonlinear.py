"""Experimental nonlinear region and matched direct-map controls (not a new planner)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsr.gtsr._validation import finite_vector, se3_transform  # noqa: E402
from tsr.gtsr.charts import GeneratorChart  # noqa: E402
from tsr.gtsr.domains import CoordinateDomain  # noqa: E402
from tsr.gtsr.metrics import se3_exp, se3_vee  # noqa: E402
from tsr.gtsr.region import GTSR  # noqa: E402

METHODS = ("gtsr_analytic", "explicit_analytic", "explicit_autodiff")


def coaxial_pose(theta, height):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, height], [0, 0, 0, 1]], dtype=float)


@dataclass(frozen=True)
class Task:
    a: float
    rho: float
    b: float = 0.12
    h: float = 0.04
    q: float = 1.0

    def __post_init__(self):
        if not np.all(np.isfinite([self.a, self.rho, self.b, self.h, self.q])):
            raise ValueError("task parameters must be finite")
        if self.rho < 0 or not 0 < self.q < np.pi:
            raise ValueError("require rho >= 0 and 0 < q < pi")

    def height(self, theta):
        return self.b + self.h * theta + self.a * theta**2

    def domain(self):
        return CoordinateDomain.from_bounds([[0, self.q], [-self.rho, self.rho]], (False, False), (None, None))


@dataclass
class Counts:
    map_calls: int = 0
    jacobian_calls: int = 0
    autodiff_map_calls: int = 0
    ik_calls: int = 0
    collision_checks: int = 0

    def reset(self):
        for key in vars(self):
            setattr(self, key, 0)


@dataclass
class AnalyticMap:
    task: Task
    counts: Counts = field(default_factory=Counts)
    coord_names = ("psi", "eta")
    periodic = (False, False)
    periods = (None, None)
    dim = 2

    def forward(self, z):
        self.counts.map_calls += 1
        theta, eta = finite_vector(z, 2, "z")
        return coaxial_pose(theta, self.task.height(theta) + eta)

    def jacobian(self, z):
        self.counts.jacobian_calls += 1
        theta, _ = finite_vector(z, 2, "z")
        result = np.zeros((6, 2))
        result[2] = [self.task.h + 2 * self.task.a * theta, 1]
        result[5, 0] = 1
        return result


class QuadraticHelixChart(AnalyticMap, GeneratorChart):
    """Non-periodic progress plus axial slack; not an independent-coordinate PoE."""


class AutodiffMap:
    """Independent float64 PyTorch forward map; no analytic derivative supplied."""

    dim = 2

    def __init__(self, task):
        import torch

        self.torch, self.task, self.counts = torch, task, Counts()

    def _tensor_map(self, z):
        torch, task = self.torch, self.task
        self.counts.map_calls += 1
        theta, eta = z.unbind()
        zero, one = torch.zeros_like(theta), torch.ones_like(theta)
        c, s = torch.cos(theta), torch.sin(theta)
        height = task.b + task.h * theta + task.a * theta.square() + eta
        return torch.stack(
            (
                torch.stack((c, -s, zero, zero)),
                torch.stack((s, c, zero, zero)),
                torch.stack((zero, zero, one, height)),
                torch.stack((zero, zero, zero, one)),
            )
        )

    def forward(self, z):
        z = finite_vector(z, 2, "z")
        return self._tensor_map(self.torch.tensor(z, dtype=self.torch.float64)).detach().numpy()

    def jacobian(self, z):
        torch = self.torch
        self.counts.jacobian_calls += 1
        z = torch.tensor(finite_vector(z, 2, "z"), dtype=torch.float64, requires_grad=True)
        self.counts.autodiff_map_calls += 1
        derivative = torch.autograd.functional.jacobian(self._tensor_map, z).detach().numpy()
        transform = self._tensor_map(z).detach().numpy()
        inverse = np.linalg.inv(transform)
        return np.column_stack([se3_vee(inverse @ derivative[:, :, j]) for j in range(2)])


class DirectRegion:
    """Minimal adapter to the SAME optimizer, without GTSR construction.

    Uses the same bounds and pose validation as GTSR. Sharing the optimizer is
    deliberate: this control tests representation/derivative overhead, not a
    competing optimizer or planning algorithm.
    """

    def __init__(self, chart, domain, anchor):
        self.chart, self.domain = chart, domain
        self.T0_w, self.Tw_e = se3_transform(anchor, "anchor"), np.eye(4)

    def to_transform(self, z):
        z = finite_vector(z, self.chart.dim, "z")
        if not all(self.domain.contains(z)):
            raise ValueError("coordinates outside domain")
        return self.T0_w @ se3_transform(self.chart.forward(z), "map(z)") @ self.Tw_e


def anchor_transform():
    return se3_exp([0.03, -0.04, 0.02, 0.2, -0.1, 0.3])


def implementation(task, method):
    if method == "gtsr_analytic":
        chart = QuadraticHelixChart(task)
        return GTSR(T0_w=anchor_transform(), chart=chart, domain=task.domain())
    if method == "explicit_analytic":
        chart = AnalyticMap(task)
    elif method == "explicit_autodiff":
        chart = AutodiffMap(task)
    else:
        raise ValueError(f"unknown method {method}")
    return DirectRegion(chart, task.domain(), anchor_transform())


def check_pose(task, transform):
    """Independent analytic task check; no projection or chart inverse."""
    local = np.linalg.solve(anchor_transform(), transform)
    theta = float(np.arctan2(local[1, 0], local[0, 0]))
    eta = float(local[2, 3] - task.b - task.h * theta - task.a * theta**2)
    expected = coaxial_pose(theta, local[2, 3])
    return dict(
        theta=theta,
        eta=eta,
        band_violation=max(0.0, abs(eta) - task.rho),
        domain_violation=max(0.0, -theta, theta - task.q),
        off_axis_error=float(np.max(np.abs(local - expected))),
    )


def finite_difference_body_jacobian(forward, z, step=1e-6):
    """Matrix central difference; independent of Lie-log Jacobian implementation."""
    z = np.asarray(z, dtype=float)
    inverse = np.linalg.inv(forward(z))
    result = []
    for j in range(len(z)):
        delta = np.zeros_like(z)
        delta[j] = step
        result.append(se3_vee(inverse @ ((forward(z + delta) - forward(z - delta)) / (2 * step))))
    return np.column_stack(result)


def connection_maximum(task, first, last, mode):
    """Continuous axial-band residual for this model only; not collision checking."""
    first, last = np.asarray(first), np.asarray(last)
    if mode == "coordinate":
        return float(max(abs(first[1]), abs(last[1])))
    if mode != "pose_linear":
        raise ValueError("unknown connection mode")
    # e(t) = eta0 + (eta1-eta0)t + a*(psi1-psi0)^2*t*(1-t).
    curvature = task.a * (last[0] - first[0]) ** 2
    linear = last[1] - first[1] + curvature
    candidates = [0.0, 1.0]
    if curvature != 0:
        critical = linear / (2 * curvature)
        if 0 < critical < 1:
            candidates.append(float(critical))
    return float(max(abs(first[1] + linear * t - curvature * t**2) for t in candidates))


def piecewise_height(task, theta, segments):
    if isinstance(segments, bool) or not isinstance(segments, (int, np.integer)) or segments < 1:
        raise ValueError("segments must be a positive integer")
    grid = np.linspace(0, task.q, segments + 1)
    return np.interp(theta, grid, task.height(grid))
