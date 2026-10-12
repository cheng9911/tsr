"""Known geometry of a layered helical gate and production TSR/G-TSR samplers."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tsr.gtsr.charts import PoEChart  # noqa: E402
from tsr.gtsr.domains import CoordinateDomain  # noqa: E402
from tsr.gtsr.region import GTSR  # noqa: E402
from tsr.tsr import TSR  # noqa: E402


@dataclass(frozen=True)
class Gate:
    angle: float = 0.8
    lower: float = 0.045
    upper: float = 0.085
    layers: int = 40
    pin_center: float = -0.037
    pin_halfheight: float = 0.001
    pin_halfx: float = 0.021
    pin_halfy: float = 0.014
    clearance: float = 0.002
    outer: float = 0.070
    yaw_tolerance: float = 0.030

    @property
    def travel(self):
        return self.upper - self.lower

    @property
    def start_depth(self):
        return self.upper - self.pin_center

    @property
    def end_depth(self):
        return self.lower - self.pin_center

    @property
    def final_depth(self):
        return self.end_depth - self.pin_halfheight - 0.004

    def pose(self, progress, yaw_error=0.0):
        yaw = self.angle * progress + yaw_error
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array(
            [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, self.start_depth - self.travel * progress], [0, 0, 0, 1]],
            dtype=float,
        )

    def walls(self):
        """Exact shared box primitives: (half sizes, local center, local yaw)."""
        inner_x, inner_y = self.pin_halfx + self.clearance, self.pin_halfy + self.clearance
        hz = self.travel / self.layers / 2
        side, cap = (self.outer - inner_x) / 2, (self.outer - inner_y) / 2
        for index in range(self.layers):
            z = self.upper - (2 * index + 1) * hz
            yaw = self.angle * (index + 0.5) / self.layers
            c, s = np.cos(yaw), np.sin(yaw)
            for half, xy in (
                ([side, self.outer, hz], [inner_x + side, 0]),
                ([side, self.outer, hz], [-inner_x - side, 0]),
                ([inner_x, cap, hz], [0, inner_y + cap]),
                ([inner_x, cap, hz], [0, -inner_y - cap]),
            ):
                x, y = xy
                yield np.array(half), np.array([c * x - s * y, s * x + c * y, z]), yaw

    def pin_clearance(self, progress, yaw):
        """Exact centered, axis-aligned pin containment in overlapping rectangular slices.

        The shaft fits inside the common inscribed circular bore. This check is
        only valid for centered pin poses with vertical axis, not arbitrary SE(3).
        Returns the minimum lateral clearance (meters); zero is contact.
        """
        z = self.upper - self.travel * float(progress)
        hz = self.travel / self.layers / 2
        centers = self.upper - (2 * np.arange(self.layers) + 1) * hz
        overlap = np.abs(centers - z) <= hz + self.pin_halfheight + 1e-12
        if not overlap.any():
            return float("inf")
        angles = self.angle * (np.arange(self.layers)[overlap] + 0.5) / self.layers
        delta = yaw - angles
        ex = self.pin_halfx * np.abs(np.cos(delta)) + self.pin_halfy * np.abs(np.sin(delta))
        ey = self.pin_halfx * np.abs(np.sin(delta)) + self.pin_halfy * np.abs(np.cos(delta))
        return float(min(np.min(self.pin_halfx + self.clearance - ex), np.min(self.pin_halfy + self.clearance - ey)))


class RegionSampler:
    """Same information and random uniforms; actual production forward APIs."""

    def __init__(self, gate, method):
        self.gate, self.method = gate, method
        self.count = int(method.rsplit("_", 1)[1]) if method.startswith("tsr_union_") else 1
        anchor = np.eye(4)
        anchor[2, 3] = gate.start_depth
        if method == "gtsr":
            first = [0, 0, -gate.travel / gate.angle, 0, 0, 1] if gate.angle else [0, 0, 1, 0, 0, 0]
            chart = PoEChart([first, [0, 0, 0, 0, 0, 1]], representation="space")
            bounds = [[0, gate.angle] if gate.angle else [-gate.travel, 0], [-gate.yaw_tolerance, gate.yaw_tolerance]]
            self.regions = [
                GTSR(
                    T0_w=anchor,
                    chart=chart,
                    domain=CoordinateDomain.from_bounds(np.array(bounds), chart.periodic, chart.periods),
                )
            ]
        elif method in ("tsr", "tsr_projected") or method.startswith("tsr_union_"):
            self.regions = []
            for index in range(self.count):
                lo, hi = index / self.count, (index + 1) / self.count
                bounds = np.zeros((6, 2))
                bounds[2] = [-gate.travel * hi, -gate.travel * lo]
                bounds[5] = [gate.angle * lo - gate.yaw_tolerance, gate.angle * hi + gate.yaw_tolerance]
                self.regions.append(TSR(T0_w=anchor, Bw=bounds))
        else:
            raise ValueError(f"unknown method: {method}")

    def sample(self, uniform):
        s, v = np.asarray(uniform, dtype=float)
        if not 0 <= s < 1 or not 0 <= v < 1:
            raise ValueError("uniforms must lie in [0,1)")
        g = self.gate
        if self.method == "gtsr":
            eta = g.yaw_tolerance * (2 * v - 1)
            value = self.regions[0].to_transform([g.angle * s if g.angle else -g.travel * s, eta])
        else:
            index = min(int(s * self.count), self.count - 1)
            yaw = g.angle * index / self.count - g.yaw_tolerance + v * (g.angle / self.count + 2 * g.yaw_tolerance)
            if self.method == "tsr_projected":
                yaw = g.angle * s + np.clip(yaw - g.angle * s, -g.yaw_tolerance, g.yaw_tolerance)
            value = self.regions[index].to_transform([0, 0, -g.travel * s, 0, 0, yaw])
        return value

    def expected_band_acceptance(self):
        g = self.gate
        return (
            1.0
            if self.method in ("gtsr", "tsr_projected")
            else (2 * g.yaw_tolerance / (g.angle / self.count + 2 * g.yaw_tolerance))
        )


def coordinates(gate, pose):
    progress = (gate.start_depth - pose[2, 3]) / gate.travel
    yaw = np.arctan2(pose[1, 0], pose[0, 0])
    return float(progress), float(yaw)


def in_band(gate, pose):
    s, yaw = coordinates(gate, pose)
    return -1e-12 <= s <= 1 + 1e-12 and abs(yaw - gate.angle * s) <= gate.yaw_tolerance + 1e-12
