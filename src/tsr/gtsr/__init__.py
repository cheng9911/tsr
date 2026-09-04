# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generalized Task Space Region (G-TSR) geometry kernel."""

from .charts import CanonicalTSRChart, GeneratorChart, PoEChart
from .domains import CircularInterval, CoordinateDomain, Interval, LinearInterval
from .metrics import SE3Metric
from .observability import (
    is_region_observable,
    region_observability_residual,
    symmetric_residual_set_discrepancy,
)
from .profile import CellStatus, ProfileIdentifier, ResponseProfile, cell_statuses, excitation_mask
from .projection import ProjectionResult
from .region import GTSR
from .response import ResponseOperator

__all__ = [
    "Interval",
    "LinearInterval",
    "CircularInterval",
    "CoordinateDomain",
    "GeneratorChart",
    "CanonicalTSRChart",
    "PoEChart",
    "SE3Metric",
    "ProjectionResult",
    "GTSR",
    "ResponseOperator",
    "ResponseProfile",
    "ProfileIdentifier",
    "CellStatus",
    "excitation_mask",
    "cell_statuses",
    "symmetric_residual_set_discrepancy",
    "region_observability_residual",
    "is_region_observable",
]
