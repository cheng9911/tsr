# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest

from tsr.gtsr.charts import CanonicalTSRChart, GeneratorChart
from tsr.gtsr.domains import CoordinateDomain, LinearInterval
from tsr.gtsr.region import GTSR


class OneDimensionalChart(GeneratorChart):
    coord_names = ("x",)
    periodic = (False,)
    periods = (None,)

    def forward(self, z):
        z = np.asarray(z)
        transform = np.eye(4)
        transform[0, 3] = z[0]
        return transform


@pytest.mark.parametrize("value", [np.zeros((6, 1)), np.zeros(5), 0.0])
def test_canonical_forward_requires_exact_vector_shape(value):
    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        CanonicalTSRChart().forward(value)


@pytest.mark.parametrize("value", [[0, 0, 0, 0, np.nan, 0], [0, 0, 0, 0, np.inf, 0]])
def test_canonical_forward_requires_finite_coordinates(value):
    with pytest.raises(ValueError, match="finite"):
        CanonicalTSRChart().forward(value)


def test_gtsr_requires_non_null_matching_chart_and_domain():
    chart = CanonicalTSRChart()
    domain = CoordinateDomain([LinearInterval(0.0, 1.0)])
    with pytest.raises(TypeError, match="chart"):
        GTSR(chart=None, domain=domain)
    with pytest.raises(TypeError, match="domain"):
        GTSR(chart=chart, domain=None)
    with pytest.raises(ValueError, match="dimension mismatch"):
        GTSR(chart=chart, domain=domain)


@pytest.mark.parametrize(
    "transform",
    [
        np.eye(3),
        np.full((4, 4), np.nan),
        np.diag([1.0, 1.0, -1.0, 1.0]),
        np.diag([2.0, 1.0, 1.0, 1.0]),
        np.vstack((np.eye(3, 4), [0.0, 0.0, 1.0, 1.0])),
    ],
)
def test_gtsr_rejects_invalid_outer_transforms(transform):
    with pytest.raises((TypeError, ValueError)):
        GTSR.canonical(T0_w=transform)
    with pytest.raises((TypeError, ValueError)):
        GTSR.canonical(Tw_e=transform)


def test_constructor_defensively_copies_array_inputs():
    T0_w = np.eye(4)
    Tw_e = np.eye(4)
    Bw = np.zeros((6, 2))
    region = GTSR.canonical(T0_w=T0_w, Tw_e=Tw_e, Bw=Bw)
    T0_w[0, 3] = 10.0
    Tw_e[1, 3] = 10.0
    Bw[:, :] = 10.0
    np.testing.assert_array_equal(region.T0_w, np.eye(4))
    np.testing.assert_array_equal(region.Tw_e, np.eye(4))
    np.testing.assert_array_equal(region.domain.cont_lower, np.zeros(6))
    np.testing.assert_array_equal(region.domain.cont_upper, np.zeros(6))


def test_public_pose_methods_validate_transform_inputs():
    region = GTSR.canonical()
    for method in [region.to_xyzrpy, region.contains, region.distance]:
        with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
            method(np.eye(3))


def test_rotation_weight_contract():
    region = GTSR.canonical()
    for weight in [np.nan, np.inf, -1.0]:
        with pytest.raises(ValueError, match="rotation_weight"):
            region.distance(np.eye(4), rotation_weight=weight)
    with pytest.raises(TypeError, match="rotation_weight"):
        region.distance(np.eye(4), rotation_weight=object())


def test_generic_chart_composition_is_available_but_legacy_methods_are_restricted():
    region = GTSR(
        chart=OneDimensionalChart(),
        domain=CoordinateDomain([LinearInterval(-1.0, 1.0)]),
    )
    np.testing.assert_allclose(region.to_transform([0.25])[:3, 3], [0.25, 0.0, 0.0])
    for method in [region.is_valid, region.contains, region.distance, region.sample_xyzrpy]:
        with pytest.raises(TypeError, match="CanonicalTSRChart"):
            method(np.eye(4) if method in [region.contains, region.distance] else [0.0])
