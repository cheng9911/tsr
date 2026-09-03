# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from tsr.gtsr.charts import PoEChart
from tsr.gtsr.domains import CoordinateDomain, LinearInterval
from tsr.gtsr.metrics import SE3Metric, se3_exp
from tsr.gtsr.region import GTSR


def _region_1d(axis, lo, hi):
    chart = PoEChart([axis], representation="space")
    domain = CoordinateDomain([LinearInterval(lo, hi)])
    return GTSR(chart=chart, domain=domain)


def test_P01_query_from_internal_coords():
    axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    region = _region_1d(axis, -1.0, 1.0)
    metric = SE3Metric()
    z_true = np.array([0.35])
    Tq = region.to_transform(z_true)
    res = region.project(Tq, metric)
    assert res.success
    assert res.distance <= 1e-10
    np.testing.assert_allclose(res.z, z_true, atol=1e-6)
    np.testing.assert_allclose(res.transform, Tq, atol=1e-10)


def test_P02_zero_width_domain():
    axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    region = _region_1d(axis, 0.2, 0.2)
    metric = SE3Metric()
    Tq = se3_exp([0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = region.project(Tq, metric)
    np.testing.assert_allclose(res.z, [0.2], atol=1e-10)
    assert res.distance == pytest.approx(abs(0.9 - 0.2), abs=1e-10)


def test_P03_canonical_chart_projection_reconstructs_on_region():
    Bw = np.array(
        [
            [-0.2, 0.2],
            [-0.2, 0.2],
            [-0.2, 0.2],
            [-np.pi / 4, np.pi / 4],
            [-np.pi / 4, np.pi / 4],
            [-np.pi / 4, np.pi / 4],
        ]
    )
    region = GTSR.canonical(Bw=Bw)
    metric = SE3Metric()
    rng = np.random.default_rng(7)
    for _ in range(10):
        z = region.domain.sample(rng)
        Tq = region.to_transform(z)
        res = region.project(Tq, metric)
        assert res.success, res.message
        assert res.distance <= 1e-8
        assert region.contains(res.transform, metric=metric, tol=1e-8)
        np.testing.assert_allclose(res.transform, Tq, atol=1e-8)


def test_P03_canonical_chart_projection_matches_analytic_translation_clip():
    Bw = np.zeros((6, 2))
    Bw[0] = [-0.2, 0.2]
    region = GTSR.canonical(Bw=Bw)
    query = se3_exp([0.7, 0.0, 0.0, 0.0, 0.0, 0.0])

    res = region.project(query, SE3Metric())

    expected_z = np.zeros(6)
    expected_z[0] = 0.2
    assert res.success, res.message
    np.testing.assert_allclose(res.z, expected_z, atol=1e-10)
    np.testing.assert_allclose(res.transform, region.to_transform(expected_z), atol=1e-12)
    assert res.distance == pytest.approx(0.5, abs=1e-10)


def test_P04_1d_translation_analytic():
    axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    region = _region_1d(axis, -0.5, 1.5)
    metric = SE3Metric()
    for q in [-1.0, 0.0, 0.7, 2.0]:
        Tq = se3_exp([q, 0.0, 0.0, 0.0, 0.0, 0.0])
        res = region.project(Tq, metric)
        expected = float(np.clip(q, -0.5, 1.5))
        np.testing.assert_allclose(res.z, [expected], atol=1e-8)
        assert res.distance == pytest.approx(abs(q - expected), abs=1e-8)


def test_P04_1d_pure_rotation_analytic():
    chart = PoEChart([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], representation="space")
    domain = CoordinateDomain.from_bounds([[-0.5, 0.5]], chart.periodic, chart.periods)
    region = GTSR(chart=chart, domain=domain)
    metric = SE3Metric(rotation_weight=0.4)
    for q in [-1.0, -0.2, 0.3, 1.0]:
        query = se3_exp([0.0, 0.0, 0.0, 0.0, 0.0, q])
        res = region.project(query, metric)
        expected = float(np.clip(q, -0.5, 0.5))
        assert res.success, res.message
        np.testing.assert_allclose(res.z, [expected], atol=1e-8)
        assert res.distance == pytest.approx(0.4 * abs(q - expected), abs=1e-8)


def test_P05_1d_nonzero_pitch_screw_reference():
    h = 0.3
    axis = [0.0, 0.0, h, 0.0, 0.0, 1.0]  # unit w, pitch h
    region = _region_1d(axis, -1.0, 1.0)
    metric = SE3Metric(rotation_weight=0.5)

    def f(q):
        return metric.distance(region.to_transform([q]), Tq)

    for qq in [0.4, 1.8, -2.5]:
        Tq = se3_exp([0.1, -0.2, 0.5, 0.2, -0.3, qq])  # off-axis query
        # 密集网格先圈定全局最优的邻域，再用黄金分割精化（稳健于非单峰）
        grid = np.linspace(-1.0, 1.0, 2001)
        fvals = np.array([f(q) for q in grid])
        i = int(np.argmin(fvals))
        lo = grid[max(0, i - 1)]
        hi = grid[min(len(grid) - 1, i + 1)]
        ref = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-14})
        res = region.project(Tq, metric)
        assert res.distance <= ref.fun + 1e-8
        np.testing.assert_allclose(res.z, [ref.x], atol=1e-6)


def test_P06_idempotence():
    axis = [0.0, 0.0, 0.3, 0.0, 0.0, 1.0]
    region = _region_1d(axis, -1.5, 1.5)
    metric = SE3Metric()
    Tq = se3_exp([0.2, -0.4, 0.6, 0.5, -0.2, 0.0])
    res1 = region.project(Tq, metric)
    res2 = region.project(res1.transform, metric)
    assert res2.distance <= 1e-10
    np.testing.assert_allclose(res2.transform, res1.transform, atol=1e-10)


def test_P07_contains_iff_projection_residual():
    axis = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    region = _region_1d(axis, -1.0, 1.0)
    metric = SE3Metric()
    for q in [-1.0, 0.0, 0.9999, 1.0]:
        Tq = se3_exp([q, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert region.contains(Tq, metric=metric) is True
    for q in [1.5, -1.5]:
        Tq = se3_exp([q, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert region.contains(Tq, metric=metric) is False


def test_P07_canonical_sample_is_contained_with_default_generic_tolerance():
    bounds = np.array(
        [
            [-0.2, 0.2],
            [-0.2, 0.2],
            [-0.2, 0.2],
            [-np.pi / 4, np.pi / 4],
            [-np.pi / 4, np.pi / 4],
            [-np.pi / 4, np.pi / 4],
        ]
    )
    T0_w = se3_exp([0.4, -0.3, 0.2, 0.3, -0.2, 0.1])
    Tw_e = se3_exp([-0.2, 0.1, 0.3, -0.1, 0.25, -0.2])
    region = GTSR.canonical(T0_w=T0_w, Tw_e=Tw_e, Bw=bounds)
    metric = SE3Metric()
    rng = np.random.default_rng(7)

    for _ in range(30):
        sampled = region.sample(rng=rng)
        result = region.project(sampled, metric)
        assert result.distance <= 1e-8
        assert region.contains(sampled, metric=metric)


def test_P07_default_generic_tolerance_rejects_pose_clearly_outside():
    region = _region_1d([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], -1.0, 1.0)
    query = se3_exp([1.0 + 5e-8, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert region.contains(query, metric=SE3Metric()) is False


def test_P07_adversarial_four_dimensional_on_region_projection_refines_to_zero():
    axes = np.array(
        [
            [
                0.3798967875308519,
                0.38120767877168804,
                0.8052634576561934,
                0.9182479662747942,
                -0.04756885729586143,
                0.393138494996064,
            ],
            [
                -0.25006825790425286,
                0.00389918876136935,
                0.45489446031005,
                0.7793759086733303,
                0.08334137480600609,
                -0.6209890564454946,
            ],
            [
                0.1684538743093157,
                0.30020351622075714,
                -0.4325693612178967,
                0.752912215539425,
                0.13292586636757228,
                0.6445571423403395,
            ],
            [-0.8497967460288005, -0.18290927999749093, -0.4943578518944165, 0, 0, 0],
        ]
    )
    chart = PoEChart(axes, representation="space")
    domain = CoordinateDomain([LinearInterval(-2.5, 2.5) for _ in range(4)])
    T0_w = np.array(
        [
            [0.979363433842017, 0.1601168981958359, 0.12332819371626942, 0.4729117433104602],
            [-0.1983440587935259, 0.8786465094217749, 0.4343270033307174, -0.11002306181593996],
            [-0.03881879434608561, -0.4498253998976436, 0.8922724980702035, -0.0754394781911677],
            [0, 0, 0, 1],
        ]
    )
    Tw_e = np.array(
        [
            [0.6668969806952015, 0.13115885506431585, -0.7335160338246478, 0.408554815347245],
            [-0.2258725044783491, 0.9736551663375292, -0.03126065874092364, 0.21998494596870574],
            [0.7100915637157663, 0.18652874256386365, 0.6789528697466722, -0.05791418680587229],
            [0, 0, 0, 1],
        ]
    )
    region = GTSR(T0_w=T0_w, Tw_e=Tw_e, chart=chart, domain=domain)
    z = np.array([-0.7787102075745456, -2.221987116055273, 1.3143633499104905, -2.492406061304726])
    query = region.to_transform(z)

    result = region.project(query, SE3Metric(rotation_weight=0.7))

    assert result.success, result.message
    assert result.distance <= 1e-8
    assert region.contains(query, metric=SE3Metric(rotation_weight=0.7))


def test_P08_multidim_nonconvex_multistart_kkT():
    # two non-commuting generators over a bounded 2D domain
    S1 = [0.0, 0.0, 0.3, 0.0, 0.0, 1.0]
    S2 = [0.1, 0.0, 0.0, 1.0, 0.0, 0.0]
    chart = PoEChart([S1, S2], representation="space")
    domain = CoordinateDomain([LinearInterval(-1.5, 1.5), LinearInterval(-1.5, 1.5)])
    region = GTSR(chart=chart, domain=domain)
    metric = SE3Metric(rotation_weight=0.7)
    Tq = se3_exp([0.4, -0.3, 0.6, 0.5, -0.4, 0.2])
    res = region.project(Tq, metric)
    assert res.success, res.message
    assert res.optimality <= 1e-6
    # grid reference: best of a dense grid is not better than the found optimum (by > 1e-8)
    xs = np.linspace(-1.5, 1.5, 201)
    grid_best = min(metric.distance(region.to_transform([a, b]), Tq) for a in xs for b in xs)
    assert res.distance <= grid_best + 1e-8


def test_P09_failure_propagation_is_visible():
    axis = [0.0, 0.0, 0.3, 0.0, 0.0, 1.0]
    region = _region_1d(axis, -1.0, 1.0)
    metric = SE3Metric()
    Tq = se3_exp([0.2, -0.4, 0.6, 0.5, -0.2, 0.0])
    from tsr.gtsr.projection import project

    res = project(region, Tq, metric, maxiter=0)
    assert res.success is False
    assert res.message != "" and res.message != "ok"
    assert np.all(np.isfinite(res.z))
    assert np.isfinite(res.distance)
    assert res.transform.shape == (4, 4)


def test_project_rejects_empty_explicit_starts():
    region = _region_1d([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], -1.0, 1.0)
    with pytest.raises(ValueError, match="starts must contain at least one"):
        from tsr.gtsr.projection import project

        project(region, np.eye(4), SE3Metric(), starts=[])
