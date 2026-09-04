# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from tests.gtsr.helpers import make_seed9917_region
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
    assert res.message == "stationary/KKT candidate; global optimality not certified"
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


def test_P08_multistart_uses_all_explicit_starts_independent_of_order():
    region, metric, _ = make_seed9917_region()
    z_query = np.array(
        [
            -0.7250909917227777,
            0.05350105481634815,
            -1.198540753054152,
            -0.9528767480423488,
            -0.7535481207602261,
            -0.06669751948316671,
        ]
    )
    bad = np.array(
        [
            0.6138846719834776,
            -0.9745710813253965,
            0.13177753171763507,
            0.11505237825646919,
            -0.064892209497172,
            -0.589829910735165,
        ]
    )
    good = np.array(
        [
            0.20823822437533823,
            0.9505431043362946,
            -0.6912116367248099,
            0.4883825844389551,
            0.38140176154321614,
            -0.6622489371555484,
        ]
    )
    query = region.to_transform(z_query)

    bad_result = region.project(query, metric, starts=[bad])
    forward = region.project(query, metric, starts=[bad, good])
    reverse = region.project(query, metric, starts=[good, bad])
    default = region.project(query, metric)

    assert bad_result.success, bad_result.message
    assert forward.success, forward.message
    assert reverse.success, reverse.message
    assert default.success, default.message
    assert bad_result.distance > 1.0
    assert forward.distance < 1e-10
    assert reverse.distance < 1e-10
    assert default.distance < 1e-10
    assert bad_result.distance - forward.distance > 1.0
    assert bad_result.distance - reverse.distance > 1.0


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


def test_projection_result_arrays_are_defensive_and_read_only():
    from tsr.gtsr.projection import ProjectionResult

    source_z = np.array([0.25])
    source_transform = np.eye(4)
    result = ProjectionResult(source_z, source_transform, 0.0, True, 0.0, "test")
    source_z[0] = 0.5
    source_transform[0, 3] = 1.0

    np.testing.assert_array_equal(result.z, [0.25])
    np.testing.assert_array_equal(result.transform, np.eye(4))
    with pytest.raises(ValueError, match="read-only"):
        result.z[0] = 0.75
    with pytest.raises(ValueError, match="read-only"):
        result.transform[0, 3] = 2.0


def test_project_rejects_empty_explicit_starts():
    region = _region_1d([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], -1.0, 1.0)
    with pytest.raises(ValueError, match="starts must contain at least one"):
        from tsr.gtsr.projection import project

        project(region, np.eye(4), SE3Metric(), starts=[])


def _seed441_region():
    """Deterministic 6D PoE region whose cold-start projection is a known false negative.

    seed 441: a genuinely on-region pose gets projected to a wrong local minimum by the
    multi-start local optimizer (distance ~0.128, success=True, contains=False). This is
    the documented contract boundary — `project` returns a stationary/KKT candidate, not a global
    certificate — and the warm-start path recovers the true zero-residual coordinate.
    """
    axes = np.array(
        [
            [
                0.5749974488564747,
                1.137659156106971,
                -0.2658632045960905,
                -0.7125973331600972,
                -0.3469752041376849,
                0.6097649124758896,
            ],
            [
                -0.5104899947281174,
                0.08200002355037285,
                1.3527619604051027,
                0.3864367037372058,
                0.00382378072136638,
                -0.9223080031669019,
            ],
            [
                1.2146276836016616,
                0.7554551516492272,
                -0.8834335453416841,
                0.10130284100039917,
                -0.7228177971862456,
                -0.6835730878816626,
            ],
            [
                0.5808667206001397,
                1.5167620243184028,
                -1.101115915125199,
                0.43214024916983707,
                0.8796552597883519,
                0.1986490094969471,
            ],
            [
                -0.9335804271926613,
                -0.5696054669192491,
                -0.4893650813350019,
                0.7551562230755159,
                0.0396575144819913,
                -0.6543442215649438,
            ],
            [
                -0.6886402460042074,
                0.3496360179696395,
                -0.6352395347596044,
                0.0,
                0.0,
                0.0,
            ],
        ]
    )
    chart = PoEChart(axes, representation="space")
    bounds = np.array([[-1.2, 1.2]] * 6)
    domain = CoordinateDomain.from_bounds(bounds, chart.periodic, chart.periods)
    T0_w = np.array(
        [
            [0.9768787685984828, 0.19778322435638954, -0.08117676776456452, -0.20859879507248927],
            [-0.18494123929239256, 0.972250980585315, 0.14326468078311377, -0.20022655723824767],
            [0.10725954256151736, -0.12493929291493361, 0.9863496153065658, 0.41161399403998583],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    Tw_e = np.array(
        [
            [0.9932767748391301, -0.0858897337203418, 0.07761573427227747, -0.2006426490020646],
            [0.04241776563525183, 0.8938675068150195, 0.4463200795604137, 0.05785359461913411],
            [-0.10771249567106342, -0.4400270831457422, 0.8915010848981403, -0.2746251827369469],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    region = GTSR(T0_w=T0_w, Tw_e=Tw_e, chart=chart, domain=domain)
    z = np.array(
        [
            -0.8113536352430053,
            -0.9810042428385752,
            1.0802377113666057,
            0.5942187059101076,
            -0.449494606703843,
            -0.6553382226474657,
        ]
    )
    return region, z, SE3Metric(rotation_weight=0.7)


def test_P10_seed441_known_cold_start_false_negative_canary():
    region, z, metric = _seed441_region()
    query = region.to_transform(z)
    # sanity: the query is genuinely on-region (built from a feasible coordinate)
    np.testing.assert_allclose(query, region.to_transform(z), atol=0.0)

    result = region.project(query, metric)  # cold start: multi-start, no warm start

    if result.distance > 1e-8:
        assert result.success, result.message
        assert np.isfinite(result.distance)
        assert np.all(np.isfinite(result.z))
        assert np.all(np.isfinite(result.transform))
        assert np.all(result.z >= region.domain.cont_lower - 1e-12)
        assert np.all(result.z <= region.domain.cont_upper + 1e-12)
        assert result.message == "stationary/KKT candidate; global optimality not certified"
        assert result.distance == pytest.approx(0.12828521902191672, abs=1e-8)
        assert result.z[-1] == pytest.approx(-1.2, abs=1e-12)
        pytest.xfail(
            "seed-441: cold-start projection falls into a local minimum "
            f"(distance={result.distance:.6g}); known false negative, warm-start recovers it"
        )
    # If the optimizer finds the true basin, this remains a normal passing
    # regression test; the suite's xfail count changes and documents that the
    # known false-negative boundary has been removed.
    assert result.success, result.message
    assert result.distance <= 1e-8
    assert np.all(result.z >= region.domain.cont_lower - 1e-12)
    assert np.all(result.z <= region.domain.cont_upper + 1e-12)
    np.testing.assert_allclose(result.transform, query, atol=1e-8)
    assert region.contains(query, metric=metric)


def test_P10_seed441_warm_start_recovers_true_membership():
    region, z, metric = _seed441_region()
    query = region.to_transform(z)
    result = region.project(query, metric, starts=[z])
    assert result.success, result.message
    assert result.distance <= 1e-10
    np.testing.assert_allclose(result.z, z, atol=1e-8)
