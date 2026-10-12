"""H0 properties and rejection gates; synthetic data only."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.linalg import expm

from tsr.gtsr import ResponseOperator, ResponseProfile6D
from tsr.gtsr.transfer import (
    CoordinateRangeSupport,
    GeneratorSemantics,
    MatchedRelationTransfer,
    MatchStatus,
    MonomialCorrespondence,
    PiecewiseLinearPhaseMap,
    compatibility_residual,
    physical_response_operator,
    se3_adjoint,
    transport_diagonal,
)


def hat(x):
    v, w = x[:3], x[3:]
    a, b, c = w
    out = np.zeros((4, 4))
    out[:3, :3] = [[0, -c, b], [c, 0, -a], [-b, a, 0]]
    out[:3, 3] = v
    return out


def phase_map():
    return PiecewiseLinearPhaseMap(
        ("align", "enter", "release", "insert"),
        ("align", "enter", "release", "insert"),
        (0, 1, 2, 3),
        ((0, 0.25, 1),) * 4,
        ((0, 0.5, 1),) * 4,
    )


def correspondence():
    return MonomialCorrespondence((2, 0, 1, 5, 3, 4), (2, -1, 0.5, -2, 1.5, 0.75))


def contract(**overrides):
    gamma = correspondence()
    ids = ("a", "b", "c", "d", "e", "f")
    args = dict(
        source_basis=np.eye(6),
        target_basis=gamma.inverse,
        frame_transform=np.eye(4),
        correspondence=gamma,
        semantics=GeneratorSemantics(ids, tuple(ids[j] for j in gamma.permutation)),
        phase_map=phase_map(),
        support=CoordinateRangeSupport((-1,) * 6, (1,) * 6, 1e-12),
        coordinate_scales=(1,) * 6,
        total_tolerance=1e-10,
        axis_tolerances=(1e-10,) * 6,
    )
    args.update(overrides)
    return MatchedRelationTransfer(**args)


def test_H001_H002_monomial_roundtrip_diagonal_and_copies():
    rng = np.random.default_rng(20260909)
    g = correspondence()
    for _ in range(100):
        x, alpha = rng.normal(size=6), rng.uniform(0, 1.25, 6)
        np.testing.assert_allclose(g.target_to_source(g.source_to_target(x)), x, atol=1e-12)
        np.testing.assert_allclose(
            g.matrix @ np.diag(alpha) @ g.inverse, np.diag(transport_diagonal(alpha, g)), atol=1e-12
        )
    g.matrix[:] = 0
    assert np.linalg.matrix_rank(g.matrix) == 6
    with pytest.raises(FrozenInstanceError):
        g.scales = (1,) * 6


@pytest.mark.parametrize(
    "perm", [(0,) * 6, (0, 1, 2, 3, 4, 6), tuple(range(5)), (False, 1, 2, 3, 4, 5), (0.0, 1, 2, 3, 4, 5)]
)
def test_invalid_monomial_permutation(perm):
    with pytest.raises(ValueError):
        MonomialCorrespondence(perm, (1,) * 6)


@pytest.mark.parametrize("scales", [(0,) * 6, (np.nan,) * 6, (np.inf,) * 6, (1,) * 5])
def test_invalid_monomial_scales(scales):
    with pytest.raises(ValueError):
        MonomialCorrespondence(tuple(range(6)), scales)


def test_H003_semantics_and_static_priority():
    ids = ("a", "b", "c", "d", "e", "f")
    wrong = contract(semantics=GeneratorSemantics(ids, ids), target_basis=np.eye(6) * 2)
    assert wrong.evaluate_contract().status is MatchStatus.UNMATCHED_FAMILY
    assert contract().evaluate_contract().status is MatchStatus.MATCHED


def test_H004_identity_compatibility():
    total, axes = compatibility_residual(
        np.eye(6), np.eye(6), np.eye(4), MonomialCorrespondence(tuple(range(6)), (1,) * 6), (1,) * 6
    )
    assert total == 0
    np.testing.assert_array_equal(axes, np.zeros(6))


def test_H005_H006_500_independent_matrix_exponentials_and_finite_actions():
    rng = np.random.default_rng(20260909)
    for _ in range(500):
        h = expm(hat(rng.normal(size=6) * 0.25))
        ad = se3_adjoint(h)
        bs = 4 * np.eye(6) + rng.normal(size=(6, 6)) * 0.2
        gamma = MonomialCorrespondence(tuple(rng.permutation(6)), tuple(rng.uniform(0.5, 2, 6)))
        bt = ad @ bs @ gamma.inverse
        alpha = rng.uniform(0, 1.25, 6)
        at = transport_diagonal(alpha, gamma)
        np.testing.assert_allclose(
            physical_response_operator(bt, at),
            ad @ physical_response_operator(bs, alpha) @ np.linalg.inv(ad),
            atol=1e-10,
            rtol=0,
        )
        xi = rng.normal(size=6) * 0.15
        np.testing.assert_allclose(expm(hat(ad @ xi)), h @ expm(hat(xi)) @ np.linalg.inv(h), atol=1e-10)
        total, axes = compatibility_residual(bs, bt, h, gamma, (1,) * 6)
        assert max(total, max(axes)) < 1e-12
        ref = expm(hat(rng.normal(size=6) * 0.1))
        relation = ref @ expm(hat(xi))
        source = ResponseOperator(ref, bs)
        target = ResponseOperator(h @ ref @ np.linalg.inv(h), bt)
        mapped_relation = h @ relation @ np.linalg.inv(h)
        np.testing.assert_allclose(
            target.coordinates(mapped_relation), gamma.source_to_target(source.coordinates(relation)), atol=1e-10
        )
        np.testing.assert_allclose(
            target.action(mapped_relation, at), h @ source.action(relation, alpha) @ np.linalg.inv(h), atol=1e-10
        )


@pytest.mark.parametrize("basis", [np.ones((6, 5)), np.zeros((6, 6)), np.eye(6) * np.nan])
def test_invalid_basis(basis):
    with pytest.raises(ValueError):
        physical_response_operator(basis, np.ones(6))
    with pytest.raises(ValueError):
        contract(source_basis=basis)


def test_adjoint_rejects_invalid_transform():
    h = np.eye(4)
    h[0, 0] = 2
    with pytest.raises(ValueError):
        se3_adjoint(h)


def test_H007_single_axis_veto_and_unit_reparameterization():
    g = correspondence()
    bt = g.inverse.copy()
    bt[:, 0] *= 1.01
    c = contract(target_basis=bt, total_tolerance=1)
    assert c.evaluate_contract().status is MatchStatus.INCOMPATIBLE_BASIS
    # Axis diagnostics are indexed by SOURCE generator, not target permutation.
    assert np.argmax(c.evaluate_contract().axis_residuals) == g.permutation[0]
    qs = np.diag([1000, 2, 3, 180 / np.pi, 5, 6])
    qt = np.diag([3, 4, 5, 6, 7, 8])
    gp = MonomialCorrespondence(g.permutation, tuple(np.diag(qt) * g.scales / np.diag(qs)[list(g.permutation)]))
    original = compatibility_residual(np.eye(6), bt, np.eye(4), g, (1,) * 6)
    changed = compatibility_residual(np.linalg.inv(qs), bt @ np.linalg.inv(qt), np.eye(4), gp, np.diag(qs))
    np.testing.assert_allclose(original[0], changed[0], atol=1e-14)
    np.testing.assert_allclose(original[1], changed[1], atol=1e-14)


def test_H008_phase_map_endpoints_warp_and_semantic_mismatch():
    p = phase_map()
    for q in range(4):
        assert p.map(q, 0) == (q, 0)
        assert p.map(q, 1) == (q, 1)
        assert p.map(q, 0.125) == (q, 0.25)
    wrong = replace(p, target_semantic_ids=("align", "enter", "still-keyed", "insert"))
    assert contract(phase_map=wrong).evaluate_contract().status is MatchStatus.UNMATCHED_FAMILY


@pytest.mark.parametrize("knots", [(0, 0, 1), (0, 0.6, 0.5, 1), (0.1, 1), (0, np.nan, 1), (0,)])
def test_phase_invalid_knots(knots):
    with pytest.raises(ValueError):
        replace(phase_map(), target_knots=(knots,) * 4)


@pytest.mark.parametrize("q,rho", [(-1, 0), (4, 0), (1.2, 0), (True, 0), (0, -0.1), (0, 1.1), (0, np.nan)])
def test_phase_invalid_query(q, rho):
    with pytest.raises(ValueError):
        phase_map().map(q, rho)


def test_phase_topology_rejected():
    with pytest.raises(ValueError):
        replace(phase_map(), source_phase_for_target=(0, 2, 1, 3))


def test_H009_mapped_support_and_tolerance():
    c = contract()
    g = c.correspondence
    for s in [-1, 1, 1 + 0.5e-12]:
        assert c.evaluate_query(g.source_to_target(np.full(6, s))).status is MatchStatus.MATCHED
    rejected = c.evaluate_query(g.source_to_target(np.full(6, 1 + 2e-12)))
    assert rejected.status is MatchStatus.OUT_OF_SUPPORT
    assert rejected.source_coordinates[0] > 1


@pytest.mark.parametrize("overrides", [dict(lower=(2,) * 6), dict(upper=(np.inf,) * 6), dict(tolerance=-1)])
def test_support_invalid(overrides):
    with pytest.raises(ValueError):
        replace(CoordinateRangeSupport((-1,) * 6, (1,) * 6, 0), **overrides)


def test_H010_profile_query_no_mutation_or_fallback():
    theta = np.arange(144).reshape(6, 24) / 50 - 1
    profile = ResponseProfile6D(theta)
    before = profile.to_dict()
    basis = np.eye(6)
    c = contract(source_basis=basis)
    basis[:] = 0
    r = c.query(profile, 2, 0.125, np.zeros(6))
    assert r.status is MatchStatus.MATCHED
    assert (r.source_phase, r.source_progress) == (2, 0.25)
    np.testing.assert_allclose(r.alpha_target, profile.alpha(2, 0.25)[list(c.correspondence.permutation)])
    assert profile.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        c.total_tolerance = 100
    failed = c.query(profile, 2, 0.125, np.full(6, 10))
    assert failed.status is MatchStatus.OUT_OF_SUPPORT
    assert failed.alpha_target is None
    assert failed.match == c.evaluate_query(np.full(6, 10))


def test_query_rejects_profile_phase_count_and_bad_parameters():
    with pytest.raises(ValueError, match="phase"):
        contract().query(ResponseProfile6D(np.zeros((6, 24)), n_phases=3), 0, 0, np.zeros(6))
    with pytest.raises(ValueError):
        contract().query(ResponseProfile6D(np.full((6, 24), np.nan)), 0, 0, np.zeros(6))


def test_H011_public_exports():
    import tsr.gtsr as public

    for name in (
        "MatchedRelationTransfer",
        "MonomialCorrespondence",
        "PiecewiseLinearPhaseMap",
        "CoordinateRangeSupport",
        "GeneratorSemantics",
        "MatchStatus",
        "MatchResult",
        "TransferQueryResult",
    ):
        assert name in public.__all__
        assert getattr(public, name) is not None


def test_adjoint_identity_cross_term_and_physical_response():
    np.testing.assert_array_equal(se3_adjoint(np.eye(4)), np.eye(6))
    h = np.eye(4)
    h[:3, 3] = [1, 2, 3]
    np.testing.assert_array_equal(se3_adjoint(h)[:3, 3:], [[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
    b = np.eye(6) + np.ones((6, 6)) * 0.2
    alpha = np.arange(6) / 5
    before = b.copy()
    np.testing.assert_allclose(physical_response_operator(b, alpha), b @ np.diag(alpha) @ np.linalg.inv(b))
    np.testing.assert_array_equal(b, before)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(total_tolerance=-1),
        dict(total_tolerance=np.nan),
        dict(axis_tolerances=(1,) * 5),
        dict(axis_tolerances=(-1,) * 6),
        dict(coordinate_scales=(0,) * 6),
        dict(coordinate_scales=(np.inf,) * 6),
        dict(source_basis=np.eye(6, dtype=complex)),
    ],
)
def test_contract_invalid_thresholds_and_units(overrides):
    with pytest.raises(ValueError):
        contract(**overrides)


def test_normalization_overflow_is_rejected_not_matching():
    # Individual source columns fit in float64; their joint norm does not.
    with pytest.raises(ValueError, match="normalization"):
        compatibility_residual(
            np.eye(6),
            np.eye(6),
            np.eye(4),
            MonomialCorrespondence(tuple(range(6)), (1,) * 6),
            (1.7e308,) * 6,
        )


def test_residual_underflow_cannot_turn_mismatch_into_match():
    bt = np.diag([1, 1, 1, 1.1, 1, 1])
    gamma = MonomialCorrespondence(tuple(range(6)), (1,) * 6)
    assert compatibility_residual(np.eye(6), bt, np.eye(4), gamma, (1,) * 6)[1][3] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="numerically unrepresentable"):
        compatibility_residual(np.eye(6), bt, np.eye(4), gamma, (1e-322,) * 6)


def test_residual_ratio_underflow_is_rejected():
    bs = 1e300 * np.eye(6)
    bt = bs.copy()
    bt[0, 1] = 1e-30
    with pytest.raises(ValueError, match="numerically unrepresentable"):
        compatibility_residual(bs, bt, np.eye(4), MonomialCorrespondence(tuple(range(6)), (1,) * 6), (1,) * 6)


def test_identity_phase_map_and_all_refusal_paths():
    p = replace(phase_map(), target_knots=((0, 1),) * 4, source_knots=((0, 1),) * 4)
    for q in range(4):
        assert p.map(q, 0.123) == (q, 0.123)
    profile = ResponseProfile6D(np.zeros((6, 24)))
    mismatched = replace(p, target_semantic_ids=("wrong", "enter", "release", "insert"))
    for c, status in [
        (contract(phase_map=mismatched), MatchStatus.UNMATCHED_FAMILY),
        (contract(target_basis=2 * correspondence().inverse), MatchStatus.INCOMPATIBLE_BASIS),
    ]:
        result = c.query(profile, 0, 0, np.zeros(6))
        assert result.status is status
        assert result.alpha_target is None
        assert result.source_phase is None
    with pytest.raises(TypeError):
        contract().query(object(), 0, 0, np.zeros(6))
    for coordinates in (np.ones(5), np.full(6, np.nan), np.zeros(6, dtype=complex)):
        with pytest.raises(ValueError):
            contract().query(profile, 0, 0, coordinates)


@pytest.mark.parametrize("kwargs", [dict(alpha_max=0), dict(alpha_max=np.inf), dict(basis_width=0)])
def test_query_invalid_profile_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        contract().query(ResponseProfile6D(np.zeros((6, 24)), **kwargs), 0, 0, np.zeros(6))


def test_identity_transfer_preserves_profile_and_query_coordinates():
    ids = tuple("abcdef")
    p = replace(phase_map(), target_knots=((0, 1),) * 4, source_knots=((0, 1),) * 4)
    c = contract(
        correspondence=MonomialCorrespondence(tuple(range(6)), (1,) * 6),
        target_basis=np.eye(6),
        semantics=GeneratorSemantics(ids, ids),
        phase_map=p,
    )
    profile = ResponseProfile6D(np.arange(144).reshape(6, 24) / 50 - 1)
    coordinates = np.linspace(-0.5, 0.5, 6)
    before = coordinates.copy()
    result = c.query(profile, 1, 0.33, coordinates)
    assert result.status is MatchStatus.MATCHED
    np.testing.assert_array_equal(result.alpha_target, profile.alpha(1, 0.33))
    np.testing.assert_array_equal(coordinates, before)
    coordinates[:] = 0
    np.testing.assert_array_equal(result.source_coordinates, before)
    with pytest.raises(FrozenInstanceError):
        result.alpha_target = (0,) * 6


@pytest.mark.parametrize("ids", ["abcdef", ("a",) * 6, tuple("abcde"), ("", "b", "c", "d", "e", "f")])
def test_invalid_semantic_ids(ids):
    with pytest.raises(ValueError):
        GeneratorSemantics(ids, tuple("abcdef"))
