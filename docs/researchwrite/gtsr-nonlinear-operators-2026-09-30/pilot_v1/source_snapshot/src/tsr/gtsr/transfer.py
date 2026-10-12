# SPDX-License-Identifier: MIT
"""Guarded transport of diagonal response laws under provided correspondences.

Twists use [v, w]. H transports local relation increments by conjugation;
Gamma transports their generator coordinates. No target law fitting is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral

import numpy as np

from ._validation import finite_matrix, finite_vector, se3_transform
from .profile6 import ResponseProfile6D

_METRIC_SCALE = (1.0, 1.0, 1.0, 0.03, 0.03, 0.03)


def _scalar(value, name: str) -> float:
    if np.ndim(value) != 0 or isinstance(value, (bool, np.bool_)) or not np.isrealobj(value):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value, n: int, name: str) -> np.ndarray:
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be real")
    return finite_vector(value, n, name)


def _basis(value, name: str) -> np.ndarray:
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be real")
    result = finite_matrix(value, (6, 6), name)
    if np.linalg.matrix_rank(result) != 6:
        raise ValueError(f"{name} must be numerically full-rank (6, 6)")
    return result


def _matrix_tuple(value) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in row) for row in value)


def _indices(value, n: int, name: str) -> tuple[int, ...]:
    values = tuple(value)
    if len(values) != n or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in values):
        raise ValueError(f"{name} must contain {n} integer indices")
    result = tuple(int(x) for x in values)
    if sorted(result) != list(range(n)):
        raise ValueError(f"{name} must be a bijection of 0..{n - 1}")
    return result


def _ids(value, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"{name} must be a sequence of semantic IDs")
    result = tuple(value)
    if not result or any(not isinstance(x, str) or not x.strip() for x in result):
        raise ValueError(f"{name} must contain nonempty string IDs")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique semantic IDs")
    return result


def se3_adjoint(H) -> np.ndarray:
    """Return Ad_H for [v,w] and Exp(Ad_H xi) = H Exp(xi) H^-1."""
    if not np.isrealobj(H):
        raise ValueError("H must be real")
    h = se3_transform(H, "H")
    r = h[:3, :3]
    x, y, z = h[:3, 3]
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.block([[r, cross @ r], [np.zeros((3, 3)), r]])


def physical_response_operator(basis, alpha) -> np.ndarray:
    """Return B diag(alpha) B^-1 using a solve; only full-rank 6D bases."""
    b = _basis(basis, "basis")
    a = _vector(alpha, 6, "alpha")
    return finite_matrix(np.linalg.solve(b.T, (b * a).T).T, (6, 6), "physical response")


@dataclass(frozen=True)
class MonomialCorrespondence:
    """c_target[i] = scales[i] * c_source[permutation[i]]."""

    permutation: tuple[int, ...]
    scales: tuple[float, ...]

    def __post_init__(self):
        p = _indices(self.permutation, 6, "permutation")
        s = _vector(self.scales, 6, "scales")
        if np.any(s == 0):
            raise ValueError("scales must be nonzero")
        with np.errstate(over="ignore", divide="ignore"):
            if not np.all(np.isfinite(1 / s)):
                raise ValueError("scales must have finite reciprocals")
        object.__setattr__(self, "permutation", p)
        object.__setattr__(self, "scales", tuple(s))

    @property
    def matrix(self) -> np.ndarray:
        result = np.zeros((6, 6))
        result[np.arange(6), self.permutation] = self.scales
        return result

    @property
    def inverse(self) -> np.ndarray:
        result = np.zeros((6, 6))
        result[self.permutation, np.arange(6)] = 1 / np.asarray(self.scales)
        return result

    def source_to_target(self, coordinates) -> np.ndarray:
        source = _vector(coordinates, 6, "source coordinates")
        with np.errstate(over="ignore", invalid="ignore"):
            return _vector(source[list(self.permutation)] * self.scales, 6, "target coordinates")

    def target_to_source(self, coordinates) -> np.ndarray:
        target = _vector(coordinates, 6, "target coordinates")
        source = np.empty(6)
        with np.errstate(over="ignore", invalid="ignore"):
            source[list(self.permutation)] = target / self.scales
        return _vector(source, 6, "source coordinates")


@dataclass(frozen=True)
class GeneratorSemantics:
    """Provided functional IDs, independent of world-axis names."""

    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("source_ids", "target_ids"):
            ids = _ids(getattr(self, name), name)
            if len(ids) != 6:
                raise ValueError(f"{name} must have six IDs")
            object.__setattr__(self, name, ids)

    def matches(self, correspondence: MonomialCorrespondence) -> bool:
        return self.target_ids == tuple(self.source_ids[j] for j in correspondence.permutation)


def transport_diagonal(alpha_source, correspondence: MonomialCorrespondence) -> np.ndarray:
    """Algebraic reordering only; use MatchedRelationTransfer.query for gated transfer."""
    if not isinstance(correspondence, MonomialCorrespondence):
        raise TypeError("correspondence must be MonomialCorrespondence")
    return _vector(alpha_source, 6, "alpha_source")[list(correspondence.permutation)]


@dataclass(frozen=True)
class PiecewiseLinearPhaseMap:
    """Provided target-to-source phase warp on equally sized ordered chains."""

    source_semantic_ids: tuple[str, ...]
    target_semantic_ids: tuple[str, ...]
    source_phase_for_target: tuple[int, ...]
    target_knots: tuple[tuple[float, ...], ...]
    source_knots: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        src = _ids(self.source_semantic_ids, "source phase IDs")
        tgt = _ids(self.target_semantic_ids, "target phase IDs")
        n = len(src)
        if len(tgt) != n:
            raise ValueError("source/target phase counts must agree")
        p = _indices(self.source_phase_for_target, n, "phase map")
        if p != tuple(range(n)):
            raise ValueError("phase map must preserve chain order")
        if len(self.target_knots) != n or len(self.source_knots) != n:
            raise ValueError("one knot pair is required per target phase")
        target, source = [], []
        for t, s in zip(self.target_knots, self.source_knots):
            if len(t) < 2 or len(t) != len(s):
                raise ValueError("paired knot arrays must have equal length >= 2")
            t, s = _vector(t, len(t), "target knots"), _vector(s, len(s), "source knots")
            for knots in (t, s):
                if knots[0] != 0 or knots[-1] != 1 or np.any(np.diff(knots) <= 0):
                    raise ValueError("knots must increase strictly from 0 to 1")
            target.append(tuple(t))
            source.append(tuple(s))
        for name, value in (
            ("source_semantic_ids", src),
            ("target_semantic_ids", tgt),
            ("source_phase_for_target", p),
            ("target_knots", tuple(target)),
            ("source_knots", tuple(source)),
        ):
            object.__setattr__(self, name, value)

    def semantics_match(self) -> bool:
        return self.target_semantic_ids == tuple(self.source_semantic_ids[j] for j in self.source_phase_for_target)

    def map(self, target_phase, target_progress) -> tuple[int, float]:
        if (
            isinstance(target_phase, (bool, np.bool_))
            or not isinstance(target_phase, Integral)
            or not 0 <= target_phase < len(self.target_semantic_ids)
        ):
            raise ValueError("target phase must be an integer phase index in range")
        rho = _scalar(target_progress, "target progress")
        if not 0 <= rho <= 1:
            raise ValueError("target progress must be in [0,1]")
        q = int(target_phase)
        return self.source_phase_for_target[q], float(np.interp(rho, self.target_knots[q], self.source_knots[q]))


@dataclass(frozen=True)
class CoordinateRangeSupport:
    """Closed source-coordinate box; does not certify joint distribution support."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    tolerance: float

    def __post_init__(self):
        lower, upper = _vector(self.lower, 6, "lower"), _vector(self.upper, 6, "upper")
        tol = _scalar(self.tolerance, "support tolerance")
        if np.any(lower > upper) or tol < 0:
            raise ValueError("support requires lower <= upper and tolerance >= 0")
        object.__setattr__(self, "lower", tuple(lower))
        object.__setattr__(self, "upper", tuple(upper))
        object.__setattr__(self, "tolerance", tol)

    def contains_source(self, coordinates) -> bool:
        c = _vector(coordinates, 6, "source coordinates")
        with np.errstate(over="ignore"):
            return bool(
                np.all(c >= np.asarray(self.lower) - self.tolerance)
                and np.all(c <= np.asarray(self.upper) + self.tolerance)
            )


def compatibility_residual(
    source_basis, target_basis, H, correspondence, coordinate_scales
) -> tuple[float, np.ndarray]:
    """Dimensionless total and source-axis residuals in the fixed target twist metric."""
    bs, bt = _basis(source_basis, "source basis"), _basis(target_basis, "target basis")
    if not isinstance(correspondence, MonomialCorrespondence):
        raise TypeError("correspondence must be MonomialCorrespondence")
    sc = _vector(coordinate_scales, 6, "coordinate scales")
    if np.any(sc <= 0):
        raise ValueError("coordinate scales must be positive")
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            reference = se3_adjoint(H) @ bs
            error = (bt @ correspondence.matrix - reference) * sc * np.asarray(_METRIC_SCALE)[:, None]
            reference = reference * sc * np.asarray(_METRIC_SCALE)[:, None]
    except FloatingPointError as exc:
        raise ValueError("scaled basis residual is numerically unrepresentable") from exc
    finite_matrix(error, (6, 6), "scaled basis difference")
    finite_matrix(reference, (6, 6), "scaled reference basis")
    # hypot avoids squaring extremely large/small values. Underflow is a numeric error,
    # never evidence of matching; there is no artificial denominator epsilon.
    with np.errstate(over="ignore"):
        denominators = np.hypot.reduce(reference, axis=0)
        total_denominator = np.hypot.reduce(reference.ravel())
    if (
        np.any(denominators == 0)
        or not np.all(np.isfinite(denominators))
        or total_denominator == 0
        or not np.isfinite(total_denominator)
    ):
        raise ValueError("reference basis normalization is numerically unrepresentable")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="raise"):
            axes = np.hypot.reduce(error, axis=0) / denominators
            total = np.hypot.reduce(error.ravel()) / total_denominator
    except FloatingPointError as exc:
        raise ValueError("normalized basis residual is numerically unrepresentable") from exc
    return _scalar(total, "total residual"), _vector(axes, 6, "axis residuals")


class MatchStatus(Enum):
    MATCHED = "matched"
    UNMATCHED_FAMILY = "unmatched_family"
    INCOMPATIBLE_BASIS = "incompatible_basis"
    OUT_OF_SUPPORT = "out_of_support"


@dataclass(frozen=True)
class MatchResult:
    """Immutable diagnostic; coordinates are absent for a static contract check."""

    status: MatchStatus
    total_residual: float
    axis_residuals: tuple[float, ...]
    source_coordinates: tuple[float, ...] | None
    reason: str


@dataclass(frozen=True)
class TransferQueryResult:
    """A rejected query always has alpha_target=None."""

    status: MatchStatus
    alpha_target: tuple[float, ...] | None
    source_phase: int | None
    source_progress: float | None
    source_coordinates: tuple[float, ...] | None
    match: MatchResult


@dataclass(frozen=True)
class MatchedRelationTransfer:
    """Immutable transfer contract with explicit thresholds and guarded profile queries.

    source_basis/target_basis/frame_transform accept arrays and are stored as tuples.
    Coordinates passed to query are Log-based generator coordinates, not Euler pose6.
    """

    source_basis: tuple[tuple[float, ...], ...]
    target_basis: tuple[tuple[float, ...], ...]
    frame_transform: tuple[tuple[float, ...], ...]
    correspondence: MonomialCorrespondence
    semantics: GeneratorSemantics
    phase_map: PiecewiseLinearPhaseMap
    support: CoordinateRangeSupport
    coordinate_scales: tuple[float, ...]
    total_tolerance: float
    axis_tolerances: tuple[float, ...]
    _total: float = field(init=False, repr=False)
    _axes: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self):
        for name, cls in (
            ("correspondence", MonomialCorrespondence),
            ("semantics", GeneratorSemantics),
            ("phase_map", PiecewiseLinearPhaseMap),
            ("support", CoordinateRangeSupport),
        ):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be {cls.__name__}")
        total = _scalar(self.total_tolerance, "total tolerance")
        axes = _vector(self.axis_tolerances, 6, "axis tolerances")
        if total < 0 or np.any(axes < 0):
            raise ValueError("match tolerances must be nonnegative")
        # Validate H before coercion, including complex inputs.
        residual, axis_residuals = compatibility_residual(
            self.source_basis, self.target_basis, self.frame_transform, self.correspondence, self.coordinate_scales
        )
        for name in ("source_basis", "target_basis", "frame_transform"):
            object.__setattr__(self, name, _matrix_tuple(getattr(self, name)))
        object.__setattr__(self, "coordinate_scales", tuple(_vector(self.coordinate_scales, 6, "coordinate scales")))
        object.__setattr__(self, "total_tolerance", total)
        object.__setattr__(self, "axis_tolerances", tuple(axes))
        object.__setattr__(self, "_total", residual)
        object.__setattr__(self, "_axes", tuple(axis_residuals))

    def evaluate_contract(self) -> MatchResult:
        if not self.semantics.matches(self.correspondence) or not self.phase_map.semantics_match():
            status, reason = MatchStatus.UNMATCHED_FAMILY, "provided generator or phase semantics disagree"
        elif self._total > self.total_tolerance or np.any(np.asarray(self._axes) > self.axis_tolerances):
            status, reason = MatchStatus.INCOMPATIBLE_BASIS, "basis compatibility tolerance exceeded"
        else:
            status, reason = MatchStatus.MATCHED, "static contract passes; query support not checked"
        return MatchResult(status, self._total, self._axes, None, reason)

    def evaluate_query(self, target_coordinates) -> MatchResult:
        target = _vector(target_coordinates, 6, "target coordinates")
        result = self.evaluate_contract()
        if result.status is not MatchStatus.MATCHED:
            return result
        source = self.correspondence.target_to_source(target)
        ok = self.support.contains_source(source)
        return MatchResult(
            MatchStatus.MATCHED if ok else MatchStatus.OUT_OF_SUPPORT,
            self._total,
            self._axes,
            tuple(source),
            "contract and coordinate range pass" if ok else "mapped coordinates exceed source range support",
        )

    def source_phase(self, target_phase, target_progress) -> tuple[int, float]:
        return self.phase_map.map(target_phase, target_progress)

    def query(self, source_profile, target_phase, target_progress, target_coordinates) -> TransferQueryResult:
        if not isinstance(source_profile, ResponseProfile6D):
            raise TypeError("source_profile must be ResponseProfile6D")
        metadata = source_profile.to_dict()
        if metadata["n_phases"] != len(self.phase_map.source_semantic_ids):
            raise ValueError("profile phase count must equal source phase-map count")
        theta = source_profile.theta
        if theta.shape[1] == 0 or not np.all(np.isfinite(theta)):
            raise ValueError("profile theta must be nonempty and finite")
        if (
            _scalar(source_profile.alpha_max, "alpha_max") <= 0
            or _scalar(source_profile.basis_width, "basis_width") <= 0
        ):
            raise ValueError("profile alpha_max and basis_width must be positive")
        q, rho = self.source_phase(target_phase, target_progress)
        result = self.evaluate_query(target_coordinates)
        if result.status is not MatchStatus.MATCHED:
            return TransferQueryResult(result.status, None, None, None, result.source_coordinates, result)
        alpha = transport_diagonal(source_profile.alpha(q, rho), self.correspondence)
        return TransferQueryResult(result.status, tuple(alpha), q, rho, result.source_coordinates, result)
