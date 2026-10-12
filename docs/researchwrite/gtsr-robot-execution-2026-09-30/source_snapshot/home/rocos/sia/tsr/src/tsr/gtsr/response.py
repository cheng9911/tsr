# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Oracle-phase response operator: a geometry-only relation -> region adapter.

Sprint D assembles a response-conditioned :class:`~tsr.gtsr.region.GTSR` from a
relation pose and an explicit finite ``alpha`` vector. It does not fit ``alpha``
(Sprint E) and does not load ManiSkill / HDF5 / learned models.
"""

import numpy as np

from ._validation import finite_matrix, finite_vector, se3_transform
from .metrics import se3_exp, se3_log
from .region import GTSR


class ResponseOperator:
    """Map a relation pose to a world-frame left action and a conditioned region.

    The finite action is ::

        A(alpha, T) = T_ref Exp(B_r @ diag(alpha) @ B_r^-1 Log(T_ref^-1 T)) T_ref^-1,

    with ``reference = T_ref`` a reference SE(3) transform and ``basis = B_r`` a
    full-rank 6x6 twist-coordinate basis (twist order ``[v, w]``). ``alpha`` is an
    explicit length-6 vector supplied by the caller (an oracle or a fitted profile);
    this class does not assume ``alpha in [0, 1]``.
    """

    def __init__(self, reference, basis):
        self._reference = se3_transform(reference, "reference")
        basis = finite_matrix(basis, (6, 6), "basis")
        if np.linalg.matrix_rank(basis) != 6:
            raise ValueError("basis must be a full-rank (invertible) 6x6 matrix")
        self._basis = basis

    @property
    def dim(self):
        """Dimension of the relation coordinate / alpha vector (always 6)."""
        return 6

    def coordinates(self, relation):
        """Return ``c = B_r^-1 Log(T_ref^-1 T)``, the local relation coordinates."""
        relation = se3_transform(relation, "relation")
        return np.linalg.solve(self._basis, se3_log(np.linalg.inv(self._reference) @ relation))

    def action(self, relation, alpha):
        """Return the finite world-frame left action ``A(alpha, T)`` as a 4x4 transform."""
        alpha = finite_vector(alpha, 6, "alpha")
        xi = self._basis @ (alpha * self.coordinates(relation))
        return self._reference @ se3_exp(xi) @ np.linalg.inv(self._reference)

    def condition(self, region, relation, alpha):
        """Return a new response-conditioned region without mutating ``region``.

        The returned region shares ``region``'s chart and domain, and its ``T0_w`` is
        left-multiplied by the action; ``Tw_e`` is preserved.
        """
        if not isinstance(region, GTSR):
            raise TypeError("region must be a GTSR")
        return GTSR(
            T0_w=self.action(relation, alpha) @ region.T0_w,
            Tw_e=region.Tw_e,
            chart=region.chart,
            domain=region.domain,
        )
