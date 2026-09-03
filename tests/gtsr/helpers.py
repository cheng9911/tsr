# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

import numpy as np
from numpy import pi

SEEDS = [0, 1, 7, 42, 20260902]
DIFF_SAMPLES_PER_SEED = 1000


def rotation_error(R_old, R_new):
    """Stable rotation error near zero (atan2 form), per §6.2."""
    R_delta = R_old.T @ R_new
    S = R_delta - R_delta.T
    vee = np.array([S[2, 1], S[0, 2], S[1, 0]])
    return np.arctan2(0.5 * np.linalg.norm(vee), 0.5 * (np.trace(R_delta) - 1.0))


def transform_error(T_old, T_new):
    e_t = np.linalg.norm(T_old[0:3, 3] - T_new[0:3, 3])
    e_R = rotation_error(T_old[0:3, 0:3], T_new[0:3, 0:3])
    return e_t, e_R


def assert_transforms_close(T_old, T_new, tol=1e-10):
    e_t, e_R = transform_error(T_old, T_new)
    assert max(e_t, e_R) <= tol, f"pose mismatch: e_t={e_t:.3e}, e_R={e_R:.3e}"


def random_rotation(rng):
    """Uniformly random SO(3) via SVD (never an arbitrary 4x4)."""
    A = rng.standard_normal((3, 3))
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def random_se3(rng):
    T = np.eye(4)
    T[0:3, 0:3] = random_rotation(rng)
    T[0:3, 3] = rng.uniform(-2.0, 2.0, size=3)
    return T


def random_bounds(rng):
    """Valid (6,2) bounds: translation lo<=hi; rotation may be inner/outer/full."""
    Bw = np.zeros((6, 2))
    Bw[0:3, 0] = rng.uniform(-1.0, 1.0, size=3)
    Bw[0:3, 1] = Bw[0:3, 0] + rng.uniform(0.0, 0.5, size=3)
    for i in range(3, 6):
        lo = rng.uniform(-pi, pi)
        width = rng.uniform(0.0, 2.0 * pi)
        Bw[i, 0] = lo
        Bw[i, 1] = lo + width
    return Bw
