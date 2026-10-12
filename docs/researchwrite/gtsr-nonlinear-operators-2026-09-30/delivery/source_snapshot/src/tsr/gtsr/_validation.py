# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Shared validation helpers for the Sprint A geometry surface."""

import numpy as np


def finite_vector(value, length, name, *, allow_nan=False):
    """Return a defensive float copy after validating a public vector input."""
    try:
        vector = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an array-like of real numbers") from exc
    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {vector.shape}")
    valid = ~np.isinf(vector) if allow_nan else np.isfinite(vector)
    if not np.all(valid):
        qualifier = "finite or NaN" if allow_nan else "finite"
        raise ValueError(f"{name} values must be {qualifier}")
    return vector


def finite_matrix(value, shape, name):
    """Return a defensive float copy after validating a public matrix input."""
    try:
        matrix = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an array-like of real numbers") from exc
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} values must be finite")
    return matrix


def rotation_matrix(value, name="rotation"):
    """Validate and defensively copy an SO(3) matrix."""
    rotation = finite_matrix(value, (3, 3), name)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must have determinant +1")
    return rotation


def se3_transform(value, name="transform"):
    """Validate and defensively copy an SE(3) homogeneous transform."""
    transform = finite_matrix(value, (4, 4), name)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12, rtol=0.0):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    rotation_matrix(transform[:3, :3], f"{name} rotation")
    return transform
