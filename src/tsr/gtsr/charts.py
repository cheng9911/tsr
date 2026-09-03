# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generator charts for the G-TSR geometry kernel."""

from abc import ABC, abstractmethod

import numpy as np
from numpy import pi

from ..utils import EPSILON, wrap_to_interval
from ._validation import finite_matrix, finite_vector, rotation_matrix, se3_transform
from .domains import CoordinateDomain
from .metrics import se3_adjoint, se3_exp

_CANONICAL_AXES = np.array(
    [
        [1, 0, 0, 0, 0, 0],  # x
        [0, 1, 0, 0, 0, 0],  # y
        [0, 0, 1, 0, 0, 0],  # z
        [0, 0, 0, 1, 0, 0],  # roll (Rx)
        [0, 0, 0, 0, 1, 0],  # pitch (Ry)
        [0, 0, 0, 0, 0, 1],  # yaw (Rz)
    ]
)
_CANONICAL_ORDER = (0, 1, 2, 5, 4, 3)  # forward 顺序：Trans, Rz, Ry, Rx


def _body_jacobian(axes, order, representation, M, z):
    """Body-frame Jacobian (6, m); column i = (T(z)^-1 dT(z)/dz_i)^vee.

    SPACE: J_i = Ad_{(Exp(S_i z_i) ... Exp(S_m z_m) M)^-1}(S_i)
    BODY:  J_i = Ad_{(Exp(B_i z_i) ... Exp(B_m z_m))^-1}(B_i)
    """
    m = len(order)
    J = np.zeros((6, m))
    right = M.copy() if representation == "space" else np.eye(4)
    for k in reversed(range(m)):
        idx = order[k]
        right = se3_exp(axes[idx] * z[idx]) @ right
        J[:, idx] = se3_adjoint(np.linalg.inv(right)) @ axes[idx]
    return J


class GeneratorChart(ABC):
    """Base protocol: owns coordinate topology, maps coordinates to local poses."""

    coord_names: tuple = ()
    periodic: tuple = ()
    periods: tuple = ()

    @property
    def dim(self):
        return len(self.coord_names)

    @abstractmethod
    def forward(self, z):
        """Map a coordinate vector ``z`` to a 4x4 local transform."""


class CanonicalTSRChart(GeneratorChart):
    """The canonical chart: ``Trans(x,y,z) Rz(yaw) Ry(pitch) Rx(roll)``."""

    coord_names = ("x", "y", "z", "roll", "pitch", "yaw")
    periodic = (False, False, False, True, True, True)
    periods = (None, None, None, 2 * pi, 2 * pi, 2 * pi)

    def forward(self, z):
        return self.xyzrpy_to_trans(z)

    def jacobian(self, z):
        z = finite_vector(z, self.dim, "z")
        return _body_jacobian(_CANONICAL_AXES, _CANONICAL_ORDER, "space", np.eye(4), z)

    @staticmethod
    def xyzrpy_to_trans(xyzrpy):
        xyzrpy = finite_vector(xyzrpy, 6, "xyzrpy")
        trans = np.zeros((4, 4))
        trans[3, 3] = 1.0
        trans[0:3, 3] = xyzrpy[0:3]
        trans[0:3, 0:3] = CanonicalTSRChart.rpy_to_rot(xyzrpy[3:6])
        return trans

    @staticmethod
    def rpy_to_rot(rpy):
        rpy = finite_vector(rpy, 3, "rpy")
        rot = np.zeros((3, 3))
        r, p, y = rpy[0], rpy[1], rpy[2]
        rot[0, 0] = np.cos(p) * np.cos(y)
        rot[1, 0] = np.cos(p) * np.sin(y)
        rot[2, 0] = -np.sin(p)
        rot[0, 1] = np.sin(r) * np.sin(p) * np.cos(y) - np.cos(r) * np.sin(y)
        rot[1, 1] = np.sin(r) * np.sin(p) * np.sin(y) + np.cos(r) * np.cos(y)
        rot[2, 1] = np.sin(r) * np.cos(p)
        rot[0, 2] = np.cos(r) * np.sin(p) * np.cos(y) + np.sin(r) * np.sin(y)
        rot[1, 2] = np.cos(r) * np.sin(p) * np.sin(y) - np.sin(r) * np.cos(y)
        rot[2, 2] = np.cos(r) * np.cos(p)
        return rot

    def coordinates(self, T_local, domain):
        """Return pose-accurate coordinates for a local transform.

        Euler coordinates are not unique.  This method chooses one canonical
        solution, but guarantees pose reconstruction even at and arbitrarily
        close to gimbal lock.  Legacy EPSILON-tolerant membership remains in
        the ``*_within_bounds`` helpers and is intentionally not used here.
        """
        T_local = se3_transform(T_local, "T_local")
        if not isinstance(domain, CoordinateDomain):
            raise TypeError("domain must be a CoordinateDomain")
        if domain.dim != self.dim:
            raise ValueError(f"chart/domain dimension mismatch: {self.dim} != {domain.dim}")
        if any(not domain.intervals[i].periodic for i in range(3, 6)):
            raise ValueError("canonical rotation coordinates require circular domain intervals")
        xyz = T_local[0:3, 3]
        rot = T_local[0:3, 0:3]
        for rpy in self._accurate_rpy_candidates(rot, domain):
            candidate = np.hstack((xyz, rpy))
            if all(domain.contains(candidate)) and self._rotation_error(rot, self.rpy_to_rot(rpy)) <= 1e-10:
                return candidate
        return np.hstack((xyz, self.accurate_rot_to_rpy(rot)))

    @staticmethod
    def accurate_rot_to_rpy(rot):
        """Invert ``Rz(yaw) Ry(pitch) Rx(roll)`` without legacy snapping."""
        rot = rotation_matrix(rot)
        cos_pitch = np.hypot(rot[0, 0], rot[1, 0])
        singular_threshold = 4.0 * np.finfo(float).eps
        if cos_pitch > singular_threshold:
            roll = np.arctan2(rot[2, 1], rot[2, 2])
            pitch = np.arctan2(-rot[2, 0], cos_pitch)
            yaw = np.arctan2(rot[1, 0], rot[0, 0])
        elif rot[2, 0] < 0.0:
            roll = np.arctan2(rot[0, 1], rot[0, 2])
            pitch = pi / 2
            yaw = 0.0
        else:
            roll = np.arctan2(-rot[0, 1], -rot[0, 2])
            pitch = -pi / 2
            yaw = 0.0
        return np.array([roll, pitch, yaw])

    @classmethod
    def _accurate_rpy_candidates(cls, rot, domain):
        """Return reconstructing Euler solutions, with domain-feasible ones first."""
        primary = cls.accurate_rot_to_rpy(rot)
        roll, pitch, yaw = primary
        cos_pitch = np.hypot(rot[0, 0], rot[1, 0])
        singular_threshold = 4.0 * np.finfo(float).eps
        if cos_pitch > singular_threshold:
            alternate = np.array([roll + pi, pi - pitch, yaw + pi])
            return [primary, cls._wrap_rpy(alternate)]

        roll_interval = domain.intervals[3]
        yaw_interval = domain.intervals[5]
        roll_values = [
            roll_interval.lo_cont,
            (roll_interval.lo_cont + roll_interval.hi_cont) / 2.0,
            roll_interval.hi_cont,
        ]
        yaw_values = [
            yaw_interval.lo_cont,
            (yaw_interval.lo_cont + yaw_interval.hi_cont) / 2.0,
            yaw_interval.hi_cont,
        ]
        if pitch > 0.0:
            offset = np.arctan2(rot[0, 1], rot[0, 2])
            singular = [np.array([value + offset, pitch, value]) for value in yaw_values]
            singular.extend(np.array([value, pitch, value - offset]) for value in roll_values)
        else:
            offset = np.arctan2(-rot[0, 1], -rot[0, 2])
            singular = [np.array([offset - value, pitch, value]) for value in yaw_values]
            singular.extend(np.array([value, pitch, offset - value]) for value in roll_values)
        singular.extend(np.array([r, pitch, y]) for r in roll_values for y in yaw_values)
        return singular + [primary]

    @staticmethod
    def _wrap_rpy(rpy):
        return (np.asarray(rpy) + pi) % (2 * pi) - pi

    @staticmethod
    def _rotation_error(first, second):
        delta = first.T @ second
        skew = delta - delta.T
        vee = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        return np.arctan2(0.5 * np.linalg.norm(vee), 0.5 * (np.trace(delta) - 1.0))

    @staticmethod
    def rot_to_rpy(rot):
        """Legacy inverse used by compatibility distance calculations."""
        rot = rotation_matrix(rot)
        rpy = np.zeros(3)
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            p = -np.arcsin(rot[2, 0])
            rpy[0] = np.arctan2((rot[2, 1] / np.cos(p)), (rot[2, 2] / np.cos(p)))
            rpy[1] = p
            rpy[2] = np.arctan2((rot[1, 0] / np.cos(p)), (rot[0, 0] / np.cos(p)))
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = np.arctan2(rot[0, 1], rot[0, 2])
                rpy[0] = r_offset
                rpy[1] = pi / 2
                rpy[2] = 0.0
            else:
                r_offset = np.arctan2(-rot[0, 1], -rot[0, 2])
                rpy[0] = r_offset
                rpy[1] = -pi / 2
                rpy[2] = 0.0
        return rpy

    @staticmethod
    def xyz_within_bounds(xyz, Bw):
        xyz = finite_vector(xyz, 3, "xyz")
        Bw = finite_matrix(Bw, (3, 2), "Bw")
        xyzcheck = []
        for i, x in enumerate(xyz):
            x_val = x.item() if hasattr(x, "item") else float(x)
            xyzcheck.append(((x_val + EPSILON) >= Bw[i, 0]) and ((x_val - EPSILON) <= Bw[i, 1]))
        return xyzcheck

    @staticmethod
    def rpy_within_bounds(rpy, Bw):
        rpy = finite_vector(rpy, 3, "rpy")
        Bw = finite_matrix(Bw, (3, 2), "Bw")
        rpy = wrap_to_interval(rpy, lower=Bw[:3, 0])
        rpycheck = [False] * 3
        for i in range(3):
            if Bw[i, 0] > Bw[i, 1] + EPSILON:
                rpycheck[i] = ((rpy[i] + EPSILON) >= Bw[i, 0]) or ((rpy[i] - EPSILON) <= Bw[i, 1])
            else:
                rpycheck[i] = ((rpy[i] + EPSILON) >= Bw[i, 0]) and ((rpy[i] - EPSILON) <= Bw[i, 1])
        return rpycheck

    @staticmethod
    def rot_within_rpy_bounds(rot, Bw):
        rot = rotation_matrix(rot)
        Bw = finite_matrix(Bw, (3, 2), "Bw")
        if not (abs(abs(rot[2, 0]) - 1) < EPSILON):
            psol = -np.arcsin(rot[2, 0])
            for p in [psol, (pi - psol)]:
                rpy = np.zeros(3)
                rpy[0] = np.arctan2((rot[2, 1] / np.cos(p)), (rot[2, 2] / np.cos(p)))
                rpy[1] = p
                rpy[2] = np.arctan2((rot[1, 0] / np.cos(p)), (rot[0, 0] / np.cos(p)))
                rpycheck = CanonicalTSRChart.rpy_within_bounds(rpy, Bw)
                if all(rpycheck):
                    return rpycheck, rpy
            return rpycheck, None
        else:
            if abs(rot[2, 0] + 1) < EPSILON:
                r_offset = np.arctan2(rot[0, 1], rot[0, 2])
                rpy_list = [
                    np.array([Bw[2, 0] + r_offset, pi / 2, Bw[2, 0]]),
                    np.array([Bw[2, 1] + r_offset, pi / 2, Bw[2, 1]]),
                    np.array([Bw[0, 0], pi / 2, Bw[0, 0] - r_offset]),
                    np.array([Bw[0, 1], pi / 2, Bw[0, 1] - r_offset]),
                ]
                for rpy in rpy_list:
                    rpycheck = CanonicalTSRChart.rpy_within_bounds(rpy, Bw)
                    if rpycheck[1] is False:
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
            else:
                r_offset = np.arctan2(-rot[0, 1], -rot[0, 2])
                rpy_list = [
                    np.array([-Bw[2, 0] + r_offset, -pi / 2, Bw[2, 0]]),
                    np.array([-Bw[2, 1] + r_offset, -pi / 2, Bw[2, 1]]),
                    np.array([Bw[0, 0], -pi / 2, -Bw[0, 0] + r_offset]),
                    np.array([Bw[0, 1], -pi / 2, -Bw[0, 1] + r_offset]),
                ]
                for rpy in rpy_list:
                    rpycheck = CanonicalTSRChart.rpy_within_bounds(rpy, Bw)
                    if rpycheck[1] is False:
                        return rpycheck, None
                    if all(rpycheck):
                        return rpycheck, rpy
        return rpycheck, None


class PoEChart(GeneratorChart):
    """Product-of-Exponentials chart.

    ``representation="space"`` -> ``Phi(z) = Exp(S1 z1) ... Exp(Sm zm) M``.
    ``representation="body"``  -> ``Phi(z) = M Exp(B1 z1) ... Exp(Bm zm)``.
    ``order`` is an explicit index permutation (default ``0..m-1``); ``M`` is the
    home transform (Sprint B restricts to ``M=I`` but the interface is kept).
    """

    _PITCH_TOL = 1e-12
    _ANGULAR_NORM_RTOL = 1e-10

    def __init__(self, axes, representation, order=None, M=None):
        axes = np.asarray(axes, dtype=float)
        if axes.ndim != 2 or axes.shape[1] != 6 or axes.shape[0] < 1:
            raise ValueError("axes must be a non-empty (m, 6) array of screw axes")
        axes = finite_matrix(axes, axes.shape, "axes")
        if representation not in ("space", "body"):
            raise ValueError("representation must be 'space' or 'body'")
        m = axes.shape[0]
        if order is None:
            order = list(range(m))
        order = [int(i) for i in order]
        if sorted(order) != list(range(m)):
            raise ValueError("order must be a permutation of 0..m-1")
        if M is None:
            M = np.eye(4)
        M = se3_transform(M, "M")
        periodic = []
        for xi in axes:
            v, w = xi[0:3], xi[3:6]
            th = float(np.linalg.norm(w))
            if th <= self._PITCH_TOL:
                periodic.append(False)  # pure translation
            else:
                if not np.isclose(th, 1.0, rtol=self._ANGULAR_NORM_RTOL, atol=self._PITCH_TOL):
                    raise ValueError(f"rotational screw axes must have a unit angular part (||omega|| ~= 1), got {th}")
                pitch = float(np.dot(w, v)) / (th * th)
                periodic.append(abs(pitch) <= self._PITCH_TOL)
        periods = tuple(2 * pi if p else None for p in periodic)
        self.coord_names = tuple(f"q{i}" for i in range(m))
        self.periodic = tuple(periodic)
        self.periods = periods
        self._axes = axes
        self._representation = representation
        self._order = order
        self._M = M

    def forward(self, z):
        z = finite_vector(z, self.dim, "z")
        T = self._M.copy()
        if self._representation == "space":
            for idx in reversed(self._order):
                T = se3_exp(self._axes[idx] * z[idx]) @ T
        else:
            for idx in self._order:
                T = T @ se3_exp(self._axes[idx] * z[idx])
        return T

    def jacobian(self, z):
        z = finite_vector(z, self.dim, "z")
        return _body_jacobian(self._axes, self._order, self._representation, self._M, z)
