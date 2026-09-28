# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Quasi-static reference for a rectangular conductor in a rectangular PEC shield.

Solves the 2D magnetic diffusion problem of a straight line, independent of the
FEM code, on a graded tensor grid whose lines follow the conductor edges:

    -div grad A + j w mu0 sigma A = mu0 sigma E0      (A = 0 on the shield)

with a uniform applied field E0 along the line. The conductor current is
I = sum sigma (E0 - j w A) dA, so the per-unit-length series impedance is
Z' = R' + j w L' = E0 / I, internal and external inductance included. The shunt
capacitance C' comes from the electrostatic problem on the same grid (conductor
at 1 V, shield at 0 V, homogeneous medium). Then gamma = sqrt(Z' j w C').

Cell-centred finite volumes, so every cell is entirely metal or entirely
dielectric and the current integral is exact per cell. The grid is graded
geometrically towards every conductor edge, where the skin current crowds.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MU0 = 4e-7 * np.pi
EPS0 = 8.8541878128e-12


def _graded(a: float, b: float, h0: float, rate: float, hmax: float) -> np.ndarray:
    """Cell edges on [a, b], size h0 at both ends, growing by `rate` to `hmax`."""
    length = b - a
    left, s, h = [0.0], 0.0, h0
    while s + h < length / 2:
        s += h
        left.append(s)
        h = min(h * rate, hmax)
    left = np.array(left)
    edges = np.unique(np.concatenate([left, [length / 2], length - left[::-1]]))
    return a + edges


def _axis(breaks: list[float], h0: float, rate: float, hmax: float) -> np.ndarray:
    parts = [_graded(breaks[i], breaks[i + 1], h0, rate, hmax) for i in range(len(breaks) - 1)]
    return np.unique(np.concatenate(parts))


class ShieldedStrip:
    """Rectangular conductor `[x0, x0+w] x [y0, y0+t]` inside the PEC shield
    `[0, W] x [0, H]`, conductivity `sigma`, homogeneous dielectric `er`.

    `h0` is the cell size at the conductor edges, `rate` the geometric growth.
    """

    def __init__(self, *, W, H, x0, y0, w, t, sigma, er=1.0,
                 h0=None, rate=1.2, hmax=None):
        self.w, self.t, self.sigma, self.er = w, t, sigma, er
        h0 = h0 or min(w, t) / 200
        hmax = hmax or max(W, H) / 20
        self.xe = _axis([0.0, x0, x0 + w, W], h0, rate, hmax)
        self.ye = _axis([0.0, y0, y0 + t, H], h0, rate, hmax)
        xc = 0.5 * (self.xe[1:] + self.xe[:-1])
        yc = 0.5 * (self.ye[1:] + self.ye[:-1])
        self.dx, self.dy = np.diff(self.xe), np.diff(self.ye)
        self.metal = ((xc[:, None] > x0) & (xc[:, None] < x0 + w)
                      & (yc[None, :] > y0) & (yc[None, :] < y0 + t)).ravel()
        self.area = (self.dx[:, None] * self.dy[None, :]).ravel()
        self._lap = self._laplacian()

    def _laplacian(self) -> sp.csr_matrix:
        """-div grad with A = 0 on the shield (ghost at half a cell)."""
        nx, ny = len(self.dx), len(self.dy)
        idx = np.arange(nx * ny).reshape(nx, ny)
        rows, cols, vals = [], [], []
        diag = np.zeros(nx * ny)

        def couple(i, j, coef):
            rows.extend([i, j, i, j])
            cols.extend([i, j, j, i])
            vals.extend([coef, coef, -coef, -coef])

        # interior faces normal to x and y: conductance = face length / distance
        gx = self.dy[None, :] / (0.5 * (self.dx[1:] + self.dx[:-1]))[:, None]
        for a, b, c in zip(idx[:-1].ravel(), idx[1:].ravel(), gx.ravel()):
            couple(a, b, c)
        gy = self.dx[:, None] / (0.5 * (self.dy[1:] + self.dy[:-1]))[None, :]
        for a, b, c in zip(idx[:, :-1].ravel(), idx[:, 1:].ravel(), gy.ravel()):
            couple(a, b, c)
        # shield faces (Dirichlet 0 at the wall, half a cell away)
        diag[idx[0]] += self.dy / (0.5 * self.dx[0])
        diag[idx[-1]] += self.dy / (0.5 * self.dx[-1])
        diag[idx[:, 0]] += self.dx / (0.5 * self.dy[0])
        diag[idx[:, -1]] += self.dx / (0.5 * self.dy[-1])
        n = nx * ny
        return (sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(diag)).tocsr()

    def series_impedance(self, f: float) -> complex:
        """Z' = R' + j w L' in ohm/m at frequency `f` (f = 0 gives R'_dc)."""
        if f == 0.0:
            return 1.0 / (self.sigma * self.w * self.t)
        w = 2 * np.pi * f
        s = self.sigma * self.metal
        K = self._lap + sp.diags(1j * w * MU0 * s * self.area)
        A = spla.spsolve(K.tocsc(), MU0 * s * self.area * 1.0)
        current = np.sum(s * (1.0 - 1j * w * A) * self.area)
        return 1.0 / current

    def capacitance(self) -> float:
        """C' in F/m: electrostatics with the conductor cells held at 1 V."""
        free = ~self.metal
        K = self._lap
        Kff = K[free][:, free]
        rhs = -K[free][:, self.metal] @ np.ones(self.metal.sum())
        v = np.ones(len(self.metal))
        v[free] = spla.spsolve(Kff.tocsc(), rhs)
        charge = (K @ v)[self.metal].sum()
        return EPS0 * self.er * charge

    def gamma(self, f: float) -> complex:
        """Propagation constant alpha + j beta in 1/m."""
        w = 2 * np.pi * f
        return np.sqrt(self.series_impedance(f) * 1j * w * self.capacitance())
