# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Build → mesh → solve helpers with a hard DOF budget for phenomenon tests.

Every geometry test follows the same shape: build a parametric geometry, attach
physics, solve, and assert an extracted quantity against `harness.references`.
These helpers carry the shared plumbing so a test module stays a geometry plus
its physics assertion. The DOF budget (< 100 000) is enforced on every solve so
the suite always runs on a laptop.

Usage (in tests/geometries/test_<name>.py):

    import numpy as np
    import rapidfem as rf
    from harness import case, references as ref

    @case.phenomenon
    def test_my_structure():
        g = case.geometry(maxh=rf.lambda_maxh(f_max=12e9))
        air = g.box(...); rf.RectWaveguidePort(...); rf.PEC(...)
        prob, res = case.sweep(g, np.linspace(8e9, 12e9, 5))
        assert np.abs(res.sparams[:, 1, 0]).min() > 0.98
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

import rapidfem as rf

#: Maximum DOF count any test geometry may produce. Keeps the whole suite on a
#: 16 GB laptop and forces authors to mesh sensibly.
DOF_BUDGET = 100_000


class DofBudgetExceeded(AssertionError):
    """Raised when a meshed problem exceeds `DOF_BUDGET`."""


def phenomenon(fn):
    """Mark a test as a physics-phenomenon geometry test (pytest marker)."""
    return pytest.mark.phenomenon(fn)


def geometry(maxh, **kw) -> "rf.Geometry":
    """A fresh `rf.Geometry` with the given target mesh size. `maxh` may be a
    float (metres) or the result of `rf.lambda_maxh(...)`."""
    return rf.Geometry(maxh=maxh, **kw)


def _enforce_budget(prob) -> int:
    n = int(prob.n_dofs)
    if n >= DOF_BUDGET:
        raise DofBudgetExceeded(
            f"{n} DOF ≥ budget {DOF_BUDGET}; coarsen maxh "
            f"({prob.n_tets} tets)"
        )
    return n


def sweep(g, frequencies, *, z0: float = 50.0, dof_budget: int = DOF_BUDGET):
    """Mesh `g`, run an FD frequency sweep, enforce the DOF budget.

    Returns `(prob, result)` where `result.sparams` has shape
    `(n_freq, n_driven, n_driven)`. Do NOT call `g.mesh()` first — this does.
    """
    g.mesh()
    prob = rf.ProblemFD(g)
    result = prob.sweep(np.asarray(frequencies, dtype=float), z0=z0)
    n = _enforce_budget(prob)
    if n >= dof_budget:
        raise DofBudgetExceeded(f"{n} DOF ≥ requested budget {dof_budget}")
    return prob, result


def eigenmodes(g, target_frequency: float, *, n_modes: int = 6,
               dof_budget: int = DOF_BUDGET):
    """Mesh `g`, solve for `n_modes` eigenmodes near `target_frequency`.

    Returns `(prob, modes)` where each mode has `.frequency_hz` and `.q_factor`.
    """
    g.mesh()
    prob = rf.ProblemFD(g)
    modes = prob.eigenmode(target_frequency, n_modes=n_modes)
    n = _enforce_budget(prob)
    if n >= dof_budget:
        raise DofBudgetExceeded(f"{n} DOF ≥ requested budget {dof_budget}")
    return prob, modes


# ── small assertion helpers ────────────────────────────────────────────────
def rel_err(measured, expected) -> float:
    """Max relative error over array-likes (expected ≠ 0)."""
    measured = np.asarray(measured)
    expected = np.asarray(expected)
    return float(np.max(np.abs(measured - expected) / np.abs(expected)))


def phase_close(measured_rad, expected_rad, tol_deg: float = 5.0) -> bool:
    """Compare two phases modulo 2π within a degree tolerance."""
    d = np.angle(np.exp(1j * (np.asarray(measured_rad) - np.asarray(expected_rad))))
    return bool(np.all(np.abs(d) <= np.deg2rad(tol_deg)))


def passivity(sparams_at_freq) -> float:
    """Σ_j |S_ij|² for a driven row; ≤ 1 for a passive network (per incident
    port). Returns the worst-case (max over rows) row power."""
    s = np.asarray(sparams_at_freq)
    return float(np.max(np.sum(np.abs(s) ** 2, axis=1)))
