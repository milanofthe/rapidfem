# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Dielectric resonator — a high-εr fill loads a metal cavity and lowers its
resonance.

A dielectric resonator (DR) is a puck of high-permittivity ceramic that stores
the field and resonates well below the empty-cavity value, the hallmark being
f_res ∝ 1/√εr. A partially-filled DR-in-cavity has no closed form, so this test
anchors on the rigorous, fully-solvable limit:

    A rectangular PEC cavity a×b×d FILLED with εr has exact resonances

        f_mnp = c/(2√εr) · sqrt((m/a)² + (n/b)² + (p/d)²)     (≥2 non-zero idx)

i.e. the empty-cavity spectrum scaled by 1/√εr (``ref.rect_cavity_freqs`` with
``er``). This pins the one thing a DR test must check: that εr enters the *mass*
term of the eigenproblem, so the resonance drops by exactly 1/√εr. Phenomena
exercised: dielectric loading, εr in the eigen mass term, monotonic 1/√εr shift.

Reference: Pozar, *Microwave Engineering*, §6.5 (dielectric resonators); §6.3
(rectangular cavity, scaled by the medium velocity c/√εr).

Note on the eigensolver
-----------------------
As documented in ``test_rect_cavity_modes``, the shift-invert Lanczos driver
returns eigenpairs nearest the shift and also surfaces non-physical single-axis
variations mid-band. The gauge-free check is to drive it the way it is meant to
be driven: place the shift near each analytic resonance and confirm the solver
locks onto *that* resonance. We reuse that approach here.
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# Distinct a,b,d → non-degenerate fundamental (TE101). εr = 37 is a typical
# DR ceramic (e.g. (Zr,Sn)TiO4). Both the loaded resonance and the in-medium
# wavelength scale as 1/√εr, so lambda_maxh(f0, er_max=εr) reproduces the empty
# cavity's mesh and the DOF count stays at the exemplar's budget.
A, B, D = 40e-3, 20e-3, 30e-3
ER = 37.0


def _filled_cavity_modes(er, target_frequency, n_modes):
    """Fresh PEC cavity fully filled with `er`, meshed + solved near the shift.

    A new geometry per call keeps each `case.eigenmodes` (which meshes `g`)
    independent. Returns `(prob, modes)`.
    """
    f0 = ref.rect_cavity_freqs(A, B, D, er=er, n_modes=1)[0]
    g = case.geometry(maxh=rf.lambda_maxh(f_max=f0, er_max=er))
    diel = g.box(A, B, D, position=(-A / 2, -B / 2, 0.0),
                 material=rf.Dielectric(er=er, tand=0.0))
    rf.PEC(*diel.faces.unassigned)  # closed cavity: every wall metal, no ports
    return case.eigenmodes(g, target_frequency=target_frequency, n_modes=n_modes)


def _best_rel_err(modes, f_analytic):
    """Smallest relative error of any computed mode against `f_analytic`."""
    return min(abs(m.frequency_hz - f_analytic) / f_analytic for m in modes)


def _lowest_physical(modes, f0):
    """Computed mode nearest f0 among those above half the fundamental (drops
    static / near-zero spurious modes)."""
    phys = [m for m in modes if m.frequency_hz > 0.5 * f0]
    assert phys, [m.frequency_hz / 1e9 for m in modes]
    return min(phys, key=lambda m: abs(m.frequency_hz - f0))


@pytest.mark.slow
@case.phenomenon
def test_dielectric_filled_cavity_resonance():
    """Rigorous analytic anchor: εr-filled cavity resonates at f_empty/√εr."""
    f0 = ref.rect_cavity_freqs(A, B, D, er=ER, n_modes=1)[0]
    f0_empty = ref.rect_cavity_freqs(A, B, D, er=1.0, n_modes=1)[0]

    # Shift at the loaded fundamental, recover TE101 within ~3 %.
    prob, modes = _filled_cavity_modes(ER, f0, n_modes=6)
    assert prob.n_dofs < case.DOF_BUDGET, prob.n_dofs

    err0 = _best_rel_err([m for m in modes if m.frequency_hz > 0.5 * f0], f0)
    assert err0 < 0.03, (
        f"loaded TE101: best computed match off by {err0 * 100:.2f}% "
        f"(analytic {f0 / 1e9:.4f} GHz; computed "
        f"{[round(m.frequency_hz / 1e9, 4) for m in modes]} GHz)"
    )

    # Physics sanity: the εr=37 fill drops the resonance to f_empty/√εr — well
    # below the empty cavity (here ≈6.16× lower).
    f_comp = _lowest_physical(modes, f0).frequency_hz
    assert f_comp < 0.5 * f0_empty, (
        f"loaded {f_comp / 1e9:.4f} GHz not well below empty "
        f"{f0_empty / 1e9:.4f} GHz"
    )
    assert abs(f_comp - f0_empty / np.sqrt(ER)) / f0 < 0.03


@pytest.mark.slow
@case.phenomenon
def test_dielectric_loading_monotonic_in_er():
    """Higher εr → lower resonance, following 1/√εr (the DR loading law).

    Solve the same cavity at two permittivities and confirm the resonance both
    decreases monotonically and tracks the analytic 1/√εr scaling. This locks
    εr into the eigen mass term across more than one value, not just εr=37.
    """
    f_res = {}
    for er in (12.0, ER):  # e.g. alumina-grade vs high-εr DR ceramic
        f0 = ref.rect_cavity_freqs(A, B, D, er=er, n_modes=1)[0]
        prob, modes = _filled_cavity_modes(er, f0, n_modes=6)
        assert prob.n_dofs < case.DOF_BUDGET, (er, prob.n_dofs)
        f_res[er] = _lowest_physical(modes, f0).frequency_hz

    # Monotonic loading: more dielectric stores more field → lower frequency.
    assert f_res[ER] < f_res[12.0], f_res

    # 1/√εr scaling between the two solves (gauge-free ratio, εr only).
    ratio = f_res[12.0] / f_res[ER]
    expected = np.sqrt(ER / 12.0)
    assert abs(ratio - expected) / expected < 0.03, (
        f"f(εr=12)/f(εr=37) = {ratio:.4f}, expected √(37/12) = {expected:.4f}"
    )
