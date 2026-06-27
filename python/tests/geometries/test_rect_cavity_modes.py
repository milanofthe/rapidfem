# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Closed rectangular PEC cavity — resonant eigenfrequencies.

A source-free, air-filled metal box has a discrete spectrum of bound
eigenmodes whose frequencies are known in closed form,

    f_mnp = c/2 · sqrt((m/a)² + (n/b)² + (p/d)²),   (≥2 non-zero indices)

The lowest is the TE101 mode. With no ports the cavity is lossless, so the
eigenfrequencies are (numerically) real and Q is infinite. Phenomena
exercised: bound resonant spectrum, mode ordering, degeneracy, PEC walls.

Reference: Pozar, *Microwave Engineering*, §6.3 (rectangular cavity).

Note on the eigensolver
-----------------------
The shift-invert Lanczos driver returns the eigenpairs *nearest the shift*.
Besides the physical resonances it also surfaces the non-physical
single-axis variations (the (m,0,0)/(0,n,0)/(0,0,p) "modes": ≈5.0 GHz for
(0,0,1), ≈7.5 GHz for (2,0,0)/(0,1,0)) which are NOT true cavity resonances
and which a blanket "below 0.5·f0" filter does not remove. The physically
meaningful, gauge-free check is therefore done the way a shift-invert solver
is meant to be driven: place the shift near each analytic resonance and
confirm the solver reproduces *that* resonance. A spurious shift would have
no eigenvalue to lock onto and the nearest returned mode would miss.
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# Air-filled box, all faces PEC. Distinct a,b,d give a non-degenerate
# fundamental (TE101 ≈ 6.25 GHz).
A, B, D = 40e-3, 20e-3, 30e-3


def _cavity_modes(target_frequency, n_modes):
    """Fresh closed-cavity geometry, meshed + solved for `n_modes` near shift.

    A new geometry per call keeps each `case.eigenmodes` (which meshes `g`)
    independent. Returns `(prob, modes)`.
    """
    f0 = ref.rect_cavity_freqs(A, B, D, er=1.0, n_modes=1)[0]
    g = case.geometry(maxh=rf.lambda_maxh(f_max=f0))
    air = g.box(A, B, D, position=(-A / 2, -B / 2, 0.0), material=rf.Air())
    rf.PEC(*air.faces.unassigned)  # closed cavity: every wall metal, no ports
    return case.eigenmodes(g, target_frequency=target_frequency, n_modes=n_modes)


def _best_rel_err(modes, f_analytic):
    """Smallest relative error of any computed mode against `f_analytic`."""
    return min(abs(m.frequency_hz - f_analytic) / f_analytic for m in modes)


def _distinct(freqs, rtol=0.01):
    """Collapse near-degenerate analytic frequencies to one representative."""
    out = []
    for f in freqs:
        if not out or abs(f - out[-1]) / out[-1] > rtol:
            out.append(f)
    return out


@pytest.mark.slow
@case.phenomenon
def test_rect_cavity_eigenfrequencies():
    analytic = ref.rect_cavity_freqs(A, B, D, er=1.0, n_modes=6)
    f0 = analytic[0]  # analytic fundamental (TE101)

    # ── Fundamental: shift at f0, recover TE101 within ~2 %. ────────────────
    prob, modes = _cavity_modes(f0, n_modes=6)
    assert prob.n_dofs < case.DOF_BUDGET, prob.n_dofs

    # Drop any static / near-zero mode below half the fundamental, then take
    # the computed resonance closest to the analytic TE101.
    physical = [m for m in modes if m.frequency_hz > 0.5 * f0]
    assert physical, [m.frequency_hz / 1e9 for m in modes]
    err0 = _best_rel_err(physical, f0)
    assert err0 < 0.02, (
        f"TE101: best computed match off by {err0 * 100:.2f}% "
        f"(analytic {f0 / 1e9:.4f} GHz; computed "
        f"{[round(m.frequency_hz / 1e9, 4) for m in modes]} GHz)"
    )

    # ── Spectrum: each of the first few distinct resonances is reproduced. ──
    # Shift 3 % below each analytic mode (do not sit on the answer) and require
    # a computed eigenfrequency within ~3 %.
    for fa in _distinct(analytic)[:4]:
        _, ms = _cavity_modes(0.97 * fa, n_modes=6)
        err = _best_rel_err(ms, fa)
        assert err < 0.03, (
            f"resonance {fa / 1e9:.4f} GHz: best match off by {err * 100:.2f}% "
            f"(computed {[round(m.frequency_hz / 1e9, 4) for m in ms]} GHz)"
        )
