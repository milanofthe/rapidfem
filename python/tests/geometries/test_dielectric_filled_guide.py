# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Dielectric-filled rectangular waveguide — TE10 dispersion shift.

Filling a rectangular guide with a lossless dielectric (εr) lowers the TE10
cutoff by √εr and raises β at a given frequency. This test drives a guide whose
operating band sits *between* the air cutoff and the dielectric cutoff: the band
only propagates because the solver carries εr in the wave operator. Phenomena
exercised: dielectric loading, cutoff shift, analytic (filled) phase slope.

Reference: Pozar, *Microwave Engineering*, §3.3 — a guide filled with εr behaves
like an air guide with k0 → √εr·k0, so β = sqrt(εr·k0² − k_c²).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# WR-90 cross-section, filled with εr = 2.2 (e.g. PTFE).
A, B = 22.86e-3, 10.16e-3
LENGTH = 30e-3
ER = 2.2

# Air TE10 cutoff ≈ 6.557 GHz; filling drops it by √εr to ≈ 4.42 GHz. The whole
# sweep sits between the two cutoffs, so it propagates ONLY because of εr.
FC_AIR = ref.rect_cutoff_freq(A, B, 1, 0, er=1.0)
FC_FILLED = ref.rect_cutoff_freq(A, B, 1, 0, er=ER)


@pytest.mark.slow
@case.phenomenon
def test_dielectric_filled_te10_dispersion():
    # Sanity on the anchor: the band straddles only the filled cutoff.
    assert FC_FILLED < 5.0e9 < 6.2e9 < FC_AIR

    g = case.geometry(maxh=rf.lambda_maxh(f_max=6.2e9, er_max=ER))
    diel = g.box(A, B, LENGTH, position=(-A / 2, -B / 2, 0.0),
                 material=rf.Dielectric(er=ER, tand=0.0))
    rf.RectWaveguidePort(diel.faces.min(axis="z"), er=ER)
    rf.RectWaveguidePort(diel.faces.max(axis="z"), er=ER)
    rf.PEC(*diel.faces.unassigned)

    freqs = np.linspace(5.0e9, 6.2e9, 7)
    prob, res = case.sweep(g, freqs)

    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    # Matched, lossless filled guide: near-total transmission, low reflection.
    # This band is BELOW the air cutoff — high |S21| here is the dielectric
    # loading at work; an air-filled operator would give an evanescent stop-band.
    assert s21.min() > 0.95, f"|S21| dipped to {s21.min():.3f}"
    assert s11.max() < 0.08, f"|S11| rose to {s11.max():.3f}"

    # Passivity: scattered power per incident port ≤ 1.
    for i in range(len(freqs)):
        assert case.passivity(res.sparams[i]) < 1.02

    # Phase SLOPE of S21 follows the FILLED-guide dispersion: Δφ = −Δβ·L between
    # two in-band frequencies, with β = sqrt(εr·k0² − k_c²). The absolute phase
    # carries a port-orientation convention; the slope is gauge-free physics and
    # only matches if the solver uses εr in the wave operator.
    fa, fb = 1, 5
    dphi = np.angle(res.sparams[fb, 1, 0]) - np.angle(res.sparams[fa, 1, 0])
    dbeta = (ref.rect_beta(freqs[fb], A, B, m=1, n=0, er=ER)
             - ref.rect_beta(freqs[fa], A, B, m=1, n=0, er=ER))
    assert case.phase_close(dphi, -dbeta * LENGTH, tol_deg=8.0)
