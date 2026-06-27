# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""WR-90 rectangular waveguide — S-parameters of a matched straight section.

EXEMPLAR for the phenomenon suite: a matched, lossless WR-90 (X-band) section
transmits its TE10 mode with near-zero reflection, and the insertion phase of
S21 equals −β·L from the analytic guide dispersion. Phenomena exercised:
TE10 propagation, matching, passivity, analytic phase.

Reference: Pozar, *Microwave Engineering*, §3.3 (rectangular waveguide).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# WR-90 (X-band, 8.2–12.4 GHz)
A, B = 22.86e-3, 10.16e-3
LENGTH = 30e-3


@pytest.mark.slow
@case.phenomenon
def test_wr90_straight_transmission_and_phase():
    g = case.geometry(maxh=rf.lambda_maxh(f_max=12e9))
    air = g.box(A, B, LENGTH, position=(-A / 2, -B / 2, 0.0), material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)

    freqs = np.linspace(8.5e9, 11.5e9, 7)
    prob, res = case.sweep(g, freqs)

    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    # Matched, lossless: near-total transmission, small reflection.
    assert s21.min() > 0.97, f"|S21| dipped to {s21.min():.3f}"
    assert s11.max() < 0.08, f"|S11| rose to {s11.max():.3f}"

    # Passivity: each incident port's scattered power ≤ 1.
    for i in range(len(freqs)):
        assert case.passivity(res.sparams[i]) < 1.02

    # Phase SLOPE of S21 follows the guide dispersion: Δφ = −Δβ·L between two
    # frequencies. The absolute S21 phase carries a port-orientation convention
    # (a constant π offset); the slope is gauge-free and is the real physics.
    fa, fb = 2, 5
    dphi = np.angle(res.sparams[fb, 1, 0]) - np.angle(res.sparams[fa, 1, 0])
    dbeta = (ref.rect_beta(freqs[fb], A, B, m=1, n=0)
             - ref.rect_beta(freqs[fa], A, B, m=1, n=0))
    assert case.phase_close(dphi, -dbeta * LENGTH, tol_deg=8.0)
