# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Coaxial line — S-parameters of a matched air-filled TEM section.

A matched, lossless coax carries a pure TEM mode (no cutoff): it transmits
with near-zero reflection and the insertion phase of S21 equals −β·L with
β = k0·√εr. Phenomena exercised: TEM propagation, matching, passivity, the
dispersionless (linear) phase slope.

Reference: Pozar, *Microwave Engineering*, §2.2 / §3.5 (coaxial line, TEM).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# Air coax: ro/ri = exp(2π·50/η0) ≈ 2.30 → Z0 ≈ 50 Ω.
RI, RO = 1.50e-3, 3.45e-3
LENGTH = 10e-3
ER = 1.0


@pytest.mark.slow
@case.phenomenon
def test_coax_matched_transmission_and_phase():
    # Sanity: chosen radii really are ~50 Ω (the matching target).
    assert abs(ref.coax_z0(RI, RO, ER) - 50.0) < 2.0

    g = case.geometry(maxh=rf.lambda_maxh(f_max=15e9))
    cx = rf.structures.coax(g, ri=RI, ro=RO, length=LENGTH, er=ER,
                            add_ports=True)
    assert len(cx.ports) == 2

    freqs = np.linspace(5e9, 15e9, 7)
    prob, res = case.sweep(g, freqs)

    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    # Matched, lossless TEM: near-total transmission, tiny reflection.
    assert s21.min() > 0.95, f"|S21| dipped to {s21.min():.3f}"
    assert s11.max() < 0.03, f"|S11| rose to {s11.max():.3f}"

    # Passivity: each incident port's scattered power ≤ 1.
    for i in range(len(freqs)):
        assert case.passivity(res.sparams[i]) < 1.02

    # Phase SLOPE of S21 follows the TEM dispersion: Δφ = −Δβ·L between two
    # frequencies, β = k0·√εr. The absolute S21 phase carries a port-orientation
    # convention offset; the slope is gauge-free and is the real physics.
    fa, fb = 1, 5
    dphi = np.angle(res.sparams[fb, 1, 0]) - np.angle(res.sparams[fa, 1, 0])
    dphi_ref = (ref.tem_phase(freqs[fb], LENGTH, ER)
                - ref.tem_phase(freqs[fa], LENGTH, ER))
    assert case.phase_close(dphi, dphi_ref, tol_deg=8.0)
