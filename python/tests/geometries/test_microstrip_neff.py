# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Microstrip line — quasi-TEM effective index from the S21 phase slope.

A signal trace on an FR4-like substrate (er = 4.4) over a ground plane, air
above, driven at each end by a full-vector wave port that de-embeds the
inhomogeneous quasi-TEM mode. That mode propagates with β = k0·√εeff, so the
insertion phase of S21 falls off linearly with k0 at slope −√εeff·L.

The gauge-free assertion is the phase SLOPE: the absolute S21 phase carries a
port-orientation convention offset, but Δφ = −Δβ·L between two frequencies is
the real physics. We compare against β = k0·√(ref.microstrip_eeff(w,h,er)), the
quasi-static Hammerstad closed form. Hammerstad is an approximation, so the FEM
εeff (more accurate) is allowed to differ by a few percent.

Reference: Pozar, *Microwave Engineering*, §3.8 (microstrip); Hammerstad εeff.
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

mm = 1e-3

# FR4-like substrate; w/h ≈ 1.9 puts a ~0.8 mm-thick line near 50 Ω.
ER = 4.4
SUB_H = 1.5 * mm
LINE_W = 2.85 * mm          # w/h ≈ 1.9 → ~50 Ω on er = 4.4
LINE_L = 12.0 * mm
SUB_W = 10.0 * mm           # narrow enough to push the box resonance above 4 GHz
AIR_H = 5.0 * mm

FREQS = np.linspace(2.0e9, 4.0e9, 5)
F0 = 0.5 * (FREQS[0] + FREQS[-1])


def _eeff_from_slope(freqs, s21, length):
    """Effective permittivity from the unwrapped S21 phase vs k0.

    φ(k0) = −√εeff·L·k0 + const, so a linear fit's slope gives √εeff."""
    k0 = ref.k0(np.asarray(freqs))
    phase = np.unwrap(np.angle(np.asarray(s21)))
    slope = np.polyfit(k0, phase, 1)[0]
    return (-slope / length) ** 2


@pytest.mark.slow
@case.phenomenon
def test_microstrip_quasi_tem_effective_index():
    # Sanity: the chosen geometry is microstrip-ish (a few-εeff quasi-TEM line).
    eeff_ref = ref.microstrip_eeff(LINE_W, SUB_H, ER)
    assert 1.0 < eeff_ref < ER

    g = case.geometry(maxh=rf.lambda_maxh(f_max=FREQS[-1], er_max=ER))
    ms = rf.structures.microstrip(
        g, line_w=LINE_W, line_l=LINE_L,
        sub_w=SUB_W, sub_h=SUB_H, air_h=AIR_H,
        er=ER, add_ports=True, f0=F0)
    assert len(ms.ports) == 2

    prob, res = case.sweep(g, FREQS)

    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    # Reasonable matching / transmission: the quasi-TEM mode carries the power,
    # it is not a dead (reflecting) line.
    assert s21.min() > 0.7, f"|S21| dipped to {s21.min():.3f}"
    assert s11.max() < 0.4, f"|S11| rose to {s11.max():.3f}"

    # Passivity: each incident port's scattered power ≤ 1.
    for i in range(len(FREQS)):
        assert case.passivity(res.sparams[i]) < 1.05

    # Phase SLOPE of S21 follows the quasi-TEM dispersion: Δφ = −Δβ·L between two
    # frequencies, β = k0·√εeff with the Hammerstad εeff. Endpoints maximise the
    # slope; |Δβ·L| < π here, so the wrapped comparison is unambiguous.
    fa, fb = 0, len(FREQS) - 1
    dphi = np.angle(res.sparams[fb, 1, 0]) - np.angle(res.sparams[fa, 1, 0])
    dbeta = (ref.k0(FREQS[fb]) - ref.k0(FREQS[fa])) * np.sqrt(eeff_ref)
    assert case.phase_close(dphi, -dbeta * LINE_L, tol_deg=10.0)

    # Cross-check: εeff extracted from the full-band phase slope vs Hammerstad.
    # The FEM solves the true inhomogeneous mode, so a few-percent gap to the
    # quasi-static closed form is expected and fine.
    eeff_meas = _eeff_from_slope(FREQS, res.sparams[:, 1, 0], LINE_L)
    assert abs(eeff_meas - eeff_ref) / eeff_ref < 0.08, (
        f"εeff measured {eeff_meas:.3f} vs Hammerstad {eeff_ref:.3f}")
