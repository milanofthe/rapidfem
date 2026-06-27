# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Skin-effect wall loss in a WR-90 section — Leontovich surface impedance.

A matched, finite-conductivity WR-90 (X-band) section attenuates its TE10
mode purely through conductor (wall) loss. Replacing the four PEC side walls
with ``rf.SurfaceImpedance`` (the Leontovich skin-effect BC, Z_s = (1+j)R_s)
turns the guide into a matched lossy line, so

    |S21| = exp(-αc · L)

and the extracted αc_measured = -ln(|S21|)/L must reproduce the analytic
conductor attenuation of Pozar eq. 3.96 (``ref.wr_te10_wall_attenuation``).
Phenomena exercised: skin-effect surface impedance, TE10 conductor loss,
matched lossy-line transmission.

The loss is small (copper αc ≈ 0.012 Np/m → ~0.1 %/L over a 100 mm section),
but the lossless PEC numerical floor here is |S21| ≈ 0.99997 (a ~40x smaller
deviation), so a single-line raw extraction is clean. Length is capped at
100 mm to stay under the harness DOF budget (this geometry: 58 792 DOF; a
200 mm section already exceeds 100 000).

Reference: Pozar, *Microwave Engineering*, §3.3 (attenuation due to
conductor loss in a rectangular waveguide).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# WR-90 (X-band, 8.2-12.4 GHz); fc(TE10) ≈ 6.56 GHz.
A, B = 22.86e-3, 10.16e-3
LENGTH = 100e-3                 # long enough for resolvable loss, < 100 k DOF
SIGMA = 5.8e7                   # copper conductivity, S/m
# Mid-band sample frequencies, all well above the 6.56 GHz cutoff.
FREQS = np.array([8.5e9, 9.0e9, 9.5e9, 10.0e9])


def _alpha_from_s21(s21_mag, length):
    """Conductor attenuation αc (Np/m) of a matched lossy line: -ln|S21|/L."""
    return -np.log(s21_mag) / length


@pytest.mark.slow
@case.phenomenon
def test_skin_effect_wall_loss():
    g = case.geometry(maxh=rf.lambda_maxh(f_max=10e9))
    air = g.box(A, B, LENGTH, position=(-A / 2, -B / 2, 0.0), material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    # Leontovich skin-effect BC on the four side walls instead of PEC.
    rf.SurfaceImpedance(*air.faces.unassigned, conductivity=SIGMA)

    prob, res = case.sweep(g, FREQS)

    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    # Lossy but well matched: each port reflects almost nothing, and the line
    # is strictly lossy so transmission stays below unity.
    assert s11.max() < 0.01, f"|S11| rose to {s11.max():.4f}"
    assert s21.max() < 1.0, f"|S21| reached {s21.max():.6f} (not lossy)"
    assert s21.min() > 0.99, f"|S21| dipped to {s21.min():.6f} (too lossy)"

    # The real physics: extracted conductor attenuation matches Pozar's αc.
    # Leontovich BC + tetrahedral discretisation vs the textbook closed form
    # is an approximation; 15 % is an honest tolerance (measured here ~2 %).
    ac_meas = _alpha_from_s21(s21, LENGTH)
    ac_ana = np.array([ref.wr_te10_wall_attenuation(f, A, B, SIGMA) for f in FREQS])
    err = np.abs(ac_meas - ac_ana) / ac_ana
    assert err.max() < 0.15, (
        "αc(measured) vs αc(analytic) [Np/m]:\n"
        + "\n".join(
            f"  f={f/1e9:4.1f} GHz  meas={m:.5f}  ana={a:.5f}  err={e*100:4.1f}%"
            for f, m, a, e in zip(FREQS, ac_meas, ac_ana, err)
        )
    )
