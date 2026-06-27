# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Dielectric (volume) loss in a filled rectangular waveguide — TE10 αd.

Filling a matched rectangular guide with a *lossy* dielectric (εr, tanδ)
turns it into a matched lossy line whose only loss mechanism is the volume
polarisation loss of the fill. With well-matched ports the transmission is

    |S21| = exp(-αd · L)

and the extracted αd_measured = -ln(|S21|)/L must reproduce the analytic
TE10 dielectric attenuation (Pozar eq. 3.29 / §3.3):

    αd = k0²·εr·tanδ / (2·β) [Np/m],   β = sqrt(εr·k0² − k_c²).

Phenomena exercised: lossy-dielectric loading, TE10 volume (dielectric)
attenuation, matched lossy-line transmission. Cross-check: with tanδ = 0 the
identical geometry must give α ≈ 0 (the lossless numerical floor).

The ports carry ``er=ER`` so the port mode impedance matches the fill (low
|S11|); the band sits well above the *filled* TE10 cutoff so the mode
propagates. Length is chosen so αd·L is comfortably resolvable while the
meshed problem stays under the harness DOF budget.

Reference: Pozar, *Microwave Engineering*, §3.3 (attenuation due to
dielectric loss in a rectangular waveguide).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# WR-90 cross-section, filled with a moderately lossy εr = 2.5, tanδ = 0.02.
A, B = 22.86e-3, 10.16e-3
LENGTH = 60e-3                  # αd·L ≈ 0.16 here: resolvable, < 100 k DOF
ER = 2.5
TAND = 0.02
# Filled TE10 cutoff ≈ 4.15 GHz; sample well above it so TE10 propagates.
FREQS = np.array([5.5e9, 6.0e9, 6.5e9, 7.0e9])
F_MAX = float(FREQS.max())

FC_FILLED = ref.rect_cutoff_freq(A, B, 1, 0, er=ER)


def _alpha_d_te10(f, a, b, er, tand):
    """Analytic TE10 dielectric attenuation αd (Np/m), Pozar §3.3:
    αd = k0²·εr·tanδ / (2·β), with β = sqrt(εr·k0² − k_c²)."""
    k0 = ref.k0(f)
    beta = ref.rect_beta(f, a, b, 1, 0, er=er)
    return k0 * k0 * er * tand / (2.0 * beta)


def _alpha_from_s21(s21_mag, length):
    """Attenuation α (Np/m) of a matched lossy line: -ln|S21|/L."""
    return -np.log(s21_mag) / length


def _build(tand):
    """A WR-90 section filled with εr=ER, given tanδ; matched WG ports + PEC."""
    g = case.geometry(maxh=rf.lambda_maxh(f_max=F_MAX, er_max=ER))
    diel = g.box(A, B, LENGTH, position=(-A / 2, -B / 2, 0.0),
                 material=rf.Dielectric(er=ER, tand=tand))
    rf.RectWaveguidePort(diel.faces.min(axis="z"), er=ER)
    rf.RectWaveguidePort(diel.faces.max(axis="z"), er=ER)
    rf.PEC(*diel.faces.unassigned)
    return g


@pytest.mark.slow
@case.phenomenon
def test_lossy_dielectric_attenuation():
    # Anchor: the whole sweep sits above the filled TE10 cutoff.
    assert FC_FILLED < FREQS.min(), f"fc_filled={FC_FILLED/1e9:.2f} GHz"

    prob, res = case.sweep(_build(TAND), FREQS)

    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    # Lossy but well matched: ports reflect almost nothing, and the line is
    # strictly lossy so transmission stays below unity. Low |S11| means the
    # raw -ln|S21|/L extraction needs no de-embedding.
    assert s11.max() < 0.05, f"|S11| rose to {s11.max():.4f}"
    assert s21.max() < 1.0, f"|S21| reached {s21.max():.6f} (not lossy)"
    assert s21.min() > 0.5, f"|S21| dipped to {s21.min():.4f} (too lossy)"

    # Passivity: scattered power per incident port ≤ 1.
    for i in range(len(FREQS)):
        assert case.passivity(res.sparams[i]) < 1.02

    # The real physics: extracted dielectric attenuation matches Pozar's αd.
    # FEM + tetrahedral discretisation vs the textbook closed form is an
    # approximation; 15 % is an honest tolerance.
    ad_meas = _alpha_from_s21(s21, LENGTH)
    ad_ana = np.array([_alpha_d_te10(f, A, B, ER, TAND) for f in FREQS])
    err = np.abs(ad_meas - ad_ana) / ad_ana
    assert err.max() < 0.15, (
        "αd(measured) vs αd(analytic) [Np/m]:\n"
        + "\n".join(
            f"  f={f/1e9:4.1f} GHz  meas={m:.4f}  ana={a:.4f}  err={e*100:4.1f}%"
            for f, m, a, e in zip(FREQS, ad_meas, ad_ana, err)
        )
    )

    # Cross-check / sanity floor: the IDENTICAL geometry with tanδ = 0 has no
    # loss mechanism, so the extracted α collapses to the numerical floor.
    _, res0 = case.sweep(_build(0.0), FREQS)
    s21_0 = np.abs(res0.sparams[:, 1, 0])
    a0 = _alpha_from_s21(s21_0, LENGTH)
    assert np.abs(a0).max() < 0.1 * ad_ana.min(), (
        f"tanδ=0 floor α={np.abs(a0).max():.4f} Np/m not << "
        f"αd={ad_ana.min():.4f} Np/m"
    )
