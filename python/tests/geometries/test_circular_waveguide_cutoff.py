# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Circular waveguide — dominant-mode (TE11) cutoff from the Bessel zero.

A hollow circular guide of radius ``R`` carries as its dominant mode the
TE11, whose cutoff wavenumber is the first zero of ``J1'``:

    k_c = p'11 / R,   p'11 = 1.8412   (first zero of J1')
    fc  = c · p'11 / (2π R)

(the next mode is TM01 at p01 = 2.4048, i.e. ~1.31·fc, so there is a clean
single-mode band above fc). Unlike a rectangular guide there is no analytic
rectangular port, so each end is driven by a numerically-solved full-vector
WavePort whose dominant mode IS the TE11.

The physics asserted is the cutoff WAVENUMBER, via the guide dispersion. The
analytic TE11 propagation constant is

    β(f) = sqrt(εr·k0² − (p'11/R)²).

The absolute S21 phase carries a port-orientation convention (a constant
offset), so — exactly as in the WR-90 exemplar — we use the gauge-free phase
SLOPE: between two frequencies Δφ_S21 = −Δβ·L. Matching that slope to the
circular-guide β proves the (p'11/R)² cutoff term, i.e. the Bessel zero. As a
recovered figure we also least-squares fit k_c back out of the measured slopes
and report fc_meas vs the analytic value.

This avoids the modal-port-below-its-own-cutoff problem by staying entirely
in the propagating single-mode band (fc < f < f_TM01).

Reference: Pozar, *Microwave Engineering*, §3.4 (circular waveguide).
"""
import numpy as np
import pytest
from scipy.special import jn_zeros, jnp_zeros

import rapidfem as rf
from rapidfem import structures as st
from harness import case, references as ref

# Geometry: a 10 mm-radius air guide. TE11 cutoff ≈ 8.785 GHz, TM01 ≈ 11.47 GHz.
R = 10.0e-3
LENGTH = 40.0e-3

# Bessel-zero cutoff constants (LOCAL — niche to this test).
P11_PRIME = float(jnp_zeros(1, 1)[0])   # 1.8412, first zero of J1' → TE11
P01 = float(jn_zeros(0, 1)[0])          # 2.4048, first zero of J0  → TM01

# Single-mode propagating band, comfortably above TE11 cutoff and below TM01.
FREQS = np.linspace(9.5e9, 11.0e9, 4)
F0 = 10.25e9                            # port eigensolve / band centre


def _circ_cutoff(p_zero: float, radius: float, er: float = 1.0) -> float:
    """Cutoff frequency of a circular-guide mode with Bessel zero ``p_zero``."""
    return ref.C0 * p_zero / (2.0 * np.pi * radius * np.sqrt(er))


def _circ_beta(f: float, radius: float, p_zero: float, er: float = 1.0) -> float:
    """TE/TM propagation constant β = sqrt(εr·k0² − (p_zero/R)²); 0 if cut off."""
    kc = p_zero / radius
    k = ref.k0(f) * np.sqrt(er)
    arg = k * k - kc * kc
    return float(np.sqrt(arg)) if arg > 0.0 else 0.0


def _fit_cutoff_wavenumber(freqs, dbeta_meas) -> float:
    """Recover k_c from the measured consecutive Δβ, by least squares.

    β(f) = sqrt(k0(f)² − k_c²) ⇒ the SET of consecutive Δβ values pins k_c
    (the unknown constant phase offset cancels in a difference). Returns the
    grid-refined k_c minimizing Σ (Δβ_model(k_c) − Δβ_meas)².
    """
    k0s = 2.0 * np.pi * np.asarray(freqs) / ref.C0

    def cost(kc):
        bm = np.sqrt(np.clip(k0s ** 2 - kc * kc, 0.0, None))
        return float(np.sum((np.diff(bm) - dbeta_meas) ** 2))

    grid = np.linspace(0.5 * P11_PRIME / R, 1.5 * P11_PRIME / R, 8001)
    return float(grid[int(np.argmin([cost(kc) for kc in grid]))])


@pytest.mark.slow
@case.phenomenon
def test_circular_waveguide_te11_cutoff():
    fc_te11 = _circ_cutoff(P11_PRIME, R)
    fc_tm01 = _circ_cutoff(P01, R)
    # Anchor: the band is single-mode and propagating — above TE11 cutoff,
    # below the next (TM01) mode, so only the dominant TE11 carries power.
    assert FREQS.min() > fc_te11 + 0.3e9, f"band dips toward cutoff {fc_te11/1e9:.2f} GHz"
    assert FREQS.max() < fc_tm01 - 0.3e9, f"band reaches TM01 {fc_tm01/1e9:.2f} GHz"

    g = case.geometry(maxh=rf.lambda_maxh(f_max=11.5e9))
    # The auto full-vector WavePort's dominant mode is the circular TE11; the
    # curved wall is set PEC by the builder.
    st.circ_waveguide(g, radius=R, length=LENGTH, add_ports=True, f0=F0)
    prob, res = case.sweep(g, FREQS)

    s21 = res.sparams[:, 1, 0]
    s11 = res.sparams[:, 0, 0]

    # Single propagating mode through a matched, lossless guide: near-total
    # transmission and small reflection across the band.
    assert np.abs(s21).min() > 0.95, f"|S21| dipped to {np.abs(s21).min():.3f}"
    assert np.abs(s11).max() < 0.15, f"|S11| rose to {np.abs(s11).max():.3f}"
    for i in range(len(FREQS)):
        assert case.passivity(res.sparams[i]) < 1.02

    # Phase SLOPE: each consecutive ΔβL is < π (no wrap ambiguity), and must
    # match the circular-guide dispersion Δφ = −Δβ·L. This is the load-bearing
    # assertion — the (p'11/R)² term in β is the Bessel-zero cutoff.
    phase = np.angle(s21)
    dphi = np.array([np.angle(np.exp(1j * (phase[i + 1] - phase[i])))
                     for i in range(len(FREQS) - 1)])
    dbeta = np.diff([_circ_beta(f, R, P11_PRIME) for f in FREQS])
    assert case.phase_close(dphi, -dbeta * LENGTH, tol_deg=8.0), (
        f"S21 phase slope {np.rad2deg(dphi)} deg vs analytic "
        f"{np.rad2deg(-dbeta * LENGTH)} deg"
    )

    # Recover the cutoff wavenumber from the measured slopes and check it lands
    # on the Bessel-zero value within 3 % (don't fudge — report it).
    dbeta_meas = -dphi / LENGTH
    kc_meas = _fit_cutoff_wavenumber(FREQS, dbeta_meas)
    fc_meas = ref.C0 * kc_meas / (2.0 * np.pi)
    err = abs(fc_meas - fc_te11) / fc_te11
    assert err < 0.03, (
        f"recovered TE11 cutoff {fc_meas/1e9:.3f} GHz vs analytic "
        f"{fc_te11/1e9:.3f} GHz ({err*100:.1f} %)"
    )

    print(f"\nn_dofs={prob.n_dofs} n_tets={prob.n_tets}")
    print(f"TE11 cutoff: analytic {fc_te11/1e9:.4f} GHz, "
          f"recovered {fc_meas/1e9:.4f} GHz ({err*100:.2f} %)")
    print(f"k_c: analytic {P11_PRIME/R:.3f} /m, recovered {kc_meas:.3f} /m")
