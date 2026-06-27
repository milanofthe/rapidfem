# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Drude (free-electron metal) slab in a waveguide — dispersive reflectivity.

A Drude medium has a frequency-dependent permittivity

    ε_r(ω) = ε∞ − ω_p² / (ω(ω + jγ)),   ω_p = 2π·f_p,  γ = 2π·f_damp

that is strongly NEGATIVE below the plasma frequency f_p (the medium reflects,
like a metal) and crosses to POSITIVE above it (the medium turns transmissive).
This test drops a Drude slab into a WR-90 cross-section, fed and collected by air
sections whose ports stay well defined. A TE10 wave hitting the slab sees an
effective propagation constant

    β_slab² = Re(ε_r(f))·k0² − k_c²,      k_c = π/a,

so the slab is an EVANESCENT barrier wherever β_slab² < 0 (high |S11|, low |S21|)
and turns PROPAGATING once β_slab² > 0 (transmission recovers). The sign change of
β_slab² — driven by the Drude dispersion sweeping Re(ε_r) from negative to
positive — predicts the reflective↔transmissive crossover. We pin the measured
|S| spectrum to that crossover, computed independently from `ref.drude_eps`.

Reference: Jackson, *Classical Electrodynamics*, §7.5 (plasma/Drude dispersion);
Pozar, *Microwave Engineering*, §3.3 (TE10 guide dispersion, β = √(εr·k0²−k_c²)).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# WR-90 cross-section (X-band, air TE10 cutoff ≈ 6.56 GHz).
A, B = 22.86e-3, 10.16e-3
L_FEED = 10.0e-3                 # air feed / collect sections (ports live here)
L_SLAB = 13.0e-3                 # Drude barrier length

# Drude model: plasma frequency in-band so Re(εr) sweeps negative → positive.
F_PLASMA = 9.0e9                 # Re(εr) crosses 0 here
F_DAMP = 0.2e9                   # light damping (small loss, keeps physics clean)
ER_INF = 1.0

FREQS = np.linspace(7.5e9, 14.0e9, 9)
F_MAX = 14.5e9


def _slab_beta_sq(f: float) -> float:
    """TE10 propagation constant² inside the Drude slab: Re(εr)·k0² − k_c².

    > 0 ⇒ the slab propagates (transmissive); < 0 ⇒ evanescent (reflective).
    """
    kc = np.pi / A
    er = ref.drude_eps(f, F_PLASMA, F_DAMP, ER_INF)
    return float(er.real * ref.k0(f) ** 2 - kc * kc)


def _crossover_freq() -> float:
    """Frequency where β_slab² changes sign, from `ref.drude_eps` alone."""
    fine = np.linspace(FREQS.min(), FREQS.max(), 2001)
    g = np.array([_slab_beta_sq(f) for f in fine])
    i = int(np.where(np.diff(np.sign(g)))[0][0])
    return float(fine[i])


def _build_and_solve():
    """air feed → Drude slab → air collect; return (prob, result, freqs)."""
    g = case.geometry(maxh=rf.lambda_maxh(f_max=F_MAX, per_lambda=12))
    # Refine the slab: the strong εr contrast at the air↔Drude interface is the
    # dominant discretization-error source, so resolve it harder than the air.
    drude = rf.Material(
        drude=rf.Drude(plasma_freq_hz=F_PLASMA, damping_freq_hz=F_DAMP,
                       er_inf=ER_INF),
        maxh=rf.lambda_maxh(f_max=F_MAX, per_lambda=14))

    inp = g.box(A, B, L_FEED, position=(-A / 2, -B / 2, 0.0),
                material=rf.Air())
    slab = g.box(A, B, L_SLAB, position=(-A / 2, -B / 2, L_FEED),
                 material=drude)
    out = g.box(A, B, L_FEED, position=(-A / 2, -B / 2, L_FEED + L_SLAB),
                material=rf.Air())
    # Fragment so the air↔Drude faces become shared conformal interfaces: those
    # internal faces carry field continuity, while the outer guide walls stay
    # exterior and default to PEC. Only the two air end-faces are driven ports.
    g.fragment(slab, inp, out)

    rf.RectWaveguidePort(inp.faces.min(axis="z"))
    rf.RectWaveguidePort(out.faces.max(axis="z"))

    prob, res = case.sweep(g, FREQS)
    return prob, res


@pytest.mark.slow
@case.phenomenon
def test_drude_slab_reflective_below_plasma_transmissive_above():
    # Anchor the prediction independently of the solver: the band must straddle
    # the air cutoff (so the feed ports propagate) and the β_slab² sign change.
    assert ref.rect_cutoff_freq(A, B, 1, 0) < FREQS.min(), "feed below cutoff"
    f_star = _crossover_freq()
    assert FREQS.min() < f_star < FREQS.max(), f"crossover {f_star/1e9:.2f} GHz"
    # Drude physics: strongly negative ε at the bottom, positive at the top.
    assert ref.drude_eps(FREQS.min(), F_PLASMA, F_DAMP, ER_INF).real < -0.2
    assert ref.drude_eps(FREQS.max(), F_PLASMA, F_DAMP, ER_INF).real > 0.2

    prob, res = _build_and_solve()
    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    print(f"\nn_dofs={prob.n_dofs}  crossover={f_star/1e9:.2f} GHz")
    for f, e, t, r in zip(FREQS, [_slab_beta_sq(f) for f in FREQS], s21, s11):
        tag = "prop" if e > 0 else "evan"
        print(f"  {f/1e9:5.2f} GHz  beta_slab^2={e:+9.0f} [{tag}]  "
              f"|S21|={t:.3f}  |S11|={r:.3f}")

    # Passivity: the lossy Drude slab can only absorb, so scattered power ≤ 1.
    # A few-% over-unity is the FEM unitarity error of the high-contrast lossy
    # air/Drude interface at a laptop mesh density — it is numerical, not a leak.
    pmax = max(case.passivity(res.sparams[i]) for i in range(len(FREQS)))
    print(f"  max passivity = {pmax:.4f}")
    assert pmax < 1.08, f"passivity {pmax:.3f} too far over unity"

    # The transition is CONTINUOUS: β_slab² passes smoothly through 0, so as the
    # Drude permittivity sweeps from strongly negative to positive, |S21| rises
    # monotonically and |S11| falls monotonically across the whole band.
    assert np.all(np.diff(s21) > 0.0), f"|S21| not monotone up: {s21}"
    assert np.all(np.diff(s11) < 0.0), f"|S11| not monotone down: {s11}"

    # Deep evanescent end (Re(ε)≪0): a metal-like mirror — near-total reflection,
    # almost nothing transmitted.
    assert s11[0] > 0.95 and s21[0] < 0.25, (
        f"bottom not reflective: |S11|={s11[0]:.3f} |S21|={s21[0]:.3f}")
    # Deep propagating end (Re(ε)>0): the slab transmits — reflection collapses.
    assert s21[-1] > 0.90 and s11[-1] < 0.35, (
        f"top not transmissive: |S21|={s21[-1]:.3f} |S11|={s11[-1]:.3f}")

    # The reflective→transmissive handover (|S21| overtaking |S11|: the slab
    # stops mirroring and starts passing power) must land at the crossover
    # PREDICTED by ref.drude_eps (β_slab² = 0). This pins the measured transition
    # to the Drude dispersion rather than to any fitted number.
    diff = s21 - s11
    k = int(np.where(np.diff(np.sign(diff)))[0][0])
    f_eq = FREQS[k] + (FREQS[k + 1] - FREQS[k]) * (-diff[k]) / (diff[k + 1] - diff[k])
    print(f"  |S21|=|S11| handover = {f_eq/1e9:.2f} GHz  (predicted {f_star/1e9:.2f} GHz)")
    assert abs(f_eq - f_star) < 1.0e9, (
        f"handover {f_eq/1e9:.2f} GHz vs predicted crossover {f_star/1e9:.2f} GHz")
