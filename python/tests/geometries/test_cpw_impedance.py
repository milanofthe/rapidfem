# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Coplanar-waveguide characteristic impedance — from a 50 Ω reflection.

A centre signal strip flanked by two coplanar ground strips on the same
dielectric surface (air above, substrate below) is a quasi-TEM line whose
characteristic impedance follows the classic conformal-mapping result

    Z0 = (30π/√εeff) · K(k')/K(k),   k = s/(s+2w),   k' = √(1−k²)

with εeff = (εr+1)/2 in the thick-substrate limit and K the complete elliptic
integral of the first kind (Simons/Wadell). Z0 depends only on the *ratio* k
and εr, so the cross-section is scaled up to ease meshing without moving Z0.
The dimensions here put Z0 ≈ 83 Ω — a deliberate, strong mismatch against the
50 Ω port reference (NOT an accidental match).

Like the stripline test, the line is fed by `rf.LumpedPort(z0=50)` delta-gaps
referenced to Zref = 50 Ω. The modal WavePort self-references to the line's own
modal impedance (reads matched, carries no Z0 information); only a fixed-Zref
lumped feed exposes a 50 Ω-referenced reflection. A lossless line of impedance
Z0 between equal Zref ports has an |S11| ENVELOPE that peaks at the quarter-wave
point (θ = βℓ = π/2), where the line transforms its far termination to
Zin = Z0²/Zref, giving

    |S11|_peak = |Z0² − Zref²| / (Z0² + Zref²)
    ⇒  Z0 = Zref·√((1+P)/(1−P))   (here Z0 > Zref)

We sweep across the band to capture that envelope peak P and invert it for Z0.

Modelling notes (the feed is the subtle part, copied from the stripline test):
the box's outer faces all ride on the default-PEC exterior, forming one
connected shield that ties both ground strips to a common potential (so the
odd/slotline mode is suppressed and a single-slot feed drives the pure CPW
mode). The signal is recessed by GAP_IN from each y-end so the feed bridges are
*interior* coplanar delta-gaps (signal→ground across one slot, on the
substrate/air interface) rather than boundary faces — otherwise the default-PEC
end walls would short the line. The asymmetric single-slot feed still measures
the full CPW mode: V = signal-to-ground across the slot, I = signal current, so
V/I = Z0 (verified lossless, |S11|²+|S21|² ≈ 1).

The conformal-mapping Z0 is itself only ~10 % accurate (thick-substrate,
infinite-ground idealisation vs the finite, shielded FEM); the FEM is the
reference and agreement within 15 % locks the extraction.

Reference: R. N. Simons, *Coplanar Waveguide Circuits, Components, and Systems*
(elliptic-integral Z0); Pozar, *Microwave Engineering*, §2.3 (the quarter-wave
/ lossless-line impedance transform).
"""
import numpy as np
import pytest
from scipy.special import ellipk

import rapidfem as rf
from harness import case

# CPW design (εr=4): the signal/gap ratio k = s/(s+2w) sets Z0 ≈ 83 Ω, a strong
# mismatch above the 50 Ω port reference. Features are scaled up (Z0 is
# scale-invariant) so a ~0.4 mm slot meshes cheaply inside the DOF budget.
ER = 4.0
SIG_W = 0.6e-3        # s: centre signal-strip width (x)
GAP_W = 0.4e-3        # w: slot between signal and each ground (x)
GND_W = 1.0e-3        # each coplanar ground-strip width (x)
SUB_H = 3.0e-3        # substrate thickness (z); air sits above. Thick enough
                      # that the default-PEC bottom does not load the line into
                      # the conductor-backed-CPW regime (h ≳ 2·(s+2w)).
AIR_H = 2.4e-3        # air region height (z); the top shield is likewise kept
                      # far so neither PEC plane drags Z0 down.
LINE_L = 8.0e-3       # line length along the propagation axis (y)
GAP_IN = 0.5e-3       # signal recess from each end (interior feed location)
FEED_L = 0.4e-3       # feed delta-gap length along y
ZREF = 50.0           # lumped-port reference impedance

SUB_W = SIG_W + 2.0 * GAP_W + 2.0 * GND_W   # total substrate/shield width (x)
FINE = GAP_W / 2.0                            # surface mesh near the slots


def _cpw_z0(s: float, w: float, er: float) -> float:
    """Analytic CPW Z0 via conformal mapping (complete elliptic integrals).

    k = s/(s+2w), k' = √(1−k²), εeff = (εr+1)/2 (thick substrate). SciPy's
    `ellipk` takes the parameter m = k², so K(k) = ellipk(k²)."""
    k = s / (s + 2.0 * w)
    kp2 = 1.0 - k * k
    eeff = (er + 1.0) / 2.0
    return 30.0 * np.pi / np.sqrt(eeff) * ellipk(kp2) / ellipk(k * k)


def _build_cpw(g):
    """Coplanar waveguide fed by a 50 Ω lumped slot delta-gap at each end.

    Substrate below, air above; signal + two grounds are coplanar PEC patches
    on the interface (z = SUB_H). The signal is recessed by GAP_IN so each feed
    plate is an interior gap bridging the signal edge to one ground across the
    right slot. Every exterior box face rides on the default-PEC shield (which
    ties the grounds together). Returns the two feed plates."""
    top_z = SUB_H
    diel = rf.Dielectric(er=ER, tand=0.0, maxh=SUB_H / 2.0)
    sub = g.box(SUB_W, LINE_L, SUB_H, position=(-SUB_W / 2, 0.0, 0.0), material=diel)
    air = g.box(SUB_W, LINE_L, AIR_H, position=(-SUB_W / 2, 0.0, top_z), material=rf.Air())

    # Coplanar conductors on the substrate/air interface.
    sig_l = LINE_L - 2.0 * GAP_IN
    signal = g.xy_plate(SIG_W, sig_l, position=(-SIG_W / 2, GAP_IN, top_z), maxh=FINE)
    gl_x0 = -SUB_W / 2
    gr_x0 = SIG_W / 2 + GAP_W
    gnd_l = g.xy_plate(GND_W, LINE_L, position=(gl_x0, 0.0, top_z), maxh=FINE)
    gnd_r = g.xy_plate(GND_W, LINE_L, position=(gr_x0, 0.0, top_z), maxh=FINE)

    # Feed bridges across the RIGHT slot (x: signal edge → right-ground edge),
    # one at each recessed signal end. Interior coplanar delta-gaps.
    feed_a = g.xy_plate(GAP_W, FEED_L, position=(SIG_W / 2, GAP_IN, top_z), maxh=FINE)
    feed_b = g.xy_plate(GAP_W, FEED_L,
                        position=(SIG_W / 2, LINE_L - GAP_IN - FEED_L, top_z), maxh=FINE)

    g.fragment(sub, air, signal, gnd_l, gnd_r, feed_a, feed_b)

    # Three conductors are internal PEC patches; the box outer faces (the shield
    # that commons the grounds) ride on the default-PEC exterior. The feed
    # plates are interior → they carry the lumped ports (E across the slot, +x).
    rf.PEC(signal, gnd_l, gnd_r)
    rf.LumpedPort(feed_a, direction=(1, 0, 0), z0=ZREF)
    rf.LumpedPort(feed_b, direction=(1, 0, 0), z0=ZREF)
    return feed_a, feed_b


@pytest.mark.slow
@case.phenomenon
def test_cpw_impedance_from_reflection():
    z0_analytic = _cpw_z0(SIG_W, GAP_W, ER)
    # Sanity: the design is a real mismatch, not an accidental 50 Ω match.
    assert abs(z0_analytic - ZREF) > 10.0, f"design Z0 {z0_analytic:.1f} too close to {ZREF}"

    g = case.geometry(maxh=rf.lambda_maxh(f_max=9.0e9, er_max=ER))
    _build_cpw(g)

    # εeff ≈ (εr+1)/2 = 2.5 puts the quarter-wave peak (θ = π/2) near 4.7 GHz
    # for this length; sweep 2–9 GHz so the |S11| envelope rises to its peak and
    # falls toward the half-wave null, resolving the peak inside the band.
    freqs = np.linspace(2.0e9, 9.0e9, 15)
    prob, res = case.sweep(g, freqs, z0=ZREF)

    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    # Lossless + passive: no dielectric/conductor loss, so |S11|²+|S21|² ≈ 1.
    for i in range(len(freqs)):
        assert case.passivity(res.sparams[i]) < 1.02, (
            f"power growth at {freqs[i]/1e9:.1f} GHz: {case.passivity(res.sparams[i]):.3f}")

    # A genuine standing-wave envelope: a clear reflection peak and a deep dip
    # as θ sweeps through π/2 toward π (not a flat matched line). A ~73 Ω FEM
    # line against the 50 Ω reference gives a peak near 0.37 — well clear of a
    # matched line's ≈0, so a 0.30 floor confirms a real envelope.
    p_peak = float(s11.max())
    assert p_peak > 0.30, f"reflection envelope too weak, peak |S11|={p_peak:.3f}"
    assert s11.min() < 0.25, f"no standing-wave dip, min |S11|={s11.min():.3f}"
    assert s21.max() > 0.85, f"line does not transmit, max |S21|={s21.max():.3f}"

    # Invert the quarter-wave envelope peak for Z0 (design Z0 > Zref).
    z0_meas = ZREF * np.sqrt((1.0 + p_peak) / (1.0 - p_peak))

    # Conformal-mapping Z0 is ~10 % accurate; the FEM is the reference.
    # Agreement within 15 % locks the extraction.
    rel = abs(z0_meas - z0_analytic) / z0_analytic
    assert rel < 0.15, (
        f"CPW Z0 mismatch: FEM {z0_meas:.2f} Ω vs conformal-map "
        f"{z0_analytic:.2f} Ω ({100*rel:.1f} %); peak |S11|={p_peak:.3f}, "
        f"DOF={prob.n_dofs}")

    # And the measured peak should track the analytic quarter-wave reflection.
    # The peak goes as the Z0²-form, so it amplifies the ~11 % Z0 gap; a 0.25
    # tolerance is the honest match given the conformal-map approximation.
    p_analytic = abs((z0_analytic**2 - ZREF**2) / (z0_analytic**2 + ZREF**2))
    assert abs(p_peak - p_analytic) / p_analytic < 0.25, (
        f"|S11| peak {p_peak:.3f} vs analytic qw-peak {p_analytic:.3f}")
