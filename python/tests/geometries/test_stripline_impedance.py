# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Stripline characteristic impedance — extracted from a 50 Ω reflection.

A center strip between two ground planes in a homogeneous dielectric is a TEM
line of impedance ``Z0 = ref.stripline_z0(w, b, er)``. Here the line is fed by
two *lumped* ports referenced to Zref = 50 Ω and deliberately designed off 50 Ω
(~30 Ω), so it is a real mismatch. A lossless line of impedance ``Z0`` between
equal Zref ports has an input reflection whose magnitude oscillates with
electrical length and whose ENVELOPE peaks at the quarter-wave point
(θ = βℓ = π/2), where the line transforms its far-port termination to
``Zin = Z0²/Zref``. The peak reflection is therefore

    |S11|_peak = |Z0² − Zref²| / (Z0² + Zref²)   ⇒   Z0 = Zref·√((1−P)/(1+P))

for Z0 < Zref (the design case). We sweep across the band to capture that
envelope peak P and invert it for Z0, then compare to the Cohn/Wheeler closed
form. (The single-interface number |(Z0−Zref)/(Z0+Zref)| is NOT the two-port
peak — a quarter-wave section doubles the impedance-transform, so the envelope
peak is the larger Z0²-form above.)

Phenomena exercised: TEM stripline propagation, the quarter-wave impedance
transform, passivity, and impedance extraction against a fixed 50 Ω reference.

Modelling note: the modal wave port self-references to the line's own modal
impedance (so it reads matched, |S11|≈0 and carries no Z0 information). To see a
50 Ω-referenced reflection the line is fed by `rf.LumpedPort(z0=50)` delta-gaps
between the strip and the lower ground at each (slightly recessed) strip end;
every other exterior face is the shielding ground (default-PEC). The asymmetric
lower-gap feed still measures the full stripline mode: V = strip-to-ground,
I = strip current, so V/I = Z0 (verified lossless, |S11|²+|S21|² ≈ 1).

Reference: Pozar, *Microwave Engineering*, §2.5 (stripline) & §2.3 (the
quarter-wave / lossless-line impedance transform).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# Stripline design: er and (w, b) chosen so the analytic Z0 ≈ 30 Ω — a
# deliberate, strong mismatch against the 50 Ω port reference.
ER = 4.0
SUB_H = 1.0e-3        # b: ground-plane spacing (strip sits at b/2)
LINE_W = 1.13e-3      # w: strip width  → ref.stripline_z0 ≈ 30 Ω
LINE_L = 8.0e-3       # dielectric length along the line (y)
SUB_W = 6.0e-3        # shield width (x), wide enough to host the strip fringe
GAP_IN = 0.4e-3       # strip recess from each end (feed delta-gap location)
ZREF = 50.0           # lumped-port reference impedance


def _build_stripline(g):
    """Shielded stripline fed by a 50 Ω lumped delta-gap at each strip end.

    Two dielectric halves meet at the strip plane (z = b/2); the strip is
    recessed by ``GAP_IN`` from both ends so the feed plates are *interior*
    delta-gaps (strip→lower ground) rather than boundary faces — that keeps the
    line from being shorted by the default-PEC end walls. Returns the two feed
    plates (the lumped-port faces)."""
    hb = SUB_H / 2.0
    diel_lo = rf.Dielectric(er=ER, tand=0.0, maxh=SUB_H / 3.0)
    diel_hi = rf.Dielectric(er=ER, tand=0.0, maxh=SUB_H / 3.0)
    lower = g.box(SUB_W, LINE_L, hb, position=(-SUB_W / 2, 0.0, 0.0), material=diel_lo)
    upper = g.box(SUB_W, LINE_L, hb, position=(-SUB_W / 2, 0.0, hb), material=diel_hi)

    trace_l = LINE_L - 2.0 * GAP_IN
    trace = g.xy_plate(LINE_W, trace_l, position=(-LINE_W / 2, GAP_IN, hb))
    feed_a = g.xz_plate(LINE_W, hb, position=(-LINE_W / 2, GAP_IN, 0.0))
    feed_b = g.xz_plate(LINE_W, hb, position=(-LINE_W / 2, LINE_L - GAP_IN, 0.0))

    g.fragment(lower, upper, trace, feed_a, feed_b)

    # Strip is an internal PEC; the box's outer faces (both grounds, the side
    # shields, and the dead end caps behind the recess) ride on the default-PEC
    # exterior. The feed plates are interior → they carry the lumped ports.
    rf.PEC(trace)
    rf.LumpedPort(feed_a, direction=(0, 0, 1), z0=ZREF)
    rf.LumpedPort(feed_b, direction=(0, 0, 1), z0=ZREF)
    return feed_a, feed_b


@pytest.mark.slow
@case.phenomenon
def test_stripline_impedance_from_reflection():
    z0_analytic = ref.stripline_z0(LINE_W, SUB_H, ER)
    # Sanity: the design is a real mismatch, not an accidental 50 Ω match.
    assert abs(z0_analytic - ZREF) > 10.0, f"design Z0 {z0_analytic:.1f} too close to {ZREF}"

    g = case.geometry(maxh=rf.lambda_maxh(f_max=9.0e9, er_max=ER))
    _build_stripline(g)

    # The quarter-wave peak (θ = π/2) sits near 5 GHz for this length/εr; sweep
    # 2–9 GHz so the |S11| envelope rises to its peak and falls toward the
    # half-wave null, resolving the peak well inside the band.
    freqs = np.linspace(2.0e9, 9.0e9, 15)
    prob, res = case.sweep(g, freqs, z0=ZREF)

    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    # Lossless + passive: no dielectric/conductor loss, so |S11|²+|S21|² ≈ 1.
    for i in range(len(freqs)):
        assert case.passivity(res.sparams[i]) < 1.02, (
            f"power growth at {freqs[i]/1e9:.1f} GHz: {case.passivity(res.sparams[i]):.3f}")

    # A genuine standing-wave envelope: a clear reflection peak and a deep
    # minimum as θ sweeps through π/2 toward π (not a flat matched line).
    p_peak = float(s11.max())
    assert p_peak > 0.40, f"reflection envelope too weak, peak |S11|={p_peak:.3f}"
    assert s11.min() < 0.25, f"no standing-wave dip, min |S11|={s11.min():.3f}"
    assert s21.max() > 0.85, f"line does not transmit, max |S21|={s21.max():.3f}"

    # Invert the quarter-wave envelope peak for Z0 (design Z0 < Zref).
    z0_meas = ZREF * np.sqrt((1.0 - p_peak) / (1.0 + p_peak))

    # The closed form (Cohn/Wheeler, thin-strip) is itself only ~10 % accurate;
    # the FEM is the reference. Agreement within 15 % locks the extraction.
    rel = abs(z0_meas - z0_analytic) / z0_analytic
    assert rel < 0.15, (
        f"stripline Z0 mismatch: FEM {z0_meas:.2f} Ω vs closed-form "
        f"{z0_analytic:.2f} Ω ({100*rel:.1f} %); peak |S11|={p_peak:.3f}, "
        f"DOF={prob.n_dofs}")

    # And the measured peak should track the analytic quarter-wave reflection.
    p_analytic = abs((z0_analytic**2 - ZREF**2) / (z0_analytic**2 + ZREF**2))
    assert abs(p_peak - p_analytic) / p_analytic < 0.20, (
        f"|S11| peak {p_peak:.3f} vs analytic qw-peak {p_analytic:.3f}")
