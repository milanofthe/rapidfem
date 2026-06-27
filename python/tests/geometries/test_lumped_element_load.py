# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Lumped R/L/C load termination — the textbook lumped-load reflection.

A uniform line driven by a single ``rf.LumpedPort(z0=50)`` and terminated by an
``rf.LumpedElement`` (a concentrated series R-L-C across the far gap) reflects
with the elementary load reflection coefficient

    Γ = (Z_L − Zref) / (Z_L + Zref),    Zref = 50 Ω (the port reference).

Both boundaries are delta-gaps over the *same* footprint: the port imposes a
surface impedance ``Zref·w/h`` and the element imposes ``Z_L·w/h``, so the
geometric ``w/h`` cancels and the gap presents the bare load impedance ``Z_L``
to the line. To keep the reflection dominated by the load (not the line) the
shielded stripline is designed at Z0 ≈ Zref = 50 Ω: on a lossless 50 Ω line the
*magnitude* |Γ| is invariant with electrical length (the line only rotates the
phase), so |S11| reads the load directly. The section is also kept electrically
small (ℓ ≪ λ) as a second guard against any residual Z0 ≠ Zref.

Cases:
  * R = 50 Ω  (matched)     → |S11| ≈ 0      (well matched).
  * R = 100 Ω (mismatched)  → |S11| ≈ 1/3    (textbook lumped-load reflection).
  * L = 2 nH  (reactive)    → |S11| ≈ 1      (a lossless reactance reflects all).

Feed pattern (recessed interior delta-gaps, shield = default-PEC) mirrors
``test_stripline_impedance``. Reference: Pozar, *Microwave Engineering*, §2.3
(load reflection coefficient).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case

# Shielded stripline designed for Z0 ≈ 50 Ω so |Γ| reads the load directly.
# ref.stripline_z0(0.5 mm, 1 mm, er=4) = 30π/√εr · b/(w+0.441b) ≈ 50 Ω.
ER = 4.0
SUB_H = 1.0e-3        # b: ground-plane spacing (strip at b/2)
LINE_W = 0.5e-3       # w: strip width  → Z0 ≈ 50 Ω
LINE_L = 5.0e-3       # short section (electrically small over the band)
SUB_W = 5.0e-3        # shield width (hosts the strip fringe)
GAP_IN = 0.4e-3       # strip recess from each end (delta-gap location)
ZREF = 50.0           # lumped-port reference impedance


def _load_gamma(z_load: complex, zref: float = ZREF) -> complex:
    """Textbook load reflection coefficient Γ = (Z_L − Zref)/(Z_L + Zref)."""
    return (z_load - zref) / (z_load + zref)


def _build_line(g, attach_load):
    """Shielded stripline: 50 Ω lumped port at one end, a lumped load at the
    other, both recessed interior delta-gaps (strip → lower ground).

    ``attach_load(load_face)`` declares the far-end termination BC on the recess
    gap face. Returns the (port_face, load_face) pair."""
    hb = SUB_H / 2.0
    diel_lo = rf.Dielectric(er=ER, tand=0.0, maxh=SUB_H / 3.0)
    diel_hi = rf.Dielectric(er=ER, tand=0.0, maxh=SUB_H / 3.0)
    lower = g.box(SUB_W, LINE_L, hb, position=(-SUB_W / 2, 0.0, 0.0), material=diel_lo)
    upper = g.box(SUB_W, LINE_L, hb, position=(-SUB_W / 2, 0.0, hb), material=diel_hi)

    trace_l = LINE_L - 2.0 * GAP_IN
    trace = g.xy_plate(LINE_W, trace_l, position=(-LINE_W / 2, GAP_IN, hb))
    feed = g.xz_plate(LINE_W, hb, position=(-LINE_W / 2, GAP_IN, 0.0))
    load = g.xz_plate(LINE_W, hb, position=(-LINE_W / 2, LINE_L - GAP_IN, 0.0))

    g.fragment(lower, upper, trace, feed, load)

    # Strip is an interior PEC; the box's outer faces (grounds, side shields,
    # dead end caps behind the recess) ride on the default-PEC exterior. The two
    # recess gaps are interior → they carry the driven port and the load.
    rf.PEC(trace)
    rf.LumpedPort(feed, direction=(0, 0, 1), z0=ZREF)
    attach_load(load)
    return feed, load


def _solve_s11(attach_load, freqs):
    """Build → mesh → sweep a single-port line; return (|S11|(f), prob)."""
    g = case.geometry(maxh=rf.lambda_maxh(f_max=float(np.max(freqs)), er_max=ER))
    _build_line(g, attach_load)
    prob, res = case.sweep(g, freqs, z0=ZREF)
    return np.abs(res.sparams[:, 0, 0]), prob


# A short electrical length keeps the section ≪ λ; with Z0 ≈ Zref the |Γ|
# magnitude is anyway length-invariant, so a few points across the band suffice.
FREQS = np.linspace(2.0e9, 4.0e9, 4)


@pytest.mark.slow
@case.phenomenon
def test_matched_resistor_load_is_reflectionless():
    """R = 50 Ω terminating a 50 Ω-referenced line → |S11| ≈ 0."""
    s11, prob = _solve_s11(lambda f: rf.LumpedElement(f, r=50.0, direction=(0, 0, 1)),
                           FREQS)
    s11_mean = float(s11.mean())
    assert s11_mean < 0.10, (
        f"matched 50 Ω load not absorbed: mean |S11|={s11_mean:.3f} "
        f"(per-freq {np.round(s11, 3)}), DOF={prob.n_dofs}")


@pytest.mark.slow
@case.phenomenon
def test_mismatched_resistor_load_reflection():
    """R = 100 Ω → |S11| ≈ |(100−50)/(100+50)| = 1/3."""
    R = 100.0
    gamma_analytic = abs(_load_gamma(R))            # = 1/3
    s11, prob = _solve_s11(lambda f: rf.LumpedElement(f, r=R, direction=(0, 0, 1)),
                           FREQS)
    s11_meas = float(np.median(s11))
    rel = abs(s11_meas - gamma_analytic) / gamma_analytic
    assert rel < 0.20, (
        f"R={R} Ω reflection off: FEM |S11|={s11_meas:.3f} vs analytic "
        f"{gamma_analytic:.3f} ({100*rel:.1f} %); per-freq {np.round(s11, 3)}, "
        f"DOF={prob.n_dofs}")


@pytest.mark.slow
@case.phenomenon
def test_reactive_load_reflects_all_power():
    """A purely reactive load (L = 2 nH, no R) → |S11| ≈ 1 (|jX±50| are equal,
    so a lossless reactance returns all incident power)."""
    s11, prob = _solve_s11(lambda f: rf.LumpedElement(f, l=2.0e-9, direction=(0, 0, 1)),
                           FREQS)
    s11_min = float(s11.min())
    assert s11_min > 0.90, (
        f"reactive load leaks power: min |S11|={s11_min:.3f} "
        f"(per-freq {np.round(s11, 3)}), DOF={prob.n_dofs}")
