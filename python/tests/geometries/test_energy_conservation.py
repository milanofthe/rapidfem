# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Scattering-matrix energy conservation — unitarity and passivity.

Two structures probe the same conservation law from opposite sides.

1. LOSSLESS unitarity. A WR-90 H-plane step (the guide width drops from
   22.86 mm to 15 mm, all PEC walls + air) is a lossless, reciprocal 2-port.
   Its scattering matrix must be unitary, so the incident power at either port
   is fully accounted for by the reflected + transmitted waves:

       |S11|² + |S21|² = 1   and   |S12|² + |S22|² = 1   at EVERY frequency.

   The step is a real discontinuity (|S11| ≈ 0.17–0.43 across the band, so the
   power genuinely splits between reflection and transmission) — this is a far
   stronger statement than mere passivity (≤ 1): with no loss path the column
   power must land exactly on 1. Asserted within 1.5 %.

2. LOSSY passivity. A matched WR-90 straight section with copper
   ``rf.SurfaceImpedance`` walls (σ = 5.8e7 S/m) dissipates a little TE10 power
   in the skin effect, so each incident column power now sits strictly below
   unity yet still bounded (a passive network never creates power):

       0.90 < |S11|² + |S21|² < 1.0.

   Operated in the lower X-band (7–9 GHz, nearer the 6.56 GHz cutoff where the
   conductor attenuation αc is largest) the loss is a clear ~0.3–0.6 %, well
   above the lossless numerical floor (~0.04 %) and far inside the 0.90 floor.

Both meshes stay under the harness DOF budget (< 100 000): the step is
~41 k DOF, the lossy guide ~43 k DOF.

Reference: Pozar, *Microwave Engineering*, §4.3 (the scattering matrix of a
lossless network is unitary; a passive network has ‖S·a‖ ≤ ‖a‖).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case

# WR-90 (X-band); TE10 cutoff ≈ 6.56 GHz.
A, B = 22.86e-3, 10.16e-3

# Lossless H-plane step: input width A, output width A2 (cutoff ≈ 9.99 GHz).
A2 = 15.0e-3
L1, L2 = 18e-3, 18e-3

# Lossy straight section.
L_LOSSY = 100e-3
SIGMA = 5.8e7                       # copper conductivity, S/m


def _column_powers(s):
    """The two incident-port column powers of a 2-port S-matrix:
    (|S11|²+|S21|², |S12|²+|S22|²). Each is the total scattered power for a
    unit wave incident on that port; = 1 (lossless), ≤ 1 (passive)."""
    s = np.asarray(s)
    return float(np.sum(np.abs(s[:, 0]) ** 2)), float(np.sum(np.abs(s[:, 1]) ** 2))


@pytest.mark.slow
@case.phenomenon
def test_lossless_step_unitarity():
    # WR-90 H-plane step, all PEC + air → lossless reciprocal 2-port.
    g = case.geometry(maxh=rf.lambda_maxh(f_max=13e9))
    box1 = g.box(A, B, L1, position=(-A / 2, -B / 2, 0.0), material=rf.Air())
    box2 = g.box(A2, B, L2, position=(-A2 / 2, -B / 2, L1), material=rf.Air())
    g.fragment(box1, box2)

    rf.RectWaveguidePort(box1.faces.min(axis="z"))
    rf.RectWaveguidePort(box2.faces.max(axis="z"), width=A2, height=B)

    # PEC the lateral walls of both sections plus the step "ring" — the part of
    # box1's top face that sits outside the narrow guide's footprint. The
    # central A2×B overlap is an interior air↔air interface (the fields pass
    # through it), so it must stay un-PEC'd or the output port is walled off.
    ring = box1.faces.where(
        lambda c, _: abs(c[2] - L1) < 1e-9 and (abs(c[0]) > A2 / 2 or abs(c[1]) > B / 2)
    )
    rf.PEC(box1.faces.min(axis="x"), box1.faces.max(axis="x"),
           box1.faces.min(axis="y"), box1.faces.max(axis="y"),
           box2.faces.min(axis="x"), box2.faces.max(axis="x"),
           box2.faces.min(axis="y"), box2.faces.max(axis="y"),
           ring)

    # Band sits above the narrow-guide cutoff (≈9.99 GHz) so both ports carry a
    # propagating TE10; the step gives a strong, frequency-dependent reflection.
    freqs = np.linspace(10.5e9, 12.5e9, 6)
    prob, res = case.sweep(g, freqs)

    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    # The step is a genuine discontinuity: power really does split.
    assert s11.max() > 0.15, f"step too weak, |S11|max={s11.max():.3f}"
    assert s21.min() > 0.5, f"step opaque, |S21|min={s21.min():.3f}"

    # Headline: a lossless reciprocal 2-port has a UNITARY S-matrix, so each
    # incident column power equals 1 at every frequency, within 1.5 %.
    worst = 0.0
    for i, f in enumerate(freqs):
        c0, c1 = _column_powers(res.sparams[i])
        worst = max(worst, abs(c0 - 1.0), abs(c1 - 1.0))
        assert abs(c0 - 1.0) < 0.015, (
            f"|S11|²+|S21|²={c0:.4f} at {f/1e9:.2f} GHz (not unitary)")
        assert abs(c1 - 1.0) < 0.015, (
            f"|S12|²+|S22|²={c1:.4f} at {f/1e9:.2f} GHz (not unitary)")
    assert worst < 0.015, f"worst unitarity deviation {worst:.5f}"


@pytest.mark.slow
@case.phenomenon
def test_lossy_section_passivity():
    # Matched WR-90 straight section with copper skin-effect walls → passive,
    # mildly lossy 2-port.
    g = case.geometry(maxh=rf.lambda_maxh(f_max=9e9))
    air = g.box(A, B, L_LOSSY, position=(-A / 2, -B / 2, 0.0), material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    # Leontovich skin-effect BC (Z_s = (1+j)R_s) on the four side walls.
    rf.SurfaceImpedance(*air.faces.unassigned, conductivity=SIGMA)

    # Lower X-band, above the 6.56 GHz cutoff — αc is largest here, so the
    # (still small) loss is comfortably above the lossless numerical floor.
    freqs = np.array([7.0e9, 7.5e9, 8.0e9, 8.5e9, 9.0e9])
    prob, res = case.sweep(g, freqs)

    s11 = np.abs(res.sparams[:, 0, 0])
    assert s11.max() < 0.01, f"|S11| rose to {s11.max():.4f} (not matched)"

    # Passive but lossy: each incident column power is < 1 (power dissipated in
    # the walls) yet bounded — a passive network never amplifies, and copper is
    # a good conductor so the loss stays modest (well above the 0.90 floor).
    worst_loss = 0.0
    for i, f in enumerate(freqs):
        c0, c1 = _column_powers(res.sparams[i])
        worst_loss = max(worst_loss, 1.0 - c0, 1.0 - c1)
        for c, name in ((c0, "|S11|²+|S21|²"), (c1, "|S12|²+|S22|²")):
            assert c < 1.0, f"{name}={c:.5f} at {f/1e9:.1f} GHz (not passive)"
            assert c > 0.90, f"{name}={c:.5f} at {f/1e9:.1f} GHz (loss too large)"
    # The loss is real, not numerical noise: clearly above the lossless floor.
    assert worst_loss > 1e-3, (
        f"loss {worst_loss:.5f} at lossless numerical floor (walls inert?)")
