# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""WR-90 iris-coupled cavity filter — resonant transmission of a lossless 2-port.

Two thin inductive irises (PEC strips that partially block the guide
cross-section, leaving a centred aperture) inside a WR-90 section form a
single resonant cavity. Each iris is a strong, reactive (lossless) reflector,
so the structure is the textbook iris-coupled bandpass resonator:

  * AT resonance the cavity between the irises is matched — the two reflections
    cancel and the TE10 mode tunnels through, |S21| → 1.
  * OFF resonance the irises simply reflect, |S11| → 1 and |S21| collapses.

That contrast is the *filter shape*: a transmission passband sitting between
reflective skirts. Because the walls and irises are all PEC and the fill is
air, the network is lossless and reciprocal, so its scattering matrix is
unitary and the incident power is conserved at every frequency:

    |S11|² + |S21|² = 1.

Phenomena exercised: TE10 propagation, iris (inductive) reflection, cavity
resonance / tunnelling, lossless unitarity (passivity), filter selectivity.

Reference: Pozar, *Microwave Engineering*, §6.4 (waveguide resonators) and
§8.x (direct-coupled cavity filters); the iris-coupled cavity is the canonical
single-pole waveguide bandpass section.
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case

# WR-90 (X-band); TE10 cutoff ≈ 6.56 GHz.
A, B = 22.86e-3, 10.16e-3

# Two identical inductive irises with a centred aperture, separated by ~λg/2
# at the design frequency so the enclosed length resonates in the band.
APERTURE = 9.0e-3      # open slot width (x) left by each iris
IRIS_T = 1.0e-3        # iris plate thickness (z)
SPACING = 16.0e-3      # gap between the two iris plates (cavity length)
END_LEN = 9.0e-3       # input / output guide lengths feeding the cavity

L = END_LEN + IRIS_T + SPACING + IRIS_T + END_LEN


def _add_iris(g, air, z0, aperture):
    """Two PEC strips along ±x at z∈[z0, z0+IRIS_T] leaving a centred slot of
    width `aperture`. Returns the dummy strip volumes (their faces become PEC
    through the fragment)."""
    strip_w = (A - aperture) / 2.0
    vols = []
    for side in (-1, +1):
        x0 = -A / 2.0 if side < 0 else aperture / 2.0
        vols.append(
            g.box(strip_w, B, IRIS_T, position=(x0, -B / 2.0, z0),
                  material=rf.Air())
        )
    return vols


@pytest.mark.slow
@case.phenomenon
def test_iris_coupled_cavity_filter():
    g = case.geometry(maxh=rf.lambda_maxh(f_max=12e9, per_lambda=10))
    air = g.box(A, B, L, position=(-A / 2, -B / 2, 0.0), material=rf.Air())

    z1 = END_LEN
    z2 = END_LEN + IRIS_T + SPACING
    iris_vols = _add_iris(g, air, z1, APERTURE) + _add_iris(g, air, z2, APERTURE)
    g.fragment(air, *iris_vols)

    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    # Everything still unassigned after the ports — outer guide walls AND the
    # iris plate faces — is metal.
    rf.PEC(*air.faces.unassigned)

    # Band spanning the cavity resonance (above the 6.56 GHz cutoff).
    freqs = np.linspace(8.0e9, 12.0e9, 17)
    prob, res = case.sweep(g, freqs)

    s11 = np.abs(res.sparams[:, 0, 0])
    s21 = np.abs(res.sparams[:, 1, 0])

    # ── Conservation: lossless ⇒ |S11|²+|S21|² ≈ 1 at every frequency ───────
    worst = 0.0
    for i, f in enumerate(freqs):
        col = s11[i] ** 2 + s21[i] ** 2
        worst = max(worst, abs(col - 1.0))
        assert abs(col - 1.0) < 0.02, (
            f"|S11|²+|S21|²={col:.4f} at {f/1e9:.2f} GHz (not conserved)")
        assert case.passivity(res.sparams[i]) <= 1.02

    # ── Filter shape: a real passband with reflective skirts ────────────────
    assert s21.max() - s21.min() > 0.4, (
        f"too flat: |S21| span {s21.max() - s21.min():.3f} "
        f"(max {s21.max():.3f}, min {s21.min():.3f})")
    assert s21.max() > 0.7, f"no passband: max|S21|={s21.max():.3f}"
    assert s11.max() > 0.7, f"no reflective skirt: max|S11|={s11.max():.3f}"

    print(f"\niris filter: n_dofs={prob.n_dofs}, n_tets={prob.n_tets}, "
          f"worst conservation err={worst:.4f}")
    for f, a, b in zip(freqs, s11, s21):
        print(f"  {f/1e9:5.2f} GHz  |S11|={a:.3f}  |S21|={b:.3f}")
