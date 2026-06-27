# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""PML termination — a matched absorber reflects almost nothing.

A WR-90 (X-band) section is driven by a single RectWaveguidePort at z=min.
The opposite end is *not* a second port but a short air slab carrying a
coordinate-stretched Perfectly Matched Layer on its outer face. A well-tuned
PML is a reflectionless absorbing termination, so the incident TE10 wave is
swallowed with little reflection: |S11| stays at the absorber floor across the
band. This is a self-consistency physics gate (matched absorber => no reflected
wave), asserted against no external reference.

Reference: Pozar, *Microwave Engineering*, §3.3 (rectangular waveguide);
Jin, *The Finite Element Method in Electromagnetics*, §9 (PML).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case

# WR-90 (X-band, 8.2–12.4 GHz)
A, B = 22.86e-3, 10.16e-3
L_INNER = 40.0e-3      # driven air section
PML_T = 15.0e-3        # PML slab thickness


@pytest.mark.slow
@case.phenomenon
def test_pml_reflectionless_termination():
    g = case.geometry(maxh=rf.lambda_maxh(f_max=12e9))

    inner = g.box(A, B, L_INNER, position=(-A / 2, -B / 2, 0.0),
                  material=rf.Air())
    # PML slab is plain Air; the absorption comes from the PML BC stretch.
    pml = g.box(A, B, PML_T, position=(-A / 2, -B / 2, L_INNER),
                material=rf.Air(), maxh=2 * rf.lambda_maxh(f_max=12e9))
    g.fragment(inner, pml)

    rf.RectWaveguidePort(inner.faces.min(axis="z"))
    rf.PML(pml, direction=(0, 0, 1), inner_face=L_INNER, thickness=PML_T,
           exponent=1.5, delta_max=8.0)
    # Side walls PEC. `.outer` drops the shared inner/PML interface; `.unassigned`
    # drops the driven port face.
    rf.PEC(*inner.faces.outer.unassigned, *pml.faces.outer.unassigned)

    freqs = np.linspace(9.0e9, 11.0e9, 5)
    prob, res = case.sweep(g, freqs)

    s11 = np.abs(res.sparams[:, 0, 0])

    # Single driven port -> 1x1 scattering matrix.
    assert res.sparams.shape == (len(freqs), 1, 1)

    # Matched absorber: reflection at the absorber floor across the band.
    assert s11.max() < 0.05, (
        f"|S11| rose to {s11.max():.4f} "
        f"(spectrum: {np.array2string(s11, precision=4)})"
    )

    # Passivity: the absorbed wave cannot scatter back more power than it
    # carried in.
    for i in range(len(freqs)):
        assert case.passivity(res.sparams[i]) < 1.02
