# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""First-order ABC termination — bounded, predictable modal reflection.

A WR-90 (X-band) section is driven by a single RectWaveguidePort at z=min and
terminated at z=max by a first-order (Sommerfeld) absorbing boundary instead of
a second port. Such an ABC enforces

    n × (∇×E) + j k0 (n × (n × E)) = 0,

i.e. it is exactly matched to a wave whose phase constant equals the free-space
k0. The TE10 mode, however, propagates with β < k0, so the ABC's assumed k0
mismatches the modal β and a residual reflection survives:

    |Γ_abc| = (k0 − β) / (k0 + β),   β = β_TE10(f).

Well above cutoff β → k0 and |Γ| → 0; near cutoff β → 0 and |Γ| → 1. This is
the textbook first-order-ABC modal reflection (a self-consistency physics gate:
the measured |S11| must track this closed form, no external solver involved).

The solver's ABC γ-coefficient is j·k0·neff with neff = 1 for air
(crates/rapidfem-fd/src/waveguide.rs), so the j·k0 reference above is exact.

Reference: Jin, *The Finite Element Method in Electromagnetics*, §9 (ABCs);
Pozar, *Microwave Engineering*, §3.3 (rectangular waveguide).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# WR-90 (X-band, 8.2–12.4 GHz)
A, B = 22.86e-3, 10.16e-3
LENGTH = 30.0e-3       # driven air section, ABC on its far face


def abc_modal_reflection(f: float, a: float, b: float) -> float:
    """Magnitude of the first-order ABC (γ = j·k0) reflection for the TE10
    mode: |Γ| = (k0 − β)/(k0 + β), the ABC-assumed-k0 vs modal-β mismatch."""
    beta = ref.rect_beta(f, a, b, m=1, n=0)
    k0 = ref.k0(f)
    return abs((k0 - beta) / (k0 + beta))


@pytest.mark.slow
@case.phenomenon
def test_abc_first_order_modal_reflection():
    g = case.geometry(maxh=rf.lambda_maxh(f_max=12e9))
    air = g.box(A, B, LENGTH, position=(-A / 2, -B / 2, 0.0), material=rf.Air())

    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.ABC(air.faces.max(axis="z"))
    # Side walls default to PEC (unassigned exterior faces); mark explicitly.
    rf.PEC(*air.faces.unassigned)

    freqs = np.linspace(8.5e9, 11.5e9, 7)
    prob, res = case.sweep(g, freqs)

    # Single driven port -> 1x1 scattering matrix.
    assert res.sparams.shape == (len(freqs), 1, 1)

    s11 = np.abs(res.sparams[:, 0, 0])
    analytic = np.array([abc_modal_reflection(f, A, B) for f in freqs])

    # The first-order ABC reflection must track |(k0−β)/(k0+β)| across the band.
    rel = np.abs(s11 - analytic) / analytic
    assert rel.max() < 0.25, (
        f"|S11| off ABC modal reflection by {rel.max():.1%}\n"
        f"  f (GHz):   {np.array2string(freqs / 1e9, precision=2)}\n"
        f"  measured:  {np.array2string(s11, precision=4)}\n"
        f"  analytic:  {np.array2string(analytic, precision=4)}"
    )

    # Trend: above cutoff β → k0, so reflection falls with frequency.
    assert np.all(np.diff(s11) < 0), (
        f"|S11| not monotonically decreasing: {np.array2string(s11, precision=4)}"
    )

    # Passivity: a dissipative ABC cannot scatter back more power than incident.
    for i in range(len(freqs)):
        assert case.passivity(res.sparams[i]) < 1.02
