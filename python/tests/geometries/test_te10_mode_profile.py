# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""WR-90 TE10 mode profile — transverse E-field follows sin(pi x / a).

A propagating TE10 mode in a rectangular guide has a single transverse field
component (E_y, along the short dimension b) whose amplitude varies as
sin(pi*(x - x0)/a) across the wide dimension a and is ~constant across b. This
test drives a straight WR-90 section at mid-band, samples the solved E-field on
a cross-sectional slab well away from both ports, and checks:

  * |E_y(x)| correlates with the analytic sin(pi*(x-x0)/a) profile,
  * the profile peaks at mid-width and vanishes at the side walls,
  * |E_x|, |E_z| are small compared to |E_y| (TE10 is E_y-dominant).

Reference: Pozar, *Microwave Engineering*, §3.3 (rectangular waveguide modes).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case

# WR-90 (X-band, 8.2-12.4 GHz). a = wide dim (x), b = short dim (y).
A, B = 22.86e-3, 10.16e-3
LENGTH = 30e-3
X0 = -A / 2.0  # box spans x in [X0, X0 + A]
FREQ = 10e9  # mid-band, well above the 6.56 GHz TE10 cutoff


def _sin_profile(x):
    """Analytic transverse amplitude sin(pi*(x - X0)/a), zero at both walls."""
    return np.sin(np.pi * (x - X0) / A)


@pytest.mark.slow
@case.phenomenon
def test_te10_transverse_field_profile():
    g = case.geometry(maxh=rf.lambda_maxh(f_max=12e9))
    air = g.box(A, B, LENGTH, position=(X0, -B / 2.0, 0.0), material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)

    prob, res = case.sweep(g, np.array([FREQ]))
    assert prob.n_dofs < case.DOF_BUDGET

    E = prob.field_at_nodes(res, 0, 0)          # (n_nodes, 3) complex, V/m
    coords = prob.mesh_nodes                     # (n_nodes, 3) float, m
    x, z = coords[:, 0], coords[:, 2]

    # Cross-sectional slab at mid-length, well away from both ports (the ports
    # sit at z=0 and z=LENGTH, so the slab centre is 15 mm = ~0.5 lambda_g away).
    z_mid = LENGTH / 2.0
    slab = np.abs(z - z_mid) < 1.5e-3
    assert slab.sum() > 50, f"only {slab.sum()} nodes in the mid-length slab"

    Ex = np.abs(E[slab, 0])
    Ey = np.abs(E[slab, 1])
    Ez = np.abs(E[slab, 2])
    xs = x[slab]

    # TE10 is E_y-dominant: transverse/longitudinal leakage is small vs |E_y|.
    ey_peak = Ey.max()
    assert Ex.max() / ey_peak < 0.20, f"|Ex|/|Ey| = {Ex.max() / ey_peak:.3f}"
    assert Ez.max() / ey_peak < 0.20, f"|Ez|/|Ey| = {Ez.max() / ey_peak:.3f}"

    # Bin |E_y| by x across the width and compare with the analytic sin profile.
    edges = np.linspace(X0, X0 + A, 13)
    centres = 0.5 * (edges[:-1] + edges[1:])
    binned = np.array([
        Ey[(xs >= lo) & (xs < hi)].mean() if np.any((xs >= lo) & (xs < hi))
        else np.nan
        for lo, hi in zip(edges[:-1], edges[1:])
    ])
    ok = ~np.isnan(binned)
    assert ok.sum() >= 10, f"only {ok.sum()} populated x-bins"

    measured = binned[ok]
    analytic = _sin_profile(centres[ok])
    corr = np.corrcoef(measured, analytic)[0, 1]
    assert corr > 0.95, f"profile correlation {corr:.3f} below 0.95"

    # Peak at mid-width, amplitude collapses toward the side walls.
    peak_x = centres[ok][np.argmax(measured)]
    assert abs(peak_x) < 0.12 * A, f"peak at x={peak_x*1e3:.2f} mm, not mid-width"
    wall = max(measured[0], measured[-1]) / measured.max()
    assert wall < 0.25, f"side-wall amplitude ratio {wall:.3f} too large"
