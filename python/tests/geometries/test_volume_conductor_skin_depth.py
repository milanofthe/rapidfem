# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Skin effect in a finite-conductivity VOLUME conductor — exponential decay.

A TE10 wave travelling down a rectangular guide is incident on a slab of good
(but finite) bulk conductor that fills the cross-section. Inside a good
conductor the field does not propagate, it diffuses: the amplitude falls off
exponentially from the surface,

    |E(depth)| = |E(0)| * exp(-depth / delta),   delta = 1/sqrt(pi f mu sigma).

The conductivity and frequency are chosen so the skin depth is ~0.8 mm — large
enough that the conductor region can be meshed (sub-delta tets) and the
exponential is actually resolved, rather than the tens-of-nanometre delta of
real copper at GHz. This test drives the structure with a single waveguide
port, samples the solved E-field along the depth axis inside the conductor with
``prob.field_at_nodes`` + ``prob.mesh_nodes``, fits log|E| vs depth, and checks
the recovered decay rate against the analytic 1/delta.

Phenomena exercised: bulk-conductor field exclusion, skin-effect penetration
depth, volumetric Ohmic medium (J = sigma E).

Reference: Pozar, *Microwave Engineering*, §1.7 (plane waves in a good
conductor / skin depth); Jackson, *Classical Electrodynamics*, §8.1.
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case, references as ref

# Guide cross-section. a sets the TE10 cutoff (fc = c/2a ~ 6.56 GHz); b is kept
# thin to keep the fine conductor mesh under the DOF budget (TE10 is
# b-independent). The drive frequency sits comfortably above cutoff.
A, B = 22.86e-3, 1.0e-3
FREQ = 8.0e9                       # well above the 6.56 GHz TE10 cutoff

# Finite-conductivity bulk conductor. sigma is moderate on purpose: it makes the
# skin depth meshable while still being a *good* conductor (sigma / (omega eps0)
# ~ 110 at 8 GHz, loss tangent >> 1, so the field genuinely decays in ~delta).
SIGMA = 50.0                       # S/m  -> delta ~= 0.80 mm at 8 GHz
DELTA = ref.skin_depth(FREQ, SIGMA)

L_AIR = 14.0e-3                    # feed air section (lets the TE10 mode settle)
T_COND = 3.5 * DELTA              # ~3.5 skin depths of conductor to decay into
COND_MAXH = DELTA / 2.5           # ~2.5 tets across one skin depth


def _decay_rate_from_field(depth, e_mag):
    """Fit slope of log|E| vs depth (returns the decay rate 1/delta_measured)."""
    coeffs = np.polyfit(depth, np.log(e_mag), 1)
    return -coeffs[0]


@pytest.mark.slow
@case.phenomenon
def test_volume_conductor_skin_depth():
    # Anchor: skin depth is in the meshable band and the slab is a few delta.
    assert 0.5e-3 < DELTA < 2.0e-3, f"delta = {DELTA*1e3:.3f} mm out of band"

    g = case.geometry(maxh=rf.lambda_maxh(f_max=FREQ * 1.2))
    air = g.box(A, B, L_AIR, position=(-A / 2, -B / 2, 0.0), material=rf.Air())
    cond = g.box(A, B, T_COND, position=(-A / 2, -B / 2, L_AIR),
                 material=rf.Conductor(conductivity=SIGMA), maxh=COND_MAXH)
    g.fragment(air, cond)

    # Single drive port at the air inlet; everything else is metal. The shared
    # air/conductor interface stays untagged (field continuity); the conductor's
    # far face is PEC (a thick conductor backstop — the field has decayed to
    # ~e^-4 there, so the reflection is negligible).
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.PEC(*air.faces.outer.unassigned, *cond.faces.outer.unassigned)

    prob, res = case.sweep(g, np.array([FREQ]))
    assert prob.n_dofs < case.DOF_BUDGET, f"{prob.n_dofs} DOF over budget"

    E = prob.field_at_nodes(res, 0, 0)           # (n_nodes, 3) complex, V/m
    coords = prob.mesh_nodes                      # (n_nodes, 3) float, m
    z = coords[:, 2]
    e_mag = np.linalg.norm(np.abs(E), axis=1)

    # Nodes inside the conductor, indexed by depth from the entry face.
    inside = z > L_AIR + 1e-9
    depth = z[inside] - L_AIR
    mag = e_mag[inside]
    assert inside.sum() > 200, f"only {inside.sum()} nodes in the conductor"

    # Bin by depth and take the cross-sectional PEAK |E| in each slice (the peak
    # of the transverse mode profile is depth-independent in shape, so its
    # envelope isolates the pure exp(-depth/delta) decay). Drop the very first
    # bin: the abrupt interface seeds short-range evanescent transverse modes.
    n_bins = 12
    edges = np.linspace(0.0, T_COND, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    peak = np.array([
        mag[(depth >= lo) & (depth < hi)].max()
        if np.any((depth >= lo) & (depth < hi)) else np.nan
        for lo, hi in zip(edges[:-1], edges[1:])
    ])

    # Fit only the clean exponential interior: skip the first ~0.3 delta (the
    # interface seeds short-range evanescent transverse modes) and the last
    # ~1 delta (the PEC backstop forces tangential E -> 0, a null that would
    # steepen the slope). The window in between is pure exp(-depth/delta).
    fit = (~np.isnan(peak)) & (centres > 0.3 * DELTA) & (centres < T_COND - DELTA)
    assert fit.sum() >= 5, f"only {fit.sum()} usable depth bins for the fit"

    rate = _decay_rate_from_field(centres[fit], peak[fit])
    delta_meas = 1.0 / rate

    # The recovered penetration depth must match the analytic skin depth. A
    # tetrahedral mesh resolving an exponential over a handful of elements is
    # coarse, so 25 % is the honest tolerance.
    rel = abs(delta_meas - DELTA) / DELTA
    print(f"\nsigma={SIGMA} S/m  f={FREQ/1e9:.1f} GHz  delta_analytic={DELTA*1e3:.3f} mm"
          f"  delta_measured={delta_meas*1e3:.3f} mm  rel_err={rel*100:.1f}%"
          f"  n_dofs={prob.n_dofs}")
    assert rel < 0.25, (
        f"skin depth mismatch: measured {delta_meas*1e3:.3f} mm vs analytic "
        f"{DELTA*1e3:.3f} mm ({rel*100:.1f}%)\n"
        f"  depth(mm): {np.array2string(centres[fit]*1e3, precision=2)}\n"
        f"  |E| peak : {np.array2string(peak[fit], precision=3)}"
    )

    # Cross-check: the field at one skin depth is ~1/e of the surface value.
    e_at_surface = peak[fit][0] * np.exp(rate * centres[fit][0])  # extrapolate to 0
    e_at_delta = e_at_surface * np.exp(-1.0)
    obs = np.interp(DELTA, centres[fit], peak[fit])
    assert 0.5 < obs / e_at_delta < 2.0, (
        f"|E(delta)|/|E_fit(delta)| = {obs/e_at_delta:.2f} (expected ~1)"
    )
