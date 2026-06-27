# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Scale invariance — the same electrical problem gives the same S-parameters.

rapidfem non-dimensionalizes geometry by a characteristic length L0 before
assembly ("lever ④"), so a structure must produce IDENTICAL S-parameters
whether it is meshed in metres, millimetres, or nanometres. The physics anchor:
a structure of size L at frequency f is electrically identical to one of size
L·s driven at f/s (same number of wavelengths across every feature). So solving
``structure(scale=s)`` at frequencies ``f/s`` must reproduce
``structure(scale=1)`` at ``f`` to within numerical noise.

Two phenomena are locked here:

  (1) a mismatched 2-port (an air→dielectric step in a WR-90 guide). The whole
      complex S-matrix — reflection AND transmission — is scale invariant.
  (2) a PML-terminated guide. The reflection off the coordinate-stretched PML
      slab is scale invariant. This is the regression gate for the lever-④ PML
      fix: a PML stretch that is NOT L0-normalized makes the stretched-layer
      impedance depend on absolute length, so |S11| would diverge between the
      two scales. The slab is deliberately UNDER-tuned (small delta_max) so the
      reflection is a robust ~0.5 — not a well-matched absorber's near-zero
      floor, which is a near-cancellation quantity dominated by mesh
      discretization noise and so unfit for a tight relative comparison.

Reference: self-consistency under coordinate scaling (no external solver). See
Pozar, *Microwave Engineering*, §3.3; Jin, *FEM in Electromagnetics*, §9 (PML).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case

# WR-90 (X-band) cross-section and lengths at the reference scale (scale = 1).
A, B = 22.86e-3, 10.16e-3
F_MAX = 12.0e9

# Two scales spanning three orders of magnitude. Both produce the same mesh
# topology and the same DOF count, because maxh tracks the (scaled) wavelength.
SCALE_REF = 1.0      # ~cm features, GHz drive
SCALE_SMALL = 1e-3   # ~µm features, THz drive — same electrical problem


def _maxh(scale, er_max=1.0):
    """Wavelength mesh cap for a structure scaled by `scale`. Scaling lengths by
    `scale` and frequencies by `1/scale` scales the wavelength (hence maxh) by
    `scale`, so the meshed problem is identical up to a global coordinate scale."""
    return rf.lambda_maxh(f_max=F_MAX / scale, er_max=er_max)


def _max_rel_sparam_diff(s_a, s_b, floor=0.05):
    """Worst-case relative difference between two S-parameter sweeps of equal
    shape. Entries with |S| below `floor` are compared against `floor` so a
    near-zero reflection term cannot blow the ratio up to meaninglessness."""
    s_a = np.asarray(s_a)
    s_b = np.asarray(s_b)
    denom = np.maximum(np.abs(s_a), floor)
    return float(np.max(np.abs(s_a - s_b) / denom))


# ── case 1: mismatched 2-port (air → dielectric step) ──────────────────────
STEP_LENGTH = 30.0e-3      # total guide length at scale = 1
STEP_ER = 2.2              # PTFE-like fill in the second half


def _build_dielectric_step(g, scale):
    """Half air / half εr WR-90 guide, two ports. The internal air→dielectric
    interface is an impedance step, so both |S11| and |S21| are non-trivial and
    the full complex S-matrix is a meaningful scale-invariance witness."""
    a, b, half = A * scale, B * scale, STEP_LENGTH * scale / 2.0
    air = g.box(a, b, half, position=(-a / 2, -b / 2, 0.0), material=rf.Air())
    diel = g.box(a, b, half, position=(-a / 2, -b / 2, half),
                 material=rf.Dielectric(er=STEP_ER, tand=0.0))
    g.fragment(air, diel)
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(diel.faces.max(axis="z"), er=STEP_ER)
    rf.PEC(*air.faces.outer.unassigned, *diel.faces.outer.unassigned)


def _solve_dielectric_step(scale):
    g = case.geometry(maxh=_maxh(scale, er_max=STEP_ER))
    _build_dielectric_step(g, scale)
    freqs = np.linspace(8.0e9, 11.0e9, 5) / scale
    prob, res = case.sweep(g, freqs)
    return prob, res.sparams


@pytest.mark.slow
@case.phenomenon
def test_dielectric_step_sparams_scale_invariant():
    prob_ref, s_ref = _solve_dielectric_step(SCALE_REF)
    prob_small, s_small = _solve_dielectric_step(SCALE_SMALL)

    # Shapes must match: same 2-port problem at both scales.
    assert s_ref.shape == s_small.shape == (5, 2, 2)

    rel = _max_rel_sparam_diff(s_ref, s_small)
    assert rel < 1e-3, (
        f"dielectric-step S-params diverge across scales: max rel diff {rel:.2e}\n"
        f"scale={SCALE_REF}:  S11={np.abs(s_ref[:, 0, 0])}\n"
        f"                    S21={np.abs(s_ref[:, 1, 0])}\n"
        f"scale={SCALE_SMALL}: S11={np.abs(s_small[:, 0, 0])}\n"
        f"                    S21={np.abs(s_small[:, 1, 0])}"
    )

    # The step is a real mismatch, not a trivial matched line: the comparison
    # has something to lock onto.
    assert np.abs(s_ref[:, 0, 0]).max() > 0.05


# ── case 2: PML-terminated guide (locks the lever-④ PML fix) ────────────────
PML_INNER = 40.0e-3        # driven air section at scale = 1
PML_T = 15.0e-3            # PML slab thickness at scale = 1
# Deliberately small delta_max: the slab absorbs only partially so |S11| lands
# at a robust ~0.5 instead of a near-zero, mesh-noise-limited absorber floor.
# The reflection still comes entirely from the coordinate stretch (with no PML
# the back wall is PEC and |S11| = 1), so its value directly probes the stretch
# profile — the exact thing lever-④ must L0-normalize.
PML_DELTA_MAX = 0.3


def _build_pml_guide(g, scale):
    """WR-90 air section driven at z=min, terminated by a coordinate-stretched
    PML slab. A correctly L0-normalized stretch reflects identically at any
    scale; an un-normalized stretch makes the stretched-layer impedance — hence
    |S11| — depend on absolute length."""
    a, b = A * scale, B * scale
    l_inner, t = PML_INNER * scale, PML_T * scale
    inner = g.box(a, b, l_inner, position=(-a / 2, -b / 2, 0.0), material=rf.Air())
    pml = g.box(a, b, t, position=(-a / 2, -b / 2, l_inner),
                material=rf.Air(), maxh=2 * _maxh(scale))
    g.fragment(inner, pml)
    rf.RectWaveguidePort(inner.faces.min(axis="z"))
    rf.PML(pml, direction=(0, 0, 1), inner_face=l_inner, thickness=t,
           exponent=1.5, delta_max=PML_DELTA_MAX)
    rf.PEC(*inner.faces.outer.unassigned, *pml.faces.outer.unassigned)


def _solve_pml_guide(scale):
    g = case.geometry(maxh=_maxh(scale))
    _build_pml_guide(g, scale)
    freqs = np.linspace(9.0e9, 11.0e9, 5) / scale
    prob, res = case.sweep(g, freqs)
    return prob, res.sparams


@pytest.mark.slow
@case.phenomenon
def test_pml_termination_scale_invariant():
    prob_ref, s_ref = _solve_pml_guide(SCALE_REF)
    prob_small, s_small = _solve_pml_guide(SCALE_SMALL)

    assert s_ref.shape == s_small.shape == (5, 1, 1)

    s11_ref = np.abs(s_ref[:, 0, 0])
    s11_small = np.abs(s_small[:, 0, 0])

    # The PML must actually be engaged: a real partial reflection (the stretch
    # is absorbing), not full PEC reflection (1.0) and not the noise floor.
    assert 0.1 < s11_ref.max() < 0.95, f"PML not engaged at ref scale: {s11_ref}"
    assert 0.1 < s11_small.max() < 0.95, f"PML not engaged at small scale: {s11_small}"

    # The lock: the stretched-layer reflection is identical across three orders
    # of magnitude in physical scale. Holds only if the PML stretch is
    # L0-normalized (lever-④); an un-normalized stretch shifts |S11| grossly.
    rel = _max_rel_sparam_diff(s_ref, s_small)
    assert rel < 1e-3, (
        f"PML |S11| is NOT scale invariant: max rel diff {rel:.2e}\n"
        f"scale={SCALE_REF}:  |S11|={s11_ref}\n"
        f"scale={SCALE_SMALL}: |S11|={s11_small}\n"
        "If this fails, the PML stretch is not L0-normalized (lever-④ bug)."
    )
