# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Proximity effect — opposing currents crowd onto the FACING surfaces.

Two nearby conductors carrying *opposing* currents (the odd / differential mode
of a coupled pair) push their current onto the surfaces that face each other.
Between the conductors the magnetic fields of the two anti-parallel currents
reinforce, so the tangential H at the surface — and hence the surface current
density ``Js = n x H`` — is larger on the inner (facing) surfaces than on the
outer surfaces. In the EVEN / common mode (same-direction currents) the gap
fields instead cancel and the current is pushed onto the OUTER surfaces. That
sign flip between the two modes is the proximity effect, the multi-conductor
companion of the skin effect.

Geometry: two parallel rectangular PEC bars (a coupled pair) raised above the
box floor, each grounded at its far end and fed at its near end by a vertical
lumped delta-gap to the floor (a PEC ground plane). A 3-D PEC bar is built the
way the iris-filter test builds its iris plates — an air box whose faces are
all tagged PEC, so the field is excluded from its interior and it acts as a
solid conductor obstacle with distinct inner/outer surfaces. Everything sits in
an air-filled shielding box whose unassigned outer walls default to PEC.

The two feeds are mirror-symmetric in x (port 0 → bar A, port 1 → bar B), so a
single solve yields both modes by superposition of the per-port fields:

    H_odd  = H(driven A) - H(driven B)     (differential: opposing currents)
    H_even = H(driven A) + H(driven B)     (common: same-direction currents)

Because both modes are sampled at the *same* mesh nodes, the inner/outer
comparison carries no sampling-distance bias — only the excitation differs. We
sample |H_tangential| in the gap (adjacent to both inner faces) and in thin
shells outside the outer faces, and assert:

  * ODD mode crowds onto the inner faces:  mean|H_gap| / mean|H_outer| > 1.3,
  * EVEN mode crowds onto the outer faces:  that ratio drops below 1,

so the same geometry flips its current distribution with the drive mode. This
phenomenon has no clean closed form at meshable resolution, so the honest
assertion is this RATIO contrast, which demonstrates current crowding
unambiguously.

Reference: Pozar, *Microwave Engineering*, §1.7 (skin / proximity effect);
Jackson, *Classical Electrodynamics*, §5 (current between parallel conductors).
"""
import numpy as np
import pytest

import rapidfem as rf
from harness import case

mm = 1e-3

# ── Coupled-pair geometry ───────────────────────────────────────────────────
# Bars run along y; they are separated along x; height is z. The floor (z = 0)
# is the PEC ground each feed/short references.
GAP = 1.0 * mm          # inner-face-to-inner-face spacing (small -> strong)
BAR_W = 1.0 * mm        # bar width   (x)
BAR_H = 2.5 * mm        # bar height  (z)
BAR_L = 9.0 * mm        # bar length  (y)
XC = (GAP + BAR_W) / 2.0   # |x| of each bar centre  -> inner faces at +-GAP/2

# Shielding box and bar placement.
BOX_X = 9.0 * mm
BOX_Y = 13.0 * mm
BOX_Z = 7.0 * mm
Y0 = 2.0 * mm                 # near-end recess (feed plane) and far-end inset
Z0 = 2.0 * mm                 # bar floor: arms raised above the ground plane

FREQ = 5.0e9                  # box is electrically small (~lambda/6): quasi-TEM

# Per-feature mesh sizes: fine on the bars and feeds, coarse in the open air.
BAR_MAXH = 0.4 * mm
FEED_MAXH = 0.3 * mm

# ── Surface-shell sampling windows ──────────────────────────────────────────
MARGIN = 0.4 * mm       # stay clear of the bar ends / top / bottom edges
OUTER_DEPTH = 1.2 * mm  # how far out from the outer faces to gather air nodes
FACE_TOL = 0.05 * mm    # drop nodes sitting exactly on a PEC face (|H|->0 there)


def _pec_bar(g, w, d, h, position):
    """A solid PEC bar: an air box whose faces are tagged PEC later (same trick
    as the iris-filter plates). Returns the GeoObject for fragmenting."""
    return g.box(w, d, h, position=position, material=rf.Air(), maxh=BAR_MAXH)


def _build_pair(g):
    """Two grounded PEC bars, each fed by a vertical lumped gap to the floor.

    Returns the two bar GeoObjects (their faces become the inner/outer metal
    surfaces). Ports: 0 -> bar A (x<0), 1 -> bar B (x>0), mirror-symmetric."""
    air = g.box(BOX_X, BOX_Y, BOX_Z,
                position=(-BOX_X / 2, 0.0, 0.0), material=rf.Air())

    # Two arms along y at x = -XC and +XC (centres); inner faces at +-GAP/2.
    arm_a = _pec_bar(g, BAR_W, BAR_L, BAR_H, position=(-XC - BAR_W / 2, Y0, Z0))
    arm_b = _pec_bar(g, BAR_W, BAR_L, BAR_H, position=(+XC - BAR_W / 2, Y0, Z0))

    # Vertical plates (normal y) bridging arm-bottom to the floor: the near ones
    # are lumped feed gaps, the far ones are PEC shorts, so each arm + its two
    # posts + the ground plane form a current loop carrying the arm current.
    def _post(x_left, y):
        return g.xz_plate(BAR_W, Z0, position=(x_left, y, 0.0), maxh=FEED_MAXH)

    feed_a = _post(-XC - BAR_W / 2, Y0)
    feed_b = _post(+XC - BAR_W / 2, Y0)
    short_a = _post(-XC - BAR_W / 2, Y0 + BAR_L)
    short_b = _post(+XC - BAR_W / 2, Y0 + BAR_L)

    g.fragment(air, arm_a, arm_b, feed_a, feed_b, short_a, short_b)

    # Port 0 feeds bar A, port 1 feeds bar B (mirror images); the shorts and the
    # bar faces are metal; the box's outer walls are the default-PEC shield.
    rf.LumpedPort(feed_a, direction=(0, 0, 1), z0=50.0)
    rf.LumpedPort(feed_b, direction=(0, 0, 1), z0=50.0)
    rf.PEC(*arm_a.faces, *arm_b.faces, *short_a.faces, *short_b.faces)
    return arm_a, arm_b


def _h_tangential(H):
    """|H_tangential| to the x-normal inner/outer faces = sqrt(Hy^2 + Hz^2).
    The surface current density is Js = n x H, so this is the crowding proxy."""
    return np.sqrt(np.abs(H[:, 1]) ** 2 + np.abs(H[:, 2]) ** 2)


def _shell_masks(coords):
    """Masks for nodes in the gap (adjacent to both inner faces) and in thin
    air shells outside the two outer faces, within the bars' y/z footprint."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    foot = (
        (y > Y0 + MARGIN) & (y < Y0 + BAR_L - MARGIN)
        & (z > Z0 + MARGIN) & (z < Z0 + BAR_H - MARGIN)
    )
    inner = foot & (np.abs(x) < GAP / 2 - FACE_TOL)         # the gap (facing)
    outer_face = XC + BAR_W / 2                              # |x| of outer faces
    outer = foot & (
        ((x < -outer_face - FACE_TOL) & (x > -outer_face - OUTER_DEPTH))
        | ((x > +outer_face + FACE_TOL) & (x < +outer_face + OUTER_DEPTH))
    )
    return inner, outer


@pytest.mark.slow
@case.phenomenon
def test_proximity_effect_inner_vs_outer_crowding():
    g = case.geometry(maxh=rf.lambda_maxh(f_max=FREQ * 1.2))
    _build_pair(g)

    prob, res = case.sweep(g, np.array([FREQ]))
    assert prob.n_dofs < case.DOF_BUDGET, f"{prob.n_dofs} DOF over budget"

    # Mirror symmetry validates the even/odd decomposition: S00≈S11, S01≈S10.
    s = res.sparams[0]
    assert abs(s[0, 0] - s[1, 1]) < 0.05, f"asymmetric self-terms: {s[0,0]} {s[1,1]}"
    assert abs(s[0, 1] - s[1, 0]) < 0.05, f"non-reciprocal coupling: {s[0,1]} {s[1,0]}"

    # Per-port H, then the two modes by superposition (same nodes, no distance
    # bias): odd = opposing currents, even = same-direction currents.
    h_a = prob.h_field_at_nodes(res, 0, 0)      # (n_nodes, 3) complex, A/m
    h_b = prob.h_field_at_nodes(res, 0, 1)
    coords = prob.mesh_nodes                     # (n_nodes, 3) float, m

    inner, outer = _shell_masks(coords)
    assert inner.sum() > 25, f"only {inner.sum()} gap (inner) nodes"
    assert outer.sum() > 8, f"only {outer.sum()} outer-shell nodes"

    def ratio(H):
        ht = _h_tangential(H)
        return ht[inner].mean(), ht[outer].mean()

    odd_in, odd_out = ratio(h_a - h_b)
    even_in, even_out = ratio(h_a + h_b)
    odd_ratio = odd_in / odd_out
    even_ratio = even_in / even_out

    print(f"\nproximity effect: n_dofs={prob.n_dofs}, n_tets={prob.n_tets}")
    print(f"  gap (inner) nodes={inner.sum()}  outer-shell nodes={outer.sum()}")
    print(f"  ODD : inner|H|={odd_in:.3g}  outer|H|={odd_out:.3g}  ratio={odd_ratio:.2f}")
    print(f"  EVEN: inner|H|={even_in:.3g}  outer|H|={even_out:.3g}  ratio={even_ratio:.2f}")

    # ODD mode (opposing currents): current crowds onto the inner/facing faces.
    assert odd_ratio > 1.3, (
        f"no inner crowding in the odd mode: inner/outer |H_tan| {odd_ratio:.2f} "
        f"<= 1.3 (inner {odd_in:.3g}, outer {odd_out:.3g} A/m)")

    # EVEN mode (same-direction currents): current crowds onto the OUTER faces,
    # so the gap field collapses and the ratio drops below unity.
    assert even_ratio < 1.0, (
        f"even mode did not push current outward: inner/outer {even_ratio:.2f} "
        f">= 1.0 (inner {even_in:.3g}, outer {even_out:.3g} A/m)")

    # The contrast is the clincher: the SAME geometry flips its current
    # distribution purely with the drive mode (mesh-symmetry-free control).
    assert odd_ratio > 2.0 * even_ratio, (
        f"odd/even contrast too weak: odd {odd_ratio:.2f} vs even {even_ratio:.2f}")
