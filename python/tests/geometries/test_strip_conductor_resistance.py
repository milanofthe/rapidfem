# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Strip resistance R'(f) of a finite-thickness RFIC trace vs a 2D reference.

A 10 um x 3 um strip (sigma = 3.03e7, TopMetal-like) runs 10 um above ground
inside a 50 um x 30 um PEC shield. Between two PEC stubs a lossy section of
length ``LL`` carries the conductor model under test; wave ports sit on the
shield ends. The line is electrically short, so the power balance gives the
series resistance directly: the dissipated fraction is ``R' LL / Z0`` with
``Z0 = 1 / (c C')`` the lossless line impedance. Two lengths are solved and
differenced, which removes the loss at the stub contacts (current
redistribution over roughly one strip width):
``R' = 2 Z0 (a2 - a1) / (LL2 - LL1)`` with ``a = -ln(|S11|^2 + |S21|^2) / 2``.

The reference is the independent quasi-static solver in ``harness.strip2d``
(magnetic diffusion in the same shielded cross-section), exact at DC and grid
converged to 0.1 %.

Two conductor models are checked, the two that ``rfic.build`` chooses from:

  * a volume conductor meshed at the skin depth: the accurate model, used
    wherever the metal is between 1.5 and 10 skin depths thick;
  * a hole whose walls carry a two-sided surface impedance with the
    volume-to-surface thickness ``2V/S`` (``Geometry._hollow``): exact at DC,
    used where the metal is thinner than 1.5 or thicker than 10 skin depths.

Measured before the fix (issue #48): the previous rfic model (SIBC with the
layer thickness on every face, interior meshed as oxide) gave 0.71 to 0.78 of
the reference R' from DC to 10 GHz.
"""
import math

import pytest

import rapidfem as rf
from harness import case
from harness.strip2d import ShieldedStrip

um = 1e-6
W, H = 50 * um, 30 * um             # shield cross-section
X0, Z0 = 20 * um, 10 * um           # strip corner
WS, TS = 10 * um, 3 * um            # strip width, thickness
SIGMA = 3.03e7
STUB = 10 * um                      # PEC stub at each port


def _skin_depth(f):
    return 1.0 / math.sqrt(math.pi * f * 4e-7 * math.pi * SIGMA)


def _reference(t=TS):
    return ShieldedStrip(W=W, H=H, x0=X0, y0=Z0, w=WS, t=t, sigma=SIGMA)


def _resistance(model, f, *, lengths, h_strip, t=TS):
    """R' in ohm/m from the power balance of two solves of different length."""
    a = [_attenuation(model, f, lossy_len=ll, h_strip=h_strip, t=t) for ll in lengths]
    z0 = 1.0 / (299792458.0 * _reference(t).capacitance())
    return 2.0 * z0 * (a[1] - a[0]) / (lengths[1] - lengths[0])


def _attenuation(model, f, *, lossy_len, h_strip, t=TS):
    """-ln(|S11|^2 + |S21|^2) / 2 of one 3D solve (the total loss in Np)."""
    g = rf.Geometry(maxh=8 * um)
    length = 2 * STUB + lossy_len
    air = g.box(W, length, H, position=(0, 0, 0), material=rf.Air())
    if model == "sheet":
        return _sheet_attenuation(g, air, f, lossy_len=lossy_len, h_strip=h_strip, t=t)
    s1 = g.box(WS, STUB, TS, position=(X0, 0, Z0))
    s2 = g.box(WS, STUB, TS, position=(X0, STUB + lossy_len, Z0))
    if model == "volume":
        mid = g.box(WS, lossy_len, TS, position=(X0, STUB, Z0),
                    material=rf.Conductor(conductivity=SIGMA, maxh=h_strip))
    else:
        mid = g.box(WS, lossy_len, TS, position=(X0, STUB, Z0),
                    material=rf.Air(maxh=h_strip))
    mid.name = "metal"
    g.fragment(air, s1, s2, mid)
    g.cut(air, s1, s2)                              # PEC stubs are holes

    eps = 0.2 * um                                  # gmsh bbox padding
    def in_strip(b):
        return (b[0] > X0 - eps and b[3] < X0 + WS + eps
                and b[2] > Z0 - eps and b[5] < Z0 + TS + eps)
    stubs = air.faces.where(lambda c, b: in_strip(b) and (
        b[4] < STUB + eps or b[1] > STUB + lossy_len - eps))
    stub_pec = rf.PEC(*stubs)

    if model == "hollow_sibc":
        for walls, t_eff in g._hollow("metal"):
            # the walls touching the stubs are PEC contacts, not lossy metal
            lateral = walls.where(lambda c, b: b[4] - b[1] > 1 * um)
            rf.SurfaceImpedance(*lateral, conductivity=SIGMA,
                                thickness=t_eff, two_sided=True)

    rf.WavePort(air.faces.min(axis="y"), f0=f, mode_kind="auto", pec=[stub_pec])
    rf.WavePort(air.faces.max(axis="y"), f0=f, mode_kind="auto", pec=[stub_pec])
    prob, res = case.sweep(g, [f])
    s = res.sparams[0]
    return -0.5 * math.log(abs(s[0, 0]) ** 2 + abs(s[1, 0]) ** 2)


def _sheet_attenuation(g, air, f, *, lossy_len, h_strip, t):
    """The strip as a zero-thickness plate at mid-thickness: PEC stubs, a
    lossy ``sheet=True`` section in between."""
    z = Z0 + t / 2
    s1 = g.xy_plate(WS, STUB, position=(X0, 0, z))
    s2 = g.xy_plate(WS, STUB, position=(X0, STUB + lossy_len, z))
    mid = g.xy_plate(WS, lossy_len, position=(X0, STUB, z))
    g.fragment(air, s1, s2, mid)
    stub_pec = rf.PEC(s1, s2)
    rf.SurfaceImpedance(mid, conductivity=SIGMA, thickness=t, sheet=True)
    mid.maxh = h_strip
    rf.WavePort(air.faces.min(axis="y"), f0=f, mode_kind="auto", pec=[stub_pec])
    rf.WavePort(air.faces.max(axis="y"), f0=f, mode_kind="auto", pec=[stub_pec])
    prob, res = case.sweep(g, [f])
    s = res.sparams[0]
    return -0.5 * math.log(abs(s[0, 0]) ** 2 + abs(s[1, 0]) ** 2)


def test_reference_dc_limit_and_tem_consistency():
    ref = _reference()
    # DC: the current is uniform, R' = 1/(sigma w t) exactly
    assert abs(ref.series_impedance(1e6).real * SIGMA * WS * TS - 1.0) < 1e-3
    # high frequency: all current on the surface, L'_ext C' = mu0 eps0 (TEM)
    w = 2 * math.pi * 1e13
    lc = ref.series_impedance(1e13).imag / w * ref.capacitance()
    assert abs(lc / (4e-7 * math.pi * 8.8541878128e-12) - 1.0) < 5e-3


@pytest.mark.slow
@case.phenomenon
@pytest.mark.parametrize("f", [1e9, 3e9])
def test_volume_conductor_matches_reference(f):
    """t/delta = 1.0 and 1.8: the band where rfic meshes the conductor."""
    got = _resistance("volume", f, lengths=(20 * um, 40 * um),
                      h_strip=min(TS / 3, _skin_depth(f) / 1.5))
    want = _reference().series_impedance(f).real
    assert abs(got / want - 1.0) < 0.03, f"R' {got:.1f} vs reference {want:.1f}"


@pytest.mark.slow
@case.phenomenon
@pytest.mark.parametrize("f, lo, hi", [(1e8, 0.97, 1.04), (1e9, 0.97, 1.04), (1e11, 0.80, 0.93)])
def test_hollow_sibc_against_reference(f, lo, hi):
    """t/delta = 0.33, 1.0 and 10.3.

    Below the skin depth the surface model is DC exact. At 10 skin depths it
    misses the loss of the current crowding into the strip edges, which a
    per-face impedance cannot carry: the converged 3D value is about 0.87 of
    the reference (0.87, 0.88, 0.87 at 1.5, 1.0, 0.7 um walls), in line with
    the 2D emulation of the same model (0.83). An earlier coarse gmsh mesh
    happened to land at 0.96.
    """
    got = _resistance("hollow_sibc", f, lengths=(50 * um, 100 * um), h_strip=1.5 * um)
    want = _reference().series_impedance(f).real
    assert lo < got / want < hi, f"R' {got:.1f} vs reference {want:.1f}"


@pytest.mark.slow
@case.phenomenon
@pytest.mark.parametrize("f", [1e8, 1e9])
def test_thin_film_as_embedded_sheet(f):
    """A 0.5 um film (t/delta = 0.06 and 0.17) as a zero-thickness sheet with
    ``sheet=True``: the two faces in parallel carry the full film current."""
    t = 0.5 * um
    got = _resistance("sheet", f, lengths=(50 * um, 100 * um), h_strip=1.5 * um, t=t)
    want = _reference(t).series_impedance(f).real
    assert abs(got / want - 1.0) < 0.03, f"R' {got:.1f} vs reference {want:.1f}"
