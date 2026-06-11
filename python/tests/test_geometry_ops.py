"""Geometry builder ops: translate / mirror / copy / array.

Fast checks (no FEM solve) on the declarative description level: centroids
move as requested, copies are independent, arrays are evenly placed, and a
polar array of boxes still meshes (boxes leave the axis-aligned world and
are rewritten as lofts). fillet / chamfer are retired with the gmsh kernel.
"""
from __future__ import annotations

import math

import pytest

import rapidfem as rf

MM = 1e-3


def _mid(obj):
    b = obj.bbox
    return tuple(0.5 * (b[k] + b[k + 3]) for k in range(3))


def test_translate_moves_centroid():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM)
    x0, y0, z0 = _mid(b)
    g.translate(b, dx=2 * MM, dz=-1 * MM)
    x1, y1, z1 = _mid(b)
    assert x1 - x0 == pytest.approx(2 * MM)
    assert y1 - y0 == pytest.approx(0.0, abs=1e-12)
    assert z1 - z0 == pytest.approx(-1 * MM)


def test_mirror_reflects_across_plane():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM, position=(1 * MM, 0, 0))
    x0 = _mid(b)[0]
    g.mirror(b, normal=(1, 0, 0))
    assert _mid(b)[0] == pytest.approx(-x0)


def test_copy_is_independent_and_inherits_material():
    g = rf.Geometry(maxh=5 * MM)
    mat = rf.Dielectric(er=4.4)
    b = g.box(1 * MM, 1 * MM, 1 * MM, material=mat)
    d = g.copy(b)
    assert d is not b
    assert d.material is mat
    # Moving the copy must not disturb the source.
    src_x = _mid(b)[0]
    g.translate(d, dx=2 * MM)
    assert _mid(b)[0] == pytest.approx(src_x)
    assert _mid(d)[0] == pytest.approx(src_x + 2 * MM)


def test_copy_does_not_inherit_name():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM)
    b.name = "src"
    d = g.copy(b)
    assert d.name is None


def test_array_linear_even_spacing():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM)
    cells = g.array(b, 4, spacing=(2 * MM, 0, 0))
    assert len(cells) == 4
    assert cells[0] is b
    xs = sorted(_mid(c)[0] for c in cells)
    deltas = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    for d in deltas:
        assert d == pytest.approx(2 * MM)


def test_array_polar_constant_radius():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM, position=(5 * MM, -0.5 * MM, -0.5 * MM))
    petals = g.array(b, 6, rotation=2 * math.pi / 6)
    assert len(petals) == 6
    radii = [math.hypot(*_mid(p)[:2]) for p in petals]
    for r in radii:
        assert r == pytest.approx(radii[0], abs=1e-6)


def test_array_polar_meshes():
    """rotated boxes become lofts; the assembled scene must mesh with every
    petal as its own region"""
    g = rf.Geometry(maxh=5 * MM)
    air = g.box(20 * MM, 20 * MM, 4 * MM, position=(-10 * MM, -10 * MM, -2 * MM),
                material=rf.Air())
    b = g.box(1 * MM, 1 * MM, 1 * MM, position=(5 * MM, -0.5 * MM, -0.5 * MM),
              material=rf.Dielectric(er=2.0))
    petals = g.array(b, 4, rotation=2 * math.pi / 4)
    for p in petals[1:]:
        p.material = rf.Dielectric(er=2.0)
    g.mesh()
    assert g.n_tets > 0
    # every petal resolved to its own volume material entry
    assert len(g._volume_materials) == 1 + len(petals)


def test_array_requires_exactly_one_mode():
    g = rf.Geometry(maxh=5 * MM)
    b = g.box(1 * MM, 1 * MM, 1 * MM)
    with pytest.raises(ValueError):
        g.array(b, 3)
    with pytest.raises(ValueError):
        g.array(b, 3, spacing=(1 * MM, 0, 0), rotation=0.1)
    with pytest.raises(ValueError):
        g.array(b, 0, spacing=(1 * MM, 0, 0))
