# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Placement and boolean operations of ``rapidfem.Geometry``: transforms,
copies, arrays, intersection and extrusion along any direction."""
from __future__ import annotations

import math

import numpy as np
import pytest

import rapidfem as rf

MM = 1e-3


def _box_of(obj) -> np.ndarray:
    """(xmin, ymin, zmin, xmax, ymax, zmax) over an object's faces."""
    b = np.array([f.bbox for f in obj.faces])
    return np.concatenate([b[:, :3].min(axis=0), b[:, 3:].max(axis=0)])


def _air(g):
    return g.box(20 * MM, 20 * MM, 20 * MM, position=(-10 * MM,) * 3, material=rf.Air())


def test_translate_and_rotate_move_the_faces():
    g = rf.Geometry(maxh=2 * MM)
    _air(g)
    b = g.box(2 * MM, 4 * MM, 1 * MM, material=rf.Dielectric(er=2.0))
    g.translate(b, dx=1 * MM)
    g.rotate(b, math.pi / 2, axis=(0, 0, 1), center=(1 * MM, 0, 0))
    np.testing.assert_allclose(_box_of(b), np.array([-3, 0, 0, 1, 2, 1]) * MM, atol=1e-12)


def test_mirror_and_stretch():
    g = rf.Geometry(maxh=2 * MM)
    _air(g)
    b = g.box(1 * MM, 1 * MM, 1 * MM, position=(1 * MM, 0, 0))
    g.mirror(b, normal=(1, 0, 0))
    np.testing.assert_allclose(_box_of(b), np.array([-2, 0, 0, -1, 1, 1]) * MM, atol=1e-12)
    g.stretch(b, fz=3.0)
    np.testing.assert_allclose(_box_of(b), np.array([-2, 0, 0, -1, 1, 3]) * MM, atol=1e-12)


def test_array_places_copies_in_their_own_regions():
    g = rf.Geometry(maxh=2 * MM)
    _air(g)
    sub = rf.Dielectric(er=3.0)
    b = g.box(1 * MM, 1 * MM, 1 * MM, material=sub)
    row = g.array(b, 3, spacing=(2 * MM, 0, 0))
    assert row[0] is b and len(row) == 3
    assert all(c.material is sub for c in row)
    np.testing.assert_allclose(_box_of(row[2])[[0, 3]], [4 * MM, 5 * MM], atol=1e-12)
    ring = g.array(g.box(1 * MM, 1 * MM, 1 * MM, position=(5 * MM, 0, 0)), 4,
                   rotation=math.pi / 2)
    np.testing.assert_allclose(_box_of(ring[2])[[0, 3]], [-6 * MM, -5 * MM], atol=1e-12)
    g.mesh()
    assert g.mesh_stats.n_tets > 0


def test_array_arguments():
    g = rf.Geometry()
    b = g.box(1, 1, 1)
    with pytest.raises(ValueError):
        g.array(b, 2)
    with pytest.raises(ValueError):
        g.array(b, 0, spacing=(1, 0, 0))


def test_intersect_keeps_the_overlap():
    g = rf.Geometry(maxh=1 * MM)
    _air(g)
    a = g.box(2 * MM, 2 * MM, 2 * MM, material=rf.Dielectric(er=2.0))
    t = g.box(2 * MM, 2 * MM, 2 * MM, position=(1 * MM,) * 3, material=rf.Dielectric(er=5.0))
    g.intersect(a, t)
    assert t.material is None
    np.testing.assert_allclose(_box_of(a), np.array([1, 1, 1, 2, 2, 2]) * MM, atol=1e-12)
    area = sum(f._info()[2] for f in a.faces)
    assert area == pytest.approx(6 * MM**2, rel=1e-9)


def test_extrude_along_a_tilted_axis():
    g = rf.Geometry(maxh=1 * MM)
    _air(g)
    p = g.xy_plate(2 * MM, 2 * MM)
    v = g.extrude(p, 2 * MM, axis=(1, 0, 1), material=rf.Dielectric(er=2.0))
    assert v.dim == 3
    assert len(v.faces) == 6
    s = 2 * MM / math.sqrt(2)
    np.testing.assert_allclose(_box_of(v), [0, 0, 0, 2 * MM + s, 2 * MM, s], atol=1e-12)
    g.mesh()
    assert g.mesh_stats.n_tets > 0


def test_extrude_after_a_rotation_sweeps_in_world_axes():
    g = rf.Geometry(maxh=1 * MM)
    _air(g)
    p = g.xy_plate(2 * MM, 2 * MM)
    g.rotate(p, math.pi / 2, axis=(1, 0, 0))  # into the xz-plane
    v = g.extrude(p, 1 * MM, axis=(0, -1, 0))
    np.testing.assert_allclose(_box_of(v), np.array([0, -1, 0, 2, 0, 2]) * MM, atol=1e-12)


def test_sweep_along_path_and_helix():
    from rapidfem import structures as st
    g = rf.Geometry(maxh=1 * MM)
    _air(g)
    pts = [(0, 0, 0), (2 * MM, 0, 2 * MM), (4 * MM, 0, 0)]
    prof = g.disc(0.3 * MM, position=pts[0], axis=(1, 0, 1))
    wire = st.sweep_along_path(g, prof, pts, material=rf.Conductor(conductivity=5.8e7))
    b = _box_of(wire)
    assert b[0] < 0 and b[3] > 4 * MM and 2 * MM < b[5] < 2.4 * MM
    coil = st.helix(g, radius=3 * MM, pitch=1.5 * MM, turns=2, wire_radius=0.2 * MM,
                    position=(0, 0, -6 * MM))
    c = _box_of(coil)
    assert c[3] == pytest.approx(3.2 * MM, rel=0.02)
    assert c[5] - c[2] == pytest.approx(3.4 * MM, rel=0.02)
    g.mesh()
    assert g.mesh_stats.n_tets > 0
    with pytest.raises(ValueError, match="disc"):
        st.sweep_along_path(g, g.xy_plate(1 * MM, 1 * MM), pts)
