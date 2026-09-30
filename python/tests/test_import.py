# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""External geometry import via ``Geometry.load``.

STEP files load one solid per body in metres (the fixture is a three-part
millimetre assembly). STL (and OBJ) surfaces load as meshable solids, placed
by unit, scale, rotation and position. A ``.msh`` volume mesh loads in mesh
mode (its groups the handles for materials and physics, no remeshing); the
fixture is written by ``save_mesh``, so the round trip is tested too. IGES
and BREP are not supported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rapidfem as rf

MM = 1e-3
ASSEMBLY = Path(__file__).parent / "data" / "assembly.step"


def _box(obj) -> np.ndarray:
    b = np.array([f.bbox for f in obj.faces])
    return np.concatenate([b[:, :3].min(axis=0), b[:, 3:].max(axis=0)])


def test_step_assembly_loads_in_metres():
    g = rf.Geometry(maxh=4 * MM)
    air = g.box(40 * MM, 40 * MM, 40 * MM, position=(-10 * MM, -10 * MM, -15 * MM),
                material=rf.Air())
    base, post, plate = g.load(str(ASSEMBLY), material=rf.Dielectric(er=3.0))
    assert all(p.dim == 3 for p in (base, post, plate))
    np.testing.assert_allclose(_box(base), np.array([0, 0, 0, 20, 20, 5]) * MM, atol=1e-9)
    np.testing.assert_allclose(_box(plate), np.array([0, 0, -4, 20, 20, 0]) * MM, atol=1e-9)
    assert _box(post)[5] == pytest.approx(17 * MM)
    g.mesh()
    assert g.mesh_stats.n_tets > 0 and g.mesh_stats.quality_min > 10
    assert len(air.faces.outer) == 6


def test_step_units_and_placement():
    g = rf.Geometry()
    parts = g.load(str(ASSEMBLY), unit="UM", position=(1 * MM, 0, 0))
    np.testing.assert_allclose(_box(parts[0]), [1e-3, 0, 0, 1e-3 + 20e-6, 20e-6, 5e-6], atol=1e-12)


def _icosphere_stl(path, radius: float) -> None:
    """A closed, outward-oriented icosahedron as an ASCII STL."""
    t = (1.0 + 5 ** 0.5) / 2.0
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
                  (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], dtype=float)
    v *= radius / np.linalg.norm(v[0])
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
         (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
         (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
         (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    lines = ["solid ico"]
    for a, b, c in f:
        n = np.cross(v[b] - v[a], v[c] - v[a])
        n /= np.linalg.norm(n)
        lines.append(f"facet normal {n[0]:e} {n[1]:e} {n[2]:e}")
        lines.append("outer loop")
        for i in (a, b, c):
            lines.append(f"vertex {v[i][0]:e} {v[i][1]:e} {v[i][2]:e}")
        lines.append("endloop")
        lines.append("endfacet")
    lines.append("endsolid ico")
    path.write_text("\n".join(lines))


@pytest.fixture(scope="module")
def stl(tmp_path_factory):
    path = tmp_path_factory.mktemp("import_fixtures") / "ico.stl"
    _icosphere_stl(path, 5 * MM)
    return path


def test_stl_loads_as_meshable_solid(stl):
    g = rf.Geometry(maxh=2 * MM)
    air = g.box(30 * MM, 30 * MM, 30 * MM, position=(-15 * MM,) * 3,
                material=rf.Air())
    part = g.load(str(stl), material=rf.Dielectric(er=4.0))
    assert part.dim == 3
    assert len(part.faces) > 0
    g.mesh()
    assert g.mesh_stats.n_tets > 0
    assert len(air.faces.outer) == 6


def test_stl_placement(stl):
    # the file in millimetres, turned about z and moved
    path = stl.parent / "ico_mm.stl"
    lines = []
    for line in stl.read_text().splitlines():
        if line.startswith("vertex"):
            x, y, z = (float(v) / MM for v in line.split()[1:])
            line = f"vertex {x:e} {y:e} {z:e}"
        lines.append(line)
    path.write_text("\n".join(lines))
    g = rf.Geometry(maxh=2 * MM)
    g.box(40 * MM, 40 * MM, 40 * MM, position=(-20 * MM,) * 3, material=rf.Air())
    part = g.load(str(path), unit="MM", rotation=(np.pi / 2, (0, 0, 1)),
                  position=(3 * MM, 0, 0))
    b = np.array([f.bbox for f in part.faces])
    lo, hi = b[:, :3].min(axis=0), b[:, 3:].max(axis=0)
    np.testing.assert_allclose((lo + hi) / 2, [3 * MM, 0, 0], atol=1e-9)
    assert hi[2] - lo[2] == pytest.approx(10 * MM * 0.85065, rel=1e-3)


def test_unsupported_extension(tmp_path):
    p = tmp_path / "part.xyz"
    p.write_text("")
    with pytest.raises(ValueError, match="unsupported extension"):
        rf.Geometry().load(str(p))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        rf.Geometry().load(str(tmp_path / "missing.stl"))


@pytest.mark.parametrize("ext", [".iges", ".igs", ".brep"])
def test_unsupported_cad_formats(tmp_path, ext):
    p = tmp_path / f"part{ext}"
    p.write_text("")
    with pytest.raises(NotImplementedError, match="STEP"):
        rf.Geometry().load(str(p))


# ── MSH: mesh mode ──────────────────────────────────────────────────────────

A, B, L = 22.86 * MM, 10.16 * MM, 30 * MM  # WR-90


def _waveguide(g):
    air = g.box(A, B, L, material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    return air


@pytest.fixture(scope="module")
def wg_msh(tmp_path_factory):
    g = rf.Geometry(maxh=4 * MM)
    _waveguide(g)
    g.mesh()
    path = tmp_path_factory.mktemp("import_fixtures") / "wg.msh"
    assert g.save_mesh(str(path)) == str(path)
    return path, g.mesh_stats.n_tets


def test_msh_exposes_named_groups(wg_msh):
    path, n_tets = wg_msh
    g = rf.Geometry()
    scene = g.load(str(path))
    assert {"air_1", "port_1", "port_2", "pec_1"} <= set(scene.groups)
    assert scene.group("air_1").material is None  # not yet bound
    np.testing.assert_allclose(scene.group("port_2")[0].bbox, [0, 0, L, A, B, L], atol=1e-12)
    with pytest.raises(KeyError, match="available"):
        scene.group("nope")


def test_msh_mode_blocks_primitives(wg_msh):
    g = rf.Geometry()
    g.load(str(wg_msh[0]))
    with pytest.raises(RuntimeError, match="mesh mode"):
        g.box(1 * MM, 1 * MM, 1 * MM)


def test_msh_mode_requires_bindings(wg_msh):
    g = rf.Geometry()
    g.load(str(wg_msh[0]))
    with pytest.raises(RuntimeError, match="no materials or physics"):
        g.mesh()


def test_msh_placement_rejected(wg_msh):
    with pytest.raises(ValueError, match="position/rotation"):
        rf.Geometry().load(str(wg_msh[0]), position=(1.0, 0.0, 0.0))


def test_msh_bake_and_solve(wg_msh):
    """The loaded mesh solves like the one it was saved from."""
    path, n_tets = wg_msh
    f = np.linspace(8e9, 12e9, 3)
    g = rf.Geometry()
    scene = g.load(str(path))
    scene.group("air_1").material = rf.Air()
    rf.RectWaveguidePort(scene.group("port_1"))
    rf.RectWaveguidePort(scene.group("port_2"))
    rf.PEC(scene.group("pec_1"))
    stats = g.mesh()
    assert stats.n_tets == n_tets
    assert len(g._material_tags) == 1
    assert len(g._physics_tags) == 3  # two ports + the PEC walls
    res = rf.Problem(g).sweep(f)
    assert res.sparams.shape == (3, 2, 2)

    ref = rf.Geometry(maxh=4 * MM)
    _waveguide(ref)
    ref.mesh()
    res_ref = rf.Problem(ref).sweep(f)
    np.testing.assert_allclose(res.sparams, res_ref.sparams, atol=1e-9)
    # matched air-filled guide: low reflection in band
    assert np.all(20 * np.log10(np.abs(res.sparams[:, 0, 0])) < -20)
