# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""External geometry import via ``Geometry.load``.

STL (and OBJ) surfaces load as meshable solids. STEP, IGES, BREP and ``.msh``
import are not available on the rapidmesh backend yet
(milanofthe/rapidmesh-dev#37, #140); their tests are strict expected failures
that flip to passing, and fail the suite as a reminder, once the backend can.
"""
from __future__ import annotations

import numpy as np
import pytest

import rapidfem as rf

MM = 1e-3


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


def test_stl_placement_is_not_available_yet(stl):
    g = rf.Geometry()
    with pytest.raises(NotImplementedError, match="identity placement"):
        g.load(str(stl), position=(1 * MM, 0, 0))


def test_unsupported_extension(tmp_path):
    p = tmp_path / "part.xyz"
    p.write_text("")
    with pytest.raises(ValueError, match="xyz"):
        rf.Geometry().load(str(p))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        rf.Geometry().load(str(tmp_path / "missing.stl"))


@pytest.mark.xfail(raises=NotImplementedError, strict=True,
                   reason="STEP/IGES/BREP import: milanofthe/rapidmesh-dev#37")
@pytest.mark.parametrize("ext", [".step", ".stp", ".iges", ".igs", ".brep"])
def test_cad_import(tmp_path, ext):
    p = tmp_path / f"part{ext}"
    p.write_text("")
    rf.Geometry().load(str(p))


@pytest.mark.xfail(raises=NotImplementedError, strict=True,
                   reason=".msh import: milanofthe/rapidmesh-dev#140")
def test_msh_import(tmp_path):
    p = tmp_path / "mesh.msh"
    p.write_text("")
    rf.Geometry().load(str(p))
