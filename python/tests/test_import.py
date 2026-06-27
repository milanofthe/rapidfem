"""External CAD / mesh import via ``Geometry.load``.

Covers the three substrates the dispatcher handles: OCC BREP formats
(STEP/BREP) that come back as composable ``GeoObject``s, STL surfaces healed
into meshable solids, and pre-built ``.msh`` meshes loaded as a ``MeshScene``
whose named physical groups carry materials and physics into a real solve.

Fixtures are generated with gmsh into a tmp dir, each in its own model, so they
don't collide with the global gmsh model rapidfem drives.
"""
from __future__ import annotations

import numpy as np
import pytest

import gmsh
import rapidfem as rf

MM = 1e-3


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    """Write box.step, box.brep, sphere.stl and wg.msh; return their paths."""
    d = tmp_path_factory.mktemp("import_fixtures")
    if not gmsh.isInitialized():
        gmsh.initialize()

    # STEP + BREP: a 10x5x3 (mm) box. gmsh writes STEP with a millimetre unit,
    # so importing with the default unit="M" yields a 0.01x0.005x0.003 m solid.
    gmsh.model.add("fix_step")
    gmsh.model.occ.addBox(0, 0, 0, 10, 5, 3)
    gmsh.model.occ.synchronize()
    gmsh.write(str(d / "box.step"))
    gmsh.write(str(d / "box.brep"))
    gmsh.model.remove()

    # STL: a unit-radius sphere surface mesh (unit-less).
    gmsh.model.add("fix_stl")
    gmsh.model.occ.addSphere(0, 0, 0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(d / "sphere.stl"))
    gmsh.model.mesh.clear()
    gmsh.model.remove()

    # MSH: a rectangular-waveguide air box with named groups (metres).
    gmsh.model.add("fix_msh")
    vtag = gmsh.model.occ.addBox(0, 0, 0, 20 * MM, 10 * MM, 30 * MM)
    gmsh.model.occ.synchronize()
    faces = gmsh.model.getBoundary([(3, vtag)], oriented=False)
    zmin = min(faces, key=lambda dt: gmsh.model.occ.getCenterOfMass(2, dt[1])[2])
    zmax = max(faces, key=lambda dt: gmsh.model.occ.getCenterOfMass(2, dt[1])[2])
    walls = [t for d_, t in faces if t not in (zmin[1], zmax[1])]
    gmsh.model.addPhysicalGroup(3, [vtag], name="air")
    gmsh.model.addPhysicalGroup(2, [zmin[1]], name="port_in")
    gmsh.model.addPhysicalGroup(2, [zmax[1]], name="port_out")
    gmsh.model.addPhysicalGroup(2, walls, name="walls")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 4 * MM)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(d / "wg.msh"))
    gmsh.model.mesh.clear()
    gmsh.model.remove()

    return {
        "step": str(d / "box.step"),
        "brep": str(d / "box.brep"),
        "stl": str(d / "sphere.stl"),
        "msh": str(d / "wg.msh"),
    }


# ── CAD: STEP / BREP as composable primitives ───────────────────────────────

def test_step_imports_as_geoobject_with_metre_units(fixtures):
    g = rf.Geometry(maxh=2.0)
    part = g.load(fixtures["step"])
    assert isinstance(part, rf.GeoObject)
    assert part.dim == 3
    xmin, ymin, zmin, xmax, ymax, zmax = part._entity.bbox
    # 10x5x3 mm STEP -> metres
    assert (xmax - xmin) == pytest.approx(10 * MM, rel=1e-3)
    assert (ymax - ymin) == pytest.approx(5 * MM, rel=1e-3)
    assert (zmax - zmin) == pytest.approx(3 * MM, rel=1e-3)
    assert len(part.faces) == 6


def test_step_scale_override(fixtures):
    # scale = metres per file unit; treat the mm box as if authored in metres.
    g = rf.Geometry(maxh=2.0)
    part = g.load(fixtures["step"], unit="MM", scale=1.0)
    dx = part._entity.bbox[3] - part._entity.bbox[0]
    assert dx == pytest.approx(10.0, rel=1e-3)


def test_step_composes_with_boolean_and_meshes(fixtures):
    g = rf.Geometry(maxh=3 * MM)
    part = g.load(fixtures["step"], material=rf.Air())
    # face selectors work on the imported solid -> composable physics
    rf.PEC(*part.faces.unassigned)
    mb, _ = g.mesh()
    assert isinstance(mb, (bytes, bytearray)) and len(mb) > 0
    assert len(g._material_tags) == 1
    assert len(g._physics_tags) == 1


def test_brep_imports(fixtures):
    g = rf.Geometry(maxh=2.0)
    part = g.load(fixtures["brep"])
    assert isinstance(part, rf.GeoObject) and part.dim == 3


def test_step_placement_position(fixtures):
    g = rf.Geometry(maxh=2.0)
    part = g.load(fixtures["step"], position=(0.1, 0.0, 0.0))
    assert part._entity.bbox[0] == pytest.approx(0.1, abs=1e-6)       # xmin
    assert part._entity.bbox[3] == pytest.approx(0.1 + 10 * MM, abs=1e-6)


def test_step_placement_rotation(fixtures):
    import math
    g = rf.Geometry(maxh=2.0)
    part = g.load(fixtures["step"], rotation=(math.pi / 2, (0, 0, 1)))
    # 90 deg about z swaps the x (10mm) and y (5mm) extents
    dx = part._entity.bbox[3] - part._entity.bbox[0]
    dy = part._entity.bbox[4] - part._entity.bbox[1]
    assert dx == pytest.approx(5 * MM, rel=1e-3)
    assert dy == pytest.approx(10 * MM, rel=1e-3)


def test_step_posthoc_transforms_work(fixtures):
    g = rf.Geometry(maxh=2.0)
    part = g.load(fixtures["step"])
    x0 = part._entity.cog[0]
    g.translate(part, dx=0.05)
    assert part._entity.cog[0] - x0 == pytest.approx(0.05, abs=1e-6)


# ── STL: healed solid ───────────────────────────────────────────────────────

def test_stl_heals_into_meshable_solid(fixtures):
    g = rf.Geometry(maxh=0.3)
    solid = g.load(fixtures["stl"], material=rf.Air())
    assert isinstance(solid, rf.GeoObject) and solid.dim == 3
    r = solid._entity.bbox
    assert (r[3] - r[0]) == pytest.approx(2.0, rel=0.05)  # diameter ~2
    mb, _ = g.mesh()
    assert len(mb) > 0


def test_stl_placement_shifts_and_meshes(fixtures):
    g = rf.Geometry(maxh=0.3)
    solid = g.load(fixtures["stl"], material=rf.Air(), position=(5.0, 0.0, 0.0))
    assert solid._entity.cog[0] == pytest.approx(5.0, abs=0.05)
    mb, _ = g.mesh()
    assert len(mb) > 0


def test_stl_rejects_posthoc_transforms(fixtures):
    g = rf.Geometry(maxh=0.3)
    solid = g.load(fixtures["stl"])
    with pytest.raises(RuntimeError, match="discrete mesh"):
        g.translate(solid, dx=1.0)
    with pytest.raises(RuntimeError, match="discrete mesh"):
        g.rotate(solid, angle=0.3)


def test_stl_cannot_mix_with_primitives(fixtures):
    # primitive after STL
    g = rf.Geometry(maxh=0.3)
    g.load(fixtures["stl"])
    with pytest.raises(RuntimeError, match="discrete"):
        g.box(1.0, 1.0, 1.0)
    # STL after primitive
    g2 = rf.Geometry(maxh=0.3)
    g2.box(1.0, 1.0, 1.0)
    with pytest.raises(RuntimeError, match="discrete"):
        g2.load(fixtures["stl"])


# ── MSH: mesh mode ──────────────────────────────────────────────────────────

def test_msh_exposes_named_groups(fixtures):
    g = rf.Geometry()
    scene = g.load(fixtures["msh"])
    assert g._mode == "mesh"
    assert set(scene.groups) == {"air", "port_in", "port_out", "walls"}
    assert scene.group("air").material is None  # not yet bound


def test_msh_mode_blocks_primitives(fixtures):
    g = rf.Geometry()
    g.load(fixtures["msh"])
    with pytest.raises(RuntimeError, match="mesh mode"):
        g.box(1 * MM, 1 * MM, 1 * MM)


def test_msh_bake_and_solve(fixtures):
    g = rf.Geometry()
    scene = g.load(fixtures["msh"])
    scene.group("air").material = rf.Air()
    rf.RectWaveguidePort(scene.group("port_in"))
    rf.RectWaveguidePort(scene.group("port_out"))
    rf.PEC(scene.group("walls"))
    mb, _ = g.mesh()
    assert len(mb) > 0
    assert len(g._material_tags) == 1
    assert len(g._physics_tags) == 3  # two ports + the PEC walls

    prob = rf.Problem(g)
    res = prob.sweep(np.linspace(8e9, 12e9, 3))
    assert res.frequencies.shape == (3,)
    assert res.sparams.shape == (3, 2, 2)
    # matched air-filled WR-90-ish guide: low reflection in band
    assert 20 * np.log10(abs(res.sparams[1, 0, 0])) < -20


def test_msh_mode_requires_bindings(fixtures):
    g = rf.Geometry()
    g.load(fixtures["msh"])
    with pytest.raises(RuntimeError, match="no materials or physics"):
        g.mesh()


# ── Dispatcher errors ───────────────────────────────────────────────────────

def test_unsupported_extension(tmp_path):
    p = tmp_path / "thing.xyz"
    p.write_text("nope")
    g = rf.Geometry(maxh=1.0)
    with pytest.raises(ValueError, match="unsupported extension"):
        g.load(str(p))


def test_missing_file():
    g = rf.Geometry(maxh=1.0)
    with pytest.raises(FileNotFoundError):
        g.load("does_not_exist.step")


def test_msh_placement_rejected(fixtures):
    g = rf.Geometry()
    with pytest.raises(ValueError, match="position/rotation"):
        g.load(fixtures["msh"], position=(1.0, 0.0, 0.0))
