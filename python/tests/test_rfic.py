# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Regression tests for the rapidfem.rfic package.

The bundled ``*.fem.json`` example exports double as fixtures: every layout
that rapidpassives ships through ``exportForFEM()`` must keep building into
a valid geometry (conductors extruded, ports resolved, fragment clean).
Meshing/solving is covered by the geometry suite; here we pin the builder
contract itself, cheap enough to run on every push.
"""
import json
from importlib.resources import files

import pytest

import rapidfem.rfic as rfic

FIXTURES = [
    "fd_rfic_spiral_from_json.fem.json",
    "fd_rfic_symmetric_inductor_from_json.fem.json",
    "fd_rfic_symmetric_transformer_from_json.fem.json",
    "fd_rfic_stacked_transformer_from_json.fem.json",
    "fd_rfic_ratrace_coupler_from_json.fem.json",
]


def _fixture(name: str) -> dict:
    with (files("rapidfem.examples") / name).open() as f:
        return json.load(f)


# ── Stack model ─────────────────────────────────────────────────────────────

def test_stack_presets_resolve():
    for pdk in ("sky130", "sg13g2"):
        stack = rfic.Stack.from_pdk(pdk)
        assert stack.metals(), pdk
        assert stack.vias(), pdk
        assert stack.top_z > stack.bottom_z


def test_stack_json_roundtrip():
    a = rfic.Stack.sky130()
    b = rfic.Stack.from_dict(a.to_dict())
    assert b.name == a.name
    assert len(b.layers) == len(a.layers)
    for la, lb in zip(a.layers, b.layers):
        assert la.name == lb.name
        assert la.gds_key == lb.gds_key
        assert la.z == pytest.approx(lb.z)
        assert la.thickness == pytest.approx(lb.thickness)
        assert la.sigma == lb.sigma
    assert b.oxide_er == a.oxide_er
    assert b.substrate_sigma == a.substrate_sigma


def test_stack_from_xml_sg13g2():
    """The bundled IHP XML must reproduce the process-spec values that the
    SG13G2 measurement validation (muehlhaus) pinned down."""
    um = 1e-6
    s = rfic.Stack.sg13g2()

    m1 = s.by_name("Metal1")
    assert m1.z == pytest.approx(1.04 * um)
    assert m1.z_top == pytest.approx(1.46 * um)
    assert m1.sigma == 2.164e7

    tm2 = s.by_name("TopMetal2")
    assert tm2.z == pytest.approx(11.2303 * um)
    assert tm2.thickness == pytest.approx(3.0 * um)
    assert tm2.sigma == 3.03e7

    assert s.by_name("TopVia2").sigma == 3.143e6
    assert s.by_name("SUBGND").is_pec          # LOWLOSS 1e10 convention

    # background dielectric stack, anchored by the Substrate Offset
    assert [d.name for d in s.dielectrics] == [
        "Substrate", "EPI", "SiO2", "Passive", "AIR"]
    assert s.dielectric_at(-1 * um).name == "EPI"
    sio2 = s.dielectric_at(5 * um)
    assert sio2.z_top == pytest.approx(15.7303 * um)
    passive = s.dielectric_at(15.8 * um)
    assert s.materials[passive.material].er == 6.6

    # legacy scalars derived from the dielectric stack
    assert s.substrate_er == 11.9
    assert s.substrate_sigma == 2.0
    assert s.oxide_er == 4.1


def test_stack_from_xml_roundtrips_dielectrics():
    s = rfic.Stack.sg13g2()
    s2 = rfic.Stack.from_dict(s.to_dict())
    assert len(s2.dielectrics) == len(s.dielectrics)
    assert len(s2.layers) == len(s.layers)
    assert s2.materials["TopMetal2"].sigma == s.materials["TopMetal2"].sigma
    assert s2.by_name("Metal1").z == pytest.approx(s.by_name("Metal1").z)
    assert s2.oxide_er == s.oxide_er


def test_stack_from_xml_rejects_non_stackup():
    with pytest.raises(ValueError, match="root element"):
        rfic.Stack.from_xml("<NotAStackup/>")


def test_stack_lookup_errors():
    stack = rfic.Stack.sky130()
    with pytest.raises(KeyError):
        stack.by_name("nope")
    assert stack.by_gds(9999) is None


# ── build(): GDS + stack -> solve-ready model ──────────────────────────────

@pytest.fixture(scope="module")
def mini_gds(tmp_path_factory):
    """A minimal SG13G2 structure: SUBGND sheet, TopMetal1 feed strip,
    TopVia2 patch, TopMetal2 strip, plus one port-marker rectangle."""
    gdstk = pytest.importorskip("gdstk")
    um = 1.0   # gds units are um
    lib = gdstk.Library(unit=1e-6)
    cell = lib.new_cell("mini")
    cell.add(gdstk.rectangle((-30 * um, -20 * um), (30 * um, 20 * um),
                             layer=250))                       # SUBGND
    cell.add(gdstk.rectangle((-25 * um, -5 * um), (25 * um, 5 * um),
                             layer=126))                       # TopMetal1
    cell.add(gdstk.rectangle((-10 * um, -5 * um), (10 * um, 5 * um),
                             layer=133))                       # TopVia2
    cell.add(gdstk.rectangle((-10 * um, -5 * um), (10 * um, 5 * um),
                             layer=134))                       # TopMetal2
    cell.add(gdstk.rectangle((-24 * um, -1 * um), (-16 * um, 1 * um),
                             layer=201))                       # port marker
    path = tmp_path_factory.mktemp("gds") / "mini.gds"
    lib.write_gds(str(path))
    return str(path)


def test_build_sg13g2(mini_gds):
    um = 1e-6
    stack = rfic.Stack.sg13g2()
    model = rfic.build(
        mini_gds, stack,
        ports=[rfic.ViaPort(z=("SUBGND", "TopMetal1"), marker=201)],
        margin=60 * um, air=40 * um, air_top=80 * um,
        mesh=rfic.MeshSpec(scale=2.0, conductor=10 * um, port=6 * um,
                           global_h=80 * um,
                           graded={"Substrate": [(60 * um, 60 * um),
                                                 (120 * um, 120 * um)]}),
    )
    g = model.geometry
    try:
        assert set(model.conductors) == {"SUBGND", "TopMetal1", "TopVia2",
                                         "TopMetal2"}
        # background slabs all present; graded substrate split into 2 boxes
        assert set(model.slabs) == {"Substrate", "EPI", "SiO2", "Passive", "AIR"}
        assert len(model.slabs["Substrate"]) == 2
        assert len(model.air_shell) == 6
        # marker-derived port plate
        assert len(model.ports) == 1
        # conductor policy: SUBGND is LOWLOSS -> PEC; TopVia2 -> volume
        # conductor (anisotropic cond_diag, no surface physics)
        from rapidfem.physics import PEC, SurfaceImpedance
        pecs = [p for p in g._physics if isinstance(p, PEC)]
        sibcs = [p for p in g._physics if isinstance(p, SurfaceImpedance)]
        assert pecs and len(sibcs) == 2          # TopMetal1 + TopMetal2
        from rapidfem.rfic.build import VIA_LATERAL_FACTOR
        via_mat = model.conductors["TopVia2"][0].material
        assert via_mat.cond_diag[2] == stack.by_name("TopVia2").sigma
        assert via_mat.cond_diag[0] == pytest.approx(
            VIA_LATERAL_FACTOR * stack.by_name("TopVia2").sigma)
        # footprint = conductor bbox + margin
        x0, y0, x1, y1 = model.footprint
        assert x0 == pytest.approx(-90 * um, abs=1e-9)
        assert y1 == pytest.approx(80 * um, abs=1e-9)
        # meshes end-to-end and reports stats
        g.mesh()
        assert g.mesh_stats.n_tets > 0
        assert any(name.startswith("port_") for name in g.mesh_stats.groups)
    finally:
        g.close()


def test_build_requires_dielectrics(mini_gds):
    stack = rfic.Stack.sky130()   # legacy preset, no background dielectrics
    with pytest.raises(ValueError, match="dielectrics"):
        rfic.build(mini_gds, stack)


# ── derived mesh presets ───────────────────────────────────────────────────

def test_meshspec_derive_sizes_from_drawn_layers():
    um = 1e-6
    stack = rfic.Stack.sg13g2()
    drawn = ["SUBGND", "TopMetal1", "TopVia2", "TopMetal2"]
    spec = rfic.MeshSpec.derive(stack, drawn, preset="accurate")

    # thinnest DRAWN layer is TopMetal1 (2 um); the stack's thin auxiliary
    # layers (MIM, single vias) must not drag the size down
    t_min = min(stack.by_name(n).thickness for n in drawn)
    assert t_min == pytest.approx(2 * um)
    assert spec.conductor == pytest.approx(4 * t_min)
    assert spec.port == pytest.approx(2 * t_min)
    assert spec.global_h == pytest.approx(32 * t_min)

    # the substrate is thick and field-poor -> graded, coarser towards the
    # backside; without this the accurate preset blows up the DOF count
    assert "Substrate" in spec.graded
    zones = spec.graded["Substrate"]
    assert len(zones) == 2 and zones[1][1] > zones[0][1]


def test_meshspec_derive_presets_are_monotone():
    stack = rfic.Stack.sg13g2()
    drawn = ["SUBGND", "TopMetal1", "TopMetal2"]
    scales = [rfic.MeshSpec.derive(stack, drawn, preset=p).scale
              for p in ("fast", "balanced", "accurate")]
    assert scales == sorted(scales, reverse=True)

    with pytest.raises(ValueError, match="fast|balanced|accurate"):
        rfic.MeshSpec.derive(stack, drawn, preset="nonsense")

    with pytest.raises(ValueError, match="no drawn layers"):
        rfic.MeshSpec.derive(stack, [], preset="fast")


def test_build_accepts_preset_name(mini_gds):
    um = 1e-6
    stack = rfic.Stack.sg13g2()
    model = rfic.build(
        mini_gds, stack,
        ports=[rfic.ViaPort(z=("SUBGND", "TopMetal1"), marker=201)],
        margin=60 * um, air=40 * um, air_top=80 * um, mesh="fast",
    )
    g = model.geometry
    try:
        g.mesh()
        assert g.mesh_stats.n_tets > 0
    finally:
        g.close()


@pytest.mark.parametrize("passv,boundary", [
    ("conformal", "abc"), ("none", "abc"), ("planar", "pml")])
def test_build_passivation_and_boundary_modes(mini_gds, passv, boundary):
    um = 1e-6
    stack = rfic.Stack.sg13g2()
    model = rfic.build(
        mini_gds, stack,
        margin=60 * um, air=40 * um, air_top=80 * um,
        passivation=passv, boundary=boundary,
        mesh=rfic.MeshSpec(scale=2.0, conductor=10 * um, port=6 * um,
                           global_h=80 * um,
                           graded={"Substrate": [(60 * um, 60 * um),
                                                 (120 * um, 120 * um)]}),
    )
    g = model.geometry
    try:
        if passv == "conformal":
            # sheet + sidewall ring + cap over the exposed TopMetal2
            assert len(model.slabs["Passive"]) >= 3
            assert len(model.slabs["AIR"]) >= 2      # polygon air prisms
        elif passv == "none":
            assert "Passive" not in model.slabs
        if boundary == "pml":
            from rapidfem.physics import ABC, PML
            pmls = [p for p in g._physics if isinstance(p, PML)]
            abcs = [p for p in g._physics if isinstance(p, ABC)]
            assert len(pmls) == 6 and not abcs
        g.mesh()
        assert g.mesh_stats.n_tets > 0
    finally:
        g.close()


# ── FEM-JSON bridge ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("fixture", FIXTURES)
def test_from_fem_json_builds(fixture):
    doc = _fixture(fixture)
    layout = rfic.from_fem_json(doc)
    try:
        # every JSON conductor layer materialised at least one volume
        json_layers = {c["layer"] for c in doc["conductors"]
                       if any(l["thickness_um"] > 0 for l in doc["stack"]["layers"]
                              if l["id"] == c["layer"])}
        assert set(layout.conductors) == json_layers
        assert all(vols for vols in layout.conductors.values())
        # every JSON port resolved into a plate
        assert set(layout.ports) == {p["name"] for p in doc["ports"]}
        # enclosure handles exist and are 3D
        for obj in (layout.substrate, layout.oxide, layout.air):
            assert obj.dim == 3
        assert layout.doc is doc
    finally:
        layout.geometry.close()


def test_from_fem_json_rejects_unknown_schema():
    doc = _fixture(FIXTURES[0])
    doc["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        rfic.from_fem_json(doc)


def test_from_fem_json_meshes():
    """One representative layout through the full mesh path, with stats."""
    layout = rfic.from_fem_json(_fixture("fd_rfic_spiral_from_json.fem.json"))
    g = layout.geometry
    try:
        import rapidfem as rf
        all_conductors = [v for vs in layout.conductors.values() for v in vs]
        rf.PEC(*(v.faces for v in all_conductors), layout.ground)
        for port in layout.ports.values():
            rf.LumpedPort(port, direction=(0, 0, 1), z0=50.0)
        rf.ABC(*layout.air.faces.outer)
        g.mesh()
        s = g.mesh_stats
        assert s is not None
        assert s.n_tets > 0
        assert s.dofs_min == s.n_edges
        assert s.dofs_max == 2 * s.n_edges + 2 * s.n_tris
        assert any(name.startswith("port_") for name in s.groups)
    finally:
        g.close()
