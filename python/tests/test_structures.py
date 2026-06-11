"""General RF structure builders in rapidfem.structures.

Fast checks: the builders compose valid geometry, return the documented
result fields, and (with add_ports) attach the canonical ports and still mesh.
"""
from __future__ import annotations

import pytest

import rapidfem as rf
from rapidfem import structures as st

MM = 1e-3


def test_coax_geometry_only():
    g = rf.Geometry(maxh=3 * MM)
    cx = st.coax(g, ri=1.5 * MM, ro=3.45 * MM, length=20 * MM)
    assert cx.dielectric is not None
    assert len(cx.port_a) >= 1
    assert len(cx.port_b) >= 1
    assert cx.ports == []  # no physics attached unless asked
    # Geometry-only result must still mesh.
    g.mesh()
    assert g.n_tets > 0


def test_coax_add_ports_meshes():
    g = rf.Geometry(maxh=3 * MM)
    cx = st.coax(g, ri=1.5 * MM, ro=3.45 * MM, length=20 * MM, add_ports=True)
    assert len(cx.ports) == 2
    g.mesh()
    assert g.n_tets > 0


def test_coax_dielectric_fill():
    g = rf.Geometry(maxh=3 * MM)
    cx = st.coax(g, ri=1.0 * MM, ro=2.3 * MM, length=10 * MM, er=2.1)
    mat = cx.dielectric.material
    assert getattr(mat, "er", None) == pytest.approx(2.1)


def test_coax_axis_x_meshes():
    g = rf.Geometry(maxh=3 * MM)
    cx = st.coax(g, ri=1.5 * MM, ro=3.45 * MM, length=15 * MM,
                 axis="x", add_ports=True)
    assert len(cx.ports) == 2
    g.mesh()
    assert g.n_tets > 0


def test_coax_rejects_bad_radii():
    g = rf.Geometry(maxh=3 * MM)
    with pytest.raises(ValueError):
        st.coax(g, ri=3.0 * MM, ro=2.0 * MM, length=10 * MM)


def test_coax_rejects_bad_axis():
    g = rf.Geometry(maxh=3 * MM)
    with pytest.raises(ValueError):
        st.coax(g, ri=1.0 * MM, ro=2.0 * MM, length=10 * MM, axis="w")


def _ms_kwargs():
    return dict(line_w=1.13 * MM, line_l=30 * MM, sub_w=20 * MM,
                sub_h=0.508 * MM, air_h=10 * MM, er=3.55, tand=0.0027)


def test_microstrip_geometry_only():
    g = rf.Geometry(maxh=5 * MM)
    ms = st.microstrip(g, **_ms_kwargs())
    assert ms.substrate is not None and ms.air is not None and ms.trace is not None
    assert len(ms.ground) >= 1
    assert len(ms.port_a) == 2 and len(ms.port_b) == 2
    assert ms.ports == [] and ms.pec is None
    g.mesh()
    assert g.n_tets > 0


def test_microstrip_add_ports_meshes():
    g = rf.Geometry(maxh=5 * MM)
    ms = st.microstrip(g, add_ports=True, f0=3.0e9, **_ms_kwargs())
    assert len(ms.ports) == 2
    assert ms.pec is not None
    g.mesh()
    assert g.n_tets > 0


def test_microstrip_add_ports_requires_f0():
    g = rf.Geometry(maxh=5 * MM)
    with pytest.raises(ValueError):
        st.microstrip(g, add_ports=True, **_ms_kwargs())


def test_microstrip_substrate_mesh_default():
    g = rf.Geometry(maxh=5 * MM)
    ms = st.microstrip(g, **_ms_kwargs())
    # Substrate material maxh defaults to sub_h / 3.
    assert ms.substrate.material.maxh == pytest.approx(0.508 * MM / 3)


def _cpw_kwargs():
    return dict(signal_w=0.4 * MM, gap=0.2 * MM, line_l=20 * MM,
                sub_w=10 * MM, sub_h=0.635 * MM, air_h=6 * MM, er=9.9)


def test_cpw_geometry_only():
    g = rf.Geometry(maxh=4 * MM)
    cw = st.cpw(g, **_cpw_kwargs())
    assert cw.signal is not None
    assert cw.ground_left is not None and cw.ground_right is not None
    assert cw.ports == []
    g.mesh()
    assert g.n_tets > 0


def test_cpw_add_ports_meshes():
    g = rf.Geometry(maxh=4 * MM)
    cw = st.cpw(g, add_ports=True, f0=10e9, backside_ground=True, **_cpw_kwargs())
    assert len(cw.ports) == 2
    g.mesh()
    assert g.n_tets > 0


def test_cpw_rejects_oversized_gap():
    g = rf.Geometry(maxh=4 * MM)
    with pytest.raises(ValueError):
        st.cpw(g, signal_w=0.4 * MM, gap=10 * MM, line_l=20 * MM,
               sub_w=10 * MM, sub_h=0.635 * MM, air_h=6 * MM, er=9.9)


def _sl_kwargs():
    return dict(line_w=0.3 * MM, line_l=20 * MM, sub_w=8 * MM,
                sub_h=1.0 * MM, er=3.38)


def test_stripline_geometry_only():
    g = rf.Geometry(maxh=4 * MM)
    sl = st.stripline(g, **_sl_kwargs())
    assert sl.lower is not None and sl.upper is not None and sl.trace is not None
    assert len(sl.port_a) == 2 and len(sl.port_b) == 2
    assert sl.ports == []
    g.mesh()
    assert g.n_tets > 0


def test_stripline_add_ports_meshes():
    g = rf.Geometry(maxh=4 * MM)
    sl = st.stripline(g, add_ports=True, f0=5e9, **_sl_kwargs())
    assert len(sl.ports) == 2 and sl.pec is not None
    g.mesh()
    assert g.n_tets > 0


def test_stripline_requires_f0():
    g = rf.Geometry(maxh=4 * MM)
    with pytest.raises(ValueError):
        st.stripline(g, add_ports=True, **_sl_kwargs())


def test_rect_waveguide_add_ports_meshes():
    g = rf.Geometry(maxh=6 * MM)
    wg = st.rect_waveguide(g, a=22.86 * MM, b=10.16 * MM, length=30 * MM,
                           add_ports=True)
    assert len(wg.ports) == 2
    g.mesh()
    assert g.n_tets > 0


def test_rect_waveguide_axis_y():
    g = rf.Geometry(maxh=6 * MM)
    wg = st.rect_waveguide(g, a=22.86 * MM, b=10.16 * MM, length=30 * MM,
                           axis="y", add_ports=True)
    assert len(wg.ports) == 2
    g.mesh()
    assert g.n_tets > 0


def test_circ_waveguide_add_ports_meshes():
    g = rf.Geometry(maxh=5 * MM)
    wg = st.circ_waveguide(g, radius=10 * MM, length=30 * MM,
                           add_ports=True, f0=12e9)
    assert len(wg.ports) == 2
    g.mesh()
    assert g.n_tets > 0


def test_circ_waveguide_requires_f0():
    g = rf.Geometry(maxh=5 * MM)
    with pytest.raises(ValueError):
        st.circ_waveguide(g, radius=10 * MM, length=30 * MM, add_ports=True)


def test_sweep_along_path_arc_meshes():
    g = rf.Geometry(maxh=0.5 * MM)
    pts = [(0, 0, 0), (0.5 * MM, 0, 0.4 * MM), (1 * MM, 0, 0)]
    wire = st.sweep_along_path(g, 50e-6, pts, material=rf.Air())
    assert wire.dim == 3
    g.mesh()
    assert g.n_tets > 0


def test_sweep_along_path_rejects_short_path():
    g = rf.Geometry(maxh=MM)
    with pytest.raises(ValueError):
        st.sweep_along_path(g, 50e-6, [(0, 0, 0)])


def test_helix_meshes():
    g = rf.Geometry(maxh=0.6 * MM)
    coil = st.helix(g, radius=2 * MM, pitch=1 * MM, turns=3,
                    wire_radius=0.15 * MM, material=rf.Air())
    assert coil.dim == 3
    g.mesh()
    assert g.n_tets > 0


def test_helix_rejects_bad_turns():
    g = rf.Geometry(maxh=MM)
    with pytest.raises(ValueError):
        st.helix(g, radius=2 * MM, pitch=1 * MM, turns=0, wire_radius=0.15 * MM)
