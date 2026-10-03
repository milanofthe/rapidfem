# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Composite builders of ``rapidfem.structures``: their layout, the physics
they attach with ``add_ports`` and the shared argument checks. The solved
behaviour of the lines is covered by the phenomenon tests in
``tests/geometries``."""
from __future__ import annotations

import numpy as np
import pytest

import rapidfem as rf
from rapidfem import structures as st

MM = 1e-3


def _box_of(obj) -> np.ndarray:
    """(xmin, ymin, zmin, xmax, ymax, zmax) over an object's faces."""
    b = np.array([f.bbox for f in obj.faces])
    return np.concatenate([b[:, :3].min(axis=0), b[:, 3:].max(axis=0)])


def test_coax_cuts_its_inner_conductor_and_ports_use_the_fill():
    g = rf.Geometry(maxh=0.5 * MM)
    cx = st.coax(g, ri=0.5 * MM, ro=1.5 * MM, length=4 * MM, axis="x",
                 material=rf.Dielectric(er=2.2), add_ports=True)
    # two annular caps, the inner-conductor wall and the shield
    assert len(cx.dielectric.faces) == 4
    assert [p.er for p in cx.ports] == [2.2, 2.2]
    assert cx.ports[1].origin == pytest.approx((4 * MM, 0.0, 0.0))
    assert cx.pec is not None
    g.mesh()
    assert g.mesh_stats.n_tets > 0


def test_rect_waveguide_lays_the_cross_section_across_the_axis():
    g = rf.Geometry(maxh=2 * MM)
    wg = st.rect_waveguide(g, a=4 * MM, b=2 * MM, length=10 * MM, axis="y",
                           add_ports=True)
    np.testing.assert_allclose(_box_of(wg.body), np.array([0, 0, 0, 4, 10, 2]) * MM,
                               atol=1e-12)
    assert [p.er for p in wg.ports] == [1.0, 1.0]
    assert wg.pec is not None


def test_planar_lines_share_one_pec_with_their_wave_ports():
    ms = st.microstrip(rf.Geometry(maxh=1 * MM), line_w=1 * MM, line_l=5 * MM,
                       sub_w=6 * MM, sub_h=0.5 * MM, air_h=3 * MM, er=3.0,
                       add_ports=True, f0=5e9)
    cw = st.cpw(rf.Geometry(maxh=1 * MM), signal_w=1 * MM, gap=0.5 * MM, line_l=5 * MM,
                sub_w=6 * MM, sub_h=0.5 * MM, air_h=3 * MM, er=3.0,
                add_ports=True, f0=5e9)
    sl = st.stripline(rf.Geometry(maxh=1 * MM), line_w=1 * MM, line_l=5 * MM,
                      sub_w=6 * MM, sub_h=1 * MM, er=3.0, add_ports=True, f0=5e9)
    for line in (ms, cw, sl):
        assert len(line.ports) == 2
        for port in line.ports:
            assert port.pec == [line.pec] and port.f0 == 5e9
    assert len(ms.port_a) == 2 and len(cw.port_b) == 2
    np.testing.assert_allclose(_box_of(sl.fill), np.array([-3, 0, 0, 3, 5, 1]) * MM,
                               atol=1e-12)


def test_circ_waveguide_ports_have_no_internal_pec():
    g = rf.Geometry(maxh=2 * MM)
    wg = st.circ_waveguide(g, radius=5 * MM, length=10 * MM, add_ports=True, f0=20e9)
    assert [p.pec for p in wg.ports] == [[], []]
    assert wg.pec is not None


@pytest.mark.parametrize("build", [
    lambda g: st.coax(g, ri=1 * MM, ro=1 * MM, length=1 * MM),
    lambda g: st.coax(g, ri=0.5 * MM, ro=1 * MM, length=1 * MM, axis="w"),
    lambda g: st.rect_waveguide(g, a=2 * MM, b=1 * MM, length=1 * MM, axis="w"),
    lambda g: st.circ_waveguide(g, radius=1 * MM, length=1 * MM, add_ports=True),
    lambda g: st.microstrip(g, line_w=1 * MM, line_l=5 * MM, sub_w=6 * MM,
                            sub_h=0.5 * MM, air_h=3 * MM, er=3.0, add_ports=True),
    lambda g: st.stripline(g, line_w=1 * MM, line_l=5 * MM, sub_w=6 * MM,
                           sub_h=1 * MM, er=3.0, add_ports=True),
    lambda g: st.cpw(g, signal_w=4 * MM, gap=1 * MM, line_l=5 * MM, sub_w=6 * MM,
                     sub_h=0.5 * MM, air_h=3 * MM, er=3.0),
], ids=["coax-radii", "coax-axis", "rect-axis", "circ-f0", "microstrip-f0",
        "stripline-f0", "cpw-ground-width"])
def test_builders_reject_bad_arguments(build):
    with pytest.raises(ValueError):
        build(rf.Geometry(maxh=1 * MM))
