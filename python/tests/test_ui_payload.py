# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""The viewer payloads of ``rapidfem.ui.serialize``: the face preview before
``g.mesh()``, the solver mesh after."""
from __future__ import annotations

import rapidfem as rf
from rapidfem.ui.serialize import geometry_to_payload

MM = 1e-3


def _guide():
    g = rf.Geometry(maxh=3 * MM)
    air = g.box(22.86 * MM, 10.16 * MM, 30 * MM, material=rf.Air())
    g.box(5 * MM, 5 * MM, 5 * MM, position=(8 * MM, 2 * MM, 10 * MM),
          material=rf.Dielectric(er=4.0))
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.PEC(*air.faces.outer.unassigned)
    return g


def test_preview_colors_faces_by_physics_and_material():
    p = geometry_to_payload(_guide())
    assert p["kind"] == "geometry"
    names = [e["name"] for e in p["entities"]]
    assert names[0] == "port_1"
    assert "dielectric_1" in names and "pec_1" in names
    for e in p["entities"]:
        assert len(e["positions"]) == len(e["normals"]) > 0
        assert len(e["positions"]) % 9 == 0
    assert p["bbox"]["max"] == [22.86 * MM, 10.16 * MM, 30 * MM]


def test_mesh_payload_after_meshing():
    g = _guide()
    g.mesh()
    p = geometry_to_payload(g)
    assert p["kind"] == "mesh"
    n = p["stats"]["n_nodes"]
    assert len(p["nodes"]) == 3 * n
    assert len(p["tets"]) == 4 * len(p["tet_phys"]) == 4 * g.mesh_stats.n_tets
    assert len(p["tris"]) == 3 * len(p["tri_phys"])
    assert max(p["tets"]) < n and max(p["tris"]) < n
    by_name = {n: t for t, n in p["phys_names"].items()}
    assert p["phys_dim"][by_name["port_1"]] == 2
    assert p["phys_dim"][by_name["dielectric_1"]] == 3
    assert set(p["tet_phys"]) == {by_name["air_1"], by_name["dielectric_1"]}
    assert set(p["tri_phys"]) == {by_name["port_1"], by_name["pec_1"]}


def test_td_trajectory_payload():
    """The native viewer export: merged corner nodes, tets and quantised
    per-node |E|, |H| frames, decimated to max_frames."""
    import numpy as np
    import rapidfem as rf
    from rapidfem.ui.api import _td_trajectory_payload

    p = rf.ProblemTD.box(size=(1, 1, 1), cells=(2, 2, 2), order=2)
    y0 = np.random.default_rng(0).standard_normal(p.n_dofs)
    traj = p.transient(y0, dt=0.05, steps=9, method="explicit", verbose=False)
    out = _td_trajectory_payload(traj, max_frames=4)
    n = out["n_node"]
    # a 2x2x2 box: 3 x 3 x 3 corner nodes shared by all tets
    assert n == 27 and len(out["nodes"]) == 3 * n
    assert len(out["tets"]) == 4 * out["n_elem"] == 4 * p.n_tets
    assert max(out["tets"]) == n - 1
    assert out["n_snapshots"] == len(out["times"]) == len(out["frames_e"]) == 4
    assert all(len(f) == n and 0 <= min(f) and max(f) <= 1000 for f in out["frames_e"])
    assert max(max(f) for f in out["frames_e"]) == 1000
    assert out["field_max"]["E"] > 0
