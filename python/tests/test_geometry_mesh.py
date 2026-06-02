"""Geometry + mesh pipeline tests with report artefacts.

These exercise the *meshing* half of rapidfem, upstream of any FE solve:
the gmsh-backed `rf.Geometry` builder, its primitives, face selectors,
material/port/BC assignment, and the mesh-level introspection that the
solver and the UI both read back out.

Two introspection surfaces are used:

- `rapidfem.ui.serialize.mesh_to_payload(g, maxh=...)` — the cheap,
  solver-free path. It meshes the live gmsh model and reads nodes / tets
  / surface tris straight out of gmsh, returning a payload whose
  ``["stats"]`` carries ``n_nodes`` / ``n_tets`` / ``n_tris`` and whose
  ``["nodes"]`` / ``["tets"]`` are the raw flat arrays (used here to
  check every tet has positive volume).
- `rf.ProblemTD(g)._op` — the native DGTD operator. After construction it
  exposes mesh-derived counts without a solve: ``n_ports()``,
  ``port_n_faces(i)``, ``port_has_mode(i)``, ``n_dispersive()``.

Meshing is cheap relative to a transient/sweep, so none of these are
marked ``slow`` — they run in the default set.
"""
from __future__ import annotations

import numpy as np
import pytest

import rapidfem as rf
from rapidfem.ui.serialize import mesh_to_payload

MM = 1e-3


def _payload_arrays(payload):
    """Reshape a mesh payload's flat node/tet lists into numpy arrays.

    Returns ``(nodes (n_nodes, 3) float64, tets (n_tets, 4) int)``.
    """
    nodes = np.asarray(payload["nodes"], dtype=float).reshape(-1, 3)
    tets = np.asarray(payload["tets"], dtype=np.int64).reshape(-1, 4)
    return nodes, tets


def _tet_volumes(nodes, tets):
    """Signed volume of every tet, as ``|det[v1-v0, v2-v0, v3-v0]| / 6``
    (returned unsigned)."""
    p0 = nodes[tets[:, 0]]
    e1 = nodes[tets[:, 1]] - p0
    e2 = nodes[tets[:, 2]] - p0
    e3 = nodes[tets[:, 3]] - p0
    det = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
    return np.abs(det) / 6.0


# -----------------------------------------------------------------------------
# 1. Box-mesh sanity: counts plausible, every tet has positive volume, and the
#    discrete volume sums to the analytic box volume.
# -----------------------------------------------------------------------------

def test_box_mesh_sanity(report):
    """Mesh a single dielectric box and gate the basic mesh invariants:
    positive node/tet counts, every tet non-degenerate (positive volume),
    and the summed tet volume recovering the analytic box volume.
    """
    report.note(
        "A single 20x20x20 mm dielectric box meshed at maxh = 5 mm. Gates the "
        "basic mesh invariants read straight out of gmsh (no FE solve): "
        "positive node and tet counts, every tetrahedron non-degenerate "
        "(strictly positive volume), and the summed discrete volume matching "
        "the analytic box volume to within meshing tolerance."
    )
    side = 20.0 * MM
    g = rf.Geometry(maxh=5.0 * MM)
    g.box(side, side, side, material=rf.Dielectric(er=2.0))
    payload = mesh_to_payload(g, maxh=5.0 * MM)
    stats = payload["stats"]
    n_nodes, n_tets = stats["n_nodes"], stats["n_tets"]
    print(f"  n_nodes={n_nodes}  n_tets={n_tets}  n_tris={stats['n_tris']}")

    assert n_nodes > 0, "mesh produced no nodes"
    assert n_tets > 0, "mesh produced no tets"
    # A 4x4x4-cell box at maxh = side/4 is comfortably in the hundreds of
    # tets; a handful would mean the size field was ignored.
    assert n_tets > 50, f"implausibly coarse mesh: only {n_tets} tets"

    nodes, tets = _payload_arrays(payload)
    assert tets.shape[0] == n_tets, "tet array / stats count disagree"
    assert nodes.shape[0] == n_nodes, "node array / stats count disagree"

    vols = _tet_volumes(nodes, tets)
    min_vol = float(vols.min())
    assert min_vol > 0.0, f"degenerate tet with volume {min_vol:.3e}"

    discrete_vol = float(vols.sum())
    analytic_vol = side ** 3
    rel_vol_err = abs(discrete_vol - analytic_vol) / analytic_vol
    print(f"  sum tet vol={discrete_vol:.4e}  analytic={analytic_vol:.4e}  "
          f"rel err={rel_vol_err:.2e}")
    # A conformal tet mesh tiles the box exactly up to floating point.
    assert rel_vol_err < 1e-6, (
        f"summed tet volume off by {rel_vol_err:.2e} from analytic"
    )

    report.table(
        "mesh statistics",
        ["quantity", "value"],
        [["nodes", n_nodes],
         ["tets", n_tets],
         ["surface tris", stats["n_tris"]],
         ["min tet volume [m^3]", f"{min_vol:.3e}"],
         ["mesh time [s]", f"{stats['mesh_time_s']:.3f}"]],
    )
    report.metric("node count", n_nodes, lower=1.0,
                  detail="gmsh getNodes() count")
    report.metric("tet count", n_tets, lower=50.0,
                  detail="4-node tetrahedra at maxh = side/4")
    report.metric("min tet volume", min_vol, lower=0.0, unit="m^3",
                  detail="smallest tet volume; > 0 means no inverted/flat tets")
    report.metric("discrete vs analytic box volume", rel_vol_err,
                  bound=1e-6,
                  detail="|sum(tet vol) - side^3| / side^3")

    # Scatter the mesh nodes projected onto z = 0 as a quick visual.
    report.plot_complex_plane(
        {"mesh nodes (xy)": nodes[:, 0] / MM + 1j * nodes[:, 1] / MM},
        title="box mesh nodes projected to xy",
        xlabel="x  [mm]",
        ylabel="y  [mm]",
        caption=f"{n_nodes} nodes of the {n_tets}-tet box mesh.",
    )


# -----------------------------------------------------------------------------
# 2. maxh refinement / monotonicity: a finer maxh yields a finer mesh.
# -----------------------------------------------------------------------------

def test_maxh_refinement_monotonic(report):
    """Mesh the same box at a sequence of decreasing maxh values and gate
    that the tet count grows monotonically as the mesh is refined. Each
    geometry is built and meshed in turn because the gmsh OCC model is a
    process-global singleton (only one live `Geometry` at a time).
    """
    report.note(
        "The same 24x24x24 mm air box meshed at three decreasing maxh values "
        "(4, 3, 2 mm). Gates that refining the mesh size strictly increases "
        "the tet count (a finer maxh must never produce fewer tets). The maxh "
        "values sit in gmsh's responsive regime: for this small box, coarse "
        "sizes (>= ~5 mm) all saturate to the same minimum-quality mesh, so "
        "the sweep starts where refinement actually bites. gmsh's OCC model "
        "is a process-global singleton, so each maxh is meshed on its own "
        "freshly built Geometry, one at a time."
    )
    side = 24.0 * MM
    maxhs = [4.0 * MM, 3.0 * MM, 2.0 * MM]
    n_tets = []
    for h in maxhs:
        g = rf.Geometry(maxh=h)
        g.box(side, side, side, material=rf.Air())
        stats = mesh_to_payload(g, maxh=h)["stats"]
        n_tets.append(stats["n_tets"])
        print(f"  maxh={h/MM:.1f} mm -> {stats['n_tets']} tets")

    assert all(n > 0 for n in n_tets), "a mesh came back empty"
    # Strict monotonicity: each refinement step adds tets.
    for coarse, fine, hc, hf in zip(n_tets, n_tets[1:], maxhs, maxhs[1:]):
        assert fine > coarse, (
            f"refining maxh {hc/MM:.1f} -> {hf/MM:.1f} mm did not add tets "
            f"({coarse} -> {fine})"
        )

    report.table(
        "maxh vs tet count",
        ["maxh [mm]", "n_tets"],
        [[h / MM, n] for h, n in zip(maxhs, n_tets)],
    )
    report.plot_xy(
        np.asarray(maxhs) / MM,
        {"n_tets": np.asarray(n_tets, dtype=float)},
        xlabel="maxh  [mm]",
        ylabel="tetrahedra",
        title="mesh refinement vs maxh",
        markers=True,
        caption="Finer maxh (smaller x) yields more tets.",
    )
    report.metric(
        "tet-count growth (finest / coarsest)",
        n_tets[-1] / n_tets[0],
        lower=1.0,
        detail=f"{n_tets[-1]} tets at {maxhs[-1]/MM:.0f} mm vs "
               f"{n_tets[0]} at {maxhs[0]/MM:.0f} mm",
    )
    report.metric(
        "finest-minus-coarsest tet delta",
        float(n_tets[-1] - n_tets[0]),
        lower=1.0,
        detail="monotone refinement adds tets",
    )


# -----------------------------------------------------------------------------
# 3. Port-face + PEC assignment on a WR-90-like guide.
# -----------------------------------------------------------------------------

def test_port_and_pec_assignment(report):
    """WR-90-like air box with a RectWaveguidePort on each z-end and PEC on
    the four side walls. Gate that the operator sees exactly two modal
    ports, each resolving a positive number of port faces.
    """
    report.note(
        "A WR-90-like air box (22.86 x 10.16 x 30 mm) with a "
        "RectWaveguidePort on each z-end face and PEC on the four side walls "
        "(everything unassigned). After operator construction we gate the "
        "mesh-level port introspection: exactly two ports, each carrying a "
        "waveguide mode and resolving a positive number of (element, face) "
        "pairs from the meshed port surface."
    )
    a_wg, b_wg, guide_l = 22.86 * MM, 10.16 * MM, 30.0 * MM
    g = rf.Geometry(maxh=7.0 * MM)
    air = g.box(a_wg, b_wg, guide_l, material=rf.Air())
    rf.RectWaveguidePort(air.faces.min(axis="z"))
    rf.RectWaveguidePort(air.faces.max(axis="z"))
    rf.PEC(*air.faces.unassigned)
    g.mesh()

    ptd = rf.ProblemTD(g, order=1, flux="upwind")
    op = ptd._op
    n_ports = op.n_ports()
    print(f"  n_ports={n_ports}")
    assert n_ports == 2, f"expected 2 ports, operator reports {n_ports}"

    face_counts = [op.port_n_faces(i) for i in range(n_ports)]
    has_mode = [op.port_has_mode(i) for i in range(n_ports)]
    print(f"  port_n_faces={face_counts}  has_mode={has_mode}")
    for i, (nf, hm) in enumerate(zip(face_counts, has_mode)):
        assert nf > 0, f"port {i} resolved no port faces"
        assert hm, f"port {i} carries no waveguide mode"

    report.table(
        "port introspection",
        ["port", "n_faces", "has_mode"],
        [[i, nf, hm] for i, (nf, hm) in enumerate(zip(face_counts, has_mode))],
    )
    report.metric("ports detected", n_ports, ref=2, atol=0,
                  detail="two RectWaveguidePort faces (z-min, z-max)")
    report.metric("port 0 faces", face_counts[0], lower=1.0,
                  detail="resolved (element, local-face) pairs on z-min port")
    report.metric("port 1 faces", face_counts[1], lower=1.0,
                  detail="resolved (element, local-face) pairs on z-max port")


# -----------------------------------------------------------------------------
# 4. Material tagging: a Debye-dispersive inclusion is seen by the operator.
# -----------------------------------------------------------------------------

def test_material_tagging_dispersive(report):
    """An air cavity with a Debye-dispersive block. The operator's
    `n_dispersive()` counts the tets carrying a dispersive material, so a
    cavity *with* the block must report a positive dispersive count while
    an otherwise-identical plain-dielectric reference reports zero — i.e.
    the material tag is actually routed through the mesh to the solver.
    """
    report.note(
        "A 24 mm air cavity containing a 9 mm centred block. With the block "
        "tagged as a Debye-dispersive material the operator's n_dispersive() "
        "must count the block's tets (> 0); with the block swapped for a "
        "plain (non-dispersive) dielectric the count must be exactly zero. "
        "This confirms the per-volume material tag survives meshing and "
        "reaches the native operator."
    )
    side = 24.0 * MM
    block = 9.0 * MM
    pos = ((side - block) / 2,) * 3

    def _build(block_material):
        g = rf.Geometry(maxh=side / 3)
        air = g.box(side, side, side, material=rf.Air())
        centre = g.box(block, block, block, position=pos,
                       material=block_material)
        g.fragment(air, centre)
        rf.PEC(*air.faces.unassigned)
        g.mesh()
        return rf.ProblemTD(g, order=1, flux="upwind")

    debye = rf.Material(
        er=4.6, debye=rf.Debye(er_inf=4.6, er_static=78.0, tau_s=8e-12),
    )
    ptd_disp = _build(debye)
    n_disp = ptd_disp._op.n_dispersive()
    print(f"  dispersive build: n_dispersive={n_disp}")

    # Reference: identical geometry, block is plain (non-dispersive).
    ptd_ref = _build(rf.Dielectric(er=4.6))
    n_ref = ptd_ref._op.n_dispersive()
    print(f"  reference build:  n_dispersive={n_ref}")

    assert n_disp > 0, (
        "Debye material did not reach the operator (n_dispersive == 0)"
    )
    assert n_ref == 0, (
        f"plain dielectric reported {n_ref} dispersive tets, expected 0"
    )

    report.table(
        "dispersive-tet count by block material",
        ["block material", "n_dispersive"],
        [["Debye (er_inf=4.6, er_s=78, tau=8 ps)", n_disp],
         ["plain Dielectric(er=4.6)", n_ref]],
    )
    report.metric("dispersive tets (Debye block)", n_disp, lower=1.0,
                  detail="operator-side count of Debye-tagged tets")
    report.metric("dispersive tets (plain ref)", n_ref, ref=0, atol=0,
                  detail="non-dispersive reference must report zero")
