# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""FEM-JSON bridge, consume rapidpassives' exportForFEM() JSON."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rapidfem.geometry import Geometry, GeoObject


FEM_JSON_SCHEMA_VERSIONS = (1,)

# Polygons coming out of mergeLayers may carry sliver edges (sub-nm jogs at
# rectangle-to-rectangle joins). Drop any vertex closer than this to its
# predecessor in xy, well below any feature of a layout, but large enough to
# wipe merge slivers.
_FEM_JSON_VERTEX_TOL_UM = 0.01


def _clean_polygon_um(poly_um: list) -> list:
    """Drop consecutive vertices closer than `_FEM_JSON_VERTEX_TOL_UM` in xy.

    rapidpassives' mergeLayers occasionally emits near-duplicate vertices at
    polygon joins (sub-nm slivers), edges far below any feature that would
    only force tiny elements. Closing-vertex duplicates are also stripped:
    polygons close themselves, an explicit trailing copy of the first
    vertex would be a zero-length edge.
    """
    if not poly_um:
        return []
    tol_sq = _FEM_JSON_VERTEX_TOL_UM ** 2
    out = [tuple(poly_um[0])]
    for x, y in poly_um[1:]:
        px, py = out[-1]
        if (x - px) ** 2 + (y - py) ** 2 < tol_sq:
            continue
        out.append((x, y))
    # Strip closing copy of the first vertex if present.
    if len(out) >= 2:
        fx, fy = out[0]
        lx, ly = out[-1]
        if (lx - fx) ** 2 + (ly - fy) ** 2 < tol_sq:
            out.pop()
    return out


@dataclass
class FemLayoutResult:
    """Output of :func:`from_fem_json`, a meshed-ready geometry plus the
    objects the caller needs to wire BCs/ports.

    Typical usage::

        from rapidfem import rfic, PEC, LumpedPort, ABC, Problem
        layout = rfic.from_fem_json("spiral.fem.json")
        all_conductors = [v for vs in layout.conductors.values() for v in vs]
        PEC(*(v.faces for v in all_conductors), layout.ground)
        for port in layout.ports.values():
            LumpedPort(port, direction=(0, 0, 1), z0=50.0)
        ABC(*layout.air.faces.outer)
        layout.geometry.mesh()
        result = Problem(layout.geometry).sweep([1e9, 10e9, 50e9])
    """
    geometry: "Geometry"
    conductors: dict[str, list["GeoObject"]]   # stack-layer id → 3-D conductor volumes
    ports: dict[str, "GeoObject"]               # port name → 2-D port plate
    ground: "GeoObject"                         # alias for the first ground patch
    ground_patches: list                        # one local ground per port (may merge)
    substrate: "GeoObject"
    oxide: "GeoObject"
    air: "GeoObject"
    doc: dict                                   # the parsed FEM-JSON (metadata + sim)


def from_fem_json(
    source,
    *,
    stack=None,
    via_mode: Literal["merged", "cells"] = "merged",
    footprint_margin: float = 0.3,
    air_height_um: float = 60.0,
    conductor_maxh_um: float = 1.5,
    port_maxh_um: float = 1.5,
    port_tab_um: float = 8.0,
    port_inset_um: float | None = None,
) -> FemLayoutResult:
    """Build a 3-D FEM geometry from a rapidpassives ``exportForFEM`` JSON.

    Conductors (metals AND vias) are extruded to their stack-layer thickness;
    every conductor surface is left for the caller to mark PEC. Ports are
    inset from each layout port's nominal location toward the layout centre
    so the plate top edge lands on the conductor's horizontal bottom face
    (PEC there constrains E_x/E_y but leaves E_z free, which the lumped-port
    drive needs).

    Parameters
    ----------
    source : str | pathlib.Path | dict
        Path to a ``.fem.json`` file or an already-parsed dict.
    stack : rfic.Stack, optional
        If given, replaces the substrate/oxide constants from the JSON's
        ``stack.substrate`` / ``stack.oxide`` block. The JSON's layer z-stack
        is always trusted (it carries the GDS-derived geometry).
    via_mode : {"merged", "cells"}
        "merged" (default) extrudes the merged bounding box of each via
        array → 1 conductor volume per array (fast). "cells" extrudes every
        individual via cell if the JSON provides ``polygon_cells``; falls
        back to the merged bbox when cells aren't present.
    footprint_margin : float
        Substrate/oxide/air enclosure margin as a fraction of the conductor
        bbox span. 0.3 = 30% on each side.
    air_height_um : float
        Air-box height above ``stack.top_z``.
    conductor_maxh_um : float
        Per-volume mesh-size cap for every extruded conductor.
    port_maxh_um : float
        Per-face mesh-size cap for the port plates + shared ground patch.
    port_tab_um : float
        Port plate width (extent perpendicular to the integration line).
    port_inset_um : float, optional
        Distance to move each port plate inward from the JSON's port
        location (toward layout centre). Default = ``port_tab_um / 2``,
        far enough to land the plate top edge inside the conductor's bottom
        face instead of on a side wall.

    Returns
    -------
    FemLayoutResult
    """
    # Local imports keep the rapidfem top-level lean even when the JSON
    # bridge isn't used.
    import json as _json
    from pathlib import Path as _Path
    from rapidfem.geometry import Geometry as _Geometry
    from rapidfem.materials import Air as _Air, Dielectric as _Dielectric

    if isinstance(source, (str, _Path)):
        with open(source) as f:
            doc = _json.load(f)
    elif isinstance(source, dict):
        doc = source
    else:
        raise TypeError(f"source must be str/Path/dict, got {type(source).__name__}")

    sv = doc.get("schema_version", 1)
    if sv not in FEM_JSON_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported FEM JSON schema_version {sv!r}; "
                         f"supported: {FEM_JSON_SCHEMA_VERSIONS}")

    stack_doc      = doc["stack"]
    layers_doc     = stack_doc["layers"]
    substrate_doc  = stack_doc["substrate"]
    oxide_doc      = stack_doc["oxide"]
    conductors_doc = doc["conductors"]
    ports_doc      = doc["ports"]

    # Layer lookup by id, used everywhere below.
    layer_by_id = {l["id"]: l for l in layers_doc}

    # Resolve port layer ids, must already be valid stack-layer ids. Older
    # exports (pre 2026-05-19) wrote generator-internal names like "m3" that
    # had no stable mapping to stack ids; those need to be re-exported.
    metals_by_z = sorted(
        (l for l in layers_doc if l["type"] == "metal"),
        key=lambda l: l["z_um"],
    )
    def _resolve_port_layer(port_layer: str) -> str:
        if port_layer in layer_by_id:
            return port_layer
        raise KeyError(
            f"port references unknown stack layer {port_layer!r}; the JSON "
            f"may have been exported with an older rapidpassives that emitted "
            f"generator-internal names (e.g. 'm3'). Re-export the layout. "
            f"Known stack layers: {sorted(layer_by_id)}")

    # Substrate + oxide constants, JSON wins unless the caller passed a stack.
    sub_thickness_um = substrate_doc.get("thickness_um", 300.0) or 300.0
    sub_er           = substrate_doc.get("er", 11.7)
    sub_rho_ohm_cm   = substrate_doc.get("rho_ohm_cm", 10.0) or 10.0
    sub_sigma        = 100.0 / sub_rho_ohm_cm   # σ [S/m] = 1 / (ρ [Ω·cm] · 0.01)
    ox_er            = oxide_doc.get("er", 4.2)
    ox_tand          = oxide_doc.get("tand", 0.0)

    if stack is not None:
        sub_er, sub_sigma = stack.substrate_er, stack.substrate_sigma
        ox_er, ox_tand    = stack.oxide_er, stack.oxide_tand

    silicon = _Dielectric(er=sub_er, conductivity=sub_sigma)
    sio2    = _Dielectric(er=ox_er,  tand=ox_tand)
    air_mat = _Air()

    # Layout bbox → footprint with margin, all in metres.
    xs, ys = [], []
    for c in conductors_doc:
        for x, y in c["polygon"]:
            xs.append(x); ys.append(y)
    if not xs:
        raise ValueError("no conductor polygons in FEM JSON")
    x_min_um, x_max_um = min(xs), max(xs)
    y_min_um, y_max_um = min(ys), max(ys)
    span_x_um = max(x_max_um - x_min_um, 1.0)
    span_y_um = max(y_max_um - y_min_um, 1.0)
    foot_w = (span_x_um + 2 * footprint_margin * span_x_um) * 1e-6
    foot_h = (span_y_um + 2 * footprint_margin * span_y_um) * 1e-6
    cx_m   = (x_min_um + x_max_um) / 2 * 1e-6
    cy_m   = (y_min_um + y_max_um) / 2 * 1e-6

    # Stack z range, bottom of lowest layer, top of highest layer.
    layers_sorted = sorted(layers_doc, key=lambda l: l["z_um"])
    z_bottom_um = layers_sorted[0]["z_um"]
    z_top_um    = layers_sorted[-1]["z_um"] + layers_sorted[-1]["thickness_um"]
    z_top_m     = z_top_um * 1e-6

    # Build the enclosure. Global maxh = ~10% of the smaller in-plane span,
    # finer-than-bulk meshing of conductors is set per-volume via maxh.
    g = _Geometry(maxh=min(foot_w, foot_h) / 10)

    substrate = g.box(foot_w, foot_h, sub_thickness_um * 1e-6,
                      position=(cx_m - foot_w / 2, cy_m - foot_h / 2,
                                z_bottom_um * 1e-6 - sub_thickness_um * 1e-6),
                      material=silicon)
    oxide = g.box(foot_w, foot_h, (z_top_um - z_bottom_um) * 1e-6,
                  position=(cx_m - foot_w / 2, cy_m - foot_h / 2, z_bottom_um * 1e-6),
                  material=sio2)
    air = g.box(foot_w, foot_h, air_height_um * 1e-6,
                position=(cx_m - foot_w / 2, cy_m - foot_h / 2, z_top_m),
                material=air_mat)

    # Extrude every conductor polygon to its stack-layer thickness.
    cond_maxh = conductor_maxh_um * 1e-6
    conductor_objects: dict[str, list] = {}
    all_conductors: list = []

    for c in conductors_doc:
        layer_id = c["layer"]
        layer = layer_by_id.get(layer_id)
        if layer is None:
            raise KeyError(f"conductor references unknown layer {layer_id!r}")
        z_lo  = layer["z_um"] * 1e-6
        thick = layer["thickness_um"] * 1e-6
        if thick <= 0:
            continue

        # Pick the polygon set, merged bbox, or per-cell array for vias.
        polys = [c["polygon"]]
        if via_mode == "cells" and c.get("polygon_cells"):
            polys = c["polygon_cells"]

        # Holes (annular conductors: rat-race ring, guard ring), only the
        # primary `polygon` entry carries holes; per-cell arrays are
        # individual solid pieces with no holes.
        holes_um = c.get("holes") if polys is not c.get("polygon_cells") else None

        for poly_um in polys:
            cleaned = _clean_polygon_um(poly_um)
            if len(cleaned) < 3:
                continue
            pts_3d = [(x * 1e-6, y * 1e-6, z_lo) for x, y in cleaned]
            poly_kwargs = {}
            if holes_um:
                cleaned_holes = [_clean_polygon_um(h) for h in holes_um]
                cleaned_holes = [
                    [(x * 1e-6, y * 1e-6, z_lo) for x, y in h]
                    for h in cleaned_holes if len(h) >= 3
                ]
                if cleaned_holes:
                    poly_kwargs["holes"] = cleaned_holes
            face = g.polygon(pts_3d, **poly_kwargs)
            vol = g.extrude(face, height=thick, material=oxide.material,
                            maxh=cond_maxh)
            vol.name = layer_id
            conductor_objects.setdefault(layer_id, []).append(vol)
            all_conductors.append(vol)

    # Shared local ground patch + one port plate per JSON port.
    # Ground sits on the lowest metal's TOP face, that's typically li1 in
    # sky130 and any other "ground shield" layer used by the generator.
    port_tab_m = port_tab_um * 1e-6
    inset_m    = (port_inset_um if port_inset_um is not None
                   else port_tab_um / 2) * 1e-6
    port_maxh  = port_maxh_um * 1e-6

    # Per-port ground reference: walk z-stack down from each port's metal
    # and use the topmost lower metal that has a conductor covering the
    # port's (inset) xy. That way ratrace / patch / microstrip pick up the
    # generator's own ground plane (port plate stays in the substrate-to-
    # trace gap), while sparser layouts like spiral fall through to the
    # lowest metal and get a synthesised local ground patch added below.
    metal_below_by_z = sorted(
        (l for l in layers_doc if l["type"] == "metal"),
        key=lambda l: l["z_um"],
    )

    def _polygon_at(layer_id, px, py):
        """Return the (xs, ys) of the conductor polygon on `layer_id` whose
        bbox covers (px, py), or None."""
        for c in conductors_doc:
            if c["layer"] != layer_id:
                continue
            poly = c["polygon"]
            xs = [pt[0] * 1e-6 for pt in poly]
            ys = [pt[1] * 1e-6 for pt in poly]
            if min(xs) <= px <= max(xs) and min(ys) <= py <= max(ys):
                return (min(xs), max(xs), min(ys), max(ys))
        return None

    def _polygons_same_net(top_id, top_xy, bot_id, bot_xy):
        """True if the polygons (top_xy on top_id, bot_xy on bot_id) are
        bridged by a via polygon (on any via-type layer between them)
        that sits inside BOTH bboxes. Means electrically same-net → not a
        valid lumped-port ground reference."""
        if top_xy is None or bot_xy is None:
            return False
        top_layer = layer_by_id[top_id]
        bot_layer = layer_by_id[bot_id]
        z_lo, z_hi = bot_layer["z_um"], top_layer["z_um"]
        for vl in layers_doc:
            if vl["type"] != "via" or not (z_lo < vl["z_um"] < z_hi):
                continue
            for c in conductors_doc:
                if c["layer"] != vl["id"]:
                    continue
                poly = c["polygon"]
                vxs = [pt[0] * 1e-6 for pt in poly]
                vys = [pt[1] * 1e-6 for pt in poly]
                cx = 0.5 * (min(vxs) + max(vxs))
                cy = 0.5 * (min(vys) + max(vys))
                tx0, tx1, ty0, ty1 = top_xy
                bx0, bx1, by0, by1 = bot_xy
                if (tx0 <= cx <= tx1 and ty0 <= cy <= ty1 and
                        bx0 <= cx <= bx1 and by0 <= cy <= by1):
                    return True
        return False

    resolved_ports = []
    for p in ports_doc:
        lay = _resolve_port_layer(p["layer"])
        port_layer = layer_by_id[lay]
        z_top = port_layer["z_um"] * 1e-6     # bottom of the port's metal
        px_raw = p["x_um"] * 1e-6
        py_raw = p["y_um"] * 1e-6

        # Inset toward the layout centre (cx_m, cy_m) so the plate top edge
        # lands on the conductor's bottom face rather than its side wall.
        dx, dy = px_raw - cx_m, py_raw - cy_m
        norm = math.hypot(dx, dy)
        if norm > 1e-15:
            px = px_raw - inset_m * dx / norm
            py = py_raw - inset_m * dy / norm
        else:
            px, py = px_raw, py_raw

        # Find this port's local ground: highest metal below the port's
        # metal that has a conductor covering (px, py). Fall back to the
        # lowest metal (li1 / poly) if none, we synthesise a local gnd
        # patch on that.
        port_gnd_z = (metal_below_by_z[0]["z_um"]
                      + metal_below_by_z[0]["thickness_um"]) * 1e-6
        port_needs_local_patch = True
        port_z = port_layer["z_um"]
        port_xy_box = _polygon_at(lay, px, py)
        for cand in reversed(metal_below_by_z):
            if cand["z_um"] >= port_z:
                continue
            cand_xy_box = _polygon_at(cand["id"], px, py)
            if cand_xy_box is None:
                continue
            # Skip same-net candidates: a via polygon between port_metal and
            # cand that lands inside both polygons connects them
            # electrically, using cand's top face as the port reference
            # would short the lumped port.
            if _polygons_same_net(lay, port_xy_box, cand["id"], cand_xy_box):
                continue
            port_gnd_z = (cand["z_um"] + cand["thickness_um"]) * 1e-6
            port_needs_local_patch = False
            break

        resolved_ports.append((p["name"], lay, px, py, z_top,
                               port_gnd_z, port_needs_local_patch))

    # Port plates with width TANGENT to the layout edge and PER-PORT gnd_z.
    # A port that sits over an existing ground plane stops at that plane;
    # a port without anything beneath drops to the lowest metal and we
    # synthesise a local ground patch there.
    port_objects: dict[str, "GeoObject"] = {}
    for name, _lay, px, py, z_top, port_gnd_z, _needs in resolved_ports:
        rx, ry = px - cx_m, py - cy_m
        rnorm = math.hypot(rx, ry)
        if rnorm > 1e-15:
            tx, ty = -ry / rnorm, rx / rnorm
        else:
            tx, ty = 0.0, 1.0     # fallback for port at the layout centre

        w_x = tx * port_tab_m
        w_y = ty * port_tab_m
        p0 = (px - tx * port_tab_m / 2,
              py - ty * port_tab_m / 2,
              port_gnd_z)
        port = g.plate(
            p0=p0,
            width=(w_x, w_y, 0),
            height=(0, 0, z_top - port_gnd_z),
            maxh=port_maxh,
        )
        port.name = name
        port_objects[name] = port

    # Only synthesise local ground patches for ports that have no real
    # ground plane below them. Cluster those by proximity so co-located
    # ports share a patch (GSG-style).
    needs_local = [
        i for i, rp in enumerate(resolved_ports) if rp[6]   # _needs flag
    ]
    cluster_dist = 4 * port_tab_m
    parent = list(range(len(resolved_ports)))
    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def _union(i, j):
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj
    for ii in range(len(needs_local)):
        for jj in range(ii + 1, len(needs_local)):
            i, j = needs_local[ii], needs_local[jj]
            xi, yi = resolved_ports[i][2], resolved_ports[i][3]
            xj, yj = resolved_ports[j][2], resolved_ports[j][3]
            if math.hypot(xi - xj, yi - yj) < cluster_dist:
                _union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in needs_local:
        clusters.setdefault(_find(i), []).append(i)

    gpad = max(port_tab_m, 4e-6)
    ground_patches: list = []
    for members in clusters.values():
        xs = [resolved_ports[i][2] for i in members]
        ys = [resolved_ports[i][3] for i in members]
        gx_min, gx_max = min(xs) - gpad, max(xs) + gpad
        gy_min, gy_max = min(ys) - gpad, max(ys) + gpad
        # All ports in a cluster share the same gnd_z (they routed there
        # because nothing covered them on any higher metal).
        cluster_gnd_z = resolved_ports[members[0]][5]
        gnd_patch = g.xy_plate(
            gx_max - gx_min, gy_max - gy_min,
            position=(gx_min, gy_min, cluster_gnd_z),
            maxh=port_maxh,
        )
        gnd_patch.name = "gnd_" + "_".join(resolved_ports[i][0] for i in members)
        ground_patches.append(gnd_patch)

    ground = ground_patches[0] if ground_patches else g.xy_plate(
        1e-6, 1e-6,
        position=(cx_m, cy_m,
                  (metals_by_z[0]['z_um'] + metals_by_z[0]['thickness_um']) * 1e-6),
        maxh=port_maxh)

    # One conformal fragment over everything that lives inside oxide + air.
    g.fragment(oxide, substrate, *all_conductors,
               *ground_patches, *port_objects.values(), air)

    return FemLayoutResult(
        geometry=g,
        conductors=conductor_objects,
        ports=port_objects,
        ground=ground,
        ground_patches=ground_patches,
        substrate=substrate,
        oxide=oxide,
        air=air,
        doc=doc,
    )


__all__ = ["from_fem_json", "FemLayoutResult", "FEM_JSON_SCHEMA_VERSIONS"]
