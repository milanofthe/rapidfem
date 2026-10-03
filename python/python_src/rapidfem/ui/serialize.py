# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Serialize rapidfem objects into JSON payloads the viewer can consume.

The bundled canvas3d viewer expects per-entity buffers in the form
``{ name, tag, color: [r,g,b], positions: number[], normals: number[] }``
where ``positions`` and ``normals`` are flat float arrays (3 components per
vertex, 3 vertices per triangle, flat-shaded, no indexing).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any


# ── Geometry → triangle payload ───────────────────────────────────────────────


def _material_label(material) -> str | None:
    """Render a tracked entity's ``.material`` into a JSON-safe display string.

    A short label like ``"Dielectric (εr=4.4)"`` so the UI legend shows
    something meaningful without dragging the whole object into the JSON
    payload; ``None`` without a material.
    """
    if material is None:
        return None
    cls = type(material).__name__
    er = getattr(material, "er", None)
    er_diag = getattr(material, "er_diag", None)
    if er_diag is not None:
        return f"{cls} (εr=[{er_diag[0]:.2g},{er_diag[1]:.2g},{er_diag[2]:.2g}])"
    if er is not None and abs(er - 1.0) > 1e-9:
        return f"{cls} (εr={er:.3g})"
    return cls


# Signature palette mirrored from `lib/theme.ts`. Kept here as floats so the
# serializer can emit RGB triples directly without pulling theme.ts into the
# Python side. Updates to either side must be matched.
# Every driven-port physics class. Their faces always render lava (the accent
# color) and carry a "Port N" label, in the geometry preview and the FEM mesh
# alike. Keep this in sync with the port classes in physics.py.
_PORT_CLASSES = frozenset({
    "RectWaveguidePort", "LumpedPort", "CoaxPort", "WavePort",
    "UserDefinedPort", "FloquetPort",
})

_COL_PORT     = [0xd9 / 255, 0x51 / 255, 0x3c / 255]   # accent (lava)
_COL_METAL    = [0xe8 / 255, 0x94 / 255, 0x4a / 255]   # accentSecondary (yellow)
_COL_AIR      = [0x5a / 255, 0x5a / 255, 0x62 / 255]   # neutral gray
_COL_PML      = [0x7b / 255, 0x5e / 255, 0x8a / 255]   # muted purple
_COL_NEUTRAL  = [0x55 / 255, 0x55 / 255, 0x5a / 255]   # untracked surfaces
_COL_DIELECTRIC_CYCLE = [
    [0x4a / 255, 0x9e / 255, 0xc2 / 255],   # blue
    [0x6b / 255, 0xbf / 255, 0x8a / 255],   # green
    [0x7b / 255, 0x5e / 255, 0x8a / 255],   # purple
    [0xa7 / 255, 0x8b / 255, 0xd9 / 255],   # light purple
    [0xc4 / 255, 0xc4 / 255, 0x6b / 255],   # olive
]


def _material_color(material) -> list[float]:
    """Signature-palette color for a Material instance."""
    if material is None:
        return _COL_NEUTRAL
    cls = type(material).__name__
    if cls == "Air":
        return _COL_AIR
    if cls == "Conductor":
        return _COL_METAL
    if cls in ("Dielectric", "Anisotropic", "Material"):
        idx = (id(material) // 16) % len(_COL_DIELECTRIC_CYCLE)
        return _COL_DIELECTRIC_CYCLE[idx]
    return _COL_NEUTRAL


def _physics_color(phys) -> list[float]:
    """Signature-palette color for a physics object."""
    cls = type(phys).__name__
    if cls in ("PEC", "PMC", "SurfaceImpedance", "LumpedElement"):
        return _COL_METAL
    if cls in _PORT_CLASSES:
        return _COL_PORT
    if cls == "PML":
        return _COL_PML
    # ABC and other transparent boundaries fall through.
    return _COL_NEUTRAL


def _physics_label(phys, port_index: int | None = None) -> str:
    """Human-readable label for a physics object (e.g. 'Port 1', 'PEC')."""
    cls = type(phys).__name__
    if cls in _PORT_CLASSES:
        return f"Port {port_index}" if port_index is not None else "Port"
    pretty = {
        "PEC": "PEC", "PMC": "PMC", "ABC": "ABC", "PML": "PML",
        "SurfaceImpedance": "Surface Impedance",
        "LumpedElement": "Lumped Element",
    }
    return pretty.get(cls, cls)


def _entity_resolution(g, ent):
    """Resolve a tracked entity to a (name, color) pair for the preview.

    The name follows the SAME ``<class>_<index>`` convention the FEM mesh emits
    at ``g.mesh()`` time (driven ports collapse to ``port_<N>``, materials to
    ``dielectric_<N>`` / ``air_<N>`` / ``conductor_<N>``), so the viewer's
    ``classify`` / ``pretty_label`` / ``color_for`` pipeline treats the geometry
    preview and the meshed model identically, ports render lava and read
    "Port N" before and after meshing alike. The color is computed here too and
    is what the wireframe fallback draws with directly.
    """
    # Faces: prefer a physics override, named exactly as the mesh would.
    if ent.dim == 2:
        key_count: dict[str, int] = {}
        for phys in getattr(g, "_physics", []):
            cls = type(phys).__name__
            key = "port" if cls in _PORT_CLASSES else cls.lower()
            key_count[key] = key_count.get(key, 0) + 1
            for pe in getattr(phys, "_entities", ()):
                if pe == ent:
                    return f"{key}_{key_count[key]}", _physics_color(phys)
        if ent.name:
            return ent.name, _color_from_name(ent.name)
        return None, _COL_NEUTRAL
    # Volumes: material-typed, indexed per material class (matches the mesh).
    if ent.dim == 3:
        mat = ent.material
        if mat is not None:
            cls = type(mat).__name__.lower()
            order: list[int] = []
            for e in g.objects:
                m = e.material
                if m is not None and type(m).__name__.lower() == cls and id(m) not in order:
                    order.append(id(m))
            idx = (order.index(id(mat)) + 1) if id(mat) in order else 1
            return f"{cls}_{idx}", _material_color(mat)
        if ent.name:
            return ent.name, _color_from_name(ent.name)
        return None, _COL_NEUTRAL
    return None, _COL_NEUTRAL


def _color_from_name(name: str) -> list[float]:
    """Stable, evenly-distributed RGB color in [0,1] from a name.

    Used only as a last-resort fallback for legacy string-named entities;
    the object-API path goes through ``_entity_resolution`` instead.
    """
    if not name:
        name = "_unnamed"
    h = hashlib.md5(name.encode("utf-8")).digest()
    # golden-ratio hue, varying saturation/value just enough to separate
    hue = (h[0] / 255.0)
    sat = 0.45 + (h[1] / 255.0) * 0.30
    val = 0.65 + (h[2] / 255.0) * 0.25
    # HSV → RGB
    i = int(hue * 6)
    f = hue * 6 - i
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)
    table = [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)]
    return list(table[i % 6])


_DEFAULT_BBOX = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}


def _bbox(b) -> dict[str, list[float]]:
    """``{min, max}`` of an ``(xmin, ymin, zmin, xmax, ymax, zmax)`` box."""
    b = [float(v) for v in b]
    if len(b) != 6 or not all(math.isfinite(v) for v in b):
        return dict(_DEFAULT_BBOX)
    return {"min": b[:3], "max": b[3:]}


def geometry_to_payload(g: Any, *, target_tris: int = 4000) -> dict:
    """Viewer payload of a Geometry: its solver mesh once ``g.mesh()`` ran,
    else a coarse triangulation of its faces.

    ``rapidfem serve`` calls this on every save; the preview surface mesh is
    separate from the solver mesh and never touches it. ``target_tris`` is
    the preview's triangle budget.
    """
    if getattr(g, "_fem_mesh", None) is not None:
        try:
            return mesh_to_payload(g, maxh=0.0)
        except Exception:
            pass
    return _surface_preview(g, target_tris)


def _surface_preview(g: Any, target_tris: int) -> dict:
    """Coarse face triangulation, filled, colored per physics and material."""
    entities: list[dict] = []
    empty = {"kind": "geometry", "bbox": dict(_DEFAULT_BBOX), "entities": entities,
             "stats": {"n_entities": 0, "n_triangles": 0, "maxh": 0.0}}
    if g._native.mesh_mode:
        return empty  # a loaded mesh shows once g.mesh() bakes it
    native = g._native
    try:
        faces = {fid: (p, n) for fid, p, n in native.preview(target_tris)}
        bbox = native.bbox()
    except Exception as e:  # noqa: BLE001
        return {**empty, "error": str(e)}
    diag = math.dist(bbox[:3], bbox[3:])

    # Three passes. Physics-targeted faces (ports, PEC, ABC, ...) claim their
    # faces first; then each volume picks up whatever of its boundary remains
    # and tints it with its material, so a substrate reads as one dielectric
    # body instead of a litter of gray faces; the rest is neutral.
    claimed: set[int] = set()

    def emit(name, tag, dim, color, ids, material):
        pos: list[float] = []
        nor: list[float] = []
        for i in ids:
            if i in claimed or i not in faces:
                continue
            claimed.add(i)
            pos.extend(faces[i][0])
            nor.extend(faces[i][1])
        if pos:
            entities.append({
                "name": name, "tag": int(tag), "dim": int(dim),
                "color": color, "positions": pos, "normals": nor,
                "material": material,
            })

    objects = [o._entity for o in g.objects]
    for ent in objects + [e for p in g._physics for e in p._entities]:
        if ent.dim != 2:
            continue
        label, color = _entity_resolution(g, ent)
        if label is None:
            continue  # untracked, a volume claims it next
        emit(label, ent.key[0], 2, color, native.face_ids([ent.key]),
             _material_label(ent.material))
    # later volumes first: an inner body added after its surrounding
    # volume takes the interface between them
    for ent in reversed(objects):
        if ent.dim != 3:
            continue
        label, color = _entity_resolution(g, ent)
        emit(label or ent.name or f"_volume_{ent.key}", ent.key, 3, color,
             native.face_ids(native.faces_of(ent.key)), _material_label(ent.material))
    for fid in sorted(faces):
        emit(f"_face_{fid}", fid, 2, _COL_NEUTRAL, [fid], None)

    return {
        "kind": "geometry",
        "bbox": _bbox(bbox),
        "entities": entities,
        "stats": {
            "n_entities": len(entities),
            "n_triangles": sum(len(e["positions"]) // 9 for e in entities),
            "maxh": diag / 10.0,
        },
    }


# ── Solver mesh → viewer payload ──────────────────────────────────────────────


def mesh_to_payload(g: Any, *, maxh: float) -> dict:
    """The Geometry's solver mesh as a viewer payload: nodes, tets and the
    boundary and group triangles, each tagged with its group.

    Meshes first (with ``maxh``, 0 for the geometry's own size) unless
    ``g.mesh()`` already ran.
    """
    import time
    t0 = time.perf_counter()
    if getattr(g, "_fem_mesh", None) is None:
        g.mesh(maxh=maxh or None)
    t_mesh = time.perf_counter() - t0
    nodes, tris, tri_tags, tets, tet_tags = g._fem_mesh.viewer()

    groups = g._native.group_names()
    phys_dim = {t: dim for t, _, dim in groups}
    phys_names = {t: name for t, name, _ in groups}
    xyz = [nodes[k::3] for k in range(3)]
    bbox = ([min(c) for c in xyz] + [max(c) for c in xyz]) if nodes else []
    return {
        "kind": "mesh",
        "bbox": _bbox(bbox),
        "nodes": nodes,
        "tris": tris,
        "tri_phys": tri_tags,
        "tets": tets,
        "tet_phys": tet_tags,
        "phys_names": phys_names,
        "phys_dim": phys_dim,
        "name_to_tag": {n: t for t, n in phys_names.items()},
        "stats": {
            "n_nodes": len(nodes) // 3,
            "n_tets": len(tet_tags),
            "n_tris": len(tri_tags),
            "mesh_time_s": t_mesh,
            "msh_bytes": 0,
        },
    }
