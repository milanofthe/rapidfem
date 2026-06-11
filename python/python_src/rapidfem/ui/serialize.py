"""Serialize rapidfem objects into JSON payloads the viewer can consume.

Two payload kinds, both built straight from the declarative geometry's
rapidmesh arrays (no gmsh):

- ``kind = "geometry"``: the pre-mesh PREVIEW. A coarse rapidmesh mesh is
  generated on a deep copy of the geometry (the user's discretization state
  is untouched), physics selections resolve on it, and every physics target /
  volume boundary becomes one flat-shaded triangle-soup entity with the
  signature palette (ports lava, PEC metal, dielectrics cycled).
- ``kind = "mesh"``: the FEM mesh after ``g.mesh()``: nodes, tets and tagged
  surface tris with physical names, plus the mesher's feature EDGES for the
  wireframe overlay.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


# ── Palette and labels ────────────────────────────────────────────────────────


def _material_label(material) -> str | None:
    """Render a ``.material`` into a JSON-safe display string."""
    if material is None:
        return None
    if isinstance(material, str):
        return material
    cls = type(material).__name__
    er = getattr(material, "er", None)
    er_diag = getattr(material, "er_diag", None)
    if er_diag is not None:
        return f"{cls} (εr=[{er_diag[0]:.2g},{er_diag[1]:.2g},{er_diag[2]:.2g}])"
    if er is not None and abs(er - 1.0) > 1e-9:
        return f"{cls} (εr={er:.3g})"
    return cls


# Signature palette mirrored from `lib/theme.ts`. Updates must be matched.
_PORT_CLASSES = frozenset({
    "RectWaveguidePort", "LumpedPort", "CoaxPort", "WavePort",
    "UserDefinedPort", "FloquetPort",
})

_COL_PORT     = [0xd9 / 255, 0x51 / 255, 0x3c / 255]   # accent (lava)
_COL_METAL    = [0xe8 / 255, 0x94 / 255, 0x4a / 255]   # accentSecondary
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
    if material is None:
        return _COL_NEUTRAL
    if isinstance(material, str):
        return _color_from_name(material)
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
    cls = type(phys).__name__
    if cls in ("PEC", "PMC", "SurfaceImpedance", "LumpedElement"):
        return _COL_METAL
    if cls in _PORT_CLASSES:
        return _COL_PORT
    if cls == "PML":
        return _COL_PML
    return _COL_NEUTRAL


def _color_from_name(name: str) -> list[float]:
    """Stable RGB fallback color for legacy string labels."""
    if not name:
        name = "_unnamed"
    h = hashlib.md5(name.encode("utf-8")).digest()
    hue = (h[0] / 255.0)
    sat = 0.45 + (h[1] / 255.0) * 0.30
    val = 0.65 + (h[2] / 255.0) * 0.25
    i = int(hue * 6)
    f = hue * 6 - i
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)
    table = [(val, t, p), (q, val, p), (p, val, t),
             (p, q, val), (t, p, val), (val, p, q)]
    return list(table[i % 6])


def _physics_names(g) -> dict[int, tuple[str, list[float], int]]:
    """``physics tag -> (name, color, dim)`` following the viewer's
    ``<kind>_<index>`` convention (ports collapse to ``port_<N>``)."""
    out: dict[int, tuple[str, list[float], int]] = {}
    key_count: dict[str, int] = {}
    for phys in getattr(g, "_physics", []):
        tag = g._physics_tags.get(id(phys))
        if tag is None:
            continue
        cls = type(phys).__name__
        key = "port" if cls in _PORT_CLASSES else cls.lower()
        key_count[key] = key_count.get(key, 0) + 1
        dim = 3 if cls == "PML" else 2
        out[tag] = (f"{key}_{key_count[key]}", _physics_color(phys), dim)
    return out


def _volume_names(g) -> dict[int, tuple[str, list[float], object]]:
    """``region tag -> (name, color, material)`` per material class index."""
    out: dict[int, tuple[str, list[float], object]] = {}
    cls_count: dict[str, int] = {}
    for region, mat in getattr(g, "_volume_materials", []):
        cls = type(mat).__name__.lower()
        cls_count[cls] = cls_count.get(cls, 0) + 1
        out[region] = (f"{cls}_{cls_count[cls]}", _material_color(mat), mat)
    return out


# ── Shared geometry helpers ───────────────────────────────────────────────────


def _bbox_payload(points: np.ndarray) -> dict[str, list[float]]:
    if len(points) == 0:
        return {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return {"min": [float(v) for v in lo], "max": [float(v) for v in hi]}


def _tri_soup(points: np.ndarray, tris: np.ndarray) -> tuple[list, list]:
    """Flat-shaded (positions, normals) for one triangle set."""
    if len(tris) == 0:
        return [], []
    p = points[tris]                       # (n, 3, 3)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    positions = p.reshape(-1).tolist()
    normals = np.repeat(n, 3, axis=0).reshape(-1).tolist()
    return positions, normals


def _tag_lookup(tris: np.ndarray, tri_tags: np.ndarray) -> dict:
    return {
        tuple(sorted(map(int, t))): int(tag)
        for t, tag in zip(tris, tri_tags)
    }


# ── Payload builders ──────────────────────────────────────────────────────────


def geometry_to_payload(g: Any, *, target_tris: int = 4000) -> dict:
    """Viewer payload for a Geometry: the FEM mesh when ``g.mesh()`` has
    run, a coarse non-mutating preview otherwise."""
    del target_tris  # the coarse preview maxh keeps its own budget
    if getattr(g, "_mesh_arrays", None) is not None:
        return mesh_to_payload(g, maxh=0.0)
    return _preview_payload(g)


def _preview_payload(g: Any) -> dict:
    """Coarse rapidmesh preview on a DEEP COPY of the geometry: the user's
    state stays untouched, missing materials are defaulted to Air for the
    render, and unresolved physics degrades to neutral surfaces instead of
    failing the save hook."""
    import copy

    g2 = copy.deepcopy(g)
    los = [np.asarray(e["bbox"][0], dtype=float) for e in g2._solids]
    his = [np.asarray(e["bbox"][1], dtype=float) for e in g2._solids]
    if not los:
        return {
            "kind": "geometry", "bbox": _bbox_payload(np.zeros((0, 3))),
            "entities": [],
            "stats": {"n_entities": 0, "n_triangles": 0, "maxh": 0.0},
        }
    lo = np.min(los, axis=0)
    hi = np.max(his, axis=0)
    diag = float(np.linalg.norm(hi - lo))
    maxh = max(diag / 10.0, 1e-9)

    from ..materials import Air
    for entry in g2._solids:
        if entry["sheet_tag"] == 0 and not entry["void"] \
                and entry["material"] is None:
            entry["material"] = Air()
        # preview meshes coarse; per-solid refinement would defeat it
        entry["maxh"] = None
        entry["surface_maxh"] = None
    g2._size_points = []

    try:
        g2.mesh(maxh=maxh)
    except Exception:
        # unresolved physics selection (or a half-built scene): drop the
        # physics and render plain geometry
        g2._physics = []
        try:
            g2.mesh(maxh=maxh)
        except Exception:
            return {
                "kind": "geometry", "bbox": _bbox_payload(np.zeros((0, 3))),
                "entities": [],
                "stats": {"n_entities": 0, "n_triangles": 0, "maxh": maxh},
            }

    m = g2._mesh
    points = m.points
    faces = m.faces.astype(np.int64)
    regions = m.face_regions.astype(np.int64)
    _, _, _, tag_tris, tag_vals = g2._mesh_arrays
    tag_of = _tag_lookup(tag_tris, tag_vals)
    face_tag = np.array(
        [tag_of.get(tuple(sorted(map(int, f))), 0) for f in faces],
        dtype=np.int64,
    )

    entities: list[dict] = []

    def _emit(name, tag, dim, color, idx, material):
        if len(idx) == 0:
            return
        pos, nor = _tri_soup(points, faces[idx])
        entities.append({
            "name": name, "tag": int(tag), "dim": int(dim), "color": color,
            "positions": pos, "normals": nor,
            "material": _material_label(material),
        })

    # Pass 1: physics-targeted faces (volume physics like PML colors its
    # volume's boundary below).
    taken = np.zeros(len(faces), dtype=bool)
    for tag, (name, color, dim) in _physics_names(g2).items():
        if dim != 2:
            continue
        idx = np.nonzero(face_tag == tag)[0]
        taken[idx] = True
        _emit(name, tag, 2, color, idx, None)

    # Pass 2: volumes claim their remaining boundary/interface faces.
    pml_regions = {
        g2._physics_tags[id(p)]
        for p in g2._physics if type(p).__name__ == "PML"
    }
    vol_names = _volume_names(g2)
    next_tag = 100000
    for region in sorted(
            set(vol_names) | pml_regions | set(np.unique(regions)) - {0}):
        in_region = ~taken & ((regions[:, 0] == region) | (regions[:, 1] == region))
        idx = np.nonzero(in_region)[0]
        taken[idx] = True
        if region in pml_regions:
            name, color, mat = "pml_1", _COL_PML, None
        elif region in vol_names:
            name, color, mat = vol_names[region]
        else:
            name, color, mat = f"_volume_{region}", _COL_NEUTRAL, None
        _emit(name, next_tag, 3, color, idx, mat)
        next_tag += 1

    # Pass 3: orphan surfaces (untagged void walls etc.), neutral.
    idx = np.nonzero(~taken)[0]
    _emit("_faces", next_tag, 2, _COL_NEUTRAL, idx, None)

    return {
        "kind": "geometry",
        "bbox": _bbox_payload(points),
        "entities": entities,
        "stats": {
            "n_entities": len(entities),
            "n_triangles": int(len(faces)),
            "maxh": maxh,
        },
    }


def mesh_to_payload(g: Any, *, maxh: float) -> dict:
    """FEM-mesh payload from the geometry's solver arrays: nodes, tets and
    tagged surface tris plus the mesher's feature edges. Face physics tags
    are offset above the volume-region tags so ``phys_names`` stays one
    flat namespace like the viewer expects."""
    del maxh  # meshing happened in g.mesh(); kept for call compatibility
    if getattr(g, "_mesh_arrays", None) is None:
        raise RuntimeError("geometry not meshed yet, call g.mesh() first")
    m = g._mesh
    points = m.points
    faces = m.faces.astype(np.int64)
    _, _, tet_tags, tag_tris, tag_vals = g._mesh_arrays

    face_offset = int(tet_tags.max()) if len(tet_tags) else 0
    tag_of = _tag_lookup(tag_tris, tag_vals)
    tri_phys = [
        (tag_of.get(tuple(sorted(map(int, f))), -face_offset) + face_offset)
        for f in faces
    ]

    phys_names: dict[int, str] = {}
    phys_dim: dict[int, int] = {}
    for tag, (name, _color, dim) in _physics_names(g).items():
        if dim == 2:
            phys_names[tag + face_offset] = name
            phys_dim[tag + face_offset] = 2
        else:
            phys_names[tag] = name
            phys_dim[tag] = 3
    for region, (name, _color, _mat) in _volume_names(g).items():
        phys_names.setdefault(region, name)
        phys_dim.setdefault(region, 3)

    stats = dict(m.stats)
    return {
        "kind": "mesh",
        "bbox": _bbox_payload(points),
        "nodes": points.reshape(-1).tolist(),
        "tris": faces.reshape(-1).tolist(),
        "tri_phys": tri_phys,
        "tets": m.tets.astype(np.int64).reshape(-1).tolist(),
        "tet_phys": tet_tags.astype(int).tolist(),
        "edges": m.edges.astype(np.int64).reshape(-1).tolist(),
        "phys_names": phys_names,
        "phys_dim": phys_dim,
        "name_to_tag": {v: k for k, v in phys_names.items()},
        "stats": {
            "n_nodes": int(len(points)),
            "n_tets": int(stats.get("n_tets", 0)),
            "n_tris": int(len(faces)),
            "n_edges": int(len(m.edges)),
            "min_dihedral_deg": float(stats.get("min_dihedral_deg", 0.0)),
            "mesh_time_s": float(stats.get("millis", 0)) / 1e3,
        },
    }
