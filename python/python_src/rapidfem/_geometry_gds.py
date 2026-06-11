"""GDSII import for rapidfem geometries.

The gmsh-backed ``Geometry.from_gds(path, ...)`` static method has been
replaced by the module-level function ``from_gds(g, path, ...)``. The
caller creates the Geometry first and passes it in; the loading machinery
is decoupled from the geometry backend.

New calling form::

    import rapidfem as rf
    import rapidfem.rfic as rfic

    g = rf.Geometry(maxh=20e-6)
    rfic.from_gds(g, "inductor.gds", stack=stack, top_cell="ind_3turn")

The retired ``Geometry.from_gds(path, ...)`` static form is gone; create
the Geometry first and pass it in.
"""
from __future__ import annotations

import numpy as np


# HELPERS ===============================================================================


def _clean_polygon(pts_m: np.ndarray) -> np.ndarray:
    """Drop the closing duplicate vertex and near-coincident consecutive
    vertices from a polygon given in metres.

    Returns a float64 array of shape (N, 2). The deduplication tolerance
    is 1 nm, tight enough to preserve nm-precision GDS geometry while
    reliably eliminating Boolean-union slivers that appear at polygon joins.
    """
    if np.allclose(pts_m[0], pts_m[-1]):
        pts_m = pts_m[:-1]
    tol = 1e-9
    keep = [pts_m[0]]
    for p in pts_m[1:]:
        if np.linalg.norm(p - keep[-1]) > tol:
            keep.append(p)
    if len(keep) > 1 and np.linalg.norm(keep[-1] - keep[0]) <= tol:
        keep = keep[:-1]
    return np.asarray(keep, dtype=np.float64)


def _material_for_layer(pdk_layer):
    """Derive a rapidfem bulk material from a PdkLayer.

    PEC is typically applied to all conductor faces by the caller, so
    the bulk material only needs to satisfy the geometry constraint that
    every non-PML solid carries a material. The assignment mirrors the
    layer's dominant physics:

    - metal/via with sigma > 0: Conductor(conductivity=sigma)
    - poly/oxide/other with er > 1: Dielectric(er=er, tand=tand)
    - everything else: Air() (lossless vacuum placeholder)
    """
    from rapidfem.materials import Conductor, Dielectric, Air

    if pdk_layer.type in ("metal", "via"):
        if pdk_layer.sigma > 0:
            return Conductor(conductivity=pdk_layer.sigma)
    if pdk_layer.er > 1.0:
        return Dielectric(er=pdk_layer.er, tand=pdk_layer.tand)
    return Air()


# PUBLIC API ============================================================================


def from_gds(
    g,
    path: str,
    stack,
    top_cell: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    flatten: bool = True,
    merge: bool = True,
    thin_conductors: bool = False,
) -> None:
    """Load a GDSII layout and extrude all matching polygons into ``g``.

    Each polygon whose (gds, datatype) tuple is present in ``stack`` is
    either extruded as a 3-D prism at the layer's z/thickness (default),
    or placed as a zero-thickness polygon sheet when
    ``thin_conductors=True`` and the layer type is "metal". Solid
    extrusions receive a bulk material derived from the PdkLayer (Conductor
    for metals/vias, Dielectric for poly/oxide); callers apply PEC or other
    surface BCs to the resulting faces.

    Parameters
    ----------
    g : rapidfem.Geometry
        Target geometry. Set a global maxh at construction time or pass
        maxh to g.mesh() later.
    path : str
        Path to the .gds (.gz) file.
    stack : rapidfem.rfic.Stack
        Maps (gds, datatype) to PdkLayer; carries z, thickness, material
        data.
    top_cell : str, optional
        Cell name to flatten and extrude. None auto-picks the unique
        top-level cell.
    bbox : tuple[float, float, float, float], optional
        (xmin, ymin, xmax, ymax) crop box in metres; polygons wholly
        outside are skipped.
    flatten : bool
        Flatten cell references before extrusion (default True).
    merge : bool
        Accepted for API compatibility; fragment() is a no-op in the
        rapidmesh backend, so co-layer polygons are always arranged
        conformally without explicit merging.
    thin_conductors : bool
        If True, metal layers become zero-thickness polygon sheets at the
        layer bottom z (thin-conductor approximation, t << wavelength/100).
        Vias and poly are still extruded as solids.
    """
    try:
        import gdstk
    except ImportError as exc:
        raise ImportError(
            "gdstk is required for from_gds(); pip install gdstk"
        ) from exc

    lib = gdstk.read_gds(path)
    cells_by_name = {c.name: c for c in lib.cells}

    if top_cell is None:
        tops = lib.top_level()
        if len(tops) != 1:
            names = ", ".join(c.name for c in tops)
            raise ValueError(
                f"GDS has {len(tops)} top-level cells ({names}); "
                f"specify top_cell=..."
            )
        cell = tops[0]
    else:
        if top_cell not in cells_by_name:
            avail = ", ".join(cells_by_name.keys())
            raise ValueError(
                f"top_cell {top_cell!r} not in GDS; available: {avail}"
            )
        cell = cells_by_name[top_cell]

    # lib.unit is metres per logical GDS unit (typically 1e-6 for µm files)
    unit = lib.unit

    flat_polys = cell.get_polygons() if not flatten else cell.flatten().polygons

    per_layer: dict[str, list[np.ndarray]] = {}
    for poly in flat_polys:
        pdk_layer = stack.by_gds(poly.layer, poly.datatype)
        if pdk_layer is None:
            continue
        pts_raw = np.asarray(poly.points, dtype=np.float64)
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            if (pts_raw[:, 0].max() < xmin or pts_raw[:, 0].min() > xmax
                    or pts_raw[:, 1].max() < ymin or pts_raw[:, 1].min() > ymax):
                continue
        per_layer.setdefault(pdk_layer.name, []).append(pts_raw * unit)

    per_layer_objs: dict[str, list] = {}
    for layer_name, polys in per_layer.items():
        pdk = stack.by_name(layer_name)
        use_2d = thin_conductors and pdk.type == "metal"
        objs = []
        for pts in polys:
            cleaned = _clean_polygon(pts)
            if len(cleaned) < 3:
                continue
            pts_2d = [(float(p[0]), float(p[1])) for p in cleaned]
            if use_2d:
                obj = g.polygon_plate(pts_2d, position=(0.0, 0.0, pdk.z))
            else:
                mat = _material_for_layer(pdk)
                obj = g.prism(
                    pts_2d, pdk.thickness,
                    position=(0.0, 0.0, pdk.z),
                    material=mat,
                )
            obj.name = layer_name
            objs.append(obj)
        if objs:
            per_layer_objs[layer_name] = objs

    # fragment() is a no-op in the rapidmesh backend; the call is retained
    # so call sites that set merge=True keep the same observable behaviour.
    if merge:
        for objs in per_layer_objs.values():
            if len(objs) >= 2:
                g.fragment(objs[0], *objs[1:])


