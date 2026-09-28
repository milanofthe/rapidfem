# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""GDSII import for :class:`rapidfem.geometry.Geometry`, mixed into it.

Every polygon on a stack layer becomes a prism at the layer's height (or a
sheet with ``thin_conductors``). Polygons of one layer are fused into one
region, so touching or overlapping traces form one conductor with no
internal faces.
"""
from __future__ import annotations

import numpy as np


def _clean(pts: np.ndarray) -> np.ndarray:
    """Drop the closing duplicate and repeated vertices of a GDS polygon."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > 1e-12:
            keep.append(p)
    if len(keep) > 1 and np.linalg.norm(keep[-1] - keep[0]) <= 1e-12:
        keep = keep[:-1]
    if len(keep) < 3:
        raise ValueError(f"polygon collapsed to {len(keep)} unique vertices")
    return np.asarray(keep)


def _area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


class _GdsMixin:
    """GDSII layout import, mixed into :class:`rapidfem.geometry.Geometry`."""

    @staticmethod
    def from_gds(
        path: str,
        stack,                          # rapidfem.rfic.Stack
        top_cell: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        flatten: bool = True,
        merge: bool = True,
        thin_conductors: bool = False,
        scale: float = 1e-6,
    ) -> "Geometry":
        """Load a GDSII layout and extrude all matching polygons into 3D primitives.

        Each polygon on a (gds, datatype) tuple in `stack` becomes a prism at
        the layer's z with the layer's thickness. All polygons of one layer
        share the layer's name on the resulting GeoObject so they can be
        batch-selected and passed to `rf.PEC`/`rf.LumpedPort` together.

        Args:
            path: Path to the .gds(.gz) file.
            stack: A `rapidfem.rfic.Stack` mapping (gds, datatype) -> PdkLayer.
            top_cell: Cell name to extrude. ``None`` auto-picks the unique
                top-level cell.
            bbox: Optional (xmin, ymin, xmax, ymax) crop box in meters; polygons
                outside are skipped (faster iteration on large layouts).
            flatten: Resolve cell references before extrusion (default True).
                Set False only if your top cell has no references.
            merge: Fuse co-layer polygons into one conductor region (default
                True), so touching traces carry no internal faces.
            thin_conductors: If True, metal layers become 2D sheets at the
                layer's bottom z (thin-conductor approximation, t << w).
            scale: accepted for compatibility; the mesher works with exact
                predicates at any scale.
        """
        from .geometry import Geometry
        try:
            import gdstk
        except ImportError as e:
            raise ImportError("gdstk is required for from_gds(). pip install gdstk") from e

        lib = gdstk.read_gds(path)
        cells_by_name = {c.name: c for c in lib.cells}
        if top_cell is None:
            tops = lib.top_level()
            if len(tops) != 1:
                names = ", ".join(c.name for c in tops)
                raise ValueError(
                    f"GDS has {len(tops)} top-level cells ({names}); specify top_cell=...")
            cell = tops[0]
        else:
            if top_cell not in cells_by_name:
                avail = ", ".join(cells_by_name.keys())
                raise ValueError(f"top_cell {top_cell!r} not in GDS; available: {avail}")
            cell = cells_by_name[top_cell]

        unit = lib.unit  # meters per GDS unit
        polys = cell.get_polygons() if not flatten else cell.flatten().polygons
        g = Geometry(name=cell.name or "gds_import")

        per_layer: dict[str, list[np.ndarray]] = {}
        for poly in polys:
            pdk_layer = stack.by_gds(poly.layer, poly.datatype)
            if pdk_layer is None:
                continue
            pts = np.asarray(poly.points, dtype=np.float64) * unit
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox
                if (pts[:, 0].max() < xmin or pts[:, 0].min() > xmax
                        or pts[:, 1].max() < ymin or pts[:, 1].min() > ymax):
                    continue
            per_layer.setdefault(pdk_layer.name, []).append(pts)

        for layer_name, layer_polys in per_layer.items():
            pdk = stack.by_name(layer_name)
            sheet = thin_conductors and pdk.type == "metal"
            objs = []
            for pts in layer_polys:
                if sheet:
                    obj = g._plate_polygon(pts, z=pdk.z)
                else:
                    obj = g._extrude_polygon(pts, z=pdk.z, thickness=pdk.thickness)
                obj.name = layer_name
                objs.append(obj)
            if merge and not sheet and len(objs) >= 2:
                g.fuse(objs[0], *objs[1:])
        return g

    def _plate_polygon(self, pts: "np.ndarray", z: float):
        """A sheet from a closed xy polygon at height z."""
        return self.polygon([tuple(p[:2]) for p in _clean(pts)], (0.0, 0.0, z))

    def _extrude_polygon(self, pts: "np.ndarray", z: float, thickness: float):
        """A prism from a closed xy polygon, from z up by thickness."""
        pts = _clean(pts)
        obj = self.polygon([tuple(p[:2]) for p in pts], (0.0, 0.0, z))
        self.extrude(obj, thickness)
        if not hasattr(self, "_prisms"):
            self._prisms = {}
        self._prisms[obj._id] = _area(pts) * float(thickness)
        return obj
