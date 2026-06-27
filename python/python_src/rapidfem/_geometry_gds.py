"""GDSII import for :class:`rapidfem.geometry.Geometry`.

Split out of ``geometry.py`` as a mixin: the GDS-to-3D extrusion machinery is
a self-contained subsystem (it only touches the scale helper, the wrap
helpers, and gmsh), so it lives here to keep the core ``Geometry`` file
navigable. ``Geometry`` inherits :class:`_GdsMixin`, so every method below is
a normal ``Geometry`` method at runtime.
"""
from __future__ import annotations

import gmsh


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

        Each polygon on a (gds, datatype) tuple in `stack` becomes a 3D box
        (rectangles) or extruded prism (general polygons) at the layer's z with
        the layer's thickness. All polygons of one layer share the layer's
        name on the resulting GeoObject so they can be batch-selected and
        passed to `rf.PEC`/`rf.LumpedPort` together.

        Args:
            path: Path to the .gds(.gz) file.
            stack: A `rapidfem.rfic.Stack` mapping (gds, datatype) -> PdkLayer.
            top_cell: Cell name to extrude. ``None`` auto-picks the unique
                top-level cell.
            bbox: Optional (xmin, ymin, xmax, ymax) crop box in meters; polygons
                outside are skipped (faster iteration on large layouts).
            flatten: Resolve cell references before extrusion (default True).
                Set False only if your top cell has no references.
            merge: Merge co-layer polygons via gmsh fragment so adjacent traces
                produce a conformal mesh (default True).
            thin_conductors: If True, metal layers become 2D PEC plates at the
                layer's bottom z (thin-conductor approximation, t << w). The
                resulting plate carries the layer's name as a SURFACE physical
                group, which is what the simulator's `.pec(...)` BC expects.
                Recommended for RFIC-style metal with thicknesses <= wavelength/100.
        """
        from .geometry import Geometry

        try:
            import gdstk
        except ImportError as e:
            raise ImportError("gdstk is required for from_gds(). pip install gdstk") from e
        import numpy as np

        lib = gdstk.read_gds(path)
        cells_by_name = {c.name: c for c in lib.cells}

        # Resolve top cell
        if top_cell is None:
            tops = lib.top_level()
            if len(tops) != 1:
                names = ", ".join(c.name for c in tops)
                raise ValueError(
                    f"GDS has {len(tops)} top-level cells ({names}); "
                    f"specify top_cell=...")
            cell = tops[0]
        else:
            if top_cell not in cells_by_name:
                avail = ", ".join(cells_by_name.keys())
                raise ValueError(f"top_cell {top_cell!r} not in GDS; available: {avail}")
            cell = cells_by_name[top_cell]

        # GDS unit (typically 1e-6 = micron, sometimes 1e-9 = nm). gdstk reports
        # the working unit in lib.unit (meters per GDS unit).
        unit = lib.unit  # meters per logical unit in the GDS

        # Flatten the cell to expose all polygons including those in references.
        flat_polys = cell.get_polygons() if not flatten else cell.flatten().polygons

        # Optional bbox crop
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox

        # scale=1e-6 normalises µm-scale GDS features so gmsh's OCC kernel
        # applies its relative tolerances at the feature scale; without it,
        # tight RFIC structures trip OCC "segment/facet intersect" errors and
        # mesh generation crawls (cf. from_fem_json).
        g = Geometry(name=cell.name or "gds_import", scale=scale)
        # Group polygons by their PdkLayer (so we name + extrude per-layer).
        per_layer: dict[str, list[np.ndarray]] = {}
        for poly in flat_polys:
            pdk_layer = stack.by_gds(poly.layer, poly.datatype)
            if pdk_layer is None:
                continue
            pts_raw = np.asarray(poly.points, dtype=np.float64)
            if bbox is not None:
                if (pts_raw[:, 0].max() < xmin or pts_raw[:, 0].min() > xmax
                        or pts_raw[:, 1].max() < ymin or pts_raw[:, 1].min() > ymax):
                    continue
            pts_m = pts_raw * unit  # convert GDS coords to meters
            per_layer.setdefault(pdk_layer.name, []).append(pts_m)

        # Build each polygon at its layer's z. With thin_conductors=True (or
        # for non-metal types) we keep it as a 2D plate so PEC BC can attach
        # to a SURFACE group; otherwise we extrude to a 3D solid.
        per_layer_objs: dict[str, list[GeoObject]] = {}
        for layer_name, polys in per_layer.items():
            pdk = stack.by_name(layer_name)
            use_2d = thin_conductors and pdk.type == "metal"
            objs: list[GeoObject] = []
            for pts in polys:
                if use_2d:
                    obj = g._plate_polygon(pts, z=pdk.z)
                else:
                    obj = g._extrude_polygon(pts, z=pdk.z, thickness=pdk.thickness)
                obj.name = layer_name
                objs.append(obj)
            per_layer_objs[layer_name] = objs

        # Optional: fragment co-layer polygons so adjacent traces share faces.
        if merge:
            for objs in per_layer_objs.values():
                if len(objs) >= 2:
                    g.fragment(objs[0], *objs[1:])

        return g

    def _plate_polygon(self, pts: "np.ndarray", z: float) -> GeoObject:
        """Create a 2D plane surface from a closed polygon at height z.

        Used by `from_gds(thin_conductors=True)` so metals stay 2D and their
        layer-name physical group lands on FACES (PEC-compatible).
        """
        import numpy as np
        s = self._s
        # Rectangle fast path
        if pts.shape[0] in (4, 5):
            p = pts[:4]
            xs, ys = sorted(set(p[:, 0])), sorted(set(p[:, 1]))
            if len(xs) == 2 and len(ys) == 2:
                tag = gmsh.model.occ.addRectangle(
                    s(xs[0]), s(ys[0]), s(z), s(xs[1] - xs[0]), s(ys[1] - ys[0])
                )
                return self._wrap_face(tag)
        # General polygon
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        tol = 1e-9
        keep = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - keep[-1]) > tol:
                keep.append(p)
        if len(keep) > 1 and np.linalg.norm(keep[-1] - keep[0]) <= tol:
            keep = keep[:-1]
        pts = np.asarray(keep)
        if len(pts) < 3:
            raise ValueError(f"Polygon collapsed to {len(pts)} unique vertices")
        pt_tags = [gmsh.model.occ.addPoint(s(p[0]), s(p[1]), s(z)) for p in pts]
        line_tags = [
            gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % len(pt_tags)])
            for i in range(len(pt_tags))
        ]
        loop = gmsh.model.occ.addCurveLoop(line_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        return self._wrap_face(surf)

    def _extrude_polygon(
        self,
        pts: "np.ndarray",
        z: float,
        thickness: float,
    ) -> GeoObject:
        """Extrude a 2D polygon (Nx2 numpy array) vertically into a 3D solid.

        Rectangular axis-aligned polygons become addBox (fast path).
        Otherwise: build a wire, plane surface, then extrude.
        """
        import numpy as np

        s = self._s
        # Detect axis-aligned rectangle (4 points, 90 degree corners)
        if pts.shape[0] in (4, 5):
            p = pts[:4]
            xs, ys = sorted(set(p[:, 0])), sorted(set(p[:, 1]))
            if len(xs) == 2 and len(ys) == 2:
                tag = gmsh.model.occ.addBox(
                    s(xs[0]), s(ys[0]), s(z),
                    s(xs[1] - xs[0]), s(ys[1] - ys[0]), s(thickness),
                )
                return self._wrap_volume(tag)

        # General polygon: wire, plane, extrude
        # Drop duplicated last point (gdstk includes it sometimes)
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        # Drop adjacent coincident vertices, gdstk boolean unions can leave
        # them at shared corners and they'd collapse OCC line endpoints into a
        # broken loop. Keep one of every coincident-pair (within ~1nm).
        tol = 1e-9
        keep = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - keep[-1]) > tol:
                keep.append(p)
        # Also check wraparound (last vs first)
        if len(keep) > 1 and np.linalg.norm(keep[-1] - keep[0]) <= tol:
            keep = keep[:-1]
        pts = np.asarray(keep)
        if len(pts) < 3:
            raise ValueError(f"Polygon collapsed to {len(pts)} unique vertices")
        pt_tags = [gmsh.model.occ.addPoint(s(p[0]), s(p[1]), s(z)) for p in pts]
        line_tags = [
            gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % len(pt_tags)])
            for i in range(len(pt_tags))
        ]
        loop = gmsh.model.occ.addCurveLoop(line_tags)
        surf = gmsh.model.occ.addPlaneSurface([loop])
        # Extrude vertically by thickness; second elt of return is the top cap
        out = gmsh.model.occ.extrude([(2, surf)], 0, 0, s(thickness))
        # gmsh's extrude returns: [top_face, volume, side_faces...]
        vol_tag = next(t for d, t in out if d == 3)
        return self._wrap_volume(vol_tag)
