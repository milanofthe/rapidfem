"""Declarative geometry layer on the rapidmesh kernel.

The public surface mirrors the original gmsh-backed module (``Geometry``
primitives, ``obj.faces`` selection, physics objects binding to faces or
volumes), but the implementation is fully declarative: primitive calls only
record descriptions, face selections are lazy queries, and :meth:`Geometry.mesh`
builds the rapidmesh scene, meshes it, resolves every selection against the
final mesh faces and assigns the solver tags.

Key semantic differences from the gmsh backend (intentional):

- ``fragment()`` is a no-op: rapidmesh arranges every operand exactly and
  conformally by construction, there is nothing left to make conforming.
- ``cut(target, *tools)`` marks the tool solids as voids (their volume is
  carved out of everything added before them, walls stay selectable).
- Face selection (``.min/.max/.where/.outer/.hull/.unassigned``) evaluates at
  ``mesh()`` time against facet groups of the final mesh. A facet group is the
  set of mesh faces sharing (sheet tag, analytic surface, region pair), which
  reproduces the granularity gmsh's ``fragment`` produced. ``where`` predicates
  receive the same ``(cog, bbox)`` tuples as before.
- Every non-PML volume must carry a material; the solver zero-fills tensors of
  unlisted volumes once any material section exists, which panics deep in
  assembly. Failing early here is the holistic fix.

All coordinates are metres.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

import rapidmesh as _rm


# SELECTION =============================================================================


class _Entity:
    """One lazy selection node: a base face/volume set plus a filter chain.

    ``base`` is ``("solid", index)``, ``("sheet", internal_tag)`` or
    ``("volume", index)``; ``ops`` is a tuple of filter steps applied to the
    base set at resolve time. Physics objects store these and the geometry
    resolves them against the mesh facet groups in :meth:`Geometry.mesh`.
    """

    __slots__ = ("_geometry", "dim", "base", "ops")

    def __init__(self, geometry: "Geometry", dim: int, base, ops: tuple = ()):
        self._geometry = geometry
        self.dim = dim
        self.base = base
        self.ops = ops

    def _with(self, op) -> "_Entity":
        return _Entity(self._geometry, self.dim, self.base, self.ops + (op,))


class EntityCollection:
    """A chainable, lazily evaluated set of faces.

    Filters return new collections; nothing touches the mesh until
    :meth:`Geometry.mesh` resolves the chain. Iteration yields the underlying
    selection nodes so the variadic physics constructors keep working
    (``rf.PEC(*obj.faces.where(...))``).
    """

    def __init__(self, geometry: "Geometry", entities: list[_Entity]):
        self._geometry = geometry
        self._entities = entities

    def __iter__(self):
        return iter(self._entities)

    def __len__(self) -> int:
        return len(self._entities)

    def _map(self, op) -> "EntityCollection":
        return EntityCollection(
            self._geometry, [e._with(op) for e in self._entities]
        )

    def min(self, axis: str = "z") -> "EntityCollection":
        """faces whose centroid sits at the set's minimum along ``axis``
        (within mesh tolerance; a planar end face resolves as one group)"""
        return self._map(("extreme", axis, -1))

    def max(self, axis: str = "z") -> "EntityCollection":
        """faces whose centroid sits at the set's maximum along ``axis``"""
        return self._map(("extreme", axis, +1))

    def where(self, predicate: Callable) -> "EntityCollection":
        """faces passing ``predicate(cog, bbox)`` with ``cog = (x, y, z)``
        and ``bbox = (xmin, ymin, zmin, xmax, ymax, zmax)`` of the group"""
        return self._map(("where", predicate))

    @property
    def outer(self) -> "EntityCollection":
        """faces that lie flat on the global model bounding box"""
        return self._map(("outer",))

    @property
    def hull(self) -> "EntityCollection":
        """faces that lie flat on this selection's own bounding box"""
        return self._map(("hull",))

    @property
    def unassigned(self) -> "EntityCollection":
        """faces not claimed by any physics object registered earlier
        (resolution follows physics registration order)"""
        return self._map(("unassigned",))


FaceCollection = EntityCollection
EdgeCollection = EntityCollection


# GEO OBJECTS ===========================================================================


class GeoObject:
    """Handle to one solid, void or sheet added to a :class:`Geometry`."""

    def __init__(self, geometry: "Geometry", dim: int, index: int):
        self._geometry = geometry
        self.dim = dim
        self._index = index

    @property
    def _entity(self) -> _Entity:
        """selection node for variadic physics targets: a sheet object selects
        its faces, a material solid selects its volume"""
        entry = self._geometry._solids[self._index]
        if self.dim == 2:
            return _Entity(self._geometry, 2, ("sheet", entry["sheet_tag"]))
        if entry["void"]:
            raise ValueError(
                "a void cannot be a volume physics target (select its walls "
                "via .faces instead)")
        return _Entity(self._geometry, 3, ("volume", self._index))

    @property
    def faces(self) -> EntityCollection:
        """lazy selection over this object's mesh faces"""
        entry = self._geometry._solids[self._index]
        if self.dim == 2:
            base = ("sheet", entry["sheet_tag"])
        else:
            base = ("solid", self._index)
        return EntityCollection(
            self._geometry, [_Entity(self._geometry, 2, base)]
        )

    @property
    def material(self):
        return self._geometry._solids[self._index].get("material")

    @material.setter
    def material(self, value) -> None:
        if isinstance(value, str):
            raise ValueError(
                f"string materials are retired: assign a material instance "
                f"(rf.Air(), rf.Dielectric(er=...)) instead of {value!r}")
        if self.dim != 3:
            raise ValueError("only solids carry a material")
        self._geometry._solids[self._index]["material"] = value

    @property
    def maxh(self):
        return self._geometry._solids[self._index].get("maxh")

    @maxh.setter
    def maxh(self, value: float) -> None:
        self._geometry._solids[self._index]["maxh"] = float(value)

    @property
    def bbox(self) -> tuple:
        lo, hi = self._geometry._solids[self._index]["bbox"]
        return (*lo, *hi)


class _FaceGroup:
    """A resolved facet group of the final mesh (selection granularity)."""

    __slots__ = ("tri_idx", "cog", "bbox", "sheet_tag", "owner", "regions", "claimed")

    def __init__(self, tri_idx, cog, bbox, sheet_tag, owner, regions):
        self.tri_idx = tri_idx          # np.ndarray of face indices
        self.cog = cog                  # (x, y, z) area-weighted centroid
        self.bbox = bbox                # (xmin, ymin, zmin, xmax, ymax, zmax)
        self.sheet_tag = sheet_tag      # internal sheet tag, 0 = none
        self.owner = owner              # owning solid index (insertion order)
        self.regions = regions          # (lo, hi) region pair
        self.claimed = False            # set when a physics object takes it


_TOL_REL = 1e-9


# GEOMETRY ==============================================================================


class Geometry:
    """Declarative geometry builder on the rapidmesh kernel.

    Primitive calls record descriptions; :meth:`mesh` assembles the exact CSG
    scene, meshes it, resolves the registered physics selections and stores
    the solver-ready arrays. ``rapidfem.Problem`` consumes those directly,
    no mesh files and no gmsh.

    Parameters
    ----------
    maxh : float
        global target tet edge length in metres
    grading : float
        Lipschitz size-grading constant (0.5 grows neighbors ~1.5x)
    """

    def __init__(self, *, maxh: float | None = None, scale: float = 1.0,
                 grading: float = 0.5, name: str = "rapidfem"):
        if scale != 1.0:
            raise ValueError("scale is retired: model in metres directly")
        self.maxh = None if maxh is None else float(maxh)
        self._grading = float(grading)
        self._name = name
        # solid AND sheet descriptions, in insertion order
        self._solids: list[dict] = []
        self._n_solid3 = 0          # rapidmesh solid counter (voids included)
        self._next_sheet_tag = 1
        self._size_points: list[tuple[tuple[float, float, float], float]] = []
        self._physics: list = []
        # filled by mesh():
        self._mesh = None
        self._groups: list[_FaceGroup] | None = None
        self._mesh_arrays = None    # (nodes, tets, tet_tags, tris, tri_tags)
        self._volume_materials: list[tuple[int, object]] = []
        self._physics_tags: dict[int, int] = {}
        self._region_of_solid: dict[int, int] = {}

    # ------------------------------------------------------------ recording helpers

    def _add_solid(self, kind: str, args: dict, *, material, maxh, bbox) -> GeoObject:
        if material is not None and getattr(material, "maxh", None) is not None:
            maxh = material.maxh if maxh is None else min(maxh, material.maxh)
        self._solids.append({
            "kind": kind, "args": args, "void": False, "material": material,
            "maxh": maxh, "surface_maxh": None, "sheet_tag": 0, "bbox": bbox,
            "solid3": self._n_solid3,
        })
        self._n_solid3 += 1
        return GeoObject(self, 3, len(self._solids) - 1)

    def _add_sheet(self, kind: str, args: dict, *, maxh, bbox) -> GeoObject:
        tag = self._next_sheet_tag
        self._next_sheet_tag += 1
        self._solids.append({
            "kind": kind, "args": args, "void": False, "material": None,
            "maxh": maxh, "surface_maxh": None, "sheet_tag": tag, "bbox": bbox,
            "solid3": None,
        })
        return GeoObject(self, 2, len(self._solids) - 1)

    # ------------------------------------------------------------ solid primitives

    def box(self, width: float, depth: float, height: float,
            position=(0, 0, 0), *, material=None, maxh: float | None = None) -> GeoObject:
        """axis-aligned box; ``position`` is the lower corner"""
        x, y, z = position
        return self._add_solid(
            "box", {"size": (width, depth, height), "position": (x, y, z)},
            material=material, maxh=maxh,
            bbox=(np.array([x, y, z]), np.array([x + width, y + depth, z + height])),
        )

    def cylinder(self, radius: float, height: float, position=(0, 0, 0),
                 axis=(0, 0, 1), *, segments: int = 24,
                 material=None, maxh: float | None = None) -> GeoObject:
        """cylinder from ``position`` along ``axis`` (length = ``height``)"""
        p = np.asarray(position, dtype=float)
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        ends = np.stack([p, p + height * a])
        pad = radius * np.sqrt(np.maximum(0.0, 1.0 - a * a))
        return self._add_solid(
            "cylinder",
            {"radius": radius, "height": height, "position": tuple(p),
             "axis": tuple(a), "segments": segments},
            material=material, maxh=maxh,
            bbox=(ends.min(axis=0) - pad, ends.max(axis=0) + pad),
        )

    def sphere(self, radius: float, position=(0, 0, 0), *, segments: int = 24,
               rings: int = 12, material=None, maxh: float | None = None,
               center=None) -> GeoObject:
        """sphere centred at ``position`` (alias ``center``)"""
        c = np.asarray(center if center is not None else position, dtype=float)
        return self._add_solid(
            "sphere", {"radius": radius, "center": tuple(c),
                       "segments": segments, "rings": rings},
            material=material, maxh=maxh, bbox=(c - radius, c + radius),
        )

    def cone(self, r1: float, r2: float, height: float, position=(0, 0, 0),
             axis=(0, 0, 1), *, segments: int = 24,
             material=None, maxh: float | None = None) -> GeoObject:
        """truncated cone: radius ``r1`` at ``position``, ``r2`` at the top"""
        p = np.asarray(position, dtype=float)
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        ends = np.stack([p, p + height * a])
        pad = max(r1, r2) * np.sqrt(np.maximum(0.0, 1.0 - a * a))
        return self._add_solid(
            "cone", {"r1": r1, "r2": r2, "height": height,
                     "position": tuple(p), "axis": tuple(a), "segments": segments},
            material=material, maxh=maxh,
            bbox=(ends.min(axis=0) - pad, ends.max(axis=0) + pad),
        )

    def wedge(self, dx: float, dy: float, dz: float, top_x: float = 0.0,
              position=(0, 0, 0), *, material=None,
              maxh: float | None = None) -> GeoObject:
        """right-angular wedge: box footprint ``dx, dy``, height ``dz``,
        top edge pulled to ``top_x``"""
        x, y, z = position
        return self._add_solid(
            "wedge", {"dx": dx, "dy": dy, "dz": dz, "top_x": top_x,
                      "position": (x, y, z)},
            material=material, maxh=maxh,
            bbox=(np.array([x, y, z]),
                  np.array([x + max(dx, top_x), y + dy, z + dz])),
        )

    def torus(self, major_radius: float, minor_radius: float, position=(0, 0, 0),
              axis=(0, 0, 1), *, segments: int = 32, tube_segments: int = 16,
              material=None, maxh: float | None = None, center=None) -> GeoObject:
        """torus around ``axis`` through ``position`` (alias ``center``)"""
        c = np.asarray(center if center is not None else position, dtype=float)
        r = major_radius + minor_radius
        return self._add_solid(
            "torus", {"major_radius": major_radius, "minor_radius": minor_radius,
                      "center": tuple(c), "axis": tuple(axis),
                      "segments": segments, "tube_segments": tube_segments},
            material=material, maxh=maxh, bbox=(c - r, c + r),
        )

    def loft(self, profile_a, profile_b, *, material=None,
             maxh: float | None = None) -> GeoObject:
        """ruled loft between two equal-count 3D point profiles
        (``polygon()`` results or raw point lists)"""
        pa = _profile_points(profile_a)
        pb = _profile_points(profile_b)
        allp = np.asarray(pa + pb, dtype=float)
        return self._add_solid(
            "loft", {"profile_a": pa, "profile_b": pb},
            material=material, maxh=maxh,
            bbox=(allp.min(axis=0), allp.max(axis=0)),
        )

    def extrude(self, profile, height: float, axis=(0, 0, 1), *,
                material=None, maxh: float | None = None) -> GeoObject:
        """extrude a planar 3D point profile along ``axis``"""
        pa = _profile_points(profile)
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        pb = [tuple(np.asarray(p) + height * a) for p in pa]
        return self.loft(pa, pb, material=material, maxh=maxh)

    def polygon(self, points, position=(0, 0, 0), *, holes=None,
                maxh: float | None = None) -> list:
        """planar polygon PROFILE for :meth:`loft` / :meth:`extrude`
        (for a meshed zero-thickness sheet use :meth:`polygon_plate`)"""
        del maxh
        if holes:
            raise ValueError(
                "polygon profiles cannot carry holes; use polygon_plate for "
                "sheets with holes")
        off = np.asarray(position, dtype=float)
        return [tuple(np.asarray(p, dtype=float) + off) for p in points]

    # ------------------------------------------------------------ sheet primitives

    def xy_plate(self, width: float, height: float, position=(0, 0, 0), *,
                 maxh: float | None = None) -> GeoObject:
        """zero-thickness rectangle in an xy-plane (PEC traces, ports)"""
        x, y, z = position
        return self._add_sheet(
            "plate", {"p0": (x, y, z), "du": (width, 0, 0), "dv": (0, height, 0)},
            maxh=maxh,
            bbox=(np.array([x, y, z]), np.array([x + width, y + height, z])),
        )

    def xz_plate(self, width: float, height: float, position=(0, 0, 0), *,
                 maxh: float | None = None) -> GeoObject:
        x, y, z = position
        return self._add_sheet(
            "plate", {"p0": (x, y, z), "du": (width, 0, 0), "dv": (0, 0, height)},
            maxh=maxh,
            bbox=(np.array([x, y, z]), np.array([x + width, y, z + height])),
        )

    def yz_plate(self, width: float, height: float, position=(0, 0, 0), *,
                 maxh: float | None = None) -> GeoObject:
        x, y, z = position
        return self._add_sheet(
            "plate", {"p0": (x, y, z), "du": (0, width, 0), "dv": (0, 0, height)},
            maxh=maxh,
            bbox=(np.array([x, y, z]), np.array([x, y + width, z + height])),
        )

    def plate(self, p0, width, height, *, maxh: float | None = None) -> GeoObject:
        """arbitrary-orientation rectangle spanned by edge vectors ``width``
        and ``height`` from corner ``p0``"""
        p = np.asarray(p0, dtype=float)
        du = np.asarray(width, dtype=float)
        dv = np.asarray(height, dtype=float)
        corners = np.stack([p, p + du, p + dv, p + du + dv])
        return self._add_sheet(
            "plate", {"p0": tuple(p), "du": tuple(du), "dv": tuple(dv)},
            maxh=maxh, bbox=(corners.min(axis=0), corners.max(axis=0)),
        )

    def disc(self, radius: float, position=(0, 0, 0), *, axis=(0, 0, 1),
             segments: int = 24, maxh: float | None = None) -> GeoObject:
        """zero-thickness disc sheet"""
        c = np.asarray(position, dtype=float)
        return self._add_sheet(
            "disc", {"radius": radius, "position": tuple(c),
                     "axis": tuple(axis), "segments": segments},
            maxh=maxh, bbox=(c - radius, c + radius),
        )

    def polygon_plate(self, points, position=(0, 0, 0), *, holes=None,
                      maxh: float | None = None) -> GeoObject:
        """zero-thickness polygon sheet in an xy-plane (2D points + holes)"""
        off = np.asarray(position, dtype=float)
        p2 = [(float(p[0]), float(p[1])) for p in points]
        allp = np.asarray(p2, dtype=float)
        lo = np.array([allp[:, 0].min(), allp[:, 1].min(), 0.0]) + off
        hi = np.array([allp[:, 0].max(), allp[:, 1].max(), 0.0]) + off
        return self._add_sheet(
            "polygon_plate",
            {"points": p2, "position": tuple(off),
             "holes": [[(float(x), float(y)) for x, y in h] for h in holes or []]},
            maxh=maxh, bbox=(lo, hi),
        )

    # ------------------------------------------------------------ booleans

    def fragment(self, *objects) -> None:
        """no-op: rapidmesh arranges all operands exactly and conformally by
        construction; shared faces and embedded sheets always conform"""

    def cut(self, target, *tools) -> None:
        """carve the tool solids out of ``target`` (and everything added
        before them): the tools become voids, their walls stay selectable"""
        del target  # priority is positional: a void carves everything earlier
        for t in tools:
            if not isinstance(t, GeoObject) or t.dim != 3:
                raise ValueError("cut tools must be solids")
            entry = self._geometry_entry(t)
            entry["void"] = True
            entry["material"] = None

    def _geometry_entry(self, obj: GeoObject) -> dict:
        if obj._geometry is not self:
            raise ValueError("object belongs to another Geometry")
        return self._solids[obj._index]

    # ------------------------------------------------------------ sizing

    def auto_refine_features(self, base_maxh: float | None = None,
                             resolution: float = 1.5,
                             min_maxh: float | None = None) -> None:
        """thin-volume sizing: every solid whose smallest bbox extent is below
        ``base_maxh`` gets ``region maxh = extent / resolution`` (clamped to
        ``min_maxh``). The validated microstrip recipe is thickness / 1.5 on
        the dielectric plus per-plate ``maxh`` on the conductor sheets."""
        base = self.maxh if base_maxh is None else float(base_maxh)
        if base is None:
            raise ValueError(
                "auto_refine_features needs a base size: pass base_maxh or "
                "construct Geometry(maxh=...)")
        for entry in self._solids:
            if entry["sheet_tag"] != 0 or entry["void"]:
                continue
            lo, hi = entry["bbox"]
            thickness = float(np.min(np.asarray(hi) - np.asarray(lo)))
            if thickness <= 0.0 or thickness >= base:
                continue
            h = thickness / resolution
            if min_maxh is not None:
                h = max(h, float(min_maxh))
            entry["maxh"] = h if entry["maxh"] is None else min(entry["maxh"], h)

    def refine_near_points(self, points, h: float, distance=None) -> None:
        """point size sources: the target shrinks to ``h`` at each point and
        recovers along the Lipschitz grading (adaptive-refinement hook)"""
        del distance  # the Lipschitz field needs no explicit radius
        for pt in points:
            self._size_points.append((tuple(float(c) for c in pt), float(h)))

    def refine_surface(self, obj: GeoObject, h: float) -> None:
        """per-solid surface sizing (reaches void walls, e.g. coax inner
        conductors): boundary patches mesh at ``h`` and grade outward"""
        self._geometry_entry(obj)["surface_maxh"] = float(h)

    # ------------------------------------------------------------ meshing

    def mesh(self, maxh: float | None = None, *, radius_edge: float = 2.0,
             max_points: int = 2_000_000, grading: float | None = None,
             algorithm: str = "rapidmesh", optimize: bool = True):
        """assemble the scene, mesh it, resolve physics selections, and store
        the solver arrays; returns ``self``"""
        del algorithm, optimize  # retired gmsh knobs
        eff_maxh = float(maxh) if maxh is not None else self.maxh
        if eff_maxh is None:
            raise ValueError(
                "no mesh size: pass maxh to Geometry(...) or g.mesh(maxh=...)")
        gm = _rm.Geometry(
            maxh=eff_maxh,
            grading=self._grading if grading is None else float(grading),
        )
        rm_solids: list = []
        for entry in self._solids:
            rm_obj = _build_rapidmesh(gm, entry)
            if entry["sheet_tag"] == 0:
                rm_solids.append(rm_obj)
                if entry["surface_maxh"] is not None:
                    gm.refine_surface(rm_obj, entry["surface_maxh"])
        for pt, h in self._size_points:
            gm.refine_near_points([pt], h)

        m = gm.mesh(radius_edge=radius_edge, max_points=max_points)
        if m.stats.get("abandoned_patches", 0):
            raise RuntimeError(
                f"mesh is not fully conforming "
                f"({m.stats['abandoned_patches']} abandoned patches); "
                f"the geometry needs review")
        self._mesh = m

        self._region_of_solid = {}
        for entry in self._solids:
            if entry["sheet_tag"] == 0 and not entry["void"]:
                s3 = entry["solid3"]
                self._region_of_solid[s3] = rm_solids[s3].region

        self._build_groups()
        self._resolve_physics()
        self._collect_materials()
        return self

    # -- facet groups -------------------------------------------------------------

    def _build_groups(self) -> None:
        m = self._mesh
        faces = m.faces.astype(np.int64)
        pts = m.points
        tri_pts = pts[faces]
        cents = tri_pts.mean(axis=1)
        areas = 0.5 * np.linalg.norm(
            np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]),
            axis=1,
        )
        r = m.face_regions.astype(np.int64)
        rlo = np.minimum(r[:, 0], r[:, 1])
        rhi = np.maximum(r[:, 0], r[:, 1])
        owners = m.surface_owners.astype(np.int64)[m.face_surfaces.astype(np.int64)]
        keys = np.stack(
            [m.face_tags.astype(np.int64), m.face_surfaces.astype(np.int64),
             rlo, rhi],
            axis=1,
        )
        _, first, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True)
        groups: list[_FaceGroup] = []
        for gi in range(len(first)):
            idx = np.nonzero(inverse == gi)[0]
            a = areas[idx]
            w = a.sum()
            cog = ((cents[idx] * a[:, None]).sum(axis=0) / w
                   if w > 0 else cents[idx].mean(axis=0))
            gp = tri_pts[idx].reshape(-1, 3)
            k = keys[first[gi]]
            groups.append(_FaceGroup(
                tri_idx=idx, cog=tuple(cog),
                bbox=(*gp.min(axis=0), *gp.max(axis=0)),
                sheet_tag=int(k[0]), owner=int(owners[idx[0]]),
                regions=(int(k[2]), int(k[3])),
            ))
        self._groups = groups
        self._model_bbox = (*pts.min(axis=0), *pts.max(axis=0))
        self._tol = _TOL_REL * max(
            self._model_bbox[3] - self._model_bbox[0],
            self._model_bbox[4] - self._model_bbox[1],
            self._model_bbox[5] - self._model_bbox[2],
        )

    def _base_groups(self, base) -> list[_FaceGroup]:
        kind, key = base
        out = []
        for g in self._groups:
            if kind == "sheet":
                if g.sheet_tag == key:
                    out.append(g)
            elif kind == "solid":
                entry = self._solids[key]
                s3 = entry["solid3"]
                region = 0 if entry["void"] else self._region_of_solid[s3]
                if g.sheet_tag != 0:
                    continue  # tagged sheets belong to their plate objects
                if g.owner == s3 or (region != 0 and region in g.regions):
                    out.append(g)
        return out

    def _resolve_entity(self, ent: _Entity) -> list[_FaceGroup]:
        groups = self._base_groups(ent.base)
        tol = self._tol
        ax = {"x": 0, "y": 1, "z": 2}
        for op in ent.ops:
            if op[0] == "extreme":
                k, sign = ax[op[1]], op[2]
                if not groups:
                    break
                vals = [g.cog[k] for g in groups]
                m = max(vals) if sign > 0 else min(vals)
                groups = [g for g in groups if abs(g.cog[k] - m) <= tol]
            elif op[0] == "where":
                groups = [g for g in groups if op[1](g.cog, g.bbox)]
            elif op[0] == "outer":
                groups = [g for g in groups
                          if _on_box(g.bbox, self._model_bbox, tol)]
            elif op[0] == "hull":
                if groups:
                    lo = np.min([g.bbox[:3] for g in groups], axis=0)
                    hi = np.max([g.bbox[3:] for g in groups], axis=0)
                    hb = (*lo, *hi)
                    groups = [g for g in groups if _on_box(g.bbox, hb, tol)]
            elif op[0] == "unassigned":
                groups = [g for g in groups if not g.claimed]
        return groups

    # -- physics resolution -------------------------------------------------------

    def _resolve_physics(self) -> None:
        from rapidfem.physics import PML

        tris: list[np.ndarray] = []
        tri_tags: list[np.ndarray] = []
        self._physics_tags = {}
        next_tag = 1
        m = self._mesh
        faces = m.faces.astype(np.int64)
        for phys in self._physics:
            if isinstance(phys, PML):
                regions = []
                for ent in phys._entities:
                    if ent.base[0] != "volume":
                        raise ValueError("PML targets must be material solids")
                    regions.append(self._region_of_solid[
                        self._solids[ent.base[1]]["solid3"]])
                if len(set(regions)) != 1:
                    raise ValueError("one PML object per volume")
                self._physics_tags[id(phys)] = regions[0]
                continue
            picked: list[np.ndarray] = []
            for ent in phys._entities:
                for g in self._resolve_entity(ent):
                    g.claimed = True
                    picked.append(g.tri_idx)
            if not picked:
                raise ValueError(
                    f"{type(phys).__name__}: face selection matched nothing "
                    f"(predicates evaluate on mesh facet groups)")
            idx = np.unique(np.concatenate(picked))
            tag = next_tag
            next_tag += 1
            self._physics_tags[id(phys)] = tag
            tris.append(faces[idx])
            tri_tags.append(np.full(len(idx), tag, dtype=np.int32))

        nodes = m.points
        tets = m.tets.astype(np.int64)
        tet_tags = m.tet_regions.astype(np.int32)
        all_tris = (np.concatenate(tris) if tris
                    else np.zeros((0, 3), dtype=np.int64))
        all_tags = (np.concatenate(tri_tags) if tri_tags
                    else np.zeros(0, dtype=np.int32))
        self._mesh_arrays = (nodes, tets, tet_tags, all_tris, all_tags)

    def _collect_materials(self) -> None:
        from rapidfem.physics import PML

        pml_regions: set[int] = set()
        for phys in self._physics:
            if isinstance(phys, PML):
                pml_regions.add(self._physics_tags[id(phys)])
        self._volume_materials = []
        for entry in self._solids:
            if entry["sheet_tag"] != 0 or entry["void"]:
                continue
            region = self._region_of_solid[entry["solid3"]]
            if region in pml_regions:
                continue  # the PML stretch profile carries its own tensors
            if entry["material"] is None:
                raise ValueError(
                    f"solid #{entry['solid3']} ({entry['kind']}) has no "
                    f"material; every non-PML volume needs one (the solver "
                    f"zero-fills unlisted volumes and fails in assembly)")
            self._volume_materials.append((region, entry["material"]))

    # ------------------------------------------------------------ misc

    @property
    def n_tets(self) -> int:
        return 0 if self._mesh is None else int(self._mesh.stats["n_tets"])

    def close(self) -> None:
        """retired gmsh-session hook, kept for API compatibility"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# BUILD HELPERS =========================================================================


def _profile_points(profile) -> list:
    pts = [tuple(float(c) for c in p) for p in profile]
    if len(pts) < 3:
        raise ValueError("profiles need at least 3 points")
    return pts


def _build_rapidmesh(gm, entry: dict):
    """replay one recorded primitive into the rapidmesh builder"""
    k, a = entry["kind"], entry["args"]
    maxh = entry["maxh"]
    void = entry["void"]
    if k == "box":
        w, d, h = a["size"]
        return gm.box(w, d, h, position=a["position"], maxh=maxh, void=void)
    if k == "cylinder":
        return gm.cylinder(a["radius"], a["height"], position=a["position"],
                           axis=a["axis"], segments=a["segments"],
                           maxh=maxh, void=void)
    if k == "sphere":
        return gm.sphere(a["radius"], position=a["center"],
                         segments=a["segments"], rings=a["rings"],
                         maxh=maxh, void=void)
    if k == "cone":
        return gm.cone(a["r1"], a["r2"], a["height"], position=a["position"],
                       axis=a["axis"], segments=a["segments"],
                       maxh=maxh, void=void)
    if k == "wedge":
        return gm.wedge(a["dx"], a["dy"], a["dz"], position=a["position"],
                        top_x=a["top_x"], maxh=maxh, void=void)
    if k == "torus":
        return gm.torus(a["major_radius"], a["minor_radius"],
                        position=a["center"], axis=a["axis"],
                        segments=a["segments"],
                        tube_segments=a["tube_segments"], maxh=maxh, void=void)
    if k == "loft":
        return gm.loft(a["profile_a"], a["profile_b"], maxh=maxh, void=void)
    if k == "plate":
        return gm.plate(a["p0"], a["du"], a["dv"],
                        tag=entry["sheet_tag"], maxh=maxh)
    if k == "disc":
        return gm.disc(a["radius"], position=a["position"], axis=a["axis"],
                       segments=a["segments"], tag=entry["sheet_tag"], maxh=maxh)
    if k == "polygon_plate":
        return gm.polygon_plate(a["points"], position=a["position"],
                                holes=a["holes"] or None,
                                tag=entry["sheet_tag"], maxh=maxh)
    raise ValueError(f"unknown primitive kind {k!r}")


def _on_box(bbox, ref, tol) -> bool:
    """True when the face bbox is degenerate in one axis AND that plane lies
    on the reference box (the gmsh-era 'outer' test)."""
    for k in range(3):
        if abs(bbox[3 + k] - bbox[k]) <= tol:
            if abs(bbox[k] - ref[k]) <= tol or abs(bbox[k] - ref[3 + k]) <= tol:
                return True
    return False
