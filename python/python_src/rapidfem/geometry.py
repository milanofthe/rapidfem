# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Geometry builder on the native rapidmesh scene.

The scene lives in Rust (``rapidfem._native.Geometry``): a list of solids and
sheets, realised into rapidmesh whenever a selection or a mesh needs it.
Solids overlap by priority, a solid added later carves its region out of the
earlier ones, and the mesh is always conformal, so there is no separate
boolean fragment step. Faces are named by their origin ``(object, role)``,
the role of the surface in its primitive (a box: -z, +z, -y, +y, -x, +x;
-1 for a sheet), which survives later changes of the scene.

This module is the Python face of that scene: :class:`Geometry`,
:class:`GeoObject` and the selectable :class:`EntityCollection`.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

from . import _native
from ._geometry_gds import _GdsMixin
from ._geometry_import import _ImportMixin
from ._geometry_primitives import _PrimitivesMixin

# Two face centroids closer than this (relative to the model size) count as
# the same position for min / max selection.
_COG_TOL_REL = 1e-9
# A face whose bounding box is thinner than this (relative) along an axis is
# planar in that axis for the hull selectors.
_HULL_TOL_REL = 1e-9


class _Entity:
    """A selectable piece of the scene: a solid object (``dim == 3``) or a
    face, named by its selection ``(object, role, side)`` (``dim == 2``;
    role -1 is a sheet; ``side`` is the solid the face was selected through,
    -1 for none, so a face split by a later solid keeps only the pieces
    bounding that solid).

    ``material``, ``name`` and ``maxh`` are the attributes the physics and
    mesh layers read; geometric properties (``cog``, ``bbox``) are looked up
    on the realised scene when asked for.
    """

    def __init__(self, geometry: "Geometry", dim: int, *, obj: int | None = None,
                 origin: tuple[int, int, int] | None = None):
        self._geometry = geometry
        self.dim = dim
        self.obj = obj
        self.origin = origin
        self.material = None
        self.name: str | None = None
        self.maxh: float | None = None

    def __repr__(self) -> str:
        what = f"object {self.obj}" if self.dim == 3 else f"face {self.origin}"
        return f"_Entity({what})"

    def _info(self):
        if self.dim == 3:
            return self._geometry._object_info(self.obj)
        return self._geometry._face_info([self.origin])[0]

    @property
    def cog(self) -> tuple[float, float, float]:
        return tuple(self._info()[0])

    @property
    def bbox(self) -> tuple[float, ...]:
        return tuple(self._info()[4])


class EntityCollection:
    """A collection of faces (or solids) with selectors and bulk attribute
    writes.

    Returned by ``obj.faces`` and ``obj.edges`` on a :class:`GeoObject`,
    and by selector methods (:meth:`min`, :meth:`max`, :meth:`where`,
    :attr:`unassigned`, :attr:`outer`, :attr:`hull`) on existing
    collections. Every selector returns a *new* collection so chains
    compose:

    .. code-block:: python

        port_face   = air.faces.min(axis="z")
        top_corner  = air.faces.where(lambda c, b: c[2] > 0.5).max(axis="x")
    """

    def __init__(self, geometry: "Geometry", entities: list[_Entity]):
        self._geometry = geometry
        self._entities = entities

    def __iter__(self):
        return iter(self._entities)

    def __len__(self):
        return len(self._entities)

    def __getitem__(self, i):
        return self._entities[i]

    def __repr__(self) -> str:
        return f"EntityCollection({len(self._entities)} entities)"

    def _infos(self):
        return [e._info() for e in self._entities]

    def _extreme(self, axis: str, pick) -> "EntityCollection":
        if not self._entities:
            return EntityCollection(self._geometry, [])
        ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        cogs = [info[0][ax] for info in self._infos()]
        m = pick(cogs)
        tol = _COG_TOL_REL * self._geometry._size()
        kept = [e for e, c in zip(self._entities, cogs) if abs(c - m) <= tol]
        return EntityCollection(self._geometry, kept)

    def min(self, axis: str = "z") -> "EntityCollection":
        """keep only entities whose centroid is at the minimum along ``axis``

        For a convex primitive (box, cylinder) this usually returns a
        single-face collection, exactly what you want for picking
        ports or specific walls.


        Example
        -------
        .. code-block:: python

            port_face = air.faces.min(axis="z")
            rf.RectWaveguidePort(port_face)


        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            axis to compare centroids on

        Returns
        -------
        EntityCollection
            subset of this collection at the min coordinate
        """
        return self._extreme(axis, min)

    def max(self, axis: str = "z") -> "EntityCollection":
        """keep only entities whose centroid is at the maximum along ``axis``

        Mirror of :meth:`min`; see that method for the worked example.


        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            axis to compare centroids on

        Returns
        -------
        EntityCollection
            subset of this collection at the max coordinate
        """
        return self._extreme(axis, max)

    def where(self, predicate: Callable[[tuple, tuple], bool]) -> "EntityCollection":
        """filter entities by a user-supplied predicate on centroid + bbox

        The most flexible selector, escape hatch for selecting faces
        by region or orientation that don't reduce to a simple ``min``
        / ``max`` along an axis.


        Example
        -------
        Horn antenna's trapezoidal side flares (faces strictly inside
        the horn region in x):

        .. code-block:: python

            rf.PEC(*horn.faces.where(lambda c, b: 1e-6 < c[0] < Lhorn - 1e-6))


        Parameters
        ----------
        predicate : Callable[[tuple, tuple], bool]
            function ``(centroid, bbox) -> bool``; ``centroid`` is a
            3-tuple and ``bbox`` is ``(xmin, ymin, zmin, xmax, ymax, zmax)``

        Returns
        -------
        EntityCollection
            entities for which the predicate returned True
        """
        kept = [e for e, info in zip(self._entities, self._infos())
                if predicate(tuple(info[0]), tuple(info[4]))]
        return EntityCollection(self._geometry, kept)

    @property
    def name(self) -> str | None:
        names = {e.name for e in self._entities}
        return names.pop() if len(names) == 1 else None

    @name.setter
    def name(self, value: str) -> None:
        for e in self._entities:
            e.name = value

    @property
    def maxh(self) -> float | None:
        hs = {e.maxh for e in self._entities}
        return hs.pop() if len(hs) == 1 else None

    @maxh.setter
    def maxh(self, value: float) -> None:
        for e in self._entities:
            e.maxh = value
        self._geometry._apply_maxh(self._entities, value)

    @property
    def material(self):
        """Assign one material to every volume in the collection.

        The canonical way to give a material to an imported-mesh group, e.g.
        ``scene.group("substrate").material = rf.Dielectric(er=4.4)``. Applies
        to volume (dim 3) members; assigning to a face collection is a no-op for
        the FEM (faces carry boundary physics, not bulk materials).
        """
        mats = {id(e.material): e.material for e in self._entities}
        return next(iter(mats.values())) if len(mats) == 1 else None

    @material.setter
    def material(self, value) -> None:
        for e in self._entities:
            e.material = value

    @property
    def unassigned(self) -> "EntityCollection":
        """subset of entities with no physics object pointing at them yet

        Filters this collection against the geometry's physics registry
        and drops any entity already targeted by a port or BC. The
        canonical use is a catch-all PEC declaration after the explicit
        port faces have been declared.


        Example
        -------
        .. code-block:: python

            rf.RectWaveguidePort(air.faces.min(axis="z"))
            rf.RectWaveguidePort(air.faces.max(axis="z"))
            rf.PEC(*air.faces.unassigned)   # everything else


        Returns
        -------
        EntityCollection
            entities not yet referenced by any physics object
        """
        targeted = set()
        for phys in self._geometry._physics:
            for ent in getattr(phys, "_entities", ()):
                targeted.add(ent.origin if ent.dim == 2 else ("obj", ent.obj))
        kept = [e for e in self._entities
                if (e.origin if e.dim == 2 else ("obj", e.obj)) not in targeted]
        return EntityCollection(self._geometry, kept)

    @property
    def outer(self) -> "EntityCollection":
        """axis-aligned faces lying on the bounding box of the whole model

        Picks the faces of this collection that lie flat on one of the six
        planes of the model's axis-aligned bounding box, typically the outer
        walls of the enclosing air box.


        Example
        -------
        .. code-block:: python

            rf.ABC(*air.faces.outer)


        Returns
        -------
        EntityCollection
            entities on the model bounding box
        """
        return self._on_box(self._geometry._model_bbox())

    @property
    def hull(self) -> "EntityCollection":
        """axis-aligned faces lying on *this collection's own* bounding box

        Like :attr:`outer`, but the reference box is the bounding box of
        the entities in *this* collection, not the whole gmsh model. A
        face is kept iff one of its axes is degenerate and its coordinate
        along that axis matches this collection's extremum.

        This is what you want for a closed surface around an object that
        is itself wrapped by something else. ``air.faces.outer`` returns
        nothing once ``air`` is surrounded by PML on every side (none of
        air's faces touch the *model* bbox any more, they are all interior
        air↔PML interfaces); ``air.faces.hull`` returns those six interface
        faces, the natural near-field-to-far-field (Huygens) surface.


        Example
        -------
        Far-field surface on the air / PML interface:

        .. code-block:: python

            rf.FarFieldSurface(*air.faces.hull)


        Returns
        -------
        EntityCollection
            entities on this collection's bounding box
        """
        infos = self._infos()
        if not infos:
            return EntityCollection(self._geometry, [])
        b = np.array([info[4] for info in infos])
        box = (*b[:, :3].min(axis=0), *b[:, 3:].max(axis=0))
        return self._on_box(box)

    def _on_box(self, box) -> "EntityCollection":
        xmin, ymin, zmin, xmax, ymax, zmax = box
        tol = _HULL_TOL_REL * self._geometry._size()
        kept = []
        for e, info in zip(self._entities, self._infos()):
            ex0, ey0, ez0, ex1, ey1, ez1 = info[4]
            on_box = (
                (abs(ex1 - ex0) < tol and (abs(ex0 - xmin) < tol or abs(ex0 - xmax) < tol))
                or (abs(ey1 - ey0) < tol and (abs(ey0 - ymin) < tol or abs(ey0 - ymax) < tol))
                or (abs(ez1 - ez0) < tol and (abs(ez0 - zmin) < tol or abs(ez0 - zmax) < tol))
            )
            if on_box:
                kept.append(e)
        return EntityCollection(self._geometry, kept)


# Back-compat aliases (and clearer naming for users)
FaceCollection = EntityCollection
EdgeCollection = EntityCollection


class _Edge:
    """An edge of a solid object, named by the roles of its two faces."""

    def __init__(self, geometry: "Geometry", obj: int, roles: tuple[int, int],
                 midpoint):
        self._geometry = geometry
        self.dim = 1
        self.obj = obj
        self.roles = roles
        self.midpoint = tuple(midpoint)

    def _info(self):
        m = self.midpoint
        return (m, (0.0, 0.0, 0.0), 0.0, [], (*m, *m))


class GeoObject:
    """A solid or sheet of a :class:`Geometry`, as returned by the primitive
    methods (:meth:`Geometry.box`, :meth:`Geometry.xy_plate`, ...).

    ``faces`` selects its bounding faces (interfaces with other solids
    included), ``edges`` its edges; ``material``, ``maxh`` and ``name``
    are read by the mesh and physics layers.
    """

    def __init__(self, geometry: "Geometry", entity: _Entity):
        self._geometry = geometry
        self._entity = entity

    def __repr__(self) -> str:
        kind = "sheet" if self.dim == 2 else "solid"
        return f"GeoObject({kind} {self._id})"

    @property
    def _id(self) -> int:
        e = self._entity
        return e.obj if e.dim == 3 else e.origin[0]

    @property
    def faces(self) -> EntityCollection:
        g = self._geometry
        if self.dim == 2:
            return EntityCollection(g, [self._entity])
        return EntityCollection(g, [g._face(o) for o in g._native.faces_of(self._id)])

    @property
    def edges(self) -> EntityCollection:
        g = self._geometry
        if self.dim == 2:
            return EntityCollection(g, [])
        return EntityCollection(
            g, [_Edge(g, self._id, roles, mid) for roles, mid in g._native.edges_of(self._id)])

    @property
    def name(self) -> str | None:
        return self._entity.name

    @name.setter
    def name(self, value: str) -> None:
        self._entity.name = value

    @property
    def material(self):
        return self._entity.material

    @material.setter
    def material(self, value) -> None:
        self._entity.material = value

    @property
    def maxh(self) -> float | None:
        return self._entity.maxh

    @maxh.setter
    def maxh(self, value: float) -> None:
        self._entity.maxh = value
        self._geometry._apply_maxh([self._entity], value)

    @property
    def dim(self) -> int:
        return self._entity.dim


class MeshStats:
    """Size and quality report of the last generated mesh.

    Stored on ``geometry.mesh_stats`` by :meth:`Geometry.mesh`. The DOF
    numbers are exact bounds for the FD solver's p-adaptive Nédélec space:
    ``dofs_min`` is the uniform order-1 count (one DOF per edge),
    ``dofs_max`` the uniform order-2 count (two per edge plus two per
    triangular face). The solver's wavelength policy lands in between, so
    the bounds gate RAM/runtime budgets before any assembly happens.
    ``quality_min`` is the smallest dihedral angle in degrees (the sliver
    indicator that governs the conditioning), ``n_slivers`` the number of
    tets below the sliver threshold.
    """
    n_nodes: int
    n_tets: int
    n_tris: int           # unique tet faces, interior + boundary
    n_edges: int          # unique tet edges
    dofs_min: int         # uniform order 1: n_edges
    dofs_max: int         # uniform order 2: 2*n_edges + 2*n_tris
    quality_min: float    # smallest dihedral angle, degrees
    n_slivers: int
    groups: dict          # group name -> element count

    def __repr__(self) -> str:
        return (f"MeshStats({self.n_nodes} nodes, {self.n_tets} tets, "
                f"{self.n_edges} edges, dofs {self.dofs_min}..{self.dofs_max}, "
                f"min dihedral {self.quality_min:.1f} deg, {self.n_slivers} slivers)")


class Geometry(_GdsMixin, _PrimitivesMixin, _ImportMixin):
    """Top-level geometry builder on the native rapidmesh scene.

    Build with primitive factory methods (:meth:`box`, :meth:`cylinder`,
    :meth:`xy_plate`, ...) each of which returns a :class:`GeoObject`.
    Attach physics via the object-API constructors
    (:class:`rapidfem.RectWaveguidePort`, :class:`rapidfem.PEC`, ...)
    pointing at faces or volumes. When the description is complete,
    call :meth:`mesh` and feed the geometry to a
    :class:`rapidfem.Problem` for analysis.

    A solid added later carves its region out of the solids added before
    it, and the mesh is always conformal: overlapping objects need no
    boolean step.


    Example
    -------
    .. code-block:: python

        g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
        air = g.box(22.86e-3, 10.16e-3, 30e-3,
                    position=(-11.43e-3, -5.08e-3, 0),
                    material=rf.Air())
        rf.RectWaveguidePort(air.faces.min(axis="z"))
        rf.RectWaveguidePort(air.faces.max(axis="z"))
        rf.PEC(*air.faces.unassigned)
        g.mesh()


    Parameters
    ----------
    maxh : float, optional
        global maximum tet edge length in metres; used by
        :meth:`mesh` when no explicit override is passed
    scale : float
        accepted for compatibility; the mesher works with exact predicates
        at any length scale
    grading : bool
        grade element sizes from fine features into the bulk
    name : str, optional
        model name (for diagnostic / log output)
    """

    def __init__(self, *, maxh: float | None = None, scale: float = 1.0,
                 grading: bool = True, name: str = "rapidfem"):
        self._native = _native.Geometry(maxh)
        self._maxh = maxh
        self._scale = 1.0
        self._grading = bool(grading)
        self.name = name
        self._objects: list[GeoObject] = []
        self._entities: list[_Entity] = []
        self._faces: dict[tuple[int, int], _Entity] = {}
        self._physics: list = []
        self._material_tags: dict[int, int] = {}
        self._physics_tags: dict = {}
        self._fem_mesh = None
        self._last_mesh = None
        self.mesh_stats: MeshStats | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Release the scene. Kept for the context-manager protocol; the
        native scene holds no global state."""

    # ── scene bookkeeping ───────────────────────────────────────────────────

    def _wrap(self, obj_id: int, *, sheet: bool, material=None,
              maxh: float | None = None) -> GeoObject:
        ent = _Entity(self, 2 if sheet else 3, obj=None if sheet else obj_id,
                      origin=(obj_id, -1, -1) if sheet else None)
        ent.material = material
        ent.maxh = maxh
        if sheet:
            self._faces[(obj_id, -1, -1)] = ent
        obj = GeoObject(self, ent)
        self._objects.append(obj)
        self._entities.append(ent)
        return obj

    def _face(self, origin) -> _Entity:
        """The face entity of an origin, one per origin (so attributes set
        on it persist)."""
        origin = (int(origin[0]), int(origin[1]), int(origin[2]))
        ent = self._faces.get(origin)
        if ent is None:
            ent = _Entity(self, 2, origin=origin)
            self._faces[origin] = ent
            self._entities.append(ent)
        return ent

    def _face_info(self, origins):
        return self._native.face_info([tuple(o) for o in origins])

    def _object_info(self, obj_id: int):
        """(centroid, 0, 0, [], bbox) of a solid from its bounding faces."""
        infos = self._face_info(self._native.faces_of(obj_id))
        if not infos:
            return ((0.0,) * 3, (0.0,) * 3, 0.0, [], (0.0,) * 6)
        b = np.array([i[4] for i in infos])
        box = (*b[:, :3].min(axis=0), *b[:, 3:].max(axis=0))
        c = tuple((np.array(box[:3]) + np.array(box[3:])) / 2)
        return (c, (0.0,) * 3, 0.0, [], box)

    def _model_bbox(self):
        infos = self._face_info(self._native.all_faces())
        if not infos:
            return (0.0,) * 6
        b = np.array([i[4] for i in infos])
        return (*b[:, :3].min(axis=0), *b[:, 3:].max(axis=0))

    def _size(self) -> float:
        b = self._model_bbox()
        return max(b[3] - b[0], b[4] - b[1], b[5] - b[2], 1e-300)

    def _apply_maxh(self, entities, h: float) -> None:
        faces = [e.origin for e in entities if e.dim == 2]
        for e in entities:
            if e.dim == 3:
                self._native.set_object_maxh(e.obj, h)
            elif e.origin[1] < 0:
                self._native.set_object_maxh(e.origin[0], h)
        if faces:
            self._native.set_face_maxh([f for f in faces if f[1] >= 0], h)

    def _solid_ids(self, objs) -> list[int]:
        ids = []
        for o in objs:
            if o.dim != 3:
                raise ValueError(f"{o!r} is not a solid")
            ids.append(o._id)
        return ids

    # ── solids from faces ───────────────────────────────────────────────────

    def extrude(self, face: GeoObject, height: float,
                axis: tuple[float, float, float] = (0, 0, 1),
                *,
                material=None,
                maxh: float | None = None) -> GeoObject:
        """extrude a 2-D face along ``axis * height`` into a 3-D volume

        The source ``face`` becomes the bottom cap of the new volume
        and remains tracked in the entity registry (so names and per-
        entity mesh sizes set on it survive).


        Example
        -------
        A 35 µm copper trace from a polygon footprint:

        .. code-block:: python

            poly = g.polygon([(0, 0), (1e-3, 0), (1e-3, 0.5e-3), (0, 0.5e-3)])
            trace = g.extrude(poly, height=35e-6)


        Parameters
        ----------
        face : GeoObject
            2-D face from :meth:`polygon`, :meth:`disc`, :meth:`plate`,
            etc.
        height : float
            sweep distance along ``axis``
        axis : tuple[float, float, float]
            sweep direction, will be scaled by ``height`` (defaults
            to +z)
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override

        Returns
        -------
        GeoObject
            new volume
        """
        ax = np.asarray(axis, dtype=float)
        n = np.linalg.norm(ax)
        if n == 0 or abs(abs(ax[2]) / n - 1.0) > 1e-12:
            raise NotImplementedError(
                "extrude: only along z so far (general directions: "
                "milanofthe/rapidmesh-dev#140)")
        h = float(height) * (1.0 if ax[2] > 0 else -1.0)
        oid = face._id
        self._native.extrude(oid, h)
        ent = face._entity
        self._faces.pop((oid, -1, -1), None)
        ent.dim, ent.obj, ent.origin = 3, oid, None
        ent.material = material
        if maxh is not None:
            ent.maxh = maxh
            self._native.set_object_maxh(oid, maxh)
        return face

    def loft(self, face_a: GeoObject, face_b: GeoObject, ruled: bool = True,
             *,
             material=None,
             maxh: float | None = None) -> GeoObject:
        """loft a volume between two coplanar / parallel 2-D faces

        Linearly interpolates the perimeter of ``face_a`` onto the
        perimeter of ``face_b``. Both faces must have the same number
        of edges in their outer boundary (a 4-edge rectangle lofts to
        a 4-edge rectangle, producing a frustum with 4 trapezoidal
        sides).


        Note
        ----
        The input faces are absorbed into the new volume's boundary
        and remain tracked as cap faces.


        Example
        -------
        Pyramidal horn between a WR-90 throat and a flared aperture:

        .. code-block:: python

            throat = g.polygon([(0, -wga/2, -wgb/2), (0,  wga/2, -wgb/2),
                                (0,  wga/2,  wgb/2), (0, -wga/2,  wgb/2)])
            aper   = g.polygon([(L, -WH/2, -HH/2),   (L,  WH/2, -HH/2),
                                (L,  WH/2,  HH/2),   (L, -WH/2,  HH/2)])
            horn = g.loft(throat, aper)


        Parameters
        ----------
        face_a, face_b : GeoObject
            two 2-D faces to bridge
        ruled : bool
            ``True`` (default) gives flat side surfaces, the right
            choice for pyramidal / frustum-style horns; ``False`` fits
            a spline through the section profiles
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override

        Returns
        -------
        GeoObject
            new volume
        """
        pa, pb = self._outline(face_a), self._outline(face_b)
        if len(pa) != len(pb):
            raise ValueError("loft: both profiles need the same vertex count")
        oid = self._native.add_loft([list(p) for p in pa], [list(p) for p in pb], maxh)
        for f in (face_a, face_b):
            self._native.remove(f._id)
        return self._wrap(oid, sheet=False, material=material, maxh=maxh)

    def revolve(self, face: GeoObject,
                axis_point: tuple[float, float, float] = (0, 0, 0),
                axis_dir: tuple[float, float, float] = (0, 0, 1),
                angle: float = 2 * math.pi,
                *,
                material=None,
                maxh: float | None = None) -> GeoObject:
        """revolve a 2-D face around an axis to create a 3-D volume

        For a full :math:`2\\pi` sweep the profile typically touches
        the axis to close the body; partial sweeps produce a wedge-shaped
        volume.


        Example
        -------
        Conical horn from a 4-point profile revolved around the x-axis:

        .. code-block:: python

            profile = g.polygon([(L, 0), (L+a, 0), (L+a, R), (L, r)])
            horn = g.revolve(profile, axis_point=(0, 0, 0), axis_dir=(1, 0, 0))


        Parameters
        ----------
        face : GeoObject
            2-D face to revolve
        axis_point : tuple[float, float, float]
            a point on the rotation axis (defaults to origin)
        axis_dir : tuple[float, float, float]
            axis direction (defaults to +z)
        angle : float
            sweep angle in radians (defaults to :math:`2\\pi`)
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override

        Returns
        -------
        GeoObject
            new volume
        """
        p0 = np.asarray(axis_point, dtype=float)
        a = np.asarray(axis_dir, dtype=float)
        a = a / np.linalg.norm(a)
        rz = []
        for p in self._outline(face):
            d = np.asarray(p, dtype=float) - p0
            z = float(d @ a)
            r = float(np.linalg.norm(d - z * a))
            rz.append([r, z])
        oid = self._native.add_revolve(rz, list(p0), list(a), math.degrees(angle), maxh)
        self._native.remove(face._id)
        return self._wrap(oid, sheet=False, material=material, maxh=maxh)

    def _outline(self, face: GeoObject) -> list[tuple[float, float, float]]:
        """The 3-D vertex loop of a planar profile made by :meth:`polygon`,
        :meth:`plate` or a ``*_plate``."""
        pts = self._profiles.get(face._id) if hasattr(self, "_profiles") else None
        if pts is None:
            raise ValueError(f"{face!r} is not a polygon or plate profile")
        return pts

    # ── booleans ────────────────────────────────────────────────────────────

    def fragment(self, target: GeoObject, *tools: GeoObject) -> None:
        """make ``target`` and ``tools`` conformal

        Kept for compatibility: the scene is always assembled conformally
        (every interface is shared by the mesh on both sides) and a solid
        added later carves its region out of the ones added before it, which
        is what fragmenting an inner object with its surrounding volume did.
        Nothing to do.


        Parameters
        ----------
        target : GeoObject
            object to fragment
        *tools : GeoObject
            objects to fragment with
        """

    def cut(self, target: GeoObject, *tools: GeoObject) -> None:
        """subtract ``tools``, leaving holes

        Each tool becomes a void: it is cut out of every solid added before
        it (``target`` among them) and its walls become boundary faces that
        physics can address (``tool`` stays usable as a selection handle, its
        ``faces`` are the walls). Boundary faces without physics are PEC.


        Parameters
        ----------
        target : GeoObject
            object to subtract from
        *tools : GeoObject
            objects to subtract
        """
        self._native.make_void(self._solid_ids(tools))

    def fuse(self, target: GeoObject, *tools: GeoObject) -> None:
        """boolean union ``target ∪ tools``

        Merges the operands into a single connected body assigned back
        to ``target``.


        Note
        ----
        Face names on the operands are **not** preserved (faces merge
        and centroids shift). Top-level volume names survive via the
        gmsh ``out_map``, but set face names AFTER ``fuse``, or use
        :meth:`fragment` if you need the interfaces themselves to
        survive as named entities.


        Parameters
        ----------
        target : GeoObject
            first operand (survives as the merged object)
        *tools : GeoObject
            operands to merge in
        """
        self._native.fuse(self._solid_ids((target, *tools)))

    def intersect(self, target: GeoObject, *tools: GeoObject) -> None:
        """boolean intersect ``target ∩ tools``

        Carves the intersection of ``target`` with every member of
        ``tools`` and assigns it back to ``target``. The tools are
        consumed by the operation.


        Note
        ----
        Tools are **consumed** by ``intersect``, do not reference
        them after the call.


        Example
        -------
        Clip a horn to the upper half-space:

        .. code-block:: python

            g.intersect(horn, halfspace)


        Parameters
        ----------
        target : GeoObject
            object to intersect (survives as the intersection region)
        *tools : GeoObject
            objects to intersect with (consumed)
        """
        raise NotImplementedError("intersect: not available yet (milanofthe/rapidmesh-dev#140)")

    # ── transforms ──────────────────────────────────────────────────────────

    def translate(self, obj: GeoObject,
                  dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """move ``obj`` (and all its child faces / edges) in place by ``(dx, dy, dz)``

        Like :meth:`rotate`, gmsh dimtags survive the transform, only
        the geometric attributes (COG, bbox) of every tracked entity
        descending from ``obj`` are refreshed, so named selectors keep
        resolving to the moved entities.


        Example
        -------
        Lift a feed line 0.5 mm in z:

        .. code-block:: python

            g.translate(feed, dz=0.5e-3)


        Parameters
        ----------
        obj : GeoObject
            volume or face to move
        dx, dy, dz : float
            translation along each axis in metres (default 0 = no move)
        """
        self._native.translate(obj._id, [float(dx), float(dy), float(dz)])
        if hasattr(self, "_profiles") and obj._id in self._profiles:
            self._profiles[obj._id] = [(x + dx, y + dy, z + dz)
                                       for x, y, z in self._profiles[obj._id]]

    def rotate(self, obj: GeoObject, angle: float,
               axis: tuple[float, float, float] = (0, 0, 1),
               center: tuple[float, float, float] = (0, 0, 0)) -> None:
        """rotate ``obj`` (and all its child faces / edges) in place

        gmsh dimtags survive the transform unchanged; only the
        geometric attributes (COG, bbox) of every tracked entity
        descending from ``obj`` are refreshed. Named selectors keep
        working, the resolver sees the new positions.


        Example
        -------
        Rotate a horn 30° around y:

        .. code-block:: python

            g.rotate(horn, math.pi / 6, axis=(0, 1, 0))


        Parameters
        ----------
        obj : GeoObject
            volume or face to rotate
        angle : float
            rotation angle in radians (right-hand rule about ``axis``)
        axis : tuple[float, float, float]
            axis direction (defaults to +z)
        center : tuple[float, float, float]
            a point on the rotation axis (defaults to origin)
        """
        raise NotImplementedError("rotate: not available yet (milanofthe/rapidmesh-dev#140)")

    def stretch(self, obj: GeoObject,
                fx: float = 1.0, fy: float = 1.0, fz: float = 1.0,
                center: tuple[float, float, float] = (0, 0, 0)) -> None:
        """anisotropic scale ``obj`` about ``center`` by ``(fx, fy, fz)``

        Per-axis dilation. The scaling centre stays fixed; everything
        else moves by :math:`(f_x x, f_y y, f_z z)` relative to it.


        Example
        -------
        Squash a circular waveguide by 0.1 % to split degenerate modes:

        .. code-block:: python

            g.stretch(feed, fy=1.001)


        Parameters
        ----------
        obj : GeoObject
            volume or face to scale
        fx, fy, fz : float
            scale factors along each axis (default 1 = no change)
        center : tuple[float, float, float]
            scaling centre (defaults to origin)
        """
        raise NotImplementedError("stretch: not available yet (milanofthe/rapidmesh-dev#140)")

    def mirror(self, obj: GeoObject,
               normal: tuple[float, float, float] = (1, 0, 0),
               point: tuple[float, float, float] = (0, 0, 0)) -> None:
        """reflect ``obj`` in place across the plane through ``point`` with ``normal``

        Useful for building symmetric structures (one half plus its
        mirror image) without re-deriving coordinates. The reflection is
        in place: dimtags survive, COG/bbox of every descendant are
        refreshed, so named selectors keep working.


        Note
        ----
        A reflection flips orientation. For a closed solid this is
        harmless (the volume is still valid), but if you mirror and then
        :meth:`fuse` the two halves, set face names AFTER the fuse (see
        :meth:`fuse`).


        Example
        -------
        Mirror a horn arm across the x = 0 plane (yz-plane):

        .. code-block:: python

            g.mirror(arm, normal=(1, 0, 0))


        Parameters
        ----------
        obj : GeoObject
            volume or face to reflect
        normal : tuple[float, float, float]
            plane normal (need not be unit length); defaults to +x,
            i.e. the yz-plane
        point : tuple[float, float, float]
            a point the plane passes through (defaults to origin)
        """
        raise NotImplementedError("mirror: not available yet (milanofthe/rapidmesh-dev#140)")

    def copy(self, obj: GeoObject, *, material=None,
             maxh: float | None = None) -> GeoObject:
        """duplicate ``obj`` into a new, independent :class:`GeoObject`

        The copy is a fresh body at the same location; move it with
        :meth:`translate` / :meth:`rotate` afterwards (or use
        :meth:`array`, which does this for you). Material and per-entity
        ``maxh`` are inherited from the source unless overridden.


        Note
        ----
        The copy's ``name`` is intentionally **not** inherited: two
        entities sharing a name would make named-face resolution
        ambiguous. Name the copy yourself (or attach physics directly)
        after placing it.


        Example
        -------
        .. code-block:: python

            via2 = g.copy(via1)
            g.translate(via2, dx=1e-3)


        Parameters
        ----------
        obj : GeoObject
            volume or face to duplicate
        material : rapidfem.Material, optional
            material for the copy (defaults to the source's material;
            volumes only)
        maxh : float, optional
            per-entity mesh size for the copy (defaults to the source's)

        Returns
        -------
        GeoObject
            the new, independent duplicate
        """
        raise NotImplementedError("copy: not available yet (milanofthe/rapidmesh-dev#140)")

    def array(self, obj: GeoObject, count: int, *,
              spacing: tuple[float, float, float] | None = None,
              rotation: float | None = None,
              axis: tuple[float, float, float] = (0, 0, 1),
              center: tuple[float, float, float] = (0, 0, 0)) -> list:
        """replicate ``obj`` into a linear or polar array of ``count`` instances

        Pass exactly one of ``spacing`` (linear array) or ``rotation``
        (polar array). The returned list has length ``count`` with the
        original ``obj`` as element ``0`` and the fresh copies after it,
        so a 4-element array yields 3 new bodies plus the original.

        Pairs naturally with :class:`rapidfem.FloquetPort` /
        :class:`rapidfem.PeriodicBoundary` for antenna arrays, frequency
        selective surfaces, and metamaterial unit-cell tilings.


        Example
        -------
        A 1x8 linear patch array on a 12 mm pitch, and a 6-fold polar ring:

        .. code-block:: python

            patches = g.array(patch, 8, spacing=(12e-3, 0, 0))
            petals  = g.array(petal, 6, rotation=2 * math.pi / 6)


        Parameters
        ----------
        obj : GeoObject
            volume or face to replicate
        count : int
            total number of instances including the original (>= 1)
        spacing : tuple[float, float, float], optional
            per-step translation in metres for a linear array
        rotation : float, optional
            per-step rotation angle in radians for a polar array
        axis : tuple[float, float, float]
            rotation axis for the polar case (defaults to +z)
        center : tuple[float, float, float]
            a point on the rotation axis for the polar case (defaults
            to origin)

        Returns
        -------
        list[GeoObject]
            ``count`` instances, ``[0]`` being the original ``obj``

        Raises
        ------
        ValueError
            if ``count < 1`` or not exactly one of ``spacing`` / ``rotation``
        """
        raise NotImplementedError("array: not available yet (milanofthe/rapidmesh-dev#140)")

    # ── edge features ───────────────────────────────────────────────────────

    def fillet(self, obj: GeoObject, radius: float,
               edges: "EntityCollection | None" = None) -> GeoObject:
        """round the edges of a volume with a constant-radius fillet

        Rounds either every edge of ``obj`` (``edges=None``) or just the
        ones in a selected :class:`EntityCollection`, replacing the
        volume with the filleted result. Realistic conductor edges and
        rounded housings need this; sharp edges also concentrate the
        field and stress the mesh.


        Note
        ----
        Filleting reshapes the boundary: the original flat faces and
        sharp edges are replaced by new rounded surfaces, so names /
        materials / ``maxh`` set on the *child faces or edges* of ``obj``
        may not survive (the top-level volume identity does). Select
        faces for physics **after** filleting, or fillet before naming.


        Example
        -------
        Round all 12 edges of a box by 0.2 mm; or just its vertical edges:

        .. code-block:: python

            g.fillet(housing, 0.2e-3)
            g.fillet(post, 50e-6, edges=post.edges.where(
                lambda c, b: b[5] - b[2] > 1e-6))  # tall (z-extent) edges


        Parameters
        ----------
        obj : GeoObject
            volume to fillet (dim must be 3)
        radius : float
            fillet radius in metres
        edges : EntityCollection, optional
            edges to round (defaults to every edge of ``obj``)

        Returns
        -------
        GeoObject
            the same ``obj``, now pointing at the filleted volume

        Raises
        ------
        ValueError
            if ``obj`` is not a volume or has no edges to round
        """
        self._cut_edges(obj, radius, True, edges)
        return obj

    def chamfer(self, obj: GeoObject, distance: float,
                edges: "EntityCollection | None" = None) -> GeoObject:
        """bevel the edges of a volume with a constant chamfer

        The flat-bevel counterpart to :meth:`fillet`: each selected edge
        is replaced by a planar facet set back ``distance`` from the
        edge. Same boundary-reshaping caveat as :meth:`fillet` (child
        face / edge names may not survive).


        Example
        -------
        .. code-block:: python

            g.chamfer(connector_body, 0.1e-3)


        Parameters
        ----------
        obj : GeoObject
            volume to chamfer (dim must be 3)
        distance : float
            chamfer setback in metres
        edges : EntityCollection, optional
            edges to bevel (defaults to every edge of ``obj``)

        Returns
        -------
        GeoObject
            the same ``obj``, now pointing at the chamfered volume

        Raises
        ------
        ValueError
            if ``obj`` is not a volume or has no edges to bevel
        """
        self._cut_edges(obj, distance, False, edges)
        return obj

    def _cut_edges(self, obj, size, fillet, edges):
        picks = None
        if edges is not None:
            picks = []
            for e in edges:
                if not isinstance(e, _Edge) or e.obj != obj._id:
                    raise ValueError(f"{e!r} is not an edge of {obj!r}")
                picks.append(tuple(e.roles))
        self._native.cut_edges(obj._id, float(size), fillet, picks)

    def _hollow(self, name: str) -> list[tuple["EntityCollection", float]]:
        """Turn every solid named ``name`` into a void: its walls become
        boundary faces that carry a surface condition. Returns one
        ``(walls, 2V/S)`` pair for the group, with the volume-to-surface
        thickness in metres (for a long trace of width w and thickness t,
        ``w t / (w + t)``: the thickness at which a two-sided surface
        impedance on every wall reproduces the DC resistance ``1/(sigma w t)``).
        """
        ids = [o._id for o in self._objects if o.dim == 3 and o.name == name]
        if not ids:
            return []
        self._native.make_void(ids)
        walls = sorted({tuple(f) for i in ids for f in self._native.faces_of(i)})
        area = sum(info[2] for info in self._face_info(walls))
        volume = sum(getattr(self, "_prisms", {}).get(i, 0.0) for i in ids)
        if volume <= 0.0 or area <= 0.0:
            raise ValueError(f"_hollow: no prism volume known for {name!r}")
        faces = EntityCollection(self, [self._face(f) for f in walls])
        return [(faces, 2.0 * volume / area)]

    # ── sizing ──────────────────────────────────────────────────────────────

    def auto_refine_features(
        self,
        base_maxh: float,
        resolution: int = 3,
        min_maxh: float | None = None,
    ) -> dict[str, float]:
        """auto-assign per-volume ``maxh`` for any volume thinner than
        ``base_maxh``

        Walks every 3-D volume in the geometry. For each, computes the
        smallest bbox dimension (the "feature size"). If that dimension
        is smaller than ``base_maxh`` *and* the user hasn't already
        set ``vol.maxh`` explicitly, sets

        .. math::

            \\mathrm{vol.maxh} = \\max\\!\\left(
                \\frac{d_{\\min}}{\\mathrm{resolution}},
                \\mathrm{min\\_maxh}
            \\right)

        so the volume is resolved with at least ``resolution`` tets
        across its thinnest axis.


        Note
        ----
        Idempotent, only writes ``maxh`` when it's currently ``None``,
        so explicit per-volume sizes (set via ``g.box(..., maxh=...)``
        or ``obj.maxh = ...``) always win.


        Example
        -------
        Resolve a 0.5 mm thin substrate against a 12 mm global cap:

        .. code-block:: python

            g.auto_refine_features(base_maxh=12e-3, resolution=3)


        Parameters
        ----------
        base_maxh : float
            reference size, volumes wider than this in all directions
            are left untouched
        resolution : int
            target number of tets across the thinnest dimension (3 is
            enough for ND-2 to capture per-element gradients; bump to
            4-5 for very high accuracy near a specific feature)
        min_maxh : float, optional
            floor on the auto-assigned size, to avoid catastrophic
            refinement on micron-scale features

        Returns
        -------
        dict[str, float]
            map ``{volume_descriptor: assigned_maxh}`` for the volumes
            touched (descriptor is the volume's ``name`` if set, else
            ``"vol@(cx,cy,cz)"``)
        """
        assigned: dict[str, float] = {}
        for obj in self._objects:
            if obj.dim != 3 or obj.maxh is not None:
                continue
            box = self._object_info(obj._id)[4]
            dims = (box[3] - box[0], box[4] - box[1], box[5] - box[2])
            positive = [d for d in dims if d > 0]
            if not positive:
                continue
            min_dim = min(positive)
            if min_dim >= base_maxh:
                continue
            h = min_dim / resolution
            if min_maxh is not None:
                h = max(h, min_maxh)
            obj.maxh = h
            c = self._object_info(obj._id)[0]
            desc = obj.name or f"vol@({c[0]*1e3:.1f},{c[1]*1e3:.1f},{c[2]*1e3:.1f})mm"
            assigned[desc] = h
        return assigned

    def refine_near_points(self, points, h: float,
                           distance: float | None = None) -> None:
        """register a local mesh-size refinement around a point cloud

        Every point becomes a size source of target ``h``; the size grows
        from there along the grading, so the element size reaches the
        surrounding target over a distance that follows from the grading
        (``distance`` is accepted for compatibility).


        Parameters
        ----------
        points : array_like, shape (N, 3)
            points in metres
        h : float
            element size at the points
        distance : float, optional
            accepted for compatibility
        """
        for p in np.asarray(points, dtype=float).reshape(-1, 3):
            self._native.add_size_point([float(v) for v in p], float(h))

    # ── meshing ─────────────────────────────────────────────────────────────

    def _assign_groups(self):
        """Tag every material and physics object and build the face and
        volume groups the solver mesh carries (what gmsh physical groups
        were): one tag per Material instance over its solids, one per
        physics object over its faces or volumes, two for a periodic pair.
        """
        next_tag = 1
        face_groups: list = []
        volume_groups: list = []
        self._material_tags = {}
        self._physics_tags = {}
        by_mat: dict[int, tuple] = {}
        for ent in self._entities:
            mat = ent.material
            if mat is None or isinstance(mat, str) or ent.dim != 3:
                continue
            by_mat.setdefault(id(mat), (mat, []))[1].append(ent.obj)
        for mat_id, (_, objs) in by_mat.items():
            self._material_tags[mat_id] = next_tag
            volume_groups.append((next_tag, objs))
            next_tag += 1

        def group(ents):
            faces = [e.origin for e in ents if e.dim == 2]
            vols = [e.obj for e in ents if e.dim == 3]
            return faces, vols

        for phys in self._physics:
            if type(phys).__name__ == "PeriodicBoundary":
                fa, _ = group(phys._entities_a)
                fb, _ = group(phys._entities_b)
                face_groups.append((next_tag, fa))
                face_groups.append((next_tag + 1, fb))
                self._physics_tags[id(phys)] = (next_tag, next_tag + 1)
                next_tag += 2
                continue
            faces, vols = group(phys._entities)
            if vols:
                volume_groups.append((next_tag, vols))
            if faces:
                face_groups.append((next_tag, faces))
            self._physics_tags[id(phys)] = next_tag
            next_tag += 1
        return face_groups, volume_groups

    def save_mesh(self, path: str) -> str:
        """Write the last generated mesh to ``path`` as gmsh ``.msh`` v4.

        The file carries every physical group exactly as the FEM solver sees
        them (materials, PEC/port/ABC targets), so it round-trips through
        :meth:`load` and is directly consumable by external gmsh-format
        solvers such as Palace, same mesh, different solver.

        Requires a prior :meth:`mesh` call; raises otherwise.
        """
        raise NotImplementedError("save_mesh: not available yet")

    def mesh(
        self,
        maxh: float | None = None,
        transition_distance: float | None = None,
        algorithm: str = "hxt",
        optimize: bool | str = True,
        *,
        cells_across: float = 1.0,
        target_elements: int | None = None,
    ):
        """tetrahedralize the scene and build the solver mesh

        Meshes the scene with rapidmesh (restricted-Delaunay refinement with
        exact predicates), honouring the global ``maxh``, per-object and
        per-face sizes and the size points of :meth:`refine_near_points`,
        then tags every material and physics object and hands the mesh to
        the solvers in memory. Fills :attr:`mesh_stats`.


        Parameters
        ----------
        maxh : float, optional
            global size override for this call
        transition_distance, algorithm : optional
            accepted for compatibility (the size field grades by itself;
            there is one algorithm)
        optimize : bool
            run the quality optimizer after meshing
        cells_across : float
            elements across the thickness of every region, so a thin layer
            gets proper tets through it (0 turns it off)
        target_elements : int, optional
            tet budget: the global size is scaled to land near it


        Returns
        -------
        MeshStats
            size and quality of the mesh
        """
        h = maxh if maxh is not None else self._maxh
        n_points, n_tets, min_dihedral, n_slivers = self._native.mesh(
            maxh=h, grading=None if self._grading else 1e9,
            cells_across=float(cells_across), optimize=bool(optimize),
            target_elements=target_elements)
        face_groups, volume_groups = self._assign_groups()
        self._fem_mesh = self._native.fem_mesh(face_groups, volume_groups)
        fm = self._fem_mesh
        stats = MeshStats()
        stats.n_nodes = fm.n_nodes
        stats.n_tets = fm.n_tets
        stats.n_tris = fm.n_tris
        stats.n_edges = fm.n_edges
        stats.dofs_min = fm.n_edges
        stats.dofs_max = 2 * fm.n_edges + 2 * fm.n_tris
        stats.quality_min = float(min_dihedral)
        stats.n_slivers = int(n_slivers)
        vols, faces = fm.group_sizes()
        stats.groups = {f"volume_{t}": n for t, n in vols}
        stats.groups.update({f"face_{t}": n for t, n in faces})
        self.mesh_stats = stats
        self._last_mesh = (self._fem_mesh, {})
        return stats



__all__ = [
    "Geometry", "GeoObject", "EntityCollection", "FaceCollection",
    "EdgeCollection", "MeshStats",
]
