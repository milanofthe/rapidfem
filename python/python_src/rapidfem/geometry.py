# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Geometry builder on the native rapidmesh scene.

The scene lives in Rust (``rapidfem._native.Geometry``): a list of solids and
sheets, realised into rapidmesh whenever a selection or a mesh needs it, and
the materials and physics placed on them. Solids overlap by priority, a
solid added later carves its region out of the earlier ones, and the mesh is
always conformal, so there is no separate boolean fragment step. Faces are
named by their origin ``(object, role)``, the role of the surface in its
primitive (a box: -z, +z, -y, +y, -x, +x; -1 for a sheet), which survives
later changes of the scene.

This module is the Python face of that scene: :class:`Geometry`,
:class:`GeoObject` and the selectable :class:`EntityCollection`, handles on
the native state.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import _native
from .materials import Material
from ._geometry_import import _ImportMixin
from ._geometry_primitives import _PrimitivesMixin

MeshStats = _native.MeshStats


class _Entity:
    """A selectable piece of the scene, a handle on its native state:

    - a solid (``dim == 3``), ``key`` its object id;
    - a face (``dim == 2``), ``key`` its selection ``(object, role, side,
      across)``: role -1 is a sheet, ``side`` the solid the face was selected
      through (-1 for none) and ``across`` what lies on the other side (an
      object, -2 for outside, -1 for anything); a face split by a later solid
      keeps its origin on every piece, these two single the pieces out;
    - an edge of a solid (``dim == 1``), ``key`` ``(object, (role_a, role_b))``;
    - a named group of a loaded mesh, ``key`` its name.

    ``material``, ``name`` and ``maxh`` are the attributes the physics and
    mesh layers read; they and the geometric properties (``cog``, ``bbox``,
    ``area``) live on the native side.
    """

    __slots__ = ("_geometry", "dim", "key")

    def __init__(self, geometry: "Geometry", dim: int, key):
        self._geometry = geometry
        self.dim = dim
        self.key = key

    def __eq__(self, other) -> bool:
        return (isinstance(other, _Entity) and other._geometry is self._geometry
                and other.key == self.key)

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"_Entity({self.key!r})"

    @property
    def material(self):
        return self._geometry._native.material(self.key)

    @material.setter
    def material(self, value) -> None:
        self._geometry._set_material([self], value)

    @property
    def name(self) -> str | None:
        return self._geometry._native.name(self.key)

    @name.setter
    def name(self, value: str | None) -> None:
        self._geometry._native.set_name(self.key, value)

    @property
    def maxh(self) -> float | None:
        return self._geometry._native.maxh(self.key)

    @maxh.setter
    def maxh(self, value: float | None) -> None:
        self._geometry._native.set_maxh([self.key], value)

    @property
    def cog(self) -> tuple[float, float, float]:
        return tuple(self._geometry._native.extents([self.key])[0][0])

    @property
    def area(self) -> float:
        return self._geometry._native.extents([self.key])[0][1]

    @property
    def bbox(self) -> tuple[float, ...]:
        return tuple(self._geometry._native.extents([self.key])[0][2])


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

    def _keys(self) -> list:
        return [e.key for e in self._entities]

    def _pick(self, positions) -> "EntityCollection":
        return EntityCollection(self._geometry, [self._entities[i] for i in positions])

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
        return self._pick(self._geometry._native.select_extreme(self._keys(), axis, False))

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
        return self._pick(self._geometry._native.select_extreme(self._keys(), axis, True))

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
        extents = self._geometry._native.extents(self._keys())
        return self._pick([i for i, (c, _, b) in enumerate(extents)
                           if predicate(tuple(c), tuple(b))])

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
        self._geometry._native.set_maxh(self._keys(), value)

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
        self._geometry._set_material(self._entities, value)

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
        return self._pick(self._geometry._native.select_unassigned(self._keys()))

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
        return self._pick(self._geometry._native.select_on_box(self._keys(), False))

    @property
    def hull(self) -> "EntityCollection":
        """axis-aligned faces lying on *this collection's own* bounding box

        Like :attr:`outer`, but the reference box is the bounding box of
        the entities in *this* collection, not the whole model. A
        face is kept iff one of its axes is degenerate and its coordinate
        along that axis matches this collection's extremum.

        This is what you want for a closed surface around an object that
        is itself wrapped by something else. ``air.faces.outer`` returns
        nothing once ``air`` is surrounded by PML on every side (none of
        air's faces touch the *model* bbox any more, they are all interior
        air/PML interfaces); ``air.faces.hull`` returns those six interface
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
        return self._pick(self._geometry._native.select_on_box(self._keys(), True))


FaceCollection = EntityCollection
EdgeCollection = EntityCollection


class GeoObject:
    """A solid or sheet of a :class:`Geometry`, as returned by the primitive
    methods (:meth:`Geometry.box`, :meth:`Geometry.xy_plate`, ...).

    ``faces`` selects its bounding faces (interfaces with other solids
    included), ``edges`` its edges; ``material``, ``maxh`` and ``name``
    are read by the mesh and physics layers.
    """

    def __init__(self, geometry: "Geometry", obj_id: int):
        self._geometry = geometry
        self._id = obj_id

    def __eq__(self, other) -> bool:
        return (isinstance(other, GeoObject) and other._geometry is self._geometry
                and other._id == self._id)

    def __hash__(self) -> int:
        return hash(self._id)

    def __repr__(self) -> str:
        kind = "sheet" if self.dim == 2 else "solid"
        return f"GeoObject({kind} {self._id})"

    @property
    def dim(self) -> int:
        return 2 if self._geometry._native.is_sheet(self._id) else 3

    @property
    def _entity(self) -> _Entity:
        """The object as a handle: a sheet is a face, a solid an object."""
        if self.dim == 2:
            return _Entity(self._geometry, 2, (self._id, -1, -1, -1))
        return _Entity(self._geometry, 3, self._id)

    @property
    def faces(self) -> EntityCollection:
        g = self._geometry
        if self.dim == 2:
            return EntityCollection(g, [self._entity])
        return EntityCollection(g, [g._face(k) for k in g._native.faces_of(self._id)])

    @property
    def edges(self) -> EntityCollection:
        g = self._geometry
        if self.dim == 2:
            return EntityCollection(g, [])
        return EntityCollection(g, [_Entity(g, 1, (self._id, roles))
                                    for roles, _ in g._native.edges_of(self._id)])

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


class Geometry(_PrimitivesMixin, _ImportMixin):
    """Top-level geometry builder on the native rapidmesh scene.

    Build with primitive factory methods (:meth:`box`, :meth:`cylinder`,
    :meth:`xy_plate`, ...) each of which returns a :class:`GeoObject`.
    Attach physics via the object-API constructors
    (:class:`rapidfem.RectWaveguidePort`, :class:`rapidfem.PEC`, ...)
    pointing at faces or volumes. When the description is complete,
    call :meth:`mesh` and feed the geometry to a
    :class:`rapidfem.ProblemFD` for analysis.

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
    grading : bool
        grade element sizes from fine features into the bulk
    name : str, optional
        model name (for diagnostic / log output)
    """

    def __init__(self, *, maxh: float | None = None, grading: bool = True,
                 name: str = "rapidfem"):
        self._native = _native.Geometry(maxh, grading=bool(grading))
        self.name = name
        # the physics objects, in the order (and by the index) the native
        # setup holds them
        self._physics: list = []
        self._fem_mesh = None
        self.mesh_stats: MeshStats | None = None

    @classmethod
    def _adopt(cls, native, name: str = "rapidfem") -> "Geometry":
        """The geometry around a native scene a native builder made."""
        g = cls.__new__(cls)
        g._native, g.name = native, name
        g._physics, g._fem_mesh, g.mesh_stats = [], None, None
        g._sync_physics()
        return g

    def _sync_physics(self) -> None:
        """The Python objects of the physics a native builder placed."""
        from . import physics
        for kind, params, keys in self._native.physics_since(len(self._physics)):
            p = object.__new__(getattr(physics, kind))
            p.__dict__.update({k: tuple(v) if isinstance(v, list) else v
                               for k, v in params.items()})
            p._entities = [self._face(k) if isinstance(k, tuple) else _Entity(self, 3, k)
                           for k in keys]
            p._geometry, p._index = self, len(self._physics)
            self._physics.append(p)

    def _objects(self, ids) -> list[GeoObject]:
        return [GeoObject(self, i) for i in ids]

    @staticmethod
    def from_gds(path: str, stack, top_cell: str | None = None,
                 bbox: tuple[float, float, float, float] | None = None,
                 merge: bool = True, thin_conductors: bool = False) -> "Geometry":
        """Load a GDSII layout and extrude every polygon on a stack layer.

        Each polygon on a (gds, datatype) of ``stack`` becomes a prism at the
        layer's z with the layer's thickness, named after the layer, so all of
        a layer can be selected at once. Cell references are resolved.

        Args:
            path: Path to the .gds file.
            stack: A `rapidfem.rfic.Stack` mapping (gds, datatype) to layers.
            top_cell: Cell name to extrude. ``None`` picks the unique
                top-level cell.
            bbox: Optional (xmin, ymin, xmax, ymax) crop box in meters;
                polygons outside are skipped.
            merge: Fuse co-layer polygons into one conductor region, so
                touching traces carry no internal faces.
            thin_conductors: Metal layers become sheets at the layer's
                bottom z (thin-conductor approximation, t << w).
        """
        native, cell = _native.rfic_from_gds(
            str(path), stack, top_cell=top_cell, bbox=bbox, merge=merge,
            thin_conductors=thin_conductors)
        return Geometry._adopt(native, name=cell or "gds_import")

    @property
    def size_scale(self) -> float:
        """factor on every target size of the mesh

        Multiplies the global ``maxh``, every object, material and face size
        and the ``maxh`` a :meth:`mesh` call passes: above 1 coarsens the whole
        mesh, below 1 refines it, the size relations of the model stay. The
        knob for a quick preview mesh or a convergence study.
        """
        return self._native.size_scale

    @size_scale.setter
    def size_scale(self, value: float) -> None:
        self._native.size_scale = float(value)

    @property
    def objects(self) -> list[GeoObject]:
        """every object of the scene, in the order they were added"""
        return [GeoObject(self, i) for i in self._native.objects()]

    def _wrap(self, obj_id: int, material=None) -> GeoObject:
        obj = GeoObject(self, obj_id)
        if material is not None:
            obj.material = material
        return obj

    def _face(self, key) -> _Entity:
        return _Entity(self, 2, tuple(key))

    def _set_material(self, entities, value) -> None:
        if value is not None and not isinstance(value, Material):
            raise TypeError(f"material must be a rapidfem Material (rf.Air(), "
                            f"rf.Dielectric(...), ...), got {type(value).__name__}")
        self._native.set_material([e.key for e in entities], value)

    # -- solids from faces ----------------------------------------------------

    def extrude(self, face: GeoObject, height: float,
                axis: tuple[float, float, float] = (0, 0, 1),
                *,
                material=None,
                maxh: float | None = None) -> GeoObject:
        """extrude a 2-D face along ``axis * height`` into a 3-D volume

        The source ``face`` becomes the bottom cap of the new volume, which
        keeps the object's identity (so names and mesh sizes set on it
        survive).


        Example
        -------
        A 35 um copper trace from a polygon footprint:

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
        self._native.extrude(face._id, height, axis, maxh)
        if material is not None:
            face.material = material
        return face

    def loft(self, face_a: GeoObject, face_b: GeoObject,
             *,
             material=None,
             maxh: float | None = None) -> GeoObject:
        """loft a volume between two coplanar / parallel 2-D faces

        Linearly interpolates the perimeter of ``face_a`` onto the
        perimeter of ``face_b``. Both faces must have the same number
        of edges in their outer boundary (a 4-edge rectangle lofts to
        a 4-edge rectangle, producing a frustum with 4 trapezoidal
        sides). The side surfaces are ruled (flat between the two
        outlines), the shape of pyramidal / frustum-style horns.


        Note
        ----
        The two profile sheets are consumed: the loft's own end faces
        replace them.


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
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override

        Returns
        -------
        GeoObject
            new volume
        """
        return self._wrap(self._native.loft(face_a._id, face_b._id, maxh), material)

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
        volume. The profile sheet is consumed.


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
        oid = self._native.revolve(face._id, axis_point, axis_dir, angle, maxh)
        return self._wrap(oid, material)

    # -- booleans -------------------------------------------------------------

    def fragment(self, target: GeoObject, *tools: GeoObject) -> None:
        """make ``target`` and ``tools`` conformal, the tools on top

        The scene is always assembled conformally (every interface is shared
        by the mesh on both sides); where solids overlap, the one on top owns
        the overlap. ``fragment`` puts the ``tools`` on top of every object,
        in the given order (the last tool wins), which is the usual intent of
        fragmenting inner objects with their surrounding volume; ``target``
        itself is not changed.


        Parameters
        ----------
        target : GeoObject
            the surrounding object, for readability only
        *tools : GeoObject
            objects to put on top, in order
        """
        self._native.bring_to_front([t._id for t in tools])

    def cut(self, target: GeoObject, *tools: GeoObject) -> None:
        """subtract ``tools``, leaving holes

        Solids: each tool becomes a void: it is cut out of every solid added
        before it (``target`` among them) and its walls become boundary faces
        that physics can address (``tool`` stays usable as a selection handle,
        its ``faces`` are the walls). Boundary faces without physics are PEC.

        Sheets: ``target`` and the tools must lie in one plane; the target
        becomes its outline minus the tools (a slot in a ground plane), the
        tools are used up. Discs take part as 96-sided polygons.


        Parameters
        ----------
        target : GeoObject
            object to subtract from
        *tools : GeoObject
            objects to subtract
        """
        ids = [t._id for t in tools]
        if target.dim == 2:
            self._native.sheet_boolean("difference", target._id, ids)
        else:
            self._native.make_void(ids)

    def fuse(self, target: GeoObject, *tools: GeoObject) -> None:
        """boolean union ``target`` and ``tools``

        Merges the operands into a single connected body assigned back
        to ``target``. Sheets must lie in one plane; their union may fall
        apart into several pieces, which stay one object, and the tools are
        used up.


        Note
        ----
        Face names on the operands are **not** preserved (faces merge
        and centroids shift). Top-level volume names survive, but set
        face names AFTER ``fuse``, or use
        :meth:`fragment` if you need the interfaces themselves to
        survive as named entities.


        Parameters
        ----------
        target : GeoObject
            first operand (survives as the merged object)
        *tools : GeoObject
            operands to merge in
        """
        ids = [t._id for t in tools]
        if target.dim == 2:
            self._native.sheet_boolean("union", target._id, ids)
        else:
            self._native.fuse([target._id, *ids])

    def intersect(self, target: GeoObject, *tools: GeoObject) -> None:
        """boolean intersect ``target`` with ``tools``

        Carves the intersection of ``target`` with every member of
        ``tools`` and assigns it back to ``target``. The tools are
        consumed by the operation (and lose their material). Sheets
        intersect in their common plane.


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
        ids = [t._id for t in tools]
        if target.dim == 2:
            self._native.sheet_boolean("intersection", target._id, ids)
        else:
            self._native.intersect(target._id, ids)

    # -- transforms -----------------------------------------------------------

    def translate(self, obj: GeoObject,
                  dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        """move ``obj`` (and all its child faces / edges) in place by ``(dx, dy, dz)``

        Like :meth:`rotate`, every face keeps its identity through the
        transform; only the geometric attributes (COG, bbox) change, so
        selections and names made before keep resolving to the moved
        faces.


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
        self._native.translate(obj._id, [dx, dy, dz])

    def rotate(self, obj: GeoObject, angle: float,
               axis: tuple[float, float, float] = (0, 0, 1),
               center: tuple[float, float, float] = (0, 0, 0)) -> None:
        """rotate ``obj`` (and all its child faces / edges) in place

        Every face keeps its identity through the transform; only the
        geometric attributes (COG, bbox) change. Named selectors keep
        working, the resolver sees the new positions.


        Example
        -------
        Rotate a horn 30 degrees around y:

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
        self._native.rotate(obj._id, angle, axis, center)

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
        self._native.stretch(obj._id, [fx, fy, fz], center)

    def mirror(self, obj: GeoObject,
               normal: tuple[float, float, float] = (1, 0, 0),
               point: tuple[float, float, float] = (0, 0, 0)) -> None:
        """reflect ``obj`` in place across the plane through ``point`` with ``normal``

        Useful for building symmetric structures (one half plus its
        mirror image) without re-deriving coordinates. The reflection is
        in place: faces keep their identity, only COG/bbox change, so
        named selectors keep working.


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
        self._native.mirror(obj._id, normal, point)

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
        new = self._wrap(self._native.copy(obj._id), material)
        if maxh is not None:
            new.maxh = maxh
        return new

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
        if count < 1:
            raise ValueError(f"array: count must be >= 1, got {count}")
        if (spacing is None) == (rotation is None):
            raise ValueError("array: pass exactly one of spacing or rotation")
        out = [obj]
        for k in range(1, count):
            c = self.copy(obj)
            if spacing is not None:
                self.translate(c, *(k * float(s) for s in spacing))
            else:
                self.rotate(c, k * float(rotation), axis=axis, center=center)
            out.append(c)
        return out

    # -- edge features --------------------------------------------------------

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
                if getattr(e, "dim", None) != 1 or e.key[0] != obj._id:
                    raise ValueError(f"{e!r} is not an edge of {obj!r}")
                picks.append(e.key[1])
        self._native.cut_edges(obj._id, float(size), fillet, picks)

    def _hollow(self, name: str) -> list[tuple["EntityCollection", float]]:
        """Turn every solid named ``name`` into a void: its walls become
        boundary faces that carry a surface condition. Returns one
        ``(walls, 2V/S)`` pair for the group, with the volume-to-surface
        thickness in metres (for a long trace of width w and thickness t,
        ``w t / (w + t)``: the thickness at which a two-sided surface
        impedance on every wall reproduces the DC resistance ``1/(sigma w t)``).
        """
        walls = self._native.hollow(name)
        if walls is None:
            return []
        keys, thickness = walls
        return [(EntityCollection(self, [self._face(k) for k in keys]), thickness)]

    # -- sizing ---------------------------------------------------------------

    def auto_refine_features(
        self,
        base_maxh: float,
        resolution: int = 3,
        min_maxh: float | None = None,
    ) -> dict[str, float]:
        """auto-assign a per-object ``maxh`` for any volume or sheet
        narrower than ``base_maxh``

        Walks every volume and sheet in the geometry. For each, computes
        the smallest non-zero bbox extent (the "feature size": a volume's
        thinnest axis, a sheet's narrowest in-plane width). If that is
        smaller than ``base_maxh`` *and* the user hasn't already set the
        object's ``maxh`` explicitly, sets

        .. math::

            \\mathrm{vol.maxh} = \\max\\!\\left(
                \\frac{d_{\\min}}{\\mathrm{resolution}},
                \\mathrm{min\\_maxh}
            \\right)

        so the feature is resolved with at least ``resolution`` elements
        across it.


        Note
        ----
        Idempotent, only writes ``maxh`` when it's currently ``None``,
        so explicit per-volume sizes (set via ``g.box(..., maxh=...)``,
        ``obj.maxh = ...`` or the material's ``maxh``) always win.


        Example
        -------
        Resolve a 0.5 mm thin substrate against a 12 mm global cap:

        .. code-block:: python

            g.auto_refine_features(base_maxh=12e-3, resolution=3)


        Parameters
        ----------
        base_maxh : float
            reference size, objects wider than this in all directions
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
            map ``{descriptor: assigned_maxh}`` for the objects touched
            (the object's ``name`` if set, else ``"vol@(cx,cy,cz)mm"`` or
            ``"sheet@(cx,cy,cz)mm"``)
        """
        return dict(self._native.auto_refine(base_maxh, resolution, min_maxh))

    def refine_near_points(self, points, h: float) -> None:
        """register a local mesh-size refinement around a point cloud

        Every point becomes a size source of target ``h``; the size grows
        from there along the grading, so the element size reaches the
        surrounding target over a distance that follows from the grading.


        Parameters
        ----------
        points : array_like, shape (N, 3)
            points in metres
        h : float
            element size at the points
        """
        self._native.add_size_points(np.asarray(points, dtype=float).reshape(-1, 3).tolist(), h)

    # -- meshing --------------------------------------------------------------

    def save_mesh(self, path: str) -> str:
        """Write the last generated mesh to ``path`` as gmsh ``.msh`` v4.

        The file carries every physical group exactly as the FEM solver sees
        them (materials, PEC/port/ABC targets), so it round-trips through
        :meth:`load` and is directly consumable by external gmsh-format
        solvers such as Palace, same mesh, different solver.

        Requires a prior :meth:`mesh` call; raises otherwise. A region
        carries one volume group: a volume group over regions an earlier
        group (a material) already holds is left out, with a warning.
        """
        dropped = self._native.save_msh(str(path))
        if dropped:
            import warnings
            warnings.warn(f"save_mesh: volume groups without regions of their own "
                          f"are not in the file: {', '.join(dropped)}", stacklevel=2)
        return str(path)

    def mesh(
        self,
        maxh: float | None = None,
        *,
        cells_across: float | None = None,
        target_elements: int | None = None,
    ):
        """tetrahedralize the scene and build the solver mesh

        Meshes the scene with rapidmesh (bottom-up: edges, then every face
        on its true surface, then every region by its constrained Delaunay
        tetrahedralization, with exact predicates), honouring the global
        ``maxh``, per-object and per-face sizes and the size points of
        :meth:`refine_near_points`, then tags every material and physics
        object and hands the mesh to the solvers in memory. A loaded mesh
        is taken as it is. Fills :attr:`mesh_stats`.


        Parameters
        ----------
        maxh : float, optional
            global size override for this call
        cells_across : float, optional
            elements across the thickness of every region, so a thin layer
            gets proper tets through it; off by default (a stack of layers far
            thinner than the size then takes flat tets through each layer)
        target_elements : int, optional
            tet budget: the global size is scaled to land near it


        Returns
        -------
        MeshStats
            size and quality of the mesh
        """
        self._fem_mesh, self.mesh_stats = self._native.mesh(maxh, cells_across, target_elements)
        return self.mesh_stats


__all__ = [
    "Geometry", "GeoObject", "EntityCollection", "FaceCollection",
    "EdgeCollection", "MeshStats",
]
