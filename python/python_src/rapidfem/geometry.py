#########################################################################################
##
##                              GEOMETRY + MESHING
##                                 (geometry.py)
##
#########################################################################################

"""Geometry + meshing builder for rapidfem, NGSolve-style.

Wraps gmsh's OpenCASCADE kernel with tracked entities that survive
boolean ops, plus a per-entity registry that carries materials, mesh
sizes, and physics targets through to the FEM solver. The user-facing
flow is

.. code-block:: python

    g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
    sub = g.box(60e-3, 60e-3, 1.6e-3, position=(-30e-3, -30e-3, 0),
                material=rf.Dielectric(er=4.4))
    patch = g.xy_plate(38e-3, 29e-3, position=(-19e-3, -14.5e-3, 1.6e-3))

    g.fragment(sub, patch)         # bool op, names + materials survive
    g.mesh()                       # generate tet mesh


Note
----
Tracking strategy: each named entity stores (cog, bbox, dim) at
registration time. After every boolean op, the geometry walks its
registry and re-resolves each entity by matching (cog, bbox) against
current gmsh entities. COG-only is ambiguous for coplanar overlapping
faces (e.g. annulus + sub-region after embedding a plate); bbox
disambiguates. ``fuse`` is supported but warned about, face merging
shifts COGs, so names cannot survive.
"""

# IMPORTS ===============================================================================

from __future__ import annotations

import math
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable

import gmsh
import numpy as np

from ._geometry_gds import _GdsMixin
from ._geometry_primitives import _PrimitivesMixin
from ._geometry_import import _ImportMixin


# TOLERANCES ============================================================================

# After a gmsh boolean op entity tags get renumbered, so entities are
# re-identified by geometry. 1e-9 m (1 nm) is the slack for those matches:
# well below any realistic mesh feature (microns and up) yet far above the
# float64 round-off gmsh introduces in O(1 m)..O(1 mm) coordinates, so it
# never merges distinct features nor splits one that the kernel nudged.
_COG_TOL = 1e-9   # distance tol for matching center-of-mass (m)
_BBOX_TOL = 1e-9  # tol for matching bounding-box corners (m)
# Tolerance for matching a face against a bounding-box extremum. Coarser than
# _BBOX_TOL on purpose: gmsh's getBoundingBox inflates the box by ~1e-7 m on
# each side, so "zero-extent" face coordinates come back as ±1e-7. Comparing
# against an extremum (not two getBoundingBox results, where the fluff cancels)
# must absorb that inflation, hence 1e-6 m.
_HULL_TOL = 1e-6
# gmsh.getBoundingBox(-1, -1) returns ±1e106 when the model holds a degenerate
# or unbounded entity; treat anything past this as "no usable bbox".
_UNBOUNDED = 1e10


# INTERNAL ENTITY TRACKING ==============================================================




@dataclass
class _Entity:
    """A tracked gmsh entity with stable identity through boolean ops.

    `tag` is the *current* gmsh tag, may be updated by Geometry._reresolve()
    after fragment/cut. The (cog, bbox, dim) triple is the stable identity used
    for re-resolution.

    ``material`` can be either a :class:`rapidfem.Material` instance (object-API
    path, set via ``g.box(..., material=...)``) or a string (legacy, set via
    ``obj.material = "fr4"``, used by rfic.Stack et al.).
    """
    dim: int
    tag: int
    cog: tuple[float, float, float]
    bbox: tuple[float, float, float, float, float, float]
    name: str | None = None
    material: object = None    # rapidfem.Material | str | None
    maxh: float | None = None
    _geometry: object = None   # back-ref to Geometry, set by registration
    _discrete: bool = False    # True for healed-STL volumes (geo/discrete kernel,
                               # not OCC): they live as mesh nodes, so OCC boolean
                               # ops and post-import transforms don't apply

    @staticmethod
    def from_dimtag(dim: int, tag: int) -> "_Entity":
        bbox = tuple(gmsh.model.getBoundingBox(dim, tag))
        return _Entity(dim=dim, tag=tag, cog=_cog(dim, tag, bbox), bbox=bbox)


def _cog(dim: int, tag: int, bbox: tuple | None = None) -> tuple:
    """Centre of mass of an entity, with a discrete-geometry fallback.

    OCC entities (primitives, imported STEP/IGES/BREP) expose an exact
    ``occ.getCenterOfMass``. Discrete entities (healed STL, pre-built ``.msh``
    groups) live outside the OCC kernel, so that call raises; we fall back to
    the bounding-box centre, which is all the ``.min``/``.max``/``.where``
    face selectors actually consult.
    """
    try:
        return tuple(gmsh.model.occ.getCenterOfMass(dim, tag))
    except Exception:
        if bbox is None:
            bbox = tuple(gmsh.model.getBoundingBox(dim, tag))
        return ((bbox[0] + bbox[3]) / 2,
                (bbox[1] + bbox[4]) / 2,
                (bbox[2] + bbox[5]) / 2)


def _bbox_match(a: tuple, b: tuple, tol: float) -> bool:
    return all(abs(a[i] - b[i]) < tol for i in range(6))


def _resolve_entity(target: _Entity) -> int | None:
    """Find a gmsh entity in `target.dim` matching the stored (bbox, cog).

    Strategy: bbox is the primary identity (stable through fragment, even when
    sub-volumes are carved out, the outer bbox doesn't change). COG is a
    secondary discriminator with a tolerance that scales with the bbox extent.
    This handles two real failure modes:
      1. COG drift after carving a sub-volume out of a larger one (e.g. air
         box minus substrate), bbox stays put, COG shifts by mass redistribution.
      2. Coplanar overlapping faces (annulus + sub-region after fragment),
         both have the same COG, but different bboxes.
    """
    # bbox extent diagonal, sets an "internal scale" for COG drift tolerance.
    extent = math.sqrt(sum((target.bbox[3 + i] - target.bbox[i]) ** 2 for i in range(3)))
    cog_tol = max(_COG_TOL, 0.01 * extent)  # 1% of diagonal, or absolute floor

    candidates = []
    for d, t in gmsh.model.getEntities(target.dim):
        if d != target.dim:
            continue
        bbox = tuple(gmsh.model.getBoundingBox(d, t))
        if not _bbox_match(bbox, target.bbox, _BBOX_TOL):
            continue
        cog = tuple(gmsh.model.occ.getCenterOfMass(d, t))
        if math.dist(cog, target.cog) < cog_tol:
            candidates.append(t)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Tie-break by smallest COG distance.
        candidates.sort(key=lambda t: math.dist(
            tuple(gmsh.model.occ.getCenterOfMass(target.dim, t)), target.cog))
        return candidates[0]
    return None


# ENTITY COLLECTION =====================================================================

class EntityCollection:
    """A collection of sub-entities (faces or edges) with selectors and
    bulk attribute writes.

    Returned by ``obj.faces`` and ``obj.edges`` on a :class:`GeoObject`,
    and by selector methods (:meth:`min`, :meth:`max`, :meth:`where`,
    :attr:`unassigned`, :attr:`outer`) on existing collections. Every
    selector returns a *new* collection so chains compose:

    .. code-block:: python

        port_face   = air.faces.min(axis="z")
        top_corner  = air.faces.where(lambda c, b: c[2] > 0.5).max(axis="x")
        loose_ends  = air.faces.outer.unassigned


    Note
    ----
    Bulk attribute writes (``coll.name = "..."`` / ``coll.maxh = ...``)
    apply to every member. The old-API pattern of assigning names this
    way still works (used by ``rfic.Stack`` and the ``from_gds``
    importer); the object-API path goes through physics constructors
    (``rf.PEC(coll)`` / ``rf.RectWaveguidePort(coll)``) and does not
    require names.


    Example
    -------
    .. code-block:: python

        # Pick the bottom face of an air box and drive it as a port
        rf.RectWaveguidePort(air.faces.min(axis="z"))

        # All un-targeted outer faces become PEC
        rf.PEC(*air.faces.outer.unassigned)


    Parameters
    ----------
    geometry : Geometry
        owning geometry (for back-references to the physics registry)
    entities : list[_Entity]
        the underlying tracked entity records
    """

    def __init__(self, geometry: "Geometry", entities: list[_Entity]):
        self._geometry = geometry
        self._entities = entities

    def __iter__(self):
        return iter(self._entities)

    def __len__(self):
        return len(self._entities)

    # Selectors
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
        ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        if not self._entities:
            return EntityCollection(self._geometry, [])
        m = min(e.cog[ax] for e in self._entities)
        kept = [e for e in self._entities if abs(e.cog[ax] - m) < _COG_TOL]
        return EntityCollection(self._geometry, kept)

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
        ax = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        if not self._entities:
            return EntityCollection(self._geometry, [])
        m = max(e.cog[ax] for e in self._entities)
        kept = [e for e in self._entities if abs(e.cog[ax] - m) < _COG_TOL]
        return EntityCollection(self._geometry, kept)

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
        kept = [e for e in self._entities if predicate(e.cog, e.bbox)]
        return EntityCollection(self._geometry, kept)

    # Bulk attribute setters (NGSolve idiom: `coll.name = "..."` writes to all)
    @property
    def name(self) -> str | None:
        names = {e.name for e in self._entities if e.name is not None}
        if len(names) == 1:
            return names.pop()
        return None

    @name.setter
    def name(self, value: str) -> None:
        for e in self._entities:
            e.name = value

    @property
    def maxh(self) -> float | None:
        vals = {e.maxh for e in self._entities if e.maxh is not None}
        if len(vals) == 1:
            return vals.pop()
        return None

    @maxh.setter
    def maxh(self, value: float) -> None:
        for e in self._entities:
            e.maxh = value

    @property
    def material(self):
        mats = {id(e.material): e.material
                for e in self._entities if e.material is not None}
        if len(mats) == 1:
            return next(iter(mats.values()))
        return None

    @material.setter
    def material(self, value) -> None:
        """Assign one material to every volume in the collection.

        The canonical way to give a material to an imported-mesh group, e.g.
        ``scene.group("substrate").material = rf.Dielectric(er=4.4)``. Applies
        to volume (dim 3) members; assigning to a face collection is a no-op for
        the FEM (faces carry boundary physics, not bulk materials).
        """
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
        targeted: set[int] = set()
        for phys in self._geometry._physics:
            for ent in getattr(phys, "_entities", ()):
                targeted.add(id(ent))
        kept = [e for e in self._entities if id(e) not in targeted]
        return EntityCollection(self._geometry, kept)

    @property
    def outer(self) -> "EntityCollection":
        """axis-aligned faces lying on the outer hull of the gmsh model

        A face is "outer" iff one of its axes is degenerate (zero
        extent, i.e. the face is planar and perpendicular to that
        axis) AND the face's coordinate along that axis matches the
        model bounding-box extremum within tolerance.

        Interior fragmented interfaces, which share the model's x/y
        bbox extents but sit at some inner z, are correctly excluded.
        Useful for tagging the external walls of a PML enclosure where
        the inner air↔PML interface must stay free.


        Note
        ----
        Uses a 1e-6 m tolerance to absorb gmsh's ``getBoundingBox``
        inflation (gmsh adds ~1e-7 m of fluff on each side, which is
        much larger than the 1e-9 m tolerance used elsewhere for
        cog/bbox identity matching).


        Example
        -------
        ABC on every external face of the air box:

        .. code-block:: python

            rf.ABC(*air.faces.outer)


        Returns
        -------
        EntityCollection
            entities on the model bounding box
        """
        try:
            bb = gmsh.model.getBoundingBox(-1, -1)
        except Exception:
            return EntityCollection(self._geometry, list(self._entities))
        xmin, ymin, zmin, xmax, ymax, zmax = bb
        # gmsh's getBoundingBox(-1, -1) returns ±1e+106 on any axis whenever
        # the model contains a degenerate or unbounded entity (we hit this
        # with the mom-cap 200-via cluster, fragment leaves slivers that
        # gmsh's bbox accumulator treats as infinite). When that happens the
        # naive extremum-match below filters every face out and the caller
        # gets an empty collection, which torpedoes any `rf.ABC(*air.faces
        # .outer, ...)` wiring downstream. Fall back to the union of every
        # tracked entity's bbox, those are real getBoundingBox results for
        # specific entities so they don't carry the infinity.
        def _is_finite_bbox(b):
            return all(abs(v) < _UNBOUNDED for v in b)
        if not _is_finite_bbox((xmin, ymin, zmin, xmax, ymax, zmax)):
            # gmsh.getBoundingBox(-1, -1) returns ±1e100 when the model has
            # degenerate entities (e.g. mom-cap's 200-via cluster leaves
            # fragment slivers gmsh treats as infinite). Substituting a
            # global union over all tracked entities doesn't help either,
            # those same slivers carry the infinite bbox, AND non-sliver
            # volumes like substrate/oxide/air sometimes end up with
            # different post-fragment footprints from each other.
            # Best fallback: pick extremes from THIS collection's faces
            # only. For `air.faces.outer`, that means we compare air's
            # outer faces against the bbox of air alone, exactly what the
            # caller intuitively expects.
            xs0, ys0, zs0, xs1, ys1, zs1 = (
                math.inf, math.inf, math.inf, -math.inf, -math.inf, -math.inf)
            for e in self._entities:
                if not _is_finite_bbox(e.bbox):
                    continue
                ex0, ey0, ez0, ex1, ey1, ez1 = e.bbox
                if ex0 < xs0: xs0 = ex0
                if ey0 < ys0: ys0 = ey0
                if ez0 < zs0: zs0 = ez0
                if ex1 > xs1: xs1 = ex1
                if ey1 > ys1: ys1 = ey1
                if ez1 > zs1: zs1 = ez1
            if math.isfinite(xs0):
                xmin, ymin, zmin = xs0, ys0, zs0
                xmax, ymax, zmax = xs1, ys1, zs1
        return self._faces_on_box(xmin, ymin, zmin, xmax, ymax, zmax)

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
        if not self._entities:
            return EntityCollection(self._geometry, [])
        xmin = ymin = zmin = math.inf
        xmax = ymax = zmax = -math.inf
        for e in self._entities:
            ex0, ey0, ez0, ex1, ey1, ez1 = e.bbox
            xmin, ymin, zmin = min(xmin, ex0), min(ymin, ey0), min(zmin, ez0)
            xmax, ymax, zmax = max(xmax, ex1), max(ymax, ey1), max(zmax, ez1)
        return self._faces_on_box(xmin, ymin, zmin, xmax, ymax, zmax)

    def _faces_on_box(self, xmin, ymin, zmin, xmax, ymax, zmax) -> "EntityCollection":
        """faces in this collection lying on the given axis-aligned box

        Shared core of :attr:`outer` (box = model bbox) and :attr:`hull`
        (box = this collection's own bbox). A face is kept iff one of its
        axes is degenerate (zero extent) and its coordinate along that
        axis matches that box's min or max within :data:`_HULL_TOL`.
        """
        kept = []
        for e in self._entities:
            ex0, ey0, ez0, ex1, ey1, ez1 = e.bbox
            on_box = (
                (abs(ex1 - ex0) < _HULL_TOL
                 and (abs(ex0 - xmin) < _HULL_TOL or abs(ex0 - xmax) < _HULL_TOL))
                or (abs(ey1 - ey0) < _HULL_TOL
                    and (abs(ey0 - ymin) < _HULL_TOL or abs(ey0 - ymax) < _HULL_TOL))
                or (abs(ez1 - ez0) < _HULL_TOL
                    and (abs(ez0 - zmin) < _HULL_TOL or abs(ez0 - zmax) < _HULL_TOL))
            )
            if on_box:
                kept.append(e)
        return EntityCollection(self._geometry, kept)


# Back-compat aliases (and clearer naming for users)
FaceCollection = EntityCollection
EdgeCollection = EntityCollection


# GEOMETRIC OBJECT ======================================================================

class GeoObject:
    """A primitive (volume or 2-D plate) in the geometry.

    The return type of every :class:`Geometry` factory method
    (:meth:`Geometry.box`, :meth:`Geometry.cylinder`, ...). Carries a
    reference to its owning geometry plus the tracked
    :class:`_Entity` record that survives boolean ops.


    Note
    ----
    The recommended path to attach materials and per-entity mesh
    sizes is the constructor kwarg form
    (``g.box(..., material=rf.Air(), maxh=2e-3)``). The
    ``obj.material = ...`` / ``obj.maxh = ...`` setters still work and
    are used by legacy paths like ``from_gds`` and ``rfic.Stack``, but
    new code should prefer the kwarg form.


    Example
    -------
    .. code-block:: python

        substrate = g.box(60e-3, 60e-3, 1.6e-3,
                          material=rf.Dielectric(er=4.4),
                          maxh=0.5e-3)
        ground_face = substrate.faces.min(axis="z")


    Attributes
    ----------
    name : str or None
        physical-group name applied at mesh time (legacy / GDS path);
        the object-API path attaches physics directly and does not
        require names
    material : rapidfem.Material or str or None
        volume material, a :class:`Material` instance under the new
        object API, or a string under the legacy ``rfic.Stack`` path
    maxh : float or None
        per-entity mesh size override in metres
    dim : int
        topological dimension (3 for volumes, 2 for plates)
    faces : EntityCollection
        bounding faces of a volume (or self, for a plate)
    edges : EntityCollection
        bounding edges of the entity
    """

    def __init__(self, geometry: "Geometry", entity: _Entity):
        self._geometry = geometry
        self._entity = entity
        # Note: `.faces` and `.edges` are computed on-demand (properties) so
        # they always reflect the CURRENT gmsh topology. After a boolean op
        # splits a face into pieces, accessing `obj.faces` re-discovers them
        # all. Names set previously persist via the geometry's entity registry
        # (matched by cog+bbox).

    def _discover_subentities(self, target_dim: int) -> list[_Entity]:
        """Re-query gmsh for the current sub-entities of dimension `target_dim`.
        Existing entries in `self._geometry._entities` matching by (cog, bbox)
        keep their name/material/maxh; new entries get fresh blank metadata."""
        if self._entity.dim == 3 and target_dim == 2:
            children = [(d, t) for d, t in
                        gmsh.model.getBoundary([(3, self._entity.tag)], oriented=False)
                        if d == 2]
        elif self._entity.dim == 3 and target_dim == 1:
            children = []
            seen: set[int] = set()
            for d, t in gmsh.model.getBoundary([(3, self._entity.tag)], oriented=False):
                if d != 2:
                    continue
                for d_e, t_e in gmsh.model.getBoundary([(2, t)], oriented=False):
                    if d_e == 1 and t_e not in seen:
                        seen.add(t_e)
                        children.append((d_e, t_e))
        elif self._entity.dim == 2 and target_dim == 1:
            children = [(d, t) for d, t in
                        gmsh.model.getBoundary([(2, self._entity.tag)], oriented=False)
                        if d == 1]
        else:
            children = []

        # Build entries, reusing existing _Entity records if cog+bbox matches
        out: list[_Entity] = []
        for d, t in children:
            bbox = tuple(gmsh.model.getBoundingBox(d, t))
            cog = _cog(d, t, bbox)
            existing = self._find_or_register_entity(d, t, cog, bbox)
            out.append(existing)
        return out

    def _find_or_register_entity(self, dim, tag, cog, bbox) -> _Entity:
        """Look up an existing _Entity in the geometry registry by (cog, bbox);
        register a new one if not found. Updates the tag to current."""
        for ent in self._geometry._entities:
            if ent.dim != dim:
                continue
            if math.dist(ent.cog, cog) < _COG_TOL and _bbox_match(ent.bbox, bbox, _BBOX_TOL):
                ent.tag = tag       # refresh tag (may have changed after fragment)
                return ent
        new_ent = _Entity(dim=dim, tag=tag, cog=cog, bbox=bbox)
        new_ent._geometry = self._geometry
        self._geometry._entities.append(new_ent)
        return new_ent

    @property
    def faces(self) -> EntityCollection:
        return EntityCollection(self._geometry, self._discover_subentities(2))

    @property
    def edges(self) -> EntityCollection:
        return EntityCollection(self._geometry, self._discover_subentities(1))

    @property
    def name(self) -> str | None:
        return self._entity.name

    @name.setter
    def name(self, value: str) -> None:
        self._entity.name = value

    @property
    def material(self) -> str | None:
        return self._entity.material

    @material.setter
    def material(self, value: str) -> None:
        self._entity.material = value

    @property
    def maxh(self) -> float | None:
        return self._entity.maxh

    @maxh.setter
    def maxh(self, value: float) -> None:
        self._entity.maxh = value

    @property
    def dim(self) -> int:
        return self._entity.dim


# GEOMETRY ==============================================================================

class Geometry(_GdsMixin, _PrimitivesMixin, _ImportMixin):
    """Top-level geometry builder. Owns a gmsh OCC model for its lifetime.

    Build with primitive factory methods (:meth:`box`, :meth:`cylinder`,
    :meth:`xy_plate`, ...) each of which returns a :class:`GeoObject`.
    Attach physics via the object-API constructors
    (:class:`rapidfem.RectWaveguidePort`, :class:`rapidfem.PEC`, ...)
    pointing at faces or volumes. When the description is complete,
    call :meth:`mesh` and feed the geometry to a
    :class:`rapidfem.Problem` for analysis.


    Note
    ----
    The Geometry holds the global gmsh OCC model exclusive while it
    lives. Constructing two ``Geometry`` instances back-to-back is
    fine (the second wipes the first's model state), but they cannot
    coexist.


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
    name : str, optional
        gmsh model name (for diagnostic / log output)


    Attributes
    ----------
    _objects : list[GeoObject]
        every primitive ever added via this builder
    _entities : list[_Entity]
        tracked sub-entity registry (volumes + faces + edges)
    _physics : list
        object-API physics registry (ports + BCs + PML)
    _material_tags : dict[int, int]
        ``id(Material) → physical-group tag`` map populated by
        :meth:`mesh`
    _physics_tags : dict[int, int]
        ``id(physics_obj) → physical-group tag`` map populated by
        :meth:`mesh`
    _last_mesh : tuple[bytes, dict] or None
        ``(mesh_bytes, name_to_tag)`` cache populated by :meth:`mesh`
    """

    def __init__(self, *, maxh: float | None = None, scale: float = 1.0,
                 grading: bool = True,
                 name: str = "rapidfem"):
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        # Working scale: gmsh internally works in (user_meters / scale) units.
        # Default scale=1.0 means gmsh sees meters directly. Set scale=1e-6
        # for µm-scale RFIC geometries so a 1 µm feature becomes 1.0 in gmsh
        # units, way above gmsh's default 1e-7 tolerance, so booleans and
        # meshing on dense layouts (interleaved combs, via clusters, …) stay
        # robust. Just before meshing every entity is dilated back to user
        # units so the resulting mesh nodes land at the original meter
        # coordinates the FEM solver expects.
        self._scale = float(scale)
        # Default tolerance is 1e-7 m, comparable to a µm feature, which
        # makes boolean ops/mesh struggle at RFIC scale. Drop two orders so
        # the tolerance lives well below any feature the FEM cares about.
        gmsh.option.setNumber("Geometry.Tolerance", 1e-9)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-9)
        # Single-threaded OCC boolean operators. The multi-threaded path
        # intermittently deadlocks on geometrically dense models (the RFIC
        # layouts with many fragmented conductors), a known OpenCASCADE
        # flakiness; the booleans are not the meshing bottleneck anyway.
        gmsh.option.setNumber("Geometry.OCCParallel", 0)
        gmsh.model.add(name)
        self._objects: list[GeoObject] = []
        self._entities: list[_Entity] = []  # all named-or-trackable entities
        self._owns_gmsh = True  # we'll finalize on close
        self._maxh = maxh                   # global mesh size cap (USER units)
        # Soft size grading from boundary into the bulk. When True (default),
        # `Mesh.MeshSizeExtendFromBoundary` is enabled in `mesh()` so the fine
        # boundary sizes (e.g. a 0.5 mm substrate next to a 10 mm air cap)
        # ramp gradually into the interior instead of HXT placing the full
        # 10 mm tet right against the 0.5 mm interface. Costs a moderate tet
        # count increase (~15-30%) in the transition zone but kills the
        # "giant air tets visible in the field viz" artefact and produces a
        # better-conditioned solve. Pass `grading=False` to recover the old
        # behaviour for benchmarks or specific mesh-count budgets.
        self._grading = bool(grading)
        # Object-API state: physics registry + post-mesh tag maps.
        self._physics: list = []            # rapidfem.physics.* instances
        self._material_tags: dict[int, int] = {}  # id(Material) -> phys group tag
        self._physics_tags: dict[int, int] = {}   # id(physics_obj) -> phys group tag
        self._last_mesh = None              # (mesh_bytes, name_to_tag) after .mesh()
        # Refinement requests added via refine_near_points(). Each entry
        # is {points: (N, 3) np.ndarray, h: float, distance: float}.
        # Consumed in mesh() as extra Distance + Threshold background
        # fields, merged into the per-entity Min combiner.
        self._refinements: list[dict] = []
        # Substrate mode: "occ" (primitives + STEP/IGES/BREP/STL, the default)
        # or "mesh" (a pre-built .msh loaded via load(); no OCC model). Set by
        # _ImportMixin._load_mesh().
        self._mode = "occ"
        self._mesh_scene = None             # MeshScene when in mesh mode
        # True once a healed STL (discrete/geo body) is present: it cannot share
        # the OCC kernel with primitives, so further OCC geometry is rejected.
        self._has_discrete = False

    # ── Scale helpers ──────────────────────────────────────────────────────
    # Every coord that goes INTO gmsh is divided by self._scale; every coord
    # read BACK from gmsh stays in the scaled space (entity bbox/cog are
    # stored scaled). mesh() dilates everything back to user units before
    # tessellation so the .msh nodes are in real meters.
    def _s(self, v: float) -> float:
        return v / self._scale

    # ── GDS-driven extrusion ────────────────────────────────────────────────
    # from_gds / _plate_polygon / _extrude_polygon live in
    # _geometry_gds._GdsMixin (inherited) to keep this file navigable.

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._owns_gmsh and gmsh.isInitialized():
            gmsh.finalize()
            self._owns_gmsh = False

    # PRIMITIVES ───────────────────────────────────────────────────────────

    def _wrap_volume(self, tag: int, *,
                     material=None,
                     maxh: float | None = None) -> GeoObject:
        if self._mode == "mesh":
            self._require_occ_mode("build OCC geometry")
        self._reject_after_discrete("add a volume")
        gmsh.model.occ.synchronize()
        ent = _Entity.from_dimtag(3, tag)
        ent._geometry = self
        ent.material = material
        ent.maxh = maxh
        obj = GeoObject(self, ent)
        self._objects.append(obj)
        self._entities.append(ent)
        return obj

    def _wrap_face(self, tag: int, *,
                   maxh: float | None = None) -> GeoObject:
        if self._mode == "mesh":
            self._require_occ_mode("build OCC geometry")
        self._reject_after_discrete("add a face")
        gmsh.model.occ.synchronize()
        ent = _Entity.from_dimtag(2, tag)
        ent._geometry = self
        ent.maxh = maxh
        obj = GeoObject(self, ent)
        self._objects.append(obj)
        self._entities.append(ent)
        return obj


    # ── Extrude / revolve ───────────────────────────────────────────────────

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
        if face.dim != 2:
            raise ValueError(f"extrude expects a 2D face, got dim={face.dim}")
        dx, dy, dz = axis[0] * height, axis[1] * height, axis[2] * height
        s = self._s
        out = gmsh.model.occ.extrude([(face.dim, face._entity.tag)], s(dx), s(dy), s(dz))
        gmsh.model.occ.synchronize()
        vol_tag = next((t for d, t in out if d == 3), None)
        if vol_tag is None:
            raise RuntimeError("extrude produced no volume")
        return self._wrap_volume(vol_tag, material=material, maxh=maxh)

    def loft(self, face_a: GeoObject, face_b: GeoObject,
             ruled: bool = True,
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
        if face_a.dim != 2 or face_b.dim != 2:
            raise ValueError("loft expects two 2D faces")
        wire_a = self._face_outer_wire(face_a)
        wire_b = self._face_outer_wire(face_b)
        out = gmsh.model.occ.addThruSections(
            [wire_a, wire_b], makeSolid=True, makeRuled=ruled
        )
        gmsh.model.occ.synchronize()
        vol_tag = next((t for d, t in out if d == 3), None)
        if vol_tag is None:
            raise RuntimeError("loft produced no volume")
        return self._wrap_volume(vol_tag, material=material, maxh=maxh)

    def _face_outer_wire(self, face: GeoObject) -> int:
        """Return a wire tag for the outer boundary of ``face``.

        gmsh ``addThruSections`` expects wire tags; we build a fresh wire
        from the face's boundary edges so the call is self-contained.
        """
        bd = gmsh.model.getBoundary(
            [(face.dim, face._entity.tag)], oriented=False, recursive=False
        )
        edge_tags = [t for d, t in bd if d == 1]
        return gmsh.model.occ.addWire(edge_tags, checkClosed=True)

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
        if face.dim != 2:
            raise ValueError(f"revolve expects a 2D face, got dim={face.dim}")
        cx, cy, cz = axis_point
        ax, ay, az = axis_dir
        s = self._s
        out = gmsh.model.occ.revolve(
            [(face.dim, face._entity.tag)], s(cx), s(cy), s(cz), ax, ay, az, angle
        )
        gmsh.model.occ.synchronize()
        vol_tag = next((t for d, t in out if d == 3), None)
        if vol_tag is None:
            raise RuntimeError("revolve produced no volume")
        return self._wrap_volume(vol_tag, material=material, maxh=maxh)

    # BOOLEAN OPS ──────────────────────────────────────────────────────────

    def fragment(self, target: GeoObject, *tools: GeoObject) -> None:
        """make every overlap between ``target`` and ``tools`` conformal

        Splits each overlap into a shared face or volume that both
        operands keep. Meshing then produces a single conformal tet
        mesh across every interface, exactly what the FEM solver
        needs. Names and per-entity mesh sizes assigned on the
        operands survive.


        Note
        ----
        Uses gmsh's ``occ.fragment`` ``out_map`` to update each input's
        tag directly, robust against COG drift that can break naive
        name-tracking after boolean ops. Child faces and edges are
        re-resolved by ``(centroid, bbox)`` matching against the
        registry.


        Example
        -------
        Substrate + air + thin patch + lumped-port plate, all
        conformally fragmented in one call:

        .. code-block:: python

            g.fragment(air, sub, patch, feed)


        Parameters
        ----------
        target : GeoObject
            first operand
        *tools : GeoObject
            additional operands to fragment with ``target``
        """
        for _op in (target, *tools):
            self._reject_discrete(_op, "boolean op")
        target_dt = [(target.dim, target._entity.tag)]
        tools_dt = [(t.dim, t._entity.tag) for t in tools]
        _, out_map = gmsh.model.occ.fragment(target_dt, tools_dt)
        gmsh.model.occ.synchronize()
        inputs = [target] + list(tools)
        self._apply_out_map(inputs, out_map)
        self._reresolve_children(top_level=set(id(o._entity) for o in inputs))

    def cut(self, target: GeoObject, *tools: GeoObject) -> None:
        """boolean subtract ``tools`` from ``target``

        Carves the volumes / faces of ``tools`` out of ``target``. The
        target survives (possibly as several pieces); tools are
        consumed by the operation.


        Note
        ----
        Tools are **consumed** by ``cut``, do not reference them after
        the call. Use :meth:`fragment` instead if you need both
        operands to survive (e.g. for a substrate-in-air model).


        Parameters
        ----------
        target : GeoObject
            object to subtract from
        *tools : GeoObject
            objects to subtract (consumed)
        """
        for _op in (target, *tools):
            self._reject_discrete(_op, "boolean op")
        target_dt = [(target.dim, target._entity.tag)]
        tools_dt = [(t.dim, t._entity.tag) for t in tools]
        _, out_map = gmsh.model.occ.cut(target_dt, tools_dt)
        gmsh.model.occ.synchronize()
        # Tools are consumed by `cut`; only the target survives (with possibly
        # multiple pieces). out_map[0] = target's new pieces.
        self._apply_out_map([target], out_map[:1] if out_map else [[]])
        self._reresolve_children(top_level={id(target._entity)})

    def _apply_out_map(self, inputs: list[GeoObject], out_map: list) -> None:
        """For each input GeoObject, update its tag/cog/bbox from gmsh's out_map.
        If an input was split into multiple pieces, the first piece keeps the
        original GeoObject; the others are registered as additional `_Entity`s
        carrying the same name/material/maxh.

        When tools sit fully inside the target (e.g. iris-strips inside an air
        box), gmsh's out_map[0] for the target enumerates EVERY piece of the
        partition that covers it, including the pieces each tool claims as
        its own primary. Without de-duplication, those tool pieces would also
        appear as "extras" under the target with the wrong material, producing
        overlapping physical groups in gmsh and ambiguous material tagging in
        the Rust solver. We collect the primary tags first and skip them when
        creating extras.
        """
        primary_tags: set[tuple[int, int]] = set()
        for input_obj, new_dimtags in zip(inputs, out_map):
            if new_dimtags:
                d0, t0 = new_dimtags[0]
                primary_tags.add((int(d0), int(t0)))
        for input_obj, new_dimtags in zip(inputs, out_map):
            if not new_dimtags:
                warnings.warn(
                    f"GeoObject (dim={input_obj.dim}, name={input_obj._entity.name!r}) "
                    f"vanished during boolean op",
                    stacklevel=3,
                )
                continue
            d0, t0 = new_dimtags[0]
            input_obj._entity.tag = t0
            input_obj._entity.cog = tuple(gmsh.model.occ.getCenterOfMass(d0, t0))
            input_obj._entity.bbox = tuple(gmsh.model.getBoundingBox(d0, t0))
            for d, t in new_dimtags[1:]:
                if (int(d), int(t)) in primary_tags:
                    continue
                extra = _Entity.from_dimtag(d, t)
                extra.name = input_obj._entity.name
                extra.material = input_obj._entity.material
                extra.maxh = input_obj._entity.maxh
                self._entities.append(extra)

    def _reresolve_children(self, top_level: set[int]) -> None:
        """Re-resolve child entities (faces, edges) by COG+bbox. Top-level
        entities (already updated via out_map) are skipped via `top_level` set
        of `_Entity` ids."""
        survived = []
        for ent in self._entities:
            if id(ent) in top_level:
                survived.append(ent)
                continue
            new_tag = _resolve_entity(ent)
            if new_tag is not None:
                ent.tag = new_tag
                survived.append(ent)
            elif ent.name or ent.material or ent.maxh:
                warnings.warn(
                    f"Tracked entity (dim={ent.dim}, name={ent.name!r}, "
                    f"cog={ent.cog}) lost during boolean op; attributes dropped.",
                    stacklevel=3,
                )
        self._entities = survived

    # TRANSFORMS ───────────────────────────────────────────────────────────

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
        self._reject_discrete(obj, "rotate")
        cx, cy, cz = center
        ax, ay, az = axis
        s = self._s
        gmsh.model.occ.rotate([(obj.dim, obj._entity.tag)],
                              s(cx), s(cy), s(cz), ax, ay, az, angle)
        gmsh.model.occ.synchronize()
        self._refresh_descendants(obj)

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
        self._reject_discrete(obj, "stretch")
        cx, cy, cz = center
        s = self._s
        gmsh.model.occ.dilate([(obj.dim, obj._entity.tag)],
                              s(cx), s(cy), s(cz), fx, fy, fz)
        gmsh.model.occ.synchronize()
        self._refresh_descendants(obj)

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
        self._reject_discrete(obj, "translate")
        s = self._s
        gmsh.model.occ.translate([(obj.dim, obj._entity.tag)], s(dx), s(dy), s(dz))
        gmsh.model.occ.synchronize()
        self._refresh_descendants(obj)

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
        self._reject_discrete(obj, "mirror")
        nx, ny, nz = normal
        px, py, pz = point
        s = self._s
        # Plane a*x + b*y + c*z + d = 0 through `point` with the given normal.
        # a,b,c are dimensionless; d carries length units, so build it from the
        # scaled point coordinates (everything inside gmsh lives in scaled space).
        d = -(nx * s(px) + ny * s(py) + nz * s(pz))
        gmsh.model.occ.mirror([(obj.dim, obj._entity.tag)], nx, ny, nz, d)
        gmsh.model.occ.synchronize()
        self._refresh_descendants(obj)

    def copy(self, obj: GeoObject, *,
             material=None,
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
        self._reject_discrete(obj, "copy")
        out = gmsh.model.occ.copy([(obj.dim, obj._entity.tag)])
        gmsh.model.occ.synchronize()
        new_dim, new_tag = out[0]
        eff_maxh = maxh if maxh is not None else obj.maxh
        if new_dim == 3:
            eff_material = material if material is not None else obj.material
            return self._wrap_volume(new_tag, material=eff_material, maxh=eff_maxh)
        return self._wrap_face(new_tag, maxh=eff_maxh)

    def array(self, obj: GeoObject, count: int, *,
              spacing: tuple[float, float, float] | None = None,
              rotation: float | None = None,
              axis: tuple[float, float, float] = (0, 0, 1),
              center: tuple[float, float, float] = (0, 0, 0)) -> list[GeoObject]:
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
            raise ValueError(
                "array: pass exactly one of spacing= (linear) or rotation= (polar)")
        instances = [obj]
        for i in range(1, count):
            inst = self.copy(obj)
            if spacing is not None:
                # Place each copy at i steps from the original (no cumulative
                # drift: always derived from the un-moved source).
                inst_shift = (spacing[0] * i, spacing[1] * i, spacing[2] * i)
                self.translate(inst, *inst_shift)
            else:
                self.rotate(inst, rotation * i, axis=axis, center=center)
            instances.append(inst)
        return instances

    def _refresh_descendants(self, obj: GeoObject) -> None:
        """Refresh COG/bbox for every tracked entity in ``obj``'s boundary
        tree (plus ``obj`` itself). In-place transforms keep dimtags but
        move centroids, without this, named-face resolvers would miss.

        gmsh's ``getBoundary(recursive=True)`` descends straight to the
        vertices, so we walk one dimension at a time to collect faces and
        edges as well.
        """
        descendants = {(obj.dim, obj._entity.tag)}
        current = [(obj.dim, obj._entity.tag)]
        while current and current[0][0] > 0:
            next_level = gmsh.model.getBoundary(current, oriented=False, recursive=False)
            descendants.update(next_level)
            current = list(next_level)
        for ent in self._entities:
            if (ent.dim, ent.tag) in descendants:
                ent.cog = tuple(gmsh.model.occ.getCenterOfMass(ent.dim, ent.tag))
                ent.bbox = tuple(gmsh.model.getBoundingBox(ent.dim, ent.tag))

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
        for _op in (target, *tools):
            self._reject_discrete(_op, "boolean op")
        target_dt = [(target.dim, target._entity.tag)]
        tools_dt = [(t.dim, t._entity.tag) for t in tools]
        _, out_map = gmsh.model.occ.intersect(target_dt, tools_dt)
        gmsh.model.occ.synchronize()
        # Tools are consumed; only target survives (possibly as several pieces).
        self._apply_out_map([target], out_map[:1] if out_map else [[]])
        self._reresolve_children(top_level={id(target._entity)})

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
        warnings.warn(
            "fuse() merges faces and shifts their COGs; named faces on the "
            "operands will not be reliably re-resolvable. Set face names AFTER "
            "fuse, or use fragment() if you need them preserved.",
            stacklevel=2,
        )
        for _op in (target, *tools):
            self._reject_discrete(_op, "boolean op")
        target_dt = [(target.dim, target._entity.tag)]
        tools_dt = [(t.dim, t._entity.tag) for t in tools]
        _, out_map = gmsh.model.occ.fuse(target_dt, tools_dt)
        gmsh.model.occ.synchronize()
        inputs = [target] + list(tools)
        self._apply_out_map(inputs, out_map)
        self._reresolve_children(top_level=set(id(o._entity) for o in inputs))

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
        self._reject_discrete(obj, "fillet")
        if obj.dim != 3:
            raise ValueError(f"fillet expects a volume (dim=3), got dim={obj.dim}")
        edge_tags = [e.tag for e in (edges if edges is not None else obj.edges)]
        if not edge_tags:
            raise ValueError("fillet: no edges to round")
        s = self._s
        # A single radius is broadcast by gmsh across every supplied curve.
        out = gmsh.model.occ.fillet([obj._entity.tag], edge_tags, [s(radius)],
                                    removeVolume=True)
        gmsh.model.occ.synchronize()
        self._apply_out_map([obj], [out])
        self._reresolve_children(top_level={id(obj._entity)})
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
        self._reject_discrete(obj, "chamfer")
        if obj.dim != 3:
            raise ValueError(f"chamfer expects a volume (dim=3), got dim={obj.dim}")
        edge_tags = [e.tag for e in (edges if edges is not None else obj.edges)]
        if not edge_tags:
            raise ValueError("chamfer: no edges to bevel")
        # gmsh's chamfer measures the setback from a reference surface per
        # curve, so pair each edge with one of its adjacent faces.
        surf_tags: list[int] = []
        for et in edge_tags:
            up, _down = gmsh.model.getAdjacencies(1, et)
            if len(up) == 0:
                raise RuntimeError(f"chamfer: edge {et} has no adjacent surface")
            surf_tags.append(int(up[0]))
        s = self._s
        out = gmsh.model.occ.chamfer([obj._entity.tag], edge_tags, surf_tags,
                                     [s(distance)], removeVolume=True)
        gmsh.model.occ.synchronize()
        self._apply_out_map([obj], [out])
        self._reresolve_children(top_level={id(obj._entity)})
        return obj

    # MESH EMIT ────────────────────────────────────────────────────────────

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
            bbox = obj._entity.bbox
            dims = (bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
            min_dim = min(d for d in dims if d > 0)
            if min_dim >= base_maxh:
                continue
            h = min_dim / resolution
            if min_maxh is not None:
                h = max(h, min_maxh)
            obj.maxh = h
            cog = obj._entity.cog
            desc = obj.name or f"vol@({cog[0]*1e3:.1f},{cog[1]*1e3:.1f},{cog[2]*1e3:.1f})mm"
            assigned[desc] = h
        return assigned

    def refine_near_points(self, points, h: float,
                           distance: float | None = None) -> None:
        """register a local mesh-size refinement around a point cloud

        On the next :meth:`mesh` call, gmsh's ``Distance`` + ``Threshold``
        background fields will enforce element size ``h`` within
        ``distance`` of any point in ``points``, smoothly relaxing back
        to the global cap further out. Multiple calls are additive,
        each request becomes its own field and merges with the others
        via ``Min``.

        Designed to consume the output of
        :meth:`rapidfem.ProblemFD.element_errors`: the user marks high-η
        tets, picks a target size relative to their current ``h_k``,
        and re-meshes. The loop is explicit (user drives it); no
        automatic AMR.


        Example
        -------
        .. code-block:: python

            errs = prob.element_errors(result, freq_idx=res_idx,
                                       theta=0.3)
            hot = errs.tet_centroids[errs.marked]
            h_target = errs.h_k[errs.marked].mean() * 0.5
            g.refine_near_points(hot, h=h_target)
            g.mesh()                # picks up the new field
            prob2 = rf.Problem(g)   # fresh problem on the refined mesh


        Parameters
        ----------
        points : array_like
            ``(N, 3)`` coordinates of refinement centres in metres
        h : float
            target tet size at the points (m)
        distance : float, optional
            transition radius; defaults to ``5 * h`` (smooth ramp back
            to the global cap)
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"refine_near_points expects (N, 3) array, got shape {pts.shape}"
            )
        if h <= 0:
            raise ValueError(f"h must be positive, got {h}")
        self._refinements.append({
            "points": pts,
            "h": float(h),
            "distance": (float(distance) if distance is not None else 5.0 * float(h)),
        })

    # ── Physical-group assignment (shared by mesh() and mesh mode) ───────────
    def _assign_physical_groups(self) -> dict[str, int]:
        """Bake the material + physics registries into gmsh physical groups.

        Three sources, in order:

        1. Object-API :class:`~rapidfem.Material` instances (one group each)
        2. Object-API physics objects (one group per ``rf.PEC``/Port/... call)
        3. Legacy string materials/names (``rfic.Stack`` and old code paths)

        Populates ``self._material_tags`` / ``self._physics_tags`` (keyed by
        ``id()``) and returns the legacy name→tag map. Assumes any prior groups
        were already cleared by the caller. Used both by :meth:`mesh` (OCC path)
        and by the mesh-mode bake (a pre-built ``.msh``), so the downstream
        :class:`~rapidfem.Problem` TOML walk is identical for both substrates.
        """
        self._material_tags = {}
        self._physics_tags = {}
        name_to_tag: dict[str, int] = {}
        # Start physical-group tags well above every entity tag, the Rust
        # mesh loader (`src/mesh_io.rs::tris_for_tag`) keys tris by either
        # the entity's physical-group tag OR (fallback) the entity tag
        # itself when no group is assigned. Sharing the integer namespace
        # means a physical-group tag like 9 collides with face entity 9
        # and Port "9" picks up the unrelated entity's triangles.
        next_tag = 100_000

        # Collect volume entities targeted by PML, they go into the PML
        # physical group (step 2) and must NOT also land in a material group,
        # otherwise the Rust solver sees them tagged twice and the PML's
        # coordinate stretch is overridden by the bulk material assignment.
        pml_volume_ids: set[int] = set()
        for phys in self._physics:
            if type(phys).__name__ == "PML":
                for ent in getattr(phys, "_entities", ()):
                    pml_volume_ids.add(id(ent))

        # Per-class counters keep physical-group names unique without leaking
        # python id()s into the viewer legend. Class lower-case + 1-based index.
        # Example: two Dielectric() instances → "dielectric_1", "dielectric_2".
        # Driven ports collapse onto a shared "port_<N>" namespace so the legend
        # reads Port 1 / Port 2 regardless of waveguide/lumped/coax mix.
        mat_class_count: dict[str, int] = {}
        phys_class_count: dict[str, int] = {}
        port_classes = {
            "RectWaveguidePort", "LumpedPort", "CoaxPort", "WavePort",
            "UserDefinedPort", "FloquetPort",
        }

        def _mat_group_name(mat) -> str:
            cls = type(mat).__name__.lower()
            mat_class_count[cls] = mat_class_count.get(cls, 0) + 1
            return f"{cls}_{mat_class_count[cls]}"

        def _phys_group_name(phys) -> str:
            cls_name = type(phys).__name__
            key = "port" if cls_name in port_classes else cls_name.lower()
            phys_class_count[key] = phys_class_count.get(key, 0) + 1
            return f"{key}_{phys_class_count[key]}"

        # 1) Material instances → volume groups (skipping PML-targeted volumes).
        mat_to_volumes: dict[int, tuple[object, list[int]]] = {}
        for ent in self._entities:
            mat = ent.material
            # Skip strings (handled in step 3) and None.
            if mat is None or isinstance(mat, str):
                continue
            if ent.dim != 3:
                continue
            if id(ent) in pml_volume_ids:
                continue
            key = id(mat)
            if key not in mat_to_volumes:
                mat_to_volumes[key] = (mat, [])
            mat_to_volumes[key][1].append(ent.tag)
        for mat_id, (mat, tags) in mat_to_volumes.items():
            phys_tag = next_tag
            next_tag += 1
            gmsh.model.addPhysicalGroup(3, tags, tag=phys_tag, name=_mat_group_name(mat))
            self._material_tags[mat_id] = phys_tag

        # 2) Physics objects → faces or volume groups.
        for phys in self._physics:
            # A PeriodicBoundary is a two-sided physics object: each side
            # must carry its own physical-group tag so the time-domain
            # backend can match the pair. The geometry stores the pair as
            # `(tag_a, tag_b)` under `_physics_tags`; downstream walkers
            # ignore it unless they are the periodic collector.
            if type(phys).__name__ == "PeriodicBoundary":
                ents_a = getattr(phys, "_entities_a", None)
                ents_b = getattr(phys, "_entities_b", None)
                if not ents_a or not ents_b:
                    continue
                name = _phys_group_name(phys)
                tag_a = next_tag
                next_tag += 1
                gmsh.model.addPhysicalGroup(
                    2, [e.tag for e in ents_a], tag=tag_a, name=f"{name}_a")
                tag_b = next_tag
                next_tag += 1
                gmsh.model.addPhysicalGroup(
                    2, [e.tag for e in ents_b], tag=tag_b, name=f"{name}_b")
                self._physics_tags[id(phys)] = (tag_a, tag_b)
                continue

            ents = getattr(phys, "_entities", None)
            if not ents:
                continue
            # All entities in one physics object share dim by construction.
            dim = ents[0].dim
            tags = [e.tag for e in ents]
            phys_tag = next_tag
            next_tag += 1
            phys_id = id(phys)
            gmsh.model.addPhysicalGroup(dim, tags, tag=phys_tag, name=_phys_group_name(phys))
            self._physics_tags[phys_id] = phys_tag

        # 3) Legacy: name/material strings (rfic.Stack + builder workflow).
        by_dim_name: dict[tuple[int, str], list[int]] = {}
        for ent in self._entities:
            if ent.name:
                by_dim_name.setdefault((ent.dim, ent.name), []).append(ent.tag)
        for ent in self._entities:
            if isinstance(ent.material, str) and ent.dim == 3:
                key = (3, f"_mat_{ent.material}")
                by_dim_name.setdefault(key, []).append(ent.tag)
        for (dim, name), tags in by_dim_name.items():
            phys_tag = next_tag
            next_tag += 1
            gmsh.model.addPhysicalGroup(dim, tags, tag=phys_tag, name=name)
            display_name = name[len("_mat_"):] if name.startswith("_mat_") else name
            name_to_tag[display_name] = phys_tag

        return name_to_tag

    def _mesh_imported(self) -> "tuple[bytes, dict[str, int]]":
        """Bake bindings into a pre-built mesh (mesh mode), no remeshing.

        A geometry loaded from a ``.msh`` (via :meth:`load`) is already
        tessellated, so there is nothing to mesh: we clear the file's original
        physical groups, re-tag from the material/physics the user attached to
        the mesh's named groups, and serialise. The element/node table is the
        loaded mesh, untouched. Downstream is identical to the OCC path because
        the same registries and tag maps drive :meth:`rapidfem.Problem` TOML.
        """
        if not self._physics and not any(
                e.material is not None for e in self._entities):
            raise RuntimeError(
                "mesh mode: no materials or physics attached. Bind them to the "
                "mesh's named groups, e.g. scene.group('air').material = "
                "rf.Air() or rf.PEC(scene.group('metal')), before g.mesh().")
        # Drop the file's own physical groups; the user's bindings are now
        # authoritative and are written back as fresh high-numbered tags.
        try:
            for dim, ptag in gmsh.model.getPhysicalGroups():
                gmsh.model.removePhysicalGroups([(dim, ptag)])
        except Exception:
            pass
        name_to_tag = self._assign_physical_groups()

        gmsh.option.setNumber("Mesh.SaveAll", 1)
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            tmp_path = f.name
        try:
            gmsh.write(tmp_path)
            with open(tmp_path, "rb") as f:
                mesh_bytes = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        self._last_mesh = (mesh_bytes, name_to_tag)
        return mesh_bytes, name_to_tag

    def mesh(
        self,
        maxh: float | None = None,
        transition_distance: float | None = None,
        algorithm: str = "hxt",
        optimize: bool | str = True,
    ) -> tuple[bytes, dict[str, int]]:
        """generate the 3-D tet mesh of the current geometry

        Calls gmsh's OCC mesher with the configured per-entity sizes
        and global cap. Per-entity ``obj.maxh = h`` is honoured via
        gmsh ``Distance + Threshold`` background fields so refinement
        transitions are smooth, not abrupt.

        The 3-D mesher and a post-pass sliver fixer are configured by the
        ``algorithm`` and ``optimize`` kwargs (defaults give the highest-
        quality mesh; the per-call overrides are escape hatches).


        Note
        ----
        Three sources of physical groups are created at mesh time:

        - Every :class:`rapidfem.Material` instance attached via
          ``g.box(..., material=...)`` gets its own physical group
          on dim 3; the resulting tag is stored in
          ``self._material_tags[id(material)]``.
        - Every physics object in ``self._physics`` (created by
          ``rf.PEC(...)``, ``rf.LumpedPort(...)``, ...) gets its own
          physical group containing all its target entities; the tag is
          stored in ``self._physics_tags[id(physics_obj)]``.
        - Legacy string materials/names (``obj.material = "fr4"``,
          ``obj.name = "ground"``) continue to work for the GDS
          import / ``rfic.Stack`` flow, they produce name-keyed groups
          in the returned ``name_to_tag`` dict.


        Example
        -------
        .. code-block:: python

            g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
            # ... primitives + physics ...
            g.mesh()                      # uses the geometry's maxh
            g.mesh(maxh=1e-3)             # one-off override


        Parameters
        ----------
        maxh : float, optional
            global mesh size cap override in metres; falls back to the
            ``maxh=`` passed to :class:`Geometry` when ``None`` (raises
            if neither is set)
        transition_distance : float, optional
            distance over which a refined region's element size grows
            from its local ``h`` to the global cap (defaults to
            :math:`5h` per-entity)

        Returns
        -------
        mesh_bytes : bytes
            gmsh ``.msh`` v4 file as bytes, also cached on
            ``self._last_mesh`` for :class:`rapidfem.Problem`
        name_to_tag : dict[str, int]
            legacy name → tag map (empty under the object-API path)
        """
        # Mesh mode (a pre-built .msh loaded via load()): nothing to tessellate,
        # just bake the user's group bindings and serialise the loaded mesh.
        if self._mode == "mesh":
            return self._mesh_imported()
        if maxh is None:
            maxh = self._maxh
        if maxh is None:
            raise ValueError(
                "no maxh set, pass it to Geometry(maxh=...) or g.mesh(maxh=...)"
            )
        gmsh.model.occ.synchronize()
        # Dilate every OCC entity from the internal scaled coords back to
        # user units BEFORE any mesh setup. Threshold fields, mesh size
        # hints and the mesher itself then all see real-meter geometry; the
        # resulting .msh ends up in user units without needing a post-mesh
        # transform. Idempotent, flag guards re-runs of mesh().
        if self._scale != 1.0 and not getattr(self, "_dilated", False):
            s = self._scale
            all_dt = gmsh.model.getEntities()
            if all_dt:
                gmsh.model.occ.dilate(all_dt, 0, 0, 0, s, s, s)
                gmsh.model.occ.synchronize()
            self._dilated = True

        # Wipe any prior mesh state AND physical groups. Without the latter,
        # re-running this cell hits "Physical surface 1 already exists".
        # Without the former, gmsh reuses stale 1D/2D meshes and partially
        # ignores the new maxh.
        try:
            gmsh.model.mesh.clear()
        except Exception:
            pass
        try:
            for dim, ptag in gmsh.model.getPhysicalGroups():
                gmsh.model.removePhysicalGroups([(dim, ptag)])
        except Exception:
            pass

        # ── Per-entity mesh size: gmsh Distance + Threshold background fields ──
        # Effective per-entity maxh = explicit `ent.maxh` first, then the
        # entity's Material.maxh as a per-material refinement floor (lets
        # users tag every conductor or thin dielectric without touching the
        # primitives that carry that material).
        threshold_field_ids: list[int] = []
        for ent in self._entities:
            eff_maxh = ent.maxh
            if eff_maxh is None:
                eff_maxh = getattr(ent.material, "maxh", None)
            if eff_maxh is None:
                continue
            dist_id = gmsh.model.mesh.field.add("Distance")
            if ent.dim == 0:
                gmsh.model.mesh.field.setNumbers(dist_id, "PointsList", [ent.tag])
            elif ent.dim == 1:
                gmsh.model.mesh.field.setNumbers(dist_id, "CurvesList", [ent.tag])
            elif ent.dim == 2:
                gmsh.model.mesh.field.setNumbers(dist_id, "SurfacesList", [ent.tag])
            elif ent.dim == 3:
                # Volumes: refine across their boundary surfaces
                boundary = gmsh.model.getBoundary([(3, ent.tag)], oriented=False)
                surf_tags = [t for d, t in boundary if d == 2]
                if not surf_tags:
                    continue
                gmsh.model.mesh.field.setNumbers(dist_id, "SurfacesList", surf_tags)
            else:
                continue

            thr_id = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thr_id, "InField", dist_id)
            # Post-dilate, gmsh is in user units → set thresholds in user
            # units too.
            gmsh.model.mesh.field.setNumber(thr_id, "SizeMin", eff_maxh)
            gmsh.model.mesh.field.setNumber(thr_id, "SizeMax", maxh)
            gmsh.model.mesh.field.setNumber(thr_id, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(
                thr_id, "DistMax",
                transition_distance if transition_distance is not None else 5 * eff_maxh,
            )
            threshold_field_ids.append(thr_id)

        # ── refine_near_points() requests ─────────────────────────────────────
        # Each registered refinement becomes an OCC point cloud + a
        # Distance + Threshold field. The OCC points must exist on the
        # synced model before the Distance field can reference them.
        # Each refinement request becomes a cloud of OCC points with a
        # ``meshSize`` attribute, embedded into the volumes that
        # contain them, plus a Distance + Threshold field that smooths
        # the transition out to ``distance``. Three pieces together,
        # ``meshSize`` + ``embed`` + ``MeshSizeFromPoints=1``, are
        # required to get HXT to actually honour the local size; the
        # field alone is too soft (HXT-Delaunay treats it as a hint,
        # not a constraint). Empirically: bare-field gives ~0%
        # refinement; full recipe gives +200% local tets.
        refinement_field_ids: list[int] = []
        refinement_has_embed = False
        # Cache the volume list once, typically a handful, and
        # ``isInside`` is cheap. After-fragment we have the final
        # volume set.
        post_frag_vols = [t for d, t in gmsh.model.getEntities(dim=3)]
        _refdbg = os.environ.get("RAPIDFEM_REFINE_DEBUG")
        if _refdbg:
            print(f"[refine] {len(self._refinements)} requests in queue",
                  file=sys.stderr)
        for req in self._refinements:
            pts = req["points"]
            h = req["h"]
            dist_radius = req["distance"]
            tags: list[int] = []
            for p in pts:
                # ``meshSize`` on the OCC point ties the local size to
                # the point, picked up when MeshSizeFromPoints=1.
                tag = gmsh.model.occ.addPoint(
                    self._s(float(p[0])), self._s(float(p[1])),
                    self._s(float(p[2])),
                    meshSize=self._s(h),
                )
                tags.append(tag)
            if not tags:
                continue
            gmsh.model.occ.synchronize()
            # Dilate the newly-added points if the geometry was dilated
            # earlier (scale != 1), keep them in user units.
            if self._scale != 1.0 and getattr(self, "_dilated", False):
                s = self._scale
                gmsh.model.occ.dilate(
                    [(0, t) for t in tags], 0, 0, 0, s, s, s,
                )
                gmsh.model.occ.synchronize()

            # Embed each point into the volume that contains it (the
            # mesher only treats embedded points as size constraints;
            # free OCC points get ignored).
            for tag, p in zip(tags, pts):
                coords = [self._s(float(p[0])), self._s(float(p[1])),
                          self._s(float(p[2]))]
                if self._scale != 1.0 and getattr(self, "_dilated", False):
                    # Post-dilate, coords are user-meters; gmsh's
                    # isInside expects model coords (= user-meters
                    # too, post-dilate).
                    coords = [float(p[0]), float(p[1]), float(p[2])]
                for vol_tag in post_frag_vols:
                    try:
                        if gmsh.model.isInside(3, vol_tag, coords) > 0:
                            gmsh.model.mesh.embed(0, [tag], 3, vol_tag)
                            refinement_has_embed = True
                            break
                    except Exception:
                        continue

            dist_id = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist_id, "PointsList", tags)
            thr_id = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thr_id, "InField", dist_id)
            gmsh.model.mesh.field.setNumber(thr_id, "SizeMin", h)
            gmsh.model.mesh.field.setNumber(thr_id, "SizeMax", maxh)
            gmsh.model.mesh.field.setNumber(thr_id, "DistMin", 0.0)
            gmsh.model.mesh.field.setNumber(thr_id, "DistMax", dist_radius)
            refinement_field_ids.append(thr_id)
            if _refdbg:
                print(f"[refine] added Distance #{dist_id} + Threshold #{thr_id} "
                      f"(N_pts={len(tags)}, h={h:.3e}, dist={dist_radius:.3e})",
                      file=sys.stderr)

        # Pin the mesh-size options unconditionally. These gmsh options are
        # process-global, NOT per-model, so a previous mesh() (e.g. one that
        # took the field branch below and set FromPoints=0) would otherwise
        # leak into a plain, field-less geometry and silently coarsen it,
        # making the result depend on call order. Set the field-less defaults
        # here; the field branch overrides them when refinement is active.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
        gmsh.option.setNumber(
            "Mesh.MeshSizeExtendFromBoundary", 1 if self._grading else 0)

        all_field_ids = threshold_field_ids + refinement_field_ids
        if _refdbg:
            print(f"[refine] all_field_ids={all_field_ids}, "
                  f"fields_list={list(gmsh.model.mesh.field.list())}",
                  file=sys.stderr)
        if all_field_ids:
            min_id = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_id, "FieldsList", all_field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_id)
            if _refdbg:
                print(f"[refine] Min combiner = field #{min_id}",
                      file=sys.stderr)
            # When threshold fields are active the user explicitly cares
            # about local size. Keep ``ExtendFromBoundary`` off so the
            # global Max applies away from refined regions, but leave
            # Curvature on (combined via Min) so curved features get
            # resolved cleanly even if the user only set per-volume
            # sizes. ``MeshSizeFromPoints`` stays ON when there are
            # refinement points so their ``meshSize`` attribute kicks
            # in (HXT-Delaunay needs both the field AND the per-point
            # size to actually refine, field alone is too soft).
            gmsh.option.setNumber(
                "Mesh.MeshSizeFromPoints",
                1 if refinement_has_embed else 0,
            )
            # Without ExtendFromBoundary, HXT-Delaunay treats the
            # background field as a soft hint and barely refines the
            # interior, and lets neighbouring tets across an interface
            # jump by an order of magnitude (e.g. 0.5 mm substrate next
            # to 10 mm air). With grading ON the boundary sizes propagate
            # smoothly inward, killing those extreme-size transitions at
            # the cost of ~15-30% more tets in the air. Point-driven
            # refinement always needs this propagation (the embedded
            # points are how the field reaches the bulk in the first
            # place), so it overrides the user's grading toggle.
            gmsh.option.setNumber(
                "Mesh.MeshSizeExtendFromBoundary",
                1 if (self._grading or refinement_has_embed) else 0,
            )

        # Curvature-based sizing: gmsh disables this by default. Turning it
        # on gives curved primitives (cylinder, sphere, cone, torus) a
        # geometry-accurate facet count without the user having to refine
        # those surfaces by hand. Value = target elements per 2π radians.
        # 12 is a reasonable balance between fidelity and DoF count for
        # second-kind Nédélec-2 (high-order absorbs some discretisation
        # error already). User can override before calling .mesh().
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)

        # Assign physical groups from the material + physics registries. Shared
        # with the mesh-mode path (a pre-built .msh that skips meshing).
        name_to_tag = self._assign_physical_groups()

        # Generate. SaveAll=1 ensures volumes without explicit material/name still
        # land in the .msh (otherwise gmsh writes only physical-group elements).
        # Post-dilate, gmsh is in user units → no further scaling needed.
        gmsh.option.setNumber("Mesh.MeshSizeMax", maxh)
        gmsh.option.setNumber("Mesh.SaveAll", 1)

        # 3-D mesher choice. HXT (algorithm 10) is gmsh's parallel Delaunay:
        # it scales across cores, and on curved bodies its tet population
        # carries fewer near-degenerate slivers than the serial Delaunay
        # (1) or Frontal (4). The original Delaunay default predates HXT
        # in gmsh and is kept only as an escape hatch, pick the relevant
        # algorithm explicitly so the mesh is reproducible across users.
        algo_codes = {
            "hxt":      10,    # parallel Delaunay (recommended default)
            "delaunay": 1,     # serial Delaunay (gmsh's historical default)
            "frontal":  4,     # frontal-Delaunay (occasionally cleaner on thin shells)
            "mmg3d":    7,     # MMG3D (anisotropic remesher; rarely needed)
        }
        algo_lc = algorithm.lower()
        if algo_lc not in algo_codes:
            raise ValueError(
                f"algorithm must be one of {sorted(algo_codes)}, got "
                f"{algorithm!r}"
            )
        gmsh.option.setNumber("Mesh.Algorithm3D", algo_codes[algo_lc])

        gmsh.model.mesh.generate(3)

        # Sliver-killing post-pass, run BEFORE writing the .msh so every
        # downstream consumer sees the optimised mesh.
        #
        # Default (``optimize=True``) uses gmsh's BUILT-IN optimiser (edge-swap
        # + node-smoothing): cheap and, crucially, crash-safe on degenerate
        # geometry. Netgen's optimiser removes slivers more aggressively but
        # HARD-CRASHES the process (a native segfault, NOT a catchable Python
        # exception) on geometry with embedded thin sheets or boolean slivers,
        # e.g. the Vivaldi feed/stub/port sheets plus the tapered-slot cut.
        # Opt into it with ``optimize="netgen"`` only when the geometry is
        # known to be well-behaved; ``optimize=False`` skips the pass.
        #
        # The try/except + re-mesh recovery below catches a *Python-level*
        # optimiser failure (the mesh state is then poisoned: boundary-face
        # physical groups silently drop elements, seen on the patch antenna's
        # PML+substrate+plate stack), but cannot catch a Netgen segfault, hence
        # the safe default.
        if optimize:
            optimizer = "Netgen" if str(optimize).lower() == "netgen" else ""
            try:
                gmsh.model.mesh.optimize(optimizer)
            except Exception as e:
                print(
                    f"warning: gmsh.optimize({optimizer!r}) failed "
                    f"({type(e).__name__}); regenerating the mesh without "
                    f"optimisation to keep port / PEC physical groups intact",
                    file=sys.stderr,
                )
                # Wipe + re-mesh. The size fields and Algorithm3D are still set
                # from above, so the second pass is parameter-identical, just
                # without the failed post-pass.
                gmsh.model.mesh.clear()
                gmsh.model.mesh.generate(3)

        # Write to a temp file, read bytes back
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
            tmp_path = f.name
        try:
            gmsh.write(tmp_path)
            with open(tmp_path, "rb") as f:
                mesh_bytes = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        self._last_mesh = (mesh_bytes, name_to_tag)
        return mesh_bytes, name_to_tag


__all__ = ["Geometry", "GeoObject", "FaceCollection"]
