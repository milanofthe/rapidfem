"""External CAD / mesh import for :class:`rapidfem.geometry.Geometry`.

Split out of ``geometry.py`` as a mixin, mirroring :class:`_GdsMixin` and
:class:`_PrimitivesMixin`: the file-import machinery (OpenCASCADE shape import,
STL surface healing, pre-built ``.msh`` ingestion) is a cohesive subsystem that
only touches the scale/wrap helpers and gmsh, so it lives here to keep the core
``Geometry`` file navigable. ``Geometry`` inherits :class:`_ImportMixin`, so
``g.load(...)`` is a normal ``Geometry`` method at runtime.

Two substrates, one entry point
-------------------------------
``g.load(path)`` dispatches on the file extension:

- **BREP/OCC formats** (``.step`` / ``.stp`` / ``.iges`` / ``.igs`` / ``.brep``)
  land in the *same* OpenCASCADE kernel the primitives use. Each imported solid
  comes back as a :class:`~rapidfem.geometry.GeoObject`, fully composable: boolean
  ops, transforms, ``material=``, ``.faces`` selectors, and physics all work
  unchanged.
- **STL** (``.stl``) is a discrete surface triangulation, not BREP. It is healed
  into a meshable solid (``classifySurfaces`` + ``createGeometry`` + a volume),
  so it can carry a material and be meshed, but boolean ops against it are
  best-effort (it is not a clean BREP).
- **Pre-built volume meshes** (``.msh``) are already tessellated. They cannot go
  through the OCC mesher, so loading one switches the geometry into *mesh mode*:
  the named physical groups in the file become selectable
  :class:`~rapidfem.geometry.EntityCollection` handles you attach materials and
  physics to. ``g.mesh()`` then bakes those bindings into physical groups and
  serialises the original mesh, no remeshing.
"""
from __future__ import annotations

import os

import gmsh

# OCC BREP formats share the primitives' kernel -> fully composable GeoObjects.
_CAD_EXTS = {".step", ".stp", ".iges", ".igs", ".brep", ".brp"}
# Discrete surface mesh: healed into a meshable solid (boolean-weak).
_STL_EXTS = {".stl"}
# Pre-built volume mesh: switches the geometry into mesh mode.
_MESH_EXTS = {".msh"}


class MeshScene:
    """Handle for a pre-built ``.msh`` loaded via :meth:`Geometry.load`.

    A loaded mesh is already discretised, so it does not expose primitives or
    boolean ops. Instead, the named *physical groups* baked into the file are
    surfaced as selectable :class:`~rapidfem.geometry.EntityCollection` handles
    (one per group) that you attach materials and physics to, exactly like the
    faces/volumes of a primitive::

        scene = g.load("antenna.msh")
        scene.group("air").material = rf.Air()
        rf.WavePort(scene.group("port1"))
        rf.PEC(scene.group("metal"))
        g.mesh()                         # bakes bindings, no remeshing

    Attributes
    ----------
    groups : dict[str, EntityCollection]
        every named physical group in the file, keyed by name
    """

    def __init__(self, geometry, groups: "dict[str, object]",
                 dims: "dict[str, int]", tags: "dict[str, int]"):
        self._geometry = geometry
        self.groups = groups          # name -> EntityCollection
        self._dims = dims             # name -> topological dim (2 or 3)
        self._file_tags = tags        # name -> physical-group tag in the file

    def group(self, name: str):
        """Return the :class:`EntityCollection` for physical group ``name``.

        Raises ``KeyError`` with the available names if ``name`` is absent.
        """
        try:
            return self.groups[name]
        except KeyError:
            avail = ", ".join(sorted(self.groups)) or "(none)"
            raise KeyError(
                f"no physical group {name!r} in the mesh; available: {avail}"
            ) from None

    def __getitem__(self, name: str):
        return self.group(name)

    def __iter__(self):
        return iter(self.groups.values())

    def __repr__(self) -> str:
        items = ", ".join(f"{n}({self._dims[n]}D)" for n in sorted(self.groups))
        return f"MeshScene(groups=[{items}])"


class _ImportMixin:
    """External-file import, mixed into :class:`rapidfem.geometry.Geometry`."""

    def load(self, path: str, *,
             material=None,
             maxh: float | None = None,
             unit: str = "M",
             scale: float = 1.0,
             heal_angle: float = 40.0):
        """Load external CAD or a pre-built mesh into this geometry.

        Single entry point for every supported external format; the action is
        chosen from the file extension:

        =====================================  ============================================
        extension                              behaviour
        =====================================  ============================================
        ``.step`` ``.stp`` ``.iges`` ``.igs``  imported into the OCC kernel as solids;
        ``.brep``                              returns a composable ``GeoObject`` per solid
        ``.stl``                               healed into a meshable solid ``GeoObject``
        ``.msh``                               loaded as a :class:`MeshScene` (mesh mode)
        =====================================  ============================================

        Example
        -------
        .. code-block:: python

            # STEP part as a first-class primitive: boolean it against an air box
            g = rf.Geometry(maxh=rf.lambda_maxh(f_max=20e9))
            part = g.load("horn.step", material=rf.Air())
            rf.WavePort(part.faces.max(axis="z"))
            rf.PEC(*part.faces.unassigned)
            g.mesh()

        Parameters
        ----------
        path : str
            path to the CAD / mesh file
        material : rapidfem.Material, optional
            volume material applied to every imported solid (CAD/STL only)
        maxh : float, optional
            per-entity mesh size override in metres (CAD/STL only)
        unit : str
            target unit OpenCASCADE converts the file into; ``"M"`` (default)
            maps a millimetre STEP file onto rapidfem's metre convention.
            Accepts the gmsh unit codes (``"M"``, ``"MM"``, ``"CM"``,
            ``"IN"``, ...). Ignored for ``.stl`` (STL carries no unit) and
            ``.msh``.
        scale : float
            extra multiplicative factor applied after import, in metres per
            file unit. Use for unit-less STL (e.g. ``scale=1e-3`` for a model
            authored in millimetres) or to correct a mis-declared STEP unit.
        heal_angle : float
            STL only, the dihedral angle in degrees below which adjacent facets
            are merged into one smooth surface patch during healing

        Returns
        -------
        GeoObject or list[GeoObject] or MeshScene
            a single ``GeoObject`` for a one-solid CAD/STL import, a list for a
            multi-solid CAD import, or a :class:`MeshScene` for a ``.msh``
        """
        ext = os.path.splitext(str(path))[1].lower()
        if not os.path.exists(path):
            raise FileNotFoundError(f"load(): no such file: {path}")
        if ext in _CAD_EXTS:
            return self._load_cad(path, material=material, maxh=maxh,
                                  unit=unit, scale=scale)
        if ext in _STL_EXTS:
            return self._load_stl(path, material=material, maxh=maxh,
                                  scale=scale, heal_angle=heal_angle)
        if ext in _MESH_EXTS:
            return self._load_mesh(path)
        supported = ", ".join(sorted(_CAD_EXTS | _STL_EXTS | _MESH_EXTS))
        raise ValueError(
            f"load(): unsupported extension {ext!r} for {path}; "
            f"supported: {supported}")

    # ── CAD: STEP / IGES / BREP ─────────────────────────────────────────────
    def _load_cad(self, path, *, material, maxh, unit, scale):
        self._require_occ_mode("import CAD")
        # OCC converts the file's declared unit into `unit` on import. With the
        # default unit="M" a millimetre STEP comes in at metre coordinates,
        # matching rapidfem's metre convention.
        gmsh.option.setString("Geometry.OCCTargetUnit", unit)
        # Snapshot the model so we can isolate *exactly* the imported entities;
        # dilating/​wrapping must not touch primitives already in the geometry.
        before = {tuple(dt) for dt in gmsh.model.occ.getEntities()}
        out = gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        if not out:
            raise RuntimeError(f"importShapes produced no entities from {path}")
        new_dt = [dt for dt in gmsh.model.occ.getEntities()
                  if tuple(dt) not in before]

        # Bring the imported entities into the geometry's working space. Coords
        # read back from gmsh stay scaled (divided by _scale); primitives apply
        # `_s` at creation, the importer applies the equivalent dilate here. The
        # optional `scale` factor (metres per file unit) folds in too. mesh()
        # dilates everything back to user metres before tessellation.
        factor = (1.0 / self._scale) * float(scale)
        if factor != 1.0:
            gmsh.model.occ.dilate(new_dt, 0, 0, 0, factor, factor, factor)
            gmsh.model.occ.synchronize()

        vols = [t for d, t in new_dt if d == 3]
        if not vols:
            # Surface-only CAD (rare): wrap free faces so the user can still
            # build with them (extrude, loft, ...).
            faces = [t for d, t in new_dt if d == 2]
            objs = [self._wrap_face(t, maxh=maxh) for t in faces]
            return objs[0] if len(objs) == 1 else objs

        objs = [self._wrap_volume(t, material=material, maxh=maxh) for t in vols]
        return objs[0] if len(objs) == 1 else objs

    # ── STL: discrete surface healed into a meshable solid ──────────────────
    def _load_stl(self, path, *, material, maxh, scale, heal_angle):
        import math

        self._require_occ_mode("import STL")
        before = {tuple(dt) for dt in gmsh.model.getEntities()}
        # `merge` loads the STL triangles as a discrete surface in the model.
        gmsh.merge(str(path))

        # STL carries no unit. Scale the raw nodes about the origin into the
        # geometry's working space *before* healing (the geo kernel has no
        # dilate, so we transform the discrete nodes directly). mesh() dilates
        # everything back to user metres before tessellation.
        factor = (1.0 / self._scale) * float(scale)
        if factor != 1.0:
            self._scale_discrete_nodes(factor)

        # Reconstruct surface patches + a parametrisation from the raw triangle
        # soup so the volume mesher has something to mesh against. The dihedral
        # `ang` controls patch merging; pi as the curve angle keeps sharp edges.
        ang = heal_angle * math.pi / 180.0
        gmsh.model.mesh.classifySurfaces(ang, True, True, math.pi)
        gmsh.model.mesh.createGeometry()

        # Stitch the (now classified) surfaces into a closed shell + volume.
        surfaces = [t for d, t in gmsh.model.getEntities(2)
                    if (d, t) not in before]
        if not surfaces:
            raise RuntimeError(f"STL healing produced no surfaces from {path}")
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        vol = gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()

        return self._wrap_volume(vol, material=material, maxh=maxh)

    @staticmethod
    def _scale_discrete_nodes(factor: float) -> None:
        """Scale every mesh node about the origin by ``factor`` (STL path)."""
        import numpy as np

        tags, coords, _ = gmsh.model.mesh.getNodes()
        if len(tags) == 0:
            return
        xyz = np.asarray(coords, dtype=float).reshape(-1, 3) * factor
        for tag, p in zip(tags, xyz):
            gmsh.model.mesh.setNode(int(tag), p.tolist(), [])

    # ── Pre-built volume mesh: mesh mode ────────────────────────────────────
    def _load_mesh(self, path):
        from .geometry import EntityCollection, _Entity

        if self._objects:
            raise RuntimeError(
                "load(.msh): a pre-built mesh cannot be combined with OCC "
                "primitives in the same Geometry. Load the mesh into a fresh "
                "Geometry().")
        # Read the file into the gmsh model to enumerate its physical groups,
        # and keep the raw bytes: the solver consumes them verbatim, the
        # physical-group tags baked by mesh() are written back over the same
        # nodes/elements.
        gmsh.open(str(path))
        self._mode = "mesh"

        groups: dict[str, object] = {}
        dims: dict[str, int] = {}
        file_tags: dict[str, int] = {}
        for dim, ptag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, ptag)
            if not name:
                name = f"group_{dim}_{ptag}"
            ents: list = []
            for etag in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
                e = _Entity.from_dimtag(dim, etag)
                e._geometry = self
                self._entities.append(e)
                ents.append(e)
            groups[name] = EntityCollection(self, ents)
            dims[name] = dim
            file_tags[name] = ptag

        scene = MeshScene(self, groups, dims, file_tags)
        self._mesh_scene = scene
        return scene

    # ── Mode guard ──────────────────────────────────────────────────────────
    def _require_occ_mode(self, what: str) -> None:
        """Raise if the geometry is in mesh mode (no OCC operations allowed)."""
        if getattr(self, "_mode", "occ") == "mesh":
            raise RuntimeError(
                f"cannot {what}: this Geometry was created from a pre-built "
                f"mesh (load('*.msh')) and holds no OCC model. Primitives and "
                f"boolean ops are unavailable in mesh mode; attach materials "
                f"and physics to the mesh's named groups instead.")
