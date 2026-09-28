# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""External geometry import for :class:`rapidfem.geometry.Geometry`, mixed
into it."""
from __future__ import annotations

from pathlib import Path

from . import _native

# metres per unit of the ``unit`` codes
_UNITS = {"M": 1.0, "CM": 1e-2, "MM": 1e-3, "UM": 1e-6, "MICRON": 1e-6,
          "NM": 1e-9, "IN": 0.0254, "INCH": 0.0254, "MIL": 2.54e-5, "FT": 0.3048}


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

    def __init__(self, geometry, groups: "dict[str, object]", dims: "dict[str, int]"):
        self._geometry = geometry
        self.groups = groups          # name -> EntityCollection
        self._dims = dims             # name -> topological dim (2 or 3)

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
    """File import, mixed into :class:`rapidfem.geometry.Geometry`."""

    def load(self, path: str, *,
             material=None,
             maxh: float | None = None,
             unit: str = "M",
             scale: float = 1.0,
             position: tuple[float, float, float] = (0.0, 0.0, 0.0),
             rotation: "tuple | None" = None,
             heal_angle: float = 40.0):
        """Load an external surface model as a solid, or a pre-built mesh.

        ``.stl`` and ``.obj`` files load as one solid each: the surface must
        be a closed, consistently oriented 2-manifold; it is split into smooth
        surfaces at creases sharper than ``heal_angle`` degrees and remeshed.
        A ``.msh`` volume mesh (gmsh MSH 4.1 or 2.2) is taken as it is and
        switches the geometry into *mesh mode*: its named physical groups
        become selectable handles (a :class:`MeshScene`) for materials and
        physics, and :meth:`mesh` bakes the bindings without remeshing.
        STEP, IGES and BREP import (milanofthe/rapidmesh-dev#37) is not
        available yet.


        Parameters
        ----------
        path : str
            path to the file
        material : rapidfem.Material, optional
            material of the imported solid
        maxh : float, optional
            mesh size on the solid in metres
        unit : str
            unit of the file's coordinates (``"M"``, ``"MM"``, ``"UM"``,
            ``"IN"``, ...); STL carries none
        scale : float
            extra factor on the coordinates, after ``unit``
        position : tuple[float, float, float]
            offset of the part in metres, applied last
        rotation : tuple, optional
            ``(angle_rad, axis)`` or ``(angle_rad, axis, centre)``, applied
            before ``position``
        heal_angle : float
            crease angle in degrees separating smooth surfaces


        Returns
        -------
        GeoObject or MeshScene
            the imported solid, or the groups of a loaded mesh
        """
        ext = Path(path).suffix.lower()
        if ext in (".step", ".stp", ".iges", ".igs", ".brep"):
            raise NotImplementedError(
                f"load: {ext} import is not available yet "
                f"(milanofthe/rapidmesh-dev#37); STL and OBJ are")
        if ext not in (".stl", ".obj", ".msh"):
            raise ValueError(f"load: unsupported extension {ext!r}; "
                             f"supported: .stl, .obj, .msh")
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        if ext == ".msh":
            if (tuple(position) != (0.0, 0.0, 0.0) or rotation is not None
                    or scale != 1.0):
                raise ValueError(
                    "load(.msh): position/rotation/scale are not supported for a "
                    "pre-built mesh; it is consumed in its own coordinates")
            return self._load_mesh(str(path))
        try:
            factor = _UNITS[unit.upper()] * float(scale)
        except KeyError:
            raise ValueError(f"load: unknown unit {unit!r}; one of {', '.join(_UNITS)}") from None
        obj = self._wrap(self._native.add_import(str(path), float(heal_angle), maxh),
                         sheet=False, material=material, maxh=maxh)
        if factor != 1.0:
            self.stretch(obj, factor, factor, factor)
        if rotation is not None:
            if len(rotation) not in (2, 3):
                raise ValueError("rotation must be (angle, axis) or (angle, axis, centre), "
                                 f"got {rotation!r}")
            angle, axis, *centre = rotation
            self.rotate(obj, float(angle), axis=axis,
                        center=centre[0] if centre else (0.0, 0.0, 0.0))
        if tuple(position) != (0.0, 0.0, 0.0):
            self.translate(obj, *position)
        return obj

    def _load_mesh(self, path: str) -> MeshScene:
        from .geometry import EntityCollection, _Entity
        if self._scene is not None or self._objects:
            raise RuntimeError("load(.msh): a loaded mesh is the whole geometry; "
                               "load it into a fresh Geometry()")
        self._scene = _native.MeshScene(path)
        groups, dims = {}, {}
        for name, dim in self._scene.groups():
            if name in groups:
                continue
            ent = _Entity(self, dim, group=name)
            self._entities.append(ent)
            groups[name] = EntityCollection(self, [ent])
            dims[name] = dim
        return MeshScene(self, groups, dims)
