# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""External geometry import for :class:`rapidfem.geometry.Geometry`, mixed
into it."""
from __future__ import annotations

from pathlib import Path


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
        """Load an external surface model as a solid.

        ``.stl`` and ``.obj`` files load as one solid each: the surface must
        be a closed, consistently oriented 2-manifold; it is split into smooth
        surfaces at creases sharper than ``heal_angle`` degrees and remeshed.
        STEP, IGES and BREP (milanofthe/rapidmesh-dev#37) and ``.msh`` import
        are not available yet.


        Parameters
        ----------
        path : str
            path to the file
        material : rapidfem.Material, optional
            material of the imported solid
        maxh : float, optional
            mesh size on the solid in metres
        unit, scale, position, rotation : optional
            placement of the part; only the identity placement is available
            yet (placement transforms: milanofthe/rapidmesh-dev#140)
        heal_angle : float
            crease angle in degrees separating smooth surfaces


        Returns
        -------
        GeoObject
            the imported solid
        """
        ext = Path(path).suffix.lower()
        if ext not in (".stl", ".obj"):
            raise NotImplementedError(
                f"load: {ext} import is not available yet (STL and OBJ are; "
                f"STEP/IGES/BREP: milanofthe/rapidmesh-dev#37)")
        if (scale != 1.0 or tuple(position) != (0.0, 0.0, 0.0) or rotation is not None
                or unit.upper() != "M"):
            raise NotImplementedError(
                "load: only the identity placement is available yet "
                "(milanofthe/rapidmesh-dev#140)")
        oid = self._native.add_import(str(path), float(heal_angle), maxh)
        return self._wrap(oid, sheet=False, material=material, maxh=maxh)
