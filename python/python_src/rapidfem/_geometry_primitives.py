# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
"""Primitive solids and sheets of :class:`rapidfem.Geometry`, on the native
rapidmesh scene. Mixed into ``Geometry``; kept apart to keep geometry.py
navigable."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from .geometry import GeoObject

_FULL_TURN = 2 * math.pi


def _full_turn(angle: float, what: str) -> None:
    if abs(angle - _FULL_TURN) > 1e-12:
        raise NotImplementedError(
            f"{what}: partial sweeps are not available yet (build the solid "
            f"with Geometry.revolve from a profile instead)")


class _PrimitivesMixin:
    """Primitive factory methods of :class:`rapidfem.Geometry`."""

    def _profile(self, obj_id: int, pts) -> None:
        if not hasattr(self, "_profiles"):
            self._profiles = {}
        self._profiles[obj_id] = [tuple(float(v) for v in p) for p in pts]

    def box(self, width: float, depth: float, height: float,
            position: tuple[float, float, float] = (0, 0, 0),
            *,
            material=None,
            maxh: float | None = None) -> "GeoObject":
        """add an axis-aligned box primitive

        The workhorse volume primitive, used for substrates, air
        regions, waveguide cavities, and PML slabs. The returned
        :class:`GeoObject` has 6 ``.faces`` and 12 ``.edges`` selectable
        via :class:`EntityCollection`.


        Example
        -------
        .. code-block:: python

            air = g.box(22.86e-3, 10.16e-3, 30e-3,
                        position=(-11.43e-3, -5.08e-3, 0),
                        material=rf.Air())


        Parameters
        ----------
        width, depth, height : float
            extents along x, y, z respectively in metres
        position : tuple[float, float, float]
            lower corner ``(xmin, ymin, zmin)`` (defaults to origin)
        material : rapidfem.Material, optional
            volume material (``rf.Air()``, ``rf.Dielectric(er=...)``,
            ...)
        maxh : float, optional
            per-volume mesh size override in metres

        Returns
        -------
        GeoObject
            volume with 6 ``.faces`` and 12 ``.edges``
        """
        oid = self._native.add_box([width, depth, height], list(position), maxh)
        return self._wrap(oid, sheet=False, material=material, maxh=maxh)

    def cylinder(self, radius: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0),
                 axis: tuple[float, float, float] = (0, 0, 1),
                 angle: float = 2 * math.pi,
                 *,
                 material=None,
                 maxh: float | None = None) -> "GeoObject":
        """add a (partial-sweep) cylinder primitive

        Curved surfaces honour ``Mesh.MeshSizeFromCurvature`` so the
        cylinder side wall meshes into geometry-accurate facets without
        manual refinement.


        Example
        -------
        Outer dielectric of a coax line:

        .. code-block:: python

            air = g.cylinder(radius=ro, height=L,
                             position=(0, 0, 0),
                             material=rf.Air())


        Parameters
        ----------
        radius : float
            cylinder radius in metres
        height : float
            extent along ``axis``
        position : tuple[float, float, float]
            base centre (defaults to origin)
        axis : tuple[float, float, float]
            cylinder axis direction (defaults to +z)
        angle : float
            sweep angle in radians; defaults to :math:`2\\pi` (full
            cylinder), :math:`<2\\pi` gives a partial cylinder
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override

        Returns
        -------
        GeoObject
            volume
        """
        _full_turn(angle, "cylinder")
        oid = self._native.add_cylinder(radius, height, list(position), list(axis), maxh)
        return self._wrap(oid, sheet=False, material=material, maxh=maxh)

    def sphere(self, radius: float,
               position: tuple[float, float, float] = (0, 0, 0),
               *,
               material=None,
               maxh: float | None = None,
               center: tuple[float, float, float] | None = None) -> "GeoObject":
        """add a sphere primitive

        Parameters
        ----------
        radius : float
            sphere radius in metres
        position : tuple[float, float, float]
            sphere centre (defaults to origin)
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override
        center : tuple[float, float, float], optional
            deprecated alias for ``position``

        Returns
        -------
        GeoObject
            volume
        """
        c = center if center is not None else position
        oid = self._native.add_sphere(radius, list(c), maxh)
        return self._wrap(oid, sheet=False, material=material, maxh=maxh)

    def cone(self, r1: float, r2: float, height: float,
             position: tuple[float, float, float] = (0, 0, 0),
             axis: tuple[float, float, float] = (0, 0, 1),
             angle: float = 2 * math.pi,
             *,
             material=None,
             maxh: float | None = None) -> "GeoObject":
        """add a truncated cone (or cylinder if ``r1 == r2``)

        Parameters
        ----------
        r1, r2 : float
            base and top radii in metres
        height : float
            extent along ``axis``
        position : tuple[float, float, float]
            base centre (defaults to origin)
        axis : tuple[float, float, float]
            cone axis direction (defaults to +z)
        angle : float
            sweep angle in radians (defaults to :math:`2\\pi`)
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override

        Returns
        -------
        GeoObject
            volume
        """
        _full_turn(angle, "cone")
        oid = self._native.add_cone(r1, r2, height, list(position), list(axis), maxh)
        return self._wrap(oid, sheet=False, material=material, maxh=maxh)

    def wedge(self, dx: float, dy: float, dz: float,
              top_x: float = 0.0,
              position: tuple[float, float, float] = (0, 0, 0),
              *,
              material=None,
              maxh: float | None = None) -> "GeoObject":
        """add a rectangular-base prism (wedge)

        The base is ``dx × dy`` at z = 0; the top edge runs from x = 0
        to x = ``top_x`` at height ``dz``, parallel to y. Useful for
        symmetric horn walls and tapered ridge waveguides.


        Parameters
        ----------
        dx, dy, dz : float
            base width, base depth, height in metres
        top_x : float
            x-extent of the top edge; ``0`` gives a triangular wedge,
            ``dx`` an ordinary box
        position : tuple[float, float, float]
            lower-left corner of the base (defaults to origin)
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override

        Returns
        -------
        GeoObject
            volume
        """
        oid = self._native.add_wedge([dx, dy, dz], top_x, list(position), maxh)
        return self._wrap(oid, sheet=False, material=material, maxh=maxh)

    def torus(self, major_radius: float, minor_radius: float,
              position: tuple[float, float, float] = (0, 0, 0),
              angle: float = 2 * math.pi,
              *,
              material=None,
              maxh: float | None = None,
              center: tuple[float, float, float] | None = None) -> "GeoObject":
        """add a torus primitive

        Parameters
        ----------
        major_radius : float
            donut radius (tube-centre to torus-axis distance) in metres
        minor_radius : float
            tube radius in metres
        position : tuple[float, float, float]
            torus centre (defaults to origin); axis is along +z
        angle : float
            sweep angle in radians; :math:`<2\\pi` gives a partial torus
        material : rapidfem.Material, optional
            volume material
        maxh : float, optional
            per-volume mesh size override
        center : tuple[float, float, float], optional
            deprecated alias for ``position``

        Returns
        -------
        GeoObject
            volume
        """
        _full_turn(angle, "torus")
        c = center if center is not None else position
        oid = self._native.add_torus(major_radius, minor_radius, list(c), [0.0, 0.0, 1.0], maxh)
        return self._wrap(oid, sheet=False, material=material, maxh=maxh)

    # ── sheets ──────────────────────────────────────────────────────────────

    def _sheet(self, corner, u, v, maxh):
        oid = self._native.add_plate(list(corner), list(u), list(v), maxh)
        c, u, v = (np.asarray(x, dtype=float) for x in (corner, u, v))
        self._profile(oid, [c, c + u, c + u + v, c + v])
        return self._wrap(oid, sheet=True, maxh=maxh)

    def xy_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0),
                 *,
                 maxh: float | None = None) -> "GeoObject":
        """add a thin rectangular plate in the xy-plane

        2-D primitive, used for thin conductors like patch antennas,
        microstrip traces, and lumped-port footprints. The returned
        object carries dim = 2 and a single ``.faces`` selector that
        points at itself.


        Note
        ----
        ``height`` here is the y-extent, *not* a vertical (z) extent.
        For an arbitrarily oriented plate (e.g. a vertical feed sheet)
        use :meth:`plate` with explicit width/height vectors.


        Example
        -------
        A patch antenna on top of a substrate:

        .. code-block:: python

            patch = g.xy_plate(38e-3, 29e-3,
                               position=(-19e-3, -14.5e-3, SUB_H))
            rf.PEC(patch)


        Parameters
        ----------
        width : float
            x-extent in metres
        height : float
            y-extent in metres
        position : tuple[float, float, float]
            lower corner (defaults to origin)
        maxh : float, optional
            per-plate mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        return self._sheet(position, (width, 0, 0), (0, height, 0), maxh)

    def xz_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0),
                 *,
                 maxh: float | None = None) -> "GeoObject":
        """add a thin rectangular plate in the xz-plane

        Convenience wrapper around :meth:`plate` for the most common
        axis-aligned vertical plate. ``width`` runs along x,
        ``height`` along z.


        Parameters
        ----------
        width : float
            x-extent in metres
        height : float
            z-extent in metres
        position : tuple[float, float, float]
            lower corner (defaults to origin)
        maxh : float, optional
            per-plate mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        return self._sheet(position, (width, 0, 0), (0, 0, height), maxh)

    def yz_plate(self, width: float, height: float,
                 position: tuple[float, float, float] = (0, 0, 0),
                 *,
                 maxh: float | None = None) -> "GeoObject":
        """add a thin rectangular plate in the yz-plane

        Convenience wrapper around :meth:`plate` for the most common
        axis-aligned vertical plate. ``width`` runs along y,
        ``height`` along z.


        Parameters
        ----------
        width : float
            y-extent in metres
        height : float
            z-extent in metres
        position : tuple[float, float, float]
            lower corner (defaults to origin)
        maxh : float, optional
            per-plate mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        return self._sheet(position, (0, width, 0), (0, 0, height), maxh)

    def plate(self, p0: tuple[float, float, float],
              width: tuple[float, float, float],
              height: tuple[float, float, float],
              *,
              maxh: float | None = None) -> "GeoObject":
        """add a thin rectangular plate at arbitrary orientation

        Used for vertical lumped-port sheets, oblique feed plates, and
        any flat 2-D region whose sides are not axis-aligned.


        Note
        ----
        The plate is the parallelogram spanned by the two edge vectors
        ``width`` and ``height``; they should be orthogonal, if they are
        not, you get a planar parallelogram, not a rectangle.


        Example
        -------
        Vertical lumped-port plate bridging substrate to a trace:

        .. code-block:: python

            port = g.plate(
                p0=(FEED_X - W/2, FEED_Y, 0),
                width=(W, 0, 0),
                height=(0, 0, SUB_H),
            )


        Parameters
        ----------
        p0 : tuple[float, float, float]
            one corner of the rectangle
        width : tuple[float, float, float]
            edge vector from ``p0`` defining one side
        height : tuple[float, float, float]
            edge vector from ``p0`` defining the perpendicular side
        maxh : float, optional
            per-plate mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        return self._sheet(p0, width, height, maxh)

    def polygon(self, points: Iterable[tuple[float, ...]],
                position: tuple[float, float, float] = (0, 0, 0),
                *,
                holes: "list[list[tuple]] | None" = None,
                maxh: float | None = None) -> "GeoObject":
        """add a planar polygon face

        2-D primitive for arbitrary outlines, combine with
        :meth:`extrude` for a non-axis-aligned trace, :meth:`revolve`
        for an axisymmetric solid, or :meth:`loft` to bridge two
        profiles into a horn-style frustum.


        Note
        ----
        2-tuple vertices are placed in the xy-plane at ``z = 0`` plus
        the ``position`` offset; 3-tuple vertices must all be coplanar,
        non-planar input raises.


        Example
        -------
        Rectangular waveguide aperture (yz-plane at ``x = L``) for a
        horn loft:

        .. code-block:: python

            aperture = g.polygon([
                (L, -W/2, -H/2), (L,  W/2, -H/2),
                (L,  W/2,  H/2), (L, -W/2,  H/2),
            ])


        Parameters
        ----------
        points : iterable of (x, y) or (x, y, z) tuples
            vertices in CCW order; polygon closes automatically
        position : tuple[float, float, float]
            offset added to every vertex (defaults to origin)
        maxh : float, optional
            per-face mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        pts = [tuple(float(v) for v in p) for p in points]
        if len(pts) < 3:
            raise ValueError("polygon needs at least 3 vertices")
        x0, y0, z0 = (float(v) for v in position)
        p3 = np.array([(p[0] + x0, p[1] + y0, (p[2] if len(p) == 3 else 0.0) + z0)
                       for p in pts])
        hole3 = [np.array([(h[0] + x0, h[1] + y0, (h[2] if len(h) == 3 else 0.0) + z0)
                           for h in hole]) for hole in (holes or [])]
        z = p3[:, 2]
        if np.ptp(z) <= 1e-12 * max(1.0, np.abs(p3).max()):
            oid = self._native.add_polygon(
                [list(p[:2]) for p in p3], float(z[0]),
                [[list(p[:2]) for p in h] for h in hole3], maxh)
            self._profile(oid, p3)
            return self._wrap(oid, sheet=True, maxh=maxh)
        # Off the xy plane: a parallelogram is a plate.
        if len(p3) == 4 and not hole3 and np.allclose(p3[0] + p3[2], p3[1] + p3[3]):
            return self._sheet(p3[0], p3[1] - p3[0], p3[3] - p3[0], maxh)
        raise NotImplementedError(
            "polygon: a general polygon off the xy plane is not available yet "
            "(milanofthe/rapidmesh-dev#140); parallelograms work in any plane")

    def disc(self, radius: float,
             position: tuple[float, float, float] = (0, 0, 0),
             *,
             axis: tuple[float, float, float] = (0, 0, 1),
             maxh: float | None = None) -> "GeoObject":
        """add a circular face with an arbitrary normal

        An exact circle: the mesh follows the curved rim at the size the
        rim's curvature asks for. Pair with
        :meth:`extrude` for a circular post or :meth:`revolve` for a
        spherical cap.


        Parameters
        ----------
        radius : float
            disc radius in metres
        position : tuple[float, float, float]
            disc centre (defaults to origin)
        axis : tuple[float, float, float]
            disc normal (defaults to +z, i.e. the xy-plane). Any direction
            is allowed, e.g. ``(1, 0, 0)`` puts the disc in the yz-plane.
            Need not be unit length.
        maxh : float, optional
            per-face mesh size override

        Returns
        -------
        GeoObject
            2-D face
        """
        oid = self._native.add_disc(radius, list(position), list(axis), maxh)
        return self._wrap(oid, sheet=True, maxh=maxh)
