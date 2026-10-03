# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""One-call RFIC model builder: GDS + Stack in, solve-ready geometry out.

``build()`` reproduces the gds2palace modelling conventions that the SG13G2
measurement validation pinned down, so a gds2palace user gets the same model
shape by default:

- the background dielectric slabs span the layout bbox plus ``margin``
- an ``air`` shell wraps the whole dielectric stack on all six sides, with
  an absorbing boundary on its outer faces (optionally a PEC floor for
  backside metallisation)
- metals become SIBC shells (two-sided, finite-thickness corrected),
  via arrays become homogenised volume conductors with anisotropic
  conductivity, LOWLOSS layers become PEC
- ports are vertical lumped plates ("via ports"), placed explicitly or
  read from GDS marker layers

The result is NOT meshed: inspect/tune, then call ``model.geometry.mesh()``
and check ``model.geometry.mesh_stats`` before committing to a sweep.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rapidfem import _native
from rapidfem._native import VIA_LATERAL_FACTOR, MeshSpec

from .stack import Stack

if TYPE_CHECKING:
    from rapidfem.geometry import Geometry, GeoObject


@dataclass
class ViaPort:
    """A vertical lumped-port plate between two stack heights.

    ``z`` bounds accept either a height in meters or a layer name: a name
    for the LOWER bound resolves to that layer's TOP face (ground reference
    top), a name for the UPPER bound to that layer's BOTTOM face (signal
    feed underside), which is the gds2palace via-port convention.

    Placement: either ``span``+``at`` explicitly, or ``marker`` to read the
    plate footprint from a GDS marker-layer rectangle (the wide axis of the
    rectangle is the plate width, the thin axis its position).
    """
    z: tuple[float | str, float | str]
    span: tuple[float, float] | None = None    # (a0, a1) along the wide axis, m
    at: float | None = None                    # position on the thin axis, m
    axis: str = "x"                            # wide axis of the plate ("x"|"y")
    marker: int | None = None                  # GDS layer number of the marker
    z0: float = 50.0                           # port reference impedance, Ohm


@dataclass
class BuiltModel:
    """Everything :func:`build` produced, with named handles for follow-up
    physics (extra BCs, experiments) before meshing."""
    geometry: "Geometry"
    stack: Stack
    conductors: dict[str, list["GeoObject"]]   # layer name -> volumes
    slabs: dict[str, list["GeoObject"]]        # dielectric name -> boxes (>1 if graded)
    air_shell: list["GeoObject"]               # 6 enclosure boxes
    ports: list["GeoObject"]                   # port plates, LumpedPort applied
    footprint: tuple[float, float, float, float]  # (x0, y0, x1, y1) incl. margin


def build(
    gds: str,
    stack: Stack,
    *,
    top_cell: str | None = None,
    ports: "tuple[ViaPort, ...] | list[ViaPort]" = (),
    margin: float = 150e-6,
    air: float = 50e-6,
    air_top: float | None = None,
    pec_floor: bool = False,
    conductor_model: dict[str, str] | None = None,
    band: tuple[float, float] | None = None,
    mesh: "MeshSpec | str | None" = None,
    passivation: str = "planar",
    pass_t_side: float = 0.6e-6,
    pass_t_top: float | None = None,
    conformal_over: str | None = None,
    boundary: str = "abc",
) -> BuiltModel:
    """Build a solve-ready FEM model from a GDS and a full process stack.

    Parameters
    ----------
    gds : str
        Path to the layout. Every stack layer present in the GDS is
        extruded; port marker layers are looked up here too.
    stack : Stack
        Full process stack, background ``dielectrics`` populated (use
        ``Stack.from_xml`` / a preset). Raises if the dielectric stack is
        empty, the enclosure needs it.
    ports : sequence of ViaPort
        Vertical lumped ports; see :class:`ViaPort`.
    margin : float
        Lateral extension of the dielectric slabs beyond the layout bbox.
    air : float
        Thickness of the air shell wrapped around the dielectric stack.
    air_top : float, optional
        Height of the air region above the stack. Defaults to the topmost
        air-like slab's own thickness (from the XML), or ``air``.
    pec_floor : bool
        PEC under the shell floor (chuck / backside metallisation)
        instead of an absorbing boundary.
    conductor_model : dict, optional
        Per-layer override of the conductor treatment: layer name ->
        ``"auto" | "sibc" | "pec" | "volume" | "volume_iso"``. Defaults:
        metals -> auto, vias -> volume (anisotropic), LOWLOSS -> pec.
        ``"sibc"`` and ``"pec"`` conductors are holes in the mesh whose walls
        carry the boundary condition; the SIBC thickness of a layer is ``2V/S``
        (its volume over its wall area), which reproduces the DC resistance.
        ``"auto"`` picks per layer from the thickness-to-skin-depth ratio over
        ``band``: SIBC where ``t/delta < 1.5`` or ``> 4`` across the whole band
        (the surface model, with its edge correction, is within a few percent
        there), otherwise an isotropic volume conductor meshed at the skin
        depth. A trace narrower than 4 skin depths keeps the uncorrected
        surface impedance, up to about 17 % low.
    band : (f_min, f_max), optional
        Frequency band the model will be solved over, in Hz. Drives the
        ``"auto"`` conductor choice and the volume-conductor mesh size. Without
        it ``"auto"`` falls back to SIBC with a warning: between 1.5 and 4 skin
        depths a surface impedance underestimates the strip resistance by up to
        about 25 %.
    mesh : MeshSpec or {"fast", "balanced", "accurate"}, optional
        Mesh sizing policy. A preset name (or the default ``None``, which
        means ``"balanced"``) derives every size from the stack and the
        layers actually drawn in the GDS, see ``MeshSpec.derive``.
        Pass a `MeshSpec` to take full control instead.
    passivation : {"planar", "conformal", "none"}
        "planar" keeps the stackup-XML sheet (the gds2palace / Momentum
        approximation). "conformal" models the real deposition: the oxide
        stops at the top metal's bottom, the passivation drapes over the
        exposed metal (``pass_t_top`` on top and field, ``pass_t_side`` on
        the sidewalls) with air beyond, built as disjoint prisms from a 2D
        offset decomposition of the metal polygons. "none" drops the sheet.
    pass_t_side : float
        Sidewall passivation thickness for the conformal mode.
    pass_t_top : float, optional
        Top/field passivation thickness; defaults to the XML sheet's own
        thickness.
    conformal_over : str, optional
        Layer name the passivation drapes over; defaults to the topmost
        metal present in the stack.
    boundary : {"abc", "pml"}
        Outer termination: first-order absorbing boundary (default), or a
        PML declared on each of the six air-shell boxes.

    Returns
    -------
    BuiltModel, un-meshed; call ``model.geometry.mesh()`` next.
    """
    from rapidfem.geometry import Geometry

    native, b = _native.rfic_build(
        str(gds), stack, top_cell=top_cell, ports=list(ports), margin=margin,
        air=air, air_top=air_top, pec_floor=pec_floor,
        conductor_model=conductor_model, band=band, mesh=mesh,
        passivation=passivation, pass_t_side=pass_t_side, pass_t_top=pass_t_top,
        conformal_over=conformal_over, boundary=boundary)
    for w in b["warnings"]:
        warnings.warn(w, stacklevel=2)
    g = Geometry._adopt(native)
    objs = g._objects
    return BuiltModel(
        geometry=g, stack=stack,
        conductors={k: objs(v) for k, v in b["conductors"].items()},
        slabs={k: objs(v) for k, v in b["slabs"].items()},
        air_shell=objs(b["air_shell"]), ports=objs(b["ports"]),
        footprint=tuple(b["footprint"]))


__all__ = ["build", "BuiltModel", "MeshSpec", "ViaPort", "VIA_LATERAL_FACTOR"]
