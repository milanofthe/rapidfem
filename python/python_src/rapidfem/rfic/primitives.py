# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Hand-coded RFIC primitives, for layouts NOT coming from GDS:
``microstrip``, ``via``, ``trace_port``, ``gsg_port``, ``differential_port``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Literal

from .stack import Stack

if TYPE_CHECKING:
    from rapidfem.geometry import Geometry, GeoObject


def microstrip(
    g: "Geometry",
    stack: Stack,
    *,
    layer: str,
    width: float,
    length: float,
    position: tuple[float, float],
    orientation: Literal["x", "y"] = "x",
    thick: bool = False,
) -> "GeoObject":
    """A metal trace on a named PDK layer. ``thick=True`` extrudes the
    trace as a 3D box of the layer's thickness; otherwise a 2D plate at the
    layer's bottom z (typical for thin-conductor approximation).
    """
    pdk_layer = stack.by_name(layer)
    if pdk_layer.type != "metal":
        raise ValueError(f"layer {layer!r} is type {pdk_layer.type!r}, not metal")
    x, y = position
    z = pdk_layer.z
    if orientation == "x":
        w_x, w_y = length, width
    else:
        w_x, w_y = width, length

    if thick:
        return g.box(w_x, w_y, pdk_layer.thickness, position=(x, y, z))
    return g.xy_plate(w_x, w_y, position=(x, y, z))


def via(
    g: "Geometry",
    stack: Stack,
    *,
    from_layer: str,
    to_layer: str,
    radius: float,
    position: tuple[float, float],
) -> "GeoObject":
    """A metal via cylinder spanning from ``from_layer`` (bottom) to
    ``to_layer`` (top). Radius in meters."""
    a = stack.by_name(from_layer)
    b = stack.by_name(to_layer)
    z0 = min(a.z, b.z)
    z1 = max(a.z_top, b.z_top)
    height = z1 - z0
    x, y = position
    return g.cylinder(radius=radius, height=height, position=(x, y, z0))


@dataclass
class TracePort:
    """Result of `rfic.trace_port`: extension pad on the trace layer, a ground
    patch below, and a vertical port plate. Both top and bottom edges of the
    port plate fully touch PEC (extension pad above, ground pad below) so the
    lumped-port BC sees a clean voltage gap.

    Wire it as::

        tp = rfic.trace_port(g, stack, layer="met5", position=(...))
        rf.PEC(trace, tp.trace_extension, tp.ground_pad)
        rf.LumpedPort(tp.port_plate, direction=(0, 0, 1), z0=50.0)
    """
    trace_extension: "GeoObject"   # small pad welded onto the trace at port location
    ground_pad: "GeoObject"
    port_plate: "GeoObject"


def trace_port(
    g: "Geometry",
    stack: Stack,
    *,
    layer: str,
    position: tuple[float, float],
    gnd_layer: str = "li1",
    extension_size: float = 4e-6,
    fragment_with: Iterable["GeoObject"] = (),
) -> TracePort:
    """Place a vertical lumped-port plate at a trace's edge with proper
    PEC references on both ends.

    Geometry:
      1. A small extension pad on the trace's `layer` (e.g. met5) at `position`,
         guarantees the port plate's TOP edge sits fully on PEC.
      2. A ground pad on `gnd_layer` (e.g. li1) directly below, anchor for
         the port plate's BOTTOM edge.
      3. A vertical port plate spanning from gnd_layer top to trace_layer bottom.

    Pass any volumes that the port should fragment with via `fragment_with`,
    typically the oxide block and the trace volume (so all three become
    conformal at the port boundary).
    """
    pdk_trace = stack.by_name(layer)
    pdk_gnd = stack.by_name(gnd_layer)
    z_trace = pdk_trace.z
    z_gnd_top = pdk_gnd.z_top   # ground patch top z (where port plate's bottom edge lands)
    cx, cy = position
    half = extension_size / 2

    # 1. Extension pad on the trace layer, co-PEC with the rest of the trace,
    # so the port plate's top edge always lands on metal.
    extension = g.box(extension_size, extension_size, pdk_trace.thickness,
                      position=(cx - half, cy - half, z_trace))

    # 2. Ground pad
    ground = g.xy_plate(extension_size, extension_size,
                        position=(cx - half, cy - half, z_gnd_top))

    # 3. Port plate, a 2D rectangle in the yz-plane at x=cx, spanning the gap
    port_plate = g.plate(
        p0=(cx, cy - half, z_gnd_top),
        width=(0, extension_size, 0),
        height=(0, 0, z_trace - z_gnd_top),
    )

    # 4. Fragment with surrounding volumes so all interfaces are conformal.
    # Always fragment ground+port with oxide at minimum.
    if fragment_with:
        first = list(fragment_with)[0]
        rest = list(fragment_with)[1:] + [extension, ground, port_plate]
        g.fragment(first, *rest)

    return TracePort(trace_extension=extension, ground_pad=ground, port_plate=port_plate)


@dataclass
class GsgPort:
    signal_pad: "GeoObject"
    ground_pads: tuple["GeoObject", "GeoObject"]
    port_plate: "GeoObject"


def gsg_port(
    g: "Geometry",
    stack: Stack,
    *,
    layer: str,
    center: tuple[float, float],
    pad_size: float = 50e-6,
    pitch: float = 100e-6,
) -> GsgPort:
    """Coplanar Ground-Signal-Ground probe pad on a named metal layer.

    Three coplanar pads (signal centered, two grounds at ``±pitch``) plus a
    vertical lumped-port plate spanning the signal-to-ground gap. Wire up as::

        gp = rfic.gsg_port(g, stack, layer="met5", center=(0, 0))
        rf.PEC(gp.signal_pad, *gp.ground_pads)
        rf.LumpedPort(gp.port_plate, direction=(1, 0, 0), z0=50.0)
    """
    pdk_layer = stack.by_name(layer)
    z_metal = pdk_layer.z
    cx, cy = center
    half = pad_size / 2

    sig = g.xy_plate(pad_size, pad_size, position=(cx - half, cy - half, z_metal))
    gleft = g.xy_plate(pad_size, pad_size,
                       position=(cx - pitch - half, cy - half, z_metal))
    gright = g.xy_plate(pad_size, pad_size,
                        position=(cx + pitch - half, cy - half, z_metal))

    # Lumped-port plate from signal → right ground at the metal layer
    port_x = cx + half
    port_plate = g.plate(
        p0=(port_x, cy - half, z_metal),
        width=(pitch - pad_size, 0, 0),
        height=(0, pad_size, 0),
    )
    return GsgPort(signal_pad=sig, ground_pads=(gleft, gright), port_plate=port_plate)


@dataclass
class DifferentialPort:
    pad_plus: "GeoObject"
    pad_minus: "GeoObject"
    port_plate: "GeoObject"


def differential_port(
    g: "Geometry",
    stack: Stack,
    *,
    layer: str,
    center: tuple[float, float],
    pad_size: float = 50e-6,
    gap: float = 30e-6,
) -> DifferentialPort:
    """Two coplanar pads with a lumped port between them, basic balanced feed."""
    pdk_layer = stack.by_name(layer)
    z_metal = pdk_layer.z
    cx, cy = center
    half = pad_size / 2
    pitch = pad_size + gap
    pad_plus = g.xy_plate(pad_size, pad_size,
                          position=(cx - pitch / 2 - half, cy - half, z_metal))
    pad_minus = g.xy_plate(pad_size, pad_size,
                           position=(cx + pitch / 2 - half, cy - half, z_metal))
    port_plate = g.plate(
        p0=(cx - pitch / 2 + half, cy - half, z_metal),
        width=(gap, 0, 0),
        height=(0, pad_size, 0),
    )
    return DifferentialPort(pad_plus=pad_plus, pad_minus=pad_minus, port_plate=port_plate)


__all__ = [
    "microstrip", "via", "trace_port", "gsg_port", "differential_port",
    "TracePort", "GsgPort", "DifferentialPort",
]
