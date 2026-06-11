"""
RFIC builder for rapidfem, PDK-grade stack definitions, GDS-driven extrusion
helpers, and hand-coded primitives (microstrip, via, GSG port).

The `Stack` data model mirrors the schema used by ``rapidpassives`` so that
the same JSON describes geometry on both sides:

- Each `PdkLayer` carries (name, gds, datatype, z, thickness, color, type)
- `Stack.from_pdk("sky130")` returns a fully-populated stack ready for GDS
  loading and FEM extrusion
- `stack.to_dict()` round-trips to the rapidpassives `Pdk` JSON shape

Typical workflow::

    import rapidfem as rf
    import rapidfem.rfic as rfic

    stack = rfic.Stack.sky130()                       # PDK preset
    g = rf.Geometry(maxh=20e-6)
    rfic.from_gds(g, "inductor.gds", stack=stack, top_cell="ind_3turn")
    subs = stack.create_substrate(g, footprint=(400e-6, 400e-6))
    air = g.box(400e-6, 400e-6, 200e-6,               # ABC enclosure
                position=(-200e-6, -200e-6, stack.top_z),
                material=rf.Air())
    rf.PEC(...)                                       # trace + ground BCs
    rf.LumpedPort(...)
    g.mesh()
    result = rf.Problem(g).sweep([1e9, 5e9, 10e9])

For hand-coded layouts, primitives live below: ``microstrip``, ``via``,
``gsg_port``, ``differential_port``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING, Iterable, Literal

from rapidfem._geometry_gds import from_gds  # noqa: F401 (re-exported)

if TYPE_CHECKING:
    from rapidfem.geometry import Geometry, GeoObject


# ─────────────────────────────────────────────────────────────────────────────
# Stack data model, mirrors rapidpassives' PdkLayer / Pdk shape
# ─────────────────────────────────────────────────────────────────────────────

LayerType = Literal["metal", "via", "poly", "diffusion", "substrate", "oxide", "other"]


@dataclass
class PdkLayer:
    """One physical layer in a process stack.

    Mirrors ``rapidpassives.web.lib.stack.pdk.PdkLayer`` 1:1 so the same JSON
    describes a stack on both rapidfem (FEM-side) and rapidpassives (viewer-side).

    Coordinate convention: z is the BOTTOM of the layer (lower face). The top
    is at z + thickness. All distances in meters (rapidpassives uses microns;
    converters below help round-trip).
    """
    name: str
    gds: int
    datatype: int
    z: float                # bottom z, in meters
    thickness: float        # in meters
    color: str = "#888"
    type: LayerType = "metal"
    # Material defaults (FEM-relevant; rapidpassives ignores these)
    er: float = 1.0
    ur: float = 1.0
    tand: float = 0.0
    sigma: float = 0.0       # bulk conductivity (S/m); 0 ⇒ treat as PEC for metals

    @property
    def z_top(self) -> float:
        return self.z + self.thickness

    @property
    def gds_key(self) -> tuple[int, int]:
        return (self.gds, self.datatype)


@dataclass
class Stack:
    """A complete process stack: ordered physical layers + substrate properties.

    Use ``Stack.sky130()`` / ``Stack.sg13g2()`` for built-in presets, or
    construct manually from a list of `PdkLayer`s.
    """
    name: str
    layers: list[PdkLayer]
    # Substrate slab below the lowest layer (silicon wafer).
    substrate_thickness: float = 300e-6   # m
    substrate_er: float = 11.9
    substrate_sigma: float = 10.0          # S/m (lossy silicon)
    # Bulk dielectric between metals (often modeled as a single effective er)
    oxide_er: float = 4.2
    oxide_tand: float = 0.0

    # ── Construction helpers ───────────────────────────────────────────────

    def __post_init__(self):
        # Sort layers bottom-to-top by z for deterministic iteration
        self.layers = sorted(self.layers, key=lambda l: (l.z, l.thickness))

    @staticmethod
    def from_pdk(name: str) -> "Stack":
        """Convenience: dispatch by lowercased PDK name."""
        normalized = name.lower().replace("-", "").replace("_", "")
        if normalized in ("sky130", "skywater130"):
            return Stack.sky130()
        if normalized in ("sg13g2", "ihpsg13g2"):
            return Stack.sg13g2()
        raise ValueError(f"unknown PDK {name!r}; available: sky130, sg13g2")

    @staticmethod
    def sky130() -> "Stack":
        """SkyWater SKY130 open-source PDK process stack.

        Layer numbering and z-positions match the rapidpassives `pdk.ts` file
        (z=0 at the bottom of li1, polysilicon below at z=-0.18 μm).
        """
        um = 1e-6
        layers = [
            # name        gds  dt   z(μm)  t(μm)   color    type      er    sigma
            PdkLayer("poly",   66, 20, -0.18 * um, 0.18 * um, "#c4725e", "poly",     er=4.2),
            PdkLayer("licon1", 66, 44, -0.10 * um, 0.10 * um, "#5a5a62", "via",      sigma=4.1e7),
            PdkLayer("li1",    67, 20,  0.00 * um, 0.10 * um, "#7b5e8a", "metal",    sigma=4.1e7),
            PdkLayer("mcon",   67, 44,  0.10 * um, 0.27 * um, "#5a5a62", "via",      sigma=4.1e7),
            PdkLayer("met1",   68, 20,  0.37 * um, 0.36 * um, "#6bbf8a", "metal",    sigma=4.1e7),
            PdkLayer("via",    68, 44,  0.73 * um, 0.27 * um, "#5a5a62", "via",      sigma=4.1e7),
            PdkLayer("met2",   69, 20,  1.00 * um, 0.36 * um, "#4a9ec2", "metal",    sigma=4.1e7),
            PdkLayer("via2",   69, 44,  1.36 * um, 0.42 * um, "#6e6e78", "via",      sigma=4.1e7),
            PdkLayer("met3",   70, 20,  1.78 * um, 0.845 * um, "#5aad78", "metal",   sigma=4.1e7),
            PdkLayer("via3",   70, 44,  2.625 * um, 0.39 * um, "#6e6e78", "via",     sigma=4.1e7),
            PdkLayer("met4",   71, 20,  3.015 * um, 0.845 * um, "#d9513c", "metal",  sigma=4.1e7),
            PdkLayer("via4",   71, 44,  3.86 * um, 0.505 * um, "#7a7a84", "via",     sigma=4.1e7),
            PdkLayer("met5",   72, 20,  4.365 * um, 1.26 * um, "#e8944a", "metal",   sigma=4.1e7),
        ]
        return Stack(
            name="SKY130", layers=layers,
            substrate_thickness=300 * um,
            substrate_er=11.9, substrate_sigma=10.0,
            oxide_er=4.2, oxide_tand=0.0,
        )

    @staticmethod
    def sg13g2() -> "Stack":
        """IHP SG13G2 open-source 130 nm SiGe BiCMOS PDK process stack."""
        um = 1e-6
        layers = [
            PdkLayer("GatPoly", 5,  0, -0.16 * um, 0.16 * um, "#c4725e", "poly",   er=4.2),
            PdkLayer("Metal1",  8,  0,  0.00 * um, 0.42 * um, "#7b5e8a", "metal",  sigma=4.1e7),
            PdkLayer("Via1",    19, 0,  0.42 * um, 0.54 * um, "#5a5a62", "via",    sigma=4.1e7),
            PdkLayer("Metal2",  10, 0,  0.96 * um, 0.49 * um, "#6bbf8a", "metal",  sigma=4.1e7),
            PdkLayer("Via2",    29, 0,  1.45 * um, 0.54 * um, "#6e6e78", "via",    sigma=4.1e7),
            PdkLayer("Metal3",  30, 0,  1.99 * um, 0.49 * um, "#4a9ec2", "metal",  sigma=4.1e7),
            PdkLayer("Via3",    49, 0,  2.48 * um, 0.54 * um, "#6e6e78", "via",    sigma=4.1e7),
            PdkLayer("Metal4",  50, 0,  3.02 * um, 0.49 * um, "#5aad78", "metal",  sigma=4.1e7),
            PdkLayer("Via4",    66, 0,  3.51 * um, 0.54 * um, "#7a7a84", "via",    sigma=4.1e7),
            PdkLayer("Metal5",  67, 0,  4.05 * um, 0.49 * um, "#d9513c", "metal",  sigma=4.1e7),
            PdkLayer("TopVia1", 125, 0, 4.54 * um, 0.85 * um, "#7a7a84", "via",    sigma=4.1e7),
            PdkLayer("TopMetal1", 126, 0, 5.39 * um, 2.0 * um, "#e8944a", "metal", sigma=4.1e7),
            PdkLayer("TopVia2", 133, 0, 7.39 * um, 2.85 * um, "#7a7a84", "via",    sigma=4.1e7),
            PdkLayer("TopMetal2", 134, 0, 10.24 * um, 3.0 * um, "#e8944a", "metal",sigma=4.1e7),
        ]
        return Stack(
            name="SG13G2", layers=layers,
            substrate_thickness=200 * um,
            substrate_er=11.9, substrate_sigma=20.0,   # SG13G2 is moderately resistive
            oxide_er=4.1, oxide_tand=0.0,
        )

    # ── Lookups ────────────────────────────────────────────────────────────

    def by_name(self, name: str) -> PdkLayer:
        for l in self.layers:
            if l.name == name:
                return l
        raise KeyError(f"layer {name!r} not in stack {self.name!r}; available: "
                       f"{[l.name for l in self.layers]}")

    def by_gds(self, gds: int, datatype: int = 0) -> PdkLayer | None:
        """Find the layer matching a GDS (number, datatype) tuple. None if absent."""
        for l in self.layers:
            if l.gds == gds and l.datatype == datatype:
                return l
        return None

    def metals(self) -> list[PdkLayer]:
        return [l for l in self.layers if l.type == "metal"]

    def vias(self) -> list[PdkLayer]:
        return [l for l in self.layers if l.type == "via"]

    @property
    def top_z(self) -> float:
        return max((l.z_top for l in self.layers), default=0.0)

    @property
    def bottom_z(self) -> float:
        """z of the highest substrate top (where active devices sit). Substrate
        slab itself sits below this at [bottom_z - substrate_thickness, bottom_z]."""
        return min((l.z for l in self.layers), default=0.0)

    # ── JSON interop with rapidpassives ────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to the rapidpassives `Pdk` JSON shape (lengths in microns)."""
        um = 1e-6
        return {
            "id": self.name.lower(),
            "name": self.name,
            "description": f"rapidfem stack: {self.name}",
            "substrate": {
                "thickness_um": self.substrate_thickness / um,
                "er": self.substrate_er,
                "sigma": self.substrate_sigma,
            },
            "oxide": {"er": self.oxide_er, "tand": self.oxide_tand},
            "layers": [
                {
                    "name": l.name, "gds": l.gds, "datatype": l.datatype,
                    "z_um": l.z / um, "thickness_um": l.thickness / um,
                    "color": l.color, "type": l.type,
                    "er": l.er, "ur": l.ur, "tand": l.tand, "sigma": l.sigma,
                }
                for l in self.layers
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "Stack":
        um = 1e-6
        sub = d.get("substrate", {})
        ox = d.get("oxide", {})
        layers = [
            PdkLayer(
                name=l["name"], gds=l["gds"], datatype=l["datatype"],
                z=l["z_um"] * um, thickness=l["thickness_um"] * um,
                color=l.get("color", "#888"), type=l.get("type", "metal"),
                er=l.get("er", 1.0), ur=l.get("ur", 1.0),
                tand=l.get("tand", 0.0), sigma=l.get("sigma", 0.0),
            )
            for l in d["layers"]
        ]
        return Stack(
            name=d["name"], layers=layers,
            substrate_thickness=sub.get("thickness_um", 300) * um,
            substrate_er=sub.get("er", 11.9), substrate_sigma=sub.get("sigma", 10.0),
            oxide_er=ox.get("er", 4.2), oxide_tand=ox.get("tand", 0.0),
        )

    # ── Geometry helpers ───────────────────────────────────────────────────

    def create_substrate(
        self,
        g: "Geometry",
        footprint: tuple[float, float],
        center: bool = True,
        z_substrate_top: float | None = None,
        fragment_existing: bool = True,
    ) -> dict[str, "GeoObject"]:
        """Instantiate the silicon substrate slab and a bulk-oxide slab spanning
        from the substrate top to the stack's top.

        Each block is created with a fully-instantiated ``rf.Dielectric``
        derived from the stack constants, drop the returned objects straight
        into the Problem API, no extra material wiring needed.

        Returns a dict of named GeoObjects (`substrate`, `oxide`).
        ``fragment_existing`` is retained for API compatibility; in the
        rapidmesh backend fragment() is a no-op (all primitives are arranged
        exactly conformal by construction without explicit fragmentation).
        """
        # Local import, avoids a circular at module load (rapidfem.__init__
        # imports rapidfem.rfic).
        from rapidfem.materials import Dielectric

        wx, wy = footprint
        x0 = -wx / 2 if center else 0.0
        y0 = -wy / 2 if center else 0.0
        z_top = z_substrate_top if z_substrate_top is not None else self.bottom_z

        silicon = Dielectric(er=self.substrate_er, conductivity=self.substrate_sigma)
        sio2 = Dielectric(er=self.oxide_er, tand=self.oxide_tand)

        sub = g.box(wx, wy, self.substrate_thickness,
                    position=(x0, y0, z_top - self.substrate_thickness),
                    material=silicon)
        sub.name = "substrate"

        oxide_height = self.top_z - z_top
        ox = None
        if oxide_height > 0:
            ox = g.box(wx, wy, oxide_height, position=(x0, y0, z_top),
                       material=sio2)
            ox.name = "oxide"

        # fragment() is a no-op in the rapidmesh backend; the call is
        # retained so fragment_existing=False callers behave correctly.
        if fragment_existing:
            g.fragment(sub)

        return {"substrate": sub} | ({"oxide": ox} if ox is not None else {})


# ─────────────────────────────────────────────────────────────────────────────
# Hand-coded primitives, for layouts NOT coming from GDS
# ─────────────────────────────────────────────────────────────────────────────

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
    z_lo = min(a.z_top, b.z_top)
    z_hi = max(a.z, b.z)
    # If layers overlap in z, just use a→b directly
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


# ─────────────────────────────────────────────────────────────────────────────
# FEM-JSON bridge, consume rapidpassives' exportForFEM() JSON
# ─────────────────────────────────────────────────────────────────────────────

FEM_JSON_SCHEMA_VERSIONS = (1,)

# Polygons coming out of mergeLayers may carry sliver edges (sub-nm jogs at
# rectangle-to-rectangle joins). Drop any vertex closer than this to its
# predecessor in xy, well below the geometric tolerance that gmsh OCC
# rejects on `addLine`, but large enough to wipe merge slivers.
_FEM_JSON_VERTEX_TOL_UM = 0.01


def _clean_polygon_um(poly_um: list) -> list:
    """Drop consecutive vertices closer than `_FEM_JSON_VERTEX_TOL_UM` in xy.

    rapidpassives' mergeLayers occasionally emits near-duplicate vertices at
    polygon joins (sub-nm slivers); gmsh's OCC kernel refuses to build a
    line for those. Closing-vertex duplicates are also stripped, gmsh's
    polygon helper closes the loop itself, an explicit trailing copy of
    the first vertex would produce a zero-length segment.
    """
    if not poly_um:
        return []
    tol_sq = _FEM_JSON_VERTEX_TOL_UM ** 2
    out = [tuple(poly_um[0])]
    for x, y in poly_um[1:]:
        px, py = out[-1]
        if (x - px) ** 2 + (y - py) ** 2 < tol_sq:
            continue
        out.append((x, y))
    # Strip closing copy of the first vertex if present.
    if len(out) >= 2:
        fx, fy = out[0]
        lx, ly = out[-1]
        if (lx - fx) ** 2 + (ly - fy) ** 2 < tol_sq:
            out.pop()
    return out


@dataclass
class FemLayoutResult:
    """Output of :func:`from_fem_json`, a meshed-ready geometry plus the
    objects the caller needs to wire BCs/ports.

    Typical usage::

        from rapidfem import rfic, PEC, LumpedPort, ABC, Problem
        layout = rfic.from_fem_json("spiral.fem.json")
        all_conductors = [v for vs in layout.conductors.values() for v in vs]
        PEC(*(v.faces for v in all_conductors), layout.ground)
        for port in layout.ports.values():
            LumpedPort(port, direction=(0, 0, 1), z0=50.0)
        ABC(*layout.air.faces.outer)
        layout.geometry.mesh()
        result = Problem(layout.geometry).sweep([1e9, 10e9, 50e9])
    """
    geometry: "Geometry"
    conductors: dict[str, list["GeoObject"]]   # stack-layer id → 3-D conductor volumes
    ports: dict[str, "GeoObject"]               # port name → 2-D port plate
    ground: "GeoObject"                         # alias for the first ground patch
    ground_patches: list                        # one local ground per port (may merge)
    substrate: "GeoObject"
    oxide: "GeoObject"
    air: "GeoObject"
    doc: dict                                   # the parsed FEM-JSON (metadata + sim)


def from_fem_json(
    source,
    *,
    stack=None,
    via_mode: Literal["merged", "cells"] = "merged",
    footprint_margin: float = 0.3,
    air_height_um: float = 60.0,
    conductor_maxh_um: float = 1.5,
    port_maxh_um: float = 1.5,
    port_tab_um: float = 8.0,
    port_inset_um: float | None = None,
) -> FemLayoutResult:
    """Build a 3-D FEM geometry from a rapidpassives ``exportForFEM`` JSON.

    Conductors (metals AND vias) are extruded to their stack-layer thickness;
    every conductor surface is left for the caller to mark PEC. Ports are
    inset from each layout port's nominal location toward the layout centre
    so the plate top edge lands on the conductor's horizontal bottom face
    (PEC there constrains E_x/E_y but leaves E_z free, which the lumped-port
    drive needs).

    Parameters
    ----------
    source : str | pathlib.Path | dict
        Path to a ``.fem.json`` file or an already-parsed dict.
    stack : rfic.Stack, optional
        If given, replaces the substrate/oxide constants from the JSON's
        ``stack.substrate`` / ``stack.oxide`` block. The JSON's layer z-stack
        is always trusted (it carries the GDS-derived geometry).
    via_mode : {"merged", "cells"}
        "merged" (default) extrudes the merged bounding box of each via
        array → 1 conductor volume per array (fast). "cells" extrudes every
        individual via cell if the JSON provides ``polygon_cells``; falls
        back to the merged bbox when cells aren't present.
    footprint_margin : float
        Substrate/oxide/air enclosure margin as a fraction of the conductor
        bbox span. 0.3 = 30% on each side.
    air_height_um : float
        Air-box height above ``stack.top_z``.
    conductor_maxh_um : float
        Per-volume mesh-size cap for every extruded conductor.
    port_maxh_um : float
        Per-face mesh-size cap for the port plates + shared ground patch.
    port_tab_um : float
        Port plate width (extent perpendicular to the integration line).
    port_inset_um : float, optional
        Distance to move each port plate inward from the JSON's port
        location (toward layout centre). Default = ``port_tab_um / 2``,
        far enough to land the plate top edge inside the conductor's bottom
        face instead of on a side wall.

    Returns
    -------
    FemLayoutResult
    """
    # Local imports keep the rapidfem top-level lean even when the JSON
    # bridge isn't used.
    import json as _json
    from pathlib import Path as _Path
    from rapidfem.geometry import Geometry as _Geometry
    from rapidfem.materials import Air as _Air, Dielectric as _Dielectric, Conductor as _Conductor

    if isinstance(source, (str, _Path)):
        with open(source) as f:
            doc = _json.load(f)
    elif isinstance(source, dict):
        doc = source
    else:
        raise TypeError(f"source must be str/Path/dict, got {type(source).__name__}")

    sv = doc.get("schema_version", 1)
    if sv not in FEM_JSON_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported FEM JSON schema_version {sv!r}; "
                         f"supported: {FEM_JSON_SCHEMA_VERSIONS}")

    stack_doc      = doc["stack"]
    layers_doc     = stack_doc["layers"]
    substrate_doc  = stack_doc["substrate"]
    oxide_doc      = stack_doc["oxide"]
    conductors_doc = doc["conductors"]
    ports_doc      = doc["ports"]

    # Layer lookup by id, used everywhere below.
    layer_by_id = {l["id"]: l for l in layers_doc}

    # Resolve port layer ids, must already be valid stack-layer ids. Older
    # exports (pre 2026-05-19) wrote generator-internal names like "m3" that
    # had no stable mapping to stack ids; those need to be re-exported.
    metals_by_z = sorted(
        (l for l in layers_doc if l["type"] == "metal"),
        key=lambda l: l["z_um"],
    )
    def _resolve_port_layer(port_layer: str) -> str:
        if port_layer in layer_by_id:
            return port_layer
        raise KeyError(
            f"port references unknown stack layer {port_layer!r}; the JSON "
            f"may have been exported with an older rapidpassives that emitted "
            f"generator-internal names (e.g. 'm3'). Re-export the layout. "
            f"Known stack layers: {sorted(layer_by_id)}")

    # Substrate + oxide constants, JSON wins unless the caller passed a stack.
    sub_thickness_um = substrate_doc.get("thickness_um", 300.0) or 300.0
    sub_er           = substrate_doc.get("er", 11.7)
    sub_rho_ohm_cm   = substrate_doc.get("rho_ohm_cm", 10.0) or 10.0
    sub_sigma        = 100.0 / sub_rho_ohm_cm   # σ [S/m] = 1 / (ρ [Ω·cm] · 0.01)
    ox_er            = oxide_doc.get("er", 4.2)
    ox_tand          = oxide_doc.get("tand", 0.0)

    if stack is not None:
        sub_er, sub_sigma = stack.substrate_er, stack.substrate_sigma
        ox_er, ox_tand    = stack.oxide_er, stack.oxide_tand

    silicon = _Dielectric(er=sub_er, conductivity=sub_sigma)
    sio2    = _Dielectric(er=ox_er,  tand=ox_tand)
    air_mat = _Air()

    # Layout bbox → footprint with margin, all in metres.
    xs, ys = [], []
    for c in conductors_doc:
        for x, y in c["polygon"]:
            xs.append(x); ys.append(y)
    if not xs:
        raise ValueError("no conductor polygons in FEM JSON")
    x_min_um, x_max_um = min(xs), max(xs)
    y_min_um, y_max_um = min(ys), max(ys)
    span_x_um = max(x_max_um - x_min_um, 1.0)
    span_y_um = max(y_max_um - y_min_um, 1.0)
    foot_w = (span_x_um + 2 * footprint_margin * span_x_um) * 1e-6
    foot_h = (span_y_um + 2 * footprint_margin * span_y_um) * 1e-6
    cx_m   = (x_min_um + x_max_um) / 2 * 1e-6
    cy_m   = (y_min_um + y_max_um) / 2 * 1e-6

    # Stack z range, bottom of lowest layer, top of highest layer.
    layers_sorted = sorted(layers_doc, key=lambda l: l["z_um"])
    z_bottom_um = layers_sorted[0]["z_um"]
    z_top_um    = layers_sorted[-1]["z_um"] + layers_sorted[-1]["thickness_um"]
    z_top_m     = z_top_um * 1e-6

    # Build the enclosure. Global maxh = ~10% of the smaller in-plane span,
    # finer-than-bulk meshing of conductors is set per-volume via maxh.
    g = _Geometry(maxh=min(foot_w, foot_h) / 10)

    substrate = g.box(foot_w, foot_h, sub_thickness_um * 1e-6,
                      position=(cx_m - foot_w / 2, cy_m - foot_h / 2,
                                z_bottom_um * 1e-6 - sub_thickness_um * 1e-6),
                      material=silicon)
    oxide = g.box(foot_w, foot_h, (z_top_um - z_bottom_um) * 1e-6,
                  position=(cx_m - foot_w / 2, cy_m - foot_h / 2, z_bottom_um * 1e-6),
                  material=sio2)
    air = g.box(foot_w, foot_h, air_height_um * 1e-6,
                position=(cx_m - foot_w / 2, cy_m - foot_h / 2, z_top_m),
                material=air_mat)

    # Extrude every conductor polygon to its stack-layer thickness.
    cond_maxh = conductor_maxh_um * 1e-6
    conductor_objects: dict[str, list] = {}
    all_conductors: list = []

    for c in conductors_doc:
        layer_id = c["layer"]
        layer = layer_by_id.get(layer_id)
        if layer is None:
            raise KeyError(f"conductor references unknown layer {layer_id!r}")
        z_lo  = layer["z_um"] * 1e-6
        thick = layer["thickness_um"] * 1e-6
        if thick <= 0:
            continue

        # Pick the polygon set: merged bbox (default) or per-cell for vias.
        polys = [c["polygon"]]
        if via_mode == "cells" and c.get("polygon_cells"):
            polys = c["polygon_cells"]

        # Holes (annular conductors: rat-race ring, guard ring), only the
        # primary `polygon` entry carries holes; per-cell arrays are
        # individual solid pieces with no holes.
        holes_um = c.get("holes") if polys is not c.get("polygon_cells") else None

        # Derive bulk material from layer data. PEC is applied to all
        # conductor surfaces by the caller so the bulk assignment only
        # needs to satisfy the "every non-PML solid needs a material" rule.
        _ltype  = layer.get("type", "metal")
        _sigma  = float(layer.get("sigma") or 0.0)
        _er     = float(layer.get("er") or 1.0)
        _tand   = float(layer.get("tand") or 0.0)
        if _ltype in ("metal", "via") and _sigma > 0:
            layer_mat = _Conductor(conductivity=_sigma)
        elif _er > 1.0:
            layer_mat = _Dielectric(er=_er, tand=_tand)
        else:
            layer_mat = _Air()

        for poly_um in polys:
            cleaned = _clean_polygon_um(poly_um)
            if len(cleaned) < 3:
                continue
            pts_2d = [(x * 1e-6, y * 1e-6) for x, y in cleaned]
            holes_2d = None
            if holes_um:
                cleaned_holes = [_clean_polygon_um(h) for h in holes_um]
                _h2d = [
                    [(x * 1e-6, y * 1e-6) for x, y in h]
                    for h in cleaned_holes if len(h) >= 3
                ]
                if _h2d:
                    holes_2d = _h2d
            vol = g.prism(
                pts_2d, height=thick, position=(0.0, 0.0, z_lo),
                holes=holes_2d, material=layer_mat, maxh=cond_maxh,
            )
            vol.name = layer_id
            conductor_objects.setdefault(layer_id, []).append(vol)
            all_conductors.append(vol)

    # Shared local ground patch + one port plate per JSON port.
    # Ground sits on the lowest metal's TOP face, that's typically li1 in
    # sky130 and any other "ground shield" layer used by the generator.
    gnd_z = (metals_by_z[0]["z_um"] + metals_by_z[0]["thickness_um"]) * 1e-6

    port_tab_m = port_tab_um * 1e-6
    inset_m    = (port_inset_um if port_inset_um is not None
                   else port_tab_um / 2) * 1e-6
    port_maxh  = port_maxh_um * 1e-6

    # Per-port ground reference: walk z-stack down from each port's metal
    # and use the topmost lower metal that has a conductor covering the
    # port's (inset) xy. That way ratrace / patch / microstrip pick up the
    # generator's own ground plane (port plate stays in the substrate-to-
    # trace gap), while sparser layouts like spiral fall through to the
    # lowest metal and get a synthesised local ground patch added below.
    metal_below_by_z = sorted(
        (l for l in layers_doc if l["type"] == "metal"),
        key=lambda l: l["z_um"],
    )

    def _conductor_covers(layer_id, px, py):
        """Naive bbox-in-polygons check on conductors emitted on `layer_id`."""
        for c in conductors_doc:
            if c["layer"] != layer_id:
                continue
            poly = c["polygon"]
            xs = [pt[0] * 1e-6 for pt in poly]
            ys = [pt[1] * 1e-6 for pt in poly]
            if min(xs) <= px <= max(xs) and min(ys) <= py <= max(ys):
                return True
        return False

    def _polygon_at(layer_id, px, py):
        """Return the (xs, ys) of the conductor polygon on `layer_id` whose
        bbox covers (px, py), or None."""
        for c in conductors_doc:
            if c["layer"] != layer_id:
                continue
            poly = c["polygon"]
            xs = [pt[0] * 1e-6 for pt in poly]
            ys = [pt[1] * 1e-6 for pt in poly]
            if min(xs) <= px <= max(xs) and min(ys) <= py <= max(ys):
                return (min(xs), max(xs), min(ys), max(ys))
        return None

    def _polygons_same_net(top_id, top_xy, bot_id, bot_xy):
        """True if the polygons (top_xy on top_id, bot_xy on bot_id) are
        bridged by a via polygon (on any via-type layer between them)
        that sits inside BOTH bboxes. Means electrically same-net → not a
        valid lumped-port ground reference."""
        if top_xy is None or bot_xy is None:
            return False
        top_layer = layer_by_id[top_id]
        bot_layer = layer_by_id[bot_id]
        z_lo, z_hi = bot_layer["z_um"], top_layer["z_um"]
        for vl in layers_doc:
            if vl["type"] != "via" or not (z_lo < vl["z_um"] < z_hi):
                continue
            for c in conductors_doc:
                if c["layer"] != vl["id"]:
                    continue
                poly = c["polygon"]
                vxs = [pt[0] * 1e-6 for pt in poly]
                vys = [pt[1] * 1e-6 for pt in poly]
                cx = 0.5 * (min(vxs) + max(vxs))
                cy = 0.5 * (min(vys) + max(vys))
                tx0, tx1, ty0, ty1 = top_xy
                bx0, bx1, by0, by1 = bot_xy
                if (tx0 <= cx <= tx1 and ty0 <= cy <= ty1 and
                        bx0 <= cx <= bx1 and by0 <= cy <= by1):
                    return True
        return False

    resolved_ports = []
    for p in ports_doc:
        lay = _resolve_port_layer(p["layer"])
        port_layer = layer_by_id[lay]
        z_top = port_layer["z_um"] * 1e-6     # bottom of the port's metal
        px_raw = p["x_um"] * 1e-6
        py_raw = p["y_um"] * 1e-6

        # Inset toward the layout centre (cx_m, cy_m) so the plate top edge
        # lands on the conductor's bottom face rather than its side wall.
        dx, dy = px_raw - cx_m, py_raw - cy_m
        norm = math.hypot(dx, dy)
        if norm > 1e-15:
            px = px_raw - inset_m * dx / norm
            py = py_raw - inset_m * dy / norm
        else:
            px, py = px_raw, py_raw

        # Find this port's local ground: highest metal below the port's
        # metal that has a conductor covering (px, py). Fall back to the
        # lowest metal (li1 / poly) if none, we synthesise a local gnd
        # patch on that.
        port_gnd_z = (metal_below_by_z[0]["z_um"]
                      + metal_below_by_z[0]["thickness_um"]) * 1e-6
        port_needs_local_patch = True
        port_z = port_layer["z_um"]
        port_xy_box = _polygon_at(lay, px, py)
        for cand in reversed(metal_below_by_z):
            if cand["z_um"] >= port_z:
                continue
            cand_xy_box = _polygon_at(cand["id"], px, py)
            if cand_xy_box is None:
                continue
            # Skip same-net candidates: a via polygon between port_metal and
            # cand that lands inside both polygons connects them
            # electrically, using cand's top face as the port reference
            # would short the lumped port.
            if _polygons_same_net(lay, port_xy_box, cand["id"], cand_xy_box):
                continue
            port_gnd_z = (cand["z_um"] + cand["thickness_um"]) * 1e-6
            port_needs_local_patch = False
            break

        resolved_ports.append((p["name"], lay, px, py, z_top,
                               port_gnd_z, port_needs_local_patch))

    # Port plates with width TANGENT to the layout edge and PER-PORT gnd_z.
    # A port that sits over an existing ground plane stops at that plane;
    # a port without anything beneath drops to the lowest metal and we
    # synthesise a local ground patch there.
    port_objects: dict[str, "GeoObject"] = {}
    for name, _lay, px, py, z_top, port_gnd_z, _needs in resolved_ports:
        rx, ry = px - cx_m, py - cy_m
        rnorm = math.hypot(rx, ry)
        if rnorm > 1e-15:
            tx, ty = -ry / rnorm, rx / rnorm
        else:
            tx, ty = 0.0, 1.0     # fallback for port at the layout centre

        w_x = tx * port_tab_m
        w_y = ty * port_tab_m
        p0 = (px - tx * port_tab_m / 2,
              py - ty * port_tab_m / 2,
              port_gnd_z)
        port = g.plate(
            p0=p0,
            width=(w_x, w_y, 0),
            height=(0, 0, z_top - port_gnd_z),
            maxh=port_maxh,
        )
        port.name = name
        port_objects[name] = port

    # Only synthesise local ground patches for ports that have no real
    # ground plane below them. Cluster those by proximity so co-located
    # ports share a patch (GSG-style).
    needs_local = [
        i for i, rp in enumerate(resolved_ports) if rp[6]   # _needs flag
    ]
    cluster_dist = 4 * port_tab_m
    parent = list(range(len(resolved_ports)))
    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def _union(i, j):
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj
    for ii in range(len(needs_local)):
        for jj in range(ii + 1, len(needs_local)):
            i, j = needs_local[ii], needs_local[jj]
            xi, yi = resolved_ports[i][2], resolved_ports[i][3]
            xj, yj = resolved_ports[j][2], resolved_ports[j][3]
            if math.hypot(xi - xj, yi - yj) < cluster_dist:
                _union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in needs_local:
        clusters.setdefault(_find(i), []).append(i)

    gpad = max(port_tab_m, 4e-6)
    ground_patches: list = []
    for members in clusters.values():
        xs = [resolved_ports[i][2] for i in members]
        ys = [resolved_ports[i][3] for i in members]
        gx_min, gx_max = min(xs) - gpad, max(xs) + gpad
        gy_min, gy_max = min(ys) - gpad, max(ys) + gpad
        # All ports in a cluster share the same gnd_z (they routed there
        # because nothing covered them on any higher metal).
        cluster_gnd_z = resolved_ports[members[0]][5]
        gnd_patch = g.xy_plate(
            gx_max - gx_min, gy_max - gy_min,
            position=(gx_min, gy_min, cluster_gnd_z),
            maxh=port_maxh,
        )
        gnd_patch.name = "gnd_" + "_".join(resolved_ports[i][0] for i in members)
        ground_patches.append(gnd_patch)

    ground = ground_patches[0] if ground_patches else g.xy_plate(
        1e-6, 1e-6,
        position=(cx_m, cy_m,
                  (metals_by_z[0]['z_um'] + metals_by_z[0]['thickness_um']) * 1e-6),
        maxh=port_maxh)

    # One conformal fragment over everything that lives inside oxide + air.
    g.fragment(oxide, substrate, *all_conductors,
               *ground_patches, *port_objects.values(), air)

    return FemLayoutResult(
        geometry=g,
        conductors=conductor_objects,
        ports=port_objects,
        ground=ground,
        ground_patches=ground_patches,
        substrate=substrate,
        oxide=oxide,
        air=air,
        doc=doc,
    )


__all__ = [
    "Stack", "PdkLayer", "LayerType",
    "microstrip", "via", "trace_port", "gsg_port", "differential_port",
    "TracePort", "GsgPort", "DifferentialPort",
    "from_gds",
    "from_fem_json", "FemLayoutResult",
]
