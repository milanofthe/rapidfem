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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .stack import Stack, StackMaterial

if TYPE_CHECKING:
    from rapidfem.geometry import Geometry, GeoObject


# gds2palace convention for homogenised via arrays: full conductivity
# vertically, one tenth laterally (the array conducts through the posts).
VIA_LATERAL_FACTOR = 0.1


@dataclass
class MeshSpec:
    """Mesh sizing policy for :func:`build`, all lengths in meters.

    ``scale`` multiplies every size (the coarsen-everything knob for fast
    iteration); per-slab overrides go into ``slabs`` keyed by the background
    dielectric's name; ``graded`` splits a slab into stacked zones of
    different mesh size, listed top-down as (thickness, h) with the last
    entry padded to the slab's remaining thickness.
    """
    scale: float = 1.0
    conductor: float = 5e-6
    port: float = 3e-6
    global_h: float = 40e-6
    slab_default: float | None = None          # None -> global_h
    slabs: dict[str, float] = field(default_factory=dict)
    graded: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    def h(self, value: float) -> float:
        return self.scale * value

    def slab_h(self, name: str) -> float:
        base = self.slabs.get(name, self.slab_default or self.global_h)
        return self.scale * base


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


def _material_for(mat: StackMaterial, mesh_h: float | None = None):
    """rapidfem material instance for a stack material record."""
    from rapidfem.materials import Air, Dielectric
    if mat.kind != "conductor" and mat.er == 1.0 and mat.sigma == 0.0:
        m = Air()
    elif mat.sigma > 0.0 and mat.kind == "semiconductor":
        m = Dielectric(er=mat.er, conductivity=mat.sigma)
    else:
        m = Dielectric(er=mat.er, tand=mat.tand)
    if mesh_h is not None:
        m.maxh = mesh_h
    return m


def _read_markers(gds_path: str, layer: int) -> list[tuple[float, float, float, str]]:
    """Rectangles on a GDS marker layer -> (a0, a1, position, axis) per rect,
    in meters. The wide axis of each rectangle is the port-plate width."""
    import gdstk
    lib = gdstk.read_gds(str(gds_path))
    tops = lib.top_level()
    out = []
    for cell in tops:
        for poly in cell.flatten().polygons:
            if poly.layer != layer:
                continue
            xs = [p[0] * lib.unit for p in poly.points]
            ys = [p[1] * lib.unit for p in poly.points]
            wx, wy = max(xs) - min(xs), max(ys) - min(ys)
            if wx >= wy:
                out.append((min(xs), max(xs), (min(ys) + max(ys)) / 2, "x"))
            else:
                out.append((min(ys), max(ys), (min(xs) + max(xs)) / 2, "y"))
    return out


def _conformal_regions(gds_path: str, gds_layer: int, datatype: int,
                       t_side: float):
    """2D region decomposition for the conformal passivation, in meters.

    Returns (expanded, rings) where ``expanded`` are the metal polygons
    offset outward by ``t_side`` (merged), and ``rings`` maps each expanded
    polygon to the original metal polygons it contains (its holes). All
    polygons are (N, 2) float arrays. The decomposition is done entirely in
    2D so every extruded volume is disjoint by construction, no OCC
    booleans needed.
    """
    import gdstk
    import numpy as np

    lib = gdstk.read_gds(str(gds_path))
    unit = lib.unit
    metal = []
    for cell in lib.top_level():
        for poly in cell.flatten().polygons:
            if poly.layer == gds_layer and poly.datatype == datatype:
                metal.append(poly)
    if not metal:
        raise ValueError(f"no polygons on GDS layer {gds_layer}/{datatype} "
                         f"for the conformal passivation")

    merged = gdstk.boolean(metal, [], "or")
    expanded = gdstk.offset(merged, t_side / unit, use_union=True)

    def pts(p):
        return np.asarray(p.points, dtype=float) * unit

    rings: list[tuple] = []   # (expanded_poly_pts, [contained metal pts])
    for ep in expanded:
        ex = [q[0] for q in ep.points]
        ey = [q[1] for q in ep.points]
        bx = (min(ex), max(ex), min(ey), max(ey))
        inside = []
        for mp in merged:
            mx = [q[0] for q in mp.points]
            my = [q[1] for q in mp.points]
            if (bx[0] <= min(mx) and max(mx) <= bx[1]
                    and bx[2] <= min(my) and max(my) <= bx[3]):
                inside.append(pts(mp))
        rings.append((pts(ep), inside))
    return rings


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
    mesh: MeshSpec | None = None,
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
        ``"sibc" | "pec" | "volume" | "volume_iso"``. Defaults: metals ->
        sibc, vias -> volume (anisotropic), LOWLOSS -> pec.
    mesh : MeshSpec, optional
        Mesh sizing policy; defaults to ``MeshSpec()``.
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
    import numpy as np
    from rapidfem.geometry import Geometry
    from rapidfem.materials import Air, Dielectric
    from rapidfem.physics import ABC, PEC, PML, LumpedPort, SurfaceImpedance

    if not stack.dielectrics:
        raise ValueError(
            "stack has no background dielectrics; build() needs the full "
            "vertical cross-section (Stack.from_xml or a preset with "
            "dielectrics)")
    if passivation not in ("planar", "conformal", "none"):
        raise ValueError(f"passivation must be planar|conformal|none, "
                         f"got {passivation!r}")
    if boundary not in ("abc", "pml"):
        raise ValueError(f"boundary must be abc|pml, got {boundary!r}")
    if boundary == "pml" and pec_floor:
        raise ValueError("pec_floor is an ABC-mode option")
    mesh = mesh or MeshSpec()
    conductor_model = conductor_model or {}

    # The passivation sheet of the stack: topmost non-air dielectric slab.
    def _is_air(d):
        m = stack.materials.get(d.material, StackMaterial(d.material))
        return m.kind != "conductor" and m.er == 1.0 and m.sigma == 0.0

    pass_slab = None
    if passivation != "planar":
        cands = [d for d in stack.dielectrics
                 if not _is_air(d)
                 and stack.materials.get(d.material, StackMaterial(d.material)).kind == "dielectric"]
        pass_slab = cands[-1] if cands else None
        if pass_slab is None:
            raise ValueError("stack has no passivation slab to modify")

    # Conformal mode: which metal the passivation drapes over.
    conf_layer = None
    if passivation == "conformal":
        metals = [l for l in stack.layers if l.type == "metal"]
        if conformal_over is not None:
            conf_layer = stack.by_name(conformal_over)
        else:
            conf_layer = max(metals, key=lambda l: l.z_top)
        if pass_t_top is None:
            pass_t_top = pass_slab.thickness

    # ── conductors from the GDS ────────────────────────────────────────────
    g = Geometry.from_gds(str(gds), stack=stack, top_cell=top_cell)
    layer_names = {l.name for l in stack.layers}
    conductors: dict[str, list] = {}
    for o in g._objects:
        if o.name in layer_names:
            conductors.setdefault(o.name, []).append(o)
    if not conductors:
        raise ValueError(f"no stack layers found in {gds!r}")

    # Layout bbox from the extruded conductors (gmsh coords are scaled;
    # entity bboxes are tracked in scaled space, dilate back via g._scale).
    bbs = np.array([o._entity.bbox for o in g._objects if o.name in layer_names])
    sc = g._scale
    x_min, y_min = bbs[:, 0].min() * sc, bbs[:, 1].min() * sc
    x_max, y_max = bbs[:, 3].max() * sc, bbs[:, 4].max() * sc
    x0, y0 = x_min - margin, y_min - margin
    x1, y1 = x_max + margin, y_max + margin
    wx, wy = x1 - x0, y1 - y0

    # ── background dielectric slabs ────────────────────────────────────────
    slabs: dict[str, list] = {}
    z_bot = stack.dielectrics[0].z
    z_top = stack.dielectrics[-1].z_top
    for d in stack.dielectrics:
        mat = stack.materials.get(d.material, StackMaterial(d.material))
        is_air_like = (mat.kind != "conductor" and mat.er == 1.0
                       and mat.sigma == 0.0)
        thickness = d.thickness
        z_lo = d.z
        if passivation != "planar" and d is pass_slab:
            # sheet is replaced (conformal) or dropped (none); the air
            # region below extends down accordingly
            continue
        if passivation == "conformal" and d.z <= conf_layer.z < d.z_top:
            # oxide stops at the exposed metal's bottom
            thickness = conf_layer.z - d.z
        if d is stack.dielectrics[-1] and is_air_like:
            # topmost air slab: cap at air_top (the XML often carries a
            # generous 200 um; the ABC does not need more than the cap)
            thickness = air_top if air_top is not None else d.thickness
            if passivation == "conformal":
                # air is built as polygon prisms below, only the upper
                # cap above the shell top stays a plain box
                z_top = d.z + thickness
                continue
            if passivation == "none" and pass_slab is not None:
                z_lo = pass_slab.z          # extend down over the dropped sheet
                thickness += pass_slab.thickness
            z_top = z_lo + thickness
        boxes = []
        zones = mesh.graded.get(d.name)
        if zones:
            # split top-down into (thickness, h) zones, last zone padded
            z_hi = z_lo + thickness
            remaining = thickness
            for i, (t_zone, h_zone) in enumerate(zones):
                t = min(t_zone, remaining) if i < len(zones) - 1 else remaining
                if t <= 0:
                    break
                boxes.append(g.box(
                    wx, wy, t, position=(x0, y0, z_hi - t),
                    material=_material_for(mat, mesh.h(h_zone))))
                z_hi -= t
                remaining -= t
        else:
            boxes.append(g.box(
                wx, wy, thickness, position=(x0, y0, z_lo),
                material=_material_for(mat, mesh.slab_h(d.name))))
        slabs[d.name] = boxes

    # ── conformal passivation shell + polygon air prisms ───────────────────
    if passivation == "conformal":
        pass_mat = stack.materials.get(pass_slab.material,
                                       StackMaterial(pass_slab.material))
        air_slab = stack.dielectrics[-1]
        h_pass = mesh.slab_h(pass_slab.name)
        h_air = mesh.h(mesh.global_h)
        zm_lo, zm_hi = conf_layer.z, conf_layer.z_top
        rings = _conformal_regions(str(gds), conf_layer.gds,
                                   conf_layer.datatype, pass_t_side)

        def _prism(outer, holes, z, height, material, h):
            face = g.polygon(
                [(p[0], p[1], z) for p in outer],
                holes=[[(p[0], p[1], z) for p in hp] for hp in holes] or None)
            return g.extrude(face, height=height, material=material, maxh=h)

        foot = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        expanded = [ep for ep, _ in rings]
        shell = []
        # field sheet: everywhere except the expanded metal footprint
        shell.append(_prism(foot, expanded, zm_lo, pass_t_top,
                            _material_for(pass_mat), h_pass))
        # sidewall ring(s): expanded minus metal, hugging the sidewalls
        for ep, metal_holes in rings:
            shell.append(_prism(ep, metal_holes, zm_lo,
                                (zm_hi - zm_lo) + pass_t_top,
                                _material_for(pass_mat), h_pass))
            # cap directly on the metal top face
            for mp in metal_holes:
                shell.append(_prism(mp, [], zm_hi, pass_t_top,
                                    _material_for(pass_mat), h_pass))
        # Air above the shell. Only the step between the field sheet
        # (zm_lo + t_top) and the shell top (zm_hi + t_top) has to follow the
        # metal outline; everything above it is one plain box. Running the
        # polygon prisms all the way to z_top instead would stamp the metal
        # outline through the full air region — vertical interfaces between
        # air and air, metres of aspect ratio, and gmsh resolving the trace
        # width over the whole height (minSICN ~0.004 on the 2 nH octagon).
        z_shell_top = zm_hi + pass_t_top
        air_boxes_low = [
            _prism(foot, expanded, zm_lo + pass_t_top,
                   z_shell_top - (zm_lo + pass_t_top), Air(), h_air),
            g.box(wx, wy, z_top - z_shell_top,
                  position=(x0, y0, z_shell_top), material=Air(), maxh=h_air),
        ]
        slabs[pass_slab.name] = shell
        slabs[air_slab.name] = air_boxes_low

    # ── air shell around the dielectric stack (6 disjoint boxes) ───────────
    stack_h = z_top - z_bot
    a = Air()
    air_shell = [
        g.box(wx + 2 * air, wy + 2 * air, air,
              position=(x0 - air, y0 - air, z_top), material=a),
        g.box(wx + 2 * air, wy + 2 * air, air,
              position=(x0 - air, y0 - air, z_bot - air), material=a),
        g.box(air, wy + 2 * air, stack_h,
              position=(x0 - air, y0 - air, z_bot), material=a),
        g.box(air, wy + 2 * air, stack_h,
              position=(x1, y0 - air, z_bot), material=a),
        g.box(wx, air, stack_h, position=(x0, y0 - air, z_bot), material=a),
        g.box(wx, air, stack_h, position=(x0, y1, z_bot), material=a),
    ]

    # ── conductor treatment ────────────────────────────────────────────────
    # Interiors of SIBC/PEC shells are meshed as the surrounding oxide
    # (decoupled by the shell); volume conductors carry their bulk sigma.
    def _treatment(name: str) -> str:
        if name in conductor_model:
            return conductor_model[name]
        layer = stack.by_name(name)
        if layer.is_pec:
            return "pec"
        if layer.type == "via":
            return "volume"
        return "sibc"

    for name, objs in conductors.items():
        layer = stack.by_name(name)
        t = _treatment(name)
        for o in objs:
            if t == "volume":
                s = layer.sigma
                o.material = Dielectric(er=1.0, cond_diag=(
                    VIA_LATERAL_FACTOR * s, VIA_LATERAL_FACTOR * s, s))
            elif t == "volume_iso":
                o.material = Dielectric(er=1.0, conductivity=layer.sigma)
            else:
                o.material = Dielectric(er=stack.oxide_er, tand=stack.oxide_tand)
            o.maxh = mesh.h(mesh.conductor)

    # ── port plates (before fragmenting, so everything is conformal) ───────
    def _z_of(bound, lower: bool) -> float:
        if isinstance(bound, str):
            layer = stack.by_name(bound)
            return layer.z_top if lower else layer.z
        return float(bound)

    plates = []
    for p in ports:
        placements = []
        if p.marker is not None:
            placements = _read_markers(gds, p.marker)
            if not placements:
                raise ValueError(f"no rectangles on marker layer {p.marker}")
        else:
            if p.span is None or p.at is None:
                raise ValueError("ViaPort needs either marker or span+at")
            placements = [(p.span[0], p.span[1], p.at, p.axis)]
        z_lo = _z_of(p.z[0], lower=True)
        z_hi = _z_of(p.z[1], lower=False)
        for a0, a1, pos, axis in placements:
            if axis == "x":
                p0, width = (a0, pos, z_lo), (a1 - a0, 0, 0)
            else:
                p0, width = (pos, a0, z_lo), (0, a1 - a0, 0)
            plate = g.plate(p0=p0, width=width, height=(0, 0, z_hi - z_lo),
                            maxh=mesh.h(mesh.port))
            plates.append((plate, p.z0))

    # ── one conformal fragment over everything ─────────────────────────────
    all_boxes = [b for bs in slabs.values() for b in bs]
    all_conductors = [o for objs in conductors.values() for o in objs]
    g.fragment(all_boxes[0], *all_boxes[1:], *air_shell, *all_conductors,
               *(pl for pl, _ in plates))

    # ── physics: conductor BCs, ports, outer boundary ──────────────────────
    for name, objs in conductors.items():
        layer = stack.by_name(name)
        t = _treatment(name)
        if t == "pec":
            PEC(*(o.faces for o in objs))
        elif t == "sibc":
            SurfaceImpedance(*(o.faces for o in objs),
                             conductivity=layer.sigma,
                             thickness=layer.thickness, two_sided=True)
        # volume conductors need no surface BC

    for plate, z0 in plates:
        LumpedPort(plate, direction=(0, 0, 1), z0=z0)

    top, bot, cxmin, cxmax, cymin, cymax = air_shell
    if boundary == "pml":
        # each air-shell box becomes a single-direction PML slab; the six
        # boxes are disjoint by construction, exactly the "one slab per
        # outer face, no overlaps" layout the PML BC requires
        PML(top,   direction=(0, 0, 1),  inner_face=z_top,   thickness=air)
        PML(bot,   direction=(0, 0, -1), inner_face=z_bot,   thickness=air)
        PML(cxmin, direction=(-1, 0, 0), inner_face=x0,      thickness=air)
        PML(cxmax, direction=(1, 0, 0),  inner_face=x1,      thickness=air)
        PML(cymin, direction=(0, -1, 0), inner_face=y0,      thickness=air)
        PML(cymax, direction=(0, 1, 0),  inner_face=y1,      thickness=air)
    else:
        outer = [
            top.faces.min(axis="x"), top.faces.max(axis="x"),
            top.faces.min(axis="y"), top.faces.max(axis="y"),
            top.faces.max(axis="z"),
            bot.faces.min(axis="x"), bot.faces.max(axis="x"),
            bot.faces.min(axis="y"), bot.faces.max(axis="y"),
            cxmin.faces.min(axis="x"),
            cxmin.faces.min(axis="y"), cxmin.faces.max(axis="y"),
            cxmax.faces.max(axis="x"),
            cxmax.faces.min(axis="y"), cxmax.faces.max(axis="y"),
            cymin.faces.min(axis="y"),
            cymax.faces.max(axis="y"),
        ]
        if pec_floor:
            PEC(bot.faces.min(axis="z"))
        else:
            outer.append(bot.faces.min(axis="z"))
        ABC(*outer)

    if g._maxh is None:
        g._maxh = mesh.h(mesh.global_h)

    return BuiltModel(
        geometry=g, stack=stack, conductors=conductors, slabs=slabs,
        air_shell=air_shell, ports=[pl for pl, _ in plates],
        footprint=(x0, y0, x1, y1),
    )


__all__ = ["build", "BuiltModel", "MeshSpec", "ViaPort", "VIA_LATERAL_FACTOR"]
