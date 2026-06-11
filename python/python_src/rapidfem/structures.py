"""Composite RF structure builders for :class:`rapidfem.geometry.Geometry`.

Module-level functions (in the spirit of :meth:`Geometry.from_gds`) that
compose the primitive builders, plus optionally the standard physics, into
common macroscopic EM setups: coaxial lines, microstrip lines, coplanar
waveguide, ... Each takes a :class:`Geometry` as its first argument, builds
into it, and returns a small dataclass holding the created objects and the
canonical port faces.

These are **not** the PDK-stack-driven RFIC helpers in
:mod:`rapidfem.rfic` (those take a :class:`rapidfem.rfic.Stack` and named
metal layers at micrometre scale). The builders here stand alone: they
create their own substrate / air / conductor geometry from physical
dimensions.

Geometry only by default. Pass ``add_ports=True`` to also attach the
canonical ports plus the enclosing PEC so the structure is immediately
solvable; otherwise use the returned port faces to wire your own physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .materials import Air, Dielectric
from .physics import ABC, CoaxPort, PEC, RectWaveguidePort, WavePort

if TYPE_CHECKING:
    from .geometry import EntityCollection, GeoObject, Geometry

__all__ = [
    "coax", "CoaxLine",
    "microstrip", "MicrostripLine",
    "cpw", "CpwLine",
    "stripline", "Stripline",
    "rect_waveguide", "RectWaveguide",
    "circ_waveguide", "CircWaveguide",
    "sweep_along_path",
    "helix",
]


# Map the axis label a builder is laid out along to its unit direction. Used
# both for the cylinder sweep direction and for selecting the end-cap faces.
_AXIS_VEC = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}

# Microstrip substrate ports: a single-element-thick substrate slab is too
# coarse for the vector wave-port eigensolve to resolve the inhomogeneous
# quasi-TEM mode, so the substrate is meshed at this fraction of its
# thickness by default. Matches the fd_microstrip_line.py example.
_SUBSTRATE_MESH_DIVISIONS = 3


@dataclass
class CoaxLine:
    """Result of :func:`coax`.

    Attributes
    ----------
    dielectric : GeoObject
        the coaxial body (outer conductor radius down to the inner-conductor
        surface); carries the fill material. Its end-cap faces are the ports.
    port_a : EntityCollection
        the end cap at the base of the line (minimum along the build axis)
    port_b : EntityCollection
        the end cap at the far end (maximum along the build axis)
    ports : list
        the two :class:`rapidfem.CoaxPort` objects, populated only when
        :func:`coax` was called with ``add_ports=True`` (else empty)
    """

    dielectric: "GeoObject"
    port_a: "EntityCollection"
    port_b: "EntityCollection"
    ports: list = field(default_factory=list)


def coax(g: "Geometry", *,
         ri: float, ro: float, length: float,
         origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
         axis: str = "z",
         er: float = 1.0,
         material=None,
         add_ports: bool = False,
         power: float = 1.0) -> CoaxLine:
    """build a straight coaxial line: a fill cylinder of outer radius ``ro``
    with the inner conductor (radius ``ri``) removed as a PEC core.

    The annular region between ``ri`` and ``ro`` carries the fill material
    (air by default, or a dielectric when ``er`` is set). The two end caps
    are the coaxial ports.


    Example
    -------
    A 20 mm matched 50 ohm air line, ready to solve:

    .. code-block:: python

        from rapidfem import structures as st
        cx = st.coax(g, ri=1.5e-3, ro=3.45e-3, length=20e-3, add_ports=True)

    Geometry only, wiring your own ports off the returned faces:

    .. code-block:: python

        cx = st.coax(g, ri=1.5e-3, ro=3.45e-3, length=20e-3)
        rf.CoaxPort(cx.port_a, ri=1.5e-3, ro=3.45e-3)


    Parameters
    ----------
    g : Geometry
        geometry to build into
    ri, ro : float
        inner and outer conductor radii in metres (``ri < ro``)
    length : float
        line length in metres along ``axis``
    origin : tuple[float, float, float]
        base-cap centre (defaults to the origin)
    axis : str
        build direction, one of ``"x"`` / ``"y"`` / ``"z"`` (defaults to z)
    er : float
        relative permittivity of the fill (defaults to 1, i.e. air); ignored
        when ``material`` is given
    material : rapidfem.Material, optional
        explicit fill material; overrides ``er``
    add_ports : bool
        when True, attach a :class:`rapidfem.CoaxPort` at each end and PEC on
        every remaining (inner-conductor + shield) face
    power : float
        port excitation power in watts (only used when ``add_ports``)

    Returns
    -------
    CoaxLine
        the built coaxial line and its port faces

    Raises
    ------
    ValueError
        if ``ri >= ro`` or ``axis`` is not one of x / y / z
    """
    if ri >= ro:
        raise ValueError(f"coax: ri ({ri}) must be < ro ({ro})")
    if axis not in _AXIS_VEC:
        raise ValueError(f"coax: axis must be 'x', 'y' or 'z', got {axis!r}")
    av = _AXIS_VEC[axis]
    fill = material if material is not None else (
        Air() if er == 1.0 else Dielectric(er=er))

    # Outer fill cylinder with the inner-conductor cylinder cut out so the
    # inner-conductor surface exists as a face we can mark PEC. The inner
    # cylinder becomes a void; its walls remain selectable via outer.faces.
    outer = g.cylinder(ro, length, position=origin, axis=av, material=fill)
    inner = g.cylinder(ri, length, position=origin, axis=av)
    g.cut(outer, inner)

    port_a = outer.faces.min(axis=axis)
    port_b = outer.faces.max(axis=axis)
    line = CoaxLine(dielectric=outer, port_a=port_a, port_b=port_b)

    if add_ports:
        far = tuple(origin[i] + av[i] * length for i in range(3))
        p0 = CoaxPort(port_a, ri=ri, ro=ro, origin=origin, z_axis=av,
                      er=er, power=power)
        p1 = CoaxPort(port_b, ri=ri, ro=ro, origin=far, z_axis=av,
                      er=er, power=power)
        # Everything left (inner-conductor surface + outer shield) is PEC.
        PEC(*outer.faces.unassigned)
        line.ports = [p0, p1]

    return line


@dataclass
class MicrostripLine:
    """Result of :func:`microstrip`.

    Attributes
    ----------
    substrate : GeoObject
        the dielectric substrate slab
    air : GeoObject
        the air region above the substrate
    trace : GeoObject
        the signal trace (a thin plate on top of the substrate)
    ground : EntityCollection
        the substrate's bottom face (the ground plane; PEC when
        ``add_ports``)
    port_a : tuple[EntityCollection, EntityCollection]
        the (substrate, air) cross-section faces at the line's near end
    port_b : tuple[EntityCollection, EntityCollection]
        the (substrate, air) cross-section faces at the line's far end
    pec : object or None
        the :class:`rapidfem.PEC` object covering trace + ground when
        ``add_ports`` was set (else None)
    ports : list
        the two :class:`rapidfem.WavePort` objects when ``add_ports`` was
        set (else empty)
    """

    substrate: "GeoObject"
    air: "GeoObject"
    trace: "GeoObject"
    ground: "EntityCollection"
    port_a: "tuple[EntityCollection, EntityCollection]"
    port_b: "tuple[EntityCollection, EntityCollection]"
    pec: object = None
    ports: list = field(default_factory=list)


def microstrip(g: "Geometry", *,
               line_w: float, line_l: float,
               sub_w: float, sub_h: float, air_h: float,
               er: float, tand: float = 0.0,
               origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
               sub_maxh: float | None = None,
               add_ports: bool = False,
               f0: float | None = None,
               power: float = 1.0) -> MicrostripLine:
    """build a microstrip line: a signal trace on a dielectric substrate
    over a ground plane, in an air region.

    Layout convention (fixed): the line propagates along **+y**, its width
    runs along **x**, and the substrate / air stack rises along **+z**. The
    substrate is centred on x = 0 at ``origin``; the trace sits on top of
    the substrate, centred over it.

    With ``add_ports`` the canonical full-vector :class:`rapidfem.WavePort`
    is placed on the substrate-plus-air cross-section at each end (which
    de-embeds the inhomogeneous quasi-TEM mode), the trace and ground plane
    are tied to one PEC, and a first-order :class:`rapidfem.ABC` opens the
    lateral x-walls and the air top. The wave-port eigensolve needs the band
    centre, so ``f0`` is required in that case.


    Example
    -------
    A 50 ohm line on 0.508 mm RO4003C, solvable around 3 GHz:

    .. code-block:: python

        from rapidfem import structures as st
        ms = st.microstrip(g, line_w=1.13e-3, line_l=30e-3,
                           sub_w=20e-3, sub_h=0.508e-3, air_h=10e-3,
                           er=3.55, tand=0.0027,
                           add_ports=True, f0=3.0e9)


    Parameters
    ----------
    g : Geometry
        geometry to build into
    line_w, line_l : float
        trace width (along x) and length (along y) in metres
    sub_w : float
        substrate width along x in metres
    sub_h : float
        substrate thickness along z in metres
    air_h : float
        air-region height above the substrate along z in metres
    er : float
        substrate relative permittivity
    tand : float
        substrate loss tangent (defaults to 0)
    origin : tuple[float, float, float]
        the substrate's lower corner reference; the substrate spans
        ``x in [-sub_w/2, sub_w/2]`` about it (defaults to the origin)
    sub_maxh : float, optional
        substrate mesh size; defaults to ``sub_h / 3`` so the wave-port
        eigensolve resolves the cross-section
    add_ports : bool
        when True, attach the two wave ports, the trace + ground PEC, and
        the open-wall ABC
    f0 : float, optional
        band-centre frequency in Hz for the wave-port phase reference;
        required when ``add_ports`` is True
    power : float
        port excitation power in watts (only used when ``add_ports``)

    Returns
    -------
    MicrostripLine
        the built line, its conductors, and its port faces

    Raises
    ------
    ValueError
        if ``add_ports`` is True but ``f0`` was not given
    """
    if add_ports and f0 is None:
        raise ValueError(
            "microstrip: add_ports=True needs f0 (band-centre Hz) for the "
            "wave-port phase reference")

    ox, oy, oz = origin
    eff_sub_maxh = sub_maxh if sub_maxh is not None else sub_h / _SUBSTRATE_MESH_DIVISIONS
    fr4 = Dielectric(er=er, tand=tand, maxh=eff_sub_maxh)

    sub = g.box(sub_w, line_l, sub_h, position=(ox - sub_w / 2, oy, oz),
                material=fr4)
    air = g.box(sub_w, line_l, air_h, position=(ox - sub_w / 2, oy, oz + sub_h),
                material=Air())
    trace = g.xy_plate(line_w, line_l, position=(ox - line_w / 2, oy, oz + sub_h))

    g.fragment(sub, air, trace)

    ground = sub.faces.min(axis="z")
    port_a = (sub.faces.min(axis="y"), air.faces.min(axis="y"))
    port_b = (sub.faces.max(axis="y"), air.faces.max(axis="y"))
    line = MicrostripLine(substrate=sub, air=air, trace=trace, ground=ground,
                          port_a=port_a, port_b=port_b)

    if add_ports:
        # Trace + ground plane on one PEC so the wave-port eigensolve can mark
        # the conductor nodes via pec=[strip].
        strip = PEC(trace, ground)
        p0 = WavePort(port_a[0], port_a[1], f0=f0, mode_kind="auto",
                      pec=[strip], power=power)
        p1 = WavePort(port_b[0], port_b[1], f0=f0, mode_kind="auto",
                      pec=[strip], power=power)
        # Open the enclosure: ABC on the lateral x-walls (substrate + air) and
        # the air top. The y-extreme faces are the ports, so they are excluded.
        ABC(sub.faces.min(axis="x"), sub.faces.max(axis="x"),
            air.faces.min(axis="x"), air.faces.max(axis="x"),
            air.faces.max(axis="z"))
        line.pec = strip
        line.ports = [p0, p1]

    return line


@dataclass
class CpwLine:
    """Result of :func:`cpw`.

    Attributes
    ----------
    substrate, air : GeoObject
        the dielectric substrate and the air region above it
    signal : GeoObject
        the centre signal trace
    ground_left, ground_right : GeoObject
        the two coplanar ground strips flanking the signal
    port_a, port_b : tuple[EntityCollection, EntityCollection]
        the (substrate, air) cross-section faces at each end
    pec : object or None
        the PEC over all conductors when ``add_ports`` (else None)
    ports : list
        the two wave ports when ``add_ports`` (else empty)
    """

    substrate: "GeoObject"
    air: "GeoObject"
    signal: "GeoObject"
    ground_left: "GeoObject"
    ground_right: "GeoObject"
    port_a: "tuple[EntityCollection, EntityCollection]"
    port_b: "tuple[EntityCollection, EntityCollection]"
    pec: object = None
    ports: list = field(default_factory=list)


def cpw(g: "Geometry", *,
        signal_w: float, gap: float, line_l: float,
        sub_w: float, sub_h: float, air_h: float,
        er: float, tand: float = 0.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sub_maxh: float | None = None,
        backside_ground: bool = False,
        add_ports: bool = False,
        f0: float | None = None,
        power: float = 1.0) -> CpwLine:
    """build a coplanar waveguide: a centre signal trace flanked by two
    coplanar ground strips across a ``gap``, all on top of a substrate.

    Same fixed layout convention as :func:`microstrip` (propagation +y,
    width +x, stack +z). The signal is centred on x = 0; each ground strip
    runs from the gap edge out to the substrate edge. Pass
    ``backside_ground=True`` for conductor-backed CPW (adds the substrate
    bottom face to the PEC).

    With ``add_ports`` a full-vector :class:`rapidfem.WavePort` is placed on
    the cross-section at each end (``f0`` required), all three conductors
    ride on one PEC, and an ABC opens the air top.


    Example
    -------
    .. code-block:: python

        from rapidfem import structures as st
        cw = st.cpw(g, signal_w=0.4e-3, gap=0.2e-3, line_l=20e-3,
                    sub_w=10e-3, sub_h=0.635e-3, air_h=6e-3,
                    er=9.9, add_ports=True, f0=10e9)


    Parameters
    ----------
    g : Geometry
        geometry to build into
    signal_w : float
        signal-trace width along x in metres
    gap : float
        gap between the signal and each ground strip in metres
    line_l : float
        line length along y in metres
    sub_w : float
        substrate width along x in metres
    sub_h : float
        substrate thickness along z in metres
    air_h : float
        air-region height above the substrate along z in metres
    er : float
        substrate relative permittivity
    tand : float
        substrate loss tangent (defaults to 0)
    origin : tuple[float, float, float]
        substrate lower-corner reference; substrate spans x about it
    sub_maxh : float, optional
        substrate mesh size (defaults to ``sub_h / 3``)
    backside_ground : bool
        add the substrate bottom face to the PEC (conductor-backed CPW)
    add_ports : bool
        attach the two wave ports, the conductor PEC, and the ABC top
    f0 : float, optional
        band-centre frequency in Hz, required when ``add_ports``
    power : float
        port excitation power in watts (only when ``add_ports``)

    Returns
    -------
    CpwLine
        the built CPW and its port faces

    Raises
    ------
    ValueError
        if the ground strips would have non-positive width, or
        ``add_ports`` is set without ``f0``
    """
    ground_w = sub_w / 2 - signal_w / 2 - gap
    if ground_w <= 0:
        raise ValueError(
            f"cpw: signal_w/2 + gap ({signal_w / 2 + gap}) must be < sub_w/2 "
            f"({sub_w / 2}); ground strips have width {ground_w}")
    if add_ports and f0 is None:
        raise ValueError("cpw: add_ports=True needs f0 (band-centre Hz)")

    ox, oy, oz = origin
    eff_sub_maxh = sub_maxh if sub_maxh is not None else sub_h / _SUBSTRATE_MESH_DIVISIONS
    diel = Dielectric(er=er, tand=tand, maxh=eff_sub_maxh)

    sub = g.box(sub_w, line_l, sub_h, position=(ox - sub_w / 2, oy, oz),
                material=diel)
    air = g.box(sub_w, line_l, air_h, position=(ox - sub_w / 2, oy, oz + sub_h),
                material=Air())

    top_z = oz + sub_h
    signal = g.xy_plate(signal_w, line_l, position=(ox - signal_w / 2, oy, top_z))
    # Left ground: from the substrate's left edge to the left gap edge.
    gl_x0 = ox - sub_w / 2
    ground_left = g.xy_plate(ground_w, line_l, position=(gl_x0, oy, top_z))
    # Right ground: from the right gap edge to the substrate's right edge.
    gr_x0 = ox + signal_w / 2 + gap
    ground_right = g.xy_plate(ground_w, line_l, position=(gr_x0, oy, top_z))

    g.fragment(sub, air, signal, ground_left, ground_right)

    port_a = (sub.faces.min(axis="y"), air.faces.min(axis="y"))
    port_b = (sub.faces.max(axis="y"), air.faces.max(axis="y"))
    line = CpwLine(substrate=sub, air=air, signal=signal,
                   ground_left=ground_left, ground_right=ground_right,
                   port_a=port_a, port_b=port_b)

    if add_ports:
        conductors = [signal, ground_left, ground_right]
        if backside_ground:
            conductors.append(sub.faces.min(axis="z"))
        strip = PEC(*conductors)
        p0 = WavePort(port_a[0], port_a[1], f0=f0, mode_kind="auto",
                      pec=[strip], power=power)
        p1 = WavePort(port_b[0], port_b[1], f0=f0, mode_kind="auto",
                      pec=[strip], power=power)
        # Lateral x-walls touch the ground strips; only the air top stays open.
        ABC(air.faces.max(axis="z"))
        line.pec = strip
        line.ports = [p0, p1]

    return line


@dataclass
class Stripline:
    """Result of :func:`stripline`.

    The dielectric is split into a lower and an upper half meeting at the
    trace plane, so the trace plate lies on the shared interface (no
    embedded floating sheet); both halves carry the same fill material.

    Attributes
    ----------
    lower, upper : GeoObject
        the lower and upper dielectric halves (below / above the trace)
    trace : GeoObject
        the centre signal trace on the mid-height interface
    port_a, port_b : tuple[EntityCollection, EntityCollection]
        the (lower, upper) cross-section faces at each end
    pec : object or None
        the PEC over trace + both grounds + side walls when ``add_ports``
    ports : list
        the two wave ports when ``add_ports`` (else empty)
    """

    lower: "GeoObject"
    upper: "GeoObject"
    trace: "GeoObject"
    port_a: "tuple[EntityCollection, EntityCollection]"
    port_b: "tuple[EntityCollection, EntityCollection]"
    pec: object = None
    ports: list = field(default_factory=list)


def stripline(g: "Geometry", *,
              line_w: float, line_l: float,
              sub_w: float, sub_h: float,
              er: float, tand: float = 0.0,
              origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
              sub_maxh: float | None = None,
              add_ports: bool = False,
              f0: float | None = None,
              power: float = 1.0) -> Stripline:
    """build a stripline: a signal trace centred at mid-height in a
    homogeneous dielectric, fully enclosed by top, bottom and side ground
    walls (boxed, shielded TEM line).

    Same fixed layout convention as :func:`microstrip` (propagation +y,
    width +x, stack +z). The trace sits at ``z = sub_h / 2`` above the
    dielectric's lower face, centred on x = 0.

    With ``add_ports`` a full-vector :class:`rapidfem.WavePort` is placed on
    the dielectric cross-section at each end (``f0`` required); the trace,
    both ground planes and both side walls ride on one PEC, so the line is
    fully shielded.


    Example
    -------
    .. code-block:: python

        from rapidfem import structures as st
        sl = st.stripline(g, line_w=0.3e-3, line_l=20e-3,
                          sub_w=8e-3, sub_h=1.0e-3, er=3.38,
                          add_ports=True, f0=5e9)


    Parameters
    ----------
    g : Geometry
        geometry to build into
    line_w : float
        trace width along x in metres
    line_l : float
        line length along y in metres
    sub_w : float
        dielectric width along x in metres
    sub_h : float
        total dielectric height along z in metres (trace sits at sub_h/2)
    er : float
        dielectric relative permittivity
    tand : float
        dielectric loss tangent (defaults to 0)
    origin : tuple[float, float, float]
        dielectric lower-corner reference; spans x about it
    sub_maxh : float, optional
        dielectric mesh size (defaults to ``sub_h / 3``)
    add_ports : bool
        attach the two wave ports and the full shielding PEC
    f0 : float, optional
        band-centre frequency in Hz, required when ``add_ports``
    power : float
        port excitation power in watts (only when ``add_ports``)

    Returns
    -------
    Stripline
        the built stripline and its port faces

    Raises
    ------
    ValueError
        if ``add_ports`` is set without ``f0``
    """
    if add_ports and f0 is None:
        raise ValueError("stripline: add_ports=True needs f0 (band-centre Hz)")

    ox, oy, oz = origin
    eff_sub_maxh = sub_maxh if sub_maxh is not None else sub_h / _SUBSTRATE_MESH_DIVISIONS
    half_h = sub_h / 2

    # Split the fill into a lower and an upper half meeting at the trace plane.
    # The trace then lies on the shared interface (a full partition surface),
    # not as a floating embedded sheet, which would crash the mesh optimizer.
    diel_lo = Dielectric(er=er, tand=tand, maxh=eff_sub_maxh)
    diel_hi = Dielectric(er=er, tand=tand, maxh=eff_sub_maxh)
    lower = g.box(sub_w, line_l, half_h, position=(ox - sub_w / 2, oy, oz),
                  material=diel_lo)
    upper = g.box(sub_w, line_l, half_h,
                  position=(ox - sub_w / 2, oy, oz + half_h), material=diel_hi)
    trace = g.xy_plate(line_w, line_l, position=(ox - line_w / 2, oy, oz + half_h))

    g.fragment(lower, upper, trace)

    port_a = (lower.faces.min(axis="y"), upper.faces.min(axis="y"))
    port_b = (lower.faces.max(axis="y"), upper.faces.max(axis="y"))
    line = Stripline(lower=lower, upper=upper, trace=trace,
                     port_a=port_a, port_b=port_b)

    if add_ports:
        # Trace + the four enclosing walls: bottom ground (lower z-min), top
        # ground (upper z-max), and the side walls on both halves.
        strip = PEC(trace,
                    lower.faces.min(axis="z"), upper.faces.max(axis="z"),
                    lower.faces.min(axis="x"), lower.faces.max(axis="x"),
                    upper.faces.min(axis="x"), upper.faces.max(axis="x"))
        p0 = WavePort(port_a[0], port_a[1], f0=f0, mode_kind="auto",
                      pec=[strip], power=power)
        p1 = WavePort(port_b[0], port_b[1], f0=f0, mode_kind="auto",
                      pec=[strip], power=power)
        line.pec = strip
        line.ports = [p0, p1]

    return line


# Box dimension order per propagation axis: which (width, depth, height) slot
# the two transverse sizes (a, b) and the length fill. Keeps the cross-section
# (a, b) transverse to the chosen propagation axis.
def _box_dims_for_axis(a: float, b: float, length: float, axis: str):
    if axis == "z":
        return (a, b, length)          # transverse x, y
    if axis == "y":
        return (a, length, b)          # transverse x, z
    if axis == "x":
        return (length, a, b)          # transverse y, z
    raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")


@dataclass
class RectWaveguide:
    """Result of :func:`rect_waveguide`.

    Attributes
    ----------
    body : GeoObject
        the waveguide fill volume
    port_a, port_b : EntityCollection
        the end-cap faces (the two waveguide ports)
    ports : list
        the two :class:`rapidfem.RectWaveguidePort` objects when
        ``add_ports`` (else empty)
    """

    body: "GeoObject"
    port_a: "EntityCollection"
    port_b: "EntityCollection"
    ports: list = field(default_factory=list)


def rect_waveguide(g: "Geometry", *,
                   a: float, b: float, length: float,
                   origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                   axis: str = "z",
                   er: float = 1.0,
                   material=None,
                   mode: tuple[int, int] = (1, 0),
                   add_ports: bool = False,
                   power: float = 1.0) -> RectWaveguide:
    """build a straight rectangular waveguide section of cross-section
    ``a`` x ``b`` and the given ``length`` along ``axis``.

    With ``add_ports`` a :class:`rapidfem.RectWaveguidePort` (default mode
    TE10) is placed at each end and the four side walls become PEC.


    Example
    -------
    A WR-90 (X-band) section, 30 mm long:

    .. code-block:: python

        from rapidfem import structures as st
        wg = st.rect_waveguide(g, a=22.86e-3, b=10.16e-3, length=30e-3,
                               add_ports=True)


    Parameters
    ----------
    g : Geometry
        geometry to build into
    a, b : float
        broad-wall and narrow-wall cross-section dimensions in metres
    length : float
        section length in metres along ``axis``
    origin : tuple[float, float, float]
        lower corner of the body box (defaults to the origin)
    axis : str
        propagation direction, one of ``"x"`` / ``"y"`` / ``"z"`` (z default)
    er : float
        relative permittivity of the fill (defaults to 1, air); ignored
        when ``material`` is given
    material : rapidfem.Material, optional
        explicit fill material; overrides ``er``
    mode : tuple[int, int]
        waveguide mode (m, n) for the ports (defaults to TE10)
    add_ports : bool
        attach a waveguide port at each end and PEC on the four side walls
    power : float
        port excitation power in watts (only when ``add_ports``)

    Returns
    -------
    RectWaveguide
        the built section and its port faces
    """
    fill = material if material is not None else (
        Air() if er == 1.0 else Dielectric(er=er))
    w, d, h = _box_dims_for_axis(a, b, length, axis)
    body = g.box(w, d, h, position=origin, material=fill)

    port_a = body.faces.min(axis=axis)
    port_b = body.faces.max(axis=axis)
    wg = RectWaveguide(body=body, port_a=port_a, port_b=port_b)

    if add_ports:
        p0 = RectWaveguidePort(port_a, mode=mode, er=er, power=power)
        p1 = RectWaveguidePort(port_b, mode=mode, er=er, power=power)
        PEC(*body.faces.unassigned)
        wg.ports = [p0, p1]

    return wg


@dataclass
class CircWaveguide:
    """Result of :func:`circ_waveguide`.

    Attributes
    ----------
    body : GeoObject
        the cylindrical fill volume
    port_a, port_b : EntityCollection
        the end-cap faces (the two waveguide ports)
    ports : list
        the two :class:`rapidfem.WavePort` objects when ``add_ports``
    """

    body: "GeoObject"
    port_a: "EntityCollection"
    port_b: "EntityCollection"
    ports: list = field(default_factory=list)


def circ_waveguide(g: "Geometry", *,
                   radius: float, length: float,
                   origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                   axis: str = "z",
                   er: float = 1.0,
                   material=None,
                   add_ports: bool = False,
                   f0: float | None = None,
                   power: float = 1.0) -> CircWaveguide:
    """build a straight circular waveguide section of the given ``radius``
    and ``length`` along ``axis``.

    Circular guides have no closed-form rectangular port, so with
    ``add_ports`` a numerically solved full-vector :class:`rapidfem.WavePort`
    is placed at each end (``f0`` required) and the curved wall becomes PEC.


    Example
    -------
    .. code-block:: python

        from rapidfem import structures as st
        wg = st.circ_waveguide(g, radius=10e-3, length=30e-3,
                               add_ports=True, f0=12e9)


    Parameters
    ----------
    g : Geometry
        geometry to build into
    radius : float
        guide radius in metres
    length : float
        section length in metres along ``axis``
    origin : tuple[float, float, float]
        base-cap centre (defaults to the origin)
    axis : str
        propagation direction, one of ``"x"`` / ``"y"`` / ``"z"`` (z default)
    er : float
        relative permittivity of the fill (defaults to 1, air); ignored
        when ``material`` is given
    material : rapidfem.Material, optional
        explicit fill material; overrides ``er``
    add_ports : bool
        attach a wave port at each end and PEC on the curved wall
    f0 : float, optional
        band-centre frequency in Hz, required when ``add_ports``
    power : float
        port excitation power in watts (only when ``add_ports``)

    Returns
    -------
    CircWaveguide
        the built section and its port faces

    Raises
    ------
    ValueError
        if ``axis`` is invalid, or ``add_ports`` is set without ``f0``
    """
    if axis not in _AXIS_VEC:
        raise ValueError(f"circ_waveguide: axis must be 'x', 'y' or 'z', got {axis!r}")
    if add_ports and f0 is None:
        raise ValueError("circ_waveguide: add_ports=True needs f0 (band-centre Hz)")
    av = _AXIS_VEC[axis]
    fill = material if material is not None else (
        Air() if er == 1.0 else Dielectric(er=er))
    body = g.cylinder(radius, length, position=origin, axis=av, material=fill)

    port_a = body.faces.min(axis=axis)
    port_b = body.faces.max(axis=axis)
    wg = CircWaveguide(body=body, port_a=port_a, port_b=port_b)

    if add_ports:
        p0 = WavePort(port_a, f0=f0, mode_kind="auto", power=power)
        p1 = WavePort(port_b, f0=f0, mode_kind="auto", power=power)
        PEC(*body.faces.unassigned)
        wg.ports = [p0, p1]

    return wg


def sweep_along_path(g: "Geometry", radius: float,
                     points: "list[tuple[float, float, float]]",
                     *,
                     material=None,
                     maxh: float | None = None) -> "GeoObject":
    """sweep a circular cross-section of ``radius`` along the polygonal
    path through ``points`` into a 3-D solid.

    The workhorse behind curved conductors: bond wires, bent traces, coax
    bends. The sweep is piecewise linear (polygonal) between the given
    waypoints; use more waypoints per segment for a smoother curve.

    Note: the old API accepted a ``profile`` :class:`GeoObject` (disc face)
    as the second argument. That form is retired because :meth:`Geometry.disc`
    now creates a zero-thickness sheet, not a sweep profile. Pass the circular
    cross-section ``radius`` directly.


    Example
    -------
    A round bond wire arcing between two pads:

    .. code-block:: python

        from rapidfem import structures as st
        pts = [(0, 0, 0), (0.5e-3, 0, 0.4e-3), (1e-3, 0, 0)]
        wire = st.sweep_along_path(g, 50e-6, pts, material=rf.Air())


    Parameters
    ----------
    g : Geometry
        geometry to build into
    radius : float
        radius of the circular cross-section in metres
    points : list[tuple[float, float, float]]
        path waypoints in metres; the sweep is polygonal (piecewise linear)
        between them
    material : rapidfem.Material, optional
        material for the swept solid
    maxh : float, optional
        per-volume mesh size override

    Returns
    -------
    GeoObject
        the swept volume

    Raises
    ------
    ValueError
        if fewer than two points are given
    """
    if len(points) < 2:
        raise ValueError("sweep_along_path needs at least two path points")
    return g.sweep(points, radius, material=material, maxh=maxh)


def helix(g: "Geometry", *,
          radius: float, pitch: float, turns: float, wire_radius: float,
          position: tuple[float, float, float] = (0.0, 0.0, 0.0),
          points_per_turn: int = 24,
          material=None,
          maxh: float | None = None) -> "GeoObject":
    """build a circular-cross-section helix (coil) wound about the +z axis.

    A round wire of radius ``wire_radius`` is swept along a helical path of
    the given coil ``radius``, axial ``pitch`` (rise per full turn) and
    number of ``turns``. The helix climbs along +z starting at
    ``position + (radius, 0, 0)``. For another orientation, build it here
    and reorient with :meth:`Geometry.translate` / :meth:`Geometry.array`.

    Useful for inductors and helical antennas. Delegates to
    :meth:`Geometry.helix` which uses the rapidmesh kernel pipe primitive.


    Example
    -------
    A 5-turn coil, 2 mm radius, 1 mm pitch, 0.1 mm wire:

    .. code-block:: python

        from rapidfem import structures as st
        coil = st.helix(g, radius=2e-3, pitch=1e-3, turns=5,
                        wire_radius=0.1e-3, material=rf.Air())


    Parameters
    ----------
    g : Geometry
        geometry to build into
    radius : float
        coil (helix) radius in metres
    pitch : float
        axial rise per full turn in metres
    turns : float
        number of turns (may be fractional)
    wire_radius : float
        radius of the round wire cross-section in metres
    position : tuple[float, float, float]
        helix-axis base point; the wire starts at ``position + (radius,0,0)``
    points_per_turn : int
        path-sampling density per turn (higher = smoother, defaults to 24)
    material : rapidfem.Material, optional
        wire material
    maxh : float, optional
        per-volume mesh size override

    Returns
    -------
    GeoObject
        the helical wire volume

    Raises
    ------
    ValueError
        if ``turns`` or ``points_per_turn`` are non-positive
    """
    if turns <= 0:
        raise ValueError(f"helix: turns must be > 0, got {turns}")
    if points_per_turn < 2:
        raise ValueError(f"helix: points_per_turn must be >= 2, got {points_per_turn}")
    return g.helix(radius, pitch, turns, wire_radius, position=position,
                   points_per_turn=points_per_turn, material=material, maxh=maxh)
