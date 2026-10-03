# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

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


# Unit direction of each axis label a straight section can be built along.
_AXIS_VEC = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}

# Planar-line substrates: a single-element-thick substrate slab is too
# coarse for the vector wave-port eigensolve to resolve the inhomogeneous
# quasi-TEM mode, so the substrate is meshed at this fraction of its
# thickness by default. Matches the fd_microstrip_line.py example.
_SUBSTRATE_MESH_DIVISIONS = 3


# SHARED HELPERS ========================================================================

def _axis(builder: str, axis: str) -> tuple[float, float, float]:
    """unit vector of the build axis label, or a ValueError naming the builder"""
    if axis not in _AXIS_VEC:
        raise ValueError(f"{builder}: axis must be 'x', 'y' or 'z', got {axis!r}")
    return _AXIS_VEC[axis]


def _check_f0(builder: str, add_ports: bool, f0: float | None) -> None:
    """wave ports solve their mode at the band centre, so they need ``f0``;
    checked before anything is built"""
    if add_ports and f0 is None:
        raise ValueError(f"{builder}: add_ports=True needs f0 (band-centre Hz) "
                         "for the wave-port mode solve")


def _fill(er: float, material):
    """the fill material and its relative permittivity: ``material`` when
    given (its ``er`` then also feeds the analytic ports), else air or a
    lossless dielectric of permittivity ``er``"""
    if material is None:
        material = Air() if er == 1.0 else Dielectric(er=er)
    return material, material.er


def _ends(axis: str, *bodies):
    """the end-cap faces (near, far) of straight sections along ``axis``;
    one collection per body, unwrapped for a single body"""
    near = tuple(b.faces.min(axis=axis) for b in bodies)
    far = tuple(b.faces.max(axis=axis) for b in bodies)
    return (near, far) if len(bodies) > 1 else (near[0], far[0])


def _slab(g, origin, width, length, z, height, material):
    """box of a planar line: centred on the origin's x, running along +y from
    the origin, spanning ``[z, z + height]`` above it"""
    ox, oy, oz = origin
    return g.box(width, length, height, position=(ox - width / 2, oy, oz + z),
                 material=material)


def _substrate(g, origin, width, length, height, er, tand, maxh):
    """dielectric slab of a planar line, meshed at ``height / 3`` unless
    ``maxh`` is given"""
    if maxh is None:
        maxh = height / _SUBSTRATE_MESH_DIVISIONS
    return _slab(g, origin, width, length, 0.0, height,
                 Dielectric(er=er, tand=tand, maxh=maxh))


def _strip(g, origin, x, width, length, z):
    """sheet conductor of a planar line along +y, its left edge ``x`` and
    its height ``z`` relative to the origin"""
    ox, oy, oz = origin
    return g.xy_plate(width, length, position=(ox + x, oy, oz + z))


def _wave_ports(line, f0: float, power: float) -> None:
    """attach a full-vector wave port at each end of ``line``; ``line.pec``,
    when already set, marks the conductors inside the cross-section"""
    pec = None if line.pec is None else [line.pec]
    line.ports = [WavePort(*(end if isinstance(end, tuple) else (end,)),
                           f0=f0, pec=pec, power=power)
                  for end in (line.port_a, line.port_b)]


@dataclass(kw_only=True)
class _Section:
    """Fields every builder result carries (documented on each subclass)."""

    port_a: "EntityCollection | tuple[EntityCollection, ...]"
    port_b: "EntityCollection | tuple[EntityCollection, ...]"
    pec: object = None
    ports: list = field(default_factory=list)


# COAX ==================================================================================

@dataclass(kw_only=True)
class CoaxLine(_Section):
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
    pec : object or None
        the :class:`rapidfem.PEC` on the inner-conductor surface and the
        shield when ``add_ports`` was set (else None)
    ports : list
        the two :class:`rapidfem.CoaxPort` objects, populated only when
        :func:`coax` was called with ``add_ports=True`` (else empty)
    """

    dielectric: "GeoObject"


def coax(g: "Geometry", *,
         ri: float, ro: float, length: float,
         origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
         axis: str = "z",
         er: float = 1.0,
         material=None,
         add_ports: bool = False,
         power: float = 1.0) -> CoaxLine:
    """build a straight coaxial line: a fill cylinder of outer radius ``ro``
    with the inner conductor (radius ``ri``) cut out of it.

    The annular region between ``ri`` and ``ro`` carries the fill material
    (air by default, or a dielectric when ``er`` is set) and is the only
    meshed volume: the inner conductor is a hole, so its surface is a
    boundary face of the fill (PEC unless other physics is placed on it).
    The two annular end caps are the coaxial ports.


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
        explicit fill material; overrides ``er`` (the ports then use its
        ``er``)
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
    av = _axis("coax", axis)
    fill, er = _fill(er, material)

    # The inner conductor is a hole in the fill: the mesh ends on its surface,
    # which stays a face of the dielectric for the PEC.
    dielectric = g.cylinder(ro, length, position=origin, axis=av, material=fill)
    g.cut(dielectric, g.cylinder(ri, length, position=origin, axis=av))

    port_a, port_b = _ends(axis, dielectric)
    line = CoaxLine(dielectric=dielectric, port_a=port_a, port_b=port_b)

    if add_ports:
        far = tuple(o + v * length for o, v in zip(origin, av))
        line.ports = [CoaxPort(end, ri=ri, ro=ro, origin=at, z_axis=av,
                               er=er, power=power)
                      for end, at in ((port_a, origin), (port_b, far))]
        # Everything left (inner-conductor surface + outer shield) is PEC.
        line.pec = PEC(*dielectric.faces.unassigned)

    return line


# PLANAR LINES ==========================================================================

@dataclass(kw_only=True)
class MicrostripLine(_Section):
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
    _check_f0("microstrip", add_ports, f0)

    sub = _substrate(g, origin, sub_w, line_l, sub_h, er, tand, sub_maxh)
    air = _slab(g, origin, sub_w, line_l, sub_h, air_h, Air())
    trace = _strip(g, origin, -line_w / 2, line_w, line_l, sub_h)

    port_a, port_b = _ends("y", sub, air)
    line = MicrostripLine(substrate=sub, air=air, trace=trace,
                          ground=sub.faces.min(axis="z"),
                          port_a=port_a, port_b=port_b)

    if add_ports:
        # Trace + ground plane on one PEC so the wave-port eigensolve can mark
        # the conductor nodes.
        line.pec = PEC(trace, line.ground)
        _wave_ports(line, f0, power)
        # Open the enclosure: ABC on the lateral x-walls (substrate + air) and
        # the air top. The y-extreme faces are the ports, so they are excluded.
        walls_lo, walls_hi = _ends("x", sub, air)
        ABC(*walls_lo, *walls_hi, air.faces.max(axis="z"))

    return line


@dataclass(kw_only=True)
class CpwLine(_Section):
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
    _check_f0("cpw", add_ports, f0)

    sub = _substrate(g, origin, sub_w, line_l, sub_h, er, tand, sub_maxh)
    air = _slab(g, origin, sub_w, line_l, sub_h, air_h, Air())
    signal = _strip(g, origin, -signal_w / 2, signal_w, line_l, sub_h)
    # Each ground strip runs from its gap edge out to the substrate edge.
    ground_left = _strip(g, origin, -sub_w / 2, ground_w, line_l, sub_h)
    ground_right = _strip(g, origin, signal_w / 2 + gap, ground_w, line_l, sub_h)

    port_a, port_b = _ends("y", sub, air)
    line = CpwLine(substrate=sub, air=air, signal=signal,
                   ground_left=ground_left, ground_right=ground_right,
                   port_a=port_a, port_b=port_b)

    if add_ports:
        conductors = [signal, ground_left, ground_right]
        if backside_ground:
            conductors.append(sub.faces.min(axis="z"))
        line.pec = PEC(*conductors)
        _wave_ports(line, f0, power)
        # Lateral x-walls touch the ground strips; only the air top stays open.
        ABC(air.faces.max(axis="z"))

    return line


@dataclass(kw_only=True)
class Stripline(_Section):
    """Result of :func:`stripline`.

    Attributes
    ----------
    fill : GeoObject
        the dielectric the trace is embedded in
    trace : GeoObject
        the centre signal trace, a sheet at mid-height
    port_a, port_b : EntityCollection
        the cross-section faces at each end
    pec : object or None
        the PEC over trace + both grounds + side walls when ``add_ports``
    ports : list
        the two wave ports when ``add_ports`` (else empty)
    """

    fill: "GeoObject"
    trace: "GeoObject"


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
    _check_f0("stripline", add_ports, f0)

    fill = _substrate(g, origin, sub_w, line_l, sub_h, er, tand, sub_maxh)
    trace = _strip(g, origin, -line_w / 2, line_w, line_l, sub_h / 2)

    port_a, port_b = _ends("y", fill)
    line = Stripline(fill=fill, trace=trace, port_a=port_a, port_b=port_b)

    if add_ports:
        # Trace + the four enclosing walls: both ground planes and both side
        # walls.
        line.pec = PEC(trace, *_ends("z", fill), *_ends("x", fill))
        _wave_ports(line, f0, power)

    return line


# WAVEGUIDES ============================================================================

@dataclass(kw_only=True)
class RectWaveguide(_Section):
    """Result of :func:`rect_waveguide`.

    Attributes
    ----------
    body : GeoObject
        the waveguide fill volume
    port_a, port_b : EntityCollection
        the end-cap faces (the two waveguide ports)
    pec : object or None
        the PEC on the four side walls when ``add_ports`` (else None)
    ports : list
        the two :class:`rapidfem.RectWaveguidePort` objects when
        ``add_ports`` (else empty)
    """

    body: "GeoObject"


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
        propagation direction, one of ``"x"`` / ``"y"`` / ``"z"`` (z default);
        ``a`` runs along the first and ``b`` along the second remaining axis
    er : float
        relative permittivity of the fill (defaults to 1, air); ignored
        when ``material`` is given
    material : rapidfem.Material, optional
        explicit fill material; overrides ``er`` (the ports then use its
        ``er``)
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

    Raises
    ------
    ValueError
        if ``axis`` is not one of x / y / z
    """
    _axis("rect_waveguide", axis)
    fill, er = _fill(er, material)
    size = {"x": (length, a, b), "y": (a, length, b), "z": (a, b, length)}[axis]
    body = g.box(*size, position=origin, material=fill)

    port_a, port_b = _ends(axis, body)
    wg = RectWaveguide(body=body, port_a=port_a, port_b=port_b)

    if add_ports:
        wg.ports = [RectWaveguidePort(end, mode=mode, er=er, power=power)
                    for end in (port_a, port_b)]
        wg.pec = PEC(*body.faces.unassigned)

    return wg


@dataclass(kw_only=True)
class CircWaveguide(_Section):
    """Result of :func:`circ_waveguide`.

    Attributes
    ----------
    body : GeoObject
        the cylindrical fill volume
    port_a, port_b : EntityCollection
        the end-cap faces (the two waveguide ports)
    pec : object or None
        the PEC on the curved wall when ``add_ports`` (else None)
    ports : list
        the two :class:`rapidfem.WavePort` objects when ``add_ports``
    """

    body: "GeoObject"


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
    av = _axis("circ_waveguide", axis)
    _check_f0("circ_waveguide", add_ports, f0)
    fill, _ = _fill(er, material)
    body = g.cylinder(radius, length, position=origin, axis=av, material=fill)

    port_a, port_b = _ends(axis, body)
    wg = CircWaveguide(body=body, port_a=port_a, port_b=port_b)

    if add_ports:
        _wave_ports(wg, f0, power)
        wg.pec = PEC(*body.faces.unassigned)

    return wg


# SWEPT CONDUCTORS ======================================================================

def sweep_along_path(g: "Geometry", profile: "GeoObject",
                     points: "list[tuple[float, float, float]]",
                     *,
                     material=None,
                     maxh: float | None = None) -> "GeoObject":
    """sweep a round ``profile`` disc along the spline through ``points`` into
    a 3-D solid.

    The workhorse behind curved conductors: bond wires, coax bends,
    helices. The ``profile`` disc (from :meth:`Geometry.disc`) must be
    positioned at ``points[0]`` with its normal along the initial path
    tangent, so the swept tube starts flush with it; the profile is used up.
    The tube is faceted (a 16-gon cross-section) along a centripetal
    Catmull-Rom spline sampled 8 times per span.


    Example
    -------
    A round bond wire arcing between two pads:

    .. code-block:: python

        from rapidfem import structures as st
        pts = [(0, 0, 0), (0.5e-3, 0, 0.4e-3), (1e-3, 0, 0)]
        prof = g.disc(50e-6, position=pts[0], axis=(0, 0, 1))
        wire = st.sweep_along_path(g, prof, pts)


    Parameters
    ----------
    g : Geometry
        geometry to build into
    profile : GeoObject
        the round cross-section, a disc
    points : list[tuple[float, float, float]]
        path control points in metres; a spline is fitted through them (a
        straight segment for two points)
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
        if ``profile`` is not an unmoved disc or fewer than two points are
        given
    """
    oid = g._native.sweep(profile._id, [tuple(float(c) for c in p) for p in points], maxh)
    return g._wrap(oid, material)


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
    and reorient with :meth:`Geometry.rotate` / :meth:`Geometry.translate`.

    Useful for inductors and helical antennas. The wire is faceted (a
    12-gon cross-section).


    Example
    -------
    A 5-turn coil, 2 mm radius, 1 mm pitch, 0.1 mm wire:

    .. code-block:: python

        from rapidfem import structures as st
        coil = st.helix(g, radius=2e-3, pitch=1e-3, turns=5, wire_radius=0.1e-3)


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
        the swept helical wire

    Raises
    ------
    ValueError
        if ``turns`` or ``points_per_turn`` are non-positive
    """
    if turns <= 0:
        raise ValueError(f"helix: turns must be > 0, got {turns}")
    if points_per_turn < 2:
        raise ValueError(f"helix: points_per_turn must be >= 2, got {points_per_turn}")
    oid = g._native.add_helix(radius, pitch, turns, wire_radius, position=tuple(position),
                              points_per_turn=int(points_per_turn), maxh=maxh)
    return g._wrap(oid, material)
