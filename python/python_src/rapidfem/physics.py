# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

#########################################################################################
##
##                            PORTS AND BOUNDARY CONDITIONS
##                                  (physics.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from __future__ import annotations

import warnings

from typing import Sequence

from .geometry import EntityCollection, GeoObject, _Entity


# HELPERS ===============================================================================


def _normalize(targets, *, expected_dim: int, cls_name: str):
    """flatten variadic geometry args to a list of _Entity + their Geometry

    Accepts :class:`GeoObject`, :class:`EntityCollection`, individual
    ``_Entity``, and any combination of those. All resolved entities
    must belong to the same :class:`Geometry` and (if ``expected_dim``
    is set) carry that dim.

    Parameters
    ----------
    targets : iterable of GeoObject, EntityCollection, or _Entity
        physics targets, variadic
    expected_dim : int
        2 for faces, 3 for volumes
    cls_name : str
        name of the calling physics class, for error messages

    Returns
    -------
    entities : list[_Entity]
        flattened target list
    geom : Geometry
        the geometry instance every target belongs to
    """
    entities: list[_Entity] = []
    geom = None
    for t in targets:
        if isinstance(t, GeoObject):
            entities.append(t._entity)
            geom = t._geometry if geom is None else geom
            if geom is not t._geometry:
                raise ValueError(
                    f"{cls_name}: targets span multiple Geometry instances")
        elif isinstance(t, EntityCollection):
            entities.extend(t._entities)
            if geom is None:
                geom = t._geometry
            elif geom is not t._geometry:
                raise ValueError(
                    f"{cls_name}: targets span multiple Geometry instances")
        elif isinstance(t, _Entity):
            entities.append(t)
            if t._geometry is None:
                raise ValueError(
                    f"{cls_name}: bare _Entity without Geometry back-ref")
            if geom is None:
                geom = t._geometry
            elif geom is not t._geometry:
                raise ValueError(
                    f"{cls_name}: targets span multiple Geometry instances")
        else:
            raise TypeError(
                f"{cls_name}: cannot use {type(t).__name__} as a target")

    if not entities:
        raise ValueError(f"{cls_name}: no targets")
    if geom is None:
        raise ValueError(f"{cls_name}: could not resolve target Geometry")
    for e in entities:
        if e.dim != expected_dim:
            kind = {2: "face", 3: "volume"}.get(expected_dim, f"dim={expected_dim}")
            raise ValueError(
                f"{cls_name}: expected {kind} targets, got dim={e.dim}")
    return entities, geom


# BASE CLASS ============================================================================

class _Physics:
    """Common base for every driven port and boundary condition.

    Subclasses set ``_expected_dim`` (2 for face physics, 3 for volume
    physics) and implement :meth:`_add_to`, which places the object on the
    native :class:`rapidfem._native.Model` under its mesh tag.


    Note
    ----
    Constructors take their target entities as the first positional
    arguments (variadic) and physics parameters as keyword arguments.
    The object registers itself with the target's :class:`Geometry` on
    ``__init__``; no further wiring step is required.

    The physics object is purely declarative, it holds no state about
    the mesh. The geometry's :meth:`Geometry.mesh` step turns it into a
    tagged group of mesh faces or tets, and :class:`rapidfem.Problem`
    reads that group tag back when it builds the model.
    """
    _expected_dim: int = 2

    def __init__(self, *targets):
        ents, geom = _normalize(targets,
                                expected_dim=self._expected_dim,
                                cls_name=type(self).__name__)
        self._entities = ents
        self._geometry = geom
        geom._physics.append(self)

    def _add_to(self, model, tag) -> None:
        """place this object on the native model under its mesh tag

        Parameters
        ----------
        model : rapidfem._native.Model
            the model being built
        tag : int
            physical-group tag assigned by ``Geometry.mesh()``
        """
        raise NotImplementedError


# DRIVEN PORTS ==========================================================================

class RectWaveguidePort(_Physics):
    """Analytic TE-mode driven port on a rectangular waveguide face.

    The port plane carries the closed-form
    :math:`\\mathrm{TE}_{mn}` mode of a rectangular waveguide with
    transverse dimensions :math:`(a, b)`. The transverse electric
    field for the dominant :math:`\\mathrm{TE}_{10}` mode is

    .. math::

        \\mathbf{E}_t(x, y) = \\hat{\\mathbf{y}}
            \\sin\\!\\left(\\frac{\\pi x}{a}\\right)

    with cutoff
    :math:`f_{c, mn} = \\frac{c}{2 \\sqrt{\\varepsilon_r}}
    \\sqrt{(m/a)^2 + (n/b)^2}`. Cross-section dimensions auto-detect
    from the port face bounding-box when ``width`` and ``height`` are
    left at 0.


    Example
    -------
    WR-90 with TE10 ports on both ends of an air box:

    .. code-block:: python

        air = g.box(A, B, L, material=rf.Air())
        rf.RectWaveguidePort(air.faces.min(axis="z"))
        rf.RectWaveguidePort(air.faces.max(axis="z"))


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        port face(s)
    mode : tuple[int, int]
        :math:`(m, n)` TE-mode indices (defaults to :math:`(1, 0)`)
    er : float
        relative permittivity inside the waveguide
    power : float
        incident power in watts
    width : float
        cross-section width override in metres (0 means auto-detect)
    height : float
        cross-section height override in metres (0 means auto-detect)
    """

    def __init__(self, *targets,
                 mode: tuple[int, int] = (1, 0),
                 er: float = 1.0,
                 power: float = 1.0,
                 width: float = 0.0,
                 height: float = 0.0):
        super().__init__(*targets)
        self.mode = (int(mode[0]), int(mode[1]))
        self.er = float(er)
        self.power = float(power)
        self.width = float(width)
        self.height = float(height)

    def _add_to(self, model, tag) -> None:
        model.add_rect_port(tag, width=self.width, height=self.height,
                            mode=list(self.mode), er=self.er, power=self.power)
class LumpedPort(_Physics):
    """Lumped voltage-source driven port between two PEC conductors.

    Models a delta-gap source bridging two conductors (e.g. a ground
    plane and a microstrip trace). The port plane spans the gap; the port
    voltage is the **area-averaged mode projection** of the solved field
    over the whole port surface,

    .. math::

        V = \\frac{1}{w} \\int_{\\text{port}}
            \\mathbf{E} \\cdot \\hat{\\ell}\\; dS ,

    (with :math:`w = A/\\ell` the port width), which stays well defined for
    tall / non-TEM ports where a single line integral would degenerate.
    The S-parameter normalises to the reference resistance :math:`R = z_0`.

    The port termination is a series **R-L-C**: :math:`Z(\\omega) = R +
    j\\omega L + 1/(j\\omega C)`. With ``l = 0`` and ``c = None`` it is the
    pure resistive reference port; ``l`` / ``c`` add a reactive termination.
    Derivation: ``derivations/lumped_port/``.


    Example
    -------
    Vertical feed plate bridging substrate to a patch antenna:

    .. code-block:: python

        feed = g.plate(p0=(0, -L/2, 0),
                       width=(W, 0, 0),
                       height=(0, 0, H))
        rf.LumpedPort(feed, direction=(0, 0, 1), z0=50.0)

        # reactive termination: 50 Ω in series with 0.1 nH
        rf.LumpedPort(feed, direction=(0, 0, 1), z0=50.0, l=0.1e-9)


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        port face(s)
    direction : Sequence[float]
        voltage-integration axis (3-vector)
    z0 : float
        reference resistance R in ohms (S-parameter reference)
    l : float
        series termination inductance in henries (0 means none)
    c : float, optional
        series termination capacitance in farads (None means none)
    power : float
        incident power in watts
    width : float
        port extent override in metres (0 means auto-detect)
    height : float
        port extent override in metres (0 means auto-detect)
    """

    def __init__(self, *targets,
                 direction: Sequence[float],
                 z0: float = 50.0,
                 l: float = 0.0,
                 c: float | None = None,
                 power: float = 1.0,
                 width: float = 0.0,
                 height: float = 0.0):
        super().__init__(*targets)
        self.direction = tuple(float(v) for v in direction)
        self.z0 = float(z0)
        self.l = float(l)
        self.c = None if c is None else float(c)
        self.power = float(power)
        self.width = float(width)
        self.height = float(height)

    def _add_to(self, model, tag) -> None:
        model.add_lumped_port(tag, z0=self.z0, l=self.l, c=self.c,
                              direction=list(self.direction), width=self.width,
                              height=self.height, power=self.power)
class CoaxPort(_Physics):
    """TEM-mode driven port on a coaxial annular face.

    Drives the analytic TEM mode of a coaxial transmission line with
    inner radius :math:`r_i` and outer radius :math:`r_o`. The
    transverse electric field is purely radial,

    .. math::

        \\mathbf{E}_t(\\rho) = \\frac{\\hat{\\boldsymbol{\\rho}}}
            {\\rho \\ln(r_o / r_i)}

    and the characteristic impedance is
    :math:`Z_0 = \\frac{\\eta_0}{2 \\pi \\sqrt{\\varepsilon_r}}
    \\ln(r_o / r_i)`. Origin and axis default to the port face
    bounding-box centre and +z.


    Example
    -------
    50 Ω air coax section with ports at both flat ends:

    .. code-block:: python

        air = g.cylinder(radius=ro, height=L, material=rf.Air())
        rf.CoaxPort(air.faces.min(axis="z"), ri=ri, ro=ro)
        rf.CoaxPort(air.faces.max(axis="z"), ri=ri, ro=ro,
                    origin=(0, 0, L))


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        port face(s)
    ri : float
        inner coax radius in metres
    ro : float
        outer coax radius in metres
    origin : Sequence[float], optional
        a point on the coax axis (defaults to the port-face centroid)
    z_axis : Sequence[float], optional
        coax axis direction (defaults to +z)
    er : float
        relative permittivity of the coax dielectric
    power : float
        incident power in watts
    """

    def __init__(self, *targets,
                 ri: float,
                 ro: float,
                 origin: Sequence[float] | None = None,
                 z_axis: Sequence[float] | None = None,
                 er: float = 1.0,
                 power: float = 1.0):
        super().__init__(*targets)
        self.ri = float(ri)
        self.ro = float(ro)
        self.origin = tuple(float(v) for v in origin) if origin is not None else None
        self.z_axis = tuple(float(v) for v in z_axis) if z_axis is not None else None
        self.er = float(er)
        self.power = float(power)

    def _add_to(self, model, tag) -> None:
        model.add_coax_port(
            tag, ri=self.ri, ro=self.ro, er=self.er, power=self.power,
            origin=None if self.origin is None else list(self.origin),
            z_axis=None if self.z_axis is None else list(self.z_axis))
class WavePort(_Physics):
    """Numerically-solved wave port, 2-D mode eigensolve on the port face.

    Computes the port's transverse mode profile by a 2-D eigensolve on
    the port-face triangulation, instead of assuming an analytic shape.
    This is the right port for a guide whose mode has no closed form: a
    ridged or arbitrary hollow waveguide (scalar :math:`\\mathrm{TE}` /
    :math:`\\mathrm{TM}` path) and a microstrip / coplanar / coupled
    line (full-vector hybrid path with per-element :math:`\\varepsilon_r`).
    The solved profile flows through the same Robin-BC injection and
    mode-projection extraction as :class:`RectWaveguidePort` and
    :class:`CoaxPort`.

    Two solver paths, selected via ``mode_kind``:

    - ``"auto"`` / ``"vector"`` / ``"hybrid"`` (default), **full-vector
      hybrid** eigenproblem (mixed Nédélec-edge :math:`E_t` + Lagrange-
      nodal :math:`E_z`). Honours per-element :math:`\\varepsilon_r` so
      the inhomogeneous quasi-TEM mode of a microstrip-class line is
      captured directly. Pair with ``pec=`` to mark any internal PEC
      conductor (the trace) that bisects the cross-section. Dispersion
      uses :math:`\\beta(k_0) = n_{\\mathrm{eff}}(f_0) \\cdot k_0`
      throughout the sweep, set ``f0`` near band centre.
    - ``"te"`` / ``"tm"``, **scalar Helmholtz** TE / TM modes on the
      homogeneously filled hollow cross-section. Cutoff and dispersion
      come from the scalar :math:`k_c`; weak frequency dependence so
      ``f0`` choice barely matters for the mode shape.

    Both paths support frequency-domain (:class:`ProblemFD`); the time-
    domain backend currently only consumes ``"te"`` / ``"tm"``.


    Example
    -------
    Microstrip line driven by a hybrid wave port at each end, with the
    trace marked as internal PEC inside the cross-section:

    .. code-block:: python

        sub = g.box(W, L, H_sub, material=fr4)
        air = g.box(W, L, H_air, position=(0, 0, H_sub), material=rf.Air())
        trace = g.xy_plate(w0, L, position=(-w0/2, 0, H_sub))
        g.fragment(sub, air, trace)

        pec_trace = rf.PEC(trace, sub.faces.min(axis="z"))
        rf.WavePort(sub.faces.min(axis="y"), air.faces.min(axis="y"),
                    f0=10e9, mode_kind="auto", pec=[pec_trace])
        rf.WavePort(sub.faces.max(axis="y"), air.faces.max(axis="y"),
                    f0=10e9, mode_kind="auto", pec=[pec_trace])

    Hollow-guide variant, dominant TE mode of an arbitrary cross-section:

    .. code-block:: python

        rf.WavePort(guide.faces.min(axis="z"), f0=10e9, mode_kind="te")


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        port face(s), multiple faces are merged into one cross-section
    mode_kind : str, optional
        ``"auto"`` / ``"vector"`` / ``"hybrid"`` (default) for the
        full-vector hybrid solve, ``"te"`` / ``"tm"`` for the scalar
        Helmholtz path.
    mode_index : int
        which mode to use, ordered by descending :math:`n_{\\mathrm{eff}}`
        (vector path) or ascending cutoff (scalar path). ``0`` = dominant.
    f0 : float
        eigensolve operating frequency in Hz. Required for the FD backend,
        the vector mode profile is computed once here and used across
        the whole sweep.
    pec : iterable of :class:`PEC`, optional
        PEC physics objects whose nodes on the port face are marked as
        internal conductors (a microstrip trace cutting through the
        cross-section). Outer-boundary PEC is inferred automatically;
        this is only for *internal* PEC.
    power : float
        incident power in watts (default ``1.0``)
    te : bool, optional
        legacy backwards-compat flag, superseded by ``mode_kind``.
    """

    def __init__(self, *targets,
                 te: bool = True,
                 mode_index: int = 0,
                 f0: float | None = None,
                 power: float = 1.0,
                 mode_kind: str | None = None,
                 pec: "Iterable | None" = None):
        super().__init__(*targets)
        self.te = bool(te)
        self.mode_index = int(mode_index)
        self.f0 = None if f0 is None else float(f0)
        self.power = float(power)
        if mode_kind is not None:
            self.mode_kind = str(mode_kind).lower()
        else:
            self.mode_kind = "auto"
        self.pec = list(pec) if pec is not None else []

    def _add_to(self, model, tag) -> None:
        # The cross-section solve: an explicit te / tm, otherwise the vector
        # solve at f0, or without f0 the scalar solve picked by ``te``.
        if self.mode_kind in ("te", "tm"):
            kind = self.mode_kind
        elif self.f0 is None:
            kind = "te" if self.te else "tm"
        else:
            kind = "vector"
        # Attached PEC objects resolve to their physical-group tags so the
        # cross-section eigensolve can mark those nodes as internal
        # conductors; Geometry.mesh() populates `_physics_tags`.
        geom = self._geometry
        pec_tags = [geom._physics_tags[id(p)] for p in self.pec
                    if isinstance(geom._physics_tags.get(id(p)), int)]
        model.add_wave_port(tag, kind=kind, mode_index=self.mode_index,
                            power=self.power, f0=self.f0, pec_tags=pec_tags)
class UserDefinedPort(_Physics):
    """Driven port with a user-supplied uniform E-field on the face.

    Escape hatch for non-standard cross-sections where the analytic
    rectangular / coaxial / Floquet ports do not apply. The user
    specifies a constant electric field vector that's imposed across
    the port face, plus a normalisation power.


    Example
    -------
    .. code-block:: python

        rf.UserDefinedPort(face,
            e_field=(0, 1, 0),
            power=1.0,
        )


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        port face(s)
    e_field : Sequence[float]
        imposed electric field vector on the port face
    power : float
        normalisation power in watts
    """

    def __init__(self, *targets,
                 e_field: Sequence[float],
                 power: float = 1.0):
        super().__init__(*targets)
        self.e_field = tuple(float(v) for v in e_field)
        self.power = float(power)

    def _add_to(self, model, tag) -> None:
        model.add_user_port(tag, e_field=list(self.e_field), power=self.power)
class FloquetPort(_Physics):
    """Floquet plane-wave port for periodic unit cells.

    Drives a periodic structure with an oblique plane wave at scan
    angles :math:`(\\theta, \\phi)`. The Floquet mode has the form

    .. math::

        \\mathbf{E}(x, y, z) = \\mathbf{E}_0
            e^{-j(k_x x + k_y y + k_z z)}

    with :math:`(k_x, k_y) = k_0 \\sin\\theta\\,(\\cos\\phi,
    \\sin\\phi)` and :math:`k_z` chosen for the desired Floquet mode
    index. Useful for frequency-selective surfaces, reflectarrays, and
    phased-array unit cells.


    Example
    -------
    Normal-incidence Floquet port on the top face of a unit cell:

    .. code-block:: python

        rf.FloquetPort(air.faces.max(axis="z"),
            scan_theta_deg=0,
            scan_phi_deg=0,
        )


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        port face(s) (typically the top or bottom of the unit cell)
    scan_theta_deg : float
        elevation scan angle :math:`\\theta` in degrees
    scan_phi_deg : float
        azimuth scan angle :math:`\\phi` in degrees
    mode_nr : int
        Floquet mode index (1 = fundamental)
    er : float
        relative permittivity of the port medium
    power : float
        incident power in watts
    """

    def __init__(self, *targets,
                 scan_theta_deg: float = 0.0,
                 scan_phi_deg: float = 0.0,
                 mode_nr: int = 1,
                 er: float = 1.0,
                 power: float = 1.0):
        super().__init__(*targets)
        self.scan_theta_deg = float(scan_theta_deg)
        self.scan_phi_deg = float(scan_phi_deg)
        self.mode_nr = int(mode_nr)
        self.er = float(er)
        self.power = float(power)

    def _add_to(self, model, tag) -> None:
        model.add_floquet_port(tag, scan_theta_deg=self.scan_theta_deg,
                               scan_phi_deg=self.scan_phi_deg,
                               mode_nr=self.mode_nr, er=self.er, power=self.power)


# BOUNDARY CONDITIONS ===================================================================

class PEC(_Physics):
    """Perfect electric conductor.

    Enforces the tangential-field condition

    .. math::

        \\hat{\\mathbf{n}} \\times \\mathbf{E} = \\mathbf{0}

    on every targeted face. Variadic constructor: pass any mix of
    :class:`GeoObject`, :class:`EntityCollection`, or single faces;
    they all share one :class:`Problem`-level ``[pec]`` block.


    Note
    ----
    Multiple ``rf.PEC(...)`` calls in the same problem are aggregated
    into one TOML ``[pec]`` block when :class:`Problem` assembles the
    config, so you can spread declarations across several lines for
    readability without worrying about runtime overhead.


    Example
    -------
    Patch antenna's conductors plus the substrate's ground plane:

    .. code-block:: python

        rf.PEC(patch_plate, sub.faces.min(axis="z"))


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        face(s) to mark as PEC, variadic
    """

    def _add_to(self, model, tag) -> None:
        model.add_pec(tag)
class PMC(_Physics):
    """Perfect magnetic conductor, symmetry boundary.

    Dual to :class:`PEC`, enforcing

    .. math::

        \\hat{\\mathbf{n}} \\times \\mathbf{H} = \\mathbf{0}

    Mostly useful as a symmetry plane when the problem's magnetic
    field is tangential to a plane (so it doesn't penetrate). Lets
    you mesh only half of a symmetric structure.


    Example
    -------
    .. code-block:: python

        rf.PMC(air.faces.min(axis="y"))


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        face(s) to mark as PMC, variadic
    """

    def _add_to(self, model, tag) -> None:
        model.add_pmc(tag)
class ABC(_Physics):
    """First-order absorbing boundary condition.

    Surface-level radiation boundary that approximates outgoing-wave
    behaviour without the cost of a volumetric absorber. The first-order
    Sommerfeld ABC enforces

    .. math::

        \\hat{\\mathbf{n}} \\times (\\nabla \\times \\mathbf{E})
            + j k_0\\, \\hat{\\mathbf{n}} \\times
            (\\hat{\\mathbf{n}} \\times \\mathbf{E}) = \\mathbf{0}

    It is a plain matched-impedance sheet, dissipative by construction, so
    ``|S| ≤ 1`` always.


    Note
    ----
    For strong absorption at the radiating face of an antenna prefer
    :class:`PML`. ABC works best when the boundary sees nearly normal
    incidence (e.g. outer faces several wavelengths away from the
    source). (A second-order ABC was removed: its Bayliss-Turkel correction
    is indefinite and not unconditionally passive, and the passivity-safe
    projection of it gave no real accuracy gain over first order, use
    :class:`PML` when first order reflects too much.)


    Example
    -------
    First-order ABC on the air-box outer hull:

    .. code-block:: python

        rf.ABC(*air.faces.outer)


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        face(s) to terminate
    """

    def __init__(self, *targets):
        super().__init__(*targets)

    def _add_to(self, model, tag) -> None:
        model.add_abc(tag)
class FarFieldSurface(_Physics):
    """Near-field-to-far-field (Huygens) integration surface.

    Marks a *closed* surface enclosing the radiator on which the
    equivalent electric and magnetic currents
    :math:`\\mathbf{J}_s = \\hat{\\mathbf{n}} \\times \\mathbf{H}`,
    :math:`\\mathbf{M}_s = -\\hat{\\mathbf{n}} \\times \\mathbf{E}` are
    sampled. :meth:`rapidfem.Problem.farfield` propagates those currents
    to the far zone via the Stratton-Chu integral.

    When the domain is truncated by an :class:`ABC`, its outer boundary
    is the Huygens surface and the solver takes it by itself, so no
    ``FarFieldSurface`` is needed. With a :class:`PML` there is no such
    surface (the outer hull is PEC-backed absorber, not free space), so
    you must mark one explicitly: the bulk-air / PML interface is the
    natural choice, a closed box sitting just inside the absorber.

    A ground or symmetry plane on the domain boundary (the PEC floor an
    antenna rests on, a PMC half-model plane) is taken as infinite: the
    parts of the surface lying on it are replaced by the images of the
    rest, and the pattern fills the half-space of the domain. This holds
    when every conducting piece of the surface lies in one plane.


    Note
    ----
    The surface must be closed and must fully enclose every radiating
    feature. Use :attr:`~rapidfem.geometry.EntityCollection.hull` (not
    ``outer``) to grab the air-box boundary: once the air box is wrapped
    in PML on every side none of its faces touch the model bounding box,
    so ``air.faces.outer`` is empty, while ``air.faces.hull`` keys off the
    air box's own bounding box and returns the air / PML interface faces.


    Example
    -------
    Far-field surface on the air / PML interface of a PML-truncated
    antenna problem:

    .. code-block:: python

        air = g.box(AW, AL, AH, material=rf.Air())
        # ... PML slabs around it ...
        rf.FarFieldSurface(*air.faces.hull)


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        face(s) forming the closed integration surface, variadic
    """

    def __init__(self, *targets):
        super().__init__(*targets)

    def _add_to(self, model, tag) -> None:
        model.set_far_field(tag)
class SurfaceImpedance(_Physics):
    """Surface impedance boundary for thin lossy conductors.

    Replaces the volumetric mesh of a thin metal sheet by a 2-D
    impedance condition

    .. math::

        \\mathbf{E}_t = Z_s\\,(\\hat{\\mathbf{n}} \\times \\mathbf{H})

    For a good conductor with skin depth
    :math:`\\delta = \\sqrt{2 / (\\omega \\mu \\sigma)}` and
    thickness :math:`t \\gg \\delta` the analytic surface impedance is

    .. math::

        Z_s = (1 + j) \\sqrt{\\frac{\\omega \\mu}{2 \\sigma}}

    Pass either the bulk parameters (``conductivity``, ``mur``,
    ``er``, optional ``thickness`` for a finite sheet) and let
    the solver compute :math:`Z_s` analytically, or override with
    an explicit ``zs = (re, im)`` in :math:`\\Omega/\\square`.

    The finite-thickness correction depends on where the face sits:

    * ``two_sided=False`` (default) — a **boundary** face with fields on one
      side only (a ground plane on the domain boundary). The face owns the
      full metal, :math:`Z = Z_{s,\\infty} \\coth(\\gamma_m t)`,
      :math:`1/(\\sigma t)` at DC.
    * ``two_sided=True`` — a **wall** of a conductor that carries the BC on
      opposing faces (the walls of a conductor cut out of the mesh). Each
      face owns half the metal, :math:`Z = Z_{s,\\infty}
      \\coth(\\gamma_m t/2)`; opposing faces in parallel recover
      :math:`1/(\\sigma t)`. For a trace of width :math:`w` pass the
      volume-to-surface thickness :math:`t_\\mathrm{eff} = w t/(w + t)`
      (``2V/S``), not the layer thickness, or the sidewalls add conductance
      and the DC resistance comes out a factor :math:`w/(w+t)` low.
    * ``sheet=True`` — a zero-thickness **sheet** embedded in the volume
      with fields on both sides, standing in for a strip of thickness
      :math:`t`: :math:`Z = Z_{s,\\infty} \\coth(\\gamma_m t/2)/2`, the
      even mode of the slab (the two faces in parallel), :math:`1/(\\sigma t)`
      at DC and :math:`Z_{s,\\infty}/2` in the skin-effect limit. Exact for
      films thinner than the skin depth; for a thick strip whose current
      runs on one face (a microstrip over ground) the default one-sided
      value is the better approximation.


    Note
    ----
    A surface impedance on the walls of a finite-thickness strip is accurate
    where the metal is thinner than about 1.5 or thicker than about 4 skin
    depths: on one-sided walls (a conductor hollowed out of the mesh) the
    impedance of the current along a convex edge rises within a few skin
    depths of it, the current crowding a per-face impedance misses, and the
    solver adds that from a universal profile (within 3 % of a 2D
    quasi-static reference from 4 skin depths up, issues #48 and #56). In
    between, and across conductors narrower than 4 skin depths, it
    underestimates the strip resistance by up to about 25 %; mesh the
    conductor as a volume there. :func:`rapidfem.rfic.build` makes this
    choice per layer from its ``band``. Never apply the BC to the faces of a
    conductor whose interior is still meshed: on internal faces it acts as a
    transition sheet and lets the field into the core.


    Example
    -------
    Copper surface on a stripline ground (an outer boundary face):

    .. code-block:: python

        rf.SurfaceImpedance(ground_face, conductivity=5.8e7)

    A 1 um thin-film strip as an embedded zero-thickness sheet:

    .. code-block:: python

        strip = g.xy_plate(w, length, position=(...))
        rf.SurfaceImpedance(strip, conductivity=3e7, thickness=1e-6,
                            sheet=True)


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        face(s) to apply the BC to
    conductivity : float
        bulk conductivity in S/m
    mur : float
        relative permeability
    er : float
        relative permittivity
    thickness : float, optional
        physical sheet thickness in metres (lossy thin-sheet model)
    two_sided : bool, optional
        the BC sits on opposing faces of the same metal volume (shell of an
        extruded conductor): each face owns half the thickness in the coth
        term. Keep ``False`` for one-sided boundary sheets (ground planes).
        Defaults to ``False``; if left unspecified while ``thickness`` is set
        and the targets cover the complete shell of a solid, a
        :class:`UserWarning` suggests the physically correct choice.
    sheet : bool, optional
        the target is a zero-thickness sheet with fields on both sides that
        stands in for a strip of ``thickness`` (see above)
    zs : tuple[float, float], optional
        explicit ``(Re, Im)`` surface impedance in :math:`\\Omega/\\square`,
        overrides the analytic value
    """

    def __init__(self, *targets,
                 conductivity: float = 0.0,
                 mur: float = 1.0,
                 er: float = 1.0,
                 thickness: float | None = None,
                 two_sided: bool | None = None,
                 sheet: bool = False,
                 zs: tuple[float, float] | None = None):
        super().__init__(*targets)
        self.conductivity = float(conductivity)
        self.mur = float(mur)
        self.er = float(er)
        self.thickness = float(thickness) if thickness is not None else None
        self.two_sided = bool(two_sided) if two_sided is not None else False
        self.sheet = bool(sheet)
        if self.sheet and self.two_sided:
            raise ValueError("SurfaceImpedance: sheet and two_sided exclude each other")
        self.zs = (float(zs[0]), float(zs[1])) if zs is not None else None
        if self._covers_solid_shell():
            warnings.warn(
                "SurfaceImpedance: the targets cover the complete shell of a "
                "solid that is still meshed. On internal faces the BC is a "
                "transition sheet and the field enters the conductor core, so "
                "the loss comes out wrong. Cut the conductor out of the mesh "
                "and put the BC on the walls (two_sided=True with "
                "thickness=2V/S), or mesh it as a volume conductor.",
                UserWarning, stacklevel=2)

    def _covers_solid_shell(self) -> bool:
        """True if the target faces include the complete shell of at least
        one 3-D object of the geometry that is still part of the mesh."""
        ids = {id(e) for e in self._entities}
        try:
            for obj in getattr(self._geometry, "_objects", []):
                if getattr(obj, "dim", None) != 3:
                    continue
                if self._geometry._native.is_void(obj._id):
                    continue            # a hole: its walls are the boundary
                shell = obj.faces._entities
                if shell and all(id(e) in ids for e in shell):
                    return True
        except Exception:
            return False
        return False

    def _add_to(self, model, tag) -> None:
        model.add_surface_impedance(
            tag, conductivity=self.conductivity, mur=self.mur, er=self.er,
            thickness=self.thickness, two_sided=self.two_sided,
            sheet=self.sheet, zs=None if self.zs is None else list(self.zs))
class LumpedElement(_Physics):
    """Series chip R-L-C element on a 2-D footprint.

    Embeds a series-RLC element across a named face, typically used
    for isolation resistors (Wilkinson dividers), shunt caps to ground,
    or matching networks. The element impedance is

    .. math::

        Z(\\omega) = R + j \\omega L + \\frac{1}{j \\omega C}

    with C optional. The current-flow direction across the element
    must be supplied explicitly via ``direction``.


    Example
    -------
    100 Ω isolation resistor across a Wilkinson port gap:

    .. code-block:: python

        rf.LumpedElement(gap_face, r=100.0, direction=(0, 1, 0))


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        face(s) hosting the element
    r : float
        series resistance in ohms
    l : float
        series inductance in henries
    c : float, optional
        series capacitance in farads (``None`` means no capacitor)
    direction : Sequence[float]
        current-flow direction across the element
    width : float
        element footprint width override in metres (0 = auto-detect)
    height : float
        element footprint height override in metres (0 = auto-detect)
    """

    def __init__(self, *targets,
                 r: float = 0.0,
                 l: float = 0.0,
                 c: float | None = None,
                 direction: Sequence[float] = (0.0, 0.0, 1.0),
                 width: float = 0.0,
                 height: float = 0.0):
        super().__init__(*targets)
        self.r = float(r)
        self.l = float(l)
        self.c = float(c) if c is not None else None
        self.direction = tuple(float(v) for v in direction)
        self.width = float(width)
        self.height = float(height)

    def _add_to(self, model, tag) -> None:
        model.add_lumped_element(tag, r=self.r, l=self.l, c=self.c,
                                 direction=list(self.direction),
                                 width=self.width, height=self.height)
class PML(_Physics):
    """Coordinate-stretched anisotropic Perfectly Matched Layer.

    Volumetric absorbing region that terminates the computational
    domain with vastly less reflection than a surface ABC. The PML
    applies a complex coordinate stretch along ``direction``,

    .. math::

        s(\\rho) = 1 + \\delta_{\\max}
            \\left( \\frac{\\rho - \\rho_0}{d} \\right)^n

    with :math:`\\rho_0` the inner-face coordinate, :math:`d` the
    slab thickness, :math:`n` the polynomial exponent (typical 1.5-3),
    and :math:`\\delta_{\\max}` the peak stretch magnitude at the
    outer face (typical 5-12).


    Note
    ----
    PML lives on a *volume* (dim=3), not a surface. Build it as an
    extra cuboid attached to the air region; assign a placeholder
    material (e.g. :class:`Air`) so the volume gets meshed, then
    declare the PML BC on the volume, the BC's stretch overrides the
    bulk permittivity for the absorption profile.

    For a closed enclosure around an antenna use one PML slab per
    outer face; the slabs must not overlap (each volume can only carry
    one absorption direction).


    Example
    -------
    Single-sided +x PML in front of a horn antenna:

    .. code-block:: python

        pml_xp = g.box(PML_T, AIR_W, AIR_H,
                       position=(AIR_X1, 0, 0),
                       material=rf.Air(),
                       maxh=2 * MAXH)
        rf.PML(pml_xp,
               direction=(1, 0, 0),
               inner_face=AIR_X1,
               thickness=PML_T)


    Parameters
    ----------
    targets : GeoObject or EntityCollection
        volume(s) to turn into PML
    direction : Sequence[float]
        outward-pointing unit vector along the absorption axis
        (axis-aligned: one of :math:`\\pm\\hat{\\mathbf{x}},
        \\pm\\hat{\\mathbf{y}}, \\pm\\hat{\\mathbf{z}}`)
    inner_face : float
        coordinate of the PML's inner face along ``direction`` (m)
    thickness : float
        PML extent in metres beyond ``inner_face``
    er_base : float
        base relative permittivity inside the PML
    ur_base : float
        base relative permeability inside the PML
    exponent : float
        polynomial profile exponent (typical 1.5-3)
    delta_max : float
        peak stretch magnitude :math:`\\delta_{\\max}` at the outer
        face (typical 5-12)
    """
    _expected_dim = 3

    def __init__(self, *targets,
                 direction: Sequence[float],
                 inner_face: float,
                 thickness: float,
                 er_base: float = 1.0,
                 ur_base: float = 1.0,
                 exponent: float = 1.5,
                 delta_max: float = 8.0):
        super().__init__(*targets)
        self.direction = tuple(float(v) for v in direction)
        self.inner_face = float(inner_face)
        self.thickness = float(thickness)
        self.er_base = float(er_base)
        self.ur_base = float(ur_base)
        self.exponent = float(exponent)
        self.delta_max = float(delta_max)

    def _add_to(self, model, tag) -> None:
        model.add_pml(tag, direction=list(self.direction),
                      inner_face=self.inner_face, thickness=self.thickness,
                      er_base=self.er_base, ur_base=self.ur_base,
                      exponent=self.exponent, delta_max=self.delta_max)
class PeriodicBoundary(_Physics):
    """Normal-incidence periodic boundary pair (time-domain backend).

    Links two opposite mesh faces as a periodic pair: a DG face on either
    side sees the partner element across the period translation as its
    neighbour, and the existing interior-face numerical flux carries the
    coupling, no special-case kernel. The translation vector is inferred
    from the two faces' centroids, and per-face-node alignment is computed
    from the transverse coordinates after applying that translation.

    Real time domain only: no Floquet phase factor (that is the oblique
    scan case, handled by :class:`FloquetPort`). The two faces must
    geometrically match, same shape, same triangle count, same in-plane
    layout up to the period translation.

    Note
    ----
    A periodic-paired face cannot also be a port or PEC: it is wired into
    the interior-face flux path, so a port / PEC declaration on the same
    triangle is rejected by the time-domain operator at build time.

    Example
    -------
    Periodic unit cell in z, PEC on the side walls, top / bottom paired:

    .. code-block:: python

        air = g.box(W, H, L, material=rf.Air())
        rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"))
        rf.PeriodicBoundary(
            air.faces.min(axis="z"),
            air.faces.max(axis="z"),
        )

    Parameters
    ----------
    face_a, face_b : GeoObject, EntityCollection, or single face
        the two opposite faces of the periodic pair, unordered
    """

    def __init__(self, face_a, face_b):
        # Run the parent's _normalize on each side so the pair check is
        # symmetric and a face-pair object stays a single physics object
        # in the geometry's _physics list, rather than registering twice.
        ents_a, geom_a = _normalize([face_a],
                                    expected_dim=2,
                                    cls_name=type(self).__name__)
        ents_b, geom_b = _normalize([face_b],
                                    expected_dim=2,
                                    cls_name=type(self).__name__)
        if geom_a is not geom_b:
            raise ValueError(
                f"{type(self).__name__}: face_a and face_b must belong "
                f"to the same Geometry"
            )
        # The base class tagging machinery assumes one tag per
        # _Physics, but we need two (one per face) for a periodic pair.
        # Store the two entity lists separately and overload the geometry
        # registration: a single PeriodicBoundary registers as two
        # physical-group tags, one per face.
        self._entities_a = ents_a
        self._entities_b = ents_b
        # _entities is kept (the union) so downstream tag walkers still see
        # something sensible.
        self._entities = list(ents_a) + list(ents_b)
        self._geometry = geom_a
        geom_a._physics.append(self)


    def _add_to(self, model, tag) -> None:
        tag_a, tag_b = tag
        model.add_periodic(tag_a, tag_b)

__all__ = [
    "RectWaveguidePort", "LumpedPort", "CoaxPort", "WavePort",
    "UserDefinedPort", "FloquetPort",
    "PEC", "PMC", "ABC", "SurfaceImpedance", "LumpedElement", "PML",
    "PeriodicBoundary",
]
