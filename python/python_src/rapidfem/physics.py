#########################################################################################
##
##                            PORTS AND BOUNDARY CONDITIONS
##                                  (physics.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from __future__ import annotations

from typing import Sequence

from .geometry import EntityCollection, GeoObject, _Entity
from ._fmt import _f64


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

    Subclasses set two class attributes that drive the serialisation
    pipeline:

    - ``_expected_dim``: 2 for face physics, 3 for volume physics
    - ``_section``: ``"ports"``, ``"pec"``, or ``"pml"``, tells the
      :class:`rapidfem.Problem` TOML assembler which block this object
      belongs to


    Note
    ----
    Constructors take their target entities as the first positional
    arguments (variadic) and physics parameters as keyword arguments.
    The object registers itself with the target's :class:`Geometry` on
    ``__init__``; no further wiring step is required.

    The physics object is purely declarative, it holds no state about
    the mesh. The geometry's :meth:`Geometry.mesh` step turns it into a
    gmsh physical group, and :class:`rapidfem.Problem` reads that group
    tag back when assembling the TOML config.
    """
    _expected_dim: int = 2
    _section: str = "ports"

    def __init__(self, *targets):
        ents, geom = _normalize(targets,
                                expected_dim=self._expected_dim,
                                cls_name=type(self).__name__)
        self._entities = ents
        self._geometry = geom
        geom._physics.append(self)

    def _to_toml(self, tag: int) -> str:
        """render this physics object as a TOML block

        Parameters
        ----------
        tag : int
            physical-group tag assigned by ``Geometry.mesh()``

        Returns
        -------
        str
            TOML fragment; an empty string for :class:`PEC`, whose
            tags are aggregated by :class:`Problem`
        """
        return ""


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

    def _to_toml(self, tag: int) -> str:
        return (
            f'[[ports]]\ntype = "rectangular"\ntag = {tag}\n'
            f'mode = [{self.mode[0]}, {self.mode[1]}]\n'
            f'er = {_f64(self.er)}\npower = {_f64(self.power)}\n'
            f'width = {_f64(self.width)}\nheight = {_f64(self.height)}\n'
        )


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

    def _to_toml(self, tag: int) -> str:
        d = self.direction
        c_line = f'c = {_f64(self.c)}\n' if self.c is not None else ''
        return (
            f'[[ports]]\ntype = "lumped"\ntag = {tag}\n'
            f'z0 = {_f64(self.z0)}\nl = {_f64(self.l)}\n{c_line}'
            f'power = {_f64(self.power)}\n'
            f'direction = [{_f64(d[0])}, {_f64(d[1])}, {_f64(d[2])}]\n'
            f'width = {_f64(self.width)}\nheight = {_f64(self.height)}\n'
        )


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

    def _to_toml(self, tag: int) -> str:
        s = (
            f'[[ports]]\ntype = "coax"\ntag = {tag}\n'
            f'ri = {_f64(self.ri)}\nro = {_f64(self.ro)}\n'
            f'er = {_f64(self.er)}\npower = {_f64(self.power)}\n'
        )
        if self.origin is not None:
            o = self.origin
            s += f'origin = [{_f64(o[0])}, {_f64(o[1])}, {_f64(o[2])}]\n'
        if self.z_axis is not None:
            z = self.z_axis
            s += f'z_axis = [{_f64(z[0])}, {_f64(z[1])}, {_f64(z[2])}]\n'
        return s


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

    def _to_toml(self, tag: int) -> str:
        if self.f0 is None:
            raise ValueError(
                "WavePort requires f0= for the FD frequency-domain backend "
                "(the operating frequency of the 2-D mode eigensolve)."
            )
        # Resolve attached PEC objects to their physical-group tags so the
        # cross-section eigensolve can mark the corresponding nodes as
        # internal conductors. Geometry.mesh() populates `_physics_tags`,
        # so this only resolves once meshing has happened.
        pec_tags: list[int] = []
        geom = self._geometry
        for phys in self.pec:
            phys_tag = geom._physics_tags.get(id(phys))
            if isinstance(phys_tag, int):
                pec_tags.append(phys_tag)
        pec_list = "[" + ", ".join(str(t) for t in pec_tags) + "]"
        return (
            f'[[ports]]\ntype = "wave_numerical"\ntag = {tag}\n'
            f'f0 = {_f64(self.f0)}\n'
            f'mode_index = {self.mode_index}\n'
            f'mode_kind = "{self.mode_kind}"\n'
            f'pec_tags = {pec_list}\n'
            f'power = {_f64(self.power)}\n'
        )


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

    def _to_toml(self, tag: int) -> str:
        e = self.e_field
        return (
            f'[[ports]]\ntype = "user_defined"\ntag = {tag}\n'
            f'e_field = [{_f64(e[0])}, {_f64(e[1])}, {_f64(e[2])}]\n'
            f'power = {_f64(self.power)}\n'
        )


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

    def _to_toml(self, tag: int) -> str:
        return (
            f'[[ports]]\ntype = "floquet"\ntag = {tag}\n'
            f'scan_theta_deg = {_f64(self.scan_theta_deg)}\n'
            f'scan_phi_deg = {_f64(self.scan_phi_deg)}\n'
            f'mode_nr = {self.mode_nr}\n'
            f'er = {_f64(self.er)}\npower = {_f64(self.power)}\n'
        )


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
    _section = "pec"

    def _to_toml(self, tag: int) -> str:
        # PEC tags are aggregated by Problem into [pec] tags=[...]; emit nothing.
        return ""


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

    def _to_toml(self, tag: int) -> str:
        return f'[[ports]]\ntype = "pmc"\ntag = {tag}\n'


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

    def _to_toml(self, tag: int) -> str:
        return f'[[ports]]\ntype = "abc"\ntag = {tag}\n'


class FarFieldSurface(_Physics):
    """Near-field-to-far-field (Huygens) integration surface.

    Marks a *closed* surface enclosing the radiator on which the
    equivalent electric and magnetic currents
    :math:`\\mathbf{J}_s = \\hat{\\mathbf{n}} \\times \\mathbf{H}`,
    :math:`\\mathbf{M}_s = -\\hat{\\mathbf{n}} \\times \\mathbf{E}` are
    sampled. :meth:`rapidfem.Problem.farfield` propagates those currents
    to the far zone via the Stratton-Chu integral.

    When the domain is truncated by an :class:`ABC`, the ABC face is
    already a closed Huygens surface and the solver auto-detects it, so
    no ``FarFieldSurface`` is needed. With a :class:`PML` there is no such
    surface (the outer hull is PEC-backed absorber, not free space), so
    you must mark one explicitly: the bulk-air / PML interface is the
    natural choice, a closed box sitting just inside the absorber.


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

    def _to_toml(self, tag: int) -> str:
        # No [[ports]] block; the tag is consumed by Problem into
        # [output] nfft_tag instead.
        return ""


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


    Note
    ----
    Apply this to the **outer face of the actual conductor geometry**, not to a
    zero-thickness sheet collapsed into the dielectric. Modelling a thick trace
    (e.g. a 3 µm microstrip line) as a single embedded plate over-estimates the
    conductor loss — the sharp edges of a zero-thickness strip carry a singular
    surface-current density, so :math:`\\int |J_s|^2` (hence the loss) and the
    extracted Z0 both run high (~1.6× the loss, ~10 % on Z0 in a microstrip
    cross-check). Extrude the conductor to its real thickness and put the
    surface impedance on its faces; the gap-facing face then carries the current
    one-sidedly, as the physical conductor does. The boundary condition itself
    is exact for a sheet of the given :math:`Z_s` — this is purely a geometry
    fidelity point.


    Example
    -------
    Copper surface on a stripline ground (an outer boundary face):

    .. code-block:: python

        rf.SurfaceImpedance(ground_face, conductivity=5.8e7)

    A finite-thickness trace, surface impedance on every outer face:

    .. code-block:: python

        trace = g.box(w, length, 3e-6, position=(...))
        rf.SurfaceImpedance(trace.faces, conductivity=3e7, thickness=3e-6)


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
    zs : tuple[float, float], optional
        explicit ``(Re, Im)`` surface impedance in :math:`\\Omega/\\square`,
        overrides the analytic value
    """

    def __init__(self, *targets,
                 conductivity: float = 0.0,
                 mur: float = 1.0,
                 er: float = 1.0,
                 thickness: float | None = None,
                 zs: tuple[float, float] | None = None):
        super().__init__(*targets)
        self.conductivity = float(conductivity)
        self.mur = float(mur)
        self.er = float(er)
        self.thickness = float(thickness) if thickness is not None else None
        self.zs = (float(zs[0]), float(zs[1])) if zs is not None else None

    def _to_toml(self, tag: int) -> str:
        s = (
            f'[[ports]]\ntype = "surface_impedance"\ntag = {tag}\n'
            f'conductivity = {_f64(self.conductivity)}\n'
            f'mur = {_f64(self.mur)}\ner = {_f64(self.er)}\n'
        )
        if self.thickness is not None:
            s += f'thickness = {_f64(self.thickness)}\n'
        if self.zs is not None:
            s += f'zs = [{_f64(self.zs[0])}, {_f64(self.zs[1])}]\n'
        return s


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

    def _to_toml(self, tag: int) -> str:
        s = (
            f'[[ports]]\ntype = "lumped_element"\ntag = {tag}\n'
            f'r = {_f64(self.r)}\nl = {_f64(self.l)}\n'
        )
        if self.c is not None:
            s += f'c = {_f64(self.c)}\n'
        d = self.direction
        s += (
            f'direction = [{_f64(d[0])}, {_f64(d[1])}, {_f64(d[2])}]\n'
            f'width = {_f64(self.width)}\nheight = {_f64(self.height)}\n'
        )
        return s


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
    _section = "pml"

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

    def _to_toml(self, tag: int) -> str:
        d = self.direction
        return (
            f'[[pml]]\nvolume_tag = {tag}\n'
            f'direction = [{_f64(d[0])}, {_f64(d[1])}, {_f64(d[2])}]\n'
            f'inner_face = {_f64(self.inner_face)}\n'
            f'thickness = {_f64(self.thickness)}\n'
            f'er_base = {_f64(self.er_base)}\n'
            f'ur_base = {_f64(self.ur_base)}\n'
            f'exponent = {_f64(self.exponent)}\n'
            f'delta_max = {_f64(self.delta_max)}\n'
        )


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
        # The base class _to_toml / tagging machinery assumes one tag per
        # _Physics, but we need two (one per face) for a periodic pair.
        # Store the two entity lists separately and overload the geometry
        # registration: a single PeriodicBoundary registers as two
        # physical-group tags, one per face.
        self._entities_a = ents_a
        self._entities_b = ents_b
        # _entities is kept (the union) so the parent _to_toml signature
        # and downstream tag walkers still see something sensible.
        self._entities = list(ents_a) + list(ents_b)
        self._geometry = geom_a
        geom_a._physics.append(self)


__all__ = [
    "RectWaveguidePort", "LumpedPort", "CoaxPort", "WavePort",
    "UserDefinedPort", "FloquetPort",
    "PEC", "PMC", "ABC", "SurfaceImpedance", "LumpedElement", "PML",
    "PeriodicBoundary",
]
