"""Time-domain DGTD problem — :class:`ProblemTD`.

`ProblemTD` is the time-domain counterpart of :class:`ProblemFD`. Where
`ProblemFD` is an analysis tool (geometry in, S-parameters out), `ProblemTD`
is a *model-export* tool: it compiles a cavity into a linear ODE
``dy/dt = A·y`` and exposes it at every level of abstraction —

* :meth:`transient`           — turnkey: propagate an initial state,
* :meth:`step`                — advance the state one exponential step,
* :meth:`rhs` / :meth:`jacobian` — the ODE right-hand side / constant Jacobian,
* :meth:`state_space`         — the verbatim sparse operator ``A``.

The current backend meshes a structured box cavity with PEC walls; general
geometry support follows the frequency-domain ``(mesh, TOML)`` path.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from .._native import TdOperator
from ..excitation import GaussianPulse

_FLUX = {"upwind": 1.0, "central": 0.0}
_FIELD = {"E": 0, "H": 1}
_COMP = {"x": 0, "y": 1, "z": 2}

# Speed of light (m/s). The DG operator runs in normalised units (c = 1, time
# measured in metres); `c` maps operator results to physical SI units —
# `t_op = c·t_seconds`, `f_Hz = c·ω_op/(2π)`.
C_LIGHT = 299_792_458.0

# S-parameter extraction tolerances. A rectangular-window DFT of the recorded
# port signals stays leakage-free only once the transient has decayed; warn
# when the tail still carries more than this fraction of the peak amplitude.
_SPARAM_DECAY_FRAC = 0.08
# |S| above 1 by more than this is non-physical for a passive structure and
# flags an under-resolved or too-short transient window.
_SPARAM_PASSIVITY_TOL = 0.02
# Fraction of the driven-port peak a port signal must exceed to count as the
# transmitted-pulse arrival; twice that arrival time is the round-trip travel
# time after which the reflection (diagonal) DFT window must close.
_SPARAM_ARRIVAL_FRAC = 0.05

# Matched-absorber loss budget. An rf.PML region wires through to the TD
# backend as a graded impedance-matched absorbing layer. The loss rate `nu`
# ramps quadratically (`nu_max·frac²`) by depth into the layer; round-trip
# attenuation through a quadratically graded slab of thickness `t` is roughly
# `exp(-2·nu_max·t/3)`. Setting `nu_max = _ABSORBER_LOSS_BUDGET / thickness`
# fixes `nu_max·t` regardless of slab depth, so the layer absorbs equally well
# at any thickness. `_ABSORBER_LOSS_BUDGET = 24` gives a round-trip reflection
# of `exp(-2·24/3) ≈ 1e-7` — far below 1 %. (`rf.PML.delta_max` is the
# frequency-domain coordinate-stretch magnitude, a different quantity, and is
# deliberately NOT used as the TD loss rate.)
_ABSORBER_LOSS_BUDGET = 24.0

# Explicit-integrator (LSERK4) CFL calibration. The exponential propagator
# is unconditionally stable; the explicit stepper is not, so `cfl_dt`
# brackets the scheme's stability limit empirically. The dimensionless
# product `z = h_solver·ρ(A)` is bisected between a known-stable and a
# known-unstable value, each candidate probed by a short explicit run from
# a delta excitation.
_CFL_POWER_ITERS = 40       # power-iteration count for the spectral radius
_CFL_PROBE_STEPS = 64       # explicit steps run per stability probe — long
                            # enough to see slow non-normal growth rates that
                            # the original 16-step probe missed.
_CFL_BISECT_ITERS = 7       # bisection steps bracketing the stability limit
_CFL_Z_STABLE = 3.0         # z = h_solver·ρ known stable for LSERK4 + DG Maxwell
_CFL_Z_UNSTABLE = 15.0      # z known to diverge
_CFL_GROWTH_FACTOR = 10.0   # absolute norm growth that fails a probe outright
_CFL_GROWTH_RATE_TOL = 1e-3 # max per-step geometric-mean amplification above
                            # 1.0 for a probe to count as stable: an upwind-DG
                            # operator is non-normal, so a mode bounded over a
                            # short probe can still drift to NaN over a long
                            # run; a strict per-step rate catches that drift.
_CFL_SAFETY = 0.8           # margin applied to the bracketed stability limit

# Adaptive embedded RK (Kennedy-Carpenter-Lewis RK4(3)5[2R+]C). The
# integrator delivers a per-step embedded-error vector; this controller
# weights it against `_RK_ATOL + _RK_RTOL·|y|`, builds a scalar `err_norm`,
# and grows or shrinks the next step with a PI rule (Söderlind /
# Gustafsson). The whole point of running this instead of the LSERK4 path
# is to drop the dependence on `cfl_dt`: a non-normal upwind-DG operator
# that slipped past a fixed CFL probe still shows up here, and the step
# gets cut before the trajectory diverges.
_RK_ATOL = 1e-8             # absolute error floor — the noise level a quiet
                            # DOF is allowed without contributing to err_norm.
_RK_RTOL = 1e-4             # relative tolerance — solution-fraction budget
                            # per DOF. 1e-4 keeps phase error well under one
                            # wavelength over a typical TD run while sitting
                            # near LSERK4's CFL on a near-uniform mesh.
_RK_SAFETY = 0.9            # safety factor in the step-size update —
                            # Hairer-Wanner standard.
_RK_GROWTH_LIMIT = 5.0      # max step-size growth between accepted steps.
_RK_SHRINK_LIMIT = 0.2      # min step-size factor per failure (lower bound).
# PI controller exponents (Söderlind / Gustafsson). The controller blends
# the current and previous error to smooth the step-size trajectory; the
# exponents are scaled by `1/p_hat` with `p_hat = min(p, p_emb)+1 = 4` for
# KCL 4(3)5.
_RK_PI_ALPHA = 0.7 / 4.0
_RK_PI_BETA = 0.4 / 4.0
# Bail-out for runaway shrinkage: the stepper aborts rather than spinning
# forever if the controller can't keep the step above this fraction of the
# requested output cadence.
_RK_MIN_STEP_FACTOR = 1e-10


def _log(msg):
    """Progress logging for long TD runs — to stderr, like the FD solver."""
    print(f"  [rapidfem-td] {msg}", file=sys.stderr, flush=True)


def _arr(y):
    """A contiguous 1-D float64 array — the zero-copy form the native
    operator reads directly from its buffer (no Python-list round-trip)."""
    return np.ascontiguousarray(y, dtype=np.float64).ravel()


def _volume_materials(geometry):
    """Yield ``(material, tag)`` for each unique volume :class:`Material`
    carrying a physical-group tag. The single walk + ``id``-dedup that
    :func:`_collect_materials` and :func:`_collect_dispersive` share.
    """
    from ..materials import Material

    seen = set()
    for ent in getattr(geometry, "_entities", []):
        mat = getattr(ent, "material", None)
        if not isinstance(mat, Material) or getattr(ent, "dim", None) != 3:
            continue
        if id(mat) in seen:
            continue
        seen.add(id(mat))
        tag = geometry._material_tags.get(id(mat))
        if tag is None:
            continue
        yield mat, int(tag)


def _collect_materials(geometry):
    """Walk the geometry's volume materials.

    Returns ``[(tag, eps_diag, mu_diag, sigma)]`` for the native TD operator.
    A material's *non-dispersive* permittivity is reported here; a Debye
    material's non-dispersive permittivity is its ``er_inf`` (the
    high-frequency limit), since the dispersion above ``er_inf`` is supplied
    by the ADE polarisation machinery (see :func:`_collect_dispersive`), not
    as a constant permittivity. A loss tangent is a frequency-domain effect
    and is not turned into a constant conductivity here.
    """
    out = []
    for mat, tag in _volume_materials(geometry):
        eps = mat.er_diag if mat.er_diag is not None else (mat.er,) * 3
        # A Debye material's non-dispersive permittivity is its er_inf; the
        # dispersive operator forces eps = er_inf on these tets anyway, but
        # reporting it here keeps the material assignment self-consistent.
        if getattr(mat, "debye", None) is not None:
            eps = (mat.debye.er_inf,) * 3
        mu = mat.ur_diag if mat.ur_diag is not None else (mat.ur,) * 3
        out.append((
            tag,
            tuple(float(v) for v in eps),
            tuple(float(v) for v in mu),
            float(mat.conductivity),
        ))
    return out


def _collect_dispersive(geometry):
    """Walk the geometry's volume materials for Debye dispersive components.

    Returns ``[(tag, er_inf, er_static, tau_s)]`` for the native TD
    operator's ``dispersive`` argument. Each volume :class:`Material`
    carrying a :class:`rapidfem.Debye` component runs the time-domain
    auxiliary-differential-equation (ADE) update — the operator appends a
    per-element polarisation field and integrates ``dP/dt = a*P + g*E``.

    The Drude model is intentionally not collected here: the time-domain
    backend's ``dispersive.rs`` carries only the first-order Debye ADE.
    Drude (a second-order auxiliary equation) is a future extension.
    """
    out = []
    for mat, tag in _volume_materials(geometry):
        debye = getattr(mat, "debye", None)
        if debye is None:
            continue
        out.append((
            tag,
            float(debye.er_inf),
            float(debye.er_static),
            float(debye.tau_s),
        ))
    return out


def _collect_ports(geometry):
    """Walk the geometry's port physics — :class:`RectWaveguidePort`,
    :class:`CoaxPort` and :class:`FloquetPort`.

    Returns ``(rect_ports, coax_ports, floquet_ports, wave_ports)``:

    - ``rect_ports`` is ``[(face_tag, mode_m, mode_n, direction, z0)]`` for
      the native TD operator's rectangular ``TE_mn`` ports
    - ``coax_ports`` is ``[(face_tag, center)]`` for its coaxial TEM ports
    - ``floquet_ports`` is ``[(face_tag, pol_mode, scan_theta, scan_phi)]``
      for its Floquet plane-wave ports — ``pol_mode`` is ``1`` (TE) or
      ``2`` (TM), matching the FD ``mode_nr`` convention; scan angles
      are radians.
    - ``wave_ports`` is ``[(face_tag, te, mode_index)]`` for numerically
      solved cross-section modes (:class:`WavePort`) — ``te`` selects
      TE vs TM, ``mode_index`` picks the mode by ascending cutoff.

    Operator port indices follow the geometry declaration order, with
    rectangular ports first, coax ports next, Floquet ports after, and
    wave ports last — matching the native ``TdOperator.from_mesh_bytes``
    layout.

    A waveguide port has ``direction`` ``None`` — its frame is auto-fit
    from the face. A coax port carries the analytic TEM ``E_ρ ∝ ρ̂/ρ``
    annular mode and forwards its optional axis ``origin`` to the native
    ``center`` override. A Floquet port carries a uniform plane wave with
    the TE / TM polarisation and the scan angles (``scan_theta_deg``,
    ``scan_phi_deg``) converted to radians; at oblique scan the
    transverse phase factor is dropped (documented approximation, see
    :class:`FloquetPort`).

    :class:`LumpedPort` is rejected — the time-domain backend has no
    lumped port (a uniform delta-gap profile only works on a genuine
    parallel-plate gap, not on concentrated quasi-TEM lines). Use a
    modal or wave port instead.
    """
    import math
    from ..physics import (
        CoaxPort, FloquetPort, LumpedPort, RectWaveguidePort, WavePort,
    )

    rect_out = []
    coax_out = []
    floquet_out = []
    wave_out = []
    for phys in getattr(geometry, "_physics", []):
        tag = geometry._physics_tags.get(id(phys))
        if tag is None:
            continue
        if isinstance(phys, RectWaveguidePort):
            mode = (int(phys.mode[0]), int(phys.mode[1]))
            rect_out.append((int(tag), mode[0], mode[1], None, 1.0))
        elif isinstance(phys, WavePort):
            # Numerically-solved cross-section mode:
            # (tag, te?, mode_index, k0). k0 = 2π f0 / c in 1/m (the mesh
            # is in metres); k0 <= 0 (f0 None) selects the scalar TE/TM
            # solve, k0 > 0 the inhomogeneous vector solve.
            k0 = (
                2.0 * math.pi * phys.f0 / 299_792_458.0
                if phys.f0 is not None
                else -1.0
            )
            wave_out.append(
                (int(tag), bool(phys.te), int(phys.mode_index), float(k0))
            )
        elif isinstance(phys, LumpedPort):
            # The time-domain backend has no lumped port. A lumped
            # (delta-gap, uniform-profile) source only carries a clean
            # mode on a genuine parallel-plate gap; on a concentrated
            # quasi-TEM line (microstrip, CPW, patch feed, spiral) the
            # uniform profile excites spurious evanescent modes and the
            # transmitted power is undercounted (see the abandoned
            # Thevenin experiments on feature/td-lumped-thevenin-v2).
            # The correct TD path for such lines is a wave port whose
            # mode profile is computed by a 2D cross-section eigensolve.
            raise NotImplementedError(
                "LumpedPort is not supported by the time-domain backend. "
                "Use a modal port (RectWaveguidePort, CoaxPort) for "
                "waveguide / TEM geometries, or a WavePort (2D "
                "cross-section eigensolve) for microstrip-class lines. "
                "The frequency-domain backend (ProblemFD) still supports "
                "LumpedPort via its Robin boundary condition."
            )
        elif isinstance(phys, CoaxPort):
            center = (
                None
                if phys.origin is None
                else (float(phys.origin[0]), float(phys.origin[1]), float(phys.origin[2]))
            )
            coax_out.append((int(tag), center))
        elif isinstance(phys, FloquetPort):
            # mode_nr 1 -> TE, mode_nr 2 -> TM, matching the FD backend's
            # FloquetPort.mode_nr field; default phys.mode_nr is 1.
            pol_mode = int(phys.mode_nr)
            theta = math.radians(float(phys.scan_theta_deg))
            phi = math.radians(float(phys.scan_phi_deg))
            floquet_out.append((int(tag), pol_mode, theta, phi))
    return rect_out, coax_out, floquet_out, wave_out


def _collect_pec(geometry):
    """Walk the geometry's :class:`PEC` physics objects and return the
    face tags of *internal* PEC plates (thin sheets inside the
    domain, e.g. a microstrip trace).

    Domain-boundary PEC faces are handled automatically by the TD
    operator (a boundary face without any port assignment is PEC by
    default); they are silently filtered out here. Only the internal
    plates - where the face has neighbour tets on both sides - need
    the explicit `pec_faces` wiring that retags both sides to behave
    as PEC walls.

    The returned tags include every face tag listed under `rf.PEC`;
    the Rust side accepts the union and only retags faces whose
    triangle list actually sits between two tets, so passing
    boundary tags through is a no-op.
    """
    from ..physics import PEC

    tags = []
    for phys in getattr(geometry, "_physics", []):
        if not isinstance(phys, PEC):
            continue
        tag = geometry._physics_tags.get(id(phys))
        if tag is None:
            continue
        if isinstance(tag, tuple):
            for t in tag:
                tags.append(int(t))
        else:
            tags.append(int(tag))
    return tags


def _collect_abc(geometry):
    """Walk the geometry's :class:`ABC` physics objects.

    Returns ``[face_tag]`` for the native TD operator's ``abc_faces``
    argument. Each face is registered with the operator as a
    pure-absorbing boundary (a :class:`PortSpec` with ``mode = None``),
    which the DG flux treats as a Silver-Mueller first-order ABC:
    near-normally-incident outgoing waves leave with negligible
    reflection, oblique incidence reflects more.

    The TD characteristic absorber is a first-order condition, the same
    order as the FD ``rf.ABC``. The user keeps a single ``rf.ABC(*faces)``
    declaration that both backends respect; only its boundary
    treatment differs.
    """
    from ..physics import ABC

    tags = []
    for phys in getattr(geometry, "_physics", []):
        if not isinstance(phys, ABC):
            continue
        tag = geometry._physics_tags.get(id(phys))
        if tag is None:
            continue
        if isinstance(tag, tuple):
            for t in tag:
                tags.append(int(t))
        else:
            tags.append(int(tag))
    return tags


def _collect_periodic(geometry):
    """Walk the geometry's :class:`PeriodicBoundary` physics objects.

    Returns ``[(face_tag_a, face_tag_b)]`` for the native TD operator's
    ``periodic_pairs`` argument. Each :class:`PeriodicBoundary` registers
    as two physical-group tags in the gmsh export (one per side), stored
    by the geometry as a ``(tag_a, tag_b)`` tuple under ``_physics_tags``;
    this walker pulls each tuple out in declaration order, so the native
    operator's periodic matcher sees the pairs in the order they were
    declared in Python.
    """
    from ..physics import PeriodicBoundary

    out = []
    for phys in getattr(geometry, "_physics", []):
        if not isinstance(phys, PeriodicBoundary):
            continue
        pair = geometry._physics_tags.get(id(phys))
        if pair is None or not isinstance(pair, tuple):
            continue
        out.append((int(pair[0]), int(pair[1])))
    return out


def _collect_absorbers(geometry):
    """Walk the geometry's :class:`PML` physics regions.

    Each :class:`rapidfem.PML` terminates the domain with a volumetric
    absorbing slab. In time domain it wires through to the native operator
    as a graded impedance-matched absorbing layer — there is no separate
    coordinate-stretch PML in the TD backend; the matched absorber is the
    TD equivalent.

    Returns ``[(volume_tag, axis, inner_face, thickness, nu_max, is_low)]``
    for the native TD operator's ``absorbers`` argument. ``axis`` is the
    index (0/1/2) of the dominant component of the PML's outward
    ``direction``; ``is_low`` is true when that component points toward
    decreasing coordinate (the layer extends to the low-coordinate end).
    ``nu_max`` is derived from the slab thickness so the round-trip
    reflection stays well below 1 % — see :data:`_ABSORBER_LOSS_BUDGET`.
    """
    from ..physics import PML

    out = []
    for phys in getattr(geometry, "_physics", []):
        if not isinstance(phys, PML):
            continue
        tag = geometry._physics_tags.get(id(phys))
        if tag is None:
            continue
        d = phys.direction
        axis = int(np.argmax(np.abs(np.asarray(d, dtype=float))))
        is_low = bool(d[axis] < 0.0)
        thickness = float(phys.thickness)
        nu_max = _ABSORBER_LOSS_BUDGET / thickness if thickness > 0.0 else 0.0
        out.append((
            int(tag), axis, float(phys.inner_face), thickness,
            float(nu_max), is_low,
        ))
    return out


class TdODE:
    """The time-domain problem as an explicit linear ODE ``dy/dt = A·y``.

    A handoff object for external integrators (e.g.
    :func:`scipy.integrate.solve_ivp`): :meth:`rhs` carries the
    integrator's ``(t, y)`` signature and is evaluated matrix-free;
    :meth:`jacobian` returns the constant sparse ``A`` for implicit
    methods. Obtained from :meth:`ProblemTD.ode`.
    """

    def __init__(self, problem):
        self._p = problem
        self.n_dof = problem.n_dof

    def rhs(self, t, y):
        """``dy/dt`` at state ``y``. The ``t`` argument is ignored — the
        system is autonomous and linear — but kept for the integrator
        signature. Matrix-free, runs on all cores."""
        return self._p._op.apply(_arr(y))

    def jacobian(self, t=None, y=None):
        """The constant Jacobian ``A`` as a :class:`scipy.sparse.csr_matrix`."""
        return self._p.state_space()

    def __repr__(self):
        return f"TdODE(n_dof={self.n_dof})"


class TdStepper:
    """A reusable one-step propagator bound to a fixed ``dt``.

    Call the stepper on a state to advance it by ``dt``. With
    ``method="exponential"`` the step is exact for the linear homogeneous
    system at any ``dt``; with ``method="explicit"`` it is the cheaper
    LSERK4 stepper, substepped to respect its CFL limit. Obtained from
    :meth:`ProblemTD.stepper`.
    """

    def __init__(self, problem, dt, krylov_dim,
                 method="exponential", cfl_dt=None):
        self._p = problem
        self.dt = float(dt)
        self.krylov_dim = int(krylov_dim)
        self.method = method
        self._cfl = cfl_dt

    def __call__(self, y):
        return self._p._advance(y, self.dt, self.method,
                                self.krylov_dim, self._cfl)

    def advance(self, y):
        """Advance ``y`` by one ``dt`` step — same as calling the stepper."""
        return self(y)

    def __repr__(self):
        return f"TdStepper(dt={self.dt:g}, method={self.method!r})"


def _point_label(spec):
    """Human-readable label for a ``(point, field, component)`` probe/source
    spec — e.g. ``"E_z @ (0.25, 0.25, 0.5)"``."""
    p, f, c = spec
    coords = ", ".join(f"{v:g}" for v in np.asarray(p, dtype=float).ravel())
    return f"{f}_{c} @ ({coords})"


class TdScattering:
    """Modal-port scattering matrix from :meth:`ProblemTD.sparams`.

    Iterates as ``(frequencies, sparams)``, so the documented tuple
    unpacking — ``freqs, S = ptd.sparams(...)`` — keeps working unchanged;
    the named attributes additionally let :func:`rapidfem.show` plot the
    result in the UI.

    Attributes
    ----------
    frequencies : ndarray
        frequency axis, shape ``[n_freq]``
    sparams : ndarray of complex
        scattering matrix, shape ``[n_freq, n_port, n_port]``
    """

    def __init__(self, frequencies, sparams):
        self.frequencies = np.asarray(frequencies)
        self.sparams = np.asarray(sparams)

    @property
    def n_ports(self):
        """Number of ports — the side length of the S-matrix."""
        return self.sparams.shape[1] if self.sparams.ndim == 3 else 0

    def __iter__(self):
        return iter((self.frequencies, self.sparams))

    def __repr__(self):
        return (f"TdScattering(n_ports={self.n_ports}, "
                f"n_freq={self.frequencies.size})")


class TdResponse:
    """Probe time series from :meth:`ProblemTD.driven_transient`.

    Iterates as ``(times, responses)`` so the documented tuple unpacking
    keeps working; the stored source / probe labels let
    :func:`rapidfem.show` annotate the time-series plot.

    Attributes
    ----------
    times : ndarray
        time axis, shape ``[steps + 1]``
    responses : ndarray
        per-probe samples, shape ``[n_probes, steps + 1]``
    source_label, probe_labels : str, list of str
        human-readable point/field/component labels
    """

    def __init__(self, times, responses, *, source_label="", probe_labels=None):
        self.times = np.asarray(times)
        self.responses = np.asarray(responses)
        self.source_label = source_label
        self.probe_labels = list(probe_labels or [])

    def __iter__(self):
        return iter((self.times, self.responses))

    def __repr__(self):
        return (f"TdResponse(n_probes={self.responses.shape[0]}, "
                f"steps={self.times.size - 1})")


class TdTransfer:
    """Scalar field-to-field frequency response from
    :meth:`ProblemTD.transfer_function`.

    Iterates as ``(frequencies, H)`` so the documented tuple unpacking
    keeps working; the labels let :func:`rapidfem.show` annotate the plot.

    Attributes
    ----------
    frequencies : ndarray
        frequency axis, shape ``[steps // 2 + 1]``
    H : ndarray of complex
        the transfer function ``R(f) / G(f)``
    source_label, probe_label : str
        human-readable point/field/component labels
    """

    def __init__(self, frequencies, H, *, source_label="", probe_label=""):
        self.frequencies = np.asarray(frequencies)
        self.H = np.asarray(H)
        self.source_label = source_label
        self.probe_label = probe_label

    def __iter__(self):
        return iter((self.frequencies, self.H))

    def __repr__(self):
        return f"TdTransfer(n_freq={self.frequencies.size})"


class TdTrajectory(np.ndarray):
    """A time-domain field trajectory — ``[n_snapshot, n_dof]``.

    For every numerical purpose this *is* a :class:`numpy.ndarray` —
    indexing, slicing, ``.shape``, arithmetic and
    :meth:`ProblemTD.export_vtk` all behave exactly as before. It
    additionally carries a back-reference to the originating
    :class:`ProblemTD` and the time step, so :func:`rapidfem.show` can
    sample the DG state onto renderable geometry for the 3-D field
    animation in the UI.
    """

    def __new__(cls, data, *, problem=None, dt=None):
        obj = np.ascontiguousarray(data, dtype=np.float64).view(cls)
        obj._problem = problem
        obj._dt = dt
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._problem = getattr(obj, "_problem", None)
        self._dt = getattr(obj, "_dt", None)


def _fmt(a):
    """Whitespace-joined ascii of a numeric array — VTK DataArray payload."""
    return " ".join(f"{v:.9g}" for v in np.asarray(a).ravel())


def _write_vtu(path, points, connectivity, offsets, cell_types, point_data):
    """Write one VTK XML UnstructuredGrid (``.vtu``), ascii."""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" '
        'byte_order="LittleEndian">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{len(points)}" '
        f'NumberOfCells="{len(offsets)}">',
        '      <Points>',
        '        <DataArray type="Float64" NumberOfComponents="3" '
        'format="ascii">',
        f'          {_fmt(points)}',
        '        </DataArray>',
        '      </Points>',
        '      <Cells>',
        '        <DataArray type="Int64" Name="connectivity" format="ascii">',
        f'          {_fmt(connectivity)}',
        '        </DataArray>',
        '        <DataArray type="Int64" Name="offsets" format="ascii">',
        f'          {_fmt(offsets)}',
        '        </DataArray>',
        '        <DataArray type="UInt8" Name="types" format="ascii">',
        f'          {_fmt(cell_types)}',
        '        </DataArray>',
        '      </Cells>',
        '      <PointData>',
    ]
    for name, data in point_data.items():
        lines.append(
            f'        <DataArray type="Float64" Name="{name}" '
            'NumberOfComponents="3" format="ascii">'
        )
        lines.append(f'          {_fmt(data)}')
        lines.append('        </DataArray>')
    lines += [
        '      </PointData>',
        '    </Piece>',
        '  </UnstructuredGrid>',
        '</VTKFile>',
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_pvd(path, entries):
    """Write a ParaView ``.pvd`` collection over ``(time, vtu-name)`` pairs."""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" '
        'byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t, fname in entries:
        lines.append(f'    <DataSet timestep="{t:.9g}" file="{fname}"/>')
    lines += ['  </Collection>', '</VTKFile>']
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class StateSpaceROM:
    """A reduced-order linear state-space model ``(A, B, C, D)`` exported
    from a :class:`TdMacroModel` for import into an external simulator.

    The model is the LTI system

    ::

        dx/dt = A x + B u
            z  = C x + D u

    with ``x`` the ``r``-dimensional reduced state, ``u`` the ``N``
    per-port incident-wave inputs and ``z`` the modal readouts. Because
    the macromodel carries two modal readouts (transverse ``E`` and
    ``H``), the output is split into ``C_e`` and ``C_h``; the
    incident / scattered split per port ``i`` is
    ``a_i = ½(z_E,i + Z_i z_H,i)``, ``b_i = ½(z_E,i − Z_i z_H,i)`` with
    ``Z`` the port reference impedance (:attr:`port_impedances`). ``D`` is
    zero — the impulse-Krylov model is strictly proper.

    All matrices are real. Units follow ``physical``: if ``True`` (the
    default of :meth:`TdMacroModel.state_space`), ``A``/``B`` are scaled
    to physical SI time (``A`` eigenvalues are physical angular
    resonances in rad/s); if ``False`` they are in the operator's
    normalised units (``c = 1``).

    Attributes
    ----------
    A : ndarray, shape (r, r)
    B : ndarray, shape (r, N)
    C_e, C_h : ndarray, shape (N, r)
    D : ndarray, shape (N, N)
        always zero, included so consumers get a complete (A, B, C, D).
    c : float
        speed of light used for the operator→physical unit mapping.
    physical : bool
        whether A/B are in physical SI time or normalised operator units.
    """

    def __init__(self, A, B, C_e, C_h, *, c, physical):
        self.A = np.ascontiguousarray(A, dtype=float)
        self.B = np.ascontiguousarray(B, dtype=float)
        self.C_e = np.ascontiguousarray(C_e, dtype=float)
        self.C_h = np.ascontiguousarray(C_h, dtype=float)
        self.D = np.zeros((C_e.shape[0], B.shape[1]), dtype=float)
        self.c = float(c)
        self.physical = bool(physical)

    @property
    def r(self):
        """Reduced state dimension."""
        return self.A.shape[0]

    @property
    def n_ports(self):
        """Number of ports ``N``."""
        return self.B.shape[1]

    def as_dict(self):
        """The realisation as a plain dict of numpy arrays plus scalar
        metadata — the form fed to :func:`scipy.io.savemat` or
        :func:`numpy.savez`."""
        return {
            "A": self.A,
            "B": self.B,
            "C_e": self.C_e,
            "C_h": self.C_h,
            "D": self.D,
            "c": np.array([[self.c]]),
            "physical": np.array([[1 if self.physical else 0]]),
        }

    def to_npz(self, path):
        """Write the realisation to a NumPy ``.npz`` archive (load with
        :func:`numpy.load`)."""
        np.savez(path, **{k: np.asarray(v) for k, v in self.as_dict().items()})
        return path

    def to_mat(self, path):
        """Write the realisation to a MATLAB ``.mat`` file via
        :func:`scipy.io.savemat` — loads directly into MATLAB / Simulink
        (``ss(A, B, C, D)``) or :func:`scipy.io.loadmat`."""
        from scipy.io import savemat

        savemat(path, self.as_dict())
        return path

    def __repr__(self):
        units = "physical-SI" if self.physical else "operator-normalised"
        return (
            f"StateSpaceROM(r={self.r}, n_ports={self.n_ports}, "
            f"units={units})"
        )


class TdMacroModel:
    """A compact MIMO macromodel of a :class:`ProblemTD`'s modal-port
    network: a block-Krylov (or multi-shift rational-Krylov) projection
    of the port-seeded Krylov subspace. Obtained from
    :meth:`ProblemTD.macromodel`.

    Evaluating the S-matrix at one frequency costs a small dense complex
    LU solve (microseconds at ``r ~ a few hundred``); a thousand-point
    sweep is milliseconds. :meth:`state_space` exports the raw
    ``(A, B, C, D)`` realisation for an external simulator;
    :meth:`to_touchstone` writes the S-matrix.

    The TD operator runs in normalised units (``c = 1``); this wrapper
    accepts and reports frequencies in physical Hertz
    (``omega_op = 2*pi*f_Hz / c``).
    """

    def __init__(self, native, c):
        self._m = native
        self._c = float(c)

    @property
    def r(self):
        """Realised reduced order (block-Krylov dimension after any
        deflation)."""
        return self._m.r

    @property
    def n_ports(self):
        """Number of modal ports — size of the S-matrix."""
        return self._m.n_ports

    def state_space(self, *, physical=True):
        """Export the reduced realisation as a :class:`StateSpaceROM`.

        Parameters
        ----------
        physical : bool
            If ``True`` (default) the ``A``/``B`` matrices are scaled to
            physical SI time, so ``A``'s eigenvalues are physical angular
            resonances (rad/s) and the model integrates against seconds.
            If ``False`` they are left in the operator's normalised units
            (``c = 1``). ``C`` is unit-independent either way.

        Returns
        -------
        StateSpaceROM
            Carries ``A, B, C_e, C_h, D`` and ``.to_mat`` / ``.to_npz``
            writers.
        """
        A, B, C_e, C_h = self._m.state_space()
        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)
        if physical:
            # omega_op = omega_phys / c, so (j*omega_op I - A) =
            # (1/c)(j*omega_phys I - c*A): the physical-time operator is
            # c*A with the input map scaled by c, C unchanged, D = 0.
            A = self._c * A
            B = self._c * B
        return StateSpaceROM(
            A, B, C_e, C_h, c=self._c, physical=physical
        )

    def port_impedances(self, frequency_hz):
        """Per-port modal reference impedances ``Z_i`` at a physical
        frequency, length ``N``. Frequency-flat for TEM / Floquet ports,
        dispersive for ``TE_mn``. Feeds the incident / scattered split on
        :class:`StateSpaceROM`."""
        omega_op = 2.0 * np.pi * float(frequency_hz) / self._c
        return np.asarray(self._m.port_impedances(omega_op))

    def evaluate(self, frequency_hz, *, passive=False):
        """Evaluate the S-matrix at one physical frequency. Returns an
        ``[n_ports, n_ports]`` complex numpy array.

        ``passive=True`` applies the passivity-enforcement perturbation:
        the SVD of ``S`` is clipped to ``sigma_max(S) <= 1``."""
        omega_op = 2.0 * np.pi * float(frequency_hz) / self._c
        if passive:
            return np.asarray(self._m.evaluate_passive(omega_op))
        return np.asarray(self._m.evaluate(omega_op))

    def sweep(self, frequencies_hz):
        """Evaluate the S-matrix on a sweep of physical frequencies.
        Returns an ``[n_freq, n_ports, n_ports]`` complex numpy array."""
        f = np.asarray(frequencies_hz, dtype=float).ravel()
        omegas_op = 2.0 * np.pi * f / self._c
        return np.asarray(self._m.sweep(omegas_op))

    def to_touchstone(self, path, frequencies_hz, *, format="MA", z_ref=50.0):
        """Write the S-matrix sweep to a Touchstone ``.s{N}p`` file.

        Parameters
        ----------
        path : str
            Output path (Touchstone 1.x: one file per N-port model).
        frequencies_hz : array_like
            Frequencies to sample, in Hertz.
        format : {"MA", "RI", "DB"}
            Per-entry format.
        z_ref : float
            Reference impedance (header ``R`` field).
        """
        f = np.asarray(frequencies_hz, dtype=float).ravel()
        f_ghz = f / 1.0e9
        omegas_op = 2.0 * np.pi * f / self._c
        self._m.to_touchstone(
            str(path), f_ghz, omegas_op, "GHZ", float(z_ref), str(format)
        )
        return path

    def __repr__(self):
        return f"TdMacroModel(r={self.r}, n_ports={self.n_ports})"


class ProblemTD:
    """Time-domain DGTD Maxwell problem ready for analysis.

    A container around a meshed :class:`~rapidfem.Geometry` and its
    attached materials, ports and BCs — the time-domain counterpart of
    :class:`~rapidfem.ProblemFD`. The curl equations are discretised in
    space with a **nodal discontinuous Galerkin** method on tetrahedra,
    giving an explicit linear ODE ``dy/dt = A·y`` with a constant, sparse
    operator ``A``; a driven port adds a rank-1 source ``b(t)``.

    ``ProblemTD`` is a **model-export tool** — it hands back that ODE at
    every level of abstraction, so the verb to call is just the level of
    detail wanted:

    - :meth:`rhs` / :meth:`state_space` — the matrix-free right-hand side,
      or the verbatim sparse operator ``A``
    - :meth:`ode` — a handoff object for an external integrator
      (e.g. ``scipy.integrate.solve_ivp``)
    - :meth:`step` / :meth:`stepper` / :meth:`transient` — exact
      exponential time stepping (matrix-free Krylov / ETD)
    - :meth:`driven_transient` / :meth:`transfer_function` / :meth:`sparams`
      — soft-source or modal-port excitation, a scalar transfer function,
      and the modal-port scattering matrix
    - :meth:`resonances` — cavity eigenfrequencies from the spectrum
    - :meth:`export_vtk` — a VTK field animation

    Because the semi-discrete system is linear with a constant ``A``, the
    exponential propagator is *exact* at any step size — the time step is
    set by the wanted output cadence, not by a CFL stability limit.

    Note
    ----
    The geometry must already be meshed (via ``g.mesh()``) before the
    ProblemTD is constructed — construction snapshots the mesh bytes.
    Re-meshing the geometry afterwards has no effect on an existing
    ProblemTD; construct a new one instead. :meth:`box` is a shortcut that
    builds directly on a structured box cavity, bypassing the geometry
    API — handy for validation.

    Example
    -------
    Build a waveguide problem, then read the model at three levels:

    .. code-block:: python

        g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
        air = g.box(22.86e-3, 10.16e-3, 30e-3, material=rf.Air())
        rf.RectWaveguidePort(air.faces.min(axis="z"))
        rf.RectWaveguidePort(air.faces.max(axis="z"))
        rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"),
               air.faces.min(axis="y"), air.faces.max(axis="y"))
        g.mesh()

        ptd = rf.ProblemTD(g, order=2, flux="upwind")
        A = ptd.state_space()                        # verbatim sparse A
        advance = ptd.stepper(dt=5e-12)              # exact exponential step
        freqs, S = ptd.sparams(np.linspace(8e9, 12e9, 21),
                               dt=3e-12, steps=820)  # modal-port S-matrix

    Attributes
    ----------
    n_dof : int
        state-vector length, ``6·Np·n_elem``
    order : int
        the DG polynomial order the operator was built at
    flux : str
        the numerical flux in use — ``"upwind"`` or ``"central"``
    c : float
        speed of light in the mesh's length units; sets the operator ↔
        physical time/frequency mapping
    """

    def __init__(self, geometry, *, order=2, flux="upwind", c=C_LIGHT):
        """
        Parameters
        ----------
        geometry : rapidfem.Geometry
            A geometry on which ``g.mesh()`` has already been called.
        order : int
            DG polynomial order.
        flux : {"upwind", "central"}
            Numerical flux. ``central`` is exactly energy-conserving;
            ``upwind`` additionally damps the discontinuous spurious modes.
        c : float
            Speed of light in the mesh's length units (default SI, metres);
            sets the operator↔physical time/frequency mapping.
        """
        if flux not in _FLUX:
            raise ValueError(f"flux must be one of {sorted(_FLUX)}")
        if getattr(geometry, "_last_mesh", None) is None:
            raise RuntimeError(
                "geometry not meshed yet — call g.mesh() before "
                "constructing a ProblemTD"
            )
        self.c = float(c)
        mesh_bytes = geometry._last_mesh[0]
        tag_materials = _collect_materials(geometry)
        tag_ports, coax_ports, floquet_ports, wave_ports = \
            _collect_ports(geometry)
        tag_absorbers = _collect_absorbers(geometry)
        tag_periodic = _collect_periodic(geometry)
        tag_abc = _collect_abc(geometry)
        tag_pec = _collect_pec(geometry)
        # The TD operator runs in normalised units (c = 1, time measured in
        # the mesh's length units), so a Debye relaxation time given in
        # seconds is scaled to operator units: tau_op = c * tau_s. eps_inf /
        # eps_static are dimensionless and pass through unchanged.
        tag_dispersive = [
            (tag, er_inf, er_static, self.c * tau_s)
            for (tag, er_inf, er_static, tau_s) in _collect_dispersive(geometry)
        ]
        self._op = TdOperator.from_mesh_bytes(
            bytes(mesh_bytes), order, _FLUX[flux],
            tag_materials or None, tag_ports or None,
            tag_absorbers or None, tag_dispersive or None,
            coax_ports or None,
            tag_periodic or None,
            floquet_ports or None,
            tag_abc or None,
            tag_pec or None,
            wave_ports or None,
        )
        self._geometry = geometry
        self.order = order
        self.flux = flux
        _log(
            f"operator built - {self.n_dof} DOFs, order {order}, "
            f"flux={flux}, {len(tag_materials)} tagged materials, "
            f"{len(tag_ports) + len(coax_ports) + len(floquet_ports) + len(wave_ports)} ports "
            f"({len(tag_ports)} rect, {len(coax_ports)} coax, "
            f"{len(floquet_ports)} floquet, {len(wave_ports)} wave), "
            f"{len(tag_absorbers)} matched-absorber (PML) regions, "
            f"{len(tag_dispersive)} Debye dispersive regions, "
            f"{len(tag_periodic)} periodic boundary pair(s), "
            f"{len(tag_abc)} ABC face(s), "
            f"{len(tag_pec)} PEC face tag(s) (internal plates only "
            f"are retagged; boundary PEC is the default)"
        )

    @classmethod
    def box(cls, *, size, cells, order=2, flux="upwind", c=1.0):
        """Build directly on a structured box cavity, bypassing the geometry
        API — handy for validation and quick experiments.

        Parameters
        ----------
        size : (lx, ly, lz)
            Cavity dimensions.
        cells : (nx, ny, nz)
            Structured-mesh cell counts per axis.
        c : float
            Speed of light in the box's length units (default 1, normalised).
        """
        if flux not in _FLUX:
            raise ValueError(f"flux must be one of {sorted(_FLUX)}")
        lx, ly, lz = size
        nx, ny, nz = cells
        obj = cls.__new__(cls)
        obj._op = TdOperator(nx, ny, nz, lx, ly, lz, order, _FLUX[flux])
        obj._geometry = None
        obj.order = order
        obj.flux = flux
        obj.c = float(c)
        obj.size = tuple(size)
        obj.cells = tuple(cells)
        _log(
            f"operator built (box) - {obj.n_dof} DOFs, order {order}, "
            f"flux={flux}"
        )
        return obj

    @property
    def n_dof(self):
        """State-vector length, ``6·Np·n_elem``."""
        return self._op.n_dof()

    @property
    def n_dofs(self):
        """Alias of :attr:`n_dof`, matching ProblemFD's attribute name."""
        return self.n_dof

    # -- low level: the ODE -------------------------------------------------
    def rhs(self, y):
        """The ODE right-hand side ``dy/dt = A·y``."""
        return self._op.apply(_arr(y))

    def field_energy(self, state):
        """Instantaneous electromagnetic field energy of a state.

        Returns ``(1/2) * integral(eps*|E|^2 + mu*|H|^2) dV`` in the
        operator's units -- the material-weighted EM field energy carried
        by ``state``. This is a physically exact diagnostic: the DG
        energy-mass matrix is block-diagonal per element, so the energy is
        a cheap per-element quadratic-form sum, evaluated matrix-free with
        no n-by-n matrix ever materialised.

        Unlike a raw state norm ``numpy.dot(y, y)``, this weights each
        component by the local permittivity / permeability, so it is the
        quantity the central flux conserves exactly and the upwind flux
        leaves non-increasing.

        Parameters
        ----------
        state : array_like
            A state vector ``[n_dof]``. Trailing auxiliary DOFs beyond the
            ``6*Np*n_elem`` E,H block are ignored.

        Returns
        -------
        float
            The field energy, finite and non-negative for any real state.
        """
        return self._op.field_energy(_arr(state))

    def jacobian(self):
        """The (constant) Jacobian of the linear system — i.e. ``A`` itself,
        as a sparse matrix. See :meth:`state_space`."""
        return self.state_space()

    def state_space(self):
        """The verbatim operator ``A`` as a :class:`scipy.sparse.csr_matrix`."""
        from scipy.sparse import csr_matrix

        n, row_ptr, col_idx, values = self._op.state_space()
        return csr_matrix((values, col_idx, row_ptr), shape=(n, n))

    def ode(self):
        """Export the problem as an explicit linear ODE ``dy/dt = A·y``.

        Returns a :class:`TdODE` carrying everything an external
        integrator needs — ``n_dof``, a matrix-free ``rhs(t, y)`` with
        the :func:`scipy.integrate.solve_ivp` signature, and
        ``jacobian()``.
        """
        return TdODE(self)

    def resonances(self, *, n=8):
        """Cavity resonant frequencies (Hz) from the operator's spectrum.

        The DG Maxwell operator's eigenvalues are `±iω`; with the upwind flux
        the physical modes are the least-damped ones — `f = c·|ω|/(2π)`.
        Dense eigenvalue solve, so for modest meshes only.
        """
        a = self.state_space().toarray()
        ev = np.linalg.eigvals(a)
        omega = np.abs(ev.imag)
        phys = omega > 1e-3 * omega.max()  # drop the near-static modes
        ev_p = ev[phys]
        out = []
        for idx in np.argsort(-ev_p.real):  # least-damped first
            f = abs(ev_p[idx].imag) * self.c / (2.0 * np.pi)
            if any(abs(f - g) <= 1e-3 * g for g in out):
                continue
            out.append(f)
            if len(out) >= n:
                break
        return np.array(sorted(out))

    # -- mid level: stepping ------------------------------------------------
    def step(self, y, h, krylov_dim=40, tol=None):
        """Advance the state by ``h`` (in the same time units as ``c``) with
        the matrix-free exponential propagator — exact for the linear
        homogeneous system at any ``h``.

        ``tol`` is the Krylov a-posteriori error tolerance; ``None`` keeps
        the solver default. A converged step uses far fewer than
        ``krylov_dim`` matvecs; ``tol=0`` forces the full ``krylov_dim``
        (the fixed-dimension worst case).

        See also :meth:`step_explicit` (the cheaper CFL-bound LSERK4) and
        :meth:`step_adaptive` (embedded KCL 4(3)5 with self-tuned step),
        plus :meth:`transient` for the high-level loop."""
        h_solver = float(self.c * h)
        if tol is None:
            return self._op.step(_arr(y), h_solver, int(krylov_dim))
        return self._op.step(_arr(y), h_solver, int(krylov_dim), float(tol))

    def step_explicit(self, y, h):
        """Advance the state by ``h`` with the explicit LSERK4 integrator
        (five matvecs, no Krylov subspace).

        Cheaper per step than :meth:`step`, but only conditionally stable:
        an ``h`` past the operator's CFL limit diverges. Prefer :meth:`step`
        (the exponential propagator) when the mesh is stiff or the step is
        set by the output cadence rather than by stability. See
        :meth:`step_adaptive` for an embedded variant that drives its own
        step size, and :meth:`transient` (``method="explicit"`` or
        ``"adaptive"``) for the higher-level loop that uses these."""
        return self._op.step_explicit(_arr(y), float(self.c * h))

    def step_adaptive(self, y, h):
        """Advance the state by ``h`` with the embedded Kennedy-Carpenter-
        Lewis RK4(3)5[2R+]C low-storage Runge-Kutta stepper.

        Same five-matvec cost as :meth:`step_explicit`, but the scheme
        carries a third-order embedded solution alongside its fourth-order
        main; the per-DOF difference between the two is returned as
        ``err`` so an adaptive controller can decide whether to accept
        the step and how to size the next one. The :meth:`transient`
        loop (with ``method="adaptive"``) is the high-level entry that
        drives this stepper with a PI controller (Söderlind / Gustafsson)
        normalising ``err`` against ``atol + rtol·|y|``; you only call
        the raw stepper here if you are writing your own controller.

        Parameters
        ----------
        y : array_like
            Current state, shape ``[n_dof]``.
        h : float
            Proposed step size in physical time units.

        Returns
        -------
        y_new : ndarray
            The advanced state, shape ``[n_dof]`` — the fourth-order main
            solution.
        err : ndarray
            Per-DOF embedded-error vector ``y_4 − y_3``, shape
            ``[n_dof]``. A controller compares its weighted L2 norm
            against 1: smaller means the step was easy and the next one
            can grow, above 1 means the step must be rejected and shrunk.
            See the ``_RK_*`` module constants for the calibration this
            module uses.

        Notes
        -----
        Conditionally stable like :meth:`step_explicit` — an ``h`` past
        the operator's CFL limit eventually NaNs. The controller in
        :meth:`transient` rejects any infinite ``err`` and shrinks ``h``,
        so the high-level loop survives a bad initial-guess step; the raw
        stepper here does not."""
        h_solver = float(self.c * h)
        y_new, err = self._op.step_kcl(_arr(y), h_solver)
        return y_new, err

    def cfl_dt(self, *, recompute=False):
        """Largest stable time step for the explicit LSERK4 integrator, in
        physical time units.

        The exponential propagator (:meth:`step`) is unconditionally
        stable; the explicit stepper (:meth:`step_explicit`) is not, and a
        step past this limit diverges. The limit is found by power-iterating
        the operator's spectral radius and bracketing the LSERK4 stability
        boundary empirically; the result is cached.

        A :meth:`transient` or :meth:`stepper` run with ``method="explicit"``
        calls this itself and substeps to stay within the limit, so the
        limit rarely needs to be read directly."""
        cached = getattr(self, "_cfl_dt_cache", None)
        if cached is not None and not recompute:
            return cached

        n = self.n_dof
        # Spectral radius by power iteration: ||A.v|| / ||v|| approaches
        # rho(A) as v aligns with the largest-magnitude eigenvector.
        rng = np.random.default_rng(0)
        v = rng.standard_normal(n)
        v /= np.linalg.norm(v)
        rho = 1.0
        for _ in range(_CFL_POWER_ITERS):
            av = self.rhs(v)
            rho = float(np.linalg.norm(av))
            v = av / rho

        # Probe at z = h_solver·ρ. Excite the *dominant* eigenvector (the
        # power-iteration leftover `v`, which is the most-unstable mode), so
        # the probe directly tests the binding mode rather than whichever
        # modes a delta happens to project onto. A stable z must keep the
        # geometric-mean per-step amplification at most slightly above 1 —
        # an upwind-DG operator is non-normal, so a probe that just stays
        # "bounded after a few steps" can hide a slow drift that NaNs the
        # real run over thousands of substeps.
        probe = v

        def stable(z):
            h = z / (self.c * rho)
            n0 = float(np.linalg.norm(probe))
            y = probe
            for _ in range(_CFL_PROBE_STEPS):
                y = self.step_explicit(y, h)
            if not np.all(np.isfinite(y)):
                return False
            nf = float(np.linalg.norm(y))
            if nf > _CFL_GROWTH_FACTOR * n0:
                return False
            rate = (nf / n0) ** (1.0 / _CFL_PROBE_STEPS) if nf > 0.0 else 0.0
            return rate <= 1.0 + _CFL_GROWTH_RATE_TOL

        lo, hi = _CFL_Z_STABLE, _CFL_Z_UNSTABLE
        while not stable(lo) and lo > 0.1:
            lo *= 0.5                       # operator stiffer than expected
        while stable(hi) and hi < 1e3:
            hi *= 1.5                       # dissipation extends the range
        for _ in range(_CFL_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            if stable(mid):
                lo = mid
            else:
                hi = mid

        self._cfl_dt_cache = _CFL_SAFETY * lo / (self.c * rho)
        return self._cfl_dt_cache

    def _advance(self, y, dt, method, krylov_dim, cfl_dt):
        """Advance ``y`` by one output step ``dt`` with the chosen
        integrator. The explicit integrator substeps so each substep stays
        within the CFL limit ``cfl_dt``."""
        if method == "exponential":
            return self.step(y, dt, krylov_dim)
        nsub = max(1, int(np.ceil(abs(dt) / cfl_dt)))
        h = dt / nsub
        for _ in range(nsub):
            y = self.step_explicit(y, h)
        return y

    def _advance_driven(self, y, t_n, dt, method, sdof, waveform,
                        krylov_dim, cfl_dt):
        """Advance ``y`` by one output step ``dt`` of the driven system
        with the chosen integrator. The explicit integrator substeps
        within the CFL limit and re-samples the waveform each substep; the
        exponential integrator holds the source constant across the step,
        as :meth:`step_driven` does."""
        if method == "exponential":
            g = float(waveform(t_n))
            return self._op.step_driven(_arr(y), sdof, g,
                                        float(self.c * dt), krylov_dim)
        nsub = max(1, int(np.ceil(abs(dt) / cfl_dt)))
        h = dt / nsub
        h_op = float(self.c * h)
        for j in range(nsub):
            g = float(waveform(t_n + j * h))
            y = self._op.step_driven_explicit(_arr(y), h_op, sdof, g)
        return y

    def _kcl_err_norm(self, y_old, y_new, err):
        """Weighted L2 error norm `sqrt(mean((err / (atol+rtol·max|y|))²))`
        the controller compares against 1.0 — the standard Hairer-Wanner
        mixed-tolerance criterion (each DOF scaled by its own magnitude).
        """
        scale = _RK_ATOL + _RK_RTOL * np.maximum(np.abs(y_old), np.abs(y_new))
        return float(np.sqrt(np.mean((err / scale) ** 2)))

    def _kcl_factor(self, err_norm, prev_err_norm, *, reject):
        """Step-size multiplier from the PI controller. On a rejected step
        the previous-error term is dropped (I-only) so a bad step's history
        is not propagated; on an accepted step the PI blend smooths the
        step trajectory across many frames."""
        if reject or prev_err_norm <= 0.0:
            f = _RK_SAFETY * err_norm ** (-_RK_PI_ALPHA)
        else:
            f = (
                _RK_SAFETY
                * err_norm ** (-_RK_PI_ALPHA)
                * prev_err_norm ** _RK_PI_BETA
            )
        if reject:
            f = max(f, _RK_SHRINK_LIMIT)
        else:
            f = max(min(f, _RK_GROWTH_LIMIT), _RK_SHRINK_LIMIT)
        return f

    def _advance_adaptive(self, y, dt, *, h, prev_err_norm, t_offset=0.0,
                          source=None, sdof=None, waveform=None):
        """Advance ``y`` by one output step ``dt`` with the KCL adaptive
        stepper. The internal PI controller takes as many sub-steps as it
        needs to land on ``t_offset + dt`` within tolerance; ``h`` carries
        the controller's current step size across frames so a long quiet
        phase doesn't re-acquire it from scratch.

        Three driving modes via the kwargs: ``source=None, sdof=None`` —
        free system (`dy/dt = A·y`); ``sdof, waveform`` — single-DOF soft
        source with zeroth-order hold; ``source, waveform`` — full source
        vector ``b`` driven by ``g(t) = waveform``, the modal-port path.

        Returns ``(y, h_next, prev_err_norm, n_acc, n_rej)``.
        """
        n_acc = 0
        n_rej = 0
        t_rel = 0.0
        h_min = _RK_MIN_STEP_FACTOR * dt
        # The KCL step is CFL-bounded like LSERK4: an h that lands the
        # solution at NaN cannot be rescued by the controller's err_norm
        # (Inf rejects, but the next attempt will retry from the same start
        # with a shrunk h). The accept/reject loop converges on a stable h.
        while t_rel < dt:
            h_try = min(h, dt - t_rel)
            t_now = t_offset + t_rel
            if source is not None:
                g = float(waveform(t_now))
                y_try, err = self._op.step_with_source_kcl(
                    _arr(y), source * g, float(self.c * h_try),
                )
            elif sdof is not None:
                g = float(waveform(t_now))
                y_try, err = self._op.step_driven_kcl(
                    _arr(y), float(self.c * h_try), int(sdof), g,
                )
            else:
                y_try, err = self._op.step_kcl(
                    _arr(y), float(self.c * h_try),
                )
            # A NaN err_norm rejects too — non-finite means the step blew
            # the CFL limit, the controller must shrink and retry.
            if not np.all(np.isfinite(y_try)):
                err_norm = np.inf
            else:
                err_norm = self._kcl_err_norm(y, y_try, err)
            if err_norm <= 1.0 and np.isfinite(err_norm):
                y = y_try
                t_rel += h_try
                n_acc += 1
                h = h_try * self._kcl_factor(
                    err_norm, prev_err_norm, reject=False,
                )
                prev_err_norm = max(err_norm, 1e-12)
            else:
                n_rej += 1
                h = h_try * self._kcl_factor(
                    err_norm if np.isfinite(err_norm) else 10.0,
                    prev_err_norm, reject=True,
                )
            if h < h_min:
                raise RuntimeError(
                    f"adaptive stepper: step size collapsed below "
                    f"{_RK_MIN_STEP_FACTOR:g}·dt after {n_acc} accepted, "
                    f"{n_rej} rejected substeps. The operator is likely "
                    f"too stiff for the chosen tolerances "
                    f"(atol={_RK_ATOL:g}, rtol={_RK_RTOL:g})."
                )
        return y, h, prev_err_norm, n_acc, n_rej

    def stepper(self, dt, *, krylov_dim=40, method="exponential"):
        """A reusable one-step propagator bound to a fixed ``dt``.

        Returns a :class:`TdStepper` — call it repeatedly to advance a
        state without re-passing ``dt``/``krylov_dim`` each time.

        ``method`` selects the integrator: ``"exponential"`` (exact at any
        ``dt``) or ``"explicit"`` (the cheaper LSERK4 stepper, substepped
        to respect its CFL limit)."""
        if method not in ("exponential", "explicit"):
            raise ValueError("method must be 'exponential' or 'explicit'")
        cfl = self.cfl_dt() if method == "explicit" else None
        return TdStepper(self, dt, krylov_dim, method, cfl)

    # -- ports: soft sources & field probes --------------------------------
    def probe_dof(self, point, *, field="E", component="z"):
        """Global DOF index for a field component at the node nearest
        ``point`` — used to place soft sources and field probes."""
        return self._op.nearest_node_dof(
            tuple(float(x) for x in point), _FIELD[field], _COMP[component]
        )

    def driven_transient(
        self, *, source, waveform, probes, dt, steps, krylov_dim=40,
        device="cpu", verbose=True,
    ):
        """Drive a soft point source and record field probes.

        Parameters
        ----------
        source : (point, field, component)
            Where and which field component to inject.
        waveform : callable
            ``g(t)`` — the excitation, e.g. a :class:`~rapidfem.GaussianPulse`.
        probes : list of (point, field, component)
            Field samples to record over the run.
        dt, steps : float, int
            Time step and step count.

        Returns
        -------
        TdResponse
            Iterates as ``(times, responses)`` — ``times`` of shape
            ``[steps+1]`` and ``responses`` of shape
            ``[n_probes, steps+1]`` — so ``times, resp = ...`` unpacking
            works unchanged; passing it to :func:`rapidfem.show` plots the
            probe signals.
        """
        sp, sf, sc = source
        sdof = self.probe_dof(sp, field=sf, component=sc)
        pdofs = [
            self.probe_dof(p, field=f, component=c) for (p, f, c) in probes
        ]
        n = self.n_dof

        # GPU path: the explicit LSERK4 driven transient device-resident,
        # probes extracted from the trajectory. Falls back to the CPU
        # exponential path when no GPU is present.
        if device == "gpu" and self._op.gpu_available():
            cfl = self.cfl_dt()
            nsub = max(1, int(np.ceil(abs(dt) / cfl)))
            h_sub = dt / nsub
            src = np.array(
                [float(waveform(i * h_sub)) for i in range(steps * nsub)],
                dtype=np.float64,
            )
            if verbose:
                _log(f"driven_transient: GPU LSERK4 "
                     f"({self._op.gpu_device()}, {nsub} substeps/step)")
            flat = self._op.gpu_transient_driven(
                np.zeros(n), float(self.c * dt), int(steps), nsub,
                int(sdof), src,
            )
            traj = np.asarray(flat).reshape(steps + 1, n)
            resp = np.array([traj[:, d] for d in pdofs])
            return TdResponse(
                np.arange(steps + 1) * dt, resp,
                source_label=_point_label(source),
                probe_labels=[_point_label(p) for p in probes],
            )

        y = np.zeros(n)
        times = np.arange(steps + 1) * dt
        resp = np.zeros((len(pdofs), steps + 1))
        for k, d in enumerate(pdofs):
            resp[k, 0] = y[d]
        t0 = time.time()
        every = max(1, steps // 10)
        for s in range(steps):
            g = float(waveform(s * dt))
            y = self._op.step_driven(
                _arr(y), sdof, g, float(self.c * dt), krylov_dim
            )
            for k, d in enumerate(pdofs):
                resp[k, s + 1] = y[d]
            if verbose and (s + 1) % every == 0:
                el = time.time() - t0
                eta = el / (s + 1) * (steps - s - 1)
                _log(
                    f"driven_transient {s + 1}/{steps}  "
                    f"({el:.1f}s elapsed, ETA {eta:.0f}s)"
                )
        if verbose:
            _log(
                f"driven_transient complete - {steps} steps "
                f"in {time.time() - t0:.1f}s"
            )
        return TdResponse(
            times, resp,
            source_label=_point_label(source),
            probe_labels=[_point_label(p) for p in probes],
        )

    def transfer_function(
        self, *, source, probe, pulse, dt, steps, krylov_dim=40,
        device="cpu", verbose=True,
    ):
        """Field-to-field frequency response by on-the-fly RFT.

        Drives a broadband ``pulse`` at ``source``, records ``probe``,
        then divides the probe spectrum by the source spectrum —
        ``H(f) = R(f) / G(f)`` — to recover the linear cavity's transfer
        function in one transient run. Peaks of ``|H(f)|`` mark the
        resonances.

        This is the scalar, on-the-fly-RFT observable. It is a transfer
        function between two *field points*, not a normalised port wave:
        true modal-port S-parameters need waveguide-mode injection /
        extraction.

        Parameters
        ----------
        source : (point, field, component)
            Soft-source location and field component to inject.
        probe : (point, field, component)
            Field sample to record.
        pulse : callable
            Broadband excitation ``g(t)`` — its spectrum sets the usable
            frequency band (a :class:`~rapidfem.GaussianPulse` is the
            typical choice).
        dt, steps : float, int
            Time step and step count; together they fix the frequency
            resolution ``1/(steps·dt)`` and the Nyquist limit
            ``1/(2·dt)``.

        Returns
        -------
        TdTransfer
            Iterates as ``(freqs, H)`` — ``freqs`` the frequency axis (Hz
            for an SI geometry, operator units for a :meth:`box`, length
            ``steps//2 + 1``) and ``H`` the complex transfer function
            ``R(f)/G(f)``, zero outside the pulse band. ``freqs, H = ...``
            unpacking works unchanged; :func:`rapidfem.show` plots it.
        """
        times, resp = self.driven_transient(
            source=source, waveform=pulse, probes=[probe],
            dt=dt, steps=steps, krylov_dim=krylov_dim, device=device,
            verbose=verbose,
        )
        g = np.asarray(pulse(times), dtype=float)
        spec_g = np.fft.rfft(g)
        spec_r = np.fft.rfft(resp[0])
        freqs = np.fft.rfftfreq(times.size, dt)
        # H = R/G only where the drive carries real energy. Outside the
        # pulse band G→0, and dividing by it amplifies pure numerical
        # noise — the classic deconvolution artefact — so H is held at
        # zero below 1 % of the peak source spectrum.
        h = np.zeros_like(spec_r)
        band = np.abs(spec_g) > 1e-2 * np.abs(spec_g).max()
        h[band] = spec_r[band] / spec_g[band]
        return TdTransfer(
            freqs, h,
            source_label=_point_label(source),
            probe_label=_point_label(probe),
        )

    # -- ports: scattering parameters --------------------------------------
    def _port_impedance(self, port_idx, f):
        """TE-mode wave impedance of a port at physical frequency ``f``."""
        wc = self._op.port_cutoff(port_idx)
        fc = self.c * wc / (2.0 * np.pi)
        r = fc / f
        return 1.0 / np.sqrt(1.0 - r * r)

    def sparams(self, freqs, *, dt, steps, pulse=None, krylov_dim=40,
                verbose=True):
        """Scattering matrix ``S(f)`` of the waveguide-port network.

        Drives each port in turn with a broadband pulse, extracts the
        modal amplitudes at every port by surface-integral projection,
        and assembles ``S_ij(f) = B_i(f)/A_j(f)`` — the wave leaving port
        ``i`` per unit wave incident at the driven port ``j``.

        Parameters
        ----------
        freqs : array_like
            Frequencies — Hz for a geometry, operator units for a
            :meth:`box`.
        dt, steps : float, int
            Time step and step count. The run must be long enough to
            capture the response; for a clean ``S₁₁`` it should stop
            before energy multiply-reflected by the (characteristic)
            ports returns.
        pulse : callable, optional
            Broadband excitation ``g(t)``; defaults to a modulated
            Gaussian spanning the frequency band.
        krylov_dim : int
            Krylov dimension of the exponential step.

        Returns
        -------
        TdScattering
            Iterates as ``(freqs, S)`` — ``freqs`` the frequency axis and
            ``S`` the complex scattering matrix of shape
            ``[n_freq, n_port, n_port]``, with ``S[f, i, j]`` in the same
            index order as the frequency-domain backend's
            ``SweepResult.sparams``. ``freqs, S = ...`` unpacking works
            unchanged; :func:`rapidfem.show` plots the S-parameters.
        """
        freqs = np.asarray(freqs, dtype=float).ravel()
        total_ports = self._op.n_ports()
        # Modal ports only — ABC faces (port_has_mode == False) are
        # absorbing-only and do not participate in S-parameter
        # extraction. The S-matrix is indexed over the modal subset
        # in declaration order.
        modal_idx = [
            p for p in range(total_ports) if self._op.port_has_mode(p)
        ]
        n_ports = len(modal_idx)
        if n_ports == 0:
            raise RuntimeError(
                "ProblemTD has no modal ports — attach "
                "RectWaveguidePort(s) or CoaxPort(s) to the "
                "geometry before constructing it; ABC faces alone do "
                "not carry a mode for extraction"
            )
        # Below a waveguide port's cutoff the modal wave impedance turns
        # imaginary and the scattering parameters are undefined; reject the
        # run up front rather than let NaN poison the whole S-matrix.
        f_min = float(freqs.min())
        for p in modal_idx:
            f_cut = self.c * self._op.port_cutoff(p) / (2.0 * np.pi)
            if f_min <= f_cut:
                raise ValueError(
                    f"frequency {f_min:.4g} is at or below the cutoff "
                    f"{f_cut:.4g} of port {p}; restrict freqs to the "
                    f"propagating band"
                )
        n = self.n_dof
        times = np.arange(steps) * dt

        if pulse is None:
            fc = float(np.mean(freqs))
            fw = float(np.ptp(freqs)) or 0.5 * fc
            tau = 1.0 / (np.pi * max(fw, 0.25 * fc))
            pulse = GaussianPulse(t0=4.0 * tau, tau=tau, f0=fc)
        g = np.asarray(pulse(times), dtype=float)
        # DFT kernel — rows index frequency, columns index time sample.
        phase = np.exp(-2j * np.pi * np.outer(freqs, times)) * dt

        h_op = float(self.c * dt)
        a_inc = np.zeros((n_ports, freqs.size), dtype=complex)
        b_out = np.zeros((n_ports, n_ports, freqs.size), dtype=complex)
        every = max(1, steps // 5)
        for jx, j_port in enumerate(modal_idx):
            src = self._op.port_source(j_port)
            y = np.zeros(n)
            pe = np.zeros((n_ports, steps))
            ph = np.zeros((n_ports, steps))
            t0 = time.time()
            for s in range(steps):
                y = self._op.step_with_source(
                    y, src * g[s], h_op, krylov_dim
                )
                for ix, i_port in enumerate(modal_idx):
                    pe[ix, s], ph[ix, s] = self._op.port_projections(
                        y, i_port
                    )
                if verbose and (s + 1) % every == 0:
                    _log(
                        f"sparams drive {jx + 1}/{n_ports}: "
                        f"step {s + 1}/{steps} ({time.time() - t0:.1f}s)"
                    )
            # Reflection (i == j) and transmission (i != j) terms need
            # opposite DFT windows. Transmission must capture the whole
            # slow, dispersive transmitted pulse, so it uses the full run.
            # Reflection must be read off before the imperfectly absorbed
            # port re-reflection makes its round trip back to the driven
            # port, so its window closes at twice the first-arrival time
            # at the nearest other port -- the round-trip travel time.
            peak = float(np.abs(pe).max())
            arrivals = []
            for ix in range(n_ports):
                if ix == jx or peak <= 0.0:
                    continue
                env = np.abs(pe[ix])
                above = np.flatnonzero(env > _SPARAM_ARRIVAL_FRAC * peak)
                if above.size:
                    arrivals.append(int(above[0]))
            refl_w = (
                int(np.clip(2 * min(arrivals), steps // 4, steps))
                if arrivals else steps
            )
            phase_r = (
                phase
                if refl_w >= steps
                else np.exp(-2j * np.pi * np.outer(freqs, times[:refl_w]))
                * dt
            )
            # Forward / backward modal split A,B = (P_e +- Z*P_h)/2 of the
            # recorded total field. Impedance is looked up on the
            # operator's port index (i_port), not the modal-list index.
            for ix, i_port in enumerate(modal_idx):
                z = np.array(
                    [self._port_impedance(i_port, f) for f in freqs]
                )
                if ix == jx:
                    pe_f = phase_r @ pe[ix, :refl_w]
                    ph_f = phase_r @ ph[ix, :refl_w]
                    a_inc[jx] = 0.5 * (pe_f + z * ph_f)
                    b_out[jx, ix] = 0.5 * (pe_f - z * ph_f)
                else:
                    pe_f = phase @ pe[ix]
                    ph_f = phase @ ph[ix]
                    b_out[jx, ix] = 0.5 * (pe_f - z * ph_f)
            # Closing the loop: the transmission DFT stays leakage-free
            # only once the transient has decayed by the window end.
            tail = max(1, steps // 20)
            resid = float(np.abs(pe[:, -tail:]).max())
            if verbose and peak > 0.0 and resid > _SPARAM_DECAY_FRAC * peak:
                _log(
                    f"sparams drive {jx + 1}: port signal still at "
                    f"{resid / peak:.0%} of peak at the window end, "
                    f"raise steps for a cleaner extraction"
                )

        s_mat = np.zeros((freqs.size, n_ports, n_ports), dtype=complex)
        for j in range(n_ports):
            for i in range(n_ports):
                s_mat[:, i, j] = b_out[j, i] / a_inc[j]
        smax = float(np.abs(s_mat).max())
        if verbose:
            _log(
                f"sparams complete - {n_ports}-port S-matrix at "
                f"{freqs.size} frequencies"
            )
            if smax > 1.0 + _SPARAM_PASSIVITY_TOL:
                _log(
                    f"sparams: |S| peaks at {smax:.3f} above unity, the "
                    f"transient window is too short or the mesh too coarse "
                    f"for a passive result"
                )
        return TdScattering(freqs, s_mat)

    # -- turnkey: a transient run ------------------------------------------
    def _modal_ports(self):
        """The geometry's modal port physics objects in the operator's
        declaration order: rect, coax, floquet, wave. Matches
        :func:`_collect_ports` and the native operator port layout, so the
        k-th entry here is the k-th port for which
        ``port_has_mode`` is true."""
        from ..physics import (
            CoaxPort, FloquetPort, RectWaveguidePort, WavePort,
        )
        if self._geometry is None:
            return []
        geom = self._geometry
        phys = [
            p for p in getattr(geom, "_physics", [])
            if geom._physics_tags.get(id(p)) is not None
        ]
        rect = [p for p in phys if isinstance(p, RectWaveguidePort)]
        coax = [p for p in phys if isinstance(p, CoaxPort)]
        floq = [p for p in phys if isinstance(p, FloquetPort)]
        wave = [p for p in phys if isinstance(p, WavePort)]
        return rect + coax + floq + wave

    def _port_operator_index(self, port):
        """Operator port index of a modal port physics object — the index
        :meth:`port_source` / :meth:`port_projections` take. Resolves the
        port's position among the modal ports (declaration order) and maps
        it onto the operator's modal subset (``port_has_mode``), so
        absorbing-only ABC faces in between are skipped."""
        modal = self._modal_ports()
        k = next((i for i, p in enumerate(modal) if p is port), None)
        if k is None:
            raise ValueError(
                "port= is not a modal port of this problem's geometry; pass "
                "a RectWaveguidePort / CoaxPort / FloquetPort / WavePort "
                "instance attached to the meshed geometry"
            )
        modal_idx = [
            p for p in range(self._op.n_ports())
            if self._op.port_has_mode(p)
        ]
        if k >= len(modal_idx):
            raise RuntimeError(
                "modal-port count mismatch between geometry and operator"
            )
        return modal_idx[k]

    def port_signals(self, traj, ports, *, dt=None, labels=None):
        """Modal wave amplitude ``P_e(t)`` at each port over a trajectory,
        as a :class:`TdResponse` for :func:`rapidfem.show` — a time-domain
        line plot of the modal port signals next to the field animation.

        Parameters
        ----------
        traj : ndarray
            A ``[n_snapshot, n_dof]`` field trajectory, e.g. the return of
            :meth:`transient`.
        ports : list
            Modal port physics objects (RectWaveguidePort / CoaxPort /
            WavePort) whose modal amplitude to read out.
        dt : float, optional
            Snapshot spacing for the time axis. Defaults to the
            trajectory's own ``dt`` when it is a :class:`TdTrajectory`.
        labels : list of str, optional
            Curve labels; default ``port 0, port 1, ...``.
        """
        if dt is None:
            dt = getattr(traj, "dt", None) or 1.0
        traj = np.asarray(traj)
        n_snap = traj.shape[0]
        idxs = [self._port_operator_index(p) for p in ports]
        rows = np.empty((len(idxs), n_snap))
        for s in range(n_snap):
            for k, idx in enumerate(idxs):
                rows[k, s] = self._op.port_projections(traj[s], idx)[0]
        labs = list(labels) if labels else [f"port {k}" for k in range(len(idxs))]
        return TdResponse(np.arange(n_snap) * dt, rows, probe_labels=labs)

    def _driven_vector_traj(self, b, waveform, *, y0, dt, steps, method,
                            device, krylov_dim, verbose):
        """Field trajectory of ``dy/dt = A·y + b·g(t)`` — the full-vector
        (modal-port) source path, routed across ``method`` ∈
        {exponential, explicit} × ``device`` ∈ {cpu, gpu}. The exponential
        step is exact at any ``dt`` (one step per snapshot); the explicit
        step is substepped within the CFL limit. Returns the
        ``[steps+1, n_dof]`` trajectory."""
        n = self.n_dof
        b = np.ascontiguousarray(b, dtype=np.float64)
        y = np.zeros(n) if y0 is None else _arr(y0)
        h_op = float(self.c * dt)

        # GPU paths, the state device-resident where the stepper allows.
        if device == "gpu" and self._op.gpu_available():
            if method == "adaptive":
                # Adaptive: PI controller on the device, only the scalar
                # err_norm per substep and the per-frame state snapshot
                # cross the bus. One waveform sample per output frame
                # (zeroth-order hold), like the LSERK4 GPU driven path.
                g_vals = np.array(
                    [float(waveform(k * dt)) for k in range(steps)],
                    dtype=np.float64,
                )
                if verbose:
                    _log(f"transient(port): GPU vector KCL adaptive "
                         f"({self._op.gpu_device()}, "
                         f"atol={_RK_ATOL:g}, rtol={_RK_RTOL:g})")
                t0 = time.time()
                flat, n_acc, n_rej, h_min, h_max = \
                    self._op.gpu_transient_kcl_driven_vec(
                        _arr(y), h_op, int(steps), b, g_vals,
                        _RK_ATOL, _RK_RTOL, _RK_SAFETY,
                        _RK_GROWTH_LIMIT, _RK_SHRINK_LIMIT,
                        _RK_PI_ALPHA, _RK_PI_BETA, _RK_MIN_STEP_FACTOR,
                    )
                traj = np.asarray(flat).reshape(steps + 1, n)
                if verbose:
                    _log(f"transient(port) complete - {steps} steps "
                         f"in {time.time() - t0:.1f}s")
                    _log(
                        f"  KCL controller: {n_acc} accepted, "
                        f"{n_rej} rejected; h ∈ "
                        f"[{h_min / self.c:.3g}, {h_max / self.c:.3g}] s"
                    )
                return traj
            if method == "explicit":
                cfl = self.cfl_dt()
                nsub = max(1, int(np.ceil(abs(dt) / cfl)))
                h_sub = dt / nsub
                gvals = np.array(
                    [float(waveform(i * h_sub))
                     for i in range(steps * nsub)],
                    dtype=np.float64,
                )
                if verbose:
                    _log(f"transient(port): GPU vector LSERK4 "
                         f"({self._op.gpu_device()}, {nsub} substeps/step)")
                # Step in chunks so progress reports incrementally. Each
                # chunk runs device-resident; only the chunk-boundary state
                # round-trips, so the overhead over one monolithic call is
                # one state up/download per chunk (negligible vs the solve).
                traj = np.empty((steps + 1, n))
                traj[0] = y
                chunk = max(1, steps // 10)
                done = 0
                t0 = time.time()
                while done < steps:
                    k = min(chunk, steps - done)
                    g_slice = gvals[done * nsub:(done + k) * nsub]
                    flat = self._op.gpu_transient_driven_vec(
                        traj[done], h_op, int(k), nsub, b, g_slice,
                    )
                    traj[done + 1:done + 1 + k] = \
                        np.asarray(flat).reshape(k + 1, n)[1:]
                    done += k
                    if verbose:
                        el = time.time() - t0
                        eta = el / done * (steps - done)
                        _log(f"transient(port) {done}/{steps}  "
                             f"({el:.1f}s elapsed, ETA {eta:.0f}s)")
                if verbose:
                    _log(f"transient(port) complete - {steps} steps "
                         f"in {time.time() - t0:.1f}s")
                return traj
            # Exponential on GPU: exact, one augmented-Arnoldi ETD step per
            # snapshot (the source is held across the step).
            if verbose:
                _log(f"transient(port): GPU vector ETD "
                     f"({self._op.gpu_device()})")
            traj = np.empty((steps + 1, n))
            traj[0] = y
            t0 = time.time()
            every = max(1, steps // 10)
            for k in range(steps):
                y = self._op.gpu_step_with_source(
                    y, b * float(waveform(k * dt)), h_op, krylov_dim,
                )
                traj[k + 1] = y
                if verbose and (k + 1) % every == 0:
                    el = time.time() - t0
                    eta = el / (k + 1) * (steps - k - 1)
                    _log(f"transient(port) {k + 1}/{steps}  "
                         f"({el:.1f}s elapsed, ETA {eta:.0f}s)")
            if verbose:
                _log(f"transient(port) complete - {steps} steps "
                     f"in {time.time() - t0:.1f}s")
            return traj
        if device == "gpu":
            _log("transient(port): no OpenCL GPU available, using CPU")

        # CPU paths.
        traj = np.empty((steps + 1, n))
        traj[0] = y
        cfl = self.cfl_dt() if method == "explicit" else None
        nsub = max(1, int(np.ceil(abs(dt) / cfl))) if cfl else 1
        if verbose and method == "explicit":
            _log(f"transient(port): CPU vector LSERK4 "
                 f"({nsub} substeps/step)")
        if verbose and method == "adaptive":
            _log(f"transient(port): CPU vector KCL adaptive "
                 f"(atol={_RK_ATOL:g}, rtol={_RK_RTOL:g})")
        # Adaptive controller state — carried across frames as in the
        # source-less path. The vector-source variant of the KCL stepper
        # receives `b·g(t_now)` as its full source, zeroth-order hold.
        h_ad = dt
        prev_err = 0.0
        total_acc, total_rej = 0, 0
        h_min_log, h_max_log = float("inf"), 0.0
        t0 = time.time()
        every = max(1, steps // 10)
        for k in range(steps):
            if method == "exponential":
                y = self._op.step_with_source(
                    _arr(y), b * float(waveform(k * dt)), h_op, krylov_dim,
                )
            elif method == "adaptive":
                y, h_ad, prev_err, n_acc, n_rej = self._advance_adaptive(
                    y, dt, h=h_ad, prev_err_norm=prev_err,
                    t_offset=k * dt, source=b, waveform=waveform,
                )
                total_acc += n_acc
                total_rej += n_rej
                h_min_log = min(h_min_log, h_ad)
                h_max_log = max(h_max_log, h_ad)
            else:
                h_sub_op = h_op / nsub
                h_sub = dt / nsub
                for j in range(nsub):
                    y = self._op.step_with_source_explicit(
                        _arr(y),
                        b * float(waveform(k * dt + j * h_sub)),
                        h_sub_op,
                    )
            traj[k + 1] = y
            if verbose and (k + 1) % every == 0:
                el = time.time() - t0
                eta = el / (k + 1) * (steps - k - 1)
                _log(f"transient(port) {k + 1}/{steps}  "
                     f"({el:.1f}s elapsed, ETA {eta:.0f}s)")
        if verbose:
            _log(f"transient(port) complete - {steps} steps "
                 f"in {time.time() - t0:.1f}s")
            if method == "adaptive":
                _log(
                    f"  KCL controller: {total_acc} accepted, "
                    f"{total_rej} rejected; h ∈ "
                    f"[{h_min_log:.3g}, {h_max_log:.3g}] s"
                )
        return traj

    def transient(self, y0=None, *, dt, steps, source=None, waveform=None,
                  port=None, krylov_dim=40, method="exponential", warmup=0,
                  device="cpu", verbose=True):
        """Propagate the field for ``steps`` steps of size ``dt``.

        With ``y0`` only this is the free (homogeneous) evolution of an
        initial state. Passing ``source`` (a ``(point, field, component)``
        spec) together with ``waveform`` (a callable ``g(t)``) instead
        drives a soft point source every step -- a driven transient whose
        full field history is returned, so :func:`rapidfem.show` animates
        the driven problem (e.g. a pulse radiating into a PML-terminated
        domain).

        Parameters
        ----------
        y0 : array_like, optional
            Initial state; defaults to zero (the rest state for a driven
            run).
        dt, steps : float, int
            Time step and step count.
        source : (point, field, component), optional
            Soft-source location and field component. Driving needs both
            ``source`` and ``waveform``.
        port : RectWaveguidePort | CoaxPort | FloquetPort | WavePort, optional
            A modal port attached to the geometry, driven by its spatial
            mode pattern ``b`` (``dy/dt = A·y + b·g(t)``) instead of a point
            source. Mutually exclusive with ``source``; needs ``waveform``.
            Routed across ``method`` × ``device`` like the rest, so the
            exponential/explicit and CPU/GPU paths all inject the same mode.
        waveform : callable, optional
            Excitation ``g(t)``, e.g. a :class:`~rapidfem.GaussianPulse`.
        method : {"exponential", "explicit", "adaptive"}
            Time integrator. ``"exponential"`` is exact at any ``dt``;
            ``"explicit"`` is the cheaper LSERK4 stepper, substepped to
            respect its CFL limit (see :meth:`cfl_dt`); ``"adaptive"`` is
            the Kennedy-Carpenter-Lewis RK4(3)5[2R+]C embedded stepper
            with PI step-size control, which drops the :meth:`cfl_dt`
            dependence and shrinks its step on its own when the operator
            is stiff. All three drive a soft source; explicit re-samples
            the waveform each substep, adaptive samples it once per
            output frame (zeroth-order hold).

            *Adaptive vs the frequency-domain* :class:`~rapidfem.Adaptive`
            *class*: unrelated — :class:`~rapidfem.Adaptive` drives mesh /
            order h-p refinement in :meth:`ProblemFD.sweep`; here the
            argument selects a *time integrator*.
        warmup : int
            Output steps to run with the exponential integrator before
            handing off to ``method``. With ``method="explicit"`` this
            covers the opening transient with the exact integrator, then
            continues on the cheaper explicit stepper. Not supported with
            ``method="adaptive"`` (the PI controller stabilises itself in
            the first few frames).
        device : {"cpu", "gpu"}
            ``"gpu"`` runs the chosen integrator on an OpenCL GPU, the
            state device-resident. Adaptive runs ~20-50× faster on a
            stiff mesh on GPU than the explicit LSERK4 path, since it
            does not pay the ``cfl_dt`` substep penalty. Falls back to
            the CPU path when no GPU is present.

        Returns
        -------
        TdTrajectory
            The field trajectory, shape ``[steps + 1, n_dof]``. It *is* a
            :class:`numpy.ndarray` for every numerical purpose (indexing,
            slicing, :meth:`export_vtk`); passing it to
            :func:`rapidfem.show` plays it back as a 3-D field animation
            in the UI.
        """
        if method not in ("exponential", "explicit", "adaptive"):
            raise ValueError(
                "method must be 'exponential', 'explicit', or 'adaptive'"
            )
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")

        # Modal-port injection: drive dy/dt = A·y + b·g(t) with b the port's
        # spatial mode pattern. Returns the field trajectory for animation.
        if port is not None:
            if source is not None:
                raise ValueError(
                    "pass either source= (point) or port= (modal port), "
                    "not both"
                )
            if waveform is None:
                raise ValueError(
                    "driving a port needs a waveform= (a callable g(t))"
                )
            if warmup:
                raise ValueError(
                    "warmup is not supported with port= injection"
                )
            b = self._op.port_source(self._port_operator_index(port))
            traj = self._driven_vector_traj(
                b, waveform, y0=y0, dt=dt, steps=steps, method=method,
                device=device, krylov_dim=krylov_dim, verbose=verbose,
            )
            return TdTrajectory(traj, problem=self, dt=dt)

        n = self.n_dof
        y = np.zeros(n) if y0 is None else _arr(y0)
        driven = source is not None and waveform is not None

        # GPU path: the explicit LSERK4 transient, state device-resident.
        # Falls back to the CPU path only when no GPU is present.
        sdof = None
        if driven:
            sp, sf, sc = source
            sdof = self.probe_dof(sp, field=sf, component=sc)
        if device == "gpu":
            if not self._op.gpu_available():
                _log("transient: no OpenCL GPU available, using CPU")
            elif method == "adaptive":
                # KCL adaptive on the GPU: controller on the device-resident
                # error vector, no cfl_dt call. One waveform sample per
                # output frame for the driven point case (zeroth-order hold).
                t0 = time.time()
                if verbose:
                    _log(f"transient: GPU KCL adaptive "
                         f"({self._op.gpu_device()}, "
                         f"atol={_RK_ATOL:g}, rtol={_RK_RTOL:g})")
                h_op = float(self.c * dt)
                if driven:
                    g_vals = np.array(
                        [float(waveform(k * dt)) for k in range(steps)],
                        dtype=np.float64,
                    )
                    flat, n_acc, n_rej, h_min, h_max = \
                        self._op.gpu_transient_kcl_driven(
                            _arr(y), h_op, int(steps), int(sdof), g_vals,
                            _RK_ATOL, _RK_RTOL, _RK_SAFETY,
                            _RK_GROWTH_LIMIT, _RK_SHRINK_LIMIT,
                            _RK_PI_ALPHA, _RK_PI_BETA,
                            _RK_MIN_STEP_FACTOR,
                        )
                else:
                    flat, n_acc, n_rej, h_min, h_max = \
                        self._op.gpu_transient_kcl(
                            _arr(y), h_op, int(steps),
                            _RK_ATOL, _RK_RTOL, _RK_SAFETY,
                            _RK_GROWTH_LIMIT, _RK_SHRINK_LIMIT,
                            _RK_PI_ALPHA, _RK_PI_BETA,
                            _RK_MIN_STEP_FACTOR,
                        )
                traj = np.asarray(flat).reshape(steps + 1, n)
                if verbose:
                    _log(f"transient complete - {steps} GPU steps "
                         f"in {time.time() - t0:.2f}s")
                    _log(
                        f"  KCL controller: {n_acc} accepted, "
                        f"{n_rej} rejected; h ∈ "
                        f"[{h_min / self.c:.3g}, {h_max / self.c:.3g}] s"
                    )
                return TdTrajectory(traj, problem=self, dt=dt)
            else:
                # Substep so each LSERK4 substep stays within the CFL
                # limit, exactly as the CPU explicit path does.
                cfl = self.cfl_dt()
                nsub = max(1, int(np.ceil(abs(dt) / cfl)))
                h_sub = dt / nsub
                t0 = time.time()
                if driven:
                    # One source amplitude per substep, re-sampled like the
                    # CPU explicit driven path.
                    src = np.array(
                        [float(waveform(i * h_sub))
                         for i in range(steps * nsub)],
                        dtype=np.float64,
                    )
                    if verbose:
                        _log(f"transient: GPU driven LSERK4 "
                             f"({self._op.gpu_device()}, "
                             f"{nsub} substeps/step)")
                    flat = self._op.gpu_transient_driven(
                        _arr(y), float(self.c * dt), int(steps), nsub,
                        int(sdof), src,
                    )
                else:
                    if verbose:
                        _log(f"transient: GPU explicit LSERK4 "
                             f"({self._op.gpu_device()}, "
                             f"{nsub} substeps/step)")
                    flat = self._op.gpu_transient(
                        _arr(y), float(self.c * dt), int(steps), nsub
                    )
                traj = np.asarray(flat).reshape(steps + 1, n)
                if verbose:
                    _log(f"transient complete - {steps} GPU steps "
                         f"in {time.time() - t0:.2f}s")
                return TdTrajectory(traj, problem=self, dt=dt)

        warmup = min(max(int(warmup), 0), steps)
        if method == "adaptive" and warmup:
            raise ValueError(
                "warmup is not supported with method='adaptive' (the "
                "controller stabilises itself in the first few frames)"
            )
        # The explicit integrator is CFL-bound; resolve the limit once so
        # the post-warmup phase can substep within it.
        cfl = self.cfl_dt() if method == "explicit" else None
        if verbose and method == "explicit":
            nsub = max(1, int(np.ceil(abs(dt) / cfl)))
            _log(f"transient: {warmup} exponential warmup step(s), then "
                 f"explicit LSERK4 ({nsub} substeps/step)")
        if verbose and method == "adaptive":
            _log(f"transient: CPU KCL adaptive "
                 f"(atol={_RK_ATOL:g}, rtol={_RK_RTOL:g})")
        traj = np.empty((steps + 1, n))
        traj[0] = y
        t0 = time.time()
        every = max(1, steps // 10)
        label = "driven transient" if driven else "transient"
        # Adaptive controller state — carried across frames so a long quiet
        # phase doesn't reacquire its step from scratch. Initial guess:
        # one full output cadence; first frame will get cut quickly by the
        # PI controller if the operator is stiffer than that.
        h_ad = dt
        prev_err = 0.0
        total_acc, total_rej = 0, 0
        h_min_log, h_max_log = float("inf"), 0.0
        for k in range(steps):
            phase = "exponential" if k < warmup else method
            if phase == "adaptive":
                if driven:
                    y, h_ad, prev_err, n_acc, n_rej = self._advance_adaptive(
                        y, dt, h=h_ad, prev_err_norm=prev_err,
                        t_offset=k * dt, sdof=sdof, waveform=waveform,
                    )
                else:
                    y, h_ad, prev_err, n_acc, n_rej = self._advance_adaptive(
                        y, dt, h=h_ad, prev_err_norm=prev_err,
                        t_offset=k * dt,
                    )
                total_acc += n_acc
                total_rej += n_rej
                h_min_log = min(h_min_log, h_ad)
                h_max_log = max(h_max_log, h_ad)
            elif driven:
                y = self._advance_driven(y, k * dt, dt, phase, sdof,
                                         waveform, krylov_dim, cfl)
            else:
                y = self._advance(y, dt, phase, krylov_dim, cfl)
            traj[k + 1] = y
            if verbose and (k + 1) % every == 0:
                el = time.time() - t0
                eta = el / (k + 1) * (steps - k - 1)
                _log(
                    f"{label} {k + 1}/{steps}  "
                    f"({el:.1f}s elapsed, ETA {eta:.0f}s)"
                )
        if verbose:
            _log(f"{label} complete - {steps} steps "
                 f"in {time.time() - t0:.1f}s")
            if method == "adaptive":
                _log(
                    f"  KCL controller: {total_acc} accepted, "
                    f"{total_rej} rejected; h ∈ "
                    f"[{h_min_log:.3g}, {h_max_log:.3g}] s"
                )
        return TdTrajectory(traj, problem=self, dt=dt)

    # -- model-order reduction --------------------------------------------
    def macromodel(self, *, r=120, sprim=False, shift_freq_hz=None,
                   shift_freqs_hz=None, n_shift_steps=2,
                   gmres_max_iter=None, gmres_tol=None):
        """Build a compact MIMO macromodel of the modal-port network.

        Block-Krylov projection of the matrix-free operator seeded by all
        port-injection vectors — the reduced state-space ``(Â, B̂, Ĉ)``
        whose ``(A, B, C, D)`` realisation exports to any external
        simulator (:meth:`TdMacroModel.state_space`). The result also
        evaluates the S-matrix in microseconds per frequency and writes
        Touchstone. See ``docs/td-macromodel-plan.md`` for the method.

        Parameters
        ----------
        r : int
            Requested block-Krylov dimension. Tens to a few hundred is
            the impulse-Krylov regime; deflation may reduce the realised
            order.
        sprim : bool
            Use the SPRIM structure-preserving E/H block-split build for
            better passivity preservation. Mutually exclusive with the
            shift options.
        shift_freq_hz : float, optional
            Single-shift rational-Krylov build with the shift at this
            physical frequency — concentrates the basis around one
            frequency.
        shift_freqs_hz : sequence of float, optional
            Multi-shift broadband rational-Krylov build (the multipole
            ROM): shift centres in Hertz across the design band. The
            union of per-shift bases spans in-band physics that
            single-shift cannot reach on a non-resonant line. Mutually
            exclusive with ``sprim`` and ``shift_freq_hz``.
        n_shift_steps : int
            Per-port ``(sigma_k I - A)^{-1}`` applications per shift
            (default 2).
        gmres_max_iter : int, optional
            Cap on the matrix-free GMRES iterations per shift-invert solve
            (shift builds only). The default suits a low-spectral-radius
            operator (order 1); raise it (e.g. 150-400) for a larger,
            higher-order operator whose wider imaginary spectrum needs a
            bigger Krylov basis to converge — this is the knob that lets
            the multipole ROM reach the in-band physics at order 2.
        gmres_tol : float, optional
            Relative residual tolerance for that GMRES (shift builds only).

        Returns
        -------
        TdMacroModel
        """
        if self._op.n_ports() == 0:
            raise RuntimeError(
                "ProblemTD has no ports — attach a port (RectWaveguidePort, "
                "CoaxPort, WavePort, FloquetPort) to the geometry before "
                "calling macromodel"
            )
        n_shift_modes = sum(
            x is not None for x in (shift_freq_hz, shift_freqs_hz)
        )
        if sprim and n_shift_modes > 0:
            raise ValueError(
                "macromodel: sprim is mutually exclusive with "
                "shift_freq_hz / shift_freqs_hz"
            )
        if n_shift_modes > 1:
            raise ValueError(
                "macromodel: pass either shift_freq_hz (single shift) "
                "or shift_freqs_hz (multi-shift), not both"
            )
        omegas_op = None
        if shift_freq_hz is not None:
            omegas_op = [2.0 * np.pi * float(shift_freq_hz) / self.c]
            mode = f"single-shift at {shift_freq_hz / 1e9:.3f} GHz"
        elif shift_freqs_hz is not None:
            fs = [float(f) for f in shift_freqs_hz]
            omegas_op = [2.0 * np.pi * f / self.c for f in fs]
            mode = (
                f"multi-shift ({len(fs)} shifts at "
                f"{fs[0] / 1e9:.2f}-{fs[-1] / 1e9:.2f} GHz, "
                f"n_shift_steps={int(n_shift_steps)})"
            )
        else:
            mode = "SPRIM" if sprim else "plain"
        _log(
            f"macromodel - {mode} block-Krylov r={int(r)} on "
            f"{self.n_dof} DOFs"
        )
        native = self._op.macromodel(
            int(r), bool(sprim), omegas_op, int(n_shift_steps),
            None if gmres_max_iter is None else int(gmres_max_iter),
            None if gmres_tol is None else float(gmres_tol),
        )
        m = TdMacroModel(native, self.c)
        _log(
            f"macromodel complete - reduced order r={m.r}, "
            f"n_ports={m.n_ports}"
        )
        return m

    # -- field export ------------------------------------------------------
    def export_vtk(self, states, path, *, times=None):
        """Write a DG field trajectory as a ParaView-openable VTK series.

        Each snapshot in ``states`` becomes a ``.vtu`` file; a ``.pvd``
        collection ties them into a time animation. The field is exported
        on **discontinuous linear tetrahedra** — one cell per DG element,
        sampled at the element corners — carrying the ``E`` and ``H``
        vector fields as point data. Corner values are exact;
        sub-element high-order variation is not rendered.

        Parameters
        ----------
        states : ndarray
            A single state ``[n_dof]`` or a trajectory
            ``[n_snapshots, n_dof]`` — e.g. the return of
            :meth:`transient`.
        path : str or os.PathLike
            Output base path. ``<path>.pvd`` and ``<path>_NNNN.vtu`` are
            written (the parent directory is created if missing).
        times : array_like, optional
            Time value per snapshot for the ``.pvd`` timeline; defaults
            to the snapshot index.

        Returns
        -------
        str
            Path of the ``.pvd`` collection file.
        """
        states = np.ascontiguousarray(states, dtype=np.float64)
        if states.ndim == 1:
            states = states[None, :]
        n_snap, n_dof = states.shape
        if n_dof != self.n_dof:
            raise ValueError(
                f"states carry {n_dof} DOFs, expected {self.n_dof}"
            )

        o = self.order
        np_ = (o + 1) * (o + 2) * (o + 3) // 6
        n_elem = self.n_dof // (6 * np_)
        corners = np.array(self._op.corner_local_nodes(), dtype=np.int64)

        # Discontinuous linear tets: 4 corner points per element.
        coords = self._op.node_coords().reshape(n_elem, np_, 3)
        corner_xyz = coords[:, corners, :].reshape(-1, 3)
        conn = np.arange(n_elem * 4, dtype=np.int64)
        offsets = np.arange(4, n_elem * 4 + 1, 4, dtype=np.int64)
        cell_types = np.full(n_elem, 10, dtype=np.uint8)  # 10 = VTK_TETRA

        if times is None:
            times = np.arange(n_snap, dtype=float)
        else:
            times = np.asarray(times, dtype=float).ravel()
            if times.size != n_snap:
                raise ValueError(
                    f"times has {times.size} entries, expected {n_snap}"
                )

        base = os.fspath(path)
        parent = os.path.dirname(base)
        if parent:
            os.makedirs(parent, exist_ok=True)
        stem = os.path.basename(base)

        entries = []
        for s in range(n_snap):
            fields = states[s].reshape(n_elem, np_, 6)[:, corners, :]
            vtu = f"{base}_{s:04d}.vtu"
            _write_vtu(
                vtu, corner_xyz, conn, offsets, cell_types,
                {
                    "E": fields[..., 0:3].reshape(-1, 3),
                    "H": fields[..., 3:6].reshape(-1, 3),
                },
            )
            entries.append((float(times[s]), f"{stem}_{s:04d}.vtu"))

        pvd = f"{base}.pvd"
        _write_pvd(pvd, entries)
        _log(f"export_vtk - {n_snap} snapshot(s) -> {pvd}")
        return pvd
