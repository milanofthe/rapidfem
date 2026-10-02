# SPDX-License-Identifier: AGPL-3.0-only
#
# Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

"""Time-domain DGTD problem, :class:`ProblemTD`.

`ProblemTD` is the time-domain counterpart of :class:`ProblemFD`. Where
`ProblemFD` is an analysis tool (geometry in, S-parameters out), `ProblemTD`
is a *model-export* tool: it compiles a cavity into a linear ODE
``dy/dt = A·y`` and exposes it at every level of abstraction,

* :meth:`transient`           : turnkey, propagate an initial state,
* :meth:`step`                : advance the state one exponential step,
* :meth:`rhs` / :meth:`jacobian`, the ODE right-hand side / constant Jacobian,
* :meth:`state_space`         : the verbatim sparse operator ``A``.

The current backend meshes a structured box cavity with PEC walls; general
geometry support follows the frequency-domain ``(mesh, TOML)`` path.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .._native import TdSession
from ._model import build_model
from ..excitation import GaussianPulse

_FLUX = {"upwind": 1.0, "central": 0.0}
_FIELD = {"E": 0, "H": 1}
_COMP = {"x": 0, "y": 1, "z": 2}

# Speed of light (m/s). The DG operator runs in normalised units (c = 1, time
# measured in metres); `c` maps operator results to physical SI units,
# `t_op = c·t_seconds`, `f_Hz = c·ω_op/(2π)`.
C_LIGHT = 299_792_458.0

def _log(msg):
    """Progress logging for long TD runs, to stderr, like the FD solver."""
    print(f"  [rapidfem-td] {msg}", file=sys.stderr, flush=True)


def _arr(y):
    """A contiguous 1-D float64 array, the zero-copy form the native
    operator reads directly from its buffer (no Python-list round-trip)."""
    return np.ascontiguousarray(y, dtype=np.float64).ravel()


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
        """``dy/dt`` at state ``y``. The ``t`` argument is ignored, the
        system is autonomous and linear, but kept for the integrator
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

    def __init__(self, problem, dt, krylov_dim, method="exponential"):
        self._p = problem
        self.dt = float(dt)
        self.krylov_dim = int(krylov_dim)
        self.method = method

    def __call__(self, y):
        frames, _ = self._p._op.transient(
            _arr(y), dt=self.dt, steps=1, method=self.method,
            krylov_dim=self.krylov_dim, verbose=False)
        return frames[-1]

    def advance(self, y):
        """Advance ``y`` by one ``dt`` step, same as calling the stepper."""
        return self(y)

    def __repr__(self):
        return f"TdStepper(dt={self.dt:g}, method={self.method!r})"


def _probe_method(device):
    """The integrator of a probe run: exponential on the CPU, the explicit
    LSERK4 on the GPU (its fast path)."""
    return "explicit" if device == "gpu" else "exponential"


def _point_label(spec):
    """Human-readable label for a ``(point, field, component)`` probe/source
    spec, e.g. ``"E_z @ (0.25, 0.25, 0.5)"``."""
    p, f, c = spec
    coords = ", ".join(f"{v:g}" for v in np.asarray(p, dtype=float).ravel())
    return f"{f}_{c} @ ({coords})"


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
    """A time-domain field trajectory, ``[n_snapshot, n_dof]``.

    For every numerical purpose this *is* a :class:`numpy.ndarray`,
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


class ProblemTD:
    """Time-domain DGTD Maxwell problem ready for analysis.

    A container around a meshed :class:`~rapidfem.Geometry` and its
    attached materials, ports and BCs, the time-domain counterpart of
    :class:`~rapidfem.ProblemFD`. The curl equations are discretised in
    space with a **nodal discontinuous Galerkin** method on tetrahedra,
    giving an explicit linear ODE ``dy/dt = A·y`` with a constant, sparse
    operator ``A``; a driven port adds a rank-1 source ``b(t)``.

    ``ProblemTD`` is a **model-export tool**, it hands back that ODE at
    every level of abstraction, so the verb to call is just the level of
    detail wanted:

    - :meth:`rhs` / :meth:`state_space`, the matrix-free right-hand side,
      or the verbatim sparse operator ``A``
    - :meth:`ode`, a handoff object for an external integrator
      (e.g. ``scipy.integrate.solve_ivp``)
    - :meth:`step` / :meth:`stepper` / :meth:`transient`, exact
      exponential time stepping (matrix-free Krylov / ETD)
    - :meth:`driven_transient` / :meth:`transfer_function`:
      soft-source or modal-port excitation and a scalar transfer function
    - :meth:`resonances`, cavity eigenfrequencies from the spectrum
    - :meth:`export_vtk`, a VTK field animation

    Because the semi-discrete system is linear with a constant ``A``, the
    exponential propagator is *exact* at any step size, the time step is
    set by the wanted output cadence, not by a CFL stability limit.

    Note
    ----
    The geometry must already be meshed (via ``g.mesh()``) before the
    ProblemTD is constructed, construction snapshots the mesh bytes.
    Re-meshing the geometry afterwards has no effect on an existing
    ProblemTD; construct a new one instead. :meth:`box` is a shortcut that
    builds directly on a structured box cavity, bypassing the geometry
    API, handy for validation.

    Example
    -------
    Build a waveguide problem, then read the model at two levels:

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

    Attributes
    ----------
    n_dof : int
        state-vector length, ``6·Np·n_elem``
    order : int
        the DG polynomial order the operator was built at
    flux : str
        the numerical flux in use, ``"upwind"`` or ``"central"``
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
                "geometry not meshed yet, call g.mesh() before "
                "constructing a ProblemTD"
            )
        self.c = float(c)
        model = build_model(geometry)
        # Near-field-to-far-field is a frequency-domain post-process; the TD
        # operator has no NFFT path, so a FarFieldSurface here is meshed and
        # tagged but never consumed. Warn rather than mislead.
        from ..physics import FarFieldSurface
        if any(isinstance(p, FarFieldSurface)
               for p in getattr(geometry, "_physics", [])):
            import warnings
            warnings.warn(
                "FarFieldSurface is ignored by the time-domain backend "
                "(near-field-to-far-field is frequency-domain only); it has "
                "no effect on a ProblemTD analysis.",
                stacklevel=2,
            )
        self._op = TdSession.from_model(
            geometry._fem_mesh, model, order, _FLUX[flux], self.c)
        self._geometry = geometry
        self.order = order
        self.flux = flux
        _log(f"operator built - {self.n_dof} DOFs, order {order}, flux={flux}")

    @classmethod
    def box(cls, *, size, cells, order=2, flux="upwind", c=1.0):
        """Build directly on a structured box cavity, bypassing the geometry
        API, handy for validation and quick experiments.

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
        obj._op = TdSession.box_cavity(nx, ny, nz, lx, ly, lz, order,
                                       _FLUX[flux], float(c))
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

    def rhs_into(self, y, out):
        """Allocation-free :meth:`rhs`: write ``dy/dt = A·y`` into ``out``.

        ``out`` must be a contiguous float64 array of length :attr:`n_dofs`.
        Reuse one buffer across a hand-rolled integration loop to avoid a
        fresh allocation per evaluation. The built-in steppers already do
        this internally, so prefer :meth:`step` / :meth:`transient` unless
        you are driving the operator yourself.

        Parameters
        ----------
        y : array_like
            state vector of length :attr:`n_dofs`
        out : numpy.ndarray
            float64 output buffer of length :attr:`n_dofs`, overwritten in place
        """
        self._op.apply_into(_arr(y), out)

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
        """The (constant) Jacobian of the linear system, i.e. ``A`` itself,
        as a sparse matrix. See :meth:`state_space`."""
        return self.state_space()

    def state_space(self):
        """The verbatim operator ``A`` as a :class:`scipy.sparse.csr_matrix`.

        Requires scipy (``pip install 'rapidfem[td]'``).
        """
        try:
            from scipy.sparse import csr_matrix
        except ImportError as e:
            raise ImportError("scipy is required for state_space() and "
                              "jacobian(). Install with: pip install 'rapidfem[td]'") from e

        n, row_ptr, col_idx, values = self._op.state_space()
        return csr_matrix((values, col_idx, row_ptr), shape=(n, n))

    def ode(self):
        """Export the problem as an explicit linear ODE ``dy/dt = A·y``.

        Returns a :class:`TdODE` carrying everything an external
        integrator needs, ``n_dof``, a matrix-free ``rhs(t, y)`` with
        the :func:`scipy.integrate.solve_ivp` signature, and
        ``jacobian()``.
        """
        return TdODE(self)

    def resonances(self, *, n=8):
        """Cavity resonant frequencies (Hz) from the operator's spectrum.

        The DG Maxwell operator's eigenvalues are `±iω`; with the upwind flux
        the physical modes are the least-damped ones, `f = c·|ω|/(2π)`.
        Dense eigenvalue solve, so for modest meshes only.
        """
        return np.asarray(self._op.resonances(n))

    # -- mid level: stepping ------------------------------------------------
    def step(self, y, h, krylov_dim=40, tol=None):
        """Advance the state by ``h`` (in the same time units as ``c``) with
        the matrix-free exponential propagator (a Krylov/Arnoldi exponential
        integrator), exact for the linear homogeneous system at any ``h``.

        ``tol`` is the Krylov a-posteriori error tolerance; ``None`` keeps
        the solver default. A converged step uses far fewer than
        ``krylov_dim`` matvecs; ``tol=0`` forces the full ``krylov_dim``
        (the fixed-dimension worst case).

        See also :meth:`step_explicit` (the cheaper CFL-bound LSERK4) and
        :meth:`step_adaptive` (embedded KCL 4(3)5 with self-tuned step),
        plus :meth:`transient` for the high-level loop."""
        if tol is None:
            return self._op.step(_arr(y), float(h), int(krylov_dim))
        return self._op.step(_arr(y), float(h), int(krylov_dim), float(tol))

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
        return self._op.step_explicit(_arr(y), float(h))

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
            The advanced state, shape ``[n_dof]``, the fourth-order main
            solution.
        err : ndarray
            Per-DOF embedded-error vector ``y_4 − y_3``, shape
            ``[n_dof]``. A controller compares its weighted L2 norm
            against 1: smaller means the step was easy and the next one
            can grow, above 1 means the step must be rejected and shrunk.
            The controller of :meth:`transient` uses ``atol = 1e-8``,
            ``rtol = 1e-4``.

        Notes
        -----
        Conditionally stable like :meth:`step_explicit`, an ``h`` past
        the operator's CFL limit eventually NaNs. The controller in
        :meth:`transient` rejects any infinite ``err`` and shrinks ``h``,
        so the high-level loop survives a bad initial-guess step; the raw
        stepper here does not."""
        return self._op.step_adaptive(_arr(y), float(h))

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
        return self._op.cfl_dt(recompute)

    def stepper(self, dt, *, krylov_dim=40, method="exponential"):
        """A reusable one-step propagator bound to a fixed ``dt``.

        Returns a :class:`TdStepper`, call it repeatedly to advance a
        state without re-passing ``dt``/``krylov_dim`` each time.

        ``method`` selects the integrator: ``"exponential"`` (exact at any
        ``dt``) or ``"explicit"`` (the cheaper LSERK4 stepper, substepped
        to respect its CFL limit)."""
        if method not in ("exponential", "explicit"):
            raise ValueError("method must be 'exponential' or 'explicit'")
        return TdStepper(self, dt, krylov_dim, method)

    # -- ports: soft sources & field probes --------------------------------
    def probe_dof(self, point, *, field="E", component="z"):
        """Global DOF index for a field component at the node nearest
        ``point``, used to place soft sources and field probes."""
        return self._op.nearest_node_dof(
            tuple(float(x) for x in point), _FIELD[field], _COMP[component]
        )

    def _spec_dof(self, spec):
        """The DOF of a ``(point, field, component)`` spec."""
        p, f, c = spec
        return self.probe_dof(p, field=f, component=c)

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
            ``g(t)``, the excitation, e.g. a :class:`~rapidfem.GaussianPulse`.
        probes : list of (point, field, component)
            Field samples to record over the run.
        dt, steps : float, int
            Time step and step count.

        Returns
        -------
        TdResponse
            Iterates as ``(times, responses)``, ``times`` of shape
            ``[steps+1]`` and ``responses`` of shape
            ``[n_probes, steps+1]``, so ``times, resp = ...`` unpacking
            works unchanged; passing it to :func:`rapidfem.show` plots the
            probe signals.
        """
        frames, _ = self._op.transient(
            None, dt=float(dt), steps=int(steps),
            source_dof=self._spec_dof(source), waveform=waveform,
            probes=[self._spec_dof(p) for p in probes],
            method=_probe_method(device), device=device,
            krylov_dim=int(krylov_dim), verbose=verbose)
        return TdResponse(
            np.arange(steps + 1) * dt, frames.T,
            source_label=_point_label(source),
            probe_labels=[_point_label(p) for p in probes],
        )

    def transfer_function(
        self, *, source, probe, pulse, dt, steps, krylov_dim=40,
        device="cpu", verbose=True,
    ):
        """Field-to-field frequency response by on-the-fly RFT.

        Drives a broadband ``pulse`` at ``source``, records ``probe``,
        then divides the probe spectrum by the source spectrum,
        ``H(f) = R(f) / G(f)``, to recover the linear cavity's transfer
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
            Broadband excitation ``g(t)``, its spectrum sets the usable
            frequency band (a :class:`~rapidfem.GaussianPulse` is the
            typical choice).
        dt, steps : float, int
            Time step and step count; together they fix the frequency
            resolution ``1/(steps·dt)`` and the Nyquist limit
            ``1/(2·dt)``.

        Returns
        -------
        TdTransfer
            Iterates as ``(freqs, H)``, ``freqs`` the frequency axis (Hz
            for an SI geometry, operator units for a :meth:`box`, length
            ``steps//2 + 1``) and ``H`` the complex transfer function
            ``R(f)/G(f)``, zero outside the pulse band. ``freqs, H = ...``
            unpacking works unchanged; :func:`rapidfem.show` plots it.
        """
        freqs, h = self._op.transfer_function(
            self._spec_dof(source), self._spec_dof(probe), pulse,
            dt=float(dt), steps=int(steps), method=_probe_method(device),
            device=device, krylov_dim=int(krylov_dim), verbose=verbose)
        return TdTransfer(
            freqs, h,
            source_label=_point_label(source),
            probe_label=_point_label(probe),
        )

    # -- turnkey: a transient run ------------------------------------------
    def _modal_ports(self):
        """The geometry's modal port physics objects in the operator's
        declaration order: rect, coax, floquet, wave. Matches
        :func:`rapidfem_td::build::operator_from_model` port layout, so the
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
        """Operator port index of a modal port physics object, the index
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
        return self._op.modal_port(k)

    def port_signals(self, traj, ports, *, dt=None, labels=None):
        """Modal wave amplitude ``P_e(t)`` at each port over a trajectory,
        as a :class:`TdResponse` for :func:`rapidfem.show`, a time-domain
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
            dt = getattr(traj, "_dt", None) or 1.0
        traj = np.ascontiguousarray(traj, dtype=np.float64)
        idxs = [self._port_operator_index(p) for p in ports]
        rows = self._op.port_signals(traj, idxs)
        labs = list(labels) if labels else [f"port {k}" for k in range(len(idxs))]
        return TdResponse(np.arange(traj.shape[0]) * dt, rows, probe_labels=labs)

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
            is stiff. All three hold the source over a step: the
            exponential integrator samples the waveform once per output
            frame, explicit once per substep, adaptive once per substep on
            the CPU and once per frame on the GPU.

            *Adaptive vs the frequency-domain* :class:`~rapidfem.Adaptive`
            *class*: unrelated, :class:`~rapidfem.Adaptive` drives mesh /
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
            the CPU when no GPU is present, for dispersive materials, and
            for the exponential integrator on a GPU without fp64 (e.g.
            Apple Silicon), each with a log line.

        Returns
        -------
        TdTrajectory
            The field trajectory, shape ``[steps + 1, n_dof]``. It *is* a
            :class:`numpy.ndarray` for every numerical purpose (indexing,
            slicing, :meth:`export_vtk`); passing it to
            :func:`rapidfem.show` plays it back as a 3-D field animation
            in the UI.
        """
        source_dof = vector = None
        if port is not None:
            if source is not None:
                raise ValueError(
                    "pass either source= (point) or port= (modal port), "
                    "not both"
                )
            vector = self._op.port_source(self._port_operator_index(port))
        elif source is not None:
            source_dof = self._spec_dof(source)
        frames, _ = self._op.transient(
            None if y0 is None else _arr(y0), dt=float(dt), steps=int(steps),
            source_dof=source_dof, source=vector, waveform=waveform,
            method=method, device=device, krylov_dim=int(krylov_dim),
            warmup=int(warmup), verbose=verbose)
        return TdTrajectory(frames, problem=self, dt=dt)

    # -- field export ------------------------------------------------------
    def export_vtk(self, states, path, *, times=None):
        """Write a DG field trajectory as a ParaView-openable VTK series.

        Each snapshot in ``states`` becomes a ``.vtu`` file; a ``.pvd``
        collection ties them into a time animation. The field is exported
        on **discontinuous linear tetrahedra**, one cell per DG element,
        sampled at the element corners, carrying the ``E`` and ``H``
        vector fields as point data. Corner values are exact;
        sub-element high-order variation is not rendered.

        Parameters
        ----------
        states : ndarray
            A single state ``[n_dof]`` or a trajectory
            ``[n_snapshots, n_dof]``, e.g. the return of
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
        n_snap = states.shape[0]
        if times is None:
            times = np.arange(n_snap, dtype=float)
        times = [float(t) for t in np.asarray(times, dtype=float).ravel()]
        pvd = self._op.export_vtk(states, times, os.fspath(path))
        _log(f"export_vtk - {n_snap} snapshot(s) -> {pvd}")
        return pvd
