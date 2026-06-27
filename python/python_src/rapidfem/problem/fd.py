#########################################################################################
##
##                                  PROBLEM
##                                 (problem.py)
##
#########################################################################################

# IMPORTS ===============================================================================

from __future__ import annotations

from typing import Iterable

import numpy as np

from .._native import Simulation as _NativeSimulation
from .._fmt import _f64
from ..geometry import Geometry
from ..physics import PEC, PML, FarFieldSurface


# HELPERS ===============================================================================


# ERROR INDICATOR =======================================================================

class ErrorIndicator:
    """Per-tetrahedron residual error indicator from :meth:`ProblemFD.element_errors`.

    Holds the Monk-style a-posteriori η values for one
    ``(frequency, port)`` combination, plus the Dörfler-marked subset
    that an AMR loop would refine.


    Note
    ----
    The indicator is purely diagnostic, calling
    :meth:`ProblemFD.element_errors` does **not** re-mesh or re-solve.
    For an end-to-end adaptive sweep that consumes the indicator, pass
    an :class:`Adaptive` to :meth:`ProblemFD.sweep`.


    Attributes
    ----------
    eta : np.ndarray
        per-tet error indicator η, shape ``(n_tets,)``, float64
    total : float
        global L² error :math:`\\sqrt{\\sum_K \\eta_K^2}`
    marked : np.ndarray
        int64 tet indices selected by Dörfler marking at ``theta``
    volume_residuals : np.ndarray
        volume-residual contribution per tet, shape ``(n_tets,)``
    face_jumps : np.ndarray
        face-jump contribution per tet (accumulated over its 4 faces)
    h_k : np.ndarray
        per-tet element diameter (max edge length), useful for
        choosing a refinement target relative to current local size
    tet_centroids : np.ndarray
        per-tet centroid coordinates, shape ``(n_tets, 3)``, m
    freq_hz : float
        frequency at which the indicator was computed
    theta : float
        Dörfler fraction used for marking
    """

    def __init__(self, *, eta, total, marked, volume_residuals, face_jumps,
                 h_k, tet_centroids, freq_hz: float, theta: float):
        self.eta = eta
        self.total = float(total)
        self.marked = marked
        self.volume_residuals = volume_residuals
        self.face_jumps = face_jumps
        self.h_k = h_k
        self.tet_centroids = tet_centroids
        self.freq_hz = float(freq_hz)
        self.theta = float(theta)

    def __repr__(self) -> str:
        n = len(self.eta)
        return (f"ErrorIndicator(n_tets={n}, total={self.total:.4e}, "
                f"marked={len(self.marked)} ({100*len(self.marked)/max(n,1):.1f}%), "
                f"freq={self.freq_hz/1e9:.3f} GHz)")


# ADAPTIVE REFINEMENT ===================================================================

class Adaptive:
    """Adaptive-mesh-refinement settings for :meth:`ProblemFD.sweep`.

    Drives a Dörfler-marking loop on top of the driven sweep, elements
    carrying the highest residual error get their local mesh size cut
    by ``refinement_ratio`` and the sweep is repeated. The default
    parameters mirror the rapidfem CLI's standard adaptive flow.


    Note
    ----
    The adaptive loop runs inside the Rust solver and is reported via
    the regular ``stderr`` log stream, there's no separate Python
    progress callback yet.


    Example
    -------
    .. code-block:: python

        result = prob.sweep(freqs, adaptive=rf.Adaptive(theta=0.6))


    Parameters
    ----------
    theta : float
        Dörfler-marking fraction (elements carrying the top
        :math:`\\theta` of the residual error are marked)
    refinement_ratio : float
        local size reduction applied to marked elements
    """

    def __init__(self, *, theta: float = 0.5, refinement_ratio: float = 0.5):
        self.theta = float(theta)
        self.refinement_ratio = float(refinement_ratio)


# PROBLEM ===============================================================================

class ProblemFD:
    """Frequency-domain FEM problem ready for analysis.

    Generic container around a meshed :class:`Geometry`, its attached
    materials, ports, and BCs. Multiple analyses can run on the same
    problem instance without re-meshing:

    - :meth:`sweep` for driven S-parameter sweeps
    - :meth:`eigenmode` for modal / resonator analysis
    - :meth:`farfield` for post-sweep radiation patterns

    Each analysis re-assembles the TOML config from the geometry's
    physics registry and hands it to the native Rust solver. The most
    recent native :class:`Simulation` instance is cached so follow-ups
    like :meth:`farfield` reuse the same assembly without re-solving.


    Note
    ----
    The geometry must already be meshed (via ``g.mesh()``) before the
    ProblemFD is constructed, the ProblemFD snapshot copies the mesh
    bytes on init. Re-meshing the geometry afterwards has no effect on
    an existing ProblemFD; construct a new one instead.


    Example
    -------
    Three analyses on a single dielectric resonator problem:

    .. code-block:: python

        g = rf.Geometry(maxh=rf.lambda_maxh(f_max=3e9))
        air = g.box(W, W, H, material=rf.Air())
        rf.PEC(air.faces.min(axis="x"), air.faces.max(axis="x"),
               air.faces.min(axis="y"), air.faces.max(axis="y"),
               air.faces.min(axis="z"), air.faces.max(axis="z"))
        g.mesh()

        prob = rf.ProblemFD(g)
        modes  = prob.eigenmode(target_frequency=2e9, n_modes=5)
        result = prob.sweep(np.linspace(1.8e9, 2.2e9, 21))
        pattern = prob.farfield(result, freq_idx=10, port_idx=0)


    Parameters
    ----------
    geometry : rapidfem.Geometry
        a geometry on which ``g.mesh()`` has already been called


    Attributes
    ----------
    native : rapidfem._native.Simulation
        the underlying native solver instance, populated after the
        first analysis call (raises if accessed before)
    n_dofs : int
        FEM degree-of-freedom count of the last assembled solver
    n_tets : int
        tetrahedra in the mesh used by the last assembled solver
    """

    def __init__(self, geometry: Geometry):
        if geometry._last_mesh is None:
            raise ValueError(
                "geometry not meshed yet, call g.mesh() before constructing a ProblemFD")
        self._geometry = geometry
        self._mesh_bytes, _ = geometry._last_mesh
        self._native: _NativeSimulation | None = None  # cached after first analysis

    # ── Analyses ──────────────────────────────────────────────────────────

    def sweep(self, frequencies: Iterable[float], *,
              z0: float = 50.0,
              adaptive: Adaptive | None = None,
              on_frequency=None):
        """run a driven frequency sweep and return the SweepResult

        Assembles the FEM operator from the geometry's material /
        port / BC registry, then factors and solves at each frequency
        in ``frequencies``. The returned :class:`SweepResult` has
        ``.frequencies``, ``.sparams`` (complex array of shape
        ``[n_freq, n_port, n_port]``), and ``.solve_time_s``.


        Example
        -------
        .. code-block:: python

            result = prob.sweep(np.linspace(8e9, 12e9, 21))


        Parameters
        ----------
        frequencies : iterable of float
            sweep points in Hz, in evaluation order
        z0 : float
            reference impedance for S-parameter normalisation in ohms
        adaptive : Adaptive, optional
            adaptive-mesh-refinement settings (``None`` disables it)
        on_frequency : callable, optional
            called after each frequency's solve as
            ``on_frequency(freq_idx, freq_hz, s_matrix)`` where ``s_matrix`` is
            the ``(n_driven, n_driven)`` complex S-block for that frequency.
            Useful for progress reporting. When ``None`` and running inside the
            UI, a callback that streams partial results to the viewer is used
            automatically.

        Returns
        -------
        SweepResult
            native solver result handle
        """
        freqs = [float(f) for f in frequencies]
        if not freqs:
            raise ValueError("sweep needs at least one frequency")
        toml = self._assemble_toml(frequencies=freqs, z0=z0, adaptive=adaptive)
        self._native = _NativeSimulation.from_bytes(self._mesh_bytes, toml)
        # The native callback is (freq_idx, freq, s_matrix). Compose an optional
        # user `on_frequency` with the UI's per-frequency streaming callback.
        from rapidfem import _show_capture
        ui_cb = _show_capture.active_sweep_callback()
        user_cb = on_frequency
        if ui_cb is None and user_cb is None:
            callback = None
        else:
            def callback(freq_idx, freq, s_matrix):
                if user_cb is not None:
                    user_cb(freq_idx, freq, s_matrix)
                if ui_cb is not None:
                    ui_cb(freq_idx, freq, s_matrix)
        return self._native.run_sweep(callback)

    def port_impedances(self, frequencies) -> "np.ndarray":
        """Modal characteristic impedance (Ω) of each driven port.

        Returns an array of shape ``(n_freq, n_driven)``: the (generally
        frequency-dependent) modal impedance that each driven port's
        S-parameters are referenced to. Requires a prior analysis call (the
        native simulation must exist). Useful with :meth:`renormalize`.
        """
        if self._native is None:
            raise RuntimeError(
                "run a sweep before querying port impedances (no native "
                "simulation yet)"
            )
        freqs = np.asarray(list(frequencies), dtype=float)
        n = self._native.n_driven_ports
        return np.array(
            [[self._native.port_z_mode(p, float(f)) for p in range(n)]
             for f in freqs]
        )

    def renormalize(self, result, z_ref: float = 50.0) -> "np.ndarray":
        """Renormalize a sweep's S-parameters to a fixed reference impedance.

        Modal ports (waveguide / wave) report S-parameters self-referenced to
        their modal impedance, so a matched modal line reads |S11|≈0 against
        that reference regardless of its characteristic impedance, and the
        ``z0`` passed to :meth:`sweep` only *labels* the output. This returns
        the S-matrix re-referenced to ``z_ref`` (e.g. 50 Ω) so mismatches
        against a standard reference appear. Lumped ports (already at a fixed
        z0) are a no-op. Returns a ``(n_freq, n_driven, n_driven)`` array;
        ``result`` is unchanged.
        """
        from .. import io
        z_old = self.port_impedances(result.frequencies)
        return io.renormalize_sparams(result.sparams, z_old, z_ref)

    def eigenmode(self, target_frequency: float, *,
                  n_modes: int = 6,
                  z0: float = 50.0):
        """run a modal solve around ``target_frequency``

        Uses shift-invert Lanczos with the configured direct factoriser
        (PARDISO when available, faer otherwise) as the inner solver.
        Returns the list of :class:`Eigenmode` instances ordered by
        distance from the shift frequency.


        Example
        -------
        Cavity resonator's first 5 modes near 2 GHz:

        .. code-block:: python

            modes = prob.eigenmode(target_frequency=2e9, n_modes=5)
            for m in modes:
                print(m.frequency_hz, m.q_factor)


        Parameters
        ----------
        target_frequency : float
            spectral shift in Hz; modes nearest this frequency are
            returned
        n_modes : int
            number of eigenpairs requested
        z0 : float
            reference impedance (only affects the output block in TOML;
            eigenmodes themselves don't depend on it)

        Returns
        -------
        list[Eigenmode]
            n_modes solver results, sorted by proximity to
            ``target_frequency``
        """
        toml = self._assemble_toml(
            frequencies=[float(target_frequency)],
            z0=z0,
            eigenmode=(float(target_frequency), int(n_modes)),
        )
        self._native = _NativeSimulation.from_bytes(self._mesh_bytes, toml)
        return self._native.run_eigenmode()

    def farfield(self, result, *,
                 freq_idx: int,
                 port_idx: int,
                 n_theta: int = 91,
                 n_phi: int = 72):
        """compute a far-field radiation pattern from a sweep result

        Evaluates the near-field-to-far-field transform on the NFFT
        surface for the chosen (frequency, driven port) combination
        and samples the result on a uniform :math:`(\\theta, \\phi)`
        grid.


        Note
        ----
        Must be called after :meth:`sweep`; raises otherwise. The far-field
        uses the most recent native solver instance, so calling
        :meth:`sweep` again invalidates earlier ``result`` handles for
        far-field purposes.


        Example
        -------
        Pattern at the resonance frequency of a patch antenna:

        .. code-block:: python

            result = prob.sweep(freqs)
            fi = int(np.argmin([abs(result.sparams[i, 0, 0])
                                for i in range(len(freqs))]))
            pattern = prob.farfield(result, freq_idx=fi, port_idx=0)
            print(pattern.peak_directivity_dbi)


        Parameters
        ----------
        result : SweepResult
            return value of a prior :meth:`sweep` call
        freq_idx : int
            frequency index into ``result.frequencies``
        port_idx : int
            driven-port index into ``result.sparams``
        n_theta : int
            number of elevation samples
        n_phi : int
            number of azimuth samples

        Returns
        -------
        RadiationPattern
            native pattern object. Scalars: ``peak_directivity_dbi``,
            ``peak_gain_dbi`` (dBi), ``radiated_power`` (W). Angle grids:
            ``theta_rad`` ``[n_theta]``, ``phi_rad`` ``[n_phi]`` (radians).
            Per-direction ``[n_phi, n_theta]`` arrays: ``directivity_dbi``,
            ``gain_dbi``, ``lcp_dbi``, ``rcp_dbi`` (dBi), ``axial_ratio_db``
            (dB), and complex ``e_theta`` / ``e_phi`` field components.
        """
        if self._native is None:
            raise ValueError(
                "call .sweep(...) before .farfield(...), far-field needs a solved problem")
        return self._native.compute_farfield(result, freq_idx, port_idx, n_theta, n_phi)

    def element_errors(self, result, *, freq_idx: int = 0, port_idx: int = 0,
                       theta: float = 0.5) -> ErrorIndicator:
        """per-tet residual error indicator for one ``(freq, port)`` slice

        Evaluates the Monk-style a-posteriori error estimator on the
        FEM solution stored in ``result``. Returns an
        :class:`ErrorIndicator` with the η values, the Dörfler-marked
        subset, and the tet centroids needed for visualisation. No
        re-mesh or re-solve.


        Example
        -------
        .. code-block:: python

            result = prob.sweep(np.linspace(2e9, 3e9, 11))
            errs = prob.element_errors(result, freq_idx=5, theta=0.3)
            print(errs)         # ErrorIndicator(n_tets=..., marked=...)
            rf.show(errs)       # 3-D field of η over the mesh


        Parameters
        ----------
        result : SweepResult
            return value of a prior :meth:`sweep` call
        freq_idx : int
            frequency index into ``result.frequencies``
        port_idx : int
            driven-port index into ``result.sparams``
        theta : float
            Dörfler-marking fraction (top η pool that accumulates to
            ``theta · total²`` gets marked)

        Returns
        -------
        ErrorIndicator
            diagnostic container with ``eta``, ``total``, ``marked``,
            ``volume_residuals``, ``face_jumps``, ``tet_centroids``
        """
        if self._native is None:
            raise ValueError(
                "call .sweep(...) before .element_errors(...), needs a solved problem")
        d = self._native.element_errors(result, freq_idx, port_idx, theta)
        if d is None:
            raise IndexError(
                f"no solution for (freq_idx={freq_idx}, port_idx={port_idx})")
        nodes = np.asarray(self._native.mesh_nodes)
        tets = np.asarray(self._native.mesh_tets)
        # Per-tet centroid (n_tets, 3), handy for sanity printing /
        # 3-D point-cloud rendering of the indicator.
        centroids = nodes[tets].mean(axis=1)
        freq_hz = float(np.asarray(result.frequencies)[freq_idx])
        return ErrorIndicator(
            eta=np.asarray(d["eta"]),
            total=float(d["total"]),
            marked=np.asarray(d["marked"]),
            volume_residuals=np.asarray(d["volume_residuals"]),
            face_jumps=np.asarray(d["face_jumps"]),
            h_k=np.asarray(d["h_k"]),
            tet_centroids=centroids,
            freq_hz=freq_hz,
            theta=theta,
        )

    # ── Field accessors ───────────────────────────────────────────────────

    def _field_accessor(self, name: str, result, freq_idx: int, port_idx: int):
        """shared body for the (freq_idx, port_idx) node-field wrappers"""
        if self._native is None:
            raise RuntimeError(
                f"call .sweep(...) before .{name}(...), needs a solved problem")
        arr = getattr(self._native, name)(result, freq_idx, port_idx)
        if arr is None:
            raise IndexError(
                f"no solution for (freq_idx={freq_idx}, port_idx={port_idx})")
        return np.asarray(arr)

    def field_at_nodes(self, result, freq_idx: int = 0, port_idx: int = 0):
        """electric field E sampled at every mesh node

        Convenience wrapper so post-processing does not have to reach
        through :attr:`native`.

        Parameters
        ----------
        result : SweepResult
            a solved sweep from :meth:`sweep`
        freq_idx : int
            frequency index into ``result.frequencies``
        port_idx : int
            driven-port index (the excitation that produced the field)

        Returns
        -------
        numpy.ndarray
            complex ``(n_nodes, 3)`` array of (Ex, Ey, Ez) per node, in V/m
        """
        return self._field_accessor("field_at_nodes", result, freq_idx, port_idx)

    def current_density_at_nodes(self, result, freq_idx: int = 0, port_idx: int = 0):
        """loss-equivalent current density J at every mesh node

        ``J = sigma_eff * E`` with ``sigma_eff = omega*eps0*eps_r*tan(delta)
        + sigma_bulk``, so both dielectric (loss tangent) and Ohmic losses
        contribute.

        Parameters
        ----------
        result : SweepResult
            a solved sweep from :meth:`sweep`
        freq_idx : int
            frequency index into ``result.frequencies``
        port_idx : int
            driven-port index

        Returns
        -------
        numpy.ndarray
            complex ``(n_nodes, 3)`` array of (Jx, Jy, Jz) per node, in A/m^2
        """
        return self._field_accessor(
            "current_density_at_nodes", result, freq_idx, port_idx)

    def h_field_at_nodes(self, result, freq_idx: int = 0, port_idx: int = 0):
        """magnetic field H sampled at every mesh node

        ``H = curl(E) / (j*omega*mu0*mu_r)``, derived from the analytic
        Nedelec-2 curl of the FEM solution.

        Parameters
        ----------
        result : SweepResult
            a solved sweep from :meth:`sweep`
        freq_idx : int
            frequency index into ``result.frequencies``
        port_idx : int
            driven-port index

        Returns
        -------
        numpy.ndarray
            complex ``(n_nodes, 3)`` array of (Hx, Hy, Hz) per node, in A/m
        """
        return self._field_accessor("h_field_at_nodes", result, freq_idx, port_idx)

    def mode_field_at_nodes(self, mode):
        """electric field E of an eigenmode sampled at every mesh node

        Parameters
        ----------
        mode : Eigenmode
            one entry returned by :meth:`eigenmode`

        Returns
        -------
        numpy.ndarray
            complex ``(n_nodes, 3)`` array of (Ex, Ey, Ez) per node. The
            magnitude is arbitrary, eigenmodes are defined up to a global scale.
        """
        if self._native is None:
            raise RuntimeError(
                "call .eigenmode(...) before .mode_field_at_nodes(...)")
        arr = self._native.mode_field_at_nodes(mode)
        if arr is None:
            raise IndexError("eigenmode carries no stored field")
        return np.asarray(arr)

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def native(self):
        """the underlying native :class:`Simulation` after an analysis

        Used by the UI serialiser (``rapidfem.ui.api``) to reach the
        low-level mesh / field accessors (``mesh_nodes``,
        ``field_at_nodes``, ``current_density_at_nodes``,
        ``compute_farfield``, ...) that live on the Rust side. Raises
        :class:`RuntimeError` if no analysis has run yet, show()ing a
        ProblemFD before any ``.sweep()`` / ``.eigenmode()`` call has
        nothing to render.
        """
        if self._native is None:
            raise RuntimeError(
                "ProblemFD.native is not available, run .sweep() or "
                ".eigenmode() first to assemble the native solver")
        return self._native

    @property
    def n_dofs(self) -> int:
        """FEM degree-of-freedom count of the last-assembled solver"""
        if self._native is None:
            raise ValueError("run an analysis first to assemble the FEM operator")
        return self._native.n_dofs

    @property
    def n_dof(self) -> int:
        """Alias of :attr:`n_dofs`, matching ProblemTD's attribute name."""
        return self.n_dofs

    @property
    def n_tets(self) -> int:
        """tetrahedron count of the mesh used by the last-assembled solver"""
        if self._native is None:
            raise ValueError("run an analysis first to assemble the FEM operator")
        return self._native.n_tets

    @property
    def mesh_nodes(self):
        """``(n_nodes, 3)`` float64 array of mesh node coordinates, in metres"""
        if self._native is None:
            raise RuntimeError("run an analysis first to assemble the mesh")
        return np.asarray(self._native.mesh_nodes)

    @property
    def mesh_tets(self):
        """``(n_tets, 4)`` int array of tetrahedron node indices"""
        if self._native is None:
            raise RuntimeError("run an analysis first to assemble the mesh")
        return np.asarray(self._native.mesh_tets)

    # ── TOML assembly ─────────────────────────────────────────────────────

    def _assemble_toml(self, *,
                       frequencies: list[float],
                       z0: float,
                       adaptive: Adaptive | None = None,
                       eigenmode: tuple[float, int] | None = None) -> str:
        """build the TOML config string the Rust solver expects

        Walks the geometry's material and physics registries; skips
        materials on PML-targeted volumes (their permittivity comes
        from the PML's stretch profile instead).

        Parameters
        ----------
        frequencies : list[float]
            sweep points to embed in the ``[frequency]`` block
        z0 : float
            S-parameter reference impedance for the ``[output]`` block
        adaptive : Adaptive, optional
            adaptive-refinement parameters
        eigenmode : tuple[float, int], optional
            ``(target_frequency, n_modes)`` for an ``[eigenmode]`` block

        Returns
        -------
        str
            TOML config text
        """
        g = self._geometry
        parts: list[str] = ['[mesh]\nfile = "(in-memory)"\n']

        freqs_str = ", ".join(_f64(f) for f in frequencies)
        parts.append(f"[frequency]\nvalues = [{freqs_str}]\n")

        # Collect volume entities targeted by PML, they get a [[pml]] block
        # and must NOT also generate a [[materials]] entry (the PML carries
        # its own er_base/ur_base, and double-tagging volumes confuses the
        # Rust solver). Mirrors the old builder workflow where a PML volume
        # had no .material at all.
        pml_volume_ids: set[int] = set()
        for phys in g._physics:
            if isinstance(phys, PML):
                for ent in phys._entities:
                    pml_volume_ids.add(id(ent))

        # Materials, group volumes by Material instance; tag came from mesh().
        # Skip Material instances whose every-volume is a PML target.
        seen_materials: set[int] = set()
        for ent in g._entities:
            mat = ent.material
            if mat is None or isinstance(mat, str) or ent.dim != 3:
                continue
            if id(ent) in pml_volume_ids:
                continue
            mat_id = id(mat)
            if mat_id in seen_materials:
                continue
            seen_materials.add(mat_id)
            tag = g._material_tags.get(mat_id)
            if tag is None:
                raise RuntimeError(
                    f"material {mat!r} has no tag, re-run g.mesh() after attaching it")
            parts.append(mat._to_toml(tag))

        # Physics, ports, BCs, PML. PEC tags get aggregated separately;
        # a FarFieldSurface tag is consumed by the [output] block.
        pec_tags: list[int] = []
        nfft_tag: int | None = None
        for phys in g._physics:
            tag = g._physics_tags.get(id(phys))
            if tag is None:
                raise RuntimeError(
                    f"physics object {phys!r} has no tag, re-run g.mesh() "
                    f"after constructing it")
            if isinstance(phys, PEC):
                pec_tags.append(tag)
            elif isinstance(phys, FarFieldSurface):
                if nfft_tag is not None:
                    raise RuntimeError(
                        "multiple FarFieldSurface objects, but only one "
                        "near-field-to-far-field surface is supported. Pass "
                        "every face to a single FarFieldSurface(...) call "
                        "(e.g. rf.FarFieldSurface(*air.faces.hull)).")
                nfft_tag = tag
            else:
                block = phys._to_toml(tag)
                if block:
                    parts.append(block)

        if pec_tags:
            tags_str = ", ".join(str(t) for t in pec_tags)
            parts.append(f"[pec]\ntags = [{tags_str}]\n")
        else:
            parts.append("[pec]\ntags = []\n")

        if eigenmode is not None:
            f0, nm = eigenmode
            parts.append(f"[eigenmode]\ntarget_frequency = {_f64(f0)}\nn_modes = {nm}\n")

        if adaptive is not None:
            parts.append(
                f"[adaptive]\ntheta = {_f64(adaptive.theta)}\n"
                f"refinement_ratio = {_f64(adaptive.refinement_ratio)}\n"
            )

        output = f"[output]\nz0 = {_f64(z0)}\n"
        if nfft_tag is not None:
            output += f"nfft_tag = {nfft_tag}\n"
        parts.append(output)
        return "\n".join(parts)


__all__ = ["ProblemFD", "Adaptive", "ErrorIndicator"]
