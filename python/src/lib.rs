// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! PyO3 bindings for rapidfem: the typed `Model`, the frequency-domain
//! `Simulation` with its results, and the time-domain `TdSession`.
//!
//! Build via `maturin develop` (dev) or `maturin build --release` (wheel).

mod geometry;
mod model;
mod rfic;
mod td;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rapidfem_fd::eigenmode::Eigenmode;
use rapidfem_fd::error_estimator::ErrorEstimate;
use rapidfem_fd::farfield::RadiationPattern;
use rapidfem_fd::network::{renormalize, TouchstoneFormat};
use rapidfem_fd::order::OrderPolicy;
use rapidfem_fd::simulation::{abc_phasor, FdSettings, Simulation, SweepResult};
use geometry::{PyFemMesh, PyGeometry};
use model::PyModel;

/// A frequency-sweep simulation. Build once, run sweeps, inspect results.
///
/// Marked `unsendable`: `Box<dyn Port>` doesn't auto-impl Send/Sync, so a Simulation
/// instance must stay on the thread that created it. Fine for typical Python use.
#[pyclass(name = "Simulation", unsendable)]
struct PySimulation {
    inner: Simulation,
}

/// Result of a frequency sweep, frequencies and S-parameters.
#[pyclass(name = "SweepResult")]
struct PySweepResult {
    inner: SweepResult,
}

/// One eigenmode of a cavity / waveguide.
#[pyclass(name = "Eigenmode", unsendable)]
struct PyEigenmode {
    inner: Eigenmode,
}

/// Far-field radiation pattern with directivity, gain, axial ratio, LCP/RCP.
#[pyclass(name = "RadiationPattern", unsendable)]
struct PyRadiationPattern {
    inner: RadiationPattern,
}

/// Per-tetrahedron residual error indicator of one ``(frequency, port)``
/// solution, from :meth:`ProblemFD.element_errors`.
///
/// Holds the Monk-style a-posteriori eta values plus the Doerfler-marked
/// subset an AMR loop would refine. Diagnostic: it does not re-mesh or
/// re-solve; refine where it points with the static size controls
/// (``maxh``, :meth:`Geometry.refine_near_points`).
///
/// Attributes
/// ----------
/// eta : np.ndarray
///     per-tet error indicator, shape ``(n_tets,)``, float64
/// total : float
///     global L2 error ``sqrt(sum_K eta_K^2)``
/// marked : np.ndarray
///     int64 tet indices selected by Doerfler marking at ``theta``
/// volume_residuals : np.ndarray
///     volume-residual contribution per tet, shape ``(n_tets,)``
/// face_jumps : np.ndarray
///     face-jump contribution per tet (accumulated over its 4 faces)
/// h_k : np.ndarray
///     per-tet element diameter (max edge length) in m, useful for choosing
///     a refinement target relative to the current local size
/// tet_centroids : np.ndarray
///     per-tet centroid coordinates, shape ``(n_tets, 3)``, m
/// freq_hz : float
///     frequency at which the indicator was computed
/// theta : float
///     Doerfler fraction used for marking
#[pyclass(name = "ErrorIndicator", module = "rapidfem", frozen)]
struct PyErrorIndicator {
    est: ErrorEstimate,
    /// `est.h_k` in metres.
    h_k: Vec<f64>,
    centroids: Vec<[f64; 3]>,
    freq_hz: f64,
    theta: f64,
}

/// The element order of a sweep: 1, 2 or "adaptive".
#[derive(FromPyObject)]
enum OrderArg {
    Uniform(i64),
    Named(String),
}

/// A reference impedance: one per port, or one for every port.
#[derive(FromPyObject)]
enum ZRef {
    Each(Vec<f64>),
    One(f64),
}

impl ZRef {
    fn values(self) -> Vec<f64> {
        match self {
            ZRef::Each(z) => z,
            ZRef::One(z) => vec![z],
        }
    }
}

#[pymethods]
impl PySimulation {
    /// Build a simulation on a solver mesh (`Geometry.fem_mesh`) and a
    /// `Model`.
    ///
    /// `order` is the uniform element order (1 or 2, default 2) or
    /// "adaptive", the wavelength order policy. `eigenmode` is
    /// `(target_hz, n_modes)`. `adaptive_tol` turns on the adaptive sweep
    /// (full solves at a few frequencies, the rest from a reduced model) with
    /// at most `adaptive_max_samples` full solves per driven port, converged
    /// after `adaptive_memory` samples in a row within the tolerance.
    #[new]
    #[pyo3(signature = (mesh, model, frequencies, *, order=None, eigenmode=None,
                        adaptive_tol=None, adaptive_max_samples=20, adaptive_memory=2))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mesh: &PyFemMesh,
        model: &PyModel,
        frequencies: Vec<f64>,
        order: Option<OrderArg>,
        eigenmode: Option<(f64, usize)>,
        adaptive_tol: Option<f64>,
        adaptive_max_samples: usize,
        adaptive_memory: usize,
    ) -> PyResult<Self> {
        if frequencies.is_empty() {
            return Err(PyValueError::new_err("sweep needs at least one frequency"));
        }
        let order = match order {
            None | Some(OrderArg::Uniform(2)) => OrderPolicy::Uniform(2),
            Some(OrderArg::Uniform(1)) => OrderPolicy::Uniform(1),
            Some(OrderArg::Named(name)) if name == "adaptive" => {
                OrderPolicy::Adaptive { theta: rapidfem_fd::order::DEFAULT_THETA }
            }
            Some(OrderArg::Uniform(p)) => {
                return Err(PyValueError::new_err(format!("order must be 1, 2 or 'adaptive', got {p}")));
            }
            Some(OrderArg::Named(name)) => {
                return Err(PyValueError::new_err(format!("order must be 1, 2 or 'adaptive', got {name:?}")));
            }
        };
        let adaptive = adaptive_tol.map(|tol| rapidfem_fd::adaptive::AdaptiveSettings {
            tol,
            max_samples: adaptive_max_samples,
            memory: adaptive_memory,
        });
        let settings = FdSettings { frequencies, order, eigenmode, adaptive };
        let inner = Simulation::new(mesh.inner.clone(), model.inner.clone(), settings)
            .map_err(PyRuntimeError::new_err)?;
        Ok(PySimulation { inner })
    }

    /// Run the configured frequency sweep. Returns a SweepResult with frequencies
    /// (float64 array, shape `[n_freq]`) and S-parameters (complex128 array,
    /// shape `[n_freq, n_driven, n_driven]`).
    ///
    /// `callback`, if given, is called after each frequency's solve as
    /// `callback(freq_idx, freq_hz, s_matrix)` where `s_matrix` is a
    /// `(n_driven, n_driven)` complex128 numpy array. Used by the UI to stream
    /// partial results; it does not change the returned SweepResult.
    #[pyo3(signature = (callback=None))]
    fn run_sweep(&self, callback: Option<Py<PyAny>>) -> PyResult<PySweepResult> {
        // No callback: release the GIL for the (potentially long) sweep so
        // other Python threads run. `Python::allow_threads` wants `Send`, but
        // Simulation is `unsendable` (Box<dyn Port> is not Send), so we drop to
        // PyO3's ffi for the same effect without the Send bound.
        let Some(cb) = callback else {
            let inner = unsafe {
                let save = pyo3::ffi::PyEval_SaveThread();
                let r = self.inner.run_sweep(None);
                pyo3::ffi::PyEval_RestoreThread(save);
                r
            }
            .map_err(PyRuntimeError::new_err)?;
            return Ok(PySweepResult { inner });
        };

        // With a callback we must call back into Python per frequency, so keep
        // the GIL held for the whole sweep (mixing PyEval_SaveThread with
        // re-acquiring the GIL on the same thread is unsound). The hook
        // re-enters via `Python::attach`, which is cheap when the GIL is
        // already held. A genuine exception from the callback (anything other
        // than KeyboardInterrupt) is stashed here, stops the sweep, and is
        // re-raised after run_sweep returns rather than silently swallowed.
        let cb_err: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
        let hook = |fi: usize, freq: f64, s: &[Vec<Complex64>]| -> bool {
            Python::attach(|py| {
                let arr = grid2(s, py);
                match cb.call1(py, (fi, freq, arr)) {
                    Ok(_) => {}
                    Err(e) => {
                        // A KeyboardInterrupt through the callback means "stop"
                        // (no error); any other exception is a real callback
                        // bug, so stash it and stop the sweep.
                        if !e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                            *cb_err.borrow_mut() = Some(e);
                        }
                        return false;
                    }
                }
                // Also honour a Ctrl-C / UI interrupt that landed between
                // callbacks (check_signals clears it on Err).
                py.check_signals().is_ok()
            })
        };
        let hook_dyn: &dyn Fn(usize, f64, &[Vec<Complex64>]) -> bool = &hook;
        let r = self.inner.run_sweep(Some(hook_dyn));
        // Surface a stashed callback exception ahead of any solver error.
        if let Some(e) = cb_err.borrow_mut().take() {
            return Err(e);
        }
        let inner = r.map_err(PyRuntimeError::new_err)?;
        Ok(PySweepResult { inner })
    }

    /// Number of tetrahedra in the mesh.
    #[getter]
    fn n_tets(&self) -> usize { self.inner.mesh.n_tets() }

    /// Number of degrees of freedom in the FEM basis.
    #[getter]
    fn n_dofs(&self) -> usize { self.inner.basis.n_field }

    /// Run an eigenmode analysis. Requires the simulation to be built with `eigenmode=`.
    /// Returns a list of `Eigenmode` (frequency, Q, field).
    fn run_eigenmode(&self) -> PyResult<Vec<PyEigenmode>> {
        if self.inner.settings.eigenmode.is_none() {
            return Err(PyRuntimeError::new_err(
                "no eigenmode target: build the Simulation with eigenmode=(f, n)",
            ));
        }
        Ok(self
            .inner
            .run_eigenmode()
            .map_err(PyRuntimeError::new_err)?
            .into_iter()
            .map(|m| PyEigenmode { inner: m })
            .collect())
    }

    /// Mesh node coordinates as a `(n_nodes, 3)` float64 numpy array.
    #[getter]
    fn mesh_nodes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.mesh.n_nodes();
        // The mesh is stored in L₀ (characteristic-length) units after lever ④;
        // report physical coordinates by multiplying back.
        let l0 = self.inner.mesh.l0;
        let mut flat: Vec<f64> = Vec::with_capacity(n * 3);
        for p in &self.inner.mesh.nodes {
            flat.extend_from_slice(&[p[0] * l0, p[1] * l0, p[2] * l0]);
        }
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), flat).expect("shape");
        arr.into_pyarray(py)
    }

    /// Mesh tetrahedra as a `(n_tets, 4)` int64 numpy array of node indices.
    #[getter]
    fn mesh_tets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let n = self.inner.mesh.n_tets();
        let mut flat: Vec<i64> = Vec::with_capacity(n * 4);
        for tet in &self.inner.mesh.tets {
            for &v in tet {
                flat.push(v as i64);
            }
        }
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 4), flat).expect("shape");
        arr.into_pyarray(py)
    }

    /// FEM E-field interpolated at every mesh node for a given (freq_idx, port_idx).
    /// Returns a `(n_nodes, 3)` complex128 numpy array (Ex, Ey, Ez per node).
    /// Use this with `pyvista` or any mesh-viz library for field visualization.
    fn field_at_nodes<'py>(
        &self,
        py: Python<'py>,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
    ) -> Option<Bound<'py, PyArray2<Complex64>>> {
        Some(per_node(self.inner.field_at_nodes(&result.inner, freq_idx, port_idx)?, py))
    }

    /// Loss-equivalent current density J = σ_eff · E at every mesh node for
    /// a given (freq_idx, port_idx). `σ_eff = ω·ε₀·εᵣ·tan(δ) + σ_bulk`
    /// covers both dielectric (loss tangent) and Ohmic losses, so substrates
    /// like Rogers with tan_δ but zero bulk σ also light up. Returns a
    /// `(n_nodes, 3)` complex128 numpy array (Jx, Jy, Jz per node) in A/m².
    fn current_density_at_nodes<'py>(
        &self,
        py: Python<'py>,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
    ) -> Option<Bound<'py, PyArray2<Complex64>>> {
        Some(per_node(self.inner.current_density_at_nodes(&result.inner, freq_idx, port_idx)?, py))
    }

    /// Magnetic field H = ∇×E / (jωμ₀μ_r) at every mesh node for a given
    /// (freq_idx, port_idx). Returns a `(n_nodes, 3)` complex128 numpy array
    /// (Hx, Hy, Hz per node) in A/m. Derived from the analytic Nédélec-2
    /// curl of the FEM solution.
    fn h_field_at_nodes<'py>(
        &self,
        py: Python<'py>,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
    ) -> Option<Bound<'py, PyArray2<Complex64>>> {
        Some(per_node(self.inner.h_field_at_nodes(&result.inner, freq_idx, port_idx)?, py))
    }

    /// Same as ``field_at_nodes`` but for an :class:`Eigenmode`. Returns a
    /// `(n_nodes, 3)` complex128 numpy array of (Ex, Ey, Ez) at each mesh
    /// node. Field magnitude is not normalised, eigenmodes are defined up
    /// to a global scale.
    fn mode_field_at_nodes<'py>(
        &self,
        py: Python<'py>,
        mode: &PyEigenmode,
    ) -> Option<Bound<'py, PyArray2<Complex64>>> {
        Some(per_node(self.inner.eigenmode_field_at_nodes(&mode.inner)?, py))
    }

    /// The field of `channel` ("E", "J" or "H") at every node at
    /// ``(freq_idx, port_idx)`` in the viewer's phasor form, flat float32
    /// ``[A, B, C]`` per node (see `abc_phasor`); None where the solver
    /// derives no such field.
    #[pyo3(signature = (result, freq_idx, port_idx, channel="E"))]
    fn field_abc<'py>(
        &self,
        py: Python<'py>,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
        channel: &str,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let (r, s) = (&result.inner, &self.inner);
        let field = match channel {
            "E" | "e" => s.field_at_nodes(r, freq_idx, port_idx),
            "J" | "j" => s.current_density_at_nodes(r, freq_idx, port_idx),
            "H" | "h" => s.h_field_at_nodes(r, freq_idx, port_idx),
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("channel must be E, J or H, got {channel:?}"))),
        };
        Ok(field.map(|f| abc_phasor(&f).into_pyarray(py)))
    }

    /// An eigenmode's E field in the viewer's phasor form (see `field_abc`).
    fn mode_field_abc<'py>(&self, py: Python<'py>, mode: &PyEigenmode) -> Option<Bound<'py, PyArray1<f32>>> {
        Some(abc_phasor(&self.inner.eigenmode_field_at_nodes(&mode.inner)?).into_pyarray(py))
    }

    /// Monk-style residual error indicator per tetrahedron at
    /// ``(freq_idx, port_idx)``, Doerfler-marked at fraction ``theta``, as
    /// an :class:`ErrorIndicator`. Raises ``IndexError`` without a solution
    /// at those indices. Diagnostic only, does not re-mesh.
    #[pyo3(signature = (result, freq_idx=0, port_idx=0, theta=0.5))]
    fn element_errors(
        &self,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
        theta: f64,
    ) -> PyResult<PyErrorIndicator> {
        let est = self
            .inner
            .element_errors_at(&result.inner, freq_idx, port_idx, theta)
            .ok_or_else(|| {
                PyIndexError::new_err(format!("no solution for (freq_idx={freq_idx}, port_idx={port_idx})"))
            })?;
        // h_k lives in mesh-internal units (l0-normalised); every other
        // length crossing this boundary (mesh_nodes, refine_near_points)
        // is in meters, so convert here.
        let l0 = self.inner.mesh.l0;
        let h_k = est.h_k.iter().map(|&h| h * l0).collect();
        Ok(PyErrorIndicator {
            est,
            h_k,
            centroids: self.inner.tet_centroids(),
            freq_hz: result.inner.frequencies[freq_idx],
            theta,
        })
    }

    /// Compute the far-field radiation pattern at (freq_idx, port_idx) on a (theta, phi) grid.
    /// Returns None if the NFFT surface is empty or out-of-bounds indices.
    #[pyo3(signature = (result, freq_idx=0, port_idx=0, n_theta=91, n_phi=72))]
    fn compute_farfield(
        &self,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
        n_theta: usize,
        n_phi: usize,
    ) -> Option<PyRadiationPattern> {
        self.inner
            .compute_farfield(&result.inner, freq_idx, port_idx, n_theta, n_phi)
            .map(|p| PyRadiationPattern { inner: p })
    }
}

#[pymethods]
impl PySweepResult {
    /// Frequencies in Hz, shape `[n_freq]`, dtype float64.
    #[getter]
    fn frequencies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.frequencies.clone().into_pyarray(py)
    }

    /// S-parameter matrix, shape `[n_freq, n_driven, n_driven]`, dtype complex128.
    /// Indexing: `S[freq_idx, observation_port, excitation_port]`.
    #[getter]
    fn sparams<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<Complex64>> {
        grid3(&self.inner.sparams, self.inner.n_driven, py)
    }

    /// Reference impedance (ohm) of each driven port per frequency, shape
    /// `[n_freq, n_driven]`: the mode impedance of a modal port (waveguide /
    /// wave), the fixed `z0` of a lumped one. `sparams` are referenced to
    /// these; `renormalize` re-references them.
    #[getter]
    fn port_impedances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        grid2(&self.inner.port_impedances, py)
    }

    /// The S-parameters renormalized from `port_impedances` to the fixed
    /// reference `z_ref` (one impedance, or one per driven port), shape
    /// `[n_freq, n_driven, n_driven]`. A matched modal line reads |S11| ~ 0
    /// against its own mode impedance; against e.g. 50 ohm its mismatch
    /// appears. Lumped ports already at `z_ref` are a no-op.
    #[pyo3(signature = (z_ref=None))]
    fn renormalize<'py>(&self, py: Python<'py>, z_ref: Option<ZRef>) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
        let z = z_ref.map_or_else(|| vec![50.0], ZRef::values);
        let s = self.inner.renormalized(&z).map_err(PyValueError::new_err)?;
        Ok(grid3(&s, self.inner.n_driven, py))
    }

    /// Writes the S-parameters to a Touchstone 1.0 file (.s1p, .s2p,
    /// .snp) at `path`, the option line carrying reference `z0`, in format
    /// `fmt`: "ri" (real / imaginary), "ma" (magnitude / degrees) or "db"
    /// (dB / degrees). The values are written as solved, each port
    /// referenced to its `port_impedances`.
    #[pyo3(signature = (path, z0=50.0, fmt="ri"))]
    fn to_touchstone(&self, path: std::path::PathBuf, z0: f64, fmt: &str) -> PyResult<()> {
        let format: TouchstoneFormat = fmt.parse().map_err(PyValueError::new_err)?;
        self.inner.write_touchstone(&path, z0, format)?;
        Ok(())
    }

    /// Number of driven ports (S-matrix dimension).
    #[getter]
    fn n_driven(&self) -> usize { self.inner.n_driven }

    /// Total wall-clock for the sweep in seconds.
    #[getter]
    fn solve_time_s(&self) -> f64 { self.inner.solve_time_s }

    /// The frequencies solved in full (Hz): every frequency of a full sweep,
    /// the samples of an adaptive one, in the order they were solved.
    #[getter]
    fn full_solve_frequencies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.full_solves.clone().into_pyarray(py)
    }
}

#[pymethods]
impl PyErrorIndicator {
    /// Per-tet error indicator eta, shape `(n_tets,)`.
    #[getter]
    fn eta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.est.element_errors.clone().into_pyarray(py)
    }

    /// Global L2 error `sqrt(sum eta^2)`.
    #[getter]
    fn total(&self) -> f64 { self.est.total_error }

    /// Doerfler-marked tet indices (int64).
    #[getter]
    fn marked<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        let marked: Vec<i64> = self.est.marked_elements.iter().map(|&i| i as i64).collect();
        marked.into_pyarray(py)
    }

    /// Volume-residual contribution per tet.
    #[getter]
    fn volume_residuals<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.est.volume_residuals.clone().into_pyarray(py)
    }

    /// Face-jump contribution per tet.
    #[getter]
    fn face_jumps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.est.face_jumps.clone().into_pyarray(py)
    }

    /// Per-tet element diameter (max edge length) in m.
    #[getter]
    fn h_k<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.h_k.clone().into_pyarray(py)
    }

    /// Per-tet centroid, shape `(n_tets, 3)`, in m.
    #[getter]
    fn tet_centroids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let flat: Vec<f64> = self.centroids.iter().flatten().copied().collect();
        numpy::ndarray::Array2::from_shape_vec((self.centroids.len(), 3), flat).expect("shape").into_pyarray(py)
    }

    /// Frequency of the indicated solution (Hz).
    #[getter]
    fn freq_hz(&self) -> f64 { self.freq_hz }

    /// Doerfler fraction used for marking.
    #[getter]
    fn theta(&self) -> f64 { self.theta }

    fn __repr__(&self) -> String {
        let n = self.est.element_errors.len();
        let m = self.est.marked_elements.len();
        format!(
            "ErrorIndicator(n_tets={n}, total={:.4e}, marked={m} ({:.1}%), freq={:.3} GHz)",
            self.est.total_error,
            100.0 * m as f64 / n.max(1) as f64,
            self.freq_hz / 1e9
        )
    }
}

#[pymethods]
impl PyEigenmode {
    /// Real part of the resonant frequency (Hz).
    #[getter]
    fn frequency_hz(&self) -> f64 { self.inner.frequency.re }

    /// Imaginary part of the resonant frequency (Hz). Non-zero for lossy / leaky modes.
    #[getter]
    fn frequency_imag_hz(&self) -> f64 { self.inner.frequency.im }

    /// Quality factor Q = f_re / (2 * f_im). Infinite for lossless modes.
    #[getter]
    fn q_factor(&self) -> f64 { self.inner.q_factor }

    /// E-field DOF coefficient vector for this mode (complex128).
    #[getter]
    fn field<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        self.inner.field.clone().into_pyarray(py)
    }
}

#[pymethods]
impl PyRadiationPattern {
    /// Theta angles (radians, 0..pi).
    #[getter]
    fn theta_rad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.theta.clone().into_pyarray(py)
    }

    /// Phi angles (radians, 0..2pi).
    #[getter]
    fn phi_rad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.phi.clone().into_pyarray(py)
    }

    /// Directivity D(theta, phi) in dBi, shape `[n_phi, n_theta]`, float64.
    #[getter]
    fn directivity_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        grid2(&self.inner.directivity_dbi, py)
    }

    /// Realised gain G(theta, phi) in dBi (directivity minus mismatch/loss),
    /// shape `[n_phi, n_theta]`, float64.
    #[getter]
    fn gain_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        grid2(&self.inner.gain_dbi, py)
    }

    /// Axial ratio in dB (0 = circular, large = linear polarisation),
    /// shape `[n_phi, n_theta]`, float64.
    #[getter]
    fn axial_ratio_db<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        grid2(&self.inner.axial_ratio_db, py)
    }

    /// Left-hand circular polarisation gain in dBi, shape `[n_phi, n_theta]`,
    /// float64.
    #[getter]
    fn lcp_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        grid2(&self.inner.lcp_dbi, py)
    }

    /// Right-hand circular polarisation gain in dBi, shape `[n_phi, n_theta]`,
    /// float64.
    #[getter]
    fn rcp_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        grid2(&self.inner.rcp_dbi, py)
    }

    /// Complex E_theta(theta, phi), shape `[n_phi, n_theta]`.
    #[getter]
    fn e_theta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<Complex64>> {
        grid2(&self.inner.e_theta, py)
    }

    /// Complex E_phi(theta, phi), shape `[n_phi, n_theta]`.
    #[getter]
    fn e_phi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<Complex64>> {
        grid2(&self.inner.e_phi, py)
    }

    /// Peak directivity over the sampled sphere, in dBi.
    #[getter]
    fn peak_directivity_dbi(&self) -> f64 { self.inner.peak_directivity_dbi }

    /// Peak realised gain over the sampled sphere, in dBi.
    #[getter]
    fn peak_gain_dbi(&self) -> f64 { self.inner.peak_gain_dbi }

    /// Total radiated power integrated over the sphere, in watts.
    #[getter]
    fn radiated_power(&self) -> f64 { self.inner.radiated_power }
}

/// A row-major grid (`[n_rows][n_cols]`, rows of equal length) as a 2D array.
fn grid2<'py, T: numpy::Element + Copy>(grid: &[Vec<T>], py: Python<'py>) -> Bound<'py, PyArray2<T>> {
    let n_cols = grid.first().map_or(0, |r| r.len());
    let flat: Vec<T> = grid.iter().flatten().copied().collect();
    numpy::ndarray::Array2::from_shape_vec((grid.len(), n_cols), flat).expect("shape").into_pyarray(py)
}

/// `[freq][n][n]` S-matrices as an `(n_freq, n, n)` array.
fn grid3<'py>(s: &[Vec<Vec<Complex64>>], n: usize, py: Python<'py>) -> Bound<'py, PyArray3<Complex64>> {
    let flat: Vec<Complex64> = s.iter().flatten().flatten().copied().collect();
    numpy::ndarray::Array3::from_shape_vec((s.len(), n, n), flat).expect("square matrices").into_pyarray(py)
}

/// A flat `[x, y, z]`-per-node vector as an `(n_nodes, 3)` array.
fn per_node<'py>(flat: Vec<Complex64>, py: Python<'py>) -> Bound<'py, PyArray2<Complex64>> {
    let n = flat.len() / 3;
    numpy::ndarray::Array2::from_shape_vec((n, 3), flat).expect("shape").into_pyarray(py)
}

/// Renormalize S-parameters `sparams` `(n_freq, n, n)`, referenced to the
/// per-frequency, per-port impedances `z_old` `(n_freq, n)`, to `z_new`
/// (one impedance, or one per port). See `SweepResult.renormalize`.
#[pyfunction]
fn renormalize_sparams<'py>(
    py: Python<'py>,
    sparams: PyReadonlyArray3<'py, Complex64>,
    z_old: PyReadonlyArray2<'py, f64>,
    z_new: ZRef,
) -> PyResult<Bound<'py, PyArray3<Complex64>>> {
    let s = sparams.as_array();
    let (n_freq, n, m) = s.dim();
    if n != m {
        return Err(PyValueError::new_err(format!("sparams must be (n_freq, n, n), got ({n_freq}, {n}, {m})")));
    }
    let s: Vec<Vec<Vec<Complex64>>> =
        (0..n_freq).map(|f| (0..n).map(|i| (0..n).map(|j| s[[f, i, j]]).collect()).collect()).collect();
    let z_old: Vec<Vec<f64>> = z_old.as_array().outer_iter().map(|row| row.to_vec()).collect();
    let out = renormalize(&s, &z_old, &z_new.values()).map_err(PyValueError::new_err)?;
    Ok(grid3(&out, n, py))
}

/// rapidfem, frequency- and time-domain EM FEM solver.
#[pymodule]
#[pyo3(name = "_native")]
fn rapidfem_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyFemMesh>()?;
    m.add_class::<geometry::PyMeshStats>()?;
    m.add_class::<PySimulation>()?;
    m.add_class::<PySweepResult>()?;
    m.add_class::<PyEigenmode>()?;
    m.add_class::<PyRadiationPattern>()?;
    m.add_class::<PyErrorIndicator>()?;
    m.add_class::<td::PyTdSession>()?;
    m.add_class::<td::PyGaussianPulse>()?;
    m.add_function(wrap_pyfunction!(renormalize_sparams, m)?)?;
    m.add_function(wrap_pyfunction!(rfic::rfic_build, m)?)?;
    m.add_function(wrap_pyfunction!(rfic::rfic_from_gds, m)?)?;
    m.add_function(wrap_pyfunction!(rfic::rfic_from_fem_json, m)?)?;
    rapidfem_geom::rfic::python::register(m)?;
    Ok(())
}
