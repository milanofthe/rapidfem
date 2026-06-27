// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! PyO3 bindings for rapidfem. Exposes `Simulation`, `SweepResult` to Python.
//!
//! Build via `maturin develop` (dev) or `maturin build --release` (wheel).
//! See `examples/wr90.py` for usage.

use num_complex::Complex64;
use numpy::{
    Complex64 as NpC64, IntoPyArray, PyArray1, PyArray2, PyArray3,
    PyReadonlyArray1, PyReadwriteArray1,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rapidfem_fd::eigenmode::Eigenmode;
use rapidfem_fd::farfield::RadiationPattern;
use rapidfem_fd::simulation::{Simulation, SweepResult};

/// A frequency-sweep simulation. Build once, run sweeps, inspect results.
///
/// Marked `unsendable`: `Box<dyn Port>` doesn't auto-impl Send/Sync, so a Simulation
/// instance must stay on the thread that created it. Fine for typical Python use.
#[pyclass(name = "Simulation", unsendable)]
struct PySimulation {
    inner: Simulation,
}

/// Result of a frequency sweep — frequencies and S-parameters.
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

#[pymethods]
impl PySimulation {
    /// Construct a simulation by loading a gmsh `.msh` file and a TOML config from disk.
    #[staticmethod]
    fn from_files(mesh_path: &str, config_path: &str) -> PyResult<Self> {
        let config = rapidfem_fd::config::load_config(config_path)
            .map_err(|e| PyRuntimeError::new_err(format!("config: {}", e)))?;
        let mesh = rapidfem_fd::mesh_io::load_mesh(mesh_path)
            .map_err(|e| PyRuntimeError::new_err(format!("mesh: {}", e)))?;
        Ok(PySimulation { inner: Simulation::new(mesh, config) })
    }

    /// Construct from in-memory mesh bytes and a TOML config string.
    /// Useful when meshes/configs are produced programmatically (no disk I/O).
    #[staticmethod]
    fn from_bytes(mesh_bytes: &[u8], config_toml: &str) -> PyResult<Self> {
        let inner = Simulation::from_bytes(mesh_bytes, config_toml)
            .map_err(|e| PyRuntimeError::new_err(e))?;
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
    fn run_sweep(&self, callback: Option<PyObject>) -> PyResult<PySweepResult> {
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
        // re-enters via `Python::with_gil`, which is cheap when the GIL is
        // already held. A genuine exception from the callback (anything other
        // than KeyboardInterrupt) is stashed here, stops the sweep, and is
        // re-raised after run_sweep returns rather than silently swallowed.
        let cb_err: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
        let hook = |fi: usize, freq: f64, s: &[Vec<Complex64>]| -> bool {
            Python::with_gil(|py| {
                let n = s.len();
                let mut flat: Vec<NpC64> = Vec::with_capacity(n * n);
                for row in s {
                    for v in row {
                        flat.push(NpC64::new(v.re, v.im));
                    }
                }
                let arr = numpy::ndarray::Array2::from_shape_vec((n, n), flat)
                    .expect("square s-matrix")
                    .into_pyarray_bound(py);
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
                !py.check_signals().is_err()
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

    /// Number of driven ports (i.e., ports with excitation: rect waveguide, lumped, coax, ...).
    #[getter]
    fn n_driven_ports(&self) -> usize {
        self.inner.ports.iter().filter(|p| p.is_driven()).count()
    }

    /// Modal characteristic impedance (Ω) of the `driven_idx`-th DRIVEN port at
    /// `freq_hz`. The index matches the S-matrix column order (driven ports
    /// only). Modal ports (waveguide / wave) return their frequency-dependent
    /// modal impedance; lumped ports return their fixed reference z0. Used by
    /// Python to renormalize the modal-referenced S-parameters to a fixed
    /// reference impedance. Returns 0.0 if the index is out of range.
    fn port_z_mode(&self, driven_idx: usize, freq_hz: f64) -> f64 {
        let exc = rapidfem_fd::excitation::Excitation::new(freq_hz, self.inner.mesh.l0);
        self.inner
            .ports
            .iter()
            .filter(|p| p.is_driven())
            .nth(driven_idx)
            .map(|p| p.z_mode(&exc))
            .unwrap_or(0.0)
    }

    /// Run an eigenmode analysis. Requires `[eigenmode]` block in the TOML config.
    /// Returns a list of `Eigenmode` (frequency, Q, field).
    fn run_eigenmode(&self) -> PyResult<Vec<PyEigenmode>> {
        if self.inner.config.eigenmode.is_none() {
            return Err(PyRuntimeError::new_err(
                "config.eigenmode block not set in TOML",
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
        arr.into_pyarray_bound(py)
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
        arr.into_pyarray_bound(py)
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
    ) -> Option<Bound<'py, PyArray2<NpC64>>> {
        let flat = self.inner.field_at_nodes(&result.inner, freq_idx, port_idx)?;
        let n = self.inner.mesh.n_nodes();
        let conv: Vec<NpC64> = flat.iter().map(|c| NpC64::new(c.re, c.im)).collect();
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), conv).expect("shape");
        Some(arr.into_pyarray_bound(py))
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
    ) -> Option<Bound<'py, PyArray2<NpC64>>> {
        let flat = self.inner.current_density_at_nodes(&result.inner, freq_idx, port_idx)?;
        let n = self.inner.mesh.n_nodes();
        let conv: Vec<NpC64> = flat.iter().map(|c| NpC64::new(c.re, c.im)).collect();
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), conv).expect("shape");
        Some(arr.into_pyarray_bound(py))
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
    ) -> Option<Bound<'py, PyArray2<NpC64>>> {
        let flat = self.inner.h_field_at_nodes(&result.inner, freq_idx, port_idx)?;
        let n = self.inner.mesh.n_nodes();
        let conv: Vec<NpC64> = flat.iter().map(|c| NpC64::new(c.re, c.im)).collect();
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), conv).expect("shape");
        Some(arr.into_pyarray_bound(py))
    }

    /// Same as ``field_at_nodes`` but for an :class:`Eigenmode`. Returns a
    /// `(n_nodes, 3)` complex128 numpy array of (Ex, Ey, Ez) at each mesh
    /// node. Field magnitude is not normalised — eigenmodes are defined up
    /// to a global scale.
    fn mode_field_at_nodes<'py>(
        &self,
        py: Python<'py>,
        mode: &PyEigenmode,
    ) -> Option<Bound<'py, PyArray2<NpC64>>> {
        let flat = self.inner.eigenmode_field_at_nodes(&mode.inner)?;
        let n = self.inner.mesh.n_nodes();
        let conv: Vec<NpC64> = flat.iter().map(|c| NpC64::new(c.re, c.im)).collect();
        let arr = numpy::ndarray::Array2::from_shape_vec((n, 3), conv).expect("shape");
        Some(arr.into_pyarray_bound(py))
    }

    /// Monk-style residual error indicator η per tetrahedron at
    /// ``(freq_idx, port_idx)``. Returns a dict ``{eta, total, marked,
    /// volume_residuals, face_jumps}`` with ``eta`` shape ``(n_tets,)``
    /// float64, ``marked`` an int64 array of Dörfler-selected tet
    /// indices at fraction ``theta``. Diagnostic only — does not
    /// re-mesh.
    #[pyo3(signature = (result, freq_idx=0, port_idx=0, theta=0.5))]
    fn element_errors<'py>(
        &self,
        py: Python<'py>,
        result: &PySweepResult,
        freq_idx: usize,
        port_idx: usize,
        theta: f64,
    ) -> Option<Bound<'py, pyo3::types::PyDict>> {
        let est = self.inner.element_errors_at(&result.inner, freq_idx, port_idx, theta)?;
        let dict = pyo3::types::PyDict::new_bound(py);
        let eta = est.element_errors.clone().into_pyarray_bound(py);
        let volr = est.volume_residuals.clone().into_pyarray_bound(py);
        let fj = est.face_jumps.clone().into_pyarray_bound(py);
        let h_k = est.h_k.clone().into_pyarray_bound(py);
        let marked: Vec<i64> = est.marked_elements.iter().map(|&i| i as i64).collect();
        let marked_arr = marked.into_pyarray_bound(py);
        dict.set_item("eta", eta).ok()?;
        dict.set_item("volume_residuals", volr).ok()?;
        dict.set_item("face_jumps", fj).ok()?;
        dict.set_item("h_k", h_k).ok()?;
        dict.set_item("total", est.total_error).ok()?;
        dict.set_item("marked", marked_arr).ok()?;
        Some(dict)
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
        self.inner.frequencies.clone().into_pyarray_bound(py)
    }

    /// S-parameter matrix, shape `[n_freq, n_driven, n_driven]`, dtype complex128.
    /// Indexing: `S[freq_idx, observation_port, excitation_port]`.
    #[getter]
    fn sparams<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<NpC64>> {
        let n_freq = self.inner.frequencies.len();
        let n = self.inner.n_driven;
        let mut flat: Vec<NpC64> = Vec::with_capacity(n_freq * n * n);
        for f_mat in &self.inner.sparams {
            for row in f_mat {
                for c in row {
                    // num_complex::Complex64 ↔ numpy::Complex64 are bit-identical layout.
                    let v: Complex64 = *c;
                    flat.push(NpC64::new(v.re, v.im));
                }
            }
        }
        let arr = numpy::ndarray::Array3::from_shape_vec((n_freq, n, n), flat)
            .expect("shape matches data");
        arr.into_pyarray_bound(py)
    }

    /// Number of driven ports (S-matrix dimension).
    #[getter]
    fn n_driven(&self) -> usize { self.inner.n_driven }

    /// Total wall-clock for the sweep in seconds.
    #[getter]
    fn solve_time_s(&self) -> f64 { self.inner.solve_time_s }
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
    fn field<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpC64>> {
        let conv: Vec<NpC64> = self.inner.field.iter().map(|c| NpC64::new(c.re, c.im)).collect();
        conv.into_pyarray_bound(py)
    }
}

#[pymethods]
impl PyRadiationPattern {
    /// Theta angles (radians, 0..pi).
    #[getter]
    fn theta_rad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.theta.clone().into_pyarray_bound(py)
    }

    /// Phi angles (radians, 0..2pi).
    #[getter]
    fn phi_rad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.phi.clone().into_pyarray_bound(py)
    }

    /// Directivity D(theta, phi) in dBi, shape `[n_phi, n_theta]`, float64.
    #[getter]
    fn directivity_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flatten_2d(&self.inner.directivity_dbi, py)
    }

    /// Realised gain G(theta, phi) in dBi (directivity minus mismatch/loss),
    /// shape `[n_phi, n_theta]`, float64.
    #[getter]
    fn gain_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flatten_2d(&self.inner.gain_dbi, py)
    }

    /// Axial ratio in dB (0 = circular, large = linear polarisation),
    /// shape `[n_phi, n_theta]`, float64.
    #[getter]
    fn axial_ratio_db<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flatten_2d(&self.inner.axial_ratio_db, py)
    }

    /// Left-hand circular polarisation gain in dBi, shape `[n_phi, n_theta]`,
    /// float64.
    #[getter]
    fn lcp_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flatten_2d(&self.inner.lcp_dbi, py)
    }

    /// Right-hand circular polarisation gain in dBi, shape `[n_phi, n_theta]`,
    /// float64.
    #[getter]
    fn rcp_dbi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flatten_2d(&self.inner.rcp_dbi, py)
    }

    /// Complex E_theta(theta, phi), shape `[n_phi, n_theta]`.
    #[getter]
    fn e_theta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpC64>> {
        flatten_2d_complex(&self.inner.e_theta, py)
    }

    /// Complex E_phi(theta, phi), shape `[n_phi, n_theta]`.
    #[getter]
    fn e_phi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpC64>> {
        flatten_2d_complex(&self.inner.e_phi, py)
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

fn flatten_2d<'py>(grid: &[Vec<f64>], py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
    let n_phi = grid.len();
    let n_theta = grid.first().map(|r| r.len()).unwrap_or(0);
    let mut flat: Vec<f64> = Vec::with_capacity(n_phi * n_theta);
    for row in grid {
        flat.extend_from_slice(row);
    }
    let arr = numpy::ndarray::Array2::from_shape_vec((n_phi, n_theta), flat).expect("shape");
    arr.into_pyarray_bound(py)
}

fn flatten_2d_complex<'py>(grid: &[Vec<Complex64>], py: Python<'py>) -> Bound<'py, PyArray2<NpC64>> {
    let n_phi = grid.len();
    let n_theta = grid.first().map(|r| r.len()).unwrap_or(0);
    let mut flat: Vec<NpC64> = Vec::with_capacity(n_phi * n_theta);
    for row in grid {
        for c in row {
            flat.push(NpC64::new(c.re, c.im));
        }
    }
    let arr = numpy::ndarray::Array2::from_shape_vec((n_phi, n_theta), flat).expect("shape");
    arr.into_pyarray_bound(py)
}

// --- Time-domain DGTD backend ----------------------------------------------

// The Krylov propagator tolerance lives with the other TD numerical
// constants — see `rapidfem_td::constants`.
use rapidfem_td::constants::KRYLOV_TOL;

/// Time-domain DGTD Maxwell operator (vacuum, PEC walls), built on a
/// structured box cavity. Wraps the Rust `MaxwellOperator`.
#[pyclass(name = "TdOperator")]
struct PyTdOperator {
    op: rapidfem_td::rhs::MaxwellOperator,
    /// Reused Krylov workspace — keeps `step` / `step_driven` allocation-free
    /// across a transient loop.
    krylov: rapidfem_td::propagator::KrylovWorkspace,
    /// Reused soft-source vector for `step_driven` — one DOF set per call.
    driven_b: Vec<f64>,
    /// Reused LSERK4 workspace — keeps the explicit `step_explicit` stepper
    /// allocation-free across a transient loop.
    lserk: rapidfem_td::explicit::LserkWorkspace,
    /// Reused KCL RK4(3)5[2R+]C workspace — keeps the adaptive `step_kcl`
    /// family allocation-free across the controller's accept/reject loop.
    kcl: rapidfem_td::explicit_adaptive::KclWorkspace,
    /// Lazily-built OpenCL GPU backend; `None` until first requested, and
    /// stays `None` if no GPU / OpenCL runtime is present.
    gpu: Option<GpuBackend>,
}

/// The GPU device context and the operator resident on it.
struct GpuBackend {
    ctx: rapidfem_td::gpu::GpuContext,
    op: rapidfem_td::gpu::GpuOperator,
}

impl PyTdOperator {
    /// Lazily build the GPU backend. Errors if there is no GPU / OpenCL
    /// runtime, or if the operator has dispersive materials (the GPU path
    /// covers the `[E,H]` block only).
    fn ensure_gpu(&mut self) -> Result<(), String> {
        if self.gpu.is_some() {
            return Ok(());
        }
        if self.op.n_dispersive() != 0 {
            return Err(
                "GPU path does not support dispersive materials"
                    .to_string(),
            );
        }
        let ctx = rapidfem_td::gpu::GpuContext::new()?;
        let op = rapidfem_td::gpu::GpuOperator::new(&ctx, &self.op)?;
        self.gpu = Some(GpuBackend { ctx, op });
        Ok(())
    }
}

#[pymethods]
impl PyTdOperator {
    /// Build the operator on a structured box cavity `[0,lx]×[0,ly]×[0,lz]`
    /// (`nx·ny·nz` cells) at polynomial `order`. `flux_alpha`: 0 = central
    /// (energy-conserving), 1 = upwind.
    #[new]
    #[pyo3(signature = (nx, ny, nz, lx, ly, lz, order, flux_alpha = 1.0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        lx: f64,
        ly: f64,
        lz: f64,
        order: usize,
        flux_alpha: f64,
    ) -> Self {
        let mesh = rapidfem_td::mesh_gen::structured_box(nx, ny, nz, lx, ly, lz);
        let op = rapidfem_td::rhs::MaxwellOperator::new(&mesh, order, flux_alpha);
        PyTdOperator {
            op,
            krylov: rapidfem_td::propagator::KrylovWorkspace::new(),
            driven_b: Vec::new(),
            lserk: rapidfem_td::explicit::LserkWorkspace::new(),
            kcl: rapidfem_td::explicit_adaptive::KclWorkspace::new(),
            gpu: None,
        }
    }

    /// Build the operator from in-memory gmsh `.msh` bytes — the path for
    /// arbitrary unstructured meshes produced by the geometry API.
    ///
    /// `tag_materials` maps a gmsh volume tag to `(eps_diag, mu_diag, sigma)`;
    /// tets in untagged volumes default to vacuum. `ports` maps a gmsh face
    /// tag to `(mode_m, mode_n, direction)` — each becomes a waveguide port,
    /// indexed in the given order. `direction` is `None` for a waveguide
    /// port, or a `(dx, dy, dz)` field axis for a lumped port.
    ///
    /// `absorbers` maps a gmsh volume tag to a graded impedance-matched
    /// absorbing layer — `(volume_tag, axis, inner_face, thickness, nu_max,
    /// is_low)`. `axis` is 0/1/2 for x/y/z; the loss ramps quadratically
    /// from zero at `inner_face` to `nu_max` at the layer's outer face,
    /// `thickness` away. `is_low` selects the low-coordinate end (the layer
    /// extends toward decreasing `axis`) versus the high-coordinate end.
    /// Each tet in the tagged volume keeps its `eps`/`mu` and gains the
    /// matched electric/magnetic loss `sigma = nu*eps`, `sigma_m = nu*mu`.
    /// Applied after `tag_materials`, so an absorber overrides a plain
    /// material assignment on the same volume.
    ///
    /// `dispersive` maps a gmsh volume tag to a Debye dispersive material —
    /// `(volume_tag, eps_inf, eps_static, tau)`. Tets in the tagged volume
    /// run the auxiliary-polarisation ADE: their non-dispersive permittivity
    /// is `eps_inf`, and an appended per-element polarisation block carries
    /// the relaxation `dP/dt = a*P + g*E`. With `dispersive` empty or None
    /// the operator is byte-identical to before — `n_dof = 6*Np*n_elem` and
    /// no polarisation state. Applied after `tag_materials` / `absorbers`,
    /// so a Debye material overrides their permittivity on the same volume.
    ///
    /// `coax_ports` declares coaxial TEM ports: `(face_tag, center)` per
    /// port, where `center` is `None` to use the face centroid or a
    /// `(cx, cy, cz)` triple to override the coax axis. Coax ports are
    /// appended to the operator's port list AFTER the rectangular `ports`,
    /// so their indices start at `len(ports)`.
    ///
    /// `periodic_pairs` declares normal-incidence periodic boundary
    /// pairs: `(face_tag_a, face_tag_b)` per pair. Each tag set is matched
    /// across the period translation (inferred from the two faces'
    /// centroids), and DG faces on either side then see the partner
    /// element across the period as their neighbour. The pair is
    /// unordered; the same triangle cannot also be tagged as a port.
    ///
    /// `floquet_ports` declares Floquet plane-wave ports for periodic
    /// unit-cell simulations: `(face_tag, polarisation_mode, scan_theta,
    /// scan_phi)` per port. `polarisation_mode` is `1` (TE / s-pol) or
    /// `2` (TM / p-pol), matching the FD `FloquetPort` convention; scan
    /// angles are radians. Floquet ports are appended to the operator's
    /// port list AFTER the rectangular `ports` and coax `coax_ports`, so
    /// their indices start at `len(ports) + len(coax_ports)`. The
    /// transverse Floquet phase factor is dropped at oblique scan (a
    /// real-valued port API approximation); normal incidence is exact.
    #[staticmethod]
    #[pyo3(signature = (mesh_bytes, order, flux_alpha = 1.0, tag_materials = None, ports = None, absorbers = None, dispersive = None, coax_ports = None, periodic_pairs = None, floquet_ports = None, abc_faces = None, pec_faces = None, wave_ports = None))]
    #[allow(clippy::too_many_arguments)]
    fn from_mesh_bytes(
        mesh_bytes: &[u8],
        order: usize,
        flux_alpha: f64,
        tag_materials: Option<
            Vec<(i32, (f64, f64, f64), (f64, f64, f64), f64)>,
        >,
        ports: Option<Vec<(i32, usize, usize, Option<(f64, f64, f64)>, f64)>>,
        absorbers: Option<Vec<(i32, usize, f64, f64, f64, bool)>>,
        dispersive: Option<Vec<(i32, f64, f64, f64)>>,
        coax_ports: Option<Vec<(i32, Option<(f64, f64, f64)>)>>,
        periodic_pairs: Option<Vec<(i32, i32)>>,
        floquet_ports: Option<Vec<(i32, u32, f64, f64)>>,
        abc_faces: Option<Vec<i32>>,
        pec_faces: Option<Vec<i32>>,
        wave_ports: Option<Vec<(i32, bool, usize, f64)>>,
    ) -> PyResult<Self> {
        use rapidfem_td::dispersive::DebyeMaterial;
        use rapidfem_td::rhs::{
            ElemMaterial, MaxwellOperator, PecSpec, PeriodicSpec, PortSpec,
        };
        use rapidfem_td::waveguide::FloquetPolarisation;
        let mesh = rapidfem_core::mesh_io::parse_mesh_bytes(mesh_bytes)
            .map_err(PyRuntimeError::new_err)?;
        let mut materials = vec![ElemMaterial::VACUUM; mesh.n_tets()];
        if let Some(tm) = tag_materials {
            for (tag, eps, mu, sigma) in tm {
                if let Some(tets) = mesh.vtag_to_tet.get(&tag) {
                    for &t in tets {
                        materials[t] = ElemMaterial {
                            eps: [eps.0, eps.1, eps.2],
                            mu: [mu.0, mu.1, mu.2],
                            sigma,
                            sigma_m: 0.0,
                        };
                    }
                }
            }
        }
        // Graded matched absorbers — per tet, depth into the layer sets a
        // quadratically ramped loss rate `nu`, mirroring `absorber.rs`. The
        // tet keeps its eps/mu; the matched pair `sigma = nu*eps`,
        // `sigma_m = nu*mu` keeps `sigma*/mu = sigma/eps = nu`, so the layer
        // is reflectionless at the interface.
        if let Some(abs_specs) = absorbers {
            for (tag, axis, inner_face, thickness, nu_max, is_low) in abs_specs
            {
                let tets = match mesh.vtag_to_tet.get(&tag) {
                    Some(t) => t,
                    None => continue,
                };
                for &t in tets {
                    let centroid: f64 = mesh.tets[t]
                        .iter()
                        .map(|&n| mesh.nodes[n][axis])
                        .sum::<f64>()
                        / 4.0;
                    let depth = if is_low {
                        inner_face - centroid
                    } else {
                        centroid - inner_face
                    };
                    if depth <= 0.0 {
                        continue;
                    }
                    let frac = (depth / thickness).clamp(0.0, 1.0);
                    let nu = nu_max * frac * frac;
                    let m = &mut materials[t];
                    m.sigma = nu * m.eps[0];
                    m.sigma_m = nu * m.mu[0];
                }
            }
        }
        // Debye dispersive volumes — applied after tag_materials / absorbers.
        // Each tagged tet's permittivity is forced to eps_inf (the static
        // curl term) and the tet is added to the ADE list; the appended
        // polarisation block then carries the dispersion. An empty list
        // leaves the operator byte-identical to the non-dispersive build.
        let mut disp_elems: Vec<(usize, DebyeMaterial)> = Vec::new();
        if let Some(dsp) = dispersive {
            for (tag, eps_inf, eps_static, tau) in dsp {
                let tets = match mesh.vtag_to_tet.get(&tag) {
                    Some(t) => t,
                    None => continue,
                };
                let mat =
                    DebyeMaterial { eps_inf, eps_static, tau };
                for &t in tets {
                    materials[t].eps = [eps_inf; 3];
                    disp_elems.push((t, mat));
                }
            }
        }
        let mut port_specs: Vec<PortSpec> = Vec::new();
        if let Some(ps) = ports {
            for (tag, m, n, dir, z0_op) in ps {
                let dir = dir.map(|(x, y, z)| [x, y, z]);
                let spec = PortSpec::from_mesh_tag_with_z0(
                    &mesh, tag, (m, n), dir, z0_op,
                )
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "port face tag {tag} has no triangles, or its \
                         direction is zero / parallel to the face"
                    ))
                })?;
                port_specs.push(spec);
            }
        }
        // Coaxial TEM ports — appended after the rectangular ports.
        if let Some(cps) = coax_ports {
            for (tag, center) in cps {
                let center = center.map(|(x, y, z)| [x, y, z]);
                let spec = PortSpec::coax_from_mesh_tag(&mesh, tag, center)
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "coax port face tag {tag} has no triangles"
                        ))
                    })?;
                port_specs.push(spec);
            }
        }
        // Floquet plane-wave ports — appended after the rectangular and
        // coax ports. `polarisation_mode` encodes the TE / TM choice as in
        // the FD backend's `mode_nr`: 1 -> TE, 2 -> TM.
        if let Some(fps) = floquet_ports {
            for (tag, pol_nr, scan_theta, scan_phi) in fps {
                let polarisation = match pol_nr {
                    1 => FloquetPolarisation::Te,
                    2 => FloquetPolarisation::Tm,
                    _ => {
                        return Err(PyRuntimeError::new_err(format!(
                            "floquet port face tag {tag}: \
                             polarisation_mode must be 1 (TE) or 2 (TM), \
                             got {pol_nr}"
                        )));
                    }
                };
                let spec = PortSpec::floquet_from_mesh_tag(
                    &mesh,
                    tag,
                    polarisation,
                    scan_theta,
                    scan_phi,
                    None,
                )
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "floquet port face tag {tag} has no triangles"
                    ))
                })?;
                port_specs.push(spec);
            }
        }
        // Numerically-solved wave ports — appended after the rectangular,
        // coax and Floquet modal ports (and before the absorbing faces),
        // so the modal-port subset stays contiguous. Each entry is
        // `(face_tag, te, mode_index, k0)`: a 2D cross-section eigensolve
        // runs at build time and the sampled profile becomes the port
        // mode. `k0 > 0` selects the inhomogeneous vector solve at that
        // operating wavenumber (microstrip-class); `k0 <= 0` the scalar
        // TE/TM solve (homogeneous hollow guide). Per-tet `ε_r` is read
        // off the already-resolved `materials`.
        if let Some(wps) = wave_ports {
            let eps_per_tet: Vec<f64> =
                materials.iter().map(|m| m.eps[0]).collect();
            // Per-node internal-PEC mask: any node on a PEC face tag is a
            // conductor node (the microstrip trace + ground). The vector
            // wave-port solve pins tangential E = 0 there, resolving the
            // quasi-TEM mode of an inhomogeneous line with an embedded
            // trace. Borrowed from `pec_faces` before it is consumed below.
            let pec_nodes: Option<Vec<bool>> = pec_faces.as_ref().map(|tags| {
                let mut mask = vec![false; mesh.n_nodes()];
                for &tag in tags {
                    if let Some(tris) = mesh.ftag_to_tri.get(&tag) {
                        for &t in tris {
                            for &nd in &mesh.tris[t] {
                                mask[nd] = true;
                            }
                        }
                    }
                }
                mask
            });
            for (tag, te, mode_index, k0) in wps {
                let spec = PortSpec::wave_from_mesh_tag(
                    &mesh,
                    tag,
                    te,
                    mode_index,
                    Some(&eps_per_tet),
                    k0,
                    pec_nodes.as_deref(),
                )
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "wave port face tag {tag}: no triangles, or the \
                         cross-section eigensolve found fewer than \
                         {} mode(s)",
                        mode_index + 1,
                    ))
                })?;
                port_specs.push(spec);
            }
        }
        // Absorbing-only (ABC) boundary faces - characteristic
        // non-reflecting flux, no waveguide mode. Appended after the
        // modal ports, so absorbing faces do NOT shift the modal-port
        // indices used by `sparams`.
        if let Some(faces) = abc_faces {
            for tag in faces {
                let spec = PortSpec::absorbing_from_mesh_tag(&mesh, tag)
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "ABC face tag {tag} has no triangles"
                        ))
                    })?;
                port_specs.push(spec);
            }
        }
        // Periodic boundary pairs, collect each `(face_a, face_b)` into a
        // PeriodicSpec. The matcher inside the operator handles the
        // transverse alignment and the face-node permutation.
        let mut periodic_specs: Vec<PeriodicSpec> = Vec::new();
        if let Some(pairs) = periodic_pairs {
            for (face_a, face_b) in pairs {
                let spec = PeriodicSpec::from_mesh_tags(&mesh, face_a, face_b)
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "periodic pair ({face_a}, {face_b}): one of the \
                             face tags has no triangles"
                        ))
                    })?;
                periodic_specs.push(spec);
            }
        }
        // Internal-PEC plates: collect from face tags.
        let mut pec_specs: Vec<PecSpec> = Vec::new();
        if let Some(tags) = pec_faces {
            for tag in tags {
                let spec = PecSpec::from_mesh_tag(&mesh, tag)
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "PEC face tag {tag} has no triangles"
                        ))
                    })?;
                pec_specs.push(spec);
            }
        }
        let op = MaxwellOperator::new_full(
            &mesh, order, flux_alpha, &materials, &port_specs, &disp_elems,
            &periodic_specs, &pec_specs,
        );
        Ok(PyTdOperator {
            op,
            krylov: rapidfem_td::propagator::KrylovWorkspace::new(),
            driven_b: Vec::new(),
            lserk: rapidfem_td::explicit::LserkWorkspace::new(),
            kcl: rapidfem_td::explicit_adaptive::KclWorkspace::new(),
            gpu: None,
        })
    }

    /// Degrees of freedom — `6·Np·n_elem` for the `[E,H]` block, plus
    /// `3·Np` per Debye dispersive element for the appended
    /// auxiliary-polarisation block. Exactly `6·Np·n_elem` with no
    /// dispersive material.
    fn n_dof(&self) -> usize {
        self.op.n_dof()
    }

    /// Number of Debye dispersive elements — the count of appended
    /// polarisation blocks. Zero for a non-dispersive problem.
    fn n_dispersive(&self) -> usize {
        self.op.n_dispersive()
    }

    /// Apply the semi-discrete operator — `dy/dt = A·y`. Zero-copy numpy in
    /// and out: `y` is read straight from its buffer, the result is handed
    /// back as a numpy array — no Python-list round-trip.
    fn apply<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(self.op.apply(y).into_pyarray_bound(py))
    }

    /// Allocation-free matvec: evaluate `dy/dt = A·y` into the caller's `out`
    /// buffer instead of returning a fresh array. Both `y` and `out` must
    /// have length `n_dof`. Useful for hand-rolled integration loops that
    /// reuse one scratch buffer across steps; the high-level steppers
    /// (`step`, `step_explicit`, ...) already do this internally.
    fn apply_into<'py>(
        &self,
        y: PyReadonlyArray1<'py, f64>,
        mut out: PyReadwriteArray1<'py, f64>,
    ) -> PyResult<()> {
        let n = self.op.n_dof();
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let out = out
            .as_slice_mut()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if y.len() != n || out.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "apply_into: expected y and out of length n_dof = {n}, got y={} out={}",
                y.len(),
                out.len()
            )));
        }
        self.op.apply_into(y, out);
        Ok(())
    }

    /// Instantaneous electromagnetic field energy
    /// `½·∫(ε|E|² + μ|H|²) dV` — the material-weighted DG energy norm,
    /// evaluated matrix-free (a cheap per-element sum, no `N×N` matrix).
    /// Zero-copy numpy in: `y` is read straight from its buffer. Only the
    /// `6·Np·n_elem` E,H entries are used, so a state with trailing
    /// auxiliary DOFs is accepted.
    fn field_energy(&self, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(self.op.field_energy(y))
    }

    /// Advance the state by `h` with the matrix-free exponential propagator.
    /// Zero-copy numpy in and out; the Krylov workspace is reused, so a
    /// transient loop allocates only the returned array per step.
    ///
    /// `tol` is the Krylov a-posteriori error tolerance: the subspace stops
    /// growing once the step's error estimate drops below it, so an easy
    /// step costs far fewer than `krylov_dim` matvecs. `tol = 0` disables
    /// the estimate and always runs the full `krylov_dim` — the
    /// fixed-dimension worst case.
    #[pyo3(signature = (y, h, krylov_dim = 40, tol = KRYLOV_TOL))]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        h: f64,
        krylov_dim: usize,
        tol: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut out = vec![0.0; y.len()];
        let op = &self.op;
        self.krylov.expmv_into(
            |x, ax| op.apply_into(x, ax),
            y,
            h,
            krylov_dim,
            tol,
            &mut out,
        );
        Ok(out.into_pyarray_bound(py))
    }

    /// Advance the state by `h` with the explicit LSERK4 integrator: five
    /// matvecs and two state registers, no Krylov subspace. Far cheaper per
    /// step than [`step`](Self::step), but only *conditionally* stable —
    /// an `h` past the operator's CFL limit diverges. Zero-copy numpy in
    /// and out; the LSERK4 workspace is reused across a transient loop.
    fn step_explicit<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        h: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut out = y.to_vec();
        let op = &self.op;
        self.lserk.step_into(|x, ax| op.apply_into(x, ax), &mut out, h);
        Ok(out.into_pyarray_bound(py))
    }

    /// Global DOF index for a field component at the node nearest `point` —
    /// the hook for a soft source or a field probe. `field`: 0 = E, 1 = H.
    /// `comp`: 0 = x, 1 = y, 2 = z.
    fn nearest_node_dof(
        &self,
        point: (f64, f64, f64),
        field: usize,
        comp: usize,
    ) -> usize {
        self.op
            .nearest_node_dof([point.0, point.1, point.2], field, comp)
    }

    /// One source-driven step — `dy/dt = A·y + b` with `b` a single-DOF soft
    /// source — via the exponential time integrator. Zero-copy numpy in/out;
    /// the Krylov workspace and source vector are reused across the loop.
    #[pyo3(signature = (y, source_dof, source_value, h, krylov_dim = 40))]
    fn step_driven<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        source_dof: usize,
        source_value: f64,
        h: f64,
        krylov_dim: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n = y.len();
        if source_dof >= n {
            return Err(PyRuntimeError::new_err("source_dof out of range"));
        }
        if self.driven_b.len() != n {
            self.driven_b = vec![0.0; n];
        }
        self.driven_b[source_dof] = source_value;
        let mut out = vec![0.0; n];
        let op = &self.op;
        self.krylov.etd_step_into(
            |x, ax| op.apply_into(x, ax),
            y,
            &self.driven_b,
            h,
            krylov_dim,
            KRYLOV_TOL,
            &mut out,
        );
        self.driven_b[source_dof] = 0.0;
        Ok(out.into_pyarray_bound(py))
    }

    /// One driven step with the explicit LSERK4 integrator — `dy/dt = A·y +
    /// b` with `b` a single-DOF soft source held constant across the step.
    /// The explicit counterpart of [`step_driven`](Self::step_driven);
    /// cheap per step but CFL-bound. Zero-copy numpy in and out.
    fn step_driven_explicit<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        h: f64,
        source_dof: usize,
        source_value: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if source_dof >= y.len() {
            return Err(PyRuntimeError::new_err("source_dof out of range"));
        }
        let mut out = y.to_vec();
        let op = &self.op;
        self.lserk.step_driven_into(
            |x, ax| op.apply_into(x, ax),
            &mut out,
            h,
            source_dof,
            source_value,
        );
        Ok(out.into_pyarray_bound(py))
    }

    /// Number of waveguide ports on the operator.
    fn n_ports(&self) -> usize {
        self.op.n_ports()
    }

    /// Whether an OpenCL GPU backend is available — built lazily on the
    /// first call. `False` if there is no GPU / OpenCL runtime, or the
    /// operator is dispersive.
    ///
    /// The probe runs under `catch_unwind`: a machine with no OpenCL ICD
    /// loader can panic in the driver-loading layer rather than return an
    /// error, and that must not crash the caller — it just means no GPU.
    fn gpu_available(&mut self) -> bool {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.ensure_gpu().is_ok()
        }))
        .unwrap_or(false)
    }

    /// Name of the GPU device, or an error if no GPU backend is available.
    fn gpu_device(&mut self) -> PyResult<String> {
        self.ensure_gpu().map_err(PyRuntimeError::new_err)?;
        Ok(self.gpu.as_ref().unwrap().ctx.device_name.clone())
    }

    /// Explicit LSERK4 transient on the GPU. Returns the flattened field
    /// trajectory `[(steps+1) * n_dof]` (row 0 is `y0`); the caller
    /// reshapes to `[steps+1, n_dof]`. `h` is the output cadence; the
    /// integrator takes `substeps` LSERK4 steps of `h/substeps` between
    /// snapshots, so the substep stays within the CFL limit. The state
    /// steps device-resident. Zero-copy numpy in.
    fn gpu_transient<'py>(
        &mut self,
        py: Python<'py>,
        y0: PyReadonlyArray1<'py, f64>,
        h: f64,
        steps: usize,
        substeps: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.ensure_gpu().map_err(PyRuntimeError::new_err)?;
        let y0 = y0
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let backend = self.gpu.as_mut().unwrap();
        let traj = backend
            .op
            .transient_traj(&backend.ctx, &y0_32, h as f32, steps, substeps)
            .map_err(PyRuntimeError::new_err)?;
        let traj64: Vec<f64> = traj.iter().map(|&v| v as f64).collect();
        Ok(traj64.into_pyarray_bound(py))
    }

    /// Driven explicit LSERK4 transient on the GPU. `source_values` holds
    /// one soft-source amplitude per substep (`steps * substeps` entries).
    /// Returns the flattened trajectory `[(steps+1) * n_dof]`; the caller
    /// reshapes. Zero-copy numpy in.
    #[allow(clippy::too_many_arguments)]
    fn gpu_transient_driven<'py>(
        &mut self,
        py: Python<'py>,
        y0: PyReadonlyArray1<'py, f64>,
        h: f64,
        steps: usize,
        substeps: usize,
        source_dof: usize,
        source_values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.ensure_gpu().map_err(PyRuntimeError::new_err)?;
        let y0 = y0
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let src = source_values
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let src_32: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        let backend = self.gpu.as_mut().unwrap();
        let traj = backend
            .op
            .transient_driven_traj(
                &backend.ctx,
                &y0_32,
                h as f32,
                steps,
                substeps,
                source_dof,
                &src_32,
            )
            .map_err(PyRuntimeError::new_err)?;
        let traj64: Vec<f64> = traj.iter().map(|&v| v as f64).collect();
        Ok(traj64.into_pyarray_bound(py))
    }

    /// Cutoff angular frequency of port `port_idx`'s mode (operator units).
    fn port_cutoff(&self, port_idx: usize) -> f64 {
        self.op.port_cutoff(port_idx)
    }

    /// True if port `port_idx` carries a waveguide / lumped mode (and
    /// thus participates in S-parameter extraction); false for a pure
    /// absorbing-only ABC face. Used by Python callers to filter
    /// the operator's port list down to its modal subset.
    fn port_has_mode(&self, port_idx: usize) -> bool {
        self.op.port_has_mode(port_idx)
    }

    /// Number of resolved (element, local-face) pairs in port
    /// `port_idx`'s face set - boundary-attached ports return one per
    /// triangle, internal-plate-attached ports return two per triangle.
    /// Diagnostic for verifying that a port plate was correctly
    /// fragmented to lie on a domain boundary.
    fn port_n_faces(&self, port_idx: usize) -> usize {
        self.op.port_n_faces(port_idx)
    }

    /// Of the port's face pairs, how many face an INTERIOR neighbor
    /// (the plate has another tet on its other side, i.e. it is
    /// internal to the domain). A correctly boundary-attached port
    /// has zero interior faces; an internal plate has all of them.
    fn port_n_interior_faces(&self, port_idx: usize) -> usize {
        self.op.port_n_interior_faces(port_idx)
    }

    /// Modal wave impedance `Z(omega)` of port `port_idx` at angular
    /// frequency `omega`, in the operator's normalised units (`Z = 1`
    /// is free space). Dispersive `Z_TE(omega)` for a `TE_mn` waveguide
    /// port, flat `Z = 1` for coax / Floquet. Returns `0` for an
    /// absorbing-only (ABC) port. The forward / backward modal split
    /// `A, B = (P_e ± Z · P_h) / 2` uses this per frequency.
    fn port_impedance(&self, port_idx: usize, omega: f64) -> f64 {
        self.op.port_impedance(port_idx, omega)
    }

    /// Spatial source vector for driving port `port_idx` with a unit
    /// waveform — the system is `dy/dt = A·y + b·g(t)`.
    fn port_source<'py>(
        &self,
        py: Python<'py>,
        port_idx: usize,
    ) -> Bound<'py, PyArray1<f64>> {
        self.op.port_source(port_idx).into_pyarray_bound(py)
    }

    /// Modal field projections `(P_e, P_h)` at port `port_idx` for the
    /// state `y` — the forward/backward split `A,B = (P_e ± Z·P_h)/2` is
    /// done per frequency on the recorded time series.
    fn port_projections(
        &self,
        y: PyReadonlyArray1<'_, f64>,
        port_idx: usize,
    ) -> PyResult<(f64, f64)> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(self.op.port_modal_projections(y, port_idx))
    }

    /// One source-driven step `dy/dt = A·y + b` with a full source vector
    /// `b` — the path for modal port excitation. Zero-copy numpy in/out;
    /// the Krylov workspace is reused across the loop.
    #[pyo3(signature = (y, source, h, krylov_dim = 40))]
    fn step_with_source<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        source: PyReadonlyArray1<'py, f64>,
        h: f64,
        krylov_dim: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let src = source
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if src.len() != y.len() {
            return Err(PyRuntimeError::new_err(
                "source length must equal n_dof",
            ));
        }
        let mut out = vec![0.0; y.len()];
        let op = &self.op;
        self.krylov.etd_step_into(
            |x, ax| op.apply_into(x, ax),
            y,
            src,
            h,
            krylov_dim,
            KRYLOV_TOL,
            &mut out,
        );
        Ok(out.into_pyarray_bound(py))
    }

    /// One source-driven step with the explicit LSERK4 integrator and a
    /// full source vector `b` — the CPU-RK counterpart of
    /// [`step_with_source`](Self::step_with_source), for modal-port
    /// injection on the explicit path. `b` held constant across the step;
    /// CFL-bound. Zero-copy numpy in/out.
    fn step_with_source_explicit<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        source: PyReadonlyArray1<'py, f64>,
        h: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let src = source
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if src.len() != y.len() {
            return Err(PyRuntimeError::new_err(
                "source length must equal n_dof",
            ));
        }
        let mut out = y.to_vec();
        let op = &self.op;
        self.lserk
            .step_with_source_into(|x, ax| op.apply_into(x, ax), &mut out, h, src);
        Ok(out.into_pyarray_bound(py))
    }

    /// One KCL RK4(3)5[2R+]C adaptive step of `dy/dt = A·y`. Returns
    /// `(y_new, err)` — the advanced state and the per-DOF embedded-error
    /// vector. Per step the matvec count is identical to `step_explicit`
    /// (five); the embedded estimate is the price an adaptive controller
    /// pays to drop the dependence on `cfl_dt`. Zero-copy numpy in; the
    /// KCL workspace is reused, so an adaptive transient loop allocates
    /// only the two returned arrays per accepted step.
    fn step_kcl<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        h: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n = y.len();
        let mut out = y.to_vec();
        let mut err = vec![0.0; n];
        let op = &self.op;
        self.kcl
            .step_into(|x, ax| op.apply_into(x, ax), &mut out, &mut err, h);
        Ok((out.into_pyarray_bound(py), err.into_pyarray_bound(py)))
    }

    /// One KCL adaptive step of the driven system `dy/dt = A·y + b` with a
    /// single-DOF soft source held constant across the step. Returns
    /// `(y_new, err)` — the advanced state and the embedded-error vector
    /// for the controller. The driven counterpart of [`step_kcl`](Self::step_kcl);
    /// zeroth-order source hold, like the LSERK4 and ETD driven paths.
    fn step_driven_kcl<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        h: f64,
        source_dof: usize,
        source_value: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n = y.len();
        if source_dof >= n {
            return Err(PyRuntimeError::new_err("source_dof out of range"));
        }
        let mut out = y.to_vec();
        let mut err = vec![0.0; n];
        let op = &self.op;
        self.kcl.step_driven_into(
            |x, ax| op.apply_into(x, ax),
            &mut out,
            &mut err,
            h,
            source_dof,
            source_value,
        );
        Ok((out.into_pyarray_bound(py), err.into_pyarray_bound(py)))
    }

    /// One KCL adaptive step of `dy/dt = A·y + b` with the **full source
    /// vector** `b = source` held constant across the step — the modal-port
    /// injection path. Returns `(y_new, err)`. Vector-source counterpart of
    /// [`step_driven_kcl`](Self::step_driven_kcl).
    fn step_with_source_kcl<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        source: PyReadonlyArray1<'py, f64>,
        h: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let src = source
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n = y.len();
        if src.len() != n {
            return Err(PyRuntimeError::new_err(
                "source length must equal n_dof",
            ));
        }
        let mut out = y.to_vec();
        let mut err = vec![0.0; n];
        let op = &self.op;
        self.kcl.step_with_source_into(
            |x, ax| op.apply_into(x, ax),
            &mut out,
            &mut err,
            h,
            src,
        );
        Ok((out.into_pyarray_bound(py), err.into_pyarray_bound(py)))
    }

    /// One source-driven exponential (ETD) step on the GPU with a full
    /// source vector `b` — `dy/dt = A·y + b` held constant across the
    /// step, via the augmented-Arnoldi propagator. The GPU-Exp counterpart
    /// of [`step_with_source`](Self::step_with_source); covers modal-port
    /// injection (point source is the `b = e_dof·val` special case). Errors
    /// if no fp64 GPU is available.
    #[pyo3(signature = (y, source, h, krylov_dim = 40))]
    fn gpu_step_with_source<'py>(
        &mut self,
        py: Python<'py>,
        y: PyReadonlyArray1<'py, f64>,
        source: PyReadonlyArray1<'py, f64>,
        h: f64,
        krylov_dim: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.ensure_gpu().map_err(PyRuntimeError::new_err)?;
        let y = y
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let src = source
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if src.len() != y.len() {
            return Err(PyRuntimeError::new_err(
                "source length must equal n_dof",
            ));
        }
        let backend = self.gpu.as_mut().unwrap();
        let out = backend
            .op
            .etd_step(&backend.ctx, y, src, h, krylov_dim)
            .map_err(PyRuntimeError::new_err)?;
        Ok(out.into_pyarray_bound(py))
    }

    /// Vector-source driven explicit LSERK4 transient on the GPU, the state
    /// device-resident — the GPU-RK counterpart of the modal-port injection
    /// loop. `source` is the spatial pattern `b` (length `n_dof`);
    /// `source_values` holds one waveform amplitude per substep
    /// (`steps * substeps` entries). Returns the flattened trajectory
    /// `[(steps+1) * n_dof]`; the caller reshapes. Zero-copy numpy in.
    #[allow(clippy::too_many_arguments)]
    fn gpu_transient_driven_vec<'py>(
        &mut self,
        py: Python<'py>,
        y0: PyReadonlyArray1<'py, f64>,
        h: f64,
        steps: usize,
        substeps: usize,
        source: PyReadonlyArray1<'py, f64>,
        source_values: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.ensure_gpu().map_err(PyRuntimeError::new_err)?;
        let y0 = y0
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let b = source
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let src = source_values
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let b_32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let src_32: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        let backend = self.gpu.as_mut().unwrap();
        let traj = backend
            .op
            .transient_driven_vec_traj(
                &backend.ctx,
                &y0_32,
                h as f32,
                steps,
                substeps,
                &b_32,
                &src_32,
            )
            .map_err(PyRuntimeError::new_err)?;
        let traj64: Vec<f64> = traj.iter().map(|&v| v as f64).collect();
        Ok(traj64.into_pyarray_bound(py))
    }

    /// Device-resident KCL RK4(3)5[2R+]C adaptive transient for the free
    /// system `dy/dt = A·y`. Returns `(traj_flat, n_accepted, n_rejected,
    /// h_min, h_max)` — the caller reshapes `traj_flat` to
    /// `[steps+1, n_dof]`. The Rust-side PI controller runs on top of a
    /// per-substep device error reduction, so only the scalar `err_norm`
    /// (per substep) and the state snapshot (per accepted frame) cross
    /// the bus.
    #[allow(clippy::too_many_arguments)]
    fn gpu_transient_kcl<'py>(
        &mut self,
        py: Python<'py>,
        y0: PyReadonlyArray1<'py, f64>,
        h: f64,
        steps: usize,
        atol: f64,
        rtol: f64,
        safety: f64,
        growth_limit: f64,
        shrink_limit: f64,
        pi_alpha: f64,
        pi_beta: f64,
        min_step_factor: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, usize, f64, f64)> {
        self.ensure_gpu().map_err(PyRuntimeError::new_err)?;
        let y0 = y0
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let backend = self.gpu.as_mut().unwrap();
        let (traj, n_acc, n_rej, h_min, h_max) = backend
            .op
            .transient_kcl_traj(
                &backend.ctx, &y0_32, h as f32, steps,
                atol as f32, rtol as f32, safety as f32,
                growth_limit as f32, shrink_limit as f32,
                pi_alpha as f32, pi_beta as f32, min_step_factor as f32,
            )
            .map_err(PyRuntimeError::new_err)?;
        let traj64: Vec<f64> = traj.iter().map(|&v| v as f64).collect();
        Ok((
            traj64.into_pyarray_bound(py),
            n_acc,
            n_rej,
            h_min as f64,
            h_max as f64,
        ))
    }

    /// Device-resident KCL adaptive transient with a single-DOF soft
    /// source held constant across each output frame. `g_values[k]` is the
    /// waveform sampled at frame `k*dt`, length `steps`.
    #[allow(clippy::too_many_arguments)]
    fn gpu_transient_kcl_driven<'py>(
        &mut self,
        py: Python<'py>,
        y0: PyReadonlyArray1<'py, f64>,
        h: f64,
        steps: usize,
        source_dof: usize,
        g_values: PyReadonlyArray1<'py, f64>,
        atol: f64,
        rtol: f64,
        safety: f64,
        growth_limit: f64,
        shrink_limit: f64,
        pi_alpha: f64,
        pi_beta: f64,
        min_step_factor: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, usize, f64, f64)> {
        self.ensure_gpu().map_err(PyRuntimeError::new_err)?;
        let y0 = y0
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let gvals = g_values
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if gvals.len() != steps {
            return Err(PyRuntimeError::new_err(
                "g_values length must equal steps",
            ));
        }
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let g_32: Vec<f32> = gvals.iter().map(|&v| v as f32).collect();
        let backend = self.gpu.as_mut().unwrap();
        let (traj, n_acc, n_rej, h_min, h_max) = backend
            .op
            .transient_kcl_traj_driven(
                &backend.ctx, &y0_32, h as f32, steps, source_dof, &g_32,
                atol as f32, rtol as f32, safety as f32,
                growth_limit as f32, shrink_limit as f32,
                pi_alpha as f32, pi_beta as f32, min_step_factor as f32,
            )
            .map_err(PyRuntimeError::new_err)?;
        let traj64: Vec<f64> = traj.iter().map(|&v| v as f64).collect();
        Ok((
            traj64.into_pyarray_bound(py),
            n_acc,
            n_rej,
            h_min as f64,
            h_max as f64,
        ))
    }

    /// Device-resident KCL adaptive transient with the **full source
    /// vector** `b = source` and waveform `g_values[k]` held across frame
    /// `k` — the modal-port injection path.
    #[allow(clippy::too_many_arguments)]
    fn gpu_transient_kcl_driven_vec<'py>(
        &mut self,
        py: Python<'py>,
        y0: PyReadonlyArray1<'py, f64>,
        h: f64,
        steps: usize,
        source: PyReadonlyArray1<'py, f64>,
        g_values: PyReadonlyArray1<'py, f64>,
        atol: f64,
        rtol: f64,
        safety: f64,
        growth_limit: f64,
        shrink_limit: f64,
        pi_alpha: f64,
        pi_beta: f64,
        min_step_factor: f64,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, usize, f64, f64)> {
        self.ensure_gpu().map_err(PyRuntimeError::new_err)?;
        let y0 = y0
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let b = source
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let gvals = g_values
            .as_slice()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if b.len() != y0.len() {
            return Err(PyRuntimeError::new_err(
                "source length must equal n_dof",
            ));
        }
        if gvals.len() != steps {
            return Err(PyRuntimeError::new_err(
                "g_values length must equal steps",
            ));
        }
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();
        let b_32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let g_32: Vec<f32> = gvals.iter().map(|&v| v as f32).collect();
        let backend = self.gpu.as_mut().unwrap();
        let (traj, n_acc, n_rej, h_min, h_max) = backend
            .op
            .transient_kcl_traj_driven_vec(
                &backend.ctx, &y0_32, h as f32, steps, &b_32, &g_32,
                atol as f32, rtol as f32, safety as f32,
                growth_limit as f32, shrink_limit as f32,
                pi_alpha as f32, pi_beta as f32, min_step_factor as f32,
            )
            .map_err(PyRuntimeError::new_err)?;
        let traj64: Vec<f64> = traj.iter().map(|&v| v as f64).collect();
        Ok((
            traj64.into_pyarray_bound(py),
            n_acc,
            n_rej,
            h_min as f64,
            h_max as f64,
        ))
    }

    /// The explicit sparse state-space matrix `A`, as a CSR quadruple
    /// `(n, row_ptr, col_idx, values)` — the index/value arrays come back
    /// as numpy arrays, so they feed straight into `scipy.sparse.csr_matrix`
    /// without a Python-list round-trip.
    fn state_space<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        usize,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let csr = self.op.assemble_sparse();
        let row_ptr: Vec<i64> =
            csr.row_ptr.iter().map(|&x| x as i64).collect();
        let col_idx: Vec<i64> =
            csr.col_idx.iter().map(|&x| x as i64).collect();
        (
            csr.n,
            row_ptr.into_pyarray_bound(py),
            col_idx.into_pyarray_bound(py),
            csr.values.into_pyarray_bound(py),
        )
    }

    /// Assemble the operator as a dense row-major `N×N` matrix, returned
    /// as a flat `N²` float64 numpy array (reshape to `(N, N)`). This is
    /// `O(N²)` memory — for **validation on small meshes only**; use
    /// `state_space` (sparse CSR) for anything sizeable.
    fn assemble_dense<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        self.op.assemble_dense().into_pyarray_bound(py)
    }

    /// Assemble the energy mass matrix `M` as a dense row-major `N×N`
    /// matrix (flat `N²` float64 array, reshape to `(N, N)`): the
    /// material-weighted DG mass matrix whose quadratic form gives the
    /// field energy `½ yᵀ M y` (matching `field_energy`). Consistent
    /// (non-lumped), so it carries off-diagonal element-mass coupling.
    /// `O(N²)` memory — small meshes / validation only.
    fn assemble_energy_mass<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        self.op.assemble_energy_mass().into_pyarray_bound(py)
    }

    /// DG node physical coordinates — an `(n_elem·Np, 3)` numpy array, in
    /// state order (`point[e*Np + node]`).
    fn node_coords<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        let pts = self.op.node_coords();
        let n = pts.len();
        let mut flat: Vec<f64> = Vec::with_capacity(n * 3);
        for p in &pts {
            flat.extend_from_slice(p);
        }
        numpy::ndarray::Array2::from_shape_vec((n, 3), flat)
            .expect("(n,3) shape")
            .into_pyarray_bound(py)
    }

    /// The four corner local-node indices `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`
    /// — the linear-tetrahedron connectivity for a VTK field export.
    fn corner_local_nodes(&self) -> (usize, usize, usize, usize) {
        let c = self.op.corner_local_nodes();
        (c[0], c[1], c[2], c[3])
    }

}

/// rapidfem — frequency- and time-domain EM FEM solver.
#[pymodule]
#[pyo3(name = "_native")]
fn rapidfem_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Solver selection is automatic: PARDISO if MKL is loadable (typically
    // 5–10× faster on complex-symmetric LU), faer otherwise. Force one with
    // RAPIDFEM_SOLVER=faer or =pardiso before importing.

    m.add_class::<PySimulation>()?;
    m.add_class::<PySweepResult>()?;
    m.add_class::<PyEigenmode>()?;
    m.add_class::<PyRadiationPattern>()?;
    m.add_class::<PyTdOperator>()?;
    Ok(())
}
