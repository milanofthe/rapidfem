// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! PyO3 bindings for rapidfem: the typed `Model`, the frequency-domain
//! `Simulation` with its results, and the time-domain `TdSession`.
//!
//! Build via `maturin develop` (dev) or `maturin build --release` (wheel).

mod geometry;
mod model;
mod td;

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rapidfem_fd::eigenmode::Eigenmode;
use rapidfem_fd::farfield::RadiationPattern;
use rapidfem_fd::order::OrderPolicy;
use rapidfem_fd::simulation::{FdSettings, Simulation, SweepResult};
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

#[pymethods]
impl PySimulation {
    /// Build a simulation on a solver mesh (`Geometry.fem_mesh`) and a
    /// `Model`.
    ///
    /// `order` is the uniform element order (1 or 2); `adaptive` selects the
    /// wavelength order policy instead. `eigenmode` is `(target_hz, n_modes)`.
    #[new]
    #[pyo3(signature = (mesh, model, frequencies, *, order=2, adaptive=false, eigenmode=None))]
    fn new(
        mesh: &PyFemMesh,
        model: &PyModel,
        frequencies: Vec<f64>,
        order: u8,
        adaptive: bool,
        eigenmode: Option<(f64, usize)>,
    ) -> PyResult<Self> {
        if !(1..=2).contains(&order) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "order must be 1 or 2, got {order}"
            )));
        }
        let order = if adaptive {
            OrderPolicy::Adaptive { theta: rapidfem_fd::order::DEFAULT_THETA }
        } else {
            OrderPolicy::Uniform(order)
        };
        let settings = FdSettings { frequencies, order, eigenmode };
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

    /// Monk-style residual error indicator η per tetrahedron at
    /// ``(freq_idx, port_idx)``. Returns a dict ``{eta, total, marked,
    /// volume_residuals, face_jumps}`` with ``eta`` shape ``(n_tets,)``
    /// float64, ``marked`` an int64 array of Dörfler-selected tet
    /// indices at fraction ``theta``. Diagnostic only, does not
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
        let dict = pyo3::types::PyDict::new(py);
        let eta = est.element_errors.clone().into_pyarray(py);
        let volr = est.volume_residuals.clone().into_pyarray(py);
        let fj = est.face_jumps.clone().into_pyarray(py);
        // h_k lives in mesh-internal units (l0-normalised); every other
        // length crossing this boundary (mesh_nodes, refine_near_points)
        // is in meters, so convert here.
        let l0 = self.inner.mesh.l0;
        let h_k: Vec<f64> = est.h_k.iter().map(|&h| h * l0).collect();
        let h_k = h_k.into_pyarray(py);
        let marked: Vec<i64> = est.marked_elements.iter().map(|&i| i as i64).collect();
        let marked_arr = marked.into_pyarray(py);
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
        self.inner.frequencies.clone().into_pyarray(py)
    }

    /// S-parameter matrix, shape `[n_freq, n_driven, n_driven]`, dtype complex128.
    /// Indexing: `S[freq_idx, observation_port, excitation_port]`.
    #[getter]
    fn sparams<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<Complex64>> {
        let n_freq = self.inner.frequencies.len();
        let n = self.inner.n_driven;
        let flat: Vec<Complex64> = self.inner.sparams.iter().flatten().flatten().copied().collect();
        let arr = numpy::ndarray::Array3::from_shape_vec((n_freq, n, n), flat)
            .expect("shape matches data");
        arr.into_pyarray(py)
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

/// A flat `[x, y, z]`-per-node vector as an `(n_nodes, 3)` array.
fn per_node<'py>(flat: Vec<Complex64>, py: Python<'py>) -> Bound<'py, PyArray2<Complex64>> {
    let n = flat.len() / 3;
    numpy::ndarray::Array2::from_shape_vec((n, 3), flat).expect("shape").into_pyarray(py)
}

/// rapidfem, frequency- and time-domain EM FEM solver.
#[pymodule]
#[pyo3(name = "_native")]
fn rapidfem_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyGeometry>()?;
    m.add_class::<PyFemMesh>()?;
    m.add_class::<geometry::PyMeshScene>()?;
    m.add_class::<PySimulation>()?;
    m.add_class::<PySweepResult>()?;
    m.add_class::<PyEigenmode>()?;
    m.add_class::<PyRadiationPattern>()?;
    m.add_class::<td::PyTdSession>()?;
    Ok(())
}
