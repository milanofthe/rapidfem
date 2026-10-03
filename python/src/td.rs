// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The time-domain session: a thin layer over `rapidfem_td::session`.
//! Times are physical throughout; `ProblemTD` only wraps the results.

use std::cell::RefCell;
use std::path::Path;

use numpy::ndarray::Array2;
use numpy::{
    AllowTypeChange, Complex64, IntoPyArray, PyArray1, PyArray2, PyArrayLikeDyn, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArray1,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyFloat;
use rapidfem_td::constants::KRYLOV_TOL;
use rapidfem_td::session::{
    flux_alpha, Device, Drive, GaussianPulse, Hook, Method, Record, RunOptions, TdSession, Waveform,
};

use crate::geometry::PyFemMesh;
use crate::model::PyModel;

fn rt(e: impl ToString) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn slice<'a>(a: &'a PyReadonlyArray1<'_, f64>) -> PyResult<&'a [f64]> {
    a.as_slice().map_err(rt)
}

/// A Gaussian pulse, optionally modulated by a sinusoidal carrier.
///
/// ``g(t) = exp(-((t-t0)/tau)^2) * cos(2*pi*f0*(t-t0))``
///
/// With ``f0 = None`` the bare Gaussian is a smooth broadband pulse; with a
/// carrier it is a band-limited pulse centred on ``f0``. The pulse is
/// native: a time-domain run samples it in Rust, without a call into Python
/// per step. Calling it evaluates ``g`` at a time or an array of times.
///
/// Parameters
/// ----------
/// t0 : float
///     Pulse-centre time, in the simulation's time units (seconds in SI).
/// tau : float
///     Gaussian width (the `1/e` half-width), in the same time units.
/// f0 : float, optional
///     Carrier frequency in the reciprocal units (Hz in SI); omit for a
///     bare Gaussian.
#[pyclass(name = "GaussianPulse", module = "rapidfem", frozen)]
pub struct PyGaussianPulse {
    inner: GaussianPulse,
}

#[pymethods]
impl PyGaussianPulse {
    #[new]
    #[pyo3(signature = (*, t0, tau, f0 = None))]
    fn new(t0: f64, tau: f64, f0: Option<f64>) -> Self {
        PyGaussianPulse { inner: GaussianPulse { t0, tau, f0 } }
    }

    /// Pulse-centre time.
    #[getter]
    fn t0(&self) -> f64 {
        self.inner.t0
    }

    /// Gaussian `1/e` half-width.
    #[getter]
    fn tau(&self) -> f64 {
        self.inner.tau
    }

    /// Carrier frequency, ``None`` for the bare Gaussian.
    #[getter]
    fn f0(&self) -> Option<f64> {
        self.inner.f0
    }

    /// ``g(t)``: a float for a scalar ``t``, an array of ``t``'s shape
    /// otherwise.
    fn __call__<'py>(&self, py: Python<'py>, t: PyArrayLikeDyn<'py, f64, AllowTypeChange>) -> Bound<'py, PyAny> {
        let g = t.as_array().mapv(|x| self.inner.eval(x));
        if g.ndim() == 0 {
            PyFloat::new(py, g.first().copied().unwrap_or(f64::NAN)).into_any()
        } else {
            g.into_pyarray(py).into_any()
        }
    }

    fn __repr__(&self) -> String {
        let GaussianPulse { t0, tau, f0 } = self.inner;
        match f0 {
            Some(f0) => format!("GaussianPulse(t0={t0:e}, tau={tau:e}, f0={f0:e})"),
            None => format!("GaussianPulse(t0={t0:e}, tau={tau:e})"),
        }
    }
}

/// The excitation of a run: a native pulse, sampled in Rust, or any Python
/// callable `g(t)`, called once per sample.
enum Excitation<'a, 'py> {
    Native(GaussianPulse),
    Python(&'a Bound<'py, PyAny>),
}

impl<'a, 'py> Excitation<'a, 'py> {
    fn of(waveform: &'a Bound<'py, PyAny>) -> Self {
        match waveform.cast::<PyGaussianPulse>() {
            Ok(pulse) => Excitation::Native(pulse.get().inner),
            Err(_) => Excitation::Python(waveform),
        }
    }
}

/// Runs `f` with the waveform as a Rust one and a hook that checks for
/// signals once per frame; an exception raised by either comes back as
/// itself.
fn with_python<T>(
    py: Python<'_>,
    waveform: Option<&Bound<'_, PyAny>>,
    f: impl FnOnce(Option<&mut Waveform>, &mut Hook) -> Result<T, String>,
) -> PyResult<T> {
    let raised: RefCell<Option<PyErr>> = RefCell::new(None);
    let keep = |e: PyErr| {
        let msg = e.to_string();
        *raised.borrow_mut() = Some(e);
        msg
    };
    let excitation = waveform.map(Excitation::of);
    let mut wave = |t: f64| -> Result<f64, String> {
        match excitation.as_ref().expect("called only with a waveform") {
            Excitation::Native(pulse) => Ok(pulse.eval(t)),
            Excitation::Python(w) => w.call1((t,)).and_then(|v| v.extract::<f64>()).map_err(keep),
        }
    };
    let mut hook = || py.check_signals().map_err(keep);
    let w: Option<&mut Waveform> = if waveform.is_some() { Some(&mut wave) } else { None };
    let out = f(w, &mut hook);
    out.map_err(|msg| raised.take().unwrap_or_else(|| PyRuntimeError::new_err(msg)))
}

/// The run options; without a `method` the device's default integrator.
fn options(method: Option<&str>, device: &str, krylov_dim: usize, warmup: usize, verbose: bool) -> PyResult<RunOptions> {
    let device: Device = device.parse().map_err(PyValueError::new_err)?;
    let method: Method = match method {
        Some(m) => m.parse().map_err(PyValueError::new_err)?,
        None => device.default_method(),
    };
    Ok(RunOptions { method, device, krylov_dim, warmup, verbose, ..Default::default() })
}

fn matrix<'py>(py: Python<'py>, rows: usize, cols: usize, data: Vec<f64>) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((rows, cols), data).expect("row-major data").into_pyarray(py)
}

/// The DGTD operator of a problem with its steppers and runs, see
/// `rapidfem_td::session`.
#[pyclass(name = "TdSession", unsendable)]
pub struct PyTdSession {
    s: TdSession,
}

impl PyTdSession {
    /// The frame count of a `[frames, n_dof]` trajectory.
    fn frames(&self, states: &PyReadonlyArray2<'_, f64>) -> PyResult<usize> {
        let (rows, cols) = states.as_array().dim();
        let n = self.s.n_dof();
        if cols != n {
            return Err(PyValueError::new_err(format!("states carry {cols} DOFs, expected {n}")));
        }
        Ok(rows)
    }

    /// The operator indices of the modal ports on face tags `tags`.
    fn ports(&self, tags: &[i32]) -> PyResult<Vec<usize>> {
        tags.iter().map(|&t| self.s.port_of_tag(t).map_err(PyValueError::new_err)).collect()
    }
}

#[pymethods]
impl PyTdSession {
    /// A structured box cavity `[0,lx]×[0,ly]×[0,lz]` of `nx·ny·nz` cells
    /// with PEC walls, DG order `order`, numerical flux `flux` ("upwind" or
    /// "central"), `c` the speed of light in the box's unit.
    #[staticmethod]
    #[pyo3(signature = (nx, ny, nz, lx, ly, lz, order, flux = "upwind", c = 1.0))]
    fn box_cavity(nx: usize, ny: usize, nz: usize, lx: f64, ly: f64, lz: f64, order: usize, flux: &str, c: f64) -> PyResult<Self> {
        let alpha = flux_alpha(flux).map_err(PyValueError::new_err)?;
        let mesh = rapidfem_td::mesh_gen::structured_box(nx, ny, nz, lx, ly, lz);
        let op = rapidfem_td::rhs::MaxwellOperator::new(&mesh, order, alpha, Default::default());
        Ok(PyTdSession { s: TdSession::new(op, c) })
    }

    /// The operator of a `Model` on a solver mesh, its ports addressed by
    /// their face tags.
    #[staticmethod]
    #[pyo3(signature = (mesh, model, order, flux = "upwind", c = 299_792_458.0))]
    fn from_model(mesh: &PyFemMesh, model: &PyModel, order: usize, flux: &str, c: f64) -> PyResult<Self> {
        let alpha = flux_alpha(flux).map_err(PyValueError::new_err)?;
        let (op, port_tags) = rapidfem_td::build::operator_from_model(&mesh.inner, &model.inner, order, alpha, c)
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyTdSession { s: TdSession::new(op, c).with_port_tags(port_tags) })
    }

    /// State length: `6·Np·n_elem`, plus `3·Np` per dispersive element.
    fn n_dofs(&self) -> usize {
        self.s.n_dof()
    }

    /// Number of tetrahedra.
    fn n_tets(&self) -> usize {
        self.s.op().n_elem()
    }

    /// `A·y`.
    fn apply<'py>(&self, py: Python<'py>, y: PyReadonlyArray1<'py, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y = slice(&y)?;
        if y.len() != self.s.n_dof() {
            return Err(PyValueError::new_err("y must have length n_dof"));
        }
        Ok(self.s.op().apply(y).into_pyarray(py))
    }

    /// `A·y` into `out`.
    fn apply_into(&self, y: PyReadonlyArray1<'_, f64>, mut out: PyReadwriteArray1<'_, f64>) -> PyResult<()> {
        let n = self.s.n_dof();
        let y = slice(&y)?;
        let out = out.as_slice_mut().map_err(rt)?;
        if y.len() != n || out.len() != n {
            return Err(PyValueError::new_err(format!(
                "apply_into: expected y and out of length n_dof = {n}, got y={} out={}",
                y.len(),
                out.len()
            )));
        }
        self.s.op().apply_into(y, out);
        Ok(())
    }

    /// `½·∫(ε|E|² + μ|H|²) dV` of a state.
    fn field_energy(&self, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        Ok(self.s.op().field_energy(slice(&y)?))
    }

    /// `exp(h·A)·y` by the Krylov propagator, `tol` its a-posteriori
    /// tolerance (`None` the default).
    #[pyo3(signature = (y, h, krylov_dim = 40, tol = None))]
    fn step<'py>(&mut self, py: Python<'py>, y: PyReadonlyArray1<'py, f64>, h: f64, krylov_dim: usize, tol: Option<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let tol = tol.unwrap_or(KRYLOV_TOL);
        Ok(self.s.step(slice(&y)?, h, krylov_dim, tol).map_err(PyValueError::new_err)?.into_pyarray(py))
    }

    /// One LSERK4 step.
    fn step_explicit<'py>(&mut self, py: Python<'py>, y: PyReadonlyArray1<'py, f64>, h: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.s.step_explicit(slice(&y)?, h).map_err(PyValueError::new_err)?.into_pyarray(py))
    }

    /// One KCL step: `(y_new, err)`.
    fn step_adaptive<'py>(&mut self, py: Python<'py>, y: PyReadonlyArray1<'py, f64>, h: f64) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let (y, e) = self.s.step_adaptive(slice(&y)?, h).map_err(PyValueError::new_err)?;
        Ok((y.into_pyarray(py), e.into_pyarray(py)))
    }

    /// The largest stable LSERK4 step, cached.
    #[pyo3(signature = (recompute = false))]
    fn cfl_dt(&mut self, recompute: bool) -> f64 {
        self.s.cfl_dt(recompute)
    }

    /// Global DOF of field `field` ("E" or "H"), component `component`
    /// ("x", "y" or "z") at the DG node nearest `point`.
    fn nearest_node_dof(&self, point: (f64, f64, f64), field: &str, component: &str) -> PyResult<usize> {
        let field = match field {
            "E" => 0,
            "H" => 1,
            _ => return Err(PyValueError::new_err(format!("field must be 'E' or 'H', got {field:?}"))),
        };
        let comp = match component {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            _ => return Err(PyValueError::new_err(format!("component must be 'x', 'y' or 'z', got {component:?}"))),
        };
        Ok(self.s.op().nearest_node_dof([point.0, point.1, point.2], field, comp))
    }

    /// A run of `steps` frames of `dt`, driven at `source_dof` or by the
    /// mode of the port on face tag `port` with `waveform(t)`. Returns the
    /// recorded frames `[steps + 1, n_dof]` (or `[steps + 1, len(probes)]`)
    /// and, for the adaptive integrator, `(accepted, rejected, h_min,
    /// h_max)`. Without a `method` the device's default integrator runs.
    #[pyo3(signature = (y0 = None, *, dt, steps, source_dof = None, port = None, waveform = None,
                        probes = None, method = None, device = "cpu", krylov_dim = 40,
                        warmup = 0, verbose = true))]
    fn transient<'py>(
        &mut self,
        py: Python<'py>,
        y0: Option<PyReadonlyArray1<'py, f64>>,
        dt: f64,
        steps: usize,
        source_dof: Option<usize>,
        port: Option<i32>,
        waveform: Option<Bound<'py, PyAny>>,
        probes: Option<Vec<usize>>,
        method: Option<&str>,
        device: &str,
        krylov_dim: usize,
        warmup: usize,
        verbose: bool,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Option<(usize, usize, f64, f64)>)> {
        let opts = options(method, device, krylov_dim, warmup, verbose)?;
        let y0 = y0.as_ref().map(slice).transpose()?;
        if source_dof.is_some() && port.is_some() {
            return Err(PyValueError::new_err("pass either source= (point) or port= (modal port), not both"));
        }
        let pattern = match port {
            Some(tag) => Some(self.s.op().port_source(self.ports(&[tag])?[0])),
            None => None,
        };
        let drive = match (source_dof, pattern.as_deref()) {
            (Some(d), _) => Drive::Point(d),
            (None, Some(b)) => Drive::Vector(b),
            (None, None) => Drive::Free,
        };
        let record = match &probes {
            Some(p) => Record::Dofs(p),
            None => Record::States,
        };
        let s = &mut self.s;
        let run = with_python(py, waveform.as_ref(), |w, hook| {
            s.transient(y0, dt, steps, drive, w, record, &opts, hook)
        })?;
        let stats = run.kcl.map(|k| (k.accepted, k.rejected, k.h_min, k.h_max));
        Ok((matrix(py, run.rows, run.width, run.data), stats))
    }

    /// `(frequencies, H)` of the field-to-field transfer function from
    /// `source_dof` to `probe_dof` under `pulse`. Without a `method` the
    /// device's default integrator runs.
    #[pyo3(signature = (source_dof, probe_dof, pulse, *, dt, steps, method = None,
                        device = "cpu", krylov_dim = 40, verbose = true))]
    fn transfer_function<'py>(
        &mut self,
        py: Python<'py>,
        source_dof: usize,
        probe_dof: usize,
        pulse: Bound<'py, PyAny>,
        dt: f64,
        steps: usize,
        method: Option<&str>,
        device: &str,
        krylov_dim: usize,
        verbose: bool,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<Complex64>>)> {
        let opts = options(method, device, krylov_dim, 0, verbose)?;
        let s = &mut self.s;
        let (f, h) = with_python(py, Some(&pulse), |w, hook| {
            s.transfer_function(source_dof, probe_dof, w.expect("pulse"), dt, steps, &opts, hook)
        })?;
        Ok((f.into_pyarray(py), h.into_pyarray(py)))
    }

    /// The `n` lowest distinct cavity resonances (Hz for an SI operator).
    #[pyo3(signature = (n = 8))]
    fn resonances<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.s.resonances(n).map_err(rt)?.into_pyarray(py))
    }

    /// The modal amplitude `P_e` of the ports on face tags `tags` over a
    /// trajectory, `[len(tags), n_frames]`.
    fn port_signals<'py>(&self, py: Python<'py>, states: PyReadonlyArray2<'py, f64>, tags: Vec<i32>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self.frames(&states)?;
        let ports = self.ports(&tags)?;
        let data = self.s.port_signals(states.as_slice().map_err(rt)?, &ports).map_err(PyValueError::new_err)?;
        Ok(matrix(py, ports.len(), rows, data))
    }

    /// Writes a state `[n_dof]` or a trajectory `[frames, n_dof]` as a VTK
    /// series over `times` (default the frame index), returns the `.pvd`
    /// path.
    #[pyo3(signature = (states, path, times = None))]
    fn export_vtk(&self, states: PyReadonlyArrayDyn<'_, f64>, path: &str, times: Option<Vec<f64>>) -> PyResult<String> {
        let states = states.as_slice().map_err(rt)?;
        let frames = states.len() / self.s.n_dof().max(1);
        let times = times.unwrap_or_else(|| (0..frames).map(|k| k as f64).collect());
        let pvd = self.s.export_vtk(states, &times, Path::new(path)).map_err(|e| {
            if e.kind() == std::io::ErrorKind::InvalidInput { PyValueError::new_err(e.to_string()) } else { e.into() }
        })?;
        Ok(pvd.to_string_lossy().into_owned())
    }

    /// `A` as CSR `(n, row_ptr, col_idx, values)`.
    fn state_space<'py>(&self, py: Python<'py>) -> (usize, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
        let csr = self.s.op().assemble_sparse();
        let row_ptr: Vec<i64> = csr.row_ptr.iter().map(|&x| x as i64).collect();
        let col_idx: Vec<i64> = csr.col_idx.iter().map(|&x| x as i64).collect();
        (csr.n, row_ptr.into_pyarray(py), col_idx.into_pyarray(py), csr.values.into_pyarray(py))
    }

    /// DG node coordinates `[n_elem·Np, 3]` in state order.
    fn node_coords<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let pts = self.s.op().node_coords();
        let n = pts.len();
        matrix(py, n, 3, pts.into_iter().flatten().collect())
    }

    /// Local node indices of the four tet corners.
    fn corner_local_nodes(&self) -> (usize, usize, usize, usize) {
        let c = self.s.op().corner_local_nodes();
        (c[0], c[1], c[2], c[3])
    }

    /// Whether a GPU backend can be built.
    fn gpu_available(&mut self) -> bool {
        self.s.gpu_device().is_ok()
    }
}
