// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The time-domain session: a thin layer over `rapidfem_td::session`.
//! Times are physical throughout; `ProblemTD` only wraps the results.

use std::cell::RefCell;
use std::path::Path;

use numpy::ndarray::Array2;
use numpy::{
    Complex64, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyReadwriteArray1,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rapidfem_td::constants::KRYLOV_TOL;
use rapidfem_td::session::{Drive, Hook, Record, RunOptions, TdSession, Waveform};

use crate::geometry::PyFemMesh;
use crate::model::PyModel;

fn rt(e: impl ToString) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn slice<'a>(a: &'a PyReadonlyArray1<'_, f64>) -> PyResult<&'a [f64]> {
    a.as_slice().map_err(rt)
}

/// Runs `f` with the Python waveform as a Rust one and a hook that checks
/// for signals once per frame; an exception raised by either comes back as
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
    let mut wave = |t: f64| -> Result<f64, String> {
        let w = waveform.expect("called only with a waveform");
        w.call1((t,)).and_then(|v| v.extract::<f64>()).map_err(keep)
    };
    let mut hook = || py.check_signals().map_err(keep);
    let w: Option<&mut Waveform> = if waveform.is_some() { Some(&mut wave) } else { None };
    let out = f(w, &mut hook);
    out.map_err(|msg| raised.take().unwrap_or_else(|| PyRuntimeError::new_err(msg)))
}

fn options(method: &str, device: &str, krylov_dim: usize, warmup: usize, verbose: bool) -> PyResult<RunOptions> {
    Ok(RunOptions {
        method: method.parse().map_err(PyValueError::new_err)?,
        device: device.parse().map_err(PyValueError::new_err)?,
        krylov_dim,
        warmup,
        verbose,
        ..Default::default()
    })
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
}

#[pymethods]
impl PyTdSession {
    /// A structured box cavity `[0,lx]×[0,ly]×[0,lz]` of `nx·ny·nz` cells
    /// with PEC walls, DG order `order`, flux blend `flux_alpha` (1 upwind,
    /// 0 central), `c` the speed of light in the box's unit.
    #[staticmethod]
    #[pyo3(signature = (nx, ny, nz, lx, ly, lz, order, flux_alpha = 1.0, c = 1.0))]
    fn box_cavity(nx: usize, ny: usize, nz: usize, lx: f64, ly: f64, lz: f64, order: usize, flux_alpha: f64, c: f64) -> Self {
        let mesh = rapidfem_td::mesh_gen::structured_box(nx, ny, nz, lx, ly, lz);
        let op = rapidfem_td::rhs::MaxwellOperator::new(&mesh, order, flux_alpha);
        PyTdSession { s: TdSession::new(op, c) }
    }

    /// The operator of a `Model` on a solver mesh.
    #[staticmethod]
    #[pyo3(signature = (mesh, model, order, flux_alpha = 1.0, c = 299_792_458.0))]
    fn from_model(mesh: &PyFemMesh, model: &PyModel, order: usize, flux_alpha: f64, c: f64) -> PyResult<Self> {
        let op = rapidfem_td::build::operator_from_model(&mesh.inner, &model.inner, order, flux_alpha, c)
            .map_err(PyRuntimeError::new_err)?;
        Ok(PyTdSession { s: TdSession::new(op, c) })
    }

    /// State length: `6·Np·n_elem`, plus `3·Np` per dispersive element.
    fn n_dofs(&self) -> usize {
        self.s.n_dof()
    }

    /// Number of tetrahedra.
    fn n_tets(&self) -> usize {
        self.s.op().n_elem()
    }

    fn n_dispersive(&self) -> usize {
        self.s.op().n_dispersive()
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

    /// `exp(h·A)·y` by the Krylov propagator.
    #[pyo3(signature = (y, h, krylov_dim = 40, tol = KRYLOV_TOL))]
    fn step<'py>(&mut self, py: Python<'py>, y: PyReadonlyArray1<'py, f64>, h: f64, krylov_dim: usize, tol: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
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

    /// Global DOF of a field component (`field` 0 = E, 1 = H; `comp` 0..3)
    /// at the DG node nearest `point`.
    fn nearest_node_dof(&self, point: (f64, f64, f64), field: usize, comp: usize) -> usize {
        self.s.op().nearest_node_dof([point.0, point.1, point.2], field, comp)
    }

    /// A run of `steps` frames of `dt`, driven at `source_dof` or by the
    /// pattern `source` with `waveform(t)`. Returns the recorded frames
    /// `[steps + 1, n_dof]` (or `[steps + 1, len(probes)]`) and, for the
    /// adaptive integrator, `(accepted, rejected, h_min, h_max)`.
    #[pyo3(signature = (y0 = None, *, dt, steps, source_dof = None, source = None, waveform = None,
                        probes = None, method = "exponential", device = "cpu", krylov_dim = 40,
                        warmup = 0, verbose = true))]
    fn transient<'py>(
        &mut self,
        py: Python<'py>,
        y0: Option<PyReadonlyArray1<'py, f64>>,
        dt: f64,
        steps: usize,
        source_dof: Option<usize>,
        source: Option<PyReadonlyArray1<'py, f64>>,
        waveform: Option<Bound<'py, PyAny>>,
        probes: Option<Vec<usize>>,
        method: &str,
        device: &str,
        krylov_dim: usize,
        warmup: usize,
        verbose: bool,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Option<(usize, usize, f64, f64)>)> {
        let opts = options(method, device, krylov_dim, warmup, verbose)?;
        let y0 = y0.as_ref().map(slice).transpose()?;
        let source = source.as_ref().map(slice).transpose()?;
        let drive = match (source_dof, source) {
            (Some(_), Some(_)) => return Err(PyValueError::new_err("pass source_dof or source, not both")),
            (Some(d), None) => Drive::Point(d),
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
    /// `source_dof` to `probe_dof` under `pulse`.
    #[pyo3(signature = (source_dof, probe_dof, pulse, *, dt, steps, method = "exponential",
                        device = "cpu", krylov_dim = 40, verbose = true))]
    fn transfer_function<'py>(
        &mut self,
        py: Python<'py>,
        source_dof: usize,
        probe_dof: usize,
        pulse: Bound<'py, PyAny>,
        dt: f64,
        steps: usize,
        method: &str,
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

    /// Operator index of the `k`-th modal port.
    fn modal_port(&self, k: usize) -> PyResult<usize> {
        self.s.modal_port(k).map_err(rt)
    }

    /// The modal amplitude `P_e` of each port over a trajectory,
    /// `[len(ports), n_frames]`.
    fn port_signals<'py>(&self, py: Python<'py>, states: PyReadonlyArray2<'py, f64>, ports: Vec<usize>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self.frames(&states)?;
        let data = self.s.port_signals(states.as_slice().map_err(rt)?, &ports).map_err(PyValueError::new_err)?;
        Ok(matrix(py, ports.len(), rows, data))
    }

    /// Writes the trajectory as a VTK series, returns the `.pvd` path.
    fn export_vtk(&self, states: PyReadonlyArray2<'_, f64>, times: Vec<f64>, path: &str) -> PyResult<String> {
        self.frames(&states)?;
        let pvd = self.s.export_vtk(states.as_slice().map_err(rt)?, &times, Path::new(path)).map_err(|e| {
            if e.kind() == std::io::ErrorKind::InvalidInput { PyValueError::new_err(e.to_string()) } else { e.into() }
        })?;
        Ok(pvd.to_string_lossy().into_owned())
    }

    fn n_ports(&self) -> usize {
        self.s.op().n_ports()
    }

    /// Whether port `idx` carries a mode (false for an absorbing-only face).
    fn port_has_mode(&self, idx: usize) -> bool {
        self.s.op().port_has_mode(idx)
    }

    /// Cutoff angular frequency of port `idx`'s mode (operator units).
    fn port_cutoff(&self, idx: usize) -> f64 {
        self.s.op().port_cutoff(idx)
    }

    /// Modal wave impedance of port `idx` at `omega` (operator units).
    fn port_impedance(&self, idx: usize, omega: f64) -> f64 {
        self.s.op().port_impedance(idx, omega)
    }

    /// (element, face) pairs of port `idx`.
    fn port_n_faces(&self, idx: usize) -> usize {
        self.s.op().port_n_faces(idx)
    }

    /// Of those, the ones with a neighbour behind the port.
    fn port_n_interior_faces(&self, idx: usize) -> usize {
        self.s.op().port_n_interior_faces(idx)
    }

    /// The source pattern `b` that drives port `idx`.
    fn port_source<'py>(&self, py: Python<'py>, idx: usize) -> Bound<'py, PyArray1<f64>> {
        self.s.op().port_source(idx).into_pyarray(py)
    }

    /// `(P_e, P_h)` of port `idx` for the state `y`.
    fn port_projections(&self, y: PyReadonlyArray1<'_, f64>, idx: usize) -> PyResult<(f64, f64)> {
        Ok(self.s.op().port_modal_projections(slice(&y)?, idx))
    }

    /// `A` as CSR `(n, row_ptr, col_idx, values)`.
    fn state_space<'py>(&self, py: Python<'py>) -> (usize, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
        let csr = self.s.op().assemble_sparse();
        let row_ptr: Vec<i64> = csr.row_ptr.iter().map(|&x| x as i64).collect();
        let col_idx: Vec<i64> = csr.col_idx.iter().map(|&x| x as i64).collect();
        (csr.n, row_ptr.into_pyarray(py), col_idx.into_pyarray(py), csr.values.into_pyarray(py))
    }

    /// `A` dense, `[n, n]` (small meshes only).
    fn assemble_dense<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.s.n_dof();
        matrix(py, n, n, self.s.op().assemble_dense())
    }

    /// The energy mass matrix, `½ yᵀ M y` the field energy, dense `[n, n]`.
    fn assemble_energy_mass<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.s.n_dof();
        matrix(py, n, n, self.s.op().assemble_energy_mass())
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

    /// The GPU device name.
    fn gpu_device(&mut self) -> PyResult<String> {
        self.s.gpu_device().map_err(rt)
    }
}
