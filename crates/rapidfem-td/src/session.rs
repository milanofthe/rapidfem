// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The time-domain session: one operator with its stepper workspaces, the
//! lazily built GPU backend and the runs on top of them.
//!
//! Every analysis of `ProblemTD` lands here: free, point-driven and
//! port-driven transients with the exponential, explicit (LSERK4, substepped
//! within the CFL limit) and adaptive (KCL RK4(3)5 with PI step control)
//! integrators, on the CPU or the GPU; the CFL limit, the cavity resonances,
//! the transfer function, the modal port signals and the VTK export.
//!
//! Times are physical: the operator runs in `c·t`, and the waveform is
//! sampled in physical time. The source is held constant over a step
//! (zeroth-order hold): once per output frame for the exponential and the
//! GPU adaptive paths, once per substep for the explicit path and the CPU
//! adaptive controller.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use num_complex::Complex64;

use crate::constants::{
    CFL_BISECT_ITERS, CFL_GROWTH_FACTOR, CFL_GROWTH_RATE_TOL, CFL_POWER_ITERS,
    CFL_PROBE_STEPS, CFL_SAFETY, CFL_Z_STABLE, CFL_Z_UNSTABLE, KCL_ATOL, KCL_ERR_FLOOR,
    KCL_GROWTH_LIMIT, KCL_MIN_STEP_FACTOR, KCL_NONFINITE_ERR, KCL_PI_ALPHA, KCL_PI_BETA,
    KCL_RTOL, KCL_SAFETY, KCL_SHRINK_LIMIT, KRYLOV_TOL, RESONANCE_MERGE_REL,
    RESONANCE_STATIC_FRACTION, RUN_LOG_LINES, TRANSFER_BAND_FRACTION,
};
use crate::explicit::{LserkWorkspace, Source};
use crate::explicit_adaptive::KclWorkspace;
use crate::propagator::KrylovWorkspace;
use crate::rhs::MaxwellOperator;

/// The time integrator of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Krylov exponential propagator, exact for the linear system at any
    /// step; one step per output frame.
    Exponential,
    /// LSERK4, substepped within the CFL limit ([`TdSession::cfl_dt`]).
    Explicit,
    /// KCL RK4(3)5[2R+]C with PI step control ([`Controller`]).
    Adaptive,
}

impl FromStr for Method {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "exponential" => Ok(Method::Exponential),
            "explicit" => Ok(Method::Explicit),
            "adaptive" => Ok(Method::Adaptive),
            _ => Err("method must be 'exponential', 'explicit', or 'adaptive'".into()),
        }
    }
}

/// Where a run executes. The GPU runs the explicit and adaptive paths in
/// f32, the exponential one only on a device with fp64; whatever it cannot
/// run goes to the CPU, with a log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

impl FromStr for Device {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "cpu" => Ok(Device::Cpu),
            "gpu" => Ok(Device::Gpu),
            _ => Err("device must be 'cpu' or 'gpu'".into()),
        }
    }
}

/// The source of `dy/dt = A·y + b·g(t)`.
#[derive(Clone, Copy)]
pub enum Drive<'a> {
    /// No source, `b = 0`.
    Free,
    /// A soft point source, `b = e_dof`.
    Point(usize),
    /// A spatial pattern, e.g. a port mode ([`MaxwellOperator::port_source`]).
    Vector(&'a [f64]),
}

/// What a run keeps per output frame.
#[derive(Clone, Copy)]
pub enum Record<'a> {
    /// The whole state.
    States,
    /// These DOFs only (field probes).
    Dofs(&'a [usize]),
}

/// The PI step-size controller of the adaptive integrator (Söderlind /
/// Gustafsson): the embedded error is weighted against `atol + rtol·|y|`
/// into a scalar norm, a step with norm at most 1 is accepted, and the next
/// step scales with `safety · err^-alpha · prev^beta`.
#[derive(Clone, Copy, Debug)]
pub struct Controller {
    pub atol: f64,
    pub rtol: f64,
    pub safety: f64,
    pub growth_limit: f64,
    pub shrink_limit: f64,
    pub pi_alpha: f64,
    pub pi_beta: f64,
    /// The run fails once the step falls below this fraction of the frame.
    pub min_step_factor: f64,
}

impl Default for Controller {
    fn default() -> Self {
        Controller {
            atol: KCL_ATOL,
            rtol: KCL_RTOL,
            safety: KCL_SAFETY,
            growth_limit: KCL_GROWTH_LIMIT,
            shrink_limit: KCL_SHRINK_LIMIT,
            pi_alpha: KCL_PI_ALPHA,
            pi_beta: KCL_PI_BETA,
            min_step_factor: KCL_MIN_STEP_FACTOR,
        }
    }
}

impl Controller {
    /// `sqrt(mean((err / (atol + rtol·max(|y_old|, |y_new|)))²))`, the
    /// mixed-tolerance norm (Hairer-Wanner); infinite for a non-finite step.
    pub fn err_norm(&self, y_old: &[f64], y_new: &[f64], err: &[f64]) -> f64 {
        let mut s = 0.0;
        for ((&a, &b), &e) in y_old.iter().zip(y_new).zip(err) {
            if !b.is_finite() {
                return f64::INFINITY;
            }
            let r = e / (self.atol + self.rtol * a.abs().max(b.abs()));
            s += r * r;
        }
        (s / err.len().max(1) as f64).sqrt()
    }

    /// The step-size factor. A rejected step drops the previous error (I
    /// control only) so a bad step's history is not carried on.
    pub fn factor(&self, err_norm: f64, prev_err_norm: f64, reject: bool) -> f64 {
        let err = err_norm.max(KCL_ERR_FLOOR);
        let f = if reject || prev_err_norm <= 0.0 {
            self.safety * err.powf(-self.pi_alpha)
        } else {
            self.safety * err.powf(-self.pi_alpha) * prev_err_norm.powf(self.pi_beta)
        };
        if reject {
            f.max(self.shrink_limit)
        } else {
            f.clamp(self.shrink_limit, self.growth_limit)
        }
    }
}

/// The settings of a run.
#[derive(Clone, Copy, Debug)]
pub struct RunOptions {
    pub method: Method,
    pub device: Device,
    /// Krylov dimension cap of the exponential propagator.
    pub krylov_dim: usize,
    /// Output frames run with the exponential propagator before `method`
    /// takes over (not with [`Method::Adaptive`]).
    pub warmup: usize,
    pub verbose: bool,
    pub controller: Controller,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            method: Method::Exponential,
            device: Device::Cpu,
            krylov_dim: 40,
            warmup: 0,
            verbose: true,
            controller: Controller::default(),
        }
    }
}

/// What the adaptive controller did over a run; steps in physical time.
#[derive(Clone, Copy, Debug)]
pub struct KclStats {
    pub accepted: usize,
    pub rejected: usize,
    pub h_min: f64,
    pub h_max: f64,
}

/// The recorded frames of a run, `rows = steps + 1` rows of `width` values
/// (the state, or the probed DOFs), row-major.
#[derive(Debug)]
pub struct Run {
    pub rows: usize,
    pub width: usize,
    pub data: Vec<f64>,
    pub kcl: Option<KclStats>,
}

/// The excitation `g(t)`, sampled in physical time.
pub type Waveform<'a> = dyn FnMut(f64) -> Result<f64, String> + 'a;

/// Called once per output frame; an `Err` aborts the run (an interrupt).
pub type Hook<'a> = dyn FnMut() -> Result<(), String> + 'a;

#[cfg(feature = "gpu")]
struct GpuBackend {
    ctx: crate::gpu::GpuContext,
    op: crate::gpu::GpuOperator,
}

#[cfg(feature = "gpu")]
enum GpuSlot {
    Untried,
    Ready(Box<GpuBackend>),
    Unavailable(String),
}

/// The operator with its workspaces and the GPU backend, see the module
/// docs.
pub struct TdSession {
    op: MaxwellOperator,
    c: f64,
    krylov: KrylovWorkspace,
    lserk: LserkWorkspace,
    kcl: KclWorkspace,
    /// The stable explicit step in operator time, once computed.
    cfl_h: Option<f64>,
    #[cfg(feature = "gpu")]
    gpu: GpuSlot,
}

fn log(verbose: bool, msg: impl AsRef<str>) {
    if verbose {
        eprintln!("  [rapidfem-td] {}", msg.as_ref());
    }
}

/// `b·g` into `src` (for a point source only the one entry is written; the
/// caller keeps the rest zero).
fn fill_source(drive: Drive, g: f64, src: &mut [f64]) {
    match drive {
        Drive::Free => {}
        Drive::Point(d) => src[d] = g,
        Drive::Vector(b) => {
            for (s, &bi) in src.iter_mut().zip(b) {
                *s = bi * g;
            }
        }
    }
}

/// The Runge-Kutta stepper source `b·g(t)` of `drive`, `None` for a free
/// run (the waveform is then not sampled). A vector drive is scaled into
/// `src`.
fn rk_source<'s>(
    drive: Drive,
    t: f64,
    wave: &mut dyn FnMut(f64) -> Result<f64, String>,
    src: &'s mut [f64],
) -> Result<Option<Source<'s>>, String> {
    Ok(match drive {
        Drive::Free => None,
        Drive::Point(dof) => Some(Source::Point { dof, value: wave(t)? }),
        Drive::Vector(_) => {
            fill_source(drive, wave(t)?, src);
            Some(Source::Vector(src))
        }
    })
}

/// Frame recorder.
struct Recorder<'a> {
    record: Record<'a>,
    width: usize,
    data: Vec<f64>,
}

impl<'a> Recorder<'a> {
    fn new(record: Record<'a>, n: usize, rows: usize) -> Self {
        let width = match record {
            Record::States => n,
            Record::Dofs(d) => d.len(),
        };
        Recorder { record, width, data: Vec::with_capacity(rows * width) }
    }

    fn push(&mut self, y: &[f64]) {
        match self.record {
            Record::States => self.data.extend_from_slice(y),
            Record::Dofs(d) => self.data.extend(d.iter().map(|&i| y[i])),
        }
    }
}

/// Progress lines, one every `steps / RUN_LOG_LINES` frames.
struct Progress {
    verbose: bool,
    label: &'static str,
    steps: usize,
    every: usize,
    t0: Instant,
}

impl Progress {
    fn new(verbose: bool, label: &'static str, steps: usize) -> Self {
        let every = (steps / RUN_LOG_LINES).max(1);
        Progress { verbose, label, steps, every, t0: Instant::now() }
    }

    fn frame(&self, done: usize) {
        if self.verbose && done.is_multiple_of(self.every) && done < self.steps {
            let el = self.t0.elapsed().as_secs_f64();
            let eta = el / done as f64 * (self.steps - done) as f64;
            log(true, format!(
                "{} {done}/{}  ({el:.1}s elapsed, ETA {eta:.0}s)",
                self.label, self.steps
            ));
        }
    }

    fn done(&self) {
        log(self.verbose, format!(
            "{} complete - {} steps in {:.1}s",
            self.label,
            self.steps,
            self.t0.elapsed().as_secs_f64()
        ));
    }
}

/// The adaptive controller's state across frames.
struct KclState {
    /// Current step, physical time.
    h: f64,
    prev_err: f64,
    stats: KclStats,
}

impl KclState {
    fn new(dt: f64) -> Self {
        KclState {
            h: dt,
            prev_err: 0.0,
            stats: KclStats { accepted: 0, rejected: 0, h_min: f64::INFINITY, h_max: 0.0 },
        }
    }
}

/// Deterministic standard-normal samples (splitmix64 and Box-Muller), the
/// power iteration's start vector.
fn normal_vector(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u, v) = (uniform(), uniform());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        })
        .collect()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

impl TdSession {
    /// A session on `op`, with `c` the speed of light in the mesh's length
    /// unit (1 for a normalised operator).
    pub fn new(op: MaxwellOperator, c: f64) -> Self {
        TdSession {
            op,
            c,
            krylov: KrylovWorkspace::new(),
            lserk: LserkWorkspace::new(),
            kcl: KclWorkspace::new(),
            cfl_h: None,
            #[cfg(feature = "gpu")]
            gpu: GpuSlot::Untried,
        }
    }

    pub fn op(&self) -> &MaxwellOperator {
        &self.op
    }

    pub fn n_dof(&self) -> usize {
        self.op.n_dof()
    }

    fn check_len(&self, what: &str, len: usize) -> Result<(), String> {
        let n = self.n_dof();
        if len != n {
            return Err(format!("{what} has {len} entries, expected n_dof = {n}"));
        }
        Ok(())
    }

    // ── single steps ─────────────────────────────────────────────────────

    /// `exp(h·A)·y` by the Krylov propagator, `tol` its a-posteriori
    /// tolerance (0 runs the full `krylov_dim`).
    pub fn step(&mut self, y: &[f64], h: f64, krylov_dim: usize, tol: f64) -> Result<Vec<f64>, String> {
        self.check_len("y", y.len())?;
        let mut out = vec![0.0; y.len()];
        let op = &self.op;
        self.krylov.expmv_into(|x, ax| op.apply_into(x, ax), y, self.c * h, krylov_dim, tol, &mut out);
        Ok(out)
    }

    /// One LSERK4 step of `h` (conditionally stable).
    pub fn step_explicit(&mut self, y: &[f64], h: f64) -> Result<Vec<f64>, String> {
        self.check_len("y", y.len())?;
        let mut out = y.to_vec();
        let op = &self.op;
        self.lserk.step_into(|x, ax| op.apply_into(x, ax), &mut out, self.c * h, None);
        Ok(out)
    }

    /// One KCL step of `h`: the fourth-order state and the embedded error.
    pub fn step_adaptive(&mut self, y: &[f64], h: f64) -> Result<(Vec<f64>, Vec<f64>), String> {
        self.check_len("y", y.len())?;
        let mut out = y.to_vec();
        let mut err = vec![0.0; y.len()];
        let op = &self.op;
        self.kcl.step_into(|x, ax| op.apply_into(x, ax), &mut out, &mut err, self.c * h, None);
        Ok((out, err))
    }

    // ── the explicit stability limit ─────────────────────────────────────

    /// The largest stable LSERK4 step (physical time), cached: the spectral
    /// radius by power iteration, then the stable `z = h·ρ` bracketed by
    /// probe runs from the dominant eigenvector, which must keep the
    /// per-step amplification at most `1 + CFL_GROWTH_RATE_TOL`.
    pub fn cfl_dt(&mut self, recompute: bool) -> f64 {
        if let (Some(h), false) = (self.cfl_h, recompute) {
            return h / self.c;
        }
        let n = self.n_dof();
        let mut v = normal_vector(n, 0);
        let s = 1.0 / norm(&v);
        v.iter_mut().for_each(|x| *x *= s);
        let mut av = vec![0.0; n];
        let mut rho = 1.0;
        for _ in 0..CFL_POWER_ITERS {
            self.op.apply_into(&v, &mut av);
            rho = norm(&av);
            for (vi, &a) in v.iter_mut().zip(&av) {
                *vi = a / rho;
            }
        }
        let probe = v;
        let n0 = norm(&probe);
        let op = &self.op;
        let lserk = &mut self.lserk;
        let mut stable = |z: f64| {
            let h = z / rho;
            let mut y = probe.clone();
            for _ in 0..CFL_PROBE_STEPS {
                lserk.step_into(|x, ax| op.apply_into(x, ax), &mut y, h, None);
            }
            if !y.iter().all(|x| x.is_finite()) {
                return false;
            }
            let nf = norm(&y);
            if nf > CFL_GROWTH_FACTOR * n0 {
                return false;
            }
            let rate = if nf > 0.0 { (nf / n0).powf(1.0 / CFL_PROBE_STEPS as f64) } else { 0.0 };
            rate <= 1.0 + CFL_GROWTH_RATE_TOL
        };
        let (mut lo, mut hi) = (CFL_Z_STABLE, CFL_Z_UNSTABLE);
        while !stable(lo) && lo > 0.1 {
            lo *= 0.5; // stiffer than expected
        }
        while stable(hi) && hi < 1e3 {
            hi *= 1.5; // dissipation widens the range
        }
        for _ in 0..CFL_BISECT_ITERS {
            let mid = 0.5 * (lo + hi);
            if stable(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let h = CFL_SAFETY * lo / rho;
        self.cfl_h = Some(h);
        h / self.c
    }

    // ── GPU ──────────────────────────────────────────────────────────────

    /// The GPU device name, or why there is none.
    pub fn gpu_device(&mut self) -> Result<String, String> {
        #[cfg(feature = "gpu")]
        {
            self.gpu_backend().map(|b| b.ctx.device_name.clone())
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err("built without GPU support".into())
        }
    }

    /// Builds the GPU backend on first use. A machine without an OpenCL
    /// loader can panic inside the driver layer instead of failing; that
    /// counts as no GPU.
    #[cfg(feature = "gpu")]
    fn gpu_backend(&mut self) -> Result<&mut GpuBackend, String> {
        if let GpuSlot::Untried = self.gpu {
            let op = &self.op;
            let built = if op.n_dispersive() != 0 {
                Err("the GPU path does not cover dispersive materials".to_string())
            } else {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let ctx = crate::gpu::GpuContext::new()?;
                    let gop = crate::gpu::GpuOperator::new(&ctx, op)?;
                    Ok(GpuBackend { ctx, op: gop })
                }))
                .unwrap_or_else(|_| Err("the OpenCL runtime panicked".to_string()))
            };
            self.gpu = match built {
                Ok(b) => GpuSlot::Ready(Box::new(b)),
                Err(e) => GpuSlot::Unavailable(e),
            };
        }
        match &mut self.gpu {
            GpuSlot::Ready(b) => Ok(b),
            GpuSlot::Unavailable(e) => Err(e.clone()),
            GpuSlot::Untried => unreachable!(),
        }
    }

    // ── runs ─────────────────────────────────────────────────────────────

    /// Runs `steps` output frames of `dt` from `y0` (zero if `None`), driven
    /// by `drive·waveform(t)`, keeping `record` per frame (`steps + 1`
    /// rows, the first the initial state). `hook` runs once per frame (and
    /// per GPU chunk); its error aborts the run.
    pub fn transient(
        &mut self,
        y0: Option<&[f64]>,
        dt: f64,
        steps: usize,
        drive: Drive,
        mut waveform: Option<&mut Waveform>,
        record: Record,
        opts: &RunOptions,
        hook: &mut Hook,
    ) -> Result<Run, String> {
        let n = self.n_dof();
        if !(dt.is_finite() && dt > 0.0) {
            return Err(format!("dt must be positive, got {dt}"));
        }
        match drive {
            Drive::Free => {}
            Drive::Point(d) if d >= n => return Err(format!("source DOF {d} out of range (n_dof = {n})")),
            Drive::Vector(b) => self.check_len("the source vector", b.len())?,
            _ => {}
        }
        if !matches!(drive, Drive::Free) && waveform.is_none() {
            return Err("a driven run needs a waveform".into());
        }
        if let Record::Dofs(d) = record
            && let Some(&bad) = d.iter().find(|&&i| i >= n) {
                return Err(format!("probe DOF {bad} out of range (n_dof = {n})"));
            }
        let warmup = opts.warmup.min(steps);
        if opts.method == Method::Adaptive && warmup > 0 {
            return Err("warmup is not supported with method='adaptive' (the controller \
                        stabilises itself in the first few frames)"
                .into());
        }
        let mut y = match y0 {
            Some(y0) => {
                self.check_len("y0", y0.len())?;
                y0.to_vec()
            }
            None => vec![0.0; n],
        };
        let mut wave = |t: f64| -> Result<f64, String> {
            match waveform.as_mut() {
                Some(w) => w(t),
                None => Ok(0.0),
            }
        };
        let label = if matches!(drive, Drive::Free) { "transient" } else { "driven transient" };
        let mut rec = Recorder::new(record, n, steps + 1);
        rec.push(&y);

        let (gpu, gpu_fp64) = self.gpu_route(opts);
        let exp_on_gpu = gpu && gpu_fp64;
        let explicit_nsub = if opts.method == Method::Explicit {
            let cfl = self.cfl_dt(false);
            (dt / cfl).ceil().max(1.0) as usize
        } else {
            1
        };
        match opts.method {
            Method::Exponential => log(opts.verbose, format!(
                "{label}: exponential ({})",
                if exp_on_gpu { "GPU" } else { "CPU" }
            )),
            Method::Explicit => log(opts.verbose, format!(
                "{label}: {}explicit LSERK4 on the {} ({explicit_nsub} substeps/step)",
                if warmup > 0 { format!("{warmup} exponential warmup step(s), then ") } else { String::new() },
                if gpu { "GPU" } else { "CPU" }
            )),
            Method::Adaptive => log(opts.verbose, format!(
                "{label}: KCL adaptive on the {} (atol={:e}, rtol={:e})",
                if gpu { "GPU" } else { "CPU" },
                opts.controller.atol,
                opts.controller.rtol
            )),
        }
        let progress = Progress::new(opts.verbose, label, steps);

        let mut src = vec![0.0; n];
        let mut out = vec![0.0; n];
        let exp_frames = if opts.method == Method::Exponential { steps } else { warmup };
        for k in 0..exp_frames {
            let t = k as f64 * dt;
            let g = if matches!(drive, Drive::Free) { 0.0 } else { wave(t)? };
            fill_source(drive, g, &mut src);
            if exp_on_gpu {
                y = self.gpu_exponential(&y, drive, &src, dt, opts.krylov_dim)?;
            } else {
                let op = &self.op;
                let h = self.c * dt;
                match drive {
                    Drive::Free => {
                        self.krylov.expmv_into(|x, ax| op.apply_into(x, ax), &y, h, opts.krylov_dim, KRYLOV_TOL, &mut out);
                    }
                    _ => self.krylov.etd_step_into(
                        |x, ax| op.apply_into(x, ax), &y, &src, h, opts.krylov_dim, KRYLOV_TOL, &mut out,
                    ),
                }
                std::mem::swap(&mut y, &mut out);
            }
            if let Drive::Point(d) = drive {
                src[d] = 0.0;
            }
            rec.push(&y);
            hook()?;
            progress.frame(k + 1);
        }

        let mut kcl = None;
        match opts.method {
            Method::Exponential => {}
            Method::Explicit if gpu => {
                self.gpu_explicit(&mut y, warmup, steps, dt, explicit_nsub, drive, &mut wave, &mut rec, &progress, hook)?;
            }
            Method::Explicit => {
                let h_sub = dt / explicit_nsub as f64;
                let h_op = self.c * h_sub;
                let op = &self.op;
                for k in warmup..steps {
                    let t = k as f64 * dt;
                    for j in 0..explicit_nsub {
                        let source = rk_source(drive, t + j as f64 * h_sub, &mut wave, &mut src)?;
                        self.lserk.step_into(|x, ax| op.apply_into(x, ax), &mut y, h_op, source);
                    }
                    rec.push(&y);
                    hook()?;
                    progress.frame(k + 1);
                }
            }
            Method::Adaptive if gpu => {
                kcl = Some(self.gpu_adaptive(&mut y, steps, dt, drive, &mut wave, &opts.controller, &mut rec)?);
                hook()?;
            }
            Method::Adaptive => {
                let mut state = KclState::new(dt);
                let mut y_try = vec![0.0; n];
                let mut err = vec![0.0; n];
                for k in 0..steps {
                    self.adaptive_frame(
                        &mut y, &mut y_try, &mut err, &mut src, k as f64 * dt, dt, drive, &mut wave,
                        &opts.controller, &mut state,
                    )?;
                    rec.push(&y);
                    hook()?;
                    progress.frame(k + 1);
                }
                kcl = Some(state.stats);
            }
        }
        progress.done();
        if let Some(s) = &kcl {
            log(opts.verbose, format!(
                "  KCL controller: {} accepted, {} rejected; h in [{:.3e}, {:.3e}] s",
                s.accepted, s.rejected, s.h_min, s.h_max
            ));
        }
        Ok(Run { rows: steps + 1, width: rec.width, data: rec.data, kcl })
    }

    /// One output frame of the CPU adaptive controller: as many KCL
    /// substeps as it takes to cover `dt`, the waveform sampled per substep.
    fn adaptive_frame(
        &mut self,
        y: &mut Vec<f64>,
        y_try: &mut Vec<f64>,
        err: &mut [f64],
        src: &mut [f64],
        t0: f64,
        dt: f64,
        drive: Drive,
        wave: &mut dyn FnMut(f64) -> Result<f64, String>,
        ctl: &Controller,
        s: &mut KclState,
    ) -> Result<(), String> {
        let op = &self.op;
        let h_floor = ctl.min_step_factor * dt;
        let mut t_rel = 0.0;
        while t_rel < dt {
            let h_try = s.h.min(dt - t_rel);
            let h_op = self.c * h_try;
            y_try.copy_from_slice(y);
            let source = rk_source(drive, t0 + t_rel, wave, src)?;
            self.kcl.step_into(|x, ax| op.apply_into(x, ax), y_try, err, h_op, source);
            let e = ctl.err_norm(y, y_try, err);
            if e.is_finite() && e <= 1.0 {
                std::mem::swap(y, y_try);
                t_rel += h_try;
                s.stats.accepted += 1;
                s.h = h_try * ctl.factor(e, s.prev_err, false);
                s.prev_err = e.max(KCL_ERR_FLOOR);
                s.stats.h_min = s.stats.h_min.min(s.h);
                s.stats.h_max = s.stats.h_max.max(s.h);
            } else {
                s.stats.rejected += 1;
                let probe = if e.is_finite() { e } else { KCL_NONFINITE_ERR };
                s.h = h_try * ctl.factor(probe, s.prev_err, true);
            }
            if s.h < h_floor {
                return Err(format!(
                    "adaptive stepper: step size collapsed below {:e}·dt after {} accepted, \
                     {} rejected substeps. The operator is likely too stiff for the chosen \
                     tolerances (atol={:e}, rtol={:e}).",
                    ctl.min_step_factor, s.stats.accepted, s.stats.rejected, ctl.atol, ctl.rtol
                ));
            }
        }
        Ok(())
    }

    /// Whether this run goes to the GPU, and whether that has fp64; logs
    /// every fallback to the CPU.
    fn gpu_route(&mut self, opts: &RunOptions) -> (bool, bool) {
        if opts.device != Device::Gpu {
            return (false, false);
        }
        #[cfg(feature = "gpu")]
        {
            let needs_exp = opts.method == Method::Exponential || opts.warmup > 0;
            match self.gpu_backend() {
                Ok(b) => {
                    let fp64 = b.ctx.fp64;
                    if needs_exp && !fp64 {
                        log(opts.verbose, format!(
                            "GPU {} has no fp64: the exponential propagator runs on the CPU",
                            b.ctx.device_name
                        ));
                    }
                    if opts.method == Method::Exponential {
                        (fp64, fp64)
                    } else {
                        (true, fp64)
                    }
                }
                Err(e) => {
                    log(opts.verbose, format!("no GPU ({e}), running on the CPU"));
                    (false, false)
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            log(opts.verbose, "built without GPU support, running on the CPU");
            (false, false)
        }
    }

    #[cfg(feature = "gpu")]
    fn gpu_exponential(&mut self, y: &[f64], drive: Drive, src: &[f64], dt: f64, m: usize) -> Result<Vec<f64>, String> {
        let h = self.c * dt;
        let b = self.gpu_backend()?;
        match drive {
            Drive::Free => b.op.expmv(&b.ctx, y, h, m),
            _ => b.op.etd_step(&b.ctx, y, src, h, m),
        }
    }

    #[cfg(not(feature = "gpu"))]
    fn gpu_exponential(&mut self, _: &[f64], _: Drive, _: &[f64], _: f64, _: usize) -> Result<Vec<f64>, String> {
        unreachable!("routed to the CPU without the gpu feature")
    }

    /// Explicit frames `k0..steps` on the GPU, in chunks so progress and
    /// interrupts come through; only the chunk-boundary state crosses the
    /// bus.
    #[cfg(feature = "gpu")]
    fn gpu_explicit(
        &mut self,
        y: &mut Vec<f64>,
        k0: usize,
        steps: usize,
        dt: f64,
        nsub: usize,
        drive: Drive,
        wave: &mut dyn FnMut(f64) -> Result<f64, String>,
        rec: &mut Recorder,
        progress: &Progress,
        hook: &mut Hook,
    ) -> Result<(), String> {
        let n = y.len();
        let h = (self.c * dt) as f32;
        let h_sub = dt / nsub as f64;
        let chunk = ((steps - k0) / RUN_LOG_LINES).max(1);
        let mut done = k0;
        while done < steps {
            let kk = chunk.min(steps - done);
            let mut g = Vec::new();
            if !matches!(drive, Drive::Free) {
                g.reserve(kk * nsub);
                for i in 0..kk * nsub {
                    g.push(wave((done * nsub + i) as f64 * h_sub)? as f32);
                }
            }
            let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
            let b = self.gpu_backend()?;
            let traj = b.op.transient(&b.ctx, &y32, h, kk, nsub, drive, &g, true)?;
            for r in 1..=kk {
                let row: Vec<f64> = traj[r * n..(r + 1) * n].iter().map(|&v| v as f64).collect();
                rec.push(&row);
                if r == kk {
                    *y = row;
                }
            }
            done += kk;
            hook()?;
            progress.frame(done);
        }
        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    fn gpu_explicit(
        &mut self, _: &mut Vec<f64>, _: usize, _: usize, _: f64, _: usize, _: Drive,
        _: &mut dyn FnMut(f64) -> Result<f64, String>, _: &mut Recorder, _: &Progress, _: &mut Hook,
    ) -> Result<(), String> {
        unreachable!("routed to the CPU without the gpu feature")
    }

    /// The adaptive run on the GPU: the controller on the host, the state
    /// and the error reduction on the device, the waveform held per frame.
    #[cfg(feature = "gpu")]
    fn gpu_adaptive(
        &mut self,
        y: &mut Vec<f64>,
        steps: usize,
        dt: f64,
        drive: Drive,
        wave: &mut dyn FnMut(f64) -> Result<f64, String>,
        ctl: &Controller,
        rec: &mut Recorder,
    ) -> Result<KclStats, String> {
        let n = y.len();
        let c = self.c;
        let h = (c * dt) as f32;
        let mut g = Vec::new();
        if !matches!(drive, Drive::Free) {
            for k in 0..steps {
                g.push(wave(k as f64 * dt)? as f32);
            }
        }
        let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let b = self.gpu_backend()?;
        let (traj, acc, rej, h_min, h_max) = b.op.transient_kcl(&b.ctx, &y32, h, steps, drive, &g, ctl)?;
        for r in 1..=steps {
            let row: Vec<f64> = traj[r * n..(r + 1) * n].iter().map(|&v| v as f64).collect();
            rec.push(&row);
            if r == steps {
                *y = row;
            }
        }
        Ok(KclStats { accepted: acc, rejected: rej, h_min: h_min as f64 / c, h_max: h_max as f64 / c })
    }

    #[cfg(not(feature = "gpu"))]
    fn gpu_adaptive(
        &mut self, _: &mut Vec<f64>, _: usize, _: f64, _: Drive,
        _: &mut dyn FnMut(f64) -> Result<f64, String>, _: &Controller, _: &mut Recorder,
    ) -> Result<KclStats, String> {
        unreachable!("routed to the CPU without the gpu feature")
    }

    /// The field-to-field transfer function `R(f)/G(f)`: a run driving a
    /// point source with `pulse` and probing one DOF, both spectra by real
    /// FFT, `H` zero where the drive spectrum is below
    /// `TRANSFER_BAND_FRACTION` of its peak. Returns the frequencies (Hz
    /// for an SI operator) and `H`.
    pub fn transfer_function(
        &mut self,
        source: usize,
        probe: usize,
        pulse: &mut Waveform,
        dt: f64,
        steps: usize,
        opts: &RunOptions,
        hook: &mut Hook,
    ) -> Result<(Vec<f64>, Vec<Complex64>), String> {
        let probes = [probe];
        let run = self.transient(None, dt, steps, Drive::Point(source), Some(&mut *pulse), Record::Dofs(&probes), opts, hook)?;
        let g = (0..=steps).map(|k| pulse(k as f64 * dt)).collect::<Result<Vec<f64>, String>>()?;
        Ok(transfer(&g, &run.data, dt))
    }

    // ── spectrum and post-processing ─────────────────────────────────────

    /// The `n` lowest distinct cavity resonances (Hz for an SI operator)
    /// from the dense spectrum: the near-static modes dropped, the
    /// least-damped eigenvalues first, `f = c·|Im λ| / 2π`. Dense, so for
    /// modest meshes only.
    pub fn resonances(&self, n: usize) -> Result<Vec<f64>, String> {
        let m = self.n_dof();
        let a = self.op.assemble_dense();
        let mat = faer::Mat::from_fn(m, m, |i, j| a[i * m + j]);
        let ev = mat.eigenvalues().map_err(|e| format!("eigenvalues: {e:?}"))?;
        let w_max = ev.iter().map(|z| z.im.abs()).fold(0.0, f64::max);
        let mut phys: Vec<_> = ev.iter().filter(|z| z.im.abs() > RESONANCE_STATIC_FRACTION * w_max).collect();
        phys.sort_by(|a, b| b.re.total_cmp(&a.re));
        let mut out: Vec<f64> = Vec::new();
        for z in phys {
            let f = z.im.abs() * self.c / (2.0 * std::f64::consts::PI);
            if out.iter().any(|&g| (f - g).abs() <= RESONANCE_MERGE_REL * g) {
                continue;
            }
            out.push(f);
            if out.len() >= n {
                break;
            }
        }
        out.sort_by(f64::total_cmp);
        Ok(out)
    }

    /// The operator index of the `k`-th modal port (the ports carrying a
    /// mode, in declaration order; absorbing-only faces skipped).
    pub fn modal_port(&self, k: usize) -> Result<usize, String> {
        (0..self.op.n_ports())
            .filter(|&p| self.op.port_has_mode(p))
            .nth(k)
            .ok_or_else(|| format!("modal port {k} does not exist on the operator"))
    }

    /// The modal amplitude `P_e` of each port over a row-major trajectory,
    /// `[ports][rows]`.
    pub fn port_signals(&self, states: &[f64], ports: &[usize]) -> Result<Vec<f64>, String> {
        let n = self.n_dof();
        if !states.len().is_multiple_of(n) {
            return Err(format!("the trajectory length {} is not a multiple of n_dof = {n}", states.len()));
        }
        if let Some(&p) = ports.iter().find(|&&p| p >= self.op.n_ports() || !self.op.port_has_mode(p)) {
            return Err(format!("port {p} is not a modal port of the operator"));
        }
        let mut out = Vec::with_capacity(ports.len() * states.len() / n);
        for &p in ports {
            out.extend(states.chunks_exact(n).map(|y| self.op.port_modal_projections(y, p).0));
        }
        Ok(out)
    }

    /// Writes a trajectory as `<base>_NNNN.vtu` per frame (discontinuous
    /// linear tets, `E` and `H` at the element corners) and the
    /// `<base>.pvd` collection over `times`; returns the `.pvd` path.
    pub fn export_vtk(&self, states: &[f64], times: &[f64], base: &Path) -> io::Result<PathBuf> {
        let n = self.n_dof();
        let invalid = |m: String| io::Error::new(io::ErrorKind::InvalidInput, m);
        if !states.len().is_multiple_of(n) {
            return Err(invalid(format!("states carry {} values, not a multiple of n_dof = {n}", states.len())));
        }
        let n_snap = states.len() / n;
        if times.len() != n_snap {
            return Err(invalid(format!("times has {} entries, expected {n_snap}", times.len())));
        }
        let coords = self.op.node_coords();
        let n_elem = self.op.n_elem();
        let np = coords.len() / n_elem;
        let corners = self.op.corner_local_nodes();
        if let Some(parent) = base.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let stem = base.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let mut entries = Vec::with_capacity(n_snap);
        for (s, y) in states.chunks_exact(n).enumerate() {
            let name = format!("{stem}_{s:04}.vtu");
            let mut w = BufWriter::new(File::create(base.with_file_name(&name))?);
            writeln!(w, "<?xml version=\"1.0\"?>")?;
            writeln!(w, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">")?;
            writeln!(w, "  <UnstructuredGrid>")?;
            writeln!(w, "    <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{n_elem}\">", 4 * n_elem)?;
            writeln!(w, "      <Points>")?;
            writeln!(w, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">")?;
            for e in 0..n_elem {
                for &c in &corners {
                    let p = coords[e * np + c];
                    writeln!(w, "          {} {} {}", p[0], p[1], p[2])?;
                }
            }
            writeln!(w, "        </DataArray>")?;
            writeln!(w, "      </Points>")?;
            writeln!(w, "      <Cells>")?;
            writeln!(w, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">")?;
            for e in 0..n_elem {
                let b = 4 * e;
                writeln!(w, "          {} {} {} {}", b, b + 1, b + 2, b + 3)?;
            }
            writeln!(w, "        </DataArray>")?;
            writeln!(w, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">")?;
            for e in 0..n_elem {
                writeln!(w, "          {}", 4 * (e + 1))?;
            }
            writeln!(w, "        </DataArray>")?;
            writeln!(w, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">")?;
            for _ in 0..n_elem {
                writeln!(w, "          10")?; // VTK_TETRA
            }
            writeln!(w, "        </DataArray>")?;
            writeln!(w, "      </Cells>")?;
            writeln!(w, "      <PointData>")?;
            for (name, off) in [("E", 0), ("H", 3)] {
                writeln!(w, "        <DataArray type=\"Float64\" Name=\"{name}\" NumberOfComponents=\"3\" format=\"ascii\">")?;
                for e in 0..n_elem {
                    for &c in &corners {
                        let i = (e * np + c) * 6 + off;
                        writeln!(w, "          {} {} {}", y[i], y[i + 1], y[i + 2])?;
                    }
                }
                writeln!(w, "        </DataArray>")?;
            }
            writeln!(w, "      </PointData>")?;
            writeln!(w, "    </Piece>")?;
            writeln!(w, "  </UnstructuredGrid>")?;
            writeln!(w, "</VTKFile>")?;
            w.flush()?;
            entries.push((times[s], name));
        }
        let pvd = base.with_file_name(format!("{stem}.pvd"));
        let mut w = BufWriter::new(File::create(&pvd)?);
        writeln!(w, "<?xml version=\"1.0\"?>")?;
        writeln!(w, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">")?;
        writeln!(w, "  <Collection>")?;
        for (t, name) in &entries {
            writeln!(w, "    <DataSet timestep=\"{t}\" file=\"{name}\"/>")?;
        }
        writeln!(w, "  </Collection>")?;
        writeln!(w, "</VTKFile>")?;
        w.flush()?;
        Ok(pvd)
    }
}

/// `R(f)/G(f)` of two equally sampled signals at spacing `dt`, by real FFT
/// on the `numpy.fft.rfftfreq` grid; zero where `|G|` is below
/// `TRANSFER_BAND_FRACTION` of its peak.
pub fn transfer(g: &[f64], r: &[f64], dt: f64) -> (Vec<f64>, Vec<Complex64>) {
    assert_eq!(g.len(), r.len(), "drive and response lengths differ");
    let n = g.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut planner = realfft::RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);
    let spectrum = |x: &[f64]| {
        let mut input = x.to_vec();
        let mut out = fft.make_output_vec();
        fft.process(&mut input, &mut out).expect("FFT buffer sizes");
        out
    };
    let (sg, sr) = (spectrum(g), spectrum(r));
    let peak = sg.iter().map(|z| z.norm()).fold(0.0, f64::max);
    let freqs = (0..sg.len()).map(|k| k as f64 / (n as f64 * dt)).collect();
    let h = sg
        .iter()
        .zip(&sr)
        .map(|(&a, &b)| if a.norm() > TRANSFER_BAND_FRACTION * peak { b / a } else { Complex64::new(0.0, 0.0) })
        .collect();
    (freqs, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_gen::structured_box;

    fn session() -> TdSession {
        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        TdSession::new(MaxwellOperator::new(&mesh, 2, 1.0, Default::default()), 1.0)
    }

    fn opts(method: Method) -> RunOptions {
        RunOptions { method, verbose: false, ..Default::default() }
    }

    fn rel(a: &[f64], b: &[f64]) -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();
        d / norm(b)
    }

    fn run(s: &mut TdSession, y0: Option<&[f64]>, drive: Drive, o: &RunOptions) -> Run {
        let mut pulse = |t: f64| Ok((-((t - 1.0) / 0.3_f64).powi(2)).exp());
        let wave: Option<&mut Waveform> = if matches!(drive, Drive::Free) { None } else { Some(&mut pulse) };
        s.transient(y0, 0.05, 12, drive, wave, Record::States, o, &mut || Ok(())).expect("run")
    }

    #[test]
    fn integrators_agree_on_free_and_driven_runs() {
        let mut s = session();
        let n = s.n_dof();
        let y0 = normal_vector(n, 7);
        let d = s.op().nearest_node_dof([0.4, 0.5, 0.5], 0, 2);
        for drive in [Drive::Free, Drive::Point(d)] {
            let y0 = if matches!(drive, Drive::Free) { Some(&y0[..]) } else { None };
            let exp = run(&mut s, y0, drive, &opts(Method::Exponential));
            assert_eq!((exp.rows, exp.width), (13, n));
            let last = |r: &Run| r.data[12 * n..].to_vec();
            for m in [Method::Explicit, Method::Adaptive] {
                let r = run(&mut s, y0, drive, &opts(m));
                let e = rel(&last(&r), &last(&exp));
                // LSERK4 at the CFL step carries a truncation error on the
                // random start; driven, the source is held per substep
                // (explicit) or per adaptive step against per frame
                // (exponential), which only bounds the adaptive run loosely
                let tol = match (drive, m) {
                    (Drive::Free, Method::Explicit) => 1e-2,
                    (Drive::Free, _) => 1e-4,
                    (_, Method::Explicit) => 5e-2,
                    _ => 0.5,
                };
                assert!(e < tol, "{m:?} vs exponential: rel {e:.2e}");
                if m == Method::Adaptive {
                    assert!(r.kcl.expect("stats").accepted >= 12);
                }
            }
        }
    }

    #[test]
    fn warmup_and_probes() {
        let mut s = session();
        let n = s.n_dof();
        let y0 = normal_vector(n, 3);
        let full = run(&mut s, Some(&y0), Drive::Free, &RunOptions { warmup: 3, ..opts(Method::Explicit) });
        let probes = [5, 17];
        let p = s
            .transient(Some(&y0), 0.05, 12, Drive::Free, None, Record::Dofs(&probes), &RunOptions { warmup: 3, ..opts(Method::Explicit) }, &mut || Ok(()))
            .expect("probe run");
        assert_eq!(p.width, 2);
        for r in 0..13 {
            assert_eq!(p.data[2 * r], full.data[r * n + 5]);
            assert_eq!(p.data[2 * r + 1], full.data[r * n + 17]);
        }
        let bad = s.transient(None, 0.05, 2, Drive::Free, None, Record::States, &RunOptions { warmup: 1, ..opts(Method::Adaptive) }, &mut || Ok(()));
        assert!(bad.is_err());
    }

    #[test]
    fn hook_aborts_the_run() {
        let mut s = session();
        let mut calls = 0;
        let mut hook = || {
            calls += 1;
            if calls == 3 { Err("interrupted".to_string()) } else { Ok(()) }
        };
        let r = s.transient(None, 0.05, 10, Drive::Free, None, Record::States, &opts(Method::Exponential), &mut hook);
        assert_eq!(r.unwrap_err(), "interrupted");
    }

    #[test]
    fn cfl_step_is_stable_and_cached() {
        let mut s = session();
        let h = s.cfl_dt(false);
        assert!(h > 0.0 && h.is_finite());
        assert_eq!(s.cfl_dt(false), h);
        let mut y = normal_vector(s.n_dof(), 1);
        let n0 = norm(&y);
        for _ in 0..500 {
            y = s.step_explicit(&y, h).unwrap();
        }
        assert!(norm(&y) <= 1.01 * n0, "growth at the CFL step");
    }

    #[test]
    fn resonance_of_the_unit_cube() {
        // TE/TM 110 of the unit PEC cube, c = 1: f = sqrt(2)/2.
        let s = session();
        let f = s.resonances(3).expect("resonances");
        assert_eq!(f.len(), 3);
        let want = 2.0_f64.sqrt() / 2.0;
        assert!((f[0] - want).abs() / want < 0.02, "fundamental {:.4}, want {want:.4}", f[0]);
        assert!(f.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn transfer_of_a_delayed_copy() {
        let dt = 0.01;
        let g: Vec<f64> = (0..256).map(|k| (-((k as f64 * dt - 0.5) / 0.05).powi(2)).exp()).collect();
        let r: Vec<f64> = (0..256).map(|k| 2.0 * g[k.max(3) - 3]).collect();
        let (f, h) = transfer(&g, &r, dt);
        assert_eq!(f.len(), 129);
        assert!((f[1] - 1.0 / (256.0 * dt)).abs() < 1e-12);
        // in band, H = 2·exp(-j·2π·f·3dt)
        for k in 1..10 {
            let want = Complex64::from_polar(2.0, -2.0 * std::f64::consts::PI * f[k] * 3.0 * dt);
            assert!((h[k] - want).norm() < 1e-6, "k={k}: {} vs {want}", h[k]);
        }
        assert_eq!(h[128], Complex64::new(0.0, 0.0));
    }

    #[test]
    fn vtk_series() {
        let s = session();
        let n = s.n_dof();
        let states = normal_vector(2 * n, 5);
        let dir = std::env::temp_dir().join(format!("rapidfem_td_vtk_{}", std::process::id()));
        let pvd = s.export_vtk(&states, &[0.0, 0.5], &dir.join("field")).expect("export");
        let text = std::fs::read_to_string(&pvd).unwrap();
        assert!(text.contains("file=\"field_0001.vtu\""));
        let vtu = std::fs::read_to_string(dir.join("field_0000.vtu")).unwrap();
        assert!(vtu.contains("NumberOfCells=\"48\""));
        assert!(s.export_vtk(&states, &[0.0], &dir.join("x")).is_err());
        std::fs::remove_dir_all(dir).ok();
    }
}
