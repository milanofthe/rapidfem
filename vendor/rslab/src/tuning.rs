//! Hardware-aware auto-tuning + resource governor (feature `tuning`).
//!
//! [`HardwareInfo::probe`] detects cores + RAM; [`Calibration::load_or_measure`]
//! measures this machine's factorization throughput once (by factoring a
//! representative grid) and caches it (FFTW-"wisdom" style, keyed by a hardware
//! fingerprint); [`plan`] then turns a [`MemoryEstimate`] + a [`Budget`] into a
//! concrete [`FactorPlan`] - picking the thread count, predicting peak memory and
//! wall-clock, and (when over budget) selecting approximations (mixed precision /
//! incomplete factor / BLR) or recommending fail-fast. Deterministic given a fixed
//! calibration, so solver-in-the-loop scheduling is reproducible.

use std::path::PathBuf;

use num_complex::Complex;

use crate::diagnostics::MemoryEstimate;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::{BlrMode, LdltSymbolic, SolverSettings};

/// Detected machine capabilities - for budgeting and the calibration key.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub total_ram_bytes: u64,
    pub available_ram_bytes: u64,
}

impl HardwareInfo {
    /// Probe the current machine (cores via std, RAM via sysinfo).
    pub fn probe() -> Self {
        let logical = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let physical = sys.physical_core_count().unwrap_or(logical);
        HardwareInfo {
            logical_cores: logical,
            physical_cores: physical.max(1),
            total_ram_bytes: sys.total_memory(),
            available_ram_bytes: sys.available_memory(),
        }
    }
    /// Stable key for the calibration cache (cores + RAM size bucket).
    pub fn fingerprint(&self) -> u64 {
        let gb = self.total_ram_bytes / (1 << 30);
        ((self.physical_cores as u64) << 40) ^ ((self.logical_cores as u64) << 20) ^ gb.min(0xFFFFF)
    }
}

/// Measured cost-model constants for this machine - the cached "wisdom".
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    /// One-thread throughput in the `factor_flops` proxy unit, ×1e9 (giga/s), for a
    /// **real** (`f64`) factorization.
    pub geom_gflops: f64,
    /// Same proxy-flops/s rate for a **complex** (`Complex<f64>`) factorization:
    /// each proxy-flop is ~4 real flops plus wider memory traffic, so the rate in
    /// the type-independent proxy unit is lower --- calibrated separately so the
    /// runtime estimate is correct for the complex-symmetric target class.
    pub geom_gflops_cplx: f64,
    /// Parallel speedup measured at `speedup_threads`.
    pub speedup: f64,
    pub speedup_threads: usize,
    /// Coefficient of variation (std/mean) of the repeated single-thread factor
    /// time on this machine --- the measurement noise floor. The tuner's deviate
    /// guard is set from this (`min_gain = z·cv`) so it never chases a speedup
    /// smaller than the noise it measured on this hardware.
    pub time_cv: f64,
    pub fingerprint: u64,
}

impl Calibration {
    fn cache_path(fp: u64) -> PathBuf {
        let dir = std::env::var_os("RLA_CALIB_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("rla-calib"));
        dir.join(format!("calib-{fp:016x}.txt"))
    }

    /// Cached calibration for `hw`, or measure it (and cache) on first use.
    pub fn load_or_measure(hw: &HardwareInfo) -> Self {
        let fp = hw.fingerprint();
        if let Some(c) = Self::load(fp) {
            return c;
        }
        let c = Self::measure(hw);
        let _ = c.save();
        c
    }

    fn load(fp: u64) -> Option<Self> {
        let s = std::fs::read_to_string(Self::cache_path(fp)).ok()?;
        let mut it = s.split_whitespace();
        let geom_gflops: f64 = it.next()?.parse().ok()?;
        let speedup: f64 = it.next()?.parse().ok()?;
        let speedup_threads: usize = it.next()?.parse().ok()?;
        // The complex rate is a later field; an older cache file lacks it, so fall
        // back to a fraction of the real rate rather than failing the whole load.
        let geom_gflops_cplx = it
            .next()
            .and_then(|t| t.parse().ok())
            .unwrap_or(geom_gflops / 3.0);
        let time_cv = it.next().and_then(|t| t.parse().ok()).unwrap_or(0.1);
        Some(Calibration {
            geom_gflops,
            geom_gflops_cplx,
            speedup,
            speedup_threads,
            time_cv,
            fingerprint: fp,
        })
    }

    fn save(&self) -> std::io::Result<()> {
        let p = Self::cache_path(self.fingerprint);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(
            p,
            format!(
                "{} {} {} {} {}",
                self.geom_gflops,
                self.speedup,
                self.speedup_threads,
                self.geom_gflops_cplx,
                self.time_cv
            ),
        )
    }

    /// A reasonable default if calibration cannot run (e.g. analyze fails).
    fn fallback(hw: &HardwareInfo) -> Self {
        Calibration {
            geom_gflops: 2.0,
            geom_gflops_cplx: 2.0 / 3.0,
            speedup: (hw.physical_cores as f64).sqrt().max(1.0),
            speedup_threads: hw.physical_cores,
            time_cv: 0.1,
            fingerprint: hw.fingerprint(),
        }
    }

    /// The one-thread proxy-flops/s rate for a scalar of `value_bytes` (real `f64`
    /// = 8, `Complex<f64>` = 16): picks the complex rate for the wider type.
    pub fn rate_for(&self, value_bytes: usize) -> f64 {
        if value_bytes >= 16 {
            self.geom_gflops_cplx
        } else {
            self.geom_gflops
        }
    }

    /// Measure throughput by factoring a representative 3D grid at 1 thread and at
    /// `physical_cores`, recording the proxy-flops/s rate and the parallel speedup.
    pub fn measure(hw: &HardwareInfo) -> Self {
        let a = grid3d_spd::<f64>(24); // ≈ 13 800 DOFs, a few hundred ms
        let Ok(sym) = LdltSymbolic::analyze(&a) else {
            return Self::fallback(hw);
        };
        let flops = sym.estimate_memory::<f64>().factor_flops as f64;
        let time_at = |t: usize| -> Option<f64> {
            let opts = SolverSettings::default().with_threads(t);
            let start = std::time::Instant::now();
            sym.factor(&a, &opts).ok()?;
            Some(start.elapsed().as_secs_f64())
        };
        // Repeat the single-thread factor a few times for the timing noise floor
        // (coefficient of variation), which sets the tuner's deviate guard.
        let mut samples = Vec::new();
        for _ in 0..4 {
            match time_at(1) {
                Some(t) => samples.push(t),
                None => return Self::fallback(hw),
            }
        }
        let Some(tn) = time_at(hw.physical_cores.max(1)) else {
            return Self::fallback(hw);
        };
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let var = samples.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let time_cv = if mean > 0.0 { var.sqrt() / mean } else { 0.0 };
        let t1 = mean;
        let geom_gflops = (flops / t1.max(1e-9)) / 1e9;
        // Complex rate: same grid, complex-typed, factored once at one thread. The
        // structure (fill, flops proxy) is identical; only the per-flop cost differs.
        let ac = grid3d_spd::<Complex<f64>>(24);
        let geom_gflops_cplx = match LdltSymbolic::analyze(&ac) {
            Ok(symc) => {
                let fc = symc.estimate_memory::<Complex<f64>>().factor_flops as f64;
                let start = std::time::Instant::now();
                match symc.factor(&ac, &SolverSettings::default().with_threads(1)) {
                    Ok(_) => (fc / start.elapsed().as_secs_f64().max(1e-9)) / 1e9,
                    Err(_) => geom_gflops / 3.0,
                }
            }
            Err(_) => geom_gflops / 3.0,
        };
        Calibration {
            geom_gflops,
            geom_gflops_cplx,
            speedup: (t1 / tn.max(1e-9)).max(1.0),
            speedup_threads: hw.physical_cores.max(1),
            time_cv,
            fingerprint: hw.fingerprint(),
        }
    }

    /// Data-driven deviate threshold for the tuner: a candidate must beat the
    /// default by more than the measured timing noise (`z·time_cv`, `z=2` for ~95%
    /// confidence) before the tuner switches to it, so it never chases a predicted
    /// gain smaller than this machine's own single-shot variance. Clamped to a
    /// sensible range in case calibration measured an implausible value.
    pub fn min_gain(&self) -> f64 {
        (2.0 * self.time_cv).clamp(0.03, 0.30)
    }

    /// Interpolated speedup at `threads`: linear toward the calibrated peak, flat
    /// beyond it (sparse-direct scaling saturates).
    pub fn speedup_for(&self, threads: usize) -> f64 {
        if threads <= 1 {
            1.0
        } else if threads >= self.speedup_threads {
            self.speedup
        } else {
            1.0 + (self.speedup - 1.0) * (threads - 1) as f64
                / (self.speedup_threads - 1).max(1) as f64
        }
    }
}

/// Resource budget the factorization must live within. The default (no memory
/// limit, all-cores, no approximations) is an unconstrained exact plan.
#[derive(Debug, Clone, Default)]
pub struct Budget {
    /// Hard memory ceiling (bytes). `None` = no limit.
    pub max_mem_bytes: Option<u64>,
    /// Thread budget. `0` = the machine's physical cores.
    pub max_threads: usize,
    /// When over the memory budget, may the planner factor in single precision
    /// (`Complex<f32>`/`f32`) and recover accuracy by refinement? Halves the factor.
    pub allow_mixed_precision: bool,
    /// When over budget, may it drop small fill (incomplete factor)? `Some(tau)`.
    pub allow_drop_tol: Option<f64>,
    /// When over budget, may it BLR-compress the big fronts?
    pub allow_blr: bool,
}

/// A concrete factorization plan: tuned options + predictions + decisions.
#[derive(Debug, Clone)]
pub struct FactorPlan {
    /// Tuned options (thread count, plus drop_tol / BLR if chosen to fit budget).
    pub opts: SolverSettings,
    /// Recommendation to factor in single precision (the caller casts the matrix);
    /// not expressible in `opts` since it is a matrix-type choice.
    pub use_mixed_precision: bool,
    /// Predicted transient peak after the chosen approximations (bytes).
    pub est_peak_bytes: u64,
    /// Predicted factor wall-clock at the chosen thread count (ms).
    pub est_runtime_ms: f64,
    /// Whether the prediction fits the memory budget.
    pub fits: bool,
    /// Human-readable summary of the decisions taken.
    pub note: String,
}

/// Learned additive-residual weights `[intercept, amdahl_frac, ln(threads),
/// ln(tree_width)]` correcting `log(measured / analytical speedup)`, fit offline by
/// `benches/fit_residual.py` on the generated thread-scaling corpus (54 matrices).
/// It is a small, dimensionless, hardware-agnostic correction: the absolute rate
/// comes from the calibration, this only reshapes the speedup curve by structure.
const TIME_RESIDUAL: [f64; 4] = [-0.4621, 0.7550, -0.0791, 0.0358];

/// Cost-model worker-count selection: the fewest workers that reach (within a
/// small margin) the minimum predicted time. Because
/// [`est_runtime_ms_threaded`](MemoryEstimate::est_runtime_ms_threaded) floors at
/// the critical path of the assembly tree, adding workers past the point where the
/// parallel term hits that floor buys nothing --- so a critical-path-bound matrix
/// (a banded chain, whose critical path is nearly its whole work) gets **fewer**
/// workers, and a wide 3D tree gets more. Deterministic given the calibration.
/// `max_threads == 0` means all physical cores.
pub fn recommend_threads_cost_model(
    estimate: &MemoryEstimate,
    calib: &Calibration,
    max_threads: usize,
    all_cores: usize,
) -> usize {
    let cap = if max_threads == 0 {
        all_cores.max(1)
    } else {
        max_threads
    };
    let rate = calib.rate_for(estimate.value_bytes);
    // Learned residual on the analytical speedup (issue #62): the bare
    // `flops / max(crit, flops/speedup)` model mispredicts the achieved scaling
    // systematically --- the critical-path flops overstate the true serial
    // fraction (sibling subtrees overlap), so it is too pessimistic on
    // serial-heavy trees and too optimistic on wide ones. A ridge fit of
    // log(measured/predicted speedup) on the dimensionless structural features
    // (fit offline by `benches/fit_residual.py`, 26% held-out RMSE reduction under
    // leave-one-matrix-out CV) corrects both. `amdahl_frac` carries almost all of
    // the signal, validating the critical-path feature. Clamped so the residual
    // refines the calibrated base, never swings the choice wildly.
    let residual_ln = |t: usize| -> f64 {
        if t <= 1 {
            return 0.0; // t=1 is the speedup reference: no correction
        }
        let flops = estimate.factor_flops.max(1) as f64;
        let crit = estimate.critical_path_flops.max(1) as f64;
        let amdahl = (crit / flops).min(1.0);
        let width = estimate.max_tree_width.max(1) as f64;
        let r = TIME_RESIDUAL[0]
            + TIME_RESIDUAL[1] * amdahl
            + TIME_RESIDUAL[2] * (t as f64).ln()
            + TIME_RESIDUAL[3] * width.ln();
        r.clamp(-0.7, 0.7) // exp factor in [0.50, 2.01]
    };
    // Corrected time: predicted speedup s -> s * exp(residual), so time /= that,
    // but never below the hard critical-path floor. The residual is linear, so at
    // `amdahl_frac -> 1` (a true chain) it would extrapolate to a speedup the
    // physics forbids; clamping to the critical-path floor (the runtime with
    // infinite workers) keeps a critical-path-bound matrix at one worker.
    let floor_ms = estimate.est_runtime_ms_threaded(rate, f64::MAX);
    let time = |t: usize| {
        let corrected =
            estimate.est_runtime_ms_threaded(rate, calib.speedup_for(t)) * (-residual_ln(t)).exp();
        corrected.max(floor_ms)
    };
    let mut best_t = 1;
    let mut best = time(1);
    for t in [2usize, 4, 6, 8, 12, 16, 20, 24, 32]
        .into_iter()
        .filter(|&t| t <= cap)
    {
        let tt = time(t);
        if tt < best * 0.97 {
            // >3% faster: worth the extra workers.
            best = tt;
            best_t = t;
        } else {
            // Diminishing returns (critical path / saturation) -- stop adding cores.
            break;
        }
    }
    best_t
}

/// Turn an a-priori [`MemoryEstimate`] + a [`Budget`] into a concrete plan, using
/// the machine's [`HardwareInfo`] and [`Calibration`]. Path-agnostic: pass the
/// estimate from `LuSymbolic`/`LdltSymbolic::estimate_memory`. Pure given the
/// inputs (deterministic scheduling).
pub fn plan(
    estimate: &MemoryEstimate,
    budget: &Budget,
    hw: &HardwareInfo,
    calib: &Calibration,
) -> FactorPlan {
    // Cost-model worker count: the fewest cores that reach near-minimum predicted
    // time (fewer when the critical path or saturation dominates), not blindly all.
    let threads =
        recommend_threads_cost_model(estimate, calib, budget.max_threads, hw.physical_cores);
    let speedup = calib.speedup_for(threads);
    let runtime = estimate.est_runtime_ms_threaded(calib.rate_for(estimate.value_bytes), speedup);
    let mut opts = SolverSettings::default().with_threads(threads);
    let mut peak = estimate.transient_peak_bytes;
    let mut use_f32 = false;
    let mut notes: Vec<String> = Vec::new();

    if let Some(maxm) = budget.max_mem_bytes {
        // Apply approximations in increasing aggressiveness until it fits. The
        // reduction factors are conservative rules of thumb (documented as such).
        if peak > maxm && budget.allow_mixed_precision && estimate.value_bytes > 4 {
            use_f32 = true;
            peak /= 2; // single precision ≈ halves the factor + panels
            notes.push("mixed-precision (factor in single)".into());
        }
        if peak > maxm {
            if let Some(tau) = budget.allow_drop_tol {
                opts = opts.with_drop_tol(tau);
                peak = (peak as f64 * 0.7) as u64; // incomplete factor ≈ −30%
                notes.push(format!("incomplete factor (drop_tol={tau:.0e})"));
            }
        }
        if peak > maxm && budget.allow_blr {
            opts = opts.with_blr(BlrMode::contribution_blocks(1e-4));
            peak = (peak as f64 * 0.7) as u64; // BLR ≈ −30% on big fronts
            notes.push("BLR compression".into());
        }
        if peak > maxm {
            notes.push(format!(
                "STILL over budget ({:.0}>{:.0} MB) - recommend fail-fast",
                peak as f64 / 1e6,
                maxm as f64 / 1e6
            ));
        }
    }

    let fits = match budget.max_mem_bytes {
        Some(m) => peak <= m,
        None => true,
    };
    if notes.is_empty() {
        notes.push(format!("exact, {threads} threads"));
    }
    FactorPlan {
        opts,
        use_mixed_precision: use_f32,
        est_peak_bytes: peak,
        est_runtime_ms: runtime,
        fits,
        note: notes.join("; "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_model_thread_count_respects_critical_path() {
        // Synthetic calibration: peak speedup 6 at 12 threads.
        let calib = Calibration {
            geom_gflops: 2.0,
            geom_gflops_cplx: 0.7,
            speedup: 6.0,
            speedup_threads: 12,
            time_cv: 0.1,
            fingerprint: 0,
        };
        let mut est = crate::diagnostics::MemoryEstimate {
            value_bytes: 8,
            factor_nnz: 0,
            factor_bytes: 0,
            panels_all_bytes: 0,
            panel_live_peak_bytes: 0,
            transient_peak_bytes: 0,
            mf_transient_peak_bytes: 0,
            factor_flops: 100_000_000_000, // 1e11 total work
            critical_path_flops: 0,
            max_tree_width: 256,
        };
        // Wide tree (critical path 1% of work): more threads keep paying off.
        est.critical_path_flops = 1_000_000_000; // 1e9
        let wide = recommend_threads_cost_model(&est, &calib, 24, 24);
        // Critical-path-bound (chain: critical path == total work): no parallel gain.
        est.critical_path_flops = est.factor_flops;
        let chain = recommend_threads_cost_model(&est, &calib, 24, 24);
        assert!(
            wide > chain,
            "wide tree uses more workers ({wide}) than a chain ({chain})"
        );
        assert_eq!(
            chain, 1,
            "a critical-path-bound matrix uses a single worker"
        );
        assert!(wide >= 8, "a wide tree uses many workers ({wide})");
    }

    #[test]
    fn probe_calibrate_plan_governor() {
        let hw = HardwareInfo::probe();
        assert!(hw.logical_cores >= 1 && hw.physical_cores >= 1);
        assert!(hw.total_ram_bytes > 0, "RAM probed");

        let calib = Calibration::measure(&hw);
        assert!(calib.geom_gflops > 0.0 && calib.speedup >= 1.0);
        assert_eq!(calib.speedup_for(1), 1.0);
        assert!(calib.speedup_for(hw.physical_cores) >= 1.0);
        // cache round-trip
        calib.save().unwrap();
        assert!(Calibration::load(hw.fingerprint()).is_some());

        let a = grid3d_spd::<f64>(16);
        let est = LdltSymbolic::analyze(&a).unwrap().estimate_memory::<f64>();
        assert!(est.factor_flops > 0);

        // Generous budget → exact, fits, predicted runtime positive.
        let plan_ok = plan(&est, &Budget::default(), &hw, &calib);
        assert!(plan_ok.fits && !plan_ok.use_mixed_precision);
        assert!(plan_ok.est_runtime_ms > 0.0);
        // v2 cost-model thread selection (#61) picks the fewest cores that reach
        // near-minimum predicted time — for a small grid the critical path or
        // saturation dominates, so it may (correctly) choose fewer than all cores.
        // The contract is 1 ≤ threads ≤ physical_cores, not "always all cores".
        match plan_ok.opts.threads {
            crate::numeric::multifrontal_ldlt::Threads::Fixed(t) => {
                assert!(
                    t >= 1 && t <= hw.physical_cores.max(1),
                    "threads within core budget"
                );
            }
            other => panic!("expected a fixed thread count, got {other:?}"),
        }

        // Tight budget with all approximations allowed → planner applies them.
        let tight = Budget {
            max_mem_bytes: Some(est.transient_peak_bytes / 8),
            max_threads: 0,
            allow_mixed_precision: true,
            allow_drop_tol: Some(1e-3),
            allow_blr: true,
        };
        let plan_tight = plan(&est, &tight, &hw, &calib);
        assert!(
            plan_tight.use_mixed_precision || plan_tight.opts.drop_tol.is_some(),
            "an approximation was selected"
        );
        assert!(!plan_tight.note.is_empty());
        assert!(
            plan_tight.est_peak_bytes < est.transient_peak_bytes,
            "approximations shrink the peak"
        );
    }
}

/// 3D 7-point Laplacian (k³ grid, Dirichlet, SPD, lower triangle), generic over
/// the scalar type - the calibration's representative matrix. Complex-typed, it is
/// real-valued but factors through the complex kernel, so it times the complex
/// proxy-flops/s rate.
fn grid3d_spd<T: Scalar>(k: usize) -> CscMatrix<T> {
    let n = k * k * k;
    let idx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals: Vec<T> = Vec::new();
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let p = idx(x, y, z);
                rows.push(p);
                cols.push(p);
                vals.push(T::from_real(6.0));
                let mut nb = |q: usize| {
                    let (hi, lo) = if p >= q { (p, q) } else { (q, p) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(T::from_real(-1.0));
                };
                if x + 1 < k {
                    nb(idx(x + 1, y, z));
                }
                if y + 1 < k {
                    nb(idx(x, y + 1, z));
                }
                if z + 1 < k {
                    nb(idx(x, y, z + 1));
                }
            }
        }
    }
    match CscMatrix::from_triplets(n, &rows, &cols, &vals) {
        Ok(m) => m,
        Err(_) => CscMatrix {
            n: 0,
            col_ptr: vec![0],
            row_idx: vec![],
            values: vec![],
        },
    }
}
