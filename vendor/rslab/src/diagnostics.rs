//! Deterministic resource diagnostics: an **a-priori** peak-memory estimate
//! (computed from the symbolic factorization, before any numeric work) and a
//! per-stage runtime/memory report collected during factorization.
//!
//! The estimate is a pure function of the analyzed structure, so it is fully
//! reproducible and lets a solver-in-the-loop scheduler decide *before* allocating
//! whether a factorization fits the memory budget (fail-fast / pick approximation).

use std::fmt;

/// A-priori estimate of the memory a factorization will use, in bytes. All fields
/// are deterministic functions of the symbolic structure and the scalar size.
#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    /// Scalar size in bytes (`16` for `Complex<f64>`, `8` for `f64`, ...).
    pub value_bytes: usize,
    /// Structural nonzeros in the factor (`L`+`U` for LU, `L` for LDL^T) - an upper
    /// bound on the emitted factor (numeric cancellation can only lower it).
    pub factor_nnz: u64,
    /// Bytes of the resident factor (the CSC output): `factor_nnz*(value+index)`.
    pub factor_bytes: u64,
    /// Dense supernode panels if **all** were held at once (the naive left-looking
    /// peak, i.e. without panel-freeing).
    pub panels_all_bytes: u64,
    /// Peak of the **live** dense panels under the refcount free-schedule - what
    /// the left-looking path actually holds at once.
    pub panel_live_peak_bytes: u64,
    /// Estimated overall transient peak for the **left-looking** path: live panels
    /// plus the accumulated compact factor plus the equilibrated input copy/copies.
    /// The number to compare against RAM for [`FactorMethod::LeftLooking`](crate::FactorMethod::LeftLooking).
    pub transient_peak_bytes: u64,
    /// Estimated transient peak for the **multifrontal** path: the
    /// contribution-block-stack model (the active front plus the live CBs of
    /// completed subtrees not yet consumed by their parent) + factor + input.
    /// Multifrontal holds more transiently than left-looking, so this is the
    /// number to compare against RAM for [`FactorMethod::Multifrontal`](crate::FactorMethod::Multifrontal).
    /// Defaults to [`transient_peak_bytes`](Self::transient_peak_bytes) until the
    /// path-specific model fills it.
    pub mf_transient_peak_bytes: u64,
    /// Geometric factorization work proxy `sum nrow^2*ncol` over supernodes (type-
    /// independent). Divide by a calibrated geometric-flops/s rate for a runtime
    /// estimate - see [`est_runtime_ms`](Self::est_runtime_ms).
    pub factor_flops: u64,
    /// Critical-path geom-flops: the longest serial chain of front work from a
    /// leaf to a root of the assembly tree (`front_flops(s) + max child`). This is
    /// the Amdahl lower bound on parallel factor time --- even with unlimited
    /// workers the tree cannot factor below `critical_path_flops / rate`, since a
    /// front depends on its children. The v2 thread-aware time model uses it to
    /// decide the worker count (a memory-bound or critical-path-bound matrix gains
    /// nothing, and may lose, from more threads). `0` until the tree pass fills it.
    pub critical_path_flops: u64,
    /// Peak assembly-tree width: the most supernodes at any one level, i.e. the
    /// maximum node-level parallelism available. Caps the useful worker count.
    pub max_tree_width: u64,
}

impl MemoryEstimate {
    pub fn transient_peak_mb(&self) -> f64 {
        self.transient_peak_bytes as f64 / 1e6
    }
    pub fn factor_mb(&self) -> f64 {
        self.factor_bytes as f64 / 1e6
    }
    /// Does the estimated transient peak fit in `available` bytes?
    pub fn fits_in(&self, available_bytes: u64) -> bool {
        self.transient_peak_bytes <= available_bytes
    }

    /// Estimated factor wall-clock in ms: `factor_flops` divided by a calibrated
    /// geometric-flops/s rate (`gflops` = giga-geom-flops/s on one thread) scaled by
    /// the measured `parallel_speedup` at the chosen thread count. Both come from
    /// the calibration (`tuning` feature); pass machine defaults otherwise.
    pub fn est_runtime_ms(&self, gflops: f64, parallel_speedup: f64) -> f64 {
        let rate = (gflops.max(1e-6) * parallel_speedup.max(1e-6)) * 1e9;
        (self.factor_flops as f64 / rate) * 1e3
    }

    /// Thread-aware runtime estimate (the v2 model): the parallel time cannot fall
    /// below the **critical path** of the assembly tree (Amdahl), so it is the max
    /// of the serial critical-path floor and the work divided by the achieved
    /// parallel rate. `gflops` is the one-thread geom-flops/s rate and
    /// `parallel_speedup` the achieved speedup at the chosen worker count (from the
    /// calibration). Unlike [`est_runtime_ms`](Self::est_runtime_ms) this does not
    /// let more threads drive the estimate below the tree's serial dependency, so
    /// argmin over the worker count correctly stops adding threads once the
    /// critical path (or, in the full v2 model, memory bandwidth) dominates.
    pub fn est_runtime_ms_threaded(&self, gflops: f64, parallel_speedup: f64) -> f64 {
        let rate1 = gflops.max(1e-6) * 1e9; // one-thread geom-flops/s
        let serial_floor = self.critical_path_flops as f64 / rate1;
        let parallel = self.factor_flops as f64 / (rate1 * parallel_speedup.max(1e-6));
        serial_floor.max(parallel) * 1e3
    }
}

impl fmt::Display for MemoryEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "transient-peak <= {:.0} MB (panels {:.0} + factor {:.0} + input/scratch); \
             factor ~{} nnz; panel-freed floor {:.0} MB",
            self.transient_peak_bytes as f64 / 1e6,
            self.panels_all_bytes as f64 / 1e6,
            self.factor_bytes as f64 / 1e6,
            self.factor_nnz,
            self.panel_live_peak_bytes as f64 / 1e6,
        )
    }
}

/// Core left-looking memory estimator. `panel_bytes(s)` is supernode `s`'s dense
/// panel size; `compact_bytes(s)` its CSC-fragment size; `update_list[s]` its
/// factored descendants (consumers). Simulates the refcount free-schedule in
/// elimination/postorder (supernodes are numbered in postorder) to get the live
/// panel peak and the accumulating compact factor - the same schedule the numeric
/// path runs, so the estimate matches what it allocates.
pub(crate) fn estimate_left_looking<'a>(
    nsuper: usize,
    panel_bytes: &dyn Fn(usize) -> u64,
    compact_bytes: &dyn Fn(usize) -> u64,
    updaters: &dyn Fn(usize) -> &'a [crate::numeric::ll_common::Li],
    value_bytes: usize,
    input_bytes: u64,
    zero_copy: bool,
) -> MemoryEstimate {
    let mut refc = vec![0usize; nsuper];
    for s in 0..nsuper {
        for &k in updaters(s) {
            refc[k as usize] += 1;
        }
    }
    let panels_all: u64 = (0..nsuper).map(panel_bytes).sum();
    let factor_bytes: u64 = (0..nsuper).map(compact_bytes).sum();

    let mut live_panels: i64 = 0;
    let mut compact: i64 = 0;
    let mut peak: i64 = 0;
    for s in 0..nsuper {
        live_panels += panel_bytes(s) as i64;
        for &k in updaters(s) {
            let k = k as usize;
            refc[k] -= 1;
            if refc[k] == 0 {
                live_panels -= panel_bytes(k) as i64;
                compact += compact_bytes(k) as i64;
            }
        }
        if refc[s] == 0 {
            live_panels -= panel_bytes(s) as i64;
            compact += compact_bytes(s) as i64;
        }
        peak = peak.max(live_panels + compact);
    }
    let panel_live_peak = peak.max(0) as u64;
    // Conservative transient upper bound. At many threads the parallel frontier of
    // a top-heavy tree holds nearly all panels at once, and the emit builds the full
    // factor CSC on top - so the safe estimate is all-resident panels + the factor +
    // the input copies + a per-thread scratch margin (cmod/cdiv buffers, gloc). This
    // is the number to compare against RAM for a fail-fast / scheduling decision; the
    // panel-freeing path makes the *actual* peak lower (down to `panel_live_peak`),
    // so this never under-predicts.
    // Per-thread scratch (cmod/cdiv buffers, gloc, the emit double-buffer) plus a
    // small absolute floor - tuned so the bound stays >= the measured peak across
    // sizes (validated: est/measured ~ 1.0-1.2x), never under-predicting.
    // With zero-copy panels (`zero_copy`: the emitted panel is the stored
    // factor, `compact_bytes == panel_bytes`) nothing is built on top of the
    // resident panels, so the bound is the panels themselves plus the input
    // and the scratch margin.
    let (scratch, transient) = if zero_copy {
        let scratch = panels_all / 4 + 32_000_000;
        (scratch, panels_all + input_bytes + scratch)
    } else {
        let scratch = (panels_all + factor_bytes) / 4 + 32_000_000;
        (scratch, panels_all + factor_bytes + input_bytes + scratch)
    };
    let _ = scratch;
    let entry_bytes = if zero_copy {
        value_bytes as u64
    } else {
        value_bytes as u64 + 8
    };
    MemoryEstimate {
        value_bytes,
        factor_nnz: factor_bytes / entry_bytes.max(1),
        factor_bytes,
        panels_all_bytes: panels_all,
        panel_live_peak_bytes: panel_live_peak,
        transient_peak_bytes: transient,
        // Default to the left-looking peak; the multifrontal model overrides this
        // in the path-aware caller (it needs the assembly-tree child structure).
        mf_transient_peak_bytes: transient,
        factor_flops: 0,        // set by the caller (needs supernode dimensions)
        critical_path_flops: 0, // set by the caller (needs the assembly tree)
        max_tree_width: 0,      // set by the caller (needs the level structure)
    }
}

/// Multifrontal transient-peak model: the **contribution-block stack** under the
/// rayon work-stealing schedule. Unlike left-looking, multifrontal holds dense
/// fronts plus the contribution blocks (packed lower triangles,
/// `cnrow*(cnrow+1)/2` each, the symmetric-LDL^T storage the numeric path
/// actually uses) of completed subtrees not yet consumed by their parent. The
/// driver factors a whole assembly-tree level concurrently, so the
/// conservative peak is, over the levels, the level's total front memory
/// (`sum nrow^2`) plus the contribution blocks of its children feeding the
/// assembly. Assuming a full level live at once never under-predicts at any
/// thread count - the transient the left-looking estimate does not capture.
/// (LDL^T-path model only; the unsymmetric LU path stores full-square CBs and
/// does not consult this.)
pub(crate) fn estimate_multifrontal_active_peak(
    by_level: &[Vec<usize>],
    nrow: &dyn Fn(usize) -> u64,
    ncol: &dyn Fn(usize) -> u64,
    children: &[Vec<usize>],
    value_bytes: u64,
) -> u64 {
    let cb = |s: usize| -> u64 {
        let cn = nrow(s).saturating_sub(ncol(s));
        cn * (cn + 1) / 2 * value_bytes
    };
    let mut peak: u64 = 0;
    for level in by_level {
        let fronts: u64 = level.iter().map(|&s| nrow(s) * nrow(s) * value_bytes).sum();
        let child_cb: u64 = level
            .iter()
            .flat_map(|&s| children[s].iter())
            .map(|&c| cb(c))
            .sum();
        peak = peak.max(fronts + child_cb);
    }
    peak
}

// ---------------------------------------------------------------------------
// Per-stage runtime/memory report, collected during a factorization.
// ---------------------------------------------------------------------------

/// One factorization stage's cost. `flops`/`bytes` are deterministic (structural);
/// `wall_ms` is observability (varies with load/threads).
#[derive(Debug, Clone)]
pub struct StageReport {
    pub name: &'static str,
    pub wall_ms: f64,
    pub flops: u64,
    pub bytes: u64,
}

/// The choices the solver made on its own for one factorization: what the
/// `Auto` settings resolved to and what the structure looked like. Plain
/// strings (the `Debug` form of the enums) so a host can display or serialise
/// them without the enum types.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Decisions {
    /// The ordering the caller asked for (`Auto` included).
    pub ordering_requested: String,
    /// The ordering actually dispatched after `Auto` resolution.
    pub ordering_used: String,
    /// The ordering preprocessor actually used (`None` / `LdltCompress`).
    pub preprocess: String,
    /// The supernode amalgamation strategy actually used.
    pub amalgamation: String,
    /// The equilibration applied before factoring (the symmetric path); the
    /// unsymmetric paths name their built-in scaling.
    pub scaling: String,
    /// The numeric kernel (`LeftLooking`, `Multifrontal`, `Klu`).
    pub method: String,
    pub n_supernodes: usize,
    /// Largest front (rows) after amalgamation.
    pub max_front: usize,
    /// Depth of the assembly tree.
    pub tree_levels: usize,
    /// KLU: number of BTF blocks (0 when BTF is off or not a KLU factor).
    pub btf_blocks: usize,
}

/// What the numeric phase did to the pivots.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NumericReport {
    /// Pivots lifted by the static-pivoting floor (`n_perturbed`).
    pub perturbed: usize,
    /// Bunch-Kaufman 2x2 pivots (`None` for the LU paths).
    pub two_by_two: Option<usize>,
    /// Inertia `(positive, negative, zero)` of a symmetric factor.
    pub inertia: Option<(usize, usize, usize)>,
}

/// Solve-phase accumulators (updated by every `solve*` call on the factor).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SolveStats {
    /// Right-hand sides solved (a `solve_many` counts each column).
    pub rhs: usize,
    /// `solve*` calls.
    pub calls: usize,
    pub wall_ms: f64,
    /// Iterative-refinement steps taken over all calls.
    pub refine_steps: usize,
}

impl SolveStats {
    pub fn record(&mut self, rhs: usize, wall_ms: f64, refine_steps: usize) {
        self.rhs += rhs;
        self.calls += 1;
        self.wall_ms += wall_ms;
        self.refine_steps += refine_steps;
    }
}

/// [`SolveStats`] behind a mutex, so a factor handle records its solves
/// through `&self`; cloning snapshots the counters (a cloned handle starts a
/// separate account).
#[derive(Debug, Default)]
pub struct SolveCounter(std::sync::Mutex<SolveStats>);

impl SolveCounter {
    pub fn record(&self, rhs: usize, wall_ms: f64, refine_steps: usize) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .record(rhs, wall_ms, refine_steps);
    }
    pub fn snapshot(&self) -> SolveStats {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Clone for SolveCounter {
    fn clone(&self) -> Self {
        SolveCounter(std::sync::Mutex::new(self.snapshot()))
    }
}

/// Everything one factorization can tell about itself: the per-stage cost,
/// the decisions taken, the numeric outcome, the settings that had no effect
/// on the chosen path, and the solve-phase accumulators. Per-call and
/// concurrency-safe (no global state), so a solver-in-the-loop with many
/// concurrent solves gets correct per-solve numbers. Carries the a-priori
/// [`MemoryEstimate`] alongside the measured factor time for estimate-vs-actual
/// feedback. Logged as one `Info` line per factorization (see
/// [`summary`](Self::summary)) and readable from the factor handle.
/// Throughput of a factorization and its solves, derived from the stage
/// records: the numbers to compare across orderings, thread counts and
/// machines. Rates are `0.0` where the stage is absent or took no time.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rates {
    /// Analysis throughput, million unknowns per second (`n` over the
    /// `analyze` stage wall time).
    pub analyze_mdof_s: f64,
    /// Numeric factorization throughput, million unknowns per second, over
    /// the latest numeric stage (`factor`, `klu-factor` or `klu-refactor`).
    pub factor_mdof_s: f64,
    /// Numeric factorization flop rate in GFlop/s (the numeric stage's flop
    /// count over its wall time; zero where the path counts no flops).
    pub factor_gflops: f64,
    /// Factor entries produced per second, in millions (`nnz(L)` over the
    /// numeric stage wall time): the memory-side throughput.
    pub factor_mnnz_s: f64,
    /// End-to-end throughput of the recorded stages (analysis, scaling and
    /// the numeric factorization together), million unknowns per second.
    pub total_mdof_s: f64,
    /// Solve throughput over every recorded solve, million unknowns per
    /// second (`rhs * n` over the accumulated solve wall time).
    pub solve_mdof_s: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    /// `analyze` (ordering + symbolic; the analysis time of the reused
    /// symbolic object), `scale`, `factor`, `refactor` in order.
    pub stages: Vec<StageReport>,
    pub threads: usize,
    pub n: usize,
    /// Stored nonzeros of the input pattern.
    pub nnz_a: u64,
    pub factor_nnz: u64,
    pub estimate: Option<MemoryEstimate>,
    pub decisions: Decisions,
    pub numeric: NumericReport,
    /// Settings the caller set to a non-default value that the chosen factor
    /// path does not read (also emitted as `Warning` log records).
    pub warnings: Vec<String>,
    pub solves: SolveStats,
}

impl Diagnostics {
    /// Wall time of the recorded stages (the analysis stage included when the
    /// symbolic object recorded it).
    pub fn total_ms(&self) -> f64 {
        self.stages.iter().map(|s| s.wall_ms).sum()
    }
    /// Wall time of one stage by name, `None` if not recorded.
    pub fn stage_ms(&self, name: &str) -> Option<f64> {
        self.stages
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.wall_ms)
    }
    /// `nnz(L) / nnz(A)` (`0` without an input count).
    /// The latest numeric stage (`factor`, `klu-factor` or `klu-refactor`).
    pub fn numeric_stage(&self) -> Option<&StageReport> {
        self.stages
            .iter()
            .rev()
            .find(|s| matches!(s.name, "factor" | "klu-factor" | "klu-refactor"))
    }

    /// Throughput figures derived from the stage records; see [`Rates`].
    pub fn rates(&self) -> Rates {
        let mdof = |wall_ms: f64| {
            if wall_ms > 0.0 {
                self.n as f64 / (wall_ms * 1e-3) / 1e6
            } else {
                0.0
            }
        };
        let mut r = Rates {
            analyze_mdof_s: mdof(self.stage_ms("analyze").unwrap_or(0.0)),
            total_mdof_s: mdof(self.total_ms()),
            ..Rates::default()
        };
        if let Some(st) = self.numeric_stage() {
            if st.wall_ms > 0.0 {
                let secs = st.wall_ms * 1e-3;
                r.factor_mdof_s = mdof(st.wall_ms);
                r.factor_gflops = st.flops as f64 / secs / 1e9;
                r.factor_mnnz_s = self.factor_nnz as f64 / secs / 1e6;
            }
        }
        if self.solves.wall_ms > 0.0 && self.solves.rhs > 0 {
            r.solve_mdof_s =
                (self.solves.rhs as f64 * self.n as f64) / (self.solves.wall_ms * 1e-3) / 1e6;
        }
        r
    }

    pub fn fill_ratio(&self) -> f64 {
        if self.nnz_a == 0 {
            0.0
        } else {
            self.factor_nnz as f64 / self.nnz_a as f64
        }
    }
    pub fn push(&mut self, name: &'static str, wall_ms: f64, flops: u64, bytes: u64) {
        self.stages.push(StageReport {
            name,
            wall_ms,
            flops,
            bytes,
        });
    }
    /// The one-line account the `Info` log carries per factorization.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} n={} nnz(A)={} nnz(L)={} fill={:.2} threads={} ordering={}",
            self.decisions.method,
            self.n,
            self.nnz_a,
            self.factor_nnz,
            self.fill_ratio(),
            self.threads,
            self.decisions.ordering_used,
        );
        if self.decisions.ordering_requested != self.decisions.ordering_used
            && !self.decisions.ordering_requested.is_empty()
        {
            s.push_str(&format!(
                " (requested {})",
                self.decisions.ordering_requested
            ));
        }
        if self.decisions.n_supernodes > 0 {
            s.push_str(&format!(
                " supernodes={} max_front={} levels={}",
                self.decisions.n_supernodes, self.decisions.max_front, self.decisions.tree_levels
            ));
        }
        if self.decisions.btf_blocks > 0 {
            s.push_str(&format!(" btf_blocks={}", self.decisions.btf_blocks));
        }
        if self.numeric.perturbed > 0 {
            s.push_str(&format!(" perturbed={}", self.numeric.perturbed));
        }
        if let Some(k) = self.numeric.two_by_two {
            s.push_str(&format!(" pivots2x2={k}"));
        }
        for st in &self.stages {
            s.push_str(&format!(" {}={:.1}ms", st.name, st.wall_ms));
        }
        let r = self.rates();
        if r.factor_mdof_s > 0.0 {
            s.push_str(&format!(
                " factor={:.2}MDOF/s {:.1}GF/s",
                r.factor_mdof_s, r.factor_gflops
            ));
        }
        if let Some(e) = &self.estimate {
            s.push_str(&format!(" est_peak={:.0}MB", e.transient_peak_mb()));
        }
        s
    }
}

impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "factorization diagnostics: {}", self.summary())?;
        writeln!(
            f,
            "  decisions: preprocess={} amalgamation={} scaling={}",
            self.decisions.preprocess, self.decisions.amalgamation, self.decisions.scaling
        )?;
        if let Some((p, n, z)) = self.numeric.inertia {
            writeln!(f, "  inertia: +{p} -{n} 0:{z}")?;
        }
        let tot = self.total_ms().max(1e-9);
        for s in &self.stages {
            writeln!(
                f,
                "  {:<10} {:8.1} ms ({:4.0}%)  {:>10} Mflop  {:>8.0} MB",
                s.name,
                s.wall_ms,
                100.0 * s.wall_ms / tot,
                s.flops / 1_000_000,
                s.bytes as f64 / 1e6,
            )?;
        }
        let r = self.rates();
        writeln!(
            f,
            "  rates: analyze {:.2} MDOF/s, factor {:.2} MDOF/s ({:.1} GF/s, {:.1} Mnnz/s), total {:.2} MDOF/s",
            r.analyze_mdof_s, r.factor_mdof_s, r.factor_gflops, r.factor_mnnz_s, r.total_mdof_s
        )?;
        if self.solves.calls > 0 {
            writeln!(
                f,
                "  solves: {} calls, {} rhs, {:.1} ms, {} refinement steps, {:.1} MDOF/s",
                self.solves.calls,
                self.solves.rhs,
                self.solves.wall_ms,
                self.solves.refine_steps,
                r.solve_mdof_s
            )?;
        }
        for w in &self.warnings {
            writeln!(f, "  warning: {w}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod rates_tests {
    use super::*;

    #[test]
    fn rates_follow_the_latest_numeric_stage_and_the_solve_totals() {
        let mut d = Diagnostics {
            n: 1_000_000,
            factor_nnz: 50_000_000,
            ..Default::default()
        };
        d.push("analyze", 250.0, 0, 0);
        d.push("factor", 500.0, 40_000_000_000, 0);
        let mut solves = SolveStats::default();
        solves.record(8, 100.0, 0);
        d.solves = solves;
        let r = d.rates();
        assert!((r.analyze_mdof_s - 4.0).abs() < 1e-9);
        assert!((r.factor_mdof_s - 2.0).abs() < 1e-9);
        assert!((r.factor_gflops - 80.0).abs() < 1e-9);
        assert!((r.factor_mnnz_s - 100.0).abs() < 1e-9);
        assert!((r.total_mdof_s - 1_000_000.0 / 0.75 / 1e6).abs() < 1e-9);
        assert!((r.solve_mdof_s - 80.0).abs() < 1e-9);
        // A refactor supersedes the first factor stage.
        d.push("klu-refactor", 100.0, 0, 0);
        assert!((d.rates().factor_mdof_s - 10.0).abs() < 1e-9);
        assert_eq!(d.rates().factor_gflops, 0.0);
        assert!(d.summary().contains("MDOF/s"));
        assert!(Diagnostics::default().rates() == Rates::default());
    }
}
