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
    /// Scalar size in bytes (`16` for `Complex<f64>`, `8` for `f64`, …).
    pub value_bytes: usize,
    /// Structural nonzeros in the factor (`L`+`U` for LU, `L` for LDLᵀ) - an upper
    /// bound on the emitted factor (numeric cancellation can only lower it).
    pub factor_nnz: u64,
    /// Bytes of the resident factor (the CSC output): `factor_nnz·(value+index)`.
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
    /// Geometric factorization work proxy `Σ nrow²·ncol` over supernodes (type-
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
            "transient-peak ≤ {:.0} MB (panels {:.0} + factor {:.0} + input/scratch); \
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
pub(crate) fn estimate_left_looking(
    nsuper: usize,
    panel_bytes: &dyn Fn(usize) -> u64,
    compact_bytes: &dyn Fn(usize) -> u64,
    update_list: &[Vec<usize>],
    value_bytes: usize,
    input_bytes: u64,
) -> MemoryEstimate {
    let mut refc = vec![0usize; nsuper];
    for ul in update_list {
        for &k in ul {
            refc[k] += 1;
        }
    }
    let panels_all: u64 = (0..nsuper).map(panel_bytes).sum();
    let factor_bytes: u64 = (0..nsuper).map(compact_bytes).sum();

    let mut live_panels: i64 = 0;
    let mut compact: i64 = 0;
    let mut peak: i64 = 0;
    for s in 0..nsuper {
        live_panels += panel_bytes(s) as i64;
        for &k in &update_list[s] {
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
    // small absolute floor - tuned so the bound stays ≥ the measured peak across
    // sizes (validated: est/measured ≈ 1.0-1.2×), never under-predicting.
    let scratch = (panels_all + factor_bytes) / 4 + 32_000_000;
    let transient = panels_all + factor_bytes + input_bytes + scratch;
    MemoryEstimate {
        value_bytes,
        factor_nnz: factor_bytes / (value_bytes as u64 + 8).max(1),
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
/// fronts plus the contribution blocks (`cnrow²` each) of completed subtrees not
/// yet consumed by their parent. The driver factors a whole assembly-tree level
/// concurrently, so the conservative peak is, over the levels, the level's total
/// front memory (`Σ nrow²`) plus the contribution blocks of its children feeding
/// the assembly. Assuming a full level live at once never under-predicts at any
/// thread count - the transient the left-looking estimate does not capture.
pub(crate) fn estimate_multifrontal_active_peak(
    by_level: &[Vec<usize>],
    nrow: &dyn Fn(usize) -> u64,
    ncol: &dyn Fn(usize) -> u64,
    children: &[Vec<usize>],
    value_bytes: u64,
) -> u64 {
    let cb = |s: usize| -> u64 {
        let cn = nrow(s).saturating_sub(ncol(s));
        cn * cn * value_bytes
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

/// Per-stage diagnostics for one factorization. Per-call and concurrency-safe (no
/// global state), so a solver-in-the-loop with many concurrent solves gets correct
/// per-solve numbers. Carries the a-priori [`MemoryEstimate`] alongside the
/// measured factor time for estimate-vs-actual feedback.
#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    pub stages: Vec<StageReport>,
    pub threads: usize,
    pub factor_nnz: u64,
    pub estimate: Option<MemoryEstimate>,
}

impl Diagnostics {
    pub fn total_ms(&self) -> f64 {
        self.stages.iter().map(|s| s.wall_ms).sum()
    }
    pub fn push(&mut self, name: &'static str, wall_ms: f64, flops: u64, bytes: u64) {
        self.stages.push(StageReport {
            name,
            wall_ms,
            flops,
            bytes,
        });
    }
}

impl fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "factorization diagnostics (threads={}, factor_nnz={}):",
            self.threads, self.factor_nnz
        )?;
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
        Ok(())
    }
}
