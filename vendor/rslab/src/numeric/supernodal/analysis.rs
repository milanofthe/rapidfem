//! The supernodal analysis shared by the LDL^T and LU paths: fill-reducing
//! ordering, elimination tree, supernodes and assembly-tree levels, computed
//! once per pattern, with the lazily built permutation program and
//! left-looking schedule every factorization of the pattern reuses.

use crate::error::RslabError;
use crate::numeric::settings::{in_scoped_pool, stack_for_depth, SolverSettings};
use crate::numeric::supernodal::LlSchedule;
use crate::symbolic::SymbolicFactorization;

/// Reusable symbolic analysis (fill-reducing ordering + assembly-tree levels)
/// for a fixed sparsity pattern. Value-independent: build once with [`analyze`]
/// and pass to [`factor_numeric`](crate::factor_numeric) for each set of numeric values sharing the
/// pattern - the PARDISO phase-1 analysis.
pub(crate) struct SupernodalAnalysis {
    pub(crate) inner: Option<SymbolicInner>,
    pub(crate) n: usize,
    pub(crate) nnz: usize,
}

impl SupernodalAnalysis {
    /// The fill-reducing ordering the analysis settled on (`perm[k]` the column that became
    /// column `k`); empty for `n = 0`.
    pub fn permutation(&self) -> &[usize] {
        self.inner.as_ref().map_or(&[], |i| &i.sym.perm[..])
    }
}

pub(crate) struct SymbolicInner {
    pub(crate) sym: SymbolicFactorization,
    /// Assembly-tree levels: `by_level[l]` are the supernodes at level `l`, all
    /// mutually independent (factored concurrently by the rayon driver).
    pub(crate) by_level: Vec<Vec<usize>>,
    /// Lazily built input program of the LDL^T path (`P^T A P`, lower fold):
    /// the permuted structure is fixed per pattern, so every (re)factorization
    /// reduces to one linear values pass. See
    /// [`crate::numeric::supernodal::InputProgram`].
    pub(crate) input: std::sync::OnceLock<crate::numeric::supernodal::InputProgram>,
    /// Lazily built left-looking schedule (row structures + updater lists),
    /// pattern-only and shared by the numeric drivers and the estimators.
    pub(crate) ll_schedule: std::sync::OnceLock<LlSchedule>,
}

impl SymbolicInner {
    /// The left-looking schedule, built on first use.
    pub(crate) fn schedule(&self) -> &LlSchedule {
        self.ll_schedule.get_or_init(|| {
            crate::logging::timed(
                || "analysis: schedule".into(),
                || LlSchedule::build(&self.sym),
            )
        })
    }
}

impl SupernodalAnalysis {
    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Internal accessor for the unsymmetric LU driver: the symbolic
    /// factorization and the precomputed assembly-tree levels. `None` for the
    /// empty (`n == 0`) analysis.
    pub(crate) fn sym_and_levels(&self) -> Option<(&SymbolicFactorization, &[Vec<usize>])> {
        self.inner.as_ref().map(|i| (&i.sym, i.by_level.as_slice()))
    }

    /// The cached pattern-only left-looking schedule (row structures + updater
    /// lists), built on first use and shared by the numeric drivers and the
    /// a-priori estimators. `None` for the empty analysis.
    pub(crate) fn ll_schedule(&self) -> Option<&LlSchedule> {
        self.inner.as_ref().map(SymbolicInner::schedule)
    }

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)`: the number of
    /// eliminated columns and the full front height. The raw material for
    /// factorization-cost diagnostics - front-size distribution (small vs dense
    /// fronts -> BLAS-2 vs BLAS-3 efficiency) and a factor-flop estimate.
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        match &self.inner {
            Some(i) => i.sym.supernodes.iter().map(|s| (s.ncol, s.nrow)).collect(),
            None => Vec::new(),
        }
    }

    pub fn n_supernodes(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.sym.supernodes.len())
    }

    /// Rows of the largest front after amalgamation.
    pub fn max_front(&self) -> usize {
        self.inner
            .as_ref()
            .and_then(|i| i.sym.supernodes.iter().map(|s| s.nrow).max())
            .unwrap_or(0)
    }

    /// The decisions the analysis took on its own (what `Auto` resolved to),
    /// for the [`Diagnostics`](crate::Diagnostics) of every factorization
    /// reusing it. `requested` is the ordering the caller asked for.
    pub fn decisions(
        &self,
        requested: crate::symbolic::OrderingMethod,
    ) -> crate::diagnostics::Decisions {
        let mut d = crate::diagnostics::Decisions {
            ordering_requested: format!("{requested:?}"),
            n_supernodes: self.n_supernodes(),
            max_front: self.max_front(),
            tree_levels: self.n_levels(),
            ..Default::default()
        };
        match &self.inner {
            Some(i) => {
                d.ordering_used = format!("{:?}", i.sym.resolved_method);
                d.amalgamation = format!("{:?}", i.sym.resolved_amalgamation);
            }
            None => d.ordering_used = d.ordering_requested.clone(),
        }
        d
    }

    /// Number of assembly-tree levels (the level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.inner.as_ref().map_or(0, |i| i.by_level.len())
    }

    /// Supernode count per assembly-tree level, leaves first. `level_widths()[l]`
    /// is the number of mutually independent fronts at level `l` - the available
    /// tree-parallelism at that depth. Wide near the leaves, narrowing to (often)
    /// a single chain at the root; the shape that decides whether tree-parallelism
    /// alone saturates the cores or the top fronts need node-parallelism.
    pub fn level_widths(&self) -> Vec<usize> {
        self.inner
            .as_ref()
            .map_or_else(Vec::new, |i| i.by_level.iter().map(|lv| lv.len()).collect())
    }
}

/// Analyze a sparsity pattern (`n`, CSC `col_ptr`/`row_idx`, lower
/// triangle): value-independent, reused by every factorization of the pattern.
pub(crate) fn analyze_with(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    opts: &SolverSettings,
) -> Result<SupernodalAnalysis, RslabError> {
    // The symbolic build (ordering, elimination tree, supernode amalgamation,
    // postorder) can recurse to O(n) on pathological patterns - dense/random
    // graphs where nested dissection finds no good separators - and would overflow
    // the caller's stack. Run it in a scoped pool whose workers have a stack sized
    // to the problem (committed on demand), the same robustness mechanism the
    // factorization uses. Shallow analyses get the floor stack at negligible cost.
    //
    // The pool is sized to the settings' thread budget (the same
    // solver-in-the-loop contract the factorization honours), so every parallel
    // step inside the analysis - notably the ND seed ensemble - respects the
    // configured worker count instead of grabbing all cores.
    in_scoped_pool(opts.resolved_threads(), stack_for_depth(n), || {
        analyze_with_inner(n, col_ptr, row_idx, opts)
    })
}

fn analyze_with_inner(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    opts: &SolverSettings,
) -> Result<SupernodalAnalysis, RslabError> {
    let nnz = row_idx.len();
    if n == 0 {
        return Ok(SupernodalAnalysis {
            inner: None,
            n: 0,
            nnz,
        });
    }
    let sym = crate::logging::timed(
        || format!("analysis: symbolic {:?}", opts.ordering.method),
        || crate::symbolic::analyze(n, col_ptr, row_idx, &opts.ordering, &opts.amalgamation),
    )?;

    // Assembly-tree levels: level(s) = 1 + max(level(children)); same-level
    // supernodes are mutually independent.
    let nsuper = sym.supernodes.len();
    let mut level = vec![0usize; nsuper];
    let mut max_level = 0usize;
    for s in 0..nsuper {
        let mut lv = 0usize;
        for &ch in &sym.supernodes[s].children {
            lv = lv.max(level[ch] + 1);
        }
        level[s] = lv;
        max_level = max_level.max(lv);
    }
    let mut by_level: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for (s, &lv) in level.iter().enumerate() {
        by_level[lv].push(s);
    }

    Ok(SupernodalAnalysis {
        inner: Some(SymbolicInner {
            sym,
            by_level,
            input: std::sync::OnceLock::new(),
            ll_schedule: std::sync::OnceLock::new(),
        }),
        n,
        nnz,
    })
}

/// Realize a [`Threads::Auto`] policy from a symbolic analysis: compute the three
/// predictive features (factor-flops, max front height, max tree width) and apply
/// the [`recommend_threads_from`] policy,
/// capped at `max_cores`. Value-independent, so it is the same for every scalar.
pub(crate) fn recommend_threads_for_sym(symb: &SupernodalAnalysis, max_cores: usize) -> usize {
    let fd = symb.front_dims();
    let flops: u64 = fd
        .iter()
        .map(|&(nc, nr)| (nr as u64) * (nr as u64) * (nc as u64))
        .sum();
    let front_nrow_max = fd.iter().map(|&(_, nr)| nr).max().unwrap_or(0);
    let tree_width_max = symb.level_widths().into_iter().max().unwrap_or(0);
    recommend_threads_from(flops, front_nrow_max, tree_width_max, max_cores)
}

/// The worker count of [`Threads::Auto`](crate::Threads::Auto), at most
/// `cap`: the calibrated cost model where the one-time install diagnosis has
/// run (feature `tuning`), else the structural predictor.
pub(crate) fn auto_threads(
    symb: &SupernodalAnalysis,
    estimate: &crate::diagnostics::MemoryEstimate,
    cap: usize,
) -> usize {
    #[cfg(feature = "tuning")]
    if let Some((cores, calib)) = crate::tuning::cached_calibration() {
        return crate::tuning::recommend_threads_cost_model(estimate, &calib, cap, cores);
    }
    let _ = estimate;
    recommend_threads_for_sym(symb, cap)
}

/// The data-driven single-solve thread-count policy, as a free function over the
/// three predictive features, so the factor path can apply it straight from the
/// symbolic analysis. Returns a worker count in `1..=max_cores`.
fn recommend_threads_from(
    factor_flops: u64,
    front_nrow_max: usize,
    tree_width_max: usize,
    max_cores: usize,
) -> usize {
    let cores = max_cores.max(1);
    // Thin fronts + narrow tree: no node-parallelism (tiny fronts) and no
    // tree-parallelism (path-like) to exploit - oversubscription only hurts.
    if front_nrow_max < 512 && tree_width_max < 128 {
        return cores.min(2);
    }
    // Tiny total work: parallel scheduling overhead dominates the factorization.
    if factor_flops < 300_000_000 {
        return cores.min(4);
    }
    cores
}
