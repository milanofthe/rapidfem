//! Generic multifrontal sparse LDLᵀ factorization over any [`Scalar`] field.
//!
//! This drives a full sparse symmetric-indefinite solve for both the real
//! (`f64`) and complex-*symmetric* (`Complex<f64>`, PARDISO `mtype 6`) paths by
//! reusing the existing **value-agnostic** symbolic analysis (ordering,
//! elimination tree, supernode amalgamation) and applying the generic dense
//! Bunch-Kaufman kernel from [`crate::dense::ldlt_generic`] front-by-front.
//!
//! This is the single, data-type-generic symmetric multifrontal driver (the
//! former f64-dedicated driver has been removed). It is rayon-parallel with a
//! `gemm` BLAS-3 Schur update and relaxed amalgamation; delayed pivoting and
//! the remaining rslab robustness features are being ported in.
//!
//! ## Current pivoting scope
//!
//! * Pivoting is restricted to the **fully-summed block** of each front (no
//!   delayed pivoting). This produces a valid factorization whenever each
//!   fully-summed block is nonsingular; pathological indefinite cases that
//!   would require delaying a pivot to the parent are out of scope for now and
//!   surface as [`RslabError::NumericallyRankDeficient`].
//! * The reassembled factor is held as a dense `n×n` global `L`. This is
//!   `O(n²)` memory and is a correctness-first choice; a sparse-CSC global `L`
//!   with a supernodal triangular solve is a later optimization.
//!
//! The result is returned as an [`LdltFactors`] in factorization order, so the
//! generic [`solve_ldlt`](crate::dense::ldlt_generic::solve_ldlt) handles the
//! triangular/diagonal solves and permutation directly.

use crate::dense::ldlt_generic::{bk_alpha, swap_sym_lower, LdltFactors};
use crate::error::RslabError;
use crate::inertia::Inertia;
use crate::scalar::Scalar;

/// Scale-invariant singularity floor for a 2×2 Bunch-Kaufman pivot: a block
/// whose `|det|` falls below `GROWTH_EPS · scale²` (scale = the largest block
/// entry magnitude) is numerically singular - rejected in exact mode and lifted
/// in static-pivot mode. Bounds the element growth `1/|det|` can otherwise
/// inject into the trailing update.
const GROWTH_EPS: f64 = 1e-14;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::{
    symbolic_factorize_with_method, OrderingMethod, RelaxAmalgamation, SupernodeParams,
    SymbolicFactorization,
};
use rayon::prelude::*;

use crate::numeric::gemm_tuning::KernelTuning;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

// Opt-in cdiv sub-phase profiler (set `RLA_PROFILE=1`): serial BK panel (getf2)
// vs the deferred Schur update, summed across worker threads. Zero cost when off.
static PROF_LDLT_GETF2_NS: AtomicU64 = AtomicU64::new(0);
static PROF_LDLT_SCHUR_NS: AtomicU64 = AtomicU64::new(0);
static PROF_LDLT_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[inline]
fn ldlt_prof_on() -> bool {
    *PROF_LDLT_FLAG.get_or_init(|| {
        std::env::var("RLA_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Action to take when a near-zero pivot is encountered during factorization.
///
/// This is the static-pivoting policy knob shared by the symmetric LDLᵀ and the
/// unsymmetric LU paths (via [`SolverSettings`] and the LU options).
#[derive(Debug, Clone)]
pub enum ZeroPivotAction {
    /// Accept the tiny pivot at face value (zero the column, count as a zero in
    /// the inertia signature, flag for iterative refinement). The perturbation
    /// magnitude is unbounded - use only when downstream code tolerates sign
    /// loss in the perturbed positions and re-checks inertia.
    ForceAccept,
    /// Return [`RslabError::NumericallyRankDeficient`].
    Fail,
    /// Replace the tiny pivot with `sign(d) · max(|d|, abs_floor)`, keeping the
    /// column live (LAPACK / MA57-style static pivoting). The factor satisfies
    /// `L·D·Lᵀ = A + Δ` for the produced `L`, `D`; `Δ` is bounded in the worst
    /// case by `‖A[:,k]‖² / abs_floor`, so drive iterative refinement against
    /// the unperturbed `A` for tight tolerances. A typical recipe is
    /// `abs_floor = eps_rel · ‖A‖∞` with `eps_rel ∈ [1e-12, 1e-8]`.
    PerturbToEps { abs_floor: f64 },
}
use std::sync::atomic::{AtomicUsize, Ordering};

/// Child-reordering strategy, selected per analysis via [`SolverSettings`] - the
/// composable replacement for the old process-wide Liu toggle. A pure scheduling
/// hint: it changes neither the factor, the fill, nor the e-numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReorderMode {
    /// Hybrid Liu (1986) contribution-stack minimization (default): reorder
    /// children to shrink the transient CB-stack peak where it is large, keep
    /// the natural leaf order elsewhere. Memory-light, ≈ throughput-neutral.
    #[default]
    HybridLiu,
    /// No child reordering: maximum leaf parallelism, larger CB-stack peak - for
    /// when memory is not the constraint.
    Off,
}

/// Factor emit/memory strategy - composable via [`SolverSettings::with_memory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryMode {
    /// Collect every front's factor, then emit the global `L`/`U`.
    Eager,
    /// Free each front's dense factor as soon as it is emitted into the global
    /// structure (default) - lower peak RSS at no accuracy cost: bit-identical
    /// factors, removes the emit-time per-front + global overlap.
    #[default]
    LowMemory,
}

/// Block-Low-Rank strategy - composable via [`SolverSettings::with_blr`]. BLR
/// makes the factor **approximate** (a preconditioner); drive iterative
/// refinement against the original matrix.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BlrMode {
    /// Dense fronts and contribution blocks (default, exact).
    #[default]
    Off,
    /// Store each large contribution block block-low-rank on the assembly stack:
    /// `eps` per-tile Frobenius tolerance, `min_cnrow` CB-size threshold, `b`
    /// tile size. Shrinks the live CB-stack transient.
    ContributionBlocks {
        eps: f64,
        min_cnrow: usize,
        b: usize,
    },
}

impl BlrMode {
    /// BLR contribution blocks at per-tile tolerance `eps` with the default
    /// `min_cnrow = 256`, `b = 256`.
    pub fn contribution_blocks(eps: f64) -> Self {
        BlrMode::ContributionBlocks {
            eps,
            min_cnrow: 256,
            b: 256,
        }
    }
}

/// Numeric factorization algorithm - composable via [`SolverSettings::with_method`].
/// Both produce the same factor (numerically equivalent); they differ in the
/// transient-memory and scheduling profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FactorMethod {
    /// Multifrontal: assembly-tree of dense fronts, rayon work-stealing parallel,
    /// with full pivoting (Bunch-Kaufman 2×2 for LDLᵀ, partial for LU). Carries
    /// the contribution-block stack + a per-front extract transient. Kept as the
    /// opt-in alternative (via [`with_method`]) for cross-checking and for fronts
    /// where the per-front extract layout is preferable; the default is
    /// [`LeftLooking`](Self::LeftLooking).
    ///
    /// [`with_method`]: SolverSettings::with_method
    Multifrontal,
    /// Supernodal left-looking (**the default**, and the [`preconditioner`]
    /// choice): each panel pulls BLAS-3 updates from its factored descendants -
    /// **no contribution-block stack, no extract phase** (the PARDISO transient
    /// profile), parallel over the assembly tree, lower fill, faster than
    /// multifrontal on the MoM matrices. Uses **Bunch-Kaufman 1×1/2×2 pivoting**
    /// (LDLᵀ) / **threshold partial pivoting** (LU), bounded to each panel's
    /// fully-summed block - pivoting parity with the multifrontal path - so it
    /// handles indefinite (zero-/tiny-diagonal) systems directly. The
    /// memory/throughput-optimal path for both exact direct solves and the
    /// equilibrated preconditioner.
    ///
    /// [`preconditioner`]: SolverSettings::preconditioner
    #[default]
    LeftLooking,
    /// Right-looking supernodal: factor the assembly-tree fronts sequentially in
    /// postorder, holding **every** front live and pushing each factored front's
    /// contribution into its parent (no contribution-block frontier free, no
    /// per-front extract). The classic right-looking schedule - higher transient
    /// memory (all fronts co-resident) traded for a simple, barrier-free push. The
    /// numeric factor is equivalent to the other methods; this is an **opt-in**
    /// method (not auto-selected) for the memory/schedule trade on classes where a
    /// dense all-fronts residency is acceptable. See Davis 2006, *Direct Methods
    /// for Sparse Linear Systems*, §4 (left-/right-looking supernodal schedules).
    RightLooking,
}

/// Options controlling the generic multifrontal factorization. Defaults give an
/// **exact** complete factorization that fails on rank deficiency. Relaxing
/// them turns the factorization into a robust, memory-light **preconditioner**.
/// All knobs compose via the `with_*` builders.
#[derive(Debug, Clone)]
pub struct SolverSettings {
    /// Near-zero pivot policy. Reuses rslab's [`ZeroPivotAction`]: `Fail`
    /// (exact, default) returns [`RslabError::NumericallyRankDeficient`] on a
    /// singular pivot; `PerturbToEps { abs_floor }` is robust static pivoting -
    /// a pivot below `abs_floor` is lifted to that floor (the
    /// complex-symmetric analogue of rslab's f64 `perturb_to_floor`), so the
    /// factorization never fails and produces `L D Lᵀ = A + E` for small `E`.
    /// That is exactly the never-fail behaviour a preconditioner needs.
    pub on_zero_pivot: ZeroPivotAction,
    /// Threshold dropping for incomplete factorization. When `Some(tau)`, fill
    /// entries of `L` with magnitude below `tau` (relative to the column) are
    /// discarded, trading factor accuracy for memory. `None` = complete
    /// factorization. (Wired in a later stage.)
    pub drop_tol: Option<f64>,
    /// Factor emit/memory strategy (peak-RSS vs simplicity). Default
    /// [`LowMemory`] (lower peak, bit-identical factors).
    ///
    /// [`LowMemory`]: MemoryMode::LowMemory
    pub memory: MemoryMode,
    /// Block-Low-Rank strategy. Default [`Off`] (exact dense fronts).
    ///
    /// [`Off`]: BlrMode::Off
    pub blr: BlrMode,
    /// Numeric factorization algorithm. Default [`LeftLooking`] (lower transient
    /// memory + faster); override with [`with_method`](Self::with_method) to force
    /// the [`Multifrontal`] path.
    ///
    /// [`LeftLooking`]: FactorMethod::LeftLooking
    /// [`Multifrontal`]: FactorMethod::Multifrontal
    pub method: FactorMethod,
    /// Worker-thread policy for this factorization, run in a **scoped** rayon pool
    /// (not the global pool). Either a [`Fixed`](Threads::Fixed) count or
    /// [`Auto`](Threads::Auto) - the data-driven per-matrix predictor, capped at a
    /// user-defined maximum. **Default [`Auto`](Threads::Auto)** (predict, up to
    /// all cores). The numeric result is bit-identical regardless of this value.
    pub threads: Threads,

    // ---- Analysis-phase knobs (read by `analyze_with`; ignored by `factor`) ----
    /// Child-reordering strategy (CB-stack peak vs leaf parallelism). Analyze-time.
    pub reorder: ReorderMode,
    /// Fill-reducing ordering (the cuDSS `REORDERING_ALG` analogue). Analyze-time.
    /// Default [`OrderingMethod::Auto`] (adaptive per-matrix choice).
    pub ordering: OrderingMethod,
    /// Supernode amalgamation `nemin` (merge-candidate column threshold). Default
    /// `16`. Smaller = finer supernodes (less fill, more per-front overhead).
    /// Analyze-time.
    pub nemin: usize,
    /// Relaxed (fill-tolerant) amalgamation thresholds, the multifrontal throughput
    /// lever. `Some` (default `≤256` wide, `≤64` extra rows) trades a little
    /// explicit-zero fill for wider, higher-rank dense fronts. Analyze-time.
    pub relax: Option<RelaxAmalgamation>,

    // ---- Kernel scheduling knobs (formerly process-wide atomics) ----
    /// Bunch-Kaufman / LU panel width (blocking factor). Default `64`. Changes the
    /// pivot search window (a different but equally valid factor), not the answer.
    /// Clamped to at least 8 on use.
    pub panel_nb: usize,
    /// Below this flop count a contribution update runs as a scalar triple loop
    /// instead of a SIMD GEMM. Default [`DEFAULT_SCALAR_GATE`](crate::DEFAULT_SCALAR_GATE).
    pub scalar_gate: usize,
    /// At/above this flop count a cmod-class GEMM runs rayon-parallel. Default
    /// [`DEFAULT_PAR_GEMM`](crate::DEFAULT_PAR_GEMM).
    pub par_gemm: usize,
    /// At/above this flop count the panel-trailing / Schur / LU-front GEMM runs
    /// rayon-parallel (the top-of-tree node-parallelism lever). Default
    /// [`DEFAULT_PAR_CDIV`](crate::DEFAULT_PAR_CDIV).
    pub par_cdiv: usize,
    /// Use the SIMD GEMM (vs the scalar triple loop) for the front Schur update.
    /// Default `true`. A kernel A/B knob for benchmarking.
    pub use_gemm_schur: bool,
    /// Threshold partial-pivoting tolerance `u ∈ [0, 1]` for the **left-looking LU**
    /// path (the shipped default for unsymmetric matrices). The diagonal pivot is
    /// kept unless it falls below `u · |colmax|` in its fully-summed block. `u = 1`
    /// is full partial pivoting; `u → 0` keeps the diagonal unless exactly zero
    /// (least fill, least stable). Default
    /// `DEFAULT_PIVOT_U = 0.1` (a `gemm_tuning` internal constant).
    /// Ignored by the LDLᵀ path (Bunch-Kaufman) and the multifrontal LU front
    /// (which uses full pivoting). Numeric-phase knob; a lower `u` trades a little
    /// stability (backed by the near-zero pivot policy) for less fill and speed on
    /// well-scaled / diagonally-dominant systems.
    pub pivot_u: f64,
    /// Symmetric equilibration strategy `Â = D A D` applied by [`LdltSolver`](crate::LdltSolver)
    /// before factoring. Default [`OnePassInfNorm`](crate::ScalingStrategy::OnePassInfNorm)
    /// (the historical one-pass ∞-norm, bit-identical to before this knob).
    /// [`Identity`](crate::ScalingStrategy::Identity) disables scaling;
    /// [`InfNorm`](crate::ScalingStrategy::InfNorm) is the iterative Knight-Ruiz
    /// (Ruiz) equilibration; [`Auto`](crate::ScalingStrategy::Auto) routes to
    /// MC64 matching on the arrow-KKT signature else ∞-norm. Scaling changes only
    /// values (not the pattern), so the a-priori memory estimate is unaffected.
    /// Consumed by the symmetric path; the unsymmetric LU path uses its own
    /// two-sided row/column equilibration.
    pub scaling: crate::scaling::ScalingStrategy,
}

/// Worker-thread policy for a factorization. The numeric result is bit-identical
/// regardless of which is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Threads {
    /// Exactly this many workers. `0` = all logical cores. Use a small fixed
    /// budget for **solver-in-the-loop** (many concurrent solves coexisting on
    /// the machine without oversubscription).
    Fixed(usize),
    /// Predict the worker count per-matrix from the structural fingerprint (the
    /// validated [`recommend_threads`](crate::StructuralFeatures::recommend_threads)
    /// policy: thin / tiny systems stay low where they would only regress, big
    /// BLAS-3-rich systems use the cores), **capped at `max`** (`0` = all logical
    /// cores). The single-solve default: best throughput without oversubscribing
    /// the matrices that do not scale.
    Auto {
        /// Upper bound on the predicted worker count (`0` = all logical cores).
        max: usize,
    },
    /// Use the **current** rayon pool as-is, without building a scoped pool. The
    /// solver-in-the-loop path: build **one** bounded pool (e.g. 4 workers) with
    /// [`with_threads`](crate::with_threads) and run the factorization *and* every
    /// iterative solve inside it, so both phases share the same capped pool with no
    /// per-call thread spawn. The numeric factor is unchanged.
    Ambient,
}

impl Default for Threads {
    fn default() -> Self {
        // Cap at 4 workers by default: our strong-scaling data puts the efficiency
        // knee at ~4-6 threads, so 4 is the pareto-optimal throughput-per-core point
        // and the safe default for concurrent / embedded (solver-in-the-loop) use.
        // `Auto` still predicts a smaller count per matrix where more would regress.
        Threads::Auto { max: 4 }
    }
}

/// All logical cores (the `0` sentinel resolution).
fn all_cores() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

impl Threads {
    /// Resolve to a concrete worker count. `recommend(cap)` is the structural
    /// predictor (already clamped to `cap`); it is only invoked in [`Auto`] mode.
    ///
    /// [`Auto`]: Threads::Auto
    pub(crate) fn resolve(self, recommend: impl FnOnce(usize) -> usize) -> usize {
        match self {
            Threads::Fixed(0) => all_cores(),
            Threads::Fixed(n) => n,
            Threads::Auto { max } => recommend(if max == 0 { all_cores() } else { max }),
            Threads::Ambient => rayon::current_num_threads().max(1),
        }
    }

    /// Run `f` under this thread policy. [`Ambient`](Threads::Ambient) runs on the
    /// current rayon pool with **no new pool spawned** (solver-in-the-loop); every
    /// other policy resolves a worker count and runs `f` in a scoped pool of that
    /// width with a `stack_bytes` worker stack. Centralizes the dispatch so all
    /// factorization paths honour `Ambient` identically.
    pub(crate) fn run<R: Send>(
        self,
        stack_bytes: usize,
        recommend: impl FnOnce(usize) -> usize,
        f: impl FnOnce() -> R + Send,
    ) -> R {
        match self {
            Threads::Ambient => f(),
            policy => in_scoped_pool(policy.resolve(recommend), stack_bytes, f),
        }
    }
}

/// Run `f` inside a **scoped** rayon thread pool of `threads` workers, so this
/// factorization's parallelism is bounded and concurrent solves coexist instead of
/// each grabbing the global pool. Falls back to running on the current pool if the
/// build fails. `threads == 0` means all logical cores.
pub(crate) fn in_scoped_pool<R: Send>(
    threads: usize,
    stack_bytes: usize,
    f: impl FnOnce() -> R + Send,
) -> R {
    let n = if threads == 0 {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        threads
    };
    let mut builder = rayon::ThreadPoolBuilder::new().num_threads(n);
    if stack_bytes > 0 {
        builder = builder.stack_size(stack_bytes);
    }
    match builder.build() {
        Ok(pool) => pool.install(f),
        Err(_) => f(),
    }
}

/// Run `f` in a scoped rayon pool of `threads` workers (`0` = all logical cores),
/// then tear the pool down. The **solver-in-the-loop / embedded** entry point:
/// build **one** capped pool and drive many solves through it without a per-call
/// thread spawn.
///
/// Typical pattern - factor once (its own bounded, depth-stacked pool via the
/// default [`Threads::Auto`]`{max:4}`), then run the multi-RHS GMRES loop capped
/// at the same width:
/// ```ignore
/// let lu = factor_general_lu(&a, &SolverSettings::default())?;   // Auto{max:4}
/// with_threads(4, || {
///     for rhs in batches { let _ = gmres_block(&a, rhs, s, &lu, tol, it, m, None)?; }
///     Ok::<_, RslabError>(())
/// })?;
/// ```
/// The block GMRES orthogonalization picks up this pool automatically (it uses the
/// ambient rayon pool). To also run the *factorization* on this shared pool (e.g.
/// re-factoring every Newton step), pass [`Threads::Ambient`] in the settings.
///
/// The pool gets a 16 MB worker stack (the factorization stack floor), so an
/// `Ambient` factorization inside is safe for typical assembly-tree depths; the
/// iterative solvers do not deep-recurse. For pathologically deep trees (banded /
/// 1D at low `nemin`) build the pool yourself with a larger `stack_size`.
pub fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    in_scoped_pool(threads, 16 * 1024 * 1024, f)
}

/// Maximum supernode-tree height (root-to-leaf), the recursion depth of the tree
/// factorization. Computed by an **iterative** post-order DFS (its own heap stack,
/// so it never recurses) and so is correct for any supernode numbering - not
/// assuming children precede their parent. O(#supernodes).
pub(crate) fn supernode_tree_depth(sym: &SymbolicFactorization) -> usize {
    let nsuper = sym.supernodes.len();
    let mut height = vec![0usize; nsuper];
    let mut is_child = vec![false; nsuper];
    for s in 0..nsuper {
        for &c in &sym.supernodes[s].children {
            is_child[c] = true;
        }
    }
    let mut max_h = 0;
    let mut stack: Vec<(usize, usize)> = Vec::new(); // (node, next child index)
    for (r, &child) in is_child.iter().enumerate() {
        if child {
            continue;
        }
        stack.push((r, 0));
        while let Some(&(node, ci)) = stack.last() {
            let children = &sym.supernodes[node].children;
            if ci < children.len() {
                if let Some(top) = stack.last_mut() {
                    top.1 += 1;
                }
                stack.push((children[ci], 0));
            } else {
                let mut h = 1;
                for &c in children {
                    h = h.max(height[c] + 1);
                }
                height[node] = h;
                max_h = max_h.max(h);
                stack.pop();
            }
        }
    }
    max_h
}

/// Worker-thread stack size for a tree of the given depth. The recursive tree
/// factorization (`factor_subtree` / `ll_factor_subtree` and the LU twins) uses
/// O(depth) native stack; depth is O(log n) for nested-dissection orderings but
/// O(#supernodes) for deep chain trees - banded / 1D-like patterns, especially
/// with low `nemin`. Sizing the worker stack to the analyzed depth keeps the
/// factorization robust for every knob setting (the address space is reserved,
/// committed only as the recursion descends), instead of a fixed guess that a
/// deep enough chain overflows. `0` (shallow trees) keeps the rayon default.
pub(crate) fn stack_for_depth(depth: usize) -> usize {
    const FRAME: usize = 32 * 1024; // per-frame budget (LL ~6.7 KB measured; MF larger)
    const MIN: usize = 16 * 1024 * 1024; // floor (>= the rayon default; covers ~depth 500)
    const MAX: usize = 8 * 1024 * 1024 * 1024; // 8 GB cap (depth ~256k)
                                               // Always set an explicit, depth-proportional stack - never fall back to the
                                               // small rayon default, which a moderate depth (a few hundred supernodes, as a
                                               // banded matrix amalgamates to) already overflows.
    depth.saturating_mul(FRAME).clamp(MIN, MAX)
}

impl Default for SolverSettings {
    fn default() -> Self {
        use crate::numeric::gemm_tuning::{
            DEFAULT_PANEL_NB, DEFAULT_PAR_CDIV, DEFAULT_PAR_GEMM, DEFAULT_PIVOT_U,
            DEFAULT_SCALAR_GATE,
        };
        Self {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: None,
            memory: MemoryMode::LowMemory,
            blr: BlrMode::Off,
            method: FactorMethod::LeftLooking,
            threads: Threads::default(),
            // Analysis-phase defaults (reproduce the historically-tuned analysis).
            reorder: ReorderMode::default(),
            ordering: OrderingMethod::default(),
            nemin: 16,
            relax: Some(RelaxAmalgamation {
                max_width: 256,
                max_extra_rows: 64,
            }),
            // Kernel defaults (reproduce the former process-wide atomic defaults).
            panel_nb: DEFAULT_PANEL_NB,
            scalar_gate: DEFAULT_SCALAR_GATE,
            par_gemm: DEFAULT_PAR_GEMM,
            par_cdiv: DEFAULT_PAR_CDIV,
            use_gemm_schur: true,
            pivot_u: DEFAULT_PIVOT_U,
            scaling: crate::scaling::ScalingStrategy::OnePassInfNorm,
        }
    }
}

impl SolverSettings {
    /// Exact, complete factorization (the default): fail on a singular pivot,
    /// no fill dropping. Use for a direct solve where accuracy is required.
    pub fn exact() -> Self {
        Self::default()
    }

    /// Robust never-fail **preconditioner** mode: static pivoting replaces any
    /// pivot below `abs_floor` (typically `eps_rel·‖A‖`) so the factorization
    /// always succeeds. Compose with [`with_drop_tol`](Self::with_drop_tol) for
    /// an incomplete preconditioner.
    pub fn preconditioner(abs_floor: f64) -> Self {
        Self {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor },
            // The equilibrated, refined preconditioner is exactly where the
            // memory/throughput-optimal left-looking path (Bunch-Kaufman 1×1/2×2)
            // belongs; override with `with_method` to force the multifrontal path.
            method: FactorMethod::LeftLooking,
            ..Self::default()
        }
    }

    /// Builder: enable incomplete-factor threshold dropping (`|fill| < tau` is
    /// discarded, relative to the column/row).
    pub fn with_drop_tol(mut self, tau: f64) -> Self {
        self.drop_tol = Some(tau);
        self
    }

    /// Builder: set the near-zero pivot policy.
    pub fn with_pivot(mut self, policy: ZeroPivotAction) -> Self {
        self.on_zero_pivot = policy;
        self
    }

    /// Builder: set the factor emit/memory strategy.
    pub fn with_memory(mut self, memory: MemoryMode) -> Self {
        self.memory = memory;
        self
    }

    /// Builder: set the Block-Low-Rank strategy (makes the factor a
    /// preconditioner - refine against the original matrix).
    pub fn with_blr(mut self, blr: BlrMode) -> Self {
        self.blr = blr;
        self
    }

    /// Builder: select the numeric factorization algorithm (multifrontal vs
    /// supernodal left-looking).
    pub fn with_method(mut self, method: FactorMethod) -> Self {
        self.method = method;
        self
    }

    /// Builder: set a **fixed** worker-thread budget (`0` = all logical cores).
    /// The factor runs in a scoped pool of this size so concurrent solves don't
    /// oversubscribe. Overrides the default [`Auto`](Threads::Auto) prediction.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = Threads::Fixed(threads);
        self
    }

    /// Builder: use the **auto** per-matrix thread predictor, capped at `max`
    /// (`0` = all logical cores). This is the default policy; use it to bound the
    /// predictor below the full core count.
    pub fn with_auto_threads(mut self, max: usize) -> Self {
        self.threads = Threads::Auto { max };
        self
    }

    /// Builder: set the worker-thread policy directly.
    pub fn with_thread_policy(mut self, threads: Threads) -> Self {
        self.threads = threads;
        self
    }

    /// Builder: set the child-reordering strategy (analyze-time).
    pub fn with_reorder(mut self, reorder: ReorderMode) -> Self {
        self.reorder = reorder;
        self
    }

    /// Builder: set the fill-reducing ordering method (analyze-time).
    pub fn with_ordering(mut self, ordering: OrderingMethod) -> Self {
        self.ordering = ordering;
        self
    }

    /// Builder: set the supernode amalgamation `nemin` (analyze-time).
    pub fn with_nemin(mut self, nemin: usize) -> Self {
        self.nemin = nemin;
        self
    }

    /// Builder: set the relaxed-amalgamation thresholds (`None` restricts to
    /// structural/size merges). Analyze-time.
    pub fn with_relax(mut self, relax: Option<RelaxAmalgamation>) -> Self {
        self.relax = relax;
        self
    }

    /// Builder: set the Bunch-Kaufman / LU panel width (kernel blocking factor).
    pub fn with_panel_nb(mut self, nb: usize) -> Self {
        self.panel_nb = nb;
        self
    }

    /// Builder: set the GEMM scheduling thresholds (scalar/SIMD and serial/parallel
    /// cutoffs) in one shot.
    pub fn with_gemm_thresholds(mut self, t: crate::numeric::gemm_tuning::GemmThresholds) -> Self {
        self.scalar_gate = t.scalar_gate;
        self.par_gemm = t.par_gemm;
        self.par_cdiv = t.par_cdiv;
        self
    }

    /// Builder: toggle the SIMD GEMM Schur update (vs the scalar triple loop).
    pub fn with_use_gemm_schur(mut self, on: bool) -> Self {
        self.use_gemm_schur = on;
        self
    }

    /// Builder: set the left-looking LU threshold partial-pivoting tolerance
    /// `u ∈ [0, 1]` (clamped). Default `0.1`; `1.0` is full partial pivoting.
    /// See [`pivot_u`](Self::pivot_u).
    pub fn with_pivot_u(mut self, u: f64) -> Self {
        self.pivot_u = u.clamp(0.0, 1.0);
        self
    }

    /// Builder: set the symmetric equilibration strategy (analyze/factor-time,
    /// symmetric path). See [`scaling`](Self::scaling).
    pub fn with_scaling(mut self, scaling: crate::scaling::ScalingStrategy) -> Self {
        self.scaling = scaling;
        self
    }

    /// The kernel scheduling knobs as a cheap `Copy` bundle, threaded into the
    /// dense-front / left-looking kernels (replaces the former atomic loads).
    pub(crate) fn kernel(&self) -> crate::numeric::gemm_tuning::KernelTuning {
        crate::numeric::gemm_tuning::KernelTuning {
            scalar_gate: self.scalar_gate,
            par_gemm: self.par_gemm,
            par_cdiv: self.par_cdiv,
            panel_nb: self.panel_nb.max(8),
            use_gemm_schur: self.use_gemm_schur,
            pivot_u: self.pivot_u.clamp(0.0, 1.0),
        }
    }

    /// A static upper bound on the worker count for *reporting*, without the
    /// structural predictor: a fixed count resolves exactly; an
    /// [`Auto`](Threads::Auto) policy reports its cap (all cores for `0`). The
    /// concrete count actually used is resolved at factor time and recorded in
    /// the [`Diagnostics`](crate::Diagnostics).
    pub fn resolved_threads(&self) -> usize {
        match self.threads {
            Threads::Fixed(0) | Threads::Auto { max: 0 } => all_cores(),
            Threads::Fixed(n) | Threads::Auto { max: n } => n,
            Threads::Ambient => rayon::current_num_threads().max(1),
        }
    }
}

/// Static-pivot perturbation, the complex-symmetric analogue of rslab's f64
/// `perturb_to_floor` (`dense::factor`): lift a pivot whose magnitude is below
/// `abs_floor` up to that floor, preserving phase. For `T = f64` this reduces
/// to `sign(d)·max(|d|, abs_floor)`, matching the real kernel.
#[inline]
pub(crate) fn perturb_pivot<T: Scalar>(d: T, abs_floor: f64) -> T {
    let mag = d.magnitude();
    if mag >= abs_floor {
        d
    } else if mag == 0.0 {
        T::from_real(abs_floor)
    } else {
        d * T::from_real(abs_floor / mag)
    }
}

/// Per-front partial-factorization output, in within-front pivot order.
struct FrontFactors<T> {
    /// Total front size (eliminated + contribution rows).
    nrow: usize,
    /// Number of eliminated (fully-summed) columns.
    nelim: usize,
    /// Pivot position → local row index (length `nrow`). Identity on the
    /// contribution rows `[nelim, nrow)`, which are never interchanged.
    perm: Vec<usize>,
    /// Unit lower `L` of the front, `nrow × nelim` column-major in pivot order.
    l: Vec<T>,
    /// `D` block diagonal, length `nelim`.
    d_diag: Vec<T>,
    /// `D` sub-diagonal, length `nelim`.
    d_subdiag: Vec<T>,
    /// `true` at the first column of each 2×2 block, length `nelim`.
    two_by_two: Vec<bool>,
    /// Number of pivots statically perturbed in this front.
    n_perturbed: usize,
    /// Inertia (signs of `D`) over this front's eliminated pivots. Exact for a
    /// real symmetric matrix; advisory (pivot real-part signs) for complex.
    inertia: Inertia,
}

/// Partially factor the first `ncol` (fully-summed) columns of a dense
/// lower-triangle front `f` (`nrow × nrow`, column-major) with Bunch-Kaufman
/// pivoting restricted to the fully-summed block. The entire trailing front is
/// updated; the trailing `[ncol, nrow)` block is returned as the contribution
/// block (`cnrow × cnrow` column-major lower triangle).
fn factor_front<T: Scalar>(
    f: &mut [T],
    nrow: usize,
    ncol: usize,
    perturb_floor: Option<f64>,
    kt: KernelTuning,
) -> Result<(FrontFactors<T>, Vec<T>), RslabError> {
    let n = nrow; // column stride
    let alpha = bk_alpha();
    let one = T::one();

    let mut perm: Vec<usize> = (0..nrow).collect();
    let mut d_diag = vec![T::zero(); ncol];
    let mut d_subdiag = vec![T::zero(); ncol];
    let mut two_by_two = vec![false; ncol];
    let mut n_perturbed = 0usize;
    let mut inertia = Inertia::new(0, 0, 0);
    // Reusable 2×2-pivot multiplier scratch, hoisted out of the pivot loop so an
    // indefinite front with many 2×2 blocks does not allocate per pivot. Only
    // entries `[k+2, n)` are ever written/read each step, so stale values left
    // below are never observed.
    let mut l1 = vec![T::zero(); nrow];
    let mut l2 = vec![T::zero(); nrow];
    // Per-panel trailing-GEMM scratch (reused across panels).
    let mut l21buf: Vec<T> = Vec::new();
    let mut gbuf: Vec<T> = Vec::new();
    let mut tmp: Vec<T> = Vec::new();

    // Blocked Bunch-Kaufman: factor the fully-summed columns in panels of width
    // `NB` with pivoting **bounded to the panel**, deferring each panel's
    // trailing Schur update to one SIMD GEMM (the BLAS-3 bulk, replacing the
    // scalar BLAS-2 column sweeps that dominated large fronts). The last column
    // of a panel has no in-panel candidate below it, so it is always a 1×1 step
    // - a 2×2 block can never straddle a panel boundary.
    let nb = kt.panel_nb;
    let mut kb = 0;
    while kb < ncol {
        let ke = (kb + nb).min(ncol);
        let mut k = kb;
        while k < ke {
            let absakk = f[k * n + k].magnitude();

            // colmax restricted to the in-panel rows (k+1)..ke.
            let mut colmax_sq = 0.0;
            let mut imax = k;
            for i in (k + 1)..ke {
                let m = f[k * n + i].magnitude_sq();
                if m > colmax_sq {
                    colmax_sq = m;
                    imax = i;
                }
            }
            let colmax = colmax_sq.sqrt();

            let kstep;
            let kp;
            if absakk.max(colmax) == 0.0 {
                // Fully zero pivot column. Exact mode fails; static-pivot mode
                // takes a 1×1 step and lets the perturbation below lift the zero
                // diagonal up to the floor.
                if perturb_floor.is_none() {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                kstep = 1;
                kp = k;
            } else if absakk >= alpha * colmax {
                kstep = 1;
                kp = k;
            } else {
                // rowmax in row imax, restricted to the fully-summed block (squared
                // domain, single final sqrt).
                let mut rowmax_sq = 0.0;
                for j in k..imax {
                    let m = f[j * n + imax].magnitude_sq();
                    if m > rowmax_sq {
                        rowmax_sq = m;
                    }
                }
                for i in (imax + 1)..ke {
                    let m = f[imax * n + i].magnitude_sq();
                    if m > rowmax_sq {
                        rowmax_sq = m;
                    }
                }
                let rowmax = rowmax_sq.sqrt();
                if absakk >= alpha * colmax * (colmax / rowmax) {
                    kstep = 1;
                    kp = k;
                } else if f[imax * n + imax].magnitude() >= alpha * rowmax {
                    kstep = 1;
                    kp = imax;
                } else {
                    kstep = 2;
                    kp = imax;
                }
            }

            if kstep == 1 {
                if kp != k {
                    swap_sym_lower(f, n, k, kp);
                    perm.swap(k, kp);
                }
                let mut d = f[k * n + k];
                match perturb_floor {
                    Some(floor) if d.magnitude() < floor => {
                        d = perturb_pivot(d, floor);
                        f[k * n + k] = d;
                        n_perturbed += 1;
                    }
                    None if d == T::zero() => return Err(RslabError::NumericallyRankDeficient),
                    _ => {}
                }
                d_diag[k] = d;
                // Inertia: sign of the 1×1 pivot (real part).
                let r = d.real();
                if r > 0.0 {
                    inertia.positive += 1;
                } else if r < 0.0 {
                    inertia.negative += 1;
                } else {
                    inertia.zero += 1;
                }
                let dinv = d.recip();
                // Update only the in-panel trailing columns `(k+1)..ke` (across all
                // rows, so the panel's L21 multiplier rows are formed). The columns
                // beyond `ke` are deferred to this panel's trailing GEMM.
                for j in (k + 1)..ke {
                    let wj_dinv = f[k * n + j] * dinv;
                    if wj_dinv != T::zero() {
                        for i in j..n {
                            f[j * n + i] = f[j * n + i] - f[k * n + i] * wj_dinv;
                        }
                    }
                }
                for i in (k + 1)..n {
                    f[k * n + i] = f[k * n + i] * dinv;
                }
                k += 1;
            } else {
                if kp != k + 1 {
                    swap_sym_lower(f, n, k + 1, kp);
                    perm.swap(k + 1, kp);
                }
                let mut d11 = f[k * n + k];
                let d21 = f[k * n + (k + 1)];
                let mut d22 = f[(k + 1) * n + (k + 1)];
                let mut det = d11 * d22 - d21 * d21;
                // Scale-invariant singularity / growth guard: a 2×2 whose `|det|`
                // is below `GROWTH_EPS · scale²` would inject `1/|det|` growth into
                // the trailing update. `scale` is the largest block-entry magnitude.
                let scale = d11.magnitude().max(d22.magnitude()).max(d21.magnitude());
                let growth_floor = GROWTH_EPS * scale * scale;
                // Static-pivot the 2×2 when its determinant is near-singular. The
                // real kernel (rslab's `perturb_2x2_to_floor`) shifts the small
                // eigenvalue; for complex-symmetric blocks the eigenvalues are
                // complex, so we shift both diagonals by the floor (lifting |det|)
                // and, as a last resort, nudge det itself - enough to keep the
                // preconditioner factor live.
                match perturb_floor {
                    Some(floor) => {
                        let fl = (floor * floor).max(growth_floor);
                        if det.magnitude() < fl {
                            let lift = floor.max(scale * GROWTH_EPS.sqrt());
                            d11 = d11 + T::from_real(lift);
                            d22 = d22 + T::from_real(lift);
                            det = d11 * d22 - d21 * d21;
                            if det.magnitude() < fl {
                                det = det + T::from_real(fl);
                            }
                            n_perturbed += 1;
                        }
                    }
                    None if det.magnitude() <= growth_floor => {
                        return Err(RslabError::NumericallyRankDeficient)
                    }
                    _ => {}
                }
                let detinv = det.recip();
                d_diag[k] = d11;
                d_subdiag[k] = d21;
                d_diag[k + 1] = d22;
                two_by_two[k] = true;
                // Inertia of the 2×2 block from det / trace (real parts): det<0 →
                // one +, one −; det>0 → two of sign(trace); det≈0 → one 0, one
                // sign(trace).
                let det_r = det.real();
                let tr_r = (d11 + d22).real();
                if det_r < 0.0 {
                    inertia.positive += 1;
                    inertia.negative += 1;
                } else if det_r > 0.0 {
                    if tr_r >= 0.0 {
                        inertia.positive += 2;
                    } else {
                        inertia.negative += 2;
                    }
                } else {
                    inertia.zero += 1;
                    if tr_r >= 0.0 {
                        inertia.positive += 1;
                    } else {
                        inertia.negative += 1;
                    }
                }

                for i in (k + 2)..n {
                    let wik = f[k * n + i];
                    let wik1 = f[(k + 1) * n + i];
                    l1[i] = (d22 * wik - d21 * wik1) * detinv;
                    l2[i] = (d11 * wik1 - d21 * wik) * detinv;
                }
                for j in (k + 2)..ke {
                    let l1j = l1[j];
                    let l2j = l2[j];
                    for i in j..n {
                        f[j * n + i] = f[j * n + i] - f[k * n + i] * l1j - f[(k + 1) * n + i] * l2j;
                    }
                }
                for i in (k + 2)..n {
                    f[k * n + i] = l1[i];
                    f[(k + 1) * n + i] = l2[i];
                }
                k += 2;
            }
        }

        // Deferred panel trailing update: f[ke.., ke..] −= L21·D·L21ᵀ. Build the
        // panel's L21 (trailing rows × panel cols) and G = L21·D (block-diagonal
        // D), GEMM into a temp, then subtract its lower triangle into `f`.
        let pw = ke - kb;
        let mt = n - ke;
        if mt > 0 && pw > 0 {
            l21buf.clear();
            l21buf.resize(mt * pw, T::zero());
            for (cc, c) in (kb..ke).enumerate() {
                for (rr, r) in (ke..n).enumerate() {
                    l21buf[cc * mt + rr] = f[c * n + r];
                }
            }
            gbuf.clear();
            gbuf.resize(mt * pw, T::zero());
            let mut c = kb;
            while c < ke {
                let cc = c - kb;
                if two_by_two[c] {
                    let (d11, d21, d22) = (d_diag[c], d_subdiag[c], d_diag[c + 1]);
                    for rr in 0..mt {
                        let a = l21buf[cc * mt + rr];
                        let b = l21buf[(cc + 1) * mt + rr];
                        gbuf[cc * mt + rr] = a * d11 + b * d21;
                        gbuf[(cc + 1) * mt + rr] = a * d21 + b * d22;
                    }
                    c += 2;
                } else {
                    let d = d_diag[c];
                    for rr in 0..mt {
                        gbuf[cc * mt + rr] = l21buf[cc * mt + rr] * d;
                    }
                    c += 1;
                }
            }
            tmp.clear();
            tmp.resize(mt * mt, T::zero());
            let par = if (mt as u128) * (mt as u128) * (pw as u128) >= kt.par_cdiv as u128 {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            if kt.use_gemm_schur {
                // SAFETY: `tmp`, `gbuf`, `l21buf` are three distinct,
                // non-overlapping allocations sized for (m,n,k)=(mt,mt,pw) under
                // the strides passed; `T` is `f64`/`Complex<f64>` (gemm-supported).
                unsafe {
                    gemm::gemm(
                        mt,
                        mt,
                        pw,
                        tmp.as_mut_ptr(),
                        mt as isize,
                        1,
                        false,
                        gbuf.as_ptr(),
                        mt as isize,
                        1,
                        l21buf.as_ptr(),
                        1,
                        mt as isize,
                        T::zero(),
                        T::one(),
                        false,
                        false,
                        false,
                        par,
                    );
                }
            } else {
                for jj in 0..mt {
                    for ii in jj..mt {
                        let mut acc = T::zero();
                        for cc in 0..pw {
                            acc = acc + gbuf[cc * mt + ii] * l21buf[cc * mt + jj];
                        }
                        tmp[jj * mt + ii] = acc;
                    }
                }
            }
            // Subtract the panel's trailing Schur block into `f`'s trailing lower
            // triangle. On a large front (top of the assembly tree, where tree
            // parallelism has dried up) this per-panel scatter is split across the
            // trailing columns: `ke..ke+mt` are contiguous columns of the
            // column-major front, so each rayon task owns a disjoint column and
            // reads the shared read-only `tmp` - the write set is a partition, so
            // the result is **bit-identical** regardless of worker count (the
            // determinism guarantee holds). 2D front parallelism complementing the
            // already-parallel Schur GEMM above; gated by the `par_cdiv` flop bar.
            if (mt as u128) * (mt as u128) >= kt.par_cdiv as u128 {
                let base = ke * n;
                f[base..base + mt * n]
                    .par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(jj, col)| {
                        for ii in jj..mt {
                            col[ke + ii] = col[ke + ii] - tmp[jj * mt + ii];
                        }
                    });
            } else {
                for jj in 0..mt {
                    let cj = ke + jj;
                    for ii in jj..mt {
                        let ri = ke + ii;
                        f[cj * n + ri] = f[cj * n + ri] - tmp[jj * mt + ii];
                    }
                }
            }
        }
        kb = ke;
    }

    // Extract the front's L (nrow × ncol, pivot order).
    let mut l = vec![T::zero(); nrow * ncol];
    let mut c = 0;
    while c < ncol {
        if two_by_two[c] {
            l[c * nrow + c] = one;
            l[(c + 1) * nrow + (c + 1)] = one;
            for i in (c + 2)..nrow {
                l[c * nrow + i] = f[c * nrow + i];
                l[(c + 1) * nrow + i] = f[(c + 1) * nrow + i];
            }
            c += 2;
        } else {
            l[c * nrow + c] = one;
            for i in (c + 1)..nrow {
                l[c * nrow + i] = f[c * nrow + i];
            }
            c += 1;
        }
    }

    // Contribution block CB = A22 − L21·D·L21ᵀ. The per-panel trailing GEMMs
    // above already applied the whole Schur update into `f`'s trailing
    // `[ncol, nrow)²` lower triangle, so extract it directly (mirrored to both
    // triangles for the parent's extend-add).
    let cnrow = nrow - ncol;
    let mut cb = vec![T::zero(); cnrow * cnrow];
    for j in 0..cnrow {
        for i in j..cnrow {
            let v = f[(ncol + j) * n + (ncol + i)];
            cb[j * cnrow + i] = v;
            cb[i * cnrow + j] = v;
        }
    }

    Ok((
        FrontFactors {
            nrow,
            nelim: ncol,
            perm,
            l,
            d_diag,
            d_subdiag,
            two_by_two,
            n_perturbed,
            inertia,
        },
        cb,
    ))
}

/// Reassembled per-front factor, retained for the global pass.
struct NodeFactor<T> {
    front: FrontFactors<T>,
    row_indices: Vec<usize>,
    /// This front's contribution block (`cnrow × cnrow` column-major lower
    /// triangle), consumed by the parent's extend-add. Kept on the node (rather
    /// than a separate take-able slot) so independent subtrees factor in
    /// parallel without a shared mutable contribution pool.
    contrib: Vec<T>,
}

thread_local! {
    /// Per-worker global→front-local index scratch (`usize`, scalar-independent),
    /// reused across every front a thread factors and held at the all-`usize::MAX`
    /// invariant between uses. Replaces the old `map_init` workspace now that the
    /// driver is a work-stealing tree recursion rather than a level `par_iter`.
    static GLOC_SCRATCH: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// A supernode's own factor plus the flat `(supernode-id, factor)` list for the
/// rest of its subtree - the return shape of [`factor_subtree`].
type SubtreeFactors<T> = (NodeFactor<T>, Vec<(usize, NodeFactor<T>)>);

/// Factor one supernode's front: build its row structure, assemble the original
/// (permuted) entries and the children's contribution blocks, then partially
/// factor the fully-summed columns. Reads only already-computed children, so
/// supernodes on the same assembly-tree level run concurrently.
fn factor_one_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &CscMatrix<T>,
    child_refs: &[&NodeFactor<T>],
    perturb_floor: Option<f64>,
    kt: KernelTuning,
) -> Result<NodeFactor<T>, RslabError> {
    let snode = &sym.supernodes[s];
    let n = sym.n;
    let ncol = snode.ncol;
    let own_last = snode.first_col + ncol;

    // Front row structure: own columns ++ sorted trailing rows (from the
    // permuted pattern of the own columns plus the children contribution rows).
    let mut trailing: Vec<usize> = Vec::new();
    for j in snode.first_col..own_last {
        for k in sym.permuted_pattern.col_ptr[j]..sym.permuted_pattern.col_ptr[j + 1] {
            let r = sym.permuted_pattern.row_idx[k];
            if r >= own_last {
                trailing.push(r);
            }
        }
    }
    for child in child_refs {
        for &r in &child.row_indices[child.front.nelim..] {
            if r >= own_last {
                trailing.push(r);
            }
        }
    }
    trailing.sort_unstable();
    trailing.dedup();
    let mut ri = Vec::with_capacity(ncol + trailing.len());
    ri.extend(snode.first_col..own_last);
    ri.extend(trailing);
    let nrow = ri.len();

    // Front buffer (transient - the unavoidable nrow² zeroing dominates, so a
    // per-front allocation adds only negligible malloc over a pooled one).
    let mut fbuf: Vec<T> = vec![T::zero(); nrow * nrow];
    let f = &mut fbuf[..];

    // Take the thread-local global→local scratch (held at all-`usize::MAX`) for
    // the assembly; returned before `factor_front` so the front GEMM's
    // work-stealing tasks can never re-enter the borrow.
    let mut gloc = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if gloc.len() < n {
        gloc.resize(n, usize::MAX);
    }
    for (li, &g) in ri.iter().enumerate() {
        gloc[g] = li;
    }

    // Scatter original entries of the eliminated columns.
    for p in 0..ncol {
        let c = snode.first_col + p;
        for k in a_perm.col_ptr[c]..a_perm.col_ptr[c + 1] {
            let g = a_perm.row_idx[k];
            let lr = gloc[g];
            debug_assert!(lr != usize::MAX, "original entry outside front");
            let (hi, lo) = if lr >= p { (lr, p) } else { (p, lr) };
            f[lo * nrow + hi] = f[lo * nrow + hi] + a_perm.values[k];
        }
    }

    // Extend-add each child's contribution block.
    for child in child_refs {
        let cn = child.front.nrow - child.front.nelim;
        let crows = &child.row_indices[child.front.nelim..];
        let cb = &child.contrib;
        for j in 0..cn {
            let lj = gloc[crows[j]];
            for i in j..cn {
                let li = gloc[crows[i]];
                let (hi, lo) = if li >= lj { (li, lj) } else { (lj, li) };
                f[lo * nrow + hi] = f[lo * nrow + hi] + cb[j * cn + i];
            }
        }
    }

    // Restore the all-`usize::MAX` invariant and return the scratch to the
    // thread-local before `factor_front` (which spawns work-stealing GEMM tasks).
    for &g in &ri {
        gloc[g] = usize::MAX;
    }
    GLOC_SCRATCH.with(|c| *c.borrow_mut() = gloc);

    let (front, contrib) = factor_front(f, nrow, ncol, perturb_floor, kt)?;
    Ok(NodeFactor {
        front,
        row_indices: ri,
        contrib,
    })
}

/// Recursively factor the assembly subtree rooted at supernode `s` with a
/// work-stealing tree schedule: the children's subtrees factor concurrently and
/// this node only after they finish. Independent subtrees fill idle threads and
/// the per-front GEMM shares the same rayon pool - no level-barrier stall. See
/// the unsymmetric twin in [`crate::numeric::multifrontal_lu`].
fn factor_subtree<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &CscMatrix<T>,
    perturb_floor: Option<f64>,
    kt: KernelTuning,
) -> Result<SubtreeFactors<T>, RslabError> {
    let children = &sym.supernodes[s].children;
    let mut outs: Vec<SubtreeFactors<T>> = children
        .par_iter()
        .map(|&ch| factor_subtree(ch, sym, a_perm, perturb_floor, kt))
        .collect::<Result<Vec<_>, _>>()?;
    let nf = {
        let child_refs: Vec<&NodeFactor<T>> = outs.iter().map(|(own, _)| own).collect();
        factor_one_node(s, sym, a_perm, &child_refs, perturb_floor, kt)?
    };
    // Free the children's contribution blocks NOW: they have been extend-added
    // into this front and are never read again (the global emit uses only the
    // L/D factor). Retaining the whole `Σ cnrow²` CB stack to the end was the
    // dominant transient-memory cost (OOMs on large fronts); dropping each CB as
    // its parent consumes it keeps only the active contribution frontier live.
    for (own, _) in outs.iter_mut() {
        own.contrib = Vec::new();
    }
    let mut subtree = Vec::new();
    for (i, (own, rest)) in outs.into_iter().enumerate() {
        subtree.push((children[i], own));
        subtree.extend(rest);
    }
    Ok((nf, subtree))
}

/// Factor a sparse symmetric matrix `A` as `Pᵀ A P = L D Lᵀ` via generic
/// multifrontal Bunch-Kaufman. Works for `T = f64` and `T = Complex<f64>`
/// (complex symmetric, `A = Aᵀ`).
///
/// Returns an [`LdltFactors`] in factorization order; solve with
/// [`solve_ldlt`](crate::dense::ldlt_generic::solve_ldlt).
pub fn factor_sparse_ldlt<T: Scalar>(a: &CscMatrix<T>) -> Result<LdltFactors<T>, RslabError> {
    factor_sparse_ldlt_with(a, &SolverSettings::default())
}

/// Like [`factor_sparse_ldlt`] but with explicit [`SolverSettings`] -
/// notably static-pivoting (preconditioner) mode via `on_zero_pivot`.
///
/// Convenience wrapper: runs [`analyze`] then [`factor_numeric`]. For the
/// PARDISO-style *analyze once, factor many* workflow - FEM Newton steps or a
/// frequency sweep that reuse one sparsity pattern - call them separately and
/// keep the [`MultifrontalSymbolic`] across factorizations.
pub fn factor_sparse_ldlt_with<T: Scalar>(
    a: &CscMatrix<T>,
    opts: &SolverSettings,
) -> Result<LdltFactors<T>, RslabError> {
    let symb = analyze(a.n, &a.col_ptr, &a.row_idx)?;
    factor_numeric(&symb, a, opts)
}

/// Reusable symbolic analysis (fill-reducing ordering + assembly-tree levels)
/// for a fixed sparsity pattern. Value-independent: build once with [`analyze`]
/// and pass to [`factor_numeric`] for each set of numeric values sharing the
/// pattern - the PARDISO phase-1 analysis.
pub struct MultifrontalSymbolic {
    inner: Option<SymbolicInner>,
    n: usize,
    nnz: usize,
}

struct SymbolicInner {
    sym: SymbolicFactorization,
    /// Assembly-tree levels: `by_level[l]` are the supernodes at level `l`, all
    /// mutually independent (factored concurrently by the rayon driver).
    by_level: Vec<Vec<usize>>,
}

impl MultifrontalSymbolic {
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

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)`: the number of
    /// eliminated columns and the full front height. The raw material for
    /// factorization-cost diagnostics - front-size distribution (small vs dense
    /// fronts → BLAS-2 vs BLAS-3 efficiency) and a factor-flop estimate.
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        match &self.inner {
            Some(i) => i.sym.supernodes.iter().map(|s| (s.ncol, s.nrow)).collect(),
            None => Vec::new(),
        }
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

/// PARDISO phase 1: analyze a sparsity pattern (`n`, CSC `col_ptr`/`row_idx`,
/// lower triangle). The result is value-independent and reusable across many
/// [`factor_numeric`] calls that share the pattern.
pub fn analyze(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> Result<MultifrontalSymbolic, RslabError> {
    analyze_with(n, col_ptr, row_idx, &SolverSettings::default())
}

/// [`analyze`] with explicit composable [`SolverSettings`] (child-reordering
/// strategy). Reuse the result across many `factor` calls that share the pattern.
pub fn analyze_with(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    opts: &SolverSettings,
) -> Result<MultifrontalSymbolic, RslabError> {
    // The symbolic build (ordering, elimination tree, supernode amalgamation,
    // postorder) can recurse to O(n) on pathological patterns - dense/random
    // graphs where nested dissection finds no good separators - and would overflow
    // the caller's stack. Run it in a scoped pool whose workers have a stack sized
    // to the problem (committed on demand), the same robustness mechanism the
    // factorization uses. Shallow analyses get the floor stack at negligible cost.
    in_scoped_pool(0, stack_for_depth(n), || {
        analyze_with_inner(n, col_ptr, row_idx, opts)
    })
}

fn analyze_with_inner(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    opts: &SolverSettings,
) -> Result<MultifrontalSymbolic, RslabError> {
    let nnz = row_idx.len();
    if n == 0 {
        return Ok(MultifrontalSymbolic {
            inner: None,
            n: 0,
            nnz,
        });
    }
    // Symbolic analysis on the structure only; feed a unit-valued f64 pattern.
    let pattern = CscMatrix::<f64> {
        n,
        col_ptr: col_ptr.to_vec(),
        row_idx: row_idx.to_vec(),
        values: vec![1.0; nnz],
    };
    // Disable LdltCompress: it transforms the pattern via a quotient-graph
    // compression beyond a plain permutation, so `sym.perm` would no longer be
    // consistent with the `A_perm` built in `factor_numeric`.
    // Relaxed/fill-tolerant amalgamation - a standard sparse-direct technique
    // (PARDISO/MUMPS apply it to every matrix): when fundamental supernodes are
    // narrow the Schur-update GEMMs are low-rank and memory-bound, so trade a
    // little explicit-zero fill for wider, higher-rank dense fronts. The width is
    // a sweet spot: too narrow → memory-bound BLAS-2; too wide → flops wasted on
    // explicit zeros. `≤256-wide, ≤64 extra rows/merge` measured best across the
    // EM FEM / MoM matrices for **both** the multifrontal and left-looking
    // kernels (≈ −15…−25 % factor time vs the previous 512/128). The lever is
    // workload-agnostic; it rides the general `SupernodeParams.relax` knob and is
    // gated to `n >= RELAX_MIN_N` inside `find_supernodes`.
    let snode_params = SupernodeParams {
        // `preprocess: None` is a correctness requirement, not a tuning knob:
        // LdltCompress rewrites the pattern beyond a permutation, breaking the
        // `sym.perm` ↔ `A_perm` consistency `factor_numeric` relies on. The
        // tunable amalgamation knobs (`nemin`, `relax`) ride the composable
        // `SolverSettings`; everything else stays at the tuned default.
        preprocess: crate::symbolic::supernode::OrderingPreprocess::None,
        nemin: opts.nemin,
        relax: opts.relax,
        ..SupernodeParams::default()
    };
    let mut sym = symbolic_factorize_with_method(&pattern, &snode_params, opts.ordering)?;

    // Liu (1986) contribution-stack minimization. Reorder each supernode's
    // children so the live contribution-block stack peak is minimized during
    // factorization. This is a pure **scheduling hint**: supernode IDs, the
    // e-numbering and the factor are unchanged (the global emit walks IDs, not
    // children, and trailing rows are sorted), so it is correctness-, fill- and
    // throughput-neutral - it only shrinks the transient CB-stack that drives
    // factorization peak RSS.
    //
    // Each node leaves a contribution block of size `cb = (nrow−ncol)²` for its
    // parent and needs `peak` working-stack to factor its subtree. Processing
    // children in order, the stack while doing child `i` is `Σ_{j<i} cb_j +
    // peak_i`; Liu's theorem minimizes `maxᵢ(Σ_{j<i} cb_j + peak_i)` by ordering
    // children by `(peak − cb)` descending. Supernodes are in postorder, so a
    // single forward sweep has every child's `(peak, cb)` ready.
    //
    // **Hybrid Liu**: reordering is only applied where the contribution stack is
    // actually large (`Σ children cb ≥ LIU_MIN_STACK`) - the upper/mid tree,
    // which is a handful of nodes carrying the spike. The vast majority of small
    // leaf nodes keep their natural order, whose rayon spawn pattern parallelizes
    // better. This keeps almost all of Liu's memory win while shedding most of
    // its throughput cost (the memory-optimal child order is not the
    // parallel-load-optimal one). `peak[s]` is always computed against the order
    // actually used, so the propagation stays exact.
    let nsuper = sym.supernodes.len();
    if opts.reorder == ReorderMode::HybridLiu {
        // ~64 MB of `Complex<f64>` contribution blocks: below this the reorder
        // saves little memory but can still disturb leaf parallelism.
        const LIU_MIN_STACK: f64 = 4_000_000.0;
        let mut cb = vec![0.0f64; nsuper];
        let mut peak = vec![0.0f64; nsuper];
        for s in 0..nsuper {
            let cn = (sym.supernodes[s].nrow - sym.supernodes[s].ncol) as f64;
            cb[s] = cn * cn;
            let mut kids = std::mem::take(&mut sym.supernodes[s].children);
            let stack_total: f64 = kids.iter().map(|&c| cb[c]).sum();
            if stack_total >= LIU_MIN_STACK {
                kids.sort_by(|&a, &b| {
                    (peak[b] - cb[b])
                        .partial_cmp(&(peak[a] - cb[a]))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            let mut acc = 0.0f64; // Σ cb of already-processed children
            let mut pk = 0.0f64;
            for &ch in &kids {
                pk = pk.max(acc + peak[ch]);
                acc += cb[ch];
            }
            // Assembly step: all children CBs live at once (acc), then this
            // node's own CB remains.
            peak[s] = pk.max(acc).max(cb[s]);
            sym.supernodes[s].children = kids;
        }
    }

    // Assembly-tree levels: level(s) = 1 + max(level(children)); same-level
    // supernodes are mutually independent.
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

    Ok(MultifrontalSymbolic {
        inner: Some(SymbolicInner { sym, by_level }),
        n,
        nnz,
    })
}

/// Postorder (children before parent) over the assembly forest, used by the
/// right-looking sequential schedule so every child is factored before its parent
/// consumes its contribution. Iterative to avoid recursion depth on deep trees.
fn supernode_postorder(sym: &SymbolicFactorization, roots: &[usize]) -> Vec<usize> {
    let mut order = Vec::with_capacity(sym.supernodes.len());
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for &r in roots {
        stack.push((r, 0));
        // Pop, then re-push with an advanced child cursor (avoids an in-place
        // mutable peek): a node is emitted only once all its children are done.
        while let Some((node, ci)) = stack.pop() {
            let children = &sym.supernodes[node].children;
            if ci < children.len() {
                stack.push((node, ci + 1));
                stack.push((children[ci], 0));
            } else {
                order.push(node);
            }
        }
    }
    order
}

/// PARDISO phases 2-3: numeric factorization reusing a [`MultifrontalSymbolic`].
/// `a` must carry the same sparsity pattern (`n`, `nnz`) the analysis was built
/// from. Honours static pivoting and incomplete-factor dropping via `opts`.
/// Realize a [`Threads::Auto`] policy from a symbolic analysis: compute the three
/// predictive features (factor-flops, max front height, max tree width) and apply
/// the [`recommend_threads_from`](crate::analysis::recommend_threads_from) policy,
/// capped at `max_cores`. Value-independent, so it is the same for every scalar.
pub(crate) fn recommend_threads_for_sym(symb: &MultifrontalSymbolic, max_cores: usize) -> usize {
    let fd = symb.front_dims();
    let flops: u64 = fd
        .iter()
        .map(|&(nc, nr)| (nr as u64) * (nr as u64) * (nc as u64))
        .sum();
    let front_nrow_max = fd.iter().map(|&(_, nr)| nr).max().unwrap_or(0);
    let tree_width_max = symb.level_widths().into_iter().max().unwrap_or(0);
    crate::analysis::recommend_threads_from(flops, front_nrow_max, tree_width_max, max_cores)
}

pub fn factor_numeric<T: Scalar>(
    symb: &MultifrontalSymbolic,
    a: &CscMatrix<T>,
    opts: &SolverSettings,
) -> Result<LdltFactors<T>, RslabError> {
    a.validate()?;
    let n = symb.n;
    if a.n != n || a.row_idx.len() != symb.nnz {
        return Err(RslabError::InvalidInput(
            "factor_numeric: matrix does not match the analyzed pattern".to_string(),
        ));
    }
    let inner = match &symb.inner {
        None => {
            return Ok(LdltFactors {
                n: 0,
                l_col_ptr: vec![0],
                l_row_idx: Vec::new(),
                l_values: Vec::new(),
                d_diag: Vec::new(),
                d_subdiag: Vec::new(),
                two_by_two: Vec::new(),
                perm: Vec::new(),
                n_perturbed: 0,
                inertia: Inertia::new(0, 0, 0),
            });
        }
        Some(i) => i,
    };
    let sym = &inner.sym;
    // Worker stack sized to the assembly-tree depth so the recursive tree
    // factorization never overflows on deep chain trees (banded / 1D + low nemin).
    let stack = stack_for_depth(supernode_tree_depth(sym));

    // Supernodal left-looking path: same factor, low transient (no CB stack). Run
    // in a scoped pool of `opts.threads` so concurrent solves don't oversubscribe.
    if opts.method == FactorMethod::LeftLooking {
        return opts.threads.run(
            stack,
            |cap| recommend_threads_for_sym(symb, cap),
            || factor_left_looking(sym, a, opts),
        );
    }

    // Static-pivot floor (absolute), translated from rslab's ZeroPivotAction.
    // `PerturbToEps { abs_floor }` is taken as given (rslab convention: an
    // absolute floor, typically `eps_rel · ‖A‖∞`); `Fail` disables perturbation.
    let perturb_floor: Option<f64> = match opts.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = a.values.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    };

    // 2. Permuted matrix A_perm = Pᵀ A P in permuted (new) numbering, lower
    //    triangle. `perm_inv` is old→new.
    let nnz = a.row_idx.len();
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let gi = sym.perm_inv[i];
            let gj = sym.perm_inv[j];
            let (r, c) = if gi >= gj { (gi, gj) } else { (gj, gi) };
            rows.push(r);
            cols.push(c);
            vals.push(a.values[k]);
        }
    }
    let a_perm = CscMatrix::<T>::from_triplets(n, &rows, &cols, &vals)?;

    // 3. Multifrontal numeric factorization with a work-stealing schedule over
    //    the assembly tree: each subtree factors independently (children before
    //    parent), filling idle threads without a level barrier, and the per-front
    //    GEMM shares the same rayon pool. The precomputed `by_level` is no longer
    //    consulted here (it remains available via `MultifrontalSymbolic::n_levels`).
    let nsuper = sym.supernodes.len();

    // Roots of the assembly forest: supernodes that are no node's child.
    let mut is_child = vec![false; nsuper];
    for snode in &sym.supernodes {
        for &ch in &snode.children {
            is_child[ch] = true;
        }
    }
    let roots: Vec<usize> = (0..nsuper).filter(|&s| !is_child[s]).collect();
    let kt = opts.kernel();
    // Run the work-stealing tree recursion in a scoped pool of `opts.threads` with
    // the depth-sized stack (honours the thread budget and is overflow-safe on
    // deep trees, like the left-looking path above).
    let recommend = |cap: usize| recommend_threads_for_sym(symb, cap);
    let mut node_results: Vec<Option<NodeFactor<T>>> = (0..nsuper).map(|_| None).collect();
    if opts.method == FactorMethod::RightLooking {
        // Right-looking schedule: factor supernodes sequentially in postorder,
        // holding every front live (no contribution-block frontier free), each
        // pushed into its parent by the shared `factor_one_node` assembly. Same
        // factor as the other paths; the trade is all-fronts residency for a
        // simple, barrier-free push (opt-in, not auto-selected).
        let order = supernode_postorder(sym, &roots);
        opts.threads
            .run(stack, recommend, || -> Result<(), RslabError> {
                for &s in &order {
                    let nf = {
                        let mut child_refs: Vec<&NodeFactor<T>> = Vec::new();
                        for &ch in &sym.supernodes[s].children {
                            match node_results[ch].as_ref() {
                                Some(c) => child_refs.push(c),
                                None => {
                                    return Err(RslabError::InvalidInput(
                                        "internal: child not factored before parent".to_string(),
                                    ))
                                }
                            }
                        }
                        factor_one_node(s, sym, &a_perm, &child_refs, perturb_floor, kt)?
                    };
                    node_results[s] = Some(nf);
                }
                Ok(())
            })?;
    } else {
        let root_outs: Vec<SubtreeFactors<T>> = opts.threads.run(stack, recommend, || {
            roots
                .par_iter()
                .map(|&r| factor_subtree(r, sym, &a_perm, perturb_floor, kt))
                .collect::<Result<Vec<_>, _>>()
        })?;
        // Scatter the subtree factors into `node_results` (by supernode id) for the
        // global emit pass, which still walks supernodes in postorder.
        for (i, (own, subtree)) in root_outs.into_iter().enumerate() {
            node_results[roots[i]] = Some(own);
            for (s, nf) in subtree {
                node_results[s] = Some(nf);
            }
        }
    }

    // Collect the factored nodes in supernode (= elimination) order.
    let mut nodes: Vec<&NodeFactor<T>> = Vec::with_capacity(nsuper);
    for node_opt in &node_results {
        match node_opt {
            Some(nd) => nodes.push(nd),
            None => {
                return Err(RslabError::InvalidInput(
                    "internal: unfactored supernode".to_string(),
                ))
            }
        }
    }

    // 4a. Assign factorization order e and gather D in e-order.
    let mut e_of_g = vec![usize::MAX; n];
    let mut perm = vec![0usize; n];
    let mut d_diag = vec![T::zero(); n];
    let mut d_subdiag = vec![T::zero(); n];
    let mut two_by_two = vec![false; n];
    let mut e = 0usize;
    for node in &nodes {
        let ff = &node.front;
        for j in 0..ff.nelim {
            let g = node.row_indices[ff.perm[j]];
            e_of_g[g] = e;
            perm[e] = sym.perm[g];
            d_diag[e] = ff.d_diag[j];
            d_subdiag[e] = ff.d_subdiag[j];
            two_by_two[e] = ff.two_by_two[j];
            e += 1;
        }
    }
    debug_assert_eq!(e, n, "every index eliminated exactly once");

    // Aggregate the additive per-front scalars before the emit, which under
    // `LowMemory` frees each front's dense factor as it is consumed (so `nodes`,
    // the immutable view, must be released first). `n_perturbed` and the inertia
    // read only the small `front` scalars, not the dense `front.l`.
    let n_perturbed: usize = nodes.iter().map(|nd| nd.front.n_perturbed).sum();
    // Inertia is additive over the assembly tree: sum the per-front signatures.
    let mut inertia = Inertia::new(0, 0, 0);
    for nd in &nodes {
        inertia.positive += nd.front.inertia.positive;
        inertia.negative += nd.front.inertia.negative;
        inertia.zero += nd.front.inertia.zero;
    }
    drop(nodes);
    // `LowMemory` (default): free each front's dense `L` the moment it is emitted
    // into the global CSC, shrinking the per-front transient as the global factor
    // grows (parity with the multifrontal LU emit). `Eager` keeps every front's
    // dense factor until the end (a throughput A/B knob; bit-identical factor).
    let low_mem = opts.memory == MemoryMode::LowMemory;

    // 4b. Emit each front's L columns into the global CSC L, in e-order. A
    //     supernode's eliminated columns form a contiguous increasing e-range,
    //     so iterating nodes then `j` yields columns in ascending CSC order;
    //     rows within a column are sorted. Iterated mutably so `LowMemory` can
    //     drop `front.l` per front (every id is `Some`, validated above).
    let one = T::one();
    let mut l_col_ptr = Vec::with_capacity(n + 1);
    l_col_ptr.push(0);
    let mut l_row_idx: Vec<usize> = Vec::new();
    let mut l_values: Vec<T> = Vec::new();
    let mut col: Vec<(usize, T)> = Vec::new();
    for node_opt in node_results.iter_mut() {
        let node = node_opt.as_mut().ok_or_else(|| {
            RslabError::InvalidInput("internal: unfactored supernode".to_string())
        })?;
        let ff = &node.front;
        let nrow = ff.nrow;
        for j in 0..ff.nelim {
            col.clear();
            let diag_e = e_of_g[node.row_indices[ff.perm[j]]];
            col.push((diag_e, one));
            for i in (j + 1)..nrow {
                let v = ff.l[j * nrow + i];
                if v != T::zero() {
                    let row_e = e_of_g[node.row_indices[ff.perm[i]]];
                    col.push((row_e, v));
                }
            }
            // Incomplete factorization: drop sub-threshold fill (relative to the
            // column's largest multiplier), keeping the unit diagonal. Shrinks
            // nnz(L) and the apply cost - an approximate factor for use as a
            // preconditioner. `None` keeps the factor complete.
            if let Some(tau) = opts.drop_tol {
                let colmax = col
                    .iter()
                    .filter(|&&(r, _)| r != diag_e)
                    .map(|&(_, v)| v.magnitude())
                    .fold(0.0, f64::max);
                let thresh = tau * colmax;
                col.retain(|&(r, v)| r == diag_e || v.magnitude() >= thresh);
            }
            col.sort_unstable_by_key(|&(r, _)| r);
            for &(r, v) in &col {
                l_row_idx.push(r);
                l_values.push(v);
            }
            l_col_ptr.push(l_row_idx.len());
        }
        if low_mem {
            node.front.l = Vec::new();
        }
    }

    Ok(LdltFactors {
        n,
        l_col_ptr,
        l_row_idx,
        l_values,
        d_diag,
        d_subdiag,
        two_by_two,
        perm,
        n_perturbed,
        inertia,
    })
}

/// Filled `L` row structure of every supernode (bottom-up, children before
/// parents), mirroring the multifrontal assembly value-free: a supernode's
/// structure is its own columns ++ the sorted union of its column patterns'
/// trailing rows and its children's off-diagonal rows. `rs[s][0..ncol]` are the
/// eliminated columns `first_col..first_col+ncol`; `rs[s][ncol..]` are the
/// (sorted) below-diagonal fill rows.
pub(crate) fn compute_supernode_row_structures(sym: &SymbolicFactorization) -> Vec<Vec<usize>> {
    let nsuper = sym.supernodes.len();
    let mut rs: Vec<Vec<usize>> = Vec::with_capacity(nsuper);
    for s in 0..nsuper {
        let snode = &sym.supernodes[s];
        let own_last = snode.first_col + snode.ncol;
        let mut trailing: Vec<usize> = Vec::new();
        for j in snode.first_col..own_last {
            for k in sym.permuted_pattern.col_ptr[j]..sym.permuted_pattern.col_ptr[j + 1] {
                let r = sym.permuted_pattern.row_idx[k];
                if r >= own_last {
                    trailing.push(r);
                }
            }
        }
        for &ch in &snode.children {
            let nck = sym.supernodes[ch].ncol;
            for &r in &rs[ch][nck..] {
                if r >= own_last {
                    trailing.push(r);
                }
            }
        }
        trailing.sort_unstable();
        trailing.dedup();
        let mut ri = Vec::with_capacity(snode.ncol + trailing.len());
        ri.extend(snode.first_col..own_last);
        ri.extend(trailing);
        rs.push(ri);
    }
    rs
}

/// Concurrently-filled store of the left-looking factor panels. Each cell is
/// written exactly once - by its owning supernode's factorization, which
/// completes before any ancestor (its only reader) runs, per the subtree
/// recursion - and concurrent writers touch disjoint indices, so the unsynchronized
/// interior mutability is sound.
struct LlStore<T> {
    panels: Vec<std::cell::UnsafeCell<Vec<T>>>,
    dvals: Vec<std::cell::UnsafeCell<Vec<T>>>,
    /// Sub-diagonal D entry of each 2×2 Bunch-Kaufman block (per eliminated
    /// column, in the panel's pivoted order; zero on 1×1 columns and on the
    /// second column of a 2×2 block).
    dsubs: Vec<std::cell::UnsafeCell<Vec<T>>>,
    /// `true` at the first column of each 2×2 block (pivoted order).
    twos: Vec<std::cell::UnsafeCell<Vec<bool>>>,
    /// Local within-panel pivot permutation (length = panel `nrow`, identity on
    /// the off-diagonal rows `[ncol, nrow)` since pivoting is bounded to the
    /// fully-summed block). Pivoted index `i` ↔ original local index `lperm[i]`.
    lperms: Vec<std::cell::UnsafeCell<Vec<usize>>>,
}
// SAFETY: see the type doc - single-writer-before-readers, disjoint indices.
unsafe impl<T: Send> Sync for LlStore<T> {}

impl<T: Scalar> LlStore<T> {
    fn new(nsuper: usize) -> Self {
        LlStore {
            panels: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(Vec::new()))
                .collect(),
            dvals: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(Vec::new()))
                .collect(),
            dsubs: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(Vec::new()))
                .collect(),
            twos: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(Vec::new()))
                .collect(),
            lperms: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(Vec::new()))
                .collect(),
        }
    }
    /// SAFETY: `k` must be a fully-factored descendant of the current node.
    unsafe fn panel(&self, k: usize) -> &Vec<T> {
        &*self.panels[k].get()
    }
    /// SAFETY: as [`panel`](Self::panel).
    unsafe fn dval(&self, k: usize) -> &Vec<T> {
        &*self.dvals[k].get()
    }
    /// SAFETY: as [`panel`](Self::panel).
    unsafe fn dsub(&self, k: usize) -> &Vec<T> {
        &*self.dsubs[k].get()
    }
    /// SAFETY: as [`panel`](Self::panel).
    unsafe fn two(&self, k: usize) -> &Vec<bool> {
        &*self.twos[k].get()
    }
    /// SAFETY: as [`panel`](Self::panel).
    unsafe fn lperm(&self, k: usize) -> &Vec<usize> {
        &*self.lperms[k].get()
    }
    /// SAFETY: only the owner of supernode `s` calls this, exactly once.
    unsafe fn set(
        &self,
        s: usize,
        panel: Vec<T>,
        d: Vec<T>,
        dsub: Vec<T>,
        two: Vec<bool>,
        lperm: Vec<usize>,
    ) {
        *self.panels[s].get() = panel;
        *self.dvals[s].get() = d;
        *self.dsubs[s].get() = dsub;
        *self.twos[s].get() = two;
        *self.lperms[s].get() = lperm;
    }
    /// Release `k`'s dense panel + D/lperm once it has been compacted.
    /// SAFETY: `k`'s last consumer is done - no other thread reads its cells.
    unsafe fn free(&self, k: usize) {
        *self.panels[k].get() = Vec::new();
        *self.dvals[k].get() = Vec::new();
        *self.dsubs[k].get() = Vec::new();
        *self.twos[k].get() = Vec::new();
        *self.lperms[k].get() = Vec::new();
    }
}

/// Compact (CSC-fragment) form of one supernode's L factor, produced the moment
/// its last consumer pulls from it so the dense panel can be freed during
/// factorization. Row indices are already final elimination positions.
struct CompactL<T> {
    ptr: Vec<usize>,
    idx: Vec<usize>,
    val: Vec<T>,
}
impl<T> Default for CompactL<T> {
    fn default() -> Self {
        CompactL {
            ptr: Vec::new(),
            idx: Vec::new(),
            val: Vec::new(),
        }
    }
}

/// Incremental-emit state for the LDLᵀ left-looking path: consumer refcounts (free
/// the panel at 0), the compact L sink, and the O(n) maps populated in-node
/// (block-aware for Bunch-Kaufman 1×1/2×2 D) and read after the barrier.
struct LlEmitLdlt<T> {
    refcount: Vec<AtomicUsize>,
    e_offset: Vec<usize>,
    compact: Vec<std::cell::UnsafeCell<CompactL<T>>>,
    e_of_g: Vec<std::cell::UnsafeCell<usize>>,
    perm: Vec<std::cell::UnsafeCell<usize>>,
    d_diag: Vec<std::cell::UnsafeCell<T>>,
    d_subdiag: Vec<std::cell::UnsafeCell<T>>,
    two_by_two: Vec<std::cell::UnsafeCell<bool>>,
    // Inertia accumulated across supernodes (block-aware).
    inertia_pos: AtomicUsize,
    inertia_neg: AtomicUsize,
    inertia_zero: AtomicUsize,
}
// SAFETY: disjoint-index writes; visibility via the refcount AcqRel + subtree join.
unsafe impl<T: Send> Sync for LlEmitLdlt<T> {}

impl<T: Scalar> LlEmitLdlt<T> {
    fn new(sym: &SymbolicFactorization, update_list: &[Vec<usize>]) -> Self {
        let nsuper = sym.supernodes.len();
        let n = sym.n;
        let mut refcount: Vec<AtomicUsize> = (0..nsuper).map(|_| AtomicUsize::new(0)).collect();
        for ul in update_list {
            for &k in ul {
                *refcount[k].get_mut() += 1;
            }
        }
        let mut e_offset = vec![0usize; nsuper];
        let mut acc = 0usize;
        for (s, snode) in sym.supernodes.iter().enumerate() {
            e_offset[s] = acc;
            acc += snode.ncol;
        }
        LlEmitLdlt {
            refcount,
            e_offset,
            compact: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(CompactL::default()))
                .collect(),
            e_of_g: (0..n)
                .map(|_| std::cell::UnsafeCell::new(usize::MAX))
                .collect(),
            perm: (0..n).map(|_| std::cell::UnsafeCell::new(0)).collect(),
            d_diag: (0..n)
                .map(|_| std::cell::UnsafeCell::new(T::zero()))
                .collect(),
            d_subdiag: (0..n)
                .map(|_| std::cell::UnsafeCell::new(T::zero()))
                .collect(),
            two_by_two: (0..n).map(|_| std::cell::UnsafeCell::new(false)).collect(),
            inertia_pos: AtomicUsize::new(0),
            inertia_neg: AtomicUsize::new(0),
            inertia_zero: AtomicUsize::new(0),
        }
    }
    #[inline]
    unsafe fn eg(&self, g: usize) -> usize {
        *self.e_of_g[g].get()
    }
}

/// Compact supernode `k`'s L factor and free its dense panel + D/lperm. Called the
/// instant `k`'s last consumer pulled from it. Mirrors the per-supernode body of
/// the legacy L emit (unit diagonal, skip the 2×2 `d21` coupling row).
fn ldlt_emit_and_free<T: Scalar>(
    k: usize,
    store: &LlStore<T>,
    emit: &LlEmitLdlt<T>,
    sym: &SymbolicFactorization,
    rs: &[Vec<usize>],
    drop_tol: Option<f64>,
) {
    let ncol = sym.supernodes[k].ncol;
    let nrow = rs[k].len();
    // SAFETY: `k` is fully factored and its last consumer is done - exclusive.
    let panel = unsafe { store.panel(k) };
    let lperm = unsafe { store.lperm(k) };
    let t2 = unsafe { store.two(k) };
    let one = T::one();
    let mut cl = CompactL::<T>::default();
    cl.ptr.reserve(ncol + 1);
    cl.ptr.push(0);
    let mut col: Vec<(usize, T)> = Vec::with_capacity(nrow);
    for p in 0..ncol {
        col.clear();
        let diag_e = unsafe { emit.eg(rs[k][lperm[p]]) };
        col.push((diag_e, one));
        // Skip the 2×2 block's `d21` coupling row (D, not an L multiplier).
        let i0 = if t2[p] { p + 2 } else { p + 1 };
        for i in i0..nrow {
            let v = panel[i + p * nrow];
            if v != T::zero() {
                col.push((unsafe { emit.eg(rs[k][lperm[i]]) }, v));
            }
        }
        if let Some(tau) = drop_tol {
            let colmax = col
                .iter()
                .filter(|&&(r, _)| r != diag_e)
                .map(|&(_, v)| v.magnitude())
                .fold(0.0, f64::max);
            let thresh = tau * colmax;
            col.retain(|&(r, v)| r == diag_e || v.magnitude() >= thresh);
        }
        col.sort_unstable_by_key(|&(r, _)| r);
        for &(r, v) in &col {
            cl.idx.push(r);
            cl.val.push(v);
        }
        cl.ptr.push(cl.idx.len());
    }
    // SAFETY: exactly one thread emits `k`; `compact[k]` is written once.
    unsafe { *emit.compact[k].get() = cl };
    // SAFETY: last consumer done - no other thread reads `k`'s cells.
    if !ldlt_no_free() {
        unsafe { store.free(k) };
    }
}

// A/B toggle (`RLA_NO_FREE=1`): keep dense panels resident, to isolate the
// live-memory effect of incremental freeing.
static LDLT_NO_FREE_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[inline]
fn ldlt_no_free() -> bool {
    *LDLT_NO_FREE_FLAG.get_or_init(|| {
        std::env::var("RLA_NO_FREE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Factor one supernode's panel: assemble `A`, apply every descendant's `cmod`
/// update (BLAS-3 with scalar fallback), then `cdiv` (partial 1×1 LDLᵀ). Reads
/// only already-factored descendant panels from `store`, so sibling subtrees run
/// concurrently. Writes the factored panel + diagonal into `store`.
#[allow(clippy::too_many_arguments)]
fn ll_factor_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &CscMatrix<T>,
    rs: &[Vec<usize>],
    update_list: &[Vec<usize>],
    store: &LlStore<T>,
    emit: &LlEmitLdlt<T>,
    perturb_floor: Option<f64>,
    n_perturbed: &AtomicUsize,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    let ll_gemm_gate = kt.scalar_gate;
    let ll_gemm_par = kt.par_gemm;
    let snode = &sym.supernodes[s];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let nrow = rs[s].len();
    let n = sym.n;
    let mut panel = vec![T::zero(); nrow * ncol];

    // Thread-local global→local scratch (held at all-`usize::MAX`).
    let mut gloc = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if gloc.len() < n {
        gloc.resize(n, usize::MAX);
    }
    for (li, &g) in rs[s].iter().enumerate() {
        gloc[g] = li;
    }
    // Assemble A's lower-triangle columns of this supernode.
    for p in 0..ncol {
        let c = first + p;
        for k in a_perm.col_ptr[c]..a_perm.col_ptr[c + 1] {
            let li = gloc[a_perm.row_idx[k]];
            panel[li + p * nrow] = panel[li + p * nrow] + a_perm.values[k];
        }
    }
    // cmod from every updater (all are factored descendants).
    let mut vc: Vec<T> = Vec::new();
    let mut vd_buf: Vec<T> = Vec::new();
    let mut u_buf: Vec<T> = Vec::new();
    for &kk in &update_list[s] {
        let nck = sym.supernodes[kk].ncol;
        let nrk = rs[kk].len();
        let ok = &rs[kk][nck..];
        let nok = ok.len();
        // SAFETY: `kk` is a factored descendant of `s` (its update reaches `s`),
        // so its panel/dval cells are written and never mutated again.
        let pk = unsafe { store.panel(kk) };
        let dk = unsafe { store.dval(kk) };
        // Bunch-Kaufman block structure of `kk`'s D (pivoted column order). The
        // cmod `L·D·Lᵀ` is invariant under `kk`'s internal column permutation, so
        // only the block-diagonal `D`-apply has to honor the 2×2 blocks.
        let dsub_k = unsafe { store.dsub(kk) };
        let two_k = unsafe { store.two(kk) };
        let p0 = ok.partition_point(|&g| g < first);
        let p1 = ok.partition_point(|&g| g < first + ncol);
        let npk = p1 - p0;
        if npk == 0 {
            continue;
        }
        if nok * npk * nck < ll_gemm_gate {
            vc.clear();
            vc.resize(nck, T::zero());
            for c_idx in p0..p1 {
                let tcol = ok[c_idx] - first;
                // vc = D · (column `c_idx` of kk's off-diagonal block), with D
                // block-diagonal (1×1 and complex-symmetric 2×2 blocks).
                let mut ck = 0;
                while ck < nck {
                    let a = pk[(nck + c_idx) + ck * nrk];
                    if two_k[ck] {
                        let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                        let b = pk[(nck + c_idx) + (ck + 1) * nrk];
                        vc[ck] = d11 * a + d21 * b;
                        vc[ck + 1] = d21 * a + d22 * b;
                        ck += 2;
                    } else {
                        vc[ck] = dk[ck] * a;
                        ck += 1;
                    }
                }
                for r_idx in c_idx..nok {
                    let trow = gloc[ok[r_idx]];
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc + pk[(nck + r_idx) + ck * nrk] * vc[ck];
                    }
                    panel[trow + tcol * nrow] = panel[trow + tcol * nrow] - acc;
                }
            }
        } else {
            vd_buf.clear();
            vd_buf.resize(npk * nck, T::zero());
            // G = (kk's in-panel off-diagonal block) · D, stored column-major as
            // `vd_buf[c + ck*npk]`. D is block-diagonal (1×1 and 2×2 blocks); a
            // 2×2 block mixes its two columns. GEMM below is unchanged.
            let mut ck = 0;
            while ck < nck {
                if two_k[ck] {
                    let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                    for i in 0..npk {
                        let a = pk[(nck + p0 + i) + ck * nrk];
                        let b = pk[(nck + p0 + i) + (ck + 1) * nrk];
                        vd_buf[i + ck * npk] = d11 * a + d21 * b;
                        vd_buf[i + (ck + 1) * npk] = d21 * a + d22 * b;
                    }
                    ck += 2;
                } else {
                    let dkc = dk[ck];
                    for i in 0..npk {
                        vd_buf[i + ck * npk] = pk[(nck + p0 + i) + ck * nrk] * dkc;
                    }
                    ck += 1;
                }
            }
            u_buf.clear();
            u_buf.resize(nok * npk, T::zero());
            let par = if nok * npk * nck >= ll_gemm_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            // SAFETY: lhs (`pk` off-diag block, read), rhs (`vd_buf`, read), dst
            // (`u_buf`, write) are pairwise-disjoint; strides in bounds.
            unsafe {
                gemm::gemm(
                    nok,
                    npk,
                    nck,
                    u_buf.as_mut_ptr(),
                    nok as isize,
                    1,
                    false,
                    pk.as_ptr().add(nck),
                    nrk as isize,
                    1,
                    vd_buf.as_ptr(),
                    1,
                    npk as isize,
                    T::zero(),
                    T::one(),
                    false,
                    false,
                    false,
                    par,
                );
            }
            for c in 0..npk {
                let tcol = ok[p0 + c] - first;
                let ucol = &u_buf[c * nok..c * nok + nok];
                for r in (p0 + c)..nok {
                    let dst = gloc[ok[r]] + tcol * nrow;
                    panel[dst] = panel[dst] - ucol[r];
                }
            }
        }
    }
    // cdiv: partial **blocked** Bunch-Kaufman LDLᵀ (1×1 and 2×2 pivots), the
    // rectangular `nrow × ncol` analogue of `factor_front`'s panel kernel. The
    // fully-summed columns are factored in panels of width `NB` with pivoting
    // **bounded to the panel** (candidate rows `(k+1)..ke`), then each panel's
    // trailing update - the remaining panel columns `[ke, ncol)` over all rows
    // `[ke, nrow)` - is deferred to one SIMD GEMM (the BLAS-3 bulk, replacing the
    // scalar rank-1/rank-2 sweeps that dominated wide separators). Unlike
    // `factor_front` there is **no `A22` block** (the panel has no columns beyond
    // `ncol`; that Schur update is the ancestors' `cmod`), so the trailing region
    // is the rectangular `(nrow-ke) × (ncol-ke)` lower part. Pivoting stays inside
    // `0..ncol`, so the off-diagonal rows `[ncol, nrow)` keep their identity and
    // `s`'s contribution to ancestors is unaffected by this internal permutation.
    let nb = kt.panel_nb;
    let ll_cdiv_par = kt.par_cdiv;
    let alpha = bk_alpha();
    let mut d = vec![T::zero(); ncol];
    let mut d_subdiag = vec![T::zero(); ncol];
    let mut two_by_two = vec![false; ncol];
    let mut lperm: Vec<usize> = (0..nrow).collect();
    // 2×2 multiplier scratch (reused; only `[k+2, nrow)` is ever read each step).
    let mut l1 = vec![T::zero(); nrow];
    let mut l2 = vec![T::zero(); nrow];
    // Per-panel deferred-GEMM scratch (reused across panels).
    let mut l21buf: Vec<T> = Vec::new();
    let mut gbuf: Vec<T> = Vec::new();
    let mut tmp: Vec<T> = Vec::new();
    let mut local_perturbed = 0usize;
    // Helper to restore the `gloc` scratch invariant before an early return.
    macro_rules! restore_gloc {
        () => {{
            for &g in &rs[s] {
                gloc[g] = usize::MAX;
            }
            GLOC_SCRATCH.with(|c| *c.borrow_mut() = gloc);
        }};
    }
    let prof = ldlt_prof_on();
    let mut kb = 0;
    while kb < ncol {
        let ke = (kb + nb).min(ncol);
        let t_g = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // getf2: unblocked Bunch-Kaufman over the panel columns [kb, ke), with the
        // pivot candidate and rank-1/rank-2 trailing updates bounded to `ke` (the
        // columns beyond `ke` are deferred to the panel GEMM below). The L21
        // multipliers are still formed over all rows down to `nrow`.
        let mut k = kb;
        while k < ke {
            let absakk = panel[k + k * nrow].magnitude();
            // colmax over the in-panel candidate rows (k+1)..ke.
            let mut colmax_sq = 0.0;
            let mut imax = k;
            for i in (k + 1)..ke {
                let m = panel[k * nrow + i].magnitude_sq();
                if m > colmax_sq {
                    colmax_sq = m;
                    imax = i;
                }
            }
            let colmax = colmax_sq.sqrt();

            let kstep;
            let kp;
            if absakk.max(colmax) == 0.0 {
                if perturb_floor.is_none() {
                    restore_gloc!();
                    return Err(RslabError::NumericallyRankDeficient);
                }
                kstep = 1;
                kp = k;
            } else if absakk >= alpha * colmax {
                kstep = 1;
                kp = k;
            } else {
                // rowmax in row `imax`, restricted to the panel.
                let mut rowmax_sq = 0.0;
                for j in k..imax {
                    let m = panel[j * nrow + imax].magnitude_sq();
                    if m > rowmax_sq {
                        rowmax_sq = m;
                    }
                }
                for i in (imax + 1)..ke {
                    let m = panel[imax * nrow + i].magnitude_sq();
                    if m > rowmax_sq {
                        rowmax_sq = m;
                    }
                }
                let rowmax = rowmax_sq.sqrt();
                if absakk >= alpha * colmax * (colmax / rowmax) {
                    kstep = 1;
                    kp = k;
                } else if panel[imax * nrow + imax].magnitude() >= alpha * rowmax {
                    kstep = 1;
                    kp = imax;
                } else {
                    kstep = 2;
                    kp = imax;
                }
            }

            if kstep == 1 {
                if kp != k {
                    swap_sym_lower(&mut panel, nrow, k, kp);
                    lperm.swap(k, kp);
                }
                let mut dk = panel[k + k * nrow];
                match perturb_floor {
                    Some(floor) if dk.magnitude() < floor => {
                        dk = perturb_pivot(dk, floor);
                        panel[k + k * nrow] = dk;
                        local_perturbed += 1;
                    }
                    None if dk == T::zero() => {
                        restore_gloc!();
                        return Err(RslabError::NumericallyRankDeficient);
                    }
                    _ => {}
                }
                d[k] = dk;
                let dinv = dk.recip();
                // Update the in-panel trailing columns (k+1)..ke (all rows, so the
                // L21 multiplier rows form), then scale column k → its L column.
                for j in (k + 1)..ke {
                    let wj_dinv = panel[k * nrow + j] * dinv;
                    if wj_dinv != T::zero() {
                        for i in j..nrow {
                            panel[j * nrow + i] =
                                panel[j * nrow + i] - panel[k * nrow + i] * wj_dinv;
                        }
                    }
                }
                for i in (k + 1)..nrow {
                    panel[k * nrow + i] = panel[k * nrow + i] * dinv;
                }
                k += 1;
            } else {
                if kp != k + 1 {
                    swap_sym_lower(&mut panel, nrow, k + 1, kp);
                    lperm.swap(k + 1, kp);
                }
                let mut d11 = panel[k + k * nrow];
                let d21 = panel[k * nrow + (k + 1)];
                let mut d22 = panel[(k + 1) + (k + 1) * nrow];
                let mut det = d11 * d22 - d21 * d21;
                let scale = d11.magnitude().max(d22.magnitude()).max(d21.magnitude());
                let growth_floor = GROWTH_EPS * scale * scale;
                match perturb_floor {
                    Some(floor) => {
                        let fl = (floor * floor).max(growth_floor);
                        if det.magnitude() < fl {
                            let lift = floor.max(scale * GROWTH_EPS.sqrt());
                            d11 = d11 + T::from_real(lift);
                            d22 = d22 + T::from_real(lift);
                            det = d11 * d22 - d21 * d21;
                            if det.magnitude() < fl {
                                det = det + T::from_real(fl);
                            }
                            local_perturbed += 1;
                        }
                    }
                    None if det.magnitude() <= growth_floor => {
                        restore_gloc!();
                        return Err(RslabError::NumericallyRankDeficient);
                    }
                    _ => {}
                }
                let detinv = det.recip();
                d[k] = d11;
                d_subdiag[k] = d21;
                d[k + 1] = d22;
                two_by_two[k] = true;
                for i in (k + 2)..nrow {
                    let wik = panel[k * nrow + i];
                    let wik1 = panel[(k + 1) * nrow + i];
                    l1[i] = (d22 * wik - d21 * wik1) * detinv;
                    l2[i] = (d11 * wik1 - d21 * wik) * detinv;
                }
                for j in (k + 2)..ke {
                    let l1j = l1[j];
                    let l2j = l2[j];
                    for i in j..nrow {
                        panel[j * nrow + i] = panel[j * nrow + i]
                            - panel[k * nrow + i] * l1j
                            - panel[(k + 1) * nrow + i] * l2j;
                    }
                }
                for i in (k + 2)..nrow {
                    panel[k * nrow + i] = l1[i];
                    panel[(k + 1) * nrow + i] = l2[i];
                }
                k += 2;
            }
        }

        if let Some(t) = t_g {
            PROF_LDLT_GETF2_NS.fetch_add(t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }
        let t_s = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // Deferred panel trailing update: panel[ke.., ke..ncol] −= L21·D·Rᵀ, where
        // L21 = panel rows [ke,nrow) × panel cols [kb,ke) (mt×pw), G = L21·D (block-
        // diagonal D), and R = the first `cw` rows of L21 (the rows that are
        // themselves remaining panel columns [ke,ncol)). The result `tmp` is the
        // rectangular `mt × cw` Schur block; only its lower part is written back.
        let pw = ke - kb;
        let cw = ncol - ke; // remaining fully-summed columns to update
        let mt = nrow - ke; // trailing rows (left-factor height)
        if pw > 0 && cw > 0 && mt > 0 {
            l21buf.clear();
            l21buf.resize(mt * pw, T::zero());
            for cc in 0..pw {
                let c = kb + cc;
                for rr in 0..mt {
                    l21buf[rr + cc * mt] = panel[(ke + rr) + c * nrow];
                }
            }
            gbuf.clear();
            gbuf.resize(mt * pw, T::zero());
            let mut cc = 0;
            while cc < pw {
                let c = kb + cc;
                if two_by_two[c] {
                    let (d11, d21, d22) = (d[c], d_subdiag[c], d[c + 1]);
                    for rr in 0..mt {
                        let a = l21buf[rr + cc * mt];
                        let b = l21buf[rr + (cc + 1) * mt];
                        gbuf[rr + cc * mt] = a * d11 + b * d21;
                        gbuf[rr + (cc + 1) * mt] = a * d21 + b * d22;
                    }
                    cc += 2;
                } else {
                    let dc = d[c];
                    for rr in 0..mt {
                        gbuf[rr + cc * mt] = l21buf[rr + cc * mt] * dc;
                    }
                    cc += 1;
                }
            }
            tmp.clear();
            tmp.resize(mt * cw, T::zero());
            let par = if (mt as u128) * (cw as u128) * (pw as u128) >= ll_cdiv_par as u128 {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            if kt.use_gemm_schur {
                // SAFETY: `tmp`, `gbuf`, `l21buf` are three distinct, non-overlapping
                // allocations sized for (m,n,k)=(mt,cw,pw); `R` is the top `cw` rows
                // of `l21buf` (cw ≤ mt), addressed via the rhs strides. `T` is
                // `f64`/`Complex<f64>` (gemm-supported).
                unsafe {
                    gemm::gemm(
                        mt,
                        cw,
                        pw,
                        tmp.as_mut_ptr(),
                        mt as isize,
                        1,
                        false,
                        gbuf.as_ptr(),
                        mt as isize,
                        1,
                        l21buf.as_ptr(),
                        1,
                        mt as isize,
                        T::zero(),
                        T::one(),
                        false,
                        false,
                        false,
                        par,
                    );
                }
            } else {
                for cc2 in 0..cw {
                    for rr in 0..mt {
                        let mut acc = T::zero();
                        for kk2 in 0..pw {
                            acc = acc + gbuf[rr + kk2 * mt] * l21buf[cc2 + kk2 * mt];
                        }
                        tmp[rr + cc2 * mt] = acc;
                    }
                }
            }
            // Subtract the lower part: column c = ke+cc2 gets rows r = ke+rr, rr ≥ cc2.
            for cc2 in 0..cw {
                let c = ke + cc2;
                for rr in cc2..mt {
                    let dst = (ke + rr) + c * nrow;
                    panel[dst] = panel[dst] - tmp[rr + cc2 * mt];
                }
            }
        }
        if let Some(t) = t_s {
            PROF_LDLT_SCHUR_NS.fetch_add(t.elapsed().as_nanos() as u64, AtomicOrdering::Relaxed);
        }
        kb = ke;
    }
    if local_perturbed > 0 {
        n_perturbed.fetch_add(local_perturbed, Ordering::Relaxed);
    }
    for &g in &rs[s] {
        gloc[g] = usize::MAX;
    }
    GLOC_SCRATCH.with(|c| *c.borrow_mut() = gloc);
    // Populate the O(n) emit maps + inertia for `s` (block-aware over its 1×1/2×2
    // Bunch-Kaufman D), mirroring the legacy pass-1 emit. The `e`-numbering is one
    // position per column, so `e_offset[s] + p` is column `p`'s elimination index.
    let eoff = emit.e_offset[s];
    let (mut ipos, mut ineg, mut izero) = (0usize, 0usize, 0usize);
    let mut pp = 0;
    while pp < ncol {
        let g = rs[s][lperm[pp]];
        let e = eoff + pp;
        // SAFETY: each global index / position is written by exactly one supernode.
        unsafe {
            *emit.e_of_g[g].get() = e;
            *emit.perm[e].get() = sym.perm[g];
            *emit.d_diag[e].get() = d[pp];
        }
        if two_by_two[pp] {
            let g2 = rs[s][lperm[pp + 1]];
            unsafe {
                *emit.e_of_g[g2].get() = e + 1;
                *emit.perm[e + 1].get() = sym.perm[g2];
                *emit.d_diag[e + 1].get() = d[pp + 1];
                *emit.d_subdiag[e].get() = d_subdiag[pp];
                *emit.two_by_two[e].get() = true;
            }
            let det_r = (d[pp] * d[pp + 1] - d_subdiag[pp] * d_subdiag[pp]).real();
            let tr_r = (d[pp] + d[pp + 1]).real();
            if det_r < 0.0 {
                ipos += 1;
                ineg += 1;
            } else if det_r > 0.0 {
                if tr_r >= 0.0 {
                    ipos += 2;
                } else {
                    ineg += 2;
                }
            } else {
                izero += 1;
                if tr_r >= 0.0 {
                    ipos += 1;
                } else {
                    ineg += 1;
                }
            }
            pp += 2;
        } else {
            let r = d[pp].real();
            if r > 0.0 {
                ipos += 1;
            } else if r < 0.0 {
                ineg += 1;
            } else {
                izero += 1;
            }
            pp += 1;
        }
    }
    emit.inertia_pos.fetch_add(ipos, Ordering::Relaxed);
    emit.inertia_neg.fetch_add(ineg, Ordering::Relaxed);
    emit.inertia_zero.fetch_add(izero, Ordering::Relaxed);
    // SAFETY: this thread owns supernode `s` and writes its cell exactly once.
    unsafe { store.set(s, panel, d, d_subdiag, two_by_two, lperm) };
    Ok(())
}

/// Factor the assembly subtree rooted at `s` with a work-stealing schedule:
/// children subtrees concurrently, then this node (whose updaters all lie in the
/// now-factored subtree). The left-looking analogue of the multifrontal driver.
#[allow(clippy::too_many_arguments)]
fn ll_factor_subtree<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &CscMatrix<T>,
    rs: &[Vec<usize>],
    update_list: &[Vec<usize>],
    store: &LlStore<T>,
    emit: &LlEmitLdlt<T>,
    perturb_floor: Option<f64>,
    drop_tol: Option<f64>,
    n_perturbed: &AtomicUsize,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    sym.supernodes[s]
        .children
        .par_iter()
        .map(|&ch| {
            ll_factor_subtree(
                ch,
                sym,
                a_perm,
                rs,
                update_list,
                store,
                emit,
                perturb_floor,
                drop_tol,
                n_perturbed,
                kt,
            )
        })
        .collect::<Result<Vec<()>, _>>()?;
    ll_factor_node(
        s,
        sym,
        a_perm,
        rs,
        update_list,
        store,
        emit,
        perturb_floor,
        n_perturbed,
        kt,
    )?;
    // Free each descendant whose last consumer this node was (refcount→0), and `s`
    // itself if it has no consumers - compacting before releasing the dense panel.
    // Disjoint `k`, so the wide top-of-tree free runs in parallel.
    const FREE_PAR: usize = 64;
    if update_list[s].len() >= FREE_PAR {
        update_list[s].par_iter().for_each(|&k| {
            if emit.refcount[k].fetch_sub(1, Ordering::AcqRel) == 1 {
                ldlt_emit_and_free(k, store, emit, sym, rs, drop_tol);
            }
        });
    } else {
        for &k in &update_list[s] {
            if emit.refcount[k].fetch_sub(1, Ordering::AcqRel) == 1 {
                ldlt_emit_and_free(k, store, emit, sym, rs, drop_tol);
            }
        }
    }
    if emit.refcount[s].load(Ordering::Relaxed) == 0 {
        ldlt_emit_and_free(s, store, emit, sym, rs, drop_tol);
    }
    Ok(())
}

/// Supernodal **left-looking** LDLᵀ with **Bunch-Kaufman 1×1/2×2 pivoting**. Each
/// supernode's dense panel is assembled from `A`, updated by every previously
/// factored descendant (`cmod`: pull the descendant's contribution columns that
/// land in this panel, applying its block-diagonal `D`), then factored in place
/// (`cdiv`: partial Bunch-Kaufman, no trailing update). Pivoting is bounded to
/// each panel's fully-summed block, so the off-diagonal rows keep their identity
/// and the descendant→ancestor `cmod` is unaffected by a panel's internal
/// permutation. There is **no contribution-block stack and no extract copy-out**
/// (the panels are the factor), so the transient is just the factor itself (the
/// PARDISO memory profile). Produces the same [`LdltFactors`] as the multifrontal
/// path (numerically equivalent up to pivot order), including indefinite
/// (zero-/tiny-diagonal) systems via the 2×2 blocks.
fn factor_left_looking<T: Scalar>(
    sym: &SymbolicFactorization,
    a: &CscMatrix<T>,
    opts: &SolverSettings,
) -> Result<LdltFactors<T>, RslabError> {
    let n = sym.n;
    let perturb_floor: Option<f64> = match opts.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = a.values.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    };

    // A_perm = Pᵀ A P, lower triangle (same build as the multifrontal path).
    let nnz = a.row_idx.len();
    let (mut rows, mut cols, mut vals) = (
        Vec::with_capacity(nnz),
        Vec::with_capacity(nnz),
        Vec::with_capacity(nnz),
    );
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let (gi, gj) = (sym.perm_inv[i], sym.perm_inv[j]);
            let (r, c) = if gi >= gj { (gi, gj) } else { (gj, gi) };
            rows.push(r);
            cols.push(c);
            vals.push(a.values[k]);
        }
    }
    let a_perm = CscMatrix::<T>::from_triplets(n, &rows, &cols, &vals)?;

    let nsuper = sym.supernodes.len();
    let rs = compute_supernode_row_structures(sym);

    // Map each global column to its owning supernode, and build per-supernode
    // updater lists: `k` updates `s` iff one of `k`'s off-diagonal rows is an
    // eliminated column of `s` (its sorted off-diag rows hit `s`'s column run).
    let mut col_to_snode = vec![0usize; n];
    for (s, snode) in sym.supernodes.iter().enumerate() {
        col_to_snode[snode.first_col..snode.first_col + snode.ncol].fill(s);
    }
    let mut update_list: Vec<Vec<usize>> = vec![Vec::new(); nsuper];
    for (k, rsk) in rs.iter().enumerate() {
        let nck = sym.supernodes[k].ncol;
        let mut last = usize::MAX;
        for &r in &rsk[nck..] {
            let s = col_to_snode[r];
            if s != last {
                update_list[s].push(k);
                last = s;
            }
        }
    }

    // Factor in parallel over the assembly forest: sibling subtrees concurrently,
    // each node after its subtree (whose panels are its only updaters). Panels are
    // written once and read only by ancestors → no synchronization needed beyond
    // the recursion structure (see `LlStore`).
    let store = LlStore::<T>::new(nsuper);
    let emit = LlEmitLdlt::<T>::new(sym, &update_list);
    let n_perturbed_atomic = AtomicUsize::new(0);
    let mut is_child = vec![false; nsuper];
    for snode in &sym.supernodes {
        for &ch in &snode.children {
            is_child[ch] = true;
        }
    }
    let roots: Vec<usize> = (0..nsuper).filter(|&s| !is_child[s]).collect();
    let kt = opts.kernel();
    roots
        .par_iter()
        .map(|&r| {
            ll_factor_subtree(
                r,
                sym,
                &a_perm,
                &rs,
                &update_list,
                &store,
                &emit,
                perturb_floor,
                opts.drop_tol,
                &n_perturbed_atomic,
                kt,
            )
        })
        .collect::<Result<Vec<()>, _>>()?;
    drop(store); // panels freed incrementally; release the shells
    let n_perturbed = n_perturbed_atomic.load(Ordering::Relaxed);
    if ldlt_prof_on() {
        let g = PROF_LDLT_GETF2_NS.swap(0, AtomicOrdering::Relaxed) as f64 / 1e6;
        let s = PROF_LDLT_SCHUR_NS.swap(0, AtomicOrdering::Relaxed) as f64 / 1e6;
        let t = (g + s).max(1.0);
        eprintln!(
            "[RLA_LDLT_CDIV] CPU-ms  getf2(BK panel) {g:.0} ({:.0}% ser)  schur(deferred GEMM) {s:.0} ({:.0}% par)",
            100.0 * g / t,
            100.0 * s / t,
        );
    }

    // Assemble global L (CSC) by concatenating the per-supernode compact fragments
    // produced (and freed) incrementally during factorization - taken by value and
    // dropped right after appending, so the peak is the growing CSC + one fragment,
    // not all fragments + the full CSC. D / perm / inertia were populated in-node.
    let mut l_col_ptr = Vec::with_capacity(n + 1);
    l_col_ptr.push(0);
    let mut l_row_idx: Vec<usize> = Vec::new();
    let mut l_values: Vec<T> = Vec::new();
    for (s, snode) in sym.supernodes.iter().enumerate() {
        // SAFETY: factorization complete; `compact[s]` written exactly once.
        let cl = unsafe { std::mem::take(&mut *emit.compact[s].get()) };
        for c in 0..snode.ncol {
            let (a, b) = (cl.ptr[c], cl.ptr[c + 1]);
            l_row_idx.extend_from_slice(&cl.idx[a..b]);
            l_values.extend_from_slice(&cl.val[a..b]);
            l_col_ptr.push(l_row_idx.len());
        }
    }
    // SAFETY: factorization complete; every position written exactly once in-node.
    let perm: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm[e].get() }).collect();
    let d_diag: Vec<T> = (0..n).map(|e| unsafe { *emit.d_diag[e].get() }).collect();
    let d_subdiag: Vec<T> = (0..n)
        .map(|e| unsafe { *emit.d_subdiag[e].get() })
        .collect();
    let two_by_two: Vec<bool> = (0..n)
        .map(|e| unsafe { *emit.two_by_two[e].get() })
        .collect();
    let inertia = Inertia::new(
        emit.inertia_pos.load(Ordering::Relaxed),
        emit.inertia_neg.load(Ordering::Relaxed),
        emit.inertia_zero.load(Ordering::Relaxed),
    );

    Ok(LdltFactors {
        n,
        l_col_ptr,
        l_row_idx,
        l_values,
        d_diag,
        d_subdiag,
        two_by_two,
        perm,
        n_perturbed,
        inertia,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::ldlt_generic::solve_ldlt;
    use num_complex::Complex;

    /// A tridiagonal (chain) matrix with no amalgamation (`nemin = 1`) builds an
    /// assembly tree as deep as the matrix; the recursive tree factorization must
    /// not overflow the worker stack on either path. Regression for the
    /// `STATUS_STACK_OVERFLOW` the auto-tuning sweep hit on banded + `nemin = 1`.
    #[test]
    fn deep_chain_tree_does_not_overflow_stack() {
        let n = 20_000usize;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(4.0f64);
            if i + 1 < n {
                rows.push(i + 1);
                cols.push(i);
                vals.push(-1.0);
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &vals).unwrap();
        for method in [FactorMethod::LeftLooking, FactorMethod::Multifrontal] {
            let s = SolverSettings::default()
                .with_method(method)
                .with_nemin(1)
                .with_threads(0);
            let f = factor_sparse_ldlt_with(&a, &s).expect("deep chain factors without overflow");
            assert_eq!(f.n, n);
        }
    }

    #[test]
    fn mf_ldlt_low_memory_emit_is_bit_identical() {
        // On the multifrontal LDLᵀ path, MemoryMode::LowMemory frees each front's
        // dense L during the global emit; it must produce exactly the same global
        // L (values, row indices, column pointers) as Eager - it changes only when
        // the per-front buffers are dropped, never the emitted factor.
        let m = 12;
        let n = m * m;
        let idx = |a: usize, b: usize| a * m + b;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                r.push(p);
                cc.push(p);
                v.push(6.0_f64);
                if b + 1 < m {
                    r.push(idx(a, b + 1));
                    cc.push(p);
                    v.push(-1.0);
                }
                if a + 1 < m {
                    r.push(idx(a + 1, b));
                    cc.push(p);
                    v.push(-1.0);
                }
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
        let mf = |mem| {
            SolverSettings::default()
                .with_method(FactorMethod::Multifrontal)
                .with_memory(mem)
                .with_threads(0)
        };
        let eager = factor_sparse_ldlt_with(&a, &mf(MemoryMode::Eager)).unwrap();
        let low = factor_sparse_ldlt_with(&a, &mf(MemoryMode::LowMemory)).unwrap();
        assert_eq!(
            eager.l_values, low.l_values,
            "L values differ under LowMemory"
        );
        assert_eq!(eager.l_row_idx, low.l_row_idx, "L row indices differ");
        assert_eq!(eager.l_col_ptr, low.l_col_ptr, "L column pointers differ");
        assert_eq!(eager.d_diag, low.d_diag, "D differs under LowMemory");
    }

    #[test]
    fn parallel_front_subtraction_is_bit_identical() {
        // 2D front parallelism (the trailing-Schur subtraction split across
        // disjoint front columns) must be bit-identical to the serial path
        // regardless of the parallel gate - the determinism guarantee. Force the
        // parallel path (par_cdiv = 0) vs the serial path (par_cdiv = MAX) and
        // compare the whole factor.
        let m = 22;
        let n = m * m;
        let idx = |a: usize, b: usize| a * m + b;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                r.push(p);
                cc.push(p);
                v.push(6.0_f64);
                if b + 1 < m {
                    r.push(idx(a, b + 1));
                    cc.push(p);
                    v.push(-1.0);
                }
                if a + 1 < m {
                    r.push(idx(a + 1, b));
                    cc.push(p);
                    v.push(-1.0);
                }
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
        let mk = |cdiv| {
            SolverSettings::default()
                .with_method(FactorMethod::Multifrontal)
                .with_threads(0)
                .with_gemm_thresholds(crate::GemmThresholds {
                    scalar_gate: 4096,
                    par_gemm: 1_000_000,
                    par_cdiv: cdiv,
                })
        };
        let parallel = factor_sparse_ldlt_with(&a, &mk(0)).unwrap();
        let serial = factor_sparse_ldlt_with(&a, &mk(usize::MAX)).unwrap();
        assert_eq!(
            parallel.l_values, serial.l_values,
            "parallel front subtraction not bit-identical"
        );
        assert_eq!(parallel.d_diag, serial.d_diag);
    }

    #[test]
    fn right_looking_matches_multifrontal_and_solves() {
        // The right-looking schedule reuses the same per-front assembly + emit as
        // the multifrontal path (only the traversal differs: sequential postorder,
        // all fronts live), so its factor must be *bit-identical* to multifrontal
        // and solve correctly (numerically equivalent to left-looking).
        let m = 12;
        let n = m * m;
        let idx = |a: usize, b: usize| a * m + b;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                r.push(p);
                cc.push(p);
                v.push(6.0_f64);
                if b + 1 < m {
                    r.push(idx(a, b + 1));
                    cc.push(p);
                    v.push(-1.0);
                }
                if a + 1 < m {
                    r.push(idx(a + 1, b));
                    cc.push(p);
                    v.push(-1.0);
                }
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        let mk = |method| {
            SolverSettings::default()
                .with_method(method)
                .with_threads(0)
        };
        let mf = factor_sparse_ldlt_with(&a, &mk(FactorMethod::Multifrontal)).unwrap();
        let rl = factor_sparse_ldlt_with(&a, &mk(FactorMethod::RightLooking)).unwrap();
        assert_eq!(
            rl.l_values, mf.l_values,
            "right-looking L differs from multifrontal"
        );
        assert_eq!(
            rl.l_row_idx, mf.l_row_idx,
            "right-looking L pattern differs"
        );
        assert_eq!(rl.d_diag, mf.d_diag, "right-looking D differs");
        let x = solve_ldlt(&rl, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-9,
            "right-looking residual {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn rcm_and_autorace_orderings_factor_and_solve() {
        // A 2D-grid SPD system must factor and solve correctly under the new RCM
        // ordering and under AutoRace (which now includes RCM as a candidate).
        let m = 14;
        let n = m * m;
        let idx = |a: usize, b: usize| a * m + b;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                r.push(p);
                cc.push(p);
                v.push(6.0_f64);
                if b + 1 < m {
                    r.push(idx(a, b + 1));
                    cc.push(p);
                    v.push(-1.0);
                }
                if a + 1 < m {
                    r.push(idx(a + 1, b));
                    cc.push(p);
                    v.push(-1.0);
                }
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        for ord in [OrderingMethod::Rcm, OrderingMethod::AutoRace] {
            let opts = SolverSettings::default().with_ordering(ord);
            let symb = analyze_with(a.n, &a.col_ptr, &a.row_idx, &opts).unwrap();
            let f = factor_numeric(&symb, &a, &opts).unwrap();
            let x = solve_ldlt(&f, &b).unwrap();
            assert!(
                residual_inf(&a, &x, &b) < 1e-9,
                "ordering {ord:?} residual {}",
                residual_inf(&a, &x, &b)
            );
        }
    }

    fn residual_inf<T: Scalar>(a: &CscMatrix<T>, x: &[T], b: &[T]) -> f64 {
        let mut ax = vec![T::zero(); a.n];
        a.symv(x, &mut ax);
        (0..a.n)
            .map(|i| (ax[i] - b[i]).magnitude())
            .fold(0.0, f64::max)
    }

    /// 1D Laplacian-style SPD tridiagonal of size n (diag 2+something, off −1).
    fn tridiag_spd_f64(n: usize) -> CscMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(4.0);
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(-1.0);
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn f64_sparse_tridiag_residual() {
        let a = tridiag_spd_f64(20);
        let b: Vec<f64> = (0..20).map(|i| (i as f64) - 9.5).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-10);
    }

    /// 2D 5-point grid (m×m), lower triangle, complex-symmetric, diagonally
    /// dominant. Branching assembly tree → exercises multi-child `cmod`.
    fn grid2d_lower<T: Scalar>(m: usize, diag: T, off: T) -> CscMatrix<T> {
        let n = m * m;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |r: usize, c: usize| r * m + c;
        let mut push = |i: usize, j: usize, v: T| {
            let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
            rows.push(hi);
            cols.push(lo);
            vals.push(v);
        };
        for r in 0..m {
            for c in 0..m {
                let p = idx(r, c);
                push(p, p, diag);
                if c + 1 < m {
                    push(p, idx(r, c + 1), off);
                }
                if r + 1 < m {
                    push(p, idx(r + 1, c), off);
                }
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn left_looking_matches_multifrontal_f64() {
        // Chain assembly tree (tridiagonal): exercises the basic left-looking
        // cmod/cdiv. Same fill and same solution as the multifrontal path.
        let a = tridiag_spd_f64(50);
        let b: Vec<f64> = (0..50).map(|i| (i % 7) as f64 - 3.0).collect();
        let mf = factor_sparse_ldlt_with(&a, &SolverSettings::default()).unwrap();
        let ll = factor_sparse_ldlt_with(
            &a,
            &SolverSettings::default().with_method(FactorMethod::LeftLooking),
        )
        .unwrap();
        assert_eq!(mf.l_values.len(), ll.l_values.len(), "fill must match");
        let xm = solve_ldlt(&mf, &b).unwrap();
        let xl = solve_ldlt(&ll, &b).unwrap();
        assert!(residual_inf(&a, &xl, &b) < 1e-9, "left-looking residual");
        let diff = (0..50).map(|i| (xm[i] - xl[i]).abs()).fold(0.0, f64::max);
        assert!(diff < 1e-9, "solutions differ by {diff}");
    }

    #[test]
    fn left_looking_2d_grid_matches_multifrontal() {
        // Branching assembly tree → multi-child cmod and deeper update lists.
        let a = grid2d_lower::<f64>(12, 8.0, -1.0);
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        let mf = factor_sparse_ldlt_with(&a, &SolverSettings::default()).unwrap();
        let ll = factor_sparse_ldlt_with(
            &a,
            &SolverSettings::default().with_method(FactorMethod::LeftLooking),
        )
        .unwrap();
        assert_eq!(mf.l_values.len(), ll.l_values.len(), "fill must match");
        let xl = solve_ldlt(&ll, &b).unwrap();
        assert!(
            residual_inf(&a, &xl, &b) < 1e-9,
            "left-looking grid residual"
        );
    }

    #[test]
    fn left_looking_complex_symmetric_type_agnostic() {
        // The left-looking path is generic over `Scalar`: complex-symmetric here.
        let c = |re: f64, im: f64| Complex::new(re, im);
        let a = grid2d_lower::<Complex<f64>>(10, c(8.0, 1.0), c(-1.0, 0.2));
        let n = a.n;
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 0.5)).collect();
        let ll = factor_sparse_ldlt_with(
            &a,
            &SolverSettings::default().with_method(FactorMethod::LeftLooking),
        )
        .unwrap();
        let xl = solve_ldlt(&ll, &b).unwrap();
        assert!(
            residual_inf(&a, &xl, &b) < 1e-9,
            "complex left-looking residual"
        );
    }

    #[test]
    fn left_looking_indefinite_2x2_inertia() {
        // [[0,1],[1,0]] (eigenvalues ±1) forces a single 2×2 Bunch-Kaufman block.
        // The left-looking path must take that 2×2 (zero diagonal → no 1×1 pivot)
        // and report inertia (1+, 1−) just like the multifrontal kernel.
        let a = CscMatrix::<f64>::from_triplets(2, &[0, 1], &[0, 0], &[0.0, 1.0]).unwrap();
        let ll = factor_sparse_ldlt_with(
            &a,
            &SolverSettings::default().with_method(FactorMethod::LeftLooking),
        )
        .unwrap();
        assert!(ll.two_by_two.iter().any(|&t| t), "expected a 2×2 block");
        assert_eq!(
            (ll.inertia.positive, ll.inertia.negative, ll.inertia.zero),
            (1, 1, 0)
        );
        let b = [1.0_f64, -2.0];
        let x = solve_ldlt(&ll, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-12, "2×2 residual");
    }

    #[test]
    fn left_looking_indefinite_matches_multifrontal() {
        // 2D 5-point grid with a *small* diagonal (0.5 ≪ 2·|off|): far from
        // diagonally dominant → genuinely indefinite, so Bunch-Kaufman must take
        // many 2×2 pivots across several supernodes. The left-looking path must
        // match the multifrontal reference in inertia and give a true solve - the
        // exact indefinite EM-FEM case the 2×2 pivoting is for.
        let a = grid2d_lower::<f64>(10, 0.5, -1.0);
        let n = a.n;
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let mf = factor_sparse_ldlt_with(&a, &SolverSettings::default()).unwrap();
        let ll = factor_sparse_ldlt_with(
            &a,
            &SolverSettings::default().with_method(FactorMethod::LeftLooking),
        )
        .unwrap();
        assert!(
            ll.two_by_two.iter().filter(|&&t| t).count() > 0,
            "indefinite system should use 2×2 pivots"
        );
        assert_eq!(
            (mf.inertia.positive, mf.inertia.negative, mf.inertia.zero),
            (ll.inertia.positive, ll.inertia.negative, ll.inertia.zero),
            "inertia must match the multifrontal reference"
        );
        let xm = solve_ldlt(&mf, &b).unwrap();
        let xl = solve_ldlt(&ll, &b).unwrap();
        assert!(
            residual_inf(&a, &xl, &b) < 1e-9,
            "left-looking indefinite residual"
        );
        assert!(
            residual_inf(&a, &xm, &b) < 1e-9,
            "multifrontal indefinite residual"
        );
        let diff = (0..n).map(|i| (xm[i] - xl[i]).abs()).fold(0.0, f64::max);
        assert!(diff < 1e-7, "solutions differ by {diff}");
    }

    #[test]
    fn left_looking_indefinite_complex_symmetric() {
        // Complex-symmetric indefinite grid: the 2×2 path is type-agnostic. The
        // 2×2 blocks here are complex-symmetric (not Hermitian), exercising the
        // generic det/detinv arithmetic. Compare inertia + solve to multifrontal.
        let c = |re: f64, im: f64| Complex::new(re, im);
        let a = grid2d_lower::<Complex<f64>>(9, c(0.4, 0.3), c(-1.0, 0.1));
        let n = a.n;
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 0.5)).collect();
        let mf = factor_sparse_ldlt_with(&a, &SolverSettings::default()).unwrap();
        let ll = factor_sparse_ldlt_with(
            &a,
            &SolverSettings::default().with_method(FactorMethod::LeftLooking),
        )
        .unwrap();
        assert!(
            ll.two_by_two.iter().filter(|&&t| t).count() > 0,
            "indefinite system should use 2×2 pivots"
        );
        assert_eq!(
            (mf.inertia.positive, mf.inertia.negative, mf.inertia.zero),
            (ll.inertia.positive, ll.inertia.negative, ll.inertia.zero),
            "inertia must match the multifrontal reference"
        );
        let xl = solve_ldlt(&ll, &b).unwrap();
        assert!(
            residual_inf(&a, &xl, &b) < 1e-9,
            "complex left-looking indefinite residual"
        );
    }

    #[test]
    fn f64_dense_front_blocked_multi_panel() {
        // A fully dense symmetric matrix is one front of width n=100 > NB(64),
        // so factoring it exercises the blocked **multi-panel** Bunch-Kaufman
        // path (which the small n≤50 tests never reach). Diagonally dominant SPD.
        let n = 100;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            for i in j..n {
                rows.push(i);
                cols.push(j);
                vals.push(if i == j {
                    n as f64 + 1.0
                } else {
                    ((i + 2 * j) % 5) as f64 - 2.0
                });
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-9,
            "residual {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn complex_dense_front_blocked_multi_panel() {
        // Dense complex-symmetric, one front of width 90 > NB → multi-panel.
        let c = |re: f64, im: f64| Complex::new(re, im);
        let n = 90;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            for i in j..n {
                rows.push(i);
                cols.push(j);
                vals.push(if i == j {
                    c(n as f64, 1.0)
                } else {
                    c(((i + 3 * j) % 5) as f64 - 2.0, 0.2)
                });
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b = vec![c(1.0, 0.5); n];
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-9);
    }

    #[test]
    fn f64_sparse_2d_grid_residual() {
        // 2D 5-point Laplacian on a 5×5 grid (n=25), SPD.
        let m = 5;
        let n = m * m;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let idx = |r: usize, c: usize| r * m + c;
        for r in 0..m {
            for c in 0..m {
                let p = idx(r, c);
                rows.push(p);
                cols.push(p);
                vals.push(4.0);
                // lower-triangle neighbors only
                if c + 1 < m {
                    let q = idx(r, c + 1);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(-1.0);
                }
                if r + 1 < m {
                    let q = idx(r + 1, c);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(-1.0);
                }
            }
        }
        let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) - 3.0).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-9,
            "residual {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn complex_sparse_tridiag_residual() {
        // Complex-symmetric Helmholtz-style tridiagonal: diagonal (4 + 0.5i),
        // off-diagonal (−1 + 0.1i). Complex symmetric (A = Aᵀ), diagonally
        // dominant so the fully-summed blocks stay nonsingular.
        let c = |re, im| Complex::new(re, im);
        let n = 16;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(c(4.0, 0.5));
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(c(-1.0, 0.1));
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 7.5, 1.0 - i as f64)).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-10,
            "residual {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn complex_sparse_large_grid_parallel() {
        // 12×12 complex-symmetric grid (n=144): a deep, bushy assembly tree
        // that genuinely exercises multiple parallel levels in the rayon driver.
        let c = |re, im| Complex::new(re, im);
        let m = 12;
        let n = m * m;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let idx = |r: usize, cc: usize| r * m + cc;
        for r in 0..m {
            for cc in 0..m {
                let p = idx(r, cc);
                rows.push(p);
                cols.push(p);
                vals.push(c(4.0, 0.5));
                if cc + 1 < m {
                    let q = idx(r, cc + 1);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(c(-1.0, 0.1));
                }
                if r + 1 < m {
                    let q = idx(r + 1, cc);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(c(-1.0, 0.1));
                }
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 11) as f64 - 5.0, 1.0)).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-9,
            "residual {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn perturb_rescues_singular_complex() {
        // Structurally singular complex-symmetric system: index 1 is fully
        // decoupled with a zero diagonal (zero row/column). Exact mode must
        // fail; static-pivoting (preconditioner) mode must succeed, report a
        // perturbation, and produce a finite, solvable factor of `A + E`.
        let c = |re, im| Complex::new(re, im);
        let n = 3;
        let rows = vec![0, 2, 1];
        let cols = vec![0, 0, 1];
        let vals = vec![c(2.0, 1.0), c(-1.0, 0.3), c(0.0, 0.0)];
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();

        assert!(
            factor_sparse_ldlt(&a).is_err(),
            "exact mode should reject the singular pivot"
        );

        let opts = SolverSettings {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-8 },
            drop_tol: None,
            ..Default::default()
        };
        let f = factor_sparse_ldlt_with(&a, &opts).unwrap();
        assert!(
            f.n_perturbed >= 1,
            "expected ≥1 perturbation, got {}",
            f.n_perturbed
        );
        let b = vec![c(1.0, 0.0); n];
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            x.iter().all(|v| v.norm().is_finite()),
            "factor must stay finite"
        );
    }

    #[test]
    fn exact_mode_never_perturbs_well_conditioned() {
        // A diagonally dominant complex-symmetric grid factors exactly with no
        // perturbation - the static-pivot path must not trigger spuriously.
        let a = {
            let c = |re, im| Complex::new(re, im);
            let n = 16;
            let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
            for j in 0..n {
                r.push(j);
                cc.push(j);
                v.push(c(4.0, 0.5));
                if j + 1 < n {
                    r.push(j + 1);
                    cc.push(j);
                    v.push(c(-1.0, 0.1));
                }
            }
            CscMatrix::<Complex<f64>>::from_triplets(n, &r, &cc, &v).unwrap()
        };
        let opts = SolverSettings {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-8 },
            drop_tol: None,
            ..Default::default()
        };
        let f = factor_sparse_ldlt_with(&a, &opts).unwrap();
        assert_eq!(
            f.n_perturbed, 0,
            "well-conditioned matrix needs no perturbation"
        );
    }

    #[test]
    fn complex_sparse_2d_grid_residual() {
        // 2D complex-symmetric grid: diagonal (4 + i), neighbor (−1 + 0.2i).
        let c = |re, im| Complex::new(re, im);
        let m = 5;
        let n = m * m;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        let idx = |r: usize, cc: usize| r * m + cc;
        for r in 0..m {
            for cc in 0..m {
                let p = idx(r, cc);
                rows.push(p);
                cols.push(p);
                vals.push(c(4.0, 1.0));
                if cc + 1 < m {
                    let q = idx(r, cc + 1);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(c(-1.0, 0.2));
                }
                if r + 1 < m {
                    let q = idx(r + 1, cc);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(c(-1.0, 0.2));
                }
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let f = factor_sparse_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-9,
            "residual {}",
            residual_inf(&a, &x, &b)
        );
    }
}
