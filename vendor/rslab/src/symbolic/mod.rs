pub mod column_counts;
pub mod ldlt_compress;
pub mod profiler;
pub mod small_leaf;
pub mod supernode;

use crate::error::RslabError;
use crate::ordering::amd::permute_pattern;
use crate::ordering::elimination_tree::EliminationTree;
use crate::ordering::postorder::{biased_postorder, postorder};
use crate::sparse::csc::{CscMatrix, CscPattern};

pub use column_counts::{column_counts, column_counts_gnp, total_factor_nnz};
pub use ldlt_compress::{build_supermap, compress_pattern, expand_permutation, SuperMap};
pub use profiler::{record_stage, StagePct, StageTiming, SymbolicProfileReport, SymbolicProfiler};
pub use small_leaf::{find_small_leaf_groups, SmallLeafGroup, SmallLeafParams};
pub use supernode::{
    find_supernodes, pick_amalgamation_strategy, AmalgamationStrategy, OrderingPreprocess,
    RelaxAmalgamation, Supernode, SupernodeParams, AUTO_MULTI_CHILD_FRAC_THRESHOLD,
};

/// Which fill-reducing ordering to use in [`symbolic_factorize_with_method`].
///
/// All methods produce a permutation; the downstream postorder
/// composition, etree construction, column counts, supernode detection,
/// and memory planning are identical regardless of method.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OrderingMethod {
    /// Approximate Minimum Degree (`rslab-amd` crate: approximate
    /// external degree with aggressive element absorption and
    /// supervariable detection, per Amestoy/Davis/Duff 1996+2004).
    /// Default. Matches SuiteSparse/faer on the oracle fixture suite.
    ///
    /// The simplified exact-external-degree implementation at
    /// `src/ordering/amd.rs` remains on disk as a reference for the
    /// algorithm's skeleton but is no longer reachable from the
    /// symbolic pipeline. See
    /// `dev/journal/2026-04-18-03.org` for the retirement evidence
    /// (34-matrix bakeoff: geomean fill tied on parity, crate
    /// 17-23% better and 18-88× faster on large).
    #[default]
    Amd,
    /// Approximate Minimum Fill (`rslab-amf` crate: HAMF4 variant
    /// of Amestoy 1999 - quotient-graph elimination scored by
    /// approximate fill `RMF(i) = (deg(i)·(deg(i)-1+2·degme) -
    /// WF(i)) / (nv(i)+1)` rather than approximate degree).
    /// Same downstream pipeline as `Amd`.
    ///
    /// Default for `n <= 10_000` per `pick_default_method`,
    /// matching MUMPS's `ana_set_ordering.F` rule for SYM=2 small
    /// matrices. Validated against MUMPS HAMF4 on the 183_293-
    /// sidecar corpus by `tests/amf_corpus_oracle.rs`: rslab nnz_L
    /// is within 1.10× MUMPS HAMF4 nnz_L on 183_277 matrices, with
    /// CHARDIS1_0000 the lone documented metric-divergence skip.
    Amf,
    /// rslab-metis multilevel nested dissection.
    MetisND,
    /// rslab-scotch nested dissection.
    ScotchND,
    /// rslab-kahip flow-based nested dissection.
    ///
    /// Includes K1 (Ost-Schulz-Strash 2021 Rule 1, conservative
    /// termination) preprocessing inside the KaHIP pipeline. Wired
    /// at `crates/rslab-kahip/src/node_nd.rs`.
    ///
    /// **Not selected by `pick_default_method`.** The session 08
    /// 41-matrix bake-off (`dev/research/ordering-kahip-driver-
    /// integration.md`) showed `KahipND` ties `MetisND` on fill
    /// geomean (1.023 vs 1.024 relative to AMD) at 4-6× the per-call
    /// symbolic-time cost (81s vs 68s vs AMD 14s, total).
    /// Reachable explicitly via `symbolic_factorize_with_method`
    /// for callers who want it.
    KahipND,
    /// Reverse Cuthill-McKee band/profile-reducing ordering
    /// (`rslab-ordering-core`: George-Liu degree-sorted BFS from a
    /// pseudo-peripheral start, reversed; Cuthill & McKee 1969,
    /// George & Liu 1981). Targets banded / structured matrices
    /// (stencils, structured FEM) where nested dissection
    /// over-separates and minimum-degree scatters fill: the band
    /// factor has less fill and factors faster. Cheap (one BFS pass).
    /// An [`AutoRace`](OrderingMethod::AutoRace) candidate; also reachable explicitly.
    Rcm,
    /// Adaptive dispatcher: picks a concrete method per-matrix from
    /// cheap pattern features (n and average degree nnz/n).
    ///
    /// Issue #50 plus its F11 follow-up (2026-05-23) collapsed the
    /// per-shape branches to one very-large-and-sparse catch on top
    /// of `pick_default_method`:
    ///   - very-large-and-sparse (n > 100_000, full nnz/n < 5) → `Amd`
    ///   - everything else delegates to `pick_default_method`
    ///     (`n <= 10_000 → Amf`, `n > 10_000 → MetisND`).
    ///
    /// **Opt-in only.** The 154k-matrix IPM bench (2026-04-18) showed
    /// `Auto` regresses sparse factor/MUMPS geomean from 0.44 (AMD)
    /// to 0.58 because the (pre-F11) small-and-sparse branch routed
    /// thousands of n<500 IPM iteration dumps to KaHIP, where K1 +
    /// multilevel setup cost 2-3× per call vs AMD. That branch is
    /// gone - `Auto`'s small-and-sparse path is now AMF via the
    /// default - but the original `Auto` warning is preserved here
    /// since the historical-bench regression evidence remains a
    /// reason to default to `Amd` outside known IPM workloads.
    ///
    /// Use `Auto` only when the workload is known to be dominated by
    /// large or `cresc132`-class matrices where the per-call setup
    /// cost amortizes. The default `symbolic_factorize` keeps `Amd`.
    /// See `dev/tried-and-rejected.md` for the full evidence.
    ///
    /// Applying `Auto` to `Auto` loops once through the dispatcher and
    /// then runs the chosen concrete method.
    Auto,
    /// Race-based dispatcher: runs full symbolic factorization on each
    /// concrete candidate in {`Amd`, `MetisND`, `ScotchND`, `KahipND`}
    /// and returns the one with the smallest `factor_nnz_estimate`.
    ///
    /// Unlike [`Auto`](OrderingMethod::Auto), which guesses the winner from cheap pattern
    /// features, `AutoRace` measures the actual symbolic outcome. Cost
    /// is ~4× a single symbolic pass (~50-500 ms total at n≈10⁵), paid
    /// once per problem because symbolic factorization is reused across
    /// numeric refactorizations with the same sparsity pattern.
    ///
    /// Motivated by issue #8: on `pinene_3200_0009` the
    /// `pick_default_method` heuristic picks `MetisND` (88 s numeric
    /// factor), but `Amd` factors in 19.5 s on the same matrix - a 4.5×
    /// win that the cheap predicate misses. Racing eliminates the
    /// guess: whichever candidate wins on this matrix is the one we
    /// use, no calibration required.
    ///
    /// Candidates that fail (e.g. external crate returns an error) are
    /// skipped; the race succeeds as long as at least one candidate
    /// produces a valid symbolic factorization. `resolved_method` on
    /// the returned `SymbolicFactorization` records the actual winner.
    AutoRace,
}

/// Resolve an `Auto` ordering to a concrete method from cheap pattern
/// features. Non-`Auto` inputs pass through unchanged.
///
/// The rule set adds shape-bakeoff branches on top of
/// [`pick_default_method`]:
///   - very-large-and-sparse (`n > 100_000`, full avg_deg < 5.0) → `Amd`
///   - arrow/bordered (issue #64): whenever the size rule would pick
///     `MetisND` (`n > 10_000`) but [`is_arrow_bordered`] detects a
///     dense border concentrating the nonzeros, override to `Amf`.
///   - thin-large (issues #67 + #73): whenever the size rule would still
///     pick `MetisND` (after the avg_deg<5 → AMD and arrow → AMF catches),
///     override to `Amf` at every `n`. Corpus A/Bs on real factor+solve
///     wall-time found AMF wins or ties MetisND across the whole population:
///     36/36 in the `(10_000, 100_000]` band (#67) and every measured
///     `n > 100_000 && avg_deg >= 5` non-arrow family (#73), including the
///     one matrix (nql180) where MetisND has smaller fill but AMF is still
///     2× faster on the real factor+solve.
///
/// Anything else delegates to `pick_default_method`. `symbolic_factorize`
/// routes through `Auto`, so the no-arg default and `Auto` resolve to the
/// same concrete method on every matrix (issue #64 unified the two paths;
/// previously the no-arg default skipped the very-large-and-sparse and
/// arrow catches).
///
/// The large-and-sparse branch swap from `ScotchND` to `Amd` is the
/// issue #50 fix (2026-05-23). On `powerflow22` (n=2.8 M,
/// full_avg_deg ≈ 3.7) the prior ScotchND route took 113.8 s
/// symbolic (15.8 M nnz_L); MetisND was 117.4 s (20.5 M nnz_L); AMD
/// was 55 s (10.4 M nnz_L). The ScotchND advantage at very large n
/// was load-bearing against the same BK pivoting cascade that
/// motivated `pick_default_method`'s chain catches; issue #46 (see
/// `pick_default_method`'s doc comment) eliminated that amplifier in
/// May 2026 and removed the justification for routing very-large
/// sparse matrices through nested dissection at all. Numeric
/// inventory: `dev/research/issue-50-numeric-inventory.csv` shows
/// the IPM corpus's [100k, 200k) bucket has AMD/MetisND num_nnz_l
/// ratio 1.00 on both representatives. See
/// `dev/research/issue-50-metisnd-symbolic-cost.md` §F7-F8.
///
/// The small-and-sparse branch (`n < 10_000 && avg_deg < 15 →
/// KahipND`) was deleted by the F11 side finding from issue #50
/// (2026-05-23). The corpus inventory in
/// `dev/research/small-sparse-inventory.csv` (838 IPM-corpus
/// matrices factored under AMD/AMF/MetisND/KahipND) shows AMF
/// dominates this population: AMF wins 169/838 per-matrix
/// (vs KahipND's 16), aggregate AMF fill is 0.87× AMD vs KahipND's
/// 0.98×, aggregate AMF time is 0.83× AMD vs KahipND's 0.99×. After
/// deletion these matrices fall through to `pick_default_method`'s
/// `n ≤ 10_000 → Amf` rule. KahipND retains 20 strict wins
/// concentrated on high-avg-deg cases (STEENBRD, HADAMARD, TABLE8),
/// all sub-22k nnz_L absolute and reachable explicitly via
/// `OrderingMethod::KahipND` for callers who want them.
///
/// `pattern` is expected to be the matrix's full-symmetric pattern (the
/// shape produced by `CscMatrix::symmetric_pattern`); the
/// `pick_default_method` call below converts to a stored-nnz
/// equivalent assuming the diagonal is included.
fn choose_adaptive(pattern: &CscPattern, method: OrderingMethod) -> OrderingMethod {
    if method != OrderingMethod::Auto {
        return method;
    }
    let n = pattern.n;
    let full_nnz = pattern.row_idx.len();
    if n == 0 {
        return OrderingMethod::Amd;
    }
    let avg_deg = full_nnz as f64 / n as f64;
    if n > 100_000 && avg_deg < 5.0 {
        return OrderingMethod::Amd;
    }
    // Convert full-symmetric nnz back to a stored-lower-triangle
    // equivalent so `pick_default_method`'s thresholds (calibrated on
    // stored nnz) apply: stored = (full + n) / 2 when the diagonal is
    // included once on each row of the symmetric pattern.
    let stored_nnz = (full_nnz + n) / 2;
    let base = pick_default_method(n, stored_nnz);
    // Issue #64 arrow/bordered-KKT catch. The size-only
    // `pick_default_method` routes every `n > 10_000` matrix to MetisND,
    // but nested dissection cannot isolate a dense border (a handful of
    // very-high-degree columns concentrating the nonzeros) and the LDLᵀ
    // factor blows up ~7-9× vs AMF/AMD. Override MetisND → AMF on the
    // arrow signature. Only the would-be-MetisND decision is touched;
    // the `n <= 10_000 → AMF` and `n > 100_000 && avg_deg < 5 → AMD`
    // (returned above) paths are untouched. See
    // `dev/research/issue-64-arrow-bordered-ordering.md`.
    if base == OrderingMethod::MetisND && is_arrow_bordered(pattern) {
        return OrderingMethod::Amf;
    }
    // Issue #67 + #73 thin-large catch. The size-only `pick_default_method`
    // routes every `n > 10_000` matrix to MetisND, but corpus A/Bs on real
    // factor+solve wall-time (not nnz_L alone) show AMF wins or ties MetisND
    // across the whole would-be-MetisND population:
    //   - #67: 36/36 in-scope `(10_000, 100_000]` families, worst case 0.99×
    //     (noise), median ~1.5×, up to 4.5×.
    //   - #73: the `n > 100_000 && avg_deg >= 5` non-arrow families - dtoc2
    //     2.49×, pinene 1.18×, cont5_1_l 2.75×, nql180 2.05×, YATP1NE 2.13× -
    //     AMF wins factor+solve on every measured matrix. Critically nql180 is
    //     the lone case where MetisND has *smaller* symbolic fill (nnz_L 0.98×)
    //     yet AMF is still 2.05× faster on real factor+solve, so fill (nnz_L /
    //     flop_proxy) is NOT a reliable speed predictor and a fill-guarded race
    //     would wrongly demote nql180. The simple unconditional reroute is the
    //     one the evidence supports - see `dev/research/issue-73-n100k-thin-
    //     regime.md` and `dev/research/issue-67-thin-large-ordering.md`.
    //
    // MetisND's separators do not pay off on these uniformly-thin discretization
    // patterns, and its symbolic ordering is 2-5× more expensive than AMF's, so
    // racing the two is a net loss. Route every would-be-MetisND decision to AMF
    // outright. Only the would-be-MetisND decision is touched; the earlier
    // `n > 100_000 && avg_deg < 5 → Amd` (#50 powerflow) and arrow → AMF (#64)
    // catches fire first and are untouched.
    if base == OrderingMethod::MetisND {
        return OrderingMethod::Amf;
    }
    base
}

/// Detect the **arrow / bordered-KKT** sparsity signature on a full
/// symmetric pattern: a *small set* of very-high-degree "border" columns
/// carrying a *large share* of the nonzeros, over an otherwise thin body.
///
/// This is the structural fingerprint of an IPM augmented system whose
/// inequality block has a few dense constraint rows (issue #64: r05's
/// iter-0 KKT has 171 of 14 842 columns at degree 502, carrying 38.5% of
/// the nonzeros). On such patterns nested dissection smears the dense
/// border across its separators and the factor blows up, whereas
/// minimum-degree / min-fill orderings (AMD/AMF) defer the border to the
/// end of the elimination where it costs one dense trailing block.
///
/// Predicate (all O(n), allocation-free), on the full symmetric pattern:
///
/// ```text
/// avg_deg   = full_nnz / n
/// heavy_thr = max(HEAVY_DEG_FLOOR, HEAVY_AVG_MULT * avg_deg)
/// heavy     = { columns with degree > heavy_thr }
/// arrow iff  1 <= heavy.count < ARROW_COUNT_FRAC * n   (a *small* set)
///        AND heavy.nnz >= ARROW_NNZ_SHARE * full_nnz   (a *large* share)
/// ```
///
/// The `ARROW_NNZ_SHARE` guard is the discriminating test: it fires on
/// r05 (38.5% share) and rejects bcsstk38 (0.3% share, despite two
/// degree-614 columns). The `ARROW_COUNT_FRAC` guard rejects "many hub"
/// patterns where a large fraction of columns are high-degree (the matrix
/// is then just dense and nested dissection is appropriate). Uniformly
/// thin matrices (PoissonControl, powerflow22, bratu3d, cont-201) have no
/// column above `heavy_thr` and are never flagged. Calibration and the
/// false-positive table are in
/// `dev/research/issue-64-arrow-bordered-ordering.md`.
fn is_arrow_bordered(pattern: &CscPattern) -> bool {
    /// A "heavy" column has degree above this absolute floor regardless
    /// of `avg_deg`, so genuinely dense small matrices (high uniform
    /// degree) are not flagged.
    const HEAVY_DEG_FLOOR: usize = 64;
    /// ...or above this multiple of the average degree.
    const HEAVY_AVG_MULT: f64 = 8.0;
    /// The heavy set must be a *handful* of columns: strictly fewer than
    /// this fraction of `n`.
    const ARROW_COUNT_FRAC: f64 = 0.05;
    /// ...that *concentrate* at least this fraction of the nonzeros.
    const ARROW_NNZ_SHARE: f64 = 0.20;

    let n = pattern.n;
    if n == 0 {
        return false;
    }
    let full_nnz = pattern.row_idx.len();
    if full_nnz == 0 {
        return false;
    }
    let avg_deg = full_nnz as f64 / n as f64;
    let heavy_thr = (HEAVY_AVG_MULT * avg_deg).ceil() as usize;
    let heavy_thr = heavy_thr.max(HEAVY_DEG_FLOOR);

    let mut heavy_count = 0usize;
    let mut heavy_nnz = 0usize;
    for j in 0..n {
        let deg = pattern.col_ptr[j + 1] - pattern.col_ptr[j];
        if deg > heavy_thr {
            heavy_count += 1;
            heavy_nnz += deg;
        }
    }

    if heavy_count == 0 {
        return false;
    }
    let count_ok = (heavy_count as f64) < ARROW_COUNT_FRAC * n as f64;
    let share_ok = (heavy_nnz as f64) >= ARROW_NNZ_SHARE * full_nnz as f64;
    count_ok && share_ok
}

/// The complete output of symbolic factorization.
///
/// Produced before any numeric work begins. Contains everything needed
/// to allocate memory and drive the numeric factorization.
#[derive(Debug)]
pub struct SymbolicFactorization {
    /// Matrix dimension.
    pub n: usize,

    /// Fill-reducing permutation (new-to-old mapping).
    /// Column `perm[k]` of the original matrix becomes column k.
    pub perm: Vec<usize>,

    /// Inverse permutation (old-to-new mapping).
    pub perm_inv: Vec<usize>,

    /// Supernodes in postorder (children before parents).
    pub supernodes: Vec<Supernode>,

    /// Estimated total NNZ in the L factor across all supernodes.
    pub factor_nnz_estimate: usize,

    /// Slack factor applied to factor_nnz_estimate. Default 1.2.
    pub factor_slack: f64,

    /// For each supernode: the size (in f64s) of its contribution block.
    pub contrib_sizes: Vec<usize>,

    /// Peak contribution pool depth (sum of all live contribution blocks
    /// at the deepest point of the tree traversal).
    pub peak_contrib_bytes: usize,

    /// Elimination tree of the permuted matrix.
    pub etree: EliminationTree,

    /// Full symmetric pattern of the permuted matrix.
    pub permuted_pattern: CscPattern,

    /// Column counts of L.
    pub col_counts: Vec<usize>,

    /// Phase 2.9 small-leaf-subtree groups (`dev/plans/phase-2.9-
    /// small-leaf-subtree.md`). Populated unconditionally at
    /// symbolic time; used at numeric time only when
    /// `NumericParams::small_leaf == SmallLeafBatch::On`.
    pub small_leaf_groups: Vec<SmallLeafGroup>,

    /// For each supernode index, `Some(g)` if the supernode is a
    /// member of `small_leaf_groups[g]`, else `None`. Length
    /// `supernodes.len()`.
    pub snode_group: Vec<Option<usize>>,

    /// Cached MC64 matching produced by the `LdltCompress`
    /// preprocessor. When `Some`, the numeric phase reuses it to
    /// derive the `Mc64Symmetric` scaling vector in O(n) instead of
    /// rerunning the Hungarian kernel. `None` when no MC64 matching
    /// was computed during symbolic factorization. (Consumed by the
    /// numeric path again once MC64 scaling is ported to the generic
    /// solver - see the rslab feature port.)
    #[allow(dead_code)]
    pub(crate) cached_mc64: Option<crate::scaling::Mc64Cache>,

    /// Concrete ordering method actually dispatched. Records the
    /// `OrderingMethod::Auto → AMD/MetisND/ScotchND/KahipND`
    /// resolution made by `choose_adaptive`. For non-`Auto` callers
    /// this is identical to the requested method.
    pub resolved_method: OrderingMethod,
    /// Concrete amalgamation strategy actually used.
    /// `AmalgamationStrategy::Auto` is resolved by
    /// `pick_amalgamation_strategy` before supernode detection; this
    /// field records the resolved value.
    pub resolved_amalgamation: supernode::AmalgamationStrategy,
    /// Concrete ordering preprocessor actually used.
    /// `OrderingPreprocess::Auto` is resolved by
    /// `pick_ordering_preprocess`; this field records `None` or
    /// `LdltCompress` after that dispatch.
    pub resolved_preprocess: supernode::OrderingPreprocess,

    /// F3.2: When this factorization was produced by
    /// [`symbolic_factorize_with_schur`], records the size of the Schur
    /// tail. The last `n_schur` columns of `perm` correspond to the
    /// user-supplied `schur_indices` in the supplied order. `None` for
    /// factorizations produced by [`symbolic_factorize`] or
    /// [`symbolic_factorize_with_method`]. The numeric phase reads this
    /// to enforce the per-front NPIV ≤ NASS − NVSCHUR stopping rule
    /// (F3.2b).
    pub is_schur_tail: Option<usize>,
}

/// Size-only base ordering rule from cheap matrix dimensions (no pattern
/// walk). Narrow on purpose - see comment on `Auto` for why a broad
/// dispatcher regressed the IPM bench. `choose_adaptive` calls this for
/// the bulk of patterns, then layers the pattern-aware catches on top
/// (very-large-and-sparse → AMD; arrow/bordered → AMF, issue #64).
///
/// Current rule (mirrors MUMPS's `ana_set_ordering.F` AMF-vs-METIS
/// heuristic):
///   - `n == 0`                                        → `Amd`
///     (avoids /0 and external-crate weirdness on the empty pattern)
///   - `n <= 10_000`                                   → `Amf`
///     (MUMPS-style "small symmetric" rule: HAMF4 fill metric is
///     within 1.10× of MUMPS HAMF4 on 183_277 of 183_293 sidecar'd
///     matrices in `tests/amf_corpus_oracle.rs`, and the in-tree
///     audit (`diag_amf_vs_amd`) shows AMF strictly better than AMD
///     on 83/782 matrices, tied on 589, AMD better on 110, geomean
///     ratio 1.003. ORBIT2_0000 alone goes from AMD's 1.4M nnz_L
///     down to AMF's 32_105.)
///   - everything else (`n > 10_000`)                  → `MetisND`
///     (large patterns where nested dissection is the standard win.)
///
/// `nnz` here is the matrix's *stored* nnz (lower triangle for
/// symmetric matrices), not the symmetric pattern's.
///
/// Issue #50 (2026-05-23) deleted two prior escape hatches:
///   - `n >= 5000 && nnz/n < 6 → MetisND` (bordered-KKT catch, CRESC132);
///   - `n >= 2000 && nnz/n < 4 → MetisND` (chain-pattern catch,
///     CHAINWOO/HYDROELL/DIXMAANH/VESUVIO).
///
/// Both were calibrated on 2026-04-27 against a Bunch-Kaufman
/// pivoting cascade that fattened the AMD-ordered factor by up to
/// 7.5× on CHAINWOO_0000 and produced a near-dense root frontal on
/// CRESC132_0000. Issue #46's fixes (`42434a5` fine-grained delayed
/// pivoting, `070840b` two-tier 2×2 partner selection) eliminated
/// the amplifier in May 2026: CHAINWOO_0000 now produces 22.9k
/// num_nnz_l with AMD vs the 2.10M it produced before, and the
/// numeric inventory in `dev/research/issue-50-numeric-inventory.csv`
/// shows zero of 250 chain-catch-class corpus matrices have
/// AMD/MetisND num_nnz_l ratio ≥ 1.5×. The catches now route 113-s
/// nested-dissection symbolic on `powerflow22` (n=2.8 M, stored
/// avg_deg ≈ 2.4) where AMD does the same job in 55 s with smaller
/// fill. See `dev/research/issue-50-metisnd-symbolic-cost.md` §F7-F8.
fn pick_default_method(n: usize, _stored_nnz: usize) -> OrderingMethod {
    if n == 0 {
        return OrderingMethod::Amd;
    }
    if n <= 10_000 {
        OrderingMethod::Amf
    } else {
        OrderingMethod::MetisND
    }
}

/// Resolve [`OrderingPreprocess::Auto`] to a concrete preprocessor
/// choice based on cheap O(nnz) shape predicates.
///
/// Returns [`OrderingPreprocess::LdltCompress`] when two conditions hold:
///
/// 1. `n >= MIN_N_FOR_COMPRESSION` (size floor). Below this, numeric
///    factor time is in the sub-ms range and the ~100-400μs compression
///    symbolic overhead dominates. Calibrated from the 154 588-matrix
///    bench: geomean regressed 0.36 → 0.48 with unconditional
///    compression, driven by small-matrix symbolic overhead.
///
/// 2. `low_degree_cols / n >= LOW_DEGREE_THRESHOLD` (arrow-KKT
///    signature). Columns with stored degree ≤ 2 (the diagonal plus at
///    most one off-diagonal) are the structural fingerprint of IPM KKT
///    slack blocks (`IpStdAugSystemSolver.cpp:250-305`: `Σ_s + δ_s I`
///    coupled to the d-row by a single identity off-diagonal). Many
///    such columns means the MC64 matching has abundant 2-cycle
///    structure for compression to exploit. This broadens the
///    `diag_only / n` predicate from `pick_scaling_strategy` because
///    Ipopt slack columns are degree-2, not degree-1.
///
/// Otherwise returns [`OrderingPreprocess::None`].
///
/// Parallels `crate::scaling::pick_scaling_strategy` in spirit.
/// Both predicates are O(nnz) and allocation-free.
///
/// No published compression-benefit predictor exists in the MUMPS /
/// SPRAL literature (see consult of 2026-04-23). These thresholds are
/// calibrated against the rslab corpus and documented in
/// `dev/journal/2026-04-23-02.org`.
pub fn pick_ordering_preprocess(matrix: &CscMatrix) -> OrderingPreprocess {
    const MIN_N_FOR_COMPRESSION: usize = 128;
    const LOW_DEGREE_THRESHOLD: f64 = 0.30;

    let n = matrix.n;
    if n < MIN_N_FOR_COMPRESSION {
        return OrderingPreprocess::None;
    }

    let mut low_degree = 0usize;
    for j in 0..n {
        let nnz_col = matrix.col_ptr[j + 1] - matrix.col_ptr[j];
        if nnz_col <= 2 {
            low_degree += 1;
        }
    }

    if low_degree as f64 / n as f64 >= LOW_DEGREE_THRESHOLD {
        OrderingPreprocess::LdltCompress
    } else {
        OrderingPreprocess::None
    }
}

/// Perform symbolic factorization of a sparse symmetric matrix.
///
/// Picks the fill-reducing ordering adaptively via [`OrderingMethod::Auto`]
/// (resolved by `choose_adaptive`): AMF for n ≤ 10_000 or arrow/bordered
/// patterns, AMD for very-large-and-sparse, MetisND otherwise. Routing
/// through `Auto` keeps this no-arg default and the explicit `Auto` caller
/// in exact agreement (issue #64). Callers who want a specific ordering
/// with no dispatcher should call `symbolic_factorize_with_method` with an
/// explicit `OrderingMethod`.
///
/// Steps:
/// 1. Pick fill-reducing ordering (resolved from `Auto` by `choose_adaptive`)
/// 2. Build elimination tree of the permuted matrix
/// 3. Compute column counts (fill prediction)
/// 4. Detect and amalgamate supernodes
/// 5. Compute MemoryPlan (factor NNZ, contribution sizes, peak memory)
pub fn symbolic_factorize(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
) -> Result<SymbolicFactorization, RslabError> {
    symbolic_factorize_with_method(matrix, snode_params, OrderingMethod::Auto)
}

/// Convert an owned-`usize` `CscPattern` into the contract's borrowed-`i32`
/// shape used by `rslab-metis` / `rslab-scotch`. Returns buffers the
/// caller must keep alive for the lifetime of the produced `CscPattern<'_>`.
fn to_contract_pattern_bufs(pattern: &CscPattern) -> Result<(Vec<i32>, Vec<i32>), RslabError> {
    let col_ptr: Result<Vec<i32>, _> = pattern.col_ptr.iter().map(|&x| i32::try_from(x)).collect();
    let col_ptr = col_ptr.map_err(|_| {
        RslabError::InvalidInput("matrix too large for i32-indexed ordering crates".to_string())
    })?;
    let row_idx: Result<Vec<i32>, _> = pattern.row_idx.iter().map(|&x| i32::try_from(x)).collect();
    let row_idx = row_idx.map_err(|_| {
        RslabError::InvalidInput("matrix too large for i32-indexed ordering crates".to_string())
    })?;
    Ok((col_ptr, row_idx))
}

/// Run an external (contract-conforming) ordering crate on `pattern` and
/// return the permutation as `Vec<usize>` in the in-tree convention
/// (new-to-old: `perm[k]` is the original column that became column `k`),
/// along with the concrete `OrderingMethod` actually dispatched (matters
/// when `method == Auto` is resolved adaptively).
fn run_external_ordering(
    pattern: &CscPattern,
    method: OrderingMethod,
) -> Result<(Vec<usize>, OrderingMethod), RslabError> {
    let (col_buf, row_buf) = to_contract_pattern_bufs(pattern)?;
    let pat = rslab_ordering_core::CscPattern::new(pattern.n, &col_buf, &row_buf)
        .ok_or_else(|| RslabError::InvalidInput("malformed CSC pattern".to_string()))?;
    // `method` is expected to be concrete here - `Auto` is resolved
    // upstream by `symbolic_factorize_with_method` against the
    // original matrix's pattern, before any preprocessing.
    debug_assert_ne!(method, OrderingMethod::Auto);
    // `actual` diverges from `method` only when ScotchND silently
    // falls back to amd_leaf for every recursion (issue #3): the
    // returned permutation is bit-identical to AMD's, so the
    // `resolved_method` field must report Amd, not ScotchND.
    let mut actual = method;
    let perm_i32 = match method {
        OrderingMethod::Amd => rslab_amd::amd_order(&pat),
        OrderingMethod::Amf => rslab_amf::amf_order(&pat),
        OrderingMethod::MetisND => rslab_metis::metis_order(&pat),
        OrderingMethod::ScotchND => {
            let opts = rslab_scotch::ScotchOptions::default();
            rslab_scotch::scotch_order_full(&pat, &opts).map(|(perm, _, sstats)| {
                if sstats.n_separator_vertices == 0 {
                    actual = OrderingMethod::Amd;
                }
                perm
            })
        }
        OrderingMethod::KahipND => rslab_kahip::kahip_order(&pat),
        OrderingMethod::Rcm => rslab_ordering_core::rcm_order(&pat),
        OrderingMethod::Auto => {
            unreachable!("Auto is resolved by symbolic_factorize_with_method")
        }
        OrderingMethod::AutoRace => {
            unreachable!("AutoRace is resolved by symbolic_factorize_with_method")
        }
    };
    let perm_i32 = perm_i32
        .map_err(|e| RslabError::InvalidInput(format!("external ordering failed: {}", e)))?;
    if perm_i32.len() != pattern.n {
        return Err(RslabError::InvalidInput(format!(
            "external ordering returned {} entries for n={}",
            perm_i32.len(),
            pattern.n
        )));
    }
    let mut out: Vec<usize> = Vec::with_capacity(perm_i32.len());
    for x in perm_i32 {
        let u = usize::try_from(x).map_err(|_| {
            RslabError::InvalidInput("external ordering returned negative index".to_string())
        })?;
        if u >= pattern.n {
            return Err(RslabError::InvalidInput(
                "external ordering returned out-of-range index".to_string(),
            ));
        }
        out.push(u);
    }
    Ok((out, actual))
}

/// Concrete candidates raced by [`OrderingMethod::AutoRace`]. See the
/// variant docstring for rationale.
const RACE_CANDIDATES: &[OrderingMethod] = &[
    OrderingMethod::Amd,
    OrderingMethod::MetisND,
    OrderingMethod::ScotchND,
    OrderingMethod::KahipND,
    // Cheap band/profile reducer: wins on banded / structured patterns where the
    // dissection candidates over-separate. Selected only if its factor_nnz is the
    // smallest, so it never hurts the race outcome.
    OrderingMethod::Rcm,
];

/// Race the [`RACE_CANDIDATES`] orderings at symbolic time and return the
/// `SymbolicFactorization` with the smallest `factor_nnz_estimate`.
///
/// Implements the [`OrderingMethod::AutoRace`] dispatcher. Candidates
/// that error out (e.g. external crate failure) are skipped; the race
/// succeeds as long as at least one candidate produces a valid result.
/// Returns an error only if every candidate fails.
fn symbolic_factorize_race(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
) -> Result<SymbolicFactorization, RslabError> {
    let mut best: Option<SymbolicFactorization> = None;
    // S7: when a symbolic profiler is attached, give each candidate its
    // own fresh profiler instead of letting all RACE_CANDIDATES append
    // into the caller's shared one. Sharing accumulated one full stage
    // list per candidate (~4x) against a single candidate's `total_us`,
    // tripping the "stage sum exceeds total" warning and inflating every
    // pct_of_total. We keep the winning candidate's profiler and copy it
    // into the caller's shared profiler at the end, so the report
    // reflects exactly one ordering run.
    let mut best_prof: Option<SymbolicProfiler> = None;
    let mut last_err: Option<RslabError> = None;
    for &cand in RACE_CANDIDATES {
        let cand_prof = snode_params
            .symbolic_profiler
            .as_ref()
            .map(|_| std::sync::Arc::new(std::sync::Mutex::new(SymbolicProfiler::new())));
        let cand_params = SupernodeParams {
            symbolic_profiler: cand_prof.clone(),
            ..snode_params.clone()
        };
        match symbolic_factorize_with_method(matrix, &cand_params, cand) {
            Ok(sym) => {
                let is_better = best
                    .as_ref()
                    .map(|b| sym.factor_nnz_estimate < b.factor_nnz_estimate)
                    .unwrap_or(true);
                if is_better {
                    best = Some(sym);
                    best_prof = cand_prof.and_then(|a| a.lock().ok().map(|g| g.clone()));
                }
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }
    // Copy the winning candidate's stage timings into the caller's shared
    // profiler so `report()` reflects one run, not all four concatenated.
    if let (Some(shared), Some(winner)) = (snode_params.symbolic_profiler.as_ref(), best_prof) {
        if let Ok(mut p) = shared.lock() {
            *p = winner;
        }
    }
    best.ok_or_else(|| {
        last_err.unwrap_or_else(|| {
            RslabError::InvalidInput("AutoRace: no candidates available".to_string())
        })
    })
}

/// Like [`symbolic_factorize`] but lets the caller pick the
/// fill-reducing ordering via [`OrderingMethod`].
///
/// `symbolic_factorize(m, p) == symbolic_factorize_with_method(m, p,
/// OrderingMethod::Amd)`.
pub fn symbolic_factorize_with_method(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
    method: OrderingMethod,
) -> Result<SymbolicFactorization, RslabError> {
    // AutoRace is resolved here by running each concrete candidate
    // through this same function and picking the smallest
    // `factor_nnz_estimate`. The recursive call passes a concrete
    // `OrderingMethod`, so there is no infinite loop.
    if method == OrderingMethod::AutoRace {
        return symbolic_factorize_race(matrix, snode_params);
    }
    let n = matrix.n;

    // Phase 2.13b per-stage profiler. Every timer is `Some` only when
    // `snode_params.symbolic_profiler.is_some()`; the `None` path
    // does no `Instant::now()` calls. See
    // `dev/research/phase-2.13b-symbolic-profiler.md`.
    let prof = snode_params.symbolic_profiler.as_ref();
    let t_total = prof.map(|_| std::time::Instant::now());

    // β refactor: scaling is no longer computed here. It moved to
    // `factorize_multifrontal` so that `SymbolicFactorization`
    // depends only on the matrix pattern (not its values) and can
    // be reused across multiple numeric factorizations of
    // structurally identical KKTs. See
    // `dev/plans/scaling-in-numeric.md`.

    // Step 1: Fill-reducing ordering. Dispatch on `method`. The
    // downstream pipeline (postorder composition, etree, column counts,
    // supernode amalgamation, memory plan) is identical regardless of
    // which ordering produced `initial_perm`.
    //
    // If `snode_params.preprocess == LdltCompress`, run MC64 symmetric
    // matching, build the super-variable map, order the compressed
    // graph, and expand the resulting super-permutation back to
    // length `n` before handing it to the rest of the pipeline. See
    // `src/symbolic/ldlt_compress.rs` and
    // `dev/plans/phase-2.6.5-ldlt-compressed-graph.md`.
    let t_sym = prof.map(|_| std::time::Instant::now());
    let full_pattern = matrix.symmetric_pattern();
    if let Some(t) = t_sym {
        record_stage(prof, "symmetric_pattern", t);
    }

    // Resolve `OrderingMethod::Auto` against the original matrix's
    // pattern *before* preprocessing. If we resolved against the
    // compressed pattern below, Auto would see a different `n` /
    // `avg_deg` and reach a different conclusion than
    // `symbolic_factorize` (which uses `pick_default_method` on the
    // matrix directly). Issue #3.
    let method = choose_adaptive(&full_pattern, method);

    let mut cached_mc64: Option<crate::scaling::Mc64Cache> = None;
    // Resolve `Auto` to `None` or `LdltCompress` before entering the
    // dispatch. Keeps the match below exhaustive on the two concrete
    // variants and keeps the dispatcher logic in one testable place.
    let t_pick = prof.map(|_| std::time::Instant::now());
    let resolved_preprocess = match snode_params.preprocess {
        OrderingPreprocess::Auto => pick_ordering_preprocess(matrix),
        other => other,
    };
    if let Some(t) = t_pick {
        record_stage(prof, "pick_preprocess", t);
    }
    // The fill-reducing ordering and (when enabled) the LdltCompress
    // preprocessor are timed under *separate* stages. The preprocessor's
    // MC64 matching can dwarf the ordering itself - on the pf22 powerflow
    // KKT (n=2.8M) MC64 is ~53s while `rslab_amd::amd_order` is ~0.3s - so
    // folding both into one "ordering" stage mis-attributes the cost and
    // led to the wrong diagnosis in issue #80. `record_ordering` wraps the
    // actual `run_external_ordering` call so every path records exactly one
    // `ordering` stage.
    let record_ordering = |pat: &CscPattern| -> Result<(Vec<usize>, OrderingMethod), RslabError> {
        let t_ord = prof.map(|_| std::time::Instant::now());
        let r = run_external_ordering(pat, method)?;
        if let Some(t) = t_ord {
            record_stage(prof, "ordering", t);
        }
        Ok(r)
    };
    let (amd_perm, resolved_method): (Vec<usize>, OrderingMethod) = match resolved_preprocess {
        OrderingPreprocess::None => record_ordering(&full_pattern)?,
        OrderingPreprocess::Auto => unreachable!("resolved above"),
        OrderingPreprocess::LdltCompress => {
            // Run the full MC64 pipeline once and keep the cache so the
            // numeric phase can reuse it for `Mc64Symmetric` scaling
            // (Phase 2.4.4: eliminates ~70% of compression symbolic
            // overhead on matrices where scaling also runs MC64). MC64 is
            // the expensive part - record it under its own `ldlt_compress`
            // stage (issue #80).
            let t_pre = prof.map(|_| std::time::Instant::now());
            let cache = crate::scaling::compute_mc64_cache(matrix)?;
            let map = build_supermap(&cache.perm);
            if let Some(t) = t_pre {
                record_stage(prof, "ldlt_compress", t);
            }
            let pair = if map.ncmp() == n {
                // Matching gives no compression leverage; fall through
                // to the uncompressed path rather than build and walk
                // an identical-size graph.
                record_ordering(&full_pattern)?
            } else {
                let t_cmp = prof.map(|_| std::time::Instant::now());
                let cpat = compress_pattern(&full_pattern, &map);
                if let Some(t) = t_cmp {
                    record_stage(prof, "compress_pattern", t);
                }
                let (super_perm, resolved) = record_ordering(&cpat)?;
                let t_exp = prof.map(|_| std::time::Instant::now());
                let expanded = expand_permutation(&super_perm, &map);
                if let Some(t) = t_exp {
                    record_stage(prof, "expand_perm", t);
                }
                (expanded, resolved)
            };
            cached_mc64 = Some(cache);
            pair
        }
    };

    // Step 2: Build the etree on the permuted pattern. This etree is
    // intermediate - we use it to compute the postorder and then discard it.
    // The local name `amd_*` is kept from the AMD-only era to minimise the
    // diff; semantically these are now "ordering output" and "permuted
    // pattern from that ordering", regardless of method.
    let t_perm1 = prof.map(|_| std::time::Instant::now());
    let amd_pattern = permute_pattern(&full_pattern, &amd_perm);
    if let Some(t) = t_perm1 {
        record_stage(prof, "permute1", t);
    }
    let t_etree0 = prof.map(|_| std::time::Instant::now());
    let amd_etree = EliminationTree::from_pattern(&amd_pattern);
    if let Some(t) = t_etree0 {
        record_stage(prof, "etree_initial", t);
    }

    // Step 3: Postorder the etree (CHOLMOD-style composition).
    // Without this step, supernode amalgamation merges columns whose indices
    // are not consecutive in the column numbering, and downstream code that
    // assumes `first_col..first_col+ncol` is the eliminated set silently
    // factors the wrong columns. See dev/research/postorder-pipeline.md.
    let t_post = prof.map(|_| std::time::Instant::now());
    let (post, post_inv) = postorder(&amd_etree);
    if let Some(t) = t_post {
        record_stage(prof, "postorder", t);
    }

    // Step 4: Compose AMD perm with the postorder.
    //   final_perm[k] = amd_perm[post[k]]
    // The composition maps postorder position k to the original column.
    let t_compose = prof.map(|_| std::time::Instant::now());
    let perm: Vec<usize> = post.iter().map(|&p| amd_perm[p]).collect();
    let mut perm_inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }
    if let Some(t) = t_compose {
        record_stage(prof, "perm_compose", t);
    }

    // Step 5: Re-permute the matrix on the composed permutation.
    let t_perm2 = prof.map(|_| std::time::Instant::now());
    let permuted_pattern = permute_pattern(&full_pattern, &perm);
    if let Some(t) = t_perm2 {
        record_stage(prof, "permute2", t);
    }

    // Step 5b: Build the final elimination tree by renumbering `amd_etree`
    // through the postorder. Postorder is a topological relabeling of the
    // elimination tree nodes, so `etree(P·A·Pᵀ) = post-renumbering of
    // etree(A)` when P is a postorder of etree(A) - the tree structure is
    // preserved and only the node labels change. This lets us produce the
    // final etree in O(n) instead of re-running `from_pattern` at
    // O(nnz · α(n)). A 3-run bench shows ~3% small-frontal p90 improvement
    // over the old two-from_pattern approach.
    let t_relabel = prof.map(|_| std::time::Instant::now());
    let final_parent: Vec<Option<usize>> = (0..n)
        .map(|new| {
            let old_amd = post[new];
            amd_etree.parent[old_amd].map(|old_par| post_inv[old_par])
        })
        .collect();
    let etree = EliminationTree {
        parent: final_parent,
        n,
    };
    if let Some(t) = t_relabel {
        record_stage(prof, "etree_relabel", t);
    }

    // Step 6: Column counts on the final pattern + etree.
    // Phase 2.5.1 switched this from the O(n²) elimination simulation
    // (still available as `column_counts`) to Gilbert-Ng-Peyton at
    // O(nnz(A) + n·α(n)). Bit-exact equivalence verified on 169585
    // KKT matrices - see `dev/validation/phase-2.5.1-*`.
    let t_cc = prof.map(|_| std::time::Instant::now());
    let mut col_counts = column_counts_gnp(&permuted_pattern, &etree);
    if let Some(t) = t_cc {
        record_stage(prof, "col_counts", t);
    }

    // Phase 2.12: optional SSIDS-style merge-biased postorder.
    // Predict desired merges using only the etree + column counts,
    // then re-postorder the etree so desired-merge children are
    // emitted adjacent to their parents. The downstream
    // `find_supernodes` adjacency check then succeeds for those
    // merges naturally.
    //
    // Rebuild path: compose perm with the bias-driven post2,
    // re-permute the matrix, rebuild etree and col_counts. The
    // structural properties are invariant under within-subtree
    // relabeling (CHOLMOD/SSIDS observation, see
    // `dev/research/phase-2.12-column-renumbering.md` §5.1).
    //
    // Fast-path: when no bias is requested (no desired merges, OR
    // the strategy is `Adjacency`), the second pass is skipped and
    // the pipeline behaves identically to pre-Phase-2.12.
    let mut permuted_pattern = permuted_pattern;
    let mut perm = perm;
    let mut etree = etree;

    // Phase 2.13a: resolve `Auto` to a concrete strategy via a cheap
    // O(n) etree shape predicate. The downstream Renumber gate and
    // `find_supernodes` reverse-iteration check need a concrete
    // variant - `Auto` is a top-level dispatch sentinel only.
    let mut effective_params = snode_params.clone();
    if matches!(
        effective_params.amalgamation_strategy,
        supernode::AmalgamationStrategy::Auto
    ) {
        effective_params.amalgamation_strategy = supernode::pick_amalgamation_strategy(&etree);
    }
    let snode_params: &SupernodeParams = &effective_params;

    let t_renumber = prof.map(|_| std::time::Instant::now());
    if matches!(
        snode_params.amalgamation_strategy,
        supernode::AmalgamationStrategy::Renumber
    ) {
        let bias = supernode::predict_merges(&etree, &col_counts, snode_params);
        if bias.iter().any(|&b| b) {
            let (post2, _post2_inv) = biased_postorder(&etree, &bias);
            // Compose: perm₂[k] = perm[post2[k]]; the existing
            // `perm` already encodes AMD ∘ post1.
            let new_perm: Vec<usize> = post2.iter().map(|&p| perm[p]).collect();
            let mut new_perm_inv = vec![0usize; n];
            for (new, &old) in new_perm.iter().enumerate() {
                new_perm_inv[old] = new;
            }
            let new_permuted_pattern = permute_pattern(&full_pattern, &new_perm);
            // Rebuild the etree on the renumbered pattern. We could
            // relabel the existing etree through post2 in O(n) (as
            // Step 5b does for the postorder), but since the
            // permutation invariant is critical and post2 is a
            // postorder of `etree`, the relabeled tree is equivalent
            // by construction. Re-derive from scratch as a defense
            // against the etree-invariance claim being subtly wrong;
            // O(nnz · α(n)) is small for the matrices we target.
            let new_etree = EliminationTree::from_pattern(&new_permuted_pattern);
            let new_col_counts = column_counts_gnp(&new_permuted_pattern, &new_etree);

            perm = new_perm;
            perm_inv = new_perm_inv;
            permuted_pattern = new_permuted_pattern;
            etree = new_etree;
            col_counts = new_col_counts;
        }
    }
    if let Some(t) = t_renumber {
        record_stage(prof, "renumber", t);
    }
    let factor_nnz = total_factor_nnz(&col_counts);

    // Step 7: Supernode detection on the postordered etree
    let t_find = prof.map(|_| std::time::Instant::now());
    let mut supernodes = find_supernodes(&etree, &col_counts, snode_params);
    // Issue #55 Phase B2: assign per-supernode incoming-delay budget.
    // Bounded-cost postorder pass; runs once per symbolic factor and
    // is cached in `SymbolicFactorization` for reuse across numeric
    // refactors. No effect until the numeric-time enforcement (B3)
    // and CB-rewire (B5) check `Supernode::delayed_capacity`.
    supernode::assign_delayed_capacities(&mut supernodes);
    if let Some(t) = t_find {
        record_stage(prof, "find_supernodes", t);
    }

    // Step 7b: Phase 2.9 small-leaf grouping. Runs unconditionally;
    // the groups are consumed at numeric time only when the
    // `small_leaf` gate is `On`. O(n_snodes), no allocations beyond
    // the groups themselves.
    let t_slg = prof.map(|_| std::time::Instant::now());
    let (small_leaf_groups, snode_group) =
        find_small_leaf_groups(&supernodes, &permuted_pattern, &snode_params.small_leaf);
    if let Some(t) = t_slg {
        record_stage(prof, "small_leaf_groups", t);
    }

    // Step 5: Compute contribution sizes and peak memory
    let t_pk = prof.map(|_| std::time::Instant::now());
    let contrib_sizes: Vec<usize> = supernodes.iter().map(|s| s.contrib_size()).collect();

    let peak_contrib_bytes = compute_peak_contrib(&supernodes, &contrib_sizes);
    if let Some(t) = t_pk {
        record_stage(prof, "peak_contrib", t);
    }

    let factor_slack = 1.2;

    if let (Some(arc), Some(t)) = (prof, t_total) {
        if let Ok(mut p) = arc.lock() {
            p.set_total(t.elapsed().as_micros() as u64);
        }
    }

    Ok(SymbolicFactorization {
        n,
        perm,
        perm_inv,
        supernodes,
        factor_nnz_estimate: (factor_nnz as f64 * factor_slack) as usize,
        factor_slack,
        contrib_sizes,
        peak_contrib_bytes,
        etree,
        permuted_pattern,
        col_counts,
        small_leaf_groups,
        snode_group,
        cached_mc64,
        resolved_method,
        resolved_amalgamation: snode_params.amalgamation_strategy,
        resolved_preprocess,
        is_schur_tail: None,
    })
}

/// Symbolic factorization with a user-supplied Schur tail (F3.2a).
///
/// Like [`symbolic_factorize_with_method`] except the last `n_schur`
/// columns of the produced permutation are pinned to `schur_indices` in
/// the supplied order - i.e. `perm[n - n_schur + i] == schur_indices[i]`
/// for every `i`. This is the symbolic side of the Schur-complement API
/// described in `dev/research/schur-complement.md` (F3.0).
///
/// The pipeline diverges from [`symbolic_factorize_with_method`] in
/// three places:
///
/// 1. **Ordering.** The fill-reducing ordering is fixed to AMD on the
///    non-Schur subgraph, via [`crate::ordering::schur::compute_schur_aware_perm`]
///    (F3.1). Other methods are not yet wired in for the Schur path
///    because each external ordering crate would need a "constrained
///    ordering" or subgraph hook to honour the Schur tail invariant.
///    See `dev/research/schur-complement.md` D3.
///
/// 2. **Postorder.** Standard CHOLMOD postorder is replaced by
///    [`crate::ordering::postorder::schur_constrained_postorder`], which
///    pins Schur nodes to their etree-index positions. The Schur subset
///    forms a top-forest of the etree (parent always strictly greater
///    than child, and Schur indices occupy `[n - n_schur, n)`), so the
///    constraint is satisfiable.
///
/// 3. **Preprocessor / amalgamation strategy.** The
///    [`OrderingPreprocess::LdltCompress`] preprocessor and the
///    [`AmalgamationStrategy::Renumber`] reorderer both rewrite the
///    column numbering and would break the tail invariant. The Schur
///    path forces `preprocess == None` and `amalgamation_strategy ==
///    Adjacency` regardless of what the caller passed in
///    `snode_params`.
///
/// Empty `schur_indices` ⇒ returns the same result as
/// [`symbolic_factorize_with_method`] with `OrderingMethod::Amd`.
///
/// `schur_indices.len() == n` ⇒ `InvalidInput` (the elimination set
/// would be empty; almost certainly an upstream logic bug).
///
/// Returns `is_schur_tail = Some(n_schur)` so the numeric phase (F3.2b)
/// can enforce the per-front `NPIV ≤ NASS − NVSCHUR` stopping rule.
pub fn symbolic_factorize_with_schur(
    matrix: &CscMatrix,
    snode_params: &SupernodeParams,
    schur_indices: &[usize],
) -> Result<SymbolicFactorization, RslabError> {
    let n = matrix.n;
    let n_schur = schur_indices.len();

    if n_schur == 0 {
        // Empty Schur ⇒ standard symbolic factorization with AMD.
        return symbolic_factorize_with_method(matrix, snode_params, OrderingMethod::Amd);
    }

    // Force the preprocessor and amalgamation strategy to values that
    // preserve the column numbering. LdltCompress rewrites columns via
    // the MC64 supermap; Renumber re-postorders. Both would break the
    // Schur tail invariant.
    let mut effective_params = snode_params.clone();
    effective_params.preprocess = OrderingPreprocess::None;
    effective_params.amalgamation_strategy = supernode::AmalgamationStrategy::Adjacency;

    // Step 1: Schur-aware ordering. AMD on the non-Schur subgraph,
    // followed by the Schur tail in user-supplied order. Validates
    // schur_indices (duplicates / out-of-range / full-n).
    let initial_perm = crate::ordering::schur::compute_schur_aware_perm(matrix, schur_indices)?;

    // Step 2: build full symmetric pattern + permute.
    let full_pattern = matrix.symmetric_pattern();
    let initial_permuted = permute_pattern(&full_pattern, &initial_perm);

    // Step 3: etree of permuted pattern. By construction Schur columns
    // sit at indices [n - n_schur, n); etree.parent[j] > j for every j,
    // so the Schur subset is closed under `parent` (top-forest).
    let initial_etree = EliminationTree::from_pattern(&initial_permuted);

    // Step 4: Schur-constrained postorder. Non-Schur descendants of
    // Schur nodes emit first (subtree-size order); Schur nodes emit at
    // their etree-index positions, preserving the user's input order.
    // Mark the highest n_schur indices in the etree as Schur. By
    // construction (compute_schur_aware_perm appends the Schur tail at
    // the end of initial_perm), these positions correspond to the user's
    // schur_indices in user-supplied order.
    let mut is_schur = vec![false; n];
    for slot in is_schur.iter_mut().skip(n - n_schur) {
        *slot = true;
    }
    let (post, post_inv) =
        crate::ordering::postorder::schur_constrained_postorder(&initial_etree, &is_schur);

    // Postorder identity check on the Schur tail (defensive - the
    // top-forest invariant should make this hold by construction).
    for (k, &p) in post.iter().enumerate().skip(n - n_schur) {
        debug_assert_eq!(
            p, k,
            "schur_constrained_postorder violated tail identity at k={}",
            k
        );
    }

    // Step 5: compose perm₀ with the postorder.
    let perm: Vec<usize> = post.iter().map(|&p| initial_perm[p]).collect();
    let mut perm_inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        perm_inv[old] = new;
    }

    // Tail-invariant assertion: this is the F3.2a contract.
    debug_assert_eq!(
        &perm[n - n_schur..],
        schur_indices,
        "Schur tail invariant violated"
    );

    // Step 6: re-permute and rebuild etree on the final pattern.
    let permuted_pattern = permute_pattern(&full_pattern, &perm);
    let final_parent: Vec<Option<usize>> = (0..n)
        .map(|new| {
            let old_initial = post[new];
            initial_etree.parent[old_initial].map(|old_par| post_inv[old_par])
        })
        .collect();
    let etree = EliminationTree {
        parent: final_parent,
        n,
    };

    // Step 7: column counts on the final pattern + etree.
    let col_counts = column_counts::column_counts_gnp(&permuted_pattern, &etree);
    let factor_nnz = column_counts::total_factor_nnz(&col_counts);

    // Step 8: supernode detection. Adjacency strategy only - Renumber
    // would re-postorder and break the tail invariant.
    let mut supernodes = supernode::find_supernodes(&etree, &col_counts, &effective_params);

    // Step 8b (F3.2b multi-supernode tail): force-merge any Schur-bearing
    // supernodes into a single tail supernode. This mirrors MUMPS's
    // HALO-SCHUR amalgamation (`PE[schur[i]] = -schur[0]`,
    // `ana_orderings.F:9187-9220`), where all Schur variables collapse
    // into one supervariable so the numeric stopping rule lives in one
    // place. rslab's adjacency-only amalgamation (size_based with
    // nemin=32, trivial_chain) does not always merge the Schur tail -
    // when the Schur block is large or the row patterns of constituent
    // Schur cols differ enough, multiple supernodes carry Schur cols,
    // and the F3.2b numeric driver would reject. The merge here keeps
    // the design contract from `dev/research/schur-complement.md` D4
    // ("the only front with nvschur > 0 is the root") satisfied without
    // requiring a multi-supernode numeric path.
    //
    // Safety: the F3.2a postorder pins Schur cols to `[n - n_schur, n)`
    // and the supernode column ranges are contiguous in this numbering,
    // so the merged supernode covers the contiguous range
    // `[n - n_schur, n)` - preserving the find_supernodes contiguity
    // invariant downstream code relies on.
    merge_schur_tail_supernodes(&mut supernodes, n, n_schur)?;

    // Issue #55 Phase B2: assign per-supernode incoming-delay budget.
    // Runs after the Schur-tail merge so the surviving root supernode
    // gets a single budget computed from its post-merge subtree.
    supernode::assign_delayed_capacities(&mut supernodes);

    // Step 9: small-leaf grouping (consumed at numeric time only when
    // the small_leaf gate is On). Same as the standard pipeline.
    let (small_leaf_groups, snode_group) =
        find_small_leaf_groups(&supernodes, &permuted_pattern, &effective_params.small_leaf);

    // Step 10: contribution sizes + peak memory.
    let contrib_sizes: Vec<usize> = supernodes.iter().map(|s| s.contrib_size()).collect();
    let peak_contrib_bytes = compute_peak_contrib(&supernodes, &contrib_sizes);

    let factor_slack = 1.2;

    Ok(SymbolicFactorization {
        n,
        perm,
        perm_inv,
        supernodes,
        factor_nnz_estimate: (factor_nnz as f64 * factor_slack) as usize,
        factor_slack,
        contrib_sizes,
        peak_contrib_bytes,
        etree,
        permuted_pattern,
        col_counts,
        small_leaf_groups,
        snode_group,
        cached_mc64: None,
        resolved_method: OrderingMethod::Amd,
        resolved_amalgamation: effective_params.amalgamation_strategy,
        resolved_preprocess: OrderingPreprocess::None,
        is_schur_tail: Some(n_schur),
    })
}

/// F3.2b helper: collapse all Schur-bearing supernodes (those whose
/// column range intersects `[n - n_schur, n)`) into a single tail
/// supernode. Mirrors MUMPS's HALO-SCHUR amalgamation
/// (`ana_orderings.F:9187-9220`).
///
/// Returns `InvalidInput` if the Schur-bearing supernodes are not
/// contiguous at the tail of the supernode list (would only arise from
/// a reducible matrix where the Schur set spans multiple etree roots -
/// not encountered in the KKT use cases this API targets).
fn merge_schur_tail_supernodes(
    supernodes: &mut Vec<Supernode>,
    n: usize,
    n_schur: usize,
) -> Result<(), RslabError> {
    let schur_lo = n - n_schur;

    // Step 0: split any supernode that straddles `schur_lo` into a
    // non-Schur half and a Schur half. This restores the invariant that
    // no supernode mixes eliminated-set columns with Schur-tail columns,
    // which the merge logic below assumes. The straddle case occurs
    // when adjacency-only amalgamation merges a small non-Schur
    // fundamental supernode into the first Schur fundamental supernode
    // via the size-based rule (both `< nemin`); without this split the
    // resulting compound supernode crosses `schur_lo` and the F3.2b
    // numeric driver would mis-locate the Schur columns.
    split_straddling_supernode(supernodes, schur_lo)?;

    // Identify Schur-bearing supernodes. Walk forward and find the
    // contiguous tail run; verify no Schur-bearing supernode lives
    // below that run (which would indicate a forest-structured Schur
    // set incompatible with F3.2a's postorder contract).
    let mut tail_start: Option<usize> = None;
    for (s, snode) in supernodes.iter().enumerate().rev() {
        let col_lo = snode.first_col;
        let col_hi = col_lo + snode.ncol();
        let intersects = col_hi > schur_lo && col_lo < n;
        if intersects {
            tail_start = Some(s);
        } else {
            // First non-Schur supernode walking back from the end
            // marks the boundary; nothing below should be Schur.
            break;
        }
    }
    // Verify nothing below `tail_start` is Schur-bearing (would imply
    // forest Schur structure).
    if let Some(start) = tail_start {
        for (s, snode) in supernodes.iter().enumerate().take(start) {
            let col_lo = snode.first_col;
            let col_hi = col_lo + snode.ncol();
            let intersects = col_hi > schur_lo && col_lo < n;
            if intersects {
                return Err(RslabError::InvalidInput(format!(
                    "Schur-bearing supernodes are not contiguous at the \
                     tail (snode {} bears Schur cols but lies below \
                     non-Schur supernode(s) preceding the tail run \
                     starting at index {}). This requires a forest- \
                     structured Schur set; see \
                     dev/research/schur-complement.md F3.2b.",
                    s, start
                )));
            }
        }
    }

    let Some(start) = tail_start else {
        return Err(RslabError::InvalidInput(
            "F3.2b merge: no Schur-bearing supernodes found despite \
             n_schur > 0 (symbolic invariant broken)"
                .to_string(),
        ));
    };
    if start == supernodes.len() - 1 {
        // Already a single Schur supernode at the tail - nothing to do.
        // Verify it covers the full Schur range.
        let last = &supernodes[start];
        let col_lo = last.first_col;
        let col_hi = col_lo + last.ncol();
        if col_lo > schur_lo || col_hi != n {
            return Err(RslabError::InvalidInput(format!(
                "F3.2b merge: single Schur supernode at index {} does not \
                 cover the full Schur tail [{}, {}) (covers [{}, {}))",
                start, schur_lo, n, col_lo, col_hi
            )));
        }
        return Ok(());
    }

    // Multi-supernode Schur tail: merge supernodes[start..] into one.
    // The merged supernode replaces supernodes[start] in place, and all
    // higher-indexed Schur supernodes are dropped from the list.
    //
    // Invariants on the merge set (verified):
    //   - Column ranges are contiguous and together cover [schur_lo, n).
    //   - Their union of children, minus the merge set itself, becomes
    //     the new merged supernode's children. (No merged supernode can
    //     be a child of another since they all bear Schur cols and the
    //     etree forces parent > child in the postordered numbering, but
    //     a child relationship would still place both in the merge set.)
    //   - nrow is bumped to cover the row pattern union; we use a
    //     conservative upper bound `(merged_first_col..n).len()` because
    //     all merged supernodes share rows in `[merged_first_col, n)`.
    let merged_first_col = supernodes[start].first_col;
    if merged_first_col != schur_lo {
        return Err(RslabError::InvalidInput(format!(
            "F3.2b merge: tail run starts at col {} but Schur tail \
             begins at {}",
            merged_first_col, schur_lo
        )));
    }

    // Verify contiguity of the column ranges.
    let mut expected = merged_first_col;
    for (s, snode) in supernodes.iter().enumerate().skip(start) {
        if snode.first_col != expected {
            return Err(RslabError::InvalidInput(format!(
                "F3.2b merge: supernode {} starts at col {} but expected {} \
                 (Schur supernode column ranges must be contiguous)",
                s, snode.first_col, expected
            )));
        }
        expected = snode.first_col + snode.ncol();
    }
    if expected != n {
        return Err(RslabError::InvalidInput(format!(
            "F3.2b merge: tail run ends at col {} but Schur tail ends at {}",
            expected, n
        )));
    }

    // Conservative nrow: max over merged supernodes of
    // `(s.first_col + s.nrow) - merged_first_col`. Each constituent
    // supernode's row pattern starts at its own first_col (in the
    // find_supernodes layout) and extends `nrow` rows. The merged
    // supernode starts at `merged_first_col`, so its row count must be
    // at least the largest constituent extent above `merged_first_col`.
    let mut merged_nrow = 0usize;
    let merge_indices: std::collections::HashSet<usize> = (start..supernodes.len()).collect();
    let mut merged_children: Vec<usize> = Vec::new();
    for snode in supernodes.iter().skip(start) {
        let extent = (snode.first_col + snode.nrow) - merged_first_col;
        if extent > merged_nrow {
            merged_nrow = extent;
        }
        for &c in &snode.children {
            if !merge_indices.contains(&c) {
                merged_children.push(c);
            }
        }
    }
    let merged_ncol = n_schur;
    if merged_nrow < merged_ncol {
        merged_nrow = merged_ncol;
    }
    let merged_row_indices: Vec<usize> =
        (merged_first_col..merged_first_col + merged_nrow).collect();

    // Replace supernodes[start] with the merged supernode and drop the
    // rest. Children indices in the surviving supernodes don't shift -
    // all merged supernodes are at the tail, so their indices were the
    // largest in the old list.
    supernodes[start] = Supernode {
        first_col: merged_first_col,
        ncol: merged_ncol,
        nrow: merged_nrow,
        row_indices: merged_row_indices,
        children: merged_children,
        // B1: merged supernode inherits the unbounded sentinel; the
        // B2 capacity-estimate pass runs after all merges complete
        // so it sees the post-merge tree and assigns a single
        // estimate per surviving supernode.
        delayed_capacity: usize::MAX,
    };
    supernodes.truncate(start + 1);
    Ok(())
}

/// F3.2b helper: split a supernode that straddles the Schur boundary
/// (`first_col < schur_lo < first_col + ncol`) into a non-Schur half
/// `[first_col, schur_lo)` and a Schur half `[schur_lo, first_col + ncol)`.
/// The Schur half inherits the original supernode's etree-parent slot
/// (the topmost cols of the original); the non-Schur half becomes the
/// only child of the Schur half.
///
/// At most one straddler can exist after `find_supernodes` (column
/// ranges are disjoint), so we either split exactly one or no-op.
///
/// Re-indexing rule: the new Schur half is inserted at position `b + 1`
/// where `b` is the original index. Any reference to a child index `> b`
/// shifts to `+1`; any reference `== b` (i.e., a parent that listed the
/// original as a child) remaps to `b + 1` since the Schur half now
/// occupies the original's etree role.
fn split_straddling_supernode(
    supernodes: &mut Vec<Supernode>,
    schur_lo: usize,
) -> Result<(), RslabError> {
    let mut straddle_idx: Option<usize> = None;
    for (s, snode) in supernodes.iter().enumerate() {
        let lo = snode.first_col;
        let hi = lo + snode.ncol;
        if lo < schur_lo && hi > schur_lo {
            if straddle_idx.is_some() {
                return Err(RslabError::InvalidInput(format!(
                    "F3.2b split: multiple supernodes straddle schur_lo={} \
                     (impossible after find_supernodes - column ranges are disjoint)",
                    schur_lo
                )));
            }
            straddle_idx = Some(s);
        }
    }
    let Some(b) = straddle_idx else {
        return Ok(());
    };

    let original = supernodes[b].clone();
    let ncol_ns = schur_lo - original.first_col;
    let ncol_sc = original.ncol - ncol_ns;
    let nrow_total = original.nrow;
    if original.row_indices.len() != nrow_total {
        return Err(RslabError::InvalidInput(format!(
            "F3.2b split: supernode {} has nrow={} but row_indices len={}",
            b,
            nrow_total,
            original.row_indices.len()
        )));
    }

    // Rewrite all child references before insertion. Indices > b shift
    // up by one; index == b (the original) remaps to b + 1 (the Schur
    // half) since the Schur half occupies the original's parental
    // position in the etree.
    for snode in supernodes.iter_mut() {
        for c in snode.children.iter_mut() {
            if *c == b {
                *c = b + 1;
            } else if *c > b {
                *c += 1;
            }
        }
    }

    // Non-Schur half replaces the original at index b. It keeps the
    // original's children (etree-children all have indices < b, so they
    // are unaffected by the shift above and still point at the right
    // supernodes after the split). Row pattern stays full nrow_total
    // because the contribution block of the non-Schur half feeds into
    // the Schur half above it.
    let non_schur = Supernode {
        first_col: original.first_col,
        ncol: ncol_ns,
        nrow: nrow_total,
        row_indices: original.row_indices.clone(),
        children: original.children,
        // B1: inherit the unbounded sentinel. The split happens
        // before the B2 capacity-estimate pass, so the post-split
        // supernodes get their real estimates uniformly.
        delayed_capacity: usize::MAX,
    };
    let schur_half = Supernode {
        first_col: schur_lo,
        ncol: ncol_sc,
        nrow: nrow_total - ncol_ns,
        row_indices: original.row_indices[ncol_ns..].to_vec(),
        children: vec![b],
        delayed_capacity: usize::MAX,
    };
    supernodes[b] = non_schur;
    supernodes.insert(b + 1, schur_half);
    Ok(())
}

/// Compute the peak contribution pool size needed during postorder traversal.
///
/// At any point during traversal, the live contribution blocks are those
/// of nodes that have been factored but whose contribution has not yet
/// been assembled into their parent. In serial postorder, a node's
/// contribution is consumed when its parent is factored.
fn compute_peak_contrib(supernodes: &[Supernode], contrib_sizes: &[usize]) -> usize {
    let n_snodes = supernodes.len();
    if n_snodes == 0 {
        return 0;
    }

    // Simulate the postorder traversal:
    // - When we process supernode k: allocate contrib[k], free contrib[child] for each child
    // - Track peak allocation
    let mut live = vec![false; n_snodes];
    let mut current_size = 0usize;
    let mut peak = 0usize;

    for k in 0..n_snodes {
        // Allocate this node's contribution block
        current_size += contrib_sizes[k];
        live[k] = true;

        if current_size > peak {
            peak = current_size;
        }

        // Free children's contribution blocks (they've been assembled)
        for &child in &supernodes[k].children {
            if live[child] {
                current_size -= contrib_sizes[child];
                live[child] = false;
            }
        }
    }

    peak * std::mem::size_of::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_factorize_basic() {
        // Simple tridiagonal
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();

        let params = SupernodeParams {
            nemin: 32,
            ..Default::default()
        };
        let sym = symbolic_factorize(&m, &params).unwrap();

        assert_eq!(sym.n, 4);
        assert_eq!(sym.perm.len(), 4);
        assert_eq!(sym.perm_inv.len(), 4);

        // Permutation should be valid
        let mut sorted_perm = sym.perm.clone();
        sorted_perm.sort();
        assert_eq!(sorted_perm, vec![0, 1, 2, 3]);

        // Factor NNZ estimate should be >= actual NNZ
        assert!(sym.factor_nnz_estimate > 0);

        // Total supernode columns = n
        let total_cols: usize = sym.supernodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total_cols, 4);
    }

    #[test]
    fn autorace_does_not_quadruple_symbolic_profiler_stages() {
        // S7 (repo-review-2026-06-09.md): AutoRace runs every
        // RACE_CANDIDATE against the *same* profiler Arc. Because
        // `SymbolicProfiler::record` appends and `set_total` overwrites,
        // the shared profiler ends with one full stage list per candidate
        // (~4x) measured against a single candidate's total. That yields
        // duplicate stage names and a spurious "stage sum exceeds total"
        // warning, and percentages that sum past 100%. The fix isolates
        // each candidate's profiler and copies only the winner's run into
        // the caller's shared profiler.
        let n = 16;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(2.0);
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(-1.0);
            }
        }
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();

        let prof = std::sync::Arc::new(std::sync::Mutex::new(SymbolicProfiler::new()));
        let params = SupernodeParams {
            symbolic_profiler: Some(prof.clone()),
            ..Default::default()
        };
        let _ = symbolic_factorize_with_method(&m, &params, OrderingMethod::AutoRace).unwrap();

        let report = prof.lock().unwrap().report();

        // Timing-independent invariant: the shared profiler must reflect
        // exactly one ordering run, so each instrumented stage name must
        // appear at most once. Pre-fix, ~RACE_CANDIDATES.len() copies of
        // each common-path stage are present regardless of how fast the
        // machine is (record() pushes even for 0 µs samples).
        let mut seen = std::collections::HashSet::new();
        for s in &report.stages {
            assert!(
                seen.insert(s.name),
                "stage '{}' recorded more than once - AutoRace leaked {} candidates' \
                 stages into the shared profiler (stages: {:?})",
                s.name,
                RACE_CANDIDATES.len(),
                report.stages.iter().map(|s| s.name).collect::<Vec<_>>(),
            );
        }
        // The stage-sum-exceeds-total warning must not fire for a single
        // ordering run.
        assert!(
            report.validation_warnings.is_empty(),
            "spurious profiler warnings: {:?}",
            report.validation_warnings
        );
    }

    #[test]
    fn test_symbolic_factorize_dense() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let sym = symbolic_factorize(&m, &params).unwrap();

        // For a dense matrix, factor NNZ = n*(n+1)/2 = 6
        assert!(sym.factor_nnz_estimate >= 6);
    }

    #[test]
    fn test_symbolic_factorize_kkt() {
        // Small KKT matrix
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 2, 2, 2],
            &[0, 1, 0, 1, 2],
            &[2.0, 3.0, 1.0, 1.0, -1e-8],
        )
        .unwrap();

        let params = SupernodeParams::default();
        let sym = symbolic_factorize(&m, &params).unwrap();

        assert_eq!(sym.n, 3);
        let total_cols: usize = sym.supernodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total_cols, 3);
    }

    #[test]
    fn test_perm_inverse_consistency() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();

        let params = SupernodeParams::default();
        let sym = symbolic_factorize(&m, &params).unwrap();

        // perm and perm_inv are inverses
        for i in 0..5 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
            assert_eq!(sym.perm_inv[sym.perm[i]], i);
        }
    }

    #[test]
    fn test_contrib_sizes_nonnegative() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();

        let params = SupernodeParams {
            nemin: 1,
            ..Default::default()
        };
        let sym = symbolic_factorize(&m, &params).unwrap();

        for &cs in &sym.contrib_sizes {
            // Contribution sizes should be non-negative (they're usize, always >= 0)
            // and for the root node it should be 0
            assert!(cs < 100000, "unreasonable contrib size: {}", cs);
        }

        // Root supernode should have 0 contribution block
        if let Some(last) = sym.supernodes.last() {
            assert_eq!(
                last.contrib_size(),
                0,
                "root should have no contribution block"
            );
        }
    }

    fn small_grid_5x5() -> CscMatrix {
        // 5x5 grid graph stored as CscMatrix (full symmetric, lower
        // triangle only). Used as a structurally non-trivial test
        // case where AMD, METIS, and SCOTCH all produce permutations
        // and the downstream pipeline must accept any of them.
        let m = 5;
        let n = 5;
        let idx = |r: usize, c: usize| r * n + c;
        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let k = idx(r, c);
                rows.push(k);
                cols.push(k);
                vals.push(4.0);
                if r + 1 < m {
                    rows.push(idx(r + 1, c));
                    cols.push(k);
                    vals.push(-1.0);
                }
                if c + 1 < n {
                    rows.push(idx(r, c + 1));
                    cols.push(k);
                    vals.push(-1.0);
                }
            }
        }
        CscMatrix::from_triplets(m * n, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn symbolic_factorize_amf_produces_valid_perm() {
        // Phase D wire-up smoke test: OrderingMethod::Amf must
        // produce a valid permutation through the full symbolic
        // pipeline (postorder composition, etree, column counts,
        // supernodes). This pins the dispatch wiring; bit-parity vs
        // MUMPS HAMF4 is the job of tests/amf_corpus_oracle.rs.
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amf).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
        assert_eq!(sym.resolved_method, OrderingMethod::Amf);
    }

    #[test]
    fn symbolic_factorize_metis_produces_valid_perm() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::MetisND).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
    }

    #[test]
    fn symbolic_factorize_scotch_produces_valid_perm() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::ScotchND).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
    }

    #[test]
    fn symbolic_factorize_kahip_produces_valid_perm() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::KahipND).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
    }

    #[test]
    fn symbolic_factorize_auto_produces_valid_perm() {
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::Auto).unwrap();
        assert_eq!(sym.n, 25);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, (0..25).collect::<Vec<_>>(), "perm is a bijection");
        for i in 0..25 {
            assert_eq!(sym.perm[sym.perm_inv[i]], i);
        }
    }

    #[test]
    fn choose_adaptive_rules() {
        // Pattern helper: diagonal pattern with n cols, nnz = density*n.
        fn pat_bufs(n: usize, avg_deg: usize) -> (Vec<usize>, Vec<usize>) {
            let total = n * avg_deg.max(1);
            let mut col_ptr = Vec::with_capacity(n + 1);
            let mut row_idx = Vec::with_capacity(total);
            let per = avg_deg.max(1);
            for j in 0..n {
                col_ptr.push(row_idx.len());
                for t in 0..per {
                    row_idx.push((j + t) % n.max(1));
                }
            }
            col_ptr.push(row_idx.len());
            (col_ptr, row_idx)
        }
        // Very-large-and-sparse (n > 100_000, avg_deg < 5.0) → AMD.
        // Issue #50 swap (2026-05-23): pre-fix this was the ScotchND
        // branch; see choose_adaptive's doc comment.
        let (cp, ri) = pat_bufs(200_000, 3);
        let p = CscPattern {
            n: 200_000,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::Amd
        );
        // Small-and-sparse (n<10_000, avg_deg<15) → delegates to
        // pick_default_method, which routes n≤10_000 to AMF. The F11
        // follow-up to issue #50 (2026-05-23) deleted the previous
        // small-and-sparse KahipND branch after the 838-matrix
        // inventory showed AMF aggregate fill 0.870× AMD vs KahipND
        // 0.984× and AMF aggregate time 0.832× AMD vs KahipND 0.990×
        // on that population; KahipND won only 16/838 matrices (1.9%)
        // vs AMF's 169/838 (20.2%). See choose_adaptive's doc comment
        // and dev/research/issue-50-metisnd-symbolic-cost.md §F12.
        let (cp, ri) = pat_bufs(500, 6);
        let p = CscPattern {
            n: 500,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::Amf
        );
        // Thin-large band (issue #67): n in (10_000, 100_000] that the
        // size rule would send to MetisND is overridden to AMF. The corpus
        // A/B (dev/research/issue-67-thin-large-ordering.md) found AMF wins
        // or ties MetisND on factor+solve across this whole band. Here
        // (n=50_000, avg_deg=20, uniform → not arrow) → AMF.
        let (cp, ri) = pat_bufs(50_000, 20);
        let p = CscPattern {
            n: 50_000,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::Amf
        );
        // Large-dense (n > 100_000, avg_deg >= 5, non-arrow) now also routes
        // to AMF (issue #73): the real factor+solve A/B found AMF wins on
        // every measured matrix in this regime, so the would-be-MetisND
        // decision is overridden to AMF at every n. The #50 avg_deg < 5 → AMD
        // catch fires first and is unaffected. (n=150_000, avg_deg=10, uniform
        // → not arrow.)
        let (cp, ri) = pat_bufs(150_000, 10);
        let p = CscPattern {
            n: 150_000,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::Auto),
            OrderingMethod::Amf
        );
        // Non-Auto passes through.
        let (cp, ri) = pat_bufs(500, 6);
        let p = CscPattern {
            n: 500,
            col_ptr: cp,
            row_idx: ri,
        };
        assert_eq!(
            choose_adaptive(&p, OrderingMethod::MetisND),
            OrderingMethod::MetisND
        );
    }

    /// Build a synthetic full-symmetric `CscPattern` with a prescribed
    /// per-column degree distribution. Connectivity is irrelevant to the
    /// degree-only arrow predicate, so row indices are filled with valid
    /// in-range values without forming a true symmetric pattern.
    fn pattern_with_degrees(degrees: &[usize]) -> CscPattern {
        let n = degrees.len();
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        for (j, &d) in degrees.iter().enumerate() {
            col_ptr.push(row_idx.len());
            for t in 0..d {
                row_idx.push((j + t) % n.max(1));
            }
        }
        col_ptr.push(row_idx.len());
        CscPattern {
            n,
            col_ptr,
            row_idx,
        }
    }

    #[test]
    fn is_arrow_bordered_fires_on_synthetic_arrow() {
        // Issue #64: a small set of very-high-degree border columns
        // carrying a large nnz share = arrow. 11_900 body columns of
        // degree 6 (71_400 nnz) + 100 border columns of degree 600
        // (60_000 nnz). avg_deg≈10.95, heavy_thr=max(64,88)=88; border
        // exceeds it. heavy_count=100 (0.83% of n < 5%); heavy_nnz share
        // 60_000/131_400 = 45.7% >= 20% → arrow.
        let mut degrees = vec![6usize; 11_900];
        degrees.extend(std::iter::repeat_n(600usize, 100));
        let pat = pattern_with_degrees(&degrees);
        assert!(is_arrow_bordered(&pat), "r05-shaped arrow must be detected");
    }

    #[test]
    fn is_arrow_bordered_rejects_uniform_sparse() {
        // Uniformly thin (PoissonControl / powerflow22 / bratu3d shape):
        // no column exceeds heavy_thr → not an arrow.
        let pat = pattern_with_degrees(&vec![8usize; 12_000]);
        assert!(
            !is_arrow_bordered(&pat),
            "uniform-sparse pattern must not be flagged as arrow"
        );
    }

    #[test]
    fn is_arrow_bordered_rejects_many_hubs() {
        // Exercises the count guard: 1000 columns of degree 1000 (10% of
        // n) carry 99% of the nnz, but a heavy set this large is not a
        // thin border - nested dissection is not obviously wrong, so the
        // arrow override must NOT fire. heavy_count=1000 = 10% > 5%.
        let mut degrees = vec![1000usize; 1000];
        degrees.extend(std::iter::repeat_n(1usize, 9000));
        let pat = pattern_with_degrees(&degrees);
        assert!(
            !is_arrow_bordered(&pat),
            "a large heavy set (10% of n) must be rejected by the count guard"
        );
    }

    #[test]
    fn is_arrow_bordered_rejects_low_nnz_share_border() {
        // bcsstk38 shape: 2 very-high-degree columns but they carry a
        // tiny nnz share (0.3%). The share guard rejects it. n must be
        // small enough that 2 cols < 5%, which is always true here.
        let mut degrees = vec![44usize; 8030];
        degrees.extend([614usize, 614usize]);
        let pat = pattern_with_degrees(&degrees);
        // heavy_thr = max(64, 8*~44) = ~355; the two 614-degree columns
        // are heavy but carry 1228 of ~354_548 nnz = 0.35% << 20%.
        assert!(
            !is_arrow_bordered(&pat),
            "a heavy set carrying a tiny nnz share must be rejected by the share guard"
        );
    }

    #[test]
    fn choose_adaptive_routes_arrow_to_amf() {
        // Issue #64: an arrow pattern with n>10_000 (which would
        // otherwise route to MetisND via pick_default_method) must be
        // overridden to Amf. Mirror the synthetic-arrow degree shape.
        let mut degrees = vec![6usize; 11_900];
        degrees.extend(std::iter::repeat_n(600usize, 100));
        let pat = pattern_with_degrees(&degrees);
        assert_eq!(
            choose_adaptive(&pat, OrderingMethod::Auto),
            OrderingMethod::Amf,
            "arrow/bordered pattern (n>10_000) must route to Amf, not MetisND"
        );
        // A uniform large-dense pattern (n > 100_000, avg_deg >= 5,
        // non-arrow) now routes to AMF via the #73 thin-large catch (the
        // would-be-MetisND decision is overridden to AMF at every n). This
        // does not exercise the arrow catch - the point is only that a
        // non-arrow shape still lands on AMF, just through #73 rather than
        // #64.
        let uniform = pattern_with_degrees(&vec![16usize; 120_000]);
        assert_eq!(
            choose_adaptive(&uniform, OrderingMethod::Auto),
            OrderingMethod::Amf,
            "uniform large-dense non-arrow pattern routes to AMF via the #73 catch"
        );
        // The arrow override must NOT fire below the size floor: a small
        // arrow already routes to Amf via the n<=10_000 rule, but assert
        // the override doesn't accidentally change a non-MetisND base.
        let mut small_arrow = vec![6usize; 4900];
        small_arrow.extend(std::iter::repeat_n(600usize, 100));
        let small = pattern_with_degrees(&small_arrow);
        assert_eq!(
            choose_adaptive(&small, OrderingMethod::Auto),
            OrderingMethod::Amf
        );
    }

    #[test]
    fn symbolic_factorize_default_uses_amf_for_small_matrices() {
        // Per Phase D of dev/plans/amf-clean-room.md: small matrices
        // (n <= 10_000) default to AMF, mirroring MUMPS's
        // ana_set_ordering.F rule for SYM=2 N≤10000.
        let m = small_grid_5x5();
        let params = SupernodeParams::default();
        let a = symbolic_factorize(&m, &params).unwrap();
        let b = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amf).unwrap();
        assert_eq!(
            a.perm, b.perm,
            "symbolic_factorize on small dense matrices must equal \
             symbolic_factorize_with_method(Amf)"
        );
        assert_eq!(a.factor_nnz_estimate, b.factor_nnz_estimate);
        assert_eq!(a.resolved_method, OrderingMethod::Amf);
    }

    #[test]
    fn pick_default_method_rules() {
        // Issue #50 (2026-05-23): the bordered-KKT and chain-pattern
        // catches that previously routed CRESC132/CHAINWOO/HYDROELL/
        // DIXMAANH/VESUVIO to MetisND were removed after the BK
        // pivoting cascade they defended against was killed by
        // issue #46's fixes (42434a5, 070840b). The numeric
        // inventory in dev/research/issue-50-numeric-inventory.csv
        // shows zero of 250 chain-catch-class matrices now have
        // AMD/MetisND num_nnz_l ratio ≥ 1.5×.
        //
        // The remaining rule is the MUMPS-style "small symmetric"
        // dispatch: n <= 10_000 → AMF, n > 10_000 → MetisND, with
        // the n == 0 sentinel returning AMD.

        // Empty matrix: AMD (avoids /0 and external-crate weirdness).
        assert_eq!(pick_default_method(0, 0), OrderingMethod::Amd);

        // Small matrices (n <= 10_000) → AMF regardless of avg_deg.
        assert_eq!(pick_default_method(715, 2839), OrderingMethod::Amf); // HAHN1
        assert_eq!(pick_default_method(3000, 8999), OrderingMethod::Amf); // DIXMAANH
        assert_eq!(pick_default_method(3000, 13_000), OrderingMethod::Amf);
        assert_eq!(pick_default_method(3083, 9484), OrderingMethod::Amf); // VESUVIO
        assert_eq!(pick_default_method(4000, 7999), OrderingMethod::Amf); // CHAINWOO
        assert_eq!(pick_default_method(5000, 20_000), OrderingMethod::Amf);
        assert_eq!(pick_default_method(5314, 22566), OrderingMethod::Amf); // CRESC132
        assert_eq!(pick_default_method(10_000, 100_000), OrderingMethod::Amf);

        // Large matrices (n > 10_000) → MetisND.
        assert_eq!(
            pick_default_method(20_000, 200_000),
            OrderingMethod::MetisND
        );
        // n=2_813_976, stored_nnz=6_622_463 (powerflow22 from #50):
        // → MetisND now (was → ScotchND via choose_adaptive's deleted
        // n>100k branch before #50). Issue #50's IPM-loop validation
        // is what justifies the deletion at this size.
        assert_eq!(
            pick_default_method(2_813_976, 6_622_463),
            OrderingMethod::MetisND
        );
    }

    #[test]
    fn pick_default_method_never_returns_kahip() {
        // Pins the session-08 driver-integration decision: KaHIP is
        // reachable only via explicit `with_method` or `Auto`. The
        // dispatcher must never return it on its own. See
        // `dev/research/ordering-kahip-driver-integration.md` for
        // the bake-off evidence (KaHIP ties METIS on fill at 4-6×
        // the per-call cost on 41 matrices). If a future change wants
        // to route some pattern to KaHIP by default, the maintainer
        // must consciously update this test and the research note.
        let shapes: &[(usize, usize)] = &[
            (0, 0),
            (10, 30),
            (500, 1500),
            (3083, 13333), // VESUVIOU
            (5314, 22566), // CRESC132
            (10_000, 50_000),
            (100_000, 500_000),
            (345_241, 1_343_126), // c-big from the shape bake-off
        ];
        for &(n, nnz) in shapes {
            let m = pick_default_method(n, nnz);
            assert_ne!(
                m,
                OrderingMethod::KahipND,
                "pick_default_method({}, {}) returned KahipND; \
                 see dev/research/ordering-kahip-driver-integration.md",
                n,
                nnz
            );
        }
    }

    /// 6×6 KKT-shaped matrix: leading 4×4 identity-like block, dense
    /// trailing 2×2 Schur, with off-diagonal coupling A_FS connecting
    /// rows {0..4} to columns {4,5}. Same structure used in the F3.1
    /// schur.rs unit tests.
    fn small_kkt_6x6() -> CscMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        // Diagonal in non-Schur block (1..=4 along positions 0..4).
        for i in 0..4 {
            rows.push(i);
            cols.push(i);
            vals.push((i + 1) as f64);
        }
        // Coupling A_FS: column 4 connects to rows 0,2; column 5 connects to rows 1,3.
        rows.push(4);
        cols.push(0);
        vals.push(0.5);
        rows.push(4);
        cols.push(2);
        vals.push(0.7);
        rows.push(5);
        cols.push(1);
        vals.push(0.3);
        rows.push(5);
        cols.push(3);
        vals.push(0.9);
        // Trailing 2×2 Schur block, dense.
        rows.push(4);
        cols.push(4);
        vals.push(1.5);
        rows.push(5);
        cols.push(4);
        vals.push(0.2);
        rows.push(5);
        cols.push(5);
        vals.push(2.5);
        CscMatrix::from_triplets(6, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn schur_symbolic_tail_invariant_user_order() {
        // schur_indices = [4, 5] in user order.
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[4, 5]).unwrap();
        assert_eq!(sym.n, 6);
        assert_eq!(sym.is_schur_tail, Some(2));
        assert_eq!(&sym.perm[4..], &[4, 5]);
    }

    #[test]
    fn schur_symbolic_tail_invariant_reversed_user_order() {
        // schur_indices = [5, 4] - user-supplied order MUST be preserved
        // exactly, not sorted.
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[5, 4]).unwrap();
        assert_eq!(sym.is_schur_tail, Some(2));
        assert_eq!(&sym.perm[4..], &[5, 4]);
    }

    #[test]
    fn schur_symbolic_perm_is_valid_permutation() {
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[4, 5]).unwrap();
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
        // perm_inv consistency.
        for (new, &old) in sym.perm.iter().enumerate() {
            assert_eq!(sym.perm_inv[old], new);
        }
    }

    #[test]
    fn schur_symbolic_empty_falls_back_to_standard() {
        // Empty schur_indices must produce a SymbolicFactorization with
        // is_schur_tail = None (delegates to symbolic_factorize_with_method).
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[]).unwrap();
        assert_eq!(sym.is_schur_tail, None);
    }

    #[test]
    fn schur_symbolic_full_n_rejected() {
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let result = symbolic_factorize_with_schur(&m, &params, &[0, 1, 2, 3, 4, 5]);
        assert!(matches!(result, Err(RslabError::InvalidInput(_))));
    }

    #[test]
    fn schur_symbolic_duplicate_rejected() {
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let result = symbolic_factorize_with_schur(&m, &params, &[4, 4]);
        assert!(matches!(result, Err(RslabError::InvalidInput(_))));
    }

    #[test]
    fn schur_symbolic_supernodes_cover_n() {
        // Sanity check: the supernode layout still covers all n columns.
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[4, 5]).unwrap();
        let total: usize = sym.supernodes.iter().map(|s| s.ncol()).sum();
        assert_eq!(total, 6);
    }

    #[test]
    fn schur_symbolic_single_schur_index() {
        let m = small_kkt_6x6();
        let params = SupernodeParams::default();
        let sym = symbolic_factorize_with_schur(&m, &params, &[5]).unwrap();
        assert_eq!(sym.is_schur_tail, Some(1));
        assert_eq!(sym.perm[5], 5);
        let mut sorted = sym.perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }

    /// PoissonControl KKT lower-triangle CSC, mirrors
    /// `src/bin/diag_poisson_kkt.rs`. n_kkt = 3K². K=20 → n=1200,
    /// large enough to exceed amd_switch=120 (so SCOTCH actually
    /// runs the multilevel pipeline) but small enough to be cheap.
    fn poisson_kkt_csc(k: usize) -> CscMatrix {
        let m = k * k;
        let n_kkt = 3 * m;
        let h = 1.0 / (k as f64 + 1.0);
        let alpha = 0.01;
        let inv_h2 = 1.0 / (h * h);

        let mut rows: Vec<usize> = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for i in 0..m {
            rows.push(i);
            cols.push(i);
            vals.push(h * h);
        }
        for i in 0..m {
            rows.push(m + i);
            cols.push(m + i);
            vals.push(alpha * h * h);
        }
        for i in 0..k {
            for j in 0..k {
                let c = i * k + j;
                let con_row = 2 * m + c;
                rows.push(con_row);
                cols.push(c);
                vals.push(4.0 * inv_h2);
                if i > 0 {
                    rows.push(con_row);
                    cols.push((i - 1) * k + j);
                    vals.push(-inv_h2);
                }
                if i + 1 < k {
                    rows.push(con_row);
                    cols.push((i + 1) * k + j);
                    vals.push(-inv_h2);
                }
                if j > 0 {
                    rows.push(con_row);
                    cols.push(i * k + (j - 1));
                    vals.push(-inv_h2);
                }
                if j + 1 < k {
                    rows.push(con_row);
                    cols.push(i * k + (j + 1));
                    vals.push(-inv_h2);
                }
                rows.push(con_row);
                cols.push(m + c);
                vals.push(-1.0);
            }
        }
        CscMatrix::from_triplets(n_kkt, &rows, &cols, &vals).expect("kkt csc")
    }

    #[test]
    fn issue_3_scotchnd_on_kkt_recurses_after_o13() {
        // History: SCOTCH bisection used to produce no separator on
        // bordered-KKT patterns - its vertex-separator FM stopped the
        // whole pass the first time both PQ heads were imbalance-
        // rejected, abandoning feasible queued moves - so the recursion
        // collapsed into amd_leaf for the entire graph and
        // `resolved_method` reported `Amd`.
        //
        // Finding O13 fixed that early stop. SCOTCH now finds a real
        // separator on this KKT pattern (verified directly in
        // `crates/rslab-scotch/tests/issue_3_kkt_repro.rs`:
        // `issue_3_scotch_recurses_on_kkt_after_o13`), so ScotchND no
        // longer degenerates: `resolved_method` reports the requested
        // ScotchND, and the ordering is genuinely SCOTCH's - not a
        // relabelled AMD leaf. This test guards that post-O13 behavior
        // at the `rslab` symbolic boundary.
        //
        // Preprocess is pinned to None so SCOTCH sees the raw KKT
        // pattern (LdltCompress would shrink it past the point the
        // degeneracy ever exercised).
        let m = poisson_kkt_csc(20);
        let params = SupernodeParams {
            preprocess: OrderingPreprocess::None,
            ..SupernodeParams::default()
        };
        let sym = symbolic_factorize_with_method(&m, &params, OrderingMethod::ScotchND).unwrap();
        assert_eq!(
            sym.resolved_method,
            OrderingMethod::ScotchND,
            "post-O13 SCOTCH recurses on this KKT pattern; resolved_method \
             must report ScotchND, not the AMD-leaf fallback"
        );

        // The permutation must be a valid bijection over 0..n.
        let n = m.n;
        assert_eq!(sym.perm.len(), n);
        let mut seen = vec![false; n];
        for &p in &sym.perm {
            assert!(p < n, "perm entry out of range");
            assert!(!seen[p], "perm is not a bijection");
            seen[p] = true;
        }

        // And it must differ from AMD's ordering - proof the recursion
        // ran SCOTCH nested dissection rather than collapsing to the
        // AMD leaf (which would return AMD's permutation verbatim).
        let amd = symbolic_factorize_with_method(&m, &params, OrderingMethod::Amd).unwrap();
        assert_ne!(
            sym.perm, amd.perm,
            "ScotchND ordering must not be bit-identical to AMD's; \
             that would indicate the degenerate AMD-leaf fallback"
        );
    }

    #[test]
    fn issue_3_auto_on_kkt_routes_via_pick_default_method() {
        // Issue #3 invariant: the `Auto` path and the no-arg
        // `symbolic_factorize` default must resolve to the *same* concrete
        // ordering on every matrix. PoissonControl K=58 (n=10092, stored
        // avg_deg≈2.67) is a uniformly-thin KKT just inside the #67
        // thin-large AMF band ((10_000, 100_000], non-arrow), so both paths
        // now resolve to AMF. (Before #67 this matrix resolved to MetisND
        // via pick_default_method's n>10_000 rule; the MetisND delegation is
        // still covered by the `choose_adaptive_rules` n=150_000 case.)
        let m = poisson_kkt_csc(58);
        let params = SupernodeParams::default();
        let auto = symbolic_factorize_with_method(&m, &params, OrderingMethod::Auto).unwrap();
        let default = symbolic_factorize(&m, &params).unwrap();
        assert_eq!(
            auto.resolved_method, default.resolved_method,
            "Auto must resolve to the same concrete method as \
             `symbolic_factorize` (which also routes through choose_adaptive)"
        );
        assert_eq!(auto.resolved_method, OrderingMethod::Amf);
    }
}
