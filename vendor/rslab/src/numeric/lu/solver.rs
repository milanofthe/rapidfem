//! The public LU interface: the reusable analysis on the symmetrized pattern
//! (with the optional MC64 row matching), the factor handle and its solves.

use super::factor::factor_general_lu_numeric;
use super::factors::LuPivots;
use super::structure::LuStructure;

use crate::error::RslabError;
use crate::numeric::settings::SolverSettings;
use crate::numeric::supernodal::analysis::analyze_with;
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;
use std::sync::Mutex;

/// Lower triangle of the symmetrized pattern `B union B^T` as CSC `(col_ptr,
/// row_idx)`, `B` the pattern of `a` with row `r` renamed `row_map[r]` (the
/// matching's row permutation, applied on the fly). The symmetric analysis
/// needs a structurally symmetric pattern so the elimination tree carries
/// fill for both `L` and `U`.
fn symmetrized_lower_pattern<T: Scalar>(
    a: &GeneralCsc<T>,
    row_map: Option<&[usize]>,
) -> (Vec<usize>, Vec<usize>) {
    use rayon::prelude::*;
    let n = a.n;
    let row = |k: usize| row_map.map_or(a.row_idx[k], |m| m[a.row_idx[k]]);
    // Counting-scatter: each entry contributes a lower-triangle pair
    // `(hi, lo)` to bucket `lo`; the buckets are then sorted and deduped.
    let mut start = vec![0usize; n + 1];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            start[row(k).min(j) + 1] += 1;
        }
    }
    for j in 0..n {
        start[j + 1] += start[j];
    }
    let mut scattered = vec![0usize; start[n]];
    let mut cursor = start[..n].to_vec();
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = row(k);
            let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
            scattered[cursor[lo]] = hi;
            cursor[lo] += 1;
        }
    }
    // Sort and dedup every bucket in place (in parallel), then compact.
    let mut buckets: Vec<&mut [usize]> = Vec::with_capacity(n);
    let mut rest: &mut [usize] = &mut scattered;
    for j in 0..n {
        let (head, tail) = rest.split_at_mut(start[j + 1] - start[j]);
        buckets.push(head);
        rest = tail;
    }
    let kept: Vec<usize> = buckets
        .into_par_iter()
        .with_min_len(256)
        .map(|seg| {
            seg.sort_unstable();
            let mut len = 0;
            for p in 0..seg.len() {
                if len == 0 || seg[p] != seg[len - 1] {
                    seg[len] = seg[p];
                    len += 1;
                }
            }
            len
        })
        .collect();
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx = Vec::with_capacity(start[n]);
    for j in 0..n {
        row_idx.extend_from_slice(&scattered[start[j]..start[j] + kept[j]]);
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx)
}

/// Whether the row matching is needed: some column's diagonal entry is
/// missing, zero, or negligible against the column (`|a_jj|` below
/// `negligible` times its largest entry). There the front-local
/// pivot search has no usable diagonal and the element growth runs away;
/// the circuit matrices of the KLU corpus all have such columns and factor to
/// roundoff only with the matching. Where every diagonal entry can pivot the
/// permutation buys nothing and costs: on the MoM near-field matrices it
/// breaks the diagonal the pivoting would use, adding 5 to 20 percent fill,
/// hundreds of perturbed pivots and residuals up to seven orders of
/// magnitude worse, and the circuits with a full diagonal factor the same
/// either way.
fn diagonal_needs_matching<T: Scalar>(a: &GeneralCsc<T>, negligible: f64) -> bool {
    use rayon::prelude::*;
    (0..a.n).into_par_iter().with_min_len(1024).any(|j| {
        let (rows, vals) = (
            &a.row_idx[a.col_ptr[j]..a.col_ptr[j + 1]],
            &a.values[a.col_ptr[j]..a.col_ptr[j + 1]],
        );
        let top = vals.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
        let diag = rows.binary_search(&j).map_or(0.0, |p| vals[p].magnitude());
        diag.is_nan() || diag <= negligible * top
    })
}

/// The MC64 row matching the analysis was done under: the factored matrix
/// is `B = diag(r) P A diag(c)` with `B` row `i` = `A` row `row_of[i]`.
pub(super) struct LuMatching {
    pub(super) row_of: Vec<usize>,
    /// `A`-row scaling.
    pub(super) r: Vec<f64>,
    pub(super) c: Vec<f64>,
}

impl LuMatching {
    /// `map[r]`: the row of `B` that row `r` of `A` becomes.
    pub(super) fn row_map(&self) -> Vec<usize> {
        let mut map = vec![0usize; self.row_of.len()];
        for (i, &r) in self.row_of.iter().enumerate() {
            map[r] = i;
        }
        map
    }
}

pub struct LuSymbolic {
    pub(super) symb: crate::numeric::supernodal::analysis::SupernodalAnalysis,
    pub(super) n: usize,
    pub(super) nnz: usize,
    pub(super) matching: Option<LuMatching>,
    /// Wall time of the analysis and the ordering it was asked for, carried
    /// into the diagnostics of every factorization reusing it.
    pub(super) analyze_ms: f64,
    pub(super) requested_ordering: crate::symbolic::OrderingMethod,
    /// [`estimate_memory`](Self::estimate_memory) results, keyed by scalar
    /// size (the estimate depends on `T` only through `size_of::<T>()`, and
    /// rebuilding the supernode row structures per call is expensive).
    pub(super) est_cache: Mutex<Vec<(usize, crate::diagnostics::MemoryEstimate)>>,
    /// The split permuted input's program ([`InputProgram::general`](crate::numeric::supernodal::InputProgram)), built at the first
    /// factorization: every (re)factorization reduces to one linear values
    /// scatter (row matching and equilibration applied on the way).
    pub(super) input: std::sync::OnceLock<crate::numeric::supernodal::InputProgram>,
    /// The exact structures of `L` and `U`, tighter than the analysis's
    /// symmetric one on an unsymmetric pattern.
    pub(super) structure: LuStructure,
}

impl LuSymbolic {
    /// The fill-reducing column ordering of the analysis (`perm[k]` the column of the
    /// (row-matched) matrix that became column `k`): pass it to
    /// [`SolverSettings::with_permutation`] to analyse a nearby pattern without a new
    /// ordering.
    pub fn permutation(&self) -> &[usize] {
        self.symb.permutation()
    }

    /// PARDISO phase 1: analyze the symmetrized pattern `A union A^T` of `a`
    /// (values ignored, so any matrix with the target pattern works), with
    /// the row matching where the diagonal needs it. Reuse the result across
    /// many [`factor`](Self::factor) calls that share the pattern.
    pub fn analyze<T: Scalar>(
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
    ) -> Result<LuSymbolic, RslabError> {
        a.validate()?;
        let n = a.n;
        let nnz = a.row_idx.len();
        if n == 0 {
            return Ok(LuSymbolic {
                symb: analyze_with(0, &[0], &[], opts)?,
                n: 0,
                nnz: 0,
                matching: None,
                analyze_ms: 0.0,
                requested_ordering: opts.ordering.method,
                est_cache: Mutex::new(Vec::new()),
                input: std::sync::OnceLock::new(),
                structure: LuStructure::empty(),
            });
        }
        let t = crate::clock::Instant::now();
        // MC64 row matching: analyze the row-permuted matrix `B` whose
        // diagonal carries the matched entries, where the diagonal of `A`
        // cannot carry the pivots itself.
        let matching = if opts.matching.enabled
            && diagonal_needs_matching(a, opts.matching.negligible_diagonal)
        {
            let cache = crate::logging::timed(
                || "lu analyze: matching".into(),
                || {
                    crate::numeric::settings::in_scoped_pool(opts.resolved_threads(), 0, || {
                        crate::scaling::mc64::compute_matching_general(a)
                    })
                },
            )?;
            if cache.n_matched == n {
                let (r, c) = crate::scaling::mc64::unsymmetric_scaling(&cache);
                // `cache.perm[j]` is the row matched to column `j`: it becomes
                // row `j` of `B`.
                Some(LuMatching {
                    row_of: cache.perm,
                    r,
                    c,
                })
            } else {
                crate::logging::warn(&format!(
                    "lu analyze: structurally rank-deficient ({} of {n} columns matched); row matching skipped",
                    cache.n_matched
                ));
                None
            }
        } else {
            None
        };
        let (col_ptr, row_idx) = crate::logging::timed(
            || "lu analyze: symmetrized pattern".into(),
            || {
                let row_map = matching.as_ref().map(LuMatching::row_map);
                crate::numeric::settings::in_scoped_pool(opts.resolved_threads(), 0, || {
                    symmetrized_lower_pattern(a, row_map.as_deref())
                })
            },
        );
        let symb = analyze_with(n, &col_ptr, &row_idx, opts)?;
        let structure = match (symb.sym_and_levels(), symb.ll_schedule()) {
            (Some((sym, _)), Some(sched)) => {
                let row_map = matching.as_ref().map(LuMatching::row_map);
                crate::logging::timed(
                    || "lu analyze: exact structure".into(),
                    || LuStructure::build(&a.col_ptr, &a.row_idx, row_map.as_deref(), sym, sched),
                )
            }
            _ => LuStructure::empty(),
        };
        let analyze_ms = t.elapsed().as_secs_f64() * 1e3;
        if crate::logging::enabled(crate::logging::LogLevel::Info) {
            let d = symb.decisions(opts.ordering.method);
            crate::logging::info(&format!(
                "lu analyze: n={n} nnz(A)={nnz} ordering={}{} supernodes={} max_front={} \
                 levels={} {analyze_ms:.1} ms",
                d.ordering_used,
                if d.ordering_used != d.ordering_requested {
                    format!(" (requested {})", d.ordering_requested)
                } else {
                    String::new()
                },
                d.n_supernodes,
                d.max_front,
                d.tree_levels
            ));
        }
        Ok(LuSymbolic {
            symb,
            n,
            nnz,
            matching,
            analyze_ms,
            requested_ordering: opts.ordering.method,
            est_cache: Mutex::new(Vec::new()),
            input: std::sync::OnceLock::new(),
            structure,
        })
    }

    /// Whether the analysis carries an MC64 row matching.
    pub fn has_matching(&self) -> bool {
        self.matching.is_some()
    }

    /// PARDISO **phases 2-3**: equilibrate and LU-factor `a`, reusing this
    /// analysis, into a ready-to-solve [`LuSolver`]. `a` must share the analyzed
    /// pattern. The unsymmetric twin of [`LdltSymbolic::factor`].
    ///
    /// [`LdltSymbolic::factor`]: crate::numeric::ldlt::LdltSymbolic::factor
    pub fn factor<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
    ) -> Result<LuSolver<T>, RslabError> {
        let estimate = self.estimate_memory::<T>();
        let resolved_threads = opts.threads.resolve(|cap| {
            crate::numeric::supernodal::analysis::auto_threads(&self.symb, &estimate, cap)
        });
        let opts = &opts.pinned(resolved_threads);
        let warnings = opts.ignored_on(crate::numeric::settings::FactorPath::Lu);
        for w in &warnings {
            crate::logging::warn(&format!("lu settings: {w}"));
        }
        let t = crate::clock::Instant::now();
        let numeric = factor_general_lu_numeric(self, a, opts)?;
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        let nnz = numeric.factor_nnz() as u64;
        let factor_bytes = numeric.bytes() as u64;
        let (l, ut, factors) = numeric.into_parts();
        let mut decisions = self.symb.decisions(self.requested_ordering);
        decisions.scaling = if self.matching.is_some() {
            "Mc64RowMatching".to_string()
        } else {
            "TwoSidedRowCol".to_string()
        };
        decisions.method = "LeftLooking".to_string();
        let mut diagnostics = crate::diagnostics::Diagnostics {
            threads: resolved_threads,
            n: self.n,
            nnz_a: self.nnz as u64,
            factor_nnz: nnz,
            estimate: Some(estimate),
            decisions,
            numeric: crate::diagnostics::NumericReport {
                perturbed: factors.n_perturbed,
                two_by_two: None,
                inertia: None,
            },
            warnings,
            ..Default::default()
        };
        // Bytes per stored entry: the scalar value plus its usize index.
        diagnostics.push("analyze", self.analyze_ms, 0, 0);
        diagnostics.push("factor", factor_ms, estimate.factor_flops, factor_bytes);
        if crate::logging::enabled(crate::logging::LogLevel::Info) {
            crate::logging::info(&format!("lu factor: {}", diagnostics.summary()));
        }
        // Solve layout: supernodal panels of `L` and `U^T` plus the tree
        // schedule; the CSC arrays are released so the factor is held once.
        let t = crate::clock::Instant::now();
        let plan_l = crate::numeric::supernodal::solve::SolvePlan::from_panels(
            l,
            &factors.supernode_parent,
            true,
            opts.solve,
        );
        let plan_u = crate::numeric::supernodal::solve::SolvePlan::from_panels(
            ut,
            &factors.supernode_parent,
            false,
            opts.solve,
        );
        diagnostics.push(
            "solve-layout",
            t.elapsed().as_secs_f64() * 1e3,
            0,
            (plan_l.bytes() + plan_u.bytes()) as u64,
        );
        Ok(LuSolver {
            solve_threads: opts.threads,
            factors,
            plan_l,
            plan_u,
            nnz: nnz as usize,
            diagnostics,
            solves: Default::default(),
        })
    }

    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)` of the symmetrized
    /// pattern - for factorization-cost diagnostics (front-size distribution and
    /// a factor-flop estimate). See `SupernodalAnalysis::front_dims`.
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        self.symb.front_dims()
    }

    /// Number of assembly-tree levels (level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.symb.n_levels()
    }

    /// Supernode count per assembly-tree level (available tree-parallelism by
    /// depth). See `SupernodalAnalysis::level_widths`.
    pub fn level_widths(&self) -> Vec<usize> {
        self.symb.level_widths()
    }

    /// **A-priori** peak-memory estimate for factoring a matrix of scalar type `T`
    /// with this analysis - computed purely from the symbolic structure, *before*
    /// any numeric work, so a scheduler can fail-fast or pick an approximation when
    /// the estimate exceeds the memory budget. Deterministic and reproducible.
    /// Exact symbolic factor fill (the compact L+U value count summed over
    /// supernodes), the reliable memory-backstop metric. Unlike
    /// [`MemoryEstimate::factor_nnz`](crate::diagnostics::MemoryEstimate::factor_nnz),
    /// a dense-panel upper bound that overshoots the real fill ~6-7x
    /// non-uniformly across orderings, this tracks the actually-stored fill.
    pub fn symbolic_factor_nnz(&self) -> usize {
        let Some((sym, _)) = self.symb.sym_and_levels() else {
            return 0;
        };
        let Some(sched) = self.symb.ll_schedule() else {
            return 0;
        };
        (0..sym.supernodes.len())
            .map(|s| {
                let nc = sym.supernodes[s].ncol;
                let cnrow = sched.rows(s).len().saturating_sub(nc);
                // L: diagonal lower-triangle + off-diagonal rows; U: upper-tri + U12.
                let l = nc * (nc + 1) / 2 + cnrow * nc;
                let u = nc * (nc + 1) / 2 + nc * cnrow;
                l + u
            })
            .sum()
    }

    pub fn estimate_memory<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        let value_bytes = std::mem::size_of::<T>();
        // Cache per scalar size: the estimate is a pure function of the
        // structure and `size_of::<T>()`, and `tuned` + phased `factor` ask
        // for it repeatedly.
        if let Ok(cache) = self.est_cache.lock() {
            if let Some(&(_, est)) = cache.iter().find(|&&(vb, _)| vb == value_bytes) {
                return est;
            }
        }
        let est = self.estimate_memory_for(value_bytes);
        if let Ok(mut cache) = self.est_cache.lock() {
            if !cache.iter().any(|&(vb, _)| vb == value_bytes) {
                cache.push((value_bytes, est));
            }
        }
        est
    }

    /// The uncached estimate body, a pure function of the symbolic structure
    /// and the scalar size.
    fn estimate_memory_for(&self, value_bytes: usize) -> crate::diagnostics::MemoryEstimate {
        let Some((sym, _levels)) = self.symb.sym_and_levels() else {
            return crate::diagnostics::estimate_left_looking(
                0,
                &|_| 0,
                &|_| 0,
                &|_| &[],
                value_bytes,
                0,
                false,
            );
        };
        let nsuper = sym.supernodes.len();
        let Some(sched) = self.symb.ll_schedule() else {
            return crate::diagnostics::estimate_left_looking(
                0,
                &|_| 0,
                &|_| 0,
                &|_| &[],
                value_bytes,
                0,
                false,
            );
        };
        // The `L` and `U^T` panels of a supernode, `(w + m_l) x w` and
        // `(w + m_u) x w` on the exact structure; they are the stored factor
        // (no compact copy, see `PanelFactor`).
        let st = &self.structure;
        let panel_bytes = |s: usize| -> u64 {
            let nc = sym.supernodes[s].ncol;
            ((st.rows_l(s).len() + st.cols_u(s).len()) * nc * value_bytes) as u64
        };
        let compact_bytes = panel_bytes;
        // The split permuted input: its values (per factorization) and structure
        // (per analysis, `u32` indices and `usize` positions).
        let input_bytes = (self.nnz * (value_bytes + 12)) as u64;
        let mut est = crate::diagnostics::estimate_left_looking(
            nsuper,
            &panel_bytes,
            &compact_bytes,
            &|s| sched.updaters(s),
            value_bytes,
            input_bytes,
            true,
        );
        est.factor_flops = (0..nsuper)
            .map(|s| {
                let nc = sym.supernodes[s].ncol as u64;
                (st.rows_l(s).len() * st.cols_u(s).len()) as u64 * nc
            })
            .sum();
        est
    }
}

/// A factored unsymmetric LU solver, ready to solve against many right-hand
/// sides, the unsymmetric twin of [`LdltSolver`](crate::numeric::ldlt::LdltSolver).
/// Build via [`LuSymbolic::factor`] (analyze once, factor many) or the one-shot
/// [`LuSolver::factor`].
pub struct LuSolver<T> {
    factors: LuPivots,
    /// `L` and `U^T` (the panels, their only storage) with the tree schedule
    /// of [`crate::numeric::supernodal::solve`]; `factors` carries the
    /// permutations, scalings and counters with empty CSC arrays.
    plan_l: crate::numeric::supernodal::solve::SolvePlan<T>,
    plan_u: crate::numeric::supernodal::solve::SolvePlan<T>,
    nnz: usize,
    diagnostics: crate::diagnostics::Diagnostics,
    /// Solve-phase accumulators (every `solve*` call records into them).
    solves: crate::diagnostics::SolveCounter,
    /// The worker policy the factorization ran with, which a Krylov solve
    /// preconditioned by this factor orthogonalizes under.
    solve_threads: crate::numeric::settings::Threads,
}

impl<T: Scalar> LuSolver<T> {
    /// One-shot analyze + equilibrate + factor of a general matrix `A`.
    pub fn factor(a: &GeneralCsc<T>, opts: &SolverSettings) -> Result<Self, RslabError> {
        LuSymbolic::analyze(a, opts)?.factor(a, opts)
    }

    /// Per-call diagnostics: measured factor time, fill, thread budget, and the
    /// a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate).
    /// Everything this factorization can tell about itself (see
    /// [`Diagnostics`](crate::Diagnostics)), the solve-phase accumulators
    /// included. A snapshot.
    pub fn diagnostics(&self) -> crate::diagnostics::Diagnostics {
        let mut d = self.diagnostics.clone();
        d.solves = self.solves.snapshot();
        d
    }

    /// Stored fill `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.nnz
    }

    /// Number of statically perturbed pivots (preconditioner mode).
    pub fn n_perturbed(&self) -> usize {
        self.factors.n_perturbed
    }

    /// The matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }
}

impl<T: Scalar> crate::numeric::direct::SolveCore<T> for LuSolver<T> {
    const NAME: &'static str = "lu";

    fn dim(&self) -> usize {
        self.factors.n
    }

    fn counter(&self) -> &crate::diagnostics::SolveCounter {
        &self.solves
    }

    /// The factored matrix is `(P_r D_r) A (D_c P_c^T) = L U`. `A x = b`
    /// gathers through the row side, sweeps `L` forward and `U` backward and
    /// scatters through the column side; `A^T x = b` is the mirror image,
    /// `U^T` forward and `L^T` backward.
    fn solve_raw(&self, b: &[T], nrhs: usize, transpose: bool) -> Result<Vec<T>, RslabError> {
        let f = &self.factors;
        let n = f.n;
        let (gather, g_scale, scatter, s_scale) = if transpose {
            (&f.perm, &f.d_col, &f.perm_row, &f.d_row)
        } else {
            (&f.perm_row, &f.d_row, &f.perm, &f.d_col)
        };
        // The sweeps take the block row-major: y[e * nrhs + c].
        let mut y = vec![T::zero(); n * nrhs];
        for (e, &orig) in gather.iter().enumerate() {
            let s = T::from_real(g_scale[orig]);
            for c in 0..nrhs {
                y[e * nrhs + c] = b[c * n + orig] * s;
            }
        }
        if transpose {
            self.plan_u.forward(nrhs, &mut y);
            self.plan_l.backward(nrhs, &mut y);
        } else {
            self.plan_l.forward(nrhs, &mut y);
            self.plan_u.backward(nrhs, &mut y);
        }
        let mut out = vec![T::zero(); n * nrhs];
        for (e, &orig) in scatter.iter().enumerate() {
            let s = T::from_real(s_scale[orig]);
            for c in 0..nrhs {
                out[c * n + orig] = y[e * nrhs + c] * s;
            }
        }
        Ok(out)
    }
}

crate::numeric::direct::direct_solver!(LuSolver);
