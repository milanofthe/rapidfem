//! The public LU interface: the reusable analysis on the symmetrized pattern
//! (with the optional MC64 row matching), the factor handle and its solves.

use super::factor::factor_general_lu_numeric;
use super::factors::LuPivots;
use super::structure::LuStructure;

use crate::error::RslabError;
use crate::numeric::settings::SolverSettings;
use crate::numeric::supernodal::analysis::analyze_with;
use crate::numeric::supernodal::panel::{PanelFactor, PanelStorage};
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
    /// size and complexity (the estimate depends on `T` only through them,
    /// and walking every node's updaters per call is expensive).
    pub(super) est_cache: Mutex<Vec<((usize, bool), crate::diagnostics::MemoryEstimate)>>,
    /// The split permuted input's program ([`InputProgram::general`](crate::numeric::supernodal::InputProgram)), built at the first
    /// factorization: every (re)factorization reduces to one linear values
    /// scatter (row matching and equilibration applied on the way).
    pub(super) input: std::sync::OnceLock<crate::numeric::supernodal::InputProgram>,
    /// The exact structures of `L` and `U`, tighter than the analysis's
    /// symmetric one on an unsymmetric pattern.
    pub(super) structure: LuStructure,
}

impl LuSymbolic {
    /// Heap bytes this analysis holds: the ordering, the supernodes, the
    /// permuted pattern and, once a factorization has built them, the
    /// schedule and the input program every later factorization reuses.
    /// The analysis stays alive while it is factored, so these bytes are part
    /// of every factorization's footprint.
    pub fn heap_bytes(&self) -> u64 {
        use crate::memory::vec_bytes;
        self.symb.heap_bytes()
            + self.structure.heap_bytes()
            + self.input.get().map_or(0, |p| p.heap_bytes())
            + self.matching.as_ref().map_or(0, |m| {
                vec_bytes(&m.row_of) + vec_bytes(&m.r) + vec_bytes(&m.c)
            })
    }

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
        let pools = super::node::LuPools::new();
        let (l, ut, factors, nnz, mut diagnostics, opts) = self.numeric(a, opts, None, &pools)?;
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
            nnz,
            diagnostics,
            solves: Default::default(),
            factored: true,
            // A one-time factor does not hold on to the scratch; the first
            // refactorization grows its own and keeps it.
            pools: super::node::LuPools::new(),
        })
    }

    /// Factor `a` again into `lu`, a factor of this analysis (the next
    /// Newton step): the numeric factorization of [`factor`](Self::factor),
    /// its panels written into `lu`'s buffers and `lu`'s solve schedule kept
    /// where pivoting left the rows unchanged. The same bits as a fresh
    /// [`factor`](Self::factor). After an error `lu` holds no factor and
    /// refuses to solve until a refactorization succeeds.
    pub fn refactor<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
        lu: &mut LuSolver<T>,
    ) -> Result<(), RslabError> {
        lu.factored = false;
        let storage = (lu.plan_l.take_storage(), lu.plan_u.take_storage());
        let (l, ut, factors, nnz, mut diagnostics, opts) =
            self.numeric(a, opts, Some(storage), &lu.pools)?;
        let t = crate::clock::Instant::now();
        lu.plan_l
            .refill(l, &factors.supernode_parent, true, opts.solve);
        lu.plan_u
            .refill(ut, &factors.supernode_parent, false, opts.solve);
        diagnostics.push(
            "solve-layout",
            t.elapsed().as_secs_f64() * 1e3,
            0,
            (lu.plan_l.bytes() + lu.plan_u.bytes()) as u64,
        );
        lu.factors = factors;
        lu.nnz = nnz;
        lu.diagnostics = diagnostics;
        lu.solve_threads = opts.threads;
        lu.factored = true;
        Ok(())
    }

    /// The numeric factorization both [`factor`](Self::factor) and
    /// [`refactor`](Self::refactor) run: the panels of `L` and `U^T` (in
    /// `storage` when given), the pivots, the fill, the diagnostics and the
    /// settings pinned to the resolved thread count.
    #[allow(clippy::type_complexity)]
    fn numeric<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
        storage: Option<(PanelStorage<T>, PanelStorage<T>)>,
        pools: &super::node::LuPools<T>,
    ) -> Result<
        (
            PanelFactor<T>,
            PanelFactor<T>,
            LuPivots,
            usize,
            crate::diagnostics::Diagnostics,
            SolverSettings,
        ),
        RslabError,
    > {
        let estimate = self.estimate_memory::<T>();
        let resolved_threads = opts.threads.resolve(|cap| {
            crate::numeric::supernodal::analysis::auto_threads(&self.symb, &estimate, cap)
        });
        let opts = opts.pinned(resolved_threads);
        let warnings = opts.ignored_on(crate::numeric::settings::FactorPath::Lu);
        for w in &warnings {
            crate::logging::warn(&format!("lu settings: {w}"));
        }
        let t = crate::clock::Instant::now();
        let numeric = factor_general_lu_numeric(self, a, &opts, storage, pools)?;
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
        Ok((l, ut, factors, nnz as usize, diagnostics, opts))
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
        // Cache per scalar kind (size and complexity): the estimate is a pure
        // function of the structure and the scalar type, and `tuned` + phased
        // `factor` ask for it repeatedly.
        let key = (std::mem::size_of::<T>(), T::COMPLEX);
        if let Ok(cache) = self.est_cache.lock() {
            if let Some(&(_, est)) = cache.iter().find(|&&(k, _)| k == key) {
                return est;
            }
        }
        let est = self.estimate_memory_for::<T>();
        if let Ok(mut cache) = self.est_cache.lock() {
            if !cache.iter().any(|&(k, _)| k == key) {
                cache.push((key, est));
            }
        }
        est
    }

    /// The uncached estimate body, a pure function of the symbolic structure
    /// and the scalar type. Its peak is the [`memory_plan`](Self::memory_plan)
    /// on all cores, the most scratch the kernels can hold.
    fn estimate_memory_for<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        let value_bytes = std::mem::size_of::<T>();
        let Some((sym, _)) = self.symb.sym_and_levels() else {
            return crate::diagnostics::MemoryEstimate {
                value_bytes,
                ..Default::default()
            };
        };
        // The `L` and `U^T` panels of a supernode, `(w + m_l) x w` and
        // `(w + m_u) x w` on the exact structure; they are the stored factor.
        let st = &self.structure;
        let nsuper = sym.supernodes.len();
        let panels: u64 = (0..nsuper)
            .map(|s| ((st.rows_l(s).len() + st.cols_u(s).len()) * sym.supernodes[s].ncol) as u64)
            .sum();
        let settings = SolverSettings {
            threads: crate::numeric::settings::Threads::Fixed(0),
            ..SolverSettings::default()
        };
        let plan = self.memory_plan::<T>(&settings, 1);
        crate::diagnostics::MemoryEstimate {
            value_bytes,
            factor_nnz: panels,
            factor_bytes: plan.factor_bytes,
            panels_all_bytes: panels * value_bytes as u64,
            panel_live_peak_bytes: panels * value_bytes as u64,
            transient_peak_bytes: plan.peak_bytes(),
            factor_flops: (0..nsuper)
                .map(|s| {
                    let nc = sym.supernodes[s].ncol as u64;
                    (st.rows_l(s).len() * st.cols_u(s).len()) as u64 * nc
                })
                .sum(),
            critical_path_flops: 0,
            max_tree_width: 0,
        }
    }

    /// The heap a factorization of scalar type `T` under `opts` needs, and a
    /// solve of `nrhs` right-hand sides after it, predicted from this
    /// analysis before any numeric work: for a preflight check
    /// ([`MemoryPlan::fits_in`](crate::MemoryPlan::fits_in)) and for
    /// scheduling factorizations side by side. The kernels' scratch grows
    /// with the worker count, so plan with the threads the factorization
    /// will run on.
    pub fn memory_plan<T: Scalar>(&self, opts: &SolverSettings, nrhs: usize) -> crate::MemoryPlan {
        let threads = opts.resolved_threads();
        let mut plan = crate::MemoryPlan {
            threads,
            nrhs,
            ..Default::default()
        };
        let (Some((sym, _)), Some(sched)) = (self.symb.sym_and_levels(), self.symb.ll_schedule())
        else {
            return plan;
        };
        // After the schedule above is built: it is part of the analysis.
        plan.analysis_bytes = self.heap_bytes();
        let vb = std::mem::size_of::<T>();
        let k = opts.kernel().k;
        let st = &self.structure;
        let (n, nnz, ns) = (sym.n, self.nnz, sym.supernodes.len());
        let w = |s: usize| sym.supernodes[s].ncol;
        let widths: Vec<usize> = (0..ns).map(w).collect();
        let off_l: Vec<usize> = (0..ns).map(|s| st.rows_l(s).len() - w(s)).collect();
        let off_u: Vec<usize> = (0..ns).map(|s| st.cols_u(s).len() - w(s)).collect();
        let panels: usize = (0..ns)
            .map(|s| (2 * w(s) + off_l[s] + off_u[s]) * w(s))
            .sum();
        let rows: usize = off_l.iter().sum::<usize>() + off_u.iter().sum::<usize>();
        let kept: Vec<bool> = sym.supernodes.iter().map(|sn| sn.ncol > 0).collect();
        let parent = crate::symbolic::supernode_parents(&sym.supernodes, &kept);
        let layout = |off: &[usize]| {
            crate::numeric::supernodal::solve::layout_size(
                &parent,
                &widths,
                off,
                &opts.solve,
                crate::numeric::settings::all_cores(),
            )
        };
        let (layout_l, layout_u) = (layout(&off_l), layout(&off_u));
        // The input program the first factorization keeps (two index arrays
        // and the positions, two pointer arrays), and the peak of its build.
        let (growth, growth_build) = if self.input.get().is_some() {
            (0, 0)
        } else {
            (12 * nnz + 16 * (n + 1), 28 * nnz + 48 * n)
        };
        // Held by the factor: the `L` and `U^T` panels with their schedules,
        // `U`'s reciprocal diagonal, both permutations and scalings, and the
        // supernode tree.
        let factor = vb * (panels + n) + (layout_l.held + layout_u.held) as usize + 32 * n + 8 * ns;
        // While the forest is factored: the scalings, the permuted values,
        // both arenas, the emit cells (refcounts, offsets, panel records,
        // positions and permutations) and the forest's lists, every node's
        // row permutation and emitted rows until the end, the workers'
        // scratch and the heaviest set of nodes running at once.
        let cells = ns * (8 + 8 + 80 + 24 + 8 + 8 + 8) + 32 * n;
        let slots = 8 * (0..ns).map(|s| st.rows_l(s).len()).sum::<usize>() + 4 * rows;
        let scratch: Vec<(u64, usize, usize)> = (0..ns)
            .map(|s| super::node::lu_node_scratch::<T>(s, sym, sched, st, &k, threads))
            .collect();
        let mut planes: Vec<(usize, usize)> = scratch.iter().map(|&(_, p, c)| (p, c)).collect();
        let (workers, kept) = crate::memory::worker_bytes::<T>(n, 2, threads, &mut planes);
        // The node kernel's, the tiled `cmod`'s and the arena's pools.
        let largest = scratch.iter().map(|&(b, _, _)| b).max().unwrap_or(0);
        let pooled = crate::memory::pooled_bytes(3, threads, largest);
        let nodes = crate::memory::concurrent_peak(sym, threads, |s| scratch[s].0);
        let during =
            (16 * n + vb * nnz + vb * panels + cells + slots) as u64 + workers + pooled + nodes;
        // After it: the factor, with the emitted row lists until the plans
        // have copied them, and a plan's build.
        let after =
            factor as u64 + (4 * rows + 48 * ns) as u64 + layout_l.build.max(layout_u.build);
        plan.analysis_growth_bytes = growth as u64;
        plan.factor_bytes = factor as u64;
        plan.kept_bytes = kept + pooled;
        plan.factor_peak_bytes = (growth_build as u64).max(growth as u64 + during.max(after))
            + crate::memory::BOOKKEEPING;
        // The permuted right-hand sides, the solution and the larger of the
        // two sweeps' scratch.
        plan.solve_bytes = (nrhs * vb) as u64
            * (2 * n as u64 + layout_l.solve_per_rhs.max(layout_u.solve_per_rhs));
        plan
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
    /// Whether it holds a factor: a failed [`LuSymbolic::refactor`] leaves
    /// it without one.
    factored: bool,
    /// The kernels' scratch: empty after [`LuSymbolic::factor`], grown by
    /// the first refactorization and kept for the next ones, which then
    /// allocate nothing of their own.
    pools: super::node::LuPools<T>,
}

impl<T: Scalar> LuSolver<T> {
    /// Heap bytes this factor holds: its values in solve layout, the pivots,
    /// the permutations and the solve schedule. `L` and `U^T` each have their own.
    pub fn heap_bytes(&self) -> u64 {
        self.plan_l.heap_bytes() + self.plan_u.heap_bytes() + self.factors.heap_bytes()
    }

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
    fn solve_raw_into(
        &self,
        b: &[T],
        nrhs: usize,
        transpose: bool,
        x: &mut [T],
        work: &mut crate::SolveWork<T>,
    ) -> Result<(), RslabError> {
        if !self.factored {
            return Err(RslabError::InvalidInput(
                "the last refactorization failed; refactor before solving".to_string(),
            ));
        }
        let f = &self.factors;
        let n = f.n;
        let (gather, g_scale, scatter, s_scale) = if transpose {
            (&f.perm, &f.d_col, &f.perm_row, &f.d_row)
        } else {
            (&f.perm_row, &f.d_row, &f.perm, &f.d_col)
        };
        // The sweeps take the block row-major: y[e * nrhs + c].
        let y = &mut work.y;
        y.clear();
        y.resize(n * nrhs, T::zero());
        for (e, &orig) in gather.iter().enumerate() {
            let s = T::from_real(g_scale[orig]);
            for c in 0..nrhs {
                y[e * nrhs + c] = b[c * n + orig] * s;
            }
        }
        let plan = &mut work.plan;
        if transpose {
            self.plan_u.forward(nrhs, y, plan);
            self.plan_l.backward(nrhs, y, plan);
        } else {
            self.plan_l.forward(nrhs, y, plan);
            self.plan_u.backward(nrhs, y, plan);
        }
        for (e, &orig) in scatter.iter().enumerate() {
            let s = T::from_real(s_scale[orig]);
            for c in 0..nrhs {
                x[c * n + orig] = y[e * nrhs + c] * s;
            }
        }
        Ok(())
    }
}

crate::numeric::direct::direct_solver!(LuSolver);
