//! High-level generic sparse symmetric direct solver.
//!
//! [`LdltSolver`] wraps the supernodal factorization
//! ([`crate::numeric::ldlt`]) with symmetric equilibration and
//! a convenient factor-once / solve-many interface. It works for both `f64`
//! (real symmetric) and `Complex<f64>` (complex symmetric, PARDISO `mtype 6`).
//!
//! ## Equilibration
//!
//! Before factoring, the matrix is symmetrically scaled `A_hat = D A D` with a
//! **real** diagonal `D = diag(s)`, `s_i = 1/sqrt(r_i)`, where `r_i = max_j |A_ij|` is
//! the row magnitude. This one-pass infinity-norm equilibration improves
//! conditioning and, because it uses off-diagonal magnitudes, tolerates a zero
//! diagonal (common in complex-symmetric and saddle-point systems). Solving
//! `A x = b` becomes: factor `A_hat`, then `x = D * (A_hat^-1 * (D b))`.
//!
//! ## Solves
//!
//! The numeric factorization produces the factor in supernodal panel form
//! (`PanelFactor`: dense column panels per front, one shared row
//! list each), which the solve plan of [`crate::numeric::supernodal::solve`]
//! takes as its storage; the `solve-layout` diagnostics stage is the tree
//! schedule built over it. [`LdltSolver::solve`] and
//! [`LdltSolver::solve_many`] then run tree-parallel sweeps whose result is
//! bit-identical for every thread count.

use super::factor::{factor_numeric, LdltNumeric};
use super::pivots::LdltPivots;
use crate::error::RslabError;
use crate::numeric::settings::SolverSettings;
use crate::numeric::supernodal::analysis::{
    analyze_with as analyze_pattern_with, SupernodalAnalysis,
};
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;

/// A factored sparse symmetric matrix, ready to solve against many right-hand
/// sides. Generic over the scalar field `T` (`f64` or `Complex<f64>`).
pub struct LdltSolver<T> {
    /// Factors of the equilibrated matrix `A_hat = D A D`, in factorization order.
    factors: LdltPivots<T>,
    /// Real symmetric equilibration diagonal `s` (`D = diag(s)`).
    scale: Vec<f64>,
    /// Per-call factor diagnostics (stages, decisions, numeric outcome).
    diagnostics: crate::diagnostics::Diagnostics,
    /// Solve-phase accumulators (every `solve*` call records into them).
    solves: crate::diagnostics::SolveCounter,
    /// The worker policy the factorization ran with, which a Krylov solve
    /// preconditioned by this factor orthogonalizes under.
    solve_threads: crate::numeric::settings::Threads,
    /// The factor `L` (the panels, its only storage) with the tree schedule;
    /// `factors` carries `D`, the permutation and the outcome with empty CSC
    /// arrays.
    pub(crate) plan: crate::numeric::supernodal::solve::SolvePlan<T>,
}

impl<T: Scalar> LdltSolver<T> {
    /// The matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }

    /// Per-call diagnostics for this factorization: measured factor time, fill,
    /// thread budget, and the a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate).
    /// Everything this factorization can tell about itself (see
    /// [`Diagnostics`](crate::Diagnostics)), the solve-phase accumulators
    /// included. A snapshot.
    pub fn diagnostics(&self) -> crate::diagnostics::Diagnostics {
        let mut d = self.diagnostics.clone();
        d.solves = self.solves.snapshot();
        d
    }

    /// Number of stored nonzeros in the global lower-triangular factor `L`
    /// (the fill). The primary sparse memory metric: RLA stores only `L` of
    /// the symmetric factorization, against which a general LU stores both
    /// `L` and `U` of the full (two-triangle) matrix.
    pub fn factor_nnz(&self) -> usize {
        self.diagnostics.factor_nnz as usize
    }

    /// Number of statically perturbed pivots (preconditioner mode). Zero for
    /// an exact factorization. A nonzero count means the stored factor is of a
    /// slightly perturbed `A + E`; solve via iterative refinement / Krylov.
    pub fn n_perturbed(&self) -> usize {
        self.factors.n_perturbed
    }

    /// Inertia (counts of positive/negative/zero eigenvalues) of the factored
    /// matrix. **Exact only for a real symmetric matrix** (`T = f64`/`f32`);
    /// equilibration uses a real diagonal `D > 0`, so the signs are preserved.
    /// For a complex-symmetric matrix the eigenvalues are complex and have no
    /// sign - there it is advisory (classified by each pivot's real part).
    pub fn inertia(&self) -> &crate::inertia::Inertia {
        &self.factors.inertia
    }

    /// Analyze, equilibrate and factor `A = P^T L D L^T P` in one call; for
    /// the *analyze once, factor many* workflow use [`LdltSymbolic`].
    pub fn factor(a: &CscMatrix<T>, opts: &SolverSettings) -> Result<Self, RslabError> {
        LdltSymbolic::analyze(a, opts)?.factor(a, opts)
    }
}

impl<T: Scalar> crate::numeric::direct::SolveCore<T> for LdltSolver<T> {
    const NAME: &'static str = "ldlt";

    fn dim(&self) -> usize {
        self.factors.n
    }

    fn counter(&self) -> &crate::diagnostics::SolveCounter {
        &self.solves
    }

    /// `x = D P (A_hat^-1 (P^T D b))`: the equilibration is fused into the
    /// permutation gather and scatter around the triangular sweeps. `A` is
    /// symmetric, so the transpose is the same solve.
    fn solve_raw(&self, b: &[T], nrhs: usize, _transpose: bool) -> Result<Vec<T>, RslabError> {
        let n = self.factors.n;
        // The sweeps take the block row-major: y[i * nrhs + c].
        let mut y = vec![T::zero(); n * nrhs];
        for (i, &p) in self.factors.perm.iter().enumerate() {
            let sp = T::from_real(self.scale[p]);
            for c in 0..nrhs {
                y[i * nrhs + c] = b[c * n + p] * sp;
            }
        }
        if nrhs == 1 {
            self.plan.solve_in_place(&self.factors, &mut y)?;
        } else {
            self.plan
                .solve_block_in_place(&self.factors, &mut y, nrhs)?;
        }
        let mut x = vec![T::zero(); n * nrhs];
        for (i, &p) in self.factors.perm.iter().enumerate() {
            let sp = T::from_real(self.scale[p]);
            for c in 0..nrhs {
                x[c * n + p] = y[i * nrhs + c] * sp;
            }
        }
        Ok(x)
    }
}

crate::numeric::direct::direct_solver!(LdltSolver);

/// Fast native one-pass inf-norm scaling on a generic (`f64`/`Complex`) matrix:
/// `s_i = 1/sqrt(max_j |A_ij|)`. The [`ScalingStrategy::OnePassInfNorm`] default, kept
/// on the generic type so the shipped path never densifies to a magnitude copy.
fn onepass_scale<T: Scalar>(a: &CscMatrix<T>) -> Vec<f64> {
    let n = a.n;
    let mut row_max = vec![0.0f64; n];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let m = a.values[k].magnitude();
            if m > row_max[i] {
                row_max[i] = m;
            }
            if i != j && m > row_max[j] {
                row_max[j] = m;
            }
        }
    }
    row_max
        .iter()
        .map(|&r| crate::scaling::inv_sqrt_scale_guarded(r))
        .collect()
}

/// Symmetric equilibration `A_hat = D A D` under the chosen [`ScalingStrategy`]:
/// the real scaling `s`, applied while the values are permuted for the
/// factorization, so no scaled copy of `A` is held; `None` for `Identity`.
///
/// The [`OnePassInfNorm`](ScalingStrategy::OnePassInfNorm) default and
/// [`Identity`](ScalingStrategy::Identity) run natively on `T` (no magnitude
/// copy, bit-identical to the historical one-pass); the iterative / matching
/// strategies ([`InfNorm`](ScalingStrategy::InfNorm),
/// [`Mc64Symmetric`](ScalingStrategy::Mc64Symmetric),
/// [`External`](ScalingStrategy::External))
/// route through [`crate::scaling::compute_scaling`] on the `|A|` magnitude
/// pattern (a real `D` derived from magnitudes is the correct congruence for a
/// complex-symmetric `A`). Scaling changes only values, so the sparsity pattern
/// and the a-priori memory estimate are unaffected.
fn equilibration<T: Scalar>(
    a: &CscMatrix<T>,
    strategy: &crate::scaling::ScalingStrategy,
) -> Result<Option<Vec<f64>>, RslabError> {
    use crate::scaling::ScalingStrategy;
    let scale = match strategy {
        ScalingStrategy::OnePassInfNorm => onepass_scale(a),
        ScalingStrategy::Identity => return Ok(None),
        other => {
            // Real magnitude view `|A|` (same pattern) for the f64 scaling machinery.
            let mag = CscMatrix::<f64> {
                n: a.n,
                col_ptr: a.col_ptr.clone(),
                row_idx: a.row_idx.clone(),
                values: a.values.iter().map(|v| v.magnitude()).collect(),
            };
            let (s, _info) = crate::scaling::compute_scaling(&mag, other)?;
            s
        }
    };
    Ok(Some(scale))
}

/// Reusable PARDISO-style **phase-1 analysis** for [`LdltSolver`].
///
/// Analyze a sparsity pattern once, then [`factor`](Self::factor) many value
/// sets that share it - FEM Newton steps, time stepping, or a frequency sweep
/// where only the matrix entries change. The analysis (fill-reducing ordering,
/// supernodes, assembly-tree levels) is the expensive value-independent part;
/// reusing it across factorizations is the core PARDISO efficiency win.
///
/// One analysis serves any scalar field: the same [`LdltSymbolic`] can
/// [`factor`](Self::factor) an `f64` matrix and a `Complex<f64>` matrix that
/// share the pattern.
///
/// ```
/// use rslab::{LdltSymbolic, SolverSettings, CscMatrix};
/// # fn demo(pattern_vals: &[f64], updated_vals: &[f64]) -> Result<(), rslab::RslabError> {
/// let a = CscMatrix::<f64>::from_triplets(2, &[0, 1], &[0, 1], &[2.0, 3.0])?;
/// let analysis = LdltSymbolic::analyze(&a, &SolverSettings::default())?;        // phase 1, once
/// let f1 = analysis.factor(&a, &SolverSettings::default())?; // phase 2/3
/// let _x = f1.solve(&[1.0, 1.0])?;
/// // ... later, same pattern, new values: analysis.factor(&a2, &opts)? ...
/// # Ok(()) }
/// ```
pub struct LdltSymbolic {
    symbolic: SupernodalAnalysis,
    nnz: usize,
    /// Wall time of the analysis and the ordering it was asked for, carried
    /// into the diagnostics of every factorization reusing it.
    analyze_ms: f64,
    requested_ordering: crate::symbolic::OrderingMethod,
    /// [`estimate_memory`](Self::estimate_memory) results, keyed by scalar
    /// size. The estimate is a pure function of the structure and
    /// `size_of::<T>()`, but computing it rebuilds the supernode row
    /// structures, expensive enough to pay only once per scalar type.
    est_cache: std::sync::Mutex<Vec<(usize, crate::diagnostics::MemoryEstimate)>>,
}

impl LdltSymbolic {
    /// Phase 1: analyze the sparsity pattern of `a` under the ordering and
    /// amalgamation settings. The values are ignored, so any matrix with the
    /// target pattern (even a zero-valued template) works.
    pub fn analyze<T: Scalar>(a: &CscMatrix<T>, opts: &SolverSettings) -> Result<Self, RslabError> {
        a.validate()?;
        let t = crate::clock::Instant::now();
        let symbolic = analyze_pattern_with(a.n, &a.col_ptr, &a.row_idx, opts)?;
        let analyze_ms = t.elapsed().as_secs_f64() * 1e3;
        let sym = Self {
            symbolic,
            nnz: a.row_idx.len(),
            analyze_ms,
            requested_ordering: opts.ordering.method,
            est_cache: std::sync::Mutex::new(Vec::new()),
        };
        if crate::logging::enabled(crate::logging::LogLevel::Info) {
            let d = sym.symbolic.decisions(opts.ordering.method);
            crate::logging::info(&format!(
                "ldlt analyze: n={} nnz(A)={} ordering={}{} supernodes={} \
                 max_front={} levels={} {analyze_ms:.1} ms",
                a.n,
                sym.nnz,
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
        Ok(sym)
    }

    /// The analyzed matrix dimension.
    pub fn n(&self) -> usize {
        self.symbolic.n()
    }

    /// Per-supernode frontal dimensions `(ncol, nrow)` of the analyzed pattern.
    /// See `SupernodalAnalysis::front_dims`.
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        self.symbolic.front_dims()
    }

    /// Number of assembly-tree levels (level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.symbolic.n_levels()
    }

    /// Supernode count per assembly-tree level (available tree-parallelism by
    /// depth). See `SupernodalAnalysis::level_widths`.
    pub fn level_widths(&self) -> Vec<usize> {
        self.symbolic.level_widths()
    }

    /// **A-priori** peak-memory estimate for factoring a matrix of scalar type `T`
    /// (LDL^T path) - a pure, deterministic function of the symbolic structure, for
    /// fail-fast / scheduling before any numeric work. See
    /// [`LuSymbolic::estimate_memory`](crate::LuSymbolic::estimate_memory).
    /// Exact symbolic factor fill (nonzeros, from the column counts, x1.2 slack),
    /// the reliable memory-backstop metric. Unlike
    /// [`MemoryEstimate::factor_nnz`](crate::diagnostics::MemoryEstimate::factor_nnz),
    /// which is a dense-supernode *upper bound* that overshoots the real fill
    /// non-uniformly across orderings (so comparing two of them can pick the worse
    /// one), this tracks the actual stored fill and is comparable across orderings.
    pub fn symbolic_factor_nnz(&self) -> usize {
        self.symbolic
            .sym_and_levels()
            .map(|(s, _)| s.factor_nnz)
            .unwrap_or(0)
    }

    pub fn estimate_memory<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        let value_bytes = std::mem::size_of::<T>();
        // The estimate depends on `T` only through `size_of::<T>()`, so cache
        // per scalar size: the row-structure rebuild below is expensive and
        // the auto-tune pipeline asks for the same estimate repeatedly.
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
        let Some((sym, levels)) = self.symbolic.sym_and_levels() else {
            return crate::diagnostics::estimate_left_looking(
                0,
                &|_| 0,
                &|_| 0,
                &|_| &[],
                value_bytes,
                0,
                true,
            );
        };
        let nsuper = sym.supernodes.len();
        let Some(sched) = self.symbolic.ll_schedule() else {
            return crate::diagnostics::estimate_left_looking(
                0,
                &|_| 0,
                &|_| 0,
                &|_| &[],
                value_bytes,
                0,
                true,
            );
        };
        // LDL^T: one dense panel per supernode (no separate U), and the panel
        // *is* the stored factor (no compact copy, see `PanelFactor`); the
        // input copy is a single lower triangle.
        let panel_bytes = |s: usize| -> u64 {
            (sched.rows(s).len() * sym.supernodes[s].ncol * value_bytes) as u64
        };
        let input_bytes = (self.nnz * (value_bytes + 8)) as u64;
        let mut est = crate::diagnostics::estimate_left_looking(
            nsuper,
            &panel_bytes,
            &panel_bytes,
            &|s| sched.updaters(s),
            value_bytes,
            input_bytes,
            true,
        );
        est.factor_flops = (0..nsuper)
            .map(|s| {
                let (nc, nr) = (sym.supernodes[s].ncol as u64, sched.rows(s).len() as u64);
                nr * nr * nc
            })
            .sum();
        // Critical path (Amdahl bound) + tree width for the thread-aware v2 model.
        // Supernodes are in elimination (postorder) order, so children precede their
        // parent and a single forward pass computes the longest leaf-to-root chain.
        let mut crit = vec![0u64; nsuper];
        let mut cp = 0u64;
        for s in 0..nsuper {
            let (nc, nr) = (sym.supernodes[s].ncol as u64, sched.rows(s).len() as u64);
            let ff = nr * nr * nc;
            let cmax = sym.supernodes[s]
                .children
                .iter()
                .map(|&c| crit[c])
                .max()
                .unwrap_or(0);
            crit[s] = ff + cmax;
            if crit[s] > cp {
                cp = crit[s];
            }
        }
        est.critical_path_flops = cp;
        est.max_tree_width = levels.iter().map(|l| l.len()).max().unwrap_or(1) as u64;
        est
    }

    /// Phases 2-3: equilibrate and factor `a`, reusing this analysis. `a` must
    /// carry the same sparsity pattern the analysis was built from (same `n`
    /// and `nnz`), otherwise an [`RslabError::InvalidInput`] is returned.
    pub fn factor<T: Scalar>(
        &self,
        a: &CscMatrix<T>,
        opts: &SolverSettings,
    ) -> Result<LdltSolver<T>, RslabError> {
        a.validate()?;
        let estimate = self.estimate_memory::<T>();
        // The concrete worker count actually used (realizes Threads::Auto).
        let resolved_threads = opts.threads.resolve(|cap| {
            crate::numeric::supernodal::analysis::auto_threads(&self.symbolic, &estimate, cap)
        });
        let opts = &opts.pinned(resolved_threads);
        let warnings = opts.ignored_on(crate::numeric::settings::FactorPath::Ldlt);
        for w in &warnings {
            crate::logging::warn(&format!("ldlt settings: {w}"));
        }
        let t = crate::clock::Instant::now();
        let scale = equilibration(a, &opts.scaling)?;
        let scale_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = crate::clock::Instant::now();
        let numeric = factor_numeric(&self.symbolic, a, scale.as_deref(), opts)?;
        let scale = scale.unwrap_or_else(|| vec![1.0; a.n]);
        let factor_nnz = (numeric.factor.nnz() - numeric.n_zeros) as u64;
        let factor_bytes = numeric.factor.bytes() as u64;
        let LdltNumeric {
            factor,
            pivots: factors,
            ..
        } = numeric;
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut decisions = self.symbolic.decisions(self.requested_ordering);
        decisions.scaling = format!("{:?}", opts.scaling);
        decisions.method = "LeftLooking".to_string();
        let mut diagnostics = crate::diagnostics::Diagnostics {
            threads: resolved_threads,
            n: a.n,
            nnz_a: self.nnz as u64,
            factor_nnz,
            estimate: Some(estimate),
            decisions,
            numeric: crate::diagnostics::NumericReport {
                perturbed: factors.n_perturbed,
                two_by_two: Some(factors.two_by_two.iter().filter(|&&b| b).count()),
                inertia: Some((
                    factors.inertia.positive,
                    factors.inertia.negative,
                    factors.inertia.zero,
                )),
            },
            warnings,
            ..Default::default()
        };
        // Bytes per stored entry: the scalar value plus its usize row index.
        diagnostics.push("analyze", self.analyze_ms, 0, 0);
        diagnostics.push("scale", scale_ms, 0, 0);
        diagnostics.push("factor", factor_ms, estimate.factor_flops, factor_bytes);
        if crate::logging::enabled(crate::logging::LogLevel::Info) {
            crate::logging::info(&format!("ldlt factor: {}", diagnostics.summary()));
        }
        // Solve layout: supernodal panels plus the tree schedule; the CSC
        // arrays are released so the factor is held once.
        let t = crate::clock::Instant::now();
        let plan = crate::numeric::supernodal::solve::SolvePlan::from_panels(
            factor,
            &factors.supernode_parent,
            true,
            opts.solve,
        );
        diagnostics.push(
            "solve-layout",
            t.elapsed().as_secs_f64() * 1e3,
            0,
            plan.bytes() as u64,
        );
        Ok(LdltSolver {
            factors,
            scale,
            diagnostics,
            solves: Default::default(),
            solve_threads: opts.threads,
            plan,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    /// 7-point 3D grid Laplacian (SPD-shifted), the canonical pattern where
    /// nested dissection beats minimum degree.
    fn grid3d(k: usize) -> CscMatrix<f64> {
        let n = k * k * k;
        let idx = |x: usize, y: usize, z: usize| (z * k + y) * k + x;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for z in 0..k {
            for y in 0..k {
                for x in 0..k {
                    let i = idx(x, y, z);
                    rows.push(i);
                    cols.push(i);
                    vals.push(6.5);
                    for (nb, is_lower) in [
                        (x.checked_sub(1).map(|xx| idx(xx, y, z)), true),
                        (y.checked_sub(1).map(|yy| idx(x, yy, z)), true),
                        (z.checked_sub(1).map(|zz| idx(x, y, zz)), true),
                    ] {
                        if let (Some(j), true) = (nb, is_lower) {
                            rows.push(i);
                            cols.push(j);
                            vals.push(-1.0);
                        }
                    }
                }
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    /// The ordering race must never return a pick with worse exact symbolic
    /// fill than the plain AMD default (it includes AMD as a candidate and
    /// selects by exact fill), and the pick must factor + solve correctly.
    #[test]
    fn race_is_pareto_on_plain_grid() {
        let a = grid3d(24); // n = 13824
        let sym_amd = LdltSymbolic::analyze(
            &a,
            &SolverSettings::default().with_ordering(crate::symbolic::OrderingMethod::Amd),
        )
        .unwrap();
        let amd_fill = sym_amd.symbolic_factor_nnz();

        let s_pick = SolverSettings::default();
        let sym_pick = LdltSymbolic::analyze(&a, &s_pick).unwrap();
        assert!(
            sym_pick.symbolic_factor_nnz() <= amd_fill,
            "fill regressed: {} > {amd_fill}",
            sym_pick.symbolic_factor_nnz()
        );
        let solver = sym_pick.factor(&a, &s_pick).unwrap();
        let b: Vec<f64> = (0..a.n).map(|i| (i % 7) as f64 - 3.0).collect();
        let x = solver.solve(&b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-8);
    }

    /// End-to-end guarantee on the default analysis for a large curl-curl
    /// system: the ordering race must realise the nested-dissection-class win
    /// over the AMD default - this is the regression that cost 10x factor
    /// time in the rapidfem FEM sweep.
    #[cfg(feature = "matgen")]
    #[test]
    fn race_finds_nd_class_win_on_curl_curl() {
        let a = crate::matgen::fem::curl_curl(&[22, 22, 22], 0.8, 0.1); // n = 31944
        let sym_amd = LdltSymbolic::analyze(
            &a,
            &SolverSettings::default().with_ordering(crate::symbolic::OrderingMethod::Amd),
        )
        .unwrap();
        let amd_fill = sym_amd.symbolic_factor_nnz();

        let sym = LdltSymbolic::analyze(&a, &SolverSettings::default()).unwrap();
        eprintln!(
            "curl_curl fill {} (amd {amd_fill})",
            sym.symbolic_factor_nnz()
        );
        assert!(
            (sym.symbolic_factor_nnz() as f64) < amd_fill as f64 * 0.75,
            "race missed the ND-class fill win"
        );
    }

    fn residual_inf<T: Scalar>(a: &CscMatrix<T>, x: &[T], b: &[T]) -> f64 {
        let mut ax = vec![T::zero(); a.n];
        a.symv(x, &mut ax);
        (0..a.n)
            .map(|i| (ax[i] - b[i]).magnitude())
            .fold(0.0, f64::max)
    }

    #[test]
    fn f64_badly_scaled_diagonal() {
        // Diagonal entries spanning ~10 orders of magnitude. Equilibration
        // should keep the solve accurate on the original system.
        let n = 12;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(10.0_f64.powi(j as i32 - 6)); // 1e-6 .. 1e5
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(1.0);
            }
        }
        let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
        let solver = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
        let x = solver.solve(&b).unwrap();
        // Relative residual (the absolute one is dominated by the 1e5 row).
        let mut ax = vec![0.0; n];
        a.symv(&x, &mut ax);
        let rel = (0..n)
            .map(|i| (ax[i] - b[i]).abs() / b[i].abs().max(1.0))
            .fold(0.0, f64::max);
        assert!(rel < 1e-10, "relative residual {}", rel);
    }

    #[test]
    fn critical_path_and_thread_aware_runtime() {
        // 3D grid: a deep assembly tree, so the critical path is a real fraction of
        // the total work. The estimate must populate a positive critical path that
        // is a subset of the total flops, a tree width >= 1, and the thread-aware
        // runtime must never fall below the Amdahl serial-critical-path floor no
        // matter how large the speedup argument.
        let m = 12;
        let n = m * m * m;
        let idx = |a: usize, b: usize, c: usize| (a * m + b) * m + c;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                for c in 0..m {
                    let p = idx(a, b, c);
                    r.push(p);
                    cc.push(p);
                    v.push(6.0_f64);
                    if c + 1 < m {
                        r.push(idx(a, b, c + 1));
                        cc.push(p);
                        v.push(-1.0);
                    }
                    if b + 1 < m {
                        r.push(idx(a, b + 1, c));
                        cc.push(p);
                        v.push(-1.0);
                    }
                    if a + 1 < m {
                        r.push(idx(a + 1, b, c));
                        cc.push(p);
                        v.push(-1.0);
                    }
                }
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
        let sym = LdltSymbolic::analyze(&a, &SolverSettings::default()).unwrap();
        let est = sym.estimate_memory::<f64>();
        assert!(est.critical_path_flops > 0, "critical path populated");
        assert!(
            est.critical_path_flops <= est.factor_flops,
            "critical path {} is a subset of total flops {}",
            est.critical_path_flops,
            est.factor_flops
        );
        assert!(est.max_tree_width >= 1, "tree width populated");
        // Amdahl floor: at a huge speedup the parallel term vanishes but the serial
        // critical path remains, so the thread-aware estimate stays >= that floor.
        let rate1 = 2.0; // gflops
        let floor_ms = est.critical_path_flops as f64 / (rate1 * 1e9) * 1e3;
        let t_huge = est.est_runtime_ms_threaded(rate1, 1e9);
        assert!(
            (t_huge - floor_ms).abs() < floor_ms * 1e-6 + 1e-9,
            "thread-aware runtime hits the critical-path floor: {t_huge} vs {floor_ms}"
        );
        // The plain model would keep shrinking with speedup (no floor).
        assert!(
            est.est_runtime_ms(rate1, 1e9) < floor_ms,
            "plain model has no Amdahl floor"
        );
    }

    #[test]
    fn scaling_strategy_knob_all_variants_solve() {
        use crate::scaling::ScalingStrategy;
        // Well-conditioned SPD tridiagonal-plus-grid: every equilibration strategy
        // (and Identity/off) must factor and solve to a tiny residual, proving the
        // knob is threaded end-to-end (SolverSettings.scaling -> equilibration ->
        // compute_scaling). The default OnePassInfNorm stays bit-identical.
        let m = 6;
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
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let sym = LdltSymbolic::analyze(&a, &SolverSettings::default()).unwrap();
        for strat in [
            ScalingStrategy::OnePassInfNorm,
            ScalingStrategy::Identity,
            ScalingStrategy::InfNorm,
            ScalingStrategy::Mc64Symmetric,
        ] {
            let opts = SolverSettings::default().with_scaling(strat.clone());
            let solver = sym.factor(&a, &opts).unwrap();
            let x = solver.solve(&b).unwrap();
            let res = residual_inf(&a, &x, &b);
            assert!(res < 1e-9, "strategy {strat:?} residual {res}");
        }
        // Default preserves the historical one-pass scaling exactly.
        assert_eq!(
            SolverSettings::default().scaling,
            ScalingStrategy::OnePassInfNorm
        );
    }

    #[test]
    fn ldlt_solve_many_matches_single() {
        let n = 7;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            r.push(j);
            cc.push(j);
            v.push(4.0_f64);
            if j + 1 < n {
                r.push(j + 1);
                cc.push(j);
                v.push(-1.0);
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
        let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
        let nrhs = 4;
        // Column-major B.
        let b: Vec<f64> = (0..n * nrhs).map(|k| (k % 5) as f64 - 2.0).collect();
        let x = f.solve_many(&b, nrhs).unwrap();
        for c in 0..nrhs {
            let bc: Vec<f64> = (0..n).map(|i| b[c * n + i]).collect();
            let xc = f.solve(&bc).unwrap();
            for i in 0..n {
                assert!((x[c * n + i] - xc[i]).abs() < 1e-10, "rhs {c} row {i}");
            }
        }
    }

    #[test]
    fn f64_inertia_diagonal_signs() {
        // Pure diagonal -> all 1x1 pivots; (positive) equilibration preserves
        // signs, so the inertia is the diagonal's signature.
        let diag = [2.0_f64, -3.0, 4.0, -1.0, 5.0];
        let n = diag.len();
        let (rows, cols): (Vec<_>, Vec<_>) = (0..n).map(|i| (i, i)).unzip();
        let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &diag).unwrap();
        let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
        let inertia = f.inertia();
        assert_eq!(
            (inertia.positive, inertia.negative, inertia.zero),
            (3, 2, 0)
        );
        assert_eq!(inertia.total(), n);
    }

    #[test]
    fn f64_inertia_indefinite_2x2() {
        // [[0,1],[1,0]] has eigenvalues +/-1 -> Bunch-Kaufman takes one 2x2 block
        // with det < 0, classified as one positive + one negative.
        let a = CscMatrix::<f64>::from_triplets(2, &[0, 1], &[0, 0], &[0.0, 1.0]).unwrap();
        let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
        assert_eq!(
            (f.inertia().positive, f.inertia().negative, f.inertia().zero),
            (1, 1, 0)
        );
    }

    #[test]
    fn phased_analyze_then_factor_many_matches_one_shot() {
        // PARDISO workflow: analyze the pattern once, factor two different
        // value sets that share it. Each must match the one-shot factor and
        // solve its own system - the FEM Newton / frequency-sweep use case.
        let c = |re, im| Complex::new(re, im);
        let n = 8;
        let (mut rows, mut cols) = (Vec::new(), Vec::new());
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
            }
        }
        // Pattern template (values irrelevant for analysis).
        let template = CscMatrix::<Complex<f64>>::from_triplets(
            n,
            &rows,
            &cols,
            &vec![c(1.0, 0.0); rows.len()],
        )
        .unwrap();
        let analysis = LdltSymbolic::analyze(&template, &SolverSettings::default()).unwrap();
        assert_eq!(analysis.n(), n);

        for shift in [0.0, 2.0, -1.5] {
            // Same pattern, different values.
            let vals: Vec<Complex<f64>> = rows
                .iter()
                .zip(&cols)
                .map(|(&i, &j)| {
                    if i == j {
                        c(4.0 + shift, 1.0)
                    } else {
                        c(-1.0, 0.2)
                    }
                })
                .collect();
            let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
            let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 4.0, 1.0)).collect();

            let phased = analysis.factor(&a, &SolverSettings::default()).unwrap();
            let one_shot = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
            let x_phased = phased.solve(&b).unwrap();
            let x_one = one_shot.solve(&b).unwrap();

            // Same factor -> identical solve.
            for (p, o) in x_phased.iter().zip(&x_one) {
                assert!((p - o).norm() < 1e-12);
            }
            assert!(residual_inf(&a, &x_phased, &b) < 1e-9);
        }
    }

    #[test]
    fn auto_threads_wiring() {
        // A thin tridiagonal: the predictor caps it low (no parallelism source),
        // a fixed budget overrides exactly, and the cap clamps.
        let n = 3000;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            r.push(j);
            cc.push(j);
            v.push(4.0_f64);
            if j + 1 < n {
                r.push(j + 1);
                cc.push(j);
                v.push(-1.0);
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
        let sym = LdltSymbolic::analyze(&a, &SolverSettings::default()).unwrap();
        // Auto capped at 8: a tridiagonal is thin/narrow -> policy returns 2.
        let auto = sym
            .factor(
                &a,
                &SolverSettings::default().with_threads(crate::Threads::Auto { max: 8 }),
            )
            .unwrap();
        assert_eq!(
            auto.diagnostics().threads,
            2,
            "thin matrix auto-capped to 2"
        );
        // Fixed overrides the predictor exactly.
        let fixed = sym
            .factor(&a, &SolverSettings::default().with_threads(5))
            .unwrap();
        assert_eq!(fixed.diagnostics().threads, 5);
        // The auto cap clamps the prediction.
        let cap1 = sym
            .factor(
                &a,
                &SolverSettings::default().with_threads(crate::Threads::Auto { max: 1 }),
            )
            .unwrap();
        assert_eq!(cap1.diagnostics().threads, 1);
        // All still solve correctly.
        let b = vec![1.0_f64; n];
        assert!(auto.solve(&b).is_ok() && fixed.solve(&b).is_ok());
    }

    #[test]
    fn analyze_options_default_matches_bare_analyze() {
        // The composable default must reproduce the bare analyze exactly: same
        // symbolic shape (fill), so existing callers are unaffected.
        let n = 200;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            r.push(j);
            c.push(j);
            v.push(4.0_f64);
            if j + 1 < n {
                r.push(j + 1);
                c.push(j);
                v.push(-1.0);
            }
        }
        let a = CscMatrix::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let bare = LdltSymbolic::analyze(&a, &SolverSettings::default()).unwrap();
        let with_default = LdltSymbolic::analyze(&a, &crate::SolverSettings::default()).unwrap();
        assert_eq!(bare.front_dims(), with_default.front_dims());
        assert_eq!(bare.level_widths(), with_default.level_widths());
    }

    #[test]
    fn analyze_with_alternative_knobs_still_solves() {
        // Changing ordering / nemin / relax changes the symbolic shape but must
        // still produce a correct factorization.
        let c = |re, im| Complex::new(re, im);
        let m = 8;
        let n = m * m;
        let idx = |r: usize, cc: usize| r * m + cc;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..m {
            for cc in 0..m {
                let p = idx(r, cc);
                rows.push(p);
                cols.push(p);
                vals.push(c(4.0, 0.5));
                for (dr, dc) in [(1usize, 0usize), (0, 1)] {
                    if r + dr < m && cc + dc < m {
                        let q = idx(r + dr, cc + dc);
                        let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                        rows.push(hi);
                        cols.push(lo);
                        vals.push(c(-1.0, 0.1));
                    }
                }
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 30.0, 1.0)).collect();
        for opts in [
            crate::SolverSettings::default().with_ordering(crate::OrderingMethod::Amd),
            crate::SolverSettings::default().with_ordering(crate::OrderingMethod::MetisND),
            crate::SolverSettings::default().with_nemin(1),
            crate::SolverSettings::default().with_relax(None),
        ] {
            let f = LdltSymbolic::analyze(&a, &opts)
                .unwrap()
                .factor(&a, &SolverSettings::default())
                .unwrap();
            let x = f.solve(&b).unwrap();
            assert!(
                residual_inf(&a, &x, &b) < 1e-9,
                "opts {opts:?} residual too large"
            );
        }
    }

    #[test]
    fn analysis_rejects_mismatched_pattern() {
        let a =
            CscMatrix::<f64>::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 2.0, 2.0]).unwrap();
        let analysis = LdltSymbolic::analyze(&a, &SolverSettings::default()).unwrap();
        // A different pattern (extra off-diagonal) must be rejected.
        let a2 = CscMatrix::<f64>::from_triplets(
            3,
            &[0, 1, 1, 2],
            &[0, 0, 1, 2],
            &[2.0, -1.0, 2.0, 2.0],
        )
        .unwrap();
        assert!(analysis.factor(&a2, &SolverSettings::default()).is_err());
    }

    #[test]
    fn complex_grid_solve() {
        let c = |re, im| Complex::new(re, im);
        let m = 6;
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
                    vals.push(c(-1.0, 0.3));
                }
                if r + 1 < m {
                    let q = idx(r + 1, cc);
                    let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                    rows.push(hi);
                    cols.push(lo);
                    vals.push(c(-1.0, 0.3));
                }
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let solver = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();

        // Solve against two different right-hand sides with the one factor.
        for shift in [0.0, 1.0] {
            let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 10.0 + shift, 1.0)).collect();
            let x = solver.solve(&b).unwrap();
            assert!(
                residual_inf(&a, &x, &b) < 1e-9,
                "residual {}",
                residual_inf(&a, &x, &b)
            );
        }
    }

    #[test]
    fn refined_solve_is_no_worse_than_plain() {
        // Complex-symmetric tridiagonal; refinement must not increase the
        // residual and should reach near machine precision.
        let c = |re, im| Complex::new(re, im);
        let n = 30;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(c(3.0, 0.4));
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(c(-1.0, 0.2));
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 15.0, 2.0)).collect();
        let solver = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();

        let x_plain = solver.solve(&b).unwrap();
        let x_ref = solver
            .solve_refined(&a, &b, &crate::RefinePolicy::steps(3))
            .unwrap()
            .0;
        let r_plain = residual_inf(&a, &x_plain, &b);
        let r_ref = residual_inf(&a, &x_ref, &b);
        assert!(
            r_ref <= r_plain.max(1e-300),
            "refined {} vs plain {}",
            r_ref,
            r_plain
        );
        assert!(r_ref < 1e-12, "refined residual {}", r_ref);
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let a = CscMatrix::<f64>::from_triplets(2, &[0, 1], &[0, 1], &[2.0, 3.0]).unwrap();
        let solver = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
        assert!(matches!(
            solver.solve(&[1.0, 2.0, 3.0]),
            Err(RslabError::DimensionMismatch { .. })
        ));
    }
}
