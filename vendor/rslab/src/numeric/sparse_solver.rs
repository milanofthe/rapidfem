//! High-level generic sparse symmetric direct solver.
//!
//! [`LdltSolver`] wraps the generic multifrontal factorization
//! ([`crate::numeric::multifrontal_ldlt`]) with symmetric equilibration and
//! a convenient factor-once / solve-many interface. It works for both `f64`
//! (real symmetric) and `Complex<f64>` (complex symmetric, PARDISO `mtype 6`).
//!
//! ## Equilibration
//!
//! Before factoring, the matrix is symmetrically scaled `Â = D A D` with a
//! **real** diagonal `D = diag(s)`, `s_i = 1/√rᵢ`, where `rᵢ = maxⱼ |Aᵢⱼ|` is
//! the row magnitude. This one-pass infinity-norm equilibration improves
//! conditioning and, because it uses off-diagonal magnitudes, tolerates a zero
//! diagonal (common in complex-symmetric and saddle-point systems). Solving
//! `A x = b` becomes: factor `Â`, then `x = D · (Â⁻¹ · (D b))`.

use crate::dense::ldlt_generic::{solve_ldlt, solve_ldlt_many, LdltFactors};
use crate::error::RslabError;
use crate::numeric::multifrontal_ldlt::{
    analyze as analyze_pattern, analyze_with as analyze_pattern_with, factor_numeric,
    MultifrontalSymbolic, SolverSettings,
};
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;

/// Floor for the [`LdltSolver::tuned`] nested-dissection bakeoff: below this
/// predicted factor cost the numeric phase is seconds at most and the extra
/// symbolic analysis isn't worth scheduling. Above it, a missed ND win on a
/// 3D-mesh pattern costs 10x the analysis price (measured on 87k-DOF
/// Nedelec-2 curl-curl: 2e11 flops AMD vs 1.8e10 MetisND).
const ND_BAKEOFF_MIN_FLOPS: u64 = 5_000_000_000;
/// Small systems never enter the bakeoff regardless of predicted flops -
/// dense-ish small matrices can post huge flops without an ND story.
const ND_BAKEOFF_MIN_N: usize = 10_000;
/// Adopt ND only on a clear predicted win, not a coin flip.
const ND_BAKEOFF_ADOPT_RATIO: f64 = 0.75;

/// A factored sparse symmetric matrix, ready to solve against many right-hand
/// sides. Generic over the scalar field `T` (`f64` or `Complex<f64>`).
pub struct LdltSolver<T> {
    /// Factors of the equilibrated matrix `Â = D A D`, in factorization order.
    factors: LdltFactors<T>,
    /// Real symmetric equilibration diagonal `s` (`D = diag(s)`).
    scale: Vec<f64>,
    /// Per-call factor diagnostics (time + a-priori memory estimate).
    diagnostics: crate::diagnostics::Diagnostics,
}

impl<T: Scalar> LdltSolver<T> {
    /// The matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }

    /// Per-call diagnostics for this factorization: measured factor time, fill,
    /// thread budget, and the a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate).
    pub fn diagnostics(&self) -> &crate::diagnostics::Diagnostics {
        &self.diagnostics
    }

    /// Number of stored nonzeros in the global lower-triangular factor `L`
    /// (the fill). The primary sparse memory metric: RLA stores only `L` of
    /// the symmetric factorization, against which a general LU stores both
    /// `L` and `U` of the full (two-triangle) matrix.
    pub fn factor_nnz(&self) -> usize {
        self.factors.l_values.len()
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

    /// Equilibrate and factor `A` as `Â = D A D = Pᵀ L D_bk Lᵀ P` (exact mode).
    pub fn factor(a: &CscMatrix<T>) -> Result<Self, RslabError> {
        Self::factor_auto(a, crate::auto_tune::DEFAULT_TUNE_WEIGHT)
    }

    /// Auto-tuned factorization at an explicit Pareto `weight` (`1` = fastest,
    /// `0` = smallest peak memory; [`DEFAULT_TUNE_WEIGHT`](crate::auto_tune::DEFAULT_TUNE_WEIGHT)
    /// leans toward speed). Picks the solver settings from the matrix's structural
    /// features via the embedded performance model, **guarded**: it only deviates
    /// from the proven default when a clear, memory-vetoed win is predicted, so it
    /// is never far worse than the default. [`factor`](Self::factor) is this at the
    /// default weight; [`factor_with`](Self::factor_with) opts out (explicit settings).
    pub fn factor_auto(a: &CscMatrix<T>, weight: f64) -> Result<Self, RslabError> {
        let (sym, s) = Self::tuned(a, weight)?;
        sym.factor(a, &s)
    }

    /// The auto-tuner's choice for `a` at Pareto `weight`: the symbolic to factor
    /// with plus the guarded, memory-backstopped [`SolverSettings`]. Runs the
    /// analysis, the model recommendation, and the deterministic memory backstop
    /// (exact fill + realistic floor, never more memory than the default). Shared by
    /// [`factor_auto`](Self::factor_auto) and the benchmark harness so both exercise
    /// identical logic.
    pub fn tuned(
        a: &CscMatrix<T>,
        weight: f64,
    ) -> Result<(LdltSymbolic, SolverSettings), RslabError> {
        let sym = LdltSymbolic::analyze(a)?;
        let est = sym.estimate_memory::<T>();
        let feat = crate::StructuralFeatures::from_symmetric(a, &sym);
        // MF/LL-floor ratio for the veto (the floor is the reliable LL reference).
        let mf_ll = if est.panel_live_peak_bytes > 0 {
            est.mf_transient_peak_bytes as f64 / est.panel_live_peak_bytes as f64
        } else {
            1.0
        };
        let s = crate::auto_tune::recommend_settings_pathed(
            &feat,
            weight,
            mf_ll,
            crate::auto_tune::SolverPath::Ldlt,
        );
        let d = SolverSettings::default();
        // Hard a-priori memory backstop (never more memory than the default). Fill is
        // compared via the *exact* symbolic fill (`symbolic_factor_nnz`), not
        // `MemoryEstimate::factor_nnz`: the latter is a dense-supernode upper bound
        // that overshoots the real fill non-uniformly across orderings, so comparing
        // two of them once let a MetisND+high-nemin pick with 2x the real fill slip
        // through on banded matrices. The realistic transient floor stays under the
        // default's (MF pick vs LL floor, LL pick floor-vs-floor).
        let default_fill = sym.symbolic_factor_nnz();
        let mem_ok =
            |e: &crate::diagnostics::MemoryEstimate, m: crate::FactorMethod, pick_fill: usize| {
                let fill_ok = pick_fill as f64 <= default_fill as f64 * 1.02;
                let flops_ok = e.factor_flops as f64 <= est.factor_flops as f64 * 1.05;
                if m == crate::FactorMethod::Multifrontal {
                    fill_ok && flops_ok && e.mf_transient_peak_bytes <= est.panel_live_peak_bytes
                } else {
                    fill_ok && flops_ok && e.panel_live_peak_bytes <= est.panel_live_peak_bytes
                }
            };
        // Reuse the default analysis unless the tuner changed an analyze-time knob.
        let (sym, s) = if (s.reorder, s.ordering, s.nemin, s.relax)
            == (d.reorder, d.ordering, d.nemin, d.relax)
        {
            if mem_ok(&est, s.method, default_fill) {
                (sym, s)
            } else {
                (sym, d)
            }
        } else {
            let sym2 = LdltSymbolic::analyze_with(a, &s)?;
            let est2 = sym2.estimate_memory::<T>();
            if mem_ok(&est2, s.method, sym2.symbolic_factor_nnz()) {
                (sym2, s)
            } else {
                (sym, d) // memory regression by the estimate -> safe default
            }
        };
        // Large systems: measured nested-dissection bakeoff (see below). The
        // model's corpus cannot see matrix provenance, and on 3D-mesh
        // patterns a minimum-degree pick misses ND wins of 10x in factor
        // time; here the extra analysis is a small fraction of the numeric
        // factorization it can save.
        if a.n >= ND_BAKEOFF_MIN_N
            && sym.estimate_memory::<T>().factor_flops >= ND_BAKEOFF_MIN_FLOPS
        {
            return Self::nd_bakeoff(a, sym, s);
        }
        Ok((sym, s))
    }

    /// Re-analyze with [`OrderingMethod::MetisND`] and keep whichever
    /// ordering the *exact* symbolic quantities favour: ND is adopted only
    /// on a clear predicted-flops win with no regression in exact fill or
    /// in the method-relevant transient peak, so the pick is Pareto-safe at
    /// any tune weight. Deterministic - both candidates are measured on
    /// this matrix, nothing is modeled.
    fn nd_bakeoff(
        a: &CscMatrix<T>,
        sym: LdltSymbolic,
        s: SolverSettings,
    ) -> Result<(LdltSymbolic, SolverSettings), RslabError> {
        use crate::symbolic::OrderingMethod;
        if s.ordering == OrderingMethod::MetisND {
            return Ok((sym, s));
        }
        let mut s_nd = s.clone();
        s_nd.ordering = OrderingMethod::MetisND;
        let sym_nd = match LdltSymbolic::analyze_with(a, &s_nd) {
            Ok(x) => x,
            Err(_) => return Ok((sym, s)), // ND analysis failed -> keep the pick
        };
        let est = sym.estimate_memory::<T>();
        let est_nd = sym_nd.estimate_memory::<T>();
        let peak = |e: &crate::diagnostics::MemoryEstimate| match s.method {
            crate::FactorMethod::Multifrontal => e.mf_transient_peak_bytes,
            _ => e.panel_live_peak_bytes,
        };
        let flops_win =
            (est_nd.factor_flops as f64) < est.factor_flops as f64 * ND_BAKEOFF_ADOPT_RATIO;
        let fill_ok = sym_nd.symbolic_factor_nnz() <= sym.symbolic_factor_nnz();
        let mem_ok = peak(&est_nd) <= peak(&est);
        if flops_win && fill_ok && mem_ok {
            Ok((sym_nd, s_nd))
        } else {
            Ok((sym, s))
        }
    }

    /// Equilibrate and factor `A` with explicit options - notably
    /// static-pivoting (never-fail preconditioner) mode. See
    /// [`SolverSettings`]. Runs analysis + numeric factorization in one
    /// call; for the *analyze once, factor many* workflow use
    /// [`LdltSymbolic`].
    pub fn factor_with(a: &CscMatrix<T>, opts: &SolverSettings) -> Result<Self, RslabError> {
        LdltSymbolic::analyze(a)?.factor(a, opts)
    }

    /// Solve `A · x = rhs` using the stored factors.
    pub fn solve(&self, rhs: &[T]) -> Result<Vec<T>, RslabError> {
        let n = self.factors.n;
        if rhs.len() != n {
            return Err(RslabError::DimensionMismatch {
                expected: n,
                got: rhs.len(),
            });
        }
        // b̂ = D b
        let b_hat: Vec<T> = rhs
            .iter()
            .zip(&self.scale)
            .map(|(&r, &s)| r * T::from_real(s))
            .collect();
        // ẑ = Â⁻¹ b̂
        let mut x = solve_ldlt(&self.factors, &b_hat)?;
        // x = D ẑ
        for (xi, &s) in x.iter_mut().zip(&self.scale) {
            *xi = *xi * T::from_real(s);
        }
        Ok(x)
    }

    /// Solve `A · X = B` for `nrhs` right-hand sides at once. `b` and the
    /// returned `x` are **row-major** `n × nrhs` buffers (`b[i*nrhs + c]` is
    /// RHS `c` at row `i`). Faster than `nrhs` separate [`solve`](Self::solve)
    /// calls - the factor structure is traversed once and each value applied to
    /// all RHS (the FEM multiple-load-case / block-Krylov use).
    pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
        let n = self.factors.n;
        if nrhs == 0 || b.len() != n * nrhs {
            return Err(RslabError::DimensionMismatch {
                expected: n * nrhs,
                got: b.len(),
            });
        }
        // B̂ = D B (real diagonal scale per row, applied to every RHS column).
        let mut b_hat = b.to_vec();
        for i in 0..n {
            let s = T::from_real(self.scale[i]);
            for c in 0..nrhs {
                b_hat[i * nrhs + c] = b_hat[i * nrhs + c] * s;
            }
        }
        let mut x = solve_ldlt_many(&self.factors, &b_hat, nrhs)?;
        // X = D X̂
        for i in 0..n {
            let s = T::from_real(self.scale[i]);
            for c in 0..nrhs {
                x[i * nrhs + c] = x[i * nrhs + c] * s;
            }
        }
        Ok(x)
    }

    /// Solve `A · x = rhs` with iterative refinement against the original
    /// matrix `a` (which must be the matrix this was factored from). Each step
    /// computes the residual `r = rhs − A x` and applies the correction
    /// `x ← x + A⁻¹ r`, stopping once `‖r‖∞` stops improving or `max_iter` is
    /// reached. This recovers accuracy lost to the within-fully-summed-block
    /// pivoting on harder indefinite systems, at the cost of a few extra solves.
    pub fn solve_refined(
        &self,
        a: &CscMatrix<T>,
        rhs: &[T],
        max_iter: usize,
    ) -> Result<Vec<T>, RslabError> {
        let n = self.factors.n;
        if a.n != n {
            return Err(RslabError::DimensionMismatch {
                expected: n,
                got: a.n,
            });
        }
        let mut x = self.solve(rhs)?;
        let mut ax = vec![T::zero(); n];
        let mut best_x = x.clone();
        let mut best_res = f64::INFINITY;
        // `max_iter` correction steps, plus the initial residual evaluation.
        for _ in 0..=max_iter {
            a.symv(&x, &mut ax);
            let r: Vec<T> = rhs.iter().zip(&ax).map(|(&b, &axi)| b - axi).collect();
            let res = r.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            // Track the best iterate - refinement can be non-monotone on very
            // ill-conditioned systems.
            if res < best_res {
                best_res = res;
                best_x.clone_from(&x);
            }
            if res == 0.0 {
                break;
            }
            let dx = self.solve(&r)?;
            for (xi, &d) in x.iter_mut().zip(&dx) {
                *xi = *xi + d;
            }
        }
        Ok(best_x)
    }
}

/// Apply a symmetric real scaling `Â = D A D`, `D = diag(scale)` (user-order),
/// producing the scaled matrix with the identical pattern.
fn apply_symmetric_scaling<T: Scalar>(a: &CscMatrix<T>, scale: &[f64]) -> CscMatrix<T> {
    let mut scaled_values = Vec::with_capacity(a.values.len());
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            scaled_values.push(a.values[k] * T::from_real(scale[i] * scale[j]));
        }
    }
    CscMatrix::<T> {
        n: a.n,
        col_ptr: a.col_ptr.clone(),
        row_idx: a.row_idx.clone(),
        values: scaled_values,
    }
}

/// Fast native one-pass ∞-norm scaling on a generic (`f64`/`Complex`) matrix:
/// `sᵢ = 1/√maxⱼ|Aᵢⱼ|`. The [`ScalingStrategy::OnePassInfNorm`] default, kept
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
        .map(|&r| if r > 0.0 { 1.0 / r.sqrt() } else { 1.0 })
        .collect()
}

/// Symmetric equilibration `Â = D A D` under the chosen [`ScalingStrategy`].
/// Returns the scaled matrix (identical pattern) and the real scaling `s`.
///
/// The [`OnePassInfNorm`](ScalingStrategy::OnePassInfNorm) default and
/// [`Identity`](ScalingStrategy::Identity) run natively on `T` (no magnitude
/// copy, bit-identical to the historical one-pass); the iterative / matching
/// strategies ([`InfNorm`](ScalingStrategy::InfNorm),
/// [`Mc64Symmetric`](ScalingStrategy::Mc64Symmetric),
/// [`Auto`](ScalingStrategy::Auto), [`External`](ScalingStrategy::External))
/// route through [`crate::scaling::compute_scaling`] on the `|A|` magnitude
/// pattern (a real `D` derived from magnitudes is the correct congruence for a
/// complex-symmetric `A`). Scaling changes only values, so the sparsity pattern
/// and the a-priori memory estimate are unaffected.
fn equilibrate_with<T: Scalar>(
    a: &CscMatrix<T>,
    strategy: &crate::scaling::ScalingStrategy,
) -> Result<(CscMatrix<T>, Vec<f64>), RslabError> {
    use crate::scaling::ScalingStrategy;
    let scale = match strategy {
        ScalingStrategy::OnePassInfNorm => onepass_scale(a),
        ScalingStrategy::Identity => return Ok((a.clone(), vec![1.0; a.n])),
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
    let scaled = apply_symmetric_scaling(a, &scale);
    Ok((scaled, scale))
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
/// let analysis = LdltSymbolic::analyze(&a)?;        // phase 1, once
/// let f1 = analysis.factor(&a, &SolverSettings::default())?; // phase 2/3
/// let _x = f1.solve(&[1.0, 1.0])?;
/// // ... later, same pattern, new values: analysis.factor(&a2, &opts)? ...
/// # Ok(()) }
/// ```
pub struct LdltSymbolic {
    symbolic: MultifrontalSymbolic,
    nnz: usize,
}

impl LdltSymbolic {
    /// Phase 1: analyze the sparsity pattern of `a`. The values are ignored, so
    /// any matrix with the target pattern (even a zero-valued template) works.
    pub fn analyze<T: Scalar>(a: &CscMatrix<T>) -> Result<Self, RslabError> {
        a.validate()?;
        Ok(Self {
            symbolic: analyze_pattern(a.n, &a.col_ptr, &a.row_idx)?,
            nnz: a.row_idx.len(),
        })
    }

    /// [`analyze`](Self::analyze) with explicit composable [`SolverSettings`] -
    /// fill-reducing ordering, supernode amalgamation, child-reordering. The
    /// tunable analysis knobs for the auto-tuning sweep.
    pub fn analyze_with<T: Scalar>(
        a: &CscMatrix<T>,
        opts: &SolverSettings,
    ) -> Result<Self, RslabError> {
        a.validate()?;
        Ok(Self {
            symbolic: analyze_pattern_with(a.n, &a.col_ptr, &a.row_idx, opts)?,
            nnz: a.row_idx.len(),
        })
    }

    /// The analyzed matrix dimension.
    pub fn n(&self) -> usize {
        self.symbolic.n()
    }

    /// Per-supernode frontal dimensions `(ncol, nrow)` of the analyzed pattern.
    /// See [`MultifrontalSymbolic::front_dims`](crate::MultifrontalSymbolic::front_dims).
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        self.symbolic.front_dims()
    }

    /// Number of assembly-tree levels (level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.symbolic.n_levels()
    }

    /// Supernode count per assembly-tree level (available tree-parallelism by
    /// depth). See [`MultifrontalSymbolic::level_widths`](crate::MultifrontalSymbolic::level_widths).
    pub fn level_widths(&self) -> Vec<usize> {
        self.symbolic.level_widths()
    }

    /// **A-priori** peak-memory estimate for factoring a matrix of scalar type `T`
    /// (LDLᵀ path) - a pure, deterministic function of the symbolic structure, for
    /// fail-fast / scheduling before any numeric work. See
    /// [`LuSymbolic::estimate_memory`](crate::LuSymbolic::estimate_memory).
    /// Exact symbolic factor fill (nonzeros, from the column counts, ×1.2 slack) —
    /// the reliable memory-backstop metric. Unlike
    /// [`MemoryEstimate::factor_nnz`](crate::diagnostics::MemoryEstimate::factor_nnz),
    /// which is a dense-supernode *upper bound* that overshoots the real fill
    /// non-uniformly across orderings (so comparing two of them can pick the worse
    /// one), this tracks the actual stored fill and is comparable across orderings.
    pub fn symbolic_factor_nnz(&self) -> usize {
        self.symbolic
            .sym_and_levels()
            .map(|(s, _)| s.factor_nnz_estimate)
            .unwrap_or(0)
    }

    pub fn estimate_memory<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        let value_bytes = std::mem::size_of::<T>();
        let Some((sym, levels)) = self.symbolic.sym_and_levels() else {
            return crate::diagnostics::estimate_left_looking(
                0,
                &|_| 0,
                &|_| 0,
                &[],
                value_bytes,
                0,
            );
        };
        let nsuper = sym.supernodes.len();
        let rs = crate::numeric::multifrontal_ldlt::compute_supernode_row_structures(sym);
        let mut col_to_snode = vec![0usize; sym.n];
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
        // LDLᵀ: one dense panel per supernode (no separate U), and the compact
        // factor is `L` only (no `U`); the input copy is a single lower triangle.
        let panel_bytes =
            |s: usize| -> u64 { (rs[s].len() * sym.supernodes[s].ncol * value_bytes) as u64 };
        let compact_bytes = |s: usize| -> u64 {
            let nc = sym.supernodes[s].ncol;
            let cnrow = rs[s].len() - nc;
            ((nc * (nc + 1) / 2 + cnrow * nc) * (value_bytes + 8)) as u64
        };
        let input_bytes = (self.nnz * (value_bytes + 8)) as u64;
        let mut est = crate::diagnostics::estimate_left_looking(
            nsuper,
            &panel_bytes,
            &compact_bytes,
            &update_list,
            value_bytes,
            input_bytes,
        );
        est.factor_flops = (0..nsuper)
            .map(|s| {
                let (nc, nr) = (sym.supernodes[s].ncol as u64, rs[s].len() as u64);
                nr * nr * nc
            })
            .sum();
        // Critical path (Amdahl bound) + tree width for the thread-aware v2 model.
        // Supernodes are in elimination (postorder) order, so children precede their
        // parent and a single forward pass computes the longest leaf-to-root chain.
        let mut crit = vec![0u64; nsuper];
        let mut cp = 0u64;
        for s in 0..nsuper {
            let (nc, nr) = (sym.supernodes[s].ncol as u64, rs[s].len() as u64);
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
        // Multifrontal transient: the contribution-block-stack model (the
        // left-looking `transient_peak_bytes` does not capture the CB stack).
        let children: Vec<Vec<usize>> = sym.supernodes.iter().map(|s| s.children.clone()).collect();
        let mf_active = crate::diagnostics::estimate_multifrontal_active_peak(
            levels,
            &|s| rs[s].len() as u64,
            &|s| sym.supernodes[s].ncol as u64,
            &children,
            value_bytes as u64,
        );
        let mf_scratch = (mf_active + est.factor_bytes) / 4 + 32_000_000;
        let mf_base = mf_active + est.factor_bytes + input_bytes + mf_scratch;
        // Work-stealing overlap margin: the rayon scheduler does not run one tree
        // level cleanly at a time - a deep subtree's leaves can be live while
        // another subtree's mid-level fronts factor, so fronts of more than one
        // level coexist, plus the per-front extract buffer. A fixed 1.4x margin
        // keeps the bound above the measured 24-thread peak across the corpus
        // (the structural / 3D matrices a single-level model under-predicted by up
        // to ~25%), matching the left-looking estimate's conservatism.
        est.mf_transient_peak_bytes = mf_base * 7 / 5;
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
            crate::numeric::multifrontal_ldlt::recommend_threads_for_sym(&self.symbolic, cap)
        });
        let t = std::time::Instant::now();
        let (scaled, scale) = equilibrate_with(a, &opts.scaling)?;
        let factors = factor_numeric(&self.symbolic, &scaled, opts)?;
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        let mut diagnostics = crate::diagnostics::Diagnostics {
            threads: resolved_threads,
            factor_nnz: factors.l_values.len() as u64,
            estimate: Some(estimate),
            ..Default::default()
        };
        diagnostics.push("factor", factor_ms, 0, factors.l_values.len() as u64 * 24);
        Ok(LdltSolver {
            factors,
            scale,
            diagnostics,
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

    /// The ND bakeoff must never return a pick with worse exact symbolic
    /// fill or worse predicted flops than the incumbent — on a plain 7-point
    /// Laplacian rslab's AMD is genuinely competitive (measured tied at 32³),
    /// so this pins the Pareto guarantee, not an adoption.
    #[test]
    fn nd_bakeoff_pareto_plain_grid() {
        let a = grid3d(24); // n = 13824
        let s_amd = SolverSettings::default();
        let sym_amd = LdltSymbolic::analyze_with(&a, &s_amd).unwrap();
        let amd_fill = sym_amd.symbolic_factor_nnz();
        let amd_flops = sym_amd.estimate_memory::<f64>().factor_flops;

        let (sym_pick, s_pick) = LdltSolver::<f64>::nd_bakeoff(&a, sym_amd, s_amd).unwrap();

        assert!(
            sym_pick.symbolic_factor_nnz() <= amd_fill,
            "fill regressed: {} > {amd_fill}",
            sym_pick.symbolic_factor_nnz()
        );
        assert!(
            sym_pick.estimate_memory::<f64>().factor_flops <= amd_flops,
            "flops regressed"
        );
        // Whatever the pick, it must factor + solve correctly.
        let solver = sym_pick.factor(&a, &s_pick).unwrap();
        let b: Vec<f64> = (0..a.n).map(|i| (i % 7) as f64 - 3.0).collect();
        let x = solver.solve(&b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-8);
    }

    /// Direct bakeoff on the edge-element curl-curl pattern (the rapidfem
    /// case that motivated it: 87k DOFs, ~2e11 AMD flops vs 1.8e10 MetisND):
    /// nested dissection wins clearly, so the bakeoff must adopt it.
    #[cfg(feature = "matgen")]
    #[test]
    fn nd_bakeoff_adopts_on_curl_curl() {
        let a = crate::matgen::fem::curl_curl(&[16, 16, 16], 0.8, 0.1); // n = 12288
        let s_amd = SolverSettings::default();
        let sym_amd = LdltSymbolic::analyze_with(&a, &s_amd).unwrap();
        let amd_flops = sym_amd.estimate_memory::<Complex<f64>>().factor_flops;

        let (sym, s) = LdltSolver::<Complex<f64>>::nd_bakeoff(&a, sym_amd, s_amd).unwrap();
        assert_eq!(s.ordering, crate::symbolic::OrderingMethod::MetisND);
        assert!(
            (sym.estimate_memory::<Complex<f64>>().factor_flops as f64)
                < amd_flops as f64 * ND_BAKEOFF_ADOPT_RATIO
        );
    }

    /// End-to-end guarantee on `tuned` for a large curl-curl system: whatever
    /// mechanism resolves it (model, OOD ordering race, or the ND bakeoff),
    /// the returned pick must realise the nested-dissection-class win over
    /// the AMD default - this is the regression that cost 10x factor time
    /// in the rapidfem FEM sweep.
    #[cfg(feature = "matgen")]
    #[test]
    fn tuned_finds_nd_class_win_on_curl_curl() {
        let a = crate::matgen::fem::curl_curl(&[22, 22, 22], 0.8, 0.1); // n = 31944
        let sym_amd = LdltSymbolic::analyze_with(&a, &SolverSettings::default()).unwrap();
        let amd_flops = sym_amd.estimate_memory::<Complex<f64>>().factor_flops;

        let (sym, s) =
            LdltSolver::<Complex<f64>>::tuned(&a, crate::auto_tune::DEFAULT_TUNE_WEIGHT).unwrap();
        eprintln!(
            "curl_curl pick {:?}: fill {} flops {}",
            s.ordering,
            sym.symbolic_factor_nnz(),
            sym.estimate_memory::<Complex<f64>>().factor_flops
        );
        assert_ne!(s.ordering, crate::symbolic::OrderingMethod::Amd);
        assert!(
            (sym.estimate_memory::<Complex<f64>>().factor_flops as f64)
                < amd_flops as f64 * ND_BAKEOFF_ADOPT_RATIO
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
        let solver = LdltSolver::factor(&a).unwrap();
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
        let sym = LdltSymbolic::analyze(&a).unwrap();
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
        // knob is threaded end-to-end (SolverSettings.scaling → equilibrate_with →
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
        let sym = LdltSymbolic::analyze(&a).unwrap();
        for strat in [
            ScalingStrategy::OnePassInfNorm,
            ScalingStrategy::Identity,
            ScalingStrategy::InfNorm,
            ScalingStrategy::Mc64Symmetric,
            ScalingStrategy::Auto,
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
        let f = LdltSolver::factor(&a).unwrap();
        let nrhs = 4;
        // Row-major B.
        let b: Vec<f64> = (0..n * nrhs).map(|k| (k % 5) as f64 - 2.0).collect();
        let x = f.solve_many(&b, nrhs).unwrap();
        for c in 0..nrhs {
            let bc: Vec<f64> = (0..n).map(|i| b[i * nrhs + c]).collect();
            let xc = f.solve(&bc).unwrap();
            for i in 0..n {
                assert!((x[i * nrhs + c] - xc[i]).abs() < 1e-10, "rhs {c} row {i}");
            }
        }
    }

    #[test]
    fn f64_inertia_diagonal_signs() {
        // Pure diagonal → all 1×1 pivots; (positive) equilibration preserves
        // signs, so the inertia is the diagonal's signature.
        let diag = [2.0_f64, -3.0, 4.0, -1.0, 5.0];
        let n = diag.len();
        let (rows, cols): (Vec<_>, Vec<_>) = (0..n).map(|i| (i, i)).unzip();
        let a = CscMatrix::<f64>::from_triplets(n, &rows, &cols, &diag).unwrap();
        let f = LdltSolver::factor(&a).unwrap();
        let inertia = f.inertia();
        assert_eq!(
            (inertia.positive, inertia.negative, inertia.zero),
            (3, 2, 0)
        );
        assert_eq!(inertia.total(), n);
    }

    #[test]
    fn f64_inertia_indefinite_2x2() {
        // [[0,1],[1,0]] has eigenvalues ±1 → Bunch-Kaufman takes one 2×2 block
        // with det < 0, classified as one positive + one negative.
        let a = CscMatrix::<f64>::from_triplets(2, &[0, 1], &[0, 0], &[0.0, 1.0]).unwrap();
        let f = LdltSolver::factor(&a).unwrap();
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
        let analysis = LdltSymbolic::analyze(&template).unwrap();
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
            let one_shot = LdltSolver::factor(&a).unwrap();
            let x_phased = phased.solve(&b).unwrap();
            let x_one = one_shot.solve(&b).unwrap();

            // Same factor → identical solve.
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
        let sym = LdltSymbolic::analyze(&a).unwrap();
        // Auto capped at 8: a tridiagonal is thin/narrow -> policy returns 2.
        let auto = sym
            .factor(&a, &SolverSettings::default().with_auto_threads(8))
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
            .factor(&a, &SolverSettings::default().with_auto_threads(1))
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
        let bare = LdltSymbolic::analyze(&a).unwrap();
        let with_default =
            LdltSymbolic::analyze_with(&a, &crate::SolverSettings::default()).unwrap();
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
            let f = LdltSymbolic::analyze_with(&a, &opts)
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
        let analysis = LdltSymbolic::analyze(&a).unwrap();
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
        let solver = LdltSolver::factor(&a).unwrap();

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
        let solver = LdltSolver::factor(&a).unwrap();

        let x_plain = solver.solve(&b).unwrap();
        let x_ref = solver.solve_refined(&a, &b, 3).unwrap();
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
        let solver = LdltSolver::factor(&a).unwrap();
        assert!(matches!(
            solver.solve(&[1.0, 2.0, 3.0]),
            Err(RslabError::DimensionMismatch { .. })
        ));
    }
}
