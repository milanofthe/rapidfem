use super::util::*;
use super::*;
use crate::error::RslabError;
use crate::numeric::ldlt::LdltSolver;
use crate::numeric::settings::SolverSettings;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use num_complex::Complex;

type C = Complex<f64>;

/// 2D complex-symmetric Helmholtz-style grid (lower triangle).
fn grid(m: usize, diag: C, off: C) -> CscMatrix<C> {
    let n = m * m;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            r.push(p);
            c.push(p);
            v.push(diag);
            if b + 1 < m {
                let (hi, lo) = (idx(a, b + 1), p);
                r.push(hi);
                c.push(lo);
                v.push(off);
            }
            if a + 1 < m {
                let (hi, lo) = (idx(a + 1, b), p);
                r.push(hi);
                c.push(lo);
                v.push(off);
            }
        }
    }
    CscMatrix::<C>::from_triplets(n, &r, &c, &v).unwrap()
}

#[test]
fn cocg_unpreconditioned_solves_complex_symmetric() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(8, c(4.0, 0.5), c(-1.0, 0.1));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let res = cocg(&a, &b, &NoPreconditioner, 1e-10, 2000).unwrap();
    assert!(res.converged, "COCG should converge, res={}", res.final_res);
    assert_eq!(res.stop, StopReason::Converged);
    // Verify against the actual residual.
    let mut ax = vec![C::default(); n];
    a.symv(&res.x, &mut ax);
    let r = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(r < 1e-7, "residual {}", r);

    // Starve the budget: the same solve capped at one iteration must report
    // `MaxIter`, not a false `Converged`.
    let capped = cocg(&a, &b, &NoPreconditioner, 1e-14, 1).unwrap();
    assert!(!capped.converged);
    assert_eq!(capped.stop, StopReason::MaxIter);
}

#[test]
fn cocr_solves_complex_symmetric_pre_and_unpre() {
    let c = |re, im| Complex::new(re, im);
    let a = grid(10, c(4.0, 0.5), c(-1.0, 0.1));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

    // Unpreconditioned COCR converges to the true solution.
    let un = cocr(&a, &b, &NoPreconditioner, 1e-10, 3000).unwrap();
    assert!(un.converged, "COCR res={}", un.final_res);
    let mut ax = vec![C::default(); n];
    a.symv(&un.x, &mut ax);
    let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(res < 1e-7, "COCR residual {}", res);

    // RLA-preconditioned COCR collapses to a handful of iterations.
    let m = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let pre = cocr(&a, &b, &m, 1e-10, 3000).unwrap();
    assert!(pre.converged && pre.iters <= 3, "iters {}", pre.iters);
}

#[test]
fn gmres_solves_unsymmetric_with_lu_preconditioner() {
    use crate::numeric::lu::LuSolver;
    use crate::sparse::general::GeneralCsc;
    // Genuinely unsymmetric complex 2D grid (right != left couplings).
    let c = |re, im| Complex::new(re, im);
    let m = 8;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(8.0, 1.0));
            if b + 1 < m {
                let q = idx(a, b + 1);
                rr.push(p);
                cc.push(q);
                vv.push(c(-1.0, 0.2));
                rr.push(q);
                cc.push(p);
                vv.push(c(-2.0, 0.1));
            }
            if a + 1 < m {
                let q = idx(a + 1, b);
                rr.push(p);
                cc.push(q);
                vv.push(c(-1.5, 0.3));
                rr.push(q);
                cc.push(p);
                vv.push(c(-0.5, 0.4));
            }
        }
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

    // Unpreconditioned GMRES converges (well-conditioned).
    let un = gmres(&a, &b, &NoPreconditioner, 1e-10, 2000, 40, None).unwrap();
    assert!(un.converged, "GMRES res={}", un.final_res);

    // LU factor as preconditioner -> 1-2 iterations.
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let pre = gmres(&a, &b, &lu, 1e-10, 200, 40, None).unwrap();
    assert!(pre.converged, "preconditioned GMRES res={}", pre.final_res);
    assert!(
        pre.iters <= 3,
        "LU-preconditioned GMRES iters {}",
        pre.iters
    );
    // Verify the true residual.
    let mut y = vec![C::default(); n];
    a.matvec(&pre.x, &mut y);
    let res = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(res < 1e-8, "residual {}", res);
}

#[test]
fn gmres_singular_operator_breaks_down_without_nan() {
    // Rank-deficient operator: A = diag(1, 1, 0) is singular and
    // `b = (1,1,1)` has a component in the null space (e_2), so GMRES cannot
    // drive the residual to zero - it stagnates. The Krylov subspace is
    // A-invariant with a *singular* restriction (eigenvalue 0), so the upper-
    // triangular Hessenberg factor acquires a ~0 diagonal. The unguarded
    // back-substitution would divide by it and emit NaN/Inf into `x` and the
    // reported residual; the guard must instead truncate to the well-
    // conditioned block, giving a deterministic breakdown: a finite iterate and
    // a truthful non-convergence report.
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    // Only the (0,0) and (1,1) entries; row/col 2 is all-zero -> A e_2 = 0.
    let a =
        GeneralCsc::<C>::from_triplets(3, &[0, 1], &[0, 1], &[c(1.0, 0.0), c(1.0, 0.0)]).unwrap();
    let b = vec![c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)];
    let res = gmres(&a, &b, &NoPreconditioner, 1e-12, 50, 10, None).unwrap();
    // No NaN/Inf reached the solution or the residual.
    assert!(
        res.x.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
        "solution has NaN/Inf: {:?}",
        res.x
    );
    assert!(
        res.final_res.is_finite(),
        "residual is NaN/Inf: {}",
        res.final_res
    );
    // Deterministic breakdown: reported as non-converged with a sane residual
    // (the singular direction pins the relative residual near 1/sqrt3 ~ 0.577 - it
    // is bounded well below the blow-up an unguarded divide would produce).
    assert!(
        !res.converged,
        "singular system must not report convergence"
    );
    assert!(
        res.final_res > 1e-12 && res.final_res <= 1.0,
        "residual not in the sane breakdown range: {}",
        res.final_res
    );
}

/// Build the genuinely unsymmetric complex grid used by the GMRES tests.
fn unsym_grid(m: usize) -> crate::sparse::general::GeneralCsc<C> {
    use crate::sparse::general::GeneralCsc;
    let c = |re, im| Complex::new(re, im);
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(8.0, 1.0));
            if b + 1 < m {
                let q = idx(a, b + 1);
                rr.push(p);
                cc.push(q);
                vv.push(c(-1.0, 0.2));
                rr.push(q);
                cc.push(p);
                vv.push(c(-2.0, 0.1));
            }
            if a + 1 < m {
                let q = idx(a + 1, b);
                rr.push(p);
                cc.push(q);
                vv.push(c(-1.5, 0.3));
                rr.push(q);
                cc.push(p);
                vv.push(c(-0.5, 0.4));
            }
        }
    }
    GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap()
}

/// Wraps a preconditioner and counts every scalar `apply` (the `M^-1` solves).
struct CountingPc<'a, M: ?Sized> {
    inner: &'a M,
    applies: std::sync::atomic::AtomicUsize,
}
impl<T: Scalar, M: Preconditioner<T> + ?Sized> Preconditioner<T> for CountingPc<'_, M> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        self.applies
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.apply(r, z)
    }
}

#[test]
fn fgmres_saves_one_precond_apply_per_restart_cycle() {
    // FGMRES: the preconditioned basis `Z` is kept, so the restart
    // update `x += Z y` costs **no** extra `M^-1` solve. The loop applies `M^-1`
    // exactly once per inner iteration and never at the restart, so over a
    // multi-cycle solve the total preconditioner-apply count equals the total
    // iteration count. Plain right-preconditioned GMRES would spend one extra
    // `M^-1` per cycle (rebuilding `M^-1(V y)`), i.e. `iters + n_cycles`.
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(10); // n = 100
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    // Weak (heavily incomplete) factor so the solve needs several restart
    // cycles at a short restart length - exercising the per-cycle update path.
    let opts = SolverSettings {
        drop_tol: Some(8e-1),
        ..Default::default()
    };
    let lu = LuSolver::factor(&a, &opts).unwrap();
    let restart = 5;
    let counting = CountingPc {
        inner: &lu,
        applies: std::sync::atomic::AtomicUsize::new(0),
    };
    let res = gmres(&a, &b, &counting, 1e-10, 2000, restart, None).unwrap();
    assert!(res.converged, "FGMRES must converge, res={}", res.final_res);
    let applies = counting.applies.load(std::sync::atomic::Ordering::Relaxed);
    // Multiple restart cycles actually occurred (proves the saving is nonzero).
    assert!(
        res.iters > restart,
        "expected multiple cycles, iters={}",
        res.iters
    );
    // FGMRES: exactly one apply per inner iteration, none at restart.
    assert_eq!(
        applies, res.iters,
        "FGMRES precond applies {} must equal iters {} (no per-cycle extra solve)",
        applies, res.iters
    );
}

#[test]
fn gmres_warm_start_cuts_total_iterations_on_related_sequence() {
    // Warm start: a sequence of related systems `A x = b_k` with a
    // slowly rotating right-hand side. Cold-starting every solve from 0 pays
    // the full iteration count each time; seeding each solve with the previous
    // solution (which is close, because the RHS barely moved) collapses the
    // per-solve count. Total iterations must drop by a clear margin.
    let a = unsym_grid(12); // n = 144
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    // Two fixed directions; b_k interpolates between them by a small angle step.
    let b0: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let b1: Vec<C> = (0..n)
        .map(|i| c(((i * 3) % 7) as f64 - 3.0, ((i % 3) as f64) - 1.0))
        .collect();
    let steps = 10;
    let (tol, maxit, restart) = (1e-8, 4000, 60);

    let bk = |k: usize| -> Vec<C> {
        let th = 5e-5 * k as f64; // slow rotation
        let (ct, st) = (th.cos(), th.sin());
        (0..n)
            .map(|i| b0[i] * c(ct, 0.0) + b1[i] * c(st, 0.0))
            .collect()
    };

    // Cold: every solve from x0 = 0.
    let mut cold_total = 0usize;
    for k in 0..steps {
        let r = gmres(&a, &bk(k), &NoPreconditioner, tol, maxit, restart, None).unwrap();
        assert!(r.converged, "cold solve {k} did not converge");
        cold_total += r.iters;
    }

    // Warm: first solve from 0, each subsequent seeded with the previous x.
    let mut warm_total = 0usize;
    let mut prev: Option<Vec<C>> = None;
    for k in 0..steps {
        let r = gmres(
            &a,
            &bk(k),
            &NoPreconditioner,
            tol,
            maxit,
            restart,
            prev.as_deref(),
        )
        .unwrap();
        assert!(r.converged, "warm solve {k} did not converge");
        warm_total += r.iters;
        prev = Some(r.x);
    }

    // A meaningful reduction (well beyond noise): warm start must cut the total
    // iteration count by at least 30% over the sequence.
    assert!(
        (warm_total as f64) < 0.7 * (cold_total as f64),
        "warm start did not help enough: cold={cold_total}, warm={warm_total}"
    );
}

#[test]
fn gmres_block_single_rhs_matches_scalar_gmres() {
    // s = 1 block GMRES reduces to the single-RHS path (same Arnoldi, same
    // Givens, default block apply = one single apply): same solution to the
    // requested tolerance, same iteration count up to a +/-1 boundary effect.
    // The paths are NOT bit-identical - block uses CGS2, single uses MGS+DGKS,
    // so the projections sum in a different order and the true residual can
    // straddle `tol` by a rounding ULP. This is documented as a design point in
    // the module-level "Orthogonalization" note, not a defect.
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(8);
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let single = gmres(&a, &b, &lu, 1e-10, 200, 40, None).unwrap();
    let blk = gmres_block(&a, &b, 1, &lu, 1e-10, 200, 40, None).unwrap();
    assert!(blk.converged);
    assert!(
        (blk.iters as i64 - single.iters as i64).abs() <= 1,
        "block(s=1) iters {} vs single {}",
        blk.iters,
        single.iters
    );
    let diff = (0..n)
        .map(|i| (blk.x[i] - single.x[i]).norm())
        .fold(0.0, f64::max);
    assert!(diff < 1e-7, "block(s=1) solution differs by {diff}");
}

#[test]
fn gmres_block_multi_rhs_solves_each_column() {
    // Several distinct right-hand sides solved in one block iteration; every
    // column must reach its own system's true residual.
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(10);
    let n = a.n;
    let s = 5;
    // Column-major nxs block: RHS `k` is shifted/scaled so columns differ.
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let res = gmres_block(&a, &bblk, s, &lu, 1e-10, 200, 40, None).unwrap();
    assert!(
        res.converged,
        "block GMRES must converge; res={:?}",
        res.final_res
    );
    // True residual per column.
    for k in 0..s {
        let mut y = vec![C::default(); n];
        a.matvec(&res.x[k * n..k * n + n], &mut y);
        let r = (0..n)
            .map(|i| (y[i] - bblk[k * n + i]).norm())
            .fold(0.0, f64::max);
        assert!(r < 1e-8, "column {k} residual {r}");
    }
    // Each column must equal the single-RHS solve of that column.
    for k in 0..s {
        let single = gmres(&a, &bblk[k * n..k * n + n], &lu, 1e-10, 200, 40, None).unwrap();
        let diff = (0..n)
            .map(|i| (res.x[k * n + i] - single.x[i]).norm())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-9,
            "column {k} differs from single solve by {diff}"
        );
    }
}

#[test]
fn gmres_block_within_cycle_deflation_shrinks_applies() {
    // Different-convergence-rate regime: a diagonal operator with
    // distinct eigenvalues, unpreconditioned, with RHS `k` supported on `k+1`
    // distinct eigenvalues. GMRES on such a RHS converges in exactly `k+1`
    // steps, so the columns finish at staggered steps within a *single* cycle.
    // The fix must (i) still solve every column exactly like single-RHS GMRES
    // and (ii) actually shrink the batched operator applies as columns deflate
    // mid-cycle - not only at restart. A counting operator records the width
    // of every `apply_block`, and we assert the panel narrows.
    use std::sync::Mutex;

    struct CountingOp<'a> {
        inner: &'a GeneralCsc<C>,
        widths: Mutex<Vec<usize>>,
    }
    impl LinearOperator<C> for CountingOp<'_> {
        fn n(&self) -> usize {
            self.inner.n()
        }
        fn apply(&self, x: &[C], y: &mut [C]) {
            self.widths.lock().unwrap().push(1);
            self.inner.apply(x, y);
        }
        fn apply_block(&self, x: &[C], y: &mut [C], s: usize) {
            self.widths.lock().unwrap().push(s);
            self.inner.apply_block(x, y, s);
        }
    }

    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 8;
    let s = 4;
    // Diagonal operator with distinct entries -> the minimal polynomial degree
    // of a RHS equals the number of distinct diagonal entries in its support.
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(2.0 + i as f64, 0.5 + 0.1 * i as f64));
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    // RHS `k` = sum of the first `k+1` unit vectors -> converges in `k+1` steps.
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..=k {
            bblk[k * n + i] = c(1.0, 0.0);
        }
    }

    let op = CountingOp {
        inner: &a,
        widths: Mutex::new(Vec::new()),
    };
    let res = gmres_block(&op, &bblk, s, &NoPreconditioner, 1e-12, 200, 40, None).unwrap();
    assert!(
        res.converged,
        "block GMRES must converge; res={:?}",
        res.final_res
    );

    // (i) Every column matches the single-RHS GMRES solve of that column.
    for k in 0..s {
        let single = gmres(
            &a,
            &bblk[k * n..k * n + n],
            &NoPreconditioner,
            1e-12,
            200,
            40,
            None,
        )
        .unwrap();
        let diff = (0..n)
            .map(|i| (res.x[k * n + i] - single.x[i]).norm())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-9,
            "column {k} differs from single solve by {diff}"
        );
    }

    // (ii) The batched applies actually shrank mid-cycle. Without within-cycle
    // deflation every `apply_block` would run at full width `s`; with it, later
    // steps run narrower. Assert a full-width apply happened (the first step),
    // that some apply narrowed below `s`, that the panel drained to width 1,
    // and that the total column-applies fell below the full-width bound.
    let widths = op.widths.into_inner().unwrap();
    let ncalls = widths.len();
    let total_cols: usize = widths.iter().sum();
    assert_eq!(
        *widths.iter().max().unwrap(),
        s,
        "the first cycle must open at full width"
    );
    assert!(
        *widths.iter().min().unwrap() < s,
        "no apply narrowed: deflation did not shrink the panel"
    );
    assert!(
        widths.contains(&1),
        "the panel must drain to a single active column"
    );
    assert!(
        total_cols < s * ncalls,
        "total column-applies {total_cols} not below the full-width bound {}",
        s * ncalls
    );
}

#[test]
fn gmres_block_bcgs2_bit_identical_across_thread_counts() {
    // The block-CGS2 orthogonalization reduces over fixed row-chunks folded in
    // chunk order, so the whole block solve is **bit-identical regardless of
    // the thread count** - the determinism guarantee. Solve the same block in
    // a 1-thread and an 8-thread rayon pool and require exact equality.
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    // Wide enough that a chunked reduction actually spans several chunks.
    let a = unsym_grid(60);
    let n = a.n;
    let s = 5;
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let solve = || gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap();
    let x1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(solve);
    let x8 = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap()
        .install(solve);
    assert_eq!(
        x1.iters, x8.iters,
        "iteration count must not depend on threads"
    );
    assert!(
        x1.x == x8.x,
        "block solve must be bit-identical across thread counts"
    );
}

#[test]
fn block_gmres_orthogonalization_respects_factor_thread_cap() {
    // The block-GMRES orthogonalization must run in a pool derived
    // from the *factor's* Threads policy, not the ambient global pool. Factor
    // with a hard cap of 2 workers; the factor then reports `Fixed(2)` as its
    // solve-phase policy, and the pool built from it caps `current_num_threads`
    // to 2 - even when the surrounding (ambient) pool is far wider. The solve
    // stays bit-identical whether run bare or inside a wide ambient pool (the
    // chunk-order reduction is thread-count independent), so the cap changes
    // only the concurrency, never the numbers.
    use crate::numeric::lu::LuSolver;
    use crate::numeric::settings::Threads;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(30);
    let n = a.n;
    let s = 4;
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..n {
            bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
        }
    }
    // Factor capped to exactly 2 workers -> solve policy is Fixed(2).
    let lu = LuSolver::factor(&a, &SolverSettings::default().with_threads(2)).unwrap();
    assert_eq!(
        Preconditioner::<C>::solve_threads(&lu),
        Threads::Fixed(2),
        "factor must carry its resolved solve-phase thread budget"
    );
    // The pool the orthogonalization installs caps the worker count to 2,
    // regardless of a wider ambient pool around it.
    let pool = solve_thread_pool(Preconditioner::<C>::solve_threads(&lu));
    let seen = with_threads(8, || {
        pool.as_ref().unwrap().install(rayon::current_num_threads)
    });
    assert_eq!(
        seen, 2,
        "the ortho pool must cap to the factor's 2-worker budget"
    );

    // The full solve is identical bare vs. inside a wide ambient pool: the
    // internal cap governs concurrency only, never the result.
    let bare = gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap();
    let in_wide = with_threads(8, || {
        gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap()
    });
    assert!(bare.converged);
    assert!(
        bare.x == in_wide.x && bare.iters == in_wide.iters,
        "capped ortho pool must not perturb the numeric result"
    );
}

#[test]
fn ambient_threads_factor_matches_default_and_runs_on_shared_pool() {
    // `Threads::Ambient` factors on the current pool (no new spawn) - the
    // re-factor-in-loop path. Inside a `with_threads(2)` pool the factor must be
    // bit-identical to the normal (scoped-pool) factor: the numeric result is
    // independent of the thread policy.
    use crate::numeric::lu::LuSolver;
    use crate::numeric::settings::Threads;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(24);
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let lu_default = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let opts_amb = SolverSettings::default().with_threads(Threads::Ambient);
    let lu_amb = with_threads(2, || {
        assert_eq!(rayon::current_num_threads(), 2);
        LuSolver::factor(&a, &opts_amb).unwrap()
    });
    let x_def = gmres_block(&a, &b, 1, &lu_default, 1e-10, 200, 40, None).unwrap();
    let x_amb = gmres_block(&a, &b, 1, &lu_amb, 1e-10, 200, 40, None).unwrap();
    assert!(
        x_def.x == x_amb.x,
        "ambient-pool factor must be bit-identical to the default factor"
    );
}

#[test]
fn default_thread_policy_caps_at_four() {
    // The pareto-optimal embedded default: predict per matrix, never exceed 4.
    use crate::numeric::settings::Threads;
    assert_eq!(SolverSettings::default().threads, Threads::Auto { max: 4 });
}

#[test]
fn f32_lu_preconditioner_keeps_f64_accuracy_in_gmres() {
    use crate::sparse::general::GeneralCsc;
    let c = |re, im| Complex::new(re, im);
    let m = 10;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(20.0, 2.0)); // strongly diagonally dominant -> f32 factor is accurate
            if b + 1 < m {
                rr.push(p);
                cc.push(idx(a, b + 1));
                vv.push(c(-1.0, 0.2));
                rr.push(idx(a, b + 1));
                cc.push(p);
                vv.push(c(-2.0, 0.1));
            }
            if a + 1 < m {
                rr.push(p);
                cc.push(idx(a + 1, b));
                vv.push(c(-1.5, 0.3));
                rr.push(idx(a + 1, b));
                cc.push(p);
                vv.push(c(-0.5, 0.4));
            }
        }
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();
    // The f32 LU factor is a half-memory preconditioner accurate to ~1e-6
    // (the f32 apply floor). f64 GMRES reaches that floor in a couple of
    // iterations - ample for a MoM/FEM Krylov tolerance, at half the factor
    // memory. (A residual below ~1e-6 is not attainable with an f32 apply;
    // use the f64 factor for tighter tolerances.)
    let pc = LowPrecisionLu::factor(&a, &SolverSettings::default()).unwrap();
    assert!(pc.factor_nnz() > 0);
    let res = gmres(&a, &b, &pc, 1e-6, 200, 50, None).unwrap();
    assert!(res.converged, "mixed-precision GMRES res={}", res.final_res);
    assert!(res.iters <= 6, "iters {}", res.iters);
    let mut y = vec![C::default(); n];
    a.matvec(&res.x, &mut y);
    let r = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(r < 1e-5, "residual {}", r);
}

#[test]
fn cocr_handles_indefinite_helmholtz() {
    // Indefinite complex-symmetric: high-frequency 2D Helmholtz with a
    // negative-real diagonal shift (`diag = -1 + 0.3i`). A robust
    // preconditioner (static-pivoted RLA) plus COCR must still converge.
    let c = |re, im| Complex::new(re, im);
    let a = grid(10, c(-1.0, 0.3), c(1.0, 0.05));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c(1.0, (i % 3) as f64 - 1.0)).collect();
    let opts = SolverSettings::preconditioner(1e-10);
    let m = LdltSolver::factor(&a, &opts).unwrap();
    let pre = cocr(&a, &b, &m, 1e-9, 500).unwrap();
    assert!(
        pre.converged,
        "indefinite COCR res={} iters={}",
        pre.final_res, pre.iters
    );
    let mut ax = vec![C::default(); n];
    a.symv(&pre.x, &mut ax);
    let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(res < 1e-6, "residual {}", res);
}

#[test]
fn incomplete_factor_reduces_fill_and_still_preconditions() {
    // Threshold dropping shrinks the factor (less memory) at the cost of a
    // weaker preconditioner - but COCG must still converge to the true
    // f64 solution. Demonstrates the memory <-> iteration tradeoff.
    let c = |re, im| Complex::new(re, im);
    let a = grid(16, c(4.0, 1.0), c(-1.0, 0.2));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

    let full = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let opts = SolverSettings::default().with_drop_tol(5e-2);
    let inc = LdltSolver::factor(&a, &opts).unwrap();

    assert!(
        inc.factor_nnz() < full.factor_nnz(),
        "dropping should reduce fill: incomplete {} vs complete {}",
        inc.factor_nnz(),
        full.factor_nnz()
    );

    let rf = cocg(&a, &b, &full, 1e-10, 1000).unwrap();
    let ri = cocg(&a, &b, &inc, 1e-10, 1000).unwrap();
    assert!(ri.converged, "incomplete-preconditioned COCG must converge");
    assert!(
        ri.iters >= rf.iters,
        "incomplete factor should need >= complete-factor iterations"
    );
    let mut ax = vec![C::default(); n];
    a.symv(&ri.x, &mut ax);
    let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(res < 1e-8, "residual {}", res);
}

#[test]
fn f32_preconditioner_keeps_f64_accuracy() {
    // Factor the preconditioner in Complex<f32> (half memory) but iterate in
    // f64: the solution must still reach f64-level residual, and the f32
    // factor - though approximate - keeps the iteration count tiny.
    let c = |re, im| Complex::new(re, im);
    let a = grid(14, c(4.0, 1.0), c(-1.0, 0.2));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

    let m = LowPrecisionPreconditioner::factor(&a, &SolverSettings::default()).unwrap();
    let res = cocg(&a, &b, &m, 1e-10, 500).unwrap();
    assert!(res.converged, "mixed-precision COCG res={}", res.final_res);
    // A few iterations suffice; the f32 factor is a strong preconditioner.
    assert!(res.iters <= 12, "f32-preconditioned iters {}", res.iters);
    // Full f64 accuracy recovered despite the single-precision factor.
    let mut ax = vec![C::default(); n];
    a.symv(&res.x, &mut ax);
    let r = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(r < 1e-8, "mixed-precision residual {}", r);
    assert!(m.factor_nnz() > 0);
}

#[test]
fn rla_preconditioner_collapses_iteration_count() {
    // A complete RLA factorization is ~ A^-1, so preconditioned COCG must
    // converge in a handful of iterations - vastly fewer than without.
    let c = |re, im| Complex::new(re, im);
    let a = grid(12, c(4.0, 1.0), c(-1.0, 0.2));
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

    let unpre = cocg(&a, &b, &NoPreconditioner, 1e-10, 5000).unwrap();
    let m = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let pre = cocg(&a, &b, &m, 1e-10, 5000).unwrap();

    assert!(pre.converged && unpre.converged);
    assert!(
        pre.iters <= 3,
        "complete-factor preconditioner should need <=3 iters, got {}",
        pre.iters
    );
    assert!(
        pre.iters * 5 < unpre.iters,
        "preconditioner should cut iterations sharply: {} vs {}",
        pre.iters,
        unpre.iters
    );
}

/// Strongly **non-normal** 1D convection-diffusion operator as a general
/// complex matrix: tridiagonal `diag = 2 (+ tiny damping)`, super `= -1+gamma`,
/// sub `= -1-gamma`. For `gamma != 0` it is far from normal - the classic GMRES-hard
/// regime where the residual stagnates for many steps before converging - yet
/// remains (weakly) diagonally dominant, so unpreconditioned GMRES does
/// converge, only after many iterations / restart cycles.
fn convection_diffusion(n: usize, gamma: f64) -> crate::sparse::general::GeneralCsc<C> {
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(2.0, 0.02));
        if i + 1 < n {
            rr.push(i);
            cc.push(i + 1);
            vv.push(c(-1.0 + gamma, 0.0));
            rr.push(i + 1);
            cc.push(i);
            vv.push(c(-1.0 - gamma, 0.0));
        }
    }
    GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap()
}

/// Wraps an operator and records the width `s` of every block apply - the
/// deflation probe: a shrinking width proves the batched
/// applies narrow as columns converge.
struct WidthCountingOp<'a> {
    inner: &'a GeneralCsc<C>,
    widths: std::sync::Mutex<Vec<usize>>,
}
impl LinearOperator<C> for WidthCountingOp<'_> {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn apply(&self, x: &[C], y: &mut [C]) {
        self.widths.lock().unwrap().push(1);
        self.inner.apply(x, y);
    }
    fn apply_block(&self, x: &[C], y: &mut [C], s: usize) {
        self.widths.lock().unwrap().push(s);
        self.inner.apply_block(x, y, s);
    }
}

#[test]
fn gmres_unpreconditioned_nonnormal_needs_many_restarts() {
    // Unpreconditioned GMRES on a strongly non-normal operator:
    // it must survive the non-normal stagnation phase and multiple restart
    // cycles, then converge to the true solution. Exercises the restart /
    // outer-loop machinery that the diagonally dominant tests never stress.
    let a = convection_diffusion(120, 0.9);
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b: Vec<C> = (0..n).map(|i| c(((i % 5) as f64) - 2.0, 0.5)).collect();
    let restart = 20;
    let res = gmres(&a, &b, &NoPreconditioner, 1e-8, 8000, restart, None).unwrap();
    assert!(
        res.converged,
        "non-normal GMRES must converge, res={}",
        res.final_res
    );
    assert!(
        res.iters > restart,
        "must span multiple restart cycles, iters={} (restart={})",
        res.iters,
        restart
    );
    let mut y = vec![C::default(); n];
    a.matvec(&res.x, &mut y);
    let r = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(r < 1e-6, "true residual {}", r);
}

#[test]
fn gmres_reorthogonalization_keeps_illconditioned_arnoldi_accurate() {
    // A near-defective, strongly non-normal operator (bidiagonal
    // Jordan-like block: clustered diagonal, dominant super-diagonal) drives
    // the Arnoldi vectors toward linear dependence, so a single MGS sweep
    // collapses the norm and the conditional DGKS second pass (`hn < eta*||w_0||`)
    // must fire to restore orthogonality. Asserting the trigger directly needs
    // an intrusive probe; instead we certify the *effect*: GMRES still drives
    // the true residual to `tol` and matches the exact (direct-LU) solution -
    // which it could not if the ill-conditioned basis went uncorrected.
    use crate::numeric::lu::LuSolver;
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 32;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(2.0, 0.0)); // clustered diagonal -> non-normal, near-defective
        if i + 1 < n {
            rr.push(i);
            cc.push(i + 1);
            vv.push(c(3.0, 0.0)); // dominant super-diagonal
        }
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<C> = (0..n).map(|_| c(1.0, 0.2)).collect();
    // Long restart (single cycle) so the ill-conditioned basis is not masked by
    // a restart - the reorthogonalization alone keeps it usable.
    let res = gmres(&a, &b, &NoPreconditioner, 1e-10, 4000, n, None).unwrap();
    assert!(
        res.converged,
        "reorth must keep GMRES converging, res={}",
        res.final_res
    );
    let lu = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let xstar = lu.solve(&b).unwrap();
    let diff = (0..n)
        .map(|i| (res.x[i] - xstar[i]).norm())
        .fold(0.0, f64::max);
    assert!(
        diff < 1e-6,
        "GMRES solution off the direct solve by {} (lost orthogonality?)",
        diff
    );
}

#[test]
fn gmres_happy_breakdown_on_eigenvector_rhs() {
    // Happy breakdown: `b` is an eigenvector of the operator, so
    // the Krylov space `K_1 = span{b}` is already `A`-invariant. The Arnoldi
    // step-1 subdiagonal `h[1][0]` is exactly `0` (the invariant-subspace
    // branch), and GMRES must produce the exact solution in a single iteration.
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 12;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        rr.push(i);
        cc.push(i);
        vv.push(c(2.0 + i as f64, 0.5 - 0.05 * i as f64)); // distinct diagonal
    }
    let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
    // `e_0` is an eigenvector (eigenvalue `a[0][0]`); its Krylov space is 1-D.
    let mut b = vec![C::default(); n];
    b[0] = c(1.0, 0.0);
    let res = gmres(&a, &b, &NoPreconditioner, 1e-12, 50, 30, None).unwrap();
    assert!(
        res.converged,
        "eigenvector RHS must converge, res={}",
        res.final_res
    );
    assert_eq!(
        res.iters, 1,
        "happy breakdown must solve in one step, got {}",
        res.iters
    );
    let mut y = vec![C::default(); n];
    a.matvec(&res.x, &mut y);
    let r = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
    assert!(r < 1e-12, "true residual {}", r);
}

#[test]
fn gmres_block_incomplete_factor_multirate_deflation() {
    // Multi-rate within-cycle deflation under a genuine
    // **factor-based** (drop-tol) preconditioner. The operator is diagonal with
    // distinct entries `d_i`; the preconditioner is a `drop_tol` LU factor of a
    // *different* diagonal matrix `diag(p_i)` - a deliberately imperfect
    // approximate inverse, so the preconditioned operator `M^-1A = diag(d_i/p_i)`
    // still has distinct eigenvalues. Right-hand side `k` is supported on the
    // first `k+1` unit vectors, so its GMRES converges in **exactly** `k+1`
    // steps: the columns finish at staggered steps *within one cycle*. The
    // within-cycle deflation must finalize each fast column and shrink the
    // batched applies to the still-active width, draining the panel to 1 - while
    // every column still matches its single-RHS solve.
    use crate::numeric::lu::LuSolver;
    use crate::sparse::general::GeneralCsc;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let n = 8;
    let s = 4;
    // Operator D = diag(d_i), d_i distinct.
    let (mut dr, mut dc, mut dv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        dr.push(i);
        dc.push(i);
        dv.push(c(2.0 + i as f64, 0.3));
    }
    let a = GeneralCsc::<C>::from_triplets(n, &dr, &dc, &dv).unwrap();
    // Preconditioning matrix P = diag(p_i), p_i chosen so d_i/p_i stay distinct
    // (p_i = 1+i => ratios 2, 1.5, 1.33, ... all different): an imperfect factor.
    let (mut pr, mut pc_, mut pv) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        pr.push(i);
        pc_.push(i);
        pv.push(c(1.0 + i as f64, 0.1));
    }
    let pmat = GeneralCsc::<C>::from_triplets(n, &pr, &pc_, &pv).unwrap();
    // drop-tol factor path (imperfect preconditioner)
    let opts = SolverSettings {
        drop_tol: Some(1e-2),
        ..Default::default()
    };
    let lu = LuSolver::factor(&pmat, &opts).unwrap();

    // RHS k = sum of the first k+1 unit vectors -> converges in exactly k+1 steps.
    let mut bblk = vec![C::default(); n * s];
    for k in 0..s {
        for i in 0..=k {
            bblk[k * n + i] = c(1.0, 0.0);
        }
    }

    let op = WidthCountingOp {
        inner: &a,
        widths: std::sync::Mutex::new(Vec::new()),
    };
    let res = gmres_block(&op, &bblk, s, &lu, 1e-10, 200, 40, None).unwrap();
    assert!(
        res.converged,
        "block GMRES must converge; res={:?}",
        res.final_res
    );

    // Every column equals its single-RHS solve (deflation must not corrupt it).
    for k in 0..s {
        let single = gmres(&a, &bblk[k * n..k * n + n], &lu, 1e-10, 200, 40, None).unwrap();
        let diff = (0..n)
            .map(|i| (res.x[k * n + i] - single.x[i]).norm())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-8,
            "column {k} differs from single solve by {diff}"
        );
    }

    // Within-cycle deflation fired: the panel opened at full width `s`, narrowed
    // as fast columns deflated, and drained to a single active column.
    let widths = op.widths.into_inner().unwrap();
    assert!(!widths.is_empty());
    assert_eq!(
        *widths.iter().max().unwrap(),
        s,
        "the first cycle must open at full width"
    );
    assert!(
        *widths.iter().min().unwrap() < s,
        "within-cycle deflation did not shrink the panel: widths={widths:?}"
    );
    assert!(
        widths.contains(&1),
        "the panel must drain to a single active column: {widths:?}"
    );
}

/// A general complex diagonal operator with a prescribed spectrum - the
/// canonical GMRES-DR / GCRO-DR demonstrator: a handful of tiny eigenvalues
/// (which restarted GMRES keeps re-discovering and discarding) sitting far
/// below a cluster of larger ones. Unpreconditioned restarted GMRES stagnates
/// on the small cluster; deflating it is exactly what recycling does.
fn diag_op(eigs: &[C]) -> crate::sparse::general::GeneralCsc<C> {
    use crate::sparse::general::GeneralCsc;
    let n = eigs.len();
    let idx: Vec<usize> = (0..n).collect();
    GeneralCsc::<C>::from_triplets(n, &idx, &idx, eigs).unwrap()
}

/// A spectrum with `n_small` tiny eigenvalues far below a spread cluster.
fn stagnation_spectrum(n: usize, n_small: usize) -> Vec<C> {
    let c = |re: f64, im: f64| Complex::new(re, im);
    (0..n)
        .map(|i| {
            if i < n_small {
                // Tiny, tightly clustered near the origin - the stagnation drivers.
                c(0.01 + 0.004 * i as f64, 0.002 * i as f64)
            } else {
                // Larger eigenvalues spread over [1, 11], mildly complex.
                let t = (i - n_small) as f64 / (n - n_small) as f64;
                c(1.0 + 10.0 * t, 0.3 * (i as f64).sin())
            }
        })
        .collect()
}

#[test]
fn gmres_recycled_matches_plain_on_hard_matrix() {
    // Correctness: the recycled solve must reach the SAME solution as plain
    // FGMRES on a hard preconditioned system (weak incomplete LU factor, short
    // restart -> many cycles), to the same tolerance.
    use crate::numeric::lu::LuSolver;
    let c = |re, im| Complex::new(re, im);
    let a = unsym_grid(12); // n = 144
    let n = a.n;
    let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let opts = SolverSettings {
        drop_tol: Some(8e-1),
        ..Default::default()
    };
    let lu = LuSolver::factor(&a, &opts).unwrap();
    let (tol, maxit, restart) = (1e-10, 4000, 12);
    let plain = gmres(&a, &b, &lu, tol, maxit, restart, None).unwrap();
    let mut rec = Recycle::new(8);
    let recd = gmres_recycled(&a, &b, &lu, tol, maxit, restart, None, &mut rec).unwrap();
    assert!(plain.converged, "plain FGMRES did not converge");
    assert!(recd.converged, "recycled did not converge");
    let diff = (0..n)
        .map(|i| (plain.x[i] - recd.x[i]).norm())
        .fold(0.0, f64::max);
    assert!(
        diff < 1e-7,
        "recycled solution differs from plain by {diff}"
    );
    // The recycle handle came back populated for the next solve.
    assert!(rec.active() > 0, "recycle subspace was not populated");
}

#[test]
fn gmres_recycled_within_solve_reduces_restarts() {
    // Within-solve deflated restarting: on a stagnating, restart-limited solve
    // (tiny eigenvalue cluster, unpreconditioned, short restart) carrying the
    // harmonic-Ritz subspace across restarts must cut total iterations by a
    // real margin versus plain FGMRES - on a SINGLE solve (fresh handle, no
    // cross-solve benefit).
    let eigs = stagnation_spectrum(40, 4);
    let a = diag_op(&eigs);
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b: Vec<C> = (0..n).map(|i| c(1.0, 0.2 * (i as f64).cos())).collect();
    let (tol, maxit, restart) = (1e-9, 5000, 10);

    let plain = gmres(&a, &b, &NoPreconditioner, tol, maxit, restart, None).unwrap();
    let mut rec = Recycle::new(6);
    let recd = gmres_recycled(
        &a,
        &b,
        &NoPreconditioner,
        tol,
        maxit,
        restart,
        None,
        &mut rec,
    )
    .unwrap();
    assert!(plain.converged, "plain did not converge ({})", plain.iters);
    assert!(recd.converged, "recycled did not converge ({})", recd.iters);
    // Both hit the same true solution.
    let diff = (0..n)
        .map(|i| (plain.x[i] - recd.x[i]).norm())
        .fold(0.0, f64::max);
    assert!(diff < 1e-6, "solutions differ by {diff}");
    // Deflated restarting must shave a clear margin off the iteration count.
    eprintln!(
        "[within-solve] plain FGMRES(10)={} iters, GCRO-DR(k=6)={} iters",
        plain.iters, recd.iters
    );
    assert!(
        (recd.iters as f64) < 0.75 * (plain.iters as f64),
        "within-solve deflation did not help enough: plain={}, recycled={}",
        plain.iters,
        recd.iters
    );
}

#[test]
fn gmres_recycled_cross_solve_beats_warm_and_cold() {
    // Cross-solve recycling on a sequence of related systems: A gets a small
    // diagonal perturbation each step and b rotates slowly. Compare total
    // iterations over the sequence for cold (x0=0), warm (x0=prev x), and
    // recycled+warm (handle carried across solves). Must order
    // recycled < warm < cold, each by a meaningful margin.
    let base = stagnation_spectrum(48, 5);
    let n = base.len();
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b0: Vec<C> = (0..n).map(|i| c(1.0, 0.15 * (i as f64).cos())).collect();
    let b1: Vec<C> = (0..n).map(|i| c(0.4 * (i as f64).sin(), 1.0)).collect();
    let steps = 8;
    let (tol, maxit, restart) = (1e-9, 6000, 12);

    // Slowly varying operator: base spectrum + eps_k on each diagonal entry.
    let ak = |kk: usize| -> crate::sparse::general::GeneralCsc<C> {
        let eps = 2e-3 * kk as f64;
        let eigs: Vec<C> = base
            .iter()
            .enumerate()
            .map(|(i, &e)| e + c(eps * (1.0 + 0.05 * i as f64), 0.0))
            .collect();
        diag_op(&eigs)
    };
    let bk = |kk: usize| -> Vec<C> {
        let th = 0.02 * kk as f64;
        let (ct, st) = (th.cos(), th.sin());
        (0..n)
            .map(|i| b0[i] * c(ct, 0.0) + b1[i] * c(st, 0.0))
            .collect()
    };

    // Cold.
    let mut cold = 0usize;
    for kk in 0..steps {
        let a = ak(kk);
        let r = gmres(&a, &bk(kk), &NoPreconditioner, tol, maxit, restart, None).unwrap();
        assert!(r.converged, "cold {kk} stalled");
        cold += r.iters;
    }
    // Warm.
    let mut warm = 0usize;
    let mut prev: Option<Vec<C>> = None;
    for kk in 0..steps {
        let a = ak(kk);
        let r = gmres(
            &a,
            &bk(kk),
            &NoPreconditioner,
            tol,
            maxit,
            restart,
            prev.as_deref(),
        )
        .unwrap();
        assert!(r.converged, "warm {kk} stalled");
        warm += r.iters;
        prev = Some(r.x);
    }
    // Recycled (+ warm start, the intended combined use).
    let mut recycled = 0usize;
    let mut rec = Recycle::new(8);
    let mut prevr: Option<Vec<C>> = None;
    for kk in 0..steps {
        let a = ak(kk);
        let r = gmres_recycled(
            &a,
            &bk(kk),
            &NoPreconditioner,
            tol,
            maxit,
            restart,
            prevr.as_deref(),
            &mut rec,
        )
        .unwrap();
        assert!(r.converged, "recycled {kk} stalled");
        recycled += r.iters;
        prevr = Some(r.x);
    }

    eprintln!(
        "[cross-solve] cold={cold}, warm={warm}, recycled={recycled} (total iters over {steps} solves)"
    );
    assert!(warm < cold, "warm ({warm}) not below cold ({cold})");
    assert!(
        recycled < warm,
        "recycled ({recycled}) not below warm ({warm})"
    );
    // Meaningful reduction, not noise.
    assert!(
        (recycled as f64) < 0.7 * (cold as f64),
        "recycled ({recycled}) did not beat cold ({cold}) by a clear margin"
    );
}

#[test]
fn gmres_recycled_composes_with_warm_start() {
    // Recycle + x0 warm start together: on a related second solve, seeding from
    // the previous solution AND recycling its stagnation subspace must converge
    // to the correct solution and take no more iterations than warm-start alone.
    let eigs = stagnation_spectrum(40, 4);
    let a = diag_op(&eigs);
    let n = a.n;
    let c = |re: f64, im: f64| Complex::new(re, im);
    let b0: Vec<C> = (0..n).map(|i| c(1.0, 0.1 * (i as f64).cos())).collect();
    let b1: Vec<C> = (0..n).map(|i| c(1.0 + 0.02 * i as f64, 0.1)).collect();
    let (tol, maxit, restart) = (1e-9, 5000, 10);

    // First solve seeds both the warm start and the recycle handle.
    let mut rec = Recycle::new(6);
    let first = gmres_recycled(
        &a,
        &b0,
        &NoPreconditioner,
        tol,
        maxit,
        restart,
        None,
        &mut rec,
    )
    .unwrap();
    assert!(first.converged);

    // Second, related solve: warm-only vs warm+recycle.
    let warm_only = gmres(
        &a,
        &b1,
        &NoPreconditioner,
        tol,
        maxit,
        restart,
        Some(&first.x),
    )
    .unwrap();
    let warm_rec = gmres_recycled(
        &a,
        &b1,
        &NoPreconditioner,
        tol,
        maxit,
        restart,
        Some(&first.x),
        &mut rec,
    )
    .unwrap();
    assert!(warm_only.converged && warm_rec.converged);
    // Same true solution.
    let mut ax = vec![C::default(); n];
    a.matvec(&warm_rec.x, &mut ax);
    let res = (0..n).map(|i| (ax[i] - b1[i]).norm()).fold(0.0, f64::max);
    assert!(res < 1e-6, "warm+recycle residual {res}");
    // Composed use is no worse than warm-start alone (typically better).
    assert!(
        warm_rec.iters <= warm_only.iters,
        "warm+recycle ({}) worse than warm-only ({})",
        warm_rec.iters,
        warm_only.iters
    );
}

#[test]
fn gmres_recycled_real_scalar_path() {
    // The real f64 field exercises the conjugate-pair reconstruction in
    // `combine_ritz` (a real diagonal has real eigenvalues, but a real
    // unsymmetric grid produces complex harmonic-Ritz pairs). Must converge to
    // the true solution - correctness of the real recycle path.
    use crate::sparse::general::GeneralCsc;
    let m = 8;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(4.0f64);
            if b + 1 < m {
                let q = idx(a, b + 1);
                rr.push(p);
                cc.push(q);
                vv.push(-1.0);
                rr.push(q);
                cc.push(p);
                vv.push(-1.8); // asymmetric => complex spectrum
            }
            if a + 1 < m {
                let q = idx(a + 1, b);
                rr.push(p);
                cc.push(q);
                vv.push(-0.7);
                rr.push(q);
                cc.push(p);
                vv.push(-1.3);
            }
        }
    }
    let a = GeneralCsc::<f64>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
    let mut rec = Recycle::<f64>::new(6);
    let r = gmres_recycled(&a, &b, &NoPreconditioner, 1e-9, 5000, 12, None, &mut rec).unwrap();
    assert!(r.converged, "real recycled solve did not converge");
    let mut ax = vec![0.0f64; n];
    a.matvec(&r.x, &mut ax);
    let res = (0..n).map(|i| (ax[i] - b[i]).abs()).fold(0.0, f64::max);
    assert!(res < 1e-6, "real recycled residual {res}");
}

/// Run `f` in a fresh rayon pool of `threads` workers.
fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(f)
}
