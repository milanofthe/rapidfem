use super::*;
use crate::numeric::settings::SolverSettings;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::OrderingMethod;
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
    let s = SolverSettings::default().with_nemin(1).with_threads(0);
    let f = LdltSolver::factor(&a, &s).expect("deep chain factors without overflow");
    assert_eq!(f.n(), n);
}

#[test]
fn rcm_and_autorace_orderings_factor_and_solve() {
    // A 2D-grid SPD system must factor and solve correctly under the new RCM
    // ordering and under the race (which includes RCM as a candidate).
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
    for ord in [OrderingMethod::Rcm, OrderingMethod::Auto] {
        let opts = SolverSettings::default().with_ordering(ord);
        let x = LdltSolver::factor(&a, &opts).unwrap().solve(&b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-9,
            "ordering {ord:?} residual {}",
            residual_inf(&a, &x, &b)
        );
    }
}

/// The number of 2x2 pivot blocks of a factorization.
fn two_by_two<T: Scalar>(f: &LdltSolver<T>) -> usize {
    f.diagnostics().numeric.two_by_two.unwrap_or(0)
}

fn residual_inf<T: Scalar>(a: &CscMatrix<T>, x: &[T], b: &[T]) -> f64 {
    let mut ax = vec![T::zero(); a.n];
    a.symv(x, &mut ax);
    (0..a.n)
        .map(|i| (ax[i] - b[i]).magnitude())
        .fold(0.0, f64::max)
}

/// 1D Laplacian-style SPD tridiagonal of size n (diag 2+something, off -1).
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
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(residual_inf(&a, &x, &b) < 1e-10);
}

/// 2D 5-point grid (mxm), lower triangle, complex-symmetric, diagonally
/// dominant. Branching assembly tree -> exercises multi-child `cmod`.
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
fn indefinite_2x2_inertia() {
    // [[0,1],[1,0]] (eigenvalues +/-1) forces a single 2x2 Bunch-Kaufman block.
    // The left-looking path must take that 2x2 (zero diagonal -> no 1x1 pivot)
    // and report inertia (1+, 1-).
    let a = CscMatrix::<f64>::from_triplets(2, &[0, 1], &[0, 0], &[0.0, 1.0]).unwrap();
    let ll = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    assert!(two_by_two(&ll) > 0, "expected a 2x2 block");
    assert_eq!(
        (
            ll.inertia().positive,
            ll.inertia().negative,
            ll.inertia().zero
        ),
        (1, 1, 0)
    );
    let b = [1.0_f64, -2.0];
    let x = ll.solve(&b).unwrap();
    assert!(residual_inf(&a, &x, &b) < 1e-12, "2x2 residual");
}

#[test]
fn indefinite_2d_grid_solves() {
    // 2D 5-point grid with a *small* diagonal (0.5 << 2*|off|): far from
    // diagonally dominant -> genuinely indefinite, so Bunch-Kaufman must take
    // many 2x2 pivots across several supernodes and still give a true solve -
    // the exact indefinite EM-FEM case the 2x2 pivoting is for.
    let a = grid2d_lower::<f64>(10, 0.5, -1.0);
    let n = a.n;
    let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
    let ll = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    assert!(
        two_by_two(&ll) > 0,
        "indefinite system should use 2x2 pivots"
    );
    assert!(
        ll.inertia().negative > 0 && ll.inertia().positive + ll.inertia().negative == n,
        "indefinite, nonsingular inertia"
    );
    let xl = ll.solve(&b).unwrap();
    assert!(
        residual_inf(&a, &xl, &b) < 1e-9,
        "left-looking indefinite residual"
    );
}

#[test]
fn indefinite_complex_symmetric() {
    // Complex-symmetric indefinite grid: the 2x2 path is type-agnostic. The
    // 2x2 blocks here are complex-symmetric (not Hermitian), exercising the
    // generic det/detinv arithmetic.
    let c = |re: f64, im: f64| Complex::new(re, im);
    let a = grid2d_lower::<Complex<f64>>(9, c(0.4, 0.3), c(-1.0, 0.1));
    let n = a.n;
    let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 0.5)).collect();
    let ll = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    assert!(
        two_by_two(&ll) > 0,
        "indefinite system should use 2x2 pivots"
    );
    assert_eq!(
        ll.inertia().positive + ll.inertia().negative + ll.inertia().zero,
        n,
        "inertia covers every pivot"
    );
    let xl = ll.solve(&b).unwrap();
    assert!(
        residual_inf(&a, &xl, &b) < 1e-9,
        "complex left-looking indefinite residual"
    );
}

#[test]
fn f64_dense_front_blocked_multi_panel() {
    // A fully dense symmetric matrix is one front of width n=100 > NB(64),
    // so factoring it exercises the blocked **multi-panel** Bunch-Kaufman
    // path (which the small n<=50 tests never reach). Diagonally dominant SPD.
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
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(
        residual_inf(&a, &x, &b) < 1e-9,
        "residual {}",
        residual_inf(&a, &x, &b)
    );
}

#[test]
fn complex_dense_front_blocked_multi_panel() {
    // Dense complex-symmetric, one front of width 90 > NB -> multi-panel.
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
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(residual_inf(&a, &x, &b) < 1e-9);
}

#[test]
fn f64_sparse_2d_grid_residual() {
    // 2D 5-point Laplacian on a 5x5 grid (n=25), SPD.
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
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(
        residual_inf(&a, &x, &b) < 1e-9,
        "residual {}",
        residual_inf(&a, &x, &b)
    );
}

#[test]
fn complex_sparse_tridiag_residual() {
    // Complex-symmetric Helmholtz-style tridiagonal: diagonal (4 + 0.5i),
    // off-diagonal (-1 + 0.1i). Complex symmetric (A = A^T), diagonally
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
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(
        residual_inf(&a, &x, &b) < 1e-10,
        "residual {}",
        residual_inf(&a, &x, &b)
    );
}

#[test]
fn complex_sparse_large_grid_parallel() {
    // 12x12 complex-symmetric grid (n=144): a deep, bushy assembly tree
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
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
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
        LdltSolver::factor(&a, &SolverSettings::default()).is_err(),
        "exact mode should reject the singular pivot"
    );

    let opts = SolverSettings::preconditioner(1e-8);
    let f = LdltSolver::factor(&a, &opts).unwrap();
    assert!(
        f.n_perturbed() >= 1,
        "expected >=1 perturbation, got {}",
        f.n_perturbed()
    );
    let b = vec![c(1.0, 0.0); n];
    let x = f.solve(&b).unwrap();
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
    let opts = SolverSettings::preconditioner(1e-8);
    let f = LdltSolver::factor(&a, &opts).unwrap();
    assert_eq!(
        f.n_perturbed(),
        0,
        "well-conditioned matrix needs no perturbation"
    );
}

#[test]
fn complex_sparse_2d_grid_residual() {
    // 2D complex-symmetric grid: diagonal (4 + i), neighbor (-1 + 0.2i).
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
    let f = LdltSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(
        residual_inf(&a, &x, &b) < 1e-9,
        "residual {}",
        residual_inf(&a, &x, &b)
    );
}
