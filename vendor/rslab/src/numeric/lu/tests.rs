use super::factor::factor_general_lu_numeric;
use super::solver::*;
use crate::numeric::settings::SolverSettings;
use crate::numeric::supernodal::Li;
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;

/// A badly scaled, row-scrambled unsymmetric system: without the MC64
/// row matching the front-restricted pivoting finds no usable pivot or
/// loses digits; with it (the default) the componentwise backward error
/// is roundoff.
#[test]
fn lu_matching_bounds_pivot_growth() {
    let m = 40usize;
    let n = m * m;
    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for j in 0..n {
        let (x, y) = (j % m, j / m);
        let rs = |i: usize| 10f64.powi(((i * 7919) % 13) as i32 - 6);
        let mut push = |i: usize, v: f64| cols[j].push(((i + 17) % n, v * rs(i)));
        push(j, 4.0);
        if x > 0 {
            push(j - 1, -1.4);
        }
        if x + 1 < m {
            push(j + 1, -0.6);
        }
        if y > 0 {
            push(j - m, -1.0);
        }
        if y + 1 < m {
            push(j + m, -1.0);
        }
    }
    let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
    for c in &mut cols {
        c.sort_by_key(|e| e.0);
        for &(r, v) in c.iter() {
            row_idx.push(r);
            values.push(v);
        }
        col_ptr.push(row_idx.len());
    }
    let a = GeneralCsc {
        n,
        col_ptr,
        row_idx,
        values,
    };
    let b: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
    // Componentwise backward error `max_i |r_i| / (|A||x| + |b|)_i`: the
    // rows span twelve decades, so a normwise residual would only
    // measure the largest rows.
    let omega = |x: &[f64]| {
        let mut r = b.clone();
        let mut d: Vec<f64> = b.iter().map(|v| v.abs()).collect();
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                r[a.row_idx[k]] -= a.values[k] * x[j];
                d[a.row_idx[k]] += a.values[k].abs() * x[j].abs();
            }
        }
        r.iter()
            .zip(&d)
            .map(|(ri, di)| if *di > 0.0 { ri.abs() / di } else { 0.0 })
            .fold(0.0, f64::max)
    };
    let opts = SolverSettings::default().with_threads(1);
    let s = LuSolver::factor(&a, &opts).unwrap();
    assert_eq!(s.diagnostics().decisions.scaling, "Mc64RowMatching");
    let x = s.solve(&b).unwrap();
    assert!(
        omega(&x) < 1e-12,
        "backward error with matching {}",
        omega(&x)
    );
    // Without the matching the shifted rows leave the fully-summed
    // blocks without a usable pivot.
    let s0 = LuSolver::factor(&a, &opts.with_matching(false));
    assert!(s0.is_err() || s0.unwrap().diagnostics().decisions.scaling == "TwoSidedRowCol");
}
use num_complex::Complex;

fn resid<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
    let mut y = vec![T::zero(); a.n];
    a.matvec(x, &mut y);
    (0..a.n)
        .map(|i| (y[i] - b[i]).magnitude())
        .fold(0.0, f64::max)
}

#[test]
fn f64_unsymmetric_tridiag() {
    // Unsymmetric real tridiagonal (full storage): diag 4, sub -1, super -2.
    let n = 20;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        r.push(i);
        c.push(i);
        v.push(4.0);
        if i + 1 < n {
            r.push(i + 1);
            c.push(i);
            v.push(-1.0);
            r.push(i);
            c.push(i + 1);
            v.push(-2.0);
        }
    }
    let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
    let b: Vec<f64> = (0..n).map(|i| i as f64 - 9.5).collect();
    let f = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(resid(&a, &x, &b) < 1e-10, "residual {}", resid(&a, &x, &b));
}

#[test]
fn lu_solve_many_matches_single() {
    let n = 12;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        r.push(i);
        c.push(i);
        v.push(5.0_f64);
        if i + 1 < n {
            r.push(i + 1);
            c.push(i);
            v.push(-1.0);
            r.push(i);
            c.push(i + 1);
            v.push(-2.0);
        }
    }
    let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
    let solver = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let nrhs = 5;
    let b: Vec<f64> = (0..n * nrhs).map(|k| (k % 7) as f64 - 3.0).collect();
    let x = solver.solve_many(&b, nrhs).unwrap();
    for col in 0..nrhs {
        let bc: Vec<f64> = (0..n).map(|i| b[col * n + i]).collect();
        let xc = solver.solve(&bc).unwrap();
        for i in 0..n {
            assert!((x[col * n + i] - xc[i]).abs() < 1e-10, "rhs {col} row {i}");
        }
    }
}

/// Every column of a block solve is BITWISE the single-column solve, whatever the
/// block width: a non-flexible Krylov method applies the solve to blocks in its Arnoldi
/// steps and to one column in its update, and any difference breaks the Arnoldi
/// relation (a single-precision factor made it 1e-3 on a saddle system). Covers the
/// leaf subtrees and the ancestor levels (a 2D grid) and the apex nodes with rows below
/// them (a 3D grid). Under FMA the fused complex multiply-add is not symmetric in its
/// factors, so the operand order of every single-column branch matters.
#[test]
fn lu_block_solve_is_bitwise_the_single_solve() {
    let mut seed = 12345u64;
    let mut rnd = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let c = |re: f64, im: f64| Complex::new(re, im);
    let grid = |m: usize, rnd: &mut dyn FnMut() -> f64| {
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let i = a * m + b;
                r.push(i);
                cc.push(i);
                v.push(c(4.5 + rnd(), rnd()));
                for j in [(a + 1 < m).then(|| i + m), (b + 1 < m).then(|| i + 1)]
                    .into_iter()
                    .flatten()
                {
                    r.push(i);
                    cc.push(j);
                    v.push(c(-1.0 + 0.3 * rnd(), 0.2 * rnd()));
                    r.push(j);
                    cc.push(i);
                    v.push(c(-1.0 + 0.3 * rnd(), 0.2 * rnd()));
                }
            }
        }
        (m * m, r, cc, v)
    };
    // a 3D grid: its top separators are large enough for the apex sweep, with rows
    // below them (the off-block product)
    let grid3 = |m: usize, rnd: &mut dyn FnMut() -> f64| {
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        let id = |a: usize, b: usize, e: usize| (a * m + b) * m + e;
        for a in 0..m {
            for b in 0..m {
                for e in 0..m {
                    let i = id(a, b, e);
                    r.push(i);
                    cc.push(i);
                    v.push(c(6.5 + rnd(), rnd()));
                    for j in [
                        (a + 1 < m).then(|| id(a + 1, b, e)),
                        (b + 1 < m).then(|| id(a, b + 1, e)),
                        (e + 1 < m).then(|| id(a, b, e + 1)),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        r.push(i);
                        cc.push(j);
                        v.push(c(-1.0 + 0.3 * rnd(), 0.2 * rnd()));
                        r.push(j);
                        cc.push(i);
                        v.push(c(-1.0 + 0.3 * rnd(), 0.2 * rnd()));
                    }
                }
            }
        }
        (m * m * m, r, cc, v)
    };
    for (n, r, cc, v) in [grid(60, &mut rnd), grid3(24, &mut rnd)] {
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &r, &cc, &v).unwrap();
        let solver = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
        for nrhs in [2usize, 3, 5] {
            let b: Vec<Complex<f64>> = (0..n * nrhs).map(|_| c(rnd(), rnd())).collect();
            let x = solver.solve_many(&b, nrhs).unwrap();
            for col in 0..nrhs {
                let bc: Vec<Complex<f64>> = (0..n).map(|i| b[col * n + i]).collect();
                let xc = solver.solve(&bc).unwrap();
                for i in 0..n {
                    assert!(
                        x[col * n + i] == xc[i],
                        "n {n} nrhs {nrhs} rhs {col} row {i}"
                    );
                }
            }
        }
    }
}

/// An analysis on a given ordering: its own ordering reproduces the factor, a
/// neighbouring pattern takes it with the elimination tree and counts recomputed, and a
/// non-permutation is refused.
#[test]
fn analysis_on_a_given_ordering() {
    let m = 12;
    let n = m * m;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for a in 0..m {
        for b in 0..m {
            let i = a * m + b;
            r.push(i);
            c.push(i);
            v.push(4.5 + 0.01 * i as f64);
            for j in [(a + 1 < m).then(|| i + m), (b + 1 < m).then(|| i + 1)]
                .into_iter()
                .flatten()
            {
                r.push(i);
                c.push(j);
                v.push(-1.0);
                r.push(j);
                c.push(i);
                v.push(-1.2);
            }
        }
    }
    let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
    let o = SolverSettings::default().with_matching(false);
    let s0 = LuSymbolic::analyze(&a, &o).unwrap();
    let f0 = s0.factor(&a, &o).unwrap();
    let o1 = o.clone().with_permutation(s0.permutation().into());
    let s1 = LuSymbolic::analyze(&a, &o1).unwrap();
    assert_eq!(s1.permutation(), s0.permutation());
    let f1 = s1.factor(&a, &o1).unwrap();
    assert_eq!(f1.factor_nnz(), f0.factor_nnz());
    let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
    assert_eq!(f1.solve(&b).unwrap(), f0.solve(&b).unwrap());
    // a neighbouring pattern (one more coupling) on the same ordering
    let (mut r2, mut c2, mut v2) = (r.clone(), c.clone(), v.clone());
    r2.extend([0, n - 1]);
    c2.extend([n - 1, 0]);
    v2.extend([-0.1, -0.1]);
    let a2 = GeneralCsc::<f64>::from_triplets(n, &r2, &c2, &v2).unwrap();
    let s2 = LuSymbolic::analyze(&a2, &o1).unwrap();
    let x = s2.factor(&a2, &o1).unwrap().solve(&b).unwrap();
    let mut ax = vec![0.0f64; n];
    for col in 0..n {
        for k in a2.col_ptr[col]..a2.col_ptr[col + 1] {
            ax[a2.row_idx[k]] += a2.values[k] * x[col];
        }
    }
    let res = ax
        .iter()
        .zip(&b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    assert!(res < 1e-12, "residual on the reused ordering {res:.1e}");
    // not a permutation
    let bad: Vec<usize> = (0..n).map(|i| i / 2).collect();
    assert!(LuSymbolic::analyze(&a, &o.clone().with_permutation(bad.into())).is_err());
}

#[test]
fn pivoting_triggered_small_diagonal() {
    // Small diagonal, large off-diagonals -> partial pivoting fires on
    // (nearly) every column. Well-conditioned overall, so the solve must
    // still hit a tiny residual: this isolates the pivoting/perm logic
    // (correctness) from numerical stability.
    let c = |re, im| Complex::new(re, im);
    let m = 6;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(0.3, 0.05)); // small diagonal
            if b + 1 < m {
                let q = idx(a, b + 1);
                rr.push(p);
                cc.push(q);
                vv.push(c(2.0, 0.3)); // large off-diagonal
                rr.push(q);
                cc.push(p);
                vv.push(c(1.5, -0.2));
            }
            if a + 1 < m {
                let q = idx(a + 1, b);
                rr.push(p);
                cc.push(q);
                vv.push(c(1.8, 0.1));
                rr.push(q);
                cc.push(p);
                vv.push(c(2.2, 0.4));
            }
        }
    }
    let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let f = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(resid(&a, &x, &b) < 1e-9, "residual {}", resid(&a, &x, &b));
}

#[test]
fn lu_pivot_u_knob_wired_and_solves() {
    // The tunable threshold `u` governs the left-looking LU pivot test. On a
    // well-scaled, diagonally-dominant grid the pivot never needs to move, so
    // every `u in [0, 1]` must solve to a tiny residual (the knob changes the
    // factor path but not correctness here). Verifies the field is threaded
    // end-to-end (SolverSettings -> KernelTuning -> kernel) and clamps.
    let c = |re, im| Complex::new(re, im);
    let m = 7;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(12.0, 1.0)); // dominant diagonal -> no interchange needed
            if b + 1 < m {
                let q = idx(a, b + 1);
                rr.push(p);
                cc.push(q);
                vv.push(c(-1.0, 0.2));
                rr.push(q);
                cc.push(p);
                vv.push(c(-1.3, -0.1));
            }
            if a + 1 < m {
                let q = idx(a + 1, b);
                rr.push(p);
                cc.push(q);
                vv.push(c(-1.1, 0.3));
                rr.push(q);
                cc.push(p);
                vv.push(c(-0.9, 0.15));
            }
        }
    }
    let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    for u in [0.0f64, 0.1, 0.5, 1.0] {
        let s = SolverSettings::default().with_pivot_threshold(u);
        let f = LuSolver::factor(&a, &s).unwrap();
        let x = f.solve(&b).unwrap();
        let mut ax = vec![Complex::new(0.0, 0.0); n];
        a.matvec(&x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-9, "pivot_u={u} residual {res}");
    }
    // Out-of-range values clamp into [0, 1].
    assert_eq!(
        SolverSettings::default()
            .with_pivot_threshold(5.0)
            .pivoting
            .threshold,
        1.0
    );
    assert_eq!(
        SolverSettings::default()
            .with_pivot_threshold(-2.0)
            .pivoting
            .threshold,
        0.0
    );
}

#[test]
fn static_pivot_reuse_across_value_sweep() {
    // Solver-in-the-loop: analyze the pattern once, then factor a *sweep* of
    // value sets that share it with static pivoting (`pivot_u = 0`, no pivot
    // search per column). On a diagonally-dominant family each static factor
    // solves accurately, and iterative refinement against the original matrix
    // recovers full accuracy - the frequency-sweep / time-stepping use case.
    let c = |re, im| Complex::new(re, im);
    let m = 8;
    let n = m * m;
    let idx = |a: usize, b: usize| a * m + b;
    let (mut rr, mut cc) = (Vec::new(), Vec::new());
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            if b + 1 < m {
                rr.push(p);
                cc.push(idx(a, b + 1));
                rr.push(idx(a, b + 1));
                cc.push(p);
            }
            if a + 1 < m {
                rr.push(p);
                cc.push(idx(a + 1, b));
                rr.push(idx(a + 1, b));
                cc.push(p);
            }
        }
    }
    let template =
        GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vec![c(1.0, 0.0); rr.len()])
            .unwrap();
    let analysis = LuSymbolic::analyze(&template, &SolverSettings::default()).unwrap();
    let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 4.0, 0.7)).collect();
    let static_opts = SolverSettings::default().with_pivot_threshold(0.0);
    for shift in [0.0, 1.5, -0.8, 3.0] {
        let vv: Vec<Complex<f64>> = rr
            .iter()
            .zip(&cc)
            .map(|(&i, &j)| {
                if i == j {
                    c(9.0 + shift, 1.0)
                } else {
                    c(-1.0, 0.2)
                }
            })
            .collect();
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        // Reuse the one analysis; static factor (no pivot search).
        let f = &analysis.factor(&a, &static_opts).unwrap();
        let x = f
            .solve_refined(&a, &b, &crate::RefinePolicy::steps(2))
            .unwrap()
            .0;
        let mut ax = vec![Complex::new(0.0, 0.0); n];
        a.matvec(&x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-9, "static reuse shift={shift} residual {res}");
    }
}

#[test]
fn complex_unsymmetric_2d_grid() {
    // 2D 5-point grid with unsymmetric neighbor couplings (right != left).
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
                vv.push(c(-1.0, 0.2)); // p,q
                rr.push(q);
                cc.push(p);
                vv.push(c(-2.0, 0.1)); // q,p (different!)
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
    let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let f = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    assert!(resid(&a, &x, &b) < 1e-9, "residual {}", resid(&a, &x, &b));
}

#[test]
fn complex_f32_lu_solves() {
    // The Complex<f32> LU path (used by the mixed-precision preconditioner).
    let c = |re: f32, im: f32| num_complex::Complex::<f32>::new(re, im);
    let m = 10;
    let n = m * m;
    let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            vv.push(c(20.0, 2.0));
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
    let a = GeneralCsc::<num_complex::Complex<f32>>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<num_complex::Complex<f32>> = (0..n).map(|i| c((i % 5) as f32 - 2.0, 1.0)).collect();
    let f = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let x = f.solve(&b).unwrap();
    let r = resid(&a, &x, &b);
    assert!(r < 1e-3, "f32 LU residual {}", r);
}

#[test]
fn phased_general_lu_analyze_once_factor_many() {
    // PARDISO workflow for the unsymmetric path: analyze the pattern once,
    // factor several value sets that share it - each must match the
    // one-shot factor's solve. The frequency-sweep / Newton use case.
    let c = |re, im| Complex::new(re, im);
    let m = 7;
    let n = m * m;
    let (mut rr, mut cc) = (Vec::new(), Vec::new());
    let idx = |a: usize, b: usize| a * m + b;
    for a in 0..m {
        for b in 0..m {
            let p = idx(a, b);
            rr.push(p);
            cc.push(p);
            if b + 1 < m {
                rr.push(p);
                cc.push(idx(a, b + 1));
                rr.push(idx(a, b + 1));
                cc.push(p);
            }
            if a + 1 < m {
                rr.push(p);
                cc.push(idx(a + 1, b));
                rr.push(idx(a + 1, b));
                cc.push(p);
            }
        }
    }
    // Template (values irrelevant) -> analyze once.
    let template =
        GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vec![c(1.0, 0.0); rr.len()])
            .unwrap();
    let analysis = LuSymbolic::analyze(&template, &SolverSettings::default()).unwrap();
    assert_eq!(analysis.n(), n);

    let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 4.0, 1.0)).collect();
    for shift in [0.0, 2.5, -1.0] {
        let vv: Vec<Complex<f64>> = rr
            .iter()
            .zip(&cc)
            .map(|(&i, &j)| {
                if i == j {
                    c(8.0 + shift, 1.0)
                } else {
                    c(-1.0, 0.2)
                }
            })
            .collect();
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let phased = &analysis.factor(&a, &SolverSettings::default()).unwrap();
        let one_shot = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
        let xp = phased.solve(&b).unwrap();
        let xo = one_shot.solve(&b).unwrap();
        for (p, o) in xp.iter().zip(&xo) {
            assert!((p - o).norm() < 1e-10);
        }
        assert!(resid(&a, &xp, &b) < 1e-8);
    }
}

#[test]
fn incomplete_lu_reduces_fill_and_still_solves() {
    // Unsymmetric grid: incomplete LU (drop_tol) must shrink nnz(L+U) yet
    // still drive iterative refinement to a small residual - the MoM
    // sparse-preconditioner configuration.
    let c = |re, im| Complex::new(re, im);
    let m = 14;
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
    let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
    let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

    let full = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
    let opts = SolverSettings::default().with_drop_tol(5e-2);
    let inc = LuSolver::factor(&a, &opts).unwrap();
    assert!(
        inc.factor_nnz() < full.factor_nnz(),
        "ILU should reduce fill: {} vs {}",
        inc.factor_nnz(),
        full.factor_nnz()
    );
    // The incomplete factor + a few refinement steps still solves accurately.
    let x = inc
        .solve_refined(&a, &b, &crate::RefinePolicy::steps(10))
        .unwrap()
        .0;
    assert!(resid(&a, &x, &b) < 1e-6, "residual {}", resid(&a, &x, &b));
}

/// A random unsymmetric matrix with a dominant diagonal (every column its
/// diagonal plus a few random rows, so `A + A^T` is much larger than `A`);
/// with `holes` every
/// seventh diagonal entry is left out, its column carrying the pivot of the
/// next row instead (a 2x2 swap the matching has to find).
fn unsymmetric_holes(n: usize, per_col: usize, seed: u64, holes: bool) -> GeneralCsc<f64> {
    let mut x = seed;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for j in 0..n {
        if holes && j % 7 == 0 && j + 1 < n {
            r.extend([j + 1, j]);
            c.extend([j, j + 1]);
            v.extend([4.0, 4.0]);
        } else {
            r.push(j);
            c.push(j);
            v.push(4.0 + (j % 3) as f64);
        }
        for _ in 0..per_col {
            let i = (next() % n as u64) as usize;
            if i != j {
                r.push(i);
                c.push(j);
                v.push(((next() % 200) as f64) / 100.0 - 1.0);
            }
        }
    }
    GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap()
}

/// The exact structure bounds the numeric factor: every `U12` column the
/// factorization keeps is in `ucols(s)`, and no supernode keeps more `L` rows
/// than `lrows(s)` has.
#[test]
fn exact_structure_bounds_the_factor() {
    for (seed, matching) in [(7u64, false), (11, true), (13, true)] {
        let a = unsymmetric_holes(600, 3, seed, matching);
        let opts = SolverSettings::default().with_matching(matching);
        let lusym = LuSymbolic::analyze(&a, &opts).unwrap();
        assert_eq!(lusym.has_matching(), matching);
        let num = factor_general_lu_numeric(&lusym, &a, &opts).unwrap();
        let (sym, _) = lusym.symb.sym_and_levels().unwrap();
        let sched = lusym.symb.ll_schedule().unwrap();
        let st = &lusym.structure;
        let (mut exact, mut symmetric, mut kept) = (0usize, 0usize, 0usize);
        let mut k = 0;
        for (s, sn) in sym.supernodes.iter().enumerate() {
            if sn.ncol == 0 {
                continue;
            }
            let (lk, uk) = (&num.l.rows[k], &num.ut.rows[k]);
            let (lrows, ucols) = (&st.rows_l(s)[sn.ncol..], &st.cols_u(s)[sn.ncol..]);
            assert!(
                lk.len() <= lrows.len(),
                "supernode {s}: {} L rows kept, {} predicted",
                lk.len(),
                lrows.len()
            );
            for &g in uk {
                assert!(
                    ucols.binary_search(&(g as Li)).is_ok(),
                    "supernode {s}: U column {g} outside the exact structure"
                );
            }
            exact += lrows.len() + ucols.len();
            symmetric += 2 * (sched.rows(s).len() - sn.ncol);
            kept += lk.len() + uk.len();
            k += 1;
        }
        eprintln!(
            "seed {seed} matching {matching}: off-block entries per column: symmetric {symmetric}, exact {exact}, kept {kept}"
        );
        assert!(
            exact < symmetric,
            "an unsymmetric pattern tightens the structure"
        );
    }
}

/// The row matching runs only where a diagonal entry is missing: with a full
/// diagonal the matrix is analyzed as given.
#[test]
fn matching_only_where_the_diagonal_needs_it() {
    let opts = SolverSettings::default();
    for (holes, expect) in [(false, false), (true, true)] {
        let a = unsymmetric_holes(300, 3, 5, holes);
        let lusym = LuSymbolic::analyze(&a, &opts).unwrap();
        assert_eq!(lusym.has_matching(), expect, "holes = {holes}");
    }
}

/// `||A^T x - b||inf / ||b||inf`.
fn transpose_residual<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
    let mut worst = 0.0f64;
    for j in 0..a.n {
        let mut acc = T::zero();
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            acc = acc + a.values[k] * x[a.row_idx[k]];
        }
        worst = worst.max((acc - b[j]).magnitude());
    }
    worst / b.iter().map(|v| v.magnitude()).fold(0.0, f64::max)
}

/// The transposed solve runs `U^T` forward and `L^T` backward on the same
/// panels, with and without the row matching, and a column-major block of
/// transposed solves is the columns solved one by one.
#[test]
fn lu_solve_transpose_on_the_same_factors() {
    use crate::numeric::krylov::Factorization;
    for holes in [false, true] {
        let a = unsymmetric_holes(500, 3, 9, holes);
        let s = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
        let b: Vec<f64> = (0..a.n).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
        let x = s.solve_transpose(&b).unwrap();
        let r = transpose_residual(&a, &x, &b);
        assert!(r < 1e-10, "holes {holes}: transposed residual {r:.1e}");
        let f: &dyn Factorization<f64> = &s;
        assert_eq!(f.solve_transpose(&b).unwrap(), x);
    }
    let c = |re, im| num_complex::Complex::new(re, im);
    let m = 30;
    let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..m * m {
        r.push(i);
        cc.push(i);
        v.push(c(4.0, 0.3));
        for (d, w) in [(1, c(-1.2, 0.1)), (m, c(-0.7, -0.2))] {
            if i + d < m * m {
                r.push(i + d);
                cc.push(i);
                v.push(w);
                r.push(i);
                cc.push(i + d);
                v.push(w * c(0.5, 0.4));
            }
        }
    }
    let a = GeneralCsc::from_triplets(m * m, &r, &cc, &v).unwrap();
    let s = LuSolver::factor(
        &a,
        &SolverSettings::default().with_ordering(crate::OrderingMethod::MetisND),
    )
    .unwrap();
    let b: Vec<_> = (0..a.n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
    let x = s.solve_transpose(&b).unwrap();
    assert!(transpose_residual(&a, &x, &b) < 1e-10);
}

/// The three direct solvers behind one trait object.
#[test]
fn every_direct_solver_is_a_factorization() {
    use crate::numeric::krylov::Factorization;
    use crate::{CscMatrix, KluSettings, KluSolver, LdltSolver};
    let n = 200;
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        r.push(i);
        c.push(i);
        v.push(4.0);
        if i + 1 < n {
            r.push(i + 1);
            c.push(i);
            v.push(-1.0);
        }
    }
    let lower = CscMatrix::<f64>::from_triplets(n, &r, &c, &v).unwrap();
    let (mut rf, mut cf, mut vf) = (r.clone(), c.clone(), v.clone());
    for k in 0..r.len() {
        if r[k] != c[k] {
            rf.push(c[k]);
            cf.push(r[k]);
            vf.push(v[k]);
        }
    }
    let full = GeneralCsc::<f64>::from_triplets(n, &rf, &cf, &vf).unwrap();
    let s = SolverSettings::default();
    let solvers: Vec<Box<dyn Factorization<f64>>> = vec![
        Box::new(LdltSolver::factor(&lower, &s).unwrap()),
        Box::new(LuSolver::factor(&full, &s).unwrap()),
        Box::new(KluSolver::factor(&full, &KluSettings::default()).unwrap()),
    ];
    let b: Vec<f64> = (0..2 * n).map(|i| (i % 9) as f64 - 4.0).collect();
    let reference = solvers[0].solve_many(&b, 2).unwrap();
    for (k, f) in solvers.iter().enumerate() {
        assert_eq!(f.n(), n);
        let x = f.solve_many(&b, 2).unwrap();
        for (p, q) in x.iter().zip(&reference) {
            assert!((p - q).abs() < 1e-10);
        }
        let (xr, out) = f
            .solve_refined(&full, &b[..n], &crate::RefinePolicy::default())
            .unwrap();
        assert!(out.omega < 1e-14 && (xr[0] - x[0]).abs() < 1e-10);
        assert_eq!(f.n_perturbed(), 0);
        // two solves here, one more for the reference on the first
        assert_eq!(f.diagnostics().solves.calls, if k == 0 { 3 } else { 2 });
    }
}
