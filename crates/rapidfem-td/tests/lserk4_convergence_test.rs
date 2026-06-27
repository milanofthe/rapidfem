// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// Analytical convergence pin for the explicit LSERK4 integrator
// (explicit.rs). The semi-discrete DG system is the linear ODE
// dy/dt = A*y, whose exact flow over time T is the matrix exponential:
// y(T) = exp(T*A)*y0. The Carpenter-Kennedy 5-stage low-storage RK4
// scheme is fourth-order accurate, so its global error must fall as
// O(h^4) as the step h is refined. This test pins both the absolute
// accuracy at a small step and the fourth-order convergence rate.
//
// Operator A: the same small, dense, SKEW-SYMMETRIC wave operator as the
// propagator golden (propagator_expm_test.rs):
//
//        A = [[ 0, -L],
//             [ L,  0]]          L = tridiag(-1, 2, -1) on 3 nodes
//
// with n = 6. Skew symmetry puts every eigenvalue on the imaginary axis,
// so the dynamics are purely oscillatory and energy-conserving: the exact
// regime for testing an explicit RK integrator's phase/amplitude error
// without artificial dissipation muddying the order. The eigenvalues of L
// are 2 - 2*cos(k*pi/4), k = 1..3, the largest being ~3.414, so the
// spectral radius of A is sqrt(3.414) ~ 1.848. LSERK4's imaginary-axis
// stability bound is |lambda|*h <~ 2.8, i.e. h <~ 1.51 here; every step
// below stays well inside it.
//
// The reference exp(T*A)*y0 is computed in-test by a dense scaling-and-
// squaring matrix exponential (Taylor series on the scaled matrix), good
// to ~machine precision on this tiny, well-conditioned operator, so the
// reference is independent of the integrator under test.

use rapidfem_td::explicit::LserkWorkspace;

const N: usize = 6;

/// Build the 6x6 row-major skew-symmetric wave operator A = [[0,-L],[L,0]],
/// L = tridiag(-1, 2, -1).
fn build_a() -> [f64; N * N] {
    // L on the 3-node grid.
    let l = [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]];
    let mut a = [0.0; N * N];
    for i in 0..3 {
        for j in 0..3 {
            // top-right block: -L  (rows 0..3, cols 3..6)
            a[i * N + (j + 3)] = -l[i][j];
            // bottom-left block: +L (rows 3..6, cols 0..3)
            a[(i + 3) * N + j] = l[i][j];
        }
    }
    a
}

/// Matrix-free A*x for a dense row-major n*n matrix.
fn matvec(a: &[f64], x: &[f64], ax: &mut [f64]) {
    for i in 0..N {
        let mut s = 0.0;
        for j in 0..N {
            s += a[i * N + j] * x[j];
        }
        ax[i] = s;
    }
}

/// Dense n*n matrix product C = A*B (row-major).
fn matmul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; N * N];
    for i in 0..N {
        for k in 0..N {
            let aik = a[i * N + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..N {
                c[i * N + j] += aik * b[k * N + j];
            }
        }
    }
    c
}

/// exp(M) for a dense n*n matrix by scaling and squaring with a Taylor
/// series on the scaled matrix. Reference-quality on a small, modestly
/// scaled operator: scale so ||M/2^s|| is tiny, sum the Taylor series to
/// high order, then square back s times.
fn dense_expm(m: &[f64]) -> Vec<f64> {
    // 1-norm (max abs column sum) to pick the scaling.
    let mut norm: f64 = 0.0;
    for j in 0..N {
        let mut col = 0.0;
        for i in 0..N {
            col += m[i * N + j].abs();
        }
        norm = norm.max(col);
    }
    // Scale so the scaled norm is well below 1 (fast, accurate Taylor).
    let mut s = 0u32;
    while norm / (1u64 << s) as f64 > 0.25 {
        s += 1;
    }
    let scale = 1.0 / (1u64 << s) as f64;
    let ms: Vec<f64> = m.iter().map(|x| x * scale).collect();

    // Taylor: E = sum_{k>=0} ms^k / k!.
    let mut e = identity();
    let mut term = identity(); // ms^k / k!
    for k in 1..=30u32 {
        term = matmul(&term, &ms);
        let inv = 1.0 / k as f64;
        for x in &mut term {
            *x *= inv;
        }
        for (ei, ti) in e.iter_mut().zip(&term) {
            *ei += *ti;
        }
    }

    // Square back: E <- E^(2^s).
    for _ in 0..s {
        e = matmul(&e, &e);
    }
    e
}

fn identity() -> Vec<f64> {
    let mut id = vec![0.0; N * N];
    for i in 0..N {
        id[i * N + i] = 1.0;
    }
    id
}

/// Dense matrix-vector product y = M*x.
fn apply(m: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; N];
    for i in 0..N {
        let mut s = 0.0;
        for j in 0..N {
            s += m[i * N + j] * x[j];
        }
        y[i] = s;
    }
    y
}

/// Integrate dy/dt = A*y from 0 to t_end with `nsteps` LSERK4 steps.
fn integrate(a: &[f64], y0: &[f64], t_end: f64, nsteps: usize) -> Vec<f64> {
    let dt = t_end / nsteps as f64;
    let mut y = y0.to_vec();
    let mut ws = LserkWorkspace::new();
    for _ in 0..nsteps {
        ws.step_into(|x, ax| matvec(a, x, ax), &mut y, dt);
    }
    y
}

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn l2_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

#[test]
fn lserk4_matches_exact_exponential_and_is_fourth_order() {
    let a = build_a();
    // A deterministic, non-symmetric initial state (i+1)/7, matching the
    // propagator golden's v0.
    let y0: Vec<f64> = (0..N).map(|i| (i as f64 + 1.0) / 7.0).collect();

    let t_end = 2.0;

    // EXACT solution y(T) = exp(T*A)*y0 (reference-quality dense expm).
    let expm = dense_expm(&a.iter().map(|x| x * t_end).collect::<Vec<_>>());
    let exact = apply(&expm, &y0);

    // Self-check the reference: A skew-symmetric => exp(T*A) orthogonal =>
    // ||exact|| == ||y0|| to machine precision.
    let n0 = l2(&y0);
    let nref = l2(&exact);
    eprintln!(
        "reference norm check: ||y0|| = {n0:.15}, ||exp(TA)y0|| = {nref:.15}, \
         rel diff = {:.2e}",
        (nref - n0).abs() / n0
    );
    assert!(
        (nref - n0).abs() / n0 < 1e-13,
        "dense expm reference is not orthogonal: {:.2e}",
        (nref - n0).abs() / n0
    );

    // (1) ACCURACY: a sufficiently small step must hit the exact solution
    // to a tight tolerance. With T = 2 over 4096 steps (h ~ 4.9e-4,
    // h*rho ~ 9e-4, far below the CFL limit) the O(h^4) error is tiny.
    let fine = integrate(&a, &y0, t_end, 4096);
    let acc_err = l2_diff(&fine, &exact) / nref;
    eprintln!("(1) accuracy: 4096 steps, rel err vs exp(TA)y0 = {acc_err:.3e}");
    assert!(
        acc_err < 1e-10,
        "LSERK4 fine-step solution off exact exp(TA)y0: rel err {acc_err:.3e}",
    );

    // (2) 4th-ORDER CONVERGENCE: halve the step repeatedly; the global
    // error must fall by ~16 each halving (observed order ~ 4).
    let step_counts = [16usize, 32, 64, 128, 256, 512];
    let errs: Vec<f64> = step_counts
        .iter()
        .map(|&n| l2_diff(&integrate(&a, &y0, t_end, n), &exact) / nref)
        .collect();
    for (n, e) in step_counts.iter().zip(&errs) {
        let h = t_end / *n as f64;
        eprintln!("    nsteps = {n:>4}  h = {h:.4e}  rel err = {e:.4e}");
    }

    let mut orders = Vec::new();
    for w in errs.windows(2) {
        let order = (w[0] / w[1]).log2();
        orders.push(order);
    }
    for (pair, ord) in step_counts.windows(2).zip(&orders) {
        eprintln!(
            "(2) order over [{:>4} -> {:>4}] steps = {ord:.3}",
            pair[0], pair[1]
        );
    }
    // Every refinement (away from any round-off floor) must show a clean
    // fourth-order rate. The defining property of LSERK4.
    for (pair, ord) in step_counts.windows(2).zip(&orders) {
        assert!(
            (3.6..=4.2).contains(ord),
            "observed order over [{} -> {}] steps = {ord:.3}, not ~4",
            pair[0],
            pair[1],
        );
    }

    // (3) ENERGY: for the lossless skew-symmetric operator the integrator
    // must preserve ||y|| over the whole transient to within its own
    // accuracy. Use the fine-step run: ||y(T)|| ~ ||y0||.
    let energy_drift = (l2(&fine) - n0).abs() / n0;
    eprintln!("(3) energy drift over T (4096 steps) = {energy_drift:.3e}");
    assert!(
        energy_drift < 1e-8,
        "energy not conserved by LSERK4: drift {energy_drift:.3e}",
    );
}
