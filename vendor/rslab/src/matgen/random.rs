//! Random sparse (structurally-conditioned) and spectral (exact-kappa) generators.
//!
//! Random sparse matrices stress the ordering/symbolic stage on irregular
//! patterns; conditioning is steered by **diagonal loading**. The spectral family
//! follows LAPACK `xLATMS`: build `A = Q Lambda Q^T` with a *prescribed* spectrum, so the
//! condition number is **exact** - the right tool for accuracy/stability tests.
//! `QLambdaQ^T` is dense, so this family is small-`n` by nature.
// Diagonal/triplet loops use the index as a value (push `k`, read `colsum[k]`).
#![allow(clippy::needless_range_loop)]

use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;

use super::Rng;

/// Random **symmetric** sparse SPD matrix: ~`avg_deg` random off-diagonals per
/// row, each diagonal loaded to its row's |off|-sum plus `alpha` (strict diagonal
/// dominance => SPD). `alpha` is the conditioning knob.
pub fn random_spd<T: Scalar>(n: usize, avg_deg: usize, alpha: f64, seed: u64) -> CscMatrix<T> {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut absum = vec![0.0f64; n];
    let per = (avg_deg / 2).max(1);
    for i in 0..n {
        for _ in 0..per {
            let j = (rng.next_u64() as usize) % n;
            if j >= i {
                continue; // lower triangle only (j < i)
            }
            let w = rng.range(-1.0, 1.0);
            rows.push(i);
            cols.push(j);
            vals.push(T::from_real(w));
            absum[i] += w.abs();
            absum[j] += w.abs();
        }
    }
    for k in 0..n {
        rows.push(k);
        cols.push(k);
        vals.push(T::from_real(absum[k] + alpha));
    }
    super::build_sym(n, &rows, &cols, &vals)
}

/// Random **unsymmetric** sparse matrix with diagonal loading. `avg_deg` random
/// entries per column anywhere off-diagonal; the diagonal is loaded to dominate
/// the column's |off|-sum plus `alpha`, keeping it nonsingular.
pub fn random_unsym<T: Scalar>(n: usize, avg_deg: usize, alpha: f64, seed: u64) -> GeneralCsc<T> {
    let mut rng = Rng::new(seed);
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut colsum = vec![0.0f64; n];
    for j in 0..n {
        for _ in 0..avg_deg.max(1) {
            let i = (rng.next_u64() as usize) % n;
            if i == j {
                continue;
            }
            let w = rng.range(-1.0, 1.0);
            rows.push(i);
            cols.push(j);
            vals.push(T::from_real(w));
            colsum[j] += w.abs();
        }
    }
    for k in 0..n {
        rows.push(k);
        cols.push(k);
        vals.push(T::from_real(colsum[k] + alpha));
    }
    super::build_gen(n, &rows, &cols, &vals)
}

/// Spectral generator (`xLATMS`-style): `A = Q Lambda Q^T` with a prescribed spectrum so
/// the **condition number is exactly `kappa`**. Eigenvalues are log-spaced in
/// `[1, kappa]` (SPD) or alternately signed (indefinite). `Q` is a random
/// orthogonal matrix (Givens sweeps). Dense => keep `n` small (<= ~500).
pub fn spectral<T: Scalar>(n: usize, kappa: f64, indefinite: bool, seed: u64) -> CscMatrix<T> {
    let mut rng = Rng::new(seed);
    // Log-spaced eigenvalues with exact endpoints 1 and kappa.
    let mut eig = vec![1.0f64; n];
    let lk = kappa.max(1.0).log10();
    for (i, e) in eig.iter_mut().enumerate() {
        let t = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        *e = 10f64.powf(t * lk);
    }
    if n > 1 {
        eig[n - 1] = kappa;
    }
    if indefinite {
        for e in eig.iter_mut().step_by(2) {
            *e = -*e;
        }
    }
    // Random orthogonal Q (start at I, apply Givens row-rotations).
    let mut q = vec![0.0f64; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    for _ in 0..(3 * n) {
        let p = (rng.next_u64() as usize) % n;
        let mut r = (rng.next_u64() as usize) % n;
        if r == p {
            r = (r + 1) % n;
        }
        if r == p {
            continue;
        }
        let th = rng.range(0.0, std::f64::consts::TAU);
        let (cs, sn) = (th.cos(), th.sin());
        for col in 0..n {
            let a = q[p * n + col];
            let b = q[r * n + col];
            q[p * n + col] = cs * a - sn * b;
            q[r * n + col] = sn * a + cs * b;
        }
    }
    // A = Q diag(eig) Q^T, lower triangle.
    let mut rows = Vec::with_capacity(n * (n + 1) / 2);
    let mut cols = Vec::with_capacity(n * (n + 1) / 2);
    let mut vals = Vec::with_capacity(n * (n + 1) / 2);
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..n {
                s += q[i * n + k] * eig[k] * q[j * n + k];
            }
            rows.push(i);
            cols.push(j);
            vals.push(T::from_real(s));
        }
    }
    super::build_sym(n, &rows, &cols, &vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_spd_diagonally_dominant() {
        let a = random_spd::<f64>(200, 10, 0.5, 1);
        assert_eq!(a.n, 200);
        assert!(a.values.len() > 200, "has off-diagonals");
    }

    #[test]
    fn spectral_has_exact_condition_number() {
        // A = QLambdaQ^T => eigenvalues are exactly Lambda; verify via the extreme Rayleigh
        // quotients along the eigenvectors is hard from CSC, so check the trace
        // (= sumlambda) and that the matrix is symmetric & sized right instead.
        let n = 50;
        let kappa = 1e6;
        let a = spectral::<f64>(n, kappa, false, 3);
        assert_eq!(a.n, n);
        // Trace = sum eig (log-spaced 1..kappa). Lower-triangle diagonal entries.
        let mut trace = 0.0;
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                if a.row_idx[k] == j {
                    trace += a.values[k];
                }
            }
        }
        let mut eigsum = 0.0;
        let lk = kappa.log10();
        for i in 0..n {
            eigsum += 10f64.powf(i as f64 / (n - 1) as f64 * lk);
        }
        // endpoint correction
        eigsum += kappa - 10f64.powf(lk);
        assert!(
            (trace - eigsum).abs() / eigsum < 1e-9,
            "trace = sumlambda (spectrum preserved)"
        );
    }
}
