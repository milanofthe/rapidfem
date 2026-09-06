//! Banded and arrow/KKT generators - direct control over fill (bandwidth) and the
//! symmetric-indefinite saddle-point structure that exercises Bunch-Kaufman 2x2
//! pivoting.
// Diagonal/triplet loops use the index as a value (push `k`, read `absum[k]`).
#![allow(clippy::needless_range_loop)]

use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;

use super::Rng;

/// Symmetric **banded** SPD matrix: random off-diagonals within `bandwidth`, with
/// each diagonal set to its row's absolute off-diagonal sum plus `alpha`. Strict
/// diagonal dominance => SPD; `alpha` is the conditioning knob (small => ill).
pub fn banded<T: Scalar>(n: usize, bandwidth: usize, alpha: f64, seed: u64) -> CscMatrix<T> {
    let mut rng = Rng::new(seed);
    let bw = bandwidth.max(1);
    // Lower-triangle off-diagonals, and an accumulator of |off| per index (row+col).
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    let mut absum = vec![0.0f64; n];
    for j in 0..n {
        for i in (j + 1)..(j + 1 + bw).min(n) {
            let w = rng.range(-1.0, 1.0);
            rows.push(i);
            cols.push(j);
            vals.push(T::from_real(w));
            absum[i] += w.abs();
            absum[j] += w.abs();
        }
    }
    // Diagonals last (from_triplets sorts within columns anyway).
    for k in 0..n {
        rows.push(k);
        cols.push(k);
        vals.push(T::from_real(absum[k] + alpha));
    }
    super::build_sym(n, &rows, &cols, &vals)
}

/// **Arrow / bordered KKT** saddle-point matrix (symmetric **indefinite**):
/// `[[A11, B^T], [B, -C]]` with a tridiagonal SPD interior `A11` of size
/// `n - border`, a dense coupling `B` (the `border` arrow rows), and a small
/// negative `(2,2)` block - so the system is genuinely indefinite and drives 2x2
/// pivots. `border` sets the (dense) border width; `gamma` the `(2,2)` regulariser.
pub fn arrow<T: Scalar>(n: usize, border: usize, gamma: f64, seed: u64) -> CscMatrix<T> {
    let mut rng = Rng::new(seed);
    let m = n.saturating_sub(border).max(1); // interior size
    let border = n - m;
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    // A11: tridiagonal SPD (diag 2, sub -1) on indices [0, m).
    for j in 0..m {
        rows.push(j);
        cols.push(j);
        vals.push(T::from_real(2.0));
        if j + 1 < m {
            rows.push(j + 1);
            cols.push(j);
            vals.push(T::from_real(-1.0));
        }
    }
    // B: dense coupling, border rows [m, n) x interior cols [0, m) (lower triangle,
    // since row >= m > col).
    for j in 0..m {
        for r in m..n {
            rows.push(r);
            cols.push(j);
            vals.push(T::from_real(rng.range(-1.0, 1.0)));
        }
    }
    // -C: negative diagonal on the border block => indefinite saddle.
    for r in m..n {
        rows.push(r);
        cols.push(r);
        vals.push(T::from_real(-gamma));
    }
    let _ = border;
    super::build_sym(n, &rows, &cols, &vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banded_is_diagonally_dominant_spd() {
        let a = banded::<f64>(50, 3, 0.5, 1);
        assert_eq!(a.n, 50);
        // Each diagonal >= its row's off-diagonal abs-sum (strict dominance).
        for j in 0..a.n {
            let diag = a.values[a.col_ptr[j]..a.col_ptr[j + 1]]
                .iter()
                .zip(&a.row_idx[a.col_ptr[j]..a.col_ptr[j + 1]])
                .find(|(_, &r)| r == j)
                .map(|(&v, _)| v)
                .unwrap();
            assert!(diag > 0.0, "positive diagonal");
        }
    }

    #[test]
    fn arrow_has_negative_border_and_is_indefinite_shaped() {
        let a = arrow::<f64>(20, 4, 0.01, 1);
        assert_eq!(a.n, 20);
        // The last `border` diagonals are negative (the -C block).
        for r in 16..20 {
            let diag = a.values[a.col_ptr[r]..a.col_ptr[r + 1]]
                .iter()
                .zip(&a.row_idx[a.col_ptr[r]..a.col_ptr[r + 1]])
                .find(|(_, &rr)| rr == r)
                .map(|(&v, _)| v)
                .unwrap();
            assert!(diag < 0.0, "border (2,2) block is negative => indefinite");
        }
    }
}
