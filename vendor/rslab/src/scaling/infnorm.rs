//! Knight-Ruiz ∞-norm iterative equilibration for sparse symmetric
//! matrices.
//!
//! Given a symmetric CSC matrix `A`, compute a diagonal `d` such that
//! each row of `D·A·D` has infinity-norm ≈ 1, where `D = diag(d)`.
//! This is the same algorithm used by the dense path in
//! `src/dense/equilibrate.rs`, adapted to iterate over lower-triangular
//! CSC storage.
//!
//! Phase 2.2.3 follow-up: the dense BK factorization succeeds on
//! HYDCAR20 / METHANL8 / SWOPF / HATFLDG because it equilibrates the
//! matrix before BK. The sparse multifrontal path was missing this
//! step - MC64 matching happens to classify these matrices as already
//! balanced even when their row norms span 4+ orders of magnitude.
//! Porting the dense path's equilibration recovers these matrices for
//! the sparse path.
//!
//! Algorithm (Jacobi-style, converges in the same number of iterations
//! as the Gauss-Seidel variant used by `dense::equilibrate` while being
//! simpler to implement over CSC lower-triangle storage):
//!
//! 1. Initialize `d = 1`.
//! 2. Repeat up to `max_iter` times:
//!    a. For each row `i`, compute `max_i = max_j |d[i]·a[i,j]·d[j]|`.
//!    b. Update `d[i] /= sqrt(max_i)` for every row whose `max_i > 0`.
//!    c. Stop when `max_i |1 − max_i|` falls below `tol`.

use crate::scaling::ScalingInfo;
use crate::sparse::csc::CscMatrix;

/// Compute the Knight-Ruiz ∞-norm symmetric scaling vector for a
/// lower-triangular symmetric CSC matrix. Returns the diagonal `d`
/// such that `D·A·D` has unit-∞ rows, paired with `ScalingInfo::Applied`.
pub fn compute_infnorm(matrix: &CscMatrix) -> (Vec<f64>, ScalingInfo) {
    let n = matrix.n;
    if n == 0 {
        return (Vec::new(), ScalingInfo::Applied);
    }
    let mut d = vec![1.0f64; n];

    // 10 iterations is the same cap the dense path uses. Most matrices
    // converge in 2-4 iterations; a few pathological ones need all 10.
    let max_iter = 10;
    let tol = 1e-8;

    // Work buffer for the row ∞-norms.
    let mut row_max = vec![0.0f64; n];

    for _ in 0..max_iter {
        // Reset the row-max buffer.
        for r in row_max.iter_mut() {
            *r = 0.0;
        }

        // Accumulate row maxes by scanning the lower triangle once.
        // Each (i, j) entry with i >= j contributes to row i (via the
        // explicit storage) and to row j (by symmetry).
        //
        // The row-j accumulation is hoisted to a register-resident
        // `col_max` across the inner k-loop, then folded into
        // `row_max[j]` once at column end. The diagonal entry (i == j)
        // is folded into `col_max` only - its `row_max[i]` write would
        // be overwritten by the end-of-column store, so we skip the
        // memory traffic and rely on `col_max` to carry the diagonal's
        // contribution. Off-diagonal entries (i > j) update both
        // `row_max[i]` and `col_max`.
        //
        // Bit-identical to the prior formulation: max(·,·) is
        // associative on non-NaN inputs (every `v` is `|·|` of finite
        // products), so combining via a register accumulator then
        // folding into `row_max[j]` produces the same value as in-place
        // updates in any iteration order.
        for j in 0..n {
            let col_start = matrix.col_ptr[j];
            let col_end = matrix.col_ptr[j + 1];
            let dj = d[j];
            let mut col_max = row_max[j];

            for k in col_start..col_end {
                let i = matrix.row_idx[k];
                let v = (d[i] * matrix.values[k] * dj).abs();
                if i != j && v > row_max[i] {
                    row_max[i] = v;
                }
                if v > col_max {
                    col_max = v;
                }
            }
            row_max[j] = col_max;
        }

        // Update diagonal and check convergence.
        let mut max_dev = 0.0f64;
        for i in 0..n {
            let m = row_max[i];
            if m > 0.0 {
                d[i] /= m.sqrt();
                let dev = (m - 1.0).abs();
                if dev > max_dev {
                    max_dev = dev;
                }
            }
            // Rows with all-zero entries keep d[i] at the current value
            // (initially 1.0) - they are structurally zero and the
            // downstream numeric phase will reject them as singular
            // pivots.
        }

        if max_dev < tol {
            break;
        }
    }

    (d, ScalingInfo::Applied)
}

/// One-pass symmetric ∞-norm equilibration `sᵢ = 1/√maxⱼ|Aᵢⱼ|` (a single
/// Knight-Ruiz step). Cheaper than the iterative [`compute_infnorm`] and the
/// historical [`crate::LdltSolver`] default: it tolerates a zero diagonal (the
/// row max is taken over off-diagonals) and never iterates. An all-zero row is
/// left unscaled (`sᵢ = 1`), surfacing as a singular pivot during factorization.
/// The scan folds in the symmetric partner `(j, i)` of every stored `(i, j)`, so
/// it is correct for a lower-triangular *or* a full symmetric store (`max` is
/// idempotent).
pub fn compute_onepass(matrix: &CscMatrix) -> (Vec<f64>, ScalingInfo) {
    let n = matrix.n;
    let mut row_max = vec![0.0f64; n];
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let m = matrix.values[k].abs();
            if m > row_max[i] {
                row_max[i] = m;
            }
            if i != j && m > row_max[j] {
                row_max[j] = m;
            }
        }
    }
    let s = row_max
        .iter()
        .map(|&r| if r > 0.0 { 1.0 / r.sqrt() } else { 1.0 })
        .collect();
    (s, ScalingInfo::Applied)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    /// Diagonal matrix diag(2, 3, 5). The oracle scaling is
    /// d = [1/sqrt(2), 1/sqrt(3), 1/sqrt(5)], so that
    /// D·A·D = diag(1, 1, 1).
    #[test]
    fn diag_3x3() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        let (d, _info) = compute_infnorm(&m);
        let expected = [1.0 / 2f64.sqrt(), 1.0 / 3f64.sqrt(), 1.0 / 5f64.sqrt()];
        for i in 0..3 {
            assert!(
                (d[i] - expected[i]).abs() < 1e-12,
                "d[{}] = {} != {}",
                i,
                d[i],
                expected[i]
            );
        }
    }

    /// 2x2 matrix [[4, 2], [2, 9]]. Row max [i=0]: max(|4|, |2|) = 4;
    /// row max [i=1]: max(|2|, |9|) = 9. After one KR sweep:
    /// d = [1/2, 1/3]. Check D·A·D row norms converge to 1.
    #[test]
    fn sym_2x2() {
        let m = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[4.0, 2.0, 9.0]).unwrap();
        let (d, _) = compute_infnorm(&m);
        // D·A·D:
        //   [d0*d0*4, d0*d1*2]
        //   [d0*d1*2, d1*d1*9]
        let a00 = d[0] * d[0] * 4.0;
        let a01 = d[0] * d[1] * 2.0;
        let a11 = d[1] * d[1] * 9.0;
        let row0 = a00.abs().max(a01.abs());
        let row1 = a01.abs().max(a11.abs());
        assert!((row0 - 1.0).abs() < 1e-6, "row0 max = {}", row0);
        assert!((row1 - 1.0).abs() < 1e-6, "row1 max = {}", row1);
    }

    /// Arrow matrix: diagonal [2, 3, 4, 5, 6, 7] with (5, 0..=4) = 1.
    /// Row 5 has 5 off-diagonal entries plus the diagonal 7; its
    /// initial ∞-norm is max(1, 1, 1, 1, 1, 7) = 7. The first KR
    /// sweep should shrink d[5] by sqrt(7).
    #[test]
    fn arrow_6x6() {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for j in 0..6 {
            rows.push(j);
            cols.push(j);
            vals.push((j + 2) as f64);
        }
        for j in 0..5 {
            rows.push(5);
            cols.push(j);
            vals.push(1.0);
        }
        let m = CscMatrix::from_triplets(6, &rows, &cols, &vals).unwrap();
        let (d, _) = compute_infnorm(&m);
        // After KR convergence, every row's max-magnitude entry in
        // D·A·D should be ≈ 1.
        for i in 0..6 {
            let mut row_max = 0.0f64;
            for j in 0..6 {
                // Look up a[i, j] from lower triangle
                let (ii, jj) = if i >= j { (i, j) } else { (j, i) };
                let mut v = 0.0;
                for k in m.col_ptr[jj]..m.col_ptr[jj + 1] {
                    if m.row_idx[k] == ii {
                        v = m.values[k];
                        break;
                    }
                }
                let scaled = (d[i] * v * d[j]).abs();
                if scaled > row_max {
                    row_max = scaled;
                }
            }
            assert!(
                (row_max - 1.0).abs() < 1e-6,
                "row {} max = {}, expected 1",
                i,
                row_max
            );
        }
    }
}
