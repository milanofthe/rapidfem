use crate::error::RslabError;
use crate::scalar::Scalar;

/// Compressed Sparse Column (CSC) matrix storage for symmetric matrices.
///
/// Only the lower triangle is stored. `col_ptr[j]..col_ptr[j+1]` gives the
/// range of entries in column j. Row indices within each column are sorted
/// in ascending order.
///
/// Generic over the scalar field `T` (defaulting to `f64`): `T = f64` is the
/// real symmetric path, `T = Complex<f64>` the complex-symmetric (PARDISO
/// `mtype 6`) path. The sparsity structure (`col_ptr`/`row_idx`) is
/// scalar-agnostic; only `values` carries the field.
#[derive(Debug, Clone)]
pub struct CscMatrix<T = f64> {
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub values: Vec<T>,
}

/// Symmetric sparsity pattern (full, not just lower triangle).
/// Used for AMD ordering and elimination tree construction.
#[derive(Debug, Clone)]
pub struct CscPattern {
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
}

impl<T: Scalar> CscMatrix<T> {
    /// Number of stored nonzeros (lower triangle only).
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Build a CSC matrix from coordinate (triplet) format.
    ///
    /// Entries must be lower-triangle (row >= col). Duplicate entries are summed.
    /// Row indices within each column are sorted.
    pub fn from_triplets(
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[T],
    ) -> Result<Self, RslabError> {
        if rows.len() != cols.len() || cols.len() != vals.len() {
            return Err(RslabError::InvalidInput(
                "triplet arrays must have equal length".to_string(),
            ));
        }

        // Count entries per column
        let mut col_counts = vec![0usize; n];
        for &c in cols {
            if c >= n {
                return Err(RslabError::InvalidInput(format!(
                    "column index {} out of bounds for n={}",
                    c, n
                )));
            }
            col_counts[c] += 1;
        }

        // Build col_ptr
        let mut col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            col_ptr[j + 1] = col_ptr[j] + col_counts[j];
        }
        let nnz = col_ptr[n];

        // Place entries
        let mut row_idx = vec![0usize; nnz];
        let mut values = vec![T::zero(); nnz];
        let mut offsets = col_ptr[..n].to_vec();
        for k in 0..rows.len() {
            let (r, c) = (rows[k], cols[k]);
            if r >= n {
                return Err(RslabError::InvalidInput(format!(
                    "row index {} out of bounds for n={}",
                    r, n
                )));
            }
            if r < c {
                return Err(RslabError::InvalidInput(format!(
                    "triplet {} ({}, {}) is upper-triangle; \
                     CscMatrix stores only the lower triangle (row >= col)",
                    k, r, c
                )));
            }
            let pos = offsets[c];
            row_idx[pos] = r;
            values[pos] = vals[k];
            offsets[c] += 1;
        }

        // Sort each column by row index, summing duplicates
        let mut result = CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        };
        result.sort_and_sum_duplicates();
        Ok(result)
    }

    /// Sort row indices within each column and sum duplicate entries.
    fn sort_and_sum_duplicates(&mut self) {
        // Two-pass approach: first sort and deduplicate into a compact representation,
        // then rebuild the arrays.
        let mut new_row_idx = Vec::with_capacity(self.row_idx.len());
        let mut new_values = Vec::with_capacity(self.values.len());
        let mut new_col_ptr = vec![0usize; self.n + 1];

        for j in 0..self.n {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            let col_start = new_row_idx.len();

            if start == end {
                new_col_ptr[j + 1] = col_start;
                continue;
            }

            // Collect (row, val) pairs for this column and sort by row
            let mut pairs: Vec<(usize, T)> = (start..end)
                .map(|k| (self.row_idx[k], self.values[k]))
                .collect();
            pairs.sort_unstable_by_key(|&(r, _)| r);

            // Deduplicate by summing
            let mut prev_row = pairs[0].0;
            let mut prev_val = pairs[0].1;
            for &(r, v) in &pairs[1..] {
                if r == prev_row {
                    prev_val = prev_val + v;
                } else {
                    new_row_idx.push(prev_row);
                    new_values.push(prev_val);
                    prev_row = r;
                    prev_val = v;
                }
            }
            new_row_idx.push(prev_row);
            new_values.push(prev_val);

            new_col_ptr[j + 1] = new_row_idx.len();
        }

        self.col_ptr = new_col_ptr;
        self.row_idx = new_row_idx;
        self.values = new_values;
    }

    /// Validate the CSC structure.
    pub fn validate(&self) -> Result<(), RslabError> {
        if self.col_ptr.len() != self.n + 1 {
            return Err(RslabError::InvalidInput(format!(
                "col_ptr length {} != n+1={}",
                self.col_ptr.len(),
                self.n + 1
            )));
        }
        if self.row_idx.len() != self.values.len() {
            return Err(RslabError::InvalidInput(
                "row_idx and values length mismatch".to_string(),
            ));
        }
        // X6 residual (repo-review-2026-06-09-verification.md): `col_ptr`
        // must start at 0. A monotone `col_ptr` beginning at `k > 0` with
        // `col_ptr[n] == nnz` passes every other check (length, monotone,
        // in-bounds, sorted, lower-triangle) while positions `0..k` of
        // `row_idx`/`values` are never covered by any column range -
        // silently dropped and never factored. Completes the column-pointer
        // contract the monotonicity check below began.
        if self.col_ptr[0] != 0 {
            return Err(RslabError::InvalidInput(format!(
                "col_ptr[0] must be 0, got {}",
                self.col_ptr[0]
            )));
        }
        if self.col_ptr[self.n] != self.row_idx.len() {
            return Err(RslabError::InvalidInput("col_ptr[n] != nnz".to_string()));
        }
        // col_ptr must be monotonically non-decreasing (X6). Without this a
        // non-monotone `ia` whose endpoints line up (col_ptr[0] == 0,
        // col_ptr[n] == nnz) passes every check below - in-bounds, sorted,
        // lower-triangle - yet `col_ptr[j] > col_ptr[j+1]` makes column j's
        // range empty and overlaps adjacent columns, silently dropping entries
        // and factoring the wrong matrix. Checked up front so the `start..end`
        // ranges used below are well-formed.
        for j in 0..self.n {
            if self.col_ptr[j + 1] < self.col_ptr[j] {
                return Err(RslabError::InvalidInput(format!(
                    "col_ptr not monotonically non-decreasing at column {} ({} > {})",
                    j,
                    self.col_ptr[j],
                    self.col_ptr[j + 1]
                )));
            }
        }
        for j in 0..self.n {
            let start = self.col_ptr[j];
            let end = self.col_ptr[j + 1];
            for k in start..end {
                if self.row_idx[k] >= self.n {
                    return Err(RslabError::InvalidInput(format!(
                        "row index {} out of bounds in column {}",
                        self.row_idx[k], j
                    )));
                }
                if self.row_idx[k] < j {
                    return Err(RslabError::InvalidInput(format!(
                        "row index {} in column {} is upper-triangle; \
                         CscMatrix stores only the lower triangle (row >= col)",
                        self.row_idx[k], j
                    )));
                }
            }
            // Check sorted
            for k in (start + 1)..end {
                if self.row_idx[k] <= self.row_idx[k - 1] {
                    return Err(RslabError::InvalidInput(format!(
                        "row indices not sorted in column {} ({}>={})",
                        j,
                        self.row_idx[k - 1],
                        self.row_idx[k]
                    )));
                }
            }
        }
        Ok(())
    }

    /// Expand the lower-triangle CSC to a full symmetric sparsity pattern.
    ///
    /// The result contains both (i,j) and (j,i) for every off-diagonal entry.
    /// Used for AMD ordering and elimination tree construction.
    pub fn symmetric_pattern(&self) -> CscPattern {
        // Count entries per column in the full pattern
        let mut col_counts = vec![0usize; self.n];
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                col_counts[j] += 1; // lower triangle entry in column j
                if i != j {
                    col_counts[i] += 1; // transpose entry in column i
                }
            }
        }

        // Build col_ptr
        let mut pat_col_ptr = vec![0usize; self.n + 1];
        for j in 0..self.n {
            pat_col_ptr[j + 1] = pat_col_ptr[j] + col_counts[j];
        }
        let pat_nnz = pat_col_ptr[self.n];
        let mut pat_row_idx = vec![0usize; pat_nnz];

        // Place entries
        let mut offsets = pat_col_ptr[..self.n].to_vec();
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                // (i, j) in lower triangle
                pat_row_idx[offsets[j]] = i;
                offsets[j] += 1;
                if i != j {
                    // (j, i) - transpose
                    pat_row_idx[offsets[i]] = j;
                    offsets[i] += 1;
                }
            }
        }

        // Sort row indices within each column
        for j in 0..self.n {
            let start = pat_col_ptr[j];
            let end = pat_col_ptr[j + 1];
            pat_row_idx[start..end].sort_unstable();
        }

        CscPattern {
            n: self.n,
            col_ptr: pat_col_ptr,
            row_idx: pat_row_idx,
        }
    }

    /// Symmetric matrix-vector product: y = A * x.
    ///
    /// Uses only the stored lower triangle; implicitly applies symmetry.
    pub fn symv(&self, x: &[T], y: &mut [T]) {
        for yi in y.iter_mut().take(self.n) {
            *yi = T::zero();
        }
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                let v = self.values[k];
                y[i] = y[i] + v * x[j];
                if i != j {
                    y[j] = y[j] + v * x[i];
                }
            }
        }
    }

    /// Convert to dense symmetric matrix.
    pub fn to_dense(&self) -> crate::dense::matrix::SymmetricMatrix<T> {
        self.to_dense_into(Vec::new())
    }

    /// Densify into a caller-provided buffer (reused to avoid the
    /// `n * n` allocation on every call). The buffer is cleared and
    /// resized to `n * n` zeros before the lower triangle is
    /// scattered in; pass `Vec::new()` for a fresh allocation.
    ///
    /// Byte-exact equivalent to `to_dense()` for the same input.
    /// Used by `FactorWorkspace` to pool the dense-fast-path buffer
    /// across calls - see
    /// `dev/research/phase-2.5.x-to-dense-pooling.md`.
    pub fn to_dense_into(&self, mut buf: Vec<T>) -> crate::dense::matrix::SymmetricMatrix<T> {
        let nn = self.n * self.n;
        buf.clear();
        buf.resize(nn, T::zero());
        // `from_triplets` guarantees all stored entries are lower-
        // triangle (row >= col), so every `(i, j)` lands at
        // `data[j*n + i]`.
        for j in 0..self.n {
            let col = j * self.n;
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                buf[col + i] = self.values[k];
            }
        }
        crate::dense::matrix::SymmetricMatrix {
            n: self.n,
            data: buf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_3x3() -> CscMatrix {
        // [ 2 -1  0 ]
        // [-1  3 -1 ]
        // [ 0 -1  4 ]
        CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[2.0, -1.0, 3.0, -1.0, 4.0],
        )
        .unwrap()
    }

    #[test]
    fn test_from_triplets_basic() {
        let m = sample_3x3();
        assert_eq!(m.n, 3);
        assert_eq!(m.nnz(), 5);
        m.validate().unwrap();
    }

    #[test]
    fn test_from_triplets_duplicate_summing() {
        let m = CscMatrix::from_triplets(2, &[0, 0, 1], &[0, 0, 1], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(m.nnz(), 2);
        assert_eq!(m.values[0], 3.0); // 1.0 + 2.0
        assert_eq!(m.values[1], 3.0);
    }

    #[test]
    fn test_symmetric_pattern() {
        let m = sample_3x3();
        let pat = m.symmetric_pattern();
        assert_eq!(pat.n, 3);
        // Full pattern: (0,0), (1,0), (0,1), (1,1), (2,1), (1,2), (2,2)
        // = 7 entries total
        assert_eq!(pat.col_ptr[3], 7);

        // Column 0: rows 0, 1
        assert_eq!(&pat.row_idx[pat.col_ptr[0]..pat.col_ptr[1]], &[0, 1]);
        // Column 1: rows 0, 1, 2
        assert_eq!(&pat.row_idx[pat.col_ptr[1]..pat.col_ptr[2]], &[0, 1, 2]);
        // Column 2: rows 1, 2
        assert_eq!(&pat.row_idx[pat.col_ptr[2]..pat.col_ptr[3]], &[1, 2]);
    }

    #[test]
    fn test_symv() {
        let m = sample_3x3();
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0; 3];
        m.symv(&x, &mut y);
        // A * x = [2-2, -1+6-3, -2+12] = [0, 2, 10]
        assert!((y[0] - 0.0).abs() < 1e-14);
        assert!((y[1] - 2.0).abs() < 1e-14);
        assert!((y[2] - 10.0).abs() < 1e-14);
    }

    #[test]
    fn test_to_dense_roundtrip() {
        let m = sample_3x3();
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), 2.0);
        assert_eq!(dense.get(1, 0), -1.0);
        assert_eq!(dense.get(0, 1), -1.0);
        assert_eq!(dense.get(1, 1), 3.0);
        assert_eq!(dense.get(2, 1), -1.0);
        assert_eq!(dense.get(1, 2), -1.0);
        assert_eq!(dense.get(2, 2), 4.0);
        assert_eq!(dense.get(2, 0), 0.0);
    }

    #[test]
    fn test_validate_rejects_bad_input() {
        let mut m = sample_3x3();
        m.row_idx[0] = 5; // out of bounds
        assert!(m.validate().is_err());
    }

    /// Issue #4: upper-triangle triplets must be rejected, not silently
    /// accepted. The two matrices below describe the same symmetric
    /// system; previously the upper-triangle form was accepted and
    /// produced different solve results downstream.
    #[test]
    fn test_from_triplets_rejects_upper_triangle() {
        // Lower-triangle form: (0,0)=2, (1,0)=1, (1,1)=2
        let lower = CscMatrix::from_triplets(2, &[0, 1, 1], &[0, 0, 1], &[2.0, 1.0, 2.0]).unwrap();
        lower.validate().unwrap();

        // Upper-triangle form of the same matrix: (0,0)=2, (0,1)=1, (1,1)=2.
        // Must be rejected - previously was silently accepted.
        let err = CscMatrix::from_triplets(2, &[0, 0, 1], &[0, 1, 1], &[2.0, 1.0, 2.0])
            .expect_err("upper-triangle triplet must be rejected");
        let msg = format!("{}", err);
        assert!(
            msg.contains("upper-triangle"),
            "error should mention upper-triangle, got: {}",
            msg
        );
    }

    /// `validate()` must also reject upper-triangle row indices, in case
    /// a `CscMatrix` is constructed by a path that bypasses
    /// `from_triplets` (e.g. direct field assignment in tests).
    #[test]
    fn test_validate_rejects_upper_triangle_row() {
        let mut m = sample_3x3();
        // Force an upper-triangle entry: column 1's first row becomes 0
        // (row 0, col 1 is upper-triangle).
        m.row_idx[2] = 0;
        let err = m
            .validate()
            .expect_err("validate must reject upper-triangle row");
        let msg = format!("{}", err);
        assert!(
            msg.contains("upper-triangle"),
            "error should mention upper-triangle, got: {}",
            msg
        );
    }

    /// The storage layer must hold a complex-symmetric matrix (A = Aᵀ with
    /// complex entries), not just `f64`. This is the point of the generic
    /// threading: structure code is shared, only `values` carries the field.
    /// Oracle hand-computed below.
    #[test]
    fn complex_storage_symv_and_dense() {
        use num_complex::Complex;
        let c = |re, im| Complex::new(re, im);
        // Lower triangle of A = [[1+i, 2], [2, 3-i]] (symmetric, A = Aᵀ).
        let m: CscMatrix<Complex<f64>> = CscMatrix::from_triplets(
            2,
            &[0, 1, 1],
            &[0, 0, 1],
            &[c(1.0, 1.0), c(2.0, 0.0), c(3.0, -1.0)],
        )
        .unwrap();
        m.validate().unwrap();
        assert_eq!(m.nnz(), 3);

        // y = A·x with x = [1, i].
        //   y0 = (1+i)·1 + 2·i        = 1 + 3i
        //   y1 = 2·1     + (3-i)·i    = 3 + 3i   (since -i² = +1)
        let x = [c(1.0, 0.0), c(0.0, 1.0)];
        let mut y = [c(0.0, 0.0); 2];
        m.symv(&x, &mut y);
        assert_eq!(y[0], c(1.0, 3.0));
        assert_eq!(y[1], c(3.0, 3.0));

        // Densify and read back, including the symmetric upper-triangle access.
        let dense = m.to_dense();
        assert_eq!(dense.get(0, 0), c(1.0, 1.0));
        assert_eq!(dense.get(1, 0), c(2.0, 0.0));
        assert_eq!(dense.get(0, 1), c(2.0, 0.0)); // symmetry
        assert_eq!(dense.get(1, 1), c(3.0, -1.0));
    }

    #[test]
    fn test_diagonal_matrix() {
        let m: CscMatrix =
            CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(m.nnz(), 3);
        let pat = m.symmetric_pattern();
        assert_eq!(pat.col_ptr[3], 3); // no off-diagonal, so 3 entries total
    }

    #[test]
    fn test_empty_matrix() {
        let m: CscMatrix = CscMatrix::from_triplets(3, &[], &[], &[]).unwrap();
        assert_eq!(m.nnz(), 0);
        m.validate().unwrap();
        let pat = m.symmetric_pattern();
        assert_eq!(pat.col_ptr[3], 0);
    }

    #[test]
    fn test_kkt_structure() {
        // Small KKT: [H  A^T; A  -delta*I]
        // H = [2 0; 0 3], A = [1 1], delta = 1e-8
        // Full matrix (3x3):
        // [ 2    0    1  ]
        // [ 0    3    1  ]
        // [ 1    1  -1e-8]
        let m: CscMatrix<f64> = CscMatrix::from_triplets(
            3,
            &[0, 1, 2, 2, 2],
            &[0, 1, 0, 1, 2],
            &[2.0, 3.0, 1.0, 1.0, -1e-8],
        )
        .unwrap();
        assert_eq!(m.nnz(), 5);
        m.validate().unwrap();

        // symv check
        let x = [1.0, 1.0, 1.0];
        let mut y = [0.0; 3];
        m.symv(&x, &mut y);
        assert!((y[0] - 3.0).abs() < 1e-14); // 2 + 0 + 1
        assert!((y[1] - 4.0).abs() < 1e-14); // 0 + 3 + 1
        assert!((y[2] - (2.0 - 1e-8)).abs() < 1e-14); // 1 + 1 - 1e-8
    }

    /// X6 (dev/research/repo-review-2026-06-09.md): `validate()` must reject a
    /// non-monotone `col_ptr`. A valid CSC requires `col_ptr` to be
    /// monotonically non-decreasing (the standard column-pointer contract);
    /// without that check a non-monotone `ia` whose endpoints line up
    /// (`col_ptr[0] == 0`, `col_ptr[n] == nnz`) passes every other check yet
    /// produces empty/overlapping column ranges, so entries are silently
    /// dropped and the wrong matrix is factored.
    ///
    /// Witness: n = 3, nnz = 2, `col_ptr = [0, 2, 1, 2]`. The endpoints line up
    /// (`col_ptr[3] == 2 == nnz`), every row index is in-bounds, lower-triangle
    /// and sorted within its (non-empty) range, but `col_ptr[1] = 2 >
    /// col_ptr[2] = 1`, so column 1's range `[2, 1)` is empty and column 2
    /// re-reads index 1. Pre-fix `validate()` returned `Ok`.
    #[test]
    fn validate_rejects_non_monotone_col_ptr() {
        let m = CscMatrix {
            n: 3,
            col_ptr: vec![0, 2, 1, 2],
            // index 0 -> (row 0, col 0); index 1 -> (row 2, col 0 and re-read
            // as col 2). Both lower-triangle and in-bounds.
            row_idx: vec![0, 2],
            values: vec![1.0, 1.0],
        };
        let err = m
            .validate()
            .expect_err("non-monotone col_ptr must be rejected (X6)");
        let msg = format!("{}", err);
        assert!(
            msg.contains("col_ptr") && msg.contains("monoton"),
            "error should mention non-monotone col_ptr, got: {}",
            msg
        );
    }

    /// X6 residual (repo-review-2026-06-09-verification.md): `validate()`
    /// must reject a `col_ptr` that does not start at 0. A monotone
    /// `col_ptr` beginning at `k > 0` with `col_ptr[n] == nnz` passes the
    /// length, monotonicity, `col_ptr[n] == nnz`, in-bounds, sorted and
    /// lower-triangle checks, yet positions `0..k` of `row_idx`/`values`
    /// fall outside every column range and are silently dropped - the
    /// matrix factored is missing those entries.
    ///
    /// Witness: n = 2, nnz = 2, `col_ptr = [1, 1, 2]`. Monotone,
    /// `col_ptr[2] == 2 == nnz`; column 0's range `[1, 1)` is empty and
    /// column 1's range `[1, 2)` reads only index 1, so `row_idx[0]` /
    /// `values[0]` are never covered. Pre-fix `validate()` returned `Ok`.
    #[test]
    fn validate_rejects_nonzero_col_ptr_start() {
        let m = CscMatrix {
            n: 2,
            col_ptr: vec![1, 1, 2],
            row_idx: vec![0, 1],
            values: vec![1.0, 1.0],
        };
        let err = m
            .validate()
            .expect_err("a col_ptr that does not start at 0 must be rejected (X6)");
        let msg = format!("{}", err);
        assert!(
            msg.contains("col_ptr[0]"),
            "error should mention col_ptr[0], got: {}",
            msg
        );
    }
}
