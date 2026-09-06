//! General (unsymmetric) sparse matrix in CSC form.
//!
//! [`CscMatrix`](crate::sparse::csc::CscMatrix) stores only the lower triangle
//! of a *symmetric* matrix. The unsymmetric LU path
//! ([`crate::numeric::multifrontal_lu`]) needs the **full** matrix with both
//! triangles and genuinely distinct `A_ij != A_ji`; this type provides that.

use crate::error::RslabError;
use crate::scalar::Scalar;

/// A general sparse matrix in compressed-sparse-column form. Every stored entry
/// is kept as given (no symmetry assumption). Column `j` occupies
/// `col_ptr[j]..col_ptr[j+1]` of `row_idx`/`values`, sorted by row with
/// duplicates summed.
#[derive(Debug, Clone)]
pub struct GeneralCsc<T> {
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub values: Vec<T>,
}

impl<T: Scalar> GeneralCsc<T> {
    /// Build from `(row, col, value)` triplets. Indices are 0-based; duplicates
    /// are summed; rows within a column are sorted.
    pub fn from_triplets(
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[T],
    ) -> Result<Self, RslabError> {
        if rows.len() != cols.len() || cols.len() != vals.len() {
            return Err(RslabError::InvalidInput(
                "GeneralCsc::from_triplets: rows/cols/vals length mismatch".to_string(),
            ));
        }
        for (&r, &c) in rows.iter().zip(cols) {
            if r >= n || c >= n {
                return Err(RslabError::InvalidInput(format!(
                    "GeneralCsc::from_triplets: index ({r}, {c}) out of bounds for {n}x{n}"
                )));
            }
        }
        // Bucket by column.
        let mut counts = vec![0usize; n];
        for &c in cols {
            counts[c] += 1;
        }
        let mut col_start = vec![0usize; n + 1];
        for j in 0..n {
            col_start[j + 1] = col_start[j] + counts[j];
        }
        let nnz = vals.len();
        let mut tmp_row = vec![0usize; nnz];
        let mut tmp_val = vec![T::zero(); nnz];
        let mut next = col_start[..n].to_vec();
        for k in 0..nnz {
            let c = cols[k];
            let pos = next[c];
            next[c] += 1;
            tmp_row[pos] = rows[k];
            tmp_val[pos] = vals[k];
        }
        // Sort within each column and sum duplicates.
        let mut col_ptr = Vec::with_capacity(n + 1);
        col_ptr.push(0);
        let mut row_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        let mut buf: Vec<(usize, T)> = Vec::new();
        for j in 0..n {
            buf.clear();
            for p in col_start[j]..col_start[j + 1] {
                buf.push((tmp_row[p], tmp_val[p]));
            }
            buf.sort_by_key(|&(r, _)| r);
            let mut i = 0;
            while i < buf.len() {
                let r = buf[i].0;
                let mut acc = buf[i].1;
                let mut j2 = i + 1;
                while j2 < buf.len() && buf[j2].0 == r {
                    acc = acc + buf[j2].1;
                    j2 += 1;
                }
                row_idx.push(r);
                values.push(acc);
                i = j2;
            }
            col_ptr.push(row_idx.len());
        }
        Ok(Self {
            n,
            col_ptr,
            row_idx,
            values,
        })
    }

    /// Number of stored nonzeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// General matrix-vector product `y = A x` (no symmetry).
    pub fn matvec(&self, x: &[T], y: &mut [T]) {
        for v in y.iter_mut() {
            *v = T::zero();
        }
        for (j, w) in self.col_ptr.windows(2).enumerate() {
            let xj = x[j];
            for k in w[0]..w[1] {
                let i = self.row_idx[k];
                y[i] = y[i] + self.values[k] * xj;
            }
        }
    }

    /// The transpose `A^T` (also general CSC). Column `i` of `A^T` is row `i` of
    /// `A`.
    pub fn transpose(&self) -> Self {
        let n = self.n;
        let nnz = self.nnz();
        let mut counts = vec![0usize; n];
        for &i in &self.row_idx {
            counts[i] += 1;
        }
        let mut col_ptr = vec![0usize; n + 1];
        for j in 0..n {
            col_ptr[j + 1] = col_ptr[j] + counts[j];
        }
        let mut row_idx = vec![0usize; nnz];
        let mut values = vec![T::zero(); nnz];
        let mut next = col_ptr[..n].to_vec();
        for j in 0..n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k]; // entry (i, j) -> transpose column i, row j
                let pos = next[i];
                next[i] += 1;
                row_idx[pos] = j;
                values[pos] = self.values[k];
            }
        }
        Self {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// One-norm `||A||_1 = max_j sum_i |a_ij|` (max absolute column sum) - the
    /// norm side of the Hager-Higham condition estimate (feral #94 port).
    pub fn one_norm(&self) -> f64 {
        let mut worst = 0.0f64;
        for j in 0..self.n {
            let mut colsum = 0.0f64;
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                colsum += self.values[k].magnitude();
            }
            if colsum > worst {
                worst = colsum;
            }
        }
        worst
    }

    /// Validate structural invariants - the full canonical-form contract the
    /// numeric paths rely on, mirroring the X6-hardened
    /// [`CscMatrix::validate`](crate::sparse::csc::CscMatrix::validate):
    /// lengths, `col_ptr[0] == 0`, monotone `col_ptr`, in-bounds rows, and
    /// **strictly increasing** row indices per column (which implies both
    /// sortedness and duplicate-freeness).
    ///
    /// The strict-increase check is load-bearing for correctness, not just
    /// hygiene: the KLU factorization scatters column values by *assignment*
    /// (`x[pre] = v`), so a duplicate row entry would silently overwrite its
    /// partner instead of summing, a wrong factor with no error, while
    /// [`matvec`](Self::matvec) sums duplicates. `from_triplets` always
    /// produces the canonical form; this guards hand-constructed matrices
    /// (the fields are `pub`).
    pub fn validate(&self) -> Result<(), RslabError> {
        if self.col_ptr.len() != self.n + 1 {
            return Err(RslabError::InvalidInput(
                "GeneralCsc: bad col_ptr length".to_string(),
            ));
        }
        if self.row_idx.len() != self.values.len() {
            return Err(RslabError::InvalidInput(
                "GeneralCsc: row_idx/values length mismatch".to_string(),
            ));
        }
        // col_ptr must start at 0: a monotone col_ptr beginning at k > 0 with
        // col_ptr[n] == nnz would silently drop positions 0..k from every
        // column range (the X6 failure mode).
        if *self.col_ptr.first().unwrap_or(&0) != 0 {
            return Err(RslabError::InvalidInput(format!(
                "GeneralCsc: col_ptr[0] must be 0, got {}",
                self.col_ptr[0]
            )));
        }
        if *self.col_ptr.last().unwrap_or(&0) != self.row_idx.len() {
            return Err(RslabError::InvalidInput(
                "GeneralCsc: col_ptr[n] != nnz".to_string(),
            ));
        }
        // Monotone col_ptr: a non-monotone pointer whose endpoints line up
        // makes column ranges empty/overlapping and silently drops entries.
        for j in 0..self.n {
            if self.col_ptr[j + 1] < self.col_ptr[j] {
                return Err(RslabError::InvalidInput(format!(
                    "GeneralCsc: col_ptr not monotonically non-decreasing at column {} ({} > {})",
                    j,
                    self.col_ptr[j],
                    self.col_ptr[j + 1]
                )));
            }
        }
        for j in 0..self.n {
            let (start, end) = (self.col_ptr[j], self.col_ptr[j + 1]);
            for k in start..end {
                if self.row_idx[k] >= self.n {
                    return Err(RslabError::InvalidInput(format!(
                        "GeneralCsc: row index {} out of bounds in column {}",
                        self.row_idx[k], j
                    )));
                }
                // Strictly increasing rows: sorted AND duplicate-free.
                if k > start && self.row_idx[k] <= self.row_idx[k - 1] {
                    return Err(RslabError::InvalidInput(format!(
                        "GeneralCsc: row indices not strictly increasing in column {} \
                         ({} then {}); sort and sum duplicates (see from_triplets)",
                        j,
                        self.row_idx[k - 1],
                        self.row_idx[k]
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical() -> GeneralCsc<f64> {
        GeneralCsc::from_triplets(
            3,
            &[0, 2, 1, 0, 2],
            &[0, 0, 1, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .unwrap()
    }

    #[test]
    fn validate_accepts_canonical() {
        assert!(canonical().validate().is_ok());
    }

    /// X6 parity with `CscMatrix::validate`: a col_ptr that does not start at
    /// 0 silently drops the uncovered prefix entries from every column range.
    #[test]
    fn validate_rejects_nonzero_col_ptr_start() {
        let m = GeneralCsc::<f64> {
            n: 2,
            col_ptr: vec![1, 1, 2],
            row_idx: vec![0, 1],
            values: vec![1.0, 1.0],
        };
        let msg = format!("{}", m.validate().unwrap_err());
        assert!(msg.contains("col_ptr[0]"), "got: {msg}");
    }

    /// X6 parity: a non-monotone col_ptr with matching endpoints makes column
    /// ranges empty/overlapping and factors the wrong matrix.
    #[test]
    fn validate_rejects_non_monotone_col_ptr() {
        let m = GeneralCsc::<f64> {
            n: 3,
            col_ptr: vec![0, 2, 1, 2],
            row_idx: vec![0, 2],
            values: vec![1.0, 1.0],
        };
        let msg = format!("{}", m.validate().unwrap_err());
        assert!(msg.contains("monoton"), "got: {msg}");
    }

    /// Duplicates are the correctness-critical case: the KLU scatter assigns
    /// (`x[pre] = v`) while `matvec` sums, so a duplicate row silently
    /// produces a wrong factor. `validate` must reject it.
    #[test]
    fn validate_rejects_duplicate_rows() {
        let m = GeneralCsc::<f64> {
            n: 2,
            col_ptr: vec![0, 2, 3],
            row_idx: vec![0, 0, 1],
            values: vec![1.0, 2.0, 3.0],
        };
        let msg = format!("{}", m.validate().unwrap_err());
        assert!(msg.contains("strictly increasing"), "got: {msg}");
    }

    #[test]
    fn validate_rejects_unsorted_rows() {
        let m = GeneralCsc::<f64> {
            n: 2,
            col_ptr: vec![0, 2, 3],
            row_idx: vec![1, 0, 1],
            values: vec![1.0, 2.0, 3.0],
        };
        let msg = format!("{}", m.validate().unwrap_err());
        assert!(msg.contains("strictly increasing"), "got: {msg}");
    }

    #[test]
    fn validate_rejects_out_of_bounds_row() {
        let m = GeneralCsc::<f64> {
            n: 2,
            col_ptr: vec![0, 1, 2],
            row_idx: vec![0, 2],
            values: vec![1.0, 1.0],
        };
        let msg = format!("{}", m.validate().unwrap_err());
        assert!(msg.contains("out of bounds"), "got: {msg}");
    }
}
