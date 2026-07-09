//! General (unsymmetric) sparse matrix in CSC form.
//!
//! [`CscMatrix`](crate::sparse::csc::CscMatrix) stores only the lower triangle
//! of a *symmetric* matrix. The unsymmetric LU path
//! ([`crate::numeric::multifrontal_lu`]) needs the **full** matrix with both
//! triangles and genuinely distinct `A_ij ≠ A_ji`; this type provides that.

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
                    "GeneralCsc::from_triplets: index ({r}, {c}) out of bounds for {n}×{n}"
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

    /// The transpose `Aᵀ` (also general CSC). Column `i` of `Aᵀ` is row `i` of
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
                let i = self.row_idx[k]; // entry (i, j) → transpose column i, row j
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

    /// Validate structural invariants (lengths, bounds).
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
        if *self.col_ptr.last().unwrap_or(&0) != self.row_idx.len() {
            return Err(RslabError::InvalidInput(
                "GeneralCsc: col_ptr[n] != nnz".to_string(),
            ));
        }
        for &i in &self.row_idx {
            if i >= self.n {
                return Err(RslabError::InvalidInput(
                    "GeneralCsc: row index out of bounds".to_string(),
                ));
            }
        }
        Ok(())
    }
}
