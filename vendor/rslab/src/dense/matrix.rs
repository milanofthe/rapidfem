use crate::error::RslabError;
use crate::scalar::Scalar;

/// Symmetric matrix stored as full nxn column-major. Only the lower triangle
/// is meaningful; the strict upper triangle is ignored on input.
/// Entry (i, j) is at index j*n + i. Size: n*n `T` values.
///
/// Generic over the scalar field `T` (defaulting to `f64`). This is the dense
/// per-front representation consumed by the Bunch-Kaufman kernel.
pub struct SymmetricMatrix<T = f64> {
    pub n: usize,
    pub data: Vec<T>,
}

impl<T: Scalar> SymmetricMatrix<T> {
    /// Create a new nxn symmetric matrix initialized to zero.
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![T::zero(); n * n],
        }
    }

    /// Create a symmetric matrix from a flat column-major vector.
    /// The lower triangle is authoritative; the upper triangle is ignored.
    pub fn from_column_major(n: usize, data: Vec<T>) -> Result<Self, RslabError> {
        if data.len() != n * n {
            return Err(RslabError::InvalidInput(format!(
                "matrix data length {} != expected {} for n={}",
                data.len(),
                n * n,
                n
            )));
        }
        Ok(Self { n, data })
    }

    /// Create a symmetric matrix from a dense 2D lower-triangular representation.
    /// `entries` provides (i, j, value) triples where i >= j.
    pub fn from_lower_triangle(n: usize, entries: &[(usize, usize, T)]) -> Self {
        let mut mat = Self::zeros(n);
        for &(i, j, v) in entries {
            mat.set(i, j, v);
        }
        mat
    }

    /// Get entry (i, j), reading from lower triangle.
    /// For i >= j, returns data[j*n + i].
    /// For i < j, returns data[i*n + j] (symmetric).
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        if i >= j {
            self.data[j * self.n + i]
        } else {
            self.data[i * self.n + j]
        }
    }

    /// Set entry (i, j) in the lower triangle.
    /// Also sets (j, i) for symmetry in the stored data.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: T) {
        if i >= j {
            self.data[j * self.n + i] = val;
        } else {
            self.data[i * self.n + j] = val;
        }
    }

    /// Validate the matrix for factorization input.
    /// Checks: n > 0, data length, no NaN/Inf in lower triangle.
    pub fn validate(&self) -> Result<(), RslabError> {
        if self.n == 0 {
            return Err(RslabError::InvalidInput(
                "matrix dimension is zero".to_string(),
            ));
        }
        if self.data.len() != self.n * self.n {
            return Err(RslabError::InvalidInput(format!(
                "matrix data length {} != expected {} for n={}",
                self.data.len(),
                self.n * self.n,
                self.n
            )));
        }
        // Check lower triangle for NaN/Inf
        for j in 0..self.n {
            for i in j..self.n {
                let val = self.data[j * self.n + i];
                if !val.is_finite() {
                    return Err(RslabError::InvalidInput(format!(
                        "matrix contains NaN or Inf at index ({},{})",
                        i, j
                    )));
                }
            }
        }
        Ok(())
    }

    /// Symmetric matrix-vector product: y = A * x.
    /// Uses only the lower triangle.
    pub fn symv(&self, x: &[T], y: &mut [T]) {
        let n = self.n;
        for yi in y.iter_mut().take(n) {
            *yi = T::zero();
        }
        for j in 0..n {
            // Diagonal
            y[j] = y[j] + self.data[j * n + j] * x[j];
            // Off-diagonal (lower triangle)
            for i in (j + 1)..n {
                let a_ij = self.data[j * n + i];
                y[i] = y[i] + a_ij * x[j];
                y[j] = y[j] + a_ij * x[i];
            }
        }
    }
}
