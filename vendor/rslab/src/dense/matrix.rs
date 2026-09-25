//! A dense symmetric matrix for checking the sparse readers in tests.

use crate::scalar::Scalar;

/// Symmetric matrix stored as full nxn column-major; entry (i, j) of the
/// lower triangle is at index j*n + i.
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
}
