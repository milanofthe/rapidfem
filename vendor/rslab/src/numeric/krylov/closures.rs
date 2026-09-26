//! Matrix-free operators and preconditioners given as closures.

use super::*;

use crate::error::RslabError;
use crate::scalar::Scalar;

/// A closure block matvec `op(x, y, s)` (`Y <- A*X`, column-major `n x s`) as a
/// [`LinearOperator`]: the natural form for a matrix-free operator that
/// captures its own assembly data and scratch. `FnMut`, since the Krylov
/// solvers issue applies one at a time.
///
/// ```
/// use rslab::{gmres_block, FnOperator, FnPreconditioner, KrylovSettings};
/// let n = 3;
/// let op = FnOperator::new(n, |x: &[f64], y: &mut [f64], s: usize| {
///     for k in 0..n * s {
///         y[k] = 2.0 * x[k];
///     }
/// });
/// let pc = FnPreconditioner::new(|r: &[f64], z: &mut [f64], _s: usize| {
///     z.copy_from_slice(r);
///     Ok(())
/// });
/// let s = KrylovSettings::default().with_tol(1e-12);
/// let r = gmres_block(&op, &[2.0; 3], 1, &pc, &s, None, None).unwrap();
/// assert!(r.x.iter().all(|&v| (v - 1.0).abs() < 1e-12));
/// ```
pub struct FnOperator<F> {
    f: std::cell::RefCell<F>,
    n: usize,
}

impl<F> FnOperator<F> {
    /// Wrap the block matvec of an `n x n` operator.
    pub fn new(n: usize, f: F) -> Self {
        Self {
            f: std::cell::RefCell::new(f),
            n,
        }
    }
}

impl<T: Scalar, F: FnMut(&[T], &mut [T], usize)> LinearOperator<T> for FnOperator<F> {
    fn n(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        (self.f.borrow_mut())(x, y, 1)
    }
    fn apply_block(&self, x: &[T], y: &mut [T], s: usize) {
        (self.f.borrow_mut())(x, y, s)
    }
}

/// A closure block preconditioner `pc(r, z, s)` (`Z <- M^-1*R`, column-major
/// `n x s`) as a [`Preconditioner`].
pub struct FnPreconditioner<G> {
    f: std::cell::RefCell<G>,
}

impl<G> FnPreconditioner<G> {
    /// Wrap a block preconditioner apply.
    pub fn new(f: G) -> Self {
        Self {
            f: std::cell::RefCell::new(f),
        }
    }
}

impl<T: Scalar, G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>> Preconditioner<T>
    for FnPreconditioner<G>
{
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        (self.f.borrow_mut())(r, z, 1)
    }
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, _n: usize) -> Result<(), RslabError> {
        (self.f.borrow_mut())(r, z, s)
    }
}
