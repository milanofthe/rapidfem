//! The operator and preconditioner interfaces of the Krylov solvers, with
//! their implementations for the sparse matrices and every factor type.

use crate::error::RslabError;
use crate::numeric::ldlt::LdltSolver;
use crate::numeric::settings::{SolverSettings, Threads};
use crate::scalar::{Demote, Scalar};
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use num_complex::Complex;

/// A linear operator `A`: applies `y = A x`. The Krylov solvers depend only on
/// this trait, so the operator may be an explicit sparse matrix
/// ([`CscMatrix`] symmetric / [`GeneralCsc`] general) **or matrix-free** - e.g.
/// a fast multipole (FMM/MLFMA) MoM operator the caller implements. RLA then
/// only factors the sparse near-field as the [`Preconditioner`].
pub trait LinearOperator<T: Scalar> {
    /// The system dimension.
    fn n(&self) -> usize;
    /// Write `y <- A x`. `x` and `y` have length `n`.
    fn apply(&self, x: &[T], y: &mut [T]);
    /// Block apply: `Y[:,c] <- A X[:,c]` for `c in 0..s`, with `X`,`Y` **column-
    /// major** `nxs` (RHS `c` is the contiguous slice `[c*n, (c+1)*n)`). The
    /// default loops the single-vector [`apply`](Self::apply); explicit-matrix
    /// operators override it with an amortized block matvec (each matrix entry
    /// loaded once for all `s` columns - the BLAS-3 arithmetic intensity that
    /// makes a multi-RHS solve pay over `s` separate ones).
    fn apply_block(&self, x: &[T], y: &mut [T], s: usize) {
        let n = self.n();
        for c in 0..s {
            self.apply(&x[c * n..c * n + n], &mut y[c * n..c * n + n]);
        }
    }
}

impl<T: Scalar> LinearOperator<T> for CscMatrix<T> {
    fn n(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.symv(x, y);
    }
    /// Amortized block symv: each lower-triangle entry `(i,j,v)` is loaded once
    /// and scattered to all `s` columns (`y[:,c] += v*x[j,c]`, and symmetrically
    /// `y[j,c] += v*x[i,c]` off the diagonal) - the BLAS-3 reuse a multi-RHS
    /// solve buys over `s` separate `symv`s.
    fn apply_block(&self, x: &[T], y: &mut [T], s: usize) {
        let n = self.n;
        for v in y.iter_mut() {
            *v = T::zero();
        }
        for j in 0..n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                let v = self.values[k];
                if i != j {
                    for c in 0..s {
                        let cb = c * n;
                        y[cb + i] = y[cb + i] + v * x[cb + j];
                        y[cb + j] = y[cb + j] + v * x[cb + i];
                    }
                } else {
                    for c in 0..s {
                        let cb = c * n;
                        y[cb + i] = y[cb + i] + v * x[cb + j];
                    }
                }
            }
        }
    }
}

impl<T: Scalar> LinearOperator<T> for GeneralCsc<T> {
    fn n(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.matvec(x, y);
    }
    /// Amortized block matvec: each entry `(i,j,v)` is loaded once and applied to
    /// all `s` columns (`y[i,c] += v*x[j,c]`).
    fn apply_block(&self, x: &[T], y: &mut [T], s: usize) {
        let n = self.n;
        for v in y.iter_mut() {
            *v = T::zero();
        }
        for j in 0..n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                let v = self.values[k];
                for c in 0..s {
                    let cb = c * n;
                    y[cb + i] = y[cb + i] + v * x[cb + j];
                }
            }
        }
    }
}

/// A preconditioner `M ~ A`: applies `z = M^-1 r`. Implemented by the direct
/// solvers (see [`Factorization`]), the low-precision factors and
/// [`NoPreconditioner`] (the unpreconditioned baseline).
pub trait Preconditioner<T: Scalar> {
    /// Write `z <- M^-1 r`. `r` and `z` have length `n`.
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError>;
    /// Block apply: `Z[:,c] <- M^-1 R[:,c]` for `c in 0..s`, with `R`,`Z` **column-
    /// major** `nxs`. The default loops [`apply`](Self::apply); a factored solver
    /// overrides it with a block triangular solve (`solve_many`) that loads each
    /// `L`/`D`/`U` value once for all `s` columns.
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        for c in 0..s {
            self.apply(&r[c * n..c * n + n], &mut z[c * n..c * n + n])?;
        }
        Ok(())
    }
    /// Thread policy the **solve phase** should honour. A factored
    /// preconditioner returns the resolved [`Threads`] budget it was built with, so
    /// [`gmres_block`](super::gmres_block)'s parallel orthogonalization runs in a pool of the **same**
    /// width - factor and solve share one concurrency budget instead of the solve
    /// silently fanning out over the global pool (the embedded / solver-in-the-loop
    /// design point). The default [`Threads::Ambient`] means "use the caller's
    /// current pool" - the behaviour for [`NoPreconditioner`] and any preconditioner
    /// that carries no factorization budget.
    fn solve_threads(&self) -> Threads {
        Threads::Ambient
    }
}

/// The identity preconditioner `M = I` (`z = r`): unpreconditioned iteration,
/// the baseline against which a real preconditioner's iteration count is read.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoPreconditioner;

impl<T: Scalar> Preconditioner<T> for NoPreconditioner {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        z.copy_from_slice(r);
        Ok(())
    }
}

/// A factor computed in a lower-precision field, applied inside an iteration
/// in the higher one: `F` factors `A` demoted to `H::Low` (see [`Demote`])
/// and every solve demotes its right-hand side, solves and promotes the
/// result. The factor takes half the memory and its kernels run in single
/// precision; the Krylov method (or [`solve_refined`](Factorization::solve_refined)
/// against the full-precision matrix) recovers the accuracy of `H`, the
/// standard mixed-precision preconditioner for large MoM and FEM systems.
///
/// Build it from any factor of the demoted matrix, the analysis reused:
///
/// ```
/// # fn main() -> Result<(), rslab::RslabError> {
/// use rslab::prelude::*;
/// use num_complex::Complex64;
/// let a = GeneralCsc::<Complex64>::from_triplets(
///     2, &[0, 1, 0], &[0, 1, 1],
///     &[Complex64::new(4.0, 1.0), Complex64::new(3.0, 0.0), Complex64::new(1.0, 0.0)],
/// )?;
/// let sym = LuSymbolic::analyze(&a, &SolverSettings::default())?;
/// let m = MixedPrecision::new(sym.factor(&a.demoted(), &SolverSettings::default())?);
/// let z = Factorization::<Complex64>::solve(&m, &[Complex64::new(1.0, 0.0); 2])?;
/// # let _ = z; Ok(()) }
/// ```
pub struct MixedPrecision<F> {
    factor: F,
}

impl<F> MixedPrecision<F> {
    /// Wrap a factor of the demoted matrix.
    pub fn new(factor: F) -> Self {
        Self { factor }
    }

    /// The low-precision factor.
    pub fn inner(&self) -> &F {
        &self.factor
    }

    /// Unwrap the low-precision factor.
    pub fn into_inner(self) -> F {
        self.factor
    }
}

/// Run `solve` on `b` demoted to `H::Low` and promote its result.
fn through_low<H: Demote>(
    b: &[H],
    solve: impl FnOnce(&[H::Low]) -> Result<Vec<H::Low>, RslabError>,
) -> Result<Vec<H>, RslabError> {
    let low: Vec<H::Low> = b.iter().map(|&v| v.demote()).collect();
    Ok(solve(&low)?.into_iter().map(H::promote).collect())
}

impl<H: Demote, F: Factorization<H::Low>> Preconditioner<H> for MixedPrecision<F> {
    fn apply(&self, r: &[H], z: &mut [H]) -> Result<(), RslabError> {
        z.copy_from_slice(&through_low(r, |r| self.factor.solve(r))?);
        Ok(())
    }
    fn apply_block(&self, r: &[H], z: &mut [H], s: usize, n: usize) -> Result<(), RslabError> {
        z[..n * s].copy_from_slice(&through_low(&r[..n * s], |r| self.factor.solve_many(r, s))?);
        Ok(())
    }
    fn solve_threads(&self) -> Threads {
        self.factor.solve_threads()
    }
}

impl<H: Demote, F: Factorization<H::Low>> Factorization<H> for MixedPrecision<F> {
    fn n(&self) -> usize {
        self.factor.n()
    }
    /// One solve with the low-precision factor: accurate to its precision.
    fn solve(&self, b: &[H]) -> Result<Vec<H>, RslabError> {
        through_low(b, |b| self.factor.solve(b))
    }
    fn solve_many(&self, b: &[H], nrhs: usize) -> Result<Vec<H>, RslabError> {
        through_low(b, |b| self.factor.solve_many(b, nrhs))
    }
    fn solve_transpose(&self, b: &[H]) -> Result<Vec<H>, RslabError> {
        through_low(b, |b| self.factor.solve_transpose(b))
    }
    /// Mixed-precision iterative refinement: residuals against `a` in `H`,
    /// corrections from the low-precision factor.
    fn solve_refined(
        &self,
        a: &dyn crate::refine::RefineOperator<H>,
        b: &[H],
        policy: &crate::refine::RefinePolicy,
    ) -> Result<(Vec<H>, crate::refine::RefineOutcome), RslabError> {
        let mut x = Factorization::solve(self, b)?;
        let outcome = crate::refine::refine_in_place(a, b, &mut x, policy, |r| {
            Factorization::solve(self, r)
        })?;
        Ok((x, outcome))
    }
    fn factor_nnz(&self) -> usize {
        self.factor.factor_nnz()
    }
    fn n_perturbed(&self) -> usize {
        self.factor.n_perturbed()
    }
    fn diagnostics(&self) -> crate::diagnostics::Diagnostics {
        self.factor.diagnostics()
    }
}

/// A `Complex<f32>` LDL^T factor preconditioning a `Complex<f64>` iteration.
pub type LowPrecisionPreconditioner = MixedPrecision<LdltSolver<Complex<f32>>>;

/// A `Complex<f32>` LU factor preconditioning a `Complex<f64>` iteration.
pub type LowPrecisionLu = MixedPrecision<crate::numeric::lu::LuSolver<Complex<f32>>>;

impl LowPrecisionPreconditioner {
    /// Demote `A` to `Complex<f32>` and factor it (static pivoting honoured
    /// via `opts`, e.g. `ZeroPivotAction::PerturbToEps`).
    pub fn factor(a: &CscMatrix<Complex<f64>>, opts: &SolverSettings) -> Result<Self, RslabError> {
        Ok(Self::new(LdltSolver::factor(&a.demoted(), opts)?))
    }

    /// Stored factor fill (nnz of `L`), in single-precision entries.
    pub fn factor_nnz(&self) -> usize {
        self.factor.factor_nnz()
    }

    /// Number of statically perturbed pivots (see [`LdltSolver::n_perturbed`]).
    pub fn n_perturbed(&self) -> usize {
        self.factor.n_perturbed()
    }
}

impl LowPrecisionLu {
    /// Demote `A` to `Complex<f32>` and LU-factor it (options honoured:
    /// static pivoting and drop tolerance for a preconditioner).
    pub fn factor(a: &GeneralCsc<Complex<f64>>, opts: &SolverSettings) -> Result<Self, RslabError> {
        Ok(Self::new(crate::numeric::lu::LuSolver::factor(
            &a.demoted(),
            opts,
        )?))
    }

    /// Stored fill `nnz(L)+nnz(U)`, in single-precision entries.
    pub fn factor_nnz(&self) -> usize {
        self.factor.factor_nnz()
    }

    /// Number of statically perturbed pivots.
    pub fn n_perturbed(&self) -> usize {
        self.factor.n_perturbed()
    }
}

/// A direct solver: [`LdltSolver`], [`LuSolver`](crate::LuSolver) and
/// [`KluSolver`](crate::KluSolver). Each is also a [`Preconditioner`], so a
/// solver loop can hold `&dyn Factorization` and swap the symmetric, general
/// and circuit paths, exact or incomplete factors, freely.
pub trait Factorization<T: Scalar>: Preconditioner<T> {
    /// The matrix dimension.
    fn n(&self) -> usize;
    /// Solve `A x = b`.
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError>;
    /// Solve `A X = B` for a column-major `n x nrhs` block.
    fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError>;
    /// Solve `A^T x = b` (the plain transpose).
    fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>, RslabError>;
    /// Solve `A x = b` with iterative refinement against `a`.
    fn solve_refined(
        &self,
        a: &dyn crate::refine::RefineOperator<T>,
        b: &[T],
        policy: &crate::refine::RefinePolicy,
    ) -> Result<(Vec<T>, crate::refine::RefineOutcome), RslabError>;
    /// Stored factor entries, the memory metric.
    fn factor_nnz(&self) -> usize;
    /// Pivots lifted by the static regularization (0 for an exact factor).
    fn n_perturbed(&self) -> usize;
    /// Everything the factorization reports about itself.
    fn diagnostics(&self) -> crate::diagnostics::Diagnostics;
}
