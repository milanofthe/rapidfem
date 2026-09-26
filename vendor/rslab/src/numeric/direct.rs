//! What the three direct solvers share: the solve entry points with their
//! accounting and iterative refinement, and the [`Preconditioner`] and
//! [`Factorization`] impls. A solver provides [`SolveCore`], the raw
//! triangular solves; [`direct_solver!`] builds the rest from it.
//!
//! Right-hand-side blocks are column-major `n x nrhs` (`b[c * n + i]` is
//! row `i` of column `c`), like the Krylov panels.

use crate::diagnostics::SolveCounter;
use crate::error::RslabError;
use crate::refine::{refine_in_place, RefineOperator, RefineOutcome, RefinePolicy};
use crate::scalar::Scalar;

/// The part of a direct solver the shared entry points build on.
pub(crate) trait SolveCore<T: Scalar> {
    /// The path's name in log records.
    const NAME: &'static str;
    fn dim(&self) -> usize;
    /// `A X = B` (or `A^T X = B`) for a column-major block `b` whose shape
    /// the caller has checked, without accounting.
    fn solve_raw(&self, b: &[T], nrhs: usize, transpose: bool) -> Result<Vec<T>, RslabError>;
    fn counter(&self) -> &SolveCounter;
}

fn record<T: Scalar, S: SolveCore<T>>(s: &S, rhs: usize, t: crate::clock::Instant, steps: usize) {
    let ms = t.elapsed().as_secs_f64() * 1e3;
    s.counter().record(rhs, ms, steps);
    if crate::logging::enabled(crate::logging::LogLevel::Debug) {
        crate::logging::debug(&format!(
            "{} solve: n={} rhs={rhs} refine_steps={steps} {ms:.3} ms",
            S::NAME,
            s.dim()
        ));
    }
}

fn check(n: usize, len: usize, nrhs: usize) -> Result<(), RslabError> {
    if nrhs == 0 || len != n * nrhs {
        return Err(RslabError::DimensionMismatch {
            expected: n * nrhs.max(1),
            got: len,
        });
    }
    Ok(())
}

pub(crate) fn solve<T: Scalar, S: SolveCore<T>>(
    s: &S,
    b: &[T],
    nrhs: usize,
    transpose: bool,
) -> Result<Vec<T>, RslabError> {
    check(s.dim(), b.len(), nrhs)?;
    let t = crate::clock::Instant::now();
    let x = s.solve_raw(b, nrhs, transpose)?;
    record(s, nrhs, t, 0);
    Ok(x)
}

pub(crate) fn refine_into<T, S, A>(
    s: &S,
    a: &A,
    b: &[T],
    x: &mut [T],
    policy: &RefinePolicy,
) -> Result<RefineOutcome, RslabError>
where
    T: Scalar,
    S: SolveCore<T>,
    A: RefineOperator<T> + ?Sized,
{
    let n = s.dim();
    if a.dim() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: a.dim(),
        });
    }
    check(n, b.len(), 1)?;
    check(n, x.len(), 1)?;
    refine_in_place(a, b, x, policy, |r| s.solve_raw(r, 1, false))
}

pub(crate) fn solve_refined<T, S, A>(
    s: &S,
    a: &A,
    b: &[T],
    policy: &RefinePolicy,
) -> Result<(Vec<T>, RefineOutcome), RslabError>
where
    T: Scalar,
    S: SolveCore<T>,
    A: RefineOperator<T> + ?Sized,
{
    check(s.dim(), b.len(), 1)?;
    let t = crate::clock::Instant::now();
    let mut x = s.solve_raw(b, 1, false)?;
    let outcome = refine_into(s, a, b, &mut x, policy)?;
    record(s, 1, t, outcome.steps);
    Ok((x, outcome))
}

pub(crate) fn refine_into_recorded<T, S, A>(
    s: &S,
    a: &A,
    b: &[T],
    x: &mut [T],
    policy: &RefinePolicy,
) -> Result<RefineOutcome, RslabError>
where
    T: Scalar,
    S: SolveCore<T>,
    A: RefineOperator<T> + ?Sized,
{
    let t = crate::clock::Instant::now();
    let outcome = refine_into(s, a, b, x, policy)?;
    record(s, 0, t, outcome.steps);
    Ok(outcome)
}

/// The shared entry points of a direct solver `$solver<T>`: the solves, the
/// refinement, and the `Preconditioner` / `Factorization` impls.
macro_rules! direct_solver {
    ($solver:ident) => {
        impl<T: $crate::scalar::Scalar> $solver<T> {
            /// Solve `A x = b`.
            pub fn solve(&self, b: &[T]) -> Result<Vec<T>, $crate::RslabError> {
                $crate::numeric::direct::solve(self, b, 1, false)
            }

            /// Solve `A X = B` for a column-major `n x nrhs` block (`b[c * n + i]`
            /// is row `i` of column `c`) in one pass over the factor, which
            /// applies each stored value to every column.
            pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, $crate::RslabError> {
                $crate::numeric::direct::solve(self, b, nrhs, false)
            }

            /// Solve `A^T x = b` on the same factors (the plain transpose; for
            /// the adjoint `A^H x = b` conjugate `b` before and `x` after).
            pub fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>, $crate::RslabError> {
                $crate::numeric::direct::solve(self, b, 1, true)
            }

            /// Solve `A x = b` with iterative refinement against `a`, the matrix
            /// this factor was computed from (or the exact one it
            /// approximates), under `policy`, reporting the backward error
            /// reached.
            pub fn solve_refined<A: $crate::refine::RefineOperator<T> + ?Sized>(
                &self,
                a: &A,
                b: &[T],
                policy: &$crate::RefinePolicy,
            ) -> Result<(Vec<T>, $crate::RefineOutcome), $crate::RslabError> {
                $crate::numeric::direct::solve_refined(self, a, b, policy)
            }

            /// Refine an existing iterate `x` of `A x = b` in place.
            pub fn refine_into<A: $crate::refine::RefineOperator<T> + ?Sized>(
                &self,
                a: &A,
                b: &[T],
                x: &mut [T],
                policy: &$crate::RefinePolicy,
            ) -> Result<$crate::RefineOutcome, $crate::RslabError> {
                $crate::numeric::direct::refine_into_recorded(self, a, b, x, policy)
            }
        }

        impl<T: $crate::scalar::Scalar> $crate::Preconditioner<T> for $solver<T> {
            fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), $crate::RslabError> {
                z.copy_from_slice(&$solver::solve(self, r)?);
                Ok(())
            }
            fn apply_block(
                &self,
                r: &[T],
                z: &mut [T],
                s: usize,
                n: usize,
            ) -> Result<(), $crate::RslabError> {
                z[..n * s].copy_from_slice(&$solver::solve_many(self, &r[..n * s], s)?);
                Ok(())
            }
            fn solve_threads(&self) -> $crate::Threads {
                self.solve_threads
            }
        }

        impl<T: $crate::scalar::Scalar> $crate::Factorization<T> for $solver<T> {
            fn n(&self) -> usize {
                $solver::n(self)
            }
            fn solve(&self, b: &[T]) -> Result<Vec<T>, $crate::RslabError> {
                $solver::solve(self, b)
            }
            fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, $crate::RslabError> {
                $solver::solve_many(self, b, nrhs)
            }
            fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>, $crate::RslabError> {
                $solver::solve_transpose(self, b)
            }
            fn solve_refined(
                &self,
                a: &dyn $crate::refine::RefineOperator<T>,
                b: &[T],
                policy: &$crate::RefinePolicy,
            ) -> Result<(Vec<T>, $crate::RefineOutcome), $crate::RslabError> {
                $solver::solve_refined(self, a, b, policy)
            }
            fn factor_nnz(&self) -> usize {
                $solver::factor_nnz(self)
            }
            fn n_perturbed(&self) -> usize {
                $solver::n_perturbed(self)
            }
            fn diagnostics(&self) -> $crate::Diagnostics {
                $solver::diagnostics(self)
            }
        }
    };
}
pub(crate) use direct_solver;
