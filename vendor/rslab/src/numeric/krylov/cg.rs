//! COCG and COCR for complex-symmetric systems (CG and CR for real ones).

use super::result::stop_reason;
use super::util::*;
use super::*;

use crate::error::RslabError;
use crate::scalar::Scalar;

/// Preconditioned COCG for a complex-symmetric `A = A^T` stored as a lower-
/// triangle [`CscMatrix`](crate::CscMatrix) (multiplied via [`CscMatrix::symv`](crate::CscMatrix::symv)).
///
/// Solves `A x = b` to relative residual `tol` (or `max_iter` iterations).
/// `precond` supplies `M^-1`; pass [`NoPreconditioner`] for the unpreconditioned
/// baseline. Starts from `x_0 = 0`.
///
/// Breakdown (a zero bilinear denominator `p^TAp` or `r^Tz`, possible for an
/// indefinite complex-symmetric operator) stops the iteration and returns the
/// best iterate with `converged = false`.
pub fn cocg<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    settings: &KrylovSettings,
) -> Result<KrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    let (tol, max_iter) = (settings.tol, settings.max_iter);
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }

    let mut x = vec![T::zero(); n];
    // r_0 = b - A x_0 = b (x_0 = 0).
    let mut r = b.to_vec();
    let bnorm = norm2(b);
    if bnorm == 0.0 {
        return Ok(KrylovResult {
            x,
            iters: 0,
            converged: true,
            final_res: 0.0,
            stop: StopReason::Converged,
        });
    }

    let mut z = vec![T::zero(); n];
    precond.apply(&r, &mut z)?;
    let mut p = z.clone();
    let mut rho = dotu(&r, &z); // r^Tz, unconjugated
    let mut q = vec![T::zero(); n];

    let mut final_res = norm2(&r) / bnorm;
    let mut converged = false;
    let mut iters = 0;
    while iters < max_iter {
        op.apply(&p, &mut q); // q = A p
        let pq = dotu(&p, &q);
        if pq == T::zero() {
            break; // breakdown
        }
        let alpha = rho * pq.recip();
        for i in 0..n {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * q[i];
        }
        iters += 1;
        final_res = norm2(&r) / bnorm;
        if final_res <= tol {
            converged = true;
            break;
        }
        precond.apply(&r, &mut z)?;
        let rho_new = dotu(&r, &z);
        if rho == T::zero() {
            break; // breakdown
        }
        let beta = rho_new * rho.recip();
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rho = rho_new;
    }

    // Not converged with iterations left => the loop broke on a zero `p^TAp`/`r^Tz`
    // denominator (breakdown); otherwise the budget was exhausted.
    let stop = stop_reason(converged, iters, max_iter);
    Ok(KrylovResult {
        x,
        iters,
        converged,
        final_res,
        stop,
    })
}

/// Preconditioned COCR (Conjugate Orthogonal Conjugate Residual, Sogabe &
/// Zhang 2007) for complex-symmetric `A = A^T`. The CR-family analogue of
/// [`cocg`]: it minimises a residual-like quantity and is typically **smoother
/// and more robust on strongly indefinite** operators (high-frequency 3D
/// Helmholtz) where COCG's residual can oscillate or break down.
///
/// Same interface and conventions as [`cocg`]. Costs one matrix-vector product
/// and one preconditioner apply per iteration. Reduces to preconditioned CR
/// for `T = f64`.
pub fn cocr<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    settings: &KrylovSettings,
) -> Result<KrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    let (tol, max_iter) = (settings.tol, settings.max_iter);
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }

    let mut x = vec![T::zero(); n];
    let mut r = b.to_vec(); // r_0 = b (x_0 = 0)
    let bnorm = norm2(b);
    if bnorm == 0.0 {
        return Ok(KrylovResult {
            x,
            iters: 0,
            converged: true,
            final_res: 0.0,
            stop: StopReason::Converged,
        });
    }

    let mut z = vec![T::zero(); n]; // z = M^-1 r
    precond.apply(&r, &mut z)?;
    let mut p = z.clone();
    let mut ap = vec![T::zero(); n];
    op.apply(&p, &mut ap); // A p
    let mut az = ap.clone(); // A z (= A p at init since p = z)
    let mut gamma = dotu(&z, &az); // z^T A z
    let mut w = vec![T::zero(); n]; // M^-1 A p
    let mut aw = vec![T::zero(); n]; // A w

    let mut final_res = norm2(&r) / bnorm;
    let mut converged = false;
    let mut iters = 0;
    while iters < max_iter {
        precond.apply(&ap, &mut w)?; // w = M^-1 A p
        op.apply(&w, &mut aw); // A w
        let denom = dotu(&ap, &w); // (A p)^T M^-1 (A p)
        if denom == T::zero() {
            break;
        }
        let alpha = gamma * denom.recip();
        for i in 0..n {
            x[i] = x[i] + alpha * p[i];
            r[i] = r[i] - alpha * ap[i];
            z[i] = z[i] - alpha * w[i];
            az[i] = az[i] - alpha * aw[i];
        }
        iters += 1;
        final_res = norm2(&r) / bnorm;
        if final_res <= tol {
            converged = true;
            break;
        }
        let gamma_new = dotu(&z, &az);
        if gamma == T::zero() {
            break;
        }
        let beta = gamma_new * gamma.recip();
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
            ap[i] = az[i] + beta * ap[i];
        }
        gamma = gamma_new;
    }

    // Same three-way classification as COCG (see [`stop_reason`]).
    let stop = stop_reason(converged, iters, max_iter);
    Ok(KrylovResult {
        x,
        iters,
        converged,
        final_res,
        stop,
    })
}
