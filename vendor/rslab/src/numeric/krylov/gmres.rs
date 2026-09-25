//! Restarted GMRES with modified Gram-Schmidt and a conditional second pass.

use super::util::*;
use super::*;

use crate::error::RslabError;
use crate::scalar::Scalar;

/// **Flexible** right-preconditioned restarted **GMRES(`restart`)** (FGMRES,
/// Saad 1993) for a general (unsymmetric) operator - the natural Krylov method
/// for unsymmetric MoM/FEM systems where COCG/COCR do not apply. `op` may be
/// matrix-free; `precond` supplies `M^-1` (e.g. an
/// [`LuSolver`](crate::LuSolver) near-field factor).
/// Solves `A x = b` from the optional initial guess `x0` (default `x_0 = 0`).
///
/// **Warm start:** pass `x0 = Some(prev)` to seed the iteration from a
/// previous, related solution - on a sequence of slowly varying systems this
/// often cuts the iteration count substantially. Convergence is still measured
/// relative to ||b||.
///
/// **Flexible variant:** the preconditioned Arnoldi vectors
/// `z_j = M^-1 v_j` (already formed to build `w = A z_j`) are *kept* as a second
/// basis `Z = [z_0 ... z_{m-1}]`, and the restart update is `x += Z y` directly.
/// This (a) removes the one extra `M^-1` solve per cycle that plain right-
/// preconditioned GMRES spends rebuilding `M^-1(V y)`, and (b) makes the method
/// flexible: `M` may **vary** between steps (an inner Krylov solve, or a
/// preconditioner strengthened across iterations). Cost: one extra `n*(m+1)`
/// basis of storage.
pub fn gmres<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
    restart: usize,
    x0: Option<&[T]>,
) -> Result<KrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // DGKS reorthogonalization threshold: redo the projection only when a single
    // MGS pass cancelled more than this fraction of the vector's length.
    const REORTH_ETA: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let m = restart.max(1);
    let bnorm = norm2(b);
    // Warm start: seed `x` from the caller's initial guess `x0`; the
    // per-cycle true residual `r = b - A x` then measures progress from that
    // guess. Convergence is still relative to ||b||. Absent `x0`, `x_0 = 0`.
    let mut x = match x0 {
        Some(g) => {
            if g.len() != n {
                return Err(RslabError::DimensionMismatch {
                    expected: n,
                    got: g.len(),
                });
            }
            g.to_vec()
        }
        None => vec![T::zero(); n],
    };
    if bnorm == 0.0 {
        x.iter_mut().for_each(|v| *v = T::zero());
        return Ok(KrylovResult {
            x,
            iters: 0,
            converged: true,
            final_res: 0.0,
            stop: StopReason::Converged,
        });
    }

    let mut total = 0usize;
    let mut w = vec![T::zero(); n];
    let mut ax = vec![T::zero(); n];
    // Arnoldi basis as one **flat** `n x (m+1)` buffer (column `i` is
    // `v[i*n .. (i+1)*n]`) - contiguous, no per-iteration vector allocation, and
    // cache-friendly for the Gram-Schmidt sweeps.
    let mut v = vec![T::zero(); n * (m + 1)];
    // FGMRES preconditioned basis `Z = [z_0 ... z_{m-1}]`, `z_j = M^-1 v_j` (issue
    // #7): stored as it is computed so the restart update is `x += Z y` with no
    // second preconditioner solve, and so a *variable* `M` is honoured exactly.
    let mut zb = vec![T::zero(); n * m];
    // Per-restart Krylov scalars hoisted out of the outer loop and cleared/reused
    // each cycle: the Hessenberg `h` as one **flat** `(m+1)xm` buffer
    // (row `i`, col `j` at `h[i*m + j]` - cache-friendlier than a `Vec<Vec<T>>`),
    // the Givens `cs`/`sn`, the LS RHS `g`, and the back-substitution `y`. A
    // many-restart solve (the ill-conditioned regime) then does no per-cycle heap
    // churn; the reused buffers are zeroed each cycle, so the numerics are
    // bit-identical to the old fresh-allocation-per-cycle path.
    let mut h = vec![T::zero(); (m + 1) * m];
    let mut cs = vec![T::zero(); m];
    let mut sn = vec![T::zero(); m];
    let mut g = vec![T::zero(); m + 1];
    let mut y = vec![T::zero(); m];

    // Outer restart loop. Each pass first measures the TRUE residual ||b-Ax|| of the
    // current iterate (the only reliable stop test on ill-conditioned MoM near-
    // field operators, where the Hessenberg LS estimate can dip below `tol` while
    // the true residual is orders larger) and records it as `final_res`. On
    // convergence *or* exhausted iterations we break with that value already in
    // hand - no separate post-loop matvec to report the residual.
    // Definitely assigned before every `break` (the only exits from the loop).
    let mut final_res;
    loop {
        op.apply(&x, &mut ax);
        let r: Vec<T> = (0..n).map(|i| b[i] - ax[i]).collect();
        let beta = norm2(&r);
        final_res = beta / bnorm;
        if final_res <= tol || total >= max_iter {
            break;
        }
        // Column 0 of the basis = r / ||r||. Reset the reused Hessenberg / Givens /
        // LS state to the fresh-zero semantics of the old per-cycle allocation.
        let inv_beta = T::from_real(1.0 / beta);
        for k in 0..n {
            v[k] = r[k] * inv_beta;
        }
        for e in h.iter_mut() {
            *e = T::zero();
        }
        for e in cs.iter_mut() {
            *e = T::zero();
        }
        for e in sn.iter_mut() {
            *e = T::zero();
        }
        for e in g.iter_mut() {
            *e = T::zero();
        }
        g[0] = T::from_real(beta);
        let mut jdim = 0usize;
        for j in 0..m {
            if total >= max_iter {
                break;
            }
            // Flexible right preconditioning: z_j = M^-1 v[j] (stored into the Z
            // basis for the restart update), then w = A z_j.
            precond.apply(&v[j * n..j * n + n], &mut zb[j * n..j * n + n])?;
            op.apply(&zb[j * n..j * n + n], &mut w);
            // Modified Gram-Schmidt against the existing basis, with **conditional**
            // reorthogonalization (DGKS): the second pass - essential on
            // ill-conditioned operators (MoM near-field) where a single MGS pass
            // loses orthogonality and the Hessenberg residual estimate drifts from
            // the true residual - runs only when the projection cancelled most of
            // the vector (||w|| dropped below `eta*||w_0||`, eta = 1/sqrt2). Well-conditioned
            // cycles skip it, halving the orthogonalization cost.
            let wnorm0 = norm2(&w);
            for i in 0..=j {
                let hij = dotc(&v[i * n..i * n + n], &w);
                h[i * m + j] = hij;
                for k in 0..n {
                    w[k] = w[k] - hij * v[i * n + k];
                }
            }
            let mut hn = norm2(&w);
            if hn < REORTH_ETA * wnorm0 {
                for i in 0..=j {
                    let s = dotc(&v[i * n..i * n + n], &w);
                    h[i * m + j] = h[i * m + j] + s;
                    for k in 0..n {
                        w[k] = w[k] - s * v[i * n + k];
                    }
                }
                hn = norm2(&w);
            }
            h[(j + 1) * m + j] = T::from_real(hn);
            if hn > 0.0 {
                let inv = T::from_real(1.0 / hn);
                for k in 0..n {
                    v[(j + 1) * n + k] = w[k] * inv;
                }
            } else {
                // Invariant subspace: zero the (reused) basis column so a later
                // sweep never reads stale data from a previous restart cycle.
                for k in 0..n {
                    v[(j + 1) * n + k] = T::zero();
                }
            }
            // Apply previous rotations to the new Hessenberg column.
            for i in 0..j {
                let temp = cs[i].conj() * h[i * m + j] + sn[i].conj() * h[(i + 1) * m + j];
                h[(i + 1) * m + j] = -sn[i] * h[i * m + j] + cs[i] * h[(i + 1) * m + j];
                h[i * m + j] = temp;
            }
            // New rotation zeroing h[j+1][j]; apply to H and the LS RHS g.
            let (c, s) = givens(h[j * m + j], h[(j + 1) * m + j]);
            cs[j] = c;
            sn[j] = s;
            h[j * m + j] = c.conj() * h[j * m + j] + s.conj() * h[(j + 1) * m + j];
            h[(j + 1) * m + j] = T::zero();
            let g_next = -s * g[j];
            g[j] = c.conj() * g[j];
            g[j + 1] = g_next;
            total += 1;
            jdim = j + 1;
            // Early-exit the inner sweep on the LS estimate, but do **not** treat it
            // as convergence: the outer loop's top-of-cycle TRUE-residual check is
            // authoritative (the estimate can drift on ill-conditioned operators).
            if g[j + 1].magnitude() / bnorm <= tol {
                break;
            }
        }
        // Back-substitute the upper-triangular H for y on the well-conditioned
        // leading block (breakdown guard), then (FGMRES) x += Z*y directly from the
        // stored preconditioned basis - no second `M^-1` solve.
        let jd = well_conditioned_dim_flat(&h, m, jdim);
        for i in (0..jd).rev() {
            let mut s = g[i];
            for k in (i + 1)..jd {
                s = s - h[i * m + k] * y[k];
            }
            y[i] = s * h[i * m + i].recip();
        }
        for i in 0..jd {
            let yi = y[i];
            for k in 0..n {
                x[k] = x[k] + zb[i * n + k] * yi;
            }
        }
    }

    Ok(KrylovResult {
        x,
        iters: total,
        converged: final_res <= tol,
        final_res,
        // GMRES-family: converged (residual met, incl. happy breakdown) or the
        // iteration budget ran out. No non-converged breakdown state.
        stop: if final_res <= tol {
            StopReason::Converged
        } else {
            StopReason::MaxIter
        },
    })
}
