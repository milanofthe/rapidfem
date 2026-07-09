//! Krylov iteration for complex-symmetric systems, preconditioned by an RLA
//! factorization.
//!
//! The target use is **3D EM FEM / MOM**: large complex-symmetric `A = Aᵀ`
//! (PARDISO `mtype 6`) systems solved iteratively, with a robust, memory-light
//! RLA factorization (static-pivoted, optionally `f32` / incomplete) as the
//! preconditioner. The iterative method of choice for `A = Aᵀ` is **COCG**
//! (Conjugate Orthogonal Conjugate Gradient, van der Vorst & Melissen 1990):
//! structurally CG, but every inner product is the *unconjugated* bilinear
//! form `xᵀy = Σ xᵢyᵢ` - the correct geometry for a complex-symmetric (not
//! Hermitian) operator. For `T = f64` it reduces exactly to preconditioned CG.
//!
//! The [`Preconditioner`] trait decouples the iteration from the factorization
//! precision: an `f64` factor applies directly; an `f32` factor (memory-halved)
//! down-/up-casts inside `apply`, while the iteration itself always runs in the
//! working precision `T`.
//!
//! ## Orthogonalization (GMRES paths)
//!
//! The two GMRES paths orthogonalize the Arnoldi basis by **different**, both
//! backward-stable, schemes (issue #8):
//!
//! - **Single-RHS [`gmres`]:** *modified* Gram-Schmidt (each projection updates
//!   `w` before the next is taken) with a conditional DGKS second pass, triggered
//!   when one MGS sweep cancels more than `1/√2` of `‖w‖`.
//! - **Block [`gmres_block`]:** *classical* Gram-Schmidt with a conditional second
//!   pass = **CGS2**, batched over the whole panel for BLAS-3 arithmetic
//!   intensity; the DGKS second pass is decided and applied **per column**.
//!
//! Consequently a **block solve with `s = 1` is *not* bit-identical to the single-
//! RHS [`gmres`]**: MGS and CGS accumulate the projections in a different order, so
//! the two can differ by a rounding ULP per step and hence by up to ±1 iteration
//! at a residual that straddles `tol`. Both converge to the same solution within
//! the requested tolerance. This is a deliberate design point (CGS2's panel-wide
//! reductions are what make the multi-RHS path fast and thread-count deterministic),
//! not a bug; see the `gmres_block_single_rhs_matches_scalar_gmres` test.

use crate::error::RslabError;
use crate::numeric::multifrontal_ldlt::{SolverSettings, Threads};
use crate::numeric::sparse_solver::LdltSolver;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use num_complex::Complex;
use rayon::prelude::*;

/// A linear operator `A`: applies `y = A x`. The Krylov solvers depend only on
/// this trait, so the operator may be an explicit sparse matrix
/// ([`CscMatrix`] symmetric / [`GeneralCsc`] general) **or matrix-free** - e.g.
/// a fast multipole (FMM/MLFMA) MoM operator the caller implements. RLA then
/// only factors the sparse near-field as the [`Preconditioner`].
pub trait LinearOperator<T: Scalar> {
    /// The system dimension.
    fn n(&self) -> usize;
    /// Write `y ← A x`. `x` and `y` have length `n`.
    fn apply(&self, x: &[T], y: &mut [T]);
    /// Block apply: `Y[:,c] ← A X[:,c]` for `c in 0..s`, with `X`,`Y` **column-
    /// major** `n×s` (RHS `c` is the contiguous slice `[c·n, (c+1)·n)`). The
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
    /// and scattered to all `s` columns (`y[:,c] += v·x[j,c]`, and symmetrically
    /// `y[j,c] += v·x[i,c]` off the diagonal) - the BLAS-3 reuse a multi-RHS
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
    /// all `s` columns (`y[i,c] += v·x[j,c]`).
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

/// A preconditioner `M ≈ A`: applies `z = M⁻¹ r`. Implemented by a factored
/// [`LdltSolver`](crate::numeric::sparse_solver::LdltSolver)
/// and by [`NoPreconditioner`] (the unpreconditioned baseline).
pub trait Preconditioner<T: Scalar> {
    /// Write `z ← M⁻¹ r`. `r` and `z` have length `n`.
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError>;
    /// Block apply: `Z[:,c] ← M⁻¹ R[:,c]` for `c in 0..s`, with `R`,`Z` **column-
    /// major** `n×s`. The default loops [`apply`](Self::apply); a factored solver
    /// overrides it with a block triangular solve (`solve_many`) that loads each
    /// `L`/`D`/`U` value once for all `s` columns.
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        for c in 0..s {
            self.apply(&r[c * n..c * n + n], &mut z[c * n..c * n + n])?;
        }
        Ok(())
    }
    /// Thread policy the **solve phase** should honour (issue #9). A factored
    /// preconditioner returns the resolved [`Threads`] budget it was built with, so
    /// [`gmres_block`]'s parallel orthogonalization runs in a pool of the **same**
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

/// A factored RLA solver is a preconditioner: `M⁻¹ r` is one forward/back
/// substitution against the stored `LDLᵀ` factor.
impl<T: Scalar> Preconditioner<T> for LdltSolver<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    /// Block apply via [`solve_many`](LdltSolver::solve_many): one block
    /// triangular solve loads each `L`/`D` value once for all `s` columns. The
    /// Krylov block is column-major; `solve_many` is row-major, so it is
    /// transposed in/out (`O(n·s)`, cheap against the solve).
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        let mut rowmaj = vec![T::zero(); n * s];
        for c in 0..s {
            for i in 0..n {
                rowmaj[i * s + c] = r[c * n + i];
            }
        }
        let x = self.solve_many(&rowmaj, s)?;
        for c in 0..s {
            for i in 0..n {
                z[c * n + i] = x[i * s + c];
            }
        }
        Ok(())
    }
}

/// A memory-halved preconditioner: factor `A` (supplied in `Complex<f64>`) in
/// `Complex<f32>` and apply it inside an `f64` Krylov iteration. The stored
/// factor occupies **half the bytes** and its triangular solves run in single
/// precision (gemm `c32` during the factor); because the outer COCG/COCR still
/// iterates in `f64`, the *solution* keeps full `f64` accuracy. This is the
/// standard mixed-precision setup for large 3D EM FEM / MOM preconditioning.
pub struct LowPrecisionPreconditioner {
    inner: LdltSolver<Complex<f32>>,
}

impl LowPrecisionPreconditioner {
    /// Down-cast `A` to `Complex<f32>` and factor it (static-pivoting honoured
    /// via `opts`, e.g. `ZeroPivotAction::PerturbToEps`).
    pub fn factor(a: &CscMatrix<Complex<f64>>, opts: &SolverSettings) -> Result<Self, RslabError> {
        let a32 = CscMatrix::<Complex<f32>> {
            n: a.n,
            col_ptr: a.col_ptr.clone(),
            row_idx: a.row_idx.clone(),
            values: a
                .values
                .iter()
                .map(|v| Complex::new(v.re as f32, v.im as f32))
                .collect(),
        };
        Ok(Self {
            inner: LdltSolver::factor_with(&a32, opts)?,
        })
    }

    /// Stored factor fill (nnz of `L`); each entry is a single-precision
    /// `Complex<f32>` (8 bytes vs 16 for `Complex<f64>`).
    pub fn factor_nnz(&self) -> usize {
        self.inner.factor_nnz()
    }

    /// Number of statically perturbed pivots (see [`LdltSolver::n_perturbed`]).
    pub fn n_perturbed(&self) -> usize {
        self.inner.n_perturbed()
    }
}

impl Preconditioner<Complex<f64>> for LowPrecisionPreconditioner {
    fn apply(&self, r: &[Complex<f64>], z: &mut [Complex<f64>]) -> Result<(), RslabError> {
        let r32: Vec<Complex<f32>> = r
            .iter()
            .map(|v| Complex::new(v.re as f32, v.im as f32))
            .collect();
        let z32 = self.inner.solve(&r32)?;
        for (zi, v) in z.iter_mut().zip(z32) {
            *zi = Complex::new(v.re as f64, v.im as f64);
        }
        Ok(())
    }
}

/// Memory-halved **unsymmetric** preconditioner: factor the general matrix `A`
/// (given in `Complex<f64>`) in `Complex<f32>` LU and apply it inside an `f64`
/// GMRES iteration. The `Complex<f32>` factor uses half the bytes (and gemm
/// `c32`); the outer GMRES keeps full `f64` accuracy. The unsymmetric analogue
/// of [`LowPrecisionPreconditioner`], for MoM/FEM general systems.
pub struct LowPrecisionLu {
    inner: crate::numeric::multifrontal_lu::LuFactors<Complex<f32>>,
}

impl LowPrecisionLu {
    /// Down-cast `A` to `Complex<f32>` and LU-factor it (options honoured -
    /// static pivoting and/or incomplete dropping for a preconditioner).
    pub fn factor(a: &GeneralCsc<Complex<f64>>, opts: &SolverSettings) -> Result<Self, RslabError> {
        let a32 = GeneralCsc::<Complex<f32>> {
            n: a.n,
            col_ptr: a.col_ptr.clone(),
            row_idx: a.row_idx.clone(),
            values: a
                .values
                .iter()
                .map(|v| Complex::new(v.re as f32, v.im as f32))
                .collect(),
        };
        Ok(Self {
            inner: crate::numeric::multifrontal_lu::factor_general_lu(&a32, opts)?,
        })
    }

    /// Stored fill `nnz(L)+nnz(U)`, in single-precision entries.
    pub fn factor_nnz(&self) -> usize {
        crate::numeric::multifrontal_lu::LuFactors::factor_nnz(&self.inner)
    }

    /// Number of statically perturbed pivots.
    pub fn n_perturbed(&self) -> usize {
        self.inner.n_perturbed
    }
}

impl Preconditioner<Complex<f64>> for LowPrecisionLu {
    fn apply(&self, r: &[Complex<f64>], z: &mut [Complex<f64>]) -> Result<(), RslabError> {
        let r32: Vec<Complex<f32>> = r
            .iter()
            .map(|v| Complex::new(v.re as f32, v.im as f32))
            .collect();
        let z32 = crate::numeric::multifrontal_lu::solve_lu(&self.inner, &r32)?;
        for (zi, v) in z.iter_mut().zip(z32) {
            *zi = Complex::new(v.re as f64, v.im as f64);
        }
        Ok(())
    }
    fn solve_threads(&self) -> Threads {
        self.inner.solve_threads
    }
}

/// Unconjugated bilinear inner product `xᵀy = Σ xᵢyᵢ` (no complex conjugation -
/// the defining choice of the COCG geometry for `A = Aᵀ`).
#[inline]
fn dotu<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + xi * yi;
    }
    s
}

/// Euclidean norm `‖x‖₂ = √Σ|xᵢ|²` (genuine modulus - used only for the stopping
/// test, never inside the Krylov recurrences).
#[inline]
fn norm2<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.magnitude_sq()).sum::<f64>().sqrt()
}

/// Why a Krylov iteration stopped (audit finding L5). Additive diagnostic
/// alongside the boolean `converged`: `Converged` is exactly `converged == true`,
/// and the two failure states distinguish a hit iteration budget (`MaxIter`)
/// from an algebraic breakdown of the short-recurrence denominator (`Breakdown`).
///
/// Only the states the solvers genuinely distinguish are represented:
/// * COCG / COCR report all three - a zero bilinear denominator (`pᵀAp` or `rᵀz`
///   in COCG, the residual-norm form in COCR), reachable on an indefinite
///   complex-symmetric operator, is a real `Breakdown` before convergence.
/// * GMRES / FGMRES / GCRO-DR / block-GMRES never report `Breakdown`: a
///   rank-deficient Arnoldi step is a *happy* breakdown that yields the exact
///   solution (so it surfaces as `Converged`), and every other non-converged
///   exit is the iteration budget, i.e. `MaxIter`. There is no stagnation
///   detector, so `Stagnation` is intentionally absent rather than guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The relative residual reached `tol`.
    Converged,
    /// The iteration budget `max_iter` was exhausted first.
    MaxIter,
    /// A short-recurrence denominator went to zero before convergence
    /// (COCG / COCR only).
    Breakdown,
    /// The progress monitor requested an early stop (block GMRES only): the
    /// caller's per-cycle callback returned `false` — typically a stagnation
    /// detector cutting a solve that stopped contracting, instead of burning
    /// the remaining iteration budget.
    Stalled,
}

impl StopReason {
    /// Lowercase tag for the Python bindings and logging: `"converged"`,
    /// `"max_iter"`, `"breakdown"`, or `"stalled"`.
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::Converged => "converged",
            StopReason::MaxIter => "max_iter",
            StopReason::Breakdown => "breakdown",
            StopReason::Stalled => "stalled",
        }
    }
}

/// Classify a COCG/COCR exit: `Converged` when the residual test passed,
/// `Breakdown` when the loop broke early (iterations still left) on a zero
/// short-recurrence denominator, else `MaxIter` (budget exhausted).
#[inline]
fn stop_reason(converged: bool, iters: usize, max_iter: usize) -> StopReason {
    if converged {
        StopReason::Converged
    } else if iters < max_iter {
        StopReason::Breakdown
    } else {
        StopReason::MaxIter
    }
}

/// Outcome of a Krylov solve.
#[derive(Debug, Clone)]
pub struct KrylovResult<T> {
    /// The computed solution.
    pub x: Vec<T>,
    /// Number of iterations actually performed.
    pub iters: usize,
    /// `true` if `‖b − Ax‖ / ‖b‖ ≤ tol` was reached within `max_iter`.
    pub converged: bool,
    /// Final relative residual `‖b − Ax‖ / ‖b‖`.
    pub final_res: f64,
    /// Why the iteration stopped (`converged` is `stop == Converged`).
    pub stop: StopReason,
}

/// Preconditioned COCG for a complex-symmetric `A = Aᵀ` stored as a lower-
/// triangle [`CscMatrix`] (multiplied via [`CscMatrix::symv`]).
///
/// Solves `A x = b` to relative residual `tol` (or `max_iter` iterations).
/// `precond` supplies `M⁻¹`; pass [`NoPreconditioner`] for the unpreconditioned
/// baseline. Starts from `x₀ = 0`.
///
/// Breakdown (a zero bilinear denominator `pᵀAp` or `rᵀz`, possible for an
/// indefinite complex-symmetric operator) stops the iteration and returns the
/// best iterate with `converged = false`.
pub fn cocg<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
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

    let mut x = vec![T::zero(); n];
    // r₀ = b − A x₀ = b (x₀ = 0).
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
    let mut rho = dotu(&r, &z); // rᵀz, unconjugated
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

    // Not converged with iterations left ⇒ the loop broke on a zero `pᵀAp`/`rᵀz`
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
/// Zhang 2007) for complex-symmetric `A = Aᵀ`. The CR-family analogue of
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
    tol: f64,
    max_iter: usize,
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

    let mut x = vec![T::zero(); n];
    let mut r = b.to_vec(); // r₀ = b (x₀ = 0)
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

    let mut z = vec![T::zero(); n]; // z = M⁻¹ r
    precond.apply(&r, &mut z)?;
    let mut p = z.clone();
    let mut ap = vec![T::zero(); n];
    op.apply(&p, &mut ap); // A p
    let mut az = ap.clone(); // A z (= A p at init since p = z)
    let mut gamma = dotu(&z, &az); // zᵀ A z
    let mut w = vec![T::zero(); n]; // M⁻¹ A p
    let mut aw = vec![T::zero(); n]; // A w

    let mut final_res = norm2(&r) / bnorm;
    let mut converged = false;
    let mut iters = 0;
    while iters < max_iter {
        precond.apply(&ap, &mut w)?; // w = M⁻¹ A p
        op.apply(&w, &mut aw); // A w
        let denom = dotu(&ap, &w); // (A p)ᵀ M⁻¹ (A p)
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

/// Conjugated (Hermitian) inner product `⟨x, y⟩ = Σ conj(xᵢ)·yᵢ` - the geometry
/// GMRES orthogonalises in (distinct from COCG's unconjugated form).
#[inline]
fn dotc<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + xi.conj() * yi;
    }
    s
}

/// Complex Givens rotation `(c, s)` that zeroes `g` against `f`: with the
/// rotation `[[conj(c), conj(s)], [-s, c]]`, `conj(c)·f + conj(s)·g = r` (real)
/// and `-s·f + c·g = 0`, `|c|²+|s|² = 1`.
#[inline]
fn givens<T: Scalar>(f: T, g: T) -> (T, T) {
    if g == T::zero() {
        return (T::one(), T::zero());
    }
    if f == T::zero() {
        return (T::zero(), T::one());
    }
    let r = (f.magnitude_sq() + g.magnitude_sq()).sqrt();
    let inv = T::from_real(1.0 / r);
    (f * inv, g * inv)
}

/// Largest leading dimension `d ≤ jdim` whose Hessenberg diagonals are all above a
/// relative breakdown threshold `eps · max_i |h[i][i]|`. The upper-triangular
/// solve for `y` back-substitutes with `1/h[i][i]`; after Givens the diagonal is
/// `√(|f|²+|g|²)` and normally nonzero, but under exact stagnation or a rank-
/// deficient Hessenberg (hard / indefinite / singular operators) some `h[i][i]`
/// can be `0`, so an unguarded `recip()` would emit `Inf`/`NaN` into `x` and the
/// residual. Truncating the solve to this well-conditioned prefix instead yields a
/// **deterministic breakdown**: the leading Krylov block is solved, the degenerate
/// tail is dropped, and the outer true-residual check reports non-convergence. In
/// the well-conditioned case every diagonal clears the threshold and `jdim` is
/// returned unchanged, so the normal path is unaffected.
#[inline]
#[allow(clippy::needless_range_loop)]
fn well_conditioned_dim<T: Scalar>(h: &[Vec<T>], jdim: usize) -> usize {
    if jdim == 0 {
        return 0;
    }
    let mut hmax = 0.0f64;
    for i in 0..jdim {
        hmax = hmax.max(h[i][i].magnitude());
    }
    let thresh = f64::EPSILON * hmax;
    for i in 0..jdim {
        if h[i][i].magnitude() <= thresh {
            return i;
        }
    }
    jdim
}

/// [`well_conditioned_dim`] for a **flat** row-major Hessenberg buffer (issue #10):
/// the single-RHS [`gmres`] stores `H` as one `(m+1)×m` `Vec<T>` (diagonal entry
/// `i` at `h[i*stride + i]`) rather than a `Vec<Vec<T>>`, so this reads the same
/// breakdown-guarded leading dimension off the flat layout. Identical logic.
#[inline]
fn well_conditioned_dim_flat<T: Scalar>(h: &[T], stride: usize, jdim: usize) -> usize {
    if jdim == 0 {
        return 0;
    }
    let mut hmax = 0.0f64;
    for i in 0..jdim {
        hmax = hmax.max(h[i * stride + i].magnitude());
    }
    let thresh = f64::EPSILON * hmax;
    for i in 0..jdim {
        if h[i * stride + i].magnitude() <= thresh {
            return i;
        }
    }
    jdim
}

/// Fixed row-block size for the block-orthogonalization reductions below. A
/// compile-time constant (never thread-count dependent), so the chunked sums are
/// **bit-identical across thread counts** - preserving the block solve's
/// determinism guarantee while spreading the reduction over all cores.
const ORTHO_CHUNK: usize = 2048;

/// Build the scoped rayon pool the block-GMRES orthogonalization reductions should
/// run in (issue #9), from the preconditioner's [`Threads`] policy. `Ambient`
/// returns `None` - the reductions then run on the caller's current pool (the
/// solver-in-the-loop path, where the caller has already installed one bounded
/// pool via [`with_threads`](crate::with_threads)). Any concrete policy builds a
/// pool of that width **once** per solve; the four `block_project` / `block_subtract`
/// calls per step reuse it (cheap `install`), so factor and solve share one
/// concurrency budget instead of the solve fanning out over the global pool. The
/// chunk-order reduction fold is thread-count independent, so the pool never
/// perturbs the bit-identical-across-thread-counts guarantee.
fn solve_thread_pool(policy: Threads) -> Option<rayon::ThreadPool> {
    match policy {
        Threads::Ambient => None,
        // `Fixed(k)` (the resolved factor budget): a `k`-worker pool (`0` = all
        // cores). `Auto` never reaches here - factors resolve it to `Fixed` up
        // front - but the `|cap| cap` fallback keeps this total.
        p => {
            let workers = p.resolve(|cap| cap);
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .ok()
        }
    }
}

/// Run an orthogonalization reduction `f` in the solve pool if one was built,
/// else on the current pool. Confined to closures capturing only Krylov *data*
/// (basis / panel slices) - never the operator or preconditioner - so it never
/// imposes a `Send`/`Sync` bound on the matrix-free (`FnOp`/`FnPc`) call path.
#[inline]
fn ortho_in_pool<R: Send>(pool: &Option<rayon::ThreadPool>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

/// Column-wise projection of a panel `W` (`n×sa`) onto each column's **own**
/// Arnoldi basis: `proj[i*sa + ap] = ⟨V_i[:,ap], W[:,ap]⟩` for block `i` in
/// `0..blocks` and active column `ap` in `0..sa`. The basis is blocks-major -
/// block `i` is the contiguous slice `vbas[i*sa*n .. (i+1)*sa*n]`, its column `ap`
/// at offset `+ap*n`; `W` column `ap` is `w[ap*n .. ap*n+n]`.
///
/// This is the classical (block) Gram-Schmidt projection: **all** projections are
/// taken against the same `W`, so the `blocks·sa` inner products are independent
/// and computed as one panel sweep instead of the `O(blocks·sa)` sequential,
/// latency-bound BLAS-1 reductions of modified Gram-Schmidt. The reduction is a
/// fixed row-chunk sum folded in chunk order → deterministic regardless of the
/// thread count.
/// `scratch` is a caller-owned reduction buffer of length `≥ nchunks · width`
/// (`nchunks = ⌈n/ORTHO_CHUNK⌉`, `width = blocks·sa`), reused across steps so the
/// hot loop allocates nothing. Each chunk writes its `width` partial sums into its
/// own slice; the slices are then folded in chunk order.
#[inline]
fn block_project<T: Scalar>(
    vbas: &[T],
    w: &[T],
    blocks: usize,
    sa: usize,
    n: usize,
    proj: &mut [T],
    scratch: &mut [T],
) {
    let width = blocks * sa;
    for p in proj[..width].iter_mut() {
        *p = T::zero();
    }
    if width == 0 || n == 0 {
        return;
    }
    let nchunks = n.div_ceil(ORTHO_CHUNK);
    let part = &mut scratch[..nchunks * width];
    part.par_chunks_mut(width)
        .enumerate()
        .for_each(|(ci, out)| {
            let r0 = ci * ORTHO_CHUNK;
            let r1 = (r0 + ORTHO_CHUNK).min(n);
            for i in 0..blocks {
                for ap in 0..sa {
                    let vb = (i * sa + ap) * n;
                    let wb = ap * n;
                    let mut sdot = T::zero();
                    for k in r0..r1 {
                        sdot = sdot + vbas[vb + k].conj() * w[wb + k];
                    }
                    out[i * sa + ap] = sdot;
                }
            }
        });
    // Fold partials in chunk order: the summation order is fixed by the chunk
    // layout, so the result does not depend on how many threads ran.
    for ci in 0..nchunks {
        let base = ci * width;
        for t in 0..width {
            proj[t] = proj[t] + part[base + t];
        }
    }
}

/// Subtract the projected components from the panel in place, per column:
/// `W[:,ap] -= Σ_i proj[i*sa+ap] · V_i[:,ap]`, accumulated in block order (`i`
/// ascending) at every element. Parallel over fixed row-chunks within each column
/// → the element-wise order is fixed, so the update is deterministic.
#[inline]
fn block_subtract<T: Scalar>(
    vbas: &[T],
    w: &mut [T],
    blocks: usize,
    sa: usize,
    n: usize,
    proj: &[T],
) {
    for ap in 0..sa {
        let wcol = &mut w[ap * n..ap * n + n];
        wcol.par_chunks_mut(ORTHO_CHUNK)
            .enumerate()
            .for_each(|(ci, wc)| {
                let r0 = ci * ORTHO_CHUNK;
                for i in 0..blocks {
                    let hij = proj[i * sa + ap];
                    if hij == T::zero() {
                        continue;
                    }
                    let vb = (i * sa + ap) * n + r0;
                    for k in 0..wc.len() {
                        wc[k] = wc[k] - hij * vbas[vb + k];
                    }
                }
            });
    }
}

/// **Flexible** right-preconditioned restarted **GMRES(`restart`)** (FGMRES,
/// Saad 1993) for a general (unsymmetric) operator - the natural Krylov method
/// for unsymmetric MoM/FEM systems where COCG/COCR do not apply. `op` may be
/// matrix-free; `precond` supplies `M⁻¹` (e.g. an RLA
/// [`LuFactors`](crate::numeric::multifrontal_lu::LuFactors) near-field factor).
/// Solves `A x = b` from the optional initial guess `x0` (default `x₀ = 0`).
///
/// **Warm start (issue #5):** pass `x0 = Some(prev)` to seed the iteration from a
/// previous, related solution - on a sequence of slowly varying systems this
/// often cuts the iteration count substantially. Convergence is still measured
/// relative to ‖b‖.
///
/// **Flexible variant (issue #7):** the preconditioned Arnoldi vectors
/// `z_j = M⁻¹ v_j` (already formed to build `w = A z_j`) are *kept* as a second
/// basis `Z = [z_0 … z_{m-1}]`, and the restart update is `x += Z y` directly.
/// This (a) removes the one extra `M⁻¹` solve per cycle that plain right-
/// preconditioned GMRES spends rebuilding `M⁻¹(V y)`, and (b) makes the method
/// flexible: `M` may **vary** between steps (an inner Krylov solve, or a
/// preconditioner strengthened across iterations). Cost: one extra `n·(m+1)`
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
    // Warm start (issue #5): seed `x` from the caller's initial guess `x0`; the
    // per-cycle true residual `r = b − A x` then measures progress from that
    // guess. Convergence is still relative to ‖b‖. Absent `x0`, `x₀ = 0`.
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
    // Arnoldi basis as one **flat** `n × (m+1)` buffer (column `i` is
    // `v[i*n .. (i+1)*n]`) - contiguous, no per-iteration vector allocation, and
    // cache-friendly for the Gram-Schmidt sweeps.
    let mut v = vec![T::zero(); n * (m + 1)];
    // FGMRES preconditioned basis `Z = [z_0 … z_{m-1}]`, `z_j = M⁻¹ v_j` (issue
    // #7): stored as it is computed so the restart update is `x += Z y` with no
    // second preconditioner solve, and so a *variable* `M` is honoured exactly.
    let mut zb = vec![T::zero(); n * m];
    // Per-restart Krylov scalars hoisted out of the outer loop and cleared/reused
    // each cycle (issue #10): the Hessenberg `h` as one **flat** `(m+1)×m` buffer
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

    // Outer restart loop. Each pass first measures the TRUE residual ‖b−Ax‖ of the
    // current iterate (the only reliable stop test on ill-conditioned MoM near-
    // field operators, where the Hessenberg LS estimate can dip below `tol` while
    // the true residual is orders larger) and records it as `final_res`. On
    // convergence *or* exhausted iterations we break with that value already in
    // hand - no separate post-loop matvec to report the residual (issue #10).
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
        // Column 0 of the basis = r / ‖r‖. Reset the reused Hessenberg / Givens /
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
            // Flexible right preconditioning: z_j = M⁻¹ v[j] (stored into the Z
            // basis for the restart update), then w = A z_j.
            precond.apply(&v[j * n..j * n + n], &mut zb[j * n..j * n + n])?;
            op.apply(&zb[j * n..j * n + n], &mut w);
            // Modified Gram-Schmidt against the existing basis, with **conditional**
            // reorthogonalization (DGKS): the second pass - essential on
            // ill-conditioned operators (MoM near-field) where a single MGS pass
            // loses orthogonality and the Hessenberg residual estimate drifts from
            // the true residual - runs only when the projection cancelled most of
            // the vector (‖w‖ dropped below `η·‖w₀‖`, η = 1/√2). Well-conditioned
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
        // leading block (breakdown guard), then (FGMRES) x += Z·y directly from the
        // stored preconditioned basis - no second `M⁻¹` solve.
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

// ===========================================================================
// GCRO-DR: Krylov subspace recycling for sequences of related solves (issue #5)
// ===========================================================================
//
// **When it helps.** RSLAB targets solver-in-the-loop workloads: many solves of
// a slowly varying `A` / `M` / `b`. On *stagnating* systems (a cluster of small
// eigenvalues that restarted GMRES keeps re-discovering and discarding), plain
// FGMRES(`m`) throws away the near-invariant subspace at every restart. GCRO-DR
// (Parks, de Sturler, Mackey, Johnson, Maiti, SISC 2006) keeps `k` harmonic-Ritz
// vectors `U` approximating that subspace and (a) *deflates* it within a solve -
// carrying `U` across restarts, which cuts restart counts on a single hard solve -
// and (b) *recycles* it across solves through an opaque [`Recycle`] handle, so the
// next related system starts already knowing where the trouble is.
//
// **Scope.** Recycling is implemented for the single-RHS FGMRES [`gmres`] path
// ([`gmres_recycled`]). The block path is out of scope: its within-cycle deflation
// + basis compaction remap the per-column Arnoldi state mid-cycle, which does not
// compose cleanly with a shared recycle subspace, and warrants its own design pass.
//
// **The split (right-preconditioned / flexible).** The Krylov space is built on
// the preconditioned operator `Ã = A M⁻¹` (FGMRES stores `Z = M⁻¹ V`, updates
// `x += Z y`), so `U` lives in the *preconditioned* space (approximate smallest
// eigenvectors of `Ã`). Each cycle recomputes `P = M⁻¹ U` (`k` preconditioner
// solves) and `C = A P = Ã U` (`k` matvecs), orthonormalizes `C = Q R` (updating
// `P, U ← ·R⁻¹` so `C = Ã U` and `P = M⁻¹ U` still hold), then runs the GCRO
// split: project the residual onto `range(C)` (`x += P C^H r`, `r -= C C^H r`)
// and build the rest of the Krylov space **orthogonal to `C`** (each Arnoldi
// vector has its `C` component removed, recorded in `B = C^H A Z`). The restart
// least-squares problem then decouples: `y` is the ordinary GMRES solution of the
// Hessenberg system and the recycle coordinate is `z₁ = −B y`, giving
// `x += P z₁ + Z y`. Recomputing `C` each solve (the default) is `k` matvecs +
// `k` M-solves and keeps the invariant exact when `A` / `M` change between solves.
//
// **Memory.** `U` and `C` are each `n·k` extra scalars (plus the same-size `P`),
// on top of the FGMRES `V`+`Z` bases. `k` defaults modest and is capped at
// `restart/2` so the deflation never starves the Arnoldi space.
//
// **Correctness is independent of `U`'s quality.** `U` only shapes the Krylov
// space to *accelerate* convergence; the outer true-residual stop test is
// authoritative, so an inaccurate / rank-deficient recycle subspace can only make
// a solve slower, never wrong. Every recycle step is accordingly guarded (rank
// dropping, singular-projection fallbacks) and degrades gracefully to plain FGMRES.

/// Sealing for [`RecycleScalar`]: only the two fields RSLAB's Krylov solvers run
/// in (`f64`, `Complex<f64>`) may carry a recycle subspace, so downstream crates
/// cannot add ill-defined implementations.
mod recycle_sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for num_complex::Complex<f64> {}
    impl Sealed for num_complex::Complex<f32> {}
}

/// Scalar fields that support GCRO-DR recycling. The harmonic-Ritz small
/// eigenproblem is always solved in `Complex<f64>` (see the `dense_eig`
/// internals); this trait provides the two field-specific
/// bridges: promoting a scalar to `Complex<f64>` for the small matrices, and
/// reconstructing the recycle vectors from complex coefficient vectors - which
/// for the **real** field must split a complex conjugate Ritz pair into its real
/// and imaginary parts (two real recycle vectors), since a real `U` cannot store
/// a complex eigenvector directly. A sealed trait (impls for the four scalar
/// fields `f64` / `f32` / `Complex<f64>` / `Complex<f32>`); not user-implementable.
pub trait RecycleScalar: Scalar + recycle_sealed::Sealed {
    /// Promote to `Complex<f64>` for the small dense harmonic-Ritz problem.
    #[doc(hidden)]
    fn to_c(self) -> Complex<f64>;
    /// Reconstruct up to `kmax` recycle vectors (column-major, length `n` each)
    /// from the search-space columns `cols` (`n × d`) and the harmonic-Ritz
    /// `(θ, g)` pairs (coefficient vectors of length `d`, sorted by ascending
    /// `|θ|`). Real fields split each complex conjugate pair into `Re g`, `Im g`.
    #[doc(hidden)]
    fn combine_ritz(
        cols: &[Self],
        n: usize,
        d: usize,
        pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
        kmax: usize,
    ) -> Vec<Self>;
}

/// Real-field recycle reconstruction (`f64` / `f32`): a real `U` cannot hold a
/// complex eigenvector, so each **complex** conjugate Ritz pair contributes its
/// `Re g` and `Im g` (two real vectors spanning the real 2-D invariant subspace),
/// while a real Ritz value contributes one. Multiplies columns only by the real
/// coefficient parts via [`Scalar::from_real`], so it is generic over the real
/// field. Emits at most `kmax` vectors.
fn combine_ritz_real<T: Scalar>(
    cols: &[T],
    n: usize,
    d: usize,
    pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
    kmax: usize,
) -> Vec<T> {
    let mut out: Vec<T> = Vec::new();
    for (theta, g) in pairs {
        if out.len() / n >= kmax {
            break;
        }
        let mag = theta.norm().max(1e-300);
        if theta.im.abs() <= 1e-8 * mag {
            let mut v = vec![T::zero(); n];
            for j in 0..d {
                let re = T::from_real(g[j].re);
                for kk in 0..n {
                    v[kk] = v[kk] + re * cols[j * n + kk];
                }
            }
            out.extend_from_slice(&v);
        } else if theta.im > 0.0 {
            let mut vr = vec![T::zero(); n];
            let mut vi = vec![T::zero(); n];
            for j in 0..d {
                let (re, im) = (T::from_real(g[j].re), T::from_real(g[j].im));
                for kk in 0..n {
                    vr[kk] = vr[kk] + re * cols[j * n + kk];
                    vi[kk] = vi[kk] + im * cols[j * n + kk];
                }
            }
            out.extend_from_slice(&vr);
            if out.len() / n < kmax {
                out.extend_from_slice(&vi);
            }
        }
    }
    out
}

/// Complex-field recycle reconstruction (`Complex<f64>` / `Complex<f32>`): the
/// harmonic-Ritz vectors are genuinely complex, so `U ← [U, V] gᵢ` directly, one
/// vector per pair. `mk` casts a `Complex<f64>` coefficient into the working field
/// (identity for `Complex<f64>`, a narrowing cast for `Complex<f32>`).
fn combine_ritz_complex<T: Scalar>(
    cols: &[T],
    n: usize,
    d: usize,
    pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
    kmax: usize,
    mk: impl Fn(Complex<f64>) -> T,
) -> Vec<T> {
    let mut out: Vec<T> = Vec::new();
    for (i, (_theta, g)) in pairs.iter().enumerate() {
        if i >= kmax {
            break;
        }
        let mut v = vec![T::zero(); n];
        for j in 0..d {
            let gj = mk(g[j]);
            for kk in 0..n {
                v[kk] = v[kk] + gj * cols[j * n + kk];
            }
        }
        out.extend_from_slice(&v);
    }
    out
}

impl RecycleScalar for f64 {
    fn to_c(self) -> Complex<f64> {
        Complex::new(self, 0.0)
    }
    fn combine_ritz(
        cols: &[f64],
        n: usize,
        d: usize,
        pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
        kmax: usize,
    ) -> Vec<f64> {
        combine_ritz_real(cols, n, d, pairs, kmax)
    }
}

impl RecycleScalar for f32 {
    fn to_c(self) -> Complex<f64> {
        Complex::new(self as f64, 0.0)
    }
    fn combine_ritz(
        cols: &[f32],
        n: usize,
        d: usize,
        pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
        kmax: usize,
    ) -> Vec<f32> {
        combine_ritz_real(cols, n, d, pairs, kmax)
    }
}

impl RecycleScalar for Complex<f64> {
    fn to_c(self) -> Complex<f64> {
        self
    }
    fn combine_ritz(
        cols: &[Complex<f64>],
        n: usize,
        d: usize,
        pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
        kmax: usize,
    ) -> Vec<Complex<f64>> {
        combine_ritz_complex(cols, n, d, pairs, kmax, |z| z)
    }
}

impl RecycleScalar for Complex<f32> {
    fn to_c(self) -> Complex<f64> {
        Complex::new(self.re as f64, self.im as f64)
    }
    fn combine_ritz(
        cols: &[Complex<f32>],
        n: usize,
        d: usize,
        pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
        kmax: usize,
    ) -> Vec<Complex<f32>> {
        combine_ritz_complex(cols, n, d, pairs, kmax, |z| {
            Complex::new(z.re as f32, z.im as f32)
        })
    }
}

/// An opaque **recycle subspace** carried across a sequence of related solves
/// ([`gmres_recycled`], issue #5). Holds `k` harmonic-Ritz vectors `U` (in the
/// preconditioned space, i.e. approximate smallest eigenvectors of `A M⁻¹`) that
/// dominate GMRES stagnation, so the next related solve deflates them from the
/// start instead of re-discovering them.
///
/// **Lifetime / validity.** Create once with [`Recycle::new`] (target dimension
/// `k`) and pass `&mut` to each solve; the handle is *updated in place* at the
/// end of every solve with the freshest harmonic-Ritz vectors. It stays useful
/// across solves with the **same or slowly varying** `A` and `M`: the invariant
/// subspace it tracks drifts slowly, and `C = A M⁻¹ U` is recomputed from `U`
/// every solve (`k` matvecs + `k` M-solves), so a changed operator is handled
/// exactly with no stale `C`. If the dimension `n` changes the handle resets
/// itself. Reusing it on an *unrelated* system is safe (never wrong) but may not
/// help. Call [`Recycle::clear`] to forget the accumulated subspace.
///
/// **Memory:** `U` is `n·k` scalars; the transient `P = M⁻¹ U` and `C = A M⁻¹ U`
/// (another `2·n·k`) live only for the duration of a solve.
#[derive(Debug, Clone)]
pub struct Recycle<T> {
    /// Target subspace dimension (capped at `restart/2` per solve).
    k: usize,
    /// Recycle vectors, column-major `n × kc` (`kc ≤ k`), in preconditioned space.
    u: Vec<T>,
    /// Current number of stored recycle vectors.
    kc: usize,
    /// Dimension the stored vectors belong to (`0` when empty).
    n: usize,
}

impl<T: Scalar> Recycle<T> {
    /// A fresh, empty recycle handle targeting `k` harmonic-Ritz vectors. The
    /// first solve seeds `U` (there is nothing to deflate yet); subsequent solves
    /// deflate and refresh it. `k` is capped at `restart/2` inside the solve.
    pub fn new(k: usize) -> Self {
        Recycle {
            k,
            u: Vec::new(),
            kc: 0,
            n: 0,
        }
    }

    /// The target subspace dimension `k`.
    pub fn dim(&self) -> usize {
        self.k
    }

    /// The number of recycle vectors currently stored (`≤ k`, `0` until the first
    /// solve populates it or after [`clear`](Self::clear)).
    pub fn active(&self) -> usize {
        self.kc
    }

    /// Forget the accumulated subspace (e.g. before an unrelated system). The
    /// target dimension `k` is retained.
    pub fn clear(&mut self) {
        self.u.clear();
        self.kc = 0;
        self.n = 0;
    }
}

/// Modified-Gram-Schmidt orthonormalization of the recycle triple `(C, P, U)`
/// (each column-major `n × kc`), maintaining the invariants `C = A P` and
/// `P = M⁻¹ U`: the same linear combination applied to `C`'s columns is applied
/// to `P` and `U`, so after `C ← Q` (orthonormal) the scaled `P, U` still satisfy
/// them. Rank-deficient columns (norm collapses under the projection) are
/// **dropped** and the survivors compacted forward. Returns the surviving count.
fn orthonormalize_recycle<T: Scalar>(
    c: &mut [T],
    p: &mut [T],
    u: &mut [T],
    n: usize,
    kc: usize,
) -> usize {
    let mut w = 0usize;
    for src in 0..kc {
        let mut ccol: Vec<T> = c[src * n..src * n + n].to_vec();
        let mut pcol: Vec<T> = p[src * n..src * n + n].to_vec();
        let mut ucol: Vec<T> = u[src * n..src * n + n].to_vec();
        let orig = norm2(&ccol);
        if orig == 0.0 {
            continue;
        }
        for j in 0..w {
            let r = dotc(&c[j * n..j * n + n], &ccol);
            for kk in 0..n {
                ccol[kk] = ccol[kk] - r * c[j * n + kk];
                pcol[kk] = pcol[kk] - r * p[j * n + kk];
                ucol[kk] = ucol[kk] - r * u[j * n + kk];
            }
        }
        let nrm = norm2(&ccol);
        if nrm <= 1e-12 * orig {
            continue; // rank-deficient direction: drop it
        }
        let inv = T::from_real(1.0 / nrm);
        for kk in 0..n {
            c[w * n + kk] = ccol[kk] * inv;
            p[w * n + kk] = pcol[kk] * inv;
            u[w * n + kk] = ucol[kk] * inv;
        }
        w += 1;
    }
    w
}

/// Modified-Gram-Schmidt orthonormalization of `nc` column-major `n`-vectors in
/// place, dropping rank-deficient columns (the guard for a defective harmonic-Ritz
/// reconstruction, requirement #4). Returns the surviving orthonormal count.
fn orthonormalize_columns<T: Scalar>(v: &mut [T], n: usize, nc: usize) -> usize {
    let mut w = 0usize;
    for src in 0..nc {
        let mut col: Vec<T> = v[src * n..src * n + n].to_vec();
        let orig = norm2(&col);
        if orig == 0.0 {
            continue;
        }
        for j in 0..w {
            let r = dotc(&v[j * n..j * n + n], &col);
            for kk in 0..n {
                col[kk] = col[kk] - r * v[j * n + kk];
            }
        }
        let nrm = norm2(&col);
        if nrm <= 1e-12 * orig {
            continue;
        }
        let inv = T::from_real(1.0 / nrm);
        for kk in 0..n {
            v[w * n + kk] = col[kk] * inv;
        }
        w += 1;
    }
    w
}

/// Recompute the recycle subspace `U` from this cycle's search space by GCRO-DR
/// harmonic-Ritz extraction. Forms the small `(d+1) × d` matrix
/// `Ḡ = [[I_k, B], [0, H̄]]` and the orthonormal image basis `Ŵ = [C, V_{p+1}]`
/// (`d = k + p`, `p = jdim` Arnoldi steps), solves the generalized eigenproblem
/// `Ḡᴴ Ḡ g = θ Ḡᴴ (Ŵᴴ [U, V]) g` for the `k` smallest `|θ|`, and reconstructs
/// `U ← [U, V] G_k` (orthonormalized, rank-guarded). Returns `(U, count)` or
/// `None` (keep the previous subspace) on a singular projection / empty spectrum.
#[allow(clippy::too_many_arguments)]
fn recompute_recycle<T: RecycleScalar>(
    cmat: &[T],
    u: &[T],
    vb: &[T],
    hbar: &[T],
    bmat: &[T],
    n: usize,
    ncur: usize,
    jdim: usize,
    p_arn: usize,
    k: usize,
) -> Option<(Vec<T>, usize)> {
    let d = ncur + jdim;
    if d == 0 || k == 0 {
        return None;
    }
    let rows = d + 1;
    let c0 = Complex::new(0.0, 0.0);
    let c1 = Complex::new(1.0, 0.0);
    // Ḡ (rows × d), row-major, in Complex<f64>.
    let mut ghat = vec![c0; rows * d];
    for i in 0..ncur {
        ghat[i * d + i] = c1; // I_k block
    }
    for i in 0..ncur {
        for j in 0..jdim {
            ghat[i * d + (ncur + j)] = bmat[i * p_arn + j].to_c(); // B block
        }
    }
    for r in 0..(jdim + 1) {
        for j in 0..jdim {
            ghat[(ncur + r) * d + (ncur + j)] = hbar[r * p_arn + j].to_c(); // H̄ block
        }
    }
    // Ŵᴴ [U, V]  (rows × d): [[Cᴴ U, 0], [Vᴴ U, [I_p; 0]]].
    let mut whatu = vec![c0; rows * d];
    for i in 0..ncur {
        for a in 0..ncur {
            whatu[i * d + a] = dotc(&cmat[i * n..i * n + n], &u[a * n..a * n + n]).to_c();
        }
    }
    for r in 0..(jdim + 1) {
        for a in 0..ncur {
            whatu[(ncur + r) * d + a] = dotc(&vb[r * n..r * n + n], &u[a * n..a * n + n]).to_c();
        }
    }
    for j in 0..jdim {
        whatu[(ncur + j) * d + (ncur + j)] = c1;
    }
    // M1 = Ḡᴴ Ḡ,  M2 = Ḡᴴ (Ŵᴴ [U, V]).
    let mut m1 = vec![c0; d * d];
    let mut m2 = vec![c0; d * d];
    for a in 0..d {
        for b in 0..d {
            let mut s1 = c0;
            let mut s2 = c0;
            for r in 0..rows {
                let ga = ghat[r * d + a].conj();
                s1 += ga * ghat[r * d + b];
                s2 += ga * whatu[r * d + b];
            }
            m1[a * d + b] = s1;
            m2[a * d + b] = s2;
        }
    }
    let pairs = crate::numeric::dense_eig::harmonic_ritz_smallest(&m1, &m2, d, k);
    if pairs.is_empty() {
        return None;
    }
    // Search-space columns [U | V_p]  (n × d), column-major.
    let mut space = vec![T::zero(); n * d];
    for a in 0..ncur {
        space[a * n..a * n + n].copy_from_slice(&u[a * n..a * n + n]);
    }
    for j in 0..jdim {
        space[(ncur + j) * n..(ncur + j) * n + n].copy_from_slice(&vb[j * n..j * n + n]);
    }
    let mut newu = T::combine_ritz(&space, n, d, &pairs, k);
    let cnt = newu.len() / n;
    let cnt2 = orthonormalize_columns(&mut newu, n, cnt);
    if cnt2 == 0 {
        return None;
    }
    newu.truncate(cnt2 * n);
    Some((newu, cnt2))
}

/// **GCRO-DR** - flexible right-preconditioned restarted GMRES with **Krylov
/// subspace recycling** (Parks/de Sturler et al. 2006, issue #5). The recycling
/// companion to [`gmres`]: identical convergence semantics and diagnostics, plus
/// a [`Recycle`] handle that (a) deflates a `k`-dimensional near-invariant
/// subspace *across restarts within this solve* and (b) *carries it to the next
/// solve*. See the module-level "GCRO-DR" section for the algorithm and when it
/// pays (stagnating, restart-limited, or slowly-varying-sequence solves).
///
/// `recycle.dim()` = `k` (capped at `restart/2`). Pass the **same** handle to a
/// sequence of related solves; it is refreshed in place at the end of each. The
/// optional `x0` warm start composes with recycling (seed from the previous
/// solution *and* recycle its stagnation subspace). With an empty handle and
/// `k = 0` this reduces to plain [`gmres`].
#[allow(clippy::too_many_arguments)]
pub fn gmres_recycled<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
    restart: usize,
    x0: Option<&[T]>,
    recycle: &mut Recycle<T>,
) -> Result<KrylovResult<T>, RslabError>
where
    T: RecycleScalar,
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
    const REORTH_ETA: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let m = restart.max(1);
    // Cap the recycle dimension at restart/2 so the Arnoldi space (m − k) is never
    // starved. `k = 0` (or an empty handle) degrades to plain FGMRES.
    let k = recycle.k.min(m / 2);
    let bnorm = norm2(b);
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
        recycle.u.clear();
        recycle.kc = 0;
        recycle.n = 0;
        return Ok(KrylovResult {
            x,
            iters: 0,
            converged: true,
            final_res: 0.0,
            stop: StopReason::Converged,
        });
    }

    // Recycle subspace carried in: reset if the dimension changed.
    if recycle.n != n {
        recycle.u.clear();
        recycle.kc = 0;
    }
    let mut u: Vec<T> = recycle.u.clone();
    let mut ncur = recycle.kc.min(k);
    u.truncate(ncur * n);

    let mut w = vec![T::zero(); n];
    let mut ax = vec![T::zero(); n];
    let mut total = 0usize;
    let mut final_res;

    loop {
        // Top-of-cycle TRUE residual (authoritative stop test).
        op.apply(&x, &mut ax);
        let mut r: Vec<T> = (0..n).map(|i| b[i] - ax[i]).collect();
        let beta0 = norm2(&r);
        final_res = beta0 / bnorm;
        if final_res <= tol || total >= max_iter {
            break;
        }

        // --- Recycle refresh: C = A M⁻¹ U, orthonormalize, GCRO project. ---
        let mut cmat = vec![T::zero(); n * ncur];
        let mut pmat = vec![T::zero(); n * ncur];
        if ncur > 0 {
            for i in 0..ncur {
                precond.apply(&u[i * n..i * n + n], &mut pmat[i * n..i * n + n])?;
                op.apply(&pmat[i * n..i * n + n], &mut cmat[i * n..i * n + n]);
            }
            ncur = orthonormalize_recycle(&mut cmat, &mut pmat, &mut u, n, ncur);
            // Outer minimization over range(C): x += P (Cᴴ r), r -= C (Cᴴ r).
            for i in 0..ncur {
                let di = dotc(&cmat[i * n..i * n + n], &r);
                for kk in 0..n {
                    x[kk] = x[kk] + pmat[i * n + kk] * di;
                    r[kk] = r[kk] - cmat[i * n + kk] * di;
                }
            }
        }
        let beta = norm2(&r);
        if beta == 0.0 {
            continue; // exact in the recycle space; loop top confirms convergence
        }

        // Arnoldi dimension this cycle (orthogonal to C).
        let p_arn = m - ncur;
        let inv_beta = T::from_real(1.0 / beta);
        let mut vb = vec![T::zero(); n * (p_arn + 1)];
        let mut zb = vec![T::zero(); n * p_arn];
        for kk in 0..n {
            vb[kk] = r[kk] * inv_beta;
        }
        // Raw (unrotated) Hessenberg for harmonic Ritz; rotated copy for the LS.
        let mut hbar = vec![T::zero(); (p_arn + 1) * p_arn];
        let mut rmat = vec![T::zero(); (p_arn + 1) * p_arn];
        let mut bmat = vec![T::zero(); ncur.max(1) * p_arn]; // C-projection coeffs
        let mut cs = vec![T::zero(); p_arn];
        let mut sn = vec![T::zero(); p_arn];
        let mut g = vec![T::zero(); p_arn + 1];
        let mut y = vec![T::zero(); p_arn];
        g[0] = T::from_real(beta);
        let mut jdim = 0usize;
        for j in 0..p_arn {
            if total >= max_iter {
                break;
            }
            precond.apply(&vb[j * n..j * n + n], &mut zb[j * n..j * n + n])?;
            op.apply(&zb[j * n..j * n + n], &mut w);
            let wnorm0 = norm2(&w);
            // Project the new direction orthogonal to C (record B[:,j] = Cᴴ w).
            for i in 0..ncur {
                let bij = dotc(&cmat[i * n..i * n + n], &w);
                bmat[i * p_arn + j] = bij;
                for kk in 0..n {
                    w[kk] = w[kk] - cmat[i * n + kk] * bij;
                }
            }
            // Modified Gram-Schmidt against V_0..V_j.
            for i in 0..=j {
                let hij = dotc(&vb[i * n..i * n + n], &w);
                hbar[i * p_arn + j] = hij;
                for kk in 0..n {
                    w[kk] = w[kk] - vb[i * n + kk] * hij;
                }
            }
            let mut hn = norm2(&w);
            // Conditional DGKS second pass (against C and V).
            if hn < REORTH_ETA * wnorm0 {
                for i in 0..ncur {
                    let s = dotc(&cmat[i * n..i * n + n], &w);
                    bmat[i * p_arn + j] = bmat[i * p_arn + j] + s;
                    for kk in 0..n {
                        w[kk] = w[kk] - cmat[i * n + kk] * s;
                    }
                }
                for i in 0..=j {
                    let s = dotc(&vb[i * n..i * n + n], &w);
                    hbar[i * p_arn + j] = hbar[i * p_arn + j] + s;
                    for kk in 0..n {
                        w[kk] = w[kk] - vb[i * n + kk] * s;
                    }
                }
                hn = norm2(&w);
            }
            hbar[(j + 1) * p_arn + j] = T::from_real(hn);
            if hn > 0.0 {
                let inv = T::from_real(1.0 / hn);
                for kk in 0..n {
                    vb[(j + 1) * n + kk] = w[kk] * inv;
                }
            } else {
                for kk in 0..n {
                    vb[(j + 1) * n + kk] = T::zero();
                }
            }
            // Copy the raw column into the rotated buffer and apply Givens for LS.
            for i in 0..=(j + 1) {
                rmat[i * p_arn + j] = hbar[i * p_arn + j];
            }
            for i in 0..j {
                let temp =
                    cs[i].conj() * rmat[i * p_arn + j] + sn[i].conj() * rmat[(i + 1) * p_arn + j];
                rmat[(i + 1) * p_arn + j] =
                    -sn[i] * rmat[i * p_arn + j] + cs[i] * rmat[(i + 1) * p_arn + j];
                rmat[i * p_arn + j] = temp;
            }
            let (c, s) = givens(rmat[j * p_arn + j], rmat[(j + 1) * p_arn + j]);
            cs[j] = c;
            sn[j] = s;
            rmat[j * p_arn + j] =
                c.conj() * rmat[j * p_arn + j] + s.conj() * rmat[(j + 1) * p_arn + j];
            rmat[(j + 1) * p_arn + j] = T::zero();
            let g_next = -s * g[j];
            g[j] = c.conj() * g[j];
            g[j + 1] = g_next;
            total += 1;
            jdim = j + 1;
            if g[j + 1].magnitude() / bnorm <= tol {
                break;
            }
        }

        // Least-squares solve on the well-conditioned leading block.
        let jd = well_conditioned_dim_flat(&rmat, p_arn, jdim);
        for i in (0..jd).rev() {
            let mut s = g[i];
            for kk in (i + 1)..jd {
                s = s - rmat[i * p_arn + kk] * y[kk];
            }
            y[i] = s * rmat[i * p_arn + i].recip();
        }
        // x += Z y   (FGMRES: preconditioned basis, no extra M-solve).
        for i in 0..jd {
            let yi = y[i];
            for kk in 0..n {
                x[kk] = x[kk] + zb[i * n + kk] * yi;
            }
        }
        // x += P z₁ with z₁ = −B y   (the recycle-coordinate update).
        for i in 0..ncur {
            let mut z1i = T::zero();
            for j in 0..jd {
                z1i = z1i - bmat[i * p_arn + j] * y[j];
            }
            for kk in 0..n {
                x[kk] = x[kk] + pmat[i * n + kk] * z1i;
            }
        }

        // --- Refresh the recycle subspace from this cycle's harmonic Ritz. ---
        if k > 0 && jdim > 0 {
            if let Some((newu, cnt)) =
                recompute_recycle(&cmat, &u, &vb, &hbar, &bmat, n, ncur, jdim, p_arn, k)
            {
                u = newu;
                ncur = cnt;
            }
        }
    }

    // Persist the freshest recycle subspace for the next related solve.
    recycle.u = u;
    recycle.kc = ncur;
    recycle.n = n;

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

/// Outcome of a block (multi-RHS) Krylov solve.
#[derive(Debug, Clone)]
pub struct BlockKrylovResult<T> {
    /// Solutions, column-major `n×s` (RHS `c` is the slice `[c·n, (c+1)·n)`).
    pub x: Vec<T>,
    /// Block iterations performed (the RHS advance in lockstep).
    pub iters: usize,
    /// `true` iff **every** RHS reached `tol`.
    pub converged: bool,
    /// Per-RHS final relative residual `‖b_c − A x_c‖ / ‖b_c‖`.
    pub final_res: Vec<f64>,
    /// Why the block iteration stopped: `Converged` when every RHS met `tol`,
    /// else `MaxIter` (the block loop has no non-converged breakdown state).
    pub stop: StopReason,
}

/// Form and add one converged column's solution contribution to the global `x`
/// **mid-cycle**, so the column can be compacted out of the active panel: back-
/// substitute `y` from that column's frozen Hessenberg/Givens state (`h`,`g`,`jd`
/// rows), build `V_ap · y` from its Arnoldi basis at the *current* stride `sa`,
/// apply the preconditioner once (`x_c += M⁻¹·(V_ap·y)`). This is the block
/// analogue of the single-RHS restart update, issued for a single column the
/// instant its Hessenberg estimate reaches `tol`, so the batched applies can then
/// shrink to the still-active width.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn finalize_block_column<T, M>(
    precond: &M,
    vbas: &[T],
    h: &[Vec<Vec<T>>],
    g: &[Vec<T>],
    jd: usize,
    ap: usize,
    sa: usize,
    n: usize,
    c: usize,
    x: &mut [T],
) -> Result<(), RslabError>
where
    T: Scalar,
    M: Preconditioner<T> + ?Sized,
{
    // Guard against a (near-)singular Hessenberg diagonal: solve only the well-
    // conditioned leading block (deterministic breakdown, no NaN into `x`).
    let jd = well_conditioned_dim(&h[ap], jd);
    if jd == 0 {
        return Ok(());
    }
    let mut y = vec![T::zero(); jd];
    for i in (0..jd).rev() {
        let mut acc = g[ap][i];
        for k in (i + 1)..jd {
            acc = acc - h[ap][i][k] * y[k];
        }
        y[i] = acc * h[ap][i][i].recip();
    }
    let mut vy = vec![T::zero(); n];
    for i in 0..jd {
        let yi = y[i];
        let vb = (i * sa + ap) * n;
        for k in 0..n {
            vy[k] = vy[k] + vbas[vb + k] * yi;
        }
    }
    let mut z = vec![T::zero(); n];
    precond.apply(&vy, &mut z)?;
    let cb = c * n;
    for k in 0..n {
        x[cb + k] = x[cb + k] + z[k];
    }
    Ok(())
}

/// Right-preconditioned restarted **block GMRES** for `s` right-hand sides `b`
/// (column-major `n×s`). The `s` systems advance in lockstep so the two expensive
/// operations - the operator matvec and the preconditioner solve - are issued
/// once per step as **block** applies ([`LinearOperator::apply_block`] /
/// [`Preconditioner::apply_block`]), reaching BLAS-3 arithmetic intensity (each
/// factor / matrix value touched once for all `s` columns). Each RHS keeps its
/// own Arnoldi basis / Hessenberg / Givens, so a column converges identically to
/// the single-RHS [`gmres`]; the systems share only the batched operator and
/// preconditioner calls. Solves `A X = B` from the optional initial guess `x0`
/// (column-major `n×s`, default `X₀ = 0`).
///
/// **Warm start (issue #5):** `x0 = Some(prev)` seeds every column from a related
/// previous solution; on a slowly varying sequence this cuts the block iteration
/// count. Each column's convergence is still relative to its own `‖B[:,c]‖`.
///
/// **Deflation:** a RHS whose true residual reaches `tol` drops out of the block,
/// so the batched applies shrink to the active width as columns converge - the
/// fast-converging RHS are never dragged along by the slowest one.
///
/// This is the MoM/FEM many-excitations path: factor (or `f32`-factor) once, then
/// drive all right-hand sides through one block iteration.
///
/// **Memory (issue #12):** the Arnoldi basis is a single up-front allocation of
/// `n·s·(restart+1)` scalars (plus a handful of `n·s` work panels), *independent*
/// of how few iterations actually run - so a large `restart` on a big `n·s` can
/// allocate many GB (`n=100k, s=10, Complex<f64>, restart=80` ≈ 13 GB). Size
/// `restart` to the memory budget; the Python binding caps an unspecified
/// `restart` automatically (an explicit value is honoured exactly).
///
/// **Threads (issue #9):** the parallel orthogonalization reductions run in a
/// scoped pool derived from the preconditioner's [`Threads`] policy
/// ([`Preconditioner::solve_threads`], resolved at factor time), so factor and
/// solve share **one** concurrency budget. A [`Threads::Ambient`] policy (or
/// [`NoPreconditioner`]) leaves the reductions on the caller's current pool - the
/// solver-in-the-loop path, where one bounded pool is installed via
/// [`with_threads`](crate::with_threads) around the whole factor+solve loop. The
/// pool width never changes the numeric result (the chunk-order reduction fold is
/// thread-count independent). The single-RHS [`gmres`] orthogonalizes serially, so
/// it has no such pool.
///
/// **Orthogonalization (issue #8):** the panel is orthogonalized by **block CGS2**
/// (classical Gram-Schmidt with a conditional, now *per-column*, second pass), not
/// the MGS+DGKS of the single-RHS [`gmres`]. The two summation orders differ, so a
/// block solve with `s = 1` is **not** bit-identical to [`gmres`] and may differ by
/// up to ±1 iteration - both still converge to `tol`. See the module-level
/// "Orthogonalization" note for the rationale.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn gmres_block<T, A, M>(
    op: &A,
    b: &[T],
    s: usize,
    precond: &M,
    tol: f64,
    max_iter: usize,
    restart: usize,
    x0: Option<&[T]>,
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    gmres_block_mon(op, b, s, precond, tol, max_iter, restart, x0, None)
}

/// [`gmres_block`] with an optional per-cycle progress MONITOR (rapidmom-local addition,
/// upstream candidate): at the start of every restart cycle — right after the true
/// residuals `‖b − A·x‖/‖b‖` of all live columns were recomputed — `mon` receives
/// `(iters_done, worst_live_residual, n_active_columns)` and returns whether the solve
/// should CONTINUE. Long solves stop being a black box: the caller can stream residual
/// trajectories to its log, and a `false` return cancels the solve early (a stagnation
/// detector cutting a stopped-contracting iteration) with `StopReason::Stalled` and the
/// best solution so far. `None` is the exact old behavior.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn gmres_block_mon<T, A, M>(
    op: &A,
    b: &[T],
    s: usize,
    precond: &M,
    tol: f64,
    max_iter: usize,
    restart: usize,
    x0: Option<&[T]>,
    mut mon: Option<&mut dyn FnMut(usize, f64, usize) -> bool>,
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if s == 0 || b.len() != n * s {
        return Err(RslabError::DimensionMismatch {
            expected: n * s,
            got: b.len(),
        });
    }
    const REORTH_ETA: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let m = restart.max(1);
    // Solve-phase thread policy (issue #9): orthogonalize in a pool of the same
    // width the preconditioner was factored with, so factor and solve share one
    // concurrency budget. `None` (Ambient / no factor) keeps the caller's pool.
    let ortho_pool = solve_thread_pool(precond.solve_threads());
    // Warm start (issue #5): seed every column from `x0` (column-major `n×s`).
    let mut x = match x0 {
        Some(g) => {
            if g.len() != n * s {
                return Err(RslabError::DimensionMismatch {
                    expected: n * s,
                    got: g.len(),
                });
            }
            g.to_vec()
        }
        None => vec![T::zero(); n * s],
    };
    let bnorm: Vec<f64> = (0..s).map(|c| norm2(&b[c * n..c * n + n])).collect();
    // Set when the progress monitor cancels the solve (`StopReason::Stalled`).
    let mut stalled = false;

    // Scratch sized for the **full** width `s` and reused; each restart cycle uses
    // only the first `sa` columns, where `sa` is the count of still-active RHS.
    // The basis stride is therefore `sa` (recomputed per cycle): block `j`, active
    // column `a` lives at `vbas[(j*sa + a)*n ..]`, so block `j` (the `n×sa` input
    // to one block apply) is the contiguous prefix `vbas[j*sa*n .. (j+1)*sa*n]`.
    let mut vbas = vec![T::zero(); n * s * (m + 1)];
    let mut zblk = vec![T::zero(); n * s]; // M⁻¹ · (block j)
    let mut wblk = vec![T::zero(); n * s]; // A · zblk
    let mut axblk = vec![T::zero(); n * s];
    let mut vyblk = vec![T::zero(); n * s];
    let mut xc = vec![T::zero(); n * s]; // compact live-RHS solutions for the residual matvec
                                         // Block Gram-Schmidt projection panels: `proj[i*sa + ap]` for blocks `0..=j`,
                                         // columns `0..sa`. One classical (block) projection pass, plus a **conditional**
                                         // reorthogonalization pass (block DGKS) taken only when a column loses
                                         // orthogonality - the backward-stable, single-thread-cheap analogue of the
                                         // old per-RHS MGS+DGKS, now batched over the whole panel.
    let mut proj1 = vec![T::zero(); m * s];
    let mut proj2 = vec![T::zero(); m * s];
    let mut wnorm0 = vec![0.0f64; s]; // panel column norms before ortho (DGKS reorth test)
    let mut reorth_col = vec![false; s]; // per-column DGKS second-pass flags (issue #8)
                                         // Reduction scratch for `block_project`: `nchunks · (m·s)`, reused every step
                                         // so the orthogonalization allocates nothing in the hot loop.
    let mut proj_scratch = vec![T::zero(); n.div_ceil(ORTHO_CHUNK) * m * s];

    // Per-active-position Arnoldi state (indexed `0..sa`, reset each cycle).
    let mut h: Vec<Vec<Vec<T>>> = (0..s).map(|_| vec![vec![T::zero(); m]; m + 1]).collect();
    let mut cs: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); m]).collect();
    let mut sn: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); m]).collect();
    let mut g: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); m + 1]).collect();
    let mut jdim = vec![0usize; s];
    let mut converged = vec![false; s];
    let mut final_res = vec![0.0f64; s];
    // Back-substitution buffer for the per-column restart update, hoisted out of
    // the hot loop and reused (issue #10): each column's back-sub overwrites the
    // `0..jd` prefix before reading it, so no per-column-per-cycle allocation.
    let mut y = vec![T::zero(); m];
    let mut total = 0usize;
    for c in 0..s {
        if bnorm[c] == 0.0 {
            converged[c] = true;
            // Exact solution of `A x = 0` is `0`; discard any warm-start seed here.
            x[c * n..c * n + n].iter_mut().for_each(|v| *v = T::zero());
        }
    }

    while total < max_iter {
        // **Deflation:** gather the not-yet-converged ("live") RHS into a compact
        // block, recompute their true residual with one block matvec, and keep
        // only those still above `tol` as the active set for this cycle. Converged
        // columns never re-enter the (expensive) inner block applies again.
        let live: Vec<usize> = (0..s).filter(|&c| !converged[c]).collect();
        if live.is_empty() {
            break;
        }
        let lw = live.len();
        for (a, &c) in live.iter().enumerate() {
            xc[a * n..a * n + n].copy_from_slice(&x[c * n..c * n + n]);
        }
        op.apply_block(&xc[..lw * n], &mut axblk[..lw * n], lw);
        // Active set = live RHS whose true residual still exceeds `tol`.
        let mut act: Vec<usize> = Vec::new();
        for (a, &c) in live.iter().enumerate() {
            let cb = c * n;
            let ab = a * n;
            let mut rn = 0.0;
            for i in 0..n {
                rn += (b[cb + i] - axblk[ab + i]).magnitude_sq();
            }
            let beta = rn.sqrt();
            final_res[c] = beta / bnorm[c];
            if final_res[c] <= tol {
                converged[c] = true;
            } else {
                // Initialize active position `act.len()` from this residual.
                let ap = act.len();
                let inv = T::from_real(1.0 / beta);
                for i in 0..n {
                    vbas[ap * n + i] = (b[cb + i] - axblk[ab + i]) * inv; // block 0, col ap
                }
                for row in h[ap].iter_mut() {
                    for e in row.iter_mut() {
                        *e = T::zero();
                    }
                }
                for e in g[ap].iter_mut() {
                    *e = T::zero();
                }
                g[ap][0] = T::from_real(beta);
                jdim[ap] = 0;
                act.push(c);
            }
        }
        let mut sa = act.len();
        // Per-cycle progress monitor (rapidmom-local addition): true residuals of all
        // live columns are fresh at this point — the honest place to report, and the
        // honest place to CANCEL (a `false` return stops a stagnated solve early).
        if let Some(m) = mon.as_mut() {
            let worst = live.iter().map(|&c| final_res[c]).fold(0.0_f64, f64::max);
            if !m(total, worst, sa) {
                stalled = true;
                break;
            }
        }
        if sa == 0 {
            break;
        }

        let mut inner_done = vec![false; sa];
        for j in 0..m {
            if total >= max_iter {
                break;
            }
            let jblock = j * sa * n;
            // Batched right preconditioning + operator apply over the active block.
            precond.apply_block(&vbas[jblock..jblock + sa * n], &mut zblk[..sa * n], sa, n)?;
            op.apply_block(&zblk[..sa * n], &mut wblk[..sa * n], sa);
            // **Block Gram-Schmidt** of the whole `sa`-column panel `W` against each
            // column's own basis `V_0..V_j`: one classical projection pass
            // (project → subtract), then a **conditional** reorthogonalization pass
            // (block DGKS) taken only when a column's norm collapses - the
            // backward-stable, single-thread-cheap analogue of the old per-RHS
            // MGS+DGKS. Both passes are panel-wide, high-arithmetic-intensity sweeps
            // parallelized over the vector dimension, replacing the `O(j·sa)`
            // sequential BLAS-1 inner products. The reorth decision is taken from
            // serial column norms, so it is identical across thread counts (the
            // whole solve stays bit-identical regardless of parallelism). Converged
            // columns are still swept (kept in the panel for batching) but their
            // Hessenberg/Givens state is frozen below.
            let blocks = j + 1;
            for ap in 0..sa {
                wnorm0[ap] = norm2(&wblk[ap * n..ap * n + n]);
            }
            ortho_in_pool(&ortho_pool, || {
                block_project(&vbas, &wblk, blocks, sa, n, &mut proj1, &mut proj_scratch)
            });
            ortho_in_pool(&ortho_pool, || {
                block_subtract(&vbas, &mut wblk, blocks, sa, n, &proj1)
            });
            // **Per-column** DGKS second pass (issue #8): decide the reorth *per
            // column* from its own norm collapse, not panel-globally. Frozen
            // (converged-this-cycle) columns are excluded (they are compacted out at
            // each step boundary, so a stale/collapsed frozen column can never
            // trigger a reorth). Only columns that actually lost orthogonality get
            // the second projection subtracted; the rest keep their pass-1 result
            // bit-for-bit - a single ill-conditioned column no longer imposes the
            // arithmetic second orthogonalization on the well-conditioned columns.
            let mut any_reorth = false;
            for ap in 0..sa {
                let need =
                    !inner_done[ap] && norm2(&wblk[ap * n..ap * n + n]) < REORTH_ETA * wnorm0[ap];
                reorth_col[ap] = need;
                any_reorth |= need;
            }
            if any_reorth {
                ortho_in_pool(&ortho_pool, || {
                    block_project(&vbas, &wblk, blocks, sa, n, &mut proj2, &mut proj_scratch)
                });
                // Zero the second-pass projection for columns that do not need it, so
                // `block_subtract` skips them (its `hij == 0` guard) and their `w`
                // and Hessenberg entries stay exactly at the pass-1 values.
                for ap in 0..sa {
                    if !reorth_col[ap] {
                        for i in 0..blocks {
                            proj2[i * sa + ap] = T::zero();
                        }
                    }
                }
                ortho_in_pool(&ortho_pool, || {
                    block_subtract(&vbas, &mut wblk, blocks, sa, n, &proj2)
                });
            }
            for ap in 0..sa {
                if inner_done[ap] {
                    continue;
                }
                let wb = ap * n;
                // Hessenberg column: projection pass, plus the per-column reorth
                // correction (zero for columns that did not reorth, so `hij == proj1`).
                for i in 0..=j {
                    let hij = if any_reorth {
                        proj1[i * sa + ap] + proj2[i * sa + ap]
                    } else {
                        proj1[i * sa + ap]
                    };
                    h[ap][i][j] = hij;
                }
                let hn = norm2(&wblk[wb..wb + n]);
                h[ap][j + 1][j] = T::from_real(hn);
                let v1 = ((j + 1) * sa + ap) * n;
                if hn > 0.0 {
                    let inv = T::from_real(1.0 / hn);
                    for k in 0..n {
                        vbas[v1 + k] = wblk[wb + k] * inv;
                    }
                } else {
                    for k in 0..n {
                        vbas[v1 + k] = T::zero();
                    }
                }
                // Previous rotations, then a new one to zero h[j+1][j]; update g.
                for i in 0..j {
                    let temp = cs[ap][i].conj() * h[ap][i][j] + sn[ap][i].conj() * h[ap][i + 1][j];
                    h[ap][i + 1][j] = -sn[ap][i] * h[ap][i][j] + cs[ap][i] * h[ap][i + 1][j];
                    h[ap][i][j] = temp;
                }
                let (cj, sj) = givens(h[ap][j][j], h[ap][j + 1][j]);
                cs[ap][j] = cj;
                sn[ap][j] = sj;
                h[ap][j][j] = cj.conj() * h[ap][j][j] + sj.conj() * h[ap][j + 1][j];
                h[ap][j + 1][j] = T::zero();
                let g_next = -sj * g[ap][j];
                g[ap][j] = cj.conj() * g[ap][j];
                g[ap][j + 1] = g_next;
                jdim[ap] = j + 1;
                if g[ap][j + 1].magnitude() / bnorm[act[ap]] <= tol {
                    inner_done[ap] = true;
                }
            }
            total += 1;
            // **Within-cycle deflation.** Columns whose Hessenberg estimate reached
            // `tol` this step are (a) finalized now - their solution contribution is
            // formed from the basis at the *current* stride and added to `x` - and
            // (b) compacted out of the active panel, remapping the per-column
            // Arnoldi state and rewriting the basis at the narrower stride `sa`.
            // The next step's batched preconditioner / operator applies therefore
            // shrink to the still-active width, exactly as promised: a fast RHS is
            // no longer dragged along by the slowest one until the next restart.
            if inner_done.iter().any(|&d| d) {
                for ap in 0..sa {
                    if inner_done[ap] {
                        finalize_block_column(
                            precond, &vbas, &h, &g, jdim[ap], ap, sa, n, act[ap], &mut x,
                        )?;
                    }
                }
                let survivors: Vec<usize> = (0..sa).filter(|&ap| !inner_done[ap]).collect();
                let sa_new = survivors.len();
                // Basis columns `0..=j+1` are populated for the survivors. Compact
                // to the new stride blocks-outer / survivors-inner so the write
                // offset is monotonically increasing and never clobbers an unread
                // source (each column's new offset is `<=` its old offset).
                let blocks_built = j + 2;
                for i in 0..blocks_built {
                    for (ap_new, &ap_old) in survivors.iter().enumerate() {
                        let src = (i * sa + ap_old) * n;
                        let dst = (i * sa_new + ap_new) * n;
                        if src != dst {
                            vbas.copy_within(src..src + n, dst);
                        }
                    }
                }
                // Remap the per-column Arnoldi state (Vec swaps are O(1) pointer
                // moves; the frozen columns' now-stale slots are never read again -
                // the next cycle re-initializes positions `0..sa`).
                for (ap_new, &ap_old) in survivors.iter().enumerate() {
                    if ap_new != ap_old {
                        h.swap(ap_new, ap_old);
                        cs.swap(ap_new, ap_old);
                        sn.swap(ap_new, ap_old);
                        g.swap(ap_new, ap_old);
                        jdim[ap_new] = jdim[ap_old];
                        act[ap_new] = act[ap_old];
                    }
                }
                sa = sa_new;
                act.truncate(sa);
                inner_done = vec![false; sa];
                if sa == 0 {
                    break;
                }
            }
        }

        // x_c += M⁻¹ (V_a y_a): back-substitute each still-active RHS, build the
        // compact VY block, one batched preconditioner apply, then scatter to
        // global `x`. Columns that deflated mid-cycle were already finalized
        // individually above, so `sa == 0` here means the whole panel converged
        // within the cycle and there is nothing left to batch.
        if sa > 0 {
            for e in vyblk[..sa * n].iter_mut() {
                *e = T::zero();
            }
            for ap in 0..sa {
                // Guard the per-column solve against a (near-)singular Hessenberg
                // diagonal: truncate to the well-conditioned leading block so a
                // rank-deficient column breaks down deterministically instead of
                // dividing by ~0 and polluting the batched applies with NaN.
                let jd = well_conditioned_dim(&h[ap], jdim[ap]);
                if jd == 0 {
                    continue;
                }
                // Reuse the hoisted `y` buffer: back-sub writes `y[0..jd]` top-down
                // (each entry before it is read), so no per-column allocation.
                for i in (0..jd).rev() {
                    let mut acc = g[ap][i];
                    for k in (i + 1)..jd {
                        acc = acc - h[ap][i][k] * y[k];
                    }
                    y[i] = acc * h[ap][i][i].recip();
                }
                let vyb = ap * n;
                for i in 0..jd {
                    let yi = y[i];
                    let vb = (i * sa + ap) * n;
                    for k in 0..n {
                        vyblk[vyb + k] = vyblk[vyb + k] + vbas[vb + k] * yi;
                    }
                }
            }
            precond.apply_block(&vyblk[..sa * n], &mut zblk[..sa * n], sa, n)?;
            for ap in 0..sa {
                let c = act[ap];
                for i in 0..n {
                    x[c * n + i] = x[c * n + i] + zblk[ap * n + i];
                }
            }
        }
    }

    // Final true residual per RHS (issue #10). A converged column was measured at
    // its top-of-cycle deflation checkpoint and frozen (never re-entered the active
    // set), so its recorded `final_res` is already the final true residual - reuse
    // it. Re-matvec **only** the columns still active at the iteration budget
    // (`converged[c] == false`), compacted into one narrow block apply, instead of a
    // full-width `s` matvec over columns that deflated cycles ago. For a fully
    // converged solve this skips the final matvec entirely; the reused values are
    // bit-identical to a full recompute (the operator is column-independent).
    let pending: Vec<usize> = (0..s)
        .filter(|&c| !converged[c] && bnorm[c] != 0.0)
        .collect();
    if !pending.is_empty() {
        let lp = pending.len();
        for (a, &c) in pending.iter().enumerate() {
            xc[a * n..a * n + n].copy_from_slice(&x[c * n..c * n + n]);
        }
        op.apply_block(&xc[..lp * n], &mut axblk[..lp * n], lp);
        for (a, &c) in pending.iter().enumerate() {
            let cb = c * n;
            let ab = a * n;
            let mut rn = 0.0;
            for i in 0..n {
                rn += (b[cb + i] - axblk[ab + i]).magnitude_sq();
            }
            final_res[c] = rn.sqrt() / bnorm[c];
        }
    }
    let mut all_conv = true;
    for c in 0..s {
        if bnorm[c] == 0.0 {
            final_res[c] = 0.0;
            continue;
        }
        if final_res[c] > tol {
            all_conv = false;
        }
    }
    Ok(BlockKrylovResult {
        x,
        iters: total,
        converged: all_conv,
        final_res,
        stop: if all_conv {
            StopReason::Converged
        } else if stalled {
            StopReason::Stalled
        } else {
            StopReason::MaxIter
        },
    })
}

/// Adapter: a closure block-matvec `op(x, y, s)` (`Y ← A·X`, column-major `n×s`) as a
/// [`LinearOperator`] for the matrix-free call path. `FnMut` (the Arnoldi issues applies
/// sequentially) so the operator's own scratch lives in the closure capture - no struct, no
/// interior-mutability dance at the call site. The `RefCell` is borrowed for one apply at a time.
struct FnOp<F> {
    f: std::cell::RefCell<F>,
    n: usize,
}
impl<T: Scalar, F: FnMut(&[T], &mut [T], usize)> LinearOperator<T> for FnOp<F> {
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

/// Adapter: a closure block-preconditioner `pc(r, z, s)` (`Z ← M⁻¹·R`) as a [`Preconditioner`].
struct FnPc<G> {
    f: std::cell::RefCell<G>,
}
impl<T: Scalar, G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>> Preconditioner<T>
    for FnPc<G>
{
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        (self.f.borrow_mut())(r, z, 1)
    }
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, _n: usize) -> Result<(), RslabError> {
        (self.f.borrow_mut())(r, z, s)
    }
}

/// Closure entry point for [`gmres_block`]: pass the block matvec and block preconditioner as
/// `FnMut` closures plus the dimension `n` - the natural form for a **matrix-free** MoM/FEM
/// operator that captures its own assembly data + scratch, with no `LinearOperator`/`Preconditioner`
/// boilerplate. For an unpreconditioned solve pass `|r, z, _| { z.copy_from_slice(r); Ok(()) }`.
#[allow(clippy::too_many_arguments)]
pub fn gmres_block_fn<T, F, G>(
    op: F,
    precond: G,
    b: &[T],
    s: usize,
    n: usize,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>,
{
    let op = FnOp {
        f: std::cell::RefCell::new(op),
        n,
    };
    let pc = FnPc {
        f: std::cell::RefCell::new(precond),
    };
    gmres_block(&op, b, s, &pc, tol, max_iter, restart, None)
}

/// Closure entry point for [`gmres_block_mon`] — [`gmres_block_fn`] plus the per-cycle
/// progress monitor (rapidmom-local addition, upstream candidate).
#[allow(clippy::too_many_arguments)]
pub fn gmres_block_fn_mon<T, F, G>(
    op: F,
    precond: G,
    b: &[T],
    s: usize,
    n: usize,
    tol: f64,
    max_iter: usize,
    restart: usize,
    mon: Option<&mut dyn FnMut(usize, f64, usize) -> bool>,
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>,
{
    let op = FnOp {
        f: std::cell::RefCell::new(op),
        n,
    };
    let pc = FnPc {
        f: std::cell::RefCell::new(precond),
    };
    gmres_block_mon(&op, b, s, &pc, tol, max_iter, restart, None, mon)
}

/// Closure entry point for [`gmres`] (single RHS) - see [`gmres_block_fn`].
#[allow(clippy::too_many_arguments)]
pub fn gmres_fn<T, F, G>(
    op: F,
    precond: G,
    b: &[T],
    n: usize,
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<KrylovResult<T>, RslabError>
where
    T: Scalar,
    F: FnMut(&[T], &mut [T], usize),
    G: FnMut(&[T], &mut [T], usize) -> Result<(), RslabError>,
{
    let op = FnOp {
        f: std::cell::RefCell::new(op),
        n,
    };
    let pc = FnPc {
        f: std::cell::RefCell::new(precond),
    };
    gmres(&op, b, &pc, tol, max_iter, restart, None)
}

/// A factorization usable as both a **direct solver** and a [`Preconditioner`].
/// Implemented by the symmetric [`LdltSolver`] and the general
/// [`LuFactors`](crate::numeric::multifrontal_lu::LuFactors), so a caller's
/// solver loop can hold `&dyn Factorization` and swap symmetric/general,
/// exact/incomplete, or `f64`/`f32` factors freely.
pub trait Factorization<T: Scalar>: Preconditioner<T> {
    /// Solve `A x = b` directly from the stored factor.
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError>;
    /// Stored fill (factor nonzeros) - the memory metric.
    fn factor_nnz(&self) -> usize;
    /// Number of statically perturbed pivots (0 for an exact factor).
    fn n_perturbed(&self) -> usize;
}

impl<T: Scalar> Factorization<T> for LdltSolver<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        LdltSolver::solve(self, b)
    }
    fn factor_nnz(&self) -> usize {
        LdltSolver::factor_nnz(self)
    }
    fn n_perturbed(&self) -> usize {
        LdltSolver::n_perturbed(self)
    }
}

impl<T: Scalar> Preconditioner<T> for crate::numeric::multifrontal_lu::LuFactors<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = crate::numeric::multifrontal_lu::solve_lu(self, r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    fn solve_threads(&self) -> Threads {
        self.solve_threads
    }
    /// Block apply via `solve_lu_many` (one block triangular solve over all `s`
    /// columns). Column-major Krylov block ↔ row-major `solve_lu_many` transpose.
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        let mut rowmaj = vec![T::zero(); n * s];
        for c in 0..s {
            for i in 0..n {
                rowmaj[i * s + c] = r[c * n + i];
            }
        }
        let x = crate::numeric::multifrontal_lu::solve_lu_many(self, &rowmaj, s)?;
        for c in 0..s {
            for i in 0..n {
                z[c * n + i] = x[i * s + c];
            }
        }
        Ok(())
    }
}

impl<T: Scalar> Factorization<T> for crate::numeric::multifrontal_lu::LuFactors<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        crate::numeric::multifrontal_lu::solve_lu(self, b)
    }
    fn factor_nnz(&self) -> usize {
        crate::numeric::multifrontal_lu::LuFactors::factor_nnz(self)
    }
    fn n_perturbed(&self) -> usize {
        self.n_perturbed
    }
}

/// The high-level [`LuSolver`](crate::numeric::multifrontal_lu::LuSolver) is a
/// preconditioner / factorization too - the unsymmetric twin of the
/// [`LdltSolver`] impls, so solver-in-the-loop code can be generic over either.
impl<T: Scalar> Preconditioner<T> for crate::numeric::multifrontal_lu::LuSolver<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    fn solve_threads(&self) -> Threads {
        self.solve_thread_policy()
    }
    /// Block apply via [`LuSolver::solve_many`](crate::numeric::multifrontal_lu::LuSolver::solve_many).
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        let mut rowmaj = vec![T::zero(); n * s];
        for c in 0..s {
            for i in 0..n {
                rowmaj[i * s + c] = r[c * n + i];
            }
        }
        let x = self.solve_many(&rowmaj, s)?;
        for c in 0..s {
            for i in 0..n {
                z[c * n + i] = x[i * s + c];
            }
        }
        Ok(())
    }
}

impl<T: Scalar> Factorization<T> for crate::numeric::multifrontal_lu::LuSolver<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        crate::numeric::multifrontal_lu::LuSolver::solve(self, b)
    }
    fn factor_nnz(&self) -> usize {
        crate::numeric::multifrontal_lu::LuSolver::factor_nnz(self)
    }
    fn n_perturbed(&self) -> usize {
        crate::numeric::multifrontal_lu::LuSolver::n_perturbed(self)
    }
}

/// The KLU path composes with the iterative stack exactly like the
/// multifrontal solvers: an exact (or sweep-refactored) `M⁻¹ = (LU)⁻¹` for
/// [`gmres`]/[`gmres_block`]. Sequential by design, so [`solve_threads`]
/// pins the orthogonalization pool to one worker.
///
/// [`solve_threads`]: Preconditioner::solve_threads
impl<T: Scalar> Preconditioner<T> for crate::numeric::klu::KluSolver<T> {
    fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
        let x = self.solve(r)?;
        z.copy_from_slice(&x);
        Ok(())
    }
    fn solve_threads(&self) -> Threads {
        self.solve_thread_policy()
    }
    /// Block apply via [`KluSolver::solve_many`](crate::KluSolver::solve_many).
    fn apply_block(&self, r: &[T], z: &mut [T], s: usize, n: usize) -> Result<(), RslabError> {
        let mut rowmaj = vec![T::zero(); n * s];
        for c in 0..s {
            for i in 0..n {
                rowmaj[i * s + c] = r[c * n + i];
            }
        }
        let x = self.solve_many(&rowmaj, s)?;
        for c in 0..s {
            for i in 0..n {
                z[c * n + i] = x[i * s + c];
            }
        }
        Ok(())
    }
}

impl<T: Scalar> Factorization<T> for crate::numeric::klu::KluSolver<T> {
    fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        crate::numeric::klu::KluSolver::solve(self, b)
    }
    fn factor_nnz(&self) -> usize {
        crate::numeric::klu::KluSolver::factor_nnz(self)
    }
    /// KLU never perturbs pivots: a vanishing pivot is a hard
    /// [`RslabError::SingularBasis`] at factor time instead.
    fn n_perturbed(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numeric::multifrontal_ldlt::{SolverSettings, ZeroPivotAction};
    use num_complex::Complex;

    type C = Complex<f64>;

    /// 2D complex-symmetric Helmholtz-style grid (lower triangle).
    fn grid(m: usize, diag: C, off: C) -> CscMatrix<C> {
        let n = m * m;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                r.push(p);
                c.push(p);
                v.push(diag);
                if b + 1 < m {
                    let (hi, lo) = (idx(a, b + 1), p);
                    r.push(hi);
                    c.push(lo);
                    v.push(off);
                }
                if a + 1 < m {
                    let (hi, lo) = (idx(a + 1, b), p);
                    r.push(hi);
                    c.push(lo);
                    v.push(off);
                }
            }
        }
        CscMatrix::<C>::from_triplets(n, &r, &c, &v).unwrap()
    }

    #[test]
    fn cocg_unpreconditioned_solves_complex_symmetric() {
        let c = |re, im| Complex::new(re, im);
        let a = grid(8, c(4.0, 0.5), c(-1.0, 0.1));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let res = cocg(&a, &b, &NoPreconditioner, 1e-10, 2000).unwrap();
        assert!(res.converged, "COCG should converge, res={}", res.final_res);
        assert_eq!(res.stop, StopReason::Converged);
        // Verify against the actual residual.
        let mut ax = vec![C::default(); n];
        a.symv(&res.x, &mut ax);
        let r = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(r < 1e-7, "residual {}", r);

        // Starve the budget: the same solve capped at one iteration must report
        // `MaxIter`, not a false `Converged`.
        let capped = cocg(&a, &b, &NoPreconditioner, 1e-14, 1).unwrap();
        assert!(!capped.converged);
        assert_eq!(capped.stop, StopReason::MaxIter);
    }

    #[test]
    fn cocr_solves_complex_symmetric_pre_and_unpre() {
        let c = |re, im| Complex::new(re, im);
        let a = grid(10, c(4.0, 0.5), c(-1.0, 0.1));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

        // Unpreconditioned COCR converges to the true solution.
        let un = cocr(&a, &b, &NoPreconditioner, 1e-10, 3000).unwrap();
        assert!(un.converged, "COCR res={}", un.final_res);
        let mut ax = vec![C::default(); n];
        a.symv(&un.x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-7, "COCR residual {}", res);

        // RLA-preconditioned COCR collapses to a handful of iterations.
        let m = LdltSolver::factor(&a).unwrap();
        let pre = cocr(&a, &b, &m, 1e-10, 3000).unwrap();
        assert!(pre.converged && pre.iters <= 3, "iters {}", pre.iters);
    }

    #[test]
    fn gmres_solves_unsymmetric_with_lu_preconditioner() {
        use crate::numeric::multifrontal_lu::factor_general_lu;
        use crate::sparse::general::GeneralCsc;
        // Genuinely unsymmetric complex 2D grid (right ≠ left couplings).
        let c = |re, im| Complex::new(re, im);
        let m = 8;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(8.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

        // Unpreconditioned GMRES converges (well-conditioned).
        let un = gmres(&a, &b, &NoPreconditioner, 1e-10, 2000, 40, None).unwrap();
        assert!(un.converged, "GMRES res={}", un.final_res);

        // LU factor as preconditioner → 1-2 iterations.
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let pre = gmres(&a, &b, &lu, 1e-10, 200, 40, None).unwrap();
        assert!(pre.converged, "preconditioned GMRES res={}", pre.final_res);
        assert!(
            pre.iters <= 3,
            "LU-preconditioned GMRES iters {}",
            pre.iters
        );
        // Verify the true residual.
        let mut y = vec![C::default(); n];
        a.matvec(&pre.x, &mut y);
        let res = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-8, "residual {}", res);
    }

    #[test]
    fn gmres_singular_operator_breaks_down_without_nan() {
        // Rank-deficient operator (issue #11): A = diag(1, 1, 0) is singular and
        // `b = (1,1,1)` has a component in the null space (e₂), so GMRES cannot
        // drive the residual to zero - it stagnates. The Krylov subspace is
        // A-invariant with a *singular* restriction (eigenvalue 0), so the upper-
        // triangular Hessenberg factor acquires a ~0 diagonal. The unguarded
        // back-substitution would divide by it and emit NaN/Inf into `x` and the
        // reported residual; the guard must instead truncate to the well-
        // conditioned block, giving a deterministic breakdown: a finite iterate and
        // a truthful non-convergence report.
        use crate::sparse::general::GeneralCsc;
        let c = |re: f64, im: f64| Complex::new(re, im);
        // Only the (0,0) and (1,1) entries; row/col 2 is all-zero → A e₂ = 0.
        let a = GeneralCsc::<C>::from_triplets(3, &[0, 1], &[0, 1], &[c(1.0, 0.0), c(1.0, 0.0)])
            .unwrap();
        let b = vec![c(1.0, 0.0), c(1.0, 0.0), c(1.0, 0.0)];
        let res = gmres(&a, &b, &NoPreconditioner, 1e-12, 50, 10, None).unwrap();
        // No NaN/Inf reached the solution or the residual.
        assert!(
            res.x.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
            "solution has NaN/Inf: {:?}",
            res.x
        );
        assert!(
            res.final_res.is_finite(),
            "residual is NaN/Inf: {}",
            res.final_res
        );
        // Deterministic breakdown: reported as non-converged with a sane residual
        // (the singular direction pins the relative residual near 1/√3 ≈ 0.577 - it
        // is bounded well below the blow-up an unguarded divide would produce).
        assert!(
            !res.converged,
            "singular system must not report convergence"
        );
        assert!(
            res.final_res > 1e-12 && res.final_res <= 1.0,
            "residual not in the sane breakdown range: {}",
            res.final_res
        );
    }

    /// Build the genuinely unsymmetric complex grid used by the GMRES tests.
    fn unsym_grid(m: usize) -> crate::sparse::general::GeneralCsc<C> {
        use crate::sparse::general::GeneralCsc;
        let c = |re, im| Complex::new(re, im);
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(8.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap()
    }

    /// Wraps a preconditioner and counts every scalar `apply` (the `M⁻¹` solves).
    struct CountingPc<'a, M: ?Sized> {
        inner: &'a M,
        applies: std::sync::atomic::AtomicUsize,
    }
    impl<T: Scalar, M: Preconditioner<T> + ?Sized> Preconditioner<T> for CountingPc<'_, M> {
        fn apply(&self, r: &[T], z: &mut [T]) -> Result<(), RslabError> {
            self.applies
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.inner.apply(r, z)
        }
    }

    #[test]
    fn fgmres_saves_one_precond_apply_per_restart_cycle() {
        // FGMRES (issue #7): the preconditioned basis `Z` is kept, so the restart
        // update `x += Z y` costs **no** extra `M⁻¹` solve. The loop applies `M⁻¹`
        // exactly once per inner iteration and never at the restart, so over a
        // multi-cycle solve the total preconditioner-apply count equals the total
        // iteration count. Plain right-preconditioned GMRES would spend one extra
        // `M⁻¹` per cycle (rebuilding `M⁻¹(V y)`), i.e. `iters + n_cycles`.
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(10); // n = 100
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        // Weak (heavily incomplete) factor so the solve needs several restart
        // cycles at a short restart length - exercising the per-cycle update path.
        let opts = SolverSettings {
            drop_tol: Some(8e-1),
            ..Default::default()
        };
        let lu = factor_general_lu(&a, &opts).unwrap();
        let restart = 5;
        let counting = CountingPc {
            inner: &lu,
            applies: std::sync::atomic::AtomicUsize::new(0),
        };
        let res = gmres(&a, &b, &counting, 1e-10, 2000, restart, None).unwrap();
        assert!(res.converged, "FGMRES must converge, res={}", res.final_res);
        let applies = counting.applies.load(std::sync::atomic::Ordering::Relaxed);
        // Multiple restart cycles actually occurred (proves the saving is nonzero).
        assert!(
            res.iters > restart,
            "expected multiple cycles, iters={}",
            res.iters
        );
        // FGMRES: exactly one apply per inner iteration, none at restart.
        assert_eq!(
            applies, res.iters,
            "FGMRES precond applies {} must equal iters {} (no per-cycle extra solve)",
            applies, res.iters
        );
    }

    #[test]
    fn gmres_warm_start_cuts_total_iterations_on_related_sequence() {
        // Warm start (issue #5): a sequence of related systems `A x = b_k` with a
        // slowly rotating right-hand side. Cold-starting every solve from 0 pays
        // the full iteration count each time; seeding each solve with the previous
        // solution (which is close, because the RHS barely moved) collapses the
        // per-solve count. Total iterations must drop by a clear margin.
        let a = unsym_grid(12); // n = 144
        let n = a.n;
        let c = |re: f64, im: f64| Complex::new(re, im);
        // Two fixed directions; b_k interpolates between them by a small angle step.
        let b0: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let b1: Vec<C> = (0..n)
            .map(|i| c(((i * 3) % 7) as f64 - 3.0, ((i % 3) as f64) - 1.0))
            .collect();
        let steps = 10;
        let (tol, maxit, restart) = (1e-8, 4000, 60);

        let bk = |k: usize| -> Vec<C> {
            let th = 5e-5 * k as f64; // slow rotation
            let (ct, st) = (th.cos(), th.sin());
            (0..n)
                .map(|i| b0[i] * c(ct, 0.0) + b1[i] * c(st, 0.0))
                .collect()
        };

        // Cold: every solve from x0 = 0.
        let mut cold_total = 0usize;
        for k in 0..steps {
            let r = gmres(&a, &bk(k), &NoPreconditioner, tol, maxit, restart, None).unwrap();
            assert!(r.converged, "cold solve {k} did not converge");
            cold_total += r.iters;
        }

        // Warm: first solve from 0, each subsequent seeded with the previous x.
        let mut warm_total = 0usize;
        let mut prev: Option<Vec<C>> = None;
        for k in 0..steps {
            let r = gmres(
                &a,
                &bk(k),
                &NoPreconditioner,
                tol,
                maxit,
                restart,
                prev.as_deref(),
            )
            .unwrap();
            assert!(r.converged, "warm solve {k} did not converge");
            warm_total += r.iters;
            prev = Some(r.x);
        }

        // A meaningful reduction (well beyond noise): warm start must cut the total
        // iteration count by at least 30% over the sequence.
        assert!(
            (warm_total as f64) < 0.7 * (cold_total as f64),
            "warm start did not help enough: cold={cold_total}, warm={warm_total}"
        );
    }

    #[test]
    fn gmres_block_single_rhs_matches_scalar_gmres() {
        // s = 1 block GMRES reduces to the single-RHS path (same Arnoldi, same
        // Givens, default block apply = one single apply): same solution to the
        // requested tolerance, same iteration count up to a ±1 boundary effect.
        // The paths are NOT bit-identical - block uses CGS2, single uses MGS+DGKS,
        // so the projections sum in a different order and the true residual can
        // straddle `tol` by a rounding ULP. This is documented as a design point in
        // the module-level "Orthogonalization" note (issue #8), not a defect.
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(8);
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let single = gmres(&a, &b, &lu, 1e-10, 200, 40, None).unwrap();
        let blk = gmres_block(&a, &b, 1, &lu, 1e-10, 200, 40, None).unwrap();
        assert!(blk.converged);
        assert!(
            (blk.iters as i64 - single.iters as i64).abs() <= 1,
            "block(s=1) iters {} vs single {}",
            blk.iters,
            single.iters
        );
        let diff = (0..n)
            .map(|i| (blk.x[i] - single.x[i]).norm())
            .fold(0.0, f64::max);
        assert!(diff < 1e-7, "block(s=1) solution differs by {diff}");
    }

    #[test]
    fn gmres_block_multi_rhs_solves_each_column() {
        // Several distinct right-hand sides solved in one block iteration; every
        // column must reach its own system's true residual.
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(10);
        let n = a.n;
        let s = 5;
        // Column-major n×s block: RHS `k` is shifted/scaled so columns differ.
        let mut bblk = vec![C::default(); n * s];
        for k in 0..s {
            for i in 0..n {
                bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
            }
        }
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let res = gmres_block(&a, &bblk, s, &lu, 1e-10, 200, 40, None).unwrap();
        assert!(
            res.converged,
            "block GMRES must converge; res={:?}",
            res.final_res
        );
        // True residual per column.
        for k in 0..s {
            let mut y = vec![C::default(); n];
            a.matvec(&res.x[k * n..k * n + n], &mut y);
            let r = (0..n)
                .map(|i| (y[i] - bblk[k * n + i]).norm())
                .fold(0.0, f64::max);
            assert!(r < 1e-8, "column {k} residual {r}");
        }
        // Each column must equal the single-RHS solve of that column.
        for k in 0..s {
            let single = gmres(&a, &bblk[k * n..k * n + n], &lu, 1e-10, 200, 40, None).unwrap();
            let diff = (0..n)
                .map(|i| (res.x[k * n + i] - single.x[i]).norm())
                .fold(0.0, f64::max);
            assert!(
                diff < 1e-9,
                "column {k} differs from single solve by {diff}"
            );
        }
    }

    #[test]
    fn gmres_block_within_cycle_deflation_shrinks_applies() {
        // Different-convergence-rate regime (issue #4): a diagonal operator with
        // distinct eigenvalues, unpreconditioned, with RHS `k` supported on `k+1`
        // distinct eigenvalues. GMRES on such a RHS converges in exactly `k+1`
        // steps, so the columns finish at staggered steps within a *single* cycle.
        // The fix must (i) still solve every column exactly like single-RHS GMRES
        // and (ii) actually shrink the batched operator applies as columns deflate
        // mid-cycle - not only at restart. A counting operator records the width
        // of every `apply_block`, and we assert the panel narrows.
        use std::sync::Mutex;

        struct CountingOp<'a> {
            inner: &'a GeneralCsc<C>,
            widths: Mutex<Vec<usize>>,
        }
        impl LinearOperator<C> for CountingOp<'_> {
            fn n(&self) -> usize {
                self.inner.n()
            }
            fn apply(&self, x: &[C], y: &mut [C]) {
                self.widths.lock().unwrap().push(1);
                self.inner.apply(x, y);
            }
            fn apply_block(&self, x: &[C], y: &mut [C], s: usize) {
                self.widths.lock().unwrap().push(s);
                self.inner.apply_block(x, y, s);
            }
        }

        let c = |re: f64, im: f64| Complex::new(re, im);
        let n = 8;
        let s = 4;
        // Diagonal operator with distinct entries → the minimal polynomial degree
        // of a RHS equals the number of distinct diagonal entries in its support.
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            rr.push(i);
            cc.push(i);
            vv.push(c(2.0 + i as f64, 0.5 + 0.1 * i as f64));
        }
        let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
        // RHS `k` = sum of the first `k+1` unit vectors → converges in `k+1` steps.
        let mut bblk = vec![C::default(); n * s];
        for k in 0..s {
            for i in 0..=k {
                bblk[k * n + i] = c(1.0, 0.0);
            }
        }

        let op = CountingOp {
            inner: &a,
            widths: Mutex::new(Vec::new()),
        };
        let res = gmres_block(&op, &bblk, s, &NoPreconditioner, 1e-12, 200, 40, None).unwrap();
        assert!(
            res.converged,
            "block GMRES must converge; res={:?}",
            res.final_res
        );

        // (i) Every column matches the single-RHS GMRES solve of that column.
        for k in 0..s {
            let single = gmres(
                &a,
                &bblk[k * n..k * n + n],
                &NoPreconditioner,
                1e-12,
                200,
                40,
                None,
            )
            .unwrap();
            let diff = (0..n)
                .map(|i| (res.x[k * n + i] - single.x[i]).norm())
                .fold(0.0, f64::max);
            assert!(
                diff < 1e-9,
                "column {k} differs from single solve by {diff}"
            );
        }

        // (ii) The batched applies actually shrank mid-cycle. Without within-cycle
        // deflation every `apply_block` would run at full width `s`; with it, later
        // steps run narrower. Assert a full-width apply happened (the first step),
        // that some apply narrowed below `s`, that the panel drained to width 1,
        // and that the total column-applies fell below the full-width bound.
        let widths = op.widths.into_inner().unwrap();
        let ncalls = widths.len();
        let total_cols: usize = widths.iter().sum();
        assert_eq!(
            *widths.iter().max().unwrap(),
            s,
            "the first cycle must open at full width"
        );
        assert!(
            *widths.iter().min().unwrap() < s,
            "no apply narrowed: deflation did not shrink the panel"
        );
        assert!(
            widths.contains(&1),
            "the panel must drain to a single active column"
        );
        assert!(
            total_cols < s * ncalls,
            "total column-applies {total_cols} not below the full-width bound {}",
            s * ncalls
        );
    }

    #[test]
    fn gmres_block_bcgs2_bit_identical_across_thread_counts() {
        // The block-CGS2 orthogonalization reduces over fixed row-chunks folded in
        // chunk order, so the whole block solve is **bit-identical regardless of
        // the thread count** - the determinism guarantee. Solve the same block in
        // a 1-thread and an 8-thread rayon pool and require exact equality.
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        // Wide enough that a chunked reduction actually spans several chunks.
        let a = unsym_grid(60);
        let n = a.n;
        let s = 5;
        let mut bblk = vec![C::default(); n * s];
        for k in 0..s {
            for i in 0..n {
                bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
            }
        }
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let solve = || gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap();
        let x1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(solve);
        let x8 = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap()
            .install(solve);
        assert_eq!(
            x1.iters, x8.iters,
            "iteration count must not depend on threads"
        );
        assert!(
            x1.x == x8.x,
            "block solve must be bit-identical across thread counts"
        );
    }

    #[test]
    fn with_threads_caps_block_gmres_pool_and_keeps_result() {
        // `with_threads(p)` runs the block solve in a scoped pool of exactly `p`
        // workers (the embedded / solver-in-the-loop cap) and produces the same
        // result as the unbounded solve.
        use crate::numeric::multifrontal_ldlt::with_threads;
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(30);
        let n = a.n;
        let s = 4;
        let mut bblk = vec![C::default(); n * s];
        for k in 0..s {
            for i in 0..n {
                bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
            }
        }
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let seen = with_threads(3, rayon::current_num_threads);
        assert_eq!(
            seen, 3,
            "with_threads must cap the pool to the requested width"
        );
        let capped = with_threads(3, || {
            gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap()
        });
        let plain = gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap();
        assert!(capped.converged);
        assert!(
            capped.x == plain.x,
            "capped-pool solve must match the unbounded solve bit-for-bit"
        );
    }

    #[test]
    fn block_gmres_orthogonalization_respects_factor_thread_cap() {
        // Issue #9: the block-GMRES orthogonalization must run in a pool derived
        // from the *factor's* Threads policy, not the ambient global pool. Factor
        // with a hard cap of 2 workers; the factor then reports `Fixed(2)` as its
        // solve-phase policy, and the pool built from it caps `current_num_threads`
        // to 2 - even when the surrounding (ambient) pool is far wider. The solve
        // stays bit-identical whether run bare or inside a wide ambient pool (the
        // chunk-order reduction is thread-count independent), so the cap changes
        // only the concurrency, never the numbers.
        use crate::numeric::multifrontal_ldlt::{with_threads, Threads};
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(30);
        let n = a.n;
        let s = 4;
        let mut bblk = vec![C::default(); n * s];
        for k in 0..s {
            for i in 0..n {
                bblk[k * n + i] = c(((i + k) % 7) as f64 - 3.0, ((i + 2 * k) % 5) as f64 - 2.0);
            }
        }
        // Factor capped to exactly 2 workers → solve policy is Fixed(2).
        let lu = factor_general_lu(&a, &SolverSettings::default().with_threads(2)).unwrap();
        assert_eq!(
            Preconditioner::<C>::solve_threads(&lu),
            Threads::Fixed(2),
            "factor must carry its resolved solve-phase thread budget"
        );
        // The pool the orthogonalization installs caps the worker count to 2,
        // regardless of a wider ambient pool around it.
        let pool = solve_thread_pool(Preconditioner::<C>::solve_threads(&lu));
        let seen = with_threads(8, || {
            pool.as_ref().unwrap().install(rayon::current_num_threads)
        });
        assert_eq!(
            seen, 2,
            "the ortho pool must cap to the factor's 2-worker budget"
        );

        // The full solve is identical bare vs. inside a wide ambient pool: the
        // internal cap governs concurrency only, never the result.
        let bare = gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap();
        let in_wide = with_threads(8, || {
            gmres_block(&a, &bblk, s, &lu, 1e-10, 300, 60, None).unwrap()
        });
        assert!(bare.converged);
        assert!(
            bare.x == in_wide.x && bare.iters == in_wide.iters,
            "capped ortho pool must not perturb the numeric result"
        );
    }

    #[test]
    fn ambient_threads_factor_matches_default_and_runs_on_shared_pool() {
        // `Threads::Ambient` factors on the current pool (no new spawn) - the
        // re-factor-in-loop path. Inside a `with_threads(2)` pool the factor must be
        // bit-identical to the normal (scoped-pool) factor: the numeric result is
        // independent of the thread policy.
        use crate::numeric::multifrontal_ldlt::{with_threads, Threads};
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(24);
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let lu_default = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let opts_amb = SolverSettings::default().with_thread_policy(Threads::Ambient);
        let lu_amb = with_threads(2, || {
            assert_eq!(rayon::current_num_threads(), 2);
            factor_general_lu(&a, &opts_amb).unwrap()
        });
        let x_def = gmres_block(&a, &b, 1, &lu_default, 1e-10, 200, 40, None).unwrap();
        let x_amb = gmres_block(&a, &b, 1, &lu_amb, 1e-10, 200, 40, None).unwrap();
        assert!(
            x_def.x == x_amb.x,
            "ambient-pool factor must be bit-identical to the default factor"
        );
    }

    #[test]
    fn default_thread_policy_caps_at_four() {
        // The pareto-optimal embedded default: predict per matrix, never exceed 4.
        use crate::numeric::multifrontal_ldlt::Threads;
        assert_eq!(SolverSettings::default().threads, Threads::Auto { max: 4 });
    }

    #[test]
    fn f32_lu_preconditioner_keeps_f64_accuracy_in_gmres() {
        use crate::sparse::general::GeneralCsc;
        let c = |re, im| Complex::new(re, im);
        let m = 10;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(20.0, 2.0)); // strongly diagonally dominant → f32 factor is accurate
                if b + 1 < m {
                    rr.push(p);
                    cc.push(idx(a, b + 1));
                    vv.push(c(-1.0, 0.2));
                    rr.push(idx(a, b + 1));
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    rr.push(p);
                    cc.push(idx(a + 1, b));
                    vv.push(c(-1.5, 0.3));
                    rr.push(idx(a + 1, b));
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();
        // The f32 LU factor is a half-memory preconditioner accurate to ~1e-6
        // (the f32 apply floor). f64 GMRES reaches that floor in a couple of
        // iterations - ample for a MoM/FEM Krylov tolerance, at half the factor
        // memory. (A residual below ~1e-6 is not attainable with an f32 apply;
        // use the f64 factor for tighter tolerances.)
        let pc = LowPrecisionLu::factor(&a, &SolverSettings::default()).unwrap();
        assert!(pc.factor_nnz() > 0);
        let res = gmres(&a, &b, &pc, 1e-6, 200, 50, None).unwrap();
        assert!(res.converged, "mixed-precision GMRES res={}", res.final_res);
        assert!(res.iters <= 6, "iters {}", res.iters);
        let mut y = vec![C::default(); n];
        a.matvec(&res.x, &mut y);
        let r = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(r < 1e-5, "residual {}", r);
    }

    #[test]
    fn cocr_handles_indefinite_helmholtz() {
        // Indefinite complex-symmetric: high-frequency 2D Helmholtz with a
        // negative-real diagonal shift (`diag = -1 + 0.3i`). A robust
        // preconditioner (static-pivoted RLA) plus COCR must still converge.
        let c = |re, im| Complex::new(re, im);
        let a = grid(10, c(-1.0, 0.3), c(1.0, 0.05));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c(1.0, (i % 3) as f64 - 1.0)).collect();
        let opts = SolverSettings {
            on_zero_pivot: ZeroPivotAction::PerturbToEps { abs_floor: 1e-10 },
            drop_tol: None,
            ..Default::default()
        };
        let m = LdltSolver::factor_with(&a, &opts).unwrap();
        let pre = cocr(&a, &b, &m, 1e-9, 500).unwrap();
        assert!(
            pre.converged,
            "indefinite COCR res={} iters={}",
            pre.final_res, pre.iters
        );
        let mut ax = vec![C::default(); n];
        a.symv(&pre.x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-6, "residual {}", res);
    }

    #[test]
    fn incomplete_factor_reduces_fill_and_still_preconditions() {
        // Threshold dropping shrinks the factor (less memory) at the cost of a
        // weaker preconditioner - but COCG must still converge to the true
        // f64 solution. Demonstrates the memory ↔ iteration tradeoff.
        let c = |re, im| Complex::new(re, im);
        let a = grid(16, c(4.0, 1.0), c(-1.0, 0.2));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

        let full = LdltSolver::factor(&a).unwrap();
        let opts = SolverSettings {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: Some(5e-2),
            ..Default::default()
        };
        let inc = LdltSolver::factor_with(&a, &opts).unwrap();

        assert!(
            inc.factor_nnz() < full.factor_nnz(),
            "dropping should reduce fill: incomplete {} vs complete {}",
            inc.factor_nnz(),
            full.factor_nnz()
        );

        let rf = cocg(&a, &b, &full, 1e-10, 1000).unwrap();
        let ri = cocg(&a, &b, &inc, 1e-10, 1000).unwrap();
        assert!(ri.converged, "incomplete-preconditioned COCG must converge");
        assert!(
            ri.iters >= rf.iters,
            "incomplete factor should need ≥ complete-factor iterations"
        );
        let mut ax = vec![C::default(); n];
        a.symv(&ri.x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-8, "residual {}", res);
    }

    #[test]
    fn f32_preconditioner_keeps_f64_accuracy() {
        // Factor the preconditioner in Complex<f32> (half memory) but iterate in
        // f64: the solution must still reach f64-level residual, and the f32
        // factor - though approximate - keeps the iteration count tiny.
        let c = |re, im| Complex::new(re, im);
        let a = grid(14, c(4.0, 1.0), c(-1.0, 0.2));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

        let m = LowPrecisionPreconditioner::factor(&a, &SolverSettings::default()).unwrap();
        let res = cocg(&a, &b, &m, 1e-10, 500).unwrap();
        assert!(res.converged, "mixed-precision COCG res={}", res.final_res);
        // A few iterations suffice; the f32 factor is a strong preconditioner.
        assert!(res.iters <= 12, "f32-preconditioned iters {}", res.iters);
        // Full f64 accuracy recovered despite the single-precision factor.
        let mut ax = vec![C::default(); n];
        a.symv(&res.x, &mut ax);
        let r = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(r < 1e-8, "mixed-precision residual {}", r);
        assert!(m.factor_nnz() > 0);
    }

    #[test]
    fn rla_preconditioner_collapses_iteration_count() {
        // A complete RLA factorization is ≈ A⁻¹, so preconditioned COCG must
        // converge in a handful of iterations - vastly fewer than without.
        let c = |re, im| Complex::new(re, im);
        let a = grid(12, c(4.0, 1.0), c(-1.0, 0.2));
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 7) as f64 - 3.0, 0.5)).collect();

        let unpre = cocg(&a, &b, &NoPreconditioner, 1e-10, 5000).unwrap();
        let m = LdltSolver::factor(&a).unwrap();
        let pre = cocg(&a, &b, &m, 1e-10, 5000).unwrap();

        assert!(pre.converged && unpre.converged);
        assert!(
            pre.iters <= 3,
            "complete-factor preconditioner should need ≤3 iters, got {}",
            pre.iters
        );
        assert!(
            pre.iters * 5 < unpre.iters,
            "preconditioner should cut iterations sharply: {} vs {}",
            pre.iters,
            unpre.iters
        );
    }

    /// Strongly **non-normal** 1D convection-diffusion operator as a general
    /// complex matrix: tridiagonal `diag = 2 (+ tiny damping)`, super `= -1+γ`,
    /// sub `= -1-γ`. For `γ ≠ 0` it is far from normal - the classic GMRES-hard
    /// regime where the residual stagnates for many steps before converging - yet
    /// remains (weakly) diagonally dominant, so unpreconditioned GMRES does
    /// converge, only after many iterations / restart cycles.
    fn convection_diffusion(n: usize, gamma: f64) -> crate::sparse::general::GeneralCsc<C> {
        use crate::sparse::general::GeneralCsc;
        let c = |re: f64, im: f64| Complex::new(re, im);
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            rr.push(i);
            cc.push(i);
            vv.push(c(2.0, 0.02));
            if i + 1 < n {
                rr.push(i);
                cc.push(i + 1);
                vv.push(c(-1.0 + gamma, 0.0));
                rr.push(i + 1);
                cc.push(i);
                vv.push(c(-1.0 - gamma, 0.0));
            }
        }
        GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap()
    }

    /// Wraps an operator and records the width `s` of every block apply - the
    /// deflation probe (issue #4 / #13): a shrinking width proves the batched
    /// applies narrow as columns converge.
    struct WidthCountingOp<'a> {
        inner: &'a GeneralCsc<C>,
        widths: std::sync::Mutex<Vec<usize>>,
    }
    impl LinearOperator<C> for WidthCountingOp<'_> {
        fn n(&self) -> usize {
            self.inner.n()
        }
        fn apply(&self, x: &[C], y: &mut [C]) {
            self.widths.lock().unwrap().push(1);
            self.inner.apply(x, y);
        }
        fn apply_block(&self, x: &[C], y: &mut [C], s: usize) {
            self.widths.lock().unwrap().push(s);
            self.inner.apply_block(x, y, s);
        }
    }

    #[test]
    fn gmres_unpreconditioned_nonnormal_needs_many_restarts() {
        // (issue #13a) Unpreconditioned GMRES on a strongly non-normal operator:
        // it must survive the non-normal stagnation phase and multiple restart
        // cycles, then converge to the true solution. Exercises the restart /
        // outer-loop machinery that the diagonally dominant tests never stress.
        let a = convection_diffusion(120, 0.9);
        let n = a.n;
        let c = |re: f64, im: f64| Complex::new(re, im);
        let b: Vec<C> = (0..n).map(|i| c(((i % 5) as f64) - 2.0, 0.5)).collect();
        let restart = 20;
        let res = gmres(&a, &b, &NoPreconditioner, 1e-8, 8000, restart, None).unwrap();
        assert!(
            res.converged,
            "non-normal GMRES must converge, res={}",
            res.final_res
        );
        assert!(
            res.iters > restart,
            "must span multiple restart cycles, iters={} (restart={})",
            res.iters,
            restart
        );
        let mut y = vec![C::default(); n];
        a.matvec(&res.x, &mut y);
        let r = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(r < 1e-6, "true residual {}", r);
    }

    #[test]
    fn gmres_reorthogonalization_keeps_illconditioned_arnoldi_accurate() {
        // (issue #13c) A near-defective, strongly non-normal operator (bidiagonal
        // Jordan-like block: clustered diagonal, dominant super-diagonal) drives
        // the Arnoldi vectors toward linear dependence, so a single MGS sweep
        // collapses the norm and the conditional DGKS second pass (`hn < η·‖w₀‖`)
        // must fire to restore orthogonality. Asserting the trigger directly needs
        // an intrusive probe; instead we certify the *effect*: GMRES still drives
        // the true residual to `tol` and matches the exact (direct-LU) solution -
        // which it could not if the ill-conditioned basis went uncorrected.
        use crate::numeric::multifrontal_lu::{factor_general_lu, solve_lu};
        use crate::sparse::general::GeneralCsc;
        let c = |re: f64, im: f64| Complex::new(re, im);
        let n = 32;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            rr.push(i);
            cc.push(i);
            vv.push(c(2.0, 0.0)); // clustered diagonal → non-normal, near-defective
            if i + 1 < n {
                rr.push(i);
                cc.push(i + 1);
                vv.push(c(3.0, 0.0)); // dominant super-diagonal
            }
        }
        let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<C> = (0..n).map(|_| c(1.0, 0.2)).collect();
        // Long restart (single cycle) so the ill-conditioned basis is not masked by
        // a restart - the reorthogonalization alone keeps it usable.
        let res = gmres(&a, &b, &NoPreconditioner, 1e-10, 4000, n, None).unwrap();
        assert!(
            res.converged,
            "reorth must keep GMRES converging, res={}",
            res.final_res
        );
        let lu = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let xstar = solve_lu(&lu, &b).unwrap();
        let diff = (0..n)
            .map(|i| (res.x[i] - xstar[i]).norm())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-6,
            "GMRES solution off the direct solve by {} (lost orthogonality?)",
            diff
        );
    }

    #[test]
    fn gmres_happy_breakdown_on_eigenvector_rhs() {
        // (issue #13d) Happy breakdown: `b` is an eigenvector of the operator, so
        // the Krylov space `K_1 = span{b}` is already `A`-invariant. The Arnoldi
        // step-1 subdiagonal `h[1][0]` is exactly `0` (the invariant-subspace
        // branch), and GMRES must produce the exact solution in a single iteration.
        use crate::sparse::general::GeneralCsc;
        let c = |re: f64, im: f64| Complex::new(re, im);
        let n = 12;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            rr.push(i);
            cc.push(i);
            vv.push(c(2.0 + i as f64, 0.5 - 0.05 * i as f64)); // distinct diagonal
        }
        let a = GeneralCsc::<C>::from_triplets(n, &rr, &cc, &vv).unwrap();
        // `e_0` is an eigenvector (eigenvalue `a[0][0]`); its Krylov space is 1-D.
        let mut b = vec![C::default(); n];
        b[0] = c(1.0, 0.0);
        let res = gmres(&a, &b, &NoPreconditioner, 1e-12, 50, 30, None).unwrap();
        assert!(
            res.converged,
            "eigenvector RHS must converge, res={}",
            res.final_res
        );
        assert_eq!(
            res.iters, 1,
            "happy breakdown must solve in one step, got {}",
            res.iters
        );
        let mut y = vec![C::default(); n];
        a.matvec(&res.x, &mut y);
        let r = (0..n).map(|i| (y[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(r < 1e-12, "true residual {}", r);
    }

    #[test]
    fn gmres_block_incomplete_factor_multirate_deflation() {
        // (issue #13b) Multi-rate within-cycle deflation under a genuine
        // **factor-based** (drop-tol) preconditioner. The operator is diagonal with
        // distinct entries `d_i`; the preconditioner is a `drop_tol` LU factor of a
        // *different* diagonal matrix `diag(p_i)` - a deliberately imperfect
        // approximate inverse, so the preconditioned operator `M⁻¹A = diag(d_i/p_i)`
        // still has distinct eigenvalues. Right-hand side `k` is supported on the
        // first `k+1` unit vectors, so its GMRES converges in **exactly** `k+1`
        // steps: the columns finish at staggered steps *within one cycle*. The
        // within-cycle deflation (#4) must finalize each fast column and shrink the
        // batched applies to the still-active width, draining the panel to 1 - while
        // every column still matches its single-RHS solve.
        use crate::numeric::multifrontal_lu::factor_general_lu;
        use crate::sparse::general::GeneralCsc;
        let c = |re: f64, im: f64| Complex::new(re, im);
        let n = 8;
        let s = 4;
        // Operator D = diag(d_i), d_i distinct.
        let (mut dr, mut dc, mut dv) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            dr.push(i);
            dc.push(i);
            dv.push(c(2.0 + i as f64, 0.3));
        }
        let a = GeneralCsc::<C>::from_triplets(n, &dr, &dc, &dv).unwrap();
        // Preconditioning matrix P = diag(p_i), p_i chosen so d_i/p_i stay distinct
        // (p_i = 1+i ⇒ ratios 2, 1.5, 1.33, … all different): an imperfect factor.
        let (mut pr, mut pc_, mut pv) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            pr.push(i);
            pc_.push(i);
            pv.push(c(1.0 + i as f64, 0.1));
        }
        let pmat = GeneralCsc::<C>::from_triplets(n, &pr, &pc_, &pv).unwrap();
        // drop-tol factor path (imperfect preconditioner)
        let opts = SolverSettings {
            drop_tol: Some(1e-2),
            ..Default::default()
        };
        let lu = factor_general_lu(&pmat, &opts).unwrap();

        // RHS k = sum of the first k+1 unit vectors → converges in exactly k+1 steps.
        let mut bblk = vec![C::default(); n * s];
        for k in 0..s {
            for i in 0..=k {
                bblk[k * n + i] = c(1.0, 0.0);
            }
        }

        let op = WidthCountingOp {
            inner: &a,
            widths: std::sync::Mutex::new(Vec::new()),
        };
        let res = gmres_block(&op, &bblk, s, &lu, 1e-10, 200, 40, None).unwrap();
        assert!(
            res.converged,
            "block GMRES must converge; res={:?}",
            res.final_res
        );

        // Every column equals its single-RHS solve (deflation must not corrupt it).
        for k in 0..s {
            let single = gmres(&a, &bblk[k * n..k * n + n], &lu, 1e-10, 200, 40, None).unwrap();
            let diff = (0..n)
                .map(|i| (res.x[k * n + i] - single.x[i]).norm())
                .fold(0.0, f64::max);
            assert!(
                diff < 1e-8,
                "column {k} differs from single solve by {diff}"
            );
        }

        // Within-cycle deflation fired: the panel opened at full width `s`, narrowed
        // as fast columns deflated, and drained to a single active column.
        let widths = op.widths.into_inner().unwrap();
        assert!(!widths.is_empty());
        assert_eq!(
            *widths.iter().max().unwrap(),
            s,
            "the first cycle must open at full width"
        );
        assert!(
            *widths.iter().min().unwrap() < s,
            "within-cycle deflation did not shrink the panel: widths={widths:?}"
        );
        assert!(
            widths.contains(&1),
            "the panel must drain to a single active column: {widths:?}"
        );
    }

    /// A general complex diagonal operator with a prescribed spectrum - the
    /// canonical GMRES-DR / GCRO-DR demonstrator: a handful of tiny eigenvalues
    /// (which restarted GMRES keeps re-discovering and discarding) sitting far
    /// below a cluster of larger ones. Unpreconditioned restarted GMRES stagnates
    /// on the small cluster; deflating it is exactly what recycling does.
    fn diag_op(eigs: &[C]) -> crate::sparse::general::GeneralCsc<C> {
        use crate::sparse::general::GeneralCsc;
        let n = eigs.len();
        let idx: Vec<usize> = (0..n).collect();
        GeneralCsc::<C>::from_triplets(n, &idx, &idx, eigs).unwrap()
    }

    /// A spectrum with `n_small` tiny eigenvalues far below a spread cluster.
    fn stagnation_spectrum(n: usize, n_small: usize) -> Vec<C> {
        let c = |re: f64, im: f64| Complex::new(re, im);
        (0..n)
            .map(|i| {
                if i < n_small {
                    // Tiny, tightly clustered near the origin - the stagnation drivers.
                    c(0.01 + 0.004 * i as f64, 0.002 * i as f64)
                } else {
                    // Larger eigenvalues spread over [1, 11], mildly complex.
                    let t = (i - n_small) as f64 / (n - n_small) as f64;
                    c(1.0 + 10.0 * t, 0.3 * (i as f64).sin())
                }
            })
            .collect()
    }

    #[test]
    fn gmres_recycled_matches_plain_on_hard_matrix() {
        // Correctness: the recycled solve must reach the SAME solution as plain
        // FGMRES on a hard preconditioned system (weak incomplete LU factor, short
        // restart → many cycles), to the same tolerance.
        use crate::numeric::multifrontal_lu::factor_general_lu;
        let c = |re, im| Complex::new(re, im);
        let a = unsym_grid(12); // n = 144
        let n = a.n;
        let b: Vec<C> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let opts = SolverSettings {
            drop_tol: Some(8e-1),
            ..Default::default()
        };
        let lu = factor_general_lu(&a, &opts).unwrap();
        let (tol, maxit, restart) = (1e-10, 4000, 12);
        let plain = gmres(&a, &b, &lu, tol, maxit, restart, None).unwrap();
        let mut rec = Recycle::new(8);
        let recd = gmres_recycled(&a, &b, &lu, tol, maxit, restart, None, &mut rec).unwrap();
        assert!(plain.converged, "plain FGMRES did not converge");
        assert!(recd.converged, "recycled did not converge");
        let diff = (0..n)
            .map(|i| (plain.x[i] - recd.x[i]).norm())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-7,
            "recycled solution differs from plain by {diff}"
        );
        // The recycle handle came back populated for the next solve.
        assert!(rec.active() > 0, "recycle subspace was not populated");
    }

    #[test]
    fn gmres_recycled_within_solve_reduces_restarts() {
        // Within-solve deflated restarting: on a stagnating, restart-limited solve
        // (tiny eigenvalue cluster, unpreconditioned, short restart) carrying the
        // harmonic-Ritz subspace across restarts must cut total iterations by a
        // real margin versus plain FGMRES - on a SINGLE solve (fresh handle, no
        // cross-solve benefit).
        let eigs = stagnation_spectrum(40, 4);
        let a = diag_op(&eigs);
        let n = a.n;
        let c = |re: f64, im: f64| Complex::new(re, im);
        let b: Vec<C> = (0..n).map(|i| c(1.0, 0.2 * (i as f64).cos())).collect();
        let (tol, maxit, restart) = (1e-9, 5000, 10);

        let plain = gmres(&a, &b, &NoPreconditioner, tol, maxit, restart, None).unwrap();
        let mut rec = Recycle::new(6);
        let recd = gmres_recycled(
            &a,
            &b,
            &NoPreconditioner,
            tol,
            maxit,
            restart,
            None,
            &mut rec,
        )
        .unwrap();
        assert!(plain.converged, "plain did not converge ({})", plain.iters);
        assert!(recd.converged, "recycled did not converge ({})", recd.iters);
        // Both hit the same true solution.
        let diff = (0..n)
            .map(|i| (plain.x[i] - recd.x[i]).norm())
            .fold(0.0, f64::max);
        assert!(diff < 1e-6, "solutions differ by {diff}");
        // Deflated restarting must shave a clear margin off the iteration count.
        eprintln!(
            "[within-solve] plain FGMRES(10)={} iters, GCRO-DR(k=6)={} iters",
            plain.iters, recd.iters
        );
        assert!(
            (recd.iters as f64) < 0.75 * (plain.iters as f64),
            "within-solve deflation did not help enough: plain={}, recycled={}",
            plain.iters,
            recd.iters
        );
    }

    #[test]
    fn gmres_recycled_cross_solve_beats_warm_and_cold() {
        // Cross-solve recycling on a sequence of related systems: A gets a small
        // diagonal perturbation each step and b rotates slowly. Compare total
        // iterations over the sequence for cold (x0=0), warm (x0=prev x), and
        // recycled+warm (handle carried across solves). Must order
        // recycled < warm < cold, each by a meaningful margin.
        let base = stagnation_spectrum(48, 5);
        let n = base.len();
        let c = |re: f64, im: f64| Complex::new(re, im);
        let b0: Vec<C> = (0..n).map(|i| c(1.0, 0.15 * (i as f64).cos())).collect();
        let b1: Vec<C> = (0..n).map(|i| c(0.4 * (i as f64).sin(), 1.0)).collect();
        let steps = 8;
        let (tol, maxit, restart) = (1e-9, 6000, 12);

        // Slowly varying operator: base spectrum + eps_k on each diagonal entry.
        let ak = |kk: usize| -> crate::sparse::general::GeneralCsc<C> {
            let eps = 2e-3 * kk as f64;
            let eigs: Vec<C> = base
                .iter()
                .enumerate()
                .map(|(i, &e)| e + c(eps * (1.0 + 0.05 * i as f64), 0.0))
                .collect();
            diag_op(&eigs)
        };
        let bk = |kk: usize| -> Vec<C> {
            let th = 0.02 * kk as f64;
            let (ct, st) = (th.cos(), th.sin());
            (0..n)
                .map(|i| b0[i] * c(ct, 0.0) + b1[i] * c(st, 0.0))
                .collect()
        };

        // Cold.
        let mut cold = 0usize;
        for kk in 0..steps {
            let a = ak(kk);
            let r = gmres(&a, &bk(kk), &NoPreconditioner, tol, maxit, restart, None).unwrap();
            assert!(r.converged, "cold {kk} stalled");
            cold += r.iters;
        }
        // Warm.
        let mut warm = 0usize;
        let mut prev: Option<Vec<C>> = None;
        for kk in 0..steps {
            let a = ak(kk);
            let r = gmres(
                &a,
                &bk(kk),
                &NoPreconditioner,
                tol,
                maxit,
                restart,
                prev.as_deref(),
            )
            .unwrap();
            assert!(r.converged, "warm {kk} stalled");
            warm += r.iters;
            prev = Some(r.x);
        }
        // Recycled (+ warm start, the intended combined use).
        let mut recycled = 0usize;
        let mut rec = Recycle::new(8);
        let mut prevr: Option<Vec<C>> = None;
        for kk in 0..steps {
            let a = ak(kk);
            let r = gmres_recycled(
                &a,
                &bk(kk),
                &NoPreconditioner,
                tol,
                maxit,
                restart,
                prevr.as_deref(),
                &mut rec,
            )
            .unwrap();
            assert!(r.converged, "recycled {kk} stalled");
            recycled += r.iters;
            prevr = Some(r.x);
        }

        eprintln!(
            "[cross-solve] cold={cold}, warm={warm}, recycled={recycled} (total iters over {steps} solves)"
        );
        assert!(warm < cold, "warm ({warm}) not below cold ({cold})");
        assert!(
            recycled < warm,
            "recycled ({recycled}) not below warm ({warm})"
        );
        // Meaningful reduction, not noise.
        assert!(
            (recycled as f64) < 0.7 * (cold as f64),
            "recycled ({recycled}) did not beat cold ({cold}) by a clear margin"
        );
    }

    #[test]
    fn gmres_recycled_composes_with_warm_start() {
        // Recycle + x0 warm start together: on a related second solve, seeding from
        // the previous solution AND recycling its stagnation subspace must converge
        // to the correct solution and take no more iterations than warm-start alone.
        let eigs = stagnation_spectrum(40, 4);
        let a = diag_op(&eigs);
        let n = a.n;
        let c = |re: f64, im: f64| Complex::new(re, im);
        let b0: Vec<C> = (0..n).map(|i| c(1.0, 0.1 * (i as f64).cos())).collect();
        let b1: Vec<C> = (0..n).map(|i| c(1.0 + 0.02 * i as f64, 0.1)).collect();
        let (tol, maxit, restart) = (1e-9, 5000, 10);

        // First solve seeds both the warm start and the recycle handle.
        let mut rec = Recycle::new(6);
        let first = gmres_recycled(
            &a,
            &b0,
            &NoPreconditioner,
            tol,
            maxit,
            restart,
            None,
            &mut rec,
        )
        .unwrap();
        assert!(first.converged);

        // Second, related solve: warm-only vs warm+recycle.
        let warm_only = gmres(
            &a,
            &b1,
            &NoPreconditioner,
            tol,
            maxit,
            restart,
            Some(&first.x),
        )
        .unwrap();
        let warm_rec = gmres_recycled(
            &a,
            &b1,
            &NoPreconditioner,
            tol,
            maxit,
            restart,
            Some(&first.x),
            &mut rec,
        )
        .unwrap();
        assert!(warm_only.converged && warm_rec.converged);
        // Same true solution.
        let mut ax = vec![C::default(); n];
        a.matvec(&warm_rec.x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b1[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-6, "warm+recycle residual {res}");
        // Composed use is no worse than warm-start alone (typically better).
        assert!(
            warm_rec.iters <= warm_only.iters,
            "warm+recycle ({}) worse than warm-only ({})",
            warm_rec.iters,
            warm_only.iters
        );
    }

    #[test]
    fn gmres_recycled_real_scalar_path() {
        // The real f64 field exercises the conjugate-pair reconstruction in
        // `combine_ritz` (a real diagonal has real eigenvalues, but a real
        // unsymmetric grid produces complex harmonic-Ritz pairs). Must converge to
        // the true solution - correctness of the real recycle path.
        use crate::sparse::general::GeneralCsc;
        let m = 8;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(4.0f64);
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(-1.0);
                    rr.push(q);
                    cc.push(p);
                    vv.push(-1.8); // asymmetric ⇒ complex spectrum
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(-0.7);
                    rr.push(q);
                    cc.push(p);
                    vv.push(-1.3);
                }
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let mut rec = Recycle::<f64>::new(6);
        let r = gmres_recycled(&a, &b, &NoPreconditioner, 1e-9, 5000, 12, None, &mut rec).unwrap();
        assert!(r.converged, "real recycled solve did not converge");
        let mut ax = vec![0.0f64; n];
        a.matvec(&r.x, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).abs()).fold(0.0, f64::max);
        assert!(res < 1e-6, "real recycled residual {res}");
    }
}
