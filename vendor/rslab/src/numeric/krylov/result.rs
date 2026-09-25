//! Outcome of a Krylov solve.

/// Why a Krylov iteration stopped (audit finding L5). Additive diagnostic
/// alongside the boolean `converged`: `Converged` is exactly `converged == true`,
/// and the two failure states distinguish a hit iteration budget (`MaxIter`)
/// from an algebraic breakdown of the short-recurrence denominator (`Breakdown`).
///
/// Only the states the solvers genuinely distinguish are represented:
/// * COCG / COCR report all three - a zero bilinear denominator (`p^TAp` or `r^Tz`
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
    /// caller's per-cycle callback returned `false`, typically a stagnation
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
pub(super) fn stop_reason(converged: bool, iters: usize, max_iter: usize) -> StopReason {
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
    /// `true` if `||b - Ax|| / ||b|| <= tol` was reached within `max_iter`.
    pub converged: bool,
    /// Final relative residual `||b - Ax|| / ||b||`.
    pub final_res: f64,
    /// Why the iteration stopped (`converged` is `stop == Converged`).
    pub stop: StopReason,
}
