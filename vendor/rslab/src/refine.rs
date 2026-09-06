//! One iterative-refinement contract for every direct path.
//!
//! Refinement exists on all three paths, and what "converged" means is a
//! property of the caller's problem, not of the factorization: a normwise
//! criterion accepts a solution whose small components carry large relative
//! error, which is exactly what a badly scaled row produces. The policy below
//! therefore names the measure, the target and the step budget in one place,
//! and every entry point reports what it achieved instead of only how many
//! steps it took.
//!
//! The two measures, with `r = b - Ax`:
//!
//! * normwise, `omega = ||r||_inf / (||A||_inf ||x||_inf + ||b||_inf)`, the
//!   backward error of the perturbed system `(A + dA) x = b + db` with
//!   `||dA|| <= omega ||A||`;
//! * componentwise, `omega = max_i |r_i| / (|A| |x| + |b|)_i`, the same with
//!   `|dA| <= omega |A|` entrywise. This is the MA57/MUMPS default and the
//!   stricter of the two: it cannot be satisfied by scaling a row away.
//!
//! Rows whose denominator underflows are skipped rather than reported as
//! infinite: a structurally empty row carries no information about the solve.

use crate::error::RslabError;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;

/// Which backward error a refinement measures and stops on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackwardError {
    /// `||r||_inf / (||A||_inf ||x||_inf + ||b||_inf)`.
    Normwise,
    /// `max_i |r_i| / (|A| |x| + |b|)_i`. The default: it is the criterion the
    /// established sparse solvers certify against, and the weaker normwise one
    /// can certify a componentwise-bad answer.
    #[default]
    Componentwise,
}

/// What the caller asks refinement to achieve.
#[derive(Debug, Clone, Copy)]
pub struct RefinePolicy {
    /// Correction steps at most. Each step is one triangular solve against the
    /// stored factor plus one residual evaluation.
    pub max_steps: usize,
    /// Stop as soon as the measured backward error is at or below this.
    pub target: f64,
    /// Which backward error to measure.
    pub measure: BackwardError,
}

impl Default for RefinePolicy {
    fn default() -> Self {
        // Two steps is what a well-conditioned system needs to reach the
        // roundoff floor; the target is a few units of double roundoff, below
        // which the measure itself is noise.
        Self {
            max_steps: 2,
            target: 4.0 * f64::EPSILON,
            measure: BackwardError::default(),
        }
    }
}

impl RefinePolicy {
    /// The historical `solve_refined(a, b, max_iter)` behaviour: a fixed step
    /// budget, no early stop.
    pub fn steps(max_steps: usize) -> Self {
        Self {
            max_steps,
            target: 0.0,
            measure: BackwardError::default(),
        }
    }

    /// Builder: measure the normwise backward error instead.
    pub fn with_measure(mut self, measure: BackwardError) -> Self {
        self.measure = measure;
        self
    }

    /// Builder: stop at this backward error.
    pub fn with_target(mut self, target: f64) -> Self {
        self.target = target;
        self
    }
}

/// What refinement achieved.
#[derive(Debug, Clone, Copy)]
pub struct RefineOutcome {
    /// Correction steps actually taken (never more than the budget).
    pub steps: usize,
    /// The measured backward error of the returned iterate.
    pub omega: f64,
    /// Whether `omega <= policy.target`. A policy with target `0.0` never
    /// certifies, which is the honest report for a fixed-step budget.
    pub certified: bool,
}

/// The two matrix products refinement needs: the residual, and the
/// componentwise denominator `|A| |x|`. Implemented for both storage forms so
/// the refinement core is written once.
pub trait RefineOperator<T: Scalar> {
    fn dim(&self) -> usize;
    /// `y <- A x`.
    fn apply(&self, x: &[T], y: &mut [T]);
    /// `y <- |A| |x|`, in magnitudes.
    fn apply_abs(&self, x: &[T], y: &mut [f64]);
}

impl<T: Scalar> RefineOperator<T> for CscMatrix<T> {
    fn dim(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.symv(x, y);
    }
    fn apply_abs(&self, x: &[T], y: &mut [f64]) {
        for v in y.iter_mut().take(self.n) {
            *v = 0.0;
        }
        // The stored lower triangle stands for both halves, exactly as in
        // `symv`: the off-diagonal entry contributes to its row and its column.
        for j in 0..self.n {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                let i = self.row_idx[k];
                let a = self.values[k].magnitude();
                y[i] += a * x[j].magnitude();
                if i != j {
                    y[j] += a * x[i].magnitude();
                }
            }
        }
    }
}

impl<T: Scalar> RefineOperator<T> for GeneralCsc<T> {
    fn dim(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.matvec(x, y);
    }
    fn apply_abs(&self, x: &[T], y: &mut [f64]) {
        for v in y.iter_mut().take(self.n) {
            *v = 0.0;
        }
        for (j, w) in self.col_ptr.windows(2).enumerate() {
            let xj = x[j].magnitude();
            for k in w[0]..w[1] {
                y[self.row_idx[k]] += self.values[k].magnitude() * xj;
            }
        }
    }
}

/// `||r||_inf / (||A||_inf ||x||_inf + ||b||_inf)`, with `a_inf = ||A||_inf`
/// supplied by the caller (it is a property of the matrix, not of the iterate).
fn normwise<T: Scalar>(r: &[T], x: &[T], b: &[T], a_inf: f64) -> f64 {
    let inf = |v: &[T]| v.iter().map(|z| z.magnitude()).fold(0.0f64, f64::max);
    let den = a_inf * inf(x) + inf(b);
    if den > 0.0 {
        inf(r) / den
    } else {
        0.0
    }
}

/// `max_i |r_i| / (|A| |x| + |b|)_i` over the rows whose denominator is
/// nonzero.
fn componentwise<T: Scalar>(r: &[T], b: &[T], abs_ax: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..r.len() {
        let den = abs_ax[i] + b[i].magnitude();
        if den > 0.0 {
            worst = worst.max(r[i].magnitude() / den);
        }
    }
    worst
}

/// Refine `x` in place against `a` and `b`, using `solve` for the correction
/// steps, and report what was achieved.
///
/// The iterate returned is the best one seen, not the last: refinement is not
/// monotone on ill-conditioned systems, and returning a worse vector than the
/// one already computed would be a regression the caller cannot detect.
pub(crate) fn refine_in_place<T, A, S>(
    a: &A,
    b: &[T],
    x: &mut [T],
    policy: &RefinePolicy,
    mut solve: S,
) -> Result<RefineOutcome, RslabError>
where
    T: Scalar,
    A: RefineOperator<T> + ?Sized,
    S: FnMut(&[T]) -> Result<Vec<T>, RslabError>,
{
    let n = a.dim();
    debug_assert_eq!(b.len(), n);
    let a_inf = match policy.measure {
        BackwardError::Normwise => {
            // ||A||_inf via one pass with a unit vector: exact for the row sums
            // of |A|, and it reuses the operator both storage forms provide.
            let ones = vec![T::one(); n];
            let mut rows = vec![0.0f64; n];
            a.apply_abs(&ones, &mut rows);
            rows.iter().fold(0.0f64, |m, &v| m.max(v))
        }
        BackwardError::Componentwise => 0.0,
    };

    let mut ax = vec![T::zero(); n];
    let mut abs_ax = vec![0.0f64; n];
    let mut best = x.to_vec();
    let mut best_omega = f64::INFINITY;
    let mut steps = 0usize;

    for it in 0..=policy.max_steps {
        a.apply(x, &mut ax);
        let r: Vec<T> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
        let omega = match policy.measure {
            BackwardError::Normwise => normwise(&r, x, b, a_inf),
            BackwardError::Componentwise => {
                a.apply_abs(x, &mut abs_ax);
                componentwise(&r, b, &abs_ax)
            }
        };
        if omega < best_omega {
            best_omega = omega;
            best.copy_from_slice(x);
        }
        // A non-finite iterate never certifies, whatever the measure says.
        let finite = x.iter().all(|v| v.magnitude().is_finite());
        if (finite && omega <= policy.target) || it == policy.max_steps {
            break;
        }
        let dx = solve(&r)?;
        for (xi, &d) in x.iter_mut().zip(&dx) {
            *xi = *xi + d;
        }
        steps += 1;
    }

    x.copy_from_slice(&best);
    let certified = best_omega <= policy.target && best.iter().all(|v| v.magnitude().is_finite());
    Ok(RefineOutcome {
        steps,
        omega: best_omega,
        certified,
    })
}
