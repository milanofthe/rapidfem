//! Matrix scaling: symmetric equilibration for the LDL^T path and MC64
//! weighted matching for the LU path.
//!
//! ## Symmetric equilibration (LDL^T)
//!
//! [`compute_scaling`] turns a [`ScalingStrategy`] into a real vector `s`
//! such that the factored matrix is the congruence `A_hat = D A D`,
//! `D = diag(s)`:
//!
//! - `OnePassInfNorm` (the `SolverSettings` default): a single
//!   Knight-Ruiz step `s_i = 1/sqrt(max_j |A_ij|)`; tolerates a zero
//!   diagonal.
//! - `InfNorm`: iterative Knight-Ruiz inf-norm equilibration.
//! - `Mc64Symmetric`: matching-based scaling after Duff & Koster (2001)
//!   and Duff & Pralet (2005), computed with a pure-Rust Hungarian
//!   algorithm, so that the largest entries of `D A D` lie on the
//!   diagonal.
//! - `Identity`, and `External` (a caller-supplied vector).
//!
//! The vector is in user-order indexing (the numbering of the input CSC).
//! `LdltSolver` computes it at factor time (through this module on the
//! `|A|` magnitude pattern, except for the one-pass default and
//! `Identity`, which it handles natively), multiplies each entry
//! `a[i,j]` by `s[i] * s[j]` while permuting the values for the
//! factorization, and fuses the scaling into the permutation
//! gather/scatter of the solve:
//! `x = D * (A_hat^-1 * (D b))`. The same vector is applied on both ends,
//! not its inverse. The sparsity pattern is unaffected.
//!
//! ## Unsymmetric matching (LU)
//!
//! `mc64::compute_matching_general` computes a maximum-product
//! transversal of a general square matrix (Hungarian kernel in
//! `hungarian.rs`), and `mc64::unsymmetric_scaling` turns its dual
//! variables into row and column scalings. The LU path uses the matching
//! as a row permutation that puts large entries on the diagonal, together
//! with those scalings; the BTF analysis uses it as the transversal.

use crate::error::RslabError;
use crate::sparse::csc::CscMatrix;

mod hungarian;
mod infnorm;
pub(crate) mod mc64;

/// One Knight-Ruiz equilibration step `d <- d / sqrt(m)`, guarded against
/// overflow/underflow. Applies the update only when
/// the result stays finite and strictly positive; otherwise `d` is held at
/// its last good value. `m` is the row/column infinity-norm and is assumed
/// `> 0` (the caller's existing `m > 0` guard).
///
/// On well-scaled matrices `d / sqrt(m)` is always finite and positive, so
/// this is **bit-identical** to the bare division - the guard bites only on
/// extreme or subnormal couplings, where the unguarded `d` would reach
/// `+-Inf` (then `NaN` on the next sweep) or `0`. Such a value silently
/// poisons every coupled row: the factorization sees a zeroed/NaN row, the
/// static pivot perturbation "repairs" it, and the solve returns garbage
/// with no error. Keeping `d` finite each sweep - rather than only
/// sanitizing at the end - also stops one overflowing row from dragging its
/// neighbours to zero.
#[inline]
pub(crate) fn kr_guarded_update(d: f64, m: f64) -> f64 {
    let cand = d / m.sqrt();
    if cand.is_finite() && cand > 0.0 {
        cand
    } else {
        d
    }
}

/// Guarded one-pass scale factor `1 / sqrt(m)` (single Knight-Ruiz
/// step): a zero, non-finite, or overflow-prone row max
/// yields the neutral `1.0` instead of a `0`/`Inf`/`NaN` factor that would
/// silently poison the equilibrated matrix. Bit-identical to the bare
/// expression for every healthy `m` (finite, `> 0`, not extreme).
#[inline]
pub(crate) fn inv_sqrt_scale_guarded(m: f64) -> f64 {
    if m > 0.0 {
        let cand = 1.0 / m.sqrt();
        if cand.is_finite() && cand > 0.0 {
            cand
        } else {
            1.0
        }
    } else {
        1.0
    }
}

/// The symmetric equilibration of the LDL^T path. `OnePassInfNorm` (the
/// default) is the cheapest; `InfNorm` iterates it to convergence;
/// `Mc64Symmetric` scales by a maximum-product matching, at its extra
/// cost, which helps matrices whose large entries sit off the diagonal
/// (saddle-point and KKT systems).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ScalingStrategy {
    /// Knight-Ruiz inf-norm iterative equilibration (the "iterative Ruiz"
    /// arm of the equilibration knob). See `infnorm::compute_infnorm`.
    InfNorm,
    /// One-pass symmetric inf-norm equilibration `s_i = 1/sqrt(max_j |A_ij|)` (a
    /// single Knight-Ruiz step). The [`crate::SolverSettings`] default:
    /// cheapest, tolerates a zero diagonal, no iteration. See
    /// `infnorm::compute_onepass`.
    #[default]
    OnePassInfNorm,
    /// MC64-style symmetric matching-based scaling. Matches the
    /// default behavior of MUMPS (SYM=2) and SSIDS
    /// (options%scaling=1). Useful on matrices where matching
    /// provides better conditioning than inf-norm balancing.
    Mc64Symmetric,
    /// Identity scaling (no-op). Use for regression testing and for
    /// inputs where any scaling is inappropriate.
    Identity,
    /// User-supplied pre-computed scaling vector in user-order
    /// indexing. Length must equal the matrix dimension.
    External(Vec<f64>),
}

/// Diagnostic information about how the scaling was computed.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingInfo {
    /// A non-trivial scaling vector was applied to the matrix and the
    /// solve path must undo it. Produced when MC64 matching ran to
    /// completion on a non-singular matrix, and when the caller
    /// supplied an `External` scaling vector (the factor applies
    /// `D = diag(s)` regardless of how `s` was obtained).
    Applied,
    /// MC64 matching found a partial solution; unmatched rows and
    /// columns fall back to identity scaling. `n_unmatched` is the
    /// number of variables that could not be matched. The returned
    /// scaling vector has `1.0` at the unmatched positions.
    PartialSingular { n_unmatched: usize },
    /// The scaling vector is all-ones - applying it is a no-op, so the
    /// solve path skips pre/post scaling entirely. Produced only by
    /// `ScalingStrategy::Identity`. (`External` reports `Applied` even
    /// when its vector happens to be all-ones, since the factor still
    /// runs the scaling loop.)
    NotApplied,
}

/// Compute the symmetric scaling vector for a sparse symmetric
/// matrix stored in CSC with only the lower triangle, following
/// `strategy`.
///
/// Returns a vector of length `n` in **user-order** indexing such
/// that applying `D = diag(scaling)` as the congruence transform
/// `D * A * D` produces a matrix whose largest-magnitude entries lie
/// on the diagonal. The off-diagonals are bounded by 1 in absolute
/// value when MC64 succeeds on a non-singular matrix.
pub fn compute_scaling(
    matrix: &CscMatrix,
    strategy: &ScalingStrategy,
) -> Result<(Vec<f64>, ScalingInfo), RslabError> {
    match strategy {
        ScalingStrategy::Identity => Ok((vec![1.0; matrix.n], ScalingInfo::NotApplied)),
        ScalingStrategy::External(s) => {
            if s.len() != matrix.n {
                return Err(RslabError::InvalidInput(format!(
                    "external scaling has length {} but matrix has n={}",
                    s.len(),
                    matrix.n,
                )));
            }
            // `Applied`, not `NotApplied`: the factor scales the matrix
            // by `D = diag(s)` unconditionally, so the solve MUST undo
            // it. `NotApplied` is a load-bearing invariant meaning "the
            // scaling vector is all-ones" - the solve keys off it to
            // skip pre/post scaling (`solve_sparse`). Pairing a real
            // `s` with `NotApplied` factors `D*A*D` but solves it as
            // `A`, returning `D^-1A^-1D^-1b`. `s` may itself be all-ones,
            // in which case `Applied` just does bit-exact `x1.0` no-ops.
            Ok((s.clone(), ScalingInfo::Applied))
        }
        ScalingStrategy::InfNorm => Ok(infnorm::compute_infnorm(matrix)),
        ScalingStrategy::OnePassInfNorm => Ok(infnorm::compute_onepass(matrix)),
        ScalingStrategy::Mc64Symmetric => mc64::compute_symmetric(matrix),
    }
}

// Hungarian kernel types, used by the `mc64` module. Not part of the
// public API.
#[allow(unused_imports)]
pub(crate) use hungarian::{hungarian_match, CostGraph, Matching};
