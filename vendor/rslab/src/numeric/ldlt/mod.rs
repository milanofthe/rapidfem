//! Sparse LDL^T factorization over any [`Scalar`] field: the real (`f64`)
//! and the complex-*symmetric* (`Complex<f64>`, PARDISO `mtype 6`) case, on
//! the value-agnostic symbolic analysis (ordering, elimination tree,
//! supernode amalgamation), with a supernodal left-looking kernel.
//!
//! ## Pivoting scope
//!
//! * Pivoting is restricted to the **fully-summed block** of each supernode:
//!   dense Bunch-Kaufman with 1x1 and 2x2 pivots, so an indefinite block (a KKT
//!   saddle, a circuit's zero-diagonal source row next to its node) factors
//!   whenever the pair sits in one supernode, which the amalgamation makes the
//!   common case (a 45k-node power grid: 1690 2x2 pivots, no failure). There
//!   is no delayed pivoting: a fully-summed block that is singular in exact
//!   mode surfaces as [`RslabError::NumericallyRankDeficient`], and the
//!   static-pivot mode ([`ZeroPivotAction`], the `preconditioner` settings)
//!   lifts the pivot to the floor instead and reports it in `n_perturbed`.
//! * The global factor `L` is kept in supernodal panel form
//!   (`PanelFactor`): each supernode's dense panel, once its last consumer
//!   is done, is finished in place (off-block rows into elimination order,
//!   the 2x2 couplings cleared, `drop_tol` applied) and becomes the stored
//!   factor, so the memory peak is the resident panels themselves (see the
//!   a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate)).

mod bunch_kaufman;
mod factor;
mod gemm;
mod node;
mod pivots;
mod solver;
#[cfg(test)]
mod tests;

pub(crate) use pivots::LdltPivots;
pub use solver::{LdltSolver, LdltSymbolic};
