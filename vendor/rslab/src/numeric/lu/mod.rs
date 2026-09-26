//! Generic **unsymmetric** sparse LU factorization over any [`Scalar`] field -
//! the general (non-symmetric) complex path, complementing the symmetric LDL^T
//! path in [`crate::numeric::ldlt`].
//!
//! It targets matrices whose *values* are unsymmetric (e.g. MoM A-EFIE
//! near-field saddle preconditioners, where the symmetric and antisymmetric
//! parts are comparable) but reuses the full symmetric machinery: the
//! fill-reducing ordering, supernodes and assembly tree
//! ([`LuSymbolic::analyze`]) and the SIMD
//! `gemm` Schur kernel. Only the dense panel kernel changes - an unsymmetric
//! LU producing separate `L` and `U` - and the analysis runs on the
//! **symmetrized pattern** `A union A^T` so the elimination structure carries
//! fill for both factors.
//!
//! ## Pivoting
//!
//! * **Threshold partial pivoting** (UMFPACK-style, `THRESH = 0.1`), bounded to
//!   each panel's fully-summed block: the diagonal is kept unless it falls below
//!   `THRESH * |colmax|`, in which case the column max is brought up. Sub-floor
//!   pivots are perturbed in preconditioner mode ([`ZeroPivotAction::PerturbToEps`])
//!   or rejected in exact mode. Pivoting stays cheap on the equilibrated,
//!   unit-diagonal MoM matrices while guarding the genuinely ill-scaled columns.
//! * The factors `L` (unit lower) and `U^T` (the pivots on its diagonal) are
//!   kept in supernodal panel form (`PanelFactor`), in factorization order:
//!   each supernode's finished panels become the stored factor. The
//!   factorization is the
//!   supernodal **left-looking** kernel (low transient, no contribution-block
//!   stack).

mod factor;
mod factors;
mod node;
mod solver;
mod structure;
#[cfg(test)]
mod tests;

pub use solver::{LuSolver, LuSymbolic};
