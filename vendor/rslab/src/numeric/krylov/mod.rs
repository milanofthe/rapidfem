//! Krylov iteration for complex-symmetric systems, preconditioned by an RLA
//! factorization.
//!
//! The target use is **3D EM FEM / MOM**: large complex-symmetric `A = A^T`
//! (PARDISO `mtype 6`) systems solved iteratively, with a robust, memory-light
//! RLA factorization (static-pivoted, optionally `f32` / incomplete) as the
//! preconditioner. The iterative method of choice for `A = A^T` is **COCG**
//! (Conjugate Orthogonal Conjugate Gradient, van der Vorst & Melissen 1990):
//! structurally CG, but every inner product is the *unconjugated* bilinear
//! form `x^T y = sum x_i y_i` - the correct geometry for a complex-symmetric (not
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
//! backward-stable, schemes:
//!
//! - **Single-RHS [`gmres`]:** *modified* Gram-Schmidt (each projection updates
//!   `w` before the next is taken) with a conditional DGKS second pass, triggered
//!   when one MGS sweep cancels more than `1/sqrt2` of `||w||`.
//! - **Block [`gmres_block`]:** *classical* Gram-Schmidt with a conditional second
//!   pass = **CGS2**, batched over the whole panel for BLAS-3 arithmetic
//!   intensity; the DGKS second pass is decided and applied **per column**.
//!
//! Consequently a **block solve with `s = 1` is *not* bit-identical to the single-
//! RHS [`gmres`]**: MGS and CGS accumulate the projections in a different order, so
//! the two can differ by a rounding ULP per step and hence by up to +/-1 iteration
//! at a residual that straddles `tol`. Both converge to the same solution within
//! the requested tolerance. This is a deliberate design point (CGS2's panel-wide
//! reductions are what make the multi-RHS path fast and thread-count deterministic),
//! not a bug; see the `gmres_block_single_rhs_matches_scalar_gmres` test.

mod block;
mod cg;
mod closures;
mod gmres;
mod operator;
mod recycle;
mod result;
mod settings;
#[cfg(test)]
mod tests;
mod util;
#[cfg(test)]
mod warmstart_tests;

pub use block::{gmres_block, BlockKrylovResult};
pub use cg::{cocg, cocr};
pub use closures::{FnOperator, FnPreconditioner};
pub use gmres::gmres;
pub use operator::{
    Factorization, LinearOperator, LowPrecisionLu, LowPrecisionPreconditioner, NoPreconditioner,
    Preconditioner,
};
pub use recycle::{gmres_recycled, Recycle, RecycleScalar};
pub use result::{KrylovResult, StopReason};
pub use settings::KrylovSettings;
