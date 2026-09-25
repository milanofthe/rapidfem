// Deny `.unwrap()` and `.expect()` in production code, but allow them in
// test modules (inside `#[cfg(test)]` blocks) where panics are acceptable.
// This is a structural enforcement of the CLAUDE.md hard rule against
// unwrap in `src/`, replacing the ad-hoc grep check in CI.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
// Style lints that fire only in test scaffolding - relaxed under cfg(test).
// The lib build keeps default clippy strictness.
#![cfg_attr(test, allow(clippy::needless_range_loop))]

//! # RSLAB - a pure-Rust sparse symmetric direct solver and preconditioner
//!
//! A self-contained replacement for PARDISO's sparse symmetric path, with no
//! MKL or other native dependency. RSLAB factors **real symmetric** (`f64`,
//! PARDISO `mtype 2`) and **complex symmetric** (`Complex<f64>`, `mtype 6`)
//! matrices as `P^T A P = L D L^T` by a rayon-parallel multifrontal
//! Bunch-Kaufman method with a SIMD (`gemm`) Schur kernel.
//!
//! Three intended uses:
//! * **FEM direct solve** - factor once, solve many right-hand sides.
//! * **MoM sparse preconditioner** - a robust, memory-light approximate factor
//!   (static pivoting, `f32` mixed precision, incomplete-factor dropping)
//!   driving a [`cocg`]/[`cocr`] iteration.
//! * **Circuit-shaped unsymmetric solve** - the sequential, bit-deterministic
//!   [`KluSolver`] (BTF + per-block Gilbert-Peierls) with a numeric-only
//!   [`refactor`](KluSolver::refactor) for fixed-pattern sweeps and a
//!   [`solve_transpose`](KluSolver::solve_transpose) (`A^Tx = b` on the same
//!   factors) for adjoint / sensitivity solves.
//!
//! ## PARDISO-style phased workflow (FEM)
//!
//! Analyze the sparsity pattern once, then factor many value sets that share it
//! (Newton steps, time stepping, frequency sweep):
//!
//! ```
//! # fn main() -> Result<(), rslab::RslabError> {
//! use rslab::prelude::*;
//! // Real symmetric matrix, lower triangle (i >= j).
//! let a = CscMatrix::<f64>::from_triplets(3, &[0, 1, 2, 1], &[0, 1, 2, 0],
//!                                         &[2.0, 2.0, 2.0, -1.0])?;
//! let analysis = LdltSymbolic::analyze(&a, &SolverSettings::default())?;                 // phase 1
//! let factor = analysis.factor(&a, &SolverSettings::default())?; // 2/3
//! let x = factor.solve(&[1.0, 2.0, 3.0])?;
//! # let _ = x; Ok(()) }
//! ```
//!
//! ## Complex-symmetric MoM preconditioner
//!
//! A robust, low-memory factor (never-fail static pivoting + incomplete
//! dropping) used to precondition COCG:
//!
//! ```
//! # fn main() -> Result<(), rslab::RslabError> {
//! use rslab::prelude::*;
//! use num_complex::Complex;
//! let c = |re, im| Complex::new(re, im);
//! let a = CscMatrix::<Complex<f64>>::from_triplets(
//!     3, &[0, 1, 2, 1], &[0, 1, 2, 0],
//!     &[c(4.0, 1.0), c(4.0, 1.0), c(4.0, 1.0), c(-1.0, 0.2)])?;
//! let opts = SolverSettings::preconditioner(1e-8).with_drop_tol(1e-2); // composable
//! let m = LdltSolver::factor(&a, &opts)?;          // preconditioner
//! let b = vec![c(1.0, 0.0); 3];
//! let res = cocg(&a, &b, &m, 1e-10, 100)?;
//! assert!(res.converged);
//! # Ok(()) }
//! ```

// -------------------------------------------------------------------------
// Module visibility (audit finding M4).
//
// The embedder contract is the curated root `pub use` set below - that is the
// ONLY surface `cargo doc` should show. Every module is therefore `pub(crate)`;
// the intended-public items are re-exported at the crate root and documented
// there. The exceptions are modules that in-tree tooling (benches, integration
// tests, xtask) reaches by their full module path for
// internal building blocks that are deliberately NOT part of the embedder API:
// those stay `pub` but `#[doc(hidden)]`, so they compile as external crates see
// them yet never appear in the public docs. Each such case is commented.
// -------------------------------------------------------------------------

/// Single-solve thread-count policy from the symbolic analysis.
/// Monotonic clock shim: std Instant natively, inert on wasm32 (no OS clock).
pub(crate) mod clock;
pub(crate) mod dense;
/// Deterministic resource diagnostics: a-priori peak-memory estimate + per-stage
/// runtime/memory report for solver-in-the-loop scheduling.
pub(crate) mod diagnostics;
pub(crate) mod error;
pub(crate) mod inertia;
pub(crate) mod io;
pub mod logging;
/// Parametrized test-matrix generators (feature `matgen`): PDE stencils, BEM/MoM
/// kernels, banded/arrow, random + spectral. Optional `matgen-download` adds a
/// SuiteSparse / Matrix Market fetcher.
///
/// Not part of the embedder API: `pub` only so the in-tree benches can build
/// test matrices; hidden from the public docs.
#[cfg(feature = "matgen")]
#[doc(hidden)]
pub mod matgen;
pub(crate) mod numeric;
/// Fill-reducing ordering internals. Not part of the embedder API: `pub` only
/// because the in-tree benches/tests reach `ordering::amd::permute_pattern` and
/// `ordering::elimination_tree::EliminationTree` by path; hidden from public docs.
#[doc(hidden)]
pub mod ordering;
pub(crate) mod scalar;
// MC64 max-product matching + equilibration. The wired surface is the
// `LdltCompress` matching (`compute_mc64_cache`/`Mc64Cache`, structural only),
// `ScalingStrategy`, and the inf-norm / one-pass equilibration used by the
// factor path.
/// Symbolic analysis internals. Not part of the embedder API beyond the root
/// re-exports (`OrderingMethod`, `RelaxAmalgamation`):
/// `pub` because the in-tree tests drive `symbolic::{symbolic_factorize,
/// column_counts_gnp, ...}` and `symbolic::supernode::OrderingPreprocess` by
/// path; hidden from public docs.
#[doc(hidden)]
pub mod refine;
pub(crate) mod scaling;
pub(crate) mod sparse;
pub mod symbolic;
/// Hardware-aware auto-tuning + resource governor (feature `tuning`): hardware
/// probe, calibration cache, and a budget-driven factorization planner.
///
/// Not part of the embedder API: `pub` only so xtask can drive calibration
/// (`tuning::{Calibration, HardwareInfo}`); hidden from the public docs.
#[cfg(feature = "tuning")]
#[doc(hidden)]
pub mod tuning;

// Flat public API re-exported at crate root - a single data-type-generic
// (`Scalar`: f64, Complex<f64>, f32, Complex<f32>) sparse direct + iterative
// stack. (The legacy f64-dedicated multifrontal path has been removed.)
pub use diagnostics::{
    Decisions, Diagnostics, MemoryEstimate, NumericReport, Rates, SolveStats, StageReport,
};
pub use error::RslabError;
pub use inertia::Inertia;
pub use io::mtx::{
    parse_mtx, parse_mtx_complex, parse_mtx_complex_general, read_mtx, read_mtx_any,
    read_mtx_complex, MtxLoaded, MtxMatrix,
};
pub use logging::{LogLevel, LogSink};
pub use refine::{BackwardError, RefineOperator, RefineOutcome, RefinePolicy};
pub use scalar::Scalar;
pub use scaling::ScalingStrategy;
// The three direct solvers: `XSymbolic::analyze -> .factor -> XSolver`.
pub use numeric::klu::{KluParallel, KluSettings, KluSolver, KluSymbolic};
pub use numeric::ldlt::{LdltSolver, LdltSymbolic};
pub use numeric::lu::{LuSolver, LuSymbolic};
pub use numeric::settings::{
    AmalgamationSettings, AmdOptions, AmfOptions, KernelSettings, MatchingSettings, MetisOptions,
    OrderingSettings, PivotSettings, RaceSettings, SolveSettings, SolverSettings, Threads,
    ZeroPivotAction,
};
// The Krylov solvers and their operator and preconditioner traits.
pub use numeric::krylov::{
    cocg, cocr, gmres, gmres_block, gmres_block_mon, gmres_recycled, BlockKrylovResult,
    Factorization, FnOperator, FnPreconditioner, KrylovResult, LinearOperator, LowPrecisionLu,
    LowPrecisionPreconditioner, NoPreconditioner, Preconditioner, Recycle, RecycleScalar,
    StopReason,
};
pub use sparse::csc::{CscMatrix, CscPattern};
pub use sparse::general::GeneralCsc;
pub use symbolic::{AmalgamationStrategy, OrderingMethod, RelaxAmalgamation};

/// Ergonomic imports for embedding RSLAB as a PARDISO-style sparse solver /
/// preconditioner. `use rslab::prelude::*;` brings in the matrix type, the
/// phased analysis/factor API, the iterative solvers and preconditioners, the
/// options enums, and the Matrix Market loaders.
pub mod prelude {
    pub use crate::{
        // iterative solvers, operators, preconditioners (solver-in-the-loop)
        cocg,
        cocr,
        gmres,
        // Matrix Market loaders + error type
        parse_mtx,
        parse_mtx_complex,
        parse_mtx_complex_general,
        read_mtx,
        read_mtx_complex,
        // matrices, options, scalar field
        CscMatrix,
        Factorization,
        GeneralCsc,
        // high-level phased solvers: `XSymbolic::analyze -> .factor -> XSolver`
        KluSettings,
        KluSolver,
        KluSymbolic,
        KrylovResult,
        LdltSolver,
        LdltSymbolic,
        LinearOperator,
        LowPrecisionLu,
        LowPrecisionPreconditioner,
        LuSolver,
        LuSymbolic,
        MtxMatrix,
        NoPreconditioner,
        Preconditioner,
        RslabError,
        Scalar,
        SolverSettings,
        ZeroPivotAction,
    };
}
