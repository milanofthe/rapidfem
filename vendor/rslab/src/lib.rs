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
//! matrices as `Pᵀ A P = L D Lᵀ` by a rayon-parallel multifrontal
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
//!   [`solve_transpose`](KluSolver::solve_transpose) (`Aᵀx = b` on the same
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
//! // Real symmetric matrix, lower triangle (i ≥ j).
//! let a = CscMatrix::<f64>::from_triplets(3, &[0, 1, 2, 1], &[0, 1, 2, 0],
//!                                         &[2.0, 2.0, 2.0, -1.0])?;
//! let analysis = LdltSymbolic::analyze(&a)?;                 // phase 1
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
//! let m = LdltSolver::factor_with(&a, &opts)?;          // preconditioner
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
// tests, the `bench_sparse` bin, xtask) reaches by their full module path for
// internal building blocks that are deliberately NOT part of the embedder API:
// those stay `pub` but `#[doc(hidden)]`, so they compile as external crates see
// them yet never appear in the public docs. Each such case is commented.
// -------------------------------------------------------------------------

/// Structural feature extraction (the "structure analyzer"): distils a matrix +
/// its symbolic analysis into a compact [`analysis::StructuralFeatures`] vector
/// for diagnostics and auto-tuning input.
pub(crate) mod analysis;
/// Auto-tuner: predict the knob config minimizing a weighted time/memory score
/// for a matrix from its [`analysis::StructuralFeatures`], via an embedded MLP
/// performance model (trained offline on the corpus sweep; pure-Rust inference).
pub(crate) mod auto_tune;
pub(crate) mod dense;
/// Deterministic resource diagnostics: a-priori peak-memory estimate + per-stage
/// runtime/memory report for solver-in-the-loop scheduling.
pub(crate) mod diagnostics;
pub(crate) mod error;
pub(crate) mod inertia;
pub(crate) mod io;
/// Parametrized test-matrix generators (feature `matgen`): PDE stencils, BEM/MoM
/// kernels, banded/arrow, random + spectral, plus a tagged catalog for benchmarks.
/// Optional `matgen-download` adds a SuiteSparse / Matrix Market fetcher.
///
/// Not part of the embedder API: `pub` only so the in-tree benches and xtask can
/// build test matrices; hidden from the public docs.
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
// MC64 max-product matching + equilibration. The wired surface is the symbolic
// MC64 cache (`compute_mc64_cache`/`Mc64Cache`), `ScalingStrategy`, and the
// inf-norm / one-pass equilibration used by the factor path.
pub(crate) mod scaling;
pub(crate) mod sparse;
/// Symbolic analysis internals. Not part of the embedder API beyond the root
/// re-exports (`OrderingMethod`, `RelaxAmalgamation`, `SymbolicProfileReport`):
/// `pub` because the in-tree tests drive `symbolic::{symbolic_factorize,
/// column_counts_gnp, ...}` and `symbolic::supernode::OrderingPreprocess` by
/// path; hidden from public docs.
#[doc(hidden)]
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
pub use analysis::{recommend_threads_from, StructuralFeatures, SymbolicShape};
pub use auto_tune::{
    apply_profile, default_profile, recommend_settings, recommend_settings_pathed,
    recommend_settings_vetoed, recommend_with_profile, SolverPath, TunerProfile,
    DEFAULT_TUNE_WEIGHT,
};
pub use dense::matrix::SymmetricMatrix;
pub use diagnostics::{Diagnostics, MemoryEstimate, StageReport};
pub use error::RslabError;
pub use numeric::gemm_tuning::{
    GemmThresholds, DEFAULT_PANEL_NB, DEFAULT_PAR_CDIV, DEFAULT_PAR_GEMM, DEFAULT_SCALAR_GATE,
};
pub use scalar::Scalar;
pub use scaling::ScalingStrategy;
// Generic dense LDLᵀ kernel (the multifrontal fronts reduce to this).
pub use dense::ldlt_generic::{
    factor_ldlt, solve_ldlt, solve_ldlt_many, CompressedLdltFactors, LdltFactors,
};
// Shared options + the low-level multifrontal symbolic/numeric building blocks.
pub use numeric::multifrontal_ldlt::{
    analyze, analyze_with, factor_numeric, factor_sparse_ldlt, factor_sparse_ldlt_with,
    with_threads, BlrMode, FactorMethod, MemoryMode, MultifrontalSymbolic, ReorderMode,
    SolverSettings, Threads, ZeroPivotAction,
};
// High-level symmetric LDLᵀ solver: `LdltSymbolic::analyze → .factor → LdltSolver`.
pub use numeric::sparse_solver::{LdltSolver, LdltSymbolic};
// High-level unsymmetric LU solver: `LuSymbolic::analyze → .factor → LuSolver`,
// plus the raw factor type and free building blocks.
pub use inertia::Inertia;
pub use io::mtx::{
    parse_mtx, parse_mtx_complex, parse_mtx_complex_general, read_mtx, read_mtx_any,
    read_mtx_complex, MtxLoaded, MtxMatrix,
};
pub use numeric::iterative::{
    cocg, cocr, gmres, gmres_block, gmres_block_fn, gmres_block_fn_mon, gmres_block_mon, gmres_fn,
    gmres_recycled, BlockKrylovResult, Factorization, KrylovResult, LinearOperator, LowPrecisionLu,
    LowPrecisionPreconditioner, NoPreconditioner, Preconditioner, Recycle, RecycleScalar,
    StopReason,
};
pub use numeric::multifrontal_lu::{
    factor_general_lu, factor_general_lu_numeric, solve_lu, solve_lu_many, solve_lu_refined,
    take_blr_cb_stats, take_front_stats, FrontStat, LuFactors, LuSolver, LuSymbolic,
};
// KLU-style third direct path (BTF + per-block Gilbert-Peierls): sequential,
// bit-deterministic, built for circuit-shaped matrices and sweep refactoring.
pub use numeric::klu::{KluSettings, KluSolver, KluSymbolic};
pub use sparse::csc::{CscMatrix, CscPattern};
pub use sparse::general::GeneralCsc;
pub use symbolic::{OrderingMethod, RelaxAmalgamation, SymbolicProfileReport};

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
        // high-level phased solvers: `XSymbolic::analyze → .factor → XSolver`
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
