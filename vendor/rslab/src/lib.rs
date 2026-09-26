// No `.unwrap()` or `.expect()` outside the tests.
#![cfg_attr(not(test), deny(clippy::unwrap_used))]
#![cfg_attr(not(test), deny(clippy::expect_used))]
#![cfg_attr(test, allow(clippy::needless_range_loop))]

//! # RSLAB
//!
//! A sparse direct solver for real and complex matrices in pure Rust (no
//! BLAS, LAPACK or MKL), generic over `f64`, `f32`, `Complex<f64>` and
//! `Complex<f32>`. Three paths, matched to their operator classes:
//!
//! | path | factorization | for |
//! |---|---|---|
//! | [`LdltSolver`] | `P^T A P = L D L^T`, Bunch-Kaufman | symmetric and complex-symmetric (PARDISO `mtype 2`, `6`) |
//! | [`LuSolver`] | `P^T A P = L U`, threshold pivoting | general unsymmetric (`mtype 11`, `13`) |
//! | [`KluSolver`] | block triangular form, per-block Gilbert-Peierls LU | circuit-shaped matrices |
//!
//! The LDL^T and LU paths are left-looking supernodal and run in parallel over
//! the elimination tree; the factor is bit-identical for every thread count.
//! Every factor is also a [`Preconditioner`] for the Krylov solvers.
//!
//! ## Analyze once, factor many
//!
//! The analysis (ordering, elimination tree, supernodes) depends on the
//! pattern only; factor as many value sets on it as needed (Newton steps,
//! frequency sweeps), and solve against one or many right-hand sides.
//!
//! ```
//! # fn main() -> Result<(), rslab::RslabError> {
//! use rslab::prelude::*;
//! // Symmetric matrices pass the lower triangle (i >= j).
//! let a = CscMatrix::<f64>::from_triplets(3, &[0, 1, 2, 1], &[0, 1, 2, 0],
//!                                         &[2.0, 2.0, 2.0, -1.0])?;
//! let s = SolverSettings::default();
//! let sym = LdltSymbolic::analyze(&a, &s)?;
//! let f = sym.factor(&a, &s)?;
//! let x = f.solve(&[1.0, 2.0, 3.0])?;
//! let xs = f.solve_many(&vec![1.0; 3 * 4], 4)?; // 4 right-hand sides, n x nrhs column-major
//! # let _ = (x, xs); Ok(()) }
//! ```
//!
//! [`LuSymbolic`] and [`KluSymbolic`] take a [`GeneralCsc`] the same way,
//! and the one-shot forms ([`LdltSolver::factor`], [`LuSolver::factor`],
//! [`KluSolver::factor`]) analyze and factor in one call. All three solvers
//! implement [`Factorization`]: `solve`, `solve_many`, `solve_transpose`,
//! `solve_refined` and the diagnostics behind one trait object.
//!
//! ## Circuit-shaped matrices
//!
//! The KLU path adds a numeric-only [`refactor`](KluSolver::refactor) (frozen
//! pattern and pivots) for sweeps, a transpose solve for adjoints, and exports
//! its factors ([`l_matrix`](KluSolver::l_matrix)).
//!
//! ```
//! # fn main() -> Result<(), rslab::RslabError> {
//! use rslab::prelude::*;
//! let a = GeneralCsc::<f64>::from_triplets(3, &[0, 1, 2, 0], &[0, 1, 2, 2],
//!                                          &[4.0, 3.0, 2.0, 1.0])?;
//! let sym = KluSymbolic::analyze(&a, &KluSettings::default())?;
//! let mut f = sym.factor(&a, &KluSettings::default())?;
//! let x = f.solve(&[1.0, 1.0, 1.0])?;
//! let xt = f.solve_transpose(&[1.0, 1.0, 1.0])?; // A^T x = b
//! f.refactor(&a)?; // new values on the same pattern, no pivot search
//! # let _ = (x, xt); Ok(()) }
//! ```
//!
//! ## Preconditioners
//!
//! Static pivoting ([`SolverSettings::preconditioner`]) never fails, and a
//! drop tolerance trades fill for iterations. [`gmres`], [`gmres_block`],
//! [`cocg`] (complex symmetric) and [`cocr`] take any [`LinearOperator`] and
//! [`Preconditioner`] with their [`KrylovSettings`]; a `Complex<f32>`
//! factor preconditions an `f64` iteration through
//! [`LowPrecisionPreconditioner`].
//!
//! ```
//! # fn main() -> Result<(), rslab::RslabError> {
//! use rslab::prelude::*;
//! use num_complex::Complex;
//! let c = |re, im| Complex::new(re, im);
//! let a = CscMatrix::<Complex<f64>>::from_triplets(
//!     3, &[0, 1, 2, 1], &[0, 1, 2, 0],
//!     &[c(4.0, 1.0), c(4.0, 1.0), c(4.0, 1.0), c(-1.0, 0.2)])?;
//! let m = LdltSolver::factor(&a, &SolverSettings::preconditioner(1e-8).with_drop_tol(1e-2))?;
//! let res = cocg(&a, &vec![c(1.0, 0.0); 3], &m, &KrylovSettings::default().with_tol(1e-10))?;
//! assert!(res.converged);
//! # Ok(()) }
//! ```
//!
//! ## Threads
//!
//! A factorization runs in a scoped pool of its own. The default
//! [`Threads::Auto`] predicts the worker count from the analysis (from the
//! calibrated cost model once the one-time install diagnosis of feature
//! `tuning` has run), capped at 4 so concurrent solves coexist;
//! [`Threads::Fixed`] pins it and [`Threads::Ambient`] runs on the
//! surrounding rayon pool, which keeps a factorization and the solves after
//! it on one pool. A caller-owned flag ([`SolverSettings::with_interrupt`])
//! cancels a running factorization at the next supernode or panel boundary.
//!
//! ## Tuning
//!
//! Every constant of the analysis, the kernels and the solves is a field of
//! [`SolverSettings`] (grouped as [`OrderingSettings`], [`RaceSettings`],
//! [`MetisOptions`], [`AmalgamationSettings`], [`KernelSettings`],
//! [`SolveSettings`]) or [`KluSettings`], with the tuned value as its
//! default.
//!
//! ## Diagnostics and estimates
//!
//! Before any numeric work, [`LdltSymbolic::estimate_memory`] (and its LU and
//! KLU twins) predicts the factor storage, the transient peak and the flops
//! from the structure alone ([`MemoryEstimate`]). After it, every factor
//! answers `diagnostics()` ([`Diagnostics`]): stage times, fill, threads, the
//! decisions taken (ordering, scaling, pivoting), and the settings the chosen
//! path did not read. [`logging`] has one level (`RLA_LOG`, default
//! `warning`) and one replaceable sink. Refined solves report what they
//! achieved ([`RefinePolicy`], [`RefineOutcome`]).
//!
//! ## Allocator
//!
//! The dense kernels allocate per product. A caching allocator such as
//! `mimalloc` as the global allocator speeds up the factorization, most on
//! Windows, whose system heap maps large blocks afresh each time.

// The public API is the root `pub use` set below. Modules are `pub(crate)`
// except those the in-tree benches, tests and xtask reach by path: those are
// `pub` but `#[doc(hidden)]`, not part of the API.

// Monotonic clock shim: std Instant natively, inert on wasm32.
pub(crate) mod clock;
pub(crate) mod dense;
pub(crate) mod diagnostics;
pub(crate) mod error;
pub(crate) mod inertia;
pub(crate) mod io;
pub mod logging;
// Test-matrix generators (feature `matgen`) for the benches and tests;
// `matgen-download` adds a SuiteSparse / Matrix Market fetcher.
#[cfg(feature = "matgen")]
#[doc(hidden)]
pub mod matgen;
pub(crate) mod numeric;
// Ordering internals the benches and tests reach by path.
#[doc(hidden)]
pub mod ordering;
pub(crate) mod refine;
pub(crate) mod scalar;
// MC64 max-product matching and the equilibrations of the factor paths.
pub(crate) mod scaling;
pub(crate) mod sparse;
// Symbolic analysis internals the tests and examples reach by path.
#[doc(hidden)]
pub mod symbolic;
// Hardware calibration (feature `tuning`), driven by xtask.
#[cfg(feature = "tuning")]
#[doc(hidden)]
pub mod tuning;

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
    cocg, cocr, gmres, gmres_block, gmres_recycled, BlockKrylovResult, Factorization, FnOperator,
    FnPreconditioner, KrylovResult, KrylovSettings, LinearOperator, LowPrecisionLu,
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
        KrylovSettings,
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
