//! KLU-style sparse LU: BTF + per-block left-looking Gilbert-Peierls.
//!
//! The third direct path next to the supernodal LDL^T and LU, built for
//! circuit-shaped matrices: extremely sparse, unsymmetric, near-triangularizable,
//! with diagonal blocks far too small for supernodal/BLAS-3 kernels to pay off.
//! Algorithmic reference: SuiteSparse KLU (Davis & Palamadai Natarajan); this is
//! an independent pure-Rust implementation, no FFI.
//!
//! Pipeline:
//!
//! 1. **Analyze** ([`KluSymbolic::analyze`]): maximum transversal + Tarjan SCC
//!    ([`crate::ordering::btf`]) permute the matrix to block *upper* triangular
//!    form with a zero-free diagonal (structural singularity is detected here),
//!    then AMD orders each irreducible diagonal block on its symmetrized
//!    pattern.
//! 2. **Factor** ([`KluSymbolic::factor`]): each diagonal block is factored by
//!    a left-looking Gilbert-Peierls LU, per-column depth-first reach on the
//!    growing L pattern, so the numeric work is proportional to the flop count,
//!    with threshold partial pivoting that prefers the (structurally nonzero)
//!    diagonal. Off-block entries are not factored; they only enter the block
//!    back-substitution. Optional row-max scaling equilibrates the rows first.
//! 3. **Refactor** ([`KluSolver::refactor`]): numeric-only re-factorization for
//!    a matrix with the *same* pattern (frequency sweeps, Newton steps): the
//!    stored pattern and pivot sequence are replayed with no symbolic work and
//!    no pivot search. A changed pattern is detected and rejected; a pivot that
//!    became zero under the frozen pivot order fails cleanly so the caller can
//!    re-[`factor`](KluSymbolic::factor) with pivoting.
//!
//! Each block factors sequentially, and independent blocks may factor in
//! parallel ([`KluParallel`]); the results are bit-identical across runs and
//! thread counts.

use crate::error::RslabError;
use crate::numeric::supernodal::PanelPtr;
use crate::ordering::btf;
use crate::scalar::{fmadd, Scalar};
use crate::sparse::general::GeneralCsc;

mod factor;
mod refactor;
mod settings;
mod solver;
mod symbolic;
#[cfg(test)]
mod tests;

use factor::{diagnostics_flops, factor_csc, factor_impl, row_scale_inv};
pub use settings::{KluParallel, KluSettings};

const UNSET: usize = usize::MAX;

/// Narrow index type for the numeric factor's row-index streams and the
/// factor-time DFS state. The Gilbert-Peierls kernel is memory-bound on
/// index chasing; 32-bit indices halve that traffic (`n < 2^32 - 1` is
/// enforced at analyze time, far beyond this path's design point).
type Ki = u32;
const KI_UNSET: Ki = Ki::MAX;
/// Tag bit in the refactor scatter program: the entry lands in an F slot
/// (off-block value) rather than the elimination work vector.
const KI_FBIT: Ki = 1 << 31;

/// Symbolic analysis for the KLU path: the BTF block structure plus the
/// per-block fill-reducing ordering. Analyze once, then factor any number of
/// matrices sharing the pattern.
#[derive(Debug, Clone)]
pub struct KluSymbolic {
    n: usize,
    nnz: usize,
    /// Pre-pivot row permutation (new-to-old): BTF matching then SCC order then
    /// per-block AMD. Partial pivoting at factor time refines this within
    /// each block.
    pre_row_perm: Vec<usize>,
    /// Column permutation (new-to-old); never changed by pivoting.
    col_perm: Vec<usize>,
    /// Diagonal-block boundaries; see [`crate::ordering::btf::BtfForm`].
    block_ptr: Vec<usize>,
    /// The analyzed pattern in the pre-pivot permuted space (column `k` is
    /// original column `col_perm[k]`, rows are pre-pivot positions). Kept so
    /// the a-priori estimators run without the matrix, like
    /// [`LuSymbolic`](crate::LuSymbolic)'s stored symbolic structure.
    pat_col_ptr: Vec<usize>,
    pat_row_idx: Vec<usize>,
    /// Lazily computed, cached symbolic fill (the estimator pass costs about
    /// as much as a numeric factor, so the phased `factor` must not pay it
    /// again on every call).
    fill: std::sync::OnceLock<KluFill>,
}

/// Exact symbolic fill of the KLU factor under the diagonal-pivoting
/// assumption (the default expectation: BTF guarantees a structurally nonzero
/// diagonal and `pivot_threshold` strongly prefers it). Threshold pivoting at factor
/// time can shift individual counts, not their order of magnitude.
#[derive(Debug, Clone, Copy)]
struct KluFill {
    l_nnz: u64,
    u_nnz: u64,
    f_nnz: u64,
    /// Gilbert-Peierls flop count (multiply-subtract pairs + divisions).
    flops: u64,
}

/// The numeric KLU factorization: `P A Q = L U` per diagonal block plus the
/// off-block entries, with row scaling folded in.
#[derive(Debug, Clone)]
struct KluFactors<T> {
    /// The flag armed for the factorization, carried so `refactor` polls it too.
    interrupt: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    n: usize,
    nnz_a: usize,
    block_ptr: Vec<usize>,
    /// Final row permutation (new-to-old), pivoting included.
    row_perm: Vec<usize>,
    /// Inverse: original row -> final position.
    pinv: Vec<usize>,
    col_perm: Vec<usize>,
    /// Per-original-row reciprocal scale factor (all 1 when scaling is off).
    rs_inv: Vec<f64>,
    scaled: bool,
    /// Parallel per-block execution resolved for the FIRST factor (a-priori
    /// proxies of the work/concurrency principle; the refactor uses the exact
    /// plan in `par_refactor`/`pipelined` instead). Read by tests only.
    #[cfg_attr(not(test), allow(dead_code))]
    parallel: bool,
    /// L: strictly-below-diagonal entries per column, unit diagonal implicit.
    /// Row indices are final positions within the column's block (narrow
    /// [`Ki`] indices: the solve/refactor loops are index-bound).
    l_colptr: Vec<usize>,
    l_rowidx: Vec<Ki>,
    l_val: Vec<T>,
    /// U: strictly-above-diagonal within-block entries per column, stored in
    /// elimination (topological) order, the refactor replay order.
    u_colptr: Vec<usize>,
    u_rowidx: Vec<Ki>,
    u_val: Vec<T>,
    udiag: Vec<T>,
    /// Off-block entries (rows in earlier blocks, final positions), per
    /// column in the input's storage order. Not factored; applied in the
    /// block back-substitution.
    f_colptr: Vec<usize>,
    f_rowidx: Vec<Ki>,
    f_val: Vec<T>,
    /// Refactor scatter program, aligned with the input's storage order.
    /// For entry `k` of the pattern-frozen matrix: `scatter_expect[k]` is
    /// the final row position recorded at factor time (`pinv[row]`), the
    /// branch-free pattern check; `scatter_target[k]` encodes the value's
    /// destination: F slot `i` as `KI_FBIT | i`, else work-vector position.
    scatter_expect: Vec<Ki>,
    scatter_target: Vec<Ki>,
    /// Blocks admitted to the pipelined parallel refactor replay, with their
    /// Amdahl-bounded worker counts (empty when none qualifies or parallelism
    /// is off), plus the block-parallel refactor decision. Both come from the
    /// exact work/critical-path plan of [`compute_replay_plan`].
    pipelined: Vec<(usize, usize)>,
    par_refactor: bool,
}

/// KLU solver handle: factor (or analyze+factor), then solve / refactor.
#[derive(Debug, Clone)]
pub struct KluSolver<T> {
    factors: KluFactors<T>,
    diagnostics: crate::diagnostics::Diagnostics,
    /// Solve-phase accumulators (every `solve*` call records into them).
    solves: crate::diagnostics::SolveCounter,
    /// The solves are sequential (the determinism guarantee), so a Krylov
    /// solve preconditioned by this factor orthogonalizes on one worker.
    solve_threads: crate::numeric::settings::Threads,
}

fn pattern_mismatch() -> RslabError {
    RslabError::InvalidInput(
        "klu: matrix pattern does not match the symbolic analysis / stored factorization"
            .to_string(),
    )
}

/// Poll a caller-owned cancellation flag. One `Option` branch when unarmed.
#[inline]
pub(crate) fn interrupt_check(
    flag: Option<&std::sync::atomic::AtomicBool>,
) -> Result<(), RslabError> {
    match flag {
        Some(f) if f.load(std::sync::atomic::Ordering::Relaxed) => Err(RslabError::Interrupted),
        _ => Ok(()),
    }
}
