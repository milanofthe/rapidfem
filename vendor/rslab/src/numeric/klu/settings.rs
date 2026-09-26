//! The options of the KLU path.

use super::*;

/// Options for the KLU path. Defaults follow SuiteSparse KLU: threshold
/// partial pivoting with strong diagonal preference (`pivot_threshold =
/// 1e-3`), row-max scaling on, BTF on.
#[derive(Debug, Clone)]
pub struct KluSettings {
    /// Threshold for diagonal preference: the diagonal entry is taken as the
    /// pivot when `|a_jj| >= pivot_threshold * max_i |a_ij|` over the
    /// eligible column. `1.0` is plain partial pivoting; small values keep
    /// the BTF/AMD-chosen diagonal (less fill) unless it is numerically tiny.
    pub pivot_threshold: f64,
    /// Divide every row by its max-magnitude entry before factoring (and
    /// scale RHS/solution accordingly). Cheap and markedly more robust on
    /// badly row-equilibrated inputs.
    pub row_scaling: bool,
    /// Permute to block upper triangular form first. Disable only for
    /// experiments; without BTF the whole matrix is one block, structural
    /// singularity surfaces as a numeric zero pivot, and the diagonal
    /// preference loses its zero-free guarantee.
    pub btf: bool,
    /// Parallel per-block execution of factor and refactor over the
    /// (independent) BTF diagonal blocks, on the ambient rayon pool.
    /// **Bit-identical to sequential in every mode**: each block is factored
    /// sequentially by construction and blocks share no state, so the result
    /// does not depend on scheduling or thread count. The default `Auto`
    /// enables it through a deterministic structural gate (no implicit
    /// measuring): several diagonal blocks, `par_min_nnz` input nonzeros,
    /// and no dominant block (largest block at most half of `n`) - real
    /// circuits are often one giant irreducible block plus thousands of
    /// singletons, where distributing blocks cannot help.
    /// Run inside a bounded rayon pool to cap it for solver-in-the-loop use,
    /// or force `Off` for strictly sequential execution.
    pub parallel: KluParallel,
    /// Maximum-product row matching (MC64) as the transversal of the block
    /// triangular form: the matched, largest-product entries become the
    /// diagonal, so the diagonal-preference pivoting rarely has to leave
    /// it. The structural transversal only guarantees a zero-free
    /// diagonal; on the ibmpg1 power grid it leaves 14k of 45k columns to
    /// off-diagonal pivots and the fill at seven times the symbolic
    /// estimate. Analysis-time (value dependent); a `refactor` keeps the
    /// matching. Default `true`; needs `btf`.
    pub matching: bool,
    /// Nonzeros from which [`KluParallel::Auto`] factors blocks in parallel.
    /// Default `8000`.
    pub par_min_nnz: usize,
    /// Replay work (fmadd count) a unit of parallel refactorization must
    /// carry: below it the spawn and handoff overhead exceeds the overlap
    /// (on the SuiteSparse circuits scircuit at about 3e7 gains nothing,
    /// ASIC_100ks at 4e8 gains 2.7x). Default `5e7`.
    pub par_min_work: u64,
    /// Simultaneous work the structure must offer for a parallel refactor:
    /// across blocks `sum work / max block work`, inside a block the mean
    /// level width of its elimination DAG. Default `2`.
    pub par_min_ratio: f64,
    /// Caller-owned cancellation flag for the numeric phase, read and never
    /// written, polled at block boundaries and inside the pipelined refactor at
    /// column boundaries. The flag armed for a factorization is carried into
    /// the factors, so a later [`refactor`](KluSolver::refactor) on them
    /// observes the same flag. **Default `None`**.
    pub interrupt: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// Parallel per-block execution policy for the KLU path
/// (see [`KluSettings::parallel`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KluParallel {
    /// Structural gate: parallel when the BTF structure has several
    /// diagonal blocks, the matrix at least
    /// [`par_min_nnz`](KluSettings::par_min_nnz) nonzeros, and the largest
    /// block holds at most half of `n` (no dominant block).
    #[default]
    Auto,
    /// Always parallel (still bit-identical; blocks are independent).
    On,
    /// Strictly sequential.
    Off,
}

impl Default for KluSettings {
    fn default() -> Self {
        Self {
            pivot_threshold: 1e-3,
            row_scaling: true,
            btf: true,
            parallel: KluParallel::Auto,
            matching: true,
            par_min_nnz: 8_000,
            par_min_work: 50_000_000,
            par_min_ratio: 2.0,
            interrupt: None,
        }
    }
}

impl KluSettings {
    /// Builder: arm the numeric phase with a caller-owned cancellation flag.
    pub fn with_interrupt(mut self, flag: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.interrupt = Some(flag);
        self
    }

    /// Poll the caller's flag; `Ok(())` when unarmed or clear.
    #[inline]
    pub(crate) fn interrupted(&self) -> Result<(), RslabError> {
        interrupt_check(self.interrupt.as_deref())
    }

    /// Set the diagonal-preference threshold (see
    /// [`pivot_threshold`](Self::pivot_threshold)). `1.0` is plain partial
    /// pivoting.
    pub fn with_pivot_threshold(mut self, u: f64) -> Self {
        self.pivot_threshold = u;
        self
    }

    /// Composable toggle for row-max scaling
    /// (see [`row_scaling`](Self::row_scaling)).
    pub fn with_row_scaling(mut self, on: bool) -> Self {
        self.row_scaling = on;
        self
    }

    /// Enable or disable the MC64 row matching (see
    /// [`KluSettings::matching`]).
    pub fn with_matching(mut self, on: bool) -> Self {
        self.matching = on;
        self
    }

    /// Composable toggle for the BTF permutation (see [`btf`](Self::btf)).
    pub fn with_btf(mut self, on: bool) -> Self {
        self.btf = on;
        self
    }

    /// Composable setter for the parallel per-block policy
    /// (see [`parallel`](Self::parallel)).
    pub fn with_parallel(mut self, p: KluParallel) -> Self {
        self.parallel = p;
        self
    }
}
