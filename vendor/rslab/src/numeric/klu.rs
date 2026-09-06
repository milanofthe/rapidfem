//! KLU-style sparse LU: BTF + per-block left-looking Gilbert-Peierls.
//!
//! The third direct path next to the multifrontal LDL^T and LU, built for
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
//! Every phase is strictly sequential and allocation-deterministic, so results
//! are **bit-identical across runs and thread counts**, this path doubles as
//! the determinism arbiter for the parallel multifrontal paths.

use crate::error::RslabError;
use crate::numeric::ll_common::PanelPtr;
use crate::ordering::btf;
use crate::scalar::{fmadd, Scalar};
use crate::sparse::general::GeneralCsc;

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

/// Options for the KLU path. Defaults follow SuiteSparse KLU: threshold
/// partial pivoting with strong diagonal preference (`pivot_tol = 1e-3`),
/// row-max scaling on, BTF on.
#[derive(Debug, Clone)]
pub struct KluSettings {
    /// Threshold for diagonal preference: the diagonal entry is taken as the
    /// pivot when `|a_jj| >= pivot_tol * max_i |a_ij|` over the eligible
    /// column. `1.0` is plain partial pivoting; small values keep the
    /// BTF/AMD-chosen diagonal (less fill) unless it is numerically tiny.
    pub pivot_tol: f64,
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
    /// measuring): at least 4 diagonal blocks, 8000 input nonzeros, and no
    /// dominant block (largest block at most half of `n`) - real circuits
    /// are often one giant irreducible block plus thousands of singletons,
    /// where distributing blocks cannot help.
    /// Cap the pool with [`with_threads`](crate::with_threads) scoping for
    /// solver-in-the-loop use, or force `Off` for strictly sequential
    /// execution.
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
    /// Structural gate: parallel when the BTF structure has at least 4
    /// diagonal blocks, the matrix at least 8000 nonzeros, and the largest
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
            pivot_tol: 1e-3,
            row_scaling: true,
            btf: true,
            parallel: KluParallel::Auto,
            matching: true,
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

    /// Composable override of the diagonal-preference threshold
    /// (see [`pivot_tol`](Self::pivot_tol)). `1.0` is plain partial pivoting.
    pub fn with_pivot_tol(mut self, tol: f64) -> Self {
        self.pivot_tol = tol;
        self
    }

    /// Composable toggle for row-max scaling
    /// (see [`row_scaling`](Self::row_scaling)).
    pub fn with_row_scaling(mut self, on: bool) -> Self {
        self.row_scaling = on;
        self
    }

    /// Composable toggle for the BTF permutation (see [`btf`](Self::btf)).
    /// Enable or disable the MC64 row matching (see
    /// [`KluSettings::matching`]).
    pub fn with_matching(mut self, on: bool) -> Self {
        self.matching = on;
        self
    }

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

    /// Convenience toggle: `true` forces [`KluParallel::On`], `false`
    /// [`KluParallel::Off`]. The default policy is [`KluParallel::Auto`].
    pub fn with_parallel_factor(mut self, on: bool) -> Self {
        self.parallel = if on {
            KluParallel::On
        } else {
            KluParallel::Off
        };
        self
    }
}

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
/// diagonal and `pivot_tol` strongly prefers it). Threshold pivoting at factor
/// time can shift individual counts, not their order of magnitude.
#[derive(Debug, Clone, Copy)]
struct KluFill {
    l_nnz: u64,
    u_nnz: u64,
    f_nnz: u64,
    /// Gilbert-Peierls flop count (multiply-subtract pairs + divisions).
    flops: u64,
}

impl KluSymbolic {
    /// Analyze with default [`KluSettings`].
    pub fn analyze<T: Scalar>(a: &GeneralCsc<T>) -> Result<Self, RslabError> {
        Self::analyze_with(a, &KluSettings::default())
    }

    /// Analyze the pattern of `a`: BTF (unless disabled) + per-block AMD.
    ///
    /// Fails with [`RslabError::StructurallySingular`] when no complete
    /// matching exists (some set of `k` columns has entries in fewer than `k`
    /// rows), such a matrix is singular for *every* value assignment.
    pub fn analyze_with<T: Scalar>(
        a: &GeneralCsc<T>,
        settings: &KluSettings,
    ) -> Result<Self, RslabError> {
        a.validate()?;
        let n = a.n;
        // Same 31-bit range gate as factor time (the whole path is Ki-based;
        // checking here lets analyze and the BTF pass use narrow indices).
        if n as u64 >= KI_FBIT as u64 || a.nnz() as u64 >= KI_FBIT as u64 {
            return Err(RslabError::InvalidInput(
                "klu: dimension/nnz exceeds the 31-bit index range of this path".to_string(),
            ));
        }

        // Matching bakeoff (BTF on): deterministic maximum-matching
        // candidates (see `btf::matching_candidates`); with more than one,
        // each candidate's AMD-ordered blocks are scored by exact Cholesky
        // lnz (Gilbert-Ng-Peyton column counts on the ordered symmetrized
        // pattern) and the cheapest matching wins. Different maximum
        // matchings differ by several-x fill on the harmonic-balance class
        // (onetone/twotone), in matrix-dependent directions - measuring
        // beats guessing. The common MNA case (complete structural
        // diagonal) short-circuits to a single candidate and pays nothing.
        let weighted = if settings.btf && settings.matching {
            let cache = crate::scaling::mc64::compute_matching_general(a)?;
            (cache.n_matched == n).then_some(cache.perm)
        } else {
            None
        };
        let (pre_row_perm, col_perm, block_ptr) = if let Some(m) = weighted {
            // The weighted matching decides the transversal; the blocks are
            // its strongly connected components.
            let form = btf::btf_from_matching(n, &a.col_ptr, &a.row_idx, m);
            let of = order_blocks(a, form, false)?;
            (of.pre_row_perm, of.col_perm, of.block_ptr)
        } else if settings.btf {
            let cands = btf::matching_candidates(n, &a.col_ptr, &a.row_idx)
                .ok_or(RslabError::StructurallySingular)?;
            let score_it = cands.len() > 1;
            let mut best: Option<OrderedForm> = None;
            for m in cands {
                let form = btf::btf_from_matching(n, &a.col_ptr, &a.row_idx, m);
                let of = order_blocks(a, form, score_it)?;
                best = match best {
                    Some(b) if b.score <= of.score => Some(b),
                    _ => Some(of),
                };
            }
            let Some(of) = best else {
                // Unreachable: `matching_candidates` returns at least one
                // matching or `None` (handled above as StructurallySingular).
                return Err(RslabError::InvalidInput(
                    "klu: no matching candidate".to_string(),
                ));
            };
            (of.pre_row_perm, of.col_perm, of.block_ptr)
        } else {
            let ident: Vec<usize> = (0..n).collect();
            let bp = if n == 0 { vec![0] } else { vec![0, n] };
            let form = btf::BtfForm {
                row_perm: ident.clone(),
                col_perm: ident,
                block_ptr: bp,
            };
            let of = order_blocks(a, form, false)?;
            (of.pre_row_perm, of.col_perm, of.block_ptr)
        };

        // Freeze the analyzed pattern in the (final) pre-pivot space for the
        // a-priori estimators. Narrow inverse permutation: the gather is
        // bound by the random `pinv_pre[r]` reads.
        let mut pinv_pre = vec![0 as Ki; n];
        for (k, &r) in pre_row_perm.iter().enumerate() {
            pinv_pre[r] = k as Ki;
        }
        let mut pat_col_ptr = Vec::with_capacity(n + 1);
        let mut pat_row_idx = Vec::with_capacity(a.nnz());
        pat_col_ptr.push(0);
        for &c in &col_perm {
            for &r in &a.row_idx[a.col_ptr[c]..a.col_ptr[c + 1]] {
                pat_row_idx.push(pinv_pre[r] as usize);
            }
            pat_col_ptr.push(pat_row_idx.len());
        }

        Ok(Self {
            n,
            nnz: a.nnz(),
            pre_row_perm,
            col_perm,
            block_ptr,
            pat_col_ptr,
            pat_row_idx,
            fill: std::sync::OnceLock::new(),
        })
    }

    /// Symbolic Gilbert-Peierls pass over the stored pattern assuming
    /// diagonal pivots: exact per-path fill and flop counts, no values.
    /// Computed once and cached, the pass costs about as much as a numeric
    /// factor, so repeated `factor`/`estimate_memory` calls must not repay it.
    fn symbolic_fill(&self) -> KluFill {
        *self.fill.get_or_init(|| self.symbolic_fill_uncached())
    }

    fn symbolic_fill_uncached(&self) -> KluFill {
        let n = self.n;
        let mut stamp = vec![0usize; n];
        let mut node_stack = vec![0usize; n];
        let mut cur_stack = vec![0usize; n];
        let mut l_colptr = vec![0usize];
        let mut l_rowidx: Vec<usize> = Vec::new();
        let mut leaves: Vec<usize> = Vec::new();
        let (mut u_nnz, mut f_nnz, mut flops) = (0u64, 0u64, 0u64);

        for b in 0..self.block_ptr.len() - 1 {
            let (bs, be) = (self.block_ptr[b], self.block_ptr[b + 1]);
            for j in bs..be {
                let sj = j + 1;
                leaves.clear();
                for k in self.pat_col_ptr[j]..self.pat_col_ptr[j + 1] {
                    let pre = self.pat_row_idx[k];
                    if pre < bs {
                        f_nnz += 1;
                        continue;
                    }
                    if stamp[pre] == sj {
                        continue;
                    }
                    stamp[pre] = sj;
                    if pre >= j {
                        leaves.push(pre);
                        continue;
                    }
                    let mut d = 0usize;
                    node_stack[0] = pre;
                    cur_stack[0] = l_colptr[pre];
                    loop {
                        let u = node_stack[d];
                        let endp = l_colptr[u + 1];
                        let mut descended = false;
                        while cur_stack[d] < endp {
                            let ch = l_rowidx[cur_stack[d]];
                            cur_stack[d] += 1;
                            if stamp[ch] == sj {
                                continue;
                            }
                            stamp[ch] = sj;
                            if ch >= j {
                                leaves.push(ch);
                                continue;
                            }
                            d += 1;
                            node_stack[d] = ch;
                            cur_stack[d] = l_colptr[ch];
                            descended = true;
                            break;
                        }
                        if descended {
                            continue;
                        }
                        // u finished: one U entry, applying its L column.
                        u_nnz += 1;
                        flops += 2 * (l_colptr[u + 1] - l_colptr[u]) as u64;
                        if d == 0 {
                            break;
                        }
                        d -= 1;
                    }
                }
                for &lv in &leaves {
                    if lv != j {
                        l_rowidx.push(lv);
                    }
                }
                flops += (l_rowidx.len() - l_colptr[j]) as u64; // divisions
                l_colptr.push(l_rowidx.len());
            }
        }
        KluFill {
            l_nnz: l_rowidx.len() as u64,
            u_nnz,
            f_nnz,
            flops,
        }
    }

    /// Exact symbolic factor fill (`L` + `U` + diagonal + off-block entries)
    /// under the diagonal-pivoting assumption, the memory-backstop metric,
    /// mirroring [`LuSymbolic::symbolic_factor_nnz`](crate::LuSymbolic::symbolic_factor_nnz).
    pub fn symbolic_factor_nnz(&self) -> usize {
        let fill = self.symbolic_fill();
        (fill.l_nnz + fill.u_nnz + self.n as u64 + fill.f_nnz) as usize
    }

    /// **A-priori** memory/work estimate for factoring a matrix of scalar
    /// type `T` with this analysis, deterministic, computed from the stored
    /// pattern alone, mirroring [`LuSymbolic::estimate_memory`](crate::LuSymbolic::estimate_memory).
    ///
    /// KLU specifics: the fill is exact under diagonal pivoting (threshold
    /// pivoting can shift it slightly); `factor_flops` is the Gilbert-Peierls
    /// flop count (not the supernodal `nrow^2*ncol` proxy); the path is
    /// strictly sequential, so `critical_path_flops == factor_flops` and
    /// `max_tree_width == 1`; there are no dense panels.
    pub fn estimate_memory<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        let fill = self.symbolic_fill();
        let value_bytes = std::mem::size_of::<T>();
        let entry = (value_bytes + std::mem::size_of::<usize>()) as u64;
        let factor_nnz = fill.l_nnz + fill.u_nnz + self.n as u64 + fill.f_nnz;
        let factor_bytes = factor_nnz * entry;
        let input_bytes = self.nnz as u64 * entry;
        let workspace_bytes = self.n as u64 * (value_bytes as u64 + 4 * 8);
        crate::diagnostics::MemoryEstimate {
            value_bytes,
            factor_nnz,
            factor_bytes,
            panels_all_bytes: 0,
            panel_live_peak_bytes: 0,
            transient_peak_bytes: factor_bytes + input_bytes + workspace_bytes,
            mf_transient_peak_bytes: factor_bytes + input_bytes + workspace_bytes,
            factor_flops: fill.flops,
            critical_path_flops: fill.flops,
            max_tree_width: 1,
        }
    }

    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of diagonal blocks in the BTF form.
    pub fn n_blocks(&self) -> usize {
        self.block_ptr.len() - 1
    }

    /// Size of the largest diagonal block (the only part that generates fill).
    pub fn max_block_size(&self) -> usize {
        (0..self.n_blocks())
            .map(|b| self.block_ptr[b + 1] - self.block_ptr[b])
            .max()
            .unwrap_or(0)
    }

    /// Diagonal-block boundaries (`n_blocks + 1` entries).
    pub fn block_ptr(&self) -> &[usize] {
        &self.block_ptr
    }

    /// Numeric factorization of `a`, which must share the analyzed pattern.
    /// Populates the solver's [`diagnostics`](KluSolver::diagnostics) with the
    /// a-priori estimate and the measured factor stage (like
    /// [`LuSymbolic::factor`](crate::LuSymbolic::factor)); the one-shot
    /// [`KluSolver::factor`] skips both for minimum latency.
    pub fn factor<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        settings: &KluSettings,
    ) -> Result<KluSolver<T>, RslabError> {
        // Attach the a-priori estimate only when it has already been computed
        // (an explicit `estimate_memory` call): the estimator is a pattern-only
        // Gilbert-Peierls pass costing about as much as a numeric factor, and
        // factor() must not silently pay it - the solvers never measure (or
        // estimate) implicitly.
        let estimate = self.fill.get().map(|_| self.estimate_memory::<T>());
        let t = crate::clock::Instant::now();
        let factors = factor_impl(self, a, settings)?;
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        let nnz =
            (factors.l_val.len() + factors.u_val.len() + factors.udiag.len() + factors.f_val.len())
                as u64;
        let entry = (std::mem::size_of::<T>() + std::mem::size_of::<Ki>()) as u64;
        let mut diagnostics = crate::diagnostics::Diagnostics {
            threads: 1,
            n: a.n,
            nnz_a: a.row_idx.len() as u64,
            factor_nnz: nnz,
            estimate,
            decisions: crate::diagnostics::Decisions {
                ordering_requested: "Amd".to_string(),
                ordering_used: "Amd".to_string(),
                scaling: if settings.row_scaling {
                    "RowMaxAbs".to_string()
                } else {
                    "Identity".to_string()
                },
                method: "Klu".to_string(),
                btf_blocks: if settings.btf { self.n_blocks() } else { 0 },
                ..Default::default()
            },
            ..Default::default()
        };
        diagnostics.push(
            "klu-factor",
            factor_ms,
            diagnostics_flops(&diagnostics),
            nnz * entry,
        );
        if crate::logging::enabled(crate::logging::LogLevel::Info) {
            crate::logging::info(&format!("klu factor: {}", diagnostics.summary()));
        }
        Ok(KluSolver {
            factors,
            diagnostics,
            solves: Default::default(),
        })
    }
}

/// The GP flop count carried on the attached estimate (0 when absent).
fn diagnostics_flops(d: &crate::diagnostics::Diagnostics) -> u64 {
    d.estimate.as_ref().map_or(0, |e| e.factor_flops)
}

/// The two-parameter principle behind EVERY parallel decision on this path:
///
/// 1. **Work floor** ([`KLU_PAR_MIN_WORK`]): a unit of parallel execution
///    must carry at least this much replay work (fmadd count) - below it,
///    spawn/handoff overhead exceeds the overlap (measured on the SuiteSparse
///    circuit suite: scircuit at ~30 M gains nothing, ASIC_100ks at ~400 M
///    gains 2.7x). The floor is overhead physics; no setting bypasses it.
/// 2. **Concurrency ratio** ([`KLU_PAR_MIN_RATIO`]): parallelism engages only
///    where the structure offers at least this much simultaneous work.
///    Across BTF blocks that is the exact Amdahl bound `sum work / max block
///    work`; inside a block it is the mean level width of the frozen
///    elimination DAG - the average number of simultaneously replayable
///    columns. (A chain-work critical-path bound would be the "exact" ratio
///    but systematically underestimates the pipeline's just-in-time overlap:
///    ASIC_100ks scores below 2 on it yet measures 2.7x.)
const KLU_PAR_MIN_WORK: u64 = 50_000_000;
const KLU_PAR_MIN_RATIO: f64 = 2.0;

/// The replay-parallelism plan, computed once at factor time from the
/// pivot-final pattern: per-block replay work `W_b = sum_j sum_{p in U(:,j)}
/// |L(:,p)|` and elimination-DAG level structure.
///
/// A block is **pipelined** (NICSLU pipeline mode, Chen/Wang/Yang TCAD 2013)
/// iff `W_b` clears the work floor and its mean level width clears the
/// concurrency ratio; the worker count is bounded by that width (more
/// workers than simultaneously ready columns cannot help). The refactor runs
/// blocks **in parallel** iff the total clears the work floor (or the user
/// forced [`KluParallel::On`]) and no single block dominates.
fn compute_replay_plan(
    block_ptr: &[usize],
    l_colptr: &[usize],
    u_colptr: &[usize],
    u_rowidx: &[Ki],
    force: bool,
) -> (Vec<(usize, usize)>, bool) {
    let mut pipelined = Vec::new();
    let mut level: Vec<Ki> = Vec::new();
    let (mut total, mut max_w): (u64, u64) = (0, 0);
    for b in 0..block_ptr.len() - 1 {
        let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
        let bn = be - bs;
        level.clear();
        level.resize(bn, 0);
        let mut nlev: usize = 1;
        let mut w_b: u64 = 0;
        for j in bs..be {
            let mut l: Ki = 0;
            for &pk in &u_rowidx[u_colptr[j]..u_colptr[j + 1]] {
                let p = pk as usize;
                w_b += (l_colptr[p + 1] - l_colptr[p]) as u64;
                l = l.max(level[p - bs] + 1);
            }
            level[j - bs] = l;
            nlev = nlev.max(l as usize + 1);
        }
        total += w_b;
        max_w = max_w.max(w_b);
        let width = (bn as f64) / (nlev as f64);
        if w_b >= KLU_PAR_MIN_WORK && width >= KLU_PAR_MIN_RATIO {
            pipelined.push((b, (width as usize).max(2)));
        }
    }
    let ratio_ok = (total as f64) >= KLU_PAR_MIN_RATIO * (max_w as f64);
    let par_blocks = ratio_ok && (force || total >= KLU_PAR_MIN_WORK);
    (pipelined, par_blocks)
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

/// Row-max scaling reciprocals (1 for empty rows / scaling off).
fn row_scale_inv<T: Scalar>(a: &GeneralCsc<T>, enabled: bool) -> Vec<f64> {
    let mut rs = vec![0.0f64; a.n];
    if enabled {
        for (k, &i) in a.row_idx.iter().enumerate() {
            let m = a.values[k].magnitude();
            if m > rs[i] {
                rs[i] = m;
            }
        }
    }
    rs.iter()
        .map(|&m| {
            if m > 0.0 && m.is_finite() {
                1.0 / m
            } else {
                1.0
            }
        })
        .collect()
}

/// One matching candidate after SCC + per-block AMD, with the exact
/// Cholesky-lnz score of its ordered symmetrized blocks (only computed in a
/// multi-candidate bakeoff).
struct OrderedForm {
    pre_row_perm: Vec<usize>,
    col_perm: Vec<usize>,
    block_ptr: Vec<usize>,
    score: u64,
}

/// Per-block AMD on the symmetrized block pattern (B + B^T, with diagonal,
/// matching what the multifrontal paths feed rslab-amd), applied
/// symmetrically to the form's permutations. Blocks of size <= 2 have
/// nothing to reorder. With `score_it`, additionally accumulates the exact
/// Cholesky lnz of each AMD-ordered block pattern (Gilbert-Ng-Peyton column
/// counts, near-linear) as the bakeoff score - the trivial blocks are
/// identical across candidates and are skipped consistently.
fn order_blocks<T: Scalar>(
    a: &GeneralCsc<T>,
    form: btf::BtfForm,
    score_it: bool,
) -> Result<OrderedForm, RslabError> {
    let n = a.n;
    let btf::BtfForm {
        row_perm: mut pre_row_perm,
        mut col_perm,
        block_ptr,
    } = form;
    // Narrow inverse permutation: this stage is bound by the random
    // `pinv0[r]` lookups over the matrix entries; 32-bit halves the lookup
    // table's cache footprint (n < 2^31 is enforced at factor time and the
    // KLU design point is far below).
    let mut pinv0 = vec![0 as Ki; n];
    for (k, &r) in pre_row_perm.iter().enumerate() {
        pinv0[r] = k as Ki;
    }
    let mut score = 0u64;
    for b in 0..block_ptr.len() - 1 {
        let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
        let bn = be - bs;
        if bn <= 2 {
            continue;
        }
        // Symmetrized block adjacency (B + B^T + diagonal), canonical form
        // (sorted, deduplicated columns). Built as: the off-diagonal
        // in-block entries B column by column (sequential writes, one random
        // `pinv0` read per entry), a per-column sort of B's short columns,
        // B^T by counting transpose (whose columns come out sorted for
        // free), then a linear three-way sorted merge per column. One
        // random counting/scatter pass over the entries instead of the two
        // of the old both-directions scatter - the random writes, not the
        // short-column sorts, dominate this stage.
        // 1) B: off-diagonal in-block entries, block-local coordinates.
        // Single pass over the matrix (the random `pinv0` reads are this
        // stage's bottleneck - no separate counting pass), sequential
        // pushes, then a short per-column sort.
        let cap: usize = (bs..be)
            .map(|j| {
                let c = col_perm[j];
                a.col_ptr[c + 1] - a.col_ptr[c]
            })
            .sum();
        let mut bcol = Vec::with_capacity(bn + 1);
        bcol.push(0usize);
        let mut bri: Vec<i32> = Vec::with_capacity(cap);
        for lj in 0..bn {
            let c = col_perm[bs + lj];
            let seg = bri.len();
            for &r in &a.row_idx[a.col_ptr[c]..a.col_ptr[c + 1]] {
                let pre = pinv0[r] as usize;
                if pre >= bs && pre < be && pre - bs != lj {
                    bri.push((pre - bs) as i32);
                }
            }
            bri[seg..].sort_unstable();
            bcol.push(bri.len());
        }
        let m = bcol[bn];
        // 2) B^T via counting transpose; columns arrive sorted because the
        // source columns are visited in ascending order.
        let mut tcol = vec![0usize; bn + 1];
        for &li in &bri {
            tcol[li as usize + 1] += 1;
        }
        for j in 0..bn {
            tcol[j + 1] += tcol[j];
        }
        let mut tri = vec![0i32; m];
        {
            let mut cur = tcol[..bn].to_vec();
            for lj in 0..bn {
                for &li in &bri[bcol[lj]..bcol[lj + 1]] {
                    tri[cur[li as usize]] = lj as i32;
                    cur[li as usize] += 1;
                }
            }
        }
        // 3) Per-column sorted merge of B, B^T, and the diagonal, deduped.
        let mut colptr_i32 = Vec::with_capacity(bn + 1);
        let mut rowidx_i32 = Vec::with_capacity(2 * m + bn);
        colptr_i32.push(0i32);
        for lj in 0..bn {
            let (mut p, pe) = (bcol[lj], bcol[lj + 1]);
            let (mut q, qe) = (tcol[lj], tcol[lj + 1]);
            let d = lj as i32;
            let mut d_pending = true;
            let mut last = -1i32;
            while p < pe || q < qe || d_pending {
                let bv = if p < pe { bri[p] } else { i32::MAX };
                let tv = if q < qe { tri[q] } else { i32::MAX };
                let dv = if d_pending { d } else { i32::MAX };
                let v = bv.min(tv).min(dv);
                if v == bv {
                    p += 1;
                } else if v == tv {
                    q += 1;
                } else {
                    d_pending = false;
                }
                if v != last {
                    rowidx_i32.push(v);
                    last = v;
                }
            }
            colptr_i32.push(rowidx_i32.len() as i32);
        }
        let pat = rslab_ordering_core::CscPattern::new(bn, &colptr_i32, &rowidx_i32)
            .ok_or_else(|| RslabError::InvalidInput("klu: malformed block pattern".to_string()))?;
        let lperm = rslab_amd::amd_order(&pat)
            .map_err(|e| RslabError::InvalidInput(format!("klu: AMD ordering failed: {e:?}")))?;
        if score_it {
            // Exact Cholesky lnz of the AMD-ordered block: permute the full
            // symmetric pattern, then etree + GNP column counts. Both accept
            // a full symmetric pattern with unsorted columns (etree uses the
            // upper entries, GNP the lower).
            let mut newpos = vec![0usize; bn];
            for (k, &lp) in lperm.iter().enumerate() {
                newpos[lp as usize] = k;
            }
            let mut pcp = Vec::with_capacity(bn + 1);
            pcp.push(0usize);
            let mut pri = Vec::with_capacity(rowidx_i32.len());
            for &lp in lperm.iter() {
                let lp = lp as usize;
                for &r in &rowidx_i32[colptr_i32[lp] as usize..colptr_i32[lp + 1] as usize] {
                    pri.push(newpos[r as usize]);
                }
                pcp.push(pri.len());
            }
            let pat_p = crate::sparse::csc::CscPattern {
                n: bn,
                col_ptr: pcp,
                row_idx: pri,
            };
            let etree = crate::ordering::elimination_tree::EliminationTree::from_pattern(&pat_p);
            let cc = crate::symbolic::column_counts_gnp(&pat_p, &etree);
            score += crate::symbolic::total_factor_nnz(&cc) as u64;
        }
        // Apply the local (new-to-old) perm symmetrically to the block's
        // segment of both permutations.
        let old_rows: Vec<usize> = pre_row_perm[bs..be].to_vec();
        let old_cols: Vec<usize> = col_perm[bs..be].to_vec();
        for (i, &lp) in lperm.iter().enumerate() {
            pre_row_perm[bs + i] = old_rows[lp as usize];
            col_perm[bs + i] = old_cols[lp as usize];
        }
    }
    Ok(OrderedForm {
        pre_row_perm,
        col_perm,
        block_ptr,
        score,
    })
}

/// Per-block factor output in absolute position spaces: row indices of
/// `l/u` are absolute final positions, `f_pre` holds absolute *pre-pivot*
/// positions (rows of earlier blocks, resolved to final positions when the
/// blocks are spliced in order), and `prog_in`/`f_k` carry the refactor
/// scatter program contributions (see [`KluFactors`]).
struct BlockOut<T> {
    /// Absolute final position for each local pre position (`len == bn`).
    fin_abs: Vec<Ki>,
    l_colptr: Vec<usize>,
    l_rowidx: Vec<Ki>,
    l_val: Vec<T>,
    u_colptr: Vec<usize>,
    u_rowidx: Vec<Ki>,
    u_val: Vec<T>,
    udiag: Vec<T>,
    f_colptr: Vec<usize>,
    f_pre: Vec<Ki>,
    f_val: Vec<T>,
    /// Input-storage index `k` per F entry (aligned with `f_pre`/`f_val`).
    f_k: Vec<Ki>,
    /// `(k, absolute final position)` for every within-block entry.
    prog_in: Vec<(Ki, Ki)>,
}

// Manual `Default` (the derive would demand `T: Default`, which `Scalar`
// does not imply); every field is an empty `Vec`.
impl<T> Default for BlockOut<T> {
    fn default() -> Self {
        Self {
            fin_abs: Vec::new(),
            l_colptr: Vec::new(),
            l_rowidx: Vec::new(),
            l_val: Vec::new(),
            u_colptr: Vec::new(),
            u_rowidx: Vec::new(),
            u_val: Vec::new(),
            udiag: Vec::new(),
            f_colptr: Vec::new(),
            f_pre: Vec::new(),
            f_val: Vec::new(),
            f_k: Vec::new(),
            prog_in: Vec::new(),
        }
    }
}

impl<T: Scalar> BlockOut<T> {
    /// Clear for a new block of size `bn` with `annz` input nonzeros,
    /// keeping allocations (the sequential driver reuses one buffer across
    /// all blocks; per-block buffers made the allocator dominate on the
    /// tens-of-thousands-of-singletons circuit class). Reserves follow the
    /// MNA reference class (~6x input fill, 4x reserve caps reallocation at
    /// one doubling without over-committing on low-fill classes).
    fn reset(&mut self, bn: usize, annz: usize) {
        self.fin_abs.clear();
        self.fin_abs.resize(bn, KI_UNSET);
        self.l_colptr.clear();
        self.l_colptr.push(0);
        self.l_rowidx.clear();
        self.l_rowidx.reserve(annz * 4);
        self.l_val.clear();
        self.l_val.reserve(annz * 4);
        self.u_colptr.clear();
        self.u_colptr.push(0);
        self.u_rowidx.clear();
        self.u_rowidx.reserve(annz * 2);
        self.u_val.clear();
        self.u_val.reserve(annz * 2);
        self.udiag.clear();
        self.udiag.reserve(bn);
        self.f_colptr.clear();
        self.f_colptr.push(0);
        self.f_pre.clear();
        self.f_val.clear();
        self.f_k.clear();
        self.prog_in.clear();
        self.prog_in.reserve(annz);
    }
}

/// Reusable per-worker scratch for [`factor_block`], sized once to the
/// largest block (SuiteSparse KLU's workspace discipline). Real circuit
/// matrices carry tens of thousands of tiny BTF blocks; allocating the DFS
/// state per block makes the allocator the dominant factor cost there.
///
/// `x` relies on the kernel invariant that every scattered value is zeroed
/// when consumed, so it is all-zero between blocks on the success path. A
/// block that fails mid-column leaves `x` dirty, which is safe: the error
/// aborts the whole factor, so no later block's output survives.
struct KluScratch<T> {
    mark: Vec<[Ki; 2]>,
    lpend: Vec<Ki>,
    x: Vec<T>,
    node_stack: Vec<Ki>,
    cur_stack: Vec<usize>,
    topo: Vec<Ki>,
    nonpiv: Vec<Ki>,
}

impl<T: Scalar> KluScratch<T> {
    fn new(max_bn: usize) -> Self {
        Self {
            mark: vec![[0, KI_UNSET]; max_bn],
            lpend: vec![KI_UNSET; max_bn],
            x: vec![T::zero(); max_bn],
            node_stack: vec![0 as Ki; max_bn],
            cur_stack: vec![0usize; max_bn],
            topo: Vec::with_capacity(max_bn),
            nonpiv: Vec::with_capacity(max_bn),
        }
    }

    /// Reset the per-block state for a block of size `bn` (cheap memsets;
    /// `x` is already zero by the kernel invariant, `topo`/`nonpiv` are
    /// cleared per column).
    fn reset(&mut self, bn: usize) {
        self.mark[..bn].fill([0, KI_UNSET]);
        self.lpend[..bn].fill(KI_UNSET);
    }
}

/// Factor one diagonal block in block-local space. Strictly sequential and
/// deterministic; the parallel driver runs one worker per block, which is
/// bit-identical to the sequential order because blocks share no state.
// The argument list mirrors the phase inputs (symbolic, matrix, settings,
// scaling, permutation, block range, workspaces); a bundling struct would
// only rename the same nine things.
#[allow(clippy::too_many_arguments)]
fn factor_block<T: Scalar>(
    sym: &KluSymbolic,
    a: &GeneralCsc<T>,
    settings: &KluSettings,
    rs_inv: &[f64],
    pinv_pre: &[Ki],
    bs: usize,
    be: usize,
    scratch: &mut KluScratch<T>,
    out: &mut BlockOut<T>,
) -> Result<(), RslabError> {
    let bn = be - bs;

    if bn == 1 {
        // Singleton block: the pivot is the (structurally nonzero) diagonal
        // entry itself; everything else in the column is off-block. Cleared
        // by hand (not `reset`) to skip its fill reserves.
        let c = sym.col_perm[bs];
        out.reset(1, 0);
        out.fin_abs[0] = bs as Ki;
        out.l_colptr.push(0);
        out.u_colptr.push(0);
        let mut diag: Option<T> = None;
        for k in a.col_ptr[c]..a.col_ptr[c + 1] {
            let pre = pinv_pre[a.row_idx[k]] as usize;
            let sv = a.values[k] * T::from_real(rs_inv[a.row_idx[k]]);
            if pre == bs {
                diag = Some(sv);
                out.prog_in.push((k as Ki, bs as Ki));
            } else if pre < bs {
                out.f_pre.push(pre as Ki);
                out.f_val.push(sv);
                out.f_k.push(k as Ki);
            } else {
                return Err(pattern_mismatch());
            }
        }
        let d = diag.ok_or(RslabError::SingularBasis { column: c })?;
        if d.magnitude() == 0.0 || !d.is_finite() {
            return Err(RslabError::SingularBasis { column: c });
        }
        out.udiag.push(d);
        out.f_colptr.push(out.f_pre.len());
        return Ok(());
    }

    // General irreducible block: left-looking Gilbert-Peierls, block-local
    // position space (`lb = pre - bs`), DFS state borrowed from the reusable
    // per-worker scratch.
    // `mark[lb] = [dfs_stamp, local_final_position]` packed pair; the DFS
    // touches both fields per visited node, packing halves its random
    // cache-line traffic.
    // `lpend`: symmetric pruning (Eisenstat & Liu, SIMAX 1992): once column
    // `s` has a symmetric pivot pair (`U(s,k) != 0` and `L(k,s) != 0`), the
    // DFS only needs the prefix of `L(:,s)` holding rows already pivotal at
    // step `k`; every pruned row is covered through column `k`'s pattern.
    // `lpend[s]` is the exclusive prefix end (KI_UNSET = unpruned).
    scratch.reset(bn);
    // Slice borrows (not `&mut Vec`) so the hot loops index through one
    // level of indirection, exactly like the previous per-block locals.
    let mark = scratch.mark.as_mut_slice();
    let lpend = scratch.lpend.as_mut_slice();
    let x = scratch.x.as_mut_slice();
    let node_stack = scratch.node_stack.as_mut_slice();
    let cur_stack = scratch.cur_stack.as_mut_slice();
    let topo = &mut scratch.topo;
    let nonpiv = &mut scratch.nonpiv;
    // Diagonal tracking through off-diagonal pivots (SuiteSparse KLU's
    // repair): `diag_row[lk]` is the block-local row currently assigned as
    // column `lk`'s diagonal candidate, `diag_col[lr]` its inverse. When a
    // pivot steals another column's diagonal row, the displaced row is
    // reassigned to that column, so the zero-free matching survives and one
    // off-diagonal pivot cannot cascade into unpivoted diagonals (and fill
    // blow-up) for the rest of the block. Materialized lazily on the first
    // off-diagonal pivot; until then both maps are the identity.
    let mut diag_row: Vec<Ki> = Vec::new();
    let mut diag_col: Vec<Ki> = Vec::new();

    // Reserve from the block's input nnz: the MNA reference class fills
    // ~6x its input, so 4x reserves cap reallocation at one doubling in the
    // common case without over-committing memory on low-fill classes.
    let annz: usize = (bs..be)
        .map(|j| {
            let c = sym.col_perm[j];
            a.col_ptr[c + 1] - a.col_ptr[c]
        })
        .sum();
    out.reset(bn, annz);
    // `fin_local[lb]` tracked inside `mark[lb][1]`; `out.fin_abs` filled at
    // the end from it.

    for lj in 0..bn {
        let j = bs + lj;
        let c = sym.col_perm[j];
        let sj = (lj + 1) as Ki; // unique DFS stamp for this column
        topo.clear();
        nonpiv.clear();

        // Pass 1, symbolic: DFS the reach of the column's within-block
        // pattern over the L columns factored so far. Pivotal nodes come
        // out in `topo` post-order; non-pivotal nodes (pivot candidates)
        // in `nonpiv`. The inner loop runs unchecked: every index is a
        // block-local position `< bn` (`debug_assert`ed), and `cur/end` are
        // positions into the already-built prefix of `l_rowidx`.
        let col_end = |p: usize, lpend: &[Ki], l_colptr: &[usize]| -> usize {
            let lp = lpend[p];
            if lp == KI_UNSET {
                l_colptr[p + 1]
            } else {
                lp as usize
            }
        };
        for k in a.col_ptr[c]..a.col_ptr[c + 1] {
            let pre = pinv_pre[a.row_idx[k]] as usize;
            if pre < bs {
                continue; // off-block, handled in pass 2
            }
            if pre >= be {
                return Err(pattern_mismatch());
            }
            let lb = pre - bs;
            let m = mark[lb];
            if m[0] == sj {
                continue;
            }
            mark[lb][0] = sj;
            if m[1] == KI_UNSET {
                nonpiv.push(lb as Ki);
                continue;
            }
            let mut d = 0usize;
            node_stack[0] = lb as Ki;
            cur_stack[0] = out.l_colptr[m[1] as usize];
            let mut end = col_end(m[1] as usize, lpend, &out.l_colptr);
            loop {
                let mut descended = false;
                while cur_stack[d] < end {
                    debug_assert!(cur_stack[d] < out.l_rowidx.len());
                    let ch = unsafe { *out.l_rowidx.get_unchecked(cur_stack[d]) } as usize;
                    cur_stack[d] += 1;
                    debug_assert!(ch < bn);
                    let mch = unsafe { *mark.get_unchecked(ch) };
                    if mch[0] == sj {
                        continue;
                    }
                    unsafe { mark.get_unchecked_mut(ch)[0] = sj };
                    if mch[1] == KI_UNSET {
                        nonpiv.push(ch as Ki);
                        continue;
                    }
                    d += 1;
                    node_stack[d] = ch as Ki;
                    let p = mch[1] as usize;
                    cur_stack[d] = out.l_colptr[p];
                    end = col_end(p, lpend, &out.l_colptr);
                    descended = true;
                    break;
                }
                if descended {
                    continue;
                }
                topo.push(node_stack[d]);
                if d == 0 {
                    break;
                }
                d -= 1;
                let up = mark[node_stack[d] as usize][1] as usize;
                end = col_end(up, lpend, &out.l_colptr);
            }
        }

        // Pass 2, scatter the scaled column values (off-block entries go
        // straight to F with their pre positions; final positions are
        // resolved when the earlier blocks are spliced).
        for k in a.col_ptr[c]..a.col_ptr[c + 1] {
            let r = a.row_idx[k];
            let pre = pinv_pre[r] as usize;
            let sv = a.values[k] * T::from_real(rs_inv[r]);
            if pre < bs {
                out.f_pre.push(pre as Ki);
                out.f_val.push(sv);
                out.f_k.push(k as Ki);
            } else {
                x[pre - bs] = sv;
                // Refactor scatter program, local target for now; translated
                // to the absolute final position once the block's pivot
                // sequence is complete (saves a second full matrix scan).
                out.prog_in.push((k as Ki, (pre - bs) as Ki));
            }
        }

        let u_start = out.u_rowidx.len();

        // Pass 3, numeric update in topological order (reverse post-order):
        // each pivotal node's final value feeds its L column into the
        // remaining work vector, and becomes a U entry. The axpy runs
        // through `fmadd` (FMA on native builds); `refactor`'s replay uses
        // the identical expression so it stays bit-identical. Unchecked: all
        // row indices are local positions `< bn` by construction.
        for &u in topo.iter().rev() {
            let u = u as usize;
            let p = mark[u][1];
            let xu = x[u];
            x[u] = T::zero();
            out.u_rowidx.push(p);
            out.u_val.push(xu);
            let nxu = T::zero() - xu;
            let p = p as usize;
            for k in out.l_colptr[p]..out.l_colptr[p + 1] {
                debug_assert!(k < out.l_rowidx.len());
                let lr = unsafe { *out.l_rowidx.get_unchecked(k) } as usize;
                debug_assert!(lr < bn);
                unsafe {
                    *x.get_unchecked_mut(lr) =
                        fmadd(nxu, *out.l_val.get_unchecked(k), *x.get_unchecked(lr));
                }
            }
        }

        // Pivot: max-magnitude candidate, overridden by the diagonal
        // (local pre position `lj`) when it clears the threshold.
        let mut piv = UNSET;
        let mut maxmag = 0.0f64;
        for &np in nonpiv.iter() {
            let m = x[np as usize].magnitude();
            if m > maxmag {
                maxmag = m;
                piv = np as usize;
            }
        }
        if piv == UNSET || maxmag == 0.0 || !maxmag.is_finite() {
            return Err(RslabError::SingularBasis { column: c });
        }
        let d = if diag_row.is_empty() {
            lj
        } else {
            diag_row[lj] as usize
        };
        if mark[d][0] == sj && mark[d][1] == KI_UNSET {
            let dm = x[d].magnitude();
            if dm > 0.0 && dm >= settings.pivot_tol * maxmag {
                piv = d;
            }
        }
        if piv != d {
            // Off-diagonal pivot: hand the displaced diagonal row `d` to the
            // (necessarily unprocessed) column that had `piv` as its
            // diagonal. Processed columns' assigned rows are always pivotal,
            // so `d` is free and the reassigned matching stays zero-free.
            if diag_row.is_empty() {
                diag_row = (0..bn as Ki).collect();
                diag_col = (0..bn as Ki).collect();
            }
            debug_assert!(mark[d][1] == KI_UNSET);
            let k2 = diag_col[piv];
            if k2 != KI_UNSET {
                diag_row[k2 as usize] = d as Ki;
                diag_col[d] = k2;
            }
            diag_col[piv] = KI_UNSET;
        }

        let dval = x[piv];
        x[piv] = T::zero();
        mark[piv][1] = lj as Ki;
        out.udiag.push(dval);
        for &np in nonpiv.iter() {
            let np = np as usize;
            if np == piv {
                continue;
            }
            // Keep structural zeros: the pattern must be value-independent
            // for the refactor replay.
            out.l_rowidx.push(np as Ki);
            out.l_val.push(x[np] / dval);
            x[np] = T::zero();
        }
        out.l_colptr.push(out.l_rowidx.len());
        out.u_colptr.push(out.u_rowidx.len());
        out.f_colptr.push(out.f_pre.len());

        // Symmetric pruning for this pivot: for each U-partner column `s`
        // (an entry `U(s,j)`), if `L(:,s)` contains the pivot row, partition
        // it so rows already pivotal come first and bound the future DFS
        // scans to that prefix.
        let pivk = piv as Ki;
        for ui in u_start..out.u_rowidx.len() {
            let s = out.u_rowidx[ui] as usize;
            if lpend[s] != KI_UNSET {
                continue;
            }
            let (cs, ce) = (out.l_colptr[s], out.l_colptr[s + 1]);
            if !out.l_rowidx[cs..ce].contains(&pivk) {
                continue;
            }
            let mut head = cs;
            for k in cs..ce {
                let r = out.l_rowidx[k] as usize;
                if mark[r][1] != KI_UNSET {
                    out.l_rowidx.swap(head, k);
                    out.l_val.swap(head, k);
                    head += 1;
                }
            }
            lpend[s] = head as Ki;
        }
    }

    // Block fully pivoted: resolve to absolute final positions. L row
    // indices go local-pre -> absolute final; U row indices go local-final
    // -> absolute final; the scatter program records absolute finals for
    // every within-block entry.
    for m in mark[..bn].iter() {
        debug_assert_ne!(m[1], KI_UNSET);
    }
    for (lb, m) in mark[..bn].iter().enumerate() {
        out.fin_abs[lb] = (bs as Ki) + m[1];
    }
    for ri in out.l_rowidx.iter_mut() {
        *ri = out.fin_abs[*ri as usize];
    }
    for ri in out.u_rowidx.iter_mut() {
        *ri += bs as Ki;
    }
    for e in out.prog_in.iter_mut() {
        e.1 = out.fin_abs[e.1 as usize];
    }
    Ok(())
}

fn factor_impl<T: Scalar>(
    sym: &KluSymbolic,
    a: &GeneralCsc<T>,
    settings: &KluSettings,
) -> Result<KluFactors<T>, RslabError> {
    a.validate()?;
    let n = sym.n;
    if a.n != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: a.n,
        });
    }
    if a.nnz() != sym.nnz {
        return Err(pattern_mismatch());
    }
    if n as u64 >= KI_FBIT as u64 || sym.nnz as u64 >= KI_FBIT as u64 {
        return Err(RslabError::InvalidInput(
            "klu: dimension/nnz exceeds the 31-bit index range of this path".to_string(),
        ));
    }

    let rs_inv = row_scale_inv(a, settings.row_scaling);
    let mut pinv_pre = vec![0 as Ki; n];
    for (k, &r) in sym.pre_row_perm.iter().enumerate() {
        pinv_pre[r] = k as Ki;
    }

    // Factor the diagonal blocks: independent by construction, so the
    // parallel driver is bit-identical to the sequential one (each worker is
    // sequential inside; nothing is shared across blocks). Parallelism is
    // opt-in and runs on the ambient rayon pool, so callers cap it with
    // `with_threads` scoping, matching the solver-in-the-loop contract.
    let nblocks = sym.block_ptr.len() - 1;
    // Resolve the FIRST-factor parallel policy from a-priori structure. The
    // exact work plan needs the pivot-final pattern, so this is the same
    // work-floor/Amdahl principle on its a-priori proxies: `nnz` as the work
    // floor, `no block holds half the matrix` as the block-level Amdahl
    // ratio. The refactor decision is replaced by the exact plan
    // ([`compute_replay_plan`]) once the pattern is frozen.
    let parallel = match settings.parallel {
        KluParallel::On => true,
        KluParallel::Off => false,
        KluParallel::Auto => sym.nnz >= 8_000 && sym.max_block_size() * 2 <= sym.n,
    } && nblocks > 1;
    let max_bn = sym.max_block_size();

    // Numeric output buffers, filled either incrementally block-by-block
    // (sequential: one reused scratch + one reused block buffer, no
    // per-block allocations and no separate splice pass - the allocator and
    // the extra copy dominated on the tens-of-thousands-of-tiny-blocks
    // circuit class) or by the two-phase parallel splice below.
    let mut l_colptr: Vec<usize> = Vec::with_capacity(n + 1);
    l_colptr.push(0);
    let mut u_colptr: Vec<usize> = Vec::with_capacity(n + 1);
    u_colptr.push(0);
    let mut f_colptr: Vec<usize> = Vec::with_capacity(n + 1);
    f_colptr.push(0);
    let mut udiag: Vec<T> = Vec::with_capacity(n);
    let mut l_rowidx: Vec<Ki> = Vec::new();
    let mut l_val: Vec<T> = Vec::new();
    let mut u_rowidx: Vec<Ki> = Vec::new();
    let mut u_val: Vec<T> = Vec::new();
    let mut f_rowidx: Vec<Ki> = Vec::new();
    let mut f_val: Vec<T> = Vec::new();
    let mut scatter_expect = vec![0 as Ki; sym.nnz];
    let mut scatter_target = vec![0 as Ki; sym.nnz];
    let mut fin_of_pre = vec![0 as Ki; n];

    if !parallel {
        // Same fill heuristics as the per-block reserves (MNA class ~6x
        // input): caps the doubling-growth copies of the append at one.
        // (The parallel branch replaces these vectors with exact-size
        // allocations instead, so the reserves live here.)
        l_rowidx.reserve(sym.nnz * 4);
        l_val.reserve(sym.nnz * 4);
        u_rowidx.reserve(sym.nnz * 2);
        u_val.reserve(sym.nnz * 2);
        f_rowidx.reserve(sym.nnz / 4);
        f_val.reserve(sym.nnz / 4);
        let mut scratch = KluScratch::<T>::new(max_bn);
        let mut out = BlockOut::<T>::default();
        for b in 0..nblocks {
            settings.interrupted()?;
            let bs = sym.block_ptr[b];
            let be = sym.block_ptr[b + 1];
            factor_block(
                sym,
                a,
                settings,
                &rs_inv,
                &pinv_pre,
                bs,
                be,
                &mut scratch,
                &mut out,
            )?;
            // Incremental append: F rows reference earlier blocks only,
            // whose final positions are already in `fin_of_pre`.
            fin_of_pre[bs..be].copy_from_slice(&out.fin_abs);
            let (lo, uo, fo) = (l_rowidx.len(), u_rowidx.len(), f_rowidx.len());
            l_rowidx.extend_from_slice(&out.l_rowidx);
            l_val.extend_from_slice(&out.l_val);
            u_rowidx.extend_from_slice(&out.u_rowidx);
            u_val.extend_from_slice(&out.u_val);
            f_rowidx.extend(out.f_pre.iter().map(|&pre| fin_of_pre[pre as usize]));
            f_val.extend_from_slice(&out.f_val);
            l_colptr.extend(out.l_colptr[1..].iter().map(|&p| lo + p));
            u_colptr.extend(out.u_colptr[1..].iter().map(|&p| uo + p));
            f_colptr.extend(out.f_colptr[1..].iter().map(|&p| fo + p));
            udiag.extend_from_slice(&out.udiag);
            for (i, &k) in out.f_k.iter().enumerate() {
                scatter_expect[k as usize] = f_rowidx[fo + i];
                scatter_target[k as usize] = KI_FBIT | (fo + i) as Ki;
            }
            for &(k, fin) in &out.prog_in {
                scatter_expect[k as usize] = fin;
                scatter_target[k as usize] = fin;
            }
        }
    } else {
        use rayon::prelude::*;
        let blocks: Vec<Result<BlockOut<T>, RslabError>> = (0..nblocks)
            .into_par_iter()
            .map_init(
                || KluScratch::<T>::new(max_bn),
                |scratch, b| {
                    settings.interrupted()?;
                    let mut out = BlockOut::<T>::default();
                    factor_block(
                        sym,
                        a,
                        settings,
                        &rs_inv,
                        &pinv_pre,
                        sym.block_ptr[b],
                        sym.block_ptr[b + 1],
                        scratch,
                        &mut out,
                    )?;
                    Ok(out)
                },
            )
            .collect();

        // Splice the per-block outputs. Two phases: exact-size global arrays are
        // carved into disjoint per-block chunks (offsets by prefix sums) and the
        // heavy value/index copies run per block, in parallel when enabled; the
        // cheap O(n)/O(nnz) bookkeeping (column pointers, the refactor scatter
        // program, `udiag`) stays sequential. Off-block F rows are resolved
        // through `fin_of_pre`, which is complete before the copy phase starts.
        let outs: Vec<BlockOut<T>> = {
            let mut v = Vec::with_capacity(nblocks);
            for o in blocks {
                v.push(o?);
            }
            v
        };
        let mut l_off = vec![0usize; nblocks + 1];
        let mut u_off = vec![0usize; nblocks + 1];
        let mut f_off = vec![0usize; nblocks + 1];
        for (b, out) in outs.iter().enumerate() {
            l_off[b + 1] = l_off[b] + out.l_rowidx.len();
            u_off[b + 1] = u_off[b] + out.u_rowidx.len();
            f_off[b + 1] = f_off[b] + out.f_pre.len();
        }

        for (b, out) in outs.iter().enumerate() {
            let bs = sym.block_ptr[b];
            fin_of_pre[bs..bs + out.fin_abs.len()].copy_from_slice(&out.fin_abs);
        }

        l_rowidx = vec![0; l_off[nblocks]];
        l_val = vec![T::zero(); l_off[nblocks]];
        u_rowidx = vec![0; u_off[nblocks]];
        u_val = vec![T::zero(); u_off[nblocks]];
        f_rowidx = vec![0; f_off[nblocks]];
        f_val = vec![T::zero(); f_off[nblocks]];
        {
            struct Chunks<'s, T> {
                l_ri: &'s mut [Ki],
                l_v: &'s mut [T],
                u_ri: &'s mut [Ki],
                u_v: &'s mut [T],
                f_ri: &'s mut [Ki],
                f_v: &'s mut [T],
            }
            let mut jobs: Vec<(Chunks<'_, T>, &BlockOut<T>)> = Vec::with_capacity(nblocks);
            let (mut lri, mut lv) = (l_rowidx.as_mut_slice(), l_val.as_mut_slice());
            let (mut uri, mut uv) = (u_rowidx.as_mut_slice(), u_val.as_mut_slice());
            let (mut fri, mut fv) = (f_rowidx.as_mut_slice(), f_val.as_mut_slice());
            for out in &outs {
                let (a1, r1) = std::mem::take(&mut lri).split_at_mut(out.l_rowidx.len());
                lri = r1;
                let (a2, r2) = std::mem::take(&mut lv).split_at_mut(out.l_val.len());
                lv = r2;
                let (a3, r3) = std::mem::take(&mut uri).split_at_mut(out.u_rowidx.len());
                uri = r3;
                let (a4, r4) = std::mem::take(&mut uv).split_at_mut(out.u_val.len());
                uv = r4;
                let (a5, r5) = std::mem::take(&mut fri).split_at_mut(out.f_pre.len());
                fri = r5;
                let (a6, r6) = std::mem::take(&mut fv).split_at_mut(out.f_val.len());
                fv = r6;
                jobs.push((
                    Chunks {
                        l_ri: a1,
                        l_v: a2,
                        u_ri: a3,
                        u_v: a4,
                        f_ri: a5,
                        f_v: a6,
                    },
                    out,
                ));
            }
            let fin = &fin_of_pre;
            let copy_one = |(c, out): (Chunks<'_, T>, &BlockOut<T>)| {
                c.l_ri.copy_from_slice(&out.l_rowidx);
                c.l_v.copy_from_slice(&out.l_val);
                c.u_ri.copy_from_slice(&out.u_rowidx);
                c.u_v.copy_from_slice(&out.u_val);
                for (dst, &pre) in c.f_ri.iter_mut().zip(&out.f_pre) {
                    *dst = fin[pre as usize];
                }
                c.f_v.copy_from_slice(&out.f_val);
            };
            jobs.into_par_iter().for_each(copy_one);
        }

        for (b, out) in outs.iter().enumerate() {
            l_colptr.extend(out.l_colptr[1..].iter().map(|&p| l_off[b] + p));
            u_colptr.extend(out.u_colptr[1..].iter().map(|&p| u_off[b] + p));
            f_colptr.extend(out.f_colptr[1..].iter().map(|&p| f_off[b] + p));
            udiag.extend_from_slice(&out.udiag);
            for (i, &k) in out.f_k.iter().enumerate() {
                let k = k as usize;
                scatter_expect[k] = f_rowidx[f_off[b] + i];
                scatter_target[k] = KI_FBIT | (f_off[b] + i) as Ki;
            }
            for &(k, fin) in &out.prog_in {
                scatter_expect[k as usize] = fin;
                scatter_target[k as usize] = fin;
            }
        }
    }

    let mut row_perm = vec![0usize; n];
    let mut pinv = vec![0usize; n];
    for (&fin, &orig) in fin_of_pre.iter().zip(&sym.pre_row_perm) {
        row_perm[fin as usize] = orig;
        pinv[orig] = fin as usize;
    }

    // Exact replay-parallelism plan from the now-frozen pattern: which blocks
    // pipeline internally (and with how many workers), and whether the blocks
    // themselves run in parallel. One work/Amdahl principle for both, see
    // [`compute_replay_plan`].
    let (pipelined, par_refactor) = if settings.parallel == KluParallel::Off {
        (Vec::new(), false)
    } else {
        compute_replay_plan(
            &sym.block_ptr,
            &l_colptr,
            &u_colptr,
            &u_rowidx,
            settings.parallel == KluParallel::On,
        )
    };

    Ok(KluFactors {
        interrupt: settings.interrupt.clone(),
        n,
        nnz_a: sym.nnz,
        block_ptr: sym.block_ptr.clone(),
        row_perm,
        pinv,
        col_perm: sym.col_perm.clone(),
        rs_inv,
        scaled: settings.row_scaling,
        parallel,
        l_colptr,
        l_rowidx,
        l_val,
        u_colptr,
        u_rowidx,
        u_val,
        udiag,
        f_colptr,
        f_rowidx,
        f_val,
        scatter_expect,
        scatter_target,
        pipelined,
        par_refactor,
    })
}

impl<T: Scalar> KluSolver<T> {
    /// One-shot analyze + factor with the given settings. Skips the a-priori
    /// estimate and stage timing (empty [`diagnostics`](Self::diagnostics)),
    /// like [`LuSolver::factor`](crate::LuSolver::factor); use the phased
    /// [`KluSymbolic::factor`] for populated diagnostics.
    pub fn factor(a: &GeneralCsc<T>, settings: &KluSettings) -> Result<Self, RslabError> {
        // Through the symbolic object, so the diagnostics are filled the same
        // way as on the analyze-once path (the former direct call returned
        // an empty `Diagnostics`).
        KluSymbolic::analyze_with(a, settings)?.factor(a, settings)
    }

    /// Per-call diagnostics: measured factor/refactor stages, fill, and the
    /// a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate).
    /// Populated by the phased [`KluSymbolic::factor`]; empty for the
    /// one-shot [`factor`](Self::factor).
    /// Everything this factorization can tell about itself (see
    /// [`Diagnostics`](crate::Diagnostics)), the solve-phase accumulators
    /// included. A snapshot.
    pub fn diagnostics(&self) -> crate::diagnostics::Diagnostics {
        let mut d = self.diagnostics.clone();
        d.solves = self.solves.snapshot();
        d
    }

    fn record_solve(&self, rhs: usize, t: crate::clock::Instant, refine_steps: usize) {
        let ms = t.elapsed().as_secs_f64() * 1e3;
        self.solves.record(rhs, ms, refine_steps);
        if crate::logging::enabled(crate::logging::LogLevel::Debug) {
            crate::logging::debug(&format!(
                "klu solve: n={} rhs={rhs} refine_steps={refine_steps} {ms:.3} ms",
                self.factors.n
            ));
        }
    }

    /// Thread policy the solve phase should honour: the KLU path is strictly
    /// sequential (that is its determinism guarantee), so this is always a
    /// fixed single-worker budget.
    pub fn solve_thread_policy(&self) -> crate::numeric::multifrontal_ldlt::Threads {
        crate::numeric::multifrontal_ldlt::Threads::Fixed(1)
    }

    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }

    /// Number of BTF diagonal blocks.
    /// Diagnostic: blocks admitted to the pipelined refactor replay, with
    /// their Amdahl-bounded worker counts.
    #[doc(hidden)]
    pub fn pipelined_blocks(&self) -> &[(usize, usize)] {
        &self.factors.pipelined
    }

    pub fn n_blocks(&self) -> usize {
        self.factors.block_ptr.len() - 1
    }

    /// Stored factor entries: L + U + diagonal + off-block.
    pub fn factor_nnz(&self) -> usize {
        self.factors.l_val.len()
            + self.factors.u_val.len()
            + self.factors.udiag.len()
            + self.factors.f_val.len()
    }

    /// Solve `A x = b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        let t = crate::clock::Instant::now();
        let x = self.solve_inner(b)?;
        self.record_solve(1, t, 0);
        Ok(x)
    }

    fn solve_inner(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        let f = &self.factors;
        if b.len() != f.n {
            return Err(RslabError::DimensionMismatch {
                expected: f.n,
                got: b.len(),
            });
        }
        let mut w = vec![T::zero(); f.n];
        for (k, &orig) in f.row_perm.iter().enumerate() {
            w[k] = b[orig] * T::from_real(f.rs_inv[orig]);
        }
        self.solve_permuted(&mut w);
        let mut xout = vec![T::zero(); f.n];
        for (k, &c) in f.col_perm.iter().enumerate() {
            xout[c] = w[k];
        }
        Ok(xout)
    }

    /// Solve the transposed system `A^T x = b` with the **same** factorization.
    ///
    /// This is the plain transpose, NOT the conjugate transpose: for a complex
    /// adjoint solve `A^H x = b`, conjugate `b` before and `x` after. (This
    /// matches the convention of the usual sparse-LU transpose solves, and is
    /// what implicit-function adjoints over holomorphic residuals need.)
    ///
    /// The stored form is `A = Rs * P_r^T * M * C` with `M` the block-upper
    /// (BTF) permuted, row-scaled matrix and `M_bb = L_b U_b` per diagonal
    /// block, so `A^T x = b` is `M^T (P_r Rs x) = C b`: gather `b` through the
    /// column permutation, run the transposed block substitution (blocks
    /// forward, per block `U^T` forward then `L^T` backward, off-block `F^T`
    /// contributions from the already-solved earlier blocks), then scatter
    /// through the row permutation and undo the row scaling. Sequential and
    /// bit-deterministic, like [`solve`](Self::solve).
    pub fn solve_transpose(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        let f = &self.factors;
        if b.len() != f.n {
            return Err(RslabError::DimensionMismatch {
                expected: f.n,
                got: b.len(),
            });
        }
        // w = C*b: position k of the permuted system reads b at its column.
        let mut w = vec![T::zero(); f.n];
        for (k, &c) in f.col_perm.iter().enumerate() {
            w[k] = b[c];
        }
        self.solve_permuted_transpose(&mut w);
        // x = Rs^-1 * P_r^T * w: scatter through the row permutation, then undo
        // the row scaling (Rs is diagonal, so it transposes onto the solution).
        let mut xout = vec![T::zero(); f.n];
        for (k, &orig) in f.row_perm.iter().enumerate() {
            xout[orig] = w[k] * T::from_real(f.rs_inv[orig]);
        }
        Ok(xout)
    }

    /// The transposed block substitution on the permuted vector: `M^T` is block
    /// **lower** triangular (the transpose of the BTF block-upper `M`), so the
    /// blocks run forward, and within a block `M_bb^T = U_b^T L_b^T` solves as
    /// `U^T` (lower, diagonal `udiag`) forward then `L^T` (unit upper) backward.
    /// Column `j` of the stored `U`/`L`/`F` is row `j` of the transpose, so
    /// every inner loop is a gather over the existing column storage.
    fn solve_permuted_transpose(&self, w: &mut [T]) {
        // Deliberately mul+sub, NOT `fmadd`: every inner loop here is a
        // gather onto a single accumulator - a latency-bound serial chain
        // where the FMA's higher latency loses to the pipelined mul + sub
        // (see the `solve_ldlt` backward-sweep note). `fmadd` stays in the
        // scatter-form sweeps of `solve_permuted`/`solve_many`.
        let f = &self.factors;
        for b in 0..f.block_ptr.len() - 1 {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            // F^T: this block's rows read the already-solved earlier blocks.
            for j in bs..be {
                let mut acc = w[j];
                for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                    acc = acc - f.f_val[k] * w[f.f_rowidx[k] as usize];
                }
                w[j] = acc;
            }
            // U^T (lower triangular, diagonal `udiag`) forward within the block.
            for j in bs..be {
                let mut acc = w[j];
                for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                    acc = acc - f.u_val[k] * w[f.u_rowidx[k] as usize];
                }
                w[j] = acc / f.udiag[j];
            }
            // L^T (unit upper) backward within the block.
            for j in (bs..be).rev() {
                let mut acc = w[j];
                for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                    acc = acc - f.l_val[k] * w[f.l_rowidx[k] as usize];
                }
                w[j] = acc;
            }
        }
    }

    /// Solve for `nrhs` right-hand sides stored row-major (`b[i * nrhs + col]`),
    /// matching [`crate::LuSolver::solve_many`]'s layout.
    ///
    /// Batched: the factor is traversed **once** and every stored entry is
    /// applied to all `nrhs` columns through contiguous per-row inner loops
    /// (SIMD-friendly and cache-reusing), the sparse-scalar factorization
    /// cannot use BLAS-3, but the wide solve can still vectorize across the
    /// right-hand sides. Each column's operation order is identical to
    /// [`solve`](Self::solve), so the result is bit-identical to `nrhs`
    /// single solves.
    pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
        let t = crate::clock::Instant::now();
        let x = self.solve_many_inner(b, nrhs)?;
        self.record_solve(nrhs, t, 0);
        Ok(x)
    }

    fn solve_many_inner(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
        let f = &self.factors;
        if nrhs == 0 || b.len() != f.n * nrhs {
            return Err(RslabError::DimensionMismatch {
                expected: f.n * nrhs.max(1),
                got: b.len(),
            });
        }
        // Permute + scale all columns into the row-major work block.
        let mut w = vec![T::zero(); f.n * nrhs];
        for (k, &orig) in f.row_perm.iter().enumerate() {
            let sv = T::from_real(f.rs_inv[orig]);
            let src = &b[orig * nrhs..orig * nrhs + nrhs];
            let dst = &mut w[k * nrhs..k * nrhs + nrhs];
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * sv;
            }
        }
        // Row j's values, staged so the axpy targets never alias the source.
        let mut xj = vec![T::zero(); nrhs];
        for blk in (0..f.block_ptr.len() - 1).rev() {
            let (bs, be) = (f.block_ptr[blk], f.block_ptr[blk + 1]);
            // L (unit lower) forward within the block. Negating the factor
            // value (loop-invariant here) instead of `xj` keeps each column's
            // FMA product bitwise equal to `solve_permuted`'s
            // (`(-a)*b == a*(-b)` exactly per real FMA).
            for j in bs..be {
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                    let (lr, nlv) = (f.l_rowidx[k] as usize, T::zero() - f.l_val[k]);
                    let row = &mut w[lr * nrhs..lr * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&xj) {
                        *r = fmadd(nlv, x, *r);
                    }
                }
            }
            // U backward within the block. Per-element division (not
            // reciprocal-multiply) keeps each column bit-identical to `solve`.
            for j in (bs..be).rev() {
                let d = f.udiag[j];
                {
                    let row = &mut w[j * nrhs..j * nrhs + nrhs];
                    for r in row.iter_mut() {
                        *r = *r / d;
                    }
                }
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                    let (ur, nuv) = (f.u_rowidx[k] as usize, T::zero() - f.u_val[k]);
                    let row = &mut w[ur * nrhs..ur * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&xj) {
                        *r = fmadd(nuv, x, *r);
                    }
                }
            }
            // Off-block columns feed the rows of earlier blocks.
            for j in bs..be {
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                    let (fr, nfv) = (f.f_rowidx[k] as usize, T::zero() - f.f_val[k]);
                    let row = &mut w[fr * nrhs..fr * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&xj) {
                        *r = fmadd(nfv, x, *r);
                    }
                }
            }
        }
        // Undo the column permutation.
        let mut xout = vec![T::zero(); f.n * nrhs];
        for (k, &c) in f.col_perm.iter().enumerate() {
            xout[c * nrhs..c * nrhs + nrhs].copy_from_slice(&w[k * nrhs..k * nrhs + nrhs]);
        }
        Ok(xout)
    }

    /// Solve with iterative refinement against the exact matrix (up to
    /// `max_iter` refinement steps, keeping the best iterate by residual
    /// max-norm), mirroring [`crate::solve_lu_refined`].
    pub fn solve_refined(
        &self,
        a: &GeneralCsc<T>,
        b: &[T],
        max_iter: usize,
    ) -> Result<Vec<T>, RslabError> {
        Ok(self
            .solve_refined_with(a, b, &crate::refine::RefinePolicy::steps(max_iter))?
            .0)
    }

    /// Iterative refinement under an explicit
    /// [`RefinePolicy`](crate::refine::RefinePolicy), reporting the achieved
    /// backward error.
    pub fn solve_refined_with(
        &self,
        a: &GeneralCsc<T>,
        b: &[T],
        policy: &crate::refine::RefinePolicy,
    ) -> Result<(Vec<T>, crate::refine::RefineOutcome), RslabError> {
        let mut x = self.solve(b)?;
        let outcome = self.refine_into(a, b, &mut x, policy)?;
        Ok((x, outcome))
    }

    /// Refine an existing iterate in place, allocating nothing for the
    /// solution.
    pub fn refine_into(
        &self,
        a: &GeneralCsc<T>,
        b: &[T],
        x: &mut [T],
        policy: &crate::refine::RefinePolicy,
    ) -> Result<crate::refine::RefineOutcome, RslabError> {
        let t = crate::clock::Instant::now();
        let outcome = self.refine_into_inner(a, b, x, policy)?;
        self.record_solve(0, t, outcome.steps);
        Ok(outcome)
    }

    fn refine_into_inner(
        &self,
        a: &GeneralCsc<T>,
        b: &[T],
        x: &mut [T],
        policy: &crate::refine::RefinePolicy,
    ) -> Result<crate::refine::RefineOutcome, RslabError> {
        let n = self.factors.n;
        if a.n != n || b.len() != n || x.len() != n {
            return Err(RslabError::DimensionMismatch {
                expected: n,
                got: a.n,
            });
        }
        crate::refine::refine_in_place(a, b, x, policy, |r| self.solve(r))
    }

    /// The block forward/backward substitution on the permuted/scaled vector.
    /// The axpys run through `fmadd` with the loop-invariant operand negated
    /// once per column, `solve_many` negates the per-entry factor value
    /// instead, which is bitwise the same product (`(-a)*b == a*(-b)` holds
    /// exactly per real FMA), so the two stay bit-identical per column.
    fn solve_permuted(&self, w: &mut [T]) {
        let f = &self.factors;
        for b in (0..f.block_ptr.len() - 1).rev() {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            // L (unit lower) forward within the block. Unchecked: all row
            // indices are final positions `< n` by construction.
            for j in bs..be {
                let xj = w[j];
                if xj != T::zero() {
                    let nxj = T::zero() - xj;
                    for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                        let lr = f.l_rowidx[k] as usize;
                        debug_assert!(lr < w.len());
                        unsafe {
                            *w.get_unchecked_mut(lr) = fmadd(f.l_val[k], nxj, *w.get_unchecked(lr));
                        }
                    }
                }
            }
            // U backward within the block.
            for j in (bs..be).rev() {
                let xj = w[j] / f.udiag[j];
                w[j] = xj;
                if xj != T::zero() {
                    let nxj = T::zero() - xj;
                    for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                        let ur = f.u_rowidx[k] as usize;
                        debug_assert!(ur < w.len());
                        unsafe {
                            *w.get_unchecked_mut(ur) = fmadd(f.u_val[k], nxj, *w.get_unchecked(ur));
                        }
                    }
                }
            }
            // Off-block columns feed the rows of earlier blocks.
            for j in bs..be {
                let xj = w[j];
                if xj != T::zero() {
                    let nxj = T::zero() - xj;
                    for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                        let fr = f.f_rowidx[k] as usize;
                        debug_assert!(fr < w.len());
                        unsafe {
                            *w.get_unchecked_mut(fr) = fmadd(f.f_val[k], nxj, *w.get_unchecked(fr));
                        }
                    }
                }
            }
        }
    }

    /// Numeric-only refactorization: replay the stored pattern and pivot
    /// sequence on new values with the **same** sparsity pattern. No symbolic
    /// work, no pivot search, the fast path for frequency sweeps and Newton
    /// steps. Fails with a pattern-mismatch error if `a`'s pattern deviates
    /// from the factored one, and with [`RslabError::SingularBasis`] if a
    /// frozen pivot becomes zero (re-`factor` with pivoting in that case).
    /// After an error the factorization is invalid; a subsequent successful
    /// `refactor` or a fresh `factor` makes it valid again.
    // The replay loops index several parallel arrays at offset positions;
    // iterator forms would obscure the offset arithmetic.
    #[allow(clippy::needless_range_loop)]
    pub fn refactor(&mut self, a: &GeneralCsc<T>) -> Result<(), RslabError> {
        a.validate()?;
        let t = crate::clock::Instant::now();
        // Cloned out of the factors so the borrow below stays exclusive; the
        // flag is read-only for us either way.
        let int_flag = self.factors.interrupt.clone();
        let int_flag = int_flag.as_deref();
        let f = &mut self.factors;
        if a.n != f.n {
            return Err(RslabError::DimensionMismatch {
                expected: f.n,
                got: a.n,
            });
        }
        if a.nnz() != f.nnz_a {
            return Err(pattern_mismatch());
        }
        let rs_inv = row_scale_inv(a, f.scaled);

        // Branch-free pattern verification against the recorded program: every
        // entry must map to the exact final position it had at factor time
        // (this subsumes the old per-column F-walk and leftover checks).
        {
            let mut acc: Ki = 0;
            for (k, &r) in a.row_idx.iter().enumerate() {
                acc |= (f.pinv[r] as Ki) ^ f.scatter_expect[k];
            }
            if acc != 0 {
                return Err(pattern_mismatch());
            }
        }

        // Per-block replay jobs over disjoint value ranges: L/U/F entries and
        // `udiag` of a block are contiguous (`colptr[bs]..colptr[be]`), so the
        // mutable arrays split cleanly and the blocks replay independently,
        // in parallel when the factor-time opt-in chose it (bit-identical:
        // per-block work is sequential and shares nothing).
        struct RJob<'s, T> {
            b: usize,
            l_v: &'s mut [T],
            u_v: &'s mut [T],
            ud: &'s mut [T],
            f_v: &'s mut [T],
        }
        let nblocks = f.block_ptr.len() - 1;
        let mut jobs: Vec<RJob<'_, T>> = Vec::with_capacity(nblocks);
        {
            let (mut lv, mut uv) = (f.l_val.as_mut_slice(), f.u_val.as_mut_slice());
            let (mut ud, mut fv) = (f.udiag.as_mut_slice(), f.f_val.as_mut_slice());
            for b in 0..nblocks {
                interrupt_check(int_flag)?;
                let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
                let (a1, r1) =
                    std::mem::take(&mut lv).split_at_mut(f.l_colptr[be] - f.l_colptr[bs]);
                lv = r1;
                let (a2, r2) =
                    std::mem::take(&mut uv).split_at_mut(f.u_colptr[be] - f.u_colptr[bs]);
                uv = r2;
                let (a3, r3) = std::mem::take(&mut ud).split_at_mut(be - bs);
                ud = r3;
                let (a4, r4) =
                    std::mem::take(&mut fv).split_at_mut(f.f_colptr[be] - f.f_colptr[bs]);
                fv = r4;
                jobs.push(RJob {
                    b,
                    l_v: a1,
                    u_v: a2,
                    ud: a3,
                    f_v: a4,
                });
            }
        }
        let (block_ptr, col_perm) = (&f.block_ptr, &f.col_perm);
        let (l_colptr, l_rowidx) = (&f.l_colptr, &f.l_rowidx);
        let (u_colptr, u_rowidx) = (&f.u_colptr, &f.u_rowidx);
        let f_colptr = &f.f_colptr;
        let scatter_target = &f.scatter_target;
        let rs_inv_ref = &rs_inv;
        // Raw column-disjoint views of one block's value arrays for the
        // level-parallel replay: every column writes only its own L/U/F/diag
        // slots and reads L columns completed in earlier levels (fenced by the
        // per-level join), so the shared-mutable access is race-free.
        struct BlockPtrs<T> {
            l_v: PanelPtr<T>,
            u_v: PanelPtr<T>,
            ud: PanelPtr<T>,
            f_v: PanelPtr<T>,
        }
        impl<T> Clone for BlockPtrs<T> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<T> Copy for BlockPtrs<T> {}

        // Replay one column through the recorded program: scatter, eliminate
        // in the stored topological order (bit-identical to `factor_impl`'s
        // pass 3: same `fmadd`, same per-column order), pivot, emit L.
        // SAFETY: caller guarantees exclusive ownership of column `j`'s output
        // slots and completed dependency columns (see `BlockPtrs`); `x` is this
        // caller's scratch, all-zero on entry and left all-zero.
        let replay_col = |j: usize,
                          bs: usize,
                          bases: (usize, usize, usize),
                          p: BlockPtrs<T>,
                          x: &mut [T],
                          sync: Option<(
            &[std::sync::atomic::AtomicBool],
            &std::sync::atomic::AtomicBool,
        )>|
         -> Result<(), RslabError> {
            let (l_base, u_base, f_base) = bases;
            let c = col_perm[j];
            unsafe {
                for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                    let r = a.row_idx[k];
                    let sv = a.values[k] * T::from_real(rs_inv_ref[r]);
                    let tv = scatter_target[k];
                    if tv & KI_FBIT != 0 {
                        *p.f_v.get().add((tv & !KI_FBIT) as usize - f_base) = sv;
                    } else {
                        x[tv as usize - bs] = sv;
                    }
                }
                for k in u_colptr[j]..u_colptr[j + 1] {
                    let pr = u_rowidx[k] as usize;
                    // Pipelined mode: consume L column `pr` only once its
                    // owner has published it (Acquire pairs with the owner's
                    // Release store after emitting the column).
                    if let Some((ready, abort)) = sync {
                        use std::sync::atomic::Ordering as AOrd;
                        let mut spins = 0u32;
                        while !ready[pr - bs].load(AOrd::Acquire) {
                            if abort.load(AOrd::Acquire) {
                                return Err(pattern_mismatch());
                            }
                            spins += 1;
                            if spins & 0x3FF == 0 {
                                std::thread::yield_now();
                            } else {
                                std::hint::spin_loop();
                            }
                        }
                    }
                    let xu = x[pr - bs];
                    x[pr - bs] = T::zero();
                    *p.u_v.get().add(k - u_base) = xu;
                    let nxu = T::zero() - xu;
                    for kl in l_colptr[pr]..l_colptr[pr + 1] {
                        let lr = l_rowidx[kl] as usize - bs;
                        debug_assert!(lr < x.len());
                        *x.get_unchecked_mut(lr) =
                            fmadd(nxu, *p.l_v.get().add(kl - l_base), *x.get_unchecked(lr));
                    }
                }
                let d = x[j - bs];
                x[j - bs] = T::zero();
                if d.magnitude() == 0.0 || !d.is_finite() {
                    return Err(RslabError::SingularBasis { column: c });
                }
                *p.ud.get().add(j - bs) = d;
                for k in l_colptr[j]..l_colptr[j + 1] {
                    let lr = l_rowidx[k] as usize - bs;
                    *p.l_v.get().add(k - l_base) = x[lr] / d;
                    x[lr] = T::zero();
                }
            }
            Ok(())
        };

        let pipelined = &f.pipelined;
        let replay_block = |job: RJob<'_, T>| -> Result<(), RslabError> {
            let b = job.b;
            let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
            let bases = (l_colptr[bs], u_colptr[bs], f_colptr[bs]);
            let ptrs = BlockPtrs {
                l_v: PanelPtr(job.l_v.as_mut_ptr()),
                u_v: PanelPtr(job.u_v.as_mut_ptr()),
                ud: PanelPtr(job.ud.as_mut_ptr()),
                f_v: PanelPtr(job.f_v.as_mut_ptr()),
            };
            let nthreads = rayon::current_num_threads().max(1);
            let pipe_nw = pipelined
                .iter()
                .find(|&&(pb, _)| pb == b)
                .map(|&(_, nw)| nw);
            if nthreads >= 2 && pipe_nw.is_some() {
                // NICSLU-style pipelined replay: worker `w` owns columns
                // `bs+w, bs+w+nw, ...` in order and spin-waits just-in-time on
                // each U-dependency's ready flag before consuming its L
                // column. Bit-identical (per-column arithmetic and writes are
                // untouched); dedicated OS threads, so a spinning peer can
                // never starve the owner of the column it waits for (a rayon
                // task pool could).
                use std::sync::atomic::{AtomicBool, Ordering as AOrd};
                let bn = be - bs;
                // Worker count bounded by the DAG's admissible speedup (the
                // plan's W/C ratio) and the thread budget.
                let nw = pipe_nw.unwrap_or(2).clamp(2, nthreads);
                let ready: Vec<AtomicBool> = (0..bn).map(|_| AtomicBool::new(false)).collect();
                let abort = AtomicBool::new(false);
                let errs: Vec<Result<(), RslabError>> = std::thread::scope(|sc| {
                    let handles: Vec<_> = (0..nw)
                        .map(|w| {
                            let (ready, abort) = (&ready, &abort);
                            sc.spawn(move || -> Result<(), RslabError> {
                                let mut x = vec![T::zero(); bn];
                                let mut jj = bs + w;
                                while jj < be {
                                    interrupt_check(int_flag)?;
                                    // SAFETY: worker-owned column; deps are
                                    // fenced by the ready Acquire loads inside
                                    // `replay_col`.
                                    let r = replay_col(
                                        jj,
                                        bs,
                                        bases,
                                        ptrs,
                                        &mut x,
                                        Some((ready, abort)),
                                    );
                                    ready[jj - bs].store(true, AOrd::Release);
                                    if let Err(e) = r {
                                        // Release everything this worker still
                                        // owns so the peers' spins terminate.
                                        abort.store(true, AOrd::Release);
                                        let mut k = jj + nw;
                                        while k < be {
                                            ready[k - bs].store(true, AOrd::Release);
                                            k += nw;
                                        }
                                        return Err(e);
                                    }
                                    if abort.load(AOrd::Acquire) {
                                        let mut k = jj + nw;
                                        while k < be {
                                            ready[k - bs].store(true, AOrd::Release);
                                            k += nw;
                                        }
                                        return Ok(());
                                    }
                                    jj += nw;
                                }
                                Ok(())
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap_or(Err(pattern_mismatch())))
                        .collect()
                });
                // Deterministic error selection: a real singular pivot wins
                // over the sympathetic aborts of the other workers (their
                // spins return `pattern_mismatch`), lowest column first.
                let mut first: Option<RslabError> = None;
                for r in errs {
                    if let Err(e) = r {
                        let better = match (&e, &first) {
                            (_, None) => true,
                            (
                                RslabError::SingularBasis { column: c1 },
                                Some(RslabError::SingularBasis { column: c0 }),
                            ) => c1 < c0,
                            (RslabError::SingularBasis { .. }, Some(_)) => true,
                            _ => false,
                        };
                        if better {
                            first = Some(e);
                        }
                    }
                }
                if let Some(e) = first {
                    return Err(e);
                }
                return Ok(());
            }
            let mut x = vec![T::zero(); be - bs];
            for j in bs..be {
                // SAFETY: this closure exclusively owns the whole block.
                replay_col(j, bs, bases, ptrs, &mut x, None)?;
            }
            Ok(())
        };
        if f.par_refactor && nblocks > 1 {
            use rayon::prelude::*;
            let results: Vec<Result<(), RslabError>> =
                jobs.into_par_iter().map(replay_block).collect();
            for r in results {
                r?;
            }
        } else {
            for job in jobs {
                replay_block(job)?;
            }
        }
        f.rs_inv = rs_inv;
        let entry = (std::mem::size_of::<T>() + std::mem::size_of::<Ki>()) as u64;
        let nnz = self.diagnostics.factor_nnz;
        self.diagnostics.push(
            "klu-refactor",
            t.elapsed().as_secs_f64() * 1e3,
            diagnostics_flops(&self.diagnostics),
            nnz * entry,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    /// The MC64 transversal keeps the diagonal-preference pivoting on the
    /// diagonal: on a row-scrambled, badly scaled grid the numeric fill
    /// equals the symbolic estimate, while the structural transversal
    /// pivots off the diagonal and fills.
    #[test]
    fn klu_matching_keeps_fill_at_the_symbolic_estimate() {
        let m = 40usize;
        let n = m * m;
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for j in 0..n {
            let (x, y) = (j % m, j / m);
            let rs = |i: usize| 10f64.powi(((i * 7919) % 13) as i32 - 6);
            let mut push = |i: usize, v: f64| cols[j].push(((i + 17) % n, v * rs(i)));
            push(j, 4.0);
            if x > 0 {
                push(j - 1, -1.4);
            }
            if x + 1 < m {
                push(j + 1, -0.6);
            }
            if y > 0 {
                push(j - m, -1.0);
            }
            if y + 1 < m {
                push(j + m, -1.0);
            }
        }
        let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
        for c in &mut cols {
            c.sort_by_key(|e| e.0);
            for &(r, v) in c.iter() {
                row_idx.push(r);
                values.push(v);
            }
            col_ptr.push(row_idx.len());
        }
        let a = GeneralCsc {
            n,
            col_ptr,
            row_idx,
            values,
        };
        let b: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
        let with = KluSettings::default();
        let sym = KluSymbolic::analyze_with(&a, &with).unwrap();
        let s = sym.factor(&a, &with).unwrap();
        assert_eq!(
            s.factor_nnz(),
            sym.symbolic_factor_nnz(),
            "no pivot-induced fill"
        );
        let x = s.solve(&b).unwrap();
        let mut r = b.clone();
        let mut d: Vec<f64> = b.iter().map(|v| v.abs()).collect();
        for j in 0..n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                r[a.row_idx[k]] -= a.values[k] * x[j];
                d[a.row_idx[k]] += a.values[k].abs() * x[j].abs();
            }
        }
        let omega = r
            .iter()
            .zip(&d)
            .map(|(ri, di)| ri.abs() / di)
            .fold(0.0, f64::max);
        assert!(omega < 1e-13, "backward error {omega}");
        let without = KluSettings::default().with_matching(false);
        let s0 = KluSolver::factor(&a, &without).unwrap();
        assert!(
            s0.factor_nnz() > s.factor_nnz(),
            "structural transversal fills more"
        );
    }
    use super::*;
    use crate::numeric::multifrontal_ldlt::SolverSettings;
    use crate::numeric::multifrontal_lu::{factor_general_lu, solve_lu};
    use num_complex::Complex;

    fn resid<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
        let mut ax = vec![T::zero(); a.n];
        a.matvec(x, &mut ax);
        let num = b
            .iter()
            .zip(&ax)
            .map(|(&bi, &axi)| (bi - axi).magnitude())
            .fold(0.0, f64::max);
        let den = b.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
        num / den.max(1e-300)
    }

    /// Deterministic xorshift for value generation (no rand dependency).
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    /// Circuit-shaped test matrix: sparse, unsymmetric, diagonally weighted,
    /// structurally nonsingular, with genuinely reducible structure (a
    /// one-directional bridge between two internally coupled halves).
    fn circuit_like(n: usize, seed: u64) -> GeneralCsc<f64> {
        let mut rng = Rng(seed | 1);
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        let half = n / 2;
        for j in 0..n {
            r.push(j);
            c.push(j);
            v.push(4.0 + rng.next_f64());
            // couplings within the same half only (keeps two SCC groups)
            let base = if j < half { 0 } else { half };
            let span = if j < half { half } else { n - half };
            for t in 1..=3usize {
                let i = base + (j - base + t * 7 + 1) % span;
                if i != j {
                    r.push(i);
                    c.push(j);
                    v.push(rng.next_f64());
                }
            }
        }
        // one-directional bridge: second half feeds the first (rows in the
        // first half, columns in the second) -> reducible, never a single SCC
        for k in 0..4usize {
            r.push(k * 3 % half);
            c.push(half + (k * 5) % (n - half));
            v.push(0.5 + rng.next_f64().abs());
        }
        GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
    }

    /// Off-diagonal pivots must not cascade: when a tiny diagonal forces an
    /// off-diagonal pivot, the displaced diagonal row is reassigned to the
    /// column whose diagonal row was stolen (SuiteSparse KLU's repair), so
    /// later columns keep their diagonal preference and the fill stays near
    /// the symbolic diagonal-pivoting prediction instead of degenerating
    /// toward partial-pivoting fill (the scircuit/rajat15 2x fill blow-up).
    #[test]
    fn klu_offdiagonal_pivot_reassigns_diagonal() {
        // scircuit's mechanism in miniature: power-net rows (uniform
        // conductances -> every entry is that row's max, so row-max scaling
        // turns them into permanent large pivot candidates in every column
        // they touch) plus a few tiny diagonals that force the first steal.
        let n = 900;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            for t in [1usize, 2] {
                r.push((j + t) % n);
                c.push(j);
                v.push(1.0);
            }
            let tiny = j % 21 == 5; // j = 2 mod 3: never on a power-net column
            r.push(j);
            c.push(j);
            v.push(if tiny { 1e-9 } else { 4.0 });
            if tiny {
                // the steal target: dominant candidate in the tiny column...
                r.push(j + 1);
                c.push(j);
                v.push(10.0);
                // ...whose own column keeps the displaced row available, but
                // small enough that plain partial pivoting would not pick it
                r.push(j);
                c.push(j + 1);
                v.push(0.3);
            }
        }
        // power nets: rows touching every 3rd column with uniform values
        for k in 0..6usize {
            let p = 100 + 130 * k;
            for j in (0..n).step_by(3) {
                if j != p && j + 1 != p && j + 2 != p {
                    r.push(p);
                    c.push(j);
                    v.push(2.0);
                }
            }
        }
        let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
        let sym = KluSymbolic::analyze(&a).unwrap();
        let f = sym.factor(&a, &KluSettings::default()).unwrap();
        assert!(
            (f.factor_nnz() as f64) < 1.5 * sym.symbolic_factor_nnz() as f64,
            "off-diagonal pivots cascaded: fill {} vs symbolic {}",
            f.factor_nnz(),
            sym.symbolic_factor_nnz()
        );
        let b: Vec<f64> = (0..n).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
        let x = f.solve(&b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-9, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn klu_solves_circuit_like_and_matches_multifrontal() {
        let a = circuit_like(200, 42);
        let b: Vec<f64> = (0..200).map(|i| (i % 11) as f64 - 5.0).collect();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert!(s.n_blocks() >= 2, "bridge structure must be reducible");
        let x = s.solve(&b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-12, "residual {}", resid(&a, &x, &b));
        // cross-check against the multifrontal LU
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let xr = solve_lu(&f, &b).unwrap();
        let diff = x
            .iter()
            .zip(&xr)
            .map(|(&p, &q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(diff < 1e-9, "klu vs multifrontal differ by {diff}");
    }

    #[test]
    fn klu_complex_small_diagonal_pivots() {
        // Small diagonal, large off-diagonals: threshold pivoting must
        // abandon the diagonal and still solve accurately (same layout as
        // the multifrontal LU pivoting test).
        let c = |re, im| Complex::new(re, im);
        let m = 6;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(0.3, 0.05));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(2.0, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(1.5, -0.2));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(1.8, 0.1));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(2.2, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let x = s.solve(&b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-12, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn klu_lower_triangular_needs_no_fill() {
        // Lower bidiagonal: BTF flips it upper triangular; every block is a
        // singleton, so the factor stores no L/U entries at all.
        let n = 50;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            r.push(i);
            c.push(i);
            v.push(2.0 + (i % 3) as f64);
            if i + 1 < n {
                r.push(i + 1);
                c.push(i);
                v.push(-1.0);
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert_eq!(s.n_blocks(), n);
        // factor_nnz = n diagonal entries + (n-1) off-block entries, zero fill
        assert_eq!(s.factor_nnz(), 2 * n - 1);
        let b: Vec<f64> = (0..n).map(|i| i as f64 - 7.0).collect();
        let x = s.solve(&b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-14);
    }

    #[test]
    fn klu_structurally_singular_detected() {
        // Column 2 shares its only row pattern with column 0 -> no complete
        // matching regardless of values.
        let a =
            GeneralCsc::<f64>::from_triplets(3, &[0, 1, 0], &[0, 1, 2], &[1.0, 1.0, 5.0]).unwrap();
        match KluSymbolic::analyze(&a) {
            Err(RslabError::StructurallySingular) => {}
            other => panic!("expected StructurallySingular, got {other:?}"),
        }
    }

    #[test]
    fn klu_numerically_singular_detected() {
        // Structurally fine 2x2 block, but rank 1 numerically: the second
        // pivot must come up exactly zero.
        let a = GeneralCsc::<f64>::from_triplets(
            2,
            &[0, 1, 0, 1],
            &[0, 0, 1, 1],
            &[1.0, 2.0, 2.0, 4.0],
        )
        .unwrap();
        match KluSolver::factor(&a, &KluSettings::default()) {
            Err(RslabError::SingularBasis { .. }) => {}
            other => panic!("expected SingularBasis, got {other:?}"),
        }
    }

    #[test]
    fn klu_factor_is_bit_deterministic() {
        let a = circuit_like(150, 7);
        let s1 = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let s2 = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert_eq!(s1.factors.l_val, s2.factors.l_val);
        assert_eq!(s1.factors.u_val, s2.factors.u_val);
        assert_eq!(s1.factors.udiag, s2.factors.udiag);
        assert_eq!(s1.factors.row_perm, s2.factors.row_perm);
        let b: Vec<f64> = (0..150).map(|i| (i % 13) as f64).collect();
        assert_eq!(s1.solve(&b).unwrap(), s2.solve(&b).unwrap());
    }

    #[test]
    fn klu_refactor_replays_and_matches_fresh_factor() {
        let a = circuit_like(150, 99);
        let mut s = KluSolver::factor(&a, &KluSettings::default()).unwrap();

        // Same values: the replay must reproduce the factor bit-identically.
        let (lv, uv, dv) = (
            s.factors.l_val.clone(),
            s.factors.u_val.clone(),
            s.factors.udiag.clone(),
        );
        s.refactor(&a).unwrap();
        assert_eq!(s.factors.l_val, lv);
        assert_eq!(s.factors.u_val, uv);
        assert_eq!(s.factors.udiag, dv);

        // New values, same pattern: the refactored solve must be accurate.
        let a2 = GeneralCsc::from_triplets(
            a.n,
            &{
                let mut rows = Vec::new();
                for j in 0..a.n {
                    for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                        rows.push(a.row_idx[k]);
                    }
                }
                rows
            },
            &{
                let mut cols = Vec::new();
                for j in 0..a.n {
                    for _ in a.col_ptr[j]..a.col_ptr[j + 1] {
                        cols.push(j);
                    }
                }
                cols
            },
            &a.values
                .iter()
                .enumerate()
                .map(|(k, &v)| v * (1.0 + 0.01 * ((k % 17) as f64)))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        s.refactor(&a2).unwrap();
        let b: Vec<f64> = (0..a.n).map(|i| (i % 9) as f64 - 4.0).collect();
        let x = s.solve(&b).unwrap();
        assert!(
            resid(&a2, &x, &b) < 1e-11,
            "refactor residual {}",
            resid(&a2, &x, &b)
        );
    }

    /// 2D convection-diffusion 5-point grid: one large irreducible block with
    /// wide elimination-DAG wavefronts - the level-schedule target shape.
    fn grid_cd(m: usize) -> (GeneralCsc<f64>, Vec<f64>) {
        let n = m * m;
        let idx = |i: usize, j: usize| i * m + j;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..m {
            for j in 0..m {
                let p = idx(i, j);
                r.push(p);
                c.push(p);
                v.push(4.0 + 0.01 * (p % 7) as f64);
                let mut off = |q: usize, w: f64| {
                    r.push(q);
                    c.push(p);
                    v.push(w);
                };
                if i > 0 {
                    off(idx(i - 1, j), -1.2);
                }
                if i + 1 < m {
                    off(idx(i + 1, j), -0.8);
                }
                if j > 0 {
                    off(idx(i, j - 1), -1.1);
                }
                if j + 1 < m {
                    off(idx(i, j + 1), -0.9);
                }
            }
        }
        let a = GeneralCsc::from_triplets(n, &r, &c, &v).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 5) as f64 - 2.0).collect();
        (a, b)
    }

    #[test]
    fn klu_pipelined_refactor_is_bit_identical_to_sequential() {
        // One irreducible 1024-column block: the level schedule must engage
        // (parallel != Off) and its refactor replay must be bit-identical to
        // the strictly sequential one - same values, not just same residual.
        let (a, b) = grid_cd(32);
        let seq = KluSettings::default().with_parallel(KluParallel::Off);
        let par = KluSettings::default().with_parallel(KluParallel::On);

        let mut s_seq = KluSolver::factor(&a, &seq).unwrap();
        let mut s_par = KluSolver::factor(&a, &par).unwrap();
        assert!(
            s_seq.factors.pipelined.is_empty(),
            "Off must not admit pipeline blocks"
        );
        // The 1024-column test block is far below the work gate; force the
        // admission so the pipelined executor itself is exercised (the gates
        // only decide when it pays, not whether it is correct).
        s_par.factors.pipelined = vec![(0, 4)];

        // Fresh values on the frozen pattern, refactor both ways.
        let mut a2 = a.clone();
        for (k, v) in a2.values.iter_mut().enumerate() {
            *v += 1e-3 * ((k % 11) as f64 - 5.0);
        }
        s_seq.refactor(&a2).unwrap();
        s_par.refactor(&a2).unwrap();
        assert_eq!(s_seq.factors.l_val, s_par.factors.l_val, "L values differ");
        assert_eq!(s_seq.factors.u_val, s_par.factors.u_val, "U values differ");
        assert_eq!(s_seq.factors.udiag, s_par.factors.udiag, "diag differs");

        let x = s_par.solve(&b).unwrap();
        assert!(
            resid(&a2, &x, &b) < 1e-10,
            "pipelined refactor solve residual"
        );
    }

    #[test]
    fn klu_refactor_rejects_changed_pattern() {
        let a = circuit_like(60, 5);
        let mut s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        // Move one off-diagonal entry to a fresh position (same nnz).
        let (mut rows, mut cols, vals): (Vec<usize>, Vec<usize>, Vec<f64>) = {
            let mut rr = Vec::new();
            let mut cc = Vec::new();
            let mut vv = Vec::new();
            for j in 0..a.n {
                for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                    rr.push(a.row_idx[k]);
                    cc.push(j);
                    vv.push(a.values[k]);
                }
            }
            (rr, cc, vv)
        };
        let moved = rows.iter().zip(&cols).position(|(&r, &c)| r != c).unwrap();
        rows[moved] = (rows[moved] + 1) % a.n;
        cols[moved] = (cols[moved] + 1) % a.n;
        let a2 = GeneralCsc::from_triplets(a.n, &rows, &cols, &vals).unwrap();
        if a2.nnz() != a.nnz() {
            return; // duplicate collapse: not the case under test
        }
        assert!(s.refactor(&a2).is_err(), "changed pattern must be rejected");
    }

    /// Max-norm relative residual of the *transposed* system `A^T x = b`.
    fn resid_t<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
        resid(&a.transpose(), x, b)
    }

    #[test]
    fn klu_solve_transpose_matches_factored_transpose() {
        // Reducible circuit-shaped matrix: solve_transpose on A's factors must
        // agree with a fresh factorization of A^T, and satisfy A^T x = b.
        let a = circuit_like(200, 42);
        let b: Vec<f64> = (0..200).map(|i| ((i * 3) % 13) as f64 - 6.0).collect();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert!(s.n_blocks() >= 2, "bridge structure must be reducible");
        let x = s.solve_transpose(&b).unwrap();
        assert!(
            resid_t(&a, &x, &b) < 1e-12,
            "residual {}",
            resid_t(&a, &x, &b)
        );
        let st = KluSolver::factor(&a.transpose(), &KluSettings::default()).unwrap();
        let xr = st.solve(&b).unwrap();
        let diff = x
            .iter()
            .zip(&xr)
            .map(|(&p, &q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(
            diff < 1e-9,
            "transpose solve vs factored transpose differ by {diff}"
        );
    }

    #[test]
    fn klu_solve_transpose_complex_plain_not_conjugate() {
        // Complex: solve_transpose must solve the PLAIN transpose A^T x = b
        // (adjoint convention: the caller conjugates for A^H). Off-diagonal
        // pivoting pressure included (small diagonal), as in the solve test.
        let c = |re, im| Complex::new(re, im);
        let m = 6;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(0.3, 0.05));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(2.0, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(1.5, -0.2));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(1.8, 0.1));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(2.2, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n)
            .map(|i| c((i % 5) as f64 - 2.0, (i % 3) as f64))
            .collect();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let x = s.solve_transpose(&b).unwrap();
        assert!(
            resid_t(&a, &x, &b) < 1e-12,
            "residual {}",
            resid_t(&a, &x, &b)
        );
        // A^H x = b via the documented conjugation recipe.
        let bc: Vec<Complex<f64>> = b.iter().map(|v| v.conj()).collect();
        let xh: Vec<Complex<f64>> = s
            .solve_transpose(&bc)
            .unwrap()
            .iter()
            .map(|v| v.conj())
            .collect();
        let ah = {
            let t = a.transpose();
            GeneralCsc::<Complex<f64>> {
                n: t.n,
                col_ptr: t.col_ptr.clone(),
                row_idx: t.row_idx.clone(),
                values: t.values.iter().map(|v| v.conj()).collect(),
            }
        };
        assert!(resid(&ah, &xh, &b) < 1e-12);
    }

    #[test]
    fn klu_solve_transpose_singleton_blocks_and_options() {
        // Lower bidiagonal (all-singleton BTF blocks, pure F off-block path),
        // plus the no-BTF and no-scaling configurations on the circuit matrix.
        let n = 50;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            r.push(i);
            c.push(i);
            v.push(2.0 + (i % 3) as f64);
            if i + 1 < n {
                r.push(i + 1);
                c.push(i);
                v.push(-1.0);
            }
        }
        let tri = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let s = KluSolver::factor(&tri, &KluSettings::default()).unwrap();
        assert_eq!(s.n_blocks(), n);
        let b: Vec<f64> = (0..n).map(|i| i as f64 - 7.0).collect();
        let x = s.solve_transpose(&b).unwrap();
        assert!(resid_t(&tri, &x, &b) < 1e-14);

        let a = circuit_like(100, 21);
        let b: Vec<f64> = (0..a.n).map(|i| (i % 5) as f64 - 2.0).collect();
        for settings in [
            KluSettings::default().with_btf(false),
            KluSettings::default().with_row_scaling(false),
            KluSettings::default()
                .with_btf(false)
                .with_row_scaling(false),
        ] {
            let s = KluSolver::factor(&a, &settings).unwrap();
            let x = s.solve_transpose(&b).unwrap();
            assert!(resid_t(&a, &x, &b) < 1e-12, "settings {settings:?}");
        }
    }

    #[test]
    fn klu_solve_transpose_after_refactor() {
        // The transpose solve must read the refactored values, not stale ones.
        let a = circuit_like(150, 99);
        let mut s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let a2 = {
            let (mut rows, mut cols) = (Vec::new(), Vec::new());
            for j in 0..a.n {
                for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                    rows.push(a.row_idx[k]);
                    cols.push(j);
                }
            }
            let vals: Vec<f64> = a
                .values
                .iter()
                .enumerate()
                .map(|(k, &v)| v * (1.0 + 0.01 * ((k % 17) as f64)))
                .collect();
            GeneralCsc::from_triplets(a.n, &rows, &cols, &vals).unwrap()
        };
        s.refactor(&a2).unwrap();
        let b: Vec<f64> = (0..a.n).map(|i| (i % 9) as f64 - 4.0).collect();
        let x = s.solve_transpose(&b).unwrap();
        assert!(
            resid_t(&a2, &x, &b) < 1e-11,
            "residual {}",
            resid_t(&a2, &x, &b)
        );
    }

    #[test]
    fn klu_solve_transpose_empty_and_dimension_check() {
        let a = GeneralCsc::<f64>::from_triplets(0, &[], &[], &[]).unwrap();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert_eq!(s.solve_transpose(&[]).unwrap(), Vec::<f64>::new());
        let a = circuit_like(20, 1);
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert!(s.solve_transpose(&[0.0; 19]).is_err());
    }

    #[test]
    fn klu_solve_many_matches_single() {
        let a = circuit_like(80, 3);
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let nrhs = 4;
        let b: Vec<f64> = (0..a.n * nrhs).map(|k| (k % 7) as f64 - 3.0).collect();
        let x = s.solve_many(&b, nrhs).unwrap();
        for col in 0..nrhs {
            let bc: Vec<f64> = (0..a.n).map(|i| b[i * nrhs + col]).collect();
            let xc = s.solve(&bc).unwrap();
            for i in 0..a.n {
                assert_eq!(x[i * nrhs + col], xc[i], "rhs {col} row {i}");
            }
        }
    }

    #[test]
    fn klu_solve_refined_tightens_residual() {
        let a = circuit_like(120, 11);
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let b: Vec<f64> = (0..a.n).map(|i| ((i * i) % 23) as f64 - 11.0).collect();
        let x = s.solve_refined(&a, &b, 2).unwrap();
        assert!(resid(&a, &x, &b) < 1e-13);
    }

    #[test]
    fn klu_without_btf_still_solves() {
        let a = circuit_like(100, 21);
        let s = KluSolver::factor(
            &a,
            &KluSettings {
                btf: false,
                ..KluSettings::default()
            },
        )
        .unwrap();
        assert_eq!(s.n_blocks(), 1);
        let b: Vec<f64> = (0..a.n).map(|i| (i % 5) as f64).collect();
        let x = s.solve(&b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-12);
    }

    #[test]
    fn klu_empty_matrix() {
        let a = GeneralCsc::<f64>::from_triplets(0, &[], &[], &[]).unwrap();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert_eq!(s.solve(&[]).unwrap(), Vec::<f64>::new());
    }

    #[test]
    fn klu_estimate_matches_actual_fill_on_dominant_matrix() {
        // Diagonally dominant -> threshold pivoting keeps every diagonal, so
        // the diagonal-pivot symbolic fill must be EXACT, and the estimate's
        // factor_nnz must equal the factored fill.
        let a = circuit_like(150, 33);
        let sym = KluSymbolic::analyze(&a).unwrap();
        let est = sym.estimate_memory::<f64>();
        let s = sym.factor(&a, &KluSettings::default()).unwrap();
        assert_eq!(est.factor_nnz as usize, s.factor_nnz());
        assert_eq!(sym.symbolic_factor_nnz(), s.factor_nnz());
        assert!(est.factor_flops > 0);
        assert_eq!(est.critical_path_flops, est.factor_flops);
        assert!(est.transient_peak_bytes >= est.factor_bytes);
    }

    #[test]
    fn klu_diagnostics_phased_vs_oneshot() {
        let a = circuit_like(100, 4);
        let sym = KluSymbolic::analyze(&a).unwrap();
        // factor() never estimates implicitly: the estimate is attached only
        // when it was computed explicitly beforehand.
        let s0 = sym.factor(&a, &KluSettings::default()).unwrap();
        assert!(s0.diagnostics().estimate.is_none());
        let _ = sym.estimate_memory::<f64>();
        let s = sym.factor(&a, &KluSettings::default()).unwrap();
        let d = s.diagnostics();
        assert_eq!(d.threads, 1);
        assert_eq!(d.factor_nnz as usize, s.factor_nnz());
        assert!(d.estimate.is_some());
        assert_eq!(d.stages.len(), 1);
        assert_eq!(d.stages[0].name, "klu-factor");

        let mut s2 = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        // the one-shot factor fills its diagnostics like the phased path
        assert_eq!(s2.diagnostics().stages.last().unwrap().name, "klu-factor");
        s2.refactor(&a).unwrap();
        assert_eq!(s2.diagnostics().stages.last().unwrap().name, "klu-refactor");
    }

    /// Cascaded stages with one-way inter-stage feeds: `stages` irreducible
    /// diagonal blocks in the BTF (the reducible shape the Auto gate keys on).
    fn cascaded(n: usize, stages: usize, seed: u64) -> GeneralCsc<f64> {
        let mut rng = Rng(seed | 1);
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        let stage = n / stages;
        for j in 0..n {
            let s = (j / stage).min(stages - 1);
            let lo = s * stage;
            let hi = if s == stages - 1 { n } else { lo + stage };
            r.push(j);
            c.push(j);
            v.push(6.0 + rng.next_f64());
            // ring coupling inside the stage keeps the block irreducible
            let fwd = lo + (j - lo + 1) % (hi - lo);
            if fwd != j {
                r.push(fwd);
                c.push(j);
                v.push(-1.0 + 0.1 * rng.next_f64());
                r.push(j);
                c.push(fwd);
                v.push(-1.0 + 0.1 * rng.next_f64());
            }
            // one-way feed from the previous stage
            if s > 0 {
                r.push(j - stage);
                c.push(j);
                v.push(0.25 * rng.next_f64());
            }
        }
        GeneralCsc::from_triplets(n, &r, &c, &v).unwrap()
    }

    #[test]
    fn klu_parallel_auto_gate_resolves_from_structure() {
        // Small: below the nnz floor, Auto stays sequential.
        let a = cascaded(400, 6, 3);
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert!(!s.factors.parallel, "small case must stay sequential");
        // Multi-block and over the floor: Auto goes parallel.
        let a = cascaded(4000, 6, 9);
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert!(s.factors.block_ptr.len() > 4);
        assert!(a.nnz() >= 8_000);
        assert!(
            s.factors.parallel,
            "large multi-block case must parallelize"
        );
        // Off always wins.
        let s =
            KluSolver::factor(&a, &KluSettings::default().with_parallel(KluParallel::Off)).unwrap();
        assert!(!s.factors.parallel);
    }

    #[test]
    fn klu_parallel_factor_bit_identical() {
        let a = circuit_like(600, 7);
        let sym = KluSymbolic::analyze(&a).unwrap();
        let s1 = sym
            .factor(&a, &KluSettings::default().with_parallel_factor(false))
            .unwrap();
        let s2 = sym
            .factor(&a, &KluSettings::default().with_parallel_factor(true))
            .unwrap();
        assert!(s2.factors.block_ptr.len() > 2, "needs a multi-block case");
        assert_eq!(s1.factors.l_rowidx, s2.factors.l_rowidx);
        assert_eq!(s1.factors.u_rowidx, s2.factors.u_rowidx);
        assert_eq!(s1.factors.row_perm, s2.factors.row_perm);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&s1.factors.l_val), bits(&s2.factors.l_val));
        assert_eq!(bits(&s1.factors.u_val), bits(&s2.factors.u_val));
        assert_eq!(bits(&s1.factors.udiag), bits(&s2.factors.udiag));
        assert_eq!(s1.factors.scatter_expect, s2.factors.scatter_expect);
        assert_eq!(s1.factors.scatter_target, s2.factors.scatter_target);
        let b: Vec<f64> = (0..a.n).map(|i| (i % 5) as f64 - 2.0).collect();
        let x1 = s1.solve(&b).unwrap();
        let x2 = s2.solve(&b).unwrap();
        assert_eq!(bits(&x1), bits(&x2));

        // Refactor honors the same opt-in and stays bit-identical too.
        let a2 = GeneralCsc {
            n: a.n,
            col_ptr: a.col_ptr.clone(),
            row_idx: a.row_idx.clone(),
            values: a.values.iter().map(|&v| v * 1.25).collect(),
        };
        let mut s1 = s1;
        let mut s2 = s2;
        s1.refactor(&a2).unwrap();
        s2.refactor(&a2).unwrap();
        assert_eq!(bits(&s1.factors.l_val), bits(&s2.factors.l_val));
        assert_eq!(bits(&s1.factors.u_val), bits(&s2.factors.u_val));
        assert_eq!(bits(&s1.factors.udiag), bits(&s2.factors.udiag));
        let y1 = s1.solve(&b).unwrap();
        let y2 = s2.solve(&b).unwrap();
        assert_eq!(bits(&y1), bits(&y2));
    }

    #[test]
    fn klu_composes_as_gmres_preconditioner() {
        use crate::numeric::iterative::gmres;
        let a = circuit_like(120, 55);
        let m = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let b: Vec<f64> = (0..a.n).map(|i| (i % 7) as f64 - 3.0).collect();
        // Exact preconditioner -> GMRES converges in one iteration.
        let res = gmres(&a, &b, &m, 1e-12, 5, 5, None).unwrap();
        assert!(res.converged);
        assert!(res.iters <= 2, "iterations {}", res.iters);
        assert!(resid(&a, &res.x, &b) < 1e-10);
    }

    #[test]
    fn klu_settings_compose() {
        let s = KluSettings::default()
            .with_pivot_tol(1.0)
            .with_row_scaling(false)
            .with_btf(false);
        assert_eq!(s.pivot_tol, 1.0);
        assert!(!s.row_scaling);
        assert!(!s.btf);
        let a = circuit_like(80, 9);
        let solver = KluSolver::factor(&a, &s).unwrap();
        let b: Vec<f64> = (0..a.n).map(|i| (i % 3) as f64).collect();
        let x = solver.solve(&b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-12);
    }
}
