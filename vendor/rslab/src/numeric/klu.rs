//! KLU-style sparse LU: BTF + per-block left-looking Gilbert-Peierls.
//!
//! The third direct path next to the multifrontal LDLᵀ and LU, built for
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
//!    a left-looking Gilbert-Peierls LU — per-column depth-first reach on the
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
//! are **bit-identical across runs and thread counts** — this path doubles as
//! the determinism arbiter for the parallel multifrontal paths.

use crate::error::RslabError;
use crate::ordering::btf;
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;

const UNSET: usize = usize::MAX;

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
}

impl Default for KluSettings {
    fn default() -> Self {
        Self {
            pivot_tol: 1e-3,
            row_scaling: true,
            btf: true,
        }
    }
}

impl KluSettings {
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
    pub fn with_btf(mut self, on: bool) -> Self {
        self.btf = on;
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
    /// Pre-pivot row permutation (new-to-old): BTF matching ∘ SCC order ∘
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
    /// rows) — such a matrix is singular for *every* value assignment.
    pub fn analyze_with<T: Scalar>(
        a: &GeneralCsc<T>,
        settings: &KluSettings,
    ) -> Result<Self, RslabError> {
        a.validate()?;
        let n = a.n;

        let (mut pre_row_perm, mut col_perm, block_ptr) = if settings.btf {
            let form = btf::block_triangular_form(n, &a.col_ptr, &a.row_idx)
                .ok_or(RslabError::StructurallySingular)?;
            (form.row_perm, form.col_perm, form.block_ptr)
        } else {
            let ident: Vec<usize> = (0..n).collect();
            let bp = if n == 0 { vec![0] } else { vec![0, n] };
            (ident.clone(), ident, bp)
        };

        // Per-block AMD on the symmetrized block pattern (B + Bᵀ, with
        // diagonal, matching what the multifrontal paths feed rslab-amd).
        // Blocks of size <= 2 have nothing to reorder.
        let mut pinv0 = vec![0usize; n];
        for (k, &r) in pre_row_perm.iter().enumerate() {
            pinv0[r] = k;
        }
        for b in 0..block_ptr.len() - 1 {
            let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
            let bn = be - bs;
            if bn <= 2 {
                continue;
            }
            let mut adj: Vec<Vec<i32>> = vec![Vec::new(); bn];
            for lj in 0..bn {
                let c = col_perm[bs + lj];
                adj[lj].push(lj as i32); // diagonal (zero-free after matching)
                for &r in &a.row_idx[a.col_ptr[c]..a.col_ptr[c + 1]] {
                    let pre = pinv0[r];
                    if pre >= bs && pre < be {
                        let li = pre - bs;
                        if li != lj {
                            adj[lj].push(li as i32);
                            adj[li].push(lj as i32);
                        }
                    }
                }
            }
            let mut colptr_i32 = Vec::with_capacity(bn + 1);
            let mut rowidx_i32 = Vec::new();
            colptr_i32.push(0i32);
            for col in adj.iter_mut() {
                col.sort_unstable();
                col.dedup();
                rowidx_i32.extend_from_slice(col);
                colptr_i32.push(rowidx_i32.len() as i32);
            }
            let pat = rslab_ordering_core::CscPattern::new(bn, &colptr_i32, &rowidx_i32)
                .ok_or_else(|| {
                    RslabError::InvalidInput("klu: malformed block pattern".to_string())
                })?;
            let lperm = rslab_amd::amd_order(&pat).map_err(|e| {
                RslabError::InvalidInput(format!("klu: AMD ordering failed: {e:?}"))
            })?;
            // Apply the local (new-to-old) perm symmetrically to the block's
            // segment of both permutations.
            let old_rows: Vec<usize> = pre_row_perm[bs..be].to_vec();
            let old_cols: Vec<usize> = col_perm[bs..be].to_vec();
            for (i, &lp) in lperm.iter().enumerate() {
                pre_row_perm[bs + i] = old_rows[lp as usize];
                col_perm[bs + i] = old_cols[lp as usize];
            }
        }

        // Freeze the analyzed pattern in the (final) pre-pivot space for the
        // a-priori estimators.
        let mut pinv_pre = vec![0usize; n];
        for (k, &r) in pre_row_perm.iter().enumerate() {
            pinv_pre[r] = k;
        }
        let mut pat_col_ptr = Vec::with_capacity(n + 1);
        let mut pat_row_idx = Vec::with_capacity(a.nnz());
        pat_col_ptr.push(0);
        for &c in &col_perm {
            for &r in &a.row_idx[a.col_ptr[c]..a.col_ptr[c + 1]] {
                pat_row_idx.push(pinv_pre[r]);
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
    /// Computed once and cached — the pass costs about as much as a numeric
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
    /// under the diagonal-pivoting assumption — the memory-backstop metric,
    /// mirroring [`LuSymbolic::symbolic_factor_nnz`](crate::LuSymbolic::symbolic_factor_nnz).
    pub fn symbolic_factor_nnz(&self) -> usize {
        let fill = self.symbolic_fill();
        (fill.l_nnz + fill.u_nnz + self.n as u64 + fill.f_nnz) as usize
    }

    /// **A-priori** memory/work estimate for factoring a matrix of scalar
    /// type `T` with this analysis — deterministic, computed from the stored
    /// pattern alone, mirroring [`LuSymbolic::estimate_memory`](crate::LuSymbolic::estimate_memory).
    ///
    /// KLU specifics: the fill is exact under diagonal pivoting (threshold
    /// pivoting can shift it slightly); `factor_flops` is the Gilbert-Peierls
    /// flop count (not the supernodal `nrow²·ncol` proxy); the path is
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
        let estimate = self.estimate_memory::<T>();
        let t = std::time::Instant::now();
        let factors = factor_impl(self, a, settings)?;
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        let nnz =
            (factors.l_val.len() + factors.u_val.len() + factors.udiag.len() + factors.f_val.len())
                as u64;
        let entry = (std::mem::size_of::<T>() + std::mem::size_of::<usize>()) as u64;
        let mut diagnostics = crate::diagnostics::Diagnostics {
            threads: 1,
            factor_nnz: nnz,
            estimate: Some(estimate),
            ..Default::default()
        };
        diagnostics.push(
            "klu-factor",
            factor_ms,
            diagnostics_flops(&diagnostics),
            nnz * entry,
        );
        Ok(KluSolver {
            factors,
            diagnostics,
        })
    }
}

/// The GP flop count carried on the attached estimate (0 when absent).
fn diagnostics_flops(d: &crate::diagnostics::Diagnostics) -> u64 {
    d.estimate.as_ref().map_or(0, |e| e.factor_flops)
}

/// The numeric KLU factorization: `P A Q = L U` per diagonal block plus the
/// off-block entries, with row scaling folded in.
#[derive(Debug, Clone)]
struct KluFactors<T> {
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
    /// L: strictly-below-diagonal entries per column, unit diagonal implicit.
    /// Row indices are final positions within the column's block.
    l_colptr: Vec<usize>,
    l_rowidx: Vec<usize>,
    l_val: Vec<T>,
    /// U: strictly-above-diagonal within-block entries per column, stored in
    /// elimination (topological) order — the refactor replay order.
    u_colptr: Vec<usize>,
    u_rowidx: Vec<usize>,
    u_val: Vec<T>,
    udiag: Vec<T>,
    /// Off-block entries (rows in earlier blocks, final positions), per
    /// column in the input's storage order. Not factored; applied in the
    /// block back-substitution.
    f_colptr: Vec<usize>,
    f_rowidx: Vec<usize>,
    f_val: Vec<T>,
}

/// KLU solver handle: factor (or analyze+factor), then solve / refactor.
#[derive(Debug, Clone)]
pub struct KluSolver<T> {
    factors: KluFactors<T>,
    diagnostics: crate::diagnostics::Diagnostics,
}

fn pattern_mismatch() -> RslabError {
    RslabError::InvalidInput(
        "klu: matrix pattern does not match the symbolic analysis / stored factorization"
            .to_string(),
    )
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

    let rs_inv = row_scale_inv(a, settings.row_scaling);
    let mut pinv_pre = vec![0usize; n];
    for (k, &r) in sym.pre_row_perm.iter().enumerate() {
        pinv_pre[r] = k;
    }

    // bpinv[pre_position] = final position; assigned when that row is pivoted.
    let mut bpinv = vec![UNSET; n];
    // Numeric work vector and DFS state, all in pre-pivot position space.
    let mut x = vec![T::zero(); n];
    let mut stamp = vec![0usize; n];
    let mut node_stack = vec![0usize; n];
    let mut cur_stack = vec![0usize; n];
    let mut topo: Vec<usize> = Vec::with_capacity(n);
    let mut nonpiv: Vec<usize> = Vec::with_capacity(n);

    let mut l_colptr = vec![0usize];
    let mut l_rowidx: Vec<usize> = Vec::new();
    let mut l_val: Vec<T> = Vec::new();
    let mut u_colptr = vec![0usize];
    let mut u_rowidx: Vec<usize> = Vec::new();
    let mut u_val: Vec<T> = Vec::new();
    let mut udiag: Vec<T> = Vec::with_capacity(n);
    let mut f_colptr = vec![0usize];
    let mut f_rowidx: Vec<usize> = Vec::new();
    let mut f_val: Vec<T> = Vec::new();

    for b in 0..sym.block_ptr.len() - 1 {
        let (bs, be) = (sym.block_ptr[b], sym.block_ptr[b + 1]);

        if be - bs == 1 {
            // Singleton block: the pivot is the (structurally nonzero)
            // diagonal entry itself; everything else in the column is
            // off-block.
            let c = sym.col_perm[bs];
            let mut diag: Option<T> = None;
            for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                let pre = pinv_pre[a.row_idx[k]];
                let sv = a.values[k] * T::from_real(rs_inv[a.row_idx[k]]);
                if pre == bs {
                    diag = Some(sv);
                } else if pre < bs {
                    f_rowidx.push(bpinv[pre]);
                    f_val.push(sv);
                } else {
                    return Err(pattern_mismatch());
                }
            }
            let d = diag.ok_or(RslabError::SingularBasis { column: c })?;
            if d.magnitude() == 0.0 || !d.is_finite() {
                return Err(RslabError::SingularBasis { column: c });
            }
            bpinv[bs] = bs;
            udiag.push(d);
            l_colptr.push(l_rowidx.len());
            u_colptr.push(u_rowidx.len());
            f_colptr.push(f_rowidx.len());
            continue;
        }

        // General irreducible block: left-looking Gilbert-Peierls.
        for j in bs..be {
            let c = sym.col_perm[j];
            let sj = j + 1; // unique DFS stamp for this column
            topo.clear();
            nonpiv.clear();

            // Pass 1 — symbolic: DFS the reach of the column's within-block
            // pattern over the L columns factored so far. Pivotal nodes come
            // out in `topo` post-order; non-pivotal nodes (pivot candidates)
            // in `nonpiv`.
            for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                let pre = pinv_pre[a.row_idx[k]];
                if pre < bs {
                    continue; // off-block, handled in pass 2
                }
                if pre >= be {
                    return Err(pattern_mismatch());
                }
                if stamp[pre] == sj {
                    continue;
                }
                stamp[pre] = sj;
                if bpinv[pre] == UNSET {
                    nonpiv.push(pre);
                    continue;
                }
                let mut d = 0usize;
                node_stack[0] = pre;
                cur_stack[0] = l_colptr[bpinv[pre]];
                loop {
                    let u = node_stack[d];
                    let endp = l_colptr[bpinv[u] + 1];
                    let mut descended = false;
                    while cur_stack[d] < endp {
                        let ch = l_rowidx[cur_stack[d]];
                        cur_stack[d] += 1;
                        if stamp[ch] == sj {
                            continue;
                        }
                        stamp[ch] = sj;
                        if bpinv[ch] == UNSET {
                            nonpiv.push(ch);
                            continue;
                        }
                        d += 1;
                        node_stack[d] = ch;
                        cur_stack[d] = l_colptr[bpinv[ch]];
                        descended = true;
                        break;
                    }
                    if descended {
                        continue;
                    }
                    topo.push(u);
                    if d == 0 {
                        break;
                    }
                    d -= 1;
                }
            }

            // Pass 2 — scatter the scaled column values (off-block entries go
            // straight to F; earlier blocks are already fully pivoted, so
            // their final positions are known).
            for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                let r = a.row_idx[k];
                let pre = pinv_pre[r];
                let sv = a.values[k] * T::from_real(rs_inv[r]);
                if pre < bs {
                    f_rowidx.push(bpinv[pre]);
                    f_val.push(sv);
                } else {
                    x[pre] = sv;
                }
            }

            // Pass 3 — numeric update in topological order (reverse
            // post-order): each pivotal node's final value feeds its L column
            // into the remaining work vector, and becomes a U entry.
            for &u in topo.iter().rev() {
                let p = bpinv[u];
                let xu = x[u];
                x[u] = T::zero();
                u_rowidx.push(p);
                u_val.push(xu);
                for k in l_colptr[p]..l_colptr[p + 1] {
                    let lr = l_rowidx[k];
                    x[lr] = x[lr] - xu * l_val[k];
                }
            }

            // Pivot: max-magnitude candidate, overridden by the diagonal
            // (pre-position `j`) when it clears the threshold.
            let mut piv = UNSET;
            let mut maxmag = 0.0f64;
            for &np in &nonpiv {
                let m = x[np].magnitude();
                if m > maxmag {
                    maxmag = m;
                    piv = np;
                }
            }
            if piv == UNSET || maxmag == 0.0 || !maxmag.is_finite() {
                // Clear the candidates before failing so the work vector
                // never leaks values into a hypothetical caller retry.
                for &np in &nonpiv {
                    x[np] = T::zero();
                }
                return Err(RslabError::SingularBasis { column: c });
            }
            if stamp[j] == sj && bpinv[j] == UNSET {
                let dm = x[j].magnitude();
                if dm > 0.0 && dm >= settings.pivot_tol * maxmag {
                    piv = j;
                }
            }

            let dval = x[piv];
            x[piv] = T::zero();
            bpinv[piv] = j;
            udiag.push(dval);
            for &np in &nonpiv {
                if np == piv {
                    continue;
                }
                // Keep structural zeros: the pattern must be value-independent
                // for the refactor replay.
                l_rowidx.push(np);
                l_val.push(x[np] / dval);
                x[np] = T::zero();
            }
            l_colptr.push(l_rowidx.len());
            u_colptr.push(u_rowidx.len());
            f_colptr.push(f_rowidx.len());
        }

        // The block is fully pivoted: fix its L row indices from pre-pivot to
        // final positions (U and F indices were final at push time).
        for k in l_colptr[bs]..l_colptr[be] {
            l_rowidx[k] = bpinv[l_rowidx[k]];
        }
    }

    let mut row_perm = vec![0usize; n];
    let mut pinv = vec![0usize; n];
    for (&fin, &orig) in bpinv.iter().zip(&sym.pre_row_perm) {
        debug_assert_ne!(fin, UNSET);
        row_perm[fin] = orig;
        pinv[orig] = fin;
    }

    Ok(KluFactors {
        n,
        nnz_a: sym.nnz,
        block_ptr: sym.block_ptr.clone(),
        row_perm,
        pinv,
        col_perm: sym.col_perm.clone(),
        rs_inv,
        scaled: settings.row_scaling,
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
    })
}

impl<T: Scalar> KluSolver<T> {
    /// One-shot analyze + factor with the given settings. Skips the a-priori
    /// estimate and stage timing (empty [`diagnostics`](Self::diagnostics)),
    /// like [`LuSolver::factor`](crate::LuSolver::factor); use the phased
    /// [`KluSymbolic::factor`] for populated diagnostics.
    pub fn factor(a: &GeneralCsc<T>, settings: &KluSettings) -> Result<Self, RslabError> {
        let sym = KluSymbolic::analyze_with(a, settings)?;
        let factors = factor_impl(&sym, a, settings)?;
        Ok(Self {
            factors,
            diagnostics: crate::diagnostics::Diagnostics::default(),
        })
    }

    /// Per-call diagnostics: measured factor/refactor stages, fill, and the
    /// a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate).
    /// Populated by the phased [`KluSymbolic::factor`]; empty for the
    /// one-shot [`factor`](Self::factor).
    pub fn diagnostics(&self) -> &crate::diagnostics::Diagnostics {
        &self.diagnostics
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

    /// Solve the transposed system `Aᵀ x = b` with the **same** factorization.
    ///
    /// This is the plain transpose, NOT the conjugate transpose: for a complex
    /// adjoint solve `Aᴴ x = b`, conjugate `b` before and `x` after. (This
    /// matches the convention of the usual sparse-LU transpose solves, and is
    /// what implicit-function adjoints over holomorphic residuals need.)
    ///
    /// The stored form is `A = Rs · P_rᵀ · M · C` with `M` the block-upper
    /// (BTF) permuted, row-scaled matrix and `M_bb = L_b U_b` per diagonal
    /// block, so `Aᵀ x = b` is `Mᵀ (P_r Rs x) = C b`: gather `b` through the
    /// column permutation, run the transposed block substitution (blocks
    /// forward, per block `Uᵀ` forward then `Lᵀ` backward, off-block `Fᵀ`
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
        // w = C·b: position k of the permuted system reads b at its column.
        let mut w = vec![T::zero(); f.n];
        for (k, &c) in f.col_perm.iter().enumerate() {
            w[k] = b[c];
        }
        self.solve_permuted_transpose(&mut w);
        // x = Rs⁻¹ · P_rᵀ · w: scatter through the row permutation, then undo
        // the row scaling (Rs is diagonal, so it transposes onto the solution).
        let mut xout = vec![T::zero(); f.n];
        for (k, &orig) in f.row_perm.iter().enumerate() {
            xout[orig] = w[k] * T::from_real(f.rs_inv[orig]);
        }
        Ok(xout)
    }

    /// The transposed block substitution on the permuted vector: `Mᵀ` is block
    /// **lower** triangular (the transpose of the BTF block-upper `M`), so the
    /// blocks run forward, and within a block `M_bbᵀ = U_bᵀ L_bᵀ` solves as
    /// `Uᵀ` (lower, diagonal `udiag`) forward then `Lᵀ` (unit upper) backward.
    /// Column `j` of the stored `U`/`L`/`F` is row `j` of the transpose, so
    /// every inner loop is a gather over the existing column storage.
    fn solve_permuted_transpose(&self, w: &mut [T]) {
        let f = &self.factors;
        for b in 0..f.block_ptr.len() - 1 {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            // Fᵀ: this block's rows read the already-solved earlier blocks.
            for j in bs..be {
                let mut acc = w[j];
                for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                    acc = acc - f.f_val[k] * w[f.f_rowidx[k]];
                }
                w[j] = acc;
            }
            // Uᵀ (lower triangular, diagonal `udiag`) forward within the block.
            for j in bs..be {
                let mut acc = w[j];
                for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                    acc = acc - f.u_val[k] * w[f.u_rowidx[k]];
                }
                w[j] = acc / f.udiag[j];
            }
            // Lᵀ (unit upper) backward within the block.
            for j in (bs..be).rev() {
                let mut acc = w[j];
                for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                    acc = acc - f.l_val[k] * w[f.l_rowidx[k]];
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
    /// (SIMD-friendly and cache-reusing) — the sparse-scalar factorization
    /// cannot use BLAS-3, but the wide solve can still vectorize across the
    /// right-hand sides. Each column's operation order is identical to
    /// [`solve`](Self::solve), so the result is bit-identical to `nrhs`
    /// single solves.
    pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
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
            // L (unit lower) forward within the block.
            for j in bs..be {
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                    let (lr, lv) = (f.l_rowidx[k], f.l_val[k]);
                    let row = &mut w[lr * nrhs..lr * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&xj) {
                        *r = *r - lv * x;
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
                    let (ur, uv) = (f.u_rowidx[k], f.u_val[k]);
                    let row = &mut w[ur * nrhs..ur * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&xj) {
                        *r = *r - uv * x;
                    }
                }
            }
            // Off-block columns feed the rows of earlier blocks.
            for j in bs..be {
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                    let (fr, fv) = (f.f_rowidx[k], f.f_val[k]);
                    let row = &mut w[fr * nrhs..fr * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&xj) {
                        *r = *r - fv * x;
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
        let n = self.factors.n;
        if a.n != n || b.len() != n {
            return Err(RslabError::DimensionMismatch {
                expected: n,
                got: b.len(),
            });
        }
        let mut x = self.solve(b)?;
        let mut ax = vec![T::zero(); n];
        let mut best_x = x.clone();
        let mut best_res = f64::INFINITY;
        for _ in 0..=max_iter {
            a.matvec(&x, &mut ax);
            let r: Vec<T> = b.iter().zip(&ax).map(|(&bi, &axi)| bi - axi).collect();
            let res = r.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            if res < best_res {
                best_res = res;
                best_x.clone_from(&x);
            }
            if res == 0.0 {
                break;
            }
            let dx = self.solve(&r)?;
            for (xi, &d) in x.iter_mut().zip(&dx) {
                *xi = *xi + d;
            }
        }
        Ok(best_x)
    }

    /// The block forward/backward substitution on the permuted/scaled vector.
    fn solve_permuted(&self, w: &mut [T]) {
        let f = &self.factors;
        for b in (0..f.block_ptr.len() - 1).rev() {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            // L (unit lower) forward within the block.
            for j in bs..be {
                let xj = w[j];
                if xj != T::zero() {
                    for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                        w[f.l_rowidx[k]] = w[f.l_rowidx[k]] - f.l_val[k] * xj;
                    }
                }
            }
            // U backward within the block.
            for j in (bs..be).rev() {
                let xj = w[j] / f.udiag[j];
                w[j] = xj;
                if xj != T::zero() {
                    for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                        w[f.u_rowidx[k]] = w[f.u_rowidx[k]] - f.u_val[k] * xj;
                    }
                }
            }
            // Off-block columns feed the rows of earlier blocks.
            for j in bs..be {
                let xj = w[j];
                if xj != T::zero() {
                    for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                        w[f.f_rowidx[k]] = w[f.f_rowidx[k]] - f.f_val[k] * xj;
                    }
                }
            }
        }
    }

    /// Numeric-only refactorization: replay the stored pattern and pivot
    /// sequence on new values with the **same** sparsity pattern. No symbolic
    /// work, no pivot search — the fast path for frequency sweeps and Newton
    /// steps. Fails with a pattern-mismatch error if `a`'s pattern deviates
    /// from the factored one, and with [`RslabError::SingularBasis`] if a
    /// frozen pivot becomes zero (re-`factor` with pivoting in that case).
    /// After an error the factorization is invalid; a subsequent successful
    /// `refactor` or a fresh `factor` makes it valid again.
    pub fn refactor(&mut self, a: &GeneralCsc<T>) -> Result<(), RslabError> {
        a.validate()?;
        let t = std::time::Instant::now();
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

        let mut x = vec![T::zero(); f.n];
        let mut scattered: Vec<usize> = Vec::new();

        for b in 0..f.block_ptr.len() - 1 {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            for j in bs..be {
                let c = f.col_perm[j];
                // Scatter (final position space). Off-block entries must hit
                // the stored F pattern slot-for-slot — the same storage-order
                // walk as at factor time.
                scattered.clear();
                let mut fcur = f.f_colptr[j];
                for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                    let r = a.row_idx[k];
                    let fin = f.pinv[r];
                    let sv = a.values[k] * T::from_real(rs_inv[r]);
                    if fin < bs {
                        if fcur >= f.f_colptr[j + 1] || f.f_rowidx[fcur] != fin {
                            return Err(pattern_mismatch());
                        }
                        f.f_val[fcur] = sv;
                        fcur += 1;
                    } else if fin >= be {
                        return Err(pattern_mismatch());
                    } else {
                        x[fin] = sv;
                        scattered.push(fin);
                    }
                }
                if fcur != f.f_colptr[j + 1] {
                    return Err(pattern_mismatch());
                }

                // Replay the elimination in the stored topological order.
                for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                    let p = f.u_rowidx[k];
                    let xu = x[p];
                    x[p] = T::zero();
                    f.u_val[k] = xu;
                    for kl in f.l_colptr[p]..f.l_colptr[p + 1] {
                        let lr = f.l_rowidx[kl];
                        x[lr] = x[lr] - xu * f.l_val[kl];
                    }
                }
                let d = x[j];
                x[j] = T::zero();
                if d.magnitude() == 0.0 || !d.is_finite() {
                    return Err(RslabError::SingularBasis { column: c });
                }
                f.udiag[j] = d;
                for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                    let lr = f.l_rowidx[k];
                    f.l_val[k] = x[lr] / d;
                    x[lr] = T::zero();
                }
                // Every scattered position must now be consumed: a leftover
                // value is an entry outside the stored pattern (the pattern
                // changed) and would otherwise be silently dropped.
                for &pos in &scattered {
                    if x[pos] != T::zero() {
                        return Err(pattern_mismatch());
                    }
                }
            }
        }
        f.rs_inv = rs_inv;
        let entry = (std::mem::size_of::<T>() + std::mem::size_of::<usize>()) as u64;
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

    /// Max-norm relative residual of the *transposed* system `Aᵀ x = b`.
    fn resid_t<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
        resid(&a.transpose(), x, b)
    }

    #[test]
    fn klu_solve_transpose_matches_factored_transpose() {
        // Reducible circuit-shaped matrix: solve_transpose on A's factors must
        // agree with a fresh factorization of Aᵀ, and satisfy Aᵀ x = b.
        let a = circuit_like(200, 42);
        let b: Vec<f64> = (0..200).map(|i| ((i * 3) % 13) as f64 - 6.0).collect();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert!(s.n_blocks() >= 2, "bridge structure must be reducible");
        let x = s.solve_transpose(&b).unwrap();
        assert!(resid_t(&a, &x, &b) < 1e-12, "residual {}", resid_t(&a, &x, &b));
        let st = KluSolver::factor(&a.transpose(), &KluSettings::default()).unwrap();
        let xr = st.solve(&b).unwrap();
        let diff = x
            .iter()
            .zip(&xr)
            .map(|(&p, &q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(diff < 1e-9, "transpose solve vs factored transpose differ by {diff}");
    }

    #[test]
    fn klu_solve_transpose_complex_plain_not_conjugate() {
        // Complex: solve_transpose must solve the PLAIN transpose Aᵀ x = b
        // (adjoint convention: the caller conjugates for Aᴴ). Off-diagonal
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
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, (i % 3) as f64)).collect();
        let s = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        let x = s.solve_transpose(&b).unwrap();
        assert!(resid_t(&a, &x, &b) < 1e-12, "residual {}", resid_t(&a, &x, &b));
        // Aᴴ x = b via the documented conjugation recipe.
        let bc: Vec<Complex<f64>> = b.iter().map(|v| v.conj()).collect();
        let xh: Vec<Complex<f64>> = s.solve_transpose(&bc).unwrap().iter().map(|v| v.conj()).collect();
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
            KluSettings::default().with_btf(false).with_row_scaling(false),
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
        assert!(resid_t(&a2, &x, &b) < 1e-11, "residual {}", resid_t(&a2, &x, &b));
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
        let s = sym.factor(&a, &KluSettings::default()).unwrap();
        let d = s.diagnostics();
        assert_eq!(d.threads, 1);
        assert_eq!(d.factor_nnz as usize, s.factor_nnz());
        assert!(d.estimate.is_some());
        assert_eq!(d.stages.len(), 1);
        assert_eq!(d.stages[0].name, "klu-factor");

        let mut s2 = KluSolver::factor(&a, &KluSettings::default()).unwrap();
        assert!(s2.diagnostics().stages.is_empty());
        s2.refactor(&a).unwrap();
        assert_eq!(s2.diagnostics().stages.last().unwrap().name, "klu-refactor");
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
