//! Generic **unsymmetric** sparse LU factorization over any [`Scalar`] field -
//! the general (non-symmetric) complex path, complementing the symmetric LDL^T
//! path in [`crate::numeric::multifrontal_ldlt`].
//!
//! It targets matrices whose *values* are unsymmetric (e.g. MoM A-EFIE
//! near-field saddle preconditioners, where the symmetric and antisymmetric
//! parts are comparable) but reuses the full symmetric machinery: the
//! fill-reducing ordering, supernodes, assembly tree, level parallelism
//! ([`analyze`](crate::numeric::multifrontal_ldlt::analyze)) and the SIMD
//! `gemm` Schur kernel. Only the per-front kernel changes - an unsymmetric LU
//! producing separate `L` and `U` - and the analysis runs on the **symmetrized
//! pattern** `A union A^T` so the elimination structure carries fill for both
//! factors.
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
//!   kept in supernodal panel form ([`PanelFactor`], see [`LuNumeric`]), in
//!   factorization order: each supernode's finished panels become the stored
//!   factor, and [`LuNumeric::into_factors`] materializes the sparse `L` (CSC)
//!   and `U` (CSR) of [`LuFactors`] on demand. The default factor path is the
//!   supernodal **left-looking** kernel (low transient, no CB stack); the
//!   multifrontal path is opt-in via [`SolverSettings::with_method`].

use crate::error::RslabError;
use crate::numeric::blr::BlrMatrix;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::multifrontal_ldlt::{
    analyze_with, perturb_pivot, BlrMode, FactorMethod, SolverSettings, ZeroPivotAction,
};
use crate::numeric::panel_factor::{finish_panel, PanelArena, PanelFactor, PanelOut};
use crate::scalar::{fmadd, Scalar};
use crate::sparse::general::GeneralCsc;
use crate::symbolic::SymbolicFactorization;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Reusable dense front-buffer pool. The multifrontal driver factors thousands
/// of fronts, each needing a transient `nrow^2 ` working buffer. Allocating one
/// per front (and freeing it) churns the system allocator with large, varying
/// sizes; on Windows the heap retains the freed blocks rather than returning
/// them to the OS, so peak RSS balloons far above the live set (the OOM the
/// pure-per-front allocation caused). This pool recycles a handful of buffers
/// (~ the concurrency level) instead, capping the transient at the live set.
/// Shared with the symmetric multifrontal twin
/// ([`crate::numeric::multifrontal_ldlt`]), whose per-front `nrow^2` buffer has
/// the same fragmentation exposure.
pub(crate) struct FrontPool<T>(Mutex<Vec<Vec<T>>>);

/// Only buffers up to this many entries are recycled. The churning majority of
/// small/medium fronts (which drive fragmentation) stay pooled; the rare huge
/// root/separator fronts are freed immediately rather than pinning their (GB-
/// scale) capacity in the pool for the whole factorization. 4M entries ~ 64 MB
/// for `Complex<f64>` (front height ~2000).
const POOL_MAX_LEN: usize = 4_000_000;

impl<T: Scalar> FrontPool<T> {
    pub(crate) fn new() -> Self {
        FrontPool(Mutex::new(Vec::new()))
    }
    /// Take a buffer (reused if available) and zero-fill it to `len`.
    pub(crate) fn take(&self, len: usize) -> Vec<T> {
        let mut buf = self
            .0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
            .unwrap_or_default();
        buf.clear();
        buf.resize(len, T::zero());
        buf
    }
    /// Return a buffer for reuse, unless it is an oversized (huge-front) buffer
    /// whose capacity we do not want to pin in the pool - those are dropped.
    pub(crate) fn give(&self, buf: Vec<T>) {
        if buf.capacity() <= POOL_MAX_LEN {
            self.0.lock().unwrap_or_else(|p| p.into_inner()).push(buf);
        }
    }
}

/// Always-on (relaxed, ~free) count of `lu_ll_factor_node` calls in flight for
/// THIS factorization - the fork-dispatch signal, ported from the LDLT twin:
/// in the separator-chain phase (few active nodes) even a small node's
/// cmod/cdiv should fork (workers idle, little foreign work for a blocked join
/// to steal); in the busy phase small nodes stay strictly serial (join-steal
/// guard - a blocked join steals whole sibling subtrees and stalls this node's
/// dependents). Scheduling-only: the dispatch never changes the computed bits
/// (identical per-entry accumulation order on every path), so a racy counter
/// read is benign and bit-identity across thread counts holds.
struct LuActiveGuard<'a>(&'a AtomicUsize);
impl Drop for LuActiveGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}
thread_local! {
    /// Per-worker global->front-local index scratch (`usize`, scalar-independent),
    /// reused across every front a thread factors. Held at the all-`usize::MAX`
    /// invariant between uses; the assembly takes it, sets only the live front
    /// rows, and restores it. Replaces the old `map_init` workspace now that the
    /// driver is a work-stealing tree recursion rather than a level `par_iter`.
    static GLOC_SCRATCH: std::cell::RefCell<Vec<Li>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

// --- BLR contribution-block compression (opt-in `RLA_BLR_CB`) --------------
//
// The dominant factorization transient is the live contribution-block (CB)
// stack - `sum cnrow^2` across the active assembly frontier, ~5x the L/U volume and
// the source of the memory spike / OOM. A CB is a frontal Schur complement of a
// smooth (MoM near-field) operator, so its off-diagonal tiles are numerically
// low-rank. Storing each large CB block-low-rank on the stack - and densifying
// it tile-by-tile only at the parent's extend-add - shrinks the live stack by
// the tile compression ratio at a small per-tile densify transient. The factor
// then becomes approximate (a preconditioner), exactly the LU/MoM use case
// where GMRES refinement absorbs the BLR tolerance. The fast dense `gemm`
// `lu_front` kernel is unchanged; only the CB *storage* is compressed.

/// A front's contribution block: dense, or BLR-compressed for the stack.
enum Contribution<T: crate::scalar::Scalar> {
    Dense(Vec<T>),
    Blr(BlrMatrix<T>),
}

impl<T: Scalar> Contribution<T> {
    /// Extend-add this CB into front `f` (`nrow`-tall, column-major) at the
    /// parent-local positions `loc` (CB index -> front-local). `cn` is the CB
    /// dimension (used by the dense case; the BLR case carries its own).
    fn extend_add_into(&self, loc: &[usize], cn: usize, f: &mut [T], nrow: usize) {
        match self {
            Contribution::Dense(cb) => {
                for jc in 0..cn {
                    let frow = loc[jc] * nrow;
                    let cb_col = &cb[jc * cn..jc * cn + cn];
                    for ic in 0..cn {
                        let dst = frow + loc[ic];
                        f[dst] = f[dst] + cb_col[ic];
                    }
                }
            }
            Contribution::Blr(blr) => {
                // Densify one tile at a time (<= bxb transient) and scatter.
                for ib in 0..blr.nbr {
                    let (r0, bm) = blr.row_extent(ib);
                    for jb in 0..blr.nbc {
                        let (c0, bn) = blr.col_extent(jb);
                        let tile = blr.block(ib, jb).to_dense();
                        for jj in 0..bn {
                            let frow = loc[c0 + jj] * nrow;
                            let tcol = &tile[jj * bm..jj * bm + bm];
                            for ii in 0..bm {
                                let dst = frow + loc[r0 + ii];
                                f[dst] = f[dst] + tcol[ii];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Per-front unsymmetric partial-factorization output, in elimination order
/// (static pivoting -> no interchange, so front-local order is pivot order).
struct FrontLu<T> {
    nrow: usize,
    nelim: usize,
    /// Unit-lower `L` of the front, `nrow x nelim` column-major (multipliers
    /// below the diagonal; unit diagonal implicit).
    l: Vec<T>,
    /// Upper `U` of the front as `nrow x nelim` column-major over the *row*
    /// index: `u[c*nrow + r]` is `U(c, r)` for `r >= c` (the eliminated row `c`
    /// against front position `r`). The diagonal `u[c*nrow + c]` is the pivot.
    u: Vec<T>,
    /// Front row permutation from partial pivoting (`rperm[k]` is the original
    /// front-local row that supplied pivot position `k`). Identity on trailing
    /// rows. Drives the global row permutation `perm_row`.
    rperm: Vec<usize>,
    n_perturbed: usize,
}

/// Reassembled per-front result retained for the global pass and the parent's
/// extend-add.
struct NodeLu<T: Scalar> {
    front: FrontLu<T>,
    row_indices: Vec<usize>,
    /// The `cnrow x cnrow` contribution block `A22 - L21*U12` - dense, or
    /// BLR-compressed for the stack (opt-in `RLA_BLR_CB`).
    contrib: Contribution<T>,
}

/// Stored unsymmetric LU factors, in factorization order. Solve with
/// [`solve_lu`]. The factored system is `P^T A P = L U`, `perm[e]` mapping
/// factorization position `e` to the original index.
pub struct LuFactors<T> {
    pub n: usize,
    /// `L` in CSC (unit lower, explicit unit diagonal).
    pub l_col_ptr: Vec<usize>,
    pub l_row_idx: Vec<usize>,
    pub l_values: Vec<T>,
    /// `U` in CSR (upper; the diagonal entry carries the pivot).
    pub u_row_ptr: Vec<usize>,
    pub u_col_idx: Vec<usize>,
    pub u_values: Vec<T>,
    /// Column permutation: factorization position -> original column index
    /// (`P^T A P = L U`, the fill-reducing ordering).
    pub perm: Vec<usize>,
    /// Row permutation: factorization position -> original row index. Differs
    /// from `perm` when partial pivoting interchanged rows.
    pub perm_row: Vec<usize>,
    /// Two-sided equilibration: the factor is of `A_hat = diag(d_row)*A*diag(d_col)`.
    /// Solve applies `D_r` to the RHS and `D_c` to the result. Both length `n`.
    pub d_row: Vec<f64>,
    pub d_col: Vec<f64>,
    /// Column partition of the factor into supernodes (the fronts): columns
    /// `supernode_ptr[s]..supernode_ptr[s + 1]` of `L` and rows of `U` share
    /// one structure. Length `ns + 1`; empty when unknown.
    pub supernode_ptr: Vec<usize>,
    /// Parent of every supernode in the assembly tree (`usize::MAX` for a
    /// root); empty when unknown.
    pub supernode_parent: Vec<usize>,
    /// Number of statically perturbed pivots.
    pub n_perturbed: usize,
    /// Thread policy the **solve phase** should honour (issue #9): resolved from
    /// the factorization's [`SolverSettings::threads`] so an iterative solve using
    /// this factor as a preconditioner runs its parallel orthogonalization in a
    /// pool of the **same** width - factor and solve share one concurrency budget
    /// instead of the solve silently fanning out over the global pool.
    /// [`Threads::Ambient`](crate::Threads::Ambient) means "use the caller's
    /// current pool" (the solver-in-the-loop path); otherwise a concrete
    /// [`Threads::Fixed`](crate::Threads::Fixed) worker count.
    pub solve_threads: crate::numeric::multifrontal_ldlt::Threads,
}

/// The numeric result of a sparse LU factorization: the unit lower `L` and
/// the transposed upper factor `U^T` (its diagonal in the panel) in
/// supernodal panel form (the storage the solves run on, written by the
/// drivers without a copy), the permutations, the scalings and the outcome.
/// [`into_factors`](Self::into_factors) materializes the compressed
/// [`LuFactors`] (`L` as CSC, `U` as CSR) for the reference solves.
#[derive(Clone, Debug)]
pub struct LuNumeric<T> {
    /// `L` in panel form, in elimination order (unit diagonal implicit).
    pub l: PanelFactor<T>,
    /// `U^T` in panel form: column `c` of the panel is row `c` of `U`, the
    /// diagonal of `U` at the panel's diagonal.
    pub ut: PanelFactor<T>,
    /// `perm[e]` is the original column eliminated at position `e`.
    pub perm: Vec<usize>,
    /// `perm_row[e]` is the original row that became pivot row `e`.
    pub perm_row: Vec<usize>,
    /// Row scaling applied before the factorization.
    pub d_row: Vec<f64>,
    /// Column scaling applied before the factorization.
    pub d_col: Vec<f64>,
    /// Supernode tree over the factor's supernodes (`usize::MAX` for a root).
    pub supernode_parent: Vec<usize>,
    /// Pivots perturbed by the static regularization.
    pub n_perturbed: usize,
    /// Structural panel slots holding an exact zero (the symmetrized
    /// pattern, cancellation or `drop_tol`); the stored nonzeros are
    /// `l.nnz() + ut.nnz() - n_zeros`.
    pub n_zeros: usize,
    /// Thread policy the solves inherit.
    pub solve_threads: crate::numeric::multifrontal_ldlt::Threads,
}

impl<T: Scalar> LuNumeric<T> {
    /// Dimension.
    pub fn n(&self) -> usize {
        self.l.n
    }

    /// Stored fill `nnz(L) + nnz(U)`: the structural panel entries minus the
    /// slots holding an exact zero.
    pub fn factor_nnz(&self) -> usize {
        self.l.nnz() + self.ut.nnz() - self.n_zeros
    }

    /// Bytes of the two panel factors.
    pub fn bytes(&self) -> usize {
        self.l.bytes() + self.ut.bytes()
    }

    fn shell(&self) -> LuFactors<T> {
        LuFactors {
            n: self.l.n,
            l_col_ptr: Vec::new(),
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
            u_row_ptr: Vec::new(),
            u_col_idx: Vec::new(),
            u_values: Vec::new(),
            perm: self.perm.clone(),
            perm_row: self.perm_row.clone(),
            d_row: self.d_row.clone(),
            d_col: self.d_col.clone(),
            supernode_ptr: self.l.sn_col.iter().map(|&c| c as usize).collect(),
            supernode_parent: self.supernode_parent.clone(),
            n_perturbed: self.n_perturbed,
            solve_threads: self.solve_threads,
        }
    }

    /// The compressed form for the reference solves (copies both factors).
    pub fn into_factors(self) -> LuFactors<T> {
        let mut f = self.shell();
        let (l_col_ptr, l_row_idx, l_values) = self.l.to_csc(true);
        let (u_row_ptr, u_col_idx, u_values) = self.ut.to_csc(false);
        f.l_col_ptr = l_col_ptr;
        f.l_row_idx = l_row_idx;
        f.l_values = l_values;
        f.u_row_ptr = u_row_ptr;
        f.u_col_idx = u_col_idx;
        f.u_values = u_values;
        f
    }

    /// Split into the two panel factors and an [`LuFactors`] shell (empty CSC
    /// arrays) carrying the permutations and scalings for the solver.
    pub(crate) fn into_parts(self) -> (PanelFactor<T>, PanelFactor<T>, LuFactors<T>) {
        let shell = self.shell();
        (self.l, self.ut, shell)
    }
}

impl<T: Scalar> LuFactors<T> {
    /// Stored fill: `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.l_values.len() + self.u_values.len()
    }
}

/// Lower triangle of the symmetrized pattern `A union A^T` as CSC `(col_ptr,
/// row_idx)`. The symmetric analysis needs a structurally symmetric pattern so
/// the elimination tree carries fill for both `L` and `U`.
fn symmetrized_lower_pattern<T: Scalar>(a: &GeneralCsc<T>) -> (Vec<usize>, Vec<usize>) {
    let n = a.n;
    // Counting-scatter (no `BTreeSet`: no per-element heap allocation, no
    // pointer-chasing). Each entry contributes a lower-triangle pair `(hi, lo)`
    // to bucket `lo`; buckets are then sorted + deduped into CSC.
    let mut counts = vec![0usize; n];
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let lo = if i < j { i } else { j };
            counts[lo] += 1;
        }
    }
    let mut start = vec![0usize; n + 1];
    for j in 0..n {
        start[j + 1] = start[j] + counts[j];
    }
    let total = start[n];
    let mut scattered = vec![0usize; total];
    let mut cursor = start[..n].to_vec();
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            let (hi, lo) = if i >= j { (i, j) } else { (j, i) };
            scattered[cursor[lo]] = hi;
            cursor[lo] += 1;
        }
    }
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx = Vec::with_capacity(total);
    for j in 0..n {
        let seg = &mut scattered[start[j]..start[j + 1]];
        seg.sort_unstable();
        let mut last = usize::MAX;
        for &i in seg.iter() {
            if i != last {
                row_idx.push(i);
                last = i;
            }
        }
        col_ptr.push(row_idx.len());
    }
    (col_ptr, row_idx)
}

/// Partially factor the first `ncol` fully-summed columns of a dense full front
/// `f` (`nrow x nrow`, column-major) by unsymmetric LU with static pivoting.
/// The trailing `[ncol, nrow)` block is returned as the contribution block
/// `A22 - L21*U12` (computed by one `gemm`).
fn lu_front<T: Scalar>(
    f: &mut [T],
    nrow: usize,
    ncol: usize,
    perturb_floor: Option<f64>,
    blr: BlrMode,
    kt: KernelTuning,
) -> Result<(FrontLu<T>, Contribution<T>), RslabError> {
    let n = nrow;
    let mut pivots = vec![T::zero(); ncol];
    let mut n_perturbed = 0usize;
    // Row permutation of the front (partial pivoting interchanges rows). Only
    // the fully-summed rows [0, ncol) are ever interchanged.
    let mut rperm: Vec<usize> = (0..nrow).collect();

    // Blocked right-looking LU (LAPACK getrf-style): factor the fully-summed
    // columns in panels of width `NB`. Each panel is factored unblocked (getf2,
    // within-block partial pivoting), then the dominant trailing update runs as
    // a single SIMD `gemm` (rank-`NB`) - routing the O(ncol^2*nrow) work through
    // the complex BLAS-3 kernel instead of scalar BLAS-2 column sweeps. This is
    // the structure MKL/PARDISO use for the supernodal panel.
    // Panel width. Smaller than the LAPACK-typical 64 because the `gemm` complex
    // kernel is very fast (~460 Gflop/s rank-64) while the unblocked getf2 panel
    // is serial BLAS-2: a narrower panel shifts work off the slow serial path
    // onto the fast parallel GEMM. NB=32 measured best on the MoM fronts.
    const NB: usize = 32;
    let mut kb = 0;
    while kb < ncol {
        kt.interrupted()?;
        let ke = (kb + NB).min(ncol);
        // --- Panel factor (getf2) over columns [kb, ke), full height ---
        for k in kb..ke {
            // Partial pivoting within the fully-summed block (rows [k, ncol));
            // compare squared magnitudes (same argmax, no per-candidate sqrt).
            let mut p = k;
            let mut best = f[k * n + k].magnitude_sq();
            for i in (k + 1)..ncol {
                let m = f[k * n + i].magnitude_sq();
                if m > best {
                    best = m;
                    p = i;
                }
            }
            if p != k {
                for c in 0..nrow {
                    f.swap(c * n + k, c * n + p);
                }
                rperm.swap(k, p);
            }
            let mut piv = f[k * n + k];
            match perturb_floor {
                Some(floor) if piv.magnitude() < floor => {
                    piv = perturb_pivot(piv, floor);
                    n_perturbed += 1;
                }
                None if piv == T::zero() => return Err(RslabError::NumericallyRankDeficient),
                _ => {}
            }
            f[k * n + k] = piv;
            let pinv = piv.recip();
            // L multipliers: column k below the diagonal (full height).
            for i in (k + 1)..n {
                f[k * n + i] = f[k * n + i] * pinv;
            }
            // Within-panel Schur update only (columns [k+1, ke)); the trailing
            // block is deferred to the panel GEMM.
            for j in (k + 1)..ke {
                let ukj = f[j * n + k];
                if ukj != T::zero() {
                    for i in (k + 1)..n {
                        f[j * n + i] = f[j * n + i] - f[k * n + i] * ukj;
                    }
                }
            }
        }
        let pw = ke - kb; // panel width
                          // --- TRSM: U[kb:ke, ke:nrow] = L11^-1 * A[kb:ke, ke:nrow] ---
                          // L11 is the unit-lower `pwxpw` diagonal block; forward-substitute each
                          // trailing column over the panel rows.
        for j in ke..n {
            for r in (kb + 1)..ke {
                let mut s = f[j * n + r];
                for i in kb..r {
                    s = s - f[i * n + r] * f[j * n + i]; // L11(r,i)*U12(i,j)
                }
                f[j * n + r] = s;
            }
        }
        // --- GEMM: A[ke:, ke:] -= L[ke:, kb:ke] * U[kb:ke, ke:] (rank-`pw`) ---
        let mt = n - ke; // trailing rows = cols
        if mt > 0 && pw > 0 {
            let flops = (mt as u128) * (mt as u128) * (pw as u128);
            let par = if flops >= kt.par_cdiv as u128 {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            // SAFETY: L21 (rows>=ke x cols[kb,ke)), U12 (rows[kb,ke) x cols>=ke)
            // and the A22 dst (rows>=ke x cols>=ke) are pairwise-disjoint
            // sub-blocks of `f`, all in bounds under col-stride `n`. `T` is a
            // supported gemm element type.
            let base = f.as_mut_ptr();
            unsafe {
                gemm::gemm(
                    mt,
                    mt,
                    pw,
                    base.add(ke * n + ke),
                    n as isize,
                    1,
                    true,
                    base.add(kb * n + ke) as *const T,
                    n as isize,
                    1,
                    base.add(ke * n + kb) as *const T,
                    n as isize,
                    1,
                    T::one(),
                    -T::one(),
                    false,
                    false,
                    false,
                    par,
                );
            }
        }
        kb = ke;
    }
    // Pivots from the factored diagonal of the fully-summed block.
    for k in 0..ncol {
        pivots[k] = f[k * n + k];
    }

    // Extract L (nrow x ncol col-major, unit lower) and U (nrow x ncol
    // col-major over the row index, with the pivot on the diagonal), plus the
    // contribution block (`f`'s Schur-updated trailing A22 = A22 - L21*U12).
    // Kept serial: the work-stealing driver already overlaps each front's serial
    // tail with other subtrees' work, so per-front extraction parallelism finds
    // no idle threads to use (measured: no gain).
    let one = T::one();
    let cnrow = nrow - ncol;
    let mut l = vec![T::zero(); nrow * ncol];
    let mut u = vec![T::zero(); nrow * ncol];
    let mut cb = vec![T::zero(); cnrow * cnrow];
    for c in 0..ncol {
        l[c * nrow + c] = one;
        u[c * nrow + c] = pivots[c];
        for r in (c + 1)..nrow {
            l[c * nrow + r] = f[c * nrow + r]; // L(r, c)
            u[c * nrow + r] = f[r * nrow + c]; // U(c, r)
        }
    }
    // Each CB column is a contiguous run of `f`'s trailing block -> memcpy.
    for c in 0..cnrow {
        let base = (ncol + c) * n + ncol;
        cb[c * cnrow..c * cnrow + cnrow].copy_from_slice(&f[base..base + cnrow]);
    }

    // Compress large CBs for the stack (BlrMode::ContributionBlocks). The
    // off-diagonal tiles of the frontal Schur complement are low-rank; storing
    // them compressed shrinks the live CB-stack transient. Densified
    // tile-by-tile at the parent extend-add.
    let contribution = match blr {
        BlrMode::ContributionBlocks {
            eps,
            min_cnrow,
            b,
            adaptive,
        } if cnrow >= min_cnrow => Contribution::Blr(BlrMatrix::from_dense_with(
            &cb, cnrow, cnrow, b, eps, adaptive,
        )),
        _ => Contribution::Dense(cb),
    };
    Ok((
        FrontLu {
            nrow,
            nelim: ncol,
            l,
            u,
            rperm,
            n_perturbed,
        },
        contribution,
    ))
}

/// Factor one supernode's full front: build the (symmetric-pattern) row
/// structure, assemble the owned columns (L side), owned rows in trailing
/// columns (U side) and children contribution blocks, then LU-factor.
#[allow(clippy::too_many_arguments)]
fn factor_one_node_lu<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &GeneralCsc<T>,
    a_perm_t: &GeneralCsc<T>,
    child_refs: &[&NodeLu<T>],
    perturb_floor: Option<f64>,
    blr: BlrMode,
    pool: &FrontPool<T>,
    kt: KernelTuning,
) -> Result<NodeLu<T>, RslabError> {
    let snode = &sym.supernodes[s];
    let n = sym.n;
    let ncol = snode.ncol;
    let own_last = snode.first_col + ncol;

    // Front row structure: own columns ++ sorted trailing rows (symmetrized
    // pattern of own columns plus children contribution rows).
    let mut trailing: Vec<usize> = Vec::new();
    for j in snode.first_col..own_last {
        for k in sym.permuted_pattern.col_ptr[j]..sym.permuted_pattern.col_ptr[j + 1] {
            let r = sym.permuted_pattern.row_idx[k];
            if r >= own_last {
                trailing.push(r);
            }
        }
    }
    for child in child_refs {
        for &r in &child.row_indices[child.front.nelim..] {
            if r >= own_last {
                trailing.push(r);
            }
        }
    }
    trailing.sort_unstable();
    trailing.dedup();
    let mut ri = Vec::with_capacity(ncol + trailing.len());
    ri.extend(snode.first_col..own_last);
    ri.extend(trailing);
    let nrow = ri.len();

    // Front buffer (transient `nrow^2`), drawn from the reuse pool so the
    // thousands of large per-front buffers become a handful of recycled ones -
    // the fragmentation fix for the transient-memory OOM. Returned via
    // `pool.give` once `lu_front` has consumed it below.
    let mut fbuf: Vec<T> = pool.take(nrow * nrow);
    let f = &mut fbuf[..];

    // Take the thread-local global->local scratch for the assembly (held at the
    // all-`usize::MAX` invariant). It is returned before `lu_front` so the
    // front GEMM's work-stealing tasks can never re-enter the borrow.
    let mut gloc = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if gloc.len() < n {
        gloc.resize(n, Li::MAX);
    }
    for (li, &g) in ri.iter().enumerate() {
        gloc[g] = li as Li;
    }

    // Owned columns (full): scatter a_perm column c into front column p.
    for p in 0..ncol {
        let c = snode.first_col + p;
        for k in a_perm.col_ptr[c]..a_perm.col_ptr[c + 1] {
            let g = a_perm.row_idx[k];
            let lr = gloc[g];
            if lr != Li::MAX {
                f[p * nrow + lr as usize] = f[p * nrow + lr as usize] + a_perm.values[k];
            }
        }
    }
    // Owned rows, trailing columns only (U12): scatter a_perm^T column r (=
    // a_perm row r) into front row p for trailing front columns.
    for p in 0..ncol {
        let r = snode.first_col + p;
        for k in a_perm_t.col_ptr[r]..a_perm_t.col_ptr[r + 1] {
            let g = a_perm_t.row_idx[k];
            let lc = gloc[g];
            if lc != Li::MAX && lc as usize >= ncol {
                f[lc as usize * nrow + p] = f[lc as usize * nrow + p] + a_perm_t.values[k];
            }
        }
    }

    // Extend-add each child's full contribution block. Map the child's
    // contribution rows to parent-front-local positions ONCE per child (`loc`,
    // reused across children) rather than re-indexing `gloc` inside the inner
    // loop - turning the cn^2 global->local lookups into cn, and slicing the
    // contiguous contribution column for the inner accumulation.
    let mut loc: Vec<usize> = Vec::new();
    for child in child_refs {
        let cn = child.front.nrow - child.front.nelim;
        let crows = &child.row_indices[child.front.nelim..];
        loc.clear();
        loc.extend(crows.iter().map(|&g| gloc[g] as usize));
        child.contrib.extend_add_into(&loc, cn, f, nrow);
    }

    // Restore the all-`Li::MAX` invariant and return the scratch to the
    // thread-local before `lu_front` (which spawns work-stealing GEMM tasks).
    for &g in &ri {
        gloc[g] = Li::MAX;
    }
    GLOC_SCRATCH.with(|c| *c.borrow_mut() = gloc);

    let (front, contrib) = lu_front(f, nrow, ncol, perturb_floor, blr, kt)?;
    // `lu_front` has copied L/U/CB out; recycle the front buffer.
    pool.give(fbuf);
    Ok(NodeLu {
        front,
        row_indices: ri,
        contrib,
    })
}

/// Fold the MC64 row matching back into the factors: pivot rows and the row
/// scaling are reported in `A`'s row indices, as the solves expect.
fn finish_matching<T: Scalar>(fac: &mut LuNumeric<T>, lusym: &LuSymbolic) {
    if let Some(m) = &lusym.matching {
        for e in fac.perm_row.iter_mut() {
            *e = m.row_of[*e];
        }
        fac.d_row = m.r.clone();
    }
}

/// A supernode's own factor plus the flat `(supernode-id, factor)` list for the
/// rest of its subtree - the return shape of [`factor_subtree`].
type SubtreeFactors<T> = (NodeLu<T>, Vec<(usize, NodeLu<T>)>);

/// Reusable symbolic analysis for the unsymmetric LU path - the symmetrized
/// pattern `A union A^T` analyzed once. Pass to [`factor_general_lu_numeric`] for
/// each value-set that shares the pattern (frequency sweep / Newton).
/// The MC64 row matching the analysis was done under: the factored matrix
/// is `B = diag(r) P A diag(c)` with `B` row `i` = `A` row `row_of[i]`.
struct LuMatching {
    row_of: Vec<usize>,
    /// `A`-row scaling.
    r: Vec<f64>,
    c: Vec<f64>,
}

pub struct LuSymbolic {
    symb: crate::numeric::multifrontal_ldlt::MultifrontalSymbolic,
    n: usize,
    nnz: usize,
    matching: Option<LuMatching>,
    /// Wall time of the analysis and the ordering it was asked for, carried
    /// into the diagnostics of every factorization reusing it.
    analyze_ms: f64,
    requested_ordering: crate::symbolic::OrderingMethod,
    /// [`estimate_memory`](Self::estimate_memory) results, keyed by scalar
    /// size (the estimate depends on `T` only through `size_of::<T>()`, and
    /// rebuilding the supernode row structures per call is expensive).
    est_cache: Mutex<Vec<(usize, crate::diagnostics::MemoryEstimate)>>,
    /// Lazily built scatter programs for `P^T A P` and its transpose: the
    /// permuted structures are fixed per pattern, so every (re)factorization
    /// reduces to one linear values scatter (equilibration applied on the way).
    perm_scatter: std::sync::OnceLock<(PermScatter, PermScatter)>,
}

impl LuSymbolic {
    /// PARDISO **phase 1**: analyze the symmetrized pattern `A union A^T` of `a`
    /// (values ignored, so any matrix with the target pattern works). Reuse the
    /// result across many [`factor`](Self::factor) calls that share the pattern
    /// - the unsymmetric twin of [`LdltSymbolic::analyze`].
    ///
    /// [`LdltSymbolic::analyze`]: crate::numeric::sparse_solver::LdltSymbolic::analyze
    pub fn analyze<T: Scalar>(a: &GeneralCsc<T>) -> Result<LuSymbolic, RslabError> {
        Self::analyze_with(a, &SolverSettings::default())
    }

    /// [`analyze`](Self::analyze) with explicit composable [`SolverSettings`]
    /// (child-reordering strategy).
    pub fn analyze_with<T: Scalar>(
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
    ) -> Result<LuSymbolic, RslabError> {
        a.validate()?;
        let n = a.n;
        let nnz = a.row_idx.len();
        if n == 0 {
            return Ok(LuSymbolic {
                symb: analyze_with(0, &[0], &[], opts)?,
                n: 0,
                nnz: 0,
                matching: None,
                analyze_ms: 0.0,
                requested_ordering: opts.ordering,
                est_cache: Mutex::new(Vec::new()),
                perm_scatter: std::sync::OnceLock::new(),
            });
        }
        let t = crate::clock::Instant::now();
        // MC64 row matching: analyze the row-permuted matrix `B` whose
        // diagonal carries the matched entries.
        let matching = if opts.lu_matching {
            let cache = crate::scaling::mc64::compute_matching_general(a)?;
            if cache.n_matched == n {
                let (r, c) = crate::scaling::mc64::unsymmetric_scaling(&cache);
                // `cache.perm[j]` is the row matched to column `j`: it becomes
                // row `j` of `B`.
                Some(LuMatching {
                    row_of: cache.perm,
                    r,
                    c,
                })
            } else {
                crate::logging::warn(&format!(
                    "lu analyze: structurally rank-deficient ({} of {n} columns matched); row matching skipped",
                    cache.n_matched
                ));
                None
            }
        } else {
            None
        };
        let (col_ptr, row_idx) = match &matching {
            Some(m) => symmetrized_lower_pattern(&Self::row_permuted(a, m)),
            None => symmetrized_lower_pattern(a),
        };
        let symb = analyze_with(n, &col_ptr, &row_idx, opts)?;
        let analyze_ms = t.elapsed().as_secs_f64() * 1e3;
        if crate::logging::enabled(crate::logging::LogLevel::Info) {
            let d = symb.decisions(opts.ordering);
            crate::logging::info(&format!(
                "lu analyze: n={n} nnz(A)={nnz} ordering={}{} supernodes={} max_front={} \
                 levels={} {analyze_ms:.1} ms",
                d.ordering_used,
                if d.ordering_used != d.ordering_requested {
                    format!(" (requested {})", d.ordering_requested)
                } else {
                    String::new()
                },
                d.n_supernodes,
                d.max_front,
                d.tree_levels
            ));
        }
        Ok(LuSymbolic {
            symb,
            n,
            nnz,
            matching,
            analyze_ms,
            requested_ordering: opts.ordering,
            est_cache: Mutex::new(Vec::new()),
            perm_scatter: std::sync::OnceLock::new(),
        })
    }

    /// `A` with its rows permuted by the matching (values untouched; the
    /// scaling is applied in the numeric phase).
    fn row_permuted<T: Scalar>(a: &GeneralCsc<T>, m: &LuMatching) -> GeneralCsc<T> {
        let n = a.n;
        let mut b_row_of_a = vec![0usize; n];
        for (b, &ar) in m.row_of.iter().enumerate() {
            b_row_of_a[ar] = b;
        }
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::with_capacity(a.row_idx.len());
        let mut values = Vec::with_capacity(a.values.len());
        col_ptr.push(0);
        let mut col: Vec<(usize, T)> = Vec::new();
        for j in 0..n {
            col.clear();
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                col.push((b_row_of_a[a.row_idx[k]], a.values[k]));
            }
            col.sort_unstable_by_key(|e| e.0);
            for &(r, v) in &col {
                row_idx.push(r);
                values.push(v);
            }
            col_ptr.push(row_idx.len());
        }
        GeneralCsc {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// Whether the analysis carries an MC64 row matching.
    pub fn has_matching(&self) -> bool {
        self.matching.is_some()
    }

    /// PARDISO **phases 2-3**: equilibrate and LU-factor `a`, reusing this
    /// analysis, into a ready-to-solve [`LuSolver`]. `a` must share the analyzed
    /// pattern. The unsymmetric twin of [`LdltSymbolic::factor`].
    ///
    /// [`LdltSymbolic::factor`]: crate::numeric::sparse_solver::LdltSymbolic::factor
    pub fn factor<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        opts: &SolverSettings,
    ) -> Result<LuSolver<T>, RslabError> {
        let estimate = self.estimate_memory::<T>();
        let resolved_threads = opts.threads.resolve(|cap| {
            crate::numeric::multifrontal_ldlt::recommend_threads_for_sym(&self.symb, cap)
        });
        let warnings = opts.ignored_on(crate::numeric::multifrontal_ldlt::FactorPath::Lu);
        for w in &warnings {
            crate::logging::warn(&format!("lu settings: {w}"));
        }
        let t = crate::clock::Instant::now();
        let numeric = factor_general_lu_numeric(self, a, opts)?;
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        let nnz = numeric.factor_nnz() as u64;
        let factor_bytes = numeric.bytes() as u64;
        let (l, ut, factors) = numeric.into_parts();
        let mut decisions = self.symb.decisions(self.requested_ordering);
        decisions.scaling = if self.matching.is_some() {
            "Mc64RowMatching".to_string()
        } else {
            "TwoSidedRowCol".to_string()
        };
        decisions.method = format!("{:?}", opts.method);
        let mut diagnostics = crate::diagnostics::Diagnostics {
            threads: resolved_threads,
            n: self.n,
            nnz_a: self.nnz as u64,
            factor_nnz: nnz,
            estimate: Some(estimate),
            decisions,
            numeric: crate::diagnostics::NumericReport {
                perturbed: factors.n_perturbed,
                two_by_two: None,
                inertia: None,
            },
            warnings,
            ..Default::default()
        };
        // Bytes per stored entry: the scalar value plus its usize index.
        diagnostics.push("analyze", self.analyze_ms, 0, 0);
        diagnostics.push("factor", factor_ms, estimate.factor_flops, factor_bytes);
        if crate::logging::enabled(crate::logging::LogLevel::Info) {
            crate::logging::info(&format!("lu factor: {}", diagnostics.summary()));
        }
        // Solve layout: supernodal panels of `L` and `U^T` plus the tree
        // schedule; the CSC arrays are released so the factor is held once.
        let t = crate::clock::Instant::now();
        let plan_l = crate::numeric::supernodal_solve::SolvePlan::from_panels(
            l,
            &factors.supernode_parent,
            true,
        );
        let plan_u = crate::numeric::supernodal_solve::SolvePlan::from_panels(
            ut,
            &factors.supernode_parent,
            false,
        );
        diagnostics.push(
            "solve-layout",
            t.elapsed().as_secs_f64() * 1e3,
            0,
            (plan_l.bytes() + plan_u.bytes()) as u64,
        );
        Ok(LuSolver {
            factors,
            plan_l,
            plan_u,
            nnz: nnz as usize,
            diagnostics,
            solves: Default::default(),
        })
    }

    /// The analyzed dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Per-supernode frontal-matrix dimensions `(ncol, nrow)` of the symmetrized
    /// pattern - for factorization-cost diagnostics (front-size distribution and
    /// a factor-flop estimate). See [`MultifrontalSymbolic::front_dims`](crate::MultifrontalSymbolic::front_dims).
    pub fn front_dims(&self) -> Vec<(usize, usize)> {
        self.symb.front_dims()
    }

    /// Number of assembly-tree levels (level-parallel factorization depth).
    pub fn n_levels(&self) -> usize {
        self.symb.n_levels()
    }

    /// Supernode count per assembly-tree level (available tree-parallelism by
    /// depth). See [`MultifrontalSymbolic::level_widths`](crate::MultifrontalSymbolic::level_widths).
    pub fn level_widths(&self) -> Vec<usize> {
        self.symb.level_widths()
    }

    /// **A-priori** peak-memory estimate for factoring a matrix of scalar type `T`
    /// with this analysis - computed purely from the symbolic structure, *before*
    /// any numeric work, so a scheduler can fail-fast or pick an approximation when
    /// the estimate exceeds the memory budget. Deterministic and reproducible.
    /// Exact symbolic factor fill (the compact L+U value count summed over
    /// supernodes), the reliable memory-backstop metric. Unlike
    /// [`MemoryEstimate::factor_nnz`](crate::diagnostics::MemoryEstimate::factor_nnz),
    /// a dense-panel upper bound that overshoots the real fill ~6-7x
    /// non-uniformly across orderings, this tracks the actually-stored fill.
    pub fn symbolic_factor_nnz(&self) -> usize {
        let Some((sym, _)) = self.symb.sym_and_levels() else {
            return 0;
        };
        let Some(sched) = self.symb.ll_schedule() else {
            return 0;
        };
        (0..sym.supernodes.len())
            .map(|s| {
                let nc = sym.supernodes[s].ncol;
                let cnrow = sched.rows(s).len().saturating_sub(nc);
                // L: diagonal lower-triangle + off-diagonal rows; U: upper-tri + U12.
                let l = nc * (nc + 1) / 2 + cnrow * nc;
                let u = nc * (nc + 1) / 2 + nc * cnrow;
                l + u
            })
            .sum()
    }

    pub fn estimate_memory<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        let value_bytes = std::mem::size_of::<T>();
        // Cache per scalar size: the estimate is a pure function of the
        // structure and `size_of::<T>()`, and `tuned` + phased `factor` ask
        // for it repeatedly.
        if let Ok(cache) = self.est_cache.lock() {
            if let Some(&(_, est)) = cache.iter().find(|&&(vb, _)| vb == value_bytes) {
                return est;
            }
        }
        let est = self.estimate_memory_for(value_bytes);
        if let Ok(mut cache) = self.est_cache.lock() {
            if !cache.iter().any(|&(vb, _)| vb == value_bytes) {
                cache.push((value_bytes, est));
            }
        }
        est
    }

    /// The uncached estimate body, a pure function of the symbolic structure
    /// and the scalar size.
    fn estimate_memory_for(&self, value_bytes: usize) -> crate::diagnostics::MemoryEstimate {
        let Some((sym, _levels)) = self.symb.sym_and_levels() else {
            return crate::diagnostics::estimate_left_looking(
                0,
                &|_| 0,
                &|_| 0,
                &|_| &[],
                value_bytes,
                0,
                false,
            );
        };
        let nsuper = sym.supernodes.len();
        let Some(sched) = self.symb.ll_schedule() else {
            return crate::diagnostics::estimate_left_looking(
                0,
                &|_| 0,
                &|_| 0,
                &|_| &[],
                value_bytes,
                0,
                false,
            );
        };
        // The `L` and `U^T` panels of a supernode, `(w + m) x w` each; they are
        // the stored factor (no compact copy, see `PanelFactor`).
        let panel_bytes = |s: usize| -> u64 {
            let nc = sym.supernodes[s].ncol;
            let nr = sched.rows(s).len();
            (2 * nr * nc * value_bytes) as u64
        };
        let compact_bytes = panel_bytes;
        // Persistent input copies: the equilibrated permuted `a_perm` and its
        // transpose `a_perm_t` (both held through the left-looking factor).
        let input_bytes = (2 * self.nnz * (value_bytes + 8)) as u64;
        let mut est = crate::diagnostics::estimate_left_looking(
            nsuper,
            &panel_bytes,
            &compact_bytes,
            &|s| sched.updaters(s),
            value_bytes,
            input_bytes,
            true,
        );
        est.factor_flops = (0..nsuper)
            .map(|s| {
                let (nc, nr) = (sym.supernodes[s].ncol as u64, sched.rows(s).len() as u64);
                nr * nr * nc
            })
            .sum();
        est
    }
}

/// A factored unsymmetric LU solver, ready to solve against many right-hand
/// sides - the high-level, equilibrated counterpart of the raw [`LuFactors`]
/// (and the unsymmetric twin of [`LdltSolver`](crate::numeric::sparse_solver::LdltSolver)).
/// Build via [`LuSymbolic::factor`] (analyze once, factor many) or the one-shot
/// [`LuSolver::factor`].
pub struct LuSolver<T> {
    factors: LuFactors<T>,
    /// `L` and `U^T` (the panels, their only storage) with the tree schedule
    /// of [`crate::numeric::supernodal_solve`]; `factors` carries the
    /// permutations, scalings and counters with empty CSC arrays.
    plan_l: crate::numeric::supernodal_solve::SolvePlan<T>,
    plan_u: crate::numeric::supernodal_solve::SolvePlan<T>,
    nnz: usize,
    diagnostics: crate::diagnostics::Diagnostics,
    /// Solve-phase accumulators (every `solve*` call records into them).
    solves: crate::diagnostics::SolveCounter,
}

impl<T: Scalar> LuSolver<T> {
    /// Thread policy the solve phase should honour (issue #9): the resolved
    /// [`Threads`](crate::Threads) budget the factorization used, carried on the
    /// stored [`LuFactors`]. An iterative solve using this factor as a
    /// preconditioner runs its parallel orthogonalization in a pool of this width.
    pub fn solve_thread_policy(&self) -> crate::numeric::multifrontal_ldlt::Threads {
        self.factors.solve_threads
    }

    /// One-shot analyze + equilibrate + factor of a general matrix `A`.
    pub fn factor(a: &GeneralCsc<T>, opts: &SolverSettings) -> Result<Self, RslabError> {
        // Through the symbolic object, so the diagnostics are filled the same
        // way as on the analyze-once path (the former direct call returned
        // an empty `Diagnostics`).
        LuSymbolic::analyze_with(a, opts)?.factor(a, opts)
    }

    /// The **heuristic** settings pick for `a` - the model-free default, the
    /// unsymmetric counterpart of [`LdltSolver::tuned`](crate::LdltSolver::tuned):
    /// analysis with the adaptive ordering heuristic, the proven default kernel
    /// configuration, and (on large systems) the exact nested-dissection bakeoff.
    pub fn tuned(a: &GeneralCsc<T>) -> Result<(LuSymbolic, SolverSettings), RslabError> {
        Self::tuned_with(a, &SolverSettings::default())
    }

    /// [`tuned`](Self::tuned) on top of the caller's settings: the analysis
    /// knobs (`nemin`, `relax`, `reorder`, `lu_matching`, ...) come from
    /// `base`, the ordering is the heuristic race, the thread count the
    /// calibrated pick.
    pub fn tuned_with(
        a: &GeneralCsc<T>,
        base: &SolverSettings,
    ) -> Result<(LuSymbolic, SolverSettings), RslabError> {
        crate::numeric::ll_common::tuned(a, base, LuSymbolic::analyze_with, |sym: &LuSymbolic| {
            sym.estimate_memory::<T>()
        })
    }

    /// Per-call diagnostics: measured factor time, fill, thread budget, and the
    /// a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate). Populated by
    /// the phased [`LuSymbolic::factor`]; empty for the one-shot
    /// [`factor`](Self::factor).
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
                "lu solve: n={} rhs={rhs} refine_steps={refine_steps} {ms:.3} ms",
                self.factors.n
            ));
        }
    }

    /// Solve `A x = b` using the stored factors.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        let t = crate::clock::Instant::now();
        let x = self.solve_inner(b, 1)?;
        self.record_solve(1, t, 0);
        Ok(x)
    }

    /// `x = A^{-1} b` on `nrhs` row-major right-hand sides: row scaling and
    /// permutation, the supernodal `L` and `U` sweeps, column permutation
    /// and scaling.
    fn solve_inner(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
        let f = &self.factors;
        let n = f.n;
        if nrhs == 0 || b.len() != n * nrhs {
            return Err(RslabError::DimensionMismatch {
                expected: n * nrhs,
                got: b.len(),
            });
        }
        let mut y = vec![T::zero(); n * nrhs];
        for e in 0..n {
            let orig = f.perm_row[e];
            let sr = T::from_real(f.d_row[orig]);
            let src = &b[orig * nrhs..(orig + 1) * nrhs];
            let dst = &mut y[e * nrhs..(e + 1) * nrhs];
            for c in 0..nrhs {
                dst[c] = src[c] * sr;
            }
        }
        self.plan_l.forward(nrhs, &mut y);
        self.plan_u.backward(nrhs, &mut y);
        let mut out = vec![T::zero(); n * nrhs];
        for e in 0..n {
            let orig = f.perm[e];
            let sc = T::from_real(f.d_col[orig]);
            let src = &y[e * nrhs..(e + 1) * nrhs];
            let dst = &mut out[orig * nrhs..(orig + 1) * nrhs];
            for c in 0..nrhs {
                dst[c] = src[c] * sc;
            }
        }
        Ok(out)
    }

    /// Solve `A * X = B` for `nrhs` right-hand sides at once. `b` and the
    /// returned `x` are **row-major** `n x nrhs` buffers (`b[i*nrhs + c]` is RHS
    /// `c` at row `i`). Faster than `nrhs` separate [`solve`](Self::solve) calls.
    pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
        let t = crate::clock::Instant::now();
        let x = self.solve_inner(b, nrhs)?;
        self.record_solve(nrhs, t, 0);
        Ok(x)
    }

    /// Solve `A x = b` with iterative refinement against the original matrix `a`
    /// (which must be the matrix this was factored from) - recovers accuracy on
    /// hard systems where the static-pivoted factor alone is insufficient.
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
        let t = crate::clock::Instant::now();
        let mut x = self.solve_inner(b, 1)?;
        let outcome = self.refine_into(a, b, &mut x, policy)?;
        self.record_solve(1, t, outcome.steps);
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
        let n = self.factors.n;
        if a.n != n || b.len() != n || x.len() != n {
            return Err(RslabError::DimensionMismatch {
                expected: n,
                got: a.n,
            });
        }
        crate::refine::refine_in_place(a, b, x, policy, |r| self.solve_inner(r, 1))
    }

    /// Stored fill `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.nnz
    }

    /// Number of statically perturbed pivots (preconditioner mode).
    pub fn n_perturbed(&self) -> usize {
        self.factors.n_perturbed
    }

    /// The matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }

    /// Borrow the underlying raw factors (CSC `L` / CSR `U`, permutations,
    /// equilibration), e.g. to use as a [`Preconditioner`](crate::Preconditioner).
    /// The factor's permutations, scalings and counters. The CSC arrays of
    /// `L` and `U` are empty here: the values live in the panels of the solve
    /// plans (use [`factor_general_lu`] for a factor with CSC arrays).
    pub fn factors(&self) -> &LuFactors<T> {
        &self.factors
    }
}

/// Factor a general (unsymmetric) sparse matrix `A` as `P^T A P = L U` via
/// generic multifrontal LU with partial pivoting. `a` holds the **full** matrix
/// (both triangles). Convenience wrapper over [`LuSymbolic::analyze`] +
/// [`factor_general_lu_numeric`]; for *analyze once, factor many* keep the
/// [`LuSymbolic`] across calls. Solve with [`solve_lu`] / [`solve_lu_refined`].
pub fn factor_general_lu<T: Scalar>(
    a: &GeneralCsc<T>,
    opts: &SolverSettings,
) -> Result<LuFactors<T>, RslabError> {
    // The analysis honours the caller's symbolic settings (ordering, child reordering):
    // `analyze` alone took the defaults and silently ignored `opts.ordering`.
    factor_general_lu_numeric(&LuSymbolic::analyze_with(a, opts)?, a, opts)
        .map(LuNumeric::into_factors)
}

// Supernodal left-looking LU (FactorMethod::LeftLooking)
//
// The unsymmetric twin of the left-looking LDL^T path: each supernode keeps two
// dense panels - `lbuf` (its columns: diagonal block + L21, full height) and
// `ubuf` (its rows' U12: the trailing-column part) - assembled from `A` and
// updated by every factored descendant. The contribution of descendant `k` is
// the rank-`ncol_k` outer product `-L_k[Ok,:]*U_k[:,Ok]`; the part landing in
// `s` splits into two GEMMs: `-L_k[Ok,:]*U_k[:,Pk]` into `lbuf` (columns of `s`)
// and `-L_k[Pk,:]*U_k[:,trailing]` into `ubuf` (U12 rows of `s`). Then the panel
// is factored in place (`cdiv`) with **no trailing/CB update** - there is no
// contribution-block stack and no per-front extract copy-out, the PARDISO
// transient profile. 1x1 static pivoting (no row interchange), as in the
// multifrontal v1; matches the equilibrated preconditioner use case.
// ===========================================================================

/// One factored supernode's left-looking payload: the dense L panel, the U12
/// rows, and the within-front row permutation from partial pivoting
/// (`rperm[i]` is the row-structure index physically at panel position `i`;
/// identity on the trailing rows, only read by the emit).
struct LuSlot<T> {
    u: Vec<T>,
    rperm: Vec<usize>,
}
impl<T> Default for LuSlot<T> {
    fn default() -> Self {
        LuSlot {
            u: Vec::new(),
            rperm: Vec::new(),
        }
    }
}
type LuLlStore<T> = crate::numeric::ll_common::SlotStore<LuSlot<T>>;

use crate::numeric::ll_common::{
    emit_refcount_offsets, Cells, Li, LlSchedule, PanelPtr, PermScatter,
};

/// Apply a factored NB-wide panel transform (column scale by `pinv`, within-panel
/// rank-1 against the stored `U11`) to rows `[r0, r1)` of a column-major buffer
/// based at `base` with column stride `nrow`. Bit-identical to the corresponding
/// rows of a full-height `getf2`. Used for the deep trailing rows, which are never
/// pivot candidates, so each caller's row range is independent.
///
/// SAFETY: `[r0, r1)` must be this caller's exclusive rows and within the buffer;
/// columns `[kb, kb+pw)` must be in bounds under stride `nrow`.
#[inline]
unsafe fn apply_panel_trailing<T: Scalar>(
    base: *mut T,
    nrow: usize,
    kb: usize,
    pw: usize,
    pinv_blk: &[T],
    r0: usize,
    r1: usize,
) {
    // `kk` indexes pinv_blk and drives the column arithmetic (`k`, `j`) and inner
    // range - not a plain slice walk.
    #[allow(clippy::needless_range_loop)]
    for kk in 0..pw {
        let k = kb + kk;
        let pinv_k = pinv_blk[kk];
        let colk = base.add(k * nrow);
        for i in r0..r1 {
            *colk.add(i) = *colk.add(i) * pinv_k;
        }
        for jj in (kk + 1)..pw {
            let j = kb + jj;
            let ukj = *base.add(j * nrow + k);
            if ukj != T::zero() {
                let colj = base.add(j * nrow);
                for i in r0..r1 {
                    *colj.add(i) = *colj.add(i) - *colk.add(i) * ukj;
                }
            }
        }
    }
}

struct LlEmit<T> {
    /// Number of consumers (ancestors that pull) still to come; freed at 0.
    refcount: Vec<AtomicUsize>,
    /// First elimination position of each supernode (symbolic prefix sum of ncol).
    e_offset: Vec<usize>,
    /// The factors' buffers: every supernode factors `L` straight into its
    /// slot of `l_arena`; `U^T` is gathered into `u_arena` at the emit.
    l_arena: PanelArena<T>,
    u_arena: PanelArena<T>,
    panels: Cells<(PanelOut, PanelOut)>,
    /// `e_of_g[g]` = elimination position of COLUMN g; `row_pos_of_g[g]` =
    /// position whose PIVOT ROW is g. Written in-node (disjoint g), read after the
    /// join barrier (and in `emit_and_free`, where the join chain makes consumer
    /// writes visible).
    e_of_g: Cells<usize>,
    row_pos_of_g: Cells<usize>,
    perm: Cells<usize>,
    perm_row: Cells<usize>,
}

impl<T: Scalar> LlEmit<T> {
    fn new(sym: &SymbolicFactorization, sched: &LlSchedule) -> Self {
        let n = sym.n;
        let (refcount, e_offset) = emit_refcount_offsets(sym, sched);
        let sizes =
            || (0..sym.supernodes.len()).map(|s| sched.rows(s).len() * sym.supernodes[s].ncol);
        LlEmit {
            refcount,
            e_offset,
            l_arena: PanelArena::new(sizes()),
            u_arena: PanelArena::new(sizes()),
            panels: Cells::new_default(sym.supernodes.len()),
            e_of_g: Cells::new(n, usize::MAX),
            row_pos_of_g: Cells::new(n, usize::MAX),
            perm: Cells::new(n, 0),
            perm_row: Cells::new(n, 0),
        }
    }
    #[inline]
    unsafe fn eg(&self, g: usize) -> usize {
        *self.e_of_g.get(g)
    }
    #[inline]
    unsafe fn rg(&self, g: usize) -> usize {
        *self.row_pos_of_g.get(g)
    }
}

/// Emit supernode `k` once its last updater is done: the left-looking panel
/// `lbuf` (`L` strictly below the diagonal, `U`'s diagonal block on and above
/// it) becomes the `L` panel as it is, and `U^T`'s panel is gathered from the
/// upper triangle and the off-block `ubuf` (`U`'s `ncol x cnrow` block,
/// column-major). Both get their off-block rows in elimination order.
fn emit_and_free<T: Scalar>(
    k: usize,
    store: &LuLlStore<T>,
    emit: &LlEmit<T>,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    drop_tol: Option<f64>,
) {
    let snode = &sym.supernodes[k];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let nrow = sched.rows(k).len();
    let cnrow = nrow - ncol;
    // SAFETY: the owner of supernode `k` emits it exactly once, after its last
    // updater has read the slots (refcount zero); nobody reads them afterwards.
    let slot = unsafe { store.take(k) };
    let (ubuf, rperm) = (&slot.u, &slot.rperm);
    debug_assert_eq!(ubuf.len(), ncol * cnrow);
    let eoff = emit.e_offset[k];
    debug_assert!(
        (0..ncol).all(|p| unsafe { emit.eg(first + p) } == eoff + p)
            && (0..ncol).all(|i| unsafe { emit.rg(sched.rows(k)[rperm[i]] as usize) } == eoff + i),
        "the diagonal block is in elimination order"
    );
    // `U^T`'s panel from the upper triangle of the `L` slot and `ubuf`
    // (`U`'s `ncol x cnrow` block, column-major), then `L`'s slot in place.
    {
        let lbuf: &[T] = unsafe { emit.l_arena.slot(k) };
        let ut = unsafe { emit.u_arena.slot_mut(k) };
        debug_assert_eq!(ut.len(), nrow * ncol);
        for p in 0..ncol {
            let col = &mut ut[p * nrow..(p + 1) * nrow];
            for (i, v) in col.iter_mut().enumerate().take(ncol).skip(p) {
                *v = lbuf[i * nrow + p];
            }
            for (t, v) in col[ncol..].iter_mut().enumerate() {
                *v = ubuf[p + t * ncol];
            }
        }
    }
    let l_rows: Vec<u32> = (ncol..nrow)
        .map(|i| unsafe { emit.rg(sched.rows(k)[rperm[i]] as usize) } as u32)
        .collect();
    let u_rows: Vec<u32> = (0..cnrow)
        .map(|t| unsafe { emit.eg(sched.rows(k)[ncol + t] as usize) } as u32)
        .collect();
    let l_out = finish_panel(
        unsafe { emit.l_arena.slot_mut(k) },
        ncol,
        l_rows,
        None,
        drop_tol,
    );
    let u_out = finish_panel(
        unsafe { emit.u_arena.slot_mut(k) },
        ncol,
        u_rows,
        None,
        drop_tol,
    );
    unsafe { emit.panels.set(k, (l_out, u_out)) };
}

#[allow(clippy::too_many_arguments)]
fn lu_ll_factor_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &GeneralCsc<T>,
    a_perm_t: &GeneralCsc<T>,
    sched: &LlSchedule,
    store: &LuLlStore<T>,
    emit: &LlEmit<T>,
    perturb_floor: Option<f64>,
    n_perturbed: &AtomicUsize,
    ll_active: &AtomicUsize,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    kt.interrupted()?;
    ll_active.fetch_add(1, Ordering::Relaxed);
    let _active = LuActiveGuard(ll_active);
    let ll_gemm_gate = kt.scalar_gate;
    let ll_gemm_par = kt.par_gemm;
    let snode = &sym.supernodes[s];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let nrow = sched.rows(s).len();
    let cnrow = nrow - ncol;
    let n = sym.n;
    // `lbuf`: nrowxncol (columns of s, full height). `ubuf`: ncolxcnrow (U12).
    // SAFETY: this task owns supernode `s`; nobody reads the slot before it
    // is published by `store.set` at the end of the node.
    let lbuf: &mut [T] = unsafe { emit.l_arena.slot_mut(s) };
    debug_assert_eq!(lbuf.len(), nrow * ncol);
    let mut ubuf = vec![T::zero(); ncol * cnrow];

    let mut gloc = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if gloc.len() < n {
        gloc.resize(n, Li::MAX);
    }
    for (li, &g) in sched.rows(s).iter().enumerate() {
        gloc[g as usize] = li as Li;
    }
    // Assemble columns of s (full) into lbuf, and the U12 rows into ubuf.
    for p in 0..ncol {
        let c = first + p;
        for k in a_perm.col_ptr[c]..a_perm.col_ptr[c + 1] {
            let li = gloc[a_perm.row_idx[k]];
            if li != Li::MAX {
                let li = li as usize;
                lbuf[p * nrow + li] = lbuf[p * nrow + li] + a_perm.values[k];
            }
        }
        for k in a_perm_t.col_ptr[c]..a_perm_t.col_ptr[c + 1] {
            let lc = gloc[a_perm_t.row_idx[k]];
            if lc != Li::MAX && lc as usize >= ncol {
                let lc = lc as usize;
                ubuf[p + (lc - ncol) * ncol] = ubuf[p + (lc - ncol) * ncol] + a_perm_t.values[k];
            }
        }
    }
    // cmod from every factored descendant. NOTE: cmod-aggregation (K-stacking many
    // descendant updates into one fat GEMM) was measured and rejected - across MoM
    // topologies 91-95 % of cmod flop already runs as large parallel GEMMs, and the
    // only aggregation reaching those dominant updates carries an 11-15x zero-pad
    // blowup (each top-of-tree descendant touches a small, distinct row/col subset
    // of the large target). The `RLA_CMOD_DIST` histogram below documents this.
    // Pre-pass over the updaters: landing ranges + update flops incl. the
    // U-side rows, the fork/tiling dispatch input (see `ll_common::cmod_spans`).
    let (spans, cmod_flops) =
        crate::numeric::ll_common::cmod_spans(sym, sched, s, first, ncol, true);
    // Fork only above real node-local work, or in the chain phase (<= 2 nodes
    // in flight - workers idle, nothing to steal). Below: strictly serial.
    const LU_CMOD_FORK_MIN_FLOPS: usize = 100_000_000;
    let fork_gate = LU_CMOD_FORK_MIN_FLOPS.max(ll_gemm_par);
    let chain_phase = ll_active.load(Ordering::Relaxed) <= 2;
    let forks = cmod_flops >= fork_gate || (chain_phase && cmod_flops >= ll_gemm_par);
    let tile_w = (ncol / 16).clamp(32, 256);
    let tile_u = (cnrow.max(1) / 16).clamp(32, 256);
    // Deterministic mode pick (the LDLT twin's tiled-cmod note applies
    // verbatim): tiled-vs-sequential is NOT bit-identical (scalar-gate
    // kernel dichotomy + shape-dependent GEMM bits), so `tiled` must be a
    // pure function of the node - never of the racy `chain_phase`, which
    // still decides only bit-neutral parallelism (`forks` for the
    // sequential GEMMs, `ll_cdiv_par`). A racy pick broke the bit-identity
    // guarantee; see tests/ll_thread_determinism.rs.
    let tiled = ncol >= 2 * tile_w && cmod_flops >= fork_gate;

    // Column-tiled parallel cmod: disjoint `&mut` slabs of the target
    // buffers; per slab every updater's contribution in updater order with a
    // serial GEMM. One fan-out per node instead of one per update; the slab
    // stays cache-hot across the updaters; slab widths are pure functions of
    // the node (never of the thread count). Every entry lives in exactly one
    // slab and receives its contributions in the same updater order. The LU
    // node has TWO target buffers, so the tiling runs as two phases: `lbuf`
    // slabs (L/U11 updates), then `ubuf` slabs (U12 updates).
    if tiled {
        let gloc_ref = &gloc;
        let spans_ref = &spans;
        lbuf.par_chunks_mut(nrow * tile_w)
            .enumerate()
            .for_each(|(ti, slab)| {
                let c0 = ti * tile_w;
                let c1 = (c0 + tile_w).min(ncol);
                let mut lupd: Vec<T> = Vec::new();
                for &(kk, p0, p1) in spans_ref {
                    let nck = sym.supernodes[kk].ncol;
                    let nrk = sched.rows(kk).len();
                    let ok = &sched.rows(kk)[nck..];
                    let nok = ok.len();
                    let q0 = p0 + ok[p0..p1].partition_point(|&g| (g as usize) < first + c0);
                    let q1 = p0 + ok[p0..p1].partition_point(|&g| (g as usize) < first + c1);
                    let npk = q1 - q0;
                    if npk == 0 {
                        continue;
                    }
                    // SAFETY: `kk` is a factored descendant of `s`.
                    let slot = unsafe { store.get(kk) };
                    let lk: &[T] = unsafe { emit.l_arena.slot(kk) };
                    let uk = &slot.u;
                    let mrows = nok - p0;
                    lupd.clear();
                    lupd.resize(mrows * npk, T::zero());
                    // SAFETY: lhs/rhs/dst pairwise disjoint; strides in bounds.
                    unsafe {
                        gemm::gemm(
                            mrows,
                            npk,
                            nck,
                            lupd.as_mut_ptr(),
                            mrows as isize,
                            1,
                            false,
                            lk.as_ptr().add(nck + p0),
                            nrk as isize,
                            1,
                            uk.as_ptr().add(q0 * nck),
                            nck as isize,
                            1,
                            T::zero(),
                            T::one(),
                            false,
                            false,
                            false,
                            gemm::Parallelism::None,
                        );
                    }
                    for jj in 0..npk {
                        let cbase = (ok[q0 + jj] as usize - first - c0) * nrow;
                        let ucol = &lupd[jj * mrows..jj * mrows + mrows];
                        for i in 0..mrows {
                            let dst = cbase + gloc_ref[ok[p0 + i] as usize] as usize;
                            slab[dst] = slab[dst] - ucol[i];
                        }
                    }
                }
            });
        if cnrow > 0 {
            let rs_s = sched.rows(s);
            ubuf.par_chunks_mut(ncol * tile_u)
                .enumerate()
                .for_each(|(ti, slab)| {
                    let u0 = ti * tile_u;
                    let u1 = (u0 + tile_u).min(cnrow);
                    let g0 = rs_s[ncol + u0];
                    let g1 = if ncol + u1 < rs_s.len() {
                        rs_s[ncol + u1]
                    } else {
                        Li::MAX
                    };
                    let mut uupd: Vec<T> = Vec::new();
                    for &(kk, p0, p1) in spans_ref {
                        let nck = sym.supernodes[kk].ncol;
                        let nrk = sched.rows(kk).len();
                        let ok = &sched.rows(kk)[nck..];
                        let nok = ok.len();
                        let t0 = p1 + ok[p1..nok].partition_point(|&g| g < g0);
                        let t1 = p1 + ok[p1..nok].partition_point(|&g| g < g1);
                        let ntr = t1 - t0;
                        let npk = p1 - p0;
                        if ntr == 0 || npk == 0 {
                            continue;
                        }
                        // SAFETY: `kk` is a factored descendant of `s`.
                        let slot = unsafe { store.get(kk) };
                        let lk: &[T] = unsafe { emit.l_arena.slot(kk) };
                        let uk = &slot.u;
                        uupd.clear();
                        uupd.resize(npk * ntr, T::zero());
                        // SAFETY: lhs/rhs/dst pairwise disjoint; strides in bounds.
                        unsafe {
                            gemm::gemm(
                                npk,
                                ntr,
                                nck,
                                uupd.as_mut_ptr(),
                                npk as isize,
                                1,
                                false,
                                lk.as_ptr().add(nck + p0),
                                nrk as isize,
                                1,
                                uk.as_ptr().add(t0 * nck),
                                nck as isize,
                                1,
                                T::zero(),
                                T::one(),
                                false,
                                false,
                                false,
                                gemm::Parallelism::None,
                            );
                        }
                        for jj in 0..ntr {
                            let ubase =
                                (gloc_ref[ok[t0 + jj] as usize] as usize - ncol - u0) * ncol;
                            let ucol = &uupd[jj * npk..jj * npk + npk];
                            for i in 0..npk {
                                let dst = ubase + (ok[p0 + i] as usize - first);
                                slab[dst] = slab[dst] - ucol[i];
                            }
                        }
                    }
                });
        }
    }

    // Sequential per-update cmod (small nodes / narrow panels).
    let mut lupd: Vec<T> = Vec::new();
    let mut uupd: Vec<T> = Vec::new();
    for &(kk, p0, p1) in spans.iter().filter(|_| !tiled) {
        let nck = sym.supernodes[kk].ncol;
        let nrk = sched.rows(kk).len();
        let ok = &sched.rows(kk)[nck..];
        let nok = ok.len();
        // SAFETY: `kk` is a factored descendant of `s`.
        let slot = unsafe { store.get(kk) };
        let lk: &[T] = unsafe { emit.l_arena.slot(kk) };
        let uk = &slot.u;
        let npk = p1 - p0;
        let mrows = nok - p0; // rows used by the L update (Ok subset sched.rows(s) from here)
        let ntrail = nok - p1;
        if mrows * npk * nck < ll_gemm_gate {
            // Scalar path.
            for jj in 0..npk {
                let tcol = ok[p0 + jj] as usize - first;
                for i in 0..mrows {
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc + lk[(nck + p0 + i) + ck * nrk] * uk[ck + (p0 + jj) * nck];
                    }
                    let trow = gloc[ok[p0 + i] as usize] as usize;
                    lbuf[tcol * nrow + trow] = lbuf[tcol * nrow + trow] - acc;
                }
            }
            for jj in 0..ntrail {
                let tu = gloc[ok[p1 + jj] as usize] as usize - ncol;
                for i in 0..npk {
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc + lk[(nck + p0 + i) + ck * nrk] * uk[ck + (p1 + jj) * nck];
                    }
                    let urow = ok[p0 + i] as usize - first;
                    ubuf[urow + tu * ncol] = ubuf[urow + tu * ncol] - acc;
                }
            }
        } else {
            // `forks` folds in the join-steal guard and the global serial
            // switch: a small node never forks here.
            let par = if forks && mrows * npk * nck >= ll_gemm_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            // L update: Lupd(mrowsxnpk) = L_k[Ok>=p0,:] * U_k[:,Pk].
            lupd.clear();
            lupd.resize(mrows * npk, T::zero());
            // SAFETY: lhs (lk off-diag rows), rhs (uk Pk cols), dst (lupd) are
            // disjoint; strides in bounds.
            unsafe {
                gemm::gemm(
                    mrows,
                    npk,
                    nck,
                    lupd.as_mut_ptr(),
                    mrows as isize,
                    1,
                    false,
                    lk.as_ptr().add(nck + p0),
                    nrk as isize,
                    1,
                    uk.as_ptr().add(p0 * nck),
                    nck as isize,
                    1,
                    T::zero(),
                    T::one(),
                    false,
                    false,
                    false,
                    par,
                );
            }
            for jj in 0..npk {
                let cbase = (ok[p0 + jj] as usize - first) * nrow;
                let ucol = &lupd[jj * mrows..jj * mrows + mrows];
                for i in 0..mrows {
                    let dst = cbase + gloc[ok[p0 + i] as usize] as usize;
                    lbuf[dst] = lbuf[dst] - ucol[i];
                }
            }
            // U update: Uupd(npkxntrail) = L_k[Pk,:] * U_k[:,trailing].
            if ntrail > 0 {
                uupd.clear();
                uupd.resize(npk * ntrail, T::zero());
                // SAFETY: as above; rhs is the trailing U columns of `uk`.
                unsafe {
                    gemm::gemm(
                        npk,
                        ntrail,
                        nck,
                        uupd.as_mut_ptr(),
                        npk as isize,
                        1,
                        false,
                        lk.as_ptr().add(nck + p0),
                        nrk as isize,
                        1,
                        uk.as_ptr().add(p1 * nck),
                        nck as isize,
                        1,
                        T::zero(),
                        T::one(),
                        false,
                        false,
                        false,
                        par,
                    );
                }
                for jj in 0..ntrail {
                    let ubase = (gloc[ok[p1 + jj] as usize] as usize - ncol) * ncol;
                    let ucol = &uupd[jj * npk..jj * npk + npk];
                    for i in 0..npk {
                        let dst = ubase + (ok[p0 + i] as usize - first);
                        ubuf[dst] = ubuf[dst] - ucol[i];
                    }
                }
            }
        }
    }
    // cdiv: in-place **blocked** panel LU (1x1 static pivoting), no trailing/CB
    // update. Mirrors the multifrontal `lu_front` getrf - unblocked `getf2` over
    // an NB-wide panel, then the dominant trailing update as a single SIMD GEMM
    // (rank-NB) - but restricted to the panel: the trailing is the remaining
    // panel columns (`lbuf`) plus the `U12` rows (`ubuf`), with no `A22`/CB. This
    // routes the `O(ncol^2*nrow)` cdiv work (the measured 77 % of the left-looking
    // factor) through BLAS-3 instead of scalar rank-1 sweeps.
    // Panel width. Swept 32/48/64/96 on the MoM fronts: 32 optimal for typical
    // panels - but root-class WIDE panels want a fatter deferred-GEMM inner
    // dimension (k = nb), the same lever as the LDLT twin's adaptive nb. Pure
    // function of `ncol`, never of the thread count.
    let nb_cdiv = if ncol >= 512 { 128 } else { 32 };
    // Join-steal guard (see the cmod fork gate above): a small node must not
    // fork inside its cdiv either - unless the chain phase makes it free.
    let cdiv_chain = ll_active.load(Ordering::Relaxed) <= 2;
    let ll_cdiv_par = if nrow * ncol * ncol >= 100_000_000 || cdiv_chain {
        kt.par_cdiv
    } else {
        usize::MAX
    };
    let mut local_perturbed = 0usize;
    // Restricted partial pivoting: row interchanges within the fully-summed block
    // `[0, ncol)` only (the standard sparse-direct choice). `rperm[i]` is the
    // row-structure index physically at position `i`; the trailing rows are never
    // interchanged, so the contribution rows `Ok` ancestors pull are unaffected
    // and `cmod` needs no permutation awareness.
    let mut rperm: Vec<usize> = (0..nrow).collect();
    // Pivot reciprocals of the current panel, reused by the parallel trailing apply.
    let mut pinv_blk: Vec<T> = vec![T::zero(); nb_cdiv];
    let mut kb = 0;
    while kb < ncol {
        kt.interrupted()?;
        let ke = (kb + nb_cdiv).min(ncol);
        // getf2: factor columns [kb, ke) over the **fully-summed rows [k+1, ncol)**
        // only - the deep trailing rows [ncol, nrow) (never pivot candidates) are
        // lifted off this serial path into the parallel apply below.
        for k in kb..ke {
            // **Threshold** partial pivoting (UMFPACK-style): keep the diagonal
            // pivot unless it is below `THRESH` of the largest candidate in the
            // fully-summed block - so a well-scaled/equilibrated matrix never
            // interchanges (no fill or accuracy cost) while small/zero diagonals
            // still get a stable pivot. `THRESH^2` compared on squared magnitudes.
            // `THRESH = kt.pivot_u` (tunable, default 0.1); `u = 1` recovers full
            // partial pivoting, `u = 0` keeps the diagonal unless it is exactly zero.
            let thresh_sq = kt.pivot_u * kt.pivot_u;
            // Static pivoting fast path (`u == 0`): keep the natural pivot order and
            // skip the argmax search entirely - the "skip pivot search" speed lever
            // for fixed-pattern value sequences (solver-in-the-loop: reuse a good
            // order across a frequency sweep / time-stepping). The search result is
            // never consumed when `u == 0` (the threshold test `diag_sq < 0` can
            // never fire), so skipping it is behaviour-identical, only faster. A
            // sub-floor / zero diagonal is still caught below by the pivot policy.
            if thresh_sq > 0.0 {
                let mut p = k;
                let mut best = lbuf[k * nrow + k].magnitude_sq();
                for i in (k + 1)..ncol {
                    let m = lbuf[k * nrow + i].magnitude_sq();
                    if m > best {
                        best = m;
                        p = i;
                    }
                }
                let diag_sq = lbuf[k * nrow + k].magnitude_sq();
                if p != k && diag_sq < thresh_sq * best {
                    for c in 0..ncol {
                        lbuf.swap(c * nrow + k, c * nrow + p);
                    }
                    for t in 0..cnrow {
                        ubuf.swap(k + t * ncol, p + t * ncol);
                    }
                    rperm.swap(k, p);
                }
            }
            let mut piv = lbuf[k * nrow + k];
            match perturb_floor {
                Some(floor) if piv.magnitude() < floor => {
                    piv = perturb_pivot(piv, floor);
                    local_perturbed += 1;
                }
                None if piv == T::zero() => {
                    for &g in sched.rows(s) {
                        gloc[g as usize] = Li::MAX;
                    }
                    GLOC_SCRATCH.with(|c| *c.borrow_mut() = gloc);
                    return Err(RslabError::NumericallyRankDeficient);
                }
                _ => {}
            }
            lbuf[k * nrow + k] = piv;
            let pinv = piv.recip();
            pinv_blk[k - kb] = pinv;
            for i in (k + 1)..ncol {
                lbuf[k * nrow + i] = lbuf[k * nrow + i] * pinv;
            }
            for j in (k + 1)..ke {
                let u_kj = lbuf[j * nrow + k];
                if u_kj != T::zero() {
                    for i in (k + 1)..ncol {
                        lbuf[j * nrow + i] = lbuf[j * nrow + i] - lbuf[k * nrow + i] * u_kj;
                    }
                }
            }
        }
        let pw = ke - kb;
        // Trailing-row L21 panel [ncol, nrow): apply the just-computed panel
        // transform (scale by `pinv_blk`, within-panel rank-1 against `U11`) to the
        // deep rows, parallel over **disjoint** row chunks. Bit-identical to the
        // full-height getf2 - same per-row op sequence - but the dominant `cnrow`
        // work now runs on all idle workers instead of the serial panel path.
        if cnrow > 0 {
            let par = cnrow * pw * pw >= ll_cdiv_par;
            if par {
                let pp = PanelPtr(lbuf.as_mut_ptr());
                let nthreads = rayon::current_num_threads().max(1);
                let cs = (nrow - ncol).div_ceil(nthreads).max(1);
                let ranges: Vec<(usize, usize)> = (0..nthreads)
                    .map(|c| {
                        let r0 = ncol + c * cs;
                        (r0.min(nrow), (r0 + cs).min(nrow))
                    })
                    .filter(|(a, b)| a < b)
                    .collect();
                // Capture the whole `pp` (Send+Sync) - destructure inside so Rust
                // does not disjoint-capture the bare `*mut T`.
                ranges.par_iter().for_each(|&(r0, r1)| {
                    // SAFETY: disjoint row chunk; see `apply_panel_trailing`.
                    unsafe { apply_panel_trailing(pp.get(), nrow, kb, pw, &pinv_blk, r0, r1) };
                });
            } else {
                // SAFETY: single-threaded over all trailing rows.
                unsafe {
                    apply_panel_trailing(lbuf.as_mut_ptr(), nrow, kb, pw, &pinv_blk, ncol, nrow)
                };
            }
        }
        // TRSM: U = L11^-1 * (trailing panel columns of lbuf) and the U12 rows.
        // Each trailing column is an independent forward substitution reading
        // only the finished panel columns [kb, ke), so the block parallelizes
        // over disjoint column chunks - bit-identical per-column op order.
        // Profiled at 22% of cdiv CPU when serial (MoM fronts).
        if (ncol - ke) * pw * pw >= ll_cdiv_par {
            let (head, tail) = lbuf.split_at_mut(ke * nrow);
            tail.par_chunks_mut(nrow).for_each(|col| {
                for r in (kb + 1)..ke {
                    let mut acc = col[r];
                    for i in kb..r {
                        acc = acc - head[i * nrow + r] * col[i];
                    }
                    col[r] = acc;
                }
            });
        } else {
            for j in ke..ncol {
                for r in (kb + 1)..ke {
                    let mut acc = lbuf[j * nrow + r];
                    for i in kb..r {
                        acc = acc - lbuf[i * nrow + r] * lbuf[j * nrow + i];
                    }
                    lbuf[j * nrow + r] = acc;
                }
            }
        }
        // U12 rows (the `cnrow` contribution columns of U): each `t`-column is an
        // independent forward-substitution over the panel rows - parallel over the
        // contiguous `ncol`-strided `ubuf` columns (safe: disjoint chunks).
        let trsm_u = |col: &mut [T], lref: &[T]| {
            for r in (kb + 1)..ke {
                let mut acc = col[r];
                for i in kb..r {
                    acc = acc - lref[i * nrow + r] * col[i];
                }
                col[r] = acc;
            }
        };
        if cnrow * pw * pw >= ll_cdiv_par {
            let lref: &[T] = lbuf;
            ubuf.par_chunks_mut(ncol).for_each(|col| trsm_u(col, lref));
        } else {
            for t in 0..cnrow {
                trsm_u(&mut ubuf[t * ncol..t * ncol + ncol], lbuf);
            }
        }
        // GEMM: lbuf[ke.., ke..ncol] -= L21[ke.., kb..ke] * U[kb..ke, ke..ncol].
        let mt = nrow - ke;
        let nt = ncol - ke;
        if mt > 0 && nt > 0 {
            let par = if (mt * nt * pw) >= ll_cdiv_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            let base = lbuf.as_mut_ptr();
            // SAFETY: the three sub-blocks of `lbuf` are disjoint; strides in bounds.
            unsafe {
                gemm::gemm(
                    mt,
                    nt,
                    pw,
                    base.add(ke * nrow + ke),
                    nrow as isize,
                    1,
                    true,
                    base.add(kb * nrow + ke),
                    nrow as isize,
                    1,
                    base.add(ke * nrow + kb),
                    nrow as isize,
                    1,
                    T::one(),
                    T::zero() - T::one(),
                    false,
                    false,
                    false,
                    par,
                );
            }
        }
        // GEMM: ubuf[ke..ncol, :] -= L[ke..ncol, kb..ke] * U12[kb..ke, :].
        if cnrow > 0 && nt > 0 {
            let par = if (nt * cnrow * pw) >= ll_cdiv_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            let lptr = lbuf.as_ptr();
            let uptr = ubuf.as_mut_ptr();
            // SAFETY: dst (`ubuf` trailing rows) is disjoint from the read
            // sub-blocks of `lbuf`/`ubuf`; strides in bounds.
            unsafe {
                gemm::gemm(
                    nt,
                    cnrow,
                    pw,
                    uptr.add(ke),
                    ncol as isize,
                    1,
                    true,
                    lptr.add(kb * nrow + ke),
                    nrow as isize,
                    1,
                    uptr.add(kb),
                    ncol as isize,
                    1,
                    T::one(),
                    T::zero() - T::one(),
                    false,
                    false,
                    false,
                    par,
                );
            }
        }
        kb = ke;
    }
    if local_perturbed > 0 {
        n_perturbed.fetch_add(local_perturbed, Ordering::Relaxed);
    }
    for &g in sched.rows(s) {
        gloc[g as usize] = Li::MAX;
    }
    GLOC_SCRATCH.with(|c| *c.borrow_mut() = gloc);
    // Populate the O(n) index maps for `s` from its (final) `rperm` and the
    // symbolic elimination offset - consumed by `emit_and_free` and the assembly.
    // Writes target disjoint global indices; visibility via the subtree join.
    let eoff = emit.e_offset[s];
    for (p, &rp) in rperm[..ncol].iter().enumerate() {
        let g_col = first + p;
        let g_row = sched.rows(s)[rp] as usize;
        // SAFETY: each global index is written by exactly one supernode.
        unsafe {
            emit.e_of_g.set(g_col, eoff + p);
            emit.row_pos_of_g.set(g_row, eoff + p);
            emit.perm.set(eoff + p, sym.perm[g_col]);
            emit.perm_row.set(eoff + p, sym.perm[g_row]);
        }
    }
    // SAFETY: this thread owns `s`, writes its cells exactly once.
    unsafe { store.set(s, LuSlot { u: ubuf, rperm }) };
    Ok(())
}

/// Supernodal left-looking LU producing the same [`LuFactors`] as the
/// multifrontal path. `a_perm`/`a_perm_t` are the equilibrated permuted matrix
/// and its transpose; `d_row`/`d_col` the equilibration carried into the result.
#[allow(clippy::too_many_arguments)]
fn factor_lu_left_looking<T: Scalar>(
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    a_perm: &GeneralCsc<T>,
    a_perm_t: &GeneralCsc<T>,
    d_row: &[f64],
    d_col: &[f64],
    perturb_floor: Option<f64>,
    drop_tol: Option<f64>,
    kt: KernelTuning,
) -> Result<LuNumeric<T>, RslabError> {
    let n = sym.n;
    let nsuper = sym.supernodes.len();
    let store = LuLlStore::<T>::new(nsuper);
    let emit = LlEmit::<T>::new(sym, sched);
    let n_perturbed_atomic = AtomicUsize::new(0);
    let roots = crate::numeric::ll_common::forest_roots(sym);
    let ll_active = AtomicUsize::new(0);
    let factor_node = |s: usize| {
        lu_ll_factor_node(
            s,
            sym,
            a_perm,
            a_perm_t,
            sched,
            &store,
            &emit,
            perturb_floor,
            &n_perturbed_atomic,
            &ll_active,
            kt,
        )
    };
    let emit_free = |k: usize| emit_and_free(k, &store, &emit, sym, sched, drop_tol);
    roots
        .par_iter()
        .map(|&r| {
            crate::numeric::ll_common::ll_subtree(
                r,
                sym,
                sched,
                &emit.refcount,
                &factor_node,
                &emit_free,
            )
        })
        .collect::<Result<Vec<()>, _>>()?;
    drop(store); // panels moved into the emit cells; release the shells
    let n_perturbed = n_perturbed_atomic.load(Ordering::Relaxed);
    let kept: Vec<bool> = sym.supernodes.iter().map(|sn| sn.ncol > 0).collect();
    let supernode_parent = crate::symbolic::supernode_parents(&sym.supernodes, &kept);
    let LlEmit {
        l_arena,
        u_arena,
        panels,
        ..
    } = emit;
    let (l, zeros_l) = l_arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).0)
    });
    let (ut, zeros_u) = u_arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).1)
    });
    let perm: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm.get(e) }).collect();
    let perm_row: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm_row.get(e) }).collect();

    Ok(LuNumeric {
        l,
        ut,
        perm,
        perm_row,
        d_row: d_row.to_vec(),
        d_col: d_col.to_vec(),
        supernode_parent,
        n_perturbed,
        n_zeros: zeros_l + zeros_u,
        solve_threads: crate::numeric::multifrontal_ldlt::Threads::Ambient,
    })
}

/// PARDISO phases 2-3 for the general path: numeric LU reusing a [`LuSymbolic`].
/// `a` must share the analyzed pattern (`n`, `nnz`).
#[allow(clippy::needless_range_loop)] // CSC column loops index col_ptr + scaling
pub fn factor_general_lu_numeric<T: Scalar>(
    lusym: &LuSymbolic,
    a: &GeneralCsc<T>,
    opts: &SolverSettings,
) -> Result<LuNumeric<T>, RslabError> {
    a.validate()?;
    let n = lusym.n;
    if a.n != n || a.row_idx.len() != lusym.nnz {
        return Err(RslabError::InvalidInput(
            "factor_general_lu_numeric: matrix does not match the analyzed pattern".to_string(),
        ));
    }
    if n == 0 {
        return Ok(LuNumeric {
            l: PanelFactor::empty(),
            ut: PanelFactor::empty(),
            perm: Vec::new(),
            perm_row: Vec::new(),
            d_row: Vec::new(),
            d_col: Vec::new(),
            supernode_parent: Vec::new(),
            n_perturbed: 0,
            n_zeros: 0,
            solve_threads: crate::numeric::multifrontal_ldlt::Threads::Ambient,
        });
    }

    // Resolve the solve-phase thread policy (issue #9): `Ambient` stays ambient
    // (caller-installed pool); every other policy is pinned to the concrete worker
    // count the factorization itself used, so a preconditioned iterative solve
    // orthogonalizes in a pool of exactly that width.
    let solve_policy = match opts.threads {
        crate::numeric::multifrontal_ldlt::Threads::Ambient => {
            crate::numeric::multifrontal_ldlt::Threads::Ambient
        }
        p => crate::numeric::multifrontal_ldlt::Threads::Fixed(p.resolve(|cap| {
            crate::numeric::multifrontal_ldlt::recommend_threads_for_sym(&lusym.symb, cap)
        })),
    };

    let perturb_floor: Option<f64> = match opts.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = a.values.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    };
    let blr = opts.blr;

    // The assembly-tree levels are no longer needed: the driver is a
    // work-stealing tree recursion, not a level-synchronous sweep.
    let (sym, _by_level) = lusym
        .symb
        .sym_and_levels()
        .ok_or_else(|| RslabError::InvalidInput("internal: empty symbolic".to_string()))?;

    // Two-sided equilibration A_hat = D_r A D_c with d_r[i] = 1/sqrt(max_j |A_ij|),
    // d_c[j] = 1/sqrt(max_i |A_ij|). Tames the dynamic range (these MoM near-field
    // matrices span ~6 orders) so the LU factor - and any incomplete drop -
    // stays well-scaled; the solve undoes it transparently. Computed from the
    // original (unpermuted) A.
    // The matrix the pipeline factors: `A` itself, or its MC64 row-permuted
    // form `B` (row `i` of `B` is row `row_of[i]` of `A`) with the matching's
    // scalings; otherwise the max-norm equilibration.
    let input;
    let a_in: &GeneralCsc<T> = match &lusym.matching {
        Some(m) => {
            input = LuSymbolic::row_permuted(a, m);
            &input
        }
        None => a,
    };
    let (d_row, d_col): (Vec<f64>, Vec<f64>) = match &lusym.matching {
        Some(m) => ((0..n).map(|i| m.r[m.row_of[i]]).collect(), m.c.clone()),
        None => {
            let mut rmax = vec![0.0f64; n];
            let mut cmax = vec![0.0f64; n];
            for j in 0..n {
                for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                    let i = a.row_idx[k];
                    let m = a.values[k].magnitude();
                    if m > rmax[i] {
                        rmax[i] = m;
                    }
                    if m > cmax[j] {
                        cmax[j] = m;
                    }
                }
            }
            (
                rmax.iter()
                    .map(|&r| crate::scaling::inv_sqrt_scale_guarded(r))
                    .collect(),
                cmax.iter()
                    .map(|&c| crate::scaling::inv_sqrt_scale_guarded(c))
                    .collect(),
            )
        }
    };

    let (fwd, bwd) = lusym.perm_scatter.get_or_init(|| {
        (
            PermScatter::build_full(n, &a_in.col_ptr, &a_in.row_idx, &sym.perm_inv),
            PermScatter::build_full_transposed(n, &a_in.col_ptr, &a_in.row_idx, &sym.perm_inv),
        )
    });
    let nnz = a_in.row_idx.len();
    let mut vals = vec![T::zero(); nnz];
    let mut vals_t = vec![T::zero(); nnz];
    for j in 0..n {
        let dc = d_col[j];
        for k in a_in.col_ptr[j]..a_in.col_ptr[j + 1] {
            let sv = a_in.values[k] * T::from_real(d_row[a_in.row_idx[k]] * dc);
            vals[fwd.pos[k]] = sv;
            vals_t[bwd.pos[k]] = sv;
        }
    }
    let a_perm = GeneralCsc {
        n,
        col_ptr: fwd.col_ptr.clone(),
        row_idx: fwd.row_idx.clone(),
        values: vals,
    };
    let a_perm_t = GeneralCsc {
        n,
        col_ptr: bwd.col_ptr.clone(),
        row_idx: bwd.row_idx.clone(),
        values: vals_t,
    };
    // Worker stack sized to the assembly-tree depth (overflow-safe on deep chain
    // trees), shared by both LU paths.
    let stack = crate::numeric::multifrontal_ldlt::stack_for_depth(
        crate::numeric::multifrontal_ldlt::supernode_tree_depth(sym),
    );

    // Supernodal left-looking LU: same factor, low transient (no CB stack). Run in
    // a scoped pool of `opts.threads` so concurrent solves don't oversubscribe.
    if opts.method == FactorMethod::LeftLooking {
        let mut fac = opts.threads.run(
            stack,
            |cap| crate::numeric::multifrontal_ldlt::recommend_threads_for_sym(&lusym.symb, cap),
            || {
                factor_lu_left_looking(
                    sym,
                    lusym.symb.ll_schedule().ok_or_else(|| {
                        RslabError::InvalidInput("internal: empty symbolic".to_string())
                    })?,
                    &a_perm,
                    &a_perm_t,
                    &d_row,
                    &d_col,
                    perturb_floor,
                    opts.drop_tol,
                    opts.kernel(),
                )
            },
        )?;
        fac.solve_threads = solve_policy;
        finish_matching(&mut fac, lusym);
        return Ok(fac);
    }

    let nsuper = sym.supernodes.len();
    let roots = crate::numeric::ll_common::forest_roots(sym);
    // Factor every root subtree with the work-stealing tree schedule; the
    // children-before-parent dependency is the recursion structure itself.
    let pool = FrontPool::<T>::new();
    let kt = opts.kernel();
    // Scoped pool of `opts.threads` with the depth-sized stack (honours the thread
    // budget and is overflow-safe on deep trees, like the left-looking path).
    let factor_one = |s: usize, child_refs: &[&NodeLu<T>]| {
        factor_one_node_lu(
            s,
            sym,
            &a_perm,
            &a_perm_t,
            child_refs,
            perturb_floor,
            blr,
            &pool,
            kt,
        )
    };
    let free_contrib = |nf: &mut NodeLu<T>| nf.contrib = Contribution::Dense(Vec::new());
    let root_outs: Vec<SubtreeFactors<T>> = opts.threads.run(
        stack,
        |cap| crate::numeric::multifrontal_ldlt::recommend_threads_for_sym(&lusym.symb, cap),
        || {
            roots
                .par_iter()
                .map(|&r| crate::numeric::ll_common::mf_subtree(r, sym, &factor_one, &free_contrib))
                .collect::<Result<Vec<_>, _>>()
        },
    )?;
    // Scatter the subtree factors into `node_results` (indexed by supernode id)
    // for the global emit pass, which still walks supernodes in postorder.
    let mut node_results: Vec<Option<NodeLu<T>>> = (0..nsuper).map(|_| None).collect();
    for (i, (own, subtree)) in root_outs.into_iter().enumerate() {
        node_results[roots[i]] = Some(own);
        for (s, nf) in subtree {
            node_results[s] = Some(nf);
        }
    }
    let mut nodes: Vec<&NodeLu<T>> = Vec::with_capacity(nsuper);
    for node_opt in &node_results {
        match node_opt {
            Some(nd) => nodes.push(nd),
            None => {
                return Err(RslabError::InvalidInput(
                    "internal: unfactored supernode".to_string(),
                ))
            }
        }
    }

    // Assign factorization order e (static pivoting -> front-local order is just
    // the column order) and the permutation.
    // Two index maps, distinct under row pivoting:
    //   col_pos_of_g[g] = factorization position eliminating COLUMN g
    //                     (-> U column indices, L column indices).
    //   row_pos_of_g[g] = factorization position whose PIVOT ROW is g
    //                     (-> L row indices). Equal to col_pos when no pivoting.
    let mut col_pos_of_g = vec![usize::MAX; n];
    let mut row_pos_of_g = vec![usize::MAX; n];
    let mut perm = vec![0usize; n];
    let mut perm_row = vec![0usize; n];
    let mut e = 0usize;
    for node in &nodes {
        let ff = &node.front;
        for j in 0..ff.nelim {
            // Columns are not interchanged, so position j maps to column
            // `row_indices[j]`; the pivot row is `row_indices[rperm[j]]`.
            let g_col = node.row_indices[j];
            let g_row = node.row_indices[ff.rperm[j]];
            col_pos_of_g[g_col] = e;
            row_pos_of_g[g_row] = e;
            perm[e] = sym.perm[g_col];
            perm_row[e] = sym.perm[g_row];
            e += 1;
        }
    }
    debug_assert_eq!(e, n, "every index eliminated exactly once");

    // Sum the static-perturbation count before the emit may free the fronts.
    let n_perturbed = nodes.iter().map(|nd| nd.front.n_perturbed).sum();
    // End the `nodes` immutable borrow so the emit can take `node_results`
    // mutably (LowMemory frees each front's dense factor as it is emitted).
    drop(nodes);
    // Every front's eliminated columns become the supernode's panels: the
    // front's `l` block is the `L` panel and its `u` block (column `j` = row
    // `j` of `U` over the front rows) is the `U^T` panel; only the off-block
    // rows need the ancestors' elimination order. Fronts are released as
    // they are emitted (the panels are the factor).
    let kept: Vec<bool> = node_results
        .iter()
        .map(|n| n.as_ref().is_some_and(|nd| nd.front.nelim > 0))
        .collect();
    let supernode_parent = crate::symbolic::supernode_parents(&sym.supernodes, &kept);
    let ncols: Vec<usize> = node_results
        .iter()
        .map(|n| n.as_ref().map_or(0, |nd| nd.front.nelim))
        .collect();
    let sizes = || {
        node_results
            .iter()
            .map(|n| n.as_ref().map_or(0, |nd| nd.front.nrow * nd.front.nelim))
    };
    let l_arena = PanelArena::<T>::new(sizes());
    let u_arena = PanelArena::<T>::new(sizes());
    let mut emit_panels = |s: usize| -> Result<(PanelOut, PanelOut), RslabError> {
        let node = node_results[s].as_mut().ok_or_else(|| {
            RslabError::InvalidInput("internal: unfactored supernode".to_string())
        })?;
        let ff = &mut node.front;
        let (nr, w) = (ff.nrow, ff.nelim);
        let diag_e = col_pos_of_g[node.row_indices[0]];
        debug_assert!(
            (0..w).all(|j| col_pos_of_g[node.row_indices[j]] == diag_e + j)
                && (0..w).all(|i| row_pos_of_g[node.row_indices[ff.rperm[i]]] == diag_e + i),
            "the diagonal block is in elimination order"
        );
        // SAFETY: the sequential emit owns every slot.
        let l = unsafe { l_arena.slot_mut(s) };
        l.copy_from_slice(&ff.l[..nr * w]);
        ff.l = Vec::new();
        let l_rows: Vec<u32> = (w..nr)
            .map(|i| row_pos_of_g[node.row_indices[ff.rperm[i]]] as u32)
            .collect();
        let ut = unsafe { u_arena.slot_mut(s) };
        ut.copy_from_slice(&ff.u[..nr * w]);
        ff.u = Vec::new();
        let u_rows: Vec<u32> = (w..nr)
            .map(|i| col_pos_of_g[node.row_indices[i]] as u32)
            .collect();
        Ok((
            finish_panel(l, w, l_rows, None, opts.drop_tol),
            finish_panel(ut, w, u_rows, None, opts.drop_tol),
        ))
    };
    let mut outs: Vec<(PanelOut, PanelOut)> = Vec::with_capacity(ncols.len());
    for (s, &w) in ncols.iter().enumerate() {
        outs.push(if w > 0 {
            emit_panels(s)?
        } else {
            Default::default()
        });
    }
    let (l, zeros_l) = l_arena.finish(n, ncols.iter().copied(), |s| std::mem::take(&mut outs[s].0));
    let (ut, zeros_u) =
        u_arena.finish(n, ncols.iter().copied(), |s| std::mem::take(&mut outs[s].1));

    let mut fac = LuNumeric {
        l,
        ut,
        perm,
        perm_row,
        d_row,
        d_col,
        supernode_parent,
        n_perturbed,
        n_zeros: zeros_l + zeros_u,
        solve_threads: solve_policy,
    };
    finish_matching(&mut fac, lusym);
    Ok(fac)
}

/// Solve `A x = b` from an unsymmetric LU factorization (`P^T A P = L U`).
#[allow(clippy::needless_range_loop)] // CSC/CSR solves index col_ptr/row_ptr + scaling
pub fn solve_lu<T: Scalar>(f: &LuFactors<T>, b: &[T]) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // y_hat = P_row * (D_r b): row-equilibrate then row-permute the RHS.
    let mut y: Vec<T> = (0..n)
        .map(|e| {
            let orig = f.perm_row[e];
            b[orig] * T::from_real(f.d_row[orig])
        })
        .collect();
    // Forward solve L y = y_hat (CSC, unit diagonal). Column-oriented: once y[e] is
    // final, eliminate it from the rows below. Axpys via `fmadd` (FMA on
    // native builds; see `scalar::fmadd`). The explicit unit diagonal is a
    // column's FIRST entry (rows sorted, lower triangular in elimination
    // numbering), so the sweep starts at `col_ptr[e] + 1` instead of
    // branching on `i != e` at every nonzero.
    for e in 0..n {
        let (s, ee) = (f.l_col_ptr[e], f.l_col_ptr[e + 1]);
        debug_assert_eq!(f.l_row_idx[s], e, "unit diagonal must lead its column");
        let nye = T::zero() - y[e];
        for k in (s + 1)..ee {
            let i = f.l_row_idx[k];
            y[i] = fmadd(f.l_values[k], nye, y[i]);
        }
    }
    // Backward solve U x = y (CSR by row). The pivot is a row's FIRST entry
    // (columns sorted, upper triangular), replacing the per-nonzero
    // diagonal-search branch. Deliberately mul+sub, NOT `fmadd`: the
    // accumulator is a latency-bound serial chain (see `solve_ldlt`'s
    // backward-sweep note).
    let mut x = vec![T::zero(); n];
    for e in (0..n).rev() {
        let (s, ee) = (f.u_row_ptr[e], f.u_row_ptr[e + 1]);
        debug_assert_eq!(f.u_col_idx[s], e, "pivot must lead its row");
        let diag = f.u_values[s];
        let mut acc = y[e];
        for k in (s + 1)..ee {
            acc = acc - f.u_values[k] * x[f.u_col_idx[k]];
        }
        x[e] = acc * diag.recip();
    }
    // Undo the column permutation and apply the column equilibration:
    // x_orig[perm[e]] = D_c[perm[e]] * x_hat[e].
    let mut out = vec![T::zero(); n];
    for e in 0..n {
        let orig = f.perm[e];
        out[orig] = x[e] * T::from_real(f.d_col[orig]);
    }
    Ok(out)
}

/// Solve `A^T * x = b` against the stored factorization of `A` (feral #94
/// enabler). The factor chain is `A^-1 = D_c P_c (LU)^-1 P_r^T D_r` (see
/// [`solve_lu`]), so `A^-T = D_r P_r L^-T U^-T P_c^T D_c`: column-equilibrate
/// and column-permute the RHS, forward-solve `U^T` (lower triangular; `U` is
/// CSR with the pivot leading each row, so once `z[e]` is final it
/// eliminates from the trailing columns of row `e`, scatter form),
/// backward-solve `L^T` (unit upper; dot column `e` of `L` against the
/// already-solved tail), then row-permute and row-equilibrate the result.
///
/// This is the plain TRANSPOSE, not the conjugate transpose - complex
/// callers wanting `A^-H b` conjugate the RHS and the result around this
/// call (`A^-H b = conj(A^-T conj(b))`), as the condition estimator does.
#[allow(clippy::needless_range_loop)] // scatter-form sweeps write w[e] and w[u_col_idx[k]]
pub fn solve_lu_transpose<T: Scalar>(f: &LuFactors<T>, b: &[T]) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // w_hat = P_c^T (D_c b): column-equilibrate then column-permute the RHS.
    let mut w: Vec<T> = (0..n)
        .map(|e| {
            let orig = f.perm[e];
            b[orig] * T::from_real(f.d_col[orig])
        })
        .collect();
    // Forward solve U^T z = w_hat, in place in `w`.
    for e in 0..n {
        let (s, ee) = (f.u_row_ptr[e], f.u_row_ptr[e + 1]);
        debug_assert_eq!(f.u_col_idx[s], e, "pivot must lead its row");
        let ze = w[e] * f.u_values[s].recip();
        w[e] = ze;
        if ze != T::zero() {
            let nze = T::zero() - ze;
            for k in (s + 1)..ee {
                let j = f.u_col_idx[k];
                w[j] = fmadd(f.u_values[k], nze, w[j]);
            }
        }
    }
    // Backward solve L^T v = z, in place in `w` (unit diagonal leads each
    // column, so the dot starts at `col_ptr[e] + 1`).
    for e in (0..n).rev() {
        let (s, ee) = (f.l_col_ptr[e], f.l_col_ptr[e + 1]);
        debug_assert_eq!(f.l_row_idx[s], e, "unit diagonal must lead its column");
        let mut acc = w[e];
        for k in (s + 1)..ee {
            acc = acc - f.l_values[k] * w[f.l_row_idx[k]];
        }
        w[e] = acc;
    }
    // x_orig[perm_row[e]] = D_r[perm_row[e]] * v[e].
    let mut out = vec![T::zero(); n];
    for e in 0..n {
        let orig = f.perm_row[e];
        out[orig] = w[e] * T::from_real(f.d_row[orig]);
    }
    Ok(out)
}

/// Solve `A * X = B` for `nrhs` right-hand sides at once. `b` and the returned
/// `x` are **row-major** `n x nrhs` buffers (`b[i*nrhs + c]` is RHS `c` at row
/// `i`). The `L`/`U` structure is traversed once and each value applied to all
/// `nrhs` columns - faster than `nrhs` separate [`solve_lu`] calls.
/// Below this RHS count / work size the LU block solve runs serially (the
/// parallel gather/scatter overhead only amortizes for wide multi-RHS).
const PAR_SOLVE_MIN_RHS: usize = 8;
const PAR_SOLVE_MIN_WORK: usize = 1 << 18;

/// Solve `A X = B` for `nrhs` right-hand sides. Row-major `n x nrhs` layout, as
/// [`solve_ldlt_many`](crate::solve_ldlt_many). For a wide RHS the columns are
/// split into per-thread chunks (each RHS independent); the result is
/// **bit-identical** to the serial block solve.
pub fn solve_lu_many<T: Scalar>(
    f: &LuFactors<T>,
    b: &[T],
    nrhs: usize,
) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if nrhs == 0 || b.len() != n * nrhs {
        return Err(RslabError::DimensionMismatch {
            expected: n * nrhs,
            got: b.len(),
        });
    }
    let nthreads = rayon::current_num_threads().max(1);
    if nrhs < PAR_SOLVE_MIN_RHS || n * nrhs < PAR_SOLVE_MIN_WORK || nthreads < 2 {
        return solve_lu_block(f, b, nrhs);
    }
    let nchunks = nthreads.min(nrhs);
    let chunk = nrhs.div_ceil(nchunks);
    let ranges: Vec<(usize, usize)> = (0..nchunks)
        .map(|t| (t * chunk, ((t + 1) * chunk).min(nrhs)))
        .filter(|&(a, e)| a < e)
        .collect();
    let parts: Result<Vec<(usize, usize, Vec<T>)>, RslabError> = ranges
        .par_iter()
        .map(|&(c0, c1)| {
            let w = c1 - c0;
            let mut sub = vec![T::zero(); n * w];
            for i in 0..n {
                let ib = i * nrhs;
                let sb = i * w;
                sub[sb..sb + w].copy_from_slice(&b[ib + c0..ib + c1]);
            }
            let xs = solve_lu_block(f, &sub, w)?;
            Ok((c0, c1, xs))
        })
        .collect();
    let parts = parts?;
    let mut x = vec![T::zero(); n * nrhs];
    for (c0, c1, xs) in parts {
        let w = c1 - c0;
        for i in 0..n {
            let ib = i * nrhs;
            let sb = i * w;
            x[ib + c0..ib + c1].copy_from_slice(&xs[sb..sb + w]);
        }
    }
    Ok(x)
}

/// Serial block solve over `nrhs` right-hand sides; fanned over column chunks by
/// the parallel [`solve_lu_many`].
fn solve_lu_block<T: Scalar>(f: &LuFactors<T>, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    // Y_hat = P_row * (D_r B): row-equilibrate then row-permute each RHS block.
    let mut y = vec![T::zero(); n * nrhs];
    for e in 0..n {
        let orig = f.perm_row[e];
        let s = T::from_real(f.d_row[orig]);
        let (eb, ob) = (e * nrhs, orig * nrhs);
        for c in 0..nrhs {
            y[eb + c] = b[ob + c] * s;
        }
    }
    // Reusable single-row scratch. Hoisting the row that is reused across an inner
    // sweep into a **local** buffer breaks the apparent aliasing of `y[..]` with
    // itself, so the `nrhs`-wide AXPY kernels below operate on non-aliasing,
    // contiguous slices the compiler can vectorize (and the hoisted row is loaded
    // once per outer step, not once per nonzero).
    let mut row = vec![T::zero(); nrhs];
    // Forward solve L Y = Y_hat (CSC, unit diagonal). `y[e]` (the column's source row)
    // is read by every nonzero of column `e` and is not written in this sweep.
    for e in 0..n {
        let eb = e * nrhs;
        row.copy_from_slice(&y[eb..eb + nrhs]);
        let (s, ee) = (f.l_col_ptr[e], f.l_col_ptr[e + 1]);
        debug_assert_eq!(f.l_row_idx[s], e);
        for k in (s + 1)..ee {
            let i = f.l_row_idx[k];
            let nlval = T::zero() - f.l_values[k];
            let ib = i * nrhs;
            let tgt = &mut y[ib..ib + nrhs];
            for c in 0..nrhs {
                tgt[c] = fmadd(nlval, row[c], tgt[c]);
            }
        }
    }
    // Backward solve U X = Y (CSR by row), in place in `y`. Accumulate row `e`'s
    // update in the local buffer (the off-diagonal sources `y[c_col]`, `c_col > e`,
    // are already solved and not touched here), then scale and write it back.
    // The pivot leads its row (sorted columns, upper triangular).
    for e in (0..n).rev() {
        let eb = e * nrhs;
        row.copy_from_slice(&y[eb..eb + nrhs]);
        let (s, ee) = (f.u_row_ptr[e], f.u_row_ptr[e + 1]);
        debug_assert_eq!(f.u_col_idx[s], e);
        let diag = f.u_values[s];
        for k in (s + 1)..ee {
            let nuval = T::zero() - f.u_values[k];
            let cb = f.u_col_idx[k] * nrhs;
            let src = &y[cb..cb + nrhs];
            for c in 0..nrhs {
                row[c] = fmadd(nuval, src[c], row[c]);
            }
        }
        let dinv = diag.recip();
        for c in 0..nrhs {
            y[eb + c] = row[c] * dinv;
        }
    }
    // Undo column permutation + column equilibration: out[perm[e]] = D_c * x_hat[e].
    let mut out = vec![T::zero(); n * nrhs];
    for e in 0..n {
        let orig = f.perm[e];
        let s = T::from_real(f.d_col[orig]);
        let (ob, eb) = (orig * nrhs, e * nrhs);
        for c in 0..nrhs {
            out[ob + c] = y[eb + c] * s;
        }
    }
    Ok(out)
}

/// Solve `A x = b` with iterative refinement against the original matrix `a`.
/// Each step computes the residual `r = b - A x` and applies the correction
/// `x <- x + (LU)^-1 r`, stopping once `||r||inf` stops improving or `max_iter` is
/// reached. This recovers the accuracy a static / within-block-pivoted factor
/// loses on ill-conditioned matrices, at the cost of a few extra solves.
pub fn solve_lu_refined<T: Scalar>(
    f: &LuFactors<T>,
    a: &GeneralCsc<T>,
    b: &[T],
    max_iter: usize,
) -> Result<Vec<T>, RslabError> {
    Ok(solve_lu_refined_with(f, a, b, &crate::refine::RefinePolicy::steps(max_iter))?.0)
}

/// Iterative refinement of an `LU` solve under an explicit
/// [`RefinePolicy`](crate::refine::RefinePolicy), reporting the achieved
/// backward error.
pub fn solve_lu_refined_with<T: Scalar>(
    f: &LuFactors<T>,
    a: &GeneralCsc<T>,
    b: &[T],
    policy: &crate::refine::RefinePolicy,
) -> Result<(Vec<T>, crate::refine::RefineOutcome), RslabError> {
    let n = f.n;
    if a.n != n || b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    let mut x = solve_lu(f, b)?;
    let outcome = crate::refine::refine_in_place(a, b, &mut x, policy, |r| solve_lu(f, r))?;
    Ok((x, outcome))
}

#[cfg(test)]
mod tests {
    use crate::numeric::multifrontal_ldlt::MemoryMode;

    /// A badly scaled, row-scrambled unsymmetric system: without the MC64
    /// row matching the front-restricted pivoting finds no usable pivot or
    /// loses digits; with it (the default) the componentwise backward error
    /// is roundoff.
    #[test]
    fn lu_matching_bounds_pivot_growth() {
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
        // Componentwise backward error `max_i |r_i| / (|A||x| + |b|)_i`: the
        // rows span twelve decades, so a normwise residual would only
        // measure the largest rows.
        let omega = |x: &[f64]| {
            let mut r = b.clone();
            let mut d: Vec<f64> = b.iter().map(|v| v.abs()).collect();
            for j in 0..n {
                for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                    r[a.row_idx[k]] -= a.values[k] * x[j];
                    d[a.row_idx[k]] += a.values[k].abs() * x[j].abs();
                }
            }
            r.iter()
                .zip(&d)
                .map(|(ri, di)| if *di > 0.0 { ri.abs() / di } else { 0.0 })
                .fold(0.0, f64::max)
        };
        for method in [FactorMethod::LeftLooking, FactorMethod::Multifrontal] {
            let opts = SolverSettings::default()
                .with_threads(1)
                .with_method(method);
            let s = LuSolver::factor(&a, &opts).unwrap();
            assert_eq!(s.diagnostics().decisions.scaling, "Mc64RowMatching");
            let x = s.solve(&b).unwrap();
            assert!(
                omega(&x) < 1e-12,
                "{method:?} backward error with matching {}",
                omega(&x)
            );
            // Without the matching the shifted rows leave the fully-summed
            // blocks without a usable pivot.
            let s0 = LuSolver::factor(&a, &opts.with_lu_matching(false));
            assert!(s0.is_err() || s0.unwrap().diagnostics().decisions.scaling == "TwoSidedRowCol");
        }
    }
    use super::*;
    use num_complex::Complex;

    fn resid<T: Scalar>(a: &GeneralCsc<T>, x: &[T], b: &[T]) -> f64 {
        let mut y = vec![T::zero(); a.n];
        a.matvec(x, &mut y);
        (0..a.n)
            .map(|i| (y[i] - b[i]).magnitude())
            .fold(0.0, f64::max)
    }

    #[test]
    fn f64_unsymmetric_tridiag() {
        // Unsymmetric real tridiagonal (full storage): diag 4, sub -1, super -2.
        let n = 20;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            r.push(i);
            c.push(i);
            v.push(4.0);
            if i + 1 < n {
                r.push(i + 1);
                c.push(i);
                v.push(-1.0);
                r.push(i);
                c.push(i + 1);
                v.push(-2.0);
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let b: Vec<f64> = (0..n).map(|i| i as f64 - 9.5).collect();
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-10, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn lu_solve_many_matches_single() {
        let n = 12;
        let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            r.push(i);
            c.push(i);
            v.push(5.0_f64);
            if i + 1 < n {
                r.push(i + 1);
                c.push(i);
                v.push(-1.0);
                r.push(i);
                c.push(i + 1);
                v.push(-2.0);
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &r, &c, &v).unwrap();
        let solver = LuSolver::factor(&a, &SolverSettings::default()).unwrap();
        let nrhs = 5;
        let b: Vec<f64> = (0..n * nrhs).map(|k| (k % 7) as f64 - 3.0).collect();
        let x = solver.solve_many(&b, nrhs).unwrap();
        for col in 0..nrhs {
            let bc: Vec<f64> = (0..n).map(|i| b[i * nrhs + col]).collect();
            let xc = solver.solve(&bc).unwrap();
            for i in 0..n {
                assert!(
                    (x[i * nrhs + col] - xc[i]).abs() < 1e-10,
                    "rhs {col} row {i}"
                );
            }
        }
    }

    #[test]
    fn pivoting_triggered_small_diagonal() {
        // Small diagonal, large off-diagonals -> partial pivoting fires on
        // (nearly) every column. Well-conditioned overall, so the solve must
        // still hit a tiny residual: this isolates the pivoting/perm logic
        // (correctness) from numerical stability.
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
                vv.push(c(0.3, 0.05)); // small diagonal
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(2.0, 0.3)); // large off-diagonal
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
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-9, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn lu_left_looking_pivoting_small_diagonal() {
        // Small diagonal, large off-diagonals -> restricted partial pivoting must
        // fire on (nearly) every column. The left-looking path (1x1 static) would
        // eliminate on the tiny pivots and lose accuracy; with pivoting it must
        // match the multifrontal and hit a tiny residual.
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
                vv.push(c(0.05, 0.01)); // tiny diagonal -> threshold pivoting fires
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
        let ll = factor_general_lu(
            &a,
            &SolverSettings::default().with_method(FactorMethod::LeftLooking),
        )
        .unwrap();
        let mf = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let xl = solve_lu(&ll, &b).unwrap();
        let xm = solve_lu(&mf, &b).unwrap();
        let mut ax = vec![Complex::new(0.0, 0.0); n];
        a.matvec(&xl, &mut ax);
        let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-9, "left-looking pivoting residual {res}");
        let diff = (0..n).map(|i| (xl[i] - xm[i]).norm()).fold(0.0, f64::max);
        assert!(diff < 1e-9, "left-looking vs multifrontal differ {diff}");
    }

    #[test]
    fn lu_pivot_u_knob_wired_and_solves() {
        // The tunable threshold `u` governs the left-looking LU pivot test. On a
        // well-scaled, diagonally-dominant grid the pivot never needs to move, so
        // every `u in [0, 1]` must solve to a tiny residual (the knob changes the
        // factor path but not correctness here). Verifies the field is threaded
        // end-to-end (SolverSettings -> KernelTuning -> kernel) and clamps.
        let c = |re, im| Complex::new(re, im);
        let m = 7;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(12.0, 1.0)); // dominant diagonal -> no interchange needed
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-1.3, -0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.1, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.9, 0.15));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        for u in [0.0f64, 0.1, 0.5, 1.0] {
            let s = SolverSettings::default()
                .with_method(FactorMethod::LeftLooking)
                .with_pivot_u(u);
            let f = factor_general_lu(&a, &s).unwrap();
            let x = solve_lu(&f, &b).unwrap();
            let mut ax = vec![Complex::new(0.0, 0.0); n];
            a.matvec(&x, &mut ax);
            let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
            assert!(res < 1e-9, "pivot_u={u} residual {res}");
        }
        // Out-of-range values clamp into [0, 1].
        assert_eq!(SolverSettings::default().with_pivot_u(5.0).pivot_u, 1.0);
        assert_eq!(SolverSettings::default().with_pivot_u(-2.0).pivot_u, 0.0);
    }

    #[test]
    fn static_pivot_reuse_across_value_sweep() {
        // Solver-in-the-loop: analyze the pattern once, then factor a *sweep* of
        // value sets that share it with static pivoting (`pivot_u = 0`, no pivot
        // search per column). On a diagonally-dominant family each static factor
        // solves accurately, and iterative refinement against the original matrix
        // recovers full accuracy - the frequency-sweep / time-stepping use case.
        let c = |re, im| Complex::new(re, im);
        let m = 8;
        let n = m * m;
        let idx = |a: usize, b: usize| a * m + b;
        let (mut rr, mut cc) = (Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                if b + 1 < m {
                    rr.push(p);
                    cc.push(idx(a, b + 1));
                    rr.push(idx(a, b + 1));
                    cc.push(p);
                }
                if a + 1 < m {
                    rr.push(p);
                    cc.push(idx(a + 1, b));
                    rr.push(idx(a + 1, b));
                    cc.push(p);
                }
            }
        }
        let template =
            GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vec![c(1.0, 0.0); rr.len()])
                .unwrap();
        let analysis = LuSymbolic::analyze(&template).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 4.0, 0.7)).collect();
        let static_opts = SolverSettings::default().with_pivot_u(0.0);
        for shift in [0.0, 1.5, -0.8, 3.0] {
            let vv: Vec<Complex<f64>> = rr
                .iter()
                .zip(&cc)
                .map(|(&i, &j)| {
                    if i == j {
                        c(9.0 + shift, 1.0)
                    } else {
                        c(-1.0, 0.2)
                    }
                })
                .collect();
            let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
            // Reuse the one analysis; static factor (no pivot search).
            let f = factor_general_lu_numeric(&analysis, &a, &static_opts)
                .map(LuNumeric::into_factors)
                .unwrap();
            let x = solve_lu_refined(&f, &a, &b, 2).unwrap();
            let mut ax = vec![Complex::new(0.0, 0.0); n];
            a.matvec(&x, &mut ax);
            let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
            assert!(res < 1e-9, "static reuse shift={shift} residual {res}");
        }
    }

    #[test]
    fn complex_unsymmetric_2d_grid() {
        // 2D 5-point grid with unsymmetric neighbor couplings (right != left).
        let c = |re, im| Complex::new(re, im);
        let m = 8;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(8.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2)); // p,q
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1)); // q,p (different!)
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        assert!(resid(&a, &x, &b) < 1e-9, "residual {}", resid(&a, &x, &b));
    }

    #[test]
    fn low_memory_emit_is_bit_identical() {
        // MemoryMode::LowMemory frees each front's dense factor during emit; it
        // must produce exactly the same global L/U (it only changes when the
        // dense per-front buffers are dropped, never the emitted values).
        let m = 10;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(7.0_f64);
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(-1.0);
                    rr.push(q);
                    cc.push(p);
                    vv.push(-2.0);
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(-1.5);
                    rr.push(q);
                    cc.push(p);
                    vv.push(-0.5);
                }
            }
        }
        let a = GeneralCsc::<f64>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let sym = LuSymbolic::analyze(&a).unwrap();
        let eager = factor_general_lu_numeric(
            &sym,
            &a,
            &SolverSettings::default().with_memory(MemoryMode::Eager),
        )
        .map(LuNumeric::into_factors)
        .unwrap();
        let low = factor_general_lu_numeric(
            &sym,
            &a,
            &SolverSettings::default().with_memory(MemoryMode::LowMemory),
        )
        .map(LuNumeric::into_factors)
        .unwrap();
        assert_eq!(eager.l_values, low.l_values, "L differs under LowMemory");
        assert_eq!(eager.u_values, low.u_values, "U differs under LowMemory");
        assert_eq!(eager.l_row_idx, low.l_row_idx);
        assert_eq!(eager.u_col_idx, low.u_col_idx);
    }

    #[test]
    fn lu_left_looking_matches_multifrontal() {
        // Unsymmetric, diagonally dominant 2D grid (no pivoting needed -> the
        // multifrontal's partial pivoting takes the diagonal, matching the
        // left-looking 1x1 path). Same solution and comparable fill.
        let c = |re, im| Complex::new(re, im);
        let m = 14;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(16.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 0.5)).collect();
        let sym = LuSymbolic::analyze(&a).unwrap();
        let mf = sym.factor(&a, &SolverSettings::default()).unwrap();
        let ll = sym
            .factor(
                &a,
                &SolverSettings::default().with_method(FactorMethod::LeftLooking),
            )
            .unwrap();
        let xm = mf.solve(&b).unwrap();
        let xl = ll.solve(&b).unwrap();
        let mut am = vec![Complex::new(0.0, 0.0); n];
        a.matvec(&xl, &mut am);
        let res = (0..n).map(|i| (am[i] - b[i]).norm()).fold(0.0, f64::max);
        assert!(res < 1e-8, "left-looking LU residual {res}");
        let diff = (0..n).map(|i| (xm[i] - xl[i]).norm()).fold(0.0, f64::max);
        assert!(diff < 1e-8, "LU solutions differ by {diff}");
        // Fill should be within a few percent (same structure; values may zero
        // out differently under a different summation order).
        let (fm, fl) = (mf.factor_nnz() as f64, ll.factor_nnz() as f64);
        assert!((fm - fl).abs() / fm < 0.05, "fill mf={fm} ll={fl}");
    }

    #[test]
    fn complex_f32_lu_solves() {
        // The Complex<f32> LU path (used by the mixed-precision preconditioner).
        let c = |re: f32, im: f32| num_complex::Complex::<f32>::new(re, im);
        let m = 10;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(20.0, 2.0));
                if b + 1 < m {
                    rr.push(p);
                    cc.push(idx(a, b + 1));
                    vv.push(c(-1.0, 0.2));
                    rr.push(idx(a, b + 1));
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    rr.push(p);
                    cc.push(idx(a + 1, b));
                    vv.push(c(-1.5, 0.3));
                    rr.push(idx(a + 1, b));
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<num_complex::Complex<f32>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<num_complex::Complex<f32>> =
            (0..n).map(|i| c((i % 5) as f32 - 2.0, 1.0)).collect();
        let f = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let x = solve_lu(&f, &b).unwrap();
        let r = resid(&a, &x, &b);
        assert!(r < 1e-3, "f32 LU residual {}", r);
    }

    #[test]
    fn phased_general_lu_analyze_once_factor_many() {
        // PARDISO workflow for the unsymmetric path: analyze the pattern once,
        // factor several value sets that share it - each must match the
        // one-shot factor's solve. The frequency-sweep / Newton use case.
        let c = |re, im| Complex::new(re, im);
        let m = 7;
        let n = m * m;
        let (mut rr, mut cc) = (Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                if b + 1 < m {
                    rr.push(p);
                    cc.push(idx(a, b + 1));
                    rr.push(idx(a, b + 1));
                    cc.push(p);
                }
                if a + 1 < m {
                    rr.push(p);
                    cc.push(idx(a + 1, b));
                    rr.push(idx(a + 1, b));
                    cc.push(p);
                }
            }
        }
        // Template (values irrelevant) -> analyze once.
        let template =
            GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vec![c(1.0, 0.0); rr.len()])
                .unwrap();
        let analysis = LuSymbolic::analyze(&template).unwrap();
        assert_eq!(analysis.n(), n);

        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 4.0, 1.0)).collect();
        for shift in [0.0, 2.5, -1.0] {
            let vv: Vec<Complex<f64>> = rr
                .iter()
                .zip(&cc)
                .map(|(&i, &j)| {
                    if i == j {
                        c(8.0 + shift, 1.0)
                    } else {
                        c(-1.0, 0.2)
                    }
                })
                .collect();
            let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
            let phased = factor_general_lu_numeric(&analysis, &a, &SolverSettings::default())
                .map(LuNumeric::into_factors)
                .unwrap();
            let one_shot = factor_general_lu(&a, &SolverSettings::default()).unwrap();
            let xp = solve_lu(&phased, &b).unwrap();
            let xo = solve_lu(&one_shot, &b).unwrap();
            for (p, o) in xp.iter().zip(&xo) {
                assert!((p - o).norm() < 1e-10);
            }
            assert!(resid(&a, &xp, &b) < 1e-8);
        }
    }

    #[test]
    fn incomplete_lu_reduces_fill_and_still_solves() {
        use crate::numeric::multifrontal_ldlt::ZeroPivotAction;
        // Unsymmetric grid: incomplete LU (drop_tol) must shrink nnz(L+U) yet
        // still drive iterative refinement to a small residual - the MoM
        // sparse-preconditioner configuration.
        let c = |re, im| Complex::new(re, im);
        let m = 14;
        let n = m * m;
        let (mut rr, mut cc, mut vv) = (Vec::new(), Vec::new(), Vec::new());
        let idx = |a: usize, b: usize| a * m + b;
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                rr.push(p);
                cc.push(p);
                vv.push(c(8.0, 1.0));
                if b + 1 < m {
                    let q = idx(a, b + 1);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.0, 0.2));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-2.0, 0.1));
                }
                if a + 1 < m {
                    let q = idx(a + 1, b);
                    rr.push(p);
                    cc.push(q);
                    vv.push(c(-1.5, 0.3));
                    rr.push(q);
                    cc.push(p);
                    vv.push(c(-0.5, 0.4));
                }
            }
        }
        let a = GeneralCsc::<Complex<f64>>::from_triplets(n, &rr, &cc, &vv).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c((i % 5) as f64 - 2.0, 1.0)).collect();

        let full = factor_general_lu(&a, &SolverSettings::default()).unwrap();
        let opts = SolverSettings {
            on_zero_pivot: ZeroPivotAction::Fail,
            drop_tol: Some(5e-2),
            ..Default::default()
        };
        let inc = factor_general_lu(&a, &opts).unwrap();
        assert!(
            inc.factor_nnz() < full.factor_nnz(),
            "ILU should reduce fill: {} vs {}",
            inc.factor_nnz(),
            full.factor_nnz()
        );
        // The incomplete factor + a few refinement steps still solves accurately.
        let x = solve_lu_refined(&inc, &a, &b, 10).unwrap();
        assert!(resid(&a, &x, &b) < 1e-6, "residual {}", resid(&a, &x, &b));
    }
}
