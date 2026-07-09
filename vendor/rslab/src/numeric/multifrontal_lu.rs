//! Generic **unsymmetric** sparse LU factorization over any [`Scalar`] field -
//! the general (non-symmetric) complex path, complementing the symmetric LDLᵀ
//! path in [`crate::numeric::multifrontal_ldlt`].
//!
//! It targets matrices whose *values* are unsymmetric (e.g. MoM A-EFIE
//! near-field saddle preconditioners, where the symmetric and antisymmetric
//! parts are comparable) but reuses the full symmetric machinery: the
//! fill-reducing ordering, supernodes, assembly tree, level parallelism
//! ([`analyze`](crate::numeric::multifrontal_ldlt::analyze)) and the SIMD
//! `gemm` Schur kernel. Only the per-front kernel changes - an unsymmetric LU
//! producing separate `L` and `U` - and the analysis runs on the **symmetrized
//! pattern** `A ∪ Aᵀ` so the elimination structure carries fill for both
//! factors.
//!
//! ## Pivoting
//!
//! * **Threshold partial pivoting** (UMFPACK-style, `THRESH = 0.1`), bounded to
//!   each panel's fully-summed block: the diagonal is kept unless it falls below
//!   `THRESH · |colmax|`, in which case the column max is brought up. Sub-floor
//!   pivots are perturbed in preconditioner mode ([`ZeroPivotAction::PerturbToEps`])
//!   or rejected in exact mode. Pivoting stays cheap on the equilibrated,
//!   unit-diagonal MoM matrices while guarding the genuinely ill-scaled columns.
//! * The reassembled factors are global sparse `L` (CSC, unit lower) and `U`
//!   (CSR, upper with the pivots on the diagonal), in factorization order. The
//!   default factor path is the supernodal **left-looking** kernel (low transient,
//!   no CB stack); the multifrontal path is opt-in via [`SolverSettings::with_method`].

use crate::error::RslabError;
use crate::numeric::blr::BlrMatrix;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::multifrontal_ldlt::{
    analyze_with, compute_supernode_row_structures, perturb_pivot, BlrMode, FactorMethod,
    MemoryMode, SolverSettings, ZeroPivotAction,
};
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;
use crate::symbolic::SymbolicFactorization;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Reusable dense front-buffer pool. The multifrontal driver factors thousands
/// of fronts, each needing a transient `nrow² ` working buffer. Allocating one
/// per front (and freeing it) churns the system allocator with large, varying
/// sizes; on Windows the heap retains the freed blocks rather than returning
/// them to the OS, so peak RSS balloons far above the live set (the OOM the
/// pure-per-front allocation caused). This pool recycles a handful of buffers
/// (≈ the concurrency level) instead, capping the transient at the live set.
struct FrontPool<T>(Mutex<Vec<Vec<T>>>);

/// Only buffers up to this many entries are recycled. The churning majority of
/// small/medium fronts (which drive fragmentation) stay pooled; the rare huge
/// root/separator fronts are freed immediately rather than pinning their (GB-
/// scale) capacity in the pool for the whole factorization. 4M entries ≈ 64 MB
/// for `Complex<f64>` (front height ~2000).
const POOL_MAX_LEN: usize = 4_000_000;

impl<T: Scalar> FrontPool<T> {
    fn new() -> Self {
        FrontPool(Mutex::new(Vec::new()))
    }
    /// Take a buffer (reused if available) and zero-fill it to `len`.
    fn take(&self, len: usize) -> Vec<T> {
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
    fn give(&self, buf: Vec<T>) {
        if buf.capacity() <= POOL_MAX_LEN {
            self.0.lock().unwrap_or_else(|p| p.into_inner()).push(buf);
        }
    }
}

// Opt-in coarse factorization profiler (set `RLA_PROFILE=1`): CPU-nanosecond
// accumulators for the assembly (scatter + extend-add) vs the per-front LU
// kernel, summed across all worker threads. Zero overhead when disabled.
static PROF_ASM_NS: AtomicU64 = AtomicU64::new(0);
static PROF_FRONT_NS: AtomicU64 = AtomicU64::new(0);
static PROF_PANEL_NS: AtomicU64 = AtomicU64::new(0);
static PROF_EXTRACT_NS: AtomicU64 = AtomicU64::new(0);
// Left-looking phase profiler (assembly / cmod updates / cdiv panel factor).
static PROF_LL_ASM_NS: AtomicU64 = AtomicU64::new(0);
static PROF_LL_CMOD_NS: AtomicU64 = AtomicU64::new(0);
static PROF_LL_CDIV_NS: AtomicU64 = AtomicU64::new(0);
// cmod descendant-distribution profiler: counts/flops split by path
// (scalar tiny / serial gemm / parallel gemm) to expose update fragmentation.
static PROF_CMOD_SCAL_N: AtomicU64 = AtomicU64::new(0);
static PROF_CMOD_SCAL_F: AtomicU64 = AtomicU64::new(0);
static PROF_CMOD_GSER_N: AtomicU64 = AtomicU64::new(0);
static PROF_CMOD_GSER_F: AtomicU64 = AtomicU64::new(0);
static PROF_CMOD_GPAR_N: AtomicU64 = AtomicU64::new(0);
static PROF_CMOD_GPAR_F: AtomicU64 = AtomicU64::new(0);
// cdiv internal sub-phase profiler: serial getf2 panel / serial TRSM / parallel
// trailing GEMM - to locate the serial fraction inside the dense node factor.
static PROF_CDIV_GETF2_NS: AtomicU64 = AtomicU64::new(0);
static PROF_CDIV_TRSM_NS: AtomicU64 = AtomicU64::new(0);
static PROF_CDIV_GEMM_NS: AtomicU64 = AtomicU64::new(0);
static PROF_CMOD_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[inline]
fn cmod_prof_on() -> bool {
    *PROF_CMOD_FLAG.get_or_init(|| {
        std::env::var("RLA_PROFILE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}
// Experiment gate: force all left-looking GEMMs serial (no nested rayon), so the
// node-internal parallelism can be isolated from the tree-level `par_iter`.
static LL_GEMM_SERIAL_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[inline]
fn ll_gemm_serial() -> bool {
    *LL_GEMM_SERIAL_FLAG.get_or_init(|| {
        std::env::var("RLA_GEMM_SERIAL")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

thread_local! {
    /// Per-worker global→front-local index scratch (`usize`, scalar-independent),
    /// reused across every front a thread factors. Held at the all-`usize::MAX`
    /// invariant between uses; the assembly takes it, sets only the live front
    /// rows, and restores it. Replaces the old `map_init` workspace now that the
    /// driver is a work-stealing tree recursion rather than a level `par_iter`.
    static GLOC_SCRATCH: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

// --- BLR contribution-block compression (opt-in `RLA_BLR_CB`) --------------
//
// The dominant factorization transient is the live contribution-block (CB)
// stack - `Σ cnrow²` across the active assembly frontier, ≈5× the L/U volume and
// the source of the memory spike / OOM. A CB is a frontal Schur complement of a
// smooth (MoM near-field) operator, so its off-diagonal tiles are numerically
// low-rank. Storing each large CB block-low-rank on the stack - and densifying
// it tile-by-tile only at the parent's extend-add - shrinks the live stack by
// the tile compression ratio at a small per-tile densify transient. The factor
// then becomes approximate (a preconditioner), exactly the LU/MoM use case
// where GMRES refinement absorbs the BLR tolerance. The fast dense `gemm`
// `lu_front` kernel is unchanged; only the CB *storage* is compressed.

/// A front's contribution block: dense, or BLR-compressed for the stack.
enum Contribution<T> {
    Dense(Vec<T>),
    Blr(BlrMatrix<T>),
}

impl<T: Scalar> Contribution<T> {
    /// Extend-add this CB into front `f` (`nrow`-tall, column-major) at the
    /// parent-local positions `loc` (CB index → front-local). `cn` is the CB
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
                // Densify one tile at a time (≤ b×b transient) and scatter.
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

// Aggregate CB-compression accounting (written when CB-BLR is active).
static CB_DENSE_SCALARS: AtomicU64 = AtomicU64::new(0);
static CB_BLR_SCALARS: AtomicU64 = AtomicU64::new(0);

/// Read and reset the BLR-CB accounting: `(dense_equiv_scalars, stored_scalars)`
/// summed over the CBs compressed since the last call - the realized CB-stack
/// compression ratio. Returns `(0, 0)` when CB compression is disabled.
pub fn take_blr_cb_stats() -> (u64, u64) {
    (
        CB_DENSE_SCALARS.swap(0, Ordering::Relaxed),
        CB_BLR_SCALARS.swap(0, Ordering::Relaxed),
    )
}

/// Per-front unsymmetric partial-factorization output, in elimination order
/// (static pivoting → no interchange, so front-local order is pivot order).
struct FrontLu<T> {
    nrow: usize,
    nelim: usize,
    /// Unit-lower `L` of the front, `nrow × nelim` column-major (multipliers
    /// below the diagonal; unit diagonal implicit).
    l: Vec<T>,
    /// Upper `U` of the front as `nrow × nelim` column-major over the *row*
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
struct NodeLu<T> {
    front: FrontLu<T>,
    row_indices: Vec<usize>,
    /// The `cnrow × cnrow` contribution block `A22 − L21·U12` - dense, or
    /// BLR-compressed for the stack (opt-in `RLA_BLR_CB`).
    contrib: Contribution<T>,
}

/// Stored unsymmetric LU factors, in factorization order. Solve with
/// [`solve_lu`]. The factored system is `Pᵀ A P = L U`, `perm[e]` mapping
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
    /// Column permutation: factorization position → original column index
    /// (`Pᵀ A P = L U`, the fill-reducing ordering).
    pub perm: Vec<usize>,
    /// Row permutation: factorization position → original row index. Differs
    /// from `perm` when partial pivoting interchanged rows.
    pub perm_row: Vec<usize>,
    /// Two-sided equilibration: the factor is of `Â = diag(d_row)·A·diag(d_col)`.
    /// Solve applies `D_r` to the RHS and `D_c` to the result. Both length `n`.
    pub d_row: Vec<f64>,
    pub d_col: Vec<f64>,
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

impl<T: Scalar> LuFactors<T> {
    /// Stored fill: `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.l_values.len() + self.u_values.len()
    }
}

/// Lower triangle of the symmetrized pattern `A ∪ Aᵀ` as CSC `(col_ptr,
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
/// `f` (`nrow × nrow`, column-major) by unsymmetric LU with static pivoting.
/// The trailing `[ncol, nrow)` block is returned as the contribution block
/// `A22 − L21·U12` (computed by one `gemm`).
fn lu_front<T: Scalar>(
    f: &mut [T],
    nrow: usize,
    ncol: usize,
    perturb_floor: Option<f64>,
    blr: BlrMode,
    profile: bool,
    kt: KernelTuning,
) -> Result<(FrontLu<T>, Contribution<T>), RslabError> {
    let n = nrow;
    // BLR compressibility probe (opt-in `RLA_BLR_PROBE`): on large fronts, report
    // how low-rank the assembled off-diagonal blocks are - the empirical go/no-go
    // and rank distribution for the BLR front factorization.
    if nrow >= 512 && std::env::var("RLA_BLR_PROBE").is_ok() {
        crate::numeric::blr::probe_front(f, nrow, ncol, 256);
    }
    let t_panel = profile.then(std::time::Instant::now);
    let mut pivots = vec![T::zero(); ncol];
    let mut n_perturbed = 0usize;
    // Row permutation of the front (partial pivoting interchanges rows). Only
    // the fully-summed rows [0, ncol) are ever interchanged.
    let mut rperm: Vec<usize> = (0..nrow).collect();

    // Blocked right-looking LU (LAPACK getrf-style): factor the fully-summed
    // columns in panels of width `NB`. Each panel is factored unblocked (getf2,
    // within-block partial pivoting), then the dominant trailing update runs as
    // a single SIMD `gemm` (rank-`NB`) - routing the O(ncol²·nrow) work through
    // the complex BLAS-3 kernel instead of scalar BLAS-2 column sweeps. This is
    // the structure MKL/PARDISO use for the supernodal panel.
    // Panel width. Smaller than the LAPACK-typical 64 because the `gemm` complex
    // kernel is very fast (≈460 Gflop/s rank-64) while the unblocked getf2 panel
    // is serial BLAS-2: a narrower panel shifts work off the slow serial path
    // onto the fast parallel GEMM. NB=32 measured best on the MoM fronts.
    const NB: usize = 32;
    let mut kb = 0;
    while kb < ncol {
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
                          // --- TRSM: U[kb:ke, ke:nrow] = L11⁻¹ · A[kb:ke, ke:nrow] ---
                          // L11 is the unit-lower `pw×pw` diagonal block; forward-substitute each
                          // trailing column over the panel rows.
        for j in ke..n {
            for r in (kb + 1)..ke {
                let mut s = f[j * n + r];
                for i in kb..r {
                    s = s - f[i * n + r] * f[j * n + i]; // L11(r,i)·U12(i,j)
                }
                f[j * n + r] = s;
            }
        }
        // --- GEMM: A[ke:, ke:] −= L[ke:, kb:ke] · U[kb:ke, ke:] (rank-`pw`) ---
        let mt = n - ke; // trailing rows = cols
        if mt > 0 && pw > 0 {
            let flops = (mt as u128) * (mt as u128) * (pw as u128);
            let par = if flops >= kt.par_cdiv as u128 {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            // SAFETY: L21 (rows≥ke × cols[kb,ke)), U12 (rows[kb,ke) × cols≥ke)
            // and the A22 dst (rows≥ke × cols≥ke) are pairwise-disjoint
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
    if let Some(t) = t_panel {
        PROF_PANEL_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_extract = profile.then(std::time::Instant::now);
    // Pivots from the factored diagonal of the fully-summed block.
    for k in 0..ncol {
        pivots[k] = f[k * n + k];
    }

    // Extract L (nrow × ncol col-major, unit lower) and U (nrow × ncol
    // col-major over the row index, with the pivot on the diagonal), plus the
    // contribution block (`f`'s Schur-updated trailing A22 = A22 − L21·U12).
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
    // Each CB column is a contiguous run of `f`'s trailing block → memcpy.
    for c in 0..cnrow {
        let base = (ncol + c) * n + ncol;
        cb[c * cnrow..c * cnrow + cnrow].copy_from_slice(&f[base..base + cnrow]);
    }

    if let Some(t) = t_extract {
        PROF_EXTRACT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    // Compress large CBs for the stack (BlrMode::ContributionBlocks). The
    // off-diagonal tiles of the frontal Schur complement are low-rank; storing
    // them compressed shrinks the live CB-stack transient. Densified
    // tile-by-tile at the parent extend-add.
    let contribution = match blr {
        BlrMode::ContributionBlocks { eps, min_cnrow, b } if cnrow >= min_cnrow => {
            let blr = BlrMatrix::from_dense(&cb, cnrow, cnrow, b, eps);
            CB_DENSE_SCALARS.fetch_add((cnrow * cnrow) as u64, Ordering::Relaxed);
            CB_BLR_SCALARS.fetch_add(blr.storage() as u64, Ordering::Relaxed);
            Contribution::Blr(blr)
        }
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
    profile: bool,
    kt: KernelTuning,
) -> Result<NodeLu<T>, RslabError> {
    let snode = &sym.supernodes[s];
    let n = sym.n;
    let ncol = snode.ncol;
    let own_last = snode.first_col + ncol;
    let t_asm = profile.then(std::time::Instant::now);

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

    // Front buffer (transient `nrow²`), drawn from the reuse pool so the
    // thousands of large per-front buffers become a handful of recycled ones -
    // the fragmentation fix for the transient-memory OOM. Returned via
    // `pool.give` once `lu_front` has consumed it below.
    let mut fbuf: Vec<T> = pool.take(nrow * nrow);
    let f = &mut fbuf[..];

    // Take the thread-local global→local scratch for the assembly (held at the
    // all-`usize::MAX` invariant). It is returned before `lu_front` so the
    // front GEMM's work-stealing tasks can never re-enter the borrow.
    let mut gloc = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if gloc.len() < n {
        gloc.resize(n, usize::MAX);
    }
    for (li, &g) in ri.iter().enumerate() {
        gloc[g] = li;
    }

    // Owned columns (full): scatter a_perm column c into front column p.
    for p in 0..ncol {
        let c = snode.first_col + p;
        for k in a_perm.col_ptr[c]..a_perm.col_ptr[c + 1] {
            let g = a_perm.row_idx[k];
            let lr = gloc[g];
            if lr != usize::MAX {
                f[p * nrow + lr] = f[p * nrow + lr] + a_perm.values[k];
            }
        }
    }
    // Owned rows, trailing columns only (U12): scatter a_permᵀ column r (=
    // a_perm row r) into front row p for trailing front columns.
    for p in 0..ncol {
        let r = snode.first_col + p;
        for k in a_perm_t.col_ptr[r]..a_perm_t.col_ptr[r + 1] {
            let g = a_perm_t.row_idx[k];
            let lc = gloc[g];
            if lc != usize::MAX && lc >= ncol {
                f[lc * nrow + p] = f[lc * nrow + p] + a_perm_t.values[k];
            }
        }
    }

    // Extend-add each child's full contribution block. Map the child's
    // contribution rows to parent-front-local positions ONCE per child (`loc`,
    // reused across children) rather than re-indexing `gloc` inside the inner
    // loop - turning the cn² global→local lookups into cn, and slicing the
    // contiguous contribution column for the inner accumulation.
    let mut loc: Vec<usize> = Vec::new();
    for child in child_refs {
        let cn = child.front.nrow - child.front.nelim;
        let crows = &child.row_indices[child.front.nelim..];
        loc.clear();
        loc.extend(crows.iter().map(|&g| gloc[g]));
        child.contrib.extend_add_into(&loc, cn, f, nrow);
    }

    // Restore the all-`usize::MAX` invariant and return the scratch to the
    // thread-local before `lu_front` (which spawns work-stealing GEMM tasks).
    for &g in &ri {
        gloc[g] = usize::MAX;
    }
    GLOC_SCRATCH.with(|c| *c.borrow_mut() = gloc);

    if let Some(t) = t_asm {
        PROF_ASM_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let t_front = profile.then(std::time::Instant::now);
    let (front, contrib) = lu_front(f, nrow, ncol, perturb_floor, blr, profile, kt)?;
    if let Some(t) = t_front {
        PROF_FRONT_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    // `lu_front` has copied L/U/CB out; recycle the front buffer.
    pool.give(fbuf);
    Ok(NodeLu {
        front,
        row_indices: ri,
        contrib,
    })
}

/// A supernode's own factor plus the flat `(supernode-id, factor)` list for the
/// rest of its subtree - the return shape of [`factor_subtree`].
type SubtreeFactors<T> = (NodeLu<T>, Vec<(usize, NodeLu<T>)>);

/// Recursively factor the assembly subtree rooted at supernode `s` with a
/// **work-stealing tree schedule**: the children's subtrees are factored
/// concurrently (`par_iter`) and this node is factored only once they are done.
/// Independent subtrees fill idle threads automatically, and the per-front GEMM
/// shares the *same* rayon pool, so there is no level-barrier stall and no
/// nested-pool contention - the parallel-efficiency win over the old
/// level-synchronous driver.
///
/// Returns this node's factor plus a flat `(supernode-id, factor)` list for the
/// whole subtree, which the caller scatters into `node_results` for the global
/// emit pass.
#[allow(clippy::too_many_arguments)]
fn factor_subtree<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &GeneralCsc<T>,
    a_perm_t: &GeneralCsc<T>,
    perturb_floor: Option<f64>,
    blr: BlrMode,
    pool: &FrontPool<T>,
    profile: bool,
    kt: KernelTuning,
) -> Result<SubtreeFactors<T>, RslabError> {
    let children = &sym.supernodes[s].children;
    // Factor the child subtrees concurrently.
    let mut outs: Vec<SubtreeFactors<T>> = children
        .par_iter()
        .map(|&ch| {
            factor_subtree(
                ch,
                sym,
                a_perm,
                a_perm_t,
                perturb_floor,
                blr,
                pool,
                profile,
                kt,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Factor this node from the children's own (subtree-root) factors.
    let nf = {
        let child_refs: Vec<&NodeLu<T>> = outs.iter().map(|(own, _)| own).collect();
        factor_one_node_lu(
            s,
            sym,
            a_perm,
            a_perm_t,
            &child_refs,
            perturb_floor,
            blr,
            pool,
            profile,
            kt,
        )?
    };
    // Free the children's contribution blocks NOW: they have just been
    // extend-added into this front and are never read again (the global emit
    // pass uses only L/U). The CB stack is `Σ cnrow²` ≈ 5× the L/U volume and,
    // when retained to the end, dominated peak memory and caused OOMs. Dropping
    // each CB the moment its parent consumes it keeps only the active
    // contribution frontier live - the standard multifrontal CB-stack.
    for (own, _) in outs.iter_mut() {
        own.contrib = Contribution::Dense(Vec::new());
    }
    // Flatten the subtree's factors for the global pass (child `i` is the i-th
    // entry of `children`).
    let mut subtree = Vec::new();
    for (i, (own, rest)) in outs.into_iter().enumerate() {
        subtree.push((children[i], own));
        subtree.extend(rest);
    }
    Ok((nf, subtree))
}

/// Reusable symbolic analysis for the unsymmetric LU path - the symmetrized
/// pattern `A ∪ Aᵀ` analyzed once. Pass to [`factor_general_lu_numeric`] for
/// each value-set that shares the pattern (frequency sweep / Newton).
pub struct LuSymbolic {
    symb: crate::numeric::multifrontal_ldlt::MultifrontalSymbolic,
    n: usize,
    nnz: usize,
}

impl LuSymbolic {
    /// PARDISO **phase 1**: analyze the symmetrized pattern `A ∪ Aᵀ` of `a`
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
            });
        }
        let (col_ptr, row_idx) = symmetrized_lower_pattern(a);
        let symb = analyze_with(n, &col_ptr, &row_idx, opts)?;
        Ok(LuSymbolic { symb, n, nnz })
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
        let t = std::time::Instant::now();
        let factors = factor_general_lu_numeric(self, a, opts)?;
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        let nnz = factors.factor_nnz() as u64;
        let mut diagnostics = crate::diagnostics::Diagnostics {
            threads: resolved_threads,
            factor_nnz: nnz,
            estimate: Some(estimate),
            ..Default::default()
        };
        diagnostics.push("factor", factor_ms, 0, nnz * 24);
        Ok(LuSolver {
            factors,
            diagnostics,
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
    /// supernodes) — the reliable memory-backstop metric. Unlike
    /// [`MemoryEstimate::factor_nnz`](crate::diagnostics::MemoryEstimate::factor_nnz),
    /// a dense-panel upper bound that overshoots the real fill ~6-7x
    /// non-uniformly across orderings, this tracks the actually-stored fill.
    pub fn symbolic_factor_nnz(&self) -> usize {
        let Some((sym, _)) = self.symb.sym_and_levels() else {
            return 0;
        };
        let rs = compute_supernode_row_structures(sym);
        (0..sym.supernodes.len())
            .map(|s| {
                let nc = sym.supernodes[s].ncol;
                let cnrow = rs[s].len().saturating_sub(nc);
                // L: diagonal lower-triangle + off-diagonal rows; U: upper-tri + U12.
                let l = nc * (nc + 1) / 2 + cnrow * nc;
                let u = nc * (nc + 1) / 2 + nc * cnrow;
                l + u
            })
            .sum()
    }

    pub fn estimate_memory<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        let value_bytes = std::mem::size_of::<T>();
        let Some((sym, _levels)) = self.symb.sym_and_levels() else {
            return crate::diagnostics::estimate_left_looking(
                0,
                &|_| 0,
                &|_| 0,
                &[],
                value_bytes,
                0,
            );
        };
        let nsuper = sym.supernodes.len();
        let rs = compute_supernode_row_structures(sym);
        let mut col_to_snode = vec![0usize; self.n];
        for (s, snode) in sym.supernodes.iter().enumerate() {
            col_to_snode[snode.first_col..snode.first_col + snode.ncol].fill(s);
        }
        let mut update_list: Vec<Vec<usize>> = vec![Vec::new(); nsuper];
        for (k, rsk) in rs.iter().enumerate() {
            let nck = sym.supernodes[k].ncol;
            let mut last = usize::MAX;
            for &r in &rsk[nck..] {
                let s = col_to_snode[r];
                if s != last {
                    update_list[s].push(k);
                    last = s;
                }
            }
        }
        let panel_bytes = |s: usize| -> u64 {
            let nc = sym.supernodes[s].ncol;
            let nr = rs[s].len();
            ((nr * nc + nc * (nr - nc)) * value_bytes) as u64
        };
        let compact_bytes = |s: usize| -> u64 {
            let nc = sym.supernodes[s].ncol;
            let cnrow = rs[s].len() - nc;
            // L: diagonal lower-triangle + off-diagonal rows; U: upper-tri + U12.
            let l = nc * (nc + 1) / 2 + cnrow * nc;
            let u = nc * (nc + 1) / 2 + nc * cnrow;
            ((l + u) * (value_bytes + 8)) as u64
        };
        // Persistent input copies: the equilibrated permuted `a_perm` and its
        // transpose `a_perm_t` (both held through the left-looking factor).
        let input_bytes = (2 * self.nnz * (value_bytes + 8)) as u64;
        let mut est = crate::diagnostics::estimate_left_looking(
            nsuper,
            &panel_bytes,
            &compact_bytes,
            &update_list,
            value_bytes,
            input_bytes,
        );
        est.factor_flops = (0..nsuper)
            .map(|s| {
                let (nc, nr) = (sym.supernodes[s].ncol as u64, rs[s].len() as u64);
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
    diagnostics: crate::diagnostics::Diagnostics,
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
        Ok(Self {
            factors: factor_general_lu(a, opts)?,
            diagnostics: crate::diagnostics::Diagnostics::default(),
        })
    }

    /// Auto-tuned factorization at Pareto `weight` (`1` = fastest, `0` = smallest
    /// peak memory). Picks the settings from the matrix's structural features via
    /// the embedded performance model, **guarded** (only deviates from the default
    /// on a clear, memory-vetoed predicted win). The unsymmetric counterpart of
    /// [`LdltSolver::factor_auto`](crate::LdltSolver::factor_auto); pass explicit
    /// settings to [`factor`](Self::factor) to opt out.
    pub fn factor_auto(a: &GeneralCsc<T>, weight: f64) -> Result<Self, RslabError> {
        let (sym, s) = Self::tuned(a, weight)?;
        sym.factor(a, &s)
    }

    /// The auto-tuner's choice for `a`: the symbolic + guarded, memory-backstopped
    /// settings (shared by [`factor_auto`](Self::factor_auto) and the benchmark).
    pub fn tuned(
        a: &GeneralCsc<T>,
        weight: f64,
    ) -> Result<(LuSymbolic, SolverSettings), RslabError> {
        let sym = LuSymbolic::analyze(a)?;
        let est = sym.estimate_memory::<T>();
        let feat = crate::StructuralFeatures::from_general(a, &sym);
        let mf_ll = if est.panel_live_peak_bytes > 0 {
            est.mf_transient_peak_bytes as f64 / est.panel_live_peak_bytes as f64
        } else {
            1.0
        };
        let s = crate::auto_tune::recommend_settings_pathed(
            &feat,
            weight,
            mf_ll,
            crate::auto_tune::SolverPath::Lu,
        );
        let d = SolverSettings::default();
        // Fill compared via the *exact* symbolic fill, not the dense-panel
        // `MemoryEstimate::factor_nnz` (which overshoots ~6-7x non-uniformly across
        // orderings, so comparing two once could pass a pick with far more real fill).
        let default_fill = sym.symbolic_factor_nnz();
        let mem_ok = |e: &crate::diagnostics::MemoryEstimate, m: FactorMethod, pick_fill: usize| {
            let fill_ok = pick_fill as f64 <= default_fill as f64 * 1.02;
            let flops_ok = e.factor_flops as f64 <= est.factor_flops as f64 * 1.05;
            if m == FactorMethod::Multifrontal {
                fill_ok && flops_ok && e.mf_transient_peak_bytes <= est.panel_live_peak_bytes
            } else {
                fill_ok && flops_ok && e.panel_live_peak_bytes <= est.panel_live_peak_bytes
            }
        };
        if (s.reorder, s.ordering, s.nemin, s.relax) == (d.reorder, d.ordering, d.nemin, d.relax) {
            if mem_ok(&est, s.method, default_fill) {
                Ok((sym, s))
            } else {
                Ok((sym, d))
            }
        } else {
            let sym2 = LuSymbolic::analyze_with(a, &s)?;
            let est2 = sym2.estimate_memory::<T>();
            if mem_ok(&est2, s.method, sym2.symbolic_factor_nnz()) {
                Ok((sym2, s))
            } else {
                Ok((sym, d))
            }
        }
    }

    /// Per-call diagnostics: measured factor time, fill, thread budget, and the
    /// a-priori [`MemoryEstimate`](crate::diagnostics::MemoryEstimate). Populated by
    /// the phased [`LuSymbolic::factor`]; empty for the one-shot
    /// [`factor`](Self::factor).
    pub fn diagnostics(&self) -> &crate::diagnostics::Diagnostics {
        &self.diagnostics
    }

    /// Solve `A x = b` using the stored factors.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, RslabError> {
        solve_lu(&self.factors, b)
    }

    /// Solve `A · X = B` for `nrhs` right-hand sides at once. `b` and the
    /// returned `x` are **row-major** `n × nrhs` buffers (`b[i*nrhs + c]` is RHS
    /// `c` at row `i`). Faster than `nrhs` separate [`solve`](Self::solve) calls.
    pub fn solve_many(&self, b: &[T], nrhs: usize) -> Result<Vec<T>, RslabError> {
        solve_lu_many(&self.factors, b, nrhs)
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
        solve_lu_refined(&self.factors, a, b, max_iter)
    }

    /// Stored fill `nnz(L) + nnz(U)`.
    pub fn factor_nnz(&self) -> usize {
        self.factors.factor_nnz()
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
    pub fn factors(&self) -> &LuFactors<T> {
        &self.factors
    }
}

/// Factor a general (unsymmetric) sparse matrix `A` as `Pᵀ A P = L U` via
/// generic multifrontal LU with partial pivoting. `a` holds the **full** matrix
/// (both triangles). Convenience wrapper over [`LuSymbolic::analyze`] +
/// [`factor_general_lu_numeric`]; for *analyze once, factor many* keep the
/// [`LuSymbolic`] across calls. Solve with [`solve_lu`] / [`solve_lu_refined`].
pub fn factor_general_lu<T: Scalar>(
    a: &GeneralCsc<T>,
    opts: &SolverSettings,
) -> Result<LuFactors<T>, RslabError> {
    factor_general_lu_numeric(&LuSymbolic::analyze(a)?, a, opts)
}

// ===========================================================================
// ===========================================================================
// #14 front-growth introspection (opt-in: RSLAB_FRONT_STATS=1)
//
// Per-supernode factor extrema collected during the LU — the instrument for
// localizing element growth that outside-in probes cannot see (a single
// ordinary-looking row can carry the discharged error of a distant front).
// Collected into a process-global store (cheap Mutex push, gated) and drained
// by the caller via [`take_front_stats`]. Column/row indices are in the
// PERMUTED (post-ordering) numbering of this factorization.
// ===========================================================================

/// One supernode's factor extrema (permuted numbering).
#[derive(Clone, Copy, Debug)]
pub struct FrontStat {
    pub s: usize,
    pub first_col: usize,
    pub ncol: usize,
    pub nrow: usize,
    pub min_piv: f64,
    pub max_l: f64,
    pub max_u: f64,
    pub perturbed: usize,
}

static FRONT_STATS: std::sync::Mutex<Vec<FrontStat>> = std::sync::Mutex::new(Vec::new());

pub(crate) fn front_stats_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RSLAB_FRONT_STATS").as_deref() == Ok("1"))
}

/// Drain the front stats collected since the last call (empty unless
/// `RSLAB_FRONT_STATS=1` was set at first factor).
pub fn take_front_stats() -> Vec<FrontStat> {
    // Stats collection is best-effort diagnostics: recover the data even if a
    // panicking factor thread poisoned the mutex.
    std::mem::take(&mut FRONT_STATS.lock().unwrap_or_else(|p| p.into_inner()))
}

// Supernodal left-looking LU (FactorMethod::LeftLooking)
//
// The unsymmetric twin of the left-looking LDLᵀ path: each supernode keeps two
// dense panels - `lbuf` (its columns: diagonal block + L21, full height) and
// `ubuf` (its rows' U12: the trailing-column part) - assembled from `A` and
// updated by every factored descendant. The contribution of descendant `k` is
// the rank-`ncol_k` outer product `−L_k[Ok,:]·U_k[:,Ok]`; the part landing in
// `s` splits into two GEMMs: `−L_k[Ok,:]·U_k[:,Pk]` into `lbuf` (columns of `s`)
// and `−L_k[Pk,:]·U_k[:,trailing]` into `ubuf` (U12 rows of `s`). Then the panel
// is factored in place (`cdiv`) with **no trailing/CB update** - there is no
// contribution-block stack and no per-front extract copy-out, the PARDISO
// transient profile. 1×1 static pivoting (no row interchange), as in the
// multifrontal v1; matches the equilibrated preconditioner use case.
// ===========================================================================

/// Concurrently-filled store of the left-looking LU factor panels (`lbuf`,
/// `ubuf` per supernode). Each cell is written once by its owner before any
/// ancestor reads it (subtree recursion); concurrent writers are disjoint.
struct LuLlStore<T> {
    lbuf: Vec<std::cell::UnsafeCell<Vec<T>>>,
    ubuf: Vec<std::cell::UnsafeCell<Vec<T>>>,
    /// Per-node within-front row permutation from partial pivoting: `rperm[i]` is
    /// the row-structure index (`rs[s]`) physically at panel position `i`.
    /// Identity on the trailing rows (never interchanged); only read by the emit.
    rperm: Vec<std::cell::UnsafeCell<Vec<usize>>>,
}
// SAFETY: single-writer-before-readers, disjoint indices (see LDLᵀ `LlStore`).
unsafe impl<T: Send> Sync for LuLlStore<T> {}

/// Raw base pointer of a panel buffer, smuggled across rayon workers so each task
/// can write its own **disjoint row range** of a column-major panel. Safe only
/// because callers partition the rows so no two tasks touch the same cell.
#[derive(Clone, Copy)]
struct PanelPtr<T>(*mut T);
// SAFETY: the pointer is only dereferenced on disjoint, caller-partitioned cells.
unsafe impl<T> Send for PanelPtr<T> {}
unsafe impl<T> Sync for PanelPtr<T> {}
impl<T> PanelPtr<T> {
    /// Extract the raw pointer. Taking `self` by value forces a closure to capture
    /// the whole (Send+Sync) wrapper rather than disjoint-capturing the bare field.
    #[inline]
    fn get(self) -> *mut T {
        self.0
    }
}

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

impl<T: Scalar> LuLlStore<T> {
    fn new(nsuper: usize) -> Self {
        LuLlStore {
            lbuf: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(Vec::new()))
                .collect(),
            ubuf: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(Vec::new()))
                .collect(),
            rperm: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(Vec::new()))
                .collect(),
        }
    }
    /// SAFETY: `k` must be a fully-factored descendant of the current node.
    unsafe fn l(&self, k: usize) -> &Vec<T> {
        &*self.lbuf[k].get()
    }
    /// SAFETY: as [`l`](Self::l).
    unsafe fn u(&self, k: usize) -> &Vec<T> {
        &*self.ubuf[k].get()
    }
    /// SAFETY: factorization of `k` is complete.
    unsafe fn rperm(&self, k: usize) -> &Vec<usize> {
        &*self.rperm[k].get()
    }
    /// SAFETY: only the owner of supernode `s` calls this, exactly once.
    unsafe fn set(&self, s: usize, l: Vec<T>, u: Vec<T>, rperm: Vec<usize>) {
        *self.lbuf[s].get() = l;
        *self.ubuf[s].get() = u;
        *self.rperm[s].get() = rperm;
    }
    /// Release the dense panels + `rperm` of `k` once it has been compacted.
    /// SAFETY: `k`'s last consumer is done - no other thread reads its panels.
    unsafe fn free(&self, k: usize) {
        *self.lbuf[k].get() = Vec::new();
        *self.ubuf[k].get() = Vec::new();
        *self.rperm[k].get() = Vec::new();
    }
}

/// Compact (CSC-fragment) form of one supernode's factor, produced the moment its
/// last consumer has pulled from it - so the bulky dense panel can be freed
/// immediately instead of living until the global emit. Row/col indices are
/// already the final elimination positions; the global assembly is concatenation.
struct CompactNode<T> {
    l_ptr: Vec<usize>,
    l_idx: Vec<usize>,
    l_val: Vec<T>,
    u_ptr: Vec<usize>,
    u_idx: Vec<usize>,
    u_val: Vec<T>,
}
// Manual (not derived) so it holds for any `T` - `Vec<T>::new()` needs no bound.
impl<T> Default for CompactNode<T> {
    fn default() -> Self {
        CompactNode {
            l_ptr: Vec::new(),
            l_idx: Vec::new(),
            l_val: Vec::new(),
            u_ptr: Vec::new(),
            u_idx: Vec::new(),
            u_val: Vec::new(),
        }
    }
}

/// Incremental-emit state shared across the factorization workers: per-supernode
/// consumer refcounts (free the panel when it hits 0), the compact factor sink,
/// and the O(n) index maps populated in-node and read back after the barrier.
struct LlEmit<T> {
    /// Number of consumers (ancestors that pull) still to come; freed at 0.
    refcount: Vec<AtomicUsize>,
    /// First elimination position of each supernode (symbolic prefix sum of ncol).
    e_offset: Vec<usize>,
    compact: Vec<std::cell::UnsafeCell<CompactNode<T>>>,
    /// `e_of_g[g]` = elimination position of COLUMN g; `row_pos_of_g[g]` =
    /// position whose PIVOT ROW is g. Written in-node (disjoint g), read after the
    /// join barrier (and in `emit_and_free`, where the join chain makes consumer
    /// writes visible).
    e_of_g: Vec<std::cell::UnsafeCell<usize>>,
    row_pos_of_g: Vec<std::cell::UnsafeCell<usize>>,
    perm: Vec<std::cell::UnsafeCell<usize>>,
    perm_row: Vec<std::cell::UnsafeCell<usize>>,
}
// SAFETY: writes target disjoint indices; cross-thread visibility is established
// by the refcount Acquire/Release and the subtree-join happens-before chain.
unsafe impl<T: Send> Sync for LlEmit<T> {}

impl<T: Scalar> LlEmit<T> {
    fn new(sym: &SymbolicFactorization, update_list: &[Vec<usize>]) -> Self {
        let nsuper = sym.supernodes.len();
        let n = sym.n;
        let mut refcount: Vec<AtomicUsize> = (0..nsuper).map(|_| AtomicUsize::new(0)).collect();
        for ul in update_list {
            for &k in ul {
                *refcount[k].get_mut() += 1;
            }
        }
        let mut e_offset = vec![0usize; nsuper];
        let mut acc = 0usize;
        for (s, snode) in sym.supernodes.iter().enumerate() {
            e_offset[s] = acc;
            acc += snode.ncol;
        }
        LlEmit {
            refcount,
            e_offset,
            compact: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(CompactNode::default()))
                .collect(),
            e_of_g: (0..n)
                .map(|_| std::cell::UnsafeCell::new(usize::MAX))
                .collect(),
            row_pos_of_g: (0..n)
                .map(|_| std::cell::UnsafeCell::new(usize::MAX))
                .collect(),
            perm: (0..n).map(|_| std::cell::UnsafeCell::new(0)).collect(),
            perm_row: (0..n).map(|_| std::cell::UnsafeCell::new(0)).collect(),
        }
    }
    #[inline]
    unsafe fn eg(&self, g: usize) -> usize {
        *self.e_of_g[g].get()
    }
    #[inline]
    unsafe fn rg(&self, g: usize) -> usize {
        *self.row_pos_of_g[g].get()
    }
}

/// Compact supernode `k` into its CSC-fragment form, then release its dense
/// panels. Called the instant `k`'s last consumer has pulled from it, so the bulk
/// (dense panels) is freed during factorization instead of at the global emit.
/// Mirrors the per-supernode body of the legacy emit, resolving to final
/// elimination positions via the now-visible index maps.
fn emit_and_free<T: Scalar>(
    k: usize,
    store: &LuLlStore<T>,
    emit: &LlEmit<T>,
    sym: &SymbolicFactorization,
    rs: &[Vec<usize>],
    drop_tol: Option<f64>,
) {
    let snode = &sym.supernodes[k];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let nrow = rs[k].len();
    let cnrow = nrow - ncol;
    // SAFETY: `k` is fully factored and its last consumer is done - exclusive.
    let lbuf = unsafe { store.l(k) };
    let ubuf = unsafe { store.u(k) };
    let rperm = unsafe { store.rperm(k) };
    let one = T::one();
    let mut cn = CompactNode::<T>::default();
    cn.l_ptr.reserve(ncol + 1);
    cn.u_ptr.reserve(ncol + 1);
    cn.l_ptr.push(0);
    cn.u_ptr.push(0);
    // Reused per-column scratch (bounded by the row count), so the hot loop does
    // not reallocate; the compact vecs grow naturally to the (sparse) true fill.
    let mut lcol: Vec<(usize, T)> = Vec::with_capacity(nrow);
    let mut urow: Vec<(usize, T)> = Vec::with_capacity(ncol + cnrow);
    for p in 0..ncol {
        let diag_e = unsafe { emit.eg(first + p) };
        // L column: unit diagonal + strict-lower (row `i` came from rs[k][rperm[i]]).
        lcol.clear();
        lcol.push((diag_e, one));
        for i in (p + 1)..nrow {
            let v = lbuf[p * nrow + i];
            if v != T::zero() {
                lcol.push((unsafe { emit.rg(rs[k][rperm[i]]) }, v));
            }
        }
        if let Some(tau) = drop_tol {
            let colmax = lcol
                .iter()
                .filter(|&&(r, _)| r != diag_e)
                .map(|&(_, v)| v.magnitude())
                .fold(0.0, f64::max);
            let thr = tau * colmax;
            lcol.retain(|&(r, v)| r == diag_e || v.magnitude() >= thr);
        }
        lcol.sort_unstable_by_key(|&(r, _)| r);
        for &(r, v) in &lcol {
            cn.l_idx.push(r);
            cn.l_val.push(v);
        }
        cn.l_ptr.push(cn.l_idx.len());
        // U row: pivot + within-block upper + U12 (trailing columns).
        urow.clear();
        urow.push((diag_e, lbuf[p * nrow + p]));
        for j in (p + 1)..ncol {
            let v = lbuf[j * nrow + p];
            if v != T::zero() {
                urow.push((unsafe { emit.eg(first + j) }, v));
            }
        }
        for t in 0..cnrow {
            let v = ubuf[p + t * ncol];
            if v != T::zero() {
                urow.push((unsafe { emit.eg(rs[k][ncol + t]) }, v));
            }
        }
        if let Some(tau) = drop_tol {
            let rowmax = urow
                .iter()
                .filter(|&&(cc, _)| cc != diag_e)
                .map(|&(_, v)| v.magnitude())
                .fold(0.0, f64::max);
            let thr = tau * rowmax;
            urow.retain(|&(cc, v)| cc == diag_e || v.magnitude() >= thr);
        }
        urow.sort_unstable_by_key(|&(c, _)| c);
        for &(c, v) in &urow {
            cn.u_idx.push(c);
            cn.u_val.push(v);
        }
        cn.u_ptr.push(cn.u_idx.len());
    }
    // SAFETY: exactly one thread emits `k`; `compact[k]` is written once.
    unsafe { *emit.compact[k].get() = cn };
    // SAFETY: last consumer done - no other thread reads `k`'s panels.
    if !ll_no_free() {
        unsafe { store.free(k) };
    }
}

// A/B toggle (`RLA_NO_FREE=1`): keep dense panels resident (legacy behaviour) so
// the live-memory effect of incremental freeing can be measured in isolation.
static LL_NO_FREE_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[inline]
fn ll_no_free() -> bool {
    *LL_NO_FREE_FLAG.get_or_init(|| {
        std::env::var("RLA_NO_FREE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

#[allow(clippy::too_many_arguments)]
fn lu_ll_factor_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &GeneralCsc<T>,
    a_perm_t: &GeneralCsc<T>,
    rs: &[Vec<usize>],
    update_list: &[Vec<usize>],
    store: &LuLlStore<T>,
    emit: &LlEmit<T>,
    perturb_floor: Option<f64>,
    n_perturbed: &AtomicUsize,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    let ll_gemm_gate = kt.scalar_gate;
    let ll_gemm_par = kt.par_gemm;
    let snode = &sym.supernodes[s];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let nrow = rs[s].len();
    let cnrow = nrow - ncol;
    let n = sym.n;
    // `lbuf`: nrow×ncol (columns of s, full height). `ubuf`: ncol×cnrow (U12).
    let mut lbuf = vec![T::zero(); nrow * ncol];
    let mut ubuf = vec![T::zero(); ncol * cnrow];

    let mut gloc = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
    if gloc.len() < n {
        gloc.resize(n, usize::MAX);
    }
    for (li, &g) in rs[s].iter().enumerate() {
        gloc[g] = li;
    }
    let t_asm = std::time::Instant::now();
    // Assemble columns of s (full) into lbuf, and the U12 rows into ubuf.
    for p in 0..ncol {
        let c = first + p;
        for k in a_perm.col_ptr[c]..a_perm.col_ptr[c + 1] {
            let li = gloc[a_perm.row_idx[k]];
            if li != usize::MAX {
                lbuf[p * nrow + li] = lbuf[p * nrow + li] + a_perm.values[k];
            }
        }
        for k in a_perm_t.col_ptr[c]..a_perm_t.col_ptr[c + 1] {
            let lc = gloc[a_perm_t.row_idx[k]];
            if lc != usize::MAX && lc >= ncol {
                ubuf[p + (lc - ncol) * ncol] = ubuf[p + (lc - ncol) * ncol] + a_perm_t.values[k];
            }
        }
    }
    PROF_LL_ASM_NS.fetch_add(t_asm.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let t_cmod = std::time::Instant::now();
    // cmod from every factored descendant. NOTE: cmod-aggregation (K-stacking many
    // descendant updates into one fat GEMM) was measured and rejected - across MoM
    // topologies 91-95 % of cmod flop already runs as large parallel GEMMs, and the
    // only aggregation reaching those dominant updates carries an 11-15× zero-pad
    // blowup (each top-of-tree descendant touches a small, distinct row/col subset
    // of the large target). The `RLA_CMOD_DIST` histogram below documents this.
    let mut lupd: Vec<T> = Vec::new();
    let mut uupd: Vec<T> = Vec::new();
    for &kk in &update_list[s] {
        let nck = sym.supernodes[kk].ncol;
        let nrk = rs[kk].len();
        let ok = &rs[kk][nck..];
        let nok = ok.len();
        // SAFETY: `kk` is a factored descendant of `s`.
        let lk = unsafe { store.l(kk) };
        let uk = unsafe { store.u(kk) };
        let p0 = ok.partition_point(|&g| g < first);
        let p1 = ok.partition_point(|&g| g < first + ncol);
        let npk = p1 - p0;
        if npk == 0 {
            continue;
        }
        let mrows = nok - p0; // rows used by the L update (Ok ⊆ rs[s] from here)
        let ntrail = nok - p1;
        if cmod_prof_on() {
            let flop = (mrows * npk * nck) as u64;
            if mrows * npk * nck < ll_gemm_gate {
                PROF_CMOD_SCAL_N.fetch_add(1, Ordering::Relaxed);
                PROF_CMOD_SCAL_F.fetch_add(flop, Ordering::Relaxed);
            } else if mrows * npk * nck >= ll_gemm_par {
                PROF_CMOD_GPAR_N.fetch_add(1, Ordering::Relaxed);
                PROF_CMOD_GPAR_F.fetch_add(flop, Ordering::Relaxed);
            } else {
                PROF_CMOD_GSER_N.fetch_add(1, Ordering::Relaxed);
                PROF_CMOD_GSER_F.fetch_add(flop, Ordering::Relaxed);
            }
        }
        if mrows * npk * nck < ll_gemm_gate {
            // Scalar path.
            for jj in 0..npk {
                let tcol = ok[p0 + jj] - first;
                for i in 0..mrows {
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc + lk[(nck + p0 + i) + ck * nrk] * uk[ck + (p0 + jj) * nck];
                    }
                    let trow = gloc[ok[p0 + i]];
                    lbuf[tcol * nrow + trow] = lbuf[tcol * nrow + trow] - acc;
                }
            }
            for jj in 0..ntrail {
                let tu = gloc[ok[p1 + jj]] - ncol;
                for i in 0..npk {
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc + lk[(nck + p0 + i) + ck * nrk] * uk[ck + (p1 + jj) * nck];
                    }
                    let urow = ok[p0 + i] - first;
                    ubuf[urow + tu * ncol] = ubuf[urow + tu * ncol] - acc;
                }
            }
        } else {
            let par = if !ll_gemm_serial() && mrows * npk * nck >= ll_gemm_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            // L update: Lupd(mrows×npk) = L_k[Ok≥p0,:] · U_k[:,Pk].
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
                let cbase = (ok[p0 + jj] - first) * nrow;
                let ucol = &lupd[jj * mrows..jj * mrows + mrows];
                for i in 0..mrows {
                    let dst = cbase + gloc[ok[p0 + i]];
                    lbuf[dst] = lbuf[dst] - ucol[i];
                }
            }
            // U update: Uupd(npk×ntrail) = L_k[Pk,:] · U_k[:,trailing].
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
                    let ubase = (gloc[ok[p1 + jj]] - ncol) * ncol;
                    let ucol = &uupd[jj * npk..jj * npk + npk];
                    for i in 0..npk {
                        let dst = ubase + (ok[p0 + i] - first);
                        ubuf[dst] = ubuf[dst] - ucol[i];
                    }
                }
            }
        }
    }
    PROF_LL_CMOD_NS.fetch_add(t_cmod.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let t_cdiv = std::time::Instant::now();
    // cdiv: in-place **blocked** panel LU (1×1 static pivoting), no trailing/CB
    // update. Mirrors the multifrontal `lu_front` getrf - unblocked `getf2` over
    // an NB-wide panel, then the dominant trailing update as a single SIMD GEMM
    // (rank-NB) - but restricted to the panel: the trailing is the remaining
    // panel columns (`lbuf`) plus the `U12` rows (`ubuf`), with no `A22`/CB. This
    // routes the `O(ncol²·nrow)` cdiv work (the measured 77 % of the left-looking
    // factor) through BLAS-3 instead of scalar rank-1 sweeps.
    // Panel width. Swept 32/48/64/96 on the MoM fronts (even with the deep trailing
    // rows now factored in a parallel apply): 32 stays optimal - wider panels add
    // serial fully-summed getf2 without a matching trailing-GEMM efficiency gain.
    const NB_CDIV: usize = 32;
    let ll_cdiv_par = kt.par_cdiv;
    let mut local_perturbed = 0usize;
    // Restricted partial pivoting: row interchanges within the fully-summed block
    // `[0, ncol)` only (the standard sparse-direct choice). `rperm[i]` is the
    // row-structure index physically at position `i`; the trailing rows are never
    // interchanged, so the contribution rows `Ok` ancestors pull are unaffected
    // and `cmod` needs no permutation awareness.
    let mut rperm: Vec<usize> = (0..nrow).collect();
    let prof = cmod_prof_on();
    // Pivot reciprocals of the current panel, reused by the parallel trailing apply.
    let mut pinv_blk: Vec<T> = vec![T::zero(); NB_CDIV];
    let mut kb = 0;
    while kb < ncol {
        let ke = (kb + NB_CDIV).min(ncol);
        let t_g = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // getf2: factor columns [kb, ke) over the **fully-summed rows [k+1, ncol)**
        // only - the deep trailing rows [ncol, nrow) (never pivot candidates) are
        // lifted off this serial path into the parallel apply below.
        for k in kb..ke {
            // **Threshold** partial pivoting (UMFPACK-style): keep the diagonal
            // pivot unless it is below `THRESH` of the largest candidate in the
            // fully-summed block - so a well-scaled/equilibrated matrix never
            // interchanges (no fill or accuracy cost) while small/zero diagonals
            // still get a stable pivot. `THRESH²` compared on squared magnitudes.
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
                    for &g in &rs[s] {
                        gloc[g] = usize::MAX;
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
            let par = !ll_gemm_serial() && cnrow * pw * pw >= ll_cdiv_par;
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
        if let Some(t) = t_g {
            PROF_CDIV_GETF2_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let t_t = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // TRSM: U = L11⁻¹ · (trailing panel columns of lbuf) and the U12 rows.
        for j in ke..ncol {
            for r in (kb + 1)..ke {
                let mut acc = lbuf[j * nrow + r];
                for i in kb..r {
                    acc = acc - lbuf[i * nrow + r] * lbuf[j * nrow + i];
                }
                lbuf[j * nrow + r] = acc;
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
        if !ll_gemm_serial() && cnrow * pw * pw >= ll_cdiv_par {
            let lref: &[T] = &lbuf;
            ubuf.par_chunks_mut(ncol).for_each(|col| trsm_u(col, lref));
        } else {
            for t in 0..cnrow {
                trsm_u(&mut ubuf[t * ncol..t * ncol + ncol], &lbuf);
            }
        }
        if let Some(t) = t_t {
            PROF_CDIV_TRSM_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let t_m = if prof {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // GEMM: lbuf[ke.., ke..ncol] −= L21[ke.., kb..ke] · U[kb..ke, ke..ncol].
        let mt = nrow - ke;
        let nt = ncol - ke;
        if mt > 0 && nt > 0 {
            let par = if !ll_gemm_serial() && (mt * nt * pw) >= ll_cdiv_par {
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
        // GEMM: ubuf[ke..ncol, :] −= L[ke..ncol, kb..ke] · U12[kb..ke, :].
        if cnrow > 0 && nt > 0 {
            let par = if !ll_gemm_serial() && (nt * cnrow * pw) >= ll_cdiv_par {
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
        if let Some(t) = t_m {
            PROF_CDIV_GEMM_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        kb = ke;
    }
    PROF_LL_CDIV_NS.fetch_add(t_cdiv.elapsed().as_nanos() as u64, Ordering::Relaxed);
    if local_perturbed > 0 {
        n_perturbed.fetch_add(local_perturbed, Ordering::Relaxed);
    }
    // #14 front-growth introspection: record this supernode's factor extrema.
    if front_stats_on() {
        let mut min_piv = f64::INFINITY;
        for k in 0..ncol {
            let m = lbuf[k * nrow + k].magnitude();
            if m < min_piv {
                min_piv = m;
            }
        }
        let max_l = lbuf.iter().map(|v| v.magnitude()).fold(0.0f64, f64::max);
        let max_u = ubuf.iter().map(|v| v.magnitude()).fold(0.0f64, f64::max);
        FRONT_STATS
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(FrontStat {
                s,
                first_col: first,
                ncol,
                nrow,
                min_piv,
                max_l,
                max_u,
                perturbed: local_perturbed,
            });
    }
    for &g in &rs[s] {
        gloc[g] = usize::MAX;
    }
    GLOC_SCRATCH.with(|c| *c.borrow_mut() = gloc);
    // Populate the O(n) index maps for `s` from its (final) `rperm` and the
    // symbolic elimination offset - consumed by `emit_and_free` and the assembly.
    // Writes target disjoint global indices; visibility via the subtree join.
    let eoff = emit.e_offset[s];
    for p in 0..ncol {
        let g_col = first + p;
        let g_row = rs[s][rperm[p]];
        // SAFETY: each global index is written by exactly one supernode.
        unsafe {
            *emit.e_of_g[g_col].get() = eoff + p;
            *emit.row_pos_of_g[g_row].get() = eoff + p;
            *emit.perm[eoff + p].get() = sym.perm[g_col];
            *emit.perm_row[eoff + p].get() = sym.perm[g_row];
        }
    }
    // SAFETY: this thread owns `s`, writes its cells exactly once.
    unsafe { store.set(s, lbuf, ubuf, rperm) };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lu_ll_factor_subtree<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    a_perm: &GeneralCsc<T>,
    a_perm_t: &GeneralCsc<T>,
    rs: &[Vec<usize>],
    update_list: &[Vec<usize>],
    store: &LuLlStore<T>,
    emit: &LlEmit<T>,
    perturb_floor: Option<f64>,
    drop_tol: Option<f64>,
    n_perturbed: &AtomicUsize,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    sym.supernodes[s]
        .children
        .par_iter()
        .map(|&ch| {
            lu_ll_factor_subtree(
                ch,
                sym,
                a_perm,
                a_perm_t,
                rs,
                update_list,
                store,
                emit,
                perturb_floor,
                drop_tol,
                n_perturbed,
                kt,
            )
        })
        .collect::<Result<Vec<()>, _>>()?;
    lu_ll_factor_node(
        s,
        sym,
        a_perm,
        a_perm_t,
        rs,
        update_list,
        store,
        emit,
        perturb_floor,
        n_perturbed,
        kt,
    )?;
    // `s` has now pulled from every descendant in its update list - for each whose
    // last consumer this was (refcount→0), compact it and free its dense panel.
    // `s` itself is freed here iff it has no consumers (root / no fill above).
    // Each `emit_and_free(k)` touches a disjoint `k`, so for the wide top-of-tree
    // nodes (where tree parallelism is exhausted) the compaction runs in parallel.
    const FREE_PAR: usize = 64;
    if update_list[s].len() >= FREE_PAR {
        update_list[s].par_iter().for_each(|&k| {
            if emit.refcount[k].fetch_sub(1, Ordering::AcqRel) == 1 {
                emit_and_free(k, store, emit, sym, rs, drop_tol);
            }
        });
    } else {
        for &k in &update_list[s] {
            if emit.refcount[k].fetch_sub(1, Ordering::AcqRel) == 1 {
                emit_and_free(k, store, emit, sym, rs, drop_tol);
            }
        }
    }
    if emit.refcount[s].load(Ordering::Relaxed) == 0 {
        emit_and_free(s, store, emit, sym, rs, drop_tol);
    }
    Ok(())
}

/// Supernodal left-looking LU producing the same [`LuFactors`] as the
/// multifrontal path. `a_perm`/`a_perm_t` are the equilibrated permuted matrix
/// and its transpose; `d_row`/`d_col` the equilibration carried into the result.
#[allow(clippy::too_many_arguments)]
fn factor_lu_left_looking<T: Scalar>(
    sym: &SymbolicFactorization,
    a_perm: &GeneralCsc<T>,
    a_perm_t: &GeneralCsc<T>,
    d_row: &[f64],
    d_col: &[f64],
    perturb_floor: Option<f64>,
    drop_tol: Option<f64>,
    kt: KernelTuning,
) -> Result<LuFactors<T>, RslabError> {
    let n = sym.n;
    let nsuper = sym.supernodes.len();
    let rs = compute_supernode_row_structures(sym);

    let mut col_to_snode = vec![0usize; n];
    for (s, snode) in sym.supernodes.iter().enumerate() {
        col_to_snode[snode.first_col..snode.first_col + snode.ncol].fill(s);
    }
    let mut update_list: Vec<Vec<usize>> = vec![Vec::new(); nsuper];
    for (k, rsk) in rs.iter().enumerate() {
        let nck = sym.supernodes[k].ncol;
        let mut last = usize::MAX;
        for &r in &rsk[nck..] {
            let s = col_to_snode[r];
            if s != last {
                update_list[s].push(k);
                last = s;
            }
        }
    }

    // Diagnostic (RLA_PROFILE): simulate the achievable peak panel memory under a
    // refcount free-schedule (free a panel once its last consumer has pulled it),
    // in elimination/postorder, vs the current all-panels-resident peak. Decides
    // whether panel-freeing is worth the concurrent machinery before building it.
    if cmod_prof_on() {
        let tb = std::mem::size_of::<T>();
        let bytes = |k: usize| -> usize {
            let nc = sym.supernodes[k].ncol;
            let nr = rs[k].len();
            (nr * nc + nc * (nr - nc)) * tb
        };
        let total: usize = (0..nsuper).map(bytes).sum();
        let mut refc = vec![0usize; nsuper];
        for ul in &update_list {
            for &k in ul {
                refc[k] += 1;
            }
        }
        let mut live = 0usize;
        let mut peak = 0usize;
        for s in 0..nsuper {
            live += bytes(s);
            for &k in &update_list[s] {
                refc[k] -= 1;
                if refc[k] == 0 {
                    live -= bytes(k);
                }
            }
            if refc[s] == 0 {
                live -= bytes(s);
            }
            peak = peak.max(live);
        }
        eprintln!(
            "[RLA_LL_MEMSIM] panels: all-resident {:.0}MB  refcount-freed peak {:.0}MB  ({:.2}x reduction)",
            total as f64 / 1e6,
            peak as f64 / 1e6,
            total as f64 / (peak.max(1) as f64),
        );
    }
    let store = LuLlStore::<T>::new(nsuper);
    let emit = LlEmit::<T>::new(sym, &update_list);
    let n_perturbed_atomic = AtomicUsize::new(0);
    let mut is_child = vec![false; nsuper];
    for snode in &sym.supernodes {
        for &ch in &snode.children {
            is_child[ch] = true;
        }
    }
    let roots: Vec<usize> = (0..nsuper).filter(|&s| !is_child[s]).collect();
    roots
        .par_iter()
        .map(|&r| {
            lu_ll_factor_subtree(
                r,
                sym,
                a_perm,
                a_perm_t,
                &rs,
                &update_list,
                &store,
                &emit,
                perturb_floor,
                drop_tol,
                &n_perturbed_atomic,
                kt,
            )
        })
        .collect::<Result<Vec<()>, _>>()?;
    drop(store); // all panels already freed incrementally; release the shells
    let n_perturbed = n_perturbed_atomic.load(Ordering::Relaxed);
    if std::env::var("RLA_PROFILE")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        let asm = PROF_LL_ASM_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let cmod = PROF_LL_CMOD_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let cdiv = PROF_LL_CDIV_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let tot = (asm + cmod + cdiv).max(1.0);
        eprintln!(
            "[RLA_LL_PROFILE] CPU-ms  asm {asm:.0} ({:.0}%)  cmod {cmod:.0} ({:.0}%)  cdiv {cdiv:.0} ({:.0}%)",
            100.0 * asm / tot,
            100.0 * cmod / tot,
            100.0 * cdiv / tot,
        );
        let sn = PROF_CMOD_SCAL_N.swap(0, Ordering::Relaxed);
        let sf = PROF_CMOD_SCAL_F.swap(0, Ordering::Relaxed);
        let rn = PROF_CMOD_GSER_N.swap(0, Ordering::Relaxed);
        let rf = PROF_CMOD_GSER_F.swap(0, Ordering::Relaxed);
        let pn = PROF_CMOD_GPAR_N.swap(0, Ordering::Relaxed);
        let pf = PROF_CMOD_GPAR_F.swap(0, Ordering::Relaxed);
        let ftot = (sf + rf + pf).max(1) as f64;
        eprintln!(
            "[RLA_CMOD_DIST] updates  scalar n={sn} ({:.1}% flop)  gemm-ser n={rn} ({:.1}% flop)  gemm-par n={pn} ({:.1}% flop)",
            100.0 * sf as f64 / ftot,
            100.0 * rf as f64 / ftot,
            100.0 * pf as f64 / ftot,
        );
        let g2 = PROF_CDIV_GETF2_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let tr = PROF_CDIV_TRSM_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let gm = PROF_CDIV_GEMM_NS.swap(0, Ordering::Relaxed) as f64 / 1e6;
        let ct = (g2 + tr + gm).max(1.0);
        eprintln!(
            "[RLA_CDIV_SUB] CPU-ms  getf2 {g2:.0} ({:.0}% ser)  trsm {tr:.0} ({:.0}% ser)  gemm {gm:.0} ({:.0}% par)",
            100.0 * g2 / ct,
            100.0 * tr / ct,
            100.0 * gm / ct,
        );
    }

    // Assemble the global L (CSC) / U (CSR) by concatenating the per-supernode
    // compact fragments produced (and freed) incrementally during factorization.
    // Supernodes are in elimination order, so concatenation yields e-ordered
    // columns/rows directly - no re-indexing, no dense panels alive here.
    let (mut l_col_ptr, mut l_row_idx, mut l_values) =
        (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
    let (mut u_row_ptr, mut u_col_idx, mut u_values) =
        (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
    l_col_ptr.push(0);
    u_row_ptr.push(0);
    for (s, snode) in sym.supernodes.iter().enumerate() {
        // Take ownership and drop each fragment right after appending, so the peak
        // is (growing final CSC) + (one supernode's fragment), not all fragments +
        // the full CSC simultaneously.
        // SAFETY: factorization complete; `compact[s]` written exactly once.
        let cn = unsafe { std::mem::take(&mut *emit.compact[s].get()) };
        for c in 0..snode.ncol {
            let (la, lb) = (cn.l_ptr[c], cn.l_ptr[c + 1]);
            l_row_idx.extend_from_slice(&cn.l_idx[la..lb]);
            l_values.extend_from_slice(&cn.l_val[la..lb]);
            l_col_ptr.push(l_row_idx.len());
            let (ua, ub) = (cn.u_ptr[c], cn.u_ptr[c + 1]);
            u_col_idx.extend_from_slice(&cn.u_idx[ua..ub]);
            u_values.extend_from_slice(&cn.u_val[ua..ub]);
            u_row_ptr.push(u_col_idx.len());
        }
    }
    // SAFETY: factorization complete; every position written exactly once in-node.
    let perm: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm[e].get() }).collect();
    let perm_row: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm_row[e].get() }).collect();

    Ok(LuFactors {
        n,
        l_col_ptr,
        l_row_idx,
        l_values,
        u_row_ptr,
        u_col_idx,
        u_values,
        perm,
        perm_row,
        d_row: d_row.to_vec(),
        d_col: d_col.to_vec(),
        n_perturbed,
        // Placeholder; the caller (`factor_general_lu_numeric`) overwrites this
        // with the resolved solve-phase policy once the factor is built.
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
) -> Result<LuFactors<T>, RslabError> {
    a.validate()?;
    let n = lusym.n;
    if a.n != n || a.row_idx.len() != lusym.nnz {
        return Err(RslabError::InvalidInput(
            "factor_general_lu_numeric: matrix does not match the analyzed pattern".to_string(),
        ));
    }
    if n == 0 {
        return Ok(LuFactors {
            n: 0,
            l_col_ptr: vec![0],
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
            u_row_ptr: vec![0],
            u_col_idx: Vec::new(),
            u_values: Vec::new(),
            perm: Vec::new(),
            perm_row: Vec::new(),
            d_row: Vec::new(),
            d_col: Vec::new(),
            n_perturbed: 0,
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

    // Two-sided equilibration Â = D_r A D_c with d_r[i] = 1/√maxⱼ|Aᵢⱼ|,
    // d_c[j] = 1/√maxᵢ|Aᵢⱼ|. Tames the dynamic range (these MoM near-field
    // matrices span ~6 orders) so the LU factor - and any incomplete drop -
    // stays well-scaled; the solve undoes it transparently. Computed from the
    // original (unpermuted) A.
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
    let d_row: Vec<f64> = rmax
        .iter()
        .map(|&r| if r > 0.0 { 1.0 / r.sqrt() } else { 1.0 })
        .collect();
    let d_col: Vec<f64> = cmax
        .iter()
        .map(|&c| if c > 0.0 { 1.0 / c.sqrt() } else { 1.0 })
        .collect();

    // Full permuted, equilibrated matrix Â_perm = Pᵀ (D_r A D_c) P and its
    // transpose (no triangle folding - unsymmetric values kept distinct).
    let nnz = a.row_idx.len();
    let (mut rows, mut cols, mut vals) = (
        Vec::with_capacity(nnz),
        Vec::with_capacity(nnz),
        Vec::with_capacity(nnz),
    );
    for j in 0..n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let i = a.row_idx[k];
            rows.push(sym.perm_inv[i]);
            cols.push(sym.perm_inv[j]);
            vals.push(a.values[k] * T::from_real(d_row[i] * d_col[j]));
        }
    }
    let a_perm = GeneralCsc::<T>::from_triplets(n, &rows, &cols, &vals)?;
    let a_perm_t = a_perm.transpose();
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
        return Ok(fac);
    }

    let profile = std::env::var("RLA_PROFILE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if profile {
        PROF_ASM_NS.store(0, Ordering::Relaxed);
        PROF_FRONT_NS.store(0, Ordering::Relaxed);
    }

    let nsuper = sym.supernodes.len();
    // Roots of the assembly forest: supernodes that are no node's child.
    let mut is_child = vec![false; nsuper];
    for snode in &sym.supernodes {
        for &ch in &snode.children {
            is_child[ch] = true;
        }
    }
    let roots: Vec<usize> = (0..nsuper).filter(|&s| !is_child[s]).collect();
    // Factor every root subtree with the work-stealing tree schedule; the
    // children-before-parent dependency is the recursion structure itself.
    let pool = FrontPool::<T>::new();
    let kt = opts.kernel();
    // Scoped pool of `opts.threads` with the depth-sized stack (honours the thread
    // budget and is overflow-safe on deep trees, like the left-looking path).
    let root_outs: Vec<SubtreeFactors<T>> = opts.threads.run(
        stack,
        |cap| crate::numeric::multifrontal_ldlt::recommend_threads_for_sym(&lusym.symb, cap),
        || {
            roots
                .par_iter()
                .map(|&r| {
                    factor_subtree(
                        r,
                        sym,
                        &a_perm,
                        &a_perm_t,
                        perturb_floor,
                        blr,
                        &pool,
                        profile,
                        kt,
                    )
                })
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
    if profile {
        let asm = PROF_ASM_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let front = PROF_FRONT_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let panel = PROF_PANEL_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let extract = PROF_EXTRACT_NS.load(Ordering::Relaxed) as f64 / 1e6;
        let total = (asm + front).max(1.0);
        eprintln!(
            "[RLA_PROFILE] CPU-ms  assembly {asm:.0} ({:.0}%)  front-LU {front:.0} ({:.0}%)  [panel {panel:.0} | extract {extract:.0}]",
            100.0 * asm / total,
            100.0 * front / total,
        );
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

    // Assign factorization order e (static pivoting → front-local order is just
    // the column order) and the permutation.
    // Two index maps, distinct under row pivoting:
    //   col_pos_of_g[g] = factorization position eliminating COLUMN g
    //                     (→ U column indices, L column indices).
    //   row_pos_of_g[g] = factorization position whose PIVOT ROW is g
    //                     (→ L row indices). Equal to col_pos when no pivoting.
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
    let low_mem = opts.memory == MemoryMode::LowMemory;

    // Emit global L (CSC, columns in ascending e) and U (CSR, rows in ascending
    // e). A supernode's eliminated columns form a contiguous increasing
    // e-range, so iterating nodes then `j` yields columns/rows in order.
    let one = T::one();
    let (mut l_col_ptr, mut l_row_idx, mut l_values) =
        (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
    let (mut u_row_ptr, mut u_col_idx, mut u_values) =
        (Vec::with_capacity(n + 1), Vec::new(), Vec::new());
    l_col_ptr.push(0);
    u_row_ptr.push(0);
    let mut lcol: Vec<(usize, T)> = Vec::new();
    let mut urow: Vec<(usize, T)> = Vec::new();
    for node_opt in node_results.iter_mut() {
        let node = node_opt.as_mut().ok_or_else(|| {
            RslabError::InvalidInput("internal: unfactored supernode".to_string())
        })?;
        let ff = &node.front;
        let nr = ff.nrow;
        for j in 0..ff.nelim {
            // Diagonal position (= column position = pivot-row position).
            let diag_e = col_pos_of_g[node.row_indices[j]];
            // L column (unit lower). Below-diagonal rows are indexed by the
            // *pivot-row* position of the front row physically at position `i`
            // (`rperm[i]`), which differs from its column position under
            // pivoting - this is the crux for a correct unsymmetric factor.
            lcol.clear();
            lcol.push((diag_e, one));
            for i in (j + 1)..nr {
                let v = ff.l[j * nr + i];
                if v != T::zero() {
                    lcol.push((row_pos_of_g[node.row_indices[ff.rperm[i]]], v));
                }
            }
            // Incomplete factorization (ILU): drop sub-threshold fill relative
            // to the column's largest multiplier, keeping the unit diagonal.
            if let Some(tau) = opts.drop_tol {
                let colmax = lcol
                    .iter()
                    .filter(|&&(r, _)| r != diag_e)
                    .map(|&(_, v)| v.magnitude())
                    .fold(0.0, f64::max);
                let thr = tau * colmax;
                lcol.retain(|&(r, v)| r == diag_e || v.magnitude() >= thr);
            }
            lcol.sort_unstable_by_key(|&(r, _)| r);
            for &(r, v) in &lcol {
                l_row_idx.push(r);
                l_values.push(v);
            }
            l_col_ptr.push(l_row_idx.len());
            // U row (upper, diagonal carries the pivot). Columns are not
            // interchanged → indexed by column position.
            urow.clear();
            urow.push((diag_e, ff.u[j * nr + j]));
            for i in (j + 1)..nr {
                let v = ff.u[j * nr + i];
                if v != T::zero() {
                    urow.push((col_pos_of_g[node.row_indices[i]], v));
                }
            }
            if let Some(tau) = opts.drop_tol {
                let rowmax = urow
                    .iter()
                    .filter(|&&(cc, _)| cc != diag_e)
                    .map(|&(_, v)| v.magnitude())
                    .fold(0.0, f64::max);
                let thr = tau * rowmax;
                urow.retain(|&(cc, v)| cc == diag_e || v.magnitude() >= thr);
            }
            urow.sort_unstable_by_key(|&(c, _)| c);
            for &(c, v) in &urow {
                u_col_idx.push(c);
                u_values.push(v);
            }
            u_row_ptr.push(u_col_idx.len());
        }
        // LowMemory: free this front's dense L/U the moment it is emitted, so
        // the per-front store shrinks as the global structure grows (removes the
        // per-front + global emit-time overlap) instead of holding every front's
        // dense factor until the end.
        if low_mem {
            node.front.l = Vec::new();
            node.front.u = Vec::new();
        }
    }

    Ok(LuFactors {
        n,
        l_col_ptr,
        l_row_idx,
        l_values,
        u_row_ptr,
        u_col_idx,
        u_values,
        perm,
        perm_row,
        d_row,
        d_col,
        n_perturbed,
        solve_threads: solve_policy,
    })
}

/// Solve `A x = b` from an unsymmetric LU factorization (`Pᵀ A P = L U`).
#[allow(clippy::needless_range_loop)] // CSC/CSR solves index col_ptr/row_ptr + scaling
pub fn solve_lu<T: Scalar>(f: &LuFactors<T>, b: &[T]) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // ŷ = P_row · (D_r b): row-equilibrate then row-permute the RHS.
    let mut y: Vec<T> = (0..n)
        .map(|e| {
            let orig = f.perm_row[e];
            b[orig] * T::from_real(f.d_row[orig])
        })
        .collect();
    // Forward solve L y = ŷ (CSC, unit diagonal). Column-oriented: once y[e] is
    // final, eliminate it from the rows below.
    for e in 0..n {
        let ye = y[e];
        for k in f.l_col_ptr[e]..f.l_col_ptr[e + 1] {
            let i = f.l_row_idx[k];
            if i != e {
                y[i] = y[i] - f.l_values[k] * ye;
            }
        }
    }
    // Backward solve U x = y (CSR by row).
    let mut x = vec![T::zero(); n];
    for e in (0..n).rev() {
        let mut acc = y[e];
        let mut diag = T::one();
        for k in f.u_row_ptr[e]..f.u_row_ptr[e + 1] {
            let c = f.u_col_idx[k];
            if c == e {
                diag = f.u_values[k];
            } else {
                acc = acc - f.u_values[k] * x[c];
            }
        }
        x[e] = acc * diag.recip();
    }
    // Undo the column permutation and apply the column equilibration:
    // x_orig[perm[e]] = D_c[perm[e]] · x̂[e].
    let mut out = vec![T::zero(); n];
    for e in 0..n {
        let orig = f.perm[e];
        out[orig] = x[e] * T::from_real(f.d_col[orig]);
    }
    Ok(out)
}

/// Solve `A · X = B` for `nrhs` right-hand sides at once. `b` and the returned
/// `x` are **row-major** `n × nrhs` buffers (`b[i*nrhs + c]` is RHS `c` at row
/// `i`). The `L`/`U` structure is traversed once and each value applied to all
/// `nrhs` columns - faster than `nrhs` separate [`solve_lu`] calls.
/// Below this RHS count / work size the LU block solve runs serially (the
/// parallel gather/scatter overhead only amortizes for wide multi-RHS).
const PAR_SOLVE_MIN_RHS: usize = 8;
const PAR_SOLVE_MIN_WORK: usize = 1 << 18;

/// Solve `A X = B` for `nrhs` right-hand sides. Row-major `n × nrhs` layout, as
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
    // Ŷ = P_row · (D_r B): row-equilibrate then row-permute each RHS block.
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
    // Forward solve L Y = Ŷ (CSC, unit diagonal). `y[e]` (the column's source row)
    // is read by every nonzero of column `e` and is not written in this sweep.
    for e in 0..n {
        let eb = e * nrhs;
        row.copy_from_slice(&y[eb..eb + nrhs]);
        for k in f.l_col_ptr[e]..f.l_col_ptr[e + 1] {
            let i = f.l_row_idx[k];
            if i != e {
                let lval = f.l_values[k];
                let ib = i * nrhs;
                let tgt = &mut y[ib..ib + nrhs];
                for c in 0..nrhs {
                    tgt[c] = tgt[c] - lval * row[c];
                }
            }
        }
    }
    // Backward solve U X = Y (CSR by row), in place in `y`. Accumulate row `e`'s
    // update in the local buffer (the off-diagonal sources `y[c_col]`, `c_col > e`,
    // are already solved and not touched here), then scale and write it back.
    for e in (0..n).rev() {
        let eb = e * nrhs;
        row.copy_from_slice(&y[eb..eb + nrhs]);
        let mut diag = T::one();
        for k in f.u_row_ptr[e]..f.u_row_ptr[e + 1] {
            let c_col = f.u_col_idx[k];
            let uval = f.u_values[k];
            if c_col == e {
                diag = uval;
            } else {
                let cb = c_col * nrhs;
                let src = &y[cb..cb + nrhs];
                for c in 0..nrhs {
                    row[c] = row[c] - uval * src[c];
                }
            }
        }
        let dinv = diag.recip();
        for c in 0..nrhs {
            y[eb + c] = row[c] * dinv;
        }
    }
    // Undo column permutation + column equilibration: out[perm[e]] = D_c · x̂[e].
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
/// Each step computes the residual `r = b − A x` and applies the correction
/// `x ← x + (LU)⁻¹ r`, stopping once `‖r‖∞` stops improving or `max_iter` is
/// reached. This recovers the accuracy a static / within-block-pivoted factor
/// loses on ill-conditioned matrices, at the cost of a few extra solves.
pub fn solve_lu_refined<T: Scalar>(
    f: &LuFactors<T>,
    a: &GeneralCsc<T>,
    b: &[T],
    max_iter: usize,
) -> Result<Vec<T>, RslabError> {
    let n = f.n;
    if a.n != n || b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    let mut x = solve_lu(f, b)?;
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
        let dx = solve_lu(f, &r)?;
        for (xi, &d) in x.iter_mut().zip(&dx) {
            *xi = *xi + d;
        }
    }
    Ok(best_x)
}

#[cfg(test)]
mod tests {
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
        // Small diagonal, large off-diagonals → partial pivoting fires on
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
        // Small diagonal, large off-diagonals → restricted partial pivoting must
        // fire on (nearly) every column. The left-looking path (1×1 static) would
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
                vv.push(c(0.05, 0.01)); // tiny diagonal → threshold pivoting fires
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
        // every `u ∈ [0, 1]` must solve to a tiny residual (the knob changes the
        // factor path but not correctness here). Verifies the field is threaded
        // end-to-end (SolverSettings → KernelTuning → kernel) and clamps.
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
                vv.push(c(12.0, 1.0)); // dominant diagonal → no interchange needed
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
            let f = factor_general_lu_numeric(&analysis, &a, &static_opts).unwrap();
            let x = solve_lu_refined(&f, &a, &b, 2).unwrap();
            let mut ax = vec![Complex::new(0.0, 0.0); n];
            a.matvec(&x, &mut ax);
            let res = (0..n).map(|i| (ax[i] - b[i]).norm()).fold(0.0, f64::max);
            assert!(res < 1e-9, "static reuse shift={shift} residual {res}");
        }
    }

    #[test]
    fn complex_unsymmetric_2d_grid() {
        // 2D 5-point grid with unsymmetric neighbor couplings (right ≠ left).
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
        .unwrap();
        let low = factor_general_lu_numeric(
            &sym,
            &a,
            &SolverSettings::default().with_memory(MemoryMode::LowMemory),
        )
        .unwrap();
        assert_eq!(eager.l_values, low.l_values, "L differs under LowMemory");
        assert_eq!(eager.u_values, low.u_values, "U differs under LowMemory");
        assert_eq!(eager.l_row_idx, low.l_row_idx);
        assert_eq!(eager.u_col_idx, low.u_col_idx);
    }

    #[test]
    fn lu_left_looking_matches_multifrontal() {
        // Unsymmetric, diagonally dominant 2D grid (no pivoting needed → the
        // multifrontal's partial pivoting takes the diagonal, matching the
        // left-looking 1×1 path). Same solution and comparable fill.
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
        // Template (values irrelevant) → analyze once.
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
            let phased =
                factor_general_lu_numeric(&analysis, &a, &SolverSettings::default()).unwrap();
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
