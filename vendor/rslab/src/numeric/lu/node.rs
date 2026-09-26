//! One supernode of the left-looking LU factorization: assembly, the updates
//! of its factored descendants (`cmod`, into both `L` and `U12`), then the
//! blocked panel LU with threshold partial pivoting.

use super::factor::{LlEmit, LuLlStore};
use super::structure::LuStructure;
use crate::numeric::supernodal::Input;

use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::supernodal::perturb_pivot;
use crate::numeric::supernodal::{Li, LlSchedule, PanelPtr};
use crate::scalar::Scalar;
use crate::symbolic::SymbolicFactorization;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

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

#[allow(clippy::too_many_arguments)]
pub(super) fn lu_ll_factor_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    inp: Input<T>,
    sched: &LlSchedule,
    st: &LuStructure,
    store: &LuLlStore,
    emit: &LlEmit<T>,
    perturb_floor: Option<f64>,
    n_perturbed: &AtomicUsize,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    kt.interrupted()?;
    let ll_gemm_gate = kt.k.scalar_gate;
    let ll_gemm_par = kt.k.par_gemm;
    let snode = &sym.supernodes[s];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let (rows_l, cols_u) = (st.rows_l(s), st.cols_u(s));
    let (nrow_l, nrow_u) = (rows_l.len(), cols_u.len());
    let (cn_l, cn_u) = (nrow_l - ncol, nrow_u - ncol);
    let n = sym.n;
    // `lbuf`: the `L` panel, `nrow_l x ncol` (the diagonal block holds `L11`
    // below and `U11` on and above the diagonal). `ut`: the `U^T` panel,
    // `nrow_u x ncol`; the factorization writes its off-block rows, `U12[p, t]`
    // at `ut[p * nrow_u + ncol + t]` (row `p` of `U` is column `p` of the
    // panel), and the emit fills the diagonal block.
    // SAFETY: this task owns supernode `s`; nobody reads the slots before they
    // are published by `store.set` at the end of the node.
    let lbuf: &mut [T] = unsafe { emit.l_arena.slot_mut(s) };
    let ut: &mut [T] = unsafe { emit.u_arena.slot_mut(s) };
    debug_assert_eq!(lbuf.len(), nrow_l * ncol);
    debug_assert_eq!(ut.len(), nrow_u * ncol);

    // Global to local: `L` rows into `lbuf`, `U` columns into `ut`.
    let gloc = crate::numeric::supernodal::Gloc::new(n, rows_l);
    let gloc_u = crate::numeric::supernodal::Gloc::second(n, cols_u);
    // Assemble columns of s (full) into lbuf, and the U12 rows into ut. The
    // exact structure holds every entry of `B`.
    for p in 0..ncol {
        let c = first + p;
        for (g, v) in inp.col(c) {
            debug_assert!(gloc[g] != Li::MAX, "entry outside the L structure");
            let li = gloc[g] as usize;
            lbuf[p * nrow_l + li] = lbuf[p * nrow_l + li] + v;
        }
        for (g, v) in inp.row(c) {
            debug_assert!(gloc_u[g] != Li::MAX, "entry outside the U structure");
            let lc = gloc_u[g] as usize;
            ut[p * nrow_u + lc] = ut[p * nrow_u + lc] + v;
        }
    }
    // cmod from every factored descendant. NOTE: cmod-aggregation (K-stacking many
    // descendant updates into one fat GEMM) was measured and rejected - across MoM
    // topologies 91-95 % of cmod flop already runs as large parallel GEMMs, and the
    // only aggregation reaching those dominant updates carries an 11-15x zero-pad
    // blowup (each top-of-tree descendant touches a small, distinct row/col subset
    // of the large target). The `RLA_CMOD_DIST` histogram below documents this.
    let plan = crate::numeric::supernodal::CmodPlan::new(
        sym,
        s,
        sched.updaters(s),
        |k| (st.rows_l(k), st.cols_u(k)),
        true,
        ll_gemm_par,
        kt.k.fork_min_flops,
    );
    let (spans, forks, tile_w, tiled) = (&plan.spans, plan.forks, plan.tile_w, plan.tiled);
    let tile_u = (cn_u.max(1) / 16).clamp(32, 256);
    // Updater `k`'s column count, off-block `L` rows and `U` columns, panels
    // and their heights.
    let updater = |k: usize| {
        let nck = sym.supernodes[k].ncol;
        let (rl, cu) = (st.rows_l(k), st.cols_u(k));
        // SAFETY: `k` is a factored descendant of `s`; its slots are published
        // and never written again.
        let (lk, uk): (&[T], &[T]) = unsafe { (emit.l_arena.slot(k), emit.u_arena.slot(k)) };
        (nck, &rl[nck..], &cu[nck..], lk, uk, rl.len(), cu.len())
    };

    // Column-tiled parallel cmod: disjoint `&mut` slabs of the target
    // buffers; per slab every updater's contribution in updater order with a
    // serial GEMM (see `CmodPlan::tiled`). The LU node has two target buffers,
    // so the tiling runs as two phases: `lbuf` slabs (the `L`/`U11` updates),
    // then runs of `U12` columns in `ut`.
    if tiled {
        let (gloc_ref, gloc_u_ref) = (&gloc, &gloc_u);
        lbuf.par_chunks_mut(nrow_l * tile_w)
            .enumerate()
            .for_each(|(ti, slab)| {
                let c0 = ti * tile_w;
                let c1 = (c0 + tile_w).min(ncol);
                let mut lupd: Vec<T> = Vec::new();
                for sp in spans {
                    let (nck, ol, ou, lk, uk, nrk_l, nrk_u) = updater(sp.k);
                    let (p0l, (p0u, p1u)) = (sp.l.0, sp.u);
                    let q0 = p0u + ou[p0u..p1u].partition_point(|&g| (g as usize) < first + c0);
                    let q1 = p0u + ou[p0u..p1u].partition_point(|&g| (g as usize) < first + c1);
                    let npk = q1 - q0;
                    let mrows = ol.len() - p0l;
                    if npk == 0 || mrows == 0 {
                        continue;
                    }
                    lupd.clear();
                    lupd.resize(mrows * npk, T::zero());
                    // SAFETY: lhs/rhs/dst pairwise disjoint; strides in bounds.
                    unsafe {
                        crate::dense::gemm_backend::gemm(
                            mrows,
                            npk,
                            nck,
                            lupd.as_mut_ptr(),
                            mrows as isize,
                            1,
                            false,
                            lk.as_ptr().add(nck + p0l),
                            nrk_l as isize,
                            1,
                            uk.as_ptr().add(nck + q0),
                            1,
                            nrk_u as isize,
                            T::zero(),
                            T::one(),
                            false,
                            false,
                            false,
                            crate::dense::gemm_backend::GemmMode::new(
                                gemm::Parallelism::None,
                                &kt.k,
                            ),
                        );
                    }
                    for jj in 0..npk {
                        let cbase = (ou[q0 + jj] as usize - first - c0) * nrow_l;
                        let ucol = &lupd[jj * mrows..jj * mrows + mrows];
                        for i in 0..mrows {
                            let dst = cbase + gloc_ref[ol[p0l + i] as usize] as usize;
                            slab[dst] = slab[dst] - ucol[i];
                        }
                    }
                }
            });
        if cn_u > 0 {
            // A slab is the run `[u0, u1)` of U12's columns: rows `ncol + u0..ncol + u1`
            // of every `ut` column, disjoint between slabs.
            let up = PanelPtr(ut.as_mut_ptr());
            (0..cn_u.div_ceil(tile_u)).into_par_iter().for_each(|ti| {
                let u0 = ti * tile_u;
                let u1 = (u0 + tile_u).min(cn_u);
                let g0 = cols_u[ncol + u0];
                let g1 = if ncol + u1 < nrow_u {
                    cols_u[ncol + u1]
                } else {
                    Li::MAX
                };
                let mut uupd: Vec<T> = Vec::new();
                for sp in spans {
                    let (nck, ol, ou, lk, uk, nrk_l, nrk_u) = updater(sp.k);
                    let ((p0l, p1l), p1u) = (sp.l, sp.u.1);
                    let t0 = p1u + ou[p1u..].partition_point(|&g| g < g0);
                    let t1 = p1u + ou[p1u..].partition_point(|&g| g < g1);
                    let (ntr, npk) = (t1 - t0, p1l - p0l);
                    if ntr == 0 || npk == 0 {
                        continue;
                    }
                    uupd.clear();
                    uupd.resize(npk * ntr, T::zero());
                    // SAFETY: lhs/rhs/dst pairwise disjoint; strides in bounds.
                    unsafe {
                        crate::dense::gemm_backend::gemm(
                            npk,
                            ntr,
                            nck,
                            uupd.as_mut_ptr(),
                            npk as isize,
                            1,
                            false,
                            lk.as_ptr().add(nck + p0l),
                            nrk_l as isize,
                            1,
                            uk.as_ptr().add(nck + t0),
                            1,
                            nrk_u as isize,
                            T::zero(),
                            T::one(),
                            false,
                            false,
                            false,
                            crate::dense::gemm_backend::GemmMode::new(
                                gemm::Parallelism::None,
                                &kt.k,
                            ),
                        );
                    }
                    for jj in 0..ntr {
                        let lt = gloc_u_ref[ou[t0 + jj] as usize] as usize;
                        let ucol = &uupd[jj * npk..jj * npk + npk];
                        for i in 0..npk {
                            // SAFETY: row `lt` lies in this slab's run.
                            unsafe {
                                let d = up.get().add((ol[p0l + i] as usize - first) * nrow_u + lt);
                                *d = *d - ucol[i];
                            }
                        }
                    }
                }
            });
        }
    }

    // Sequential per-update cmod (small nodes / narrow panels).
    let mut lupd: Vec<T> = Vec::new();
    let mut uupd: Vec<T> = Vec::new();
    for sp in spans.iter().filter(|_| !tiled) {
        let (nck, ol, ou, lk, uk, nrk_l, nrk_u) = updater(sp.k);
        let ((p0l, p1l), (p0u, p1u)) = (sp.l, sp.u);
        // L update: rows `ol[p0l..]` times the `U` columns landing here;
        // U12 update: the `L` rows landing here times the `U` columns past here.
        let (mrows, npk_u) = (ol.len() - p0l, p1u - p0u);
        let (npk_l, ntrail) = (p1l - p0l, ou.len() - p1u);
        if (mrows * npk_u + npk_l * ntrail) * nck < ll_gemm_gate {
            // Scalar path.
            for jj in 0..npk_u {
                let tcol = ou[p0u + jj] as usize - first;
                for i in 0..mrows {
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc
                            + lk[(nck + p0l + i) + ck * nrk_l] * uk[ck * nrk_u + nck + p0u + jj];
                    }
                    let trow = gloc[ol[p0l + i] as usize] as usize;
                    lbuf[tcol * nrow_l + trow] = lbuf[tcol * nrow_l + trow] - acc;
                }
            }
            for jj in 0..ntrail {
                let tu = gloc_u[ou[p1u + jj] as usize] as usize;
                for i in 0..npk_l {
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc
                            + lk[(nck + p0l + i) + ck * nrk_l] * uk[ck * nrk_u + nck + p1u + jj];
                    }
                    let urow = ol[p0l + i] as usize - first;
                    ut[urow * nrow_u + tu] = ut[urow * nrow_u + tu] - acc;
                }
            }
            continue;
        }
        // `forks` folds in the join-steal guard: a small node never forks here.
        let par = |work: usize| {
            if forks && work >= ll_gemm_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            }
        };
        if mrows > 0 && npk_u > 0 {
            // Lupd(mrows x npk_u) = L_k[rows from p0l, :] * U_k[:, landing columns].
            lupd.clear();
            lupd.resize(mrows * npk_u, T::zero());
            // SAFETY: lhs (lk off-diag rows), rhs (uk landing columns), dst
            // (lupd) are disjoint; strides in bounds.
            unsafe {
                crate::dense::gemm_backend::gemm(
                    mrows,
                    npk_u,
                    nck,
                    lupd.as_mut_ptr(),
                    mrows as isize,
                    1,
                    false,
                    lk.as_ptr().add(nck + p0l),
                    nrk_l as isize,
                    1,
                    uk.as_ptr().add(nck + p0u),
                    1,
                    nrk_u as isize,
                    T::zero(),
                    T::one(),
                    false,
                    false,
                    false,
                    crate::dense::gemm_backend::GemmMode::new(par(mrows * npk_u * nck), &kt.k),
                );
            }
            for jj in 0..npk_u {
                let cbase = (ou[p0u + jj] as usize - first) * nrow_l;
                let ucol = &lupd[jj * mrows..jj * mrows + mrows];
                for i in 0..mrows {
                    let dst = cbase + gloc[ol[p0l + i] as usize] as usize;
                    lbuf[dst] = lbuf[dst] - ucol[i];
                }
            }
        }
        if npk_l > 0 && ntrail > 0 {
            // Uupd(npk_l x ntrail) = L_k[landing rows, :] * U_k[:, columns past s].
            uupd.clear();
            uupd.resize(npk_l * ntrail, T::zero());
            // SAFETY: as above; rhs is the trailing U columns of `uk`.
            unsafe {
                crate::dense::gemm_backend::gemm(
                    npk_l,
                    ntrail,
                    nck,
                    uupd.as_mut_ptr(),
                    npk_l as isize,
                    1,
                    false,
                    lk.as_ptr().add(nck + p0l),
                    nrk_l as isize,
                    1,
                    uk.as_ptr().add(nck + p1u),
                    1,
                    nrk_u as isize,
                    T::zero(),
                    T::one(),
                    false,
                    false,
                    false,
                    crate::dense::gemm_backend::GemmMode::new(par(npk_l * ntrail * nck), &kt.k),
                );
            }
            for jj in 0..ntrail {
                let lt = gloc_u[ou[p1u + jj] as usize] as usize;
                let ucol = &uupd[jj * npk_l..jj * npk_l + npk_l];
                for i in 0..npk_l {
                    let dst = (ol[p0l + i] as usize - first) * nrow_u + lt;
                    ut[dst] = ut[dst] - ucol[i];
                }
            }
        }
    }
    // cdiv: in-place **blocked** panel LU (1x1 static pivoting), no trailing
    // update outside the panel: a getrf - unblocked `getf2` over
    // an NB-wide panel, then the dominant trailing update as a single SIMD GEMM
    // (rank-NB) - but restricted to the panel: the trailing is the remaining
    // panel columns (`lbuf`) plus the `U12` rows (in `ut`), with no `A22`/CB. This
    // routes the `O(ncol^2*nrow_l)` cdiv work (the measured 77 % of the left-looking
    // factor) through BLAS-3 instead of scalar rank-1 sweeps.
    // Panel width. Swept 32/48/64/96 on the MoM fronts: 32 optimal for typical
    // panels - but root-class WIDE panels want a fatter deferred-GEMM inner
    // dimension (k = nb), the same lever as the LDLT twin's adaptive nb. Pure
    // function of `ncol`, never of the thread count.
    let nb_cdiv = if ncol >= 512 { 128 } else { 32 };
    // Join-steal guard (see the cmod fork gate above): a small node must not
    // fork inside its cdiv either.
    let ll_cdiv_par = if nrow_l * ncol * ncol >= 100_000_000 {
        kt.k.par_cdiv
    } else {
        usize::MAX
    };
    let mut local_perturbed = 0usize;
    // Restricted partial pivoting: row interchanges within the fully-summed block
    // `[0, ncol)` only (the standard sparse-direct choice). `rperm[i]` is the
    // row-structure index physically at position `i`; the trailing rows are never
    // interchanged, so the contribution rows `Ok` ancestors pull are unaffected
    // and `cmod` needs no permutation awareness.
    let mut rperm: Vec<usize> = (0..nrow_l).collect();
    // Pivot reciprocals of the current panel, reused by the parallel trailing apply.
    let mut pinv_blk: Vec<T> = vec![T::zero(); nb_cdiv];
    let mut kb = 0;
    while kb < ncol {
        kt.interrupted()?;
        let ke = (kb + nb_cdiv).min(ncol);
        // getf2: factor columns [kb, ke) over the **fully-summed rows [k+1, ncol)**
        // only - the deep trailing rows [ncol, nrow_l) (never pivot candidates) are
        // lifted off this serial path into the parallel apply below.
        for k in kb..ke {
            // **Threshold** partial pivoting (UMFPACK-style): keep the diagonal
            // pivot unless it is below `THRESH` of the largest candidate in the
            // fully-summed block - so a well-scaled/equilibrated matrix never
            // interchanges (no fill or accuracy cost) while small/zero diagonals
            // still get a stable pivot. `THRESH^2` compared on squared magnitudes.
            // `THRESH = kt.pivot_threshold` (tunable, default 0.1); `u = 1` recovers full
            // partial pivoting, `u = 0` keeps the diagonal unless it is exactly zero.
            let thresh_sq = kt.pivot_threshold * kt.pivot_threshold;
            // Static pivoting fast path (`u == 0`): keep the natural pivot order and
            // skip the argmax search entirely - the "skip pivot search" speed lever
            // for fixed-pattern value sequences (solver-in-the-loop: reuse a good
            // order across a frequency sweep / time-stepping). The search result is
            // never consumed when `u == 0` (the threshold test `diag_sq < 0` can
            // never fire), so skipping it is behaviour-identical, only faster. A
            // sub-floor / zero diagonal is still caught below by the pivot policy.
            if thresh_sq > 0.0 {
                let mut p = k;
                let mut best = lbuf[k * nrow_l + k].magnitude_sq();
                for i in (k + 1)..ncol {
                    let m = lbuf[k * nrow_l + i].magnitude_sq();
                    if m > best {
                        best = m;
                        p = i;
                    }
                }
                let diag_sq = lbuf[k * nrow_l + k].magnitude_sq();
                if p != k && diag_sq < thresh_sq * best {
                    for c in 0..ncol {
                        lbuf.swap(c * nrow_l + k, c * nrow_l + p);
                    }
                    for t in ncol..nrow_u {
                        ut.swap(k * nrow_u + t, p * nrow_u + t);
                    }
                    rperm.swap(k, p);
                }
            }
            let mut piv = lbuf[k * nrow_l + k];
            match perturb_floor {
                Some(floor) if piv.magnitude() < floor => {
                    piv = perturb_pivot(piv, floor);
                    local_perturbed += 1;
                }
                None if piv == T::zero() => {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                _ => {}
            }
            lbuf[k * nrow_l + k] = piv;
            let pinv = piv.recip();
            pinv_blk[k - kb] = pinv;
            for i in (k + 1)..ncol {
                lbuf[k * nrow_l + i] = lbuf[k * nrow_l + i] * pinv;
            }
            for j in (k + 1)..ke {
                let u_kj = lbuf[j * nrow_l + k];
                if u_kj != T::zero() {
                    for i in (k + 1)..ncol {
                        lbuf[j * nrow_l + i] = lbuf[j * nrow_l + i] - lbuf[k * nrow_l + i] * u_kj;
                    }
                }
            }
        }
        let pw = ke - kb;
        // Trailing-row L21 panel [ncol, nrow_l): apply the just-computed panel
        // transform (scale by `pinv_blk`, within-panel rank-1 against `U11`) to the
        // deep rows, parallel over **disjoint** row chunks. Bit-identical to the
        // full-height getf2 - same per-row op sequence - but the dominant `cn_l`
        // work now runs on all idle workers instead of the serial panel path.
        if cn_l > 0 {
            let par = cn_l * pw * pw >= ll_cdiv_par;
            if par {
                let pp = PanelPtr(lbuf.as_mut_ptr());
                let nthreads = rayon::current_num_threads().max(1);
                let cs = cn_l.div_ceil(nthreads).max(1);
                let ranges: Vec<(usize, usize)> = (0..nthreads)
                    .map(|c| {
                        let r0 = ncol + c * cs;
                        (r0.min(nrow_l), (r0 + cs).min(nrow_l))
                    })
                    .filter(|(a, b)| a < b)
                    .collect();
                // Capture the whole `pp` (Send+Sync) - destructure inside so Rust
                // does not disjoint-capture the bare `*mut T`.
                ranges.par_iter().for_each(|&(r0, r1)| {
                    // SAFETY: disjoint row chunk; see `apply_panel_trailing`.
                    unsafe { apply_panel_trailing(pp.get(), nrow_l, kb, pw, &pinv_blk, r0, r1) };
                });
            } else {
                // SAFETY: single-threaded over all trailing rows.
                unsafe {
                    apply_panel_trailing(lbuf.as_mut_ptr(), nrow_l, kb, pw, &pinv_blk, ncol, nrow_l)
                };
            }
        }
        // TRSM: U = L11^-1 * (trailing panel columns of lbuf) and the U12 rows.
        // Each trailing column is an independent forward substitution reading
        // only the finished panel columns [kb, ke), so the block parallelizes
        // over disjoint column chunks - bit-identical per-column op order.
        // Profiled at 22% of cdiv CPU when serial (MoM fronts).
        if (ncol - ke) * pw * pw >= ll_cdiv_par {
            let (head, tail) = lbuf.split_at_mut(ke * nrow_l);
            tail.par_chunks_mut(nrow_l).for_each(|col| {
                for r in (kb + 1)..ke {
                    let mut acc = col[r];
                    for i in kb..r {
                        acc = acc - head[i * nrow_l + r] * col[i];
                    }
                    col[r] = acc;
                }
            });
        } else {
            for j in ke..ncol {
                for r in (kb + 1)..ke {
                    let mut acc = lbuf[j * nrow_l + r];
                    for i in kb..r {
                        acc = acc - lbuf[i * nrow_l + r] * lbuf[j * nrow_l + i];
                    }
                    lbuf[j * nrow_l + r] = acc;
                }
            }
        }
        // U12 rows (the `cn_u` contribution columns of U): the forward substitution
        // over the panel rows, `x_r -= L[r, i] x_i` for `i` ascending, on whole rows
        // of U12 (the contiguous runs `ut[r * nrow_u + ncol..(r + 1) * nrow_u]`), so every
        // entry sees the operations of its own column's substitution in order;
        // parallel over disjoint runs of the columns.
        let trsm_u = |t0: usize, t1: usize, u: PanelPtr<T>, lref: &[T]| {
            for r in (kb + 1)..ke {
                for i in kb..r {
                    let l = lref[i * nrow_l + r];
                    // SAFETY: rows `r != i` of `ut`, the caller's columns `[t0, t1)`.
                    unsafe {
                        let (xr, xi) = (u.get().add(r * nrow_u), u.get().add(i * nrow_u));
                        for t in t0..t1 {
                            *xr.add(t) = *xr.add(t) - l * *xi.add(t);
                        }
                    }
                }
            }
        };
        let up = PanelPtr(ut.as_mut_ptr());
        if cn_u * pw * pw >= ll_cdiv_par {
            let lref: &[T] = lbuf;
            (ncol..nrow_u)
                .into_par_iter()
                .step_by(256)
                .for_each(|t0| trsm_u(t0, (t0 + 256).min(nrow_u), up, lref));
        } else {
            trsm_u(ncol, nrow_u, up, lbuf);
        }
        // GEMM: lbuf[ke.., ke..ncol] -= L21[ke.., kb..ke] * U[kb..ke, ke..ncol].
        let mt = nrow_l - ke;
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
                crate::dense::gemm_backend::gemm(
                    mt,
                    nt,
                    pw,
                    base.add(ke * nrow_l + ke),
                    nrow_l as isize,
                    1,
                    true,
                    base.add(kb * nrow_l + ke),
                    nrow_l as isize,
                    1,
                    base.add(ke * nrow_l + kb),
                    nrow_l as isize,
                    1,
                    T::one(),
                    T::zero() - T::one(),
                    false,
                    false,
                    false,
                    crate::dense::gemm_backend::GemmMode::new(par, &kt.k),
                );
            }
        }
        // GEMM: U12[ke..ncol, :] -= L[ke..ncol, kb..ke] * U12[kb..ke, :].
        if cn_u > 0 && nt > 0 {
            let par = if (nt * cn_u * pw) >= ll_cdiv_par {
                gemm::Parallelism::Rayon(0)
            } else {
                gemm::Parallelism::None
            };
            let lptr = lbuf.as_ptr();
            let uptr = ut.as_mut_ptr();
            // SAFETY: dst (U12 rows `ke..ncol`, `ut` columns) is disjoint from the
            // read sub-blocks of `lbuf` and U12 (rows `kb..ke`); strides in bounds.
            unsafe {
                crate::dense::gemm_backend::gemm(
                    nt,
                    cn_u,
                    pw,
                    uptr.add(ke * nrow_u + ncol),
                    1,
                    nrow_u as isize,
                    true,
                    lptr.add(kb * nrow_l + ke),
                    nrow_l as isize,
                    1,
                    uptr.add(kb * nrow_u + ncol),
                    1,
                    nrow_u as isize,
                    T::one(),
                    T::zero() - T::one(),
                    false,
                    false,
                    false,
                    crate::dense::gemm_backend::GemmMode::new(par, &kt.k),
                );
            }
        }
        kb = ke;
    }
    if local_perturbed > 0 {
        n_perturbed.fetch_add(local_perturbed, Ordering::Relaxed);
    }
    // Populate the O(n) index maps for `s` from its (final) `rperm` and the
    // symbolic elimination offset - consumed by `emit_and_free` and the assembly.
    // Writes target disjoint global indices; visibility via the subtree join.
    let eoff = emit.e_offset[s];
    for (p, &rp) in rperm[..ncol].iter().enumerate() {
        let g_col = first + p;
        let g_row = rows_l[rp] as usize;
        // SAFETY: each global index is written by exactly one supernode.
        unsafe {
            emit.e_of_g.set(g_col, eoff + p);
            emit.row_pos_of_g.set(g_row, eoff + p);
            emit.perm.set(eoff + p, sym.perm[g_col]);
            emit.perm_row.set(eoff + p, sym.perm[g_row]);
        }
    }
    // SAFETY: this thread owns `s`, writes its cells exactly once.
    unsafe { store.set(s, rperm) };
    Ok(())
}
