//! One supernode of the left-looking LDL^T factorization: assembly of `A`,
//! the updates of its factored descendants (`cmod`), then the panel
//! factorization.

use super::bunch_kaufman::ll_cdiv_emit;
use super::bunch_kaufman::BkPools;
use super::factor::LlEmitLdlt;
use super::gemm::{grow_scratch, lower_tile_gemm};
use crate::numeric::supernodal::{Input, Span};

use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::supernodal::LlSchedule;
use crate::scalar::Scalar;
use crate::symbolic::SymbolicFactorization;
use rayon::prelude::*;
use std::sync::atomic::AtomicUsize;

/// Factor one supernode's panel: assemble `A`, apply every descendant's `cmod`
/// update (BLAS-3 with scalar fallback), then `cdiv` (partial 1x1 LDL^T). Reads
/// only already-factored descendant panels and their `D` from the emit state,
/// so sibling subtrees run concurrently. Writes the factored panel and `D`
/// there.
#[allow(clippy::too_many_arguments)]
pub(super) fn ll_factor_node<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    inp: Input<T>,
    sched: &LlSchedule,
    pools: &BkPools<T>,
    emit: &LlEmitLdlt<T>,
    perturb_floor: Option<f64>,
    n_perturbed: &AtomicUsize,
    kt: KernelTuning,
) -> Result<(), RslabError> {
    kt.interrupted()?;
    let ll_gemm_gate = kt.k.scalar_gate;
    let ll_gemm_par = kt.k.par_gemm;
    let snode = &sym.supernodes[s];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let nrow = sched.rows(s).len();
    let n = sym.n;
    // SAFETY: this task owns supernode `s`; nobody reads its slot before the
    // node is done (its ancestors start after it).
    let panel: &mut [T] = unsafe { emit.arena.slot_mut(s) };
    debug_assert_eq!(panel.len(), nrow * ncol);

    // Global-to-local rows (narrow entries halve the table's random-access
    // footprint); restored when the node returns.
    let gloc = crate::numeric::supernodal::Gloc::new(n, sched.rows(s));
    // Assemble A's lower-triangle columns of this supernode.
    for p in 0..ncol {
        let c = first + p;
        for (g, v) in inp.col(c) {
            let li = gloc[g] as usize;
            panel[li + p * nrow] = panel[li + p * nrow] + v;
        }
    }
    let plan = crate::numeric::supernodal::CmodPlan::new(
        sym,
        s,
        sched.updaters(s),
        |k| (sched.rows(k), sched.rows(k)),
        false,
        ll_gemm_par,
        kt.k.fork_min_flops,
    );
    let (spans, tile_w, tiled) = (&plan.spans, plan.tile_w, plan.tiled);
    let seq_gemm_par = if plan.forks { ll_gemm_par } else { usize::MAX };
    if tiled {
        let gloc_ref = &gloc;
        let spans_ref = spans;
        panel
            .par_chunks_mut(nrow * tile_w)
            .enumerate()
            .for_each(|(ti, tile)| {
                let c0 = ti * tile_w;
                let c1 = (c0 + tile_w).min(ncol);
                let mut bufs = pools.cmod.take();
                let (_, vd_buf, u_buf) = &mut *bufs;
                for &Span {
                    k: kk, l: (p0, p1), ..
                } in spans_ref
                {
                    let nck = sym.supernodes[kk].ncol;
                    let nrk = sched.rows(kk).len();
                    let ok = &sched.rows(kk)[nck..];
                    let nok = ok.len();
                    // Updater columns landing in this slab.
                    let q0 = p0 + ok[p0..p1].partition_point(|&g| (g as usize) < first + c0);
                    let q1 = p0 + ok[p0..p1].partition_point(|&g| (g as usize) < first + c1);
                    let npk = q1 - q0;
                    if npk == 0 {
                        continue;
                    }
                    // SAFETY: `kk` is a factored descendant of `s`, its cells
                    // are written and never mutated again.
                    let pk: &[T] = unsafe { emit.arena.slot(kk) };
                    let (dk, dsub_k, two_k) = unsafe { emit.d_of(kk, nck) };
                    // G = (kk's block rows q0..q1) * D, column-major npk x nck.
                    grow_scratch(vd_buf, npk * nck);
                    let mut ck = 0;
                    while ck < nck {
                        if two_k[ck] {
                            let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                            for i in 0..npk {
                                let a = pk[(nck + q0 + i) + ck * nrk];
                                let b = pk[(nck + q0 + i) + (ck + 1) * nrk];
                                vd_buf[i + ck * npk] = d11 * a + d21 * b;
                                vd_buf[i + (ck + 1) * npk] = d21 * a + d22 * b;
                            }
                            ck += 2;
                        } else {
                            let dkc = dk[ck];
                            for i in 0..npk {
                                vd_buf[i + ck * npk] = pk[(nck + q0 + i) + ck * nrk] * dkc;
                            }
                            ck += 1;
                        }
                    }
                    let mrows = nok - q0;
                    grow_scratch(u_buf, mrows * npk);
                    // Serial per slab - the parallelism is across slabs.
                    // SAFETY: lhs (read), rhs (read), dst (write) pairwise
                    // disjoint; strides in bounds.
                    unsafe {
                        lower_tile_gemm(
                            &mut *u_buf,
                            mrows,
                            npk,
                            nck,
                            pk.as_ptr().add(nck + q0),
                            nrk as isize,
                            vd_buf.as_ptr(),
                            npk as isize,
                            usize::MAX,
                            &kt.k,
                        )
                    };
                    for c in 0..npk {
                        let tcol = ok[q0 + c] as usize - first;
                        let ucol = &u_buf[c * mrows..c * mrows + mrows];
                        let dst_col = &mut tile[(tcol - c0) * nrow..(tcol - c0 + 1) * nrow];
                        for r in (q0 + c)..nok {
                            let dst = gloc_ref[ok[r] as usize] as usize;
                            dst_col[dst] = dst_col[dst] - ucol[r - q0];
                        }
                    }
                }
            });
    }

    // Sequential per-update cmod (small nodes / small total update work).
    let mut bufs = pools.cmod.take();
    let (vc, vd_buf, u_buf) = &mut *bufs;
    for &Span {
        k: kk, l: (p0, p1), ..
    } in spans.iter().filter(|_| !tiled)
    {
        let nck = sym.supernodes[kk].ncol;
        let nrk = sched.rows(kk).len();
        let ok = &sched.rows(kk)[nck..];
        let nok = ok.len();
        // SAFETY: `kk` is a factored descendant of `s` (its update reaches `s`),
        // so its panel/dval cells are written and never mutated again.
        let pk: &[T] = unsafe { emit.arena.slot(kk) };
        let (dk, dsub_k, two_k) = unsafe { emit.d_of(kk, nck) };
        // Bunch-Kaufman block structure of `kk`'s D (pivoted column order). The
        // cmod `L*D*L^T` is invariant under `kk`'s internal column permutation, so
        // only the block-diagonal `D`-apply has to honor the 2x2 blocks.
        let npk = p1 - p0;
        // Gate on the REAL work (rows >= p0); the scalar path already
        // iterates from the target block, so small tails route there.
        if (nok - p0) * npk * nck < ll_gemm_gate {
            grow_scratch(vc, nck);
            for c_idx in p0..p1 {
                let tcol = ok[c_idx] as usize - first;
                // vc = D * (column `c_idx` of kk's off-diagonal block), with D
                // block-diagonal (1x1 and complex-symmetric 2x2 blocks).
                let mut ck = 0;
                while ck < nck {
                    let a = pk[(nck + c_idx) + ck * nrk];
                    if two_k[ck] {
                        let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                        let b = pk[(nck + c_idx) + (ck + 1) * nrk];
                        vc[ck] = d11 * a + d21 * b;
                        vc[ck + 1] = d21 * a + d22 * b;
                        ck += 2;
                    } else {
                        vc[ck] = dk[ck] * a;
                        ck += 1;
                    }
                }
                for r_idx in c_idx..nok {
                    let trow = gloc[ok[r_idx] as usize] as usize;
                    let mut acc = T::zero();
                    for ck in 0..nck {
                        acc = acc + pk[(nck + r_idx) + ck * nrk] * vc[ck];
                    }
                    panel[trow + tcol * nrow] = panel[trow + tcol * nrow] - acc;
                }
            }
        } else {
            grow_scratch(vd_buf, npk * nck);
            // G = (kk's in-panel off-diagonal block) * D, stored column-major as
            // `vd_buf[c + ck*npk]`. D is block-diagonal (1x1 and 2x2 blocks); a
            // 2x2 block mixes its two columns. GEMM below is unchanged.
            let mut ck = 0;
            while ck < nck {
                if two_k[ck] {
                    let (d11, d21, d22) = (dk[ck], dsub_k[ck], dk[ck + 1]);
                    for i in 0..npk {
                        let a = pk[(nck + p0 + i) + ck * nrk];
                        let b = pk[(nck + p0 + i) + (ck + 1) * nrk];
                        vd_buf[i + ck * npk] = d11 * a + d21 * b;
                        vd_buf[i + (ck + 1) * npk] = d21 * a + d22 * b;
                    }
                    ck += 2;
                } else {
                    let dkc = dk[ck];
                    for i in 0..npk {
                        vd_buf[i + ck * npk] = pk[(nck + p0 + i) + ck * nrk] * dkc;
                    }
                    ck += 1;
                }
            }
            // Only rows >= p0 land in (or below) the target block: computing
            // the full `nok`-tall product and discarding rows `< p0` in the
            // write-back wasted `p0*npk*nck` flops per update - large for
            // updates into high supernodes, where most of the updater's
            // off-diagonal rows lie above the target. Mirror the LU twin:
            // offset the lhs by `p0` and compute `mrows = nok - p0` rows.
            // The write-back below also reads only rows `>= c` per column
            // (the symmetric lower part), so the product is computed
            // tile-wise from each tile's diagonal downward - the same
            // `lower_tile_gemm` that serves the panel Schur updates. For
            // updates into the topmost supernodes (`mrows ~ npk`) the full
            // rectangle wasted another ~half of the flops.
            let mrows = nok - p0;
            grow_scratch(u_buf, mrows * npk);
            // SAFETY: lhs (`pk` off-diag block from row p0, read), rhs
            // (`vd_buf`, read), dst (`u_buf`, write) are pairwise-disjoint;
            // strides in bounds.
            unsafe {
                lower_tile_gemm(
                    &mut *u_buf,
                    mrows,
                    npk,
                    nck,
                    pk.as_ptr().add(nck + p0),
                    nrk as isize,
                    vd_buf.as_ptr(),
                    npk as isize,
                    seq_gemm_par,
                    &kt.k,
                )
            };
            for c in 0..npk {
                let tcol = ok[p0 + c] as usize - first;
                let ucol = &u_buf[c * mrows..c * mrows + mrows];
                for r in (p0 + c)..nok {
                    let dst = gloc[ok[r] as usize] as usize + tcol * nrow;
                    panel[dst] = panel[dst] - ucol[r - p0];
                }
            }
        }
    }
    ll_cdiv_emit(
        s,
        sym,
        sched,
        pools,
        emit,
        perturb_floor,
        n_perturbed,
        kt,
        panel,
    )
}

/// Heap bytes [`ll_factor_node`] allocates for supernode `s` while it runs,
/// for the memory plan: the tiled cmod's per-slab buffers on up to
/// `workers` threads at once, or the sequential cmod's buffers, which stay
/// allocated through the cdiv ([`ll_cdiv_scratch`](super::bunch_kaufman::ll_cdiv_scratch));
/// the largest split planes its GEMMs take on a worker (complex fields)
/// and how many workers may take them at once. Replays the kernel's plan
/// and growth of its buffers.
pub(super) fn ll_node_scratch<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    k: &crate::KernelSettings,
    workers: usize,
) -> (u64, usize, usize) {
    use super::gemm::lower_tile_planes;
    use crate::memory::grown;
    let vb = std::mem::size_of::<T>();
    let (first, ncol) = (sym.supernodes[s].first_col, sym.supernodes[s].ncol);
    let (cdiv, mut planes) =
        super::bunch_kaufman::ll_cdiv_scratch::<T>(ncol, sched.rows(s).len(), k);
    let plan = crate::numeric::supernodal::CmodPlan::new(
        sym,
        s,
        sched.updaters(s),
        |k| (sched.rows(k), sched.rows(k)),
        false,
        k.par_gemm,
        k.fork_min_flops,
    );
    let updater = |kk: usize| {
        let nck = sym.supernodes[kk].ncol;
        (nck, &sched.rows(kk)[nck..])
    };
    if plan.tiled {
        let tw = plan.tile_w;
        let slabs = ncol.div_ceil(tw);
        let (mut worst, mut slab_planes) = (0, 0);
        for ti in 0..slabs {
            let (c0, c1) = (ti * tw, (ti * tw + tw).min(ncol));
            let (mut vd, mut u) = (0, 0);
            for sp in &plan.spans {
                let (nck, ok) = updater(sp.k);
                let (p0, p1) = sp.l;
                let q0 = p0 + ok[p0..p1].partition_point(|&g| (g as usize) < first + c0);
                let q1 = p0 + ok[p0..p1].partition_point(|&g| (g as usize) < first + c1);
                if q1 > q0 {
                    vd = grown(vd, (q1 - q0) * nck);
                    u = grown(u, (ok.len() - q0) * (q1 - q0));
                    if T::COMPLEX {
                        let p = lower_tile_planes(ok.len() - q0, q1 - q0, nck, usize::MAX, k);
                        slab_planes = slab_planes.max(p);
                    }
                }
            }
            worst = worst.max(vd + u);
        }
        let copies = if slab_planes > planes {
            slabs.min(workers.max(1))
        } else {
            1
        };
        let bytes = ((worst * slabs.min(workers.max(1)) * vb) as u64).max(cdiv);
        (bytes, planes.max(slab_planes), copies)
    } else {
        let (mut vc, mut vd, mut u) = (0, 0, 0);
        for sp in &plan.spans {
            let (nck, ok) = updater(sp.k);
            let (p0, p1) = sp.l;
            let npk = p1 - p0;
            if (ok.len() - p0) * npk * nck < k.scalar_gate {
                vc = grown(vc, nck);
            } else {
                vd = grown(vd, npk * nck);
                u = grown(u, (ok.len() - p0) * npk);
                if T::COMPLEX {
                    let par = if plan.forks { k.par_gemm } else { usize::MAX };
                    planes = planes.max(lower_tile_planes(ok.len() - p0, npk, nck, par, k));
                }
            }
        }
        (((vc + vd + u) * vb) as u64 + cdiv, planes, 1)
    }
}
