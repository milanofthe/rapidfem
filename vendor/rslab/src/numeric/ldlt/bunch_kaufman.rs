//! The dense panel factorization of the LDL^T kernel (`cdiv`): blocked
//! Bunch-Kaufman with 1x1 and 2x2 pivots bounded to a supernode's
//! fully-summed block, the row-parallel replay of each panel's transform on
//! the deep rows, and the deferred trailing update with panel lookahead.

use super::factor::{LdltSlot, LlEmitLdlt, LlStore};
use super::gemm::lower_tile_gemm;
use crate::numeric::supernodal::perturb_pivot;

use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::supernodal::LlSchedule;
use crate::numeric::supernodal::PanelPtr as LdltPanelPtr;
use crate::scalar::Scalar;
use crate::symbolic::SymbolicFactorization;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Scale-invariant singularity floor for a 2x2 Bunch-Kaufman pivot: a block
/// whose `|det|` falls below `GROWTH_EPS * scale^2` (scale = the largest block
/// entry magnitude) is numerically singular - rejected in exact mode and lifted
/// in static-pivot mode. Bounds the element growth `1/|det|` can otherwise
/// inject into the trailing update.
const GROWTH_EPS: f64 = 1e-14;

/// Apply a factored Bunch-Kaufman panel's transform sequence to rows
/// `[r0, r1)` of the column-major `panel` (stride `nrow`), for pivot steps
/// `[kb, ke)`. Bit-identical to the corresponding rows of the full-height
/// panel factorization: per 1x1 step the in-panel updates use the **final**
/// column-`k` multipliers (`w_j*d^-1` is exactly the stored `L(j,k)`), then
/// the column is scaled by `d^-1`; per 2x2 step the multiplier pair is
/// rebuilt from the (already perturbed) stored `D` block with the same
/// expressions and order. Deep rows are never pivot candidates, so each
/// caller's row range is independent - the lever that lifts the dominant
/// `O((nrow-ke)*pw^2)` panel work off the serial getf2 path onto all idle
/// workers (ports the LU twin's `apply_panel_trailing` to Bunch-Kaufman).
///
/// `deep_swaps[k - kb]` records the pivot interchange partner of step `k`
/// (`usize::MAX` when the step did not interchange): getf2 bounds its swaps
/// to the panel rows, so the deep-row segments of each interchange are
/// replayed here, immediately before the step's transform - the original
/// full-height order, row by row.
///
/// `mult_snap` holds the in-panel multipliers **as of each step's time**
/// (`mult_snap[(k - kb)*nb + (j - kb)]` is step `k`'s coefficient for
/// in-panel row `j`). Reading them from the final panel would be wrong:
/// later symmetric interchanges permute the rows of earlier multiplier
/// columns (unlike LU, where produced pivot rows never move again).
///
/// SAFETY: `[r0, r1)` must be this caller's exclusive rows and within the
/// buffer; columns `[kb, ke)` must be in bounds under stride `nrow`.
#[allow(clippy::too_many_arguments)]
unsafe fn apply_bk_panel_trailing<T: Scalar>(
    base: *mut T,
    nrow: usize,
    kb: usize,
    ke: usize,
    d_diag: &[T],
    d_subdiag: &[T],
    two_by_two: &[bool],
    deep_swaps: &[usize],
    mult_snap: &[T],
    nb: usize,
    r0: usize,
    r1: usize,
    sb: usize,
) {
    // Blocked form of the per-pivot sweep. Pivots are taken in sub-blocks of
    // `sb` columns ([`KernelSettings::trailing_block`](crate::KernelSettings::trailing_block);
    // 16 measured best: 32 doubles the scalar within-block work, 8 halves
    // the GEMM efficiency): inside a sub-block a pivot's rank-1 (rank-2)
    // update reaches only the sub-block's remaining columns (scalar loops),
    // while its contribution to the columns beyond the sub-block is deferred
    // and applied as one GEMM `B[:, ke2..ke] -= W * M` per sub-block, where
    // `W` holds the pivot columns before their `D^{-1}` scaling and `M` the
    // multipliers of `mult_snap`. A pivot swap that reaches beyond the
    // sub-block first flushes the pending pivots (the swapped-in column must
    // carry every earlier update, as it does in the sequential sweep). Rows
    // are independent, so any row range gives the same values.
    let deep = r1.saturating_sub(r0);
    if deep == 0 || ke <= kb {
        return;
    }
    // One spare column: a 2x2 pivot that starts on a sub-block's last
    // column extends the sub-block by one, the pair is never split.
    let sb = sb.max(1);
    let mut w: Vec<T> = vec![T::zero(); deep * (sb + 1)];
    let mut kb2 = kb;
    while kb2 < ke {
        let mut ke2 = (kb2 + sb).min(ke);
        if ke2 < ke && two_by_two[ke2 - 1] {
            ke2 += 1;
        }
        // Pending pivots [pend0, k) whose deferred update has not been applied.
        let mut pend0 = kb2;
        let mut k = kb2;
        while k < ke2 {
            let kp = deep_swaps[k - kb];
            if kp != usize::MAX {
                if kp >= ke2 {
                    flush_trailing(
                        base, nrow, kb, ke, kb2, ke2, pend0, k, &w, deep, mult_snap, nb, r0,
                    );
                    pend0 = k;
                }
                let src = if two_by_two[k] { k + 1 } else { k };
                let ca = base.add(src * nrow);
                let cb = base.add(kp * nrow);
                for i in r0..r1 {
                    core::ptr::swap(ca.add(i), cb.add(i));
                }
            }
            if two_by_two[k] {
                let (d11, d21, d22) = (d_diag[k], d_subdiag[k], d_diag[k + 1]);
                let det = d11 * d22 - d21 * d21;
                let detinv = det.recip();
                let colk = base.add(k * nrow);
                let colk1 = base.add((k + 1) * nrow);
                for j in (k + 2)..ke2 {
                    let l1j = mult_snap[(k - kb) * nb + (j - kb)];
                    let l2j = mult_snap[(k + 1 - kb) * nb + (j - kb)];
                    let colj = base.add(j * nrow);
                    for i in r0..r1 {
                        *colj.add(i) = *colj.add(i) - *colk.add(i) * l1j - *colk1.add(i) * l2j;
                    }
                }
                let (head, tail) = w.split_at_mut((k + 1 - kb2) * deep);
                let wk = &mut head[(k - kb2) * deep..];
                let wk1 = &mut tail[..deep];
                for i in r0..r1 {
                    let wik = *colk.add(i);
                    let wik1 = *colk1.add(i);
                    wk[i - r0] = wik;
                    wk1[i - r0] = wik1;
                    *colk.add(i) = (d22 * wik - d21 * wik1) * detinv;
                    *colk1.add(i) = (d11 * wik1 - d21 * wik) * detinv;
                }
                k += 2;
            } else {
                let dinv = d_diag[k].recip();
                let colk = base.add(k * nrow);
                for j in (k + 1)..ke2 {
                    let wj_dinv = mult_snap[(k - kb) * nb + (j - kb)];
                    if wj_dinv != T::zero() {
                        let colj = base.add(j * nrow);
                        for i in r0..r1 {
                            *colj.add(i) = *colj.add(i) - *colk.add(i) * wj_dinv;
                        }
                    }
                }
                let wk = &mut w[(k - kb2) * deep..(k - kb2 + 1) * deep];
                for i in r0..r1 {
                    let v = *colk.add(i);
                    wk[i - r0] = v;
                    *colk.add(i) = v * dinv;
                }
                k += 1;
            }
        }
        flush_trailing(
            base, nrow, kb, ke, kb2, ke2, pend0, ke2, &w, deep, mult_snap, nb, r0,
        );
        kb2 = ke2;
    }
}

/// Apply the deferred updates of pivots `[p0, p1)` of the sub-block
/// `[kb2, ke2)` (their unscaled columns in `w`, indexed from `kb2`) to the
/// columns `[ke2, ke)` of rows `r0..r0 + deep`:
/// `B[:, ke2..ke] -= W[:, p0..p1] * M[p0..p1, ke2..ke]`.
///
/// # Safety
/// `base` is the panel with leading dimension `nrow`; the row range is
/// this task's own.
#[allow(clippy::too_many_arguments)]
unsafe fn flush_trailing<T: Scalar>(
    base: *mut T,
    nrow: usize,
    kb: usize,
    ke: usize,
    kb2: usize,
    ke2: usize,
    p0: usize,
    p1: usize,
    w: &[T],
    deep: usize,
    mult_snap: &[T],
    nb: usize,
    r0: usize,
) {
    let npend = p1 - p0;
    let ncols = ke - ke2;
    if npend == 0 || ncols == 0 {
        return;
    }
    let lhs = w.as_ptr().add((p0 - kb2) * deep);
    // M rows p0..p1, columns ke2..ke: row-major with stride `nb`.
    let rhs = mult_snap.as_ptr().add((p0 - kb) * nb + (ke2 - kb));
    let dst = base.add(ke2 * nrow + r0);
    // The direct kernel, not the backend entry: these products are skinny
    // (k = trailing_block), where the complex split's plane copies cost more
    // than they save (measured: 1.84 s against 1.98 s single-core).
    gemm::gemm(
        deep,
        ncols,
        npend,
        dst,
        nrow as isize,
        1,
        true,
        lhs,
        deep as isize,
        1,
        rhs,
        1,
        nb as isize,
        T::one(),
        T::zero() - T::one(),
        false,
        false,
        false,
        gemm::Parallelism::None,
    );
}

/// One blocked Bunch-Kaufman panel step of the left-looking cdiv over the
/// fully-summed columns `[kb, ke)`: the in-panel getf2 (rows `< ke`) plus the
/// row-parallel deep replay. Touches ONLY panel columns `[kb, ke)` (their full
/// `nrow` height), which is what makes the cdiv panel lookahead sound: the
/// step for panel `p+1` may run concurrently with the wide part of panel
/// `p`'s deferred Schur update (columns `>= ke2`), the two column ranges are
/// disjoint. Returns the number of perturbed pivots.
#[allow(clippy::too_many_arguments)]
fn ll_bk_panel_step<T: Scalar>(
    panel: &mut [T],
    nrow: usize,
    kb: usize,
    ke: usize,
    nb: usize,
    alpha: f64,
    perturb_floor: Option<f64>,
    ll_cdiv_par: usize,
    trailing_block: usize,
    d: &mut [T],
    d_subdiag: &mut [T],
    two_by_two: &mut [bool],
    lperm: &mut [usize],
    l1: &mut [T],
    l2: &mut [T],
    deep_swaps: &mut [usize],
    mult_snap: &mut [T],
) -> Result<usize, RslabError> {
    let mut perturbed = 0usize;
    // getf2: unblocked Bunch-Kaufman over the panel columns [kb, ke), with
    // EVERYTHING bounded to the panel rows `< ke`: pivot candidates,
    // rank-1/rank-2 updates, interchanges. The deep rows `[ke, nrow)` -
    // the dominant `O((nrow-ke)*pw^2)` share on tall panels - are lifted
    // off this serial path into the parallel `apply_bk_panel_trailing`
    // below (bit-identical replay; ports the LU twin's lever).
    for ds in deep_swaps.iter_mut() {
        *ds = usize::MAX;
    }
    let mut k = kb;
    while k < ke {
        let absakk = panel[k + k * nrow].magnitude();
        // colmax over the in-panel candidate rows (k+1)..ke.
        let mut colmax_sq = 0.0;
        let mut imax = k;
        for i in (k + 1)..ke {
            let m = panel[k * nrow + i].magnitude_sq();
            if m > colmax_sq {
                colmax_sq = m;
                imax = i;
            }
        }
        let colmax = colmax_sq.sqrt();

        let kstep;
        let kp;
        if absakk.max(colmax) == 0.0 {
            if perturb_floor.is_none() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            kstep = 1;
            kp = k;
        } else if absakk >= alpha * colmax {
            kstep = 1;
            kp = k;
        } else {
            // rowmax in row `imax`, restricted to the panel.
            let mut rowmax_sq = 0.0;
            for j in k..imax {
                let m = panel[j * nrow + imax].magnitude_sq();
                if m > rowmax_sq {
                    rowmax_sq = m;
                }
            }
            for i in (imax + 1)..ke {
                let m = panel[imax * nrow + i].magnitude_sq();
                if m > rowmax_sq {
                    rowmax_sq = m;
                }
            }
            let rowmax = rowmax_sq.sqrt();
            if absakk >= alpha * colmax * (colmax / rowmax) {
                kstep = 1;
                kp = k;
            } else if panel[imax * nrow + imax].magnitude() >= alpha * rowmax {
                kstep = 1;
                kp = imax;
            } else {
                kstep = 2;
                kp = imax;
            }
        }

        if kstep == 1 {
            if kp != k {
                swap_sym_lower_bounded(panel, nrow, k, kp, ke);
                lperm.swap(k, kp);
                deep_swaps[k - kb] = kp;
            }
            let mut dk = panel[k + k * nrow];
            match perturb_floor {
                Some(floor) if dk.magnitude() < floor => {
                    dk = perturb_pivot(dk, floor);
                    panel[k + k * nrow] = dk;
                    perturbed += 1;
                }
                None if dk == T::zero() => {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                _ => {}
            }
            d[k] = dk;
            let dinv = dk.recip();
            // Update the in-panel trailing columns (k+1)..ke over the
            // panel rows, then scale column k's panel rows (deep rows
            // replayed in the parallel apply).
            for j in (k + 1)..ke {
                let wj_dinv = panel[k * nrow + j] * dinv;
                mult_snap[(k - kb) * nb + (j - kb)] = wj_dinv;
                if wj_dinv != T::zero() {
                    for i in j..ke {
                        panel[j * nrow + i] = panel[j * nrow + i] - panel[k * nrow + i] * wj_dinv;
                    }
                }
            }
            for i in (k + 1)..ke {
                panel[k * nrow + i] = panel[k * nrow + i] * dinv;
            }
            k += 1;
        } else {
            if kp != k + 1 {
                swap_sym_lower_bounded(panel, nrow, k + 1, kp, ke);
                lperm.swap(k + 1, kp);
                deep_swaps[k - kb] = kp;
            }
            let mut d11 = panel[k + k * nrow];
            let d21 = panel[k * nrow + (k + 1)];
            let mut d22 = panel[(k + 1) + (k + 1) * nrow];
            let mut det = d11 * d22 - d21 * d21;
            let scale = d11.magnitude().max(d22.magnitude()).max(d21.magnitude());
            let growth_floor = GROWTH_EPS * scale * scale;
            match perturb_floor {
                Some(floor) => {
                    let fl = (floor * floor).max(growth_floor);
                    if det.magnitude() < fl {
                        let lift = floor.max(scale * GROWTH_EPS.sqrt());
                        d11 = d11 + T::from_real(lift);
                        d22 = d22 + T::from_real(lift);
                        det = d11 * d22 - d21 * d21;
                        if det.magnitude() < fl {
                            det = det + T::from_real(fl);
                        }
                        perturbed += 1;
                    }
                }
                None if det.magnitude() <= growth_floor => {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                _ => {}
            }
            let detinv = det.recip();
            d[k] = d11;
            d_subdiag[k] = d21;
            d[k + 1] = d22;
            two_by_two[k] = true;
            for i in (k + 2)..ke {
                let wik = panel[k * nrow + i];
                let wik1 = panel[(k + 1) * nrow + i];
                l1[i] = (d22 * wik - d21 * wik1) * detinv;
                l2[i] = (d11 * wik1 - d21 * wik) * detinv;
                mult_snap[(k - kb) * nb + (i - kb)] = l1[i];
                mult_snap[(k + 1 - kb) * nb + (i - kb)] = l2[i];
            }
            for j in (k + 2)..ke {
                let l1j = l1[j];
                let l2j = l2[j];
                for i in j..ke {
                    panel[j * nrow + i] = panel[j * nrow + i]
                        - panel[k * nrow + i] * l1j
                        - panel[(k + 1) * nrow + i] * l2j;
                }
            }
            for i in (k + 2)..ke {
                panel[k * nrow + i] = l1[i];
                panel[(k + 1) * nrow + i] = l2[i];
            }
            k += 2;
        }
    }
    // Deep rows [ke, nrow): replay this panel's interchanges + pivot
    // transforms row-parallel (bit-identical to the old full-height
    // getf2 - same per-row op sequence). This is the dominant panel
    // work on tall supernodes; it now runs on all idle workers instead
    // of the serial getf2 path.
    if nrow > ke {
        let deep = nrow - ke;
        let pw = ke - kb;
        let par = deep * pw * pw >= ll_cdiv_par;
        if par {
            let pp = LdltPanelPtr(panel.as_mut_ptr());
            let nthreads = rayon::current_num_threads().max(1);
            let cs = deep.div_ceil(nthreads).max(1);
            let ranges: Vec<(usize, usize)> = (0..nthreads)
                .map(|c| {
                    let r0 = ke + c * cs;
                    (r0.min(nrow), (r0 + cs).min(nrow))
                })
                .filter(|(a, b)| a < b)
                .collect();
            ranges.par_iter().for_each(|&(r0, r1)| {
                // SAFETY: disjoint row chunk; see `apply_bk_panel_trailing`.
                unsafe {
                    apply_bk_panel_trailing(
                        pp.get(),
                        nrow,
                        kb,
                        ke,
                        d,
                        d_subdiag,
                        two_by_two,
                        deep_swaps,
                        mult_snap,
                        nb,
                        r0,
                        r1,
                        trailing_block,
                    )
                };
            });
        } else {
            // Row chunks that keep a chunk of every panel column in L1 across
            // the whole pivot sweep (the sweep streams each column once per
            // pivot; unchunked, a deep panel is re-read from L2 `pw` times).
            // Rows are independent, so every row's arithmetic is unchanged:
            // bit-identical to one call over all rows.
            // SAFETY: single task over all deep rows.
            unsafe {
                apply_bk_panel_trailing(
                    panel.as_mut_ptr(),
                    nrow,
                    kb,
                    ke,
                    d,
                    d_subdiag,
                    two_by_two,
                    deep_swaps,
                    mult_snap,
                    nb,
                    ke,
                    nrow,
                    trailing_block,
                )
            };
        }
    }

    Ok(perturbed)
}

/// cdiv + store + emit for supernode `s` on an already fully cmod-updated
/// `panel` - the tail of [`ll_factor_node`], extracted so the spine
/// pipeline executor can drive assembly/cmod itself and reuse
/// the identical factor kernel. Takes `panel` and the global->local scratch
/// `gloc` by value (`gloc` is returned to the thread-local scratch slot on
/// every exit path).
#[allow(clippy::too_many_arguments)]
pub(super) fn ll_cdiv_emit<T: Scalar>(
    s: usize,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    store: &LlStore<T>,
    emit: &LlEmitLdlt<T>,
    perturb_floor: Option<f64>,
    n_perturbed: &AtomicUsize,
    kt: KernelTuning,
    panel: &mut [T],
) -> Result<(), RslabError> {
    let snode = &sym.supernodes[s];
    let ncol = snode.ncol;
    let nrow = sched.rows(s).len();
    // cdiv: partial **blocked** Bunch-Kaufman LDL^T (1x1 and 2x2 pivots), the
    // rectangular `nrow x ncol` analogue of `factor_front`'s panel kernel. The
    // fully-summed columns are factored in panels of width `NB` with pivoting
    // **bounded to the panel** (candidate rows `(k+1)..ke`), then each panel's
    // trailing update - the remaining panel columns `[ke, ncol)` over all rows
    // `[ke, nrow)` - is deferred to one SIMD GEMM (the BLAS-3 bulk, replacing the
    // scalar rank-1/rank-2 sweeps that dominated wide separators). Unlike
    // `factor_front` there is **no `A22` block** (the panel has no columns beyond
    // `ncol`; that Schur update is the ancestors' `cmod`), so the trailing region
    // is the rectangular `(nrow-ke) x (ncol-ke)` lower part. Pivoting stays inside
    // `0..ncol`, so the off-diagonal rows `[ncol, nrow)` keep their identity and
    // `s`'s contribution to ancestors is unaffected by this internal permutation.
    //
    // Adaptive panel width: wide separators get double-width panels - the
    // deferred Schur GEMM's inner dimension is `nb`, and k = 64 is too thin
    // to reach peak on root-class panels (measured ~79 Gflop/s-eq). The
    // extra serial getf2 work is O(nb^3) per panel - negligible against the
    // GEMM gain at this size. The global nb sweep said 128 loses overall
    // because SMALL panels pay; widening only above `ncol >= 512` (a pure
    // function of the node, thread-count independent) keeps them at default.
    let nb = if ncol >= 512 {
        kt.k.panel_nb.max(128)
    } else {
        kt.k.panel_nb
    };
    // Same join-steal guard as cmod: a small node must not fork inside its
    // cdiv (deep-row apply / deferred Schur GEMM) - the blocked join steals
    // foreign subtree work and stalls this node's dependents. Total cdiv
    // work ~ nrow*ncol^2 (panel + trailing updates).
    let ll_cdiv_par = if nrow * ncol * ncol >= 100_000_000 {
        kt.k.par_cdiv
    } else {
        usize::MAX
    };
    let alpha = bk_alpha();
    let mut d = vec![T::zero(); ncol];
    let mut d_subdiag = vec![T::zero(); ncol];
    let mut two_by_two = vec![false; ncol];
    let mut lperm: Vec<usize> = (0..nrow).collect();
    // 2x2 multiplier scratch (reused; only `[k+2, nrow)` is ever read each step).
    let mut l1 = vec![T::zero(); nrow];
    let mut l2 = vec![T::zero(); nrow];
    // Per-panel deferred-GEMM scratch (reused across panels).
    let mut l21buf: Vec<T> = Vec::new();
    let mut gbuf: Vec<T> = Vec::new();
    let mut tmp: Vec<T> = Vec::new();
    // Per-step pivot-interchange partners of the current panel (`usize::MAX`
    // = no interchange), consumed by the deep-row replay.
    let mut deep_swaps = vec![usize::MAX; nb];
    // Time-of-step in-panel multipliers (`nb x nb`, column = step), consumed
    // by the deep-row replay (later interchanges permute the final panel's
    // multiplier rows, so the finals cannot be read back).
    let mut mult_snap = vec![T::zero(); nb * nb];
    let mut local_perturbed = 0usize;
    // Panel-lookahead state: a second scratch set for the joined next-panel
    // step, the wide-Schur staging buffer, and the high-water mark of columns
    // already factored ahead by the lookahead join.
    let mut l1b = vec![T::zero(); nrow];
    let mut l2b = vec![T::zero(); nrow];
    let mut deep_swaps_b = vec![usize::MAX; nb];
    let mut mult_snap_b = vec![T::zero(); nb * nb];
    let mut tmp_w: Vec<T> = Vec::new();
    let mut done_through = 0usize;
    let mut kb = 0;
    while kb < ncol {
        let ke = (kb + nb).min(ncol);
        if kb >= done_through {
            let r = ll_bk_panel_step(
                panel,
                nrow,
                kb,
                ke,
                nb,
                alpha,
                perturb_floor,
                ll_cdiv_par,
                kt.k.trailing_block,
                &mut d,
                &mut d_subdiag,
                &mut two_by_two,
                &mut lperm,
                &mut l1,
                &mut l2,
                &mut deep_swaps,
                &mut mult_snap,
            );
            match r {
                Ok(np) => local_perturbed += np,
                Err(e) => {
                    return Err(e);
                }
            }
        }
        // Deferred panel trailing update: panel[ke.., ke..ncol] -= L21*D*R^T, where
        // L21 = panel rows [ke,nrow) x panel cols [kb,ke) (mtxpw), G = L21*D (block-
        // diagonal D), and R = the first `cw` rows of L21 (the rows that are
        // themselves remaining panel columns [ke,ncol)). The result `tmp` is the
        // rectangular `mt x cw` Schur block; only its lower part is written back.
        let pw = ke - kb;
        let cw = ncol - ke; // remaining fully-summed columns to update
        let mt = nrow - ke; // trailing rows (left-factor height)
        if pw > 0 && cw > 0 && mt > 0 {
            l21buf.clear();
            l21buf.resize(mt * pw, T::zero());
            for cc in 0..pw {
                let c = kb + cc;
                for rr in 0..mt {
                    l21buf[rr + cc * mt] = panel[(ke + rr) + c * nrow];
                }
            }
            gbuf.clear();
            gbuf.resize(mt * pw, T::zero());
            let mut cc = 0;
            while cc < pw {
                let c = kb + cc;
                if two_by_two[c] {
                    let (d11, d21, d22) = (d[c], d_subdiag[c], d[c + 1]);
                    for rr in 0..mt {
                        let a = l21buf[rr + cc * mt];
                        let b = l21buf[rr + (cc + 1) * mt];
                        gbuf[rr + cc * mt] = a * d11 + b * d21;
                        gbuf[rr + (cc + 1) * mt] = a * d21 + b * d22;
                    }
                    cc += 2;
                } else {
                    let dc = d[c];
                    for rr in 0..mt {
                        gbuf[rr + cc * mt] = l21buf[rr + cc * mt] * dc;
                    }
                    cc += 1;
                }
            }
            // Panel lookahead: split this panel's Schur into the NARROW part
            // (the next panel's columns [ke, ke2)) and the WIDE rest
            // ([ke2, ncol)), then factor the next panel concurrently with the
            // wide GEMM - the two touch disjoint column ranges. The gate is a
            // pure function of the node shape (never of thread count or the
            // racy chain state), so the GEMM split, and therefore the bits,
            // are deterministic per matrix; `ll_thread_determinism` holds.
            let ke2 = (ke + nb).min(ncol);
            let cw_n = ke2 - ke;
            let wide = cw - cw_n;
            let look = kt.k.use_gemm_schur && wide > 0 && mt * wide * pw >= kt.k.par_cdiv;
            if look {
                // Narrow Schur into the next panel's columns.
                tmp.clear();
                tmp.resize(mt * cw_n, T::zero());
                // SAFETY: `tmp`, `gbuf`, `l21buf` are distinct allocations
                // sized for the (mt, cw_n, pw) strides.
                unsafe {
                    lower_tile_gemm(
                        &mut tmp,
                        mt,
                        cw_n,
                        pw,
                        gbuf.as_ptr(),
                        mt as isize,
                        l21buf.as_ptr(),
                        mt as isize,
                        ll_cdiv_par,
                        &kt.k,
                    )
                };
                for cc2 in 0..cw_n {
                    let c = ke + cc2;
                    for rr in cc2..mt {
                        let dst = (ke + rr) + c * nrow;
                        panel[dst] = panel[dst] - tmp[rr + cc2 * mt];
                    }
                }
                // Join: next panel's getf2 + deep replay (columns [ke, ke2))
                // alongside the wide Schur (columns [ke2, ncol)).
                tmp_w.clear();
                tmp_w.resize(mt * wide, T::zero());
                let (left, right) = panel.split_at_mut(ke2 * nrow);
                let (gbuf_ref, l21_ref, tw_ref) = (&gbuf, &l21buf, &mut tmp_w);
                let (step_res, ()) = rayon::join(
                    || {
                        ll_bk_panel_step(
                            left,
                            nrow,
                            ke,
                            ke2,
                            nb,
                            alpha,
                            perturb_floor,
                            ll_cdiv_par,
                            kt.k.trailing_block,
                            &mut d,
                            &mut d_subdiag,
                            &mut two_by_two,
                            &mut lperm,
                            &mut l1b,
                            &mut l2b,
                            &mut deep_swaps_b,
                            &mut mult_snap_b,
                        )
                    },
                    || {
                        // SAFETY: distinct allocations; the rhs offset selects
                        // the wide columns' R rows (row = column index).
                        unsafe {
                            lower_tile_gemm(
                                tw_ref,
                                mt,
                                wide,
                                pw,
                                gbuf_ref.as_ptr(),
                                mt as isize,
                                l21_ref.as_ptr().add(cw_n),
                                mt as isize,
                                ll_cdiv_par,
                                &kt.k,
                            )
                        };
                        for cc2 in cw_n..cw {
                            let c = ke + cc2;
                            let col = &mut right[(c - ke2) * nrow..(c - ke2 + 1) * nrow];
                            let tcol = &tw_ref[(cc2 - cw_n) * mt..(cc2 - cw_n + 1) * mt];
                            for rr in cc2..mt {
                                col[ke + rr] = col[ke + rr] - tcol[rr];
                            }
                        }
                    },
                );
                match step_res {
                    Ok(np) => local_perturbed += np,
                    Err(e) => {
                        return Err(e);
                    }
                }
                done_through = ke2;
            } else {
                tmp.clear();
                tmp.resize(mt * cw, T::zero());
                if kt.k.use_gemm_schur {
                    // The write-back below reads only `rr >= cc2`, so compute the
                    // rectangular product tile-by-tile from each tile's diagonal
                    // downward. Matters most at the tree root where `cw ~ mt`
                    // (nearly-square panel) and the full product wasted ~half its
                    // flops; for tall separator panels (`mt >> cw`) the saving is
                    // small but never negative.
                    // SAFETY: `tmp`, `gbuf`, `l21buf` are distinct allocations sized
                    // for the (mt, cw, pw) strides.
                    unsafe {
                        lower_tile_gemm(
                            &mut tmp,
                            mt,
                            cw,
                            pw,
                            gbuf.as_ptr(),
                            mt as isize,
                            l21buf.as_ptr(),
                            mt as isize,
                            ll_cdiv_par,
                            &kt.k,
                        )
                    };
                } else {
                    for cc2 in 0..cw {
                        for rr in 0..mt {
                            let mut acc = T::zero();
                            for kk2 in 0..pw {
                                acc = acc + gbuf[rr + kk2 * mt] * l21buf[cc2 + kk2 * mt];
                            }
                            tmp[rr + cc2 * mt] = acc;
                        }
                    }
                }
                // Subtract the lower part: column c = ke+cc2 gets rows r = ke+rr, rr >= cc2.
                for cc2 in 0..cw {
                    let c = ke + cc2;
                    for rr in cc2..mt {
                        let dst = (ke + rr) + c * nrow;
                        panel[dst] = panel[dst] - tmp[rr + cc2 * mt];
                    }
                }
            }
        }
        kb = ke;
    }
    if local_perturbed > 0 {
        n_perturbed.fetch_add(local_perturbed, Ordering::Relaxed);
    }
    // Populate the O(n) emit maps + inertia for `s` (block-aware over its 1x1/2x2
    // Bunch-Kaufman D), mirroring the legacy pass-1 emit. The `e`-numbering is one
    // position per column, so `e_offset[s] + p` is column `p`'s elimination index.
    let eoff = emit.e_offset[s];
    let (mut ipos, mut ineg, mut izero) = (0usize, 0usize, 0usize);
    let mut pp = 0;
    while pp < ncol {
        let g = sched.rows(s)[lperm[pp]] as usize;
        let e = eoff + pp;
        // SAFETY: each global index / position is written by exactly one supernode.
        unsafe {
            emit.e_of_g.set(g, e);
            emit.perm.set(e, sym.perm[g]);
            emit.d_diag.set(e, d[pp]);
        }
        if two_by_two[pp] {
            let g2 = sched.rows(s)[lperm[pp + 1]] as usize;
            unsafe {
                emit.e_of_g.set(g2, e + 1);
                emit.perm.set(e + 1, sym.perm[g2]);
                emit.d_diag.set(e + 1, d[pp + 1]);
                emit.d_subdiag.set(e, d_subdiag[pp]);
                emit.two_by_two.set(e, true);
            }
            let det_r = (d[pp] * d[pp + 1] - d_subdiag[pp] * d_subdiag[pp]).real();
            let tr_r = (d[pp] + d[pp + 1]).real();
            if det_r < 0.0 {
                ipos += 1;
                ineg += 1;
            } else if det_r > 0.0 {
                if tr_r >= 0.0 {
                    ipos += 2;
                } else {
                    ineg += 2;
                }
            } else {
                izero += 1;
                if tr_r >= 0.0 {
                    ipos += 1;
                } else {
                    ineg += 1;
                }
            }
            pp += 2;
        } else {
            let r = d[pp].real();
            if r > 0.0 {
                ipos += 1;
            } else if r < 0.0 {
                ineg += 1;
            } else {
                izero += 1;
            }
            pp += 1;
        }
    }
    emit.inertia_pos.fetch_add(ipos, Ordering::Relaxed);
    emit.inertia_neg.fetch_add(ineg, Ordering::Relaxed);
    emit.inertia_zero.fetch_add(izero, Ordering::Relaxed);
    // SAFETY: this thread owns supernode `s` and writes its cell exactly once.
    unsafe {
        store.set(
            s,
            LdltSlot {
                d,
                dsub: d_subdiag,
                two: two_by_two,
                lperm,
            },
        )
    };
    Ok(())
}

/// The Bunch-Kaufman pivot threshold `alpha = (1 + sqrt17)/8 ~ 0.6404`.
#[inline]
pub(crate) fn bk_alpha() -> f64 {
    (1.0 + 17.0_f64.sqrt()) / 8.0
}

/// Symmetric interchange of rows and columns `p < q` in a column-major
/// lower-triangle panel of leading dimension `n`, with the below-`q`
/// column-segment swap bounded to rows `< row_limit`. The blocked Bunch-Kaufman panel kernels keep their pivot
/// interchanges inside the panel rows and replay the deep-row segments later
/// in the parallel trailing apply (`apply_bk_panel_trailing`), so the
/// interchange sequence reaches every row exactly once, in step order.
pub(crate) fn swap_sym_lower_bounded<T: Scalar>(
    a: &mut [T],
    n: usize,
    p: usize,
    q: usize,
    row_limit: usize,
) {
    debug_assert!(p < q && q < n && q < row_limit);
    // Column segment strictly below q: (i, p) <-> (i, q) for i > q.
    for i in (q + 1)..row_limit {
        a.swap(p * n + i, q * n + i);
    }
    // Middle cross strip: (i, p) <-> (q, i) for p < i < q.
    for i in (p + 1)..q {
        a.swap(p * n + i, i * n + q);
    }
    // Diagonal: (p, p) <-> (q, q).
    a.swap(p * n + p, q * n + q);
    // Left row segments: (p, j) <-> (q, j) for j < p.
    for j in 0..p {
        a.swap(j * n + p, j * n + q);
    }
}
