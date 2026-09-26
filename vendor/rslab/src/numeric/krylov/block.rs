//! Block GMRES for many right-hand sides with CGS2 over the whole panel.

use super::util::*;
use super::*;

use crate::error::RslabError;
use crate::scalar::Scalar;

/// Outcome of a block (multi-RHS) Krylov solve.
#[derive(Debug, Clone)]
pub struct BlockKrylovResult<T> {
    /// Solutions, column-major `nxs` (RHS `c` is the slice `[c*n, (c+1)*n)`).
    pub x: Vec<T>,
    /// Block iterations performed (the RHS advance in lockstep).
    pub iters: usize,
    /// `true` iff **every** RHS reached `tol`.
    pub converged: bool,
    /// Per-RHS final relative residual `||b_c - A x_c|| / ||b_c||`.
    pub final_res: Vec<f64>,
    /// Why the block iteration stopped: `Converged` when every RHS met `tol`,
    /// else `MaxIter` (the block loop has no non-converged breakdown state).
    pub stop: StopReason,
}

/// Form and add one converged column's solution contribution to the global `x`
/// **mid-cycle**, so the column can be compacted out of the active panel: back-
/// substitute `y` from that column's frozen Hessenberg/Givens state (`h`,`g`,`jd`
/// rows), build `V_ap * y` from its Arnoldi basis at the *current* stride `sa`,
/// apply the preconditioner once (`x_c += M^-1*(V_ap*y)`). This is the block
/// analogue of the single-RHS restart update, issued for a single column the
/// instant its Hessenberg estimate reaches `tol`, so the batched applies can then
/// shrink to the still-active width.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn finalize_block_column<T, M>(
    precond: &M,
    vbas: &[T],
    h: &[Vec<Vec<T>>],
    g: &[Vec<T>],
    jd: usize,
    ap: usize,
    sa: usize,
    n: usize,
    c: usize,
    x: &mut [T],
) -> Result<(), RslabError>
where
    T: Scalar,
    M: Preconditioner<T> + ?Sized,
{
    // Guard against a (near-)singular Hessenberg diagonal: solve only the well-
    // conditioned leading block (deterministic breakdown, no NaN into `x`).
    let jd = well_conditioned_dim(&h[ap], jd);
    if jd == 0 {
        return Ok(());
    }
    let mut y = vec![T::zero(); jd];
    for i in (0..jd).rev() {
        let mut acc = g[ap][i];
        for k in (i + 1)..jd {
            acc = acc - h[ap][i][k] * y[k];
        }
        y[i] = acc * h[ap][i][i].recip();
    }
    let mut vy = vec![T::zero(); n];
    for i in 0..jd {
        let yi = y[i];
        let vb = (i * sa + ap) * n;
        for k in 0..n {
            vy[k] = vy[k] + vbas[vb + k] * yi;
        }
    }
    let mut z = vec![T::zero(); n];
    precond.apply(&vy, &mut z)?;
    let cb = c * n;
    for k in 0..n {
        x[cb + k] = x[cb + k] + z[k];
    }
    Ok(())
}

/// Right-preconditioned restarted **block GMRES** for `s` right-hand sides `b`
/// (column-major `nxs`). The `s` systems advance in lockstep so the two expensive
/// operations - the operator matvec and the preconditioner solve - are issued
/// once per step as **block** applies ([`LinearOperator::apply_block`] /
/// [`Preconditioner::apply_block`]), reaching BLAS-3 arithmetic intensity (each
/// factor / matrix value touched once for all `s` columns). Each RHS keeps its
/// own Arnoldi basis / Hessenberg / Givens, so a column converges identically to
/// the single-RHS [`gmres`]; the systems share only the batched operator and
/// preconditioner calls. Solves `A X = B` from the optional initial guess `x0`
/// (column-major `nxs`, default `X_0 = 0`).
///
/// **Warm start:** `x0 = Some(prev)` seeds every column from a related
/// previous solution; on a slowly varying sequence this cuts the block iteration
/// count. Each column's convergence is still relative to its own `||B[:,c]||`.
///
/// **Deflation:** a RHS whose true residual reaches `tol` drops out of the block,
/// so the batched applies shrink to the active width as columns converge - the
/// fast-converging RHS are never dragged along by the slowest one.
///
/// This is the MoM/FEM many-excitations path: factor (or `f32`-factor) once, then
/// drive all right-hand sides through one block iteration.
///
/// **Memory:** the Arnoldi basis is a single up-front allocation of
/// `n*s*(restart+1)` scalars (plus a handful of `n*s` work panels), *independent*
/// of how few iterations actually run - so a large `restart` on a big `n*s` can
/// allocate many GB (`n=100k, s=10, Complex<f64>, restart=80` ~ 13 GB). An
/// unset [`KrylovSettings::restart`] is capped to
/// [`KrylovSettings::basis_budget_bytes`]; an explicit value is honoured
/// exactly.
///
/// **Threads:** the parallel orthogonalization reductions run in a
/// scoped pool derived from the preconditioner's [`Threads`](crate::Threads) policy
/// ([`Preconditioner::solve_threads`], resolved at factor time), so factor and
/// solve share **one** concurrency budget. A [`Threads::Ambient`](crate::Threads::Ambient) policy (or
/// [`NoPreconditioner`]) leaves the reductions on the caller's current pool - the
/// solver-in-the-loop path, where one bounded rayon pool is installed around
/// the whole factor+solve loop. The
/// pool width never changes the numeric result (the chunk-order reduction fold is
/// thread-count independent). The single-RHS [`gmres`] orthogonalizes serially, so
/// it has no such pool.
///
/// **Orthogonalization:** the panel is orthogonalized by **block CGS2**
/// (classical Gram-Schmidt with a conditional, now *per-column*, second pass), not
/// the MGS+DGKS of the single-RHS [`gmres`]. The two summation orders differ, so a
/// block solve with `s = 1` is **not** bit-identical to [`gmres`] and may differ by
/// up to +/-1 iteration - both still converge to `tol`. See the module-level
/// "Orthogonalization" note for the rationale.
/// **Monitor:** at the start of every restart cycle, right after the true
/// residuals `||b - A*x||/||b||` of all live columns were recomputed, `mon` receives
/// `(iters_done, worst_live_residual, n_active_columns)` and returns whether the solve
/// should CONTINUE. Long solves stop being a black box: the caller can stream residual
/// trajectories to its log, and a `false` return cancels the solve early (a stagnation
/// detector cutting a stopped-contracting iteration) with `StopReason::Stalled` and the
/// best solution so far.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn gmres_block<T, A, M>(
    op: &A,
    b: &[T],
    s: usize,
    precond: &M,
    settings: &KrylovSettings,
    x0: Option<&[T]>,
    mut mon: Option<&mut dyn FnMut(usize, f64, usize) -> bool>,
) -> Result<BlockKrylovResult<T>, RslabError>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if s == 0 || b.len() != n * s {
        return Err(RslabError::DimensionMismatch {
            expected: n * s,
            got: b.len(),
        });
    }
    let (tol, max_iter, reorth_eta) = (settings.tol, settings.max_iter, settings.reorth_eta);
    let chunk = settings.ortho_chunk.max(1);
    let m = settings.restart_for(n, s, std::mem::size_of::<T>(), 1);
    // Solve-phase thread policy: orthogonalize in a pool of the same
    // width the preconditioner was factored with, so factor and solve share one
    // concurrency budget. `None` (Ambient / no factor) keeps the caller's pool.
    let ortho_pool = solve_thread_pool(precond.solve_threads());
    // Warm start: seed every column from `x0` (column-major `nxs`).
    let mut x = match x0 {
        Some(g) => {
            if g.len() != n * s {
                return Err(RslabError::DimensionMismatch {
                    expected: n * s,
                    got: g.len(),
                });
            }
            g.to_vec()
        }
        None => vec![T::zero(); n * s],
    };
    let bnorm: Vec<f64> = (0..s).map(|c| norm2(&b[c * n..c * n + n])).collect();
    // Set when the progress monitor cancels the solve (`StopReason::Stalled`).
    let mut stalled = false;

    // Scratch sized for the **full** width `s` and reused; each restart cycle uses
    // only the first `sa` columns, where `sa` is the count of still-active RHS.
    // The basis stride is therefore `sa` (recomputed per cycle): block `j`, active
    // column `a` lives at `vbas[(j*sa + a)*n ..]`, so block `j` (the `nxsa` input
    // to one block apply) is the contiguous prefix `vbas[j*sa*n .. (j+1)*sa*n]`.
    let mut vbas = vec![T::zero(); n * s * (m + 1)];
    let mut zblk = vec![T::zero(); n * s]; // M^-1 * (block j)
    let mut wblk = vec![T::zero(); n * s]; // A * zblk
    let mut axblk = vec![T::zero(); n * s];
    let mut vyblk = vec![T::zero(); n * s];
    let mut xc = vec![T::zero(); n * s]; // compact live-RHS solutions for the residual matvec
                                         // Block Gram-Schmidt projection panels: `proj[i*sa + ap]` for blocks `0..=j`,
                                         // columns `0..sa`. One classical (block) projection pass, plus a **conditional**
                                         // reorthogonalization pass (block DGKS) taken only when a column loses
                                         // orthogonality - the backward-stable, single-thread-cheap analogue of the
                                         // old per-RHS MGS+DGKS, now batched over the whole panel.
    let mut proj1 = vec![T::zero(); m * s];
    let mut proj2 = vec![T::zero(); m * s];
    let mut wnorm0 = vec![0.0f64; s]; // panel column norms before ortho (DGKS reorth test)
    let mut reorth_col = vec![false; s]; // per-column DGKS second-pass flags
                                         // Reduction scratch for `block_project`: `nchunks * (m*s)`, reused every step
                                         // so the orthogonalization allocates nothing in the hot loop.
    let mut proj_scratch = vec![T::zero(); n.div_ceil(chunk) * m * s];

    // Per-active-position Arnoldi state (indexed `0..sa`, reset each cycle).
    let mut h: Vec<Vec<Vec<T>>> = (0..s).map(|_| vec![vec![T::zero(); m]; m + 1]).collect();
    let mut cs: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); m]).collect();
    let mut sn: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); m]).collect();
    let mut g: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); m + 1]).collect();
    let mut jdim = vec![0usize; s];
    let mut converged = vec![false; s];
    let mut final_res = vec![0.0f64; s];
    // Back-substitution buffer for the per-column restart update, hoisted out of
    // the hot loop and reused: each column's back-sub overwrites the
    // `0..jd` prefix before reading it, so no per-column-per-cycle allocation.
    let mut y = vec![T::zero(); m];
    let mut total = 0usize;
    for c in 0..s {
        if bnorm[c] == 0.0 {
            converged[c] = true;
            // Exact solution of `A x = 0` is `0`; discard any warm-start seed here.
            x[c * n..c * n + n].iter_mut().for_each(|v| *v = T::zero());
        }
    }

    while total < max_iter {
        // **Deflation:** gather the not-yet-converged ("live") RHS into a compact
        // block, recompute their true residual with one block matvec, and keep
        // only those still above `tol` as the active set for this cycle. Converged
        // columns never re-enter the (expensive) inner block applies again.
        let live: Vec<usize> = (0..s).filter(|&c| !converged[c]).collect();
        if live.is_empty() {
            break;
        }
        let lw = live.len();
        for (a, &c) in live.iter().enumerate() {
            xc[a * n..a * n + n].copy_from_slice(&x[c * n..c * n + n]);
        }
        op.apply_block(&xc[..lw * n], &mut axblk[..lw * n], lw);
        // Active set = live RHS whose true residual still exceeds `tol`.
        let mut act: Vec<usize> = Vec::new();
        for (a, &c) in live.iter().enumerate() {
            let cb = c * n;
            let ab = a * n;
            let mut rn = 0.0;
            for i in 0..n {
                rn += (b[cb + i] - axblk[ab + i]).magnitude_sq();
            }
            let beta = rn.sqrt();
            final_res[c] = beta / bnorm[c];
            if final_res[c] <= tol {
                converged[c] = true;
            } else {
                // Initialize active position `act.len()` from this residual.
                let ap = act.len();
                let inv = T::from_real(1.0 / beta);
                for i in 0..n {
                    vbas[ap * n + i] = (b[cb + i] - axblk[ab + i]) * inv; // block 0, col ap
                }
                for row in h[ap].iter_mut() {
                    for e in row.iter_mut() {
                        *e = T::zero();
                    }
                }
                for e in g[ap].iter_mut() {
                    *e = T::zero();
                }
                g[ap][0] = T::from_real(beta);
                jdim[ap] = 0;
                act.push(c);
            }
        }
        let mut sa = act.len();
        // Per-cycle progress monitor (rapidmom-local addition): true residuals of all
        // live columns are fresh at this point, the honest place to report, and the
        // honest place to CANCEL (a `false` return stops a stagnated solve early).
        if let Some(m) = mon.as_mut() {
            let worst = live.iter().map(|&c| final_res[c]).fold(0.0_f64, f64::max);
            if !m(total, worst, sa) {
                stalled = true;
                break;
            }
        }
        if sa == 0 {
            break;
        }

        let mut inner_done = vec![false; sa];
        for j in 0..m {
            if total >= max_iter {
                break;
            }
            let jblock = j * sa * n;
            // Batched right preconditioning + operator apply over the active block.
            precond.apply_block(&vbas[jblock..jblock + sa * n], &mut zblk[..sa * n], sa, n)?;
            op.apply_block(&zblk[..sa * n], &mut wblk[..sa * n], sa);
            // **Block Gram-Schmidt** of the whole `sa`-column panel `W` against each
            // column's own basis `V_0..V_j`: one classical projection pass
            // (project -> subtract), then a **conditional** reorthogonalization pass
            // (block DGKS) taken only when a column's norm collapses - the
            // backward-stable, single-thread-cheap analogue of the old per-RHS
            // MGS+DGKS. Both passes are panel-wide, high-arithmetic-intensity sweeps
            // parallelized over the vector dimension, replacing the `O(j*sa)`
            // sequential BLAS-1 inner products. The reorth decision is taken from
            // serial column norms, so it is identical across thread counts (the
            // whole solve stays bit-identical regardless of parallelism). Converged
            // columns are still swept (kept in the panel for batching) but their
            // Hessenberg/Givens state is frozen below.
            let blocks = j + 1;
            for ap in 0..sa {
                wnorm0[ap] = norm2(&wblk[ap * n..ap * n + n]);
            }
            ortho_in_pool(&ortho_pool, || {
                block_project(
                    &vbas,
                    &wblk,
                    blocks,
                    sa,
                    n,
                    chunk,
                    &mut proj1,
                    &mut proj_scratch,
                )
            });
            ortho_in_pool(&ortho_pool, || {
                block_subtract(&vbas, &mut wblk, blocks, sa, n, chunk, &proj1)
            });
            // **Per-column** DGKS second pass: decide the reorth *per
            // column* from its own norm collapse, not panel-globally. Frozen
            // (converged-this-cycle) columns are excluded (they are compacted out at
            // each step boundary, so a stale/collapsed frozen column can never
            // trigger a reorth). Only columns that actually lost orthogonality get
            // the second projection subtracted; the rest keep their pass-1 result
            // bit-for-bit - a single ill-conditioned column no longer imposes the
            // arithmetic second orthogonalization on the well-conditioned columns.
            let mut any_reorth = false;
            for ap in 0..sa {
                let need =
                    !inner_done[ap] && norm2(&wblk[ap * n..ap * n + n]) < reorth_eta * wnorm0[ap];
                reorth_col[ap] = need;
                any_reorth |= need;
            }
            if any_reorth {
                ortho_in_pool(&ortho_pool, || {
                    block_project(
                        &vbas,
                        &wblk,
                        blocks,
                        sa,
                        n,
                        chunk,
                        &mut proj2,
                        &mut proj_scratch,
                    )
                });
                // Zero the second-pass projection for columns that do not need it, so
                // `block_subtract` skips them (its `hij == 0` guard) and their `w`
                // and Hessenberg entries stay exactly at the pass-1 values.
                for ap in 0..sa {
                    if !reorth_col[ap] {
                        for i in 0..blocks {
                            proj2[i * sa + ap] = T::zero();
                        }
                    }
                }
                ortho_in_pool(&ortho_pool, || {
                    block_subtract(&vbas, &mut wblk, blocks, sa, n, chunk, &proj2)
                });
            }
            for ap in 0..sa {
                if inner_done[ap] {
                    continue;
                }
                let wb = ap * n;
                // Hessenberg column: projection pass, plus the per-column reorth
                // correction (zero for columns that did not reorth, so `hij == proj1`).
                for i in 0..=j {
                    let hij = if any_reorth {
                        proj1[i * sa + ap] + proj2[i * sa + ap]
                    } else {
                        proj1[i * sa + ap]
                    };
                    h[ap][i][j] = hij;
                }
                let hn = norm2(&wblk[wb..wb + n]);
                h[ap][j + 1][j] = T::from_real(hn);
                let v1 = ((j + 1) * sa + ap) * n;
                if hn > 0.0 {
                    let inv = T::from_real(1.0 / hn);
                    for k in 0..n {
                        vbas[v1 + k] = wblk[wb + k] * inv;
                    }
                } else {
                    for k in 0..n {
                        vbas[v1 + k] = T::zero();
                    }
                }
                // Previous rotations, then a new one to zero h[j+1][j]; update g.
                for i in 0..j {
                    let temp = cs[ap][i].conj() * h[ap][i][j] + sn[ap][i].conj() * h[ap][i + 1][j];
                    h[ap][i + 1][j] = -sn[ap][i] * h[ap][i][j] + cs[ap][i] * h[ap][i + 1][j];
                    h[ap][i][j] = temp;
                }
                let (cj, sj) = givens(h[ap][j][j], h[ap][j + 1][j]);
                cs[ap][j] = cj;
                sn[ap][j] = sj;
                h[ap][j][j] = cj.conj() * h[ap][j][j] + sj.conj() * h[ap][j + 1][j];
                h[ap][j + 1][j] = T::zero();
                let g_next = -sj * g[ap][j];
                g[ap][j] = cj.conj() * g[ap][j];
                g[ap][j + 1] = g_next;
                jdim[ap] = j + 1;
                if g[ap][j + 1].magnitude() / bnorm[act[ap]] <= tol {
                    inner_done[ap] = true;
                }
            }
            total += 1;
            // **Within-cycle deflation.** Columns whose Hessenberg estimate reached
            // `tol` this step are (a) finalized now - their solution contribution is
            // formed from the basis at the *current* stride and added to `x` - and
            // (b) compacted out of the active panel, remapping the per-column
            // Arnoldi state and rewriting the basis at the narrower stride `sa`.
            // The next step's batched preconditioner / operator applies therefore
            // shrink to the still-active width, exactly as promised: a fast RHS is
            // no longer dragged along by the slowest one until the next restart.
            if inner_done.iter().any(|&d| d) {
                for ap in 0..sa {
                    if inner_done[ap] {
                        finalize_block_column(
                            precond, &vbas, &h, &g, jdim[ap], ap, sa, n, act[ap], &mut x,
                        )?;
                    }
                }
                let survivors: Vec<usize> = (0..sa).filter(|&ap| !inner_done[ap]).collect();
                let sa_new = survivors.len();
                // Basis columns `0..=j+1` are populated for the survivors. Compact
                // to the new stride blocks-outer / survivors-inner so the write
                // offset is monotonically increasing and never clobbers an unread
                // source (each column's new offset is `<=` its old offset).
                let blocks_built = j + 2;
                for i in 0..blocks_built {
                    for (ap_new, &ap_old) in survivors.iter().enumerate() {
                        let src = (i * sa + ap_old) * n;
                        let dst = (i * sa_new + ap_new) * n;
                        if src != dst {
                            vbas.copy_within(src..src + n, dst);
                        }
                    }
                }
                // Remap the per-column Arnoldi state (Vec swaps are O(1) pointer
                // moves; the frozen columns' now-stale slots are never read again -
                // the next cycle re-initializes positions `0..sa`).
                for (ap_new, &ap_old) in survivors.iter().enumerate() {
                    if ap_new != ap_old {
                        h.swap(ap_new, ap_old);
                        cs.swap(ap_new, ap_old);
                        sn.swap(ap_new, ap_old);
                        g.swap(ap_new, ap_old);
                        jdim[ap_new] = jdim[ap_old];
                        act[ap_new] = act[ap_old];
                    }
                }
                sa = sa_new;
                act.truncate(sa);
                inner_done = vec![false; sa];
                if sa == 0 {
                    break;
                }
            }
        }

        // x_c += M^-1 (V_a y_a): back-substitute each still-active RHS, build the
        // compact VY block, one batched preconditioner apply, then scatter to
        // global `x`. Columns that deflated mid-cycle were already finalized
        // individually above, so `sa == 0` here means the whole panel converged
        // within the cycle and there is nothing left to batch.
        if sa > 0 {
            for e in vyblk[..sa * n].iter_mut() {
                *e = T::zero();
            }
            for ap in 0..sa {
                // Guard the per-column solve against a (near-)singular Hessenberg
                // diagonal: truncate to the well-conditioned leading block so a
                // rank-deficient column breaks down deterministically instead of
                // dividing by ~0 and polluting the batched applies with NaN.
                let jd = well_conditioned_dim(&h[ap], jdim[ap]);
                if jd == 0 {
                    continue;
                }
                // Reuse the hoisted `y` buffer: back-sub writes `y[0..jd]` top-down
                // (each entry before it is read), so no per-column allocation.
                for i in (0..jd).rev() {
                    let mut acc = g[ap][i];
                    for k in (i + 1)..jd {
                        acc = acc - h[ap][i][k] * y[k];
                    }
                    y[i] = acc * h[ap][i][i].recip();
                }
                let vyb = ap * n;
                for i in 0..jd {
                    let yi = y[i];
                    let vb = (i * sa + ap) * n;
                    for k in 0..n {
                        vyblk[vyb + k] = vyblk[vyb + k] + vbas[vb + k] * yi;
                    }
                }
            }
            precond.apply_block(&vyblk[..sa * n], &mut zblk[..sa * n], sa, n)?;
            for ap in 0..sa {
                let c = act[ap];
                for i in 0..n {
                    x[c * n + i] = x[c * n + i] + zblk[ap * n + i];
                }
            }
        }
    }

    // Final true residual per RHS. A converged column was measured at
    // its top-of-cycle deflation checkpoint and frozen (never re-entered the active
    // set), so its recorded `final_res` is already the final true residual - reuse
    // it. Re-matvec **only** the columns still active at the iteration budget
    // (`converged[c] == false`), compacted into one narrow block apply, instead of a
    // full-width `s` matvec over columns that deflated cycles ago. For a fully
    // converged solve this skips the final matvec entirely; the reused values are
    // bit-identical to a full recompute (the operator is column-independent).
    let pending: Vec<usize> = (0..s)
        .filter(|&c| !converged[c] && bnorm[c] != 0.0)
        .collect();
    if !pending.is_empty() {
        let lp = pending.len();
        for (a, &c) in pending.iter().enumerate() {
            xc[a * n..a * n + n].copy_from_slice(&x[c * n..c * n + n]);
        }
        op.apply_block(&xc[..lp * n], &mut axblk[..lp * n], lp);
        for (a, &c) in pending.iter().enumerate() {
            let cb = c * n;
            let ab = a * n;
            let mut rn = 0.0;
            for i in 0..n {
                rn += (b[cb + i] - axblk[ab + i]).magnitude_sq();
            }
            final_res[c] = rn.sqrt() / bnorm[c];
        }
    }
    let mut all_conv = true;
    for c in 0..s {
        if bnorm[c] == 0.0 {
            final_res[c] = 0.0;
            continue;
        }
        if final_res[c] > tol {
            all_conv = false;
        }
    }
    Ok(BlockKrylovResult {
        x,
        iters: total,
        converged: all_conv,
        final_res,
        stop: if all_conv {
            StopReason::Converged
        } else if stalled {
            StopReason::Stalled
        } else {
            StopReason::MaxIter
        },
    })
}
