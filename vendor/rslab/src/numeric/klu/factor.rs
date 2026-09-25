//! The numeric factorization: per-block Gilbert-Peierls LU with threshold
//! pivoting, and the replay plan of the refactorization.

use super::*;

/// The GP flop count carried on the attached estimate (0 when absent).
pub(super) fn diagnostics_flops(d: &crate::diagnostics::Diagnostics) -> u64 {
    d.estimate.as_ref().map_or(0, |e| e.factor_flops)
}

/// The two-parameter principle behind EVERY parallel decision on this path:
///
/// 1. **Work floor** ([`KluSettings::par_min_work`]): a unit of parallel
///    execution must carry at least this much replay work (fmadd count) -
///    below it, spawn/handoff overhead exceeds the overlap.
/// 2. **Concurrency ratio** ([`KluSettings::par_min_ratio`]): parallelism engages only
///    where the structure offers at least this much simultaneous work.
///    Across BTF blocks that is the exact Amdahl bound `sum work / max block
///    work`; inside a block it is the mean level width of the frozen
///    elimination DAG - the average number of simultaneously replayable
///    columns. (A chain-work critical-path bound would be the "exact" ratio
///    but systematically underestimates the pipeline's just-in-time overlap:
///    ASIC_100ks scores below 2 on it yet measures 2.7x.)
///
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
    min_work: u64,
    min_ratio: f64,
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
        if w_b >= min_work && width >= min_ratio {
            pipelined.push((b, (width as usize).max(2)));
        }
    }
    let ratio_ok = (total as f64) >= min_ratio * (max_w as f64);
    let par_blocks = ratio_ok && (force || total >= min_work);
    (pipelined, par_blocks)
}

/// The columns of a factor array (`colptr`, `rowidx`, `val`) as a matrix with
/// sorted rows, plus the diagonal entry `diag(j)` of each column `j` where
/// there is one.
pub(super) fn factor_csc<T: Scalar>(
    n: usize,
    colptr: &[usize],
    rowidx: &[Ki],
    val: &[T],
    diag: impl Fn(usize) -> Option<T>,
) -> GeneralCsc<T> {
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx = Vec::with_capacity(val.len() + n);
    let mut values = Vec::with_capacity(val.len() + n);
    let mut col: Vec<(usize, T)> = Vec::new();
    for j in 0..n {
        col.clear();
        col.extend((colptr[j]..colptr[j + 1]).map(|k| (rowidx[k] as usize, val[k])));
        col.extend(diag(j).map(|d| (j, d)));
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

/// Row-max scaling reciprocals (1 for empty rows / scaling off).
pub(super) fn row_scale_inv<T: Scalar>(a: &GeneralCsc<T>, enabled: bool) -> Vec<f64> {
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

/// Per-block factor output in absolute position spaces: row indices of
/// `l/u` are absolute final positions, `f_pre` holds absolute *pre-pivot*
/// positions (rows of earlier blocks, resolved to final positions when the
/// blocks are spliced in order), and `prog_in`/`f_k` carry the refactor
/// scatter program contributions (see [`KluFactors`]).
pub(super) struct BlockOut<T> {
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
pub(super) struct KluScratch<T> {
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
pub(super) fn factor_block<T: Scalar>(
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
            if dm > 0.0 && dm >= settings.pivot_threshold * maxmag {
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

pub(super) fn factor_impl<T: Scalar>(
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
        KluParallel::Auto => sym.nnz >= settings.par_min_nnz && sym.max_block_size() * 2 <= sym.n,
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
        // allocations instead, so the reserves live here.) A single block
        // (one irreducible matrix, the common case for a logic circuit) is
        // moved out of the block buffer instead of copied, so nothing is
        // reserved for it.
        if nblocks > 1 {
            l_rowidx.reserve(sym.nnz * 4);
            l_val.reserve(sym.nnz * 4);
            u_rowidx.reserve(sym.nnz * 2);
            u_val.reserve(sym.nnz * 2);
            f_rowidx.reserve(sym.nnz / 4);
            f_val.reserve(sym.nnz / 4);
        }
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
            if nblocks == 1 {
                l_rowidx = std::mem::take(&mut out.l_rowidx);
                l_val = std::mem::take(&mut out.l_val);
                u_rowidx = std::mem::take(&mut out.u_rowidx);
                u_val = std::mem::take(&mut out.u_val);
                l_colptr = std::mem::take(&mut out.l_colptr);
                u_colptr = std::mem::take(&mut out.u_colptr);
                udiag = std::mem::take(&mut out.udiag);
            } else {
                l_rowidx.extend_from_slice(&out.l_rowidx);
                l_val.extend_from_slice(&out.l_val);
                u_rowidx.extend_from_slice(&out.u_rowidx);
                u_val.extend_from_slice(&out.u_val);
                l_colptr.extend(out.l_colptr[1..].iter().map(|&p| lo + p));
                u_colptr.extend(out.u_colptr[1..].iter().map(|&p| uo + p));
                udiag.extend_from_slice(&out.udiag);
            }
            f_rowidx.extend(out.f_pre.iter().map(|&pre| fin_of_pre[pre as usize]));
            f_val.extend_from_slice(&out.f_val);
            f_colptr.extend(out.f_colptr[1..].iter().map(|&p| fo + p));
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
    // One block has nothing to pipeline: skip the replay plan.
    let (pipelined, par_refactor) = if settings.parallel == KluParallel::Off || nblocks == 1 {
        (Vec::new(), false)
    } else {
        compute_replay_plan(
            &sym.block_ptr,
            &l_colptr,
            &u_colptr,
            &u_rowidx,
            settings.parallel == KluParallel::On,
            settings.par_min_work,
            settings.par_min_ratio,
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
