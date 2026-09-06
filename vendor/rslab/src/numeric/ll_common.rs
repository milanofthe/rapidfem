//! Shared scaffolding of the supernodal left-looking drivers - the pieces that
//! are identical between the LDL^T and the LU twin: the per-supernode slot
//! store, the raw panel pointer used for disjoint-row parallel writes, and the
//! heuristic `tuned` driver (ND bakeoff + calibrated worker count).

use crate::diagnostics::MemoryEstimate;
use crate::error::RslabError;
use crate::numeric::multifrontal_ldlt::SolverSettings;
use crate::symbolic::OrderingMethod;

/// One `UnsafeCell` payload per supernode, written exactly once by the
/// supernode's owner and read only by nodes that are (transitively) its
/// assembly-tree ancestors - the single-writer-before-readers discipline the
/// left-looking schedule guarantees. `free` resets a slot once its last
/// consumer is done.
pub(crate) struct SlotStore<P> {
    slots: Vec<std::cell::UnsafeCell<P>>,
}

// SAFETY: single-writer-before-readers, disjoint indices (see the type doc).
unsafe impl<P: Send> Sync for SlotStore<P> {}

impl<P: Default> SlotStore<P> {
    pub fn new(nsuper: usize) -> Self {
        SlotStore {
            slots: (0..nsuper)
                .map(|_| std::cell::UnsafeCell::new(P::default()))
                .collect(),
        }
    }

    /// SAFETY: `k` must be a fully-factored descendant of the current node
    /// (its owner's write happened-before this read).
    pub unsafe fn get(&self, k: usize) -> &P {
        &*self.slots[k].get()
    }

    /// SAFETY: only the owner of supernode `s` calls this, exactly once.
    pub unsafe fn set(&self, s: usize, p: P) {
        *self.slots[s].get() = p;
    }

    /// Move the panel out, leaving the default in its place. SAFETY: the
    /// owner of supernode `k`, after its last reader is done.
    pub unsafe fn take(&self, k: usize) -> P {
        std::mem::take(&mut *self.slots[k].get())
    }
}

/// Raw base pointer of a panel buffer, smuggled across rayon workers so each
/// task can write its own **disjoint row range** of a column-major panel. Safe
/// only because callers partition the rows so no two tasks touch the same cell.
pub(crate) struct PanelPtr<T>(pub *mut T);
impl<T> Clone for PanelPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for PanelPtr<T> {}
// SAFETY: the pointer is only dereferenced on disjoint, caller-partitioned cells.
unsafe impl<T> Send for PanelPtr<T> {}
unsafe impl<T> Sync for PanelPtr<T> {}
impl<T> PanelPtr<T> {
    /// Extract the raw pointer. Taking `self` by value forces a closure to
    /// capture the whole (Send+Sync) wrapper rather than disjoint-capturing the
    /// bare field.
    #[inline]
    pub fn get(self) -> *mut T {
        self.0
    }
}

/// The deterministic heuristic settings pick shared by `LdltSolver::tuned` and
/// `LuSolver::tuned`: default settings, an exact ND bakeoff on large systems,
/// and (feature `tuning`, when the one-time install diagnosis has run) the
/// calibrated cost-model worker count.
pub(crate) fn tuned<A: ?Sized, S>(
    a: &A,
    base: &SolverSettings,
    analyze_with: impl Fn(&A, &SolverSettings) -> Result<S, RslabError>,
    estimate: impl Fn(&S) -> MemoryEstimate,
) -> Result<(S, SolverSettings), RslabError> {
    #[cfg(not(feature = "tuning"))]
    let _ = &estimate;
    // Ordering by the exact prefix race (`AutoRace`): every candidate's true
    // factor nnz, computed concurrently, smallest wins - replacing the former
    // Amd-pinned default plus flops-gated ND bakeoff with one exact
    // measurement (the race is what the bakeoff approximated).
    #[allow(unused_mut)]
    let mut s = base.clone().with_ordering(OrderingMethod::AutoRace);
    let sym = analyze_with(a, &s)?;
    // Install-diagnosed worker count: only when a calibration cache exists
    // (written once by `tuning::install_diagnose`); never measures here.
    #[cfg(feature = "tuning")]
    if let Some((cores, calib)) = crate::tuning::cached_calibration() {
        let est = estimate(&sym);
        let t = crate::tuning::recommend_threads_cost_model(&est, &calib, 0, cores);
        s.threads = crate::numeric::multifrontal_ldlt::Threads::Fixed(t);
    }
    Ok((sym, s))
}

/// A fixed-size array of independently written cells: each index is written by
/// exactly one owner (disjoint indices) and read only after a happens-before
/// barrier (subtree join / refcount Acquire-Release). Centralizes the
/// `Vec<UnsafeCell<V>>` pattern of the left-looking emit state.
pub(crate) struct Cells<V>(Vec<std::cell::UnsafeCell<V>>);

// SAFETY: disjoint-index writes; cross-thread visibility is the caller's
// barrier (see the type doc).
unsafe impl<V: Send> Sync for Cells<V> {}

impl<V: Default> Cells<V> {
    /// `n` default-initialized cells (for payloads without a cheap `Clone`).
    pub fn new_default(n: usize) -> Self {
        Cells(
            (0..n)
                .map(|_| std::cell::UnsafeCell::new(V::default()))
                .collect(),
        )
    }
}

impl<V: Clone> Cells<V> {
    pub fn new(n: usize, init: V) -> Self {
        Cells(
            (0..n)
                .map(|_| std::cell::UnsafeCell::new(init.clone()))
                .collect(),
        )
    }
}

impl<V> Cells<V> {
    /// SAFETY: `i` is this caller's exclusively owned index.
    #[inline]
    pub unsafe fn set(&self, i: usize, v: V) {
        *self.0[i].get() = v;
    }
    /// SAFETY: the write to `i` happened-before this read.
    #[inline]
    pub unsafe fn get(&self, i: usize) -> &V {
        &*self.0[i].get()
    }
    /// SAFETY: as [`set`](Self::set) - exclusive owner, e.g. for in-place take.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    pub unsafe fn get_mut(&self, i: usize) -> &mut V {
        &mut *self.0[i].get()
    }
}

/// Consumer refcounts (free a panel when its count reaches zero) and the
/// per-supernode first elimination position - the shared head of both
/// left-looking emit states.
pub(crate) fn emit_refcount_offsets(
    sym: &crate::symbolic::SymbolicFactorization,
    sched: &LlSchedule,
) -> (Vec<std::sync::atomic::AtomicUsize>, Vec<usize>) {
    use std::sync::atomic::AtomicUsize;
    let nsuper = sym.supernodes.len();
    let mut refcount: Vec<AtomicUsize> = (0..nsuper).map(|_| AtomicUsize::new(0)).collect();
    for s in 0..nsuper {
        for &k in sched.updaters(s) {
            *refcount[k as usize].get_mut() += 1;
        }
    }
    let mut e_offset = vec![0usize; nsuper];
    let mut acc = 0usize;
    for (s, snode) in sym.supernodes.iter().enumerate() {
        e_offset[s] = acc;
        acc += snode.ncol;
    }
    (refcount, e_offset)
}

/// Cached scatter program for the numeric phase's permuted matrix (the KLU
/// pattern applied to the supernodal twins): the permuted CSC *structure* is a
/// pure function of the analyzed pattern and the fill-reducing permutation, so
/// it is built once per symbolic analysis; every (re)factorization then only
/// scatters the new values through `pos` - one linear pass, no counting sort,
/// no per-column sorting.
pub(crate) struct PermScatter {
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    /// `pos[k]` = slot of original entry `k` in the permuted values array.
    pub pos: Vec<usize>,
}

impl PermScatter {
    /// Build for the symmetric lower-triangle fold `P^T A P` (LDL^T path):
    /// original entry `(i, j)` lands at permuted `(max(gi, gj), min(gi, gj))`
    /// with `g = perm_inv[*]`. Columns come out row-sorted.
    pub fn build_lower(
        n: usize,
        a_col_ptr: &[usize],
        a_row_idx: &[usize],
        perm_inv: &[usize],
    ) -> Self {
        Self::build_with(
            n,
            a_col_ptr,
            a_row_idx,
            |gi, gj| {
                if gi >= gj {
                    (gi, gj)
                } else {
                    (gj, gi)
                }
            },
            perm_inv,
        )
    }

    /// Build for the full (unfolded) permutation `P^T A P` (LU path).
    pub fn build_full(
        n: usize,
        a_col_ptr: &[usize],
        a_row_idx: &[usize],
        perm_inv: &[usize],
    ) -> Self {
        Self::build_with(n, a_col_ptr, a_row_idx, |gi, gj| (gi, gj), perm_inv)
    }

    /// Build for the transpose of the full permutation, `(P^T A P)^T` (the LU
    /// path's `a_perm_t`): entry `(i, j)` lands at `(gj, gi)`.
    pub fn build_full_transposed(
        n: usize,
        a_col_ptr: &[usize],
        a_row_idx: &[usize],
        perm_inv: &[usize],
    ) -> Self {
        Self::build_with(n, a_col_ptr, a_row_idx, |gi, gj| (gj, gi), perm_inv)
    }

    fn build_with(
        n: usize,
        a_col_ptr: &[usize],
        a_row_idx: &[usize],
        target: impl Fn(usize, usize) -> (usize, usize),
        perm_inv: &[usize],
    ) -> Self {
        let nnz = a_row_idx.len();
        // Pass 1: count entries per target column.
        let mut col_ptr = vec![0usize; n + 1];
        for (j, &gj) in perm_inv.iter().enumerate() {
            for k in a_col_ptr[j]..a_col_ptr[j + 1] {
                let (_, c) = target(perm_inv[a_row_idx[k]], gj);
                col_ptr[c + 1] += 1;
            }
        }
        for c in 0..n {
            col_ptr[c + 1] += col_ptr[c];
        }
        // Pass 2: place (row, original-entry) pairs per column.
        let mut cursor = col_ptr[..n].to_vec();
        let mut pairs: Vec<(usize, usize)> = vec![(0, 0); nnz];
        for (j, &gj) in perm_inv.iter().enumerate() {
            for k in a_col_ptr[j]..a_col_ptr[j + 1] {
                let (r, c) = target(perm_inv[a_row_idx[k]], gj);
                let p = cursor[c];
                cursor[c] += 1;
                pairs[p] = (r, k);
            }
        }
        // Pass 3: sort each column by row (once, at build time) and freeze the
        // structure + position map.
        let mut row_idx = vec![0usize; nnz];
        let mut pos = vec![0usize; nnz];
        for c in 0..n {
            let (s, e) = (col_ptr[c], col_ptr[c + 1]);
            pairs[s..e].sort_unstable_by_key(|&(r, _)| r);
            for (p, &(r, k)) in (s..e).zip(pairs[s..e].iter()) {
                row_idx[p] = r;
                pos[k] = p;
            }
        }
        PermScatter {
            col_ptr,
            row_idx,
            pos,
        }
    }

    /// Scatter a fresh value set through the frozen structure: one linear pass.
    /// `scale(k)` maps the original entry's value (e.g. equilibration); pass
    /// the identity for the plain permutation.
    pub fn scatter<T: crate::scalar::Scalar>(
        &self,
        values: &[T],
        mut map: impl FnMut(usize, T) -> T,
    ) -> Vec<T> {
        debug_assert_eq!(values.len(), self.pos.len());
        let mut out = vec![T::zero(); values.len()];
        for (k, &p) in self.pos.iter().enumerate() {
            out[p] = map(k, values[k]);
        }
        out
    }
}

/// Narrow index type of the left-looking streams (row structures, updater
/// lists, the gloc scatter table): the supernodal sweeps are index-bound, and
/// halving the index traffic was measured as a real win on the KLU path (the
/// same lever, see the M3 audit). The row values are global column positions
/// `< n` and the updater values supernode ids `< nsuper`; both fit u32 for
/// every reachable problem size (an `n >= 2^32` factor would not fit any
/// memory this library targets - asserted at schedule build).
pub(crate) type Li = u32;

/// Pattern-only left-looking schedule, built once per symbolic analysis and
/// shared by the LDLT/LU drivers and the a-priori memory estimators (it was
/// previously rebuilt by each of them, per factorization): the per-supernode
/// row structures and the updater lists, both in flat CSR-like storage with
/// narrow [`Li`] values.
pub(crate) struct LlSchedule {
    rs_off: Vec<usize>,
    rs: Vec<Li>,
    ul_off: Vec<usize>,
    ul: Vec<Li>,
}

impl LlSchedule {
    /// Rows of supernode `s`: `rows(s)[0..ncol]` are its eliminated columns
    /// `first_col..first_col+ncol`; `rows(s)[ncol..]` the sorted
    /// below-diagonal fill rows (the multifrontal assembly, value-free).
    #[inline]
    pub fn rows(&self, s: usize) -> &[Li] {
        &self.rs[self.rs_off[s]..self.rs_off[s + 1]]
    }

    /// Updaters of supernode `s`: every factored `k` whose off-diagonal rows
    /// hit `s`'s column run (each exactly once, ascending).
    #[inline]
    pub fn updaters(&self, s: usize) -> &[Li] {
        &self.ul[self.ul_off[s]..self.ul_off[s + 1]]
    }

    pub fn build(sym: &crate::symbolic::SymbolicFactorization) -> Self {
        let nsuper = sym.supernodes.len();
        assert!(
            sym.n <= Li::MAX as usize,
            "left-looking schedule requires n < 2^32"
        );
        // Row structures: own columns ++ sorted union of the column patterns'
        // trailing rows and the children's off-diagonal rows.
        let mut rs_off = Vec::with_capacity(nsuper + 1);
        rs_off.push(0usize);
        let mut rs: Vec<Li> = Vec::new();
        let mut trailing: Vec<Li> = Vec::new();
        for s in 0..nsuper {
            let snode = &sym.supernodes[s];
            let own_last = snode.first_col + snode.ncol;
            trailing.clear();
            for j in snode.first_col..own_last {
                for k in sym.permuted_pattern.col_ptr[j]..sym.permuted_pattern.col_ptr[j + 1] {
                    let r = sym.permuted_pattern.row_idx[k];
                    if r >= own_last {
                        trailing.push(r as Li);
                    }
                }
            }
            for &ch in &snode.children {
                let nck = sym.supernodes[ch].ncol;
                for &r in &rs[rs_off[ch] + nck..rs_off[ch + 1]] {
                    if r as usize >= own_last {
                        trailing.push(r);
                    }
                }
            }
            trailing.sort_unstable();
            trailing.dedup();
            rs.extend((snode.first_col..own_last).map(|c| c as Li));
            rs.extend_from_slice(&trailing);
            rs_off.push(rs.len());
        }

        // Updater lists: `k` updates `s` iff one of `k`'s off-diagonal rows is
        // an eliminated column of `s`. Two counting passes over the flat rows.
        let mut col_to_snode = vec![0usize; sym.n];
        for (s, snode) in sym.supernodes.iter().enumerate() {
            col_to_snode[snode.first_col..snode.first_col + snode.ncol].fill(s);
        }
        let mut ul_off = vec![0usize; nsuper + 1];
        let each_hit = |mut f: Box<dyn FnMut(usize, usize) + '_>| {
            for k in 0..nsuper {
                let nck = sym.supernodes[k].ncol;
                let mut last = usize::MAX;
                for &r in &rs[rs_off[k] + nck..rs_off[k + 1]] {
                    let s = col_to_snode[r as usize];
                    if s != last {
                        f(s, k);
                        last = s;
                    }
                }
            }
        };
        each_hit(Box::new(|s, _k| ul_off[s + 1] += 1));
        for s in 0..nsuper {
            ul_off[s + 1] += ul_off[s];
        }
        let mut cursor = ul_off[..nsuper].to_vec();
        let mut ul = vec![0 as Li; ul_off[nsuper]];
        each_hit(Box::new(|s, k| {
            ul[cursor[s]] = k as Li;
            cursor[s] += 1;
        }));
        LlSchedule {
            rs_off,
            rs,
            ul_off,
            ul,
        }
    }
}

/// The left-looking assembly-forest recursion shared by the LDLT/LU twins:
/// factor every child subtree concurrently, then this node, then compact and
/// free every descendant whose last consumer this node was (refcount->0), and
/// the node itself if nothing above consumes it. `factor_node` and `emit_free`
/// carry the path-specific kernels; everything else is the shared schedule.
pub(crate) fn ll_subtree(
    s: usize,
    sym: &crate::symbolic::SymbolicFactorization,
    sched: &LlSchedule,
    refcount: &[std::sync::atomic::AtomicUsize],
    factor_node: &(dyn Fn(usize) -> Result<(), RslabError> + Sync),
    emit_free: &(dyn Fn(usize) + Sync),
) -> Result<(), RslabError> {
    use rayon::prelude::*;
    use std::sync::atomic::Ordering;
    sym.supernodes[s]
        .children
        .par_iter()
        .map(|&ch| ll_subtree(ch, sym, sched, refcount, factor_node, emit_free))
        .collect::<Result<Vec<()>, _>>()?;
    factor_node(s)?;
    // Disjoint `k`, so the wide top-of-tree free runs in parallel.
    const FREE_PAR: usize = 64;
    let free = |k: usize| {
        if refcount[k].fetch_sub(1, Ordering::AcqRel) == 1 {
            emit_free(k);
        }
    };
    if sched.updaters(s).len() >= FREE_PAR {
        sched.updaters(s).par_iter().for_each(|&k| free(k as usize));
    } else {
        for &k in sched.updaters(s) {
            free(k as usize);
        }
    }
    if refcount[s].load(Ordering::Relaxed) == 0 {
        emit_free(s);
    }
    Ok(())
}

/// The multifrontal assembly-forest recursion shared by the LDLT/LU twins:
/// factor every child subtree concurrently, factor this node from the
/// children's fronts, free the children's contribution blocks the moment they
/// have been extend-added (the CB stack is the dominant transient - keeping it
/// to the end OOMed on large fronts), and flatten the subtree's node results.
/// Returns `(own, [(supernode, node)...])` for the parent / forest scatter.
pub(crate) type NodeFactorFn<'a, N> = dyn Fn(usize, &[&N]) -> Result<N, RslabError> + Sync + 'a;

pub(crate) fn mf_subtree<N: Send>(
    s: usize,
    sym: &crate::symbolic::SymbolicFactorization,
    factor_one: &NodeFactorFn<'_, N>,
    free_contrib: &(dyn Fn(&mut N) + Sync),
) -> Result<(N, Vec<(usize, N)>), RslabError> {
    use rayon::prelude::*;
    let children = &sym.supernodes[s].children;
    let mut outs: Vec<(N, Vec<(usize, N)>)> = children
        .par_iter()
        .map(|&ch| mf_subtree(ch, sym, factor_one, free_contrib))
        .collect::<Result<Vec<_>, _>>()?;
    let nf = {
        let child_refs: Vec<&N> = outs.iter().map(|(own, _)| own).collect();
        factor_one(s, &child_refs)?
    };
    for (own, _) in outs.iter_mut() {
        free_contrib(own);
    }
    let mut subtree = Vec::new();
    for (i, (own, rest)) in outs.into_iter().enumerate() {
        subtree.push((children[i], own));
        subtree.extend(rest);
    }
    Ok((nf, subtree))
}

/// Roots of the assembly forest: supernodes that are no node's child.
pub(crate) fn forest_roots(sym: &crate::symbolic::SymbolicFactorization) -> Vec<usize> {
    let nsuper = sym.supernodes.len();
    let mut is_child = vec![false; nsuper];
    for snode in &sym.supernodes {
        for &ch in &snode.children {
            is_child[ch] = true;
        }
    }
    (0..nsuper).filter(|&s| !is_child[s]).collect()
}

/// cmod pre-pass shared by the LDLT/LU node kernels: locate each updater's
/// landing range in the panel of supernode `s` (`[p0, p1)` of its off-diagonal
/// rows) and total the update flops - the fork/tiling dispatch input. With
/// `count_u`, the U-side rows beyond the landing range are counted too (the
/// LU twin updates both `L` and `U12`).
pub(crate) fn cmod_spans(
    sym: &crate::symbolic::SymbolicFactorization,
    sched: &LlSchedule,
    s: usize,
    first: usize,
    ncol: usize,
    count_u: bool,
) -> (Vec<(usize, usize, usize)>, usize) {
    let mut spans: Vec<(usize, usize, usize)> = Vec::with_capacity(sched.updaters(s).len());
    let mut cmod_flops: usize = 0;
    for &kk in sched.updaters(s) {
        let kk = kk as usize;
        let nck = sym.supernodes[kk].ncol;
        let ok = &sched.rows(kk)[nck..];
        let nok = ok.len();
        let p0 = ok.partition_point(|&g| (g as usize) < first);
        let p1 = ok.partition_point(|&g| (g as usize) < first + ncol);
        let npk = p1 - p0;
        if npk == 0 {
            continue;
        }
        let flop = (nok - p0) * npk * nck;
        cmod_flops += flop;
        if count_u {
            cmod_flops += npk * (nok - p1) * nck;
        }
        spans.push((kk, p0, p1));
    }
    (spans, cmod_flops)
}
