//! The pattern-only left-looking schedule: per-supernode row structures and
//! updater lists, and the consumer refcounts that free a panel after its
//! last updater.

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
    /// below-diagonal rows of the panel.
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
