//! The exact supernodal structure of `L` and `U`.
//!
//! The analysis runs on the symmetrized pattern `A + A^T`, and its row
//! structures bound both factors. On a structurally unsymmetric matrix they
//! bound them loosely: the MoM near-field systems carry 16 to 19 percent of
//! the panel entries in off-block rows that stay zero. The exact structures
//! follow from the left-looking updates themselves. Descendant `k` adds its
//! `L` rows to `s` when its `U` columns reach `s`'s columns, and its `U`
//! columns when its `L` rows do, so per supernode
//!
//! - the `L` rows of `s`: rows of `B` below the diagonal block in `s`'s
//!   columns, and the rows past `s` of every `k` whose `U` columns reach `s`;
//! - the `U` columns of `s`: columns of `B` past the block in `s`'s rows, and
//!   the columns past `s` of every `k` whose `L` rows reach `s`.
//!
//! Pivoting stays inside a supernode's fully-summed block, so row
//! interchanges permute the block rows among themselves and leave both sets
//! unchanged: they are exact up to numerical cancellation. Both are subsets
//! of the symmetric structure, and every contributing `k` is among `s`'s
//! symmetric updaters, which the construction walks.

use crate::numeric::supernodal::{Li, LlSchedule};
use crate::symbolic::SymbolicFactorization;

pub(super) struct LuStructure {
    l_off: Vec<usize>,
    l: Vec<Li>,
    u_off: Vec<usize>,
    u: Vec<Li>,
}

impl LuStructure {
    /// The structure of an empty analysis.
    pub fn empty() -> Self {
        LuStructure {
            l_off: vec![0],
            l: Vec::new(),
            u_off: vec![0],
            u: Vec::new(),
        }
    }

    /// Rows of `s`'s `L` panel: its own columns, then the off-block rows
    /// ascending (the layout of `LlSchedule::rows`).
    #[inline]
    pub fn rows_l(&self, s: usize) -> &[Li] {
        &self.l[self.l_off[s]..self.l_off[s + 1]]
    }

    /// Columns of `s`'s `U` panel: its own columns, then the `U12` columns
    /// ascending.
    #[inline]
    pub fn cols_u(&self, s: usize) -> &[Li] {
        &self.u[self.u_off[s]..self.u_off[s + 1]]
    }

    /// From the pattern of `A` (`col_ptr`, `row_idx`), the row matching
    /// (`row_map[r]`: the row of `B` that row `r` of `A` becomes) and the
    /// analysis of `B + B^T`.
    pub fn build(
        col_ptr: &[usize],
        row_idx: &[usize],
        row_map: Option<&[usize]>,
        sym: &SymbolicFactorization,
        sched: &LlSchedule,
    ) -> Self {
        let n = sym.n;
        let nsuper = sym.supernodes.len();
        let mut snode_of = vec![0usize; n];
        for (s, sn) in sym.supernodes.iter().enumerate() {
            snode_of[sn.first_col..sn.first_col + sn.ncol].fill(s);
        }
        let end = |s: usize| sym.supernodes[s].first_col + sym.supernodes[s].ncol;
        // Seeds: the entries of `B` below or right of their diagonal block.
        let (mut lseed, mut useed): (Vec<Vec<Li>>, Vec<Vec<Li>>) =
            (vec![Vec::new(); nsuper], vec![Vec::new(); nsuper]);
        for j in 0..n {
            let gj = sym.perm_inv[j];
            for &r in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
                let gi = sym.perm_inv[row_map.map_or(r, |m| m[r])];
                let (sj, si) = (snode_of[gj], snode_of[gi]);
                if gi >= end(sj) {
                    lseed[sj].push(gi as Li);
                } else if gj >= end(si) {
                    useed[si].push(gj as Li);
                }
            }
        }
        let mut st = Self::empty();
        // `s`'s columns `[first, last)` hit by the ascending list `v`.
        let hits = |v: &[Li], first: usize, last: usize| {
            let p = v.partition_point(|&g| (g as usize) < first);
            p < v.len() && (v[p] as usize) < last
        };
        for s in 0..nsuper {
            let (first, last) = (sym.supernodes[s].first_col, end(s));
            let (mut l, mut u) = (std::mem::take(&mut lseed[s]), std::mem::take(&mut useed[s]));
            for &k in sched.updaters(s) {
                let k = k as usize;
                let nck = sym.supernodes[k].ncol;
                let (lk, uk) = (&st.rows_l(k)[nck..], &st.cols_u(k)[nck..]);
                if hits(uk, first, last) {
                    l.extend(lk.iter().filter(|&&g| g as usize >= last));
                }
                if hits(lk, first, last) {
                    u.extend(uk.iter().filter(|&&g| g as usize >= last));
                }
            }
            for (v, out, off) in [
                (&mut l, &mut st.l, &mut st.l_off),
                (&mut u, &mut st.u, &mut st.u_off),
            ] {
                v.sort_unstable();
                v.dedup();
                out.extend((first..last).map(|c| c as Li));
                out.extend_from_slice(v);
                off.push(out.len());
            }
            debug_assert!(
                {
                    let ncol = last - first;
                    let sym_rows = &sched.rows(s)[ncol..];
                    let inside = |v: &[Li]| v.iter().all(|g| sym_rows.binary_search(g).is_ok());
                    inside(&st.rows_l(s)[ncol..]) && inside(&st.cols_u(s)[ncol..])
                },
                "the exact structure lies in the symmetric one"
            );
        }
        st
    }
}
