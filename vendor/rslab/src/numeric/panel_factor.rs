//! The supernodal panel form of a triangular factor: the one representation
//! the numeric drivers write and the solves read.
//!
//! A factor `L` (unit lower triangular, or a general lower triangular `U^T`)
//! is stored per supernode as one dense column-major panel of `(w + m) x w`
//! entries with leading dimension `w + m`: `w` columns of the supernode, the
//! `w x w` diagonal block on top (only its lower triangle is meaningful, the
//! unit diagonal is implicit for a unit factor), the `m` off-block rows below
//! in ascending elimination order. The row list is shared by the panel's
//! columns, so the index overhead is `m` words per supernode rather than one
//! per entry, and the values are exactly the dense panels the factorization
//! kernels produce -- a left-looking driver hands its panel over without a
//! copy, and the solves run BLAS-3 style on the same memory. Compared with a
//! compressed-column factor with one `usize` index per entry (24 bytes per
//! complex entry, and a second copy for the solve layout) this is the whole
//! factor at 16 bytes per complex entry, once.
//!
//! [`PanelFactor::to_csc`] materializes the compressed-column form on demand
//! for the reference solves and the public CSC factor types.

use crate::scalar::Scalar;

/// A lower triangular factor in supernodal panel form (see the module docs).
#[derive(Clone, Debug)]
pub struct PanelFactor<T> {
    /// Matrix dimension.
    pub n: usize,
    /// Supernode column starts in elimination order, `ns + 1` entries.
    pub sn_col: Vec<u32>,
    /// Off-block row indices per supernode, ascending.
    pub rows: Vec<Vec<u32>>,
    /// Per supernode the `(w + m) x w` column-major panel (leading dimension
    /// `w + m`); entries above the diagonal of the diagonal block are unused.
    pub panels: Vec<Vec<T>>,
}

impl<T: Scalar> PanelFactor<T> {
    /// An empty factor of dimension 0.
    pub fn empty() -> Self {
        PanelFactor {
            n: 0,
            sn_col: vec![0],
            rows: Vec::new(),
            panels: Vec::new(),
        }
    }

    /// Number of supernodes.
    #[inline]
    pub fn n_supernodes(&self) -> usize {
        self.sn_col.len() - 1
    }

    /// `(c0, w, m)` of supernode `s`: first column, column count, off-block rows.
    #[inline]
    pub fn shape(&self, s: usize) -> (usize, usize, usize) {
        let c0 = self.sn_col[s] as usize;
        let w = self.sn_col[s + 1] as usize - c0;
        (c0, w, self.rows[s].len())
    }

    /// Structural entry count of the factor: the lower triangles of the
    /// diagonal blocks (diagonal included) plus the off-block rows.
    pub fn nnz(&self) -> usize {
        (0..self.n_supernodes())
            .map(|s| {
                let (_, w, m) = self.shape(s);
                w * (w + 1) / 2 + m * w
            })
            .sum()
    }

    /// Bytes of the panel storage (values plus row indices).
    pub fn bytes(&self) -> usize {
        let vals: usize = self.panels.iter().map(|p| p.len()).sum();
        let rows: usize = self.rows.iter().map(|r| r.len()).sum();
        vals * std::mem::size_of::<T>() + rows * 4
    }

    /// The compressed-column form `(col_ptr, row_idx, values)` with the rows of
    /// every column ascending. For a unit factor the explicit unit diagonal
    /// leads each column; exact zeros are dropped, as a sparse factor would
    /// never have stored them.
    pub fn to_csc(&self, unit: bool) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        let n = self.n;
        let mut col_ptr = Vec::with_capacity(n + 1);
        col_ptr.push(0);
        let mut row_idx: Vec<usize> = Vec::with_capacity(self.nnz());
        let mut values: Vec<T> = Vec::with_capacity(self.nnz());
        let zero = T::zero();
        for s in 0..self.n_supernodes() {
            let (c0, w, m) = self.shape(s);
            let ld = w + m;
            let panel = &self.panels[s];
            let rows = &self.rows[s];
            for k in 0..w {
                let col = &panel[k * ld..(k + 1) * ld];
                if unit {
                    row_idx.push(c0 + k);
                    values.push(T::one());
                } else {
                    row_idx.push(c0 + k);
                    values.push(col[k]);
                }
                for (i, &v) in col.iter().enumerate().take(w).skip(k + 1) {
                    if v != zero {
                        row_idx.push(c0 + i);
                        values.push(v);
                    }
                }
                for (i, &r) in rows.iter().enumerate() {
                    let v = col[w + i];
                    if v != zero {
                        row_idx.push(r as usize);
                        values.push(v);
                    }
                }
                col_ptr.push(row_idx.len());
            }
        }
        debug_assert_eq!(col_ptr.len(), n + 1, "every column emitted once");
        (col_ptr, row_idx, values)
    }

    /// Build from a lower-triangular CSC factor whose columns list the
    /// diagonal first and the rows ascending. `supernode_ptr` gives the column
    /// partition (`ns + 1` entries, first 0, last `n`); an unusable partition
    /// is replaced by maximal runs of nested columns.
    pub fn from_csc(
        n: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[T],
        supernode_ptr: &[usize],
    ) -> Self {
        const NONE: u32 = u32::MAX;
        let zero = T::zero();
        let mut sn_col: Vec<u32> = vec![0];
        let known = supernode_ptr.len() >= 2
            && supernode_ptr[0] == 0
            && supernode_ptr.last().copied() == Some(n)
            && supernode_ptr.windows(2).all(|p| p[0] < p[1]);
        if known {
            for p in supernode_ptr.windows(2) {
                sn_col.push(p[1] as u32);
            }
        } else {
            // mark[r] = c0 of the run whose first column has row r
            let mut mark = vec![NONE; n];
            let mut j = 0;
            while j < n {
                let c0 = j;
                for &r in &row_idx[col_ptr[c0]..col_ptr[c0 + 1]] {
                    mark[r] = c0 as u32;
                }
                let mut c1 = c0 + 1;
                while c1 < n {
                    let fits = row_idx[col_ptr[c1]..col_ptr[c1 + 1]]
                        .iter()
                        .all(|&r| mark[r] == c0 as u32 || (r >= c0 && r <= c1));
                    if !fits {
                        break;
                    }
                    c1 += 1;
                }
                sn_col.push(c1 as u32);
                j = c1;
            }
        }
        let ns = sn_col.len() - 1;
        let mut rows: Vec<Vec<u32>> = Vec::with_capacity(ns);
        let mut panels: Vec<Vec<T>> = Vec::with_capacity(ns);
        let mut pos = vec![NONE; n];
        for s in 0..ns {
            let (c0, c1) = (sn_col[s] as usize, sn_col[s + 1] as usize);
            let w = c1 - c0;
            let mut sn_rows: Vec<u32> = Vec::new();
            for c in c0..c1 {
                for &r in &row_idx[col_ptr[c]..col_ptr[c + 1]] {
                    if r >= c1 && pos[r] == NONE {
                        pos[r] = 0;
                        sn_rows.push(r as u32);
                    }
                }
            }
            sn_rows.sort_unstable();
            let m = sn_rows.len();
            let ld = w + m;
            for (i, &r) in sn_rows.iter().enumerate() {
                pos[r as usize] = (w + i) as u32;
            }
            for (k, p) in pos[c0..c1].iter_mut().enumerate() {
                *p = k as u32;
            }
            let mut panel = vec![zero; ld * w];
            for c in c0..c1 {
                let k = c - c0;
                let col = &mut panel[k * ld..(k + 1) * ld];
                for e in col_ptr[c]..col_ptr[c + 1] {
                    let r = row_idx[e];
                    debug_assert_ne!(pos[r], NONE, "row outside the supernode panel");
                    col[pos[r] as usize] = values[e];
                }
            }
            for &r in &sn_rows {
                pos[r as usize] = NONE;
            }
            for p in &mut pos[c0..c1] {
                *p = NONE;
            }
            rows.push(sn_rows);
            panels.push(panel);
        }
        PanelFactor {
            n,
            sn_col,
            rows,
            panels,
        }
    }
}

/// Reorder the rows of a column-major `(ld) x w` panel in place so that row
/// `i` of the result is row `order[i]` of the input (`order` is a permutation
/// of `0..ld`). One column of scratch, no allocation per call beyond it.
pub fn permute_panel_rows<T: Copy>(
    panel: &mut [T],
    ld: usize,
    w: usize,
    order: &[u32],
    tmp: &mut Vec<T>,
) {
    debug_assert_eq!(order.len(), ld);
    debug_assert_eq!(panel.len(), ld * w);
    if order.iter().enumerate().all(|(i, &o)| o as usize == i) {
        return;
    }
    tmp.clear();
    for k in 0..w {
        let col = &mut panel[k * ld..(k + 1) * ld];
        tmp.extend_from_slice(col);
        for (i, &o) in order.iter().enumerate() {
            col[i] = tmp[o as usize];
        }
        tmp.clear();
    }
}

/// One supernode's emitted panel: its off-block rows (elimination indices,
/// ascending) and the `(w + m) x w` column-major values.
pub(crate) struct PanelOut<T> {
    pub rows: Vec<u32>,
    pub panel: Vec<T>,
    /// Structural slots (diagonal excluded) holding an exact zero: numeric
    /// cancellation, the symmetrized pattern of an unsymmetric matrix, or
    /// `drop_tol`. The panel keeps the slots; `nnz() - zeros` is the stored
    /// nonzero count a sparse factor would report.
    pub zeros: usize,
}
impl<T> Default for PanelOut<T> {
    fn default() -> Self {
        PanelOut {
            rows: Vec::new(),
            panel: Vec::new(),
            zeros: 0,
        }
    }
}

/// Finish one supernode's panel for the store: rows `0..w` are the
/// supernode's own columns in elimination order, the `m` rows below are
/// permuted into ascending elimination order (`e_rows[i]` is the elimination
/// index of panel row `w + i` on entry), for an LDL^T factor the `(p+1, p)`
/// entry of each 2x2 pivot is cleared (that coupling lives in `D`), and
/// `drop_tol` zeroes the entries below `tau * max|col|` of their column
/// (the diagonal excluded). Consumes `e_rows`.
pub(crate) fn finish_panel<T: Scalar>(
    mut panel: Vec<T>,
    w: usize,
    mut e_rows: Vec<u32>,
    two_by_two: Option<&[bool]>,
    drop_tol: Option<f64>,
) -> PanelOut<T> {
    let m = e_rows.len();
    let ld = w + m;
    debug_assert_eq!(panel.len(), ld * w);
    let sorted = e_rows.windows(2).all(|p| p[0] < p[1]);
    if !sorted {
        let mut order: Vec<u32> = (0..m as u32).collect();
        order.sort_unstable_by_key(|&i| e_rows[i as usize]);
        let full: Vec<u32> = (0..w as u32)
            .chain(order.iter().map(|&i| w as u32 + i))
            .collect();
        let mut tmp = Vec::with_capacity(ld);
        permute_panel_rows(&mut panel, ld, w, &full, &mut tmp);
        e_rows = order.iter().map(|&i| e_rows[i as usize]).collect();
    }
    let zero = T::zero();
    if let Some(two_by_two) = two_by_two {
        for p in 0..w {
            if two_by_two[p] && p + 1 < w {
                panel[p * ld + p + 1] = zero;
            }
        }
    }
    if let Some(tau) = drop_tol {
        for p in 0..w {
            let col = &mut panel[p * ld..(p + 1) * ld];
            let colmax = col[p + 1..]
                .iter()
                .map(|v| v.magnitude())
                .fold(0.0, f64::max);
            let thresh = tau * colmax;
            for v in col[p + 1..].iter_mut() {
                if v.magnitude() < thresh {
                    *v = zero;
                }
            }
        }
    }
    let zeros = (0..w)
        .map(|p| {
            panel[p * ld + p + 1..(p + 1) * ld]
                .iter()
                .filter(|&&v| v == zero)
                .count()
        })
        .sum();
    PanelOut {
        rows: e_rows,
        panel,
        zeros,
    }
}

/// Assemble the per-supernode panels (elimination order, empty supernodes
/// skipped) into the factor.
pub(crate) fn assemble_panels<T: Scalar>(
    n: usize,
    ncols: impl Iterator<Item = usize>,
    mut outs: impl FnMut(usize) -> PanelOut<T>,
) -> (PanelFactor<T>, usize) {
    let mut sn_col: Vec<u32> = vec![0];
    let mut rows = Vec::new();
    let mut panels = Vec::new();
    let mut zeros = 0usize;
    for (s, w) in ncols.enumerate() {
        if w == 0 {
            continue;
        }
        let out = outs(s);
        debug_assert_eq!(out.panel.len(), (w + out.rows.len()) * w);
        sn_col.push(sn_col.last().copied().unwrap_or(0) + w as u32);
        rows.push(out.rows);
        panels.push(out.panel);
        zeros += out.zeros;
    }
    debug_assert_eq!(sn_col.last().copied(), Some(n as u32));
    (
        PanelFactor {
            n,
            sn_col,
            rows,
            panels,
        },
        zeros,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csc_round_trip_keeps_values() {
        // Two supernodes: columns {0,1} with off-block rows {2,3}, column {2},
        // column {3}.
        let col_ptr = vec![0, 4, 7, 9, 10];
        let row_idx = vec![0, 1, 2, 3, 1, 2, 3, 2, 3, 3];
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 5.0, 6.0, 1.0, 7.0, 1.0];
        let f = PanelFactor::from_csc(4, &col_ptr, &row_idx, &values, &[0, 2, 3, 4]);
        assert_eq!(f.n_supernodes(), 3);
        assert_eq!(f.rows[0], vec![2, 3]);
        assert_eq!(f.panels[0].len(), 8);
        let (cp, ri, v) = f.to_csc(true);
        assert_eq!(cp, col_ptr);
        assert_eq!(ri, row_idx);
        assert_eq!(v, values);
        assert_eq!(f.nnz(), 10);
    }

    #[test]
    fn row_permutation_in_place() {
        // 3 x 2 panel, column-major.
        let mut panel = vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
        let mut tmp = Vec::new();
        permute_panel_rows(&mut panel, 3, 2, &[2, 0, 1], &mut tmp);
        assert_eq!(panel, vec![2.0, 0.0, 1.0, 12.0, 10.0, 11.0]);
    }
}
