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
//! kernels produce. All panels live in **one buffer in supernode order** (the
//! [`PanelArena`] the drivers factor into), so a sweep streams through memory
//! the way the elimination tree is walked -- a left-looking driver hands its
//! panel over without a copy, and the solves run BLAS-3 style on the same
//! memory. Compared with a compressed-column factor with one `usize` index
//! per entry (24 bytes per complex entry, and a second copy for the solve
//! layout) this is the whole factor at 16 bytes per complex entry, once.
//!
//! [`PanelFactor::to_csc`] materializes the compressed-column form on demand
//! for the reference solves and the public CSC factor types.

use crate::numeric::supernodal::PanelPtr;
use crate::scalar::Scalar;
use rayon::prelude::*;

/// A lower triangular factor in supernodal panel form (see the module docs).
#[derive(Clone, Debug)]
pub struct PanelFactor<T> {
    /// Matrix dimension.
    pub n: usize,
    /// Supernode column starts in elimination order, `ns + 1` entries.
    pub sn_col: Vec<u32>,
    /// Off-block row indices per supernode, ascending.
    pub rows: Vec<Vec<u32>>,
    /// Panel starts into `vals`, `ns + 1` entries.
    pub val_ptr: Vec<usize>,
    /// The panels back to back in supernode order: supernode `s` is the
    /// `(w + m) x w` column-major block `vals[val_ptr[s]..val_ptr[s + 1]]`
    /// (leading dimension `w + m`); entries above the diagonal of the
    /// diagonal block are unused.
    pub vals: Vec<T>,
}

impl<T: Scalar> PanelFactor<T> {
    /// An empty factor of dimension 0.
    pub fn empty() -> Self {
        PanelFactor {
            n: 0,
            sn_col: vec![0],
            rows: Vec::new(),
            val_ptr: vec![0],
            vals: Vec::new(),
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

    /// The panel of supernode `s`.
    #[inline]
    pub fn panel(&self, s: usize) -> &[T] {
        &self.vals[self.val_ptr[s]..self.val_ptr[s + 1]]
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
        let rows: usize = self.rows.iter().map(|r| r.len()).sum();
        self.vals.len() * std::mem::size_of::<T>() + rows * 4
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

/// The factor's buffer while a numeric driver fills it: one slot per
/// supernode of the analysis, sized `(w + m) x w` from the symbolic row
/// counts, back to back in supernode order. Each slot is written by the one
/// task that owns its supernode (the drivers factor straight into it), read
/// by the
/// tasks that update from it once it is published, and finished in place
/// ([`finish_panel`]). [`finish`](Self::finish) then closes the gaps left by
/// dropped rows and yields the [`PanelFactor`].
pub(crate) struct PanelArena<T> {
    /// Slot starts per supernode of the analysis, `nsuper + 1` entries.
    slot_ptr: Vec<usize>,
    vals: Vec<T>,
    base: PanelPtr<T>,
}

// SAFETY: slots are disjoint and each has a single writer before any reader
// (the drivers' publication discipline, see the type docs).
unsafe impl<T: Send> Sync for PanelArena<T> {}

impl<T: Scalar> PanelArena<T> {
    /// Allocate slots of the given sizes (entries per supernode, 0 for an
    /// empty supernode). The buffer is zero-initialized, so a slot starts as
    /// the zero panel the assembly expects.
    pub fn new(sizes: impl Iterator<Item = usize>) -> Self {
        let mut slot_ptr = vec![0usize];
        for len in sizes {
            slot_ptr.push(slot_ptr.last().copied().unwrap_or(0) + len);
        }
        let total = slot_ptr.last().copied().unwrap_or(0);
        // The whole factor, often hundreds of MB: zero it across the calling
        // pool rather than on one thread (41 ms serial on a 465 MB factor).
        let mut vals: Vec<T> = Vec::with_capacity(total);
        vals.spare_capacity_mut()
            .par_chunks_mut(1 << 16)
            .for_each(|chunk| {
                chunk.iter_mut().for_each(|v| {
                    v.write(T::zero());
                })
            });
        // SAFETY: every element of the first `total` was written above.
        unsafe { vals.set_len(total) };
        let base = PanelPtr(vals.as_mut_ptr());
        PanelArena {
            slot_ptr,
            vals,
            base,
        }
    }

    /// Entries of slot `s`.
    #[inline]
    pub fn slot_len(&self, s: usize) -> usize {
        self.slot_ptr[s + 1] - self.slot_ptr[s]
    }

    /// The slot of supernode `s` for writing.
    ///
    /// # Safety
    /// Only the owner of `s` may call this, and not while any reader holds a
    /// reference from [`slot`](Self::slot).
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slot_mut(&self, s: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.base.get().add(self.slot_ptr[s]), self.slot_len(s))
    }

    /// The slot of supernode `s` for reading.
    ///
    /// # Safety
    /// The owner must have published the slot (all writes done) before any
    /// reader calls this, and must not write again while readers exist.
    #[inline]
    pub unsafe fn slot(&self, s: usize) -> &[T] {
        std::slice::from_raw_parts(self.base.get().add(self.slot_ptr[s]), self.slot_len(s))
    }

    /// Close the arena into the factor: `ncols` gives the column count of
    /// every supernode of the analysis (0 skipped), `outs(s)` the finished
    /// panel's rows and compacted length. Slots are moved down in place to
    /// close the gaps of dropped rows (reads stay ahead of writes), so the
    /// factor's buffer is exact. Returns the factor and the zero-slot count.
    pub fn finish(
        mut self,
        n: usize,
        ncols: impl Iterator<Item = usize>,
        mut outs: impl FnMut(usize) -> PanelOut,
    ) -> (PanelFactor<T>, usize) {
        let mut sn_col: Vec<u32> = vec![0];
        let mut rows = Vec::new();
        let mut val_ptr = vec![0usize];
        let mut zeros = 0usize;
        let mut dst = 0usize;
        let mut moved = false;
        for (s, w) in ncols.enumerate() {
            if w == 0 {
                continue;
            }
            let out = outs(s);
            let src = self.slot_ptr[s];
            debug_assert_eq!(out.len, (w + out.rows.len()) * w);
            debug_assert!(out.len <= self.slot_len(s));
            if dst != src {
                self.vals.copy_within(src..src + out.len, dst);
                moved = true;
            }
            dst += out.len;
            sn_col.push(sn_col.last().copied().unwrap_or(0) + w as u32);
            rows.push(out.rows);
            val_ptr.push(dst);
            zeros += out.zeros;
        }
        debug_assert_eq!(sn_col.last().copied(), Some(n as u32));
        let mut vals = self.vals;
        if moved || dst < vals.len() {
            vals.truncate(dst);
            vals.shrink_to_fit();
        }
        (
            PanelFactor {
                n,
                sn_col,
                rows,
                val_ptr,
                vals,
            },
            zeros,
        )
    }
}

/// One supernode's finished panel, as left in its arena slot: its off-block
/// rows (elimination indices, ascending), the compacted panel length
/// `(w + m) * w`, and the zero-slot count.
#[derive(Default)]
pub(crate) struct PanelOut {
    pub rows: Vec<u32>,
    pub len: usize,
    /// Structural slots (diagonal excluded) holding an exact zero: numeric
    /// cancellation, the symmetrized pattern of an unsymmetric matrix, or
    /// `drop_tol`. `nnz() - zeros` is the stored nonzero count a sparse
    /// factor would report.
    pub zeros: usize,
}

/// Finish one supernode's panel in its slot: rows `0..w` are the supernode's
/// own columns in elimination order, the `m` rows below are permuted into
/// ascending elimination order (`e_rows[i]` is the elimination index of
/// panel row `w + i` on entry), for an LDL^T factor the `(p+1, p)` entry of
/// each 2x2 pivot is cleared (that coupling lives in `D`), `drop_tol` zeroes
/// the entries below `tau * max|col|` of their column (the diagonal
/// excluded), and off-block rows that end up without a nonzero in any column
/// are dropped from the panel (compacted in place, the tail of the slot is
/// left behind). Consumes `e_rows`.
pub(crate) fn finish_panel<T: Scalar>(
    panel: &mut [T],
    w: usize,
    mut e_rows: Vec<u32>,
    two_by_two: Option<&[bool]>,
    drop_tol: Option<f64>,
) -> PanelOut {
    let m = e_rows.len();
    let ld = w + m;
    debug_assert!(panel.len() >= ld * w);
    let panel = &mut panel[..ld * w];
    let sorted = e_rows.windows(2).all(|p| p[0] < p[1]);
    if !sorted {
        let mut order: Vec<u32> = (0..m as u32).collect();
        order.sort_unstable_by_key(|&i| e_rows[i as usize]);
        let full: Vec<u32> = (0..w as u32)
            .chain(order.iter().map(|&i| w as u32 + i))
            .collect();
        let mut tmp = Vec::with_capacity(ld);
        permute_panel_rows(panel, ld, w, &full, &mut tmp);
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
    // One pass over the strictly lower part: count the zero slots and find
    // the off-block rows without a nonzero in any column (the symmetrized
    // pattern of an unsymmetric matrix, relaxed amalgamation).
    let mut zeros = 0usize;
    let mut row_nz = vec![false; m];
    for p in 0..w {
        let col = &panel[p * ld..(p + 1) * ld];
        zeros += col[p + 1..w].iter().filter(|&&v| v == zero).count();
        for (i, &v) in col[w..].iter().enumerate() {
            if v == zero {
                zeros += 1;
            } else {
                row_nz[i] = true;
            }
        }
    }
    let m2 = row_nz.iter().filter(|&&b| b).count();
    let mut len = ld * w;
    if m2 < m {
        let ld2 = w + m2;
        let mut dst = 0;
        for p in 0..w {
            let src = p * ld;
            // Reads stay ahead of writes: `dst <= src` and the kept row `i`
            // lands at an offset no larger than its source.
            panel.copy_within(src..src + w, dst);
            let mut k = w;
            for (i, &keep) in row_nz.iter().enumerate() {
                if keep {
                    panel[dst + k] = panel[src + w + i];
                    k += 1;
                }
            }
            dst += ld2;
        }
        len = ld2 * w;
        e_rows = e_rows
            .iter()
            .zip(&row_nz)
            .filter(|(_, &keep)| keep)
            .map(|(&r, _)| r)
            .collect();
        zeros -= (m - m2) * w;
    }
    PanelOut {
        rows: e_rows,
        len,
        zeros,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_permutation_in_place() {
        // 3 x 2 panel, column-major.
        let mut panel = vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
        let mut tmp = Vec::new();
        permute_panel_rows(&mut panel, 3, 2, &[2, 0, 1], &mut tmp);
        assert_eq!(panel, vec![2.0, 0.0, 1.0, 12.0, 10.0, 11.0]);
    }

    #[test]
    fn arena_finish_closes_gaps_and_drops_empty_rows() {
        // Supernode 0: w=1, rows {1, 2}; supernode 1: w=1, row {2}; supernode 2: w=1.
        let arena = PanelArena::<f64>::new([3usize, 2, 1].into_iter());
        unsafe {
            arena.slot_mut(0).copy_from_slice(&[1.0, 0.0, 5.0]); // row 1 is empty
            arena.slot_mut(1).copy_from_slice(&[1.0, 6.0]);
            arena.slot_mut(2).copy_from_slice(&[1.0]);
        }
        let outs = [
            finish_panel(unsafe { arena.slot_mut(0) }, 1, vec![2, 1], None, None),
            finish_panel(unsafe { arena.slot_mut(1) }, 1, vec![2], None, None),
            finish_panel(unsafe { arena.slot_mut(2) }, 1, vec![], None, None),
        ];
        // Rows come in as {2, 1}: sorted to {1, 2}, and row 2 (the zero) is dropped.
        assert_eq!(outs[0].rows, vec![1]);
        assert_eq!(outs[0].len, 2);
        let mut outs = outs.into_iter();
        let (f, zeros) = arena.finish(3, [1usize, 1, 1].into_iter(), |_| outs.next().unwrap());
        assert_eq!(zeros, 0);
        assert_eq!(f.vals, vec![1.0, 5.0, 1.0, 6.0, 1.0]);
        assert_eq!(f.val_ptr, vec![0, 2, 4, 5]);
        assert_eq!(f.rows, vec![vec![1], vec![2], vec![]]);
    }
}
