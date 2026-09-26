//! The input stage shared by the LDL^T and LU factorizations: the frozen
//! program that moves a value set of `A` into the factorization order, and
//! the view the supernodes assemble from.
//!
//! The permuted structure is a pure function of the analyzed pattern and the
//! permutation, so it is built once per analysis; every (re)factorization
//! then only scatters its values through `pos` (one linear pass, the scaling
//! applied on the way), with no counting sort, no per-column sorting and no
//! copy of the indices.

use super::Li;
use crate::scalar::Scalar;
use crate::symbolic::SymbolicFactorization;

/// Where the entries of the permuted matrix live: a column part (by column:
/// `col_ptr`, `row_idx`, then the values `[..split]`) and, on the LU path, a
/// row part (by row: `row_ptr`, `col_idx`, values from `split`). Indices are
/// sorted within each column and row, the order the assembly walks them in.
pub(crate) struct InputProgram {
    n: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<Li>,
    row_ptr: Vec<usize>,
    col_idx: Vec<Li>,
    /// `pos[k]`: the slot of entry `k` of `A` in the program's value array.
    pos: Vec<usize>,
}

impl InputProgram {
    /// The symmetric fold of a lower triangle (LDL^T): entry `(i, j)` lands
    /// at `(max(gi, gj), min(gi, gj))` of `P^T A P`, `g = perm_inv[*]`.
    pub fn symmetric(col_ptr: &[usize], row_idx: &[usize], perm_inv: &[usize]) -> Self {
        Self::build(col_ptr, row_idx, |i, j| {
            let (gi, gj) = (perm_inv[i], perm_inv[j]);
            Part::Col {
                col: gi.min(gj),
                row: gi.max(gj),
            }
        })
    }

    /// The split of a general matrix (LU): entry `(i, j)` of `P^T B P` (`B`
    /// the matrix factored; row `r` of `A` is row `row_map[r]` of `B`) goes to
    /// the columns of `j`'s supernode when its row lies in that supernode's
    /// diagonal block or below, else to the `U12` rows of `i`'s supernode.
    pub fn general(
        col_ptr: &[usize],
        row_idx: &[usize],
        row_map: Option<&[usize]>,
        sym: &SymbolicFactorization,
    ) -> Self {
        let mut first = vec![0usize; sym.n];
        for sn in &sym.supernodes {
            first[sn.first_col..sn.first_col + sn.ncol].fill(sn.first_col);
        }
        Self::build(col_ptr, row_idx, |i, j| {
            let (gi, gj) = (sym.perm_inv[row_map.map_or(i, |m| m[i])], sym.perm_inv[j]);
            if gi >= first[gj] {
                Part::Col { col: gj, row: gi }
            } else {
                Part::Row { row: gi, col: gj }
            }
        })
    }

    fn build(col_ptr: &[usize], row_idx: &[usize], place: impl Fn(usize, usize) -> Part) -> Self {
        let n = col_ptr.len() - 1;
        let entries = || (0..n).flat_map(|j| (col_ptr[j]..col_ptr[j + 1]).map(move |k| (j, k)));
        // Count per target column / row, place `(index, entry)` pairs, then
        // sort each column / row by index.
        let (mut cptr, mut rptr) = (vec![0usize; n + 1], vec![0usize; n + 1]);
        for (j, k) in entries() {
            match place(row_idx[k], j) {
                Part::Col { col, .. } => cptr[col + 1] += 1,
                Part::Row { row, .. } => rptr[row + 1] += 1,
            }
        }
        for c in 0..n {
            cptr[c + 1] += cptr[c];
            rptr[c + 1] += rptr[c];
        }
        let split = cptr[n];
        let mut pairs: Vec<(Li, usize)> = vec![(0, 0); row_idx.len()];
        let (mut cc, mut rc) = (cptr[..n].to_vec(), rptr[..n].to_vec());
        for (j, k) in entries() {
            match place(row_idx[k], j) {
                Part::Col { col, row } => {
                    pairs[cc[col]] = (row as Li, k);
                    cc[col] += 1;
                }
                Part::Row { row, col } => {
                    pairs[split + rc[row]] = (col as Li, k);
                    rc[row] += 1;
                }
            }
        }
        let mut pos = vec![0usize; pairs.len()];
        let mut index = |ptr: &[usize], base: usize| -> Vec<Li> {
            let mut out = vec![0 as Li; ptr[n]];
            for c in 0..n {
                let run = &mut pairs[base + ptr[c]..base + ptr[c + 1]];
                run.sort_unstable_by_key(|&(g, _)| g);
                for (p, &(g, k)) in (ptr[c]..ptr[c + 1]).zip(run.iter()) {
                    out[p] = g;
                    pos[k] = base + p;
                }
            }
            out
        };
        let row_idx_out = index(&cptr, 0);
        let col_idx = index(&rptr, split);
        InputProgram {
            n,
            col_ptr: cptr,
            row_idx: row_idx_out,
            row_ptr: rptr,
            col_idx,
            pos,
        }
    }

    /// Scatter a value set of `A` (the pattern this was built from) into the
    /// program's order in one pass. With `weight`, entry `(i, j)` of `A` is
    /// multiplied by `weight(i, j)` on the way (the equilibration), so no
    /// scaled copy of `A` is ever held.
    pub fn values<T: Scalar>(
        &self,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[T],
        weight: Option<&dyn Fn(usize, usize) -> f64>,
    ) -> Vec<T> {
        debug_assert_eq!(values.len(), self.pos.len());
        let mut out = vec![T::zero(); values.len()];
        match weight {
            None => {
                for (k, &p) in self.pos.iter().enumerate() {
                    out[p] = values[k];
                }
            }
            Some(w) => {
                for j in 0..self.n {
                    for k in col_ptr[j]..col_ptr[j + 1] {
                        out[self.pos[k]] = values[k] * T::from_real(w(row_idx[k], j));
                    }
                }
            }
        }
        out
    }
}

/// Target of one entry of `A` in the permuted matrix.
enum Part {
    Col { col: usize, row: usize },
    Row { row: usize, col: usize },
}

/// One factorization's permuted input: the program and its values.
#[derive(Clone, Copy)]
pub(crate) struct Input<'a, T> {
    prog: &'a InputProgram,
    vals: &'a [T],
}

impl<'a, T: Scalar> Input<'a, T> {
    pub fn new(prog: &'a InputProgram, vals: &'a [T]) -> Self {
        debug_assert_eq!(vals.len(), prog.pos.len());
        Input { prog, vals }
    }

    /// Every value, in no particular order.
    pub fn values(self) -> &'a [T] {
        self.vals
    }

    /// Rows and values of column `c` (in its supernode's columns, rows
    /// ascending).
    #[inline]
    pub fn col(self, c: usize) -> impl Iterator<Item = (usize, T)> + 'a {
        let r = self.prog.col_ptr[c]..self.prog.col_ptr[c + 1];
        let (idx, vals) = (&self.prog.row_idx[r.clone()], &self.vals[r]);
        idx.iter().zip(vals).map(|(&g, &v)| (g as usize, v))
    }

    /// Columns and values of row `r` in its supernode's `U12` (columns past
    /// the supernode, ascending); empty for the symmetric fold.
    #[inline]
    pub fn row(self, r: usize) -> impl Iterator<Item = (usize, T)> + 'a {
        let split = self.prog.col_ptr[self.prog.n];
        let k = split + self.prog.row_ptr[r]..split + self.prog.row_ptr[r + 1];
        let idx = &self.prog.col_idx[self.prog.row_ptr[r]..self.prog.row_ptr[r + 1]];
        idx.iter()
            .zip(&self.vals[k])
            .map(|(&g, &v)| (g as usize, v))
    }
}
