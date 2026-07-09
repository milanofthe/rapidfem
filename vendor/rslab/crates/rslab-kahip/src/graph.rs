//! Internal undirected weighted graph in CSR form.
//!
//! Shared by K3 (flow refinement), K4 (flow separator), K5 (V/F
//! cycles) and K6 (driver). Sorted neighbor lists per vertex; edge
//! weights parallel to the adjacency array. Symmetric: every edge
//! `(u, v, w)` appears as `v ∈ N(u)` and `u ∈ N(v)` with the same
//! weight.

use rslab_ordering_core::{CscPattern, OrderingError};

/// Undirected weighted graph stored in CSR.
///
/// Self-loops and duplicate edges are rejected at construction.
/// Empty graphs (`n == 0`) are permitted and degenerate into no-ops
/// in all downstream algorithms.
#[derive(Debug, Clone)]
pub(crate) struct UndirectedGraph {
    pub n: usize,
    /// Length `n + 1`; `xadj[v]..xadj[v+1]` is the slice of
    /// `adjncy` / `eweight` for vertex `v`.
    pub xadj: Vec<usize>,
    /// Neighbor list, sorted ascending within each vertex.
    pub adjncy: Vec<usize>,
    /// Edge weights parallel to `adjncy`. Always `> 0`.
    pub eweight: Vec<i64>,
    /// Vertex weights, length `n`, every entry `> 0`. On a finest-level
    /// graph these are all `1`; on a coarse graph produced by
    /// multilevel coarsening each vertex stands for a supervertex and
    /// carries the summed mass of the original vertices it absorbed.
    /// K3 flow refinement measures its balance constraint against these
    /// weights, not against vertex counts - a count-balanced cut on a
    /// graph with `vweight ≫ 1` can be badly weight-imbalanced. K4's flow
    /// node separator *consumes* `vweight` but does not itself enforce a
    /// balance constraint (the separator is always returned; balance
    /// checking is the caller's responsibility - see
    /// `dev/research/ordering-kahip-k4.md`).
    pub vweight: Vec<i64>,
}

impl UndirectedGraph {
    /// Neighbor indices slice for `v`.
    #[inline]
    pub fn neighbors(&self, v: usize) -> &[usize] {
        &self.adjncy[self.xadj[v]..self.xadj[v + 1]]
    }

    /// Edge weights slice for `v`, parallel to [`Self::neighbors`].
    #[inline]
    pub fn eweights(&self, v: usize) -> &[i64] {
        &self.eweight[self.xadj[v]..self.xadj[v + 1]]
    }

    /// Summed vertex weight of each side of a bisection
    /// `where_[v] ∈ {0, 1}`. Vertices with any other label contribute
    /// to neither side.
    pub fn part_weights(&self, where_: &[u8]) -> (i64, i64) {
        debug_assert_eq!(where_.len(), self.n);
        let mut w0: i64 = 0;
        let mut w1: i64 = 0;
        for (v, &p) in where_.iter().enumerate() {
            match p {
                0 => w0 += self.vweight[v],
                1 => w1 += self.vweight[v],
                _ => {}
            }
        }
        (w0, w1)
    }

    /// Total vertex weight of the graph (`Σ vweight`).
    pub fn total_weight(&self) -> i64 {
        self.vweight.iter().sum()
    }

    /// Total weighted cut of a bisection `where_[v] ∈ {0, 1}`.
    /// Each undirected edge is counted once (via `u < v` guard).
    pub fn cut_weight(&self, where_: &[u8]) -> i64 {
        debug_assert_eq!(where_.len(), self.n);
        let mut cut: i64 = 0;
        for u in 0..self.n {
            for (j, &v) in self.neighbors(u).iter().enumerate() {
                if v > u && where_[u] != where_[v] {
                    cut += self.eweights(u)[j];
                }
            }
        }
        cut
    }
}

/// Build an [`UndirectedGraph`] from a full-symmetric
/// [`CscPattern`] with unit edge weights. Diagonal entries are
/// ignored (they don't represent graph edges).
///
/// Returns [`OrderingError::MalformedInput`] if `pattern` is
/// structurally invalid.
pub(crate) fn from_csc_unit_weights(
    pattern: &CscPattern<'_>,
) -> Result<UndirectedGraph, OrderingError> {
    let n = pattern.n;
    if pattern.col_ptr.len() != n + 1 {
        return Err(OrderingError::MalformedInput);
    }
    let mut xadj = Vec::with_capacity(n + 1);
    let mut adjncy = Vec::new();
    let mut eweight = Vec::new();
    xadj.push(0);
    for j in 0..n {
        let start = pattern.col_ptr[j] as usize;
        let end = pattern.col_ptr[j + 1] as usize;
        if start > end || end > pattern.row_idx.len() {
            return Err(OrderingError::MalformedInput);
        }
        let mut col_neighbors: Vec<usize> = pattern.row_idx[start..end]
            .iter()
            .filter_map(|&r| {
                let r = r as usize;
                if r == j {
                    None
                } else {
                    Some(r)
                }
            })
            .collect();
        col_neighbors.sort_unstable();
        col_neighbors.dedup();
        for &v in &col_neighbors {
            if v >= n {
                return Err(OrderingError::MalformedInput);
            }
            adjncy.push(v);
            eweight.push(1);
        }
        xadj.push(adjncy.len());
    }
    Ok(UndirectedGraph {
        n,
        xadj,
        adjncy,
        eweight,
        vweight: vec![1; n],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(n: usize, edges: &[(usize, usize, i64)]) -> UndirectedGraph {
        // Helper that builds a symmetric CSR from an edge list.
        let mut per_v: Vec<Vec<(usize, i64)>> = vec![Vec::new(); n];
        for &(u, v, w) in edges {
            assert!(u != v, "self-loops not allowed");
            assert!(w > 0, "weights must be positive");
            per_v[u].push((v, w));
            per_v[v].push((u, w));
        }
        let mut xadj = Vec::with_capacity(n + 1);
        let mut adjncy = Vec::new();
        let mut eweight = Vec::new();
        xadj.push(0);
        for list in per_v.iter_mut() {
            list.sort_by_key(|&(v, _)| v);
            for &(v, w) in list.iter() {
                adjncy.push(v);
                eweight.push(w);
            }
            xadj.push(adjncy.len());
        }
        UndirectedGraph {
            n,
            xadj,
            adjncy,
            eweight,
            vweight: vec![1; n],
        }
    }

    #[test]
    fn cut_weight_matches_expected() {
        // Path 0-1-2-3-4 (unit weights); bisection {0,1} | {2,3,4}.
        let g = build(5, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)]);
        let where_ = [0u8, 0, 1, 1, 1];
        assert_eq!(g.cut_weight(&where_), 1);
    }

    #[test]
    fn cut_weight_weighted_path() {
        let g = build(4, &[(0, 1, 3), (1, 2, 5), (2, 3, 2)]);
        // Cut at 1|2: weight 5.
        assert_eq!(g.cut_weight(&[0, 0, 1, 1]), 5);
        // Cut at 2|3: weight 2.
        assert_eq!(g.cut_weight(&[0, 0, 0, 1]), 2);
    }

    #[test]
    fn from_csc_roundtrip() {
        // Full-symmetric CSC for path 0-1-2.
        let col_ptr = [0i32, 1, 3, 4];
        let row_idx = [1i32, 0, 2, 1];
        let pattern = CscPattern::new(3, &col_ptr, &row_idx).unwrap();
        let g = from_csc_unit_weights(&pattern).unwrap();
        assert_eq!(g.n, 3);
        assert_eq!(g.neighbors(0), &[1]);
        assert_eq!(g.neighbors(1), &[0, 2]);
        assert_eq!(g.neighbors(2), &[1]);
        assert!(g.eweights(0).iter().all(|&w| w == 1));
    }

    #[test]
    fn from_csc_ignores_diagonal() {
        let col_ptr = [0i32, 2, 4, 5];
        let row_idx = [0i32, 1, 0, 1, 2];
        let pattern = CscPattern::new(3, &col_ptr, &row_idx).unwrap();
        let g = from_csc_unit_weights(&pattern).unwrap();
        assert_eq!(g.neighbors(0), &[1]);
        assert_eq!(g.neighbors(1), &[0]);
        assert_eq!(g.neighbors(2), &[]);
    }
}
