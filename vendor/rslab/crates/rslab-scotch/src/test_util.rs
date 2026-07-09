//! Test-only helpers shared across module-level `#[cfg(test)]` blocks.
//!
//! Compiled out of release builds.

use std::collections::BTreeSet;

/// Build a full-symmetric CSC pattern from undirected edges.
///
/// Each `(i, j)` in `edges` produces both `(i, j)` and `(j, i)`
/// entries. Self-loops `(i, i)` are accepted and emitted; the
/// downstream [`crate::graph::Graph::from_csc_pattern`] will drop
/// them. Row indices within each column are sorted ascending and
/// deduplicated.
pub(crate) fn csc_from_edges(n: usize, edges: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
    let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(i, j) in edges {
        set.insert((i, j));
        set.insert((j, i));
    }
    let mut cols: Vec<Vec<i32>> = vec![Vec::new(); n];
    for &(r, c) in &set {
        cols[c].push(r as i32);
    }
    for col in &mut cols {
        col.sort();
    }
    let mut col_ptr = Vec::with_capacity(n + 1);
    col_ptr.push(0);
    let mut row_idx = Vec::new();
    for col in &cols {
        for &r in col {
            row_idx.push(r);
        }
        col_ptr.push(row_idx.len() as i32);
    }
    (col_ptr, row_idx)
}
