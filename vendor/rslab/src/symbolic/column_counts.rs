use crate::ordering::elimination_tree::EliminationTree;
use crate::sparse::csc::CscPattern;

/// Compute the number of nonzeros in each column of the Cholesky factor L.
///
/// Uses elimination graph simulation: process columns left to right,
/// maintaining the fill pattern. For column j, `L[:,j]` contains:
/// - The diagonal entry (j,j)
/// - All original entries (i,j) with i > j
/// - All fill entries propagated from earlier columns
///
/// For indefinite factorization (LDL^T), the fill pattern is the same as
/// Cholesky - pivoting changes values but not structure (ignoring delayed
/// pivots, which are Phase 2).
///
/// Input `pattern` should be the full symmetric pattern (both triangles).
///
/// Returns a vector of length n where `counts[j]` is the number of nonzeros
/// in column j of L (including the diagonal).
pub fn column_counts(pattern: &CscPattern, _etree: &EliminationTree) -> Vec<usize> {
    let n = pattern.n;
    if n == 0 {
        return Vec::new();
    }

    // Simulate the elimination to compute the exact fill pattern.
    // For each column j, track the set of row indices i > j that will
    // have nonzeros in L[:,j].
    //
    // When column j is eliminated, for every pair of rows (i1, i2) in
    // L[:,j] with i1 > j and i2 > j, a fill entry is created at (max(i1,i2), min(i1,i2)).
    // These fill entries propagate to subsequent columns.

    // Build adjacency: for each column j, the set of rows > j
    let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, col_j) in col_rows.iter_mut().enumerate() {
        for k in pattern.col_ptr[j]..pattern.col_ptr[j + 1] {
            let i = pattern.row_idx[k];
            if i > j {
                col_j.push(i);
            }
        }
        col_j.sort_unstable();
        col_j.dedup();
    }

    let mut counts = vec![1usize; n]; // diagonal always present

    for j in 0..n {
        let rows = std::mem::take(&mut col_rows[j]);
        counts[j] += rows.len();

        // Propagate fill: all rows in this column become connected.
        // The minimum row index inherits all other row indices.
        if rows.len() > 1 {
            let min_row = rows[0]; // smallest row > j
            for &row in &rows[1..] {
                // Add row to column min_row's pattern (if not already present)
                if !col_rows[min_row].contains(&row) {
                    col_rows[min_row].push(row);
                }
            }
            col_rows[min_row].sort_unstable();
            col_rows[min_row].dedup();
        }
    }

    counts
}

/// Compute the total number of nonzeros in L from column counts.
pub fn total_factor_nnz(counts: &[usize]) -> usize {
    counts.iter().sum()
}

/// Gilbert-Ng-Peyton column counts (O(nnz(A) + n·α(n))).
///
/// Equivalent to `column_counts` but asymptotically much faster on
/// dense-ish patterns. Uses Liu's row-subtree characterization:
///
/// - L(i, j) ≠ 0 iff j ∈ T^r_i (the row subtree for row i)
/// - `c[j] = |{i : j ∈ T^r_i}|` (column count)
/// - `c[j]` = sum over T^r_i of (leaves at j) − (LCAs at j), accumulated
///   up the etree
///
/// References:
/// - Gilbert, Ng, Peyton, *SIAM J. Matrix Anal. Appl.* 15(4):1075-1091, 1994
/// - Davis, *Direct Methods for Sparse Linear Systems* §4.4
/// - CSparse `cs_counts.c` (BSD, structural reference)
pub fn column_counts_gnp(pattern: &CscPattern, etree: &EliminationTree) -> Vec<usize> {
    let n = pattern.n;
    if n == 0 {
        return Vec::new();
    }

    let post = etree.postorder();
    let first = etree.first_descendants(&post);
    let children = etree.children();

    // delta[i] starts at 1 iff i is a leaf of the etree (the row subtree
    // T^r_i trivially contains i as a leaf whenever i has no etree children
    // - the contribution of every node i to its own column count).
    let mut delta: Vec<i64> = (0..n)
        .map(|i| if children[i].is_empty() { 1 } else { 0 })
        .collect();

    // maxfirst[k]: max first[i_prev] over previously-seen row-subtree leaves
    //              that contributed to column k. -1 until any leaf is seen.
    // prevleaf[k]: the previous such i, used to find the LCA with the
    //              current i via disjoint-set `find` on the partial etree.
    // ancestor[i]: DSU pointer; after node i is "processed" (below), it
    //              points (eventually) to the deepest still-unprocessed
    //              ancestor - which is the LCA of any two descendants.
    let mut maxfirst = vec![-1i64; n];
    let mut prevleaf = vec![-1i64; n];
    let mut ancestor: Vec<usize> = (0..n).collect();

    for &i in &post {
        // Head step (cs_counts.c): every non-root node contributes -1 to
        // its parent's delta, canceling the double-count produced when
        // the accumulation pass merges i and parent(i)'s subtree sums.
        if let Some(p) = etree.parent[i] {
            delta[p] -= 1;
        }

        // Walk column i of the symmetric pattern. Entries with row
        // partner > i are the below-diagonal nonzeros A(partner, i).
        // For each such partner, test whether i is a leaf of the row
        // subtree T^r_{partner}. Condition: first[i] > maxfirst[partner].
        let fi = first[i] as i64;
        let row_start = pattern.col_ptr[i];
        let row_end = pattern.col_ptr[i + 1];
        for k in row_start..row_end {
            let partner = pattern.row_idx[k];
            if partner <= i {
                continue;
            }
            if fi > maxfirst[partner] {
                delta[i] += 1;
                let pl = prevleaf[partner];
                if pl != -1 {
                    // LCA of pl and i via path-compressed find on the
                    // partially-unioned etree.
                    let mut q = pl as usize;
                    while ancestor[q] != q {
                        q = ancestor[q];
                    }
                    let root = q;
                    let mut cur = pl as usize;
                    while cur != root {
                        let next = ancestor[cur];
                        ancestor[cur] = root;
                        cur = next;
                    }
                    delta[root] -= 1;
                }
                prevleaf[partner] = i as i64;
                maxfirst[partner] = fi;
            }
        }
        // Union i into its parent so subsequent `find`s from descendants
        // of i stop at the parent (the LCA with any later-processed node).
        if let Some(p) = etree.parent[i] {
            ancestor[i] = p;
        }
    }

    // Accumulate deltas up the etree (children into parents) in postorder.
    for &i in &post {
        if let Some(p) = etree.parent[i] {
            delta[p] += delta[i];
        }
    }

    delta.iter().map(|&d| d as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    #[test]
    fn test_column_counts_diagonal() {
        // Diagonal matrix: each column of L has exactly 1 nonzero (the diagonal)
        let m = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0; 4]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![1, 1, 1, 1]);
        assert_eq!(total_factor_nnz(&counts), 4);
    }

    #[test]
    fn test_column_counts_tridiagonal() {
        // Tridiagonal 4x4: L has entries on diagonal and one subdiagonal
        // Column 0: rows 0, 1 → count = 2
        // Column 1: rows 1, 2 → count = 2
        // Column 2: rows 2, 3 → count = 2
        // Column 3: row 3      → count = 1
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![2, 2, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 7);
    }

    #[test]
    fn test_column_counts_dense() {
        // Dense 3x3: L is full lower triangle
        // Column 0: rows 0, 1, 2 → count = 3
        // Column 1: rows 1, 2    → count = 2
        // Column 2: row 2        → count = 1
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![3, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 6); // n*(n+1)/2
    }

    #[test]
    fn test_column_counts_arrow() {
        // Arrow 5x5: column 0 has entries at rows 0-4, others are diagonal
        // Eliminating column 0 creates fill among rows 1-4
        // Column 0: rows 0,1,2,3,4 → count = 5
        // Column 1: rows 1,2,3,4 (fill from col 0) → count = 4
        // Column 2: rows 2,3,4 → count = 3
        // Column 3: rows 3,4 → count = 2
        // Column 4: row 4 → count = 1
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![5, 4, 3, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 15); // fully dense: n*(n+1)/2
    }

    #[test]
    fn test_column_counts_block_diagonal() {
        // Two 2x2 dense blocks: no fill between blocks
        // [a b 0 0]
        // [b c 0 0]
        // [0 0 d e]
        // [0 0 e f]
        let m = CscMatrix::from_triplets(4, &[0, 1, 1, 2, 3, 3], &[0, 0, 1, 2, 2, 3], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let counts = column_counts(&pat, &etree);

        assert_eq!(counts, vec![2, 1, 2, 1]);
        assert_eq!(total_factor_nnz(&counts), 6);
    }

    // --- Phase 2.5.1: GNP column-count parity tests ---
    //
    // Each reuses the exact pattern from the reference tests above and
    // asserts column_counts_gnp returns the same vector as column_counts.

    #[test]
    fn gnp_matches_reference_diagonal() {
        let m = CscMatrix::from_triplets(4, &[0, 1, 2, 3], &[0, 1, 2, 3], &[1.0; 4]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_tridiagonal() {
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_dense() {
        let m = CscMatrix::from_triplets(3, &[0, 1, 2, 1, 2, 2], &[0, 0, 0, 1, 1, 2], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_arrow() {
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_block_diagonal() {
        let m = CscMatrix::from_triplets(4, &[0, 1, 1, 2, 3, 3], &[0, 0, 1, 2, 2, 3], &[1.0; 6])
            .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }

    #[test]
    fn gnp_matches_reference_banded() {
        // 7x7 banded: entries on diagonal, sub-diagonal, and 2 below diagonal.
        // Produces a non-trivial row-subtree pattern (more interesting than chain).
        let rows = vec![0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 6];
        let cols = vec![0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6];
        let m = CscMatrix::from_triplets(7, &rows, &cols, &vec![1.0; rows.len()]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        assert_eq!(column_counts_gnp(&pat, &etree), column_counts(&pat, &etree));
    }
}
