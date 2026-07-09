use crate::sparse::csc::CscPattern;

/// Apply a permutation to row/column indices: compute P·A·Pᵀ pattern.
///
/// Given a symmetric CscPattern (both triangles stored, the form produced
/// by `CscMatrix::symmetric_pattern`) and a permutation `perm`
/// (new-to-old mapping), returns the permuted pattern with both
/// triangles, sorted within each column.
///
/// Uses a two-pass counting-sort layout (O(nnz)) rather than a
/// `Vec<Vec<usize>>` with per-column sort+dedup. On near-dense inputs
/// like DMN15103 (n=99 fully full) this is ~7× faster because (a) each
/// entry is copied exactly once instead of being pushed once from each
/// triangle and then deduped, and (b) the final per-column sort runs on
/// pre-placed contiguous slices.
///
/// Note: the fill-reducing ordering itself is produced by the standalone
/// `rslab_amd` crate (a quotient-graph AMD); this module only retains the
/// permutation-application helper used throughout `symbolic`.
#[allow(clippy::needless_range_loop)]
pub fn permute_pattern(pattern: &CscPattern, perm: &[usize]) -> CscPattern {
    let n = pattern.n;

    // Build inverse permutation: inv_perm[old] = new
    let mut inv_perm = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        inv_perm[old] = new;
    }

    // Pass 1: count entries per new column. Since the input is a full
    // symmetric pattern, column `old_j` has exactly one entry for every
    // off-diagonal neighbor (plus any diagonal) - we just re-bucket them
    // into column `inv_perm[old_j]` one-for-one.
    let mut col_ptr = vec![0usize; n + 1];
    for old_j in 0..n {
        let new_j = inv_perm[old_j];
        let nnz_j = pattern.col_ptr[old_j + 1] - pattern.col_ptr[old_j];
        col_ptr[new_j + 1] = nnz_j;
    }
    // Prefix sum
    for j in 0..n {
        col_ptr[j + 1] += col_ptr[j];
    }

    let nnz = col_ptr[n];
    let mut row_idx = vec![0usize; nnz];
    let mut offsets: Vec<usize> = col_ptr[..n].to_vec();

    // Pass 2: fill row_idx with the permuted row values.
    for old_j in 0..n {
        let new_j = inv_perm[old_j];
        let start = pattern.col_ptr[old_j];
        let end = pattern.col_ptr[old_j + 1];
        for k in start..end {
            let new_i = inv_perm[pattern.row_idx[k]];
            row_idx[offsets[new_j]] = new_i;
            offsets[new_j] += 1;
        }
    }

    // Sort each column's row indices. Downstream code (column_counts,
    // factorization) does not strictly require sorted order, but the
    // previous implementation produced sorted columns and keeping that
    // invariant avoids subtle coupling with callers that may rely on it.
    for j in 0..n {
        let start = col_ptr[j];
        let end = col_ptr[j + 1];
        row_idx[start..end].sort_unstable();
    }

    CscPattern {
        n,
        col_ptr,
        row_idx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    #[test]
    fn test_permute_pattern() {
        // Simple 3x3 tridiagonal: [[1,-1,0],[-1,2,-1],[0,-1,1]]
        let m = CscMatrix::from_triplets(
            3,
            &[0, 1, 1, 2, 2],
            &[0, 0, 1, 1, 2],
            &[1.0, -1.0, 2.0, -1.0, 1.0],
        )
        .unwrap();
        let pat = m.symmetric_pattern();

        // Reverse permutation: [2, 1, 0]
        let perm = vec![2, 1, 0];
        let permuted = permute_pattern(&pat, &perm);

        // After reversing, the pattern should be the same (tridiagonal is symmetric)
        assert_eq!(permuted.n, 3);
        assert_eq!(permuted.col_ptr[3], pat.col_ptr[3]);
    }
}
