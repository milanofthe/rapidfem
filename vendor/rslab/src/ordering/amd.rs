use crate::sparse::csc::CscPattern;

/// Apply a permutation to row/column indices: compute P*A*P^T pattern.
///
/// Given a symmetric CscPattern (both triangles stored, the form produced
/// by `CscMatrix::symmetric_pattern`) and a permutation `perm`
/// (new-to-old mapping), returns the permuted pattern with both
/// triangles, sorted within each column.
///
/// Uses a two-pass counting-sort layout (O(nnz)) rather than a
/// `Vec<Vec<usize>>` with per-column sort+dedup. On near-dense inputs
/// like DMN15103 (n=99 fully full) this is ~7x faster because (a) each
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

    // Pass 2: new column `j` is old column `perm[j]` with its rows renumbered,
    // sorted. Columns are independent, so contiguous blocks of them, balanced
    // by entry count, fill in parallel; every column comes out the same as
    // serially, whatever the thread count.
    let fill = |j0: usize, j1: usize, out: &mut [usize]| {
        let base = col_ptr[j0];
        for j in j0..j1 {
            let dst = &mut out[col_ptr[j] - base..col_ptr[j + 1] - base];
            let old = perm[j];
            let src = &pattern.row_idx[pattern.col_ptr[old]..pattern.col_ptr[old + 1]];
            for (d, &r) in dst.iter_mut().zip(src) {
                *d = inv_perm[r];
            }
            dst.sort_unstable();
        }
    };
    const PARALLEL_MIN_NNZ: usize = 1 << 16;
    if nnz < PARALLEL_MIN_NNZ || rayon::current_num_threads() == 1 {
        fill(0, n, &mut row_idx);
    } else {
        use rayon::prelude::*;
        let target = nnz.div_ceil(8 * rayon::current_num_threads());
        let mut blocks: Vec<(usize, usize, &mut [usize])> = Vec::new();
        let mut rest: &mut [usize] = &mut row_idx;
        let mut j0 = 0;
        while j0 < n {
            let mut j1 = j0 + 1;
            while j1 < n && col_ptr[j1] - col_ptr[j0] < target {
                j1 += 1;
            }
            let (head, tail) = rest.split_at_mut(col_ptr[j1] - col_ptr[j0]);
            blocks.push((j0, j1, head));
            rest = tail;
            j0 = j1;
        }
        blocks
            .into_par_iter()
            .for_each(|(j0, j1, out)| fill(j0, j1, out));
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
