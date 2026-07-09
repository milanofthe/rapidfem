//! Schur-aware fill-reducing ordering primitive (F3.1).
//!
//! Given a symmetric matrix and a list of Schur-block variables, produce
//! a permutation where the Schur tail occupies positions
//! `[n - n_schur, n)` in the user-supplied order, and the non-Schur
//! prefix is ordered by AMD applied to the **non-Schur subgraph**.
//!
//! Per `dev/research/schur-complement.md` D3, this primitive runs the
//! ordering algorithm on the subgraph induced by non-Schur indices,
//! then appends the Schur tail. This is a clean-room divergence from
//! MUMPS's HALO-SCHUR mechanism (which keeps Schur variables in the
//! AMD graph but constrained to come last via amalgamation): rslab does
//! not own the `rslab-amd` driver API, and the subgraph approach is
//! composable across all four ordering methods (AMD, AMF, MetisND,
//! ScotchND) without any external-crate change.
//!
//! **Trade-off:** the strict-subgraph approach ignores edges to Schur
//! variables when computing the non-Schur ordering. For typical KKT
//! shapes (Schur block ≪ eliminated block), this is a small loss; for
//! Schur-heavy patterns it could meaningfully degrade fill. F3.3
//! cross-validation against MUMPS HALO-SCHUR will quantify this.
//!
//! See `dev/plans/kkt-feature-gaps.md` §F3.

use crate::error::RslabError;
use crate::sparse::csc::{CscMatrix, CscPattern};

/// Compute a Schur-aware permutation for `matrix` given a list of
/// variables to keep in the Schur tail.
///
/// Returns a permutation `perm` of length `n` such that:
/// - `perm[n - n_schur + i] == schur_indices[i]` for each `i`.
/// - The prefix `perm[0..n - n_schur]` is the AMD ordering of the
///   non-Schur subgraph, lifted back to original-index space.
///
/// `perm` is "new-to-old": `perm[k]` is the original column that
/// becomes column `k` after permutation.
///
/// # Errors
/// - `InvalidInput("schur_indices contains duplicates")`
/// - `InvalidInput("schur_indices entry out of range")`
/// - `InvalidInput("schur_indices.len() == n is not allowed")`
///   (would mean elimination set is empty; the user almost
///   certainly meant a partial Schur and `n_schur == n` is a
///   logic bug upstream - see F3.0 D-Question 3.)
pub fn compute_schur_aware_perm(
    matrix: &CscMatrix,
    schur_indices: &[usize],
) -> Result<Vec<usize>, RslabError> {
    let n = matrix.n;
    let n_schur = schur_indices.len();

    // Empty Schur ⇒ standard AMD on the full pattern, no Schur at all.
    if n_schur == 0 {
        let pattern = matrix.symmetric_pattern();
        return run_amd(&pattern);
    }

    if n_schur == n {
        return Err(RslabError::InvalidInput(
            "schur_indices.len() == n is not allowed; elimination set would be empty".to_string(),
        ));
    }

    // Validate: in-range, no duplicates.
    let mut is_schur = vec![false; n];
    for &s in schur_indices {
        if s >= n {
            return Err(RslabError::InvalidInput(format!(
                "schur_indices entry {} out of range for n={}",
                s, n
            )));
        }
        if is_schur[s] {
            return Err(RslabError::InvalidInput(format!(
                "schur_indices contains duplicate entry {}",
                s
            )));
        }
        is_schur[s] = true;
    }

    // Build the non-Schur index list and its inverse map.
    // non_schur_indices[k] = original index of the k-th non-Schur variable.
    // sub_of[orig] = local index in non_schur_indices, or usize::MAX for Schur.
    let n_f = n - n_schur;
    let mut non_schur_indices = Vec::with_capacity(n_f);
    let mut sub_of = vec![usize::MAX; n];
    for orig in 0..n {
        if !is_schur[orig] {
            sub_of[orig] = non_schur_indices.len();
            non_schur_indices.push(orig);
        }
    }

    // Build the non-Schur subgraph pattern by restricting the symmetric
    // pattern to entries (i, j) where both i and j are non-Schur. Index
    // remapping: original index -> local index in non_schur_indices.
    let full_pattern = matrix.symmetric_pattern();
    let sub_pattern = restrict_pattern_to_subgraph(&full_pattern, &sub_of, n_f);

    // Run AMD on the sub-pattern. sub_perm has length n_f, in subgraph
    // index space (0..n_f).
    let sub_perm = run_amd(&sub_pattern)?;

    // Lift the sub-permutation back to original-index space, then append
    // the Schur tail in user-supplied order.
    let mut perm = Vec::with_capacity(n);
    for &sub_idx in &sub_perm {
        perm.push(non_schur_indices[sub_idx]);
    }
    for &s in schur_indices {
        perm.push(s);
    }

    debug_assert_eq!(perm.len(), n);
    Ok(perm)
}

/// Restrict a symmetric pattern to the subgraph induced by indices for
/// which `sub_of[orig] != usize::MAX`. The output pattern has dimension
/// `n_f` and uses the local index `sub_of[orig]` in place of `orig`.
fn restrict_pattern_to_subgraph(full: &CscPattern, sub_of: &[usize], n_f: usize) -> CscPattern {
    // First pass: count entries per local column.
    let mut col_counts = vec![0usize; n_f];
    for j_orig in 0..full.n {
        let j_loc = sub_of[j_orig];
        if j_loc == usize::MAX {
            continue;
        }
        for k in full.col_ptr[j_orig]..full.col_ptr[j_orig + 1] {
            let i_orig = full.row_idx[k];
            if sub_of[i_orig] != usize::MAX {
                col_counts[j_loc] += 1;
            }
        }
    }

    let mut col_ptr = vec![0usize; n_f + 1];
    for j in 0..n_f {
        col_ptr[j + 1] = col_ptr[j] + col_counts[j];
    }
    let nnz = col_ptr[n_f];
    let mut row_idx = vec![0usize; nnz];

    // Second pass: place entries.
    let mut offsets = col_ptr[..n_f].to_vec();
    for j_orig in 0..full.n {
        let j_loc = sub_of[j_orig];
        if j_loc == usize::MAX {
            continue;
        }
        for k in full.col_ptr[j_orig]..full.col_ptr[j_orig + 1] {
            let i_orig = full.row_idx[k];
            let i_loc = sub_of[i_orig];
            if i_loc != usize::MAX {
                row_idx[offsets[j_loc]] = i_loc;
                offsets[j_loc] += 1;
            }
        }
    }

    // Sort row indices within each column (full.symmetric_pattern() returns
    // sorted columns; restriction preserves order, but sort defensively in
    // case the caller passes a non-canonical pattern in future).
    for j in 0..n_f {
        let start = col_ptr[j];
        let end = col_ptr[j + 1];
        row_idx[start..end].sort_unstable();
    }

    CscPattern {
        n: n_f,
        col_ptr,
        row_idx,
    }
}

/// Run AMD on a CscPattern, returning the new-to-old permutation.
///
/// Mirrors the `run_external_ordering` path in `src/symbolic/mod.rs`
/// but specialized to AMD only. We keep this local because F3.1 doesn't
/// need to plumb a `method` parameter - F3.2 will, when it integrates
/// the Schur-aware ordering into `symbolic_factorize_with_schur` and
/// dispatches over all four ordering methods.
fn run_amd(pattern: &CscPattern) -> Result<Vec<usize>, RslabError> {
    if pattern.n == 0 {
        return Ok(Vec::new());
    }
    let col_buf: Result<Vec<i32>, _> = pattern.col_ptr.iter().map(|&x| i32::try_from(x)).collect();
    let col_buf = col_buf.map_err(|_| {
        RslabError::InvalidInput("matrix too large for i32-indexed AMD".to_string())
    })?;
    let row_buf: Result<Vec<i32>, _> = pattern.row_idx.iter().map(|&x| i32::try_from(x)).collect();
    let row_buf = row_buf.map_err(|_| {
        RslabError::InvalidInput("matrix too large for i32-indexed AMD".to_string())
    })?;
    let pat = rslab_ordering_core::CscPattern::new(pattern.n, &col_buf, &row_buf)
        .ok_or_else(|| RslabError::InvalidInput("malformed CSC pattern".to_string()))?;
    let perm_i32 = rslab_amd::amd_order(&pat)
        .map_err(|e| RslabError::InvalidInput(format!("AMD failed: {}", e)))?;
    let mut out: Vec<usize> = Vec::with_capacity(perm_i32.len());
    for x in perm_i32 {
        let u = usize::try_from(x)
            .map_err(|_| RslabError::InvalidInput("AMD returned negative index".to_string()))?;
        if u >= pattern.n {
            return Err(RslabError::InvalidInput(
                "AMD returned out-of-range index".to_string(),
            ));
        }
        out.push(u);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_kkt() -> CscMatrix {
        // 6x6 KKT with two-block structure: the trailing 2x2 is the dense
        // (primal, dual) Schur block; the leading 4x4 has identity-diagonal
        // structure with off-diagonal connections to the Schur.
        //   [ 1            *  *  ]
        //   [    1         *  *  ]
        //   [       1      *  *  ]
        //   [          1   *  *  ]
        //   [ *  *  *  *   a  b  ]
        //   [ *  *  *  *   b  c  ]
        // (lower triangle stored)
        let rows = vec![0, 1, 2, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5];
        let cols = vec![0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5];
        let vals = vec![
            1.0, 1.0, 1.0, 1.0, // diagonal of first 4
            0.5, 0.5, 0.5, 0.5, 2.0, // row 4
            0.3, 0.3, 0.3, 0.3, 0.7, 3.0, // row 5
        ];
        CscMatrix::from_triplets(6, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn perm_places_schur_tail_at_end_in_user_order() {
        let m = small_kkt();
        // Schur indices in a non-monotone order to verify the tail order
        // is preserved.
        let schur = vec![5, 4];
        let perm = compute_schur_aware_perm(&m, &schur).unwrap();
        assert_eq!(perm.len(), 6);
        // Last two positions = schur in user order.
        assert_eq!(perm[4], 5);
        assert_eq!(perm[5], 4);
        // Prefix is some permutation of {0, 1, 2, 3}.
        let mut prefix = perm[..4].to_vec();
        prefix.sort();
        assert_eq!(prefix, vec![0, 1, 2, 3]);
    }

    #[test]
    fn perm_is_a_valid_permutation() {
        let m = small_kkt();
        let schur = vec![4, 5];
        let perm = compute_schur_aware_perm(&m, &schur).unwrap();
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn empty_schur_falls_back_to_full_amd() {
        let m = small_kkt();
        let perm = compute_schur_aware_perm(&m, &[]).unwrap();
        assert_eq!(perm.len(), 6);
        // Valid permutation of 0..6.
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn duplicate_schur_indices_rejected() {
        let m = small_kkt();
        let r = compute_schur_aware_perm(&m, &[4, 4]);
        assert!(matches!(r, Err(RslabError::InvalidInput(_))));
    }

    #[test]
    fn out_of_range_schur_index_rejected() {
        let m = small_kkt();
        let r = compute_schur_aware_perm(&m, &[6]);
        assert!(matches!(r, Err(RslabError::InvalidInput(_))));
    }

    #[test]
    fn full_schur_rejected() {
        let m = small_kkt();
        let r = compute_schur_aware_perm(&m, &[0, 1, 2, 3, 4, 5]);
        assert!(matches!(r, Err(RslabError::InvalidInput(_))));
    }

    #[test]
    fn n_zero_with_empty_schur_returns_empty() {
        let m = CscMatrix::from_triplets(0, &[], &[], &[]).unwrap();
        let perm = compute_schur_aware_perm(&m, &[]).unwrap();
        assert!(perm.is_empty());
    }

    #[test]
    fn schur_size_one_works() {
        let m = small_kkt();
        let perm = compute_schur_aware_perm(&m, &[3]).unwrap();
        assert_eq!(perm.len(), 6);
        assert_eq!(perm[5], 3);
        // Prefix is a perm of {0, 1, 2, 4, 5}.
        let mut prefix = perm[..5].to_vec();
        prefix.sort();
        assert_eq!(prefix, vec![0, 1, 2, 4, 5]);
    }

    #[test]
    fn restrict_pattern_drops_schur_edges() {
        // Verify the subgraph restriction is correct: for the small KKT
        // with Schur = {4, 5}, the subgraph is the upper-left 4x4 which
        // is the identity (diagonal-only).
        let m = small_kkt();
        let full = m.symmetric_pattern();
        let mut sub_of = vec![usize::MAX; 6];
        sub_of[0] = 0;
        sub_of[1] = 1;
        sub_of[2] = 2;
        sub_of[3] = 3;
        let sub = restrict_pattern_to_subgraph(&full, &sub_of, 4);
        assert_eq!(sub.n, 4);
        // Diagonal entries - each column has exactly 1 entry (the diagonal).
        for j in 0..4 {
            let nnz_j = sub.col_ptr[j + 1] - sub.col_ptr[j];
            assert_eq!(nnz_j, 1, "column {} expected 1 entry, got {}", j, nnz_j);
            assert_eq!(sub.row_idx[sub.col_ptr[j]], j);
        }
    }
}
