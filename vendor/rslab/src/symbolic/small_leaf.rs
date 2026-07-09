//! Small-leaf-subtree batching (Phase 2.9).
//!
//! Groups consecutive true-leaf supernodes at the bottom of the
//! elimination tree so the numeric phase can factor them in a single
//! batched sweep instead of paying the full per-supernode dispatch
//! cost on each one.
//!
//! Research: `dev/research/phase-2.9-small-leaf-subtree.md`.
//! Plan:     `dev/plans/phase-2.9-small-leaf-subtree.md`.
//!
//! The grouping logic here is cheap (one pass over `supernodes`) and runs
//! unconditionally at symbolic time. Whether the groups are actually used is a
//! numeric-time choice; when unused, `small_leaf_groups` is ignored and the
//! regular per-supernode driver runs.

use crate::sparse::csc::CscPattern;
use crate::symbolic::supernode::Supernode;

/// Parameters controlling small-leaf grouping.
///
/// A supernode is "small" when `nrow <= nrow_max && ncol <= ncol_max`.
/// A group closes when cumulative `nrow * nrow` would exceed
/// `arena_budget`. Defaults are initial calibration values; they will
/// be tuned against the 154k-matrix bench.
#[derive(Debug, Clone)]
pub struct SmallLeafParams {
    pub nrow_max: usize,
    pub ncol_max: usize,
    pub arena_budget: usize,
}

impl Default for SmallLeafParams {
    fn default() -> Self {
        Self {
            nrow_max: 16,
            ncol_max: 8,
            arena_budget: 4096,
        }
    }
}

/// One batch of consecutive small-leaf supernodes.
#[derive(Debug, Clone)]
pub struct SmallLeafGroup {
    /// Indices into `SymbolicFactorization::supernodes`, in postorder.
    pub members: Vec<usize>,
    /// Per-leaf precomputed frontal row layouts. `member_rows[k]` is
    /// the full row_indices Vec for `members[k]` - what the numeric
    /// `build_row_indices` would produce, but computed once at
    /// symbolic time so the batched numeric path can reuse it across
    /// every factorization that shares this `SymbolicFactorization`.
    ///
    /// Layout is `[first_col..first_col+ncol, trailing rows sorted]`,
    /// identical to the build_row_indices output for a leaf (no
    /// children, no delayed columns).
    ///
    /// The actual per-member frontal dimension is
    /// `member_rows[k].len()`, which may differ from the symbolic
    /// `nrow` stored on the Supernode (that field is a placeholder
    /// sized from `col_counts` and approximates but does not
    /// necessarily equal the true A-pattern row set).
    pub member_rows: Vec<Vec<usize>>,
    /// Sum of per-leaf `nrow_actual * nrow_actual` (where
    /// `nrow_actual == member_rows[k].len()`). Used by the numeric
    /// driver as a sizing hint for the pooled workspace.
    pub arena_size: usize,
    /// Per-leaf offsets into the arena. Length == `members.len() + 1`;
    /// `offsets[k]..offsets[k+1]` is the slice for `members[k]`.
    /// Computed from `member_rows[k].len()^2` prefix-summed.
    pub offsets: Vec<usize>,
}

/// Identify small-leaf groups in a supernode forest.
///
/// Returns:
///
/// * `groups` - the groups, in postorder. Each group carries the
///   precomputed per-member row_indices so the numeric batched path
///   can skip `build_row_indices`.
/// * `snode_group` - for each supernode, `Some(g)` if it belongs to
///   `groups[g]`, else `None`.
///
/// A supernode qualifies as a batch member iff:
///
///   1. `children.is_empty()` (true leaf - no contributions to
///      assemble).
///   2. `ncol <= params.ncol_max`.
///   3. `nrow <= params.nrow_max`.
///
/// Adjacent qualifying supernodes in postorder are greedily packed
/// into a single group until adding the next would push the arena
/// size over `params.arena_budget`, at which point a new group is
/// started.
///
/// A non-leaf or over-size supernode breaks the current group even
/// if the group has capacity - the batched path assumes strict
/// postorder adjacency so that non-grouped siblings can be processed
/// in the regular per-supernode loop without gap tracking.
///
/// Row layout is computed from the symmetric `permuted_pattern`
/// (not the matrix values) so the groups are value-agnostic and
/// reusable across IPM iterations via `SymbolicFactorization`
/// caching.
pub fn find_small_leaf_groups(
    supernodes: &[Supernode],
    permuted_pattern: &CscPattern,
    params: &SmallLeafParams,
) -> (Vec<SmallLeafGroup>, Vec<Option<usize>>) {
    let mut groups: Vec<SmallLeafGroup> = Vec::new();
    let mut snode_group: Vec<Option<usize>> = vec![None; supernodes.len()];

    // Pooled scratch for leaf row computation. Entries kept at
    // `false` between leaves; touched entries cleared explicitly so
    // the full `vec![false; n]` is paid once.
    let mut seen: Vec<bool> = vec![false; permuted_pattern.n];
    let mut trailing: Vec<usize> = Vec::new();

    let mut current: Option<SmallLeafGroup> = None;

    // Flush the currently-open group into `groups`, updating
    // `snode_group`. Local closure so every early-close site uses
    // the same path.
    let flush = |current: &mut Option<SmallLeafGroup>,
                 groups: &mut Vec<SmallLeafGroup>,
                 snode_group: &mut [Option<usize>]| {
        if let Some(g) = current.take() {
            let gid = groups.len();
            for &m in &g.members {
                snode_group[m] = Some(gid);
            }
            groups.push(g);
        }
    };

    for (idx, snode) in supernodes.iter().enumerate() {
        let qualifies = snode.children.is_empty()
            && snode.ncol <= params.ncol_max
            && snode.nrow <= params.nrow_max
            && snode.nrow > 0;

        if !qualifies {
            flush(&mut current, &mut groups, &mut snode_group);
            continue;
        }

        // Compute the true row_indices for this leaf. Equivalent to
        // `build_row_indices` for a leaf (no children, no delayed).
        let rows = compute_leaf_rows(snode, permuted_pattern, &mut seen, &mut trailing);
        let leaf_size = rows.len() * rows.len();

        // Start a new group if none is open, or if the next leaf would
        // overflow the arena budget.
        let must_close = match &current {
            Some(g) => g.arena_size + leaf_size > params.arena_budget,
            None => false,
        };
        if must_close {
            flush(&mut current, &mut groups, &mut snode_group);
        }

        let g = current.get_or_insert_with(|| SmallLeafGroup {
            members: Vec::new(),
            member_rows: Vec::new(),
            arena_size: 0,
            offsets: vec![0],
        });
        g.members.push(idx);
        g.member_rows.push(rows);
        g.arena_size += leaf_size;
        g.offsets.push(g.arena_size);
    }

    flush(&mut current, &mut groups, &mut snode_group);

    (groups, snode_group)
}

/// Compute the frontal row layout for a single leaf supernode.
///
/// Mirrors `numeric::factorize::build_row_indices` restricted to the
/// leaf case: no children, no delayed columns. The layout is:
///
/// ```text
/// [first_col..first_col+ncol]   // own columns
/// [trailing rows, sorted]       // non-own rows from the A-pattern
/// ```
///
/// `seen` and `trailing` are caller-owned pooled scratch. On entry
/// `seen` must be all-`false`; this function restores that invariant
/// before returning.
fn compute_leaf_rows(
    snode: &Supernode,
    pattern: &CscPattern,
    seen: &mut [bool],
    trailing: &mut Vec<usize>,
) -> Vec<usize> {
    let first_col = snode.first_col;
    let ncol = snode.ncol;

    // Mark own columns as "seen" so the trailing scan skips them.
    for s in seen.iter_mut().skip(first_col).take(ncol) {
        *s = true;
    }

    trailing.clear();
    for j in first_col..first_col + ncol {
        for k in pattern.col_ptr[j]..pattern.col_ptr[j + 1] {
            let r = pattern.row_idx[k];
            if !seen[r] {
                seen[r] = true;
                trailing.push(r);
            }
        }
    }
    trailing.sort_unstable();

    let mut rows = Vec::with_capacity(ncol + trailing.len());
    rows.extend(first_col..first_col + ncol);
    rows.extend_from_slice(trailing);

    // Restore the `seen` invariant.
    for s in seen.iter_mut().skip(first_col).take(ncol) {
        *s = false;
    }
    for &r in trailing.iter() {
        seen[r] = false;
    }

    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small leaf at column offset `first_col`. `ncol == nrow`
    /// and the pattern is a self-diagonal block - the simplest
    /// pattern that produces row_indices equal to just the own cols.
    fn mk_leaf(first_col: usize, ncol: usize, nrow: usize) -> Supernode {
        Supernode {
            first_col,
            ncol,
            nrow,
            row_indices: Vec::new(),
            children: Vec::new(),
            delayed_capacity: usize::MAX,
        }
    }

    fn mk_nonleaf(first_col: usize, ncol: usize, nrow: usize) -> Supernode {
        Supernode {
            first_col,
            ncol,
            nrow,
            row_indices: Vec::new(),
            children: vec![0],
            delayed_capacity: usize::MAX,
        }
    }

    /// Diagonal pattern spanning `n` columns. Each column has a single
    /// diagonal entry, so every leaf's row_indices is exactly its own
    /// column range.
    fn diag_pattern(n: usize) -> CscPattern {
        let col_ptr: Vec<usize> = (0..=n).collect();
        let row_idx: Vec<usize> = (0..n).collect();
        CscPattern {
            n,
            col_ptr,
            row_idx,
        }
    }

    #[test]
    fn empty_input_yields_empty_output() {
        let pat = diag_pattern(0);
        let (g, m) = find_small_leaf_groups(&[], &pat, &SmallLeafParams::default());
        assert!(g.is_empty());
        assert!(m.is_empty());
    }

    #[test]
    fn all_small_leaves_pack_into_one_group() {
        // Five tiny leaves in a diagonal pattern: row_indices for each
        // equals its own column range, so arena_size sums ncol².
        let snodes = vec![
            mk_leaf(0, 2, 2),
            mk_leaf(2, 2, 2),
            mk_leaf(4, 2, 2),
            mk_leaf(6, 1, 1),
            mk_leaf(7, 2, 2),
        ];
        let pat = diag_pattern(9);
        let (groups, snode_group) =
            find_small_leaf_groups(&snodes, &pat, &SmallLeafParams::default());
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members, vec![0, 1, 2, 3, 4]);
        let expected_arena = 4 + 4 + 4 + 1 + 4;
        assert_eq!(groups[0].arena_size, expected_arena);
        assert_eq!(groups[0].offsets.len(), 6);
        assert_eq!(groups[0].offsets.first(), Some(&0));
        assert_eq!(groups[0].offsets.last(), Some(&expected_arena));
        for i in 0..5 {
            assert_eq!(snode_group[i], Some(0));
        }
        // Row layouts: for a diagonal pattern, each leaf's rows are
        // just its own column range.
        assert_eq!(groups[0].member_rows[0], vec![0, 1]);
        assert_eq!(groups[0].member_rows[3], vec![6]);
    }

    #[test]
    fn non_leaf_breaks_group() {
        // Leaf, leaf, non-leaf, leaf, leaf → two groups.
        let snodes = vec![
            mk_leaf(0, 2, 2),
            mk_leaf(2, 2, 2),
            mk_nonleaf(4, 3, 3),
            mk_leaf(7, 2, 2),
            mk_leaf(9, 2, 2),
        ];
        let pat = diag_pattern(11);
        let (groups, snode_group) =
            find_small_leaf_groups(&snodes, &pat, &SmallLeafParams::default());
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].members, vec![0, 1]);
        assert_eq!(groups[1].members, vec![3, 4]);
        assert_eq!(snode_group[2], None);
    }

    #[test]
    fn oversize_leaf_breaks_group() {
        // ncol=9 exceeds the default 8 ncol_max.
        let snodes = vec![mk_leaf(0, 2, 2), mk_leaf(2, 9, 9), mk_leaf(11, 2, 2)];
        let pat = diag_pattern(13);
        let (groups, snode_group) =
            find_small_leaf_groups(&snodes, &pat, &SmallLeafParams::default());
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].members, vec![0]);
        assert_eq!(groups[1].members, vec![2]);
        assert_eq!(snode_group[1], None);
    }

    #[test]
    fn budget_forces_split() {
        let params = SmallLeafParams {
            nrow_max: 16,
            ncol_max: 8,
            arena_budget: 5,
        };
        // Three 2×2 leaves (4 each): first fits alone, second would
        // push arena to 8 > 5 → new group.
        let snodes = vec![mk_leaf(0, 2, 2), mk_leaf(2, 2, 2), mk_leaf(4, 2, 2)];
        let pat = diag_pattern(6);
        let (groups, snode_group) = find_small_leaf_groups(&snodes, &pat, &params);
        assert_eq!(groups.len(), 3);
        assert_eq!(snode_group[0], Some(0));
        assert_eq!(snode_group[1], Some(1));
        assert_eq!(snode_group[2], Some(2));
    }

    #[test]
    fn zero_nrow_is_skipped() {
        let snodes = vec![mk_leaf(0, 0, 0), mk_leaf(0, 2, 2)];
        let pat = diag_pattern(2);
        let (groups, snode_group) =
            find_small_leaf_groups(&snodes, &pat, &SmallLeafParams::default());
        assert_eq!(snode_group[0], None);
        assert_eq!(snode_group[1], Some(0));
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members, vec![1]);
    }

    #[test]
    fn offsets_are_prefix_sums() {
        let snodes = vec![mk_leaf(0, 2, 2), mk_leaf(2, 3, 3), mk_leaf(5, 4, 4)];
        let pat = diag_pattern(9);
        let (groups, _) = find_small_leaf_groups(&snodes, &pat, &SmallLeafParams::default());
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.offsets, vec![0, 4, 4 + 9, 4 + 9 + 16]);
        assert_eq!(g.arena_size, *g.offsets.last().unwrap());
        for w in g.offsets.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn compute_leaf_rows_with_offdiagonal_nonzeros() {
        // Leaf at cols 2..4, with off-diagonal nonzeros pulling in
        // rows 5 and 7. Verify the returned layout is
        // [own cols | sorted trailing].
        // Pattern: col 2 has rows [2, 7]; col 3 has rows [3, 5, 7];
        // cols 5 and 7 just have their diagonal.
        let col_ptr = vec![0, 0, 0, 2, 5, 5, 6, 6, 7];
        let row_idx = vec![2, 7, 3, 5, 7, 5, 7];
        let pat = CscPattern {
            n: 8,
            col_ptr,
            row_idx,
        };
        let leaf = mk_leaf(2, 2, 2);
        let mut seen = vec![false; 8];
        let mut trailing = Vec::new();
        let rows = compute_leaf_rows(&leaf, &pat, &mut seen, &mut trailing);
        assert_eq!(rows, vec![2, 3, 5, 7]);
        assert!(seen.iter().all(|&b| !b), "seen invariant restored");
    }
}
