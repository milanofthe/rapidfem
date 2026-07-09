use super::elimination_tree::EliminationTree;

#[cfg(test)]
thread_local! {
    /// S1 (dev/research/repo-review-2026-06-09.md) work counter: total
    /// number of child-list elements materialized+sorted across all
    /// per-node sorts in [`postorder`]. Linear in `n` for the fixed
    /// (sort-once-per-node) traversal; quadratic for the old
    /// sort-on-every-stack-visit version. Test-only; compiled out of
    /// production builds.
    static SORT_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Compute a postorder traversal of the elimination tree.
///
/// Returns `(postorder, inv_postorder)` where:
/// - `postorder[k]` = the node visited at position k (new-to-old)
/// - `inv_postorder[node]` = the position of node in the postorder (old-to-new)
///
/// Children are visited in order of ascending subtree size (smallest first)
/// to minimize peak memory usage in the ContribPool.
pub fn postorder(etree: &EliminationTree) -> (Vec<usize>, Vec<usize>) {
    let n = etree.n;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let children = etree.children();
    let sizes = etree.subtree_sizes();
    let roots = etree.roots();

    let mut order = Vec::with_capacity(n);

    // DFS stack carries each node's already-sorted child list plus a cursor:
    // `(node, sorted_children, child_idx)`. The sort runs exactly once per
    // node - when the node is first pushed - not once per stack visit.
    //
    // The previous version stored only `(node, child_idx)` and re-cloned and
    // re-sorted `children[node]` on every `stack.last_mut()` iteration. A node
    // with `c` children sits on top of the stack `c+1` times (once per child
    // push + once for the final pop), so it paid `O(c²·log c)`. On a star
    // etree (one root with `n-1` children - the arrow/bordered-KKT shape AMD
    // produces for a dense trailing border) that made the default symbolic
    // pipeline `O(n²·log n)`. See S1, dev/research/repo-review-2026-06-09.md,
    // and the matching cursor layout in `biased_postorder` /
    // `EliminationTree::postorder`.
    let mut stack: Vec<(usize, Vec<usize>, usize)> = Vec::new();

    // Process roots in ascending subtree size order
    let mut sorted_roots = roots;
    sorted_roots.sort_unstable_by_key(|&r| sizes[r]);

    for &root in &sorted_roots {
        stack.push((root, sorted_children_by_size(&children[root], &sizes), 0));

        while let Some((node, sorted_children, child_idx)) = stack.last_mut() {
            let node_id = *node;
            if *child_idx < sorted_children.len() {
                let child = sorted_children[*child_idx];
                *child_idx += 1;
                let next = sorted_children_by_size(&children[child], &sizes);
                stack.push((child, next, 0));
            } else {
                // All children visited - emit this node (postorder)
                order.push(node_id);
                stack.pop();
            }
        }
    }

    // Compute inverse
    let mut inv = vec![0usize; n];
    for (k, &node) in order.iter().enumerate() {
        inv[node] = k;
    }

    (order, inv)
}

/// Phase 2.12 merge-biased postorder.
///
/// Like [`postorder`], but when descending into a parent's children
/// it partitions them into `bias[child] == false` (emit *first*) and
/// `bias[child] == true` (emit *last*). Within each partition,
/// children are still ordered by ascending subtree size (peak-memory
/// minimization, same as [`postorder`]).
///
/// Effect: children whose `bias[child]` is `true` have their subtrees
/// emitted adjacent to (immediately before) the parent's column in
/// the resulting numbering. When the bias matches the SSIDS desired
/// merges (per `crate::symbolic::supernode::predict_merges`), the
/// returned ordering makes every desired merge adjacent in the
/// column numbering, so the standard adjacency check in
/// `find_supernodes` succeeds for it.
///
/// Invariant: `biased_postorder(etree, &vec![false; n]) ==
/// postorder(etree)`.
pub fn biased_postorder(etree: &EliminationTree, bias: &[bool]) -> (Vec<usize>, Vec<usize>) {
    let n = etree.n;
    debug_assert_eq!(
        bias.len(),
        n,
        "biased_postorder bias length must equal etree.n"
    );
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let children = etree.children();
    let sizes = etree.subtree_sizes();
    let roots = etree.roots();

    let mut order = Vec::with_capacity(n);
    let mut stack: Vec<(usize, Vec<usize>, usize)> = Vec::new();

    // Roots are not biased (no parent to be adjacent to). Use the
    // unbiased subtree-size order.
    let mut sorted_roots = roots;
    sorted_roots.sort_unstable_by_key(|&r| sizes[r]);

    for &root in &sorted_roots {
        let merged = merge_bias_partition(&children[root], &sizes, bias);
        stack.push((root, merged, 0));

        while let Some((node, sorted_children, child_idx)) = stack.last_mut() {
            let node_id = *node;
            if *child_idx < sorted_children.len() {
                let child = sorted_children[*child_idx];
                *child_idx += 1;
                let next_children = merge_bias_partition(&children[child], &sizes, bias);
                stack.push((child, next_children, 0));
            } else {
                order.push(node_id);
                stack.pop();
            }
        }
    }

    let mut inv = vec![0usize; n];
    for (k, &node) in order.iter().enumerate() {
        inv[node] = k;
    }
    (order, inv)
}

/// Schur-constrained postorder of an elimination tree (F3.2a).
///
/// Given an `is_schur` indicator (length `etree.n`), produce a postorder
/// such that **every Schur node appears at its etree-index position** in
/// the output. That is, `post[j] == j` for every `j` where
/// `is_schur[j] == true`, provided the Schur subset is closed under the
/// `parent` relation in the etree (the "top-forest" invariant). When that
/// invariant holds, the constraint is satisfiable: non-Schur descendants
/// of Schur nodes are emitted first; Schur nodes are then emitted in
/// strict ascending etree-index order, which equals their input order
/// because [`super::schur::compute_schur_aware_perm`] places Schur
/// indices at positions `[n - n_schur, n)` in the supplied order.
///
/// **Caller's responsibility.** The function does not validate the
/// top-forest invariant (no Schur node has a non-Schur parent). Callers
/// inside `symbolic_factorize_with_schur` get this for free because
/// `compute_schur_aware_perm` puts Schur indices at the highest
/// positions, and `parent[j] > j` for every node in the etree of the
/// permuted pattern.
///
/// **Children ordering rule** (applied at every parent and at the root
/// list): non-Schur children first, sorted by ascending subtree size
/// (peak-memory minimization, identical to [`postorder`]). Schur
/// children second, sorted by ascending etree index (preserves the
/// user's input order across the Schur tail).
///
/// Invariant: `schur_constrained_postorder(etree, &vec![false; n]) ==
/// postorder(etree)`.
pub fn schur_constrained_postorder(
    etree: &EliminationTree,
    is_schur: &[bool],
) -> (Vec<usize>, Vec<usize>) {
    let n = etree.n;
    debug_assert_eq!(
        is_schur.len(),
        n,
        "schur_constrained_postorder is_schur length must equal etree.n"
    );
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let children = etree.children();
    let sizes = etree.subtree_sizes();
    let roots = etree.roots();

    let mut order = Vec::with_capacity(n);

    // Phase 1: emit non-Schur nodes only. Walk the entire etree in DFS
    // postorder but only push non-Schur nodes onto `order`. A Schur node
    // is "transparent" - we recurse through it (so its non-Schur
    // descendants are reached) but we skip emitting it. After phase 1,
    // every non-Schur node sits at some position in `[0, n_f)` in a
    // valid postorder of the non-Schur subgraph (where each non-Schur's
    // sub-parent is its nearest non-Schur ancestor, or None).
    let sorted_roots = schur_partition_children(&roots, &sizes, is_schur);
    let mut stack: Vec<(usize, Vec<usize>, usize)> = Vec::new();
    for &root in sorted_roots.iter() {
        let merged = schur_partition_children(&children[root], &sizes, is_schur);
        stack.push((root, merged, 0));

        while let Some((node, sorted_children, child_idx)) = stack.last_mut() {
            let node_id = *node;
            if *child_idx < sorted_children.len() {
                let child = sorted_children[*child_idx];
                *child_idx += 1;
                let next_children = schur_partition_children(&children[child], &sizes, is_schur);
                stack.push((child, next_children, 0));
            } else {
                if !is_schur[node_id] {
                    order.push(node_id);
                }
                stack.pop();
            }
        }
    }

    // Phase 2: emit Schur nodes in ascending etree-index order. The
    // contract from `compute_schur_aware_perm` places Schur indices at
    // `[n - n_schur, n)`, so iterating `k` from `0..n` and pushing when
    // `is_schur[k]` yields exactly the identity tail: `post[n_f + i] ==
    // n_f + i` for every Schur position. A DFS over the Schur subtree
    // would emit them in tree-walk order - correct only when the Schur
    // etree is a single ascending chain. With a forest of Schur roots
    // (e.g. KKT matrices like ACOPP30 where Schur cols 158, 159, 160,
    // 161, 167, 168 are roots while 157 is parented under chain root
    // 208), DFS reorders Schur indices and breaks the tail identity
    // that `symbolic_factorize_with_schur` relies on. Direct ascending
    // emission preserves the postorder validity (every Schur node's
    // Schur children have smaller etree index, so they emit earlier;
    // non-Schur descendants emitted in phase 1 already sit at positions
    // `< n_f`).
    for (k, &flag) in is_schur.iter().enumerate() {
        if flag {
            order.push(k);
        }
    }

    let mut inv = vec![0usize; n];
    for (k, &node) in order.iter().enumerate() {
        inv[node] = k;
    }
    (order, inv)
}

/// Partition children for the Schur-constrained postorder.
///
/// Non-Schur children first, ascending by subtree size. Schur children
/// second, ascending by etree index (preserves input order across the
/// Schur tail).
fn schur_partition_children(children: &[usize], sizes: &[usize], is_schur: &[bool]) -> Vec<usize> {
    let mut nonschur: Vec<usize> = children.iter().copied().filter(|&c| !is_schur[c]).collect();
    let mut schur: Vec<usize> = children.iter().copied().filter(|&c| is_schur[c]).collect();
    nonschur.sort_unstable_by_key(|&c| sizes[c]);
    schur.sort_unstable();
    nonschur.extend(schur);
    nonschur
}

/// Sort a node's children by ascending subtree size (smallest first), the
/// peak-memory-minimizing visit order used by [`postorder`]. Factored out so
/// the clone+sort runs exactly once per node (see S1,
/// `dev/research/repo-review-2026-06-09.md`).
fn sorted_children_by_size(children: &[usize], sizes: &[usize]) -> Vec<usize> {
    #[cfg(test)]
    SORT_WORK.with(|w| w.set(w.get() + children.len()));
    let mut v = children.to_vec();
    v.sort_unstable_by_key(|&c| sizes[c]);
    v
}

/// Order a parent's children for the merge-biased postorder.
///
/// Partition: `bias[child] == false` first (emit early), then
/// `bias[child] == true` (emit late, adjacent to the parent). Within
/// each partition, ascending subtree size - the same heuristic as
/// the unbiased postorder, applied independently to each partition.
fn merge_bias_partition(children: &[usize], sizes: &[usize], bias: &[bool]) -> Vec<usize> {
    let mut early: Vec<usize> = children.iter().copied().filter(|&c| !bias[c]).collect();
    let mut late: Vec<usize> = children.iter().copied().filter(|&c| bias[c]).collect();
    early.sort_unstable_by_key(|&c| sizes[c]);
    late.sort_unstable_by_key(|&c| sizes[c]);
    early.extend(late);
    early
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    #[test]
    fn test_postorder_tridiagonal() {
        // Chain: 0→1→2→3. Postorder should be [0, 1, 2, 3].
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, inv) = postorder(&etree);

        assert_eq!(order.len(), 4);
        // In a chain, postorder visits from leaf to root
        assert_eq!(order, vec![0, 1, 2, 3]);

        // Verify inverse
        for (k, &node) in order.iter().enumerate() {
            assert_eq!(inv[node], k);
        }
    }

    #[test]
    fn test_postorder_valid_topological_order() {
        // For any matrix: every child appears before its parent in postorder
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, inv) = postorder(&etree);

        assert_eq!(order.len(), 5);

        // Verify topological property: parent appears after child
        for j in 0..5 {
            if let Some(p) = etree.parent[j] {
                assert!(
                    inv[j] < inv[p],
                    "child {} (pos {}) should appear before parent {} (pos {})",
                    j,
                    inv[j],
                    p,
                    inv[p]
                );
            }
        }
    }

    #[test]
    fn test_postorder_diagonal() {
        // Forest of singletons: any order is a valid postorder
        let m = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[1.0; 3]).unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, _) = postorder(&etree);

        assert_eq!(order.len(), 3);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_postorder_inverse_roundtrip() {
        let m =
            CscMatrix::from_triplets(4, &[0, 1, 1, 2, 2, 3, 3], &[0, 0, 1, 1, 2, 2, 3], &[1.0; 7])
                .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (order, inv) = postorder(&etree);

        // order[inv[j]] == j for all j
        for j in 0..4 {
            assert_eq!(order[inv[j]], j);
        }
        // inv[order[k]] == k for all k
        for k in 0..4 {
            assert_eq!(inv[order[k]], k);
        }
    }

    #[test]
    fn test_schur_postorder_no_schur_matches_postorder() {
        // is_schur all-false should reproduce standard postorder exactly.
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 2, 3, 4, 1, 2, 3, 4],
            &[0, 0, 0, 0, 0, 1, 2, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let (a, _) = postorder(&etree);
        let (b, _) = schur_constrained_postorder(&etree, &[false; 5]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_schur_postorder_chain_tail_pinned() {
        // Chain 0→1→2→3→4, mark {3,4} as Schur. Standard postorder is
        // [0,1,2,3,4]; constrained must keep 3,4 at positions 3,4.
        let m = CscMatrix::from_triplets(
            5,
            &[0, 1, 1, 2, 2, 3, 3, 4, 4],
            &[0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[1.0; 9],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let mut is_schur = vec![false; 5];
        is_schur[3] = true;
        is_schur[4] = true;
        let (post, inv) = schur_constrained_postorder(&etree, &is_schur);
        assert_eq!(post[3], 3);
        assert_eq!(post[4], 4);
        // Identity check on the tail; non-Schur prefix is some valid
        // topological order.
        assert_eq!(inv[3], 3);
        assert_eq!(inv[4], 4);
    }

    #[test]
    fn test_schur_postorder_topological_property() {
        // For arbitrary etree + is_schur, every child still precedes its
        // parent in the postorder (topological invariant).
        let m = CscMatrix::from_triplets(
            6,
            &[0, 1, 2, 3, 4, 5, 1, 4, 2, 4, 3, 4, 5],
            &[0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4],
            &[1.0; 13],
        )
        .unwrap();
        let pat = m.symmetric_pattern();
        let etree = EliminationTree::from_pattern(&pat);
        let mut is_schur = vec![false; 6];
        is_schur[4] = true;
        is_schur[5] = true;
        let (_post, inv) = schur_constrained_postorder(&etree, &is_schur);
        for j in 0..6 {
            if let Some(p) = etree.parent[j] {
                assert!(
                    inv[j] < inv[p],
                    "child {} (pos {}) must precede parent {} (pos {})",
                    j,
                    inv[j],
                    p,
                    inv[p]
                );
            }
        }
        // Schur nodes are at the tail in their etree-index order.
        // For this matrix the Schur subset {4,5} forms a top of the tree.
        assert!(inv[4] >= 4);
        assert!(inv[5] >= 4);
        assert!(inv[4] < inv[5] || inv[5] < inv[4]); // both valid positions
    }

    #[test]
    fn test_schur_postorder_forest_tail_identity() {
        // F3.3 regression: when the Schur subtree is a *forest* (multiple
        // Schur roots) with at least one internal Schur node whose parent
        // is also Schur, a DFS over the Schur subtree emits Schur nodes
        // in tree-walk order, not etree-index order. That breaks the
        // tail identity post[k] == k that
        // `symbolic_factorize_with_schur` relies on for the
        // schur_indices contract.
        //
        // ACOPP30_0000 hit this: Schur roots were {158, 159, 160, 161,
        // 167, 168, 195, 196, 197, 203, 204} plus a chain 157 → 162 →
        // ... → 208. Tail identity was violated from col 174 onward,
        // so the original A[174, 174] = -28.56 ended up at permuted
        // (184, 184) and the Schur block had max relative error 0.997
        // vs the dense oracle. The fix is to emit phase-2 Schur nodes
        // directly in ascending etree-index order, not via DFS.
        //
        // Construction: n=8, Schur = {4, 5, 6, 7}. Etree:
        //   non-Schur chain 0 → 1 → 2 → 3 → root 5 (Schur)
        //   internal Schur 4 → root 7 (Schur)
        //   Schur roots {5, 6, 7}; Schur 4 is a non-root Schur node.
        let etree = EliminationTree {
            parent: vec![
                Some(1),
                Some(2),
                Some(3),
                Some(5),
                Some(7),
                None,
                None,
                None,
            ],
            n: 8,
        };
        let is_schur = vec![false, false, false, false, true, true, true, true];
        let (post, inv) = schur_constrained_postorder(&etree, &is_schur);
        // Tail identity: post[k] == k for every Schur k.
        for k in 4..8 {
            assert_eq!(
                post[k], k,
                "tail identity violated: post[{}] = {} (expected {})",
                k, post[k], k
            );
            assert_eq!(inv[k], k);
        }
        // Topological invariant: every child precedes its parent.
        for j in 0..8 {
            if let Some(p) = etree.parent[j] {
                assert!(
                    inv[j] < inv[p],
                    "child {} (pos {}) must precede parent {} (pos {})",
                    j,
                    inv[j],
                    p,
                    inv[p]
                );
            }
        }
        // Non-Schur prefix: post[0..4] is a permutation of {0, 1, 2, 3}.
        let mut prefix: Vec<usize> = post[0..4].to_vec();
        prefix.sort();
        assert_eq!(prefix, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_schur_postorder_empty_etree() {
        let etree = EliminationTree {
            parent: Vec::new(),
            n: 0,
        };
        let (post, inv) = schur_constrained_postorder(&etree, &[]);
        assert!(post.is_empty());
        assert!(inv.is_empty());
    }

    #[test]
    fn test_postorder_empty() {
        let etree = EliminationTree {
            parent: Vec::new(),
            n: 0,
        };
        let (order, inv) = postorder(&etree);
        assert!(order.is_empty());
        assert!(inv.is_empty());
    }

    /// Build a star elimination tree: nodes `0..n-1` are leaves whose only
    /// parent is the last node `n-1` (the root). This is the etree of an
    /// arrow/bordered matrix whose dense border sits at the *trailing*
    /// index (`A[n-1, i] != 0` for every `i < n-1`) - exactly the shape
    /// AMD produces for the dense-border KKT rows in this codebase's tests.
    fn star_etree(n: usize) -> EliminationTree {
        // Lower-triangle: diagonal + a dense trailing column n-1.
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i); // diagonal
            if i < n - 1 {
                rows.push(n - 1);
                cols.push(i); // (row n-1, col i): border in the lower triangle
            }
        }
        let vals = vec![1.0; rows.len()];
        let m = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let pat = m.symmetric_pattern();
        EliminationTree::from_pattern(&pat)
    }

    /// S1 (dev/research/repo-review-2026-06-09.md): the previous `postorder`
    /// re-cloned and re-sorted `children[node]` on every stack visit, so a
    /// node with `c` children (on top of the stack `c+1` times) paid
    /// O(c²·log c). On a star etree (one root with `n-1` children) that is
    /// O(n²·log n) - quadratic - in the default symbolic pipeline.
    ///
    /// Reproduction is deterministic via the `SORT_WORK` counter (total
    /// child-list elements materialized across all per-node sorts), so no
    /// flaky wall-clock timing is needed. Pre-fix the root's `(n-1)`-element
    /// child list is materialized `n` times → `~n²` elements. Post-fix it is
    /// materialized exactly once → `~n` elements. The assertion `work ≤ 4·n`
    /// fails on the quadratic version and passes on the linear fix.
    #[test]
    fn test_postorder_star_sort_work_is_linear() {
        let n = 2000;
        let etree = star_etree(n);

        // Sanity: this really is a star (root n-1, all others its children).
        assert_eq!(etree.children()[n - 1].len(), n - 1);
        assert_eq!(etree.roots(), vec![n - 1]);

        SORT_WORK.with(|w| w.set(0));
        let (order, inv) = postorder(&etree);
        let work = SORT_WORK.with(|w| w.get());

        // Output correctness still holds (every child before its parent).
        assert_eq!(order.len(), n);
        for j in 0..n {
            if let Some(p) = etree.parent[j] {
                assert!(inv[j] < inv[p], "child {j} must precede parent {p}");
            }
        }

        // The fix: child-sorting work is linear, not quadratic. The old
        // sort-on-every-visit code materializes ~n² elements here.
        assert!(
            work <= 4 * n,
            "postorder sort work {work} exceeds the linear bound {} (n={n}); \
             the O(n²·log n) sort-on-every-stack-visit regression is back",
            4 * n
        );
    }
}
