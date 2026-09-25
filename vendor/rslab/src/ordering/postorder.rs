use super::elimination_tree::EliminationTree;

#[cfg(test)]
thread_local! {
    /// Work counter: total number of child-list elements
    /// materialized+sorted across all per-node sorts in [`postorder`].
    /// Linear in `n` for the sort-once-per-node traversal; quadratic for
    /// a sort-on-every-stack-visit version. Test-only; compiled out of
    /// production builds.
    static SORT_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Compute a postorder traversal of the elimination tree.
///
/// Returns `(postorder, inv_postorder)` where:
/// - `postorder[k]` = the node visited at position k (new-to-old)
/// - `inv_postorder[node]` = the position of node in the postorder (old-to-new)
///
/// Children are visited in order of ascending subtree size (smallest first).
pub fn postorder(etree: &EliminationTree) -> (Vec<usize>, Vec<usize>) {
    postorder_with(etree, |kids, sizes| {
        #[cfg(test)]
        SORT_WORK.with(|w| w.set(w.get() + kids.len()));
        kids.sort_unstable_by_key(|&c| sizes[c]);
    })
}

/// Merge-biased postorder.
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
    debug_assert_eq!(
        bias.len(),
        etree.n,
        "biased_postorder bias length must equal etree.n"
    );
    let mut late = Vec::new();
    postorder_with(etree, |kids, sizes| {
        // Unbiased children first, the biased ones (to be merged into the
        // parent) last, next to it; each part by subtree size.
        late.clear();
        late.extend(kids.iter().copied().filter(|&c| bias[c]));
        let mut w = 0;
        for r in 0..kids.len() {
            if !bias[kids[r]] {
                kids[w] = kids[r];
                w += 1;
            }
        }
        kids[w..].copy_from_slice(&late);
        let (early, tail) = kids.split_at_mut(w);
        early.sort_unstable_by_key(|&c| sizes[c]);
        tail.sort_unstable_by_key(|&c| sizes[c]);
    })
}

/// The postorder visiting each node's children in the order `order_children`
/// leaves them in (called once per node on its children, ascending, with the
/// subtree sizes); roots by subtree size. Returns `(postorder, inverse)`.
fn postorder_with(
    etree: &EliminationTree,
    mut order_children: impl FnMut(&mut [usize], &[usize]),
) -> (Vec<usize>, Vec<usize>) {
    let n = etree.n;
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let sizes = etree.subtree_sizes();
    let (ptr, mut idx) = etree.children_flat();
    for v in 0..n {
        order_children(&mut idx[ptr[v]..ptr[v + 1]], &sizes);
    }
    let mut roots = etree.roots();
    roots.sort_unstable_by_key(|&r| sizes[r]);
    let mut order = Vec::with_capacity(n);
    let mut next = ptr[..n].to_vec();
    let mut stack: Vec<usize> = Vec::new();
    for root in roots {
        stack.push(root);
        while let Some(&v) = stack.last() {
            if next[v] < ptr[v + 1] {
                stack.push(idx[next[v]]);
                next[v] += 1;
            } else {
                order.push(v);
                stack.pop();
            }
        }
    }
    let mut inv = vec![0usize; n];
    for (k, &v) in order.iter().enumerate() {
        inv[v] = k;
    }
    (order, inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::csc::CscMatrix;

    #[test]
    fn test_postorder_tridiagonal() {
        // Chain: 0->1->2->3. Postorder should be [0, 1, 2, 3].
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

    /// A `postorder` that re-cloned and re-sorted `children[node]` on every
    /// stack visit would make a node with `c` children (on top of the stack
    /// `c+1` times) pay O(c^2*log c). On a star etree (one root with `n-1`
    /// children) that is O(n^2*log n) - quadratic - in the symbolic
    /// pipeline.
    ///
    /// The check is deterministic via the `SORT_WORK` counter (total
    /// child-list elements materialized across all per-node sorts), so no
    /// flaky wall-clock timing is needed. Sorting per visit materializes the
    /// root's `(n-1)`-element child list `n` times -> `~n^2` elements;
    /// sorting once per node materializes it exactly once -> `~n` elements.
    /// The assertion `work <= 4*n` separates the two.
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

        // Child-sorting work is linear, not quadratic. Sort-on-every-visit
        // code would materialize ~n^2 elements here.
        assert!(
            work <= 4 * n,
            "postorder sort work {work} exceeds the linear bound {} (n={n}); \
             the O(n^2*log n) sort-on-every-stack-visit regression is back",
            4 * n
        );
    }
}
