//! Block triangular form (BTF) for unsymmetric matrices.
//!
//! Two purely structural, sequential, deterministic passes used by the KLU
//! path ([`crate::numeric::klu`]):
//!
//! 1. **Maximum transversal** ([`max_transversal`]): an MC21-style
//!    augmenting-path matching that pairs every column with a distinct row
//!    holding a structural nonzero, giving the permuted matrix a zero-free
//!    diagonal. An incomplete matching proves the matrix structurally
//!    singular (its structural rank is the matching size).
//! 2. **Tarjan SCC** ([`block_triangular_form`]): strongly connected
//!    components of the matched matrix's directed graph, emitted in reverse
//!    topological order, yield the symmetric permutation to **block upper
//!    triangular** form: all entries outside the diagonal blocks lie above
//!    them, and each diagonal block is irreducible (strongly connected).
//!
//! Both passes are iterative (explicit stacks, no recursion, safe on
//! arbitrarily deep graphs) and allocation-linear in `n + nnz`.

/// Result of the BTF permutation of an `nxn` pattern.
#[derive(Debug, Clone)]
pub(crate) struct BtfForm {
    /// Row permutation, new-to-old: permuted row `k` is original row
    /// `row_perm[k]`.
    pub row_perm: Vec<usize>,
    /// Column permutation, new-to-old.
    pub col_perm: Vec<usize>,
    /// Diagonal-block boundaries in the permuted matrix: block `b` spans
    /// rows/columns `block_ptr[b]..block_ptr[b + 1]`. `block_ptr.len()` is
    /// `n_blocks + 1`, first entry `0`, last entry `n`.
    pub block_ptr: Vec<usize>,
}

/// Maximum transversal on a CSC pattern (default candidate: diagonal-seeded
/// Hopcroft-Karp). See [`hk_transversal`] and [`matching_candidates`].
#[cfg(test)]
pub(crate) fn max_transversal(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> (Vec<usize>, usize) {
    hk_transversal(n, col_ptr, row_idx, true)
}

/// Hopcroft-Karp maximum transversal on a CSC pattern.
///
/// Returns `row_match` where `row_match[j]` is the row matched to column `j`
/// (`usize::MAX` if unmatched), together with the number of matched columns
/// (the structural rank). An optional diagonal seed plus a first-free greedy
/// pass seed the matching; the rest runs Hopcroft-Karp phases: a BFS
/// layering over alternating paths from all free columns, then one sweep of
/// vertex-disjoint shortest augmenting DFS, `O(sqrt(n) * nnz)` worst case.
/// (The classic MC21 one-column-at-a-time DFS degenerates on the
/// harmonic-balance circuit class - seconds instead of tens of milliseconds
/// on `twotone` - because late columns re-walk ever longer alternating
/// paths; phases amortize that.) Fully deterministic: columns are processed
/// in natural order and adjacency in stored order.
pub(crate) fn hk_transversal(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    diag_seed: bool,
) -> (Vec<usize>, usize) {
    const UNMATCHED: usize = usize::MAX;
    let mut row_match = vec![UNMATCHED; n]; // column -> row
    let mut col_of_row = vec![UNMATCHED; n]; // row -> column
    let mut n_matched = 0usize;

    // Seed pass 1: match every column to its structural diagonal entry when
    // present. On circuit matrices the diagonal is the physically meaningful
    // pairing (MNA conductances) - it is what the KLU pivoting prefers later
    // - and seeding it keeps the final matching (and thus the per-block
    // symmetrized pattern AMD orders) from drifting on augmenting-path
    // accidents.
    if diag_seed {
        for j in 0..n {
            for &r in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
                if r == j {
                    col_of_row[j] = j;
                    row_match[j] = j;
                    n_matched += 1;
                    break;
                }
            }
        }
    }
    // Seed pass 2: first free row in each remaining column.
    for j in 0..n {
        if row_match[j] != UNMATCHED {
            continue;
        }
        for &r in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
            if col_of_row[r] == UNMATCHED {
                col_of_row[r] = j;
                row_match[j] = r;
                n_matched += 1;
                break;
            }
        }
    }
    if n_matched == n {
        return (row_match, n_matched);
    }

    // Hopcroft-Karp phases. `level[c]` is the alternating-path BFS distance
    // of column `c` from the free columns this phase (`UNSET` = unreached,
    // or consumed by an earlier DFS of the same phase).
    const UNSET: usize = usize::MAX;
    let mut level = vec![UNSET; n];
    let mut queue: Vec<usize> = Vec::with_capacity(n);
    // Per-phase adjacency cursor: every edge is scanned at most once per
    // phase across all DFS of that phase.
    let mut ptr: Vec<usize> = vec![0; n];
    let mut col_stack = vec![0usize; n];
    let mut path_row = vec![0usize; n];

    loop {
        // BFS layering from all free columns, stopping at the level of the
        // shortest augmenting path (deeper layers cannot host a shortest
        // path and would only waste work).
        level.iter_mut().for_each(|v| *v = UNSET);
        queue.clear();
        for c in 0..n {
            if row_match[c] == UNMATCHED {
                level[c] = 0;
                queue.push(c);
            }
        }
        let mut shortest = UNSET;
        let mut head = 0usize;
        while head < queue.len() {
            let c = queue[head];
            head += 1;
            if level[c] >= shortest {
                break;
            }
            for &r in &row_idx[col_ptr[c]..col_ptr[c + 1]] {
                let c2 = col_of_row[r];
                if c2 == UNMATCHED {
                    if shortest == UNSET {
                        shortest = level[c];
                    }
                } else if level[c2] == UNSET {
                    level[c2] = level[c] + 1;
                    queue.push(c2);
                }
            }
        }
        if shortest == UNSET {
            break; // no augmenting path left: the matching is maximum
        }

        // Vertex-disjoint shortest augmenting DFS from every free column,
        // following strictly increasing levels. Exhausted or used columns
        // drop out of the phase via `level = UNSET`.
        ptr[..n].copy_from_slice(&col_ptr[..n]);
        for jstart in 0..n {
            if row_match[jstart] != UNMATCHED || level[jstart] != 0 {
                continue;
            }
            let mut depth = 0usize;
            col_stack[0] = jstart;
            'dfs: loop {
                let c = col_stack[depth];
                let mut descended = false;
                while ptr[c] < col_ptr[c + 1] {
                    let r = row_idx[ptr[c]];
                    ptr[c] += 1;
                    let c2 = col_of_row[r];
                    if c2 == UNMATCHED {
                        // Augment along the stack; the used columns leave
                        // the phase (vertex-disjointness).
                        path_row[depth] = r;
                        for k in (0..=depth).rev() {
                            let ck = col_stack[k];
                            let rk = path_row[k];
                            col_of_row[rk] = ck;
                            row_match[ck] = rk;
                            level[ck] = UNSET;
                        }
                        n_matched += 1;
                        break 'dfs;
                    }
                    if level[c2] == level[c] + 1 {
                        path_row[depth] = r;
                        depth += 1;
                        col_stack[depth] = c2;
                        descended = true;
                        break;
                    }
                }
                if descended {
                    continue;
                }
                // Column exhausted for this phase.
                level[c] = UNSET;
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
        }
    }
    (row_match, n_matched)
}

/// MC21 maximum transversal (first-free greedy seed + one-column-at-a-time
/// augmenting DFS) with a deterministic work budget in adjacency steps.
///
/// MC21 walks different alternating paths than Hopcroft-Karp and therefore
/// lands on a different (equally maximum) matching - on some circuit
/// families (`onetone2`) a markedly better one for the downstream per-block
/// AMD. It is kept as a *candidate* for [`matching_candidates`]; the budget
/// bounds its known degeneration (`twotone`-class), where it returns `None`
/// and the candidate is simply dropped.
pub(crate) fn mc21_transversal(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    budget: usize,
) -> Option<(Vec<usize>, usize)> {
    const UNMATCHED: usize = usize::MAX;
    let mut row_match = vec![UNMATCHED; n];
    let mut col_of_row = vec![UNMATCHED; n];
    let mut n_matched = 0usize;
    for j in 0..n {
        for &r in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
            if col_of_row[r] == UNMATCHED {
                col_of_row[r] = j;
                row_match[j] = r;
                n_matched += 1;
                break;
            }
        }
    }
    if n_matched < n {
        n_matched = mc21_augment(
            n,
            col_ptr,
            row_idx,
            &mut row_match,
            &mut col_of_row,
            n_matched,
            budget,
        )?;
    }
    Some((row_match, n_matched))
}

/// MC21 augmenting-path DFS (one free column at a time, global cheap
/// lookahead), bounded by `budget` adjacency steps (`None` on exhaustion).
#[allow(clippy::too_many_arguments)]
fn mc21_augment(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    row_match: &mut [usize],
    col_of_row: &mut [usize],
    mut n_matched: usize,
    budget: usize,
) -> Option<usize> {
    const UNMATCHED: usize = usize::MAX;
    let mut work = 0usize;
    let mut visited = vec![UNMATCHED; n];
    let mut cheap: Vec<usize> = col_ptr[..n].to_vec();
    let mut col_stack = vec![0usize; n];
    let mut cursor = vec![0usize; n];
    let mut path_row = vec![0usize; n];

    for jstart in 0..n {
        if row_match[jstart] != UNMATCHED {
            continue;
        }
        let stamp = jstart;
        let mut depth = 0usize;
        col_stack[0] = jstart;
        cursor[0] = col_ptr[jstart];
        visited[jstart] = stamp;
        let mut augment_depth = None;

        'dfs: while augment_depth.is_none() {
            let c = col_stack[depth];
            while cheap[c] < col_ptr[c + 1] {
                let r = row_idx[cheap[c]];
                cheap[c] += 1;
                work += 1;
                if col_of_row[r] == UNMATCHED {
                    path_row[depth] = r;
                    augment_depth = Some(depth);
                    continue 'dfs;
                }
            }
            let mut advanced = false;
            while cursor[depth] < col_ptr[c + 1] {
                let r = row_idx[cursor[depth]];
                cursor[depth] += 1;
                work += 1;
                let c2 = col_of_row[r];
                if visited[c2] == stamp {
                    continue;
                }
                visited[c2] = stamp;
                path_row[depth] = r;
                depth += 1;
                col_stack[depth] = c2;
                cursor[depth] = col_ptr[c2];
                advanced = true;
                break;
            }
            if advanced {
                continue;
            }
            if depth == 0 {
                break;
            }
            depth -= 1;
        }
        if work > budget {
            return None;
        }

        if let Some(d) = augment_depth {
            for k in (0..=d).rev() {
                let c = col_stack[k];
                let r = path_row[k];
                col_of_row[r] = c;
                row_match[c] = r;
            }
            n_matched += 1;
        }
    }
    Some(n_matched)
}

/// Deterministic matching candidates for the KLU analyze bakeoff, or `None`
/// when the pattern is structurally singular.
///
/// Fast path: a complete structural diagonal yields the single identity
/// matching (the typical MNA case) at seed-pass cost. Otherwise up to three
/// distinct maximum matchings: diagonal-seeded HK, free-seeded HK, and
/// budgeted MC21 (dropped when its budget runs out). Different maximum
/// matchings hand the downstream per-block AMD different symmetrized
/// patterns - with several-x fill differences on the harmonic-balance
/// class - so the caller scores the candidates and keeps the cheapest.
pub(crate) fn matching_candidates(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> Option<Vec<Vec<usize>>> {
    let (m1, k1) = hk_transversal(n, col_ptr, row_idx, true);
    if k1 != n {
        return None;
    }
    // Complete structural diagonal: the diagonal-seeded matching IS the
    // identity, the canonical pairing; alternates would only wander off it.
    if m1.iter().enumerate().all(|(j, &r)| r == j) {
        return Some(vec![m1]);
    }
    let mut out = vec![m1];
    let (m2, k2) = hk_transversal(n, col_ptr, row_idx, false);
    if k2 == n && !out.contains(&m2) {
        out.push(m2);
    }
    // Budget in adjacency steps: sized so the class where MC21's matching
    // wins (onetone: ~34M steps) completes, while its degenerate class
    // (twotone: >300M) is cut off after a hard-capped prefix (~150ms)
    // instead of burning seconds. MC21 is an opportunistic extra candidate;
    // losing it to the budget only removes one bakeoff entrant.
    let nnz = col_ptr[n];
    let budget = (128 * (nnz + n)).min(50_000_000);
    if let Some((m3, k3)) = mc21_transversal(n, col_ptr, row_idx, budget) {
        if k3 == n && !out.contains(&m3) {
            out.push(m3);
        }
    }
    Some(out)
}

/// Compute the block upper triangular form of a structurally nonsingular CSC
/// pattern. Returns `None` if the pattern is structurally singular (see
/// [`max_transversal`]); the caller maps that to its error type.
///
/// The row permutation composes the matching with the SCC order; the column
/// permutation is the SCC order alone, so `row_perm[k]` and `col_perm[k]`
/// address the same diagonal-block slot `k`.
#[cfg(test)]
pub(crate) fn block_triangular_form(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
) -> Option<BtfForm> {
    if n == 0 {
        return Some(BtfForm {
            row_perm: Vec::new(),
            col_perm: Vec::new(),
            block_ptr: vec![0],
        });
    }
    let (row_match, n_matched) = max_transversal(n, col_ptr, row_idx);
    if n_matched != n {
        return None;
    }
    Some(btf_from_matching(n, col_ptr, row_idx, row_match))
}

/// Tarjan SCC pass over the matched digraph: turns one maximum matching into
/// the block-upper-triangular permutation pair (see module docs).
///
/// The row permutation composes the matching with the SCC order; the column
/// permutation is the SCC order alone, so `row_perm[k]` and `col_perm[k]`
/// address the same diagonal-block slot `k`.
pub(crate) fn btf_from_matching(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    row_match: Vec<usize>,
) -> BtfForm {
    if n == 0 {
        return BtfForm {
            row_perm: Vec::new(),
            col_perm: Vec::new(),
            block_ptr: vec![0],
        };
    }
    // col_of_row[r] = the column matched to row r: node id of row r in the
    // matched digraph (node per column; edge j -> col_of_row[i] for each
    // stored entry (i, j)). 32-bit working arrays throughout: the pass is
    // bound by the random `col_of_row`/`index`/`lowlink` accesses, and the
    // KLU caller gates `n` to the 31-bit range before calling in.
    debug_assert!(n < u32::MAX as usize);
    let mut col_of_row = vec![0u32; n];
    for (j, &r) in row_match.iter().enumerate() {
        col_of_row[r] = j as u32;
    }

    // Iterative Tarjan. Components are emitted in reverse topological order
    // of the edge direction above; placing them in emission order makes every
    // cross-block entry land ABOVE its diagonal block (block upper
    // triangular), see the module docs.
    const UNSET32: u32 = u32::MAX;
    let mut index = vec![UNSET32; n]; // discovery index
    let mut lowlink = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut scc_stack: Vec<u32> = Vec::with_capacity(n);
    // DFS state: (node, adjacency cursor) frames.
    let mut frame_node = vec![0u32; n];
    let mut frame_cursor = vec![0usize; n];
    let mut next_index = 0u32;
    let mut col_perm: Vec<usize> = Vec::with_capacity(n);
    let mut block_ptr: Vec<usize> = vec![0];

    for root in 0..n {
        if index[root] != UNSET32 {
            continue;
        }
        let mut top = 0usize;
        frame_node[0] = root as u32;
        frame_cursor[0] = col_ptr[root];
        index[root] = next_index;
        lowlink[root] = next_index;
        next_index += 1;
        scc_stack.push(root as u32);
        on_stack[root] = true;

        loop {
            let v = frame_node[top] as usize;
            let mut descended = false;
            while frame_cursor[top] < col_ptr[v + 1] {
                let w = col_of_row[row_idx[frame_cursor[top]]] as usize;
                frame_cursor[top] += 1;
                if index[w] == UNSET32 {
                    index[w] = next_index;
                    lowlink[w] = next_index;
                    next_index += 1;
                    scc_stack.push(w as u32);
                    on_stack[w] = true;
                    top += 1;
                    frame_node[top] = w as u32;
                    frame_cursor[top] = col_ptr[w];
                    descended = true;
                    break;
                } else if on_stack[w] && index[w] < lowlink[v] {
                    lowlink[v] = index[w];
                }
            }
            if descended {
                continue;
            }
            // v is finished: emit its SCC if it is a root.
            if lowlink[v] == index[v] {
                let block_start = col_perm.len();
                loop {
                    // Tarjan invariant: `v` was pushed when discovered and is
                    // still on the stack, so the pop cannot fail. A violation
                    // is a bug in this function - it must NOT be reported as
                    // structural singularity (the `None` return means that).
                    let Some(w) = scc_stack.pop() else {
                        unreachable!("Tarjan: v is on the stack");
                    };
                    let w = w as usize;
                    on_stack[w] = false;
                    col_perm.push(w);
                    if w == v {
                        break;
                    }
                }
                // Tarjan pops the SCC in reverse discovery order; restore
                // discovery order inside the block so the permutation is
                // independent of stack mechanics (and stable for tests).
                col_perm[block_start..].sort_unstable_by_key(|&w| index[w]);
                block_ptr.push(col_perm.len());
            }
            if top == 0 {
                break;
            }
            top -= 1;
            let parent = frame_node[top] as usize;
            if lowlink[v] < lowlink[parent] {
                lowlink[parent] = lowlink[v];
            }
        }
    }
    debug_assert_eq!(col_perm.len(), n);

    let row_perm: Vec<usize> = col_perm.iter().map(|&j| row_match[j]).collect();
    BtfForm {
        row_perm,
        col_perm,
        block_ptr,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CSC pattern from (row, col) pairs.
    fn pattern(n: usize, entries: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(r, c) in entries {
            cols[c].push(r);
        }
        let mut col_ptr = vec![0usize];
        let mut row_idx = Vec::new();
        for c in cols {
            let mut c = c;
            c.sort_unstable();
            row_idx.extend_from_slice(&c);
            col_ptr.push(row_idx.len());
        }
        (col_ptr, row_idx)
    }

    #[test]
    fn transversal_identity_diagonal() {
        let (cp, ri) = pattern(3, &[(0, 0), (1, 1), (2, 2), (1, 0)]);
        let (m, k) = max_transversal(3, &cp, &ri);
        assert_eq!(k, 3);
        assert_eq!(m, vec![0, 1, 2]);
    }

    #[test]
    fn transversal_needs_augmenting_path() {
        // Column 0 -> rows {0,1}, column 1 -> row {0}: greedy matches (0,0)
        // and must re-route via an augmenting path to (0->1, 1->0).
        let (cp, ri) = pattern(2, &[(0, 0), (1, 0), (0, 1)]);
        let (m, k) = max_transversal(2, &cp, &ri);
        assert_eq!(k, 2);
        assert_eq!(m, vec![1, 0]);
    }

    #[test]
    fn transversal_detects_structural_singularity() {
        // Column 2 is empty.
        let (cp, ri) = pattern(3, &[(0, 0), (1, 1), (2, 0)]);
        let (_, k) = max_transversal(3, &cp, &ri);
        assert_eq!(k, 2);
    }

    /// Check the BTF invariant: every entry of the permuted matrix lies in or
    /// above its diagonal block.
    fn assert_block_upper(n: usize, col_ptr: &[usize], row_idx: &[usize], f: &BtfForm) {
        let mut inv_row = vec![0usize; n];
        for (k, &r) in f.row_perm.iter().enumerate() {
            inv_row[r] = k;
        }
        let mut block_of = vec![0usize; n];
        for b in 0..f.block_ptr.len() - 1 {
            for k in f.block_ptr[b]..f.block_ptr[b + 1] {
                block_of[k] = b;
            }
        }
        for (kc, &jc) in f.col_perm.iter().enumerate() {
            for &r in &row_idx[col_ptr[jc]..col_ptr[jc + 1]] {
                let kr = inv_row[r];
                assert!(
                    block_of[kr] <= block_of[kc],
                    "entry ({kr}, {kc}) below its diagonal block"
                );
            }
        }
    }

    #[test]
    fn btf_lower_triangular_becomes_upper() {
        // Strictly lower triangular plus diagonal: n trivial blocks, and the
        // permutation must flip the fill to the upper triangle.
        let n = 5;
        let mut e = Vec::new();
        for i in 0..n {
            e.push((i, i));
            if i + 1 < n {
                e.push((i + 1, i));
            }
        }
        let (cp, ri) = pattern(n, &e);
        let f = block_triangular_form(n, &cp, &ri).unwrap();
        assert_eq!(f.block_ptr.len() - 1, n, "all blocks must be 1x1");
        assert_block_upper(n, &cp, &ri, &f);
    }

    #[test]
    fn btf_finds_irreducible_blocks() {
        // Two 2-cycles {0,1} and {2,3} coupled 'upward' by (0,2): two 2x2
        // blocks, and the {0,1} block must come first (it feeds nothing).
        let entries = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (0, 1),
            (1, 0),
            (2, 3),
            (3, 2),
            (0, 2),
        ];
        let (cp, ri) = pattern(4, &entries);
        let f = block_triangular_form(4, &cp, &ri).unwrap();
        assert_eq!(f.block_ptr, vec![0, 2, 4]);
        assert_block_upper(4, &cp, &ri, &f);
        let first: Vec<usize> = f.col_perm[0..2].to_vec();
        assert!(first.contains(&0) && first.contains(&1));
    }

    #[test]
    fn btf_fully_coupled_is_one_block() {
        // A single cycle through all columns: one irreducible block.
        let n = 4;
        let mut e = Vec::new();
        for i in 0..n {
            e.push((i, i));
            e.push(((i + 1) % n, i));
        }
        let (cp, ri) = pattern(n, &e);
        let f = block_triangular_form(n, &cp, &ri).unwrap();
        assert_eq!(f.block_ptr, vec![0, n]);
    }

    #[test]
    fn btf_structurally_singular_is_none() {
        let (cp, ri) = pattern(3, &[(0, 0), (1, 1), (2, 0)]);
        assert!(block_triangular_form(3, &cp, &ri).is_none());
    }

    #[test]
    fn btf_empty_matrix() {
        let f = block_triangular_form(0, &[0], &[]).unwrap();
        assert_eq!(f.block_ptr, vec![0]);
        assert!(f.row_perm.is_empty());
    }
}
