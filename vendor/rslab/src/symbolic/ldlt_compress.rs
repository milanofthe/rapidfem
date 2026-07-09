//! LDLᵀ-aware ordering preprocessing: Duff-Pralet symmetric matching
//! plus quotient-graph compression. Port of MUMPS `ICNTL(12) = 2`
//! (`DMUMPS_LDLT_COMPRESS`, `DMUMPS_EXPAND_PERMUTATION`).
//! See `dev/research/phase-2.6.5-ldlt-aware-ordering.md` for the
//! algorithm and `dev/plans/phase-2.6.5-ldlt-compressed-graph.md` for
//! the plan.
//!
//! The compression walks the MC64 matching permutation's cycle
//! structure, contracts each 2-cycle into one super-variable, and
//! decomposes k-cycles (k ≥ 3) into ⌊k/2⌋ pairs + an odd singleton.
//! The compressed graph is fed to AMD/METIS/SCOTCH; the returned
//! super-permutation is then expanded so paired originals sit
//! adjacent in the output permutation (so the downstream BK kernel
//! sees each tentative 2×2 pair in consecutive columns).

use crate::sparse::csc::CscPattern;

/// Super-variable map produced from a symmetric MC64 matching.
///
/// `icmp[i]` is the super-variable id of original variable `i`. Super
/// ids are contiguous `[0, ncmp)` where `ncmp = pairs.len() +
/// singletons.len()`. Pair super-ids come first: super `s < pairs.len()`
/// means `map.pairs[s]`; super `s >= pairs.len()` means
/// `map.singletons[s - pairs.len()]`.
#[derive(Debug, Clone)]
pub struct SuperMap {
    pub icmp: Vec<usize>,
    pub pairs: Vec<(usize, usize)>,
    pub singletons: Vec<usize>,
}

impl SuperMap {
    /// Number of super-variables (dimension of the compressed graph).
    pub fn ncmp(&self) -> usize {
        self.pairs.len() + self.singletons.len()
    }
}

/// Walk the MC64 matching permutation and emit pairs / singletons per
/// the Duff-Pralet rule (matches MUMPS `DMUMPS_SYM_MWM`):
///
/// - `perm[j] == j`     → singleton
/// - `perm[j] != j` and `perm[perm[j]] == j` → pair `(j, perm[j])`
///   (canonicalised as `(min, max)` and emitted once)
/// - length-k cycle (k ≥ 3) `j0 → j1 → ... → j_{k-1} → j0`:
///   pairs `(j0, j1), (j2, j3), ...` and `j_{k-1}` as singleton if
///   k is odd.
/// - `perm[j] == usize::MAX` (unmatched) → singleton
///
/// Emission order follows discovery (ascending `j`), deterministic.
pub fn build_supermap(perm: &[usize]) -> SuperMap {
    let n = perm.len();
    let mut icmp = vec![usize::MAX; n];
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut singletons: Vec<usize> = Vec::new();
    let mut visited = vec![false; n];

    for start in 0..n {
        if visited[start] {
            continue;
        }
        if perm[start] == usize::MAX {
            visited[start] = true;
            singletons.push(start);
            continue;
        }
        // Walk the cycle starting at `start`, collecting nodes in
        // visit order.
        let mut cycle: Vec<usize> = Vec::new();
        let mut j = start;
        loop {
            if visited[j] {
                break;
            }
            visited[j] = true;
            cycle.push(j);
            let next = perm[j];
            if next == usize::MAX {
                break;
            }
            if next == start {
                break;
            }
            j = next;
        }
        // Decompose according to cycle length.
        match cycle.len() {
            1 => singletons.push(cycle[0]),
            2 => {
                let (a, b) = (cycle[0], cycle[1]);
                let pair = if a < b { (a, b) } else { (b, a) };
                pairs.push(pair);
            }
            _ => {
                // k ≥ 3: pair consecutive, odd leftover singleton.
                let mut i = 0;
                while i + 1 < cycle.len() {
                    let (a, b) = (cycle[i], cycle[i + 1]);
                    let pair = if a < b { (a, b) } else { (b, a) };
                    pairs.push(pair);
                    i += 2;
                }
                if cycle.len() % 2 == 1 {
                    singletons.push(cycle[cycle.len() - 1]);
                }
            }
        }
    }

    // Stamp `icmp`: pair super-ids come first.
    for (sid, &(a, b)) in pairs.iter().enumerate() {
        icmp[a] = sid;
        icmp[b] = sid;
    }
    let pair_count = pairs.len();
    for (k, &s) in singletons.iter().enumerate() {
        icmp[s] = pair_count + k;
    }

    SuperMap {
        icmp,
        pairs,
        singletons,
    }
}

/// Contract a full symmetric `CscPattern` onto the super-variable map.
///
/// For every stored edge `(i, j)` (with `i != j`), emit the contracted
/// edge `(icmp[i], icmp[j])` in the output pattern. Self-loops
/// (`icmp[i] == icmp[j]`) are dropped, and duplicates produced by two
/// originals inside the same super-variable or by already-present
/// parallel contracted edges are merged. Output is full-symmetric,
/// row-sorted, with strictly increasing rows per column.
///
/// Uses a per-column `Vec<u32>` mark array (tag-reset trick, same
/// pattern as AMD) to deduplicate in O(nnz · avg_super_deg) without
/// an explicit sort.
pub fn compress_pattern(pat: &CscPattern, map: &SuperMap) -> CscPattern {
    let ncmp = map.ncmp();
    if ncmp == 0 {
        return CscPattern {
            n: 0,
            col_ptr: vec![0],
            row_idx: Vec::new(),
        };
    }

    // Per-super-column accumulator: mark[r] == current_tag means row
    // super-id `r` is already present in the current super-column.
    // Using u32 tags avoids per-column zeroing of the whole array.
    let mut mark: Vec<u32> = vec![0; ncmp];
    let mut tag: u32 = 0;

    let mut col_ptr: Vec<usize> = Vec::with_capacity(ncmp + 1);
    col_ptr.push(0);
    let mut row_idx: Vec<usize> = Vec::new();

    // Inverse index: for each super-column `sc`, the list of originals
    // that map to it. Build once from `map.icmp`.
    let mut super_cols: Vec<Vec<usize>> = vec![Vec::new(); ncmp];
    for (orig, &sid) in map.icmp.iter().enumerate() {
        if sid < ncmp {
            super_cols[sid].push(orig);
        }
    }

    let mut col_buf: Vec<usize> = Vec::new();
    for (sc, originals) in super_cols.iter().enumerate() {
        tag = tag.wrapping_add(1);
        if tag == 0 {
            // Tag wrapped; reset all marks and start again.
            mark.iter_mut().for_each(|m| *m = 0);
            tag = 1;
        }
        col_buf.clear();

        for &orig_c in originals {
            // Walk the originals' column in `pat` and contract rows.
            let start = pat.col_ptr[orig_c];
            let end = pat.col_ptr[orig_c + 1];
            for k in start..end {
                let orig_r = pat.row_idx[k];
                let sr = map.icmp[orig_r];
                if sr == sc {
                    continue;
                }
                if mark[sr] != tag {
                    mark[sr] = tag;
                    col_buf.push(sr);
                }
            }
        }

        col_buf.sort_unstable();
        row_idx.extend_from_slice(&col_buf);
        col_ptr.push(row_idx.len());
    }

    CscPattern {
        n: ncmp,
        col_ptr,
        row_idx,
    }
}

/// Expand a super-permutation of length `ncmp` back to a length-`n`
/// permutation of `0..n`. Pair super-ids emit their two originals
/// adjacent in the order `(pair.0, pair.1)`; singleton super-ids emit
/// the single original.
pub fn expand_permutation(super_perm: &[usize], map: &SuperMap) -> Vec<usize> {
    let n = map.icmp.len();
    let mut out: Vec<usize> = Vec::with_capacity(n);
    let pair_count = map.pairs.len();
    for &s in super_perm {
        if s < pair_count {
            let (a, b) = map.pairs[s];
            out.push(a);
            out.push(b);
        } else {
            out.push(map.singletons[s - pair_count]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supermap_all_singletons_identity_perm() {
        // perm[j] = j for all j → all 1-cycles.
        let perm: Vec<usize> = (0..5).collect();
        let map = build_supermap(&perm);
        assert!(map.pairs.is_empty());
        assert_eq!(map.singletons, vec![0, 1, 2, 3, 4]);
        assert_eq!(map.icmp, vec![0, 1, 2, 3, 4]);
        assert_eq!(map.ncmp(), 5);
    }

    #[test]
    fn supermap_two_2cycles() {
        // 0↔2 and 1↔3. perm[0]=2, perm[2]=0, perm[1]=3, perm[3]=1.
        let perm = vec![2, 3, 0, 1];
        let map = build_supermap(&perm);
        assert_eq!(map.pairs, vec![(0, 2), (1, 3)]);
        assert!(map.singletons.is_empty());
        assert_eq!(map.icmp, vec![0, 1, 0, 1]);
        assert_eq!(map.ncmp(), 2);
    }

    #[test]
    fn supermap_three_cycle_plus_singletons() {
        // 0→1→2→0, plus 3,4,5 fixed.
        let perm = vec![1, 2, 0, 3, 4, 5];
        let map = build_supermap(&perm);
        // Cycle 0→1→2 discovered at start=0, cycle = [0,1,2].
        // Decomposes into pair (0,1) and singleton 2.
        assert_eq!(map.pairs, vec![(0, 1)]);
        assert_eq!(map.singletons, vec![2, 3, 4, 5]);
        // icmp: 0→0, 1→0, 2→1 (first singleton, sid=pair_count+0=1),
        //       3→2, 4→3, 5→4.
        assert_eq!(map.icmp, vec![0, 0, 1, 2, 3, 4]);
        assert_eq!(map.ncmp(), 5);
    }

    #[test]
    fn supermap_unmatched_is_singleton() {
        // perm[2] unmatched; 0↔1 paired.
        let perm = vec![1, 0, usize::MAX];
        let map = build_supermap(&perm);
        assert_eq!(map.pairs, vec![(0, 1)]);
        assert_eq!(map.singletons, vec![2]);
        assert_eq!(map.ncmp(), 2);
    }

    #[test]
    fn expand_is_identity_when_super_perm_is_iota() {
        // After building the supermap, expanding `super_perm = 0..ncmp`
        // reproduces a permutation of 0..n consistent with the
        // pair-then-singleton layout.
        let perm = vec![2, 3, 0, 1];
        let map = build_supermap(&perm);
        let super_iota: Vec<usize> = (0..map.ncmp()).collect();
        let expanded = expand_permutation(&super_iota, &map);
        // Pairs emit (0,2),(1,3) → [0,2,1,3]. Length n=4.
        assert_eq!(expanded, vec![0, 2, 1, 3]);
        let mut sorted = expanded.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn expand_is_bijection_with_3cycle() {
        let perm = vec![1, 2, 0, 3, 4, 5];
        let map = build_supermap(&perm);
        let super_iota: Vec<usize> = (0..map.ncmp()).collect();
        let expanded = expand_permutation(&super_iota, &map);
        assert_eq!(expanded.len(), 6);
        let mut sorted = expanded.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }

    fn build_full_pattern(n: usize, edges: &[(usize, usize)]) -> CscPattern {
        // Build a full-symmetric `CscPattern` from the given undirected
        // edges (also inserts the diagonal). Used by the contraction
        // tests below.
        let mut cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (j, col) in cols.iter_mut().enumerate() {
            col.push(j);
        }
        for &(i, j) in edges {
            cols[i].push(j);
            cols[j].push(i);
        }
        let mut col_ptr = Vec::with_capacity(n + 1);
        let mut row_idx = Vec::new();
        col_ptr.push(0);
        for col in &mut cols {
            col.sort_unstable();
            col.dedup();
            row_idx.extend_from_slice(col);
            col_ptr.push(row_idx.len());
        }
        CscPattern {
            n,
            col_ptr,
            row_idx,
        }
    }

    #[test]
    fn compress_contracts_paired_columns_and_drops_selfloops() {
        // 4 variables, edges 0-2, 1-3, 0-1. Pair {0,2} → super 0;
        // pair {1,3} → super 1. After contraction:
        //   - edge 0-2 becomes self-loop on super 0 → dropped.
        //   - edge 1-3 becomes self-loop on super 1 → dropped.
        //   - edge 0-1 becomes edge super-0 ↔ super-1.
        let pat = build_full_pattern(4, &[(0, 2), (1, 3), (0, 1)]);
        let map = build_supermap(&[2, 3, 0, 1]);
        let cpat = compress_pattern(&pat, &map);
        assert_eq!(cpat.n, 2);
        // Super-column 0: rows = [1]. Super-column 1: rows = [0].
        // Diagonals in the input become self-loops and are dropped.
        assert_eq!(cpat.col_ptr, vec![0, 1, 2]);
        assert_eq!(cpat.row_idx, vec![1, 0]);
    }

    #[test]
    fn compress_dedups_parallel_edges() {
        // 4 variables, pair {0,2}, pair {1,3}. Edges 0-1 and 2-3 both
        // contract to super-0 ↔ super-1 - the accumulator must emit
        // the edge once per direction.
        let pat = build_full_pattern(4, &[(0, 1), (2, 3)]);
        let map = build_supermap(&[2, 3, 0, 1]);
        let cpat = compress_pattern(&pat, &map);
        assert_eq!(cpat.n, 2);
        assert_eq!(cpat.col_ptr, vec![0, 1, 2]);
        assert_eq!(cpat.row_idx, vec![1, 0]);
    }

    #[test]
    fn compress_preserves_symmetry() {
        // Random-ish small pattern with mixed pairs and singletons.
        // Verify the compressed pattern is full-symmetric:
        // row `r` in col `c` iff row `c` in col `r`.
        let pat = build_full_pattern(6, &[(0, 1), (2, 3), (0, 4), (2, 5), (1, 5)]);
        // perm: 0↔1 pair, 2↔3 pair, 4,5 singletons.
        let map = build_supermap(&[1, 0, 3, 2, 4, 5]);
        let cpat = compress_pattern(&pat, &map);
        // Build adjacency set and verify symmetry.
        let mut edges = std::collections::HashSet::new();
        for c in 0..cpat.n {
            for k in cpat.col_ptr[c]..cpat.col_ptr[c + 1] {
                edges.insert((cpat.row_idx[k], c));
            }
        }
        for &(r, c) in &edges {
            assert!(
                edges.contains(&(c, r)),
                "edge ({}, {}) present but ({}, {}) not - asymmetry",
                r,
                c,
                c,
                r
            );
        }
        // Rows within each column are sorted and unique.
        for c in 0..cpat.n {
            let s = cpat.col_ptr[c];
            let e = cpat.col_ptr[c + 1];
            let col = &cpat.row_idx[s..e];
            for w in col.windows(2) {
                assert!(w[0] < w[1], "col {} not strictly sorted: {:?}", c, col);
            }
        }
    }
}
