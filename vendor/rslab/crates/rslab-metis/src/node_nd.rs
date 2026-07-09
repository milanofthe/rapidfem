//! Recursive nested-dissection driver.
//!
//! Top-level algorithm (Karypis & Kumar 1998, §4, George 1973):
//!
//! 1. Split graph into connected components. Each component is
//!    ordered independently and its numbering is concatenated.
//! 2. On a single component:
//!    - If its size is below `nd_to_amd_switch`, hand off to AMD.
//!      (AMD dominates at small scales where ND overhead would
//!      exceed its quality benefit; METIS 5.2.0 switches at 200.)
//!    - Otherwise, run a multilevel node bisection, number the
//!      separator last, and recurse on the two sides.
//!
//! The numbering convention is "permutation" = new-position → old-id.
//! Internally we populate `iperm[original_vertex] = new_position` and
//! invert at the end.

use crate::coarsen::{coarsen, CoarsenCounters};
use crate::fm_refine::{refine_bisection, refine_separator};
use crate::graph::Graph;
use crate::initial_partition::{initial_bisect_bfs, initial_bisect_ggp, PART_A, PART_B};
use crate::rng::SplitMix;
use crate::separator::construct_separator;
use crate::{MetisOptions, MetisStats};
use rslab_ordering_core::{CscPattern, OrderingError};

/// Entry point. Produces a permutation `perm` where `perm[i]` is the
/// old vertex id placed at new position `i` (new-to-old).
pub(crate) fn nd_order(
    pattern: &CscPattern<'_>,
    opts: &MetisOptions,
    stats: &mut MetisStats,
) -> Result<Vec<i32>, OrderingError> {
    let graph = Graph::from_csc_pattern(pattern)?;
    let n = graph.nvtxs as usize;
    let mut iperm: Vec<i32> = vec![-1; n];
    let mut rng = SplitMix::new(opts.seed);

    // Iterative dissection over an explicit work stack of `(subgraph, vtx_map,
    // offset)` items - **not** native recursion, so the depth of the dissection
    // tree (O(log n) for good separators, but O(n) for graphs without them, e.g.
    // dense/random patterns) never overflows the call stack. Each item owns a
    // disjoint output range `[offset, offset + nvtxs)`, so siblings are order
    // independent.
    let mut work: Vec<(Graph, Vec<i32>, usize)> = Vec::new();
    let (cc_label, ncc) = connected_components(&graph);
    stats.n_components = ncc as u32;
    let mut offset: usize = 0;
    for c in 0..ncc {
        let (sub, vtx_map) = extract_by_label(&graph, &cc_label, c as i32);
        let count = sub.nvtxs as usize;
        if count > 0 {
            work.push((sub, vtx_map, offset));
        }
        offset += count;
    }

    while let Some((subgraph, vtx_map, offset)) = work.pop() {
        let n = subgraph.nvtxs as usize;
        if n == 0 {
            continue;
        }
        if n == 1 {
            iperm[vtx_map[0] as usize] = offset as i32;
            continue;
        }

        // Connected-component split (sub-problem may be disconnected).
        let (cc_label, ncc) = connected_components(&subgraph);
        if ncc > 1 {
            let mut off = offset;
            for c in 0..ncc {
                let (sub, map) = extract_by_label(&subgraph, &cc_label, c as i32);
                let map_to_orig: Vec<i32> =
                    map.iter().map(|&local| vtx_map[local as usize]).collect();
                let count = sub.nvtxs as usize;
                if count > 0 {
                    work.push((sub, map_to_orig, off));
                }
                off += count;
            }
            continue;
        }

        // AMD leaf.
        if n <= opts.nd_to_amd_switch as usize {
            amd_leaf(&subgraph, &vtx_map, offset, &mut iperm, stats)?;
            continue;
        }

        // Multilevel node bisection.
        let labels = multilevel_node_bisection(&subgraph, opts, &mut rng, stats);
        let mut a_verts: Vec<i32> = Vec::new();
        let mut b_verts: Vec<i32> = Vec::new();
        let mut s_verts: Vec<i32> = Vec::new();
        for (v, &l) in labels.iter().enumerate() {
            match l {
                PART_A => a_verts.push(v as i32),
                PART_B => b_verts.push(v as i32),
                _ => s_verts.push(v as i32),
            }
        }

        // Safety: a degenerate split - one side empty, or so unbalanced that the
        // larger side is >=90% of the subgraph - dissects ~O(n) deep for no gain
        // (each level barely shrinks) on graphs without good separators. Order the
        // whole subgraph with the (non-recursive) AMD leaf instead; this also keeps
        // the work-stack shallow.
        let big = a_verts.len().max(b_verts.len());
        if a_verts.is_empty() || b_verts.is_empty() || big as f64 >= 0.9 * n as f64 {
            amd_leaf(&subgraph, &vtx_map, offset, &mut iperm, stats)?;
            continue;
        }

        stats.n_separator_vertices += s_verts.len() as u32;
        let na = a_verts.len();
        let nb = b_verts.len();

        // Number separator last: positions [offset + na + nb, offset + n).
        for (i, &v) in s_verts.iter().enumerate() {
            let orig = vtx_map[v as usize];
            iperm[orig as usize] = (offset + na + nb + i) as i32;
        }

        let (sub_a, map_a_local) = extract_by_list(&subgraph, &a_verts);
        let map_a: Vec<i32> = map_a_local
            .iter()
            .map(|&local| vtx_map[local as usize])
            .collect();
        let (sub_b, map_b_local) = extract_by_list(&subgraph, &b_verts);
        let map_b: Vec<i32> = map_b_local
            .iter()
            .map(|&local| vtx_map[local as usize])
            .collect();
        work.push((sub_a, map_a, offset));
        work.push((sub_b, map_b, offset + na));
    }

    invert_iperm(&iperm, n)
}

/// Multilevel node bisection: coarsen → initial bisection with niparts
/// trials → uncoarsen with FM → convert edge bisection to node
/// separator → refine separator. Returns labels in {PART_A, PART_B,
/// PART_SEP}.
fn multilevel_node_bisection(
    graph: &Graph,
    opts: &MetisOptions,
    rng: &mut SplitMix,
    stats: &mut MetisStats,
) -> Vec<u8> {
    let mut counters = CoarsenCounters::default();
    let levels = coarsen(graph, opts, rng, &mut counters);
    stats.n_two_hop_fallbacks += counters.n_two_hop_fallbacks;
    stats.n_levels += levels.len() as u32;

    // Coarsest graph for initial bisection.
    let coarsest: &Graph = match levels.last() {
        Some(cg) => &cg.graph,
        None => graph,
    };
    let total: i64 = coarsest.vwgt.iter().map(|&w| w as i64).sum();
    let target = total / 2;

    // niparts trials; keep best post-FM cut.
    let mut best_labels: Vec<u8> = vec![PART_A; coarsest.nvtxs as usize];
    let mut best_cut: i32 = i32::MAX;
    for trial in 0..opts.niparts {
        let mut trial_labels = if trial % 2 == 0 {
            initial_bisect_ggp(coarsest, rng, target)
        } else {
            initial_bisect_bfs(coarsest, rng, target)
        };
        let cut = refine_bisection(
            coarsest,
            &mut trial_labels,
            opts.max_imbalance,
            opts.fm_passes,
        );
        stats.n_fm_passes += opts.fm_passes;
        if cut < best_cut {
            best_cut = cut;
            best_labels = trial_labels;
        }
    }
    let mut labels = best_labels;

    // Uncoarsen: walk levels in reverse. `cmap` at level i maps
    // previous-graph vertices to level-i graph vertices.
    for level_idx in (0..levels.len()).rev() {
        let cg = &levels[level_idx];
        let prev_graph: &Graph = if level_idx == 0 {
            graph
        } else {
            &levels[level_idx - 1].graph
        };
        let prev_n = prev_graph.nvtxs as usize;
        let mut proj: Vec<u8> = vec![PART_A; prev_n];
        for (v, p) in proj.iter_mut().enumerate().take(prev_n) {
            let c = cg.cmap[v] as usize;
            *p = labels[c];
        }
        labels = proj;
        refine_bisection(prev_graph, &mut labels, opts.max_imbalance, opts.fm_passes);
        stats.n_fm_passes += opts.fm_passes;
    }

    // Convert edge bisection to node separator.
    construct_separator(graph, &mut labels);
    // Refine separator (greedy).
    refine_separator(graph, &mut labels, opts.max_imbalance, opts.fm_passes);
    stats.n_fm_passes += opts.fm_passes;

    labels
}

/// Hand off to AMD on an uncoarsened subgraph. Writes new positions
/// `[offset, offset + n)` into `iperm`.
fn amd_leaf(
    subgraph: &Graph,
    vtx_map: &[i32],
    offset: usize,
    iperm: &mut [i32],
    stats: &mut MetisStats,
) -> Result<(), OrderingError> {
    stats.n_amd_leaf_calls += 1;
    let n = subgraph.nvtxs as usize;
    if n == 0 {
        return Ok(());
    }
    let (col_ptr, row_idx) = graph_to_csc_pattern(subgraph);
    let pattern = CscPattern::new(n, &col_ptr, &row_idx).ok_or(OrderingError::MalformedInput)?;
    let perm_local = rslab_amd::amd_order(&pattern)?;
    debug_assert_eq!(perm_local.len(), n);
    for (new_pos, &local_id) in perm_local.iter().enumerate() {
        let orig = vtx_map[local_id as usize];
        iperm[orig as usize] = (offset + new_pos) as i32;
    }
    Ok(())
}

/// Build a full-symmetric CSC pattern from an internal `Graph`.
/// Adjacency in `Graph` is already full-symmetric with diagonal
/// dropped; we reinsert the diagonal for downstream consumers that
/// expect it. Row indices within each column remain sorted.
fn graph_to_csc_pattern(graph: &Graph) -> (Vec<i32>, Vec<i32>) {
    let n = graph.nvtxs as usize;
    let mut col_ptr: Vec<i32> = Vec::with_capacity(n + 1);
    let mut row_idx: Vec<i32> = Vec::with_capacity(graph.adjncy.len() + n);
    col_ptr.push(0);
    for v in 0..n {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        // Insert neighbors keeping sorted order, splicing in the
        // diagonal at its correct position.
        let mut diag_inserted = false;
        for k in lo..hi {
            let u = graph.adjncy[k];
            if !diag_inserted && u as usize > v {
                row_idx.push(v as i32);
                diag_inserted = true;
            }
            row_idx.push(u);
        }
        if !diag_inserted {
            row_idx.push(v as i32);
        }
        col_ptr.push(row_idx.len() as i32);
    }
    (col_ptr, row_idx)
}

/// Invert the new→old position map `iperm` (where `iperm[old] = new_pos`)
/// into the old→new permutation `perm` (where `perm[new_pos] = old`).
///
/// Rejects an out-of-range or duplicated target position rather than
/// silently emitting a non-bijection - parity with the scotch/kahip
/// `invert_iperm` helpers (O20).
fn invert_iperm(iperm: &[i32], n: usize) -> Result<Vec<i32>, OrderingError> {
    let mut perm: Vec<i32> = vec![-1; n];
    for (old, &new_pos) in iperm.iter().enumerate() {
        if new_pos < 0 || (new_pos as usize) >= n {
            return Err(OrderingError::Internal(
                "metis nd produced invalid permutation",
            ));
        }
        let np = new_pos as usize;
        if perm[np] >= 0 {
            return Err(OrderingError::Internal(
                "metis nd produced duplicate position",
            ));
        }
        perm[np] = old as i32;
    }
    Ok(perm)
}

/// Connected-component labeling via BFS. Returns `(cc_label, ncc)`
/// where `cc_label[v] ∈ 0..ncc`.
fn connected_components(graph: &Graph) -> (Vec<i32>, usize) {
    let n = graph.nvtxs as usize;
    let mut cc: Vec<i32> = vec![-1; n];
    let mut ncc: i32 = 0;
    let mut queue: Vec<i32> = Vec::new();
    for start in 0..n {
        if cc[start] >= 0 {
            continue;
        }
        cc[start] = ncc;
        queue.clear();
        queue.push(start as i32);
        while let Some(v) = queue.pop() {
            let vu = v as usize;
            let lo = graph.xadj[vu] as usize;
            let hi = graph.xadj[vu + 1] as usize;
            for k in lo..hi {
                let u = graph.adjncy[k];
                if cc[u as usize] < 0 {
                    cc[u as usize] = ncc;
                    queue.push(u);
                }
            }
        }
        ncc += 1;
    }
    (cc, ncc as usize)
}

/// Extract the induced subgraph on the vertex set `{v : label[v] == c}`.
/// Returns the subgraph and the mapping `local_id → original_id`.
fn extract_by_label(graph: &Graph, label: &[i32], c: i32) -> (Graph, Vec<i32>) {
    let n = graph.nvtxs as usize;
    let mut vtx_map: Vec<i32> = Vec::new();
    let mut local_id: Vec<i32> = vec![-1; n];
    for v in 0..n {
        if label[v] == c {
            local_id[v] = vtx_map.len() as i32;
            vtx_map.push(v as i32);
        }
    }
    (build_induced(graph, &vtx_map, &local_id), vtx_map)
}

/// Extract the induced subgraph on the given list of vertex ids.
fn extract_by_list(graph: &Graph, verts: &[i32]) -> (Graph, Vec<i32>) {
    let n = graph.nvtxs as usize;
    let mut local_id: Vec<i32> = vec![-1; n];
    for (i, &v) in verts.iter().enumerate() {
        local_id[v as usize] = i as i32;
    }
    let vtx_map = verts.to_vec();
    (build_induced(graph, &vtx_map, &local_id), vtx_map)
}

fn build_induced(graph: &Graph, vtx_map: &[i32], local_id: &[i32]) -> Graph {
    let sub_n = vtx_map.len();
    let mut xadj: Vec<i32> = Vec::with_capacity(sub_n + 1);
    let mut adjncy: Vec<i32> = Vec::new();
    let mut adjwgt: Vec<i32> = Vec::new();
    let mut vwgt: Vec<i32> = Vec::with_capacity(sub_n);
    xadj.push(0);
    for &orig in vtx_map {
        let v = orig as usize;
        vwgt.push(graph.vwgt[v]);
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            let lu = local_id[u];
            if lu >= 0 {
                adjncy.push(lu);
                adjwgt.push(graph.adjwgt[k]);
            }
        }
        xadj.push(adjncy.len() as i32);
    }
    Graph {
        nvtxs: sub_n as i32,
        xadj,
        adjncy,
        vwgt,
        adjwgt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rslab_ordering_core::CscPattern;
    use std::collections::BTreeSet;

    fn csc_from_triples(n: usize, triples: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
        let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &(i, j) in triples {
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
        let mut col_ptr: Vec<i32> = vec![0];
        let mut row_idx: Vec<i32> = Vec::new();
        for col in &cols {
            for &r in col {
                row_idx.push(r);
            }
            col_ptr.push(row_idx.len() as i32);
        }
        (col_ptr, row_idx)
    }

    fn grid_triples(m: usize, n: usize) -> Vec<(usize, usize)> {
        let idx = |r: usize, c: usize| r * n + c;
        let mut t = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let k = idx(r, c);
                t.push((k, k));
                if r + 1 < m {
                    t.push((k, idx(r + 1, c)));
                }
                if c + 1 < n {
                    t.push((k, idx(r, c + 1)));
                }
            }
        }
        t
    }

    fn assert_permutation(perm: &[i32]) {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &p in perm {
            let p = p as usize;
            assert!(p < n, "index {} out of bounds", p);
            assert!(!seen[p], "duplicate {}", p);
            seen[p] = true;
        }
    }

    #[test]
    fn invert_iperm_rejects_duplicate_positions() {
        // A valid new→old inversion of a bijection.
        // iperm[old] = new_pos: old0→2, old1→0, old2→1 ⇒ perm = [1, 2, 0].
        assert_eq!(invert_iperm(&[2, 0, 1], 3).unwrap(), vec![1, 2, 0]);
        // Two olds claiming the same position must be rejected, not
        // silently overwritten into a non-bijection (parity with the
        // scotch/kahip duplicate-position check; O20).
        assert!(matches!(
            invert_iperm(&[0, 0, 2], 3),
            Err(OrderingError::Internal(_))
        ));
        // Out-of-range target position is rejected.
        assert!(matches!(
            invert_iperm(&[3, 0, 1], 3),
            Err(OrderingError::Internal(_))
        ));
    }

    #[test]
    fn cc_disconnected_blocks() {
        // Two disconnected 3x3 grids.
        let mut t = grid_triples(3, 3);
        // second block at ids 9..18
        for &(i, j) in grid_triples(3, 3).iter() {
            t.push((i + 9, j + 9));
        }
        let (cp, ri) = csc_from_triples(18, &t);
        let pat = CscPattern::new(18, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let (_, ncc) = connected_components(&g);
        assert_eq!(ncc, 2);
    }

    #[test]
    fn nd_order_small_grid_is_permutation() {
        // 10x10 grid, 100 vertices. With defaults (nd_to_amd_switch=200)
        // this falls into the AMD leaf branch at the top level.
        let t = grid_triples(10, 10);
        let (cp, ri) = csc_from_triples(100, &t);
        let pat = CscPattern::new(100, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let perm = nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 100);
        assert_permutation(&perm);
        assert!(stats.n_amd_leaf_calls >= 1);
    }

    #[test]
    fn nd_order_large_grid_uses_multilevel() {
        // 20x20 grid, 400 vertices > 200 (nd_to_amd_switch) so the top
        // level runs a real multilevel bisection.
        let t = grid_triples(20, 20);
        let (cp, ri) = csc_from_triples(400, &t);
        let pat = CscPattern::new(400, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let perm = nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 400);
        assert_permutation(&perm);
        assert!(
            stats.n_separator_vertices > 0,
            "expected a top-level separator"
        );
    }

    #[test]
    fn nd_order_deterministic() {
        let t = grid_triples(12, 12);
        let n = 144;
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut s1 = MetisStats::default();
        let mut s2 = MetisStats::default();
        let p1 = nd_order(&pat, &opts, &mut s1).unwrap();
        let p2 = nd_order(&pat, &opts, &mut s2).unwrap();
        assert_eq!(p1, p2);
    }

    #[test]
    fn nd_order_handles_disconnected_graph() {
        // Two separate 6x6 grids.
        let mut t = grid_triples(6, 6);
        for &(i, j) in grid_triples(6, 6).iter() {
            t.push((i + 36, j + 36));
        }
        let (cp, ri) = csc_from_triples(72, &t);
        let pat = CscPattern::new(72, &cp, &ri).unwrap();
        let opts = MetisOptions::default();
        let mut stats = MetisStats::default();
        let perm = nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 72);
        assert_permutation(&perm);
        assert_eq!(stats.n_components, 2);
    }

    #[test]
    fn extract_induced_preserves_edges() {
        // 3x3 grid, extract top row {0,1,2}.
        let t = grid_triples(3, 3);
        let (cp, ri) = csc_from_triples(9, &t);
        let pat = CscPattern::new(9, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let (sub, map) = extract_by_list(&g, &[0, 1, 2]);
        assert_eq!(sub.nvtxs, 3);
        assert_eq!(map, vec![0, 1, 2]);
        // Top row: 0-1, 1-2 → 2 edges, each stored twice.
        assert_eq!(sub.adjncy.len(), 4);
    }
}
