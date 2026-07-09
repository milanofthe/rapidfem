//! KaHIP Phase K6 - recursive nested-dissection driver.
//!
//! Walks connected components of the input graph; for each, recurses
//! with a contiguous numbering window. At each recursion level: if the
//! subgraph is small (≤ `amd_switch`), hand off to AMD; otherwise run
//! K5 to produce an edge bisection, K4 to lift it to a node separator,
//! number the separator last in the current window, and recurse on
//! the two sides.
//!
//! Mirrors `rslab-scotch::node_nd`, swapping SCOTCH's halo FM + direct
//! vertex separator for KaHIP's flow refinement (K3) + boundary-
//! bipartite separator (K4). The numbering convention is
//! `perm[i] = old vertex id at new position i`.

use rslab_metis::internals::graph::Graph;
use rslab_metis::internals::initial_partition::{PART_A, PART_B};
use rslab_metis::internals::rng::SplitMix;
use rslab_ordering_core::{CscPattern, OrderingError};

use crate::cycle::{graph_to_undirected, multilevel_bisection};
use crate::data_reduction::{expand_permutation, reduce_graph, ReduceOptions};
use crate::node_separator::flow_node_separator;
use crate::{KahipMode, KahipOptions, KahipStats};

const PART_SEP: u8 = 2;

/// Entry point for KaHIP nested dissection.
///
/// Pipeline: K1 data reduction → K2-K6 nested dissection on the
/// reduced graph → expand the reduced-graph permutation back to
/// original indices. `stats` accumulates crate-specific diagnostics
/// across the recursion.
pub(crate) fn kahip_nd_order(
    pattern: &CscPattern<'_>,
    opts: &KahipOptions,
    stats: &mut KahipStats,
) -> Result<Vec<i32>, OrderingError> {
    // K1: Ost-Schulz-Strash data reduction. We apply only Rule 1
    // (degree-1 cascading) - Rules 2-4 hurt fill empirically on our
    // corpus even with a corrected expansion; see `ReduceOptions`.
    // Rule 1 alone cleanly strips the leaf-heavy parts of arrow-
    // structured KKTs before we hand the core to multilevel
    // partitioning, and its elimination order (leaves before their
    // owners) matches what AMD would do for the leaves anyway.
    if let Some(reduced) = reduce_graph(pattern, 0.99, ReduceOptions::conservative())? {
        if reduced.n < pattern.n {
            stats.reduced_n = reduced.n;
            let reduced_pat = CscPattern::new(reduced.n, &reduced.col_ptr, &reduced.row_idx)
                .ok_or(OrderingError::MalformedInput)?;
            let reduced_perm = kahip_nd_inner(&reduced_pat, opts, stats)?;
            return expand_permutation(&reduced, &reduced_perm, pattern.n);
        }
    }

    stats.reduced_n = pattern.n;
    kahip_nd_inner(pattern, opts, stats)
}

fn kahip_nd_inner(
    pattern: &CscPattern<'_>,
    opts: &KahipOptions,
    stats: &mut KahipStats,
) -> Result<Vec<i32>, OrderingError> {
    let graph = Graph::from_csc_pattern(pattern)?;
    let n = graph.nvtxs as usize;

    let mut iperm: Vec<i32> = vec![-1; n];
    let mut rng = SplitMix::new(opts.seed);
    run_top(&graph, &mut iperm, opts, &mut rng, stats)?;
    invert_iperm(&iperm, n)
}

fn amd_switch_for(mode: KahipMode) -> usize {
    match mode {
        KahipMode::Fast => 200,
        KahipMode::Eco => 120,
        KahipMode::Strong => 80,
    }
}

fn run_top(
    graph: &Graph,
    iperm: &mut [i32],
    opts: &KahipOptions,
    rng: &mut SplitMix,
    stats: &mut KahipStats,
) -> Result<(), OrderingError> {
    let n = graph.nvtxs as usize;
    let (cc_label, ncc) = connected_components(graph);
    stats.n_components = ncc as u32;
    let mut offset: usize = 0;
    for c in 0..ncc {
        let (sub, vtx_map) = extract_by_label(graph, &cc_label, c as i32);
        let count = sub.nvtxs as usize;
        if count > 0 {
            recurse(&sub, &vtx_map, offset, iperm, opts, rng, stats)?;
        }
        offset += count;
    }
    debug_assert_eq!(offset, n);
    Ok(())
}

fn recurse(
    subgraph: &Graph,
    vtx_map: &[i32],
    offset: usize,
    iperm: &mut [i32],
    opts: &KahipOptions,
    rng: &mut SplitMix,
    stats: &mut KahipStats,
) -> Result<(), OrderingError> {
    let n = subgraph.nvtxs as usize;
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        iperm[vtx_map[0] as usize] = offset as i32;
        return Ok(());
    }

    let (cc_label, ncc) = connected_components(subgraph);
    if ncc > 1 {
        let mut off = offset;
        for c in 0..ncc {
            let (sub, map) = extract_by_label(subgraph, &cc_label, c as i32);
            let map_to_orig: Vec<i32> = map.iter().map(|&local| vtx_map[local as usize]).collect();
            let count = sub.nvtxs as usize;
            recurse(&sub, &map_to_orig, off, iperm, opts, rng, stats)?;
            off += count;
        }
        return Ok(());
    }

    let switch = amd_switch_for(opts.mode);
    if n <= switch {
        return amd_leaf(subgraph, vtx_map, offset, iperm);
    }

    // K5: edge bisection.
    let mut labels = multilevel_bisection(subgraph, opts, rng, stats);

    // K4: lift the edge bisection to a node separator.
    let ug = graph_to_undirected(subgraph);
    match flow_node_separator(&ug, &labels, None) {
        Some(sep) => {
            labels = sep.part;
        }
        None => {
            // No cross edges → degenerate. Fall through with A/B only,
            // which means both sides are independent and we can just
            // recurse directly on the connected components of A and B.
        }
    }

    let mut a_verts: Vec<i32> = Vec::new();
    let mut b_verts: Vec<i32> = Vec::new();
    let mut s_verts: Vec<i32> = Vec::new();
    for (v, &l) in labels.iter().enumerate() {
        match l {
            PART_A => a_verts.push(v as i32),
            PART_B => b_verts.push(v as i32),
            PART_SEP => s_verts.push(v as i32),
            _ => a_verts.push(v as i32),
        }
    }

    if a_verts.is_empty() || b_verts.is_empty() {
        return amd_leaf(subgraph, vtx_map, offset, iperm);
    }

    let na = a_verts.len();
    let nb = b_verts.len();
    let ns = s_verts.len();

    for (i, &v) in s_verts.iter().enumerate() {
        let orig = vtx_map[v as usize];
        iperm[orig as usize] = (offset + na + nb + i) as i32;
    }
    let _ = ns;

    let (sub_a, map_a_local) = extract_by_list(subgraph, &a_verts);
    let map_a: Vec<i32> = map_a_local
        .iter()
        .map(|&local| vtx_map[local as usize])
        .collect();
    let (sub_b, map_b_local) = extract_by_list(subgraph, &b_verts);
    let map_b: Vec<i32> = map_b_local
        .iter()
        .map(|&local| vtx_map[local as usize])
        .collect();

    recurse(&sub_a, &map_a, offset, iperm, opts, rng, stats)?;
    recurse(&sub_b, &map_b, offset + na, iperm, opts, rng, stats)?;
    Ok(())
}

fn amd_leaf(
    subgraph: &Graph,
    vtx_map: &[i32],
    offset: usize,
    iperm: &mut [i32],
) -> Result<(), OrderingError> {
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

fn graph_to_csc_pattern(graph: &Graph) -> (Vec<i32>, Vec<i32>) {
    let n = graph.nvtxs as usize;
    let mut col_ptr: Vec<i32> = Vec::with_capacity(n + 1);
    let mut row_idx: Vec<i32> = Vec::with_capacity(graph.adjncy.len() + n);
    col_ptr.push(0);
    for v in 0..n {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        let mut neighbors: Vec<i32> = graph.adjncy[lo..hi].to_vec();
        neighbors.sort_unstable();
        let mut diag_inserted = false;
        for u in neighbors {
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

fn invert_iperm(iperm: &[i32], n: usize) -> Result<Vec<i32>, OrderingError> {
    let mut perm: Vec<i32> = vec![-1; n];
    for (old, &new_pos) in iperm.iter().enumerate() {
        if new_pos < 0 || (new_pos as usize) >= n {
            return Err(OrderingError::Internal(
                "kahip nd produced invalid permutation",
            ));
        }
        let np = new_pos as usize;
        if perm[np] >= 0 {
            return Err(OrderingError::Internal(
                "kahip nd produced duplicate position",
            ));
        }
        perm[np] = old as i32;
    }
    Ok(perm)
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn diagonal_pattern_yields_permutation() {
        let cp = vec![0, 1, 2, 3];
        let ri = vec![0, 1, 2];
        let pat = CscPattern::new(3, &cp, &ri).unwrap();
        let opts = KahipOptions::default();
        let mut stats = KahipStats::default();
        let perm = kahip_nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 3);
        assert_permutation(&perm);
    }

    #[test]
    fn small_grid_uses_amd_leaf() {
        let t = grid_triples(10, 10);
        let (cp, ri) = csc_from_triples(100, &t);
        let pat = CscPattern::new(100, &cp, &ri).unwrap();
        let opts = KahipOptions::default(); // Fast: amd_switch=200, so 100 ≤ switch
        let mut stats = KahipStats::default();
        let perm = kahip_nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 100);
        assert_permutation(&perm);
    }

    #[test]
    fn large_grid_runs_multilevel() {
        let t = grid_triples(16, 16);
        let (cp, ri) = csc_from_triples(256, &t);
        let pat = CscPattern::new(256, &cp, &ri).unwrap();
        let opts = KahipOptions::default(); // Fast: amd_switch=200 < 256
        let mut stats = KahipStats::default();
        let perm = kahip_nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 256);
        assert_permutation(&perm);
        assert!(stats.cycles > 0, "expected at least one K5 pass on 16x16");
    }

    #[test]
    fn deterministic_across_runs() {
        let t = grid_triples(14, 14);
        let n = 196;
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let opts = KahipOptions::default();
        let mut s1 = KahipStats::default();
        let mut s2 = KahipStats::default();
        let p1 = kahip_nd_order(&pat, &opts, &mut s1).unwrap();
        let p2 = kahip_nd_order(&pat, &opts, &mut s2).unwrap();
        assert_eq!(p1, p2, "KaHIP ND must be deterministic for fixed seed");
    }

    #[test]
    fn handles_disconnected_components() {
        let mut t = grid_triples(8, 8);
        for &(i, j) in grid_triples(8, 8).iter() {
            t.push((i + 64, j + 64));
        }
        let (cp, ri) = csc_from_triples(128, &t);
        let pat = CscPattern::new(128, &cp, &ri).unwrap();
        let opts = KahipOptions::default();
        let mut stats = KahipStats::default();
        let perm = kahip_nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 128);
        assert_permutation(&perm);
        // Two disjoint 8×8 grids ⇒ exactly two top-level components.
        // Mirrors metis (`node_nd.rs` `nd_order_handles_disconnected_graph`)
        // and scotch, which assert the same on their stats (O18).
        assert_eq!(stats.n_components, 2);
    }

    #[test]
    fn fast_eco_strong_all_produce_valid_permutations() {
        let t = grid_triples(12, 12);
        let (cp, ri) = csc_from_triples(144, &t);
        let pat = CscPattern::new(144, &cp, &ri).unwrap();
        for mode in [KahipMode::Fast, KahipMode::Eco, KahipMode::Strong] {
            let opts = KahipOptions { seed: 3, mode };
            let mut stats = KahipStats::default();
            let perm = kahip_nd_order(&pat, &opts, &mut stats)
                .unwrap_or_else(|_| panic!("mode {:?} failed", mode));
            assert_eq!(perm.len(), 144);
            assert_permutation(&perm);
        }
    }

    #[test]
    fn empty_graph_returns_empty_permutation() {
        let cp = vec![0];
        let ri: Vec<i32> = Vec::new();
        let pat = CscPattern::new(0, &cp, &ri).unwrap();
        let opts = KahipOptions::default();
        let mut stats = KahipStats::default();
        let perm = kahip_nd_order(&pat, &opts, &mut stats).unwrap();
        assert!(perm.is_empty());
    }
}
