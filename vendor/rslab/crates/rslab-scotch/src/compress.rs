//! Graph compression by indistinguishable-vertex merging.
//!
//! S1 of `dev/plans/ordering-scotch.md`. Algorithm and invariants are
//! covered in `dev/research/scotch-graph-compression.md`. The short
//! version: two vertices `u`, `v` are *indistinguishable* (a
//! "supervariable") when their closed neighborhoods coincide,
//!
//!   `N[u] = N(u) ∪ {u} = N(v) ∪ {v} = N[v]`.
//!
//! In a structurally symmetric graph this implies `(u, v) ∈ E` plus
//! `N(u) \ {v} = N(v) \ {u}`, which is exactly the case that arises
//! from repeated rows/columns in structured KKT and FE-mesh patterns.
//!
//! We **do not** detect the disjoint case `(u, v) ∉ E ∧ N(u) = N(v)`;
//! it is rare in the SCOTCH target workloads, and conflating it
//! with the closed-neighborhood test would require either two hash
//! passes or a per-pair fix-up at compare time. The research note
//! documents this as a deliberate, conservative S1 decision.
//!
//! Hashing uses [`std::collections::hash_map::DefaultHasher`]
//! (SipHash-1-3, deterministic). Cryptographic strength is not
//! required because we exact-verify every bucket.
//!
//! ## Invariants enforced by tests
//!
//! - Vertex-weight conservation: `Σ vwgt_c = Σ vwgt`.
//! - Edge-weight conservation: external edges sum to
//!   `Σ adjwgt_c` (intra-class edges become self-loops and drop).
//! - Bijection of expansion: `vertex_map` flattened in order is a
//!   permutation of `0..n`.
//! - Symmetry preserved: `(c, c')` in compressed adjacency iff
//!   `(c', c)` is.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::graph::Graph;

/// Result of a successful graph compression.
///
/// `vertex_map[c]` lists the original-vertex indices that merged into
/// supervariable `c`, in ascending original-index order. The lowest
/// original index in each class is the class representative; classes
/// are emitted in increasing-representative order, so iterating
/// `vertex_map` and flattening produces a deterministic permutation
/// of `0..n` (the "expansion").
pub(crate) struct CompressedGraph {
    /// The compressed graph (with summed vertex and edge weights).
    pub graph: Graph,
    /// `vertex_map[c]` = original-vertex indices in supervariable `c`.
    pub vertex_map: Vec<Vec<i32>>,
    /// `n_compressed / n_original`. Smaller = more aggressive compression.
    pub ratio: f64,
}

/// Attempt to compress `g` by merging indistinguishable vertices.
///
/// Returns `Some(compressed)` only if the resulting vertex-count
/// ratio `n_compressed / n_original` is **strictly less than**
/// `max_ratio`. With the default `max_ratio = 0.7` (per
/// [`crate::ScotchOptions::compress_ratio`]), compression is accepted
/// when it removes at least 30 % of the vertices.
///
/// Returns `None` when:
/// - the graph is empty (`nvtxs == 0`), or
/// - no merges are possible (compression ratio is 1.0), or
/// - merges exist but the saved fraction is below the threshold.
///
/// Complexity: `O(|E|)` for hashing + bucket assembly, plus
/// `O(|E_compressed| log d_max)` for the per-class adjacency dedup.
pub(crate) fn compress_graph(g: &Graph, max_ratio: f64) -> Option<CompressedGraph> {
    let n = g.nvtxs as usize;
    if n == 0 {
        return None;
    }

    // -- Step 1: build closed neighborhoods + per-vertex hashes.
    let mut closed: Vec<Vec<i32>> = Vec::with_capacity(n);
    let mut hashes: Vec<u64> = Vec::with_capacity(n);
    for v in 0..n as i32 {
        let cn = closed_nbhd_sorted(g, v);
        let mut h = DefaultHasher::new();
        cn.hash(&mut h);
        g.vwgt[v as usize].hash(&mut h);
        hashes.push(h.finish());
        closed.push(cn);
    }

    // -- Step 2: hash buckets.
    let mut buckets: HashMap<u64, Vec<i32>> = HashMap::new();
    for v in 0..n as i32 {
        buckets.entry(hashes[v as usize]).or_default().push(v);
    }

    // -- Step 3: equivalence classes via exact compare within buckets.
    // Process vertices in original order so the first-seen vertex in a
    // class is its representative; class indices come out in
    // representative-ascending order (deterministic).
    let mut class_of: Vec<i32> = vec![-1; n];
    let mut classes: Vec<Vec<i32>> = Vec::new();
    for v in 0..n as i32 {
        if class_of[v as usize] >= 0 {
            continue;
        }
        let cls_idx = classes.len() as i32;
        class_of[v as usize] = cls_idx;
        let mut members = vec![v];
        if let Some(bucket) = buckets.get(&hashes[v as usize]) {
            for &u in bucket {
                if u <= v || class_of[u as usize] >= 0 {
                    continue;
                }
                if g.vwgt[u as usize] != g.vwgt[v as usize] {
                    continue;
                }
                if closed[v as usize] == closed[u as usize] {
                    class_of[u as usize] = cls_idx;
                    members.push(u);
                }
            }
        }
        classes.push(members);
    }

    let n_compressed = classes.len();
    let ratio = n_compressed as f64 / n as f64;
    if ratio >= max_ratio {
        return None;
    }

    // -- Step 4: assemble compressed CSR.
    let vwgt_c: Vec<i32> = classes
        .iter()
        .map(|cls| cls.iter().map(|&v| g.vwgt[v as usize]).sum())
        .collect();

    // For each class, collect (compressed_neighbor, weight) entries
    // from every original member. Self-loops are dropped.
    let mut adj_per_c: Vec<Vec<(i32, i32)>> = vec![Vec::new(); n_compressed];
    for v in 0..n as i32 {
        let cv = class_of[v as usize] as usize;
        let nbrs = g.neighbors(v);
        let weights = g.edge_weights(v);
        for (idx, &u) in nbrs.iter().enumerate() {
            let cu = class_of[u as usize];
            if cu as usize == cv {
                continue; // self-loop after merge
            }
            adj_per_c[cv].push((cu, weights[idx]));
        }
    }

    // Flatten into CSR with per-class sort + run-length sum (parallel-edge merge).
    let mut xadj_c: Vec<i32> = Vec::with_capacity(n_compressed + 1);
    let mut adjncy_c: Vec<i32> = Vec::new();
    let mut adjwgt_c: Vec<i32> = Vec::new();
    xadj_c.push(0);
    for adj in &mut adj_per_c {
        adj.sort_unstable_by_key(|&(nb, _)| nb);
        let mut i = 0;
        while i < adj.len() {
            let nb = adj[i].0;
            let mut wsum = 0i32;
            while i < adj.len() && adj[i].0 == nb {
                wsum += adj[i].1;
                i += 1;
            }
            adjncy_c.push(nb);
            adjwgt_c.push(wsum);
        }
        xadj_c.push(adjncy_c.len() as i32);
    }

    Some(CompressedGraph {
        graph: Graph {
            nvtxs: n_compressed as i32,
            xadj: xadj_c,
            adjncy: adjncy_c,
            vwgt: vwgt_c,
            adjwgt: adjwgt_c,
        },
        vertex_map: classes,
        ratio,
    })
}

/// Build the sorted closed neighborhood `N[v] = N(v) ∪ {v}` of `v`.
///
/// Relies on `g.neighbors(v)` already being sorted ascending - the
/// from-CSC constructor guarantees this on intake. Uses
/// `partition_point` to insert `v` at the correct sorted position
/// without re-sorting.
fn closed_nbhd_sorted(g: &Graph, v: i32) -> Vec<i32> {
    let nbrs = g.neighbors(v);
    let pos = nbrs.partition_point(|&x| x < v);
    let mut out = Vec::with_capacity(nbrs.len() + 1);
    out.extend_from_slice(&nbrs[..pos]);
    out.push(v);
    out.extend_from_slice(&nbrs[pos..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::csc_from_edges;
    use rslab_ordering_core::CscPattern;

    /// Helper: build a [`Graph`] from undirected edges. Diagonal is
    /// added automatically because `from_csc_pattern` drops it.
    fn graph_from_edges(n: usize, edges: &[(usize, usize)]) -> Graph {
        let mut e: Vec<(usize, usize)> = edges.to_vec();
        for i in 0..n {
            e.push((i, i));
        }
        let (cp, ri) = csc_from_edges(n, &e);
        let pat = CscPattern::new(n, &cp, &ri).expect("valid CSC");
        Graph::from_csc_pattern(&pat).expect("valid graph")
    }

    fn complete_graph(n: usize) -> Graph {
        let mut e = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                e.push((i, j));
            }
        }
        graph_from_edges(n, &e)
    }

    /// Three disconnected `K_4` blocks at vertex offsets 0, 4, 8.
    fn three_k4_blocks() -> Graph {
        let mut e = Vec::new();
        for off in [0, 4, 8] {
            for i in 0..4 {
                for j in (i + 1)..4 {
                    e.push((off + i, off + j));
                }
            }
        }
        graph_from_edges(12, &e)
    }

    /// 4x4 grid (16 vertices). Index = row*4 + col.
    fn grid_4x4() -> Graph {
        let mut e = Vec::new();
        for r in 0..4 {
            for c in 0..4 {
                let v = r * 4 + c;
                if r + 1 < 4 {
                    e.push((v, v + 4));
                }
                if c + 1 < 4 {
                    e.push((v, v + 1));
                }
            }
        }
        graph_from_edges(16, &e)
    }

    fn path_graph(n: usize) -> Graph {
        let mut e = Vec::new();
        for i in 0..(n - 1) {
            e.push((i, i + 1));
        }
        graph_from_edges(n, &e)
    }

    // --- Negative tests: no compression possible -----------------------

    #[test]
    fn grid_4x4_has_no_indistinguishable_vertices() {
        let g = grid_4x4();
        let r = compress_graph(&g, 0.7);
        assert!(
            r.is_none(),
            "4x4 grid: every vertex has a unique adjacency set"
        );
    }

    #[test]
    fn path_graph_has_no_indistinguishable_vertices() {
        let g = path_graph(8);
        let r = compress_graph(&g, 0.7);
        assert!(r.is_none(), "P_8: each interior vertex has unique nbrs");
    }

    #[test]
    fn high_threshold_rejects_modest_compression() {
        // Block-diagonal compresses to ratio 0.25 - below 0.7 default.
        // With threshold = 0.2 (only accept *very* aggressive
        // compression), 0.25 >= 0.2 so we reject.
        let g = three_k4_blocks();
        let r = compress_graph(&g, 0.2);
        assert!(r.is_none(), "ratio 0.25 fails strict-less-than 0.2");
    }

    // --- Positive tests: known compression outcomes --------------------

    #[test]
    fn three_k4_blocks_compress_to_three_supervariables() {
        let g = three_k4_blocks();
        let cg = compress_graph(&g, 0.7).expect("should compress");
        assert_eq!(cg.graph.nvtxs, 3);
        assert_eq!(cg.vwgt(), &[4, 4, 4][..]); // each block = supervariable of weight 4
        assert!(
            cg.graph.adjncy.is_empty(),
            "intra-block edges become self-loops; no inter-block edges exist"
        );
        assert!(cg.ratio < 0.7);
        assert!((cg.ratio - 0.25).abs() < 1e-12);
        assert_eq!(cg.vertex_map.len(), 3);
        assert_eq!(cg.vertex_map[0], vec![0, 1, 2, 3]);
        assert_eq!(cg.vertex_map[1], vec![4, 5, 6, 7]);
        assert_eq!(cg.vertex_map[2], vec![8, 9, 10, 11]);
    }

    #[test]
    fn complete_graph_collapses_to_single_supervariable() {
        let g = complete_graph(6);
        let cg = compress_graph(&g, 0.7).expect("K_6 should compress");
        assert_eq!(cg.graph.nvtxs, 1);
        assert_eq!(cg.vwgt(), &[6][..]);
        assert!(cg.graph.adjncy.is_empty());
        assert_eq!(cg.vertex_map[0], (0..6).collect::<Vec<i32>>());
    }

    // --- Edge-weight summing -------------------------------------------

    /// Two K_3 blocks bridged by a single edge (vertex 2 - vertex 3).
    /// After compression: 4 supervariables { {0,1}, {2}, {3}, {4,5} }.
    /// The {0,1}→{2} compressed edge must carry weight 2 (one
    /// contribution from edge (0,2), one from edge (1,2)). Likewise
    /// {3}→{4,5} totals 2 in each direction. {2}→{3} stays at weight 1.
    #[test]
    fn parallel_edges_sum_their_weights_on_merge() {
        let mut e = Vec::new();
        // K_3 on {0,1,2}
        for i in 0..3 {
            for j in (i + 1)..3 {
                e.push((i, j));
            }
        }
        // K_3 on {3,4,5}
        for i in 3..6 {
            for j in (i + 1)..6 {
                e.push((i, j));
            }
        }
        // Bridge
        e.push((2, 3));
        let g = graph_from_edges(6, &e);
        let cg = compress_graph(&g, 0.7).expect("should compress to 4 superv.");
        assert_eq!(cg.graph.nvtxs, 4);

        // Identify supervariables by their member sets.
        // Class 0 = {0,1}, 1 = {2}, 2 = {3}, 3 = {4,5} (representative-ascending).
        assert_eq!(cg.vertex_map[0], vec![0, 1]);
        assert_eq!(cg.vertex_map[1], vec![2]);
        assert_eq!(cg.vertex_map[2], vec![3]);
        assert_eq!(cg.vertex_map[3], vec![4, 5]);
        assert_eq!(cg.vwgt(), &[2, 1, 1, 2][..]);

        // Weight-2 edges between {0,1}↔{2} and {3}↔{4,5}; weight-1 across the bridge.
        let nb0: Vec<(i32, i32)> = cg
            .graph
            .neighbors(0)
            .iter()
            .zip(cg.graph.edge_weights(0))
            .map(|(&n, &w)| (n, w))
            .collect();
        assert_eq!(nb0, vec![(1, 2)]);

        let nb1: Vec<(i32, i32)> = cg
            .graph
            .neighbors(1)
            .iter()
            .zip(cg.graph.edge_weights(1))
            .map(|(&n, &w)| (n, w))
            .collect();
        assert_eq!(nb1, vec![(0, 2), (2, 1)]);

        let nb2: Vec<(i32, i32)> = cg
            .graph
            .neighbors(2)
            .iter()
            .zip(cg.graph.edge_weights(2))
            .map(|(&n, &w)| (n, w))
            .collect();
        assert_eq!(nb2, vec![(1, 1), (3, 2)]);

        let nb3: Vec<(i32, i32)> = cg
            .graph
            .neighbors(3)
            .iter()
            .zip(cg.graph.edge_weights(3))
            .map(|(&n, &w)| (n, w))
            .collect();
        assert_eq!(nb3, vec![(2, 2)]);
    }

    // --- Bijection on expansion ----------------------------------------

    #[test]
    fn vertex_map_flattened_is_a_permutation() {
        let g = three_k4_blocks();
        let cg = compress_graph(&g, 0.7).expect("should compress");
        let mut flat: Vec<i32> = Vec::new();
        for cls in &cg.vertex_map {
            flat.extend_from_slice(cls);
        }
        assert_eq!(flat.len(), 12);
        let mut sorted = flat.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..12).collect::<Vec<i32>>());
    }

    // --- Conservation invariants ---------------------------------------

    #[test]
    fn vertex_weights_conserve_under_compression() {
        let g = complete_graph(7);
        let cg = compress_graph(&g, 0.7).expect("K_7 compresses");
        let total_in: i32 = g.vwgt.iter().sum();
        let total_out: i32 = cg.graph.vwgt.iter().sum();
        assert_eq!(total_in, total_out);
    }

    #[test]
    fn external_edge_weights_conserve_under_compression() {
        // Use the bridged-K_3 graph: total external weight after
        // compression should match the sum of edge weights in the
        // original graph minus those that became self-loops.
        let mut e = Vec::new();
        for i in 0..3 {
            for j in (i + 1)..3 {
                e.push((i, j));
            }
        }
        for i in 3..6 {
            for j in (i + 1)..6 {
                e.push((i, j));
            }
        }
        e.push((2, 3));
        let g = graph_from_edges(6, &e);
        let cg = compress_graph(&g, 0.7).unwrap();

        // Original edges: 3 in each K_3 + 1 bridge = 7 undirected,
        // each stored twice = 14 directed entries with unit weight.
        // Self-loops after compression: all 3 K_3 edges in each block
        // collapse - 6 undirected = 12 directed entries.
        // Surviving directed weight: 14 - 12 = 2 (the bridge x2).
        // Plus the inter-class edges {0,1}↔{2} and {3}↔{4,5} are
        // unchanged in directed-weight total because each original
        // directed entry maps 1:1 to a directed compressed entry.
        // Wait - those K_3 edges are intra-class for the {0,1} and
        // {4,5} classes, but the (0,2), (1,2), (3,4), (3,5) edges are
        // inter-class. Let me redo:
        //   Intra-class K_3 edges (collapsed to self-loops):
        //     (0,1) - {0,1} ↔ {0,1}: self-loop. 2 directed entries.
        //     (4,5) - {4,5} ↔ {4,5}: self-loop. 2 directed entries.
        //   Total intra-class = 4 directed entries dropped.
        //   Surviving directed weight = 14 - 4 = 10.
        let total_out: i32 = cg.graph.adjwgt.iter().sum();
        assert_eq!(total_out, 10);
    }

    // --- Helper accessors used in assertions above ---------------------

    impl CompressedGraph {
        fn vwgt(&self) -> &[i32] {
            &self.graph.vwgt
        }
    }
}
