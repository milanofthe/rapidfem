//! Flow-based node separator (KaHIP Phase K4).
//!
//! Converts a refined edge bisection (from K3) into a minimum
//! weighted node separator via a bipartite vertex-cover reduction:
//! the boundary vertices of each side form the two parts of a
//! bipartite graph whose edges are the cross-cut edges, and
//! König's theorem equates min-weight vertex cover with the
//! min `s-t` cut on a directed network with source/sink arcs
//! carrying vertex weights and cross-arcs carrying infinite
//! capacity.
//!
//! Reference: Sanders & Schulz 2011 §4.4. See
//! `dev/research/ordering-kahip-k4.md` for the full construction
//! and test-oracle list.
//!
//! v1 scope: unit vertex weights when `vweight = None`; no
//! internal balance enforcement (caller decides). Callers are
//! K5 (V/F-cycle controller) and K6 (driver) which will land
//! later; module remains internal until then.

use crate::flow::push_relabel;
use crate::graph::UndirectedGraph;

/// Node separator produced from an edge bisection.
#[derive(Debug, Clone)]
pub(crate) struct NodeSeparator {
    /// Per-vertex label in `{0, 1, 2}`: 0 = part A, 1 = part B,
    /// 2 = separator. Parallel to the graph's vertex set.
    pub part: Vec<u8>,
    /// Total vertex weight of the separator.
    pub weight: i64,
}

/// Compute a min-weight node separator from an edge bisection via
/// the boundary-bipartite vertex-cover reduction.
///
/// Returns `None` if `where_` is malformed or if there are no
/// cross-cut edges (empty separator is trivially optimal and the
/// caller should short-circuit).
pub(crate) fn flow_node_separator(
    graph: &UndirectedGraph,
    where_: &[u8],
    vweight: Option<&[i64]>,
) -> Option<NodeSeparator> {
    if where_.len() != graph.n {
        return None;
    }
    for &p in where_ {
        if p > 1 {
            return None;
        }
    }
    if let Some(w) = vweight {
        if w.len() != graph.n {
            return None;
        }
        for &wv in w {
            if wv < 1 {
                return None;
            }
        }
    }
    let get_vw = |v: usize| -> i64 { vweight.map(|w| w[v]).unwrap_or(1) };

    // Boundary = vertices with a neighbor in the opposite part.
    let mut on_boundary = vec![false; graph.n];
    for u in 0..graph.n {
        for &v in graph.neighbors(u) {
            if where_[u] != where_[v] {
                on_boundary[u] = true;
                break;
            }
        }
    }
    let b0: Vec<usize> = (0..graph.n)
        .filter(|&v| where_[v] == 0 && on_boundary[v])
        .collect();
    let b1: Vec<usize> = (0..graph.n)
        .filter(|&v| where_[v] == 1 && on_boundary[v])
        .collect();
    if b0.is_empty() || b1.is_empty() {
        // No cross-cut edges - trivial empty separator.
        return None;
    }

    // Network id mapping.
    let mut net_id: Vec<i32> = vec![-1; graph.n];
    for (i, &v) in b0.iter().enumerate() {
        net_id[v] = i as i32;
    }
    for (i, &v) in b1.iter().enumerate() {
        net_id[v] = (b0.len() + i) as i32;
    }
    let src = b0.len() + b1.len();
    let sink = src + 1;
    let n_net = sink + 1;

    // INF = (sum of boundary vertex weights) + 1. Bounded by the
    // sum of source-arc and sink-arc capacities, so no flow can
    // actually saturate INF on a cross-arc - guaranteeing the min
    // cut lies on source/sink arcs only.
    let mut total_bnd_weight: i64 = 0;
    for &v in b0.iter().chain(b1.iter()) {
        total_bnd_weight = total_bnd_weight.saturating_add(get_vw(v));
    }
    let inf_cap: i64 = total_bnd_weight.saturating_add(1);

    let mut edges: Vec<(usize, usize, i64)> = Vec::new();
    // Source arcs.
    for &v in &b0 {
        edges.push((src, net_id[v] as usize, get_vw(v)));
    }
    // Sink arcs.
    for &v in &b1 {
        edges.push((net_id[v] as usize, sink, get_vw(v)));
    }
    // Cross arcs (directed B_0 → B_1 only).
    for &u in &b0 {
        for (j, &v) in graph.neighbors(u).iter().enumerate() {
            if where_[v] != 1 {
                continue;
            }
            if net_id[v] < 0 {
                continue;
            }
            let _w = graph.eweights(u)[j]; // edge weight unused (cross-arc is INF).
            edges.push((net_id[u] as usize, net_id[v] as usize, inf_cap));
        }
    }

    let (_flow_value, is_source_side) = match push_relabel(n_net, &edges, src, sink) {
        Ok(p) => p,
        Err(_) => return None,
    };

    // Extract separator: v ∈ B_0 is picked iff NOT source-side
    // (its source-arc is saturated); v ∈ B_1 is picked iff
    // source-side (its sink-arc is saturated).
    let mut part: Vec<u8> = where_.to_vec();
    let mut weight: i64 = 0;
    for &v in &b0 {
        let nid = net_id[v] as usize;
        if !is_source_side[nid] {
            part[v] = 2;
            weight += get_vw(v);
        }
    }
    for &v in &b1 {
        let nid = net_id[v] as usize;
        if is_source_side[nid] {
            part[v] = 2;
            weight += get_vw(v);
        }
    }

    debug_assert!(is_valid_separator(graph, &part));

    Some(NodeSeparator { part, weight })
}

/// Returns true iff every edge with endpoints in parts 0 and 1
/// has at least one endpoint in part 2 (the separator).
fn is_valid_separator(graph: &UndirectedGraph, part: &[u8]) -> bool {
    for u in 0..graph.n {
        if part[u] != 0 {
            continue;
        }
        for &v in graph.neighbors(u) {
            if part[v] == 1 {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(n: usize, edges: &[(usize, usize, i64)]) -> UndirectedGraph {
        let mut per_v: Vec<Vec<(usize, i64)>> = vec![Vec::new(); n];
        for &(u, v, w) in edges {
            per_v[u].push((v, w));
            per_v[v].push((u, w));
        }
        let mut xadj = Vec::with_capacity(n + 1);
        let mut adjncy = Vec::new();
        let mut eweight = Vec::new();
        xadj.push(0);
        for list in per_v.iter_mut() {
            list.sort_by_key(|&(v, _)| v);
            for &(v, w) in list.iter() {
                adjncy.push(v);
                eweight.push(w);
            }
            xadj.push(adjncy.len());
        }
        UndirectedGraph {
            n,
            xadj,
            adjncy,
            eweight,
            vweight: vec![1; n],
        }
    }

    fn grid(k: usize) -> UndirectedGraph {
        let n = k * k;
        let idx = |r: usize, c: usize| r * k + c;
        let mut edges = Vec::new();
        for r in 0..k {
            for c in 0..k {
                if c + 1 < k {
                    edges.push((idx(r, c), idx(r, c + 1), 1));
                }
                if r + 1 < k {
                    edges.push((idx(r, c), idx(r + 1, c), 1));
                }
            }
        }
        build(n, &edges)
    }

    #[test]
    fn empty_boundary_returns_none() {
        let g = build(3, &[(0, 1, 1), (1, 2, 1)]);
        let w = vec![0u8, 0, 0];
        assert!(flow_node_separator(&g, &w, None).is_none());
    }

    #[test]
    fn malformed_inputs_return_none() {
        let g = build(3, &[(0, 1, 1), (1, 2, 1)]);
        // Wrong where_ length.
        assert!(flow_node_separator(&g, &[0u8, 0], None).is_none());
        // Bad label.
        assert!(flow_node_separator(&g, &[0u8, 5, 1], None).is_none());
        // Bad vweight length.
        assert!(flow_node_separator(&g, &[0u8, 0, 1], Some(&[1, 1])).is_none());
        // Zero vweight.
        assert!(flow_node_separator(&g, &[0u8, 0, 1], Some(&[1, 0, 1])).is_none());
    }

    #[test]
    fn path_nine_bisected_gives_separator_of_one() {
        // 0-1-2-3-4-5-6-7-8; partition {0..3} | {4..8}. Only cut
        // edge is (3, 4); min vertex cover = 1 vertex.
        let g = build(
            9,
            &[
                (0, 1, 1),
                (1, 2, 1),
                (2, 3, 1),
                (3, 4, 1),
                (4, 5, 1),
                (5, 6, 1),
                (6, 7, 1),
                (7, 8, 1),
            ],
        );
        let w = vec![0u8, 0, 0, 0, 1, 1, 1, 1, 1];
        let sep = flow_node_separator(&g, &w, None).unwrap();
        assert_eq!(sep.weight, 1);
        let sep_vertices: Vec<usize> = (0..9).filter(|&v| sep.part[v] == 2).collect();
        assert_eq!(sep_vertices.len(), 1);
        assert!(sep_vertices[0] == 3 || sep_vertices[0] == 4);
    }

    #[test]
    fn grid_7x7_horizontal_bisection_separator_size_7() {
        // rows 0..3 in part 0, rows 4..6 in part 1. Crossing
        // edges form a perfect matching (r=3,c)-(r=4,c) for
        // c = 0..6 → König min vertex cover = 7.
        let g = grid(7);
        let mut w = vec![0u8; 49];
        for r in 0..7 {
            for c in 0..7 {
                w[r * 7 + c] = if r < 4 { 0 } else { 1 };
            }
        }
        let sep = flow_node_separator(&g, &w, None).unwrap();
        assert_eq!(sep.weight, 7);
        let sep_count = sep.part.iter().filter(|&&p| p == 2).count();
        assert_eq!(sep_count, 7);
        // Every separator vertex must lie on row 3 or row 4
        // (both are valid König covers).
        for (v, &p) in sep.part.iter().enumerate() {
            if p == 2 {
                let r = v / 7;
                assert!(
                    r == 3 || r == 4,
                    "separator vertex {} at row {} is neither 3 nor 4",
                    v,
                    r
                );
            }
        }
    }

    #[test]
    fn k33_bipartite_separator_size_3() {
        // K_{3,3}: 0,1,2 on left, 3,4,5 on right, every pair
        // connected. Bisection {0,1,2} | {3,4,5}; min vertex
        // cover = 3.
        let mut edges = Vec::new();
        for u in 0..3 {
            for v in 3..6 {
                edges.push((u, v, 1));
            }
        }
        let g = build(6, &edges);
        let w = vec![0u8, 0, 0, 1, 1, 1];
        let sep = flow_node_separator(&g, &w, None).unwrap();
        assert_eq!(sep.weight, 3);
    }

    #[test]
    fn determinism_across_repeats() {
        let g = grid(5);
        let mut w = vec![0u8; 25];
        for r in 0..5 {
            for c in 0..5 {
                w[r * 5 + c] = if r < 3 { 0 } else { 1 };
            }
        }
        let s1 = flow_node_separator(&g, &w, None).unwrap();
        let s2 = flow_node_separator(&g, &w, None).unwrap();
        assert_eq!(s1.part, s2.part);
        assert_eq!(s1.weight, s2.weight);
    }

    #[test]
    fn separator_is_valid_on_random_bisection() {
        // Deterministic LCG random graph; verify the separator
        // property for any refined bisection.
        let n = 30;
        let mut state: u64 = 0xCAFEBABE;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        };
        let mut edges: Vec<(usize, usize, i64)> = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if (next() % 100) < 15 {
                    edges.push((u, v, 1));
                }
            }
        }
        let g = build(n, &edges);
        let w: Vec<u8> = (0..n).map(|_| (next() % 2) as u8).collect();
        if let Some(sep) = flow_node_separator(&g, &w, None) {
            // Validated by the debug_assert inside the function,
            // but re-check on a release build too.
            assert!(is_valid_separator(&g, &sep.part));
        }
    }

    #[test]
    fn weighted_separator_prefers_lighter_cover() {
        // Path 0-1-2. Partition {0} | {2} with 1 as the only
        // cross-path. Bisection is impossible here because
        // vertex 1 is neighbor to both parts but must belong to
        // one of {0, 1}. Instead: path 0-1-2-3, {0,1} | {2,3};
        // cross edge (1, 2). Heavy weight on vertex 1, light
        // weight on vertex 2 → min-weight cover is {2}.
        let g = build(4, &[(0, 1, 1), (1, 2, 1), (2, 3, 1)]);
        let w = vec![0u8, 0, 1, 1];
        let vw = vec![1i64, 100, 1, 1];
        let sep = flow_node_separator(&g, &w, Some(&vw)).unwrap();
        assert_eq!(sep.weight, 1);
        assert_eq!(sep.part[2], 2);
        assert_ne!(sep.part[1], 2);
    }

    #[test]
    fn interior_vertices_are_not_in_separator() {
        // 2D grid 5x5, horizontal bisection at row 2/3. Interior
        // vertices (rows 0, 1 in part 0; rows 4 in part 1) must
        // never be in the separator since they have no cross-
        // part neighbors.
        let g = grid(5);
        let mut w = vec![0u8; 25];
        for r in 0..5 {
            for c in 0..5 {
                w[r * 5 + c] = if r < 3 { 0 } else { 1 };
            }
        }
        let sep = flow_node_separator(&g, &w, None).unwrap();
        for r in 0..5 {
            for c in 0..5 {
                let v = r * 5 + c;
                let interior_p0 = r < 2;
                let interior_p1 = r > 3;
                if interior_p0 || interior_p1 {
                    assert_ne!(
                        sep.part[v], 2,
                        "interior vertex ({},{}) should not be in separator",
                        r, c
                    );
                }
            }
        }
    }
}
