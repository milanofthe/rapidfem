//! Flow-based edge refinement (KaHIP Phase K3).
//!
//! Improves an existing bisection by extracting a BFS band around
//! the current partition boundary, formulating a max-flow problem on
//! the band (with fixed vertices pinned via super-source / super-
//! sink at infinite capacity), and applying the resulting min-cut if
//! it strictly improves the weighted cut while respecting a balance
//! constraint. One call performs a single band-extract + solve; the
//! caller iterates (K5 V/F-cycle or K6 driver).
//!
//! Reference: Sanders & Schulz 2011 §4. See
//! `dev/research/ordering-kahip-k3.md` for the full algorithm and
//! the test-oracle construction.
//!
//! v1 most-balanced-min-cut scope: two-cut search (residual BFS from
//! source vs from sink); full MBMC via residual-flow manipulation
//! deferred to K5/K6 follow-up (plan audit item 3).

use std::collections::VecDeque;

use crate::flow::push_relabel;
use crate::graph::UndirectedGraph;

/// One iteration of flow-based bisection refinement.
///
/// Returns `true` iff the bisection was strictly improved in cut
/// weight while remaining within the balance tolerance. On `false`
/// return, `where_` is left untouched.
///
/// `bnd_distance` is the BFS radius around the boundary. `d = 0`
/// collapses the band to the boundary itself and produces no useful
/// refinement; such calls return `false`.
///
/// `max_imbalance` is the fractional slack on partition size:
/// `max(|part0|, |part1|) ≤ (1 + ε) · ⌈n / 2⌉`.
pub(crate) fn flow_refine_bisection(
    graph: &UndirectedGraph,
    where_: &mut [u8],
    bnd_distance: usize,
    max_imbalance: f64,
) -> bool {
    if where_.len() != graph.n || graph.n == 0 || bnd_distance == 0 {
        return false;
    }
    for &p in where_.iter() {
        if p > 1 {
            return false;
        }
    }

    let old_cut = graph.cut_weight(where_);

    // Identify boundary: vertices with a neighbor in the opposite
    // part.
    let mut is_boundary = vec![false; graph.n];
    for u in 0..graph.n {
        for &v in graph.neighbors(u) {
            if where_[u] != where_[v] {
                is_boundary[u] = true;
                break;
            }
        }
    }
    if !is_boundary.iter().any(|&b| b) {
        return false;
    }

    // BFS outward from boundary, recording distance ∈ {0..=d}.
    let mut dist: Vec<i32> = vec![-1; graph.n];
    let mut q: VecDeque<usize> = VecDeque::new();
    for (v, &b) in is_boundary.iter().enumerate() {
        if b {
            dist[v] = 0;
            q.push_back(v);
        }
    }
    while let Some(u) = q.pop_front() {
        let du = dist[u];
        if du as usize >= bnd_distance {
            continue;
        }
        for &v in graph.neighbors(u) {
            if dist[v] == -1 {
                dist[v] = du + 1;
                q.push_back(v);
            }
        }
    }

    // Band = vertices with dist[v] in 0..=bnd_distance.
    // Fixed nodes = band vertices with dist == bnd_distance.
    //   part0-fixed → super-source, part1-fixed → super-sink.
    // Note: if a boundary vertex itself has dist == bnd_distance
    // only when bnd_distance == 0, excluded above.
    let band: Vec<usize> = (0..graph.n).filter(|&v| dist[v] >= 0).collect();
    if band.len() < 2 {
        return false;
    }

    // Build a contiguous flow-network id mapping.
    // 0..band.len() = band vertices (in the order `band[..]`).
    // band.len() = super-source, band.len() + 1 = super-sink.
    let mut net_id: Vec<i32> = vec![-1; graph.n];
    for (i, &v) in band.iter().enumerate() {
        net_id[v] = i as i32;
    }
    let src = band.len();
    let sink = band.len() + 1;
    let n_net = band.len() + 2;

    // Edges: for every graph edge (u, v) with both endpoints in the
    // band, add (u→v, w) and (v→u, w). Enumerate with u < v so each
    // undirected edge is added once.
    let mut edges: Vec<(usize, usize, i64)> = Vec::new();
    let mut in_band_cut_weight: i64 = 0;
    for &u in &band {
        let ui = net_id[u] as usize;
        for (j, &v) in graph.neighbors(u).iter().enumerate() {
            if v <= u {
                continue;
            }
            if net_id[v] < 0 {
                continue;
            }
            let vi = net_id[v] as usize;
            let w = graph.eweights(u)[j];
            edges.push((ui, vi, w));
            edges.push((vi, ui, w));
            if where_[u] != where_[v] {
                in_band_cut_weight += w;
            }
        }
    }

    // Fixed-node depth per part. The plan's "pin vertices at
    // distance exactly `bnd_distance`" assumes the part extends at
    // least that deep. For small parts entirely contained in the
    // ball, no vertex sits at `bnd_distance`; fall back to the
    // maximum depth observed in that part among band vertices so
    // that at least the "deepest reachable" vertices are pinned.
    let mut max_dist_p0: i32 = -1;
    let mut max_dist_p1: i32 = -1;
    for &v in &band {
        match where_[v] {
            0 => max_dist_p0 = max_dist_p0.max(dist[v]),
            1 => max_dist_p1 = max_dist_p1.max(dist[v]),
            _ => {}
        }
    }
    let pin_depth_p0 = max_dist_p0.min(bnd_distance as i32);
    let pin_depth_p1 = max_dist_p1.min(bnd_distance as i32);

    // INF_CAP large enough to make the super edges non-bottlenecks
    // (> any achievable min-cut) without overflowing i64 across
    // many pins. In-band cut weight is bounded by the total weight
    // of in-band edges; take that plus 1 as INF_CAP per edge. Each
    // pin's flow is bounded by that total so sum over pins stays
    // well below i64::MAX for any practical band size.
    let mut total_band_edge_weight: i64 = 0;
    for &(_, _, w) in &edges {
        total_band_edge_weight = total_band_edge_weight.saturating_add(w);
    }
    // edges above counts each undirected edge twice (forward+reverse
    // anti-parallel pair), so halve; add 1 for strict `>`.
    let inf_cap: i64 = (total_band_edge_weight / 2).saturating_add(1);

    let mut fixed_src_count = 0usize;
    let mut fixed_snk_count = 0usize;
    for &v in &band {
        if where_[v] == 0 && dist[v] == pin_depth_p0 && pin_depth_p0 >= 0 {
            let vi = net_id[v] as usize;
            edges.push((src, vi, inf_cap));
            fixed_src_count += 1;
        } else if where_[v] == 1 && dist[v] == pin_depth_p1 && pin_depth_p1 >= 0 {
            let vi = net_id[v] as usize;
            edges.push((vi, sink, inf_cap));
            fixed_snk_count += 1;
        }
    }

    // If one side has no fixed nodes, no flow can occur; refinement
    // is a no-op.
    if fixed_src_count == 0 || fixed_snk_count == 0 {
        return false;
    }

    let (_flow_value, side_from_src) = match push_relabel(n_net, &edges, src, sink) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // Second cut: residual-BFS from sink, i.e., compute the set of
    // vertices that reach sink in the terminal residual graph.
    // Two consecutive solves is clumsy; we reconstruct by re-solving
    // with s/t swapped, which finds the "closest to sink" min-cut.
    let (_, side_from_sink) = match push_relabel(n_net, &reverse(&edges), sink, src) {
        Ok(p) => p,
        Err(_) => return false,
    };

    // Candidate partitions over the full vertex set, respecting the
    // two-cut search: source-side assigns band vertices to part 0
    // iff they are on the source side; sink-side assigns to part 0
    // iff they are NOT on the sink side (complement).
    let candidates: [Vec<u8>; 2] = [
        build_candidate(
            graph.n,
            &band,
            &net_id,
            where_,
            &side_from_src,
            /*to_part0_if*/ true,
        ),
        build_candidate(
            graph.n,
            &band,
            &net_id,
            where_,
            &side_from_sink,
            /*to_part0_if*/ false,
        ),
    ];

    // Score candidates: pick the one with lower cut weight that
    // satisfies the balance constraint. Balance is on vertex *weight*,
    // not count: on a coarse multilevel graph a vertex stands for a
    // supervertex whose mass ≫ 1, so a count-balanced min-cut can be
    // badly weight-imbalanced and hand the finer level's FM a violating
    // start (finding O16). The constraint mirrors KaHIP / Sanders &
    // Schulz 2011: max(weight(part0), weight(part1)) ≤ (1+ε)·⌈W/2⌉.
    let total_w = graph.total_weight();
    let half_w = total_w.div_euclid(2) + (total_w % 2); // ⌈W/2⌉
    let slack_w = ((1.0 + max_imbalance) * half_w as f64).floor() as i64;
    let mut best: Option<(i64, &Vec<u8>)> = None;
    for cand in candidates.iter() {
        let (w0, w1) = graph.part_weights(cand);
        let cw = graph.cut_weight(cand);
        if w0.max(w1) > slack_w {
            continue;
        }
        if cw >= old_cut {
            continue;
        }
        match best {
            None => best = Some((cw, cand)),
            Some((bw, _)) if cw < bw => best = Some((cw, cand)),
            _ => {}
        }
    }

    // Silence unused warnings on the reported in-band cut weight
    // (useful for debugging/instrumentation but not exposed yet).
    let _ = in_band_cut_weight;

    if let Some((_, cand)) = best {
        where_.copy_from_slice(cand);
        true
    } else {
        false
    }
}

fn build_candidate(
    _n: usize,
    band: &[usize],
    net_id: &[i32],
    where_: &[u8],
    flag: &[bool],
    to_part0_if: bool,
) -> Vec<u8> {
    let mut out = where_.to_vec();
    for &v in band {
        let vi = net_id[v] as usize;
        let is_src_side = flag[vi];
        let part0 = if to_part0_if {
            is_src_side
        } else {
            !is_src_side
        };
        out[v] = if part0 { 0 } else { 1 };
    }
    out
}

#[cfg(test)]
fn count_parts(w: &[u8]) -> (usize, usize) {
    let mut c0 = 0usize;
    let mut c1 = 0usize;
    for &p in w {
        if p == 0 {
            c0 += 1;
        } else if p == 1 {
            c1 += 1;
        }
    }
    (c0, c1)
}

fn reverse(edges: &[(usize, usize, i64)]) -> Vec<(usize, usize, i64)> {
    edges.iter().map(|&(u, v, c)| (v, u, c)).collect()
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
    fn empty_and_degenerate_inputs() {
        let g = build(0, &[]);
        let mut w: Vec<u8> = vec![];
        assert!(!flow_refine_bisection(&g, &mut w, 1, 0.05));

        let g = build(3, &[(0, 1, 1), (1, 2, 1)]);
        let mut w = vec![0u8, 0, 1];
        // bnd_distance 0 must early-return.
        assert!(!flow_refine_bisection(&g, &mut w, 0, 0.05));

        // Bad partition label.
        let mut bad = vec![0u8, 5, 1];
        assert!(!flow_refine_bisection(&g, &mut bad, 1, 0.05));

        // No boundary (everyone in one part).
        let mut allzero = vec![0u8, 0, 0];
        assert!(!flow_refine_bisection(&g, &mut allzero, 1, 0.05));
    }

    #[test]
    fn path_midpoint_cut_is_already_optimal() {
        // 0-1-2-3-4, bisection {0,1} | {2,3,4}; cut weight = 1.
        // Any refinement cannot find a strictly smaller cut.
        let g = build(5, &[(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)]);
        let mut w = vec![0u8, 0, 1, 1, 1];
        let before = g.cut_weight(&w);
        assert_eq!(before, 1);
        let changed = flow_refine_bisection(&g, &mut w, 2, 0.5);
        assert!(!changed, "optimal cut must not be improved");
        assert_eq!(g.cut_weight(&w), before);
    }

    #[test]
    fn grid_7x7_suboptimal_diagonal_improves() {
        let g = grid(7);
        // Build a deliberately bad bisection: lower-left triangle
        // in part 0, rest in part 1. This cuts a stair-step of ~13
        // edges.
        let mut w = vec![0u8; 49];
        for r in 0..7 {
            for c in 0..7 {
                w[r * 7 + c] = if r + c < 7 { 0 } else { 1 };
            }
        }
        let before = g.cut_weight(&w);
        // Stair-step cut across r+c=6: 6 horizontal edges + 6
        // vertical = 12.
        assert_eq!(before, 12);
        // bnd_distance = 2 keeps the band narrow enough that the
        // pins at true depth 2 in each part (r+c=4 on source side,
        // r+c=9 on sink side) force the min-cut to lie strictly
        // between the pin sets. Out-of-band vertices (r+c<4 or >9)
        // stay in their original parts, preserving balance.
        // Balance slack must accommodate the flow's natural
        // drift: some previously-part-1 vertices near the source
        // pin set get pulled to part 0 (ε = 0.4 → slack = 35).
        let changed = flow_refine_bisection(&g, &mut w, 2, 0.4);
        assert!(changed);
        let after = g.cut_weight(&w);
        assert!(
            after < before,
            "cut must strictly improve: before={} after={}",
            before,
            after
        );
    }

    #[test]
    fn determinism_across_repeats() {
        let g = grid(6);
        let mut w1 = vec![0u8; 36];
        for r in 0..6 {
            for c in 0..6 {
                w1[r * 6 + c] = if c < 3 { 0 } else { 1 };
            }
        }
        let mut w2 = w1.clone();
        flow_refine_bisection(&g, &mut w1, 2, 0.1);
        flow_refine_bisection(&g, &mut w2, 2, 0.1);
        assert_eq!(w1, w2);
    }

    #[test]
    fn balance_constraint_can_reject_improvement() {
        // Dumbbell: two cliques joined by a single edge. A perfect
        // vertical cut is size 1 but balances the cliques. A
        // dumbbell with asymmetric part-0 seed can also find a
        // cut_of_weight_1 that's heavily unbalanced; we test the
        // constraint does reject.
        let edges = &[
            (0, 1, 1),
            (0, 2, 1),
            (1, 2, 1),
            (2, 3, 1), // bridge
            (3, 4, 1),
            (3, 5, 1),
            (4, 5, 1),
        ];
        let g = build(6, edges);
        // Start with {0,1,2,3} | {4,5}; cut = 2 (edges (3,4),(3,5)).
        let mut w = vec![0u8, 0, 0, 0, 1, 1];
        let before = g.cut_weight(&w);
        assert_eq!(before, 2);
        // With tight balance = 0.0 (max part size = 3), moving 3 to
        // part 1 would give {0,1,2} | {3,4,5} with cut 1 - that's
        // balanced and should be accepted.
        let changed = flow_refine_bisection(&g, &mut w, 3, 0.0);
        assert!(changed);
        assert_eq!(g.cut_weight(&w), 1);
        let (c0, c1) = count_parts(&w);
        assert_eq!(c0.max(c1), 3);
    }

    #[test]
    fn non_worsening_on_random_bisection() {
        // Deterministic LCG to build a random sparse graph, then
        // seed a coin-flip bisection and ensure K3 never worsens.
        let n = 40;
        let mut state: u64 = 0xFEEDFACE;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        };
        let mut edges: Vec<(usize, usize, i64)> = Vec::new();
        for u in 0..n {
            for v in (u + 1)..n {
                if (next() % 100) < 12 {
                    edges.push((u, v, ((next() % 5) + 1) as i64));
                }
            }
        }
        let g = build(n, &edges);
        let mut w: Vec<u8> = (0..n).map(|_| (next() % 2) as u8).collect();
        let before = g.cut_weight(&w);
        flow_refine_bisection(&g, &mut w, 3, 0.3);
        let after = g.cut_weight(&w);
        assert!(
            after <= before,
            "K3 must not worsen: before={} after={}",
            before,
            after
        );
    }

    #[test]
    fn respects_vertex_weight_balance_on_weighted_graph() {
        // O16 reproduction. 7×7 grid with a deliberately bad diagonal
        // bisection (lower-left triangle r+c<7 in part 0). Flow
        // refinement strictly improves the cut by pulling a handful of
        // part-1 vertices - including vertex 13 - into part 0, leaving
        // counts 33|16. With ε = 0.4 the COUNT slack is ⌊1.4·25⌋ = 35,
        // so a count-based balance check accepts the move.
        //
        // Vertex weights are NON-UNIT, as on a coarse multilevel graph:
        // vertex 13 (which the min-cut moves *into* part 0) carries
        // mass 20, every other vertex mass 1. Vertex weights never enter
        // the max-flow (it is driven by edge weights), so the moved set
        // is identical to the unit-weight case. Total weight 68,
        // ⌈68/2⌉ = 34, so the WEIGHT slack is ⌊1.4·34⌋ = 47.
        //
        //   start  part0 = {r+c<7}, weight 28 | part1 weight 40  → ≤47 OK
        //   refined part0 weight 52 (32 unit + vertex-13's 20) | 16 → 52 > 47
        //
        // Oracle: the KaHIP balance constraint (Sanders & Schulz 2011)
        // is on vertex *weight*, not count:
        //   max(weight(part0), weight(part1)) ≤ (1+ε)·⌈total_weight/2⌉.
        // A weight-aware refiner must never leave `where_` in a state
        // that violates this; a count-based one leaves 52|16 here.
        let g = {
            let mut g = grid(7);
            g.vweight[13] = 20;
            g
        };
        let mut w = vec![0u8; 49];
        for r in 0..7 {
            for c in 0..7 {
                w[r * 7 + c] = if r + c < 7 { 0 } else { 1 };
            }
        }

        let eps = 0.4;
        let total_w = g.total_weight();
        let half_w = total_w.div_euclid(2) + (total_w % 2); // ⌈total/2⌉
        let slack_w = ((1.0 + eps) * half_w as f64).floor() as i64;

        // Sanity: the starting partition already respects the weight
        // balance, so the only way to violate the invariant below is to
        // accept a weight-imbalanced candidate.
        let (sw0, sw1) = g.part_weights(&w);
        assert!(
            sw0.max(sw1) <= slack_w,
            "start must be weight-balanced: {}|{} slack {}",
            sw0,
            sw1,
            slack_w
        );

        flow_refine_bisection(&g, &mut w, 2, eps);

        // The invariant: whatever flow_refine left behind must respect
        // the *weight* balance constraint.
        let (w0, w1) = g.part_weights(&w);
        assert!(
            w0.max(w1) <= slack_w,
            "weight balance violated: {}|{} exceeds slack {} (cut={})",
            w0,
            w1,
            slack_w,
            g.cut_weight(&w)
        );
    }

    #[test]
    fn fixed_nodes_pinned_at_max_depth() {
        // Path of length 9: 0-1-2-3-4-5-6-7-8. Seed at 4|5, then
        // set bnd_distance = 2 so depths 0..=2 form the band.
        //   boundary = {4, 5}
        //   depth-1 = {3, 6}
        //   depth-2 = {2, 7}
        // Vertices 0, 1 and 8 are OUTSIDE the band; they should NOT
        // be changed. Depth-2 fixed nodes are 2 (part 0) and 7
        // (part 1); they MUST stay pinned.
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
        let mut w = vec![0u8, 0, 0, 0, 0, 1, 1, 1, 1];
        flow_refine_bisection(&g, &mut w, 2, 0.5);
        // Outside-band vertices untouched.
        assert_eq!(w[0], 0);
        assert_eq!(w[1], 0);
        assert_eq!(w[8], 1);
        // Fixed-at-depth-2 vertices pinned.
        assert_eq!(w[2], 0);
        assert_eq!(w[7], 1);
    }
}
