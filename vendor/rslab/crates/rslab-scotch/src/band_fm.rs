//! Band FM: edge-bisection FM restricted to a band around the boundary.
//!
//! Extracts a sub-graph of vertices within `width` BFS hops of the
//! current boundary, adds two artificial **anchor supervertices**
//! (one per side) whose weight equals the total out-of-band weight on
//! that side, runs FM on the band, and projects the refined labels
//! back to the original graph.
//!
//! Anchors solve the balance-accounting bug flagged in audit finding
//! 2 of `dev/plans/ordering-scotch.md`: without them, FM in the band
//! cannot see how much weight is on each side outside the band, and
//! happily produces band-internal "improvements" that violate global
//! balance. Anchors carry that mass and are pinned for the entire
//! pass so FM cannot move them.
//!
//! Clean-room from Pellegrini 1996 §3 plus the SCOTCH band-graph
//! construction described in audit finding 2 (no code is paraphrased
//! from `bgraph_bipart_bd.c`). See research note
//! `dev/research/scotch-band-fm.md`.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use rslab_metis::internals::graph::Graph;
use rslab_metis::internals::initial_partition::{cut_size, part_weight, PART_A, PART_B};

/// Refine an edge bisection by band FM with anchor supervertices.
///
/// `labels[v] ∈ {PART_A, PART_B}` on entry and exit. `width` is the
/// BFS depth of the band around the boundary; SCOTCH defaults to
/// 3. Returns the final edge cut on the *original* graph.
pub fn band_fm_refine(
    graph: &Graph,
    labels: &mut [u8],
    width: u32,
    max_imbalance: f64,
    max_passes: u32,
) -> i32 {
    let n = graph.nvtxs as usize;
    if n < 2 {
        return cut_size(graph, labels);
    }
    if width == 0 {
        return cut_size(graph, labels);
    }

    // 1. Identify boundary vertices.
    let boundary = boundary_vertices(graph, labels);
    if boundary.is_empty() {
        return cut_size(graph, labels);
    }

    // 2. BFS to depth `width` from boundary; mark in-band vertices.
    let in_band = bfs_band(graph, &boundary, width);

    // 3. Build sub-graph with anchors.
    let band = BandGraph::build(graph, labels, &in_band);
    if band.sub.nvtxs <= 2 {
        // No band-internal vertices to refine (only anchors).
        return cut_size(graph, labels);
    }

    // 4. Refine the band via local FM. Anchors are locked.
    let mut sub_labels = band.labels.clone();
    refine_band(
        &band.sub,
        &mut sub_labels,
        band.anchor_a,
        band.anchor_b,
        max_imbalance,
        max_passes,
    );

    // 5. Project labels back. Out-of-band vertices keep their old
    //    labels; in-band vertices take the sub-graph's verdict.
    // `orig_of_sub` is sub-indexed and yields the original-graph
    // vertex, so `enumerate()` gives `(sub_index, orig_vertex)`.
    for (sub_i, &orig_v) in band.orig_of_sub.iter().enumerate() {
        if sub_i == band.anchor_a as usize || sub_i == band.anchor_b as usize {
            continue;
        }
        labels[orig_v as usize] = sub_labels[sub_i];
    }

    cut_size(graph, labels)
}

/// All vertices with at least one neighbour on the opposite side.
fn boundary_vertices(graph: &Graph, labels: &[u8]) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    for v in 0..graph.nvtxs {
        let lv = labels[v as usize];
        for &u in graph.neighbors(v) {
            if labels[u as usize] != lv {
                out.push(v);
                break;
            }
        }
    }
    out
}

/// BFS from `start` set to depth `width`. Returns a bitmap of
/// in-band vertices.
fn bfs_band(graph: &Graph, start: &[i32], width: u32) -> Vec<bool> {
    let n = graph.nvtxs as usize;
    let mut in_band: Vec<bool> = vec![false; n];
    let mut depth: Vec<u32> = vec![u32::MAX; n];
    let mut q: VecDeque<i32> = VecDeque::new();
    let mut sorted_start = start.to_vec();
    sorted_start.sort();
    for &v in &sorted_start {
        in_band[v as usize] = true;
        depth[v as usize] = 0;
        q.push_back(v);
    }
    while let Some(v) = q.pop_front() {
        let d = depth[v as usize];
        if d >= width {
            continue;
        }
        for &u in graph.neighbors(v) {
            let uu = u as usize;
            if !in_band[uu] {
                in_band[uu] = true;
                depth[uu] = d + 1;
                q.push_back(u);
            }
        }
    }
    in_band
}

/// Sub-graph extracted from `graph` covering all in-band vertices
/// plus two anchor supervertices.
struct BandGraph {
    /// Sub-graph in CSR. Last two vertices are anchors:
    /// `anchor_a = nvtxs - 2`, `anchor_b = nvtxs - 1`.
    sub: Graph,
    /// `orig_of_sub[i]` = original-graph vertex ID for sub vertex `i`,
    /// or `i32::MAX` if `i` is an anchor.
    orig_of_sub: Vec<i32>,
    /// Initial sub-graph labels (band vertices keep their original
    /// label; anchors get their fixed side).
    labels: Vec<u8>,
    /// Sub-graph index of `anchor_a`, `anchor_b`.
    anchor_a: i32,
    anchor_b: i32,
}

impl BandGraph {
    fn build(graph: &Graph, labels: &[u8], in_band: &[bool]) -> Self {
        let n = graph.nvtxs as usize;
        // Map original -> sub for in-band vertices.
        let mut sub_of_orig: Vec<i32> = vec![-1; n];
        let mut orig_of_sub: Vec<i32> = Vec::new();
        for v in 0..n {
            if in_band[v] {
                sub_of_orig[v] = orig_of_sub.len() as i32;
                orig_of_sub.push(v as i32);
            }
        }
        let n_band = orig_of_sub.len();
        let anchor_a = n_band as i32;
        let anchor_b = (n_band + 1) as i32;
        // Reserve anchor slots in orig_of_sub.
        orig_of_sub.push(i32::MAX);
        orig_of_sub.push(i32::MAX);

        // Adjacency: for each sub vertex collect neighbours and
        // accumulate edge to anchors for any out-of-band crossing.
        // We also accumulate anchor weight.
        let mut adj: Vec<Vec<(i32, i32)>> = vec![Vec::new(); n_band + 2];
        let mut vwgt: Vec<i32> = vec![0; n_band + 2];
        let mut sub_labels: Vec<u8> = vec![PART_A; n_band + 2];
        sub_labels[anchor_a as usize] = PART_A;
        sub_labels[anchor_b as usize] = PART_B;

        // anchor weight = sum of out-of-band vwgt per side.
        let mut anchor_a_w: i64 = 0;
        let mut anchor_b_w: i64 = 0;
        for v in 0..n {
            if !in_band[v] {
                let w = graph.vwgt[v] as i64;
                if labels[v] == PART_A {
                    anchor_a_w += w;
                } else if labels[v] == PART_B {
                    anchor_b_w += w;
                }
            }
        }
        // Saturate at i32::MAX (the band sub-graph uses i32 vwgt).
        vwgt[anchor_a as usize] = anchor_a_w.min(i32::MAX as i64) as i32;
        vwgt[anchor_b as usize] = anchor_b_w.min(i32::MAX as i64) as i32;

        // For each in-band vertex, walk its neighbours.
        for orig_v in 0..n {
            let sv = sub_of_orig[orig_v];
            if sv < 0 {
                continue;
            }
            vwgt[sv as usize] = graph.vwgt[orig_v];
            sub_labels[sv as usize] = labels[orig_v];
            // Aggregate by destination sub-vertex.
            let mut to_anchor_a: i32 = 0;
            let mut to_anchor_b: i32 = 0;
            let lo = graph.xadj[orig_v] as usize;
            let hi = graph.xadj[orig_v + 1] as usize;
            for k in lo..hi {
                let u = graph.adjncy[k] as usize;
                let w = graph.adjwgt[k];
                if let Some(&su) = sub_of_orig.get(u) {
                    if su >= 0 {
                        adj[sv as usize].push((su, w));
                        continue;
                    }
                }
                // u is out-of-band: route to its anchor.
                if labels[u] == PART_A {
                    to_anchor_a = to_anchor_a.saturating_add(w);
                } else if labels[u] == PART_B {
                    to_anchor_b = to_anchor_b.saturating_add(w);
                }
            }
            if to_anchor_a > 0 {
                adj[sv as usize].push((anchor_a, to_anchor_a));
                adj[anchor_a as usize].push((sv, to_anchor_a));
            }
            if to_anchor_b > 0 {
                adj[sv as usize].push((anchor_b, to_anchor_b));
                adj[anchor_b as usize].push((sv, to_anchor_b));
            }
        }

        // Compact into Graph CSR.
        let nvtxs_sub = (n_band + 2) as i32;
        let mut xadj: Vec<i32> = Vec::with_capacity(n_band + 3);
        let mut adjncy: Vec<i32> = Vec::new();
        let mut adjwgt: Vec<i32> = Vec::new();
        xadj.push(0);
        for adj_sv in adj.iter().take(n_band + 2) {
            for &(u, w) in adj_sv {
                adjncy.push(u);
                adjwgt.push(w);
            }
            xadj.push(adjncy.len() as i32);
        }
        let sub = Graph {
            nvtxs: nvtxs_sub,
            xadj,
            adjncy,
            vwgt,
            adjwgt,
        };
        BandGraph {
            sub,
            orig_of_sub,
            labels: sub_labels,
            anchor_a,
            anchor_b,
        }
    }
}

/// Inner FM pass for the band sub-graph. Same gain accounting as
/// halo_fm but with the two anchors permanently locked.
///
/// Anchors have correct vwgt so balance checks see the full
/// out-of-band mass. This is what fixes the balance-accounting bug
/// flagged in audit finding 2.
fn refine_band(
    sub: &Graph,
    labels: &mut [u8],
    anchor_a: i32,
    anchor_b: i32,
    max_imbalance: f64,
    max_passes: u32,
) {
    let n = sub.nvtxs as usize;
    let total: i64 = sub.vwgt.iter().map(|&w| w as i64).sum();
    let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;

    let mut pass_cut = cut_size(sub, labels);

    for _pass in 0..max_passes {
        let before_pass = pass_cut;
        let mut gain: Vec<i32> = vec![0; n];
        compute_gains(sub, labels, &mut gain);
        let mut locked: Vec<bool> = vec![false; n];
        // Pin anchors.
        locked[anchor_a as usize] = true;
        locked[anchor_b as usize] = true;

        let mut heap: BinaryHeap<(i32, Reverse<i32>, i32)> = BinaryHeap::new();
        for (v, &g) in gain.iter().enumerate().take(n) {
            if !locked[v] {
                heap.push((g, Reverse(v as i32), g));
            }
        }

        let mut cur_cut = pass_cut;
        let mut moves: Vec<i32> = Vec::new();
        let mut a_w = part_weight(sub, labels, PART_A);
        let mut b_w = total - a_w;
        // See rslab_metis::fm_refine::refine_bisection for rationale:
        // best_prefix=None is the "no balanced state seen yet"
        // sentinel for imbalanced pass-starts.
        let starts_balanced = a_w.max(b_w) <= max_side;
        let mut best_prefix: Option<usize> = if starts_balanced { Some(0) } else { None };
        let mut best_cut = pass_cut;
        let mut best_a_w = a_w;
        let mut best_b_w = b_w;
        let mut no_improve: u32 = 0;

        while let Some((_, Reverse(v), stamp)) = heap.pop() {
            let vu = v as usize;
            if locked[vu] {
                continue;
            }
            if stamp != gain[vu] {
                continue;
            }
            let from = labels[vu];
            let to = if from == PART_A { PART_B } else { PART_A };
            let (new_a_w, new_b_w) = if to == PART_A {
                (a_w + sub.vwgt[vu] as i64, b_w - sub.vwgt[vu] as i64)
            } else {
                (a_w - sub.vwgt[vu] as i64, b_w + sub.vwgt[vu] as i64)
            };
            let side_max = new_a_w.max(new_b_w);

            labels[vu] = to;
            a_w = new_a_w;
            b_w = new_b_w;
            cur_cut -= gain[vu];
            locked[vu] = true;
            moves.push(v);

            // Update neighbour gains using the *correct* sign
            // convention for gain = ed - id.
            let lo = sub.xadj[vu] as usize;
            let hi = sub.xadj[vu + 1] as usize;
            for k in lo..hi {
                let u = sub.adjncy[k] as usize;
                if locked[u] {
                    continue;
                }
                let w = sub.adjwgt[k];
                if labels[u] == from {
                    gain[u] += 2 * w;
                } else {
                    gain[u] -= 2 * w;
                }
                heap.push((gain[u], Reverse(u as i32), gain[u]));
            }

            if side_max <= max_side {
                let is_first_balanced = best_prefix.is_none();
                if is_first_balanced || cur_cut < best_cut {
                    best_cut = cur_cut;
                    best_prefix = Some(moves.len());
                    best_a_w = a_w;
                    best_b_w = b_w;
                    no_improve = 0;
                } else {
                    no_improve += 1;
                }
            } else {
                no_improve += 1;
            }
            if no_improve >= 50 {
                break;
            }
        }

        match best_prefix {
            Some(prefix) => {
                for &v in moves.iter().skip(prefix) {
                    let vu = v as usize;
                    labels[vu] = if labels[vu] == PART_A { PART_B } else { PART_A };
                }
                pass_cut = best_cut;
            }
            None => {
                // No balanced state reached; keep the move sequence.
                pass_cut = cur_cut;
            }
        }
        let _ = (best_a_w, best_b_w);
        a_w = part_weight(sub, labels, PART_A);
        b_w = total - a_w;
        let _ = (a_w, b_w);
        if pass_cut == before_pass {
            break;
        }
    }
}

fn compute_gains(graph: &Graph, labels: &[u8], gain: &mut [i32]) {
    for (v, &lv) in labels.iter().enumerate() {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        let mut ed: i32 = 0;
        let mut id: i32 = 0;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            let w = graph.adjwgt[k];
            if labels[u] == lv {
                id = id.saturating_add(w);
            } else {
                ed = ed.saturating_add(w);
            }
        }
        gain[v] = ed - id;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::csc_from_edges;
    use rslab_ordering_core::CscPattern;

    fn build(n: usize, edges: &[(usize, usize)]) -> Graph {
        let (cp, ri) = csc_from_edges(n, edges);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    fn grid(m: usize, n: usize) -> Graph {
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let idx = |r: usize, c: usize| r * n + c;
        for r in 0..m {
            for c in 0..n {
                if r + 1 < m {
                    edges.push((idx(r, c), idx(r + 1, c)));
                }
                if c + 1 < n {
                    edges.push((idx(r, c), idx(r, c + 1)));
                }
            }
        }
        build(m * n, &edges)
    }

    #[test]
    fn band_extraction_anchor_weights_sum_correctly() {
        let g = grid(5, 5);
        let labels: Vec<u8> = (0..25u8)
            .map(|k| if (k as usize / 5) < 2 { PART_A } else { PART_B })
            .collect();
        // Boundary: rows 1 and 2 (rows 1-2 transition). With width=1
        // band includes rows 0,1,2,3 → out-of-band = row 4 only.
        let bnd = boundary_vertices(&g, &labels);
        let in_band = bfs_band(&g, &bnd, 1);
        let band = BandGraph::build(&g, &labels, &in_band);
        let n_in_band: usize = in_band.iter().filter(|&&b| b).count();
        let n_out: usize = 25 - n_in_band;
        let out_a: i64 = (0..25)
            .filter(|&v| !in_band[v] && labels[v] == PART_A)
            .map(|v| g.vwgt[v] as i64)
            .sum();
        let out_b: i64 = (0..25)
            .filter(|&v| !in_band[v] && labels[v] == PART_B)
            .map(|v| g.vwgt[v] as i64)
            .sum();
        assert_eq!(out_a + out_b, n_out as i64);
        assert_eq!(band.sub.vwgt[band.anchor_a as usize] as i64, out_a);
        assert_eq!(band.sub.vwgt[band.anchor_b as usize] as i64, out_b);
    }

    #[test]
    fn cut_never_grows_on_grid() {
        let g = grid(6, 6);
        let mut labels: Vec<u8> = (0..36u8)
            .map(|k| if (k as usize / 6) < 3 { PART_A } else { PART_B })
            .collect();
        let cut_before = cut_size(&g, &labels);
        let cut_after = band_fm_refine(&g, &mut labels, 2, 0.10, 16);
        // I1 (bookkeeping consistency).
        assert_eq!(cut_after, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(
            cut_after <= cut_before,
            "band FM grew the cut: {} -> {}",
            cut_before,
            cut_after
        );
    }

    #[test]
    fn out_of_band_labels_preserved() {
        let g = grid(7, 7);
        let labels_init: Vec<u8> = (0..49u8)
            .map(|k| if (k as usize / 7) < 3 { PART_A } else { PART_B })
            .collect();
        let mut labels = labels_init.clone();
        // Width 1 - band stays close to the row 2/3 boundary.
        let returned = band_fm_refine(&g, &mut labels, 1, 0.10, 8);
        // I1 - band FM also reports a cut; it must match.
        assert_eq!(returned, cut_size(&g, &labels), "I1: bookkeeping");
        // Out-of-band: rows 5 and 6 (and row 0). With width=1, band
        // is rows 1, 2, 3, 4. Verify rows 0, 5, 6 unchanged.
        for r in [0usize, 5, 6] {
            for c in 0..7 {
                let v = r * 7 + c;
                assert_eq!(
                    labels[v], labels_init[v],
                    "row {} col {} (out-of-band) was changed",
                    r, c
                );
            }
        }
    }

    #[test]
    fn determinism() {
        let g = grid(6, 6);
        let init: Vec<u8> = (0..36u8)
            .map(|k| if (k as usize / 6) < 3 { PART_A } else { PART_B })
            .collect();
        let mut a = init.clone();
        let mut b = init.clone();
        let ca = band_fm_refine(&g, &mut a, 2, 0.05, 8);
        let cb = band_fm_refine(&g, &mut b, 2, 0.05, 8);
        assert_eq!(a, b);
        assert_eq!(ca, cb);
        assert_eq!(ca, cut_size(&g, &a), "I1: bookkeeping (run a)");
        assert_eq!(cb, cut_size(&g, &b), "I1: bookkeeping (run b)");
    }

    #[test]
    fn empty_graph() {
        let cp: Vec<i32> = vec![0];
        let ri: Vec<i32> = vec![];
        let pat = CscPattern::new(0, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut labels: Vec<u8> = vec![];
        let c = band_fm_refine(&g, &mut labels, 3, 0.10, 8);
        assert_eq!(c, 0);
        assert_eq!(c, cut_size(&g, &labels), "I1: bookkeeping");
    }

    #[test]
    fn no_boundary_no_op() {
        // Two disjoint K_3 with each on its own side → no boundary,
        // band FM is a no-op.
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let g = build(6, &edges);
        let mut labels: Vec<u8> = vec![PART_A, PART_A, PART_A, PART_B, PART_B, PART_B];
        let c = band_fm_refine(&g, &mut labels, 3, 0.10, 8);
        assert_eq!(c, 0);
        assert_eq!(c, cut_size(&g, &labels), "I1: bookkeeping");
        assert_eq!(labels, vec![PART_A, PART_A, PART_A, PART_B, PART_B, PART_B]);
    }
}
