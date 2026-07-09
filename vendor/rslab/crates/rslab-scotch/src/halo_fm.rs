//! Halo FM: edge-bisection FM with a one-hop dynamic halo.
//!
//! Standard boundary FM (`rslab_metis::internals::fm_refine::refine_bisection`)
//! considers only vertices adjacent to the opposite side as move
//! candidates. Halo FM widens the candidate set to include vertices
//! one hop *off* the boundary. The halo is **dynamic** (audit
//! finding 3 of `dev/plans/ordering-scotch.md`): every move that
//! changes the boundary is propagated to the halo set within the
//! same pass.
//!
//! Clean-room from Pellegrini 1996 §3 and the research note
//! `dev/research/scotch-halo-fm.md`. No code is paraphrased from
//! SCOTCH's `bgraph_bipart_fm.c`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rslab_metis::internals::graph::Graph;
use rslab_metis::internals::initial_partition::{cut_size, part_weight, PART_A, PART_B};

/// Refine an edge bisection with halo-extended FM.
///
/// `labels[v] ∈ {PART_A, PART_B}` on entry and exit. Returns the
/// final edge cut. The candidate set on each pass is
/// `boundary ∪ halo` where `halo = {v ∉ boundary : ∃ u ∈ N(v) with
/// u ∈ boundary}`. Both sets are updated incrementally as moves
/// commit.
pub fn halo_fm_refine(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    max_passes: u32,
) -> i32 {
    let n = graph.nvtxs as usize;
    if n < 2 {
        return cut_size(graph, labels);
    }
    let total: i64 = graph.vwgt.iter().map(|&w| w as i64).sum();
    let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;

    let mut pass_cut = cut_size(graph, labels);

    for _pass in 0..max_passes {
        let before_pass = pass_cut;

        // Per-pass scratch.
        let mut boundary_cnt: Vec<i32> = vec![0; n];
        let mut halo_cnt: Vec<i32> = vec![0; n];
        let mut gain: Vec<i32> = vec![0; n];
        compute_pass_state(graph, labels, &mut boundary_cnt, &mut halo_cnt, &mut gain);

        let mut locked: Vec<bool> = vec![false; n];
        let mut heap: BinaryHeap<(i32, Reverse<i32>, i32)> = BinaryHeap::new();
        for (v, &g) in gain.iter().enumerate().take(n) {
            if is_candidate(v, &boundary_cnt, &halo_cnt) {
                heap.push((g, Reverse(v as i32), g));
            }
        }

        let mut cur_cut = pass_cut;
        let mut moves: Vec<i32> = Vec::new();
        let mut a_w = part_weight(graph, labels, PART_A);
        let mut b_w = total - a_w;
        // Mirrors rslab_metis::fm_refine::refine_bisection: an
        // imbalanced pass-start has no valid rollback target at
        // prefix 0. best_prefix=None means "no balanced state seen
        // yet"; the first balanced state encountered is
        // unconditionally recorded.
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
            // The candidate may have left the candidate set since it
            // was queued (e.g. a neighbour move pulled it off the
            // boundary and out of the halo).
            if !is_candidate(vu, &boundary_cnt, &halo_cnt) {
                continue;
            }
            let from = labels[vu];
            let to = if from == PART_A { PART_B } else { PART_A };
            let (new_a_w, new_b_w) = if to == PART_A {
                (a_w + graph.vwgt[vu] as i64, b_w - graph.vwgt[vu] as i64)
            } else {
                (a_w - graph.vwgt[vu] as i64, b_w + graph.vwgt[vu] as i64)
            };
            let side_max = new_a_w.max(new_b_w);

            // Always commit (even if balance fails) so FM can climb
            // out of poor configurations; only consider for `best` if
            // the resulting state is balanced.
            labels[vu] = to;
            a_w = new_a_w;
            b_w = new_b_w;
            cur_cut -= gain[vu];
            locked[vu] = true;
            moves.push(v);

            // Update boundary_cnt, halo_cnt, gain for v's neighbours,
            // and v itself.
            update_after_move(
                graph,
                labels,
                vu,
                from,
                to,
                &mut boundary_cnt,
                &mut halo_cnt,
                &mut gain,
                &locked,
                &mut heap,
            );

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
                a_w = best_a_w;
                b_w = best_b_w;
                pass_cut = best_cut;
            }
            None => {
                // No balanced state reached this pass; keep the full
                // move sequence so the next pass starts closer to
                // balance instead of rolling back to the violating
                // start.
                pass_cut = cur_cut;
            }
        }
        let _ = (a_w, b_w);
        if pass_cut == before_pass {
            break;
        }
    }
    pass_cut
}

/// `v` is a candidate iff `v` is on the boundary or in the one-hop halo.
fn is_candidate(v: usize, boundary_cnt: &[i32], halo_cnt: &[i32]) -> bool {
    boundary_cnt[v] > 0 || (boundary_cnt[v] == 0 && halo_cnt[v] > 0)
}

/// Per-pass state: opposite-side neighbour weight (`boundary_cnt`,
/// counted in unweighted form), boundary neighbour count (`halo_cnt`),
/// and standard FM gain.
fn compute_pass_state(
    graph: &Graph,
    labels: &[u8],
    boundary_cnt: &mut [i32],
    halo_cnt: &mut [i32],
    gain: &mut [i32],
) {
    let n = graph.nvtxs as usize;
    // First pass: gain and boundary_cnt.
    for v in 0..n {
        let lv = labels[v];
        let mut ed: i32 = 0;
        let mut id: i32 = 0;
        let mut bcnt: i32 = 0;
        for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
            let u = graph.adjncy[k] as usize;
            let w = graph.adjwgt[k];
            if labels[u] == lv {
                id = id.saturating_add(w);
            } else {
                ed = ed.saturating_add(w);
                bcnt += 1;
            }
        }
        gain[v] = ed - id;
        boundary_cnt[v] = bcnt;
    }
    // Second pass: halo_cnt = number of neighbours that ARE on the
    // boundary (boundary_cnt[u] > 0). A vertex is *in* the halo only
    // if it itself is *not* on the boundary.
    for (v, hslot) in halo_cnt.iter_mut().enumerate().take(n) {
        let mut hcnt: i32 = 0;
        for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
            let u = graph.adjncy[k] as usize;
            if boundary_cnt[u] > 0 {
                hcnt += 1;
            }
        }
        *hslot = hcnt;
    }
}

/// After moving `v` from `from` to `to`, update neighbour boundary
/// counts, halo counts, and gains. Push newly-eligible candidates
/// into the heap.
#[allow(clippy::too_many_arguments)]
fn update_after_move(
    graph: &Graph,
    labels: &[u8],
    vu: usize,
    from: u8,
    to: u8,
    boundary_cnt: &mut [i32],
    halo_cnt: &mut [i32],
    gain: &mut [i32],
    locked: &[bool],
    heap: &mut BinaryHeap<(i32, Reverse<i32>, i32)>,
) {
    // Snapshot v's neighbours.
    let nbrs: Vec<i32> = graph.neighbors(vu as i32).to_vec();
    let weights: Vec<i32> = graph.edge_weights(vu as i32).to_vec();

    // Set we'll re-evaluate halo_cnt for: neighbours of any vertex
    // whose boundary_cnt crossed the 0 threshold this move.
    let mut boundary_changed: Vec<usize> = Vec::new();

    // Recompute boundary_cnt[vu] from scratch given the new label.
    let mut bcnt_v: i32 = 0;
    for &u in &nbrs {
        if labels[u as usize] != to {
            bcnt_v += 1;
        }
    }
    let prev_bcnt_v = boundary_cnt[vu];
    boundary_cnt[vu] = bcnt_v;
    if (prev_bcnt_v > 0) != (bcnt_v > 0) {
        boundary_changed.push(vu);
    }
    // Recompute gain[vu]: now it reflects flipping back to `from`.
    gain[vu] = -gain[vu];

    // For each neighbour u, update gain and boundary_cnt deltas.
    for (k, &u) in nbrs.iter().enumerate() {
        let uu = u as usize;
        let w = weights[k];
        let lu = labels[uu];
        let prev_b = boundary_cnt[uu];
        if lu == from {
            // u stayed on `from`. Edge (u,v): was same-side for u
            // (in id[u]), now crosses (in ed[u]). With gain = ed - id,
            // Δgain[u] = +w - (-w) = +2w.
            boundary_cnt[uu] += 1;
            if !locked[uu] {
                gain[uu] += 2 * w;
            }
        } else if lu == to {
            // u stayed on `to`. Edge (u,v): was crossing (in ed[u]),
            // now same-side (in id[u]). Δgain[u] = -2w.
            boundary_cnt[uu] -= 1;
            if !locked[uu] {
                gain[uu] -= 2 * w;
            }
        }
        // (lu can't be PART_SEP since we're in pure edge-bisection space.)
        if (prev_b > 0) != (boundary_cnt[uu] > 0) {
            boundary_changed.push(uu);
        }
    }

    // Halo update: for every vertex w whose boundary status flipped
    // (entered or left the boundary), recompute halo_cnt for w's
    // neighbours.
    for &x in &boundary_changed {
        // Collect neighbours of x (snapshot).
        let nx: Vec<i32> = graph.neighbors(x as i32).to_vec();
        for &y in &nx {
            let yy = y as usize;
            // Recompute halo_cnt[yy] from scratch - small fanout in
            // typical graphs, and avoids tracking signed deltas
            // through arbitrary boundary transitions.
            let mut h: i32 = 0;
            for &z in graph.neighbors(yy as i32) {
                if boundary_cnt[z as usize] > 0 {
                    h += 1;
                }
            }
            halo_cnt[yy] = h;
        }
    }

    // Re-stamp gains for unlocked neighbours and push if candidate.
    for &u in &nbrs {
        let uu = u as usize;
        if !locked[uu] && is_candidate(uu, boundary_cnt, halo_cnt) {
            heap.push((gain[uu], Reverse(u), gain[uu]));
        }
    }
    // Also push any halo-newly-eligible vertices among the
    // changed-boundary set's neighbours - they may not be in `nbrs`.
    for &x in &boundary_changed {
        for &y in graph.neighbors(x as i32) {
            let yy = y as usize;
            if !locked[yy] && is_candidate(yy, boundary_cnt, halo_cnt) {
                heap.push((gain[yy], Reverse(y), gain[yy]));
            }
        }
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

    fn balanced(graph: &Graph, labels: &[u8], max_imbalance: f64) -> bool {
        let total: i64 = graph.vwgt.iter().map(|&w| w as i64).sum();
        let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;
        let a = part_weight(graph, labels, PART_A);
        let b = part_weight(graph, labels, PART_B);
        a.max(b) <= max_side
    }

    #[test]
    fn cut_never_grows_on_grid() {
        let g = grid(6, 6);
        // Hand-perturbed bisection: alternate 18 vs 18.
        let mut labels: Vec<u8> = (0..36u8)
            .map(|k| if (k as usize % 6) < 3 { PART_A } else { PART_B })
            .collect();
        let cut_before = cut_size(&g, &labels);
        let cut_after = halo_fm_refine(&g, &mut labels, 0.05, 32);
        // I1 (bookkeeping consistency): returned cut matches labels.
        assert_eq!(cut_after, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(
            cut_after <= cut_before,
            "halo FM grew the cut: {} -> {}",
            cut_before,
            cut_after
        );
        assert!(balanced(&g, &labels, 0.10));
    }

    #[test]
    fn matches_optimal_on_4x4_grid() {
        // 4x4 grid: optimal half-half cut is 4 (one row or column).
        let g = grid(4, 4);
        let mut labels: Vec<u8> = (0..16u8)
            .map(|k| if (k as usize % 4) < 2 { PART_A } else { PART_B })
            .collect();
        let cut = halo_fm_refine(&g, &mut labels, 0.05, 32);
        assert_eq!(cut, cut_size(&g, &labels), "I1: bookkeeping");
        assert!(cut <= 4, "expected cut <= 4 on 4x4, got {}", cut);
    }

    #[test]
    fn determinism() {
        let g = grid(5, 5);
        let init: Vec<u8> = (0..25u8)
            .map(|k| if (k as usize / 5) < 3 { PART_A } else { PART_B })
            .collect();
        let mut a = init.clone();
        let mut b = init.clone();
        let ca = halo_fm_refine(&g, &mut a, 0.10, 16);
        let cb = halo_fm_refine(&g, &mut b, 0.10, 16);
        assert_eq!(a, b);
        assert_eq!(ca, cb);
        // I1 on both runs.
        assert_eq!(ca, cut_size(&g, &a), "I1: bookkeeping (run a)");
        assert_eq!(cb, cut_size(&g, &b), "I1: bookkeeping (run b)");
    }

    #[test]
    fn empty_returns_zero() {
        let cp: Vec<i32> = vec![0];
        let ri: Vec<i32> = vec![];
        let pat = CscPattern::new(0, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut labels: Vec<u8> = vec![];
        let c = halo_fm_refine(&g, &mut labels, 0.05, 8);
        assert_eq!(c, 0);
        assert_eq!(c, cut_size(&g, &labels), "I1: bookkeeping");
    }

    #[test]
    fn already_optimal_zero_cut() {
        // Two disjoint K_3, each side gets one. No crossing edges.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        edges.extend([(0, 1), (1, 2), (0, 2)]);
        edges.extend([(3, 4), (4, 5), (3, 5)]);
        let g = build(6, &edges);
        let mut labels: Vec<u8> = vec![PART_A, PART_A, PART_A, PART_B, PART_B, PART_B];
        let c = halo_fm_refine(&g, &mut labels, 0.05, 8);
        assert_eq!(c, 0);
        assert_eq!(c, cut_size(&g, &labels), "I1: bookkeeping");
        // Verify partition wasn't shuffled.
        assert_eq!(labels, vec![PART_A, PART_A, PART_A, PART_B, PART_B, PART_B]);
    }

    #[test]
    fn no_better_than_boundary_on_path() {
        // Path graph: optimal split is the middle edge, cut = 1.
        let edges: Vec<(usize, usize)> = (0..9).map(|i| (i, i + 1)).collect();
        let g = build(10, &edges);
        let mut labels: Vec<u8> = (0..10u8)
            .map(|k| if k < 5 { PART_A } else { PART_B })
            .collect();
        let c = halo_fm_refine(&g, &mut labels, 0.05, 8);
        assert_eq!(c, cut_size(&g, &labels), "I1: bookkeeping");
        assert_eq!(c, 1, "path with balanced split has cut 1");
    }
}
