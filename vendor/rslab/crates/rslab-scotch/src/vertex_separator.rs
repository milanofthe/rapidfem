//! Direct vertex-separator computation via two-sided Fiduccia-Mattheyses.
//!
//! Clean-room reconstruction of the algorithm in Pellegrini, "SCOTCH:
//! A software package for static mapping..." (HPCN Europe, 1996), §3.
//! No code is paraphrased from SCOTCH's `vgraph_separate_fm.c`
//! (CeCILL-C). The corrected gain formula, the incremental
//! `frontier_load` arrays, the per-move *and* final-prefix imbalance
//! checks, and the lighter-side initial separator construction follow
//! the audit findings in `dev/plans/ordering-scotch.md` (findings 1,
//! 6, 7) and the research note `dev/research/scotch-vertex-separator.md`.
//!
//! ## Difference from rslab-metis
//!
//! `rslab_metis::internals::separator::construct_separator` derives a
//! node separator *post-hoc* from an edge bisection by computing a
//! minimum vertex cover of the boundary bipartite graph (König's
//! theorem). That route minimises a tight upper bound on the
//! separator size, but the separator weight itself is not the
//! optimisation target.
//!
//! This module instead treats *separator weight* as the primary
//! objective. Starting from the boundary of the lighter side (a
//! trivially-valid but non-minimal separator), two-sided FM moves
//! shrink the separator while keeping the surviving sides balanced.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// One committed FM move within a pass: `(vertex moved out of S,
/// destination side, list of (neighbour, prev_label) pulled into S
/// because they were on the opposite side)`. Used for rollback when a
/// pass overshoots the best-balanced prefix.
type Move = (i32, u8, Vec<(i32, u8)>);

use rslab_metis::internals::fm_refine::PART_SEP;
use rslab_metis::internals::graph::Graph;
use rslab_metis::internals::initial_partition::{part_weight, PART_A, PART_B};

/// Compute a node separator from an edge bisection by direct
/// two-sided FM.
///
/// `labels[v] ∈ {PART_A, PART_B}` on entry; on exit `labels[v] ∈
/// {PART_A, PART_B, PART_SEP}`. The post-call invariants are:
///
/// * No edge connects a `PART_A` vertex directly to a `PART_B` vertex.
/// * `max(part_weight(A), part_weight(B)) <= (1 + max_imbalance) *
///   total / 2`, where `total = Σ vwgt`.
///
/// `move_cap` bounds moves per pass; `pass_cap` bounds passes. Both
/// match SCOTCH's per-call FM defaults (200, 32).
///
/// Returns the total vertex weight assigned to the separator.
pub fn compute_vertex_separator(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    move_cap: u32,
    pass_cap: u32,
) -> i64 {
    let n = graph.nvtxs as usize;
    debug_assert_eq!(labels.len(), n);
    if n == 0 {
        return 0;
    }

    let total: i64 = graph.vwgt.iter().map(|&w| w as i64).sum();
    let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;

    initial_separator(graph, labels);

    let mut load_a: Vec<i64> = vec![0; n];
    let mut load_b: Vec<i64> = vec![0; n];
    recompute_loads(graph, labels, &mut load_a, &mut load_b);

    let mut sep_w = sep_weight(graph, labels);
    if sep_w == 0 {
        return 0;
    }
    let mut a_w = part_weight(graph, labels, PART_A);
    let mut b_w = part_weight(graph, labels, PART_B);

    for _ in 0..pass_cap {
        let before = sep_w;
        let (new_sep_w, new_a_w, new_b_w) = fm_pass(
            graph,
            labels,
            &mut load_a,
            &mut load_b,
            sep_w,
            a_w,
            b_w,
            max_side,
            move_cap,
        );
        sep_w = new_sep_w;
        a_w = new_a_w;
        b_w = new_b_w;
        if sep_w >= before {
            break;
        }
    }

    sep_w
}

/// Build an initial separator from the boundary of the lighter side.
fn initial_separator(graph: &Graph, labels: &mut [u8]) {
    let n = graph.nvtxs as usize;
    let total_a = part_weight(graph, labels, PART_A);
    let total_b = part_weight(graph, labels, PART_B);
    let lighter = if total_a <= total_b { PART_A } else { PART_B };
    let other = if lighter == PART_A { PART_B } else { PART_A };

    // Mark every lighter-side vertex with at least one neighbour in
    // `other` as separator. Two-pass to avoid mutating labels mid-scan.
    let mut to_sep: Vec<u8> = vec![0; n];
    for v in 0..n {
        if labels[v] != lighter {
            continue;
        }
        for &u in graph.neighbors(v as i32) {
            if labels[u as usize] == other {
                to_sep[v] = 1;
                break;
            }
        }
    }
    for v in 0..n {
        if to_sep[v] == 1 {
            labels[v] = PART_SEP;
        }
    }
}

/// Recompute `load_a[v]` and `load_b[v]` for every separator vertex.
/// Non-separator vertices keep zero entries (unused).
fn recompute_loads(graph: &Graph, labels: &[u8], load_a: &mut [i64], load_b: &mut [i64]) {
    let n = graph.nvtxs as usize;
    for v in 0..n {
        load_a[v] = 0;
        load_b[v] = 0;
        if labels[v] != PART_SEP {
            continue;
        }
        for &u in graph.neighbors(v as i32) {
            let lu = labels[u as usize];
            let w = graph.vwgt[u as usize] as i64;
            if lu == PART_A {
                load_a[v] += w;
            } else if lu == PART_B {
                load_b[v] += w;
            }
        }
    }
}

/// Total weight of separator vertices.
fn sep_weight(graph: &Graph, labels: &[u8]) -> i64 {
    let mut s: i64 = 0;
    for (v, &l) in labels.iter().enumerate() {
        if l == PART_SEP {
            s += graph.vwgt[v] as i64;
        }
    }
    s
}

/// Gain of moving separator vertex `v` to `side`:
/// `vwgt[v] - load_other[v]`.
///
/// Audit-corrected: `load_other` is the load on the *opposite* side,
/// because those are the neighbours that must enter the separator.
fn gain_to_side(v: usize, side: u8, vwgt: &[i32], load_a: &[i64], load_b: &[i64]) -> i64 {
    let opp_load = match side {
        PART_A => load_b[v],
        PART_B => load_a[v],
        _ => 0,
    };
    vwgt[v] as i64 - opp_load
}

/// Single FM pass. Returns `(sep_w, a_w, b_w)` rolled back to the
/// best balanced prefix.
#[allow(clippy::too_many_arguments)]
fn fm_pass(
    graph: &Graph,
    labels: &mut [u8],
    load_a: &mut [i64],
    load_b: &mut [i64],
    init_sep_w: i64,
    init_a_w: i64,
    init_b_w: i64,
    max_side: i64,
    move_cap: u32,
) -> (i64, i64, i64) {
    let n = graph.nvtxs as usize;
    let mut locked: Vec<bool> = vec![false; n];

    // Two PQs: pq_to_a[v] = best gain of moving v from S to A, etc.
    // Stored as (gain, Reverse(v), stamp) - stamp lets us discard
    // stale entries cheaply.
    let mut pq_to_a: BinaryHeap<(i64, Reverse<i32>, i64)> = BinaryHeap::new();
    let mut pq_to_b: BinaryHeap<(i64, Reverse<i32>, i64)> = BinaryHeap::new();
    for v in 0..n {
        if labels[v] != PART_SEP {
            continue;
        }
        if load_a[v] > 0 {
            let g = gain_to_side(v, PART_A, &graph.vwgt, load_a, load_b);
            pq_to_a.push((g, Reverse(v as i32), g));
        }
        if load_b[v] > 0 {
            let g = gain_to_side(v, PART_B, &graph.vwgt, load_a, load_b);
            pq_to_b.push((g, Reverse(v as i32), g));
        }
    }

    let mut sep_w = init_sep_w;
    let mut a_w = init_a_w;
    let mut b_w = init_b_w;

    let mut best_sep_w = sep_w;
    let mut best_a_w = a_w;
    let mut best_b_w = b_w;
    let mut best_prefix: usize = 0;
    let mut moves: Vec<Move> = Vec::new();
    // (v, side_v_moved_to, list of (u, prev_label_of_u_pulled_into_S))

    let mut moves_done: u32 = 0;
    while moves_done < move_cap {
        // Choose the side that currently weighs less; if equal, prefer
        // the one whose PQ has a higher-gain head. Deterministic
        // breaking via choosing PART_A on a true tie.
        let pick_a_first = match (pq_to_a.peek(), pq_to_b.peek()) {
            (None, None) => break,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (Some(a), Some(b)) => {
                if a_w < b_w {
                    true
                } else if b_w < a_w {
                    false
                } else {
                    // equal weights: pick the higher-gain head, A on tie.
                    a.0 >= b.0
                }
            }
        };

        // Try the chosen side; if its move is rejected (imbalance,
        // stale, locked), we'll fall through and try the other side.
        let mut moved_this_iter = false;
        // An imbalance-rejected head is popped, not abandoned: feasible
        // lower-gain moves may still sit below it in either PQ, so a
        // rejection must NOT terminate the pass (O13). Only a genuinely
        // empty/stale frontier (no move and no rejection) ends it.
        let mut rejected_this_iter = false;
        for attempt in 0..2 {
            let pick_a = if attempt == 0 {
                pick_a_first
            } else {
                !pick_a_first
            };
            let pq = if pick_a { &mut pq_to_a } else { &mut pq_to_b };
            let side = if pick_a { PART_A } else { PART_B };
            // Drain stale/locked entries until we find a valid head.
            let mut chosen: Option<i32> = None;
            while let Some(&(_g, Reverse(v), stamp)) = pq.peek() {
                let vu = v as usize;
                if locked[vu] {
                    pq.pop();
                    continue;
                }
                if labels[vu] != PART_SEP {
                    pq.pop();
                    continue;
                }
                let cur = gain_to_side(vu, side, &graph.vwgt, load_a, load_b);
                if cur != stamp {
                    pq.pop();
                    // Re-push at current gain only if still relevant.
                    let still_relevant = match side {
                        PART_A => load_a[vu] > 0,
                        PART_B => load_b[vu] > 0,
                        _ => false,
                    };
                    if still_relevant {
                        pq.push((cur, Reverse(v), cur));
                    }
                    continue;
                }
                chosen = Some(v);
                break;
            }
            let v = match chosen {
                Some(v) => v,
                None => continue,
            };
            let vu = v as usize;
            // Per-move imbalance: would moving v + pulling its opposite
            // neighbours into S blow the destination cap?
            let dest_delta = graph.vwgt[vu] as i64;
            let (new_a_w, new_b_w) = if side == PART_A {
                (a_w + dest_delta, b_w)
            } else {
                (a_w, b_w + dest_delta)
            };
            // Pulling neighbours from the opposite side into S reduces
            // the opposite side's weight; that can only improve balance
            // on the destination side. So checking only the destination
            // is safe. But the pulled-in vertices reduce the *opposite*
            // weight, which may cause it to drop too low. We'll check
            // the post-move side weights for both bounds.
            let opp_load = if side == PART_A {
                load_b[vu]
            } else {
                load_a[vu]
            };
            let (post_a_w, post_b_w) = if side == PART_A {
                (new_a_w, new_b_w - opp_load)
            } else {
                (new_a_w - opp_load, new_b_w)
            };
            if post_a_w.max(post_b_w) > max_side {
                // Infeasible: pop this head and try the next candidate.
                // The pop drains the heap monotonically, so the pass
                // still terminates; but we must keep going this outer
                // iteration (and across iterations) because a feasible
                // lower-gain move may sit below this head - SCOTCH skips
                // the infeasible head and continues (O13). Record the
                // rejection so the post-loop break does not mistake "no
                // move" for "frontier exhausted". (Don't pop the other
                // side's peek - that's a separate, independent move.)
                let pq2 = if pick_a { &mut pq_to_a } else { &mut pq_to_b };
                pq2.pop();
                rejected_this_iter = true;
                continue;
            }

            // Commit the move. v leaves S into `side`; opposite-side
            // neighbours of v that aren't already in S get pulled in.
            let pq2 = if pick_a { &mut pq_to_a } else { &mut pq_to_b };
            pq2.pop();
            labels[vu] = side;
            locked[vu] = true;
            sep_w -= graph.vwgt[vu] as i64;
            if side == PART_A {
                a_w += graph.vwgt[vu] as i64;
            } else {
                b_w += graph.vwgt[vu] as i64;
            }

            // For each neighbour u:
            //   - if labels[u] == opp(side): pull u into S, lock it.
            //     Update its load_* arrays (need to inspect *its*
            //     neighbours).
            //   - if labels[u] == side: nothing.
            //   - if labels[u] == PART_SEP: u's load_side increased by
            //     vwgt[v] (v is now in side). We re-stamp u's PQ
            //     entries.
            let opposite = if side == PART_A { PART_B } else { PART_A };
            let mut pulled: Vec<(i32, u8)> = Vec::new();
            // Snapshot v's neighbours up front.
            let nbrs_v: Vec<i32> = graph.neighbors(v).to_vec();
            for &u in &nbrs_v {
                let uu = u as usize;
                let lu = labels[uu];
                if lu == opposite {
                    pulled.push((u, opposite));
                    labels[uu] = PART_SEP;
                    locked[uu] = true;
                    if side == PART_A {
                        b_w -= graph.vwgt[uu] as i64;
                    } else {
                        a_w -= graph.vwgt[uu] as i64;
                    }
                    sep_w += graph.vwgt[uu] as i64;
                    // Compute u's loads from scratch.
                    load_a[uu] = 0;
                    load_b[uu] = 0;
                    for &uu_n in graph.neighbors(u) {
                        match labels[uu_n as usize] {
                            PART_A => load_a[uu] += graph.vwgt[uu_n as usize] as i64,
                            PART_B => load_b[uu] += graph.vwgt[uu_n as usize] as i64,
                            _ => {}
                        }
                    }
                    // For u's neighbours that are themselves in S,
                    // their load_<opposite> must drop by vwgt[u]
                    // (since u is no longer on opposite side).
                    for &uu_n in graph.neighbors(u) {
                        let uun = uu_n as usize;
                        if labels[uun] == PART_SEP && uun != vu {
                            if opposite == PART_A {
                                load_a[uun] -= graph.vwgt[uu] as i64;
                            } else {
                                load_b[uun] -= graph.vwgt[uu] as i64;
                            }
                        }
                    }
                } else if lu == PART_SEP {
                    // v moved from S to side. From u's perspective, v
                    // was previously in S (so contributed to neither
                    // load) and is now in `side`, so u's load_side
                    // increases by vwgt[v].
                    if side == PART_A {
                        load_a[uu] += graph.vwgt[vu] as i64;
                    } else {
                        load_b[uu] += graph.vwgt[vu] as i64;
                    }
                }
                // lu == side: v was in S contributing 0 to u's loads;
                // now in side contributing vwgt[v]. But u is in `side`
                // too - its load arrays only matter while u is in S,
                // and it isn't - so no update.
            }

            // Re-stamp PQ entries for any S-vertex whose load may have
            // changed. We touched: u's freshly pulled into S (already
            // pushed below), u's neighbours that are in S (load_<opp>
            // changed), and S-vertices adjacent to v (load_side
            // changed).
            for &u in &nbrs_v {
                let uu = u as usize;
                if labels[uu] == PART_SEP && !locked[uu] {
                    if load_a[uu] > 0 {
                        let g = gain_to_side(uu, PART_A, &graph.vwgt, load_a, load_b);
                        pq_to_a.push((g, Reverse(u), g));
                    }
                    if load_b[uu] > 0 {
                        let g = gain_to_side(uu, PART_B, &graph.vwgt, load_a, load_b);
                        pq_to_b.push((g, Reverse(u), g));
                    }
                }
            }
            // For each freshly-pulled u, push its PQ entries (if not
            // locked - and we just locked them above, so they will
            // never enter the heap as candidates this pass; but their
            // load updates have been applied).
            // Locked vertices are skipped at pop time, so even if they
            // were in the heap from before, they're harmless.

            // For each pulled u's S-neighbours (other than v) re-stamp.
            for (u, _) in &pulled {
                for &n2 in graph.neighbors(*u) {
                    let nn = n2 as usize;
                    if labels[nn] == PART_SEP && !locked[nn] {
                        if load_a[nn] > 0 {
                            let g = gain_to_side(nn, PART_A, &graph.vwgt, load_a, load_b);
                            pq_to_a.push((g, Reverse(n2), g));
                        }
                        if load_b[nn] > 0 {
                            let g = gain_to_side(nn, PART_B, &graph.vwgt, load_a, load_b);
                            pq_to_b.push((g, Reverse(n2), g));
                        }
                    }
                }
            }

            moves.push((v, side, pulled));
            moves_done += 1;
            moved_this_iter = true;

            // Final-prefix imbalance check (audit finding 7b): only
            // record as best if the current side weights still respect
            // the cap.
            if sep_w < best_sep_w && a_w.max(b_w) <= max_side {
                best_sep_w = sep_w;
                best_a_w = a_w;
                best_b_w = b_w;
                best_prefix = moves.len();
            }
            break;
        }
        // End the pass only when the frontier is truly exhausted: no
        // move committed AND nothing was merely imbalance-rejected. A
        // rejection means feasible lower-gain moves may remain queued
        // (O13), so we keep going.
        if !moved_this_iter && !rejected_this_iter {
            break;
        }
    }

    // Roll back to best_prefix.
    if best_prefix < moves.len() {
        for i in (best_prefix..moves.len()).rev() {
            let (v, side, pulled) = &moves[i];
            let vu = *v as usize;
            // Undo: pulled vertices return to their original side, v
            // returns to S.
            for &(u, prev) in pulled {
                labels[u as usize] = prev;
            }
            labels[vu] = PART_SEP;
            let _ = side; // side is implied by the move
        }
        // Recompute loads from scratch - cheaper than mirroring every
        // delta in reverse.
        recompute_loads(graph, labels, load_a, load_b);
    }

    let _ = best_a_w;
    let _ = best_b_w;
    let final_a_w = part_weight(graph, labels, PART_A);
    let final_b_w = part_weight(graph, labels, PART_B);
    (best_sep_w, final_a_w, final_b_w)
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

    fn assert_valid_separator(graph: &Graph, labels: &[u8]) {
        for v in 0..graph.nvtxs as usize {
            let lv = labels[v];
            if lv != PART_A && lv != PART_B {
                continue;
            }
            for &u in graph.neighbors(v as i32) {
                let lu = labels[u as usize];
                assert!(
                    !((lv == PART_A && lu == PART_B) || (lv == PART_B && lu == PART_A)),
                    "edge ({}, {}) crosses A-B",
                    v,
                    u
                );
            }
        }
    }

    fn balanced(graph: &Graph, labels: &[u8], max_imbalance: f64) -> bool {
        let total: i64 = graph.vwgt.iter().map(|&w| w as i64).sum();
        let max_side = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;
        let a = part_weight(graph, labels, PART_A);
        let b = part_weight(graph, labels, PART_B);
        a.max(b) <= max_side
    }

    fn sep_w(graph: &Graph, labels: &[u8]) -> i64 {
        let mut s: i64 = 0;
        for (v, &l) in labels.iter().enumerate() {
            if l == PART_SEP {
                s += graph.vwgt[v] as i64;
            }
        }
        s
    }

    #[test]
    fn path_p11_separator_is_one_vertex() {
        // P_11: 0-1-2-...-10. Initial cut [0..5] | [5..10].
        let edges: Vec<(usize, usize)> = (0..10).map(|i| (i, i + 1)).collect();
        let g = build(11, &edges);
        let mut labels: Vec<u8> = (0..11u8)
            .map(|k| if k < 5 { PART_A } else { PART_B })
            .collect();
        let s = compute_vertex_separator(&g, &mut labels, 0.20, 200, 32);
        assert_valid_separator(&g, &labels);
        assert_eq!(s, sep_w(&g, &labels), "I1: bookkeeping");
        assert!(s >= 1, "non-trivial separator");
        assert!(s <= 1, "expected singleton separator on a path, got {}", s);
        assert!(balanced(&g, &labels, 0.20));
    }

    #[test]
    fn cycle_c8_separator_at_most_two() {
        // C_8: 8-cycle. Min separator weight is 2.
        let mut edges: Vec<(usize, usize)> = (0..7).map(|i| (i, i + 1)).collect();
        edges.push((7, 0));
        let g = build(8, &edges);
        let mut labels: Vec<u8> = (0..8u8)
            .map(|k| if k < 4 { PART_A } else { PART_B })
            .collect();
        let s = compute_vertex_separator(&g, &mut labels, 0.20, 200, 32);
        assert_valid_separator(&g, &labels);
        assert_eq!(s, sep_w(&g, &labels), "I1: bookkeeping");
        assert!(s <= 2, "cycle min-separator is 2, got {}", s);
        assert!(s >= 2, "cycle requires at least 2 separator vertices");
    }

    #[test]
    fn k33_separator_is_valid_and_bounded() {
        // K_{3,3}: nodes 0..3 vs 3..6. Any valid node separator that
        // keeps both sides non-empty must contain a whole side
        // (weight 3). However, the FM contract here only enforces
        // the *upper* balance bound; with a loose tolerance FM may
        // collapse one side to ∅ and shrink the separator further.
        // We therefore assert validity and weight bound, not exact 3.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..3 {
            for j in 3..6 {
                edges.push((i, j));
            }
        }
        let g = build(6, &edges);
        let mut labels: Vec<u8> = (0..6u8)
            .map(|k| if k < 3 { PART_A } else { PART_B })
            .collect();
        let s = compute_vertex_separator(&g, &mut labels, 0.50, 200, 32);
        assert_valid_separator(&g, &labels);
        assert_eq!(s, sep_w(&g, &labels), "I1: bookkeeping");
        assert!(
            (1..=3).contains(&s),
            "K_{{3,3}} separator must be in [1,3], got {}",
            s
        );
    }

    #[test]
    fn disjoint_components_no_separator() {
        // Two disjoint K_4: nodes 0..4 (one block) and 4..8 (another).
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for i in 0..4 {
            for j in (i + 1)..4 {
                edges.push((i, j));
            }
        }
        for i in 4..8 {
            for j in (i + 1)..8 {
                edges.push((i, j));
            }
        }
        let g = build(8, &edges);
        let mut labels: Vec<u8> = (0..8u8)
            .map(|k| if k < 4 { PART_A } else { PART_B })
            .collect();
        let s = compute_vertex_separator(&g, &mut labels, 0.20, 200, 32);
        assert_valid_separator(&g, &labels);
        assert_eq!(s, sep_w(&g, &labels), "I1: bookkeeping");
        assert_eq!(s, 0, "no crossing edges → no separator");
    }

    #[test]
    fn grid_5x5_row_separator_bounded() {
        // 5x5 grid, hand-cut into two 5x2 + middle row.
        let m = 5;
        let n = 5;
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
        let g = build(m * n, &edges);
        // Initial: top 2 rows = A, bottom 2 rows = B, middle row split arbitrarily.
        let mut labels: Vec<u8> = (0..(m * n) as u8)
            .map(|k| {
                let r = (k as usize) / n;
                if r < 2 {
                    PART_A
                } else {
                    PART_B
                }
            })
            .collect();
        let s = compute_vertex_separator(&g, &mut labels, 0.30, 200, 32);
        assert_valid_separator(&g, &labels);
        assert_eq!(s, sep_w(&g, &labels), "I1: bookkeeping");
        // Expected: one row of 5 vertices is enough to separate.
        // Allow some slack for FM not finding the true optimum.
        assert!(s <= 7, "grid 5x5 separator should be <= 7, got {}", s);
        assert!(s >= 5, "grid 5x5 needs at least 5 separator vertices");
    }

    #[test]
    fn separator_is_deterministic() {
        // Same input twice → same output.
        let m = 4;
        let n = 4;
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
        let g = build(m * n, &edges);
        let init: Vec<u8> = (0..(m * n) as u8)
            .map(|k| if (k as usize / n) < 2 { PART_A } else { PART_B })
            .collect();
        let mut l1 = init.clone();
        let mut l2 = init.clone();
        let s1 = compute_vertex_separator(&g, &mut l1, 0.20, 200, 32);
        let s2 = compute_vertex_separator(&g, &mut l2, 0.20, 200, 32);
        assert_eq!(s1, sep_w(&g, &l1), "I1: bookkeeping (run a)");
        assert_eq!(s2, sep_w(&g, &l2), "I1: bookkeeping (run b)");
        assert_eq!(l1, l2, "FM must be deterministic");
    }

    #[test]
    fn empty_graph_returns_zero() {
        // n=0
        let cp: Vec<i32> = vec![0];
        let ri: Vec<i32> = vec![];
        let pat = CscPattern::new(0, &cp, &ri).unwrap();
        let g = Graph::from_csc_pattern(&pat).unwrap();
        let mut labels: Vec<u8> = vec![];
        let s = compute_vertex_separator(&g, &mut labels, 0.20, 200, 32);
        assert_eq!(s, sep_w(&g, &labels), "I1: bookkeeping");
        assert_eq!(s, 0);
    }

    #[test]
    fn weight_matches_label_count() {
        let m = 5;
        let n = 5;
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
        let g = build(m * n, &edges);
        let mut labels: Vec<u8> = (0..(m * n) as u8)
            .map(|k| if (k as usize / n) < 2 { PART_A } else { PART_B })
            .collect();
        let s = compute_vertex_separator(&g, &mut labels, 0.30, 200, 32);
        assert_eq!(s, sep_w(&g, &labels));
    }

    #[test]
    fn fm_pass_continues_past_imbalance_rejected_heads() {
        // O13 regression: when BOTH PQ heads are imbalance-rejected in
        // one outer FM iteration, the pass must keep trying the
        // next-lower feasible moves, not break with feasible moves
        // still queued (SCOTCH skips infeasible heads and continues).
        //
        // Weighted graph; each S vertex is adjacent to exactly one
        // side, so moving it pulls nothing - fully predictable:
        //   0=a1 (A, w4) - 3=s1 (S, w3)
        //   1=a3 (A, w4) - 5=s3 (S, w1)
        //   2=b2 (B, w9) - 4=s2 (S, w2)
        // Initial: A={a1,a3}=8, B={b2}=9, S={s1,s2,s3}=6.
        // With max_side=10:
        //   s1->A: post A=11 > 10  reject  (pq_to_a head, gain 3)
        //   s2->B: post B=11 > 10  reject  (pq_to_b head, gain 2)
        //   s3->A: post A= 9 <=10  FEASIBLE (pq_to_a, gain 1) sep 6->5
        // Pre-fix code rejects both heads in iter 1, sets
        // moved_this_iter=false, and breaks - never reaching s3, so it
        // returns sep_w=6. The fix continues to s3 for sep_w=5.
        let mut g = build(6, &[(0, 3), (1, 5), (2, 4)]);
        g.vwgt = vec![4, 4, 9, 3, 2, 1];
        let mut labels = vec![PART_A, PART_A, PART_B, PART_SEP, PART_SEP, PART_SEP];
        let mut load_a = vec![0i64; 6];
        let mut load_b = vec![0i64; 6];
        recompute_loads(&g, &labels, &mut load_a, &mut load_b);
        // Loads as expected (each S vertex sees only one side).
        assert_eq!((load_a[3], load_b[3]), (4, 0), "s1 sees A only");
        assert_eq!((load_a[4], load_b[4]), (0, 9), "s2 sees B only");
        assert_eq!((load_a[5], load_b[5]), (4, 0), "s3 sees A only");

        let (sep_w_out, _a, _b) = fm_pass(
            &g,
            &mut labels,
            &mut load_a,
            &mut load_b,
            6,   // init_sep_w
            8,   // init_a_w
            9,   // init_b_w
            10,  // max_side
            200, // move_cap
        );

        assert_eq!(
            sep_w_out, 5,
            "FM must skip the imbalance-rejected heads and reach the \
             feasible lower-gain move s3->A; got sep_w={}",
            sep_w_out
        );
        assert_eq!(labels[5], PART_A, "s3 should have moved into A");
        assert_eq!(labels[3], PART_SEP, "s1 stays in S (infeasible)");
        assert_eq!(labels[4], PART_SEP, "s2 stays in S (infeasible)");
    }
}
