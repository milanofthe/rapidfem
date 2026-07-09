//! Push-relabel max-flow / min-cut solver (KaHIP Phase K2).
//!
//! Pure-Rust clean-room implementation per Goldberg & Tarjan 1988 and
//! Cherkassky & Goldberg 1995. See `dev/research/ordering-kahip-k2.md`
//! for the algorithm notes and test-oracle construction.
//!
//! Scope for K2:
//! - Highest-label selection with deterministic lowest-index admissible
//!   tie-breaking.
//! - Gap relabeling (required per audit item 6 of
//!   `dev/plans/ordering-kahip.md`).
//! - Min-cut extraction via residual-graph BFS from source.
//!
//! Out of scope (deferred): global relabeling, most-balanced-min-cut,
//! vertex-capacitated flow (K4 concern via node-splitting reduction).
//!
//! The module is `pub(crate)` and unused outside its own tests until
//! Phase K3 (flow-based refinement) lands.

use std::collections::VecDeque;

use rslab_ordering_core::OrderingError;

/// Internal edge representation for the residual network.
///
/// Each directed edge `(u, v, cap)` is stored with an index `rev`
/// pointing at its reverse edge in the same edges vector. Reverse
/// edges start with zero capacity so that the antiparallel pair
/// models a purely forward-directed capacitated edge in the flow
/// network while still allowing residual "cancellation" flow via the
/// reverse-edge capacity that grows as flow is pushed forward.
#[derive(Debug, Clone, Copy)]
struct Edge {
    to: usize,
    rev: usize,
    cap: i64,
    flow: i64,
}

impl Edge {
    #[inline]
    fn residual(&self) -> i64 {
        self.cap - self.flow
    }
}

/// Solve max-flow / min-cut on a directed capacitated graph.
///
/// `edges` is a slice of `(from, to, cap)` with nonnegative integer
/// capacities. Parallel edges are allowed. Self-loops are ignored.
/// Negative capacities, `source == sink`, or any out-of-bounds
/// vertex return [`OrderingError::MalformedInput`].
///
/// Returns `(flow_value, is_source_side)` where `is_source_side[v]`
/// holds iff `v` is reachable from `source` in the terminal residual
/// graph, which is a valid min-cut partition.
pub(crate) fn push_relabel(
    n: usize,
    edges: &[(usize, usize, i64)],
    source: usize,
    sink: usize,
) -> Result<(i64, Vec<bool>), OrderingError> {
    if n == 0 || source >= n || sink >= n || source == sink {
        return Err(OrderingError::MalformedInput);
    }
    for &(u, v, c) in edges {
        if u >= n || v >= n || c < 0 {
            return Err(OrderingError::MalformedInput);
        }
    }

    // Build CSR-like residual network: each input edge becomes a
    // forward/reverse pair. `adj[v]` is the list of indices into
    // `net` for edges out of v.
    let mut net: Vec<Edge> = Vec::with_capacity(2 * edges.len());
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v, c) in edges {
        if u == v {
            continue;
        }
        let fwd = net.len();
        let rev = fwd + 1;
        net.push(Edge {
            to: v,
            rev,
            cap: c,
            flow: 0,
        });
        net.push(Edge {
            to: u,
            rev: fwd,
            cap: 0,
            flow: 0,
        });
        adj[u].push(fwd);
        adj[v].push(rev);
    }

    // State arrays.
    let mut height: Vec<usize> = vec![0; n];
    let mut excess: Vec<i64> = vec![0; n];
    // height_count[h] = number of non-sink, non-disconnected vertices
    // currently at height h. Size 2n+1 is generous; any h >= n is
    // considered "disconnected from sink" and is excluded from the
    // histogram (we never tally those).
    let mut height_count: Vec<usize> = vec![0; 2 * n + 1];
    // Buckets of active vertices keyed by height. We pull the
    // highest non-empty bucket; within a bucket we use FIFO order.
    let mut bucket: Vec<VecDeque<usize>> = vec![VecDeque::new(); 2 * n + 1];
    let mut max_active: usize = 0;
    // `in_bucket[v]` prevents duplicate insertions when a single
    // vertex receives multiple pushes before being discharged.
    let mut in_bucket: Vec<bool> = vec![false; n];

    // Initial height function: source at height n, everyone else 0.
    // height_count is the count over V \ {source, sink}; the source's
    // height n is outside the tracked range, and sink stays at 0 and
    // is explicitly skipped in gap logic below.
    height[source] = n;
    // Tally non-source, non-sink vertices at height 0 (all of them).
    height_count[0] = n.saturating_sub(2);

    // Preflow initialization: saturate every out-edge of source.
    let source_edges: Vec<usize> = adj[source].clone();
    for ei in source_edges {
        let cap = net[ei].cap;
        if cap == 0 {
            continue;
        }
        let to = net[ei].to;
        net[ei].flow += cap;
        let rev = net[ei].rev;
        net[rev].flow -= cap;
        excess[source] -= cap;
        excess[to] += cap;
        if to != sink && !in_bucket[to] {
            bucket[height[to]].push_back(to);
            in_bucket[to] = true;
            if height[to] > max_active {
                max_active = height[to];
            }
        }
    }

    // Main discharge loop.
    while let Some(u) = pop_highest(&mut bucket, &mut max_active) {
        in_bucket[u] = false;
        if u == source || u == sink || excess[u] == 0 {
            continue;
        }
        discharge(
            u,
            &mut net,
            &adj,
            &mut height,
            &mut excess,
            &mut height_count,
            &mut bucket,
            &mut in_bucket,
            &mut max_active,
            n,
            source,
            sink,
        );
    }

    // Flow value: either -excess[source] or excess[sink]; both are
    // equal in magnitude at termination. Use excess[sink] for
    // clarity.
    let flow_value = excess[sink];

    // Min-cut: BFS from source on the residual graph.
    let mut is_source_side = vec![false; n];
    is_source_side[source] = true;
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(source);
    while let Some(u) = q.pop_front() {
        for &ei in &adj[u] {
            let e = net[ei];
            if e.residual() > 0 && !is_source_side[e.to] {
                is_source_side[e.to] = true;
                q.push_back(e.to);
            }
        }
    }

    Ok((flow_value, is_source_side))
}

/// Pull the highest-label active vertex, decrementing `max_active`
/// down to the next non-empty bucket on the fly. Returns `None` when
/// all buckets are empty.
fn pop_highest(bucket: &mut [VecDeque<usize>], max_active: &mut usize) -> Option<usize> {
    loop {
        if let Some(u) = bucket[*max_active].pop_front() {
            return Some(u);
        }
        if *max_active == 0 {
            return None;
        }
        *max_active -= 1;
    }
}

/// Discharge a single active vertex: push to admissible neighbors in
/// ascending adjacency-list order; if excess remains, relabel (with
/// gap detection) and repeat. Returns when excess drops to zero or
/// the vertex reaches height `n` (disconnected from sink and source).
#[allow(clippy::too_many_arguments)]
fn discharge(
    u: usize,
    net: &mut [Edge],
    adj: &[Vec<usize>],
    height: &mut [usize],
    excess: &mut [i64],
    height_count: &mut [usize],
    bucket: &mut [VecDeque<usize>],
    in_bucket: &mut [bool],
    max_active: &mut usize,
    n: usize,
    source: usize,
    sink: usize,
) {
    while excess[u] > 0 {
        let mut pushed_any = false;
        // Iterate neighbors in stored adjacency order (determinism).
        for &ei in &adj[u] {
            let e = net[ei];
            if e.residual() <= 0 {
                continue;
            }
            if height[u] != height[e.to] + 1 {
                continue;
            }
            // Admissible: push min(excess, residual).
            let delta = excess[u].min(e.residual());
            net[ei].flow += delta;
            let rev = net[ei].rev;
            net[rev].flow -= delta;
            excess[u] -= delta;
            excess[e.to] += delta;
            if e.to != source && e.to != sink && !in_bucket[e.to] {
                bucket[height[e.to]].push_back(e.to);
                in_bucket[e.to] = true;
                if height[e.to] > *max_active {
                    *max_active = height[e.to];
                }
            }
            pushed_any = true;
            if excess[u] == 0 {
                break;
            }
        }
        if excess[u] == 0 {
            return;
        }
        if pushed_any {
            // Still has excess but made progress; re-enter loop to
            // see if new admissible edges appeared (they don't, but
            // the idiom keeps discharge monotone).
            continue;
        }
        // Relabel: h(u) = 1 + min h(v) over residual out-edges.
        let mut new_height = usize::MAX;
        for &ei in &adj[u] {
            let e = net[ei];
            if e.residual() > 0 && height[e.to] + 1 < new_height {
                new_height = height[e.to] + 1;
            }
        }
        // An active vertex always has a residual reverse edge - the reverse
        // of whichever edge delivered its excess - so the relabel scan above
        // always finds a finite height. The branch below is therefore
        // unreachable while `excess[u] > 0`, which holds here: we did not
        // return at the `excess[u] == 0` check above. The debug_assert pins
        // that invariant against future changes (O19, repo-review-2026-06-09).
        debug_assert_ne!(
            new_height,
            usize::MAX,
            "push-relabel: active vertex {u} (excess {}) has no residual \
             out-edge; the reverse of its in-flow edge must be residual",
            excess[u]
        );
        if new_height == usize::MAX {
            // Unreachable defensive fallback (see the debug_assert above).
            // NOTE: this returns *without* the `height_count[old_h] -= 1`
            // that the normal relabel path performs below, so reaching it
            // would corrupt the gap histogram - another reason to assert
            // entry rather than silently proceed on a bad count.
            height[u] = 2 * n;
            return;
        }

        let old_h = height[u];
        // Update the height histogram. Source stays pinned at n and
        // sink stays pinned at 0; neither is ever relabeled. Gap
        // detection only fires for `0 < g < n`: a gap at 0 would
        // falsely claim the sink (at height 0) is disconnected.
        if old_h < n {
            height_count[old_h] -= 1;
            if old_h > 0 && height_count[old_h] == 0 {
                // Gap at old_h: every vertex w (other than source /
                // sink) with old_h < h(w) < n is disconnected from
                // sink. Lift to n + 1 so the algorithm's second
                // phase can drain their excess back to source via
                // reverse edges.
                for w in 0..n {
                    if w == source || w == sink {
                        continue;
                    }
                    if height[w] > old_h && height[w] < n {
                        height_count[height[w]] -= 1;
                        height[w] = n + 1;
                        if excess[w] > 0 && !in_bucket[w] {
                            bucket[n + 1].push_back(w);
                            in_bucket[w] = true;
                        }
                    }
                }
                height[u] = n + 1;
                if excess[u] > 0 && !in_bucket[u] {
                    bucket[n + 1].push_back(u);
                    in_bucket[u] = true;
                }
                if n + 1 > *max_active {
                    *max_active = n + 1;
                }
                return;
            }
        }

        height[u] = new_height;
        if new_height < n {
            height_count[new_height] += 1;
        }
        if new_height > *max_active {
            *max_active = new_height;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_inputs() {
        assert!(matches!(
            push_relabel(0, &[], 0, 0),
            Err(OrderingError::MalformedInput)
        ));
        assert!(matches!(
            push_relabel(2, &[], 0, 0),
            Err(OrderingError::MalformedInput)
        ));
        assert!(matches!(
            push_relabel(2, &[], 5, 1),
            Err(OrderingError::MalformedInput)
        ));
        assert!(matches!(
            push_relabel(2, &[(0, 1, -1)], 0, 1),
            Err(OrderingError::MalformedInput)
        ));
        assert!(matches!(
            push_relabel(2, &[(0, 3, 1)], 0, 1),
            Err(OrderingError::MalformedInput)
        ));
    }

    #[test]
    fn empty_graph_zero_flow() {
        let (f, side) = push_relabel(2, &[], 0, 1).unwrap();
        assert_eq!(f, 0);
        assert!(side[0]);
        assert!(!side[1]);
    }

    #[test]
    fn single_edge_saturates() {
        let (f, side) = push_relabel(2, &[(0, 1, 7)], 0, 1).unwrap();
        assert_eq!(f, 7);
        assert!(side[0]);
        assert!(!side[1]);
    }

    #[test]
    fn parallel_edges_sum() {
        let (f, _) = push_relabel(2, &[(0, 1, 3), (0, 1, 5)], 0, 1).unwrap();
        assert_eq!(f, 8);
    }

    #[test]
    fn unit_capacity_path() {
        // 0 -> 1 -> 2 -> 3 -> 4 with each capacity 1.
        let n = 5;
        let mut edges = Vec::new();
        for i in 0..n - 1 {
            edges.push((i, i + 1, 1));
        }
        let (f, side) = push_relabel(n, &edges, 0, n - 1).unwrap();
        assert_eq!(f, 1);
        // Min-cut here is any single edge along the path - the first
        // saturating edge is the first (0->1), so source side = {0}.
        assert!(side[0]);
        assert!(!side[n - 1]);
    }

    #[test]
    fn self_loops_ignored() {
        let (f, _) = push_relabel(2, &[(0, 0, 99), (0, 1, 4), (1, 1, 99)], 0, 1).unwrap();
        assert_eq!(f, 4);
    }

    #[test]
    fn diamond_bottleneck() {
        // 0 --10--> 1 --1--> 3
        // 0 --1 --> 2 --10--> 3
        let edges = [(0, 1, 10), (1, 3, 1), (0, 2, 1), (2, 3, 10)];
        let (f, side) = push_relabel(4, &edges, 0, 3).unwrap();
        assert_eq!(f, 2);
        assert!(side[0]);
        assert!(!side[3]);
    }

    #[test]
    fn clrs_figure_26_1() {
        // CLRS 3e Figure 26.1: max-flow = 23.
        // Vertices: s=0, v1=1, v2=2, v3=3, v4=4, t=5
        // Edges (directed):
        //   s->v1 16, s->v2 13
        //   v1->v3 12
        //   v2->v1 4, v2->v4 14
        //   v3->v2 9, v3->t 20
        //   v4->v3 7, v4->t 4
        let edges = [
            (0, 1, 16),
            (0, 2, 13),
            (1, 3, 12),
            (2, 1, 4),
            (2, 4, 14),
            (3, 2, 9),
            (3, 5, 20),
            (4, 3, 7),
            (4, 5, 4),
        ];
        let (f, _) = push_relabel(6, &edges, 0, 5).unwrap();
        assert_eq!(f, 23);
    }

    #[test]
    fn grid_horizontal_cut_equals_k() {
        // k x k grid with unit-capacity horizontal edges going
        // left-to-right; vertical edges are high-capacity so the
        // minimum cut is a vertical slice of k horizontal edges.
        // Super-source connects to the left column; super-sink to
        // the right column. Max-flow = k.
        for k in [2usize, 3, 4, 5] {
            let n = k * k + 2;
            let src = k * k;
            let sink = k * k + 1;
            let idx = |r: usize, c: usize| r * k + c;
            let mut edges = Vec::new();
            // Super-source to left column, infinite (= big).
            for r in 0..k {
                edges.push((src, idx(r, 0), 1_000_000));
            }
            // Right column to super-sink.
            for r in 0..k {
                edges.push((idx(r, k - 1), sink, 1_000_000));
            }
            // Horizontal unit edges.
            for r in 0..k {
                for c in 0..k - 1 {
                    edges.push((idx(r, c), idx(r, c + 1), 1));
                }
            }
            // Vertical high-cap edges (both directions).
            for r in 0..k - 1 {
                for c in 0..k {
                    edges.push((idx(r, c), idx(r + 1, c), 1_000_000));
                    edges.push((idx(r + 1, c), idx(r, c), 1_000_000));
                }
            }
            let (f, _) = push_relabel(n, &edges, src, sink).unwrap();
            assert_eq!(f, k as i64, "grid {}x{}", k, k);
        }
    }

    #[test]
    fn bipartite_matching_k_3_3() {
        // K_{3,3} bipartite matching via max-flow.
        // Left: 0,1,2   Right: 3,4,5   src=6   sink=7
        let src = 6;
        let sink = 7;
        let mut edges = vec![
            (src, 0, 1),
            (src, 1, 1),
            (src, 2, 1),
            (3, sink, 1),
            (4, sink, 1),
            (5, sink, 1),
        ];
        for l in 0..3 {
            for r in 3..6 {
                edges.push((l, r, 1));
            }
        }
        let (f, _) = push_relabel(8, &edges, src, sink).unwrap();
        assert_eq!(f, 3);
    }

    #[test]
    fn cut_saturation_invariant_on_random_graph() {
        // Deterministic pseudo-random directed graph; verify that
        // forward cut edges are saturated and backward cut edges
        // carry zero flow, i.e., |f| equals the sum of forward-cut
        // capacities.
        let n = 30;
        let mut edges = Vec::new();
        // Simple LCG for reproducibility (not a security RNG).
        let mut state: u64 = 0xC0FFEE;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state
        };
        for u in 0..n {
            for v in 0..n {
                if u == v {
                    continue;
                }
                if (next() % 100) < 15 {
                    let c = ((next() % 20) + 1) as i64;
                    edges.push((u, v, c));
                }
            }
        }
        let (f, side) = push_relabel(n, &edges, 0, n - 1).unwrap();
        assert!(side[0]);
        let mut forward_cap: i64 = 0;
        for &(u, v, c) in &edges {
            if side[u] && !side[v] {
                forward_cap += c;
            }
        }
        if side[n - 1] {
            // Source and sink on same side ⇒ graph has no s-t path,
            // so max-flow = 0 is the only correct answer.
            assert_eq!(f, 0);
            return;
        }
        assert_eq!(f, forward_cap, "max-flow must equal forward-cut capacity");
    }

    #[test]
    fn cut_saturation_invariant_on_connected_graph() {
        // A hand-laid network guaranteed to have positive max-flow.
        // Layered s -> A -> B -> t with cross and back edges.
        let edges = [
            // Source = 0, sink = 5.
            (0, 1, 6),
            (0, 2, 4),
            (1, 3, 3),
            (1, 4, 4),
            (2, 3, 2),
            (2, 4, 5),
            (3, 5, 5),
            (4, 5, 6),
            // Some back edges (should not affect max-flow here).
            (3, 1, 1),
            (4, 2, 2),
        ];
        let (f, side) = push_relabel(6, &edges, 0, 5).unwrap();
        // Hand check: capacity out of source = 10. Into sink = 11.
        // Middle bottleneck through {3,4}: 3 has 2 in (from 1 + 2) → 5 out; 4 has 9 in → 6 out. So min of (10, 3+4+2+5=14 into {3,4}, 5+6=11 into sink) = 10.
        assert_eq!(f, 10);
        assert!(side[0] && !side[5]);
        let mut forward_cap: i64 = 0;
        for &(u, v, c) in &edges {
            if side[u] && !side[v] {
                forward_cap += c;
            }
        }
        assert_eq!(f, forward_cap);
    }

    #[test]
    fn disconnected_sink_zero_flow() {
        // Two components: source in one, sink in another. Max-flow = 0
        // and source side is everything reachable from source.
        let edges = [(0, 1, 5), (2, 3, 5)];
        let (f, side) = push_relabel(4, &edges, 0, 3).unwrap();
        assert_eq!(f, 0);
        assert!(side[0]);
        assert!(side[1]);
        assert!(!side[2]);
        assert!(!side[3]);
    }

    #[test]
    fn deterministic_under_adjacency_order() {
        // The same graph built twice must return the same flow
        // value and cut partition.
        let edges = [(0, 1, 3), (0, 2, 2), (1, 2, 1), (1, 3, 2), (2, 3, 4)];
        let a = push_relabel(4, &edges, 0, 3).unwrap();
        let b = push_relabel(4, &edges, 0, 3).unwrap();
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }
}
