//! Node-separator refinement (two-sided FM on the vertex separator).
//!
//! A pass moves separator vertices into one side; every neighbor of the moved
//! vertex that sat on the other side is pulled into the separator. The gain of
//! moving `v` into side `s` is `w(v) - w(N(v) on the far side)`. Vertices are
//! taken from an indexed max-heap keyed by gain (in-place key updates, one
//! entry per vertex, most recently touched vertex first among equal gains),
//! moves are journaled, and the pass rolls back to the best separator seen.
//! A pass stops when the heap drains, after [`MOVE_LIMIT`] consecutive moves
//! without improvement, or when the separator has grown past
//! [`MAX_OVERSHOOT`] times the best one.

use crate::fm_refine::PART_SEP;
use crate::graph::Graph;
use crate::initial_partition::{PART_A, PART_B};
use crate::rng::SplitMix;

/// Consecutive non-improving moves after which a pass gives up. Effectively
/// unbounded: the one-sided passes find their best separators at the end of
/// long hill traversals (a METIS-style limit of 300 costs 10 to 35 percent
/// fill on 2D and 3D grids for a 15 percent time saving), so a pass runs
/// until the heap drains or the overshoot cap trips.
const MOVE_LIMIT: usize = 1 << 20;

/// A pass also stops when the separator has grown to this multiple of the
/// best separator seen (a safety net for pathological hill climbs).
const MAX_OVERSHOOT: f64 = 4.0;

/// Weight of `v`'s neighbors that carry label `s`.
#[inline]
fn far_weight(graph: &Graph, labels: &[u8], v: usize, s: usize) -> i64 {
    let mut w = 0i64;
    for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
        let u = graph.adjncy[k] as usize;
        if labels[u] == s as u8 {
            w += graph.vwgt[u] as i64;
        }
    }
    w
}

/// Total vertex weight per label class `[A, B, SEP]`.
fn class_weights(graph: &Graph, labels: &[u8]) -> [i64; 3] {
    let mut w = [0i64; 3];
    for v in 0..graph.nvtxs as usize {
        w[labels[v] as usize] += graph.vwgt[v] as i64;
    }
    w
}

/// Gain of moving separator vertex `v` into side `into`.
#[inline]
fn gain(graph: &Graph, labels: &[u8], v: usize, into: usize) -> i64 {
    graph.vwgt[v] as i64 - far_weight(graph, labels, v, 1 - into)
}

const NONE: u32 = u32::MAX;

/// Indexed binary max-heap over vertices: `key` orders, the most recently
/// pushed or updated vertex wins ties (the `tie` stamp; a touched vertex is
/// handled next, which keeps the moves contiguous along the separator), and
/// `pos` maps a vertex to its slot so keys update in place.
pub(crate) struct GainHeap {
    heap: Vec<u32>,
    pos: Vec<u32>,
    key: Vec<i64>,
    tie: Vec<u64>,
    seq: u64,
}

impl GainHeap {
    pub fn new(n: usize) -> Self {
        Self {
            heap: Vec::with_capacity(n),
            pos: vec![NONE; n],
            key: vec![0; n],
            tie: vec![0; n],
            seq: 0,
        }
    }

    /// Empty the heap (`O(len)`), keeping the allocations.
    pub fn clear(&mut self) {
        for &v in &self.heap {
            self.pos[v as usize] = NONE;
        }
        self.heap.clear();
    }

    #[inline]
    pub fn contains(&self, v: usize) -> bool {
        self.pos[v] != NONE
    }

    #[inline]
    fn before(&self, a: u32, b: u32) -> bool {
        let (ka, kb) = (self.key[a as usize], self.key[b as usize]);
        ka > kb || (ka == kb && self.tie[a as usize] > self.tie[b as usize])
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if self.before(self.heap[i], self.heap[p]) {
                self.heap.swap(i, p);
                self.pos[self.heap[i] as usize] = i as u32;
                self.pos[self.heap[p] as usize] = p as u32;
                i = p;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let (l, r) = (2 * i + 1, 2 * i + 2);
            let mut best = i;
            if l < n && self.before(self.heap[l], self.heap[best]) {
                best = l;
            }
            if r < n && self.before(self.heap[r], self.heap[best]) {
                best = r;
            }
            if best == i {
                break;
            }
            self.heap.swap(i, best);
            self.pos[self.heap[i] as usize] = i as u32;
            self.pos[self.heap[best] as usize] = best as u32;
            i = best;
        }
    }

    /// Insert `v` (not in the heap) with `key`.
    pub fn push(&mut self, v: usize, key: i64) {
        debug_assert!(!self.contains(v));
        self.seq += 1;
        self.key[v] = key;
        self.tie[v] = self.seq;
        self.pos[v] = self.heap.len() as u32;
        self.heap.push(v as u32);
        self.sift_up(self.heap.len() - 1);
    }

    /// Set the key of `v` (in the heap), stamp it as most recent, and restore
    /// the heap order.
    pub fn update(&mut self, v: usize, key: i64) {
        let i = self.pos[v] as usize;
        let old = self.key[v];
        self.seq += 1;
        self.key[v] = key;
        self.tie[v] = self.seq;
        if key >= old {
            self.sift_up(i);
        } else {
            self.sift_down(i);
        }
    }

    /// Remove and return the vertex with the largest key.
    pub fn pop(&mut self) -> Option<usize> {
        let top = *self.heap.first()?;
        let last = self.heap.pop().expect("non-empty");
        self.pos[top as usize] = NONE;
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.pos[last as usize] = 0;
            self.sift_down(0);
        }
        Some(top as usize)
    }
}

/// One label change, for the rollback journal.
#[derive(Clone, Copy)]
struct Transition {
    vertex: u32,
    old_label: u8,
}

/// Per-call scratch: the heap plus the move journal.
pub(crate) struct Workspace {
    heap: GainHeap,
    journal: Vec<Transition>,
    order: Vec<u32>,
}

impl Workspace {
    pub fn new(n: usize) -> Self {
        Self {
            heap: GainHeap::new(n),
            journal: Vec::new(),
            order: Vec::new(),
        }
    }
}

/// Move separator vertex `v` into side `into`, pulling its far-side neighbors
/// into the separator; keeps the heap keys of the affected separator vertices
/// current. Returns the change of the separator weight.
fn apply_move(
    graph: &Graph,
    labels: &mut [u8],
    class_w: &mut [i64; 3],
    ws: &mut Workspace,
    v: usize,
    into: usize,
) -> i64 {
    let from_side = 1 - into;
    let wv = graph.vwgt[v] as i64;

    ws.journal.push(Transition {
        vertex: v as u32,
        old_label: PART_SEP,
    });
    labels[v] = into as u8;
    class_w[PART_SEP as usize] -= wv;
    class_w[into] += wv;
    let mut delta = -wv;

    for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
        let u = graph.adjncy[k] as usize;
        if labels[u] != from_side as u8 {
            continue;
        }
        // `u` joins the separator.
        let wu = graph.vwgt[u] as i64;
        ws.journal.push(Transition {
            vertex: u as u32,
            old_label: from_side as u8,
        });
        labels[u] = PART_SEP;
        class_w[from_side] -= wu;
        class_w[PART_SEP as usize] += wu;
        delta += wu;
        ws.heap.push(u, gain(graph, labels, u, into));
        // Every separator neighbor of `u` lost `u` from its far side: its
        // gain for moving into `into` grows by `w(u)`.
        for kk in graph.xadj[u] as usize..graph.xadj[u + 1] as usize {
            let t = graph.adjncy[kk] as usize;
            if t == u || labels[t] != PART_SEP {
                continue;
            }
            if ws.heap.contains(t) {
                let key = ws.heap.key[t] + wu;
                ws.heap.update(t, key);
            } else {
                // Dropped earlier as too heavy for the side cap; a fresh gain
                // makes it a candidate again.
                ws.heap.push(t, gain(graph, labels, t, into));
            }
        }
    }
    delta
}

/// Roll the journal back to length `keep`, restoring labels and class weights.
fn unwind(
    graph: &Graph,
    labels: &mut [u8],
    class_w: &mut [i64; 3],
    ws: &mut Workspace,
    keep: usize,
) {
    while ws.journal.len() > keep {
        let t = ws.journal.pop().expect("journal non-empty");
        let v = t.vertex as usize;
        let wv = graph.vwgt[v] as i64;
        class_w[labels[v] as usize] -= wv;
        class_w[t.old_label as usize] += wv;
        labels[v] = t.old_label;
    }
}

/// Seed the heap with every separator vertex's gain for moving into `into`,
/// in a random order (the tie-break among equal gains, seed-deterministic).
fn seed_heap(graph: &Graph, labels: &[u8], ws: &mut Workspace, rng: &mut SplitMix, into: usize) {
    ws.heap.clear();
    ws.order.clear();
    ws.order
        .extend((0..graph.nvtxs as u32).filter(|&v| labels[v as usize] == PART_SEP));
    rng.shuffle(&mut ws.order);
    for i in 0..ws.order.len() {
        let v = ws.order[i] as usize;
        ws.heap.push(v, gain(graph, labels, v, into));
    }
}

/// One refinement pass moving separator vertices into side `into` while that
/// side stays under `side_cap`. Returns the (best) separator weight reached.
fn side_pass(
    graph: &Graph,
    labels: &mut [u8],
    into: usize,
    side_cap: i64,
    rng: &mut SplitMix,
    ws: &mut Workspace,
) -> i64 {
    let mut class_w = class_weights(graph, labels);
    seed_heap(graph, labels, ws, rng, into);
    ws.journal.clear();

    let mut sep_w = class_w[PART_SEP as usize];
    let mut best_w = sep_w;
    let mut best_len = 0usize;
    let mut since_best = 0usize;

    while let Some(v) = ws.heap.pop() {
        debug_assert_eq!(labels[v], PART_SEP);
        if class_w[into] + graph.vwgt[v] as i64 > side_cap {
            continue; // too heavy for the target side right now
        }
        sep_w += apply_move(graph, labels, &mut class_w, ws, v, into);
        if sep_w < best_w {
            best_w = sep_w;
            best_len = ws.journal.len();
            since_best = 0;
        } else {
            since_best += 1;
            let overshoot = sep_w - best_w;
            if since_best > MOVE_LIMIT || overshoot as f64 > MAX_OVERSHOOT * best_w.max(1) as f64 {
                break;
            }
        }
    }

    unwind(graph, labels, &mut class_w, ws, best_len);
    debug_assert_eq!(class_w[PART_SEP as usize], best_w);
    best_w
}

/// Refine a node separator (labels `PART_A` / `PART_B` / `PART_SEP`) for up
/// to `max_rounds` rounds of two side passes each, keeping either side under
/// `(1 + max_imbalance) / 2` of the total weight. Returns the separator
/// weight.
pub(crate) fn refine_node_separator(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    max_rounds: u32,
    rng: &mut SplitMix,
) -> i64 {
    let class_w = class_weights(graph, labels);
    let mut sep_w = class_w[PART_SEP as usize];
    if sep_w == 0 {
        return 0;
    }
    let total = class_w[0] + class_w[1] + class_w[2];
    let side_cap = ((1.0 + max_imbalance) * total as f64 / 2.0).ceil() as i64;
    let mut ws = Workspace::new(graph.nvtxs as usize);

    for _round in 0..max_rounds {
        let w = class_weights(graph, labels);
        let first = if w[PART_A as usize] <= w[PART_B as usize] {
            PART_A as usize
        } else {
            PART_B as usize
        };
        let after_first = side_pass(graph, labels, first, side_cap, rng, &mut ws);
        let after_second = side_pass(graph, labels, 1 - first, side_cap, rng, &mut ws);
        let round_end = after_first.min(after_second);
        if round_end >= sep_w {
            sep_w = round_end;
            break;
        }
        sep_w = round_end;
    }
    sep_w
}

/// Rebalance a projected separator whose sides differ by more than the
/// tolerance: move separator vertices (best gain first) into the lighter
/// side until the sides balance. Moves are kept, never rolled back.
pub(crate) fn balance_node_separator(
    graph: &Graph,
    labels: &mut [u8],
    max_imbalance: f64,
    rng: &mut SplitMix,
) {
    let mut class_w = class_weights(graph, labels);
    let a = class_w[PART_A as usize];
    let b = class_w[PART_B as usize];
    let tolerance = ((a + b) as f64 * max_imbalance / 2.0).ceil() as i64;
    if (a - b).abs() <= tolerance.max(1) {
        return;
    }
    let into = if a < b {
        PART_A as usize
    } else {
        PART_B as usize
    };
    let heavy = 1 - into;

    let mut ws = Workspace::new(graph.nvtxs as usize);
    seed_heap(graph, labels, &mut ws, rng, into);
    while let Some(v) = ws.heap.pop() {
        if class_w[into] + graph.vwgt[v] as i64 > class_w[heavy] {
            continue;
        }
        apply_move(graph, labels, &mut class_w, &mut ws, v, into);
        if (class_w[PART_A as usize] - class_w[PART_B as usize]).abs() <= tolerance.max(1) {
            break;
        }
    }
}

#[cfg(test)]
pub(crate) fn is_valid_trisection(graph: &Graph, labels: &[u8]) -> bool {
    for v in 0..graph.nvtxs as usize {
        let lv = labels[v];
        if lv != PART_A && lv != PART_B {
            continue;
        }
        for k in graph.xadj[v] as usize..graph.xadj[v + 1] as usize {
            let u = graph.adjncy[k] as usize;
            let lu = labels[u];
            if (lv == PART_A && lu == PART_B) || (lv == PART_B && lu == PART_A) {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fm_refine::separator_weight;
    use crate::initial_partition::initial_bisect_ggp;
    use crate::separator::construct_separator;
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

    fn grid(m: usize, n: usize) -> Graph {
        let idx = |r: usize, c: usize| r * n + c;
        let total = m * n;
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
        let (cp, ri) = csc_from_triples(total, &t);
        let pat = CscPattern::new(total, &cp, &ri).unwrap();
        Graph::from_csc_pattern(&pat).unwrap()
    }

    /// Build a valid trisection on a grid via GGP + König, then check
    /// the refiner's invariants.
    fn refined_grid_case(m: usize, n: usize, seed: u64) -> (Graph, Vec<u8>, i64) {
        let g = grid(m, n);
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let mut rng = SplitMix::new(seed);
        let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
        construct_separator(&g, &mut labels);
        let before = separator_weight(&g, &labels);
        let after = refine_node_separator(&g, &mut labels, 0.20, 10, &mut rng);
        (g, labels, before - after)
    }

    #[test]
    fn refine_preserves_trisection_and_bookkeeping() {
        for seed in [1u64, 7, 21, 33] {
            let (g, labels, _) = refined_grid_case(12, 12, seed);
            assert!(is_valid_trisection(&g, &labels), "seed {seed}");
            // Returned weight must match a from-scratch recount.
            let mut labels2 = labels.clone();
            let mut rng = SplitMix::new(99);
            let w = refine_node_separator(&g, &mut labels2, 0.20, 10, &mut rng);
            assert_eq!(w, separator_weight(&g, &labels2), "bookkeeping");
            assert!(is_valid_trisection(&g, &labels2));
        }
    }

    #[test]
    fn refine_never_grows_separator() {
        for seed in [1u64, 5, 17] {
            let (_, _, saved) = refined_grid_case(16, 16, seed);
            assert!(saved >= 0, "separator grew by {} (seed {seed})", -saved);
        }
    }

    #[test]
    fn refine_finds_thin_separator_on_grid_band() {
        // 8x8 grid with a fat 3-column separator band: columns 3,4,5
        // SEP, cols 0-2 A, cols 6-7 B. Optimal is a single column (8).
        let g = grid(8, 8);
        let mut labels: Vec<u8> = (0..64u8)
            .map(|k| match k % 8 {
                0..=2 => PART_A,
                3..=5 => PART_SEP,
                _ => PART_B,
            })
            .collect();
        let before = separator_weight(&g, &labels);
        assert_eq!(before, 24);
        let mut rng = SplitMix::new(3);
        let after = refine_node_separator(&g, &mut labels, 0.20, 10, &mut rng);
        assert!(is_valid_trisection(&g, &labels));
        assert_eq!(after, separator_weight(&g, &labels), "bookkeeping");
        assert!(
            after <= 10,
            "node FM must thin a 3-wide band toward a column, got {after}"
        );
    }

    #[test]
    fn balance_moves_toward_lighter_side() {
        // Heavily imbalanced trisection on a 12x12 grid: col 1 = SEP,
        // col 0 = A (12 vertices), cols 2.. = B (120 vertices).
        let g = grid(12, 12);
        let mut labels: Vec<u8> = (0..144u16)
            .map(|k| match k % 12 {
                0 => PART_A,
                1 => PART_SEP,
                _ => PART_B,
            })
            .collect();
        assert!(is_valid_trisection(&g, &labels));
        let mut rng = SplitMix::new(11);
        balance_node_separator(&g, &mut labels, 0.20, &mut rng);
        assert!(is_valid_trisection(&g, &labels));
        let a: i64 = labels
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l == PART_A)
            .map(|(v, _)| g.vwgt[v] as i64)
            .sum();
        let b: i64 = labels
            .iter()
            .enumerate()
            .filter(|&(_, &l)| l == PART_B)
            .map(|(v, _)| g.vwgt[v] as i64)
            .sum();
        assert!(
            a.max(b) < 132,
            "balance must reduce the 12/120 imbalance, got {a}/{b}"
        );
    }

    #[test]
    fn refine_deterministic_with_seed() {
        let g = grid(14, 14);
        let total: i64 = g.vwgt.iter().map(|&w| w as i64).sum();
        let mk = || {
            let mut rng = SplitMix::new(42);
            let mut labels = initial_bisect_ggp(&g, &mut rng, total / 2);
            construct_separator(&g, &mut labels);
            let w = refine_node_separator(&g, &mut labels, 0.20, 10, &mut rng);
            (labels, w)
        };
        let (l1, w1) = mk();
        let (l2, w2) = mk();
        assert_eq!(w1, w2);
        assert_eq!(l1, l2);
    }

    #[test]
    fn empty_separator_is_noop() {
        let g = grid(4, 4);
        let mut labels = vec![PART_A; 16];
        let mut rng = SplitMix::new(1);
        let w = refine_node_separator(&g, &mut labels, 0.20, 10, &mut rng);
        assert_eq!(w, 0);
        assert_eq!(labels, vec![PART_A; 16]);
        balance_node_separator(&g, &mut labels, 0.20, &mut rng);
        assert_eq!(labels, vec![PART_A; 16]);
    }
}
