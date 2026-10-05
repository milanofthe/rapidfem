//! The outline of a face in its chart: points with the slot each stands
//! for (a point the face shares with its neighbours, or one of its own),
//! the constraint segments, and the loops for the inside test.

use rapidmesh_exact::vector::{V2, V3};
use rustc_hash::{FxHashMap, FxHashSet};

/// A point of a face's chart: one shared with the face's neighbours (a
/// corner or edge sample, by global id) or one of the face's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Slot {
    Global(u32),
    Own(u32),
}

/// The fixed outline of a face in its chart: points with their slots (a
/// global point may appear twice, on either side of a seam), the face's own
/// points so far (a seam's), the constraint segments and the loops for the
/// inside test.
#[derive(Default)]
pub struct Domain {
    pub pts: Vec<V2>,
    pub slots: Vec<Slot>,
    pub own: Vec<V3>,
    pub segments: FxHashSet<(usize, usize)>,
    pub loops: Vec<Vec<V2>>,
    index: FxHashMap<Slot, Vec<usize>>,
}

impl Domain {
    /// A new point of the face's own, by its index among them.
    pub fn add_own(&mut self, p: V3) -> u32 {
        self.own.push(p);
        (self.own.len() - 1) as u32
    }

    /// The local index of `slot` at `q`: one per slot and place (a seam
    /// puts a slot in two places a turn apart; the same place reached by
    /// two roundings is one).
    pub fn add_point(&mut self, slot: Slot, q: V2) -> usize {
        if let Some(ids) = self.index.get(&slot) {
            for &i in ids {
                let p = self.pts[i];
                // Relative to the larger of the two places and the
                // domain's first point: two roundings of a place at the
                // chart's origin (an apex) are one too.
                let scale = [p, q, self.pts[0]]
                    .iter()
                    .fold(0.0f64, |m, x| m.max(x[0].abs()).max(x[1].abs()));
                let tol = 1e-9 * scale.max(1e-300);
                if (p[0] - q[0]).abs() <= tol && (p[1] - q[1]).abs() <= tol {
                    return i;
                }
            }
        }
        self.pts.push(q);
        self.slots.push(slot);
        self.index.entry(slot).or_default().push(self.pts.len() - 1);
        self.pts.len() - 1
    }

    pub fn add_segment(&mut self, a: usize, b: usize) {
        if a != b {
            self.segments.insert((a.min(b), a.max(b)));
        }
    }

    /// A closed loop of the outline.
    pub fn add_loop(&mut self, ring: &[(Slot, V2)]) {
        let ids: Vec<usize> = ring.iter().map(|&(s, q)| self.add_point(s, q)).collect();
        for k in 0..ids.len() {
            self.add_segment(ids[k], ids[(k + 1) % ids.len()]);
        }
        self.loops.push(ring.iter().map(|x| x.1).collect());
    }

    /// The loops of the outline from its segments (for a domain built from
    /// chains): open chains (an edge inside) pruned, the rest split into
    /// cycles. The even-odd inside test counts each segment once however
    /// the cycles group them, so an outline that touches itself (at a pole)
    /// is as good as any.
    pub fn close_loops(&mut self) {
        let mut adj: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for &(a, b) in &self.segments {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
        // Prune open chains.
        let mut ends: Vec<usize> = adj
            .iter()
            .filter(|(_, n)| n.len() == 1)
            .map(|(&v, _)| v)
            .collect();
        while let Some(v) = ends.pop() {
            let Some(ns) = adj.get(&v) else {
                continue;
            };
            if ns.len() != 1 {
                continue;
            }
            let w = ns[0];
            adj.remove(&v);
            if let Some(nw) = adj.get_mut(&w) {
                nw.retain(|&x| x != v);
                if nw.len() == 1 {
                    ends.push(w);
                } else if nw.is_empty() {
                    adj.remove(&w);
                }
            }
        }
        // Hierholzer: walk unused segments until back at the start.
        let mut starts: Vec<usize> = adj.keys().copied().collect();
        starts.sort_unstable();
        for s in starts {
            while adj.get(&s).is_some_and(|n| !n.is_empty()) {
                let mut ring = vec![s];
                let mut cur = s;
                while let Some(next) = adj.get_mut(&cur).and_then(|n| n.pop()) {
                    if let Some(nn) = adj.get_mut(&next) {
                        if let Some(k) = nn.iter().position(|&x| x == cur) {
                            nn.swap_remove(k);
                        }
                    }
                    cur = next;
                    if cur == s {
                        break;
                    }
                    ring.push(cur);
                    if ring.len() > self.pts.len() + 1 {
                        break;
                    }
                }
                if cur == s && ring.len() >= 3 {
                    self.loops.push(ring.iter().map(|&i| self.pts[i]).collect());
                }
            }
        }
    }

    /// An open chain of constraint segments inside the face.
    pub fn add_chain(&mut self, chain: &[(Slot, V2)]) {
        let ids: Vec<usize> = chain.iter().map(|&(s, q)| self.add_point(s, q)).collect();
        for w in ids.windows(2) {
            self.add_segment(w[0], w[1]);
        }
    }
}
