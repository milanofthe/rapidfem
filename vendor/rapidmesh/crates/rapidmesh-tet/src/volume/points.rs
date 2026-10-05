//! A spatial index of points whose cells follow the points: a k-d tree
//! with small leaves, so a graded point set (fine at a trace, coarse in the
//! air around it) costs the same per query everywhere, which a uniform
//! grid sized by the mean density does not. Points by distance from a
//! place, and points in a box.

use crate::simplex::Ordered;
use rapidmesh_exact::vector::V3;
use rapidmesh_exact::vector::{box_d2, dist2};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Points per leaf at most.
const LEAF: usize = 8;

#[derive(Clone, Copy)]
enum Kind {
    Inner(u32, u32),
    /// A range of `order`.
    Leaf(u32, u32),
}

#[derive(Clone, Copy)]
struct Node {
    lo: V3,
    hi: V3,
    kind: Kind,
}

/// The index over a point set (indices into it).
pub struct PointTree {
    nodes: Vec<Node>,
    /// The points, leaf by leaf.
    order: Vec<u32>,
}

impl PointTree {
    pub fn new(pts: &[V3]) -> PointTree {
        let mut t = PointTree {
            nodes: Vec::new(),
            order: (0..pts.len() as u32).collect(),
        };
        if !pts.is_empty() {
            t.build(pts, 0, pts.len());
        }
        t
    }

    fn build(&mut self, pts: &[V3], a: usize, b: usize) -> u32 {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for &i in &self.order[a..b] {
            for k in 0..3 {
                lo[k] = lo[k].min(pts[i as usize][k]);
                hi[k] = hi[k].max(pts[i as usize][k]);
            }
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(Node {
            lo,
            hi,
            kind: Kind::Leaf(a as u32, b as u32),
        });
        if b - a > LEAF {
            // The longest side split at the median.
            let k = (0..3)
                .max_by(|&x, &y| (hi[x] - lo[x]).total_cmp(&(hi[y] - lo[y])))
                .unwrap_or(0);
            let m = (a + b) / 2;
            self.order[a..b].select_nth_unstable_by(m - a, |&x, &y| {
                pts[x as usize][k].total_cmp(&pts[y as usize][k])
            });
            let left = self.build(pts, a, m);
            let right = self.build(pts, m, b);
            self.nodes[id as usize].kind = Kind::Inner(left, right);
        }
        id
    }

    /// The points by their distance from `q`, nearest first (ties in no
    /// particular order).
    pub fn by_distance<'a>(&'a self, pts: &'a [V3], q: V3) -> impl Iterator<Item = u32> + 'a {
        // Nodes by the distance of their boxes, points by theirs: a point
        // comes out once nothing left can be nearer.
        let mut heap: BinaryHeap<(Reverse<Ordered>, Reverse<u64>)> = BinaryHeap::new();
        // Entries: a node `n` as `2 n`, a point `v` as `2 v + 1`.
        if !self.nodes.is_empty() {
            let n = &self.nodes[0];
            heap.push((Reverse(Ordered(box_d2(n.lo, n.hi, q))), Reverse(0)));
        }
        std::iter::from_fn(move || loop {
            let (_, Reverse(e)) = heap.pop()?;
            if e & 1 == 1 {
                return Some((e >> 1) as u32);
            }
            match self.nodes[(e >> 1) as usize].kind {
                Kind::Inner(l, r) => {
                    for c in [l, r] {
                        let n = &self.nodes[c as usize];
                        heap.push((
                            Reverse(Ordered(box_d2(n.lo, n.hi, q))),
                            Reverse((c as u64) << 1),
                        ));
                    }
                }
                Kind::Leaf(a, b) => {
                    for &v in &self.order[a as usize..b as usize] {
                        heap.push((
                            Reverse(Ordered(dist2(pts[v as usize], q))),
                            Reverse(((v as u64) << 1) | 1),
                        ));
                    }
                }
            }
        })
    }

    /// The points in the box `lo..hi`.
    pub fn in_box(&self, pts: &[V3], lo: V3, hi: V3, out: &mut Vec<u32>) {
        if self.nodes.is_empty() {
            return;
        }
        let mut stack = vec![0u32];
        while let Some(n) = stack.pop() {
            let node = &self.nodes[n as usize];
            if (0..3).any(|k| node.hi[k] < lo[k] || node.lo[k] > hi[k]) {
                continue;
            }
            match node.kind {
                Kind::Inner(l, r) => stack.extend([l, r]),
                Kind::Leaf(a, b) => out.extend(
                    self.order[a as usize..b as usize]
                        .iter()
                        .copied()
                        .filter(|&v| {
                            (0..3)
                                .all(|k| pts[v as usize][k] >= lo[k] && pts[v as usize][k] <= hi[k])
                        }),
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A graded cloud: points come out by distance, and a box holds what a
    /// scan finds in it.
    #[test]
    fn distance_order_and_boxes_match_a_scan() {
        let pts: Vec<V3> = (0..500)
            .map(|i| {
                let t = i as f64;
                let r = (t / 500.0).powi(3) * 10.0;
                [r * (t * 0.7).cos(), r * (t * 1.3).sin(), (t * 0.37) % 2.0]
            })
            .collect();
        let tree = PointTree::new(&pts);
        let q = [0.3, -0.2, 1.0];
        let order: Vec<u32> = tree.by_distance(&pts, q).collect();
        assert_eq!(order.len(), pts.len());
        for w in order.windows(2) {
            assert!(dist2(pts[w[0] as usize], q) <= dist2(pts[w[1] as usize], q));
        }
        let (lo, hi) = ([-1.0, -1.0, 0.0], [2.0, 0.5, 1.5]);
        let mut found = Vec::new();
        tree.in_box(&pts, lo, hi, &mut found);
        found.sort_unstable();
        let scan: Vec<u32> = (0..pts.len() as u32)
            .filter(|&v| (0..3).all(|k| pts[v as usize][k] >= lo[k] && pts[v as usize][k] <= hi[k]))
            .collect();
        assert_eq!(found, scan);
    }
}
