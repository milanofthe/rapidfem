//! The tube carrier's centerline: a polyline with a [`Bvh`] over its
//! segments for closest-point queries. The projection
//! onto a `Surface::Tube` runs for every point the mesher puts on it; a
//! linear scan over a helix path would cost the whole path each time.

use crate::bvh::Bvh;
use rapidmesh_exact::vector::{bbox, closest_on_segment, dist2, V3};

/// A polyline sweep centerline with an AABB segment tree for closest queries.
#[derive(Debug)]
pub struct TubePath {
    /// The ordered path nodes.
    pub pts: Vec<V3>,
    /// The tree over the segments (segment `i` from node `i` to `i + 1`).
    bvh: Bvh,
}

impl TubePath {
    /// Builds the segment tree. `pts` needs at least 2 nodes.
    pub fn new(pts: Vec<V3>) -> TubePath {
        assert!(pts.len() >= 2, "tube path needs at least 2 nodes");
        let mids: Vec<V3> = pts
            .windows(2)
            .map(|w| std::array::from_fn(|k| 0.5 * (w[0][k] + w[1][k])))
            .collect();
        let boxes = pts.windows(2).map(bbox).collect();
        TubePath {
            bvh: Bvh::build(boxes, &mids),
            pts,
        }
    }

    /// The closest point of the polyline to `p` and its segment (exact; the
    /// tree only prunes).
    fn closest_with_segment(&self, p: V3) -> (V3, usize) {
        let on = |s: usize| closest_on_segment(p, self.pts[s], self.pts[s + 1]);
        let s = self
            .bvh
            .nearest(p, f64::INFINITY, |s| Some(dist2(p, on(s as usize))))
            .map_or(0, |(s, _)| s as usize);
        (on(s), s)
    }

    /// Closest point on the polyline to `p` (exact; the tree only prunes).
    pub fn closest(&self, p: V3) -> V3 {
        self.closest_with_segment(p).0
    }

    /// The segment (from node `i` to `i + 1`) with the closest point of the
    /// polyline to `p`.
    pub fn closest_segment(&self, p: V3) -> usize {
        self.closest_with_segment(p).1
    }

    /// The closest point of the path to `p` and its segment, walking from
    /// segment `start` along the path while the segments come nearer (a
    /// search where the answer lies close, not over the whole path).
    pub fn closest_near(&self, p: V3, start: usize) -> (V3, usize) {
        let n_seg = self.pts.len() - 1;
        let at = |s: usize| {
            let q = closest_on_segment(p, self.pts[s], self.pts[s + 1]);
            (dist2(q, p), q)
        };
        let mut s = start.min(n_seg - 1);
        let (mut best, mut q) = at(s);
        loop {
            let next = [s.wrapping_sub(1), s + 1]
                .into_iter()
                .filter(|&k| k < n_seg)
                .map(|k| (k, at(k)))
                .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0));
            match next {
                Some((k, (d, x))) if d < best => {
                    (s, best, q) = (k, d, x);
                }
                _ => return (q, s),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_exact::vector::dist;

    #[test]
    fn tree_closest_matches_linear_scan() {
        // helix-like path
        let pts: Vec<V3> = (0..=180)
            .map(|i| {
                let t = i as f64 / 28.0 * std::f64::consts::TAU;
                [0.8 * t.cos(), 0.8 * t.sin(), 0.42 * i as f64 / 28.0]
            })
            .collect();
        let tube = TubePath::new(pts.clone());
        let linear = |p: V3| -> V3 {
            let mut best = (pts[0], f64::MAX);
            for w in pts.windows(2) {
                let q = closest_on_segment(p, w[0], w[1]);
                if dist2(p, q) < best.1 {
                    best = (q, dist2(p, q));
                }
            }
            best.0
        };
        let mut s = 12345u64;
        let mut frac = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..500 {
            let p: V3 = [4.0 * frac() - 2.0, 4.0 * frac() - 2.0, 4.0 * frac()];
            let (a, b) = (tube.closest(p), linear(p));
            assert!(dist(a, b) < 1e-9, "tree {a:?} vs linear {b:?} at {p:?}");
        }
    }
}
