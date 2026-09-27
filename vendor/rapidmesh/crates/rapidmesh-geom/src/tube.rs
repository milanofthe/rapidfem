//! The tube carrier's centerline: a polyline with an AABB-tree accelerator
//! for closest-point queries (the same shape as
//! [`crate::discrete::DiscreteSurface`]'s triangle tree). The projection
//! oracle of a `SurfaceKind::Tube` runs millions of times per mesh (sphere
//! tracing, POCS, wall predicates); a plain linear scan over a helix path
//! made the coil geometries 10x slower than their discrete-carrier
//! predecessor, and flat prune passes only doubled the scan.

use crate::vec3::{dot, sub, V3};

/// A polyline sweep centerline with an AABB segment tree for closest queries.
#[derive(Debug)]
pub struct TubePath {
    /// The ordered path nodes.
    pub pts: Vec<V3>,
    /// Flat AABB tree (node = (lo, hi, left, right | leaf segment range)).
    nodes: Vec<Node>,
    /// Segment order for leaf ranges.
    order: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct Node {
    lo: V3,
    hi: V3,
    /// Child indices, or `(start, !count)` leaf encoding when `right < 0`.
    left: i32,
    right: i32,
}

fn d2(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

fn box_d2(lo: V3, hi: V3, p: V3) -> f64 {
    let mut s = 0.0;
    for k in 0..3 {
        let d = (lo[k] - p[k]).max(0.0).max(p[k] - hi[k]);
        s += d * d;
    }
    s
}

/// Closest point on segment `a -> b` to `p`.
fn closest_on_seg(p: V3, a: V3, b: V3) -> V3 {
    let ab = sub(b, a);
    let len2 = dot(ab, ab);
    let t = if len2 > 0.0 {
        (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    std::array::from_fn(|k| a[k] + t * ab[k])
}

impl TubePath {
    /// Builds the segment tree. `pts` needs at least 2 nodes.
    pub fn new(pts: Vec<V3>) -> TubePath {
        assert!(pts.len() >= 2, "tube path needs at least 2 nodes");
        let n_seg = pts.len() - 1;
        let mids: Vec<V3> = pts
            .windows(2)
            .map(|w| std::array::from_fn(|k| 0.5 * (w[0][k] + w[1][k])))
            .collect();
        let mut order: Vec<u32> = (0..n_seg as u32).collect();
        let mut nodes = Vec::new();
        build(&pts, &mids, &mut order, 0, n_seg, &mut nodes);
        TubePath { pts, nodes, order }
    }

    /// Closest point on the polyline to `p` (exact; the tree only prunes).
    pub fn closest(&self, p: V3) -> V3 {
        let mut best = (f64::INFINITY, self.pts[0]);
        self.closest_rec(0, p, &mut best);
        best.1
    }

    /// Calls `f` with the index of every path segment whose AABB overlaps
    /// the AABB of query segment `a -> b` inflated by `pad` (a conservative
    /// superset of the segments within `pad` of the query). Zero-alloc; the
    /// analytic tube-crossing solver collects its candidate quadratics here.
    pub fn for_segments_near(&self, a: V3, b: V3, pad: f64, f: &mut impl FnMut(usize)) {
        let lo: V3 = std::array::from_fn(|k| a[k].min(b[k]) - pad);
        let hi: V3 = std::array::from_fn(|k| a[k].max(b[k]) + pad);
        self.near_rec(0, lo, hi, f);
    }

    fn near_rec(&self, ni: usize, lo: V3, hi: V3, f: &mut impl FnMut(usize)) {
        let n = self.nodes[ni];
        if (0..3).any(|k| n.hi[k] < lo[k] || n.lo[k] > hi[k]) {
            return;
        }
        if n.right < 0 {
            let (start, count) = (n.left as usize, (!n.right) as usize);
            for &si in &self.order[start..start + count] {
                f(si as usize);
            }
            return;
        }
        self.near_rec(n.left as usize, lo, hi, f);
        self.near_rec(n.right as usize, lo, hi, f);
    }

    fn closest_rec(&self, ni: usize, p: V3, best: &mut (f64, V3)) {
        let n = self.nodes[ni];
        if box_d2(n.lo, n.hi, p) >= best.0 {
            return;
        }
        if n.right < 0 {
            let (start, count) = (n.left as usize, (!n.right) as usize);
            for &si in &self.order[start..start + count] {
                let s = si as usize;
                let q = closest_on_seg(p, self.pts[s], self.pts[s + 1]);
                let dd = d2(p, q);
                if dd < best.0 {
                    *best = (dd, q);
                }
            }
            return;
        }
        let (l, r) = (n.left as usize, n.right as usize);
        let (dl, dr) = (
            box_d2(self.nodes[l].lo, self.nodes[l].hi, p),
            box_d2(self.nodes[r].lo, self.nodes[r].hi, p),
        );
        if dl <= dr {
            self.closest_rec(l, p, best);
            self.closest_rec(r, p, best);
        } else {
            self.closest_rec(r, p, best);
            self.closest_rec(l, p, best);
        }
    }
}

fn build(
    pts: &[V3],
    mids: &[V3],
    order: &mut [u32],
    start: usize,
    end: usize,
    nodes: &mut Vec<Node>,
) -> i32 {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for &si in &order[start..end] {
        for &q in &[pts[si as usize], pts[si as usize + 1]] {
            for k in 0..3 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
    }
    let idx = nodes.len() as i32;
    nodes.push(Node {
        lo,
        hi,
        left: 0,
        right: 0,
    });
    let count = end - start;
    if count <= 8 {
        nodes[idx as usize].left = start as i32;
        nodes[idx as usize].right = !(count as i32);
        return idx;
    }
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).partial_cmp(&(hi[b] - lo[b])).unwrap())
        .unwrap();
    order[start..end].sort_unstable_by(|&a, &b| {
        mids[a as usize][axis]
            .partial_cmp(&mids[b as usize][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = start + count / 2;
    let l = build(pts, mids, order, start, mid, nodes);
    let r = build(pts, mids, order, mid, end, nodes);
    nodes[idx as usize].left = l;
    nodes[idx as usize].right = r;
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::dist;

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
                let q = closest_on_seg(p, w[0], w[1]);
                if d2(p, q) < best.1 {
                    best = (q, d2(p, q));
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
