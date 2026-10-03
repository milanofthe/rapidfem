//! One bounding-volume hierarchy over a triangle set, for `O(log F)`
//! nearest-facet distance, segment candidates and graded distance fields.
//!
//! The model builds it once over its PLC facets ([`crate::Model::index`]);
//! the sizing field, the region query and the mesher's oracle all query that
//! one index. Per-facet values (size targets) are not part of the geometry:
//! a [`Targets`] pairs values with the tree's per-node minima, so one tree
//! serves every set of targets.
//!
//! Median/SAH splits on facet centroids: each facet sits in exactly one leaf,
//! internal nodes carry the subtree box, so the queries prune by a node
//! lower bound (branch and bound).

use rapidmesh_csg::Tri;
use rapidmesh_geom::vec3::{dot, sub, V3};

/// Squared distance from point `p` to triangle `t` (closest-point clamp).
pub fn point_tri_dist2(p: V3, t: &Tri) -> f64 {
    let (a, b, c) = (t.v[0], t.v[1], t.v[2]);
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return dot(ap, ap);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return dot(bp, bp);
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return dot(cp, cp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q: V3 = std::array::from_fn(|k| a[k] + v * ab[k]);
        return dot(sub(p, q), sub(p, q));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q: V3 = std::array::from_fn(|k| a[k] + w * ac[k]);
        return dot(sub(p, q), sub(p, q));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let q: V3 = std::array::from_fn(|k| b[k] + w * (c[k] - b[k]));
        return dot(sub(p, q), sub(p, q));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q: V3 = std::array::from_fn(|k| a[k] + ab[k] * v + ac[k] * w);
    dot(sub(p, q), sub(p, q))
}

/// Squared distance from `p` to the axis-aligned box `[lo, hi]` (0 if inside).
fn box_dist2(lo: V3, hi: V3, p: V3) -> f64 {
    let mut d2 = 0.0;
    for k in 0..3 {
        let e = if p[k] < lo[k] {
            lo[k] - p[k]
        } else if p[k] > hi[k] {
            p[k] - hi[k]
        } else {
            0.0
        };
        d2 += e * e;
    }
    d2
}

struct Node {
    lo: V3,
    hi: V3,
    /// Child node indices (a flat-array BVH: the left subtree spans many slots,
    /// so the right child is NOT `left + 1`). Unused for a leaf (`count > 0`).
    left: u32,
    right: u32,
    start: u32,
    count: u32,
}

/// Facets per leaf at most.
const LEAF_MAX: usize = 4;

pub struct FacetBvh {
    tris: Vec<Tri>,
    /// Bounding box per facet (by facet index).
    boxes: Vec<(V3, V3)>,
    /// Facet indices grouped by leaf (a permutation of `0..tris.len()`).
    order: Vec<u32>,
    nodes: Vec<Node>,
}

impl std::fmt::Debug for FacetBvh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FacetBvh({} facets, {} nodes)",
            self.tris.len(),
            self.nodes.len()
        )
    }
}

/// Values per facet of a [`FacetBvh`] (size targets) with their minimum per
/// tree node, the lower bound of the graded queries.
pub struct Targets {
    values: Vec<f64>,
    node_min: Vec<f64>,
}

impl Targets {
    /// `values[i]` for facet `i` of `bvh`.
    pub fn new(bvh: &FacetBvh, values: Vec<f64>) -> Targets {
        assert_eq!(values.len(), bvh.tris.len(), "one value per facet");
        let mut node_min = vec![f64::INFINITY; bvh.nodes.len()];
        // Children come after their parent in the flat array, so a pass
        // from the back sees every child before its parent.
        for ni in (0..bvh.nodes.len()).rev() {
            let n = &bvh.nodes[ni];
            node_min[ni] = if n.count > 0 {
                bvh.order[n.start as usize..(n.start + n.count) as usize]
                    .iter()
                    .map(|&fi| values[fi as usize])
                    .fold(f64::INFINITY, f64::min)
            } else {
                node_min[n.left as usize].min(node_min[n.right as usize])
            };
        }
        Targets { values, node_min }
    }
}

impl FacetBvh {
    /// The facets, in build order.
    pub fn tris(&self) -> &[Tri] {
        &self.tris
    }

    /// Builds a BVH over `tris` (facet `i` is `tris[i]`). Empty input is
    /// queryable (`nearest_dist` returns INFINITY).
    pub fn build(tris: &[Tri]) -> FacetBvh {
        let tris = tris.to_vec();
        let centroids: Vec<V3> = tris
            .iter()
            .map(|t| std::array::from_fn(|k| (t.v[0][k] + t.v[1][k] + t.v[2][k]) / 3.0))
            .collect();
        let boxes: Vec<(V3, V3)> = tris
            .iter()
            .map(|t| {
                (
                    std::array::from_fn(|k| t.v[0][k].min(t.v[1][k]).min(t.v[2][k])),
                    std::array::from_fn(|k| t.v[0][k].max(t.v[1][k]).max(t.v[2][k])),
                )
            })
            .collect();
        let mut order: Vec<u32> = (0..tris.len() as u32).collect();
        let mut nodes: Vec<Node> = Vec::new();
        if !tris.is_empty() {
            let b = Build {
                boxes: &boxes,
                centroids: &centroids,
            };
            b.node(&mut order, 0, tris.len(), 0, &mut nodes);
        }
        FacetBvh {
            tris,
            boxes,
            order,
            nodes,
        }
    }

    /// Distance from `p` to the nearest facet (INFINITY if empty).
    pub fn nearest_dist(&self, p: V3) -> f64 {
        self.nearest(p).map_or(f64::INFINITY, |(_, d)| d)
    }

    /// The distance from `p` to the nearest facet, or `r` if none is
    /// closer: the search skips everything beyond `r`, so a small `r` is
    /// cheap. Exact whenever the result is below `r`.
    pub fn nearest_dist_within(&self, p: V3, r: f64) -> f64 {
        if self.nodes.is_empty() || !(r < f64::INFINITY) {
            return self.nearest_dist(p).min(r);
        }
        let mut best = (u32::MAX, r * r);
        self.nearest_rec(0, p, &|_| true, &mut best);
        if best.0 == u32::MAX {
            r
        } else {
            best.1.sqrt()
        }
    }

    /// The nearest facet to `p` (its index in the build input) and its
    /// distance, `None` if empty.
    pub fn nearest(&self, p: V3) -> Option<(u32, f64)> {
        self.nearest_where(p, &|_| true)
    }

    /// The nearest facet to `p` among those `keep` accepts (by index in the
    /// build input) and its distance, `None` if there is none.
    pub fn nearest_where(&self, p: V3, keep: &dyn Fn(u32) -> bool) -> Option<(u32, f64)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = (u32::MAX, f64::INFINITY);
        self.nearest_rec(0, p, keep, &mut best);
        (best.0 != u32::MAX).then(|| (best.0, best.1.sqrt()))
    }

    fn nearest_rec(&self, ni: usize, p: V3, keep: &dyn Fn(u32) -> bool, best: &mut (u32, f64)) {
        let n = &self.nodes[ni];
        if box_dist2(n.lo, n.hi, p) >= best.1 {
            return;
        }
        if n.count > 0 {
            for &fi in &self.order[n.start as usize..(n.start + n.count) as usize] {
                if !keep(fi) {
                    continue;
                }
                let d2 = point_tri_dist2(p, &self.tris[fi as usize]);
                if d2 < best.1 {
                    *best = (fi, d2);
                }
            }
            return;
        }
        // Visit the nearer child first so the farther one prunes more often.
        let (l, r) = (n.left as usize, n.right as usize);
        let dl = box_dist2(self.nodes[l].lo, self.nodes[l].hi, p);
        let dr = box_dist2(self.nodes[r].lo, self.nodes[r].hi, p);
        if dl <= dr {
            self.nearest_rec(l, p, keep, best);
            self.nearest_rec(r, p, keep, best);
        } else {
            self.nearest_rec(r, p, keep, best);
            self.nearest_rec(l, p, keep, best);
        }
    }

    /// Facet indices whose bounding box, grown by `pad`, meets the segment
    /// `a -> b`, appended to `out` (unsorted, no duplicates). A superset of
    /// the facets within `pad` of the segment: the candidate set for
    /// segment-surface crossings.
    pub fn facets_near_segment(&self, a: V3, b: V3, pad: f64, out: &mut Vec<u32>) {
        if self.nodes.is_empty() || !(pad >= 0.0) {
            return;
        }
        let seg = Seg::new(a, b);
        let mut stack = [0u32; 64];
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let n = &self.nodes[stack[sp] as usize];
            if seg.interval(&n.lo, &n.hi, pad).is_none() {
                continue;
            }
            if n.count > 0 {
                for &fi in &self.order[n.start as usize..(n.start + n.count) as usize] {
                    let (lo, hi) = &self.boxes[fi as usize];
                    if seg.interval(lo, hi, pad).is_some() {
                        out.push(fi);
                    }
                }
                continue;
            }
            stack[sp] = n.right;
            stack[sp + 1] = n.left;
            sp += 2;
        }
    }

    /// The finest target among the facets within `r` of `p` (INFINITY if
    /// none).
    pub fn min_target_within(&self, targets: &Targets, p: V3, r: f64) -> f64 {
        let mut best = f64::INFINITY;
        if self.nodes.is_empty() || !(r >= 0.0) {
            return best;
        }
        let r2 = r * r;
        let mut stack = [0u32; 64];
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let ni = stack[sp] as usize;
            let n = &self.nodes[ni];
            if targets.node_min[ni] >= best || box_dist2(n.lo, n.hi, p) > r2 {
                continue;
            }
            if n.count > 0 {
                for &fi in &self.order[n.start as usize..(n.start + n.count) as usize] {
                    let t = targets.values[fi as usize];
                    if t < best && point_tri_dist2(p, &self.tris[fi as usize]) <= r2 {
                        best = t;
                    }
                }
                continue;
            }
            stack[sp] = n.right;
            stack[sp + 1] = n.left;
            sp += 2;
        }
        best
    }

    /// `min over facets ( target + grading * dist(p, facet) )`: the graded
    /// distance field that grows the sizing field from the fine wall targets.
    pub fn graded_min(&self, targets: &Targets, p: V3, grading: f64) -> f64 {
        if self.nodes.is_empty() {
            return f64::INFINITY;
        }
        let mut best = f64::INFINITY;
        self.graded_rec(targets, 0, p, grading, &mut best);
        best
    }

    /// [`FacetBvh::graded_min`], or `bound` if it is not below: the search
    /// skips every subtree that cannot go below `bound`. Exact whenever the
    /// result is below `bound`.
    pub fn graded_min_within(&self, targets: &Targets, p: V3, grading: f64, bound: f64) -> f64 {
        if self.nodes.is_empty() {
            return bound.min(f64::INFINITY);
        }
        let mut best = bound;
        self.graded_rec(targets, 0, p, grading, &mut best);
        best
    }

    fn graded_rec(&self, targets: &Targets, ni: usize, p: V3, grading: f64, best: &mut f64) {
        let n = &self.nodes[ni];
        // Lower bound for anything in this subtree: the finest target plus the
        // graded distance to the subtree box.
        let bound = targets.node_min[ni] + grading * box_dist2(n.lo, n.hi, p).sqrt();
        if bound >= *best {
            return;
        }
        if n.count > 0 {
            for &fi in &self.order[n.start as usize..(n.start + n.count) as usize] {
                let v = targets.values[fi as usize]
                    + grading * point_tri_dist2(p, &self.tris[fi as usize]).sqrt();
                if v < *best {
                    *best = v;
                }
            }
            return;
        }
        let (l, r) = (n.left as usize, n.right as usize);
        let bl =
            targets.node_min[l] + grading * box_dist2(self.nodes[l].lo, self.nodes[l].hi, p).sqrt();
        let br =
            targets.node_min[r] + grading * box_dist2(self.nodes[r].lo, self.nodes[r].hi, p).sqrt();
        if bl <= br {
            self.graded_rec(targets, l, p, grading, best);
            self.graded_rec(targets, r, p, grading, best);
        } else {
            self.graded_rec(targets, r, p, grading, best);
            self.graded_rec(targets, l, p, grading, best);
        }
    }
}

/// True if the segment `a + t d`, `t` in `[0, 1]`, meets the box
/// `[lo - pad, hi + pad]` (slab test).
/// A segment `a + t d`, `t` in `[0, 1]`, with its inverse direction for
/// slab tests.
struct Seg {
    a: V3,
    d: V3,
    inv: V3,
}

impl Seg {
    fn new(a: V3, b: V3) -> Seg {
        let d = sub(b, a);
        Seg {
            a,
            d,
            inv: d.map(|x| 1.0 / x),
        }
    }

    /// The parameter interval inside the box grown by `pad`, `None` when
    /// the segment misses it. Boxes are closed, so a touch counts.
    fn interval(&self, lo: &V3, hi: &V3, pad: f64) -> Option<(f64, f64)> {
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        for k in 0..3 {
            let (l, h) = (lo[k] - pad, hi[k] + pad);
            if self.d[k] == 0.0 {
                if self.a[k] < l || self.a[k] > h {
                    return None;
                }
                continue;
            }
            let (mut u, mut v) = ((l - self.a[k]) * self.inv[k], (h - self.a[k]) * self.inv[k]);
            if u > v {
                std::mem::swap(&mut u, &mut v);
            }
            t0 = t0.max(u);
            t1 = t1.min(v);
            if t0 > t1 {
                return None;
            }
        }
        Some((t0, t1))
    }
}

/// Splits deeper than this take the median, so a traversal stack of 64
/// entries always suffices.
const SAH_DEPTH: usize = 40;
/// Centroid bins per axis of the surface area heuristic.
const SAH_BINS: usize = 16;

fn area(lo: &V3, hi: &V3) -> f64 {
    let e: V3 = std::array::from_fn(|k| (hi[k] - lo[k]).max(0.0));
    e[0] * e[1] + e[1] * e[2] + e[2] * e[0]
}

fn grow(b: &mut (V3, V3), o: &(V3, V3)) {
    for k in 0..3 {
        b.0[k] = b.0[k].min(o.0[k]);
        b.1[k] = b.1[k].max(o.1[k]);
    }
}

const EMPTY: (V3, V3) = ([f64::MAX; 3], [f64::MIN; 3]);

/// The inputs of a BVH build.
struct Build<'a> {
    boxes: &'a [(V3, V3)],
    centroids: &'a [V3],
}

impl Build<'_> {
    /// Builds the node spanning `order[start..end]` and returns its index.
    /// Splits by the binned surface area heuristic (the split whose child
    /// boxes, weighted by their facet counts, have the least area), which
    /// keeps long thin facets (swept tubes) from overlapping every node; the
    /// median of the widest centroid axis where the heuristic finds nothing
    /// or the tree gets deep.
    fn node(
        &self,
        order: &mut [u32],
        start: usize,
        end: usize,
        depth: usize,
        nodes: &mut Vec<Node>,
    ) -> u32 {
        let mut bb = EMPTY;
        for &fi in &order[start..end] {
            grow(&mut bb, &self.boxes[fi as usize]);
        }
        let idx = nodes.len() as u32;
        let count = end - start;
        nodes.push(Node {
            lo: bb.0,
            hi: bb.1,
            left: 0,
            right: 0,
            start: start as u32,
            count: count as u32,
        });
        if count <= LEAF_MAX {
            return idx;
        }
        let mut cb = EMPTY;
        for &fi in &order[start..end] {
            let c = self.centroids[fi as usize];
            grow(&mut cb, &(c, c));
        }
        let mid = self
            .sah_split(order, start, end, &cb, depth)
            .unwrap_or_else(|| self.median_split(order, start, end, &cb));
        nodes[idx as usize].start = 0;
        nodes[idx as usize].count = 0;
        let left = self.node(order, start, mid, depth + 1, nodes);
        let right = self.node(order, mid, end, depth + 1, nodes);
        nodes[idx as usize].left = left;
        nodes[idx as usize].right = right;
        idx
    }

    /// Partitions at the best binned split; `None` when there is none (all
    /// centroids in one bin) or the tree is deep.
    fn sah_split(
        &self,
        order: &mut [u32],
        start: usize,
        end: usize,
        cb: &(V3, V3),
        depth: usize,
    ) -> Option<usize> {
        if depth >= SAH_DEPTH {
            return None;
        }
        let bin = |axis: usize, c: &V3| -> usize {
            let w = cb.1[axis] - cb.0[axis];
            (((c[axis] - cb.0[axis]) / w * SAH_BINS as f64) as usize).min(SAH_BINS - 1)
        };
        let mut best: Option<(f64, usize, usize)> = None; // (cost, axis, split bin)
        for axis in 0..3 {
            if !(cb.1[axis] - cb.0[axis] > 0.0) {
                continue;
            }
            let mut bins = [(EMPTY, 0usize); SAH_BINS];
            for &fi in &order[start..end] {
                let b = &mut bins[bin(axis, &self.centroids[fi as usize])];
                grow(&mut b.0, &self.boxes[fi as usize]);
                b.1 += 1;
            }
            // Areas and counts to the right of each split, then a sweep from
            // the left.
            let mut right = [(0.0f64, 0usize); SAH_BINS];
            let (mut acc, mut n) = (EMPTY, 0usize);
            for s in (1..SAH_BINS).rev() {
                grow(&mut acc, &bins[s].0);
                n += bins[s].1;
                right[s] = (if n > 0 { area(&acc.0, &acc.1) } else { 0.0 }, n);
            }
            let (mut acc, mut n) = (EMPTY, 0usize);
            for s in 1..SAH_BINS {
                grow(&mut acc, &bins[s - 1].0);
                n += bins[s - 1].1;
                let (ra, rn) = right[s];
                if n == 0 || rn == 0 {
                    continue;
                }
                let cost = area(&acc.0, &acc.1) * n as f64 + ra * rn as f64;
                if best.is_none_or(|b| cost < b.0) {
                    best = Some((cost, axis, s));
                }
            }
        }
        let (_, axis, s) = best?;
        // Partition: bins below `s` to the left.
        let slice = &mut order[start..end];
        let mut i = 0;
        for j in 0..slice.len() {
            if bin(axis, &self.centroids[slice[j] as usize]) < s {
                slice.swap(i, j);
                i += 1;
            }
        }
        Some(start + i)
    }

    /// Partitions at the median centroid of the widest centroid axis.
    fn median_split(&self, order: &mut [u32], start: usize, end: usize, cb: &(V3, V3)) -> usize {
        let axis = (0..3)
            .max_by(|&a, &b| (cb.1[a] - cb.0[a]).total_cmp(&(cb.1[b] - cb.0[b])))
            .unwrap_or(0);
        let count = end - start;
        order[start..end].select_nth_unstable_by(count / 2, |&a, &b| {
            self.centroids[a as usize][axis].total_cmp(&self.centroids[b as usize][axis])
        });
        start + count / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targeted(f: &[(Tri, f64)]) -> (FacetBvh, Targets) {
        let bvh = FacetBvh::build(&f.iter().map(|x| x.0).collect::<Vec<_>>());
        let tg = Targets::new(&bvh, f.iter().map(|x| x.1).collect());
        (bvh, tg)
    }

    fn tri(a: V3, b: V3, c: V3, target: f64) -> (Tri, f64) {
        (Tri::new(a, b, c), target)
    }

    fn brute_nearest(facets: &[(Tri, f64)], p: V3) -> f64 {
        facets
            .iter()
            .map(|(t, _)| point_tri_dist2(p, t))
            .fold(f64::MAX, f64::min)
            .sqrt()
    }

    fn brute_graded(facets: &[(Tri, f64)], p: V3, g: f64) -> f64 {
        facets
            .iter()
            .map(|(t, tg)| tg + g * point_tri_dist2(p, t).sqrt())
            .fold(f64::MAX, f64::min)
    }

    fn box_facets() -> Vec<(Tri, f64)> {
        // Two triangles per face of the unit cube, target varying per face.
        let mut f = Vec::new();
        let c = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.1),
            ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], 0.1),
            ([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], 0.5),
            ([0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], 0.5),
            ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0], 0.3),
            ([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], 0.3),
        ];
        for (a, b, cc, t) in c {
            f.push(tri(a, b, cc, t));
        }
        f
    }

    #[test]
    fn nearest_matches_brute() {
        let f = box_facets();
        let (bvh, _) = targeted(&f);
        for p in [
            [0.5, 0.5, 0.5],
            [0.1, 0.2, 0.9],
            [-1.0, 0.5, 0.5],
            [0.5, 0.5, 2.0],
        ] {
            let got = bvh.nearest_dist(p);
            let want = brute_nearest(&f, p);
            assert!(
                (got - want).abs() < 1e-12,
                "nearest at {p:?}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn graded_min_matches_brute() {
        let f = box_facets();
        let (bvh, tg) = targeted(&f);
        let g = 0.5;
        for p in [
            [0.5, 0.5, 0.5],
            [0.1, 0.2, 0.9],
            [0.5, 0.5, 0.05],
            [0.95, 0.5, 0.5],
        ] {
            let got = bvh.graded_min(&tg, p, g);
            let want = brute_graded(&f, p, g);
            assert!(
                (got - want).abs() < 1e-12,
                "graded at {p:?}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn deep_tree_matches_brute() {
        // Many facets force a multi-level tree, so the flat-array child indices
        // (right != left+1) are exercised: a grid of small triangles at varied
        // depths with varied targets.
        let mut f = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                let (x, y) = (i as f64 * 0.5, j as f64 * 0.5);
                let z = 0.1 * (i + j) as f64;
                let target = 0.05 + 0.02 * ((i * 7 + j) % 5) as f64;
                f.push(tri(
                    [x, y, z],
                    [x + 0.4, y, z],
                    [x, y + 0.4, z + 0.2],
                    target,
                ));
            }
        }
        let (bvh, tg) = targeted(&f);
        for p in [
            [1.3, 1.7, 0.5],
            [-2.0, 0.5, 1.0],
            [3.1, 3.2, 0.0],
            [0.25, 0.25, 5.0],
            [1.0, 2.0, -1.0],
        ] {
            assert!(
                (bvh.nearest_dist(p) - brute_nearest(&f, p)).abs() < 1e-9,
                "nearest at {p:?}"
            );
            assert!(
                (bvh.graded_min(&tg, p, 0.5) - brute_graded(&f, p, 0.5)).abs() < 1e-9,
                "graded at {p:?}"
            );
        }
    }

    #[test]
    fn segment_candidates_cover_every_facet_near_the_segment() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut f = Vec::new();
        for _ in 0..300 {
            let a: V3 = std::array::from_fn(|_| 4.0 * rnd());
            let b: V3 = std::array::from_fn(|k| a[k] + 0.3 * (rnd() - 0.5));
            let c: V3 = std::array::from_fn(|k| a[k] + 0.3 * (rnd() - 0.5));
            f.push(tri(a, b, c, 0.1));
        }
        let (bvh, _) = targeted(&f);
        let mut out = Vec::new();
        for _ in 0..500 {
            let a: V3 = std::array::from_fn(|_| 5.0 * rnd() - 0.5);
            let b: V3 = std::array::from_fn(|k| a[k] + 2.0 * (rnd() - 0.5));
            let pad = 0.1 * rnd();
            out.clear();
            bvh.facets_near_segment(a, b, pad, &mut out);
            let mut sorted = out.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), out.len(), "duplicates");
            // Every facet within `pad` of a dense sample of the segment is a
            // candidate.
            for (fi, (t, _)) in f.iter().enumerate() {
                let near = (0..=200).any(|i| {
                    let u = i as f64 / 200.0;
                    let p: V3 = std::array::from_fn(|k| a[k] + u * (b[k] - a[k]));
                    point_tri_dist2(p, t).sqrt() < pad
                });
                if near {
                    assert!(out.contains(&(fi as u32)), "facet {fi} missed");
                }
            }
        }
    }

    #[test]
    fn empty_is_safe() {
        let bvh = FacetBvh::build(&[]);
        assert_eq!(bvh.nearest_dist([0.0; 3]), f64::INFINITY);
    }
}
