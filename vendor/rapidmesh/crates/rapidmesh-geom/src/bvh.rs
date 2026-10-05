//! A bounding-volume hierarchy over boxes: the one tree under the facet
//! index of a model, the discrete carriers and the tube centerlines. It
//! knows boxes only; what lies in them (triangles, segments) and how far a
//! point is from it is the caller's, passed to the queries as closures.
//!
//! Splits by the binned surface area heuristic (the split whose child boxes,
//! weighted by their counts, have the least area), which keeps long thin
//! primitives (swept tubes) from overlapping every node; the median of the
//! widest centroid axis where the heuristic finds nothing or the tree gets
//! deep. Each primitive sits in exactly one leaf; inner nodes carry the box
//! of their subtree, so the queries prune by a node lower bound.

use rapidmesh_exact::vector::{box_d2, sub, V3};

/// A node: the box of its subtree, its children (an inner node) or its range
/// of the primitive order (a leaf, `count > 0`; see [`Bvh::leaf`]). Children come after their
/// parent in the flat array, the right one not right after the left.
#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub lo: V3,
    pub hi: V3,
    pub left: u32,
    pub right: u32,
    pub start: u32,
    pub count: u32,
}

/// Primitives per leaf at most.
const LEAF_MAX: usize = 4;
/// Splits deeper than this take the median, so a traversal stack of 64
/// entries always suffices.
const SAH_DEPTH: usize = 40;
/// Centroid bins per axis of the surface area heuristic.
const SAH_BINS: usize = 16;

#[derive(Debug, Clone, Default)]
pub struct Bvh {
    /// Box per primitive (by primitive index).
    boxes: Vec<(V3, V3)>,
    /// Primitive indices grouped by leaf (a permutation of `0..n`).
    order: Vec<u32>,
    nodes: Vec<Node>,
}

impl Bvh {
    /// The tree over the primitives with these boxes (primitive `i` has box
    /// `boxes[i]`); `centroids` place them for the splits.
    pub fn build(boxes: Vec<(V3, V3)>, centroids: &[V3]) -> Bvh {
        let mut order: Vec<u32> = (0..boxes.len() as u32).collect();
        let mut nodes: Vec<Node> = Vec::new();
        if !boxes.is_empty() {
            let b = Build {
                boxes: &boxes,
                centroids,
            };
            b.node(&mut order, 0, boxes.len(), 0, &mut nodes);
        }
        Bvh {
            boxes,
            order,
            nodes,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The primitives of leaf `n`.
    pub fn leaf(&self, n: &Node) -> &[u32] {
        &self.order[n.start as usize..(n.start + n.count) as usize]
    }

    /// The least `value` over the primitives, by branch and bound: `bound`
    /// gives a lower bound of the values in node `ni`, `value` the value of
    /// a primitive (it may return anything not below the best so far when
    /// the primitive cannot beat it). Starts from `best` and returns the
    /// primitive that went below it, if any, with its value. Visits the
    /// child of the lower bound first (the left on a tie); a primitive
    /// replaces the best only when strictly below it.
    pub fn minimize(
        &self,
        best: f64,
        bound: impl Fn(usize) -> f64,
        mut value: impl FnMut(u32, f64) -> f64,
    ) -> (Option<u32>, f64) {
        let mut out = (None, best);
        if !self.nodes.is_empty() {
            self.minimize_rec(0, &bound, &mut value, &mut out);
        }
        out
    }

    fn minimize_rec(
        &self,
        ni: usize,
        bound: &impl Fn(usize) -> f64,
        value: &mut impl FnMut(u32, f64) -> f64,
        best: &mut (Option<u32>, f64),
    ) {
        if bound(ni) >= best.1 {
            return;
        }
        let n = &self.nodes[ni];
        if n.count > 0 {
            for &i in self.leaf(n) {
                let v = value(i, best.1);
                if v < best.1 {
                    *best = (Some(i), v);
                }
            }
            return;
        }
        let (l, r) = (n.left as usize, n.right as usize);
        if bound(l) <= bound(r) {
            self.minimize_rec(l, bound, value, best);
            self.minimize_rec(r, bound, value, best);
        } else {
            self.minimize_rec(r, bound, value, best);
            self.minimize_rec(l, bound, value, best);
        }
    }

    /// The primitive nearest `p` by `d2`, a squared distance at least that
    /// to its box, among the primitives `d2` gives a value for and below
    /// `limit2`; with its squared distance.
    pub fn nearest(
        &self,
        p: V3,
        limit2: f64,
        mut d2: impl FnMut(u32) -> Option<f64>,
    ) -> Option<(u32, f64)> {
        let (i, d) = self.minimize(
            limit2,
            |ni| box_d2(self.nodes[ni].lo, self.nodes[ni].hi, p),
            |i, _| d2(i).unwrap_or(f64::INFINITY),
        );
        i.map(|i| (i, d))
    }

    /// Primitives whose box, grown by `pad`, meets the segment `a -> b`,
    /// appended to `out` (unsorted, no duplicates).
    pub fn along_segment(&self, a: V3, b: V3, pad: f64, out: &mut Vec<u32>) {
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
                for &i in self.leaf(n) {
                    let (lo, hi) = &self.boxes[i as usize];
                    if seg.interval(lo, hi, pad).is_some() {
                        out.push(i);
                    }
                }
                continue;
            }
            stack[sp] = n.right;
            stack[sp + 1] = n.left;
            sp += 2;
        }
    }
}

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

/// The inputs of a build.
struct Build<'a> {
    boxes: &'a [(V3, V3)],
    centroids: &'a [V3],
}

impl Build<'_> {
    /// Builds the node spanning `order[start..end]` and returns its index.
    fn node(
        &self,
        order: &mut [u32],
        start: usize,
        end: usize,
        depth: usize,
        nodes: &mut Vec<Node>,
    ) -> u32 {
        let mut bb = EMPTY;
        for &i in &order[start..end] {
            grow(&mut bb, &self.boxes[i as usize]);
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
        for &i in &order[start..end] {
            let c = self.centroids[i as usize];
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
            for &i in &order[start..end] {
                let b = &mut bins[bin(axis, &self.centroids[i as usize])];
                grow(&mut b.0, &self.boxes[i as usize]);
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
