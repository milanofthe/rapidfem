//! A quadtree-cached scalar field: an arbitrary field `f(x, y)` (e.g. a target
//! edge length, or a distance to geometry) sampled ONCE onto an adaptively
//! refined quadtree, then evaluated in O(depth) with bilinear interpolation
//! inside each leaf. The tree is fine where the field is small (near features)
//! and coarse where it is large -- the natural structure for graded meshing and
//! adaptive refinement (AMR), and far cheaper than re-evaluating an expensive
//! field at every one of the mesher's millions of queries.
//!
//! Distinct from the mesher's own gradient-limited *sizing field* (what size
//! to mesh to, see [`crate::domain`]): a `QuadtreeField` is a generic
//! background cache that any field can be baked into for fast lookup.

type P2 = [f64; 2];

enum Node {
    /// Field values at the four corners (SW, SE, NW, NE) of the leaf cell.
    Leaf([f64; 4]),
    /// Four children in row-major quadrant order: SW, SE, NW, NE.
    Branch(Box<[Node; 4]>),
}

/// A scalar field baked onto a square quadtree over a bounding box.
pub struct QuadtreeField {
    lo: P2,
    size: f64,
    root: Node,
}

impl QuadtreeField {
    /// Bake `f` onto a quadtree over `[lo, hi]`, refining each cell until it is
    /// no larger than the field value there (so the field is resolved to within
    /// its own gradient), down to `min_cell` and capped at `max_depth` levels.
    pub fn from_fn(lo: P2, hi: P2, min_cell: f64, max_depth: u32, f: impl Fn(P2) -> f64) -> Self {
        let size = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(min_cell.max(1e-12));
        let root = Self::build(lo, size, min_cell, max_depth, &f);
        QuadtreeField { lo, size, root }
    }

    fn build(o: P2, s: f64, min_cell: f64, depth: u32, f: &impl Fn(P2) -> f64) -> Node {
        let center = [o[0] + s * 0.5, o[1] + s * 0.5];
        if depth == 0 || s <= f(center).max(min_cell) {
            return Node::Leaf([
                f(o),
                f([o[0] + s, o[1]]),
                f([o[0], o[1] + s]),
                f([o[0] + s, o[1] + s]),
            ]);
        }
        let h = s * 0.5;
        Node::Branch(Box::new([
            Self::build(o, h, min_cell, depth - 1, f),
            Self::build([o[0] + h, o[1]], h, min_cell, depth - 1, f),
            Self::build([o[0], o[1] + h], h, min_cell, depth - 1, f),
            Self::build([o[0] + h, o[1] + h], h, min_cell, depth - 1, f),
        ]))
    }

    /// Field value at `p`: descend to its leaf (O(depth)) and bilinearly
    /// interpolate the corner values. Points outside the box clamp to the border.
    pub fn eval(&self, p: P2) -> f64 {
        let mut node = &self.root;
        let mut o = self.lo;
        let mut s = self.size;
        loop {
            match node {
                Node::Leaf(h) => {
                    let u = ((p[0] - o[0]) / s).clamp(0.0, 1.0);
                    let v = ((p[1] - o[1]) / s).clamp(0.0, 1.0);
                    let bot = h[0] * (1.0 - u) + h[1] * u;
                    let top = h[2] * (1.0 - u) + h[3] * u;
                    return bot * (1.0 - v) + top * v;
                }
                Node::Branch(kids) => {
                    let half = s * 0.5;
                    let qx = (p[0] >= o[0] + half) as usize;
                    let qy = (p[1] >= o[1] + half) as usize;
                    o = [o[0] + qx as f64 * half, o[1] + qy as f64 * half];
                    s = half;
                    node = &kids[qy * 2 + qx];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_a_linear_field() {
        // h(x,y) = 1 + 0.1*x: a gradient-limited ramp. The quadtree + bilinear
        // interpolation must reproduce it closely everywhere.
        let f = |p: P2| 1.0 + 0.1 * p[0];
        let field = QuadtreeField::from_fn([0.0, 0.0], [10.0, 10.0], 0.5, 12, f);
        for &(x, y) in &[(0.3, 0.7), (4.5, 9.1), (9.9, 0.1), (2.0, 2.0)] {
            let got = field.eval([x, y]);
            let want = f([x, y]);
            assert!((got - want).abs() < 0.05, "at ({x},{y}): {got} vs {want}");
        }
    }

    #[test]
    fn clamps_outside_the_box() {
        let field = QuadtreeField::from_fn([0.0, 0.0], [1.0, 1.0], 0.25, 8, |_| 0.3);
        assert!((field.eval([-5.0, 20.0]) - 0.3).abs() < 1e-9);
    }
}
