//! Triangle primitives and exact containment tests.

use rapidmesh_exact::{orient2d, Axis, Point3, Sign};

/// A triangle of the input soup, with explicit f64 vertices.
///
/// Must be non-degenerate (positive area); degenerate triangles are rejected
/// when a projection axis is requested.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tri {
    /// The three vertices.
    pub v: [[f64; 3]; 3],
}

impl Tri {
    /// New triangle from vertices.
    pub fn new(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Tri {
        Tri { v: [a, b, c] }
    }

    /// Vertex as a [`Point3`].
    pub fn point(&self, i: usize) -> Point3 {
        Point3::Explicit(self.v[i])
    }

    /// A projection axis in which this triangle has exactly nonzero area,
    /// together with its 2D orientation there.
    ///
    /// Axes are tried in order of decreasing approximate normal component, so
    /// the chosen projection is also the numerically best-conditioned one.
    /// Panics on exactly degenerate (zero-area) triangles.
    pub fn projection_axis(&self) -> (Axis, Sign) {
        let [a, b, c] = self.v;
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        let mut axes = [Axis::X, Axis::Y, Axis::Z];
        axes.sort_by(|p, q| {
            n[q.index()]
                .abs()
                .partial_cmp(&n[p.index()].abs())
                .expect("finite coordinates")
        });
        for axis in axes {
            let s = orient2d(&self.point(0), &self.point(1), &self.point(2), axis)
                .expect("explicit points are always valid");
            if s != Sign::Zero {
                return (axis, s);
            }
        }
        panic!("degenerate (zero-area) triangle: {:?}", self.v);
    }

    /// Exact closed containment: true if `p` lies inside this triangle or on
    /// its boundary. `p` must be a valid point lying in this triangle's
    /// plane (containment is evaluated in the triangle's projection).
    pub fn contains_coplanar(&self, p: &Point3, axis: Axis, orientation: Sign) -> bool {
        for i in 0..3 {
            let a = self.point(i);
            let b = self.point((i + 1) % 3);
            let s = orient2d(&a, &b, p, axis).expect("valid point");
            if s != Sign::Zero && s != orientation {
                return false;
            }
        }
        true
    }
}
