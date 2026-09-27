//! Small shared geometry primitives used by more than one stage of the tet
//! pipeline. Each lived as a byte-identical copy in two modules before; keeping
//! one definition here removes the "fix it in both places" hazard.

use rapidmesh_geom::vec3::{cross, dist, len, sub, V3};

/// Circumradius of triangle `(a, b, c)` in 3-space: `|ab||bc||ca| / (4 * area)`,
/// computed from the cross-product area. Returns `INFINITY` for a degenerate
/// (collinear) triangle.
pub(crate) fn circumradius(a: V3, b: V3, c: V3) -> f64 {
    let (ab, bc, ca) = (dist(a, b), dist(b, c), dist(c, a));
    // 2 * area, from the cross product of two edges.
    let area2 = len(cross(sub(b, a), sub(c, a)));
    if area2 <= 1e-300 {
        f64::INFINITY
    } else {
        ab * bc * ca / (2.0 * area2)
    }
}
