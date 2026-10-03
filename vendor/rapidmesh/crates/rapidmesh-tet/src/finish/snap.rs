//! Moving boundary vertices onto the true shape.
//!
//! The refinement runs on an exact discrete model (a PLC); the analytic
//! carriers and curves it approximates come back here. Each curve and patch
//! vertex moves to its projection onto its [`Shape`], curves before patches,
//! as far as every incident tet stays positively oriented and none drops
//! below `FLOOR_DEG` (or below where its star already was): the full
//! move, else the largest of a few halvings, else none. Labels, faces and
//! connectivity never change, so every invariant but orientation carries
//! over, and orientation is checked exactly. The moves run on the index of
//! [`crate::finish::improve`], which snaps again what stayed short once it has
//! reshaped the star ([`crate::finish::improve::finish`]).
//!
//! A volume vertex that ended up on faces of one patch only (a cell point
//! inserted so close to the surface that the surface points meant to
//! replace it were rejected as its duplicates) is a surface sample in all
//! but name: it becomes a vertex of that patch first, and snaps with the
//! rest.

use crate::finish::P3;
use crate::finish::{Complex, PointClass};

/// The true shape behind a discrete model.
pub trait Shape: Sync {
    /// The point of the shape a vertex of `kind` at `p` belongs at: on its
    /// patch's carrier or its curve. `None` leaves the vertex where it is.
    fn project(&self, kind: PointClass, p: P3) -> Option<P3>;

    /// [`Shape::project`] to within a small share of the local size, where
    /// that is much cheaper (a curve's dense samples instead of the curve):
    /// for comparing candidate places, the chosen one projected exactly.
    fn project_near(&self, kind: PointClass, p: P3) -> Option<P3> {
        self.project(kind, p)
    }

    /// The parameters of `p` on the carrier of `kind` where a search from
    /// them is cheaper than [`Shape::project`] (a spline carrier).
    fn param(&self, _kind: PointClass, _p: P3) -> Option<[f64; 2]> {
        None
    }

    /// [`Shape::project`] searched from `uv`, the parameters of a point
    /// near the answer, with the parameters of the answer.
    fn project_from(&self, kind: PointClass, p: P3, uv: [f64; 2]) -> Option<(P3, [f64; 2])> {
        self.project(kind, p).map(|q| (q, uv))
    }

    /// A point of the shape near `p` for comparing candidate places, searched
    /// from `uv` like [`Shape::project_from`] but only as exact as a
    /// candidate needs (one step on a spline carrier); the point lies on the
    /// shape.
    fn project_near_from(&self, kind: PointClass, p: P3, uv: [f64; 2]) -> Option<P3> {
        self.project_from(kind, p, uv).map(|r| r.0)
    }

    /// Whether a vertex of `kind` may slide on its carrier or curve for
    /// quality: a smooth one, not a faceted patch or a polyline, whose
    /// facets and kinks a moved vertex would cut.
    fn smooth(&self, _kind: PointClass) -> bool {
        true
    }
}

/// Halvings tried before a vertex is left in place.
pub(crate) const HALVINGS: usize = 4;
/// A snap step may not take a tet of the star below this dihedral (degrees)
/// unless the star was there already: fidelity to the shape is not worth a
/// sliver.
pub(crate) const FLOOR_DEG: f64 = 10.0;

/// Gives every volume vertex that lies on faces of exactly one patch that
/// patch's kind.
pub(crate) fn adopt_face_vertices(c: &mut Complex) {
    const NONE: u32 = u32::MAX;
    const MIXED: u32 = u32::MAX - 1;
    let mut patch = vec![NONE; c.points.len()];
    for f in &c.faces {
        for &v in &f.tri {
            let p = &mut patch[v as usize];
            *p = if f.patch >= MIXED {
                MIXED
            } else if *p == NONE || *p == f.patch {
                f.patch
            } else {
                MIXED
            };
        }
    }
    for (v, &p) in patch.iter().enumerate() {
        if p < MIXED && c.classes[v] == PointClass::Interior {
            c.classes[v] = PointClass::Face(p);
        }
    }
}
