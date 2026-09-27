//! Constraints injected into a facet by intersections with other facets.
//!
//! A constraint carries the provenance of the line it lies on, so that the
//! intersection point of two constraints can be constructed exactly:
//! two plane-cut constraints meet in a TPI of the three original planes, a
//! plane-cut and an edge constraint meet in an LPI, and two (coplanar) edge
//! constraints meet in a synthesized-plane LPI.

use rapidmesh_exact::Point3;

/// The line a constraint segment lies on, by provenance.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintLine {
    /// The intersection line of the facet's plane with another triangle's
    /// plane (given by that triangle's vertices).
    PlaneCut([[f64; 3]; 3]),
    /// The line through two explicit points (an edge of a coplanar triangle).
    Edge([f64; 3], [f64; 3]),
}

/// A constraint segment on a facet, with exact (possibly implicit) endpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    /// First endpoint.
    pub a: Point3,
    /// Second endpoint.
    pub b: Point3,
    /// Provenance of the supporting line.
    pub line: ConstraintLine,
}

impl Constraint {
    /// The exact intersection point of this constraint's supporting line with
    /// another constraint's supporting line, both lying in the plane of the
    /// facet given by `facet_plane`. `None` if the lines are parallel or
    /// identical.
    pub fn line_intersection(
        &self,
        other: &Constraint,
        facet_plane: [[f64; 3]; 3],
    ) -> Option<Point3> {
        match (&self.line, &other.line) {
            (ConstraintLine::PlaneCut(p1), ConstraintLine::PlaneCut(p2)) => {
                let p = Point3::tpi(facet_plane, *p1, *p2);
                p.is_valid().then_some(p)
            }
            (ConstraintLine::PlaneCut(plane), ConstraintLine::Edge(u, v))
            | (ConstraintLine::Edge(u, v), ConstraintLine::PlaneCut(plane)) => {
                let p = Point3::lpi(*u, *v, plane[0], plane[1], plane[2]);
                p.is_valid().then_some(p)
            }
            (ConstraintLine::Edge(u, v), ConstraintLine::Edge(a, b)) => {
                Point3::lli_coplanar(*u, *v, *a, *b)
            }
        }
    }
}
