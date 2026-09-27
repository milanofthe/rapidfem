//! Exact triangle-triangle intersection with implicit intersection points.
//!
//! The workhorse of the arrangement stage. Intersection points are produced
//! as implicit [`Point3`]s (line-plane intersections, or original vertices) —
//! never as rounded coordinates — so cascaded predicates downstream stay
//! exact.

use crate::tri::Tri;
use rapidmesh_exact::{orient3d, Point3, Sign};

/// Exact intersection of two non-degenerate triangles.
#[derive(Debug, Clone, PartialEq)]
pub enum TriTriIsect {
    /// The triangles do not intersect.
    Disjoint,
    /// The triangles touch in a single point.
    Touching(Point3),
    /// The triangles intersect in a proper segment.
    Segment(Point3, Point3),
    /// The triangles are exactly coplanar (intersection is 2-dimensional and
    /// handled by the in-plane pipeline).
    Coplanar,
}

/// Exact signs of `t`'s vertices against the plane of `plane_tri`.
fn plane_signs(plane_tri: &Tri, t: &Tri) -> [Sign; 3] {
    std::array::from_fn(|i| {
        orient3d(
            &plane_tri.point(0),
            &plane_tri.point(1),
            &plane_tri.point(2),
            &t.point(i),
        )
        .expect("explicit points are always valid")
    })
}

/// Candidate endpoints contributed by `t`'s boundary against `other`:
/// vertices of `t` lying exactly on `other`'s plane and inside `other`, and
/// intersections of `t`'s edges strictly crossing `other`'s plane inside
/// `other`.
fn boundary_candidates(t: &Tri, signs: &[Sign; 3], other: &Tri, out: &mut Vec<Point3>) {
    let (axis, orientation) = other.projection_axis();
    // On-plane vertices.
    for (i, &s) in signs.iter().enumerate() {
        if s == Sign::Zero {
            let p = t.point(i);
            if other.contains_coplanar(&p, axis, orientation) {
                out.push(p);
            }
        }
    }
    // Strictly crossing edges.
    for i in 0..3 {
        let j = (i + 1) % 3;
        if signs[i].combine(signs[j]) == Sign::Negative {
            let p = Point3::lpi(t.v[i], t.v[j], other.v[0], other.v[1], other.v[2]);
            debug_assert!(p.is_valid(), "strict sign change implies a valid LPI");
            if other.contains_coplanar(&p, axis, orientation) {
                out.push(p);
            }
        }
    }
}

/// Exact intersection of two non-degenerate triangles.
///
/// Candidate endpoints are the points where one triangle's boundary meets the
/// other triangle (on-plane vertices and edge-plane crossings, kept only when
/// contained in the other triangle); every endpoint of the intersection
/// segment is such a point. Deduplication is exact ([`Point3::coincides`]).
pub fn tri_tri_intersection(t0: &Tri, t1: &Tri) -> TriTriIsect {
    let s1 = plane_signs(t0, t1);
    if s1 == [Sign::Zero; 3] {
        return TriTriIsect::Coplanar;
    }
    if s1.iter().all(|&s| s == Sign::Positive) || s1.iter().all(|&s| s == Sign::Negative) {
        return TriTriIsect::Disjoint;
    }
    let s0 = plane_signs(t1, t0);
    if s0.iter().all(|&s| s == Sign::Positive) || s0.iter().all(|&s| s == Sign::Negative) {
        return TriTriIsect::Disjoint;
    }

    let mut candidates: Vec<Point3> = Vec::new();
    boundary_candidates(t1, &s1, t0, &mut candidates);
    boundary_candidates(t0, &s0, t1, &mut candidates);

    // Exact dedup.
    let mut distinct: Vec<Point3> = Vec::new();
    for c in candidates {
        if !distinct.iter().any(|d| d.coincides(&c)) {
            distinct.push(c);
        }
    }

    match distinct.len() {
        0 => TriTriIsect::Disjoint,
        1 => TriTriIsect::Touching(distinct.pop().expect("len 1")),
        2 => {
            let b = distinct.pop().expect("len 2");
            let a = distinct.pop().expect("len 2");
            TriTriIsect::Segment(a, b)
        }
        n => {
            unreachable!("non-coplanar tri-tri intersection produced {n} distinct candidate points")
        }
    }
}
