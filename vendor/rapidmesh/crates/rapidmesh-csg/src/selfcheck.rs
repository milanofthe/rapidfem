//! Exact check of an indexed triangle surface: triangles that meet other
//! than in the vertices and edges they share.
//!
//! Two triangles of a valid surface meet only where they share: nowhere if
//! they share nothing, in the one vertex, along the one edge (without
//! folding onto each other in a plane). Anything else is a crossing, a
//! touch or an overlap the surface must not have; a triangle of zero area
//! is reported against itself.

use crate::tri::Tri;
use crate::tri_tri::{tri_tri_intersection, TriTriIsect};
use rapidmesh_exact::{lex_cmp, orient2d, orient3d, Point3, Sign};
use rayon::prelude::*;

/// The pairs `[i, j]` (`i <= j`, ascending) of triangles of `tris` over
/// `verts` that meet other than in what they share.
pub fn improper_pairs(verts: &[[f64; 3]], tris: &[[u32; 3]]) -> Vec<[u32; 2]> {
    let n = tris.len();
    let corners = |t: usize| tris[t].map(|v| verts[v as usize]);
    let boxes: Vec<([f64; 3], [f64; 3])> = (0..n)
        .map(|t| {
            let c = corners(t);
            (
                std::array::from_fn(|k| c[0][k].min(c[1][k]).min(c[2][k])),
                std::array::from_fn(|k| c[0][k].max(c[1][k]).max(c[2][k])),
            )
        })
        .collect();
    // Sweep along the longest extent: sorted by the low end, a pair can
    // overlap only while the earlier one's high end reaches the later low.
    let (lo, hi) = boxes
        .iter()
        .fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), (a, b)| {
            (
                std::array::from_fn(|k| lo[k].min(a[k])),
                std::array::from_fn(|k| hi[k].max(b[k])),
            )
        });
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .unwrap_or(0);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| boxes[a].0[axis].total_cmp(&boxes[b].0[axis]));
    let overlap = |a: usize, b: usize| {
        (0..3).all(|k| boxes[a].0[k] <= boxes[b].1[k] && boxes[b].0[k] <= boxes[a].1[k])
    };
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for (i, &a) in order.iter().enumerate() {
        for &b in &order[i + 1..] {
            if boxes[b].0[axis] > boxes[a].1[axis] {
                break;
            }
            if overlap(a, b) {
                candidates.push((a.min(b), a.max(b)));
            }
        }
    }
    let tri = |t: usize| {
        let c = corners(t);
        Tri::new(c[0], c[1], c[2])
    };
    let mut out: Vec<[u32; 2]> = candidates
        .par_iter()
        .filter(|&&(a, b)| improper(tris[a], tris[b], &tri(a), &tri(b)))
        .map(|&(a, b)| [a as u32, b as u32])
        .collect();
    out.extend(
        (0..n)
            .filter(|&t| tri(t).is_degenerate())
            .map(|t| [t as u32, t as u32]),
    );
    out.sort_unstable();
    out.dedup();
    out
}

/// Whether triangles `a` and `b` (vertex ids, and as triangles `ta`, `tb`)
/// meet other than in what they share.
fn improper(a: [u32; 3], b: [u32; 3], ta: &Tri, tb: &Tri) -> bool {
    if ta.is_degenerate() || tb.is_degenerate() {
        return false; // reported on their own
    }
    let shared: Vec<u32> = a.iter().copied().filter(|v| b.contains(v)).collect();
    let pa = |i: usize| ta.point(i);
    let other = |t: &[u32; 3], tt: &Tri, not: &[u32]| -> Vec<Point3> {
        (0..3)
            .filter(|&i| !not.contains(&t[i]))
            .map(|i| tt.point(i))
            .collect()
    };
    let at = |t: &[u32; 3], tt: &Tri, v: u32| tt.point(t.iter().position(|&x| x == v).unwrap());
    let same = |p: &Point3, q: &Point3| lex_cmp(p, q) == Some(std::cmp::Ordering::Equal);
    match shared.len() {
        3 => true,
        2 => {
            // Along the shared edge they only fold: coplanar, with their
            // third corners on one side of it.
            let (u, v) = (at(&a, ta, shared[0]), at(&a, ta, shared[1]));
            let (c, d) = (
                other(&a, ta, &shared)[0].clone(),
                other(&b, tb, &shared)[0].clone(),
            );
            if orient3d(&pa(0), &pa(1), &pa(2), &d) != Some(Sign::Zero) {
                return false;
            }
            let (axis, _) = ta.projection_axis();
            let (sc, sd) = (orient2d(&u, &v, &c, axis), orient2d(&u, &v, &d, axis));
            sc == sd
        }
        1 => {
            let v = at(&a, ta, shared[0]);
            match tri_tri_intersection(ta, tb) {
                TriTriIsect::Disjoint => false,
                TriTriIsect::Touching(p) => !same(&p, &v),
                TriTriIsect::Segment(..) => true,
                TriTriIsect::Coplanar => {
                    // Their corners at the shared vertex: they overlap where
                    // an edge of one runs into (or along) the other.
                    let (axis, _) = ta.projection_axis();
                    let (oa, ob) = (other(&a, ta, &shared), other(&b, tb, &shared));
                    let inside = |p: &Point3, e: &[Point3]| {
                        let o = orient2d(&v, &e[0], &e[1], axis);
                        let s1 = orient2d(&v, &e[0], p, axis);
                        let s2 = orient2d(&v, p, &e[1], axis);
                        let with = |s: Option<Sign>| s == o || s == Some(Sign::Zero);
                        with(s1) && with(s2)
                    };
                    ob.iter().any(|p| inside(p, &oa)) || oa.iter().any(|p| inside(p, &ob))
                }
            }
        }
        _ => match tri_tri_intersection(ta, tb) {
            TriTriIsect::Disjoint => false,
            TriTriIsect::Coplanar => coplanar_meet(ta, tb),
            _ => true,
        },
    }
}

/// Whether two coplanar triangles without a shared vertex meet (closed): an
/// edge of one touches an edge of the other, or a corner lies in the other.
fn coplanar_meet(ta: &Tri, tb: &Tri) -> bool {
    // One plane: `ta`'s projection shows `tb` with area too.
    let (axis, oa) = ta.projection_axis();
    let ob = orient2d(&tb.point(0), &tb.point(1), &tb.point(2), axis).unwrap_or(Sign::Zero);
    let o = |p: &Point3, q: &Point3, r: &Point3| orient2d(p, q, r, axis);
    let touch = |p: &Point3, q: &Point3, r: &Point3, s: &Point3| {
        let (d1, d2) = (o(p, q, r), o(p, q, s));
        let (d3, d4) = (o(r, s, p), o(r, s, q));
        let opposite = |x: Option<Sign>, y: Option<Sign>| {
            x == Some(Sign::Zero) || y == Some(Sign::Zero) || x != y
        };
        if d1 == Some(Sign::Zero) && d2 == Some(Sign::Zero) {
            // Collinear: they touch where their extents along the line do.
            let key = |x: &Point3| x.approx().unwrap_or([0.0; 3]);
            let (lo1, hi1) = sorted(key(p), key(q));
            let (lo2, hi2) = sorted(key(r), key(s));
            return lo1 <= hi2 && lo2 <= hi1;
        }
        opposite(d1, d2) && opposite(d3, d4)
    };
    let (a, b) = (
        [0, 1, 2].map(|i| ta.point(i)),
        [0, 1, 2].map(|i| tb.point(i)),
    );
    for i in 0..3 {
        for j in 0..3 {
            if touch(&a[i], &a[(i + 1) % 3], &b[j], &b[(j + 1) % 3]) {
                return true;
            }
        }
    }
    tb.contains_coplanar(&a[0], axis, ob) || ta.contains_coplanar(&b[0], axis, oa)
}

/// Two points in lexicographic order.
fn sorted(p: [f64; 3], q: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    if p <= q {
        (p, q)
    } else {
        (q, p)
    }
}
