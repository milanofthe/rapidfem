//! Points prepared for many 2D predicates in one projection.
//!
//! [`orient2d`](crate::orient2d) and [`incircle2d`](crate::incircle2d)
//! rebuild the homogeneous coordinates of an implicit point on every call:
//! as intervals for the filter and as expansions when it fails. A
//! triangulation of one facet asks thousands of predicates about the same
//! few hundred points, so [`Projected`] keeps both: the interval enclosure
//! from the start, the exact expansions from their first use. The signs are
//! the same as those of the uncached predicates.

use crate::expansion::Expansion;
use crate::geom::{det3, det4};
use crate::interval::Interval;
use crate::point::Point3;
use crate::ring::Ring;
use crate::{Axis, Sign};
use std::cell::OnceCell;

/// A point with its homogeneous coordinates in the projection that drops
/// one axis, `(u, v, w)` in the order of [`Point3::hom2`].
pub struct Projected {
    point: Point3,
    drop: Axis,
    /// The projected coordinates of an explicit point.
    explicit: Option<[f64; 2]>,
    interval: [Interval; 3],
    /// The affine projected coordinates `(u / w, v / w)` as intervals, when
    /// `w` is certainly nonzero: far tighter than the homogeneous rows once
    /// lifted, so the filters below decide on them first.
    affine: Option<[Interval; 2]>,
    exact: OnceCell<[Expansion; 3]>,
}

impl Projected {
    /// Prepares `point` for predicates in the projection dropping `drop`.
    pub fn new(point: Point3, drop: Axis) -> Projected {
        let explicit = point.as_explicit().map(|p| match drop {
            Axis::X => [p[1], p[2]],
            Axis::Y => [p[2], p[0]],
            Axis::Z => [p[0], p[1]],
        });
        let interval = point.hom2::<Interval>(drop);
        let affine = match explicit {
            Some([u, v]) => Some([Interval::point(u), Interval::point(v)]),
            None => interval[0]
                .checked_div(interval[2])
                .zip(interval[1].checked_div(interval[2]))
                .map(|(u, v)| [u, v]),
        };
        Projected {
            point,
            drop,
            explicit,
            interval,
            affine,
            exact: OnceCell::new(),
        }
    }

    /// The point itself.
    pub fn point(&self) -> &Point3 {
        &self.point
    }

    /// False when the two points certainly differ in the projection: their
    /// affine coordinates are explicit and unequal, or their intervals are
    /// apart. On points of one plane (the projection keeps it) true is
    /// necessary for coincidence.
    pub fn may_coincide(&self, other: &Projected) -> bool {
        if let (Some(a), Some(b)) = (self.explicit, other.explicit) {
            return a == b;
        }
        match (&self.affine, &other.affine) {
            (Some(a), Some(b)) => (0..2).all(|k| a[k].lo() <= b[k].hi() && b[k].lo() <= a[k].hi()),
            _ => true,
        }
    }

    fn exact(&self) -> &[Expansion; 3] {
        self.exact
            .get_or_init(|| self.point.hom2::<Expansion>(self.drop))
    }
}

/// The line an implicit point lies on by construction: the line `p q` of a
/// line-plane intersection, endpoints in a fixed order.
fn carrier(p: &Point3) -> Option<[[f64; 3]; 2]> {
    match p {
        Point3::Lpi { p, q, .. } => Some(if lex_le(p, q) { [*p, *q] } else { [*q, *p] }),
        _ => None,
    }
}

fn lex_le(a: &[f64; 3], b: &[f64; 3]) -> bool {
    (a[0], a[1], a[2]).partial_cmp(&(b[0], b[1], b[2])) != Some(std::cmp::Ordering::Greater)
}

/// True if every point lies on one line by construction: a line-plane
/// intersection of that line, or an explicit point defining it.
fn on_common_line(pts: &[&Projected]) -> bool {
    let Some(line) = pts.iter().find_map(|p| carrier(&p.point)) else {
        return false;
    };
    pts.iter().all(|p| match &p.point {
        Point3::Explicit(c) => *c == line[0] || *c == line[1],
        other => carrier(other) == Some(line),
    })
}

/// [`orient2d`](crate::orient2d) of prepared points (same projection).
pub fn orient2d(a: &Projected, b: &Projected, c: &Projected) -> Option<Sign> {
    if let (Some(pa), Some(pb), Some(pc)) = (a.explicit, b.explicit, c.explicit) {
        return Some(Sign::of_f64(geometry_predicates::orient2d(pa, pb, pc)));
    }
    let pts = [a, b, c];
    // Affine filter: det2 of the differences to `c`.
    if let (Some(pa), Some(pb), Some(pc)) = (a.affine, b.affine, c.affine) {
        let (ax, ay) = (pa[0].sub(pc[0]), pa[1].sub(pc[1]));
        let (bx, by) = (pb[0].sub(pc[0]), pb[1].sub(pc[1]));
        if let Some(sign) = ax.mul(by).sub(ay.mul(bx)).sign() {
            return Some(sign);
        }
    }
    // det3 of homogeneous rows = (product of w) * det2[[a - c], [b - c]].
    'filter: {
        let rows: [[Interval; 3]; 3] = std::array::from_fn(|i| pts[i].interval);
        let Some(mut sign) = det3(&rows).sign() else {
            break 'filter;
        };
        for p in &pts {
            match p.interval[2].sign() {
                Some(Sign::Positive) => {}
                Some(Sign::Negative) => sign = sign.flip(),
                _ => break 'filter,
            }
        }
        return Some(sign);
    }
    // Three points on one line by construction (line-plane intersections of
    // one line and that line's defining points) are collinear: no arithmetic.
    if on_common_line(&pts)
        && pts
            .iter()
            .all(|p| matches!(p.interval[2].sign(), Some(s) if s != Sign::Zero))
    {
        return Some(Sign::Zero);
    }
    let rows: [[Expansion; 3]; 3] = std::array::from_fn(|i| pts[i].exact().clone());
    let mut sign = det3(&rows).sign();
    for p in &pts {
        match p.exact()[2].sign() {
            Sign::Zero => return None,
            s => sign = sign.combine(s),
        }
    }
    Some(sign)
}

/// [`incircle2d`](crate::incircle2d) of prepared points (same projection).
pub fn incircle2d(a: &Projected, b: &Projected, c: &Projected, d: &Projected) -> Option<Sign> {
    if let Some(sign) = incircle2d_filtered(a, b, c, d) {
        return Some(sign);
    }
    let pts = [a, b, c, d];
    if pts.iter().any(|p| p.exact()[2].sign() == Sign::Zero) {
        return None;
    }
    let rows: [[Expansion; 4]; 4] = std::array::from_fn(|i| lifted(pts[i].exact()));
    Some(det4(&rows).sign())
}

/// [`incircle2d`] as far as the floating-point filters certify it: `None`
/// where only the exact stage could tell (near-cocircular points), which is
/// by far the most expensive case with implicit points. For callers that
/// may leave such a case undecided (a Delaunay pass that needs no canonical
/// form: both diagonals of cocircular points are equally good).
pub fn incircle2d_filtered(
    a: &Projected,
    b: &Projected,
    c: &Projected,
    d: &Projected,
) -> Option<Sign> {
    if let (Some(pa), Some(pb), Some(pc), Some(pd)) =
        (a.explicit, b.explicit, c.explicit, d.explicit)
    {
        return Some(Sign::of_f64(geometry_predicates::incircle(pa, pb, pc, pd)));
    }
    let pts = [a, b, c, d];
    // Affine filter: the classic det3 of the differences to `d`, lifted.
    if let (Some(pa), Some(pb), Some(pc), Some(pd)) = (a.affine, b.affine, c.affine, d.affine) {
        let row = |p: [Interval; 2]| -> [Interval; 3] {
            let (x, y) = (p[0].sub(pd[0]), p[1].sub(pd[1]));
            [x, y, x.mul(x).add(y.mul(y))]
        };
        if let Some(sign) = det3(&[row(pa), row(pb), row(pc)]).sign() {
            return Some(sign);
        }
    }
    if pts
        .iter()
        .all(|p| matches!(p.interval[2].sign(), Some(s) if s != Sign::Zero))
    {
        let rows: [[Interval; 4]; 4] = std::array::from_fn(|i| lifted(&pts[i].interval));
        if let Some(sign) = det4(&rows).sign() {
            return Some(sign);
        }
    }
    None
}

/// The lifted homogeneous row `(x w, y w, x^2 + y^2, w^2)` of a point.
fn lifted<T: Ring>(h: &[T; 3]) -> [T; 4] {
    let (x, y, w) = (&h[0], &h[1], &h[2]);
    [x.mul(w), y.mul(w), x.mul(x).add(&y.mul(y)), w.mul(w)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points on one line by construction (intersections of the line with
    /// several planes, and the line's own endpoints) are collinear in the
    /// structural shortcut exactly as in the full evaluation, and a point
    /// off the line is not.
    #[test]
    fn points_on_one_constructed_line_are_collinear() {
        let (p, q) = ([0.1, -0.3, 0.7], [0.9, 0.4, -0.2]);
        let cut = |h: f64| Point3::Lpi {
            p,
            q,
            r: [0.0, 0.0, h],
            s: [1.0, 0.0, h + 0.1],
            t: [0.0, 1.0, h - 0.05],
        };
        let pts = [
            Point3::Explicit(p),
            Point3::Explicit(q),
            cut(0.0),
            cut(0.2),
            cut(-0.1),
            Point3::Explicit([0.5, 0.5, 0.5]),
        ];
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let prep: Vec<Projected> = pts
                .iter()
                .map(|x| Projected::new(x.clone(), axis))
                .collect();
            for i in 0..pts.len() {
                for j in 0..pts.len() {
                    for k in 0..pts.len() {
                        assert_eq!(
                            orient2d(&prep[i], &prep[j], &prep[k]),
                            crate::orient2d(&pts[i], &pts[j], &pts[k], axis),
                            "{i} {j} {k} {axis:?}"
                        );
                    }
                }
            }
            assert_eq!(orient2d(&prep[2], &prep[3], &prep[4]), Some(Sign::Zero));
        }
    }

    /// The prepared predicates agree with the uncached ones on implicit
    /// points, degenerate configurations included.
    #[test]
    fn prepared_signs_match_the_uncached_ones() {
        let mut s = 0x243f_6a88_85a3_08d3u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) % 9) as f64 * 0.25 - 1.0
        };
        let mut pts: Vec<Point3> = Vec::new();
        for _ in 0..24 {
            let e: [f64; 3] = [r(), r(), 0.0];
            pts.push(Point3::Explicit(e));
            // A line through the plane z = 0: an implicit point on it.
            let (p, q) = ([r(), r(), -1.0], [r(), r(), 1.0]);
            pts.push(Point3::Lpi {
                p,
                q,
                r: [0.0, 0.0, 0.0],
                s: [1.0, 0.0, 0.0],
                t: [0.0, 1.0, 0.0],
            });
        }
        let prep: Vec<Projected> = pts
            .iter()
            .map(|p| Projected::new(p.clone(), Axis::Z))
            .collect();
        let n = pts.len();
        for i in 0..n {
            for j in 0..n {
                for k in (0..n).step_by(5) {
                    assert_eq!(
                        orient2d(&prep[i], &prep[j], &prep[k]),
                        crate::orient2d(&pts[i], &pts[j], &pts[k], Axis::Z)
                    );
                    let l = (i + j + k) % n;
                    assert_eq!(
                        incircle2d(&prep[i], &prep[j], &prep[k], &prep[l]),
                        crate::incircle2d(&pts[i], &pts[j], &pts[k], &pts[l], Axis::Z)
                    );
                }
            }
        }
    }
}
