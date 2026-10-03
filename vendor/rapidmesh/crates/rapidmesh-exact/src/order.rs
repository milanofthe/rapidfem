//! Ordering predicates along lines, for points that may be implicit.

use crate::expansion::Expansion;
use crate::interval::Interval;
use crate::orient::orient2d;
use crate::point::Point3;
use crate::ring::Ring;
use crate::{Axis, Sign};

/// (q - p) · (b - a) over homogeneous coordinates, times the (separately
/// sign-corrected) product of the four w's.
fn dot_diff<T: Ring>(ha: &[T; 4], hb: &[T; 4], hp: &[T; 4], hq: &[T; 4]) -> T {
    let mut acc: Option<T> = None;
    for i in 0..3 {
        let dq = hq[i].mul(&hp[3]).sub(&hp[i].mul(&hq[3]));
        let db = hb[i].mul(&ha[3]).sub(&ha[i].mul(&hb[3]));
        let term = dq.mul(&db);
        acc = Some(match acc {
            None => term,
            Some(s) => s.add(&term),
        });
    }
    acc.expect("three components")
}

/// Exact sign of (q - p) · (b - a): orders `p` vs `q` along the direction
/// from `a` to `b`. Positive means `q` lies further along a→b than `p`.
///
/// All four points may be implicit. Returns `None` if any point is invalid.
pub fn cmp_along(a: &Point3, b: &Point3, p: &Point3, q: &Point3) -> Option<Sign> {
    // Interval filter.
    'filter: {
        let ha = a.hom::<Interval>();
        let hb = b.hom::<Interval>();
        let hp = p.hom::<Interval>();
        let hq = q.hom::<Interval>();
        let Some(mut sign) = dot_diff(&ha, &hb, &hp, &hq).sign() else {
            break 'filter;
        };
        for h in [&ha, &hb, &hp, &hq] {
            match h[3].sign() {
                Some(Sign::Positive) => {}
                Some(Sign::Negative) => sign = sign.flip(),
                _ => break 'filter,
            }
        }
        return Some(sign);
    }
    // Exact stage.
    let ha = a.hom::<Expansion>();
    let hb = b.hom::<Expansion>();
    let hp = p.hom::<Expansion>();
    let hq = q.hom::<Expansion>();
    let mut sign = dot_diff(&ha, &hb, &hp, &hq).sign();
    for h in [&ha, &hb, &hp, &hq] {
        match h[3].sign() {
            Sign::Zero => return None,
            s => sign = sign.combine(s),
        }
    }
    Some(sign)
}

/// Exact lexicographic comparison of two (possibly implicit) points by
/// (x, y, z). `None` if a point is invalid.
pub fn lex_cmp(p: &Point3, q: &Point3) -> Option<std::cmp::Ordering> {
    let origin = Point3::explicit(0.0, 0.0, 0.0);
    let axes = [
        Point3::explicit(1.0, 0.0, 0.0),
        Point3::explicit(0.0, 1.0, 0.0),
        Point3::explicit(0.0, 0.0, 1.0),
    ];
    for axis in &axes {
        match cmp_along(&origin, axis, p, q)? {
            Sign::Positive => return Some(std::cmp::Ordering::Less),
            Sign::Negative => return Some(std::cmp::Ordering::Greater),
            Sign::Zero => {}
        }
    }
    Some(std::cmp::Ordering::Equal)
}

/// Exact 3D collinearity: true if `a`, `b`, `c` lie on one line (zero
/// orientation in every axis projection). `None` if a point is invalid.
pub fn collinear(a: &Point3, b: &Point3, c: &Point3) -> Option<bool> {
    for drop in [Axis::X, Axis::Y, Axis::Z] {
        if orient2d(a, b, c, drop)? != Sign::Zero {
            return Some(false);
        }
    }
    Some(true)
}

/// Exact closed betweenness on the segment [a, b]: true if the (collinear)
/// point `p` satisfies a ≤ p ≤ b along the segment. The caller is responsible
/// for `p` being on the line through `a`, `b`.
pub fn within_closed(a: &Point3, b: &Point3, p: &Point3) -> Option<bool> {
    Some(cmp_along(a, b, a, p)? != Sign::Negative && cmp_along(a, b, p, b)? != Sign::Negative)
}

/// Exact open betweenness: true if `p` lies strictly between `a` and `b`.
pub fn strictly_between(a: &Point3, b: &Point3, p: &Point3) -> Option<bool> {
    Some(cmp_along(a, b, a, p)? == Sign::Positive && cmp_along(a, b, p, b)? == Sign::Positive)
}
