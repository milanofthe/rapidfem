//! Staged-exact orientation predicates over explicit and implicit points.

use crate::expansion::Expansion;
use crate::geom::{det3, det4};
use crate::interval::Interval;
use crate::point::Point3;
use crate::{Axis, Sign};

/// Affine interval coordinates for points whose homogeneous w is exactly 1
/// (explicit, Pac); `None` for the projective kinds (Lpi, Tpi, Bary). Lets
/// the filter of [`orient3d`] use the plain affine difference determinant
/// instead of the homogeneous lift: w-sign folding disappears and the
/// determinant shrinks by one dimension.
#[inline]
fn affine_interval(p: &Point3) -> Option<[Interval; 3]> {
    match p {
        Point3::Explicit(c) => Some(c.map(Interval::point)),
        Point3::Pac { .. } => {
            let h = p.hom::<Interval>();
            Some([h[0], h[1], h[2]])
        }
        _ => None,
    }
}

/// Exact 3D orientation of four points, any of which may be implicit.
///
/// Sign convention: equals the sign of det [[a-d], [b-d], [c-d]] (rows), the
/// same convention as Shewchuk's `orient3d` -- positive when `d` lies below the
/// plane through `a`, `b`, `c` oriented counterclockwise as seen from above
/// the plane.
///
/// Returns `None` if any implicit point is invalid (its defining primitives
/// do not intersect in a single point, exact w == 0).
///
/// Evaluation is staged: fast adaptive path for all-explicit inputs
/// (`geometry-predicates`), conservative interval filter for implicit inputs,
/// exact expansion arithmetic as the final word.
pub fn orient3d(a: &Point3, b: &Point3, c: &Point3, d: &Point3) -> Option<Sign> {
    // Fast adaptive path: all points explicit.
    if let (Some(pa), Some(pb), Some(pc), Some(pd)) = (
        a.as_explicit(),
        b.as_explicit(),
        c.as_explicit(),
        d.as_explicit(),
    ) {
        return Some(Sign::of_f64(geometry_predicates::orient3d(pa, pb, pc, pd)));
    }

    let pts = [a, b, c, d];

    // Affine interval filter (all w exactly 1: explicit, Pac): the homogeneous det4 equals det3 of the rows
    // a-d, b-d, c-d, with no w corrections. When indecisive, the exact
    // stage decides directly (the projective filter sees the same widths).
    if let (Some(pa), Some(pb), Some(pc), Some(pd)) = (
        affine_interval(a),
        affine_interval(b),
        affine_interval(c),
        affine_interval(d),
    ) {
        let row = |p: &[Interval; 3]| -> [Interval; 3] { std::array::from_fn(|k| p[k].sub(pd[k])) };
        if let Some(sign) = det3(&[row(&pa), row(&pb), row(&pc)]).sign() {
            return Some(sign);
        }
    } else {
        // The 4x4 homogeneous determinant relates to the affine orientation
        // by det4 = (prod of w_i) * det3[[a-d],[b-d],[c-d]], so the
        // orientation sign is the det4 sign combined with each w sign.

        // Projective interval filter (some w != 1).
        'filter: {
            let homs: [[Interval; 4]; 4] = std::array::from_fn(|i| pts[i].hom::<Interval>());
            let Some(mut sign) = det4(&homs).sign() else {
                break 'filter;
            };
            for h in &homs {
                match h[3].sign() {
                    // Strictly signed w: fold into the result.
                    Some(Sign::Positive) => {}
                    Some(Sign::Negative) => sign = sign.flip(),
                    // w == 0 exactly or uncertain: let the exact stage decide
                    // validity.
                    _ => break 'filter,
                }
            }
            return Some(sign);
        }
    }

    // Exact stage.
    let homs: [[Expansion; 4]; 4] = std::array::from_fn(|i| pts[i].hom::<Expansion>());
    let mut sign = det4(&homs).sign();
    for h in &homs {
        match h[3].sign() {
            Sign::Zero => return None,
            s => sign = sign.combine(s),
        }
    }
    Some(sign)
}

/// A point prepared for many orientation tests against explicit points
/// ([`orient3d_explicit`]): its homogeneous coordinates as intervals once,
/// and exactly on first need. An implicit point's coordinates (a
/// barycenter of intersection points) cost far more than one determinant.
pub struct Prepared3 {
    point: Point3,
    explicit: Option<[f64; 3]>,
    affine: Option<[Interval; 3]>,
    hom: [Interval; 4],
    exact: std::cell::OnceCell<[Expansion; 4]>,
}

impl Prepared3 {
    /// Prepares `point`.
    pub fn new(point: Point3) -> Prepared3 {
        Prepared3 {
            explicit: point.as_explicit(),
            affine: affine_interval(&point),
            hom: point.hom::<Interval>(),
            exact: std::cell::OnceCell::new(),
            point,
        }
    }

    /// The point itself.
    pub fn point(&self) -> &Point3 {
        &self.point
    }
}

/// [`orient3d`] of explicit `a`, `b`, `c` and the prepared `p`: the same
/// sign by the same stages (adaptive f64, interval, exact), `p`'s
/// coordinates computed once for all calls.
pub fn orient3d_explicit(a: [f64; 3], b: [f64; 3], c: [f64; 3], p: &Prepared3) -> Option<Sign> {
    if let Some(d) = p.explicit {
        return Some(Sign::of_f64(geometry_predicates::orient3d(a, b, c, d)));
    }
    let lift = |q: [f64; 3]| -> [Interval; 4] {
        [
            Interval::point(q[0]),
            Interval::point(q[1]),
            Interval::point(q[2]),
            Interval::point(1.0),
        ]
    };
    if let Some(pd) = &p.affine {
        if let Some(sign) = orient3d_near(a, b, c, pd) {
            return Some(sign);
        }
        let row = |q: [f64; 3]| -> [Interval; 3] {
            std::array::from_fn(|k| Interval::point(q[k]).sub(pd[k]))
        };
        if let Some(sign) = det3(&[row(a), row(b), row(c)]).sign() {
            return Some(sign);
        }
    } else {
        'filter: {
            let Some(mut sign) = det4(&[lift(a), lift(b), lift(c), p.hom]).sign() else {
                break 'filter;
            };
            match p.hom[3].sign() {
                Some(Sign::Positive) => {}
                Some(Sign::Negative) => sign = sign.flip(),
                _ => break 'filter,
            }
            return Some(sign);
        }
    }
    let e = p.exact.get_or_init(|| p.point.hom::<Expansion>());
    let lift = |q: [f64; 3]| -> [Expansion; 4] {
        [
            Expansion::from_f64(q[0]),
            Expansion::from_f64(q[1]),
            Expansion::from_f64(q[2]),
            Expansion::from_f64(1.0),
        ]
    };
    let sign = det4(&[lift(a), lift(b), lift(c), e.clone()]).sign();
    match e[3].sign() {
        Sign::Zero => None,
        w => Some(sign.combine(w)),
    }
}

/// A static filter for [`orient3d_explicit`] with `p` known to lie in the
/// box `pd`: the orientation is affine in `p`, `(a - p) . n` with
/// `n = (b - a) x (c - a)`, so it differs from its value at the box center
/// by at most `sum |n_k| * width_k`; that value in f64 errs by at most
/// Shewchuk's first bound times the permanent. `None` when the two bounds
/// leave the sign open.
fn orient3d_near(a: [f64; 3], b: [f64; 3], c: [f64; 3], pd: &[Interval; 3]) -> Option<Sign> {
    let m: [f64; 3] = std::array::from_fn(|k| 0.5 * (pd[k].lo() + pd[k].hi()));
    let width: [f64; 3] = std::array::from_fn(|k| pd[k].hi() - pd[k].lo());
    let ad: [f64; 3] = std::array::from_fn(|k| a[k] - m[k]);
    let bd: [f64; 3] = std::array::from_fn(|k| b[k] - m[k]);
    let cd: [f64; 3] = std::array::from_fn(|k| c[k] - m[k]);
    let det = ad[2] * (bd[0] * cd[1] - cd[0] * bd[1])
        + bd[2] * (cd[0] * ad[1] - ad[0] * cd[1])
        + cd[2] * (ad[0] * bd[1] - bd[0] * ad[1]);
    let permanent = ((bd[0] * cd[1]).abs() + (cd[0] * bd[1]).abs()) * ad[2].abs()
        + ((cd[0] * ad[1]).abs() + (ad[0] * cd[1]).abs()) * bd[2].abs()
        + ((ad[0] * bd[1]).abs() + (bd[0] * ad[1]).abs()) * cd[2].abs();
    // Shewchuk's orient3d error bound A, (7 + 56 eps) eps with eps = 2^-53.
    let eps = f64::EPSILON / 2.0;
    let rounding = (7.0 + 56.0 * eps) * eps * permanent;
    // |n_k| bounded by the magnitudes of its two products; the factor covers
    // the rounding of the differences and products.
    let u: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
    let v: [f64; 3] = std::array::from_fn(|k| c[k] - a[k]);
    let n = [
        (u[1] * v[2]).abs() + (u[2] * v[1]).abs(),
        (u[2] * v[0]).abs() + (u[0] * v[2]).abs(),
        (u[0] * v[1]).abs() + (u[1] * v[0]).abs(),
    ];
    let spread = (n[0] * width[0] + n[1] * width[1] + n[2] * width[2]) * (1.0 + 1e-6);
    let bound = rounding + spread;
    if !bound.is_finite() {
        return None;
    }
    if det > bound {
        Some(Sign::Positive)
    } else if det < -bound {
        Some(Sign::Negative)
    } else {
        None
    }
}

/// Exact 2D in-circle test in the axis-aligned projection that drops the
/// given axis. Points may be implicit.
///
/// Positive iff `d` lies strictly inside the circumcircle of the
/// counterclockwise triangle (a, b, c); the sign flips for clockwise
/// (a, b, c). Returns `None` if any point is invalid.
///
/// Homogeneous lifting: the classic row (x, y, x^2 + y^2, 1) scaled by w^2
/// becomes (X W, Y W, X^2 + Y^2, W^2), polynomial in the homogeneous
/// coordinates, and the scaling factors are strictly positive so the
/// determinant sign needs no w correction.
pub fn incircle2d(a: &Point3, b: &Point3, c: &Point3, d: &Point3, drop: Axis) -> Option<Sign> {
    // Fast adaptive path: all points explicit.
    if let (Some(pa), Some(pb), Some(pc), Some(pd)) = (
        a.as_explicit(),
        b.as_explicit(),
        c.as_explicit(),
        d.as_explicit(),
    ) {
        let proj = |p: [f64; 3]| drop.project(p);
        return Some(Sign::of_f64(geometry_predicates::incircle(
            proj(pa),
            proj(pb),
            proj(pc),
            proj(pd),
        )));
    }

    fn lifted_row<T: crate::ring::Ring>(h: &[T; 3]) -> [T; 4] {
        let (x, y, w) = (&h[0], &h[1], &h[2]);
        [x.mul(w), y.mul(w), x.mul(x).add(&y.mul(y)), w.mul(w)]
    }

    let pts = [a, b, c, d];

    // Interval filter.
    {
        let rows: [[Interval; 4]; 4] =
            std::array::from_fn(|i| lifted_row(&pts[i].hom2::<Interval>(drop)));
        let ws_known = pts
            .iter()
            .all(|p| matches!(p.hom2::<Interval>(drop)[2].sign(), Some(s) if s != Sign::Zero));
        if ws_known {
            if let Some(sign) = det4(&rows).sign() {
                return Some(sign);
            }
        }
    }

    // Exact stage.
    for p in &pts {
        if p.hom2::<Expansion>(drop)[2].sign() == Sign::Zero {
            return None;
        }
    }
    let rows: [[Expansion; 4]; 4] =
        std::array::from_fn(|i| lifted_row(&pts[i].hom2::<Expansion>(drop)));
    Some(det4(&rows).sign())
}

/// Exact 2D orientation of three points in the axis-aligned projection that
/// drops the given axis. Points may be implicit.
///
/// Sign convention: in the projected coordinate pair (see
/// [`Point3::hom2`] for the cyclic pairing), positive when `a`, `b`, `c` are
/// counterclockwise -- equivalently, the sign of the `drop` component of the
/// normal of triangle (a, b, c) in 3D.
///
/// Returns `None` if any implicit point is invalid (exact w == 0).
pub fn orient2d(a: &Point3, b: &Point3, c: &Point3, drop: Axis) -> Option<Sign> {
    // Fast adaptive path: all points explicit.
    if let (Some(pa), Some(pb), Some(pc)) = (a.as_explicit(), b.as_explicit(), c.as_explicit()) {
        let proj = |p: [f64; 3]| drop.project(p);
        return Some(Sign::of_f64(geometry_predicates::orient2d(
            proj(pa),
            proj(pb),
            proj(pc),
        )));
    }

    let pts = [a, b, c];

    // det3 of homogeneous rows = (prod of w_i) * det2[[a-c],[b-c]].

    // Interval filter.
    'filter: {
        let homs: [[Interval; 3]; 3] = std::array::from_fn(|i| pts[i].hom2::<Interval>(drop));
        let Some(mut sign) = det3(&homs).sign() else {
            break 'filter;
        };
        for h in &homs {
            match h[2].sign() {
                Some(Sign::Positive) => {}
                Some(Sign::Negative) => sign = sign.flip(),
                _ => break 'filter,
            }
        }
        return Some(sign);
    }

    // Exact stage.
    let homs: [[Expansion; 3]; 3] = std::array::from_fn(|i| pts[i].hom2::<Expansion>(drop));
    let mut sign = det3(&homs).sign();
    for h in &homs {
        match h[2].sign() {
            Sign::Zero => return None,
            s => sign = sign.combine(s),
        }
    }
    Some(sign)
}

#[cfg(test)]
mod prepared_tests {
    use super::*;

    /// The prepared test against the general one: explicit, line-plane and
    /// barycenter points, on and off the plane of the explicit three.
    #[test]
    fn prepared_orientation_agrees() {
        let mut s = 0xa409_3822_299f_31d0u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 8.0).round() / 4.0 - 1.0
        };
        let mut on_plane = 0;
        for case in 0..3000 {
            let mut pt = || [r(), r(), r()];
            let (a, b, c) = (pt(), pt(), pt());
            // Line-plane points: on the plane abc for every third case.
            let lpi = |r: &mut dyn FnMut() -> f64, on: bool| {
                let (p, q) = ([r(), r(), r()], [r(), r(), r()]);
                let (u, v, w) = if on {
                    (a, b, c)
                } else {
                    ([r(), r(), r()], [r(), r(), r()], [r(), r(), r()])
                };
                Point3::lpi(p, q, u, v, w)
            };
            let on = case % 3 == 0;
            let p = match case % 4 {
                0 => Point3::explicit(r(), r(), r()),
                1 | 2 => lpi(&mut r, on),
                _ => Point3::bary(lpi(&mut r, on), lpi(&mut r, on), lpi(&mut r, on)),
            };
            let (ea, eb, ec) = (
                Point3::Explicit(a),
                Point3::Explicit(b),
                Point3::Explicit(c),
            );
            let want = orient3d(&ea, &eb, &ec, &p);
            on_plane += usize::from(want == Some(Sign::Zero));
            assert_eq!(
                orient3d_explicit(a, b, c, &Prepared3::new(p)),
                want,
                "case {case}"
            );
        }
        assert!(on_plane > 100, "only {on_plane} points on the plane");
    }
}
