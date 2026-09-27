//! Staged-exact orientation predicates over explicit and implicit points.

use crate::expansion::Expansion;
use crate::geom::{det3, det4, det4_lift, det5};
use crate::interval::Interval;
use crate::point::Point3;
use crate::{Axis, Sign};

/// Exact 3D orientation of four points, any of which may be implicit.
///
/// Sign convention: equals the sign of det [[a-d], [b-d], [c-d]] (rows), the
/// same convention as Shewchuk's `orient3d` — positive when `d` lies below the
/// plane through `a`, `b`, `c` oriented counterclockwise as seen from above
/// the plane.
///
/// Returns `None` if any implicit point is invalid (its defining primitives
/// do not intersect in a single point, exact w == 0).
///
/// Evaluation is staged: fast adaptive path for all-explicit inputs
/// (`geometry-predicates`), conservative interval filter for implicit inputs,
/// exact expansion arithmetic as the final word.
/// Affine interval coordinates for points whose homogeneous w is exactly 1
/// (explicit, Lnc, Pac); `None` for the projective kinds (Lpi, Tpi, Bary).
/// Lets the filters of [`orient3d`] and [`insphere3d`] use the plain affine
/// difference determinants instead of the homogeneous lifts: w-sign folding
/// disappears and the determinant shrinks by one dimension, several times
/// fewer interval operations on the dominant Lnc/Pac meshing path.
#[inline]
fn affine_interval(p: &Point3) -> Option<[Interval; 3]> {
    match p {
        Point3::Explicit(c) => Some(c.map(Interval::point)),
        Point3::Lnc { .. } | Point3::Pac { .. } => {
            let h = p.hom::<Interval>();
            Some([h[0], h[1], h[2]])
        }
        _ => None,
    }
}

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

    // Affine-reduction fast path. orient3d is multilinear in each argument, so
    // if exactly one point is an affine Steiner point (Lnc) -- a convex
    // combination `(1-t) a + t b` of explicit parents with t in (0, 1) -- its
    // orientation is `(1-t) O_a + t O_b` where O_a, O_b are the orientations
    // with the parent substituted in. With both weights strictly positive, when
    // O_a and O_b share a sign (or are zero) the point shares it; each O_i is a
    // fully explicit predicate (the fast adaptive path). This resolves the
    // common "Steiner clearly on one side of the facet plane" case without the
    // implicit interval / expansion machinery; the straddling case (O_a, O_b
    // opposite) falls through to the exact stages below. Exact: the identity is
    // a real-number identity (the point IS `(1-t) a + t b` exactly) and the
    // per-parent signs are exact. (Lpi/Tpi/Bary and Pac -- whose `1-u-v` weight
    // is not as cheaply sign-certified in f64 -- take the path below.)
    {
        let mut implicit_idx: Option<usize> = None;
        let mut explicit = [[0.0f64; 3]; 4];
        let mut single = true;
        for (k, p) in pts.iter().enumerate() {
            match p.as_explicit() {
                Some(c) => explicit[k] = c,
                None if implicit_idx.is_none() => implicit_idx = Some(k),
                None => {
                    single = false;
                    break;
                }
            }
        }
        if single {
            if let Some(k) = implicit_idx {
                if let Some((parents, weights, 2)) = pts[k].affine_combo() {
                    if weights[0] > 0.0 && weights[1] > 0.0 {
                        let (mut pos, mut neg) = (false, false);
                        for parent in &parents[..2] {
                            let mut q = explicit;
                            q[k] = *parent;
                            match Sign::of_f64(geometry_predicates::orient3d(
                                q[0], q[1], q[2], q[3],
                            )) {
                                Sign::Positive => pos = true,
                                Sign::Negative => neg = true,
                                Sign::Zero => {}
                            }
                        }
                        if !(pos && neg) {
                            return Some(if pos {
                                Sign::Positive
                            } else if neg {
                                Sign::Negative
                            } else {
                                Sign::Zero
                            });
                        }
                    }
                }
            }
        }
    }

    // Affine interval filter (all w exactly 1: explicit, Lnc, Pac — the
    // dominant meshing path): the homogeneous det4 equals det3 of the rows
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
        let proj = |p: [f64; 3]| match drop {
            Axis::X => [p[1], p[2]],
            Axis::Y => [p[2], p[0]],
            Axis::Z => [p[0], p[1]],
        };
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

/// Exact 3D in-sphere test, any point may be implicit.
///
/// Positive iff `e` lies strictly inside the circumsphere of the POSITIVELY
/// ORIENTED tetrahedron (a, b, c, d) (Shewchuk's `insphere` convention; the
/// sign flips for a negatively oriented tet). Returns `None` if any point is
/// invalid (exact w == 0).
///
/// Homogeneous lifting: the classic row (x, y, z, x^2 + y^2 + z^2, 1) scaled
/// by w^2 becomes (X W, Y W, Z W, X^2 + Y^2 + Z^2, W^2), polynomial in the
/// homogeneous coordinates; the scaling factors w^2 are strictly positive
/// for valid points, so the determinant sign needs no w correction.
pub fn insphere3d(a: &Point3, b: &Point3, c: &Point3, d: &Point3, e: &Point3) -> Option<Sign> {
    // Fast adaptive path: all points explicit.
    if let (Some(pa), Some(pb), Some(pc), Some(pd), Some(pe)) = (
        a.as_explicit(),
        b.as_explicit(),
        c.as_explicit(),
        d.as_explicit(),
        e.as_explicit(),
    ) {
        return Some(Sign::of_f64(geometry_predicates::insphere(
            pa, pb, pc, pd, pe,
        )));
    }

    fn lifted_row<T: crate::ring::Ring>(h: &[T; 4]) -> [T; 5] {
        let (x, y, z, w) = (&h[0], &h[1], &h[2], &h[3]);
        [
            x.mul(w),
            y.mul(w),
            z.mul(w),
            x.mul(x).add(&y.mul(y)).add(&z.mul(z)),
            w.mul(w),
        ]
    }

    let pts = [a, b, c, d, e];

    // Affine interval filter (all w exactly 1): column operations reduce the
    // homogeneous 5x5 lift to Shewchuk's difference form, det4 of rows
    // (p - e, |p - e|^2) — several times fewer interval operations than the
    // projective det5. This is the dominant meshing path (explicit, Lnc and
    // Pac points all have w = 1); when indecisive it falls through to the
    // exact stage directly (the projective filter sees the same widths).
    if let (Some(pa), Some(pb), Some(pc), Some(pd), Some(pe)) = (
        affine_interval(a),
        affine_interval(b),
        affine_interval(c),
        affine_interval(d),
        affine_interval(e),
    ) {
        let row = |p: &[Interval; 3]| -> [Interval; 4] {
            let d: [Interval; 3] = std::array::from_fn(|k| p[k].sub(pe[k]));
            let lift = d[0].mul(d[0]).add(d[1].mul(d[1])).add(d[2].mul(d[2]));
            [d[0], d[1], d[2], lift]
        };
        if let Some(sign) = det4_lift(&[row(&pa), row(&pb), row(&pc), row(&pd)]).sign() {
            return Some(sign);
        }
    } else {
        // Projective interval filter (some w != 1).
        let homs: [[Interval; 4]; 5] = std::array::from_fn(|i| pts[i].hom::<Interval>());
        let ws_known = homs
            .iter()
            .all(|h| matches!(h[3].sign(), Some(s) if s != Sign::Zero));
        if ws_known {
            let rows: [[Interval; 5]; 5] = std::array::from_fn(|i| lifted_row(&homs[i]));
            if let Some(sign) = det5(&rows).sign() {
                return Some(sign);
            }
        }
    }

    // Exact stage.
    let homs: [[Expansion; 4]; 5] = std::array::from_fn(|i| pts[i].hom::<Expansion>());
    for h in &homs {
        if h[3].sign() == Sign::Zero {
            return None;
        }
    }
    let rows: [[Expansion; 5]; 5] = std::array::from_fn(|i| lifted_row(&homs[i]));
    Some(det5(&rows).sign())
}

/// Exact weighted in-sphere (power / regularity) test over EXPLICIT points --
/// the sliver-exudation predicate. Positive iff the weighted point `(e, we)`
/// has NEGATIVE power distance to the orthosphere of the positively oriented
/// weighted tet `(a, wa) .. (d, wd)`: `e` violates the tet's regularity, so a
/// regular (weighted-Delaunay) flip removes the shared facet. With all
/// weights zero this is exactly [`insphere3d`]'s convention.
///
/// Shewchuk difference form with the weight correction on the lift:
/// `sign det4 of rows (p - e, |p - e|^2 - (wp - we))`. Staged: a static f64
/// filter, exact expansion arithmetic when indecisive. An interval stage in
/// between decided well under 1% of the calls the static filter left open
/// (those are nearly cospherical, e.g. points on one circle), so there is
/// none. Points here are plain f64, so no implicit machinery is involved.
pub fn power_test3d(
    a: [f64; 3],
    wa: f64,
    b: [f64; 3],
    wb: f64,
    c: [f64; 3],
    wc: f64,
    d: [f64; 3],
    wd: f64,
    e: [f64; 3],
    we: f64,
) -> Sign {
    fn wrow<T: crate::ring::Ring>(p: [f64; 3], wp: f64, e: [f64; 3], we: f64) -> [T; 4] {
        let d: [T; 3] = std::array::from_fn(|k| T::from_f64(p[k]).sub(&T::from_f64(e[k])));
        let lift = d[0]
            .mul(&d[0])
            .add(&d[1].mul(&d[1]))
            .add(&d[2].mul(&d[2]))
            .sub(&T::from_f64(wp).sub(&T::from_f64(we)));
        [d[0].clone(), d[1].clone(), d[2].clone(), lift]
    }
    let pts = [(a, wa), (b, wb), (c, wc), (d, wd)];
    // Static filter: the determinant in plain f64 against a bound on its
    // rounding error (a multiple of the unit roundoff times the permanent,
    // the determinant of the absolute values). Decides almost every call.
    {
        let rows: [[f64; 4]; 4] = std::array::from_fn(|i| {
            let (p, wp) = pts[i];
            let d = [p[0] - e[0], p[1] - e[1], p[2] - e[2]];
            [
                d[0],
                d[1],
                d[2],
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - (wp - we),
            ]
        });
        // The lift's error scales with |d|^2 + |dw|, not with its possibly
        // cancelled value.
        let abs: [[f64; 4]; 4] = std::array::from_fn(|i| {
            let r = rows[i];
            let lift = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + (pts[i].1 - we).abs();
            [r[0].abs(), r[1].abs(), r[2].abs(), lift]
        });
        let det = det4_f64(&rows);
        let perm = permanent4(&abs);
        // 64 unit roundoffs per unit of permanent covers the differences,
        // the lift and the 4x4 expansion with a wide margin.
        let bound = 64.0 * f64::EPSILON * perm;
        if det > bound {
            return Sign::Positive;
        }
        if det < -bound {
            return Sign::Negative;
        }
    }
    let rows: [[Expansion; 4]; 4] =
        std::array::from_fn(|i| wrow::<Expansion>(pts[i].0, pts[i].1, e, we));
    det4_lift(&rows).sign()
}

/// Determinant of a 4x4 f64 matrix by expansion along the first row (for
/// the static filter only: rounded, not exact).
fn det4_f64(m: &[[f64; 4]; 4]) -> f64 {
    let det3 = |r: [[f64; 3]; 3]| -> f64 {
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    };
    (0..4)
        .map(|col| {
            let minor: [[f64; 3]; 3] = std::array::from_fn(|i| {
                let row = &m[i + 1];
                let mut it = (0..4).filter(|&j| j != col);
                std::array::from_fn(|_| row[it.next().expect("3 columns remain")])
            });
            let s = if col % 2 == 0 { 1.0 } else { -1.0 };
            s * m[0][col] * det3(minor)
        })
        .sum()
}

/// Permanent of a 4x4 matrix of non-negative entries (the determinant's
/// expansion with every sign positive): bounds the magnitude every term of
/// the determinant can reach.
fn permanent4(m: &[[f64; 4]; 4]) -> f64 {
    let perm3 = |r: [[f64; 3]; 3]| -> f64 {
        r[0][0] * (r[1][1] * r[2][2] + r[1][2] * r[2][1])
            + r[0][1] * (r[1][0] * r[2][2] + r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] + r[1][1] * r[2][0])
    };
    (0..4)
        .map(|col| {
            let minor: [[f64; 3]; 3] = std::array::from_fn(|i| {
                let row = &m[i + 1];
                let mut it = (0..4).filter(|&j| j != col);
                std::array::from_fn(|_| row[it.next().expect("3 columns remain")])
            });
            m[0][col] * perm3(minor)
        })
        .sum()
}

/// Exact 2D orientation of three points in the axis-aligned projection that
/// drops the given axis. Points may be implicit.
///
/// Sign convention: in the projected coordinate pair (see
/// [`Point3::hom2`] for the cyclic pairing), positive when `a`, `b`, `c` are
/// counterclockwise — equivalently, the sign of the `drop` component of the
/// normal of triangle (a, b, c) in 3D.
///
/// Returns `None` if any implicit point is invalid (exact w == 0).
pub fn orient2d(a: &Point3, b: &Point3, c: &Point3, drop: Axis) -> Option<Sign> {
    // Fast adaptive path: all points explicit.
    if let (Some(pa), Some(pb), Some(pc)) = (a.as_explicit(), b.as_explicit(), c.as_explicit()) {
        let proj = |p: [f64; 3]| match drop {
            Axis::X => [p[1], p[2]],
            Axis::Y => [p[2], p[0]],
            Axis::Z => [p[0], p[1]],
        };
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
mod power_filter_tests {
    use super::*;

    /// The exact sign, bypassing both filters.
    fn exact(pts: [([f64; 3], f64); 4], e: [f64; 3], we: f64) -> Sign {
        let row = |p: [f64; 3], wp: f64| -> [Expansion; 4] {
            let d: [Expansion; 3] =
                std::array::from_fn(|k| Expansion::from_f64(p[k]).sub(&Expansion::from_f64(e[k])));
            let lift = d[0]
                .mul(&d[0])
                .add(&d[1].mul(&d[1]))
                .add(&d[2].mul(&d[2]))
                .sub(&Expansion::from_f64(wp).sub(&Expansion::from_f64(we)));
            [d[0].clone(), d[1].clone(), d[2].clone(), lift]
        };
        let rows: [[Expansion; 4]; 4] = std::array::from_fn(|i| row(pts[i].0, pts[i].1));
        det4(&rows).sign()
    }

    #[test]
    fn filters_agree_with_exact_arithmetic() {
        let mut s = 0x243f_6a88_85a3_08d3u64;
        let mut r = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..20_000 {
            // Random, grid-snapped (cospherical and coplanar ties) and tiny
            // scaled configurations, with and without weights.
            let snap = case % 3 == 1;
            let scale = if case % 5 == 0 { 1e-3 } else { 1.0 };
            let pt = |r: &mut dyn FnMut() -> f64| -> [f64; 3] {
                std::array::from_fn(|_| {
                    let v = r();
                    scale * if snap { (v * 4.0).round() / 4.0 } else { v }
                })
            };
            let pts: [([f64; 3], f64); 4] = std::array::from_fn(|_| {
                let p = pt(&mut r);
                let w = if case % 2 == 0 {
                    (scale * 0.3 * r()).powi(2)
                } else {
                    0.0
                };
                (p, w)
            });
            let e = pt(&mut r);
            let we = if case % 4 == 0 {
                (scale * 0.3 * r()).powi(2)
            } else {
                0.0
            };
            let [(a, wa), (b, wb), (c, wc), (d, wd)] = pts;
            let got = power_test3d(a, wa, b, wb, c, wc, d, wd, e, we);
            assert_eq!(got, exact(pts, e, we), "case {case}");
        }
    }
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
