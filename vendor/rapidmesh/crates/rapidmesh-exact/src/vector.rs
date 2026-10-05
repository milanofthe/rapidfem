//! The plain float vector algebra every crate shares: points and vectors in
//! 2 and 3 dimensions as arrays, the operations on them (written once for
//! any dimension where they make sense), orthonormal frames and affine maps.
//!
//! Conventions: [`len`] is the magnitude; [`unit`] the direction or `None`
//! for a zero or non-finite vector; [`normalize`] the direction, the input
//! back for a zero vector. Angles are in radians.

use std::f64::consts::{PI, TAU};

/// A point or vector in the plane.
pub type V2 = [f64; 2];
/// A point or vector in space.
pub type V3 = [f64; 3];
/// A 3x3 matrix by rows.
pub type M3 = [[f64; 3]; 3];

/// The 3x3 identity.
pub const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// `a + b`.
#[inline]
pub fn add<const N: usize>(a: [f64; N], b: [f64; N]) -> [f64; N] {
    std::array::from_fn(|k| a[k] + b[k])
}

/// `a - b`.
#[inline]
pub fn sub<const N: usize>(a: [f64; N], b: [f64; N]) -> [f64; N] {
    std::array::from_fn(|k| a[k] - b[k])
}

/// `a * s`.
#[inline]
pub fn scale<const N: usize>(a: [f64; N], s: f64) -> [f64; N] {
    a.map(|x| x * s)
}

/// `a + s * d`: the point `s` along `d` from `a`.
#[inline]
pub fn along<const N: usize>(a: [f64; N], d: [f64; N], s: f64) -> [f64; N] {
    std::array::from_fn(|k| a[k] + s * d[k])
}

/// The point at the share `t` from `a` to `b`.
#[inline]
pub fn lerp<const N: usize>(a: [f64; N], b: [f64; N], t: f64) -> [f64; N] {
    std::array::from_fn(|k| a[k] + t * (b[k] - a[k]))
}

/// The middle of `a` and `b`.
#[inline]
pub fn mid<const N: usize>(a: [f64; N], b: [f64; N]) -> [f64; N] {
    std::array::from_fn(|k| 0.5 * (a[k] + b[k]))
}

/// `a . b`.
#[inline]
pub fn dot<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
    (0..N).map(|k| a[k] * b[k]).sum()
}

/// `|a|`.
#[inline]
pub fn len<const N: usize>(a: [f64; N]) -> f64 {
    dot(a, a).sqrt()
}

/// `|a - b|^2`.
#[inline]
pub fn dist2<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

/// `|a - b|`.
#[inline]
pub fn dist<const N: usize>(a: [f64; N], b: [f64; N]) -> f64 {
    dist2(a, b).sqrt()
}

/// The unit vector along `a`, `None` for a zero or non-finite one.
#[inline]
pub fn unit<const N: usize>(a: [f64; N]) -> Option<[f64; N]> {
    let l = len(a);
    (l > 0.0 && l.is_finite()).then(|| a.map(|x| x / l))
}

/// The unit vector along `a`; `a` back when it has no direction.
#[inline]
pub fn normalize<const N: usize>(a: [f64; N]) -> [f64; N] {
    let l = len(a);
    if l > 0.0 {
        scale(a, 1.0 / l)
    } else {
        a
    }
}

/// The bounding box `(lo, hi)` of points (`lo` at `f64::MAX`, `hi` at
/// `f64::MIN` for none).
pub fn bbox<const N: usize, P: std::borrow::Borrow<[f64; N]>>(
    pts: impl IntoIterator<Item = P>,
) -> ([f64; N], [f64; N]) {
    pts.into_iter()
        .fold(([f64::MAX; N], [f64::MIN; N]), |(lo, hi), p| {
            let p = p.borrow();
            (
                std::array::from_fn(|k| lo[k].min(p[k])),
                std::array::from_fn(|k| hi[k].max(p[k])),
            )
        })
}

/// Squared distance from `p` to the box `lo..hi` (0 inside).
#[inline]
pub fn box_d2<const N: usize>(lo: [f64; N], hi: [f64; N], p: [f64; N]) -> f64 {
    (0..N)
        .map(|k| (lo[k] - p[k]).max(0.0).max(p[k] - hi[k]).powi(2))
        .sum()
}

/// The mean of points.
pub fn centroid<const N: usize, P: std::borrow::Borrow<[f64; N]>>(
    ps: impl IntoIterator<Item = P>,
) -> [f64; N] {
    let (mut s, mut n) = ([0.0; N], 0usize);
    for p in ps {
        s = add(s, *p.borrow());
        n += 1;
    }
    s.map(|x| x / n as f64)
}

/// The share along the segment `a b` of the point of it nearest `p`, in
/// `[0, 1]` (0 for a segment of no length).
#[inline]
pub fn segment_param<const N: usize>(p: [f64; N], a: [f64; N], b: [f64; N]) -> f64 {
    let d = sub(b, a);
    let l2 = dot(d, d);
    if l2 > 0.0 {
        (dot(sub(p, a), d) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The point of the segment `a b` nearest `p`.
#[inline]
pub fn closest_on_segment<const N: usize>(p: [f64; N], a: [f64; N], b: [f64; N]) -> [f64; N] {
    lerp(a, b, segment_param(p, a, b))
}

/// The squared distance from `p` to the segment `a b`.
#[inline]
pub fn segment_dist2<const N: usize>(p: [f64; N], a: [f64; N], b: [f64; N]) -> f64 {
    dist2(p, closest_on_segment(p, a, b))
}

/// `a x b`.
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The z component of `a x b` for plane vectors (twice the signed area of
/// the triangle `0 a b`).
#[inline]
pub fn cross2(a: V2, b: V2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// The normal of the triangle `a b c` with the length of twice its area,
/// turned by the right hand along `a b c`.
#[inline]
pub fn tri_normal(a: V3, b: V3, c: V3) -> V3 {
    cross(sub(b, a), sub(c, a))
}

/// The normal of a closed polygon by Newell's method, its length twice the
/// area: exact for a planar polygon, the best plane's for a bent one.
pub fn newell(pts: &[V3]) -> V3 {
    let mut n = [0.0; 3];
    for (i, p) in pts.iter().enumerate() {
        let q = pts[(i + 1) % pts.len()];
        n[0] += (p[1] - q[1]) * (p[2] + q[2]);
        n[1] += (p[2] - q[2]) * (p[0] + q[0]);
        n[2] += (p[0] - q[0]) * (p[1] + q[1]);
    }
    n
}

/// The determinant of a 3x3 matrix.
#[inline]
pub fn det3(m: M3) -> f64 {
    dot(m[0], cross(m[1], m[2]))
}

/// `m v`.
#[inline]
pub fn mul_vec(m: M3, v: V3) -> V3 {
    m.map(|row| dot(row, v))
}

/// `a b`.
pub fn mul(a: M3, b: M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// The transpose of `m`.
pub fn transpose(m: M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| m[j][i]))
}

/// The cofactor matrix of `m` (its determinant times its inverse
/// transpose): it maps the normals of a plane the way `m` maps the plane,
/// and stays defined where `m` is singular.
pub fn cofactor(m: M3) -> M3 {
    let e = |i: usize, j: usize| m[i % 3][j % 3];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            e(i + 1, j + 1) * e(i + 2, j + 2) - e(i + 1, j + 2) * e(i + 2, j + 1)
        })
    })
}

/// The inverse of `m`, `None` where it is singular.
pub fn inverse(m: M3) -> Option<M3> {
    let det = det3(m);
    (det != 0.0 && det.is_finite()).then(|| transpose(cofactor(m)).map(|r| scale(r, 1.0 / det)))
}

/// A vector square to `n` (not unit): `n` crossed with the axis it has
/// the least of.
#[inline]
pub fn perp(n: V3) -> V3 {
    let k = (0..3)
        .min_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
        .unwrap_or(0);
    let mut e = [0.0; 3];
    e[k] = 1.0;
    cross(n, e)
}

/// The unit vector square to the unit `a` where nothing else fixes one (the
/// angle 0 of a carrier about `a`, the `x` of a frame without a hint): `a`
/// crossed with the x axis, or with the y axis where `a` lies close to x.
#[inline]
pub fn ortho_unit(a: V3) -> V3 {
    let t = if a[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize(cross(a, t))
}

/// The rotation by `angle` about the unit `axis` (right-handed; Rodrigues).
pub fn rotation(axis: V3, angle: f64) -> M3 {
    let (s, c) = angle.sin_cos();
    let [x, y, z] = axis;
    let k = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| IDENTITY[i][j] * c + s * k[i][j] + (1.0 - c) * axis[i] * axis[j])
    })
}

/// `v` turned by the least rotation that takes the unit `a` to the unit
/// `b` (`v` back where they are opposite: no least one).
pub fn turn(v: V3, a: V3, b: V3) -> V3 {
    let k = cross(a, b);
    let c = dot(a, b);
    if c <= -1.0 + 1e-12 {
        return v;
    }
    add(
        add(scale(v, c), cross(k, v)),
        scale(k, dot(k, v) / (1.0 + c)),
    )
}

/// The angle of `d` about an axis, from the unit `x` towards the unit `y`
/// square to it, in `(-pi, pi]`.
#[inline]
pub fn angle_about(d: V3, x: V3, y: V3) -> f64 {
    dot(d, y).atan2(dot(d, x))
}

/// `a` wrapped into `(-pi, pi]`.
#[inline]
pub fn wrap_pm(a: f64) -> f64 {
    let x = a.rem_euclid(TAU);
    if x > PI {
        x - TAU
    } else {
        x
    }
}

/// `a` shifted by whole `period`s to the nearest of `near`.
#[inline]
pub fn wrap_near(a: f64, near: f64, period: f64) -> f64 {
    a - period * ((a - near) / period).round()
}

/// The point of the triangle `a b c` nearest `p`.
pub fn closest_on_tri(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return along(a, ab, d1 / (d1 - d3));
    }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return along(a, ac, d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return lerp(b, c, (d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    along(along(a, ab, vb * denom), ac, vc * denom)
}

/// An orthonormal frame: the origin `o` and the axes `x`, `y`, `z`
/// (right-handed as [`Frame::new`] makes it; a mirror turns it
/// left-handed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub o: V3,
    pub x: V3,
    pub y: V3,
    pub z: V3,
}

impl Frame {
    /// The frame at `o` with the axis `z` and `x` towards `x_hint` (its part
    /// square to `z`); without a hint (or one along `z`), [`ortho_unit`] of
    /// `z`. `None` for a zero `z`.
    pub fn new(o: V3, z: V3, x_hint: Option<V3>) -> Option<Frame> {
        let z = unit(z)?;
        let square = |h: V3| unit(sub(h, scale(z, dot(h, z))));
        let x = x_hint.and_then(square).unwrap_or_else(|| ortho_unit(z));
        Some(Frame {
            o,
            x,
            y: cross(z, x),
            z,
        })
    }

    /// The point with the coordinates `(a, b, c)` in the frame.
    #[inline]
    pub fn at(&self, a: f64, b: f64, c: f64) -> V3 {
        along(along(along(self.o, self.x, a), self.y, b), self.z, c)
    }

    /// The coordinates of `p` in the frame.
    #[inline]
    pub fn local(&self, p: V3) -> V3 {
        let d = sub(p, self.o);
        [dot(d, self.x), dot(d, self.y), dot(d, self.z)]
    }

    /// The angle of `p` about the axis, from `x` towards `y`.
    #[inline]
    pub fn angle(&self, p: V3) -> f64 {
        angle_about(sub(p, self.o), self.x, self.y)
    }

    /// The frame carried by the affine map `m` (whose linear part keeps
    /// angles: a move, turn, mirror or uniform stretch), its axes kept
    /// unit.
    pub fn mapped(&self, m: &Affine) -> Frame {
        let axis = |d: V3| normalize(m.vector(d));
        Frame {
            o: m.point(self.o),
            x: axis(self.x),
            y: axis(self.y),
            z: axis(self.z),
        }
    }
}

/// An affine map `p -> linear p + offset`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine {
    pub linear: M3,
    pub offset: V3,
}

impl Affine {
    pub const IDENTITY: Affine = Affine {
        linear: IDENTITY,
        offset: [0.0; 3],
    };

    /// The move by `v`.
    pub fn translation(v: V3) -> Affine {
        Affine {
            linear: IDENTITY,
            offset: v,
        }
    }

    /// The linear map `m` about the fixed point `center`.
    pub fn about(center: V3, m: M3) -> Affine {
        Affine {
            linear: m,
            offset: sub(center, mul_vec(m, center)),
        }
    }

    /// The turn by `angle` about the axis along `axis` through `center`
    /// (right-handed); `None` for a zero axis.
    pub fn rotation(center: V3, axis: V3, angle: f64) -> Option<Affine> {
        Some(Affine::about(center, rotation(unit(axis)?, angle)))
    }

    /// The mirror across the plane through `point` square to `normal`;
    /// `None` for a zero normal.
    pub fn mirror(point: V3, normal: V3) -> Option<Affine> {
        let n = unit(normal)?;
        let m =
            std::array::from_fn(|i| std::array::from_fn(|j| IDENTITY[i][j] - 2.0 * n[i] * n[j]));
        Some(Affine::about(point, m))
    }

    /// The stretch by `factors` along x, y and z about `center`.
    pub fn stretch(center: V3, factors: V3) -> Affine {
        let m = std::array::from_fn(|i| std::array::from_fn(|j| IDENTITY[i][j] * factors[i]));
        Affine::about(center, m)
    }

    /// This map followed by `next`.
    pub fn then(&self, next: &Affine) -> Affine {
        Affine {
            linear: mul(next.linear, self.linear),
            offset: next.point(self.offset),
        }
    }

    /// The image of the point `p`.
    #[inline]
    pub fn point(&self, p: V3) -> V3 {
        add(mul_vec(self.linear, p), self.offset)
    }

    /// The image of the direction `d`.
    #[inline]
    pub fn vector(&self, d: V3) -> V3 {
        mul_vec(self.linear, d)
    }

    /// The image of the normal `n` of a plane (not unit; turned with the
    /// winding where the map mirrors).
    #[inline]
    pub fn normal(&self, n: V3) -> V3 {
        mul_vec(cofactor(self.linear), n)
    }

    /// The determinant of the linear part (negative for a mirror).
    pub fn det(&self) -> f64 {
        det3(self.linear)
    }

    /// The inverse map, `None` where it is singular.
    pub fn inverse(&self) -> Option<Affine> {
        let linear = inverse(self.linear)?;
        Some(Affine {
            linear,
            offset: scale(mul_vec(linear, self.offset), -1.0),
        })
    }

    /// The single factor every length is stretched by, where there is one
    /// (a move, turn or mirror: 1; a uniform stretch: its factor).
    pub fn uniform_factor(&self) -> Option<f64> {
        let cols = transpose(self.linear);
        let s = len(cols[0]);
        let tol = 1e-12 * s.max(1.0);
        let same = cols.iter().all(|c| (len(*c) - s).abs() <= tol)
            && (0..3).all(|i| (i + 1..3).all(|j| dot(cols[i], cols[j]).abs() <= tol * s));
        (s > 0.0 && same).then_some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: V3, b: V3) -> bool {
        dist(a, b) < 1e-12
    }

    #[test]
    fn a_frame_is_orthonormal_and_inverts() {
        let f = Frame::new([1.0, 2.0, 3.0], [0.0, 0.0, 2.0], Some([1.0, 1.0, 5.0])).unwrap();
        assert!((len(f.x) - 1.0).abs() < 1e-15 && dot(f.x, f.z).abs() < 1e-15);
        assert!(close(f.x, normalize([1.0, 1.0, 0.0])));
        let p = [0.3, -0.7, 9.0];
        let l = f.local(p);
        assert!(close(f.at(l[0], l[1], l[2]), p));
        let g = Frame::new([0.0; 3], [1.0, 0.0, 0.0], None).unwrap();
        assert!(dot(g.x, g.z).abs() < 1e-15 && (det3([g.x, g.y, g.z]) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn affine_maps_compose_and_invert() {
        let r = Affine::rotation(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            std::f64::consts::FRAC_PI_2,
        )
        .unwrap();
        assert!(close(r.point([2.0, 0.0, 0.0]), [1.0, 1.0, 0.0]));
        let m = Affine::mirror([0.0, 0.0, 1.0], [0.0, 0.0, 3.0]).unwrap();
        assert!(close(m.point([5.0, 6.0, 0.0]), [5.0, 6.0, 2.0]) && m.det() < 0.0);
        let s = Affine::stretch([1.0; 3], [2.0, 2.0, 2.0]);
        let all = r.then(&m).then(&s);
        let p = [0.4, -1.2, 3.3];
        assert!(close(all.inverse().unwrap().point(all.point(p)), p));
        assert_eq!(
            all.uniform_factor().map(|f| (f * 1e9).round() / 1e9),
            Some(2.0)
        );
        assert_eq!(
            Affine::stretch([0.0; 3], [1.0, 2.0, 1.0]).uniform_factor(),
            None
        );
        // A normal stays square to the mapped plane.
        let t = Affine::stretch([0.0; 3], [1.0, 3.0, 0.5]).then(&r);
        let (u, v) = ([1.0, 2.0, 0.0], [0.0, 1.0, 4.0]);
        let n = t.normal(cross(u, v));
        assert!(dot(n, t.vector(u)).abs() < 1e-12 && dot(n, t.vector(v)).abs() < 1e-12);
    }

    #[test]
    fn closest_points_on_segments_and_triangles() {
        let (a, b) = ([0.0, 0.0], [2.0, 0.0]);
        assert_eq!(closest_on_segment([1.0, 3.0], a, b), [1.0, 0.0]);
        assert_eq!(closest_on_segment([-1.0, 3.0], a, b), a);
        assert_eq!(segment_dist2([3.0, 1.0], a, b), 2.0);
        let t = ([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(close(
            closest_on_tri([0.2, 0.2, 5.0], t.0, t.1, t.2),
            [0.2, 0.2, 0.0]
        ));
        assert!(close(
            closest_on_tri([2.0, 2.0, 0.0], t.0, t.1, t.2),
            [0.5, 0.5, 0.0]
        ));
    }

    #[test]
    fn angles_wrap() {
        assert!((wrap_pm(3.0 * PI) - PI).abs() < 1e-12);
        assert!((wrap_near(0.1 + TAU, 0.0, TAU) - 0.1).abs() < 1e-12);
        let r = rotation([0.0, 0.0, 1.0], 0.3);
        let v = mul_vec(r, [1.0, 0.0, 0.0]);
        assert!((angle_about(v, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) - 0.3).abs() < 1e-15);
        assert!(close(
            turn([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            [1.0, 0.0, 0.0]
        ));
    }
}
