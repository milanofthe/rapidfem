//! The 3-vector helpers of every crate above `rapidmesh-csg` (csg sits
//! below geom, `rapidmesh-exact` is generic over its rings, and the core of
//! `rapidmesh-topo` builds without geom). `len` is the magnitude,
//! `normalize` and `unit` the direction (the input back for a zero vector,
//! or `None`).

/// A point or vector in 3-space.
pub type V3 = [f64; 3];

/// Component-wise difference `a - b`.
#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise sum `a + b`.
#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scalar multiple `a * s`.
#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product `a . b`.
#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b`.
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean magnitude `|a|`.
#[inline]
pub fn len(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// Euclidean distance `|a - b|`.
#[inline]
pub fn dist(a: V3, b: V3) -> f64 {
    len(sub(a, b))
}

/// Unit vector in the direction of `a`; returns `a` unchanged when `a` is the
/// zero vector (degenerate, no defined direction).
#[inline]
pub fn normalize(a: V3) -> V3 {
    let l = len(a);
    if l > 0.0 {
        scale(a, 1.0 / l)
    } else {
        a
    }
}

/// Squared distance `|a - b|^2`.
#[inline]
pub fn dist2(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

/// The unit vector along `a`, `None` for a zero or non-finite one.
#[inline]
pub fn unit(a: V3) -> Option<V3> {
    let l = len(a);
    (l > 0.0 && l.is_finite()).then(|| a.map(|x| x / l))
}

/// A vector perpendicular to `n` (not unit): `n` crossed with the axis it
/// has the least of.
#[inline]
pub fn perp(n: V3) -> V3 {
    let k = (0..3)
        .min_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
        .unwrap_or(0);
    let mut e = [0.0; 3];
    e[k] = 1.0;
    cross(n, e)
}

/// Squared distance from `p` to the box `lo..hi` (0 inside).
#[inline]
pub fn box_d2(lo: V3, hi: V3, p: V3) -> f64 {
    let mut s = 0.0;
    for k in 0..3 {
        let d = (lo[k] - p[k]).max(0.0).max(p[k] - hi[k]);
        s += d * d;
    }
    s
}

/// The bounding box `(lo, hi)` of points (`lo` at `f64::MAX`, `hi` at
/// `f64::MIN` for none).
pub fn bbox<P: std::borrow::Borrow<V3>>(pts: impl IntoIterator<Item = P>) -> (V3, V3) {
    pts.into_iter()
        .fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), p| {
            let p = p.borrow();
            (
                std::array::from_fn(|k| lo[k].min(p[k])),
                std::array::from_fn(|k| hi[k].max(p[k])),
            )
        })
}

/// The mean of points.
pub fn centroid(ps: &[V3]) -> V3 {
    let n = ps.len() as f64;
    std::array::from_fn(|k| ps.iter().map(|p| p[k]).sum::<f64>() / n)
}

/// A unit vector square to the unit `a`: `a` crossed with the x axis, or
/// with the y axis where `a` lies close to x. The angle 0 of a carrier
/// about `a` where nothing else fixes it.
pub fn ortho_unit(a: V3) -> V3 {
    let t = if a[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    normalize(cross(a, t))
}
