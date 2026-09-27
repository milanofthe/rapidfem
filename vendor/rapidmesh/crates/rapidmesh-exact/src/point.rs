//! Explicit and implicit (lazily evaluated) 3D points.

use crate::expansion::Expansion;
use crate::geom::{lpi_hom, tpi_hom};
use crate::interval::Interval;
use crate::ring::Ring;
use crate::{Axis, Sign};

/// A 3D point, either explicit or defined implicitly by the primitives whose
/// intersection it is.
///
/// Implicit points are never rounded to f64 coordinates inside predicates:
/// every predicate evaluates their homogeneous coordinates symbolically (as
/// polynomials in the defining inputs) at the precision the staged evaluation
/// requires. This is what makes cascaded constructions (intersection points of
/// intersection segments, Steiner points on recovered boundaries) exact.
#[derive(Debug, Clone, PartialEq)]
pub enum Point3 {
    /// An ordinary coordinate point.
    Explicit([f64; 3]),
    /// Line-plane intersection: line through `p`, `q`; plane through
    /// `r`, `s`, `t`.
    Lpi {
        /// First line point.
        p: [f64; 3],
        /// Second line point.
        q: [f64; 3],
        /// First plane point.
        r: [f64; 3],
        /// Second plane point.
        s: [f64; 3],
        /// Third plane point.
        t: [f64; 3],
    },
    /// Three-plane intersection, each plane through its three points.
    Tpi {
        /// The three defining planes, each as three points.
        planes: Box<[[[f64; 3]; 3]; 3]>,
    },
    /// Barycenter of three points (which may themselves be implicit).
    ///
    /// Stays implicit even for explicit children: the division by 3 would
    /// round, and the barycenter is used in predicates (e.g. as the interior
    /// representative of an arrangement sub-triangle during inside/outside
    /// classification), where exactness is required.
    Bary {
        /// The three points being averaged.
        pts: Box<[Point3; 3]>,
    },
    /// Linear combination on a segment: `a + t * (b - a)`.
    ///
    /// The CDT Steiner-point type (Diazzi et al. 2023, Sec. 4.2): a point
    /// constrained to lie EXACTLY on the segment through `a`, `b` for any
    /// f64 parameter `t` — rounding `t` only slides the point along the
    /// carrier line, never off it. Degree 1 in the inputs (w = 1), so every
    /// staged predicate stays cheap. Splits of sub-segments fold back onto
    /// the original carrier with a recomputed `t`, keeping the
    /// representation closed under recovery.
    Lnc {
        /// Segment start.
        a: [f64; 3],
        /// Segment end.
        b: [f64; 3],
        /// Position parameter, meaningful in (0, 1).
        t: f64,
    },
    /// Planar affine combination on a triangle: `a + u (b - a) + v (c - a)`.
    ///
    /// The 2D analog of [`Point3::Lnc`]: a point constrained to lie EXACTLY
    /// on the plane through `a`, `b`, `c` for any f64 parameters — rounding
    /// `u`, `v` only slides the point within the plane, never off it. The
    /// CDT facet-interior Steiner type (surface refinement points must stay
    /// exactly on their constraint facet or face recovery would see the
    /// facet pierced next round). Degree 1 in the inputs, w = 1.
    Pac {
        /// Triangle corner the parameters are anchored at.
        a: [f64; 3],
        /// Second corner (`u` direction).
        b: [f64; 3],
        /// Third corner (`v` direction).
        c: [f64; 3],
        /// Coordinate along `b - a`.
        u: f64,
        /// Coordinate along `c - a`.
        v: f64,
    },
}

impl Point3 {
    /// An explicit point.
    pub fn explicit(x: f64, y: f64, z: f64) -> Point3 {
        Point3::Explicit([x, y, z])
    }

    /// The intersection of the line through `p`, `q` with the plane through
    /// `r`, `s`, `t`.
    pub fn lpi(p: [f64; 3], q: [f64; 3], r: [f64; 3], s: [f64; 3], t: [f64; 3]) -> Point3 {
        Point3::Lpi { p, q, r, s, t }
    }

    /// The intersection of three planes, each given by three points.
    pub fn tpi(plane0: [[f64; 3]; 3], plane1: [[f64; 3]; 3], plane2: [[f64; 3]; 3]) -> Point3 {
        Point3::Tpi {
            planes: Box::new([plane0, plane1, plane2]),
        }
    }

    /// The intersection of two coplanar lines: line through `p`, `q` and line
    /// through `a`, `b`. Returns `None` if the four points are not exactly
    /// coplanar, or if the lines are parallel or identical.
    ///
    /// Construction: the point is the LPI of line (p, q) with a plane that
    /// contains line (a, b) but not the common plane — its third defining
    /// point `x` is synthesized off-plane. Any `x` works as long as the
    /// resulting LPI is valid: if `x` accidentally lands in the common plane
    /// or collinear with (a, b), the LPI's w is exactly zero and the next
    /// candidate is tried; exactness never depends on the accuracy of `x`.
    pub fn lli_coplanar(p: [f64; 3], q: [f64; 3], a: [f64; 3], b: [f64; 3]) -> Option<Point3> {
        // Require exact coplanarity, otherwise the construction below would
        // produce a point that is not on line (a, b).
        if crate::orient::orient3d(
            &Point3::Explicit(p),
            &Point3::Explicit(q),
            &Point3::Explicit(a),
            &Point3::Explicit(b),
        ) != Some(Sign::Zero)
        {
            return None;
        }
        // Preferred synthesis: offset along the (approximate) common-plane
        // normal; unit-axis offsets as fallbacks.
        let d1 = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        let d2 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let n = [
            d2[1] * d1[2] - d2[2] * d1[1],
            d2[2] * d1[0] - d2[0] * d1[2],
            d2[0] * d1[1] - d2[1] * d1[0],
        ];
        let candidates = [
            [a[0] + n[0], a[1] + n[1], a[2] + n[2]],
            [a[0] + 1.0, a[1], a[2]],
            [a[0], a[1] + 1.0, a[2]],
            [a[0], a[1], a[2] + 1.0],
        ];
        for x in candidates {
            let cand = Point3::Lpi {
                p,
                q,
                r: a,
                s: b,
                t: x,
            };
            if cand.is_valid() {
                return Some(cand);
            }
        }
        // All candidates invalid: the lines are parallel or identical.
        None
    }

    /// The exact barycenter of three points.
    pub fn bary(a: Point3, b: Point3, c: Point3) -> Point3 {
        Point3::Bary {
            pts: Box::new([a, b, c]),
        }
    }

    /// A point on the segment from `a` to `b` at parameter `t` (exact on the
    /// carrier line for ANY f64 `t`; meaningful as a Steiner point for
    /// `t` in (0, 1)).
    pub fn lnc(a: [f64; 3], b: [f64; 3], t: f64) -> Point3 {
        Point3::Lnc { a, b, t }
    }

    /// A point in the plane of the triangle (a, b, c) at barycentric-style
    /// parameters (u, v) (exact on the carrier plane for ANY f64 values;
    /// inside the triangle for u, v > 0, u + v < 1).
    pub fn pac(a: [f64; 3], b: [f64; 3], c: [f64; 3], u: f64, v: f64) -> Point3 {
        Point3::Pac { a, b, c, u, v }
    }

    /// The coordinates if this point is explicit.
    pub fn as_explicit(&self) -> Option<[f64; 3]> {
        match self {
            Point3::Explicit(c) => Some(*c),
            _ => None,
        }
    }

    /// Affine (degree-1, w = 1) decomposition into explicit parent points and
    /// barycentric weights, for the Steiner types [`Point3::Lnc`] and
    /// [`Point3::Pac`]: the point equals `sum_i weight_i * parent_i` exactly
    /// (the weights are the real numbers `1 - t`, `t`, etc.; the returned f64
    /// values are their roundings, used only for a strictly-positive guard).
    /// Returns the parents, weights, and the count `n` (2 for Lnc, 3 for Pac).
    /// `None` for explicit points and the projective types (Lpi/Tpi/Bary).
    ///
    /// Multilinear predicates (orient3d) can substitute the parents for the
    /// point: the predicate's value is the same weighted combination of the
    /// per-parent values, so when every parent shares an orientation sign the
    /// point shares it too -- resolved by fast explicit predicates instead of
    /// the implicit interval/expansion path.
    pub fn affine_combo(&self) -> Option<([[f64; 3]; 3], [f64; 3], usize)> {
        match self {
            Point3::Lnc { a, b, t } => Some(([*a, *b, [0.0; 3]], [1.0 - t, *t, 0.0], 2)),
            Point3::Pac { a, b, c, u, v } => Some(([*a, *b, *c], [1.0 - u - v, *u, *v], 3)),
            _ => None,
        }
    }

    /// Homogeneous coordinates (x, y, z, w) in the given ring. For explicit
    /// points w = 1.
    pub fn hom<T: Ring>(&self) -> [T; 4] {
        match self {
            Point3::Explicit(c) => [
                T::from_f64(c[0]),
                T::from_f64(c[1]),
                T::from_f64(c[2]),
                T::from_f64(1.0),
            ],
            Point3::Lpi { p, q, r, s, t } => lpi_hom(*p, *q, *r, *s, *t),
            Point3::Tpi { planes } => tpi_hom(planes),
            Point3::Bary { pts } => {
                // (p0 + p1 + p2) / 3 over homogeneous children:
                // X_i = x0 w1 w2 + x1 w0 w2 + x2 w0 w1, W = 3 w0 w1 w2.
                let h: [[T; 4]; 3] = std::array::from_fn(|k| pts[k].hom::<T>());
                let w01 = h[0][3].mul(&h[1][3]);
                let w12 = h[1][3].mul(&h[2][3]);
                let w02 = h[0][3].mul(&h[2][3]);
                let coord = |i: usize| {
                    h[0][i]
                        .mul(&w12)
                        .add(&h[1][i].mul(&w02))
                        .add(&h[2][i].mul(&w01))
                };
                let w = T::from_f64(3.0).mul(&w01).mul(&h[2][3]);
                [coord(0), coord(1), coord(2), w]
            }
            Point3::Lnc { a, b, t } => {
                // a + t (b - a), w = 1: degree 1 in the f64 inputs.
                let tt = T::from_f64(*t);
                let coord = |i: usize| {
                    let ai = T::from_f64(a[i]);
                    let bi = T::from_f64(b[i]);
                    ai.add(&tt.mul(&Ring::sub(&bi, &ai)))
                };
                [coord(0), coord(1), coord(2), T::from_f64(1.0)]
            }
            Point3::Pac { a, b, c, u, v } => {
                // a + u (b - a) + v (c - a), w = 1: degree 1 in the inputs.
                let uu = T::from_f64(*u);
                let vv = T::from_f64(*v);
                let coord = |i: usize| {
                    let ai = T::from_f64(a[i]);
                    let bi = T::from_f64(b[i]);
                    let ci = T::from_f64(c[i]);
                    ai.add(&uu.mul(&Ring::sub(&bi, &ai)))
                        .add(&vv.mul(&Ring::sub(&ci, &ai)))
                };
                [coord(0), coord(1), coord(2), T::from_f64(1.0)]
            }
        }
    }

    /// Homogeneous 2D coordinates in the projection that drops the given
    /// axis. The pairing is cyclic — drop X gives (y, z), drop Y gives (z, x),
    /// drop Z gives (x, y) — so the projected orientation of a triangle equals
    /// the sign of the dropped component of its normal.
    pub fn hom2<T: Ring>(&self, drop: Axis) -> [T; 3] {
        let [x, y, z, w] = self.hom::<T>();
        match drop {
            Axis::X => [y, z, w],
            Axis::Y => [z, x, w],
            Axis::Z => [x, y, w],
        }
    }

    /// Exact coincidence test: true if both points are the same point of R^3.
    ///
    /// Both points must be valid (w != 0); cross-ratio equality
    /// x_a * w_b == x_b * w_a (etc.) is then equivalent to equality of the
    /// affine points.
    pub fn coincides(&self, other: &Point3) -> bool {
        if let (Some(a), Some(b)) = (self.as_explicit(), other.as_explicit()) {
            return a == b;
        }
        // Interval filter: any strictly nonzero cross-difference rules out
        // coincidence; all exactly-zero proves it.
        let ha = self.hom::<Interval>();
        let hb = other.hom::<Interval>();
        let mut undecided = false;
        for i in 0..3 {
            let d = Ring::sub(&Ring::mul(&ha[i], &hb[3]), &Ring::mul(&hb[i], &ha[3]));
            match d.sign() {
                Some(Sign::Zero) => {}
                Some(_) => return false,
                None => undecided = true,
            }
        }
        if !undecided {
            return true;
        }
        // Exact stage.
        let ea = self.hom::<Expansion>();
        let eb = other.hom::<Expansion>();
        (0..3).all(|i| Ring::sub(&ea[i].mul(&eb[3]), &eb[i].mul(&ea[3])).is_zero())
    }

    /// Exact sign of the homogeneous w coordinate. [`Sign::Zero`] means the
    /// defining primitives do not intersect in a single point (the point is
    /// invalid and must not be used in predicates).
    pub fn w_sign(&self) -> Sign {
        match self {
            Point3::Explicit(_) | Point3::Lnc { .. } | Point3::Pac { .. } => Sign::Positive,
            _ => {
                // Interval filter first, exact fallback.
                if let Some(s) = self.hom::<Interval>()[3].sign() {
                    return s;
                }
                self.hom::<Expansion>()[3].sign()
            }
        }
    }

    /// True if the point is well defined (w != 0). Exact.
    pub fn is_valid(&self) -> bool {
        self.w_sign() != Sign::Zero
    }

    /// Approximate f64 coordinates (for output/visualization, never for
    /// predicates). `None` if the f64 evaluation of w underflows to zero.
    pub fn approx(&self) -> Option<[f64; 3]> {
        // Explicit points round-trip untouched.
        if let Point3::Explicit(p) = self {
            return Some(*p);
        }
        // Constructed points: correctly rounded coordinates from the exact
        // homogeneous representation. Interval-midpoint evaluation is off
        // by an ulp or two, which is enough to knock a point off a plane it
        // exactly lies on (and exact planarity is what patches, creases,
        // and region volumes are built from).
        let h = self.hom::<Expansion>();
        if h[3].sign() == Sign::Zero {
            return None;
        }
        Some([
            crate::expansion::div_round(&h[0], &h[3]),
            crate::expansion::div_round(&h[1], &h[3]),
            crate::expansion::div_round(&h[2], &h[3]),
        ])
    }
}
