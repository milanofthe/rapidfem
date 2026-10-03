//! Bridge from a [`rapidmesh_brep::Brep`] to the bottom-up mesher stages.
//!
//! The B-rep is the geometry source; this module turns its analytic edges and
//! trimmed faces into the inputs the existing stages consume -- stage 1 here
//! (an edge's analytic curve for `curve::distribute`), the face and volume
//! stages follow. The B-rep only changes WHERE the surface points come from; the
//! volume Lloyd, region classification and restricted-Delaunay extraction are
//! unchanged.

use crate::curve::{Curve, PolylineCurve};
use rapidmesh_brep::{Brep, Curve as BCurve, Edge as BEdge, Surface};
use rapidmesh_geom::nurbs::NurbsCurve;
use rapidmesh_geom::vec3::{add, cross, dot, scale, sub, V3};
use std::sync::Arc;

fn dist3(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d).sqrt()
}

struct ProfileCurve {
    profile: Arc<NurbsCurve>,
    base: V3,
    u: V3,
    v: V3,
    axis: V3,
    z: f64,
    ts: Vec<f64>,
    ss: Vec<f64>,
}

impl ProfileCurve {
    fn new(
        profile: Arc<NurbsCurve>,
        base: V3,
        u: V3,
        v: V3,
        axis: V3,
        t: [f64; 2],
        z: f64,
    ) -> Option<ProfileCurve> {
        let (lo, hi) = (t[0].min(t[1]), t[0].max(t[1]));
        if !(hi > lo) {
            return None;
        }
        let n = 256usize;
        let (mut ts, mut ss) = (vec![lo], vec![0.0f64]);
        let (mut prev, mut acc) = (lo, 0.0);
        for i in 1..=n {
            let tt = lo + (hi - lo) * i as f64 / n as f64;
            acc += profile.arc_length(prev, tt, 2);
            ts.push(tt);
            ss.push(acc);
            prev = tt;
        }
        Some(ProfileCurve {
            profile,
            base,
            u,
            v,
            axis,
            z,
            ts,
            ss,
        })
    }
    fn s_to_t(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, self.ss[self.ss.len() - 1]);
        let i = self
            .ss
            .partition_point(|&x| x < s)
            .clamp(1, self.ss.len() - 1);
        let (s0, s1) = (self.ss[i - 1], self.ss[i]);
        let f = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
        self.ts[i - 1] + f * (self.ts[i] - self.ts[i - 1])
    }
    fn at3(&self, t: f64) -> V3 {
        let c = self.profile.eval(t);
        add(
            add(self.base, scale(self.axis, self.z)),
            add(scale(self.u, c[0]), scale(self.v, c[1])),
        )
    }
}

impl Curve for ProfileCurve {
    fn length(&self) -> f64 {
        self.ss[self.ss.len() - 1]
    }
    fn point_at(&self, s: f64) -> V3 {
        self.at3(self.s_to_t(s))
    }
    fn radius_at(&self, s: f64) -> f64 {
        let k = self.profile.curvature(self.s_to_t(s));
        if k > 1e-12 {
            1.0 / k
        } else {
            f64::INFINITY
        }
    }
    fn ders_at(&self, s: f64) -> [V3; 3] {
        let t = self.s_to_t(s);
        let (_, c1, c2) = self.profile.ders2(t);
        let lift = |c: [f64; 2]| add(scale(self.u, c[0]), scale(self.v, c[1]));
        let [d1, d2] = arc_ders(lift(c1), lift(c2), 1.0);
        [self.at3(t), d1, d2]
    }
}

/// Arc-length derivatives from those in a parameter `t`: the unit tangent
/// (oriented by `sign`, the sign of dt/ds) and the curvature vector.
fn arc_ders(c_t: V3, c_tt: V3, sign: f64) -> [V3; 2] {
    let l2 = dot(c_t, c_t);
    if !(l2 > 0.0) {
        return [[0.0; 3], [0.0; 3]];
    }
    let t = scale(c_t, 1.0 / l2.sqrt());
    let k = scale(sub(c_tt, scale(t, dot(c_tt, t))), 1.0 / l2);
    [scale(t, sign), k]
}

/// A circular arc (or full circle) parametrised by arc length. Radius is constant,
/// so the sagitta sizing places uniform points; the arc range is taken from the
/// edge's chain endpoints (a closed rim spans the full `2*pi`).
struct CircleCurve {
    center: V3,
    x: V3,
    y: V3,
    radius: f64,
    a0: f64,
    span: f64,
}

impl CircleCurve {
    fn new(center: V3, axis: V3, x: V3, radius: f64, chain: &[V3]) -> Option<CircleCurve> {
        if !(radius > 0.0) || chain.len() < 2 {
            return None;
        }
        let y = cross(axis, x);
        let ang = |p: V3| {
            let d = sub(p, center);
            dot(d, y).atan2(dot(d, x))
        };
        // Total signed swept angle = sum of per-segment increments (each in
        // (-pi, pi]); robust for an arc (partial) and a closed rim (sums to +-2*pi).
        let pi = std::f64::consts::PI;
        let wrap = |a: f64| (a + pi).rem_euclid(2.0 * pi) - pi;
        let a0 = ang(chain[0]);
        let mut span = 0.0;
        for w in chain.windows(2) {
            span += wrap(ang(w[1]) - ang(w[0]));
        }
        if span.abs() < 1e-9 {
            return None;
        }
        Some(CircleCurve {
            center,
            x,
            y,
            radius,
            a0,
            span,
        })
    }
}

impl Curve for CircleCurve {
    fn length(&self) -> f64 {
        self.radius * self.span.abs()
    }
    fn point_at(&self, s: f64) -> V3 {
        let f = (s / self.length()).clamp(0.0, 1.0);
        let t = self.a0 + self.span * f;
        let (st, ct) = t.sin_cos();
        std::array::from_fn(|k| self.center[k] + self.radius * (ct * self.x[k] + st * self.y[k]))
    }
    fn radius_at(&self, _s: f64) -> f64 {
        self.radius
    }
    fn ders_at(&self, s: f64) -> [V3; 3] {
        let f = (s / self.length()).clamp(0.0, 1.0);
        let (st, ct) = (self.a0 + self.span * f).sin_cos();
        let radial: V3 = std::array::from_fn(|k| ct * self.x[k] + st * self.y[k]);
        let along: V3 = std::array::from_fn(|k| -st * self.x[k] + ct * self.y[k]);
        [
            add(self.center, scale(radial, self.radius)),
            scale(along, self.span.signum()),
            scale(radial, -1.0 / self.radius),
        ]
    }
}

/// An elliptic arc (oblique plane∩cylinder section) parametrised by arc length,
/// via a dense angle→arc-length table (like [`ProfileCurve`]). Curvature is the
/// exact analytic `R(t) = (a²sin²t + b²cos²t)^{3/2} / (a·b)`, so the sagitta
/// sizing refines the high-curvature ends of the major axis.
struct EllipseCurve {
    center: V3,
    major: V3,
    minor: V3,
    a: f64,
    b: f64,
    ts: Vec<f64>, // angles t0..t0+span
    ss: Vec<f64>, // cumulative arc length
}

impl EllipseCurve {
    fn new(center: V3, major: V3, minor: V3, a: f64, b: f64, chain: &[V3]) -> Option<EllipseCurve> {
        if !(a > 0.0 && b > 0.0) || chain.len() < 2 {
            return None;
        }
        // Angle of a chain point in the ellipse's own (normalised) frame.
        let ang = |p: V3| {
            let d = sub(p, center);
            (dot(d, minor) / b).atan2(dot(d, major) / a)
        };
        // Signed swept angle from the chain (each increment in (-pi, pi]), robust
        // for a partial arc and a closed section (sums to ±2π) -- as CircleCurve.
        let pi = std::f64::consts::PI;
        let wrap = |x: f64| (x + pi).rem_euclid(2.0 * pi) - pi;
        let t0 = ang(chain[0]);
        let mut span = 0.0;
        for w in chain.windows(2) {
            span += wrap(ang(w[1]) - ang(w[0]));
        }
        if span.abs() < 1e-9 {
            return None;
        }
        // Dense arc-length table over [t0, t0+span].
        let n = 512usize;
        let at = |t: f64| -> V3 {
            let (st, ct) = t.sin_cos();
            std::array::from_fn(|k| center[k] + a * ct * major[k] + b * st * minor[k])
        };
        let (mut ts, mut ss) = (vec![t0], vec![0.0f64]);
        let mut prev = at(t0);
        for i in 1..=n {
            let t = t0 + span * i as f64 / n as f64;
            let p = at(t);
            ts.push(t);
            ss.push(ss[i - 1] + dist3(prev, p));
            prev = p;
        }
        Some(EllipseCurve {
            center,
            major,
            minor,
            a,
            b,
            ts,
            ss,
        })
    }
    fn s_to_t(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, self.ss[self.ss.len() - 1]);
        let i = self
            .ss
            .partition_point(|&x| x < s)
            .clamp(1, self.ss.len() - 1);
        let (s0, s1) = (self.ss[i - 1], self.ss[i]);
        let f = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
        self.ts[i - 1] + f * (self.ts[i] - self.ts[i - 1])
    }
}

impl Curve for EllipseCurve {
    fn length(&self) -> f64 {
        self.ss[self.ss.len() - 1]
    }
    fn point_at(&self, s: f64) -> V3 {
        let t = self.s_to_t(s);
        let (st, ct) = t.sin_cos();
        std::array::from_fn(|k| {
            self.center[k] + self.a * ct * self.major[k] + self.b * st * self.minor[k]
        })
    }
    fn radius_at(&self, s: f64) -> f64 {
        let t = self.s_to_t(s);
        let (st, ct) = t.sin_cos();
        (self.a * self.a * st * st + self.b * self.b * ct * ct).powf(1.5) / (self.a * self.b)
    }
    fn ders_at(&self, s: f64) -> [V3; 3] {
        let t = self.s_to_t(s);
        let (st, ct) = t.sin_cos();
        let at = |ca: f64, cb: f64| -> V3 {
            std::array::from_fn(|k| self.a * ca * self.major[k] + self.b * cb * self.minor[k])
        };
        let sign = (self.ts[self.ts.len() - 1] - self.ts[0]).signum();
        let [d1, d2] = arc_ders(at(-st, ct), at(-ct, -st), sign);
        [add(self.center, at(ct, st)), d1, d2]
    }
}

/// Pulls `p` onto the intersection of two surfaces: a two-tangent-plane Newton
/// step (solve `p' = p + α·n_a + β·n_b` against both tangent-plane constraints;
/// quadratic convergence for transversal intersections -- plain alternating
/// projection converges only linearly, too slow near shallow crossings), falling
/// back to one alternating-projection step where the surfaces are near-tangential.
/// The caller guards against divergence.
fn pocs(sa: &Surface, sb: &Surface, mut p: V3, tol: f64) -> V3 {
    for _ in 0..32 {
        let (fa, na) = sa.closest(p);
        let (fb, nb) = sb.closest(p);
        let c = dot(na, nb);
        let det = 1.0 - c * c;
        let (ra, rb) = (dot(sub(fa, p), na), dot(sub(fb, p), nb));
        let q: V3 = if det > 1e-6 {
            let alpha = (ra - c * rb) / det;
            let beta = (rb - c * ra) / det;
            std::array::from_fn(|k| p[k] + alpha * na[k] + beta * nb[k])
        } else {
            sb.closest(fa).0
        };
        let moved = dist3(p, q);
        p = q;
        if moved < tol {
            break;
        }
    }
    p
}

/// The true intersection curve of two analytic carriers: a densely resampled
/// on-curve polyline for arc length + curvature, with `point_at` output pulled
/// onto BOTH carriers (the dense polyline's own chord sagitta, `h²/8R` between
/// samples, would otherwise leave distributed points measurably off-surface).
struct IntersectionCurve {
    poly: PolylineCurve,
    sa: Surface,
    sb: Surface,
    tol: f64,
}

impl Curve for IntersectionCurve {
    fn length(&self) -> f64 {
        self.poly.length()
    }
    fn point_at(&self, s: f64) -> V3 {
        // Endpoints stay EXACTLY the (pinned, shared) chain corners.
        if s <= 0.0 || s >= self.poly.length() {
            return self.poly.point_at(s);
        }
        let p0 = self.poly.point_at(s);
        let p = pocs(&self.sa, &self.sb, p0, self.tol);
        // Divergence guard, as in the polyline construction.
        if dist3(p, p0) <= 0.05 * self.poly.length() {
            p
        } else {
            p0
        }
    }
    fn radius_at(&self, s: f64) -> f64 {
        self.poly.radius_at(s)
    }
    fn ders_at(&self, s: f64) -> [V3; 3] {
        // The tangent of the true curve is square to both normals, oriented
        // along the polyline; the curvature term is left out (Gauss-Newton,
        // which converges for points near the curve).
        let p = self.point_at(s);
        let t = cross(self.sa.closest(p).1, self.sb.closest(p).1);
        let along = self.poly.ders_at(s)[1];
        let l = dot(t, t).sqrt();
        if !(l > 1e-9) {
            return self.poly.ders_at(s);
        }
        [p, scale(t, dot(t, along).signum() / l), [0.0; 3]]
    }
}

/// The dense on-curve polyline backing [`IntersectionCurve`]: the faceted chain
/// is subdivided (so the discrete curvature of [`PolylineCurve`] resolves the
/// real one) and every sample pulled onto BOTH surfaces by alternating
/// projection. A sample that diverges (a tangential contact, a projection into
/// another basin) keeps its chain position, so the result never degrades below
/// the input chain.
fn intersection_polyline(sa: &Surface, sb: &Surface, chain: &[V3]) -> Option<PolylineCurve> {
    if chain.len() < 2 {
        return None;
    }
    let total: f64 = chain.windows(2).map(|w| dist3(w[0], w[1])).sum();
    if !(total > 0.0) {
        return None;
    }
    let tol = 1e-12 * total;
    let target = total / 256.0; // dense enough for discrete curvature + sizing
    let mut out: Vec<V3> = Vec::with_capacity(512);
    for w in chain.windows(2) {
        let seg = dist3(w[0], w[1]);
        let n = (seg / target).ceil().max(1.0) as usize;
        for k in 0..n {
            let f = k as f64 / n as f64;
            let p0: V3 = std::array::from_fn(|c| w[0][c] + f * (w[1][c] - w[0][c]));
            let p = pocs(sa, sb, p0, tol);
            // Divergence guard: a projected point that left the segment's own
            // neighbourhood is a failed projection -- keep the chain point.
            out.push(if dist3(p, p0) <= seg.max(0.05 * total) {
                p
            } else {
                p0
            });
        }
    }
    // Endpoints: corners are shared pinned sites -- keep them EXACTLY as the
    // chain ends (the interior samples are the ones the POCS pulls onto the curve).
    out[0] = chain[0];
    out.push(chain[chain.len() - 1]);
    PolylineCurve::new(&out)
}

/// The analytic curve to distribute points on for a B-rep edge: the exact profile,
/// circle or ellipse where recovered, the POCS-densified intersection curve of two
/// analytic carriers, else the faceted chain polyline (a straight `Line` is
/// exactly a 2-point polyline, so it reduces to uniform spacing).
pub fn edge_curve(brep: &Brep, edge: &BEdge) -> Option<Box<dyn Curve>> {
    match &edge.curve {
        BCurve::Profile {
            profile,
            base,
            u,
            v,
            axis,
            t,
            z,
        } => ProfileCurve::new(profile.clone(), *base, *u, *v, *axis, *t, *z)
            .map(|c| Box::new(c) as Box<dyn Curve>),
        BCurve::Circle {
            center,
            axis,
            radius,
            x,
        } => CircleCurve::new(*center, *axis, *x, *radius, &edge.chain)
            .map(|c| Box::new(c) as Box<dyn Curve>),
        BCurve::Ellipse {
            center,
            major,
            minor,
            a,
            b,
        } => EllipseCurve::new(*center, *major, *minor, *a, *b, &edge.chain)
            .map(|c| Box::new(c) as Box<dyn Curve>)
            .or_else(|| PolylineCurve::new(&edge.chain).map(|c| Box::new(c) as Box<dyn Curve>)),
        BCurve::Intersection { a, b } => {
            let (sa, sb) = (brep.surface(*a), brep.surface(*b));
            match intersection_polyline(sa, sb, &edge.chain) {
                Some(poly) => {
                    let tol = 1e-12 * poly.length();
                    Some(Box::new(IntersectionCurve {
                        poly,
                        sa: sa.clone(),
                        sb: sb.clone(),
                        tol,
                    }) as Box<dyn Curve>)
                }
                None => PolylineCurve::new(&edge.chain).map(|c| Box::new(c) as Box<dyn Curve>),
            }
        }
        _ => PolylineCurve::new(&edge.chain).map(|c| Box::new(c) as Box<dyn Curve>),
    }
}

#[cfg(test)]
mod curve_tests {
    use super::*;
    use crate::curve::{closest_arc, distribute_floored};
    use rapidmesh_brep::build::from_plc;
    use rapidmesh_geom::{cylinder, solid_box, Scene};

    /// Distance of `p` from the analytic cylinder (axis line through `c`, dir `a`).
    fn cyl_dev(p: V3, c: V3, a: V3, r: f64) -> f64 {
        let al = (dot(a, a)).sqrt();
        let an: V3 = [a[0] / al, a[1] / al, a[2] / al];
        let d = sub(p, c);
        let z = dot(d, an);
        let rho = (dot(d, d) - z * z).max(0.0).sqrt();
        (rho - r).abs()
    }

    /// The oblique rim: distributed points must lie EXACTLY on the analytic
    /// cylinder AND the cut plane (the exact ellipse), not on the faceted chain
    /// (whose diagonal vertices sit a chord-sagitta ~4e-3 inside the barrel).
    #[test]
    fn ellipse_edge_points_lie_on_both_carriers() {
        let mut scene = Scene::new();
        scene.add_solid(cylinder([0.0, 0.0, -2.0], [1.0, 0.0, 2.0], 0.5, 24));
        scene.add_void(solid_box([-3.0, -3.0, 0.0], [3.0, 3.0, 3.0]));
        let b = from_plc(&scene.assemble());
        let e = b
            .edges
            .iter()
            .find(|e| matches!(e.curve, BCurve::Ellipse { .. }))
            .expect("an ellipse edge");
        let c = edge_curve(&b, e).unwrap();
        let s = distribute_floored(&*c, 1e-2, &|_| 0.2, 0.5, 0.0);
        assert!(s.len() > 4, "several points on the rim");
        for &si in &s {
            let p = c.point_at(si);
            assert!(
                cyl_dev(p, [0.0, 0.0, -2.0], [1.0, 0.0, 2.0], 0.5) < 1e-9,
                "point off the cylinder by {}",
                cyl_dev(p, [0.0, 0.0, -2.0], [1.0, 0.0, 2.0], 0.5)
            );
            assert!(p[2].abs() < 1e-9, "point off the cut plane by {}", p[2]);
        }
    }

    /// The cyl∩cyl hole rim: POCS-refined points must lie on BOTH cylinders
    /// (the faceted chain deviates by the facet sagitta, ~1e-3 at 24 segments).
    #[test]
    fn intersection_edge_points_lie_on_both_cylinders() {
        let mut scene = Scene::new();
        scene.add_solid(cylinder([-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.8, 24));
        scene.add_void(cylinder([0.0, -2.0, 0.0], [0.0, 4.0, 0.0], 0.4, 24));
        let b = from_plc(&scene.assemble());
        let e = b
            .edges
            .iter()
            .find(|e| matches!(e.curve, BCurve::Intersection { .. }))
            .expect("an intersection edge");
        let c = edge_curve(&b, e).unwrap();
        let s = distribute_floored(&*c, 1e-2, &|_| 0.2, 0.5, 0.0);
        assert!(s.len() > 6, "several points on the rim");
        // Interior points (endpoints stay pinned to the chain corners).
        for &si in &s[1..s.len() - 1] {
            let p = c.point_at(si);
            let d1 = cyl_dev(p, [-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.8);
            let d2 = cyl_dev(p, [0.0, -2.0, 0.0], [0.0, 4.0, 0.0], 0.4);
            assert!(d1 < 1e-6 && d2 < 1e-6, "point off carriers: {d1} / {d2}");
        }
    }

    /// The default numeric derivatives of `point_at`, for comparison.
    struct Numeric<'a>(&'a dyn Curve);
    impl Curve for Numeric<'_> {
        fn length(&self) -> f64 {
            self.0.length()
        }
        fn point_at(&self, s: f64) -> V3 {
            self.0.point_at(s)
        }
        fn radius_at(&self, s: f64) -> f64 {
            self.0.radius_at(s)
        }
    }

    fn samples(c: &dyn Curve, n: usize) -> Vec<(f64, V3)> {
        (0..=n)
            .map(|i| {
                let s = c.length() * i as f64 / n as f64;
                (s, c.point_at(s))
            })
            .collect()
    }

    fn scan(c: &dyn Curve, p: V3) -> f64 {
        (0..=200_000)
            .map(|i| dist3(c.point_at(c.length() * i as f64 / 200_000.0), p))
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn closed_form_curve_derivatives_match_numeric_ones() {
        let (x, y, z) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let chain: Vec<V3> = (0..=12)
            .map(|i| {
                let t = -0.4 + 2.2 * i as f64 / 12.0;
                [3.0 * t.cos(), 1.2 * t.sin(), 0.5]
            })
            .collect();
        let ellipse = EllipseCurve::new([0.0, 0.0, 0.5], x, y, 3.0, 1.2, &chain).unwrap();
        let circle = CircleCurve::new([0.0, 0.0, 0.0], z, x, 2.0, &chain).unwrap();
        for c in [&ellipse as &dyn Curve, &circle] {
            for f in [0.2, 0.5, 0.8] {
                let s = f * c.length();
                let [p, t, k] = c.ders_at(s);
                let numeric = Numeric(c).ders_at(s)[1];
                assert!(dist3(p, c.point_at(s)) < 1e-12);
                assert!((dot(t, t) - 1.0).abs() < 1e-12, "unit tangent");
                let cos = dot(t, numeric) / dot(numeric, numeric).sqrt();
                assert!(cos > 1.0 - 1e-9, "tangent along the curve at {f}: {cos}");
                assert!(dot(t, k).abs() < 1e-9, "curvature square to the tangent");
                let bend = dot(k, k).sqrt() * c.radius_at(s);
                assert!(
                    (bend - 1.0).abs() < 1e-9,
                    "curvature 1 / radius at {f}: {bend}"
                );
            }
        }
        // Exact arc length on the circle: the numeric derivatives agree too.
        let s = 0.3 * circle.length();
        let (a, b) = (circle.ders_at(s), Numeric(&circle).ders_at(s));
        assert!(
            dist3(a[1], b[1]) < 1e-6 && dist3(a[2], b[2]) < 1e-4,
            "{a:?} vs {b:?}"
        );
    }

    #[test]
    fn newton_finds_the_nearest_arc_length() {
        let (x, y) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let chain: Vec<V3> = (0..=12)
            .map(|i| {
                let t = -0.4 + 2.2 * i as f64 / 12.0;
                [3.0 * t.cos(), 1.2 * t.sin(), 0.5]
            })
            .collect();
        let ellipse = EllipseCurve::new([0.0, 0.0, 0.5], x, y, 3.0, 1.2, &chain).unwrap();
        let smp = samples(&ellipse, 16);
        for p in [[2.5, 1.5, 0.9], [0.3, 0.2, 0.5], [-1.0, 2.0, 0.0]] {
            let d = dist3(ellipse.point_at(closest_arc(&ellipse, &smp, p)), p);
            let reference = scan(&ellipse, p);
            assert!(
                d <= reference + 1e-9 && reference - d < 1e-6,
                "{p:?}: {d} vs {reference}"
            );
        }
        // A closed circle, a point just across the seam from the first sample.
        let ring: Vec<V3> = (0..=24)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 24.0;
                [2.0 * t.cos(), 2.0 * t.sin(), 0.0]
            })
            .collect();
        let circle = CircleCurve::new([0.0; 3], [0.0, 0.0, 1.0], x, 2.0, &ring).unwrap();
        let smp = samples(&circle, 8);
        let p = [3.0, -0.05, 0.4];
        let q = circle.point_at(closest_arc(&circle, &smp, p));
        let want = [
            2.0 * 3.0 / 3.0f64.hypot(0.05),
            -2.0 * 0.05 / 3.0f64.hypot(0.05),
            0.0,
        ];
        assert!(dist3(q, want) < 1e-9, "{q:?} vs {want:?}");
    }
}
