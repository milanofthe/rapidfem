//! The analytic curve of a B-rep edge ([`edge_curve`]): what the edge
//! samples are distributed along and the snap projects onto.

use crate::curve::{Curve, PolylineCurve};
use rapidmesh_brep::{Brep, Curve as BCurve, Edge as BEdge};
use rapidmesh_exact::vector::{cross, dist, dot, scale, sub, V3};
use rapidmesh_geom::Surface;

/// Arc length along a parametric curve and back: `ts` the parameters
/// from the curve's first end to its last (rising or falling), `ss` the arc
/// length at each.
struct ArcTable {
    ts: Vec<f64>,
    ss: Vec<f64>,
}

impl ArcTable {
    /// The table over `t0..t1` at `samples` pieces, the length of each by
    /// `arc(a, b)` with `a < b`.
    fn new(t0: f64, t1: f64, samples: usize, arc: impl Fn(f64, f64) -> f64) -> Option<ArcTable> {
        if !(t0 != t1 && t0.is_finite() && t1.is_finite()) {
            return None;
        }
        let (mut ts, mut ss) = (vec![t0], vec![0.0f64]);
        let (mut prev, mut acc) = (t0, 0.0);
        for i in 1..=samples {
            let t = t0 + (t1 - t0) * i as f64 / samples as f64;
            acc += arc(prev.min(t), prev.max(t));
            ts.push(t);
            ss.push(acc);
            prev = t;
        }
        (acc > 0.0).then_some(ArcTable { ts, ss })
    }

    fn length(&self) -> f64 {
        self.ss[self.ss.len() - 1]
    }

    /// The sign of dt/ds.
    fn sign(&self) -> f64 {
        (self.ts[self.ts.len() - 1] - self.ts[0]).signum()
    }

    fn t(&self, s: f64) -> f64 {
        let s = s.clamp(0.0, self.length());
        let i = self
            .ss
            .partition_point(|&x| x < s)
            .clamp(1, self.ss.len() - 1);
        let (s0, s1) = (self.ss[i - 1], self.ss[i]);
        let f = if s1 > s0 { (s - s0) / (s1 - s0) } else { 0.0 };
        self.ts[i - 1] + f * (self.ts[i] - self.ts[i - 1])
    }
}

/// A piece of a carrier curve by arc length from the edge's first end.
struct Piece {
    curve: rapidmesh_geom::Curve<3>,
    arc: ArcTable,
}

impl Piece {
    fn new(curve: rapidmesh_geom::Curve<3>, t: [f64; 2]) -> Option<Piece> {
        let arc = ArcTable::new(t[0], t[1], 256, |a, b| curve.arc_length(a, b))?;
        Some(Piece { curve, arc })
    }
}

impl Curve for Piece {
    fn length(&self) -> f64 {
        self.arc.length()
    }
    fn point_at(&self, s: f64) -> V3 {
        self.curve.eval(self.arc.t(s))
    }
    fn radius_at(&self, s: f64) -> f64 {
        let k = self.curve.curvature(self.arc.t(s));
        if k > 1e-12 {
            1.0 / k
        } else {
            f64::INFINITY
        }
    }
    fn ders_at(&self, s: f64) -> [V3; 3] {
        let (c0, c1, c2) = self.curve.ders(self.arc.t(s));
        let [d1, d2] = arc_ders(c1, c2, self.arc.sign());
        [c0, d1, d2]
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

/// The true intersection curve of two analytic carriers: a densely resampled
/// on-curve polyline for arc length + curvature, with `point_at` output pulled
/// onto BOTH carriers (the dense polyline's own chord sagitta, `h^2/8R` between
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
        let p = self.sa.meet(&self.sb, p0, self.tol);
        // Divergence guard, as in the polyline construction.
        if dist(p, p0) <= 0.05 * self.poly.length() {
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
    let total: f64 = chain.windows(2).map(|w| dist(w[0], w[1])).sum();
    if !(total > 0.0) {
        return None;
    }
    let tol = 1e-12 * total;
    let target = total / 256.0; // dense enough for discrete curvature + sizing
    let mut out: Vec<V3> = Vec::with_capacity(512);
    for w in chain.windows(2) {
        let seg = dist(w[0], w[1]);
        let n = (seg / target).ceil().max(1.0) as usize;
        for k in 0..n {
            let f = k as f64 / n as f64;
            let p0: V3 = std::array::from_fn(|c| w[0][c] + f * (w[1][c] - w[0][c]));
            let p = sa.meet(sb, p0, tol);
            // Divergence guard: a projected point that left the segment's own
            // neighbourhood is a failed projection -- keep the chain point.
            out.push(if dist(p, p0) <= seg.max(0.05 * total) {
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

/// The analytic curve to distribute points on for a B-rep edge: the piece
/// of its carrier, the intersection of two carriers pulled onto both, else
/// the chain.
pub fn edge_curve(brep: &Brep, edge: &BEdge) -> Option<Box<dyn Curve>> {
    let chain = || PolylineCurve::new(&edge.chain).map(|c| Box::new(c) as Box<dyn Curve>);
    match &edge.curve {
        BCurve::Piece { curve, t } => Piece::new(curve.clone(), *t)
            .map(|c| Box::new(c) as Box<dyn Curve>)
            .or_else(chain),
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
                None => chain(),
            }
        }
        BCurve::Polyline => chain(),
    }
}

#[cfg(test)]
mod curve_tests {
    use super::*;
    use crate::curve::{closest_arc, distribute_floored};
    use crate::sizing::CurvatureLaw;
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
            .find(|e| e.curve.carrier().is_some_and(|c| c.name() == "ellipse"))
            .expect("an ellipse edge");
        let c = edge_curve(&b, e).unwrap();
        let s = distribute_floored(
            &*c,
            &|r| CurvatureLaw::Chord(1e-2).curve(r),
            &|_| 0.2,
            0.5,
            0.0,
        );
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

    /// The cyl-cyl hole rim: POCS-refined points must lie on BOTH cylinders
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
        let s = distribute_floored(
            &*c,
            &|r| CurvatureLaw::Chord(1e-2).curve(r),
            &|_| 0.2,
            0.5,
            0.0,
        );
        assert!(s.len() > 6, "several points on the rim");
        // Interior points (endpoints stay pinned to the chain corners).
        for &si in &s[1..s.len() - 1] {
            let p = c.point_at(si);
            let d1 = cyl_dev(p, [-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.8);
            let d2 = cyl_dev(p, [0.0, -2.0, 0.0], [0.0, 4.0, 0.0], 0.4);
            assert!(d1 < 1e-6 && d2 < 1e-6, "point off carriers: {d1} / {d2}");
        }
    }

    /// The piece of `curve` along `chain`.
    fn piece(curve: rapidmesh_geom::Curve<3>, chain: &[V3]) -> Piece {
        let t = curve.piece(chain, 1e-9).unwrap();
        Piece::new(curve, t).unwrap()
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
            .map(|i| dist(c.point_at(c.length() * i as f64 / 200_000.0), p))
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
        let ellipse = piece(
            rapidmesh_geom::Curve::Ellipse {
                c: [0.0, 0.0, 0.5],
                p: scale(x, 3.0),
                q: scale(y, 1.2),
            },
            &chain,
        );
        let circle = piece(
            rapidmesh_geom::Curve::circle([0.0; 3], z, x, 2.0).unwrap(),
            &chain,
        );
        for c in [&ellipse as &dyn Curve, &circle] {
            for f in [0.2, 0.5, 0.8] {
                let s = f * c.length();
                let [p, t, k] = c.ders_at(s);
                let numeric = Numeric(c).ders_at(s)[1];
                assert!(dist(p, c.point_at(s)) < 1e-12);
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
            dist(a[1], b[1]) < 1e-6 && dist(a[2], b[2]) < 1e-4,
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
        let ellipse = piece(
            rapidmesh_geom::Curve::Ellipse {
                c: [0.0, 0.0, 0.5],
                p: scale(x, 3.0),
                q: scale(y, 1.2),
            },
            &chain,
        );
        let smp = samples(&ellipse, 16);
        for p in [[2.5, 1.5, 0.9], [0.3, 0.2, 0.5], [-1.0, 2.0, 0.0]] {
            let d = dist(ellipse.point_at(closest_arc(&ellipse, &smp, p)), p);
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
        let circle = piece(
            rapidmesh_geom::Curve::circle([0.0; 3], [0.0, 0.0, 1.0], x, 2.0).unwrap(),
            &ring,
        );
        let smp = samples(&circle, 8);
        let p = [3.0, -0.05, 0.4];
        let q = circle.point_at(closest_arc(&circle, &smp, p));
        let want = [
            2.0 * 3.0 / 3.0f64.hypot(0.05),
            -2.0 * 0.05 / 3.0f64.hypot(0.05),
            0.0,
        ];
        assert!(dist(q, want) < 1e-9, "{q:?} vs {want:?}");
    }
}
