//! The carrier curves: lines, conics and B-splines in the plane (the
//! pcurves of a CAD file, the profiles of swept shapes) and in space (the
//! edges), with one interface: evaluation and derivatives, curvature, the
//! parameter of a point, arc length, the piece of a curve a chain of points
//! runs along, and affine maps.
//!
//! A conic is kept by its centre and two vectors, its parameter as a CAD
//! file gives it: an ellipse `c + cos t p + sin t q`, a hyperbola `c +
//! cosh t p + sinh t q`, a parabola `c + t^2 p + t q`; a line `p + t d`.
//! The vectors need be neither square nor of one length (conjugate
//! semi-diameters), so an affine map carries a conic by its centre and
//! vectors alone; an ellipse whose vectors are square and of one length is
//! a circle.

use crate::nurbs::NurbsCurve;
use rapidmesh_exact::vector::{
    add, along, cross, dist, dot, len, normalize, scale, sub, wrap_pm, Affine, Frame, V3,
};
use std::f64::consts::{PI, TAU};
use std::sync::Arc;

/// A carrier curve in `N` dimensions.
#[derive(Debug, Clone)]
pub enum Curve<const N: usize> {
    /// `p + t d`.
    Line {
        p: [f64; N],
        d: [f64; N],
    },
    /// `c + cos t p + sin t q`: a circle where `p` and `q` are square and
    /// of one length.
    Ellipse {
        c: [f64; N],
        p: [f64; N],
        q: [f64; N],
    },
    /// `c + cosh t p + sinh t q`.
    Hyperbola {
        c: [f64; N],
        p: [f64; N],
        q: [f64; N],
    },
    /// `c + t^2 p + t q`.
    Parabola {
        c: [f64; N],
        p: [f64; N],
        q: [f64; N],
    },
    Nurbs(Arc<NurbsCurve<N>>),
}

/// How near square and of one length an ellipse's vectors are to be a
/// circle (relative to their length).
const ROUND: f64 = 1e-12;

impl<const N: usize> Curve<N> {
    /// The centre and radius of an ellipse that is a circle.
    pub fn as_circle(&self) -> Option<([f64; N], f64)> {
        let Curve::Ellipse { c, p, q } = self else {
            return None;
        };
        let (a, b) = (len(*p), len(*q));
        ((a - b).abs() <= ROUND * a && dot(*p, *q).abs() <= ROUND * a * b).then_some((*c, a))
    }

    /// The name of the kind: "line", "circle", "ellipse", "hyperbola",
    /// "parabola" or "spline".
    pub fn name(&self) -> &'static str {
        match self {
            Curve::Line { .. } => "line",
            Curve::Ellipse { .. } if self.as_circle().is_some() => "circle",
            Curve::Ellipse { .. } => "ellipse",
            Curve::Hyperbola { .. } => "hyperbola",
            Curve::Parabola { .. } => "parabola",
            Curve::Nurbs(_) => "spline",
        }
    }

    /// The point at `t`.
    pub fn eval(&self, t: f64) -> [f64; N] {
        self.ders(t).0
    }

    /// The point at `t` and its first two derivatives in `t`.
    pub fn ders(&self, t: f64) -> ([f64; N], [f64; N], [f64; N]) {
        // `c + f p + g q` with `f`, `g` and their derivatives.
        let mix = |c: [f64; N], p: [f64; N], q: [f64; N], f: [f64; 3], g: [f64; 3]| {
            let at = |u: f64, v: f64| add(scale(p, u), scale(q, v));
            (add(c, at(f[0], g[0])), at(f[1], g[1]), at(f[2], g[2]))
        };
        match self {
            Curve::Line { p, d } => (along(*p, *d, t), *d, [0.0; N]),
            Curve::Ellipse { c, p, q } => {
                let (s, co) = t.sin_cos();
                mix(*c, *p, *q, [co, -s, -co], [s, co, -s])
            }
            Curve::Hyperbola { c, p, q } => {
                let (sh, ch) = (t.sinh(), t.cosh());
                mix(*c, *p, *q, [ch, sh, ch], [sh, ch, sh])
            }
            Curve::Parabola { c, p, q } => mix(*c, *p, *q, [t * t, 2.0 * t, 2.0], [t, 1.0, 0.0]),
            Curve::Nurbs(n) => n.ders2(t),
        }
    }

    /// The curvature at `t` (0 on a line).
    pub fn curvature(&self, t: f64) -> f64 {
        if let Some((_, r)) = self.as_circle() {
            return 1.0 / r;
        }
        let (_, d1, d2) = self.ders(t);
        let (l1, l2, m) = (dot(d1, d1), dot(d2, d2), dot(d1, d2));
        if l1 > 0.0 {
            (l1 * l2 - m * m).max(0.0).sqrt() / (l1.sqrt() * l1)
        } else {
            0.0
        }
    }

    /// The period of a closed carrier: a full turn of an ellipse, the
    /// domain of a B-spline whose ends meet; none for the others.
    pub fn period(&self) -> Option<f64> {
        match self {
            Curve::Ellipse { .. } => Some(TAU),
            Curve::Nurbs(n) => {
                let (lo, hi) = n.domain();
                let size = n
                    .ctrl
                    .iter()
                    .map(|c| dist(*c, n.ctrl[0]))
                    .fold(0.0, f64::max);
                (dist(n.eval(lo), n.eval(hi)) <= 1e-9 * size.max(1e-300)).then_some(hi - lo)
            }
            _ => None,
        }
    }

    /// The parameter of the point of the curve nearest `x`.
    pub fn param(&self, x: [f64; N]) -> f64 {
        match self {
            Curve::Line { p, d } => dot(sub(x, *p), *d) / dot(*d, *d),
            Curve::Nurbs(n) => n.closest_param(x),
            Curve::Ellipse { c, p, q } => {
                let [a, b] = coordinates(sub(x, *c), *p, *q);
                let t = b.atan2(a);
                if self.as_circle().is_some() {
                    t
                } else {
                    self.newton(x, t)
                }
            }
            // From where `x` lies in the frame of the vectors, refined.
            Curve::Hyperbola { c, p, q } => {
                let [_, b] = coordinates(sub(x, *c), *p, *q);
                self.newton(x, b.asinh())
            }
            Curve::Parabola { c, p, q } => {
                let [_, b] = coordinates(sub(x, *c), *p, *q);
                self.newton(x, b)
            }
        }
    }

    /// [`Curve::param`] searched from `t0`, near the answer.
    pub fn param_near(&self, x: [f64; N], t0: f64) -> f64 {
        match self {
            Curve::Nurbs(n) => n.closest_param_near(x, t0),
            Curve::Line { .. } => self.param(x),
            _ if self.as_circle().is_some() => self.param(x),
            _ => self.newton(x, t0),
        }
    }

    /// Newton on `(C(t) - x) . C'(t) = 0` from `t`, each step halved until
    /// the distance drops.
    fn newton(&self, x: [f64; N], mut t: f64) -> f64 {
        let d2 = |t: f64| dist(self.eval(t), x).powi(2);
        for _ in 0..32 {
            let (c, d1, dd) = self.ders(t);
            let w = sub(c, x);
            let fp = dot(d1, d1) + dot(w, dd);
            if !(fp > 0.0) {
                break;
            }
            let mut step = -dot(w, d1) / fp;
            let now = d2(t);
            while step.abs() > 1e-15 && d2(t + step) > now {
                step *= 0.5;
            }
            t += step;
            if step.abs() <= 1e-14 {
                break;
            }
        }
        t
    }

    /// The length of the curve from `t0` to `t1` (either way).
    pub fn arc_length(&self, t0: f64, t1: f64) -> f64 {
        let (lo, hi) = (t0.min(t1), t0.max(t1));
        if let Some((_, r)) = self.as_circle() {
            return r * (hi - lo);
        }
        match self {
            Curve::Line { d, .. } => len(*d) * (hi - lo),
            Curve::Nurbs(n) => n.arc_length(lo, hi, 2),
            _ => {
                // Gauss-Legendre on |C'|, a piece per sixteenth of a turn
                // of the parameter.
                let pieces = ((hi - lo) / (TAU / 16.0)).ceil().max(1.0) as usize;
                let h = (hi - lo) / pieces as f64;
                const X: [f64; 5] = [
                    -0.906_179_845_938_664,
                    -0.538_469_310_105_683,
                    0.0,
                    0.538_469_310_105_683,
                    0.906_179_845_938_664,
                ];
                const W: [f64; 5] = [
                    0.236_926_885_056_189,
                    0.478_628_670_499_366,
                    0.568_888_888_888_889,
                    0.478_628_670_499_366,
                    0.236_926_885_056_189,
                ];
                (0..pieces)
                    .map(|k| {
                        let m = lo + (k as f64 + 0.5) * h;
                        X.iter()
                            .zip(W)
                            .map(|(x, w)| w * len(self.ders(m + 0.5 * h * x).1))
                            .sum::<f64>()
                            * 0.5
                            * h
                    })
                    .sum()
            }
        }
    }

    /// The piece of the curve the chain `pts` runs along, as the parameters
    /// of its first and its last point (the first greater where the chain
    /// runs against the parameter); none where the chain does not follow
    /// the curve monotonically (a step of half a turn on an ellipse, which
    /// goes round either way; a B-spline whose parameters double back). A
    /// chain that closes on itself spans a whole turn, or the whole of a
    /// B-spline.
    pub fn piece(&self, pts: &[[f64; N]], tol: f64) -> Option<[f64; 2]> {
        let n = pts.len().checked_sub(1).filter(|&n| n > 0)?;
        match self {
            Curve::Ellipse { .. } => {
                let t0 = self.param(pts[0]);
                let mut span = 0.0;
                let mut last = t0;
                for &x in &pts[1..] {
                    let t = self.param_near(x, last);
                    let step = wrap_pm(t - last);
                    if step.abs() > PI * (1.0 - 1e-6) {
                        return None;
                    }
                    span += step;
                    last = t;
                }
                (span.abs() > 1e-9).then_some([t0, t0 + span])
            }
            Curve::Nurbs(c) if dist(pts[0], pts[n]) <= tol => {
                // The whole curve, the way the chain runs.
                let (lo, hi) = c.domain();
                let t = if self.param(pts[1]) > self.param(pts[0]) {
                    [lo, hi]
                } else {
                    [hi, lo]
                };
                self.monotonic(pts, t)
            }
            _ => self.monotonic(pts, [self.param(pts[0]), self.param(pts[n])]),
        }
    }

    /// `t` where the inner points of `pts` run along it in order.
    fn monotonic(&self, pts: &[[f64; N]], t: [f64; 2]) -> Option<[f64; 2]> {
        if t[0] == t[1] {
            return None;
        }
        let mut last = 0.0;
        for &x in &pts[1..pts.len() - 1] {
            let s = (self.param(x) - t[0]) / (t[1] - t[0]);
            if !(s > last && s < 1.0) {
                return None;
            }
            last = s;
        }
        Some(t)
    }

    /// The curve under the affine map given by `point` and its linear part
    /// `vector` (a B-spline's control points move; a conic's centre and
    /// vectors).
    fn map<const M: usize>(
        &self,
        point: impl Fn([f64; N]) -> [f64; M],
        vector: impl Fn([f64; N]) -> [f64; M],
    ) -> Curve<M> {
        match self {
            Curve::Line { p, d } => Curve::Line {
                p: point(*p),
                d: vector(*d),
            },
            Curve::Ellipse { c, p, q } => Curve::Ellipse {
                c: point(*c),
                p: vector(*p),
                q: vector(*q),
            },
            Curve::Hyperbola { c, p, q } => Curve::Hyperbola {
                c: point(*c),
                p: vector(*p),
                q: vector(*q),
            },
            Curve::Parabola { c, p, q } => Curve::Parabola {
                c: point(*c),
                p: vector(*p),
                q: vector(*q),
            },
            Curve::Nurbs(n) => Curve::Nurbs(Arc::new(NurbsCurve {
                ctrl: n.ctrl.iter().map(|&x| point(x)).collect(),
                degree: n.degree,
                knots: n.knots.clone(),
                weights: n.weights.clone(),
            })),
        }
    }
}

/// The coordinates `[a, b]` of `w` in the plane of `p` and `q` (`w = a p +
/// b q` for a `w` in it, of its foot otherwise).
fn coordinates<const N: usize>(w: [f64; N], p: [f64; N], q: [f64; N]) -> [f64; 2] {
    let (pp, pq, qq) = (dot(p, p), dot(p, q), dot(q, q));
    let (wp, wq) = (dot(w, p), dot(w, q));
    let det = pp * qq - pq * pq;
    if !(det > 0.0) {
        return [0.0, 0.0];
    }
    [(wp * qq - wq * pq) / det, (wq * pp - wp * pq) / det]
}

impl Curve<2> {
    /// The curve under `(u, v) -> (s[0] u + t[0], s[1] v + t[1])`: a
    /// parameter curve carried into another parameterisation of its
    /// surface.
    pub fn rescaled(&self, s: [f64; 2], t: [f64; 2]) -> Curve<2> {
        self.map(
            |x| [s[0] * x[0] + t[0], s[1] * x[1] + t[1]],
            |d| [s[0] * d[0], s[1] * d[1]],
        )
    }

    /// The plane curve set into space in the plane `z` of `frame`: a point
    /// `(u, v)` goes to `frame.at(u, v, z)` (an affine map, so every kind
    /// stays exact).
    pub fn lifted(&self, frame: &Frame, z: f64) -> Curve<3> {
        self.map(
            |x| frame.at(x[0], x[1], z),
            |d| add(scale(frame.x, d[0]), scale(frame.y, d[1])),
        )
    }
}

impl Curve<3> {
    /// The circle about `center` of `radius` square to `axis`, its angle 0
    /// towards `x` (its part square to the axis).
    pub fn circle(center: V3, axis: V3, x: V3, radius: f64) -> Option<Curve<3>> {
        let f = Frame::new(center, axis, Some(x))?;
        Some(Curve::Ellipse {
            c: center,
            p: scale(f.x, radius),
            q: scale(f.y, radius),
        })
    }

    /// The unit normal of the plane of a conic, the conic running
    /// counterclockwise about it.
    pub fn axis(&self) -> Option<V3> {
        match self {
            Curve::Ellipse { p, q, .. }
            | Curve::Hyperbola { p, q, .. }
            | Curve::Parabola { p, q, .. } => Some(normalize(cross(*p, *q))),
            _ => None,
        }
    }

    /// The curve carried by the affine map `m`: every kind under any map
    /// (an ellipse whose vectors an unequal stretch makes unequal is no
    /// circle any more).
    pub fn mapped(&self, m: &Affine) -> Curve<3> {
        self.map(|x| m.point(x), |d| m.vector(d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close<const N: usize>(a: [f64; N], b: [f64; N]) -> bool {
        dist(a, b) < 1e-9
    }

    #[test]
    fn conics_invert_and_measure() {
        let e: Curve<3> = Curve::Ellipse {
            c: [1.0, 2.0, 3.0],
            p: [3.0, 0.0, 0.0],
            q: [0.0, 0.0, 1.0],
        };
        assert_eq!(e.name(), "ellipse");
        for t in [-2.5, 0.3, 1.2, 3.0] {
            let x = e.eval(t);
            assert!(wrap_pm(e.param(x) - t).abs() < 1e-9, "{t}");
            // Off the curve, along its normal: the same parameter.
            let (_, d1, d2) = e.ders(t);
            let n = normalize(sub(d2, scale(d1, dot(d1, d2) / dot(d1, d1))));
            assert!(wrap_pm(e.param(sub(x, scale(n, 0.2))) - t).abs() < 1e-8);
        }
        // Ramanujan's perimeter of a 3 x 1 ellipse.
        let h: f64 = (2.0f64 / 4.0).powi(2);
        let perimeter = PI * 4.0 * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt()));
        assert!((e.arc_length(0.0, TAU) - perimeter).abs() < 1e-6);
        assert!(
            (e.curvature(0.0) - 3.0).abs() < 1e-12,
            "a / b^2 at the end of the major axis"
        );
        let c = Curve::circle([0.0; 3], [0.0, 0.0, 2.0], [1.0, 1.0, 0.0], 2.0).unwrap();
        assert_eq!(c.name(), "circle");
        assert!((c.curvature(0.4) - 0.5).abs() < 1e-15);
        assert!(close(c.eval(0.0), scale(normalize([1.0, 1.0, 0.0]), 2.0)));
        // A hyperbola and a parabola by their file parameters.
        let hyp: Curve<2> = Curve::Hyperbola {
            c: [0.0, 0.0],
            p: [2.0, 0.0],
            q: [0.0, 1.0],
        };
        let par: Curve<2> = Curve::Parabola {
            c: [0.0, 0.0],
            p: [0.5, 0.0],
            q: [0.0, 1.0],
        };
        for t in [-1.0, 0.2, 0.9] {
            assert!((hyp.param(hyp.eval(t)) - t).abs() < 1e-9);
            assert!((par.param(par.eval(t)) - t).abs() < 1e-9);
        }
        // A line by the length of its direction, as a file gives it.
        let line: Curve<3> = Curve::Line {
            p: [1.0, 0.0, 0.0],
            d: [0.0, 2.0, 0.0],
        };
        assert!((line.param([5.0, 3.0, 1.0]) - 1.5).abs() < 1e-15);
        assert!((line.arc_length(0.0, 1.5) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn a_chain_finds_its_piece() {
        let c = Curve::circle([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 1.0).unwrap();
        let arc: Vec<V3> = (0..=6).map(|k| c.eval(-0.5 * k as f64)).collect();
        let t = c.piece(&arc, 1e-9).unwrap();
        assert!(t[0].abs() < 1e-12 && (t[1] + 3.0).abs() < 1e-12);
        // A whole turn the way the chain runs.
        let ring: Vec<V3> = (0..=8).map(|k| c.eval(TAU * k as f64 / 8.0)).collect();
        let t = c.piece(&ring, 1e-9).unwrap();
        assert!((t[1] - t[0] - TAU).abs() < 1e-12);
        // Two points half a turn apart: either way round.
        assert!(c.piece(&[c.eval(0.0), c.eval(PI)], 1e-9).is_none());
    }

    #[test]
    fn a_map_carries_a_circle_to_its_ellipse() {
        let c = Curve::circle([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 2.0).unwrap();
        let m = Affine::stretch([0.0; 3], [3.0, 1.0, 1.0]);
        let e = c.mapped(&m);
        assert_eq!(e.name(), "ellipse");
        for t in [0.0, 0.7, 2.0, 4.0] {
            let x = m.point(c.eval(t));
            assert!(close(e.eval(e.param(x)), x));
            assert!(close(e.eval(t), x), "the parameter carries over");
        }
        let turned = c.mapped(&Affine::rotation([0.0; 3], [1.0, 2.0, 3.0], 1.0).unwrap());
        assert_eq!(turned.name(), "circle");
    }

    #[test]
    fn a_profile_lifts_exactly() {
        let n = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0],
        );
        let f = Frame::new([0.0, 0.0, 5.0], [0.0, 0.0, 1.0], Some([0.0, 1.0, 0.0])).unwrap();
        let plane = Curve::<2>::Nurbs(Arc::new(n));
        let space = plane.lifted(&f, 2.0);
        for t in [0.1, 0.5, 0.9] {
            let x = plane.eval(t);
            assert!(close(space.eval(t), f.at(x[0], x[1], 2.0)));
            assert!((space.curvature(t) - 1.0).abs() < 1e-9);
        }
    }
}
