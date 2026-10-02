//! The geometry of a STEP file: placements, curves and surfaces, each with
//! its evaluation and the parameter of a point on it.

use rapidmesh_geom::{NurbsSurface, SurfaceKind};
use std::f64::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

pub type P3 = [f64; 3];

pub(crate) fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub(crate) fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub(crate) fn scale(a: P3, s: f64) -> P3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub(crate) fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub(crate) fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub(crate) fn norm(a: P3) -> f64 {
    dot(a, a).sqrt()
}
pub(crate) fn unit(a: P3) -> P3 {
    let l = norm(a);
    if l > 0.0 {
        scale(a, 1.0 / l)
    } else {
        a
    }
}
pub(crate) fn dist(a: P3, b: P3) -> f64 {
    norm(sub(a, b))
}

/// A right-handed orthonormal frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub o: P3,
    pub x: P3,
    pub y: P3,
    pub z: P3,
}

impl Frame {
    /// The frame at `o` with axis `z` and `x` toward `x_hint` (any
    /// perpendicular direction where there is none).
    pub fn new(o: P3, z: P3, x_hint: Option<P3>) -> Frame {
        let z = unit(z);
        let hint = x_hint.unwrap_or(if z[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        });
        let x = unit(sub(hint, scale(z, dot(hint, z))));
        Frame {
            o,
            x,
            y: cross(z, x),
            z,
        }
    }

    fn at(&self, a: f64, b: f64, c: f64) -> P3 {
        add(
            self.o,
            add(scale(self.x, a), add(scale(self.y, b), scale(self.z, c))),
        )
    }

    fn local(&self, p: P3) -> P3 {
        let d = sub(p, self.o);
        [dot(d, self.x), dot(d, self.y), dot(d, self.z)]
    }
}

/// A rational B-spline curve in `N` dimensions, with its knots expanded.
#[derive(Clone, Debug)]
pub struct Spline<const N: usize> {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub ctrl: Vec<[f64; N]>,
    pub weights: Vec<f64>,
}

impl<const N: usize> Spline<N> {
    pub fn domain(&self) -> (f64, f64) {
        (
            self.knots[self.degree],
            self.knots[self.knots.len() - self.degree - 1],
        )
    }

    /// De Boor's algorithm in homogeneous coordinates.
    pub fn eval(&self, t: f64) -> [f64; N] {
        let p = self.degree;
        let (lo, hi) = self.domain();
        let t = t.clamp(lo, hi);
        let n = self.ctrl.len();
        let mut k = p;
        while k + 1 < n && self.knots[k + 1] <= t {
            k += 1;
        }
        // Homogeneous points: the weighted coordinates, then the weight.
        let mut d: Vec<([f64; N], f64)> = (0..=p)
            .map(|j| {
                let i = k + j - p;
                let w = self.weights[i];
                (self.ctrl[i].map(|c| c * w), w)
            })
            .collect();
        for r in 1..=p {
            for j in (r..=p).rev() {
                let i = k + j - p;
                let den = self.knots[i + p + 1 - r] - self.knots[i];
                let a = if den > 0.0 {
                    (t - self.knots[i]) / den
                } else {
                    0.0
                };
                let (prev, cur) = (d[j - 1], d[j]);
                d[j] = (
                    std::array::from_fn(|c| (1.0 - a) * prev.0[c] + a * cur.0[c]),
                    (1.0 - a) * prev.1 + a * cur.1,
                );
            }
        }
        let (h, w) = d[p];
        h.map(|c| c / w)
    }
}

/// A curve in a surface's parameters (a PCURVE of the file).
#[derive(Clone, Debug)]
pub enum Curve2 {
    Line {
        p: [f64; 2],
        d: [f64; 2],
    },
    /// Centre, unit `x` direction (`y` is it turned a quarter), radius.
    Circle {
        o: [f64; 2],
        x: [f64; 2],
        r: f64,
    },
    Ellipse {
        o: [f64; 2],
        x: [f64; 2],
        a: f64,
        b: f64,
    },
    Spline(Spline<2>),
}

impl Curve2 {
    pub fn eval(&self, t: f64) -> [f64; 2] {
        let at = |o: [f64; 2], x: [f64; 2], a: f64, b: f64| {
            let y = [-x[1], x[0]];
            let (c, s) = (a * t.cos(), b * t.sin());
            [o[0] + c * x[0] + s * y[0], o[1] + c * x[1] + s * y[1]]
        };
        match self {
            Curve2::Line { p, d } => [p[0] + t * d[0], p[1] + t * d[1]],
            Curve2::Circle { o, x, r } => at(*o, *x, *r, *r),
            Curve2::Ellipse { o, x, a, b } => at(*o, *x, *a, *b),
            Curve2::Spline(s) => s.eval(t),
        }
    }
}

/// A curve of the file.
#[derive(Clone, Debug)]
pub enum Curve {
    Line {
        p: P3,
        d: P3,
    },
    Circle {
        f: Frame,
        r: f64,
    },
    Ellipse {
        f: Frame,
        a: f64,
        b: f64,
    },
    /// `a cosh t` along x, `b sinh t` along y.
    Hyperbola {
        f: Frame,
        a: f64,
        b: f64,
    },
    /// `focal t^2` along x, `2 focal t` along y.
    Parabola {
        f: Frame,
        focal: f64,
    },
    Spline(Spline<3>),
}

impl Curve {
    pub fn eval(&self, t: f64) -> P3 {
        match self {
            Curve::Line { p, d } => add(*p, scale(*d, t)),
            Curve::Circle { f, r } => f.at(r * t.cos(), r * t.sin(), 0.0),
            Curve::Ellipse { f, a, b } => f.at(a * t.cos(), b * t.sin(), 0.0),
            Curve::Hyperbola { f, a, b } => f.at(a * t.cosh(), b * t.sinh(), 0.0),
            Curve::Parabola { f, focal } => f.at(focal * t * t, 2.0 * focal * t, 0.0),
            Curve::Spline(s) => s.eval(t),
        }
    }

    /// The period of a closed curve: 2 pi for the conics, the domain of a
    /// B-spline whose ends coincide.
    pub fn period(&self) -> Option<f64> {
        match self {
            Curve::Circle { .. } | Curve::Ellipse { .. } => Some(TAU),
            Curve::Line { .. } | Curve::Hyperbola { .. } | Curve::Parabola { .. } => None,
            Curve::Spline(s) => {
                let (lo, hi) = s.domain();
                let size = s
                    .ctrl
                    .iter()
                    .fold(0.0f64, |m, c| m.max(norm(sub(*c, s.ctrl[0]))))
                    .max(1e-300);
                (dist(s.eval(lo), s.eval(hi)) <= 1e-9 * size).then_some(hi - lo)
            }
        }
    }

    /// The parameter of the point of the curve nearest `q`.
    pub fn param(&self, q: P3) -> f64 {
        match self {
            Curve::Line { p, d } => dot(sub(q, *p), *d) / dot(*d, *d),
            Curve::Circle { f, .. } => {
                let l = f.local(q);
                l[1].atan2(l[0])
            }
            Curve::Ellipse { f, a, b } => {
                let l = f.local(q);
                // The angle of the point, refined by Newton on the distance.
                let mut t = (l[1] / b).atan2(l[0] / a);
                for _ in 0..20 {
                    let (c, s) = (t.cos(), t.sin());
                    let (x, y) = (a * c - l[0], b * s - l[1]);
                    let g = -x * a * s + y * b * c;
                    let h = a * a * s * s + b * b * c * c - x * a * c - y * b * s;
                    if h.abs() < 1e-300 {
                        break;
                    }
                    let step = g / h;
                    t -= step;
                    if step.abs() < 1e-14 {
                        break;
                    }
                }
                t
            }
            Curve::Hyperbola { f, a, b } => {
                let l = f.local(q);
                // From the height, refined by Newton on the distance.
                newton(
                    (l[1] / b).asinh(),
                    |t| [a * t.cosh() - l[0], b * t.sinh() - l[1]],
                    |t| [a * t.sinh(), b * t.cosh()],
                    |t| [a * t.cosh(), b * t.sinh()],
                )
            }
            Curve::Parabola { f, focal } => {
                let l = f.local(q);
                newton(
                    l[1] / (2.0 * focal),
                    |t| [focal * t * t - l[0], 2.0 * focal * t - l[1]],
                    |t| [2.0 * focal * t, 2.0 * focal],
                    |_| [2.0 * focal, 0.0],
                )
            }
            Curve::Spline(s) => {
                let (lo, hi) = s.domain();
                // The best of a sampling, refined by golden section.
                let n = 16 * s.ctrl.len().max(4);
                let mut best = (lo, f64::INFINITY);
                for i in 0..=n {
                    let t = lo + (hi - lo) * i as f64 / n as f64;
                    let d = dist(s.eval(t), q);
                    if d < best.1 {
                        best = (t, d);
                    }
                }
                let h = (hi - lo) / n as f64;
                let (mut a, mut b) = ((best.0 - h).max(lo), (best.0 + h).min(hi));
                let g = (5f64.sqrt() - 1.0) / 2.0;
                for _ in 0..80 {
                    let (c, d) = (b - g * (b - a), a + g * (b - a));
                    if dist(s.eval(c), q) < dist(s.eval(d), q) {
                        b = d;
                    } else {
                        a = c;
                    }
                }
                0.5 * (a + b)
            }
        }
    }
}

/// The parameter nearest from `t` of a plane curve whose offset from the
/// point is `r(t)`, with first and second derivatives `d1`, `d2`: Newton
/// on half the squared distance.
fn newton(
    mut t: f64,
    r: impl Fn(f64) -> [f64; 2],
    d1: impl Fn(f64) -> [f64; 2],
    d2: impl Fn(f64) -> [f64; 2],
) -> f64 {
    for _ in 0..30 {
        let (x, a, b) = (r(t), d1(t), d2(t));
        let g = x[0] * a[0] + x[1] * a[1];
        let h = a[0] * a[0] + a[1] * a[1] + x[0] * b[0] + x[1] * b[1];
        if h <= 1e-300 {
            break;
        }
        let step = g / h;
        t -= step;
        if step.abs() < 1e-14 * (1.0 + t.abs()) {
            break;
        }
    }
    t
}

/// How a surface bends at a point (see [`Surface::bend`]).
#[derive(Clone, Copy, Debug)]
pub struct Bend {
    pub stretch: [f64; 2],
    pub bend: [f64; 2],
    pub twist: f64,
}

/// A surface of the file.
#[derive(Clone, Debug)]
pub enum Surface {
    Plane(Frame),
    /// `(theta, height)`
    Cylinder(Frame, f64),
    /// `(theta, height)`, the radius `r + height tan(semi)`.
    Cone(Frame, f64, f64),
    /// `(theta, latitude)`
    Sphere(Frame, f64),
    /// `(theta, phi)`: major and minor radius.
    Torus(Frame, f64, f64),
    Spline(Arc<NurbsSurface>),
}

impl Surface {
    pub fn eval(&self, uv: [f64; 2]) -> P3 {
        let [u, v] = uv;
        match self {
            Surface::Plane(f) => f.at(u, v, 0.0),
            Surface::Cylinder(f, r) => f.at(r * u.cos(), r * u.sin(), v),
            Surface::Cone(f, r, semi) => {
                let rr = r + v * semi.tan();
                f.at(rr * u.cos(), rr * u.sin(), v)
            }
            Surface::Sphere(f, r) => {
                f.at(r * v.cos() * u.cos(), r * v.cos() * u.sin(), r * v.sin())
            }
            Surface::Torus(f, big, small) => {
                let rr = big + small * v.cos();
                f.at(rr * u.cos(), rr * u.sin(), small * v.sin())
            }
            Surface::Spline(s) => s.eval(u, v),
        }
    }

    /// The parameters of the point of the surface nearest `p`, the angles
    /// in `(-pi, pi]`.
    pub fn param(&self, p: P3) -> [f64; 2] {
        match self {
            Surface::Plane(f) => {
                let l = f.local(p);
                [l[0], l[1]]
            }
            Surface::Cylinder(f, _) => {
                let l = f.local(p);
                [l[1].atan2(l[0]), l[2]]
            }
            Surface::Cone(f, r, semi) => {
                let l = f.local(p);
                // The nearest point on the meridian line through (r, 0)
                // with slope tan(semi).
                let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
                let (s, c) = semi.sin_cos();
                let along = (rho - r) * s + l[2] * c;
                [l[1].atan2(l[0]), along * c]
            }
            Surface::Sphere(f, _) => {
                let l = f.local(p);
                let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
                [l[1].atan2(l[0]), l[2].atan2(rho)]
            }
            Surface::Torus(f, big, _) => {
                let l = f.local(p);
                let rho = (l[0] * l[0] + l[1] * l[1]).sqrt();
                [l[1].atan2(l[0]), l[2].atan2(rho - big)]
            }
            Surface::Spline(s) => s.closest_param(p),
        }
    }

    /// The period of each parameter that wraps around: 2 pi for the
    /// angles, the domain of a B-spline direction whose two ends coincide.
    pub fn periods(&self) -> [Option<f64>; 2] {
        match self {
            Surface::Plane(_) => [None, None],
            Surface::Cylinder(..) | Surface::Cone(..) | Surface::Sphere(..) => [Some(TAU), None],
            Surface::Torus(..) => [Some(TAU), Some(TAU)],
            Surface::Spline(s) => {
                let (du, dv) = s.domain();
                let size = s
                    .ctrl
                    .iter()
                    .fold(0.0f64, |m, c| m.max(norm(sub(*c, s.ctrl[0]))))
                    .max(1e-300);
                let closed = |k: usize| {
                    (0..=8).all(|i| {
                        let t = i as f64 / 8.0;
                        let (a, b) = if k == 0 {
                            let v = dv[0] + t * (dv[1] - dv[0]);
                            (s.eval(du[0], v), s.eval(du[1], v))
                        } else {
                            let u = du[0] + t * (du[1] - du[0]);
                            (s.eval(u, dv[0]), s.eval(u, dv[1]))
                        };
                        dist(a, b) <= 1e-9 * size
                    })
                };
                [
                    closed(0).then_some(du[1] - du[0]),
                    closed(1).then_some(dv[1] - dv[0]),
                ]
            }
        }
    }

    /// The lines of the parameters that are one point in space: the
    /// parameter held fixed along one, its value there and the point. The
    /// poles of a sphere, the apex of a cone, a side of a B-spline drawn
    /// together into a point.
    pub fn poles(&self) -> Vec<(usize, f64, P3)> {
        match self {
            Surface::Sphere(..) => [-FRAC_PI_2, FRAC_PI_2]
                .map(|v| (1, v, self.eval([0.0, v])))
                .to_vec(),
            Surface::Cone(_, r, semi) if semi.tan().abs() > 1e-12 => {
                let v = -r / semi.tan();
                vec![(1, v, self.eval([0.0, v]))]
            }
            Surface::Spline(s) => {
                let (du, dv) = s.domain();
                let size = s
                    .ctrl
                    .iter()
                    .fold(0.0f64, |m, c| m.max(norm(sub(*c, s.ctrl[0]))))
                    .max(1e-300);
                let mut out = Vec::new();
                for (k, ends, other) in [(0, du, dv), (1, dv, du)] {
                    for value in ends {
                        let at = |t: f64| {
                            let w = other[0] + t * (other[1] - other[0]);
                            if k == 0 {
                                s.eval(value, w)
                            } else {
                                s.eval(w, value)
                            }
                        };
                        let p = at(0.0);
                        if (1..=8).all(|i| dist(at(i as f64 / 8.0), p) <= 1e-9 * size) {
                            out.push((k, value, p));
                        }
                    }
                }
                out
            }
            _ => Vec::new(),
        }
    }

    /// How the surface bends at `uv`, by differences over `h`: the lengths
    /// of the first derivatives, the normal parts of the second along each
    /// parameter and of the mixed one. A step `d` along parameter `k` strays
    /// from the surface by about `d^2 bend[k] / 8`, one of `du` and `dv`
    /// across a cell by `du dv twist / 4`.
    pub fn bend(&self, uv: [f64; 2], h: [f64; 2]) -> Bend {
        let at = |du: f64, dv: f64| self.eval([uv[0] + du * h[0], uv[1] + dv * h[1]]);
        let p = at(0.0, 0.0);
        let (u0, u1, v0, v1) = (at(-1.0, 0.0), at(1.0, 0.0), at(0.0, -1.0), at(0.0, 1.0));
        let s_u = scale(sub(u1, u0), 0.5 / h[0]);
        let s_v = scale(sub(v1, v0), 0.5 / h[1]);
        let s_uu = scale(add(sub(u1, scale(p, 2.0)), u0), 1.0 / (h[0] * h[0]));
        let s_vv = scale(add(sub(v1, scale(p, 2.0)), v0), 1.0 / (h[1] * h[1]));
        let s_uv = scale(
            sub(
                add(at(1.0, 1.0), at(-1.0, -1.0)),
                add(at(1.0, -1.0), at(-1.0, 1.0)),
            ),
            0.25 / (h[0] * h[1]),
        );
        let c = cross(s_u, s_v);
        // Where the normal is lost (a pole), the whole second derivative.
        let part = |x: P3| {
            if norm(c) > 1e-12 * norm(s_u) * norm(s_v) {
                dot(x, c).abs() / norm(c)
            } else {
                norm(x)
            }
        };
        Bend {
            stretch: [norm(s_u), norm(s_v)],
            bend: [part(s_uu), part(s_vv)],
            twist: part(s_uv),
        }
    }

    /// The carrier for the model.
    pub fn kind(&self) -> SurfaceKind {
        match self {
            Surface::Plane(f) => SurfaceKind::Plane {
                point: f.o,
                normal: f.z,
            },
            Surface::Cylinder(f, r) => SurfaceKind::Cylinder {
                center: f.o,
                axis: f.z,
                radius: *r,
            },
            Surface::Cone(f, r, semi) => SurfaceKind::Cone {
                apex: add(f.o, scale(f.z, -r / semi.tan())),
                axis: f.z,
                tan_half_angle: semi.tan(),
            },
            Surface::Sphere(f, r) => SurfaceKind::Sphere {
                center: f.o,
                radius: *r,
            },
            Surface::Torus(f, big, small) => SurfaceKind::Torus {
                center: f.o,
                axis: f.z,
                major_radius: *big,
                minor_radius: *small,
            },
            Surface::Spline(s) => SurfaceKind::Nurbs(s.clone()),
        }
    }
}

/// `x` shifted by whole periods into `(ref - period/2, ref + period/2]`.
pub fn near(x: f64, reference: f64, period: f64) -> f64 {
    x - period * ((x - reference + 0.5 * period) / period).floor()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A point of a hyperbola or a parabola gives back its parameter.
    #[test]
    fn open_conics_give_back_their_parameters() {
        let f = Frame::new([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], Some([1.0, 0.0, 0.0]));
        for c in [
            Curve::Hyperbola { f, a: 2.0, b: 0.5 },
            Curve::Parabola { f, focal: 0.7 },
        ] {
            for t in [-1.5, -0.2, 0.0, 0.9, 2.0] {
                assert!((c.param(c.eval(t)) - t).abs() < 1e-10, "{c:?} {t}");
            }
        }
    }

    #[test]
    fn surfaces_invert_their_parameters() {
        let f = Frame::new([1.0, 2.0, 3.0], [0.3, -0.2, 1.0], Some([1.0, 0.0, 0.0]));
        for s in [
            Surface::Plane(f),
            Surface::Cylinder(f, 2.0),
            Surface::Cone(f, 2.0, 0.4),
            Surface::Sphere(f, 2.0),
            Surface::Torus(f, 3.0, 1.0),
        ] {
            for uv in [[0.3, 0.2], [-2.0, 0.5], [2.5, -0.7]] {
                let back = s.param(s.eval(uv));
                assert!(
                    dist(s.eval(back), s.eval(uv)) < 1e-9,
                    "{s:?} {uv:?} {back:?}"
                );
            }
        }
    }

    #[test]
    fn a_spline_with_unit_weights_is_polynomial() {
        // A quadratic Bezier arc.
        let s = Spline::<3> {
            degree: 2,
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ctrl: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]],
            weights: vec![1.0; 3],
        };
        let m = s.eval(0.5);
        assert!(dist(m, [1.0, 1.0, 0.0]) < 1e-12);
        let c = Curve::Spline(s);
        assert!((c.param([1.0, 1.2, 0.0]) - 0.5).abs() < 1e-6);
    }
}
