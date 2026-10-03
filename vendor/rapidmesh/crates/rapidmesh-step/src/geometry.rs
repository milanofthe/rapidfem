//! The geometry of a STEP file: placements, curves and surfaces, each with
//! its evaluation and the parameter of a point on it.

use rapidmesh_geom::vec3::{add, cross, dist, dot, len, normalize, scale, sub};
use rapidmesh_geom::{NurbsCurve, NurbsSurface, SurfaceKind};
use std::f64::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

pub type P3 = [f64; 3];

/// A right-handed orthonormal frame: an axis placement of the file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Axes {
    pub o: P3,
    pub x: P3,
    pub y: P3,
    pub z: P3,
}

impl Axes {
    /// The frame at `o` with axis `z` and `x` toward `x_hint` (any
    /// perpendicular direction where there is none).
    pub fn new(o: P3, z: P3, x_hint: Option<P3>) -> Axes {
        let z = normalize(z);
        let hint = x_hint.unwrap_or(if z[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        });
        let x = normalize(sub(hint, scale(z, dot(hint, z))));
        Axes {
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
    Spline(NurbsCurve<2>),
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
        f: Axes,
        r: f64,
    },
    Ellipse {
        f: Axes,
        a: f64,
        b: f64,
    },
    /// `a cosh t` along x, `b sinh t` along y.
    Hyperbola {
        f: Axes,
        a: f64,
        b: f64,
    },
    /// `focal t^2` along x, `2 focal t` along y.
    Parabola {
        f: Axes,
        focal: f64,
    },
    Spline(NurbsCurve<3>),
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
                    .fold(0.0f64, |m, c| m.max(len(sub(*c, s.ctrl[0]))))
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

/// A swept surface of the file as one the model knows, and whether its
/// normal runs against the swept one's (the faces on it then turn).
pub struct Swept {
    pub surface: Surface,
    pub flipped: bool,
}

/// The surface `profile` sweeps turning about the axis through `o` along
/// `axis` (SURFACE_OF_REVOLUTION, its normal the turn's direction crossed
/// with the profile's): a line in a plane of the axis sweeps a cylinder, a
/// cone or a plane, a circle there a torus or a sphere, a B-spline the
/// rational B-spline surface of the turn. `None` for any other.
pub fn revolved(profile: &Curve, o: P3, axis: P3) -> Option<Swept> {
    let a = normalize(axis);
    let foot = |p: P3| add(o, scale(a, dot(sub(p, o), a)));
    let surface = match profile {
        Curve::Line { p, d } => {
            let e = sub(*p, foot(*p));
            let (along, across) = (dot(*d, a), len(cross(*d, a)));
            // A line off the planes of the axis sweeps a hyperboloid.
            if dot(cross(*d, a), sub(*p, o)).abs() > 1e-9 * len(*d) * (1.0 + len(sub(*p, o))) {
                return None;
            }
            if across <= 1e-12 * len(*d) {
                Surface::Cylinder(Axes::new(foot(*p), a, Some(e)), len(e))
            } else if along.abs() <= 1e-12 * len(*d) {
                Surface::Plane(Axes::new(foot(*p), a, None))
            } else {
                let out = if len(e) > 0.0 {
                    normalize(e)
                } else {
                    normalize(sub(*d, scale(a, along)))
                };
                let semi = (dot(*d, out) / along).atan();
                Surface::Cone(Axes::new(foot(*p), a, Some(out)), len(e), semi)
            }
        }
        Curve::Circle { f, r } => {
            if dot(f.z, a).abs() > 1e-9 || dot(sub(o, f.o), f.z).abs() > 1e-9 * (1.0 + *r) {
                return None;
            }
            let e = sub(f.o, foot(f.o));
            if len(e) <= 1e-12 * r {
                Surface::Sphere(Axes::new(foot(f.o), a, None), *r)
            } else {
                Surface::Torus(Axes::new(foot(f.o), a, Some(e)), len(e), *r)
            }
        }
        Curve::Spline(c) => {
            // The turn as a rational quadratic through nine points a
            // quarter turn apart (Piegl and Tiller, A8.1), per control.
            let h = std::f64::consts::FRAC_1_SQRT_2;
            let nv = c.ctrl.len();
            let (mut ctrl, mut weights) = (vec![[0.0; 3]; 9 * nv], vec![0.0; 9 * nv]);
            for (j, (&q, &w)) in c.ctrl.iter().zip(&c.weights).enumerate() {
                let base = foot(q);
                let x = sub(q, base);
                let y = cross(a, x);
                for i in 0..9 {
                    let t = i as f64 * std::f64::consts::FRAC_PI_4;
                    let (k, wk) = if i % 2 == 0 { (1.0, 1.0) } else { (1.0 / h, h) };
                    ctrl[i * nv + j] =
                        add(base, scale(add(scale(x, t.cos()), scale(y, t.sin())), k));
                    weights[i * nv + j] = w * wk;
                }
            }
            let turn = [
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ]
            .map(|k| k * TAU)
            .to_vec();
            let s = NurbsSurface::try_new(
                [2, c.degree],
                [turn, c.knots.clone()],
                [9, nv],
                ctrl,
                weights,
            )
            .ok()?;
            Surface::Spline(Arc::new(s))
        }
        _ => return None,
    };
    // The normal at a point of the profile off the axis: the turn's
    // direction there crossed with the profile's.
    let at = |t: f64| {
        let (p, q) = (profile.eval(t), profile.eval(t + 1e-6));
        (p, cross(cross(a, sub(p, foot(p))), sub(q, p)))
    };
    Some(oriented(surface, at, profile_probes(profile)))
}

/// The surface `profile` sweeps moving along `d` (SURFACE_OF_LINEAR_EXTRUSION,
/// its normal the profile's direction crossed with `d`): a line sweeps a
/// plane, a circle about `d` a cylinder, a B-spline the B-spline surface of
/// the sweep. `None` for any other.
pub fn extruded(profile: &Curve, d: P3) -> Option<Swept> {
    let surface = match profile {
        Curve::Line { p, d: along } => {
            let n = cross(*along, d);
            if len(n) <= 1e-12 * len(*along) * len(d) {
                return None;
            }
            Surface::Plane(Axes::new(*p, n, Some(*along)))
        }
        Curve::Circle { f, r } if len(cross(f.z, normalize(d))) <= 1e-9 => {
            Surface::Cylinder(Axes::new(f.o, normalize(d), Some(f.x)), *r)
        }
        Curve::Spline(c) => {
            let nu = c.ctrl.len();
            let mut ctrl = Vec::with_capacity(2 * nu);
            let mut weights = Vec::with_capacity(2 * nu);
            for (&q, &w) in c.ctrl.iter().zip(&c.weights) {
                ctrl.extend([q, add(q, d)]);
                weights.extend([w, w]);
            }
            let s = NurbsSurface::try_new(
                [c.degree, 1],
                [c.knots.clone(), vec![0.0, 0.0, 1.0, 1.0]],
                [nu, 2],
                ctrl,
                weights,
            )
            .ok()?;
            Surface::Spline(Arc::new(s))
        }
        _ => return None,
    };
    let at = |t: f64| {
        let (p, q) = (profile.eval(t), profile.eval(t + 1e-6));
        (p, cross(sub(q, p), d))
    };
    Some(oriented(surface, at, profile_probes(profile)))
}

/// Parameters along a profile to compare normals at: within a spline's
/// domain, a line's vector (from its point on, where a profile starts),
/// all round a conic.
fn profile_probes(profile: &Curve) -> Vec<f64> {
    let (lo, hi) = match profile {
        Curve::Spline(s) => s.domain(),
        Curve::Line { .. } => (0.0, 1.0),
        _ => (0.0, TAU),
    };
    (1..8).map(|i| lo + (hi - lo) * i as f64 / 8.0).collect()
}

/// `surface` with whether its normal runs against the swept one, `at(t)`
/// a point of the profile and the swept normal there: compared where that
/// normal is longest.
fn oriented(surface: Surface, at: impl Fn(f64) -> (P3, P3), probes: Vec<f64>) -> Swept {
    let (p, n) = probes
        .into_iter()
        .map(at)
        .max_by(|x, y| len(x.1).total_cmp(&len(y.1)))
        .expect("probes");
    let flipped = dot(surface.normal(surface.param(p)), n) < 0.0;
    Swept { surface, flipped }
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
    Plane(Axes),
    /// `(theta, height)`
    Cylinder(Axes, f64),
    /// `(theta, height)`, the radius `r + height tan(semi)`.
    Cone(Axes, f64, f64),
    /// `(theta, latitude)`
    Sphere(Axes, f64),
    /// `(theta, phi)`: major and minor radius.
    Torus(Axes, f64, f64),
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
                    .fold(0.0f64, |m, c| m.max(len(sub(*c, s.ctrl[0]))))
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
                    .fold(0.0f64, |m, c| m.max(len(sub(*c, s.ctrl[0]))))
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
            if len(c) > 1e-12 * len(s_u) * len(s_v) {
                dot(x, c).abs() / len(c)
            } else {
                len(x)
            }
        };
        Bend {
            stretch: [len(s_u), len(s_v)],
            bend: [part(s_uu), part(s_vv)],
            twist: part(s_uv),
        }
    }

    /// The normal at `uv` (unit, from the parameters' directions by
    /// differences): the first parameter's direction crossed with the
    /// second's.
    pub fn normal(&self, uv: [f64; 2]) -> P3 {
        let h = 1e-6;
        let s_u = sub(self.eval([uv[0] + h, uv[1]]), self.eval([uv[0] - h, uv[1]]));
        let s_v = sub(self.eval([uv[0], uv[1] + h]), self.eval([uv[0], uv[1] - h]));
        normalize(cross(s_u, s_v))
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
                x: f.x,
                radius: *r,
            },
            Surface::Cone(f, r, semi) => SurfaceKind::Cone {
                apex: add(f.o, scale(f.z, -r / semi.tan())),
                axis: f.z,
                x: f.x,
                tan_half_angle: semi.tan(),
            },
            Surface::Sphere(f, r) => SurfaceKind::Sphere {
                center: f.o,
                axis: f.z,
                x: f.x,
                radius: *r,
            },
            Surface::Torus(f, big, small) => SurfaceKind::Torus {
                center: f.o,
                axis: f.z,
                x: f.x,
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
        let f = Axes::new([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], Some([1.0, 0.0, 0.0]));
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
        let f = Axes::new([1.0, 2.0, 3.0], [0.3, -0.2, 1.0], Some([1.0, 0.0, 0.0]));
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

    /// Points a profile sweeps turning about an axis lie on the surface it
    /// reads as, the normals alike once a flipped one turns (but on a
    /// sphere: a whole circle about its centre covers it twice, facing
    /// both ways, and the faces on it settle which).
    #[test]
    fn swept_profiles_read_as_the_surfaces_they_sweep() {
        let (o, a) = ([1.0, 0.5, 0.0], normalize([0.0, 0.2, 1.0]));
        let up = Axes::new(
            [1.0, 0.5, 0.0],
            cross(a, [1.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
        );
        let profiles = [
            Curve::Line {
                p: [2.0, 0.5, 0.0],
                d: a,
            },
            Curve::Line {
                p: [2.0, 0.5, 0.0],
                d: add(a, [0.5, 0.0, 0.0]),
            },
            Curve::Line {
                p: [2.0, 0.5, 0.0],
                d: [-1.0, 0.0, 0.0],
            },
            Curve::Circle {
                f: Axes {
                    z: scale(up.z, -1.0),
                    y: scale(up.y, -1.0),
                    ..up
                },
                r: 0.5,
            },
            Curve::Circle {
                f: Axes {
                    o: [2.5, 0.5, 0.0],
                    ..up
                },
                r: 0.5,
            },
            Curve::Spline(NurbsCurve::<3> {
                degree: 2,
                knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                ctrl: vec![[2.0, 0.5, 0.0], [3.0, 0.7, 1.0], [2.0, 0.9, 2.0]],
                weights: vec![1.0, 0.7, 1.0],
            }),
        ];
        for c in &profiles {
            let swept = revolved(c, o, a).unwrap_or_else(|| panic!("{c:?}"));
            for t in [0.1, 0.4, 0.8] {
                for turn in [0.3f64, 2.0, 4.5] {
                    // Turned about the axis by Rodrigues' formula.
                    let q = sub(c.eval(t), o);
                    let (s, k) = turn.sin_cos();
                    let r = |q: P3| {
                        add(
                            add(scale(q, k), scale(cross(a, q), s)),
                            scale(a, dot(a, q) * (1.0 - k)),
                        )
                    };
                    let p = add(o, r(q));
                    let back = swept.surface.eval(swept.surface.param(p));
                    assert!(
                        dist(back, p) < 1e-6,
                        "{c:?}: {p:?} off by {}",
                        dist(back, p)
                    );
                    let d = sub(c.eval(t + 1e-6), c.eval(t));
                    let n = cross(cross(a, sub(p, o)), r(d));
                    let sphere = matches!(swept.surface, Surface::Sphere(..));
                    if !sphere && len(cross(a, sub(p, o))) > 1e-6 {
                        let mut m = swept.surface.normal(swept.surface.param(p));
                        if swept.flipped {
                            m = scale(m, -1.0);
                        }
                        assert!(dot(m, normalize(n)) > 0.99, "{c:?}: normal {m:?} {n:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_spline_with_unit_weights_is_polynomial() {
        // A quadratic Bezier arc.
        let s = NurbsCurve::<3> {
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
