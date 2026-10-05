//! The geometry of a STEP file in the model's terms: placements by the
//! file's rule, its cone by the model's parameters, and its swept surfaces
//! as the kinds the model knows. Curves and surfaces themselves are the
//! model's ([`rapidmesh_geom::Curve`], [`rapidmesh_geom::Surface`]).

use rapidmesh_exact::vector::{add, cross, dot, len, normalize, scale, sub, unit, Frame, V3};
use rapidmesh_geom::{Curve, NurbsSurface, Surface};
use std::f64::consts::TAU;
use std::sync::Arc;

/// The frame of an axis placement: `x` towards `x_hint` (its part square
/// to `z`), else towards the x axis (the y axis where `z` lies close to it),
/// the file's rule for a placement without a reference direction.
pub fn placement(o: V3, z: V3, x_hint: Option<V3>) -> Option<Frame> {
    let z = unit(z)?;
    let hint = x_hint.unwrap_or(if z[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    });
    Frame::new(o, z, Some(hint))
}

/// How the file's parameters of a surface map to the model's: `(u, v)` to
/// `(s[0] u + t[0], s[1] v + t[1])` (the identity but on a cone).
pub type Reparam = ([f64; 2], [f64; 2]);

/// The cone of the file whose radius is `r + v tan(semi)` at height `v`
/// over the placement `f`: the model's, at its apex and opening by a
/// positive half angle, its `v` the distance from the apex; with how the
/// file's parameters map to it.
pub fn cone(f: Frame, r: f64, semi: f64) -> (Surface, Reparam) {
    let shift = r / semi.tan();
    let apex = f.at(0.0, 0.0, -shift);
    if semi > 0.0 {
        let frame = Frame { o: apex, ..f };
        let s = Surface::Cone {
            frame,
            half_angle: semi,
        };
        (s, ([1.0, 1.0], [0.0, shift]))
    } else {
        // Opening against the axis: the axis turned round (`y` with it, the
        // frame staying right-handed), the angle and the height turned too.
        let frame = Frame {
            o: apex,
            x: f.x,
            y: scale(f.y, -1.0),
            z: scale(f.z, -1.0),
        };
        let s = Surface::Cone {
            frame,
            half_angle: -semi,
        };
        (s, ([-1.0, -1.0], [0.0, -shift]))
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
pub fn revolved(profile: &Curve<3>, o: V3, axis: V3) -> Option<Swept> {
    let a = normalize(axis);
    let foot = |p: V3| add(o, scale(a, dot(sub(p, o), a)));
    let surface = match profile {
        Curve::Line { p, d } => {
            let e = sub(*p, foot(*p));
            let (along, across) = (dot(*d, a), len(cross(*d, a)));
            // A line off the planes of the axis sweeps a hyperboloid.
            if dot(cross(*d, a), sub(*p, o)).abs() > 1e-9 * len(*d) * (1.0 + len(sub(*p, o))) {
                return None;
            }
            if across <= 1e-12 * len(*d) {
                Surface::Cylinder {
                    frame: placement(foot(*p), a, Some(e))?,
                    radius: len(e),
                }
            } else if along.abs() <= 1e-12 * len(*d) {
                Surface::Plane(placement(foot(*p), a, None)?)
            } else {
                let out = if len(e) > 0.0 {
                    normalize(e)
                } else {
                    normalize(sub(*d, scale(a, along)))
                };
                let semi = (dot(*d, out) / along).atan();
                cone(placement(foot(*p), a, Some(out))?, len(e), semi).0
            }
        }
        Curve::Ellipse { c, .. } => {
            let (_, r) = profile.as_circle()?;
            let z = profile.axis()?;
            if dot(z, a).abs() > 1e-9 || dot(sub(o, *c), z).abs() > 1e-9 * (1.0 + r) {
                return None;
            }
            let e = sub(*c, foot(*c));
            if len(e) <= 1e-12 * r {
                Surface::Sphere {
                    frame: placement(foot(*c), a, None)?,
                    radius: r,
                }
            } else {
                Surface::Torus {
                    frame: placement(foot(*c), a, Some(e))?,
                    major: len(e),
                    minor: r,
                }
            }
        }
        Curve::Nurbs(c) => {
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
            Surface::Nurbs(Arc::new(s))
        }
        _ => return None,
    };
    // The normal at a point of the profile off the axis: the turn's
    // direction there crossed with the profile's.
    let at = |t: f64| {
        let (p, d, _) = profile.ders(t);
        (p, cross(cross(a, sub(p, foot(p))), d))
    };
    Some(oriented(surface, at, probes(profile)))
}

/// The surface `profile` sweeps moving along `d` (SURFACE_OF_LINEAR_EXTRUSION,
/// its normal the profile's direction crossed with `d`): a line sweeps a
/// plane, a circle about `d` a cylinder, a B-spline the B-spline surface of
/// the sweep. `None` for any other.
pub fn extruded(profile: &Curve<3>, d: V3) -> Option<Swept> {
    let surface = match profile {
        Curve::Line { p, d: along } => {
            let n = cross(*along, d);
            if len(n) <= 1e-12 * len(*along) * len(d) {
                return None;
            }
            Surface::Plane(placement(*p, n, Some(*along))?)
        }
        Curve::Ellipse { c, p, .. }
            if profile.as_circle().is_some()
                && profile
                    .axis()
                    .is_some_and(|z| len(cross(z, normalize(d))) <= 1e-9) =>
        {
            Surface::Cylinder {
                frame: placement(*c, normalize(d), Some(*p))?,
                radius: profile.as_circle()?.1,
            }
        }
        Curve::Nurbs(c) => {
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
            Surface::Nurbs(Arc::new(s))
        }
        _ => return None,
    };
    let at = |t: f64| {
        let (p, along, _) = profile.ders(t);
        (p, cross(along, d))
    };
    Some(oriented(surface, at, probes(profile)))
}

/// Parameters along a profile to compare normals at: within a spline's
/// domain, a line's vector (from its point on, where a profile starts),
/// all round a conic.
fn probes(profile: &Curve<3>) -> Vec<f64> {
    let (lo, hi) = match profile {
        Curve::Nurbs(s) => s.domain(),
        Curve::Line { .. } => (0.0, 1.0),
        _ => (0.0, TAU),
    };
    (1..8).map(|i| lo + (hi - lo) * i as f64 / 8.0).collect()
}

/// `surface` with whether its normal runs against the swept one, `at(t)`
/// a point of the profile and the swept normal there: compared where that
/// normal is longest.
fn oriented(surface: Surface, at: impl Fn(f64) -> (V3, V3), probes: Vec<f64>) -> Swept {
    let (p, n) = probes
        .into_iter()
        .map(at)
        .max_by(|x, y| len(x.1).total_cmp(&len(y.1)))
        .expect("probes");
    let flipped = dot(surface.normal(surface.param(p)), n) < 0.0;
    Swept { surface, flipped }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_exact::vector::dist;
    use rapidmesh_geom::NurbsCurve;

    /// A file's cone and the model's agree point for point once the
    /// parameters are carried over, either way it opens.
    #[test]
    fn a_cone_carries_its_parameters_over() {
        let f = placement([1.0, 2.0, 3.0], [0.3, -0.2, 1.0], Some([1.0, 0.0, 0.0])).unwrap();
        for semi in [0.4, -0.3] {
            let (r, k) = (2.0, f64::tan(semi));
            let (s, (sc, sh)) = cone(f, r, semi);
            for [u, v] in [[0.3, 0.2], [-2.0, 0.5], [2.5, -0.7]] {
                let file = add(
                    f.at(0.0, 0.0, v),
                    scale(add(scale(f.x, u.cos()), scale(f.y, u.sin())), r + v * k),
                );
                let ours = s.eval([sc[0] * u + sh[0], sc[1] * v + sh[1]]);
                assert!(dist(file, ours) < 1e-12, "{semi}: {file:?} {ours:?}");
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
        let up = placement(
            [1.0, 0.5, 0.0],
            cross(a, [1.0, 0.0, 0.0]),
            Some([1.0, 0.0, 0.0]),
        )
        .unwrap();
        let circle = |c: V3, x: V3, y: V3, r: f64| Curve::Ellipse {
            c,
            p: scale(x, r),
            q: scale(y, r),
        };
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
            circle(up.o, up.x, scale(up.y, -1.0), 0.5),
            circle([2.5, 0.5, 0.0], up.x, up.y, 0.5),
            Curve::Nurbs(Arc::new(NurbsCurve::<3> {
                degree: 2,
                knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                ctrl: vec![[2.0, 0.5, 0.0], [3.0, 0.7, 1.0], [2.0, 0.9, 2.0]],
                weights: vec![1.0, 0.7, 1.0],
            })),
        ];
        for c in &profiles {
            let swept = revolved(c, o, a).unwrap_or_else(|| panic!("{c:?}"));
            for t in [0.1, 0.4, 0.8] {
                for turn in [0.3f64, 2.0, 4.5] {
                    // Turned about the axis by Rodrigues' formula.
                    let q = sub(c.eval(t), o);
                    let (s, k) = turn.sin_cos();
                    let r = |q: V3| {
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
                    let sphere = matches!(swept.surface, Surface::Sphere { .. });
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
}
