//! Surfaces of revolution charted conformally.
//!
//! On a surface turned about an axis, its meridian `(rho, z)` by arc length
//! `s`, lengths go by `ds^2 + rho^2 du^2 = rho^2 (du^2 + dsigma^2)` with
//! `dsigma = ds / rho`: the coordinates `(u, sigma)` keep angles and scale
//! lengths by `rho`. Two charts follow from them:
//!
//! - the strip `r0 (u, sigma)` for a face off the axis: a cylinder (where
//!   it is the unrolling), a band of a torus, a part of a revolved profile;
//! - the polar `R (cos k u, sin k u)`, `R` growing as `exp(k sigma)`, round
//!   a pole where the meridian meets the axis at slope `k`: on a cone (`k`
//!   the sine of its half angle) the unrolling, on a sphere (`k = 1`) the
//!   stereographic projection.

use crate::{Curve, Surface};
use rapidmesh_exact::vector::{len, V2};
use std::f64::consts::{FRAC_PI_2, TAU};

/// A surface of revolution about the axis of its frame, by its meridian
/// `(distance from the axis, height along it)` and the form of its chart.
/// The meridian's parameter `t` is the surface's `v` but on a line, whose
/// `v` is the height (see [`Revolution::t`]).
pub(super) struct Revolution {
    meridian: Curve<2>,
    pub(super) form: Form,
    /// The period of the meridian's parameter where it closes (a torus's
    /// tube angle, a closed profile's domain).
    pub(super) period: Option<f64>,
}

pub(super) enum Form {
    /// A line square to nothing but parallel to the axis at distance `rho`
    /// (a cylinder), `len` the length of its direction: `sigma = len t /
    /// rho`.
    StripLine { rho: f64, len: f64 },
    /// A circle about `(c, z)` of radius `r` off the axis (`c > r`, a ring
    /// torus), by its angle `t`.
    StripCircle { c: f64, r: f64 },
    /// A B-spline meridian clear of the axis: `sigma` at the parameters
    /// `t` (increasing, over its domain).
    StripTable { t: Vec<f64>, sigma: Vec<f64> },
    /// A line meeting the axis at `t0` (a cone): `R = len side (t - t0)`,
    /// the distance from the apex, `side` the sign of `t - t0` on the
    /// surface, `k` the sine of the half angle.
    PolarLine {
        t0: f64,
        len: f64,
        side: f64,
        k: f64,
    },
    /// A circle of radius `r` centred on the axis (a sphere), round its
    /// pole at angle `pi / 2`: `R = 2 r tan(psi / 2)`, `psi = pi / 2 - t`
    /// the angle from the pole.
    PolarSphere { r: f64 },
}

impl Revolution {
    /// The chart of a surface of revolution about its axis, or none where
    /// it has none of these forms: a cone near flat or near a cylinder, a
    /// spindle torus, a profile reaching the axis.
    pub(super) fn of(surface: &Surface) -> Option<Revolution> {
        let frame = *surface.frame()?;
        let size = match surface {
            Surface::Cylinder { radius, .. } | Surface::Sphere { radius, .. } => *radius,
            Surface::Torus { major, .. } => *major,
            _ => 1.0,
        };
        let meridian = surface.meridian(frame.o, frame.z, 1e-9 * size.max(1.0))?;
        let form = match &meridian {
            Curve::Line { p, d } => {
                let l = len(*d);
                let b = d[0] / l;
                if b.abs() <= 1e-12 {
                    if !(p[0] > 0.0) {
                        return None;
                    }
                    Form::StripLine { rho: p[0], len: l }
                } else {
                    // A cone is charted where it neither lies flat nor
                    // stands near a cylinder.
                    if !(b.abs() > 1e-6 && b.abs() < 1.0 - 1e-12) {
                        return None;
                    }
                    Form::PolarLine {
                        t0: -p[0] / d[0],
                        len: l,
                        side: d[0].signum(),
                        k: b.abs(),
                    }
                }
            }
            Curve::Ellipse { .. } => {
                let (c, r) = meridian.as_circle()?;
                if c[0].abs() <= 1e-9 * r {
                    Form::PolarSphere { r }
                } else if c[0] > r * (1.0 + 1e-9) {
                    Form::StripCircle { c: c[0], r }
                } else {
                    return None;
                }
            }
            Curve::Nurbs(n) => {
                let (lo, hi) = n.domain();
                const N: usize = 2048;
                let t: Vec<f64> = (0..=N)
                    .map(|i| lo + (hi - lo) * i as f64 / N as f64)
                    .collect();
                let extent = n
                    .ctrl
                    .iter()
                    .map(|c| c[0].abs().max(c[1].abs()))
                    .fold(0.0, f64::max);
                let rate = |t: f64| {
                    let (q, d, _) = meridian.ders(t);
                    (q[0] > 1e-9 * extent).then(|| len(d) / q[0])
                };
                let mut sigma = vec![0.0];
                for w in t.windows(2) {
                    let m = 0.5 * (w[0] + w[1]);
                    // Simpson on each step.
                    let s = (rate(w[0])? + 4.0 * rate(m)? + rate(w[1])?) * (w[1] - w[0]) / 6.0;
                    sigma.push(sigma[sigma.len() - 1] + s);
                }
                Form::StripTable { t, sigma }
            }
            _ => return None,
        };
        let period = match (&form, &meridian) {
            (Form::StripCircle { .. }, _) => Some(TAU),
            (Form::StripTable { .. }, Curve::Nurbs(n)) => {
                let (lo, hi) = n.domain();
                let extent = n.ctrl.iter().map(|c| len(*c)).fold(0.0, f64::max);
                (rapidmesh_exact::vector::dist(n.eval(lo), n.eval(hi)) <= 1e-9 * extent)
                    .then_some(hi - lo)
            }
            _ => None,
        };
        Some(Revolution {
            meridian,
            form,
            period,
        })
    }

    /// Whether the chart is polar (round a pole) rather than a strip.
    pub(super) fn polar(&self) -> bool {
        matches!(self.form, Form::PolarLine { .. } | Form::PolarSphere { .. })
    }

    /// The slope at the pole: the polar chart turns by `k` times the angle
    /// round the axis.
    pub(super) fn k(&self) -> f64 {
        match self.form {
            Form::PolarLine { k, .. } => k,
            _ => 1.0,
        }
    }

    /// The meridian's parameter at the surface's `v`: on a line (a
    /// cylinder, a cone) `v` is the height.
    pub(super) fn t(&self, v: f64) -> f64 {
        match &self.meridian {
            Curve::Line { p, d } => (v - p[1]) / d[1],
            _ => v,
        }
    }

    /// The surface's `v` at the meridian's parameter `t`.
    pub(super) fn v(&self, t: f64) -> f64 {
        match &self.meridian {
            Curve::Line { p, d } => p[1] + t * d[1],
            _ => t,
        }
    }

    /// Whether the chart is a sphere's stereographic projection, which
    /// goes round its pole whole.
    pub(super) fn sphere(&self) -> bool {
        matches!(self.form, Form::PolarSphere { .. })
    }

    /// The distance from the axis at `t`.
    pub(super) fn rho(&self, t: f64) -> f64 {
        self.meridian.eval(t)[0]
    }

    /// The conformal height at `t` of a strip.
    pub(super) fn sigma(&self, t: f64) -> f64 {
        match &self.form {
            Form::StripLine { rho, len } => t * len / rho,
            &Form::StripCircle { c, r } => {
                let (k, e) = circle_constants(c, r);
                2.0 * r / k * unwrapped_atan(e, 0.5 * t)
            }
            Form::StripTable { t: ts, sigma } => {
                // A closed profile goes on round: by whole turns, each
                // adding the conformal height of one.
                let turns = self.period.map_or(0.0, |p| ((t - ts[0]) / p).floor());
                interpolate(ts, sigma, t - turns * self.period.unwrap_or(0.0))
                    + turns * (sigma[sigma.len() - 1] - sigma[0])
            }
            _ => 0.0,
        }
    }

    /// The parameter of conformal height `s` of a strip.
    pub(super) fn sigma_inv(&self, s: f64) -> f64 {
        match &self.form {
            Form::StripLine { rho, len } => s * rho / len,
            &Form::StripCircle { c, r } => {
                let (k, e) = circle_constants(c, r);
                let y = s * k / (2.0 * r);
                let n = (y / std::f64::consts::PI).round();
                let y = y - n * std::f64::consts::PI;
                2.0 * ((y.tan() / e).atan() + n * std::f64::consts::PI)
            }
            Form::StripTable { t, sigma } => {
                let total = sigma[sigma.len() - 1] - sigma[0];
                let turns = match self.period {
                    Some(_) if total > 0.0 => ((s - sigma[0]) / total).floor(),
                    _ => 0.0,
                };
                interpolate(sigma, t, s - turns * total) + turns * self.period.unwrap_or(0.0)
            }
            _ => 0.0,
        }
    }

    /// The radius in a polar chart at `t`.
    pub(super) fn radius(&self, t: f64) -> f64 {
        match self.form {
            Form::PolarLine { t0, len, side, .. } => len * side * (t - t0),
            Form::PolarSphere { r } => 2.0 * r * (0.5 * (FRAC_PI_2 - t)).tan(),
            _ => 0.0,
        }
    }

    /// The parameter at radius `radius` of a polar chart.
    pub(super) fn radius_inv(&self, radius: f64) -> f64 {
        match self.form {
            Form::PolarLine { t0, len, side, .. } => t0 + side * radius / len,
            Form::PolarSphere { r } => FRAC_PI_2 - 2.0 * (radius / (2.0 * r)).atan(),
            _ => 0.0,
        }
    }

    /// The chart length of a unit length on a polar chart at radius `radius`.
    pub(super) fn polar_shrink(&self, radius: f64) -> f64 {
        match self.form {
            Form::PolarSphere { r } => 1.0 + radius * radius / (4.0 * r * r),
            _ => 1.0,
        }
    }
}

/// For a ring torus `c > r`: `k = sqrt(c^2 - r^2)` and `e = sqrt((c - r) /
/// (c + r))`, by which the conformal height is `2 r / k atan(e tan(t /
/// 2))`.
fn circle_constants(c: f64, r: f64) -> (f64, f64) {
    ((c * c - r * r).sqrt(), ((c - r) / (c + r)).sqrt())
}

/// `atan(e tan(psi))` continued through the poles of the tangent: it grows
/// by `pi` each half turn of `psi` (`e > 0`).
fn unwrapped_atan(e: f64, psi: f64) -> f64 {
    let (s, c) = psi.sin_cos();
    psi + ((e - 1.0) * s * c).atan2(c * c + e * s * s)
}

/// Linear interpolation from one increasing table to the other, the end
/// steps carried on outside.
fn interpolate(from: &[f64], to: &[f64], x: f64) -> f64 {
    let i = from.partition_point(|&v| v <= x).clamp(1, from.len() - 1);
    let (a, b) = (from[i - 1], from[i]);
    let w = if b > a { (x - a) / (b - a) } else { 0.0 };
    to[i - 1] + w * (to[i] - to[i - 1])
}

/// The point at `radius` and `angle` of a polar chart.
pub(super) fn polar_point(radius: f64, angle: f64) -> V2 {
    [radius * angle.cos(), radius * angle.sin()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NurbsCurve;
    use rapidmesh_exact::vector::{dist, dot, sub, Frame};
    use std::sync::Arc;

    /// Every form keeps angles: a step along the chart's axes maps to two
    /// square steps on the surface of one length, as the chart says.
    #[test]
    fn every_form_is_conformal_and_inverts() {
        let f = Frame::new([0.5, -1.0, 2.0], [0.2, 0.3, 1.0], None).unwrap();
        let profile = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![[1.0, 0.0], [2.0, 1.0], [1.5, 2.0]],
            vec![1.0, 0.8, 1.0],
        );
        let surfaces = [
            Surface::cylinder(f.o, f.z, 1.5),
            Surface::cone(f.o, f.z, 0.4),
            Surface::sphere(f.o, 2.0),
            Surface::torus(f.o, f.z, 3.0, 1.0),
            Surface::Revolved {
                frame: f,
                profile: Arc::new(profile),
            },
        ];
        for s in &surfaces {
            let rev = Revolution::of(s).unwrap_or_else(|| panic!("{}", s.name()));
            let chart = |uv: V2| -> V2 {
                let t = rev.t(uv[1]);
                if rev.polar() {
                    polar_point(rev.radius(t), rev.k() * uv[0])
                } else {
                    [uv[0], rev.sigma(t)]
                }
            };
            let back = |q: V2| -> V2 {
                if rev.polar() {
                    [q[1].atan2(q[0]) / rev.k(), rev.v(rev.radius_inv(len(q)))]
                } else {
                    [q[0], rev.v(rev.sigma_inv(q[1]))]
                }
            };
            for uv in [[0.3, 0.4], [2.0, 0.7], [-1.0, 0.2]] {
                let p = s.eval(uv);
                let q = chart(uv);
                assert!(dist(s.eval(back(q)), p) < 1e-9, "{} inverse", s.name());
                let h = 1e-6;
                let (x, y) = (
                    s.eval(back([q[0] + h, q[1]])),
                    s.eval(back([q[0], q[1] + h])),
                );
                let (dx, dy) = (sub(x, p), sub(y, p));
                let (lx, ly) = (len(dx), len(dy));
                assert!(
                    (lx - ly).abs() < 1e-3 * lx,
                    "{} lengths {lx} {ly}",
                    s.name()
                );
                assert!(dot(dx, dy).abs() < 1e-3 * lx * ly, "{} square", s.name());
                // The strip's chart length per length is 1 / rho, the
                // polar one's as `polar_shrink` gives.
                let want = if rev.polar() {
                    rev.polar_shrink(len(q))
                } else {
                    1.0 / rev.rho(rev.t(uv[1]))
                };
                assert!((h / lx - want).abs() < 1e-3 * want, "{} scale", s.name());
            }
        }
    }

    /// The torus's conformal height grows by `r / rho` per tube angle all
    /// the way round, through the inside of the ring.
    #[test]
    fn a_torus_strip_runs_on_through_the_ring() {
        let rev = Revolution::of(&Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 0.575, 0.125)).unwrap();
        for t in [-3.0, -1.2, 0.0, 0.7, 2.5, 3.1, 3.3, 4.0, 9.0] {
            assert!((rev.sigma_inv(rev.sigma(t)) - t).abs() < 1e-12, "{t}");
            let h = 1e-6;
            let slope = (rev.sigma(t + h) - rev.sigma(t - h)) / (2.0 * h);
            let want = 0.125 / (0.575 + 0.125 * t.cos());
            assert!((slope - want).abs() < 1e-6 * want, "{t}: {slope} {want}");
        }
    }
}
