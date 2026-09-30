//! Cylinders and cones unrolled into the plane: both are developable, so
//! the chart keeps lengths and angles, and a face on them is meshed there
//! without distortion. A tube round an open path unrolls the same way as a
//! cylinder, by arc length along its path and angle round it; it stretches
//! by its path's curvature times its radius at most.
//!
//! A face that does not go all the way round is cut where its angles leave
//! the largest gap. A face that does (a barrel between two circles, a cone
//! down to its apex) is cut along a seam: a straight line of the unrolled
//! plane (a helix on a cylinder, a generator on a cone) between a point of
//! one winding loop and the nearest point of the other, or the apex. The
//! seam's points are the face's own; they appear twice in the chart, once on
//! either side, and the lifted triangles meet across the seam.

use super::surface::{Domain2, Slot};
use rapidmesh_brep::Surface;
use rapidmesh_geom::TubePath;

type P2 = [f64; 2];
type P3 = [f64; 3];

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

/// An unrolled cylinder or cone, with the angle the chart is measured from.
pub struct Unroll<'a> {
    surface: &'a Surface,
    kind: Kind,
    /// The angle of the chart's origin.
    theta0: f64,
    /// Whether the chart is cut along a seam (angles from its origin up to
    /// a turn) rather than in a gap (angles either side of its origin).
    seam: bool,
    /// The slant distance of the ring round an apex (0 for none): inside
    /// it the ring, not the size, keeps three points round the axis.
    ring: f64,
    /// A tube's path with its frames.
    tube: Option<Frames<'a>>,
    /// An extruded face's profile by arc length.
    profile: Option<Profile>,
}

/// The profile of an extrusion by arc length: an extrusion is developable,
/// unrolled by arc length along its profile and height along its axis
/// without distortion. A closed profile goes once round (angle `2 pi s /
/// L`), an open one half round, so its ends stay apart.
struct Profile {
    /// Parameters of the profile and the arc lengths there, increasing.
    t: Vec<f64>,
    s: Vec<f64>,
    closed: bool,
}

impl Profile {
    const SAMPLES: usize = 1024;

    fn of(curve: &rapidmesh_geom::NurbsCurve) -> Option<Profile> {
        let (t0, t1) = curve.domain();
        if !(t1 > t0) {
            return None;
        }
        let t: Vec<f64> = (0..=Self::SAMPLES)
            .map(|i| t0 + (t1 - t0) * i as f64 / Self::SAMPLES as f64)
            .collect();
        let mut s = vec![0.0];
        for w in t.windows(2) {
            let (a, b) = (curve.eval(w[0]), curve.eval(w[1]));
            s.push(s[s.len() - 1] + (b[0] - a[0]).hypot(b[1] - a[1]));
        }
        let len = s[s.len() - 1];
        if !(len > 0.0) {
            return None;
        }
        let (a, b) = (curve.eval(t0), curve.eval(t1));
        let closed = (b[0] - a[0]).hypot(b[1] - a[1]) <= 1e-9 * len;
        Some(Profile { t, s, closed })
    }

    fn len(&self) -> f64 {
        self.s[self.s.len() - 1]
    }

    /// The radius of the unrolled cylinder: the profile's length over the
    /// turn it takes.
    fn radius(&self) -> f64 {
        self.len() / if self.closed { TAU } else { PI }
    }

    /// Linear interpolation from one table to the other.
    fn map(from: &[f64], to: &[f64], x: f64) -> f64 {
        let i = from.partition_point(|&v| v <= x).clamp(1, from.len() - 1);
        let (a, b) = (from[i - 1], from[i]);
        let w = if b > a {
            ((x - a) / (b - a)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        to[i - 1] + w * (to[i] - to[i - 1])
    }

    fn arc(&self, t: f64) -> f64 {
        Self::map(&self.t, &self.s, t)
    }

    fn param(&self, s: f64) -> f64 {
        let s = if self.closed {
            s.rem_euclid(self.len())
        } else {
            s
        };
        Self::map(&self.s, &self.t, s)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    /// Arc length round the axis, height along it (along the path for a
    /// tube).
    Cylinder { radius: f64 },
    /// Polar: slant distance from the apex, angle scaled by `sin`.
    Cone { sin: f64, cos: f64 },
    /// A slender full torus: arc length round the axis on the ring of the
    /// tube's centres, arc length round the tube. Periodic both ways; it
    /// stretches by `minor / major` at most.
    Torus { major: f64, minor: f64 },
    /// A band of a torus between two circles round its axis (a fillet),
    /// unrolled conformally: `radius` (the distance from the axis at the
    /// tube angle `v0`) times the angle round the axis, and times the
    /// integral of `minor / rho` over the tube angle from `v0`, `rho` the
    /// distance from the axis. Angles keep; lengths scale by `rho /
    /// radius`, which the sizes follow.
    Band {
        radius: f64,
        major: f64,
        minor: f64,
        v0: f64,
    },
}

/// A torus is unrolled when its tube is at most this share of its ring.
const SLENDER_TORUS: f64 = 0.25;

/// The conformal height of tube angle `v` on a torus of radii `major` and
/// `minor` (ring, `major > minor`), up to a constant: `minor / k` times
/// `2 atan(c tan(v / 2))`, continuous through the inside of the ring, with
/// `k = sqrt(major^2 - minor^2)` and `c = sqrt((major - minor) / (major +
/// minor))`; its derivative is `minor / (major + minor cos v)`.
fn band_height(major: f64, minor: f64, v: f64) -> f64 {
    let k = (major * major - minor * minor).sqrt();
    let c = ((major - minor) / (major + minor)).sqrt();
    minor / k * 2.0 * (c * (0.5 * v).sin()).atan2((0.5 * v).cos())
}

/// The tube angle of conformal height `y` (the inverse of [`band_height`]).
fn band_angle(major: f64, minor: f64, y: f64) -> f64 {
    let k = (major * major - minor * minor).sqrt();
    let c = ((major - minor) / (major + minor)).sqrt();
    let phi = y * k / minor;
    2.0 * (0.5 * phi).sin().atan2(c * (0.5 * phi).cos())
}

/// An open tube path with a frame at each node that turns with it
/// (parallel transport): arc length and angle round the path, continuous
/// across its kinks.
struct Frames<'a> {
    path: &'a TubePath,
    /// The arc length at each node.
    at: Vec<f64>,
    /// The normal of the plane through each node that halves the turn there
    /// (the tangent at the ends): a point between two such planes belongs to
    /// the segment between, at the share of its distances to them.
    plane: Vec<P3>,
    /// The frame in that plane at each node.
    frame: Vec<(P3, P3)>,
}

impl<'a> Frames<'a> {
    fn of(path: &'a TubePath) -> Option<Frames<'a>> {
        let p = &path.pts;
        let m = p.len() - 1;
        let tangent: Vec<P3> = p.windows(2).filter_map(|w| unit(sub(w[1], w[0]))).collect();
        if tangent.len() != m {
            return None;
        }
        let mut at = vec![0.0];
        for w in p.windows(2) {
            at.push(at[at.len() - 1] + dot(sub(w[1], w[0]), sub(w[1], w[0])).sqrt());
        }
        // A closed path turns the face into a torus: no seam of one chart.
        if dot(sub(p[m], p[0]), sub(p[m], p[0])).sqrt() <= 1e-9 * at[m] {
            return None;
        }
        let plane: Vec<P3> = (0..=m)
            .map(|i| match i {
                0 => tangent[0],
                _ if i == m => tangent[m - 1],
                _ => unit(add(tangent[i - 1], tangent[i])).unwrap_or(tangent[i]),
            })
            .collect();
        let mut frame = Vec::with_capacity(m + 1);
        let mut n = unit(perp(plane[0]))?;
        for i in 0..=m {
            if i > 0 {
                n = turn(n, plane[i - 1], plane[i]);
            }
            let x = unit(sub(n, scale(plane[i], dot(n, plane[i]))))?;
            n = x;
            frame.push((x, cross(plane[i], x)));
        }
        Some(Frames {
            path,
            at,
            plane,
            frame,
        })
    }

    /// Angle round the path and arc length along it of `q`.
    fn angle_length(&self, q: P3) -> (f64, f64) {
        let p = &self.path.pts;
        let m = p.len() - 1;
        let mut i = self.path.closest_segment(q);
        let side = |k: usize| dot(sub(q, p[k]), self.plane[k]);
        for _ in 0..m {
            if i > 0 && side(i) < 0.0 {
                i -= 1;
            } else if i + 1 < m && side(i + 1) > 0.0 {
                i += 1;
            } else {
                break;
            }
        }
        let (d0, d1) = (side(i), side(i + 1));
        let t = if d0 - d1 > 0.0 { d0 / (d0 - d1) } else { 0.5 };
        // Past the path's ends the ends' segments go on.
        let t = match (i == 0, i + 1 == m) {
            (true, true) => t,
            (true, false) => t.min(1.0),
            (false, true) => t.max(0.0),
            (false, false) => t.clamp(0.0, 1.0),
        };
        let angle = |k: usize| {
            let w = sub(q, p[k]);
            let (x, y) = self.frame[k];
            dot(w, y).atan2(dot(w, x))
        };
        let (a0, a1) = (angle(i), angle(i + 1));
        let a = a0 + t.clamp(0.0, 1.0) * wrap_pm(a1 - a0);
        (a, self.at[i] + t * (self.at[i + 1] - self.at[i]))
    }

    /// The point at angle `a` and arc length `s` at `radius` from the path
    /// (near the tube; its closest point is on it).
    fn point(&self, a: f64, s: f64, radius: f64) -> P3 {
        let p = &self.path.pts;
        let m = p.len() - 1;
        let i = self.at.partition_point(|&x| x <= s).clamp(1, m) - 1;
        let t = (s - self.at[i]) / (self.at[i + 1] - self.at[i]).max(1e-300);
        let axis: P3 = std::array::from_fn(|k| p[i][k] + t * (p[i + 1][k] - p[i][k]));
        let dir = |k: usize| {
            let (x, y) = self.frame[k];
            add(scale(x, a.cos()), scale(y, a.sin()))
        };
        let tc = t.clamp(0.0, 1.0);
        let d = add(scale(dir(i), 1.0 - tc), scale(dir(i + 1), tc));
        let d = unit(d).unwrap_or(dir(i));
        add(axis, scale(d, radius))
    }
}

impl<'a> Unroll<'a> {
    /// The unrolling of a cylinder or cone carrier.
    pub fn of(surface: &'a Surface) -> Option<Unroll<'a>> {
        let kind = match *surface {
            Surface::Cylinder { radius, .. } => Kind::Cylinder { radius },
            Surface::Torus { major, minor, .. } => Kind::Torus { major, minor },
            Surface::Tube { ref path, radius } => {
                return Some(Unroll {
                    surface,
                    kind: Kind::Cylinder { radius },
                    theta0: 0.0,
                    seam: false,
                    ring: 0.0,
                    tube: Some(Frames::of(path)?),
                    profile: None,
                });
            }
            Surface::Extruded { ref profile, .. } => {
                let pr = Profile::of(profile)?;
                return Some(Unroll {
                    surface,
                    kind: Kind::Cylinder {
                        radius: pr.radius(),
                    },
                    theta0: 0.0,
                    seam: false,
                    ring: 0.0,
                    tube: None,
                    profile: Some(pr),
                });
            }
            Surface::Cone { half_angle, .. } => {
                let (sin, cos) = half_angle.sin_cos();
                if !(sin > 1e-6 && cos > 1e-6) {
                    return None;
                }
                Kind::Cone { sin, cos }
            }
            _ => return None,
        };
        Some(Unroll {
            surface,
            kind,
            theta0: 0.0,
            seam: false,
            ring: 0.0,
            tube: None,
            profile: None,
        })
    }

    /// Angle round the axis and height (axial from the apex for a cone).
    fn angle_height(&self, p: P3) -> (f64, f64) {
        if let Some(f) = &self.tube {
            return f.angle_length(p);
        }
        if let Some(pr) = &self.profile {
            let uv = self.surface.project_uv(p);
            return (pr.arc(uv[0]) / pr.radius(), uv[1]);
        }
        let uv = self.surface.project_uv(p);
        match self.kind {
            Kind::Band { v0, .. } => (uv[0], wrap_pm(uv[1] - v0)),
            _ => (uv[0], uv[1]),
        }
    }

    /// The chart point of angle `a` (from the chart's origin, unwrapped)
    /// and height `h`.
    fn chart(&self, a: f64, h: f64) -> P2 {
        match self.kind {
            Kind::Cylinder { radius } => [radius * a, h],
            Kind::Cone { sin, cos } => {
                let (t, phi) = (h / cos, a * sin);
                [t * phi.cos(), t * phi.sin()]
            }
            Kind::Torus { major, minor } => [major * a, minor * h],
            Kind::Band {
                radius,
                major,
                minor,
                v0,
            } => [
                radius * a,
                radius * (band_height(major, minor, v0 + h) - band_height(major, minor, v0)),
            ],
        }
    }

    /// The surface point at chart point `q`.
    pub fn lift(&self, q: P2) -> P3 {
        if self.tube.is_some() {
            return self.surface.closest(self.near(q)).0;
        }
        if let (Some(pr), Kind::Cylinder { radius }) = (&self.profile, self.kind) {
            let a = q[0] / radius + self.theta0;
            return self.surface.eval_uv([pr.param(a * radius), q[1]]);
        }
        match self.kind {
            Kind::Cylinder { radius } => self.surface.eval_uv([q[0] / radius + self.theta0, q[1]]),
            Kind::Cone { sin, cos } => {
                let t = (q[0] * q[0] + q[1] * q[1]).sqrt();
                // The chart's angles run from 0 up to a turn times `sin` along
                // a seam (past a half turn for a wide cone), either side of
                // 0 in a gap cut.
                let mut phi = q[1].atan2(q[0]);
                if self.seam && phi < 0.0 && phi + TAU <= TAU * sin * (1.0 + 1e-9) {
                    phi += TAU;
                }
                self.surface.eval_uv([phi / sin + self.theta0, t * cos])
            }
            Kind::Torus { major, minor } => self
                .surface
                .eval_uv([q[0] / major + self.theta0, q[1] / minor]),
            Kind::Band {
                radius,
                major,
                minor,
                v0,
            } => {
                let y = q[1] / radius + band_height(major, minor, v0);
                self.surface
                    .eval_uv([q[0] / radius + self.theta0, band_angle(major, minor, y)])
            }
        }
    }

    /// A point at chart point `q` near the face (on it but for a tube, whose
    /// lift projects this onto it).
    pub fn near(&self, q: P2) -> P3 {
        match (&self.tube, self.kind) {
            (Some(f), Kind::Cylinder { radius }) => {
                f.point(q[0] / radius + self.theta0, q[1], radius)
            }
            _ => self.lift(q),
        }
    }

    /// The chart point of a point of the face (once its chart is set up).
    pub fn to_chart(&self, p: P3) -> P2 {
        let (t, h) = self.angle_height(p);
        let h = match self.kind {
            Kind::Torus { .. } => h.rem_euclid(TAU),
            _ => h,
        };
        let a = if self.seam {
            (t - self.theta0).rem_euclid(TAU)
        } else {
            wrap_pm(t - self.theta0)
        };
        self.chart(a, h)
    }

    /// The largest size at chart point `q` that still puts three points on
    /// the circle round the axis there (a coarser mesh would join a seam's
    /// two sides).
    pub fn cap(&self, q: P2) -> f64 {
        // A profile that closes keeps three points round it; an open one
        // bounds nothing.
        if let Some(pr) = &self.profile {
            return if pr.closed {
                pr.len() / 3.0
            } else {
                f64::INFINITY
            };
        }
        match self.kind {
            Kind::Cylinder { radius } => TAU * radius / 3.0,
            Kind::Torus { minor, .. } => TAU * minor / 3.0,
            Kind::Band { radius, .. } => TAU * radius / 3.0,
            Kind::Cone { sin, .. } => {
                TAU * (q[0] * q[0] + q[1] * q[1]).sqrt().max(self.ring) * sin / 3.0
            }
        }
    }

    /// Chart length per length on the face at chart point `q`: 1 but on a
    /// band, whose lengths scale with the distance from the axis.
    pub fn shrink(&self, q: P2) -> f64 {
        match self.kind {
            Kind::Band {
                radius,
                major,
                minor,
                v0,
            } => {
                let y = q[1] / radius + band_height(major, minor, v0);
                radius / (major + minor * band_angle(major, minor, y).cos())
            }
            _ => 1.0,
        }
    }

    /// Whether `h` is the apex (a cone's point where the angle is undefined):
    /// within a millionth of the model's scale, as near as a file gives an
    /// apex (a cone's angle written to ten digits puts it that far off).
    fn is_apex(&self, h: f64, scale: f64) -> bool {
        matches!(self.kind, Kind::Cone { .. }) && h.abs() <= 1e-6 * scale
    }

    /// The chart of a full torus (no loops, no edges inside): the square
    /// of both angles, cut along a circle round the axis and one round the
    /// tube, whose points appear on both sides of the square and the point
    /// where they meet at its four corners.
    fn torus_domain(
        &mut self,
        major: f64,
        minor: f64,
        rings: &[Vec<u32>],
        inner: &[Vec<u32>],
        extra: &[u32],
        points: &[P3],
        size: &dyn Fn(P3) -> f64,
    ) -> Option<Domain2> {
        if !rings.is_empty() || !inner.is_empty() {
            return None;
        }
        self.theta0 = 0.0;
        self.seam = true;
        let mut d = Domain2::default();
        let finest = |along: &dyn Fn(f64) -> P3| {
            (0..32)
                .map(|k| size(along(TAU * k as f64 / 32.0)))
                .fold(f64::INFINITY, f64::min)
                .max(1e-300)
        };
        let round = |a: f64| self.lift(self.chart(a, 0.0));
        let across = |b: f64| self.lift(self.chart(0.0, b));
        let nu = ((TAU * (major + minor) / finest(&round)).ceil() as usize).max(3);
        let nv = ((TAU * minor / finest(&across)).ceil() as usize).max(3);
        let corner = d.add_own(self.lift(self.chart(0.0, 0.0)));
        let us: Vec<(f64, u32)> = (1..nu)
            .map(|k| {
                let a = TAU * k as f64 / nu as f64;
                (a, d.add_own(round(a)))
            })
            .collect();
        let vs: Vec<(f64, u32)> = (1..nv)
            .map(|k| {
                let b = TAU * k as f64 / nv as f64;
                (b, d.add_own(across(b)))
            })
            .collect();
        let c = Slot::Own(corner);
        let mut outline: Vec<(Slot, P2)> = vec![(c, self.chart(0.0, 0.0))];
        outline.extend(us.iter().map(|&(a, k)| (Slot::Own(k), self.chart(a, 0.0))));
        outline.push((c, self.chart(TAU, 0.0)));
        outline.extend(vs.iter().map(|&(b, k)| (Slot::Own(k), self.chart(TAU, b))));
        outline.push((c, self.chart(TAU, TAU)));
        outline.extend(
            us.iter()
                .rev()
                .map(|&(a, k)| (Slot::Own(k), self.chart(a, TAU))),
        );
        outline.push((c, self.chart(0.0, TAU)));
        outline.extend(
            vs.iter()
                .rev()
                .map(|&(b, k)| (Slot::Own(k), self.chart(0.0, b))),
        );
        d.add_loop(&outline);
        for &g in extra {
            d.add_point(Slot::Global(g), self.to_chart(points[g as usize]));
        }
        Some(d)
    }

    /// The chart of a face whose loops are `rings` (global ids, in loop
    /// order), whose inner edges are `inner` and whose other fixed points
    /// are `extra`, with seam points spaced by `size`; none when the face
    /// has no chart of this kind.
    pub(crate) fn domain(
        &mut self,
        points: &[P3],
        rings: &[Vec<u32>],
        inner: &[Vec<u32>],
        extra: &[u32],
        size: &dyn Fn(P3) -> f64,
    ) -> Option<Domain2> {
        if let Kind::Torus { major, minor } = self.kind {
            if rings.is_empty() {
                if minor > SLENDER_TORUS * major {
                    return None;
                }
                return self.torus_domain(major, minor, rings, inner, extra, points, size);
            }
            // A face with loops: a band round the axis, where it keeps to
            // less than half the tube of a ring torus.
            let vs: Vec<f64> = rings
                .iter()
                .flatten()
                .map(|&g| self.surface.project_uv(points[g as usize])[1])
                .collect();
            let (s, c) = vs
                .iter()
                .fold((0.0, 0.0), |(s, c), v| (s + v.sin(), c + v.cos()));
            let v0 = s.atan2(c);
            let spread = vs.iter().map(|v| wrap_pm(v - v0).abs()).fold(0.0, f64::max);
            if !(spread < 0.5 * PI && major > minor) {
                return None;
            }
            self.kind = Kind::Band {
                radius: major + minor * v0.cos(),
                major,
                minor,
                v0,
            };
        }
        let scale = rings
            .iter()
            .flatten()
            .map(|&g| {
                points[g as usize]
                    .iter()
                    .fold(0.0_f64, |m, x| m.max(x.abs()))
            })
            .fold(1e-300, f64::max);
        let ah: Vec<Vec<(f64, f64)>> = rings
            .iter()
            .map(|r| {
                r.iter()
                    .map(|&g| self.angle_height(points[g as usize]))
                    .collect()
            })
            .collect();
        // The winding of each loop round the axis.
        let winding = |r: &[(f64, f64)]| -> i64 {
            let mut sum = 0.0;
            let pts: Vec<f64> = r
                .iter()
                .filter(|x| !self.is_apex(x.1, scale))
                .map(|x| x.0)
                .collect();
            for k in 0..pts.len() {
                sum += wrap_pm(pts[(k + 1) % pts.len()] - pts[k]);
            }
            (sum / TAU).round() as i64
        };
        let windings: Vec<i64> = ah.iter().map(|r| winding(r)).collect();
        let winding_loops: Vec<usize> = (0..rings.len()).filter(|&i| windings[i] != 0).collect();
        let mut d = Domain2::default();
        if winding_loops.is_empty() {
            // No loop winds round: each loop's angles unwrapped along it (an
            // apex, whose angle is arbitrary, stepped over), the chart
            // centred on the widest. Where the face leaves a gap this is
            // the cut in it; where a seam edge closes the face round (its
            // loop runs up the seam, round, down it and back), the seam's
            // two sides land at either end.
            // A loop through the apex starts right after it: the step
            // across the apex, whose size nothing tells, is then the
            // closing one, never taken.
            let unwrapped: Vec<Vec<f64>> = ah
                .iter()
                .map(|a| {
                    let n = a.len();
                    let start = (0..n)
                        .find(|&k| self.is_apex(a[k].1, scale))
                        .map_or(0, |k| (k + 1) % n);
                    let mut out = vec![0.0; n];
                    let mut last: Option<(f64, f64)> = None;
                    for step in 0..n {
                        let k = (start + step) % n;
                        let (t, h) = a[k];
                        if self.is_apex(h, scale) {
                            out[k] = last.map_or(t, |l| l.0);
                            continue;
                        }
                        let u = last.map_or(t, |(u, raw)| u + wrap_pm(t - raw));
                        last = Some((u, t));
                        out[k] = u;
                    }
                    out
                })
                .collect();
            let range = |u: &[f64]| {
                u.iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                        (lo.min(x), hi.max(x))
                    })
            };
            let widest = (0..unwrapped.len())
                .max_by(|&i, &j| {
                    let w = |k: usize| {
                        let (lo, hi) = range(&unwrapped[k]);
                        hi - lo
                    };
                    w(i).total_cmp(&w(j))
                })
                .unwrap_or(0);
            let (lo, hi) = range(&unwrapped[widest]);
            if !(hi - lo <= TAU * (1.0 + 1e-9)) {
                return None;
            }
            self.theta0 = 0.5 * (lo + hi);
            for ((r, a), u) in rings.iter().zip(&ah).zip(&unwrapped) {
                // A loop into the turn of the widest one.
                let mean = u.iter().sum::<f64>() / u.len().max(1) as f64;
                let shift = TAU * ((self.theta0 - mean) / TAU).round();
                let ring: Vec<(Slot, P2)> = r
                    .iter()
                    .zip(a)
                    .zip(u)
                    .map(|((&g, &(_, h)), &x)| {
                        (Slot::Global(g), self.chart(x + shift - self.theta0, h))
                    })
                    .collect();
                d.add_loop(&ring);
            }
            for c in inner {
                let chain: Vec<(Slot, P2)> = c
                    .iter()
                    .map(|&g| {
                        let (t, h) = self.angle_height(points[g as usize]);
                        (Slot::Global(g), self.chart(wrap_pm(t - self.theta0), h))
                    })
                    .collect();
                d.add_chain(&chain);
            }
            for &g in extra {
                let (t, h) = self.angle_height(points[g as usize]);
                d.add_point(Slot::Global(g), self.chart(wrap_pm(t - self.theta0), h));
            }
            return Some(d);
        }
        // A face all the way round: one or two winding loops (or one and
        // the apex), holes that do not.
        let holes: Vec<usize> = (0..rings.len()).filter(|&i| windings[i] == 0).collect();
        let (l1, l2) = match winding_loops.as_slice() {
            [a, b] => (*a, Some(*b)),
            [a] => (*a, None),
            _ => return None,
        };
        let apex: Option<u32> = if l2.is_none() {
            // A cone down to its apex: the apex is a corner of the face.
            extra
                .iter()
                .chain(rings.iter().flatten())
                .copied()
                .find(|&g| self.is_apex(self.angle_height(points[g as usize]).1, scale))
        } else {
            None
        };
        // An apex that is no corner of the B-rep (the carrier's singular
        // point inside the face) is a point of the face's own.
        let own_apex: Option<P3> = match (l2, apex) {
            (None, None) => Some(self.surface.singular_point()?),
            _ => None,
        };
        // The seam starts at a point of the first winding loop and runs
        // straight in the chart to the nearest point in angle of the other
        // (or the apex). Of the starts whose seam crosses no edge of the
        // face, the one farthest from every other point of its edges (a
        // notch in the loop, a hole, an edge inside); failing that, the
        // one farthest in angle from the holes.
        let far_height = |theta: f64| -> (f64, f64) {
            match l2 {
                Some(l2) => {
                    let near = (0..rings[l2].len())
                        .min_by(|&i, &j| {
                            wrap_pm(ah[l2][i].0 - theta)
                                .abs()
                                .total_cmp(&wrap_pm(ah[l2][j].0 - theta).abs())
                        })
                        .unwrap_or(0);
                    (wrap_pm(ah[l2][near].0 - theta), ah[l2][near].1)
                }
                None => (
                    0.0,
                    apex.map_or(0.0, |g| self.angle_height(points[g as usize]).1),
                ),
            }
        };
        let mut edges: Vec<((f64, f64), (f64, f64))> = Vec::new();
        for a in &ah {
            for k in 0..a.len() {
                edges.push((a[k], a[(k + 1) % a.len()]));
            }
        }
        for c in inner {
            let at: Vec<(f64, f64)> = c
                .iter()
                .map(|&g| self.angle_height(points[g as usize]))
                .collect();
            edges.extend(at.windows(2).map(|w| (w[0], w[1])));
        }
        let n1 = rings[l1].len();
        let stride = n1.div_ceil(128).max(1);
        // A loop may run through the apex (a generator to it is an edge):
        // no seam starts there, where the angle is undefined.
        let clear = |k: usize| -> Option<f64> {
            if self.is_apex(ah[l1][k].1, scale) {
                return None;
            }
            let theta = ah[l1][k].0;
            let (fa, fh) = far_height(theta);
            let (a, b) = (self.chart(0.0, ah[l1][k].1), self.chart(fa, fh));
            let tol = 1e-9 * (dist2d(a, b) + scale);
            let mut clearance = f64::INFINITY;
            for &(p, q) in &edges {
                let ap = wrap_pm(p.0 - theta);
                let (x, y) = (
                    self.chart(ap, p.1),
                    self.chart(ap + wrap_pm(q.0 - p.0), q.1),
                );
                if crosses(a, b, x, y) {
                    return None;
                }
                if dist2d(x, a) > tol && dist2d(x, b) > tol {
                    clearance = clearance.min(seg_dist(x, a, b));
                }
            }
            Some(clearance)
        };
        let hole_angles: Vec<f64> = holes
            .iter()
            .flat_map(|&i| ah[i].iter().map(|x| x.0))
            .collect();
        let pick = (0..n1)
            .step_by(stride)
            .filter_map(|k| clear(k).map(|c| (k, c)))
            .max_by(|x, y| x.1.total_cmp(&y.1))
            .map(|x| x.0)
            .unwrap_or_else(|| {
                (0..n1)
                    .filter(|&k| !self.is_apex(ah[l1][k].1, scale))
                    .max_by(|&i, &j| {
                        let far = |k: usize| {
                            hole_angles
                                .iter()
                                .map(|&h| wrap_pm(ah[l1][k].0 - h).abs())
                                .fold(PI, f64::min)
                        };
                        far(i).total_cmp(&far(j))
                    })
                    .unwrap_or(0)
            });
        // A loop that runs through the apex (out along a generator, round,
        // back along it) cuts the face open itself: no seam, its chain
        // starts next to the apex and ends in it.
        let through: Option<usize> = match (l2, apex) {
            (None, Some(g)) => rings[l1].iter().position(|&x| x == g),
            _ => None,
        };
        let pick = match through {
            Some(i) => {
                let dir = if windings[l1] > 0 { 1 } else { -1 };
                (i as i64 + dir).rem_euclid(n1 as i64) as usize
            }
            None => pick,
        };
        self.theta0 = ah[l1][pick].0;
        self.seam = true;
        // A winding loop from `start`, turned to run up in angle, unwrapped
        // from `first`: its points and chart angles, the start again at the
        // end one turn on.
        let unwrapped = |li: usize, start: usize, first: f64| -> Vec<(u32, f64, f64)> {
            let (r, a) = (&rings[li], &ah[li]);
            let n = r.len();
            let dir: i64 = if windings[li] > 0 { 1 } else { -1 };
            let mut out = Vec::with_capacity(n + 1);
            let mut ang = first;
            let mut k = start;
            // The angle goes on from the last point off the apex, whose own
            // angle is arbitrary.
            let mut last = start;
            for step in 0..=n {
                if step > 0 && !self.is_apex(a[k].1, scale) {
                    ang += wrap_pm(a[k].0 - a[last].0);
                    last = k;
                }
                out.push((r[k], ang, a[k].1));
                k = (k as i64 + dir).rem_euclid(n as i64) as usize;
            }
            out
        };
        let chain1 = unwrapped(l1, pick, 0.0);
        if (chain1[chain1.len() - 1].1 - TAU).abs() > 1e-6 {
            return None;
        }
        // The far end of the seam: on the second loop the point nearest in
        // angle, or the apex.
        let (chain2, far): (Vec<(u32, f64, f64)>, (Slot, f64, f64)) = match l2 {
            Some(l2) => {
                let near = (0..rings[l2].len())
                    .min_by(|&i, &j| {
                        wrap_pm(ah[l2][i].0 - self.theta0)
                            .abs()
                            .total_cmp(&wrap_pm(ah[l2][j].0 - self.theta0).abs())
                    })
                    .unwrap_or(0);
                let first = wrap_pm(ah[l2][near].0 - self.theta0);
                let c = unwrapped(l2, near, first);
                if (c[c.len() - 1].1 - first - TAU).abs() > 1e-6 {
                    return None;
                }
                let f = (Slot::Global(c[0].0), c[0].1, c[0].2);
                (c, f)
            }
            None => match (apex, own_apex) {
                (Some(g), _) => (
                    Vec::new(),
                    (
                        Slot::Global(g),
                        0.0,
                        self.angle_height(points[g as usize]).1,
                    ),
                ),
                (None, Some(p)) => {
                    let k = d.add_own(p);
                    (Vec::new(), (Slot::Own(k), 0.0, 0.0))
                }
                (None, None) => return None,
            },
        };
        // Holes into the window of the chart.
        let hole_chart = |i: usize| -> Option<Vec<(Slot, P2)>> {
            let mut ang: Vec<f64> = Vec::with_capacity(rings[i].len());
            let mut cur = (ah[i][0].0 - self.theta0).rem_euclid(TAU);
            ang.push(cur);
            for k in 1..rings[i].len() {
                cur += wrap_pm(ah[i][k].0 - ah[i][k - 1].0);
                ang.push(cur);
            }
            let mean = ang.iter().sum::<f64>() / ang.len() as f64;
            let shift = TAU * ((PI - mean) / TAU).round();
            let ring: Vec<(Slot, P2)> = rings[i]
                .iter()
                .zip(&ang)
                .zip(&ah[i])
                .map(|((&g, &a), &(_, h))| (Slot::Global(g), self.chart(a + shift, h)))
                .collect();
            let inside = ang.iter().all(|&a| a + shift > 0.0 && a + shift < TAU);
            inside.then_some(ring)
        };
        // The seam: the face's own points along the chart line between its
        // ends, twice (one turn apart).
        let (a_chart, b_chart) = (self.chart(0.0, chain1[0].2), self.chart(far.1, far.2));
        let len = ((b_chart[0] - a_chart[0]).powi(2) + (b_chart[1] - a_chart[1]).powi(2)).sqrt();
        let mid = self.lift([
            (a_chart[0] + b_chart[0]) / 2.0,
            (a_chart[1] + b_chart[1]) / 2.0,
        ]);
        let n = if through.is_some() {
            1
        } else {
            ((len / size(mid).max(1e-300)).ceil() as usize).max(1)
        };
        let seam: Vec<(f64, f64)> = (1..n)
            .map(|k| {
                let t = k as f64 / n as f64;
                (far.1 * t, chain1[0].2 + t * (far.2 - chain1[0].2))
            })
            .collect();
        let own: Vec<u32> = seam
            .iter()
            .map(|&(a, h)| d.add_own(self.lift(self.chart(a, h))))
            .collect();
        // The outline: the first loop one turn up, the seam copy one turn
        // on, the second loop back down (or the apex), the seam back.
        let mut outline: Vec<(Slot, P2)> = chain1
            .iter()
            .map(|&(g, a, h)| (Slot::Global(g), self.chart(a, h)))
            .collect();
        if through.is_some() {
            // Its start again one turn on is the start: the loop closes.
            outline.pop();
        } else {
            for (k, &(a, h)) in seam.iter().enumerate() {
                outline.push((Slot::Own(own[k]), self.chart(a + TAU, h)));
            }
            if chain2.is_empty() {
                outline.push((far.0, self.chart(far.1, far.2)));
            } else {
                for &(g, a, h) in chain2.iter().rev() {
                    outline.push((Slot::Global(g), self.chart(a, h)));
                }
            }
            for (k, &(a, h)) in seam.iter().enumerate().rev() {
                outline.push((Slot::Own(own[k]), self.chart(a, h)));
            }
        }
        d.add_loop(&outline);
        // Round an apex, a ring at the height of the seam's last point, so
        // its fan has three neighbours on the surface, not the seam point
        // twice over one other.
        let mut ring = 0.0;
        if chain2.is_empty() {
            let h = seam.last().map_or(0.5 * chain1[0].2, |x| x.1);
            if let Kind::Cone { cos, .. } = self.kind {
                ring = h / cos;
            }
            for k in 1..3 {
                let q = self.chart(TAU * k as f64 / 3.0, h);
                let own = d.add_own(self.lift(q));
                d.add_point(Slot::Own(own), q);
            }
        }
        for &i in &holes {
            d.add_loop(&hole_chart(i)?);
        }
        for c in inner {
            let chain: Vec<(Slot, P2)> = c
                .iter()
                .map(|&g| {
                    let (t, h) = self.angle_height(points[g as usize]);
                    (
                        Slot::Global(g),
                        self.chart((t - self.theta0).rem_euclid(TAU), h),
                    )
                })
                .collect();
            d.add_chain(&chain);
        }
        for &g in extra {
            if Some(g) == apex {
                continue;
            }
            let (t, h) = self.angle_height(points[g as usize]);
            d.add_point(
                Slot::Global(g),
                self.chart((t - self.theta0).rem_euclid(TAU), h),
            );
        }
        self.ring = ring;
        Some(d)
    }
}

/// `v` turned by the least rotation that takes unit `a` to unit `b`.
fn turn(v: P3, a: P3, b: P3) -> P3 {
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

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: P3, b: P3) -> P3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: P3, s: f64) -> P3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(a: P3) -> Option<P3> {
    let l = dot(a, a).sqrt();
    (l > 0.0 && l.is_finite()).then(|| a.map(|x| x / l))
}

/// A vector square to `n`.
fn perp(n: P3) -> P3 {
    let k = (0..3)
        .min_by(|&i, &j| n[i].abs().total_cmp(&n[j].abs()))
        .unwrap_or(0);
    let mut e = [0.0; 3];
    e[k] = 1.0;
    cross(n, e)
}

fn dist2d(a: P2, b: P2) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// The distance of `p` from the segment `a b`.
fn seg_dist(p: P2, a: P2, b: P2) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let l2 = d[0] * d[0] + d[1] * d[1];
    let t = if l2 > 0.0 {
        (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    dist2d(p, [a[0] + t * d[0], a[1] + t * d[1]])
}

/// Whether the segments `a b` and `x y` cross (at a point inside both).
fn crosses(a: P2, b: P2, x: P2, y: P2) -> bool {
    let side = |p: P2, q: P2, r: P2| (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]);
    let (s1, s2) = (side(a, b, x), side(a, b, y));
    let (s3, s4) = (side(x, y, a), side(x, y, b));
    s1 * s2 < 0.0 && s3 * s4 < 0.0
}

/// An angle wrapped into `(-pi, pi]`.
fn wrap_pm(a: f64) -> f64 {
    let x = a.rem_euclid(TAU);
    if x > PI {
        x - TAU
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The conformal height of a torus band inverts, and grows by `minor /
    /// rho` per tube angle, through the inside of the ring too.
    #[test]
    fn a_torus_band_unrolls_conformally() {
        let (major, minor) = (0.575, 0.125);
        for v in [-3.0, -1.2, 0.0, 0.7, 2.5, 3.1, 3.3, 4.0] {
            let y = band_height(major, minor, v);
            let back = band_angle(major, minor, y);
            assert!((wrap_pm(back - v)).abs() < 1e-12, "{v} {back}");
            let h = 1e-6;
            let slope =
                (band_height(major, minor, v + h) - band_height(major, minor, v - h)) / (2.0 * h);
            let want = minor / (major + minor * v.cos());
            assert!((slope - want).abs() < 1e-6 * want, "{v}: {slope} {want}");
        }
    }
}
