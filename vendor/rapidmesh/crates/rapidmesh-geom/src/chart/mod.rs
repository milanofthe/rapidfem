//! Charts: the plane a face is meshed in and the way back onto it.
//!
//! A face's bounds are first laid out in its surface's own parameters (see
//! [`uv`]: unwrapped round the periods, split at the poles, the rims that
//! wind round joined by a seam), then mapped into the plane by a chart that
//! keeps angles, so an isotropic mesh in it, its sizes scaled by the
//! chart's [`Chart::shrink`] at each place, lifts to a mesh whose triangles
//! keep their shapes on the face:
//!
//! - a plane: its frame;
//! - a surface of revolution (cylinder, cone, sphere, torus, revolved
//!   profile): the strip or the polar chart of its conformal coordinates
//!   (see [`revolution`]); a sphere is turned so that its pole lies in the
//!   face, and its chart is the stereographic projection, which goes round
//!   the pole whole and is taken as it lies;
//! - an extrusion: arc length along its profile and height; a tube round an
//!   open path: angle round the path and arc length along it;
//! - a B-spline: its parameters made isotropic on average (see [`param`]).

mod domain;
mod param;
mod revolution;
mod sweep;
pub mod uv;

pub use domain::{Domain, Slot};

use crate::Surface;
use param::Param;
use rapidmesh_exact::vector::{add, centroid, dist, dot, len, sub, unit, wrap_near, Frame, V2, V3};
use revolution::{polar_point, Form, Revolution};
use std::f64::consts::TAU;
use sweep::{Frames, Profile};
use uv::{Join, Mark};

/// The pole of a sphere's chart stays at least this far (in degrees seen
/// from the centre) from every sample of the face.
const POLE_CLEARANCE_DEG: f64 = 15.0;

/// The chart of one face on its carrier.
pub struct Chart {
    /// The surface as charted: a sphere turned so its pole lies in the
    /// face.
    surface: Surface,
    kind: Kind,
    periods: [Option<f64>; 2],
    poles: Vec<(usize, f64, V3)>,
    /// The parameters of samples spread over the face.
    samples: Vec<V2>,
    /// The start of a B-spline's or a profile's domain.
    lo: V2,
    /// The angle of the chart's origin, and whether the face's outline
    /// spans a whole turn of it (cut along a seam).
    u0: f64,
    seam: bool,
    /// The middle of the outline in the parameters: points inside are put
    /// in its turn.
    mid: V2,
    /// A strip's distance from the axis whose lengths it keeps, and the
    /// conformal height of its origin.
    r0: f64,
    s0: f64,
    /// The polar radius of the ring round an apex (0 for none): inside it
    /// the ring, not the size, keeps three points round the axis.
    ring: f64,
}

enum Kind {
    Plane,
    Revolution(Revolution),
    Extruded(Profile),
    Tube { frames: Frames, radius: f64 },
    Param(Box<Param>),
}

impl Chart {
    /// The chart of a face on `surface`, its loops through `rings` and
    /// `samples` spread over it (a sphere's pole is chosen clear of them);
    /// none for a carrier without one, a sphere face without loops or
    /// without room for the pole.
    pub fn of(surface: &Surface, samples: &[V3], rings: &[Vec<V3>]) -> Option<Chart> {
        let (surface, kind) = match surface {
            Surface::Plane(_) => (surface.clone(), Kind::Plane),
            Surface::Sphere { frame, radius } => {
                let axis = sphere_axis(frame.o, samples, rings)?;
                let turned = Surface::Sphere {
                    frame: Frame::new(frame.o, axis, None)?,
                    radius: *radius,
                };
                let r = Revolution::of(&turned)?;
                (turned, Kind::Revolution(r))
            }
            Surface::Cylinder { .. }
            | Surface::Cone { .. }
            | Surface::Torus { .. }
            | Surface::Revolved { .. } => {
                (surface.clone(), Kind::Revolution(Revolution::of(surface)?))
            }
            Surface::Extruded { profile, .. } => {
                (surface.clone(), Kind::Extruded(Profile::of(profile)?))
            }
            Surface::Tube { path, radius } => (
                surface.clone(),
                Kind::Tube {
                    frames: Frames::of(path)?,
                    radius: *radius,
                },
            ),
            Surface::Nurbs(_) => (
                surface.clone(),
                Kind::Param(Box::new(Param::of(surface, samples)?)),
            ),
            _ => return None,
        };
        Some(Chart::with(surface, kind, samples))
    }

    fn with(surface: Surface, kind: Kind, samples: &[V3]) -> Chart {
        let (periods, poles) = match &kind {
            Kind::Tube { .. } => ([Some(TAU), None], Vec::new()),
            _ => (surface.periods(), surface.poles()),
        };
        let lo = match &surface {
            Surface::Nurbs(n) => n.domain().0,
            Surface::Extruded { profile, .. } => [profile.domain().0, 0.0],
            _ => [0.0; 2],
        };
        let mut c = Chart {
            surface,
            kind,
            periods,
            poles,
            samples: Vec::new(),
            lo,
            u0: 0.0,
            seam: false,
            mid: [0.0; 2],
            r0: 1.0,
            s0: 0.0,
            ring: 0.0,
        };
        c.samples = samples.iter().map(|&p| c.params(p)).collect();
        c
    }

    /// Whether the chart is the face's plane (its points exact).
    pub fn is_plane(&self) -> bool {
        matches!(self.kind, Kind::Plane)
    }

    /// Whether the chart takes the face as it lies, without unwrapping:
    /// a plane, a sphere seen from its pole.
    fn direct(&self) -> bool {
        match &self.kind {
            Kind::Plane => true,
            Kind::Revolution(r) => r.sphere(),
            _ => false,
        }
    }

    /// The parameters of a point of the face.
    fn params(&self, p: V3) -> V2 {
        match &self.kind {
            Kind::Tube { frames, .. } => {
                let (a, s) = frames.angle_length(p);
                [a, s]
            }
            _ => self.surface.param(p),
        }
    }

    /// The point at parameters `uv` (any turn of a closed one).
    fn point(&self, uv: V2) -> V3 {
        match &self.kind {
            Kind::Tube { frames, radius } => {
                self.surface.closest(frames.point(uv[0], uv[1], *radius)).0
            }
            Kind::Extruded(_) | Kind::Param(_) => {
                let mut uv = uv;
                for k in 0..2 {
                    if let Some(p) = self.periods[k] {
                        uv[k] = self.lo[k] + (uv[k] - self.lo[k]).rem_euclid(p);
                    }
                }
                self.surface.eval(uv)
            }
            _ => self.surface.eval(uv),
        }
    }

    /// The chart point at parameters `uv`.
    fn chart_point(&self, uv: V2) -> V2 {
        let [u, v] = uv;
        match &self.kind {
            Kind::Plane => uv,
            Kind::Revolution(r) if r.polar() => {
                polar_point(r.radius(r.t(v)), r.k() * (u - self.u0))
            }
            Kind::Revolution(r) => [
                self.r0 * (u - self.u0),
                self.r0 * (r.sigma(r.t(v)) - self.s0),
            ],
            Kind::Extruded(pr) => [pr.arc(u) - pr.arc(self.u0), v],
            Kind::Tube { radius, .. } => [radius * (u - self.u0), v],
            Kind::Param(pm) => pm.to_chart(uv),
        }
    }

    /// The parameters at chart point `q`.
    fn params_at(&self, q: V2) -> V2 {
        match &self.kind {
            Kind::Plane => q,
            Kind::Revolution(r) if r.polar() => {
                let k = r.k();
                // The chart's angles run from 0 up to a turn times `k`
                // along a seam (past a half turn for a wide cone), either
                // side of 0 in a gap cut.
                let mut phi = q[1].atan2(q[0]);
                if self.seam && phi < 0.0 && phi + TAU <= TAU * k * (1.0 + 1e-9) {
                    phi += TAU;
                }
                [phi / k + self.u0, r.v(r.radius_inv(len(q)))]
            }
            Kind::Revolution(r) => [
                q[0] / self.r0 + self.u0,
                r.v(r.sigma_inv(q[1] / self.r0 + self.s0)),
            ],
            Kind::Extruded(pr) => [pr.param(q[0] + pr.arc(self.u0)), q[1]],
            Kind::Tube { radius, .. } => [q[0] / radius + self.u0, q[1]],
            Kind::Param(pm) => pm.params(q, 0.0),
        }
    }

    /// The surface point at chart point `q`.
    pub fn lift(&self, q: V2) -> V3 {
        self.point(self.params_at(q))
    }

    /// A point at chart point `q` near the face (on it but for a tube, whose
    /// lift projects this onto it).
    pub fn near(&self, q: V2) -> V3 {
        match &self.kind {
            Kind::Tube { frames, radius } => {
                let uv = self.params_at(q);
                frames.point(uv[0], uv[1], *radius)
            }
            _ => self.lift(q),
        }
    }

    /// The parameters of a point of the face in the turn of its outline.
    fn inside(&self, p: V3) -> V2 {
        let mut uv = self.params(p);
        for k in 0..2 {
            if let Some(per) = self.periods[k] {
                uv[k] = if k == 0 && self.seam {
                    self.u0 + (uv[k] - self.u0).rem_euclid(per)
                } else {
                    wrap_near(uv[k], self.mid[k], per)
                };
            }
        }
        uv
    }

    /// The chart point of a point of the face (once its chart is set up).
    pub fn to_chart(&self, p: V3) -> V2 {
        self.chart_point(self.inside(p))
    }

    /// The largest size at chart point `q` that still puts three points on
    /// a circle the face closes round (a coarser mesh would join a seam's
    /// two sides).
    pub fn cap(&self, q: V2) -> f64 {
        match &self.kind {
            Kind::Plane => f64::INFINITY,
            Kind::Revolution(r) if r.sphere() => f64::INFINITY,
            Kind::Revolution(r) if r.polar() => TAU * len(q).max(self.ring) * r.k() / 3.0,
            Kind::Revolution(r) => {
                let round = TAU * self.r0;
                let tube = match self.periods[1] {
                    Some(per) => {
                        let v = self.params_at(q)[1];
                        self.r0 * (r.sigma(r.t(v + per)) - r.sigma(r.t(v))).abs()
                    }
                    None => f64::INFINITY,
                };
                round.min(tube) / 3.0
            }
            Kind::Extruded(pr) if pr.closed => pr.len() / 3.0,
            Kind::Extruded(_) => f64::INFINITY,
            Kind::Tube { radius, .. } => TAU * radius / 3.0,
            Kind::Param(pm) => pm.period_length() / 3.0,
        }
    }

    /// Chart length per length on the face at chart point `q`.
    pub fn shrink(&self, q: V2) -> f64 {
        match &self.kind {
            Kind::Revolution(r) if r.polar() => r.polar_shrink(len(q)),
            Kind::Revolution(r) => self.r0 / r.rho(r.t(self.params_at(q)[1])),
            Kind::Param(pm) => pm.shrink(&self.surface, self.params_at(q)),
            _ => 1.0,
        }
    }

    /// The chart of a face whose loops are `rings` (global ids, in loop
    /// order), whose inner edges are `inner` and whose other fixed points
    /// are `extra`, with seam points spaced by `size`; none when the face
    /// has no chart of this kind.
    pub fn domain(
        &mut self,
        points: &[V3],
        rings: &[Vec<u32>],
        inner: &[Vec<u32>],
        extra: &[u32],
        size: &dyn Fn(V3) -> f64,
        grading: f64,
    ) -> Option<Domain> {
        if self.direct() {
            return Some(self.direct_domain(points, rings, inner, extra));
        }
        if rings.is_empty() {
            return self.whole_domain(extra, points, size);
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
        let marks: Vec<Vec<Mark<Slot>>> = rings
            .iter()
            .map(|r| {
                r.iter()
                    .map(|&g| {
                        let p = points[g as usize];
                        Mark {
                            tag: Slot::Global(g),
                            p,
                            uv: self.params(p),
                        }
                    })
                    .collect()
            })
            .collect();
        // A strip keeps the lengths at the middle of the face's samples.
        if let Kind::Revolution(r) = &self.kind {
            if !r.polar() {
                let from: Vec<V2> = if self.samples.is_empty() {
                    marks.iter().flatten().map(|m| m.uv).collect()
                } else {
                    self.samples.clone()
                };
                let v = mean_of(&from, 1, self.periods[1]);
                (self.r0, self.s0) = match r.form {
                    Form::StripLine { rho, .. } => (rho, 0.0),
                    _ => (r.rho(r.t(v)), r.sigma(r.t(v))),
                };
            }
        }
        // The spacing of each loop point from its neighbours in the chart,
        // which a seam from it starts with.
        let mut spacing: Vec<(V3, f64)> = Vec::new();
        for r in &marks {
            let n = r.len();
            for i in 0..n {
                let d = |j: usize| dist(self.chart_point(r[i].uv), self.chart_point(r[j].uv));
                spacing.push((r[i].p, d((i + 1) % n).min(d((i + n - 1) % n))));
            }
        }
        let fit = 1e-6 * scale;
        let end = |p: V3| {
            spacing
                .iter()
                .filter(|s| dist(s.0, p) <= fit)
                .map(|s| s.1)
                .fold(f64::INFINITY, f64::min)
        };
        let seam = |a: V2, b: V2| -> Vec<(V2, V3)> {
            let at = |t: f64| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
            let l = dist(self.chart_point(a), self.chart_point(b));
            let ends = (end(self.point(a)), end(self.point(b)));
            // Its size: the face's, graded from the spacing of the loops at
            // its ends. The seam is fixed in the face's mesh: a rim sampled
            // far finer than the face would meet one of its segments in
            // slivers the refinement inside does not mend.
            graded(l, ends.0.min(ends.1), |t| {
                let q = self.chart_point(at(t));
                (size(self.lift(q)) * self.shrink(q))
                    .min(ends.0 + grading * t * l)
                    .min(ends.1 + grading * (1.0 - t) * l)
            })
            .into_iter()
            .map(|t| (at(t), self.point(at(t))))
            .collect()
        };
        // The edges inside the face, unwrapped along each.
        let mut avoid: Vec<Vec<(V2, V3)>> = Vec::with_capacity(inner.len());
        for c in inner {
            let mut chain: Vec<(V2, V3)> = Vec::with_capacity(c.len());
            for &g in c {
                let p = points[g as usize];
                let mut uv = self.params(p);
                if let Some(&(q, _)) = chain.last() {
                    for k in 0..2 {
                        if let Some(per) = self.periods[k] {
                            uv[k] = wrap_near(uv[k], q[k], per);
                        }
                    }
                }
                chain.push((uv, p));
            }
            avoid.push(chain);
        }
        let mut own: Vec<V3> = Vec::new();
        let mut add = |p: V3| {
            own.push(p);
            Slot::Own((own.len() - 1) as u32)
        };
        // The face lies on the side of a rim its samples do.
        let side = |k: usize, _: i64, ring: &[Mark<Slot>]| -> f64 {
            let j = 1 - k;
            let m = uv::mean(ring, j);
            let d: f64 = self
                .samples
                .iter()
                .map(|s| match self.periods[j] {
                    Some(per) => wrap_near(s[j], m, per) - m,
                    None => s[j] - m,
                })
                .sum();
            if d < 0.0 {
                -1.0
            } else {
                1.0
            }
        };
        let b = uv::bounds(
            self.periods,
            &self.poles,
            marks,
            fit,
            &mut Join {
                seam: &seam,
                own: &mut add,
                side: &side,
                pole_step: TAU / 12.0,
                avoid: &avoid,
            },
        )
        .ok()?;
        // The chart's origin: where a seam cuts the outline, else the
        // middle of its angles.
        let outer = &b.rings[b.outer];
        let (lo, hi) = outer
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), m| {
                (lo.min(m.uv[0]), hi.max(m.uv[0]))
            });
        self.mid = [0.5 * (lo + hi), uv::mean(outer, 1)];
        if let Some(per) = self.periods[0] {
            self.seam = hi - lo >= per * (1.0 - 1e-9);
            self.u0 = if self.seam { lo } else { self.mid[0] };
        }
        let mut d = Domain::default();
        d.own = own;
        for ring in &b.rings {
            let r: Vec<(Slot, V2)> = ring
                .iter()
                .map(|m| (m.tag, self.chart_point(m.uv)))
                .collect();
            d.add_loop(&r);
        }
        for c in inner {
            let chain: Vec<(Slot, V2)> = c
                .iter()
                .map(|&g| (Slot::Global(g), self.to_chart(points[g as usize])))
                .collect();
            d.add_chain(&chain);
        }
        for &g in extra {
            d.add_point(Slot::Global(g), self.to_chart(points[g as usize]));
        }
        // Round an apex, a ring at the nearest point's distance, so its fan
        // has three neighbours on the surface, not a seam point twice over
        // one other.
        if let Kind::Revolution(r) = &self.kind {
            let tiny = 1e-9 * scale;
            if r.polar() && d.pts.iter().any(|q| len(*q) <= tiny) {
                let radius = d
                    .pts
                    .iter()
                    .map(|q| len(*q))
                    .filter(|&l| l > tiny)
                    .fold(f64::INFINITY, f64::min);
                if radius.is_finite() {
                    let k = r.k();
                    for i in 1..3 {
                        let q = polar_point(radius, TAU * k * i as f64 / 3.0);
                        let p = self.lift(q);
                        let own = d.add_own(p);
                        d.add_point(Slot::Own(own), q);
                    }
                    self.ring = radius;
                }
            }
        }
        Some(d)
    }

    /// The chart of a face taken as it lies (a plane, a sphere seen from
    /// its pole): every point where the chart puts it.
    fn direct_domain(
        &self,
        points: &[V3],
        rings: &[Vec<u32>],
        inner: &[Vec<u32>],
        extra: &[u32],
    ) -> Domain {
        let mut d = Domain::default();
        let at = |g: u32| (Slot::Global(g), self.to_chart(points[g as usize]));
        for r in rings {
            d.add_loop(&r.iter().map(|&g| at(g)).collect::<Vec<_>>());
        }
        for c in inner {
            d.add_chain(&c.iter().map(|&g| at(g)).collect::<Vec<_>>());
        }
        for &g in extra {
            let (slot, q) = at(g);
            d.add_point(slot, q);
        }
        d
    }

    /// The chart of a whole surface closed both ways (a torus, a closed
    /// profile turned round, without loops or edges inside): the window of
    /// a turn round the axis and one of the meridian, cut along a circle
    /// round the axis and the meridian, whose points appear on both sides
    /// of it and the point where they meet at its four corners.
    fn whole_domain(
        &mut self,
        extra: &[u32],
        points: &[V3],
        size: &dyn Fn(V3) -> f64,
    ) -> Option<Domain> {
        let Kind::Revolution(r) = &self.kind else {
            return None;
        };
        let (Some(pu), Some(pv)) = (self.periods[0], self.periods[1]) else {
            return None;
        };
        let v0 = r.v(match &r.form {
            Form::StripTable { t, .. } => t[0],
            _ => 0.0,
        });
        (self.r0, self.s0) = (r.rho(r.t(v0)), r.sigma(r.t(v0)));
        self.u0 = 0.0;
        self.seam = true;
        self.mid = [0.5 * pu, v0 + 0.5 * pv];
        let mut d = Domain::default();
        let at = |u: f64, v: f64| self.chart_point([u, v0 + v]);
        let length = |f: &dyn Fn(f64) -> V2, span: f64| {
            (0..64)
                .map(|k| dist(f(span * k as f64 / 64.0), f(span * (k + 1) as f64 / 64.0)))
                .sum::<f64>()
        };
        let finest = |f: &dyn Fn(f64) -> V2, span: f64| {
            (0..32)
                .map(|k| {
                    let q = f(span * k as f64 / 32.0);
                    size(self.lift(q)) * self.shrink(q)
                })
                .fold(f64::INFINITY, f64::min)
                .max(1e-300)
        };
        let round = |a: f64| at(a, 0.0);
        let across = |b: f64| at(0.0, b);
        let nu = ((length(&round, pu) / finest(&round, pu)).ceil() as usize).max(3);
        let nv = ((length(&across, pv) / finest(&across, pv)).ceil() as usize).max(3);
        let corner = d.add_own(self.lift(at(0.0, 0.0)));
        let us: Vec<(f64, u32)> = (1..nu)
            .map(|k| {
                let a = pu * k as f64 / nu as f64;
                (a, d.add_own(self.lift(round(a))))
            })
            .collect();
        let vs: Vec<(f64, u32)> = (1..nv)
            .map(|k| {
                let b = pv * k as f64 / nv as f64;
                (b, d.add_own(self.lift(across(b))))
            })
            .collect();
        let c = Slot::Own(corner);
        let mut outline: Vec<(Slot, V2)> = vec![(c, at(0.0, 0.0))];
        outline.extend(us.iter().map(|&(a, k)| (Slot::Own(k), at(a, 0.0))));
        outline.push((c, at(pu, 0.0)));
        outline.extend(vs.iter().map(|&(b, k)| (Slot::Own(k), at(pu, b))));
        outline.push((c, at(pu, pv)));
        outline.extend(us.iter().rev().map(|&(a, k)| (Slot::Own(k), at(a, pv))));
        outline.push((c, at(0.0, pv)));
        outline.extend(vs.iter().rev().map(|&(b, k)| (Slot::Own(k), at(0.0, b))));
        d.add_loop(&outline);
        for &g in extra {
            d.add_point(Slot::Global(g), self.to_chart(points[g as usize]));
        }
        Some(d)
    }

    /// The charts of a whole sphere (no loops, no edges inside): the two
    /// caps either side of its equator, each seen from the pole of the
    /// other. They share the equator's points, spaced by `size`: the first
    /// own points of both domains. The fixed points `extra` go to the cap
    /// they lie on; the equator keeps as far from them as it can (square
    /// to the direction of one of them or an axis of the sphere's frame,
    /// the one that leaves the nearest of them highest above it), so none
    /// comes to lie beside its points.
    pub fn sphere_caps(
        surface: &Surface,
        points: &[V3],
        extra: &[u32],
        size: &dyn Fn(V3) -> f64,
    ) -> Option<[(Chart, Domain); 2]> {
        let Surface::Sphere { frame, radius } = surface else {
            return None;
        };
        let dirs: Vec<V3> = extra
            .iter()
            .filter_map(|&g| unit(sub(points[g as usize], frame.o)))
            .collect();
        let height = |a: V3| dirs.iter().map(|d| dot(*d, a).abs()).fold(1.0, f64::min);
        // Ties go to the last, the frame's own axis.
        let axis = dirs
            .iter()
            .copied()
            .chain([frame.x, frame.y, frame.z])
            .max_by(|a, b| height(*a).total_cmp(&height(*b)))?;
        let frame = &if axis == frame.z {
            *frame
        } else {
            Frame::new(frame.o, axis, None)?
        };
        let round = |a: f64| frame.at(radius * a.cos(), radius * a.sin(), 0.0);
        let finest = (0..32)
            .map(|k| size(round(TAU * k as f64 / 32.0)))
            .fold(f64::INFINITY, f64::min)
            .max(1e-300);
        let n = ((TAU * radius / finest).ceil() as usize).max(3);
        let equator: Vec<V3> = (0..n).map(|k| round(TAU * k as f64 / n as f64)).collect();
        let cap = |up: f64| -> Option<(Chart, Domain)> {
            let axis = frame.z.map(|x| up * x);
            let turned = Surface::Sphere {
                frame: Frame::new(frame.o, axis, None)?,
                radius: *radius,
            };
            let kind = Kind::Revolution(Revolution::of(&turned)?);
            let c = Chart::with(turned, kind, &[]);
            let mut d = Domain::default();
            let ring: Vec<(Slot, V2)> = equator
                .iter()
                .map(|&p| (Slot::Own(d.add_own(p)), c.to_chart(p)))
                .collect();
            d.add_loop(&ring);
            for &g in extra {
                let p = points[g as usize];
                let h = dot(sub(p, frame.o), axis);
                if h > 0.0 || (h == 0.0 && up > 0.0) {
                    d.add_point(Slot::Global(g), c.to_chart(p));
                }
            }
            Some((c, d))
        };
        Some([cap(1.0)?, cap(-1.0)?])
    }
}

/// The mean of coordinate `k` of `pts`, round its period where it has one.
fn mean_of(pts: &[V2], k: usize, period: Option<f64>) -> f64 {
    match period {
        Some(per) => {
            let w = TAU / per;
            let (s, c) = pts.iter().fold((0.0, 0.0), |(s, c), q| {
                (s + (w * q[k]).sin(), c + (w * q[k]).cos())
            });
            s.atan2(c) / w
        }
        None => pts.iter().map(|q| q[k]).sum::<f64>() / pts.len().max(1) as f64,
    }
}

/// The places, as shares of a line of length `len` strictly between its
/// ends, of samples spaced by `h` at each share: at equal steps of the
/// integral of `1 / h`, read on a grid fine enough for the `finest` size.
fn graded(len: f64, finest: f64, h: impl Fn(f64) -> f64) -> Vec<f64> {
    if !(len > 0.0) {
        return Vec::new();
    }
    let cells = (4.0 * len / finest.max(1e-6 * len))
        .ceil()
        .clamp(256.0, 65536.0) as usize;
    let mut cum = vec![0.0; cells + 1];
    for i in 0..cells {
        let t = (i as f64 + 0.5) / cells as f64;
        cum[i + 1] = cum[i] + len / cells as f64 / h(t).max(1e-300);
    }
    let total = cum[cells];
    let n = (total.round() as usize).max(1);
    (1..n)
        .map(|k| {
            let want = total * k as f64 / n as f64;
            let i = cum.partition_point(|&c| c < want).clamp(1, cells) - 1;
            let f = (want - cum[i]) / (cum[i + 1] - cum[i]).max(1e-300);
            (i as f64 + f.clamp(0.0, 1.0)) / cells as f64
        })
        .collect()
}

/// The axis a sphere face is charted about: from its centre away from the
/// pole the chart projects from, which lies off the face, as far from it as
/// the candidates go (opposite the face's mean direction, or in a hole of
/// it). None for a face without loops or without room for the pole.
fn sphere_axis(center: V3, samples: &[V3], rings: &[Vec<V3>]) -> Option<V3> {
    if rings.is_empty() {
        return None;
    }
    let dirs: Vec<V3> = samples
        .iter()
        .filter_map(|&p| unit(sub(p, center)))
        .collect();
    if dirs.is_empty() {
        return None;
    }
    let mut candidates: Vec<V3> = Vec::new();
    if let Some(m) = unit(dirs.iter().fold([0.0; 3], |s, d| add(s, *d))) {
        candidates.push(m.map(|x| -x));
    }
    for ring in rings {
        if let Some(m) = unit(sub(centroid(ring), center)) {
            candidates.push(m);
            candidates.push(m.map(|x| -x));
        }
    }
    // The candidate farthest from the face (the least cosine to it).
    let (n, closest) = candidates
        .into_iter()
        .map(|n| {
            let c = dirs
                .iter()
                .map(|d| dot(*d, n))
                .fold(f64::NEG_INFINITY, f64::max);
            (n, c)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))?;
    if closest > POLE_CLEARANCE_DEG.to_radians().cos() {
        return None;
    }
    Some(n.map(|x| -x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NurbsCurve;
    use std::sync::Arc;

    /// `n` points of the circle round the z axis at height `z` and radius
    /// `r`, turning `dir` (1 or -1).
    fn circle(n: usize, r: f64, z: f64, dir: f64) -> Vec<V3> {
        (0..n)
            .map(|k| {
                let a = dir * TAU * k as f64 / n as f64;
                [r * a.cos(), r * a.sin(), z]
            })
            .collect()
    }

    /// The domain of a face on `s` with loops `loops`, and checks that every
    /// loop point lifts back where it is and every own point lies on `s`.
    fn domain_of(s: &Surface, loops: &[Vec<V3>], extra: &[V3]) -> (Chart, Domain, Vec<V3>) {
        let points: Vec<V3> = loops.iter().flatten().chain(extra).copied().collect();
        let mut next = 0u32;
        let rings: Vec<Vec<u32>> = loops
            .iter()
            .map(|l| {
                l.iter()
                    .map(|_| {
                        next += 1;
                        next - 1
                    })
                    .collect()
            })
            .collect();
        let extra: Vec<u32> = (next..points.len() as u32).collect();
        let samples: Vec<V3> = loops.iter().map(|l| centroid(l.as_slice())).collect();
        let mut c = Chart::of(s, &samples, loops).unwrap_or_else(|| panic!("{}", s.name()));
        let d = c
            .domain(&points, &rings, &[], &extra, &|_| 0.3, 0.3)
            .unwrap_or_else(|| panic!("{}", s.name()));
        for (i, &slot) in d.slots.iter().enumerate() {
            let p = match slot {
                Slot::Global(g) => points[g as usize],
                Slot::Own(k) => d.own[k as usize],
            };
            assert!(dist(s.closest(p).0, p) < 1e-9, "{} point on it", s.name());
            assert!(dist(c.lift(d.pts[i]), p) < 1e-7, "{} lift {i}", s.name());
        }
        (c, d, points)
    }

    #[test]
    fn a_barrel_is_cut_along_a_seam() {
        let s = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0);
        let (c, d, _) = domain_of(
            &s,
            &[circle(24, 1.0, 0.0, 1.0), circle(24, 1.0, 2.0, -1.0)],
            &[],
        );
        assert!(c.seam && !d.own.is_empty(), "seam points of its own");
        assert_eq!(d.loops.len(), 1, "one outline round the cut barrel");
        // Unrolled: the outline spans a turn of the circumference.
        let xs: Vec<f64> = d.pts.iter().map(|q| q[0]).collect();
        let span = xs.iter().cloned().fold(f64::MIN, f64::max)
            - xs.iter().cloned().fold(f64::MAX, f64::min);
        assert!((span - TAU).abs() < 1e-9, "{span}");
    }

    #[test]
    fn a_cone_down_to_its_apex_unrolls_into_a_sector() {
        let s = Surface::cone([0.0; 3], [0.0, 0.0, 1.0], 0.5);
        let r = 2.0 * 0.5f64.tan();
        let (_, d, _) = domain_of(&s, &[circle(24, r, 2.0, 1.0)], &[]);
        // The slant distance keeps: the rim lies at the slant length.
        let slant = 2.0 / 0.5f64.cos();
        let rim = d.pts.iter().map(|q| len(*q)).fold(0.0, f64::max);
        assert!((rim - slant).abs() < 1e-9, "{rim} {slant}");
    }

    #[test]
    fn a_spherical_cap_is_seen_from_its_far_pole() {
        let s = Surface::sphere([0.0; 3], 1.0);
        let z = 0.3f64;
        let (c, d, _) = domain_of(&s, &[circle(24, (1.0 - z * z).sqrt(), z, 1.0)], &[]);
        assert!(!c.seam && d.own.is_empty(), "no seam");
        // Stereographic: the cap's rim a circle in the chart.
        let r: Vec<f64> = d.pts.iter().map(|q| len(*q)).collect();
        assert!(r.iter().all(|x| (x - r[0]).abs() < 1e-9));
    }

    #[test]
    fn a_whole_sphere_is_two_caps_on_one_equator() {
        let s = Surface::sphere([1.0, -2.0, 0.5], 2.0);
        let [(a, da), (b, db)] = Chart::sphere_caps(&s, &[], &[], &|_| 0.4).unwrap();
        assert_eq!(da.own.len(), db.own.len());
        for ((c, d), k) in [(&a, &da), (&b, &db)].into_iter().zip(0..) {
            for (i, &slot) in d.slots.iter().enumerate() {
                let Slot::Own(j) = slot else { panic!() };
                let p = d.own[j as usize];
                assert!(dist(s.closest(p).0, p) < 1e-9, "cap {k}");
                assert!(dist(c.lift(d.pts[i]), p) < 1e-9, "cap {k} lift");
                // The equator, seen from a pole, at twice the radius.
                assert!((len(d.pts[i]) - 4.0).abs() < 1e-9);
            }
            // The cap's middle lifts to its pole, on its own side.
            let z = c.lift([0.0, 0.0])[2] - 0.5;
            assert!((z.abs() - 2.0).abs() < 1e-9 && (z > 0.0) == (k == 0));
        }
        assert_eq!(da.own, db.own);
    }

    #[test]
    fn a_whole_torus_is_cut_round_both_ways() {
        let s = Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 2.0, 0.5);
        let points: Vec<V3> = Vec::new();
        let mut c = Chart::of(&s, &[], &[]).unwrap();
        let d = c.domain(&points, &[], &[], &[], &|_| 0.3, 0.3).unwrap();
        assert_eq!(d.loops.len(), 1);
        for (i, &slot) in d.slots.iter().enumerate() {
            let Slot::Own(k) = slot else { panic!() };
            let p = d.own[k as usize];
            assert!(dist(s.closest(p).0, p) < 1e-9);
            assert!(dist(c.lift(d.pts[i]), p) < 1e-9);
        }
    }

    #[test]
    fn the_inner_half_of_a_torus_lies_where_its_samples_do() {
        // Both halves of a torus between its top and bottom circles share
        // their loops: the samples tell the inner one from the outer.
        let s = Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 1.0, 0.3);
        let loops = [circle(32, 1.0, 0.3, 1.0), circle(32, 1.0, -0.3, -1.0)];
        let points: Vec<V3> = loops.iter().flatten().copied().collect();
        let rings: Vec<Vec<u32>> = vec![(0..32).collect(), (32..64).collect()];
        let inner: Vec<V3> = circle(16, 0.7, 0.0, 1.0);
        let mut c = Chart::of(&s, &inner, &loops).unwrap();
        let d = c.domain(&points, &rings, &[], &[], &|_| 0.1, 0.3).unwrap();
        // The middle of the chart lifts onto the inside of the ring.
        let (lo, hi) = d
            .pts
            .iter()
            .fold(([f64::MAX; 2], [f64::MIN; 2]), |(lo, hi), q| {
                (
                    [lo[0].min(q[0]), lo[1].min(q[1])],
                    [hi[0].max(q[0]), hi[1].max(q[1])],
                )
            });
        let p = c.lift([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])]);
        let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
        assert!((rho - 0.7).abs() < 1e-9, "{rho}");
    }

    #[test]
    fn a_band_of_a_revolved_profile_gets_a_strip() {
        let profile = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![[1.0, 0.0], [2.0, 1.0], [1.5, 2.0]],
            vec![1.0, 0.8, 1.0],
        );
        let s = Surface::Revolved {
            frame: Frame::new([0.0; 3], [0.0, 0.0, 1.0], None).unwrap(),
            profile: Arc::new(profile.clone()),
        };
        let [r0, z0] = profile.eval(0.0);
        let [r1, z1] = profile.eval(1.0);
        let (c, d, _) = domain_of(
            &s,
            &[circle(32, r0, z0, 1.0), circle(32, r1, z1, -1.0)],
            &[],
        );
        assert!(c.seam && d.loops.len() == 1);
    }
}
