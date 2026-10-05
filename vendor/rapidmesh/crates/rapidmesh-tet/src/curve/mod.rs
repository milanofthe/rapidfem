//! Stage 1 of the bottom-up sizing hierarchy: point distribution on a general
//! edge curve.
//!
//! A curve is parametrized by arc length `s in [0, length]`. Points are placed to
//! meet a chord (sagitta) error bound -- an element of length `h` on a curve of
//! radius `R` deviates by `eps ~ h^2/(8R)`, so bounding the relative deviation
//! `delta = eps/R` gives `h <= R*sqrt(8*delta)` -- under a SMOOTHNESS constraint:
//! the size field is gradient-limited so adjacent elements differ by at most a
//! fixed RATIO (`1 + grad`). The ratio (multiplicative) limit is what gives both a
//! NARROW transition (few points -- size coarsens geometrically away from a tight
//! feature) AND no abrupt density jump (every neighbour pair is within `1+grad`).
//! An additive `h0 + slope*dist` limit cannot have both: a gentle slope floods a
//! wide band with fine points, a steep slope jumps.
//!
//! Points are then placed at equal increments of the cumulative density
//! `C(s) = integral 1/h ds` -- the 1D centroidal-Voronoi (equal-error) optimum --
//! so no separate relaxation pass is needed. This module is geometry only: it does
//! not know about the mesh, surfaces, or the PLC; higher stages feed it curves and
//! consume its points.

pub(crate) mod kinds;

use crate::simplex::circumradius;
use rapidmesh_exact::vector::{
    add, bbox, closest_on_segment, dist, dist2, dot, normalize, scale, sub, V3,
};
use rapidmesh_geom::bvh::Bvh;
/// A general edge curve, parametrized by arc length `s in [0, length()]`.
/// `Send + Sync` supertraits: curve evaluators are plain data, and the
/// refiner is shared across rayon workers for its read-only stages.
pub trait Curve: Send + Sync {
    /// Total arc length.
    fn length(&self) -> f64;
    /// Position at arc length `s` (clamped to `[0, length]`).
    fn point_at(&self, s: f64) -> V3;
    /// Local principal radius of curvature `R = 1/kappa` at `s` (`INFINITY` where
    /// the curve is straight). The input to the sagitta size bound.
    fn radius_at(&self, s: f64) -> f64;
    /// Position and its first two derivatives in arc length at `s`: the point,
    /// the unit tangent and the curvature vector. The default differentiates
    /// `point_at` numerically; analytic curves give theirs in closed form.
    fn ders_at(&self, s: f64) -> [V3; 3] {
        let len = self.length();
        let h = 1e-4 * len;
        let (a, b) = ((s - h).max(0.0), (s + h).min(len));
        let half = 0.5 * (b - a);
        if !(half > 0.0) {
            return [self.point_at(s), [0.0; 3], [0.0; 3]];
        }
        let (pa, pm, pb) = (self.point_at(a), self.point_at(a + half), self.point_at(b));
        [
            self.point_at(s),
            scale(sub(pb, pa), 0.5 / half),
            scale(add(sub(pb, pm), sub(pa, pm)), 1.0 / (half * half)),
        ]
    }
}

/// The arc length of the point of `curve` nearest `p`. `samples` are
/// `(arc length, point)` pairs in ascending order covering the curve: the
/// nearest one starts safeguarded Newton steps on `(C(s) - p) . C'(s) = 0`
/// between its neighbours, each step halved until the distance drops. On a
/// closed curve (first and last sample one point) both sides of the seam are
/// searched. For many queries on one curve, [`CurveSamples`] finds the
/// nearest sample by its index instead of by a scan.
pub fn closest_arc(curve: &dyn Curve, samples: &[(f64, V3)], p: V3) -> f64 {
    let i = (0..samples.len())
        .min_by(|&a, &b| dist(samples[a].1, p).total_cmp(&dist(samples[b].1, p)))
        .unwrap_or(0);
    arc_from(curve, samples, i, p)
}

/// [`closest_arc`] from the sample `i` nearest `p`.
fn arc_from(curve: &dyn Curve, samples: &[(f64, V3)], i: usize, p: V3) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let bracket = |j: usize| {
        (
            samples[j].0,
            samples[j.saturating_sub(1)].0,
            samples[(j + 1).min(n - 1)].0,
        )
    };
    let mut starts = vec![bracket(i)];
    let closed = n > 2 && dist(samples[0].1, samples[n - 1].1) <= 1e-12 * curve.length();
    if closed && (i == 0 || i == n - 1) {
        starts.push(bracket(n - 1 - i));
    }
    starts
        .into_iter()
        .map(|(s0, lo, hi)| newton_arc(curve, s0, lo, hi, p))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(0.0, |(s, _)| s)
}

/// A dense sample of a curve, `(arc length, point)` in ascending order,
/// with a tree over the segments between consecutive samples: the nearest
/// sample and the nearest point of the polyline in logarithmic time, for a
/// curve projected onto again and again (the finish moves its vertices).
pub struct CurveSamples {
    pub samples: Vec<(f64, V3)>,
    bvh: Bvh,
}

impl CurveSamples {
    pub fn new(samples: Vec<(f64, V3)>) -> CurveSamples {
        let segs = || samples.windows(2).map(|w| [w[0].1, w[1].1]);
        let boxes = segs().map(bbox).collect();
        let mids: Vec<V3> = segs().map(|[a, b]| scale(add(a, b), 0.5)).collect();
        let bvh = Bvh::build(boxes, &mids);
        CurveSamples { samples, bvh }
    }

    /// The index of the sample nearest `p` (the first on a tie).
    fn nearest_sample(&self, p: V3) -> usize {
        if self.samples.len() < 2 {
            return 0;
        }
        let d2 = |i: usize| dist2(self.samples[i].1, p);
        let near = self.bvh.nearest(p, f64::INFINITY, |j| {
            let j = j as usize;
            Some(d2(j).min(d2(j + 1)))
        });
        near.map_or(0, |(j, _)| {
            let j = j as usize;
            if d2(j + 1) < d2(j) {
                j + 1
            } else {
                j
            }
        })
    }

    /// [`closest_arc`] on these samples of `curve`.
    pub fn closest_arc(&self, curve: &dyn Curve, p: V3) -> f64 {
        arc_from(curve, &self.samples, self.nearest_sample(p), p)
    }

    /// The point of the polyline through the samples nearest `p`.
    pub fn nearest_on_polyline(&self, p: V3) -> Option<V3> {
        let foot = |j: u32| {
            let (a, b) = (self.samples[j as usize].1, self.samples[j as usize + 1].1);
            closest_on_segment(p, a, b)
        };
        let (j, _) = self
            .bvh
            .nearest(p, f64::INFINITY, |j| Some(dist2(foot(j), p)))?;
        Some(foot(j))
    }
}

/// Safeguarded Newton for the nearest arc length in `[lo, hi]` from `s0`,
/// with the squared distance there.
fn newton_arc(curve: &dyn Curve, s0: f64, lo: f64, hi: f64, p: V3) -> (f64, f64) {
    let d2 = |s: f64| {
        let r = sub(curve.point_at(s), p);
        dot(r, r)
    };
    let (mut s, mut d) = (s0, d2(s0));
    for _ in 0..24 {
        let [c, t, k] = curve.ders_at(s);
        let r = sub(c, p);
        let g = dot(r, t);
        let gp = dot(t, t) + dot(r, k);
        // Newton where the distance is convex along the curve, else a
        // gradient step.
        let step = if gp > 0.0 {
            g / gp
        } else {
            g / dot(t, t).max(f64::MIN_POSITIVE)
        };
        let mut lambda = 1.0;
        let mut moved = false;
        for _ in 0..12 {
            let c = (s - lambda * step).clamp(lo, hi);
            let dc = d2(c);
            if dc < d {
                moved = (c - s).abs() > 1e-15 * (hi - lo);
                s = c;
                d = dc;
                break;
            }
            lambda *= 0.5;
        }
        if !moved {
            break;
        }
    }
    (s, d)
}

/// A curve given as a polyline of on-curve sample points. Curvature is the
/// discrete osculating-circle radius (the circumradius of three consecutive
/// samples). Works for any edge; an analytic edge supplies a finely resampled,
/// exactly-on-curve polyline so the discrete radius matches the true one.
pub struct PolylineCurve {
    pts: Vec<V3>,
    cum: Vec<f64>, // cumulative arc length at each sample
}

impl PolylineCurve {
    /// Builds a polyline curve from samples (>= 2). Duplicate consecutive points
    /// are dropped so the arc-length table is strictly increasing.
    pub fn new(samples: &[V3]) -> Option<PolylineCurve> {
        let mut pts: Vec<V3> = Vec::with_capacity(samples.len());
        for &p in samples {
            if pts.last().map(|&q| dist(p, q) > 1e-15).unwrap_or(true) {
                pts.push(p);
            }
        }
        if pts.len() < 2 {
            return None;
        }
        let mut cum = vec![0.0f64; pts.len()];
        for i in 1..pts.len() {
            cum[i] = cum[i - 1] + dist(pts[i], pts[i - 1]);
        }
        Some(PolylineCurve { pts, cum })
    }

    /// Sample index `i` with `cum[i] <= s` (for interpolation).
    fn seg(&self, s: f64) -> usize {
        let s = s.clamp(0.0, self.cum[self.cum.len() - 1]);
        self.cum
            .partition_point(|&c| c < s)
            .clamp(1, self.cum.len() - 1)
            - 1
    }
}

impl Curve for PolylineCurve {
    fn length(&self) -> f64 {
        self.cum[self.cum.len() - 1]
    }

    fn point_at(&self, s: f64) -> V3 {
        let i = self.seg(s);
        let (s0, s1) = (self.cum[i], self.cum[i + 1]);
        let f = if s1 > s0 {
            (s.clamp(s0, s1) - s0) / (s1 - s0)
        } else {
            0.0
        };
        std::array::from_fn(|k| self.pts[i][k] + f * (self.pts[i + 1][k] - self.pts[i][k]))
    }

    fn radius_at(&self, s: f64) -> f64 {
        // Osculating radius at the sample nearest `s` (its two polyline neighbours).
        let i = self.seg(s);
        // Use the vertex closest to s as the apex of the triple.
        let apex = if i + 1 < self.pts.len() && (s - self.cum[i]) > (self.cum[i + 1] - s) {
            i + 1
        } else {
            i
        };
        if apex == 0 || apex + 1 >= self.pts.len() {
            return f64::INFINITY; // endpoints: no curvature defined
        }
        circumradius(self.pts[apex - 1], self.pts[apex], self.pts[apex + 1])
    }

    fn ders_at(&self, s: f64) -> [V3; 3] {
        // Straight within each segment.
        let i = self.seg(s);
        [
            self.point_at(s),
            normalize(sub(self.pts[i + 1], self.pts[i])),
            [0.0; 3],
        ]
    }
}

/// A curve sampled along another: the points of `along` (the chain of
/// facets an edge follows), the radius of `by` (its own curve) at the same
/// share of the length. The facets' irregular spacing shows spurious small
/// radii; the curve's own are the ones the sampling wants.
pub struct Guided<'a> {
    pub along: &'a dyn Curve,
    pub by: &'a dyn Curve,
}

impl Curve for Guided<'_> {
    fn length(&self) -> f64 {
        self.along.length()
    }
    fn point_at(&self, s: f64) -> V3 {
        self.along.point_at(s)
    }
    fn radius_at(&self, s: f64) -> f64 {
        let len = self.along.length();
        let share = if len > 0.0 { s / len } else { 0.0 };
        self.by.radius_at(share * self.by.length())
    }
    fn ders_at(&self, s: f64) -> [V3; 3] {
        self.along.ders_at(s)
    }
}

/// The most a curve turns per segment, floor or not: a third of a turn, as
/// a closed edge takes three segments at least.
const MAX_TURN: f64 = std::f64::consts::TAU / 3.0;

/// Arc-length samples of `curve` spaced by the target `size(s)` at arc
/// length `s`, refined where the curvature needs it (`bent(r)`: the size
/// at radius of curvature `r`, see `sizing::CurvatureLaw`), and graded by `grad`. A FLOOR on
/// the bend: a curvature-radius spike of the curve (the sharp turn of an
/// intersection curve, a micro-rim) may not drive the sampling below
/// `minh`; the size it is given (`size`, which the faces beside it take
/// too) it follows however fine. `0` = off.
pub fn distribute_floored(
    curve: &dyn Curve,
    bent: &dyn Fn(f64) -> f64,
    size: &dyn Fn(f64) -> f64,
    grad: f64,
    minh: f64,
) -> Vec<f64> {
    let len = curve.length();
    // The finest target along the curve sets the resolution of the samples.
    let maxh = (0..=32)
        .map(|i| size(len * i as f64 / 32.0))
        .fold(f64::INFINITY, f64::min);
    if !(len > 0.0) || !(maxh > 0.0) {
        return vec![0.0];
    }
    // Fine arc-length samples: enough to resolve the finest target (and a
    // bend of radius about that size). Cap the count.
    let m = ((len / bent(maxh).max(maxh * 0.05)).ceil() as usize * 4).clamp(64, 8192);
    let ds = len / m as f64;
    let mut h = vec![0.0f64; m + 1];
    for i in 0..=m {
        let s = (i as f64) * ds;
        let r = curve.radius_at(s);
        let target = size(s);
        // The floor holds the curve's own bend (a spike of its radius), not
        // the size it is given: the faces beside it take that size too.
        let bend = if r.is_finite() {
            bent(r).max(minh)
        } else {
            f64::INFINITY
        };
        h[i] = bend.min(target).max(1e-12);
    }
    // Multiplicative gradient limit: h cannot grow faster than the ratio (1+grad)
    // per element of its own length. Over a sub-element sample step `ds`, that is
    // h[i] <= h[i-1] * (1+grad)^(ds/h[i-1]). Forward then backward sweep makes the
    // field two-sided Lipschitz in log-space -> smooth, narrow, no jump.
    let g = grad.max(1e-6);
    for i in 1..=m {
        let cap = h[i - 1] * (1.0 + g).powf(ds / h[i - 1]);
        if h[i] > cap {
            h[i] = cap;
        }
    }
    for i in (0..m).rev() {
        let cap = h[i + 1] * (1.0 + g).powf(ds / h[i + 1]);
        if h[i] > cap {
            h[i] = cap;
        }
    }
    // Cumulative density C(s) = integral 1/h ds (trapezoid over the samples).
    let mut cum = vec![0.0f64; m + 1];
    for i in 1..=m {
        cum[i] = cum[i - 1] + 0.5 * (1.0 / h[i] + 1.0 / h[i - 1]) * ds;
    }
    let total = cum[m];
    // The floor keeps tiny features from refining the mesh, but the curve
    // takes a segment per MAX_TURN it turns at least: a bend stays one and
    // is no chord through the solid (a rounded plate edge of a radius below
    // the floor), where the sizes alone would give it none.
    let at: Vec<V3> = (0..=m).map(|i| curve.point_at(i as f64 * ds)).collect();
    let turned: f64 = (1..m)
        .map(|i| {
            let (u, v) = (sub(at[i], at[i - 1]), sub(at[i + 1], at[i]));
            let (lu, lv) = (dot(u, u).sqrt(), dot(v, v).sqrt());
            if lu > 0.0 && lv > 0.0 {
                (dot(u, v) / (lu * lv)).clamp(-1.0, 1.0).acos()
            } else {
                0.0
            }
        })
        .sum();
    let n = (total.round() as usize)
        .max(1)
        .max((turned / MAX_TURN - 1e-9).ceil() as usize); // number of elements
    let mut out = Vec::with_capacity(n + 1);
    out.push(0.0);
    let mut j = 1usize;
    for k in 1..n {
        let target = k as f64 / n as f64 * total;
        while j <= m && cum[j] < target {
            j += 1;
        }
        let j = j.min(m);
        let seg = (cum[j] - cum[j - 1]).max(1e-30);
        let f = ((target - cum[j - 1]) / seg).clamp(0.0, 1.0);
        out.push(((j - 1) as f64 + f) * ds);
    }
    out.push(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sizing::CurvatureLaw;
    use std::f64::consts::PI;

    /// A bend below the floor keeps a segment per third of a turn: a rounded
    /// edge of half a turn is no chord through the solid; a straight edge
    /// stays one segment.
    #[test]
    fn a_bend_below_the_floor_keeps_its_segments() {
        let arc: Vec<V3> = (0..=16)
            .map(|k| {
                let t = PI * k as f64 / 16.0;
                [2.0 * t.cos(), 2.0 * t.sin(), 0.0]
            })
            .collect();
        let bend = PolylineCurve::new(&arc).unwrap();
        let ss = distribute_floored(
            &bend,
            &|r| CurvatureLaw::Chord(0.05).curve(r),
            &|_| 50.0,
            0.5,
            5.0,
        );
        assert_eq!(ss.len(), 3, "{ss:?}");
        let line = PolylineCurve::new(&[[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]).unwrap();
        let ss = distribute_floored(
            &line,
            &|r| CurvatureLaw::Chord(0.05).curve(r),
            &|_| 50.0,
            0.5,
            5.0,
        );
        assert_eq!(ss.len(), 2, "{ss:?}");
    }

    /// The floor holds a curve's own bend, not the size it is given: a
    /// straight edge along a band whose faces are finer than the floor (a
    /// thin fillet at a fine geometric error) takes the faces' size (#299).
    #[test]
    fn a_size_below_the_floor_is_kept() {
        let line = PolylineCurve::new(&[[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]]).unwrap();
        let ss = distribute_floored(
            &line,
            &|r| CurvatureLaw::Chord(0.05).curve(r),
            &|_| 0.5,
            0.5,
            5.0,
        );
        assert_eq!(ss.len(), 13, "{ss:?}");
    }

    fn circle(r: f64, n: usize) -> PolylineCurve {
        let pts: Vec<V3> = (0..=n)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / n as f64;
                [r * t.cos(), r * t.sin(), 0.0]
            })
            .collect();
        PolylineCurve::new(&pts).unwrap()
    }

    #[test]
    fn polyline_arc_length_and_radius() {
        let c = circle(2.0, 400);
        assert!((c.length() - 2.0 * PI * 2.0).abs() < 1e-2);
        // discrete radius ~ true radius
        let r = c.radius_at(c.length() * 0.3);
        assert!((r - 2.0).abs() < 0.05, "radius {r}");
    }

    #[test]
    fn circle_distribution_is_uniform_and_meets_bound() {
        let r = 2.0;
        let delta = 0.02;
        let c = circle(r, 2000);
        let s = distribute_floored(
            &c,
            &|r| CurvatureLaw::Chord(delta).curve(r),
            &|_| 100.0,
            0.3,
            0.0,
        );
        // Expected element length ~ R*sqrt(8*delta); count ~ circumference / h.
        let h = r * (8.0 * delta).sqrt();
        let expect = (2.0 * PI * r / h).round() as usize;
        let n = s.len() - 1; // elements
        assert!(
            (n as i64 - expect as i64).abs() <= 2,
            "count {n} vs expected {expect}"
        );
        // Spacing is near-uniform on a circle (constant curvature).
        let mut spc: Vec<f64> = s.windows(2).map(|w| w[1] - w[0]).collect();
        spc.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ratio = spc[spc.len() - 1] / spc[0];
        assert!(ratio < 1.2, "circle spacing ratio {ratio}");
    }

    #[test]
    fn straight_edge_uses_maxh() {
        let pts: Vec<V3> = (0..=50)
            .map(|i| [i as f64 / 50.0 * 10.0, 0.0, 0.0])
            .collect();
        let c = PolylineCurve::new(&pts).unwrap();
        let s = distribute_floored(
            &c,
            &|r| CurvatureLaw::Chord(0.02).curve(r),
            &|_| 1.0,
            0.3,
            0.0,
        );
        let n = s.len() - 1;
        assert_eq!(n, 10, "10 elements of maxh=1 on a length-10 line, got {n}");
    }

    #[test]
    fn high_curvature_spot_grades_smoothly() {
        // A curve: long straight arms with a tight semicircle bump in the middle,
        // so the field is fine at the bump and coarse on the arms. Verify the
        // OUTPUT spacing ratio between adjacent elements stays bounded (smooth).
        let mut pts: Vec<V3> = Vec::new();
        for i in 0..=100 {
            pts.push([-5.0 + i as f64 / 100.0 * 5.0, 0.0, 0.0]); // arm to origin
        }
        let rb = 0.1;
        for i in 1..100 {
            let t = PI * i as f64 / 100.0;
            pts.push([rb * t.sin(), rb * (1.0 - t.cos()), 0.0]); // semicircle bump
        }
        for i in 0..=100 {
            pts.push([i as f64 / 100.0 * 5.0, 0.0, 0.0]); // arm away
        }
        let c = PolylineCurve::new(&pts).unwrap();
        let s = distribute_floored(
            &c,
            &|r| CurvatureLaw::Chord(0.02).curve(r),
            &|_| 1.0,
            0.3,
            0.0,
        );
        let spc: Vec<f64> = s.windows(2).map(|w| w[1] - w[0]).collect();
        // No adjacent pair jumps by more than ~ (1+grad) plus a sampling margin.
        let mut worst = 1.0f64;
        for w in spc.windows(2) {
            worst = worst.max(w[1] / w[0]).max(w[0] / w[1]);
        }
        assert!(worst < 1.6, "adjacent spacing ratio {worst} not smooth");
        // And it must actually refine the bump (some element far below maxh).
        assert!(
            spc.iter().cloned().fold(f64::MAX, f64::min) < 0.1,
            "bump not refined"
        );
    }
}
