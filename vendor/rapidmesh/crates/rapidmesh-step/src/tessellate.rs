//! The solids of a STEP model as faceted solids for the model: every edge
//! sampled once and shared by the faces on it, so each body is closed by
//! construction; every face triangulated in its surface's parameters (its
//! boundary there from the file's parameter curves, else projected), the
//! facets carrying the face's surface as their carrier. Edges and faces are
//! done in parallel.

use crate::entities::{Bound, Edge, Face, Model, StepError};
use crate::geometry::{near, Curve, Curve2, Surface, P3};
use rapidmesh_csg::Tri;
use rapidmesh_exact::Sign;
use rapidmesh_geom::cdt2::triangulate_constrained;
use rapidmesh_geom::vec3::{bbox, dist, dot, sub, unit};
use rapidmesh_geom::{polygon_orientation, CurveKind, EdgeCurve, Faceted, SurfaceKind};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::f64::consts::TAU;
use std::sync::Arc;

/// A body of the file.
pub struct Body {
    pub name: String,
    pub solid: Faceted,
}

/// How finely the carriers are faceted.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    /// The largest distance of a facet from its surface, relative to the
    /// model's size.
    pub chord: f64,
    /// The fewest segments on a full circle.
    pub min_segments: usize,
}

impl Default for Tolerance {
    fn default() -> Tolerance {
        Tolerance {
            chord: 1e-3,
            min_segments: 12,
        }
    }
}

/// How often the edges whose samples cross in a face's parameters are
/// sampled twice as finely, at most.
const REFINE_ROUNDS: usize = 4;

/// The samples of an edge from its first vertex to its second, with the
/// parameter of each on the edge's curve.
struct Samples {
    ts: Vec<f64>,
    pts: Vec<P3>,
}

struct Tess<'a> {
    m: &'a Model,
    /// The largest distance of a facet from its surface.
    chord: f64,
    /// How far a point from a parameter curve may lie off the edge.
    fit: f64,
    min_segments: usize,
    samples: Vec<Samples>,
}

/// The parameters along an edge's curve from `p0` to `p1` (all of it
/// when `closed`), with its parameter or, not `forward`, against it.
fn span(curve: &Curve, p0: P3, p1: P3, forward: bool, closed: bool) -> (f64, f64) {
    let t0 = curve.param(p0);
    let mut t1 = curve.param(p1);
    if let Some(period) = curve.period() {
        let sign = if forward { 1.0 } else { -1.0 };
        // Onward from t0 by less than a period, a whole one if closed.
        let mut delta = sign * (t1 - t0);
        delta -= period * (delta / period).floor();
        if closed || delta < 1e-12 {
            delta = period;
        }
        t1 = t0 + sign * delta;
    }
    (t0, t1)
}

impl Tess<'_> {
    /// The samples of edge `e`, `parts` times as many on a curve as its
    /// chord asks for.
    fn sample(&self, e: &Edge, parts: usize) -> Samples {
        let curve = &self.m.curves[e.curve];
        let (p0, p1) = (self.m.vertices[e.ends[0]], self.m.vertices[e.ends[1]]);
        let (t0, t1) = span(curve, p0, p1, e.forward, e.ends[0] == e.ends[1]);
        let mut ts: Vec<f64> = match curve {
            Curve::Line { .. } => vec![t0, t1],
            Curve::Circle { r, .. } => {
                self.even(t0, t1, parts * self.segments(*r, (t1 - t0).abs()))
            }
            Curve::Ellipse { a, b, .. } => self.even(
                t0,
                t1,
                parts * self.segments(a.min(*b).max(1e-12), (t1 - t0).abs()),
            ),
            Curve::Spline(_) | Curve::Hyperbola { .. } | Curve::Parabola { .. } => {
                // Halved where the curve strays from its chord by more than
                // the tolerance, from two spans per control point (eight
                // on a conic).
                let start = parts
                    * match curve {
                        Curve::Spline(s) => 2 * s.ctrl.len(),
                        _ => 8,
                    };
                let mut ts = self.even(t0, t1, start);
                let mut i = 0;
                while i + 1 < ts.len() && ts.len() < 4000 * parts {
                    let (a, b) = (ts[i], ts[i + 1]);
                    let (pa, pb, pm) = (curve.eval(a), curve.eval(b), curve.eval(0.5 * (a + b)));
                    let mid: P3 = std::array::from_fn(|k| 0.5 * (pa[k] + pb[k]));
                    if dist(pm, mid) > self.chord {
                        ts.insert(i + 1, 0.5 * (a + b));
                    } else {
                        i += 1;
                    }
                }
                ts
            }
        };
        let mut pts: Vec<P3> = ts.iter().map(|&t| curve.eval(t)).collect();
        // The ends are the vertices, exactly: the edges meeting there share
        // them.
        let last = pts.len() - 1;
        pts[0] = p0;
        pts[last] = p1;
        ts[0] = t0;
        ts[last] = t1;
        Samples { ts, pts }
    }

    fn even(&self, t0: f64, t1: f64, n: usize) -> Vec<f64> {
        let n = n.max(1);
        (0..=n)
            .map(|i| t0 + (t1 - t0) * i as f64 / n as f64)
            .collect()
    }

    /// Segments on an arc of `angle` at radius `r` within the chord.
    fn segments(&self, r: f64, angle: f64) -> usize {
        let step = 2.0 * (1.0 - (self.chord / r).min(1.0)).acos();
        let full = (TAU / step.max(1e-6)).ceil().max(self.min_segments as f64);
        ((full * angle / TAU).ceil() as usize).max(1)
    }

    /// The parameters on `surf` (surface `si`) of the samples of edge `e`,
    /// in the order given: from the edge's parameter curve on the surface
    /// that continues from `prev` (a seam has two), where it lies on the
    /// edge, else projected.
    fn edge_uv(
        &self,
        surf: &Surface,
        si: usize,
        e: usize,
        ts: &[f64],
        pts: &[P3],
        prev: Option<[f64; 2]>,
    ) -> Vec<[f64; 2]> {
        let on: Vec<&Curve2> = self.m.edges[e]
            .pcurves
            .iter()
            .filter(|(s, _)| *s == si)
            .map(|(_, c)| c)
            .collect();
        let first = |c: &Curve2| c.eval(ts[0]);
        let pick = match (on.len(), prev) {
            (0, _) => None,
            (_, None) => Some(on[0]),
            (_, Some(q)) => on.iter().copied().min_by(|a, b| {
                let d = |f: [f64; 2]| (f[0] - q[0]).powi(2) + (f[1] - q[1]).powi(2);
                d(first(a)).total_cmp(&d(first(b)))
            }),
        };
        if let Some(c) = pick {
            let uv: Vec<[f64; 2]> = ts.iter().map(|&t| c.eval(t)).collect();
            let n = uv.len();
            let fits = [0, n / 2, n - 1]
                .iter()
                .all(|&i| dist(surf.eval(uv[i]), pts[i]) <= self.fit);
            if fits {
                return uv;
            }
        }
        pts.iter().map(|&p| surf.param(p)).collect()
    }

    /// The facets of face `f`: its carrier and its triangles. With `check`,
    /// bounds whose sides cross in the surface's parameters (an edge's
    /// chords cutting across a neighbour's where the face is thinner than
    /// they stray from their curves) come back as those sides instead.
    fn face(&self, f: &Face, check: bool) -> Result<FaceOut, StepError> {
        let err = |message: String| StepError { id: f.id, message };
        let surf = &self.m.surfaces[f.surface];
        let mut rings: Vec<Vec<P3>> = Vec::new();
        let mut uv: Vec<Vec<[f64; 2]>> = Vec::new();
        for b in &f.bounds {
            let Bound::Edges(edges) = b else {
                continue;
            };
            let (mut ring, mut ring_uv) = (Vec::new(), Vec::new());
            for &(e, along) in edges {
                let s = &self.samples[e];
                let (mut ts, mut pts) = (s.ts.clone(), s.pts.clone());
                if !along {
                    ts.reverse();
                    pts.reverse();
                }
                let q = self.edge_uv(surf, f.surface, e, &ts, &pts, ring_uv.last().copied());
                let n = pts.len() - 1;
                ring.extend_from_slice(&pts[..n]);
                ring_uv.extend_from_slice(&q[..n]);
            }
            if ring.len() >= 3 {
                rings.push(ring);
                uv.push(ring_uv);
            }
        }
        if rings.is_empty() {
            // Bounded by no edge (a pole's vertex at most): all of a closed
            // surface.
            return match self.whole(surf, f) {
                Some(tris) => Ok(FaceOut::Facets(surf.kind(), tris)),
                None => Err(err("no bounds".into())),
            };
        }
        // Periodic parameters unwrapped along each ring (projected points
        // come back within one period), every ring moved into the turn of
        // the outer one. A point at a pole has no angle of its own: it is
        // skipped and then split in two, one with the angle of the point
        // before it, one with that of the point after, the pole's line
        // between them.
        let periods = surf.periods();
        let poles = surf.poles();
        let pole_at = |p: P3| {
            poles
                .iter()
                .find(|pole| dist(pole.2, p) <= 1e-2 * self.fit)
                .map(|pole| (pole.0, pole.1))
        };
        for (ring, r) in rings.iter_mut().zip(uv.iter_mut()) {
            let at: Vec<Option<(usize, f64)>> = ring.iter().map(|&p| pole_at(p)).collect();
            let mut prev: Option<[f64; 2]> = None;
            for i in 0..r.len() {
                if at[i].is_some() {
                    continue;
                }
                if let Some(q) = prev {
                    for k in 0..2 {
                        if let Some(period) = periods[k] {
                            r[i][k] = near(r[i][k], q[k], period);
                        }
                    }
                }
                prev = Some(r[i]);
            }
            if at.iter().all(Option::is_none) {
                continue;
            }
            if at.iter().all(Option::is_some) {
                return Err(err("a bound lies in a pole".into()));
            }
            let n = r.len();
            let (mut split, mut split_uv) = (Vec::with_capacity(n + 2), Vec::with_capacity(n + 2));
            for i in 0..n {
                let Some((fixed, value)) = at[i] else {
                    split.push(ring[i]);
                    split_uv.push(r[i]);
                    continue;
                };
                let free = |step: usize| {
                    let j = (1..n)
                        .map(|d| (i + step * d) % n)
                        .find(|&j| at[j].is_none())
                        .unwrap_or(i);
                    let mut q = r[j];
                    q[fixed] = value;
                    q
                };
                let (from, to) = (free(n - 1), free(1));
                split.push(ring[i]);
                split_uv.push(from);
                if from != to {
                    split.push(ring[i]);
                    split_uv.push(to);
                }
            }
            *ring = split;
            *r = split_uv;
        }
        // The points of a seam between `a` and `b`: halved until each
        // piece keeps to the chord (the seam is no edge, nothing sampled it).
        let seam = |a: [f64; 2], b: [f64; 2]| -> Vec<([f64; 2], P3)> {
            let at = |w: f64| [a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1])];
            let mut parts = 1usize;
            while parts < 256
                && (0..parts).any(|i| {
                    let (w0, w1) = (i as f64 / parts as f64, (i + 1) as f64 / parts as f64);
                    let (p0, p1) = (surf.eval(at(w0)), surf.eval(at(w1)));
                    let mid: P3 = std::array::from_fn(|k| 0.5 * (p0[k] + p1[k]));
                    dist(surf.eval(at(0.5 * (w0 + w1))), mid) > self.chord
                })
            {
                parts *= 2;
            }
            (1..parts)
                .map(|i| {
                    let t = at(i as f64 / parts as f64);
                    (t, surf.eval(t))
                })
                .collect()
        };
        join_windings(&mut rings, &mut uv, periods, &poles, f.same_sense, &seam).map_err(err)?;
        // A side of a ring along a pole (one point in space) takes points
        // as closely as a turn of an edge's segment: facets fan into the
        // pole from next to each other, not from across the face.
        let turn = TAU / self.min_segments as f64;
        for (ring, r) in rings.iter_mut().zip(uv.iter_mut()) {
            let n = r.len();
            let (mut dense, mut dense_uv) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for i in 0..n {
                let j = (i + 1) % n;
                dense.push(ring[i]);
                dense_uv.push(r[i]);
                if ring[i] == ring[j] && r[i] != r[j] {
                    let span = (r[j][0] - r[i][0]).abs().max((r[j][1] - r[i][1]).abs());
                    let parts = (span / turn).ceil() as usize;
                    for m in 1..parts {
                        let w = m as f64 / parts as f64;
                        dense.push(ring[i]);
                        dense_uv.push([
                            r[i][0] + w * (r[j][0] - r[i][0]),
                            r[i][1] + w * (r[j][1] - r[i][1]),
                        ]);
                    }
                }
            }
            *ring = dense;
            *r = dense_uv;
        }
        let mean = |r: &[[f64; 2]], k: usize| r.iter().map(|q| q[k]).sum::<f64>() / r.len() as f64;
        let outer = uv
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| area(a).abs().total_cmp(&area(b).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0);
        for k in 0..2 {
            let Some(period) = periods[k] else {
                continue;
            };
            let m0 = mean(&uv[outer], k);
            for (i, r) in uv.iter_mut().enumerate() {
                if i != outer {
                    let shift = near(mean(r, k), m0, period) - mean(r, k);
                    r.iter_mut().for_each(|q| q[k] += shift);
                }
            }
        }
        let mut pts3: Vec<P3> = rings.iter().flatten().copied().collect();
        let mut pts2: Vec<[f64; 2]> = uv.iter().flatten().copied().collect();
        let holes: Vec<&[[f64; 2]]> = uv
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != outer)
            .map(|(_, r)| r.as_slice())
            .collect();
        // The constrained Delaunay of the rings' points and the inner ones
        // with the rings' sides forced, less what lies outside the outer
        // ring or in a hole.
        let mut segments = Vec::with_capacity(pts2.len());
        let mut first = 0;
        for r in &uv {
            segments.extend((0..r.len()).map(|i| (first + i, first + (i + 1) % r.len())));
            first += r.len();
        }
        let (inner, stretch) = self.inner_points(surf, &uv[outer], &holes);
        pts2.extend_from_slice(&inner);
        pts3.extend(inner.iter().map(|&q| surf.eval(q)));
        // Triangulated where the parameters are scaled by how far the
        // surface runs along them (lengths on a cylinder or a cone): an
        // angle and a height are not measured alike, and a Delaunay of the
        // raw parameters joins points right across a curved face.
        let scaled: Vec<[f64; 2]> = pts2
            .iter()
            .map(|q| [q[0] * stretch[0], q[1] * stretch[1]])
            .collect();
        let inside = |q: [f64; 2]| {
            let q = [q[0] / stretch[0], q[1] / stretch[1]];
            contains(&uv[outer], q) && !holes.iter().any(|h| contains(h, q))
        };
        // The sides of the bounds that cross, then those no facet takes
        // (a fold where two bounds leave a point of tangency side by side):
        // the longest of them, whose edges then take more samples (more of
        // both would fold alike).
        let longest = |sides: Vec<usize>| -> FaceOut {
            let length = |k: usize| dist(pts3[segments[k].0], pts3[segments[k].1]);
            let most = sides.iter().map(|&k| length(k)).fold(0.0, f64::max);
            FaceOut::Crossing(
                sides
                    .into_iter()
                    .filter(|&k| length(k) >= 0.95 * most)
                    .map(|k| (pts3[segments[k].0], pts3[segments[k].1]))
                    .collect(),
            )
        };
        if check {
            let crossed = crossings(&scaled, &segments);
            if !crossed.is_empty() {
                return Ok(longest(crossed));
            }
        }
        let mut tris = triangulate_constrained(&scaled, &segments, inside);
        if check && !tris.is_empty() {
            let mut taken: rustc_hash::FxHashSet<(usize, usize)> = Default::default();
            for t in &tris {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    taken.insert((a.min(b), a.max(b)));
                }
            }
            let bare: Vec<usize> = (0..segments.len())
                .filter(|&k| {
                    let (a, b) = segments[k];
                    pts3[a] != pts3[b] && !taken.contains(&(a.min(b), a.max(b)))
                })
                .collect();
            if !bare.is_empty() {
                return Ok(longest(bare));
            }
        }
        if tris.is_empty() {
            // Bounds that run back along themselves (two edges on one arc,
            // there and back) enclose nothing: the face has no facets.
            let length: f64 = segments
                .iter()
                .map(|&(i, j)| {
                    let (a, b) = (scaled[i], scaled[j]);
                    (a[0] - b[0]).hypot(a[1] - b[1])
                })
                .sum();
            let enclosed: f64 = uv
                .iter()
                .map(|r| {
                    let r: Vec<[f64; 2]> = r
                        .iter()
                        .map(|q| [q[0] * stretch[0], q[1] * stretch[1]])
                        .collect();
                    area(&r).abs()
                })
                .sum();
            if enclosed <= 1e-6 * length * length {
                rapidmesh_exact::log::debug(
                    "step.facets",
                    format!("face #{}: its bounds enclose nothing, no facets", f.id),
                );
                return Ok(FaceOut::Facets(surf.kind(), Vec::new()));
            }
            return Err(err("its bounds do not triangulate in its parameters".into()));
        }
        // A new diagonal must keep to the chord: its middle off the surface
        // by no more than it.
        let on_surface = |i: usize, j: usize| {
            let mid2 = [
                0.5 * (pts2[i][0] + pts2[j][0]),
                0.5 * (pts2[i][1] + pts2[j][1]),
            ];
            let mid3: P3 = std::array::from_fn(|k| 0.5 * (pts3[i][k] + pts3[j][k]));
            dist(surf.eval(mid2), mid3) <= self.chord
        };
        flip_to_shape(&mut tris, &pts2, &pts3, on_surface);
        // A facet far off its surface is a face triangulated over the wrong
        // part of its parameters: said, for the diagnosis.
        let far = tris
            .iter()
            .map(|t| {
                let c: P3 =
                    std::array::from_fn(|k| (pts3[t[0]][k] + pts3[t[1]][k] + pts3[t[2]][k]) / 3.0);
                dist(c, surf.eval(surf.param(c)))
            })
            .fold(0.0f64, f64::max);
        if far > 4.0 * self.chord {
            rapidmesh_exact::log::debug(
                "step.facets",
                format!(
                    "face #{}: a facet {far:.3e} off its surface (chord {:.3e})",
                    f.id, self.chord
                ),
            );
        }
        let out = tris
            .into_iter()
            // A facet on a pole has two corners there: nothing in space.
            .filter(|t| {
                pts3[t[0]] != pts3[t[1]] && pts3[t[1]] != pts3[t[2]] && pts3[t[2]] != pts3[t[0]]
            })
            .map(|t| {
                let [a, b, c] = t.map(|i| pts3[i]);
                // Counterclockwise in the parameters is the surface's normal;
                // the face's is it or its opposite.
                let ccw = polygon_orientation(&t.map(|i| pts2[i])) == Sign::Positive;
                if ccw == f.same_sense {
                    Tri::new(a, b, c)
                } else {
                    Tri::new(a, c, b)
                }
            })
            .collect();
        Ok(FaceOut::Facets(surf.kind(), out))
    }

    /// The facets of all of a closed surface (a sphere, a torus) for face
    /// `f` on it: a grid of its parameters spaced within the chord, the
    /// rows at a pole drawn into the pole's point (the face's vertex there,
    /// exactly).
    fn whole(&self, surf: &Surface, f: &Face) -> Option<Vec<Tri>> {
        use std::f64::consts::{FRAC_PI_2, PI};
        let (nu, nv, v0, wraps) = match surf {
            Surface::Sphere(_, r) => (
                self.segments(*r, TAU),
                self.segments(*r, PI),
                -FRAC_PI_2,
                false,
            ),
            Surface::Torus(_, big, small) => (
                self.segments(big + small, TAU),
                self.segments(*small, TAU),
                -PI,
                true,
            ),
            _ => return None,
        };
        let (nu, nv) = (nu.max(3), nv.max(if wraps { 3 } else { 2 }));
        let span = if wraps { TAU } else { PI };
        let vertices: Vec<P3> = f
            .bounds
            .iter()
            .filter_map(|b| match b {
                Bound::Vertex(v) => Some(self.m.vertices[*v]),
                Bound::Edges(_) => None,
            })
            .collect();
        let at = |i: usize, j: usize| -> P3 {
            let v = v0 + span * (j % if wraps { nv } else { nv + 1 }) as f64 / nv as f64;
            if !wraps && (j == 0 || j == nv) {
                let pole = surf.eval([0.0, v]);
                return vertices
                    .iter()
                    .copied()
                    .find(|&q| dist(q, pole) <= self.fit)
                    .unwrap_or(pole);
            }
            surf.eval([-PI + TAU * (i % nu) as f64 / nu as f64, v])
        };
        let mut out = Vec::with_capacity(2 * nu * nv);
        for i in 0..nu {
            for j in 0..nv {
                let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                // Counterclockwise in the parameters: the surface's normal.
                for t in [[a, b, c], [a, c, d]] {
                    if t[0] == t[1] || t[1] == t[2] || t[2] == t[0] {
                        continue;
                    }
                    out.push(if f.same_sense {
                        Tri::new(t[0], t[1], t[2])
                    } else {
                        Tri::new(t[0], t[2], t[1])
                    });
                }
            }
        }
        Some(out)
    }

    /// Points inside the parameter polygon `outer` less `holes`, spaced so
    /// the facets between them stay within the chord of the surface, and
    /// how far the surface runs along each parameter at most.
    fn inner_points(
        &self,
        surf: &Surface,
        outer: &[[f64; 2]],
        holes: &[&[[f64; 2]]],
    ) -> (Vec<[f64; 2]>, [f64; 2]) {
        if matches!(surf, Surface::Plane(_)) {
            return (Vec::new(), [1.0, 1.0]);
        }
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for q in outer {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        // How much the surface stretches and bends over the face's box, at
        // most.
        let range = [hi[0] - lo[0], hi[1] - lo[1]].map(|r| r.max(1e-12));
        let h = range.map(|r| 1e-4 * r);
        let (mut st, mut bend, mut twist) = ([1e-12f64; 2], [0.0f64; 2], 0.0f64);
        for i in 0..=4 {
            for j in 0..=4 {
                let q = [
                    lo[0] + range[0] * i as f64 / 4.0,
                    lo[1] + range[1] * j as f64 / 4.0,
                ];
                let b = surf.bend(q, h);
                for k in 0..2 {
                    st[k] = st[k].max(b.stretch[k]);
                    bend[k] = bend[k].max(b.bend[k]);
                }
                twist = twist.max(b.twist);
            }
        }
        // Per parameter the step whose chord stays within the tolerance
        // along it and which turns by no more than an edge's segment of a
        // full circle (a whole side where it runs straight), both shortened
        // where a cell's twist takes it beyond; then no facet longer in
        // space than FACET_ASPECT times its breadth.
        let turn = TAU / self.min_segments as f64;
        let mut steps = [0, 1].map(|k| {
            if bend[k] > 0.0 {
                (8.0 * self.chord / bend[k])
                    .sqrt()
                    .min(turn * st[k] / bend[k])
                    .min(range[k])
            } else {
                range[k]
            }
        });
        let across = steps[0] * steps[1] * twist / 4.0;
        if across > self.chord {
            let f = (self.chord / across).sqrt();
            steps = steps.map(|d| d * f);
        }
        for k in 0..2 {
            let other = steps[1 - k] * st[1 - k];
            if steps[k] * st[k] > FACET_ASPECT * other {
                steps[k] = FACET_ASPECT * other / st[k];
            }
        }
        let n = [0, 1].map(|k| (range[k] / steps[k]).ceil().clamp(1.0, 200.0) as usize);
        let mut out = Vec::new();
        for i in 1..n[0] {
            for j in 1..n[1] {
                let q = [
                    lo[0] + (hi[0] - lo[0]) * i as f64 / n[0] as f64,
                    lo[1] + (hi[1] - lo[1]) * j as f64 / n[1] as f64,
                ];
                let inside = contains(outer, q) && !holes.iter().any(|h| contains(h, q));
                // Clear of the boundary by a third of a step.
                let clear = std::iter::once(outer)
                    .chain(holes.iter().copied())
                    .all(|ring| seg_dist(ring, q, steps) > 0.33);
                if inside && clear {
                    out.push(q);
                }
            }
        }
        (out, st)
    }
}

/// Turns the faces whose facets run against most of their neighbours'
/// along the sides they share: a face the file leaves facing either way (a
/// sphere swept from a whole circle covers itself twice, facing out and
/// in) faces as the shell around it.
fn settle_orientation(faces: &mut [(SurfaceKind, Vec<Tri>)]) {
    type Side = [[u64; 3]; 2];
    let side = |t: &Tri, k: usize| -> Side {
        [t.v[k].map(f64::to_bits), t.v[(k + 1) % 3].map(f64::to_bits)]
    };
    let mut by_side: FxHashMap<Side, Vec<usize>> = FxHashMap::default();
    for (i, (_, tris)) in faces.iter().enumerate() {
        for t in tris {
            for k in 0..3 {
                by_side.entry(side(t, k)).or_default().push(i);
            }
        }
    }
    let turn: Vec<usize> = (0..faces.len())
        .filter(|&i| {
            let (mut with, mut against) = (0usize, 0usize);
            for t in &faces[i].1 {
                for k in 0..3 {
                    let [p, q] = side(t, k);
                    let others = |s: &Side| {
                        by_side
                            .get(s)
                            .into_iter()
                            .flatten()
                            .filter(|&&j| j != i)
                            .count()
                    };
                    with += others(&[q, p]);
                    against += others(&[p, q]);
                }
            }
            against > with
        })
        .collect();
    for i in turn {
        rapidmesh_exact::log::debug(
            "step.facets",
            format!("face {i} of a solid turned to face as its neighbours"),
        );
        for t in &mut faces[i].1 {
            *t = Tri::new(t.v[0], t.v[2], t.v[1]);
        }
    }
}

/// What a face comes to.
enum FaceOut {
    /// Its carrier and its triangles.
    Facets(SurfaceKind, Vec<Tri>),
    /// The sides of its bounds that cross in its parameters, in space.
    Crossing(Vec<(P3, P3)>),
}

/// The segments (indices into `segments`, sides between `pts`) that cross
/// another one properly: through each other's inside, not where they share
/// an end.
fn crossings(pts: &[[f64; 2]], segments: &[(usize, usize)]) -> Vec<usize> {
    let orient = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| polygon_orientation(&[a, b, c]);
    let span = |k: usize, d: usize| {
        let (a, b) = (pts[segments[k].0][d], pts[segments[k].1][d]);
        (a.min(b), a.max(b))
    };
    let mut order: Vec<usize> = (0..segments.len()).collect();
    order.sort_by(|&a, &b| span(a, 0).0.total_cmp(&span(b, 0).0));
    let mut out = Vec::new();
    for (n, &i) in order.iter().enumerate() {
        let (a, b) = (pts[segments[i].0], pts[segments[i].1]);
        for &j in &order[n + 1..] {
            if span(j, 0).0 > span(i, 0).1 {
                break;
            }
            let ((y0, y1), (z0, z1)) = (span(i, 1), span(j, 1));
            let (c, d) = (pts[segments[j].0], pts[segments[j].1]);
            if y1 < z0 || z1 < y0 || [a, b].iter().any(|p| *p == c || *p == d) {
                continue;
            }
            let apart = |s: Sign, t: Sign| {
                matches!(
                    (s, t),
                    (Sign::Positive, Sign::Negative) | (Sign::Negative, Sign::Positive)
                )
            };
            if apart(orient(a, b, c), orient(a, b, d)) && apart(orient(c, d, a), orient(c, d, b)) {
                out.extend([i, j]);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Edges between the same two vertices along the same path (a file may
/// model an arc twice, on two circles of opposite axes) take the samples
/// of the first of them, so the faces on either meet point for point.
fn share_coincident(m: &Model, samples: &mut [Samples], fit: f64) {
    let mut by_ends: FxHashMap<[usize; 2], Vec<usize>> = FxHashMap::default();
    for (i, e) in m.edges.iter().enumerate() {
        if e.ends[0] != e.ends[1] {
            let mut key = e.ends;
            key.sort_unstable();
            by_ends.entry(key).or_default().push(i);
        }
    }
    for group in by_ends.values().filter(|g| g.len() > 1) {
        for (k, &b) in group.iter().enumerate() {
            // Points within b's span of its curve lie on a's curve, within
            // a's span.
            let own = &m.curves[m.edges[b].curve];
            let (u0, u1) = (samples[b].ts[0], *samples[b].ts.last().unwrap());
            let probes: Vec<P3> = [0.25, 0.5, 0.75]
                .iter()
                .map(|w| own.eval(u0 + w * (u1 - u0)))
                .collect();
            let Some(&a) = group[..k].iter().find(|&&a| {
                let curve = &m.curves[m.edges[a].curve];
                let (t0, t1) = (samples[a].ts[0], *samples[a].ts.last().unwrap());
                probes.iter().all(|&p| {
                    let mut t = curve.param(p);
                    if let Some(period) = curve.period() {
                        t = near(t, 0.5 * (t0 + t1), period);
                    }
                    t0.min(t1) <= t && t <= t0.max(t1) && dist(curve.eval(t), p) <= fit
                })
            }) else {
                continue;
            };
            let mut pts = samples[a].pts.clone();
            if m.edges[a].ends[0] != m.edges[b].ends[0] {
                pts.reverse();
            }
            // The parameters of the points on b's own curve, onward from
            // its first.
            let curve = &m.curves[m.edges[b].curve];
            let (t0, t1) = (samples[b].ts[0], *samples[b].ts.last().unwrap());
            let n = pts.len() - 1;
            let mut ts = pts
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let guess = t0 + (t1 - t0) * i as f64 / n as f64;
                    match curve.period() {
                        Some(period) => near(curve.param(p), guess, period),
                        None => curve.param(p),
                    }
                })
                .collect::<Vec<_>>();
            ts[0] = t0;
            ts[n] = t1;
            samples[b] = Samples { ts, pts };
        }
    }
}

/// Twice the signed area of a ring.
fn area(r: &[[f64; 2]]) -> f64 {
    (0..r.len())
        .map(|i| {
            let (a, b) = (r[i], r[(i + 1) % r.len()]);
            a[0] * b[1] - a[1] * b[0]
        })
        .sum()
}

/// Joins the rings (`rings` in space, `uv` unwrapped in the parameters)
/// that wind once round a period of the surface into one that bounds a
/// region of the parameters. The two rims of a cylinder face with no seam
/// edge each run over a whole period, and neither encloses anything: they
/// become one ring along the first rim, across a seam to the second, back
/// along it and across the seam again. A rim alone closes through the pole
/// on the side of the face (a spherical cap): the face lies to the left of
/// its bounds seen from its normal, which is the surface's where
/// `same_sense`. The seam's points are those of the rims, twice, a period
/// apart.
fn join_windings(
    rings: &mut Vec<Vec<P3>>,
    uv: &mut Vec<Vec<[f64; 2]>>,
    periods: [Option<f64>; 2],
    poles: &[(usize, f64, P3)],
    same_sense: bool,
    seam: &dyn Fn([f64; 2], [f64; 2]) -> Vec<([f64; 2], P3)>,
) -> Result<(), String> {
    // The turns of each ring round each period, its closing step included.
    let turns = |r: &[[f64; 2]], k: usize| -> i64 {
        let Some(period) = periods[k] else {
            return 0;
        };
        // The steps along the ring add up to last - first; the closing one
        // takes the first point next to the last.
        let (a, b) = (r[0][k], r[r.len() - 1][k]);
        ((near(a, b, period) - a) / period).round() as i64
    };
    let winding: Vec<(usize, usize, i64)> = uv
        .iter()
        .enumerate()
        .flat_map(|(i, r)| (0..2).map(move |k| (i, k, turns(r, k))))
        .filter(|&(_, _, w)| w != 0)
        .collect();
    let Some(&(_, k, _)) = winding.first() else {
        return Ok(());
    };
    if winding.iter().any(|w| w.1 != k || w.2.abs() != 1) {
        return Err("a bound winds round the surface more than once".into());
    }
    let period = periods[k].unwrap_or(0.0);
    let shift = |q: [f64; 2], by: f64| -> [f64; 2] {
        let mut q = q;
        q[k] += by;
        q
    };
    // The ring from point `from` on round, ending on the copy of that point
    // a turn on.
    let open = |ring: &[P3], r: &[[f64; 2]], from: usize, w: i64| -> (Vec<P3>, Vec<[f64; 2]>) {
        let n = r.len();
        let turn = w as f64 * period;
        let pts = (0..=n).map(|t| ring[(from + t) % n]).collect();
        let q = (0..=n)
            .map(|t| {
                let i = from + t;
                if i < n {
                    r[i]
                } else {
                    shift(r[i - n], turn)
                }
            })
            .collect();
        (pts, q)
    };
    let (joined, joined_uv) = match *winding.as_slice() {
        [(a, _, wa), (b, _, wb)] => {
            if wa == wb {
                return Err("the rims of a band run the same way round".into());
            }
            // The seam is the shortest way between the rims: the pair of
            // their points nearest each other in space, so it crosses
            // neither (a rim with a step up in it has points of one angle
            // at two heights). The first rim opens there, the second is
            // moved into its turn.
            let (start, from) = (0..uv[a].len())
                .flat_map(|i| (0..uv[b].len()).map(move |j| (i, j)))
                .min_by(|&(i, j), &(x, y)| {
                    dist(rings[a][i], rings[b][j]).total_cmp(&dist(rings[a][x], rings[b][y]))
                })
                .unwrap_or((0, 0));
            let (mut pts, mut q) = open(&rings[a], &uv[a], start, wa);
            let end = q[q.len() - 1][k];
            let by = near(uv[b][from][k], end, period) - uv[b][from][k];
            let mut moved: Vec<[f64; 2]> = uv[b].iter().map(|&p| shift(p, by)).collect();
            // Where the other parameter wraps too (a torus), the second rim
            // goes to the face's side of the first, within a turn: a fillet
            // round a hole is the quarter between its rims, not the rest.
            let j = 1 - k;
            if let Some(turn) = periods[j] {
                let side = face_side(k, wa, same_sense);
                let mean = |r: &[[f64; 2]]| r.iter().map(|q| q[j]).sum::<f64>() / r.len() as f64;
                let (ma, mb) = (mean(&uv[a]), mean(&moved));
                let ahead = (side * (mb - ma)).rem_euclid(turn);
                let to = ma + side * ahead;
                moved.iter_mut().for_each(|q| q[j] += to - mb);
            }
            // Across the seam to the second rim, round it, and back across
            // the seam a turn on: the same points of the surface.
            let across = seam(q[q.len() - 1], moved[from]);
            let back: Vec<([f64; 2], P3)> = across
                .iter()
                .rev()
                .map(|&(t, p)| (shift(t, -(wa as f64) * period), p))
                .collect();
            let (pts_b, q_b) = open(&rings[b], &moved, from, wb);
            for (t, p) in across {
                q.push(t);
                pts.push(p);
            }
            pts.extend(pts_b);
            q.extend(q_b);
            for (t, p) in back {
                q.push(t);
                pts.push(p);
            }
            (pts, q)
        }
        [(a, _, wa)] => {
            let j = 1 - k;
            let side = face_side(k, wa, same_sense);
            let mean = uv[a].iter().map(|q| q[j]).sum::<f64>() / uv[a].len() as f64;
            let Some(&(_, value, pole)) = poles
                .iter()
                .filter(|p| p.0 == j && (p.1 - mean) * side > 0.0)
                .min_by(|x, y| (x.1 - mean).abs().total_cmp(&(y.1 - mean).abs()))
            else {
                return Err("a bound winds round the surface alone".into());
            };
            let (mut pts, mut q) = open(&rings[a], &uv[a], 0, wa);
            let (first, last) = (q[0], q[q.len() - 1]);
            let mut p0 = last;
            p0[j] = value;
            let mut p1 = first;
            p1[j] = value;
            // Along the seam to the pole and back from it a turn on.
            let across = seam(last, p0);
            let back: Vec<([f64; 2], P3)> = across
                .iter()
                .rev()
                .map(|&(t, p)| (shift(t, -(wa as f64) * period), p))
                .collect();
            for (t, p) in across {
                q.push(t);
                pts.push(p);
            }
            pts.extend([pole, pole]);
            q.extend([p0, p1]);
            for (t, p) in back {
                q.push(t);
                pts.push(p);
            }
            (pts, q)
        }
        _ => return Err("more than two bounds wind round the surface".into()),
    };
    // Rims that touch (a bore cut by another) meet where the seam starts:
    // a seam of no length leaves the point there twice, once is enough.
    let (mut joined, mut joined_uv) = (joined, joined_uv);
    let mut i = 0;
    while joined.len() > 3 && i < joined.len() {
        let j = (i + 1) % joined.len();
        if joined[i] == joined[j] && joined_uv[i] == joined_uv[j] {
            joined.remove(j);
            joined_uv.remove(j);
        } else {
            i += 1;
        }
    }
    let mut gone: Vec<usize> = winding.iter().map(|w| w.0).collect();
    gone.sort_unstable();
    for i in gone.into_iter().rev() {
        rings.remove(i);
        uv.remove(i);
    }
    rings.push(joined);
    uv.push(joined_uv);
    Ok(())
}

/// The longest a facet of a curved face is in space against its breadth.
const FACET_ASPECT: f64 = 4.0;

/// The sign of the other parameter on the side of a rim running `w` turns
/// round parameter `k` where its face lies: to the left of its bounds seen
/// from its normal, the surface's where `same_sense`. Left of a run along +u
/// is +v, left of one along +v is -u.
fn face_side(k: usize, w: i64, same_sense: bool) -> f64 {
    let left = if k == 0 { 1.0 } else { -1.0 } * w as f64;
    if same_sense {
        left
    } else {
        -left
    }
}

/// Flips the diagonal of each pair of facets `tris` (counterclockwise in
/// the parameters `p2`) whose smallest angle in space (`p3`) it enlarges,
/// where their quad is convex in the parameters and the new diagonal
/// stays `on_surface`, until none does. A
/// Delaunay triangulation in the parameters chooses the diagonals of
/// cells the parameters shear in space by chance, a grid of inner points
/// being cocircular throughout; in space one of them is the better. The
/// rings' sides have one facet on them and stay.
fn flip_to_shape(
    tris: &mut [[usize; 3]],
    p2: &[[f64; 2]],
    p3: &[P3],
    on_surface: impl Fn(usize, usize) -> bool,
) {
    let ccw = |a: usize, b: usize, c: usize| {
        let (p, q, r) = (p2[a], p2[b], p2[c]);
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) > 0.0
    };
    // The cosine of the smallest angle: the largest of the three.
    let worst = |t: [usize; 3]| -> f64 {
        (0..3)
            .map(|k| {
                let (o, a, b) = (p3[t[k]], p3[t[(k + 1) % 3]], p3[t[(k + 2) % 3]]);
                let (u, v) = (sub(a, o), sub(b, o));
                dot(u, v) / (dot(u, u) * dot(v, v)).sqrt().max(f64::MIN_POSITIVE)
            })
            .fold(f64::MIN, f64::max)
    };
    for _ in 0..32 {
        let mut at: FxHashMap<(usize, usize), (usize, usize)> = FxHashMap::default();
        for (t, tri) in tris.iter().enumerate() {
            for k in 0..3 {
                at.insert((tri[k], tri[(k + 1) % 3]), (t, k));
            }
        }
        let mut touched = vec![false; tris.len()];
        let mut flipped = false;
        for t in 0..tris.len() {
            for k in 0..3 {
                if touched[t] {
                    break;
                }
                let (a, b, c) = (tris[t][k], tris[t][(k + 1) % 3], tris[t][(k + 2) % 3]);
                let Some(&(s, _)) = at.get(&(b, a)) else {
                    continue;
                };
                if touched[s] {
                    continue;
                }
                let d = tris[s]
                    .iter()
                    .copied()
                    .find(|&v| v != a && v != b)
                    .unwrap_or(a);
                let (x, y) = ([a, d, c], [d, b, c]);
                if !ccw(x[0], x[1], x[2]) || !ccw(y[0], y[1], y[2]) {
                    continue;
                }
                let before = worst(tris[t]).max(worst(tris[s]));
                if worst(x).max(worst(y)) < before - 1e-9 && on_surface(c, d) {
                    tris[t] = x;
                    tris[s] = y;
                    touched[t] = true;
                    touched[s] = true;
                    flipped = true;
                }
            }
        }
        if !flipped {
            break;
        }
    }
}

/// Even-odd point in polygon.
fn contains(r: &[[f64; 2]], q: [f64; 2]) -> bool {
    let mut inside = false;
    for i in 0..r.len() {
        let (a, b) = (r[i], r[(i + 1) % r.len()]);
        if (a[1] > q[1]) != (b[1] > q[1]) {
            let x = a[0] + (q[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if q[0] < x {
                inside = !inside;
            }
        }
    }
    inside
}

/// The distance from `q` to the ring in steps (each direction scaled by
/// its step).
fn seg_dist(r: &[[f64; 2]], q: [f64; 2], steps: [f64; 2]) -> f64 {
    let s = |p: [f64; 2]| [p[0] / steps[0], p[1] / steps[1]];
    let q = s(q);
    (0..r.len())
        .map(|i| {
            let (a, b) = (s(r[i]), s(r[(i + 1) % r.len()]));
            let d = [b[0] - a[0], b[1] - a[1]];
            let l = d[0] * d[0] + d[1] * d[1];
            let t = if l > 0.0 {
                (((q[0] - a[0]) * d[0] + (q[1] - a[1]) * d[1]) / l).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let p = [a[0] + t * d[0] - q[0], a[1] + t * d[1] - q[1]];
            (p[0] * p[0] + p[1] * p[1]).sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

/// The carrier of a STEP edge curve, where it has one of the kinds the
/// B-rep keeps (a hyperbola or a parabola has none, nor an invalid
/// B-spline).
fn curve_kind(c: &Curve) -> Option<CurveKind> {
    match c {
        Curve::Line { p, d } => Some(CurveKind::Line {
            p0: *p,
            dir: unit(*d)?,
        }),
        Curve::Circle { f, r } => Some(CurveKind::Circle {
            center: f.o,
            axis: f.z,
            x: f.x,
            radius: *r,
        }),
        Curve::Ellipse { f, a, b } => Some(CurveKind::Ellipse {
            center: f.o,
            major: f.x,
            minor: f.y,
            a: *a,
            b: *b,
        }),
        Curve::Spline(sp) => Some(CurveKind::Nurbs(Arc::new(sp.clone()))),
        Curve::Hyperbola { .. } | Curve::Parabola { .. } => None,
    }
}

/// The bodies of `m`: one per solid, closed, oriented outward and placed
/// where the assembly puts it.
pub fn bodies(m: &Model, tol: Tolerance) -> Result<Vec<Body>, StepError> {
    let (lo, hi) = bbox(&m.vertices);
    let size = dist(lo, hi).max(1e-12);
    let mut t = Tess {
        m,
        chord: tol.chord * size,
        fit: 1e-4 * size,
        min_segments: tol.min_segments,
        samples: Vec::new(),
    };
    // Every face of the file once; the edges whose samples cross in a
    // face's parameters sampled twice as finely, until none do.
    let mut used: Vec<usize> = m
        .solids
        .iter()
        .flat_map(|s| s.faces.iter().copied())
        .collect();
    used.sort_unstable();
    used.dedup();
    let mut parts = vec![1usize; m.edges.len()];
    let mut facets: FxHashMap<usize, (SurfaceKind, Vec<Tri>)> = FxHashMap::default();
    for round in 0..=REFINE_ROUNDS {
        t.samples = m
            .edges
            .par_iter()
            .zip(&parts)
            .map(|(e, &k)| t.sample(e, k))
            .collect();
        share_coincident(m, &mut t.samples, t.fit);
        let check = round < REFINE_ROUNDS;
        let outs: Vec<FaceOut> = used
            .par_iter()
            .map(|&f| t.face(&m.faces[f], check))
            .collect::<Result<_, _>>()?;
        // The edges each pair of neighbouring samples lies on.
        let mut along: FxHashMap<[[u64; 3]; 2], Vec<usize>> = FxHashMap::default();
        let key = |p: P3, q: P3| {
            let (p, q) = (p.map(f64::to_bits), q.map(f64::to_bits));
            [p.min(q), p.max(q)]
        };
        for (e, s) in t.samples.iter().enumerate() {
            for w in s.pts.windows(2) {
                along.entry(key(w[0], w[1])).or_default().push(e);
            }
        }
        let mut refine: Vec<usize> = Vec::new();
        for out in &outs {
            if let FaceOut::Crossing(sides) = out {
                for &(p, q) in sides {
                    refine.extend(along.get(&key(p, q)).into_iter().flatten());
                }
            }
        }
        refine.sort_unstable();
        refine.dedup();
        if refine.is_empty() && outs.iter().any(|o| matches!(o, FaceOut::Crossing(_))) {
            // Sides no edge refines (straight ones): as they are.
            facets = used
                .par_iter()
                .map(|&f| Ok((f, t.face(&m.faces[f], false)?)))
                .collect::<Result<Vec<_>, StepError>>()?
                .into_iter()
                .filter_map(|(f, o)| match o {
                    FaceOut::Facets(k, tris) => Some((f, (k, tris))),
                    FaceOut::Crossing(_) => None,
                })
                .collect();
            break;
        }
        if refine.is_empty() {
            facets = used
                .iter()
                .zip(outs)
                .filter_map(|(&f, o)| match o {
                    FaceOut::Facets(k, tris) => Some((f, (k, tris))),
                    FaceOut::Crossing(_) => None,
                })
                .collect();
            break;
        }
        rapidmesh_exact::log::debug(
            "step.facets",
            format!(
                "round {round}: {} edges sampled finer where bounds cross",
                refine.len()
            ),
        );
        for e in refine {
            parts[e] *= 2;
        }
    }
    let mut out = Vec::with_capacity(m.solids.len());
    for solid in &m.solids {
        let mut faces: Vec<(SurfaceKind, Vec<Tri>)> =
            solid.faces.iter().map(|fi| facets[fi].clone()).collect();
        settle_orientation(&mut faces);
        let mut f = Faceted::new();
        for (kind, tris) in faces {
            let s = f.add_surface(kind);
            for tri in tris {
                f.push_tri(tri, s);
            }
        }
        // The solid's vertices are its corners: its edges end there.
        let mut corners: Vec<P3> = solid
            .faces
            .iter()
            .flat_map(|&fi| &m.faces[fi].bounds)
            .flat_map(|b| match b {
                Bound::Edges(es) => es
                    .iter()
                    .flat_map(|&(e, _)| m.edges[e].ends)
                    .collect::<Vec<_>>(),
                Bound::Vertex(v) => vec![*v],
            })
            .map(|v| m.vertices[v])
            .collect();
        corners.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        corners.dedup();
        f.corners = corners;
        // Its edge curves, with the samples its faces were built on: the
        // B-rep takes them as the carriers of its edges.
        let mut edges: Vec<usize> = solid
            .faces
            .iter()
            .flat_map(|&fi| &m.faces[fi].bounds)
            .flat_map(|b| match b {
                Bound::Edges(es) => es.iter().map(|&(e, _)| e).collect::<Vec<_>>(),
                Bound::Vertex(_) => Vec::new(),
            })
            .collect();
        edges.sort_unstable();
        edges.dedup();
        f.curves = edges
            .into_iter()
            .filter_map(|e| {
                Some(EdgeCurve {
                    kind: curve_kind(&m.curves[m.edges[e].curve])?,
                    points: t.samples[e].pts.clone(),
                })
            })
            .collect();
        if solid.placement != rapidmesh_geom::Frame::IDENTITY {
            f = f.transformed(solid.placement.linear, solid.placement.offset);
        }
        out.push(Body {
            name: solid.name.clone(),
            solid: f,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two sides through each other's inside cross; sides that only meet
    /// at a shared end, or touch, do not.
    #[test]
    fn crossing_sides_are_found() {
        let pts = [
            [0.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
        ];
        let segments = [(0, 1), (2, 3), (3, 4), (1, 5)];
        assert_eq!(crossings(&pts, &segments), vec![0, 1]);
    }

    /// A face of a closed shell facing in turns to face out like the rest.
    #[test]
    fn a_face_against_its_neighbours_turns() {
        let c = |i: usize| -> P3 { [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64] };
        // The cube's faces as quads facing out, x = 1 facing in.
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 5, 7, 3],
        ];
        let kind = SurfaceKind::Plane {
            point: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
        };
        let mut faces: Vec<(SurfaceKind, Vec<Tri>)> = quads
            .iter()
            .map(|q| {
                let [a, b, c_, d] = q.map(c);
                (kind.clone(), vec![Tri::new(a, b, c_), Tri::new(a, c_, d)])
            })
            .collect();
        let before = faces[5].1.clone();
        settle_orientation(&mut faces);
        for (i, (_, tris)) in faces.iter().enumerate().take(5) {
            assert_eq!(tris.len(), 2, "face {i}");
        }
        assert!(faces[5]
            .1
            .iter()
            .zip(&before)
            .all(|(t, u)| t.v[1] == u.v[2] && t.v[2] == u.v[1]));
    }
}
