//! The solids of a STEP model as faceted solids for the model: every edge
//! sampled once and shared by the faces on it, so each body is closed by
//! construction; every face triangulated in its surface's parameters (its
//! boundary there from the file's parameter curves, else projected), the
//! facets carrying the face's surface as their carrier. Edges and faces are
//! done in parallel.

use crate::entities::{Bound, Edge, Face, Model, StepError};
use rapidmesh_csg::Tri;
use rapidmesh_exact::vector::V3;
use rapidmesh_exact::vector::{bbox, dist, dot, len, segment_dist2, sub, wrap_near};
use rapidmesh_exact::Sign;
use rapidmesh_geom::cdt2::triangulate_constrained;
use rapidmesh_geom::chart::uv::{self, area, Join, Mark};
use rapidmesh_geom::{crossings, polygon_orientation, Curve, EdgeCurve, Faceted, Surface};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::f64::consts::TAU;

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
    pts: Vec<V3>,
}

struct Tess<'a> {
    m: &'a Model,
    /// The largest distance of a facet from its surface.
    chord: f64,
    /// How far a point from a parameter curve may lie off the edge.
    fit: f64,
    min_segments: usize,
    samples: Vec<Samples>,
    /// The model's size (the diagonal of its vertices' box).
    size: f64,
}

/// The parameters along an edge's curve from `p0` to `p1` (all of it
/// when `closed`), with its parameter or, not `forward`, against it.
fn span(curve: &Curve<3>, p0: V3, p1: V3, forward: bool, closed: bool) -> (f64, f64) {
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
            Curve::Ellipse { p, q, .. } => self.even(
                t0,
                t1,
                parts * self.segments(len(*p).min(len(*q)).max(1e-12), (t1 - t0).abs()),
            ),
            Curve::Nurbs(_) | Curve::Hyperbola { .. } | Curve::Parabola { .. } => {
                // Halved where the curve strays from its chord by more than
                // the tolerance, from two spans per control point (eight
                // on a conic).
                let start = parts
                    * match curve {
                        Curve::Nurbs(s) => 2 * s.ctrl.len(),
                        _ => 8,
                    };
                let mut ts = self.even(t0, t1, start);
                let mut i = 0;
                while i + 1 < ts.len() && ts.len() < 4000 * parts {
                    let (a, b) = (ts[i], ts[i + 1]);
                    let (pa, pb, pm) = (curve.eval(a), curve.eval(b), curve.eval(0.5 * (a + b)));
                    let mid: V3 = std::array::from_fn(|k| 0.5 * (pa[k] + pb[k]));
                    if dist(pm, mid) > self.chord {
                        ts.insert(i + 1, 0.5 * (a + b));
                    } else {
                        i += 1;
                    }
                }
                ts
            }
        };
        let mut pts: Vec<V3> = ts.iter().map(|&t| curve.eval(t)).collect();
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
        pts: &[V3],
        prev: Option<[f64; 2]>,
        taken: Option<usize>,
    ) -> (Vec<[f64; 2]>, Option<usize>) {
        let on: Vec<(usize, &Curve<2>)> = self.m.edges[e]
            .pcurves
            .iter()
            .enumerate()
            .filter(|(_, (s, _))| *s == si)
            .map(|(k, (_, c))| (k, c))
            .collect();
        let first = |c: &Curve<2>| c.eval(ts[0]);
        // A seam's second use in the face takes its other curve: where it
        // starts (at a pole, say) need not tell them apart.
        let free: Vec<(usize, &Curve<2>)> = match taken {
            Some(k) if on.len() > 1 => on.iter().copied().filter(|&(j, _)| j != k).collect(),
            _ => on.clone(),
        };
        let pick = match (free.len(), prev) {
            (0, _) => None,
            (_, None) => Some(free[0]),
            (_, Some(q)) => free.iter().copied().min_by(|a, b| {
                let d = |f: [f64; 2]| (f[0] - q[0]).powi(2) + (f[1] - q[1]).powi(2);
                d(first(a.1)).total_cmp(&d(first(b.1)))
            }),
        };
        if let Some((k, c)) = pick {
            let uv: Vec<[f64; 2]> = ts.iter().map(|&t| c.eval(t)).collect();
            let n = uv.len();
            let fits = [0, n / 2, n - 1]
                .iter()
                .all(|&i| dist(surf.eval(uv[i]), pts[i]) <= self.fit);
            if fits {
                // A sphere's latitude from the sphere: a file's curve may give
                // it a turn off (2 pi down, the same point by its sines).
                let uv = match surf {
                    Surface::Sphere { .. } => uv
                        .iter()
                        .zip(pts)
                        .map(|(&q, &p)| [q[0], surf.param(p)[1]])
                        .collect(),
                    _ => uv,
                };
                return (uv, Some(k));
            }
        }
        (pts.iter().map(|&p| surf.param(p)).collect(), None)
    }

    /// The facets of face `f`: its carrier and its triangles. With `check`,
    /// bounds whose sides cross in the surface's parameters (an edge's
    /// chords cutting across a neighbour's where the face is thinner than
    /// they stray from their curves) come back as those sides instead.
    fn face(&self, f: &Face, check: bool) -> Result<FaceOut, StepError> {
        let surf = &self.m.surfaces[f.surface];
        let b = match self.bounds(f, surf, f.surface)? {
            Shape::Whole(tris) => return Ok(FaceOut::Facets(Some(surf.clone()), tris)),
            Shape::Bounds(b) => b,
        };
        let part = Part {
            face: f,
            b: &b,
            front: f.same_sense,
        };
        Ok(match self.facets(surf, &[part], check)? {
            Ok(mut per) => FaceOut::Facets(Some(surf.clone()), per.pop().unwrap_or_default()),
            Err(sides) => FaceOut::Crossing(sides),
        })
    }

    /// The bounds of face `f` in the parameters of `surf` (surface `si`,
    /// the face's own or one it lies on): its rings in space and in the
    /// parameters, periodic parameters unwrapped and moved into the turn of
    /// the outer ring, points at a pole split, rings that wind round a
    /// period joined; or the facets of all of a closed surface where no
    /// edge bounds it.
    fn bounds(&self, f: &Face, surf: &Surface, si: usize) -> Result<Shape, StepError> {
        let err = |message: String| StepError { id: f.id, message };
        let mut rings: Vec<Vec<V3>> = Vec::new();
        let mut uv: Vec<Vec<[f64; 2]>> = Vec::new();
        let mut origin: Vec<Vec<Origin>> = Vec::new();
        // The parameter curve each edge took in this face.
        let mut taken: FxHashMap<usize, usize> = FxHashMap::default();
        for b in &f.bounds {
            let Bound::Edges(edges) = b else {
                continue;
            };
            let (mut ring, mut ring_uv, mut ring_origin) = (Vec::new(), Vec::new(), Vec::new());
            for &(e, along) in edges {
                let s = &self.samples[e];
                let (mut ts, mut pts) = (s.ts.clone(), s.pts.clone());
                if !along {
                    ts.reverse();
                    pts.reverse();
                }
                let (q, k) = self.edge_uv(
                    surf,
                    si,
                    e,
                    &ts,
                    &pts,
                    ring_uv.last().copied(),
                    taken.get(&e).copied(),
                );
                if let Some(k) = k {
                    taken.insert(e, k);
                }
                let n = pts.len() - 1;
                ring.extend_from_slice(&pts[..n]);
                ring_uv.extend_from_slice(&q[..n]);
                ring_origin.extend((0..n).map(|j| Origin {
                    edge: e,
                    at: if along { j } else { n - j },
                    along,
                }));
            }
            if ring.len() >= 3 {
                rings.push(ring);
                uv.push(ring_uv);
                origin.push(ring_origin);
            }
        }
        if rings.is_empty() {
            // Bounded by no edge (a pole's vertex at most): all of a closed
            // surface.
            return match self.whole(surf, f) {
                Some(tris) => Ok(Shape::Whole(tris)),
                None => Err(err("no bounds".into())),
            };
        }
        let marks: Vec<Vec<Mark<Option<Origin>>>> = rings
            .iter()
            .zip(&uv)
            .zip(&origin)
            .map(|((ring, r), o)| {
                ring.iter()
                    .zip(r)
                    .zip(o)
                    .map(|((&p, &uv), &o)| Mark {
                        tag: Some(o),
                        p,
                        uv,
                    })
                    .collect()
            })
            .collect();
        // The points of a seam between `a` and `b`: halved until each
        // piece keeps to the chord (the seam is no edge, nothing sampled it).
        let seam = |a: [f64; 2], b: [f64; 2]| -> Vec<([f64; 2], V3)> {
            let at = |w: f64| [a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1])];
            let mut parts = 1usize;
            while parts < 256
                && (0..parts).any(|i| {
                    let (w0, w1) = (i as f64 / parts as f64, (i + 1) as f64 / parts as f64);
                    let (p0, p1) = (surf.eval(at(w0)), surf.eval(at(w1)));
                    let mid: V3 = std::array::from_fn(|k| 0.5 * (p0[k] + p1[k]));
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
        let mut own = |_: V3| None;
        let side = |k: usize, w: i64, _: &[Mark<Option<Origin>>]| face_side(k, w, f.same_sense);
        let b = uv::bounds(
            surf.periods(),
            &surf.poles(),
            marks,
            1e-2 * self.fit,
            &mut Join {
                seam: &seam,
                own: &mut own,
                side: &side,
                pole_step: TAU / self.min_segments as f64,
                avoid: &[],
            },
        )
        .map_err(err)?;
        Ok(Shape::Bounds(Bounds {
            rings: b
                .rings
                .iter()
                .map(|r| r.iter().map(|m| m.p).collect())
                .collect(),
            uv: b
                .rings
                .iter()
                .map(|r| r.iter().map(|m| m.uv).collect())
                .collect(),
            outer: b.outer,
            origin: if b.plain {
                b.rings
                    .iter()
                    .map(|r| r.iter().filter_map(|m| m.tag).collect())
                    .collect()
            } else {
                Vec::new()
            },
        }))
    }

    /// The facets of the faces `parts` on `surf`, triangulated together in
    /// its parameters (one face, or faces of several bodies that touch on
    /// it: where they overlap they take the same triangles), each wound to
    /// its front: per part its triangles. With `check`, bounds whose sides
    /// cross or fold come back as those sides instead.
    #[allow(clippy::type_complexity)]
    fn facets(
        &self,
        surf: &Surface,
        parts: &[Part],
        check: bool,
    ) -> Result<Result<Vec<Vec<Tri>>, Vec<(V3, V3)>>, StepError> {
        let err = |message: String| StepError {
            id: parts[0].face.id,
            message,
        };
        // The rings' points, those of several parts at one place in space
        // taken once, and their sides.
        let mut pts3: Vec<V3> = Vec::new();
        let mut pts2: Vec<[f64; 2]> = Vec::new();
        let mut index: FxHashMap<[u64; 3], usize> = FxHashMap::default();
        let mut segments: Vec<(usize, usize)> = Vec::new();
        let mut seen: rustc_hash::FxHashSet<(usize, usize)> = Default::default();
        let shared = parts.len() > 1;
        for part in parts {
            for (ring, r) in part.b.rings.iter().zip(&part.b.uv) {
                let ids: Vec<usize> = ring
                    .iter()
                    .zip(r)
                    .map(|(&p, &q)| {
                        if shared {
                            if let Some(&i) = index.get(&p.map(f64::to_bits)) {
                                return i;
                            }
                            index.insert(p.map(f64::to_bits), pts3.len());
                        }
                        pts3.push(p);
                        pts2.push(q);
                        pts3.len() - 1
                    })
                    .collect();
                for i in 0..ids.len() {
                    let s = (ids[i], ids[(i + 1) % ids.len()]);
                    if !shared || (s.0 != s.1 && seen.insert((s.0.min(s.1), s.0.max(s.1)))) {
                        segments.push(s);
                    }
                }
            }
        }
        // Inside a part: in its outer ring and in none of its holes.
        let within = |part: &Part, q: [f64; 2]| {
            let b = part.b;
            contains(&b.uv[b.outer], q)
                && !b
                    .uv
                    .iter()
                    .enumerate()
                    .any(|(i, h)| i != b.outer && contains(h, q))
        };
        let rings: Vec<&[[f64; 2]]> = parts
            .iter()
            .flat_map(|p| p.b.uv.iter().map(|r| r.as_slice()))
            .collect();
        let frame: Vec<[f64; 2]> = parts
            .iter()
            .flat_map(|p| p.b.uv[p.b.outer].iter().copied())
            .collect();
        let (inner, stretch) = self.inner_points(
            surf,
            &frame,
            &|q| parts.iter().any(|p| within(p, q)),
            &rings,
        );
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
            parts.iter().any(|p| within(p, q))
        };
        // The sides of the bounds that cross, then those no facet takes
        // (a fold where two bounds leave a point of tangency side by side):
        // the longest of them, whose edges then take more samples (more of
        // both would fold alike).
        let longest = |sides: Vec<usize>| -> Vec<(V3, V3)> {
            let length = |k: usize| dist(pts3[segments[k].0], pts3[segments[k].1]);
            let most = sides.iter().map(|&k| length(k)).fold(0.0, f64::max);
            sides
                .into_iter()
                .filter(|&k| length(k) >= 0.95 * most)
                .map(|k| (pts3[segments[k].0], pts3[segments[k].1]))
                .collect()
        };
        if check {
            let crossed = crossings(&scaled, &segments);
            if !crossed.is_empty() {
                return Ok(Err(longest(crossed)));
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
                return Ok(Err(longest(bare)));
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
            let enclosed: f64 = rings
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
                    format!(
                        "face #{}: its bounds enclose nothing, no facets",
                        parts[0].face.id
                    ),
                );
                return Ok(Ok(vec![Vec::new(); parts.len()]));
            }
            return Err(err("its bounds do not triangulate in its parameters".into()));
        }
        // A new diagonal must keep to the chord: its middle off the surface
        // by no more than it; a side of a bound stays.
        let on_surface = |i: usize, j: usize| {
            let mid2 = [
                0.5 * (pts2[i][0] + pts2[j][0]),
                0.5 * (pts2[i][1] + pts2[j][1]),
            ];
            let mid3: V3 = std::array::from_fn(|k| 0.5 * (pts3[i][k] + pts3[j][k]));
            dist(surf.eval(mid2), mid3) <= self.chord
        };
        let fixed: rustc_hash::FxHashSet<(usize, usize)> = if shared {
            segments
                .iter()
                .map(|&(a, b)| (a.min(b), a.max(b)))
                .collect()
        } else {
            Default::default()
        };
        flip_to_shape(&mut tris, &pts2, &pts3, on_surface, &fixed);
        // A facet far off its surface is a face triangulated over the wrong
        // part of its parameters: said, for the diagnosis.
        let far = tris
            .iter()
            .map(|t| {
                let c: V3 =
                    std::array::from_fn(|k| (pts3[t[0]][k] + pts3[t[1]][k] + pts3[t[2]][k]) / 3.0);
                dist(c, surf.eval(surf.param(c)))
            })
            .fold(0.0f64, f64::max);
        if far > 4.0 * self.chord {
            rapidmesh_exact::log::debug(
                "step.facets",
                format!(
                    "face #{}: a facet {far:.3e} off its surface (chord {:.3e})",
                    parts[0].face.id, self.chord
                ),
            );
        }
        Ok(Ok(parts
            .iter()
            .map(|part| {
                tris.iter()
                    // A facet on a pole has two corners there: nothing in space.
                    .filter(|t| {
                        pts3[t[0]] != pts3[t[1]]
                            && pts3[t[1]] != pts3[t[2]]
                            && pts3[t[2]] != pts3[t[0]]
                    })
                    // Of several parts, each takes the facets in it.
                    .filter(|t| {
                        !shared || {
                            let c: [f64; 2] = std::array::from_fn(|k| {
                                (pts2[t[0]][k] + pts2[t[1]][k] + pts2[t[2]][k]) / 3.0
                            });
                            within(part, c)
                        }
                    })
                    .map(|t| {
                        let [a, b, c] = t.map(|i| pts3[i]);
                        // Counterclockwise in the parameters is the surface's
                        // normal; the part's front is it or its opposite.
                        let ccw = polygon_orientation(&t.map(|i| pts2[i])) == Sign::Positive;
                        if ccw == part.front {
                            Tri::new(a, b, c)
                        } else {
                            Tri::new(a, c, b)
                        }
                    })
                    .collect()
            })
            .collect()))
    }

    /// What each face of `used` comes to (in that order): the faces of a
    /// group of touching ones triangulated together, the others each alone.
    fn all_faces(
        &self,
        used: &[usize],
        groups: &[Vec<usize>],
        check: bool,
    ) -> Result<Vec<FaceOut>, StepError> {
        let mut out: FxHashMap<usize, FaceOut> = groups
            .par_iter()
            .map(|g| self.group(g, check))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        let alone: Vec<(usize, FaceOut)> = used
            .par_iter()
            .filter(|f| !out.contains_key(f))
            .map(|&f| Ok((f, self.face(&self.m.faces[f], check)?)))
            .collect::<Result<_, StepError>>()?;
        out.extend(alone);
        Ok(used
            .iter()
            .map(|f| out.remove(f).expect("every face"))
            .collect())
    }

    /// The facets of the faces `faces` (indices into the model's faces), of
    /// several bodies on one carrier, triangulated together in the
    /// parameters of the first one's surface, so where they overlap they
    /// take the same triangles. Faces whose bounds need a pole or a seam
    /// are triangulated each on its own.
    fn group(&self, faces: &[usize], check: bool) -> Result<Vec<(usize, FaceOut)>, StepError> {
        let each = || -> Result<Vec<(usize, FaceOut)>, StepError> {
            faces
                .iter()
                .map(|&fi| Ok((fi, self.face(&self.m.faces[fi], check)?)))
                .collect()
        };
        let Some(bounds) = self.group_bounds(faces)? else {
            return each();
        };
        let (si, surf) = self.carrier(faces);
        let parts: Vec<Part> = faces
            .iter()
            .zip(&bounds)
            .map(|(&fi, b)| {
                let f = &self.m.faces[fi];
                Part {
                    face: f,
                    b,
                    front: f.same_sense == self.agrees(f.surface, si, b.rings[b.outer][0]),
                }
            })
            .collect();
        Ok(match self.facets(surf, &parts, check)? {
            Ok(per) => faces
                .iter()
                .zip(per)
                .map(|(&fi, tris)| {
                    let kind = Some(self.m.surfaces[self.m.faces[fi].surface].clone());
                    (fi, FaceOut::Facets(kind, tris))
                })
                .collect(),
            Err(sides) => faces
                .iter()
                .enumerate()
                .map(|(k, &fi)| {
                    let sides = if k == 0 { sides.clone() } else { Vec::new() };
                    (fi, FaceOut::Crossing(sides))
                })
                .collect(),
        })
    }

    /// The surface of the first face of a group: the parameters it is
    /// triangulated and imprinted in.
    fn carrier(&self, faces: &[usize]) -> (usize, &Surface) {
        let si = self.m.faces[faces[0]].surface;
        (si, &self.m.surfaces[si])
    }

    /// Whether surface `a` faces the way surface `b` does at `p` (on both).
    fn agrees(&self, a: usize, b: usize, p: V3) -> bool {
        let (sa, sb) = (&self.m.surfaces[a], &self.m.surfaces[b]);
        dot(sa.normal(sa.param(p)), sb.normal(sb.param(p))) >= 0.0
    }

    /// The bounds of each face of a group in the parameters of its carrier,
    /// every face moved into the turn of the first; `None` where one needs
    /// a pole or a seam (a full turn: its points would meet themselves a
    /// period apart).
    fn group_bounds(&self, faces: &[usize]) -> Result<Option<Vec<Bounds>>, StepError> {
        let (si, surf) = self.carrier(faces);
        let mut out: Vec<Bounds> = Vec::with_capacity(faces.len());
        for &fi in faces {
            match self.bounds(&self.m.faces[fi], surf, si)? {
                Shape::Bounds(b) if !b.origin.is_empty() => {
                    // A seam: its points twice, a period apart.
                    let mut seen: rustc_hash::FxHashSet<[u64; 3]> = Default::default();
                    if !b
                        .rings
                        .iter()
                        .flatten()
                        .all(|p| seen.insert(p.map(f64::to_bits)))
                    {
                        return Ok(None);
                    }
                    out.push(b);
                }
                _ => return Ok(None),
            }
        }
        for (k, period) in surf.periods().into_iter().enumerate() {
            let Some(period) = period else {
                continue;
            };
            let m0 = mean(&out[0].uv[out[0].outer], k);
            for b in &mut out[1..] {
                let m = mean(&b.uv[b.outer], k);
                let shift = wrap_near(m, m0, period) - m;
                b.uv.iter_mut().flatten().for_each(|q| q[k] += shift);
            }
        }
        Ok(Some(out))
    }

    /// What the faces of a group must share before they are triangulated
    /// together: where a point of one face's bounds lies on a side of
    /// another's, or two sides cross, the point goes into the edges of
    /// both (so the bodies' other faces on those edges take it too); a
    /// point within the tolerance of another's is moved onto it.
    fn imprints(&self, faces: &[usize], solid_of: &[usize]) -> Result<Vec<Imprint>, StepError> {
        let Some(bounds) = self.group_bounds(faces)? else {
            return Ok(Vec::new());
        };
        let (_, surf) = self.carrier(faces);
        // Lengths in the parameters: scaled by how far the surface runs
        // along each at the first point.
        let q0 = bounds[0].uv[bounds[0].outer][0];
        let st = surf.bend(q0).stretch.map(|s| s.max(1e-12));
        let sc = |q: [f64; 2]| [q[0] * st[0], q[1] * st[1]];
        let tol = 1e-7 * self.size;
        // Every side: the scaled ends, the ends in space, and the edge with
        // the indices of its samples there.
        struct Side {
            a: [f64; 2],
            b: [f64; 2],
            pa: V3,
            pb: V3,
            edge: usize,
            after: usize,
        }
        let sides: Vec<Vec<Side>> = bounds
            .iter()
            .map(|bd| {
                let mut out = Vec::new();
                for ((ring, r), o) in bd.rings.iter().zip(&bd.uv).zip(&bd.origin) {
                    let n = ring.len();
                    for i in 0..n {
                        let j = (i + 1) % n;
                        let at = if o[i].along { o[i].at } else { o[i].at - 1 };
                        out.push(Side {
                            a: sc(r[i]),
                            b: sc(r[j]),
                            pa: ring[i],
                            pb: ring[j],
                            edge: o[i].edge,
                            after: at,
                        });
                    }
                }
                out
            })
            .collect();
        let points = |k: usize| {
            bounds[k]
                .rings
                .iter()
                .zip(&bounds[k].uv)
                .flat_map(|(ring, r)| ring.iter().zip(r).map(|(&p, &q)| (p, sc(q))))
                .collect::<Vec<_>>()
        };
        let mut out = Vec::new();
        for x in 0..faces.len() {
            for y in 0..faces.len() {
                if solid_of[faces[x]] == solid_of[faces[y]] {
                    continue;
                }
                for (p, q) in points(x) {
                    for s in &sides[y] {
                        let d = [s.b[0] - s.a[0], s.b[1] - s.a[1]];
                        let l2 = d[0] * d[0] + d[1] * d[1];
                        if l2 <= 0.0 {
                            continue;
                        }
                        let w = ((q[0] - s.a[0]) * d[0] + (q[1] - s.a[1]) * d[1]) / l2;
                        let foot = [s.a[0] + w * d[0], s.a[1] + w * d[1]];
                        if (foot[0] - q[0]).hypot(foot[1] - q[1]) > tol {
                            continue;
                        }
                        // On an end: the same point (moved onto it where
                        // apart by less than the tolerance), else inside.
                        for end in [s.pa, s.pb] {
                            if p != end && dist(p, end) <= tol && x < y {
                                out.push(Imprint::Move { from: end, to: p });
                            }
                        }
                        if dist(p, s.pa) > tol && dist(p, s.pb) > tol && w > 0.0 && w < 1.0 {
                            out.push(Imprint::Insert {
                                edge: s.edge,
                                after: s.after,
                                p,
                            });
                        }
                    }
                }
                if x < y {
                    for s in &sides[x] {
                        for u in &sides[y] {
                            let o = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| {
                                polygon_orientation(&[a, b, c])
                            };
                            let apart = |s: Sign, t: Sign| {
                                matches!(
                                    (s, t),
                                    (Sign::Positive, Sign::Negative)
                                        | (Sign::Negative, Sign::Positive)
                                )
                            };
                            if !(apart(o(s.a, s.b, u.a), o(s.a, s.b, u.b))
                                && apart(o(u.a, u.b, s.a), o(u.a, u.b, s.b)))
                            {
                                continue;
                            }
                            // The crossing, on the carrier.
                            let (d, e) = (
                                [s.b[0] - s.a[0], s.b[1] - s.a[1]],
                                [u.b[0] - u.a[0], u.b[1] - u.a[1]],
                            );
                            let den = d[0] * e[1] - d[1] * e[0];
                            let w = ((u.a[0] - s.a[0]) * e[1] - (u.a[1] - s.a[1]) * e[0]) / den;
                            let c = [(s.a[0] + w * d[0]) / st[0], (s.a[1] + w * d[1]) / st[1]];
                            let p = surf.eval(c);
                            out.push(Imprint::Insert {
                                edge: s.edge,
                                after: s.after,
                                p,
                            });
                            out.push(Imprint::Insert {
                                edge: u.edge,
                                after: u.after,
                                p,
                            });
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// The facets of all of a closed surface (a sphere, a torus) for face
    /// `f` on it: a grid of its parameters spaced within the chord, the
    /// rows at a pole drawn into the pole's point (the face's vertex there,
    /// exactly).
    fn whole(&self, surf: &Surface, f: &Face) -> Option<Vec<Tri>> {
        use std::f64::consts::{FRAC_PI_2, PI};
        let (nu, nv, v0, wraps) = match surf {
            Surface::Sphere { radius, .. } => (
                self.segments(*radius, TAU),
                self.segments(*radius, PI),
                -FRAC_PI_2,
                false,
            ),
            Surface::Torus { major, minor, .. } => (
                self.segments(major + minor, TAU),
                self.segments(*minor, TAU),
                -PI,
                true,
            ),
            _ => return None,
        };
        let (nu, nv) = (nu.max(3), nv.max(if wraps { 3 } else { 2 }));
        let span = if wraps { TAU } else { PI };
        let vertices: Vec<V3> = f
            .bounds
            .iter()
            .filter_map(|b| match b {
                Bound::Vertex(v) => Some(self.m.vertices[*v]),
                Bound::Edges(_) => None,
            })
            .collect();
        let at = |i: usize, j: usize| -> V3 {
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
        frame: &[[f64; 2]],
        inside: &dyn Fn([f64; 2]) -> bool,
        rings: &[&[[f64; 2]]],
    ) -> (Vec<[f64; 2]>, [f64; 2]) {
        if matches!(surf, Surface::Plane(_)) {
            return (Vec::new(), [1.0, 1.0]);
        }
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for q in frame {
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        // How much the surface stretches and bends over the face's box, at
        // most.
        let range = [hi[0] - lo[0], hi[1] - lo[1]].map(|r| r.max(1e-12));
        let (mut st, mut bend, mut twist) = ([1e-12f64; 2], [0.0f64; 2], 0.0f64);
        for i in 0..=4 {
            for j in 0..=4 {
                let q = [
                    lo[0] + range[0] * i as f64 / 4.0,
                    lo[1] + range[1] * j as f64 / 4.0,
                ];
                let b = surf.bend(q);
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
                // Clear of the boundary by a third of a step.
                let clear = rings.iter().all(|ring| seg_dist(ring, q, steps) > 0.33);
                if inside(q) && clear {
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
fn settle_orientation(faces: &mut [(Option<Surface>, Vec<Tri>)]) {
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

/// Where a point of a face's bounds comes from: edge `edge`'s sample `at`
/// (in the edge's own order), the edge run along itself or against.
#[derive(Clone, Copy)]
struct Origin {
    edge: usize,
    at: usize,
    along: bool,
}

/// A face's bounds in the parameters of a surface: its rings in space and
/// there, the outer one, and per point where it comes from (empty where a
/// pole or a seam added points).
struct Bounds {
    rings: Vec<Vec<V3>>,
    uv: Vec<Vec<[f64; 2]>>,
    outer: usize,
    origin: Vec<Vec<Origin>>,
}

/// What a face comes to before it is triangulated: its bounds, or all of a
/// closed surface's facets.
enum Shape {
    Bounds(Bounds),
    Whole(Vec<Tri>),
}

/// A face triangulated with others on one surface: its bounds there and
/// whether its front is the surface's normal.
struct Part<'a> {
    face: &'a Face,
    b: &'a Bounds,
    front: bool,
}

/// A change to the edges' samples so faces of several bodies on one
/// carrier share their points.
#[derive(Clone, Copy, Debug)]
enum Imprint {
    /// Point `p` into edge `edge` after its sample `after`.
    Insert { edge: usize, after: usize, p: V3 },
    /// Every sample at `from` to `to`.
    Move { from: V3, to: V3 },
}

/// The mean of parameter `k` over a ring.
fn mean(r: &[[f64; 2]], k: usize) -> f64 {
    r.iter().map(|q| q[k]).sum::<f64>() / r.len() as f64
}

/// Rounds of imprints before the faces of a group are triangulated: each
/// may add points the next finds on another face's sides.
const IMPRINT_ROUNDS: usize = 4;

/// The groups of faces that touch: faces of different bodies on one
/// curved carrier (a cylinder, a cone, a sphere, a torus) whose boxes
/// meet, the points of each on the other's surface; with the faces of
/// the same bodies that touch those. `samples` the edges' samples, `tol`
/// how far off a surface a point counts as on it.
fn contact_groups(m: &Model, samples: &[Samples], solid_of: &[usize], tol: f64) -> Vec<Vec<usize>> {
    let curved = |f: &Face| {
        matches!(
            m.surfaces[f.surface],
            Surface::Cylinder { .. }
                | Surface::Cone { .. }
                | Surface::Sphere { .. }
                | Surface::Torus { .. }
        )
    };
    // Per candidate face: some points of its bounds and their box.
    let candidates: Vec<(usize, Vec<V3>, V3, V3)> = (0..m.faces.len())
        .filter(|&fi| solid_of[fi] != usize::MAX && curved(&m.faces[fi]))
        .filter_map(|fi| {
            let pts: Vec<V3> = m.faces[fi]
                .bounds
                .iter()
                .flat_map(|b| match b {
                    Bound::Edges(es) => es
                        .iter()
                        .flat_map(|&(e, _)| samples[e].pts.iter().copied())
                        .collect(),
                    Bound::Vertex(_) => Vec::new(),
                })
                .collect();
            if pts.is_empty() {
                return None;
            }
            let (lo, hi) = bbox(&pts);
            let step = pts.len().div_ceil(16);
            Some((fi, pts.into_iter().step_by(step).collect(), lo, hi))
        })
        .collect();
    let on = |s: &Surface, pts: &[V3]| pts.iter().all(|&p| dist(s.eval(s.param(p)), p) <= tol);
    let mut parent: Vec<usize> = (0..m.faces.len()).collect();
    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut grouped = vec![false; m.faces.len()];
    for (i, (a, pa, la, ha)) in candidates.iter().enumerate() {
        for (b, pb, lb, hb) in &candidates[i + 1..] {
            if solid_of[*a] == solid_of[*b]
                || (0..3).any(|k| la[k] > hb[k] + tol || lb[k] > ha[k] + tol)
            {
                continue;
            }
            let (sa, sb) = (
                &m.surfaces[m.faces[*a].surface],
                &m.surfaces[m.faces[*b].surface],
            );
            if on(sa, pb) && on(sb, pa) {
                let (ra, rb) = (root(&mut parent, *a), root(&mut parent, *b));
                parent[ra] = rb;
                grouped[*a] = true;
                grouped[*b] = true;
            }
        }
    }
    let mut groups: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for fi in 0..m.faces.len() {
        if grouped[fi] {
            groups.entry(root(&mut parent, fi)).or_default().push(fi);
        }
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().collect();
    out.sort_unstable();
    out
}

/// Applies `imprints` to the edges' samples: moves first, then each
/// edge's new points in order along it. Returns how many samples changed.
fn apply_imprints(m: &Model, samples: &mut [Samples], imprints: &[Imprint], tol: f64) -> usize {
    let mut changed = 0;
    for im in imprints {
        if let Imprint::Move { from, to } = *im {
            for s in samples.iter_mut() {
                for p in &mut s.pts {
                    if *p == from {
                        *p = to;
                        changed += 1;
                    }
                }
            }
        }
    }
    let mut by_edge: FxHashMap<usize, Vec<(usize, V3)>> = FxHashMap::default();
    for im in imprints {
        if let Imprint::Insert { edge, after, p } = *im {
            by_edge.entry(edge).or_default().push((after, p));
        }
    }
    let mut edges: Vec<usize> = by_edge.keys().copied().collect();
    edges.sort_unstable();
    for e in edges {
        let curve = &m.curves[m.edges[e].curve];
        let s = &mut samples[e];
        let mut add: Vec<(f64, V3)> = Vec::new();
        for &(after, p) in &by_edge[&e] {
            if after + 1 >= s.pts.len()
                || s.pts.iter().any(|&q| dist(q, p) <= tol)
                || add.iter().any(|&(_, q)| dist(q, p) <= tol)
            {
                continue;
            }
            // Its parameter on the edge's curve, between the samples it
            // goes between.
            let (t0, t1) = (s.ts[after], s.ts[after + 1]);
            let mut t = curve.param(p);
            if let Some(period) = curve.period() {
                t = wrap_near(t, 0.5 * (t0 + t1), period);
            }
            let t = t.clamp(t0.min(t1), t0.max(t1));
            add.push((t, p));
        }
        if add.is_empty() {
            continue;
        }
        changed += add.len();
        let dir = (*s.ts.last().unwrap() - s.ts[0]).signum();
        let mut all: Vec<(f64, V3)> = s.ts.iter().copied().zip(s.pts.iter().copied()).collect();
        all.extend(add);
        let (first, last) = (all[0], all[s.ts.len() - 1]);
        let mut middle: Vec<(f64, V3)> = all
            .into_iter()
            .filter(|&(_, p)| p != first.1 && p != last.1)
            .collect();
        middle.sort_by(|a, b| (dir * a.0).total_cmp(&(dir * b.0)));
        s.ts = std::iter::once(first.0)
            .chain(middle.iter().map(|x| x.0))
            .chain(std::iter::once(last.0))
            .collect();
        s.pts = std::iter::once(first.1)
            .chain(middle.iter().map(|x| x.1))
            .chain(std::iter::once(last.1))
            .collect();
    }
    changed
}

/// What a face comes to.
enum FaceOut {
    /// Its carrier and its triangles.
    Facets(Option<Surface>, Vec<Tri>),
    /// The sides of its bounds that cross in its parameters, in space.
    Crossing(Vec<(V3, V3)>),
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
            let probes: Vec<V3> = [0.25, 0.5, 0.75]
                .iter()
                .map(|w| own.eval(u0 + w * (u1 - u0)))
                .collect();
            let Some(&a) = group[..k].iter().find(|&&a| {
                let curve = &m.curves[m.edges[a].curve];
                let (t0, t1) = (samples[a].ts[0], *samples[a].ts.last().unwrap());
                probes.iter().all(|&p| {
                    let mut t = curve.param(p);
                    if let Some(period) = curve.period() {
                        t = wrap_near(t, 0.5 * (t0 + t1), period);
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
                        Some(period) => wrap_near(curve.param(p), guess, period),
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
    p3: &[V3],
    on_surface: impl Fn(usize, usize) -> bool,
    fixed: &rustc_hash::FxHashSet<(usize, usize)>,
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
                if touched[s] || fixed.contains(&(a.min(b), a.max(b))) {
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
        .map(|i| segment_dist2(q, s(r[i]), s(r[(i + 1) % r.len()])).sqrt())
        .fold(f64::INFINITY, f64::min)
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
        size,
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
    let mut facets: FxHashMap<usize, (Option<Surface>, Vec<Tri>)> = FxHashMap::default();
    // The body of each face (none for a face of a moved body or of
    // several): faces of different bodies that touch on a carrier are
    // triangulated together.
    let mut solid_of = vec![usize::MAX; m.faces.len()];
    let mut users = vec![0usize; m.faces.len()];
    for (k, solid) in m.solids.iter().enumerate() {
        for &fi in &solid.faces {
            users[fi] += 1;
            if solid.placement == rapidmesh_exact::vector::Affine::IDENTITY {
                solid_of[fi] = k;
            }
        }
    }
    for (fi, &n) in users.iter().enumerate() {
        if n != 1 {
            solid_of[fi] = usize::MAX;
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for round in 0..=REFINE_ROUNDS {
        t.samples = m
            .edges
            .par_iter()
            .zip(&parts)
            .map(|(e, &k)| t.sample(e, k))
            .collect();
        share_coincident(m, &mut t.samples, t.fit);
        if round == 0 {
            groups = contact_groups(m, &t.samples, &solid_of, 1e-6 * size);
            rapidmesh_exact::log::debug(
                "step.contact",
                format!(
                    "{} groups of touching faces, {} faces",
                    groups.len(),
                    groups.iter().map(Vec::len).sum::<usize>()
                ),
            );
        }
        for _ in 0..IMPRINT_ROUNDS {
            let imprints: Vec<Imprint> = groups
                .par_iter()
                .map(|g| t.imprints(g, &solid_of))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            if apply_imprints(m, &mut t.samples, &imprints, 1e-7 * size) == 0 {
                break;
            }
        }
        let check = round < REFINE_ROUNDS;
        let outs = t.all_faces(&used, &groups, check)?;
        // The edges each pair of neighbouring samples lies on.
        let mut along: FxHashMap<[[u64; 3]; 2], Vec<usize>> = FxHashMap::default();
        let key = |p: V3, q: V3| {
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
                .iter()
                .copied()
                .zip(t.all_faces(&used, &groups, false)?)
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
        let mut faces: Vec<(Option<Surface>, Vec<Tri>)> =
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
        // (The samples' ends: a vertex moved onto another body's is where
        // the edges end.)
        let mut corners: Vec<V3> = solid
            .faces
            .iter()
            .flat_map(|&fi| &m.faces[fi].bounds)
            .flat_map(|b| match b {
                Bound::Edges(es) => es
                    .iter()
                    .flat_map(|&(e, _)| {
                        let s = &t.samples[e].pts;
                        [s[0], s[s.len() - 1]]
                    })
                    .collect::<Vec<_>>(),
                Bound::Vertex(v) => vec![m.vertices[*v]],
            })
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
            .map(|e| EdgeCurve {
                curve: m.curves[m.edges[e].curve].clone(),
                points: t.samples[e].pts.clone(),
            })
            .collect();
        if solid.placement != rapidmesh_exact::vector::Affine::IDENTITY {
            f = f.transformed(&solid.placement);
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
        let c = |i: usize| -> V3 { [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64] };
        // The cube's faces as quads facing out, x = 1 facing in.
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 5, 7, 3],
        ];
        let kind = Surface::plane([0.0; 3], [0.0, 0.0, 1.0]);
        let mut faces: Vec<(Option<Surface>, Vec<Tri>)> = quads
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
