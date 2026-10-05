//! Edges and faces: every B-rep edge sampled once by the size field, every
//! face meshed alone in 2D with the samples of its edges as its fixed
//! boundary. Two faces on an edge take the same samples, so the faces of a
//! model close up without looking at each other.

pub(crate) mod chart;
pub(crate) mod periodic;
pub(crate) mod planar;
pub(crate) mod remesh;
pub(crate) mod topology;

use crate::curve::{distribute_floored, Curve, Guided, PolylineCurve};
use crate::params::MeshParams;
use crate::sizing::tree::DomainTree;
use crate::surface::planar::{mesh_constrained, PipRows};
use rapidmesh_brep::{Brep, Model};
use rapidmesh_exact::vector::{add, bbox, cross, dist2, dot, sub};
use rapidmesh_exact::vector::{V2, V3};
use rapidmesh_geom::chart::{Chart, Domain, Slot};
use rapidmesh_geom::crossings;
use rapidmesh_geom::grid::HashGrid;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// The surface mesh of a model, by B-rep entity.
#[derive(Debug, Clone, Default)]
pub struct Boundary {
    pub points: Vec<V3>,
    /// Per B-rep edge, its points from `ends[0]` to `ends[1]` (a closed
    /// edge ends on its first point).
    pub edges: Vec<Vec<u32>>,
    /// Per B-rep face, its triangles, wound so their normal points into the
    /// face's front region (`regions[0]`).
    pub faces: Vec<Vec<[u32; 3]>>,
}

/// Why a model has no boundary mesh (yet).
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryError {
    /// A face on a curved carrier no chart covers (yet), with the carrier's
    /// kind and what stopped it.
    Curved {
        face: u32,
        kind: &'static str,
        why: &'static str,
    },
    /// A face whose 2D mesh came out empty.
    Empty { face: u32 },
    /// A periodic face whose original's mesh does not land on its edges.
    Periodic { face: u32 },
    /// A face meshed with a hole, a fold or a segment of its edges left out.
    Broken {
        face: u32,
        kind: &'static str,
        open: usize,
        over: usize,
        unused: usize,
    },
    /// Samples the edges need before a face can be meshed (edge, arc
    /// length); the boundary takes them and meshes again.
    Refine(Vec<(u32, f64)>),
    /// Edges the regions miss that grow in number round after round:
    /// splitting them makes more than it mends; the faces missing most.
    Diverged { left: usize, near: Vec<Near> },
}

/// A face where the boundary edges the regions miss gather: its carrier,
/// how many it misses, a point of one, its shortest edge and the size
/// there.
#[derive(Debug, Clone, PartialEq)]
pub struct Near {
    pub face: u32,
    pub kind: &'static str,
    pub missing: usize,
    pub at: V3,
    pub shortest: f64,
    pub size: f64,
}

impl std::fmt::Display for BoundaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundaryError::Curved { face, kind, why } => {
                write!(f, "face {face} is curved ({kind}) beyond its chart: {why}")
            }
            BoundaryError::Empty { face } => write!(f, "face {face} has an empty mesh"),
            BoundaryError::Periodic { face } => {
                write!(f, "periodic face {face} does not match its original")
            }
            BoundaryError::Refine(at) => write!(f, "{} edge samples still needed", at.len()),
            BoundaryError::Diverged { left, near } => {
                write!(f, "{left} edges the regions miss, more each round")?;
                for n in near {
                    write!(
                        f,
                        "; face {} ({}) misses {} at {:?}, shortest edge {:.3e}, size {:.3e}",
                        n.face, n.kind, n.missing, n.at, n.shortest, n.size
                    )?;
                }
                Ok(())
            }
            BoundaryError::Broken {
                face,
                kind,
                open,
                over,
                unused,
            } => write!(
                f,
                "face {face} ({kind}) meshed with {open} open edges, {over} edges on more than two triangles and {unused} segments left out"
            ),
        }
    }
}

impl std::error::Error for BoundaryError {}

impl Boundary {
    /// The directed edges of the boundary of region `r` without a partner
    /// in the opposite direction (none when it is a closed surface) and a
    /// point of one. Sheets inside the region count on both sides, so they
    /// never open it.
    pub fn open_edges(&self, brep: &Brep, r: u32) -> (usize, Option<V3>) {
        let mut count: FxHashMap<(u32, u32), i64> = FxHashMap::default();
        for (face, tris) in brep.faces.iter().zip(&self.faces) {
            let [front, back] = face.regions.map(|x| x.0);
            let sides: &[bool] = match (front == r, back == r) {
                (true, true) => &[true, false],
                // The normal points into the front: outward seen from the back.
                (true, false) => &[true],
                (false, true) => &[false],
                (false, false) => &[],
            };
            for &flip in sides {
                for t in tris {
                    let t = if flip { [t[0], t[2], t[1]] } else { *t };
                    for k in 0..3 {
                        let (a, b) = (t[k], t[(k + 1) % 3]);
                        *count.entry((a.min(b), a.max(b))).or_default() +=
                            if a < b { 1 } else { -1 };
                    }
                }
            }
        }
        let mut open: Vec<(u32, u32)> = count
            .iter()
            .filter(|(_, &c)| c != 0)
            .map(|(&e, _)| e)
            .collect();
        open.sort_unstable();
        if !open.is_empty() {
            // Which faces touch each open edge, for the diagnosis.
            let touching = |e: (u32, u32)| -> Vec<(usize, [u32; 2], i64)> {
                self.faces
                    .iter()
                    .enumerate()
                    .filter_map(|(f, tris)| {
                        let dir: i64 = tris
                            .iter()
                            .flat_map(|t| (0..3).map(move |k| (t[k], t[(k + 1) % 3])))
                            .filter(|&(a, b)| (a.min(b), a.max(b)) == e)
                            .map(|(a, _)| if a == e.0 { 1 } else { -1 })
                            .sum();
                        let any = tris.iter().any(|t| {
                            (0..3).any(|k| {
                                let (a, b) = (t[k], t[(k + 1) % 3]);
                                (a.min(b), a.max(b)) == e
                            })
                        });
                        any.then(|| (f, brep.faces[f].regions.map(|x| x.0), dir))
                    })
                    .collect()
            };
            rapidmesh_exact::log::debug(
                "surface.open",
                format!(
                    "region {r}: open edges {:?}",
                    open.iter()
                        .take(6)
                        .map(|&(a, b)| (
                            self.points[a as usize],
                            self.points[b as usize],
                            touching((a, b))
                        ))
                        .collect::<Vec<_>>()
                ),
            );
        }
        (open.len(), open.first().map(|e| self.points[e.0 as usize]))
    }
}

/// The boundary mesh of `model` under the size field of `domain`.
pub fn boundary(
    model: &Model,
    domain: &DomainTree,
    params: &MeshParams,
) -> Result<Boundary, BoundaryError> {
    boundary_keeping(model, domain, params).map(|x| x.0)
}

/// [`boundary`], with the Delaunay tetrahedralization of each region as
/// its last check made it.
///
/// The edges are sampled between their corners, then round after round the
/// faces are meshed on the samples, the regions they bound are checked,
/// and a segment that is no edge of the Delaunay tetrahedralization of a
/// region it bounds is split (an edge inside a curved face flipped or split
/// in its kept mesh), until each region's constrained Delaunay
/// tetrahedralization exists.
pub fn boundary_keeping(
    model: &Model,
    domain: &DomainTree,
    params: &MeshParams,
) -> Result<(Boundary, FxHashMap<u32, crate::volume::region::Kept>), BoundaryError> {
    let brep = &model.brep;
    let mut s = Rounds::new(model, domain, params);
    let mut dirty: Vec<usize> = (0..brep.faces.len()).collect();
    // Faces flipped but not remeshed, whose regions the next round checks.
    let mut recheck: Vec<usize> = Vec::new();
    let mut rounds = 0;
    // The edges missed at the round before, the fewest any round missed,
    // and the rounds in a row that missed more.
    let mut before = usize::MAX;
    let mut fewest = usize::MAX;
    let mut grew = 0;
    loop {
        let t = rapidmesh_exact::clock::Instant::now();
        s.with_originals(&mut dirty);
        s.faces(&mut dirty)?;
        rapidmesh_exact::log::stage("surface.faces", t.elapsed().as_secs_f64());
        // A face whose mesh has a hole or a fold stays so whatever the
        // rounds do: said at once.
        let seen: Vec<usize> = dirty.iter().chain(&recheck).copied().collect();
        if let Some(e) = broken_face(brep, &s.b, &seen, &s.comps) {
            return Err(e);
        }
        let t = rapidmesh_exact::clock::Instant::now();
        let Checked {
            missing,
            inside,
            left,
        } = s.check(&dirty, &recheck);
        rapidmesh_exact::log::stage("surface.segment_check", t.elapsed().as_secs_f64());
        let near = s.near(&inside);
        rapidmesh_exact::log::debug(
            "surface.round",
            format!(
                "{rounds}: {} faces, {} segments, {} edges inside missing; most on {:?}",
                dirty.len(),
                missing.len(),
                inside.len(),
                near
            ),
        );
        grew = if left > before { grew + 1 } else { 0 };
        before = left;
        fewest = fewest.min(left);
        // Rounds that keep missing more, or fewer of them that already
        // miss far more than the best round did: splitting makes more.
        let far = left as f64 > DIVERGED_GROWTH * fewest as f64;
        if grew >= DIVERGED_ROUNDS || (grew + 1 >= DIVERGED_ROUNDS && far) {
            return Err(BoundaryError::Diverged { left, near });
        }
        if (missing.is_empty() && inside.is_empty()) || rounds == MAX_SPLIT_ROUNDS {
            let log = rapidmesh_exact::log::stat;
            log("surface.split_rounds", rounds as f64);
            log("surface.segments_missing", left as f64);
            return Ok(s.finish());
        }
        let (changed, outline) = s.split(missing);
        dirty = changed
            .iter()
            .flat_map(|&ei| s.edge_faces[ei].iter().copied())
            .collect();
        let in_place = s.in_place(&dirty);
        let mut edited = s.edit_inside(inside, &dirty, &in_place);
        let failed = s.insert_outline(&outline, &in_place);
        for &f in &in_place {
            if !failed.contains(&f) {
                edited.push(f);
            }
        }
        dirty.retain(|f| !in_place.contains(f) || failed.contains(f));
        dirty.sort_unstable();
        dirty.dedup();
        edited.sort_unstable();
        edited.dedup();
        // The regions of the remeshed and the edited faces are checked
        // afresh.
        for &fi in dirty.iter().chain(&edited) {
            for r in brep.faces[fi].regions.map(|r| r.0) {
                s.region_missing.remove(&r);
            }
        }
        recheck = edited;
        rounds += 1;
    }
}

/// The share of an edge's sample spacing its chain may lie off its curve
/// and the samples still be made on the chain.
const OFF_CHORDS: f64 = 0.5;

/// How far the chords of `chain` lie off `curve` at most: the sagitta of
/// each by the curve's radius there.
fn chord_depth(chain: &[V3], curve: &dyn Curve) -> f64 {
    let total: f64 = chain.windows(2).map(|w| dist2(w[0], w[1]).sqrt()).sum();
    if !(total > 0.0) {
        return 0.0;
    }
    let (mut at, mut depth) = (0.0, 0.0f64);
    for w in chain.windows(2) {
        let l = dist2(w[0], w[1]).sqrt();
        let r = curve.radius_at((at + 0.5 * l) / total * curve.length());
        if r.is_finite() && r > 0.5 * l {
            depth = depth.max(r - (r * r - 0.25 * l * l).sqrt());
        }
        at += l;
    }
    depth
}

/// The closest two of the samples `arcs` of an edge of length `len` and
/// its ends are apart.
fn spacing(arcs: &[f64], len: f64) -> f64 {
    let mut all = Vec::with_capacity(arcs.len() + 2);
    all.push(0.0);
    all.extend_from_slice(arcs);
    all.push(len);
    all.sort_by(f64::total_cmp);
    all.windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min)
}

/// Samples closer than `gap` to each other or to a corner are one: a
/// segment of no length is no edge of any tetrahedralization.
fn spaced(arcs: &mut Vec<f64>, len: f64, gap: f64) {
    arcs.sort_by(f64::total_cmp);
    let mut last = 0.0;
    arcs.retain(|&s| {
        let keep = s - last > gap && len - s > gap;
        if keep {
            last = s;
        }
        keep
    });
}

/// What a check of the regions found missing: the segments, the edges
/// inside faces (face, ends, whether a flip may mend it), and the count
/// over all regions.
struct Checked {
    missing: Vec<[u32; 2]>,
    inside: Vec<(u32, [u32; 2], bool)>,
    left: usize,
}

/// The edge of a point on none: a corner or a face's own point.
const NO_EDGE: u32 = u32::MAX;

/// The state the rounds of [`boundary_keeping`] carry from one to the
/// next: the samples of each edge, the boundary they make, the kept face
/// meshes and what each region still misses.
struct Rounds<'m> {
    model: &'m Model,
    domain: &'m DomainTree,
    params: &'m MeshParams,
    extent: f64,
    /// Samples closer than this are one.
    gap: f64,
    curves: Vec<Option<Box<dyn Curve>>>,
    /// The samples of each edge, by arc length.
    arcs: Vec<Vec<f64>>,
    /// Periodic faces: edge classes sampled from their roots, and the faces
    /// that copy another.
    classes: crate::surface::periodic::Classes,
    /// Faces smaller than the size joined into composites.
    comps: crate::surface::topology::Composites,
    face_coedges: Vec<Vec<u32>>,
    face_edges: Vec<Vec<usize>>,
    face_corners: Vec<Vec<u32>>,
    edge_faces: Vec<Vec<usize>>,
    /// The boundary as the rounds keep it. A point keeps its id for good:
    /// a new sample or face point takes the next one, and a point no edge
    /// or face holds any more stays unused until the end.
    b: Boundary,
    /// The point of each edge sample, by its edge and arc length.
    sample: FxHashMap<(u32, u64), u32>,
    /// The edge each point is a sample of ([`NO_EDGE`] for a corner or a
    /// face's own point).
    edge_of: Vec<u32>,
    /// Each round remeshes only the faces on a split edge and checks only
    /// the regions around them; the others keep their meshes, edited in
    /// place (a face joined into a composite has an empty one).
    meshes: Vec<Option<Stars>>,
    /// Points a curved face must take (its triangles held to being Delaunay).
    required: Vec<Vec<V3>>,
    /// The edges flips made.
    flipped_in: FxHashSet<(u32, u32)>,
    /// Rounds that took samples a face asked for.
    refines: usize,
    /// The Delaunay tetrahedralization of each region at its last check.
    dts: FxHashMap<u32, crate::volume::region::Kept>,
    region_missing: FxHashMap<u32, usize>,
}

impl<'m> Rounds<'m> {
    /// The edges sampled between their corners by the size, the periodic
    /// classes and the composites made.
    fn new(model: &'m Model, domain: &'m DomainTree, params: &'m MeshParams) -> Rounds<'m> {
        let (plc, brep) = (&model.plc, &model.brep);
        let (lo, hi) = bbox(&plc.vertices);
        let extent = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max).max(1e-12);
        let floor = params.h_floor(extent);
        let (_, edge_laws) = crate::sizing::curvature_laws(model, params);
        let grading = params.grade();
        // Edges sharing both corners (and closed edges) bound a face only with
        // points between their corners: two samples for a closed edge, one for
        // each of several edges between two corners.
        let mut between: FxHashMap<(u32, u32), usize> = FxHashMap::default();
        for e in &brep.edges {
            let (a, b) = (e.ends[0].0, e.ends[1].0);
            *between.entry((a.min(b), a.max(b))).or_default() += 1;
        }
        // An edge is sampled along the chain of facets it follows, the
        // finish snapping the samples onto its curve; but where its samples
        // come closer than the chain is off the curve (a tolerance on the
        // edge far finer than the faceting), they are made on the curve
        // itself: on the chords, the faces would be dented there. Not where
        // a face is meshed on its own facets (a B-spline or scanned face, a
        // revolved one), which takes its samples on them.
        let on_carrier = |e: &rapidmesh_brep::Edge| {
            e.coedges.iter().all(|&c| {
                let f = &brep.faces[brep.coedge(c).face.0 as usize];
                !matches!(
                    brep.surface(f.surface),
                    rapidmesh_geom::Surface::Nurbs(_)
                        | rapidmesh_geom::Surface::Discrete(_)
                        | rapidmesh_geom::Surface::Revolved { .. }
                )
            })
        };
        let (curves, mut arcs): (Vec<Option<Box<dyn Curve>>>, Vec<Vec<f64>>) = brep
            .edges
            .par_iter()
            .enumerate()
            .map(|(ei, e)| {
                let cap = params.edge_maxh_for(ei);
                // A circle has no spike for the floor to stop, so it takes
                // as many segments per turn as the tolerance asks, however
                // small it is (a small hole stays round).
                let law = edge_laws[ei];
                let bent = |r: f64| law.curve(r);
                let floor = if e.curve.is_circle() { 0.0 } else { floor };
                let sample = |c: &dyn Curve| -> Vec<f64> {
                    let size = |s: f64| domain.h_at_surf(c.point_at(s)).min(cap);
                    let len = c.length();
                    let ss = distribute_floored(c, &bent, &size, grading, floor);
                    let mut arcs: Vec<f64> =
                        ss.into_iter().filter(|&s| s > 0.0 && s < len).collect();
                    let (a, b) = (e.ends[0].0, e.ends[1].0);
                    if a == b && arcs.len() < 2 {
                        arcs = vec![len / 3.0, 2.0 * len / 3.0];
                    } else if between[&(a.min(b), a.max(b))] > 1 && arcs.is_empty() {
                        arcs = vec![len / 2.0];
                    }
                    arcs
                };
                let Some(chain) = PolylineCurve::new(&e.chain) else {
                    return (None, Vec::new());
                };
                let own = crate::curve::kinds::edge_curve(brep, e);
                let arcs = match &own {
                    // A circle by its own radius (as the finish snaps it).
                    Some(x) if e.curve.is_circle() => sample(&Guided {
                        along: &chain,
                        by: &**x,
                    }),
                    _ => sample(&chain),
                };
                let deep = |x: &dyn Curve| {
                    chord_depth(&e.chain, x) > OFF_CHORDS * spacing(&arcs, chain.length())
                };
                match own {
                    Some(x) if on_carrier(e) && deep(&*x) => {
                        let arcs = sample(&*x);
                        (Some(x), arcs)
                    }
                    _ => (Some(Box::new(chain) as Box<dyn Curve>), arcs),
                }
            })
            .unzip();
        rapidmesh_exact::log::debug(
            "surface.exact_edges",
            format!(
                "{} of {} edges sampled on their curves, finer than their chains are off them",
                curves
                    .iter()
                    .flatten()
                    .zip(&brep.edges)
                    .filter(|(c, e)| (c.length()
                        - PolylineCurve::new(&e.chain).map_or(0.0, |p| p.length()))
                    .abs()
                        > 0.0)
                    .count(),
                brep.edges.len()
            ),
        );
        let gap = 1e-9 * extent;
        for (a, c) in arcs.iter_mut().zip(&curves) {
            if let Some(c) = c {
                spaced(a, c.length(), gap);
            }
        }
        let mut face_coedges: Vec<Vec<u32>> = vec![Vec::new(); brep.faces.len()];
        for (ci, c) in brep.coedges.iter().enumerate() {
            face_coedges[c.face.0 as usize].push(ci as u32);
        }
        let face_edges: Vec<Vec<usize>> = face_coedges
            .iter()
            .map(|cs| {
                cs.iter()
                    .map(|&c| brep.coedges[c as usize].edge.0 as usize)
                    .collect()
            })
            .collect();
        let classes = crate::surface::periodic::Classes::new(
            brep,
            &curves,
            &face_edges,
            &params.periodic,
            1e-7 * extent,
        );
        classes.sync(&mut arcs, &curves, &|a, l| spaced(a, l, gap));
        // Faces smaller than the size joined into composites (none with
        // periodic faces), and the edges between them unsampled.
        let comps = if params.periodic.is_empty() {
            let kept = |f: usize| {
                params.surf_maxh.iter().any(|&(id, _)| id as usize == f)
                    || params.surf_tol.iter().any(|&(id, _)| id as usize == f)
            };
            crate::surface::topology::Composites::new(model, &|p| domain.h_at_surf(p), &kept)
        } else {
            crate::surface::topology::Composites::alone(brep)
        };
        for (a, &inside) in arcs.iter_mut().zip(&comps.internal) {
            if inside {
                a.clear();
            }
        }
        let mut face_corners: Vec<Vec<u32>> = vec![Vec::new(); brep.faces.len()];
        for (vi, v) in brep.vertices.iter().enumerate() {
            for f in &v.faces {
                face_corners[f.0 as usize].push(vi as u32);
            }
        }
        let mut edge_faces: Vec<Vec<usize>> = vec![Vec::new(); brep.edges.len()];
        for c in &brep.coedges {
            edge_faces[c.edge.0 as usize].push(c.face.0 as usize);
        }
        let points: Vec<V3> = brep
            .vertices
            .iter()
            .map(|v| on_planes(brep, v.pos, v.faces.iter().map(|f| f.0 as usize)))
            .collect();
        let meshes = (0..brep.faces.len())
            .map(|f| (comps.root[f] != f).then(Stars::default))
            .collect();
        let mut s = Rounds {
            model,
            domain,
            params,
            extent,
            gap,
            curves,
            arcs,
            classes,
            comps,
            face_coedges,
            face_edges,
            face_corners,
            edge_faces,
            edge_of: vec![NO_EDGE; points.len()],
            b: Boundary {
                points,
                edges: vec![Vec::new(); brep.edges.len()],
                faces: vec![Vec::new(); brep.faces.len()],
            },
            sample: FxHashMap::default(),
            meshes,
            required: vec![Vec::new(); brep.faces.len()],
            flipped_in: FxHashSet::default(),
            refines: 0,
            dts: FxHashMap::default(),
            region_missing: FxHashMap::default(),
        };
        s.refresh_edges(0..brep.edges.len());
        s
    }

    /// The points of the edges `es` from their samples: a sample keeps its
    /// point round after round, a new one takes the next id. An edge inside
    /// a composite face is no edge of the mesh.
    fn refresh_edges(&mut self, es: impl IntoIterator<Item = usize>) {
        let brep = &self.model.brep;
        for e in es {
            if self.comps.internal[e] {
                self.b.edges[e].clear();
                continue;
            }
            let ends = brep.edges[e].ends;
            let mut ids = vec![ends[0].0];
            if let Some(c) = &self.curves[e] {
                for &s in &self.arcs[e] {
                    let id = *self
                        .sample
                        .entry((e as u32, s.to_bits()))
                        .or_insert_with(|| {
                            let p =
                                on_planes(brep, c.point_at(s), self.edge_faces[e].iter().copied());
                            self.b.points.push(p);
                            self.edge_of.push(e as u32);
                            (self.b.points.len() - 1) as u32
                        });
                    ids.push(id);
                }
            }
            ids.push(ends[1].0);
            self.b.edges[e] = ids;
        }
    }

    /// Whether a point is a corner or an edge sample (fixed for the faces
    /// on it), not a face's own.
    fn fixed(&self) -> impl Fn(u32) -> bool + '_ {
        let corners = self.model.brep.vertices.len() as u32;
        move |g| g < corners || self.edge_of[g as usize] != NO_EDGE
    }

    /// Which points the boundary uses: the corners and the points of its
    /// edges and faces.
    fn live(&self) -> Vec<bool> {
        let mut live = vec![false; self.b.points.len()];
        live[..self.model.brep.vertices.len()].fill(true);
        for &g in self.b.edges.iter().flatten() {
            live[g as usize] = true;
        }
        for t in self.b.faces.iter().flatten() {
            for &g in t {
                live[g as usize] = true;
            }
        }
        live
    }

    /// The boundary the rounds came to, with the points it uses numbered
    /// afresh in the order they came and a composite's triangles back on
    /// its faces; and the Delaunay tetrahedralization of each region.
    fn finish(self) -> (Boundary, FxHashMap<u32, crate::volume::region::Kept>) {
        let live = self.live();
        let mut id = vec![u32::MAX; live.len()];
        let mut points = Vec::with_capacity(live.iter().filter(|&&l| l).count());
        for (i, _) in live.iter().enumerate().filter(|x| *x.1) {
            id[i] = points.len() as u32;
            points.push(self.b.points[i]);
        }
        let at = |g: &u32| id[*g as usize];
        let mut b = Boundary {
            points,
            edges: self
                .b
                .edges
                .iter()
                .map(|ids| ids.iter().map(at).collect())
                .collect(),
            faces: self
                .b
                .faces
                .iter()
                .map(|ts| ts.iter().map(|t| t.each_ref().map(at)).collect())
                .collect(),
        };
        if self.comps.any() {
            to_members(self.model, &self.comps, &mut b);
        }
        (b, self.dts)
    }

    /// The samples of edge `e` spaced, and those of its periodic class
    /// kept in step.
    fn respace(&mut self, edges: impl IntoIterator<Item = usize>) {
        let gap = self.gap;
        for e in edges {
            if let Some(c) = &self.curves[e] {
                spaced(&mut self.arcs[e], c.length(), gap);
            }
        }
    }

    /// A copied face takes the mesh of its original, which is meshed for
    /// it; a face joined into a composite is meshed as its root.
    fn with_originals(&self, dirty: &mut Vec<usize>) {
        let extra: Vec<usize> = dirty
            .iter()
            .filter_map(|&f| self.classes.copy_of[f].map(|c| c.0))
            .collect();
        dirty.extend(extra);
        for f in dirty.iter_mut() {
            *f = self.comps.root[*f];
        }
        dirty.sort_unstable();
        dirty.dedup();
    }

    /// The faces `dirty` meshed on the samples; the samples a face asks for
    /// taken first (through their roots in a periodic class), and the faces
    /// on those edges meshed again with the ones that asked.
    fn faces(&mut self, dirty: &mut Vec<usize>) -> Result<(), BoundaryError> {
        loop {
            match self.mesh_faces(dirty) {
                Ok(()) => return Ok(()),
                Err(BoundaryError::Refine(at)) if self.refines < MAX_REFINES => {
                    self.refines += 1;
                    let mut grown: FxHashSet<usize> = FxHashSet::default();
                    for (e, arc) in at {
                        let e = e as usize;
                        let Some(c) = &self.curves[e] else { continue };
                        let (root, s) = self.classes.to_root(e, arc, c.length());
                        self.arcs[root].push(s);
                        grown.insert(root);
                    }
                    self.respace(grown.iter().copied());
                    if self.classes.any() {
                        let gap = self.gap;
                        self.classes
                            .sync(&mut self.arcs, &self.curves, &|a, l| spaced(a, l, gap));
                        grown = grown
                            .iter()
                            .flat_map(|&r| self.classes.members(r))
                            .collect();
                    }
                    self.refresh_edges(grown.iter().copied());
                    dirty.extend(
                        grown
                            .iter()
                            .flat_map(|&e| self.edge_faces[e].iter().copied()),
                    );
                    self.with_originals(dirty);
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// The faces `dirty` meshed afresh, and every copied face from its
    /// original. Where a face asks for samples, the faces that meshed keep
    /// their meshes and leave `dirty`.
    fn mesh_faces(&mut self, dirty: &mut Vec<usize>) -> Result<(), BoundaryError> {
        let fresh: Vec<(usize, Result<FaceOut, BoundaryError>)> = dirty
            .par_iter()
            .filter(|&&fi| self.classes.copy_of[fi].is_none() && self.comps.root[fi] == fi)
            .map(|&fi| {
                let members = self.comps.members(fi);
                let m = if members.len() > 1 {
                    self.mesh_composite(&members)
                } else {
                    self.mesh_face(fi)
                };
                (fi, m)
            })
            .collect();
        // Samples asked for by any face come first: with them, every face may
        // mesh.
        let needed: Vec<(u32, f64)> = fresh
            .iter()
            .filter_map(|(_, m)| match m {
                Err(BoundaryError::Refine(at)) => Some(at.clone()),
                _ => None,
            })
            .flatten()
            .collect();
        if !needed.is_empty() {
            // The faces that meshed keep their meshes: only the faces on the
            // edges that take the samples are meshed again.
            let mut meshed: FxHashSet<usize> = FxHashSet::default();
            for (fi, m) in fresh {
                if let Ok(m) = m {
                    self.take(fi, m);
                    meshed.insert(fi);
                }
            }
            dirty.retain(|f| !meshed.contains(f));
            return Err(BoundaryError::Refine(needed));
        }
        for (fi, m) in fresh {
            self.take(fi, m?);
        }
        if self.classes.copy_of.iter().any(Option::is_some) {
            self.copy_faces()?;
        }
        Ok(())
    }

    /// The fresh mesh of face `fi` into the boundary: its own points take
    /// the next ids (a point on a seam, there from either side, is one).
    fn take(&mut self, fi: usize, m: FaceOut) {
        let base = self.b.points.len() as u32;
        self.b.points.extend_from_slice(&m.own);
        self.edge_of.resize(self.b.points.len(), NO_EDGE);
        let id = |i: usize| match m.slots[i] {
            Slot::Global(g) => g,
            Slot::Own(k) => base + k,
        };
        let tris: Vec<[u32; 3]> = m.tris.iter().map(|t| t.map(id)).collect();
        self.meshes[fi] = Some(Stars::of(&tris));
        self.b.faces[fi] = tris;
    }

    /// Each copied face afresh from its original: its fixed points found by
    /// position among the corners and samples, its own points moved.
    fn copy_faces(&mut self) -> Result<(), BoundaryError> {
        let tol = 1e-7 * self.extent;
        let mut index = crate::finish::periodic::PointIndex::new(tol);
        let corners = self.model.brep.vertices.len() as u32;
        for g in (0..corners).chain(self.b.edges.iter().flatten().copied()) {
            index.insert(self.b.points[g as usize], g as usize);
        }
        let copies: Vec<(usize, usize, V3)> = self
            .classes
            .copy_of
            .iter()
            .enumerate()
            .filter_map(|(b, c)| c.map(|(a, shift)| (a, b, shift)))
            .collect();
        for (a, b, shift) in copies {
            if self.meshes[a].is_none() {
                continue;
            }
            let moved = |p: V3| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
            let turn = dot(face_front(self.model, a), face_front(self.model, b)) < 0.0;
            let mut own: FxHashMap<u32, u32> = FxHashMap::default();
            let mut tris = Vec::with_capacity(self.b.faces[a].len());
            for i in 0..self.b.faces[a].len() {
                let mut t = self.b.faces[a][i];
                for g in &mut t {
                    let p = moved(self.b.points[*g as usize]);
                    *g = if self.fixed()(*g) {
                        let points = &self.b.points;
                        index
                            .find(p, &|i| points[i], tol)
                            .ok_or(BoundaryError::Periodic { face: b as u32 })?
                            as u32
                    } else {
                        *own.entry(*g).or_insert_with(|| {
                            self.b.points.push(p);
                            self.edge_of.push(NO_EDGE);
                            (self.b.points.len() - 1) as u32
                        })
                    };
                }
                tris.push(if turn { [t[0], t[2], t[1]] } else { t });
            }
            self.meshes[b] = Some(Stars::of(&tris));
            self.b.faces[b] = tris;
        }
        Ok(())
    }

    /// The edge of face `fi` the segment between points `a` and `b` runs
    /// along, and the arc length at its middle (where a face whose chords
    /// cross asks for a sample).
    fn mid(&self, fi: usize, a: u32, b: u32) -> Option<(u32, f64)> {
        self.face_edges[fi].iter().find_map(|&e| {
            let ids = &self.b.edges[e];
            let k = ids
                .windows(2)
                .position(|w| (w[0], w[1]) == (a, b) || (w[0], w[1]) == (b, a))?;
            let len = self.curves[e].as_ref()?.length();
            let arc = |k: usize| match k {
                0 => 0.0,
                k if k == ids.len() - 1 => len,
                k => self.arcs[e][k - 1],
            };
            Some((e as u32, 0.5 * (arc(k) + arc(k + 1))))
        })
    }

    /// The regions of the faces `dirty` and `recheck`, each from its
    /// tetrahedralization of the round before (which takes the points the
    /// splits added), checked for what they miss.
    fn check(&mut self, dirty: &[usize], recheck: &[usize]) -> Checked {
        let (b, brep) = (&self.b, &self.model.brep);
        let mut touched: Vec<u32> = dirty
            .iter()
            .chain(recheck)
            .flat_map(|&fi| brep.faces[fi].regions.map(|r| r.0))
            .filter(|&r| r != 0)
            .collect();
        touched.sort_unstable();
        touched.dedup();
        let jobs: Vec<(u32, Option<crate::volume::region::Kept>)> =
            touched.iter().map(|&r| (r, self.dts.remove(&r))).collect();
        let checked: Vec<(
            u32,
            (crate::volume::region::Check, crate::volume::region::Kept),
        )> = jobs
            .into_par_iter()
            .map(|(r, prev)| (r, crate::volume::region::check_keeping(b, brep, r, prev)))
            .collect();
        let checked: Vec<(u32, crate::volume::region::Check)> = checked
            .into_iter()
            .map(|(r, (c, k))| {
                self.dts.insert(r, k);
                (r, c)
            })
            .collect();
        let missing: Vec<[u32; 2]> = checked
            .iter()
            .flat_map(|x| x.1.segments.iter().copied())
            .collect();
        // A face between two regions is reported by both.
        let mut inside: Vec<(u32, [u32; 2], bool)> = checked
            .iter()
            .flat_map(|x| x.1.edges.iter().copied())
            .collect();
        inside.sort_unstable();
        inside.dedup();
        // Edited in the order of where the edges lie, not of the faces' and
        // points' numbers: the edits of one face meet each other.
        let at = |g: u32| b.points[g as usize].map(f64::to_bits);
        inside.sort_by_cached_key(|&(_, [x, y], _)| {
            let (p, q) = (at(x), at(y));
            (p.min(q), p.max(q))
        });
        self.region_missing.extend(
            checked
                .iter()
                .map(|(r, c)| (*r, c.segments.len() + c.edges.len())),
        );
        Checked {
            missing,
            inside,
            left: self.region_missing.values().sum(),
        }
    }

    /// The faces missing most, with their carrier, a point of each, their
    /// shortest edge and the size there.
    fn near(&self, inside: &[(u32, [u32; 2], bool)]) -> Vec<Near> {
        let brep = &self.model.brep;
        let mut per: FxHashMap<u32, (usize, [u32; 2])> = FxHashMap::default();
        for &(f, e, _) in inside {
            per.entry(f).or_insert((0, e)).0 += 1;
        }
        let mut worst: Vec<(u32, (usize, [u32; 2]))> = per.into_iter().collect();
        worst.sort_unstable_by_key(|&(f, (n, _))| (std::cmp::Reverse(n), f));
        worst
            .into_iter()
            .take(4)
            .map(|(f, (missing, [a, _]))| {
                let at = self.b.points[a as usize];
                Near {
                    face: f,
                    kind: surface_kind(brep.surface(brep.faces[f as usize].surface)),
                    missing,
                    at,
                    shortest: self.face_edges[f as usize]
                        .iter()
                        .filter_map(|&e| self.curves[e].as_ref().map(|c| c.length()))
                        .fold(f64::INFINITY, f64::min),
                    size: self.domain.h_at_surf(at),
                }
            })
            .collect()
    }

    /// The missing segments split: each where the point deepest in its
    /// diametral ball projects onto it (that point then lies outside the
    /// balls of both halves; at the middle where no point is inside, a tie
    /// broken by the perturbation). Returns the edges that took samples and
    /// the splits a kept face mesh may take in place (the edge, the
    /// segment's ends and the new sample).
    #[allow(clippy::type_complexity)]
    fn split(&mut self, missing: Vec<[u32; 2]>) -> (FxHashSet<usize>, Vec<(usize, [u32; 2], u32)>) {
        let mut at: FxHashMap<(u32, u32), (usize, usize)> = FxHashMap::default();
        for (ei, ids) in self.b.edges.iter().enumerate() {
            for (k, w) in ids.windows(2).enumerate() {
                at.insert((w[0].min(w[1]), w[0].max(w[1])), (ei, k));
            }
        }
        let near = (!missing.is_empty()).then(|| point_grid(&self.b.points, &self.live()));
        let mut split: FxHashMap<(usize, usize), f64> = FxHashMap::default();
        for [a, c] in missing {
            let Some(&x) = at.get(&(a.min(c), a.max(c))) else {
                continue;
            };
            let (pa, pc) = (self.b.points[a as usize], self.b.points[c as usize]);
            let t = near
                .as_ref()
                .and_then(|g| deepest_in_ball(g, &self.b.points, pa, pc, [a, c]))
                .map(|q| {
                    let d = sub(pc, pa);
                    (dot(sub(q, pa), d) / dot(d, d).max(1e-300)).clamp(0.2, 0.8)
                })
                .unwrap_or(0.5);
            split.insert(x, t);
        }
        let mut changed: FxHashSet<usize> = FxHashSet::default();
        let mut outline: Vec<(usize, [u32; 2], f64)> = Vec::new();
        for ((ei, k), t) in split {
            let Some(c) = &self.curves[ei] else {
                continue;
            };
            let len = c.length();
            let arcs = &self.arcs[ei];
            let s = |i: usize| -> f64 {
                if i == 0 {
                    0.0
                } else if i > arcs.len() {
                    len
                } else {
                    arcs[i - 1]
                }
            };
            let (lo, hi) = (s(k), s(k + 1));
            if hi - lo > 2.0 * self.gap {
                // A split of an edge in a periodic class is its root's.
                let (root, at) = self.classes.to_root(ei, lo + t * (hi - lo), len);
                self.arcs[root].push(at);
                changed.insert(root);
                if !self.classes.any() {
                    let ids = &self.b.edges[ei];
                    outline.push((ei, [ids[k], ids[k + 1]], at));
                }
            }
        }
        self.respace(changed.iter().copied());
        if self.classes.any() {
            let gap = self.gap;
            self.classes
                .sync(&mut self.arcs, &self.curves, &|a, l| spaced(a, l, gap));
            changed = changed
                .iter()
                .flat_map(|&r| self.classes.members(r))
                .collect();
        }
        self.refresh_edges(changed.iter().copied());
        let mut outline: Vec<(usize, [u32; 2], u32)> = outline
            .into_iter()
            .filter_map(|(ei, ends, at)| {
                let g = self.sample.get(&(ei as u32, at.to_bits()))?;
                Some((ei, ends, *g))
            })
            .collect();
        // Put in by where the new samples lie (the splits come out of a
        // map): one face's take each other's in turn.
        outline.sort_by_key(|o| self.b.points[o.2 as usize].map(f64::to_bits));
        (changed, outline)
    }

    /// Faces whose outline takes the new samples in place: their points
    /// stay, so the regions' kept tetrahedralizations take only the new
    /// ones (a planar face is flipped back to its exact constrained
    /// Delaunay triangulation after, see [`Rounds::insert_outline`]). A
    /// discrete face is meshed afresh (remeshed on its facets it comes out
    /// better), so is a face of a periodic class (its copies follow the
    /// original).
    fn in_place(&self, dirty: &[usize]) -> FxHashSet<usize> {
        let brep = &self.model.brep;
        if self.classes.copy_of.iter().any(|c| c.is_some()) {
            return FxHashSet::default();
        }
        dirty
            .iter()
            .copied()
            .filter(|&f| {
                self.meshes[f].is_some()
                    && !matches!(
                        brep.surface(brep.faces[f].surface),
                        rapidmesh_geom::Surface::Discrete(_)
                    )
            })
            .collect()
    }

    /// An edge inside a curved face that is no Delaunay edge is flipped in
    /// the kept mesh where its other diagonal is one, else split at its
    /// middle on the carrier; the face keeps the point when it is meshed
    /// afresh; where the point on the carrier would fold the face, the edge
    /// splits at its middle. An edge short against the size is left (a
    /// sharp angle, where splitting would not end). Returns the faces
    /// edited.
    fn edit_inside(
        &mut self,
        inside: Vec<(u32, [u32; 2], bool)>,
        dirty: &[usize],
        in_place: &FxHashSet<usize>,
    ) -> Vec<usize> {
        let brep = &self.model.brep;
        let tol = 1e-7 * self.extent;
        let mut edited: Vec<usize> = Vec::new();
        // An edge of a copied face is its original's, moved back.
        let back = self.classes.copy_of.iter().any(|c| c.is_some()).then(|| {
            let mut index = crate::finish::periodic::PointIndex::new(tol);
            for (i, _) in self.live().iter().enumerate().filter(|x| *x.1) {
                index.insert(self.b.points[i], i);
            }
            index
        });
        for (fi, [ga, gb], flip) in inside {
            let (fi, ga, gb) = match (self.classes.copy_of[fi as usize], &back) {
                (Some((a, shift)), Some(index)) => {
                    let moved = |g: u32| {
                        let p = self.b.points[g as usize];
                        let q = [p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]];
                        index.find(q, &|i| self.b.points[i], tol)
                    };
                    let (Some(x), Some(y)) = (moved(ga), moved(gb)) else {
                        continue;
                    };
                    (a as u32, x as u32, y as u32)
                }
                _ => (fi, ga, gb),
            };
            let f = fi as usize;
            if dirty.contains(&f) && !in_place.contains(&f) {
                continue;
            }
            let Some(stars) = self.meshes[f].as_mut() else {
                continue;
            };
            let tris = &mut self.b.faces[f];
            let points = &self.b.points;
            // An edge a flip made is not flipped back (a face between two
            // regions can have its Delaunay edge in one and not the other):
            // it splits.
            if flip && !self.flipped_in.contains(&key(ga, gb)) {
                if let Some((c, d)) = flip_kept(tris, stars, ga, gb, points) {
                    self.flipped_in.insert(key(c, d));
                    edited.push(f);
                    continue;
                }
            }
            let (pa, pb) = (points[ga as usize], points[gb as usize]);
            let mid: V3 = std::array::from_fn(|k| 0.5 * (pa[k] + pb[k]));
            let q = brep.surface(brep.faces[f].surface).closest(mid).0;
            let least = 2.0 * REQUIRED_SPACING * self.domain.h_at_surf(q);
            if dist2(pa, pb) <= least * least {
                continue;
            }
            // On the carrier, else (where that folds the face) on the chord,
            // off the carrier by no more than the edge already is.
            let v = points.len() as u32;
            if let Some(q) = [q, mid]
                .into_iter()
                .find(|&x| split_kept(tris, stars, ga, gb, v, x, points, 2))
            {
                self.b.points.push(q);
                self.edge_of.push(NO_EDGE);
                self.required[f].push(q);
                edited.push(f);
            } else {
                rapidmesh_exact::log::debug(
                    "surface.split",
                    format!(
                        "face {f} ({}): edge {ga}-{gb} on {} triangles kept no split",
                        surface_kind(brep.surface(brep.faces[f].surface)),
                        stars.on_edge(tris, ga, gb).len()
                    ),
                );
            }
        }
        edited
    }

    /// The new samples into the kept meshes, after the edits inside them.
    /// Returns the faces that could not take one, to be meshed afresh.
    fn insert_outline(
        &mut self,
        outline: &[(usize, [u32; 2], u32)],
        in_place: &FxHashSet<usize>,
    ) -> FxHashSet<usize> {
        let mut failed: FxHashSet<usize> = FxHashSet::default();
        let corners = self.model.brep.vertices.len() as u32;
        let edge_of = &self.edge_of;
        let fixed = |g: u32| g < corners || edge_of[g as usize] != NO_EDGE;
        for &(ei, [ga, gb], g) in outline {
            for &f in &self.edge_faces[ei] {
                if !in_place.contains(&f) || failed.contains(&f) {
                    continue;
                }
                let Some(stars) = self.meshes[f].as_mut() else {
                    failed.insert(f);
                    continue;
                };
                let (tris, points) = (&mut self.b.faces[f], &self.b.points);
                if split_kept(tris, stars, ga, gb, g, points[g as usize], points, 1) {
                    legalize(tris, stars, g, points, &fixed);
                } else {
                    failed.insert(f);
                }
            }
        }
        // A planar face goes back to its exact constrained Delaunay
        // triangulation, the one the volume stage recovers.
        let brep = &self.model.brep;
        for &f in in_place {
            if failed.contains(&f) || !brep.surface(brep.faces[f].surface).is_plane() {
                continue;
            }
            let front = face_front(self.model, f);
            if let Some(stars) = self.meshes[f].as_mut() {
                exact_flips(
                    &mut self.b.faces[f],
                    stars,
                    brep,
                    &self.b.points,
                    edge_of,
                    front,
                );
            }
        }
        failed
    }
}

/// Points in a uniform grid, for the point deepest in a segment's
/// diametral ball.
/// The points in use (`live`) in a grid of about one a cell.
fn point_grid(points: &[V3], live: &[bool]) -> HashGrid<u32> {
    let (lo, hi) = bbox(points);
    let span = (0..3)
        .map(|k| hi[k] - lo[k])
        .fold(0.0, f64::max)
        .max(1e-300);
    let n = live.iter().filter(|&&l| l).count();
    let cell = (span / (n.max(1) as f64).cbrt()).max(1e-12 * span);
    let mut g = HashGrid::with_origin(lo, cell);
    for (i, &p) in points.iter().enumerate().filter(|x| live[x.0]) {
        g.insert(p, i as u32);
    }
    g
}

/// The point (not in `skip`) deepest inside the ball on the diameter `a b`.
fn deepest_in_ball(
    grid: &HashGrid<u32>,
    points: &[V3],
    a: V3,
    b: V3,
    skip: [u32; 2],
) -> Option<V3> {
    let m: V3 = std::array::from_fn(|k| 0.5 * (a[k] + b[k]));
    let r2 = 0.25 * dist2(a, b);
    let r = r2.sqrt();
    let mut best: Option<(f64, V3)> = None;
    for &v in grid.in_box(m.map(|x| x - r), m.map(|x| x + r)) {
        if skip.contains(&v) {
            continue;
        }
        let q = points[v as usize];
        let depth = r2 - dist2(q, m);
        if depth > 0.0 && best.is_none_or(|(d, _)| depth > d) {
            best = Some((depth, q));
        }
    }
    best.map(|x| x.1)
}

/// The triangles of each composite face back to the faces it joined: each
/// to the face of the facet nearest its centroid.
fn to_members(model: &Model, comps: &crate::surface::topology::Composites, b: &mut Boundary) {
    let (plc, brep) = (&model.plc, &model.brep);
    for root in 0..brep.faces.len() {
        if comps.root[root] != root {
            continue;
        }
        let members = comps.members(root);
        if members.len() < 2 {
            continue;
        }
        // The facet centroids of the members in a grid about a facet wide.
        let mut cents: Vec<(V3, usize)> = Vec::new();
        let mut span = 0.0;
        for &f in &members {
            for &t in &brep.faces[f].facets {
                let q = plc.triangles[t as usize].map(|i| plc.vertices[i as usize]);
                span += dist2(q[0], q[1]).sqrt();
                cents.push((
                    std::array::from_fn(|k| (q[0][k] + q[1][k] + q[2][k]) / 3.0),
                    f,
                ));
            }
        }
        let cell = (2.0 * span / cents.len().max(1) as f64).max(1e-300);
        let mut grid: HashGrid<usize> = HashGrid::new(cell);
        for (i, (c, _)) in cents.iter().enumerate() {
            grid.insert(*c, i);
        }
        let nearest = |p: V3| -> usize {
            grid.nearest(p, 0.0, |&i| dist2(cents[i].0, p))
                .map_or(root, |(&i, _)| cents[i].1)
        };
        let tris = std::mem::take(&mut b.faces[root]);
        for t in tris {
            let q = t.map(|v| b.points[v as usize]);
            let c: V3 = std::array::from_fn(|k| (q[0][k] + q[1][k] + q[2][k]) / 3.0);
            b.faces[nearest(c)].push(t);
        }
    }
}

/// The first face among `faces` whose mesh has an edge with one triangle
/// that is no segment of a B-rep edge (a hole in the face), an edge on more
/// than two triangles, or leaves a segment of its edges out.
fn broken_face(
    brep: &Brep,
    b: &Boundary,
    faces: &[usize],
    comps: &crate::surface::topology::Composites,
) -> Option<BoundaryError> {
    let segments: FxHashSet<(u32, u32)> = b
        .edges
        .iter()
        .flat_map(|ids| ids.windows(2).map(|w| (w[0].min(w[1]), w[0].max(w[1]))))
        .collect();
    let mut roots: Vec<usize> = faces.iter().map(|&f| comps.root[f]).collect();
    roots.sort_unstable();
    roots.dedup();
    for fi in roots {
        let members = comps.members(fi);
        let mut count: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        for t in &b.faces[fi] {
            for k in 0..3 {
                let (a, c) = (t[k], t[(k + 1) % 3]);
                *count.entry((a.min(c), a.max(c))).or_default() += 1;
            }
        }
        let open = count
            .iter()
            .filter(|(e, &n)| n == 1 && !segments.contains(e))
            .count();
        let over = count.values().filter(|&&n| n > 2).count();
        // The segments of the face's own edges its triangles leave out.
        let unused = brep
            .coedges
            .iter()
            .filter(|c| members.contains(&(c.face.0 as usize)))
            .flat_map(|c| b.edges[c.edge.0 as usize].windows(2))
            .filter(|w| !count.contains_key(&(w[0].min(w[1]), w[0].max(w[1]))))
            .count();
        if open > 0 || over > 0 || unused > 0 {
            let at = |&(a, c): &(u32, u32)| (b.points[a as usize], b.points[c as usize]);
            let bad: Vec<_> = count
                .iter()
                .filter(|(e, &n)| (n == 1 && !segments.contains(e)) || n > 2)
                .map(|(e, &n)| (n, at(e)))
                .collect();
            rapidmesh_exact::log::debug(
                "surface.broken",
                format!(
                    "face {fi} ({}): {} triangles, edges (use count, ends) {bad:?}",
                    surface_kind(brep.surface(brep.faces[fi].surface)),
                    b.faces[fi].len()
                ),
            );
            return Some(BoundaryError::Broken {
                face: fi as u32,
                kind: surface_kind(brep.surface(brep.faces[fi].surface)),
                open,
                over,
                unused,
            });
        }
    }
    None
}

/// The facets of face `fi`, each wound toward the face's front (the side
/// of its first region).
pub(crate) fn front_facets(model: &Model, fi: usize) -> impl Iterator<Item = [V3; 3]> + '_ {
    let (plc, face) = (&model.plc, &model.brep.faces[fi]);
    face.facets.iter().map(move |&t| {
        let p = plc.triangles[t as usize].map(|i| plc.vertices[i as usize]);
        if plc.region_tags[t as usize] == face.regions {
            p
        } else {
            [p[0], p[2], p[1]]
        }
    })
}

/// The front side of face `fi`: the summed normal of its facets.
fn face_front(model: &Model, fi: usize) -> V3 {
    front_facets(model, fi).fold([0.0; 3], |s, p| {
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        [s[0] + n[0], s[1] + n[1], s[2] + n[2]]
    })
}

/// A point a curved face must take stays this multiple of the size away
/// from its triangle's corners and the face's other such points.
const REQUIRED_SPACING: f64 = 0.1;

/// Rounds of samples asked for by faces before a face that still asks
/// fails.
const MAX_REFINES: usize = 8;

/// Split rounds before a region is left without its constrained Delaunay
/// tetrahedralization.
const MAX_SPLIT_ROUNDS: usize = 40;

/// Rounds in a row that miss more edges than the one before, after which
/// the boundary gives up; a round fewer where the last misses this many
/// times as many as the best round did.
const DIVERGED_ROUNDS: usize = 3;
const DIVERGED_GROWTH: f64 = 1.5;

/// The axis and coordinate of a plane normal to an axis (to a rounding of
/// its normal), if it is one.
fn axis_plane(s: &rapidmesh_geom::Surface) -> Option<(usize, f64)> {
    let rapidmesh_geom::Surface::Plane(f) = s else {
        return None;
    };
    let (o, normal) = (f.o, f.z);
    let k = (0..3).max_by(|&a, &b| normal[a].abs().total_cmp(&normal[b].abs()))?;
    let off: f64 = (0..3).filter(|&j| j != k).map(|j| normal[j].abs()).sum();
    (off <= 1e-12 * normal[k].abs()).then_some((k, o[k]))
}

/// `p` on the axis-aligned planes among `faces`: it takes each one's
/// coordinate exactly. A curve's samples and a corner are on such a plane
/// only to a rounding otherwise, and a plane whose points are off it is no
/// single facet but one per triangle, each of whose edges the regions must
/// then have.
fn on_planes(brep: &Brep, mut p: V3, faces: impl IntoIterator<Item = usize>) -> V3 {
    for f in faces {
        if let Some((k, x)) = axis_plane(brep.surface(brep.faces[f].surface)) {
            p[k] = x;
        }
    }
    p
}

/// The points of a loop of co-edges, each edge's from its first on, run
/// the way the co-edges run.
fn ring(brep: &Brep, edges: &[Vec<u32>], coedges: impl IntoIterator<Item = u32>) -> Vec<u32> {
    let mut ring: Vec<u32> = Vec::new();
    for ce in coedges {
        let c = &brep.coedges[ce as usize];
        let pts = &edges[c.edge.0 as usize];
        if c.forward {
            ring.extend(&pts[..pts.len() - 1]);
        } else {
            ring.extend(pts[1..].iter().rev());
        }
    }
    ring
}

/// An edge by its ends, whichever way round.
fn key(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// The triangles at each point of a kept face mesh (indices into the
/// face's triangles in the boundary), for editing it in place.
#[derive(Default)]
struct Stars(FxHashMap<u32, Vec<usize>>);

impl Stars {
    fn of(tris: &[[u32; 3]]) -> Stars {
        let mut at: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        for (ti, t) in tris.iter().enumerate() {
            for &v in t {
                at.entry(v).or_default().push(ti);
            }
        }
        Stars(at)
    }

    /// The triangles of `tris` on the edge `a b`.
    fn on_edge(&self, tris: &[[u32; 3]], a: u32, b: u32) -> Vec<usize> {
        self.0.get(&a).map_or(Vec::new(), |ts| {
            ts.iter()
                .copied()
                .filter(|&t| tris[t].contains(&b))
                .collect()
        })
    }

    fn link(&mut self, v: u32, t: usize) {
        self.0.entry(v).or_default().push(t);
    }

    fn unlink(&mut self, v: u32, t: usize) {
        if let Some(ts) = self.0.get_mut(&v) {
            ts.retain(|&x| x != t);
        }
    }
}

/// Flips the edge `a b` of a kept face mesh where its two triangles turn
/// the same way after (wound as they were), returning the new diagonal;
/// none when the edge is not inside the face, the other diagonal is an
/// edge already (round a point of three triangles: the flip would put it
/// on four) or the new triangles would fold.
fn flip_kept(
    tris: &mut [[u32; 3]],
    stars: &mut Stars,
    a: u32,
    b: u32,
    points: &[V3],
) -> Option<(u32, u32)> {
    let [t0, t1] = stars.on_edge(tris, a, b)[..] else {
        return None;
    };
    let third = |t: [u32; 3]| t.iter().copied().find(|&v| v != a && v != b);
    let (Some(c), Some(d)) = (third(tris[t0]), third(tris[t1])) else {
        return None;
    };
    if c == d || !stars.on_edge(tris, c, d).is_empty() {
        return None;
    }
    let (n0, n1) = if wound(tris[t0], a, b) {
        ([c, a, d], [d, b, c])
    } else {
        ([c, d, a], [d, c, b])
    };
    let p = |i: u32| points[i as usize];
    let normal = |t: [u32; 3]| cross(sub(p(t[1]), p(t[0])), sub(p(t[2]), p(t[0])));
    let old = add(normal(tris[t0]), normal(tris[t1]));
    if !(dot(normal(n0), old) > 0.0 && dot(normal(n1), old) > 0.0) {
        return None;
    }
    tris[t0] = n0;
    tris[t1] = n1;
    stars.unlink(a, t1);
    stars.unlink(b, t0);
    stars.link(c, t1);
    stars.link(d, t0);
    Some((c, d))
}

/// Lawson flips round the new point `v` of a kept face mesh: each edge
/// across from it whose opposite angles sum past a half turn flips, and
/// the two it then faces are looked at in turn. The outline stays (an
/// edge on one triangle has nothing to flip with), and so does an edge
/// between two `fixed` points.
fn legalize(
    tris: &mut [[u32; 3]],
    stars: &mut Stars,
    v: u32,
    points: &[V3],
    fixed: &dyn Fn(u32) -> bool,
) {
    let p = |i: u32| points[i as usize];
    let mut stack: Vec<(u32, u32)> = stars.0.get(&v).map_or(Vec::new(), |ts| {
        ts.iter()
            .map(|&t| {
                let e: Vec<u32> = tris[t].iter().copied().filter(|&x| x != v).collect();
                (e[0], e[1])
            })
            .collect()
    });
    let mut guard = 0;
    while let Some((a, b)) = stack.pop() {
        guard += 1;
        if guard > 256 {
            break;
        }
        let [t0, t1] = stars.on_edge(tris, a, b)[..] else {
            continue;
        };
        let third = |t: [u32; 3]| t.iter().copied().find(|&x| x != a && x != b);
        let (Some(c), Some(d)) = (third(tris[t0]), third(tris[t1])) else {
            continue;
        };
        if c != v && d != v {
            continue;
        }
        let far = if c == v { d } else { c };
        // An edge between two fixed points may be a segment of an edge
        // inside the face: it stays.
        if fixed(a) && fixed(b) {
            continue;
        }
        if !across_too_wide(p(a), p(b), p(v), p(far)) {
            continue;
        }
        if flip_kept(tris, stars, a, b, points).is_some() {
            stack.push((a, far));
            stack.push((far, b));
        }
    }
}

/// Splits the edge `a b` of a kept face mesh, on `sides` triangles, at the
/// point `v` at `q` (each triangle on the edge becomes two, wound the same
/// way); false when the edge is not on so many triangles or a new
/// triangle would fold. `v` need not be among `points` yet.
#[allow(clippy::too_many_arguments)]
fn split_kept(
    tris: &mut Vec<[u32; 3]>,
    stars: &mut Stars,
    a: u32,
    b: u32,
    v: u32,
    q: V3,
    points: &[V3],
    sides: usize,
) -> bool {
    let ts = stars.on_edge(tris, a, b);
    if ts.len() != sides {
        return false;
    }
    let p = |i: u32| points[i as usize];
    let normal = |x: V3, y: V3, z: V3| cross(sub(y, x), sub(z, x));
    // Each triangle x y c with the edge wound x to y becomes x v c and
    // v y c.
    let mut halves: Vec<(usize, [u32; 3], [u32; 3])> = Vec::new();
    for &t in &ts {
        let tri = tris[t];
        let (x, y) = if wound(tri, a, b) { (a, b) } else { (b, a) };
        let Some(c) = tri.iter().copied().find(|&w| w != a && w != b) else {
            return false;
        };
        let old = normal(p(x), p(y), p(c));
        if !(dot(normal(p(x), q, p(c)), old) > 0.0 && dot(normal(q, p(y), p(c)), old) > 0.0) {
            return false;
        }
        halves.push((t, [x, v, c], [v, y, c]));
    }
    for (t, first, second) in halves {
        let n = tris.len();
        let (y, c) = (second[1], second[2]);
        tris[t] = first;
        tris.push(second);
        stars.unlink(y, t);
        stars.link(y, n);
        stars.link(c, n);
        stars.link(v, t);
        stars.link(v, n);
    }
    true
}

/// Whether the edge `a b` of a surface mesh, with `c` and `d` across it, is
/// no Delaunay edge: the angles at `c` and `d` sum past a half turn.
pub(crate) fn across_too_wide(a: V3, b: V3, c: V3, d: V3) -> bool {
    let angle = |x: V3| {
        let (u, w) = (sub(a, x), sub(b, x));
        (dot(u, w) / (dot(u, u) * dot(w, w)).sqrt().max(1e-300))
            .clamp(-1.0, 1.0)
            .acos()
    };
    angle(c) + angle(d) > std::f64::consts::PI + 1e-9
}

/// Whether triangle `t` runs from `x` to `y` along one of its edges.
fn wound(t: [u32; 3], x: u32, y: u32) -> bool {
    (0..3).any(|k| t[k] == x && t[(k + 1) % 3] == y)
}

/// Flips the kept mesh of a planar face to the constrained Delaunay
/// triangulation of its points under the volume stage's perturbation (see
/// [`delaunay_flips`]), its outline and the samples along an edge of
/// `brep` (by `edge_of`) constrained.
fn exact_flips(
    tris: &mut [[u32; 3]],
    stars: &mut Stars,
    brep: &Brep,
    points: &[V3],
    edge_of: &[u32],
    front: V3,
) {
    let corners = brep.vertices.len() as u32;
    let ends = |c: u32, e: u32| {
        let ends = brep.edges[e as usize].ends;
        ends[0].0 == c || ends[1].0 == c
    };
    let on_edge = |a: u32, b: u32| -> bool {
        let (ea, eb) = (edge_of[a as usize], edge_of[b as usize]);
        match (a < corners, b < corners) {
            (false, false) => ea != NO_EDGE && ea == eb,
            (true, false) => eb != NO_EDGE && ends(a, eb),
            (false, true) => ea != NO_EDGE && ends(b, ea),
            (true, true) => brep
                .edges
                .iter()
                .any(|e| key(e.ends[0].0, e.ends[1].0) == key(a, b)),
        }
    };
    let mut count: FxHashMap<(u32, u32), u32> = FxHashMap::default();
    for t in tris.iter() {
        for k in 0..3 {
            *count.entry(key(t[k], t[(k + 1) % 3])).or_default() += 1;
        }
    }
    let constrained: FxHashSet<(u32, u32)> = count
        .into_iter()
        .filter(|&((a, b), n)| n == 1 || on_edge(a, b))
        .map(|(e, _)| e)
        .collect();
    delaunay_flips(tris, &|i| points[i as usize], &constrained, front);
    *stars = Stars::of(tris);
}

/// The mesh of one face in a round: a slot per local point (the outline's
/// first, then the interior), the face's own points, and triangles over the
/// local points, wound to the face's front.
pub(crate) struct FaceOut {
    pub slots: Vec<Slot>,
    pub own: Vec<V3>,
    pub tris: Vec<[usize; 3]>,
}

/// How a face's chart maps back onto it: the chart of its carrier, or a
/// height field over its facets (a curved face no carrier chart takes).
enum Map<'a> {
    Carrier(Chart),
    Facets(crate::surface::chart::FacetChart<'a>),
}

impl Map<'_> {
    fn lift(&self, q: V2) -> V3 {
        match self {
            Map::Carrier(c) => c.lift(q),
            Map::Facets(c) => c.lift(q),
        }
    }

    fn to_chart(&self, p: V3) -> V2 {
        match self {
            Map::Carrier(c) => c.to_chart(p),
            Map::Facets(c) => c.to2(p),
        }
    }

    /// A point near the face at `q`, good enough for the size there (on
    /// the facets, without the carrier's projection).
    fn near(&self, q: V2) -> V3 {
        match self {
            Map::Carrier(c) => c.near(q),
            Map::Facets(c) => c.on_facets(q),
        }
    }

    /// The direction the face's front faces over a height field chart
    /// (none for a carrier's, measured on the carrier).
    fn facing(&self) -> Option<V3> {
        match self {
            Map::Carrier(_) => None,
            Map::Facets(c) => Some(c.front),
        }
    }

    /// The largest size the chart allows at `q`.
    fn cap(&self, q: V2) -> f64 {
        match self {
            Map::Carrier(c) => c.cap(q),
            Map::Facets(_) => f64::INFINITY,
        }
    }

    fn shrink(&self, q: V2) -> f64 {
        match self {
            Map::Carrier(c) => c.shrink(q),
            Map::Facets(c) => c.shrink(q),
        }
    }

    /// Whether the chart is the face's own plane (its points exact).
    fn exact_plane(&self) -> bool {
        matches!(self, Map::Carrier(c) if c.is_plane())
    }
}

impl Rounds<'_> {
    /// Meshes the composite face of `members` (the root first) on the facets of
    /// them all: its outline runs along the edges its faces do not share.
    fn mesh_composite(&self, members: &[usize]) -> Result<FaceOut, BoundaryError> {
        let (model, domain, params) = (self.model, self.domain, self.params);
        let (points, edges) = (&self.b.points[..], &self.b.edges[..]);
        let brep = &model.brep;
        let root = members[0];
        let (loops, inner_coedges) = self.comps.outline(brep, root);
        let rings: Vec<Vec<u32>> = loops
            .iter()
            .map(|lp| ring(brep, edges, lp.iter().copied()))
            .collect();
        let inner_edges: Vec<u32> = inner_coedges
            .iter()
            .map(|&ce| brep.coedges[ce as usize].edge.0)
            .collect();
        let inner: Vec<Vec<u32>> = inner_edges
            .iter()
            .map(|&e| edges[e as usize].clone())
            .collect();
        let chains: Vec<&[V3]> = inner_edges
            .iter()
            .map(|&e| brep.edges[e as usize].chain.as_slice())
            .collect();
        let facets: Vec<[V3; 3]> = members
            .iter()
            .flat_map(|&f| front_facets(model, f))
            .collect();
        let mut corners: Vec<u32> = members
            .iter()
            .flat_map(|&f| self.face_corners[f].iter().copied())
            .collect();
        corners.sort_unstable();
        corners.dedup();
        let cap = members
            .iter()
            .map(|&f| params.surf_maxh_for(f))
            .fold(f64::INFINITY, f64::min);
        let size3 = |p: V3| domain.h_at_surf(p).min(cap);
        crate::surface::remesh::remesh(
            &facets,
            &rings,
            &inner,
            &chains,
            &corners,
            &self.required[root],
            points,
            &size3,
            REQUIRED_SPACING,
            None,
        )
        .map_err(|why| BoundaryError::Curved {
            face: root as u32,
            kind: "composite",
            why,
        })
    }

    /// Meshes face `fi` in the chart of its carrier, or (where no chart takes
    /// it) on its facets.
    fn mesh_face(&self, fi: usize) -> Result<FaceOut, BoundaryError> {
        let (model, domain, params) = (self.model, self.domain, self.params);
        let (points, edges) = (&self.b.points[..], &self.b.edges[..]);
        let (corners, required) = (&self.face_corners[fi][..], &self.required[fi][..]);
        let brep = &model.brep;
        let face = &brep.faces[fi];
        // The loops, the edges inside the face (a crease, a sheet meeting it)
        // and the corners it only touches, as global ids.
        let in_loop: FxHashSet<u32> = face
            .loops
            .iter()
            .flat_map(|lp| lp.coedges.iter().map(|c| c.0))
            .collect();
        let rings: Vec<Vec<u32>> = face
            .loops
            .iter()
            .map(|lp| ring(brep, edges, lp.coedges.iter().map(|c| c.0)))
            .collect();
        let inner_edges: Vec<u32> = self.face_coedges[fi]
            .iter()
            .filter(|ce| !in_loop.contains(ce))
            .map(|&ce| brep.coedges[ce as usize].edge.0)
            .collect();
        let inner: Vec<Vec<u32>> = inner_edges
            .iter()
            .map(|&e| edges[e as usize].clone())
            .collect();
        let front = face_front(model, fi);
        let cap = params.surf_maxh_for(fi);
        let size3 = |p: V3| domain.h_at_surf(p).min(cap);
        let surface = brep.surface(face.surface);
        let curved = |why: &'static str| BoundaryError::Curved {
            face: fi as u32,
            kind: surface_kind(surface),
            why,
        };
        // Which way the face's front lies from its carrier's normal (measured
        // on its facets: a full barrel's normals sum to nothing).
        let side: f64 = front_facets(model, fi)
            .map(|p| {
                let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                let c: V3 = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0);
                dot(n, surface.closest(c).1)
            })
            .sum::<f64>()
            .signum();
        let mid = |a: u32, b: u32| self.mid(fi, a, b);
        let ctx = Ctx {
            fi,
            points,
            mid: &mid,
            surface,
            size3: &size3,
            params,
            side,
            front,
        };
        // A whole sphere: its two caps, meshed each alone and put together
        // over the points of their equator.
        if rings.is_empty() && inner.is_empty() {
            if let Some(caps) = Chart::sphere_caps(surface, points, corners, &size3) {
                let shared = caps[0].1.own.len();
                let mut outs = Vec::with_capacity(2);
                for (c, d) in caps {
                    outs.push(mesh_domain(&ctx, &Map::Carrier(c), d, required)?);
                }
                return Ok(merge(shared, outs));
            }
        }
        // The chart of the carrier where it has one that takes the face; else
        // a height field over the face's facets.
        let ring_points: Vec<Vec<V3>> = rings
            .iter()
            .map(|r| r.iter().map(|&g| points[g as usize]).collect())
            .collect();
        let samples: Vec<V3> = front_facets(model, fi)
            .map(|p| std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0))
            .collect();
        let carrier = Chart::of(surface, &samples, &ring_points).and_then(|mut c| {
            c.domain(points, &rings, &inner, corners, &size3, params.grade())
                .map(|d| (Map::Carrier(c), d))
        });
        let (map, d) = match carrier {
            Some(x) => x,
            None => {
                let Some(mut chart) = crate::surface::chart::FacetChart::of(model, fi) else {
                    // No chart at all (a scan, a B-spline the parameters do
                    // not chart, such as a closed band of a CAD loft): remeshed
                    // on its own facets, where no piece of a thin part can come
                    // to lie over another, and onto its carrier where it has
                    // one; the finish then puts every point on it.
                    let facets: Vec<[V3; 3]> = front_facets(model, fi).collect();
                    let chains: Vec<&[V3]> = inner_edges
                        .iter()
                        .map(|&e| brep.edges[e as usize].chain.as_slice())
                        .collect();
                    let m = crate::surface::remesh::remesh(
                        &facets,
                        &rings,
                        &inner,
                        &chains,
                        corners,
                        required,
                        points,
                        &size3,
                        REQUIRED_SPACING,
                        (!matches!(surface, rapidmesh_geom::Surface::Discrete(_)))
                            .then_some(surface),
                    )
                    .map_err(curved)?;
                    return Ok(FaceOut {
                        slots: m.slots,
                        own: m.own,
                        tris: m.tris,
                    });
                };
                // The chart's normal turned to the front, by the face's facets.
                let lean = dot(face_front(model, fi), chart.n);
                chart.front = if lean < 0.0 {
                    chart.n.map(|x| -x)
                } else {
                    chart.n
                };
                let mut d = Domain::default();
                let at = |g: u32| (Slot::Global(g), chart.to2(points[g as usize]));
                for r in &rings {
                    d.add_loop(&r.iter().map(|&g| at(g)).collect::<Vec<_>>());
                }
                for c in &inner {
                    d.add_chain(&c.iter().map(|&g| at(g)).collect::<Vec<_>>());
                }
                for &g in corners {
                    let (s, q) = at(g);
                    d.add_point(s, q);
                }
                (Map::Facets(chart), d)
            }
        };
        mesh_domain(&ctx, &map, d, required)
    }
}

/// The pieces of a face (a sphere's caps) as one face mesh: the own points
/// they share (`shared` of them, the same in each) once, each piece's
/// others after.
fn merge(shared: usize, outs: Vec<FaceOut>) -> FaceOut {
    let mut all = FaceOut {
        slots: Vec::new(),
        own: outs
            .first()
            .map_or(Vec::new(), |o| o.own[..shared].to_vec()),
        tris: Vec::new(),
    };
    for out in outs {
        let base = all.slots.len();
        let own_base = all.own.len() as u32;
        all.own.extend_from_slice(&out.own[shared..]);
        all.slots.extend(out.slots.iter().map(|&s| match s {
            Slot::Own(k) if k as usize >= shared => Slot::Own(own_base + k - shared as u32),
            s => s,
        }));
        all.tris
            .extend(out.tris.iter().map(|t| t.map(|i| base + i)));
    }
    all
}

/// What meshing a face's domain needs from the face.
struct Ctx<'a> {
    fi: usize,
    points: &'a [V3],
    /// The edge and middle arc length of an edge segment, by its ends.
    mid: &'a dyn Fn(u32, u32) -> Option<(u32, f64)>,
    surface: &'a rapidmesh_geom::Surface,
    size3: &'a dyn Fn(V3) -> f64,
    params: &'a MeshParams,
    /// Which way the front lies from the carrier's normal.
    side: f64,
    front: V3,
}

/// Meshes a face's domain in its chart: the required points it takes, the
/// 2D mesh, the lift, the winding towards the front, the flips.
fn mesh_domain(
    ctx: &Ctx<'_>,
    map: &Map<'_>,
    d: Domain,
    required: &[V3],
) -> Result<FaceOut, BoundaryError> {
    let (fi, points, surface, size3, params, side, front) = (
        ctx.fi,
        ctx.points,
        ctx.surface,
        ctx.size3,
        ctx.params,
        ctx.side,
        ctx.front,
    );
    // The points a curved face must take, where they fall inside it in the
    // chart and clear of its outline.
    let mut d = d;
    let pip0 = PipRows::build(&d.loops);
    let within = |q: V2| pip0.inside(q);
    for &p in required {
        let q = map.to_chart(p);
        let clear = REQUIRED_SPACING * size3(p);
        let free = d
            .pts
            .iter()
            .all(|x| (x[0] - q[0]).hypot(x[1] - q[1]) > clear);
        if within(q) && free {
            let k = d.add_own(p);
            d.add_point(Slot::Own(k), q);
        }
    }
    let slot_point = |s: Slot, own: &[V3]| -> V3 {
        match s {
            Slot::Global(g) => points[g as usize],
            Slot::Own(k) => own[k as usize],
        }
    };

    // In a curved face's chart the sizes shrink with the tilt, so the lifted
    // triangles keep theirs.
    let target = |q: V2| {
        let h = size3(map.near(q)) * map.shrink(q);
        h.min(map.cap(q).max(1e-3 * h))
    };
    let step = d
        .pts
        .iter()
        .map(|&q| target(q))
        .fold(f64::INFINITY, f64::min);
    let pip = PipRows::build(&d.loops);
    let min_angle = if params.surf_min_angle > 0.0 {
        params.surf_min_angle
    } else {
        28.0
    };
    let nb = d.pts.len();
    let mut segments: Vec<(usize, usize)> = d.segments.iter().copied().collect();
    segments.sort_unstable();
    // Chords of the outline that cross (the face is narrower there than
    // its samples are apart) have no triangulation between them: their
    // edges take samples at their middles first.
    let crossed = crossings(&d.pts, &segments);
    let split: Vec<(u32, f64)> = crossed
        .into_iter()
        .filter_map(|i| {
            let (a, b) = segments[i];
            match (d.slots[a], d.slots[b]) {
                (Slot::Global(a), Slot::Global(b)) => (ctx.mid)(a, b),
                _ => None,
            }
        })
        .collect();
    if !split.is_empty() {
        rapidmesh_exact::log::debug(
            "surface.refine",
            format!(
                "face {fi} ({}): {} crossing chords of {} in its chart",
                surface_kind(surface),
                split.len(),
                segments.len()
            ),
        );
        return Err(BoundaryError::Refine(split));
    }
    let (p2, mut tris) = mesh_constrained(
        d.pts.clone(),
        segments,
        target,
        |q| pip.inside(q),
        step,
        min_angle,
        4,
        12,
    );
    if tris.is_empty() {
        rapidmesh_exact::log::debug(
            "surface.empty",
            format!(
                "face {fi} ({}): no triangle inside its {} loops of {} points",
                surface_kind(surface),
                d.loops.len(),
                d.pts.len()
            ),
        );
        return Err(BoundaryError::Empty { face: fi as u32 });
    }
    // The face's own points: the seam's, then the interior lifted.
    let mut own = d.own.clone();
    let first_inner = own.len() as u32;
    own.extend(p2[nb..].iter().map(|&q| map.lift(q)));
    let slots: Vec<Slot> = d
        .slots
        .iter()
        .copied()
        .chain((0..(p2.len() - nb) as u32).map(|k| Slot::Own(first_inner + k)))
        .collect();
    // A face in a plane of constant coordinate keeps its points exactly on
    // it: the lift through the fitted frame rounds off it, and the volume
    // stage's exact predicates would see a face with creases.
    if map.exact_plane() {
        let fixed: Vec<V3> = d
            .slots
            .iter()
            .filter_map(|&s| match s {
                Slot::Global(g) => Some(points[g as usize]),
                Slot::Own(_) => None,
            })
            .collect();
        for k in 0..3 {
            if let Some(x) = fixed.first().map(|p| p[k]) {
                if fixed.iter().all(|p| p[k] == x) {
                    for p in &mut own[first_inner as usize..] {
                        p[k] = x;
                    }
                }
            }
        }
    }
    // Wound so the lifted triangles face the front region.
    for t in &mut tris {
        if geometry_predicates::orient2d(p2[t[0]], p2[t[1]], p2[t[2]]) < 0.0 {
            t.swap(1, 2);
        }
    }
    let facing: f64 = tris
        .iter()
        .map(|t| {
            let q = t.map(|i| slot_point(slots[i], &own));
            let c: V3 = std::array::from_fn(|k| (q[0][k] + q[1][k] + q[2][k]) / 3.0);
            let n = cross(sub(q[1], q[0]), sub(q[2], q[0]));
            match map.facing() {
                Some(f) => dot(n, f),
                None => dot(n, surface.closest(c).1) * side,
            }
        })
        .sum();
    if facing < 0.0 {
        for t in &mut tris {
            t.swap(1, 2);
        }
    }
    // The constrained Delaunay triangulation of the face under the
    // perturbation of the volume stage: a planar face flips on its exact
    // points, a curved one in its chart.
    let exact = map.exact_plane();
    let at = |i: usize| -> V3 {
        if exact {
            slot_point(slots[i], &own)
        } else {
            [p2[i][0], p2[i][1], 0.0]
        }
    };
    let normal = if exact { front } else { [0.0, 0.0, 1.0] };
    delaunay_flips(&mut tris, &at, &d.segments, normal);
    Ok(FaceOut { slots, own, tris })
}

/// Lawson flips to the constrained Delaunay triangulation of a planar face:
/// every edge but the constrained ones is flipped while the vertex across
/// it lies inside the circle of its triangle, ties decided by the
/// perturbation of [`crate::predicates`] through a point off the plane
/// (the circle is the sphere's trace on the plane; the point off it never
/// decides, the other four being coplanar, so the tie falls to the planar
/// points as it does in the volume).
fn delaunay_flips<I: Copy + Ord + std::hash::Hash>(
    tris: &mut [[I; 3]],
    at: &dyn Fn(I) -> V3,
    constrained: &FxHashSet<(I, I)>,
    normal: V3,
) {
    use crate::predicates::{inside, orient};
    let mut owner: FxHashMap<(I, I), Vec<usize>> = FxHashMap::default();
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            owner.entry((a.min(b), a.max(b))).or_default().push(ti);
        }
    }
    let mut queue: Vec<(I, I)> = owner
        .iter()
        .filter(|(e, ts)| ts.len() == 2 && !constrained.contains(e))
        .map(|(e, _)| *e)
        .collect();
    queue.sort_unstable();
    let len = (0..3)
        .map(|k| normal[k] * normal[k])
        .sum::<f64>()
        .sqrt()
        .max(1e-300);
    let mut guard = 0usize;
    let cap = 64 * tris.len() + 64;
    while let Some((a, b)) = queue.pop() {
        guard += 1;
        if guard > cap {
            break;
        }
        let Some(ts) = owner.get(&(a, b)).cloned() else {
            continue;
        };
        if ts.len() != 2 {
            continue;
        }
        let third = |t: [I; 3]| t.iter().copied().find(|&v| v != a && v != b);
        let (Some(c), Some(d)) = (third(tris[ts[0]]), third(tris[ts[1]])) else {
            continue;
        };
        // The apex off the plane, beside the triangle a b c.
        let (pa, pb, pc, pd) = (at(a), at(b), at(c), at(d));
        let span = (0..3)
            .map(|k| (pa[k] - pc[k]).abs() + (pb[k] - pc[k]).abs())
            .fold(0.0, f64::max);
        let m: V3 = std::array::from_fn(|k| (pa[k] + pb[k] + pc[k]) / 3.0 + span * normal[k] / len);
        let mut t = [pa, pb, pc, m];
        if orient(t[0], t[1], t[2], t[3]) < 0 {
            t.swap(0, 1);
        }
        // Only a strictly convex quad flips: both new triangles turn like
        // the old one around the point off the plane.
        let turn = |x: V3, y: V3, z: V3| orient(x, y, z, m);
        let old = turn(pa, pb, pc);
        let convex = c != d && old != 0 && turn(pc, pa, pd) == old && turn(pd, pb, pc) == old;
        let flip = convex && orient(t[0], t[1], t[2], t[3]) > 0 && inside(t, pd);
        if !flip {
            continue;
        }
        // Replace a b c | b a d by c d b | d c a, keeping the winding.
        let (t0, t1) = (ts[0], ts[1]);
        let wind =
            |t: [I; 3], x: I, y: I| -> bool { (0..3).any(|k| t[k] == x && t[(k + 1) % 3] == y) };
        // t0 holds a -> b or b -> a; write both new triangles in its turn.
        let ab = wind(tris[t0], a, b);
        let (n0, n1) = if ab {
            ([c, a, d], [d, b, c])
        } else {
            ([c, d, a], [d, c, b])
        };
        for (ti, old) in [(t0, tris[t0]), (t1, tris[t1])] {
            for k in 0..3 {
                let (x, y) = (old[k], old[(k + 1) % 3]);
                if let Some(v) = owner.get_mut(&(x.min(y), x.max(y))) {
                    v.retain(|&u| u != ti);
                }
            }
        }
        tris[t0] = n0;
        tris[t1] = n1;
        owner.remove(&(a, b));
        for (ti, t) in [(t0, n0), (t1, n1)] {
            for k in 0..3 {
                let (x, y) = (t[k], t[(k + 1) % 3]);
                let e = (x.min(y), x.max(y));
                owner.entry(e).or_default().push(ti);
                if e != (c.min(d), c.max(d)) && !constrained.contains(&e) {
                    queue.push(e);
                }
            }
        }
    }
}

/// The name of a carrier's kind.
pub(crate) fn surface_kind(s: &rapidmesh_geom::Surface) -> &'static str {
    use rapidmesh_geom::Surface as S;
    match s {
        S::Plane { .. } => "plane",
        S::Cylinder { .. } => "cylinder",
        S::Sphere { .. } => "sphere",
        S::Cone { .. } => "cone",
        S::Torus { .. } => "torus",
        S::Extruded { .. } => "extruded",
        S::Revolved { .. } => "revolved",
        S::Nurbs(_) => "nurbs",
        S::Discrete(_) => "discrete",
        S::Tube { .. } => "tube",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_geom::{solid_box, Scene};

    fn area(b: &Boundary, tris: &[[u32; 3]]) -> f64 {
        tris.iter()
            .map(|t| {
                let p = t.map(|i| b.points[i as usize]);
                let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                0.5 * dot(n, n).sqrt()
            })
            .sum()
    }

    /// A stack of a thick and a very thin box with a box inside the thick
    /// one: every face meshed alone, every region closed, every face of its
    /// full area, and the thin layer costs no more than its faces.
    #[test]
    fn a_thin_stack_closes_face_by_face() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [100.0, 80.0, 40.0]));
        scene.add_solid(solid_box([0.0, 0.0, 40.0], [100.0, 80.0, 40.2]));
        scene.add_solid(solid_box([30.0, 20.0, 10.0], [60.0, 40.0, 20.0]));
        let model = Model::new(scene.assemble());
        let params = MeshParams {
            maxh: 16.0,
            ..Default::default()
        };
        let domain = crate::sizing::build_sizing_domain(&model, &params);
        let b = boundary(&model, &domain, &params).unwrap();
        let brep = &model.brep;
        let mut regions: Vec<u32> = brep
            .faces
            .iter()
            .flat_map(|f| f.regions.map(|r| r.0))
            .filter(|&r| r != 0)
            .collect();
        regions.sort_unstable();
        regions.dedup();
        assert_eq!(regions.len(), 3);
        for r in regions {
            assert_eq!(b.open_edges(brep, r).0, 0, "region {r}");
        }
        for (fi, f) in brep.faces.iter().enumerate() {
            let want: f64 = f
                .facets
                .iter()
                .map(|&t| {
                    let p = model.plc.triangles[t as usize].map(|i| model.plc.vertices[i as usize]);
                    let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                    0.5 * dot(n, n).sqrt()
                })
                .sum();
            let got = area(&b, &b.faces[fi]);
            assert!(
                (got - want).abs() < 1e-9 * want.max(1.0),
                "face {fi}: {got} vs {want}"
            );
        }
        // Sized by 16 on faces of up to 100 x 80: a few hundred triangles,
        // not the tens of thousands the 0.2 layer would ask of a sample
        // dense on its thickness.
        let n: usize = b.faces.iter().map(Vec::len).sum();
        assert!(n < 2000, "{n} triangles");
    }

    fn closed_everywhere(model: &Model, maxh: f64) -> Boundary {
        let params = MeshParams {
            maxh,
            ..Default::default()
        };
        let domain = crate::sizing::build_sizing_domain(model, &params);
        let b = boundary(model, &domain, &params).unwrap();
        let brep = &model.brep;
        let mut regions: Vec<u32> = brep
            .faces
            .iter()
            .flat_map(|f| f.regions.map(|r| r.0))
            .filter(|&r| r != 0)
            .collect();
        regions.sort_unstable();
        regions.dedup();
        for r in regions {
            assert_eq!(b.open_edges(brep, r).0, 0, "region {r}");
        }
        b
    }

    /// Sheets inside a region (an L shaped one floating, a plate standing
    /// on the floor and reaching a wall) and a void through the top: the
    /// sheets are meshed as faces, the walls they touch take their edges and
    /// corners, and the top face takes the hole.
    #[test]
    fn sheets_and_holes_are_faces_of_their_own() {
        use rapidmesh_geom::{sheet_polygon, sheet_rect, FaceTag};
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [10.0, 10.0, 4.0]));
        scene.add_void(solid_box([4.0, 4.0, 2.0], [6.0, 6.0, 5.0]));
        let l = [
            [1.0, 1.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [1.0, 3.0],
        ];
        scene.add_sheet(
            sheet_polygon(&l, &[], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            FaceTag(7),
        );
        scene.add_sheet(
            sheet_rect([0.0, 7.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 2.0]),
            FaceTag(8),
        );
        let model = Model::new(scene.assemble());
        let b = closed_everywhere(&model, 0.7);
        let tagged = |tag: u32| -> f64 {
            model
                .brep
                .faces
                .iter()
                .zip(&b.faces)
                .filter(|(f, _)| f.face_tag.0 == tag)
                .map(|(_, t)| area(&b, t))
                .sum()
        };
        assert!((tagged(7) - 3.0).abs() < 1e-9, "L sheet {}", tagged(7));
        assert!((tagged(8) - 6.0).abs() < 1e-9, "plate {}", tagged(8));
        // Every sheet edge on a wall is an edge of the wall's triangles.
        let mut wall_edges: FxHashSet<(u32, u32)> = FxHashSet::default();
        for (f, tris) in model.brep.faces.iter().zip(&b.faces) {
            if f.regions[0] != f.regions[1] {
                for t in tris {
                    for k in 0..3 {
                        let (a, c) = (t[k], t[(k + 1) % 3]);
                        wall_edges.insert((a.min(c), a.max(c)));
                    }
                }
            }
        }
        let plate = model
            .brep
            .faces
            .iter()
            .position(|f| f.face_tag.0 == 8)
            .unwrap();
        let on_wall = |p: V3| p[0] == 0.0 || p[2] == 0.0;
        let mut shared = 0;
        for t in &b.faces[plate] {
            for k in 0..3 {
                let (a, c) = (t[k], t[(k + 1) % 3]);
                if on_wall(b.points[a as usize]) && on_wall(b.points[c as usize]) {
                    let both_x = b.points[a as usize][0] == 0.0 && b.points[c as usize][0] == 0.0;
                    let both_z = b.points[a as usize][2] == 0.0 && b.points[c as usize][2] == 0.0;
                    if both_x || both_z {
                        assert!(wall_edges.contains(&(a.min(c), a.max(c))), "edge {a} {c}");
                        shared += 1;
                    }
                }
            }
        }
        assert!(shared >= 2, "{shared}");
    }
}
