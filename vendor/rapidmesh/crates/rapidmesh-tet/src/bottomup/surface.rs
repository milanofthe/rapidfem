//! Edges and faces: every B-rep edge sampled once by the size field, every
//! face meshed alone in 2D with the samples of its edges as its fixed
//! boundary. Two faces on an edge take the same samples, so the faces of a
//! model close up without looking at each other.

use crate::conform::MeshParams;
use crate::curve::{distribute_floored, Curve, PolylineCurve};
use crate::domain::DomainTree;
use crate::surf2d::{mesh_constrained, PipRows};
use rapidmesh_brep::{Brep, Model};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

type P2 = [f64; 2];
type P3 = [f64; 3];

/// The surface mesh of a model, by B-rep entity.
#[derive(Debug, Clone, Default)]
pub struct Boundary {
    pub points: Vec<P3>,
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
    /// splitting them makes more than it mends.
    Diverged { left: usize },
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
            BoundaryError::Diverged { left } => {
                write!(f, "{left} edges the regions miss, more each round")
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
    /// in the opposite direction: none when it is a closed surface. Sheets
    /// inside the region count on both sides, so they never open it.
    pub fn open_edges(&self, brep: &Brep, r: u32) -> usize {
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
        let open: Vec<(u32, u32)> = count
            .iter()
            .filter(|(_, &c)| c != 0)
            .map(|(&e, _)| e)
            .collect();
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
                "bottomup.open",
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
        open.len()
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
pub fn boundary_keeping(
    model: &Model,
    domain: &DomainTree,
    params: &MeshParams,
) -> Result<(Boundary, FxHashMap<u32, super::region::Kept>), BoundaryError> {
    let (plc, brep) = (&model.plc, &model.brep);
    let (lo, hi) = plc
        .vertices
        .iter()
        .fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), p| {
            (
                std::array::from_fn(|k| lo[k].min(p[k])),
                std::array::from_fn(|k| hi[k].max(p[k])),
            )
        });
    let extent = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max).max(1e-12);

    // ---- edges: the samples between the corners
    let floor = params.h_floor(extent);
    if rapidmesh_exact::log::level() == Some(rapidmesh_exact::log::Level::Debug) {
        log_feature_sizes(model, floor);
    }
    let grading = if params.grading > 0.0 {
        params.grading
    } else {
        0.5
    };
    // Edges sharing both corners (and closed edges) bound a face only with
    // points between their corners: two samples for a closed edge, one for
    // each of several edges between two corners.
    let mut between: FxHashMap<(u32, u32), usize> = FxHashMap::default();
    for e in &brep.edges {
        let (a, b) = (e.ends[0].0, e.ends[1].0);
        *between.entry((a.min(b), a.max(b))).or_default() += 1;
    }
    let curves: Vec<Option<PolylineCurve>> = brep
        .edges
        .iter()
        .map(|e| PolylineCurve::new(&e.chain))
        .collect();
    let mut arcs: Vec<Vec<f64>> = brep
        .edges
        .par_iter()
        .enumerate()
        .map(|(ei, e)| {
            let Some(c) = &curves[ei] else {
                return Vec::new();
            };
            let cap = params.edge_maxh_for(ei);
            let size = |s: f64| domain.h_at_surf(c.point_at(s)).min(cap);
            let len = c.length();
            let ss = distribute_floored(c, params.edge_tol_for(ei), &size, grading, floor);
            let mut arcs: Vec<f64> = ss.into_iter().filter(|&s| s > 0.0 && s < len).collect();
            let (a, b) = (e.ends[0].0, e.ends[1].0);
            if a == b && arcs.len() < 2 {
                arcs = vec![len / 3.0, 2.0 * len / 3.0];
            } else if between[&(a.min(b), a.max(b))] > 1 && arcs.is_empty() {
                arcs = vec![len / 2.0];
            }
            arcs
        })
        .collect();
    // Samples closer than this to each other or to a corner are one: a
    // segment of no length is no edge of any tetrahedralization.
    let gap = 1e-9 * extent;
    let spaced = |arcs: &mut Vec<f64>, len: f64| {
        arcs.sort_by(f64::total_cmp);
        let mut last = 0.0;
        arcs.retain(|&s| {
            let keep = s - last > gap && len - s > gap;
            if keep {
                last = s;
            }
            keep
        });
    };
    for (a, c) in arcs.iter_mut().zip(&curves) {
        if let Some(c) = c {
            spaced(a, c.length());
        }
    }

    // ---- faces on the edges; an edge segment that is no edge of the
    // Delaunay tetrahedralization of a region it bounds is split and the
    // faces are meshed again, until the region's constrained Delaunay
    // tetrahedralization exists
    let mut regions: Vec<u32> = brep
        .faces
        .iter()
        .flat_map(|f| f.regions.map(|r| r.0))
        .filter(|&r| r != 0)
        .collect();
    regions.sort_unstable();
    regions.dedup();
    let mut face_coedges: Vec<Vec<u32>> = vec![Vec::new(); brep.faces.len()];
    for (ci, c) in brep.coedges.iter().enumerate() {
        face_coedges[c.face.0 as usize].push(ci as u32);
    }
    // Periodic faces: edge classes sampled from their roots, and the faces
    // that copy another.
    let face_edges: Vec<Vec<usize>> = face_coedges
        .iter()
        .map(|cs| {
            cs.iter()
                .map(|&c| brep.coedges[c as usize].edge.0 as usize)
                .collect()
        })
        .collect();
    let classes =
        super::periodic::Classes::new(brep, &curves, &face_edges, &params.periodic, 1e-7 * extent);
    classes.sync(&mut arcs, &curves, &spaced);
    let copies = &classes.copy_of;
    // Faces smaller than the size joined into composites (none with
    // periodic faces), and the edges between them unsampled.
    let comps = if params.periodic.is_empty() {
        let kept = |f: usize| {
            params.surf_maxh.iter().any(|&(id, _)| id as usize == f)
                || params.surf_tol.iter().any(|&(id, _)| id as usize == f)
        };
        super::topology::Composites::new(model, &|p| domain.h_at_surf(p), &kept)
    } else {
        super::topology::Composites::alone(brep)
    };
    for (a, &inside) in arcs.iter_mut().zip(&comps.internal) {
        if inside {
            a.clear();
        }
    }
    // A copied face takes the mesh of its original, which is meshed for it;
    // a face joined into a composite is meshed as its root.
    let with_originals = |dirty: &mut Vec<usize>| {
        let extra: Vec<usize> = dirty
            .iter()
            .filter_map(|&f| copies[f].map(|c| c.0))
            .collect();
        dirty.extend(extra);
        for f in dirty.iter_mut() {
            *f = comps.root[*f];
        }
        dirty.sort_unstable();
        dirty.dedup();
    };
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
    let region_of_face = |fi: usize| brep.faces[fi].regions.map(|r| r.0);
    // Each round remeshes only the faces on a split edge and checks only
    // the regions around them; the rest is kept from the round before.
    let mut cache: Vec<Option<FaceMesh>> = (0..brep.faces.len())
        .map(|f| (comps.root[f] != f).then(FaceMesh::empty))
        .collect();
    let mut dirty: Vec<usize> = (0..brep.faces.len()).collect();
    let mut region_missing: FxHashMap<u32, usize> = FxHashMap::default();
    // Points a curved face must take (its triangles held to being Delaunay).
    let mut required: Vec<Vec<P3>> = vec![Vec::new(); brep.faces.len()];
    // Faces flipped but not remeshed, whose regions the next round checks.
    let mut recheck: Vec<usize> = Vec::new();
    // The edges flips made, by their ends.
    let mut flipped_in: FxHashSet<[[u64; 3]; 2]> = FxHashSet::default();
    let mut rounds = 0;
    // The edges missed at the round before, and the rounds in a row that
    // missed more.
    let mut before = usize::MAX;
    let mut grew = 0;
    // Rounds that took samples a face asked for.
    let mut refines = 0;
    // The Delaunay tetrahedralization of each region at its last check.
    let mut kept: FxHashMap<u32, super::region::Kept> = FxHashMap::default();
    loop {
        let t = rapidmesh_exact::clock::Instant::now();
        with_originals(&mut dirty);
        let b = loop {
            match faces_on(
                model,
                &curves,
                &arcs,
                &face_coedges,
                &face_corners,
                &mut cache,
                &dirty,
                &required,
                copies,
                &comps,
                domain,
                params,
            ) {
                Ok(b) => break b,
                // The samples a face asked for, taken (through their roots
                // in a periodic class), and the faces on those edges meshed
                // again with the ones that asked.
                Err(BoundaryError::Refine(at)) if refines < MAX_REFINES => {
                    refines += 1;
                    let mut grown: FxHashSet<usize> = FxHashSet::default();
                    for (e, arc) in at {
                        let e = e as usize;
                        let Some(c) = &curves[e] else { continue };
                        let (root, s) = classes.to_root(e, arc, c.length());
                        arcs[root].push(s);
                        grown.insert(root);
                    }
                    for &e in &grown {
                        if let Some(c) = &curves[e] {
                            spaced(&mut arcs[e], c.length());
                        }
                    }
                    if classes.any() {
                        classes.sync(&mut arcs, &curves, &spaced);
                        grown = grown.iter().flat_map(|&r| classes.members(r)).collect();
                    }
                    dirty.extend(grown.iter().flat_map(|&e| edge_faces[e].iter().copied()));
                    with_originals(&mut dirty);
                }
                Err(e) => return Err(e),
            }
        };
        rapidmesh_exact::log::stage("bottomup.faces", t.elapsed().as_secs_f64());
        // A face whose mesh has a hole or a fold stays so whatever the
        // rounds do: said at once.
        let seen: Vec<usize> = dirty.iter().chain(&recheck).copied().collect();
        if let Some(e) = broken_face(brep, &b, &seen, &comps) {
            return Err(e);
        }
        let t = rapidmesh_exact::clock::Instant::now();
        let mut touched: Vec<u32> = dirty
            .iter()
            .chain(&recheck)
            .flat_map(|&fi| region_of_face(fi))
            .filter(|&r| r != 0)
            .collect();
        touched.sort_unstable();
        touched.dedup();
        let checked: Vec<(u32, (super::region::Check, super::region::Kept))> = touched
            .par_iter()
            .map(|&r| (r, super::region::check_keeping(&b, brep, r)))
            .collect();
        let checked: Vec<(u32, super::region::Check)> = checked
            .into_iter()
            .map(|(r, (c, k))| {
                kept.insert(r, k);
                (r, c)
            })
            .collect();
        // Only this round's findings name points by this round's ids; a kept
        // region's count only counts (it holds what nothing could fix).
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
        region_missing.extend(
            checked
                .iter()
                .map(|(r, c)| (*r, c.segments.len() + c.edges.len())),
        );
        let left: usize = region_missing.values().sum();
        rapidmesh_exact::log::stage("bottomup.segment_check", t.elapsed().as_secs_f64());
        rapidmesh_exact::log::debug(
            "bottomup.round",
            format!(
                "{rounds}: {} faces, {} segments, {} edges inside missing, e.g. {:?}",
                dirty.len(),
                missing.len(),
                inside.len(),
                inside
                    .iter()
                    .take(3)
                    .map(|&(r, [a, c], _)| (r, b.points[a as usize], b.points[c as usize]))
                    .collect::<Vec<_>>()
            ),
        );
        let max_rounds = MAX_SPLIT_ROUNDS;
        grew = if left > before { grew + 1 } else { 0 };
        before = left;
        if grew >= DIVERGED_ROUNDS {
            return Err(BoundaryError::Diverged { left });
        }
        if (missing.is_empty() && inside.is_empty()) || rounds == max_rounds {
            let log = rapidmesh_exact::log::stat;
            log("bottomup.split_rounds", rounds as f64);
            log("bottomup.segments_missing", left as f64);
            let mut b = b;
            if comps.any() {
                to_members(model, &comps, &mut b);
            }
            return Ok((b, kept));
        }
        let mut at: FxHashMap<(u32, u32), (usize, usize)> = FxHashMap::default();
        for (ei, ids) in b.edges.iter().enumerate() {
            for (k, w) in ids.windows(2).enumerate() {
                at.insert((w[0].min(w[1]), w[0].max(w[1])), (ei, k));
            }
        }
        // A missing segment splits where the point deepest in its diametral
        // ball projects onto it: that point then lies outside the balls of
        // both halves (at the middle where no point is inside, a tie broken
        // by the perturbation).
        let near = (!missing.is_empty()).then(|| PointGrid::new(&b.points));
        let mut split: FxHashMap<(usize, usize), f64> = FxHashMap::default();
        for [a, c] in missing {
            let Some(&x) = at.get(&(a.min(c), a.max(c))) else {
                continue;
            };
            let (pa, pc) = (b.points[a as usize], b.points[c as usize]);
            let t = near
                .as_ref()
                .and_then(|g| g.deepest_in_ball(&b.points, pa, pc, [a, c]))
                .map(|q| {
                    let d = sub(pc, pa);
                    (dot(sub(q, pa), d) / dot(d, d).max(1e-300)).clamp(0.2, 0.8)
                })
                .unwrap_or(0.5);
            split.insert(x, t);
        }
        let mut changed: FxHashSet<usize> = FxHashSet::default();
        // The splits a kept face mesh may take in place: the edge, the
        // segment's ends, the new sample's arc length and place.
        let mut outline: Vec<(usize, [u32; 2], f64, P3)> = Vec::new();
        for ((ei, k), t) in split {
            let Some(c) = &curves[ei] else {
                continue;
            };
            let len = c.length();
            let s = |i: usize| -> f64 {
                if i == 0 {
                    0.0
                } else if i > arcs[ei].len() {
                    len
                } else {
                    arcs[ei][i - 1]
                }
            };
            let (lo, hi) = (s(k), s(k + 1));
            if hi - lo > 2.0 * gap {
                // A split of an edge in a periodic class is its root's.
                let (root, at) = classes.to_root(ei, lo + t * (hi - lo), len);
                arcs[root].push(at);
                changed.insert(root);
                if !classes.any() {
                    let ids = &b.edges[ei];
                    outline.push((ei, [ids[k], ids[k + 1]], at, c.point_at(at)));
                }
            }
        }
        for &ei in &changed {
            if let Some(c) = &curves[ei] {
                spaced(&mut arcs[ei], c.length());
            }
        }
        if classes.any() {
            classes.sync(&mut arcs, &curves, &spaced);
            changed = changed.iter().flat_map(|&r| classes.members(r)).collect();
        }
        dirty = changed
            .iter()
            .flat_map(|&ei| edge_faces[ei].iter().copied())
            .collect();
        // Curved faces whose outline takes the new samples in place. A
        // planar face is meshed afresh (cheap, and it must stay the exact
        // constrained Delaunay triangulation the volume stage expects), so
        // is a discrete one (remeshed on its facets it comes out better)
        // and a face of a periodic class (its copies follow the original).
        let in_place: FxHashSet<usize> = if copies.iter().any(|c| c.is_some()) {
            FxHashSet::default()
        } else {
            dirty
                .iter()
                .copied()
                .filter(|&f| {
                    cache[f].is_some()
                        && !matches!(
                            brep.surface(brep.faces[f].surface),
                            rapidmesh_brep::Surface::Plane { .. }
                                | rapidmesh_brep::Surface::Discrete(_)
                        )
                })
                .collect()
        };
        // An edge inside a curved face that is no Delaunay edge is flipped
        // in the kept mesh where its other diagonal is one, else split at
        // its middle on the carrier; the face keeps the point when it is
        // meshed afresh; where the point on the carrier would fold the face,
        // the edge splits at its middle. An edge short against the size is
        // left (a sharp angle, where splitting would not end).
        let mut edited: Vec<usize> = Vec::new();
        // An edge of a copied face is its original's, moved back.
        let back = copies.iter().any(|c| c.is_some()).then(|| {
            let mut index = crate::mesh3::periodic::PointIndex::new(1e-7 * extent);
            for (i, &p) in b.points.iter().enumerate() {
                index.insert(p, i);
            }
            index
        });
        for (fi, [ga, gb], flip) in inside {
            let (fi, ga, gb) = match (copies[fi as usize], &back) {
                (Some((a, shift)), Some(index)) => {
                    let moved = |g: u32| {
                        let p = b.points[g as usize];
                        let q = [p[0] - shift[0], p[1] - shift[1], p[2] - shift[2]];
                        index.find(q, &|i| b.points[i], 1e-7 * extent)
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
            let Some(m) = cache[f].as_mut() else {
                continue;
            };
            let (pa, pb) = (b.points[ga as usize], b.points[gb as usize]);
            // An edge a flip made is not flipped back (a face between two
            // regions can have its Delaunay edge in one and not the other):
            // it splits.
            if flip && !flipped_in.contains(&edge_key(pa, pb)) {
                if let Some((c, d)) = flip_kept(m, ga, gb, &b.points) {
                    flipped_in.insert(edge_key(c, d));
                    edited.push(f);
                    continue;
                }
            }
            let mid: P3 = std::array::from_fn(|k| 0.5 * (pa[k] + pb[k]));
            let q = brep.surface(brep.faces[f].surface).closest(mid).0;
            let least = 2.0 * REQUIRED_SPACING * domain.h_at_surf(q);
            if dist2(pa, pb) <= least * least {
                continue;
            }
            // On the carrier, else (where that folds the face) on the chord,
            // off the carrier by no more than the edge already is.
            if let Some(q) = [q, mid]
                .into_iter()
                .find(|&x| split_kept(m, ga, gb, x, &b.points))
            {
                required[f].push(q);
                edited.push(f);
            } else {
                rapidmesh_exact::log::debug(
                    "bottomup.split",
                    format!(
                        "face {f} ({}): edge {ga}-{gb} on {:?} triangles kept no split",
                        surface_kind(brep.surface(brep.faces[f].surface)),
                        m.on_edge(ga, gb).map(|x| x.2.len())
                    ),
                );
            }
        }
        // The new samples into the kept meshes, after the edits inside them
        // (which name points by this round's ids); a face that cannot take
        // one is meshed afresh.
        let mut failed: FxHashSet<usize> = FxHashSet::default();
        for &(ei, [ga, gb], at, q) in &outline {
            for &f in &edge_faces[ei] {
                if !in_place.contains(&f) || failed.contains(&f) {
                    continue;
                }
                let Some(m) = cache[f].as_mut() else {
                    failed.insert(f);
                    continue;
                };
                let sample = Fixed::Sample(ei as u32, at.to_bits());
                if split_outline_kept(m, ga, gb, q, &b.points, sample) {
                    legalize(m, m.slots.len() - 1, &b.points);
                } else {
                    failed.insert(f);
                }
            }
        }
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
            for r in region_of_face(fi) {
                region_missing.remove(&r);
            }
        }
        recheck = edited;
        rounds += 1;
    }
}

/// Points in a uniform grid, for the point deepest in a segment's
/// diametral ball.
struct PointGrid {
    lo: P3,
    cell: f64,
    map: FxHashMap<[i64; 3], Vec<u32>>,
}

impl PointGrid {
    fn new(points: &[P3]) -> PointGrid {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for p in points {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let span = (0..3)
            .map(|k| hi[k] - lo[k])
            .fold(0.0, f64::max)
            .max(1e-300);
        let cell = (span / (points.len().max(1) as f64).cbrt()).max(1e-12 * span);
        let mut g = PointGrid {
            lo,
            cell,
            map: FxHashMap::default(),
        };
        for (i, &p) in points.iter().enumerate() {
            g.map.entry(g.key(p)).or_default().push(i as u32);
        }
        g
    }

    fn key(&self, p: P3) -> [i64; 3] {
        std::array::from_fn(|k| ((p[k] - self.lo[k]) / self.cell).floor() as i64)
    }

    /// The point strictly inside the ball on segment `a b` as diameter
    /// that lies deepest in it (none for an empty ball), `skip` aside.
    fn deepest_in_ball(&self, points: &[P3], a: P3, b: P3, skip: [u32; 2]) -> Option<P3> {
        let m: P3 = std::array::from_fn(|k| 0.5 * (a[k] + b[k]));
        let r2 = 0.25 * dist2(a, b);
        let r = r2.sqrt();
        let (lo, hi) = (self.key(m.map(|x| x - r)), self.key(m.map(|x| x + r)));
        let mut best: Option<(f64, P3)> = None;
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    let Some(vs) = self.map.get(&[x, y, z]) else {
                        continue;
                    };
                    for &v in vs {
                        if skip.contains(&v) {
                            continue;
                        }
                        let q = points[v as usize];
                        let depth = r2 - dist2(q, m);
                        if depth > 0.0 && best.is_none_or(|(d, _)| depth > d) {
                            best = Some((depth, q));
                        }
                    }
                }
            }
        }
        best.map(|x| x.1)
    }
}

/// The first face among `faces` whose mesh has an edge with one triangle
/// that is no segment of a B-rep edge (a hole in the face), an edge on more
/// than two triangles, or leaves a segment of its edges out.
/// The triangles of each composite face back to the faces it joined: each
/// to the face of the facet nearest its centroid.
fn to_members(model: &Model, comps: &super::topology::Composites, b: &mut Boundary) {
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
        let mut cents: Vec<(P3, usize)> = Vec::new();
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
        let key = |p: P3| -> [i64; 3] { std::array::from_fn(|k| (p[k] / cell).floor() as i64) };
        let mut grid: FxHashMap<[i64; 3], Vec<usize>> = FxHashMap::default();
        for (i, (c, _)) in cents.iter().enumerate() {
            grid.entry(key(*c)).or_default().push(i);
        }
        let nearest = |p: P3| -> usize {
            let k = key(p);
            let mut best = (f64::INFINITY, root);
            for r in 0..64i64 {
                for x in k[0] - r..=k[0] + r {
                    for y in k[1] - r..=k[1] + r {
                        for z in k[2] - r..=k[2] + r {
                            let off = (x - k[0]).abs().max((y - k[1]).abs()).max((z - k[2]).abs());
                            if off != r {
                                continue;
                            }
                            for &i in grid.get(&[x, y, z]).into_iter().flatten() {
                                let d = dist2(cents[i].0, p);
                                if d < best.0 {
                                    best = (d, cents[i].1);
                                }
                            }
                        }
                    }
                }
                if best.0.sqrt() <= r as f64 * cell {
                    break;
                }
            }
            best.1
        };
        let tris = std::mem::take(&mut b.faces[root]);
        for t in tris {
            let q = t.map(|v| b.points[v as usize]);
            let c: P3 = std::array::from_fn(|k| (q[0][k] + q[1][k] + q[2][k]) / 3.0);
            b.faces[nearest(c)].push(t);
        }
    }
}

fn broken_face(
    brep: &Brep,
    b: &Boundary,
    faces: &[usize],
    comps: &super::topology::Composites,
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
                "bottomup.broken",
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

/// The front side of face `fi`: the summed normal of its facets, turned to
/// match its regions.
fn face_front(model: &Model, fi: usize) -> P3 {
    let (plc, face) = (&model.plc, &model.brep.faces[fi]);
    face.facets.iter().fold([0.0; 3], |s, &t| {
        let p = plc.triangles[t as usize].map(|i| plc.vertices[i as usize]);
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        let n = if plc.region_tags[t as usize] == face.regions {
            n
        } else {
            n.map(|x| -x)
        };
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
/// the boundary gives up.
const DIVERGED_ROUNDS: usize = 3;

/// The boundary with the edges sampled at `arcs` (arc lengths strictly
/// between the corners), the faces in `dirty` meshed on them afresh and the
/// others taken from `cache`.
#[allow(clippy::too_many_arguments)]
/// The axis and coordinate of a plane normal to an axis (to a rounding of
/// its normal), if it is one.
fn axis_plane(s: &rapidmesh_brep::Surface) -> Option<(usize, f64)> {
    let rapidmesh_brep::Surface::Plane { o, normal, .. } = s else {
        return None;
    };
    let k = (0..3).max_by(|&a, &b| normal[a].abs().total_cmp(&normal[b].abs()))?;
    let off: f64 = (0..3).filter(|&j| j != k).map(|j| normal[j].abs()).sum();
    (off <= 1e-12 * normal[k].abs()).then_some((k, o[k]))
}

fn faces_on(
    model: &Model,
    curves: &[Option<PolylineCurve>],
    arcs: &[Vec<f64>],
    face_coedges: &[Vec<u32>],
    face_corners: &[Vec<u32>],
    cache: &mut [Option<FaceMesh>],
    dirty: &[usize],
    required: &[Vec<P3>],
    copies: &[Option<(usize, P3)>],
    comps: &super::topology::Composites,
    domain: &DomainTree,
    params: &MeshParams,
) -> Result<Boundary, BoundaryError> {
    let brep = &model.brep;
    // A point on an axis-aligned plane takes the plane's coordinate exactly:
    // a curve's samples and a corner are on it only to a rounding otherwise,
    // and a plane whose points are off it is no single facet but one per
    // triangle, each of whose edges the regions must then have.
    let on_planes = |mut p: P3, faces: &mut dyn Iterator<Item = usize>| -> P3 {
        for f in faces {
            if let Some((k, x)) = axis_plane(brep.surface(brep.faces[f].surface)) {
                p[k] = x;
            }
        }
        p
    };
    let mut edge_faces: Vec<Vec<usize>> = vec![Vec::new(); brep.edges.len()];
    for c in &brep.coedges {
        edge_faces[c.edge.0 as usize].push(c.face.0 as usize);
    }
    let mut points: Vec<P3> = brep
        .vertices
        .iter()
        .map(|v| on_planes(v.pos, &mut v.faces.iter().map(|f| f.0 as usize)))
        .collect();
    let mut of_sample: FxHashMap<Fixed, u32> = FxHashMap::default();
    let edges: Vec<Vec<u32>> = brep
        .edges
        .iter()
        .enumerate()
        .map(|(ei, e)| {
            let mut ids = vec![e.ends[0].0];
            if let Some(c) = &curves[ei] {
                for &s in &arcs[ei] {
                    let id = points.len() as u32;
                    of_sample.insert(Fixed::Sample(ei as u32, s.to_bits()), id);
                    ids.push(id);
                    points.push(on_planes(
                        c.point_at(s),
                        &mut edge_faces[ei].iter().copied(),
                    ));
                }
            }
            ids.push(e.ends[1].0);
            ids
        })
        .collect();
    let corners = brep.vertices.len() as u32;
    // The sample behind each global id, to key the fixed points of a face
    // across rounds.
    let mut sample_of: Vec<Fixed> = (0..corners).map(Fixed::Corner).collect();
    for (ei, ids) in edges.iter().enumerate() {
        for (k, &id) in ids[1..ids.len() - 1].iter().enumerate() {
            debug_assert_eq!(id as usize, sample_of.len());
            sample_of.push(Fixed::Sample(ei as u32, arcs[ei][k].to_bits()));
        }
    }
    // Per segment of an edge (its ends' global ids): the edge and the arc
    // length at its middle, where a face whose chords cross asks for a
    // sample.
    let mut mids: FxHashMap<(u32, u32), (u32, f64)> = FxHashMap::default();
    for (ei, ids) in edges.iter().enumerate() {
        let Some(c) = &curves[ei] else { continue };
        let arc = |k: usize| match k {
            0 => 0.0,
            k if k == ids.len() - 1 => c.length(),
            k => arcs[ei][k - 1],
        };
        for k in 0..ids.len() - 1 {
            let (a, b) = (ids[k], ids[k + 1]);
            mids.insert(
                (a.min(b), a.max(b)),
                (ei as u32, 0.5 * (arc(k) + arc(k + 1))),
            );
        }
    }
    let fresh: Vec<(usize, Result<FaceMesh, BoundaryError>)> = dirty
        .par_iter()
        .filter(|&&fi| copies[fi].is_none() && comps.root[fi] == fi)
        .map(|&fi| {
            let members = comps.members(fi);
            let m = if members.len() > 1 {
                mesh_composite(
                    model,
                    &members,
                    comps,
                    &points,
                    &edges,
                    face_corners,
                    &required[fi],
                    domain,
                    params,
                )
            } else {
                mesh_face(
                    model,
                    fi,
                    &points,
                    &edges,
                    &mids,
                    &face_coedges[fi],
                    &face_corners[fi],
                    &required[fi],
                    domain,
                    params,
                )
            };
            (
                fi,
                m.map(|m| {
                    // A point on a seam is there once from either side.
                    let mut slots: Vec<Kept> = Vec::new();
                    let mut index: FxHashMap<Kept, usize> = FxHashMap::default();
                    let to: Vec<usize> = m
                        .slots
                        .iter()
                        .map(|&s| {
                            let k = match s {
                                Slot::Global(g) => Kept::Fixed(sample_of[g as usize]),
                                Slot::Own(k) => Kept::Own(k),
                            };
                            *index.entry(k).or_insert_with(|| {
                                slots.push(k);
                                slots.len() - 1
                            })
                        })
                        .collect();
                    FaceMesh {
                        slots,
                        own: m.own,
                        tris: m.tris.iter().map(|t| t.map(|i| to[i])).collect(),
                        ids: Vec::new(),
                        local: FxHashMap::default(),
                        at: Vec::new(),
                        fresh: FxHashMap::default(),
                    }
                }),
            )
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
        return Err(BoundaryError::Refine(needed));
    }
    for (fi, m) in fresh {
        cache[fi] = Some(m?);
    }
    // Each copied face afresh from its original: its fixed points found by
    // position among the corners and samples, its own points moved.
    if copies.iter().any(|c| c.is_some()) {
        let (lo, hi) = points
            .iter()
            .fold(([f64::MAX; 3], [f64::MIN; 3]), |(lo, hi), p| {
                (
                    std::array::from_fn(|k| lo[k].min(p[k])),
                    std::array::from_fn(|k| hi[k].max(p[k])),
                )
            });
        let tol = 1e-7 * (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max).max(1e-12);
        let mut index = crate::mesh3::periodic::PointIndex::new(tol);
        for (i, &p) in points.iter().enumerate() {
            index.insert(p, i);
        }
        for (b, copy) in copies.iter().enumerate() {
            let Some((a, shift)) = *copy else { continue };
            let Some(src) = cache[a].as_ref() else {
                continue;
            };
            let moved = |p: P3| [p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]];
            let mut slots = Vec::with_capacity(src.slots.len());
            for &k in &src.slots {
                slots.push(match k {
                    Kept::Own(o) => Kept::Own(o),
                    Kept::Fixed(f) => {
                        let g = match f {
                            Fixed::Corner(v) => v,
                            _ => of_sample[&f],
                        };
                        let Some(gb) = index.find(moved(points[g as usize]), &|i| points[i], tol)
                        else {
                            return Err(BoundaryError::Periodic { face: b as u32 });
                        };
                        Kept::Fixed(sample_of[gb])
                    }
                });
            }
            let turn = dot(face_front(model, a), face_front(model, b)) < 0.0;
            cache[b] = Some(FaceMesh {
                slots,
                own: src.own.iter().map(|&p| moved(p)).collect(),
                tris: src
                    .tris
                    .iter()
                    .map(|&t| if turn { [t[0], t[2], t[1]] } else { t })
                    .collect(),
                ids: Vec::new(),
                local: FxHashMap::default(),
                at: Vec::new(),
                fresh: FxHashMap::default(),
            });
        }
    }
    let mut faces = Vec::with_capacity(cache.len());
    for m in cache.iter_mut() {
        let m = m.as_mut().expect("every face meshed");
        let base = points.len() as u32;
        points.extend_from_slice(&m.own);
        let ids: Vec<u32> = m
            .slots
            .iter()
            .map(|&k| match k {
                Kept::Fixed(Fixed::Corner(v)) => v,
                Kept::Fixed(f) => of_sample[&f],
                Kept::Own(k) => base + k,
            })
            .collect();
        faces.push(m.tris.iter().map(|t| t.map(|i| ids[i])).collect());
        m.local = ids.iter().enumerate().map(|(i, &g)| (g, i)).collect();
        m.fresh.clear();
        m.at = vec![Vec::new(); ids.len()];
        for (ti, t) in m.tris.iter().enumerate() {
            for &v in t {
                m.at[v].push(ti);
            }
        }
        m.ids = ids;
    }
    // An edge inside a composite face is no edge of the mesh.
    let edges = edges
        .into_iter()
        .zip(&comps.internal)
        .map(|(ids, &inside)| if inside { Vec::new() } else { ids })
        .collect();
    Ok(Boundary {
        points,
        edges,
        faces,
    })
}

/// Flips the edge between global points `ga` and `gb` of a kept face mesh
/// (their two triangles become the two on the other diagonal, wound the
/// same way), returning the ends of the new edge; none when the edge is
/// not inside the face, the other diagonal is an edge already or the new
/// triangles would fold.
fn flip_kept(m: &mut FaceMesh, ga: u32, gb: u32, points: &[P3]) -> Option<(P3, P3)> {
    let (a, b, _) = m.on_edge(ga, gb)?;
    let (c, d) = flip_local(m, a, b, points)?;
    Some((m.pos(c, points), m.pos(d, points)))
}

/// Flips the edge between local points `a` and `b` of a kept face mesh
/// where its two triangles turn the same way after; the new diagonal.
fn flip_local(m: &mut FaceMesh, a: usize, b: usize, points: &[P3]) -> Option<(usize, usize)> {
    let ts: Vec<usize> = m.at[a]
        .iter()
        .copied()
        .filter(|&t| m.tris[t].contains(&b))
        .collect();
    let [t0, t1] = ts[..] else {
        return None;
    };
    let third = |t: [usize; 3]| t.iter().copied().find(|&v| v != a && v != b);
    let (Some(c), Some(d)) = (third(m.tris[t0]), third(m.tris[t1])) else {
        return None;
    };
    // The other diagonal already an edge (round a point of three
    // triangles): the flip would put it on four.
    if c == d || m.at[c].iter().any(|&t| m.tris[t].contains(&d)) {
        return None;
    }
    let (n0, n1) = if wound(m.tris[t0], a, b) {
        ([c, a, d], [d, b, c])
    } else {
        ([c, d, a], [d, c, b])
    };
    let p = |i: usize| m.pos(i, points);
    let normal = |t: [usize; 3]| cross(sub(p(t[1]), p(t[0])), sub(p(t[2]), p(t[0])));
    let old = {
        let (x, y) = (normal(m.tris[t0]), normal(m.tris[t1]));
        [x[0] + y[0], x[1] + y[1], x[2] + y[2]]
    };
    if !(dot(normal(n0), old) > 0.0 && dot(normal(n1), old) > 0.0) {
        return None;
    }
    m.tris[t0] = n0;
    m.tris[t1] = n1;
    m.unlink(a, t1);
    m.unlink(b, t0);
    m.at[c].push(t1);
    m.at[d].push(t0);
    Some((c, d))
}

/// Lawson flips round the new local point `v` of a kept face mesh: each
/// edge across from it whose opposite angles sum past a half turn flips,
/// and the two it then faces are looked at in turn. The outline stays (an
/// edge on one triangle has nothing to flip with), and so does an edge
/// between two fixed points.
fn legalize(m: &mut FaceMesh, v: usize, points: &[P3]) {
    let angle = |m: &FaceMesh, x: usize, y: usize, z: usize| {
        let (px, py, pz) = (m.pos(x, points), m.pos(y, points), m.pos(z, points));
        let (u, w) = (sub(py, px), sub(pz, px));
        (dot(u, w) / (dot(u, u) * dot(w, w)).sqrt().max(1e-300))
            .clamp(-1.0, 1.0)
            .acos()
    };
    let mut stack: Vec<(usize, usize)> = m.at[v]
        .iter()
        .map(|&t| {
            let e: Vec<usize> = m.tris[t].iter().copied().filter(|&x| x != v).collect();
            (e[0], e[1])
        })
        .collect();
    let mut guard = 0;
    while let Some((a, b)) = stack.pop() {
        guard += 1;
        if guard > 256 {
            break;
        }
        let ts: Vec<usize> = m.at[a]
            .iter()
            .copied()
            .filter(|&t| m.tris[t].contains(&b))
            .collect();
        let [t0, t1] = ts[..] else {
            continue;
        };
        let third = |t: [usize; 3]| t.iter().copied().find(|&x| x != a && x != b);
        let (Some(c), Some(d)) = (third(m.tris[t0]), third(m.tris[t1])) else {
            continue;
        };
        if c != v && d != v {
            continue;
        }
        let far = if c == v { d } else { c };
        // An edge between two fixed points may be a segment of an edge
        // inside the face: it stays.
        let fixed = |x: usize| matches!(m.slots[x], Kept::Fixed(_));
        if fixed(a) && fixed(b) {
            continue;
        }
        if angle(m, v, a, b) + angle(m, far, a, b) <= std::f64::consts::PI + 1e-9 {
            continue;
        }
        if flip_local(m, a, b, points).is_some() {
            stack.push((a, far));
            stack.push((far, b));
        }
    }
}

/// Splits the edge between global points `ga` and `gb` of a kept face mesh
/// at `q`, a new point of the face's own (each triangle on the edge
/// becomes two, wound the same way); false when the edge is not inside the
/// face or a new triangle would fold.
fn split_kept(m: &mut FaceMesh, ga: u32, gb: u32, q: P3, points: &[P3]) -> bool {
    let own = Kept::Own(m.own.len() as u32);
    split_kept_as(m, ga, gb, q, points, own, 2)
}

/// Splits the outline segment between global points `ga` and `gb` of a
/// kept face mesh at the new edge sample `sample` at `q`: the triangle on
/// it becomes two, and the face need not be meshed afresh.
fn split_outline_kept(
    m: &mut FaceMesh,
    ga: u32,
    gb: u32,
    q: P3,
    points: &[P3],
    sample: Fixed,
) -> bool {
    split_kept_as(m, ga, gb, q, points, Kept::Fixed(sample), 1)
}

/// Splits the edge between global points `ga` and `gb` of a kept face mesh
/// at `q`, a new point kept as `slot`, where the edge has `sides`
/// triangles and neither half turns over.
fn split_kept_as(
    m: &mut FaceMesh,
    ga: u32,
    gb: u32,
    q: P3,
    points: &[P3],
    slot: Kept,
    sides: usize,
) -> bool {
    let Some((a, b, ts)) = m.on_edge(ga, gb) else {
        return false;
    };
    if ts.len() != sides {
        return false;
    }
    let p = |i: usize| m.pos(i, points);
    let normal = |x: P3, y: P3, z: P3| cross(sub(y, x), sub(z, x));
    // Each triangle x y c with the edge wound x to y becomes x q c and
    // q y c.
    let mut halves: Vec<(usize, [usize; 3], [usize; 3])> = Vec::new();
    let v = m.slots.len();
    for &t in &ts {
        let tri = m.tris[t];
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
    m.slots.push(slot);
    match slot {
        Kept::Own(_) => m.own.push(q),
        Kept::Fixed(_) => {
            m.fresh.insert(v, q);
        }
    }
    m.ids.push(u32::MAX);
    m.at.push(Vec::new());
    for (t, first, second) in halves {
        let n = m.tris.len();
        let (y, c) = (second[1], second[2]);
        m.tris[t] = first;
        m.tris.push(second);
        m.unlink(y, t);
        m.at[y].push(n);
        m.at[c].push(n);
        m.at[v].extend([t, n]);
    }
    true
}

/// An edge by its ends, whichever way round.
fn edge_key(a: P3, b: P3) -> [[u64; 3]; 2] {
    let (x, y) = (a.map(f64::to_bits), b.map(f64::to_bits));
    if x <= y {
        [x, y]
    } else {
        [y, x]
    }
}

/// Whether triangle `t` runs from `x` to `y` along one of its edges.
fn wound(t: [usize; 3], x: usize, y: usize) -> bool {
    (0..3).any(|k| t[k] == x && t[(k + 1) % 3] == y)
}

/// A fixed point of a face, named so it outlives the numbering of a round:
/// a corner, or an edge sample by its arc length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Fixed {
    Corner(u32),
    Sample(u32, u64),
}

/// A face mesh as kept between rounds: a kept slot per local point, the
/// face's own points, and triangles over the local points.
struct FaceMesh {
    slots: Vec<Kept>,
    own: Vec<P3>,
    tris: Vec<[usize; 3]>,
    /// The global id of each local point at the last assembly, the local
    /// point of each global id, and the triangles at each local point.
    ids: Vec<u32>,
    local: FxHashMap<u32, usize>,
    at: Vec<Vec<usize>>,
    /// Edge samples put in since the last assembly (no global id yet), by
    /// local point, and where they lie.
    fresh: FxHashMap<usize, P3>,
}

impl FaceMesh {
    /// The mesh of a face meshed as part of a composite (none of its own).
    fn empty() -> FaceMesh {
        FaceMesh {
            slots: Vec::new(),
            own: Vec::new(),
            tris: Vec::new(),
            ids: Vec::new(),
            local: FxHashMap::default(),
            at: Vec::new(),
            fresh: FxHashMap::default(),
        }
    }

    /// The triangles on the edge between global points `ga` and `gb`, and
    /// the edge's local points.
    fn on_edge(&self, ga: u32, gb: u32) -> Option<(usize, usize, Vec<usize>)> {
        let (&a, &b) = (self.local.get(&ga)?, self.local.get(&gb)?);
        let ts = self.at[a]
            .iter()
            .copied()
            .filter(|&t| self.tris[t].contains(&b))
            .collect();
        Some((a, b, ts))
    }

    /// Where local point `i` lies (a point split in this round has no
    /// global id yet).
    fn pos(&self, i: usize, points: &[P3]) -> P3 {
        match self.slots[i] {
            Kept::Own(k) => self.own[k as usize],
            Kept::Fixed(_) => match self.fresh.get(&i) {
                Some(&p) => p,
                None => points[self.ids[i] as usize],
            },
        }
    }

    fn unlink(&mut self, v: usize, t: usize) {
        self.at[v].retain(|&x| x != t);
    }
}

/// A local point of a kept face mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kept {
    Fixed(Fixed),
    Own(u32),
}

/// A point of a face's chart: one shared with the face's neighbours (a
/// corner or edge sample, by global id) or one of the face's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Slot {
    Global(u32),
    Own(u32),
}

/// The fixed outline of a face in its chart: points with their slots (a
/// global point may appear twice, on either side of a seam), the face's own
/// points so far (a seam's), the constraint segments and the loops for the
/// inside test.
#[derive(Default)]
pub(crate) struct Domain2 {
    pub(crate) pts: Vec<P2>,
    pub(crate) slots: Vec<Slot>,
    pub(crate) own: Vec<P3>,
    pub(crate) segments: FxHashSet<(usize, usize)>,
    pub(crate) loops: Vec<Vec<P2>>,
    index: FxHashMap<Slot, Vec<usize>>,
}

impl Domain2 {
    /// A new point of the face's own, by its index among them.
    pub(crate) fn add_own(&mut self, p: P3) -> u32 {
        self.own.push(p);
        (self.own.len() - 1) as u32
    }

    /// The local index of `slot` at `q`: one per slot and place (a seam
    /// puts a slot in two places a turn apart; the same place reached by
    /// two roundings is one).
    pub(crate) fn add_point(&mut self, slot: Slot, q: P2) -> usize {
        if let Some(ids) = self.index.get(&slot) {
            for &i in ids {
                let p = self.pts[i];
                // Relative to the larger of the two places and the
                // domain's first point: two roundings of a place at the
                // chart's origin (an apex) are one too.
                let scale = [p, q, self.pts[0]]
                    .iter()
                    .fold(0.0f64, |m, x| m.max(x[0].abs()).max(x[1].abs()));
                let tol = 1e-9 * scale.max(1e-300);
                if (p[0] - q[0]).abs() <= tol && (p[1] - q[1]).abs() <= tol {
                    return i;
                }
            }
        }
        self.pts.push(q);
        self.slots.push(slot);
        self.index.entry(slot).or_default().push(self.pts.len() - 1);
        self.pts.len() - 1
    }

    fn add_segment(&mut self, a: usize, b: usize) {
        if a != b {
            self.segments.insert((a.min(b), a.max(b)));
        }
    }

    /// A closed loop of the outline.
    pub(crate) fn add_loop(&mut self, ring: &[(Slot, P2)]) {
        let ids: Vec<usize> = ring.iter().map(|&(s, q)| self.add_point(s, q)).collect();
        for k in 0..ids.len() {
            self.add_segment(ids[k], ids[(k + 1) % ids.len()]);
        }
        self.loops.push(ring.iter().map(|x| x.1).collect());
    }

    /// The loops of the outline from its segments (for a domain built from
    /// chains): open chains (an edge inside) pruned, the rest split into
    /// cycles. The even-odd inside test counts each segment once however
    /// the cycles group them, so an outline that touches itself (at a pole)
    /// is as good as any.
    pub(crate) fn close_loops(&mut self) {
        let mut adj: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for &(a, b) in &self.segments {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
        // Prune open chains.
        let mut ends: Vec<usize> = adj
            .iter()
            .filter(|(_, n)| n.len() == 1)
            .map(|(&v, _)| v)
            .collect();
        while let Some(v) = ends.pop() {
            let Some(ns) = adj.get(&v) else {
                continue;
            };
            if ns.len() != 1 {
                continue;
            }
            let w = ns[0];
            adj.remove(&v);
            if let Some(nw) = adj.get_mut(&w) {
                nw.retain(|&x| x != v);
                if nw.len() == 1 {
                    ends.push(w);
                } else if nw.is_empty() {
                    adj.remove(&w);
                }
            }
        }
        // Hierholzer: walk unused segments until back at the start.
        let mut starts: Vec<usize> = adj.keys().copied().collect();
        starts.sort_unstable();
        for s in starts {
            while adj.get(&s).is_some_and(|n| !n.is_empty()) {
                let mut ring = vec![s];
                let mut cur = s;
                while let Some(next) = adj.get_mut(&cur).and_then(|n| n.pop()) {
                    if let Some(nn) = adj.get_mut(&next) {
                        if let Some(k) = nn.iter().position(|&x| x == cur) {
                            nn.swap_remove(k);
                        }
                    }
                    cur = next;
                    if cur == s {
                        break;
                    }
                    ring.push(cur);
                    if ring.len() > self.pts.len() + 1 {
                        break;
                    }
                }
                if cur == s && ring.len() >= 3 {
                    self.loops.push(ring.iter().map(|&i| self.pts[i]).collect());
                }
            }
        }
    }

    /// An open chain of constraint segments inside the face.
    pub(crate) fn add_chain(&mut self, chain: &[(Slot, P2)]) {
        let ids: Vec<usize> = chain.iter().map(|&(s, q)| self.add_point(s, q)).collect();
        for w in ids.windows(2) {
            self.add_segment(w[0], w[1]);
        }
    }
}

/// The mesh of one face in a round: a slot per local point (the outline's
/// first, then the interior), the face's own points, and triangles over the
/// local points.
struct FaceOut {
    slots: Vec<Slot>,
    own: Vec<P3>,
    tris: Vec<[usize; 3]>,
}

/// How a face's chart maps back onto it.
enum Map<'a> {
    Chart(super::chart::Chart<'a>),
    Unroll(super::unroll::Unroll<'a>),
    Stereo(super::stereo::Stereo),
}

impl Map<'_> {
    fn lift(&self, q: P2) -> P3 {
        match self {
            Map::Chart(c) => c.lift(q),
            Map::Unroll(u) => u.lift(q),
            Map::Stereo(s) => s.lift(q),
        }
    }

    fn to_chart(&self, p: P3) -> P2 {
        match self {
            Map::Chart(c) => c.to2(p),
            Map::Unroll(u) => u.to_chart(p),
            Map::Stereo(s) => s.to_chart(p),
        }
    }

    /// A point near the face at `q`, good enough for the size there (on
    /// the facets, without the carrier's projection).
    fn near(&self, q: P2) -> P3 {
        match self {
            Map::Chart(c) => c.on_facets(q),
            Map::Unroll(u) => u.near(q),
            Map::Stereo(s) => s.lift(q),
        }
    }

    /// The direction the face's front faces over a height field chart
    /// (none for an unrolled one, measured on its carrier).
    fn facing(&self) -> Option<P3> {
        match self {
            Map::Chart(c) => Some(c.front),
            Map::Unroll(_) | Map::Stereo(_) => None,
        }
    }

    /// Whether the chart covers `q` (for a piece of an atlas: falls in one
    /// of its facets).
    fn contains(&self, q: P2) -> bool {
        match self {
            Map::Chart(c) => c.contains(q),
            Map::Unroll(_) | Map::Stereo(_) => true,
        }
    }

    /// The largest size the chart allows at `q`.
    fn cap(&self, q: P2) -> f64 {
        match self {
            Map::Chart(_) | Map::Stereo(_) => f64::INFINITY,
            Map::Unroll(u) => u.cap(q),
        }
    }

    fn shrink(&self, q: P2) -> f64 {
        match self {
            Map::Chart(c) => c.shrink(q),
            Map::Unroll(u) => u.shrink(q),
            Map::Stereo(s) => s.shrink(q),
        }
    }

    /// Whether the chart is the face's own plane (its points exact).
    fn exact_plane(&self) -> bool {
        matches!(self, Map::Chart(c) if !c.is_curved())
    }
}

/// Meshes the composite face of `members` (the root first) on the facets of
/// them all: its outline runs along the edges its faces do not share.
#[allow(clippy::too_many_arguments)]
fn mesh_composite(
    model: &Model,
    members: &[usize],
    comps: &super::topology::Composites,
    points: &[P3],
    edges: &[Vec<u32>],
    face_corners: &[Vec<u32>],
    required: &[P3],
    domain: &DomainTree,
    params: &MeshParams,
) -> Result<FaceOut, BoundaryError> {
    let (plc, brep) = (&model.plc, &model.brep);
    let root = members[0];
    let (loops, inner_coedges) = comps.outline(brep, root);
    let rings: Vec<Vec<u32>> = loops
        .iter()
        .map(|lp| {
            let mut ring: Vec<u32> = Vec::new();
            for &ce in lp {
                let c = &brep.coedges[ce as usize];
                let pts = &edges[c.edge.0 as usize];
                if c.forward {
                    ring.extend(&pts[..pts.len() - 1]);
                } else {
                    ring.extend(pts[1..].iter().rev());
                }
            }
            ring
        })
        .collect();
    let inner_edges: Vec<u32> = inner_coedges
        .iter()
        .map(|&ce| brep.coedges[ce as usize].edge.0)
        .collect();
    let inner: Vec<Vec<u32>> = inner_edges
        .iter()
        .map(|&e| edges[e as usize].clone())
        .collect();
    let chains: Vec<&[P3]> = inner_edges
        .iter()
        .map(|&e| brep.edges[e as usize].chain.as_slice())
        .collect();
    let facets: Vec<[P3; 3]> = members
        .iter()
        .flat_map(|&f| {
            let face = &brep.faces[f];
            face.facets.iter().map(move |&t| {
                let p = plc.triangles[t as usize].map(|i| plc.vertices[i as usize]);
                if plc.region_tags[t as usize] == face.regions {
                    p
                } else {
                    [p[0], p[2], p[1]]
                }
            })
        })
        .collect();
    let mut corners: Vec<u32> = members
        .iter()
        .flat_map(|&f| face_corners[f].iter().copied())
        .collect();
    corners.sort_unstable();
    corners.dedup();
    let cap = members
        .iter()
        .map(|&f| params.surf_maxh_for(f))
        .fold(f64::INFINITY, f64::min);
    let size3 = |p: P3| domain.h_at_surf(p).min(cap);
    let m = super::remesh::remesh(
        &facets,
        &rings,
        &inner,
        &chains,
        &corners,
        required,
        points,
        &size3,
        REQUIRED_SPACING,
        None,
    )
    .map_err(|why| BoundaryError::Curved {
        face: root as u32,
        kind: "composite",
        why,
    })?;
    Ok(FaceOut {
        slots: m.slots,
        own: m.own,
        tris: m.tris,
    })
}

#[allow(clippy::too_many_arguments)]
fn mesh_face(
    model: &Model,
    fi: usize,
    points: &[P3],
    edges: &[Vec<u32>],
    mids: &FxHashMap<(u32, u32), (u32, f64)>,
    coedges: &[u32],
    corners: &[u32],
    required: &[P3],
    domain: &DomainTree,
    params: &MeshParams,
) -> Result<FaceOut, BoundaryError> {
    let (plc, brep) = (&model.plc, &model.brep);
    let face = &brep.faces[fi];
    // The loops, the edges inside the face (a crease, a sheet meeting it)
    // and the corners it only touches, as global ids.
    let mut in_loop: FxHashSet<u32> = FxHashSet::default();
    let rings: Vec<Vec<u32>> = face
        .loops
        .iter()
        .map(|lp| {
            let mut ring: Vec<u32> = Vec::new();
            for &ce in &lp.coedges {
                in_loop.insert(ce.0);
                let c = brep.coedge(ce);
                let pts = &edges[c.edge.0 as usize];
                if c.forward {
                    ring.extend(&pts[..pts.len() - 1]);
                } else {
                    ring.extend(pts[1..].iter().rev());
                }
            }
            ring
        })
        .collect();
    let inner_edges: Vec<u32> = coedges
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
    let size3 = |p: P3| domain.h_at_surf(p).min(cap);
    let surface = brep.surface(face.surface);
    let curved = |why: &'static str| BoundaryError::Curved {
        face: fi as u32,
        kind: surface_kind(surface),
        why,
    };
    // Which way the face's front lies from its carrier's normal (measured
    // on its facets: a full barrel's normals sum to nothing).
    let side: f64 = face
        .facets
        .iter()
        .map(|&t| {
            let p = plc.triangles[t as usize].map(|i| plc.vertices[i as usize]);
            let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
            let n = if plc.region_tags[t as usize] == face.regions {
                n
            } else {
                n.map(|x| -x)
            };
            let c: P3 = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0);
            dot(n, surface.closest(c).1)
        })
        .sum::<f64>()
        .signum();
    let ctx = Ctx {
        fi,
        points,
        mids,
        surface,
        size3: &size3,
        params,
        side,
        front,
    };
    // A part of a sphere projected stereographically; unrolled where the
    // carrier unrolls and the face lies on it as the unrolling takes it (a
    // partial torus does not); else charted.
    let ring_points: Vec<Vec<P3>> = rings
        .iter()
        .map(|r| r.iter().map(|&g| points[g as usize]).collect())
        .collect();
    let projected = super::stereo::Stereo::of(model, fi, &ring_points).map(|s| {
        let mut d = Domain2::default();
        let at = |g: u32| (Slot::Global(g), s.to_chart(points[g as usize]));
        for r in &rings {
            d.add_loop(&r.iter().map(|&g| at(g)).collect::<Vec<_>>());
        }
        for c in &inner {
            d.add_chain(&c.iter().map(|&g| at(g)).collect::<Vec<_>>());
        }
        for &g in corners {
            let (slot, q) = at(g);
            d.add_point(slot, q);
        }
        (Map::Stereo(s), d)
    });
    let unrolled = || {
        super::unroll::Unroll::of(surface).and_then(|mut u| {
            u.domain(points, &rings, &inner, corners, &size3)
                .map(|d| (Map::Unroll(u), d))
        })
    };
    let (map, d) = match projected.or_else(unrolled) {
        Some(x) => x,
        None => {
            let Some(mut chart) = super::chart::Chart::of(model, fi) else {
                // No one chart of a smooth discrete face (a scan) or a
                // NURBS face (a closed band of a CAD loft): remeshed on its
                // own facets, where no piece of a thin part can come to lie
                // over another; the finish then puts every point on the
                // carrier. One with ridges (a loft round the corners of a
                // polygon) goes to the atlas, whose pieces part at them.
                let facets: Vec<[P3; 3]> = face
                    .facets
                    .iter()
                    .map(|&t| {
                        let p = plc.triangles[t as usize].map(|i| plc.vertices[i as usize]);
                        if plc.region_tags[t as usize] == face.regions {
                            p
                        } else {
                            [p[0], p[2], p[1]]
                        }
                    })
                    .collect();
                if matches!(
                    surface,
                    rapidmesh_brep::Surface::Discrete(_) | rapidmesh_brep::Surface::Nurbs(_)
                ) && !super::remesh::ridged(&facets)
                {
                    let chains: Vec<&[P3]> = inner_edges
                        .iter()
                        .map(|&e| brep.edges[e as usize].chain.as_slice())
                        .collect();
                    let m = super::remesh::remesh(
                        &facets,
                        &rings,
                        &inner,
                        &chains,
                        corners,
                        required,
                        points,
                        &size3,
                        REQUIRED_SPACING,
                        matches!(surface, rapidmesh_brep::Surface::Nurbs(_)).then_some(surface),
                    )
                    .map_err(curved)?;
                    return Ok(FaceOut {
                        slots: m.slots,
                        own: m.own,
                        tris: m.tris,
                    });
                }
                // No one chart: an atlas of pieces, meshed each alone and
                // put together over the points of their cuts.
                let (shared, pieces) = super::atlas::atlas(
                    model,
                    fi,
                    points,
                    &rings,
                    &inner,
                    &inner_edges,
                    edges,
                    corners,
                    required,
                    &size3,
                )
                .map_err(|e| match e {
                    super::atlas::AtlasError::Curved(why) => curved(why),
                    super::atlas::AtlasError::Refine(at) => BoundaryError::Refine(at),
                })?;
                rapidmesh_exact::log::debug(
                    "bottomup.atlas",
                    format!(
                        "face {fi} ({}): {} pieces",
                        surface_kind(surface),
                        pieces.len()
                    ),
                );
                let mut outs = Vec::with_capacity(pieces.len());
                for p in pieces {
                    outs.push(mesh_domain(
                        &ctx,
                        &Map::Chart(p.chart),
                        p.domain,
                        &p.required,
                    )?);
                }
                return Ok(merge(shared.len(), outs));
            };
            // The chart's normal turned to the front, by the face's facets.
            let lean: f64 = face
                .facets
                .iter()
                .map(|&t| {
                    let p = plc.triangles[t as usize].map(|i| plc.vertices[i as usize]);
                    let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                    let n = if plc.region_tags[t as usize] == face.regions {
                        n
                    } else {
                        n.map(|x| -x)
                    };
                    dot(n, chart.n)
                })
                .sum();
            chart.front = if lean < 0.0 {
                chart.n.map(|x| -x)
            } else {
                chart.n
            };
            let mut d = Domain2::default();
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
            (Map::Chart(chart), d)
        }
    };
    mesh_domain(&ctx, &map, d, required)
}

/// The pieces of an atlas as one face mesh: the own points they share
/// (`shared` of them, the same in each) once, each piece's others after.
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
    points: &'a [P3],
    /// The edge and middle arc length of each edge segment.
    mids: &'a FxHashMap<(u32, u32), (u32, f64)>,
    surface: &'a rapidmesh_brep::Surface,
    size3: &'a dyn Fn(P3) -> f64,
    params: &'a MeshParams,
    /// Which way the front lies from the carrier's normal.
    side: f64,
    front: P3,
}

/// Meshes a face's domain in its chart: the required points it takes, the
/// 2D mesh, the lift, the winding towards the front, the flips.
fn mesh_domain(
    ctx: &Ctx<'_>,
    map: &Map<'_>,
    d: Domain2,
    required: &[P3],
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
    // A domain without loops (a piece of an atlas) is what its chart covers.
    let loops0 = !d.loops.is_empty();
    let within = |q: P2| {
        if loops0 {
            pip0.inside(q)
        } else {
            map.contains(q)
        }
    };
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
    let slot_point = |s: Slot, own: &[P3]| -> P3 {
        match s {
            Slot::Global(g) => points[g as usize],
            Slot::Own(k) => own[k as usize],
        }
    };

    // In a curved face's chart the sizes shrink with the tilt, so the lifted
    // triangles keep theirs.
    let target = |q: P2| {
        let h = size3(map.near(q)) * map.shrink(q);
        h.min(map.cap(q).max(1e-3 * h))
    };
    let step = d
        .pts
        .iter()
        .map(|&q| target(q))
        .fold(f64::INFINITY, f64::min);
    let pip = PipRows::build(&d.loops);
    let loops = !d.loops.is_empty();
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
    let split: Vec<(u32, f64)> = crossings(&d.pts, &segments)
        .into_iter()
        .filter_map(|i| {
            let (a, b) = segments[i];
            match (d.slots[a], d.slots[b]) {
                (Slot::Global(a), Slot::Global(b)) => ctx.mids.get(&(a.min(b), a.max(b))).copied(),
                _ => None,
            }
        })
        .collect();
    if !split.is_empty() {
        rapidmesh_exact::log::debug(
            "bottomup.refine",
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
        |q| {
            if loops {
                pip.inside(q)
            } else {
                map.contains(q)
            }
        },
        step,
        min_angle,
        4,
        12,
        |_| false,
    );
    if tris.is_empty() {
        rapidmesh_exact::log::debug(
            "bottomup.empty",
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
        let fixed: Vec<P3> = d
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
            let c: P3 = std::array::from_fn(|k| (q[0][k] + q[1][k] + q[2][k]) / 3.0);
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
    let at = |i: usize| -> P3 {
        if exact {
            slot_point(slots[i], &own)
        } else {
            [p2[i][0], p2[i][1], 0.0]
        }
    };
    // Keys in the order of the global ids: the shared points keep theirs,
    // the face's own follow them in order.
    let key = |i: usize| -> u32 {
        match slots[i] {
            Slot::Global(g) => g,
            Slot::Own(k) => (u32::MAX / 2) + k,
        }
    };
    let normal = if exact { front } else { [0.0, 0.0, 1.0] };
    delaunay_flips(&mut tris, &at, &key, &d.segments, normal);
    Ok(FaceOut { slots, own, tris })
}

/// The segments (indices into `segments`) that cross another or pass
/// through a point of it in the plane: an outline whose chords cross (a
/// face narrower than its samples are apart) has no triangulation.
pub(crate) fn crossings(pts: &[P2], segments: &[(usize, usize)]) -> Vec<usize> {
    if segments.len() < 2 {
        return Vec::new();
    }
    let mut lens: Vec<f64> = segments
        .iter()
        .map(|&(a, b)| (pts[a][0] - pts[b][0]).hypot(pts[a][1] - pts[b][1]))
        .collect();
    lens.sort_by(f64::total_cmp);
    let cell = lens[lens.len() / 2].max(1e-300);
    let key = |x: f64| (x / cell).floor() as i64;
    let mut grid: FxHashMap<(i64, i64), Vec<usize>> = FxHashMap::default();
    for (i, &(a, b)) in segments.iter().enumerate() {
        let (p, q) = (pts[a], pts[b]);
        for x in key(p[0].min(q[0]))..=key(p[0].max(q[0])) {
            for y in key(p[1].min(q[1]))..=key(p[1].max(q[1])) {
                grid.entry((x, y)).or_default().push(i);
            }
        }
    }
    let orient = |a: P2, b: P2, c: P2| geometry_predicates::orient2d(a, b, c);
    // Whether `c` lies inside the segment `a b` it is collinear with.
    let within = |a: P2, b: P2, c: P2| {
        let t = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1]) * (b[1] - a[1]);
        let l = (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2);
        t > 0.0 && t < l
    };
    let meet = |i: usize, j: usize| {
        let ((a, b), (c, d)) = (segments[i], segments[j]);
        if a == c || a == d || b == c || b == d {
            return false;
        }
        let (pa, pb, pc, pd) = (pts[a], pts[b], pts[c], pts[d]);
        let (o1, o2) = (orient(pa, pb, pc), orient(pa, pb, pd));
        let (o3, o4) = (orient(pc, pd, pa), orient(pc, pd, pb));
        if o1 * o2 < 0.0 && o3 * o4 < 0.0 {
            return true;
        }
        (o1 == 0.0 && within(pa, pb, pc))
            || (o2 == 0.0 && within(pa, pb, pd))
            || (o3 == 0.0 && within(pc, pd, pa))
            || (o4 == 0.0 && within(pc, pd, pb))
    };
    let mut out: FxHashSet<usize> = FxHashSet::default();
    for ids in grid.values() {
        for (k, &i) in ids.iter().enumerate() {
            for &j in &ids[k + 1..] {
                if meet(i, j) {
                    out.insert(i);
                    out.insert(j);
                }
            }
        }
    }
    let mut out: Vec<usize> = out.into_iter().collect();
    out.sort_unstable();
    out
}

/// Lawson flips to the constrained Delaunay triangulation of a planar face:
/// every edge but the constrained ones is flipped while the vertex across
/// it lies inside the circle of its triangle, ties decided by the
/// perturbation of [`super::predicates`] through a point off the plane
/// with a key below all (the circle is the sphere's trace on the plane, and
/// the tie falls to the largest planar key as it does in the volume).
fn delaunay_flips(
    tris: &mut [[usize; 3]],
    at: &dyn Fn(usize) -> P3,
    key: &dyn Fn(usize) -> u32,
    constrained: &FxHashSet<(usize, usize)>,
    normal: P3,
) {
    use super::predicates::{inside, orient};
    let mut owner: FxHashMap<(usize, usize), Vec<usize>> = FxHashMap::default();
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            owner.entry((a.min(b), a.max(b))).or_default().push(ti);
        }
    }
    let mut queue: Vec<(usize, usize)> = owner
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
        let third = |t: [usize; 3]| t.iter().copied().find(|&v| v != a && v != b);
        let (Some(c), Some(d)) = (third(tris[ts[0]]), third(tris[ts[1]])) else {
            continue;
        };
        // The apex off the plane, beside the triangle a b c.
        let (pa, pb, pc, pd) = (at(a), at(b), at(c), at(d));
        let span = (0..3)
            .map(|k| (pa[k] - pc[k]).abs() + (pb[k] - pc[k]).abs())
            .fold(0.0, f64::max);
        let m: P3 = std::array::from_fn(|k| (pa[k] + pb[k] + pc[k]) / 3.0 + span * normal[k] / len);
        let mut t = [(pa, key(a) + 1), (pb, key(b) + 1), (pc, key(c) + 1), (m, 0)];
        if orient(t[0].0, t[1].0, t[2].0, t[3].0) < 0 {
            t.swap(0, 1);
        }
        // Only a strictly convex quad flips: both new triangles turn like
        // the old one around the point off the plane.
        let turn = |x: P3, y: P3, z: P3| orient(x, y, z, m);
        let old = turn(pa, pb, pc);
        let convex = c != d && old != 0 && turn(pc, pa, pd) == old && turn(pd, pb, pc) == old;
        let flip = convex
            && orient(t[0].0, t[1].0, t[2].0, t[3].0) > 0
            && inside(
                t.map(|x| x.0),
                pd,
                [t[0].1, t[1].1, t[2].1, t[3].1, key(d) + 1],
            );
        if !flip {
            continue;
        }
        // Replace a b c | b a d by c d b | d c a, keeping the winding.
        let (t0, t1) = (ts[0], ts[1]);
        let wind = |t: [usize; 3], x: usize, y: usize| -> bool {
            (0..3).any(|k| t[k] == x && t[(k + 1) % 3] == y)
        };
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
fn surface_kind(s: &rapidmesh_brep::Surface) -> &'static str {
    use rapidmesh_brep::Surface as S;
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

fn dist2(a: P3, b: P3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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

/// The local feature sizes of the model's corners and edges against the size
/// floor (#217, measured before protection uses them).
fn log_feature_sizes(model: &Model, floor: f64) {
    let brep = &model.brep;
    let fs = rapidmesh_brep::feature::FeatureSize::new(model);
    let corners: Vec<(f64, P3)> = (0..brep.vertices.len() as u32)
        .into_par_iter()
        .map(|v| (fs.at_corner(brep, v), brep.vertices[v as usize].pos))
        .collect();
    let edges: Vec<(f64, P3)> = (0..brep.edges.len())
        .into_par_iter()
        .flat_map_iter(|e| {
            let chain = &brep.edges[e].chain;
            let fs = &fs;
            chain.windows(2).map(move |w| {
                let m: P3 = std::array::from_fn(|k| 0.5 * (w[0][k] + w[1][k]));
                (fs.at_edge(brep, e as u32, m), m)
            })
        })
        .collect();
    let summary = |name: &str, xs: &[(f64, P3)]| {
        let mut v: Vec<(f64, P3)> = xs.iter().copied().filter(|x| x.0.is_finite()).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let below = v.iter().filter(|x| x.0 < floor).count();
        let q = |f: f64| {
            v.get(((v.len() as f64 - 1.0) * f) as usize)
                .map_or(f64::NAN, |x| x.0)
        };
        rapidmesh_exact::log::debug(
            "bottomup.features",
            format!(
                "{name}: {} measured, {below} below the floor {floor:.3e}; least {:.3e}, 10% {:.3e}, median {:.3e}; smallest at {:?}",
                v.len(),
                q(0.0),
                q(0.1),
                q(0.5),
                v.iter().take(3).map(|x| x.1).collect::<Vec<_>>()
            ),
        );
    };
    summary("corners", &corners);
    summary("edges", &edges);
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
        let domain = crate::cvt::build_sizing_domain(&model, &params);
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
            assert_eq!(b.open_edges(brep, r), 0, "region {r}");
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
        let domain = crate::cvt::build_sizing_domain(model, &params);
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
            assert_eq!(b.open_edges(brep, r), 0, "region {r}");
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
        let on_wall = |p: P3| p[0] == 0.0 || p[2] == 0.0;
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
