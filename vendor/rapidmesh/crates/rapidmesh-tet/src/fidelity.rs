//! Boundary fidelity: how faithfully a mesh reproduces its input PLC.
//!
//! The quality diagnostics ([`crate::diagnostics`]) measure the tets and
//! check that the surface is closed, but a closed surface can still miss part
//! of the geometry (a crater where a region was lost) or smear a crease, and
//! their surface checks only see curved analytic surfaces. This module
//! compares the two surfaces geometrically, without trusting any face label:
//! - the mesh interfaces (where the tet region changes, where the tets end,
//!   and sheet faces) against the PLC facets, in both directions, and on a
//!   curved analytic face against its carrier (the facets are its chords);
//! - the sharp edges of the PLC against the sharp edges of the mesh.
//!
//! Distances are relative to the local mesh size, the longest edge of the
//! nearest mesh interface face. Two surfaces match where they are closer than
//! [`FIDELITY_REL`] of it.

use crate::conform::TetMesh;
use crate::constants::{
    FIDELITY_MESH_SHARP_DEG, FIDELITY_REL, FIDELITY_SAMPLES_PER_FACE, FIDELITY_SHARP_DEG, TET_FACES,
};
use crate::diagnostics::{Defect, DefectKind};
use rapidmesh_brep::index::FacetBvh;
use rapidmesh_csg::Tri;
use rapidmesh_geom::vec3::{cross, dist, dot, len, V3};
use rapidmesh_geom::{SurfaceKind, CREASE_DEG};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Subdivisions per PLC facet edge at most (the samples grow with its square).
const MAX_SPLIT: usize = 32;

/// How faithfully a mesh reproduces its PLC.
#[derive(Debug, Clone, Default)]
pub struct Fidelity {
    /// Largest centroid distance of a mesh interface face from the PLC, over
    /// the face's longest edge.
    pub mesh_to_plc: f64,
    /// Largest distance of a PLC point from the mesh interfaces, over the
    /// local mesh size.
    pub plc_to_mesh: f64,
    /// Share of the mesh interface area that does not lie on the PLC
    /// (invented geometry).
    pub excess_area: f64,
    /// Share of the PLC area the mesh interfaces do not cover (lost geometry).
    pub uncovered_area: f64,
    /// Largest distance of a point on a sharp PLC edge from the sharp mesh
    /// edges, over the local mesh size.
    pub feature_dev: f64,
    /// Share of the sharp PLC edge length without a sharp mesh edge nearby
    /// (lost or smeared creases).
    pub feature_missed: f64,
    /// Share of the labelled mesh face area that lies off every PLC facet of
    /// its own surface (a face tagged with the wrong surface).
    pub mislabeled_area: f64,
    /// Where the surfaces disagree: the worst point per PLC facet and edge,
    /// every mesh face off the PLC or off its own surface.
    pub defects: Vec<Defect>,
}

/// Measures how faithfully `mesh` reproduces the PLC of `model`. A PLC
/// facet carries the surface of its B-rep face, the geometry the mesher is
/// given: a strip the B-rep absorbed into a neighbour (the faceting of a
/// tangent contact) is that neighbour's surface, with no seam of its own.
pub fn measure(mesh: &TetMesh, model: &rapidmesh_brep::Model) -> Fidelity {
    let plc = &model.plc;
    let mut label: Vec<u32> = plc.surface_refs.iter().map(|s| s.0).collect();
    for f in &model.brep.faces {
        for &t in &f.facets {
            label[t as usize] = f.plc_surface;
        }
    }
    let mpt = |i: usize| mesh.points[i];
    let ppt = |i: usize| plc.vertices[i];
    // The curved analytic carrier of each PLC facet's B-rep face: near its
    // face, the surface is measured there, not on the facets (a mesh finer
    // than the faceting lies off the facets by their sagitta, and is right).
    let mut facet_face: Vec<u32> = vec![u32::MAX; plc.triangles.len()];
    for (fi, f) in model.brep.faces.iter().enumerate() {
        for &t in &f.facets {
            facet_face[t as usize] = fi as u32;
        }
    }
    let carrier = |t: u32| -> Option<&rapidmesh_brep::Surface> {
        let f = *facet_face.get(t as usize)?;
        let face = model.brep.faces.get(f as usize)?;
        let s = model.brep.surface(face.surface);
        (!matches!(
            s,
            rapidmesh_brep::Surface::Plane { .. } | rapidmesh_brep::Surface::Discrete(_)
        ))
        .then_some(s)
    };
    // The distance of `p` from the surface near facet `t` found `d` away:
    // from its carrier where it has one and `p` is within `reach` of it.
    let true_dist = |p: V3, t: u32, d: f64, reach: f64| -> f64 {
        match carrier(t) {
            Some(s) if d <= reach => d.min(dist(p, s.closest(p).0)),
            _ => d,
        }
    };
    let mtris = interfaces(mesh);
    let ptris: Vec<[usize; 3]> = plc
        .triangles
        .iter()
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .collect();
    let mut fid = Fidelity::default();
    let plc_area: f64 = ptris.iter().map(|t| area(corners(&ppt, t))).sum();
    if mtris.is_empty() {
        fid.uncovered_area = if plc_area > 0.0 { 1.0 } else { 0.0 };
        return fid;
    }

    // Mesh -> PLC: every interface face centroid against the PLC facets.
    let plc_bvh = FacetBvh::build(
        &ptris
            .iter()
            .map(|t| tri(corners(&ppt, t)))
            .collect::<Vec<_>>(),
    );
    let mesh_long: Vec<f64> = mtris.iter().map(|t| longest(corners(&mpt, t))).collect();
    // The faces closing a filled contact wedge are off the geometry by
    // design: not measured against it.
    let sorted = |t: [usize; 3]| {
        let mut k = t;
        k.sort_unstable();
        k
    };
    let contact: FxHashSet<[usize; 3]> = mesh
        .contact_faces
        .iter()
        .filter_map(|&i| mesh.faces.get(i).map(|f| sorted(f.tri)))
        .collect();
    // Per face: its area, its deviation (relative to its size) and where.
    let faces: Vec<Option<(f64, f64, V3)>> = mtris
        .par_iter()
        .zip(&mesh_long)
        .map(|(t, &l)| {
            if contact.contains(&sorted(*t)) {
                return None;
            }
            let v = corners(&mpt, t);
            let c = centroid(v);
            let d = plc_bvh
                .nearest(c)
                .map_or(f64::INFINITY, |(t, d)| true_dist(c, t, d, l));
            let rel = if l > 0.0 && d.is_finite() { d / l } else { 0.0 };
            Some((area(v), rel, c))
        })
        .collect();
    let (mut mesh_area, mut excess) = (0.0, 0.0);
    for &(a, rel, c) in faces.iter().flatten() {
        mesh_area += a;
        fid.mesh_to_plc = fid.mesh_to_plc.max(rel);
        if rel > FIDELITY_REL {
            excess += a;
            fid.defects.push(Defect {
                kind: DefectKind::Excess,
                pos: c,
                value: rel,
            });
        }
    }
    fid.excess_area = ratio(excess, mesh_area);

    // Labels: every labelled face against the PLC facets of its own surface
    // (one index per surface: a filter on one index for all could not prune
    // before it met a facet of the surface, far into the search).
    let mut facets_of: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (t, &l) in label.iter().enumerate() {
        facets_of.entry(l).or_default().push(t as u32);
    }
    let by_label: FxHashMap<u32, (FacetBvh, Vec<u32>)> = facets_of
        .into_par_iter()
        .map(|(l, ids)| {
            let tris: Vec<Tri> = ids
                .iter()
                .map(|&t| tri(corners(&ppt, &ptris[t as usize])))
                .collect();
            (l, (FacetBvh::build(&tris), ids))
        })
        .collect();
    let faces: Vec<Option<(f64, f64, V3)>> = mesh
        .faces
        .par_iter()
        .map(|sf| {
            if contact.contains(&sorted(sf.tri)) {
                return None;
            }
            let v = corners(&mpt, &sf.tri);
            let (a, l) = (area(v), longest(v));
            let c = centroid(v);
            let rel = by_label
                .get(&sf.surface)
                .and_then(|(bvh, ids)| bvh.nearest(c).map(|(i, d)| (ids[i as usize], d)))
                .map_or(f64::INFINITY, |(t, d)| true_dist(c, t, d, l) / l);
            Some((a, if l > 0.0 { rel } else { 0.0 }, c))
        })
        .collect();
    let (mut labelled, mut mislabeled) = (0.0, 0.0);
    for &(a, rel, c) in faces.iter().flatten() {
        labelled += a;
        if rel > FIDELITY_REL {
            mislabeled += a;
            fid.defects.push(Defect {
                kind: DefectKind::Mislabeled,
                pos: c,
                value: rel,
            });
        }
    }
    fid.mislabeled_area = ratio(mislabeled, labelled);
    // PLC -> mesh: samples on every PLC facet against the interface faces.
    let mesh_bvh = FacetBvh::build(
        &mtris
            .iter()
            .map(|t| tri(corners(&mpt, t)))
            .collect::<Vec<_>>(),
    );
    // A point of the geometry within a face's size of a face closing a
    // filled contact wedge lies in or at the fill (the wedge is material
    // now): not measured.
    let fill_faces: Vec<[usize; 3]> = mesh
        .contact_faces
        .iter()
        .filter_map(|&i| mesh.faces.get(i).map(|f| f.tri))
        .collect();
    let fill_long: Vec<f64> = fill_faces
        .iter()
        .map(|t| longest(corners(&mpt, t)))
        .collect();
    let fill_bvh = FacetBvh::build(
        &fill_faces
            .iter()
            .map(|t| tri(corners(&mpt, t)))
            .collect::<Vec<_>>(),
    );
    let in_fill = |p: V3| {
        fill_bvh
            .nearest(p)
            .is_some_and(|(fi, d)| d <= fill_long[fi as usize])
    };
    let local = |p: V3| -> Option<f64> {
        let (fi, d) = mesh_bvh.nearest(p)?;
        let l = mesh_long[fi as usize];
        (l > 0.0 && !in_fill(p)).then(|| d / l)
    };
    let step = sample_step(
        &mesh_long,
        &ptris,
        &ppt,
        FIDELITY_SAMPLES_PER_FACE * mtris.len(),
    );
    // Per facet: its uncovered area, its largest deviation and the worst
    // uncovered sample.
    let facets: Vec<(f64, f64, Option<(V3, f64)>)> = ptris
        .par_iter()
        .enumerate()
        .map_init(Vec::new, |samples, (ti, t)| {
            let v = corners(&ppt, t);
            let k = splits(longest(v), step);
            samples.clear();
            tri_samples(v, k, samples);
            let w = area(v) / samples.len() as f64;
            let (mut uncovered, mut largest) = (0.0, 0.0f64);
            let mut worst: Option<(V3, f64)> = None;
            // A sample of a curved facet stands for its carrier's point.
            if let Some(s) = carrier(ti as u32) {
                for p in samples.iter_mut() {
                    *p = s.closest(*p).0;
                }
            }
            for &p in samples.iter() {
                let Some(rel) = local(p) else { continue };
                largest = largest.max(rel);
                if rel > FIDELITY_REL {
                    uncovered += w;
                    if worst.is_none_or(|(_, r)| rel > r) {
                        worst = Some((p, rel));
                    }
                }
            }
            (uncovered, largest, worst)
        })
        .collect();
    let mut uncovered = 0.0;
    for &(u, largest, worst) in &facets {
        uncovered += u;
        fid.plc_to_mesh = fid.plc_to_mesh.max(largest);
        if let Some((pos, value)) = worst {
            fid.defects.push(Defect {
                kind: DefectKind::Uncovered,
                pos,
                value,
            });
        }
    }
    fid.uncovered_area = ratio(uncovered, plc_area);

    // Sharp PLC edges against sharp mesh edges. A segment is the degenerate
    // triangle (a, b, b), for which the closest-point clamp is exact.
    // Facets of one analytic curved surface meet at facet seams, never at a
    // crease, however coarse the tessellation. A discrete patch of an import
    // can wrap around a crease through a smooth detour, so there only bends
    // below the import's crease angle are seams.
    let cos_crease = CREASE_DEG.to_radians().cos();
    let seam = |a: u32, b: u32, cos_bend: f64| {
        let (sa, sb) = (label[a as usize], label[b as usize]);
        sa == sb
            && match plc.surfaces[sa as usize] {
                SurfaceKind::Plane { .. } | SurfaceKind::Facets => false,
                SurfaceKind::Discrete(_) => cos_bend > cos_crease,
                _ => true,
            }
    };
    // Creases are where the surface bends or where one surface meets
    // another, however flat the junction (a tangent seam, a shallow cut).
    // Where surfaces meet, the crease is the B-rep edge along its curve, the
    // one the mesher follows (the facets of a tangent contact cross each
    // other in a band around it); bends within a surface come from the PLC.
    let within = |a: u32, b: u32, cos_bend: f64| {
        label[a as usize] != label[b as usize] || seam(a, b, cos_bend)
    };
    let mut creases: Vec<(V3, V3)> = sharp_edges(&ppt, &ptris, FIDELITY_SHARP_DEG, &within)
        .into_iter()
        .map(|e| (ppt(e[0]), ppt(e[1])))
        .collect();
    let surface_of = |c: &rapidmesh_brep::CoEdgeId| {
        model.brep.faces[model.brep.coedge(*c).face.0 as usize].plc_surface
    };
    for e in &model.brep.edges {
        // Only where the surface changes, as on the mesh side (a boundary
        // of face tags or regions within one surface is not measured).
        let first = e.coedges.first().map(surface_of);
        if e.coedges.iter().all(|c| Some(surface_of(c)) == first) {
            continue;
        }
        let pts: Vec<V3> = match crate::brep_mesh::edge_curve(&model.brep, e) {
            Some(c) => {
                let n = 2 * e.chain.len().max(2);
                (0..=n)
                    .map(|i| c.point_at(c.length() * i as f64 / n as f64))
                    .collect()
            }
            None => e.chain.clone(),
        };
        creases.extend(pts.windows(2).map(|w| (w[0], w[1])));
    }
    let mut mesh_sharp = sharp_edges(&mpt, &mtris, FIDELITY_MESH_SHARP_DEG, &|_, _, _| false);
    let labelled: Vec<[usize; 3]> = mesh.faces.iter().map(|f| f.tri).collect();
    mesh_sharp.extend(label_edges(&labelled, |f| mesh.faces[f].surface));
    let seg_bvh = FacetBvh::build(
        &mesh_sharp
            .iter()
            .map(|e| Tri::new(mpt(e[0]), mpt(e[1]), mpt(e[1])))
            .collect::<Vec<_>>(),
    );
    // Per crease: its length, the length missed, the largest deviation and
    // the worst sample.
    let per_crease: Vec<(f64, f64, f64, Option<(V3, f64)>)> = creases
        .par_iter()
        .map(|&(a, b)| {
            let l = dist(a, b);
            let k = splits(l, step);
            let (mut missed, mut largest) = (0.0, 0.0f64);
            let mut worst: Option<(V3, f64)> = None;
            for i in 0..k {
                let s = (i as f64 + 0.5) / k as f64;
                let p: V3 = std::array::from_fn(|j| a[j] + s * (b[j] - a[j]));
                let Some((fi, _)) = mesh_bvh.nearest(p) else {
                    continue;
                };
                let size = mesh_long[fi as usize];
                if size <= 0.0 || in_fill(p) {
                    continue;
                }
                let rel = seg_bvh.nearest_dist(p) / size;
                largest = largest.max(rel);
                if rel > FIDELITY_REL {
                    missed += l / k as f64;
                    if worst.is_none_or(|(_, r)| rel > r) {
                        worst = Some((p, rel));
                    }
                }
            }
            (l, missed, largest, worst)
        })
        .collect();
    let (mut sharp_len, mut missed) = (0.0, 0.0);
    for &(l, m, largest, worst) in &per_crease {
        sharp_len += l;
        missed += m;
        fid.feature_dev = fid.feature_dev.max(largest);
        if let Some((pos, value)) = worst {
            fid.defects.push(Defect {
                kind: DefectKind::FeatureMissed,
                pos,
                value,
            });
        }
    }
    fid.feature_missed = ratio(missed, sharp_len);
    fid
}

/// The mesh interfaces: tet faces where the region changes or the tets end,
/// plus the sheet faces (same region on both sides).
fn interfaces(mesh: &TetMesh) -> Vec<[usize; 3]> {
    // (sorted corners, region, tet << 2 | face), grouped by corners.
    // Written in place: a parallel collect would hold its pieces and the
    // whole at once.
    let mut faces: Vec<([u32; 3], u32, u32)> = vec![([0; 3], 0, 0); 4 * mesh.tets.len()];
    faces
        .par_chunks_mut(4)
        .zip(mesh.tets.par_iter())
        .enumerate()
        .for_each(|(ti, (out, t))| {
            for (fi, f) in TET_FACES.iter().enumerate() {
                let mut k = [t[f[0]] as u32, t[f[1]] as u32, t[f[2]] as u32];
                k.sort_unstable();
                out[fi] = (k, mesh.tet_regions[ti].0, (ti as u32) << 2 | fi as u32);
            }
        });
    // The whole entry as key: the same order in any thread count.
    faces.par_sort_unstable();
    let mut out = Vec::new();
    let mut have: FxHashSet<[u32; 3]> = FxHashSet::default();
    for group in faces.chunk_by(|a, b| a.0 == b.0) {
        if group.len() == 1 || group.iter().any(|f| f.1 != group[0].1) {
            let tf = group[0].2;
            let t = mesh.tets[(tf >> 2) as usize];
            let f = TET_FACES[(tf & 3) as usize];
            out.push([t[f[0]], t[f[1]], t[f[2]]]);
            have.insert(group[0].0);
        }
    }
    for sf in &mesh.faces {
        let mut k = sf.tri.map(|v| v as u32);
        k.sort_unstable();
        if have.insert(k) {
            out.push(sf.tri);
        }
    }
    out
}

/// Edges where a triangle set bends by more than `deg` degrees, or where it
/// ends or branches (not exactly two triangles on the edge). A pair for which
/// `smooth(f, g, cos_bend)` holds is a seam and never sharp.
fn sharp_edges(
    pt: &impl Fn(usize) -> V3,
    tris: &[[usize; 3]],
    deg: f64,
    smooth: &impl Fn(u32, u32, f64) -> bool,
) -> Vec<[usize; 2]> {
    let cos_max = deg.to_radians().cos();
    // (sorted edge, triangle, traversed low -> high)
    let mut edges: Vec<([usize; 2], u32, bool)> = Vec::with_capacity(3 * tris.len());
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.push(([a.min(b), a.max(b)], ti as u32, a < b));
        }
    }
    edges.sort_unstable_by_key(|e| e.0);
    let normal = |ti: u32| {
        let v = corners(pt, &tris[ti as usize]);
        cross(sub3(v[1], v[0]), sub3(v[2], v[0]))
    };
    let mut out = Vec::new();
    for group in edges.chunk_by(|a, b| a.0 == b.0) {
        let sharp = match group {
            [f, g] => {
                // Coherently oriented neighbours traverse their shared edge in
                // opposite directions; flip one otherwise.
                let (n0, mut n1) = (normal(f.1), normal(g.1));
                if f.2 == g.2 {
                    n1 = n1.map(|x| -x);
                }
                let l = len(n0) * len(n1);
                l > 0.0 && {
                    let cos_bend = dot(n0, n1) / l;
                    cos_bend < cos_max && !smooth(f.1, g.1, cos_bend)
                }
            }
            _ => true,
        };
        if sharp {
            out.push(group[0].0);
        }
    }
    out
}

/// Edges where triangles with different labels meet.
fn label_edges(tris: &[[usize; 3]], label: impl Fn(usize) -> u32) -> Vec<[usize; 2]> {
    let mut edges: Vec<([usize; 2], u32)> = Vec::with_capacity(3 * tris.len());
    for (ti, t) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.push(([a.min(b), a.max(b)], label(ti)));
        }
    }
    edges.sort_unstable();
    edges
        .chunk_by(|a, b| a.0 == b.0)
        .filter(|g| g.iter().any(|e| e.1 != g[0].1))
        .map(|g| g[0].0)
        .collect()
}

/// The sample spacing: half the median mesh interface edge, widened until
/// the PLC samples fit in `budget`.
fn sample_step(
    mesh_long: &[f64],
    ptris: &[[usize; 3]],
    ppt: &impl Fn(usize) -> V3,
    budget: usize,
) -> f64 {
    let mut ls: Vec<f64> = mesh_long.iter().copied().filter(|&l| l > 0.0).collect();
    if ls.is_empty() {
        return f64::INFINITY;
    }
    let mid = ls.len() / 2;
    let median = *ls.select_nth_unstable_by(mid, f64::total_cmp).1;
    let step = 0.5 * median;
    let n: usize = ptris
        .iter()
        .map(|t| splits(longest(corners(ppt, t)), step).pow(2))
        .sum();
    if n > budget.max(1) {
        step * (n as f64 / budget.max(1) as f64).sqrt()
    } else {
        step
    }
}

/// Segments per edge of length `l` at spacing `step`.
fn splits(l: f64, step: f64) -> usize {
    if step.is_finite() && step > 0.0 {
        ((l / step).ceil() as usize).clamp(1, MAX_SPLIT)
    } else {
        1
    }
}

/// Centroids of the `k * k` triangles of the regular split of `v` with `k`
/// segments per edge (all of equal area).
fn tri_samples(v: [V3; 3], k: usize, out: &mut Vec<V3>) {
    let (e1, e2) = (sub3(v[1], v[0]), sub3(v[2], v[0]));
    let at = |u: f64, w: f64| -> V3 { std::array::from_fn(|j| v[0][j] + u * e1[j] + w * e2[j]) };
    let kf = k as f64;
    for i in 0..k {
        for j in 0..k - i {
            let (u, w) = (i as f64, j as f64);
            out.push(at((u + 1.0 / 3.0) / kf, (w + 1.0 / 3.0) / kf));
            if i + j + 1 < k {
                out.push(at((u + 2.0 / 3.0) / kf, (w + 2.0 / 3.0) / kf));
            }
        }
    }
}

fn corners(pt: &impl Fn(usize) -> V3, t: &[usize; 3]) -> [V3; 3] {
    [pt(t[0]), pt(t[1]), pt(t[2])]
}

fn tri(v: [V3; 3]) -> Tri {
    Tri::new(v[0], v[1], v[2])
}

fn sub3(a: V3, b: V3) -> V3 {
    std::array::from_fn(|k| a[k] - b[k])
}

fn area(v: [V3; 3]) -> f64 {
    0.5 * len(cross(sub3(v[1], v[0]), sub3(v[2], v[0])))
}

fn longest(v: [V3; 3]) -> f64 {
    dist(v[0], v[1]).max(dist(v[1], v[2])).max(dist(v[2], v[0]))
}

fn centroid(v: [V3; 3]) -> V3 {
    std::array::from_fn(|k| (v[0][k] + v[1][k] + v[2][k]) / 3.0)
}

fn ratio(part: f64, whole: f64) -> f64 {
    if whole > 0.0 {
        part / whole
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regular split covers the triangle with k^2 samples of equal
    /// weight, all inside.
    #[test]
    fn samples_cover_the_triangle() {
        let v = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        for k in 1..6 {
            let mut s = Vec::new();
            tri_samples(v, k, &mut s);
            assert_eq!(s.len(), k * k);
            assert!(s
                .iter()
                .all(|p| p[0] > 0.0 && p[1] > 0.0 && p[0] + p[1] < 1.0));
        }
    }

    /// A folded pair of triangles is sharp, a flat pair is not, whatever
    /// their orientation; a lone triangle's edges are all sharp.
    #[test]
    fn sharp_edges_ignore_orientation() {
        let pts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, -1.0, 0.0],
            [0.5, 0.0, 1.0],
        ];
        let pt = |i: usize| pts[i];
        for flip in [false, true] {
            let other = if flip { [0, 3, 1] } else { [1, 3, 0] };
            assert!(sharp_edges(&pt, &[[0, 1, 2], other], 30.0, &|_, _, _| false).len() == 4);
            let fold = if flip { [0, 4, 1] } else { [1, 4, 0] };
            assert!(sharp_edges(&pt, &[[0, 1, 2], fold], 30.0, &|_, _, _| false).len() == 5);
        }
    }
}
