//! The volume of one region from its boundary mesh.

use super::surface::Boundary;
use rapidmesh_brep::Brep;
use rustc_hash::FxHashSet;

/// The points (global ids) on the boundary of region `r`: its faces and the
/// sheets inside it.
pub fn region_points(b: &Boundary, brep: &Brep, r: u32) -> Vec<u32> {
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    for (f, tris) in brep.faces.iter().zip(&b.faces) {
        if f.regions.iter().any(|x| x.0 == r) {
            seen.extend(tris.iter().flatten());
        }
    }
    let mut v: Vec<u32> = seen.into_iter().collect();
    v.sort_unstable();
    v
}

/// What keeps a region's constrained Delaunay tetrahedralization from
/// existing without further points: edge segments that are no edge of the
/// Delaunay tetrahedralization of its points (a segment must be strongly
/// Delaunay), and edges inside its curved faces that are none (a curved
/// face is no planar facet: each of its triangles is a facet of its own,
/// whose edges are segments too).
#[derive(Debug, Default, Clone)]
pub struct Check {
    pub segments: Vec<[u32; 2]>,
    /// Missing edges inside a curved face: the face, the edge, and whether
    /// its other diagonal is a Delaunay edge (flipped, it needs no point).
    pub edges: Vec<(u32, [u32; 2], bool)>,
}

/// The check of region `r` of a boundary.
pub fn check(b: &Boundary, brep: &Brep, r: u32) -> Check {
    check_keeping(b, brep, r).0
}

/// [`check`], with the Delaunay tetrahedralization it was made on and the
/// points it is over (the region's, in global order), for the constrained
/// stage to take up.
pub fn check_keeping(b: &Boundary, brep: &Brep, r: u32) -> (Check, Kept) {
    let ids = region_points(b, brep, r);
    let pts: Vec<[f64; 3]> = ids.iter().map(|&v| b.points[v as usize]).collect();
    let dt = super::delaunay::Delaunay::new(&pts);
    let mut dt_edges: FxHashSet<(u32, u32)> = FxHashSet::default();
    for t in dt.tets() {
        let g = t.map(|v| ids[v as usize]);
        for i in 0..4 {
            for j in i + 1..4 {
                dt_edges.insert((g[i].min(g[j]), g[i].max(g[j])));
            }
        }
    }
    // Every edge on a face of the region: its loops and the edges inside it.
    let on_region: FxHashSet<u32> = brep
        .coedges
        .iter()
        .filter(|c| {
            brep.faces[c.face.0 as usize]
                .regions
                .iter()
                .any(|x| x.0 == r)
        })
        .map(|c| c.edge.0)
        .collect();
    let mut segments = Vec::new();
    let mut seen: FxHashSet<(u32, u32)> = FxHashSet::default();
    for &e in &on_region {
        for w in b.edges[e as usize].windows(2) {
            let key = (w[0].min(w[1]), w[0].max(w[1]));
            if seen.insert(key) && !dt_edges.contains(&key) {
                segments.push([w[0], w[1]]);
            }
        }
    }
    segments.sort_unstable();
    let mut edges = Vec::new();
    for (fi, (f, tris)) in brep.faces.iter().zip(&b.faces).enumerate() {
        let planar = matches!(
            brep.surface(f.surface),
            rapidmesh_brep::Surface::Plane { .. }
        );
        if planar || !f.regions.iter().any(|x| x.0 == r) {
            continue;
        }
        // The third corner on either side of each edge of the face.
        let mut across: rustc_hash::FxHashMap<(u32, u32), Vec<u32>> =
            rustc_hash::FxHashMap::default();
        for t in tris {
            for k in 0..3 {
                let (a, c) = (t[k], t[(k + 1) % 3]);
                across
                    .entry((a.min(c), a.max(c)))
                    .or_default()
                    .push(t[(k + 2) % 3]);
            }
        }
        let mut keys: Vec<&(u32, u32)> = across.keys().collect();
        keys.sort_unstable();
        for &e in keys {
            // Segments are split on their edge, above.
            if dt_edges.contains(&e) || seen.contains(&e) {
                continue;
            }
            let others = &across[&e];
            let flip = others.len() == 2
                && dt_edges.contains(&(others[0].min(others[1]), others[0].max(others[1])));
            edges.push((fi as u32, [e.0, e.1], flip));
        }
    }
    (Check { segments, edges }, Kept { pts, dt })
}

/// A region's Delaunay tetrahedralization and the points it is over.
pub struct Kept {
    pub pts: Vec<[f64; 3]>,
    pub dt: super::delaunay::Delaunay,
}
