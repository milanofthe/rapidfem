//! The volume of one region from its boundary mesh.

use crate::surface::Boundary;
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
/// whose edges are segments too; a planar face is one facet).
#[derive(Debug, Default, Clone)]
pub struct Check {
    pub segments: Vec<[u32; 2]>,
    /// Missing edges inside a curved face: the face, the edge, and whether
    /// its other diagonal is a Delaunay edge (flipped, it needs no point).
    pub edges: Vec<(u32, [u32; 2], bool)>,
}

/// What keeps region `r` of a boundary from its constrained Delaunay
/// tetrahedralization (see [`Check`]), with the Delaunay tetrahedralization
/// it was made on and the points it is over, for the next round and the
/// constrained stage to take up. The one kept from the round before
/// (`prev`) drops the points the region lost and takes those it gained (the
/// perturbation is a property of the points, so it is the one made
/// afresh), unless so many changed that one made afresh is cheaper.
pub fn check_keeping(b: &Boundary, brep: &Brep, r: u32, prev: Option<Kept>) -> (Check, Kept) {
    let ids = region_points(b, brep, r);
    let bits = |p: &[f64; 3]| p.map(f64::to_bits);
    let id_of: rustc_hash::FxHashMap<[u64; 3], u32> = ids
        .iter()
        .map(|&v| (bits(&b.points[v as usize]), v))
        .collect();
    let updated = prev.and_then(|mut k| {
        let gone: Vec<u32> = (0..k.pts.len() as u32)
            .filter(|&i| !id_of.contains_key(&bits(&k.pts[i as usize])))
            .collect();
        if gone.len() * REBUILD_SHARE > k.pts.len() {
            rapidmesh_exact::log::debug(
                "volume.check",
                format!(
                    "region {r}: {} of {} points gone, made afresh",
                    gone.len(),
                    k.pts.len()
                ),
            );
            return None;
        }
        let had: FxHashSet<[u64; 3]> = k.pts.iter().map(bits).collect();
        let new: Vec<[f64; 3]> = ids
            .iter()
            .map(|&v| b.points[v as usize])
            .filter(|p| !had.contains(&bits(p)))
            .collect();
        rapidmesh_exact::log::debug(
            "volume.check",
            format!(
                "region {r}: {} points gone, {} new, kept",
                gone.len(),
                new.len()
            ),
        );
        let Some(order) = k.dt.update(&gone, &new) else {
            rapidmesh_exact::log::debug(
                "volume.check",
                format!("region {r}: the kept tetrahedralization does not update, made afresh"),
            );
            return None;
        };
        let mut pts: Vec<[f64; 3]> = order.iter().map(|&i| k.pts[i as usize]).collect();
        pts.extend(new);
        Some(Kept { pts, dt: k.dt })
    });
    let kept = updated.unwrap_or_else(|| {
        let pts: Vec<[f64; 3]> = ids.iter().map(|&v| b.points[v as usize]).collect();
        let dt = crate::volume::delaunay::Delaunay::new(&pts);
        Kept { pts, dt }
    });
    // The global id of each point of the tetrahedralization, in its order.
    let gid: Vec<u32> = kept.pts.iter().map(|p| id_of[&bits(p)]).collect();
    let dt = &kept.dt;
    let mut dt_edges: FxHashSet<(u32, u32)> = FxHashSet::default();
    for t in dt.tets() {
        let g = t.map(|v| gid[v as usize]);
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
        if !f.regions.iter().any(|x| x.0 == r) {
            continue;
        }
        // A planar face is a facet of its own, a tilted plane's too, whose
        // points are off it by their rounding only: the wrapping takes its
        // triangles as they are. Asking each inner edge of it to be
        // Delaunay instead splits without end where two faces nearly meet
        // (the walls of a sharp notch, #369).
        if matches!(
            brep.surface(f.surface),
            rapidmesh_geom::Surface::Plane { .. }
        ) {
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
    (Check { segments, edges }, kept)
}

/// A kept tetrahedralization that would lose more than this share of its
/// points (one in so many) is made afresh instead.
const REBUILD_SHARE: usize = 8;

/// A region's Delaunay tetrahedralization and the points it is over.
pub struct Kept {
    pub pts: Vec<[f64; 3]>,
    pub dt: crate::volume::delaunay::Delaunay,
}
