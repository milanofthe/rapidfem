//! Mesh diagnostics: quality metrics and located defects. Quality (dihedral
//! histogram, slivers, radius-edge ratio), conformity (non-manifold edges per
//! region, region volumes) and faithfulness to the surfaces (straddlers,
//! bridges). Every defect carries its position, so a corpus run shows where a
//! mesh is wrong, not only that it is.

use crate::constants::FIDELITY_REL;
use crate::mesh::TetMesh;

use crate::quality::{quality_stats, QualityStats};
use crate::simplex::tet_min_dihedral;
use rapidmesh_brep::Surface;
use rapidmesh_geom::vec3::{centroid, dist, V3};
use rapidmesh_geom::SurfaceKind;
use rayon::prelude::*;

/// The kind of a located mesh defect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefectKind {
    /// A tet whose smallest dihedral angle is below [`crate::SLIVER_DEG`] (poorly
    /// conditioned; `value` = the angle in degrees).
    Sliver,
    /// A surface edge not shared by exactly two faces (a boundary leak /
    /// non-manifold incidence; `value` = the incidence count).
    NonManifoldEdge,
    /// A boundary face with a vertex OFF its analytic surface: an interior point
    /// leaked into the boundary. `value` = the off-surface distance.
    Straddler,
    /// A boundary face whose vertices all sit ON surfaces but whose INTERIOR
    /// spans far off every one of them: a lid/bridge over a cavity opening
    /// (topologically watertight, geometrically false -- the mold_block
    /// class the vertex-based straddler test is blind to). `value` = the
    /// centroid's off-surface distance.
    BridgeFace,
    /// A point of the input PLC far from every mesh interface: geometry the
    /// mesh lost ([`crate::measure`]). `value` = the distance over the local
    /// mesh size.
    Uncovered,
    /// A mesh interface face far from every PLC facet: geometry the mesh
    /// invented. `value` = the centroid distance over the face's longest edge.
    Excess,
    /// A point on a sharp PLC edge far from every sharp mesh edge: a lost or
    /// smeared crease. `value` = the distance over the local mesh size.
    FeatureMissed,
    /// A labelled mesh face off every PLC facet of its own surface: a face
    /// tagged with the wrong surface. `value` = the centroid distance over
    /// the face's longest edge (infinite if the surface has no facet).
    Mislabeled,
    /// A face with a region on a side that is no face of a tet of that
    /// region: an interface or sheet the tets there do not have, so it is
    /// not embedded in the volume mesh. `value` = the tets of its regions
    /// that have it (it needs one per side, a sheet two).
    LooseFace,
}

impl DefectKind {
    /// The name of the kind, in snake case.
    pub fn name(self) -> &'static str {
        match self {
            DefectKind::Sliver => "sliver",
            DefectKind::NonManifoldEdge => "nonmanifold_edge",
            DefectKind::Straddler => "straddler",
            DefectKind::BridgeFace => "bridge_face",
            DefectKind::Uncovered => "uncovered",
            DefectKind::Excess => "excess",
            DefectKind::FeatureMissed => "feature_missed",
            DefectKind::Mislabeled => "mislabeled",
            DefectKind::LooseFace => "loose_face",
        }
    }
}

/// A defect with its 3D location and a severity `value` (units per [`DefectKind`]).
#[derive(Debug, Clone)]
pub struct Defect {
    pub kind: DefectKind,
    pub pos: V3,
    pub value: f64,
}

/// Quality + conformity diagnostics of a tet mesh, with located defects.
#[derive(Debug, Clone)]
pub struct MeshDiagnostics {
    pub quality: QualityStats,
    pub n_points: usize,
    pub n_faces: usize,
    /// Every surface edge is shared by exactly two faces.
    pub watertight: bool,
    pub n_nonmanifold_edges: usize,
    pub n_straddlers: usize,
    /// Boundary faces bridging far off every analytic surface (see
    /// [`DefectKind::BridgeFace`]).
    pub n_bridge_faces: usize,
    /// Faces the tets of their regions do not have (see
    /// [`DefectKind::LooseFace`]).
    pub n_loose_faces: usize,
    /// Largest distance of a curved boundary face's centroid from its analytic
    /// surface (the chord sagitta -- the realised geometric accuracy vs `tol`).
    pub max_surface_deviation: f64,
    /// Located defects (slivers, non-manifold edges, straddlers).
    pub defects: Vec<Defect>,
}

/// Computes quality + conformity diagnostics with located defects.
pub fn diagnose(mesh: &TetMesh) -> MeshDiagnostics {
    let pt = |i: usize| mesh.points[i];

    // ---- quality, and the slivers as defects ------------------------------
    let quality = quality_stats(mesh);
    let mut defects: Vec<Defect> = quality
        .slivers
        .iter()
        .map(|&t| {
            let p = mesh.tets[t].map(pt);
            Defect {
                kind: DefectKind::Sliver,
                pos: centroid(&p),
                value: tet_min_dihedral(p),
            }
        })
        .collect();

    // ---- conformity: non-manifold surface edges, per region ---------------
    // Each region's boundary (the faces that touch it) must be a closed 2-manifold:
    // every edge shared by exactly TWO of that region's faces. A triple curve --
    // where an interface (region A|B) meets the outer boundary -- carries 3+ faces
    // globally but exactly 2 per region, so it is not a defect. A real non-manifold (a barrel-seam pinch, a crack) shows
    // up within a single region and is still caught. An embedded sheet (the
    // same region on both sides) lies inside its region and bounds nothing,
    // so it is not part of this count.
    let mut region_edge: rustc_hash::FxHashMap<(u32, usize, usize), u32> =
        rustc_hash::FxHashMap::default();
    for f in &mesh.faces {
        if f.regions[0] == f.regions[1] {
            continue;
        }
        for &rt in &f.regions {
            if rt.0 == 0 {
                continue; // the void has no boundary of its own
            }
            for k in 0..3 {
                let (a, b) = (f.tri[k], f.tri[(k + 1) % 3]);
                *region_edge.entry((rt.0, a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
    }
    // Where two bodies touch along a curve, the region around them has four
    // faces at it, two of each body: the geometry pinches there (a contact
    // has no volume between), no leak. Odd counts are leaks, and four faces
    // off every curve are a fold.
    let on_curve: rustc_hash::FxHashSet<(usize, usize)> = mesh
        .curve_edges
        .iter()
        .map(|c| (c.v[0].min(c.v[1]), c.v[0].max(c.v[1])))
        .collect();
    let mut nm: rustc_hash::FxHashMap<(usize, usize), u32> = rustc_hash::FxHashMap::default();
    for (&(_, a, b), &cnt) in &region_edge {
        if cnt != 2 && !(cnt == 4 && on_curve.contains(&(a, b))) {
            let e = nm.entry((a, b)).or_insert(0);
            *e = (*e).max(cnt);
        }
    }
    let n_nonmanifold = nm.len();
    let mut nm: Vec<((usize, usize), u32)> = nm.into_iter().collect();
    nm.sort_unstable();
    for &((a, b), cnt) in &nm {
        defects.push(Defect {
            kind: DefectKind::NonManifoldEdge,
            pos: centroid(&[pt(a), pt(b)]),
            value: cnt as f64,
        });
    }

    // ---- conformity: every face is a face of the tets on its sides --------
    // A face between regions a and b needs a tet of a and one of b on it; a
    // sheet inside region r two tets of r; a face on the outside (region 0)
    // one tet of the region on its other side. Only the faces are indexed,
    // the tets looked up against them.
    let key = |t: [usize; 3]| {
        let mut k = t;
        k.sort_unstable();
        k
    };
    let face_of: rustc_hash::FxHashMap<[usize; 3], usize> = mesh
        .faces
        .iter()
        .enumerate()
        .map(|(i, f)| (key(f.tri), i))
        .collect();
    // A triangle listed twice, with a region on both (a | b and b | c): b has
    // no thickness there (one body cut down to a face of another), so it
    // needs no tet on it.
    let mut listed: rustc_hash::FxHashMap<[usize; 3], Vec<usize>> =
        rustc_hash::FxHashMap::default();
    for (i, f) in mesh.faces.iter().enumerate() {
        listed.entry(key(f.tri)).or_default().push(i);
    }
    let flat = |i: usize, r: rapidmesh_geom::RegionTag| {
        listed[&key(mesh.faces[i].tri)]
            .iter()
            .any(|&j| j != i && mesh.faces[j].regions.contains(&r))
    };
    // The tets of each side's region on each face (a sheet counts both on
    // its first side).
    let mut held: Vec<[u32; 2]> = vec![[0, 0]; mesh.faces.len()];
    for (t, tv) in mesh.tets.iter().enumerate() {
        for f in crate::simplex::TET_FACES {
            if let Some(&i) = face_of.get(&key(f.map(|j| tv[j]))) {
                let r = mesh.tet_regions[t];
                let [a, b] = mesh.faces[i].regions;
                if r == a {
                    held[i][0] += 1;
                } else if r == b {
                    held[i][1] += 1;
                }
            }
        }
    }
    let mut n_loose = 0usize;
    for (i, f) in mesh.faces.iter().enumerate() {
        let [a, b] = f.regions;
        let [ha, hb] = held[i];
        let loose = if a == b {
            a.0 != 0 && ha < 2
        } else {
            (a.0 != 0 && ha == 0 && !flat(i, a)) || (b.0 != 0 && hb == 0 && !flat(i, b))
        };
        if loose {
            n_loose += 1;
            defects.push(Defect {
                kind: DefectKind::LooseFace,
                pos: centroid(&f.tri.map(pt)),
                value: (ha + hb) as f64,
            });
        }
    }

    // ---- conformity: straddlers + surface deviation (curved faces) --------
    // A boundary face whose vertex sits far OFF its analytic surface means an
    // interior point leaked into the boundary (under-sampling). The chord sagitta
    // (centroid off-surface) is the realised geometric accuracy.
    let mut max_dev = 0.0f64;
    let mut n_straddlers = 0usize;
    let mut n_bridge_faces = 0usize;
    // The curved analytic surfaces, used as a best-fit basis. A boundary point is
    // measured against the surface it ACTUALLY lies on (the nearest one), not the
    // face's tagged surface: near an intersection ring a face is easily tagged with
    // the wrong sphere, and a tagged-only test then reports a phantom straddler for
    // every on-surface vertex it mislabels. Planes and faceted faces are left
    // out: they cannot anchor the fit.
    let curved: Vec<Surface> = mesh.surfaces.iter().filter_map(Surface::curved).collect();
    let nearest_off = |q: V3| -> f64 {
        curved
            .iter()
            .map(|s| dist(q, s.closest(q).0))
            .fold(f64::INFINITY, f64::min)
    };
    // Per curved face: its corners, longest edge, the largest distance of a
    // corner and of its centroid from the nearest surface.
    let offs: Vec<Option<([V3; 3], f64, f64, f64)>> = mesh
        .faces
        .par_iter()
        .map(|f| {
            let kind = &mesh.surfaces[f.surface as usize];
            if matches!(kind, SurfaceKind::Plane { .. } | SurfaceKind::Facets) || curved.is_empty()
            {
                return None; // planar faces are exact; deviation is 0
            }
            let v = [pt(f.tri[0]), pt(f.tri[1]), pt(f.tri[2])];
            let longest = (0..3)
                .map(|k| dist(v[k], v[(k + 1) % 3]))
                .fold(0.0f64, f64::max);
            // straddler: a VERTEX off EVERY analytic surface (a genuinely
            // leaked interior point), not merely off this face's tagged surface.
            let vmax_off = v.iter().map(|&q| nearest_off(q)).fold(0.0f64, f64::max);
            Some((v, longest, vmax_off, nearest_off(centroid(&v))))
        })
        .collect();
    for &(v, longest, vmax_off, c_off) in offs.iter().flatten() {
        if longest > 0.0 && vmax_off > FIDELITY_REL * longest {
            n_straddlers += 1;
            defects.push(Defect {
                kind: DefectKind::Straddler,
                pos: centroid(&v),
                value: vmax_off,
            });
        }
        // accuracy: chord sagitta = centroid distance to the nearest true surface
        // (a real bridge face -- flat over a concave crease -- still reports far off).
        max_dev = max_dev.max(c_off);
        // bridge face / lid: every VERTEX passes the straddler test (all on
        // some surface), but the face INTERIOR spans far off everything --
        // the cavity-lid class (mold_block), topologically watertight yet
        // geometrically false. Same relative threshold as the straddler.
        if longest > 0.0 && c_off > FIDELITY_REL * longest && vmax_off <= FIDELITY_REL * longest {
            n_bridge_faces += 1;
            defects.push(Defect {
                kind: DefectKind::BridgeFace,
                pos: centroid(&v),
                value: c_off,
            });
        }
    }

    MeshDiagnostics {
        quality,
        n_points: mesh.points.len(),
        n_faces: mesh.faces.len(),
        watertight: n_nonmanifold == 0,
        n_nonmanifold_edges: n_nonmanifold,
        n_straddlers,
        n_bridge_faces,
        n_loose_faces: n_loose,
        max_surface_deviation: max_dev,
        defects,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_geom::{solid_box, Scene};

    #[test]
    fn box_is_clean_and_watertight() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let model = rapidmesh_brep::Model::try_of_scene(&scene).expect("model");
        let m = crate::mesher::mesh_scene(
            &scene,
            &model,
            &crate::params::MeshParams {
                maxh: 1.0,
                ..Default::default()
            },
        )
        .expect("bottom-up mesh");
        let d = diagnose(&m);
        assert!(
            d.watertight,
            "box must be watertight ({} non-manifold)",
            d.n_nonmanifold_edges
        );
        assert_eq!(d.n_straddlers, 0, "a box has no curved straddlers");
        assert!(
            d.max_surface_deviation < 1e-9,
            "planar faces have zero deviation"
        );
        assert!(d.quality.min_dihedral_deg > 0.0, "well-defined dihedral");
        let vol: f64 = d.quality.per_region.iter().map(|r| r.volume).sum();
        assert!((vol - 24.0).abs() < 1e-6, "box volume 24, got {vol}");
    }

    #[test]
    fn sphere_is_watertight_and_on_surface() {
        use rapidmesh_geom::sphere;
        let mut scene = Scene::new();
        scene.add_solid(sphere([0.0, 0.0, 0.0], 1.0, 24, 12));
        let model = rapidmesh_brep::Model::try_of_scene(&scene).expect("model");
        let m = crate::mesher::mesh_scene(
            &scene,
            &model,
            &crate::params::MeshParams {
                maxh: 0.4,
                tol_surf: 1e-2,
                ..Default::default()
            },
        )
        .expect("bottom-up mesh");
        let d = diagnose(&m);
        assert!(
            d.watertight,
            "sphere must be watertight ({} non-manifold)",
            d.n_nonmanifold_edges
        );
        assert_eq!(d.n_straddlers, 0, "a well-sampled sphere has no straddlers");
        // chord sagitta of a radius-1 sphere at this density is small but nonzero
        assert!(
            d.max_surface_deviation < 0.1,
            "sphere deviation within tol, got {}",
            d.max_surface_deviation
        );
    }
}
