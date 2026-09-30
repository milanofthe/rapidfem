//! Mesh diagnostics: quality metrics plus spatially-located defects.
//!
//! This is the instrument that turns the corpus run into a MAP rather than a
//! pass/fail light, and -- crucially -- it locates WHERE the mesh is wrong, which
//! is exactly the trigger the curved-surface refinement (straddler repair) needs:
//! a straddler is a boundary face whose vertex is OFF its analytic surface (an
//! interior point leaked into the boundary), and its position is where a surface
//! point must be inserted. Quality (dihedral histogram, slivers, radius-edge) and
//! conformity (non-manifold edges, surface deviation, region volumes) round out
//! the picture so every later step is measurable and visible.

use crate::conform::TetMesh;
use rapidmesh_brep::Surface;
use rapidmesh_geom::vec3::{dist, sub, V3};
use rapidmesh_geom::SurfaceKind;
use rayon::prelude::*;

/// The kind of a located mesh defect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefectKind {
    /// A tet whose smallest dihedral angle is below [`SLIVER_DEG`] (poorly
    /// conditioned; `value` = the angle in degrees).
    Sliver,
    /// A surface edge not shared by exactly two faces (a boundary leak /
    /// non-manifold incidence; `value` = the incidence count).
    NonManifoldEdge,
    /// A boundary face with a vertex OFF its analytic surface: an interior point
    /// leaked into the boundary (the restricted Delaunay under-sampled the
    /// surface). `value` = the off-surface distance. The repair site for refinement.
    Straddler,
    /// A boundary face whose vertices all sit ON surfaces but whose INTERIOR
    /// spans far off every one of them: a lid/bridge over a cavity opening
    /// (topologically watertight, geometrically false -- the mold_block
    /// class the vertex-based straddler test is blind to). `value` = the
    /// centroid's off-surface distance.
    BridgeFace,
    /// A point of the input PLC far from every mesh interface: geometry the
    /// mesh lost ([`crate::fidelity`]). `value` = the distance over the local
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
    pub n_tets: usize,
    pub n_points: usize,
    pub n_faces: usize,
    /// Smallest dihedral angle over all tets (degrees).
    pub min_dihedral_deg: f64,
    /// Mean of the per-tet smallest dihedral angle (degrees).
    pub mean_min_dihedral_deg: f64,
    /// Count of per-tet min-dihedral in each 10-degree bin `[0,10),...,[170,180)`.
    pub dihedral_histogram: [usize; 18],
    /// Tets with min dihedral below [`SLIVER_DEG`].
    pub n_slivers: usize,
    /// Largest circumradius / shortest-edge ratio.
    pub max_radius_edge: f64,
    /// Every surface edge is shared by exactly two faces.
    pub watertight: bool,
    pub n_nonmanifold_edges: usize,
    pub n_straddlers: usize,
    /// Boundary faces bridging far off every analytic surface (see
    /// [`DefectKind::BridgeFace`]).
    pub n_bridge_faces: usize,
    /// Largest distance of a curved boundary face's centroid from its analytic
    /// surface (the chord sagitta -- the realised geometric accuracy vs `tol`).
    pub max_surface_deviation: f64,
    /// Total tet volume per region, ascending region tag.
    pub region_volumes: Vec<(u32, f64)>,
    /// Located defects (slivers, non-manifold edges, straddlers).
    pub defects: Vec<Defect>,
}

/// Tets below this smallest-dihedral angle (degrees) are counted as slivers.
pub use crate::constants::SLIVER_DEG;

fn centroid(ps: &[V3]) -> V3 {
    let n = ps.len() as f64;
    std::array::from_fn(|k| ps.iter().map(|p| p[k]).sum::<f64>() / n)
}

/// Smallest dihedral angle of a tet, in degrees (0 for a flat or
/// degenerate tet): at each edge the angle between the two faces through
/// it, from their outward normals (four cross products for the six edges),
/// the arccosine of the largest cosine, so one `acos` per tet. The one
/// implementation the refinement, the improvement and the statistics share.
pub fn tet_min_dihedral(p: [V3; 4]) -> f64 {
    let cross = |a: V3, b: V3| -> V3 {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: V3, b: V3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    // The normal of the face opposite each corner, turned away from it.
    let mut n = [[0.0; 3]; 4];
    let mut len2 = [0.0; 4];
    for k in 0..4 {
        let (a, b, c) = (p[(k + 1) % 4], p[(k + 2) % 4], p[(k + 3) % 4]);
        let m = cross(sub(b, a), sub(c, a));
        let s = dot(m, sub(p[k], a));
        let l = dot(m, m);
        if !(s != 0.0 && l > 0.0) {
            return 0.0;
        }
        n[k] = if s > 0.0 { m.map(|x| -x) } else { m };
        len2[k] = l;
    }
    // The edge off corners k and l lies on the faces opposite them.
    let mut cos = -1.0f64;
    for (k, l) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        cos = cos.max(-dot(n[k], n[l]) / (len2[k] * len2[l]).sqrt());
    }
    cos.clamp(-1.0, 1.0).acos().to_degrees()
}

fn tet_volume(p: [V3; 4]) -> f64 {
    let (a, b, c, d) = (p[0], p[1], p[2], p[3]);
    let (ab, ac, ad) = (sub(b, a), sub(c, a), sub(d, a));
    let cr = [
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    ];
    (ab[0] * cr[0] + ab[1] * cr[1] + ab[2] * cr[2]).abs() / 6.0
}

/// Circumradius / shortest-edge of one tet (`None` if degenerate).
fn radius_edge(p: [V3; 4]) -> Option<f64> {
    // Circumcenter via the linear system on squared-distance differences.
    let a = p[0];
    let mut m = [[0.0f64; 3]; 3];
    let mut rhs = [0.0f64; 3];
    for i in 0..3 {
        let q = p[i + 1];
        for k in 0..3 {
            m[i][k] = 2.0 * (q[k] - a[k]);
        }
        rhs[i] = (0..3).map(|k| q[k] * q[k] - a[k] * a[k]).sum();
    }
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if det.abs() < 1e-30 {
        return None;
    }
    let mut cc = [0.0f64; 3];
    for k in 0..3 {
        let mut mk = m;
        for r in 0..3 {
            mk[r][k] = rhs[r];
        }
        let dk = mk[0][0] * (mk[1][1] * mk[2][2] - mk[1][2] * mk[2][1])
            - mk[0][1] * (mk[1][0] * mk[2][2] - mk[1][2] * mk[2][0])
            + mk[0][2] * (mk[1][0] * mk[2][1] - mk[1][1] * mk[2][0]);
        cc[k] = dk / det;
    }
    let r = dist(cc, a);
    let mut lmin = f64::MAX;
    for i in 0..4 {
        for j in i + 1..4 {
            lmin = lmin.min(dist(p[i], p[j]));
        }
    }
    (lmin > 0.0).then(|| r / lmin)
}

/// Computes quality + conformity diagnostics with located defects.
pub fn diagnose(mesh: &TetMesh) -> MeshDiagnostics {
    let pt = |i: usize| mesh.points[i];

    // ---- quality: dihedral histogram, slivers, radius-edge ----------------
    let mut hist = [0usize; 18];
    let mut min_dih = f64::MAX;
    let mut sum_dih = 0.0;
    let mut max_re = 0.0f64;
    let mut n_slivers = 0usize;
    let mut defects: Vec<Defect> = Vec::new();
    let mut region_vol: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
    // Per tet: its smallest dihedral, radius-edge ratio and volume.
    let per_tet: Vec<(f64, Option<f64>, f64)> = mesh
        .tets
        .par_iter()
        .map(|t| {
            let p = [pt(t[0]), pt(t[1]), pt(t[2]), pt(t[3])];
            (tet_min_dihedral(p), radius_edge(p), tet_volume(p))
        })
        .collect();
    for (ti, (t, &(md, re, vol))) in mesh.tets.iter().zip(&per_tet).enumerate() {
        let p = [pt(t[0]), pt(t[1]), pt(t[2]), pt(t[3])];
        if md.is_finite() {
            min_dih = min_dih.min(md);
            sum_dih += md;
            let bin = ((md / 10.0).floor() as usize).min(17);
            hist[bin] += 1;
            if md < SLIVER_DEG {
                n_slivers += 1;
                defects.push(Defect {
                    kind: DefectKind::Sliver,
                    pos: centroid(&p),
                    value: md,
                });
            }
        }
        if let Some(re) = re {
            max_re = max_re.max(re);
        }
        *region_vol.entry(mesh.tet_regions[ti].0).or_insert(0.0) += vol;
    }
    let mean_dih = if mesh.tets.is_empty() {
        0.0
    } else {
        sum_dih / mesh.tets.len() as f64
    };

    // Sliver census (RAPIDMESH_SLIVER_CENSUS): classify every sliver tet by
    // its surface entanglement -- how many of its vertices sit on surface
    // faces, whether all four share ONE analytic surface (the on-carrier
    // chord/cap configuration no smoothing can lift), and how many of its
    // facets are constrained. The histogram picks the repair strategy.
    if std::env::var_os("RAPIDMESH_SLIVER_CENSUS").is_some() {
        use std::collections::{HashMap, HashSet};
        // vertex -> surfaces of its incident faces
        let mut vsurf: HashMap<usize, Vec<u32>> = HashMap::new();
        let mut ftris: HashSet<[usize; 3]> = HashSet::new();
        for f in &mesh.faces {
            for &v in &f.tri {
                let e = vsurf.entry(v).or_default();
                if !e.contains(&f.surface) {
                    e.push(f.surface);
                }
            }
            let mut k = f.tri;
            k.sort_unstable();
            ftris.insert(k);
        }
        let mut census: HashMap<(usize, bool, usize), usize> = HashMap::new();
        for t in &mesh.tets {
            let p = [pt(t[0]), pt(t[1]), pt(t[2]), pt(t[3])];
            let md = tet_min_dihedral(p);
            if !md.is_finite() || md >= SLIVER_DEG {
                continue;
            }
            let on_surf = t.iter().filter(|v| vsurf.contains_key(v)).count();
            // all four on ONE common surface?
            let common = t
                .iter()
                .filter_map(|v| vsurf.get(v))
                .fold(None::<Vec<u32>>, |acc, s| match acc {
                    None => Some(s.clone()),
                    Some(a) => Some(a.into_iter().filter(|x| s.contains(x)).collect()),
                })
                .is_some_and(|c| on_surf == 4 && !c.is_empty());
            let mut cfaces = 0usize;
            for i in 0..4 {
                let mut f: [usize; 3] = [t[(i + 1) % 4], t[(i + 2) % 4], t[(i + 3) % 4]];
                f.sort_unstable();
                if ftris.contains(&f) {
                    cfaces += 1;
                }
            }
            *census.entry((on_surf, common, cfaces)).or_insert(0) += 1;
        }
        let mut rows: Vec<_> = census.into_iter().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.1));
        eprintln!("SLIVER CENSUS (on_surf_verts, all4_one_surface, constrained_facets) -> count");
        for ((os, com, cf), n) in rows.iter().take(12) {
            eprintln!("  ({os}, {com}, {cf}) -> {n}");
        }
    }

    // ---- conformity: non-manifold surface edges, PER REGION (Diag5) -------
    // Each region's boundary (the faces that touch it) must be a closed 2-manifold:
    // every edge shared by exactly TWO of that region's faces. A triple curve --
    // where an interface (region A|B) meets the outer boundary -- carries 3+ faces
    // GLOBALLY but exactly 2 PER REGION, so it is NOT a defect (the old global count
    // wrongly flagged it). A real non-manifold (a barrel-seam pinch, a crack) shows
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
    let mut nm: rustc_hash::FxHashMap<(usize, usize), u32> = rustc_hash::FxHashMap::default();
    for (&(_, a, b), &cnt) in &region_edge {
        if cnt != 2 {
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
    // every on-surface vertex it mislabels. Planes carry no geometry here
    // (the kind lacks the frame), so they cannot anchor the fit.
    let curved: Vec<Surface> = mesh.surfaces.iter().filter_map(Surface::curved).collect();
    let nearest_off = |q: V3| -> f64 {
        curved
            .iter()
            .map(|s| dist(q, s.closest(q).0))
            .fold(f64::INFINITY, f64::min)
    };
    // A face closing a filled contact wedge spans the wedge by design.
    let contact: rustc_hash::FxHashSet<usize> = mesh.contact_faces.iter().copied().collect();
    // Per curved face: its corners, longest edge, the largest distance of a
    // corner and of its centroid from the nearest surface.
    let offs: Vec<Option<([V3; 3], f64, f64, f64)>> = mesh
        .faces
        .par_iter()
        .enumerate()
        .map(|(fi, f)| {
            let kind = &mesh.surfaces[f.surface as usize];
            if matches!(kind, SurfaceKind::Plane) || curved.is_empty() || contact.contains(&fi) {
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
        if longest > 0.0 && vmax_off > 0.25 * longest {
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
        if longest > 0.0 && c_off > 0.25 * longest && vmax_off <= 0.25 * longest {
            n_bridge_faces += 1;
            defects.push(Defect {
                kind: DefectKind::BridgeFace,
                pos: centroid(&v),
                value: c_off,
            });
        }
    }

    MeshDiagnostics {
        n_tets: mesh.tets.len(),
        n_points: mesh.points.len(),
        n_faces: mesh.faces.len(),
        min_dihedral_deg: if min_dih.is_finite() { min_dih } else { 0.0 },
        mean_min_dihedral_deg: mean_dih,
        dihedral_histogram: hist,
        n_slivers,
        max_radius_edge: max_re,
        watertight: n_nonmanifold == 0,
        n_nonmanifold_edges: n_nonmanifold,
        n_straddlers,
        n_bridge_faces,
        max_surface_deviation: max_dev,
        region_volumes: region_vol.into_iter().collect(),
        defects,
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn dihedral_of_known_tets() {
        let r = 1.0 / 3f64.sqrt();
        let regular = [[r, r, r], [r, -r, -r], [-r, r, -r], [-r, -r, r]];
        assert!((tet_min_dihedral(regular) - (1.0f64 / 3.0).acos().to_degrees()).abs() < 1e-9);
        let corner = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let want = (1.0 / 3f64.sqrt()).acos().to_degrees();
        assert!((tet_min_dihedral(corner) - want).abs() < 1e-9);
        let flipped = [corner[1], corner[0], corner[2], corner[3]];
        assert!((tet_min_dihedral(flipped) - want).abs() < 1e-9);
        let flat = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        assert_eq!(tet_min_dihedral(flat), 0.0);
    }

    /// The face-normal form agrees with the angle between the half-planes
    /// at each edge, on random and on flat tets.
    #[test]
    fn dihedral_matches_the_edge_form() {
        fn by_edges(p: [V3; 4]) -> f64 {
            let cross = |a: V3, b: V3| -> V3 {
                [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ]
            };
            let dot = |a: V3, b: V3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let mut cos = -1.0f64;
            for (i, j, k, l) in [
                (0, 1, 2, 3),
                (0, 2, 1, 3),
                (0, 3, 1, 2),
                (1, 2, 0, 3),
                (1, 3, 0, 2),
                (2, 3, 0, 1),
            ] {
                let e = sub(p[j], p[i]);
                let (n1, n2) = (cross(e, sub(p[k], p[i])), cross(e, sub(p[l], p[i])));
                cos = cos.max(dot(n1, n2) / (dot(n1, n1) * dot(n2, n2)).sqrt());
            }
            cos.clamp(-1.0, 1.0).acos().to_degrees()
        }
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut r = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for i in 0..20_000 {
            let mut p: [V3; 4] = std::array::from_fn(|_| [r(), r(), r()]);
            if i % 4 == 0 {
                // Nearly flat: the fourth corner close to the plane of the others.
                let t = [r().abs(), r().abs()];
                p[3] = std::array::from_fn(|k| {
                    p[0][k] + t[0] * (p[1][k] - p[0][k]) + t[1] * (p[2][k] - p[0][k])
                });
                p[3][2] += 1e-3 * r();
            }
            let (a, b) = (tet_min_dihedral(p), by_edges(p));
            assert!((a - b).abs() < 1e-6 * (1.0 + b), "{p:?}: {a} vs {b}");
        }
    }

    use super::*;
    use rapidmesh_geom::{solid_box, Scene};

    #[test]
    fn box_is_clean_and_watertight() {
        let mut scene = Scene::new();
        scene.add_solid(solid_box([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]));
        let plc = scene.assemble();
        let m = crate::conform::mesh_plc_with(
            &plc,
            &crate::conform::MeshParams {
                maxh: 1.0,
                ..Default::default()
            },
        );
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
        assert!(d.min_dihedral_deg > 0.0, "well-defined dihedral");
        let vol: f64 = d.region_volumes.iter().map(|&(_, v)| v).sum();
        assert!((vol - 24.0).abs() < 1e-6, "box volume 24, got {vol}");
    }

    #[test]
    fn sphere_is_watertight_and_on_surface() {
        use rapidmesh_geom::sphere;
        let mut scene = Scene::new();
        scene.add_solid(sphere([0.0, 0.0, 0.0], 1.0, 24, 12));
        let plc = scene.assemble();
        let m = crate::conform::mesh_plc_with(
            &plc,
            &crate::conform::MeshParams {
                maxh: 0.4,
                tol_surf: 1e-2,
                ..Default::default()
            },
        );
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
