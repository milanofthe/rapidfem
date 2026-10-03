//! The quality of a finished mesh: one pass over its tets
//! ([`quality_stats`]) and the headline metrics in the log.

use crate::mesh::{SurfaceMesh, TetMesh};
/// Quality summary of a tet mesh, with where the worst element is, where
/// the slivers are and a per-region breakdown.
#[derive(Debug, Clone)]
pub struct QualityStats {
    /// Number of tets.
    pub n_tets: usize,
    /// Smallest dihedral angle in degrees (sliver indicator; the load-bearing
    /// metric for Nedelec conditioning).
    pub min_dihedral_deg: f64,
    /// Mean of the per-tet smallest dihedral angle (degrees).
    pub mean_min_dihedral_deg: f64,
    /// Count of per-tet smallest dihedral angles in each 10-degree bin
    /// `[0,10), ..., [170,180)`.
    pub dihedral_histogram: [usize; 18],
    /// The slivers, in order: tets with a smallest dihedral below
    /// [`crate::SLIVER_DEG`].
    pub slivers: Vec<usize>,
    /// Largest circumradius / shortest-edge ratio.
    pub max_radius_edge: f64,
    /// Longest edge in the mesh.
    pub max_edge: f64,
    /// Index of the tet holding the smallest dihedral angle (`usize::MAX` for
    /// an empty mesh).
    pub worst_tet: usize,
    /// Centroid of the worst tet: where the worst sliver sits.
    pub worst_location: [f64; 3],
    /// Region tag of the worst tet.
    pub worst_region: u32,
    /// Per region with tets, in ascending tag order.
    pub per_region: Vec<RegionQuality>,
}

/// The quality of one region's tets.
#[derive(Debug, Clone, Copy)]
pub struct RegionQuality {
    pub region: u32,
    /// Smallest dihedral angle (degrees).
    pub min_dihedral_deg: f64,
    pub n_tets: usize,
    /// Total tet volume.
    pub volume: f64,
}

use crate::simplex::{radius_edge, tet_min_dihedral, tet_volume};
use rapidmesh_geom::vec3::dist2;

/// The quality statistics of a mesh, in one parallel pass over its tets.
pub fn quality_stats(mesh: &TetMesh) -> QualityStats {
    use rayon::prelude::*;
    /// The statistics of a block of tets.
    #[derive(Clone)]
    struct Part {
        min_dihedral: f64,
        worst_tet: usize,
        sum_dihedral: f64,
        histogram: [usize; 18],
        slivers: Vec<usize>,
        max_re: f64,
        max_edge2: f64,
        /// Per region: smallest dihedral, tets, volume.
        per_region: Vec<(f64, usize, f64)>,
    }
    let nreg = mesh
        .tet_regions
        .iter()
        .map(|r| r.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let empty = Part {
        min_dihedral: f64::MAX,
        worst_tet: usize::MAX,
        sum_dihedral: 0.0,
        histogram: [0; 18],
        slivers: Vec::new(),
        max_re: 0.0,
        max_edge2: 0.0,
        per_region: vec![(f64::MAX, 0, 0.0); nreg],
    };
    const BLOCK: usize = 1 << 14;
    let parts: Vec<Part> = mesh
        .tets
        .par_chunks(BLOCK)
        .enumerate()
        .map(|(bi, tets)| {
            let mut q = empty.clone();
            for (k, t) in tets.iter().enumerate() {
                let ti = bi * BLOCK + k;
                let p: [[f64; 3]; 4] = std::array::from_fn(|k| mesh.points[t[k]]);
                for i in 0..4 {
                    for j in i + 1..4 {
                        q.max_edge2 = q.max_edge2.max(dist2(p[i], p[j]));
                    }
                }
                if let Some(re) = radius_edge(p) {
                    q.max_re = q.max_re.max(re);
                }
                let md = tet_min_dihedral(p);
                q.sum_dihedral += md;
                q.histogram[((md / 10.0).floor() as usize).min(17)] += 1;
                if md < crate::SLIVER_DEG {
                    q.slivers.push(ti);
                }
                if md < q.min_dihedral {
                    q.min_dihedral = md;
                    q.worst_tet = ti;
                }
                let e = &mut q.per_region[mesh.tet_regions[ti].0 as usize];
                e.0 = e.0.min(md);
                e.1 += 1;
                e.2 += tet_volume(p);
            }
            q
        })
        .collect();
    // Blocks in order: the first of equal worst tets wins, as serially.
    let mut q = empty;
    for b in parts {
        if b.min_dihedral < q.min_dihedral {
            q.min_dihedral = b.min_dihedral;
            q.worst_tet = b.worst_tet;
        }
        q.sum_dihedral += b.sum_dihedral;
        for (h, x) in q.histogram.iter_mut().zip(b.histogram) {
            *h += x;
        }
        q.slivers.extend(b.slivers);
        q.max_re = q.max_re.max(b.max_re);
        q.max_edge2 = q.max_edge2.max(b.max_edge2);
        for (e, x) in q.per_region.iter_mut().zip(&b.per_region) {
            e.0 = e.0.min(x.0);
            e.1 += x.1;
            e.2 += x.2;
        }
    }
    let worst = (q.worst_tet != usize::MAX).then_some(q.worst_tet);
    QualityStats {
        n_tets: mesh.tets.len(),
        min_dihedral_deg: q.min_dihedral,
        mean_min_dihedral_deg: if mesh.tets.is_empty() {
            0.0
        } else {
            q.sum_dihedral / mesh.tets.len() as f64
        },
        dihedral_histogram: q.histogram,
        slivers: q.slivers,
        max_radius_edge: q.max_re,
        max_edge: q.max_edge2.sqrt(),
        worst_tet: q.worst_tet,
        worst_location: worst.map_or([0.0; 3], |w| {
            let t = mesh.tets[w];
            std::array::from_fn(|k| (0..4).map(|c| mesh.points[t[c]][k]).sum::<f64>() / 4.0)
        }),
        worst_region: worst.map_or(0, |w| mesh.tet_regions[w].0),
        per_region: q
            .per_region
            .into_iter()
            .enumerate()
            .filter(|(_, (_, n, _))| *n > 0)
            .map(|(r, (m, n, v))| RegionQuality {
                region: r as u32,
                min_dihedral_deg: m,
                n_tets: n,
                volume: v,
            })
            .collect(),
    }
}

/// Emits the headline VOLUME-mesh metrics through [`rapidmesh_exact::log`]: the
/// element + vertex counts, the quality summary, the sliver count (a `warn` when
/// any sliver survives), and -- for multi-region meshes -- a per-region quality
/// breakdown. Recorded both live (when the log level allows) and into
/// `mesh.stats`, so the important numbers are visible without re-deriving them.
pub fn log_metrics(q: &QualityStats, n_points: usize) {
    use rapidmesh_exact::log;
    log::stat("mesh.tets", q.n_tets as f64);
    log::stat("mesh.points", n_points as f64);
    log::stat("mesh.min_dihedral_deg", q.min_dihedral_deg);
    log::stat("mesh.max_radius_edge", q.max_radius_edge);
    log::stat("mesh.longest_edge", q.max_edge);
    let n_slivers = q.slivers.len();
    log::stat("mesh.slivers", n_slivers as f64);
    log::info("metrics", format!("tets {}  points {}", q.n_tets, n_points));
    log::info(
        "metrics",
        format!(
            "min-dihedral {:.1} deg   max-radius-edge {:.2}   longest-edge {:.4}",
            q.min_dihedral_deg, q.max_radius_edge, q.max_edge
        ),
    );
    let pct = if q.n_tets > 0 {
        100.0 * n_slivers as f64 / q.n_tets as f64
    } else {
        0.0
    };
    let lvl = if n_slivers > 0 {
        log::Level::Warn
    } else {
        log::Level::Info
    };
    log::event(
        lvl,
        "metrics",
        format!(
            "slivers {} / {} ({pct:.2}%, below {:.0} deg)",
            n_slivers,
            q.n_tets,
            crate::SLIVER_DEG
        ),
    );
    if n_slivers > 0 {
        log::warn(
            "metrics",
            format!(
                "worst {:.1} deg in region {} near ({:.4}, {:.4}, {:.4})",
                q.min_dihedral_deg,
                q.worst_region,
                q.worst_location[0],
                q.worst_location[1],
                q.worst_location[2]
            ),
        );
    }
    if q.per_region.len() > 1 {
        let parts: Vec<String> = q
            .per_region
            .iter()
            .map(|r| format!("r{}:{:.1}deg/{}", r.region, r.min_dihedral_deg, r.n_tets))
            .collect();
        log::info("metrics", format!("regions  {}", parts.join("  ")));
    }
}

/// Emits the headline SURFACE-mesh metrics through [`rapidmesh_exact::log`]:
/// triangle + vertex counts and the minimum interior angle, with a count of thin
/// (`< 15 deg`) triangles (a `warn` when any survive).
pub fn log_surface_metrics(mesh: &SurfaceMesh) {
    use rapidmesh_exact::log;
    let angles: Vec<f64> = mesh
        .faces
        .iter()
        .map(|f| crate::simplex::tri_min_angle(f.tri.map(|v| mesh.points[v])))
        .collect();
    let min_ang = angles.iter().copied().fold(f64::MAX, f64::min);
    let min_ang = if min_ang.is_finite() { min_ang } else { 0.0 };
    let n_bad = angles.iter().filter(|&&a| a < 15.0).count();
    log::stat("surface.faces", mesh.faces.len() as f64);
    log::stat("surface.points", mesh.points.len() as f64);
    log::stat("surface.min_angle_deg", min_ang);
    log::stat("surface.thin", n_bad as f64);
    log::info(
        "metrics",
        format!("faces {}  points {}", mesh.faces.len(), mesh.points.len()),
    );
    let lvl = if n_bad > 0 {
        log::Level::Warn
    } else {
        log::Level::Info
    };
    log::event(
        lvl,
        "metrics",
        format!("min-angle {min_ang:.1} deg   {n_bad} below 15 deg"),
    );
}
