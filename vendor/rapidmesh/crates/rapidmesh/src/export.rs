//! Viewer and comparison exports.
//!
//! [`write_mesh_json`] writes one mesh in the viewer schema
//! (`<mesher>_<name>.json`); the gmsh/tetgen reference scripts write the same
//! schema under their own prefix. [`write_manifest`] regenerates the viewer
//! manifest by scanning for rapidmesh outputs, so the scene exporter and the
//! benchmark harness can both contribute geometries in any order.
//! [`write_plc_json`] dumps the assembled PLC with region seeds and sizing so
//! the tetgen reference meshes the IDENTICAL input.

use rapidmesh_tet::{MeshParams, QualityStats, TetMesh};
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
struct FaceJson {
    tri: [usize; 3],
    tag: u32,
    regions: [u32; 2],
}

#[derive(Serialize)]
struct StatsJson {
    n_points: usize,
    n_tets: usize,
    min_dihedral_deg: f64,
    max_radius_edge: f64,
    max_edge: f64,
    millis: u128,
}

#[derive(Serialize)]
struct MeshJson {
    name: String,
    mesher: String,
    points: Vec<[f64; 3]>,
    tets: Vec<[usize; 4]>,
    tet_regions: Vec<u32>,
    faces: Vec<FaceJson>,
    stats: StatsJson,
}

/// Writes `<out_dir>/<mesher>_<name>.json` in the viewer schema.
pub fn write_mesh_json(
    out_dir: &Path,
    mesher: &str,
    name: &str,
    mesh: &TetMesh,
    q: &QualityStats,
    millis: u128,
) {
    let json = MeshJson {
        name: name.to_string(),
        mesher: mesher.to_string(),
        points: mesh.points.clone(),
        tets: mesh.tets.clone(),
        tet_regions: mesh.tet_regions.iter().map(|r| r.0).collect(),
        faces: mesh
            .faces
            .iter()
            .map(|f| FaceJson {
                tri: f.tri,
                tag: f.face_tag.0,
                regions: [f.regions[0].0, f.regions[1].0],
            })
            .collect(),
        stats: StatsJson {
            n_points: mesh.points.len(),
            n_tets: mesh.tets.len(),
            min_dihedral_deg: q.min_dihedral_deg,
            max_radius_edge: q.max_radius_edge,
            max_edge: q.max_edge,
            millis,
        },
    };
    let path = out_dir.join(format!("{mesher}_{name}.json"));
    std::fs::write(&path, serde_json::to_string(&json).expect("serialize")).expect("write");
    println!("  -> {}", path.display());
}

/// Regenerates the viewer manifest from the `rapidmesh_*.json` files present
/// in `out_dir`. Known comparison scenes come first in canonical order,
/// everything else (imported surface models) alphabetically after. Returns
/// the names written.
pub fn write_manifest(out_dir: &Path) -> std::io::Result<Vec<String>> {
    let canonical = [
        "em_scene",
        "via",
        "microstrip",
        "sphere",
        "l_prism",
        "density_transition",
    ];
    let mut names: Vec<String> = std::fs::read_dir(out_dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let f = e.file_name().into_string().ok()?;
            let stem = f.strip_prefix("rapidmesh_")?.strip_suffix(".json")?;
            Some(stem.to_string())
        })
        .collect();
    names.sort_by_key(|n| {
        let rank = canonical
            .iter()
            .position(|&c| c == n)
            .unwrap_or(canonical.len());
        (rank, n.clone())
    });
    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_string(&names).expect("serialize"),
    )?;
    Ok(names)
}

#[derive(Serialize)]
struct PlcRegionJson {
    tag: u32,
    /// A point strictly inside the region (barycenter of its largest tet).
    seed: [f64; 3],
    maxh: f64,
}

#[derive(Serialize)]
struct PlcJson {
    name: String,
    vertices: Vec<[f64; 3]>,
    triangles: Vec<[u32; 3]>,
    maxh: f64,
    radius_edge_bound: f64,
    regions: Vec<PlcRegionJson>,
}

/// Writes `<plc_dir>/<name>.json`: the assembled PLC plus per-region seed
/// points and sizing, derived from an already-computed rapidmesh mesh of the
/// same PLC. The tetgen reference script consumes this.
pub fn write_plc_json(
    plc_dir: &Path,
    name: &str,
    plc: &rapidmesh_geom::TaggedPlc,
    params: &MeshParams,
    mesh: &TetMesh,
) {
    // Per region: barycenter of the region's largest tet as inner seed.
    let mut best: std::collections::BTreeMap<u32, (f64, [f64; 3])> = Default::default();
    for (t, r) in mesh.tets.iter().zip(&mesh.tet_regions) {
        let p: [[f64; 3]; 4] = std::array::from_fn(|i| mesh.points[t[i]]);
        let u: [f64; 3] = std::array::from_fn(|k| p[1][k] - p[0][k]);
        let v: [f64; 3] = std::array::from_fn(|k| p[2][k] - p[0][k]);
        let w: [f64; 3] = std::array::from_fn(|k| p[3][k] - p[0][k]);
        let vol = (u[0] * (v[1] * w[2] - v[2] * w[1])
            + u[1] * (v[2] * w[0] - v[0] * w[2])
            + u[2] * (v[0] * w[1] - v[1] * w[0]))
            .abs();
        let bary: [f64; 3] = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k] + p[3][k]) / 4.0);
        let e = best.entry(r.0).or_insert((-1.0, bary));
        if vol > e.0 {
            *e = (vol, bary);
        }
    }
    let json = PlcJson {
        name: name.to_string(),
        vertices: plc.vertices.clone(),
        triangles: plc.triangles.clone(),
        maxh: params.maxh,
        radius_edge_bound: params.radius_edge_bound,
        regions: best
            .iter()
            .map(|(&tag, &(_, seed))| PlcRegionJson {
                tag,
                seed,
                maxh: params
                    .region_maxh
                    .iter()
                    .find(|(r, _)| *r == tag)
                    .map(|&(_, h)| h)
                    .unwrap_or(params.maxh),
            })
            .collect(),
    };
    let path = plc_dir.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_string(&json).expect("serialize")).expect("write");
}
