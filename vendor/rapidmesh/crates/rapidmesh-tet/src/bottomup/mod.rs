//! The bottom-up mesher: corners, then edges, then faces, then each region
//! on its own. Every mesh of a lower dimension is final and shared by the
//! higher ones on it, so neighbours conform by construction and no stage
//! sees more than its own piece: a face is meshed alone whatever lies close
//! to it, and a thin region costs only its surface.

pub mod atlas;
pub mod blocks;
pub mod cdt;
pub mod chart;
pub mod contact;
pub mod delaunay;
pub mod periodic;
pub mod predicates;
pub mod refine;
pub mod region;
pub(crate) mod remesh;
pub mod stereo;
pub mod surface;
pub(crate) mod topology;
pub mod unroll;

pub use cdt::CdtError;
pub use surface::{boundary, Boundary, BoundaryError};

use crate::conform::{CurveEdge, MeshParams, PointClass, SurfaceFace, SurfaceMesh, TetMesh};
use rapidmesh_brep::{Brep, Model};
use rapidmesh_geom::Scene;
use rayon::prelude::*;

/// Why the bottom-up mesher gave no mesh.
#[derive(Debug, Clone, PartialEq)]
pub enum MeshError {
    Boundary(BoundaryError),
    /// A region whose boundary mesh is not closed (edges without a partner
    /// in the opposite direction): no tetrahedralization can fill it.
    Open {
        region: u32,
        edges: usize,
    },
    /// The region with its constrained tetrahedralization left undone.
    Region {
        region: u32,
        error: CdtError,
    },
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::Boundary(e) => write!(f, "{e}"),
            MeshError::Open { region, edges } => {
                write!(f, "region {region} is not closed ({edges} open edges)")
            }
            MeshError::Region { region, error } => write!(f, "region {region}: {error}"),
        }
    }
}

impl std::error::Error for MeshError {}

impl From<BoundaryError> for MeshError {
    fn from(e: BoundaryError) -> MeshError {
        MeshError::Boundary(e)
    }
}

/// The first id of the points a region's refinement adds, before they are
/// numbered after every boundary point.
const NEW: u32 = 1 << 31;

/// The regions of a model (its background, region 0, left out).
fn regions(brep: &Brep) -> Vec<u32> {
    let mut rs: Vec<u32> = brep
        .faces
        .iter()
        .flat_map(|f| f.regions.map(|r| r.0))
        .filter(|&r| r != 0)
        .collect();
    rs.sort_unstable();
    rs.dedup();
    rs
}

/// The faces of region `r`, each turned so the region lies on its positive
/// side; a sheet inside the region in both orientations.
pub fn region_faces(brep: &Brep, b: &Boundary, r: u32) -> Vec<[u32; 3]> {
    region_faces_beyond(brep, b, r)
        .into_iter()
        .map(|x| x.0)
        .collect()
}

/// [`region_faces`], each with the region on its other side (`r` for a
/// sheet inside it).
pub fn region_faces_beyond(brep: &Brep, b: &Boundary, r: u32) -> Vec<([u32; 3], u32)> {
    let mut out = Vec::new();
    let mut beyond: Vec<u32> = Vec::new();
    let mut from: Vec<usize> = Vec::new();
    for (fi, (f, tris)) in brep.faces.iter().zip(&b.faces).enumerate() {
        let [front, back] = f.regions.map(|x| x.0);
        for t in tris {
            // A triangle's normal points into the front region, the side a
            // positive orientation calls negative.
            if front == r {
                out.push([t[0], t[2], t[1]]);
                beyond.push(back);
                from.push(fi);
            }
            if back == r {
                out.push(*t);
                beyond.push(front);
                from.push(fi);
            }
        }
    }
    // One triangle on two faces, turned both ways into the region, bounds
    // nothing of it (where two faces coincide, the region between them has
    // no thickness): both go, and the regions beyond meet on it. A sheet's
    // two sides are one face and stay.
    let key = |t: [u32; 3]| {
        let mut k = t;
        k.sort_unstable();
        k
    };
    let mut seen: rustc_hash::FxHashMap<[u32; 3], Vec<usize>> = rustc_hash::FxHashMap::default();
    for (i, &t) in out.iter().enumerate() {
        seen.entry(key(t)).or_default().push(i);
    }
    let parity = |t: [u32; 3]| {
        // Whether `t` is an even rotation of its sorted corners.
        let k = key(t);
        let at = |v: u32| k.iter().position(|&x| x == v).unwrap_or(0);
        let (a, b, c) = (at(t[0]), at(t[1]), at(t[2]));
        (a + 1) % 3 == b && (b + 1) % 3 == c
    };
    let mut drop = vec![false; out.len()];
    for ids in seen.values() {
        if ids.len() == 2
            && from[ids[0]] != from[ids[1]]
            && parity(out[ids[0]]) != parity(out[ids[1]])
        {
            drop[ids[0]] = true;
            drop[ids[1]] = true;
        }
    }
    out.into_iter()
        .zip(beyond)
        .zip(drop)
        .filter(|(_, d)| !d)
        .map(|(x, _)| x)
        .collect()
}

/// What each boundary point lies on, and the mesh edges along B-rep edges.
fn classes(brep: &Brep, b: &Boundary) -> (Vec<PointClass>, Vec<CurveEdge>) {
    let mut point_class = vec![PointClass::Interior; b.points.len()];
    for (fi, tris) in b.faces.iter().enumerate() {
        for t in tris {
            for &v in t {
                point_class[v as usize] = PointClass::Face(fi as u32);
            }
        }
    }
    let mut curve_edges = Vec::new();
    for (ei, pts) in b.edges.iter().enumerate() {
        for &v in pts {
            point_class[v as usize] = PointClass::Edge(ei as u32);
        }
        for w in pts.windows(2) {
            curve_edges.push(CurveEdge {
                v: [w[0] as usize, w[1] as usize],
                edge: ei as u32,
            });
        }
    }
    for v in 0..brep.vertices.len() {
        point_class[v] = PointClass::Vertex(v as u32);
    }
    (point_class, curve_edges)
}

/// The tet mesh of `model` by the bottom-up stages: the boundary, then each
/// region filled by its constrained Delaunay tetrahedralization (no
/// interior points yet).
pub fn mesh(model: &Model, params: &MeshParams) -> Result<TetMesh, MeshError> {
    mesh_in_blocks(model, None, params)
}

/// [`mesh`] of the model of `scene`, a large one in blocks (see
/// [`blocks`]).
pub fn mesh_scene(scene: &Scene, model: &Model, params: &MeshParams) -> Result<TetMesh, MeshError> {
    // RAPIDMESH_BLOCK_TETS plans smaller blocks, to try them on small
    // models; RAPIDMESH_NO_BLOCKS meshes any model whole, for comparison.
    if std::env::var_os("RAPIDMESH_NO_BLOCKS").is_some() {
        return mesh(model, params);
    }
    let block = std::env::var("RAPIDMESH_BLOCK_TETS")
        .ok()
        .and_then(|x| x.parse().ok());
    mesh_in_blocks(model, Some((scene, block)), params)
}

fn mesh_in_blocks(
    source: &Model,
    scene: Option<(&Scene, Option<f64>)>,
    params: &MeshParams,
) -> Result<TetMesh, MeshError> {
    // Cells across a region, where asked for, bound its size.
    let params = &params.with_thickness_caps(&source.plc);
    let t = rapidmesh_exact::clock::Instant::now();
    let domain = crate::cvt::build_sizing_domain(source, params);
    rapidmesh_exact::log::stage("bottomup.sizing", t.elapsed().as_secs_f64());
    // A periodic face must stay whole: its partner is meshed as its image.
    let t = rapidmesh_exact::clock::Instant::now();
    let cut = scene
        .filter(|_| params.periodic.is_empty())
        .and_then(|(scene, block)| {
            let plan = blocks::Plan::new(source, &domain, block);
            if plan.is_empty() {
                return None;
            }
            rapidmesh_exact::log::stat("bottomup.cells", plan.cells() as f64);
            blocks::Blocks::new(scene, source, &plan)
        });
    rapidmesh_exact::log::stage("bottomup.blocks", t.elapsed().as_secs_f64());
    rapidmesh_exact::log::stat(
        "bottomup.blocks",
        cut.as_ref().map_or(1, |c| c.count()) as f64,
    );
    let cut_params = cut.as_ref().map(|c| c.params(params));
    let (model, params_on) = match (&cut, &cut_params) {
        (Some(c), Some(p)) => (&c.model, p),
        _ => (source, params),
    };
    let (b, mut kept) = surface::boundary_keeping(model, &domain, params_on)?;
    rapidmesh_exact::log::heap("boundary");
    let brep = &model.brep;
    // A region open anywhere has no tetrahedralization: said at once, not
    // after a search of the wrapping through all of it.
    if let Some((region, edges)) = regions(brep)
        .into_par_iter()
        .map(|r| (r, b.open_edges(brep, r)))
        .find_any(|&(_, n)| n > 0)
    {
        return Err(MeshError::Open { region, edges });
    }
    let t = rapidmesh_exact::clock::Instant::now();
    let jobs: Vec<(u32, Option<region::Kept>)> = regions(brep)
        .into_iter()
        .map(|r| (r, kept.remove(&r)))
        .collect();
    let filled: Vec<(u32, Result<cdt::Filled, CdtError>)> = jobs
        .into_par_iter()
        .map(|(r, dt)| {
            (
                r,
                cdt::tetrahedralize_on(&b.points, &region_faces(brep, &b, r), dt),
            )
        })
        .collect();
    rapidmesh_exact::log::stage("bottomup.cdt", t.elapsed().as_secs_f64());
    rapidmesh_exact::log::heap("cdt");
    let mut filled_ok: Vec<(u32, cdt::Filled)> = Vec::with_capacity(filled.len());
    for (r, ts) in filled {
        // A region left without its tetrahedralization is written to the
        // file named by RAPIDMESH_DUMP_REGION (points, then its faces turned
        // into it), for `cdt::tests::region_from_file` to trace.
        if ts.is_err() {
            if let Ok(path) = std::env::var("RAPIDMESH_DUMP_REGION") {
                let faces = region_faces(brep, &b, r);
                let mut txt = format!("{} {}\n", b.points.len(), faces.len());
                for p in &b.points {
                    txt += &format!("{:?} {:?} {:?}\n", p[0], p[1], p[2]);
                }
                for t in &faces {
                    txt += &format!("{} {} {}\n", t[0], t[1], t[2]);
                }
                std::fs::write(path, txt).ok();
            }
        }
        let ts = ts.map_err(|error| MeshError::Region { region: r, error })?;
        filled_ok.push((r, ts));
    }
    // The points the wrapping added, numbered after the boundary's.
    let mut points = b.points.clone();
    let filled_ok: Vec<(u32, Vec<[u32; 4]>)> = filled_ok
        .into_iter()
        .map(|(r, f)| {
            let base = points.len() as u32;
            points.extend_from_slice(&f.steiner);
            let tets = f
                .tets
                .into_iter()
                .map(|t| {
                    t.map(|v| {
                        if v >= cdt::STEINER {
                            base + (v - cdt::STEINER)
                        } else {
                            v
                        }
                    })
                })
                .collect();
            (r, tets)
        })
        .collect();
    rapidmesh_exact::log::stat(
        "bottomup.steiner_points",
        (points.len() - b.points.len()) as f64,
    );
    // ---- interior points: each region refined to the size on its own
    let t = rapidmesh_exact::clock::Instant::now();
    let cap = params.vol_cap();
    let size = |p: [f64; 3]| domain.h_at(p).min(cap);
    let budget = params.max_points.max(1);
    let refined: Vec<(u32, refine::Refined)> = filled_ok
        .into_par_iter()
        .map(|(r, ts)| {
            let (faces, beyond): (Vec<[u32; 3]>, Vec<u32>) =
                region_faces_beyond(brep, &b, r).into_iter().unzip();
            (
                r,
                refine::refine(&points, &ts, &faces, &beyond, &size, NEW, budget),
            )
        })
        .collect();
    rapidmesh_exact::log::stage("bottomup.refine", t.elapsed().as_secs_f64());
    let mut tets: Vec<[u32; 4]> = Vec::new();
    let mut tet_regions: Vec<u32> = Vec::new();
    for (r, rf) in refined {
        let base = points.len() as u32;
        points.extend_from_slice(&rf.points);
        tet_regions.extend(std::iter::repeat_n(r, rf.tets.len()));
        tets.extend(
            rf.tets
                .into_iter()
                .map(|t| t.map(|v| if v >= NEW { base + (v - NEW) } else { v })),
        );
    }
    rapidmesh_exact::log::stat(
        "bottomup.interior_points",
        (points.len() - b.points.len()) as f64,
    );
    rapidmesh_exact::log::heap("refine");
    let (mut point_class, curve_edges) = classes(brep, &b);
    point_class.resize(points.len(), PointClass::Interior);
    // The faces wound into their front region, with their B-rep face.
    let mut faces: Vec<crate::mesh3::Face> = brep
        .faces
        .iter()
        .zip(&b.faces)
        .enumerate()
        .flat_map(|(fi, (f, tris))| {
            tris.iter().map(move |t| crate::mesh3::Face {
                tri: *t,
                regions: f.regions.map(|r| r.0),
                patch: fi as u32,
            })
        })
        .collect();
    let mut edges: Vec<([u32; 2], u32)> = curve_edges
        .iter()
        .map(|e| ([e.v[0] as u32, e.v[1] as u32], e.edge))
        .collect();
    if let Some(cut) = &cut {
        from_blocks(
            cut,
            source,
            &mut points,
            &mut point_class,
            &mut tets,
            &mut tet_regions,
            &mut faces,
            &mut edges,
        );
    }
    rapidmesh_exact::log::heap("finish");
    Ok(crate::mesh3::brep::finish_classified(
        source,
        params,
        &domain,
        points,
        &point_class,
        tets,
        tet_regions,
        faces,
        &edges,
    ))
}

/// Names a mesh of the cut model of `cut` in the terms of `source`: each
/// block's tets by its region, the pieces of faces and edges by theirs,
/// points by what they lie on there (a point on a cut alone lies inside a
/// region), the source corners first and in order; the cut faces go.
#[allow(clippy::too_many_arguments)]
fn from_blocks(
    cut: &blocks::Blocks,
    source: &Model,
    points: &mut Vec<[f64; 3]>,
    point_class: &mut Vec<PointClass>,
    tets: &mut [[u32; 4]],
    tet_regions: &mut [u32],
    faces: &mut Vec<crate::mesh3::Face>,
    edges: &mut Vec<([u32; 2], u32)>,
) {
    let corners = source.brep.vertices.len() as u32;
    let mut next = corners;
    let order: Vec<u32> = (0..points.len() as u32)
        .map(|v| {
            let at = (v < cut.model.brep.vertices.len() as u32)
                .then(|| cut.corner(v))
                .flatten();
            at.unwrap_or_else(|| {
                next += 1;
                next - 1
            })
        })
        .collect();
    let mut moved = vec![[0.0; 3]; points.len()];
    let mut classes = vec![PointClass::Interior; points.len()];
    for (v, &to) in order.iter().enumerate() {
        moved[to as usize] = points[v];
        classes[to as usize] = cut.class(point_class[v]);
    }
    *points = moved;
    *point_class = classes;
    for t in tets.iter_mut() {
        *t = t.map(|v| order[v as usize]);
    }
    for r in tet_regions.iter_mut() {
        *r = cut.region(*r);
    }
    faces.retain_mut(|f| match cut.face(f.patch) {
        Some(s) => {
            f.patch = s;
            f.tri = f.tri.map(|v| order[v as usize]);
            f.regions = f.regions.map(|r| cut.region(r));
            true
        }
        None => false,
    });
    edges.retain_mut(|(e, id)| match cut.edge(*id) {
        Some(s) => {
            *id = s;
            *e = e.map(|v| order[v as usize]);
            true
        }
        None => false,
    });
}

/// The surface mesh of `model` by the bottom-up stages: each B-rep face
/// meshed alone on the shared samples of its edges.
pub fn surface_mesh(model: &Model, params: &MeshParams) -> Result<SurfaceMesh, BoundaryError> {
    let domain = crate::cvt::build_sizing_domain(model, params);
    let b = boundary(model, &domain, params)?;
    let brep = &model.brep;
    // The faces meet only in the vertices and edges they share.
    let tris: Vec<[u32; 3]> = b.faces.iter().flatten().copied().collect();
    let improper = rapidmesh_csg::improper_pairs(&b.points, &tris);
    rapidmesh_exact::log::stat("bottomup.improper_pairs", improper.len() as f64);
    if let Some(&[i, j]) = improper.first() {
        let show = |t: usize| tris[t].map(|v| (v, b.points[v as usize]));
        rapidmesh_exact::log::warn(
            "bottomup",
            format!(
                "faces meet: {:?} and {:?}",
                show(i as usize),
                show(j as usize)
            ),
        );
    }
    let (point_class, curve_edges) = classes(brep, &b);
    let faces = brep
        .faces
        .iter()
        .zip(&b.faces)
        .enumerate()
        .flat_map(|(fi, (f, tris))| {
            tris.iter().map(move |t| SurfaceFace {
                tri: t.map(|v| v as usize),
                face_tag: f.face_tag,
                regions: f.regions,
                patch: fi as u32,
                surface: f.plc_surface,
            })
        })
        .collect();
    Ok(SurfaceMesh {
        points: b.points,
        faces,
        surfaces: model.plc.surfaces.clone(),
        surface_owners: model.plc.surface_owners.clone(),
        curve_edges,
        point_class,
    })
}
