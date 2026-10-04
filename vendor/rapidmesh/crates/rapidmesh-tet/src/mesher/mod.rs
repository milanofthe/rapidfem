//! The bottom-up mesher: corners, then edges, then faces, then each region
//! on its own. Every mesh of a lower dimension is final and shared by the
//! higher ones on it, so neighbours conform by construction and no stage
//! sees more than its own piece: a face is meshed alone whatever lies close
//! to it, and a thin region costs only its surface.

pub mod blocks;

pub use crate::surface::{boundary, Boundary, BoundaryError};
pub use crate::volume::cdt::CdtError;

use crate::finish::Complex;
use crate::surface;
use crate::volume::{cdt, refine, region};

use crate::mesh::{CurveEdge, PointClass, SurfaceFace, SurfaceMesh, TetMesh};

use crate::params::MeshParams;
use crate::sizing::tree::DomainTree;
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
        at: [f64; 3],
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
            MeshError::Open { region, edges, at } => {
                write!(
                    f,
                    "region {region} is not closed ({edges} open edges, one at {at:?})"
                )
            }
            MeshError::Region { region, error } => write!(f, "region {region}: {error}"),
        }
    }
}

impl std::error::Error for MeshError {}

impl MeshError {
    /// Where it went wrong, where one point says it.
    pub fn at(&self) -> Option<[f64; 3]> {
        match self {
            MeshError::Boundary(BoundaryError::Diverged { near, .. }) => near.first().map(|n| n.at),
            MeshError::Open { at, .. } => Some(*at),
            MeshError::Region {
                error: CdtError::Stuck { at, .. },
                ..
            } => Some(*at),
            MeshError::Boundary(_) => None,
        }
    }

    /// What went wrong, where on `model` and what to do about it, in the
    /// words of someone who has to repair the geometry.
    pub fn explain(&self, model: &Model) -> String {
        let brep = &model.brep;
        let p = |x: [f64; 3]| format!("({:.6}, {:.6}, {:.6})", x[0], x[1], x[2]);
        let face = |f: u32| -> String {
            let kind = brep
                .faces
                .get(f as usize)
                .map_or("?", |x| surface::surface_kind(brep.surface(x.surface)));
            let at =
                face_at(model, f as usize).map_or(String::new(), |x| format!(" near {}", p(x)));
            format!("face {f} ({kind}){at}")
        };
        match self {
            MeshError::Boundary(BoundaryError::Diverged { left, near }) => {
                let mut out = format!(
                    "cannot mesh: {left} edges of the surface mesh cannot be made Delaunay; \
                     splitting them only makes more"
                );
                for (i, n) in near.iter().enumerate() {
                    let edge = if n.shortest < n.size {
                        format!(
                            ", whose shortest edge {:.3e} is below the mesh size {:.3e} there",
                            n.shortest, n.size
                        )
                    } else {
                        format!(", at a mesh size of {:.3e} there", n.size)
                    };
                    out += &format!(
                        "{} at {} on face {} ({}){edge}",
                        if i == 0 { ", worst" } else { "; also" },
                        p(n.at),
                        n.face,
                        n.kind,
                    );
                }
                out + ". The geometry there is finer than the mesh can follow (tiny \
                       edges, sliver faces, faces meeting at a grazing angle or nearly \
                       touching): repair it, or give the faces there a smaller size."
            }
            MeshError::Boundary(BoundaryError::Curved { face: f, why, .. }) => format!(
                "cannot mesh {}: {why}. The face is degenerate or folds over its \
                 surface: check it in the CAD.",
                face(*f)
            ),
            MeshError::Boundary(BoundaryError::Empty { face: f }) => format!(
                "cannot mesh {}: its mesh is empty. The face is degenerate (no area at \
                 the mesh size): remove it or merge it into its neighbour.",
                face(*f)
            ),
            MeshError::Boundary(BoundaryError::Broken {
                face: f,
                open,
                over,
                unused,
                ..
            }) => format!(
                "cannot mesh {}: its mesh has {open} open edges, {over} edges on more \
                 than two triangles and {unused} boundary segments left out. The face \
                 crosses itself or is degenerate: check it in the CAD.",
                face(*f)
            ),
            MeshError::Boundary(BoundaryError::Periodic { face: f }) => format!(
                "cannot mesh {}: it does not match its periodic partner. The two faces \
                 must be exact moved copies of each other.",
                face(*f)
            ),
            MeshError::Boundary(e @ BoundaryError::Refine(_)) => {
                format!("cannot mesh: {e} (an internal error, please report it with the input)")
            }
            MeshError::Open { region, edges, at } => format!(
                "cannot mesh: region {region} is not closed ({edges} open edges, one at {}). \
                 Its solids have gaps, or faces that overlap without meeting exactly: \
                 repair the input there.",
                p(*at)
            ),
            MeshError::Region {
                region,
                error: CdtError::Stuck { at, .. },
            } => format!(
                "cannot mesh: region {region} cannot be filled near {}: its surface mesh \
                 is not Delaunay there, which comes from geometry far below the mesh \
                 size. Repair it there, or give the faces there a smaller size.",
                p(*at)
            ),
        }
    }
}

/// A point of face `f` of `model`: the centroid of one of its facets.
fn face_at(model: &Model, f: usize) -> Option<[f64; 3]> {
    let face = model.brep.faces.get(f)?;
    let t = model.plc.triangles[*face.facets.first()? as usize];
    let v = t.map(|i| model.plc.vertices[i as usize]);
    Some(std::array::from_fn(|k| (v[0][k] + v[1][k] + v[2][k]) / 3.0))
}

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

/// The tet mesh of the model of `scene`: the boundary, then each region
/// filled by its constrained Delaunay tetrahedralization, refined and
/// finished; a large one in blocks.
pub fn mesh_scene(scene: &Scene, model: &Model, params: &MeshParams) -> Result<TetMesh, MeshError> {
    mesh_in_blocks(model, Some((scene, None)), params)
}

fn mesh_in_blocks(
    source: &Model,
    scene: Option<(&Scene, Option<f64>)>,
    params: &MeshParams,
) -> Result<TetMesh, MeshError> {
    // Cells across a region, where asked for, bound its size.
    let params = &params.with_thickness_caps(&source.plc);
    let t = rapidmesh_exact::clock::Instant::now();
    let domain = crate::sizing::build_sizing_domain(source, params);
    rapidmesh_exact::log::stage("mesher.sizing", t.elapsed().as_secs_f64());
    // A periodic face must stay whole: its partner is meshed as its image.
    let t = rapidmesh_exact::clock::Instant::now();
    let cut = scene
        .filter(|_| params.periodic.is_empty())
        .and_then(|(scene, block)| {
            let plan = blocks::Plan::new(source, &domain, block);
            if plan.is_empty() {
                return None;
            }
            rapidmesh_exact::log::stat("mesher.cells", plan.cells() as f64);
            blocks::Blocks::new(scene, source, &plan).map(|b| (plan, b))
        });
    rapidmesh_exact::log::stage("mesher.blocks", t.elapsed().as_secs_f64());
    rapidmesh_exact::log::stat(
        "mesher.blocks",
        cut.as_ref().map_or(1, |c| c.1.count()) as f64,
    );
    // Blocks are a matter of speed: where a block fails at a cut (one
    // through a fine feature), the model is meshed whole; a failure away
    // from the cuts is the geometry's, and said at once. A boundary that
    // diverges is the cuts' only where every face it misses most on is a
    // cut or a piece of a face one splits, near a cut.
    let at_cut = |e: &MeshError| {
        let Some((plan, blocks)) = cut.as_ref() else {
            return false;
        };
        let near = |p: [f64; 3]| plan.near(p, AT_CUT * domain.h_at(p));
        match e {
            MeshError::Boundary(BoundaryError::Diverged { near: faces, .. }) => {
                faces.iter().all(|n| near(n.at) && blocks.at_cut(n.face))
            }
            _ => e.at().is_none_or(near),
        }
    };
    match mesh_cut(source, cut.as_ref().map(|c| &c.1), &domain, params) {
        Err(e) if at_cut(&e) => {
            rapidmesh_exact::log::info(
                "mesher.blocks",
                format!("in blocks: {e}; meshed whole instead"),
            );
            mesh_cut(source, None, &domain, params)
        }
        x => x,
    }
}

/// A failure within this many sizes of a cut may be the cut's.
const AT_CUT: f64 = 4.0;

/// [`mesh_scene`] of `source`, of its blocks where `cut` is given.
fn mesh_cut(
    source: &Model,
    cut: Option<&blocks::Blocks>,
    domain: &DomainTree,
    params: &MeshParams,
) -> Result<TetMesh, MeshError> {
    let raw = raw(source, cut, domain, params)?;
    rapidmesh_exact::log::heap("finish");
    Ok(crate::finish::brep::finish_classified(source, params, raw))
}

/// The [`Complex`] mesh of `source`, of its blocks where `cut` is given.
fn raw(
    source: &Model,
    cut: Option<&blocks::Blocks>,
    domain: &DomainTree,
    params: &MeshParams,
) -> Result<Complex, MeshError> {
    let cut_params = cut.map(|c| c.params(params));
    let (model, params_on) = match (cut, &cut_params) {
        (Some(c), Some(p)) => (&c.model, p),
        _ => (source, params),
    };
    let (b, mut kept) = surface::boundary_keeping(model, domain, params_on)?;
    rapidmesh_exact::log::heap("boundary");
    let brep = &model.brep;
    // A region open anywhere has no tetrahedralization: said at once, not
    // after a search of the wrapping through all of it.
    if let Some((region, (edges, Some(at)))) = regions(brep)
        .into_par_iter()
        .map(|r| (r, b.open_edges(brep, r)))
        .find_any(|&(_, (n, _))| n > 0)
    {
        return Err(MeshError::Open { region, edges, at });
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
    rapidmesh_exact::log::stage("volume.cdt", t.elapsed().as_secs_f64());
    rapidmesh_exact::log::heap("cdt");
    let mut filled_ok: Vec<(u32, cdt::Filled)> = Vec::with_capacity(filled.len());
    for (r, ts) in filled {
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
        "volume.steiner_points",
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
    rapidmesh_exact::log::stage("volume.refine", t.elapsed().as_secs_f64());
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
        "volume.interior_points",
        (points.len() - b.points.len()) as f64,
    );
    rapidmesh_exact::log::heap("refine");
    let (mut point_class, curve_edges) = classes(brep, &b);
    point_class.resize(points.len(), PointClass::Interior);
    // The faces wound into their front region, with their B-rep face.
    let mut faces: Vec<crate::finish::Face> = brep
        .faces
        .iter()
        .zip(&b.faces)
        .enumerate()
        .flat_map(|(fi, (f, tris))| {
            tris.iter().map(move |t| crate::finish::Face {
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
    if let Some(cut) = cut {
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
    Ok(Complex {
        points,
        classes: point_class,
        tets,
        regions: tet_regions,
        faces,
        edges,
    })
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
    faces: &mut Vec<crate::finish::Face>,
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
/// meshed alone on the shared samples of its edges. A triangle budget
/// (`surf_target_count`, 0 = none) coarsens the sizes by one global factor
/// over a few remeshes until the count is at most a little over it.
pub fn surface_mesh(model: &Model, params: &MeshParams) -> Result<SurfaceMesh, BoundaryError> {
    use crate::sizing::{budget, Budget};
    let mesh = |p: &MeshParams| surface_once(model, p);
    budget(
        params,
        params.surf_target_count,
        Budget::Cap(2.0),
        mesh,
        |m| m.faces.len(),
    )
    .map(|(m, _)| m)
}

fn surface_once(model: &Model, params: &MeshParams) -> Result<SurfaceMesh, BoundaryError> {
    let domain = crate::sizing::build_sizing_domain(model, params);
    let b = boundary(model, &domain, params)?;
    let brep = &model.brep;
    // The faces meet only in the vertices and edges they share.
    let tris: Vec<[u32; 3]> = b.faces.iter().flatten().copied().collect();
    let improper = rapidmesh_csg::improper_pairs(&b.points, &tris);
    rapidmesh_exact::log::stat("mesher.improper_pairs", improper.len() as f64);
    if let Some(&[i, j]) = improper.first() {
        let show = |t: usize| tris[t].map(|v| (v, b.points[v as usize]));
        rapidmesh_exact::log::warn(
            "mesher",
            format!(
                "faces meet: {:?} and {:?}",
                show(i as usize),
                show(j as usize)
            ),
        );
    }
    let (point_class, curve_edges) = classes(brep, &b);
    // The samples of a smooth edge onto its exact curve: they were placed
    // on the polyline of its facets (the volume finish snaps them, a surface
    // mesh has no finish), so a circle stays round.
    let mut points = b.points;
    {
        use crate::finish::snap::Shape;
        let shape = crate::finish::brep::BrepShape::new(brep);
        for (p, &k) in points.iter_mut().zip(&point_class) {
            if matches!(k, PointClass::Edge(_)) && shape.smooth(k) {
                if let Some(q) = shape.project(k, *p) {
                    *p = q;
                }
            }
        }
    }
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
        points,
        faces,
        surfaces: model.plc.surfaces.clone(),
        surface_owners: model.plc.surface_owners.clone(),
        curve_edges,
        point_class,
    })
}
