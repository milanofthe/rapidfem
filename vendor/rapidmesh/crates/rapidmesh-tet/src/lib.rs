//! Tetrahedral meshing, bottom up (`bottomup`): the edges are sampled once, each face is
//! meshed alone on the samples of its edges, each region is filled by its constrained
//! Delaunay tetrahedralization and refined, and the boundary is snapped onto the analytic
//! carriers and the worst tets improved locally (`mesh3::improve`).
//!
//! The restricted-Delaunay refinement (`mesh3::refine`, `mesh_budgeted`) is still the
//! fallback where the bottom-up mesher fails and the surface mesher under a triangle budget
//! (#237, #238).
//!
//! Public surface: the entry points re-exported below, the 2D core (`surf2d`), adaptive
//! marking (`adapt`) and `diagnostics`. Topology and quality accessors live in
//! `rapidmesh_topo`.
pub mod adapt;
pub mod bottomup;
pub mod diagnostics;
pub mod fidelity;
pub mod mesh3;
pub mod surf2d;

pub(crate) mod brep_mesh;
pub(crate) mod conform;
pub(crate) mod constants;
pub(crate) mod curve;
pub(crate) mod cvt;
pub(crate) mod domain;
mod geomutil;
pub mod tri;

pub use adapt::{dorfler_mark, Dorfler};
pub use conform::{
    log_metrics, log_surface_metrics, mesh_model, mesh_plc, mesh_plc_with, quality_stats,
    CurveEdge, MeshParams, PointClass, QualityStats, SurfaceFace, SurfaceMesh, TetMesh,
};
pub use cvt::{budgeted, mesh_budgeted};
pub use tri::{tetrahedralize, Triangulation};
