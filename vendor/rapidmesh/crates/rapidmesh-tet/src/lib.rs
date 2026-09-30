//! Tetrahedral meshing: restricted-Delaunay refinement against an exact
//! discrete model of the B-rep (`mesh3`), behind every entry point:
//! 1. Protect the corners and feature curves with weighted balls.
//! 2. Refine restricted facets (surface balls on the PLC) and cells
//!    (circumcenters under a size and radius-edge bound); label cells by
//!    region.
//! 3. Snap boundary vertices onto the analytic carriers and improve the
//!    worst tets locally (flips, vertex moves, boundary peels), in rounds.
//!
//! The legacy quality optimizer (`optimize`) remains available as an
//! optional pass on the finished mesh.

// Public surface: the 2D core (`surf2d`), adaptive marking (`adapt`), the
// sizing-field cache (`quadfield`), and `diagnostics`. The MoM/FEM topology +
// quality accessors live in the downstream `rapidmesh_topo` layer (one
// implementation for both the 2D and the 3D-surface endpoint). Everything else
// is the internal mesher engine -- `pub(crate)`, reached only through the
// re-exported entry points below (`mesh_budgeted` / `surface_mesh` / `mesh_plc` /
// `tetrahedralize` / ...). The canonical embedding front door is
// `rapidmesh_topo::{mesh_2d, mesh_3d}`.
pub mod adapt;
pub mod bottomup;
pub mod diagnostics;
pub mod fidelity;
pub mod gradefield;
pub mod mesh3;
pub mod quadfield;
pub mod surf2d;

pub(crate) mod brep_mesh;
pub(crate) mod conform;
pub(crate) mod constants;
pub(crate) mod curve;
pub(crate) mod cvt;
pub(crate) mod domain;
mod geomutil;
pub(crate) mod optimize;
pub mod tri;

pub use adapt::dorfler_mark;
pub use conform::{
    log_metrics, log_surface_metrics, mesh_model, mesh_plc, mesh_plc_with, quality_stats,
    CurveEdge, MeshParams, PointClass, QualityStats, SurfaceFace, SurfaceMesh, TetMesh,
};
pub use cvt::{budgeted, mesh_budgeted};
pub use mesh3::brep::surface_mesh;
pub use optimize::{optimize, OptimizeParams};
pub use tri::{tetrahedralize, Triangulation};
