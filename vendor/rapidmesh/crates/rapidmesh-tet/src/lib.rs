//! Tetrahedral meshing, bottom up (`mesher`): the size field is built
//! (`sizing`), the edges are sampled once (`curve`), each face is meshed
//! alone on the samples of its edges (`surface`), each region is filled by
//! its constrained Delaunay tetrahedralization and refined (`volume`), and
//! the boundary is snapped onto the analytic carriers and the worst tets
//! improved locally (`finish`). The output types are in `mesh`, the
//! parameters in `params`, the quality in `quality` and
//! `diagnostics`.
//!
//! Public surface: the entry points re-exported below. Topology accessors
//! live in `rapidmesh_topo`.
pub(crate) mod adapt;
pub(crate) mod constants;
pub(crate) mod curve;
pub(crate) mod diagnostics;
pub(crate) mod fidelity;
pub(crate) mod finish;
pub(crate) mod mesh;
pub(crate) mod mesher;
pub(crate) mod params;
pub(crate) mod predicates;
pub(crate) mod quality;
pub(crate) mod simplex;
pub(crate) mod sizing;
pub(crate) mod surface;
pub(crate) mod volume;

pub use adapt::{dorfler_mark, Dorfler};
pub use constants::SLIVER_DEG;
pub use diagnostics::{diagnose, Defect, DefectKind, MeshDiagnostics};
pub use fidelity::{measure, Fidelity};
pub use finish::brep::edge_projection;
pub use mesh::{CurveEdge, PointClass, SurfaceFace, SurfaceMesh, TetMesh};
pub use mesher::{mesh_scene, surface_mesh, MeshError};
pub use params::{MeshParams, PeriodicPair};
pub use quality::{log_metrics, log_surface_metrics, quality_stats, QualityStats, RegionQuality};
pub use sizing::{angled, budgeted, CurvatureLaw};
