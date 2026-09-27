//! rapidmesh: conforming tetrahedral and surface meshes for EM FEM and MoM.
//!
//! The Rust API is the whole mesher; the Python package is a thin layer
//! over it. Build a [`Geometry`] from [`shapes`] and sheets, size it
//! through [`Scope`]s, name the faces and edges a solver needs, pair
//! periodic faces, and mesh:
//!
//! ```no_run
//! use rapidmesh::shapes::{Cuboid, Sheet};
//! use rapidmesh::{FaceFilter, Geometry, MeshOptions, Scope};
//!
//! let mut g = Geometry::new(Some(0.5));
//! let air = g.add(Cuboid::new([4.0, 4.0, 4.0]))?;
//! let sub = g.add_solid(Cuboid::new([4.0, 4.0, 1.0]), Some(0.25), false)?;
//! g.label_solid(air, "air");
//! g.label_solid(sub, "substrate");
//! g.add_sheet(&Sheet::xy(1.0, 1.0, [1.5, 1.5, 1.0]), 7, None)?;
//! g.label_tag(7, "patch");
//! g.name(&Scope::surf(Some(FaceFilter::normal([0.0, 0.0, 1.0]))), "top")?;
//!
//! let mesh = g.mesh(&MeshOptions::default())?;
//! let sets = mesh.sets();
//! println!("{mesh}: {} tets in the substrate", sets.cells["substrate"].len());
//! mesh.write_msh("cell.msh")?;
//! # Ok::<(), rapidmesh::Error>(())
//! ```
//!
//! The planar endpoints ([`mesh_2d`], [`mesh_layers`]) mesh tagged 2D
//! regions for MoM.

pub mod export;
mod geometry;
mod mesh;
pub mod scenes;
pub mod shapes;

pub use geometry::{
    Geometry, Level, MeshOptions, Scope, Solid, SurfaceOptions, DEFAULT_CELLS_ACROSS,
};
pub use mesh::{Diagnostics, Labels, Mesh, Run, Sets, SolidInfo, SurfaceMesh, TetView, TriView};
pub use rapidmesh_brep::{EdgeFilter, EdgeKind, FaceFilter, Topology};
pub use rapidmesh_exact::log::{Event, Level as LogLevel};
pub use rapidmesh_geom::TaggedPlc;
pub use rapidmesh_tet::diagnostics::{Defect, DefectKind, MeshDiagnostics};
pub use rapidmesh_tet::fidelity::Fidelity;
pub use rapidmesh_tet::mesh3::PeriodicPair;
pub use rapidmesh_tet::{dorfler_mark, PointClass, QualityStats, SurfaceFace, TetMesh};
pub use rapidmesh_topo::{
    mesh_2d, mesh_layers, overlay_regions, union_regions, Classification, Mesh2D, Mesh2DOptions,
    OverlayRule, Region2D, TetGeometry, TetTopology, TriClassification, TriGeometry, TriTopology,
    FACE_PERMS, NONE,
};

/// What can go wrong building or meshing a geometry.
#[derive(Debug)]
pub enum Error {
    /// A call that does not fit the geometry (a zero axis, a selection
    /// that matches nothing, a file that is no closed solid).
    Invalid(String),
    Io(std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::Invalid(m) => f.write_str(m),
            Error::Io(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Invalid(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::Io(e)
    }
}

/// The level from which the log of a run is printed live (`None` silent;
/// the `RAPIDMESH_LOG` variable does the same).
pub fn set_log_level(level: Option<LogLevel>) {
    rapidmesh_exact::log::set_level(level);
}
