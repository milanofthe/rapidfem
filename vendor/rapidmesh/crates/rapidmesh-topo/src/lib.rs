//! Analysis-ready cell-complex view of a mesh.
//!
//! The solver-agnostic, dimension-uniform derivation of a mesh's 0/1/2/3-cell
//! incidence and per-element geometry — the connectivity downstream FEM/MoM
//! solvers otherwise rebuild from scratch. 2D and 3D run through the same code:
//! a triangle mesh's *topology* is identical whether it is planar (MoM) or
//! embedded in 3D (a surface); only *geometry* is coordinate-aware.
//!
//! This crate is basis-free. RWG / Nédélec DOF maps and quadrature layer on top.
//!
//! # The two endpoints (with the `mesher` feature)
//!
//! For embedding, there are exactly two front doors — one for 2D, one for 3D —
//! each returning a complete bundle (mesh + topology + geometry). Use these; the
//! lower-level pieces (`TriTopology`, `TetGeometry`, …) are what they are built
//! from.
//!
//! - [`mesh_2d`] → [`Mesh2D`] — planar / MoM, via the production 2D path
//!   (`surf2d`, the gmsh-grade mesher the wasm landing uses); raw tagged 2D
//!   polygons in.
//! - [`mesh_3d`] → [`Mesh3D`] — volume / FEM; a 3D PLC in.
//!
//! For an anisotropic boundary layer, [`Region2D::local_width`] answers how wide
//! the shape is at a boundary point and [`Region2D::offset_chains`] lays the
//! inward offsets at a distance that varies along it — chains to hand straight
//! back as [`Region2D::constraints`].
//!
//! ```
//! use rapidmesh_topo::{TetTopology, Tets};
//! // one tet -> 6 edges, 4 faces, every face on the boundary
//! let topo = TetTopology::build(&Tets { tets: &[[0, 1, 2, 3]], n_verts: 4 });
//! assert_eq!(topo.edges.len(), 6);
//! assert_eq!(topo.faces.len(), 4);
//! ```

pub mod convention;
pub mod csr;
mod math;
mod source;
mod tet;
mod tri;

#[cfg(feature = "mesher")]
mod bundle;
#[cfg(feature = "mesher")]
mod classes;
#[cfg(feature = "mesher")]
pub mod export;
#[cfg(feature = "mesher")]
pub mod mesher;
#[cfg(feature = "mesher")]
mod offset;

pub use convention::{
    canonical_edge, face_perm, sort3_sign, FACE_PERMS, NONE, TET_EDGE_LOCAL, TET_FACE_LOCAL,
    TRI_EDGE_LOCAL,
};
pub use csr::Csr;
pub use source::{TetSource, Tets, TriSource, Tris};
pub use tet::{TetGeometry, TetTopology};
pub use tri::{TriGeometry, TriTopology};

#[cfg(feature = "mesher")]
pub use bundle::{
    mesh_2d, mesh_3d, mesh_layers, overlay_regions, union_regions, Mesh2D, Mesh2DOptions, Mesh3D,
    Region2D,
};
#[cfg(feature = "mesher")]
pub use classes::{Classification, TriClassification};
#[cfg(feature = "mesher")]
pub use i_overlay::core::overlay_rule::OverlayRule;
