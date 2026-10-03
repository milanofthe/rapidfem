//! Analysis-ready cell-complex view of a mesh.
//!
//! The solver-agnostic, dimension-uniform derivation of a mesh's 0/1/2/3-cell
//! incidence and per-element geometry — the connectivity downstream FEM/FVM
//! solvers otherwise rebuild from scratch. 2D and 3D run through the same code:
//! a triangle mesh's *topology* is identical whether it is planar or
//! embedded in 3D (a surface); only *geometry* is coordinate-aware.
//!
//! This crate is basis-free. RWG / Nédélec DOF maps and quadrature layer on top.
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
pub mod foam;
mod math;
mod source;
mod tet;
mod tri;

#[cfg(feature = "mesher")]
mod classes;
#[cfg(feature = "mesher")]
pub mod export;
#[cfg(feature = "mesher")]
pub mod mesher;

pub use convention::{
    canonical_edge, face_perm, sort3_sign, FACE_PERMS, NONE, TET_EDGE_LOCAL, TET_FACE_LOCAL,
    TRI_EDGE_LOCAL,
};
pub use csr::Csr;
pub use source::{TetSource, Tets, TriSource, Tris};
pub use tet::{TetGeometry, TetTopology};
pub use tri::{TriGeometry, TriTopology};

#[cfg(feature = "mesher")]
pub use classes::{Classification, TriClassification};
