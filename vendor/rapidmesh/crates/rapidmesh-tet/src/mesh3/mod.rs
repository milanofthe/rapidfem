//! The volume mesher rebuilt around one interface to the geometry.
//!
//! Restricted-Delaunay refinement in the manner of CGAL Mesh_3: the mesher
//! asks the geometry only through [`oracle::DomainOracle`] (region of a
//! point, crossings of a segment, features) and [`oracle::SizeField`], and
//! produces a [`Complex`]: tets labelled by the region at their
//! circumcenter, boundary faces where the label changes, and sheet faces
//! where a restricted facet lies inside one region. Conformity holds by
//! construction: every tet has one label, and the boundary between labels is
//! a closed set of mesh facets. [`snap`] then moves the boundary onto the
//! analytic carriers and [`improve`] repairs the worst tets locally.

pub mod brep;
pub mod improve;
pub(crate) mod layered;
pub mod oracle;
pub(crate) mod periodic;
pub mod refine;
pub mod snap;
pub(crate) mod surfopt;
pub mod verify;

pub use oracle::{Crossing, DomainOracle, FeatureCurve, Patch, SizeField, Uniform, P3};
pub use periodic::PeriodicPair;

/// Where a mesh vertex lies on the geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexKind {
    /// A corner of the domain (index into [`DomainOracle::corners`]).
    Corner(u32),
    /// On a feature curve (index into [`DomainOracle::curves`]).
    Curve(u32),
    /// On a surface patch.
    Patch(u32),
    /// Inside a region.
    Volume,
}

/// A boundary or sheet face of a [`Complex`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Face {
    /// Vertex indices, wound so the normal points into `regions[0]`.
    pub tri: [u32; 3],
    /// The regions in front (`regions[0]`, the normal side) and behind.
    /// Equal for a sheet face.
    pub regions: [u32; 2],
    /// The patch the face lies on (`u32::MAX` when no patch was found for a
    /// label change, which the verifier reports).
    pub patch: u32,
}

/// The mesher's output: a labelled tetrahedral complex.
#[derive(Clone, Debug, Default)]
pub struct Complex {
    pub points: Vec<P3>,
    pub kinds: Vec<VertexKind>,
    /// Positively oriented tets with a nonzero label.
    pub tets: Vec<[u32; 4]>,
    /// Region label per tet (never 0).
    pub regions: Vec<u32>,
    /// Faces where the label changes, plus sheet faces.
    pub faces: Vec<Face>,
    /// The protected curve segments (vertex pair, curve index): every one
    /// must be an edge of the mesh.
    pub feature_edges: Vec<([u32; 2], u32)>,
}
