//! The finish of a raw mesh ([`brep::finish_classified`]): a tetrahedral
//! complex (tets labelled by region, boundary and sheet faces, protected
//! curve segments) whose boundary [`snap`] moves onto the analytic carriers
//! and whose worst tets [`improve`] repairs locally.

pub(crate) mod brep;
pub(crate) mod improve;
pub(crate) mod periodic;
pub(crate) mod snap;

/// A point.
pub type P3 = [f64; 3];

pub use crate::mesh::PointClass;

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

/// A labelled tetrahedral complex: the mesher's raw output and what the
/// finish works on.
#[derive(Clone, Debug, Default)]
pub struct Complex {
    pub points: Vec<P3>,
    /// What each point lies on in the source model.
    pub classes: Vec<PointClass>,
    /// Positively oriented tets with a nonzero label.
    pub tets: Vec<[u32; 4]>,
    /// Region label per tet (never 0).
    pub regions: Vec<u32>,
    /// Faces where the label changes, plus sheet faces, wound into their
    /// front region.
    pub faces: Vec<Face>,
    /// The segments of the B-rep edges (vertex pair, edge): every one must
    /// be an edge of the mesh.
    pub edges: Vec<([u32; 2], u32)>,
}
