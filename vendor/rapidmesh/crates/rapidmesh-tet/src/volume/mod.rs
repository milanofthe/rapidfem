//! The volume of each region: the constrained Delaunay tetrahedralization of
//! its boundary ([`cdt`], on the Delaunay tetrahedralization of [`delaunay`]
//! in the tet store of [`tets`]), checked against the boundary ([`region`])
//! and refined to the size ([`refine`]).

pub(crate) mod cdt;
pub(crate) mod delaunay;
pub(crate) mod points;
pub(crate) mod refine;
pub(crate) mod region;
pub(crate) mod tets;
