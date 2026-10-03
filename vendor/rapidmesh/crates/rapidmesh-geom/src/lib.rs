//! Geometry front-end: solid primitives and the tagged PLC.
//!
//! The tagged PLC (piecewise-linear complex: watertight triangle surface with
//! face/region tags and back-references to the originating analytic surface) is
//! the central intermediate representation of rapidmesh. The CSG path
//! (primitives + booleans), the imports and the STEP path converge on it; the
//! B-rep is built from it. The surface back-references give each face its
//! carrier, which the mesher samples and the second-order nodes are projected
//! onto.

pub mod bvh;
pub mod cdt2;
pub mod discrete;
mod faceted;
pub mod grid;
pub mod import;
pub mod nurbs;
pub mod nurbs_surface;
pub mod plc;
pub mod polygon;
pub mod prim;
pub mod scene;
pub mod tube;
pub mod vec3;

pub use discrete::DiscreteSurface;
pub use faceted::{CurveKind, EdgeCurve, Faceted, FlatFacet, Frame, SurfaceKind};
pub use import::{import_obj, import_stl, validate_closed, ImportError, CREASE_DEG};
pub use nurbs::NurbsCurve;
pub use nurbs_surface::NurbsSurface;
pub use plc::{FaceTag, RegionTag, SurfaceRef, TaggedPlc};
pub use polygon::{polygon_orientation, polygon_union, triangulate_polygon};
pub use prim::{
    cylinder, cylinder_iso, extrude_polygon, extrude_profile, extrude_sheet,
    extrude_spline_profile, facet_count, facet_subdivisions, frustum, frustum_iso, helix,
    icosphere, loft, mesh_solid, naca0012_profile, pipe, revolve, revolve_at, sheet_disk,
    sheet_nurbs, sheet_polygon, sheet_rect, solid_box, sphere, torus, wedge, ProfileEdge,
};
pub use scene::{AssembleError, Scene};
pub use tube::TubePath;
