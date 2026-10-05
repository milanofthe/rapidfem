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
pub mod chart;
pub mod curve;
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
pub mod sheet;
pub mod surface;
pub mod tube;

pub use curve::Curve;
pub use discrete::DiscreteSurface;
pub use faceted::{EdgeCurve, Faceted, FlatFacet};
pub use import::{import_obj, import_stl, validate_closed, ImportError, CREASE_DEG};
pub use nurbs::NurbsCurve;
pub use nurbs_surface::NurbsSurface;
pub use plc::{FaceTag, RegionTag, SurfaceRef, TaggedPlc};
pub use polygon::{crossings, polygon_orientation, polygon_union, triangulate_polygon};
pub use prim::{
    cylinder, extrude_polygon, extrude_profile, extrude_sheet, facet_subdivisions, frustum, helix,
    icosphere, loft, mesh_solid, naca0012_points, pipe, revolve, revolve_at, sheet_disk,
    sheet_nurbs, sheet_polygon, sheet_rect, solid_box, torus, wedge, ProfileEdge,
};
pub use rapidmesh_csg::BoolOp;
pub use scene::{AssembleError, Scene};
pub use sheet::sheet_boolean;
pub use surface::{Bend, Surface};
pub use tube::TubePath;
