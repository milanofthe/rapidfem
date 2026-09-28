//! Geometry front-end: solid primitives and the tagged PLC.
//!
//! The tagged PLC (piecewise-linear complex: watertight triangle surface with
//! face/region tags and back-references to the originating analytic surface) is
//! the central intermediate representation of rapidmesh. Both the CSG path
//! (primitives + booleans) and the later STEP path converge on it; the tet
//! mesher consumes it. Surface back-references exist so the order-2 snapping
//! stage can project midside nodes onto the true surface.

pub mod discrete;
mod faceted;
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
pub use faceted::{Faceted, FlatFacet, Frame, SurfaceKind};
pub use import::{
    import_obj, import_obj_creased, import_stl, import_stl_creased, min_height_ratio,
    validate_closed, ImportError, CREASE_DEG,
};
pub use nurbs::NurbsCurve;
pub use nurbs_surface::NurbsSurface;
pub use plc::{FaceTag, RegionTag, SurfaceRef, TaggedPlc};
pub use polygon::{polygon_orientation, triangulate_polygon};
pub use prim::{
    cylinder, cylinder_iso, extrude_polygon, extrude_profile, extrude_spline_profile, facet_count,
    facet_subdivisions, frustum, frustum_iso, helix, icosphere, loft, mesh_solid, naca0012_profile,
    pipe, revolve, revolve_at, sheet_disk, sheet_nurbs, sheet_polygon, sheet_rect, solid_box,
    sphere, torus, wedge, ProfileEdge,
};
pub use scene::{AssembleError, Scene};
pub use tube::TubePath;
