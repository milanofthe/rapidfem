//! `rapidmesh-brep`: a boundary-representation (B-rep) layer between the exact CSG
//! and the mesher.
//!
//! Faces are TRIMMED analytic surfaces, edges are analytic CURVES (including the
//! intersection curves a boolean creates), vertices are corner points. The mesher
//! re-meshes from this geometry -- distribute on each edge curve, mesh each
//! trimmed face in its (u,v) parameter space, fill the volume -- independent of
//! any input tessellation.
//!
//! Topology is **non-manifold** (Weiler radial-edge): an edge radially links ALL
//! faces meeting along it, and a face carries front/back material labels, so
//! multi-material interfaces and embedded sheets -- rapidmesh's core domain -- are
//! first-class.
//!
//! Deliberately MINIMAL: this layer carries only what the mesher consumes
//! (vertices, edges with an analytic curve + radial face list, faces with
//! oriented boundary loops + region/tag labels). Everything else the mesher
//! already does -- parameter-space mapping (`surfchart`), point distribution
//! (`curve`), region classification (`region_at`), volume filling -- so there is
//! no half-edge/pcurve/shell/region machinery here (a face chart maps an
//! edge chain into (u,v) on demand).

use rapidmesh_geom::vec3::V3;
use rapidmesh_geom::{FaceTag, RegionTag, Scene, TaggedPlc};
use std::sync::Arc;

pub mod build;
pub mod feature;
pub mod index;
pub mod surface;
pub mod topology;

pub use surface::Surface;
pub use topology::{
    extract_topology, EdgeFilter, EdgeKind, EdgeTopo, FaceFilter, FaceTopo, Topology,
};

// ---- ids -----------------------------------------------------------------

macro_rules! id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);
    };
}
id!(VertexId);
id!(EdgeId);
id!(CoEdgeId);
id!(FaceId);
id!(SurfaceId);

// ---- geometry (analytic) -------------------------------------------------

/// An analytic edge curve. The edge also stores its on-PLC vertex chain
/// (`Edge::verts`), so `Polyline` needs no data and `Intersection` uses the chain
/// as the projection seed. The mesher evaluates these (it owns the curve / chart
/// machinery); the B-rep only RECOGNISES and stores the form.
#[derive(Debug, Clone)]
pub enum Curve {
    /// Straight segment through `p0` with unit direction `dir`.
    Line { p0: V3, dir: V3 },
    /// Circle: center, unit `axis` (normal), `radius`, in-plane unit `x` axis.
    Circle {
        center: V3,
        axis: V3,
        radius: f64,
        x: V3,
    },
    /// A 2D profile NURBS lifted to 3D on an extrusion frame at height `z` over the
    /// parameter range `t`: point = `base + axis*z + u*profile(t).x + v*profile(t).y`.
    /// Self-contained (the airfoil outline edge); the analytic curvature drives the
    /// sizing, tessellation-independent.
    Profile {
        profile: Arc<rapidmesh_geom::nurbs::NurbsCurve>,
        base: V3,
        u: V3,
        v: V3,
        axis: V3,
        t: [f64; 2],
        z: f64,
    },
    /// Ellipse: an oblique plane section of a cylinder. The point is `center`
    /// plus `a·cos(t)·major` plus `b·sin(t)·minor`, with `axis = major x minor`
    /// the section-plane normal. Exact closed form (curvature drives the sizing
    /// analytically, like `Circle`).
    Ellipse {
        center: V3,
        major: V3,
        minor: V3,
        a: f64,
        b: f64,
    },
    /// Intersection of two surfaces, evaluated lazily by projecting the vertex
    /// chain onto both (the mesher reuses its surface projections). Covers every
    /// analytic∩analytic curve with no closed form (cylinder∩cylinder, oblique
    /// cone sections, torus intersections): the chain is densified and each
    /// sample pulled onto BOTH carriers by alternating projection, so the edge
    /// follows the true curve instead of the faceted arrangement chain (whose
    /// sagitta error is the straddler-sliver root cause).
    Intersection { a: SurfaceId, b: SurfaceId },
    /// Faceted fallback: the edge IS its vertex chain (no analytic refinement).
    Polyline,
}

// ---- topology (non-manifold radial-edge) ---------------------------------

/// A corner point (an endpoint of one or more edge chains). Interior facet
/// vertices are NOT B-rep vertices.
#[derive(Debug, Clone)]
pub struct Vertex {
    pub pos: V3,
    /// Every face the point lies on: those of its edges and any it only
    /// touches (a sheet's corner on a wall).
    pub faces: Vec<FaceId>,
}

/// A B-rep edge: a maximal chain of PLC boundary edges between two corners, with
/// its recovered analytic `curve` and the radial list of all uses meeting it.
#[derive(Debug, Clone)]
pub struct Edge {
    /// The corner endpoints (`ends[0]` = `chain.first()`, `ends[1]` = `chain.last()`).
    pub ends: [VertexId; 2],
    /// The ordered on-PLC vertex chain (the polyline the edge follows); the curve
    /// runs from `chain[0]` to `chain[last]`.
    pub chain: Vec<V3>,
    /// The recovered analytic curve.
    pub curve: Curve,
    /// All uses (co-edges) around this edge -- the radial cycle (non-manifold): 2
    /// for a box edge, 3+ at a multi-material interface or a sheet rim. Each
    /// co-edge ties the edge to one adjacent face with its parameter-space trim.
    pub coedges: Vec<CoEdgeId>,
}

/// A directed use of an edge by one face (a "co-edge"): by a loop of the face,
/// or by the face around an edge inside it (a crease of an import, the line
/// where a sheet meets a wall). Every face the edge lies on has one.
#[derive(Debug, Clone)]
pub struct CoEdge {
    pub edge: EdgeId,
    pub face: FaceId,
    /// True if the loop traverses the edge from `ends[0]` to `ends[1]`.
    pub forward: bool,
}

/// An oriented boundary cycle of a face, as co-edges. `loops[0]` of a face is the
/// outer boundary, the rest are holes.
#[derive(Debug, Clone, Default)]
pub struct Loop {
    pub coedges: Vec<CoEdgeId>,
}

/// A trimmed analytic surface. `regions` are the materials on the front
/// (`+normal`) and back sides: equal for an embedded sheet, one being
/// `RegionTag(0)` (background) for an outer wall.
#[derive(Debug, Clone)]
pub struct Face {
    pub surface: SurfaceId,
    pub loops: Vec<Loop>,
    pub regions: [RegionTag; 2],
    pub face_tag: FaceTag,
    /// Index of the originating analytic surface in the source `TaggedPlc`
    /// (`plc.surfaces` / `TetMesh.surfaces`): the mesher tags output faces by it
    /// and reads the `SurfaceKind` for on-surface carriers.
    pub plc_surface: u32,
    /// Scene-solid owner (parallel to `TaggedPlc::surface_owners`).
    pub owner: u32,
    /// The role of its surface in the shape it came from (see
    /// `TaggedPlc::surface_roles`): with `owner` (and the tag of a sheet)
    /// the origin of the face.
    pub role: u32,
    /// Indices of the source `TaggedPlc` triangles that make up this face. The
    /// mesher uses them to seed on-surface points and as a parameter-free
    /// inside/ownership test (a point belongs to the face whose triangle it is
    /// nearest to).
    pub facets: Vec<u32>,
}

/// The boundary-representation model: arena-allocated topology + geometry, linked
/// by ids. Built from the exact CSG arrangement (see [`build::from_plc`]),
/// consumed by the mesher.
#[derive(Debug, Clone, Default)]
pub struct Brep {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub coedges: Vec<CoEdge>,
    pub faces: Vec<Face>,
    pub surfaces: Vec<Surface>,
}

impl Brep {
    pub fn new() -> Brep {
        Brep::default()
    }
    pub fn vertex(&self, v: VertexId) -> &Vertex {
        &self.vertices[v.0 as usize]
    }
    pub fn edge(&self, e: EdgeId) -> &Edge {
        &self.edges[e.0 as usize]
    }
    pub fn coedge(&self, c: CoEdgeId) -> &CoEdge {
        &self.coedges[c.0 as usize]
    }
    pub fn face(&self, f: FaceId) -> &Face {
        &self.faces[f.0 as usize]
    }
    pub fn surface(&self, s: SurfaceId) -> &Surface {
        &self.surfaces[s.0 as usize]
    }
}

/// The geometry model of a scene: its conforming PLC and the B-rep over it,
/// built once and shared by every consumer (the sizing ids, volume and
/// surface meshing), so their ids agree by construction.
#[derive(Debug, Clone)]
pub struct Model {
    pub plc: TaggedPlc,
    pub brep: Brep,
    /// The spatial index over the PLC facets (facet `i` = PLC triangle
    /// `i`), built on first use and shared by the sizing field, the region
    /// query and the mesher's oracle.
    index: std::sync::OnceLock<Arc<index::FacetBvh>>,
}

impl Model {
    /// The model of an assembled PLC.
    pub fn new(plc: TaggedPlc) -> Model {
        let brep = build::from_plc(&plc);
        Model {
            plc,
            brep,
            index: std::sync::OnceLock::new(),
        }
    }

    /// The spatial index over the PLC facets.
    pub fn index(&self) -> Arc<index::FacetBvh> {
        self.index
            .get_or_init(|| {
                let v = |i: u32| self.plc.vertices[i as usize];
                let tris: Vec<rapidmesh_csg::Tri> = self
                    .plc
                    .triangles
                    .iter()
                    .map(|t| rapidmesh_csg::Tri::new(v(t[0]), v(t[1]), v(t[2])))
                    .collect();
                Arc::new(index::FacetBvh::build(&tris))
            })
            .clone()
    }

    /// Assembles `scene` and builds its model.
    pub fn of_scene(scene: &Scene) -> Model {
        Model::new(scene.assemble())
    }

    /// [`Model::of_scene`], or the input the scene could not assemble.
    pub fn try_of_scene(scene: &Scene) -> Result<Model, rapidmesh_geom::AssembleError> {
        Ok(Model::new(scene.try_assemble()?))
    }

    /// The region, face and edge read model (ids are this model's).
    pub fn topology(&self) -> Topology {
        extract_topology(&self.plc, &self.brep)
    }
}
