//! `rapidmesh-brep`: a boundary-representation (B-rep) layer between the exact CSG
//! and the mesher.
//!
//! Faces are TRIMMED analytic surfaces, edges are analytic CURVES (including the
//! intersection curves a boolean creates), vertices are corner points. The mesher
//! meshes from this geometry (samples on each edge curve, each face in its
//! chart, then the volume), independent of any input tessellation.
//!
//! Topology is **non-manifold** (Weiler radial-edge): an edge radially links ALL
//! faces meeting along it, and a face carries front/back material labels, so
//! multi-material interfaces and embedded sheets -- rapidmesh's core domain -- are
//! first-class.
//!
//! Deliberately MINIMAL: this layer carries only what the mesher consumes
//! (vertices, edges with an analytic curve + radial face list, faces with
//! oriented boundary loops + region/tag labels). Charts, edge sampling and
//! the volume are the mesher's, so there is no half-edge/pcurve/shell/region
//! machinery here.

use rapidmesh_exact::vector::V3;
use rapidmesh_geom::{FaceTag, RegionTag, Scene, TaggedPlc};
use std::sync::Arc;

pub mod build;
pub mod index;
pub mod topology;

use rapidmesh_geom::Surface;
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

/// The geometry of an edge. The edge also keeps the chain of PLC points it
/// follows, from its first end to its last: a polyline edge is that chain,
/// and the chain seeds the projection onto an intersection. The mesher
/// evaluates these; the B-rep recognises and stores them.
#[derive(Debug, Clone)]
pub enum Curve {
    /// The piece of a carrier curve between its parameters `t[0]` (at the
    /// edge's first end) and `t[1]` (at its last): a line, a circle, an
    /// ellipse, a B-spline (a CAD file's, or a swept profile set in space).
    Piece {
        curve: rapidmesh_geom::Curve<3>,
        t: [f64; 2],
    },
    /// Where two carriers meet with no closed form (cylinder on cylinder,
    /// oblique cone sections, tori): the mesher pulls the chain onto both,
    /// so the edge follows the true curve, not the faceted chain (whose
    /// sagitta would leave slivers astride it).
    Intersection { a: SurfaceId, b: SurfaceId },
    /// The edge is its chain (no carrier known).
    Polyline,
}

impl Curve {
    /// The carrier curve of a piece.
    pub fn carrier(&self) -> Option<&rapidmesh_geom::Curve<3>> {
        match self {
            Curve::Piece { curve, .. } => Some(curve),
            _ => None,
        }
    }

    /// Whether the edge is a piece of a circle.
    pub fn is_circle(&self) -> bool {
        self.carrier().is_some_and(|c| c.as_circle().is_some())
    }
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
    /// and reads its [`Surface`] for on-surface carriers.
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
    /// query and the mesher.
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

    /// Assembles `scene` and builds its model, or names the input the scene
    /// could not assemble.
    pub fn try_of_scene(scene: &Scene) -> Result<Model, rapidmesh_geom::AssembleError> {
        Ok(Model::new(scene.try_assemble()?))
    }

    /// The region, face and edge read model (ids are this model's).
    pub fn topology(&self) -> Topology {
        extract_topology(&self.plc, &self.brep)
    }
}
