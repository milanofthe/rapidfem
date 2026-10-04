//! The mesher's output: [`TetMesh`] and [`SurfaceMesh`], and what their
//! faces, edges and points carry.

use rapidmesh_geom::{FaceTag, RegionTag, SurfaceKind};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

/// Deterministic hashing: meshing decisions iterate these containers, and a
/// mesher must be reproducible run-to-run (std's RandomState is not).
type DState = BuildHasherDefault<rustc_hash::FxHasher>;
type DMap<K, V> = HashMap<K, V, DState>;

/// A boundary surface mesh: the conforming triangulation of the PLC patches,
/// produced by the surface path ([`crate::mesher::surface_mesh`])
/// without the volume tetrahedralization. Same face schema as [`TetMesh`].
#[derive(Debug)]
pub struct SurfaceMesh {
    /// Surface vertices (PLC corners, graded edge points, patch interior).
    pub points: Vec<[f64; 3]>,
    /// The triangulation of every patch, tagged.
    pub faces: Vec<SurfaceFace>,
    /// The analytic surfaces referenced by [SurfaceFace::surface].
    pub surfaces: Vec<SurfaceKind>,
    /// Per-surface owner solid index, parallel to `surfaces`.
    pub surface_owners: Vec<u32>,
    /// The mesh edges on B-rep edges.
    pub curve_edges: Vec<CurveEdge>,
    /// What each point lies on, parallel to `points`.
    pub point_class: Vec<PointClass>,
}

/// A conforming surface face of the tet mesh, with its PLC tags.
#[derive(Debug, Clone)]
pub struct SurfaceFace {
    /// Global vertex indices.
    pub tri: [usize; 3],
    /// Face tag inherited from the PLC patch (sheets, ports).
    pub face_tag: FaceTag,
    /// Region tags on (front, back) of the source patch.
    pub regions: [RegionTag; 2],
    /// The B-rep face this face lies on.
    pub patch: u32,
    /// The analytic surface this face approximates (index into
    /// [TetMesh::surfaces]).
    pub surface: u32,
}

/// What a mesh vertex lies on: a B-rep vertex, edge or face (ids of the
/// B-rep the mesher builds from the PLC; face ids match
/// [`SurfaceFace::patch`]), or the interior of the region around it. The
/// mesher knows it for every vertex it places; solvers and later passes read
/// it instead of re-deriving it (curved elements project by it, boundary
/// conditions select by it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointClass {
    Vertex(u32),
    Edge(u32),
    Face(u32),
    Interior,
}

impl PointClass {
    /// The dimension of what the point lies on (0 vertex, 1 edge, 2 face,
    /// 3 interior) and its id (`u32::MAX` for the interior).
    pub fn dim_id(self) -> (u8, u32) {
        match self {
            PointClass::Vertex(i) => (0, i),
            PointClass::Edge(i) => (1, i),
            PointClass::Face(i) => (2, i),
            PointClass::Interior => (3, u32::MAX),
        }
    }

    /// The class of [`PointClass::dim_id`]; the interior for any other
    /// dimension or no id.
    pub fn of_dim_id(dim: u8, id: u32) -> PointClass {
        match (dim, id) {
            (_, u32::MAX) => PointClass::Interior,
            (0, i) => PointClass::Vertex(i),
            (1, i) => PointClass::Edge(i),
            (2, i) => PointClass::Face(i),
            _ => PointClass::Interior,
        }
    }
}

/// A mesh edge on a B-rep edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurveEdge {
    /// The two mesh points.
    pub v: [usize; 2],
    /// The B-rep edge it lies on.
    pub edge: u32,
}

/// A region-tagged conforming tetrahedral mesh.
#[derive(Debug)]
pub struct TetMesh {
    /// Mesh vertices (PLC vertices plus the points the mesher added).
    pub points: Vec<[f64; 3]>,
    /// Positively oriented tets.
    pub tets: Vec<[usize; 4]>,
    /// Region of every tet.
    pub tet_regions: Vec<RegionTag>,
    /// The mesh faces tiling the PLC patches, with tags.
    pub faces: Vec<SurfaceFace>,
    /// The analytic surfaces referenced by [SurfaceFace::surface].
    pub surfaces: Vec<SurfaceKind>,
    /// Per-surface owner solid index (scene insertion order, voids included);
    /// `u32::MAX` for sheet surfaces. Parallel to `surfaces`.
    pub surface_owners: Vec<u32>,
    /// `points[..plc_points]` are the PLC's own vertices (the geometry);
    /// everything after is an interior point the mesher added.
    pub plc_points: usize,
    /// What each point lies on, parallel to `points`.
    pub point_class: Vec<PointClass>,
    /// The mesh edges on B-rep curves (the mesher's protected curve
    /// segments), those inside one face included (an import's open crease).
    pub curve_edges: Vec<CurveEdge>,
    /// Periodic pairs: every point on a face `a` of a pair with its image
    /// on face `b`.
    pub periodic_points: Vec<[usize; 2]>,
}

impl TetMesh {
    /// Feature (crease) edges of the surface mesh, derived from the faces.
    /// An edge is a
    /// feature edge iff it is not interior to one smooth surface group:
    /// boundary/non-manifold incidence (face count != 2), or the two faces
    /// differ in analytic surface, face tag, or region pair. Within ONE
    /// `Plane` surface entry that collects several non-coplanar walls (loft
    /// flanks, pipe segments), the planar patch id discriminates, so true
    /// geometric creases survive while the facet seams of curved analytic
    /// surfaces (cylinder barrel) stay smooth.
    ///
    /// The mesher's curve edges come on top, as long as they are still edges
    /// of the surface: they carry the features no label change marks.
    pub fn feature_edges(&self) -> Vec<[usize; 2]> {
        // group key per face: planes split by patch, curved by surface
        // (surface, smooth-id, face-tag, region-lo, region-hi)
        type FaceKey = (u32, u32, u32, u32, u32);
        let face_key = |sf: &SurfaceFace| -> FaceKey {
            let smooth = match self.surfaces[sf.surface as usize] {
                SurfaceKind::Plane { .. } | SurfaceKind::Facets => sf.patch,
                _ => u32::MAX,
            };
            let (r0, r1) = (
                sf.regions[0].0.min(sf.regions[1].0),
                sf.regions[0].0.max(sf.regions[1].0),
            );
            (sf.surface, smooth, sf.face_tag.0, r0, r1)
        };
        // edge -> (incidence count, first face key seen, mixed-key flag)
        let mut edges: DMap<(usize, usize), (u32, FaceKey, bool)> = DMap::default();
        for sf in &self.faces {
            let key = face_key(sf);
            for k in 0..3 {
                let (a, b) = (sf.tri[k], sf.tri[(k + 1) % 3]);
                let e = (a.min(b), a.max(b));
                let entry = edges.entry(e).or_insert((0, key, false));
                entry.0 += 1;
                if entry.1 != key {
                    entry.2 = true;
                }
            }
        }
        let mut out: Vec<[usize; 2]> = edges
            .iter()
            .filter(|(_, &(cnt, _, mixed))| cnt != 2 || mixed)
            .map(|(&(a, b), _)| [a, b])
            .collect();
        out.extend(
            self.curve_edges
                .iter()
                .map(|e| [e.v[0].min(e.v[1]), e.v[0].max(e.v[1])])
                .filter(|e| edges.contains_key(&(e[0], e[1]))),
        );
        out.sort_unstable();
        out.dedup();
        out
    }
}
