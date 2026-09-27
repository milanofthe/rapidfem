//! Where the cells of a tet mesh lie on the geometry: every face and edge of
//! the topology with the B-rep face or edge it lies on, and the regions on
//! the two sides of every face. With the vertex classification of the mesh
//! (`TetMesh::point_class`) this is what a solver selects boundary
//! conditions, ports and materials by, without matching coordinates.

use crate::convention::NONE;
use crate::tet::TetTopology;
use rapidmesh_tet::TetMesh;
use std::collections::HashMap;

/// The classification of a tet mesh's topology.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Classification {
    /// Per topology face: the B-rep face it lies on, [`NONE`] inside a region.
    pub face_patch: Vec<u32>,
    /// Per topology face: its face tag (0 for none), from the B-rep face.
    pub face_tag: Vec<u32>,
    /// Per topology face: the regions of `face_tets[0]` and `face_tets[1]`
    /// (0 where there is no tet: outside the mesh).
    pub face_regions: Vec<[u32; 2]>,
    /// Per topology edge: the B-rep edge it lies on, [`NONE`] elsewhere.
    pub edge_curve: Vec<u32>,
}

impl Classification {
    /// Classifies the faces and edges of `topo`, the topology of `mesh`.
    pub fn build(mesh: &TetMesh, topo: &TetTopology) -> Classification {
        let face_id: HashMap<[u32; 3], u32> = topo
            .faces
            .iter()
            .enumerate()
            .map(|(i, &f)| (f, i as u32))
            .collect();
        let mut face_patch = vec![NONE; topo.faces.len()];
        let mut face_tag = vec![0u32; topo.faces.len()];
        for sf in &mesh.faces {
            let mut k = sf.tri.map(|v| v as u32);
            k.sort_unstable();
            if let Some(&f) = face_id.get(&k) {
                face_patch[f as usize] = sf.patch;
                face_tag[f as usize] = sf.face_tag.0;
            }
        }
        let region = |t: u32| {
            if t == NONE {
                0
            } else {
                mesh.tet_regions[t as usize].0
            }
        };
        let face_regions = topo
            .face_tets
            .iter()
            .map(|&[a, b]| [region(a), region(b)])
            .collect();
        let edge_id: HashMap<[u32; 2], u32> = topo
            .edges
            .iter()
            .enumerate()
            .map(|(i, &e)| (e, i as u32))
            .collect();
        let mut edge_curve = vec![NONE; topo.edges.len()];
        for ce in &mesh.curve_edges {
            let (a, b) = (ce.v[0] as u32, ce.v[1] as u32);
            if let Some(&e) = edge_id.get(&[a.min(b), a.max(b)]) {
                edge_curve[e as usize] = ce.edge;
            }
        }
        Classification {
            face_patch,
            face_tag,
            face_regions,
            edge_curve,
        }
    }
}

/// The classification of a surface mesh's topology.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TriClassification {
    /// Per triangle: the B-rep face it lies on.
    pub tri_patch: Vec<u32>,
    /// Per topology edge: the B-rep edge it lies on, [`NONE`] elsewhere.
    pub edge_curve: Vec<u32>,
}

impl TriClassification {
    /// Classifies the triangles and edges of `topo`, the topology of `mesh`.
    pub fn build(
        mesh: &rapidmesh_tet::SurfaceMesh,
        topo: &crate::tri::TriTopology,
    ) -> TriClassification {
        let edge_id: HashMap<[u32; 2], u32> = topo
            .edges
            .iter()
            .enumerate()
            .map(|(i, &e)| (e, i as u32))
            .collect();
        let mut edge_curve = vec![NONE; topo.edges.len()];
        for ce in &mesh.curve_edges {
            let (a, b) = (ce.v[0] as u32, ce.v[1] as u32);
            if let Some(&e) = edge_id.get(&[a.min(b), a.max(b)]) {
                edge_curve[e as usize] = ce.edge;
            }
        }
        TriClassification {
            tri_patch: mesh.faces.iter().map(|f| f.patch).collect(),
            edge_curve,
        }
    }
}
