//! Source adapters for the rapidmesh mesher's own output types, behind the
//! `mesher` feature. The core stays std-only and builds for any external mesh;
//! this wires our `TetMesh` / `SurfaceMesh` into the same complex builders.

use crate::source::{TetSource, TriSource};
use rapidmesh_tet::{SurfaceMesh, TetMesh};

impl TetSource for TetMesh {
    fn n_verts(&self) -> usize {
        self.points.len()
    }
    fn n_tets(&self) -> usize {
        self.tets.len()
    }
    fn tet(&self, i: usize) -> [u32; 4] {
        let t = self.tets[i];
        [t[0] as u32, t[1] as u32, t[2] as u32, t[3] as u32]
    }
}

impl TriSource for SurfaceMesh {
    fn n_verts(&self) -> usize {
        self.points.len()
    }
    fn n_tris(&self) -> usize {
        self.faces.len()
    }
    fn tri(&self, i: usize) -> [u32; 3] {
        let t = self.faces[i].tri;
        [t[0] as u32, t[1] as u32, t[2] as u32]
    }
    fn tri_tag(&self, i: usize) -> i64 {
        self.faces[i].face_tag.0 as i64
    }
}

/// The boundary surface of a tet mesh as a triangle source (its tagged faces).
impl TriSource for TetMesh {
    fn n_verts(&self) -> usize {
        self.points.len()
    }
    fn n_tris(&self) -> usize {
        self.faces.len()
    }
    fn tri(&self, i: usize) -> [u32; 3] {
        let t = self.faces[i].tri;
        [t[0] as u32, t[1] as u32, t[2] as u32]
    }
    fn tri_tag(&self, i: usize) -> i64 {
        self.faces[i].face_tag.0 as i64
    }
}
