// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! DG mesh topology, per-element face adjacency, outward normals and areas.
//!
//! The core [`Mesh`] already deduplicates triangular faces and records, per
//! tet, its 4 faces (`tet_to_tri`) and, per face, its up-to-2 adjacent tets
//! (`tri_to_tet`). A discontinuous-Galerkin discretisation additionally needs,
//! for each (element, local face): the neighbouring element *and which of its
//! local faces this is* (to match face-node traces), the outward unit normal,
//! and the face area (surface-integral scaling).

use crate::mesh::{Mesh, TET_FACE_LOCAL};

/// Local index of the tet vertex opposite each of the 4 local faces
/// (`TET_FACE_LOCAL` lists 3 of the 4 local node indices; this is the 4th).
const TET_FACE_OPPOSITE: [usize; 4] = [3, 1, 2, 0];

/// One face of one element.
#[derive(Clone, Copy, Debug)]
pub struct DgFace {
    /// Neighbouring element across this face, or `usize::MAX` on a domain boundary.
    pub neighbor: usize,
    /// Which local face (0..4) of the neighbour this is; `usize::MAX` on a boundary.
    pub neighbor_local_face: usize,
    /// Global triangle index into `Mesh::tris`.
    pub tri: usize,
    /// Outward unit normal, as seen from the owning element.
    pub normal: [f64; 3],
    /// Triangle area.
    pub area: f64,
}

/// Per-element face topology for a tetrahedral DG mesh.
pub struct FaceTopology {
    /// 4 entries per tet, flattened: `faces[elem * 4 + local_face]`.
    faces: Vec<DgFace>,
}

impl FaceTopology {
    /// The face record for a given element and local face (0..4).
    #[inline]
    pub fn face(&self, elem: usize, local_face: usize) -> &DgFace {
        &self.faces[elem * 4 + local_face]
    }

    /// Build the face topology from a meshed [`Mesh`].
    pub fn build(mesh: &Mesh) -> Self {
        let mut faces = Vec::with_capacity(mesh.n_tets() * 4);
        for (t, tet) in mesh.tets.iter().enumerate() {
            for lf in 0..4 {
                let [a, b, c] = TET_FACE_LOCAL[lf];
                let v0 = mesh.nodes[tet[a]];
                let v1 = mesh.nodes[tet[b]];
                let v2 = mesh.nodes[tet[c]];
                let vo = mesh.nodes[tet[TET_FACE_OPPOSITE[lf]]];

                let mut n = cross(sub(v1, v0), sub(v2, v0));
                let len = norm(n);
                let area = 0.5 * len;
                for k in 0..3 {
                    n[k] /= len;
                }
                // Flip so the normal points away from the opposite vertex
                // (which lies inside the element).
                if dot(sub(vo, v0), n) > 0.0 {
                    for k in 0..3 {
                        n[k] = -n[k];
                    }
                }

                let tri = mesh.tet_to_tri[t][lf];
                let [t0, t1] = mesh.tri_to_tet[tri];
                let neighbor = if t0 == t { t1 } else { t0 };
                let neighbor_local_face = if neighbor == usize::MAX {
                    usize::MAX
                } else {
                    mesh.tet_to_tri[neighbor]
                        .iter()
                        .position(|&x| x == tri)
                        .expect("a shared face must appear in the neighbour's faces")
                };

                faces.push(DgFace { neighbor, neighbor_local_face, tri, normal: n, area });
            }
        }
        FaceTopology { faces }
    }
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;

    /// The reference tetrahedron (one element, four boundary faces).
    fn unit_tet() -> Mesh {
        Mesh::from_tets(
            vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            vec![[0, 1, 2, 3]],
        )
    }

    /// Two tets sharing triangle {1,2,3}, on opposite sides of the plane
    /// x + y + z = 1.
    fn two_tets() -> Mesh {
        Mesh::from_tets(
            vec![
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 1., 1.],
            ],
            vec![[0, 1, 2, 3], [1, 2, 3, 4]],
        )
    }

    #[test]
    fn single_tet_has_four_boundary_faces() {
        let m = unit_tet();
        let topo = FaceTopology::build(&m);
        for lf in 0..4 {
            let f = topo.face(0, lf);
            assert_eq!(f.neighbor, usize::MAX);
            assert_eq!(f.neighbor_local_face, usize::MAX);
            assert!((norm(f.normal) - 1.0).abs() < 1e-12, "normal not unit");
        }
    }

    #[test]
    fn outward_normals_close_the_tet() {
        // For any closed polyhedron, Σ area_k · n_k = 0.
        let m = unit_tet();
        let topo = FaceTopology::build(&m);
        let mut s = [0.0; 3];
        for lf in 0..4 {
            let f = topo.face(0, lf);
            for k in 0..3 {
                s[k] += f.area * f.normal[k];
            }
        }
        assert!(norm(s) < 1e-12, "Σ a·n = {s:?}");
    }

    #[test]
    fn normals_point_outward() {
        let m = unit_tet();
        let topo = FaceTopology::build(&m);
        let centroid = [0.25, 0.25, 0.25];
        for lf in 0..4 {
            let f = topo.face(0, lf);
            let [a, _, _] = TET_FACE_LOCAL[lf];
            let v0 = m.nodes[m.tets[0][a]];
            // The outward normal points away from the (interior) centroid.
            assert!(dot(f.normal, sub(v0, centroid)) > 0.0);
        }
    }

    #[test]
    fn shared_face_links_both_elements() {
        let m = two_tets();
        let topo = FaceTopology::build(&m);
        // tet 0 local face 3 = {1,2,3}; tet 1 local face 0 = {1,2,3}.
        let f0 = topo.face(0, 3);
        let f1 = topo.face(1, 0);
        assert_eq!(f0.neighbor, 1);
        assert_eq!(f0.neighbor_local_face, 0);
        assert_eq!(f1.neighbor, 0);
        assert_eq!(f1.neighbor_local_face, 3);
        assert_eq!(f0.tri, f1.tri);
        // The two elements see opposite normals on the shared face.
        for k in 0..3 {
            assert!((f0.normal[k] + f1.normal[k]).abs() < 1e-12);
        }
        assert!((f0.area - f1.area).abs() < 1e-12);
    }
}
