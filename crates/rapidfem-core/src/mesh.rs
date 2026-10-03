// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Mesh data structure: nodes, edges, tris, tets, and connectivity.
//!
//! Edges and faces and their incidences (tet↔edge, tet↔face, face↔edge,
//! face↔tet) come from rapidmesh's [`TetTopology`]: edges `(min, max)`, faces
//! with ascending vertices. The local edge/face traversal orders below are the
//! element's fixed interface convention; together with the sorted global
//! vertex keys they give every shared edge/face a consistent orientation
//! across elements, required by the curl-conforming Nédélec DOFs. They hold
//! the same edges (face vertex sets) per slot as rapidmesh's local orders,
//! the faces in another slot order (`FACE_OF_TOPOLOGY`).

use hashbrown::HashMap;
use crate::geom::{centroid, dot, norm, scale, sub, tri_area_vector, unit};
use rapidmesh_topo::{TetTopology, Tets, NONE};

/// Local edge order within a tetrahedron, as 0-indexed node pairs.
/// The reversed entry (3,1) for the 5th edge is part of the convention and is
/// load-bearing for DOF orientation; do not "normalise" it.
pub const TET_EDGE_LOCAL: [[usize; 2]; 6] = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [3, 1], // reversed on purpose
    [2, 3],
];

/// Local edge order within a triangle, as 0-indexed node pairs of the SORTED
/// triangle. Because the triangle's nodes are stored ascending, each pair is also
/// ascending globally, so an edge carries the same orientation whether it is
/// reached through a triangle or through a tetrahedron. That is what lets the
/// surface element share the volume element's edge DOFs.
pub const TRI_EDGE_LOCAL: [[usize; 2]; 3] = [[0, 1], [1, 2], [0, 2]];

/// The rapidmesh local face (face `i` leaves out vertex `i`) in each slot of
/// [`TET_FACE_LOCAL`].
const FACE_OF_TOPOLOGY: [usize; 4] = [3, 1, 2, 0];

/// Local face order within a tetrahedron, as 0-indexed node triples.
/// The 3rd entry (0,3,1) is intentionally not in ascending order.
pub const TET_FACE_LOCAL: [[usize; 3]; 4] = [
    [0, 1, 2], // (1,2,3)
    [0, 2, 3], // (1,3,4)
    [0, 3, 1], // (1,4,2), note reversed!
    [1, 2, 3], // (2,3,4)
];

#[derive(Clone)]
pub struct Mesh {
    /// Node coordinates: `nodes[i] = [x, y, z]`
    pub nodes: Vec<[f64; 3]>,
    /// Edges: `edges[e] = [n1, n2]` sorted (min, max)
    pub edges: Vec<[usize; 2]>,
    /// Triangles: `tris[t] = [n1, n2, n3]` sorted
    pub tris: Vec<[usize; 3]>,
    /// Tetrahedra: `tets[t] = [n1, n2, n3, n4]`, positively oriented
    pub tets: Vec<[usize; 4]>,

    /// Per-tet: 6 edge indices in TET_EDGE_LOCAL order
    pub tet_to_edge: Vec<[usize; 6]>,
    /// Per-tet: 4 tri indices in TET_FACE_LOCAL order
    pub tet_to_tri: Vec<[usize; 4]>,
    /// Per-tri: 3 edge indices
    pub tri_to_edge: Vec<[usize; 3]>,
    /// Per-tri: up to 2 adjacent tet indices (usize::MAX = no neighbor)
    pub tri_to_tet: Vec<[usize; 2]>,

    /// Edge lengths
    pub edge_lengths: Vec<f64>,

    /// Triangle index by its sorted vertices.
    pub inv_tris: HashMap<(usize, usize, usize), usize>,

    /// Face group tag → list of triangle indices
    pub ftag_to_tri: HashMap<i32, Vec<usize>>,
    /// Volume group tag → list of tet indices
    pub vtag_to_tet: HashMap<i32, Vec<usize>>,

    /// Characteristic length L₀ (m) the node coordinates were divided by to
    /// non-dimensionalize the geometry (lever ④). `1.0` means the mesh is in its
    /// original physical units (no normalization applied). When > 0 and ≠ 1, the
    /// stored `nodes`/`edge_lengths` are in units of L₀ and physical coordinates
    /// are recovered by multiplying by `l0`. See `derivations/basis_nondim/`.
    pub l0: f64,
}

impl Mesh {
    /// Build all connectivity from raw nodes and tets, through rapidmesh's
    /// topology.
    pub fn from_tets(nodes: Vec<[f64; 3]>, tets: Vec<[usize; 4]>) -> Self {
        let tets32: Vec<[u32; 4]> = tets.iter().map(|t| t.map(|v| v as u32)).collect();
        let topo = TetTopology::build(&Tets { tets: &tets32, n_verts: nodes.len() });
        let idx = |v: u32| if v == NONE { usize::MAX } else { v as usize };
        let edges: Vec<[usize; 2]> = topo.edges.iter().map(|e| e.map(|v| v as usize)).collect();
        let tris: Vec<[usize; 3]> = topo.faces.iter().map(|f| f.map(|v| v as usize)).collect();
        let tet_to_edge = topo.tet_edges.iter().map(|e| e.map(|v| v as usize)).collect();
        let tet_to_tri = topo
            .tet_faces
            .iter()
            .map(|f| FACE_OF_TOPOLOGY.map(|k| f[k] as usize))
            .collect();
        let tri_to_edge = topo.face_edges.iter().map(|e| e.map(|v| v as usize)).collect();
        let tri_to_tet = topo.face_tets.iter().map(|t| t.map(idx)).collect();

        // A face shared by three or more tets means a non-manifold mesh: the
        // topology keeps the first two, the DG face-jump terms on it are wrong.
        let mut on_face = vec![0u8; tris.len()];
        for f in topo.tet_faces.iter().flatten() {
            on_face[*f as usize] = on_face[*f as usize].saturating_add(1);
        }
        let non_manifold: usize = on_face.iter().map(|&c| c.saturating_sub(2) as usize).sum();
        if non_manifold > 0 {
            eprintln!(
                "WARNING: non-manifold mesh: {} face-tet incidences beyond \
                 the two-per-face limit were dropped; face-jump terms on \
                 those faces will be wrong",
                non_manifold
            );
        }

        let edge_lengths: Vec<f64> = edges
            .iter()
            .map(|&[a, b]| {
                let d = [nodes[b][0] - nodes[a][0], nodes[b][1] - nodes[a][1], nodes[b][2] - nodes[a][2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .collect();
        let inv_tris = tris.iter().enumerate().map(|(i, t)| ((t[0], t[1], t[2]), i)).collect();

        Mesh {
            nodes, edges, tris, tets,
            tet_to_edge, tet_to_tri, tri_to_edge, tri_to_tet,
            edge_lengths, inv_tris,
            ftag_to_tri: HashMap::new(),
            vtag_to_tet: HashMap::new(),
            l0: 1.0,
        }
    }

    /// Non-dimensionalize the geometry (lever ④): divide all node coordinates by
    /// the characteristic length L₀ = mean edge length, so the solver assembles
    /// on O(1) coordinates regardless of the mesh's physical scale. Returns L₀.
    ///
    /// Idempotent: a no-op if already normalized (`l0 != 1.0`) or degenerate
    /// (zero mean edge). The transform is exactly reversible, physical
    /// coordinates are `node * l0`, so callers restore physical units for
    /// output by multiplying back. Connectivity is coordinate-independent and
    /// untouched; `edge_lengths` is rescaled in step.
    pub fn normalize_characteristic_length(&mut self) -> f64 {
        if self.l0 != 1.0 || self.edge_lengths.is_empty() {
            return self.l0;
        }
        let mean: f64 = self.edge_lengths.iter().sum::<f64>() / self.edge_lengths.len() as f64;
        if mean.is_nan() || mean <= 0.0 {
            return 1.0;
        }
        let inv = 1.0 / mean;
        for n in &mut self.nodes {
            n[0] *= inv; n[1] *= inv; n[2] *= inv;
        }
        for e in &mut self.edge_lengths {
            *e *= inv;
        }
        self.l0 = mean;
        mean
    }

    pub fn n_nodes(&self) -> usize { self.nodes.len() }
    pub fn n_edges(&self) -> usize { self.edges.len() }
    pub fn n_tris(&self) -> usize { self.tris.len() }
    pub fn n_tets(&self) -> usize { self.tets.len() }

    /// Get triangles for a face tag.
    pub fn tris_for_tag(&self, tag: i32) -> &[usize] {
        self.ftag_to_tri.get(&tag).map_or(&[], |v| v.as_slice())
    }

    /// A mask over the nodes: `true` for a node on a triangle of any of the
    /// face groups `tags`.
    pub fn nodes_on_tags(&self, tags: &[i32]) -> Vec<bool> {
        let mut mask = vec![false; self.n_nodes()];
        for &tag in tags {
            for &t in self.tris_for_tag(tag) {
                for n in self.tris[t] {
                    mask[n] = true;
                }
            }
        }
        mask
    }

    /// The distinct nodes of triangles `tris`, in first-seen order.
    pub fn tri_nodes(&self, tris: &[usize]) -> Vec<usize> {
        let mut seen = hashbrown::HashSet::new();
        tris.iter()
            .flat_map(|&t| self.tris[t])
            .filter(|&n| seen.insert(n))
            .collect()
    }

    /// The unit normal of triangle `t` pointing into the first tet it
    /// bounds, `None` for a degenerate triangle or one no tet touches.
    pub fn tri_inward_normal(&self, t: usize) -> Option<[f64; 3]> {
        let [a, b, c] = self.tris[t].map(|n| self.nodes[n]);
        let area = tri_area_vector(a, b, c);
        if norm(area) < 1e-300 {
            return None;
        }
        let tet = self.tri_to_tet[t].into_iter().find(|&x| x != usize::MAX)?;
        let n = unit(area);
        let into = sub(centroid(&self.nodes, self.tets[tet]), a);
        Some(if dot(n, into) < 0.0 { scale(n, -1.0) } else { n })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_manifold_face_keeps_the_first_two_tets() {
        // Three tets all listing nodes {0,1,2} as a face is a non-manifold
        // connectivity. from_tets must keep the first two incidences in
        // tri_to_tet rather than overwriting slot [1] with the third.
        let nodes = vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0],
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 5]];
        let mesh = Mesh::from_tets(nodes, tets);
        assert_eq!(mesh.n_tets(), 3);

        let shared = mesh.inv_tris[&(0, 1, 2)];
        let adj = mesh.tri_to_tet[shared];
        assert_eq!(adj[0], 0, "first tet kept");
        assert_eq!(adj[1], 1, "second tet kept, not overwritten by the third");
    }

    /// `normalize_characteristic_length` divides coordinates by the mean edge
    /// length, records it in `l0`, rescales `edge_lengths`, and is idempotent.
    #[test]
    fn normalize_sets_l0_and_unit_mean_edge() {
        // A tet scaled to a small (RFIC-like) size; mean edge ~ µm.
        let s = 3e-6;
        let nodes = vec![
            [0.0, 0.0, 0.0], [s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s],
        ];
        let mut mesh = Mesh::from_tets(nodes, vec![[0, 1, 2, 3]]);
        let mean_before: f64 =
            mesh.edge_lengths.iter().sum::<f64>() / mesh.edge_lengths.len() as f64;

        let l0 = mesh.normalize_characteristic_length();
        assert!((l0 - mean_before).abs() < 1e-18, "l0 = mean edge length");
        assert_eq!(mesh.l0, l0);

        // Mean edge length of the normalized mesh is exactly 1.
        let mean_after: f64 =
            mesh.edge_lengths.iter().sum::<f64>() / mesh.edge_lengths.len() as f64;
        assert!((mean_after - 1.0).abs() < 1e-12, "mean edge normalized to 1");

        // Physical coordinates are recovered by multiplying back by l0.
        assert!((mesh.nodes[1][0] * mesh.l0 - s).abs() < 1e-18);

        // Idempotent: a second call is a no-op.
        let again = mesh.normalize_characteristic_length();
        assert_eq!(again, l0);
    }
}
