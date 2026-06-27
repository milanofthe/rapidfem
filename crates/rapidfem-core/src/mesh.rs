// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Mesh data structure: nodes, edges, tris, tets, and connectivity.
//!
//! Edges and faces are extracted from the tetrahedra, deduplicated by sorted
//! node keys, and cross-referenced (tet↔edge, tet↔face, face↔edge, face↔tet).
//! The local edge/face traversal orders below are a fixed interface convention
//! that, together with sorted global node keys, gives every shared edge/face a
//! consistent orientation across elements — required by the curl-conforming
//! Nédélec DOFs.

use hashbrown::HashMap;

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

/// Local face order within a tetrahedron, as 0-indexed node triples.
/// The 3rd entry (0,3,1) is intentionally not in ascending order.
pub const TET_FACE_LOCAL: [[usize; 3]; 4] = [
    [0, 1, 2], // (1,2,3)
    [0, 2, 3], // (1,3,4)
    [0, 3, 1], // (1,4,2), note reversed!
    [1, 2, 3], // (2,3,4)
];

pub struct Mesh {
    /// Node coordinates: nodes[i] = [x, y, z]
    pub nodes: Vec<[f64; 3]>,
    /// Edges: edges[e] = [n1, n2] sorted (min, max)
    pub edges: Vec<[usize; 2]>,
    /// Triangles: tris[t] = [n1, n2, n3] sorted
    pub tris: Vec<[usize; 3]>,
    /// Tetrahedra: tets[t] = [n1, n2, n3, n4] in gmsh node order (sorted)
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

    /// Inverse maps for fast lookup during construction
    pub inv_edges: HashMap<(usize, usize), usize>,
    pub inv_tris: HashMap<(usize, usize, usize), usize>,

    /// Gmsh face tag → list of triangle indices
    pub ftag_to_tri: HashMap<i32, Vec<usize>>,
    /// Gmsh volume tag → list of tet indices
    pub vtag_to_tet: HashMap<i32, Vec<usize>>,

    /// Characteristic length L₀ (m) the node coordinates were divided by to
    /// non-dimensionalize the geometry (lever ④). `1.0` means the mesh is in its
    /// original physical units (no normalization applied). When > 0 and ≠ 1, the
    /// stored `nodes`/`edge_lengths` are in units of L₀ and physical coordinates
    /// are recovered by multiplying by `l0`. See `derivations/basis_nondim/`.
    pub l0: f64,
}

impl Mesh {
    /// Build all connectivity from raw nodes and tets.
    /// Extracts edges and triangles from tetrahedra, builds inverse maps.
    pub fn from_tets(nodes: Vec<[f64; 3]>, tets: Vec<[usize; 4]>) -> Self {
        let n_tets = tets.len();
        let mut inv_edges: HashMap<(usize, usize), usize> = HashMap::new();
        let mut inv_tris: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut edges: Vec<[usize; 2]> = Vec::new();
        let mut tris: Vec<[usize; 3]> = Vec::new();
        let mut tet_to_edge = vec![[0usize; 6]; n_tets];
        let mut tet_to_tri = vec![[0usize; 4]; n_tets];

        for (ti, tet) in tets.iter().enumerate() {
            // Extract 6 edges
            for (ei, &[li, lj]) in TET_EDGE_LOCAL.iter().enumerate() {
                let (a, b) = (tet[li], tet[lj]);
                let key = if a < b { (a, b) } else { (b, a) };
                let edge_idx = *inv_edges.entry(key).or_insert_with(|| {
                    let idx = edges.len();
                    edges.push([key.0, key.1]);
                    idx
                });
                tet_to_edge[ti][ei] = edge_idx;
            }

            // Extract 4 faces
            for (fi, &[li, lj, lk]) in TET_FACE_LOCAL.iter().enumerate() {
                let mut face = [tet[li], tet[lj], tet[lk]];
                face.sort();
                let key = (face[0], face[1], face[2]);
                let tri_idx = *inv_tris.entry(key).or_insert_with(|| {
                    let idx = tris.len();
                    tris.push(face);
                    idx
                });
                tet_to_tri[ti][fi] = tri_idx;
            }
        }

        // Per-tri edge order convention: (0,1), (1,2), (0,2) of the sorted
        // triangle nodes — matches the surface DOF layout in `basis`.
        let n_tris = tris.len();
        let mut tri_to_edge = vec![[0usize; 3]; n_tris];
        for (ti, tri) in tris.iter().enumerate() {
            let edge_pairs = [
                (tri[0].min(tri[1]), tri[0].max(tri[1])), // edge(0,1)
                (tri[1].min(tri[2]), tri[1].max(tri[2])), // edge(1,2)
                (tri[0].min(tri[2]), tri[0].max(tri[2])), // edge(0,2)
            ];
            for (ei, &key) in edge_pairs.iter().enumerate() {
                tri_to_edge[ti][ei] = inv_edges[&key];
            }
        }

        // Build tri_to_tet. An interior face is shared by exactly two tets,
        // a boundary face by one. A face shared by three or more tets means
        // a non-manifold mesh; report it rather than silently overwriting
        // slot [1], which would corrupt the DG face-jump terms downstream.
        let mut tri_to_tet = vec![[usize::MAX; 2]; n_tris];
        let mut non_manifold = 0usize;
        for (ti, tet_tris) in tet_to_tri.iter().enumerate() {
            for &tri_idx in tet_tris {
                if tri_to_tet[tri_idx][0] == usize::MAX {
                    tri_to_tet[tri_idx][0] = ti;
                } else if tri_to_tet[tri_idx][1] == usize::MAX {
                    tri_to_tet[tri_idx][1] = ti;
                } else {
                    non_manifold += 1;
                }
            }
        }
        if non_manifold > 0 {
            eprintln!(
                "WARNING: non-manifold mesh: {} face-tet incidences beyond \
                 the two-per-face limit were dropped; face-jump terms on \
                 those faces will be wrong",
                non_manifold
            );
        }

        // Compute edge lengths
        let edge_lengths: Vec<f64> = edges.iter().map(|&[a, b]| {
            let dx = nodes[b][0] - nodes[a][0];
            let dy = nodes[b][1] - nodes[a][1];
            let dz = nodes[b][2] - nodes[a][2];
            (dx*dx + dy*dy + dz*dz).sqrt()
        }).collect();

        Mesh {
            nodes, edges, tris, tets,
            tet_to_edge, tet_to_tri, tri_to_edge, tri_to_tet,
            edge_lengths, inv_edges, inv_tris,
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
    /// (zero mean edge). The transform is exactly reversible — physical
    /// coordinates are `node * l0` — so callers restore physical units for
    /// output by multiplying back. Connectivity is coordinate-independent and
    /// untouched; `edge_lengths` is rescaled in step.
    pub fn normalize_characteristic_length(&mut self) -> f64 {
        if self.l0 != 1.0 || self.edge_lengths.is_empty() {
            return self.l0;
        }
        let mean: f64 = self.edge_lengths.iter().sum::<f64>() / self.edge_lengths.len() as f64;
        if !(mean > 0.0) {
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

    /// Get boundary triangles (only one adjacent tet).
    pub fn boundary_tris(&self) -> Vec<usize> {
        (0..self.n_tris())
            .filter(|&i| self.tri_to_tet[i][1] == usize::MAX)
            .collect()
    }

    /// Get triangles for a face tag.
    pub fn tris_for_tag(&self, tag: i32) -> &[usize] {
        self.ftag_to_tri.get(&tag).map_or(&[], |v| v.as_slice())
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
