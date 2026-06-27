// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Nédélec order-2 (R2) degree-of-freedom mapping.
//!
//! Assigns global DOF indices to the 20 per-tet and 8 per-surface-triangle
//! basis functions of the canonical R2 element (see `tet_assembly_r2`). The
//! global numbering interleaves the two modes: edge·m1, face·m1, edge·m2,
//! face·m2, with n_field = 2·n_edges + 2·n_tris.
//!
//! DOF structure per tetrahedron (20 DOFs total):
//!   [0..6]   = edge DOFs (mode 1), mapped to tet_to_edge[0..6]
//!   [6..10]  = face DOFs (mode 1), mapped to tet_to_tri[0..4] + n_edges
//!   [10..16] = edge DOFs (mode 2), mapped to tet_to_edge[0..6] + (n_tris + n_edges)
//!   [16..20] = face DOFs (mode 2), mapped to tet_to_tri[0..4] + (n_tris + 2*n_edges)
//!
//! DOF structure per surface triangle (8 DOFs total):
//!   [0..3] = edge DOFs (mode 1), mapped to tri_to_edge[0..3]
//!   [3]    = face DOF  (mode 1), mapped to tri_index + n_edges
//!   [4..7] = edge DOFs (mode 2), mapped to tri_to_edge[0..3] + (n_tris + n_edges)
//!   [7]    = face DOF  (mode 2), mapped to tri_index + (n_tris + 2*n_edges)

use crate::mesh::Mesh;

pub struct Nedelec2Basis {
    /// Total number of DOFs in the system: 2*n_edges + 2*n_tris
    pub n_field: usize,
    pub n_tets: usize,
    pub n_tris: usize,
    pub n_edges: usize,
    /// Per-tet DOF indices (20 per tet)
    pub tet_to_field: Vec<[usize; 20]>,
    /// Per-surface-tri DOF indices (8 per tri)
    pub tri_to_field: Vec<[usize; 8]>,
    /// Per-edge DOF indices (2 per edge): [mode1, mode2]
    pub edge_to_field: Vec<[usize; 2]>,
    /// Precomputed row indices for the surface (tri) sparse matrix
    pub tri_rows: Vec<usize>,
    /// Precomputed col indices for tri sparse matrix
    pub tri_cols: Vec<usize>,
}

impl Nedelec2Basis {
    pub fn new(mesh: &Mesh) -> Self {
        let n_edges = mesh.n_edges();
        let n_tris = mesh.n_tris();
        let n_tets = mesh.n_tets();
        let n_field = 2 * n_edges + 2 * n_tris;

        // tet_to_field: 20 DOFs per tet
        let mut tet_to_field = vec![[0usize; 20]; n_tets];
        for ti in 0..n_tets {
            let edges = &mesh.tet_to_edge[ti];
            let faces = &mesh.tet_to_tri[ti];

            // Mode 1 edges (DOFs 0-5)
            for i in 0..6 {
                tet_to_field[ti][i] = edges[i];
            }
            // Mode 1 faces (DOFs 6-9)
            for i in 0..4 {
                tet_to_field[ti][6 + i] = faces[i] + n_edges;
            }
            // Mode 2 edges (DOFs 10-15)
            for i in 0..6 {
                tet_to_field[ti][10 + i] = edges[i] + n_tris + n_edges;
            }
            // Mode 2 faces (DOFs 16-19)
            for i in 0..4 {
                tet_to_field[ti][16 + i] = faces[i] + n_tris + 2 * n_edges;
            }
        }

        // tri_to_field: 8 DOFs per surface triangle
        let mut tri_to_field = vec![[0usize; 8]; n_tris];
        for ti in 0..n_tris {
            let edges = &mesh.tri_to_edge[ti];

            // Mode 1 edges (DOFs 0-2)
            for i in 0..3 {
                tri_to_field[ti][i] = edges[i];
            }
            // Mode 1 face (DOF 3)
            tri_to_field[ti][3] = ti + n_edges;
            // Mode 2 edges (DOFs 4-6)
            for i in 0..3 {
                tri_to_field[ti][4 + i] = edges[i] + n_tris + n_edges;
            }
            // Mode 2 face (DOF 7)
            tri_to_field[ti][7] = ti + n_tris + 2 * n_edges;
        }

        // edge_to_field: 2 DOFs per edge
        let mut edge_to_field = vec![[0usize; 2]; n_edges];
        for ei in 0..n_edges {
            edge_to_field[ei][0] = ei;                        // mode 1
            edge_to_field[ei][1] = ei + n_tris + n_edges;    // mode 2
        }

        // Precompute COO row/col arrays for the n_tris × 64 surface entries
        let n_tri_dofs = 8usize;
        let n2 = n_tri_dofs * n_tri_dofs; // 64
        let nnz_tri = n_tris * n2;
        let mut tri_rows_arr = vec![0usize; nnz_tri];
        let mut tri_cols_arr = vec![0usize; nnz_tri];

        for itri in 0..n_tris {
            let p = itri * n2;
            let indices = &tri_to_field[itri];
            for ii in 0..n_tri_dofs {
                // rows[p+N*ii:p+N*(ii+1)] = indices[ii]
                for k in 0..n_tri_dofs {
                    tri_rows_arr[p + n_tri_dofs * ii + k] = indices[ii];
                }
                // cols[p+ii:p+N2:N] = indices[ii]
                for k in 0..n_tri_dofs {
                    tri_cols_arr[p + ii + n_tri_dofs * k] = indices[ii];
                }
            }
        }

        Nedelec2Basis {
            n_field,
            n_tets,
            n_tris,
            n_edges,
            tet_to_field,
            tri_to_field,
            edge_to_field,
            tri_rows: tri_rows_arr,
            tri_cols: tri_cols_arr,
        }
    }

    /// Flat zero array of size n_tris*64 for accumulating surface (tri) entries.
    pub fn empty_tri_matrix(&self) -> Vec<num_complex::Complex64> {
        vec![num_complex::Complex64::new(0.0, 0.0); self.n_tris * 64]
    }

    /// Convert the flat surface (tri) data array to a CSR matrix, dropping zeros.
    pub fn generate_csr(&self, data: &[num_complex::Complex64]) -> sprs::CsMat<num_complex::Complex64> {
        use sprs::TriMat;
        let mut tri_mat = TriMat::new((self.n_field, self.n_field));
        for (idx, &val) in data.iter().enumerate() {
            if val.re != 0.0 || val.im != 0.0 {
                tri_mat.add_triplet(self.tri_rows[idx], self.tri_cols[idx], val);
            }
        }
        tri_mat.to_csr()
    }
}

/// Convert global node indices in edge/tri arrays to local tet indices (0-3).
///
/// Given tet vertex IDs [v0,v1,v2,v3] and a set of global node IDs,
/// returns the local index (0-3) of each node within the tet.
pub fn local_mapping(tet_verts: &[usize; 4], global_ids: &[[usize; 2]; 6]) -> [[usize; 2]; 6] {
    let mut out = [[0usize; 2]; 6];
    for (i, pair) in global_ids.iter().enumerate() {
        for (j, &gid) in pair.iter().enumerate() {
            for k in 0..4 {
                if tet_verts[k] == gid {
                    out[i][j] = k;
                    break;
                }
            }
        }
    }
    out
}

/// Same as local_mapping but for triangle face nodes (3 nodes per face).
pub fn local_mapping_tri(tet_verts: &[usize; 4], global_ids: &[[usize; 3]; 4]) -> [[usize; 3]; 4] {
    let mut out = [[0usize; 3]; 4];
    for (i, triple) in global_ids.iter().enumerate() {
        for (j, &gid) in triple.iter().enumerate() {
            for k in 0..4 {
                if tet_verts[k] == gid {
                    out[i][j] = k;
                    break;
                }
            }
        }
    }
    out
}
