// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! The global degree-of-freedom layout for the volume and surface elements.
//!
//! The numbering itself lives in [`crate::dofmap::DofMap`], which is a prefix-sum
//! over the mesh entities and so carries no assumption that every entity holds the
//! same number of DOFs. This module is what connects it to the elements: it says
//! which entity each of an element's local DOFs sits on ([`R2_TET_OWNERS`],
//! [`R2_TRI_OWNERS`]), resolves that against the mesh, and caches the resulting
//! local-to-global lists in flat, offset-indexed arrays.
//!
//! Nothing here is `[usize; 20]` or `[usize; 8]` any more. The counts come from
//! the map, so an element with a different DOF count needs no change on this side
//! (docs/fd-basis-plan.md, stage 1).
//!
//! Local DOF order of the R2 tetrahedron (20 DOFs), as emitted by
//! `tet_assembly_r2::build_basis`:
//!
//!   [0..6]   edge functions, first mode, on local edges 0-5
//!   [6..10]  face functions, first mode, on local faces 0-3
//!   [10..16] edge functions, second mode
//!   [16..20] face functions, second mode
//!
//! Local DOF order of the R2 triangle (8 DOFs):
//!
//!   [0..3] edge, first mode   [3] face, first mode
//!   [4..7] edge, second mode  [7] face, second mode

use crate::dofmap::{DofMap, DofOwner};
use crate::mesh::Mesh;

/// The entity each local DOF of the R2 tetrahedron belongs to.
pub const R2_TET_OWNERS: [DofOwner; 20] = [
    DofOwner::Edge { entity: 0, k: 0 },
    DofOwner::Edge { entity: 1, k: 0 },
    DofOwner::Edge { entity: 2, k: 0 },
    DofOwner::Edge { entity: 3, k: 0 },
    DofOwner::Edge { entity: 4, k: 0 },
    DofOwner::Edge { entity: 5, k: 0 },
    DofOwner::Face { entity: 0, k: 0 },
    DofOwner::Face { entity: 1, k: 0 },
    DofOwner::Face { entity: 2, k: 0 },
    DofOwner::Face { entity: 3, k: 0 },
    DofOwner::Edge { entity: 0, k: 1 },
    DofOwner::Edge { entity: 1, k: 1 },
    DofOwner::Edge { entity: 2, k: 1 },
    DofOwner::Edge { entity: 3, k: 1 },
    DofOwner::Edge { entity: 4, k: 1 },
    DofOwner::Edge { entity: 5, k: 1 },
    DofOwner::Face { entity: 0, k: 1 },
    DofOwner::Face { entity: 1, k: 1 },
    DofOwner::Face { entity: 2, k: 1 },
    DofOwner::Face { entity: 3, k: 1 },
];

/// The entity each local DOF of the R2 surface triangle belongs to. The triangle
/// is its own face, hence `entity: 0` on the face DOFs.
pub const R2_TRI_OWNERS: [DofOwner; 8] = [
    DofOwner::Edge { entity: 0, k: 0 },
    DofOwner::Edge { entity: 1, k: 0 },
    DofOwner::Edge { entity: 2, k: 0 },
    DofOwner::Face { entity: 0, k: 0 },
    DofOwner::Edge { entity: 0, k: 1 },
    DofOwner::Edge { entity: 1, k: 1 },
    DofOwner::Edge { entity: 2, k: 1 },
    DofOwner::Face { entity: 0, k: 1 },
];

/// A ragged list of per-entity DOF indices: `data[off[i]..off[i+1]]`.
struct Ragged {
    data: Vec<usize>,
    off: Vec<u32>,
}

impl Ragged {
    #[inline]
    fn get(&self, i: usize) -> &[usize] {
        &self.data[self.off[i] as usize..self.off[i + 1] as usize]
    }
}

pub struct Nedelec2Basis {
    /// Total number of DOFs in the system.
    pub n_field: usize,
    pub n_tets: usize,
    pub n_tris: usize,
    pub n_edges: usize,
    /// The entity-based numbering these tables were resolved from.
    pub dofs: DofMap,
    tet: Ragged,
    tri: Ragged,
    edge: Ragged,
    /// `tet_nnz[i]..tet_nnz[i+1]` is tet `i`'s slice of the volume COO triplets:
    /// n² entries for an element with n DOFs. Replaces the fixed stride of 400.
    tet_nnz: Vec<usize>,
    /// The same, for the n_i² surface entries per triangle.
    tri_nnz: Vec<usize>,
    /// Precomputed row indices for the surface (tri) sparse matrix.
    pub tri_rows: Vec<usize>,
    /// Precomputed col indices for the surface (tri) sparse matrix.
    pub tri_cols: Vec<usize>,
}

/// Exclusive prefix sum with a trailing total, over the squares of `counts`.
fn square_offsets(counts: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut off = vec![0usize];
    let mut acc = 0usize;
    for n in counts {
        acc += n * n;
        off.push(acc);
    }
    off
}

impl Nedelec2Basis {
    pub fn new(mesh: &Mesh) -> Self {
        let n_edges = mesh.n_edges();
        let n_tris = mesh.n_tris();
        let n_tets = mesh.n_tets();

        let dofs = DofMap::uniform_r2(mesh);

        // Volume elements: resolve the R2 owner list against each tet's entities.
        let mut tet_data = Vec::with_capacity(n_tets * R2_TET_OWNERS.len());
        let mut tet_off = Vec::with_capacity(n_tets + 1);
        let mut scratch = Vec::new();
        tet_off.push(0u32);
        for ti in 0..n_tets {
            dofs.resolve(
                &R2_TET_OWNERS,
                ti,
                &mesh.tet_to_edge[ti],
                &mesh.tet_to_tri[ti],
                &mut scratch,
            );
            tet_data.extend_from_slice(&scratch);
            tet_off.push(tet_data.len() as u32);
        }

        // Surface elements: the triangle is face `ti` of the mesh, and is its own
        // single face entity.
        let mut tri_data = Vec::with_capacity(n_tris * R2_TRI_OWNERS.len());
        let mut tri_off = Vec::with_capacity(n_tris + 1);
        tri_off.push(0u32);
        for ti in 0..n_tris {
            dofs.resolve(&R2_TRI_OWNERS, ti, &mesh.tri_to_edge[ti], &[ti], &mut scratch);
            tri_data.extend_from_slice(&scratch);
            tri_off.push(tri_data.len() as u32);
        }

        // Edge DOFs, for the PEC elimination.
        let mut edge_data = Vec::with_capacity(2 * n_edges);
        let mut edge_off = Vec::with_capacity(n_edges + 1);
        edge_off.push(0u32);
        for ei in 0..n_edges {
            for k in 0..dofs.n_edge_dofs(ei) {
                edge_data.push(dofs.edge_dof(ei, k));
            }
            edge_off.push(edge_data.len() as u32);
        }

        let tet = Ragged { data: tet_data, off: tet_off };
        let tri = Ragged { data: tri_data, off: tri_off };
        let edge = Ragged { data: edge_data, off: edge_off };

        let tet_nnz = square_offsets((0..n_tets).map(|i| tet.get(i).len()));
        let tri_nnz = square_offsets((0..n_tris).map(|i| tri.get(i).len()));

        // COO row/col arrays for the surface entries, one n×n block per triangle.
        let nnz_tri = *tri_nnz.last().unwrap();
        let mut tri_rows = vec![0usize; nnz_tri];
        let mut tri_cols = vec![0usize; nnz_tri];
        for itri in 0..n_tris {
            let indices = tri.get(itri);
            let n = indices.len();
            let p = tri_nnz[itri];
            for ii in 0..n {
                for jj in 0..n {
                    tri_rows[p + n * ii + jj] = indices[ii];
                    tri_cols[p + n * ii + jj] = indices[jj];
                }
            }
        }

        Nedelec2Basis {
            n_field: dofs.n_field,
            n_tets,
            n_tris,
            n_edges,
            dofs,
            tet,
            tri,
            edge,
            tet_nnz,
            tri_nnz,
            tri_rows,
            tri_cols,
        }
    }

    /// Global DOF indices of tetrahedron `i`, in the element's local order.
    #[inline]
    pub fn tet_dofs(&self, i: usize) -> &[usize] {
        self.tet.get(i)
    }

    /// Global DOF indices of surface triangle `i`, in the element's local order.
    #[inline]
    pub fn tri_dofs(&self, i: usize) -> &[usize] {
        self.tri.get(i)
    }

    /// Global DOF indices carried by edge `i`.
    #[inline]
    pub fn edge_dofs(&self, i: usize) -> &[usize] {
        self.edge.get(i)
    }

    /// Offsets into the volume COO arrays: tet `i` owns `[tet_nnz[i], tet_nnz[i+1])`.
    #[inline]
    pub fn tet_nnz_offsets(&self) -> &[usize] {
        &self.tet_nnz
    }

    /// Total number of volume COO triplets, Σ n_i².
    #[inline]
    pub fn n_tet_nnz(&self) -> usize {
        *self.tet_nnz.last().unwrap()
    }

    /// Flat zero array for accumulating the surface (tri) entries, Σ n_i² long.
    pub fn empty_tri_matrix(&self) -> Vec<num_complex::Complex64> {
        vec![num_complex::Complex64::new(0.0, 0.0); *self.tri_nnz.last().unwrap()]
    }

    /// Offset of surface triangle `i`'s n×n block within the flat surface array.
    #[inline]
    pub fn tri_block(&self, i: usize) -> usize {
        self.tri_nnz[i]
    }

    /// Convert the flat surface (tri) data array to a CSR matrix, dropping zeros.
    pub fn generate_csr(
        &self,
        data: &[num_complex::Complex64],
    ) -> sprs::CsMat<num_complex::Complex64> {
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
