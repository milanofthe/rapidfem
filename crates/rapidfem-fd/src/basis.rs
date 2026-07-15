// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! The degree-of-freedom layout of the volume and surface elements.
//!
//! Three things meet here, and it is worth being precise about which does what.
//!
//! [`crate::order::OrderMap`] says what order every cell and every entity is.
//! [`crate::dofmap::DofMap`] turns the per-entity DOF counts into global indices, by
//! a prefix sum, so it assumes nothing about those counts being equal. This module
//! is the join: [`tet_dof_owners`] and [`tri_dof_owners`] enumerate an element's
//! local DOFs from its entities' orders, and `NedelecBasis` resolves that against
//! the mesh once and caches the local-to-global lists in flat, offset-indexed
//! arrays.
//!
//! **The owner list is the element definition.** `tet_assembly::build_basis`
//! builds one basis function per entry of it and enumerates nothing itself. So
//! there is no second enumeration that could disagree with the DOF map about how
//! many DOFs an element has, which they are, or what order they come in. At a
//! uniform order that would be a nicety; once the count varies per cell it is the
//! only thing keeping the two in step.
//!
//! At uniform order 2 a tetrahedron has 20 DOFs and a boundary triangle 8, in the
//! mode-major order the goldens are pinned to:
//!
//!   tet:  [0..6] edges m0, [6..10] faces m0, [10..16] edges m1, [16..20] faces m1
//!   tri:  [0..3] edges m0, [3] face m0,      [4..7] edges m1,   [7] face m1
//!
//! Under the minimum rule an entity that did not reach a given mode is simply
//! skipped, and both counts shrink. Nothing outside this module may assume 20 or 8.

use crate::dofmap::{DofMap, DofOwner};
use crate::mesh::Mesh;
use crate::order::{self, OrderMap};
use crate::tet_assembly::BasisKind;

/// The local DOFs of a tetrahedral element, as the entities they belong to.
///
/// **This list IS the element definition.** `tet_assembly::build_basis` builds
/// one basis function per entry, by asking the entity's generator for its `k`-th
/// function — it does not enumerate anything itself. So the basis and the DOF map
/// cannot disagree about how many DOFs there are, which ones they are, or what
/// order they come in. With a variable order that is not a nicety; it is the only
/// way to keep the two in step.
///
/// The order within the list is mode-major — all entities' function 0, then all
/// entities' function 1 — because that is the order the uniform order-2 element
/// has always used and the goldens are pinned to. Entities whose order does not
/// reach a given mode are simply skipped.
///
/// `edge_order` and `face_order` are the entity orders after the minimum rule
/// (see [`crate::order`]), in local `TET_EDGE_LOCAL` / `TET_FACE_LOCAL` order.
pub fn tet_dof_owners(edge_order: &[u8; 6], face_order: &[u8; 4]) -> Vec<DofOwner> {
    let mut out = Vec::with_capacity(20);
    for k in 0..order::P_MAX as usize {
        for e in 0..6 {
            if k < order::n_edge_dofs(edge_order[e]) {
                out.push(DofOwner::Edge { entity: e as u8, k: k as u8 });
            }
        }
        for f in 0..4 {
            if k < order::n_face_dofs(face_order[f]) {
                out.push(DofOwner::Face { entity: f as u8, k: k as u8 });
            }
        }
    }
    out
}

/// The same, for a surface triangle. The triangle is its own single face entity,
/// hence `entity: 0` on the face DOFs.
pub fn tri_dof_owners(edge_order: &[u8; 3], face_order: u8) -> Vec<DofOwner> {
    let mut out = Vec::with_capacity(8);
    for k in 0..order::P_MAX as usize {
        for e in 0..3 {
            if k < order::n_edge_dofs(edge_order[e]) {
                out.push(DofOwner::Edge { entity: e as u8, k: k as u8 });
            }
        }
        if k < order::n_face_dofs(face_order) {
            out.push(DofOwner::Face { entity: 0, k: k as u8 });
        }
    }
    out
}

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

pub struct NedelecBasis {
    /// Which basis of the R2 space the elements are built from.
    pub kind: BasisKind,
    /// The order of every cell and, by the minimum rule, of every entity.
    pub orders: OrderMap,
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

impl NedelecBasis {
    /// The default element: the interpolatory R2 basis, uniform order 2.
    pub fn new(mesh: &Mesh) -> Self {
        NedelecBasis::with_kind(mesh, BasisKind::Interpolatory)
    }

    /// A uniform order-2 space in the given basis.
    pub fn with_kind(mesh: &Mesh, kind: BasisKind) -> Self {
        NedelecBasis::with_orders(mesh, kind, OrderMap::uniform(mesh, 2))
    }

    /// A space of arbitrary per-cell order.
    ///
    /// Anything other than uniform order 2 requires the hierarchical basis: the
    /// minimum rule works by keeping the functions *up to* an entity's order, which
    /// is only meaningful when the lower-order space is a coordinate subspace of the
    /// higher one. It is for `Hierarchical` (mode 0 of an edge is the Whitney
    /// function); it is not for `Interpolatory`, whose mode-0 block contains no
    /// Whitney function at all (`derivations/nedelec2/hierarchical.py`, P2).
    /// Allowing it there would silently discretise a space that is neither order 1
    /// nor order 2 and is not conforming across an order jump.
    pub fn with_orders(mesh: &Mesh, kind: BasisKind, orders: OrderMap) -> Self {
        assert!(
            orders.is_uniform(2) || kind == BasisKind::Hierarchical,
            "variable or reduced order requires BasisKind::Hierarchical: the interpolatory \
             basis does not nest, so 'the functions up to order p' is not a subset of its DOFs"
        );

        let n_edges = mesh.n_edges();
        let n_tris = mesh.n_tris();
        let n_tets = mesh.n_tets();

        let dofs = DofMap::new(
            n_edges,
            n_tris,
            n_tets,
            |e| order::n_edge_dofs(orders.edge[e]) as u32,
            |f| order::n_face_dofs(orders.face[f]) as u32,
            |c| order::n_cell_dofs(orders.cell[c]) as u32,
        );

        // Volume elements: the owner list of each tet, from its entities' orders.
        let mut tet_data = Vec::with_capacity(n_tets * 20);
        let mut tet_off = Vec::with_capacity(n_tets + 1);
        let mut scratch = Vec::new();
        tet_off.push(0u32);
        for ti in 0..n_tets {
            let owners = tet_dof_owners(
                &orders.tet_edge_orders(mesh, ti),
                &orders.tet_face_orders(mesh, ti),
            );
            dofs.resolve(&owners, ti, &mesh.tet_to_edge[ti], &mesh.tet_to_tri[ti], &mut scratch);
            tet_data.extend_from_slice(&scratch);
            tet_off.push(tet_data.len() as u32);
        }

        // Surface elements: the triangle is face `ti` of the mesh, and is its own
        // single face entity.
        let mut tri_data = Vec::with_capacity(n_tris * 8);
        let mut tri_off = Vec::with_capacity(n_tris + 1);
        tri_off.push(0u32);
        for ti in 0..n_tris {
            let owners = tri_dof_owners(&orders.tri_edge_orders(mesh, ti), orders.face[ti]);
            dofs.resolve(&owners, ti, &mesh.tri_to_edge[ti], &[ti], &mut scratch);
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

        NedelecBasis {
            kind,
            orders,
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
