// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Global degree-of-freedom numbering, by geometric entity.
//!
//! A curl-conforming basis attaches its DOFs to edges, faces and (from order 3)
//! cell interiors. Conformity is then automatic: two elements sharing an edge
//! see the same global DOFs on it, so their tangential traces agree without any
//! constraint equation.
//!
//! The count per entity is **data, not a constant**. That is the whole point:
//! a mixed-order space gives different entities different orders, so the map is a
//! prefix-sum table rather than a fixed stride (see `docs/fd-basis-plan.md`).
//! The layout is entity-major,
//!
//!   [ edge 0 DOFs | edge 1 DOFs | ... | face 0 DOFs | ... | cell 0 DOFs | ... ]
//!
//! which keeps an entity's DOFs contiguous. (The previous layout was mode-major,
//! `[all edges m1][all faces m1][all edges m2][all faces m2]`, which only works
//! when every entity has the same count. The two differ by a permutation of the
//! unknowns; the assembled system is the same up to that relabelling.)

use crate::mesh::Mesh;

/// Which geometric entity a local element DOF belongs to, and its index within
/// that entity. An element reports this once per local DOF; the map turns it
/// into a global index. Nothing else needs to know the element's internal order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DofOwner {
    /// Local edge 0-5 of the tetrahedron (or 0-2 of a triangle), k-th DOF on it.
    Edge { entity: u8, k: u8 },
    /// Local face 0-3 of the tetrahedron (or the triangle itself), k-th DOF.
    Face { entity: u8, k: u8 },
    /// Cell interior, k-th DOF. Empty below order 3.
    Cell { k: u8 },
}

/// Prefix-sum DOF numbering over the mesh entities.
pub struct DofMap {
    /// `edge_off[e] .. edge_off[e+1]` are edge `e`'s DOFs, offset by `edge_base`.
    edge_off: Vec<u32>,
    face_off: Vec<u32>,
    cell_off: Vec<u32>,
    edge_base: usize,
    face_base: usize,
    cell_base: usize,
    pub n_field: usize,
}

impl DofMap {
    /// Build from a per-entity DOF count. The counts are what an element's
    /// `n_dofs_on(kind, order)` reports; for a uniform order-2 space they are
    /// 2 per edge, 2 per face, 0 per cell.
    pub fn new(
        n_edges: usize,
        n_faces: usize,
        n_cells: usize,
        edge_count: impl Fn(usize) -> u32,
        face_count: impl Fn(usize) -> u32,
        cell_count: impl Fn(usize) -> u32,
    ) -> DofMap {
        let prefix = |n: usize, f: &dyn Fn(usize) -> u32| -> (Vec<u32>, usize) {
            let mut off = Vec::with_capacity(n + 1);
            let mut acc: u32 = 0;
            off.push(0);
            for i in 0..n {
                acc += f(i);
                off.push(acc);
            }
            (off, acc as usize)
        };
        let (edge_off, n_edge_dofs) = prefix(n_edges, &edge_count);
        let (face_off, n_face_dofs) = prefix(n_faces, &face_count);
        let (cell_off, n_cell_dofs) = prefix(n_cells, &cell_count);

        let edge_base = 0;
        let face_base = edge_base + n_edge_dofs;
        let cell_base = face_base + n_face_dofs;
        DofMap {
            edge_off,
            face_off,
            cell_off,
            edge_base,
            face_base,
            cell_base,
            n_field: cell_base + n_cell_dofs,
        }
    }

    /// The uniform Nédélec first-kind order-2 space: 2 DOFs on every edge, 2 on
    /// every face, none in the interior.
    pub fn uniform_r2(mesh: &Mesh) -> DofMap {
        DofMap::new(
            mesh.n_edges(),
            mesh.n_tris(),
            mesh.n_tets(),
            |_| 2,
            |_| 2,
            |_| 0,
        )
    }

    #[inline]
    pub fn edge_dof(&self, edge: usize, k: usize) -> usize {
        let base = self.edge_base + self.edge_off[edge] as usize;
        debug_assert!(k < self.n_edge_dofs(edge), "edge {edge} has no DOF {k}");
        base + k
    }

    #[inline]
    pub fn face_dof(&self, face: usize, k: usize) -> usize {
        let base = self.face_base + self.face_off[face] as usize;
        debug_assert!(k < self.n_face_dofs(face), "face {face} has no DOF {k}");
        base + k
    }

    #[inline]
    pub fn cell_dof(&self, cell: usize, k: usize) -> usize {
        let base = self.cell_base + self.cell_off[cell] as usize;
        debug_assert!(k < self.n_cell_dofs(cell), "cell {cell} has no DOF {k}");
        base + k
    }

    #[inline]
    pub fn n_edge_dofs(&self, edge: usize) -> usize {
        (self.edge_off[edge + 1] - self.edge_off[edge]) as usize
    }

    #[inline]
    pub fn n_face_dofs(&self, face: usize) -> usize {
        (self.face_off[face + 1] - self.face_off[face]) as usize
    }

    #[inline]
    pub fn n_cell_dofs(&self, cell: usize) -> usize {
        (self.cell_off[cell + 1] - self.cell_off[cell]) as usize
    }

    /// Resolve an element's local DOF owners against the mesh entities of one
    /// cell, producing the local-to-global index list.
    ///
    /// `edges` and `faces` are the cell's global entity indices, in the local
    /// order the element used when it reported its owners.
    pub fn resolve(
        &self,
        owners: &[DofOwner],
        cell: usize,
        edges: &[usize],
        faces: &[usize],
        out: &mut Vec<usize>,
    ) {
        out.clear();
        out.reserve(owners.len());
        for o in owners {
            out.push(match *o {
                DofOwner::Edge { entity, k } => self.edge_dof(edges[entity as usize], k as usize),
                DofOwner::Face { entity, k } => self.face_dof(faces[entity as usize], k as usize),
                DofOwner::Cell { k } => self.cell_dof(cell, k as usize),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-uniform map: the offsets must place every entity's DOFs
    /// contiguously and account for every global index exactly once.
    #[test]
    fn variable_counts_are_contiguous_and_complete() {
        // 3 edges with 1, 2, 3 DOFs; 2 faces with 0 and 2; 1 cell with 3.
        let m = DofMap::new(3, 2, 1, |e| (e + 1) as u32, |f| (f * 2) as u32, |_| 3);
        assert_eq!(m.n_field, (1 + 2 + 3) + (0 + 2) + 3);

        let mut seen = vec![0u32; m.n_field];
        for e in 0..3 {
            assert_eq!(m.n_edge_dofs(e), e + 1);
            for k in 0..m.n_edge_dofs(e) {
                seen[m.edge_dof(e, k)] += 1;
            }
        }
        for f in 0..2 {
            for k in 0..m.n_face_dofs(f) {
                seen[m.face_dof(f, k)] += 1;
            }
        }
        for k in 0..m.n_cell_dofs(0) {
            seen[m.cell_dof(0, k)] += 1;
        }
        assert!(seen.iter().all(|&c| c == 1), "every DOF claimed exactly once: {seen:?}");

        // An entity's DOFs must be consecutive.
        assert_eq!(m.edge_dof(2, 0) + 1, m.edge_dof(2, 1));
        assert_eq!(m.edge_dof(2, 1) + 1, m.edge_dof(2, 2));
    }

    /// A face with zero DOFs (what a p=1/p=2 interface produces) must not leave a
    /// hole or shift its neighbours.
    #[test]
    fn empty_entities_take_no_space() {
        let m = DofMap::new(1, 3, 0, |_| 1, |f| if f == 1 { 0 } else { 2 }, |_| 0);
        assert_eq!(m.n_field, 1 + 2 + 0 + 2);
        assert_eq!(m.n_face_dofs(1), 0);
        // face 2 follows face 0 immediately, because face 1 is empty
        assert_eq!(m.face_dof(2, 0), m.face_dof(0, 1) + 1);
    }
}
