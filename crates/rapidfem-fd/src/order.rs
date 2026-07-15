// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Per-cell and per-entity polynomial orders, and the minimum rule that connects
//! them.
//!
//! A mixed-order space assigns each cell its own order `p_K`. The DOFs, though,
//! live on the *entities*: an edge shared by a `p = 2` cell and a `p = 1` cell can
//! carry only one number of DOFs, and the tangential trace on it has to be the same
//! polynomial seen from both sides. The rule that makes this work is the **minimum
//! rule**:
//!
//!   p_E = min { p_K : K contains E }
//!
//! Each entity takes the lowest order of any cell touching it. A cell then uses,
//! on each of its entities, only the functions up to that entity's order — so a
//! `p = 2` cell next to a `p = 1` cell simply drops the second function on the
//! shared edge. Both cells then see the same trace, and conformity is automatic
//! with no constraint equations, no hanging-node machinery and no projection.
//!
//! The price is that the reduction leaks one cell inward: a `p = 1` cell drags its
//! neighbours' shared entities down with it. That is inherent to the minimum rule
//! and is why the order policy (stage 5) should not produce isolated `p = 1` cells.
//!
//! **This requires a hierarchical basis.** The rule assumes that "the functions up
//! to order p" is a meaningful subset of the DOFs — that the order-1 space is a
//! coordinate subspace of the order-2 space. It is, for `BasisKind::Hierarchical`
//! (mode 0 of an edge is exactly the Whitney function), and it is *not* for
//! `BasisKind::Interpolatory`, whose mode-0 block is disjoint from the Whitney
//! space. `NedelecBasis` enforces that.

use crate::mesh::Mesh;

/// The lowest and highest order this element family supports.
pub const P_MIN: u8 = 1;
pub const P_MAX: u8 = 2;

/// How many DOFs an edge of order `p` carries.
///
/// p = 1: the Whitney function alone.
/// p = 2: the Whitney function and the edge gradient ∇(L_a L_b).
#[inline]
pub fn n_edge_dofs(p: u8) -> usize {
    p as usize
}

/// How many DOFs a face of order `p` carries: the two face bubbles, which only
/// exist from order 2. (Cell-interior DOFs would appear at order 3.)
#[inline]
pub fn n_face_dofs(p: u8) -> usize {
    if p >= 2 {
        2
    } else {
        0
    }
}

/// How many DOFs a cell interior carries. Zero below order 3, so zero for now.
#[inline]
pub fn n_cell_dofs(_p: u8) -> usize {
    0
}

/// Orders of the cells, and the entity orders the minimum rule derives from them.
#[derive(Clone, Debug)]
pub struct OrderMap {
    pub cell: Vec<u8>,
    pub edge: Vec<u8>,
    pub face: Vec<u8>,
}

impl OrderMap {
    /// Every cell at the same order. This is the case the whole solver ran at
    /// before mixed order existed, and `p = 2` must reproduce it exactly.
    pub fn uniform(mesh: &Mesh, p: u8) -> OrderMap {
        assert!((P_MIN..=P_MAX).contains(&p), "order {p} is outside [{P_MIN}, {P_MAX}]");
        OrderMap::from_cells(mesh, vec![p; mesh.n_tets()])
    }

    /// Apply the minimum rule to a per-cell order assignment.
    pub fn from_cells(mesh: &Mesh, cell: Vec<u8>) -> OrderMap {
        assert_eq!(cell.len(), mesh.n_tets(), "one order per tetrahedron");
        for &p in &cell {
            assert!((P_MIN..=P_MAX).contains(&p), "order {p} is outside [{P_MIN}, {P_MAX}]");
        }

        let mut edge = vec![u8::MAX; mesh.n_edges()];
        let mut face = vec![u8::MAX; mesh.n_tris()];
        for t in 0..mesh.n_tets() {
            let p = cell[t];
            for &e in &mesh.tet_to_edge[t] {
                edge[e] = edge[e].min(p);
            }
            for &f in &mesh.tet_to_tri[t] {
                face[f] = face[f].min(p);
            }
        }
        // Every edge and face of a tet mesh belongs to at least one tet, so nothing
        // can still be at the sentinel. If it is, the connectivity is broken.
        debug_assert!(edge.iter().all(|&p| p != u8::MAX), "an edge belongs to no tetrahedron");
        debug_assert!(face.iter().all(|&p| p != u8::MAX), "a face belongs to no tetrahedron");

        OrderMap { cell, edge, face }
    }

    /// True when every cell is at the same order — the case in which the element
    /// is the plain uniform one and the interpolatory basis is still valid.
    pub fn is_uniform(&self, p: u8) -> bool {
        self.cell.iter().all(|&q| q == p)
    }

    /// The orders of tet `t`'s six edges, in `TET_EDGE_LOCAL` order.
    #[inline]
    pub fn tet_edge_orders(&self, mesh: &Mesh, t: usize) -> [u8; 6] {
        std::array::from_fn(|i| self.edge[mesh.tet_to_edge[t][i]])
    }

    /// The orders of tet `t`'s four faces, in `TET_FACE_LOCAL` order.
    #[inline]
    pub fn tet_face_orders(&self, mesh: &Mesh, t: usize) -> [u8; 4] {
        std::array::from_fn(|i| self.face[mesh.tet_to_tri[t][i]])
    }

    /// The orders of surface triangle `t`'s three edges, in `TRI_EDGE_LOCAL` order.
    #[inline]
    pub fn tri_edge_orders(&self, mesh: &Mesh, t: usize) -> [u8; 3] {
        std::array::from_fn(|i| self.edge[mesh.tri_to_edge[t][i]])
    }

    /// Total DOFs, for reporting.
    pub fn n_dofs(&self) -> usize {
        self.edge.iter().map(|&p| n_edge_dofs(p)).sum::<usize>()
            + self.face.iter().map(|&p| n_face_dofs(p)).sum::<usize>()
            + self.cell.iter().map(|&p| n_cell_dofs(p)).sum::<usize>()
    }
}

/// The default `θ` of the order policy. See [`wavelength_policy`].
pub const DEFAULT_THETA: f64 = 0.35;

/// The diameter of a tetrahedron: its longest edge.
///
/// The right length scale, and not the same as a mean or a cube root of the
/// volume. A cell that is thin across but long along is *not* well resolved for a
/// wave travelling along it, and its diameter says so where an averaged measure
/// would not.
pub fn cell_diameter(mesh: &Mesh, t: usize) -> f64 {
    mesh.tet_to_edge[t]
        .iter()
        .map(|&e| mesh.edge_lengths[e])
        .fold(0.0_f64, f64::max)
}

/// The wavenumber in each cell, from the cell's own material.
///
/// `k_K = ω·√(εr·μr)/c₀`. The cell's material, not the background: a dielectric
/// slab shortens the wavelength inside it, and a cell that resolves free space is
/// not the same cell that resolves the dielectric.
pub fn cell_wavenumbers(
    mesh: &Mesh,
    er: &[[[num_complex::Complex64; 3]; 3]],
    ur: &[[[num_complex::Complex64; 3]; 3]],
    freq: f64,
) -> Vec<f64> {
    let k0 = 2.0 * std::f64::consts::PI * freq / crate::constants::C0 * mesh.l0;
    (0..mesh.n_tets())
        .map(|t| {
            // The largest refractive index the tensor presents in any direction:
            // the worst case is what has to be resolved.
            let n_eps = (0..3).map(|i| er[t][i][i].re).fold(1.0_f64, f64::max);
            let n_mu = (0..3).map(|i| ur[t][i][i].re).fold(1.0_f64, f64::max);
            k0 * (n_eps.max(0.0) * n_mu.max(0.0)).sqrt()
        })
        .collect()
}

/// The a-priori order policy: reduce to order 1 where the mesh is fine for a reason
/// that has nothing to do with the wavelength.
///
/// ```text
///     p_K = 1   if  k_K · h_K < θ      (geometry-driven refinement)
///     p_K = 2   otherwise              (wavelength-driven)
/// ```
///
/// The reasoning is the dispersion error, which for an order-p element scales as
/// `(k·h)^{2p}`. Where `k·h` is already tiny — a cell shrunk to resolve a gap, a
/// fillet, a thin trace, not to resolve a wave — the order-1 error `(k·h)²` is
/// below anything the rest of the model can deliver, and buying `(k·h)⁴` there is
/// paying six DOFs an edge for accuracy nobody can measure. Those cells go to
/// order 1 and give their DOFs back.
///
/// This needs no solve: it is a-priori, computable from the mesh, the materials and
/// the frequency alone. (A p-decay indicator, which *is* free with a hierarchical
/// basis — the magnitude of an element's top-mode coefficients — is the a-posteriori
/// refinement of this, and is not implemented yet.)
///
/// The minimum rule then propagates the reduction to the entities, which means it
/// leaks one cell outward into the p = 2 region. That is inherent, and it is why θ
/// should not be pushed so high that the p = 1 region becomes a scatter of isolated
/// cells: a compact region is cheaper than the same number of scattered cells.
pub fn wavelength_policy(mesh: &Mesh, k: &[f64], theta: f64) -> OrderMap {
    assert_eq!(k.len(), mesh.n_tets(), "one wavenumber per tetrahedron");
    let cell: Vec<u8> = (0..mesh.n_tets())
        .map(|t| {
            if k[t] * cell_diameter(mesh, t) < theta {
                1
            } else {
                2
            }
        })
        .collect();
    OrderMap::from_cells(mesh, cell)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two cubes side by side, six tets each, so there is a genuine shared face.
    fn two_cubes() -> Mesh {
        let mut nodes = Vec::new();
        for i in 0..3 {
            for j in 0..2 {
                for k in 0..2 {
                    nodes.push([i as f64, j as f64, k as f64]);
                }
            }
        }
        let idx = |i: usize, j: usize, k: usize| (i * 2 + j) * 2 + k;
        let mut tets = Vec::new();
        for i in 0..2 {
            let c = [
                idx(i, 0, 0), idx(i + 1, 0, 0), idx(i + 1, 1, 0), idx(i, 1, 0),
                idx(i, 0, 1), idx(i + 1, 0, 1), idx(i + 1, 1, 1), idx(i, 1, 1),
            ];
            for t in [
                [c[0], c[1], c[2], c[6]],
                [c[0], c[2], c[3], c[6]],
                [c[0], c[3], c[7], c[6]],
                [c[0], c[7], c[4], c[6]],
                [c[0], c[4], c[5], c[6]],
                [c[0], c[5], c[1], c[6]],
            ] {
                tets.push(t);
            }
        }
        Mesh::from_tets(nodes, tets)
    }

    #[test]
    fn uniform_order_two_gives_every_entity_two_dofs() {
        let mesh = two_cubes();
        let o = OrderMap::uniform(&mesh, 2);
        assert!(o.edge.iter().all(|&p| p == 2));
        assert!(o.face.iter().all(|&p| p == 2));
        assert_eq!(o.n_dofs(), 2 * mesh.n_edges() + 2 * mesh.n_tris());
    }

    #[test]
    fn uniform_order_one_is_one_dof_per_edge_and_none_on_faces() {
        let mesh = two_cubes();
        let o = OrderMap::uniform(&mesh, 1);
        assert_eq!(o.n_dofs(), mesh.n_edges());
    }

    /// The minimum rule: an entity touched by any p=1 cell drops to 1, and every
    /// entity of a p=1 cell does, including the ones it shares with p=2 cells.
    #[test]
    fn the_minimum_rule_pulls_shared_entities_down() {
        let mesh = two_cubes();
        let mut cells = vec![2u8; mesh.n_tets()];
        cells[0] = 1; // one cell of the first cube

        let o = OrderMap::from_cells(&mesh, cells);

        // Exactly the entities of tet 0 are at order 1, and nothing else.
        for e in 0..mesh.n_edges() {
            let on_tet0 = mesh.tet_to_edge[0].contains(&e);
            assert_eq!(o.edge[e] == 1, on_tet0, "edge {e}: order {} but on tet0 = {on_tet0}", o.edge[e]);
        }
        for f in 0..mesh.n_tris() {
            let on_tet0 = mesh.tet_to_tri[0].contains(&f);
            assert_eq!(o.face[f] == 1, on_tet0, "face {f}: order {} but on tet0 = {on_tet0}", o.face[f]);
        }

        // The reduction costs DOFs: 6 edges lose one each, 4 faces lose two each.
        let full = OrderMap::uniform(&mesh, 2);
        assert_eq!(full.n_dofs() - o.n_dofs(), 6 * 1 + 4 * 2);
    }
}
