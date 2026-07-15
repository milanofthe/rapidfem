// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.
//
// Properties of the hierarchical basis — the only basis, once the interpolatory
// one was removed.
//
// The two things that make it the basis, checked directly on the element matrix:
//
//   1. Mode 1 of every edge is a pure gradient, so it is exactly curl-free and its
//      row of the stiffness matrix is identically zero. That is the local
//      exact-sequence property: the curl kernel is explicit in the element.
//   2. Mode 0 of every edge IS the Whitney function (times the edge length),
//      exactly, so order 1 is a coordinate subspace and the hierarchy nests.
//
// That the basis spans the full R2 space, and the earlier fact that it spans the
// same space the removed interpolatory basis did, are proved symbolically in
// `derivations/nedelec2/hierarchical.py`.

use rapidfem_fd::basis::{local_mapping, local_mapping_tri, tet_dof_owners};
use rapidfem_fd::mesh::Mesh;
use rapidfem_fd::tet_assembly::{barycentric_grads, build_basis, tet_stiff_mass, BasisFn};

type V3 = [f64; 3];

fn test_tets() -> Vec<[V3; 4]> {
    vec![
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.3, -0.2, 0.1], [1.7, 0.4, -0.3], [0.1, 1.9, 0.6], [-0.4, 0.2, 2.2]],
        [[-1.1, 0.7, 0.2], [0.9, -0.3, 1.4], [2.1, 1.6, -0.5], [0.2, 2.4, 1.9]],
    ]
}

/// The tet's local maps, taken from a single-tet mesh so the conventions are the
/// assembler's and not this file's.
fn tet_setup(verts: &[V3; 4]) -> ([f64; 4], [f64; 4], [f64; 4], [f64; 6], [[usize; 2]; 6], [[usize; 3]; 4]) {
    let mesh = Mesh::from_tets(verts.to_vec(), vec![[0, 1, 2, 3]]);
    let tet = &mesh.tets[0];
    let xs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][0]);
    let ys: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][1]);
    let zs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][2]);

    let te = &mesh.tet_to_edge[0];
    let edge_len: [f64; 6] = std::array::from_fn(|i| mesh.edge_lengths[te[i]]);
    let edge_nodes: [[usize; 2]; 6] = std::array::from_fn(|i| mesh.edges[te[i]]);
    let edge_map = local_mapping(tet, &edge_nodes);

    let tt = &mesh.tet_to_tri[0];
    let tri_nodes: [[usize; 3]; 4] = std::array::from_fn(|i| mesh.tris[tt[i]]);
    let tri_map = local_mapping_tri(tet, &tri_nodes);

    (xs, ys, zs, edge_len, edge_map, tri_map)
}

fn basis_of(verts: &[V3; 4]) -> (Vec<BasisFn>, [V3; 4], f64) {
    let (xs, ys, zs, edge_len, edge_map, tri_map) = tet_setup(verts);
    let (grads, six_v) = barycentric_grads(&xs, &ys, &zs);
    let node_dist = |i: usize, j: usize| {
        ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2) + (zs[i]-zs[j]).powi(2)).sqrt()
    };
    let owners = tet_dof_owners(&[2; 6], &[2; 4]);
    (build_basis(&owners, &edge_len, &edge_map, &tri_map, &node_dist), grads, six_v)
}

/// The point of the hierarchical basis: mode 1 of every edge is a pure gradient,
/// so its curl vanishes identically and its row of the stiffness matrix is exactly
/// zero. That is the local exact-sequence property, visible in the element matrix.
#[test]
fn the_gradient_dofs_are_curl_free() {
    let ident = {
        let z = num_complex::Complex64::new(0.0, 0.0);
        let o = num_complex::Complex64::new(1.0, 0.0);
        [[o, z, z], [z, o, z], [z, z, o]]
    };
    for (ti, verts) in test_tets().iter().enumerate() {
        let (xs, ys, zs, el, em, tm) = tet_setup(verts);
        let owners = tet_dof_owners(&[2; 6], &[2; 4]);
        let (d, f) = tet_stiff_mass(&owners, &xs, &ys, &zs, &el, &em, &tm, &ident, &ident);

        // Scale by the biggest stiffness entry, so "zero" means zero relative to
        // the matrix and not to an absolute constant that depends on the mesh size.
        let scale = d.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
        assert!(scale > 0.0);

        // Local DOFs 10..16 are the edge mode-1 functions (basis::tet_dof_owners).
        for i in 10..16 {
            for j in 0..20 {
                let v = d[i * 20 + j].norm() / scale;
                assert!(v < 1e-14, "tet {ti}: D[{i},{j}] = {v:.3e}, but DOF {i} must be curl-free");
            }
        }
        // The other 14 must NOT be curl-free, or the space would be too small.
        for i in (0..10).chain(16..20) {
            let row: f64 = (0..20).map(|j| d[i * 20 + j].norm()).sum::<f64>() / scale;
            assert!(row > 1e-6, "tet {ti}: DOF {i} is unexpectedly curl-free");
        }

        // The mass matrix must stay nonsingular for every DOF (the gradient DOFs
        // are only invisible to the curl, not to the field).
        for i in 0..20 {
            assert!(f[i * 20 + i].norm() > 0.0, "tet {ti}: mass diagonal {i} is zero");
        }
    }
}

/// Mode 0 of every edge is the Whitney function times the edge length, exactly.
/// This is what makes order 1 a coordinate subspace: dropping mode 1 leaves the
/// lowest-order Nédélec element and nothing else. A sign or scale slip in
/// `edge_fns` would break the nesting silently, so pin the functions themselves.
#[test]
fn hierarchical_mode0_is_the_whitney_function() {
    for (ti, verts) in test_tets().iter().enumerate() {
        let (xs, ys, zs, el, em, _tm) = tet_setup(verts);
        let (grads, _) = barycentric_grads(&xs, &ys, &zs);
        let (hier, _, _) = basis_of(verts);

        // Sample points inside the tet, in barycentric coordinates.
        let samples = [
            [0.25, 0.25, 0.25, 0.25],
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.2, 0.1],
            [0.05, 0.15, 0.3, 0.5],
        ];
        for e in 0..6 {
            let (a, b) = (em[e][0], em[e][1]);
            let len = el[e];
            for lam in &samples {
                // W_ab = L_a ∇L_b − L_b ∇L_a, times the edge length.
                let want: V3 = std::array::from_fn(|k| {
                    len * (lam[a] * grads[b][k] - lam[b] * grads[a][k])
                });
                // The basis function, evaluated from its term list.
                let f = &hier[e];
                let mut got = [0.0_f64; 3];
                for t in &f.terms {
                    let mut c = f.scale * t.coeff;
                    for (i, &ex) in t.exps.iter().enumerate() {
                        for _ in 0..ex {
                            c *= lam[i];
                        }
                    }
                    for k in 0..3 {
                        got[k] += c * grads[t.grad as usize][k];
                    }
                }
                let mag = want.iter().map(|v| v.abs()).fold(1e-30, f64::max);
                for k in 0..3 {
                    let err = (got[k] - want[k]).abs() / mag;
                    assert!(
                        err < 1e-13,
                        "tet {ti} edge {e}: mode-0 component {k} is {} but the Whitney \
                         function is {} (rel {err:.2e})",
                        got[k], want[k]
                    );
                }
            }
        }
    }
}
