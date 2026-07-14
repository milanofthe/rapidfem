// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.
//
// The hierarchical basis is a second basis of the SAME R2 space.
//
// Everything downstream depends on that. If it were a different space, the
// hierarchical element would silently solve a different problem, and no golden
// test would notice: goldens pin one basis against its own derivation, not two
// bases against each other.
//
// So this proves it, and proves it as an if-and-only-if rather than a spot check.
// Two bases of a finite-dimensional inner-product space span the same subspace iff
// every function of each lies in the span of the other. "Lies in the span" is
// exactly "the L² projection has zero residual", and the residual is
//
//     r_j = ||ψ_j||² − n_jᵀ M⁻¹ n_j,   M = Gram(φ),  n_j = (∫φ_i·ψ_j)_i
//
// which is computable from the two Gram matrices and the cross-Gram matrix, all of
// which the element integrates exactly. Zero residual both ways, and equal
// dimension, gives span equality.
//
// The symbolic counterpart, with the same conclusion plus the exact-sequence and
// conformity properties, is `derivations/nedelec2/hierarchical.py`.

use rapidfem_fd::basis::{local_mapping, local_mapping_tri};
use rapidfem_fd::mesh::Mesh;
use rapidfem_fd::tet_assembly_r2::{
    barycentric_grads, build_basis, cross_mass, r2_tet_stiff_mass, BasisFn, BasisKind,
};

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

fn basis_of(kind: BasisKind, verts: &[V3; 4]) -> (Vec<BasisFn>, [V3; 4], f64) {
    let (xs, ys, zs, edge_len, edge_map, tri_map) = tet_setup(verts);
    let (grads, six_v) = barycentric_grads(&xs, &ys, &zs);
    let node_dist = |i: usize, j: usize| {
        ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2) + (zs[i]-zs[j]).powi(2)).sqrt()
    };
    (build_basis(kind, &edge_len, &edge_map, &tri_map, &node_dist), grads, six_v)
}

/// Cholesky solve for an SPD system, row-major. The Gram matrix of a basis is SPD
/// by construction, so this cannot fail unless the "basis" is degenerate — which
/// is itself something worth failing on.
fn spd_solve(a: &[f64], n: usize, rhs: &[f64]) -> Vec<f64> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                assert!(s > 0.0, "Gram matrix is not positive definite: the basis is degenerate");
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    // forward, then back
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// The largest relative L² residual of projecting each function of `b` onto
/// span(`a`). Zero iff span(b) ⊆ span(a).
fn projection_residual(a: &[BasisFn], b: &[BasisFn], grads: &[V3; 4], six_v: f64) -> f64 {
    let n = a.len();
    let m = b.len();
    let gram_a = cross_mass(a, a, grads, six_v);
    let gram_b = cross_mass(b, b, grads, six_v);
    let cross = cross_mass(a, b, grads, six_v); // n x m

    let mut worst = 0.0_f64;
    for j in 0..m {
        let nj: Vec<f64> = (0..n).map(|i| cross[i * m + j]).collect();
        let c = spd_solve(&gram_a, n, &nj);
        // ||ψ_j − Σ c_i φ_i||² = ||ψ_j||² − n_jᵀ c
        let sq_norm = gram_b[j * m + j];
        let residual = sq_norm - nj.iter().zip(c.iter()).map(|(x, y)| x * y).sum::<f64>();
        worst = worst.max(residual.abs() / sq_norm);
    }
    worst
}

#[test]
fn both_bases_span_the_same_space() {
    for (ti, verts) in test_tets().iter().enumerate() {
        let (interp, grads, six_v) = basis_of(BasisKind::Interpolatory, verts);
        let (hier, _, _) = basis_of(BasisKind::Hierarchical, verts);
        assert_eq!(interp.len(), 20);
        assert_eq!(hier.len(), 20);

        let h_in_i = projection_residual(&interp, &hier, &grads, six_v);
        let i_in_h = projection_residual(&hier, &interp, &grads, six_v);
        eprintln!("tet {ti}: hier ⊆ interp residual {h_in_i:.3e}, interp ⊆ hier residual {i_in_h:.3e}");

        assert!(h_in_i < 1e-12, "tet {ti}: the hierarchical basis leaves R2 (residual {h_in_i:.3e})");
        assert!(i_in_h < 1e-12, "tet {ti}: the interpolatory basis leaves the hierarchical span (residual {i_in_h:.3e})");
        // Both are 20 functions, each set inside the other's span, and each set is
        // linearly independent (spd_solve would have panicked otherwise). Hence the
        // spans are equal.
    }
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
        let (d, f) = r2_tet_stiff_mass(BasisKind::Hierarchical, &xs, &ys, &zs, &el, &em, &tm, &ident, &ident);

        // Scale by the biggest stiffness entry, so "zero" means zero relative to
        // the matrix and not to an absolute constant that depends on the mesh size.
        let scale = d.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
        assert!(scale > 0.0);

        // Local DOFs 10..16 are the edge mode-1 functions (basis::R2_TET_OWNERS).
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
/// `r2_edge_fns` would break the nesting silently, so pin the functions themselves.
#[test]
fn hierarchical_mode0_is_the_whitney_function() {
    for (ti, verts) in test_tets().iter().enumerate() {
        let (xs, ys, zs, el, em, _tm) = tet_setup(verts);
        let (grads, _) = barycentric_grads(&xs, &ys, &zs);
        let (hier, _, _) = basis_of(BasisKind::Hierarchical, verts);

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
