// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.
//
// The surface element IS the tangential trace of the volume element.
//
// The Robin/port boundary term is assembled with `ned2_tri_stiff` on a boundary
// triangle, and its 8 DOFs are the same global DOFs the volume element carries on
// that face. That only makes sense if the surface functions are literally the
// traces of the volume functions. Until this test, that was an assertion in a
// comment ("sign-matched to volume") backing a second, hand-written construction.
//
// What is proved here, on sheared tetrahedra in a general position:
//
//   1. exactly 8 of the 20 volume DOFs have a nonzero tangential trace on a face,
//      and they are exactly the 8 the DOF map shares with the surface triangle;
//   2. the other 12 trace to zero, so the surface element loses nothing;
//   3. integrating the surviving traces over the face reproduces `ned2_tri_stiff`
//      entrywise, in the DOF order the assembler uses.
//
// (3) is the operative statement: it says the Robin matrix the solver adds to K is
// the one the volume discretisation implies, sign convention and all. The
// symbolic counterpart is `derivations/nedelec2/face_trace.py`.

use rapidfem_core::quadrature::gaus_quad_tri;
use rapidfem_fd::basis::{local_mapping, local_mapping_tri, Nedelec2Basis};
use rapidfem_fd::mesh::Mesh;
use rapidfem_fd::tet_assembly_r2::{barycentric_grads, build_basis, BasisFn};
use rapidfem_fd::tri_assembly_r2::ned2_tri_stiff;

type V3 = [f64; 3];

/// Tetrahedra in a general position: no right angles, no equal edges, node order
/// deliberately not ascending in space.
fn test_tets() -> Vec<[V3; 4]> {
    vec![
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.3, -0.2, 0.1], [1.7, 0.4, -0.3], [0.1, 1.9, 0.6], [-0.4, 0.2, 2.2]],
        [[-1.1, 0.7, 0.2], [0.9, -0.3, 1.4], [2.1, 1.6, -0.5], [0.2, 2.4, 1.9]],
        // A flattish one, to make sure the trace does not quietly depend on shape.
        [[0.0, 0.0, 0.0], [2.0, 0.1, 0.0], [0.7, 1.5, 0.05], [0.9, 0.6, 0.35]],
    ]
}

fn sub(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: &V3, b: &V3) -> V3 {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

fn dot(a: &V3, b: &V3) -> f64 {
    a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

fn norm(a: &V3) -> f64 {
    dot(a, a).sqrt()
}

/// A volume basis function at barycentric `lam`: scale · Σ coeff · L^e · ∇L_g.
fn eval_volume_fn(f: &BasisFn, lam: &[f64; 4], grads: &[V3; 4]) -> V3 {
    let mut v = [0.0; 3];
    for t in &f.terms {
        let mut c = f.scale * t.coeff;
        for (i, &e) in t.exps.iter().enumerate() {
            for _ in 0..e {
                c *= lam[i];
            }
        }
        let g = &grads[t.grad as usize];
        for k in 0..3 {
            v[k] += c * g[k];
        }
    }
    v
}

/// The part of `v` in the plane with unit normal `n`.
fn tangential(v: &V3, n: &V3) -> V3 {
    let d = dot(v, n);
    [v[0] - d*n[0], v[1] - d*n[1], v[2] - d*n[2]]
}

/// The 20 volume basis functions of a single-tet mesh, plus its ∇L.
fn volume_basis(mesh: &Mesh) -> (Vec<BasisFn>, [V3; 4]) {
    let tet = &mesh.tets[0];
    let xs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][0]);
    let ys: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][1]);
    let zs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][2]);
    let (grads, _) = barycentric_grads(&xs, &ys, &zs);

    let tet_edges = &mesh.tet_to_edge[0];
    let edge_len: [f64; 6] = std::array::from_fn(|i| mesh.edge_lengths[tet_edges[i]]);
    let edge_nodes: [[usize; 2]; 6] = std::array::from_fn(|i| mesh.edges[tet_edges[i]]);
    let edge_map = local_mapping(tet, &edge_nodes);

    let tet_tris = &mesh.tet_to_tri[0];
    let tri_nodes: [[usize; 3]; 4] = std::array::from_fn(|i| mesh.tris[tet_tris[i]]);
    let tri_map = local_mapping_tri(tet, &tri_nodes);

    let node_dist = |i: usize, j: usize| {
        ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2) + (zs[i]-zs[j]).powi(2)).sqrt()
    };
    (build_basis(&edge_len, &edge_map, &tri_map, &node_dist), grads)
}

/// The trace-carrying volume DOFs of face `f`, in the surface element's DOF order:
/// the local volume index whose GLOBAL DOF is the surface element's local DOF `s`.
/// This is the correspondence the assembler relies on — a shared global index —
/// not one re-derived here.
fn trace_dofs(basis: &Nedelec2Basis, tri: usize) -> [usize; 8] {
    let tet_dofs = basis.tet_dofs(0);
    let tri_dofs = basis.tri_dofs(tri);
    std::array::from_fn(|s| {
        tet_dofs
            .iter()
            .position(|&g| g == tri_dofs[s])
            .unwrap_or_else(|| panic!("surface DOF {s} is on no volume DOF of this tet"))
    })
}

#[test]
fn only_the_shared_dofs_have_a_tangential_trace() {
    let pts = gaus_quad_tri(4);

    for (ti, verts) in test_tets().iter().enumerate() {
        let mesh = Mesh::from_tets(verts.to_vec(), vec![[0, 1, 2, 3]]);
        let basis = Nedelec2Basis::new(&mesh);
        let (fns, grads) = volume_basis(&mesh);
        let tet = &mesh.tets[0];

        for f in 0..4 {
            let tri = mesh.tet_to_tri[0][f];
            let nodes = mesh.tris[tri];
            let tv: [V3; 3] = std::array::from_fn(|k| mesh.nodes[nodes[k]]);
            let n = {
                let c = cross(&sub(&tv[1], &tv[0]), &sub(&tv[2], &tv[0]));
                let l = norm(&c);
                [c[0]/l, c[1]/l, c[2]/l]
            };
            // Tet-local indices of the face's three nodes, so a face barycentric
            // triple lifts to a tet barycentric quadruple.
            let lmap = local_mapping_tri(tet, &[nodes, nodes, nodes, nodes])[0];

            let shared = trace_dofs(&basis, tri);
            let scale: f64 = mesh.edge_lengths.iter().cloned().fold(0.0, f64::max);

            for v in 0..20 {
                let on_face = shared.contains(&v);
                let mut peak = 0.0_f64;
                for qp in &pts {
                    let mut lam = [0.0; 4];
                    for k in 0..3 {
                        lam[lmap[k]] = qp[k + 1];
                    }
                    let phi = eval_volume_fn(&fns[v], &lam, &grads);
                    peak = peak.max(norm(&tangential(&phi, &n)));
                }
                if on_face {
                    assert!(
                        peak > 1e-3 * scale,
                        "tet {ti} face {f}: shared DOF {v} traces to (near) zero, peak {peak:.3e}"
                    );
                } else {
                    assert!(
                        peak < 1e-12 * scale.max(1.0),
                        "tet {ti} face {f}: DOF {v} is not on the face but has a \
                         tangential trace, peak {peak:.3e}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_traced_robin_matrix_is_the_surface_element() {
    // Degree 4 is exact for the product of two degree-2 functions.
    let pts = gaus_quad_tri(4);
    let gamma = num_complex::Complex64::new(1.0, 0.0);

    for (ti, verts) in test_tets().iter().enumerate() {
        let mesh = Mesh::from_tets(verts.to_vec(), vec![[0, 1, 2, 3]]);
        let basis = Nedelec2Basis::new(&mesh);
        let (fns, grads) = volume_basis(&mesh);
        let tet = &mesh.tets[0];

        for f in 0..4 {
            let tri = mesh.tet_to_tri[0][f];
            let nodes = mesh.tris[tri];
            let tv: [V3; 3] = std::array::from_fn(|k| mesh.nodes[nodes[k]]);
            let c = cross(&sub(&tv[1], &tv[0]), &sub(&tv[2], &tv[0]));
            let area = 0.5 * norm(&c);
            let nrm = [c[0]/norm(&c), c[1]/norm(&c), c[2]/norm(&c)];
            let lmap = local_mapping_tri(tet, &[nodes, nodes, nodes, nodes])[0];
            let shared = trace_dofs(&basis, tri);

            // M_st = ∫_F (n̂×φ_s)·(n̂×φ_t) dA = ∫_F φ_s,tangential · φ_t,tangential dA,
            // by quadrature on the traces of the volume functions.
            let mut m = [[0.0_f64; 8]; 8];
            for qp in &pts {
                let mut lam = [0.0; 4];
                for k in 0..3 {
                    lam[lmap[k]] = qp[k + 1];
                }
                let phi: Vec<V3> = shared
                    .iter()
                    .map(|&v| tangential(&eval_volume_fn(&fns[v], &lam, &grads), &nrm))
                    .collect();
                for s in 0..8 {
                    for t in 0..8 {
                        m[s][t] += qp[0] * area * dot(&phi[s], &phi[t]);
                    }
                }
            }

            // The surface element, on the same triangle, with γ = 1.
            let want = ned2_tri_stiff(&tv, gamma);

            let mut peak = 0.0_f64;
            let mut scale = 1e-300_f64;
            for s in 0..8 {
                for t in 0..8 {
                    scale = scale.max(want[s][t].norm());
                    peak = peak.max((m[s][t] - want[s][t].re).abs());
                    assert!(
                        want[s][t].im.abs() < 1e-30,
                        "the γ=1 surface element must be real"
                    );
                }
            }
            assert!(
                peak / scale < 1e-12,
                "tet {ti} face {f}: the traced Robin matrix differs from ned2_tri_stiff, \
                 rel {:.3e}",
                peak / scale
            );
        }
    }
}
