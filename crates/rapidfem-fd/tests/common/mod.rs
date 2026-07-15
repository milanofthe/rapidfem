// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.
//
// Shared test scaffolding: box meshes, and the TRUE discrete spectrum of a system,
// computed densely.
//
// The dense spectrum is the reference that the iterative eigensolver is checked
// against, so it must not share any machinery with it. It is built from first
// principles: assemble E and B, eliminate the constrained DOFs, and solve the dense
// generalised symmetric problem E·x = λ·B·x by reducing it to standard form with a
// Cholesky factor of B. No shift, no iteration, no convergence criterion — the whole
// spectrum, exactly.

#![allow(dead_code)] // each test file uses a subset

use faer::Mat;
use num_complex::Complex64 as C64;
use rapidfem_fd::basis::NedelecBasis;
use rapidfem_fd::mesh::Mesh;
use rapidfem_fd::tet_assembly::assemble_global_matrices;
use std::collections::HashSet;

pub const C0: f64 = 299_792_458.0;

/// A box of nx×ny×nz cubes, each cut into six tetrahedra (Kuhn subdivision).
pub fn box_mesh(lx: f64, ly: f64, lz: f64, nx: usize, ny: usize, nz: usize) -> Mesh {
    graded_box(lx, ly, lz, nx, ny, nz, 1.0)
}

/// The same, graded toward the origin in ALL THREE axes when `grade > 1`.
///
/// Grading a single axis would be pointless for anything that looks at a cell's
/// diameter: the diameter is the LONGEST edge, so a cell that is thin in z but
/// full-width in x and y is no smaller than a coarse one. Only a cell that is small
/// in every direction is genuinely refined.
pub fn graded_box(
    lx: f64,
    ly: f64,
    lz: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    grade: f64,
) -> Mesh {
    let idx = |i: usize, j: usize, k: usize| (i * (ny + 1) + j) * (nz + 1) + k;
    let g = |n: usize, i: usize| (i as f64 / n as f64).powf(grade);
    let mut nodes = Vec::new();
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..=nz {
                nodes.push([lx * g(nx, i), ly * g(ny, j), lz * g(nz, k)]);
            }
        }
    }
    let mut tets = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let c = [
                    idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                    idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1),
                    idx(i, j + 1, k + 1),
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
        }
    }
    Mesh::from_tets(nodes, tets)
}

/// Every triangle on the outer boundary: a face with only one adjacent tetrahedron.
pub fn boundary_tris(mesh: &Mesh) -> Vec<usize> {
    (0..mesh.n_tris())
        .filter(|&t| mesh.tri_to_tet[t][1] == usize::MAX)
        .collect()
}

/// The DOFs a PEC condition on `pec_tris` removes.
pub fn pec_dofs(basis: &NedelecBasis, mesh: &Mesh, pec_tris: &[usize]) -> HashSet<usize> {
    let mut pec = HashSet::new();
    for &t in pec_tris {
        for &e in &mesh.tri_to_edge[t] {
            pec.extend(basis.edge_dofs(e));
        }
        pec.extend(basis.tri_dofs(t));
    }
    pec
}

/// Identity material tensors on every tetrahedron: air.
pub fn air(n_tets: usize) -> (Vec<[[C64; 3]; 3]>, Vec<[[C64; 3]; 3]>) {
    let (z, o) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0));
    let id = [[o, z, z], [z, o, z], [z, z, o]];
    (vec![id; n_tets], vec![id; n_tets])
}

/// The generalised eigenvalues of E·x = λ·B·x on the free DOFs, ascending.
///
/// Real materials only: with air, E and B come out real symmetric.
pub fn dense_spectrum(basis: &NedelecBasis, mesh: &Mesh, pec_tris: &[usize]) -> Vec<f64> {
    let pec = pec_dofs(basis, mesh, pec_tris);
    let free: Vec<usize> = (0..basis.n_field).filter(|d| !pec.contains(d)).collect();
    let mut to_free = vec![usize::MAX; basis.n_field];
    for (i, &d) in free.iter().enumerate() {
        to_free[d] = i;
    }
    let n = free.len();

    let (er, ur) = air(mesh.n_tets());
    let (rows, cols, de, db) = assemble_global_matrices(mesh, basis, &er, &ur);

    let mut e = Mat::<f64>::zeros(n, n);
    let mut b = Mat::<f64>::zeros(n, n);
    for k in 0..rows.len() {
        let (r, c) = (to_free[rows[k]], to_free[cols[k]]);
        if r == usize::MAX || c == usize::MAX {
            continue;
        }
        e[(r, c)] += de[k].re;
        b[(r, c)] += db[k].re;
    }

    // B = L Lᵀ.
    let mut l = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut s = b[(i, j)];
            for k in 0..j {
                s -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                assert!(s > 0.0, "the mass matrix is not positive definite");
                l[(i, i)] = s.sqrt();
            } else {
                l[(i, j)] = s / l[(j, j)];
            }
        }
    }

    // C = L⁻¹ E L⁻ᵀ, by two triangular solves.
    let mut c = e.clone();
    for col in 0..n {
        for i in 0..n {
            let mut s = c[(i, col)];
            for k in 0..i {
                s -= l[(i, k)] * c[(k, col)];
            }
            c[(i, col)] = s / l[(i, i)];
        }
    }
    for row in 0..n {
        for j in 0..n {
            let mut s = c[(row, j)];
            for k in 0..j {
                s -= c[(row, k)] * l[(j, k)];
            }
            c[(row, j)] = s / l[(j, j)];
        }
    }

    let eig = c.eigen().expect("the dense eigendecomposition must succeed");
    let vals = eig.S().column_vector();
    let mut out: Vec<f64> = (0..n).map(|i| vals[i].re).collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

/// The nonzero part of a spectrum, and the dimension of the kernel it sat on.
pub fn resonances(basis: &NedelecBasis, mesh: &Mesh, pec: &[usize]) -> (Vec<f64>, usize) {
    let s = dense_spectrum(basis, mesh, pec);
    let scale = s.last().copied().unwrap();
    let k = s.iter().filter(|&&l| l.abs() < 1e-9 * scale).count();
    (s[k..].to_vec(), k)
}

/// A WR-90-shaped cavity, coarse enough for a dense eigendecomposition.
/// Returns the mesh, its PEC faces, and the closed-form TE101 frequency.
pub fn cavity() -> (Mesh, Vec<usize>, f64) {
    let (a, b, d): (f64, f64, f64) = (0.02286, 0.01016, 0.030);
    let mesh = box_mesh(a, b, d, 3, 1, 4);
    let pec = boundary_tris(&mesh);
    let f101 = 0.5 * C0 * ((1.0 / a).powi(2) + (1.0 / d).powi(2)).sqrt();
    (mesh, pec, f101)
}

/// λ = k₀², so f = c₀·√λ / 2π, in GHz.
pub fn to_ghz(lambda: f64) -> f64 {
    C0 * lambda.max(0.0).sqrt() / (2.0 * std::f64::consts::PI) / 1e9
}

/// k₀² for a frequency in Hz — the eigenvalue the pencil actually carries.
pub fn to_lambda(f_hz: f64) -> f64 {
    (2.0 * std::f64::consts::PI * f_hz / C0).powi(2)
}
