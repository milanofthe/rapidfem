// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.
//
// The discrete spectrum of a PEC cavity, computed densely and exactly.
//
// This is the stage-3 oracle of docs/fd-basis-plan.md, and it is deliberately
// built without `eigenmode::solve_eigenmode`. That routine runs a Lanczos
// recurrence in the Euclidean inner product on the operator (E−σB)⁻¹B, which is
// NOT self-adjoint there (it is self-adjoint in the B inner product), and it
// accepts every Ritz value of the tridiagonal without a residual check. Its
// eigenVALUES come out roughly right; its eigenVECTORS have an O(1) eigenpair
// residual, and it reports ghost modes below the fundamental. That is a
// pre-existing defect, unrelated to the element basis, and an oracle must not be
// built on it.
//
// So the spectrum is computed here from first principles: assemble E and B,
// eliminate the PEC DOFs, and solve the dense generalised symmetric problem
// E·x = λ·B·x by reducing it to standard form with a Cholesky factor of B.
//
// What this establishes:
//
//   1. The physics. The lowest nonzero eigenvalue is the cavity's fundamental,
//      TE101, which has a closed form. That checks the element, the DOF map, the
//      orientation convention and the PEC elimination together.
//   2. The kernel. The curl-curl operator's null space (the discrete gradients)
//      sits at exactly λ = 0. Its dimension is a property of the space and must
//      match between the two bases.
//   3. The oracle. An eigenvalue of the discrete operator is a property of the
//      SPACE, not of the basis chosen for it. `hierarchical_basis_test` proves the
//      two bases span one space algebraically, on a single element. Here that
//      claim has to survive assembly, the global DOF map and the constraint
//      elimination — which is exactly where an ownership or orientation mistake
//      would hide.

use faer::Mat;
use num_complex::Complex64 as C64;
use rapidfem_fd::basis::Nedelec2Basis;
use rapidfem_fd::mesh::Mesh;
use rapidfem_fd::tet_assembly_r2::{assemble_global_matrices, BasisKind};
use std::collections::HashSet;

const C0: f64 = 299_792_458.0;

/// A box of nx×ny×nz cubes, each cut into six tetrahedra (Kuhn subdivision).
fn box_mesh(lx: f64, ly: f64, lz: f64, nx: usize, ny: usize, nz: usize) -> Mesh {
    let idx = |i: usize, j: usize, k: usize| (i * (ny + 1) + j) * (nz + 1) + k;
    let mut nodes = Vec::new();
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..=nz {
                nodes.push([
                    lx * i as f64 / nx as f64,
                    ly * j as f64 / ny as f64,
                    lz * k as f64 / nz as f64,
                ]);
            }
        }
    }
    let mut tets = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let c = [
                    idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                    idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1),
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

fn boundary_tris(mesh: &Mesh) -> Vec<usize> {
    (0..mesh.n_tris())
        .filter(|&t| mesh.tri_to_tet[t][1] == usize::MAX)
        .collect()
}

/// The generalised eigenvalues of E·x = λ·B·x on the free DOFs, ascending.
///
/// B (the mass matrix) is symmetric positive definite, so B = LLᵀ and the pencil
/// reduces to the standard symmetric problem (L⁻¹EL⁻ᵀ)y = λy with y = Lᵀx. No
/// iteration, no shift, no convergence criterion: this is the whole spectrum.
fn dense_spectrum(kind: BasisKind, mesh: &Mesh, pec_tris: &[usize]) -> Vec<f64> {
    let basis = Nedelec2Basis::with_kind(mesh, kind);

    let mut pec: HashSet<usize> = HashSet::new();
    for &t in pec_tris {
        for &e in &mesh.tri_to_edge[t] {
            pec.extend(basis.edge_dofs(e));
        }
        pec.extend(basis.tri_dofs(t));
    }
    let free: Vec<usize> = (0..basis.n_field).filter(|d| !pec.contains(d)).collect();
    let mut to_free = vec![usize::MAX; basis.n_field];
    for (i, &d) in free.iter().enumerate() {
        to_free[d] = i;
    }
    let n = free.len();

    let id = {
        let (z, o) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0));
        [[o, z, z], [z, o, z], [z, z, o]]
    };
    let (rows, cols, de, db) = assemble_global_matrices(
        mesh,
        &basis,
        &vec![id; mesh.n_tets()],
        &vec![id; mesh.n_tets()],
    );

    // Air and vacuum: the tensors are real, so E and B are real symmetric.
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

    // B = L Lᵀ, in place, lower triangle.
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

/// A cavity small enough for a dense eigendecomposition, coarse but resolved.
fn cavity() -> (Mesh, Vec<usize>, f64) {
    let (a, b, d) = (0.02286, 0.01016, 0.030);
    let mesh = box_mesh(a, b, d, 3, 1, 4);
    let pec = boundary_tris(&mesh);
    // TE101: f = (c₀/2)·√((1/a)² + (1/d)²)
    let f101 = 0.5 * C0 * ((1.0 / a).powi(2) + (1.0 / d).powi(2)).sqrt();
    (mesh, pec, f101)
}

/// λ = k₀², so f = c₀·√λ / 2π.
fn to_ghz(lambda: f64) -> f64 {
    C0 * lambda.max(0.0).sqrt() / (2.0 * std::f64::consts::PI) / 1e9
}

#[test]
fn the_cavity_fundamental_matches_the_closed_form() {
    let (mesh, pec, f101) = cavity();
    eprintln!(
        "cavity: {} tets, {} PEC faces; TE101 (closed form) = {:.6} GHz",
        mesh.n_tets(),
        pec.len(),
        f101 / 1e9
    );

    for kind in [BasisKind::Interpolatory, BasisKind::Hierarchical] {
        let s = dense_spectrum(kind, &mesh, &pec);
        // The discrete gradients sit at λ = 0. Everything above them is physical;
        // there is no spurious branch in between, which is the whole point of a
        // curl-conforming element.
        let scale = s.last().copied().unwrap();
        let n_kernel = s.iter().filter(|&&l| l.abs() < 1e-9 * scale).count();
        let first = s
            .iter()
            .cloned()
            .find(|&l| l.abs() >= 1e-9 * scale)
            .expect("the spectrum is entirely kernel");

        eprintln!(
            "  {kind:?}: {} DOFs, kernel dim {n_kernel}, lowest resonance {:.6} GHz",
            s.len(),
            to_ghz(first)
        );

        let err = (to_ghz(first) * 1e9 - f101).abs() / f101;
        assert!(
            err < 1e-2,
            "{kind:?}: the fundamental came out at {:.6} GHz, closed form {:.6} GHz (rel {err:.3e})",
            to_ghz(first),
            f101 / 1e9
        );
    }
}

/// The oracle. Same space => same spectrum, all of it: the kernel dimension, every
/// resonance, in order. Not bit-identical — the element matrices are genuinely
/// different numbers and the roundoff differs — but far tighter than any
/// discretisation error.
#[test]
fn both_bases_give_the_same_discrete_spectrum() {
    let (mesh, pec, _) = cavity();

    let si = dense_spectrum(BasisKind::Interpolatory, &mesh, &pec);
    let sh = dense_spectrum(BasisKind::Hierarchical, &mesh, &pec);
    assert_eq!(si.len(), sh.len(), "the two bases produced different DOF counts");

    let scale = si.last().copied().unwrap();
    let ki = si.iter().filter(|&&l| l.abs() < 1e-9 * scale).count();
    let kh = sh.iter().filter(|&&l| l.abs() < 1e-9 * scale).count();
    eprintln!("  kernel dimension: interpolatory {ki}, hierarchical {kh}");
    assert_eq!(ki, kh, "the two bases disagree on the dimension of the curl kernel");

    // Compare the nonzero part: the kernel eigenvalues are all "zero" and their
    // roundoff is meaningless to compare relatively.
    let mut worst = 0.0_f64;
    let mut worst_at = 0;
    for i in ki..si.len() {
        let rel = (si[i] - sh[i]).abs() / si[i].abs();
        if rel > worst {
            worst = rel;
            worst_at = i;
        }
    }
    eprintln!(
        "  {} resonances compared; worst relative disagreement {worst:.3e} at index {worst_at} \
         ({:.6} vs {:.6} GHz)",
        si.len() - ki,
        to_ghz(si[worst_at]),
        to_ghz(sh[worst_at])
    );
    for i in ki..(ki + 6).min(si.len()) {
        eprintln!(
            "    mode {}: {:.9} GHz  vs  {:.9} GHz",
            i - ki,
            to_ghz(si[i]),
            to_ghz(sh[i])
        );
    }

    assert!(
        worst < 1e-9,
        "the two bases give different spectra (worst rel {worst:.3e}): they do not span the same \
         space"
    );
}
