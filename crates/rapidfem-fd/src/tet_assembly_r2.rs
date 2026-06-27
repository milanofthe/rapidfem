// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Clean-room Nédélec first-kind order-2 (R2) tetrahedral element.
//!
//! Independently derived element assembly for the 20-DOF curl-conforming
//! element. The 20 basis functions are built from the Whitney function
//! W_ab = L_a ∇L_b − L_b ∇L_a scaled by a nodal barycentric weight:
//!
//!   edge e=(a,b), length ℓ:   φ_e1 = ℓ L_a W_ab,   φ_e2 = ℓ L_b W_ab
//!   face f=(n0,n1,n2):
//!       φ_f1 = |n0 n2| · L_n1 (L_n0 ∇L_n2 − L_n2 ∇L_n0)
//!       φ_f2 = |n0 n1| · L_n2 (L_n0 ∇L_n1 − L_n1 ∇L_n0)
//!
//! This basis spans the canonical R2 space (P1)³ ⊕ {p ∈ H̃2³ : x·p = 0};
//! the derivation, completeness proof and spectral identification live in
//! `derivations/nedelec2/` (element.py, canonical_r2.py). Element matrices:
//!
//!   stiffness  D_ij = ∫ (∇×φ_i)·μ⁻¹·(∇×φ_j) dV
//!   mass       F_ij = ∫  φ_i·ε·φ_j           dV
//!
//! Every basis function is a sum of terms `coeff · L_p L_q · ∇L_g`. Products
//! of barycentric coordinates integrate in closed form (see `coefficients`),
//! so the whole element is assembled exactly with small fixed loops — no
//! quadrature. DOF order matches the assembler: [6 edge·m1][4 face·m1]
//! [6 edge·m2][4 face·m2].

use num_complex::Complex64 as C64;
use crate::coefficients::volume_coeff;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;

type V3 = [f64; 3];

/// Inverse of a complex 3×3 tensor (textbook cofactor / determinant form).
/// Panics on a (near-)singular tensor rather than emit NaNs into the system.
fn matinv3(m: &[[C64; 3]; 3]) -> [[C64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    assert!(
        det.norm() > crate::constants::SINGULAR_EPS,
        "matinv3: singular 3x3 material tensor (|det| = {:.3e})",
        det.norm()
    );
    let inv = C64::new(1.0, 0.0) / det;
    [
        [(m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
         (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
         (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv],
        [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
         (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
         (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv],
        [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
         (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
         (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv],
    ]
}

#[inline]
fn cross(a: &V3, b: &V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// (a · T · b) for real vectors a,b and a complex 3×3 tensor T.
#[inline]
fn vtv(a: &V3, t: &[[C64; 3]; 3], b: &V3) -> C64 {
    let mut s = C64::new(0.0, 0.0);
    for i in 0..3 {
        // (T·b)_i
        let tb = t[i][0] * b[0] + t[i][1] * b[1] + t[i][2] * b[2];
        s += tb * a[i];
    }
    s
}

/// Barycentric gradients ∇L_i and 6·Volume for a tet, from the standard
/// cofactor expansion of L_i = (a_i + b_i x + c_i y + d_i z)/(6V). The
/// (b_i, c_i, d_i) cofactors are signed minors of the vertex matrix.
pub fn barycentric_grads(xs: &[f64; 4], ys: &[f64; 4], zs: &[f64; 4]) -> ([V3; 4], f64) {
    let (x1, x2, x3, x4) = (xs[0], xs[1], xs[2], xs[3]);
    let (y1, y2, y3, y4) = (ys[0], ys[1], ys[2], ys[3]);
    let (z1, z2, z3, z4) = (zs[0], zs[1], zs[2], zs[3]);

    let six_v = -x1 * y2 * z3 + x1 * y2 * z4 + x1 * y3 * z2 - x1 * y3 * z4 - x1 * y4 * z2
        + x1 * y4 * z3 + x2 * y1 * z3 - x2 * y1 * z4 - x2 * y3 * z1 + x2 * y3 * z4
        + x2 * y4 * z1 - x2 * y4 * z3 - x3 * y1 * z2 + x3 * y1 * z4 + x3 * y2 * z1
        - x3 * y2 * z4 - x3 * y4 * z1 + x3 * y4 * z2 + x4 * y1 * z2 - x4 * y1 * z3
        - x4 * y2 * z1 + x4 * y2 * z3 + x4 * y3 * z1 - x4 * y3 * z2;

    // b_i, c_i, d_i cofactors (∇L_i = (b_i,c_i,d_i)/6V)
    let bbs = [
        -y2 * z3 + y2 * z4 + y3 * z2 - y3 * z4 - y4 * z2 + y4 * z3,
        y1 * z3 - y1 * z4 - y3 * z1 + y3 * z4 + y4 * z1 - y4 * z3,
        -y1 * z2 + y1 * z4 + y2 * z1 - y2 * z4 - y4 * z1 + y4 * z2,
        y1 * z2 - y1 * z3 - y2 * z1 + y2 * z3 + y3 * z1 - y3 * z2,
    ];
    let ccs = [
        x2 * z3 - x2 * z4 - x3 * z2 + x3 * z4 + x4 * z2 - x4 * z3,
        -x1 * z3 + x1 * z4 + x3 * z1 - x3 * z4 - x4 * z1 + x4 * z3,
        x1 * z2 - x1 * z4 - x2 * z1 + x2 * z4 + x4 * z1 - x4 * z2,
        -x1 * z2 + x1 * z3 + x2 * z1 - x2 * z3 - x3 * z1 + x3 * z2,
    ];
    let dds = [
        -x2 * y3 + x2 * y4 + x3 * y2 - x3 * y4 - x4 * y2 + x4 * y3,
        x1 * y3 - x1 * y4 - x3 * y1 + x3 * y4 + x4 * y1 - x4 * y3,
        -x1 * y2 + x1 * y4 + x2 * y1 - x2 * y4 - x4 * y1 + x4 * y2,
        x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2,
    ];
    // Sliver guard: a near-degenerate tet has 6V → 0 while edges stay O(1),
    // so ∇L = (b,c,d)/6V would blow up to ±∞/NaN and poison the whole global
    // factorization. Floor |6V| at q = SLIVER_NORMVOL_FLOOR of h_mean³ so one
    // bad tet stays locally wrong but bounded. (Diagnostics: `core::quality`.)
    let mut sum_len = 0.0;
    for &(a, b) in &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        let dx = xs[a] - xs[b];
        let dy = ys[a] - ys[b];
        let dz = zs[a] - zs[b];
        sum_len += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    let h_mean = sum_len / 6.0;
    let floor = crate::constants::SLIVER_NORMVOL_FLOOR * h_mean * h_mean * h_mean;
    let six_v_eff = if six_v.abs() < floor {
        floor.copysign(if six_v == 0.0 { 1.0 } else { six_v })
    } else {
        six_v
    };

    let inv = 1.0 / six_v_eff;
    let grads = std::array::from_fn(|i| [bbs[i] * inv, ccs[i] * inv, dds[i] * inv]);
    (grads, six_v_eff.abs())
}

/// One term of a basis function: `coeff · L_mono[0] · L_mono[1] · ∇L_grad`.
#[derive(Clone, Copy)]
pub struct Term {
    pub coeff: f64,
    pub mono: [usize; 2],
    pub grad: usize,
}

/// A basis function = `scale · Σ terms`. Every R2 function has exactly 2 terms.
/// This is the single source of truth for the canonical R2 basis, shared by
/// the element assembly here and the field reconstruction in `interp`.
pub struct BasisFn {
    pub scale: f64,
    pub terms: [Term; 2],
}

/// Build the 20 R2 basis functions for this tet from its local edge/face maps.
/// DOF order matches `basis::Nedelec2Basis`: edge·m1, face·m1, edge·m2, face·m2.
pub fn build_basis(
    edge_len: &[f64; 6],
    edge_map: &[[usize; 2]; 6],
    tri_map: &[[usize; 3]; 4],
    node_dist: &dyn Fn(usize, usize) -> f64,
) -> Vec<BasisFn> {
    let mut edge_m1 = Vec::with_capacity(6);
    let mut edge_m2 = Vec::with_capacity(6);
    for e in 0..6 {
        let (a, b) = (edge_map[e][0], edge_map[e][1]);
        let l = edge_len[e];
        // φ_e1 = ℓ L_a (L_a ∇L_b − L_b ∇L_a)
        edge_m1.push(BasisFn {
            scale: l,
            terms: [
                Term { coeff: 1.0, mono: [a, a], grad: b },
                Term { coeff: -1.0, mono: [a, b], grad: a },
            ],
        });
        // φ_e2 = ℓ L_b (L_a ∇L_b − L_b ∇L_a)
        edge_m2.push(BasisFn {
            scale: l,
            terms: [
                Term { coeff: 1.0, mono: [a, b], grad: b },
                Term { coeff: -1.0, mono: [b, b], grad: a },
            ],
        });
    }
    let mut face_m1 = Vec::with_capacity(4);
    let mut face_m2 = Vec::with_capacity(4);
    for f in 0..4 {
        let (n0, n1, n2) = (tri_map[f][0], tri_map[f][1], tri_map[f][2]);
        // φ_f1 = |n0 n2| L_n1 (L_n0 ∇L_n2 − L_n2 ∇L_n0)
        // (sign convention flipped to match the pipeline's face-mode-1 DOF)
        face_m1.push(BasisFn {
            scale: node_dist(n0, n2),
            terms: [
                Term { coeff: -1.0, mono: [n1, n0], grad: n2 },
                Term { coeff: 1.0, mono: [n1, n2], grad: n0 },
            ],
        });
        // φ_f2 = |n0 n1| L_n2 (L_n0 ∇L_n1 − L_n1 ∇L_n0)
        face_m2.push(BasisFn {
            scale: node_dist(n0, n1),
            terms: [
                Term { coeff: 1.0, mono: [n2, n0], grad: n1 },
                Term { coeff: -1.0, mono: [n2, n1], grad: n0 },
            ],
        });
    }
    // DOF order: edge·m1, face·m1, edge·m2, face·m2
    let mut basis = Vec::with_capacity(20);
    basis.extend(edge_m1);
    basis.extend(face_m1);
    basis.extend(edge_m2);
    basis.extend(face_m2);
    basis
}

/// ∫ L_p L_q L_r L_s dV with local node indices 0-3 (degree-4 monomial).
#[inline]
fn integ4(p: usize, q: usize, r: usize, s: usize, six_v: f64) -> f64 {
    // volume_coeff takes 1-based indices, 0 = unused; our nodes are 0-3.
    volume_coeff(p + 1, q + 1, r + 1, s + 1) * six_v
}

/// ∫ L_p L_q dV with local node indices 0-3 (degree-2 monomial).
#[inline]
fn integ2(p: usize, q: usize, six_v: f64) -> f64 {
    volume_coeff(p + 1, q + 1, 0, 0) * six_v
}

/// Per-tet 20×20 stiffness (`D`) and mass (`F`) matrices for the R2 element.
///
/// `ms` is μ⁻¹ and `mm` is ε (per-tet constant tensors), matching the
/// assembler's convention.
pub fn r2_tet_stiff_mass(
    xs: &[f64; 4],
    ys: &[f64; 4],
    zs: &[f64; 4],
    edge_lengths: &[f64; 6],
    local_edge_map: &[[usize; 2]; 6],
    local_tri_map: &[[usize; 3]; 4],
    ms: &[[C64; 3]; 3], // μ⁻¹
    mm: &[[C64; 3]; 3], // ε
) -> ([[C64; 20]; 20], [[C64; 20]; 20]) {
    let (grads, six_v) = barycentric_grads(xs, ys, zs);
    let node_dist = |i: usize, j: usize| -> f64 {
        ((xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2) + (zs[i] - zs[j]).powi(2)).sqrt()
    };
    let basis = build_basis(edge_lengths, local_edge_map, local_tri_map, &node_dist);

    let zero = C64::new(0.0, 0.0);
    let mut d = [[zero; 20]; 20];
    let mut f = [[zero; 20]; 20];

    for i in 0..20 {
        for j in i..20 {
            let bi = &basis[i];
            let bj = &basis[j];
            let sc = bi.scale * bj.scale;

            // --- mass: φ_i · ε · φ_j ---
            let mut fij = zero;
            for ti in &bi.terms {
                for tj in &bj.terms {
                    let coeff = ti.coeff * tj.coeff;
                    let quad = vtv(&grads[ti.grad], mm, &grads[tj.grad]);
                    let intg = integ4(ti.mono[0], ti.mono[1], tj.mono[0], tj.mono[1], six_v);
                    fij += quad * (coeff * intg);
                }
            }
            fij *= C64::new(sc, 0.0);

            // --- stiffness: (∇×φ_i) · μ⁻¹ · (∇×φ_j) ---
            // curl(L_p L_q ∇L_g) = L_q (∇L_p×∇L_g) + L_p (∇L_q×∇L_g)
            let mut dij = zero;
            for ti in &bi.terms {
                let curls_i = [
                    (ti.mono[1], cross(&grads[ti.mono[0]], &grads[ti.grad])),
                    (ti.mono[0], cross(&grads[ti.mono[1]], &grads[ti.grad])),
                ];
                for tj in &bj.terms {
                    let curls_j = [
                        (tj.mono[1], cross(&grads[tj.mono[0]], &grads[tj.grad])),
                        (tj.mono[0], cross(&grads[tj.mono[1]], &grads[tj.grad])),
                    ];
                    let coeff = ti.coeff * tj.coeff;
                    for (mi, ci) in &curls_i {
                        for (mj, cj) in &curls_j {
                            let quad = vtv(ci, ms, cj);
                            let intg = integ2(*mi, *mj, six_v);
                            dij += quad * (coeff * intg);
                        }
                    }
                }
            }
            dij *= C64::new(sc, 0.0);

            d[i][j] = dij;
            d[j][i] = dij;
            f[i][j] = fij;
            f[j][i] = fij;
        }
    }
    (d, f)
}

/// Assemble global stiffness (E) and mass (B) COO triplets from all tets using
/// the canonical R2 element. Drop-in replacement for the legacy
/// `tet_assembly::assemble_global_matrices` (same signature and DOF mapping);
/// `ur` is permeability (inverted per tet to μ⁻¹), `er` is permittivity.
pub fn assemble_global_matrices(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    er: &[[[C64; 3]; 3]],
    ur: &[[[C64; 3]; 3]],
) -> (Vec<usize>, Vec<usize>, Vec<C64>, Vec<C64>) {
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    let n_tets = mesh.n_tets();
    let nnz = n_tets * 400;
    let mut rows = vec![0usize; nnz];
    let mut cols = vec![0usize; nnz];
    let mut data_e = vec![C64::new(0.0, 0.0); nnz];
    let mut data_b = vec![C64::new(0.0, 0.0); nnz];

    let chunks: Vec<(usize, &mut [usize], &mut [usize], &mut [C64], &mut [C64])> = {
        let rc: Vec<&mut [usize]> = rows.chunks_mut(400).collect();
        let cc: Vec<&mut [usize]> = cols.chunks_mut(400).collect();
        let de: Vec<&mut [C64]> = data_e.chunks_mut(400).collect();
        let db: Vec<&mut [C64]> = data_b.chunks_mut(400).collect();
        (0..n_tets).zip(rc).zip(cc).zip(de).zip(db)
            .map(|((((i, r), c), e), b)| (i, r, c, e, b))
            .collect()
    };

    #[cfg(feature = "parallel")]
    let it = chunks.into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let it = chunks.into_iter();

    it.for_each(|(itet, row_slice, col_slice, de_slice, db_slice)| {
        let tet = &mesh.tets[itet];
        let xs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][0]);
        let ys: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][1]);
        let zs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][2]);

        let tet_edges = &mesh.tet_to_edge[itet];
        let edge_lengths: [f64; 6] = std::array::from_fn(|i| mesh.edge_lengths[tet_edges[i]]);
        let global_edge_nodes: [[usize; 2]; 6] = std::array::from_fn(|i| mesh.edges[tet_edges[i]]);
        let local_edge_map = crate::basis::local_mapping(tet, &global_edge_nodes);

        let tet_tris = &mesh.tet_to_tri[itet];
        let global_tri_nodes: [[usize; 3]; 4] = std::array::from_fn(|i| mesh.tris[tet_tris[i]]);
        let local_tri_map = crate::basis::local_mapping_tri(tet, &global_tri_nodes);

        let ms = matinv3(&ur[itet]);
        let mm = &er[itet];

        let (esub, bsub) = r2_tet_stiff_mass(
            &xs, &ys, &zs, &edge_lengths, &local_edge_map, &local_tri_map, &ms, mm,
        );

        let indices = &basis.tet_to_field[itet];
        for ii in 0..20 {
            for jj in 0..20 {
                let idx = ii * 20 + jj;
                row_slice[idx] = indices[ii];
                col_slice[idx] = indices[jj];
                de_slice[idx] = esub[ii][jj];
                db_slice[idx] = bsub[ii][jj];
            }
        }
    });

    (rows, cols, data_e, data_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident() -> [[C64; 3]; 3] {
        [[C64::new(1.0,0.0),C64::new(0.0,0.0),C64::new(0.0,0.0)],
         [C64::new(0.0,0.0),C64::new(1.0,0.0),C64::new(0.0,0.0)],
         [C64::new(0.0,0.0),C64::new(0.0,0.0),C64::new(1.0,0.0)]]
    }

    /// A fully coplanar (degenerate) tet must not produce inf/NaN gradients —
    /// the volume floor keeps 1/6V finite.
    #[test]
    fn degenerate_tet_grads_are_finite() {
        let xs = [0.0, 1.0, 0.0, 0.5];
        let ys = [0.0, 0.0, 1.0, 0.5];
        let zs = [0.0, 0.0, 0.0, 0.0]; // 4th node coplanar -> 6V = 0
        let (grads, six_v) = barycentric_grads(&xs, &ys, &zs);
        assert!(six_v.is_finite() && six_v > 0.0, "floored 6V must be positive finite");
        for g in &grads {
            for &c in g { assert!(c.is_finite(), "gradient component must be finite"); }
        }
    }

    /// A sliver tet must yield finite element matrices (no NaN poisoning).
    #[test]
    fn sliver_element_matrices_are_finite() {
        let xs = [0.0, 1.0, 0.0, 0.333];
        let ys = [0.0, 0.0, 1.0, 0.333];
        let zs = [0.0, 0.0, 0.0, 1e-12]; // near-flat sliver
        let s2 = 2.0_f64.sqrt();
        let el = [1.0, 1.0, 1.0, s2, s2, s2];
        let em = [[0,1],[0,2],[0,3],[1,2],[3,1],[2,3]];
        let tm = [[0,1,2],[0,2,3],[0,3,1],[1,2,3]];
        let (d, f) = r2_tet_stiff_mass(&xs,&ys,&zs,&el,&em,&tm,&ident(),&ident());
        for i in 0..20 { for j in 0..20 {
            assert!(d[i][j].re.is_finite() && d[i][j].im.is_finite(), "D finite");
            assert!(f[i][j].re.is_finite() && f[i][j].im.is_finite(), "F finite");
        }}
    }
}
