// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! The curl-conforming tetrahedral element: Nédélec first kind, orders 1 and 2.
//!
//! The space is the canonical R2 = (P1)³ ⊕ {p ∈ H̃2³ : x·p = 0}, 20-dimensional on a
//! tetrahedron, with the lowest-order (Whitney/Nédélec-0) space, 6-dimensional,
//! nested inside it. All of it is built from the Whitney function
//! W_ab = L_a ∇L_b − L_b ∇L_a. Two bases of that same space are available; see
//! [`BasisKind`] for which and why.
//!
//!   edge (a,b), length ℓ    interpolatory: ℓ·L_a·W_ab and ℓ·L_b·W_ab
//!                           hierarchical:  ℓ·W_ab      and ℓ·∇(L_a·L_b)
//!   face (n0,n1,n2)         φ_f0 = |n0 n2| · L_n1 (L_n2 ∇L_n0 − L_n0 ∇L_n2)
//!                           φ_f1 = |n0 n1| · L_n2 (L_n0 ∇L_n1 − L_n1 ∇L_n0)
//!
//! The derivation, completeness proof and spectral identification live in
//! `derivations/nedelec2/` (element.py, canonical_r2.py); the proof that the two
//! bases span one space, and that only the hierarchical one nests, is in
//! hierarchical.py. Element matrices:
//!
//!   stiffness  D_ij = ∫ (∇×φ_i)·μ⁻¹·(∇×φ_j) dV
//!   mass       F_ij = ∫  φ_i·ε·φ_j           dV
//!
//! Every basis function is a sum of terms `coeff · L^e · ∇L_g` over a barycentric
//! exponent multi-index. Products of barycentric coordinates integrate in closed
//! form (see `coefficients`), so the element is assembled exactly, with no
//! quadrature.
//!
//! **The element does not decide how many DOFs it has.** `build_basis` takes the
//! owner list that `basis::tet_dof_owners` produced from the entity orders, and
//! builds one function per entry. Under a mixed order that list is shorter than 20,
//! and nothing here notices or cares.

use num_complex::Complex64 as C64;
use crate::coefficients::volume_coeff_exps;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::dofmap::DofOwner;

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

/// One term of a basis function:
///
///   `coeff · L_1^e1 L_2^e2 L_3^e3 L_4^e4 · ∇L_grad`
///
/// The exponent multi-index is the general form (it is what
/// `derivations/nedelec2/element.py` uses), so this type carries any polynomial
/// H(curl) basis, not only R2. Order 2 happens to produce degree-2 monomials;
/// nothing here assumes that.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Term {
    pub coeff: f64,
    pub exps: [u8; 4],
    pub grad: u8,
}

impl Term {
    /// A degree-2 term `coeff · L_p L_q · ∇L_g` from local node indices 0-3.
    #[inline]
    pub fn quad(coeff: f64, p: usize, q: usize, g: usize) -> Term {
        let mut exps = [0u8; 4];
        exps[p] += 1;
        exps[q] += 1;
        Term { coeff, exps, grad: g as u8 }
    }

    /// A degree-1 term `coeff · L_p · ∇L_g`. The hierarchical basis needs it: its
    /// Whitney and gradient functions are degree 1, not degree 2.
    #[inline]
    pub fn lin(coeff: f64, p: usize, g: usize) -> Term {
        let mut exps = [0u8; 4];
        exps[p] += 1;
        Term { coeff, exps, grad: g as u8 }
    }
}

/// A basis function = `scale · Σ terms`.
///
/// The term count is not fixed: an R2 function has two, a higher-order or
/// hierarchical one may have more. `TERMS_INLINE` is sized so that R2 needs no
/// heap allocation; a longer function spills to the heap without any other code
/// noticing.
pub const TERMS_INLINE: usize = 4;

#[derive(Clone, Debug)]
pub struct BasisFn {
    pub scale: f64,
    pub terms: Vec<Term>,
}

impl BasisFn {
    #[inline]
    pub fn new(scale: f64, terms: Vec<Term>) -> BasisFn {
        BasisFn { scale, terms }
    }
}

/// Which basis of the R2 space to use.
///
/// Both span the same 20-dimensional space — that is proved, not assumed, in
/// `derivations/nedelec2/hierarchical.py` and checked in Rust by
/// `tests/hierarchical_basis_test.rs`. They differ in which functions carry the
/// DOFs, and that difference is what stages 4-5 of `docs/fd-basis-plan.md` need.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BasisKind {
    /// `ℓ·L_a·W_ab` and `ℓ·L_b·W_ab`. Both edge functions are degree 2, so the
    /// lowest-order (Whitney) space is **not** spanned by any subset of the DOFs:
    /// there is no way to run a cell at order 1 by dropping DOFs. This is the
    /// element the goldens were generated from and the default.
    #[default]
    Interpolatory,
    /// `ℓ·W_ab` and `ℓ·∇(L_a·L_b)`. Mode 0 *is* the Whitney space, so order 1 is a
    /// coordinate subspace and the hierarchy nests. Mode 1 is a pure gradient,
    /// hence curl-free, which makes the kernel of the curl operator explicit —
    /// the local exact-sequence property (Schöberl & Zaglmayr, COMPEL 24 (2005)
    /// 374). Better conditioned, and the only basis a p-decay indicator can be
    /// read off.
    Hierarchical,
}

/// The two R2 functions of the edge (a, b) of length ℓ, in mode order.
///
/// Interpolatory:  φ_e0 = ℓ·L_a·W_ab,   φ_e1 = ℓ·L_b·W_ab
/// Hierarchical:   φ_e0 = ℓ·W_ab,       φ_e1 = ℓ·∇(L_a L_b)
///
/// with the Whitney function W_ab = L_a ∇L_b − L_b ∇L_a and, expanded,
/// ∇(L_a L_b) = L_a ∇L_b + L_b ∇L_a. Note the hierarchical pair is degree 1 and
/// the interpolatory pair degree 2: **per edge they span different 2-D spaces.**
/// Only the total 20-dimensional spaces coincide, once the face functions are
/// included. The derivation proves exactly that.
///
/// `a` and `b` are node indices in whatever local numbering the caller uses, so
/// this serves the tetrahedron and the triangle alike.
///
/// This, and `r2_face_fns`, are the ONLY places the R2 functions are written
/// down. The surface element is the tangential trace of this one and is built by
/// calling the same two functions, so the two cannot drift apart in sign — which
/// they previously could, being written out twice.
pub fn r2_edge_fns(kind: BasisKind, a: usize, b: usize, len: f64) -> [BasisFn; 2] {
    match kind {
        BasisKind::Interpolatory => [
            BasisFn::new(len, vec![Term::quad(1.0, a, a, b), Term::quad(-1.0, a, b, a)]),
            BasisFn::new(len, vec![Term::quad(1.0, a, b, b), Term::quad(-1.0, b, b, a)]),
        ],
        BasisKind::Hierarchical => [
            BasisFn::new(len, vec![Term::lin(1.0, a, b), Term::lin(-1.0, b, a)]),
            BasisFn::new(len, vec![Term::lin(1.0, a, b), Term::lin(1.0, b, a)]),
        ],
    }
}

/// The two R2 functions of the face (n0, n1, n2), in mode order:
///
///   φ_f1 = |n0 n2| · L_n1 (L_n2 ∇L_n0 − L_n0 ∇L_n2)
///   φ_f2 = |n0 n1| · L_n2 (L_n0 ∇L_n1 − L_n1 ∇L_n0)
///
/// `d02` and `d01` are the distances |n0 n2| and |n0 n1|. Both functions have a
/// vanishing tangential trace on the tetrahedron's other three faces (proved in
/// `derivations/nedelec2/`), which is what makes them face-interior DOFs.
pub fn r2_face_fns(n0: usize, n1: usize, n2: usize, d02: f64, d01: f64) -> [BasisFn; 2] {
    [
        BasisFn::new(d02, vec![Term::quad(-1.0, n1, n0, n2), Term::quad(1.0, n1, n2, n0)]),
        BasisFn::new(d01, vec![Term::quad(1.0, n2, n0, n1), Term::quad(-1.0, n2, n1, n0)]),
    ]
}

/// Build one basis function per entry of `owners`.
///
/// The element does not enumerate its own DOFs. The owner list — produced by
/// `basis::tet_dof_owners` from the entity orders — says which entity each local
/// DOF belongs to and which of that entity's functions it is, and this reads that
/// list off. So a DOF exists in the element exactly when it exists in the DOF map,
/// in the same position, whatever the orders happen to be. There is no second
/// enumeration that could disagree.
pub fn build_basis(
    kind: BasisKind,
    owners: &[DofOwner],
    edge_len: &[f64; 6],
    edge_map: &[[usize; 2]; 6],
    tri_map: &[[usize; 3]; 4],
    node_dist: &dyn Fn(usize, usize) -> f64,
) -> Vec<BasisFn> {
    let edges: Vec<[BasisFn; 2]> = (0..6)
        .map(|e| r2_edge_fns(kind, edge_map[e][0], edge_map[e][1], edge_len[e]))
        .collect();
    let faces: Vec<[BasisFn; 2]> = (0..4)
        .map(|f| {
            let (n0, n1, n2) = (tri_map[f][0], tri_map[f][1], tri_map[f][2]);
            r2_face_fns(n0, n1, n2, node_dist(n0, n2), node_dist(n0, n1))
        })
        .collect();

    owners
        .iter()
        .map(|o| match *o {
            DofOwner::Edge { entity, k } => edges[entity as usize][k as usize].clone(),
            DofOwner::Face { entity, k } => faces[entity as usize][k as usize].clone(),
            DofOwner::Cell { .. } => {
                unreachable!("no cell-interior DOFs below order 3")
            }
        })
        .collect()
}

/// `∫ L^(ea + eb) dV`, the mass integrand of two terms. Exponents add.
#[inline]
fn integ_mass(ea: [u8; 4], eb: [u8; 4], six_v: f64) -> f64 {
    let e = [ea[0] + eb[0], ea[1] + eb[1], ea[2] + eb[2], ea[3] + eb[3]];
    volume_coeff_exps(e) * six_v
}

/// `∫ L^(ea + eb) dV` for two CURL terms, whose exponents are one degree lower
/// than the functions they came from. Same closed form; the separate name keeps
/// the two call sites readable.
#[inline]
fn integ_stiff(ea: [u8; 4], eb: [u8; 4], six_v: f64) -> f64 {
    integ_mass(ea, eb, six_v)
}

/// The curl of one term, as a list of terms with a CONSTANT vector instead of a
/// `∇L`:
///
///   ∇×(L^e · ∇L_g) = Σ_k e_k · L^(e − 1_k) · (∇L_k × ∇L_g)
///
/// One input term produces at most four output terms (one per nonzero exponent).
/// This is `element.py::curl_field`, ported.
#[inline]
fn curl_term(t: &Term, grads: &[V3; 4], out: &mut Vec<(f64, [u8; 4], V3)>) {
    for k in 0..4 {
        let ek = t.exps[k];
        if ek == 0 {
            continue;
        }
        let mut e = t.exps;
        e[k] -= 1;
        out.push((
            t.coeff * ek as f64,
            e,
            cross(&grads[k], &grads[t.grad as usize]),
        ));
    }
}

/// Element stiffness (`D`) and mass (`F`), row-major `n×n`, for ANY basis given
/// as a term list. This is the whole `O(n²)` cost of the element assembly, and
/// it does not know which element it is integrating:
///
///   D_ij = ∫ (∇×φ_i) · μ⁻¹ · (∇×φ_j) dV
///   F_ij = ∫  φ_i     ·  ε   ·  φ_j    dV
///
/// Every term is `c · L^e · ∇L_g`, so both integrands are barycentric monomials
/// times a constant tensor contraction and integrate exactly by the closed form
/// in `coefficients` (no quadrature). `ms` is μ⁻¹, `mm` is ε.
pub fn element_stiff_mass(
    basis: &[BasisFn],
    grads: &[V3; 4],
    six_v: f64,
    ms: &[[C64; 3]; 3], // μ⁻¹
    mm: &[[C64; 3]; 3], // ε
) -> (Vec<C64>, Vec<C64>) {
    let n = basis.len();
    let zero = C64::new(0.0, 0.0);
    let mut d = vec![zero; n * n];
    let mut f = vec![zero; n * n];

    // Curls are needed once per function, not once per (i, j) pair.
    let mut curls: Vec<Vec<(f64, [u8; 4], V3)>> = Vec::with_capacity(n);
    for b in basis {
        let mut c = Vec::with_capacity(b.terms.len() * 4);
        for t in &b.terms {
            curl_term(t, grads, &mut c);
        }
        curls.push(c);
    }

    for i in 0..n {
        for j in i..n {
            let (bi, bj) = (&basis[i], &basis[j]);
            let sc = bi.scale * bj.scale;

            // --- mass: φ_i · ε · φ_j ---
            let mut fij = zero;
            for ti in &bi.terms {
                for tj in &bj.terms {
                    let coeff = ti.coeff * tj.coeff;
                    let quad = vtv(&grads[ti.grad as usize], mm, &grads[tj.grad as usize]);
                    let intg = integ_mass(ti.exps, tj.exps, six_v);
                    fij += quad * (coeff * intg);
                }
            }
            fij *= C64::new(sc, 0.0);

            // --- stiffness: (∇×φ_i) · μ⁻¹ · (∇×φ_j) ---
            let mut dij = zero;
            for (ci, ei, vi) in &curls[i] {
                for (cj, ej, vj) in &curls[j] {
                    let quad = vtv(vi, ms, vj);
                    let intg = integ_stiff(*ei, *ej, six_v);
                    dij += quad * (ci * cj * intg);
                }
            }
            dij *= C64::new(sc, 0.0);

            d[i * n + j] = dij;
            d[j * n + i] = dij;
            f[i * n + j] = fij;
            f[j * n + i] = fij;
        }
    }
    (d, f)
}

/// `∫ φ_i · ψ_j dV` for two bases on the same tetrahedron. Row-major `|a| × |b|`.
///
/// The cross-Gram matrix of two bases is what decides whether they span the same
/// space: `ψ_j` lies in `span(φ)` exactly when the L² projection of `ψ_j` onto
/// `span(φ)` has zero residual, and both the projection and the residual are built
/// from this matrix and the two Gram matrices. `tests/hierarchical_basis_test.rs`
/// uses it to prove the interpolatory and hierarchical bases span one space —
/// an if-and-only-if, not a spot check.
pub fn cross_mass(a: &[BasisFn], b: &[BasisFn], grads: &[V3; 4], six_v: f64) -> Vec<f64> {
    let mut m = vec![0.0_f64; a.len() * b.len()];
    for (i, fi) in a.iter().enumerate() {
        for (j, fj) in b.iter().enumerate() {
            let mut acc = 0.0;
            for ti in &fi.terms {
                for tj in &fj.terms {
                    let g = grads[ti.grad as usize]
                        .iter()
                        .zip(grads[tj.grad as usize].iter())
                        .map(|(x, y)| x * y)
                        .sum::<f64>();
                    acc += ti.coeff * tj.coeff * g * integ_mass(ti.exps, tj.exps, six_v);
                }
            }
            m[i * b.len() + j] = fi.scale * fj.scale * acc;
        }
    }
    m
}

/// Per-tet stiffness and mass for the R2 element: build the basis, then hand it
/// to the basis-agnostic `element_stiff_mass`. Returns row-major `20×20`.
pub fn r2_tet_stiff_mass(
    kind: BasisKind,
    owners: &[DofOwner],
    xs: &[f64; 4],
    ys: &[f64; 4],
    zs: &[f64; 4],
    edge_lengths: &[f64; 6],
    local_edge_map: &[[usize; 2]; 6],
    local_tri_map: &[[usize; 3]; 4],
    ms: &[[C64; 3]; 3], // μ⁻¹
    mm: &[[C64; 3]; 3], // ε
) -> (Vec<C64>, Vec<C64>) {
    let (grads, six_v) = barycentric_grads(xs, ys, zs);
    let node_dist = |i: usize, j: usize| -> f64 {
        ((xs[i] - xs[j]).powi(2) + (ys[i] - ys[j]).powi(2) + (zs[i] - zs[j]).powi(2)).sqrt()
    };
    let basis = build_basis(kind, owners, edge_lengths, local_edge_map, local_tri_map, &node_dist);
    element_stiff_mass(&basis, &grads, six_v, ms, mm)
}

/// Cut `s` into consecutive mutable blocks of the widths given by the prefix-sum
/// table `off` (`off[i]..off[i+1]` is block `i`). The variable-width counterpart
/// of `chunks_mut`: each element writes only into its own block, so the scatter
/// stays lock-free even when the elements have different DOF counts.
fn ragged_chunks_mut<'a, T>(s: &'a mut [T], off: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(off.len().saturating_sub(1));
    let mut rest = s;
    for w in off.windows(2) {
        let (head, tail) = rest.split_at_mut(w[1] - w[0]);
        out.push(head);
        rest = tail;
    }
    out
}

/// Assemble global stiffness (E) and mass (B) COO triplets from all tets using
/// the canonical R2 element. `ur` is permeability (inverted per tet to μ⁻¹), `er`
/// is permittivity.
///
/// The triplet layout comes from the basis, not from a constant: element `i` owns
/// `basis.tet_nnz_offsets()[i] .. [i+1]`, which is n_i² wide. For a uniform R2
/// space that is 400 for every element; for a mixed-order space it is not.
pub fn assemble_global_matrices(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    er: &[[[C64; 3]; 3]],
    ur: &[[[C64; 3]; 3]],
) -> (Vec<usize>, Vec<usize>, Vec<C64>, Vec<C64>) {
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    let n_tets = mesh.n_tets();
    let off = basis.tet_nnz_offsets();
    let nnz = basis.n_tet_nnz();
    let mut rows = vec![0usize; nnz];
    let mut cols = vec![0usize; nnz];
    let mut data_e = vec![C64::new(0.0, 0.0); nnz];
    let mut data_b = vec![C64::new(0.0, 0.0); nnz];

    let chunks: Vec<(usize, &mut [usize], &mut [usize], &mut [C64], &mut [C64])> = {
        let rc = ragged_chunks_mut(&mut rows, off);
        let cc = ragged_chunks_mut(&mut cols, off);
        let de = ragged_chunks_mut(&mut data_e, off);
        let db = ragged_chunks_mut(&mut data_b, off);
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

        let owners = crate::basis::tet_dof_owners(
            &basis.orders.tet_edge_orders(mesh, itet),
            &basis.orders.tet_face_orders(mesh, itet),
        );
        let (esub, bsub) = r2_tet_stiff_mass(
            basis.kind, &owners, &xs, &ys, &zs, &edge_lengths, &local_edge_map, &local_tri_map,
            &ms, mm,
        );

        // The element matrices are row-major n×n, and the block reserved for this
        // element is n² wide. Neither side knows what n is.
        let indices = basis.tet_dofs(itet);
        let n = indices.len();
        debug_assert_eq!(esub.len(), n * n);
        debug_assert_eq!(row_slice.len(), n * n);
        for ii in 0..n {
            for jj in 0..n {
                let idx = ii * n + jj;
                row_slice[idx] = indices[ii];
                col_slice[idx] = indices[jj];
                de_slice[idx] = esub[idx];
                db_slice[idx] = bsub[idx];
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
        let owners = crate::basis::tet_dof_owners(&[2;6], &[2;4]);
        let (d, f) = r2_tet_stiff_mass(BasisKind::Interpolatory,&owners,&xs,&ys,&zs,&el,&em,&tm,&ident(),&ident());
        assert_eq!(d.len(), 400);
        for (dv, fv) in d.iter().zip(f.iter()) {
            assert!(dv.re.is_finite() && dv.im.is_finite(), "D finite");
            assert!(fv.re.is_finite() && fv.im.is_finite(), "F finite");
        }
    }

    /// The curl of a term must be what the exponent rule says it is:
    /// ∇×(L^e ∇L_g) = Σ_k e_k · L^(e−1_k) · (∇L_k × ∇L_g).
    ///
    /// The old code special-cased a degree-2 monomial as `L_p L_q` and emitted
    /// two curl terms even when p == q. The general rule emits ONE term with a
    /// doubled coefficient instead. Both are the same field; this pins that.
    #[test]
    fn curl_of_a_repeated_exponent_term_is_the_doubled_single_term() {
        let xs = [0.0, 1.0, 0.0, 0.0];
        let ys = [0.0, 0.0, 1.0, 0.0];
        let zs = [0.0, 0.0, 0.0, 1.0];
        let (grads, _) = barycentric_grads(&xs, &ys, &zs);

        // t = 1.0 · L_0^2 · ∇L_1   (this is exactly the p == q case)
        let t = Term::quad(1.0, 0, 0, 1);
        assert_eq!(t.exps, [2, 0, 0, 0]);

        let mut out = Vec::new();
        curl_term(&t, &grads, &mut out);
        assert_eq!(out.len(), 1, "one nonzero exponent -> one curl term");

        let (coeff, exps, vec) = out[0];
        assert_eq!(coeff, 2.0, "e_0 = 2 -> coefficient 2");
        assert_eq!(exps, [1, 0, 0, 0], "L_0^2 -> L_0");
        let want = cross(&grads[0], &grads[1]);
        for k in 0..3 {
            assert!((vec[k] - want[k]).abs() < 1e-14);
        }
    }
}
