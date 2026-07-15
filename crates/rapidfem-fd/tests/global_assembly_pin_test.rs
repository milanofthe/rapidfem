// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.
//
// Bit-identity pin for the GLOBAL frequency-domain assembly.
//
// The element goldens (`r2_element_golden_test`, `tri_mass_golden_test`,
// `interp_golden_test`) pin the per-element kernels entrywise. Nothing pinned
// what the assembly does WITH them: the DOF map, the scatter into the COO
// triplets, and the global ordering. This test does.
//
// It exists to make the element-basis refactor (docs/fd-basis-plan.md) safe:
// stages 0-2 of that plan restructure the term representation, the DOF map and
// the surface element with ZERO behaviour change, and this is the oracle that
// says so. The numbers below were captured from the interpolatory R2 element
// before any of it was touched.
//
// If a change to the element or the assembly moves these numbers, that change
// is not a refactor.

use num_complex::Complex64 as C64;
use rapidfem_fd::basis::NedelecBasis;
use rapidfem_fd::mesh::Mesh;
use rapidfem_fd::tet_assembly::assemble_global_matrices;

/// A unit cube cut into six tetrahedra (the standard Kuhn triangulation). Small
/// enough to reason about, big enough that every entity kind is shared: it has
/// interior edges and interior faces, so the DOF map is genuinely exercised.
fn cube_6tet() -> Mesh {
    let nodes = vec![
        [0.0, 0.0, 0.0], // 0
        [1.0, 0.0, 0.0], // 1
        [1.0, 1.0, 0.0], // 2
        [0.0, 1.0, 0.0], // 3
        [0.0, 0.0, 1.0], // 4
        [1.0, 0.0, 1.0], // 5
        [1.0, 1.0, 1.0], // 6
        [0.0, 1.0, 1.0], // 7
    ];
    // Kuhn / Freudenthal: six tets sharing the main diagonal 0-6.
    let tets = vec![
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ];
    Mesh::from_tets(nodes, tets)
}

fn identity() -> [[C64; 3]; 3] {
    let z = C64::new(0.0, 0.0);
    let o = C64::new(1.0, 0.0);
    [[o, z, z], [z, o, z], [z, z, o]]
}

/// A material tensor with distinct, non-symmetric-in-index entries, so an error
/// that transposes or mis-contracts the tensor cannot hide behind the identity.
fn skew_tensor() -> [[C64; 3]; 3] {
    [
        [C64::new(2.5, -0.3), C64::new(0.4, 0.1), C64::new(-0.2, 0.05)],
        [C64::new(0.4, 0.1), C64::new(1.7, -0.9), C64::new(0.3, -0.2)],
        [C64::new(-0.2, 0.05), C64::new(0.3, -0.2), C64::new(3.1, 0.6)],
    ]
}

/// Sum of |value| over the triplets, and a pattern checksum that is sensitive to
/// the ORDER in which the triplets are emitted (so a reordering of the scatter
/// is caught, not just a change of values).
fn summarise(rows: &[usize], cols: &[usize], vals: &[C64]) -> (f64, f64, u64) {
    let mut abs_sum = 0.0_f64;
    let mut sq_sum = 0.0_f64;
    for v in vals {
        abs_sum += v.norm();
        sq_sum += v.norm_sqr();
    }
    // FNV-1a over the (index, row, col) stream: order-sensitive by construction.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, (&r, &c)) in rows.iter().zip(cols.iter()).enumerate() {
        for b in i.to_le_bytes().iter().chain(r.to_le_bytes().iter()).chain(c.to_le_bytes().iter())
        {
            h ^= *b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    (abs_sum, sq_sum.sqrt(), h)
}

/// The generalised term representation regroups the curl of a repeated-exponent
/// term (`L_a^2 ∇L_g`) from two identical contributions into one with a doubled
/// coefficient. That is a change of summation order, so exact equality is not
/// guaranteed in principle. It does in fact hold here (the regrouping introduces
/// only powers of two, which scale exactly), and it was checked with `rel == 0`.
/// The permanent gate is a tight relative bound rather than bitwise equality,
/// because FMA contraction and instruction selection are not portable.
fn close(got: f64, want: f64, what: &str) {
    let rel = (got - want).abs() / want.abs().max(1e-300);
    assert!(
        rel < 1e-14,
        "{what}: got {got:.17e}, want {want:.17e}, rel {rel:.2e}"
    );
}

#[test]
fn global_dof_map_is_pinned() {
    let mesh = cube_6tet();
    let basis = NedelecBasis::new(&mesh);

    assert_eq!(mesh.n_tets(), 6, "tets");
    assert_eq!(mesh.n_edges(), 19, "edges");
    assert_eq!(mesh.n_tris(), 18, "faces");
    // n_field = 2*edges + 2*faces
    assert_eq!(basis.n_field, 2 * 19 + 2 * 18, "n_field");
    assert_eq!(basis.tet_dofs(0).len(), 20);
    assert_eq!(basis.tri_dofs(0).len(), 8);

    // Every global DOF must be reachable from some tetrahedron, and no DOF may
    // be claimed outside the range. An offset bug shows up here immediately.
    let mut seen = vec![false; basis.n_field];
    for t in 0..mesh.n_tets() {
        for &d in basis.tet_dofs(t) {
            assert!(d < basis.n_field, "DOF {d} out of range");
            seen[d] = true;
        }
    }
    assert!(seen.iter().all(|&s| s), "some global DOF is on no tetrahedron");
}

/// The entity-major layout (stage 1) replaced a mode-major one. A relabelling of
/// the unknowns is not a change of the discretisation: the assembled system must
/// be the *same* matrix, permuted. This proves it, rather than asserting it.
///
/// The previous layout, in closed form:
///   edge e, mode m -> e + m·(n_edges + n_tris)
///   face f, mode m -> f + n_edges + m·(n_edges + n_tris)
/// which is exactly what `NedelecBasis::new` used to compute.
#[test]
fn numbering_is_a_relabelling_of_the_mode_major_layout() {
    let mesh = cube_6tet();
    let basis = NedelecBasis::new(&mesh);
    let (ne, nt) = (mesh.n_edges(), mesh.n_tris());
    let stride = ne + nt;

    // perm[old] = new, built from the two maps agreeing on every local DOF.
    let mut perm = vec![usize::MAX; basis.n_field];
    for ti in 0..mesh.n_tets() {
        let edges = &mesh.tet_to_edge[ti];
        let faces = &mesh.tet_to_tri[ti];
        let old: Vec<usize> = (0..6)
            .map(|i| edges[i])
            .chain((0..4).map(|i| faces[i] + ne))
            .chain((0..6).map(|i| edges[i] + stride))
            .chain((0..4).map(|i| faces[i] + ne + stride))
            .collect();
        let new = basis.tet_dofs(ti);
        assert_eq!(old.len(), new.len());
        for (&o, &n) in old.iter().zip(new) {
            assert!(
                perm[o] == usize::MAX || perm[o] == n,
                "DOF {o} maps to both {} and {n}: not a function",
                perm[o]
            );
            perm[o] = n;
        }
    }
    assert!(perm.iter().all(|&p| p != usize::MAX), "the old layout has an unmapped DOF");

    let mut hit = vec![false; basis.n_field];
    for &p in &perm {
        assert!(!hit[p], "two old DOFs map to new DOF {p}: not injective");
        hit[p] = true;
    }
    // Bijective on a finite set of equal size: the two layouts are a permutation
    // of one another, so the assembled systems are P·K·Pᵀ. The abs-sum and the
    // Frobenius norm pinned below are invariant under that, which is why they did
    // not move when the layout did.
}

#[test]
fn global_assembly_is_pinned_identity_material() {
    let mesh = cube_6tet();
    let basis = NedelecBasis::new(&mesh);
    let er = vec![identity(); mesh.n_tets()];
    let ur = vec![identity(); mesh.n_tets()];

    let (rows, cols, de, db) = assemble_global_matrices(&mesh, &basis, &er, &ur);
    assert_eq!(rows.len(), 6 * 400, "nnz triplets");

    let (e_abs, e_fro, e_hash) = summarise(&rows, &cols, &de);
    let (b_abs, b_fro, b_hash) = summarise(&rows, &cols, &db);

    eprintln!("E: abs={e_abs:.17e} fro={e_fro:.17e} hash={e_hash:#018x}");
    eprintln!("B: abs={b_abs:.17e} fro={b_fro:.17e} hash={b_hash:#018x}");

    assert_eq!(e_hash, b_hash, "E and B must share one index pattern");
    assert_eq!(e_hash, PATTERN_HASH, "triplet (order, row, col) pattern moved");

    close(e_abs, E_ABS_IDENT, "E abs-sum");
    close(e_fro, E_FRO_IDENT, "E Frobenius");
    close(b_abs, B_ABS_IDENT, "B abs-sum");
    close(b_fro, B_FRO_IDENT, "B Frobenius");
}

#[test]
fn global_assembly_is_pinned_anisotropic_material() {
    let mesh = cube_6tet();
    let basis = NedelecBasis::new(&mesh);
    // A different tensor on each tet, so the per-element material path is
    // exercised rather than a single global constant.
    let er: Vec<_> = (0..mesh.n_tets())
        .map(|i| if i % 2 == 0 { skew_tensor() } else { identity() })
        .collect();
    let ur: Vec<_> = (0..mesh.n_tets())
        .map(|i| if i % 3 == 0 { skew_tensor() } else { identity() })
        .collect();

    let (rows, cols, de, db) = assemble_global_matrices(&mesh, &basis, &er, &ur);
    let (e_abs, e_fro, _) = summarise(&rows, &cols, &de);
    let (b_abs, b_fro, _) = summarise(&rows, &cols, &db);

    eprintln!("E: abs={e_abs:.17e} fro={e_fro:.17e}");
    eprintln!("B: abs={b_abs:.17e} fro={b_fro:.17e}");

    close(e_abs, E_ABS_ANISO, "E abs-sum (aniso)");
    close(e_fro, E_FRO_ANISO, "E Frobenius (aniso)");
    close(b_abs, B_ABS_ANISO, "B abs-sum (aniso)");
    close(b_fro, B_FRO_ANISO, "B Frobenius (aniso)");
}

// ---------------------------------------------------------------------------
// Captured from the interpolatory R2 element, before the modular-basis work.
//
// The four abs-sum / Frobenius pins are the real oracle: they are invariant under
// a relabelling of the unknowns, so they hold across stage 1's move from the
// mode-major to the entity-major layout, and they would still catch a wrong VALUE.
//
// PATTERN_HASH is not invariant, and did move at stage 1 (from 0xc6a1417eefe03ae5).
// That is licensed by `numbering_is_a_relabelling_of_the_mode_major_layout`, which
// proves the new numbering is a permutation of the old. Nothing else may move it.
// ---------------------------------------------------------------------------
const PATTERN_HASH: u64 = 0x2cb6_abad_262c_36f9;

const E_ABS_IDENT: f64 = 2.38082947741327047e2;
const E_FRO_IDENT: f64 = 6.96409442146721247e0;
const B_ABS_IDENT: f64 = 5.81700637508123641e0;
const B_FRO_IDENT: f64 = 1.80341098872539324e-1;

const E_ABS_ANISO: f64 = 1.90885714581524127e2;
const E_FRO_ANISO: f64 = 5.90644022984038841e0;
const B_ABS_ANISO: f64 = 1.03549859184304172e1;
const B_FRO_ANISO: f64 = 3.42254138110637207e-1;
