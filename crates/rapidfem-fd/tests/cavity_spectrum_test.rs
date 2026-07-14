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
use rapidfem_fd::order::{cell_diameter, cell_wavenumbers, wavelength_policy, OrderMap};
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
    spectrum_of(&Nedelec2Basis::with_kind(mesh, kind), mesh, pec_tris)
}

/// The same, for an arbitrary order map.
fn spectrum_of(basis: &Nedelec2Basis, mesh: &Mesh, pec_tris: &[usize]) -> Vec<f64> {
    let basis = basis;

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

// ===========================================================================
// Stage 4: variable order.
// ===========================================================================

/// The nonzero part of a spectrum, and the size of the kernel it sat on.
fn resonances_of(basis: &Nedelec2Basis, mesh: &Mesh, pec: &[usize]) -> (Vec<f64>, usize) {
    let s = spectrum_of(basis, mesh, pec);
    let scale = s.last().copied().unwrap();
    let k = s.iter().filter(|&&l| l.abs() < 1e-9 * scale).count();
    (s[k..].to_vec(), k)
}

/// The kernel of the discrete curl is the space of discrete gradients. That is not
/// a vague statement: it has an exactly countable dimension, because the scalar
/// potentials are the DOFs of the Lagrange space one rung down the de Rham
/// sequence, with the boundary ones removed by PEC.
///
///   order 1 (Whitney):  grad of P1, zero on the boundary  ->  #interior nodes
///   order 2:            grad of P2, zero on the boundary  ->  #interior nodes
///                                                             + #interior edges
///
/// If the assembled kernel has any other dimension, the global space is not the one
/// the element claims to build. This is the exact-sequence property, checked on the
/// assembled system rather than on a single element.
#[test]
fn the_kernel_dimension_is_the_number_of_discrete_gradients() {
    let (mesh, pec, _) = cavity();

    let boundary_nodes: HashSet<usize> = pec.iter().flat_map(|&t| mesh.tris[t]).collect();
    let boundary_edges: HashSet<usize> = pec.iter().flat_map(|&t| mesh.tri_to_edge[t]).collect();
    let interior_nodes = mesh.nodes.len() - boundary_nodes.len();
    let interior_edges = mesh.n_edges() - boundary_edges.len();

    for (p, want) in [(1u8, interior_nodes), (2u8, interior_nodes + interior_edges)] {
        let basis =
            Nedelec2Basis::with_orders(&mesh, BasisKind::Hierarchical, OrderMap::uniform(&mesh, p));
        let (_, kernel) = resonances_of(&basis, &mesh, &pec);
        eprintln!(
            "  p = {p}: {} DOFs, kernel dim {kernel} (discrete gradients: {want})",
            basis.n_field
        );
        assert_eq!(
            kernel, want,
            "p = {p}: the curl kernel has dimension {kernel}, but the space of discrete \
             gradients has dimension {want}"
        );
    }
}

/// Order 1 is the Whitney element, and it converges like one.
///
/// The eigenvalue error of a curl-conforming element of order p is O(h^2p), so
/// halving h must cut the order-1 eigenvalue error by about 4. Measuring the rate
/// is what distinguishes "order 1 works" from "order 1 happens to produce a number".
#[test]
fn order_1_converges_at_the_whitney_rate() {
    let (a, b, d): (f64, f64, f64) = (0.02286, 0.01016, 0.030);
    let f101 = 0.5 * C0 * ((1.0 / a).powi(2) + (1.0 / d).powi(2)).sqrt();
    let lambda = (2.0 * std::f64::consts::PI * f101 / C0).powi(2);

    let mut errs = Vec::new();
    for m in [1usize, 2, 4] {
        let mesh = box_mesh(a, b, d, 2 * m, m, 3 * m);
        let pec = boundary_tris(&mesh);
        let basis =
            Nedelec2Basis::with_orders(&mesh, BasisKind::Hierarchical, OrderMap::uniform(&mesh, 1));
        let (res, _) = resonances_of(&basis, &mesh, &pec);
        let err = (res[0] - lambda).abs() / lambda;
        eprintln!(
            "  h/{m}: {} tets, {} DOFs, lambda error {err:.4e}",
            mesh.n_tets(),
            basis.n_field
        );
        errs.push(err);
    }

    let rates: Vec<f64> = errs.windows(2).map(|w| (w[0] / w[1]).log2()).collect();
    for (i, r) in rates.iter().enumerate() {
        eprintln!("  rate h/{} -> h/{}: {r:.2}", 1 << i, 1 << (i + 1));
    }
    // The coarsest mesh is 36 tetrahedra across a whole cavity: deeply
    // pre-asymptotic, and it flatters the first rate. The finest pair is the one
    // that measures the element rather than the mesh.
    let rate = *rates.last().unwrap();
    assert!(
        rate > 1.7 && rate < 2.4,
        "order 1 converged at rate {rate:.2} on the finest pair, but the Whitney element is O(h^2)"
    );
}

/// The whole point of mixed order, checked as a hard mathematical constraint.
///
/// The mixed-order space is spanned by a SUBSET of the uniform order-2 basis
/// functions (drop the second function on the reduced edges and the bubbles on the
/// reduced faces), and it CONTAINS the uniform order-1 space (every Whitney function
/// survives the minimum rule). So
///
///     V(p=1)  subset  V(mixed)  subset  V(p=2)
///
/// and by the Courant-Fischer min-max principle a smaller space can only push the
/// eigenvalues up:
///
///     lambda_k(p=2)  <=  lambda_k(mixed)  <=  lambda_k(p=1)   for every k.
///
/// This must hold for every eigenvalue, and a broken element could not fake it. If
/// the minimum rule dropped the wrong function, or a p=2 cell next to a p=1 cell
/// kept a function whose partner across the shared face was gone, the space would
/// not be conforming, would not be a subspace, and the bracket would break. It is
/// the conformity test, written as an inequality.
#[test]
fn mixed_order_is_bracketed_by_the_uniform_spaces() {
    let (mesh, pec, _) = cavity();

    // Reduce the cells in the first quarter of the box (in z) to order 1. That
    // produces a genuine interface: cells at p = 2 whose edges and faces have been
    // pulled down to p = 1 by their neighbours.
    let zmax = 0.030;
    let cells: Vec<u8> = (0..mesh.n_tets())
        .map(|t| {
            let zc: f64 = mesh.tets[t].iter().map(|&n| mesh.nodes[n][2]).sum::<f64>() / 4.0;
            if zc < 0.25 * zmax {
                1
            } else {
                2
            }
        })
        .collect();
    let n_reduced = cells.iter().filter(|&&p| p == 1).count();
    assert!(
        n_reduced > 0 && n_reduced < mesh.n_tets(),
        "the test needs a genuine order interface"
    );

    let b1 =
        Nedelec2Basis::with_orders(&mesh, BasisKind::Hierarchical, OrderMap::uniform(&mesh, 1));
    let bm = Nedelec2Basis::with_orders(
        &mesh,
        BasisKind::Hierarchical,
        OrderMap::from_cells(&mesh, cells),
    );
    let b2 =
        Nedelec2Basis::with_orders(&mesh, BasisKind::Hierarchical, OrderMap::uniform(&mesh, 2));

    eprintln!(
        "  {} of {} cells reduced to p=1; DOFs: p1 {}, mixed {}, p2 {}",
        n_reduced,
        mesh.n_tets(),
        b1.n_field,
        bm.n_field,
        b2.n_field
    );
    assert!(
        b1.n_field < bm.n_field && bm.n_field < b2.n_field,
        "the spaces must nest strictly"
    );

    // Courant-Fischer compares the k-th eigenvalue of the FULL spectra at the SAME
    // index k, kernel included. It must not be applied to "the first resonance of
    // each": the three spaces hold different numbers of discrete gradients, so their
    // kernels have different dimensions and the first resonance sits at a different
    // index in each. Comparing those compares λ_i to λ_j with i ≠ j, about which
    // Courant-Fischer says nothing at all.
    let s1 = spectrum_of(&b1, &mesh, &pec);
    let sm = spectrum_of(&bm, &mesh, &pec);
    let s2 = spectrum_of(&b2, &mesh, &pec);

    let slack = |x: f64| 1e-8 * x.abs().max(1.0);
    for k in 0..sm.len() {
        assert!(
            sm[k] >= s2[k] - slack(s2[k]),
            "eigenvalue {k}: the mixed space ({:.6e}) fell below the order-2 space ({:.6e}), so \
             it is not a subspace of it: the element is not conforming across the order jump",
            sm[k],
            s2[k]
        );
    }
    for k in 0..s1.len() {
        assert!(
            s1[k] >= sm[k] - slack(sm[k]),
            "eigenvalue {k}: the order-1 space ({:.6e}) fell below the mixed space ({:.6e}), so \
             the mixed space does not contain it: the minimum rule dropped a Whitney function",
            s1[k],
            sm[k]
        );
    }
    eprintln!(
        "  Courant-Fischer holds index by index: all {} of p1, all {} of mixed",
        s1.len(),
        sm.len()
    );

    let (r1, k1) = resonances_of(&b1, &mesh, &pec);
    let (rm, km) = resonances_of(&bm, &mesh, &pec);
    let (r2, k2) = resonances_of(&b2, &mesh, &pec);
    eprintln!("  kernel dims: p1 {k1}, mixed {km}, p2 {k2}");
    eprintln!(
        "  fundamental: p1 {:.6} GHz, mixed {:.6} GHz, p2 {:.6} GHz",
        to_ghz(r1[0]),
        to_ghz(rm[0]),
        to_ghz(r2[0])
    );

    // And the point of the exercise: the mixed space buys most of order 2's accuracy
    // for fewer DOFs than order 2 needs.
    let (a, _b, d): (f64, f64, f64) = (0.02286, 0.01016, 0.030);
    let f101 = 0.5 * C0 * ((1.0 / a).powi(2) + (1.0 / d).powi(2)).sqrt();
    let lambda = (2.0 * std::f64::consts::PI * f101 / C0).powi(2);
    let e = |x: f64| (x - lambda).abs() / lambda;
    eprintln!(
        "  fundamental error: p1 {:.3e}, mixed {:.3e}, p2 {:.3e}",
        e(r1[0]),
        e(rm[0]),
        e(r2[0])
    );
    assert!(e(rm[0]) < e(r1[0]), "the mixed space must beat uniform order 1");
}

// ===========================================================================
// Stage 5: the a-priori order policy.
// ===========================================================================

/// A box graded in z: the cells near z = 0 are small for a reason that has nothing
/// to do with the wavelength. This is the situation the policy exists for. A
/// uniform mesh has nothing to decide.
fn graded_box(lx: f64, ly: f64, lz: f64, nx: usize, ny: usize, nz: usize, grade: f64) -> Mesh {
    let idx = |i: usize, j: usize, k: usize| (i * (ny + 1) + j) * (nz + 1) + k;
    // Graded in ALL THREE axes toward the corner at the origin. Grading only one
    // axis would be pointless: a cell's diameter is its LONGEST edge, so a cell that
    // is thin in z but full-width in x and y has the same diameter as a coarse one,
    // and the policy would rightly refuse to reduce it. Only a cell that is small in
    // every direction is genuinely geometry-refined.
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

/// The policy's oracle: on a mesh with geometry-driven refinement, it must give back
/// DOFs without giving back accuracy.
///
/// The claim being tested is not "mixed order is cheaper" (dropping DOFs is always
/// cheaper) and not "mixed order is accurate" (keeping them always is). It is that
/// the policy picks the RIGHT cells: the ones where the order-1 dispersion error
/// (k·h)² is already far below the model's overall error, so that reducing them
/// costs essentially nothing. If the policy chose badly — reducing cells where the
/// mode has structure — the error would jump.
#[test]
fn the_order_policy_gives_back_dofs_without_giving_back_accuracy() {
    let (a, b, d): (f64, f64, f64) = (0.02286, 0.01016, 0.030);
    let f101 = 0.5 * C0 * ((1.0 / a).powi(2) + (1.0 / d).powi(2)).sqrt();
    let lambda = (2.0 * std::f64::consts::PI * f101 / C0).powi(2);

    let mesh = graded_box(a, b, d, 3, 2, 4, 2.5);
    let pec = boundary_tris(&mesh);

    let id = {
        let (z, o) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0));
        [[o, z, z], [z, o, z], [z, z, o]]
    };
    let er = vec![id; mesh.n_tets()];
    let ur = vec![id; mesh.n_tets()];
    let k = cell_wavenumbers(&mesh, &er, &ur, f101);

    let hs: Vec<f64> = (0..mesh.n_tets()).map(|t| cell_diameter(&mesh, t)).collect();
    let khs: Vec<f64> = (0..mesh.n_tets()).map(|t| k[t] * hs[t]).collect();
    let kh_min = khs.iter().cloned().fold(f64::INFINITY, f64::min);
    let kh_max = khs.iter().cloned().fold(0.0_f64, f64::max);
    eprintln!("  {} tets, k*h from {kh_min:.3} to {kh_max:.3}", mesh.n_tets());

    let full = Nedelec2Basis::with_orders(&mesh, BasisKind::Hierarchical, OrderMap::uniform(&mesh, 2));
    let (r_full, _) = resonances_of(&full, &mesh, &pec);
    let e_full = (r_full[0] - lambda).abs() / lambda;
    eprintln!("  uniform p=2: {} DOFs, error {e_full:.3e}", full.n_field);

    // Sweep theta so the trade-off is a measurement, not an assertion at one point.
    let mut any = false;
    for theta in [0.5, 0.75, 1.0, 1.5] {
        let orders = wavelength_policy(&mesh, &k, theta);
        let n1 = orders.cell.iter().filter(|&&p| p == 1).count();
        if n1 == 0 || n1 == mesh.n_tets() {
            eprintln!("  theta = {theta:.2}: {n1} cells at p=1 (nothing to decide, skipped)");
            continue;
        }
        any = true;
        let basis = Nedelec2Basis::with_orders(&mesh, BasisKind::Hierarchical, orders);
        let (r, _) = resonances_of(&basis, &mesh, &pec);
        let err = (r[0] - lambda).abs() / lambda;
        let saved = 1.0 - basis.n_field as f64 / full.n_field as f64;
        eprintln!(
            "  theta = {theta:.2}: {n1}/{} cells at p=1, {} DOFs ({:.0}% saved), error {err:.3e} \
             ({:.1}x the uniform error)",
            mesh.n_tets(),
            basis.n_field,
            100.0 * saved,
            err / e_full
        );

        assert!(saved > 0.0, "theta = {theta}: the policy saved no DOFs");
        // The policy is only allowed to reduce cells whose order-1 dispersion error
        // is negligible. If it did, the eigenvalue can barely move: an order of
        // magnitude of headroom is generous and still catches a policy that reduces
        // the wrong cells (uniform p=1 on this mesh is ~100x worse).
        assert!(
            err < 10.0 * e_full,
            "theta = {theta}: the policy reduced cells that mattered — the error went from \
             {e_full:.3e} to {err:.3e}"
        );
    }

    // The control. Reducing EVERYTHING is materially worse, so the free lunch above
    // is the policy's CHOICE of cells and not an artefact of a mesh on which order 2
    // was never earning its keep anywhere.
    //
    // (This mesh is a poor one for this mode on purpose: grading toward a corner
    // starves the region the field actually lives in, so even uniform order 2 is
    // only good to ~2%. That is what makes the comparison meaningful — the fine
    // corner cells contribute nothing either way, which is exactly the situation the
    // policy is supposed to detect and exploit.)
    let all_p1 = Nedelec2Basis::with_orders(&mesh, BasisKind::Hierarchical, OrderMap::uniform(&mesh, 1));
    let (r1, _) = resonances_of(&all_p1, &mesh, &pec);
    let e1 = (r1[0] - lambda).abs() / lambda;
    eprintln!(
        "  uniform p=1: {} DOFs, error {e1:.3e} ({:.1}x the uniform-p2 error)",
        all_p1.n_field,
        e1 / e_full
    );
    assert!(
        e1 > 2.0 * e_full,
        "the control is not a control: blanket order 1 is not materially worse than order 2 \
         on this mesh, so 'the policy reduced the harmless cells' is not something this mesh \
         can demonstrate"
    );
}

/// The policy reduces exactly the cells it says it does, and reduces them for the
/// stated reason. A cell is at order 1 iff k*h < theta.
#[test]
fn the_policy_reduces_exactly_the_cells_below_theta() {
    let (a, b, d): (f64, f64, f64) = (0.02286, 0.01016, 0.030);
    let mesh = graded_box(a, b, d, 3, 2, 4, 2.5);

    let id = {
        let (z, o) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0));
        [[o, z, z], [z, o, z], [z, z, o]]
    };
    // A dielectric in half the box: the wavelength there is shorter, so a cell of
    // the same size is LESS resolved and must not be reduced as readily. The policy
    // has to read the cell's own material, not a global one.
    let eps_r = 9.0;
    let mut er = vec![id; mesh.n_tets()];
    for t in 0..mesh.n_tets() {
        let xc: f64 = mesh.tets[t].iter().map(|&n| mesh.nodes[n][0]).sum::<f64>() / 4.0;
        if xc > 0.5 * a {
            er[t][0][0] = C64::new(eps_r, 0.0);
            er[t][1][1] = C64::new(eps_r, 0.0);
            er[t][2][2] = C64::new(eps_r, 0.0);
        }
    }
    let ur = vec![id; mesh.n_tets()];
    let freq = 8.24e9;
    let k = cell_wavenumbers(&mesh, &er, &ur, freq);

    // The dielectric cells must carry a wavenumber sqrt(9) = 3x the air ones.
    let air: Vec<f64> = (0..mesh.n_tets()).filter(|&t| er[t][0][0].re == 1.0).map(|t| k[t]).collect();
    let die: Vec<f64> = (0..mesh.n_tets()).filter(|&t| er[t][0][0].re == eps_r).map(|t| k[t]).collect();
    assert!(!air.is_empty() && !die.is_empty());
    let ratio = die[0] / air[0];
    eprintln!("  wavenumber ratio dielectric/air: {ratio:.4} (expect {:.4})", eps_r.sqrt());
    assert!((ratio - eps_r.sqrt()).abs() < 1e-12, "the policy is not reading the cell's material");

    let theta = 0.75;
    let orders = wavelength_policy(&mesh, &k, theta);
    for t in 0..mesh.n_tets() {
        let kh = k[t] * cell_diameter(&mesh, t);
        let want = if kh < theta { 1 } else { 2 };
        assert_eq!(
            orders.cell[t], want,
            "tet {t}: k*h = {kh:.4}, theta = {theta}, so p should be {want} but is {}",
            orders.cell[t]
        );
    }
    let n1 = orders.cell.iter().filter(|&&p| p == 1).count();
    eprintln!("  {n1}/{} cells below theta = {theta}", mesh.n_tets());
    assert!(n1 > 0, "no cell is below theta: the check is vacuous");
}
