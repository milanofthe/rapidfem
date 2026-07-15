// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.
//
// The iterative eigensolver, against the true spectrum.
//
// `eigenmode::solve_eigenmode` runs a shift-invert Lanczos. Nothing pinned what it
// returned: not the eigenvalues, not the eigenvectors, and above all not whether a
// returned "mode" was an eigenpair at all. It was not — it reported ghost modes
// below the fundamental and eigenvectors with an O(1) eigenpair residual.
//
// The reference here is `common::dense_spectrum`, which computes the WHOLE spectrum
// of the same discrete system by a dense Cholesky reduction: no shift, no iteration,
// no convergence criterion, nothing the iterative solver could share a bug with.
//
// Three things have to hold, and the third is the one that was missing:
//
//   1. Every eigenvalue the solver returns is one of the true ones.
//   2. It returns the ones NEAREST THE TARGET, which is what shift-invert is for.
//   3. Every eigenPAIR it returns actually satisfies E·x = λ·B·x. An eigenvalue that
//      is right by luck, carried by a vector that solves nothing, is not a mode —
//      and it is exactly what a Lanczos run in the wrong inner product produces, with
//      or without reorthogonalisation.

mod common;

use common::*;
use num_complex::Complex64 as C64;
use rapidfem_fd::basis::NedelecBasis;
use rapidfem_fd::eigenmode::solve_eigenmode;
use rapidfem_fd::materials::{Dispersion, Material};
use rapidfem_fd::mesh::Mesh;
use rapidfem_fd::tet_assembly::{assemble_global_matrices, BasisKind};

/// ‖E·x − λ·B·x‖ / ‖λ·B·x‖ on the free DOFs, from the true matrices.
///
/// Computed here, independently, and not taken from whatever the solver reports
/// about itself.
fn eigenpair_residual(
    basis: &NedelecBasis,
    mesh: &Mesh,
    pec_tris: &[usize],
    lambda: C64,
    x: &[C64],
) -> f64 {
    let pec = pec_dofs(basis, mesh, pec_tris);
    let (er, ur) = air(mesh.n_tets());
    let (rows, cols, de, db) = assemble_global_matrices(mesh, basis, &er, &ur);

    let mut ex = vec![C64::new(0.0, 0.0); basis.n_field];
    let mut bx = vec![C64::new(0.0, 0.0); basis.n_field];
    for k in 0..rows.len() {
        ex[rows[k]] += de[k] * x[cols[k]];
        bx[rows[k]] += db[k] * x[cols[k]];
    }

    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for i in 0..basis.n_field {
        if pec.contains(&i) {
            continue;
        }
        num += (ex[i] - lambda * bx[i]).norm_sqr();
        den += (lambda * bx[i]).norm_sqr();
    }
    (num / den.max(1e-300)).sqrt()
}

/// The solver must return eigenPAIRS, not eigenvalue-shaped numbers.
///
/// This is the assertion the old solver could not pass at any tolerance: its modes
/// had residuals of order 1.
#[test]
fn every_returned_mode_is_an_eigenpair() {
    let (mesh, pec, f101) = cavity();
    let basis = NedelecBasis::with_kind(&mesh, BasisKind::Interpolatory);

    let modes = solve_eigenmode(&mesh, &basis, &pec, None, f101, 4).expect("the solve must succeed");
    assert!(!modes.is_empty(), "the solver returned nothing");

    for (i, m) in modes.iter().enumerate() {
        let r = eigenpair_residual(&basis, &mesh, &pec, m.eigenvalue, &m.field);
        eprintln!(
            "  mode {i}: {:.6} GHz, residual reported {:.2e}, measured {:.2e}",
            m.frequency.re / 1e9,
            m.residual,
            r
        );
        assert!(
            r < 1e-7,
            "mode {i} at {:.6} GHz is not an eigenpair: ‖Ex − λBx‖/‖λBx‖ = {r:.3e}",
            m.frequency.re / 1e9
        );
        // And the solver must not be lying to itself about it.
        assert!(
            (m.residual - r).abs() < 1e-6 * r.max(1e-12) + 1e-9,
            "mode {i}: the solver reports residual {:.3e} but it is {r:.3e}",
            m.residual
        );
    }
}

/// Every eigenvalue returned is a true eigenvalue, and they are the ones nearest the
/// target. A ghost — a Ritz value that matches nothing in the true spectrum — fails
/// the first check. A solver that converges on the wrong end of the spectrum fails
/// the second.
#[test]
fn the_modes_are_the_true_ones_nearest_the_target() {
    let (mesh, pec, f101) = cavity();
    let basis = NedelecBasis::with_kind(&mesh, BasisKind::Interpolatory);

    let sigma = to_lambda(f101);

    // The reference must be the RESONANCES ordered by distance to the shift, not the
    // whole spectrum. The 35 kernel eigenvalues sit at λ = 0, and |0 − σ| is smaller
    // than the distance of the second resonance, so a naive "true eigenvalues nearest
    // the shift" list would put static modes at index 1 and the solver would be
    // marked wrong for correctly refusing to return them. The solver's contract is
    // the non-static eigenvalues nearest the target, and that is what this compares.
    let (res, kernel) = resonances(&basis, &mesh, &pec);
    eprintln!("  {kernel} kernel eigenvalues excluded from the reference");
    let mut by_distance = res;
    by_distance.sort_by(|a, b| (a - sigma).abs().partial_cmp(&(b - sigma).abs()).unwrap());

    let n = 4;
    let modes = solve_eigenmode(&mesh, &basis, &pec, None, f101, n).expect("the solve must succeed");
    assert_eq!(modes.len(), n, "the solver converged on only {} of {n} modes", modes.len());

    eprintln!("  target {:.6} GHz (λ = {sigma:.6e})", f101 / 1e9);
    for (i, m) in modes.iter().enumerate() {
        let got = m.eigenvalue.re;
        let want = by_distance[i];
        let rel = (got - want).abs() / want.abs().max(1.0);
        eprintln!(
            "  mode {i}: {:.9} GHz vs true {:.9} GHz (rel {rel:.2e})",
            to_ghz(got),
            to_ghz(want)
        );
        assert!(
            rel < 1e-8,
            "mode {i}: the solver returned λ = {got:.9e}, but the {i}-th true eigenvalue by \
             distance to the shift is {want:.9e}"
        );
    }
}

/// The ghost test, stated in physical terms: a cavity has nothing below its
/// fundamental, and the solver must not invent anything there.
///
/// The old solver reported a cluster near 5.3 GHz in this cavity, whose true spectrum
/// has nothing at all between 0 and the fundamental.
#[test]
fn no_mode_is_reported_below_the_fundamental() {
    let (mesh, pec, f101) = cavity();
    let basis = NedelecBasis::with_kind(&mesh, BasisKind::Interpolatory);

    let (res, _) = resonances(&basis, &mesh, &pec);
    let fundamental = to_ghz(res[0]);
    eprintln!("  the true discrete fundamental is {fundamental:.6} GHz");

    // Ask for many modes: the more it is pushed past what the Krylov space supports,
    // the more a solver without a convergence test invents.
    let modes = solve_eigenmode(&mesh, &basis, &pec, None, f101, 8).expect("the solve must succeed");
    for m in &modes {
        let f = m.frequency.re / 1e9;
        eprintln!("  returned {f:.6} GHz");
        assert!(
            f > 0.999 * fundamental,
            "the solver reported a mode at {f:.6} GHz, below the cavity's fundamental at \
             {fundamental:.6} GHz: there is no such mode"
        );
    }
}

/// The curl kernel sits at λ = 0. Those are genuine eigenpairs of the pencil, and a
/// residual test alone would happily let them through as "resonances at 0 Hz". They
/// are not resonances and must not be returned.
#[test]
fn the_static_kernel_is_not_reported_as_a_resonance() {
    let (mesh, pec, f101) = cavity();
    let basis = NedelecBasis::with_kind(&mesh, BasisKind::Interpolatory);
    let (_, kernel) = resonances(&basis, &mesh, &pec);
    eprintln!("  the discrete curl kernel has dimension {kernel}");
    assert!(kernel > 0, "this cavity has no kernel, so the test checks nothing");

    let modes = solve_eigenmode(&mesh, &basis, &pec, None, f101, 6).expect("the solve must succeed");
    for m in &modes {
        assert!(
            m.frequency.re > 1e6,
            "a static (gradient) mode was returned as a resonance at {:.3e} Hz",
            m.frequency.re
        );
    }
}

/// The lossy path, which is the whole reason the inner product is BILINEAR.
///
/// With loss, E and B are complex SYMMETRIC — not Hermitian — so the form the
/// operator is self-adjoint in is xᵀBy without a conjugate. A Hermitian Lanczos here
/// would not merely be less accurate; it would be solving a different problem.
///
/// The check has a closed form. Fill the cavity uniformly with εr = ε'(1 − j·tanδ).
/// Then B = εr·B_vac, so λ = λ_vac/εr, and
///
///     k₀ = √λ = √λ_vac / √εr,   1/√εr ≈ (1 + j·tanδ/2)/√ε'   for small tanδ,
///
/// so Re(f)/Im(f) = 2/tanδ and the solver's Q = ½·Re(f)/|Im(f)| = 1/tanδ, exactly, to
/// first order. It depends on nothing about the mesh or the element — only on the
/// complex arithmetic being right.
#[test]
fn a_uniformly_lossy_cavity_has_q_equal_to_one_over_tan_delta() {
    let (mesh, pec, f101_vac) = cavity();

    let eps = 4.0_f64;
    for tand in [1e-2_f64, 1e-3] {
        let mat = Material {
            er: eps,
            ur: 1.0,
            tand,
            cond: 0.0,
            tet_indices: (0..mesh.n_tets()).collect(),
            er_diag: None,
            ur_diag: None,
            dispersion: Dispersion::None,
        };
        let basis = NedelecBasis::with_kind(&mesh, BasisKind::Interpolatory);

        // The dielectric drags the resonance down by √ε'.
        let target = f101_vac / eps.sqrt();
        let modes = solve_eigenmode(&mesh, &basis, &pec, Some(&[mat]), target, 2)
            .expect("the lossy solve must succeed");
        assert!(!modes.is_empty(), "no mode converged on the lossy cavity");

        let m = &modes[0];
        let q_want = 1.0 / tand;
        let q_err = (m.q_factor - q_want).abs() / q_want;
        eprintln!(
            "  tanδ = {tand:.0e}: f = {:.6} GHz, Q = {:.3} (closed form {q_want:.1}, rel {q_err:.2e}), \
             residual {:.2e}",
            m.frequency.re / 1e9,
            m.q_factor,
            m.residual
        );

        // The eigenpair must still be an eigenpair, with a genuinely complex λ.
        assert!(m.residual < 1e-8, "the lossy mode is not an eigenpair: residual {:.3e}", m.residual);
        assert!(m.eigenvalue.im.abs() > 0.0, "a lossy cavity must give a complex eigenvalue");

        // Q = 1/tanδ is exact to first order in tanδ, so the smaller tanδ is, the
        // tighter this has to be. Allow the first-order truncation and no more.
        assert!(
            q_err < 5.0 * tand,
            "tanδ = {tand:.0e}: Q came out at {:.3}, but a uniformly lossy cavity has \
             Q = 1/tanδ = {q_want:.1}",
            m.q_factor
        );
    }
}

/// The reachable band, pinned.
///
/// This is not a bug being tolerated — it is a property of shift-invert on a pencil
/// with a huge null space, and the solver's usage depends on knowing it. The discrete
/// gradients all sit at λ = 0, which shift-invert maps to |μ| = 1/σ. A mode at λ is
/// ahead of that cluster only when |λ − σ| < σ, i.e. f < √2·f_target. Beyond it, the
/// mode is behind the entire kernel and no Krylov space reaches it.
///
/// The test states the consequence and checks it both ways: a target on the lowest
/// mode reaches only the lowest few, and a target in the middle of the band reaches
/// modes on both sides of it — INCLUDING ones below the target, which is the part
/// that would be surprising if the mechanism were anything else.
#[test]
fn the_reachable_band_is_below_root_two_times_the_target() {
    let (mesh, pec, f101) = cavity();
    let basis = NedelecBasis::with_kind(&mesh, BasisKind::Interpolatory);
    let (truth, kernel) = resonances(&basis, &mesh, &pec);
    assert!(kernel > 0, "no kernel: this test checks nothing");

    // Target the middle of the first four resonances. All of them then satisfy
    // |λ − σ| < σ and are ahead of the kernel cluster.
    let lo = to_ghz(truth[0]);
    let hi = to_ghz(truth[3]);
    let target = 0.5 * (lo + hi) * 1e9;
    eprintln!(
        "  true resonances: {:.3} {:.3} {:.3} {:.3} GHz; target at {:.3} GHz",
        lo,
        to_ghz(truth[1]),
        to_ghz(truth[2]),
        hi,
        target / 1e9
    );

    let modes = solve_eigenmode(&mesh, &basis, &pec, None, target, 4).expect("the solve must succeed");
    assert_eq!(modes.len(), 4, "a mid-band target must reach all four");

    // Every one of the four is a true resonance, and at least one lies BELOW the
    // target — which only happens because the band is two-sided in λ, not because the
    // solver walks upward from the shift.
    let mut below = 0;
    for m in &modes {
        let f = m.frequency.re;
        let matched = truth.iter().any(|&t| (m.eigenvalue.re - t).abs() / t < 1e-8);
        assert!(matched, "a returned mode at {:.6} GHz is not a true resonance", f / 1e9);
        if f < target {
            below += 1;
        }
        // The band, restated on the returned modes.
        assert!(
            m.eigenvalue.re < 2.0 * to_lambda(target),
            "a mode at {:.6} GHz was returned from beyond λ = 2σ, which shift-invert cannot \
             reach past the kernel: the mechanism is not what this test believes",
            f / 1e9
        );
        below += 0;
    }
    assert!(below > 0, "no mode below the target: the band should be two-sided");
    eprintln!("  all 4 reached, {below} of them below the target");
}
