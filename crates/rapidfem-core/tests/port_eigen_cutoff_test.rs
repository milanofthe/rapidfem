// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Analytic golden test for the 2-D port-mode eigensolver
//! ([`rapidfem_core::port_eigen::solve_modes`]).
//!
//! A hollow PEC rectangular waveguide of cross-section `a × b` has the
//! *exact* scalar-Helmholtz cutoff spectrum
//!
//! ```text
//!   k_c(m, n) = π · √((m/a)² + (n/b)²)
//! ```
//!
//! with `m, n ≥ 1` for `TM` (Dirichlet, `E_z = 0` on the wall) and
//! `m, n ≥ 0`, not both zero, for `TE` (Neumann, `∂H_z/∂n = 0`). These
//! closed forms are the anchors here, so no codegen / derivation script is
//! needed — the constants are written out directly.
//!
//! The P1-nodal scalar solver gives an *eigenvalue upper bound*, but the
//! row-sum-lumped mass shifts it back down, so the two errors nearly
//! cancel: on a structured `24×14` rectangle the lowest TE/TM cutoffs land
//! within ~0.1–0.5 % of the closed form (measured: TM11 0.07 %, TM21 0.09 %
//! on a finer `40×22` mesh). This is still a convergence-grade golden, not
//! a `1e-10` pin, so we assert the lowest few *sorted* computed `k_c`
//! against the analytic ladder within a sane 1 % relative FEM tolerance.
//! The TE constant (`k_c = 0`) mode is dropped inside the solver; the
//! scalar Helmholtz problem produces no spurious modes, so no further
//! filtering is needed. The dense per-face eigensolve is `O(N³)`, so the
//! mesh is kept modest (a few hundred DOFs) to stay fast in debug builds.

use rapidfem_core::port_eigen::{solve_modes, ModeKind, PortMesh2D};
use std::f64::consts::PI;

/// Cross-section dimensions. `b` is deliberately *not* a simple ratio of
/// `a`, so the low TE/TM cutoffs are non-degenerate and each computed mode
/// maps to a single analytic value (no accidental TE20 == TE01 collision).
const A: f64 = 2.0;
const B: f64 = 1.1;

/// Relative tolerance on each cutoff wavenumber. Lumped-mass P1 nearly
/// cancels the consistent-mass eigenvalue bias; on the mesh below the
/// lowest handful land within ~0.5 %, so 1 % is a sane, non-fudged FEM
/// tolerance with ~2× margin.
const REL_TOL: f64 = 0.01;

/// Analytic rectangular-guide cutoff `k_c(m, n)`.
fn kc(m: u32, n: u32) -> f64 {
    PI * ((m as f64 / A).powi(2) + (n as f64 / B).powi(2)).sqrt()
}

/// Build a structured right-triangle mesh of the `A × B` rectangle with
/// `nx × ny` cells (each split into two triangles), on the z = 0 plane,
/// and flatten it to a [`PortMesh2D`]. The outer rectangle perimeter is
/// auto-detected as the PEC boundary by `from_face`.
fn rect_port(nx: usize, ny: usize) -> PortMesh2D {
    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            nodes.push([A * i as f64 / nx as f64, B * j as f64 / ny as f64, 0.0]);
        }
    }
    let id = |i: usize, j: usize| j * (nx + 1) + i;
    let mut tris = Vec::with_capacity(2 * nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let (n00, n10, n01, n11) =
                (id(i, j), id(i + 1, j), id(i, j + 1), id(i + 1, j + 1));
            tris.push([n00, n10, n11]);
            tris.push([n00, n11, n01]);
        }
    }
    PortMesh2D::from_face(&nodes, &tris, [0.0, 0.0, 1.0], None)
}

/// Assert the solver's lowest `kind` cutoffs match the analytic ladder
/// `want` (already sorted ascending) within [`REL_TOL`], printing the full
/// computed-vs-analytic table for the report.
fn assert_cutoffs(kind: ModeKind, want: &[(f64, &str)]) {
    // ~0.08 element size: lumped-mass P1 still resolves the lowest
    // cutoffs to ~0.5 %, while keeping the dense N×N eigensolve (N a few
    // hundred) fast in an unoptimised debug test build.
    let nx = 24;
    let ny = 14;
    let pm = rect_port(nx, ny);
    let modes = solve_modes(&pm, kind, want.len());
    assert_eq!(
        modes.len(),
        want.len(),
        "{kind:?}: solver returned {} modes, wanted {}",
        modes.len(),
        want.len()
    );
    eprintln!("{kind:?} cutoffs (mesh {nx}x{ny}):");
    let mut worst = 0.0_f64;
    for (mode, &(w, label)) in modes.iter().zip(want) {
        let rel = (mode.k_c - w).abs() / w;
        worst = worst.max(rel);
        eprintln!(
            "  {label:7}: computed k_c = {:.5}, analytic = {:.5}, rel = {:.4}",
            mode.k_c, w, rel
        );
    }
    for (mode, &(w, label)) in modes.iter().zip(want) {
        let rel = (mode.k_c - w).abs() / w;
        assert!(
            rel < REL_TOL,
            "{kind:?} {label}: computed k_c = {:.5}, analytic = {:.5}, \
             rel = {:.4} exceeds tol {REL_TOL}",
            mode.k_c,
            w,
            rel
        );
    }
    eprintln!("{kind:?} worst rel error: {worst:.4}");
}

/// TE (Neumann) ladder: the three lowest cutoffs of the `A × B = 2 × 1.1`
/// guide are TE10 = π/a, TE01 = π/b, TE20 = 2π/a, all distinct.
#[test]
fn te_rectangular_cutoffs_match_analytic() {
    let want = [
        (kc(1, 0), "TE10"),
        (kc(0, 1), "TE01"),
        (kc(2, 0), "TE20"),
    ];
    // Sanity: the analytic ladder really is sorted ascending and distinct.
    assert!(want[0].0 < want[1].0 && want[1].0 < want[2].0);
    assert_cutoffs(ModeKind::Te, &want);
}

/// TM (Dirichlet) ladder: the two lowest cutoffs are TM11 and TM21.
#[test]
fn tm_rectangular_cutoffs_match_analytic() {
    let want = [(kc(1, 1), "TM11"), (kc(2, 1), "TM21")];
    assert!(want[0].0 < want[1].0);
    assert_cutoffs(ModeKind::Tm, &want);
}
