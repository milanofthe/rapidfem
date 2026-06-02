// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Sparse direct solver abstraction.
//!
//! Each backend (PARDISO, faer LU, Apple Accelerate Bunch-Kaufman, …) lives in
//! its own submodule and implements `SparseSolver`. Callers don't care which
//! backend they got — they assemble the COO triplets, hand them to the trait,
//! and ask for solutions.
//!
//! Backend selection is via `SolverChoice::from_env()` (env var
//! `RAPIDFEM_SOLVER=auto|pardiso|accelerate|faer`, default `auto`). The auto
//! order is PARDISO → Accelerate (macOS) → faer LU.

use num_complex::Complex64 as C64;

pub mod pardiso;
pub mod faer_lu;
pub mod schur;
#[cfg(target_os = "macos")]
pub mod accelerate;

/// Sparse direct solver for a complex-symmetric matrix.
///
/// Input convention: full COO triplets `(rows, cols, vals)` of dimension `n`.
/// Off-diagonal entries appear in both halves (the FEM assembly produces them
/// that way naturally); each backend filters to the form it prefers
/// (upper-CSR for PARDISO, full triplets for faer, real-block CSC for
/// Accelerate). The same factorisation is reused for many RHS via `solve`.
pub trait SparseSolver: Send {
    /// Build the symbolic + numeric factorisation from full COO triplets.
    /// Resets any previously stored factor.
    fn factorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[C64],
    ) -> Result<(), String>;

    /// Re-factor with new values for the same sparsity pattern. Backends that
    /// can amortise the symbolic step override this; the default falls back
    /// to a full re-factorisation.
    fn refactorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[C64],
    ) -> Result<(), String> {
        self.factorize(n, rows, cols, vals)
    }

    /// Solve `K · x = b` using the cached factorisation.
    fn solve(&mut self, b: &[C64]) -> Result<Vec<C64>, String>;

    /// Backend name, for logs.
    fn name(&self) -> &'static str;
}

/// User-facing backend selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverChoice {
    /// Try PARDISO → Accelerate (macOS) → faer in that order.
    Auto,
    /// Intel MKL PARDISO (dynamic load of `mkl_rt`).
    Pardiso,
    /// Apple Accelerate sparse Bunch-Kaufman (macOS only).
    Accelerate,
    /// Pure-Rust faer sparse LU.
    Faer,
}

impl SolverChoice {
    /// Read `RAPIDFEM_SOLVER` and parse into a choice. Unknown values fall
    /// back to `Auto`.
    pub fn from_env() -> Self {
        match std::env::var("RAPIDFEM_SOLVER").ok().as_deref() {
            Some(s) => match s.to_ascii_lowercase().as_str() {
                "pardiso" => Self::Pardiso,
                "accelerate" => Self::Accelerate,
                "faer" => Self::Faer,
                _ => Self::Auto,
            },
            None => Self::Auto,
        }
    }
}

/// Instantiate a solver matching `choice`, falling back gracefully when the
/// requested backend isn't available at runtime (e.g. PARDISO without MKL,
/// Accelerate off macOS). Logs the actual choice to stderr.
pub fn pick(choice: SolverChoice) -> Box<dyn SparseSolver> {
    let try_pardiso = || pardiso::PardisoSolver::try_new()
        .map(|s| Box::new(s) as Box<dyn SparseSolver>);
    #[cfg(target_os = "macos")]
    let try_accelerate = || accelerate::AccelerateSolver::try_new()
        .map(|s| Box::new(s) as Box<dyn SparseSolver>);
    #[cfg(not(target_os = "macos"))]
    let try_accelerate = || -> Option<Box<dyn SparseSolver>> { None };
    let make_faer = || Box::new(faer_lu::FaerLuSolver::new()) as Box<dyn SparseSolver>;

    let solver = match choice {
        SolverChoice::Pardiso => try_pardiso().unwrap_or_else(|| {
            eprintln!("  solver: PARDISO requested but unavailable, falling back to faer LU");
            make_faer()
        }),
        SolverChoice::Accelerate => try_accelerate().unwrap_or_else(|| {
            eprintln!("  solver: Accelerate requested but unavailable, falling back to faer LU");
            make_faer()
        }),
        SolverChoice::Faer => make_faer(),
        SolverChoice::Auto => try_pardiso()
            .or_else(try_accelerate)
            .unwrap_or_else(make_faer),
    };
    eprintln!("  solver: using {}", solver.name());
    solver
}
