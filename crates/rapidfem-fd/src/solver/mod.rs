// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Sparse direct solver abstraction.
//!
//! Two backends implement `SparseSolver`: Intel MKL PARDISO (dynamically
//! loaded, if installed) and rslab's pure-Rust complex-symmetric LDLᵀ, which
//! is also the fallback everywhere. Callers assemble the COO triplets, hand
//! them to the trait, and ask for solutions.
//!
//! Backend selection is via `SolverChoice::from_env()` (env var
//! `RAPIDFEM_SOLVER=auto|pardiso|rslab`, default `auto`). The auto order is
//! PARDISO → rslab.

use num_complex::Complex64 as C64;

pub mod pardiso;
pub mod rslab_ldlt;

/// Sparse direct solver for a complex-symmetric matrix.
///
/// Input convention: full COO triplets `(rows, cols, vals)` of dimension `n`.
/// Off-diagonal entries appear in both halves (the FEM assembly produces them
/// that way naturally); each backend filters to the form it prefers
/// (upper-CSR for PARDISO, lower-CSC for rslab). The same factorisation is
/// reused for many RHS via `solve`.
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
    /// Try PARDISO → rslab in that order.
    Auto,
    /// Intel MKL PARDISO (dynamic load of `mkl_rt`).
    Pardiso,
    /// Pure-Rust rslab complex-symmetric LDLᵀ.
    Rslab,
}

impl SolverChoice {
    /// Read `RAPIDFEM_SOLVER` and parse into a choice. Unknown values
    /// (including the retired `faer` / `accelerate`) warn and fall back to
    /// `Auto`.
    pub fn from_env() -> Self {
        match std::env::var("RAPIDFEM_SOLVER").ok().as_deref() {
            Some(s) => match s.to_ascii_lowercase().as_str() {
                "pardiso" => Self::Pardiso,
                "rslab" => Self::Rslab,
                "auto" => Self::Auto,
                other => {
                    eprintln!(
                        "  solver: RAPIDFEM_SOLVER={other:?} is not a backend \
                         (valid: auto|pardiso|rslab), using auto"
                    );
                    Self::Auto
                }
            },
            None => Self::Auto,
        }
    }
}

/// Instantiate a solver matching `choice`, falling back to rslab when PARDISO
/// isn't available at runtime. Logs the actual choice to stderr.
pub fn pick(choice: SolverChoice) -> Box<dyn SparseSolver> {
    let try_pardiso = || pardiso::PardisoSolver::try_new()
        .map(|s| Box::new(s) as Box<dyn SparseSolver>);
    let make_rslab = || Box::new(rslab_ldlt::RslabSolver::new()) as Box<dyn SparseSolver>;

    let solver = match choice {
        SolverChoice::Pardiso => try_pardiso().unwrap_or_else(|| {
            eprintln!("  solver: PARDISO requested but unavailable, falling back to rslab");
            make_rslab()
        }),
        SolverChoice::Rslab => make_rslab(),
        SolverChoice::Auto => try_pardiso().unwrap_or_else(make_rslab),
    };
    eprintln!("  solver: using {}", solver.name());
    solver
}
