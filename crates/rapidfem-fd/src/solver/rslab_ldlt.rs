// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! `SparseSolver` impl backed by rslab's complex-symmetric LDLᵀ (Bunch-Kaufman).
//!
//! rslab factors the complex-symmetric `A` directly (PARDISO mtype-6
//! analogue), so unlike the Accelerate backend no 2N real-block reformulation
//! is needed, and unlike faer no general-LU on the full pattern. The trait's
//! full COO triplets are filtered to the lower triangle (rslab's `CscMatrix`
//! convention, duplicates summed by `from_triplets`).
//!
//! Sweep amortisation: the first `factorize` runs `LdltSolver::tuned` (symbolic
//! analysis + auto-tuned settings) and caches both; `refactorize` reuses them
//! and only redoes the numeric phase. rslab validates that the pattern (n,
//! nnz) is unchanged and errors otherwise — `refactorize` then falls back to a
//! fresh `factorize` instead of solving on a stale symbolic.
//!
//! Solver-in-the-loop: rslab's a-priori `MemoryEstimate` (deterministic, from
//! the symbolic structure alone) is checked against the machine's RAM BEFORE
//! any numeric work — a too-big factorisation fails fast with a clear message
//! instead of driving the machine into swap/OOM mid-sweep. The estimate and
//! the post-factor diagnostics (factor nnz, perturbed pivots) go to the log.

use num_complex::Complex64 as C64;
use rslab::{CscMatrix, FactorMethod, LdltSolver, LdltSymbolic, SolverSettings};
use super::SparseSolver;

/// Refuse to factor when the estimated transient peak exceeds this fraction
/// of TOTAL system RAM. Headroom for the OS, the assembly buffers and the
/// caller's field data; beyond it the machine swaps long before OOM.
const MEM_BUDGET_FRACTION: f64 = 0.8;

/// A-priori memory gate: estimate the factorisation's transient peak for the
/// tuned method and error out (before any numeric work) if it exceeds the
/// budget. Returns the log line describing the estimate.
fn check_memory(sym: &LdltSymbolic, settings: &SolverSettings) -> Result<String, String> {
    let est = sym.estimate_memory::<C64>();
    // The transient peak differs per factorisation schedule; compare the one
    // the tuner actually picked (multifrontal holds more than left-looking).
    let peak = match settings.method {
        FactorMethod::Multifrontal => est.mf_transient_peak_bytes,
        _ => est.transient_peak_bytes,
    };
    let hw = rslab::tuning::HardwareInfo::probe();
    let budget = (hw.total_ram_bytes as f64 * MEM_BUDGET_FRACTION) as u64;
    let line = format!(
        "factor nnz {:.2e}, est. peak {:.0} MB (RAM {:.0} MB, {:.0} MB free)",
        est.factor_nnz as f64,
        peak as f64 / 1e6,
        hw.total_ram_bytes as f64 / 1e6,
        hw.available_ram_bytes as f64 / 1e6,
    );
    if peak > budget {
        return Err(format!(
            "rslab: estimated factorisation peak {:.0} MB exceeds {:.0}% of system \
             RAM ({:.0} MB) — refine the mesh less, or run on a bigger machine \
             ({line})",
            peak as f64 / 1e6,
            MEM_BUDGET_FRACTION * 100.0,
            hw.total_ram_bytes as f64 / 1e6,
        ));
    }
    Ok(line)
}

pub struct RslabSolver {
    n: usize,
    symbolic: Option<(LdltSymbolic, SolverSettings)>,
    solver: Option<LdltSolver<C64>>,
    // Lower-triangle triplet buffers, reused across refactorizations.
    lo_rows: Vec<usize>,
    lo_cols: Vec<usize>,
    lo_vals: Vec<C64>,
}

impl RslabSolver {
    pub fn new() -> Self {
        Self { n: 0, symbolic: None, solver: None,
               lo_rows: Vec::new(), lo_cols: Vec::new(), lo_vals: Vec::new() }
    }

    /// Filter the full COO triplets to the lower triangle (row ≥ col) into the
    /// reused buffers and build rslab's CSC (duplicates summed there).
    fn build_matrix(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[C64],
    ) -> Result<CscMatrix<C64>, String> {
        self.lo_rows.clear();
        self.lo_cols.clear();
        self.lo_vals.clear();
        for i in 0..rows.len() {
            if rows[i] >= cols[i] {
                self.lo_rows.push(rows[i]);
                self.lo_cols.push(cols[i]);
                self.lo_vals.push(vals[i]);
            }
        }
        CscMatrix::from_triplets(n, &self.lo_rows, &self.lo_cols, &self.lo_vals)
            .map_err(|e| format!("rslab matrix build: {e:?}"))
    }
}

impl Default for RslabSolver {
    fn default() -> Self { Self::new() }
}

impl SparseSolver for RslabSolver {
    fn factorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[C64],
    ) -> Result<(), String> {
        let a = self.build_matrix(n, rows, cols, vals)?;
        let (sym, settings) = LdltSolver::<C64>::tuned(&a, rslab::DEFAULT_TUNE_WEIGHT)
            .map_err(|e| format!("rslab analyze/tune: {e:?}"))?;
        let mem_line = check_memory(&sym, &settings)?;
        eprintln!("  rslab: {mem_line}");
        let solver = sym.factor(&a, &settings)
            .map_err(|e| format!("rslab factor: {e:?}"))?;
        if solver.n_perturbed() > 0 {
            eprintln!(
                "  rslab: WARNING {} perturbed pivots (near-singular system?), \
                 residuals may degrade",
                solver.n_perturbed()
            );
        }
        self.n = n;
        self.symbolic = Some((sym, settings));
        self.solver = Some(solver);
        Ok(())
    }

    /// Numeric-only refactor on the cached symbolic + tuned settings. Falls
    /// back to a full `factorize` when no symbolic is cached or the sparsity
    /// pattern changed (rslab rejects a pattern mismatch explicitly).
    fn refactorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[C64],
    ) -> Result<(), String> {
        if self.symbolic.is_none() || self.n != n {
            return self.factorize(n, rows, cols, vals);
        }
        let a = self.build_matrix(n, rows, cols, vals)?;
        let (sym, settings) = self.symbolic.as_ref().unwrap();
        match sym.factor(&a, settings) {
            Ok(solver) => {
                if solver.n_perturbed() > 0 {
                    eprintln!(
                        "  rslab: WARNING {} perturbed pivots on refactorize",
                        solver.n_perturbed()
                    );
                }
                self.solver = Some(solver);
                Ok(())
            }
            // Pattern drift (e.g. a Robin entry that is exactly zero at one
            // frequency): redo the analysis instead of failing the sweep.
            Err(_) => self.factorize(n, rows, cols, vals),
        }
    }

    fn solve(&mut self, b: &[C64]) -> Result<Vec<C64>, String> {
        let solver = self.solver.as_ref()
            .ok_or_else(|| "rslab: solve before factorize".to_string())?;
        if b.len() != self.n {
            return Err(format!("rslab: RHS length {} ≠ n = {}", b.len(), self.n));
        }
        solver.solve(b).map_err(|e| format!("rslab solve: {e:?}"))
    }

    fn name(&self) -> &'static str { "rslab LDLᵀ" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip on the same tiny complex-symmetric system the Accelerate
    /// backend test uses, plus a numeric-only refactorize on scaled values.
    #[test]
    fn solve_3x3_round_trip_and_refactor() {
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let vals = vec![
            C64::new(2.0, 0.0),  C64::new(1.0, 0.5),
            C64::new(1.0, 0.5),  C64::new(4.0, -1.0), C64::new(0.0, 0.3),
            C64::new(0.0, 0.3),  C64::new(3.0, 0.2),
        ];
        let mut solver = RslabSolver::new();
        solver.factorize(3, &rows, &cols, &vals).unwrap();

        let check = |solver: &mut RslabSolver, vals: &[C64]| {
            let x = [C64::new(1.0, 0.0), C64::new(0.5, -0.7), C64::new(-0.3, 0.1)];
            let mut b = [C64::new(0.0, 0.0); 3];
            for k in 0..rows.len() {
                b[rows[k]] += vals[k] * x[cols[k]];
            }
            let x_back = solver.solve(&b).unwrap();
            let err: f64 = x_back.iter().zip(x.iter())
                .map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt();
            let xn: f64 = x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            assert!(err / xn < 1e-10, "rel err {} too large", err / xn);
        };
        check(&mut solver, &vals);

        let vals2: Vec<C64> = vals.iter().map(|v| v * C64::new(1.3, 0.1)).collect();
        solver.refactorize(3, &rows, &cols, &vals2).unwrap();
        check(&mut solver, &vals2);
    }
}
