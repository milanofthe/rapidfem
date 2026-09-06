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
use rslab::OrderingMethod;
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
        dump_matrix(&a);
        let (mut sym, mut settings) = LdltSolver::<C64>::tuned(&a)
            .map_err(|e| format!("rslab analyze/tune: {e:?}"))?;
        // Escape hatches over the heuristic pick (rslab >= 0.19: deterministic
        // tuned() = adaptive ordering + exact ND bakeoff + calibrated worker
        // count; the ML tuner moved to the opt-in tuned_model). These are for
        // experiments, not correctness.
        // RAPIDFEM_RSLAB_ORDERING=amd|amf|metis|scotch, RAPIDFEM_RSLAB_METHOD=ll|mf.
        if let Ok(v) = std::env::var("RAPIDFEM_RSLAB_ORDERING") {
            let ord = match v.to_ascii_lowercase().as_str() {
                "amd" => Some(OrderingMethod::Amd),
                "amf" => Some(OrderingMethod::Amf),
                "metis" => Some(OrderingMethod::MetisND),
                other => {
                    eprintln!("  rslab: unknown RAPIDFEM_RSLAB_ORDERING={other:?}, ignoring");
                    None
                }
            };
            if let Some(ord) = ord {
                if ord != settings.ordering {
                    settings.ordering = ord;
                    sym = LdltSymbolic::analyze_with(&a, &settings)
                        .map_err(|e| format!("rslab analyze ({v}): {e:?}"))?;
                }
            }
        }
        if let Ok(v) = std::env::var("RAPIDFEM_RSLAB_METHOD") {
            match v.to_ascii_lowercase().as_str() {
                "ll" => settings.method = FactorMethod::LeftLooking,
                "mf" => settings.method = FactorMethod::Multifrontal,
                other => eprintln!("  rslab: unknown RAPIDFEM_RSLAB_METHOD={other:?}, ignoring"),
            }
        }
        let mem_line = check_memory(&sym, &settings)?;
        eprintln!(
            "  rslab: {:?}/{:?}, {mem_line}, est. {:.2e} flops",
            settings.method, settings.ordering,
            sym.estimate_memory::<C64>().factor_flops as f64,
        );
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

    /// Batched multi-RHS solve: one factor traversal for all RHS. Falls back
    /// to sequential solves when the row-major staging buffers (~3·n·nrhs
    /// complex values: packed input, equilibrated copy, output) would not
    /// comfortably fit in the currently AVAILABLE RAM.
    fn solve_many(&mut self, bs: &[Vec<C64>]) -> Result<Vec<Vec<C64>>, String> {
        let nrhs = bs.len();
        if nrhs <= 1 || self.solver.is_none() {
            return bs.iter().map(|b| self.solve(b)).collect();
        }
        let n = self.n;
        for b in bs {
            if b.len() != n {
                return Err(format!("rslab: RHS length {} ≠ n = {}", b.len(), n));
            }
        }
        let staging = 3 * n * nrhs * std::mem::size_of::<C64>();
        let hw = rslab::tuning::HardwareInfo::probe();
        if staging as u64 > hw.available_ram_bytes / 2 {
            eprintln!(
                "  rslab: batched solve would stage ~{:.0} MB (>{:.0} MB free/2), \
                 solving {} RHS sequentially",
                staging as f64 / 1e6,
                hw.available_ram_bytes as f64 / 1e6,
                nrhs,
            );
            return bs.iter().map(|b| self.solve(b)).collect();
        }
        let solver = self.solver.as_ref().unwrap();
        // Pack row-major n×nrhs (rslab's solve_many layout), solve, unpack.
        let mut packed = vec![C64::new(0.0, 0.0); n * nrhs];
        for (c, b) in bs.iter().enumerate() {
            for i in 0..n {
                packed[i * nrhs + c] = b[i];
            }
        }
        let x = solver.solve_many(&packed, nrhs)
            .map_err(|e| format!("rslab solve_many: {e:?}"))?;
        Ok((0..nrhs)
            .map(|c| (0..n).map(|i| x[i * nrhs + c]).collect())
            .collect())
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

    /// Batched solve must reproduce the sequential per-RHS solutions.
    #[test]
    fn solve_many_matches_sequential() {
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let vals = vec![
            C64::new(2.0, 0.0),  C64::new(1.0, 0.5),
            C64::new(1.0, 0.5),  C64::new(4.0, -1.0), C64::new(0.0, 0.3),
            C64::new(0.0, 0.3),  C64::new(3.0, 0.2),
        ];
        let mut solver = RslabSolver::new();
        solver.factorize(3, &rows, &cols, &vals).unwrap();

        let bs: Vec<Vec<C64>> = (0..3)
            .map(|k| (0..3)
                .map(|i| C64::new((i + k) as f64 + 0.5, (i * k) as f64 - 0.25))
                .collect())
            .collect();
        let batched = solver.solve_many(&bs).unwrap();
        for (b, xb) in bs.iter().zip(&batched) {
            let xs = solver.solve(b).unwrap();
            let diff: f64 = xs.iter().zip(xb)
                .map(|(a, c)| (a - c).norm_sqr()).sum::<f64>().sqrt();
            assert!(diff < 1e-12, "batched ≠ sequential, diff {diff}");
        }
    }
}

/// `RAPIDFEM_DUMP_MATRIX=<dir>`: write the first assembled system matrix of
/// the process as Matrix Market (`<dir>/rapidfem_<n>.mtx`, complex symmetric
/// coordinate, lower triangle) for solver benchmarks on real FEM matrices.
fn dump_matrix(a: &CscMatrix<C64>) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DONE: AtomicBool = AtomicBool::new(false);
    let Ok(dir) = std::env::var("RAPIDFEM_DUMP_MATRIX") else {
        return;
    };
    if DONE.swap(true, Ordering::Relaxed) {
        return;
    }
    let path = std::path::Path::new(&dir).join(format!("rapidfem_{}.mtx", a.n));
    let mut out = String::new();
    out.push_str("%%MatrixMarket matrix coordinate complex symmetric\n");
    out.push_str(&format!("{} {} {}\n", a.n, a.n, a.row_idx.len()));
    for j in 0..a.n {
        for k in a.col_ptr[j]..a.col_ptr[j + 1] {
            let v = a.values[k];
            out.push_str(&format!("{} {} {:e} {:e}\n", a.row_idx[k] + 1, j + 1, v.re, v.im));
        }
    }
    match std::fs::write(&path, out) {
        Ok(()) => eprintln!("RAPIDFEM_DUMP_MATRIX: wrote {}", path.display()),
        Err(e) => eprintln!("RAPIDFEM_DUMP_MATRIX: cannot write {}: {e}", path.display()),
    }
}
