// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Sparse symmetric direct solve on the vendored rslab LDLᵀ (Bunch-Kaufman),
//! the one factorisation path of rapidfem: the driven sweep, the eigenmode
//! shift-invert and the port-mode shift-invert all run through
//! [`SymmetricSolver`], real (`f64`) or complex symmetric (`Complex64`).
//!
//! Callers hand over full COO triplets (off-diagonal entries in both halves,
//! as an assembly produces them); they are filtered to the lower triangle
//! (rslab's `CscMatrix` convention, duplicates summed by `from_triplets`).
//! rslab equilibrates the matrix itself (`ScalingStrategy::OnePassInfNorm`),
//! so callers pass the raw system.
//!
//! Pattern reuse: the first `factorize` runs the symbolic analysis (the
//! ordering race; the worker count comes from the calibration when
//! `install_diagnose` has run) and caches it with the settings; `refactorize`
//! reuses both and redoes only the numeric phase, in place: the new factor
//! is written into the old one's buffers and keeps its solve schedule and
//! kernel scratch (rslab's `refactor`, the same bits as a fresh factor),
//! which is what a frequency sweep or a series of eigen shifts on one
//! pattern needs. rslab validates that the pattern (n, nnz) is unchanged and
//! errors otherwise; `refactorize` then falls back to a fresh `factorize`
//! instead of solving on a stale symbolic.
//!
//! Before any numeric work, rslab's `MemoryPlan` (from the symbolic structure,
//! the worker count and the number of right-hand sides) is checked against
//! the machine's RAM, so a factorisation too big for the machine fails fast
//! with a clear message instead of driving it into swap mid-sweep.

use rslab::{CscMatrix, Inertia, LdltSolver, LdltSymbolic, OrderingMethod, Scalar, SolverSettings};

/// Refuse to factor when the estimated transient peak exceeds this fraction
/// of TOTAL system RAM. Headroom for the OS, the assembly buffers and the
/// caller's field data; beyond it the machine swaps long before OOM.
const MEM_BUDGET_FRACTION: f64 = 0.8;

/// A-priori memory gate: plan the factorisation and a solve of `nrhs`
/// right-hand sides under `settings` and error out (before any numeric work)
/// if the peak exceeds the budget. Returns the log line describing the plan.
fn check_memory<T: Scalar>(
    sym: &LdltSymbolic,
    settings: &SolverSettings,
    nrhs: usize,
) -> Result<String, String> {
    let plan = sym.memory_plan::<T>(settings, nrhs);
    let peak = plan.peak_bytes();
    let hw = rslab::tuning::HardwareInfo::probe();
    let budget = (hw.total_ram_bytes as f64 * MEM_BUDGET_FRACTION) as u64;
    let line = format!(
        "factor nnz {:.2e}, est. peak {:.0} MB at {} threads, {} rhs (RAM {:.0} MB, {:.0} MB free)",
        sym.estimate_memory::<T>().factor_nnz as f64,
        peak as f64 / 1e6,
        plan.threads,
        plan.nrhs,
        hw.total_ram_bytes as f64 / 1e6,
        hw.available_ram_bytes as f64 / 1e6,
    );
    if peak > budget {
        return Err(format!(
            "rslab: estimated factorisation peak {:.0} MB exceeds {:.0}% of system \
             RAM ({:.0} MB), refine the mesh less, or run on a bigger machine \
             ({line})",
            peak as f64 / 1e6,
            MEM_BUDGET_FRACTION * 100.0,
            hw.total_ram_bytes as f64 / 1e6,
        ));
    }
    Ok(line)
}

/// Factor-once, solve-many sparse symmetric solver (see the module docs).
pub struct SymmetricSolver<T: Scalar> {
    n: usize,
    symbolic: Option<(LdltSymbolic, SolverSettings)>,
    solver: Option<LdltSolver<T>>,
    /// rslab's a-priori estimate of the factorisation: (flops, factor nnz).
    estimate: Option<(u64, u64)>,
    /// Right-hand sides per solve, for the memory plan.
    nrhs: usize,
    // Lower-triangle triplet buffers, reused across refactorizations.
    lo_rows: Vec<usize>,
    lo_cols: Vec<usize>,
    lo_vals: Vec<T>,
}

impl<T: Scalar> SymmetricSolver<T> {
    pub fn new() -> Self {
        Self { n: 0, symbolic: None, solver: None, estimate: None, nrhs: 1,
               lo_rows: Vec::new(), lo_cols: Vec::new(), lo_vals: Vec::new() }
    }

    /// The number of right-hand sides solved at once (the driven ports),
    /// so the memory plan counts their work vectors.
    pub fn expect_rhs(&mut self, nrhs: usize) {
        self.nrhs = nrhs.max(1);
    }

    /// Filter the full COO triplets to the lower triangle (row ≥ col) into the
    /// reused buffers and build rslab's CSC (duplicates summed there).
    fn build_matrix(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[T],
    ) -> Result<CscMatrix<T>, String> {
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

impl<T: Scalar> Default for SymmetricSolver<T> {
    fn default() -> Self { Self::new() }
}

impl<T: Scalar> SymmetricSolver<T> {
    /// Symbolic analysis plus numeric factorisation from full COO triplets.
    /// Resets any previously stored factor.
    pub fn factorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[T],
    ) -> Result<(), String> {
        let a = self.build_matrix(n, rows, cols, vals)?;
        // The default is the ordering race; RAPIDFEM_RSLAB_ORDERING=
        // amd|amf|metis|rcm pins one, for experiments, not correctness.
        let mut settings = SolverSettings::default();
        if let Ok(v) = std::env::var("RAPIDFEM_RSLAB_ORDERING") {
            match v.to_ascii_lowercase().as_str() {
                "amd" => settings.ordering.method = OrderingMethod::Amd,
                "amf" => settings.ordering.method = OrderingMethod::Amf,
                "metis" => settings.ordering.method = OrderingMethod::MetisND,
                "rcm" => settings.ordering.method = OrderingMethod::Rcm,
                other => eprintln!("  rslab: unknown RAPIDFEM_RSLAB_ORDERING={other:?}, ignoring"),
            }
        }
        let sym = LdltSymbolic::analyze(&a, &settings)
            .map_err(|e| format!("rslab analyze: {e:?}"))?;
        let mem_line = check_memory::<T>(&sym, &settings, self.nrhs)?;
        let est = sym.estimate_memory::<T>();
        self.estimate = Some((est.factor_flops, est.factor_nnz));
        eprintln!(
            "  rslab: {:?}, {mem_line}, est. {:.2e} flops",
            settings.ordering.method,
            sym.estimate_memory::<T>().factor_flops as f64,
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

    /// Numeric-only refactor on the cached symbolic and settings. Falls
    /// back to a full `factorize` when no symbolic is cached or the sparsity
    /// pattern changed (rslab rejects a pattern mismatch explicitly).
    pub fn refactorize(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[T],
    ) -> Result<(), String> {
        if self.symbolic.is_none() || self.n != n {
            return self.factorize(n, rows, cols, vals);
        }
        let a = self.build_matrix(n, rows, cols, vals)?;
        let (sym, settings) = self.symbolic.as_ref().unwrap();
        let done = match self.solver.as_mut() {
            Some(solver) => sym.refactor(&a, settings, solver).is_ok(),
            None => match sym.factor(&a, settings) {
                Ok(solver) => {
                    self.solver = Some(solver);
                    true
                }
                Err(_) => false,
            },
        };
        if !done {
            // Pattern drift (e.g. a Robin entry that is exactly zero at one
            // frequency): redo the analysis instead of failing the sweep.
            return self.factorize(n, rows, cols, vals);
        }
        let perturbed = self.solver.as_ref().map_or(0, |s| s.n_perturbed());
        if perturbed > 0 {
            eprintln!("  rslab: WARNING {perturbed} perturbed pivots on refactorize");
        }
        Ok(())
    }

    /// Solve `K · x = b` on the cached factorisation.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, String> {
        let solver = self.solver.as_ref()
            .ok_or_else(|| "rslab: solve before factorize".to_string())?;
        if b.len() != self.n {
            return Err(format!("rslab: RHS length {} ≠ n = {}", b.len(), self.n));
        }
        solver.solve(b).map_err(|e| format!("rslab solve: {e:?}"))
    }

    /// Batched multi-RHS solve: one factor traversal for all RHS. Falls back
    /// to sequential solves when the staging buffers (~3·n·nrhs
    /// complex values: packed input, equilibrated copy, output) would not
    /// comfortably fit in the currently AVAILABLE RAM.
    pub fn solve_many(&self, bs: &[Vec<T>]) -> Result<Vec<Vec<T>>, String> {
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
        let staging = 3 * n * nrhs * std::mem::size_of::<T>();
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
        // rslab takes the block column-major: the right-hand sides back to back.
        let x = solver.solve_many(&bs.concat(), nrhs)
            .map_err(|e| format!("rslab solve_many: {e:?}"))?;
        Ok(x.chunks(n).map(<[T]>::to_vec).collect())
    }

    /// Solve a nearby system `A' X = B` by COCG, preconditioned with the
    /// current factorisation of `A` (a neighbouring frequency of a sweep: the
    /// same pattern, slightly different values). Returns `None` when there is
    /// no factorisation yet or any right-hand side misses the relative
    /// residual `tol` within `max_iter` iterations; the caller then refactors.
    /// Also returns the largest iteration count.
    pub fn solve_nearby(
        &mut self,
        n: usize,
        rows: &[usize],
        cols: &[usize],
        vals: &[T],
        bs: &[Vec<T>],
        tol: f64,
        max_iter: usize,
    ) -> Option<(Vec<Vec<T>>, usize)> {
        if self.solver.is_none() || self.n != n {
            return None;
        }
        let a = self.build_matrix(n, rows, cols, vals).ok()?;
        let precond = self.solver.as_ref()?;
        let settings = rslab::KrylovSettings { tol, max_iter, ..Default::default() };
        let mut xs = Vec::with_capacity(bs.len());
        let mut iters = 0;
        for b in bs {
            let r = rslab::cocg(&a, b, precond, &settings).ok()?;
            if !r.converged {
                return None;
            }
            iters = iters.max(r.iters);
            xs.push(r.x);
        }
        Some((xs, iters))
    }

    /// rslab's a-priori estimate of the factorisation, `(flops, factor nnz)`,
    /// from the symbolic analysis: deterministic, unlike a measured time.
    pub fn factor_estimate(&self) -> Option<(u64, u64)> {
        self.estimate
    }

    /// Inertia (positive, negative, zero pivots) of the last factorisation.
    /// For a real symmetric matrix `A - sigma B` with `B` positive definite
    /// this counts the eigenvalues below `sigma` (Sylvester's law); it has no
    /// such meaning for a complex-symmetric matrix.
    pub fn inertia(&self) -> Option<Inertia> {
        self.solver.as_ref().map(|s| s.inertia().clone())
    }

    /// Backend name, for logs.
    pub fn name(&self) -> &'static str { "rslab LDLᵀ" }
}

/// Compress the unknowns `0..n` that are not `fixed` into a contiguous
/// range: returns the free indices in order and the map from a full index to
/// its free position (`usize::MAX` for a fixed one).
pub fn free_index(n: usize, fixed: impl Fn(usize) -> bool) -> (Vec<usize>, Vec<usize>) {
    let free: Vec<usize> = (0..n).filter(|&i| !fixed(i)).collect();
    let mut to_free = vec![usize::MAX; n];
    for (r, &i) in free.iter().enumerate() {
        to_free[i] = r;
    }
    (free, to_free)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64 as C64;

    /// Round-trip on a tiny complex-symmetric system, plus a numeric-only
    /// refactorize on scaled values.
    #[test]
    fn solve_3x3_round_trip_and_refactor() {
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let vals = vec![
            C64::new(2.0, 0.0),  C64::new(1.0, 0.5),
            C64::new(1.0, 0.5),  C64::new(4.0, -1.0), C64::new(0.0, 0.3),
            C64::new(0.0, 0.3),  C64::new(3.0, 0.2),
        ];
        let mut solver = SymmetricSolver::<C64>::new();
        solver.factorize(3, &rows, &cols, &vals).unwrap();

        let check = |solver: &mut SymmetricSolver<C64>, vals: &[C64]| {
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
        let mut solver = SymmetricSolver::<C64>::new();
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

    /// A nearby system (values perturbed by a few percent, same pattern)
    /// solved by COCG on the old factorisation matches its direct solve.
    #[test]
    fn solve_nearby_matches_the_direct_solve() {
        // 1D Helmholtz-like tridiagonal, complex symmetric.
        let n = 200;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        let system = |k2: f64| -> Vec<C64> {
            let mut v = Vec::new();
            for i in 0..n {
                v.push(C64::new(2.0 - k2, 0.01));
                if i + 1 < n {
                    v.push(C64::new(-1.0, 0.0));
                    v.push(C64::new(-1.0, 0.0));
                }
            }
            v
        };
        for i in 0..n {
            rows.push(i); cols.push(i);
            if i + 1 < n {
                rows.push(i); cols.push(i + 1);
                rows.push(i + 1); cols.push(i);
            }
        }
        vals.extend(system(0.30));
        let mut solver = SymmetricSolver::<C64>::new();
        solver.factorize(n, &rows, &cols, &vals).unwrap();

        let near = system(0.31);
        let b: Vec<C64> = (0..n).map(|i| C64::new(1.0 + i as f64 * 0.01, 0.0)).collect();
        let (xs, iters) = solver
            .solve_nearby(n, &rows, &cols, &near, std::slice::from_ref(&b), 1e-12, 200)
            .expect("COCG must converge on a nearby system");
        let mut direct = SymmetricSolver::<C64>::new();
        direct.factorize(n, &rows, &cols, &near).unwrap();
        let want = direct.solve(&b).unwrap();
        let err: f64 = xs[0].iter().zip(&want).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt();
        let scale: f64 = want.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        assert!(err / scale < 1e-9, "rel err {} after {iters} iterations", err / scale);
    }
}
