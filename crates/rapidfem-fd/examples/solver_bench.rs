// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Solver-backend micro-benchmark on a real rapidfem system matrix.
//!
//! Assembles K = E − k0²·B on a small structured tet box (Nédélec-2, lossy
//! dielectric fill so K is genuinely complex symmetric), then times per
//! available backend (rslab always, PARDISO when `mkl_rt` is installed):
//! factorize, refactorize (numeric-only, simulating a frequency sweep) and
//! solve, and checks the residual ‖Kx−b‖/‖b‖ against the COO matvec.
//!
//! Historical reference (M3 MacBook Air, n = 21 424, this benchmark, commit
//! feat/rslab-solver): faer LU 2 338 ms factor / 2 421 ms per sweep
//! frequency; Apple Accelerate real-block LDLᵀ 1 035 / 1 031 ms; rslab
//! 1 717 / 344 ms — the numbers that retired the faer and Accelerate
//! backends.
//!
//! Deliberately small (resource-friendly): `NX` controls the mesh; NX=8 is
//! ~3k tets / ~25k complex DOFs. Run:
//!
//!   cargo run --release -j 4 -p rapidfem-fd --example solver_bench [NX]

use num_complex::Complex64 as C64;
use rapidfem_core::mesh::Mesh;
use rapidfem_fd::basis::NedelecBasis;
use rapidfem_fd::solver::SparseSolver;
use rapidfem_fd::tet_assembly::assemble_global_matrices;
use std::time::Instant;

/// Structured box mesh: nx×ny×nz cells, each split into 6 tets (Kuhn).
fn structured_box(nx: usize, ny: usize, nz: usize, lx: f64, ly: f64, lz: f64) -> Mesh {
    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([
                    lx * i as f64 / nx as f64,
                    ly * j as f64 / ny as f64,
                    lz * k as f64 / nz as f64,
                ]);
            }
        }
    }
    let id = |i: usize, j: usize, k: usize| (k * (ny + 1) + j) * (nx + 1) + i;
    // Kuhn split of the unit cube into 6 tets around the main diagonal.
    const KUHN: [[usize; 4]; 6] = [
        [0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7],
        [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7],
    ];
    let mut tets = Vec::with_capacity(nx * ny * nz * 6);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = [
                    id(i, j, k), id(i + 1, j, k), id(i, j + 1, k), id(i + 1, j + 1, k),
                    id(i, j, k + 1), id(i + 1, j, k + 1), id(i, j + 1, k + 1), id(i + 1, j + 1, k + 1),
                ];
                for t in &KUHN {
                    tets.push([c[t[0]], c[t[1]], c[t[2]], c[t[3]]]);
                }
            }
        }
    }
    Mesh::from_tets(nodes, tets)
}

/// COO matvec y = K·x for the residual check.
fn coo_matvec(n: usize, rows: &[usize], cols: &[usize], vals: &[C64], x: &[C64]) -> Vec<C64> {
    let mut y = vec![C64::new(0.0, 0.0); n];
    for k in 0..rows.len() {
        y[rows[k]] += vals[k] * x[cols[k]];
    }
    y
}

fn bench_backend(
    name: &str,
    solver: &mut dyn SparseSolver,
    n: usize,
    rows: &[usize],
    cols: &[usize],
    k_freqs: &[Vec<C64>],
    b: &[C64],
) {
    // Cold factorize at the first frequency.
    let t0 = Instant::now();
    if let Err(e) = solver.factorize(n, rows, cols, &k_freqs[0]) {
        println!("{name:32} FACTORIZE FAILED: {e}");
        return;
    }
    let t_factor = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let x = match solver.solve(b) {
        Ok(x) => x,
        Err(e) => { println!("{name:32} SOLVE FAILED: {e}"); return; }
    };
    let t_solve = t1.elapsed().as_secs_f64();

    let r = coo_matvec(n, rows, cols, &k_freqs[0], &x);
    let rnorm: f64 = r.iter().zip(b).map(|(ri, bi)| (ri - bi).norm_sqr()).sum::<f64>().sqrt();
    let bnorm: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    let resid = rnorm / bnorm;

    // Sweep: numeric refactorize + solve at the remaining frequencies.
    let t2 = Instant::now();
    let mut ok = true;
    for kv in &k_freqs[1..] {
        if let Err(e) = solver.refactorize(n, rows, cols, kv) {
            println!("{name:32} REFACTORIZE FAILED: {e}");
            ok = false;
            break;
        }
        if solver.solve(b).is_err() { ok = false; break; }
    }
    let n_re = k_freqs.len() - 1;
    let t_refactor = if ok && n_re > 0 { t2.elapsed().as_secs_f64() / n_re as f64 } else { f64::NAN };

    println!(
        "{name:32} factor {:8.1} ms   refactor+solve {:8.1} ms/freq   solve {:6.1} ms   resid {:.2e}",
        t_factor * 1e3, t_refactor * 1e3, t_solve * 1e3, resid
    );
}

fn main() {
    let nx: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let mesh = structured_box(nx, nx, nx, 0.02, 0.02, 0.02);
    let basis = NedelecBasis::new(&mesh);
    let n = basis.n_field;
    println!(
        "mesh: {} tets, {} edges, {} tris → n_field = {}",
        mesh.n_tets(), mesh.n_edges(), mesh.n_tris(), n
    );

    // Lossy dielectric fill → genuinely complex-symmetric K.
    let er_t: [[C64; 3]; 3] = [
        [C64::new(4.0, -0.04), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(4.0, -0.04), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(4.0, -0.04)],
    ];
    let ur_t: [[C64; 3]; 3] = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
    ];
    let er = vec![er_t; mesh.n_tets()];
    let ur = vec![ur_t; mesh.n_tets()];

    let t0 = Instant::now();
    let (rows, cols, data_e, data_b) = assemble_global_matrices(&mesh, &basis, &er, &ur);
    println!(
        "assembled E,B in {:.1} ms ({} COO entries)",
        t0.elapsed().as_secs_f64() * 1e3, rows.len()
    );

    // Three "frequencies" around 10 GHz on the 2 cm box (k0·h ≈ realistic),
    // K(f) = E − k0²·B, same pattern, different values — like the FD sweep.
    let c0 = 299_792_458.0;
    let k_freqs: Vec<Vec<C64>> = [9.5e9, 10.0e9, 10.5e9]
        .iter()
        .map(|f| {
            let k0 = 2.0 * std::f64::consts::PI * f / c0;
            let k0sq = C64::from(k0 * k0);
            (0..rows.len()).map(|i| data_e[i] - k0sq * data_b[i]).collect()
        })
        .collect();

    // Deterministic RHS.
    let b: Vec<C64> = (0..n)
        .map(|i| C64::new((((i * 7 + 3) % 101) as f64 / 101.0) - 0.5,
                          (((i * 13 + 5) % 89) as f64 / 89.0) - 0.5))
        .collect();

    println!("\nbackend                          (n = {n}, 3-point sweep)");
    if let Some(mut pardiso) = rapidfem_fd::solver::pardiso::PardisoSolver::try_new() {
        bench_backend("PARDISO (mtype 6)", &mut pardiso, n, &rows, &cols, &k_freqs, &b);
    } else {
        println!("PARDISO                          not available (no mkl_rt), skipped");
    }

    let mut rs = rapidfem_fd::solver::rslab_ldlt::RslabSolver::new();
    bench_backend("rslab LDLᵀ (complex-symmetric)", &mut rs, n, &rows, &cols, &k_freqs, &b);
}
