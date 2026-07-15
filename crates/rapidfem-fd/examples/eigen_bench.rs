// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Eigenmode-solver timing on a PEC cavity.
//!
//! The rewritten solver reorthogonalises the Lanczos basis fully, in the B inner
//! product, twice per step. That is O(m²·n) and is the price of not producing ghost
//! modes — but it is a price, so measure it against the factorisation it sits next to
//! rather than assume it is negligible.
//!
//!   cargo run --release -p rapidfem-fd --example eigen_bench [N] [MODES]

use rapidfem_fd::basis::NedelecBasis;
use rapidfem_fd::eigenmode::solve_eigenmode;
use rapidfem_fd::mesh::Mesh;

const C0: f64 = 299_792_458.0;

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
                    idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1),
                    idx(i, j + 1, k + 1),
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(6);
    let n_modes: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);
    let f_ghz: Option<f64> = args.get(3).and_then(|s| s.parse().ok());

    let (a, b, d): (f64, f64, f64) = (0.02286, 0.01016, 0.030);
    let mesh = box_mesh(a, b, d, 2 * n, n, 3 * n);
    let pec: Vec<usize> = (0..mesh.n_tris())
        .filter(|&t| mesh.tri_to_tet[t][1] == usize::MAX)
        .collect();
    let basis = NedelecBasis::new(&mesh);
    let f101 = 0.5 * C0 * ((1.0 / a).powi(2) + (1.0 / d).powi(2)).sqrt();
    let target = f_ghz.map(|g| g * 1e9).unwrap_or(f101);

    println!("cavity: {} tets, {} DOFs", mesh.n_tets(), basis.n_field);
    let t0 = std::time::Instant::now();
    let modes = solve_eigenmode(&mesh, &basis, &pec, None, target, n_modes).expect("solve");
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    for (i, m) in modes.iter().enumerate() {
        println!(
            "  mode {i}: {:.6} GHz, Q = {:.3e}, residual {:.2e}",
            m.frequency.re / 1e9,
            m.q_factor,
            m.residual
        );
    }
    println!("total: {ms:.1} ms   (closed-form TE101 = {:.6} GHz)", f101 / 1e9);
}
