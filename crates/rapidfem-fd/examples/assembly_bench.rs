// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Element-assembly micro-benchmark.
//!
//! Times ONLY `assemble_global_matrices`: the per-element basis construction and
//! the O(n²) stiffness/mass integration, scattered into the COO triplets. The
//! linear solve is `solver_bench`'s job; this one exists to guard the element
//! hot path across the basis refactor (docs/fd-basis-plan.md), which replaces a
//! fixed two-term / degree-2-monomial representation with a general exponent
//! multi-index and a variable term count.
//!
//! Run on both sides of a change and compare:
//!
//!   cargo run --release -j 4 -p rapidfem-fd --example assembly_bench [NX] [REPS]
//!
//! NX controls the mesh (NX=12 is ~10k tets). Deliberately modest so it does not
//! saturate the machine.

use num_complex::Complex64 as C64;
use rapidfem_fd::basis::NedelecBasis;
use rapidfem_fd::mesh::Mesh;
use rapidfem_fd::tet_assembly::assemble_global_matrices;

/// A structured box of NX³ cubes, each cut into six tetrahedra.
fn box_mesh(nx: usize) -> Mesh {
    let n = nx + 1;
    let idx = |i: usize, j: usize, k: usize| (i * n + j) * n + k;
    let h = 1.0 / nx as f64;
    let mut nodes = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                nodes.push([i as f64 * h, j as f64 * h, k as f64 * h]);
            }
        }
    }
    let mut tets = Vec::with_capacity(6 * nx * nx * nx);
    for i in 0..nx {
        for j in 0..nx {
            for k in 0..nx {
                // Kuhn subdivision of the cube: six tets on the main diagonal.
                let c = [
                    idx(i, j, k),
                    idx(i + 1, j, k),
                    idx(i + 1, j + 1, k),
                    idx(i, j + 1, k),
                    idx(i, j, k + 1),
                    idx(i + 1, j, k + 1),
                    idx(i + 1, j + 1, k + 1),
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
    let nx: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    let mesh = box_mesh(nx);
    let basis = NedelecBasis::new(&mesh);
    println!(
        "mesh: {} tets, {} edges, {} faces -> {} DOFs",
        mesh.n_tets(),
        mesh.n_edges(),
        mesh.n_tris(),
        basis.n_field
    );

    // A lossy anisotropic fill, so the complex tensor path is the one timed and
    // not a real-arithmetic fast path.
    let eps = [
        [C64::new(4.4, -0.09), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(4.1, -0.07), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(3.8, -0.05)],
    ];
    let mu = [
        [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
        [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
    ];
    let er = vec![eps; mesh.n_tets()];
    let ur = vec![mu; mesh.n_tets()];

    // One warm-up pass, then take the best of `reps` (the minimum is the least
    // noisy estimator here: scheduler noise only ever adds time).
    let _ = assemble_global_matrices(&mesh, &basis, &er, &ur);

    let mut best = f64::INFINITY;
    let mut checksum = 0.0_f64;
    for r in 0..reps {
        let t0 = std::time::Instant::now();
        let (_rows, _cols, de, db) = assemble_global_matrices(&mesh, &basis, &er, &ur);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        // Keep the result alive so nothing is optimised away, and carry a value
        // that would move if the assembly changed.
        checksum = de[0].norm() + db[de.len() / 2].norm();
        println!("  rep {r}: {ms:8.2} ms");
        best = best.min(ms);
    }
    println!("best: {best:.2} ms   ({:.0} tets/ms)", mesh.n_tets() as f64 / best);
    println!("checksum: {checksum:.17e}");
}
