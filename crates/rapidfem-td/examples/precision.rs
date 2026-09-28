// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Mixed-precision probe: how far the `Field`-precision build drifts from
//! the f64 reference.
//!
//! The TD backend is written against the `Field` scalar alias. With
//! `Field = f64` this run is the reference: it records one matvec output
//! and a propagated transient state to a side file. With `Field = f32` it
//! reads that reference back and reports the relative L2 error of each,
//! the numbers that set `GPU_REL_TOL` for the GPU path.
//!
//! ```text
//! # constants.rs: Field = f64
//! cargo run --release -p rapidfem-td --example precision
//! # constants.rs: Field = f32
//! cargo run --release -p rapidfem-td --example precision
//! ```
//!
//! The build mode is detected from `size_of::<Field>()` (8 = f64 reference
//! writer, 4 = f32 comparator), so no flag is needed.

use std::mem::size_of;

use rapidfem_td::constants::Field;
use rapidfem_td::explicit::LserkWorkspace;
use rapidfem_td::mesh_gen::structured_box;
use rapidfem_td::rhs::MaxwellOperator;

const STEPS: usize = 500;

fn ref_path() -> std::path::PathBuf {
    std::env::temp_dir().join("rapidfem_td_precision_ref.bin")
}

/// Relative L2 error of an f32-path result against the f64 reference.
fn rel_l2(got: &[Field], reference: &[f64]) -> f64 {
    let err: f64 = got
        .iter()
        .zip(reference)
        .map(|(&a, &b)| (a as f64 - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale: f64 = reference.iter().map(|b| b * b).sum::<f64>().sqrt();
    err / scale
}

fn main() {
    let mesh = structured_box(3, 3, 3, 1.0, 1.0, 1.0);
    let op = MaxwellOperator::new(&mesh, 2, 1.0); // upwind, the common case
    let n = op.n_dof();

    // A deterministic initial state.
    let y0: Vec<Field> =
        (0..n).map(|i| (0.2 + i as Field * 0.013).sin()).collect();

    // One matvec, the per-call operator error.
    let dy = op.apply(&y0);

    // Spectral radius by power iteration, for a sub-CFL explicit step.
    let mut v = y0.clone();
    let mut rho = 1.0_f32 as Field;
    for _ in 0..30 {
        let av = op.apply(&v);
        rho = av.iter().map(|x| x * x).sum::<Field>().sqrt();
        let inv = 1.0 / rho;
        for (vi, &a) in v.iter_mut().zip(&av) {
            *vi = a * inv;
        }
    }
    let dt = 1.0 / rho;

    // A propagated transient, the accumulated operator error over a run.
    let mut y = y0.clone();
    let mut ws = LserkWorkspace::new();
    for _ in 0..STEPS {
        ws.step_into(|x, ax| op.apply_into(x, ax), &mut y, dt);
    }

    println!(
        "precision probe, Field = {}-byte float",
        size_of::<Field>()
    );
    println!("  n_dof {n}, upwind flux, {STEPS} LSERK4 steps");

    let path = ref_path();
    if size_of::<Field>() == 8 {
        let mut bytes = Vec::with_capacity(2 * n * 8);
        for &v in dy.iter().chain(y.iter()) {
            bytes.extend_from_slice(&(v as f64).to_le_bytes());
        }
        std::fs::write(&path, bytes).expect("write reference");
        println!("  wrote f64 reference           {}", path.display());
    } else {
        match std::fs::read(&path) {
            Ok(bytes) => {
                let r: Vec<f64> = bytes
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                assert_eq!(r.len(), 2 * n, "reference dof count mismatch");
                println!(
                    "  rel L2 error, one matvec      {:.3e}",
                    rel_l2(&dy, &r[..n])
                );
                println!(
                    "  rel L2 error, {STEPS}-step transient {:.3e}",
                    rel_l2(&y, &r[n..])
                );
            }
            Err(_) => println!(
                "  no f64 reference at {}, run the f64 build first",
                path.display()
            ),
        }
    }
}
