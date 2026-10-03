// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! GPU scaling benchmark for the time-domain backend.
//!
//! Sweeps structured-box grids from a few hundred thousand to ~10M state
//! DOFs in three geometries, a cube, an elongated beam, a flat slab, and
//! reports the GPU throughput of the explicit LSERK4 transient and the
//! Krylov exponential propagator. The cube / beam / slab differ in
//! surface-to-volume ratio, so the flux-to-volume work balance shifts
//! across them.
//!
//! The CPU is timed alongside only at the smaller grids; above
//! `CPU_DOF_LIMIT` its run dominates the benchmark without adding
//! information (the speedup is already clear by then).
//!
//! ```text
//! cargo run --release -p rapidfem-td --features gpu --example gpu_bench
//! ```

use std::time::Instant;

use rapidfem_td::constants::Field;
use rapidfem_td::explicit::LserkWorkspace;
use rapidfem_td::gpu::{GpuContext, GpuOperator};
use rapidfem_td::mesh_gen::structured_box;
use rapidfem_td::propagator::expmv;
use rapidfem_td::rhs::MaxwellOperator;
use rapidfem_td::session::Drive;

const STEPS: usize = 100;
const KRYLOV_DIM: usize = 40;
/// Polynomial order of the sweep (order 2: 60 state DOFs per tet).
const ORDER: usize = 2;
/// Above this state-DOF count the CPU reference is skipped, its run
/// would dominate the benchmark without adding information.
const CPU_DOF_LIMIT: usize = 1_500_000;

/// Median wall-clock seconds of `reps` runs of `f`.
fn time_median(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut ts: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[reps / 2]
}

/// Spectral radius of the operator, for a sub-CFL explicit step.
fn spectral_radius(op: &MaxwellOperator, n: usize) -> Field {
    let mut v: Vec<Field> = (0..n).map(|i| (i as Field * 0.05).sin()).collect();
    let mut rho = 1.0;
    for _ in 0..20 {
        let av = op.apply(&v);
        rho = av.iter().map(|x| x * x).sum::<Field>().sqrt();
        let inv = 1.0 / rho;
        for (vi, &a) in v.iter_mut().zip(&av) {
            *vi = a * inv;
        }
    }
    rho
}

fn main() {
    let gpu = match GpuContext::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU available: {e}");
            return;
        }
    };
    println!("GPU scaling benchmark, time-domain backend");
    println!("device: {}", gpu.device_name);
    println!(
        "order {ORDER}, {STEPS} LSERK4 steps, krylov-dim {KRYLOV_DIM}, \
         rayon threads {}\n",
        rayon::current_num_threads(),
    );

    // (geometry label, nx, ny, nz), three shapes across a size sweep,
    // each reaching ~10M state DOFs at the top end.
    let cases: &[(&str, usize, usize, usize)] = &[
        ("cube", 8, 8, 8),
        ("cube", 14, 14, 14),
        ("cube", 20, 20, 20),
        ("cube", 26, 26, 26),
        ("cube", 30, 30, 30),
        ("beam", 48, 8, 8),
        ("beam", 110, 11, 11),
        ("beam", 140, 14, 14),
        ("slab", 28, 28, 6),
        ("slab", 60, 60, 6),
        ("slab", 96, 96, 6),
    ];

    println!(
        "{:>6} {:>11} {:>9} {:>12} {:>11} {:>12} {:>9} {:>12} {:>9}",
        "geom",
        "n_dof",
        "tets",
        "lserk GPU",
        "GDOF-st/s",
        "lserk CPU",
        "lserk x",
        "expmv GPU",
        "expmv x",
    );

    for &(label, nx, ny, nz) in cases {
        let mesh = structured_box(nx, ny, nz, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, ORDER, 1.0, Default::default());
        let n = op.n_dof();
        let tets = n / 60;
        let y0: Vec<Field> =
            (0..n).map(|i| (i as Field * 0.05).sin()).collect();
        let dt = 1.0 / spectral_radius(&op, n);
        let mut gop = GpuOperator::new(&gpu, &op).expect("GpuOperator");
        let y0_32: Vec<f32> = y0.iter().map(|&v| v as f32).collect();

        // GPU explicit LSERK4 transient.
        gop.transient(&gpu, &y0_32, dt as f32, 3, 1, Drive::Free, &[], false).unwrap();
        let gpu_lserk = time_median(5, || {
            gop.transient(&gpu, &y0_32, dt as f32, STEPS, 1, Drive::Free, &[], false).unwrap();
        });
        let gdof = n as f64 * STEPS as f64 / gpu_lserk / 1e9;

        // GPU Krylov exponential propagator.
        let t_exp = 0.5 * dt * STEPS as Field;
        gop.expmv(&gpu, &y0, t_exp, KRYLOV_DIM).unwrap();
        let gpu_expmv = time_median(5, || {
            gop.expmv(&gpu, &y0, t_exp, KRYLOV_DIM).unwrap();
        });

        // CPU reference, only at the smaller grids.
        let cpu = if n <= CPU_DOF_LIMIT {
            let cpu_lserk = time_median(3, || {
                let mut y = y0.clone();
                let mut ws = LserkWorkspace::new();
                for _ in 0..STEPS {
                    ws.step_into(|x, ax| op.apply_into(x, ax), &mut y, dt, None);
                }
            });
            let cpu_expmv = time_median(3, || {
                expmv(|x| op.apply(x), &y0, t_exp, KRYLOV_DIM);
            });
            Some((cpu_lserk, cpu_expmv))
        } else {
            None
        };

        let ms = |s: f64| s * 1e3;
        let (cpu_lserk, lserk_x, expmv_x) = match cpu {
            Some((cl, ce)) => (
                format!("{:.1}", ms(cl)),
                format!("{:.1}x", cl / gpu_lserk),
                format!("{:.1}x", ce / gpu_expmv),
            ),
            None => ("-".to_string(), "-".to_string(), "-".to_string()),
        };
        println!(
            "{label:>6} {n:>11} {tets:>9} {:>12.1} {gdof:>11.2} \
             {cpu_lserk:>12} {lserk_x:>9} {:>12.1} {expmv_x:>9}",
            ms(gpu_lserk),
            ms(gpu_expmv),
        );
    }
}
