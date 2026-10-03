// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Arnoldi orthogonalisation benchmark, MGS vs CGS2, on realistic meshes.
//!
//! The production Krylov stepper (`propagator::expmv_into`) orthogonalises
//! with serial modified Gram-Schmidt (MGS). This pits it against classical
//! Gram-Schmidt with one reorthogonalisation (CGS2), serial, and
//! rayon-parallel, on structured-box meshes large enough to be
//! representative. (An earlier "CGS2 measured slower" note was taken on
//! meshes of a few dozen tetrahedra, far too small to generalise from.)
//!
//! MGS issues `j+1` *sequential* dot/axpy pairs per Arnoldi step. CGS2
//! batches the `j+1` projections into two passes whose dot products are
//! independent, cache-friendly and parallelisable, at the cost of 2x the
//! orthogonalisation flops. The matvec is identical for both, so the
//! `X - matvec` columns isolate the orthogonalisation itself.
//!
//! ```text
//! cargo run --release -p rapidfem-td --example ortho_bench
//! ```

use std::time::Instant;

use rapidfem_td::mesh_gen::structured_box;
use rapidfem_td::rhs::MaxwellOperator;
use rayon::prelude::*;

fn norm2(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

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

/// `m` matvecs over a fixed basis, the cost both orthogonalisations share.
fn matvec_only(op: &MaxwellOperator, basis: &[f64], n: usize, m: usize, w: &mut [f64]) {
    for j in 0..m {
        op.apply_into(&basis[j * n..j * n + n], w);
    }
}

/// One full `m`-step Arnoldi, modified Gram-Schmidt, the production path.
fn arnoldi_mgs(op: &MaxwellOperator, v: &[f64], m: usize) -> f64 {
    let n = v.len();
    let mut basis = vec![0.0; (m + 1) * n];
    let mut w = vec![0.0; n];
    let beta = norm2(v);
    for k in 0..n {
        basis[k] = v[k] / beta;
    }
    let mut tail = 0.0;
    for j in 0..m {
        op.apply_into(&basis[j * n..j * n + n], &mut w);
        for i in 0..=j {
            let b0 = i * n;
            let mut hij = 0.0;
            for k in 0..n {
                hij += w[k] * basis[b0 + k];
            }
            for k in 0..n {
                w[k] -= hij * basis[b0 + k];
            }
        }
        tail = norm2(&w);
        if tail < 1e-12 || j + 1 == m {
            break;
        }
        let inv = 1.0 / tail;
        let dst = (j + 1) * n;
        for k in 0..n {
            basis[dst + k] = w[k] * inv;
        }
    }
    tail
}

/// One full `m`-step Arnoldi, classical GS with one reorthogonalisation
/// (CGS2). `parallel` runs the batched projection on the rayon pool. The
/// Hessenberg bookkeeping (`O(m²)` adds) is omitted, negligible against
/// the `O(m²·n)` projection work this benchmark measures.
fn arnoldi_cgs2(op: &MaxwellOperator, v: &[f64], m: usize, parallel: bool) -> f64 {
    let n = v.len();
    let mut basis = vec![0.0; (m + 1) * n];
    let mut w = vec![0.0; n];
    let beta = norm2(v);
    for k in 0..n {
        basis[k] = v[k] / beta;
    }
    let mut tail = 0.0;
    for j in 0..m {
        op.apply_into(&basis[j * n..j * n + n], &mut w);
        let cols = j + 1;
        // Two projection passes against the `cols` prior basis vectors.
        for _pass in 0..2 {
            // c[i] = <basis[i], w>, independent across i.
            let c: Vec<f64> = if parallel {
                (0..cols)
                    .into_par_iter()
                    .map(|i| {
                        let b = &basis[i * n..i * n + n];
                        b.iter().zip(&w).map(|(a, x)| a * x).sum()
                    })
                    .collect()
            } else {
                (0..cols)
                    .map(|i| {
                        let b = &basis[i * n..i * n + n];
                        b.iter().zip(&w).map(|(a, x)| a * x).sum()
                    })
                    .collect()
            };
            // w -= Σ_i c[i]·basis[i].
            if parallel {
                let chunk = 8192usize;
                let basis_ref: &[f64] = &basis;
                w.par_chunks_mut(chunk).enumerate().for_each(|(ci, wc)| {
                    let k0 = ci * chunk;
                    let len = wc.len();
                    for i in 0..cols {
                        let coeff = c[i];
                        let b = &basis_ref[i * n + k0..i * n + k0 + len];
                        for (wk, bk) in wc.iter_mut().zip(b) {
                            *wk -= coeff * bk;
                        }
                    }
                });
            } else {
                for i in 0..cols {
                    let b0 = i * n;
                    let coeff = c[i];
                    for k in 0..n {
                        w[k] -= coeff * basis[b0 + k];
                    }
                }
            }
        }
        tail = norm2(&w);
        if tail < 1e-12 || j + 1 == m {
            break;
        }
        let inv = 1.0 / tail;
        let dst = (j + 1) * n;
        for k in 0..n {
            basis[dst + k] = w[k] * inv;
        }
    }
    tail
}

fn main() {
    let order = 2;
    let m = 40;
    let reps = 7;
    println!("rapidfem-td, Arnoldi orthogonalisation benchmark");
    println!(
        "order {order}, krylov m = {m}, rayon threads: {}\n",
        rayon::current_num_threads()
    );
    println!(
        "{:>6} {:>9} {:>11} {:>11} {:>11} {:>11}  | {:>10} {:>10} {:>10}  {:>8}",
        "cells", "n_dof", "matvec", "MGS", "CGS2-ser", "CGS2-par",
        "ortho:MGS", "CGS2-ser", "CGS2-par", "par/MGS"
    );
    for &c in &[4usize, 6, 8, 10, 12] {
        let mesh = structured_box(c, c, c, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, order, 1.0, Default::default());
        let n = op.n_dof();
        let v: Vec<f64> = (0..n).map(|i| (0.3 + i as f64 * 0.011).sin()).collect();

        // A fixed `m`-vector basis for the matvec-only baseline.
        let basis: Vec<f64> =
            (0..m * n).map(|i| (0.1 + i as f64 * 0.003).cos()).collect();
        let mut w = vec![0.0; n];

        arnoldi_mgs(&op, &v, m); // warm

        let t_mv = time_median(reps, || matvec_only(&op, &basis, n, m, &mut w));
        let t_mgs = time_median(reps, || {
            arnoldi_mgs(&op, &v, m);
        });
        let t_cs = time_median(reps, || {
            arnoldi_cgs2(&op, &v, m, false);
        });
        let t_cp = time_median(reps, || {
            arnoldi_cgs2(&op, &v, m, true);
        });

        // Orthogonalisation only, the full Arnoldi minus the shared matvec.
        let o_mgs = (t_mgs - t_mv).max(0.0);
        let o_cs = (t_cs - t_mv).max(0.0);
        let o_cp = (t_cp - t_mv).max(0.0);

        println!(
            "{:>6} {:>9} {:>9.2}ms {:>9.2}ms {:>9.2}ms {:>9.2}ms  | \
             {:>8.2}ms {:>8.2}ms {:>8.2}ms  {:>7.2}x",
            c * c * c,
            n,
            t_mv * 1e3,
            t_mgs * 1e3,
            t_cs * 1e3,
            t_cp * 1e3,
            o_mgs * 1e3,
            o_cs * 1e3,
            o_cp * 1e3,
            if o_cp > 0.0 { o_mgs / o_cp } else { 0.0 },
        );
    }
    println!(
        "\nmatvec = m matvecs alone; ortho columns = full Arnoldi - matvec.\n\
         par/MGS = MGS-orthogonalisation time / CGS2-parallel orthogonalisation time."
    );
}
