// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Allocation audit, counts heap allocations on the TD hot paths.
//!
//! A counting global allocator wraps the system allocator; the audit then
//! reports how many allocations each operator call performs and how that
//! scales with the mesh. Run:
//!
//! ```text
//! cargo run --release -p rapidfem-td --example allocaudit
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rapidfem_td::mesh_gen::structured_box;
use rapidfem_td::propagator::{KrylovWorkspace, etd_step, expmv};
use rapidfem_td::rhs::MaxwellOperator;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn snap() -> (usize, usize) {
    (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed))
}

fn main() {
    println!("rapidfem-td, allocation audit  (release, order 2)\n");

    // --- apply: allocations per operator call, vs mesh size --------------
    println!(
        "{:>7} {:>9} {:>10} {:>14} {:>12}",
        "cells", "n_elem", "n_dof", "apply allocs", "allocs/elem"
    );
    for &c in &[2usize, 4, 6] {
        let mesh = structured_box(c, c, c, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0, Default::default());
        let n = op.n_dof();
        let n_elem = mesh.n_tets();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        op.apply(&y); // warm
        let (a0, _) = snap();
        let _ = op.apply(&y);
        let (a1, _) = snap();
        let da = a1 - a0;
        println!(
            "{:>7} {:>9} {:>10} {:>14} {:>12.1}",
            c * c * c,
            n_elem,
            n,
            da,
            da as f64 / n_elem as f64,
        );
    }

    // --- one exponential step: apply count × per-apply allocations -------
    let mesh = structured_box(4, 4, 4, 1.0, 1.0, 1.0);
    let op = MaxwellOperator::new(&mesh, 2, 1.0, Default::default());
    let n = op.n_dof();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.07).cos()).collect();

    println!("\nn_dof = {n}:");
    {
        expmv(|x| op.apply(x), &y, 0.02, 40); // warm
        let (a0, b0) = snap();
        let _ = expmv(|x| op.apply(x), &y, 0.02, 40);
        let (a1, b1) = snap();
        println!(
            "  expmv      krylov 40 (allocating wrapper): {} allocs, {} MiB",
            a1 - a0,
            (b1 - b0) / (1 << 20),
        );
    }
    {
        let mut ws = KrylovWorkspace::new();
        let mut out = vec![0.0; n];
        ws.expmv_into(|x, ax| op.apply_into(x, ax), &y, 0.02, 40, 0.0, &mut out);
        let (a0, b0) = snap();
        let t = Instant::now();
        ws.expmv_into(|x, ax| op.apply_into(x, ax), &y, 0.02, 40, 0.0, &mut out);
        let dt = t.elapsed();
        let (a1, b1) = snap();
        println!(
            "  expmv_into krylov 40 (reused workspace):    {} allocs, \
             {} KiB, {:.2} ms",
            a1 - a0,
            (b1 - b0) / 1024,
            dt.as_secs_f64() * 1e3,
        );
    }
    {
        let b = vec![0.0; n];
        etd_step(|x| op.apply(x), &y, &b, 0.02, 40); // warm
        let (a0, _) = snap();
        let _ = etd_step(|x| op.apply(x), &y, &b, 0.02, 40);
        let (a1, _) = snap();
        println!("  etd_step   krylov 40 (allocating):          {} allocs", a1 - a0);
    }

    // --- sparse assembly --------------------------------------------------
    {
        let (a0, _) = snap();
        let _ = op.assemble_sparse();
        let (a1, _) = snap();
        println!("  assemble_sparse: {} allocs", a1 - a0);
    }
}
