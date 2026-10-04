// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! rslab settings on rapidfem's own systems: every `.sparta` file given
//! (written by a sweep with `RAPIDFEM_DUMP_DIR`, see `dump.rs`) is analysed,
//! factored and solved under each settings variant, and one line per (file,
//! variant) reports the analysis, factor and solve times (best of
//! `RSLAB_TUNE_REPS`, default 2), the planned peak and the largest relative
//! residual. `RSLAB_TUNE_VARIANTS` (comma separated names) picks variants.
//!
//!     cargo run --release -p rapidfem-fd --example rslab_tune -- dump/*.sparta

use std::io::Read;
use std::time::Instant;

use num_complex::Complex64 as C64;
use rslab::{AmalgamationStrategy, CscMatrix, LdltSymbolic, OrderingMethod, RelaxAmalgamation, SolverSettings, Threads};

/// One system of a dump: the lower triangle and the right-hand sides.
struct System {
    a: CscMatrix<C64>,
    rhs: Vec<Vec<C64>>,
}

fn read(path: &str) -> std::io::Result<System> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{path}: {m}"));
    if buf.len() < 48 || &buf[..8] != b"SPARTA02" {
        return Err(bad("not a SPARTA02 file"));
    }
    let mut at = 8;
    let mut u64_ = || {
        let v = u64::from_le_bytes(buf[at..at + 8].try_into().unwrap());
        at += 8;
        v
    };
    let (n, nnz, nrhs, _flags) = (u64_() as usize, u64_() as usize, u64_() as usize, u64_());
    let _k0 = u64_();
    let col_ptr: Vec<usize> = (0..=n).map(|_| u64_() as usize).collect();
    let row_idx: Vec<usize> = (0..nnz).map(|_| u64_() as usize).collect();
    let mut f64_ = || {
        let v = f64::from_le_bytes(buf[at..at + 8].try_into().unwrap());
        at += 8;
        v
    };
    let mut c64 = || C64::new(f64_(), f64_());
    let values: Vec<C64> = (0..nnz).map(|_| c64()).collect();
    let rhs = (0..nrhs).map(|_| (0..n).map(|_| c64()).collect()).collect();
    Ok(System { a: CscMatrix { n, col_ptr, row_idx, values }, rhs })
}

/// `max_k |A x_k - b_k| / |b_k|` with `A` symmetric from its lower triangle.
fn residual(a: &CscMatrix<C64>, x: &[Vec<C64>], b: &[Vec<C64>]) -> f64 {
    let mut worst = 0.0f64;
    for (xk, bk) in x.iter().zip(b) {
        let mut y = vec![C64::new(0.0, 0.0); a.n];
        a.symv(xk, &mut y);
        let num: f64 = y.iter().zip(bk).map(|(u, v)| (u - v).norm_sqr()).sum::<f64>().sqrt();
        let den: f64 = bk.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        worst = worst.max(num / den);
    }
    worst
}

/// The variants: a name and its change to the default settings.
fn variants() -> Vec<(&'static str, Box<dyn Fn(&mut SolverSettings)>)> {
    let cores = std::thread::available_parallelism().map_or(4, |p| p.get());
    vec![
        ("default", Box::new(|_| {})),
        ("amd", Box::new(|s| s.ordering.method = OrderingMethod::Amd)),
        ("amf", Box::new(|s| s.ordering.method = OrderingMethod::Amf)),
        ("metis", Box::new(|s| s.ordering.method = OrderingMethod::MetisND)),
        ("threads_all", Box::new(move |s| s.threads = Threads::Auto { max: cores })),
        ("threads_fixed_all", Box::new(move |s| s.threads = Threads::Fixed(cores))),
        ("threads_2", Box::new(|s| s.threads = Threads::Fixed(2))),
        ("nemin_8", Box::new(|s| s.amalgamation.nemin = 8)),
        ("nemin_32", Box::new(|s| s.amalgamation.nemin = 32)),
        ("nemin_64", Box::new(|s| s.amalgamation.nemin = 64)),
        ("relax", Box::new(|s| s.amalgamation.relax = Some(RelaxAmalgamation::default()))),
        ("adjacency", Box::new(|s| s.amalgamation.strategy = AmalgamationStrategy::Adjacency)),
        ("renumber", Box::new(|s| s.amalgamation.strategy = AmalgamationStrategy::Renumber)),
        ("panel_32", Box::new(|s| s.kernels.panel_nb = 32)),
        ("panel_128", Box::new(|s| s.kernels.panel_nb = 128)),
        ("schur_128", Box::new(|s| s.kernels.schur_tile = 128)),
        ("schur_512", Box::new(|s| s.kernels.schur_tile = 512)),
        ("trailing_32", Box::new(|s| s.kernels.trailing_block = 32)),
        ("solve_block_2048", Box::new(|s| s.solve.block = 2048)),
    ]
}

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = std::env::var("RSLAB_TUNE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let pick: Option<Vec<String>> = std::env::var("RSLAB_TUNE_VARIANTS").ok().map(|v| v.split(',').map(str::to_string).collect());
    println!("{:<32} {:<18} {:>9} {:>11} {:>9} {:>9} {:>9} {:>10} {:>9}", "file", "variant", "n", "factor_nnz", "analyze", "factor", "solve", "peak_MB", "resid");
    for f in &files {
        let sys = match read(f) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{e}");
                continue;
            }
        };
        let name = std::path::Path::new(f).file_name().unwrap().to_string_lossy().into_owned();
        for (vname, change) in variants() {
            if pick.as_ref().is_some_and(|p| !p.iter().any(|x| x == vname)) {
                continue;
            }
            let mut s = SolverSettings::default();
            change(&mut s);
            let (mut ta, mut tf, mut ts) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut row = None;
            for _ in 0..reps {
                let t = Instant::now();
                let sym = match LdltSymbolic::analyze(&sys.a, &s) {
                    Ok(sym) => sym,
                    Err(e) => {
                        println!("{name:<32} {vname:<18} analyze failed: {e:?}");
                        break;
                    }
                };
                ta = ta.min(t.elapsed().as_secs_f64());
                let plan = sym.memory_plan::<C64>(&s, sys.rhs.len());
                let nnz = sym.estimate_memory::<C64>().factor_nnz;
                let t = Instant::now();
                let solver = match sym.factor(&sys.a, &s) {
                    Ok(x) => x,
                    Err(e) => {
                        println!("{name:<32} {vname:<18} factor failed: {e:?}");
                        break;
                    }
                };
                tf = tf.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                let x = solver.solve_many(&sys.rhs.concat(), sys.rhs.len()).expect("solve");
                ts = ts.min(t.elapsed().as_secs_f64());
                let xs: Vec<Vec<C64>> = x.chunks(sys.a.n).map(<[C64]>::to_vec).collect();
                row = Some((nnz, plan.peak_bytes() as f64 / 1e6, residual(&sys.a, &xs, &sys.rhs)));
            }
            if let Some((nnz, peak, res)) = row {
                println!("{name:<32} {vname:<18} {:>9} {:>11.3e} {ta:>9.3} {tf:>9.3} {ts:>9.3} {peak:>10.0} {res:>9.1e}", sys.a.n, nnz as f64);
            }
        }
    }
}
