// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Adaptive frequency sweep: full solves at a few frequencies chosen greedily,
//! every requested frequency from a projection-based reduced model. The
//! method of Palace's adaptive driven solver (`drivers/drivensolver.cpp`
//! `SweepAdaptive`, `models/romoperator.cpp`), see issue #67:
//!
//! - The system splits into frequency-independent pieces, `A(k₀) = E −
//!   k₀²·Σ_m ε_m(ω)·B_m + s(k₀)·B_σ + Σ_p γ_p(k₀)·S_p`: B_0 the fixed mass
//!   matrix, B_m that of a dispersive material at unit εr (a port whose Robin
//!   term is no scalar multiple of one matrix is projected per frequency).
//!   Palace's sample choice assumes the quadratic part; the other terms make
//!   it less sharp, never the result less accurate (each sample is checked
//!   against a full solve).
//! - The basis is real: each full solution adds its real and imaginary part,
//!   orthogonalised by classical Gram-Schmidt run twice in the Euclidean dof
//!   product. The pieces are projected with Vᵀ (not Vᴴ), which keeps the
//!   reduced system complex symmetric, and grow one row and column per new
//!   basis vector.
//! - A reduced solve is a dense column-pivoted QR (the real basis makes the
//!   reduced matrix poorly conditioned, as Palace notes).
//! - The band ends are sampled first. The next frequency comes from minimal
//!   rational interpolation (Pradovera, SINUM 58 (2020); Adv. Comput. Math.
//!   49 (2023)): the snapshots of an excitation, linearised as `[u; j·t·u]`,
//!   give barycentric weights q (the right singular vector of the smallest
//!   singular value), and the next sample is where `|Σ q_i/(t − t_i)|` is
//!   smallest over the band.
//! - At a new sample the full solution is compared with the reduced one of
//!   the basis before it; the sweep stops when `memory` samples in a row are
//!   within `tol` for every excitation.
//!
//! Unlike Palace, one factorisation solves every excitation (a direct solve
//! makes the extra right-hand sides cheap), so each sample feeds the basis
//! with all of them, and the symbolic analysis is reused across samples.

use num_complex::Complex64 as C64;

use crate::assembly::{DrivenSystem, SolveResult};
use crate::excitation::Excitation;

/// Settings of the adaptive sweep (Palace's `AdaptiveTol`,
/// `AdaptiveMaxSamples`, `AdaptiveConvergenceMemory`).
#[derive(Clone, Debug)]
pub struct AdaptiveSettings {
    /// Relative error of the reduced solution at a new sample below which it
    /// counts as converged there.
    pub tol: f64,
    /// Full solves per excitation at most.
    pub max_samples: usize,
    /// Converged samples in a row that end the sweep.
    pub memory: usize,
}

impl Default for AdaptiveSettings {
    fn default() -> Self {
        AdaptiveSettings { tol: 1e-3, max_samples: 20, memory: 2 }
    }
}

/// Grid points of the next-sample search over the band.
const MRI_GRID: usize = 1_000_000;
/// A new basis vector whose part outside the basis is below this fraction of
/// its norm adds nothing.
const ORTHO_DROP: f64 = 1e-12;

/// A sparse operator on the free DOFs, (row, col, value) triplets.
struct Triplets(Vec<(usize, usize, C64)>);

impl Triplets {
    fn apply(&self, x: &[f64], n: usize) -> Vec<C64> {
        let mut y = vec![C64::new(0.0, 0.0); n];
        for &(r, c, v) in &self.0 {
            y[r] += v * x[c];
        }
        y
    }
}

/// A dense, symmetric reduced matrix that grows by one row and column.
#[derive(Default)]
struct Reduced(Vec<Vec<C64>>);

impl Reduced {
    /// Adds the row and column of the new basis vector from `w` = op·v_new.
    fn grow(&mut self, basis: &[Vec<f64>], w: &[C64]) {
        let col: Vec<C64> = dots(basis, w);
        let k = col.len() - 1;
        for (j, row) in self.0.iter_mut().enumerate() {
            row.push(col[j]);
        }
        self.0.push(col);
        debug_assert_eq!(self.0[k].len(), k + 1);
    }
}

/// `v_jᵀ·w` for every basis vector.
fn dots(basis: &[Vec<f64>], w: &[C64]) -> Vec<C64> {
    let one = |v: &Vec<f64>| v.iter().zip(w).fold(C64::new(0.0, 0.0), |s, (&a, &b)| s + b * a);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        basis.par_iter().map(one).collect()
    }
    #[cfg(not(feature = "parallel"))]
    basis.iter().map(one).collect()
}

/// `V·y` on the free DOFs.
fn expand(basis: &[Vec<f64>], y: &[C64], n: usize) -> Vec<C64> {
    let chunk = |range: std::ops::Range<usize>| -> Vec<C64> {
        range.map(|i| basis.iter().zip(y).fold(C64::new(0.0, 0.0), |s, (v, &c)| s + c * v[i])).collect()
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        const BLOCK: usize = 4096;
        (0..n.div_ceil(BLOCK))
            .into_par_iter()
            .flat_map_iter(|b| chunk(b * BLOCK..((b + 1) * BLOCK).min(n)))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    chunk(0..n)
}

fn norm(x: &[C64]) -> f64 {
    x.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// The reduced model: the basis and the projected pieces.
struct Rom {
    n: usize,
    basis: Vec<Vec<f64>>,
    e: Reduced,
    /// The parts of B, see [`DrivenSystem::b_scales`].
    b: Vec<Reduced>,
    bs: Option<Reduced>,
    /// Ports whose Robin term is γ(k₀) times a fixed matrix: the port, its
    /// unit matrix and its projection.
    scalar_ports: Vec<(usize, Triplets, Reduced)>,
    /// Ports projected at every frequency.
    tensor_ports: Vec<usize>,
}

impl Rom {
    fn new(sys: &DrivenSystem, exc: &Excitation) -> Rom {
        let mut scalar_ports = Vec::new();
        let mut tensor_ports = Vec::new();
        for (p, port) in sys.ports.iter().enumerate() {
            if port.robin_is_scalar() {
                scalar_ports.push((p, Triplets(sys.port_robin(p, exc, true)), Reduced::default()));
            } else {
                tensor_ports.push(p);
            }
        }
        Rom {
            n: sys.n_free(),
            basis: Vec::new(),
            e: Reduced::default(),
            b: (0..1 + sys.dispersive.len()).map(|_| Reduced::default()).collect(),
            bs: sys.data_bsigma.as_ref().map(|_| Reduced::default()),
            scalar_ports,
            tensor_ports,
        }
    }

    fn dim(&self) -> usize {
        self.basis.len()
    }

    /// Adds the part of `x` outside the basis (two Gram-Schmidt passes) and
    /// projects every piece onto it; false when nothing is left.
    fn add(&mut self, sys: &DrivenSystem, x: Vec<f64>) -> bool {
        let pre = x.iter().map(|a| a * a).sum::<f64>().sqrt();
        if pre == 0.0 {
            return false;
        }
        let mut v = x;
        for _ in 0..2 {
            let w: Vec<C64> = v.iter().map(|&a| C64::new(a, 0.0)).collect();
            let h = dots(&self.basis, &w);
            for (bj, hj) in self.basis.iter().zip(&h) {
                for (vi, &bi) in v.iter_mut().zip(bj) {
                    *vi -= hj.re * bi;
                }
            }
        }
        let rest = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        if rest <= ORTHO_DROP * pre {
            return false;
        }
        v.iter_mut().for_each(|a| *a /= rest);
        let (ev, bv, bsv) = sys.volume_apply(&v);
        let port_w: Vec<Vec<C64>> = self.scalar_ports.iter().map(|(_, s, _)| s.apply(&v, self.n)).collect();
        self.basis.push(v);
        self.e.grow(&self.basis, &ev);
        for (r, w) in self.b.iter_mut().zip(&bv) {
            r.grow(&self.basis, w);
        }
        if let (Some(r), Some(w)) = (self.bs.as_mut(), bsv) {
            r.grow(&self.basis, &w);
        }
        for ((_, _, r), w) in self.scalar_ports.iter_mut().zip(port_w) {
            r.grow(&self.basis, &w);
        }
        true
    }

    /// The reduced solutions of every excitation at `freq`, one coefficient
    /// vector each.
    fn solve(&self, sys: &DrivenSystem, freq: f64) -> Result<Vec<Vec<C64>>, String> {
        let exc = Excitation::new(freq, sys.mesh.l0);
        let k = self.dim();
        let k0_sq = C64::from(exc.k0 * exc.k0);
        let mut a: Vec<Vec<C64>> = self.e.0.clone();
        let add = |a: &mut [Vec<C64>], f: C64, m: &Reduced| {
            for (row, mrow) in a.iter_mut().zip(&m.0) {
                for (x, &v) in row.iter_mut().zip(mrow) {
                    *x += f * v;
                }
            }
        };
        for (scale, r) in sys.b_scales(freq).into_iter().zip(&self.b) {
            add(&mut a, -k0_sq * scale, r);
        }
        if let Some(bs) = &self.bs {
            add(&mut a, DrivenSystem::sigma_scale(&exc), bs);
        }
        for (p, _, r) in &self.scalar_ports {
            add(&mut a, sys.ports[*p].get_gamma(&exc), r);
        }
        for &p in &self.tensor_ports {
            // W = R·V on the rows R touches, then Vᵀ·W
            let mut rows: Vec<usize> = Vec::new();
            let mut slot: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            let mut w: Vec<Vec<C64>> = Vec::new();
            for (r, c, val) in sys.port_robin(p, &exc, false) {
                let s = *slot.entry(r).or_insert_with(|| {
                    rows.push(r);
                    w.push(vec![C64::new(0.0, 0.0); k]);
                    rows.len() - 1
                });
                for (x, bj) in w[s].iter_mut().zip(&self.basis) {
                    *x += val * bj[c];
                }
            }
            for (ws, &r) in w.iter().zip(&rows) {
                for (row, bi) in a.iter_mut().zip(&self.basis) {
                    let vi = bi[r];
                    if vi != 0.0 {
                        for (x, &wj) in row.iter_mut().zip(ws) {
                            *x += wj * vi;
                        }
                    }
                }
            }
        }
        let rhs: Vec<Vec<C64>> = sys
            .rhs(&exc)
            .iter()
            .map(|b| {
                let nz: Vec<usize> = (0..b.len()).filter(|&i| b[i] != C64::new(0.0, 0.0)).collect();
                self.basis.iter().map(|v| nz.iter().fold(C64::new(0.0, 0.0), |s, &i| s + b[i] * v[i])).collect()
            })
            .collect();
        dense_solve(&a, &rhs)
    }
}

/// `A·Y = R` for a small dense complex `A`, by QR with column pivoting.
fn dense_solve(a: &[Vec<C64>], rhs: &[Vec<C64>]) -> Result<Vec<Vec<C64>>, String> {
    use faer::linalg::solvers::Solve;
    let k = a.len();
    let z = |c: C64| faer::c64 { re: c.re, im: c.im };
    let am = faer::Mat::<faer::c64>::from_fn(k, k, |i, j| z(a[i][j]));
    let bm = faer::Mat::<faer::c64>::from_fn(k, rhs.len(), |i, j| z(rhs[j][i]));
    let x = am.col_piv_qr().solve(&bm);
    let out: Vec<Vec<C64>> = (0..rhs.len())
        .map(|j| (0..k).map(|i| C64::new(x[(i, j)].re, x[(i, j)].im)).collect())
        .collect();
    if out.iter().flatten().any(|c| !c.re.is_finite() || !c.im.is_finite()) {
        return Err("adaptive sweep: the reduced system is singular".into());
    }
    Ok(out)
}

/// The snapshots of one excitation for the next-sample choice: the sample
/// (frequency over the band's top) and the snapshot's coordinates in the
/// basis. The basis holds every snapshot and grows only by parts orthogonal
/// to them, so older coordinates pad with zeros.
#[derive(Default)]
struct Mri {
    samples: Vec<(f64, Vec<C64>)>,
}

impl Mri {
    /// The point of `[lo, hi]` where the barycentric denominator of the
    /// minimal rational interpolant is smallest.
    fn next(&self, lo: f64, hi: f64) -> Result<f64, String> {
        let s = self.samples.len();
        let k = self.samples.iter().map(|(_, c)| c.len()).max().unwrap_or(0);
        let z = |c: C64| faer::c64 { re: c.re, im: c.im };
        // [c; j·t·c] per snapshot, as columns
        let zm = faer::Mat::<faer::c64>::from_fn(2 * k, s, |i, j| {
            let (t, c) = &self.samples[j];
            let ci = |i: usize| c.get(i).copied().unwrap_or(C64::new(0.0, 0.0));
            if i < k { z(ci(i)) } else { z(C64::new(0.0, *t) * ci(i - k)) }
        });
        let svd = zm.thin_svd().map_err(|e| format!("adaptive sweep: SVD of the snapshots failed: {e:?}"))?;
        let sv = svd.S().column_vector();
        let (s_max, s_min) = (sv[0].re, sv[s - 1].re);
        if s_min < 1e-12 * s_max {
            eprintln!("  adaptive: the snapshots are nearly dependent (smallest singular value {:.1e} of {:.1e})", s_min, s_max);
        }
        let v = svd.V();
        let q: Vec<C64> = (0..s).map(|i| C64::new(v[(i, s - 1)].re, v[(i, s - 1)].im)).collect();
        let ts: Vec<f64> = self.samples.iter().map(|(t, _)| *t).collect();
        let denom = |t: f64| q.iter().zip(&ts).fold(C64::new(0.0, 0.0), |acc, (&qi, &ti)| acc + qi / (t - ti)).norm();
        let (mut best, mut best_t) = (f64::INFINITY, 0.5 * (lo + hi));
        for g in 0..=MRI_GRID {
            let t = lo + (hi - lo) * g as f64 / MRI_GRID as f64;
            let d = denom(t);
            if d < best {
                (best, best_t) = (d, t);
            }
        }
        Ok(best_t)
    }
}

/// The full solves of the sweep: one solver whose symbolic analysis every
/// sample reuses.
struct Full {
    solver: rapidfem_core::linalg::SymmetricSolver<C64>,
    coo_rows: Vec<usize>,
    coo_cols: Vec<usize>,
    coo_vals: Vec<C64>,
    sampled: Vec<f64>,
}

impl Full {
    fn new(n_exc: usize) -> Full {
        let mut solver = rapidfem_core::linalg::SymmetricSolver::<C64>::new();
        solver.expect_rhs(n_exc);
        Full { solver, coo_rows: Vec::new(), coo_cols: Vec::new(), coo_vals: Vec::new(), sampled: Vec::new() }
    }

    /// One full solve at `freq`: the relative error of the reduced model
    /// before it per excitation (infinite while the basis is empty); the
    /// solutions then join the basis and the snapshots (`freq / hi`).
    fn sample(&mut self, sys: &mut DrivenSystem, rom: &mut Rom, mri: &mut [Mri], freq: f64, hi: f64) -> Result<Vec<f64>, String> {
        let t = web_time::Instant::now();
        let n = sys.n_free();
        let b = sys.assemble(freq, &mut self.coo_rows, &mut self.coo_cols, &mut self.coo_vals);
        if self.sampled.is_empty() {
            self.solver.factorize(n, &self.coo_rows, &self.coo_cols, &self.coo_vals)?;
        } else {
            self.solver.refactorize(n, &self.coo_rows, &self.coo_cols, &self.coo_vals)?;
        }
        let xs = self.solver.solve_many(&b)?;
        let t_full = t.elapsed().as_secs_f64();
        let errors: Vec<f64> = if rom.dim() == 0 {
            vec![f64::INFINITY; xs.len()]
        } else {
            let ys = rom.solve(sys, freq)?;
            xs.iter()
                .zip(&ys)
                .map(|(x, y)| {
                    let r = expand(&rom.basis, y, n);
                    let d: Vec<C64> = x.iter().zip(&r).map(|(a, b)| a - b).collect();
                    norm(&d) / norm(x).max(f64::MIN_POSITIVE)
                })
                .collect()
        };
        for x in &xs {
            rom.add(sys, x.iter().map(|c| c.re).collect());
            rom.add(sys, x.iter().map(|c| c.im).collect());
        }
        for (m, x) in mri.iter_mut().zip(&xs) {
            m.samples.push((freq / hi, dots(&rom.basis, x)));
        }
        self.sampled.push(freq);
        eprintln!(
            "  adaptive sample {:>2} at f={:>10.4e} Hz: full solve {:>6.1}ms, reduced error {}, basis {}",
            self.sampled.len(), freq, t_full * 1e3,
            errors.iter().map(|e| if e.is_finite() { format!("{e:.1e}") } else { "-".into() }).collect::<Vec<_>>().join(" "),
            rom.dim(),
        );
        Ok(errors)
    }
}

/// Adaptive sweep over `frequencies` (see the module doc). Calls `on_solve`
/// for every frequency with the reduced solutions, like
/// [`crate::assembly::frequency_sweep`]; returns them with the frequencies
/// solved in full.
pub fn adaptive_sweep(
    sys: &mut DrivenSystem,
    frequencies: &[f64],
    settings: &AdaptiveSettings,
    mut on_solve: Option<&mut dyn FnMut(usize, f64, &SolveResult) -> bool>,
) -> Result<(Vec<SolveResult>, Vec<f64>), String> {
    if settings.tol.is_nan() || settings.tol <= 0.0 || settings.max_samples < 2 || settings.memory == 0 {
        return Err(format!("adaptive sweep: need tol > 0, max_samples >= 2 and memory >= 1, got {settings:?}"));
    }
    let lo = frequencies.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = frequencies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !lo.is_finite() || lo >= hi {
        let results = crate::assembly::frequency_sweep(sys, frequencies, on_solve)?;
        let solved = frequencies[..results.len()].to_vec();
        return Ok((results, solved));
    }
    let t_all = web_time::Instant::now();
    let n_exc = sys.ports.iter().filter(|p| p.is_driven()).count();
    if n_exc == 0 {
        return Err("adaptive sweep: no driven port".into());
    }
    let mut rom = Rom::new(sys, &Excitation::new(lo, sys.mesh.l0));
    let mut mri: Vec<Mri> = (0..n_exc).map(|_| Mri::default()).collect();
    let mut full = Full::new(n_exc);
    full.sample(sys, &mut rom, &mut mri, lo, hi)?;
    full.sample(sys, &mut rom, &mut mri, hi, hi)?;
    let mut memory = vec![0usize; n_exc];
    let cap = settings.max_samples * n_exc;
    let mut n_samples = 2;
    while n_samples < cap {
        let Some(e) = memory.iter().position(|&m| m < settings.memory) else { break };
        let f = mri[e].next(lo / hi, 1.0)? * hi;
        let errors = full.sample(sys, &mut rom, &mut mri, f, hi)?;
        n_samples += 1;
        for (m, err) in memory.iter_mut().zip(&errors) {
            *m = if *err < settings.tol { *m + 1 } else { 0 };
        }
    }
    let converged = memory.iter().all(|&m| m >= settings.memory);
    let t_offline = t_all.elapsed().as_secs_f64();
    eprintln!(
        "  adaptive sweep {} after {n_samples} full solves in {:.1}s, basis {}",
        if converged { "converged" } else { "stopped at max_samples, NOT converged" },
        t_offline, rom.dim(),
    );

    let t_online = web_time::Instant::now();
    let n_field = sys.basis.n_field;
    let mut results = Vec::with_capacity(frequencies.len());
    for (fi, &freq) in frequencies.iter().enumerate() {
        let ys = rom.solve(sys, freq)?;
        let solutions = ys.iter().map(|y| sys.to_field(&expand(&rom.basis, y, rom.n))).collect();
        results.push(SolveResult { solutions, n_field });
        if let Some(cb) = on_solve.as_deref_mut()
            && !cb(fi, freq, results.last().unwrap()) {
                eprintln!("  sweep stopped early after frequency {}/{} (interrupt)", fi + 1, frequencies.len());
                break;
            }
    }
    eprintln!(
        "  adaptive sweep: {} frequencies from the reduced model in {:.1}s",
        results.len(), t_online.elapsed().as_secs_f64(),
    );
    Ok((results, full.sampled))
}
