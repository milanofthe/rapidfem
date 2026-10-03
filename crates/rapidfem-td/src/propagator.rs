// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Krylov-subspace exponential propagator.
//!
//! The semi-discrete DG system is linear, `dy/dt = A·y`, so a step of size `h`
//! is exactly `y ← exp(h·A)·y`. `A` is large and only available as a
//! matrix-free `apply`, so the action `exp(h·A)·v` is formed in an `m`-step
//! Krylov subspace: Arnoldi gives `A·V_m ≈ V_m·H_m`, and
//! `exp(h·A)·v ≈ ‖v‖·V_m·exp(h·H_m)·e₁` with the small `exp(h·H_m)` dense.
//!
//! [`KrylovWorkspace`] holds the Arnoldi basis and the dense buffers so a
//! repeated time step allocates nothing.

use rayon::prelude::*;

use crate::constants::{
    ARNOLDI_BREAKDOWN, ARNOLDI_MIN_CHUNK, ARNOLDI_TASKS_PER_THREAD, Accum,
    EXPM_SCALE_THRESHOLD, EXPM_TAYLOR_TERMS,
};
use crate::dg_basis::matmul_into;

/// Dense matrix exponential of an `n×n` row-major matrix, via
/// scaling-and-squaring with a Taylor core. Allocating wrapper around
/// `expm_into`.
pub fn expm(a: &[Accum], n: usize) -> Vec<Accum> {
    let mut out = vec![0.0; n * n];
    expm_into(a, n, &mut out, &mut ExpmScratch::new());
    out
}

/// Scratch for the dense matrix exponential, five `n×n` buffers, grown on
/// demand and reused.
struct ExpmScratch {
    cap: usize,
    b: Vec<Accum>,
    result: Vec<Accum>,
    term: Vec<Accum>,
    term2: Vec<Accum>,
    tmp: Vec<Accum>,
}

impl ExpmScratch {
    fn new() -> Self {
        ExpmScratch {
            cap: 0,
            b: Vec::new(),
            result: Vec::new(),
            term: Vec::new(),
            term2: Vec::new(),
            tmp: Vec::new(),
        }
    }
    fn ensure(&mut self, nn: usize) {
        if self.cap < nn {
            self.b.resize(nn, 0.0);
            self.result.resize(nn, 0.0);
            self.term.resize(nn, 0.0);
            self.term2.resize(nn, 0.0);
            self.tmp.resize(nn, 0.0);
            self.cap = nn;
        }
    }
}

/// `exp(A)` of an `n×n` row-major matrix into `out`, reusing `s`, the
/// allocation-free form of [`expm`].
fn expm_into(a: &[Accum], n: usize, out: &mut [Accum], s: &mut ExpmScratch) {
    let nn = n * n;
    s.ensure(nn);

    // Infinity norm, then scale so it is within EXPM_SCALE_THRESHOLD.
    let mut norm = 0.0_f64;
    for i in 0..n {
        let row: Accum = (0..n).map(|j| a[i * n + j].abs()).sum();
        norm = norm.max(row);
    }
    let sq: u32 = if norm > EXPM_SCALE_THRESHOLD {
        (norm.log2().ceil() as i64 + 1).max(0) as u32
    } else {
        0
    };
    let scale = 2.0_f64.powi(sq as i32);
    for k in 0..nn {
        s.b[k] = a[k] / scale;
    }

    // result = I, term = I.
    s.result[..nn].fill(0.0);
    s.term[..nn].fill(0.0);
    for i in 0..n {
        s.result[i * n + i] = 1.0;
        s.term[i * n + i] = 1.0;
    }
    // exp(B) = Σ Bᵏ/k!  (≈18 terms suffice for ‖B‖ ≤ 1/2).
    for k in 1..=EXPM_TAYLOR_TERMS {
        matmul_into(&s.term[..nn], &s.b[..nn], n, n, n, &mut s.term2[..nn]);
        let inv = 1.0 / k as Accum;
        for x in s.term2[..nn].iter_mut() {
            *x *= inv;
        }
        for (r, t) in s.result[..nn].iter_mut().zip(&s.term2[..nn]) {
            *r += *t;
        }
        s.term[..nn].copy_from_slice(&s.term2[..nn]);
    }
    // Square sq times.
    for _ in 0..sq {
        matmul_into(&s.result[..nn], &s.result[..nn], n, n, n, &mut s.tmp[..nn]);
        s.result[..nn].copy_from_slice(&s.tmp[..nn]);
    }
    out[..nn].copy_from_slice(&s.result[..nn]);
}

/// Reusable workspace for the Krylov exponential propagator, owns the
/// Arnoldi basis and the dense `H` buffers, so
/// [`expmv_into`](Self::expmv_into) allocates nothing once warmed.
pub struct KrylovWorkspace {
    /// Arnoldi basis, flat, vector `j` occupies `basis[j*n .. (j+1)*n]`.
    basis: Vec<Accum>,
    /// Arnoldi working vector / matvec output.
    w: Vec<Accum>,
    /// CGS2 projection coefficients, `Vᵀ·w` for one orthogonalisation pass.
    proj: Vec<Accum>,
    /// Upper Hessenberg `H`, `m×m` row-major.
    h: Vec<Accum>,
    /// `t·H` and `exp(t·H)`, packed `dim×dim`.
    th: Vec<Accum>,
    exp_th: Vec<Accum>,
    expm_scratch: ExpmScratch,
    /// Augmented state / result buffers for [`etd_step_into`](Self::etd_step_into).
    aug_in: Vec<Accum>,
    aug_out: Vec<Accum>,
}

impl Default for KrylovWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl KrylovWorkspace {
    /// An empty workspace; its buffers grow to fit on the first call.
    pub fn new() -> Self {
        KrylovWorkspace {
            basis: Vec::new(),
            w: Vec::new(),
            proj: Vec::new(),
            h: Vec::new(),
            th: Vec::new(),
            exp_th: Vec::new(),
            expm_scratch: ExpmScratch::new(),
            aug_in: Vec::new(),
            aug_out: Vec::new(),
        }
    }

    fn ensure(&mut self, n: usize, m: usize) {
        if self.basis.len() < (m + 1) * n {
            self.basis.resize((m + 1) * n, 0.0);
        }
        if self.w.len() < n {
            self.w.resize(n, 0.0);
        }
        if self.proj.len() < m {
            self.proj.resize(m, 0.0);
        }
        if self.h.len() < m * m {
            self.h.resize(m * m, 0.0);
            self.th.resize(m * m, 0.0);
            self.exp_th.resize(m * m, 0.0);
        }
    }

    /// Matrix-free `exp(t·A)·v` into `out`, with `matvec(x, ax)` writing
    /// `A·x` into `ax`. After the buffers have grown once to fit `n` and
    /// `max_dim`, this allocates nothing, the form to call in a step loop.
    ///
    /// `max_dim` caps the Krylov dimension. `tol` is a *relative*
    /// a-posteriori error tolerance: after each new basis vector the
    /// estimate `h_{j+1,j}·|exp(tH)_{dim-1,0}|` is checked, and the subspace
    /// stops growing once it drops below `tol` (or on a lucky breakdown).
    /// `tol = 0` skips the estimate and runs the full `max_dim`, the
    /// fixed-dimension behaviour. Returns the Krylov dimension actually used.
    pub fn expmv_into<F>(
        &mut self,
        matvec: F,
        v: &[Accum],
        t: Accum,
        max_dim: usize,
        tol: Accum,
        out: &mut [Accum],
    ) -> usize
    where
        F: Fn(&[Accum], &mut [Accum]),
    {
        let n = v.len();
        self.ensure(n, max_dim);

        let beta = norm2(v);
        if beta == 0.0 {
            out[..n].fill(0.0);
            return 0;
        }
        let inv_beta = 1.0 / beta;
        for k in 0..n {
            self.basis[k] = v[k] * inv_beta;
        }
        self.h[..max_dim * max_dim].fill(0.0);

        // CGS2 `w -= V·c` chunk: sized so the rayon pool is over-subscribed
        // a few-fold rather than left with a fixed `⌈n/chunk⌉` tasks that
        // starved every core but a handful on small and medium meshes.
        let ortho_chunk = (n
            / (ARNOLDI_TASKS_PER_THREAD * rayon::current_num_threads()))
        .max(ARNOLDI_MIN_CHUNK);

        // Arnoldi with classical Gram-Schmidt + one reorthogonalisation
        // (CGS2). Unlike modified Gram-Schmidt's sequential dot/axpy chain
        // the `cols` projections of each pass are independent, so the
        // batched `Vᵀ·w` and `w -= V·c` fan out across the rayon pool, a
        // large win once the mesh is past a few hundred elements (see the
        // `ortho_bench` example). Two passes recover MGS-grade
        // orthogonality. After each new basis vector the relative
        // a-posteriori estimate `h_{j+1,j}·|exp(tH)_{dim-1,0}|` is checked
        // against `tol`; the subspace stops growing once it converges. The
        // dim×dim `exp(tH)` for the estimate is cheap, O(dim³) against the
        // O(cols·n) projection work.
        let mut dim = max_dim;
        for j in 0..max_dim {
            matvec(&self.basis[j * n..j * n + n], &mut self.w[..n]);
            let cols = j + 1;
            for _pass in 0..2 {
                // proj[i] = ⟨basis[i], w⟩, independent across i.
                {
                    let basis = &self.basis;
                    let w = &self.w[..n];
                    self.proj[..cols]
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, p)| {
                            let bi = &basis[i * n..i * n + n];
                            *p = bi.iter().zip(w).map(|(a, x)| a * x).sum();
                        });
                }
                // w -= Σ_i proj[i]·basis[i], fanned out over w in chunks.
                {
                    let basis = &self.basis;
                    let proj = &self.proj;
                    self.w[..n]
                        .par_chunks_mut(ortho_chunk)
                        .enumerate()
                        .for_each(|(ci, wc)| {
                            let k0 = ci * ortho_chunk;
                            let len = wc.len();
                            for i in 0..cols {
                                let coeff = proj[i];
                                let bi = &basis[i * n + k0..i * n + k0 + len];
                                for (wk, bk) in wc.iter_mut().zip(bi) {
                                    *wk -= coeff * bk;
                                }
                            }
                        });
                }
                // H column j accumulates both passes' coefficients.
                for i in 0..cols {
                    self.h[i * max_dim + j] += self.proj[i];
                }
            }
            let hnext = norm2(&self.w[..n]);
            let d = j + 1;

            if hnext < ARNOLDI_BREAKDOWN || d == max_dim {
                dim = d;
                break;
            }
            if tol > 0.0 {
                for a in 0..d {
                    for b in 0..d {
                        self.th[a * d + b] = t * self.h[a * max_dim + b];
                    }
                }
                expm_into(
                    &self.th[..d * d],
                    d,
                    &mut self.exp_th[..d * d],
                    &mut self.expm_scratch,
                );
                if hnext * self.exp_th[(d - 1) * d].abs() < tol {
                    dim = d;
                    break;
                }
            }
            self.h[(j + 1) * max_dim + j] = hnext;
            let inv = 1.0 / hnext;
            let dst = (j + 1) * n;
            for k in 0..n {
                self.basis[dst + k] = self.w[k] * inv;
            }
        }

        // exp(t·H) on the dim×dim leading block, packed tightly.
        for i in 0..dim {
            for j in 0..dim {
                self.th[i * dim + j] = t * self.h[i * max_dim + j];
            }
        }
        expm_into(
            &self.th[..dim * dim],
            dim,
            &mut self.exp_th[..dim * dim],
            &mut self.expm_scratch,
        );

        // out = β · Σ_i basis[i] · exp_th[i,0].
        out[..n].fill(0.0);
        for i in 0..dim {
            let c = beta * self.exp_th[i * dim];
            let bi = &self.basis[i * n..i * n + n];
            for k in 0..n {
                out[k] += c * bi[k];
            }
        }
        dim
    }

    /// Allocation-free exponential-time-differencing step of
    /// `dy/dt = A·y + b` with the source `b` held constant over the step:
    /// `y ← exp(hA)·y + h·φ₁(hA)·b`, written into `out` (`n`).
    ///
    /// Uses the augmented-matrix identity
    /// `exp(h·[[A, b],[0, 0]])·[y; 1] = [exp(hA)y + h·φ₁(hA)b ; 1]`, so the
    /// Krylov projection on the `(n+1)`-dimensional augmented system handles
    /// the φ-function with no extra machinery, reusing this workspace
    /// throughout. `max_dim` / `tol` cap and adaptively truncate the Krylov
    /// subspace exactly as in [`expmv_into`](Self::expmv_into).
    pub fn etd_step_into<F>(
        &mut self,
        matvec: F,
        y: &[Accum],
        b: &[Accum],
        h: Accum,
        max_dim: usize,
        tol: Accum,
        out: &mut [Accum],
    ) where
        F: Fn(&[Accum], &mut [Accum]),
    {
        let n = y.len();
        // Borrow the augmented buffers out so `expmv_into` can take `&mut self`.
        let mut z = std::mem::take(&mut self.aug_in);
        let mut r = std::mem::take(&mut self.aug_out);
        z.resize(n + 1, 0.0);
        r.resize(n + 1, 0.0);
        z[..n].copy_from_slice(y);
        z[n] = 1.0;

        // Augmented operator: [[A, b], [0, 0]] applied to [x; ξ].
        let aug = |zz: &[Accum], o: &mut [Accum]| {
            matvec(&zz[..n], &mut o[..n]);
            let xi = zz[n];
            for k in 0..n {
                o[k] += xi * b[k];
            }
            o[n] = 0.0;
        };
        self.expmv_into(aug, &z, h, max_dim, tol, &mut r);
        out[..n].copy_from_slice(&r[..n]);

        self.aug_in = z;
        self.aug_out = r;
    }
}

/// Matrix-free action `exp(t·A)·v`, via an `m`-step Krylov projection.
///
/// `matvec` computes `A·x`. `m` is the Krylov dimension; Arnoldi stops early
/// on a lucky breakdown. Allocating wrapper around
/// [`KrylovWorkspace::expmv_into`] with `tol = 0` (the fixed full-`m`
/// subspace), reuse a [`KrylovWorkspace`] directly to step without
/// allocating.
pub fn expmv<F>(matvec: F, v: &[Accum], t: Accum, m: usize) -> Vec<Accum>
where
    F: Fn(&[Accum]) -> Vec<Accum>,
{
    let mut ws = KrylovWorkspace::new();
    let mut out = vec![0.0; v.len()];
    ws.expmv_into(|x, ax| ax.copy_from_slice(&matvec(x)), v, t, m, 0.0, &mut out);
    out
}

/// One exponential-time-differencing step of `dy/dt = A·y + b`, with the
/// source `b` held constant across the step:
/// `y ← exp(h·A)·y + h·φ₁(h·A)·b`. Allocating wrapper around
/// [`KrylovWorkspace::etd_step_into`] with `tol = 0` (the fixed full-`m`
/// subspace).
pub fn etd_step<F>(matvec: F, y: &[Accum], b: &[Accum], h: Accum, m: usize) -> Vec<Accum>
where
    F: Fn(&[Accum]) -> Vec<Accum>,
{
    let mut ws = KrylovWorkspace::new();
    let mut out = vec![0.0; y.len()];
    ws.etd_step_into(|x, ax| ax.copy_from_slice(&matvec(x)), y, b, h, m, 0.0, &mut out);
    out
}

fn norm2(a: &[Accum]) -> Accum {
    a.iter().map(|x| x * x).sum::<Accum>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expm_of_zero_is_identity() {
        let e = expm(&[0.0; 9], 3);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((e[i * 3 + j] - want).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn expm_of_rotation_generator() {
        // exp([[0,-θ],[θ,0]]) = [[cosθ,-sinθ],[sinθ,cosθ]].
        let theta = 0.7;
        let e = expm(&[0.0, -theta, theta, 0.0], 2);
        assert!((e[0] - theta.cos()).abs() < 1e-12);
        assert!((e[1] + theta.sin()).abs() < 1e-12);
        assert!((e[2] - theta.sin()).abs() < 1e-12);
        assert!((e[3] - theta.cos()).abs() < 1e-12);
    }

    #[test]
    fn expm_of_diagonal() {
        let e = expm(&[1.5, 0.0, 0.0, -2.0], 2);
        assert!((e[0] - 1.5_f64.exp()).abs() < 1e-11);
        assert!((e[3] - (-2.0_f64).exp()).abs() < 1e-11);
        assert!(e[1].abs() < 1e-13 && e[2].abs() < 1e-13);
    }

    #[test]
    fn expmv_matches_dense_exponential() {
        // Matrix-free Krylov action vs the dense reference exp(tA)·v on the
        // DG Maxwell operator of a small cavity.
        use crate::mesh_gen::structured_box;
        use crate::rhs::MaxwellOperator;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0, Default::default());
        let n = op.n_dof();
        let a = op.assemble_dense();
        let t = 0.05;

        // Dense reference.
        let mut ta = a.clone();
        for x in ta.iter_mut() {
            *x *= t;
        }
        let dense_exp = expm(&ta, n);

        // A deterministic test vector.
        let v: Vec<f64> =
            (0..n).map(|i| (0.3 + i as f64 * 0.017).sin()).collect();
        let mut want = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                want[i] += dense_exp[i * n + j] * v[j];
            }
        }

        let got = expmv(|x| op.apply(x), &v, t, 60);
        let err: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 =
            want.iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!(err < 1e-8 * scale, "Krylov vs dense: err {err}, scale {scale}");
    }

    #[test]
    fn adaptive_expmv_into_stops_early_and_stays_accurate() {
        // With a tolerance, expmv_into must truncate the Krylov subspace
        // well before `max_dim` yet still match the full-dimension result.
        use crate::mesh_gen::structured_box;
        use crate::rhs::MaxwellOperator;
        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0, Default::default());
        let n = op.n_dof();
        let v: Vec<f64> =
            (0..n).map(|i| (0.3 + i as f64 * 0.011).sin()).collect();
        let t = 0.02;

        let mut ws = KrylovWorkspace::new();
        // tol = 0 → the full fixed subspace, as the reference.
        let mut reference = vec![0.0; n];
        let full = ws.expmv_into(
            |x, ax| ax.copy_from_slice(&op.apply(x)),
            &v, t, 80, 0.0, &mut reference,
        );

        // A moderate tolerance must stop well short of `max_dim`.
        let mut got = vec![0.0; n];
        let dim = ws.expmv_into(
            |x, ax| ax.copy_from_slice(&op.apply(x)),
            &v, t, 80, 1e-9, &mut got,
        );
        assert!(dim < full, "no early stop, adaptive {dim}, fixed {full}");

        let err: f64 = got
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 =
            reference.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err < 1e-7 * scale, "adaptive rel.err {}", err / scale);

        // A tighter tolerance never truncates earlier than a looser one.
        let mut tight = vec![0.0; n];
        let dim_tight = ws.expmv_into(
            |x, ax| ax.copy_from_slice(&op.apply(x)),
            &v, t, 80, 1e-13, &mut tight,
        );
        assert!(dim_tight >= dim, "tighter tol {dim_tight} < looser {dim}");
    }

    #[test]
    fn etd_step_matches_analytic_linear_ode() {
        // dy/dt = A·y + b,  A = [[0,-ω],[ω,0]],  b constant.
        // Exact: y(h) = exp(hA)·(y₀ + A⁻¹b) - A⁻¹b.
        let omega = 1.3;
        let a = [0.0, -omega, omega, 0.0];
        let matvec = |x: &[f64]| {
            vec![a[0] * x[0] + a[1] * x[1], a[2] * x[0] + a[3] * x[1]]
        };
        let b = [0.4, -0.7];
        let y0 = [1.0, 0.5];
        let h = 0.6;

        let ainv_b = [b[1] / omega, -b[0] / omega];
        let (c, s) = ((omega * h).cos(), (omega * h).sin());
        let shifted = [y0[0] + ainv_b[0], y0[1] + ainv_b[1]];
        let want = [
            c * shifted[0] - s * shifted[1] - ainv_b[0],
            s * shifted[0] + c * shifted[1] - ainv_b[1],
        ];
        // The augmented system is 3-dimensional; m ≥ 3 makes Krylov exact.
        let got = etd_step(matvec, &y0, &b, h, 8);
        assert!((got[0] - want[0]).abs() < 1e-12, "{got:?} vs {want:?}");
        assert!((got[1] - want[1]).abs() < 1e-12, "{got:?} vs {want:?}");
    }

    #[test]
    fn central_flux_propagation_conserves_energy() {
        // P4.4: a central-flux transient run conserves the discrete field
        // energy yᵀM̃y exactly (up to the Krylov tolerance) over many steps.
        use crate::mesh_gen::structured_box;
        use crate::rhs::MaxwellOperator;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 0.0, Default::default());
        let n = op.n_dof();
        let mm = op.assemble_energy_mass();
        let energy = |y: &[f64]| -> f64 {
            let mut e = 0.0;
            for i in 0..n {
                for j in 0..n {
                    e += y[i] * mm[i * n + j] * y[j];
                }
            }
            e
        };
        let mut y: Vec<f64> =
            (0..n).map(|i| (0.2 + i as f64 * 0.013).sin()).collect();
        let e0 = energy(&y);
        for _ in 0..30 {
            y = expmv(|x| op.apply(x), &y, 0.02, 40);
        }
        let drift = ((energy(&y) - e0) / e0).abs();
        assert!(drift < 1e-7, "energy drift {drift:e}");
    }

    #[test]
    fn adaptive_krylov_dimension_meets_tolerance() {
        // With a tolerance, expmv_into picks the Krylov dimension itself; the
        // result must match the dense reference, and the chosen dimension
        // stay modest.
        use crate::mesh_gen::structured_box;
        use crate::rhs::MaxwellOperator;
        let mesh = structured_box(1, 1, 1, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0, Default::default());
        let n = op.n_dof();
        let t = 0.05;

        let a = op.assemble_dense();
        let ta: Vec<f64> = a.iter().map(|x| x * t).collect();
        let dense_exp = expm(&ta, n);
        let v: Vec<f64> =
            (0..n).map(|i| (0.3 + i as f64 * 0.017).sin()).collect();
        let mut want = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                want[i] += dense_exp[i * n + j] * v[j];
            }
        }

        let mut ws = KrylovWorkspace::new();
        let mut got = vec![0.0; n];
        let dim = ws.expmv_into(
            |x, ax| ax.copy_from_slice(&op.apply(x)),
            &v, t, 200, 1e-9, &mut got,
        );
        let err: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = want.iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!(err < 1e-7 * scale, "adaptive expmv err {}", err / scale);
        assert!(dim > 0 && dim < n, "chosen Krylov dim {dim}");
    }
}
