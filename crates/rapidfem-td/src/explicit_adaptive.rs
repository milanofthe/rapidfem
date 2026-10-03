// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Adaptive embedded low-storage Runge-Kutta time integration.
//!
//! The Kennedy-Carpenter-Lewis RK4(3)5[2R+]C scheme, five stages, fourth
//! order with an embedded third-order error estimator, in the "2R+"
//! low-storage form (two state-shaped registers plus a third for the
//! embedded-error accumulator). Per step the matvec count is identical to
//! [`crate::explicit::LserkWorkspace`]'s LSERK4 (five), but each step also
//! yields a per-DOF error vector the caller's controller can read to grow
//! or shrink the next step.
//!
//! That decouples the integrator from `cfl_dt`: a non-normal upwind-DG
//! operator that smuggles transient growth past a fixed CFL probe still
//! shows up in this stepper's embedded error, and the controller cuts the
//! step before the trajectory diverges. The LSERK4 stepper is left in place
//! for the non-adaptive path where the user pins `dt` explicitly.

use rayon::prelude::*;

use crate::constants::{Field, KCL_A, KCL_B, KCL_BHAT, KCL_STAGES};

/// Reusable workspace for the KCL RK4(3)5[2R+]C embedded adaptive stepper,
/// owns the stage register, the embedded-error accumulator, and the matvec
/// scratch so a stepped transient allocates nothing once warmed.
///
/// Storage layout matches NodePy's `TwoRRungeKuttaPair.__step__`
/// (Kennedy-Carpenter-Lewis 2000 + Ketcheson 2010): the caller's `y`
/// plays the role of the main accumulator `S2`, this struct owns the
/// stage-state register `Y_i`, the `dt·F` register, and the embedded-error
/// accumulator. The "+" in 2R+ is the embedded-error register, 2 main
/// state registers (`y` + `stage`) plus 1 for `e`, plus the matvec scratch.
pub struct KclWorkspace {
    /// Stage state register `Y_i`, built fresh each interior stage as
    /// `S2 + (A_{i+1,i} - b_i)·dt·F_{i-1}`, then fed to the matvec.
    stage: Vec<Field>,
    /// Embedded-error accumulator `e = Σ_i (b̂_i - b_i)·dt·F_i`, the per-
    /// DOF difference between the fourth-order and third-order solutions,
    /// the quantity an adaptive controller normalises and tests.
    e: Vec<Field>,
    /// Holds `dt · F_i` for the current stage. Reused as the matvec
    /// output buffer and then in-place dt-scaled, then read by the post-
    /// stage updates to `y`, `e`, and the next stage's evaluation point.
    k: Vec<Field>,
}

impl Default for KclWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl KclWorkspace {
    /// An empty workspace; its buffers grow to fit on the first step.
    pub fn new() -> Self {
        KclWorkspace { stage: Vec::new(), e: Vec::new(), k: Vec::new() }
    }

    fn ensure(&mut self, n: usize) {
        if self.stage.len() < n {
            self.stage.resize(n, 0.0);
        }
        if self.e.len() < n {
            self.e.resize(n, 0.0);
        }
        if self.k.len() < n {
            self.k.resize(n, 0.0);
        }
    }

    /// One KCL RK4(3)5[2R+]C step of `dy/dt = A·y`, advancing `y` in place
    /// by `dt` and writing the embedded-error vector into `err`.
    /// `matvec(x, ax)` writes `A·x` into `ax`.
    ///
    /// `err` is the per-DOF difference between the fourth-order main
    /// solution and the third-order embedded one, a controller takes its
    /// weighted L2 norm against `atol + rtol·|y|` and accepts or rejects
    /// the step. Allocation-free once warmed.
    pub fn step_into<F>(
        &mut self,
        matvec: F,
        y: &mut [Field],
        err: &mut [Field],
        dt: Field,
    ) where
        F: Fn(&[Field], &mut [Field]),
    {
        let n = y.len();
        assert_eq!(err.len(), n, "err length must equal state length");
        self.ensure(n);

        // Stage 0, matvec at y_n, accumulate b₀ into y (which becomes the
        // running S2), seed e with (b̂₀-b₀)·dt·F₀.
        matvec(&y[..n], &mut self.k[..n]);
        let b0 = KCL_B[0];
        let e0 = KCL_BHAT[0] - b0;
        y[..n]
            .par_iter_mut()
            .zip(self.k[..n].par_iter_mut())
            .zip(self.e[..n].par_iter_mut())
            .for_each(|((yi, ki), ei)| {
                let f = dt * *ki;          // f = dt·F₀
                *ki = f;                   // keep dt·F₀ in self.k for stage 1
                *yi += b0 * f;             // S2 = y_n + b₀·dt·F₀
                *ei = e0 * f;              // e  = (b̂₀-b₀)·dt·F₀
            });

        // Stages 1..s, build the new stage state from current S2 (=y) and
        // the carried-over dt·F_{i-1}, evaluate F_i, accumulate b_i·dt·F_i
        // into y and (b̂_i-b_i)·dt·F_i into e. Across iterations self.k
        // holds dt·F_{i-1} on entry and dt·F_i on exit.
        for stage in 1..KCL_STAGES {
            let amb = KCL_A[stage - 1] - KCL_B[stage - 1];
            // Y_i := S2 + (A_{i,i-1} - b_{i-1})·dt·F_{i-1}.
            y[..n]
                .par_iter()
                .zip(self.k[..n].par_iter())
                .zip(self.stage[..n].par_iter_mut())
                .for_each(|((yi, ki), si)| {
                    *si = *yi + amb * *ki;
                });
            matvec(&self.stage[..n], &mut self.k[..n]);
            let b = KCL_B[stage];
            let eweight = KCL_BHAT[stage] - b;
            // Scale to dt·F_i in place, accumulate into S2 and e.
            self.k[..n]
                .par_iter_mut()
                .zip(y[..n].par_iter_mut())
                .zip(self.e[..n].par_iter_mut())
                .for_each(|((ki, yi), ei)| {
                    let f = dt * *ki;
                    *ki = f;
                    *yi += b * f;
                    *ei += eweight * f;
                });
        }

        err[..n].copy_from_slice(&self.e[..n]);
    }

    /// One KCL step of the driven system `dy/dt = A·y + b`, with the soft
    /// point source `b = e_{source_dof}·source_value` held constant across
    /// the step. Advances `y` in place by `dt` and writes the embedded-error
    /// vector into `err`.
    ///
    /// The zeroth-order source hold mirrors [`crate::explicit::LserkWorkspace::step_driven_into`]
    /// and the exponential `step_driven`, so the integrators drive a port
    /// transient identically bar their own truncation error.
    pub fn step_driven_into<F>(
        &mut self,
        matvec: F,
        y: &mut [Field],
        err: &mut [Field],
        dt: Field,
        source_dof: usize,
        source_value: Field,
    ) where
        F: Fn(&[Field], &mut [Field]),
    {
        let n = y.len();
        assert_eq!(err.len(), n, "err length must equal state length");
        self.ensure(n);

        // Stage 0, RHS is A·y_n + b·g (zeroth-order hold across the step,
        // same convention as the LSERK4 driven path).
        matvec(&y[..n], &mut self.k[..n]);
        if source_dof < n {
            self.k[source_dof] += source_value;
        }
        let b0 = KCL_B[0];
        let e0 = KCL_BHAT[0] - b0;
        y[..n]
            .par_iter_mut()
            .zip(self.k[..n].par_iter_mut())
            .zip(self.e[..n].par_iter_mut())
            .for_each(|((yi, ki), ei)| {
                let f = dt * *ki;
                *ki = f;
                *yi += b0 * f;
                *ei = e0 * f;
            });

        for stage in 1..KCL_STAGES {
            let amb = KCL_A[stage - 1] - KCL_B[stage - 1];
            y[..n]
                .par_iter()
                .zip(self.k[..n].par_iter())
                .zip(self.stage[..n].par_iter_mut())
                .for_each(|((yi, ki), si)| {
                    *si = *yi + amb * *ki;
                });
            matvec(&self.stage[..n], &mut self.k[..n]);
            if source_dof < n {
                self.k[source_dof] += source_value;
            }
            let b = KCL_B[stage];
            let eweight = KCL_BHAT[stage] - b;
            self.k[..n]
                .par_iter_mut()
                .zip(y[..n].par_iter_mut())
                .zip(self.e[..n].par_iter_mut())
                .for_each(|((ki, yi), ei)| {
                    let f = dt * *ki;
                    *ki = f;
                    *yi += b * f;
                    *ei += eweight * f;
                });
        }

        err[..n].copy_from_slice(&self.e[..n]);
    }

    /// One KCL step of the driven system `dy/dt = A·y + b`, with the
    /// **full source vector** `b = source` held constant across the step.
    /// Advances `y` in place by `dt` and writes the embedded-error vector
    /// into `err`.
    ///
    /// The vector-source generalisation of [`step_driven_into`](Self::step_driven_into),
    /// the modal-port injection path, where the source spreads over every
    /// port-face DOF. Zeroth-order hold over the step, like the explicit
    /// and exponential counterparts.
    pub fn step_with_source_into<F>(
        &mut self,
        matvec: F,
        y: &mut [Field],
        err: &mut [Field],
        dt: Field,
        source: &[Field],
    ) where
        F: Fn(&[Field], &mut [Field]),
    {
        let n = y.len();
        assert_eq!(err.len(), n, "err length must equal state length");
        assert_eq!(source.len(), n, "source length must equal state length");
        self.ensure(n);

        // Stage 0, RHS is A·y_n + b (full source vector, zeroth-order
        // hold), the modal-port injection path.
        matvec(&y[..n], &mut self.k[..n]);
        let b0 = KCL_B[0];
        let e0 = KCL_BHAT[0] - b0;
        y[..n]
            .par_iter_mut()
            .zip(self.k[..n].par_iter_mut())
            .zip(self.e[..n].par_iter_mut())
            .zip(&source[..n])
            .for_each(|(((yi, ki), ei), si)| {
                let f = dt * (*ki + *si);
                *ki = f;
                *yi += b0 * f;
                *ei = e0 * f;
            });

        for stage in 1..KCL_STAGES {
            let amb = KCL_A[stage - 1] - KCL_B[stage - 1];
            y[..n]
                .par_iter()
                .zip(self.k[..n].par_iter())
                .zip(self.stage[..n].par_iter_mut())
                .for_each(|((yi, ki), si)| {
                    *si = *yi + amb * *ki;
                });
            matvec(&self.stage[..n], &mut self.k[..n]);
            let b = KCL_B[stage];
            let eweight = KCL_BHAT[stage] - b;
            self.k[..n]
                .par_iter_mut()
                .zip(y[..n].par_iter_mut())
                .zip(self.e[..n].par_iter_mut())
                .zip(&source[..n])
                .for_each(|(((ki, yi), ei), si)| {
                    let f = dt * (*ki + *si);
                    *ki = f;
                    *yi += b * f;
                    *ei += eweight * f;
                });
        }

        err[..n].copy_from_slice(&self.e[..n]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kcl_is_fourth_order_on_a_linear_ode() {
        // dy/dt = A·y with A = [[0,-ω],[ω,0]], a pure rotation, exact
        // solution y(t) = R(ωt)·y₀. Halving the step must quarter the error
        // twice over (a fourth-order rate ≈ 16). Same probe as LSERK4's so
        // the orders can be compared apples-to-apples.
        let omega = 1.3_f64;
        let matvec = |x: &[Field], ax: &mut [Field]| {
            ax[0] = -omega * x[1];
            ax[1] = omega * x[0];
        };
        let t_end = 1.7_f64;
        let integrate = |nsteps: usize| -> [Field; 2] {
            let dt = t_end / nsteps as Field;
            let mut y = [1.0_f64, 0.4_f64];
            let mut err = [0.0_f64, 0.0_f64];
            let mut ws = KclWorkspace::new();
            for _ in 0..nsteps {
                ws.step_into(matvec, &mut y, &mut err, dt);
            }
            y
        };
        let (c, s) = ((omega * t_end).cos(), (omega * t_end).sin());
        let exact = [c * 1.0 - s * 0.4, s * 1.0 + c * 0.4];
        let err = |n: usize| -> f64 {
            let y = integrate(n);
            ((y[0] - exact[0]).powi(2) + (y[1] - exact[1]).powi(2)).sqrt()
        };
        let rate = err(20) / err(40);
        assert!(rate > 12.0, "KCL not ~4th order, error ratio {rate:.1}");
    }

    #[test]
    fn kcl_embedded_error_is_third_order() {
        // The embedded-error vector e = y_emb - y_main should be O(dt^4) for
        // a 5(4) method (the embedded is third-order accurate, so its
        // difference from the fourth-order main scales as the embedded
        // truncation: dt^4). Halving dt should drop ‖e‖ by ~16. Probe on
        // the same rotation ODE.
        let omega = 1.3_f64;
        let matvec = |x: &[Field], ax: &mut [Field]| {
            ax[0] = -omega * x[1];
            ax[1] = omega * x[0];
        };
        let probe = |dt: Field| -> f64 {
            let mut y = [1.0_f64, 0.4_f64];
            let mut err = [0.0_f64, 0.0_f64];
            let mut ws = KclWorkspace::new();
            ws.step_into(matvec, &mut y, &mut err, dt);
            (err[0] * err[0] + err[1] * err[1]).sqrt()
        };
        let e1 = probe(0.05);
        let e2 = probe(0.025);
        let rate = e1 / e2;
        // Embedded is third-order ⇒ difference to fourth-order is O(dt^4) ⇒
        // rate ≈ 16 under step halving. Loosen to >10 for noise margin.
        assert!(rate > 10.0, "KCL embedded error rate {rate:.1} (want ≈16)");
    }

    #[test]
    fn kcl_matches_the_exponential_propagator_below_cfl() {
        // Against the already-validated Krylov exponential propagator: for
        // a step well inside the CFL limit, one KCL step agrees with
        // exp(dt·A)·y to the scheme's O(dt^5) truncation error. Same probe
        // as the LSERK4 cross-check.
        use crate::mesh_gen::structured_box;
        use crate::propagator::expmv;
        use crate::rhs::MaxwellOperator;

        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let y0: Vec<Field> =
            (0..n).map(|i| (0.3 + i as Field * 0.017).sin()).collect();

        let dt = 1e-3;
        let mut y_rk = y0.clone();
        let mut err = vec![0.0; n];
        let mut ws = KclWorkspace::new();
        ws.step_into(|x, ax| op.apply_into(x, ax), &mut y_rk, &mut err, dt);

        let y_exp = expmv(|x| op.apply(x), &y0, dt, 40);

        let e: f64 = y_rk
            .iter()
            .zip(&y_exp)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = y_exp.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            e < 1e-9 * scale,
            "KCL vs exponential step: rel.err {}",
            e / scale,
        );
    }

    #[test]
    fn kcl_driven_matches_etd_below_cfl() {
        // The driven KCL step against the exponential ETD step (single-DOF
        // source held constant across the step): one driven KCL step well
        // inside the CFL limit agrees with `etd_step` to O(dt^5). The
        // zeroth-order hold convention is shared.
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;
        use crate::rhs::MaxwellOperator;

        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let y0: Vec<Field> =
            (0..n).map(|i| (0.2 + i as Field * 0.019).sin()).collect();

        let sdof = n / 3;
        let src = 0.7;
        let dt = 1e-3;

        let mut y_rk = y0.clone();
        let mut err = vec![0.0; n];
        let mut ws = KclWorkspace::new();
        ws.step_driven_into(
            |x, ax| op.apply_into(x, ax), &mut y_rk, &mut err, dt, sdof, src,
        );

        let mut b = vec![0.0; n];
        b[sdof] = src;
        let y_etd = etd_step(|x| op.apply(x), &y0, &b, dt, 40);

        let e: f64 = y_rk
            .iter()
            .zip(&y_etd)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = y_etd.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            e < 1e-9 * scale,
            "driven KCL vs ETD step: rel.err {}",
            e / scale,
        );
    }

    #[test]
    fn kcl_vector_source_matches_etd_below_cfl() {
        // The vector-source KCL step against the exponential ETD step with
        // the same full source vector `b`: a single driven KCL step well
        // inside the CFL limit agrees with `etd_step` to O(dt^5). This is
        // the modal-port injection path (b spread over many DOFs).
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;
        use crate::rhs::MaxwellOperator;

        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0);
        let n = op.n_dof();
        let y0: Vec<Field> =
            (0..n).map(|i| (0.2 + i as Field * 0.019).sin()).collect();

        let b: Vec<Field> =
            (0..n).map(|i| 0.4 * (0.11 * i as Field).cos()).collect();
        let dt = 1e-3;

        let mut y_rk = y0.clone();
        let mut err = vec![0.0; n];
        let mut ws = KclWorkspace::new();
        ws.step_with_source_into(
            |x, ax| op.apply_into(x, ax), &mut y_rk, &mut err, dt, &b,
        );

        let y_etd = etd_step(|x| op.apply(x), &y0, &b, dt, 40);

        let e: f64 = y_rk
            .iter()
            .zip(&y_etd)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = y_etd.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            e < 1e-9 * scale,
            "vector-source KCL vs ETD step: rel.err {}",
            e / scale,
        );
    }
}
