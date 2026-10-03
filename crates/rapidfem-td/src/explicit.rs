// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Explicit low-storage Runge-Kutta time integration.
//!
//! The semi-discrete DG system `dy/dt = A·y` can be advanced two ways. The
//! [`propagator`](crate::propagator) takes the exponential route, exact at
//! any step size but costing a Krylov subspace per step. This module is the
//! explicit alternative: the Carpenter-Kennedy 5-stage 4th-order
//! low-storage Runge-Kutta scheme (LSERK4), the standard nodal-DG
//! integrator (Hesthaven & Warburton, *Nodal DG Methods*).
//!
//! A step is five matvecs and two state registers, far cheaper than an
//! exponential step, but the scheme is only **conditionally** stable: the
//! step is bounded by a CFL limit `dt ≲ C / ρ(A)` set by the spectral
//! radius of the operator. On a mesh with a wide element-size range the
//! exponential integrator, immune to that limit, can still win per unit of
//! simulated time; on a near-uniform mesh the cheap explicit step wins.
//! The two are kept side by side so the choice can be measured, not
//! guessed.

use rayon::prelude::*;

use crate::constants::{Field, LSERK4_A, LSERK4_B, LSERK4_STAGES};

/// The source `b` of the driven system `dy/dt = A·y + b`, held constant
/// across a step (zeroth-order hold). The hold mirrors the exponential ETD
/// step of the [`propagator`](crate::propagator), so the explicit, adaptive
/// and exponential integrators drive a transient identically bar their own
/// truncation error.
#[derive(Clone, Copy, Debug)]
pub enum Source<'a> {
    /// A soft point source, `b = e_dof·value`.
    Point { dof: usize, value: Field },
    /// A full source vector, e.g. a modal port pattern spread over every
    /// port-face DOF.
    Vector(&'a [Field]),
}

impl Source<'_> {
    /// `k += b`, called on each stage's matvec output. Adding into `k`
    /// before the stage update rounds exactly like adding inside it.
    pub(crate) fn add_to(self, k: &mut [Field]) {
        match self {
            Source::Point { dof, value } => {
                if dof < k.len() {
                    k[dof] += value;
                }
            }
            Source::Vector(b) => {
                assert_eq!(b.len(), k.len(), "source length must equal state length");
                k.par_iter_mut().zip(b.par_iter()).for_each(|(ki, bi)| *ki += *bi);
            }
        }
    }
}

/// Reusable workspace for the LSERK4 explicit stepper, owns the residual
/// register and the matvec scratch, so a stepped transient allocates
/// nothing once warmed.
pub struct LserkWorkspace {
    /// Residual register `p`, the one extra state vector low-storage RK
    /// carries between stages.
    p: Vec<Field>,
    /// Matvec output `A·y` for the current stage.
    k: Vec<Field>,
}

impl Default for LserkWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

impl LserkWorkspace {
    /// An empty workspace; its buffers grow to fit on the first step.
    pub fn new() -> Self {
        LserkWorkspace { p: Vec::new(), k: Vec::new() }
    }

    fn ensure(&mut self, n: usize) {
        if self.p.len() < n {
            self.p.resize(n, 0.0);
        }
        if self.k.len() < n {
            self.k.resize(n, 0.0);
        }
    }

    /// One LSERK4 step of `dy/dt = A·y + b`, advancing `y` in place by
    /// `dt`; `source` is `b` held across the step, `None` for the free
    /// system. `matvec(x, ax)` writes `A·x` into `ax`. After the buffers
    /// have grown once to fit `n`, this allocates nothing, the form to call
    /// in a step loop.
    ///
    /// The scheme is fourth-order accurate and **conditionally stable**:
    /// `dt` past the operator's CFL limit makes the iteration diverge. The
    /// caller owns the step-size choice.
    pub fn step_into<F>(
        &mut self,
        matvec: F,
        y: &mut [Field],
        dt: Field,
        source: Option<Source<'_>>,
    ) where
        F: Fn(&[Field], &mut [Field]),
    {
        let n = y.len();
        self.ensure(n);
        // The residual register starts each step at zero (a[0] = 0 would
        // zero it on the first stage anyway; the explicit fill keeps the
        // first stage from reading a stale register).
        self.p[..n].fill(0.0);

        for stage in 0..LSERK4_STAGES {
            // k = A·y + b at the current stage state.
            matvec(y, &mut self.k[..n]);
            if let Some(s) = source {
                s.add_to(&mut self.k[..n]);
            }
            let (a, b) = (LSERK4_A[stage], LSERK4_B[stage]);
            // p ← a·p + dt·k ;  y ← y + b·p. Both updates are per-index
            // independent, so they fan out across the rayon pool exactly
            // like the operator's own apply.
            self.p[..n]
                .par_iter_mut()
                .zip(y[..n].par_iter_mut())
                .zip(&self.k[..n])
                .for_each(|((pi, yi), ki)| {
                    *pi = a * *pi + dt * *ki;
                    *yi += b * *pi;
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lserk4_is_fourth_order_on_a_linear_ode() {
        // dy/dt = A·y with A = [[0,-ω],[ω,0]], a pure rotation, exact
        // solution y(t) = R(ωt)·y₀. Halving the step must quarter the error
        // twice over (a fourth-order rate ≈ 16).
        let omega = 1.3_f64;
        let matvec = |x: &[Field], ax: &mut [Field]| {
            ax[0] = -omega * x[1];
            ax[1] = omega * x[0];
        };
        let t_end = 1.7_f64;
        let integrate = |nsteps: usize| -> [Field; 2] {
            let dt = t_end / nsteps as Field;
            let mut y = [1.0_f64, 0.4_f64];
            let mut ws = LserkWorkspace::new();
            for _ in 0..nsteps {
                ws.step_into(matvec, &mut y, dt, None);
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
        assert!(rate > 12.0, "LSERK4 not ~4th order, error ratio {rate:.1}");
    }

    #[test]
    fn lserk4_matches_the_exponential_propagator_below_cfl() {
        // Against the already-validated Krylov exponential propagator: for
        // a step well inside the CFL limit, one LSERK4 step agrees with
        // exp(dt·A)·y to the scheme's O(dt⁵) truncation error.
        use crate::mesh_gen::structured_box;
        use crate::propagator::expmv;
        use crate::rhs::MaxwellOperator;

        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0, Default::default());
        let n = op.n_dof();
        let y0: Vec<Field> =
            (0..n).map(|i| (0.3 + i as Field * 0.017).sin()).collect();

        // A conservatively small step, deep inside any CFL limit, so the
        // O(dt⁵) error is what is being measured.
        let dt = 1e-3;
        let mut y_rk = y0.clone();
        let mut ws = LserkWorkspace::new();
        ws.step_into(|x, ax| op.apply_into(x, ax), &mut y_rk, dt, None);

        let y_exp = expmv(|x| op.apply(x), &y0, dt, 40);

        let err: f64 = y_rk
            .iter()
            .zip(&y_exp)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = y_exp.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            err < 1e-9 * scale,
            "LSERK4 vs exponential step: rel.err {}",
            err / scale,
        );
    }

    #[test]
    fn lserk4_is_stable_below_cfl_and_diverges_above() {
        // The defining property of the explicit scheme: bounded well below
        // the CFL limit, divergent well above it. The exact limit in
        // `dt·ρ(A)` depends on where the spectrum sits, upwind-flux
        // dissipation pushes it past the bare 2.8 imaginary-axis bound,
        // so the test brackets it with a wide margin rather than pinning a
        // constant. `ρ(A)` is estimated by a short power iteration on the
        // magnitude.
        use crate::mesh_gen::structured_box;
        use crate::rhs::MaxwellOperator;

        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0, Default::default());
        let n = op.n_dof();

        // Power iteration: ‖A·v‖/‖v‖ approaches ρ(A) as v aligns with the
        // largest-magnitude eigenvector.
        let mut v: Vec<Field> =
            (0..n).map(|i| (0.7 + i as Field * 0.013).sin()).collect();
        let mut rho = 0.0;
        for _ in 0..60 {
            let av = op.apply(&v);
            let nv: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let na: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
            rho = na / nv;
            let inv = 1.0 / na;
            v = av.iter().map(|x| x * inv).collect();
        }
        assert!(rho > 0.0, "power iteration gave ρ = {rho}");

        let run = |dt: Field, steps: usize| -> f64 {
            let mut y: Vec<Field> =
                (0..n).map(|i| (i as Field * 0.05).cos()).collect();
            let mut ws = LserkWorkspace::new();
            for _ in 0..steps {
                ws.step_into(|x, ax| op.apply_into(x, ax), &mut y, dt, None);
            }
            y.iter().map(|x| x * x).sum::<f64>().sqrt()
        };

        // Well below the limit (dt·ρ ≈ 2.0): the field stays bounded.
        let stable = run(2.0 / rho, 400);
        assert!(stable.is_finite() && stable < 1e3, "below CFL grew: {stable}");

        // Well above any RK4 stability region (dt·ρ ≈ 12, far past the
        // region's ~3.5 maximum extent): the iteration diverges.
        let unstable = run(12.0 / rho, 400);
        assert!(
            !unstable.is_finite() || unstable > 1e6,
            "above CFL stayed bounded: {unstable}",
        );
    }

    #[test]
    fn lserk4_driven_matches_etd_below_cfl() {
        // The driven explicit step against the exponential ETD step with
        // the same held source `b`: for a step well inside the CFL limit,
        // one driven LSERK4 step agrees with `etd_step` (dy/dt = A·y + b)
        // to the scheme's O(dt^5) truncation error. Both source kinds: a
        // single-DOF point source and a vector spread over many DOFs (the
        // modal-port injection path).
        use crate::mesh_gen::structured_box;
        use crate::propagator::etd_step;
        use crate::rhs::MaxwellOperator;

        let mesh = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        let op = MaxwellOperator::new(&mesh, 2, 1.0, Default::default());
        let n = op.n_dof();
        let y0: Vec<Field> =
            (0..n).map(|i| (0.2 + i as Field * 0.019).sin()).collect();
        let dt = 1e-3; // deep inside the CFL limit

        let sdof = n / 3; // an arbitrary interior source DOF
        let mut point = vec![0.0; n];
        point[sdof] = 0.7;
        let spread: Vec<Field> =
            (0..n).map(|i| 0.4 * (0.11 * i as Field).cos()).collect();
        let cases = [
            ("point", Source::Point { dof: sdof, value: 0.7 }, &point),
            ("vector", Source::Vector(&spread), &spread),
        ];

        for (label, source, b) in cases {
            let mut y_rk = y0.clone();
            let mut ws = LserkWorkspace::new();
            ws.step_into(|x, ax| op.apply_into(x, ax), &mut y_rk, dt, Some(source));

            let y_etd = etd_step(|x| op.apply(x), &y0, b, dt, 40);

            let err: f64 = y_rk
                .iter()
                .zip(&y_etd)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            let scale: f64 = y_etd.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                err < 1e-9 * scale,
                "{label} driven LSERK4 vs ETD step: rel.err {}",
                err / scale,
            );
        }
    }
}
