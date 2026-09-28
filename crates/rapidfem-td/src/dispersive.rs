// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Dispersive materials via auxiliary differential equations (ADE).
//!
//! A frequency-dependent permittivity becomes, in the time domain, an
//! auxiliary polarisation field with its own ODE. For a Debye medium
//!
//! ```text
//!   ε(ω) = ε_∞ + (ε_s − ε_∞) / (1 + iωτ)
//! ```
//!
//! the polarisation `P` obeys the first-order relaxation equation
//!
//! ```text
//!   τ·Ṗ + P = (ε_s − ε_∞)·E ,    D = ε_∞·E + P .
//! ```
//!
//! The augmented `(E, H, …, P)` system stays linear with constant
//! coefficients, so the Krylov/ETD propagator carries it unchanged, `P` is
//! just extra per-node state. (Drude / Lorentz media are analogous, with a
//! second-order auxiliary equation.)

use crate::constants::Field;

/// A Debye dispersive material, the relaxation parameters mirror the
/// frequency-domain `rapidfem.Debye`.
#[derive(Clone, Copy, Debug)]
pub struct DebyeMaterial {
    /// High-frequency permittivity `ε_∞`.
    pub eps_inf: Field,
    /// Static (zero-frequency) permittivity `ε_s`.
    pub eps_static: Field,
    /// Relaxation time `τ`.
    pub tau: Field,
}

impl DebyeMaterial {
    /// Analytic complex relative permittivity at angular frequency `ω`,
    /// returned as `(real, imag)`.
    pub fn permittivity(&self, omega: Field) -> (Field, Field) {
        let d = self.eps_static - self.eps_inf;
        let denom = 1.0 + (omega * self.tau).powi(2);
        (self.eps_inf + d / denom, -d * omega * self.tau / denom)
    }

    /// The relaxation ODE in the form `Ṗ = a·P + b(t)`: returns the constant
    /// `a = -1/τ` and the source gain `g` such that `b(t) = g·E(t)`.
    pub fn relaxation_coeffs(&self) -> (Field, Field) {
        (-1.0 / self.tau, (self.eps_static - self.eps_inf) / self.tau)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagator::etd_step2;

    #[test]
    fn debye_ade_reproduces_the_analytic_permittivity() {
        // Integrate the relaxation ODE Ṗ = a·P + g·cos(ωt) with the ETD
        // propagator; the steady-state polarisation phasor must reconstruct
        // the analytic Debye ε(ω) across a frequency sweep.
        let mat = DebyeMaterial { eps_inf: 2.0, eps_static: 5.0, tau: 0.3 };
        let (a, g) = mat.relaxation_coeffs();
        let matvec = |p: &[f64]| vec![a * p[0]];

        for &omega in &[0.5, 1.5, 4.0, 10.0] {
            let src = |t: f64| g * (omega * t).cos();
            let n_per = 200;
            let h = 2.0 * std::f64::consts::PI / omega / n_per as f64;

            // Settle into the periodic steady state.
            let mut p = vec![0.0];
            let mut t = 0.0;
            for _ in 0..40 * n_per {
                p = etd_step2(matvec, &p, &[src(t)], &[src(t + h)], h, 4);
                t += h;
            }
            // Extract the polarisation phasor over one period.
            let (mut pc, mut ps) = (0.0, 0.0);
            for _ in 0..n_per {
                p = etd_step2(matvec, &p, &[src(t)], &[src(t + h)], h, 4);
                t += h;
                pc += p[0] * (omega * t).cos();
                ps += p[0] * (omega * t).sin();
            }
            let norm = 2.0 / n_per as f64;
            // P(t) = Re[(pc·norm - i·ps·norm)·e^{iωt}];  E phasor = 1;
            // ε(ω) = ε_∞ + P_phasor.
            let eps_re = mat.eps_inf + pc * norm;
            let eps_im = -ps * norm;
            let (want_re, want_im) = mat.permittivity(omega);
            assert!(
                (eps_re - want_re).abs() < 2e-3,
                "ω={omega}: Re ε = {eps_re:.4}, analytic {want_re:.4}"
            );
            assert!(
                (eps_im - want_im).abs() < 2e-3,
                "ω={omega}: Im ε = {eps_im:.4}, analytic {want_im:.4}"
            );
        }
    }
}
