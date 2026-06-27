// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! The per-frequency electrical state, in one place.
//!
//! Before this existed, every site that needed a wavenumber recomputed
//! `k0 = 2π·freq/c0` (9 call sites) and every port that needed the angular
//! frequency recovered it as `ω = k0·c0` (4 sites) — a round-trip that is both
//! duplicated and, once `k0` is non-dimensionalized, wrong. `Excitation` is the
//! single source of truth: build it once per frequency, pass `&Excitation` to
//! assembly and ports, read `k0` for wavenumbers and `omega` for the angular
//! frequency directly (never recovered from `k0`).
//!
//! This separation is what makes the characteristic-length non-dimensionalization
//! (lever ④) a one-line change: `k0` can be scaled to `κ = k0·L0` at the
//! constructor while `omega` stays the true physical angular frequency, so the
//! ω-dependent material/impedance physics is untouched. See
//! `derivations/basis_nondim/`.

use crate::constants::{C0, PI};

/// The electromagnetic state at one frequency: the angular frequency and the
/// (possibly length-normalized) wavenumber, derived once and carried together.
#[derive(Clone, Copy, Debug)]
pub struct Excitation {
    /// Drive frequency f (Hz).
    pub freq: f64,
    /// Wavenumber in the working length units: κ = k₀·L₀ when the mesh is
    /// non-dimensionalized by the characteristic length L₀, else the physical
    /// k₀ = ω/c₀ (rad/m). This is the quantity the wave operator and the
    /// propagation constants β must use to stay consistent with the geometry.
    pub k0: f64,
    /// Angular frequency ω = 2πf (rad/s). The true physical ω — used by the
    /// frequency-dependent material/circuit terms (skin depth, R+jωL+1/jωC),
    /// which are *not* length-coupled and must see the real frequency.
    pub omega: f64,
    /// Characteristic length L₀ (m) the geometry was divided by (1.0 = none).
    pub l0: f64,
}

impl Excitation {
    /// Build the electrical state for a drive frequency `freq` (Hz) on a mesh
    /// non-dimensionalized by characteristic length `l0` (use `1.0` for an
    /// un-normalized, physical-unit mesh).
    ///
    /// The free-space wavenumber is scaled to `κ = k₀·L₀` so it matches the
    /// `x/L₀` coordinates the solver assembles on, while `ω` stays the physical
    /// angular frequency. For `l0 = 1.0` this is exactly the physical state.
    pub fn new(freq: f64, l0: f64) -> Self {
        let omega = 2.0 * PI * freq;
        Excitation { freq, k0: omega / C0 * l0, omega, l0 }
    }

    /// Length-coupled angular frequency ω̃ = κ·c₀ = ω·L₀, used by wave
    /// impedances `Z = ω̃·μ/β` so they stay invariant under the L₀ scaling
    /// (both ω̃ and β carry one factor of L₀, which cancels). Equals the
    /// physical ω when `l0 = 1`.
    #[inline]
    pub fn omega_scaled(&self) -> f64 { self.k0 * C0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// l0 = 1 is the physical state: k0 = ω/c0, ω̃ = ω.
    #[test]
    fn physical_state_at_unit_l0() {
        let e = Excitation::new(10e9, 1.0);
        assert!((e.k0 - e.omega / C0).abs() < 1e-9 * e.k0);
        assert!((e.omega_scaled() - e.omega).abs() < 1e-6 * e.omega);
    }

    /// Non-dimensionalization scales the wavenumber by L₀ (κ = k₀·L₀) while ω
    /// stays physical, and the length-coupled ω̃ = κ·c₀ scales with L₀ in step.
    #[test]
    fn wavenumber_scales_with_l0_omega_stays_physical() {
        let f = 10e9;
        let phys = Excitation::new(f, 1.0);
        let l0 = 4.5e-3;
        let norm = Excitation::new(f, l0);
        assert!((norm.k0 - phys.k0 * l0).abs() < 1e-12 * norm.k0, "κ = k0·L0");
        assert!((norm.omega - phys.omega).abs() < 1e-9 * phys.omega, "ω physical");
        assert!((norm.omega_scaled() - phys.omega * l0).abs() < 1e-6 * norm.omega_scaled(),
            "ω̃ = ω·L0");
    }
}
