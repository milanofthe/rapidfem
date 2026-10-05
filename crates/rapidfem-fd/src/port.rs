// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Port trait: unified interface for all boundary condition types.
//! Assembly uses this trait to handle the driven ports and the passive
//! boundaries (ABC, SIBC, lumped elements) uniformly.

use num_complex::Complex64 as C64;
use crate::excitation::Excitation;

/// Unified port interface for the Robin boundary-condition ports.
///
/// A driven modal port supplies its number, `beta` and `port_mode_3d_global`;
/// the Robin coefficient γ = j·β and the incident source −2j·β·E_mode follow.
/// A passive boundary supplies only `get_gamma`.
pub trait Port {
    /// S-matrix index (1-based) of a driven port; 0 for a passive boundary.
    fn port_number(&self) -> usize { 0 }

    /// Whether this port is driven (has an excitation vector).
    fn is_driven(&self) -> bool { self.port_number() > 0 }

    /// Propagation constant β of the port mode at the live k0.
    fn beta(&self, _exc: &Excitation) -> f64 { 0.0 }

    /// Robin BC coefficient γ.
    fn get_gamma(&self, exc: &Excitation) -> C64 { C64::new(0.0, self.beta(exc)) }

    /// An anisotropic Robin coefficient (3×3, global frame) for the port's
    /// `k`-th triangle, where it differs from γ.
    fn tri_tensor(&self, _exc: &Excitation, _k: usize) -> Option<[[C64; 3]; 3]> { None }

    /// Whether the Robin term is γ(k₀) times one fixed surface matrix at
    /// every frequency (no [`Port::tri_tensor`]): the adaptive sweep then
    /// projects that matrix once instead of per frequency.
    fn robin_is_scalar(&self) -> bool { true }

    /// Power-normalised mode field at a global point (S-parameter
    /// extraction and the excitation); `None` for a passive boundary.
    fn port_mode_3d_global(&self, _x: f64, _y: f64, _z: f64, _exc: &Excitation) -> Option<(f64, f64, f64)> {
        None
    }

    /// Incident-wave source term U_inc = −2j·β·E_mode at a global point.
    fn get_uinc(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> Option<[C64; 3]> {
        let (ex, ey, ez) = self.port_mode_3d_global(x, y, z, exc)?;
        let f = C64::new(0.0, -2.0 * self.beta(exc));
        Some([f * ex, f * ey, f * ez])
    }

    /// Mode impedance `Z_mode` (port impedances, renormalisation).
    fn z_mode(&self, _exc: &Excitation) -> f64 { 0.0 }

    /// A lumped port's voltage extraction: (field direction, gap height,
    /// incident voltage). `None` for a modal port (mode projection).
    fn lumped(&self) -> Option<([f64; 3], f64, f64)> { None }
}
