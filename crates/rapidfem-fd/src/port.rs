// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Port trait: unified interface for all boundary condition types.
//! Assembly uses this trait to handle RectWaveguide, LumpedPort, and ABC uniformly.

use num_complex::Complex64 as C64;
use crate::excitation::Excitation;

/// Unified port interface for the Robin boundary-condition ports.
pub trait Port {
    /// Robin BC impedance parameter γ
    fn get_gamma(&self, exc: &Excitation) -> C64;

    /// Incident field U_inc at a global point (for excitation vector).
    /// Returns None for non-driven ports (e.g. ABC).
    fn get_uinc(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> Option<[C64; 3]>;

    /// Whether this port is driven (has an excitation vector)
    fn is_driven(&self) -> bool;

    /// Port mode field at a global point (for S-param extraction).
    /// Returns None for ABC (no S-param extraction).
    fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> Option<(f64, f64, f64)>;

    /// Mode impedance Z_mode (for sparam_field_power/mode_power)
    fn z_mode(&self, exc: &Excitation) -> f64;

    /// Port number (for S-matrix indexing)
    fn port_number(&self) -> usize;

    /// Whether this port is a lumped port (uses voltage extraction)
    fn is_lumped(&self) -> bool { false }

    /// For lumped ports: return (direction, z0, voltage) for voltage integration
    fn lumped_voltage_params(&self) -> Option<([f64; 3], f64, f64)> { None }

    /// For lumped ports: extent along the field direction (height parameter)
    fn port_height(&self) -> Option<f64> { None }
}

/// Passive boundary: no excitation and no S-param extraction (ABC-style).
/// Every such type forwards `get_gamma` to its inherent method and stubs the
/// rest, so they share one impl body.
macro_rules! impl_passive_port {
    ($ty:ty) => {
        impl Port for $ty {
            fn get_gamma(&self, exc: &Excitation) -> C64 { self.get_gamma(exc) }
            fn get_uinc(&self, _x: f64, _y: f64, _z: f64, _exc: &Excitation) -> Option<[C64; 3]> { None }
            fn is_driven(&self) -> bool { false }
            fn port_mode_3d_global(&self, _x: f64, _y: f64, _z: f64, _exc: &Excitation)
                -> Option<(f64, f64, f64)> { None }
            fn z_mode(&self, _exc: &Excitation) -> f64 { 0.0 }
            fn port_number(&self) -> usize { 0 }
        }
    };
}

/// Driven port: forwards excitation and mode field to inherent methods and
/// indexes by `self.port_number`. Only the mode impedance differs per type,
/// so it is supplied as a `|port, k0|` closure.
macro_rules! impl_driven_port {
    ($ty:ty, $z_mode:expr) => {
        impl Port for $ty {
            fn get_gamma(&self, exc: &Excitation) -> C64 { self.get_gamma(exc) }
            fn get_uinc(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> Option<[C64; 3]> {
                Some(self.get_uinc(x, y, z, exc))
            }
            fn is_driven(&self) -> bool { true }
            fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, exc: &Excitation)
                -> Option<(f64, f64, f64)> {
                Some(self.port_mode_3d_global(x, y, z, exc))
            }
            fn z_mode(&self, exc: &Excitation) -> f64 { ($z_mode)(self, exc) }
            fn port_number(&self) -> usize { self.port_number }
        }
    };
}

impl_passive_port!(crate::waveguide::LumpedElement);
impl_passive_port!(crate::waveguide::SurfaceImpedance);
impl_passive_port!(crate::waveguide::AbsorbingBoundary);

impl_driven_port!(crate::waveguide::RectWaveguide, |p: &crate::waveguide::RectWaveguide, exc| p.z_mode(exc));
impl_driven_port!(crate::waveguide::FloquetPort, |_p: &crate::waveguide::FloquetPort, _exc: &Excitation| crate::constants::Z0);
// UserDefinedPort: dummy mode impedance, the user scales via incident power.
impl_driven_port!(crate::waveguide::UserDefinedPort, |_p: &crate::waveguide::UserDefinedPort, _exc: &Excitation| 1.0);
impl_driven_port!(crate::waveguide::CoaxPort, |p: &crate::waveguide::CoaxPort, _exc: &Excitation| p.port_z());
impl_driven_port!(crate::waveguide::NumericalWavePort, |p: &crate::waveguide::NumericalWavePort, exc| p.z_mode(exc));

// LumpedPort is driven AND lumped (voltage extraction), so it adds the lumped
// trait methods on top of the driven pattern and stays hand-written.
impl Port for crate::waveguide::LumpedPort {
    fn get_gamma(&self, exc: &Excitation) -> C64 { self.get_gamma(exc) }
    fn get_uinc(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> Option<[C64; 3]> {
        Some(self.get_uinc(x, y, z, exc))
    }
    fn is_driven(&self) -> bool { true }
    fn port_mode_3d_global(&self, x: f64, y: f64, z: f64, exc: &Excitation) -> Option<(f64, f64, f64)> {
        Some(self.port_mode_3d_global(x, y, z, exc))
    }
    fn z_mode(&self, _exc: &Excitation) -> f64 { self.z0 }
    fn port_number(&self) -> usize { self.port_number }
    fn is_lumped(&self) -> bool { true }
    fn lumped_voltage_params(&self) -> Option<([f64; 3], f64, f64)> {
        Some((self.direction, self.z0, self.voltage()))
    }
    fn port_height(&self) -> Option<f64> { Some(self.height) }
}
