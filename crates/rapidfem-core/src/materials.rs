// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Per-tet material tensors for the frequency-domain assembly.
//!
//! The effective complex relative permittivity combines a loss tangent and a
//! bulk conductivity into the standard lossy-dielectric form (e.g. Pozar,
//! *Microwave Engineering*, §1.3; Jackson, *Classical Electrodynamics*):
//!
//!   εr*(ω) = εr·(1 − j·tanδ) − j·σ/(ω·ε₀)
//!
//! with optional diagonal anisotropy, Debye/Drude dispersion and uniaxial
//! PML coordinate stretching layered on top.

use num_complex::Complex64 as C64;
use crate::constants::EPS0;

/// Frequency dispersion model for εr.
///
/// Debye: ε(ω) = ε∞ + (εs - ε∞) / (1 + jωτ)
/// Drude: ε(ω) = ε∞ - ωp² / (ω² + jγω)   where ωp = 2π·plasma_freq, γ = 2π·damping_freq
pub enum Dispersion {
    None,
    Debye { er_inf: f64, er_static: f64, tau_s: f64 },
    Drude { er_inf: f64, plasma_freq_hz: f64, damping_freq_hz: f64 },
}

impl Dispersion {
    /// Returns the effective complex εr at the given frequency. The base `er` value (passed
    /// in) is used when this is `None`; otherwise the dispersion model overrides it.
    pub fn evaluate(&self, base_er: f64, frequency: f64) -> C64 {
        let omega = 2.0 * std::f64::consts::PI * frequency;
        match self {
            Dispersion::None => C64::from(base_er),
            Dispersion::Debye { er_inf, er_static, tau_s } => {
                C64::from(*er_inf) + (C64::from(*er_static - *er_inf)) / (C64::new(1.0, 0.0) + C64::new(0.0, omega * *tau_s))
            }
            Dispersion::Drude { er_inf, plasma_freq_hz, damping_freq_hz } => {
                let wp = 2.0 * std::f64::consts::PI * plasma_freq_hz;
                let gamma = 2.0 * std::f64::consts::PI * damping_freq_hz;
                // ε(ω) = ε∞ - ωp² / (ω² + jγω) = ε∞ - ωp² / (ω(ω + jγ))
                let denom = C64::from(omega) * C64::new(omega, gamma);
                C64::from(*er_inf) - C64::from(wp * wp) / denom
            }
        }
    }

    pub fn is_dispersive(&self) -> bool {
        !matches!(self, Dispersion::None)
    }
}

/// Material definition for a region of the mesh. Supports diagonal anisotropy:
/// `er_diag` and `ur_diag` override the scalar values when present.
pub struct Material {
    /// Relative permittivity (scalar, isotropic baseline)
    pub er: f64,
    /// Relative permeability (scalar, isotropic baseline)
    pub ur: f64,
    /// Loss tangent tan(δ)
    pub tand: f64,
    /// Conductivity σ (S/m)
    pub cond: f64,
    /// Which tets this material applies to (indices into mesh.tets)
    pub tet_indices: Vec<usize>,
    /// Optional diagonal εr tensor [εxx, εyy, εzz]; if Some, overrides the scalar `er`.
    pub er_diag: Option<[f64; 3]>,
    /// Optional diagonal μr tensor [μxx, μyy, μzz]; if Some, overrides the scalar `ur`.
    pub ur_diag: Option<[f64; 3]>,
    /// Optional frequency dispersion model. When set, εr is evaluated at each frequency.
    pub dispersion: Dispersion,
}

/// Build per-tet εr and μr tensors from material definitions.
/// Accumulates each region's diagonal εr/μr/tanδ/σ onto its tets, then applies
/// the complex-permittivity relation εr* = εr(1 − j·tanδ) − j·σ/(ω·ε₀).
///
/// Returns (er_tensors, ur_tensors) where each is Vec of 3x3 complex tensors.
pub fn build_material_tensors(
    n_tets: usize,
    materials: &[Material],
    frequency: f64,
) -> (Vec<[[C64; 3]; 3]>, Vec<[[C64; 3]; 3]>) {
    let w0 = 2.0 * std::f64::consts::PI * frequency;

    let zero3x3 = [[C64::new(0.0, 0.0); 3]; 3];
    let mut er = vec![zero3x3; n_tets];
    let mut ur = vec![zero3x3; n_tets];
    let mut tand = vec![zero3x3; n_tets];
    let mut cond = vec![zero3x3; n_tets];

    // Accumulate each region's diagonal properties onto its tets.
    // For dispersive materials, εr is replaced by Dispersion::evaluate(εr_base, frequency).
    for mat in materials {
        let er_diag = mat.er_diag.unwrap_or([mat.er; 3]);
        let ur_diag = mat.ur_diag.unwrap_or([mat.ur; 3]);
        let er_eff: [C64; 3] = if mat.dispersion.is_dispersive() {
            // Dispersion currently isotropic: same complex εr on all diagonal elements.
            let e = mat.dispersion.evaluate(mat.er, frequency);
            [e, e, e]
        } else {
            [C64::from(er_diag[0]), C64::from(er_diag[1]), C64::from(er_diag[2])]
        };
        for &ti in &mat.tet_indices {
            for k in 0..3 {
                er[ti][k][k] += er_eff[k];
                ur[ti][k][k] += C64::from(ur_diag[k]);
                tand[ti][k][k] += C64::from(mat.tand);
                cond[ti][k][k] += C64::from(mat.cond);
            }
        }
    }

    // Complex permittivity: εr* = εr·(1 − j·tanδ) − j·σ/(ω·ε₀)
    for ti in 0..n_tets {
        for i in 0..3 {
            for j in 0..3 {
                let er_val = er[ti][i][j];
                let tand_val = tand[ti][i][j];
                let cond_val = cond[ti][i][j];
                er[ti][i][j] = er_val * (C64::new(1.0, 0.0) - C64::new(0.0, 1.0) * tand_val)
                    - C64::new(0.0, 1.0) * cond_val / C64::from(w0 * EPS0);
            }
        }
    }

    (er, ur)
}

/// PML (Perfectly Matched Layer) region, port of geo/pmlbox.py.
///
/// Absorbing layer using anisotropic complex coordinate-stretched material tensors:
///   sₐ(coord) = 1 - j · u(coord)^n · δmax,  u = (coord - inner_face)·sign / thickness
///   ε_aa = εr · (s_b · s_c) / s_a   (and cyclic for the orthogonal pair)
///
/// Stretch is applied along `direction` (one of ±x̂, ±ŷ, ±ẑ). `inner_face` is the coordinate
/// of the PML's inner boundary along that direction; `thickness` extends outward from there.
/// Stretch factors in the orthogonal directions remain 1 (uniaxial PML).
pub struct PmlRegion {
    /// Tets that belong to this PML region
    pub tet_indices: Vec<usize>,
    /// Base relative permittivity (real)
    pub er_base: f64,
    /// Base relative permeability (real)
    pub ur_base: f64,
    /// Direction the layer absorbs: e.g. (1,0,0) for +x face PML, (0,0,-1) for -z
    pub direction: [f64; 3],
    /// Coordinate of the inner face along the absorption direction (m)
    pub inner_face: f64,
    /// PML layer thickness (m)
    pub thickness: f64,
    /// Stretch profile exponent (typical 1.5 - 3.0)
    pub exponent: f64,
    /// Maximum stretch magnitude δmax (typical 5 - 12)
    pub delta_max: f64,
}

impl PmlRegion {
    /// Stretch factors (s_x, s_y, s_z) at a point (x,y,z).
    pub fn stretch(&self, x: f64, y: f64, z: f64) -> [C64; 3] {
        let d = self.direction;
        let (axis, sign) = if d[0].abs() > d[1].abs() && d[0].abs() > d[2].abs() {
            (0, d[0].signum())
        } else if d[1].abs() > d[2].abs() {
            (1, d[1].signum())
        } else {
            (2, d[2].signum())
        };
        let coord = [x, y, z][axis];
        let u_raw = sign * (coord - self.inner_face) / self.thickness;
        let u = u_raw.max(0.0);  // clamp inside the PML
        let s_d = C64::new(1.0, 0.0) - C64::new(0.0, 1.0) * C64::from(u.powf(self.exponent) * self.delta_max);
        let mut s = [C64::new(1.0, 0.0); 3];
        s[axis] = s_d;
        s
    }

    /// Anisotropic εr / μr tensors at a tet centroid.
    pub fn material_tensors_at(&self, x: f64, y: f64, z: f64) -> ([[C64; 3]; 3], [[C64; 3]; 3]) {
        let s = self.stretch(x, y, z);
        // Diagonal: ε_aa = εr · (s_b · s_c) / s_a
        let mut er_t = [[C64::new(0.0, 0.0); 3]; 3];
        let mut ur_t = [[C64::new(0.0, 0.0); 3]; 3];
        for a in 0..3 {
            let b = (a + 1) % 3;
            let c = (a + 2) % 3;
            let factor = (s[b] * s[c]) / s[a];
            er_t[a][a] = C64::from(self.er_base) * factor;
            ur_t[a][a] = C64::from(self.ur_base) * factor;
        }
        (er_t, ur_t)
    }
}

/// Build per-tet εr and μr tensors with PML overrides.
///
/// First runs `build_material_tensors` on regular materials, then for every tet in any PML
/// region OVERWRITES the tensor with the coordinate-stretched anisotropic value evaluated at
/// the tet centroid. (PML regions take precedence over isotropic materials.)
pub fn build_material_tensors_with_pml(
    n_tets: usize,
    materials: &[Material],
    pml_regions: &[PmlRegion],
    mesh: &crate::mesh::Mesh,
    frequency: f64,
) -> (Vec<[[C64; 3]; 3]>, Vec<[[C64; 3]; 3]>) {
    let (mut er, mut ur) = build_material_tensors(n_tets, materials, frequency);
    for region in pml_regions {
        for &ti in &region.tet_indices {
            let tet = &mesh.tets[ti];
            let cx = (mesh.nodes[tet[0]][0] + mesh.nodes[tet[1]][0]
                + mesh.nodes[tet[2]][0] + mesh.nodes[tet[3]][0]) * 0.25;
            let cy = (mesh.nodes[tet[0]][1] + mesh.nodes[tet[1]][1]
                + mesh.nodes[tet[2]][1] + mesh.nodes[tet[3]][1]) * 0.25;
            let cz = (mesh.nodes[tet[0]][2] + mesh.nodes[tet[1]][2]
                + mesh.nodes[tet[2]][2] + mesh.nodes[tet[3]][2]) * 0.25;
            let (er_t, ur_t) = region.material_tensors_at(cx, cy, cz);
            er[ti] = er_t;
            ur[ti] = ur_t;
        }
    }
    (er, ur)
}
