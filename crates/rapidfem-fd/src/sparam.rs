// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! S-parameter extraction via port surface integrals.
//!
//! Standard modal post-processing (Pozar, *Microwave Engineering*, ch. 4;
//! Jin, *FEM in Electromagnetics*): a port wave amplitude is the overlap of
//! the simulated field with the port mode, weighted by the local wave
//! admittance / Poynting factor. `sparam_waveport` returns the amplitude
//! ratio S; the `*_power` helpers return the Poynting-normalised powers; and
//! `sparam_voltage_surface` extracts a lumped port's S from the area-averaged
//! mode-projected voltage `V = (1/w)∫E·l̂ dS` (derivations/lumped_port/).
//! All accept `&dyn Port` and integrate over the port triangles by Gaussian
//! quadrature (`surface_integral`).

use num_complex::Complex64 as C64;
use crate::quadrature::gaus_quad_tri;
use crate::port::Port;
use crate::excitation::Excitation;

/// Gauss-quadrature surface integral of a scalar function over a triangle set.
pub fn surface_integral(
    nodes: &[[f64; 3]],
    triangles: &[[usize; 3]],
    function: &dyn Fn(f64, f64, f64) -> C64,
    gq_order: usize,
) -> C64 {
    let dpts = gaus_quad_tri(gq_order);
    let mut total = C64::new(0.0, 0.0);

    for tri in triangles {
        let v1 = nodes[tri[0]];
        let v2 = nodes[tri[1]];
        let v3 = nodes[tri[2]];

        let e1 = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]];
        let e2 = [v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]];
        let cr = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
        let area = 0.5 * (cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]).sqrt();

        let mut tri_sum = C64::new(0.0, 0.0);
        for qp in &dpts {
            let (w, l1, l2, l3) = (qp[0], qp[1], qp[2], qp[3]);
            let x = v1[0]*l1 + v2[0]*l2 + v3[0]*l3;
            let y = v1[1]*l1 + v2[1]*l2 + v3[1]*l3;
            let z = v1[2]*l1 + v2[2]*l2 + v3[2]*l3;
            tri_sum += C64::from(w) * function(x, y, z);
        }
        total += tri_sum * C64::from(area);
    }

    total
}

/// Modal-port S-parameter: amplitude ratio of the reflected/transmitted
/// power-wave to the incident, by mode overlap.
///
/// S = ∫ (E_field − Q·E_mode)·conj(E_mode)·c dS / ∫ |E_mode|²·c dS
///
/// where `c(x,y,z)` is the **local wave admittance weight**, `√(εᵣ/μᵣ)` for a
/// TEM/quasi-TEM mode (`1/μᵣ` for TE, `1/εᵣ` for TM), supplied by `weight`.
/// The weight turns the bare field overlap (`|E|²`) into the power / Poynting
/// overlap: the power-wave amplitude is `b ∝ ∫E×H*·n̂`, and for a TEM mode
/// `H_mode ∝ c·(n̂×E_mode)`, so `b ∝ ∫ c·E·conj(E_mode)`. Without it the
/// overlap mis-weights an *inhomogeneous* quasi-TEM mode (the cross-section
/// impedance varies), giving a passivity error `|S|² > 1` ∝ the inhomogeneity.
/// The ratio stays amplitude-invariant, so the mode normalisation cancels.
pub fn sparam_waveport(
    nodes: &[[f64; 3]],
    tri_verts: &[[usize; 3]],
    port: &dyn Port,
    exc: &Excitation,
    active: bool,
    fieldf: &dyn Fn(f64, f64, f64) -> (C64, C64, C64),
    weight: &dyn Fn(f64, f64, f64) -> f64,
    gq_order: usize,
) -> C64 {
    let q = if active { 1.0 } else { 0.0 };

    let mode_dot_field = surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, exc).unwrap_or((0.0, 0.0, 0.0));
        let (fx, fy, fz) = fieldf(x, y, z);
        let c = C64::from(weight(x, y, z));

        let ex1 = fx - C64::from(q * mx);
        let ey1 = fy - C64::from(q * my);
        let ez1 = fz - C64::from(q * mz);

        c * (ex1 * C64::from(mx) + ey1 * C64::from(my) + ez1 * C64::from(mz))
    }, gq_order);

    let norm = surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, exc).unwrap_or((0.0, 0.0, 0.0));
        C64::from(weight(x, y, z) * (mx*mx + my*my + mz*mz))
    }, gq_order);

    if norm.norm() < crate::constants::SINGULAR_EPS {
        return C64::new(0.0, 0.0);
    }

    mode_dot_field / norm
}

/// Field power crossing the port: P = ∫ (E−Q·E_mode)·conj(E_mode)/(2·Z_mode) dS.
///
/// P_field = ∫ (E_field - Q·E_mode) · conj(E_mode) / (2·Z_mode) dS
pub fn sparam_field_power(
    nodes: &[[f64; 3]],
    tri_verts: &[[usize; 3]],
    port: &dyn Port,
    exc: &Excitation,
    active: bool,
    fieldf: &dyn Fn(f64, f64, f64) -> (C64, C64, C64),
    gq_order: usize,
) -> C64 {
    let q = if active { 1.0 } else { 0.0 };
    let z_mode = port.z_mode(exc);

    surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, exc).unwrap_or((0.0, 0.0, 0.0));
        let (fx, fy, fz) = fieldf(x, y, z);

        let ex1 = fx - C64::from(q * mx);
        let ey1 = fy - C64::from(q * my);
        let ez1 = fz - C64::from(q * mz);

        let ex2 = C64::from(mx).conj();
        let ey2 = C64::from(my).conj();
        let ez2 = C64::from(mz).conj();

        (ex1*ex2 + ey1*ey2 + ez1*ez2) / C64::from(2.0 * z_mode)
    }, gq_order)
}

/// Reference mode power: P = ∫ |E_mode|²/(2·Z_mode) dS.
///
/// P_mode = ∫ |E_mode|² / (2·Z_mode) dS
pub fn sparam_mode_power(
    nodes: &[[f64; 3]],
    tri_verts: &[[usize; 3]],
    port: &dyn Port,
    exc: &Excitation,
    gq_order: usize,
) -> C64 {
    let z_mode = port.z_mode(exc);

    surface_integral(nodes, tri_verts, &|x, y, z| {
        let (mx, my, mz) = port.port_mode_3d_global(x, y, z, exc).unwrap_or((0.0, 0.0, 0.0));
        let ex2 = C64::from(mx).conj();
        let ey2 = C64::from(my).conj();
        let ez2 = C64::from(mz).conj();

        (C64::from(mx)*ex2 + C64::from(my)*ey2 + C64::from(mz)*ez2) / C64::from(2.0 * z_mode)
    }, gq_order)
}

/// Voltage-based S-parameter extraction for lumped ports.
///
/// Integrates E·dl along a line through the port to get the port voltage,
/// then computes S-parameter from incident/reflected wave decomposition.
///
/// v_inc = sqrt(2 * P * Z0), the incident voltage
/// V_total = ∫ E · dl across the port gap
///
/// For active port (self-excitation):  S = (V_total - V_inc) / V_inc
/// For passive port (observation only): S = V_total / V_inc
///
/// The reference impedance Z₀ cancels in the wave-amplitude ratio (it enters
/// `a` and `b` identically), so it is not a parameter here.
/// Area-averaged mode-projected voltage S-parameter for lumped ports.
///
/// Clean-room derivation in `derivations/lumped_port/`: the port voltage is the
/// transverse-averaged path integral of the SOLVED field over the whole port
/// surface,
///
///   V = (1/w) ∫_Γ E·l̂ dS,   w = area/l   ⇒   V = (l/A) ∫_Γ E·l̂ dS
///
/// (`l̂` = port direction, `l` = height along it, `A` = port area). For a
/// uniform mode `a·l̂` this is exactly the gap voltage `a·l`, but unlike a few
/// discrete `∫E·dl` lines it stays well-defined when the field is non-uniform
/// over a TALL port (e.g. a 184 µm RFIC feed), where the line integrals
/// disagree and the extraction degenerates.
///
/// `S_ii = (V - V_inc)/V_inc` (active), `S_ij = V/V_inc` (passive).
pub fn sparam_voltage_surface(
    nodes: &[[f64; 3]],
    tri_verts: &[[usize; 3]],
    direction: [f64; 3],
    height: f64,
    v_inc: f64,
    active: bool,
    fieldf: &dyn Fn(f64, f64, f64) -> (C64, C64, C64),
    gq_order: usize,
) -> C64 {
    // Port area A = Σ triangle areas.
    let mut area = 0.0_f64;
    for tri in tri_verts {
        let v1 = nodes[tri[0]];
        let v2 = nodes[tri[1]];
        let v3 = nodes[tri[2]];
        let e1 = [v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2]];
        let e2 = [v3[0]-v1[0], v3[1]-v1[1], v3[2]-v1[2]];
        let cr = [e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]];
        area += 0.5 * (cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]).sqrt();
    }
    if area < crate::constants::SINGULAR_EPS || v_inc == 0.0 {
        return C64::new(0.0, 0.0);
    }

    // ∫_Γ E·l̂ dS over the port surface.
    let flux = surface_integral(nodes, tri_verts, &|x, y, z| {
        let (fx, fy, fz) = fieldf(x, y, z);
        fx * C64::from(direction[0]) + fy * C64::from(direction[1]) + fz * C64::from(direction[2])
    }, gq_order);

    // V = (l/A) ∫ E·l̂ dS  (the w=A/l normalisation, mode-projected gap voltage).
    let v_total = flux * C64::from(height / area);
    let v_inc_c = C64::from(v_inc);
    if active {
        (v_total - v_inc_c) / v_inc_c
    } else {
        v_total / v_inc_c
    }
}

