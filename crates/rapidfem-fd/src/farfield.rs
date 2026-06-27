// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Near-field to far-field transformation (NFFT) for radiation patterns.
//!
//! Uses the equivalence principle on a closed surface (typically the ABC boundary):
//!   J_s = n̂ × H     (equivalent electric current)
//!   M_s = -n̂ × E    (equivalent magnetic current)
//!
//! Far-field radiation integrals:
//!   N(θ,φ) = ∫∫ J_s · e^{jk r̂·r'} dS'
//!   L(θ,φ) = ∫∫ M_s · e^{jk r̂·r'} dS'
//!
//! E_θ^far = -jk/(4π) (L_φ + η₀ N_θ)
//! E_φ^far = -jk/(4π) (-L_θ + η₀ N_φ)

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::interp;
use crate::error_estimator::eval_curl_in_tet;
use crate::quadrature::gaus_quad_tri;
use crate::constants::*;

/// Far-field radiation pattern result.
///
/// All angle-indexed arrays are `[phi_idx][theta_idx]`.
pub struct RadiationPattern {
    /// Theta angles in radians (0 to π)
    pub theta: Vec<f64>,
    /// Phi angles in radians (0 to 2π)
    pub phi: Vec<f64>,
    /// Complex E_θ component (V/m at unit reference distance)
    pub e_theta: Vec<Vec<C64>>,
    /// Complex E_φ component
    pub e_phi: Vec<Vec<C64>>,
    /// Directivity in dBi
    pub directivity_dbi: Vec<Vec<f64>>,
    /// Gain in dBi, accounts for input mismatch (= directivity × (1-|S11|²) when input_power is provided)
    pub gain_dbi: Vec<Vec<f64>>,
    /// Axial ratio in dB. AR=0dB → circular polarization, AR=∞ → linear.
    pub axial_ratio_db: Vec<Vec<f64>>,
    /// Left-hand circular polarization (LCP) component, in dBi (relative to isotropic)
    pub lcp_dbi: Vec<Vec<f64>>,
    /// Right-hand circular polarization (RCP) component, in dBi
    pub rcp_dbi: Vec<Vec<f64>>,
    /// Peak directivity in dBi
    pub peak_directivity_dbi: f64,
    /// Peak gain in dBi
    pub peak_gain_dbi: f64,
    /// Total radiated power (W)
    pub radiated_power: f64,
}

/// Compute the far-field radiation pattern from the FEM solution.
///
/// `nfft_tri_ids`: triangle indices of the closed surface (ABC boundary)
/// `solution`: FEM solution vector (complex DOF coefficients)
/// `frequency`: operating frequency (Hz)
/// `n_theta`: number of theta angles (0 to π)
/// `n_phi`: number of phi angles (0 to 2π)
/// `gq_order`: Gauss quadrature order for surface integration
pub fn compute_farfield(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    nfft_tri_ids: &[usize],
    frequency: f64,
    n_theta: usize,
    n_phi: usize,
    gq_order: usize,
) -> RadiationPattern {
    compute_farfield_with_input(mesh, basis, solution, nfft_tri_ids, frequency, n_theta, n_phi, gq_order, None)
}

/// Same as `compute_farfield`, but lets the caller pass a radiation efficiency η = P_rad/P_in
/// for accurate gain calculation. The caller computes η from the S-parameters of the port
/// (typically η = 1 - Σ|S_i1|² for a single-driven port). When `radiation_efficiency = None`,
/// gain == directivity (lossless, matched assumption).
///
/// Note: we do NOT use the FEM-integrated radiated power for the gain offset, because the
/// FEM E-field scale is an internal-to-the-solver convention (see LumpedPort `get_uinc`).
/// Directivity is unit-invariant; gain requires an externally known efficiency.
pub fn compute_farfield_with_input(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    nfft_tri_ids: &[usize],
    frequency: f64,
    n_theta: usize,
    n_phi: usize,
    gq_order: usize,
    radiation_efficiency: Option<f64>,
) -> RadiationPattern {
    compute_farfield_full(mesh, basis, solution, nfft_tri_ids, &[], frequency,
                          n_theta, n_phi, gq_order, radiation_efficiency)
}

/// Full-fidelity far-field with separate ABC (radiation) and PEC (image-closing) surfaces.
///
/// On the ABC surface, both J_s = n̂×H and M_s = -n̂×E contribute (radiating equivalence).
/// On PEC surfaces, tangential E = 0 by construction so M_s = 0, only J_s = n̂×H from the
/// air-side is included. Adding the PEC surface CLOSES the integration boundary around the
/// antenna, eliminating the phantom back lobe that an open-surface NFFT produces over a
/// half-space antenna with a finite ground plane.
///
/// `nfft_tri_ids`: open radiating surface (typically ABC tri tags).
/// `pec_tri_ids`: PEC closing surfaces (ground, side walls); empty = no closure.
pub fn compute_farfield_full(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    nfft_tri_ids: &[usize],
    pec_tri_ids: &[usize],
    frequency: f64,
    n_theta: usize,
    n_phi: usize,
    gq_order: usize,
    radiation_efficiency: Option<f64>,
) -> RadiationPattern {
    let exc = crate::excitation::Excitation::new(frequency, mesh.l0);
    // Lever ④: the near-to-far transform is done entirely in physical units —
    // surface points/fields are converted to physical below — so use the
    // physical k₀ here (= κ/L₀), not the length-normalized κ.
    let l0 = mesh.l0;
    let k0 = exc.k0 / l0;
    let omega = exc.omega;
    let j = C64::new(0.0, 1.0);

    // Build theta/phi grids
    let thetas: Vec<f64> = (0..n_theta).map(|i| PI * i as f64 / (n_theta - 1) as f64).collect();
    let phis: Vec<f64> = (0..n_phi).map(|i| 2.0 * PI * i as f64 / n_phi as f64).collect();

    // Build spatial hash for point-in-tet queries
    let grid = interp::TetGrid::new(mesh);

    // Precompute quadrature points and fields on the NFFT surface
    let quad_pts = gaus_quad_tri(gq_order);

    // For each observation direction (theta, phi), compute N and L integrals
    let mut directivity = vec![vec![0.0f64; n_theta]; n_phi];
    let mut e_theta = vec![vec![C64::new(0.0, 0.0); n_theta]; n_phi];
    let mut e_phi = vec![vec![C64::new(0.0, 0.0); n_theta]; n_phi];

    // Precompute surface data: for each triangle, store quadrature point data
    // (position, E-field, H-field, normal, area*weight, is_pec)
    struct SurfPoint {
        pos: [f64; 3],
        e: [C64; 3],
        h: [C64; 3],
        normal: [f64; 3],
        aw: f64,    // area * quadrature weight
        is_pec: bool, // true → M_s = 0 (tangential E = 0 on PEC by construction)
    }

    let mut surf_data: Vec<SurfPoint> = Vec::new();
    let all_tris: Vec<(usize, bool)> = nfft_tri_ids.iter().map(|&t| (t, false))
        .chain(pec_tri_ids.iter().map(|&t| (t, true)))
        .collect();

    for &(tri_idx, is_pec) in &all_tris {
        let tri = mesh.tris[tri_idx];
        let v0 = mesh.nodes[tri[0]];
        let v1 = mesh.nodes[tri[1]];
        let v2 = mesh.nodes[tri[2]];

        // Triangle edges and normal
        let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
        let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
        let cr = [
            e1[1]*e2[2] - e1[2]*e2[1],
            e1[2]*e2[0] - e1[0]*e2[2],
            e1[0]*e2[1] - e1[1]*e2[0],
        ];
        let area = 0.5 * (cr[0]*cr[0] + cr[1]*cr[1] + cr[2]*cr[2]).sqrt();
        let nn = 2.0 * area;
        let mut normal = [cr[0]/nn, cr[1]/nn, cr[2]/nn];

        // Orient normal outward using adjacent tet
        let adj_tet = mesh.tri_to_tet[tri_idx][0];
        if adj_tet != usize::MAX {
            let tet = &mesh.tets[adj_tet];
            let tc = [
                (mesh.nodes[tet[0]][0]+mesh.nodes[tet[1]][0]+mesh.nodes[tet[2]][0]+mesh.nodes[tet[3]][0])/4.0,
                (mesh.nodes[tet[0]][1]+mesh.nodes[tet[1]][1]+mesh.nodes[tet[2]][1]+mesh.nodes[tet[3]][1])/4.0,
                (mesh.nodes[tet[0]][2]+mesh.nodes[tet[1]][2]+mesh.nodes[tet[2]][2]+mesh.nodes[tet[3]][2])/4.0,
            ];
            let tri_center = [
                (v0[0]+v1[0]+v2[0])/3.0,
                (v0[1]+v1[1]+v2[1])/3.0,
                (v0[2]+v1[2]+v2[2])/3.0,
            ];
            let to_tet = [tc[0]-tri_center[0], tc[1]-tri_center[1], tc[2]-tri_center[2]];
            if normal[0]*to_tet[0] + normal[1]*to_tet[1] + normal[2]*to_tet[2] > 0.0 {
                normal = [-normal[0], -normal[1], -normal[2]];
            }
        }

        // Evaluate E and H at quadrature points
        for qp in &quad_pts {
            let (w, l1, l2, l3) = (qp[0], qp[1], qp[2], qp[3]);
            let x = v0[0]*l1 + v1[0]*l2 + v2[0]*l3;
            let y = v0[1]*l1 + v1[1]*l2 + v2[1]*l3;
            let z = v0[2]*l1 + v1[2]*l2 + v2[2]*l3;

            // Find containing tet for this point
            let tet_idx = grid.find_containing_tet(mesh, x, y, z)
                .or_else(|| {
                    // Fallback: use the adjacent tet
                    let t = mesh.tri_to_tet[tri_idx][0];
                    if t != usize::MAX { Some(t) } else { None }
                });

            if let Some(tet) = tet_idx {
                // Reconstruct on the (L₀-normalized) mesh, then convert to
                // physical units: E_recon = L₀·E_phys, curl_recon = L₀²·curl_phys.
                let il = 1.0 / l0;
                let (ex, ey, ez) = interp::eval_field_in_tet(mesh, basis, solution, tet, x, y, z);

                // curl(E) = jωμ₀ H  =>  H = curl(E) / (jωμ₀); the extra L₀²
                // (curl normalization) is folded into the denominator.
                let curl_e = eval_curl_in_tet(mesh, basis, solution, tet, x, y, z);
                let denom = j * C64::from(omega * MU0 * l0 * l0);
                let hx = curl_e[0] / denom;
                let hy = curl_e[1] / denom;
                let hz = curl_e[2] / denom;

                surf_data.push(SurfPoint {
                    pos: [x * l0, y * l0, z * l0],          // physical position
                    e: [ex * il, ey * il, ez * il],         // physical E (V/m)
                    h: [hx, hy, hz],                         // physical H (A/m)
                    normal,                                  // unit vector, scale-free
                    aw: area * w * l0 * l0,                  // physical area element
                    is_pec,
                });
            }
        }
    }

    eprintln!("  Far-field: {} surface integration points ({} ABC tris + {} PEC tris)",
        surf_data.len(), nfft_tri_ids.len(), pec_tri_ids.len());

    // Compute far-field for each (theta, phi) direction.
    // The outer loop is embarrassingly parallel: each direction integrates the same surf_data
    // independently. With the `parallel` feature, rayon distributes ip across cores.
    let dtheta = PI / (n_theta - 1) as f64;
    let dphi = 2.0 * PI / n_phi as f64;

    let direction_results: Vec<(usize, usize, C64, C64, f64)> = {
        let pairs: Vec<(usize, usize)> = (0..n_phi)
            .flat_map(|ip| (0..n_theta).map(move |it| (ip, it)))
            .collect();

        let compute_one = |&(ip, it): &(usize, usize)| -> (usize, usize, C64, C64, f64) {
            let theta = thetas[it];
            let phi = phis[ip];
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            let sin_p = phi.sin();
            let cos_p = phi.cos();

            let r_hat = [sin_t * cos_p, sin_t * sin_p, cos_t];
            let theta_hat = [cos_t * cos_p, cos_t * sin_p, -sin_t];
            let phi_hat = [-sin_p, cos_p, 0.0];

            let mut nt = C64::new(0.0, 0.0);
            let mut np = C64::new(0.0, 0.0);
            let mut lt = C64::new(0.0, 0.0);
            let mut lp = C64::new(0.0, 0.0);

            for sp in &surf_data {
                let rdot = r_hat[0] * sp.pos[0] + r_hat[1] * sp.pos[1] + r_hat[2] * sp.pos[2];
                let phase = (j * C64::from(k0 * rdot)).exp();
                let daw = C64::from(sp.aw) * phase;

                let jx = C64::from(sp.normal[1]) * sp.h[2] - C64::from(sp.normal[2]) * sp.h[1];
                let jy = C64::from(sp.normal[2]) * sp.h[0] - C64::from(sp.normal[0]) * sp.h[2];
                let jz = C64::from(sp.normal[0]) * sp.h[1] - C64::from(sp.normal[1]) * sp.h[0];

                let (mx, my, mz) = if sp.is_pec {
                    (C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0))
                } else {
                    (
                        -(C64::from(sp.normal[1]) * sp.e[2] - C64::from(sp.normal[2]) * sp.e[1]),
                        -(C64::from(sp.normal[2]) * sp.e[0] - C64::from(sp.normal[0]) * sp.e[2]),
                        -(C64::from(sp.normal[0]) * sp.e[1] - C64::from(sp.normal[1]) * sp.e[0]),
                    )
                };

                let j_t = jx * C64::from(theta_hat[0]) + jy * C64::from(theta_hat[1]) + jz * C64::from(theta_hat[2]);
                let j_p = jx * C64::from(phi_hat[0]) + jy * C64::from(phi_hat[1]) + jz * C64::from(phi_hat[2]);
                let m_t = mx * C64::from(theta_hat[0]) + my * C64::from(theta_hat[1]) + mz * C64::from(theta_hat[2]);
                let m_p = mx * C64::from(phi_hat[0]) + my * C64::from(phi_hat[1]) + mz * C64::from(phi_hat[2]);

                nt += j_t * daw;
                np += j_p * daw;
                lt += m_t * daw;
                lp += m_p * daw;
            }

            let factor = -j * C64::from(k0 / (4.0 * PI));
            let e_t = factor * (lp + C64::from(Z0) * nt);
            let e_p = factor * (-lt + C64::from(Z0) * np);
            let u = (e_t.norm().powi(2) + e_p.norm().powi(2)) / (2.0 * Z0);
            (ip, it, e_t, e_p, u)
        };

        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            pairs.par_iter().map(compute_one).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            pairs.iter().map(compute_one).collect()
        }
    };

    let mut total_power = 0.0;
    for (ip, it, e_t, e_p, u) in &direction_results {
        e_theta[*ip][*it] = *e_t;
        e_phi[*ip][*it] = *e_p;
        directivity[*ip][*it] = *u;
        total_power += u * thetas[*it].sin() * dtheta * dphi;
    }

    // D(θ,φ) = 4π U(θ,φ) / P_rad
    let mut peak_d = 0.0f64;
    for ip in 0..n_phi {
        for it in 0..n_theta {
            let d = if total_power > 0.0 {
                4.0 * PI * directivity[ip][it] / total_power
            } else {
                0.0
            };
            let d_dbi = if d > SINGULAR_EPS { 10.0 * d.log10() } else { FARFIELD_DB_FLOOR };
            directivity[ip][it] = d_dbi;
            peak_d = peak_d.max(d_dbi);
        }
    }

    // Gain: directivity scaled by user-supplied radiation efficiency.
    let efficiency = radiation_efficiency.unwrap_or(1.0).clamp(0.0, 1.0);
    let efficiency_db_offset = if efficiency > SINGULAR_EPS { 10.0 * efficiency.log10() } else { FARFIELD_DB_FLOOR };

    let mut gain = vec![vec![0.0f64; n_theta]; n_phi];
    let mut ar = vec![vec![0.0f64; n_theta]; n_phi];
    let mut lcp = vec![vec![0.0f64; n_theta]; n_phi];
    let mut rcp = vec![vec![0.0f64; n_theta]; n_phi];

    let mut peak_g = f64::NEG_INFINITY;
    for ip in 0..n_phi {
        for it in 0..n_theta {
            // Gain in dBi
            let g_dbi = directivity[ip][it] + efficiency_db_offset;
            gain[ip][it] = g_dbi;
            peak_g = peak_g.max(g_dbi);

            // Axial ratio from polarization ellipse: needs |E_θ|, |E_φ|, phase difference δ.
            //   AR² = (a/b)² where a,b are major/minor semi-axes.
            //   a²+b² = |E_θ|² + |E_φ|²
            //   a²-b² = sqrt((|E_θ|² - |E_φ|²)² + 4|E_θ|²|E_φ|² cos²(δ))
            //   ab    = |E_θ|·|E_φ|·|sin(δ)|
            let et = e_theta[ip][it];
            let ep = e_phi[ip][it];
            let etn = et.norm();
            let epn = ep.norm();
            let mag_sq_sum = etn * etn + epn * epn;
            if mag_sq_sum > SINGULAR_EPS {
                let delta = ep.arg() - et.arg();
                let cos_d = delta.cos();
                let diff_sq = (etn * etn - epn * epn).powi(2) + 4.0 * etn * etn * epn * epn * cos_d * cos_d;
                let a_sq_minus_b_sq = diff_sq.sqrt();
                let a_sq = 0.5 * (mag_sq_sum + a_sq_minus_b_sq);
                let b_sq = 0.5 * (mag_sq_sum - a_sq_minus_b_sq).max(0.0);
                ar[ip][it] = if b_sq > SINGULAR_EPS {
                    10.0 * (a_sq / b_sq).log10()
                } else {
                    99.0  // effectively linear polarization
                };
            } else {
                ar[ip][it] = 99.0;
            }

            // LCP/RCP decomposition. Using the IEEE convention (E_LCP = (E_θ - jE_φ)/√2,
            // E_RCP = (E_θ + jE_φ)/√2). |E_LCP|² and |E_RCP|² convert to a directivity-style
            // dBi via the same 4π·U/P_rad scaling, but only the ratio of LCP:RCP magnitudes
            // is conventionally meaningful here, so we report each as a directivity-equivalent.
            let inv_sqrt2 = 1.0 / std::f64::consts::SQRT_2;
            let e_lcp = (et - C64::new(0.0, 1.0) * ep) * C64::from(inv_sqrt2);
            let e_rcp = (et + C64::new(0.0, 1.0) * ep) * C64::from(inv_sqrt2);
            // |E|² → directivity scaling. Each pol contains half the power (in pure linear case)
            // or all of one and none of the other (in pure circular). Convert |E|² to a dBi value
            // via the same 4π·U/P normalization used for directivity.
            let u_lcp = e_lcp.norm_sqr() / (2.0 * Z0);
            let u_rcp = e_rcp.norm_sqr() / (2.0 * Z0);
            let denom = total_power.max(SINGULAR_EPS);
            let d_lcp = 4.0 * PI * u_lcp / denom;
            let d_rcp = 4.0 * PI * u_rcp / denom;
            lcp[ip][it] = if d_lcp > SINGULAR_EPS { 10.0 * d_lcp.log10() } else { FARFIELD_DB_FLOOR };
            rcp[ip][it] = if d_rcp > SINGULAR_EPS { 10.0 * d_rcp.log10() } else { FARFIELD_DB_FLOOR };
        }
    }

    eprintln!("  Peak directivity: {:.2} dBi, peak gain: {:.2} dBi, radiation efficiency: {:.1}%",
        peak_d, peak_g, efficiency * 100.0);

    RadiationPattern {
        theta: thetas,
        phi: phis,
        e_theta,
        e_phi,
        directivity_dbi: directivity,
        gain_dbi: gain,
        axial_ratio_db: ar,
        lcp_dbi: lcp,
        rcp_dbi: rcp,
        peak_directivity_dbi: peak_d,
        peak_gain_dbi: peak_g,
        radiated_power: total_power,
    }
}

/// Write radiation pattern to a CSV file for plotting.
pub fn write_pattern_csv(
    path: &str,
    pattern: &RadiationPattern,
) -> Result<(), String> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("Cannot create {}: {}", path, e))?;

    writeln!(file, "phi_deg,theta_deg,directivity_dBi,gain_dBi,AR_dB,LCP_dBi,RCP_dBi,E_theta_re,E_theta_im,E_phi_re,E_phi_im")
        .map_err(|e| e.to_string())?;

    for (ip, &phi) in pattern.phi.iter().enumerate() {
        for (it, &theta) in pattern.theta.iter().enumerate() {
            let et = pattern.e_theta[ip][it];
            let ep = pattern.e_phi[ip][it];
            writeln!(file,
                "{:.1},{:.1},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6e},{:.6e},{:.6e},{:.6e}",
                phi.to_degrees(),
                theta.to_degrees(),
                pattern.directivity_dbi[ip][it],
                pattern.gain_dbi[ip][it],
                pattern.axial_ratio_db[ip][it],
                pattern.lcp_dbi[ip][it],
                pattern.rcp_dbi[ip][it],
                et.re, et.im, ep.re, ep.im,
            ).map_err(|e| e.to_string())?;
        }
    }

    Ok(())
}

/// Write principal plane cuts (E-plane and H-plane) to CSV.
pub fn write_plane_cuts_csv(
    path: &str,
    pattern: &RadiationPattern,
) -> Result<(), String> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("Cannot create {}: {}", path, e))?;

    writeln!(file, "plane,theta_deg,directivity_dBi,gain_dBi,AR_dB,LCP_dBi,RCP_dBi")
        .map_err(|e| e.to_string())?;

    let mut write_cut = |ip: usize, label: &str| -> Result<(), String> {
        for (it, &theta) in pattern.theta.iter().enumerate() {
            writeln!(file, "{},{:.1},{:.4},{:.4},{:.4},{:.4},{:.4}",
                label,
                theta.to_degrees(),
                pattern.directivity_dbi[ip][it],
                pattern.gain_dbi[ip][it],
                pattern.axial_ratio_db[ip][it],
                pattern.lcp_dbi[ip][it],
                pattern.rcp_dbi[ip][it],
            ).map_err(|e| e.to_string())?;
        }
        Ok(())
    };

    // E-plane: φ=0° (xz-plane)
    write_cut(0, "E")?;

    // H-plane: φ=90°
    let ip_h = pattern.phi.iter().position(|&p| (p - PI/2.0).abs() < 0.01)
        .unwrap_or(pattern.phi.len() / 4);
    write_cut(ip_h, "H")?;

    Ok(())
}
