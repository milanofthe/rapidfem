// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Near-field to far-field transformation (NFFT) for radiation patterns.
//!
//! Love's equivalence on a closed surface S enclosing every source: with the
//! outward normal n̂,
//!   J_s = n̂ × H     (equivalent electric current)
//!   M_s = -n̂ × E    (equivalent magnetic current)
//! radiate the field outside S in free space. S is the outer boundary of the
//! computational domain (the ABC, the ground and walls: n̂ × E vanishes on
//! PEC by construction, so M_s drops out there by itself), or a surface
//! marked explicitly (the air / PML interface). Sheets inside S (patches,
//! strips) are not part of it: their currents are sources the fields on S
//! already carry.
//!
//! Far-field radiation integrals:
//!   N(θ,φ) = ∫∫ J_s · e^{jk r̂·r'} dS'
//!   L(θ,φ) = ∫∫ M_s · e^{jk r̂·r'} dS'
//!
//! E_θ^far = -jk/(4π) (L_φ + η₀ N_θ)
//! E_φ^far = -jk/(4π) (-L_θ + η₀ N_φ)
//!
//! A domain resting on an infinite PEC (or PMC) plane, a ground plane or a
//! symmetry plane, has that plane as part of its boundary: the surface is
//! then the rest of the boundary, closed by the images of its currents in
//! the plane (PEC: J' = -R J, M' = R M; PMC: J' = R J, M' = -R M, with R the
//! reflection), and the far field lives in the half-space of the domain.

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::NedelecBasis;
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
    /// Realized gain in dBi: directivity × the accepted power fraction
    /// (1 - Σ|S_i1|²) when it is given, which a lossless antenna radiates
    /// in full.
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

/// An infinite PEC or PMC plane through `point` (mesh units), `normal`
/// pointing into the domain's half-space.
#[derive(Clone, Copy, Debug)]
pub struct ImagePlane {
    pub point: [f64; 3],
    pub normal: [f64; 3],
    pub pec: bool,
}

/// The far-field pattern of the FEM solution from the closed surface
/// `surface_tris`. A boundary triangle takes its fields from its one tet and
/// its normal away from it; a triangle inside the domain (a marked Huygens
/// surface) takes them from its first tet, its normal away from the
/// surface's centroid (the surface is taken to be star-shaped about it, as
/// a box is).
///
/// With an `image` plane the surface is closed by the images of its
/// currents and the pattern is zero outside the domain's half-space.
///
/// `accepted_fraction`: the fraction of the incident power the antenna
/// accepts, for the realized gain; `None` makes the gain the directivity.
#[allow(clippy::too_many_arguments)]
pub fn compute_farfield(
    mesh: &Mesh,
    basis: &NedelecBasis,
    solution: &[C64],
    surface_tris: &[usize],
    image: Option<ImagePlane>,
    frequency: f64,
    n_theta: usize,
    n_phi: usize,
    gq_order: usize,
    accepted_fraction: Option<f64>,
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

    let quad_pts = gaus_quad_tri(gq_order);

    // For each observation direction (theta, phi), compute N and L integrals
    let mut directivity = vec![vec![0.0f64; n_theta]; n_phi];
    let mut e_theta = vec![vec![C64::new(0.0, 0.0); n_theta]; n_phi];
    let mut e_phi = vec![vec![C64::new(0.0, 0.0); n_theta]; n_phi];

    let centroid_of = |tri: [usize; 3]| -> [f64; 3] {
        let [a, b, c] = tri.map(|v| mesh.nodes[v]);
        std::array::from_fn(|k| (a[k] + b[k] + c[k]) / 3.0)
    };
    // area-weighted centroid of the surface, for the inner triangles' normals
    let mut surface_centre = [0.0; 3];
    let mut surface_area = 0.0;
    for &t in surface_tris {
        let [a, b, c] = mesh.tris[t].map(|v| mesh.nodes[v]);
        let area = 0.5 * norm3(cross3(sub3(b, a), sub3(c, a)));
        let m = centroid_of(mesh.tris[t]);
        for k in 0..3 {
            surface_centre[k] += area * m[k];
        }
        surface_area += area;
    }
    let surface_centre = surface_centre.map(|v| v / surface_area.max(f64::MIN_POSITIVE));

    // Per quadrature point: position, the equivalent currents J = n̂ × H and
    // M = -n̂ × E, area × weight (physical units).
    struct SurfPoint {
        pos: [f64; 3],
        j: [C64; 3],
        m: [C64; 3],
        aw: f64,
    }
    let ccross = |n: [f64; 3], v: [C64; 3]| -> [C64; 3] {
        [
            C64::from(n[1]) * v[2] - C64::from(n[2]) * v[1],
            C64::from(n[2]) * v[0] - C64::from(n[0]) * v[2],
            C64::from(n[0]) * v[1] - C64::from(n[1]) * v[0],
        ]
    };
    let mut surf_data: Vec<SurfPoint> = Vec::with_capacity(surface_tris.len() * quad_pts.len());
    for &tri_idx in surface_tris {
        let tri = mesh.tris[tri_idx];
        let [v0, v1, v2] = tri.map(|v| mesh.nodes[v]);
        let cr = cross3(sub3(v1, v0), sub3(v2, v0));
        let area = 0.5 * norm3(cr);
        let mut normal = cr.map(|c| c / (2.0 * area));
        let [t0, t1] = mesh.tri_to_tet[tri_idx];
        let centre = centroid_of(tri);
        let away_from = if t1 == usize::MAX {
            // outward: away from the one tet
            let tet = mesh.tets[t0];
            std::array::from_fn(|k| tet.iter().map(|&v| mesh.nodes[v][k]).sum::<f64>() / 4.0)
        } else {
            surface_centre
        };
        if dot3(normal, sub3(centre, away_from)) < 0.0 {
            normal = normal.map(|c| -c);
        }
        let tet = t0;
        for qp in &quad_pts {
            let (w, l1, l2, l3) = (qp[0], qp[1], qp[2], qp[3]);
            let x = v0[0] * l1 + v1[0] * l2 + v2[0] * l3;
            let y = v0[1] * l1 + v1[1] * l2 + v2[1] * l3;
            let z = v0[2] * l1 + v1[2] * l2 + v2[2] * l3;
            // Reconstruct on the (L₀-normalized) mesh, then convert to
            // physical units: E_recon = L₀·E_phys, curl_recon = L₀²·curl_phys.
            let il = 1.0 / l0;
            let (ex, ey, ez) = interp::eval_field_in_tet(mesh, basis, solution, tet, x, y, z);
            // curl(E) = -jωμ₀ H  =>  H = curl(E) / (-jωμ₀); the extra L₀²
            // (curl normalization) is folded into the denominator.
            let curl_e = eval_curl_in_tet(mesh, basis, solution, tet, x, y, z);
            let denom = -j * C64::from(omega * MU0 * l0 * l0);
            let e = [ex * il, ey * il, ez * il];
            let h = curl_e.map(|c| c / denom);
            surf_data.push(SurfPoint {
                pos: [x * l0, y * l0, z * l0],
                j: ccross(normal, h),
                m: ccross(normal, e).map(|c| -c),
                aw: area * w * l0 * l0,
            });
        }
    }
    // The images in the plane close the surface.
    if let Some(ip) = image {
        let n = ip.normal;
        let p0 = ip.point.map(|c| c * l0);
        let reflect = |v: [C64; 3]| -> [C64; 3] {
            let vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
            std::array::from_fn(|k| v[k] - vn * C64::from(2.0 * n[k]))
        };
        let (sj, sm) = if ip.pec { (-1.0, 1.0) } else { (1.0, -1.0) };
        let images: Vec<SurfPoint> = surf_data
            .iter()
            .map(|sp| {
                let d = dot3(sub3(sp.pos, p0), n);
                SurfPoint {
                    pos: std::array::from_fn(|k| sp.pos[k] - 2.0 * d * n[k]),
                    j: reflect(sp.j).map(|c| c * sj),
                    m: reflect(sp.m).map(|c| c * sm),
                    aw: sp.aw,
                }
            })
            .collect();
        surf_data.extend(images);
    }

    eprintln!("  Far-field: {} surface integration points on {} tris",
        surf_data.len(), surface_tris.len());

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
            // behind an image plane there is no field
            if image.is_some_and(|ip| dot3(r_hat, ip.normal) < -1e-12) {
                return (ip, it, C64::new(0.0, 0.0), C64::new(0.0, 0.0), 0.0);
            }

            let mut nt = C64::new(0.0, 0.0);
            let mut np = C64::new(0.0, 0.0);
            let mut lt = C64::new(0.0, 0.0);
            let mut lp = C64::new(0.0, 0.0);

            for sp in &surf_data {
                let rdot = r_hat[0] * sp.pos[0] + r_hat[1] * sp.pos[1] + r_hat[2] * sp.pos[2];
                let phase = (j * C64::from(k0 * rdot)).exp();
                let daw = C64::from(sp.aw) * phase;

                let [jx, jy, jz] = sp.j;
                let [mx, my, mz] = sp.m;

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

    // Realized gain: directivity scaled by the accepted power fraction.
    let efficiency = accepted_fraction.unwrap_or(1.0).clamp(0.0, 1.0);
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

    eprintln!("  Peak directivity: {:.2} dBi, peak realized gain: {:.2} dBi, accepted power: {:.1}%",
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

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}
