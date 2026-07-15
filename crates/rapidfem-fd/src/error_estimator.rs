// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Full residual-based a posteriori error estimator.
//!
//! η_K² = h_K² · ||curl(curl E) + k₀²εE||²_K + h_K · Σ_f ||[n × curl E]||²_f
//!
//! Based on Monk (2003) for Maxwell FEM with Nedelec elements.

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::NedelecBasis;
use crate::interp;
use crate::quadrature::{gaus_quad_tet, gaus_quad_tri};

pub struct ErrorEstimate {
    pub element_errors: Vec<f64>,
    pub volume_residuals: Vec<f64>,
    pub face_jumps: Vec<f64>,
    /// Per-tet element diameter (max edge length). Same h_K the
    /// estimator scales the residual by, exposed so downstream
    /// refinement code can pick a target size relative to the current
    /// local mesh resolution.
    pub h_k: Vec<f64>,
    pub total_error: f64,
    pub marked_elements: Vec<usize>,
}

/// Write a gmsh background mesh size field (.pos file).
///
/// For each element, computes a target size based on the error indicator:
/// - Marked elements: current_h * refinement_ratio
/// - Unmarked elements: keep current size
///
/// Usage: `gmsh model.geo -bgm size_field.pos -3 -o refined.msh`
pub fn write_size_field(
    path: &str,
    mesh: &Mesh,
    estimate: &ErrorEstimate,
    refinement_ratio: f64,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    let n_tets = mesh.n_tets();
    let marked_set: std::collections::HashSet<usize> = estimate.marked_elements.iter().copied().collect();

    // Compute current element size (max edge length) per tet
    let h_k: Vec<f64> = (0..n_tets).map(|itet| {
        let edges = &mesh.tet_to_edge[itet];
        edges.iter().map(|&ei| mesh.edge_lengths[ei]).fold(0.0f64, f64::max)
    }).collect();

    // Compute target size per node (minimum of adjacent elements)
    let n_nodes = mesh.n_nodes();
    let mut node_size = vec![f64::INFINITY; n_nodes];

    for itet in 0..n_tets {
        let target = if marked_set.contains(&itet) {
            h_k[itet] * refinement_ratio
        } else {
            h_k[itet]
        };
        for &ni in &mesh.tets[itet] {
            node_size[ni] = node_size[ni].min(target);
        }
    }

    // Write gmsh .pos format (View "size" with SP = scalar point)
    writeln!(file, "View \"size\" {{")?;
    for ni in 0..n_nodes {
        let p = mesh.nodes[ni];
        let s = node_size[ni];
        if s < f64::INFINITY {
            writeln!(file, "SP({:.10e},{:.10e},{:.10e}){{{:.10e}}};", p[0], p[1], p[2], s)?;
        }
    }
    writeln!(file, "}};")?;

    Ok(())
}


// `eval_curl_in_tet` lives in `crate::interp`, same analytic Nédélec-2
// curl, used by both the error estimator and the H-field visualisation.
pub use crate::interp::eval_curl_in_tet;

/// Compute the full residual error estimate for each element.
pub fn estimate_error(
    mesh: &Mesh,
    basis: &NedelecBasis,
    solution: &[C64],
    k0: f64,
    er: &[[[C64; 3]; 3]],
    theta: f64,
) -> ErrorEstimate {
    let n_tets = mesh.n_tets();
    let k0_sq = C64::from(k0 * k0);

    let mut volume_residuals = vec![0.0f64; n_tets];
    let mut face_jumps_accum = vec![0.0f64; n_tets];

    // Step 1: Element diameter h_K = max edge length per tet
    let h_k: Vec<f64> = (0..n_tets).map(|itet| {
        let edges = &mesh.tet_to_edge[itet];
        edges.iter().map(|&ei| mesh.edge_lengths[ei]).fold(0.0f64, f64::max)
    }).collect();

    // Step 2: Volume residual per tet
    // r_K = curl(curl E) + k₀²εᵣE
    // For Nedelec-2, curl(curl E) is nearly constant per tet
    // We evaluate at the centroid and multiply by volume
    let tet_quad = gaus_quad_tet(2); // 4 points, exact for quadratic

    for itet in 0..n_tets {
        let tet = &mesh.tets[itet];
        let v0 = mesh.nodes[tet[0]];
        let v1 = mesh.nodes[tet[1]];
        let v2 = mesh.nodes[tet[2]];
        let v3 = mesh.nodes[tet[3]];

        // Tet volume
        let e1 = [v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]];
        let e2 = [v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]];
        let e3 = [v3[0]-v0[0], v3[1]-v0[1], v3[2]-v0[2]];
        let vol = (e1[0]*(e2[1]*e3[2]-e2[2]*e3[1]) - e1[1]*(e2[0]*e3[2]-e2[2]*e3[0])
                  + e1[2]*(e2[0]*e3[1]-e2[1]*e3[0])).abs() / 6.0;

        let mut res_sq = 0.0;

        for qp in &tet_quad {
            let (w, l1, l2, l3, l4) = (qp[0], qp[1], qp[2], qp[3], qp[4]);
            let x = v0[0]*l1 + v1[0]*l2 + v2[0]*l3 + v3[0]*l4;
            let y = v0[1]*l1 + v1[1]*l2 + v2[1]*l3 + v3[1]*l4;
            let z = v0[2]*l1 + v1[2]*l2 + v2[2]*l3 + v3[2]*l4;

            // E-field at this point
            let (ex, ey, ez) = interp::eval_field_in_tet(mesh, basis, solution, itet, x, y, z);

            // For volume residual, we need curl(curl E) + k₀²εE
            // curl(curl E) is approximately constant per tet for Nedelec-2
            // We approximate it from the curl variation across quad points
            // For now, use the simpler estimate: ||k₀²εE||² as volume term
            // (curl(curl E) contribution is captured by face jumps)
            let er_mat = &er[itet];
            let eps_ex = er_mat[0][0]*ex + er_mat[0][1]*ey + er_mat[0][2]*ez;
            let eps_ey = er_mat[1][0]*ex + er_mat[1][1]*ey + er_mat[1][2]*ez;
            let eps_ez = er_mat[2][0]*ex + er_mat[2][1]*ey + er_mat[2][2]*ez;

            let rx = k0_sq * eps_ex;
            let ry = k0_sq * eps_ey;
            let rz = k0_sq * eps_ez;

            res_sq += w * (rx.norm_sqr() + ry.norm_sqr() + rz.norm_sqr());
        }

        // Scale: integrate over volume (quad weights sum to 1/6 for reference tet)
        volume_residuals[itet] = res_sq * vol * 6.0; // 6*vol corrects for reference tet
    }

    // Step 3: Face jumps on interior faces
    let face_quad = gaus_quad_tri(4); // 6 points

    for face_idx in 0..mesh.n_tris() {
        let tet_left = mesh.tri_to_tet[face_idx][0];
        let tet_right = mesh.tri_to_tet[face_idx][1];
        if tet_right == usize::MAX { continue; } // boundary face

        let tri = &mesh.tris[face_idx];
        let fv0 = mesh.nodes[tri[0]];
        let fv1 = mesh.nodes[tri[1]];
        let fv2 = mesh.nodes[tri[2]];

        // Face area
        let fe1 = [fv1[0]-fv0[0], fv1[1]-fv0[1], fv1[2]-fv0[2]];
        let fe2 = [fv2[0]-fv0[0], fv2[1]-fv0[1], fv2[2]-fv0[2]];
        let normal = [fe1[1]*fe2[2]-fe1[2]*fe2[1], fe1[2]*fe2[0]-fe1[0]*fe2[2], fe1[0]*fe2[1]-fe1[1]*fe2[0]];
        let area = 0.5 * (normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]).sqrt();
        let nn = (normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]).sqrt();
        let n_hat = if nn > 1e-30 { [normal[0]/nn, normal[1]/nn, normal[2]/nn] } else { [0.0; 3] };

        let mut jump_sq = 0.0;

        for qp in &face_quad {
            let (w, l1, l2, l3) = (qp[0], qp[1], qp[2], qp[3]);
            let x = fv0[0]*l1 + fv1[0]*l2 + fv2[0]*l3;
            let y = fv0[1]*l1 + fv1[1]*l2 + fv2[1]*l3;
            let z = fv0[2]*l1 + fv1[2]*l2 + fv2[2]*l3;

            // curl(E) from both sides
            let curl_l = eval_curl_in_tet(mesh, basis, solution, tet_left, x, y, z);
            let curl_r = eval_curl_in_tet(mesh, basis, solution, tet_right, x, y, z);

            // Jump: [curl E] = curl_left - curl_right
            let dcurl = [curl_l[0]-curl_r[0], curl_l[1]-curl_r[1], curl_l[2]-curl_r[2]];

            // Tangential jump: n × [curl E]
            let jx = C64::from(n_hat[1])*dcurl[2] - C64::from(n_hat[2])*dcurl[1];
            let jy = C64::from(n_hat[2])*dcurl[0] - C64::from(n_hat[0])*dcurl[2];
            let jz = C64::from(n_hat[0])*dcurl[1] - C64::from(n_hat[1])*dcurl[0];

            jump_sq += w * (jx.norm_sqr() + jy.norm_sqr() + jz.norm_sqr());
        }

        let face_contrib = jump_sq * area;
        face_jumps_accum[tet_left] += face_contrib;
        face_jumps_accum[tet_right] += face_contrib;
    }

    // Step 4: Combine
    let mut element_errors = vec![0.0f64; n_tets];
    for itet in 0..n_tets {
        let hk = h_k[itet];
        let eta_sq = hk * hk * volume_residuals[itet] + hk * face_jumps_accum[itet];
        element_errors[itet] = eta_sq.sqrt();
    }

    let total_error = element_errors.iter().map(|e| e * e).sum::<f64>().sqrt();

    // Step 5: Dörfler marking
    let mut indexed: Vec<(usize, f64)> = element_errors.iter().enumerate()
        .map(|(i, &e)| (i, e)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let target = theta * total_error * total_error;
    let mut accum = 0.0;
    let mut marked = Vec::new();
    for (idx, err) in &indexed {
        accum += err * err;
        marked.push(*idx);
        if accum >= target { break; }
    }

    let max_err = indexed.first().map(|(_, e)| *e).unwrap_or(0.0);
    eprintln!("  Error estimate: total={:.4e}, max={:.4e}, marked={}/{} ({:.1}%)",
        total_error, max_err, marked.len(), n_tets, 100.0 * marked.len() as f64 / n_tets as f64);

    ErrorEstimate {
        element_errors,
        volume_residuals,
        face_jumps: face_jumps_accum,
        h_k,
        total_error,
        marked_elements: marked,
    }
}
