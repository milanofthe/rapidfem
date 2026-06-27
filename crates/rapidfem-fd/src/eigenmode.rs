// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Eigenmode solver: find resonant frequencies and Q-factors.
//!
//! Solves E*x = λ*B*x using shift-invert Lanczos.
//! Uses PARDISO (if available) or faer for the shift-invert linear solve.
//! Handles complex-symmetric matrices (lossy materials).

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::tet_assembly_r2::assemble_global_matrices;
use crate::constants::*;
use std::collections::HashSet;

pub struct Eigenmode {
    pub frequency: C64,
    pub q_factor: f64,
    pub eigenvalue: C64,
    pub field: Vec<C64>,
}

pub fn solve_eigenmode(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    pec_tri_indices: &[usize],
    materials: Option<&[crate::materials::Material]>,
    target_freq: f64,
    n_modes: usize,
) -> Result<Vec<Eigenmode>, String> {
    let n_tets = mesh.n_tets();
    let n_field = basis.n_field;

    // Assemble E and B
    let (er, ur) = if let Some(mats) = materials {
        crate::materials::build_material_tensors(n_tets, mats, target_freq)
    } else {
        let id: [[C64; 3]; 3] = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
        ];
        (vec![id; n_tets], vec![id; n_tets])
    };

    let t0 = web_time::Instant::now();
    let (rows, cols, data_e, data_b) = assemble_global_matrices(mesh, basis, &er, &ur);
    eprintln!("  Eigenmode: assembled E,B in {:.1}ms", t0.elapsed().as_secs_f64()*1e3);

    // PEC DOFs
    let mut pec_ids: HashSet<usize> = HashSet::new();
    for &ti in pec_tri_indices {
        for &ei in &mesh.tri_to_edge[ti] {
            for &d in &basis.edge_to_field[ei] { pec_ids.insert(d); }
        }
        for &d in &basis.tri_to_field[ti] { pec_ids.insert(d); }
    }
    let free_dofs: Vec<usize> = (0..n_field).filter(|d| !pec_ids.contains(d)).collect();
    let n_free = free_dofs.len();
    let mut dof_to_free = vec![usize::MAX; n_field];
    for (fi, &d) in free_dofs.iter().enumerate() { dof_to_free[d] = fi; }

    eprintln!("  Eigenmode: {} free DOFs, target={:.4e} Hz", n_free, target_freq);

    // Shift σ = (2πf/c)²
    let k0_target = crate::excitation::Excitation::new(target_freq, mesh.l0).k0;
    let sigma = C64::from(k0_target * k0_target);

    // Build COO for (E - σB) and B, filtered to free DOFs
    let mut shift_rows: Vec<usize> = Vec::new();
    let mut shift_cols: Vec<usize> = Vec::new();
    let mut shift_vals: Vec<C64> = Vec::new();
    let mut b_rows_free: Vec<usize> = Vec::new();
    let mut b_cols_free: Vec<usize> = Vec::new();
    let mut b_vals_free: Vec<C64> = Vec::new();

    for i in 0..rows.len() {
        let r = rows[i]; let c = cols[i];
        if pec_ids.contains(&r) || pec_ids.contains(&c) { continue; }
        let fi = dof_to_free[r];
        let fj = dof_to_free[c];
        shift_rows.push(fi);
        shift_cols.push(fj);
        shift_vals.push(data_e[i] - sigma * data_b[i]);
        if data_b[i].re != 0.0 || data_b[i].im != 0.0 {
            b_rows_free.push(fi);
            b_cols_free.push(fj);
            b_vals_free.push(data_b[i]);
        }
    }

    // Factor (E - σB) via the backend-agnostic SparseSolver trait.
    let t1 = web_time::Instant::now();
    let mut solver = crate::solver::pick(crate::solver::SolverChoice::from_env());
    solver.factorize(n_free, &shift_rows, &shift_cols, &shift_vals)?;
    eprintln!("  Eigenmode: {} shift-invert in {:.1}ms",
        solver.name(), t1.elapsed().as_secs_f64()*1e3);

    // Sparse mat-vec: y = B * x
    let b_matvec = |x: &[C64]| -> Vec<C64> {
        let mut y = vec![C64::new(0.0, 0.0); n_free];
        for k in 0..b_rows_free.len() {
            let i = b_rows_free[k];
            let j = b_cols_free[k];
            let v = b_vals_free[k];
            y[i] += v * x[j];
            if i != j { y[j] += v * x[i]; } // symmetric: add transpose
        }
        y
    };

    // Shift-invert Lanczos: find eigenvalues of (E-σB)⁻¹ B near σ
    let t2 = web_time::Instant::now();
    let n_lanczos = (3 * n_modes + 20).min(n_free).min(100);

    // Random start vector
    let mut v: Vec<C64> = (0..n_free).map(|i| C64::from(((i * 7 + 13) % 97) as f64 / 97.0)).collect();
    let norm: f64 = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    for x in &mut v { *x /= C64::from(norm); }

    let mut vecs: Vec<Vec<C64>> = Vec::new();
    let mut alphas: Vec<C64> = Vec::new();
    let mut betas: Vec<f64> = Vec::new();
    let mut v_prev = vec![C64::new(0.0, 0.0); n_free];
    let mut beta_prev = 0.0f64;

    for _j in 0..n_lanczos {
        vecs.push(v.clone());

        // w = (E - σB)⁻¹ * B * v
        let bv = b_matvec(&v);
        let mut w = solver.solve(&bv)?;

        // α = v^T * w (complex-symmetric inner product, NOT Hermitian)
        let alpha: C64 = v.iter().zip(w.iter()).map(|(vi, wi)| vi * wi).sum();
        alphas.push(alpha);

        // w = w - α*v - β*v_prev
        for i in 0..n_free {
            w[i] -= alpha * v[i] + C64::from(beta_prev) * v_prev[i];
        }

        // Re-orthogonalize against all previous vectors (full reorthogonalization)
        for prev in &vecs {
            let dot: C64 = prev.iter().zip(w.iter()).map(|(pi, wi)| pi * wi).sum();
            for i in 0..n_free { w[i] -= dot * prev[i]; }
        }

        // β = ||w||
        let w_norm: f64 = w.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        betas.push(w_norm);

        if w_norm < LANCZOS_BREAKDOWN { break; }

        v_prev = v;
        beta_prev = w_norm;
        v = w.iter().map(|x| x / C64::from(w_norm)).collect();
    }

    let m = alphas.len();
    eprintln!("  Eigenmode: {} Lanczos iterations in {:.1}ms", m, t2.elapsed().as_secs_f64()*1e3);

    // Build tridiagonal T and solve dense eigenvalue problem
    let t_mat = faer::Mat::<faer::c64>::from_fn(m, m, |i, j| {
        if i == j {
            faer::c64 { re: alphas[i].re, im: alphas[i].im }
        } else if j == i + 1 && i < betas.len() {
            faer::c64 { re: betas[i], im: 0.0 }
        } else if i == j + 1 && j < betas.len() {
            faer::c64 { re: betas[j], im: 0.0 }
        } else {
            faer::c64 { re: 0.0, im: 0.0 }
        }
    });

    let eig = t_mat.eigen()
        .map_err(|e| format!("eigendecomposition failed: {:?}", e))?;
    let eigenvalues = eig.S().column_vector();
    let eigenvectors = eig.U();

    // Convert: μ → λ = σ + 1/μ, then f = c√λ / 2π
    let mut modes: Vec<(C64, Vec<C64>)> = Vec::new();

    for k in 0..m {
        let mu = C64::new(eigenvalues[k].re, eigenvalues[k].im);
        if mu.norm() < SINGULAR_EPS { continue; }
        let lambda = sigma + C64::new(1.0, 0.0) / mu;
        if lambda.re <= 0.0 { continue; }

        // Ritz vector: x = V * y
        let mut x_free = vec![C64::new(0.0, 0.0); n_free];
        for j in 0..m.min(vecs.len()) {
            let coeff = C64::new(eigenvectors[(j, k)].re, eigenvectors[(j, k)].im);
            for i in 0..n_free {
                x_free[i] += coeff * vecs[j][i];
            }
        }

        // Map to full DOF vector
        let mut x_full = vec![C64::new(0.0, 0.0); n_field];
        for (fi, &d) in free_dofs.iter().enumerate() {
            x_full[d] = x_free[fi];
        }

        modes.push((lambda, x_full));
    }

    // Sort by distance to target
    modes.sort_by(|a, b| (a.0 - sigma).norm().partial_cmp(&(b.0 - sigma).norm()).unwrap());

    Ok(modes.into_iter().take(n_modes).map(|(lambda, field)| {
        // λ = κ², so √λ = κ = k₀·L₀; recover the physical frequency by /L₀.
        let k0 = lambda.sqrt() / C64::from(mesh.l0);
        let freq = k0 * C64::from(C0 / (2.0 * PI));
        let q = if freq.im.abs() > SINGULAR_EPS { 0.5 * freq.re / freq.im.abs() } else { f64::INFINITY };
        Eigenmode { frequency: freq, q_factor: q, eigenvalue: lambda, field }
    }).collect())
}
