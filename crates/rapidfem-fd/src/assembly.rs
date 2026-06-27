// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Frequency-domain assemble-and-solve pipeline.
//!
//! The standard vector-FEM driven-problem assembly:
//! 1. element stiffness E and mass B from the R2 volume assembly,
//! 2. system matrix K = E − k₀²·B,
//! 3. PEC: drop the DOFs on perfect-conductor faces,
//! 4. Robin: add the port-surface boundary term to K,
//! 5. port excitation vector b from the incident modes,
//! 6. eliminate the constrained DOFs and solve K·x = b per driven port.
//!
//! The sweep variant caches E/B for frequency-independent materials and reuses
//! the solver's symbolic factorisation across frequencies.

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::port::Port;
use crate::tet_assembly_r2::assemble_global_matrices;
use crate::tri_assembly_r2::{ned2_tri_stiff, ned2_tri_force};
use crate::coefficients::AreaCoeffCache;
use crate::quadrature::gaus_quad_tri;
use std::collections::HashSet;

pub struct SolveResult {
    pub solutions: Vec<Vec<C64>>,
    pub n_field: usize,
}

/// Symmetric diagonal (Jacobi) equilibration of a COO system: returns S with
/// S_i = 1/√|K_ii|. Solving (S K S)(S⁻¹x) = S b and recovering x = S y is a
/// diagonal similarity, so the solution is exact up to the now-better-
/// conditioned factorisation. It improves the backward error when DOFs span
/// very different scales (mixed materials/PML, sliver-adjacent rows). On a
/// uniform, well-shaped mesh S is ≈ constant and the transform is a no-op.
/// See `derivations/conditioning/`.
fn equilibration_scaling(n: usize, rows: &[usize], cols: &[usize], vals: &[C64]) -> Vec<f64> {
    let mut diag = vec![C64::new(0.0, 0.0); n];
    for k in 0..rows.len() {
        if rows[k] == cols[k] {
            diag[rows[k]] += vals[k];
        }
    }
    diag.iter().map(|d| {
        let m = d.norm();
        if m > crate::constants::SINGULAR_EPS { 1.0 / m.sqrt() } else { 1.0 }
    }).collect()
}

/// Scale a COO system in place to S K S.
#[inline]
fn apply_equilibration(rows: &[usize], cols: &[usize], vals: &mut [C64], s: &[f64]) {
    for k in 0..rows.len() {
        vals[k] *= s[rows[k]] * s[cols[k]];
    }
}

/// Assemble the driven system and solve for each driven port. Accepts any
/// Port type via trait objects.
pub fn assemble_and_solve(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    ports: &[&dyn Port],
    port_tri_indices: &[&[usize]],
    pec_tri_indices: &[usize],
    freq: f64,
    materials: Option<&[crate::materials::Material]>,
) -> Result<SolveResult, String> {
    assemble_and_solve_with_pml(mesh, basis, ports, port_tri_indices, pec_tri_indices, freq, materials, None)
}

pub fn assemble_and_solve_with_pml(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    ports: &[&dyn Port],
    port_tri_indices: &[&[usize]],
    pec_tri_indices: &[usize],
    freq: f64,
    materials: Option<&[crate::materials::Material]>,
    pml_regions: Option<&[crate::materials::PmlRegion]>,
) -> Result<SolveResult, String> {
    let exc = crate::excitation::Excitation::new(freq, mesh.l0);
    let k0 = exc.k0;
    let n_field = basis.n_field;
    let n_tets = mesh.n_tets();

    // Step 1: Build per-tet material tensors.
    let (er, ur) = if let Some(pml) = pml_regions {
        crate::materials::build_material_tensors_with_pml(
            n_tets, materials.unwrap_or(&[]), pml, mesh, freq,
        )
    } else if let Some(mats) = materials {
        crate::materials::build_material_tensors(n_tets, mats, freq)
    } else {
        // Default: air (identity tensors)
        let identity: [[C64; 3]; 3] = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
        ];
        (vec![identity; n_tets], vec![identity; n_tets])
    };

    let t0 = web_time::Instant::now();
    let (rows, cols, data_e, data_b) = assemble_global_matrices(mesh, basis, &er, &ur);
    eprintln!("  Assembled E,B in {:.1}ms ({} entries)", t0.elapsed().as_secs_f64()*1e3, rows.len());

    // Step 2: K = E - k0² * B (defer CSR construction, build faer triplets directly later)
    let t1 = web_time::Instant::now();
    let k0_sq = C64::from(k0 * k0);

    // Step 3: collect the PEC (perfect-conductor) DOFs to constrain.
    let mut pec_ids: HashSet<usize> = HashSet::new();

    for &ti in pec_tri_indices {
        // edge_ids = list(mesh.tri_to_edge[:,tri_ids].flatten())
        let edges = &mesh.tri_to_edge[ti];
        for &ei in edges {
            // eids = field.edge_to_field[:, ii]
            let edofs = &basis.edge_to_field[ei];
            for &d in edofs {
                pec_ids.insert(d);
            }
        }
        // tids = field.tri_to_field[:, ii]
        let tdofs = &basis.tri_to_field[ti];
        for &d in tdofs {
            pec_ids.insert(d);
        }
    }
    eprintln!("  PEC DOFs: {} of {}", pec_ids.len(), n_field);

    // Step 4: Robin / port boundary term, accumulated into a flat per-tri
    // buffer (Bempty) and added to K as COO triplets.
    let ac_base = AreaCoeffCache::new();
    let gauss_points = gaus_quad_tri(4);

    // Bempty = field.empty_tri_matrix()
    let mut bempty = basis.empty_tri_matrix();

    for (pi, (port, tri_ids)) in ports.iter().zip(port_tri_indices.iter()).enumerate() {
        let gamma = port.get_gamma(&exc);

        // Robin BC stiffness: for each port tri, compute 8x8 and write into flat array
        for &ti in *tri_ids {
            let tri = &mesh.tris[ti];
            let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];
            let bsub = ned2_tri_stiff(&verts, gamma, &ac_base);
            let p = ti * 64;
            for ii in 0..8 {
                for jj in 0..8 {
                    bempty[p + ii * 8 + jj] += bsub[ii][jj];
                }
            }
        }

        eprintln!("  Port {} Robin: gamma={:.4e}, {} tris, driven={}", pi, gamma, tri_ids.len(), port.is_driven());
    }

    eprintln!("  Robin BC assembled in {:.1}ms", t1.elapsed().as_secs_f64()*1e3);

    // Step 5: Port excitation vectors, only for driven ports
    let mut port_vectors: Vec<Vec<C64>> = Vec::new();
    let mut driven_port_indices: Vec<usize> = Vec::new();

    for (pi, (port, tri_ids)) in ports.iter().zip(port_tri_indices.iter()).enumerate() {
        if !port.is_driven() {
            continue; // ABC: no excitation vector
        }
        driven_port_indices.push(pi);

        let mut bvec = vec![C64::new(0.0, 0.0); n_field];

        for &ti in *tri_ids {
            let tri = &mesh.tris[ti];
            let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];

            let u_inc_at_qp: Vec<[C64; 3]> = gauss_points.iter().filter_map(|qp| {
                let (l1, l2, l3) = (qp[1], qp[2], qp[3]);
                let x = verts[0][0]*l1 + verts[1][0]*l2 + verts[2][0]*l3;
                let y = verts[0][1]*l1 + verts[1][1]*l2 + verts[2][1]*l3;
                let z = verts[0][2]*l1 + verts[1][2]*l2 + verts[2][2]*l3;
                port.get_uinc(x, y, z, &exc)
            }).collect();

            if u_inc_at_qp.len() == gauss_points.len() {
                let b_tri = ned2_tri_force(&verts, &u_inc_at_qp, &gauss_points);
                let dofs = &basis.tri_to_field[ti];
                for i in 0..8 {
                    bvec[dofs[i]] += b_tri[i];
                }
            }
        }

        let bnorm: f64 = bvec.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        eprintln!("  Port {} ||b|| = {:.6e}", pi, bnorm);
        port_vectors.push(bvec);
    }

    // Step 6: Eliminate PEC DOFs, build reduced system, solve
    let free_dofs: Vec<usize> = (0..n_field).filter(|d| !pec_ids.contains(d)).collect();
    let n_free = free_dofs.len();
    eprintln!("  Free DOFs: {}", n_free);

    let mut dof_to_free = vec![usize::MAX; n_field];
    for (fi, &d) in free_dofs.iter().enumerate() {
        dof_to_free[d] = fi;
    }

    // Build COO triplets for reduced system: K = (E - k0²*B) + Robin
    let t2 = web_time::Instant::now();
    let mut coo_rows: Vec<usize> = Vec::new();
    let mut coo_cols: Vec<usize> = Vec::new();
    let mut coo_vals: Vec<C64> = Vec::new();

    for i in 0..rows.len() {
        let r = rows[i];
        let c = cols[i];
        if pec_ids.contains(&r) || pec_ids.contains(&c) { continue; }
        coo_rows.push(dof_to_free[r]);
        coo_cols.push(dof_to_free[c]);
        coo_vals.push(data_e[i] - k0_sq * data_b[i]);
    }
    // Precompute non-zero Robin indices (avoids iterating all n_tris*64 entries)
    let robin_nonzero: Vec<usize> = (0..bempty.len())
        .filter(|&i| (bempty[i].re != 0.0 || bempty[i].im != 0.0)
            && !pec_ids.contains(&basis.tri_rows[i])
            && !pec_ids.contains(&basis.tri_cols[i]))
        .collect();
    for &idx in &robin_nonzero {
        coo_rows.push(dof_to_free[basis.tri_rows[idx]]);
        coo_cols.push(dof_to_free[basis.tri_cols[idx]]);
        coo_vals.push(bempty[idx]);
    }
    eprintln!("  COO: {} entries, built in {:.1}ms", coo_rows.len(), t2.elapsed().as_secs_f64()*1e3);

    // Symmetric diagonal equilibration: solve (S K S)(S⁻¹x) = S b, recover x = S y.
    let s_eq = equilibration_scaling(n_free, &coo_rows, &coo_cols, &coo_vals);
    apply_equilibration(&coo_rows, &coo_cols, &mut coo_vals, &s_eq);

    // Backend-agnostic factor + solve via the SparseSolver trait. Selection
    // honours RAPIDFEM_SOLVER (auto|pardiso|accelerate|faer).
    let mut solver = crate::solver::pick(crate::solver::SolverChoice::from_env());
    let t_solve = web_time::Instant::now();
    solver.factorize(n_free, &coo_rows, &coo_cols, &coo_vals)?;
    eprintln!("  {}: factorized in {:.1}ms", solver.name(), t_solve.elapsed().as_secs_f64()*1e3);

    let mut solutions = Vec::new();
    for (pi, bvec) in port_vectors.iter().enumerate() {
        let b_free: Vec<C64> = free_dofs.iter().enumerate()
            .map(|(fi, &d)| bvec[d] * C64::from(s_eq[fi])).collect();
        let x_free = solver.solve(&b_free)?;
        let mut x_full = vec![C64::new(0.0, 0.0); n_field];
        for (fi, &d) in free_dofs.iter().enumerate() {
            x_full[d] = x_free[fi] * C64::from(s_eq[fi]);
        }
        let xnorm: f64 = x_full.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        eprintln!("  Port {} solved ({}) in {:.1}ms, ||x|| = {:.6e}",
            pi, solver.name(), t_solve.elapsed().as_secs_f64()*1e3, xnorm);
        solutions.push(x_full);
    }

    Ok(SolveResult { solutions, n_field })
}

/// Frequency sweep: solve at multiple frequencies.
///
/// For frequency-independent materials, caches E and B matrices.
/// Returns solutions per frequency: Vec<SolveResult>.
pub fn frequency_sweep(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    ports: &[&dyn Port],
    port_tri_indices: &[&[usize]],
    pec_tri_indices: &[usize],
    frequencies: &[f64],
    materials: Option<&[crate::materials::Material]>,
) -> Result<Vec<SolveResult>, String> {
    frequency_sweep_with_pml(mesh, basis, ports, port_tri_indices, pec_tri_indices, frequencies, materials, None, None)
}

pub fn frequency_sweep_with_pml(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    ports: &[&dyn Port],
    port_tri_indices: &[&[usize]],
    pec_tri_indices: &[usize],
    frequencies: &[f64],
    materials: Option<&[crate::materials::Material]>,
    pml_regions: Option<&[crate::materials::PmlRegion]>,
    // Optional per-frequency hook, called after each frequency's solve with
    // (freq_idx, freq_hz, &SolveResult). Lets a caller stream partial results
    // (e.g. progressive S-parameters in the UI) without changing the output.
    // Returns `false` to stop the sweep early (e.g. on a user interrupt); the
    // frequencies solved so far are returned.
    mut on_solve: Option<&mut dyn FnMut(usize, f64, &SolveResult) -> bool>,
) -> Result<Vec<SolveResult>, String> {
    // Detect if any material is frequency-dependent, if so, K must be rebuilt every frequency
    let materials_dispersive = materials
        .map(|m| m.iter().any(|x| x.dispersion.is_dispersive()))
        .unwrap_or(false);
    if materials_dispersive {
        eprintln!("  Frequency-dependent materials detected - rebuilding K every frequency");
    }

    // Cache E, B for frequency-independent materials
    let n_tets = mesh.n_tets();
    let (er, ur) = if let Some(pml) = pml_regions {
        crate::materials::build_material_tensors_with_pml(
            n_tets, materials.unwrap_or(&[]), pml, mesh, frequencies[0],
        )
    } else if let Some(mats) = materials {
        crate::materials::build_material_tensors(n_tets, mats, frequencies[0])
    } else {
        let identity: [[C64; 3]; 3] = [
            [C64::new(1.0, 0.0), C64::new(0.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            [C64::new(0.0, 0.0), C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
        ];
        (vec![identity; n_tets], vec![identity; n_tets])
    };

    let t0 = web_time::Instant::now();
    let (rows, cols, mut data_e, mut data_b) = assemble_global_matrices(mesh, basis, &er, &ur);
    eprintln!("  Assembled E,B in {:.1}ms{}", t0.elapsed().as_secs_f64()*1e3,
        if materials_dispersive { "" } else { " (cached for sweep)" });

    // PEC DOFs (frequency-independent)
    let mut pec_ids: HashSet<usize> = HashSet::new();
    for &ti in pec_tri_indices {
        for &ei in &mesh.tri_to_edge[ti] {
            for &d in &basis.edge_to_field[ei] { pec_ids.insert(d); }
        }
        for &d in &basis.tri_to_field[ti] { pec_ids.insert(d); }
    }

    let free_dofs: Vec<usize> = (0..basis.n_field).filter(|d| !pec_ids.contains(d)).collect();
    let n_free = free_dofs.len();
    let mut dof_to_free = vec![usize::MAX; basis.n_field];
    for (fi, &d) in free_dofs.iter().enumerate() { dof_to_free[d] = fi; }

    let ac_base = crate::coefficients::AreaCoeffCache::new();
    let gauss_points = crate::quadrature::gaus_quad_tri(4);

    let mut results = Vec::with_capacity(frequencies.len());

    // Precompute non-PEC COO indices for K entries (reused every frequency)
    let k_free_indices: Vec<usize> = (0..rows.len())
        .filter(|&i| !pec_ids.contains(&rows[i]) && !pec_ids.contains(&cols[i]))
        .collect();
    let k_free_rows: Vec<usize> = k_free_indices.iter().map(|&i| dof_to_free[rows[i]]).collect();
    let k_free_cols: Vec<usize> = k_free_indices.iter().map(|&i| dof_to_free[cols[i]]).collect();

    // Precompute non-PEC Robin indices (reused every frequency)
    let robin_free_indices: Vec<usize> = (0..basis.n_tris * 64)
        .filter(|&idx| {
            let r = basis.tri_rows[idx];
            let c = basis.tri_cols[idx];
            !pec_ids.contains(&r) && !pec_ids.contains(&c)
        })
        .collect();

    // Pick backend once for the whole sweep, symbolic factorisation is
    // amortised across frequencies via `solver.refactorize`.
    let mut solver = crate::solver::pick(crate::solver::SolverChoice::from_env());
    let mut first_factor = true;

    // COO buffers for the per-frequency system matrix, reused across the
    // sweep. Capacity covers the K block plus the Robin upper bound.
    let coo_cap = k_free_indices.len() + robin_free_indices.len();
    let mut coo_rows: Vec<usize> = Vec::with_capacity(coo_cap);
    let mut coo_cols: Vec<usize> = Vec::with_capacity(coo_cap);
    let mut coo_vals: Vec<C64> = Vec::with_capacity(coo_cap);
    let mut bempty = basis.empty_tri_matrix();

    for (fi, &freq) in frequencies.iter().enumerate() {
        let t_freq = web_time::Instant::now();
        let exc = crate::excitation::Excitation::new(freq, mesh.l0);
        let k0 = exc.k0;
        let k0_sq = C64::from(k0 * k0);
        let n_field = basis.n_field;

        // Rebuild element matrices when materials are frequency-dependent
        if materials_dispersive && fi > 0 {
            let (er_f, ur_f) = if let Some(pml) = pml_regions {
                crate::materials::build_material_tensors_with_pml(
                    n_tets, materials.unwrap_or(&[]), pml, mesh, freq,
                )
            } else {
                crate::materials::build_material_tensors(n_tets, materials.unwrap_or(&[]), freq)
            };
            let (_, _, de, db) = assemble_global_matrices(mesh, basis, &er_f, &ur_f);
            data_e = de;
            data_b = db;
        }

        // Robin BC (γ frequency-dependent), reuse bempty buffer
        bempty.fill(C64::new(0.0, 0.0));
        for (_, (port, tri_ids)) in ports.iter().zip(port_tri_indices.iter()).enumerate() {
            let gamma = port.get_gamma(&exc);
            for &ti in *tri_ids {
                let tri = &mesh.tris[ti];
                let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];
                let bsub = ned2_tri_stiff(&verts, gamma, &ac_base);
                let p = ti * 64;
                for ii in 0..8 { for jj in 0..8 { bempty[p + ii*8 + jj] += bsub[ii][jj]; } }
            }
        }

        // Port excitation vectors
        let mut port_bvecs: Vec<Vec<C64>> = Vec::new();
        for (port, tri_ids) in ports.iter().zip(port_tri_indices.iter()) {
            if !port.is_driven() { continue; }
            let mut bvec = vec![C64::new(0.0, 0.0); n_field];
            for &ti in *tri_ids {
                let tri = &mesh.tris[ti];
                let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];
                let u_at_qp: Vec<[C64; 3]> = gauss_points.iter()
                    .filter_map(|qp| {
                        let (l1,l2,l3) = (qp[1],qp[2],qp[3]);
                        port.get_uinc(
                            verts[0][0]*l1+verts[1][0]*l2+verts[2][0]*l3,
                            verts[0][1]*l1+verts[1][1]*l2+verts[2][1]*l3,
                            verts[0][2]*l1+verts[1][2]*l2+verts[2][2]*l3, &exc)
                    }).collect();
                if u_at_qp.len() == gauss_points.len() {
                    let b_tri = ned2_tri_force(&verts, &u_at_qp, &gauss_points);
                    let dofs = &basis.tri_to_field[ti];
                    for i in 0..8 { bvec[dofs[i]] += b_tri[i]; }
                }
            }
            port_bvecs.push(bvec);
        }

        // Build the system matrix COO: K = (E - k0^2*B) + Robin, straight
        // into the solver's COO buffers, reusing the allocation.
        coo_rows.clear();
        coo_cols.clear();
        coo_vals.clear();

        for (ti, &orig_i) in k_free_indices.iter().enumerate() {
            coo_rows.push(k_free_rows[ti]);
            coo_cols.push(k_free_cols[ti]);
            coo_vals.push(data_e[orig_i] - k0_sq * data_b[orig_i]);
        }
        for &idx in &robin_free_indices {
            let val = bempty[idx];
            if val.re == 0.0 && val.im == 0.0 { continue; }
            coo_rows.push(dof_to_free[basis.tri_rows[idx]]);
            coo_cols.push(dof_to_free[basis.tri_cols[idx]]);
            coo_vals.push(val);
        }

        // Symmetric diagonal equilibration (recomputed per frequency; the
        // sparsity pattern is unchanged, so the symbolic factorisation reuse
        // via `refactorize` still holds).
        let s_eq = equilibration_scaling(n_free, &coo_rows, &coo_cols, &coo_vals);
        apply_equilibration(&coo_rows, &coo_cols, &mut coo_vals, &s_eq);

        // Factor (symbolic once via `factorize`, then `refactorize` per freq
        // reusing the sparsity pattern) and solve via the backend-agnostic
        // SparseSolver trait.
        if first_factor {
            solver.factorize(n_free, &coo_rows, &coo_cols, &coo_vals)?;
            first_factor = false;
        } else {
            solver.refactorize(n_free, &coo_rows, &coo_cols, &coo_vals)?;
        }

        let mut solutions = Vec::new();
        for bvec in &port_bvecs {
            let b_free: Vec<C64> = free_dofs.iter().enumerate()
                .map(|(fi_d, &d)| bvec[d] * C64::from(s_eq[fi_d])).collect();
            let x_free = solver.solve(&b_free)?;
            let mut x_full = vec![C64::new(0.0, 0.0); n_field];
            for (fi_d, &d) in free_dofs.iter().enumerate() {
                x_full[d] = x_free[fi_d] * C64::from(s_eq[fi_d]);
            }
            solutions.push(x_full);
        }

        eprintln!(
            "  f={:>8.4e} Hz [{:>2}/{:>2}]  {:>6.1}ms  {}",
            freq, fi + 1, frequencies.len(), t_freq.elapsed().as_secs_f64() * 1e3,
            solver.name(),
        );
        results.push(SolveResult { solutions, n_field });
        if let Some(cb) = on_solve.as_deref_mut() {
            if !cb(fi, freq, results.last().unwrap()) {
                eprintln!(
                    "  sweep stopped early after frequency {}/{} (interrupt)",
                    fi + 1, frequencies.len(),
                );
                break;
            }
        }
    }

    Ok(results)
}
#[cfg(test)]
mod conditioning_tests {
    use super::*;

    /// Equilibration must drive the diagonal magnitudes to ~1 (it is exactly
    /// 1/√|K_ii| symmetric scaling) — this is what tames a scale-mixed system.
    #[test]
    fn equilibration_unit_diagonal() {
        // 2-DOF system with a 1e8 spread between the two diagonals.
        let rows = vec![0, 0, 1, 1];
        let cols = vec![0, 1, 0, 1];
        let mut vals = vec![
            C64::new(1e6, 0.0), C64::new(1.0, 0.0),
            C64::new(1.0, 0.0), C64::new(1e-2, 0.0),
        ];
        let s = equilibration_scaling(2, &rows, &cols, &vals);
        apply_equilibration(&rows, &cols, &mut vals, &s);
        // diagonal entries are vals[0] (0,0) and vals[3] (1,1)
        assert!((vals[0].norm() - 1.0).abs() < 1e-12, "diag0 = {}", vals[0].norm());
        assert!((vals[3].norm() - 1.0).abs() < 1e-12, "diag1 = {}", vals[3].norm());
    }

    /// On a system that is already uniformly scaled, equilibration is ~identity.
    #[test]
    fn equilibration_noop_on_uniform() {
        let rows = vec![0, 1];
        let cols = vec![0, 1];
        let vals = vec![C64::new(2.0, 0.0), C64::new(2.0, 0.0)];
        let s = equilibration_scaling(2, &rows, &cols, &vals);
        for &si in &s { assert!((si - 1.0 / 2.0_f64.sqrt()).abs() < 1e-12); }
    }
}
