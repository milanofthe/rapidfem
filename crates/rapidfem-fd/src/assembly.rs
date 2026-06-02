// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
// Copyright (C) Robert Fennis (original EMerge source)
//
// This file is part of rapidfem and contains code ported from EMerge
// (https://github.com/FennisRobert/EMerge), originally licensed under
// GPL-2.0-or-later with the Gmsh additional permission; redistributed
// here under GPL-3.0-or-later with that permission preserved.
// See LICENSE and NOTICE for the full terms.

//! Exact port of assembler.py: assemble_freq_matrix + solve pipeline.
//!
//! Follows EMerge's assembly order exactly:
//! 1. E, B = tet_mass_stiffness_matrices
//! 2. K = (E - k0² * B).tocsr()
//! 3. PEC: collect DOFs from edge_to_field and tri_to_field for PEC faces
//! 4. Robin: Bempty = empty_tri_matrix(); compute_bc_entries; K += generate_csr(Bempty)
//! 5. Port vectors: assemble_robin_bc_bvec (generate_points_3d + compute_force_entries)
//! 6. Eliminate PEC DOFs, solve K*x = b

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::port::Port;
use crate::tet_assembly::assemble_global_matrices;
use crate::tri_assembly::{ned2_tri_stiff, ned2_tri_force};
use crate::coefficients::AreaCoeffCache;
use crate::quadrature::gaus_quad_tri;
use crate::constants::PI;
use std::collections::HashSet;

pub struct SolveResult {
    pub solutions: Vec<Vec<C64>>,
    pub n_field: usize,
}

/// Exact port of assembler.py:assemble_freq_matrix + solve.
/// Now accepts any Port type via trait objects.
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

/// The PEC-eliminated reduced system `K x = b`, free-DOF indexed.
///
/// This is the common product of FD assembly: the reduced COO triplets, the
/// free-DOF map, and one free-indexed RHS per driven port. Both the monolithic
/// solve and the Schur-DD solve consume it, so they are guaranteed to see the
/// identical system (the basis of the bit-for-bit Schur validation gate).
pub struct ReducedSystem {
    pub n_free: usize,
    pub n_field: usize,
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<C64>,
    /// Global DOF for each free index (`free_dofs[fi]` → global DOF).
    pub free_dofs: Vec<usize>,
    /// One free-indexed RHS per driven port.
    pub rhs: Vec<Vec<C64>>,
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
    let sys = assemble_reduced_with_pml(
        mesh, basis, ports, port_tri_indices, pec_tri_indices, freq, materials, pml_regions,
    )?;

    let mut solver = crate::solver::pick(crate::solver::SolverChoice::from_env());
    let t_solve = web_time::Instant::now();
    solver.factorize(sys.n_free, &sys.rows, &sys.cols, &sys.vals)?;
    eprintln!("  {}: factorized in {:.1}ms", solver.name(), t_solve.elapsed().as_secs_f64()*1e3);

    let mut solutions = Vec::new();
    for (pi, b_free) in sys.rhs.iter().enumerate() {
        let x_free = solver.solve(b_free)?;
        let mut x_full = vec![C64::new(0.0, 0.0); sys.n_field];
        for (fi, &d) in sys.free_dofs.iter().enumerate() {
            x_full[d] = x_free[fi];
        }
        let xnorm: f64 = x_full.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        eprintln!("  Port {} solved ({}) in {:.1}ms, ||x|| = {:.6e}",
            pi, solver.name(), t_solve.elapsed().as_secs_f64()*1e3, xnorm);
        solutions.push(x_full);
    }

    Ok(SolveResult { solutions, n_field: sys.n_field })
}

/// Build the PEC-eliminated reduced system without solving it.
/// Factored out of `assemble_and_solve_with_pml` so the Schur-DD path and the
/// monolithic path share one assembly.
pub fn assemble_reduced_with_pml(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    ports: &[&dyn Port],
    port_tri_indices: &[&[usize]],
    pec_tri_indices: &[usize],
    freq: f64,
    materials: Option<&[crate::materials::Material]>,
    pml_regions: Option<&[crate::materials::PmlRegion]>,
) -> Result<ReducedSystem, String> {
    let c0 = crate::constants::C0;
    let k0 = 2.0 * PI * freq / c0;
    let n_field = basis.n_field;
    let n_tets = mesh.n_tets();

    // Step 1: Build material tensors (exact port of assembler.py lines 280-303)
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

    // Step 2: K = E - k0² * B (defer CSR construction — build faer triplets directly later)
    let t1 = web_time::Instant::now();
    let k0_sq = C64::from(k0 * k0);

    // Step 3: PEC DOFs — exact port of assembler.py lines 356-373
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

    // Step 4: Robin BC — exact port of assembler.py lines 380-413
    // Uses EMerge's flat array mechanism: Bempty + compute_bc_entries + generate_csr
    let ac_base = AreaCoeffCache::new();
    let gauss_points = gaus_quad_tri(4);

    // Bempty = field.empty_tri_matrix()
    let mut bempty = basis.empty_tri_matrix();

    for (pi, (port, tri_ids)) in ports.iter().zip(port_tri_indices.iter()).enumerate() {
        let gamma = port.get_gamma(k0);

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

    // Step 5: Port excitation vectors — only for driven ports
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
                port.get_uinc(x, y, z, k0)
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

    // Map each driven port's full-length excitation to free-DOF indexing.
    let rhs: Vec<Vec<C64>> = port_vectors
        .iter()
        .map(|bvec| free_dofs.iter().map(|&d| bvec[d]).collect())
        .collect();
    let _ = &driven_port_indices; // kept for parity with the solve-side logging

    Ok(ReducedSystem {
        n_free,
        n_field,
        rows: coo_rows,
        cols: coo_cols,
        vals: coo_vals,
        free_dofs,
        rhs,
    })
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
    frequency_sweep_with_pml(mesh, basis, ports, port_tri_indices, pec_tri_indices, frequencies, materials, None, 1)
}

/// `n_subdomains > 1` switches the per-frequency solve from a monolithic
/// factorization to primal Schur-complement domain decomposition (issue #12):
/// the mesh is partitioned once (frequency-independent), and each frequency is
/// solved by `schur::schur_solve`, bounding peak memory by the largest
/// subdomain interior block rather than the global factor.
pub fn frequency_sweep_with_pml(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    ports: &[&dyn Port],
    port_tri_indices: &[&[usize]],
    pec_tri_indices: &[usize],
    frequencies: &[f64],
    materials: Option<&[crate::materials::Material]>,
    pml_regions: Option<&[crate::materials::PmlRegion]>,
    n_subdomains: usize,
) -> Result<Vec<SolveResult>, String> {
    // Detect if any material is frequency-dependent — if so, K must be rebuilt every frequency
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

    // Schur DD: partition the mesh once (frequency-independent). `None` keeps
    // the monolithic path.
    let schur_class: Option<Vec<crate::solver::schur::DofClass>> = if n_subdomains > 1 {
        use crate::solver::schur;
        let centroids = schur::tet_centroids(mesh);
        let part = schur::partition_rcb(&centroids, n_subdomains);
        let cls = schur::classify_free_dofs(mesh, basis, &free_dofs, &part);
        let n_if = cls.iter().filter(|c| matches!(c, schur::DofClass::Interface)).count();
        eprintln!(
            "  Schur DD: {} subdomains, {} interface DOFs ({:.1}% of {} free)",
            n_subdomains, n_if, 100.0 * n_if as f64 / n_free as f64, n_free);
        Some(cls)
    } else {
        None
    };

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

    // Pick backend once for the whole sweep — symbolic factorisation is
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
        let k0 = 2.0 * PI * freq / crate::constants::C0;
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

        // Robin BC (γ frequency-dependent) — reuse bempty buffer
        bempty.fill(C64::new(0.0, 0.0));
        for (_, (port, tri_ids)) in ports.iter().zip(port_tri_indices.iter()).enumerate() {
            let gamma = port.get_gamma(k0);
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
                            verts[0][2]*l1+verts[1][2]*l2+verts[2][2]*l3, k0)
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

        // Free-indexed RHS (shared by both solve paths).
        let rhs_free: Vec<Vec<C64>> = port_bvecs
            .iter()
            .map(|bvec| free_dofs.iter().map(|&d| bvec[d]).collect())
            .collect();

        // Solve: Schur DD when partitioned, else the monolithic factorization
        // (symbolic once via `factorize`, then `refactorize` reusing the
        // sparsity pattern across the sweep).
        let sols_free: Vec<Vec<C64>> = if let Some(cls) = &schur_class {
            crate::solver::schur::schur_solve(
                n_free, &coo_rows, &coo_cols, &coo_vals, &rhs_free, cls,
                crate::solver::SolverChoice::from_env(),
            )?
        } else {
            if first_factor {
                solver.factorize(n_free, &coo_rows, &coo_cols, &coo_vals)?;
                first_factor = false;
            } else {
                solver.refactorize(n_free, &coo_rows, &coo_cols, &coo_vals)?;
            }
            rhs_free.iter().map(|b| solver.solve(b)).collect::<Result<_, _>>()?
        };

        let mut solutions = Vec::new();
        for x_free in &sols_free {
            let mut x_full = vec![C64::new(0.0, 0.0); n_field];
            for (fi_d, &d) in free_dofs.iter().enumerate() {
                x_full[d] = x_free[fi_d];
            }
            solutions.push(x_full);
        }

        eprintln!(
            "  f={:>8.4e} Hz [{:>2}/{:>2}]  {:>6.1}ms  {}",
            freq, fi + 1, frequencies.len(), t_freq.elapsed().as_secs_f64() * 1e3,
            if schur_class.is_some() { "schur-dd" } else { solver.name() },
        );
        results.push(SolveResult { solutions, n_field });
    }

    Ok(results)
}