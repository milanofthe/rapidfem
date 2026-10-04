// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

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
use crate::basis::NedelecBasis;
use crate::port::Port;
use crate::tet_assembly::assemble_global_matrices;
use crate::tri_assembly::{tri_force, tri_stiff, tri_stiff_tensor};

/// The surface element's DOF owners for triangle `ti`, from the entity orders the
/// minimum rule produced. The element and the DOF map read the same list, so they
/// agree on the size by construction.
fn tri_owners(basis: &NedelecBasis, mesh: &Mesh, ti: usize) -> Vec<crate::dofmap::DofOwner> {
    crate::basis::tri_dof_owners(&basis.orders.tri_edge_orders(mesh, ti), basis.orders.face[ti])
}

pub struct SolveResult {
    pub solutions: Vec<Vec<C64>>,
    pub n_field: usize,
}

/// Relative residual of the preconditioned iteration (S-parameters then
/// agree with the direct solve to about 1e-10).
const SWEEP_ITERATE_TOL: f64 = 1e-10;
/// Cost of one preconditioned iteration per right-hand side, in factorisation
/// flops per factor nonzero. The triangular solves are memory bound where the
/// factorisation runs BLAS-3, so a solve costs far more per flop; measured on
/// WR-90 filters from 50k to 680k DOF (issue #42).
const SWEEP_ITERATION_FLOPS_PER_NNZ: f64 = 35.0;
/// Fewer affordable iterations than this and a sweep point refactors directly.
const SWEEP_ITERATE_MIN_BUDGET: usize = 4;

/// Iteration budget of a sweep point: half an estimated refactorisation,
/// spent on preconditioned iterations. From rslab's a-priori estimate, so the
/// choice between iterating and refactoring (and with it every result) does not
/// depend on the machine or its load.
fn sweep_iteration_budget(solver: &rapidfem_core::linalg::SymmetricSolver<C64>, n_rhs: usize) -> usize {
    let Some((flops, nnz)) = solver.factor_estimate() else { return 0 };
    let per_iteration = SWEEP_ITERATION_FLOPS_PER_NNZ * nnz as f64 * n_rhs.max(1) as f64;
    (0.5 * flops as f64 / per_iteration) as usize
}

/// Frequency sweep: assembles the driven system and solves it for every
/// driven port at every frequency (a single frequency is a sweep of one).
///
/// For frequency-independent materials the E and B matrices are cached.
/// Returns the solutions per frequency (`Vec<SolveResult>`).
pub fn frequency_sweep(
    mesh: &Mesh,
    basis: &NedelecBasis,
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

    // Bulk conductivity also makes εr* frequency-dependent (−j·σ/(ω·ε₀)),
    // but linearly enough to avoid re-assembly: on the cached path the σ term
    // is kept OUT of B and carried in a separate mass matrix B_σ, added per
    // frequency as +j·k₀²/(ω·ε₀)·B_σ, algebraically identical to rebuilding
    // εr*(ω) every frequency. On the dispersive path the full rebuild already
    // evaluates σ at each frequency, so no split is needed there.
    let sigma_split = !materials_dispersive
        && materials.map(|m| m.iter().any(|x| {
            x.cond != 0.0 || x.cond_diag.map(|d| d != [0.0; 3]).unwrap_or(false)
        })).unwrap_or(false);

    // Cache E, B for frequency-independent materials
    let n_tets = mesh.n_tets();
    let pml = pml_regions.map(|p| (p, mesh));
    let (er, ur) = crate::materials::material_tensors(n_tets, materials, frequencies[0], !sigma_split, pml);

    let t0 = web_time::Instant::now();
    let (rows, cols, mut data_e, mut data_b) = assemble_global_matrices(mesh, basis, &er, &ur);

    // B_σ shares the sparsity of B (same mesh/basis); assembled once with the
    // σ tensors as "permittivity" and a unit μr (the curl-curl part of that
    // assembly is discarded).
    let data_bsigma: Option<Vec<C64>> = if sigma_split {
        let mut sig = crate::materials::build_sigma_tensors(n_tets, materials.unwrap_or(&[]));
        if let Some(pml) = pml_regions {
            // PML tets carry stretched εr only; defend against σ-material overlap.
            let zero3x3 = [[C64::new(0.0, 0.0); 3]; 3];
            for region in pml {
                for &ti in &region.tet_indices { sig[ti] = zero3x3; }
            }
        }
        let ur_id = vec![crate::materials::IDENTITY; n_tets];
        let (_, _, _, bs) = assemble_global_matrices(mesh, basis, &sig, &ur_id);
        Some(bs)
    } else {
        None
    };

    eprintln!("  Assembled E,B in {:.1}ms{}", t0.elapsed().as_secs_f64()*1e3,
        if materials_dispersive { "" }
        else if sigma_split { " (cached for sweep, σ mass matrix scaled per frequency)" }
        else { " (cached for sweep)" });

    // PEC DOFs (frequency-independent) and the free-DOF renumbering
    let (free_dofs, dof_to_free) = basis.free_dofs(mesh, pec_tri_indices);
    let n_free = free_dofs.len();
    let is_free = |d: usize| dof_to_free[d] != usize::MAX;

    let gauss_points = crate::quadrature::gaus_quad_tri(4);

    let mut results = Vec::with_capacity(frequencies.len());

    // Precompute non-PEC COO indices for K entries (reused every frequency)
    let k_free_indices: Vec<usize> = (0..rows.len())
        .filter(|&i| is_free(rows[i]) && is_free(cols[i]))
        .collect();
    let k_free_rows: Vec<usize> = k_free_indices.iter().map(|&i| dof_to_free[rows[i]]).collect();
    let k_free_cols: Vec<usize> = k_free_indices.iter().map(|&i| dof_to_free[cols[i]]).collect();

    // Precompute non-PEC Robin indices (reused every frequency), restricted
    // to the PORT triangles: only they carry a Robin term, and the port set
    // is frequency-independent. The COO entries at these indices are then
    // emitted UNCONDITIONALLY per frequency (no skip of exact-zero values),
    // so the sparsity pattern is guaranteed stable across the sweep, which
    // the numeric-only `refactorize` (rslab frozen-pattern factor) relies on.
    let mut robin_free_indices: Vec<usize> = port_tri_indices
        .iter()
        .flat_map(|tri_ids| tri_ids.iter().copied())
        .flat_map(|ti| basis.tri_block(ti)..basis.tri_block(ti + 1))
        .filter(|&idx| {
            let r = basis.tri_rows[idx];
            let c = basis.tri_cols[idx];
            is_free(r) && is_free(c)
        })
        .collect();
    // Ports share no triangles by construction; dedup defends the pattern
    // (and the entry values) against a model that lists one twice.
    robin_free_indices.sort_unstable();
    robin_free_indices.dedup();

    // One solver for the whole sweep: the symbolic factorisation is
    // amortised across frequencies via `solver.refactorize`.
    let mut solver = rapidfem_core::linalg::SymmetricSolver::<C64>::new();
    let mut first_factor = true;
    let mut have_factor = false;

    // COO buffers for the per-frequency system matrix, reused across the
    // sweep. Capacity covers the K block plus the Robin upper bound.
    let coo_cap = k_free_indices.len() + robin_free_indices.len();
    let mut coo_rows: Vec<usize> = Vec::with_capacity(coo_cap);
    let mut coo_cols: Vec<usize> = Vec::with_capacity(coo_cap);
    let mut coo_vals: Vec<C64> = Vec::with_capacity(coo_cap);
    let mut bempty = basis.empty_tri_matrix();

    let dump = crate::dump::target();
    for (fi, &freq) in frequencies.iter().enumerate() {
        let t_freq = web_time::Instant::now();
        let exc = crate::excitation::Excitation::new(freq, mesh.l0);
        let k0 = exc.k0;
        let k0_sq = C64::from(k0 * k0);
        let n_field = basis.n_field;

        // Rebuild element matrices when materials are frequency-dependent
        if materials_dispersive && fi > 0 {
            let (er_f, ur_f) = crate::materials::material_tensors(n_tets, materials, freq, true, pml);
            let (_, _, de, db) = assemble_global_matrices(mesh, basis, &er_f, &ur_f);
            data_e = de;
            data_b = db;
        }

        // Robin BC (γ frequency-dependent), reuse bempty buffer
        bempty.fill(C64::new(0.0, 0.0));
        for (port, tri_ids) in ports.iter().zip(port_tri_indices.iter()) {
            let gamma = port.get_gamma(&exc);
            for (k, &ti) in tri_ids.iter().enumerate() {
                let tri = &mesh.tris[ti];
                let verts = [mesh.nodes[tri[0]], mesh.nodes[tri[1]], mesh.nodes[tri[2]]];
                let owners = tri_owners(basis, mesh, ti);
                let bsub = match port.tri_tensor(&exc, k) {
                    Some(tensor) => tri_stiff_tensor(&owners, &verts, &tensor),
                    None => tri_stiff(&owners, &verts, gamma),
                };
                let n = basis.tri_dofs(ti).len();
                let p = basis.tri_block(ti);
                for ii in 0..n { for jj in 0..n { bempty[p + ii*n + jj] += bsub[ii*n + jj]; } }
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
                    let owners = tri_owners(basis, mesh, ti);
                    let b_tri = tri_force(&owners, &verts, &u_at_qp, &gauss_points);
                    let dofs = basis.tri_dofs(ti);
                    for (i, &d) in dofs.iter().enumerate() { bvec[d] += b_tri[i]; }
                }
            }
            port_bvecs.push(bvec);
        }

        // Build the system matrix COO: K = (E - k0^2*B) + Robin, straight
        // into the solver's COO buffers, reusing the allocation.
        coo_rows.clear();
        coo_cols.clear();
        coo_vals.clear();

        // Per-frequency σ term: −k₀²·(−j·σ/(ω·ε₀)) = +j·k₀²/(ω·ε₀)·B_σ.
        let sigma_scale = C64::new(0.0, 1.0) * k0_sq
            / C64::from(2.0 * std::f64::consts::PI * freq * crate::constants::EPS0);
        if let Some(bs) = &data_bsigma {
            for (ti, &orig_i) in k_free_indices.iter().enumerate() {
                coo_rows.push(k_free_rows[ti]);
                coo_cols.push(k_free_cols[ti]);
                coo_vals.push(data_e[orig_i] - k0_sq * data_b[orig_i] + sigma_scale * bs[orig_i]);
            }
        } else {
            for (ti, &orig_i) in k_free_indices.iter().enumerate() {
                coo_rows.push(k_free_rows[ti]);
                coo_cols.push(k_free_cols[ti]);
                coo_vals.push(data_e[orig_i] - k0_sq * data_b[orig_i]);
            }
        }
        // Unconditional emit (zeros included): the pattern must not drift
        // between frequencies, see `robin_free_indices` above.
        for &idx in &robin_free_indices {
            coo_rows.push(dof_to_free[basis.tri_rows[idx]]);
            coo_cols.push(dof_to_free[basis.tri_cols[idx]]);
            coo_vals.push(bempty[idx]);
        }

        // Port right-hand sides on the free DOFs.
        let b_frees: Vec<Vec<C64>> = port_bvecs.iter()
            .map(|bvec| free_dofs.iter()
                .map(|&d| bvec[d]).collect())
            .collect();
        let asm_ms = t_freq.elapsed().as_secs_f64() * 1e3;
        if let Some(target) = &dump {
            crate::dump::write_system(target, fi, 2.0 * std::f64::consts::PI * freq / 299_792_458.0, n_free, &coo_rows, &coo_cols, &coo_vals, &b_frees)?;
        }

        // Neighbouring frequencies: COCG preconditioned with the factorisation
        // of an earlier frequency (same pattern, nearby values) instead of a
        // refactorisation, when a factorisation is expensive enough to be
        // worth avoiding. A 681k-DOF iris filter swept over 21 points ran
        // 2.6x faster this way, S-parameters equal to 5e-11. A point that does
        // not converge within its budget refactors and becomes the reference.
        solver.expect_rhs(b_frees.len());
        let budget = sweep_iteration_budget(&solver, b_frees.len());
        let nearby = if have_factor && budget >= SWEEP_ITERATE_MIN_BUDGET {
            solver.solve_nearby(n_free, &coo_rows, &coo_cols, &coo_vals, &b_frees,
                                SWEEP_ITERATE_TOL, budget)
        } else {
            None
        };
        let (x_frees, how) = if let Some((xs, it)) = nearby {
            (xs, format!("cocg {it} it on the last factor"))
        } else {
            // Factor (symbolic once via `factorize`, then `refactorize` per
            // freq reusing the sparsity pattern) and solve all ports batched.
            if first_factor {
                solver.factorize(n_free, &coo_rows, &coo_cols, &coo_vals)?;
                first_factor = false;
            } else {
                solver.refactorize(n_free, &coo_rows, &coo_cols, &coo_vals)?;
            }
            have_factor = true;
            (solver.solve_many(&b_frees)?, solver.name().to_string())
        };
        let mut solutions = Vec::new();
        for x_free in x_frees {
            let mut x_full = vec![C64::new(0.0, 0.0); n_field];
            for (fi_d, &d) in free_dofs.iter().enumerate() {
                x_full[d] = x_free[fi_d];
            }
            solutions.push(x_full);
        }

        eprintln!(
            "  f={:>8.4e} Hz [{:>2}/{:>2}]  {:>6.1}ms  {} (assembly {asm_ms:.0}ms)",
            freq, fi + 1, frequencies.len(), t_freq.elapsed().as_secs_f64() * 1e3,
            how,
        );
        results.push(SolveResult { solutions, n_field });
        if let Some(cb) = on_solve.as_deref_mut()
            && !cb(fi, freq, results.last().unwrap()) {
                eprintln!(
                    "  sweep stopped early after frequency {}/{} (interrupt)",
                    fi + 1, frequencies.len(),
                );
                break;
            }
    }

    Ok(results)
}
