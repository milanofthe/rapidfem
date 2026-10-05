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

/// The driven system of a sweep: everything that does not change with the
/// frequency (the volume matrices E, B and B_σ, the free-DOF numbering, the
/// port triangles), and [`DrivenSystem::assemble`] for the system matrix and
/// the port right-hand sides at one frequency. The full sweep and the adaptive
/// sweep ([`crate::adaptive`]) build their systems from the same pieces.
pub struct DrivenSystem<'a> {
    pub mesh: &'a Mesh,
    pub basis: &'a NedelecBasis,
    pub ports: &'a [&'a dyn Port],
    pub port_tri_indices: &'a [&'a [usize]],
    /// The dispersive materials: B holds them with a unit εr (times their
    /// 1 − j·tanδ), scaled per frequency by their εr(ω).
    pub dispersive: Vec<&'a crate::materials::Material>,
    /// Per volume triplet on free DOFs (parallel to `k_free_indices`): 0, or
    /// 1 + the index of the dispersive material of its tet. Empty without
    /// dispersive materials.
    k_group: Vec<u16>,
    data_e: Vec<C64>,
    data_b: Vec<C64>,
    /// The bulk-conductivity mass matrix (B's pattern), scaled per frequency
    /// by [`DrivenSystem::sigma_scale`].
    pub data_bsigma: Option<Vec<C64>>,
    pub free_dofs: Vec<usize>,
    pub dof_to_free: Vec<usize>,
    /// The volume triplets on free DOFs: their index into E/B and their
    /// free row and column.
    pub k_free_indices: Vec<usize>,
    pub k_free_rows: Vec<usize>,
    pub k_free_cols: Vec<usize>,
    /// The surface triplets of the port triangles on free DOFs.
    robin_free_indices: Vec<usize>,
    gauss_points: Vec<[f64; 4]>,
    bempty: Vec<C64>,
}

impl<'a> DrivenSystem<'a> {
    /// Assembles the frequency-independent pieces, E and B at `f_first`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mesh: &'a Mesh,
        basis: &'a NedelecBasis,
        ports: &'a [&'a dyn Port],
        port_tri_indices: &'a [&'a [usize]],
        pec_tri_indices: &[usize],
        f_first: f64,
        materials: Option<&'a [crate::materials::Material]>,
        pml_regions: Option<&'a [crate::materials::PmlRegion]>,
    ) -> DrivenSystem<'a> {
        // A dispersive material's εr(ω) is one complex scalar on its tets
        // (isotropic, times 1 − j·tanδ), so its part of B is εr(ω) times its
        // mass matrix at unit εr: assembled once, scaled per frequency.
        let dispersive: Vec<&crate::materials::Material> = materials
            .map(|m| m.iter().filter(|x| x.dispersion.is_dispersive()).collect())
            .unwrap_or_default();

        // Bulk conductivity also makes εr* frequency-dependent (−j·σ/(ω·ε₀)),
        // but linearly enough to avoid re-assembly: the σ term is kept OUT of B
        // and carried in a separate mass matrix B_σ, added per frequency as
        // +j·k₀²/(ω·ε₀)·B_σ, algebraically identical to rebuilding εr*(ω) every
        // frequency.
        let sigma_split = materials.map(|m| m.iter().any(|x| {
                x.cond != 0.0 || x.cond_diag.map(|d| d != [0.0; 3]).unwrap_or(false)
            })).unwrap_or(false);

        let n_tets = mesh.n_tets();
        let pml = pml_regions.map(|p| (p, mesh));
        let (mut er, ur) = crate::materials::material_tensors(n_tets, materials, f_first, !sigma_split, pml);
        let mut tet_group = vec![0u16; if dispersive.is_empty() { 0 } else { n_tets }];
        if !dispersive.is_empty() {
            // PML tets keep their stretched tensors
            let mut in_pml = vec![false; n_tets];
            for region in pml_regions.unwrap_or(&[]) {
                for &ti in &region.tet_indices { in_pml[ti] = true; }
            }
            for (g, m) in dispersive.iter().enumerate() {
                let unit = C64::new(1.0, -m.tand);
                let zero = C64::new(0.0, 0.0);
                for &ti in m.tet_indices.iter().filter(|&&ti| !in_pml[ti]) {
                    er[ti] = [[unit, zero, zero], [zero, unit, zero], [zero, zero, unit]];
                    tet_group[ti] = g as u16 + 1;
                }
            }
        }

        let t0 = web_time::Instant::now();
        let (rows, cols, data_e, data_b) = assemble_global_matrices(mesh, basis, &er, &ur);

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

        eprintln!("  Assembled E,B in {:.1}ms (cached for sweep{}{})", t0.elapsed().as_secs_f64()*1e3,
            if sigma_split { ", σ mass matrix scaled per frequency" } else { "" },
            if dispersive.is_empty() { "" } else { ", dispersive mass matrices scaled per frequency" });

        // PEC DOFs (frequency-independent) and the free-DOF renumbering
        let (free_dofs, dof_to_free) = basis.free_dofs(mesh, pec_tri_indices);
        let is_free = |d: usize| dof_to_free[d] != usize::MAX;

        // Precompute non-PEC COO indices for K entries (reused every frequency)
        let k_free_indices: Vec<usize> = (0..rows.len())
            .filter(|&i| is_free(rows[i]) && is_free(cols[i]))
            .collect();
        let k_free_rows: Vec<usize> = k_free_indices.iter().map(|&i| dof_to_free[rows[i]]).collect();
        let k_free_cols: Vec<usize> = k_free_indices.iter().map(|&i| dof_to_free[cols[i]]).collect();
        // the tet of a triplet: the element blocks follow the tets in order
        let offsets = basis.tet_nnz_offsets();
        let k_group: Vec<u16> = if tet_group.is_empty() {
            Vec::new()
        } else {
            k_free_indices.iter().map(|&i| tet_group[offsets.partition_point(|&o| o <= i) - 1]).collect()
        };

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

        DrivenSystem {
            mesh,
            basis,
            ports,
            port_tri_indices,
            dispersive,
            k_group,
            data_e,
            data_b,
            data_bsigma,
            free_dofs,
            dof_to_free,
            k_free_indices,
            k_free_rows,
            k_free_cols,
            robin_free_indices,
            gauss_points: crate::quadrature::gaus_quad_tri(4),
            bempty: basis.empty_tri_matrix(),
        }
    }

    pub fn n_free(&self) -> usize {
        self.free_dofs.len()
    }

    /// The scale of each part of B at `freq`: 1 for the fixed part, then
    /// εr(ω) of each dispersive material.
    pub fn b_scales(&self, freq: f64) -> Vec<C64> {
        std::iter::once(C64::new(1.0, 0.0))
            .chain(self.dispersive.iter().map(|m| m.dispersion.evaluate(m.er, freq)))
            .collect()
    }

    /// E·x, the parts of B·x (see [`DrivenSystem::b_scales`]) and B_σ·x on
    /// the free DOFs for a real `x` (the reduced basis of the adaptive
    /// sweep), straight from the cached triplets.
    pub fn volume_apply(&self, x: &[f64]) -> (Vec<C64>, Vec<Vec<C64>>, Option<Vec<C64>>) {
        let n = self.n_free();
        let zero = C64::new(0.0, 0.0);
        let mut e = vec![zero; n];
        let mut b = vec![vec![zero; n]; 1 + self.dispersive.len()];
        let mut bs = self.data_bsigma.as_ref().map(|_| vec![zero; n]);
        for (t, &i) in self.k_free_indices.iter().enumerate() {
            let (r, xc) = (self.k_free_rows[t], x[self.k_free_cols[t]]);
            if xc == 0.0 {
                continue;
            }
            e[r] += self.data_e[i] * xc;
            let g = self.k_group.get(t).map_or(0, |&g| g as usize);
            b[g][r] += self.data_b[i] * xc;
            if let (Some(bs), Some(d)) = (bs.as_mut(), self.data_bsigma.as_ref()) {
                bs[r] += d[i] * xc;
            }
        }
        (e, b, bs)
    }

    /// The factor of B_σ at `exc`: −k₀²·(−j·σ/(ω·ε₀)) = +j·k₀²/(ω·ε₀).
    pub fn sigma_scale(exc: &crate::excitation::Excitation) -> C64 {
        C64::new(0.0, 1.0) * C64::from(exc.k0 * exc.k0)
            / C64::from(exc.omega * crate::constants::EPS0)
    }

    /// The surface triplets of port `p`'s triangles on free DOFs with their
    /// values at `exc`: (free row, free col, value). With `unit` the scalar
    /// Robin coefficient is 1 (the port's γ factored out).
    pub fn port_robin(&self, p: usize, exc: &crate::excitation::Excitation, unit: bool) -> Vec<(usize, usize, C64)> {
        let port = self.ports[p];
        let gamma = if unit { C64::new(1.0, 0.0) } else { port.get_gamma(exc) };
        let mut out = Vec::new();
        for (k, &ti) in self.port_tri_indices[p].iter().enumerate() {
            let (verts, owners) = self.tri(ti);
            let sub = match (unit, port.tri_tensor(exc, k)) {
                (false, Some(tensor)) => tri_stiff_tensor(&owners, &verts, &tensor),
                _ => tri_stiff(&owners, &verts, gamma),
            };
            let start = self.basis.tri_block(ti);
            for (j, v) in sub.into_iter().enumerate() {
                let (r, c) = (self.dof_to_free[self.basis.tri_rows[start + j]], self.dof_to_free[self.basis.tri_cols[start + j]]);
                if r != usize::MAX && c != usize::MAX {
                    out.push((r, c, v));
                }
            }
        }
        out
    }

    fn tri(&self, ti: usize) -> ([[f64; 3]; 3], Vec<crate::dofmap::DofOwner>) {
        let tri = &self.mesh.tris[ti];
        let verts = [self.mesh.nodes[tri[0]], self.mesh.nodes[tri[1]], self.mesh.nodes[tri[2]]];
        (verts, tri_owners(self.basis, self.mesh, ti))
    }

    /// The right-hand side of every driven port at `exc`, on the free DOFs.
    pub fn rhs(&self, exc: &crate::excitation::Excitation) -> Vec<Vec<C64>> {
        let n_field = self.basis.n_field;
        let gauss_points = &self.gauss_points;
        let mut port_bvecs: Vec<Vec<C64>> = Vec::new();
        for (port, tri_ids) in self.ports.iter().zip(self.port_tri_indices.iter()) {
            if !port.is_driven() { continue; }
            let mut bvec = vec![C64::new(0.0, 0.0); n_field];
            for &ti in *tri_ids {
                let (verts, owners) = self.tri(ti);
                let u_at_qp: Vec<[C64; 3]> = gauss_points.iter()
                    .filter_map(|qp| {
                        let (l1,l2,l3) = (qp[1],qp[2],qp[3]);
                        port.get_uinc(
                            verts[0][0]*l1+verts[1][0]*l2+verts[2][0]*l3,
                            verts[0][1]*l1+verts[1][1]*l2+verts[2][1]*l3,
                            verts[0][2]*l1+verts[1][2]*l2+verts[2][2]*l3, exc)
                    }).collect();
                if u_at_qp.len() == gauss_points.len() {
                    let b_tri = tri_force(&owners, &verts, &u_at_qp, gauss_points);
                    let dofs = self.basis.tri_dofs(ti);
                    for (i, &d) in dofs.iter().enumerate() { bvec[d] += b_tri[i]; }
                }
            }
            port_bvecs.push(bvec);
        }
        port_bvecs.iter()
            .map(|bvec| self.free_dofs.iter().map(|&d| bvec[d]).collect())
            .collect()
    }

    /// The system matrix at `freq` into the COO buffers (K = E − k₀²·B, the
    /// σ term, the Robin terms; the pattern is the same at every frequency)
    /// and the port right-hand sides.
    pub fn assemble(
        &mut self,
        freq: f64,
        coo_rows: &mut Vec<usize>,
        coo_cols: &mut Vec<usize>,
        coo_vals: &mut Vec<C64>,
    ) -> Vec<Vec<C64>> {
        let exc = crate::excitation::Excitation::new(freq, self.mesh.l0);
        let k0_sq = C64::from(exc.k0 * exc.k0);
        // −k₀² times each part of B
        let b_coef: Vec<C64> = self.b_scales(freq).into_iter().map(|s| -k0_sq * s).collect();
        let coef = |t: usize| b_coef[self.k_group.get(t).map_or(0, |&g| g as usize)];

        // Robin BC (γ frequency-dependent), reuse bempty buffer
        self.bempty.fill(C64::new(0.0, 0.0));
        for (port, tri_ids) in self.ports.iter().zip(self.port_tri_indices.iter()) {
            let gamma = port.get_gamma(&exc);
            for (k, &ti) in tri_ids.iter().enumerate() {
                let (verts, owners) = self.tri(ti);
                let bsub = match port.tri_tensor(&exc, k) {
                    Some(tensor) => tri_stiff_tensor(&owners, &verts, &tensor),
                    None => tri_stiff(&owners, &verts, gamma),
                };
                let n = self.basis.tri_dofs(ti).len();
                let p = self.basis.tri_block(ti);
                for ii in 0..n { for jj in 0..n { self.bempty[p + ii*n + jj] += bsub[ii*n + jj]; } }
            }
        }

        // Build the system matrix COO: K = (E - k0^2*B) + Robin, straight
        // into the solver's COO buffers, reusing the allocation.
        coo_rows.clear();
        coo_cols.clear();
        coo_vals.clear();
        let (data_e, data_b) = (&self.data_e, &self.data_b);
        if let Some(bs) = &self.data_bsigma {
            let sigma_scale = Self::sigma_scale(&exc);
            for (ti, &orig_i) in self.k_free_indices.iter().enumerate() {
                coo_rows.push(self.k_free_rows[ti]);
                coo_cols.push(self.k_free_cols[ti]);
                coo_vals.push(data_e[orig_i] + coef(ti) * data_b[orig_i] + sigma_scale * bs[orig_i]);
            }
        } else {
            for (ti, &orig_i) in self.k_free_indices.iter().enumerate() {
                coo_rows.push(self.k_free_rows[ti]);
                coo_cols.push(self.k_free_cols[ti]);
                coo_vals.push(data_e[orig_i] + coef(ti) * data_b[orig_i]);
            }
        }
        // Unconditional emit (zeros included): the pattern must not drift
        // between frequencies, see `robin_free_indices` above.
        for &idx in &self.robin_free_indices {
            coo_rows.push(self.dof_to_free[self.basis.tri_rows[idx]]);
            coo_cols.push(self.dof_to_free[self.basis.tri_cols[idx]]);
            coo_vals.push(self.bempty[idx]);
        }
        self.rhs(&exc)
    }

    /// A free-DOF solution spread onto the full field (zero on PEC DOFs).
    pub fn to_field(&self, x_free: &[C64]) -> Vec<C64> {
        let mut x_full = vec![C64::new(0.0, 0.0); self.basis.n_field];
        for (fi_d, &d) in self.free_dofs.iter().enumerate() {
            x_full[d] = x_free[fi_d];
        }
        x_full
    }
}

/// Frequency sweep: assembles the driven system and solves it for every
/// driven port at every frequency (a single frequency is a sweep of one).
///
/// For frequency-independent materials the E and B matrices are cached.
/// Returns the solutions per frequency (`Vec<SolveResult>`).
pub fn frequency_sweep(
    sys: &mut DrivenSystem,
    frequencies: &[f64],
    // Optional per-frequency hook, called after each frequency's solve with
    // (freq_idx, freq_hz, &SolveResult). Lets a caller stream partial results
    // (e.g. progressive S-parameters in the UI) without changing the output.
    // Returns `false` to stop the sweep early (e.g. on a user interrupt); the
    // frequencies solved so far are returned.
    mut on_solve: Option<&mut dyn FnMut(usize, f64, &SolveResult) -> bool>,
) -> Result<Vec<SolveResult>, String> {
    let n_free = sys.n_free();
    let n_field = sys.basis.n_field;
    let mut results = Vec::with_capacity(frequencies.len());

    // One solver for the whole sweep: the symbolic factorisation is
    // amortised across frequencies via `solver.refactorize`.
    let mut solver = rapidfem_core::linalg::SymmetricSolver::<C64>::new();
    let mut first_factor = true;
    let mut have_factor = false;

    // COO buffers for the per-frequency system matrix, reused across the
    // sweep.
    let mut coo_rows: Vec<usize> = Vec::new();
    let mut coo_cols: Vec<usize> = Vec::new();
    let mut coo_vals: Vec<C64> = Vec::new();

    let dump = crate::dump::target();
    for (fi, &freq) in frequencies.iter().enumerate() {
        let t_freq = web_time::Instant::now();
        let b_frees = sys.assemble(freq, &mut coo_rows, &mut coo_cols, &mut coo_vals);
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
        let solutions = x_frees.iter().map(|x| sys.to_field(x)).collect();

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
