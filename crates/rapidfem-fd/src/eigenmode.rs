// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Eigenmode solver: the resonances of a cavity, and their Q factors.
//!
//! Solves the pencil `E·x = λ·B·x` for the eigenvalues nearest a target, by
//! shift-invert Lanczos. `λ = k₀²`, so `f = c₀·√λ / 2π`; a lossy material makes
//! `λ` complex and its imaginary part is the Q.
//!
//! # The inner product is not optional
//!
//! Shift-invert replaces the pencil by the single operator
//!
//!   OP = (E − σB)⁻¹ · B
//!
//! whose eigenvalues `μ` cluster the wanted `λ` away from the rest: `λ = σ + 1/μ`.
//! Lanczos then needs OP to be **self-adjoint**, and OP is not self-adjoint in the
//! Euclidean inner product. It is self-adjoint in the one induced by `B`:
//!
//!   ⟨x, y⟩_B = xᵀ·B·y      (bilinear, NOT conjugated: E and B are complex
//!                           SYMMETRIC under loss, not Hermitian)
//!
//! Run it in the Euclidean form instead — `α = vᵀw`, `β = ‖w‖₂`, and both the
//! reorthogonalisation and the normalisation to match — and it is a three-term
//! recurrence for an operator that has no three-term recurrence. The tridiagonal
//! matrix it produces is not similar to anything, its Ritz values are not
//! eigenvalues, and its Ritz vectors solve nothing.
//!
//! That is what this solver used to do, and it is why it reported ghost modes below a
//! cavity's fundamental and eigenvectors with an O(1) eigenpair residual. (It was not
//! for want of reorthogonalisation: it reorthogonalised against the whole basis, just
//! in the wrong form.) All of it is now checked against a densely computed spectrum in
//! `tests/eigensolver_test.rs`.
//!
//! # The residual test is not optional either
//!
//! A Krylov space of dimension m cannot resolve more than a handful of eigenvalues
//! well, and the rest of its Ritz values are junk. There is no way to tell which is
//! which except to ask: a Ritz pair `(λ, x)` is a mode only if it actually satisfies
//! the equation. So every candidate is checked,
//!
//!   ‖E·x − λ·B·x‖ / ‖λ·B·x‖ < EIGEN_RESIDUAL_TOL,
//!
//! and the ones that do not pass are not returned. Two sparse matrix-vector products
//! per candidate, against a factorisation that cost far more, and it makes a ghost
//! impossible rather than unlikely.
//!
//! # Which modes are reachable at all: `f < √2 · f_target`
//!
//! This is a property of the problem, not a shortcoming of the implementation, and
//! it decides how the solver must be *used*.
//!
//! The curl operator has an enormous null space — the discrete gradients, one per
//! interior node and, at order 2, one per interior edge. On a 400 k-DOF mesh that is
//! some 200 000 eigenvalues sitting at exactly `λ = 0`. Under shift-invert they all
//! land on the single point
//!
//!   μ_kernel = 1/(0 − σ) = −1/σ,   so   |μ_kernel| = 1/σ.
//!
//! Lanczos converges on the eigenvalues of LARGEST magnitude first. A mode at `λ` has
//! `|μ| = 1/|λ − σ|`, so it is ahead of that vast degenerate cluster only when
//!
//!   |λ − σ| < σ,   i.e.   0 < λ < 2σ,   i.e.   f < √2 · f_target.
//!
//! A mode outside that band sits BEHIND ~200 000 identical Ritz values, and no
//! Krylov space of any affordable size will reach it. Measured on a 410 k-DOF cavity:
//! targeting the fundamental at 8.24 GHz and asking for four modes gives ONE, after
//! 92 Lanczos steps; moving the target to 12 GHz — which puts all four inside the
//! band — gives five, in 30 steps and a third of the time.
//!
//! So: **put the target in the middle of the frequency range you care about**, not on
//! the lowest mode of it. The solver says so when it cannot deliver.
//!
//! Removing the restriction means deflating the gradient space explicitly (projecting
//! the Krylov space onto the complement of the range of the discrete gradient
//! matrix). That is a real addition and is not done here.

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::tet_assembly_r2::assemble_global_matrices;
use crate::constants::*;
use std::collections::HashSet;

/// A Ritz pair is accepted as a mode only if its relative eigenpair residual is
/// below this. Far tighter than any physical tolerance, because a converged mode
/// reaches machine precision and an unconverged one is off by orders of magnitude:
/// there is nothing in between to calibrate against.
pub const EIGEN_RESIDUAL_TOL: f64 = 1e-8;

/// The curl operator's kernel — the discrete gradients — sits at `λ = 0`. Those are
/// genuine eigenpairs and would sail through the residual test, but they are static
/// fields, not resonances. Anything with `|λ|` below this fraction of the shift is
/// one of them.
const STATIC_MODE_FLOOR: f64 = 1e-6;

pub struct Eigenmode {
    pub frequency: C64,
    pub q_factor: f64,
    pub eigenvalue: C64,
    pub field: Vec<C64>,
    /// The relative eigenpair residual `‖E·x − λ·B·x‖ / ‖λ·B·x‖` of this mode.
    ///
    /// Every returned mode has passed [`EIGEN_RESIDUAL_TOL`], so this is small by
    /// construction. It is reported anyway: it is the one number that says whether
    /// the field can be trusted, and a caller that does not look at it should at
    /// least be able to.
    pub residual: f64,
}

/// A sparse matrix as COO triplets over the free DOFs.
///
/// The assembly emits the FULL n×n block of every element, so both `(i,j)` and
/// `(j,i)` are already present. A matvec must therefore NOT add the transpose — the
/// old one did, and silently computed `(2B − diag B)·x` instead of `B·x`.
struct Coo {
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<C64>,
    n: usize,
}

impl Coo {
    fn matvec(&self, x: &[C64]) -> Vec<C64> {
        let mut y = vec![C64::new(0.0, 0.0); self.n];
        for k in 0..self.rows.len() {
            y[self.rows[k]] += self.vals[k] * x[self.cols[k]];
        }
        y
    }
}

/// The bilinear form `xᵀ·y`. Not the Hermitian one: E and B are complex SYMMETRIC
/// when the materials are lossy, and it is the bilinear form they are symmetric in.
#[inline]
fn bilinear(x: &[C64], y: &[C64]) -> C64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

#[inline]
fn norm2(x: &[C64]) -> f64 {
    x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
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
    eprintln!("  Eigenmode: assembled E,B in {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);

    // PEC DOFs, and the free-DOF renumbering.
    let mut pec_ids: HashSet<usize> = HashSet::new();
    for &ti in pec_tri_indices {
        for &ei in &mesh.tri_to_edge[ti] {
            for &d in basis.edge_dofs(ei) {
                pec_ids.insert(d);
            }
        }
        for &d in basis.tri_dofs(ti) {
            pec_ids.insert(d);
        }
    }
    let free_dofs: Vec<usize> = (0..n_field).filter(|d| !pec_ids.contains(d)).collect();
    let n_free = free_dofs.len();
    if n_free == 0 {
        return Err("every DOF is constrained: nothing to solve".to_string());
    }
    let mut dof_to_free = vec![usize::MAX; n_field];
    for (fi, &d) in free_dofs.iter().enumerate() {
        dof_to_free[d] = fi;
    }
    eprintln!("  Eigenmode: {} free DOFs, target = {:.4e} Hz", n_free, target_freq);

    // The shift σ = k₀(target)².
    let k0_target = crate::excitation::Excitation::new(target_freq, mesh.l0).k0;
    let sigma = C64::from(k0_target * k0_target);

    // E, B and (E − σB) on the free DOFs, sharing one index list.
    let mut idx_r: Vec<usize> = Vec::new();
    let mut idx_c: Vec<usize> = Vec::new();
    let mut e_vals: Vec<C64> = Vec::new();
    let mut b_vals: Vec<C64> = Vec::new();
    let mut shift_vals: Vec<C64> = Vec::new();
    for i in 0..rows.len() {
        let (r, c) = (rows[i], cols[i]);
        if pec_ids.contains(&r) || pec_ids.contains(&c) {
            continue;
        }
        idx_r.push(dof_to_free[r]);
        idx_c.push(dof_to_free[c]);
        e_vals.push(data_e[i]);
        b_vals.push(data_b[i]);
        shift_vals.push(data_e[i] - sigma * data_b[i]);
    }
    let e_mat = Coo { rows: idx_r.clone(), cols: idx_c.clone(), vals: e_vals, n: n_free };
    let b_mat = Coo { rows: idx_r.clone(), cols: idx_c.clone(), vals: b_vals, n: n_free };

    let t1 = web_time::Instant::now();
    let mut solver = crate::solver::pick(crate::solver::SolverChoice::from_env());
    solver.factorize(n_free, &idx_r, &idx_c, &shift_vals)?;
    eprintln!(
        "  Eigenmode: {} shift-invert in {:.1}ms",
        solver.name(),
        t1.elapsed().as_secs_f64() * 1e3
    );

    // ---------------------------------------------------------------------
    // Lanczos in the B inner product.
    //
    // OP = (E − σB)⁻¹·B is self-adjoint with respect to ⟨x,y⟩_B = xᵀ·B·y, so that is
    // the product every dot, every normalisation and every reorthogonalisation below
    // is taken in.
    //
    // Only the Lanczos vectors are stored, not B·v alongside them. ⟨v_k, w⟩_B is
    // v_kᵀ·(B·w) just as much as it is (B·v_k)ᵀ·w, and one B·w serves the whole
    // reorthogonalisation sweep — so a single extra sparse matvec per pass buys back
    // half the memory, which on a large problem is the difference between fitting and
    // not.
    // ---------------------------------------------------------------------
    let t2 = web_time::Instant::now();
    // The Krylov space may grow well past the number of modes wanted, because on a
    // large problem the shift-invert spectrum crowds and a short space resolves
    // nothing. It is affordable because the loop STOPS as soon as enough modes have
    // actually converged — the cap is a ceiling, not a plan.
    let m_max = (8 * n_modes + 60).min(n_free).min(400);
    const CHECK_EVERY: usize = 10;
    // A few more candidates than asked for: a mode may converge out of order, and one
    // that is nearly there should not be discarded before the next check.
    let n_cand = n_modes + 4;

    let mut v: Vec<C64> = (0..n_free)
        .map(|i| C64::from(((i * 7 + 13) % 97) as f64 / 97.0 - 0.5))
        .collect();
    let mut bv = b_mat.matvec(&v);
    let n0 = bilinear(&v, &bv).sqrt();
    if n0.norm() < LANCZOS_BREAKDOWN {
        return Err("the start vector is B-null: cannot begin the recurrence".to_string());
    }
    for i in 0..n_free {
        v[i] /= n0;
        bv[i] /= n0;
    }

    let mut vecs: Vec<Vec<C64>> = Vec::new();
    let mut alphas: Vec<C64> = Vec::new();
    let mut betas: Vec<C64> = Vec::new();
    let mut v_prev = vec![C64::new(0.0, 0.0); n_free];
    let mut beta_prev = C64::new(0.0, 0.0);
    let mut converged: Vec<Eigenmode> = Vec::new();
    let mut n_solves = 0usize;

    for j in 0..m_max {
        vecs.push(v.clone());

        // w = OP·v = (E − σB)⁻¹·(B·v)
        let mut w = solver.solve(&bv)?;
        n_solves += 1;

        // α = ⟨v, w⟩_B = (B·v)ᵀ·w, from the B·v already in hand.
        let alpha = bilinear(&bv, &w);
        alphas.push(alpha);

        // The three-term recurrence, then re-establish B-orthogonality against the
        // whole basis. The recurrence is exact in theory and loses orthogonality in
        // practice, and it is precisely that loss which manufactures ghost Ritz
        // values — the defect this rewrite exists to remove.
        for i in 0..n_free {
            w[i] -= alpha * v[i] + beta_prev * v_prev[i];
        }

        // Classical Gram-Schmidt, twice ("twice is enough", Kahan). The second sweep
        // is skipped when the first barely changed w, which is the usual case.
        let mut bw = b_mat.matvec(&w);
        for pass in 0..2 {
            let before = norm2(&w);
            for vk in vecs.iter() {
                let c = bilinear(vk, &bw);
                if c.norm() == 0.0 {
                    continue;
                }
                for i in 0..n_free {
                    w[i] -= c * vk[i];
                }
            }
            bw = b_mat.matvec(&w);
            if pass == 0 && norm2(&w) > 0.7 * before {
                break; // orthogonality was not materially disturbed
            }
        }

        // β = √(⟨w, w⟩_B). Complex; its sign is a free choice as long as the same β
        // goes into T and into the recurrence.
        let beta = bilinear(&w, &bw).sqrt();

        // ⟨w,w⟩_B can vanish for w ≠ 0: the form is bilinear, not definite, so a
        // complex-symmetric Lanczos can break down without having found an invariant
        // subspace. Stop the recurrence rather than divide by it; whatever Ritz pairs
        // exist so far are still checked on their residual, so nothing unsound leaks.
        if beta.norm() < LANCZOS_BREAKDOWN * norm2(&w).max(1.0) {
            break;
        }
        betas.push(beta);

        for i in 0..n_free {
            w[i] /= beta;
            bw[i] /= beta;
        }
        v_prev = v;
        beta_prev = beta;
        v = w;
        bv = bw;

        // Stop as soon as enough modes are genuinely converged. Checking costs an
        // eigendecomposition of a tiny tridiagonal plus two sparse matvecs per
        // candidate, against a Krylov step that costs a sparse back-substitution and
        // an O(m·n) sweep — cheap enough to do often, and it turns the cap above from
        // a cost into a ceiling.
        let m = alphas.len();
        if m >= n_modes + 2 && (j + 1) % CHECK_EVERY == 0 {
            converged = extract_modes(
                &alphas, &betas, &vecs, &e_mat, &b_mat, sigma, &free_dofs, n_field, mesh.l0,
                n_cand,
            )?
            .0;
            if converged.len() >= n_modes {
                break;
            }
        }
    }

    let m = alphas.len();
    if converged.len() < n_modes {
        converged = extract_modes(
            &alphas, &betas, &vecs, &e_mat, &b_mat, sigma, &free_dofs, n_field, mesh.l0, n_cand,
        )?
        .0;
    }
    let mut modes = converged;

    eprintln!(
        "  Eigenmode: {} Lanczos steps ({} solves) in {:.1}ms, {} modes converged",
        m,
        n_solves,
        t2.elapsed().as_secs_f64() * 1e3,
        modes.len()
    );

    if modes.len() < n_modes {
        // Say so, and return what is real. Padding the list with unconverged Ritz
        // pairs to reach n_modes is exactly the behaviour this rewrite removes.
        //
        // The likeliest cause is the reachable band (see the module docs): a mode
        // beyond √2·f_target sits behind the whole discrete-gradient kernel in the
        // shift-invert spectrum and cannot be reached at any Krylov size. Say what to
        // do about it, not just that it happened.
        let f_reach = target_freq * std::f64::consts::SQRT_2;
        eprintln!(
            "  Eigenmode: WARNING - only {} of the {} requested modes converged to a residual \
             below {:.0e} in {} Lanczos steps.",
            modes.len(),
            n_modes,
            EIGEN_RESIDUAL_TOL,
            m
        );
        eprintln!(
            "  Eigenmode:   Shift-invert can only reach f < √2·f_target = {:.4e} Hz here; beyond \
             that a mode sits behind the discrete-gradient kernel. Put the target in the MIDDLE \
             of the band of interest rather than on its lowest mode.",
            f_reach
        );
    }
    modes.truncate(n_modes);
    Ok(modes)
}

/// Turn the Lanczos state into the Ritz pairs that are actually eigenpairs.
///
/// Every Ritz pair is a candidate and nothing more. It becomes a mode only by
/// satisfying `E·x = λ·B·x` to [`EIGEN_RESIDUAL_TOL`]; there is no other way to tell
/// a converged pair from an artefact of an unresolved Krylov space, and guessing was
/// what produced ghost modes below a cavity's fundamental.
///
/// Only the `limit` candidates NEAREST THE SHIFT are formed and tested. The rest are
/// not wanted and, on a large problem, forming them is what dominates the cost:
/// a Ritz vector is an O(m·n) sweep over the whole Lanczos basis, and there are m of
/// them. Sorting by `|λ − σ|` costs nothing (it needs only the tridiagonal's
/// eigenvalues), so the expensive part is done for a handful of vectors instead of
/// all of them.
///
/// Returns the modes, sorted by distance to the shift, plus how many candidates were
/// rejected as unconverged and how many as static.
#[allow(clippy::too_many_arguments)]
fn extract_modes(
    alphas: &[C64],
    betas: &[C64],
    vecs: &[Vec<C64>],
    e_mat: &Coo,
    b_mat: &Coo,
    sigma: C64,
    free_dofs: &[usize],
    n_field: usize,
    l0: f64,
    limit: usize,
) -> Result<(Vec<Eigenmode>, usize, usize), String> {
    let m = alphas.len();
    let n_free = e_mat.n;

    // The complex-symmetric tridiagonal T: α on the diagonal, β on both off-diagonals.
    let t_mat = faer::Mat::<faer::c64>::from_fn(m, m, |i, j| {
        let z = |c: C64| faer::c64 { re: c.re, im: c.im };
        if i == j {
            z(alphas[i])
        } else if j == i + 1 && i < betas.len() {
            z(betas[i])
        } else if i == j + 1 && j < betas.len() {
            z(betas[j])
        } else {
            faer::c64 { re: 0.0, im: 0.0 }
        }
    });

    let eig = t_mat.eigen().map_err(|e| format!("eigendecomposition of T failed: {e:?}"))?;
    let eigenvalues = eig.S().column_vector();
    let eigenvectors = eig.U();

    // Candidates: the Ritz values that are neither degenerate nor static, ordered by
    // distance to the shift. This is pure scalar work on the tridiagonal.
    let mut n_static = 0usize;
    let mut cands: Vec<(usize, C64)> = Vec::new();
    for k in 0..m {
        let mu = C64::new(eigenvalues[k].re, eigenvalues[k].im);
        if mu.norm() < SINGULAR_EPS {
            continue;
        }
        let lambda = sigma + C64::new(1.0, 0.0) / mu;

        // The static (gradient) modes are genuine eigenpairs at λ = 0 and would sail
        // through the residual test. They are not resonances.
        if lambda.norm() < STATIC_MODE_FLOOR * sigma.norm() {
            n_static += 1;
            continue;
        }
        cands.push((k, lambda));
    }
    cands.sort_by(|a, b| {
        (a.1 - sigma).norm().partial_cmp(&(b.1 - sigma).norm()).unwrap()
    });
    cands.truncate(limit);

    let mut modes: Vec<Eigenmode> = Vec::new();
    let mut n_unconverged = 0usize;

    for (k, lambda) in cands {
        // Ritz vector x = V·y. This is the expensive step, which is why only the
        // wanted candidates get one.
        let mut x = vec![C64::new(0.0, 0.0); n_free];
        for j in 0..m {
            let c = C64::new(eigenvectors[(j, k)].re, eigenvectors[(j, k)].im);
            if c.norm() == 0.0 {
                continue;
            }
            for i in 0..n_free {
                x[i] += c * vecs[j][i];
            }
        }

        let ex = e_mat.matvec(&x);
        let bx = b_mat.matvec(&x);
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in 0..n_free {
            num += (ex[i] - lambda * bx[i]).norm_sqr();
            den += (lambda * bx[i]).norm_sqr();
        }
        let residual = (num / den.max(1e-300)).sqrt();

        if !residual.is_finite() || residual > EIGEN_RESIDUAL_TOL {
            n_unconverged += 1;
            continue;
        }

        let mut field = vec![C64::new(0.0, 0.0); n_field];
        for (fi, &d) in free_dofs.iter().enumerate() {
            field[d] = x[fi];
        }

        // λ = κ², so √λ = κ = k₀·L₀; the physical frequency is recovered by /L₀.
        let k0 = lambda.sqrt() / C64::from(l0);
        let freq = k0 * C64::from(C0 / (2.0 * PI));
        let q = if freq.im.abs() > SINGULAR_EPS {
            0.5 * freq.re / freq.im.abs()
        } else {
            f64::INFINITY
        };

        modes.push(Eigenmode { frequency: freq, q_factor: q, eigenvalue: lambda, field, residual });
    }

    // Already in shift-distance order, since the candidates were.
    Ok((modes, n_unconverged, n_static))
}
