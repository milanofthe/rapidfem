// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Primal Schur-complement domain-decomposition solver (phase-0 spike).
//!
//! Partitions the free DOFs into per-subdomain **interior** sets and a shared
//! **interface** (skeleton) set Γ, factorizes each interior block `K_II^(s)`
//! independently, condenses to the interface system
//!
//! ```text
//!   S = K_ΓΓ − Σ_s K_ΓI^(s) (K_II^(s))^{-1} K_IΓ^(s)
//!   g = b_Γ  − Σ_s K_ΓI^(s) (K_II^(s))^{-1} b_I^(s)
//! ```
//!
//! solves `S xΓ = g`, then back-substitutes `x_I^(s)` per subdomain. The global
//! matrix is never factorized — only the interior blocks plus the (much
//! smaller) interface system. This is the memory-bounded path for large FD
//! problems; see GH issue #12.
//!
//! Key structural fact this relies on: an *interior* DOF is, by construction,
//! supported only by tets of a single subdomain, so an entry coupling two
//! interior DOFs always lies within one subdomain (`K_II` is block-diagonal),
//! and every block can be extracted from the global reduced COO purely by DOF
//! class — no per-element routing needed.
//!
//! Phase-0 scope: explicit (sparse-direct) interface solve, serial. The
//! matrix-free interface GMRES and parallel subdomain factorization are
//! phase-2 (issue #12). `RAPIDFEM_SOLVER` selects the backend used for every
//! interior block and the interface system.

use num_complex::Complex64 as C64;

use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use super::{SolverChoice, pick};

/// Class of a (free) DOF under a subdomain partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DofClass {
    /// DOF supported by exactly one subdomain's tets.
    Interior(usize),
    /// DOF on a subdomain boundary (touched by ≥2 subdomains).
    Interface,
}

/// Recursive coordinate bisection of tets into `k` subdomains.
///
/// Splits the longest-spread axis at a count-proportional median, recursively.
/// Dependency-free and deterministic. Returns `tet_subdomain[itet] ∈ 0..k`.
pub fn partition_rcb(centroids: &[[f64; 3]], k: usize) -> Vec<usize> {
    let n = centroids.len();
    let mut sub = vec![0usize; n];
    if k <= 1 || n == 0 {
        return sub;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    let mut next_id = 0usize;
    rcb_rec(centroids, &mut idx, k, &mut sub, &mut next_id);
    sub
}

fn rcb_rec(
    c: &[[f64; 3]],
    idx: &mut [usize],
    k: usize,
    sub: &mut [usize],
    next_id: &mut usize,
) {
    if k <= 1 || idx.len() <= 1 {
        let id = *next_id;
        *next_id += 1;
        for &t in idx.iter() {
            sub[t] = id;
        }
        return;
    }
    // Axis of maximum spread over this slice.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &t in idx.iter() {
        for a in 0..3 {
            lo[a] = lo[a].min(c[t][a]);
            hi[a] = hi[a].max(c[t][a]);
        }
    }
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).partial_cmp(&(hi[b] - lo[b])).unwrap())
        .unwrap();
    idx.sort_by(|&a, &b| c[a][axis].partial_cmp(&c[b][axis]).unwrap());

    let k1 = k / 2;
    let k2 = k - k1;
    // Left subtree gets a proportional share of the tets; clamp so neither
    // side is empty.
    let mut m = idx.len() * k1 / k;
    m = m.clamp(1, idx.len() - 1);
    let (left, right) = idx.split_at_mut(m);
    rcb_rec(c, left, k1, sub, next_id);
    rcb_rec(c, right, k2, sub, next_id);
}

/// Per-tet centroids (mean of the 4 vertex coordinates).
pub fn tet_centroids(mesh: &Mesh) -> Vec<[f64; 3]> {
    mesh.tets
        .iter()
        .map(|tet| {
            let mut c = [0.0; 3];
            for &n in tet.iter() {
                for a in 0..3 {
                    c[a] += mesh.nodes[n][a];
                }
            }
            for a in 0..3 {
                c[a] *= 0.25;
            }
            c
        })
        .collect()
}

/// Decode a global Nédélec-2 DOF index into its supporting geometry.
/// Returns `(is_edge, geo_index)`. DOF layout (see `Nedelec2Basis::new`):
/// `[edge m1 | face m1 | edge m2 | face m2]`.
#[inline]
fn decode_dof(d: usize, n_edges: usize, n_tris: usize) -> (bool, usize) {
    if d < n_edges {
        (true, d)
    } else if d < n_edges + n_tris {
        (false, d - n_edges)
    } else if d < 2 * n_edges + n_tris {
        (true, d - n_edges - n_tris)
    } else {
        (false, d - 2 * n_edges - n_tris)
    }
}

/// Classify each *free* DOF as interior-to-a-subdomain or interface.
///
/// `free_dofs[fi]` is the global DOF at free index `fi`. A DOF is interface iff
/// its supporting edge/face is touched by tets in ≥2 subdomains. Returns a
/// vector indexed by free index (`0..n_free`).
pub fn classify_free_dofs(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    free_dofs: &[usize],
    tet_subdomain: &[usize],
) -> Vec<DofClass> {
    let n_edges = basis.n_edges;
    let n_tris = basis.n_tris;

    // Invert tet_to_edge → edge_to_tets.
    let mut edge_tets: Vec<Vec<usize>> = vec![Vec::new(); n_edges];
    for (t, edges) in mesh.tet_to_edge.iter().enumerate() {
        for &e in edges.iter() {
            edge_tets[e].push(t);
        }
    }

    let classify_tets = |tets: &[usize]| -> DofClass {
        let mut first = usize::MAX;
        let mut multi = false;
        for &t in tets {
            if t == usize::MAX {
                continue;
            }
            let s = tet_subdomain[t];
            if first == usize::MAX {
                first = s;
            } else if s != first {
                multi = true;
            }
        }
        if multi || first == usize::MAX {
            // multi → interface; first==MAX shouldn't happen for a real DOF,
            // but treat as interface (safe: it just lands in the skeleton).
            DofClass::Interface
        } else {
            DofClass::Interior(first)
        }
    };

    free_dofs
        .iter()
        .map(|&d| {
            let (is_edge, geo) = decode_dof(d, n_edges, n_tris);
            if is_edge {
                classify_tets(&edge_tets[geo])
            } else {
                classify_tets(&mesh.tri_to_tet[geo])
            }
        })
        .collect()
}

/// A COO triplet accumulator with a local (row, col) numbering.
#[derive(Default)]
struct Coo {
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<C64>,
}
impl Coo {
    #[inline]
    fn push(&mut self, r: usize, c: usize, v: C64) {
        self.rows.push(r);
        self.cols.push(c);
        self.vals.push(v);
    }
}

#[inline]
fn norm2(v: &[C64]) -> f64 {
    v.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

/// Conjugate dot product `<a, b> = Σ conj(a_i) b_i` (Arnoldi inner product).
#[inline]
fn cdot(a: &[C64], b: &[C64]) -> C64 {
    let mut s = C64::new(0.0, 0.0);
    for i in 0..a.len() {
        s += a[i].conj() * b[i];
    }
    s
}

/// Complex Givens rotation zeroing the 2nd component of `[a; b]`. Returns
/// `(c, s)` with `c` real, applied as `x1' = c·x1 + s·x2`,
/// `x2' = -conj(s)·x1 + c·x2`.
#[inline]
fn givens(a: C64, b: C64) -> (f64, C64) {
    let (na, nb) = (a.norm(), b.norm());
    if nb == 0.0 {
        (1.0, C64::new(0.0, 0.0))
    } else if na == 0.0 {
        (0.0, b.conj() / C64::new(nb, 0.0))
    } else {
        let denom = (na * na + nb * nb).sqrt();
        (na / denom, (b.conj() * a / C64::new(na, 0.0)) / C64::new(denom, 0.0))
    }
}

/// Neumann-Neumann interface preconditioner data: for each subdomain a local
/// interface index list (local→global) and a factorized LOCAL Schur complement
/// `S_s` on that subdomain's interface DOFs, plus the global multiplicity of
/// each interface DOF (how many subdomains touch it) for the partition of unity.
struct NnPrecond {
    local_iface: Vec<Vec<usize>>,             // per subdomain: global iface idx
    s_solver: Vec<Option<Box<dyn super::SparseSolver>>>, // factorized S_s
    mult: Vec<f64>,                            // per global iface DOF
}

/// Build the Neumann-Neumann preconditioner. Captures the per-subdomain
/// interior correction that `K_ΓΓ` alone misses (the reason plain-`K_ΓΓ` GMRES
/// stalls on bulky 3D). Cost: Σ_s |Γ_s| interior back-solves + small dense
/// local factorizations — no global dense S.
fn build_nn_precond(
    k: usize,
    n_iface: usize,
    n_int: &[usize],
    kgg: &Coo,
    kig: &[Coo],
    kgi: &[Coo],
    isolv: &mut [Box<dyn super::SparseSolver>],
    choice: SolverChoice,
) -> Result<NnPrecond, String> {
    // Per-subdomain local interface set Γ_s = interface DOFs adjacent to s.
    let mut local_iface: Vec<Vec<usize>> = Vec::with_capacity(k);
    let mut g2l: Vec<Vec<usize>> = Vec::with_capacity(k); // global→local (MAX if absent)
    let mut mult = vec![0.0f64; n_iface];
    for s in 0..k {
        let mut present = vec![false; n_iface];
        for &c in &kig[s].cols {
            present[c] = true;
        }
        for &r in &kgi[s].rows {
            present[r] = true;
        }
        let mut gl = Vec::new();
        let mut map = vec![usize::MAX; n_iface];
        for g in 0..n_iface {
            if present[g] {
                map[g] = gl.len();
                gl.push(g);
                mult[g] += 1.0;
            }
        }
        local_iface.push(gl);
        g2l.push(map);
    }

    // Build + factorize each local Schur complement S_s (dense, small).
    let mut s_solver: Vec<Option<Box<dyn super::SparseSolver>>> = Vec::with_capacity(k);
    for s in 0..k {
        let m = local_iface[s].len();
        if m == 0 || n_int[s] == 0 {
            s_solver.push(None);
            continue;
        }
        // K_IΓ^s columns keyed by GLOBAL interface index.
        let mut kig_cols: Vec<Vec<(usize, C64)>> = vec![Vec::new(); n_iface];
        for i in 0..kig[s].rows.len() {
            kig_cols[kig[s].cols[i]].push((kig[s].rows[i], kig[s].vals[i]));
        }
        let mut sd = vec![C64::new(0.0, 0.0); m * m];
        // K_ΓΓ restricted to Γ_s.
        for i in 0..kgg.rows.len() {
            let (rg, cg) = (kgg.rows[i], kgg.cols[i]);
            let (rl, cl) = (g2l[s][rg], g2l[s][cg]);
            if rl != usize::MAX && cl != usize::MAX {
                sd[rl * m + cl] += kgg.vals[i];
            }
        }
        // − K_ΓI^s (K_II^s)^{-1} K_IΓ^s restricted to Γ_s, column by column.
        for jl in 0..m {
            let jg = local_iface[s][jl];
            let col = &kig_cols[jg];
            if col.is_empty() {
                continue;
            }
            let mut b = vec![C64::new(0.0, 0.0); n_int[s]];
            for &(ir, val) in col {
                b[ir] += val;
            }
            let w = isolv[s].solve(&b)?;
            for t in 0..kgi[s].rows.len() {
                let rl = g2l[s][kgi[s].rows[t]];
                if rl != usize::MAX {
                    sd[rl * m + jl] -= kgi[s].vals[t] * w[kgi[s].cols[t]];
                }
            }
        }
        // Factorize S_s via the backend (dense-as-sparse: emit all entries).
        let mut sr = Vec::new();
        let mut sc = Vec::new();
        let mut sv = Vec::new();
        for i in 0..m {
            for j in 0..m {
                let v = sd[i * m + j];
                if i == j || v != C64::new(0.0, 0.0) {
                    sr.push(i);
                    sc.push(j);
                    sv.push(v);
                }
            }
        }
        let mut solver = pick(choice);
        solver.factorize(m, &sr, &sc, &sv)
            .map_err(|e| format!("local Schur S_{s} factorize failed: {e}"))?;
        s_solver.push(Some(solver));
    }
    Ok(NnPrecond { local_iface, s_solver, mult })
}

/// Apply the Neumann-Neumann preconditioner: M^{-1} r = Σ_s R_s^T D_s S_s^{-1} D_s R_s r,
/// with D_s the inverse-multiplicity partition of unity.
fn nn_apply(nn: &mut NnPrecond, r: &[C64]) -> Vec<C64> {
    let n_iface = r.len();
    let mut out = vec![C64::new(0.0, 0.0); n_iface];
    for s in 0..nn.local_iface.len() {
        let solver = match nn.s_solver[s].as_mut() {
            Some(sv) => sv,
            None => continue,
        };
        let gl = &nn.local_iface[s];
        let m = gl.len();
        let mut rs = vec![C64::new(0.0, 0.0); m];
        for (jl, &jg) in gl.iter().enumerate() {
            rs[jl] = r[jg] / C64::new(nn.mult[jg], 0.0);
        }
        if let Ok(ws) = solver.solve(&rs) {
            for (jl, &jg) in gl.iter().enumerate() {
                out[jg] += ws[jl] / C64::new(nn.mult[jg], 0.0);
            }
        }
    }
    out
}

/// Right-preconditioned restarted GMRES for `op(x) = b`, preconditioner
/// `prec(v) ≈ A^{-1} v`. Returns `(x, total_iters, final_rel_resid)`.
/// `op`/`prec` are `FnMut` so they can hold the (mutable) subdomain solvers.
fn gmres_solve(
    op: &mut dyn FnMut(&[C64]) -> Vec<C64>,
    prec: &mut dyn FnMut(&[C64]) -> Vec<C64>,
    b: &[C64],
    tol: f64,
    restart: usize,
    max_outer: usize,
) -> (Vec<C64>, usize, f64) {
    let n = b.len();
    let zero = C64::new(0.0, 0.0);
    let bnorm = norm2(b);
    let mut x = vec![zero; n];
    if bnorm == 0.0 {
        return (x, 0, 0.0);
    }
    let mut total = 0usize;
    for _outer in 0..max_outer {
        let sx = op(&x);
        let mut r: Vec<C64> = (0..n).map(|i| b[i] - sx[i]).collect();
        let beta = norm2(&r);
        if beta <= tol * bnorm {
            return (x, total, beta / bnorm);
        }
        let mut v: Vec<Vec<C64>> = Vec::with_capacity(restart + 1);
        let mut z: Vec<Vec<C64>> = Vec::with_capacity(restart);
        let inv_beta = C64::new(1.0 / beta, 0.0);
        for c in r.iter_mut() {
            *c *= inv_beta;
        }
        v.push(r);
        let mut h = vec![vec![zero; restart]; restart + 1];
        let mut cs = vec![0.0f64; restart];
        let mut sn = vec![zero; restart];
        let mut gbar = vec![zero; restart + 1];
        gbar[0] = C64::new(beta, 0.0);
        let mut jused = 0usize;
        for j in 0..restart {
            total += 1;
            let zj = prec(&v[j]);
            let mut w = op(&zj);
            z.push(zj);
            for i in 0..=j {
                let hij = cdot(&v[i], &w);
                h[i][j] = hij;
                for t in 0..n {
                    w[t] -= hij * v[i][t];
                }
            }
            let hnext = norm2(&w);
            h[j + 1][j] = C64::new(hnext, 0.0);
            if hnext > 0.0 {
                let inv = C64::new(1.0 / hnext, 0.0);
                for c in w.iter_mut() {
                    *c *= inv;
                }
                v.push(w);
            } else {
                v.push(vec![zero; n]);
            }
            for i in 0..j {
                let temp = C64::new(cs[i], 0.0) * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i].conj() * h[i][j] + C64::new(cs[i], 0.0) * h[i + 1][j];
                h[i][j] = temp;
            }
            let (c, s) = givens(h[j][j], h[j + 1][j]);
            cs[j] = c;
            sn[j] = s;
            h[j][j] = C64::new(c, 0.0) * h[j][j] + s * h[j + 1][j];
            h[j + 1][j] = zero;
            let gj = gbar[j];
            gbar[j] = C64::new(c, 0.0) * gj;
            gbar[j + 1] = -s.conj() * gj;
            jused = j + 1;
            if gbar[j + 1].norm() <= tol * bnorm {
                break;
            }
        }
        // Back-solve the (jused × jused) upper-triangular least-squares system.
        let mut y = vec![zero; jused];
        for i in (0..jused).rev() {
            let mut acc = gbar[i];
            for col in (i + 1)..jused {
                acc -= h[i][col] * y[col];
            }
            y[i] = acc / h[i][i];
        }
        for i in 0..jused {
            for t in 0..n {
                x[t] += y[i] * z[i][t];
            }
        }
    }
    let sx = op(&x);
    let rr: f64 = (0..n).map(|i| (b[i] - sx[i]).norm_sqr()).sum::<f64>().sqrt();
    (x, total, rr / bnorm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schur_3x3_hand_checked() {
        // A = [[4,1,0],[1,3,1],[0,1,2]], b=[1,2,3]. Exact: x=[2/9,1/9,13/9].
        // Classes: DOF0 Interior(0), DOF1 Interface, DOF2 Interior(1).
        let c = |x: f64| C64::new(x, 0.0);
        // both halves (the real FEM COO is stored symmetrically)
        let rows = vec![0, 1, 2, 0, 1, 1, 2];
        let cols = vec![0, 1, 2, 1, 0, 2, 1];
        let vals = vec![c(4.0), c(3.0), c(2.0), c(1.0), c(1.0), c(1.0), c(1.0)];
        let rhs = vec![vec![c(1.0), c(2.0), c(3.0)]];
        let cls = vec![DofClass::Interior(0), DofClass::Interface, DofClass::Interior(1)];
        let x = schur_solve(3, &rows, &cols, &vals, &rhs, &cls, SolverChoice::Faer).unwrap();
        let expect = [2.0 / 9.0, 1.0 / 9.0, 13.0 / 9.0];
        for i in 0..3 {
            let e = (x[0][i].re - expect[i]).abs();
            assert!(e < 1e-10, "x[{i}]={:.6} expected {:.6} (err {e:.2e})", x[0][i].re, expect[i]);
        }
    }
}

/// Solve the reduced free-DOF system by primal Schur DD.
///
/// Inputs are exactly what the monolithic path already builds: the reduced COO
/// `(rows, cols, vals)` of dimension `n_free` (free-DOF indexed, PEC already
/// eliminated), one or more RHS vectors `rhs` (each length `n_free`), and the
/// per-free-DOF classification. Returns one solution vector (length `n_free`)
/// per RHS — identical to the monolithic solve up to solver precision.
pub fn schur_solve(
    n_free: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[C64],
    rhs: &[Vec<C64>],
    dof_class: &[DofClass],
    choice: SolverChoice,
) -> Result<Vec<Vec<C64>>, String> {
    assert_eq!(dof_class.len(), n_free, "dof_class must cover all free DOFs");

    // Number of subdomains present.
    let k = dof_class
        .iter()
        .filter_map(|c| match c {
            DofClass::Interior(s) => Some(*s + 1),
            DofClass::Interface => None,
        })
        .max()
        .unwrap_or(0);

    // Local numbering: per-subdomain interior index, and a global interface
    // index. `loc[fi]` is the in-block index; the block is given by class.
    let mut loc = vec![usize::MAX; n_free];
    let mut n_int = vec![0usize; k];
    let mut n_iface = 0usize;
    for (fi, c) in dof_class.iter().enumerate() {
        match *c {
            DofClass::Interior(s) => {
                loc[fi] = n_int[s];
                n_int[s] += 1;
            }
            DofClass::Interface => {
                loc[fi] = n_iface;
                n_iface += 1;
            }
        }
    }

    // Route the global reduced COO into blocks.
    let mut kii: Vec<Coo> = (0..k).map(|_| Coo::default()).collect();
    let mut kig: Vec<Coo> = (0..k).map(|_| Coo::default()).collect(); // interior×iface
    let mut kgi: Vec<Coo> = (0..k).map(|_| Coo::default()).collect(); // iface×interior
    let mut kgg = Coo::default();

    for i in 0..rows.len() {
        let (r, c, v) = (rows[i], cols[i], vals[i]);
        match (dof_class[r], dof_class[c]) {
            (DofClass::Interior(sr), DofClass::Interior(sc)) => {
                debug_assert_eq!(sr, sc, "interior-interior coupling across subdomains");
                kii[sr].push(loc[r], loc[c], v);
            }
            (DofClass::Interior(sr), DofClass::Interface) => {
                kig[sr].push(loc[r], loc[c], v);
            }
            (DofClass::Interface, DofClass::Interior(sc)) => {
                kgi[sc].push(loc[r], loc[c], v);
            }
            (DofClass::Interface, DofClass::Interface) => {
                kgg.push(loc[r], loc[c], v);
            }
        }
    }

    // Memory proxy: a single direct factorization's fill-in scales super-
    // linearly with the matrix size, so the relevant bound is the LARGEST
    // interior block we ever factorize, not the global n_free. Log it (and the
    // interface size) so the win over the monolithic factor is visible.
    let max_int = n_int.iter().copied().max().unwrap_or(0);
    eprintln!(
        "  Schur: {k} subdomains, largest interior block {max_int}/{n_free} \
         ({:.0}% of global), interface {n_iface} ({:.0}%)",
        100.0 * max_int as f64 / n_free as f64,
        100.0 * n_iface as f64 / n_free as f64,
    );

    // Factorize each interior block once (reused for all back-solves and RHS).
    let mut isolv: Vec<Box<dyn super::SparseSolver>> = Vec::with_capacity(k);
    for s in 0..k {
        let mut solver = pick(choice);
        if n_int[s] > 0 {
            solver
                .factorize(n_int[s], &kii[s].rows, &kii[s].cols, &kii[s].vals)
                .map_err(|e| format!("interior block {s} factorize failed: {e} \
                    (possible interior resonance — see issue #12)"))?;
        }
        isolv.push(solver);
    }

    // Column views of K_IΓ^(s): iface column j → list of (interior_row, val).
    // Used both to form S and to back-substitute.
    let kig_cols: Vec<Vec<Vec<(usize, C64)>>> = (0..k)
        .map(|s| {
            let mut cols_v: Vec<Vec<(usize, C64)>> = vec![Vec::new(); n_iface];
            for i in 0..kig[s].rows.len() {
                cols_v[kig[s].cols[i]].push((kig[s].rows[i], kig[s].vals[i]));
            }
            cols_v
        })
        .collect();

    // Interface-solve setup. Small interface → form the dense Schur matrix S
    // and factorize it directly (exact, cheap). Large interface → matrix-free
    // GMRES with the sparse K_ΓΓ as preconditioner: never store the dense S,
    // and replace O(n_iface) back-solves with ~iters×k (issue #12).
    // Interface-DOF threshold for switching to matrix-free GMRES; the const
    // default can be overridden via RAPIDFEM_SCHUR_EXPLICIT_MAX (also lets a
    // test force the GMRES path on a small system).
    let explicit_max = std::env::var("RAPIDFEM_SCHUR_EXPLICIT_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(crate::constants::SCHUR_EXPLICIT_INTERFACE_MAX);
    let use_gmres = n_iface > explicit_max;
    let mut iface_solver: Option<Box<dyn super::SparseSolver>> = None; // dense-S path
    let mut nn_precond: Option<NnPrecond> = None;                      // GMRES preconditioner

    if n_iface > 0 && use_gmres {
        nn_precond = Some(build_nn_precond(
            k, n_iface, &n_int, &kgg, &kig, &kgi, &mut isolv, choice)?);
        eprintln!("  Schur: matrix-free interface GMRES (n_iface={n_iface}, Neumann-Neumann preconditioned)");
    } else if n_iface > 0 {
        // Dense S = K_ΓΓ − Σ_s K_ΓI^(s) (K_II^(s))^{-1} K_IΓ^(s), column by column.
        let mut s_dense = vec![C64::new(0.0, 0.0); n_iface * n_iface];
        for i in 0..kgg.rows.len() {
            s_dense[kgg.rows[i] * n_iface + kgg.cols[i]] += kgg.vals[i];
        }
        for s in 0..k {
            if n_int[s] == 0 {
                continue;
            }
            for j in 0..n_iface {
                let col = &kig_cols[s][j];
                if col.is_empty() {
                    continue;
                }
                let mut b = vec![C64::new(0.0, 0.0); n_int[s]];
                for &(ir, val) in col {
                    b[ir] += val;
                }
                let w = isolv[s].solve(&b)?;
                for t in 0..kgi[s].rows.len() {
                    s_dense[kgi[s].rows[t] * n_iface + j] -= kgi[s].vals[t] * w[kgi[s].cols[t]];
                }
            }
        }
        let mut sr = Vec::new();
        let mut sc = Vec::new();
        let mut sv = Vec::new();
        for i in 0..n_iface {
            for j in 0..n_iface {
                let v = s_dense[i * n_iface + j];
                if i == j || v != C64::new(0.0, 0.0) {
                    sr.push(i);
                    sc.push(j);
                    sv.push(v);
                }
            }
        }
        let mut solver = pick(choice);
        solver
            .factorize(n_iface, &sr, &sc, &sv)
            .map_err(|e| format!("interface system factorize failed: {e}"))?;
        iface_solver = Some(solver);
    }

    // Per-RHS condense → interface solve → back-substitute.
    let mut solutions = Vec::with_capacity(rhs.len());
    for b_full in rhs {
        // u_s = K_II^(s)^{-1} b_I^(s); g = b_Γ − Σ_s K_ΓI^(s) u_s.
        let mut g = vec![C64::new(0.0, 0.0); n_iface];
        for (fi, c) in dof_class.iter().enumerate() {
            if let DofClass::Interface = c {
                g[loc[fi]] += b_full[fi];
            }
        }
        let mut u: Vec<Vec<C64>> = Vec::with_capacity(k);
        for s in 0..k {
            if n_int[s] == 0 {
                u.push(Vec::new());
                continue;
            }
            let mut b_i = vec![C64::new(0.0, 0.0); n_int[s]];
            for (fi, c) in dof_class.iter().enumerate() {
                if let DofClass::Interior(ss) = c {
                    if *ss == s {
                        b_i[loc[fi]] = b_full[fi];
                    }
                }
            }
            let u_s = isolv[s].solve(&b_i)?;
            for t in 0..kgi[s].rows.len() {
                g[kgi[s].rows[t]] -= kgi[s].vals[t] * u_s[kgi[s].cols[t]];
            }
            u.push(u_s);
        }

        // Interface solve: matrix-free GMRES (large) or direct dense S (small).
        let x_g = if n_iface == 0 {
            Vec::new()
        } else if use_gmres {
            // S·v = K_ΓΓ·v − Σ_s K_ΓI^(s) (K_II^(s))^{-1} (K_IΓ^(s)·v), matrix-free.
            let mut op = |v: &[C64]| -> Vec<C64> {
                let mut acc = vec![C64::new(0.0, 0.0); n_iface];
                for i in 0..kgg.rows.len() {
                    acc[kgg.rows[i]] += kgg.vals[i] * v[kgg.cols[i]];
                }
                for s in 0..k {
                    if n_int[s] == 0 {
                        continue;
                    }
                    let mut t = vec![C64::new(0.0, 0.0); n_int[s]];
                    for e in 0..kig[s].rows.len() {
                        t[kig[s].rows[e]] += kig[s].vals[e] * v[kig[s].cols[e]];
                    }
                    let w = isolv[s].solve(&t).expect("interior solve in Schur matvec");
                    for e in 0..kgi[s].rows.len() {
                        acc[kgi[s].rows[e]] -= kgi[s].vals[e] * w[kgi[s].cols[e]];
                    }
                }
                acc
            };
            let nn = nn_precond.as_mut().unwrap();
            let mut prec = |v: &[C64]| -> Vec<C64> { nn_apply(nn, v) };
            let (xg, iters, resid) = gmres_solve(
                &mut op,
                &mut prec,
                &g,
                crate::constants::SCHUR_GMRES_TOL,
                crate::constants::SCHUR_GMRES_RESTART,
                crate::constants::SCHUR_GMRES_MAX_OUTER,
            );
            eprintln!("  Schur interface GMRES: {iters} iters, rel.resid {resid:.2e}");
            xg
        } else {
            iface_solver.as_mut().unwrap().solve(&g)?
        };

        // Back-substitute: x_I^(s) = u_s − K_II^(s)^{-1} (K_IΓ^(s) xΓ).
        let mut x_full = vec![C64::new(0.0, 0.0); n_free];
        for (fi, c) in dof_class.iter().enumerate() {
            if let DofClass::Interface = c {
                x_full[fi] = x_g[loc[fi]];
            }
        }
        for s in 0..k {
            if n_int[s] == 0 {
                continue;
            }
            // rhs2 = K_IΓ^(s) xΓ
            let mut rhs2 = vec![C64::new(0.0, 0.0); n_int[s]];
            if n_iface > 0 {
                for j in 0..n_iface {
                    let xj = x_g[j];
                    if xj == C64::new(0.0, 0.0) {
                        continue;
                    }
                    for &(ir, val) in &kig_cols[s][j] {
                        rhs2[ir] += val * xj;
                    }
                }
            }
            let corr = if rhs2.iter().any(|v| *v != C64::new(0.0, 0.0)) {
                isolv[s].solve(&rhs2)?
            } else {
                vec![C64::new(0.0, 0.0); n_int[s]]
            };
            for (fi, c) in dof_class.iter().enumerate() {
                if let DofClass::Interior(ss) = c {
                    if *ss == s {
                        x_full[fi] = u[s][loc[fi]] - corr[loc[fi]];
                    }
                }
            }
        }
        solutions.push(x_full);
    }

    Ok(solutions)
}
