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

    // Dense interface matrix S (n_iface × n_iface), seeded with K_ΓΓ.
    let mut s_dense = vec![C64::new(0.0, 0.0); n_iface * n_iface];
    for i in 0..kgg.rows.len() {
        s_dense[kgg.rows[i] * n_iface + kgg.cols[i]] += kgg.vals[i];
    }

    // Subtract Σ_s K_ΓI^(s) (K_II^(s))^{-1} K_IΓ^(s), column by column.
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
            let w = isolv[s].solve(&b)?; // w = K_II^{-1} K_IΓ[:, j]
            // S[:, j] -= K_ΓI^(s) w   (K_ΓI triplets: iface_row, interior_col)
            for t in 0..kgi[s].rows.len() {
                let ir = kgi[s].rows[t];
                let ic = kgi[s].cols[t];
                s_dense[ir * n_iface + j] -= kgi[s].vals[t] * w[ic];
            }
        }
    }

    // Build + factorize the interface system once (skip if no interface).
    let mut iface_solver = pick(choice);
    if n_iface > 0 {
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
        iface_solver
            .factorize(n_iface, &sr, &sc, &sv)
            .map_err(|e| format!("interface system factorize failed: {e}"))?;
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

        // Interface solve.
        let x_g = if n_iface > 0 {
            iface_solver.solve(&g)?
        } else {
            Vec::new()
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
