// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Clean-room surface (boundary-triangle) assembly for the Robin / port BC.
//!
//! On a boundary face the curl-conforming element restricts to the canonical
//! R2 *surface* element: 8 DOFs = 3 edges × 2 modes + 1 face × 2 modes, with
//! the same Whitney-times-nodal-weight form as the volume basis but in the
//! triangle's tangent plane (W_ab = L_a ∇L_b − L_b ∇L_a, 2-D gradients):
//!
//!   edge e=(a,b): φ_e1 = ℓ L_a W_ab,  φ_e2 = ℓ L_b W_ab
//!   face (0,1,2): φ_f1 = |0,2| L_1 W_02 (sign-matched),  φ_f2 = |0,1| L_2 W_01
//!
//! The Robin term ∫ γ (n̂×φ_i)·(n̂×φ_j) dA reduces, for tangential fields, to
//! γ ∫ φ_i·φ_j dA — the surface mass matrix. The forcing is ∫ φ_i·u_inc dA.
//! Both integrate exactly with the barycentric area coefficients.
//!
//! DOF order matches `basis::Nedelec2Basis`: [e0 e1 e2 (m1)], face·m1,
//! [e0 e1 e2 (m2)], face·m2 → indices 0,1,2,3,4,5,6,7.

use num_complex::Complex64 as C64;
use crate::coefficients::AreaCoeffCache;

type V2 = [f64; 2];

/// Surface local edge order (sorted-triangle convention, matches `mesh`).
const TRI_EDGE_MAP: [[usize; 2]; 3] = [[0, 1], [1, 2], [0, 2]];

#[inline]
fn dot2(a: &V2, b: &V2) -> f64 { a[0]*b[0] + a[1]*b[1] }

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

#[inline]
fn norm3(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0]*a[0] + a[1]*a[1] + a[2]*a[2]).sqrt();
    [a[0]/n, a[1]/n, a[2]/n]
}

/// Local right-handed 2-D frame of a triangle: returns (rotation rows, xs, ys)
/// with vertex 0 at the origin, edge 0→1 along x̂, n̂ = ê1×ê2 as ẑ.
fn tri_local_cs(v: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [f64; 3]) {
    let o = v[0];
    let e1 = [v[1][0]-o[0], v[1][1]-o[1], v[1][2]-o[2]];
    let e2 = [v[2][0]-o[0], v[2][1]-o[1], v[2][2]-o[2]];
    let zhat = norm3(cross3(e1, e2));
    let xhat = norm3(e1);
    let yhat = norm3(cross3(zhat, xhat));
    let basis = [xhat, yhat, zhat];
    let mut xs = [0.0; 3];
    let mut ys = [0.0; 3];
    for k in 0..3 {
        let d = [v[k][0]-o[0], v[k][1]-o[1], v[k][2]-o[2]];
        xs[k] = basis[0][0]*d[0] + basis[0][1]*d[1] + basis[0][2]*d[2];
        ys[k] = basis[1][0]*d[0] + basis[1][1]*d[1] + basis[1][2]*d[2];
    }
    (basis, xs, ys)
}

/// 2-D barycentric gradients ∇L_i = (b_i, c_i)/(2A) and the signed 2A.
fn bary_grads_2d(xs: &[f64; 3], ys: &[f64; 3]) -> ([V2; 3], f64) {
    let (x1, x2, x3) = (xs[0], xs[1], xs[2]);
    let (y1, y2, y3) = (ys[0], ys[1], ys[2]);
    let two_a = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2); // signed 2·Area
    let b = [y2-y3, y3-y1, y1-y2];
    let c = [x3-x2, x1-x3, x2-x1];

    // Sliver guard (2-D analogue of the tet floor): a near-collinear boundary
    // triangle has 2A → 0; floor it so ∇L stays finite.
    let mut sum_len = 0.0;
    for &(i, j) in &[(0, 1), (1, 2), (0, 2)] {
        let dx = xs[i] - xs[j];
        let dy = ys[i] - ys[j];
        sum_len += (dx * dx + dy * dy).sqrt();
    }
    let h_mean = sum_len / 3.0;
    let floor = crate::constants::SLIVER_NORMVOL_FLOOR * h_mean * h_mean;
    let two_a_eff = if two_a.abs() < floor {
        floor.copysign(if two_a == 0.0 { 1.0 } else { two_a })
    } else {
        two_a
    };

    let inv = 1.0 / two_a_eff;
    let grads = std::array::from_fn(|i| [b[i]*inv, c[i]*inv]);
    (grads, two_a_eff)
}

#[derive(Clone, Copy)]
struct Term { coeff: f64, mono: [usize; 2], grad: usize }
struct SurfFn { scale: f64, terms: [Term; 2] }

fn node_dist(xs: &[f64; 3], ys: &[f64; 3], i: usize, j: usize) -> f64 {
    ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2)).sqrt()
}

/// Build the 8 surface basis functions in DOF order.
fn build_surface_basis(xs: &[f64; 3], ys: &[f64; 3]) -> [SurfFn; 8] {
    let d = |i, j| node_dist(xs, ys, i, j);
    let mk = |scale: f64, c0: f64, m0: [usize; 2], g0: usize,
              c1: f64, m1: [usize; 2], g1: usize| SurfFn {
        scale, terms: [Term { coeff: c0, mono: m0, grad: g0 },
                       Term { coeff: c1, mono: m1, grad: g1 }],
    };
    // edges m1: ℓ L_a (L_a ∇L_b − L_b ∇L_a)
    let e = |a: usize, b: usize, weight: usize| {
        let l = d(a, b);
        mk(l, 1.0, [weight, a], b, -1.0, [weight, b], a)
    };
    let e0 = TRI_EDGE_MAP[0]; let e1 = TRI_EDGE_MAP[1]; let e2 = TRI_EDGE_MAP[2];
    // face: φ_f1 = |0,2| L_1 (L_2 ∇L_0 − L_0 ∇L_2) (sign-matched to volume),
    //       φ_f2 = |0,1| L_2 (L_0 ∇L_1 − L_1 ∇L_0)
    let f1 = mk(d(0, 2), -1.0, [1, 0], 2, 1.0, [1, 2], 0);
    let f2 = mk(d(0, 1), 1.0, [2, 0], 1, -1.0, [2, 1], 0);
    [
        e(e0[0], e0[1], e0[0]),   // 0: edge0 m1
        e(e1[0], e1[1], e1[0]),   // 1: edge1 m1
        e(e2[0], e2[1], e2[0]),   // 2: edge2 m1
        f1,                        // 3: face m1
        e(e0[0], e0[1], e0[1]),   // 4: edge0 m2
        e(e1[0], e1[1], e1[1]),   // 5: edge1 m2
        e(e2[0], e2[1], e2[1]),   // 6: edge2 m2
        f2,                        // 7: face m2
    ]
}

/// ∫ L_p L_q L_r L_s dA over the triangle (local 0-based node indices).
#[inline]
fn integ_area(p: usize, q: usize, r: usize, s: usize, area: f64, ac: &AreaCoeffCache) -> f64 {
    // area_coeff returns ∫/A for 1-based indices, 0 = unused.
    ac.get(p + 1, q + 1, r + 1, s + 1) * area
}

/// Surface Robin stiffness: `γ ∫ φ_i·φ_j dA`, an 8×8 complex matrix.
pub fn ned2_tri_stiff(
    glob_vertices: &[[f64; 3]; 3],
    gamma: C64,
    ac_base: &AreaCoeffCache,
) -> [[C64; 8]; 8] {
    let (_, xs, ys) = tri_local_cs(glob_vertices);
    let (grads, two_a) = bary_grads_2d(&xs, &ys);
    let area = 0.5 * two_a.abs();
    let fns = build_surface_basis(&xs, &ys);

    let mut bmat = [[C64::new(0.0, 0.0); 8]; 8];
    for i in 0..8 {
        for j in i..8 {
            let sc = fns[i].scale * fns[j].scale;
            let mut acc = 0.0;
            for ti in &fns[i].terms {
                for tj in &fns[j].terms {
                    let g = dot2(&grads[ti.grad], &grads[tj.grad]);
                    let intg = integ_area(ti.mono[0], ti.mono[1], tj.mono[0], tj.mono[1], area, ac_base);
                    acc += ti.coeff * tj.coeff * g * intg;
                }
            }
            let val = gamma * C64::from(sc * acc);
            bmat[i][j] = val;
            bmat[j][i] = val;
        }
    }
    bmat
}

/// Surface excitation: `∫ φ_i·u_inc dA` by quadrature, an 8-vector.
/// `dpts[q] = [w, L1, L2, L3]`, `glob_uinc[q]` the incident field at that point.
pub fn ned2_tri_force(
    glob_vertices: &[[f64; 3]; 3],
    glob_uinc: &[[C64; 3]],
    dpts: &[[f64; 4]],
) -> [C64; 8] {
    let (basis, xs, ys) = tri_local_cs(glob_vertices);
    let (grads, two_a) = bary_grads_2d(&xs, &ys);
    let area = 0.5 * two_a.abs();
    let fns = build_surface_basis(&xs, &ys);

    // incident field rotated into the local frame (tangential x,y components)
    let lcs_uinc: Vec<[C64; 3]> = glob_uinc.iter().map(|c| [
        C64::from(basis[0][0])*c[0] + C64::from(basis[0][1])*c[1] + C64::from(basis[0][2])*c[2],
        C64::from(basis[1][0])*c[0] + C64::from(basis[1][1])*c[1] + C64::from(basis[1][2])*c[2],
        C64::from(basis[2][0])*c[0] + C64::from(basis[2][1])*c[1] + C64::from(basis[2][2])*c[2],
    ]).collect();

    // barycentric L_i at a quad point from its (L1,L2,L3) — direct.
    let mut bvec = [C64::new(0.0, 0.0); 8];
    for (fi, f) in fns.iter().enumerate() {
        let mut sum = C64::new(0.0, 0.0);
        for (qi, qp) in dpts.iter().enumerate() {
            let w = qp[0];
            let lam = [qp[1], qp[2], qp[3]];
            // φ(point) tangential vector = scale·Σ coeff·L_p·L_q·∇L_g
            // force pairs with the volume assembly basis (build_basis sign),
            // not the interp reconstruction sign.
            let mut phi = [0.0_f64; 2];
            for t in &f.terms {
                let s = f.scale * t.coeff * lam[t.mono[0]] * lam[t.mono[1]];
                phi[0] += s * grads[t.grad][0];
                phi[1] += s * grads[t.grad][1];
            }
            let ux = lcs_uinc[qi][0];
            let uy = lcs_uinc[qi][1];
            sum += C64::from(w) * (C64::from(phi[0])*ux + C64::from(phi[1])*uy);
        }
        bvec[fi] = C64::from(area) * sum;
    }
    bvec
}
