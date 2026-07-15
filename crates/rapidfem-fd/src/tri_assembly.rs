// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Surface (boundary-triangle) assembly for the Robin / port BC.
//!
//! The surface element is the **tangential trace of the volume element**, not a
//! second element. On a boundary face F of a tetrahedron:
//!
//!   * the barycentric coordinate of the opposite vertex vanishes on F, and
//!   * the tangential part of its gradient vanishes on F as well, while for the
//!     three vertices of F the tangential part of ∇L is exactly the triangle's own
//!     2-D barycentric gradient.
//!
//! So the trace of a volume function whose terms name only F's vertices is
//! obtained by reading the same terms with 2-D gradients, and the volume functions
//! that name the opposite vertex trace to zero. Exactly 8 of the 20 volume DOFs
//! survive: F's three edges × 2 modes, and F itself × 2 modes. That is the
//! surface element.
//!
//! `build_surface_basis` therefore calls `tet_assembly::edge_fns` and
//! `face_fns` — the same generators the volume element is built from — on the
//! triangle's own three nodes. It does not restate the functions, so it cannot
//! disagree with the volume element about their sign. (It used to restate them,
//! with a comment claiming the signs had been matched by hand.) The identity is
//! proved symbolically in `derivations/nedelec2/face_trace.py` and checked against
//! the volume element on real tetrahedra in `tests/face_trace_test.rs`.
//!
//! The Robin term ∫ γ (n̂×φ_i)·(n̂×φ_j) dA reduces, for tangential fields, to
//! γ ∫ φ_i·φ_j dA — the surface mass matrix. The forcing is ∫ φ_i·u_inc dA. Both
//! integrate exactly with the barycentric area coefficients; no quadrature is
//! needed for the mass.
//!
//! DOF order matches `basis::tri_dof_owners`: [e0 e1 e2]·m1, face·m1,
//! [e0 e1 e2]·m2, face·m2 → indices 0..8.

use num_complex::Complex64 as C64;
use rapidfem_core::mesh::TRI_EDGE_LOCAL;

use crate::coefficients::area_coeff_exps;
use crate::dofmap::DofOwner;
use crate::tet_assembly::{edge_fns, face_fns, BasisFn, BasisKind};

type V2 = [f64; 2];

/// Number of DOFs on the surface element at uniform order 2: 3 edges × 2 modes
/// + 1 face × 2. Under the minimum rule it can be less; ask the DOF map, do not
/// assume this.
pub const N_TRI_DOFS_P2: usize = 8;

#[inline]
fn dot2(a: &V2, b: &V2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

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
pub fn tri_local_cs(v: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3], [f64; 3]) {
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
pub fn bary_grads_2d(xs: &[f64; 3], ys: &[f64; 3]) -> ([V2; 3], f64) {
    let (x1, x2, x3) = (xs[0], xs[1], xs[2]);
    let (y1, y2, y3) = (ys[0], ys[1], ys[2]);
    let two_a = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2); // signed 2·Area
    let b = [y2-y3, y3-y1, y1-y2];
    let c = [x3-x2, x1-x3, x2-x1];

    // Sliver guard (2-D analogue of the tet floor): a near-collinear boundary
    // triangle has 2A → 0; floor it so ∇L stays finite.
    let mut sum_len = 0.0;
    for &[i, j] in &TRI_EDGE_LOCAL {
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

fn node_dist(xs: &[f64; 3], ys: &[f64; 3], i: usize, j: usize) -> f64 {
    ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2)).sqrt()
}

/// The 8 surface basis functions, in DOF order.
///
/// Built from the volume element's own generators on the triangle's three nodes
/// (see the module docs). The triangle has no fourth node, so `exps[3]` is zero
/// and no term gradients it — asserted below, because that is precisely the trace
/// property the construction relies on.
pub fn build_surface_basis(
    kind: BasisKind,
    owners: &[DofOwner],
    xs: &[f64; 3],
    ys: &[f64; 3],
) -> Vec<BasisFn> {
    let d = |i: usize, j: usize| node_dist(xs, ys, i, j);

    let edges: Vec<[BasisFn; 2]> = TRI_EDGE_LOCAL
        .iter()
        .map(|&[a, b]| edge_fns(kind, a, b, d(a, b)))
        .collect();
    let face = face_fns(0, 1, 2, d(0, 2), d(0, 1));

    let fns: Vec<BasisFn> = owners
        .iter()
        .map(|o| match *o {
            DofOwner::Edge { entity, k } => edges[entity as usize][k as usize].clone(),
            DofOwner::Face { k, .. } => face[k as usize].clone(),
            DofOwner::Cell { .. } => unreachable!("a triangle has no cell DOFs"),
        })
        .collect();
    debug_assert!(
        fns.iter().all(|f| f.terms.iter().all(|t| t.exps[3] == 0 && t.grad < 3)),
        "a surface function names the opposite vertex: it is not a tangential trace"
    );
    fns
}

/// Evaluate one surface function's tangential vector at barycentric `lam`.
#[inline]
fn eval_surface_fn(f: &BasisFn, lam: &[f64; 3], grads: &[V2; 3]) -> V2 {
    let mut v = [0.0_f64; 2];
    for t in &f.terms {
        let mut mono = f.scale * t.coeff;
        for (i, &e) in t.exps[..3].iter().enumerate() {
            for _ in 0..e {
                mono *= lam[i];
            }
        }
        let g = &grads[t.grad as usize];
        v[0] += mono * g[0];
        v[1] += mono * g[1];
    }
    v
}

/// Surface mass matrix `∫ φ_i·φ_j dA`, row-major n×n, for any surface basis.
///
/// Exact: a product of two functions is a sum of terms `c · L^e · (∇L_a·∇L_b)`
/// with constant gradients, and `∫ L^e dA / A` is the closed-form area
/// coefficient. The 2-D counterpart of `tet_assembly::element_stiff_mass`; the
/// Robin term needs no curl, so there is no stiffness half.
pub fn surface_mass(basis: &[BasisFn], grads: &[V2; 3], area: f64) -> Vec<f64> {
    let n = basis.len();
    let mut m = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let mut acc = 0.0;
            for ti in &basis[i].terms {
                for tj in &basis[j].terms {
                    let g = dot2(&grads[ti.grad as usize], &grads[tj.grad as usize]);
                    let e: [u8; 3] = std::array::from_fn(|k| ti.exps[k] + tj.exps[k]);
                    acc += ti.coeff * tj.coeff * g * (area_coeff_exps(e) * area);
                }
            }
            let val = basis[i].scale * basis[j].scale * acc;
            m[i * n + j] = val;
            m[j * n + i] = val;
        }
    }
    m
}

/// Surface Robin stiffness: `γ ∫ φ_i·φ_j dA`, row-major n×n with n = owners.len().
/// n is 8 at uniform order 2 and smaller where the minimum rule has reduced the
/// triangle's entities.
pub fn tri_stiff(
    kind: BasisKind,
    owners: &[DofOwner],
    glob_vertices: &[[f64; 3]; 3],
    gamma: C64,
) -> Vec<C64> {
    let (_, xs, ys) = tri_local_cs(glob_vertices);
    let (grads, two_a) = bary_grads_2d(&xs, &ys);
    let fns = build_surface_basis(kind, owners, &xs, &ys);
    let m = surface_mass(&fns, &grads, 0.5 * two_a.abs());
    m.iter().map(|&v| gamma * C64::from(v)).collect()
}

/// Surface excitation: `∫ φ_i·u_inc dA` by quadrature, an 8-vector.
/// `dpts[q] = [w, L1, L2, L3]`, `glob_uinc[q]` the incident field at that point.
pub fn tri_force(
    kind: BasisKind,
    owners: &[DofOwner],
    glob_vertices: &[[f64; 3]; 3],
    glob_uinc: &[[C64; 3]],
    dpts: &[[f64; 4]],
) -> Vec<C64> {
    let (frame, xs, ys) = tri_local_cs(glob_vertices);
    let (grads, two_a) = bary_grads_2d(&xs, &ys);
    let area = 0.5 * two_a.abs();
    let fns = build_surface_basis(kind, owners, &xs, &ys);

    // Incident field rotated into the local frame; only the tangential (x,y)
    // components pair with φ.
    let lcs_uinc: Vec<[C64; 2]> = glob_uinc
        .iter()
        .map(|c| {
            std::array::from_fn(|r| {
                C64::from(frame[r][0]) * c[0]
                    + C64::from(frame[r][1]) * c[1]
                    + C64::from(frame[r][2]) * c[2]
            })
        })
        .collect();

    let mut bvec = vec![C64::new(0.0, 0.0); fns.len()];
    for (fi, f) in fns.iter().enumerate() {
        let mut sum = C64::new(0.0, 0.0);
        for (qi, qp) in dpts.iter().enumerate() {
            let phi = eval_surface_fn(f, &[qp[1], qp[2], qp[3]], &grads);
            sum += C64::from(qp[0])
                * (C64::from(phi[0]) * lcs_uinc[qi][0] + C64::from(phi[1]) * lcs_uinc[qi][1]);
        }
        bvec[fi] = C64::from(area) * sum;
    }
    bvec
}
