// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! DG reference element on the unit tetrahedron.
//!
//! Reference tet: vertices `(0,0,0), (1,0,0), (0,1,0), (0,0,1)`, i.e.
//! `r,s,t ≥ 0` and `r+s+t ≤ 1`.
//!
//! A nodal (Lagrange) basis of order `p` is built on equispaced nodes. The
//! basis is expressed in monomials via the inverse Vandermonde, which lets
//! every reference operator be assembled in closed form:
//!
//! * the mass matrix from the exact monomial integral
//!   `∫ rᵃsᵇtᶜ dV = a!·b!·c! / (a+b+c+3)!`,
//! * the differentiation matrices from the analytic monomial derivatives.
//!
//! Equispaced nodes keep this first version simple; for `p ≳ 5` the
//! Vandermonde conditioning degrades and Warp&Blend nodes become worthwhile.

use crate::constants::Field;

/// Reference DG element of polynomial order `p` on the unit tetrahedron.
///
/// All matrices are `n_nodes × n_nodes`, stored row-major.
pub struct ReferenceElement {
    /// Polynomial order.
    pub order: usize,
    /// Node count, `Np = (p+1)(p+2)(p+3)/6`.
    pub n_nodes: usize,
    /// Node coordinates on the reference tet, `(r, s, t)`.
    pub nodes: Vec<[Field; 3]>,
    /// Mass matrix, `M[i,j] = ∫ φ_i φ_j dV`.
    pub mass: Vec<Field>,
    /// Inverse mass matrix.
    pub mass_inv: Vec<Field>,
    /// `∂/∂r` differentiation matrix: `(∂u/∂r)(node_i) = Σ_j diff_r[i,j] u_j`.
    pub diff_r: Vec<Field>,
    /// `∂/∂s` differentiation matrix.
    pub diff_s: Vec<Field>,
    /// `∂/∂t` differentiation matrix.
    pub diff_t: Vec<Field>,
    /// Face-node count per face, `Nfp = (p+1)(p+2)/2`.
    pub n_face_nodes: usize,
    /// Volume-node indices on each of the 4 local faces, aligned with
    /// `TET_FACE_LOCAL`: face 0 = (t=0), 1 = (r=0), 2 = (s=0), 3 = (r+s+t=1).
    pub face_nodes: [Vec<usize>; 4],
    /// Lift matrix, `Np x (4*Nfp)` row-major. Maps face-trace values to volume
    /// nodes (`M^-1 * surface integral of phi_i * trace`). Every face is
    /// parametrised as the unit right triangle (reference area 1/2), so the
    /// per-face physical scaling `Fscale = area_phys / (1/2)` is applied later
    /// by the RHS operator.
    pub lift: Vec<Field>,
    /// Per-face nodal surface-integration weights, `face_node_weights[f][m]`
    /// is `∮ φ_m dA` over the reference face (the unit right triangle), so
    /// they sum to `1/2`. A physical surface integral is
    /// `Σ_m 2·area_phys·weight[m]·F_m`.
    pub face_node_weights: [Vec<Field>; 4],
}

impl ReferenceElement {
    /// Build the reference element of order `p` (`p ≥ 1`).
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "DG reference element needs order >= 1");
        let monos = monomials(p);
        let nodes = equispaced_nodes(p);
        let n = monos.len();
        assert_eq!(n, nodes.len(), "monomial / node count mismatch");

        // Vandermonde V[node, mono] = mono(node); C = V^-1 gives the nodal
        // basis coefficients in the monomial basis: φ_i = Σ_m C[m,i]·mono_m.
        let mut vander = vec![0.0; n * n];
        for (ni, nd) in nodes.iter().enumerate() {
            for (mi, m) in monos.iter().enumerate() {
                vander[ni * n + mi] = eval_mono(*m, *nd);
            }
        }
        let c = invert(&vander, n);

        // Monomial Gram matrix G[m,n] = ∫ mono_m·mono_n dV (exact).
        let mut gram = vec![0.0; n * n];
        for (mi, m) in monos.iter().enumerate() {
            for (ni, q) in monos.iter().enumerate() {
                gram[mi * n + ni] =
                    mono_integral(m[0] + q[0], m[1] + q[1], m[2] + q[2]);
            }
        }
        // Mass M = Cᵀ G C.
        let mass = triple_product(&c, &gram, &c, n);
        let mass_inv = invert(&mass, n);

        // Derivative Vandermondes Vr/Vs/Vt, then D = V_d · C.
        let (mut vr, mut vs, mut vt) =
            (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
        for (ni, nd) in nodes.iter().enumerate() {
            for (mi, m) in monos.iter().enumerate() {
                vr[ni * n + mi] = eval_mono_d(*m, *nd, 0);
                vs[ni * n + mi] = eval_mono_d(*m, *nd, 1);
                vt[ni * n + mi] = eval_mono_d(*m, *nd, 2);
            }
        }
        let diff_r = matmul(&vr, &c, n);
        let diff_s = matmul(&vs, &c, n);
        let diff_t = matmul(&vt, &c, n);

        let (n_face_nodes, face_nodes, lift, face_node_weights) =
            build_lift(p, &mass_inv, n);

        ReferenceElement {
            order: p,
            n_nodes: n,
            nodes,
            mass,
            mass_inv,
            diff_r,
            diff_s,
            diff_t,
            n_face_nodes,
            face_nodes,
            lift,
            face_node_weights,
        }
    }
}

/// Monomial exponents `(a,b,c)` for total degree `≤ p`.
fn monomials(p: usize) -> Vec<[usize; 3]> {
    let mut v = Vec::new();
    for deg in 0..=p {
        for a in 0..=deg {
            for b in 0..=(deg - a) {
                v.push([a, b, deg - a - b]);
            }
        }
    }
    v
}

/// Equispaced nodes `(i,j,k)/p` with `i+j+k ≤ p` on the reference tet.
fn equispaced_nodes(p: usize) -> Vec<[Field; 3]> {
    let mut v = Vec::new();
    let pp = p as Field;
    for i in 0..=p {
        for j in 0..=(p - i) {
            for k in 0..=(p - i - j) {
                v.push([i as Field / pp, j as Field / pp, k as Field / pp]);
            }
        }
    }
    v
}

/// `n!` as an `Field`.
fn factorial(n: usize) -> Field {
    (1..=n).map(|k| k as Field).product()
}

/// Exact integral of `rᵃ sᵇ tᶜ` over the unit reference tetrahedron.
fn mono_integral(a: usize, b: usize, c: usize) -> Field {
    factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3)
}

/// Evaluate monomial `rᵃ sᵇ tᶜ` at a point.
fn eval_mono(m: [usize; 3], x: [Field; 3]) -> Field {
    x[0].powi(m[0] as i32) * x[1].powi(m[1] as i32) * x[2].powi(m[2] as i32)
}

/// Evaluate `∂/∂x_axis` of monomial `m` at a point (`axis` 0=r, 1=s, 2=t).
fn eval_mono_d(m: [usize; 3], x: [Field; 3], axis: usize) -> Field {
    let e = m[axis];
    if e == 0 {
        return 0.0;
    }
    let mut d = m;
    d[axis] -= 1;
    e as Field * eval_mono(d, x)
}

/// Invert an `n×n` row-major matrix via Gauss-Jordan with partial pivoting.
fn invert(src: &[Field], n: usize) -> Vec<Field> {
    let mut a = src.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        assert!(best > 1e-300, "singular matrix in invert()");
        if piv != col {
            for k in 0..n {
                a.swap(col * n + k, piv * n + k);
                inv.swap(col * n + k, piv * n + k);
            }
        }
        let d = a[col * n + col];
        for k in 0..n {
            a[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col];
            if f == 0.0 {
                continue;
            }
            for k in 0..n {
                a[r * n + k] -= f * a[col * n + k];
                inv[r * n + k] -= f * inv[col * n + k];
            }
        }
    }
    inv
}

/// `A · B` for `n×n` row-major matrices.
fn matmul(a: &[Field], b: &[Field], n: usize) -> Vec<Field> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// `Aᵀ · G · A` for `n×n` row-major matrices.
fn triple_product(a: &[Field], g: &[Field], a2: &[Field], n: usize) -> Vec<Field> {
    // (Aᵀ G) then · A
    let mut at_g = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let mut acc = 0.0;
            for m in 0..n {
                acc += a[m * n + i] * g[m * n + k];
            }
            at_g[i * n + k] = acc;
        }
    }
    matmul(&at_g, a2, n)
}

/// Integer node multi-indices `(i,j,k)`, same order as [`equispaced_nodes`].
fn equispaced_ijk(p: usize) -> Vec<[usize; 3]> {
    let mut v = Vec::new();
    for i in 0..=p {
        for j in 0..=(p - i) {
            for k in 0..=(p - i - j) {
                v.push([i, j, k]);
            }
        }
    }
    v
}

/// Exact integral of `u^a v^b` over the unit right triangle.
fn tri_integral(a: usize, b: usize) -> Field {
    factorial(a) * factorial(b) / factorial(a + b + 2)
}

/// `A·B` for general row-major matrices (`ar×ac` times `ac×bc`).
fn mat_mul(a: &[Field], ar: usize, ac: usize, b: &[Field], bc: usize) -> Vec<Field> {
    let mut c = vec![0.0; ar * bc];
    for i in 0..ar {
        for k in 0..ac {
            let aik = a[i * ac + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..bc {
                c[i * bc + j] += aik * b[k * bc + j];
            }
        }
    }
    c
}

/// Build the per-face node sets and the lift matrix.
fn build_lift(
    p: usize,
    mass_inv: &[Field],
    n: usize,
) -> (usize, [Vec<usize>; 4], Vec<Field>, [Vec<Field>; 4]) {
    let ijk = equispaced_ijk(p);
    // Face membership, aligned with TET_FACE_LOCAL.
    let on_face = |f: usize, c: [usize; 3]| match f {
        0 => c[2] == 0,
        1 => c[0] == 0,
        2 => c[1] == 0,
        3 => c[0] + c[1] + c[2] == p,
        _ => unreachable!(),
    };
    // 2D coordinate of a face node, drop the constrained index.
    let coord2d = |f: usize, c: [usize; 3]| -> [usize; 2] {
        match f {
            0 => [c[0], c[1]],
            1 => [c[1], c[2]],
            2 => [c[0], c[2]],
            3 => [c[0], c[1]],
            _ => unreachable!(),
        }
    };

    let nfp = (p + 1) * (p + 2) / 2;
    let mut face_nodes: [Vec<usize>; 4] =
        [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for (f, fnodes) in face_nodes.iter_mut().enumerate() {
        for (idx, &c) in ijk.iter().enumerate() {
            if on_face(f, c) {
                fnodes.push(idx);
            }
        }
        assert_eq!(fnodes.len(), nfp, "face {f} node count");
    }

    // 2D monomials (a,b) with a+b <= p.
    let monos2: Vec<[usize; 2]> = {
        let mut v = Vec::new();
        for deg in 0..=p {
            for a in 0..=deg {
                v.push([a, deg - a]);
            }
        }
        v
    };

    // Emat: Np x (4*Nfp). Column block f holds the face mass matrix scattered
    // onto the rows of face f's volume nodes.
    let cols = 4 * nfp;
    let mut emat = vec![0.0; n * cols];
    let mut face_node_weights: [Vec<Field>; 4] =
        [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let pf = p as Field;
    for f in 0..4 {
        // 2D node coordinates for this face.
        let c2: Vec<[Field; 2]> = face_nodes[f]
            .iter()
            .map(|&vi| {
                let c = coord2d(f, ijk[vi]);
                [c[0] as Field / pf, c[1] as Field / pf]
            })
            .collect();
        // 2D Vandermonde and its inverse.
        let mut v2 = vec![0.0; nfp * nfp];
        for (i, xy) in c2.iter().enumerate() {
            for (j, m) in monos2.iter().enumerate() {
                v2[i * nfp + j] =
                    xy[0].powi(m[0] as i32) * xy[1].powi(m[1] as i32);
            }
        }
        let c2inv = invert(&v2, nfp);
        // 2D monomial Gram matrix (exact).
        let mut g2 = vec![0.0; nfp * nfp];
        for (i, m) in monos2.iter().enumerate() {
            for (j, q) in monos2.iter().enumerate() {
                g2[i * nfp + j] = tri_integral(m[0] + q[0], m[1] + q[1]);
            }
        }
        // Face mass Mf = C2ᵀ G2 C2.
        let mf = triple_product(&c2inv, &g2, &c2inv, nfp);
        // Scatter into Emat.
        for a in 0..nfp {
            let vi = face_nodes[f][a];
            for m in 0..nfp {
                emat[vi * cols + f * nfp + m] = mf[a * nfp + m];
            }
        }
        // Nodal surface-integration weights, row sums of the face mass.
        face_node_weights[f] = (0..nfp)
            .map(|a| (0..nfp).map(|m| mf[a * nfp + m]).sum())
            .collect();
    }

    let lift = mat_mul(mass_inv, n, n, &emat, cols);
    (nfp, face_nodes, lift, face_node_weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn np(p: usize) -> usize {
        (p + 1) * (p + 2) * (p + 3) / 6
    }

    #[test]
    fn node_counts() {
        for p in 1..=4 {
            let re = ReferenceElement::new(p);
            assert_eq!(re.n_nodes, np(p), "Np wrong for p={p}");
        }
    }

    #[test]
    fn mass_is_partition_of_unity() {
        // Σ_ij M[i,j] = ∫ (Σφ_i)(Σφ_j) = ∫ 1·1 dV = 1/6 (unit tet volume).
        for p in 1..=4 {
            let re = ReferenceElement::new(p);
            let sum: f64 = re.mass.iter().sum();
            assert!(
                (sum - 1.0 / 6.0).abs() < 1e-11,
                "p={p}: Σ M = {sum}, expected 1/6"
            );
        }
    }

    #[test]
    fn mass_is_symmetric() {
        for p in 1..=4 {
            let re = ReferenceElement::new(p);
            let n = re.n_nodes;
            for i in 0..n {
                for j in 0..n {
                    assert!(
                        (re.mass[i * n + j] - re.mass[j * n + i]).abs() < 1e-13
                    );
                }
            }
        }
    }

    #[test]
    fn mass_inverse_is_correct() {
        for p in 1..=4 {
            let re = ReferenceElement::new(p);
            let n = re.n_nodes;
            let id = matmul(&re.mass, &re.mass_inv, n);
            for i in 0..n {
                for j in 0..n {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (id[i * n + j] - want).abs() < 1e-9,
                        "p={p}: M·M⁻¹ off at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn diff_rows_sum_to_zero() {
        // ∂(constant)/∂x = 0  ⇒  each row of every D sums to 0.
        for p in 1..=4 {
            let re = ReferenceElement::new(p);
            let n = re.n_nodes;
            for d in [&re.diff_r, &re.diff_s, &re.diff_t] {
                for i in 0..n {
                    let row: f64 = (0..n).map(|j| d[i * n + j]).sum();
                    assert!(row.abs() < 1e-9, "p={p}: D row {i} sum = {row}");
                }
            }
        }
    }

    #[test]
    fn diff_is_exact_on_polynomials() {
        // Differentiate a degree-p polynomial exactly at the nodes.
        let p = 3;
        let re = ReferenceElement::new(p);
        let n = re.n_nodes;
        // u = 1 + 2r + 3s - t + r·s + r·t² + s³
        let u = |r: f64, s: f64, t: f64| {
            1.0 + 2.0 * r + 3.0 * s - t + r * s + r * t * t + s * s * s
        };
        let du_dr = |_r: f64, s: f64, t: f64| 2.0 + s + t * t;
        let du_ds = |r: f64, s: f64, _t: f64| 3.0 + r + 3.0 * s * s;
        let du_dt = |r: f64, _s: f64, t: f64| -1.0 + 2.0 * r * t;

        let un: Vec<f64> =
            re.nodes.iter().map(|x| u(x[0], x[1], x[2])).collect();
        for (d, exact) in [
            (&re.diff_r, &du_dr as &dyn Fn(f64, f64, f64) -> f64),
            (&re.diff_s, &du_ds),
            (&re.diff_t, &du_dt),
        ] {
            for i in 0..n {
                let got: f64 = (0..n).map(|j| d[i * n + j] * un[j]).sum();
                let want = exact(re.nodes[i][0], re.nodes[i][1], re.nodes[i][2]);
                assert!(
                    (got - want).abs() < 1e-8,
                    "node {i}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn equispaced_vandermonde_conditioning() {
        // WP1.4: quantify the monomial-Vandermonde conditioning of the
        // equispaced node set across orders, to know the safe order range.
        let norm_inf = |m: &[f64], n: usize| -> f64 {
            (0..n)
                .map(|i| (0..n).map(|j| m[i * n + j].abs()).sum::<f64>())
                .fold(0.0_f64, f64::max)
        };
        for p in 1..=5 {
            let monos = monomials(p);
            let nodes = equispaced_nodes(p);
            let n = monos.len();
            let mut v = vec![0.0; n * n];
            for (ni, nd) in nodes.iter().enumerate() {
                for (mi, m) in monos.iter().enumerate() {
                    v[ni * n + mi] = eval_mono(*m, *nd);
                }
            }
            let c = invert(&v, n);
            let cond = norm_inf(&v, n) * norm_inf(&c, n);
            eprintln!("DIAG conditioning: p={p} Np={n} cond~{cond:.2e}");
            // f64 holds ~15 digits; cond < 1e10 still leaves ~5 safe digits.
            assert!(cond < 1e10, "p={p}: Vandermonde cond {cond:.2e} too high");
        }
    }

    #[test]
    fn lift_integrates_face_traces() {
        // 1ᵀ M (LIFT · trace) reduces to ∮ trace dA. For an all-ones trace on
        // one face this is the reference face area, 1/2 in the unit
        // right-triangle parametrisation used by every face.
        for p in 1..=4 {
            let re = ReferenceElement::new(p);
            let n = re.n_nodes;
            let nfp = re.n_face_nodes;
            let cols = 4 * nfp;
            for f in 0..4 {
                let mut trace = vec![0.0; cols];
                for m in 0..nfp {
                    trace[f * nfp + m] = 1.0;
                }
                let g: Vec<f64> = (0..n)
                    .map(|i| {
                        (0..cols)
                            .map(|c| re.lift[i * cols + c] * trace[c])
                            .sum()
                    })
                    .collect();
                let mut val = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        val += re.mass[i * n + j] * g[j];
                    }
                }
                assert!(
                    (val - 0.5).abs() < 1e-9,
                    "p={p} face {f}: ∮ = {val}, expected 1/2"
                );
            }
        }
    }
}
