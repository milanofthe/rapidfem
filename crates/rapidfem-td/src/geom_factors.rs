// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Geometric factors, the affine map from the reference tetrahedron to each
//! physical element.
//!
//! For a tet with vertices `v0..v3` the map is
//! `x(r,s,t) = v0 + J·(r,s,t)`, with the constant Jacobian
//! `J = [v1-v0 | v2-v0 | v3-v0]`. The metric `J⁻¹` turns reference
//! derivatives into physical ones:
//! `∂u/∂x_i = Σ_k (J⁻¹)[k,i] · ∂u/∂ξ_k`.

use crate::constants::Field;
use rapidfem_core::geom::{cross, dot, sub};
use rapidfem_core::mesh::Mesh;

/// Per-element geometric factors of the reference→physical affine map.
#[derive(Clone, Copy, Debug)]
pub struct GeometricFactors {
    /// `v0`, image of the reference origin.
    pub origin: [Field; 3],
    /// Jacobian `J[i][k] = ∂x_i/∂ξ_k`; column `k` is `v_{k+1} - v0`.
    pub jacobian: [[Field; 3]; 3],
    /// Inverse Jacobian `J⁻¹`, the metric terms `∂ξ_k/∂x_i`.
    pub jacobian_inv: [[Field; 3]; 3],
    /// Signed determinant of `J`, six times the signed element volume.
    pub det: Field,
}

impl GeometricFactors {
    /// Build the geometric factors for a tet given its four vertices.
    pub fn for_tet(v: &[[Field; 3]; 4]) -> Self {
        let (c0, c1, c2) = (sub(v[1], v[0]), sub(v[2], v[0]), sub(v[3], v[0]));
        // J[i][k], row i, column k.
        let jacobian = [
            [c0[0], c1[0], c2[0]],
            [c0[1], c1[1], c2[1]],
            [c0[2], c1[2], c2[2]],
        ];
        let (jacobian_inv, det) = inv3(jacobian);
        GeometricFactors {
            origin: v[0],
            jacobian,
            jacobian_inv,
            det,
        }
    }

    /// Map a reference point `(r,s,t)` to physical coordinates.
    pub fn map(&self, xi: [Field; 3]) -> [Field; 3] {
        let mut x = self.origin;
        for i in 0..3 {
            for k in 0..3 {
                x[i] += self.jacobian[i][k] * xi[k];
            }
        }
        x
    }
}

/// Geometric factors for every tet of a mesh.
pub fn all_geometric_factors(mesh: &Mesh) -> Vec<GeometricFactors> {
    mesh.tets
        .iter()
        .map(|tet| {
            let v: [[Field; 3]; 4] = [
                mesh.nodes[tet[0]],
                mesh.nodes[tet[1]],
                mesh.nodes[tet[2]],
                mesh.nodes[tet[3]],
            ]
            .map(|p| p.map(|x| x as Field));
            GeometricFactors::for_tet(&v)
        })
        .collect()
}

/// Closed-form inverse and determinant of a 3x3 matrix: the columns of the
/// adjugate are the cross products of the row pairs.
fn inv3(m: [[Field; 3]; 3]) -> ([[Field; 3]; 3], Field) {
    let adj_cols = [cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1])];
    let det = dot(m[0], adj_cols[0]);
    let id = 1.0 / det;
    let inv: [[Field; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| adj_cols[j][i] * id));
    (inv, det)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dg_basis::ReferenceElement;

    const REF_TET: [[f64; 3]; 4] =
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];

    #[test]
    fn reference_tet_is_identity() {
        let g = GeometricFactors::for_tet(&REF_TET);
        assert!((g.det - 1.0).abs() < 1e-14);
        for i in 0..3 {
            for k in 0..3 {
                let want = if i == k { 1.0 } else { 0.0 };
                assert!((g.jacobian[i][k] - want).abs() < 1e-14);
                assert!((g.jacobian_inv[i][k] - want).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn inverse_jacobian_is_consistent() {
        // A sheared, scaled, translated tet.
        let v = [
            [1.0, 2.0, -1.0],
            [3.0, 2.5, -1.0],
            [1.5, 5.0, 0.5],
            [2.0, 2.0, 4.0],
        ];
        let g = GeometricFactors::for_tet(&v);
        // J · J⁻¹ = I
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += g.jacobian[i][k] * g.jacobian_inv[k][j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((acc - want).abs() < 1e-12, "J·J⁻¹ off at ({i},{j})");
            }
        }
    }

    #[test]
    fn det_matches_triple_product() {
        let v = [
            [1.0, 2.0, -1.0],
            [3.0, 2.5, -1.0],
            [1.5, 5.0, 0.5],
            [2.0, 2.0, 4.0],
        ];
        let g = GeometricFactors::for_tet(&v);
        // det J = (v1-v0) . ((v2-v0) x (v3-v0)), six times the volume.
        let (a, b, c) = (sub(v[1], v[0]), sub(v[2], v[0]), sub(v[3], v[0]));
        let triple = dot(a, cross(b, c));
        assert!((g.det - triple).abs() < 1e-12);
        assert!(g.det > 0.0);
    }

    #[test]
    fn det_flips_with_orientation() {
        // Swapping two vertices flips the sign of det(J), not its magnitude.
        let v = [REF_TET[0], REF_TET[2], REF_TET[1], REF_TET[3]];
        let g = GeometricFactors::for_tet(&v);
        assert!(g.det < 0.0, "swapped tet should have negative det");
        assert!((g.det + 1.0).abs() < 1e-14);
    }

    #[test]
    fn physical_derivatives_are_exact() {
        // End-to-end: reference operators + metric reproduce ∂/∂x,y,z exactly
        // for a physical polynomial on a sheared physical element.
        let v = [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 3.0],
        ];
        let g = GeometricFactors::for_tet(&v);
        let re = ReferenceElement::new(2);
        let n = re.n_nodes;
        let pn: Vec<[f64; 3]> = re.nodes.iter().map(|&xi| g.map(xi)).collect();

        // u = x·y + 2x + z²   ⇒   u_x = y+2,  u_y = x,  u_z = 2z.
        let u = |p: [f64; 3]| p[0] * p[1] + 2.0 * p[0] + p[2] * p[2];
        let un: Vec<f64> = pn.iter().map(|&p| u(p)).collect();
        let exacts: [&dyn Fn([f64; 3]) -> f64; 3] = [
            &|p: [f64; 3]| p[1] + 2.0,
            &|p: [f64; 3]| p[0],
            &|p: [f64; 3]| 2.0 * p[2],
        ];

        let dref = [&re.diff_r, &re.diff_s, &re.diff_t];
        for axis in 0..3 {
            for i in 0..n {
                // (D_x u)_i = sum_k Jinv[k][axis] sum_j D_ref[k][i,j] u_j
                let mut got = 0.0;
                for k in 0..3 {
                    let mut row = 0.0;
                    for j in 0..n {
                        row += dref[k][i * n + j] * un[j];
                    }
                    got += g.jacobian_inv[k][axis] * row;
                }
                let want = exacts[axis](pn[i]);
                assert!(
                    (got - want).abs() < 1e-9,
                    "axis {axis} node {i}: got {got}, want {want}"
                );
            }
        }
    }
}
