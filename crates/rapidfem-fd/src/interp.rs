// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Field reconstruction: evaluate the FEM E-field and its curl at a point.
//!
//! The solution vector holds DOF coefficients of the canonical R2 basis. The
//! field inside a tet is the weighted sum of its 20 basis functions evaluated
//! at the point; the curl follows from ∇×(s·∇L_g) = ∇s × ∇L_g. Both reuse the
//! *same* basis definition as the assembly (`tet_assembly_r2::build_basis`),
//! so reconstruction and assembly cannot drift apart.
//!
//! Sign convention: the reconstructed field carries a global minus relative to
//! `build_basis`. A global basis sign is physically immaterial (it cancels in
//! the assembled system), but it must match the modal-overlap convention used
//! by `sparam` and the port excitation. `RECON_SIGN` pins that choice in one
//! place; the self-validating curl test is invariant to it.

use num_complex::Complex64 as C64;
use crate::mesh::Mesh;
use crate::basis::Nedelec2Basis;
use crate::tet_assembly_r2::{barycentric_grads, build_basis, BasisFn};

type V3 = [f64; 3];

/// Global sign matching the reconstruction to the `sparam`/excitation
/// convention (see module docs). Physically immaterial; pinned here.
const RECON_SIGN: f64 = -1.0;

#[inline]
fn cross3(a: &V3, b: &V3) -> V3 {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}

/// Barycentric L_i of local node i at point `p`: L_i = δ_{i0} + ∇L_i·(p − v0).
#[inline]
fn lambdas_at(grads: &[V3; 4], v0: &V3, p: &V3) -> [f64; 4] {
    let d = [p[0]-v0[0], p[1]-v0[1], p[2]-v0[2]];
    std::array::from_fn(|i| {
        (if i == 0 { 1.0 } else { 0.0 })
            + grads[i][0]*d[0] + grads[i][1]*d[1] + grads[i][2]*d[2]
    })
}

/// Per-tet geometry shared by field and curl evaluation: the four ∇L_i, the
/// 20 canonical basis functions (in DOF order), and node 0 as the affine base.
fn tet_basis(mesh: &Mesh, tet_idx: usize) -> ([V3; 4], Vec<BasisFn>, V3) {
    let tet = &mesh.tets[tet_idx];
    let xs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][0]);
    let ys: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][1]);
    let zs: [f64; 4] = std::array::from_fn(|i| mesh.nodes[tet[i]][2]);
    let (grads, _six_v) = barycentric_grads(&xs, &ys, &zs);

    let tet_edges = &mesh.tet_to_edge[tet_idx];
    let edge_lengths: [f64; 6] = std::array::from_fn(|i| mesh.edge_lengths[tet_edges[i]]);
    let global_edge_nodes: [[usize; 2]; 6] = std::array::from_fn(|i| mesh.edges[tet_edges[i]]);
    let edge_map = crate::basis::local_mapping(tet, &global_edge_nodes);

    let tet_tris = &mesh.tet_to_tri[tet_idx];
    let global_tri_nodes: [[usize; 3]; 4] = std::array::from_fn(|i| mesh.tris[tet_tris[i]]);
    let tri_map = crate::basis::local_mapping_tri(tet, &global_tri_nodes);

    let node_dist = |i: usize, j: usize| -> f64 {
        ((xs[i]-xs[j]).powi(2) + (ys[i]-ys[j]).powi(2) + (zs[i]-zs[j]).powi(2)).sqrt()
    };
    let fns = build_basis(&edge_lengths, &edge_map, &tri_map, &node_dist);
    (grads, fns, mesh.nodes[tet[0]])
}

/// Evaluate the FEM E-field at point (x,y,z) inside tet `tet_idx`, as the
/// DOF-weighted sum of the 20 canonical R2 basis functions: each function is
/// `scale·Σ coeff·L_p·L_q·∇L_g`, so the field is a simple polynomial sum.
pub fn eval_field_in_tet(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    tet_idx: usize,
    x: f64, y: f64, z: f64,
) -> (C64, C64, C64) {
    let (grads, fns, v0) = tet_basis(mesh, tet_idx);
    let lam = lambdas_at(&grads, &v0, &[x, y, z]);
    let field_ids = &basis.tet_to_field[tet_idx];

    let mut e = [C64::new(0.0, 0.0); 3];
    for (i, bf) in fns.iter().enumerate() {
        let dof = solution[field_ids[i]];
        for t in &bf.terms {
            // term value = scale·coeff·L_p·L_q·∇L_g
            let s = RECON_SIGN * bf.scale * t.coeff * lam[t.mono[0]] * lam[t.mono[1]];
            let g = &grads[t.grad];
            for k in 0..3 {
                e[k] += dof * C64::from(s * g[k]);
            }
        }
    }
    (e[0], e[1], e[2])
}

/// Spatial hash grid for fast point-in-tet lookup.
pub struct TetGrid {
    cells: hashbrown::HashMap<(i32, i32, i32), Vec<usize>>,
    cell_size: f64,
}

impl TetGrid {
    /// Build a spatial grid from the mesh. Each tet is assigned to the cell containing its centroid.
    pub fn new(mesh: &Mesh) -> Self {
        // Compute bounding box
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for node in &mesh.nodes {
            for k in 0..3 { min[k] = min[k].min(node[k]); max[k] = max[k].max(node[k]); }
        }
        let diag = ((max[0]-min[0]).powi(2) + (max[1]-min[1]).powi(2) + (max[2]-min[2]).powi(2)).sqrt();
        let cell_size = diag / (mesh.n_tets() as f64).cbrt().max(2.0);

        let mut cells: hashbrown::HashMap<(i32, i32, i32), Vec<usize>> = hashbrown::HashMap::new();
        for itet in 0..mesh.n_tets() {
            let tet = &mesh.tets[itet];
            let cx = (mesh.nodes[tet[0]][0] + mesh.nodes[tet[1]][0] + mesh.nodes[tet[2]][0] + mesh.nodes[tet[3]][0]) / 4.0;
            let cy = (mesh.nodes[tet[0]][1] + mesh.nodes[tet[1]][1] + mesh.nodes[tet[2]][1] + mesh.nodes[tet[3]][1]) / 4.0;
            let cz = (mesh.nodes[tet[0]][2] + mesh.nodes[tet[1]][2] + mesh.nodes[tet[2]][2] + mesh.nodes[tet[3]][2]) / 4.0;
            let key = ((cx / cell_size).floor() as i32, (cy / cell_size).floor() as i32, (cz / cell_size).floor() as i32);
            cells.entry(key).or_default().push(itet);
        }

        TetGrid { cells, cell_size }
    }

    /// Find the tet containing a point using the spatial grid. Falls back to brute force if not found.
    pub fn find_containing_tet(&self, mesh: &Mesh, x: f64, y: f64, z: f64) -> Option<usize> {
        let cs = self.cell_size;
        let cx = (x / cs).floor() as i32;
        let cy = (y / cs).floor() as i32;
        let cz = (z / cs).floor() as i32;

        // Search 3x3x3 neighborhood
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(tets) = self.cells.get(&(cx+dx, cy+dy, cz+dz)) {
                        for &itet in tets {
                            if point_in_tet(mesh, itet, x, y, z) {
                                return Some(itet);
                            }
                        }
                    }
                }
            }
        }
        // Fallback: brute force (handles edge cases)
        find_containing_tet_brute(mesh, x, y, z)
    }
}

fn point_in_tet(mesh: &Mesh, itet: usize, x: f64, y: f64, z: f64) -> bool {
    let eps = crate::constants::POINT_IN_TET_EPS;
    let tet = &mesh.tets[itet];
    let v1 = mesh.nodes[tet[0]];
    let v2 = mesh.nodes[tet[1]];
    let v3 = mesh.nodes[tet[2]];
    let v4 = mesh.nodes[tet[3]];

    let m00 = v2[0]-v1[0]; let m01 = v3[0]-v1[0]; let m02 = v4[0]-v1[0];
    let m10 = v2[1]-v1[1]; let m11 = v3[1]-v1[1]; let m12 = v4[1]-v1[1];
    let m20 = v2[2]-v1[2]; let m21 = v3[2]-v1[2]; let m22 = v4[2]-v1[2];

    let det = m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20);
    if det.abs() < crate::constants::SINGULAR_EPS { return false; }
    let inv_det = 1.0 / det;

    let dx = x - v1[0];
    let dy = y - v1[1];
    let dz = z - v1[2];

    let u = ((m11*m22-m12*m21)*dx + (m02*m21-m01*m22)*dy + (m01*m12-m02*m11)*dz) * inv_det;
    let v = ((m12*m20-m10*m22)*dx + (m00*m22-m02*m20)*dy + (m02*m10-m00*m12)*dz) * inv_det;
    let w = ((m10*m21-m11*m20)*dx + (m01*m20-m00*m21)*dy + (m00*m11-m01*m10)*dz) * inv_det;

    u >= -eps && v >= -eps && w >= -eps && u + v + w <= 1.0 + eps
}

/// Brute-force fallback for find_containing_tet.
fn find_containing_tet_brute(mesh: &Mesh, x: f64, y: f64, z: f64) -> Option<usize> {
    for itet in 0..mesh.n_tets() {
        if point_in_tet(mesh, itet, x, y, z) {
            return Some(itet);
        }
    }
    None
}

/// Find the tet containing a point (brute force, for backward compatibility).
pub fn find_containing_tet(mesh: &Mesh, x: f64, y: f64, z: f64) -> Option<usize> {
    find_containing_tet_brute(mesh, x, y, z)
}

/// Analytic curl of the FEM E-field inside a known tet at point `(x, y, z)`.
///
/// Each canonical basis function is `scale·Σ coeff·L_p·L_q·∇L_g`, so by
/// `∇×(s·∇L_g) = ∇s × ∇L_g` with ∇s = coeff·(L_q·∇L_p + L_p·∇L_q):
///
///   ∇×φ = scale·coeff·[ L_q·(∇L_p×∇L_g) + L_p·(∇L_q×∇L_g) ]
///
/// linear in position via the L's (no constant-per-tet approximation). Used by
/// the error estimator, far-field integration, and H = ∇×E / (jωμ).
pub fn eval_curl_in_tet(
    mesh: &Mesh,
    basis: &Nedelec2Basis,
    solution: &[C64],
    tet_idx: usize,
    x: f64, y: f64, z: f64,
) -> [C64; 3] {
    let (grads, fns, v0) = tet_basis(mesh, tet_idx);
    let lam = lambdas_at(&grads, &v0, &[x, y, z]);
    let field_ids = &basis.tet_to_field[tet_idx];

    let mut curl = [C64::new(0.0, 0.0); 3];
    for (i, bf) in fns.iter().enumerate() {
        let dof = solution[field_ids[i]];
        for t in &bf.terms {
            let (p, q, g) = (t.mono[0], t.mono[1], t.grad);
            let cp = cross3(&grads[p], &grads[g]); // ∇L_p × ∇L_g, weight L_q
            let cq = cross3(&grads[q], &grads[g]); // ∇L_q × ∇L_g, weight L_p
            let w = RECON_SIGN * bf.scale * t.coeff;
            for k in 0..3 {
                curl[k] += dof * C64::from(w * (lam[q] * cp[k] + lam[p] * cq[k]));
            }
        }
    }
    curl
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::Nedelec2Basis;
    use crate::mesh::Mesh;

    /// Build a single-tet mesh with non-degenerate vertices.
    fn single_tet_mesh() -> Mesh {
        let nodes = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let tets = vec![[0, 1, 2, 3]];
        Mesh::from_tets(nodes, tets)
    }

    /// Numerical 6-point central-difference curl of `eval_field_in_tet`. Used
    /// as a ground-truth reference for the analytic curl: the Nédélec-2 field
    /// is at most quadratic in position, so its curl is at most linear, and
    /// central differences are EXACT (to FP rounding) for linear functions.
    fn numerical_curl(
        mesh: &Mesh,
        basis: &Nedelec2Basis,
        solution: &[C64],
        tet_idx: usize,
        x: f64, y: f64, z: f64,
        h: f64,
    ) -> [C64; 3] {
        let eval = |x, y, z| eval_field_in_tet(mesh, basis, solution, tet_idx, x, y, z);
        let (_, eyp_x, ezp_x) = eval(x + h, y, z);
        let (_, eym_x, ezm_x) = eval(x - h, y, z);
        let (exp_y, _, ezp_y) = eval(x, y + h, z);
        let (exm_y, _, ezm_y) = eval(x, y - h, z);
        let (exp_z, eyp_z, _) = eval(x, y, z + h);
        let (exm_z, eym_z, _) = eval(x, y, z - h);
        let inv2h = C64::from(0.5 / h);
        let d_ez_dy = (ezp_y - ezm_y) * inv2h;
        let d_ey_dz = (eyp_z - eym_z) * inv2h;
        let d_ex_dz = (exp_z - exm_z) * inv2h;
        let d_ez_dx = (ezp_x - ezm_x) * inv2h;
        let d_ey_dx = (eyp_x - eym_x) * inv2h;
        let d_ex_dy = (exp_y - exm_y) * inv2h;
        [
            d_ez_dy - d_ey_dz,
            d_ex_dz - d_ez_dx,
            d_ey_dx - d_ex_dy,
        ]
    }

    /// Analytic curl must agree with the FD reference to FP precision at any
    /// point inside the tet, for any DOF configuration. Sweeps several
    /// (x, y, z) and a deterministic-pseudo-random DOF vector touching every
    /// basis function (6 edges × 2 modes + 4 faces × 2 modes = 20 DOFs).
    #[test]
    fn analytic_curl_matches_numerical() {
        let mesh = single_tet_mesh();
        let basis = Nedelec2Basis::new(&mesh);
        assert_eq!(basis.n_field, 20);

        // Deterministic seed, every DOF gets a distinct, non-trivial complex value.
        let solution: Vec<C64> = (0..20)
            .map(|i| C64::new(0.37 + 0.11 * i as f64, -0.21 + 0.07 * i as f64))
            .collect();

        // Sample at several interior points, including off-centre ones that
        // expose the linear position dependence.
        let pts = [
            (0.25, 0.25, 0.25),  // centroid
            (0.10, 0.20, 0.50),
            (0.50, 0.10, 0.10),
            (0.05, 0.05, 0.85),
        ];
        let h = 1e-3;
        for &(x, y, z) in &pts {
            let analytic = eval_curl_in_tet(&mesh, &basis, &solution, 0, x, y, z);
            let numerical = numerical_curl(&mesh, &basis, &solution, 0, x, y, z, h);
            for k in 0..3 {
                let diff = (analytic[k] - numerical[k]).norm();
                let scale = analytic[k].norm().max(numerical[k].norm()).max(1e-12);
                assert!(
                    diff / scale < 1e-6,
                    "component {k} at ({x},{y},{z}): analytic={:?}, numerical={:?}, rel_err={}",
                    analytic[k], numerical[k], diff / scale,
                );
            }
        }
    }
}
