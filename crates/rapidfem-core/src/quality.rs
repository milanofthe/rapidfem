// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Mesh element-quality metrics for conditioning diagnostics.
//!
//! Slivers (near-flat tets with O(1) edges but vanishing volume) are the worst
//! case for vector-FEM conditioning: the canonical R2 element's condition
//! number scales like (1/q)² in the normalized volume
//!
//!   q = 6V / h_mean³        (≈ 0.6 regular, → 0 sliver)
//!
//! (see `derivations/conditioning/`). This module measures q per tet so a bad
//! mesh is flagged at load time rather than silently producing an
//! ill-conditioned — or NaN-poisoned — system. It is diagnostic only; the
//! assembly floors degenerate volumes independently (`SLIVER_NORMVOL_FLOOR`).

use crate::constants::SLIVER_NORMVOL_WARN;
use crate::mesh::Mesh;

/// Normalized volume q = 6V / h_mean³ of a tetrahedron (h_mean = mean edge
/// length). Returns 0 for a fully degenerate tet.
pub fn tet_normalized_volume(v: &[[f64; 3]; 4]) -> f64 {
    let sub = |a: usize, b: usize| [v[b][0] - v[a][0], v[b][1] - v[a][1], v[b][2] - v[a][2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    let six_v = dot(sub(0, 1), cross(sub(0, 2), sub(0, 3))).abs();

    let mut sum_len = 0.0;
    let pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for (a, b) in pairs {
        let e = sub(a, b);
        sum_len += dot(e, e).sqrt();
    }
    let h_mean = sum_len / 6.0;
    if h_mean <= 0.0 { return 0.0; }
    six_v / (h_mean * h_mean * h_mean)
}

/// Summary of a mesh's element quality.
pub struct MeshQuality {
    pub n_tets: usize,
    pub min_q: f64,
    pub p01_q: f64,
    /// Tets below the warning threshold (a real conditioning concern).
    pub n_warn: usize,
}

/// Assess the per-tet normalized volume of a mesh.
pub fn assess(mesh: &Mesh) -> MeshQuality {
    let n = mesh.n_tets();
    if n == 0 {
        return MeshQuality { n_tets: 0, min_q: 1.0, p01_q: 1.0, n_warn: 0 };
    }
    let mut qs: Vec<f64> = Vec::with_capacity(n);
    let mut n_warn = 0;
    for tet in &mesh.tets {
        let v = [mesh.nodes[tet[0]], mesh.nodes[tet[1]], mesh.nodes[tet[2]], mesh.nodes[tet[3]]];
        let q = tet_normalized_volume(&v);
        if q < SLIVER_NORMVOL_WARN { n_warn += 1; }
        qs.push(q);
    }
    qs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p01 = qs[(n as f64 * 0.01) as usize];
    MeshQuality { n_tets: n, min_q: qs[0], p01_q: p01, n_warn }
}

impl MeshQuality {
    /// Emit a one-line warning if any tet is below the conditioning threshold.
    /// Returns true if a warning was emitted.
    pub fn warn_if_poor(&self) -> bool {
        if self.n_warn > 0 {
            eprintln!(
                "WARNING: {} of {} tets have normalized volume q < {:.0e} \
                 (min q = {:.2e}); the FEM system will be ill-conditioned. \
                 Consider remeshing / sliver removal.",
                self.n_warn, self.n_tets, SLIVER_NORMVOL_WARN, self.min_q
            );
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_tet_has_unit_order_q() {
        // Near-regular tet → q ≈ 0.5–0.7.
        let v = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                 [0.5, 0.866_025, 0.0], [0.5, 0.288_675, 0.816_497]];
        let q = tet_normalized_volume(&v);
        assert!(q > 0.4 && q < 0.8, "regular tet q = {q}");
    }

    #[test]
    fn sliver_tet_has_tiny_q() {
        // Apex lifted by only 1e-4 → a sliver; q should be ~1e-4.
        let v = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0], [0.333, 0.333, 1e-4]];
        let q = tet_normalized_volume(&v);
        assert!(q < 1e-3, "sliver q = {q}");
        assert!(q > 0.0, "sliver q must stay positive");
    }

    #[test]
    fn degenerate_tet_is_zero() {
        // All four nodes coplanar → q = 0.
        let v = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]];
        assert_eq!(tet_normalized_volume(&v), 0.0);
    }

    #[test]
    fn assess_flags_a_sliver_in_the_mesh() {
        // Two tets sharing face {0,1,2}: one healthy (apex high), one a sliver
        // (apex barely off the base plane).
        let nodes = vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.3, 0.3, 0.8],     // healthy apex
            [0.3, 0.3, -1e-7],   // sliver apex
        ];
        let tets = vec![[0, 1, 2, 3], [0, 1, 2, 4]];
        let mesh = Mesh::from_tets(nodes, tets);
        let q = assess(&mesh);
        assert_eq!(q.n_tets, 2);
        assert_eq!(q.n_warn, 1, "exactly the sliver should be flagged");
        assert!(q.min_q < SLIVER_NORMVOL_WARN);
        assert!(q.warn_if_poor());
    }
}
