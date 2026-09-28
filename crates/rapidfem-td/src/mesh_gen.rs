// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Structured tetrahedral mesh generation.
//!
//! A box mesher via the conforming Kuhn (Freudenthal) triangulation, six
//! tets per cell through a fixed main diagonal, which keeps shared cell faces
//! matched. [`structured_box_jittered`] perturbs the interior to produce
//! irregular meshes for validating the solver on non-uniform elements.

use rapidfem_core::mesh::Mesh;

/// Six tets per cell, all sharing the 0-7 main diagonal. Cell-corner index
/// is `x + 2y + 4z`.
const KUHN: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 6, 4, 7],
    [0, 4, 5, 7],
    [0, 5, 1, 7],
];

/// Tetrahedra of an `nx·ny·nz` grid of cells via the Kuhn triangulation.
fn kuhn_tets(nx: usize, ny: usize, nz: usize) -> Vec<[usize; 4]> {
    let (nxp, nyp) = (nx + 1, ny + 1);
    let node_id = |i: usize, j: usize, k: usize| (k * nyp + j) * nxp + i;
    let mut tets = Vec::with_capacity(nx * ny * nz * 6);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let corner = |c: usize| {
                    node_id(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1))
                };
                for kt in KUHN {
                    let mut t = [
                        corner(kt[0]),
                        corner(kt[1]),
                        corner(kt[2]),
                        corner(kt[3]),
                    ];
                    t.sort_unstable();
                    tets.push(t);
                }
            }
        }
    }
    tets
}

/// Tetrahedral mesh of the box `[0,lx] × [0,ly] × [0,lz]` with `nx·ny·nz`
/// cells, each split into 6 tets.
pub fn structured_box(
    nx: usize,
    ny: usize,
    nz: usize,
    lx: f64,
    ly: f64,
    lz: f64,
) -> Mesh {
    assert!(nx >= 1 && ny >= 1 && nz >= 1);
    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                nodes.push([
                    lx * i as f64 / nx as f64,
                    ly * j as f64 / ny as f64,
                    lz * k as f64 / nz as f64,
                ]);
            }
        }
    }
    Mesh::from_tets(nodes, kuhn_tets(nx, ny, nz))
}

/// Like [`structured_box`] but every node is pseudo-randomly displaced.
///
/// The box shape stays exact, a displacement component is zeroed whenever
/// the node lies on that axis's boundary face, so the cavity is unchanged,
/// but every tetrahedron becomes irregular and skewed. For validating the
/// solver on non-uniform meshes. `amplitude` is a fraction of the cell size
/// (keep `< 0.5` to avoid inverted elements).
pub fn structured_box_jittered(
    nx: usize,
    ny: usize,
    nz: usize,
    lx: f64,
    ly: f64,
    lz: f64,
    amplitude: f64,
    seed: u64,
) -> Mesh {
    assert!(nx >= 1 && ny >= 1 && nz >= 1);
    let (hx, hy, hz) =
        (lx / nx as f64, ly / ny as f64, lz / nz as f64);
    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                let mut p =
                    [hx * i as f64, hy * j as f64, hz * k as f64];
                let off = jitter_offset(i, j, k, seed);
                if i != 0 && i != nx {
                    p[0] += amplitude * hx * off[0];
                }
                if j != 0 && j != ny {
                    p[1] += amplitude * hy * off[1];
                }
                if k != 0 && k != nz {
                    p[2] += amplitude * hz * off[2];
                }
                nodes.push(p);
            }
        }
    }
    Mesh::from_tets(nodes, kuhn_tets(nx, ny, nz))
}

/// Deterministic pseudo-random displacement in `[-1,1]³` from a node index.
fn jitter_offset(i: usize, j: usize, k: usize, seed: u64) -> [f64; 3] {
    let mix = |mut x: u64| -> f64 {
        x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        (x as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    let base = seed
        .wrapping_add((i as u64).wrapping_mul(0x0100_0193))
        .wrapping_add((j as u64).wrapping_mul(0x0001_0000_01B3))
        .wrapping_add((k as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    [mix(base), mix(base ^ 0xAAAA), mix(base ^ 0x5555)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_mesh_is_conforming_and_fills_volume() {
        let m = structured_box(2, 2, 2, 1.0, 1.0, 1.0);
        assert_eq!(m.n_tets(), 8 * 6);
        for adj in &m.tri_to_tet {
            assert!(adj[0] != usize::MAX, "every face has an owner");
        }
        assert!((mesh_volume(&m) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jittered_box_keeps_shape_and_fills_volume() {
        // Jittering preserves the box: total volume unchanged, no inverted
        // elements, still conforming.
        let m = structured_box_jittered(2, 2, 2, 1.0, 1.0, 1.0, 0.3, 42);
        assert_eq!(m.n_tets(), 8 * 6);
        for adj in &m.tri_to_tet {
            assert!(adj[0] != usize::MAX);
        }
        assert!(
            (mesh_volume(&m) - 1.0).abs() < 1e-12,
            "jittered box volume = {}",
            mesh_volume(&m)
        );
    }

    fn mesh_volume(m: &Mesh) -> f64 {
        let mut vol = 0.0;
        for tet in &m.tets {
            let v: Vec<[f64; 3]> = tet.iter().map(|&n| m.nodes[n]).collect();
            let e = |a: [f64; 3]| {
                [a[0] - v[0][0], a[1] - v[0][1], a[2] - v[0][2]]
            };
            let (a, b, c) = (e(v[1]), e(v[2]), e(v[3]));
            let det = a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
            vol += det.abs() / 6.0;
        }
        vol
    }
}
