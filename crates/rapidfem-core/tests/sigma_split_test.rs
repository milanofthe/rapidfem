// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

// The σ-split used by the cached frequency sweep must reproduce the full
// lossy-dielectric form exactly:
//   εr*(ω) = [εr·(1 − j·tanδ)]  +  (−j/(ω·ε₀))·σ
//   full      wo_sigma part          sigma part (scaled per frequency)

use num_complex::Complex64 as C64;
use rapidfem_core::constants::EPS0;
use rapidfem_core::materials::{
    build_sigma_tensors, material_tensors, Dispersion,
    Material,
};

fn mat(er: f64, tand: f64, cond: f64, tets: Vec<usize>) -> Material {
    Material {
        er,
        ur: 1.0,
        tand,
        cond,
        cond_diag: None,
        tet_indices: tets,
        er_diag: None,
        ur_diag: None,
        dispersion: Dispersion::None,
    }
}

#[test]
fn wo_sigma_plus_scaled_sigma_equals_full_tensors() {
    // SG13G2-like: lossy oxide, EPI and substrate with bulk conductivity.
    let materials = vec![
        mat(4.1, 0.01, 0.0, vec![0]),
        mat(11.9, 0.0, 5.0, vec![1]),
        mat(11.9, 0.0, 2.0, vec![2]),
    ];
    for freq in [5.0e8, 2.0e9, 3.0e10] {
        let w = 2.0 * std::f64::consts::PI * freq;
        let (full, _) = material_tensors(3, Some(&materials), freq, true, None);
        let (base, _) = material_tensors(3, Some(&materials), freq, false, None);
        let sigma = build_sigma_tensors(3, &materials);
        let scale = C64::new(0.0, -1.0) / C64::from(w * EPS0);
        for ti in 0..3 {
            for i in 0..3 {
                for j in 0..3 {
                    let recomposed = base[ti][i][j] + scale * sigma[ti][i][j];
                    let d = (full[ti][i][j] - recomposed).norm();
                    let s = full[ti][i][j].norm().max(1e-300);
                    assert!(
                        d / s < 1e-14,
                        "tet {ti} [{i}][{j}] f={freq:.1e}: full {:?} vs recomposed {:?}",
                        full[ti][i][j],
                        recomposed
                    );
                }
            }
        }
    }
}

#[test]
fn anisotropic_cond_diag_lands_on_the_diagonal() {
    // Homogenised via array: weak lateral, strong vertical conduction.
    let mut m = mat(1.0, 0.0, 0.0, vec![0]);
    m.cond_diag = Some([3.143e5, 3.143e5, 3.143e6]);
    let freq = 1.0e9;
    let w = 2.0 * std::f64::consts::PI * freq;
    let sigma = build_sigma_tensors(1, &[m.clone_for_test()]);
    let (full, _) = material_tensors(1, Some(&[m]), freq, true, None);
    for (k, s) in [3.143e5, 3.143e5, 3.143e6].iter().enumerate() {
        assert!((sigma[0][k][k] - C64::from(*s)).norm() < 1e-9);
        let want_im = -s / (w * EPS0);
        assert!(
            (full[0][k][k].im - want_im).abs() / want_im.abs() < 1e-14,
            "axis {k}: {} vs {}",
            full[0][k][k].im,
            want_im
        );
    }
    assert_eq!(sigma[0][0][1], C64::new(0.0, 0.0));
}

trait CloneForTest {
    fn clone_for_test(&self) -> Material;
}
impl CloneForTest for Material {
    fn clone_for_test(&self) -> Material {
        Material {
            er: self.er,
            ur: self.ur,
            tand: self.tand,
            cond: self.cond,
            cond_diag: self.cond_diag,
            tet_indices: self.tet_indices.clone(),
            er_diag: self.er_diag,
            ur_diag: self.ur_diag,
            dispersion: Dispersion::None,
        }
    }
}

#[test]
fn sigma_tensors_are_zero_without_conductivity() {
    let materials = vec![mat(4.1, 0.01, 0.0, vec![0])];
    let sigma = build_sigma_tensors(1, &materials);
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(sigma[0][i][j], C64::new(0.0, 0.0));
        }
    }
}
