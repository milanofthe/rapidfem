// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Barycentric (natural-coordinate) integration coefficients.
//!
//! Integrals of products of barycentric coordinates over a simplex have a
//! classical closed form in terms of factorials (Eisenberg & Malvern, "On
//! finite element integration in natural coordinates", Int. J. Numer.
//! Methods Eng. 7 (1973) 574-575):
//!
//!   ∫_tet L₁^p L₂^q L₃^r L₄^s dV = (p! q! r! s!)/(p+q+r+s+3)! · 6V
//!   ∫_tri L₁^p L₂^q L₃^r   dA = (p! q! r!)/(p+q+r+2)!     · 2A
//!
//! The two helpers below return the mesh-independent prefactors used by the
//! Nédélec-2 element assembly. The same identity is re-derived from scratch
//! by symbolic integration in `derivations/nedelec2/barycentric.py`, whose
//! exact rationals back the golden test in `tests/coefficients_golden_test.rs`.

const FACTORIALS: [u64; 10] = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880];

/// Volume integration coefficient for the barycentric product L_a L_b L_c L_d.
/// Indices a,b,c,d are in range [0,4] where 0 means "not used" and 1-4 select
/// a vertex coordinate. Result: ∫_tet L_a L_b L_c L_d dV / (6V).
pub fn volume_coeff(a: usize, b: usize, c: usize, d: usize) -> f64 {
    // Count occurrences of each index 0-6 (only 1-4 matter)
    let mut klmn = [0usize; 7];
    klmn[a] += 1;
    klmn[b] += 1;
    klmn[c] += 1;
    klmn[d] += 1;
    let numerator = FACTORIALS[klmn[1]] * FACTORIALS[klmn[2]]
        * FACTORIALS[klmn[3]] * FACTORIALS[klmn[4]]
        * FACTORIALS[klmn[5]] * FACTORIALS[klmn[6]];
    let sum: usize = klmn[1..].iter().sum();
    let denominator = FACTORIALS[sum + 3];
    numerator as f64 / denominator as f64
}

/// Area integration coefficient for the barycentric product on a triangle.
/// Result: ∫_tri L_a L_b L_c L_d dA / A = 2 · (factorial products)/(sum+2)!
pub fn area_coeff(a: usize, b: usize, c: usize, d: usize) -> f64 {
    let mut klmn = [0usize; 7];
    klmn[a] += 1;
    klmn[b] += 1;
    klmn[c] += 1;
    klmn[d] += 1;
    let numerator = 2 * FACTORIALS[klmn[1]] * FACTORIALS[klmn[2]]
        * FACTORIALS[klmn[3]] * FACTORIALS[klmn[4]]
        * FACTORIALS[klmn[5]] * FACTORIALS[klmn[6]];
    let sum: usize = klmn[1..].iter().sum();
    let denominator = FACTORIALS[sum + 2];
    numerator as f64 / denominator as f64
}

/// Precomputed 5×5×5×5 volume coefficient cache (indices 0-4).
/// VOLUME_COEFF_CACHE[i][j][k][l] = volume_coeff(i,j,k,l)
/// Note: at runtime, multiply by 6*V for physical scaling.
pub struct VolumeCoeffCache {
    pub data: [[[[f64; 5]; 5]; 5]; 5],
}

impl VolumeCoeffCache {
    pub fn new() -> Self {
        let mut data = [[[[0.0f64; 5]; 5]; 5]; 5];
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    for l in 0..5 {
                        data[i][j][k][l] = volume_coeff(i, j, k, l);
                    }
                }
            }
        }
        VolumeCoeffCache { data }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.data[i][j][k][l]
    }
}

/// Precomputed 5×5×5×5 area coefficient cache.
pub struct AreaCoeffCache {
    pub data: [[[[f64; 5]; 5]; 5]; 5],
}

impl AreaCoeffCache {
    pub fn new() -> Self {
        let mut data = [[[[0.0f64; 5]; 5]; 5]; 5];
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    for l in 0..5 {
                        data[i][j][k][l] = area_coeff(i, j, k, l);
                    }
                }
            }
        }
        AreaCoeffCache { data }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.data[i][j][k][l]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_coeff_known_values() {
        // volume_coeff(0,0,0,0) = 1!/0!/0!/0!/0!/0! / 3! = 1/6
        // Actually with all indices = 0, klmn = [4,0,0,0,0,0,0]
        // Only klmn[1..] matter, so all are 0 → numerator = 1
        // sum = 0, denominator = 3! = 6
        assert!((volume_coeff(0, 0, 0, 0) - 1.0/6.0).abs() < 1e-15);

        // volume_coeff(1,1,0,0) → klmn = [2,2,0,0,0,0,0]
        // klmn[1]=2, rest 0 → num = 2! = 2, sum = 2, denom = 5! = 120
        assert!((volume_coeff(1, 1, 0, 0) - 2.0/120.0).abs() < 1e-15);

        // volume_coeff(1,2,3,4) → klmn = [0,1,1,1,1,0,0]
        // num = 1*1*1*1*1*1 = 1, sum = 4, denom = 7! = 5040
        assert!((volume_coeff(1, 2, 3, 4) - 1.0/5040.0).abs() < 1e-15);
    }

    #[test]
    fn test_area_coeff_known_values() {
        // area_coeff(0,0,0,0) → num = 2, sum = 0, denom = 2! = 2 → 1.0
        assert!((area_coeff(0, 0, 0, 0) - 1.0).abs() < 1e-15);

        // area_coeff(1,1,0,0) → klmn[1]=2, num = 2*2! = 4, sum = 2, denom = 4! = 24 → 4/24 = 1/6
        assert!((area_coeff(1, 1, 0, 0) - 1.0/6.0).abs() < 1e-15);
    }

    #[test]
    fn test_cache_matches_function() {
        let vc = VolumeCoeffCache::new();
        let ac = AreaCoeffCache::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    for l in 0..5 {
                        assert!((vc.get(i,j,k,l) - volume_coeff(i,j,k,l)).abs() < 1e-15);
                        assert!((ac.get(i,j,k,l) - area_coeff(i,j,k,l)).abs() < 1e-15);
                    }
                }
            }
        }
    }
}
