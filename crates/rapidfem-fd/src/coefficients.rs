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
//! Nédélec-2 element assembly. The same identity is reproduced by symbolic
//! integration in `derivations/nedelec2/barycentric.py`, whose
//! exact rationals back the golden test in `tests/coefficients_golden_test.rs`.

/// Factorials up to 16!, which is 2·(max element order) + 3 for orders well past
/// anything we run. Order p functions have barycentric degree p, so a mass
/// integrand reaches degree 2p and the volume denominator is (2p+3)!; order 6
/// still fits. u64 overflows at 21!, so the table cannot silently grow past that.
const FACTORIALS: [u64; 17] = [
    1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000, 20922789888000,
];

/// Volume integration coefficient for a barycentric monomial, by EXPONENTS.
///
///   ∫_tet L1^e1 L2^e2 L3^e3 L4^e4 dV / (6V)  =  (∏ e_i!) / (Σ e_i + 3)!
///
/// This is the general form (Eisenberg & Malvern); it takes the exponent
/// multi-index directly, so it is not tied to an element of any particular
/// order. See `derivations/nedelec2/barycentric.py`.
#[inline]
pub fn volume_coeff_exps(e: [u8; 4]) -> f64 {
    let mut num: u64 = 1;
    let mut sum: usize = 0;
    for &ei in &e {
        num *= FACTORIALS[ei as usize];
        sum += ei as usize;
    }
    num as f64 / FACTORIALS[sum + 3] as f64
}

/// Area integration coefficient for a barycentric monomial on a triangle, by
/// EXPONENTS. Returns ∫/A (not ∫/2A), matching the historic `area_coeff`:
///
///   ∫_tri L1^e1 L2^e2 L3^e3 dA / A  =  2 · (∏ e_i!) / (Σ e_i + 2)!
#[inline]
pub fn area_coeff_exps(e: [u8; 3]) -> f64 {
    let mut num: u64 = 2;
    let mut sum: usize = 0;
    for &ei in &e {
        num *= FACTORIALS[ei as usize];
        sum += ei as usize;
    }
    num as f64 / FACTORIALS[sum + 2] as f64
}

/// Volume coefficient by VERTEX INDICES: a,b,c,d in [0,4], where 0 means
/// "unused" and 1-4 select a barycentric coordinate. A thin wrapper over
/// `volume_coeff_exps` that converts the index list to an exponent multi-index;
/// kept because the golden test and the derivation are written against it.
pub fn volume_coeff(a: usize, b: usize, c: usize, d: usize) -> f64 {
    volume_coeff_exps(indices_to_exps4(a, b, c, d))
}

/// Area coefficient by vertex indices, same convention as `volume_coeff`.
///
/// Index 4 names a coordinate a triangle does not have. The historic formula
/// counted it anyway, in both the factorial product and the degree sum, so
/// `area_coeff(4, ..)` returns a value that is not a triangle integral at all.
/// `AreaCoeffCache` still fills those slots (its loop runs 0..5 on every axis)
/// and nothing reads them: the only caller, `tri_assembly_r2`, passes local
/// triangle nodes, i.e. indices 1-3. The behaviour is preserved verbatim rather
/// than "fixed", so this stays a refactor.
pub fn area_coeff(a: usize, b: usize, c: usize, d: usize) -> f64 {
    let e = indices_to_exps4(a, b, c, d);
    let mut num: u64 = 2;
    let mut sum: usize = 0;
    for &ei in &e {
        num *= FACTORIALS[ei as usize];
        sum += ei as usize;
    }
    num as f64 / FACTORIALS[sum + 2] as f64
}

/// Count how often each of the four barycentric coordinates is named. Index 0
/// means "unused" and is discarded.
#[inline]
fn indices_to_exps4(a: usize, b: usize, c: usize, d: usize) -> [u8; 4] {
    let mut e = [0u8; 4];
    for idx in [a, b, c, d] {
        if idx >= 1 && idx <= 4 {
            e[idx - 1] += 1;
        }
    }
    e
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
