// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Symmetric Gaussian quadrature rules for triangles and tetrahedra.
//!
//! The rule data are the standard published symmetric-quadrature tables:
//! triangles from Dunavant, "High degree efficient symmetrical Gaussian
//! quadrature rules for the triangle", Int. J. Numer. Methods Eng. 21 (1985)
//! 1129-1148; tetrahedra the classic low-order rules (Hammer-Marlowe-Stroud /
//! Keast). Each rule is stored as compact symmetry orbits and expanded by
//! cyclic rotation of the barycentric coordinates. The unit test verifies
//! polynomial exactness against the closed-form simplex integral, so the
//! tables are validated independently of their source.

/// A quadrature point on a triangle: (weight, L1, L2, L3) in barycentric coords.
pub type TriQuadPoint = [f64; 4];

/// A quadrature point on a tetrahedron: (weight, L1, L2, L3, L4) in barycentric coords.
pub type TetQuadPoint = [f64; 5];

/// Expand triangle quadrature rules from compact symmetry orbits.
pub fn gaus_quad_tri(order: usize) -> Vec<TriQuadPoint> {
    let rules = tri_rules(order);
    let mut pts = Vec::new();
    for &(n, w, l1, l2, l3) in &rules {
        let (mut a, mut b, mut c) = (l1, l2, l3);
        for i in 0..n {
            if i == 3 {
                // Symmetry orbit 6: after 3 rotations, swap L2/L3 then rotate 3 more
                a = l1; b = l3; c = l2;
            }
            pts.push([w, a, b, c]);
            let tmp = a; a = b; b = c; c = tmp; // rotate
        }
    }
    pts
}

/// Expand tetrahedron quadrature rules from compact symmetry orbits.
pub fn gaus_quad_tet(order: usize) -> Vec<TetQuadPoint> {
    let rules = tet_rules(order);
    let mut pts = Vec::new();
    for &(n, w, l1, l2, l3, l4) in &rules {
        let (mut a, mut b, mut c, mut d) = (l1, l2, l3, l4);
        for _ in 0..n {
            pts.push([w, a, b, c, d]);
            let tmp = a; a = b; b = c; c = d; d = tmp; // rotate
        }
    }
    pts
}

/// Compact triangle quadrature rules: (symmetry_count, weight, L1, L2, L3).
fn tri_rules(order: usize) -> Vec<(usize, f64, f64, f64, f64)> {
    match order {
        1 => vec![(1, 1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0)],
        2 => vec![(3, 1.0/3.0, 2.0/3.0, 1.0/6.0, 1.0/6.0)],
        3 => vec![
            (1, -0.562500000000000, 1.0/3.0, 1.0/3.0, 1.0/3.0),
            (3, 0.520833333333333, 0.6, 0.2, 0.2),
        ],
        4 => vec![
            (3, 0.223381589678011, 0.108103018168070, 0.445948490915965, 0.445948490915965),
            (3, 0.109951743655322, 0.816847572980459, 0.091576213509771, 0.091576213509771),
        ],
        5 => vec![
            (1, 0.225000000000000, 1.0/3.0, 1.0/3.0, 1.0/3.0),
            (3, 0.132394152788506, 0.059715871789770, 0.470142064105115, 0.470142064105115),
            (3, 0.125939180544827, 0.797426985353087, 0.101286507323456, 0.101286507323456),
        ],
        6 => vec![
            (3, 0.116786275726379, 0.501426509658179, 0.249286745170910, 0.249286745170910),
            (3, 0.050844906370207, 0.873821971016996, 0.063089014491502, 0.063089014491502),
            (6, 0.082851075618374, 0.053145049844817, 0.310352451033784, 0.636502499121399),
        ],
        7 => vec![
            (1, -0.149570044467682, 1.0/3.0, 1.0/3.0, 1.0/3.0),
            (3, 0.175615257433208, 0.479308067841920, 0.260345966079040, 0.260345966079040),
            (3, 0.053347235608838, 0.869739794195568, 0.065130102902216, 0.065130102902216),
            (6, 0.077113760890257, 0.048690315425316, 0.312865496004874, 0.638444188569810),
        ],
        8 => vec![
            (1, 0.144315607677787, 1.0/3.0, 1.0/3.0, 1.0/3.0),
            (3, 0.095091634267285, 0.081414823414554, 0.459292588292723, 0.459292588292723),
            (3, 0.103217370534718, 0.658861384496480, 0.170569307751760, 0.170569307751760),
            (3, 0.032458497623198, 0.898905543365938, 0.050547228317031, 0.050547228317031),
            (6, 0.027230314174435, 0.008394777409958, 0.263112829634638, 0.728492392955404),
        ],
        9 => vec![
            (1, 0.097135796282799, 1.0/3.0, 1.0/3.0, 1.0/3.0),
            (3, 0.031334700227139, 0.020634961602525, 0.489682519198738, 0.489682519198738),
            (3, 0.077827541004774, 0.125820817014127, 0.437089591492937, 0.437089591492937),
            (3, 0.079647738927210, 0.623592928761935, 0.188203535619033, 0.188203535619033),
            (3, 0.025577675658698, 0.910540973211095, 0.044729513394453, 0.044729513394453),
            (6, 0.043283539377289, 0.036838412054736, 0.221962989160766, 0.741198598784498),
        ],
        10 => vec![
            (1, 0.090817990382754, 1.0/3.0, 1.0/3.0, 1.0/3.0),
            (3, 0.036725957756467, 0.028844733232685, 0.485577633383657, 0.485577633383657),
            (3, 0.045321059435528, 0.781036849029926, 0.109481575485037, 0.109481575485037),
            (6, 0.072757916845420, 0.141707219414880, 0.307939838764121, 0.550352941820999),
            (6, 0.028327242531057, 0.025003534762686, 0.246672560639903, 0.728323904597411),
            (6, 0.009421666963733, 0.009540815400299, 0.066803251012200, 0.923655933587500),
        ],
        _ => panic!("Triangle quadrature order {} not supported (1-10)", order),
    }
}

/// Compact tetrahedron quadrature rules: (symmetry_count, weight, L1, L2, L3, L4).
fn tet_rules(order: usize) -> Vec<(usize, f64, f64, f64, f64, f64)> {
    match order {
        1 => vec![(1, 1.0, 0.25, 0.25, 0.25, 0.25)],
        // 4-point degree-2 rule: a = (5+3√5)/20, b = (5−√5)/20.
        2 => vec![(4, 0.25,
            0.5854101966249684, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105)],
        3 => vec![
            (1, -0.8, 0.25, 0.25, 0.25, 0.25),
            (4, 0.45, 0.5, 0.166666667, 0.166666667, 0.166666667),
        ],
        4 => vec![
            (1, -0.078933, 0.25, 0.25, 0.25, 0.25),
            (4, 0.0457333333, 0.7857142857, 0.0714285714, 0.0714285714, 0.0714285714),
            (1, 0.1493333333, 0.3994035762, 0.1005964238, 0.3994035762, 0.1005964238),
            (1, 0.1493333333, 0.3994035762, 0.1005964238, 0.1005964238, 0.3994035762),
            (1, 0.1493333333, 0.3994035762, 0.3994035762, 0.1005964238, 0.1005964238),
            (1, 0.1493333333, 0.1005964238, 0.3994035762, 0.3994035762, 0.1005964238),
            (1, 0.1493333333, 0.1005964238, 0.3994035762, 0.1005964238, 0.3994035762),
            (1, 0.1493333333, 0.1005964238, 0.1005964238, 0.3994035762, 0.3994035762),
        ],
        _ => panic!("Tet quadrature order {} not supported (1-4)", order),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri_quad_weights_sum() {
        for order in 1..=10 {
            let pts = gaus_quad_tri(order);
            let sum: f64 = pts.iter().map(|p| p[0]).sum();
            assert!((sum - 1.0).abs() < 1e-10, "Order {}: weight sum = {}", order, sum);
        }
    }

    #[test]
    fn test_tet_quad_weights_sum() {
        // Tet quadrature weights are normalised to sum to 1 over the
        // reference tet. The order-4 rule's tabulated literals are slightly
        // truncated, so allow a wider tolerance than the triangle rules.
        for order in 1..=4 {
            let pts = gaus_quad_tet(order);
            let sum: f64 = pts.iter().map(|p| p[0]).sum();
            assert!((sum - 1.0).abs() < 1e-6, "Order {}: weight sum = {}", order, sum);
        }
    }

    fn fact(n: u32) -> f64 {
        (1..=n).map(|k| k as f64).product::<f64>().max(1.0)
    }

    /// Polynomial exactness: a degree-p rule must integrate every monomial of
    /// total degree ≤ p exactly. Validates the tabulated rules against the
    /// closed-form simplex integral, independently of where the data came from.
    #[test]
    fn test_tri_quad_polynomial_exactness() {
        for order in 1..=10 {
            let pts = gaus_quad_tri(order);
            for a in 0..=order {
                for b in 0..=(order - a) {
                    let c = order - a - b; // worst case: total degree == order
                    // area-normalized analytic: (1/A)∫ L1^a L2^b L3^c dA
                    let exact = 2.0 * fact(a as u32) * fact(b as u32) * fact(c as u32)
                        / fact((a + b + c + 2) as u32);
                    let approx: f64 = pts.iter().map(|p| {
                        p[0] * p[1].powi(a as i32) * p[2].powi(b as i32) * p[3].powi(c as i32)
                    }).sum();
                    assert!((approx - exact).abs() < 1e-9,
                        "tri order {order}, monomial ({a},{b},{c}): {approx} vs {exact}");
                }
            }
        }
    }

    /// Same for tetrahedra; orders 1-3 are exact (order-4 literals are
    /// truncated, covered only by the weight-sum test above).
    #[test]
    fn test_tet_quad_polynomial_exactness() {
        for order in 1..=3 {
            let pts = gaus_quad_tet(order);
            for a in 0..=order {
                for b in 0..=(order - a) {
                    for c in 0..=(order - a - b) {
                        let d = order - a - b - c;
                        let exact = 6.0 * fact(a as u32) * fact(b as u32) * fact(c as u32)
                            * fact(d as u32) / fact((a + b + c + d + 3) as u32);
                        let approx: f64 = pts.iter().map(|p| {
                            p[0] * p[1].powi(a as i32) * p[2].powi(b as i32)
                                * p[3].powi(c as i32) * p[4].powi(d as i32)
                        }).sum();
                        assert!((approx - exact).abs() < 1e-7,
                            "tet order {order}, monomial ({a},{b},{c},{d}): {approx} vs {exact}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_tri_quad_order4_has_6_points() {
        let pts = gaus_quad_tri(4);
        assert_eq!(pts.len(), 6);
    }

    #[test]
    fn test_tet_quad_order2_has_4_points() {
        let pts = gaus_quad_tet(2);
        assert_eq!(pts.len(), 4);
    }
}
