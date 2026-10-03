// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Planar regions of a layout: polygon sets with holes, their union,
//! difference and outward offset, by i_overlay. A region is a list of
//! shapes, a shape its outer contour followed by its holes.

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;

pub type P2 = [f64; 2];
pub type Region = Vec<Vec<Vec<P2>>>;

/// Gds2palace / gdstk offset convention: a miter longer than this many
/// offset distances is bevelled.
pub const MITER_LIMIT: f64 = 2.0;

fn overlay(a: &Region, b: &Region, rule: OverlayRule) -> Region {
    a.overlay(b, rule, FillRule::NonZero)
}

/// The union of possibly overlapping polygons.
pub fn union(r: &Region) -> Region {
    overlay(r, &Region::new(), OverlayRule::Union)
}

/// `a` without `b`.
pub fn difference(a: &Region, b: &Region) -> Region {
    overlay(a, b, OverlayRule::Difference)
}

/// Twice the signed area of a contour (positive counter-clockwise).
fn signed_area2(c: &[P2]) -> f64 {
    (0..c.len()).map(|k| {
        let (a, b) = (c[k], c[(k + 1) % c.len()]);
        a[0] * b[1] - b[0] * a[1]
    }).sum()
}

/// The region grown outward by `d` with mitered corners (bevelled beyond
/// [`MITER_LIMIT`]): the union of the region, a strip of width `d` outside
/// each edge and the miter wedge at each corner.
pub fn offset(r: &Region, d: f64) -> Region {
    let mut parts: Region = r.clone();
    for shape in r {
        for (k, c) in shape.iter().enumerate() {
            let n = c.len();
            if n < 3 {
                continue;
            }
            // the right-hand normal points away from the material for a
            // counter-clockwise outer contour and a clockwise hole
            let ccw = signed_area2(c) > 0.0;
            let away = if (k == 0) == ccw { 1.0 } else { -1.0 };
            let normal = |a: P2, b: P2| -> Option<P2> {
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let l = dx.hypot(dy);
                (l > 0.0).then(|| [away * dy / l, -away * dx / l])
            };
            for i in 0..n {
                let (a, b, c2) = (c[(i + n - 1) % n], c[i], c[(i + 1) % n]);
                let (Some(n1), Some(n2)) = (normal(a, b), normal(b, c2)) else { continue };
                let off = |p: P2, m: P2, s: f64| [p[0] + s * m[0], p[1] + s * m[1]];
                // the strip outside edge b..c2
                parts.push(vec![vec![b, c2, off(c2, n2, d), off(b, n2, d)]]);
                // the corner at b: miter point, or a bevel past the limit
                let cosine = n1[0] * n2[0] + n1[1] * n2[1];
                let miter = [n1[0] + n2[0], n1[1] + n2[1]];
                let wedge = if 1.0 + cosine > 2.0 / (MITER_LIMIT * MITER_LIMIT) {
                    vec![b, off(b, n1, d), off(b, miter, d / (1.0 + cosine)), off(b, n2, d)]
                } else {
                    vec![b, off(b, n1, d), off(b, n2, d)]
                };
                parts.push(vec![wedge]);
            }
        }
    }
    union(&parts)
}

/// The axis-aligned bounding box `(xmin, ymin, xmax, ymax)` of a region.
pub fn bbox(r: &Region) -> [f64; 4] {
    let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
    for p in r.iter().flatten().flatten() {
        b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(r: &Region) -> f64 {
        r.iter().flatten().map(|c| signed_area2(c)).sum::<f64>().abs() / 2.0
    }

    #[test]
    fn square_offset_is_mitered() {
        let sq: Region = vec![vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]];
        let grown = offset(&sq, 0.1);
        assert!((area(&grown) - 1.44).abs() < 1e-6, "{}", area(&grown));
        let cw: Region = vec![vec![vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]]];
        assert!((area(&offset(&cw, 0.1)) - 1.44).abs() < 1e-6, "{}", area(&offset(&cw, 0.1)));
    }

    #[test]
    fn ring_offset_narrows_the_hole() {
        let ring: Region = vec![vec![
            vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            vec![[1.0, 1.0], [1.0, 3.0], [3.0, 3.0], [3.0, 1.0]],
        ]];
        let grown = offset(&union(&ring), 0.5);
        // outer 5 x 5, hole 1 x 1
        assert!((area(&grown) - 24.0).abs() < 1e-5, "{}", area(&grown));
    }
}
