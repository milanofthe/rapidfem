// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Planar regions of a layout: polygon sets with holes, their union,
//! difference and offset, by i_overlay, and the via-array merge built on them. A region is a list of
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

/// The region shrunk inward by `d` (mitered): its complement in a frame
/// around it, grown by `d`, taken away.
pub fn shrink(r: &Region, d: f64) -> Region {
    let [x0, y0, x1, y1] = bbox(r);
    let m = 2.0 * d;
    let frame: Region = vec![vec![vec![[x0 - m, y0 - m], [x1 + m, y0 - m], [x1 + m, y1 + m], [x0 - m, y1 + m]]]];
    difference(r, &offset(&difference(&frame, r), d))
}

/// The area of a shape (outer contour minus its holes).
fn area(shape: &[Vec<P2>]) -> f64 {
    let a = |c: &Vec<P2>| signed_area2(c).abs() / 2.0;
    shape.first().map_or(0.0, a) - shape.iter().skip(1).map(a).sum::<f64>()
}

fn inside(p: P2, c: &[P2]) -> bool {
    let mut odd = false;
    for k in 0..c.len() {
        let (a, b) = (c[k], c[(k + 1) % c.len()]);
        if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]) {
            odd = !odd;
        }
    }
    odd
}

/// The via cells closer than `spacing` to each other merged into one shape
/// per array, as gds2palace's `merge_via_array` does: each cell grown by
/// half the spacing plus 0.01 um, the union shrunk back. Each shape comes with
/// its fill factor, the area of the cells inside it over its own area.
pub fn merge_vias(cells: &[Vec<P2>], spacing: f64) -> Vec<(Vec<Vec<P2>>, f64)> {
    let o = spacing / 2.0 + 0.01e-6;
    let r: Region = cells.iter().map(|c| vec![c.clone()]).collect();
    let merged = shrink(&offset(&union(&r), o), o);
    merged
        .into_iter()
        .map(|shape| {
            let covered: f64 = cells
                .iter()
                .filter(|c| {
                    let n = c.len() as f64;
                    let centre = [c.iter().map(|p| p[0]).sum::<f64>() / n, c.iter().map(|p| p[1]).sum::<f64>() / n];
                    inside(centre, &shape[0]) && !shape[1..].iter().any(|h| inside(centre, h))
                })
                .map(|c| signed_area2(c).abs() / 2.0)
                .sum();
            let ff = (covered / area(&shape)).min(1.0);
            (shape, ff)
        })
        .collect()
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
    fn a_via_array_merges_into_one_block_with_its_fill_factor() {
        // 3 x 3 cells of 1 x 1 at a pitch of 2: one 5 x 5 block, 9/25 filled
        let cells: Vec<Vec<P2>> = (0..9)
            .map(|k| {
                let (x, y) = (2.0 * (k % 3) as f64, 2.0 * (k / 3) as f64);
                vec![[x, y], [x + 1.0, y], [x + 1.0, y + 1.0], [x, y + 1.0]]
            })
            .collect();
        let merged = merge_vias(&cells, 1.5);
        assert_eq!(merged.len(), 1);
        let (shape, ff) = &merged[0];
        assert!((super::area(shape) - 25.0).abs() < 1e-6, "{}", super::area(shape));
        assert!((ff - 9.0 / 25.0).abs() < 1e-6, "{ff}");
        // a spacing below the gap leaves the cells apart
        assert_eq!(merge_vias(&cells, 0.5).len(), 9);
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
