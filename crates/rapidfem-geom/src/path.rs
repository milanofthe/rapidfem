// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Paths for swept solids.

type P3 = [f64; 3];

/// The centripetal Catmull-Rom spline through `points` as a polyline,
/// `samples` points per span (the ends extended by reflection). Two points
/// stay a straight segment.
pub fn spline(points: &[P3], samples: usize) -> Vec<P3> {
    let n = points.len();
    if n < 3 || samples < 2 {
        return points.to_vec();
    }
    let sub = |a: P3, b: P3| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let add = |a: P3, b: P3| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let at = |i: isize| -> P3 {
        if i < 0 {
            add(points[0], sub(points[0], points[1]))
        } else if i as usize >= n {
            add(points[n - 1], sub(points[n - 1], points[n - 2]))
        } else {
            points[i as usize]
        }
    };
    // knot spacing: square root of the chord length
    let dt = |a: P3, b: P3| {
        let d = sub(b, a);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().sqrt().max(1e-300)
    };
    let lerp = |a: P3, b: P3, ta: f64, tb: f64, t: f64| {
        let (u, v) = ((tb - t) / (tb - ta), (t - ta) / (tb - ta));
        [u * a[0] + v * b[0], u * a[1] + v * b[1], u * a[2] + v * b[2]]
    };
    let mut out = vec![points[0]];
    for i in 0..n - 1 {
        let (p0, p1, p2, p3) = (at(i as isize - 1), at(i as isize), at(i as isize + 1), at(i as isize + 2));
        let t0 = 0.0;
        let t1 = t0 + dt(p0, p1);
        let t2 = t1 + dt(p1, p2);
        let t3 = t2 + dt(p2, p3);
        for k in 1..=samples {
            let t = t1 + (t2 - t1) * k as f64 / samples as f64;
            let a1 = lerp(p0, p1, t0, t1, t);
            let a2 = lerp(p1, p2, t1, t2, t);
            let a3 = lerp(p2, p3, t2, t3, t);
            let b1 = lerp(a1, a2, t0, t2, t);
            let b2 = lerp(a2, a3, t1, t3, t);
            out.push(if k == samples { p2 } else { lerp(b1, b2, t1, t2, t) });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spline_passes_through_its_points() {
        let pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]];
        let s = spline(&pts, 6);
        assert_eq!(s.len(), 1 + 3 * 6);
        for (i, p) in pts.iter().enumerate() {
            assert_eq!(s[i * 6], *p);
        }
        // a straight run stays straight
        let line = spline(&[[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], 4);
        assert!(line.iter().all(|p| p[1] == 0.0 && p[2] == 0.0));
        assert_eq!(spline(&pts[..2], 6), pts[..2].to_vec());
    }
}
