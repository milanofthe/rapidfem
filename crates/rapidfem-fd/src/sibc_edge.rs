// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Edge correction of the surface impedance of a good conductor.
//!
//! A per-face Leontovich impedance Zs misses the loss of the current crowding
//! into a conductor's convex edges: within a few skin depths δ of an edge the
//! field enters from two faces, and the ratio of the tangential E along the
//! edge to the surface current along it rises to about 2 Zs at the edge. The
//! ratio is universal in r/δ (r the distance to the edge) once the conductor
//! is a few δ thick: `G_TABLE`, from the 2D magneto-quasistatic solution of
//! rectangular conductors (`derivations/sibc_edge/`), the same within 2 % for
//! aspect ratios 1:3 to 1:20 and t/δ from 14 to 50.
//!
//! The impedance of the current along the edge is Zs·g(r/δ); the current
//! across it keeps Zs. On a triangle the factor is averaged over the
//! triangle's distances to the edge (uniformly: weighting the edge with the
//! r^(-2/3) of a singular field over-corrects by 5 to 12 % on meshes of 2 to
//! 5 δ, the uniform average stays within 3.5 % of the 2D reference from 1.5 δ
//! meshes up, `test_strip_conductor_resistance.py`).
//!
//! Below a few skin depths across, the corners of the cross-section
//! interact and the profile no longer holds; the correction fades in between
//! 3 and 4 δ of the conductor's local size ([`fade`]: the distance through
//! the metal along either face normal at the edge), where `rfic.build`
//! meshes the metal as a volume conductor.

use crate::mesh::Mesh;
use num_complex::Complex64 as C64;

/// g(r/δ): (r/δ, Re g, Im g), from the 2D reference at t/δ = 50.
const G_TABLE: [(f64, f64, f64); 15] = [
    (0.0, 2.200, -0.050),
    (0.02, 2.168, -0.072),
    (0.05, 2.063, -0.125),
    (0.1, 1.915, -0.165),
    (0.2, 1.752, -0.241),
    (0.3, 1.620, -0.268),
    (0.5, 1.428, -0.269),
    (0.75, 1.282, -0.261),
    (1.0, 1.180, -0.234),
    (1.5, 1.058, -0.172),
    (2.0, 0.999, -0.119),
    (3.0, 0.974, -0.034),
    (4.0, 0.988, -0.005),
    (6.0, 1.001, -0.002),
    (8.0, 1.0, 0.0),
];

/// Beyond this many skin depths from an edge the impedance is Zs.
pub const REACH: f64 = 8.0;

/// Two faces meeting at more than this angle between their normals make an
/// edge.
const EDGE_ANGLE_DEG: f64 = 30.0;

pub fn g(u: f64) -> C64 {
    if u >= REACH {
        return C64::new(1.0, 0.0);
    }
    let k = G_TABLE.iter().position(|e| e.0 > u).unwrap_or(G_TABLE.len() - 1).max(1);
    let (a, b) = (G_TABLE[k - 1], G_TABLE[k]);
    let s = ((u - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
    C64::new(a.1 + s * (b.1 - a.1), a.2 + s * (b.2 - a.2))
}

/// One triangle near an edge: the edge direction, the face normal, the
/// distances of its corners to the edge and the conductor's local size
/// (mesh units; infinite beyond the reach of the fade).
#[derive(Clone, Debug)]
struct TriNearEdge {
    dir: [f64; 3],
    normal: [f64; 3],
    r: [f64; 3],
    size: f64,
}

/// The triangles of a conductor surface next to its convex edges.
#[derive(Clone, Debug, Default)]
pub struct EdgeProfile {
    /// Per triangle of the surface (in its order).
    tris: Vec<Option<TriNearEdge>>,
}

impl EdgeProfile {
    /// The convex edges of the one-sided surface `tri_ids` (boundary
    /// triangles, the conductor behind them) and the triangles within
    /// `max_reach` (mesh units) of one; `None` without edges.
    pub fn build(mesh: &Mesh, tri_ids: &[usize], max_reach: f64) -> Option<EdgeProfile> {
        let one_sided: Vec<bool> = tri_ids.iter().map(|&t| mesh.tri_to_tet[t][1] == usize::MAX).collect();
        // unit normal into the conductor (away from the one tet) and centroid
        let geo: Vec<([f64; 3], [f64; 3])> = tri_ids
            .iter()
            .map(|&t| {
                let [a, b, c] = mesh.tris[t].map(|v| mesh.nodes[v]);
                let n = unit(cross(sub(b, a), sub(c, a)));
                let centre: [f64; 3] = std::array::from_fn(|k| (a[k] + b[k] + c[k]) / 3.0);
                let tet = mesh.tets[mesh.tri_to_tet[t][0]];
                let tc: [f64; 3] = std::array::from_fn(|k| tet.iter().map(|&v| mesh.nodes[v][k]).sum::<f64>() / 4.0);
                let n = if dot(n, sub(tc, centre)) > 0.0 { n.map(|x| -x) } else { n };
                (n, centre)
            })
            .collect();
        // mesh edge -> the surface triangles on it
        let mut on_edge: hashbrown::HashMap<usize, Vec<usize>> = hashbrown::HashMap::new();
        for (k, &t) in tri_ids.iter().enumerate() {
            if one_sided[k] {
                for &e in &mesh.tri_to_edge[t] {
                    on_edge.entry(e).or_default().push(k);
                }
            }
        }
        let cos_max = EDGE_ANGLE_DEG.to_radians().cos();
        let mut segments: Vec<[[f64; 3]; 2]> = Vec::new();
        let mut seg_normals: Vec<[[f64; 3]; 2]> = Vec::new();
        for (&e, ks) in &on_edge {
            if let [ka, kb] = ks[..] {
                let (na, ca) = geo[ka];
                let (nb, cb) = geo[kb];
                // convex: the other face lies behind this one (on the metal side)
                if dot(na, nb) < cos_max && dot(na, sub(cb, ca)) > 0.0 {
                    let [p, q] = mesh.edges[e];
                    segments.push([mesh.nodes[p], mesh.nodes[q]]);
                    seg_normals.push([na, nb]);
                }
            }
        }
        if segments.is_empty() {
            return None;
        }
        // bucket the segments on a grid of cell `max_reach`
        let cell = max_reach.max(1e-12);
        let key = |p: [f64; 3]| -> [i64; 3] { p.map(|x| (x / cell).floor() as i64) };
        let mut grid: hashbrown::HashMap<[i64; 3], Vec<usize>> = hashbrown::HashMap::new();
        for (i, s) in segments.iter().enumerate() {
            let (lo, hi) = (key(min3(s[0], s[1])), key(max3(s[0], s[1])));
            for x in lo[0] - 1..=hi[0] + 1 {
                for y in lo[1] - 1..=hi[1] + 1 {
                    for z in lo[2] - 1..=hi[2] + 1 {
                        grid.entry([x, y, z]).or_default().push(i);
                    }
                }
            }
        }
        // The conductor's local size, for the fade: a ray into the metal
        // along either face normal at the edge, to the surface opposite.
        let delta_max = max_reach / REACH;
        let cap = FADE_FULL * delta_max * 1.05;
        let surface: Vec<[[f64; 3]; 3]> = tri_ids.iter().map(|&t| mesh.tris[t].map(|v| mesh.nodes[v])).collect();
        let rays = RayGrid::new(&surface);
        let tris = tri_ids
            .iter()
            .enumerate()
            .map(|(k, &t)| {
                if !one_sided[k] {
                    return None;
                }
                let verts = mesh.tris[t].map(|v| mesh.nodes[v]);
                let near = grid.get(&key(geo[k].1))?;
                // the edge nearest to the centroid decides the direction
                let (best, _) = near
                    .iter()
                    .map(|&i| (i, seg_dist(geo[k].1, segments[i])))
                    .min_by(|a, b| a.1.total_cmp(&b.1))?;
                let s = segments[best];
                let r = verts.map(|v| near.iter().map(|&i| seg_dist(v, segments[i])).fold(f64::INFINITY, f64::min));
                if r.iter().cloned().fold(f64::INFINITY, f64::min) > max_reach {
                    return None;
                }
                let (n, c) = geo[k];
                // from the edge point next to the triangle, just inside the
                // metal, across the cross-section along either face normal
                let [na, nb] = seg_normals[best];
                let foot = seg_foot(c, s);
                let inward = unit([na[0] + nb[0], na[1] + nb[1], na[2] + nb[2]]);
                let eps = 1e-6 * norm(sub(s[1], s[0])).max(cap);
                let origin: [f64; 3] = std::array::from_fn(|i| foot[i] + eps * inward[i]);
                let size = [na, nb]
                    .iter()
                    .map(|&m| rays.distance(&surface, origin, m, usize::MAX, cap))
                    .fold(f64::INFINITY, f64::min);
                Some(TriNearEdge { dir: unit(sub(s[1], s[0])), normal: n, r, size })
            })
            .collect();
        Some(EdgeProfile { tris })
    }

    /// The admittance tensor γ·[(1/ḡ)·êê + (P - êê)] of triangle `k` (P the
    /// tangential projector), for the scalar Robin coefficient `gamma` and
    /// the skin depth `delta` (mesh units); `None` where it is `gamma`.
    pub fn tensor(&self, k: usize, gamma: C64, delta: f64) -> Option<[[C64; 3]; 3]> {
        let t = self.tris.get(k)?.as_ref()?;
        let fade = fade(t.size / delta);
        if fade <= 0.0 || t.r.iter().cloned().fold(f64::INFINITY, f64::min) > REACH * delta {
            return None;
        }
        let gbar = C64::new(1.0, 0.0) + (average_g(t.r, delta) - 1.0) * fade;
        let (e, n) = (t.dir, t.normal);
        let along = gamma / gbar;
        Some(std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let id = if i == j { 1.0 } else { 0.0 };
                let p = id - n[i] * n[j] - e[i] * e[j];
                gamma * C64::from(p) + along * C64::from(e[i] * e[j])
            })
        }))
    }
}

/// The conductor size (in δ) from which the correction holds in full, and
/// below which it is gone. Under about 4 δ across, the walls' own
/// finite-thickness term (coth) already carries part of the crowding, and
/// the two together overshoot by up to 18 % (a 2 x 3 µm strip at 10 to
/// 20 GHz); `rfic.build` meshes such metal as a volume conductor.
const FADE_FULL: f64 = 4.0;
const FADE_NONE: f64 = 3.0;

/// The share of the edge correction a conductor `size_over_delta` skin
/// depths across takes: a smoothstep from `FADE_NONE` to `FADE_FULL`.
pub fn fade(size_over_delta: f64) -> f64 {
    let s = ((size_over_delta - FADE_NONE) / (FADE_FULL - FADE_NONE)).clamp(0.0, 1.0);
    s * s * (3.0 - 2.0 * s)
}

/// g(r/δ) averaged over a triangle whose distance to the edge runs linearly
/// between its corner values `r`.
fn average_g(r: [f64; 3], delta: f64) -> C64 {
    let mut r = r;
    r.sort_by(f64::total_cmp);
    let [r0, r1, r2] = r;
    // Density of a linear function over a triangle: rising on [r0, r1],
    // falling on [r1, r2]; both pieces integrated by Gauss-Legendre, the
    // singular weight through r = r0 + (b - r0)·x³.
    let (xs, ws) = gauss_legendre_16();
    let mut num = C64::new(0.0, 0.0);
    let mut den = 0.0;
    let span = (r2 - r0).max(1e-300);
    let density = |x: f64| -> f64 {
        if x <= r1 {
            if r1 > r0 { (x - r0) / (r1 - r0) } else { 1.0 }
        } else if r2 > r1 {
            (r2 - x) / (r2 - r1)
        } else {
            1.0
        }
    };
    let pieces: [(f64, f64); 2] = [(r0, r1), (r1, r2)];
    for (a, b) in pieces {
        if b - a <= 1e-15 * span {
            continue;
        }
        for (x, w) in xs.iter().zip(ws.iter()) {
            let t = 0.5 * (x + 1.0);
            // x³ substitution clusters the points at the edge
            let (rr, jac) = (a + (b - a) * t * t * t, (b - a) * 3.0 * t * t * 0.5);
            let m = w * jac * density(rr);
            num += g(rr / delta) * m;
            den += m;
        }
    }
    if den > 0.0 { num / den } else { g(r0 / delta) }
}

/// The surface triangles bucketed on a grid, for rays from the surface into
/// the conductor.
struct RayGrid {
    cell: f64,
    cells: hashbrown::HashMap<[i64; 3], Vec<usize>>,
}

impl RayGrid {
    fn new(tris: &[[[f64; 3]; 3]]) -> RayGrid {
        let mean_edge = tris.iter().map(|t| norm(sub(t[1], t[0]))).sum::<f64>() / tris.len().max(1) as f64;
        let cell = mean_edge.max(1e-12);
        let mut cells: hashbrown::HashMap<[i64; 3], Vec<usize>> = hashbrown::HashMap::new();
        for (i, t) in tris.iter().enumerate() {
            let lo = min3(min3(t[0], t[1]), t[2]).map(|x| (x / cell).floor() as i64);
            let hi = max3(max3(t[0], t[1]), t[2]).map(|x| (x / cell).floor() as i64);
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        cells.entry([x, y, z]).or_default().push(i);
                    }
                }
            }
        }
        RayGrid { cell, cells }
    }

    /// Distance from `origin` along `dir` to the first triangle other than
    /// `skip`, infinite beyond `cap`.
    fn distance(&self, tris: &[[[f64; 3]; 3]], origin: [f64; 3], dir: [f64; 3], skip: usize, cap: f64) -> f64 {
        let mut seen = hashbrown::HashSet::new();
        let mut best = f64::INFINITY;
        let steps = (cap / (0.5 * self.cell)).ceil() as usize + 1;
        for i in 0..=steps {
            let s = (i as f64 * 0.5 * self.cell).min(cap);
            let p: [f64; 3] = std::array::from_fn(|k| origin[k] + s * dir[k]);
            let key = p.map(|x| (x / self.cell).floor() as i64);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(list) = self.cells.get(&[key[0] + dx, key[1] + dy, key[2] + dz]) else { continue };
                        for &t in list {
                            if t != skip && seen.insert(t)
                                && let Some(d) = ray_tri(origin, dir, tris[t]) {
                                    best = best.min(d);
                                }
                        }
                    }
                }
            }
            if best <= s {
                break;
            }
        }
        if best <= cap { best } else { f64::INFINITY }
    }
}

/// Möller-Trumbore: the distance along `dir` from `o` to the triangle, if
/// it is hit in front.
fn ray_tri(o: [f64; 3], dir: [f64; 3], t: [[f64; 3]; 3]) -> Option<f64> {
    let (e1, e2) = (sub(t[1], t[0]), sub(t[2], t[0]));
    let p = cross(dir, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-14 * dot(e1, e1) {
        return None;
    }
    let s = sub(o, t[0]);
    let u = dot(s, p) / det;
    let q = cross(s, e1);
    let v = dot(dir, q) / det;
    let d = dot(e2, q) / det;
    let tol = 1e-9;
    (u >= -tol && v >= -tol && u + v <= 1.0 + tol && d > 1e-9 * norm(e1)).then_some(d)
}

fn gauss_legendre_16() -> ([f64; 16], [f64; 16]) {
    let x = [
        -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.755_404_408_355_003,
        -0.6178762444026438, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
        0.0950125098376374, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
        0.755_404_408_355_003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499,
    ];
    let w = [
        0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
        0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
        0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
        0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541,
    ];
    (x, w)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    let l = norm(a).max(1e-300);
    a.map(|x| x / l)
}
fn min3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| a[k].min(b[k]))
}
fn max3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| a[k].max(b[k]))
}
fn seg_foot(p: [f64; 3], s: [[f64; 3]; 2]) -> [f64; 3] {
    let d = sub(s[1], s[0]);
    let t = (dot(sub(p, s[0]), d) / dot(d, d).max(1e-300)).clamp(0.0, 1.0);
    [s[0][0] + t * d[0], s[0][1] + t * d[1], s[0][2] + t * d[2]]
}

fn seg_dist(p: [f64; 3], s: [[f64; 3]; 2]) -> f64 {
    norm(sub(p, seg_foot(p, s)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn far_from_the_edge_the_factor_is_one() {
        assert_eq!(g(10.0), C64::new(1.0, 0.0));
        let gbar = average_g([20.0, 25.0, 30.0], 1.0);
        assert!((gbar - C64::new(1.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn a_triangle_on_the_edge_averages_the_profile() {
        // small against δ: the average is g near the edge
        let gbar = average_g([0.0, 0.0, 0.01], 1.0);
        assert!((gbar.re - 2.18).abs() < 0.03, "{gbar}");
        // large against δ: the band's share of the triangle
        let big = average_g([0.0, 0.0, 10.0], 1.0);
        assert!(big.re > 1.0 && big.re < 1.2, "{big}");
    }

    #[test]
    fn the_correction_fades_in_with_thickness() {
        assert_eq!(fade(3.0), 0.0);
        assert_eq!(fade(4.0), 1.0);
        assert!((fade(3.5) - 0.5).abs() < 1e-12);
    }
}
