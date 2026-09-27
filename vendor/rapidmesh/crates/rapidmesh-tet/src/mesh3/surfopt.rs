//! Quality optimization of a surface mesh: edge flips and smoothing inside
//! each patch, the pass gmsh and the chart tiling before ran after meshing.
//!
//! The restricted Delaunay facets meet the angle bound of the refinement;
//! most sit well above it, but a band near the bound stays. Flipping the
//! diagonal of two triangles of one patch where that raises their smallest
//! angle, and moving every free vertex towards the centroid of its
//! neighbours (then onto its carrier, or along its curve) where the
//! smallest angle around it does not fall, lifts that band. Corners stay,
//! feature edges and patch borders are never flipped, and no triangle turns
//! over.

use super::oracle::P3;
use rustc_hash::{FxHashMap, FxHashSet};

fn sub(a: P3, b: P3) -> P3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: P3, b: P3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normal(p: [P3; 3]) -> P3 {
    cross(sub(p[1], p[0]), sub(p[2], p[0]))
}

/// Smallest angle of a triangle, in degrees (0 when degenerate).
pub(crate) fn min_angle(p: [P3; 3]) -> f64 {
    let mut m = f64::INFINITY;
    for k in 0..3 {
        let (u, v) = (sub(p[(k + 1) % 3], p[k]), sub(p[(k + 2) % 3], p[k]));
        let d = (dot(u, u) * dot(v, v)).sqrt();
        if !(d > 0.0) {
            return 0.0;
        }
        m = m.min((dot(u, v) / d).clamp(-1.0, 1.0).acos().to_degrees());
    }
    m
}

/// A triangle of the surface mesh being optimized.
pub(crate) struct Tri {
    pub v: [usize; 3],
    pub patch: u32,
}

/// Two flipped triangles must bend by less than this (cos 30 deg): a flip
/// across a ridge of a curved patch would fold it.
const FLIP_BEND_COS: f64 = 0.866;

/// A change must raise the smallest angle by at least this (degrees).
const GAIN: f64 = 1e-3;

/// Optimizes the triangles in place over at most `passes` rounds of flips
/// and smoothing. `free[v]` marks the vertices that may move (not
/// corners); `project(v, x)` puts vertex `v` at `x` onto its carrier or
/// curve;
/// `fixed` holds the edges that must stay (feature curves).
pub(crate) fn optimize(
    points: &mut [P3],
    tris: &mut [Tri],
    free: &[bool],
    fixed: &FxHashSet<(usize, usize)>,
    project: &dyn Fn(usize, P3) -> Option<P3>,
    passes: usize,
) {
    for _ in 0..passes {
        let flipped = flip_pass(points, tris, fixed);
        let moved = smooth_pass(points, tris, free, project);
        if flipped + moved == 0 {
            break;
        }
    }
}

fn key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// Flips every diagonal whose flip raises the smallest angle of its two
/// triangles; returns the number of flips.
fn flip_pass(points: &[P3], tris: &mut [Tri], fixed: &FxHashSet<(usize, usize)>) -> usize {
    let mut edges: FxHashMap<(usize, usize), Vec<usize>> = FxHashMap::default();
    for (i, t) in tris.iter().enumerate() {
        for k in 0..3 {
            edges
                .entry(key(t.v[k], t.v[(k + 1) % 3]))
                .or_default()
                .push(i);
        }
    }
    let mut keys: Vec<(usize, usize)> = edges
        .iter()
        .filter(|(e, ts)| ts.len() == 2 && !fixed.contains(e))
        .map(|(e, _)| *e)
        .collect();
    keys.sort_unstable();
    let mut touched = vec![false; tris.len()];
    let mut flips = 0;
    for e in keys {
        let (i, j) = (edges[&e][0], edges[&e][1]);
        if touched[i] || touched[j] || tris[i].patch != tris[j].patch {
            continue;
        }
        // Triangle i runs a -> b, j runs b -> a.
        let ti = tris[i].v;
        let Some(k) = (0..3).find(|&k| key(ti[k], ti[(k + 1) % 3]) == e) else {
            continue;
        };
        let (a, b, c) = (ti[k], ti[(k + 1) % 3], ti[(k + 2) % 3]);
        let tj = tris[j].v;
        let Some(d) = tj.iter().copied().find(|&x| x != a && x != b) else {
            continue;
        };
        let consistent = (0..3).any(|m| tj[m] == b && tj[(m + 1) % 3] == a);
        if !consistent || c == d || fixed.contains(&key(c, d)) || edges.contains_key(&key(c, d)) {
            continue;
        }
        let p = |x: usize| points[x];
        let old = min_angle([p(a), p(b), p(c)]).min(min_angle([p(b), p(a), p(d)]));
        let (n1, n2) = ([a, d, c], [d, b, c]);
        let new = min_angle(n1.map(p)).min(min_angle(n2.map(p)));
        if !(new > old + GAIN) {
            continue;
        }
        // The new pair keeps the side of the old one and does not fold.
        let (o1, o2) = (normal([p(a), p(b), p(c)]), normal([p(b), p(a), p(d)]));
        let (m1, m2) = (normal(n1.map(p)), normal(n2.map(p)));
        let unit = |v: P3| {
            let l = dot(v, v).sqrt();
            v.map(|x| x / l)
        };
        let side = unit([o1[0] + o2[0], o1[1] + o2[1], o1[2] + o2[2]]);
        if !(dot(unit(m1), side) > 0.0
            && dot(unit(m2), side) > 0.0
            && dot(unit(m1), unit(m2)) > FLIP_BEND_COS)
        {
            continue;
        }
        tris[i].v = n1;
        tris[j].v = n2;
        touched[i] = true;
        touched[j] = true;
        flips += 1;
    }
    flips
}

/// Moves every free vertex to the centroid of its neighbours, projected
/// onto its carrier, where the smallest angle of its triangles does not
/// fall and none turns over; returns the number of moves.
fn smooth_pass(
    points: &mut [P3],
    tris: &[Tri],
    free: &[bool],
    project: &dyn Fn(usize, P3) -> Option<P3>,
) -> usize {
    let mut star: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    for (i, t) in tris.iter().enumerate() {
        for &v in &t.v {
            star[v].push(i);
        }
    }
    let mut moves = 0;
    for v in 0..points.len() {
        if !free[v] || star[v].is_empty() {
            continue;
        }
        let mut c = [0.0; 3];
        let mut n = 0.0;
        for &i in &star[v] {
            for &w in &tris[i].v {
                if w != v {
                    for k in 0..3 {
                        c[k] += points[w][k];
                    }
                    n += 1.0;
                }
            }
        }
        let Some(x) = project(v, c.map(|s| s / n)) else {
            continue;
        };
        let at = |w: usize, moved: bool| if moved && w == v { x } else { points[w] };
        let tri_at = |i: usize, moved: bool| tris[i].v.map(|w| at(w, moved));
        let before = star[v]
            .iter()
            .map(|&i| min_angle(tri_at(i, false)))
            .fold(f64::INFINITY, f64::min);
        let after = star[v]
            .iter()
            .map(|&i| min_angle(tri_at(i, true)))
            .fold(f64::INFINITY, f64::min);
        let upright = star[v]
            .iter()
            .all(|&i| dot(normal(tri_at(i, false)), normal(tri_at(i, true))) > 0.0);
        if upright && after >= before - 1e-9 && (after > before + GAIN || x != points[v]) {
            points[v] = x;
            moves += 1;
        }
    }
    moves
}
