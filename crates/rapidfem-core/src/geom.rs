// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Small 3-vector helpers shared by every crate: the arithmetic, the
//! centroid of a tet or a triangle and a triangle's area vector.

/// A point or vector in 3D.
pub type V3 = [f64; 3];

#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// `a` scaled to unit length (`a` must not be zero).
#[inline]
pub fn unit(a: V3) -> V3 {
    scale(a, 1.0 / norm(a))
}

/// The mean of points `ids` of `nodes` (a tet's or a triangle's centroid).
#[inline]
pub fn centroid<const N: usize>(nodes: &[V3], ids: [usize; N]) -> V3 {
    let mut c = [0.0; 3];
    for &i in &ids {
        c = add(c, nodes[i]);
    }
    scale(c, 1.0 / N as f64)
}

/// The triangle's area vector, `(b - a) × (c - a) / 2`: its length is the
/// area, its direction the normal by the right-hand rule.
#[inline]
pub fn tri_area_vector(a: V3, b: V3, c: V3) -> V3 {
    scale(cross(sub(b, a), sub(c, a)), 0.5)
}
