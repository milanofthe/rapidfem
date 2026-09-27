//! Minimal 3-vector helpers shared across the geometry/mesh crates.
//!
//! These `[f64; 3]` operations were independently re-declared, byte-identically,
//! in ~18 modules (every tet and brep file carried its own `sub`/`dot`/`cross`).
//! Consolidating them here removes the "fix the same one-liner in 18 places"
//! hazard. Two names that previously collided are split explicitly: `len`
//! returns the magnitude (the tet crates' old `norm`), `normalize` returns the
//! unit vector (the brep crates' old `norm`); both degenerate to the input.

/// A point or vector in 3-space.
pub type V3 = [f64; 3];

/// Component-wise difference `a - b`.
#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Component-wise sum `a + b`.
#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scalar multiple `a * s`.
#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product `a . b`.
#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product `a x b`.
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Euclidean magnitude `|a|`.
#[inline]
pub fn len(a: V3) -> f64 {
    dot(a, a).sqrt()
}

/// Euclidean distance `|a - b|`.
#[inline]
pub fn dist(a: V3, b: V3) -> f64 {
    len(sub(a, b))
}

/// Unit vector in the direction of `a`; returns `a` unchanged when `a` is the
/// zero vector (degenerate, no defined direction).
#[inline]
pub fn normalize(a: V3) -> V3 {
    let l = len(a);
    if l > 0.0 {
        scale(a, 1.0 / l)
    } else {
        a
    }
}
