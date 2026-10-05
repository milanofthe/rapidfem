// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The sheet booleans and sheet outlines.
//!
//! The booleans themselves are rapidmesh's: a union keeps the pieces under
//! one tag and rapidmesh merges them, a difference or intersection is its
//! exact boolean in the common plane (`Geometry::sheet_boolean`, see
//! [`crate::geometry::Piece`]). What stays here is the outline of one
//! sheet in 3D, the profile a loft or a revolution sweeps.

use rapidmesh::shapes::Sheet;
use rapidmesh::Transform;
use rapidfem_core::geom::{add, cross, dot, scale, sub, unit};

type P3 = [f64; 3];

/// Sides of the polygon a disc's outline has.
pub const DISC_SEGMENTS: usize = 96;

/// A sheet boolean.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SheetOp {
    Union,
    Difference,
    Intersection,
}

/// `p` rotated by `angle` about the axis `axis` through `center` (Rodrigues).
fn rotate(p: P3, angle: f64, axis: P3, center: P3) -> P3 {
    let k = unit(axis);
    let v = sub(p, center);
    let (s, c) = angle.sin_cos();
    let r = add(add(scale(v, c), scale(cross(k, v), s)), scale(k, dot(k, v) * (1.0 - c)));
    add(r, center)
}

fn apply(t: &Transform, p: P3) -> P3 {
    match *t {
        Transform::Translate(d) => add(p, d),
        Transform::Rotate { angle, axis, center } => rotate(p, angle, axis, center),
        Transform::Mirror { normal, point } => {
            let n = unit(normal);
            sub(p, scale(n, 2.0 * dot(sub(p, point), n)))
        }
        Transform::Stretch { factors, center } => {
            let v = sub(p, center);
            add(center, [v[0] * factors[0], v[1] * factors[1], v[2] * factors[2]])
        }
    }
}

/// The outer vertex loop of `sheet` in 3D, `transforms` applied: the profile
/// a loft or a revolution sweeps (a disc as a [`DISC_SEGMENTS`]-gon).
pub(crate) fn outline(sheet: &Sheet, transforms: &[Transform]) -> Result<Vec<P3>, String> {
    let pts = match sheet {
        Sheet::Rect { corner, u, v } => vec![*corner, add(*corner, *u), add(add(*corner, *u), *v), add(*corner, *v)],
        Sheet::Disc { radius, center, axis, .. } => {
            let n = unit(*axis);
            let helper = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let a = unit(cross(n, helper));
            let b = cross(n, a);
            (0..DISC_SEGMENTS)
                .map(|k| {
                    let t = 2.0 * std::f64::consts::PI * k as f64 / DISC_SEGMENTS as f64;
                    add(*center, add(scale(a, radius * t.cos()), scale(b, radius * t.sin())))
                })
                .collect()
        }
        Sheet::Polygon { points, position, u, v, .. } => {
            points.iter().map(|q| add(*position, add(scale(*u, q[0]), scale(*v, q[1])))).collect()
        }
        Sheet::Nurbs { .. } => return Err("a NURBS sheet has no polygon outline".into()),
    };
    Ok(pts.into_iter().map(|p| transforms.iter().fold(p, |q, t| apply(t, q))).collect())
}
