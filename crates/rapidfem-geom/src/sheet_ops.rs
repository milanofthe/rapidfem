// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! Boolean operations on coplanar sheets: union, difference, intersection.
//!
//! rapidmesh's CSG works on solids; sheets (traces, ground planes, slots)
//! need the 2D booleans in their plane. Every operand becomes its outline in
//! 3D (its transforms applied), the outlines are checked to share one plane,
//! projected into that plane's frame, combined by i_overlay and handed back
//! as `Sheet::Polygon`s with holes plus the placement that puts the frame
//! back. A plane `z = const` gets no placement, so an xy result stays an xy
//! polygon (and extrudes as a prism). Discs enter as polygons of
//! [`DISC_SEGMENTS`] sides. To move into rapidmesh once it offers sheet
//! booleans (milanofthe/rapidmesh-dev#292).

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::simplify::SimplifyShape;
use i_overlay::float::single::SingleFloatOverlay;
use rapidmesh::shapes::Sheet;
use rapidmesh::Transform;
use rapidfem_core::geom::{add, cross, dot, norm, scale, sub, unit};

type P2 = [f64; 2];
type P3 = [f64; 3];
type Shapes = Vec<Vec<Vec<P2>>>;

/// Sides of the polygon a disc enters a boolean as.
pub const DISC_SEGMENTS: usize = 96;

/// Relative distance from the plane (to the operands' size) within which
/// an outline counts as coplanar.
const PLANE_TOL: f64 = 1e-9;

/// The boolean of [`boolean`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SheetOp {
    Union,
    Difference,
    Intersection,
}

/// One outline and its holes, in 3D.
struct Loops {
    outer: Vec<P3>,
    holes: Vec<Vec<P3>>,
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

/// The outline of `sheet` in 3D, `transforms` applied.
fn loops_of(sheet: &Sheet, transforms: &[Transform]) -> Result<Loops, String> {
    let place = |pts: Vec<P3>| -> Vec<P3> {
        pts.into_iter().map(|p| transforms.iter().fold(p, |q, t| apply(t, q))).collect()
    };
    let lift = |pts: &[P2], at: P3| pts.iter().map(|q| [q[0] + at[0], q[1] + at[1], at[2]]).collect::<Vec<_>>();
    Ok(match sheet {
        Sheet::Rect { corner, u, v } => Loops {
            outer: place(vec![*corner, add(*corner, *u), add(add(*corner, *u), *v), add(*corner, *v)]),
            holes: Vec::new(),
        },
        Sheet::Disc { radius, center, axis, .. } => {
            let n = unit(*axis);
            let helper = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let a = unit(cross(n, helper));
            let b = cross(n, a);
            let pts = (0..DISC_SEGMENTS)
                .map(|k| {
                    let t = 2.0 * std::f64::consts::PI * k as f64 / DISC_SEGMENTS as f64;
                    add(*center, add(scale(a, radius * t.cos()), scale(b, radius * t.sin())))
                })
                .collect();
            Loops { outer: place(pts), holes: Vec::new() }
        }
        Sheet::Polygon { points, holes, position } => Loops {
            outer: place(lift(points, *position)),
            holes: holes.iter().map(|h| place(lift(h, *position))).collect(),
        },
        Sheet::Nurbs { .. } => return Err("a NURBS sheet does not take part in sheet booleans".into()),
    })
}

/// The outer vertex loop of `sheet` in 3D, `transforms` applied: the profile
/// a loft or a revolution sweeps (a disc as a [`DISC_SEGMENTS`]-gon).
pub(crate) fn outline(sheet: &Sheet, transforms: &[Transform]) -> Result<Vec<P3>, String> {
    Ok(loops_of(sheet, transforms)?.outer)
}

/// The plane of an outline: a point on it and its unit normal (Newell).
fn plane_of(outline: &[P3]) -> Result<(P3, P3), String> {
    let mut n = [0.0; 3];
    for k in 0..outline.len() {
        let (a, b) = (outline[k], outline[(k + 1) % outline.len()]);
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    if norm(n) == 0.0 {
        return Err("a degenerate sheet outline has no plane".into());
    }
    Ok((outline[0], unit(n)))
}

/// The frame of the plane through `point` with normal `n`: origin, the
/// in-plane axes and the placement taking the xy plane onto it (empty for
/// a plane `z = const`).
fn frame(point: P3, n: P3) -> (P3, P3, P3, Vec<Transform>) {
    let z = [0.0, 0.0, 1.0];
    if n[0].abs() < 1e-12 && n[1].abs() < 1e-12 {
        return ([0.0, 0.0, point[2]], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], Vec::new());
    }
    let axis = cross(z, n);
    let angle = dot(z, n).clamp(-1.0, 1.0).acos();
    let origin = scale(n, dot(point, n));
    let o = [0.0; 3];
    let u = rotate([1.0, 0.0, 0.0], angle, axis, o);
    let v = rotate([0.0, 1.0, 0.0], angle, axis, o);
    (origin, u, v, vec![Transform::Rotate { angle, axis, center: o }, Transform::Translate(origin)])
}

/// The shapes of one operand in the frame, outer counter-clockwise and the
/// holes clockwise.
fn shapes_of(l: &Loops, origin: P3, u: P3, v: P3) -> Shapes {
    let to2 = |pts: &[P3]| pts.iter().map(|p| {
        let d = sub(*p, origin);
        [dot(d, u), dot(d, v)]
    }).collect::<Vec<P2>>();
    let mut contours = vec![to2(&l.outer)];
    contours.extend(l.holes.iter().map(|h| to2(h)));
    contours.simplify_shape(FillRule::EvenOdd, 0.0)
}

/// One operand of [`boolean`]: its pieces and the transforms they share.
pub type Operand<'a> = (&'a [Sheet], &'a [Transform]);

/// The union of an operand's pieces in the frame.
fn operand_shapes<'a>(pieces: impl IntoIterator<Item = &'a Loops>, origin: P3, u: P3, v: P3) -> Shapes {
    pieces
        .into_iter()
        .flat_map(|l| shapes_of(l, origin, u, v).into_iter().flatten())
        .collect::<Vec<_>>()
        .simplify_shape(FillRule::NonZero, 0.0)
}

/// `target` combined with `tools` by `op`: the resulting sheets, xy
/// polygons with holes in the plane's frame, and the placement they need.
/// Errors when the operands do not share a plane; an empty result is an
/// error too (the target would vanish).
pub fn boolean(op: SheetOp, target: Operand, tools: &[Operand]) -> Result<(Vec<Sheet>, Vec<Transform>), String> {
    let loops = |(pieces, tr): Operand| pieces.iter().map(|s| loops_of(s, tr)).collect::<Result<Vec<_>, _>>();
    let t = loops(target)?;
    let others = tools.iter().map(|&o| loops(o)).collect::<Result<Vec<_>, _>>()?;
    let first = t.first().ok_or("an empty sheet operand")?;
    let (point, n) = plane_of(&first.outer)?;
    let all = || t.iter().chain(others.iter().flatten());
    let size = all()
        .flat_map(|l| l.outer.iter())
        .map(|p| norm(sub(*p, point)))
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    for l in all() {
        for p in l.outer.iter().chain(l.holes.iter().flatten()) {
            if dot(sub(*p, point), n).abs() > PLANE_TOL * size {
                return Err("sheet booleans need every sheet in one plane".into());
            }
        }
    }
    let (origin, u, v, placement) = frame(point, n);
    let subject = operand_shapes(&t, origin, u, v);
    let clip = operand_shapes(others.iter().flatten(), origin, u, v);
    let rule = match op {
        SheetOp::Union => OverlayRule::Union,
        SheetOp::Difference => OverlayRule::Difference,
        SheetOp::Intersection => OverlayRule::Intersect,
    };
    let result = subject.overlay(&clip, rule, FillRule::NonZero);
    if result.is_empty() {
        return Err("the sheet boolean leaves nothing".into());
    }
    // a plane z = const keeps its height in the polygon, any other is placed
    let z = if placement.is_empty() { origin[2] } else { 0.0 };
    let sheets = result
        .into_iter()
        .map(|shape| {
            let mut it = shape.into_iter();
            Sheet::Polygon {
                points: it.next().unwrap_or_default(),
                holes: it.collect(),
                position: [0.0, 0.0, z],
            }
        })
        .collect();
    Ok((sheets, placement))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area(points: &[P2]) -> f64 {
        0.5 * (0..points.len())
            .map(|k| {
                let (a, b) = (points[k], points[(k + 1) % points.len()]);
                a[0] * b[1] - b[0] * a[1]
            })
            .sum::<f64>()
    }

    fn total_area(sheets: &[Sheet]) -> f64 {
        sheets
            .iter()
            .map(|s| match s {
                Sheet::Polygon { points, holes, .. } => {
                    area(points).abs() - holes.iter().map(|h| area(h).abs()).sum::<f64>()
                }
                _ => unreachable!(),
            })
            .sum()
    }

    #[test]
    fn plate_minus_disc_has_a_hole_and_stays_at_its_height() {
        let plate = Sheet::xy(4.0, 4.0, [0.0, 0.0, 1.5]);
        let disc = Sheet::disc(1.0, [2.0, 2.0, 1.5], [0.0, 0.0, 1.0]);
        let (out, placement) = boolean(SheetOp::Difference, (std::slice::from_ref(&plate), &[]), &[(std::slice::from_ref(&disc), &[])]).unwrap();
        assert!(placement.is_empty());
        assert_eq!(out.len(), 1);
        let Sheet::Polygon { holes, position, .. } = &out[0] else { unreachable!() };
        assert_eq!(holes.len(), 1);
        assert_eq!(position[2], 1.5);
        let want = 16.0 - std::f64::consts::PI;
        assert!((total_area(&out) - want).abs() < 2e-3 * want);
    }

    #[test]
    fn union_of_abutting_strips_is_one_piece_and_a_split_leaves_two() {
        let a = Sheet::xy(2.0, 1.0, [0.0, 0.0, 0.0]);
        let b = Sheet::xy(2.0, 1.0, [2.0, 0.0, 0.0]);
        let (u, _) = boolean(SheetOp::Union, (std::slice::from_ref(&a), &[]), &[(std::slice::from_ref(&b), &[])]).unwrap();
        assert_eq!(u.len(), 1);
        assert!((total_area(&u) - 4.0).abs() < 1e-12);
        let wide = Sheet::xy(5.0, 1.0, [0.0, 0.0, 0.0]);
        let gap = Sheet::xy(1.0, 3.0, [2.0, -1.0, 0.0]);
        let (d, _) = boolean(SheetOp::Difference, (std::slice::from_ref(&wide), &[]), &[(std::slice::from_ref(&gap), &[])]).unwrap();
        assert_eq!(d.len(), 2);
        assert!((total_area(&d) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn tilted_plane_round_trips_through_its_placement() {
        // two plates in the plane x = 1, the second one moved there
        let a = Sheet::yz(2.0, 2.0, [1.0, 0.0, 0.0]);
        let b = Sheet::xy(1.0, 1.0, [0.0, 0.0, 0.0]);
        let to_plane = [
            Transform::Rotate { angle: -std::f64::consts::FRAC_PI_2, axis: [0.0, 1.0, 0.0], center: [0.0; 3] },
            Transform::Translate([1.0, 0.5, 0.5]),
        ];
        let (out, placement) = boolean(SheetOp::Intersection, (std::slice::from_ref(&a), &[]), &[(std::slice::from_ref(&b), &to_plane)]).unwrap();
        assert_eq!(out.len(), 1);
        assert!((total_area(&out) - 1.0).abs() < 1e-9);
        let Sheet::Polygon { points, .. } = &out[0] else { unreachable!() };
        for q in points {
            let p = placement.iter().fold([q[0], q[1], 0.0], |p, t| apply(t, p));
            assert!((p[0] - 1.0).abs() < 1e-9, "{p:?} is off the plane x = 1");
            assert!(p[1] > 0.5 - 1e-9 && p[1] < 1.5 + 1e-9 && p[2] > 0.5 - 1e-9 && p[2] < 1.5 + 1e-9);
        }
    }

    #[test]
    fn sheets_in_different_planes_are_refused() {
        let a = Sheet::xy(1.0, 1.0, [0.0, 0.0, 0.0]);
        let b = Sheet::xy(1.0, 1.0, [0.0, 0.0, 0.5]);
        assert!(boolean(SheetOp::Union, (std::slice::from_ref(&a), &[]), &[(std::slice::from_ref(&b), &[])]).is_err());
    }
}
