//! Edge features of a solid: chamfers and fillets, cut from the solid alone
//! by the exact boolean, with exact carriers for the new faces.
//!
//! An edge is found on the B-rep of the scene, as an edge of one of the
//! solid's faces (so the rim of a hole cut into it counts), and named by
//! the origins of its two faces. Both faces must be straight across the edge: two planes along
//! a straight edge, or, along a circle, two coaxial surfaces of revolution
//! with straight meridians (a plane square to the axis, a cylinder, a cone).
//! In the section square to the edge (the meridian for a circle) both faces
//! are then lines through the edge point. A chamfer is the line between the
//! points `distance` into each face, a fillet the arc of `radius` tangent to
//! both lines. The cutter is the region between the edge and that line or
//! arc, grown outward, swept along the edge (a prism) or about the axis (a
//! ring): the new face is exactly a plane or cone, a cylinder or torus.
//!
//! A fillet meets its faces tangentially, yet the cutter's faceted arc has
//! its first and last vertex on the lines of tangency, so its facets leave
//! the faces at half a facet angle: the boolean cuts them cleanly along
//! those lines, without the strips two tessellations crossing about a
//! tangent contact would leave.

use crate::{Error, Result};
use rapidmesh_brep::{Brep, Curve, Model, Surface};
use rapidmesh_geom::vec3::{add, cross, dot, normalize as unit, scale, sub, V3};
use rapidmesh_geom::{extrude_profile, revolve_at, Faceted, ProfileEdge};
use std::f64::consts::{PI, TAU};

/// Which edges of a solid, its faces by their role in it: every edge of
/// its faces, those of one face, the one between two of its faces, or the
/// one between its face and a face of another solid (the rim of a hole cut
/// into it: the void's index and the role of its face).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgePick {
    All,
    Of(u32),
    Between(u32, u32),
    With(u32, u32, u32),
}

/// How an edge is cut: a flat chamfer `distance` into both faces, or a
/// fillet of `radius` tangent to both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EdgeCut {
    Chamfer(f64),
    Fillet(f64),
}

impl EdgeCut {
    fn name(self) -> &'static str {
        match self {
            EdgeCut::Chamfer(_) => "chamfer",
            EdgeCut::Fillet(_) => "fillet",
        }
    }
    fn size(self) -> f64 {
        match self {
            EdgeCut::Chamfer(d) | EdgeCut::Fillet(d) => d,
        }
    }
}

/// One face along an edge at a point of it: its outward normal and the
/// direction into it, away from the edge, both square to the edge.
struct Side {
    normal: V3,
    into: V3,
    surface: Surface,
}

/// A solid with edges cut: its new shape, the material it loses (one piece
/// per edge, the solid's surfaces with the cutter's after them) and the
/// roles of the new faces in the new shape.
pub(crate) struct Cut {
    pub shape: Faceted,
    pub removed: Vec<Faceted>,
    pub roles: Vec<u32>,
}

/// Solid `owner` (shape `f`, in `region`) of the scene `model` with the
/// picked edges cut by `cut`.
pub(crate) fn cut_edges(
    model: &Model,
    owner: u32,
    region: u32,
    f: &Faceted,
    picks: &[EdgePick],
    cut: EdgeCut,
    maxh: Option<f64>,
) -> Result<Cut> {
    let what = cut.name();
    if !(cut.size() > 0.0) {
        return Err(Error::Invalid(format!(
            "{what} size {} must be positive",
            cut.size()
        )));
    }
    let brep = &model.brep;
    let mut cutters = Vec::new();
    for (e, edge) in brep.edges.iter().enumerate() {
        // The faces along it that bound the solid (others meet it where
        // more materials do, like a hole on through the air above).
        let mut faces: Vec<usize> = edge
            .coedges
            .iter()
            .map(|&c| brep.coedges[c.0 as usize].face.0 as usize)
            .filter(|&f| brep.faces[f].regions.iter().any(|r| r.0 == region))
            .collect();
        faces.sort_unstable();
        faces.dedup();
        let origin: Vec<(u32, u32)> = faces
            .iter()
            .map(|&i| (brep.faces[i].owner, brep.faces[i].role))
            .collect();
        let own = |r: u32| origin.contains(&(owner, r));
        let picked = picks.iter().any(|p| match *p {
            EdgePick::All => origin.iter().any(|o| o.0 == owner),
            EdgePick::Of(r) => own(r),
            EdgePick::Between(a, b) => a != b && own(a) && own(b),
            EdgePick::With(r, other, role) => own(r) && origin.contains(&(other, role)),
        });
        if !picked {
            continue;
        }
        let name = format!("edge between faces {origin:?} (solid, role)");
        if faces.len() != 2 {
            return Err(Error::Invalid(format!(
                "{what}: the {name} does not bound the solid on two faces"
            )));
        }
        cutters.push(
            cutter(model, e, [faces[0], faces[1]], region, cut, maxh)
                .map_err(|why| Error::Invalid(format!("{what}: the {name}: {why}")))?,
        );
    }
    if cutters.is_empty() {
        return Err(Error::Invalid(format!("{what}: no edge matches")));
    }
    // Each cut appends the cutter's surfaces after the shape's; the new
    // face is the cutter's first.
    let mut roles = Vec::with_capacity(cutters.len());
    let mut base = f.surfaces.len() as u32;
    for c in &cutters {
        roles.push(base);
        base += c.surfaces.len() as u32;
    }
    Ok(Cut {
        shape: cutters
            .iter()
            .try_fold(f.clone(), |acc, c| acc.minus(c))
            .map_err(|e| Error::Invalid(format!("{what}: the cut does not assemble: {e}")))?,
        removed: cutters
            .iter()
            .map(|c| f.common(c))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| Error::Invalid(format!("{what}: the cut does not assemble: {e}")))?,
        roles,
    })
}

/// The section square to an edge: a plane swept along a straight edge, or
/// the meridian turned about the axis of a circle.
enum Section {
    Straight {
        base: V3,
        u: V3,
        v: V3,
        h: V3,
    },
    /// About the axis through `center`, at the angles (from `x`) of the
    /// circle's own vertices: the ring's facets then meet the faces along
    /// the circle where theirs do, instead of crossing them.
    Meridian {
        center: V3,
        axis: V3,
        radial: V3,
        x: V3,
        angles: Vec<f64>,
        full: bool,
    },
}

impl Section {
    fn flat(&self, q: V3) -> [f64; 2] {
        match *self {
            Section::Straight { base, u, v, .. } => [dot(sub(q, base), u), dot(sub(q, base), v)],
            Section::Meridian {
                center,
                axis,
                radial,
                ..
            } => [dot(sub(q, center), radial), dot(sub(q, center), axis)],
        }
    }

    fn solid(
        &self,
        pts: &[[f64; 2]],
        edges: &[ProfileEdge],
        maxh: Option<f64>,
    ) -> std::result::Result<Faceted, String> {
        match *self {
            Section::Straight { base, u, v, h } => {
                extrude_profile(pts, edges, base, u, v, h, maxh, 1e-2)
            }
            Section::Meridian {
                center,
                axis,
                x,
                ref angles,
                full,
                ..
            } => {
                if pts.iter().any(|m| m[0] < 0.0) {
                    return Err("the cut reaches past the axis".into());
                }
                revolve_at(pts, edges, center, axis, x, angles, full, maxh, 1e-2)
            }
        }
    }
}

/// The cutter of edge `e` between `faces` of the solid in `region`; its
/// first surface is the new face.
fn cutter(
    model: &Model,
    e: usize,
    faces: [usize; 2],
    region: u32,
    cut: EdgeCut,
    maxh: Option<f64>,
) -> std::result::Result<Faceted, String> {
    let brep = &model.brep;
    let edge = &brep.edges[e];
    let chain = &edge.chain;
    if chain.len() < 2 {
        return Err("it has no length".into());
    }
    let size = cut.size();
    // A point of the edge, on its exact curve, and the tangent there.
    let k = (chain.len() - 1) / 2;
    let (p, t) = match edge.curve {
        Curve::Line { p0, dir } => {
            let m = scale(add(chain[k], chain[k + 1]), 0.5);
            (add(p0, scale(dir, dot(sub(m, p0), dir))), dir)
        }
        Curve::Circle {
            center,
            axis,
            radius,
            ..
        } => {
            let d = sub(chain[k], center);
            let radial = unit(sub(d, scale(axis, dot(d, axis))));
            (add(center, scale(radial, radius)), cross(axis, radial))
        }
        _ => return Err("only straight and circular edges take one".into()),
    };
    let sides = [0, 1].map(|i| side(model, faces[i], region, [chain[k], chain[k + 1]], p, t));
    let [a, b] = match sides {
        [Some(a), Some(b)] => [a, b],
        _ => return Err("the solid is not on one side of both faces".into()),
    };
    if !(dot(a.into, b.normal) < 0.0 && dot(b.into, a.normal) < 0.0) {
        return Err("it is not convex; the cut would add material".into());
    }
    let section = match edge.curve {
        Curve::Line { .. } => {
            if !matches!(a.surface, Surface::Plane { .. })
                || !matches!(b.surface, Surface::Plane { .. })
            {
                return Err("a straight edge takes one between two planes".into());
            }
            // Longer than the edge by the cut's size at both ends.
            let (e0, e1) = (chain[0], chain[chain.len() - 1]);
            let t = if dot(sub(e1, e0), t) >= 0.0 {
                t
            } else {
                scale(t, -1.0)
            };
            let len = dot(sub(e1, e0), t) + 2.0 * size;
            Section::Straight {
                base: sub(e0, scale(t, size)),
                u: a.into,
                v: cross(t, a.into),
                h: scale(t, len),
            }
        }
        Curve::Circle {
            center,
            axis,
            radius,
            ..
        } => {
            for s in [&a, &b] {
                if !straight_meridian(&s.surface, center, axis, radius) {
                    return Err("a circle takes one between a plane square to its axis, \
                         a cylinder or a cone about it"
                        .into());
                }
            }
            let radial_of =
                |q: V3| unit(sub(sub(q, center), scale(axis, dot(sub(q, center), axis))));
            let x = radial_of(chain[0]);
            let y = cross(axis, x);
            let angle = |q: V3| {
                let d = radial_of(q);
                dot(d, y).atan2(dot(d, x))
            };
            let full = chain[0] == chain[chain.len() - 1];
            let angles = if full {
                let mut a: Vec<f64> = chain[..chain.len() - 1]
                    .iter()
                    .map(|&q| angle(q).rem_euclid(TAU))
                    .collect();
                a.sort_by(f64::total_cmp);
                a.dedup();
                a
            } else {
                // An arc: its angles unwound, grown by the cut's size at
                // both ends like a straight edge.
                let mut a = vec![0.0];
                for w in chain.windows(2) {
                    let step = (angle(w[1]) - angle(w[0]) + PI).rem_euclid(TAU) - PI;
                    a.push(a[a.len() - 1] + step);
                }
                if a[a.len() - 1] < 0.0 {
                    a.iter_mut().for_each(|t| *t = -*t);
                    a.reverse();
                }
                let grow = size / radius;
                a.insert(0, a[0] - grow);
                a.push(a[a.len() - 1] + grow);
                a
            };
            // Mirrored arcs ran backwards: turn them with a mirrored frame.
            let backwards = !full && chain.len() > 1 && {
                let step = (angle(chain[1]) - angle(chain[0]) + PI).rem_euclid(TAU) - PI;
                step < 0.0
            };
            let axis = if backwards { scale(axis, -1.0) } else { axis };
            Section::Meridian {
                center,
                axis,
                radial: radial_of(p),
                x,
                angles,
                full,
            }
        }
        _ => unreachable!(),
    };
    let (pts, edges) = profile(p, &a, &b, cut, &section);
    section.solid(&pts, &edges, maxh)
}

/// The cutter's profile in the section, its first edge the new face: the
/// region between the edge point `p` and the chamfer line or fillet arc,
/// grown by the cut's size outward (past the faces, out of the solid).
fn profile(
    p: V3,
    a: &Side,
    b: &Side,
    cut: EdgeCut,
    section: &Section,
) -> (Vec<[f64; 2]>, Vec<ProfileEdge>) {
    let out = add(p, scale(add(a.normal, b.normal), cut.size()));
    match cut {
        EdgeCut::Chamfer(d) => {
            let (qa, qb) = (add(p, scale(a.into, d)), add(p, scale(b.into, d)));
            let u = unit(sub(qa, qb));
            let pts = [add(qa, scale(u, d)), sub(qb, scale(u, d)), out];
            (
                pts.iter().map(|&q| section.flat(q)).collect(),
                vec![ProfileEdge::Line; 3],
            )
        }
        EdgeCut::Fillet(r) => {
            // The circle of radius r tangent to both faces: its centre on
            // the bisector, the tangent points r / tan(half the angle) in.
            let half = 0.5 * dot(a.into, b.into).clamp(-1.0, 1.0).acos();
            let (ta, tb) = (
                add(p, scale(a.into, r / half.tan())),
                add(p, scale(b.into, r / half.tan())),
            );
            let c = add(p, scale(unit(add(a.into, b.into)), r / half.sin()));
            let [fa, fb, fc] = [ta, tb, c].map(|q| section.flat(q));
            let (xa, xb) = (
                [fa[0] - fc[0], fa[1] - fc[1]],
                [fb[0] - fc[0], fb[1] - fc[1]],
            );
            let sweep = (xa[0] * xb[1] - xa[1] * xb[0]).atan2(xa[0] * xb[0] + xa[1] * xb[1]);
            let pts = [
                ta,
                tb,
                add(tb, scale(b.normal, r)),
                out,
                add(ta, scale(a.normal, r)),
            ];
            let mut edges = vec![ProfileEdge::Line; 5];
            edges[0] = ProfileEdge::Arc((0.25 * sweep).tan());
            (pts.iter().map(|&q| section.flat(q)).collect(), edges)
        }
    }
}

/// The face `f` at the edge point `p` (tangent `t`): from its triangle on
/// the edge segment `seg`, the side of the solid in `region`, and from its
/// carrier the exact normal.
fn side(model: &Model, f: usize, region: u32, seg: [V3; 2], p: V3, t: V3) -> Option<Side> {
    let (plc, brep): (&rapidmesh_geom::TaggedPlc, &Brep) = (&model.plc, &model.brep);
    let face = &brep.faces[f];
    let at = |v: u32| plc.vertices[v as usize];
    let tri = face.facets.iter().map(|&i| i as usize).find(|&i| {
        let v = plc.triangles[i].map(at);
        seg.iter().all(|q| v.contains(q))
    })?;
    let v = plc.triangles[tri].map(at);
    let third = *v.iter().find(|q| !seg.contains(q))?;
    // The triangle normal, turned out of the solid.
    let n = unit(cross(sub(v[1], v[0]), sub(v[2], v[0])));
    let [front, back] = plc.region_tags[tri].map(|r| r.0);
    let out = match (front == region, back == region) {
        (false, true) => n,
        (true, false) => scale(n, -1.0),
        _ => return None,
    };
    let surface = brep.surface(face.surface).clone();
    let exact = surface.closest(p).1;
    let normal = if dot(exact, out) >= 0.0 {
        exact
    } else {
        scale(exact, -1.0)
    };
    let across = unit(cross(t, normal));
    let into = if dot(sub(third, p), across) >= 0.0 {
        across
    } else {
        scale(across, -1.0)
    };
    Some(Side {
        normal,
        into,
        surface,
    })
}

/// A plane square to `axis` or a cylinder or cone about the axis through
/// `center`: a surface of revolution about it with a straight meridian.
fn straight_meridian(s: &Surface, center: V3, axis: V3, radius: f64) -> bool {
    let tol = 1e-9 * radius.max(1.0);
    let on_axis = |q: V3| {
        let d = sub(q, center);
        let off = sub(d, scale(axis, dot(d, axis)));
        dot(off, off).sqrt() <= tol
    };
    let along = |b: V3| dot(unit(b), axis).abs() >= 1.0 - 1e-9;
    match s {
        Surface::Plane { normal, .. } => along(*normal),
        Surface::Cylinder {
            center: c, axis: a, ..
        } => along(*a) && on_axis(*c),
        Surface::Cone { apex, axis: a, .. } => along(*a) && on_axis(*apex),
        _ => false,
    }
}
