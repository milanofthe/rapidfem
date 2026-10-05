//! The carrier surfaces: one type for every analytic kind, a B-spline
//! surface, a discrete patch of an import and a tube about a path, with one
//! interface the whole pipeline queries (the STEP reader, the shapes, the
//! B-rep, the mesher, its sizing and finish, the facade).
//!
//! Each analytic kind keeps an orthonormal [`Frame`]: a plane its origin and
//! in-plane axes with `z` its normal; a cylinder, cone, sphere, torus or
//! surface of revolution its axis `z` with `x` the direction of angle 0
//! (`y = z x x` the direction of angle `pi / 2`); an extrusion its profile
//! plane `x, y` and its direction `z`. The parameters `(u, v)`:
//!
//! | kind | `u` | `v` |
//! |---|---|---|
//! | plane | along `x` | along `y` |
//! | cylinder | angle about `z` | height along `z` |
//! | cone | angle about `z` | distance from the apex along `z` |
//! | sphere | angle about `z` | latitude |
//! | torus | angle about `z` | angle about the tube |
//! | extruded | profile parameter | height along `z` |
//! | revolved | angle about `z` | profile parameter |
//! | B-spline | its own | its own |
//!
//! A discrete patch and a tube have no parameter map: they are queried by
//! their nearest point ([`Surface::closest`]).

use crate::nurbs::NurbsCurve;
use crate::Curve;
use crate::{DiscreteSurface, NurbsSurface, TubePath};
use rapidmesh_exact::vector::{
    add, angle_about, centroid, cross, dist, dot, len, newell, normalize, scale, sub, unit, Affine,
    Frame, V2, V3,
};
use std::f64::consts::{FRAC_PI_2, TAU};
use std::sync::Arc;

/// A carrier surface.
#[derive(Debug, Clone)]
pub enum Surface {
    /// The plane through `o` square to `z`.
    Plane(Frame),
    Cylinder {
        frame: Frame,
        radius: f64,
    },
    /// The cone with its apex at `o`, opening along `z` by `half_angle`.
    Cone {
        frame: Frame,
        half_angle: f64,
    },
    Sphere {
        frame: Frame,
        radius: f64,
    },
    /// The torus about `z` through `o`: the ring of the tube's centres of
    /// radius `major`, the tube of radius `minor`.
    Torus {
        frame: Frame,
        major: f64,
        minor: f64,
    },
    /// The 2D `profile` in the plane `o + a x + b y`, swept along `z`. Covers
    /// airfoils and any extruded section; the curve carries the exact
    /// curvature.
    Extruded {
        frame: Frame,
        profile: Arc<NurbsCurve>,
    },
    /// The meridian `profile(t) = (r, z)` turned about the axis `z` through
    /// `o`: at angle `theta` the point `o + z z + r (cos theta x + sin theta
    /// y)`. The solid lies left of the profile, so `(dz, -dr)` points out.
    Revolved {
        frame: Frame,
        profile: Arc<NurbsCurve>,
    },
    /// A NURBS surface, the general free-form carrier. Affine maps act on
    /// its control points exactly.
    Nurbs(Arc<NurbsSurface>),
    /// A smooth patch of an imported triangle soup (between crease edges):
    /// the patch is the carrier, queried by its nearest point.
    Discrete(Arc<DiscreteSurface>),
    /// The surface at distance `radius` from a smooth path (swept pipes,
    /// helical coils), analytic per segment of the path.
    Tube {
        path: Arc<TubePath>,
        radius: f64,
    },
}

impl Surface {
    /// The plane through `point` square to `normal`, its `x` the
    /// [`ortho_unit`](rapidmesh_exact::vector::ortho_unit) of the normal;
    /// none for a zero normal.
    pub fn plane(point: V3, normal: V3) -> Option<Surface> {
        Frame::new(point, normal, None).map(Surface::Plane)
    }

    /// The plane of the flat polygon `pts` (at least three points, not all
    /// on a line): through their centroid, square to their Newell normal.
    pub fn plane_of(pts: &[V3]) -> Option<Surface> {
        if pts.is_empty() {
            return None;
        }
        Surface::plane(centroid(pts), newell(pts))
    }

    /// An order of surfaces by their geometry alone: the kind, then the
    /// frame and the sizes, bit by bit (kinds with a profile or a net by
    /// their kind). Where the facets of several solids lie on one face, the
    /// face takes the carrier first in it, whichever solid came first.
    pub fn geometry_order(&self, other: &Surface) -> std::cmp::Ordering {
        fn key(s: &Surface) -> (u8, Vec<u64>) {
            let framed = |f: &Frame, sizes: &[f64]| -> Vec<u64> {
                [f.o, f.x, f.y, f.z]
                    .iter()
                    .flat_map(|v| v.map(f64::to_bits))
                    .chain(sizes.iter().map(|x| x.to_bits()))
                    .collect()
            };
            match s {
                Surface::Plane(f) => (0, framed(f, &[])),
                Surface::Cylinder { frame, radius } => (1, framed(frame, &[*radius])),
                Surface::Cone { frame, half_angle } => (2, framed(frame, &[*half_angle])),
                Surface::Sphere { frame, radius } => (3, framed(frame, &[*radius])),
                Surface::Torus {
                    frame,
                    major,
                    minor,
                } => (4, framed(frame, &[*major, *minor])),
                Surface::Extruded { .. } => (5, Vec::new()),
                Surface::Revolved { .. } => (6, Vec::new()),
                Surface::Nurbs(_) => (7, Vec::new()),
                Surface::Discrete(_) => (8, Vec::new()),
                Surface::Tube { .. } => (9, Vec::new()),
            }
        }
        key(self).cmp(&key(other))
    }

    /// [`Surface::plane_of`] where the points lie in one plane to a
    /// billionth of their extent; none where they are bent.
    pub fn plane_through(pts: &[V3]) -> Option<Surface> {
        let plane = Surface::plane_of(pts)?;
        let f = plane.frame()?;
        let extent = pts.iter().map(|p| dist(*p, f.o)).fold(0.0, f64::max);
        pts.iter()
            .all(|p| dot(sub(*p, f.o), f.z).abs() <= 1e-9 * extent)
            .then_some(plane)
    }

    /// A cylinder about `axis` through `center`.
    pub fn cylinder(center: V3, axis: V3, radius: f64) -> Surface {
        Surface::Cylinder {
            frame: framed(center, axis),
            radius,
        }
    }

    /// A cone with its apex at `apex`, opening along `axis` by `half_angle`.
    pub fn cone(apex: V3, axis: V3, half_angle: f64) -> Surface {
        Surface::Cone {
            frame: framed(apex, axis),
            half_angle,
        }
    }

    /// A sphere about the z axis.
    pub fn sphere(center: V3, radius: f64) -> Surface {
        Surface::Sphere {
            frame: framed(center, [0.0, 0.0, 1.0]),
            radius,
        }
    }

    /// A torus about `axis` through `center`.
    pub fn torus(center: V3, axis: V3, major: f64, minor: f64) -> Surface {
        Surface::Torus {
            frame: framed(center, axis),
            major,
            minor,
        }
    }

    /// The name of the kind: "plane", "cylinder", "cone", "sphere",
    /// "torus", "extruded", "revolved", "nurbs", "discrete", "tube".
    pub fn name(&self) -> &'static str {
        match self {
            Surface::Plane(_) => "plane",
            Surface::Cylinder { .. } => "cylinder",
            Surface::Cone { .. } => "cone",
            Surface::Sphere { .. } => "sphere",
            Surface::Torus { .. } => "torus",
            Surface::Extruded { .. } => "extruded",
            Surface::Revolved { .. } => "revolved",
            Surface::Nurbs(_) => "nurbs",
            Surface::Discrete(_) => "discrete",
            Surface::Tube { .. } => "tube",
        }
    }

    /// The frame of an analytic kind; none for a B-spline, a discrete patch
    /// and a tube.
    pub fn frame(&self) -> Option<&Frame> {
        match self {
            Surface::Plane(f)
            | Surface::Cylinder { frame: f, .. }
            | Surface::Cone { frame: f, .. }
            | Surface::Sphere { frame: f, .. }
            | Surface::Torus { frame: f, .. }
            | Surface::Extruded { frame: f, .. }
            | Surface::Revolved { frame: f, .. } => Some(f),
            Surface::Nurbs(_) | Surface::Discrete(_) | Surface::Tube { .. } => None,
        }
    }

    /// Whether the surface is a plane.
    pub fn is_plane(&self) -> bool {
        matches!(self, Surface::Plane(_))
    }

    /// The plane set on the points `pts` of a face: its origin their
    /// centroid's foot, its `x` towards the first of them off it, so the
    /// coordinates of the face stay small; every other kind as it is.
    pub fn fitted(&self, pts: &[V3]) -> Surface {
        let Surface::Plane(f) = self else {
            return self.clone();
        };
        if pts.is_empty() {
            return self.clone();
        }
        let onto = |p: V3| sub(p, scale(f.z, dot(sub(p, f.o), f.z)));
        let o = onto(centroid(pts));
        let hint = pts
            .iter()
            .map(|&p| sub(onto(p), o))
            .find(|d| dot(*d, *d) > 1e-20);
        Surface::Plane(Frame::new(o, f.z, hint).unwrap_or(*f))
    }

    /// The surface carried by the affine map `m`. A plane, a B-spline and a
    /// discrete patch under any map; the other kinds under one that keeps
    /// their shape (a move, turn, mirror or uniform stretch, their lengths
    /// scaled by its factor); none under an unequal stretch, where they
    /// have no closed form left.
    pub fn mapped(&self, m: &Affine) -> Option<Surface> {
        match self {
            Surface::Plane(f) => Surface::plane(m.point(f.o), m.normal(f.z)),
            Surface::Nurbs(n) => Some(Surface::Nurbs(Arc::new(NurbsSurface::new(
                n.degree,
                n.knots.clone(),
                n.n,
                n.ctrl.iter().map(|&q| m.point(q)).collect(),
                n.weights.clone(),
            )))),
            // The discrete carrier is its point set; its normals follow the
            // winding of the mapped facets.
            Surface::Discrete(d) => Some(Surface::Discrete(Arc::new(DiscreteSurface::new(
                d.points.iter().map(|&q| m.point(q)).collect(),
                d.tris.clone(),
            )))),
            _ => {
                let s = m.uniform_factor()?;
                Some(match self {
                    Surface::Cylinder { frame, radius } => Surface::Cylinder {
                        frame: frame.mapped(m),
                        radius: radius * s,
                    },
                    Surface::Cone { frame, half_angle } => Surface::Cone {
                        frame: frame.mapped(m),
                        half_angle: *half_angle,
                    },
                    Surface::Sphere { frame, radius } => Surface::Sphere {
                        frame: frame.mapped(m),
                        radius: radius * s,
                    },
                    Surface::Torus {
                        frame,
                        major,
                        minor,
                    } => Surface::Torus {
                        frame: frame.mapped(m),
                        major: major * s,
                        minor: minor * s,
                    },
                    Surface::Extruded { frame, profile } => Surface::Extruded {
                        frame: frame.mapped(m),
                        profile: Arc::new(profile.scaled(s)),
                    },
                    Surface::Revolved { frame, profile } => Surface::Revolved {
                        frame: frame.mapped(m),
                        profile: Arc::new(profile.scaled(s)),
                    },
                    Surface::Tube { path, radius } => Surface::Tube {
                        path: Arc::new(TubePath::new(
                            path.pts.iter().map(|&q| m.point(q)).collect(),
                        )),
                        radius: radius * s,
                    },
                    Surface::Plane(_) | Surface::Nurbs(_) | Surface::Discrete(_) => {
                        unreachable!("mapped above")
                    }
                })
            }
        }
    }

    /// The nearest point of the surface to `p` and the outward normal there:
    /// in closed form without trigonometry for every analytic kind, through
    /// the parameter map for the spline kinds, through the facets for a
    /// discrete patch.
    pub fn closest(&self, p: V3) -> (V3, V3) {
        // A unit direction, or `fallback` where it is undefined (on an axis).
        let dir = |d: V3, fallback: V3| unit(d).unwrap_or(fallback);
        match self {
            // In-plane, not `p - n (n . d)`: an axis-aligned plane then
            // keeps its coordinate bit-exact.
            Surface::Plane(f) => {
                let l = f.local(p);
                (f.at(l[0], l[1], 0.0), f.z)
            }
            Surface::Cylinder { frame: f, radius } => {
                let foot = add(f.o, scale(f.z, dot(sub(p, f.o), f.z)));
                let n = dir(sub(p, foot), f.x);
                (add(foot, scale(n, *radius)), n)
            }
            Surface::Sphere { frame: f, radius } => {
                let n = dir(sub(p, f.o), f.x);
                (add(f.o, scale(n, *radius)), n)
            }
            Surface::Cone {
                frame: f,
                half_angle,
            } => {
                // In the meridian half-plane the nappe is the ray from the
                // apex along (sin, cos) of the half angle; behind the apex
                // the apex itself is nearest.
                let d = sub(p, f.o);
                let h = dot(d, f.z);
                let rd = sub(d, scale(f.z, h));
                let radial = dir(rd, f.x);
                let (sin, cos) = half_angle.sin_cos();
                let t = (len(rd) * sin + h * cos).max(0.0);
                let q = add(f.o, scale(add(scale(radial, sin), scale(f.z, cos)), t));
                (q, sub(scale(radial, cos), scale(f.z, sin)))
            }
            Surface::Torus {
                frame: f,
                major,
                minor,
            } => {
                let d = sub(p, f.o);
                let radial = dir(sub(d, scale(f.z, dot(d, f.z))), f.x);
                let ring = add(f.o, scale(radial, *major));
                let n = dir(sub(p, ring), radial);
                (add(ring, scale(n, *minor)), n)
            }
            Surface::Discrete(d) => d.closest(p),
            Surface::Tube { path, radius } => {
                let q = path.closest(p);
                // On the path: any radial direction.
                let n = dir(sub(p, q), [0.0, 0.0, 1.0]);
                (add(q, scale(n, *radius)), n)
            }
            Surface::Extruded { .. } | Surface::Nurbs(_) | Surface::Revolved { .. } => {
                let uv = self.param(p);
                (self.eval(uv), self.normal(uv))
            }
        }
    }

    /// Whether the nearest point is a search (an extrusion, a revolution, a
    /// B-spline, a tube, a discrete patch): one from the parameters of a
    /// point nearby is far cheaper than [`Surface::closest`].
    pub fn searches(&self) -> bool {
        matches!(
            self,
            Surface::Extruded { .. }
                | Surface::Nurbs(_)
                | Surface::Revolved { .. }
                | Surface::Tube { .. }
                | Surface::Discrete(_)
        )
    }

    /// Where a search for the nearest point to `p` starts later: its
    /// parameters on a spline carrier, its path segment on a tube, its
    /// facet on a discrete patch.
    pub fn search_start(&self, p: V3) -> V2 {
        match self {
            Surface::Tube { path, .. } => [path.closest_segment(p) as f64, 0.0],
            Surface::Discrete(d) => [d.closest_facet(p).2 as f64, 0.0],
            _ => self.param(p),
        }
    }

    /// [`Surface::closest`] searched from `uv0`, where the search for a
    /// point near the answer started (see [`Surface::search_start`]), with
    /// where the answer's starts (`uv0` again on a carrier that does not
    /// search).
    pub fn closest_near(&self, p: V3, uv0: V2) -> (V3, V2) {
        if !self.searches() {
            return (self.closest(p).0, uv0);
        }
        let uv = match self {
            Surface::Tube { path, radius } => {
                // Walked from the segment before; a point farther from the
                // path than the tube is wide may be nearer another turn.
                let (mut q, mut s) = path.closest_near(p, uv0[0] as usize);
                if dist(p, q) > 2.0 * radius {
                    q = path.closest(p);
                    s = path.closest_segment(p);
                }
                let n = unit(sub(p, q)).unwrap_or([0.0, 0.0, 1.0]);
                return (add(q, scale(n, *radius)), [s as f64, 0.0]);
            }
            Surface::Extruded { frame: f, profile } => {
                let l = f.local(p);
                [profile.closest_param_near([l[0], l[1]], uv0[0]), l[2]]
            }
            Surface::Nurbs(s) => s.closest_param_near(p, uv0),
            Surface::Discrete(d) => {
                let (q, _, t) = d.closest_facet_near(p, uv0[0] as usize);
                return (q, [t as f64, 0.0]);
            }
            Surface::Revolved { frame: f, profile } => {
                let (theta, r, z) = cylindrical(f, p);
                [theta, profile.closest_param_near([r, z], uv0[1])]
            }
            _ => unreachable!("searches"),
        };
        (self.eval(uv), uv)
    }

    /// A point of the surface near `p`, cheaper than
    /// [`Surface::closest_near`] where that searches a spline: one
    /// Gauss-Newton step from `uv0` on a B-spline (a point on it, not quite
    /// the nearest), the nearest point elsewhere. For comparing candidate
    /// places close to `uv0`.
    pub fn toward(&self, p: V3, uv0: V2) -> V3 {
        match self {
            Surface::Nurbs(s) => {
                let uv = s.step_toward(p, uv0);
                s.eval(uv[0], uv[1])
            }
            _ => self.closest_near(p, uv0).0,
        }
    }

    /// The point at the parameters `uv`; a discrete patch and a tube have
    /// none (they give `uv` back as a point of the plane z = 0).
    pub fn eval(&self, uv: V2) -> V3 {
        let [u, v] = uv;
        let round = |f: &Frame| add(scale(f.x, u.cos()), scale(f.y, u.sin()));
        match self {
            Surface::Plane(f) => f.at(u, v, 0.0),
            Surface::Cylinder { frame: f, radius } => {
                add(f.at(0.0, 0.0, v), scale(round(f), *radius))
            }
            Surface::Cone {
                frame: f,
                half_angle,
            } => add(f.at(0.0, 0.0, v), scale(round(f), v * half_angle.tan())),
            Surface::Sphere { frame: f, radius } => add(
                f.o,
                scale(add(scale(round(f), v.cos()), scale(f.z, v.sin())), *radius),
            ),
            Surface::Torus {
                frame: f,
                major,
                minor,
            } => add(
                f.at(0.0, 0.0, minor * v.sin()),
                scale(round(f), major + minor * v.cos()),
            ),
            Surface::Extruded { frame: f, profile } => {
                let c = profile.eval(u);
                f.at(c[0], c[1], v)
            }
            Surface::Revolved { frame: f, profile } => {
                let c = profile.eval(v);
                add(f.at(0.0, 0.0, c[1]), scale(round(f), c[0]))
            }
            Surface::Nurbs(s) => s.eval(u, v),
            Surface::Discrete(_) | Surface::Tube { .. } => [u, v, 0.0],
        }
    }

    /// The parameters of the nearest point to `p`; a discrete patch and a
    /// tube have none (they give `p`'s x and y back).
    pub fn param(&self, p: V3) -> V2 {
        match self {
            Surface::Plane(f) => {
                let l = f.local(p);
                [l[0], l[1]]
            }
            Surface::Cylinder { frame: f, .. } => {
                let (theta, _, z) = cylindrical(f, p);
                [theta, z]
            }
            Surface::Sphere { frame: f, .. } => {
                let d = normalize(sub(p, f.o));
                [
                    angle_about(d, f.x, f.y),
                    dot(d, f.z).clamp(-1.0, 1.0).asin(),
                ]
            }
            Surface::Cone {
                frame: f,
                half_angle,
            } => {
                // In the meridian half-plane, the foot on the ray from the
                // apex (the apex behind it).
                let (theta, r, z) = cylindrical(f, p);
                let (sin, cos) = half_angle.sin_cos();
                [theta, (r * sin + z * cos).max(0.0) * cos]
            }
            Surface::Torus {
                frame: f, major, ..
            } => {
                let (theta, r, z) = cylindrical(f, p);
                [theta, z.atan2(r - major)]
            }
            Surface::Extruded { frame: f, profile } => {
                let l = f.local(p);
                [profile.closest_param([l[0], l[1]]), l[2]]
            }
            Surface::Revolved { frame: f, profile } => {
                let (theta, r, z) = cylindrical(f, p);
                [theta, profile.closest_param([r, z])]
            }
            Surface::Nurbs(s) => s.closest_param(p),
            Surface::Discrete(_) | Surface::Tube { .. } => [p[0], p[1]],
        }
    }

    /// The outward unit normal at the parameters `uv` (+z for a discrete
    /// patch and a tube, which have none).
    pub fn normal(&self, uv: V2) -> V3 {
        let [u, v] = uv;
        let round = |f: &Frame| add(scale(f.x, u.cos()), scale(f.y, u.sin()));
        match self {
            Surface::Plane(f) => f.z,
            Surface::Cylinder { frame: f, .. } => round(f),
            Surface::Sphere { frame: f, .. } => add(scale(round(f), v.cos()), scale(f.z, v.sin())),
            Surface::Cone {
                frame: f,
                half_angle,
            } => sub(
                scale(round(f), half_angle.cos()),
                scale(f.z, half_angle.sin()),
            ),
            Surface::Torus { frame: f, .. } => add(scale(round(f), v.cos()), scale(f.z, v.sin())),
            Surface::Extruded { frame: f, profile } => {
                let (_, c1, _) = profile.ders2(u);
                normalize(rapidmesh_exact::vector::cross(
                    add(scale(f.x, c1[0]), scale(f.y, c1[1])),
                    f.z,
                ))
            }
            Surface::Revolved { frame: f, profile } => {
                // (dz, -dr) in the meridian, the solid on the profile's left.
                let (_, c1, _) = profile.ders2(v);
                normalize(sub(scale(round(f), c1[1]), scale(f.z, c1[0])))
            }
            Surface::Nurbs(s) => s.normal(u, v),
            Surface::Discrete(_) | Surface::Tube { .. } => [0.0, 0.0, 1.0],
        }
    }

    /// The point at `uv` and its derivatives `[S, S_u, S_v, S_uu, S_uv,
    /// S_vv]`, in closed form for every kind (a discrete patch and a tube,
    /// with no parameter map, give their point and no derivatives).
    pub fn ders(&self, uv: V2) -> [V3; 6] {
        let [u, v] = uv;
        let (s, c) = u.sin_cos();
        // The radial direction about z and its turn.
        let ring = |f: &Frame| {
            (
                add(scale(f.x, c), scale(f.y, s)),
                sub(scale(f.y, c), scale(f.x, s)),
            )
        };
        let zero = [0.0; 3];
        match self {
            Surface::Plane(f) => [f.at(u, v, 0.0), f.x, f.y, zero, zero, zero],
            Surface::Cylinder { frame: f, radius } => {
                let (r, t) = ring(f);
                let at = add(f.at(0.0, 0.0, v), scale(r, *radius));
                [at, scale(t, *radius), f.z, scale(r, -radius), zero, zero]
            }
            Surface::Cone {
                frame: f,
                half_angle,
            } => {
                let (r, t) = ring(f);
                let k = half_angle.tan();
                [
                    add(f.at(0.0, 0.0, v), scale(r, v * k)),
                    scale(t, v * k),
                    add(f.z, scale(r, k)),
                    scale(r, -v * k),
                    scale(t, k),
                    zero,
                ]
            }
            Surface::Sphere { frame: f, radius } => {
                let (r, t) = ring(f);
                let (sv, cv) = v.sin_cos();
                let out = add(scale(r, cv), scale(f.z, sv));
                [
                    add(f.o, scale(out, *radius)),
                    scale(t, radius * cv),
                    scale(sub(scale(f.z, cv), scale(r, sv)), *radius),
                    scale(r, -radius * cv),
                    scale(t, -radius * sv),
                    scale(out, -radius),
                ]
            }
            Surface::Torus {
                frame: f,
                major,
                minor,
            } => {
                let (r, t) = ring(f);
                let (sv, cv) = v.sin_cos();
                let rho = major + minor * cv;
                [
                    add(f.at(0.0, 0.0, minor * sv), scale(r, rho)),
                    scale(t, rho),
                    scale(sub(scale(f.z, cv), scale(r, sv)), *minor),
                    scale(r, -rho),
                    scale(t, -minor * sv),
                    scale(add(scale(r, cv), scale(f.z, sv)), -minor),
                ]
            }
            Surface::Extruded { frame: f, profile } => {
                let (p, d1, d2) = profile.ders2(u);
                let lift = |d: [f64; 2]| add(scale(f.x, d[0]), scale(f.y, d[1]));
                [f.at(p[0], p[1], v), lift(d1), f.z, lift(d2), zero, zero]
            }
            Surface::Revolved { frame: f, profile } => {
                // The meridian (rho, h) at v turned by u.
                let (r, t) = ring(f);
                let ([rho, h], [rho1, h1], [rho2, h2]) = profile.ders2(v);
                [
                    add(f.at(0.0, 0.0, h), scale(r, rho)),
                    scale(t, rho),
                    add(scale(f.z, h1), scale(r, rho1)),
                    scale(r, -rho),
                    scale(t, rho1),
                    add(scale(f.z, h2), scale(r, rho2)),
                ]
            }
            Surface::Nurbs(n) => n.ders2(u, v),
            Surface::Discrete(_) | Surface::Tube { .. } => {
                [self.eval(uv), zero, zero, zero, zero, zero]
            }
        }
    }

    /// How the surface bends at `uv`: the lengths of the first derivatives,
    /// the normal parts of the second along each parameter and of the mixed
    /// one. A step `d` along parameter `k` strays from the surface by about
    /// `d^2 bend[k] / 8`, one of `du` and `dv` across a cell by `du dv
    /// twist / 4`. Where the normal is lost, the whole second derivatives:
    /// at a pole one derivative vanishes beside the other (to rounding, so
    /// their cross product is measured against the longer one).
    pub fn bend(&self, uv: V2) -> Bend {
        let [_, su, sv, suu, suv, svv] = self.ders(uv);
        let n = cross(su, sv);
        let side = len(su).max(len(sv));
        let part = |x: V3| {
            if len(n) > 1e-9 * side * side {
                dot(x, n).abs() / len(n)
            } else {
                len(x)
            }
        };
        Bend {
            stretch: [len(su), len(sv)],
            bend: [part(suu), part(svv)],
            twist: part(suv),
        }
    }

    /// The period of each parameter that wraps around: a full turn for the
    /// angles, the domain of a B-spline direction (or of a profile) whose
    /// two ends meet.
    pub fn periods(&self) -> [Option<f64>; 2] {
        match self {
            Surface::Cylinder { .. } | Surface::Cone { .. } | Surface::Sphere { .. } => {
                [Some(TAU), None]
            }
            Surface::Torus { .. } => [Some(TAU), Some(TAU)],
            Surface::Extruded { profile, .. } => [closed_profile(profile), None],
            Surface::Revolved { profile, .. } => [Some(TAU), closed_profile(profile)],
            Surface::Nurbs(n) => {
                let (du, dv) = n.domain();
                let size = n
                    .ctrl
                    .iter()
                    .map(|c| dist(*c, n.ctrl[0]))
                    .fold(0.0, f64::max)
                    .max(1e-300);
                let closed = |k: usize| {
                    (0..=8).all(|i| {
                        let t = i as f64 / 8.0;
                        let (a, b) = if k == 0 {
                            let v = dv[0] + t * (dv[1] - dv[0]);
                            (n.eval(du[0], v), n.eval(du[1], v))
                        } else {
                            let u = du[0] + t * (du[1] - du[0]);
                            (n.eval(u, dv[0]), n.eval(u, dv[1]))
                        };
                        dist(a, b) <= 1e-9 * size
                    })
                };
                [
                    closed(0).then_some(du[1] - du[0]),
                    closed(1).then_some(dv[1] - dv[0]),
                ]
            }
            Surface::Plane(_) | Surface::Discrete(_) | Surface::Tube { .. } => [None, None],
        }
    }

    /// The lines of the parameters that are one point in space: the
    /// parameter held fixed along one, its value there and the point. The
    /// poles of a sphere, the apex of a cone, a meridian's end on the axis
    /// of a revolution, a side of a B-spline drawn together into a point.
    pub fn poles(&self) -> Vec<(usize, f64, V3)> {
        match self {
            Surface::Sphere { .. } => [-FRAC_PI_2, FRAC_PI_2]
                .map(|v| (1, v, self.eval([0.0, v])))
                .to_vec(),
            Surface::Cone { frame, .. } => vec![(1, 0.0, frame.o)],
            Surface::Revolved { profile, .. } => {
                let (lo, hi) = profile.domain();
                let size = profile
                    .ctrl
                    .iter()
                    .map(|c| c[0].abs())
                    .fold(0.0, f64::max)
                    .max(1e-300);
                [lo, hi]
                    .into_iter()
                    .filter(|&t| profile.eval(t)[0].abs() <= 1e-9 * size)
                    .map(|t| (1, t, self.eval([0.0, t])))
                    .collect()
            }
            Surface::Nurbs(n) => {
                let (du, dv) = n.domain();
                let size = n
                    .ctrl
                    .iter()
                    .map(|c| dist(*c, n.ctrl[0]))
                    .fold(0.0, f64::max)
                    .max(1e-300);
                let mut out = Vec::new();
                for (k, ends, other) in [(0, du, dv), (1, dv, du)] {
                    for value in ends {
                        let at = |t: f64| {
                            let w = other[0] + t * (other[1] - other[0]);
                            if k == 0 {
                                n.eval(value, w)
                            } else {
                                n.eval(w, value)
                            }
                        };
                        let p = at(0.0);
                        if (1..=8).all(|i| dist(at(i as f64 / 8.0), p) <= 1e-9 * size) {
                            out.push((k, value, p));
                        }
                    }
                }
                out
            }
            _ => Vec::new(),
        }
    }

    /// The principal curvatures `[k_max, k_min]` (magnitudes, 1/length) at
    /// the nearest point to `p`; none for a discrete patch (it has only a
    /// radius estimate) and at the apex of a cone.
    pub fn principal_curvatures(&self, p: V3) -> Option<[f64; 2]> {
        match self {
            Surface::Plane(_) => Some([0.0, 0.0]),
            Surface::Cylinder { radius, .. } | Surface::Tube { radius, .. } => {
                Some([1.0 / radius, 0.0])
            }
            Surface::Sphere { radius, .. } => Some([1.0 / radius, 1.0 / radius]),
            Surface::Cone { half_angle, .. } => {
                // Across the generator, the circle of radius rho seen on the
                // normal section: cos(half angle) / rho; along it, straight.
                let rho = self.param(p)[1] * half_angle.tan();
                (rho > 0.0).then(|| [half_angle.cos() / rho, 0.0])
            }
            Surface::Torus { major, minor, .. } => {
                // The tube circle, and the parallel circle seen on the
                // normal section: cos(phi) / (R + r cos(phi)).
                let c = self.param(p)[1].cos();
                let k = [1.0 / minor, (c / (major + minor * c)).abs()];
                Some([k[0].max(k[1]), k[0].min(k[1])])
            }
            Surface::Extruded { profile, .. } => Some([profile.curvature(self.param(p)[0]), 0.0]),
            Surface::Revolved { profile, .. } => {
                // The meridian, and the parallel circle of radius r seen on
                // the normal section: |n_r| / r (on the axis, a smooth pole
                // bends like the meridian).
                let t = self.param(p)[1];
                let (c, c1, _) = profile.ders2(t);
                let km = profile.curvature(t);
                let speed = c1[0].hypot(c1[1]);
                let kp = if c[0] > 1e-12 * speed.max(1.0) && speed > 0.0 {
                    (c1[1] / speed).abs() / c[0]
                } else {
                    km
                };
                Some([km.max(kp), km.min(kp)])
            }
            Surface::Nurbs(s) => {
                let uv = s.closest_param(p);
                s.principal_curvatures(uv[0], uv[1])
            }
            Surface::Discrete(_) => None,
        }
    }

    /// The smallest principal radius of curvature at the nearest point to
    /// `p` (infinite where flat): the input to the sagitta size bound.
    pub fn curvature_radius(&self, p: V3) -> f64 {
        match self {
            Surface::Discrete(d) => d.curvature_radius(p),
            _ => match self.principal_curvatures(p) {
                Some([k, _]) if k > 1e-12 => 1.0 / k,
                Some(_) => f64::INFINITY,
                // The apex of a cone.
                None => 1e-12,
            },
        }
    }

    /// The point nearest `p` where this surface and `other` meet: Newton on
    /// the two tangent planes (`p + alpha n_a + beta n_b` on both), which
    /// converges quadratically where they cross, an alternating projection
    /// where they nearly touch; stopped once a step moves less than `tol`.
    /// The caller guards against a start in another basin.
    pub fn meet(&self, other: &Surface, mut p: V3, tol: f64) -> V3 {
        for _ in 0..32 {
            let (fa, na) = self.closest(p);
            let (fb, nb) = other.closest(p);
            let c = dot(na, nb);
            let det = 1.0 - c * c;
            let q = if det > 1e-6 {
                let (ra, rb) = (dot(sub(fa, p), na), dot(sub(fb, p), nb));
                add(
                    add(p, scale(na, (ra - c * rb) / det)),
                    scale(nb, (rb - c * ra) / det),
                )
            } else {
                other.closest(fa).0
            };
            let moved = dist(p, q);
            p = q;
            if moved < tol {
                break;
            }
        }
        p
    }

    /// The axis the surface turns about by itself: a point on it and its
    /// unit direction. A plane and a sphere fix none (any axis along the
    /// normal, through the centre).
    pub fn axis(&self) -> Option<(V3, V3)> {
        match self {
            Surface::Cylinder { frame, .. }
            | Surface::Cone { frame, .. }
            | Surface::Torus { frame, .. }
            | Surface::Revolved { frame, .. } => Some((frame.o, frame.z)),
            _ => None,
        }
    }

    /// The meridian about the axis through `o` along the unit `a`: the
    /// section with a half-plane of the axis in (distance from it, height
    /// along it), where the surface is one of revolution about it (a plane
    /// square to it, a sphere centred on it, a cylinder, cone, torus or
    /// revolved profile about it); none otherwise. A centre counts as on
    /// the axis within `tol`, a direction as along it to a cosine within
    /// 1e-9 of 1.
    pub fn meridian(&self, o: V3, a: V3, tol: f64) -> Option<Curve<2>> {
        let z = |c: V3| dot(sub(c, o), a);
        let on_axis = |c: V3| len(sub(sub(c, o), scale(a, z(c)))) <= tol;
        let along = |b: V3| dot(b, a).abs() >= 1.0 - 1e-9;
        let about = |f: &Frame| along(f.z) && on_axis(f.o);
        let circle = |c: V2, r: f64| Curve::Ellipse {
            c,
            p: [r, 0.0],
            q: [0.0, r],
        };
        match self {
            Surface::Plane(f) => along(f.z).then_some(Curve::Line {
                p: [0.0, z(f.o)],
                d: [1.0, 0.0],
            }),
            Surface::Sphere { frame: f, radius } => {
                on_axis(f.o).then(|| circle([0.0, z(f.o)], *radius))
            }
            Surface::Cylinder { frame: f, radius } => about(f).then_some(Curve::Line {
                p: [*radius, 0.0],
                d: [0.0, 1.0],
            }),
            // The generator leaves the apex into the nappe, whichever way
            // the axis runs.
            Surface::Cone {
                frame: f,
                half_angle,
            } => about(f).then(|| {
                let (sin, cos) = half_angle.sin_cos();
                Curve::Line {
                    p: [0.0, z(f.o)],
                    d: [sin, dot(f.z, a).signum() * cos],
                }
            }),
            Surface::Torus {
                frame: f,
                major,
                minor,
            } => about(f).then(|| circle([*major, z(f.o)], *minor)),
            Surface::Revolved { frame: f, profile } => about(f).then(|| {
                Curve::Nurbs(profile.clone()).rescaled([1.0, dot(f.z, a).signum()], [0.0, z(f.o)])
            }),
            _ => None,
        }
    }

    /// The isolated singular point of the surface, if any (the apex of a
    /// cone): its tangent plane is undefined, and it may lie inside a face
    /// with no edge at it.
    pub fn singular_point(&self) -> Option<V3> {
        match self {
            Surface::Cone { frame, .. } => Some(frame.o),
            _ => None,
        }
    }
}

/// How a surface bends at a point (see [`Surface::bend`]).
#[derive(Clone, Copy, Debug)]
pub struct Bend {
    pub stretch: [f64; 2],
    pub bend: [f64; 2],
    pub twist: f64,
}

/// The domain length of a profile whose ends meet.
fn closed_profile(profile: &NurbsCurve) -> Option<f64> {
    let (lo, hi) = profile.domain();
    let size = profile
        .ctrl
        .iter()
        .map(|c| dist(*c, profile.ctrl[0]))
        .fold(0.0, f64::max)
        .max(1e-300);
    (dist(profile.eval(lo), profile.eval(hi)) <= 1e-9 * size).then_some(hi - lo)
}

/// The frame about `axis` through `o`, its `x` the ortho unit of the axis.
fn framed(o: V3, axis: V3) -> Frame {
    Frame::new(o, axis, None).expect("the axis of a surface is a nonzero vector")
}

/// The angle of `p` about the frame's `z` (from `x`), its distance from the
/// axis and its height along it.
fn cylindrical(f: &Frame, p: V3) -> (f64, f64, f64) {
    let l = f.local(p);
    (l[1].atan2(l[0]), l[0].hypot(l[1]), l[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn close(a: V3, b: V3) -> bool {
        dist(a, b) < 1e-9
    }

    #[test]
    fn a_plane_is_set_on_its_face() {
        let pts = [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
        ];
        let s = Surface::plane_through(&pts).unwrap().fitted(&pts);
        let p = [1.3, 2.1, 1.0];
        assert!(close(s.eval(s.param(p)), p));
        assert!(close(s.eval([0.0, 0.0]), [1.0, 1.5, 1.0]));
        assert!(s.curvature_radius(p).is_infinite());
        assert!(Surface::plane_through(&[
            [0.0; 3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.5]
        ])
        .is_none());
    }

    #[test]
    fn a_cylinder_maps_its_parameters_both_ways() {
        let s = Surface::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0);
        let p = s.eval([0.7, 1.5]);
        assert!((p[0].hypot(p[1]) - 2.0).abs() < 1e-12);
        assert!(close(s.eval(s.param(p)), p));
        assert_eq!(s.curvature_radius(p), 2.0);
    }

    #[test]
    fn a_sphere_maps_its_parameters_both_ways() {
        let c = [1.0, 0.0, 0.0];
        let s = Surface::sphere(c, 3.0);
        for uv in [[0.5, 0.4], [-1.2, -0.8], [PI * 0.5, 0.0]] {
            let p = s.eval(uv);
            assert!((dist(p, c) - 3.0).abs() < 1e-12);
            assert!(close(s.eval(s.param(p)), p));
            assert!(close(s.normal(uv), scale(sub(p, c), 1.0 / 3.0)));
        }
        assert_eq!(s.curvature_radius([4.0, 0.0, 0.0]), 3.0);
    }

    /// The nearest of a dense scan of `(u, v)`, as a reference.
    fn scanned(s: &Surface, p: V3, u: (f64, f64), v: (f64, f64)) -> f64 {
        let n = 400;
        let mut best = f64::INFINITY;
        for i in 0..=n {
            for j in 0..=n {
                let q = s.eval([
                    u.0 + (u.1 - u.0) * i as f64 / n as f64,
                    v.0 + (v.1 - v.0) * j as f64 / n as f64,
                ]);
                best = best.min(dist(q, p));
            }
        }
        best
    }

    #[test]
    fn a_cone_footpoint_is_the_nearest_point() {
        let half = 0.6f64.atan();
        let s = Surface::cone([0.0; 3], [0.0, 0.0, 1.0], half);
        // Off the barrel, outside and inside, and behind the apex.
        for p in [[2.0, 0.3, 0.5], [0.1, 0.0, 2.0], [0.4, 0.2, -0.7]] {
            let (q, _) = s.closest(p);
            let d = dist(q, p);
            let reference = scanned(&s, p, (-PI, PI), (0.0, 4.0));
            assert!(d <= reference + 1e-9, "{p:?}: {d} vs scan {reference}");
            assert!(reference - d < 1e-3, "{p:?}: {d} vs scan {reference}");
            assert!(close(s.eval(s.param(q)), q));
        }
        // Across the generator the normal section bends by cos / rho.
        let p = s.eval([0.3, 2.0]);
        let k = s.principal_curvatures(p).unwrap();
        assert!((k[0] - half.cos() / (2.0 * 0.6)).abs() < 1e-12 && k[1] == 0.0);
    }

    #[test]
    fn a_fat_torus_bends_tighter_than_its_tube_inside() {
        // R < 2r: on the inner equator the parallel circle, 1 / (R - r),
        // bends more than the tube, 1 / r.
        let s = Surface::torus([0.0; 3], [0.0, 0.0, 1.0], 1.5, 1.0);
        assert!((s.curvature_radius([0.5, 0.0, 0.0]) - 0.5).abs() < 1e-12);
        assert!((s.curvature_radius([2.5, 0.0, 0.0]) - 1.0).abs() < 1e-12);
        let p = s.eval([0.4, 2.0]);
        assert!(close(s.eval(s.param(p)), p));
        assert!(close(s.closest(p).1, s.normal([0.4, 2.0])));
    }

    #[test]
    fn a_revolved_half_circle_is_a_sphere() {
        // Rational quadratic half circle from the south to the north pole.
        let r = 1.5;
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let profile = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
            vec![[0.0, -r], [r, -r], [r, 0.0], [r, r], [0.0, r]],
            vec![1.0, w, 1.0, w, 1.0],
        );
        let c = [0.5, -1.0, 2.0];
        let s = Surface::Revolved {
            frame: framed(c, [0.0, 0.0, 1.0]),
            profile: Arc::new(profile),
        };
        for p in [[3.0, 0.2, 2.5], [0.4, -1.1, 2.1], [-2.0, -3.0, -1.0]] {
            let (q, n) = s.closest(p);
            let d = normalize(sub(p, c));
            assert!(close(q, add(c, scale(d, r))), "{q:?}");
            assert!(close(n, d), "outward normal {n:?}");
            let k = s.principal_curvatures(p).unwrap();
            assert!((k[0] - 1.0 / r).abs() < 1e-9 && (k[1] - 1.0 / r).abs() < 1e-9);
        }
    }

    #[test]
    fn derivatives_match_differences_on_every_kind() {
        let quarter = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0],
        );
        let f = Frame::new([0.3, -0.2, 1.0], [0.2, 1.0, 0.4], Some([1.0, 0.0, 0.0])).unwrap();
        let kinds = [
            Surface::Plane(f),
            Surface::Cylinder {
                frame: f,
                radius: 1.5,
            },
            Surface::Cone {
                frame: f,
                half_angle: 0.4,
            },
            Surface::Sphere {
                frame: f,
                radius: 2.0,
            },
            Surface::Torus {
                frame: f,
                major: 3.0,
                minor: 1.0,
            },
            Surface::Extruded {
                frame: f,
                profile: Arc::new(quarter.clone()),
            },
            Surface::Revolved {
                frame: f,
                profile: Arc::new(quarter),
            },
        ];
        let h = 1e-4;
        for s in &kinds {
            for uv in [[0.3, 0.4], [0.7, 0.2]] {
                let d = s.ders(uv);
                assert!(dist(d[0], s.eval(uv)) < 1e-12, "{}", s.name());
                let at = |du: f64, dv: f64| s.eval([uv[0] + du, uv[1] + dv]);
                let diff = |a: V3, b: V3, w: f64| scale(sub(a, b), 1.0 / w);
                let numeric = [
                    diff(at(h, 0.0), at(-h, 0.0), 2.0 * h),
                    diff(at(0.0, h), at(0.0, -h), 2.0 * h),
                    diff(add(at(h, 0.0), at(-h, 0.0)), scale(d[0], 2.0), h * h),
                    diff(
                        add(at(h, h), at(-h, -h)),
                        add(at(h, -h), at(-h, h)),
                        4.0 * h * h,
                    ),
                    diff(add(at(0.0, h), at(0.0, -h)), scale(d[0], 2.0), h * h),
                ];
                for (k, n) in numeric.iter().enumerate() {
                    assert!(
                        dist(d[k + 1], *n) < 1e-5,
                        "{} derivative {}",
                        s.name(),
                        k + 1
                    );
                }
            }
        }
    }

    #[test]
    fn meridians_are_the_sections_through_the_axis() {
        // Every kind about the axis through (1, 2, 3) along -z: a point of
        // the surface lies on its meridian at (distance, height).
        let (o, a) = ([1.0, 2.0, 3.0], [0.0, 0.0, -1.0]);
        let profile = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![[1.0, 0.0], [2.0, 1.0], [1.5, 2.0]],
            vec![1.0, 0.8, 1.0],
        );
        let kinds = [
            Surface::plane([0.0, 0.0, 1.0], a).unwrap(),
            Surface::sphere([1.0, 2.0, 0.5], 2.0),
            Surface::cylinder([1.0, 2.0, 0.0], [0.0, 0.0, 1.0], 1.5),
            Surface::cone([1.0, 2.0, 1.0], [0.0, 0.0, 1.0], 0.4),
            Surface::torus([1.0, 2.0, -1.0], [0.0, 0.0, 1.0], 3.0, 1.0),
            Surface::Revolved {
                frame: framed([1.0, 2.0, 0.2], [0.0, 0.0, 1.0]),
                profile: Arc::new(profile),
            },
        ];
        for s in &kinds {
            let m = s
                .meridian(o, a, 1e-12)
                .unwrap_or_else(|| panic!("{}", s.name()));
            for uv in [[0.3, 0.4], [2.0, 0.7]] {
                let p = s.eval(uv);
                let d = sub(p, o);
                let h = dot(d, a);
                let q = [len(sub(d, scale(a, h))), h];
                let t = m.param(q);
                let e = m.eval(t);
                assert!(dist(e, q) < 1e-9, "{} {q:?} vs {e:?}", s.name());
            }
            assert!(s.meridian([1.5, 2.0, 0.0], a, 1e-12).is_none() || s.is_plane());
        }
    }

    #[test]
    fn a_pole_bends_by_its_whole_second_derivatives() {
        // At the pole the turn's derivative is rounding beside the
        // latitude's: no normal to take parts along, the twist is all of r.
        let s = Surface::sphere([0.0; 3], 5.0);
        let b = s.bend([0.6, FRAC_PI_2]);
        assert!((b.twist - 5.0).abs() < 1e-9, "twist {}", b.twist);
        let b = s.bend([0.6, 0.3]);
        assert!(b.twist < 1e-12 && (b.bend[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn a_map_carries_the_carriers() {
        let turn = Affine::rotation([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.8).unwrap();
        let grow = Affine::stretch([0.0; 3], [2.0, 2.0, 2.0]);
        let m = turn.then(&grow);
        let s = Surface::torus([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 2.0, 0.5);
        let t = s.mapped(&m).unwrap();
        for uv in [[0.3, 1.0], [2.0, -2.5]] {
            assert!(close(t.closest(m.point(s.eval(uv))).0, m.point(s.eval(uv))));
        }
        assert!(s
            .mapped(&Affine::stretch([0.0; 3], [1.0, 2.0, 1.0]))
            .is_none());
        let mirror = Affine::mirror([0.0; 3], [1.0, 1.0, 0.0]).unwrap();
        let c = Surface::cylinder([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], 0.7)
            .mapped(&mirror)
            .unwrap();
        let p =
            mirror.point(Surface::cylinder([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], 0.7).eval([1.1, 0.4]));
        assert!(close(c.closest(p).0, p));
    }
}
