//! The unified surface interface.
//!
//! Every B-rep face references one [`Surface`]. It is a self-contained enum --
//! each variant carries the full parameters it needs to evaluate and invert,
//! including the frame a plane lacks in [`SurfaceKind`] -- with a single interface
//! (`closest` / `principal_curvatures` / `curvature_radius`, and the parameter
//! maps `eval_uv` / `project_uv` / `normal`). It is the one projection path:
//! the builder, the mesher, sizing and the diagnostics all query it. Every analytic carrier has a closed-form footpoint and curvature; the
//! extruded profile and a NURBS surface project by safeguarded Newton on their
//! own derivatives and take their curvature from them.

use rapidmesh_geom::nurbs::NurbsCurve;
use rapidmesh_geom::vec3::{add, cross, dist, dot, normalize as norm, ortho_unit, scale, sub, V3};
use rapidmesh_geom::{NurbsSurface, SurfaceKind};
use std::sync::Arc;

type P2 = [f64; 2];

/// The unit `x` made square to the unit axis `a` (it should be already;
/// rounding and transforms leave it a hair off), or [`ortho_unit`] of `a`
/// where `x` has no part off the axis.
fn square_to(x: V3, a: V3) -> V3 {
    let off = sub(x, scale(a, dot(x, a)));
    if dot(off, off) > 1e-24 {
        norm(off)
    } else {
        ortho_unit(a)
    }
}

/// A trimmed surface's underlying geometry, with one parameter-map interface.
#[derive(Debug, Clone)]
pub enum Surface {
    /// Plane with orthonormal frame `(o; u, v)` and `normal = u x v`.
    Plane { o: V3, u: V3, v: V3, normal: V3 },
    /// Cylinder: `(theta, h)`, `theta` from `x` about `axis`, `h` along `axis`.
    Cylinder {
        center: V3,
        axis: V3,
        x: V3,
        radius: f64,
    },
    /// Sphere: `(theta, phi)`, `phi` the latitude about `z`.
    Sphere {
        center: V3,
        x: V3,
        z: V3,
        radius: f64,
    },
    /// Cone: `(theta, s)`, `s` the signed distance from `apex` along `axis`.
    Cone {
        apex: V3,
        axis: V3,
        x: V3,
        half_angle: f64,
    },
    /// Torus: `(theta_major, phi_minor)`.
    Torus {
        center: V3,
        axis: V3,
        x: V3,
        major: f64,
        minor: f64,
    },
    /// Extruded profile: `(t, h)`, `profile(t)` in the `(u, v)` plane, `h` along `axis`.
    Extruded {
        base: V3,
        u: V3,
        v: V3,
        axis: V3,
        profile: Arc<NurbsCurve>,
    },
    /// Surface of revolution: `(theta, t)`, the meridian `profile(t) = (r,
    /// z)` turned by `theta` from `x` about `axis` through `origin`.
    Revolved {
        origin: V3,
        axis: V3,
        x: V3,
        profile: Arc<NurbsCurve>,
    },
    /// A NURBS surface, mapped by its own `(u, v)`.
    Nurbs(Arc<NurbsSurface>),
    /// A discrete smooth patch of an imported soup, queried by closest-point
    /// projection (no parameter map -- use [`Surface::closest`]).
    Discrete(Arc<rapidmesh_geom::DiscreteSurface>),
    /// Constant-radius tube about a polyline path (swept pipes, helix coils):
    /// `dist(p, path) = radius`. Queried by closest-point projection like
    /// [`Surface::Discrete`], but analytic per segment: smooth where it
    /// matters and with exact curvature `radius` for sizing.
    Tube {
        path: Arc<rapidmesh_geom::TubePath>,
        radius: f64,
    },
}

impl Surface {
    /// Builds the surface from a CSG [`SurfaceKind`]. `frame_pts` are points
    /// of the face, in order: a plane's frame starts at the first and points
    /// its `u` toward the next, so chart coordinates stay small; a faceted
    /// kind is framed by the plane they span. Every other kind is
    /// self-contained from its parameters.
    pub fn from_kind(kind: &SurfaceKind, frame_pts: &[V3]) -> Surface {
        match kind {
            SurfaceKind::Plane { .. } | SurfaceKind::Facets => {
                let (point, normal) = kind
                    .plane()
                    .or_else(|| SurfaceKind::plane_of(frame_pts).plane())
                    .unwrap_or(([0.0; 3], [0.0, 0.0, 1.0]));
                let onto = |p: V3| sub(p, scale(normal, dot(sub(p, point), normal)));
                let o = if frame_pts.is_empty() {
                    point
                } else {
                    let n = frame_pts.len() as f64;
                    onto(std::array::from_fn(|k| {
                        frame_pts.iter().map(|p| p[k]).sum::<f64>() / n
                    }))
                };
                let u = frame_pts
                    .iter()
                    .map(|&p| sub(onto(p), o))
                    .find(|d| dot(*d, *d) > 1e-20)
                    .map_or_else(|| ortho_unit(normal), norm);
                Surface::Plane {
                    o,
                    u,
                    v: cross(normal, u),
                    normal,
                }
            }
            SurfaceKind::Cylinder {
                center,
                axis,
                x,
                radius,
            } => {
                let a = norm(*axis);
                Surface::Cylinder {
                    center: *center,
                    axis: a,
                    x: square_to(*x, a),
                    radius: *radius,
                }
            }
            SurfaceKind::Sphere {
                center,
                axis,
                x,
                radius,
            } => {
                let z = norm(*axis);
                Surface::Sphere {
                    center: *center,
                    x: square_to(*x, z),
                    z,
                    radius: *radius,
                }
            }
            SurfaceKind::Cone {
                apex,
                axis,
                x,
                tan_half_angle,
            } => {
                let a = norm(*axis);
                Surface::Cone {
                    apex: *apex,
                    axis: a,
                    x: square_to(*x, a),
                    half_angle: tan_half_angle.atan(),
                }
            }
            SurfaceKind::Torus {
                center,
                axis,
                x,
                major_radius,
                minor_radius,
            } => {
                let a = norm(*axis);
                Surface::Torus {
                    center: *center,
                    axis: a,
                    x: square_to(*x, a),
                    major: *major_radius,
                    minor: *minor_radius,
                }
            }
            SurfaceKind::Extruded {
                profile,
                base,
                udir,
                vdir,
                axis,
            } => Surface::Extruded {
                base: *base,
                u: norm(*udir),
                v: norm(*vdir),
                axis: norm(*axis),
                profile: profile.clone(),
            },
            SurfaceKind::Discrete(d) => Surface::Discrete(d.clone()),
            SurfaceKind::Nurbs(n) => Surface::Nurbs(n.clone()),
            SurfaceKind::Revolved {
                profile,
                origin,
                axis,
                x,
            } => Surface::Revolved {
                origin: *origin,
                axis: norm(*axis),
                x: norm(*x),
                profile: profile.clone(),
            },
            SurfaceKind::Tube { path, radius } => Surface::Tube {
                path: path.clone(),
                radius: *radius,
            },
        }
    }

    /// The plane through `o` with unit normal `normal`, framed by an arbitrary
    /// in-plane `u`.
    pub fn plane(o: V3, normal: V3) -> Surface {
        let u = ortho_unit(normal);
        Surface::Plane {
            o,
            u,
            v: cross(normal, u),
            normal,
        }
    }

    /// The carrier of a curved kind; none for a plane or faceted kind.
    pub fn curved(kind: &SurfaceKind) -> Option<Surface> {
        (!matches!(kind, SurfaceKind::Plane { .. } | SurfaceKind::Facets))
            .then(|| Surface::from_kind(kind, &[]))
    }

    /// Closest point on the surface and the outward normal there: the
    /// projection of the meshing path, the improver and the diagnostics.
    /// Closed form and
    /// without trigonometry for every analytic kind; the extruded profile goes
    /// through its parameter map, a discrete patch through its facets.
    pub fn closest(&self, p: V3) -> (V3, V3) {
        // A unit direction, or `fallback` where it is undefined (on an axis).
        let unit = |d: V3, fallback: V3| -> V3 {
            let l = dot(d, d).sqrt();
            if l > 0.0 {
                scale(d, 1.0 / l)
            } else {
                fallback
            }
        };
        match self {
            // In-plane frame, not `p - n (n . d)`: an axis-aligned plane
            // then keeps its coordinate bit-exact.
            Surface::Plane { o, u, v, normal } => {
                let d = sub(p, *o);
                (
                    add(*o, add(scale(*u, dot(d, *u)), scale(*v, dot(d, *v)))),
                    *normal,
                )
            }
            Surface::Cylinder {
                center,
                axis,
                x,
                radius,
            } => {
                let d = sub(p, *center);
                let foot = add(*center, scale(*axis, dot(d, *axis)));
                let n = unit(sub(p, foot), *x);
                (add(foot, scale(n, *radius)), n)
            }
            Surface::Sphere {
                center, x, radius, ..
            } => {
                let n = unit(sub(p, *center), *x);
                (add(*center, scale(n, *radius)), n)
            }
            Surface::Cone {
                apex,
                axis,
                x,
                half_angle,
            } => {
                // In the meridian half-plane the nappe is the ray from the
                // apex along (sin, cos) of the half angle; behind the apex the
                // apex itself is closest.
                let d = sub(p, *apex);
                let h = dot(d, *axis);
                let rd = sub(d, scale(*axis, h));
                let r = dot(rd, rd).sqrt();
                let dir = unit(rd, *x);
                let (sin, cos) = half_angle.sin_cos();
                let t = (r * sin + h * cos).max(0.0);
                let q = add(*apex, scale(add(scale(dir, sin), scale(*axis, cos)), t));
                (q, sub(scale(dir, cos), scale(*axis, sin)))
            }
            Surface::Torus {
                center,
                axis,
                x,
                major,
                minor,
            } => {
                let d = sub(p, *center);
                let pdir = unit(sub(d, scale(*axis, dot(d, *axis))), *x);
                let ring = add(*center, scale(pdir, *major));
                let n = unit(sub(p, ring), pdir);
                (add(ring, scale(n, *minor)), n)
            }
            Surface::Discrete(d) => d.closest(p),
            Surface::Tube { path, radius } => {
                let q = path.closest(p);
                // p ON the axis: any radial direction
                let n = unit(sub(p, q), [0.0, 0.0, 1.0]);
                (add(q, scale(n, *radius)), n)
            }
            Surface::Extruded { .. } | Surface::Nurbs(_) | Surface::Revolved { .. } => {
                let uv = self.project_uv(p);
                (self.eval_uv(uv), self.normal(uv))
            }
        }
    }

    /// Whether the carrier is a spline (an extrusion, a revolution, a
    /// B-spline surface), whose nearest point is a search: one from the
    /// parameters of a point nearby is far cheaper than [`Surface::closest`].
    pub fn searches(&self) -> bool {
        matches!(
            self,
            Surface::Extruded { .. }
                | Surface::Nurbs(_)
                | Surface::Revolved { .. }
                | Surface::Tube { .. }
        )
    }

    /// Where a search for the nearest point to `p` starts later: its
    /// parameters on a spline carrier, its path segment on a tube.
    pub fn search_start(&self, p: V3) -> P2 {
        match self {
            Surface::Tube { path, .. } => [path.closest_segment(p) as f64, 0.0],
            _ => self.project_uv(p),
        }
    }

    /// [`Surface::closest`] searched from `uv0`, where the search for a
    /// point near the answer started (see [`Surface::search_start`]), with
    /// where the answer's starts (`uv0` again on a carrier that does not
    /// search).
    pub fn closest_near(&self, p: V3, uv0: P2) -> (V3, V3, P2) {
        if !self.searches() {
            let (q, n) = self.closest(p);
            return (q, n, uv0);
        }
        if let Surface::Tube { path, radius } = self {
            // Walked from the segment before; a point farther from the path
            // than the tube is wide may be nearer another turn of it.
            let (mut q, mut s) = path.closest_near(p, uv0[0] as usize);
            if dist(p, q) > 2.0 * radius {
                q = path.closest(p);
                s = path.closest_segment(p);
            }
            let d = sub(p, q);
            let l = dot(d, d).sqrt();
            let n = if l > 0.0 {
                scale(d, 1.0 / l)
            } else {
                [0.0, 0.0, 1.0]
            };
            return (add(q, scale(n, *radius)), n, [s as f64, 0.0]);
        }
        let uv = match self {
            Surface::Extruded {
                base,
                u,
                v,
                axis,
                profile,
            } => {
                let rel = sub(p, *base);
                [
                    profile.closest_param_near([dot(rel, *u), dot(rel, *v)], uv0[0]),
                    dot(rel, *axis),
                ]
            }
            Surface::Nurbs(s) => s.closest_param_near(p, uv0),
            Surface::Revolved {
                origin,
                axis,
                x,
                profile,
            } => {
                let y = cross(*axis, *x);
                let d = sub(p, *origin);
                let z = dot(d, *axis);
                let rd = sub(d, scale(*axis, z));
                let r = dot(rd, rd).sqrt();
                [
                    dot(rd, y).atan2(dot(rd, *x)),
                    profile.closest_param_near([r, z], uv0[1]),
                ]
            }
            _ => self.project_uv(p),
        };
        (self.eval_uv(uv), self.normal(uv), uv)
    }

    /// A point of the surface near `p`, cheaper than [`Surface::closest_near`]
    /// where that searches a spline: one Gauss-Newton step from `uv0` on a
    /// NURBS carrier (a point on it, not quite the nearest), the nearest
    /// point elsewhere. For comparing candidate places close to `uv0`.
    pub fn toward(&self, p: V3, uv0: P2) -> V3 {
        match self {
            Surface::Nurbs(s) => {
                let uv = s.step_toward(p, uv0);
                s.eval(uv[0], uv[1])
            }
            _ => self.closest_near(p, uv0).0,
        }
    }

    /// Parameter point `(u, v)` -> 3D.
    pub fn eval_uv(&self, p: P2) -> V3 {
        match self {
            Surface::Plane { o, u, v, .. } => add(*o, add(scale(*u, p[0]), scale(*v, p[1]))),
            Surface::Cylinder {
                center,
                axis,
                x,
                radius,
            } => {
                let y = cross(*axis, *x);
                let r = add(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                add(add(*center, scale(*axis, p[1])), scale(r, *radius))
            }
            Surface::Sphere {
                center,
                x,
                z,
                radius,
            } => {
                let y = cross(*z, *x);
                let eq = add(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                let dir = add(scale(eq, p[1].cos()), scale(*z, p[1].sin()));
                add(*center, scale(dir, *radius))
            }
            Surface::Cone {
                apex,
                axis,
                x,
                half_angle,
            } => {
                let y = cross(*axis, *x);
                let rho = p[1] * half_angle.tan();
                let r = add(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                add(add(*apex, scale(*axis, p[1])), scale(r, rho))
            }
            Surface::Torus {
                center,
                axis,
                x,
                major,
                minor,
            } => {
                let y = cross(*axis, *x);
                let dir = add(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                let ring = scale(dir, major + minor * p[1].cos());
                add(add(*center, ring), scale(*axis, minor * p[1].sin()))
            }
            Surface::Extruded {
                base,
                u,
                v,
                axis,
                profile,
            } => {
                let c = profile.eval(p[0]);
                add(
                    add(*base, scale(*axis, p[1])),
                    add(scale(*u, c[0]), scale(*v, c[1])),
                )
            }
            Surface::Nurbs(s) => s.eval(p[0], p[1]),
            Surface::Revolved {
                origin,
                axis,
                x,
                profile,
            } => {
                let c = profile.eval(p[1]);
                let y = cross(*axis, *x);
                let dir = add(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                add(add(*origin, scale(*axis, c[1])), scale(dir, c[0]))
            }
            // no parameter map: the (u,v) API is chart territory, and discrete
            // patches never chart -- callers on the meshing path use `closest`
            Surface::Discrete(_) => [p[0], p[1], 0.0],
            Surface::Tube { .. } => [p[0], p[1], 0.0],
        }
    }

    /// 3D point -> parameter `(u, v)` (closest point on the surface).
    pub fn project_uv(&self, p: V3) -> P2 {
        match self {
            Surface::Plane { o, u, v, .. } => [dot(sub(p, *o), *u), dot(sub(p, *o), *v)],
            Surface::Cylinder {
                center, axis, x, ..
            } => {
                let y = cross(*axis, *x);
                let d = sub(p, *center);
                let h = dot(d, *axis);
                let rd = sub(d, scale(*axis, h));
                [dot(rd, y).atan2(dot(rd, *x)), h]
            }
            Surface::Sphere { center, x, z, .. } => {
                let y = cross(*z, *x);
                let d = norm(sub(p, *center));
                [
                    dot(d, y).atan2(dot(d, *x)),
                    dot(d, *z).clamp(-1.0, 1.0).asin(),
                ]
            }
            Surface::Cone {
                apex,
                axis,
                x,
                half_angle,
            } => {
                // In the meridian half-plane the nappe is the ray from the
                // apex along (sin, cos) of the half angle; behind the apex the
                // apex itself is closest.
                let y = cross(*axis, *x);
                let d = sub(p, *apex);
                let h = dot(d, *axis);
                let rd = sub(d, scale(*axis, h));
                let r = dot(rd, rd).sqrt();
                let (sin, cos) = half_angle.sin_cos();
                let t = (r * sin + h * cos).max(0.0);
                [dot(rd, y).atan2(dot(rd, *x)), t * cos]
            }
            Surface::Torus {
                center,
                axis,
                x,
                major,
                ..
            } => {
                let y = cross(*axis, *x);
                let d = sub(p, *center);
                let zc = dot(d, *axis);
                let pd = sub(d, scale(*axis, zc));
                let theta = dot(pd, y).atan2(dot(pd, *x));
                let rho = dot(pd, pd).sqrt() - major;
                [theta, zc.atan2(rho)]
            }
            Surface::Extruded {
                base,
                u,
                v,
                axis,
                profile,
            } => {
                let rel = sub(p, *base);
                [
                    profile.closest_param([dot(rel, *u), dot(rel, *v)]),
                    dot(rel, *axis),
                ]
            }
            Surface::Nurbs(s) => s.closest_param(p),
            Surface::Revolved {
                origin,
                axis,
                x,
                profile,
            } => {
                // Into the meridian half-plane of `p`, then onto the profile.
                let y = cross(*axis, *x);
                let d = sub(p, *origin);
                let z = dot(d, *axis);
                let rd = sub(d, scale(*axis, z));
                let r = dot(rd, rd).sqrt();
                [dot(rd, y).atan2(dot(rd, *x)), profile.closest_param([r, z])]
            }
            Surface::Discrete(_) => [p[0], p[1]],
            Surface::Tube { .. } => [p[0], p[1]],
        }
    }

    /// Outward unit normal at parameter `(u, v)`.
    pub fn normal(&self, p: P2) -> V3 {
        match self {
            Surface::Plane { normal, .. } => *normal,
            Surface::Cylinder { axis, x, .. } => {
                let y = cross(*axis, *x);
                add(scale(*x, p[0].cos()), scale(y, p[0].sin()))
            }
            Surface::Sphere { center, .. } => norm(sub(self.eval_uv(p), *center)),
            Surface::Cone {
                axis,
                x,
                half_angle,
                ..
            } => {
                let y = cross(*axis, *x);
                let radial = add(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                norm(sub(
                    scale(radial, half_angle.cos()),
                    scale(*axis, half_angle.sin()),
                ))
            }
            Surface::Torus {
                center,
                axis,
                x,
                major,
                ..
            } => {
                let y = cross(*axis, *x);
                let dir = add(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                let tube_center = add(*center, scale(dir, *major));
                norm(sub(self.eval_uv(p), tube_center))
            }
            Surface::Extruded {
                u,
                v,
                axis,
                profile,
                ..
            } => {
                let (_, c1, _) = profile.ders2(p[0]);
                let tangent = add(scale(*u, c1[0]), scale(*v, c1[1]));
                norm(cross(tangent, *axis))
            }
            Surface::Nurbs(s) => s.normal(p[0], p[1]),
            Surface::Revolved {
                axis, x, profile, ..
            } => {
                // (dz, -dr) in the meridian, the solid on the profile's left.
                let (_, c1, _) = profile.ders2(p[1]);
                let y = cross(*axis, *x);
                let dir = add(scale(*x, p[0].cos()), scale(y, p[0].sin()));
                norm(sub(scale(dir, c1[1]), scale(*axis, c1[0])))
            }
            Surface::Discrete(_) => [0.0, 0.0, 1.0],
            Surface::Tube { .. } => [0.0, 0.0, 1.0],
        }
    }

    /// Principal curvatures `[k_max, k_min]` (magnitudes, 1/length) at the
    /// footpoint of `p`, in closed form; none for a discrete patch (only a
    /// radius estimate exists there) and at the apex of a cone.
    pub fn principal_curvatures(&self, p: V3) -> Option<[f64; 2]> {
        match self {
            Surface::Plane { .. } => Some([0.0, 0.0]),
            Surface::Cylinder { radius, .. } | Surface::Tube { radius, .. } => {
                Some([1.0 / radius, 0.0])
            }
            Surface::Sphere { radius, .. } => Some([1.0 / radius, 1.0 / radius]),
            Surface::Cone { half_angle, .. } => {
                // Across the generator, the circle of radius rho seen on the
                // normal section: cos(half angle) / rho; along it, straight.
                let rho = self.project_uv(p)[1] * half_angle.tan();
                (rho > 0.0).then(|| [half_angle.cos() / rho, 0.0])
            }
            Surface::Torus { major, minor, .. } => {
                // The tube circle, and the parallel circle seen on the normal
                // section: cos(phi) / (R + r cos(phi)).
                let c = self.project_uv(p)[1].cos();
                let k = [1.0 / minor, (c / (major + minor * c)).abs()];
                Some([k[0].max(k[1]), k[0].min(k[1])])
            }
            Surface::Extruded { profile, .. } => {
                Some([profile.curvature(self.project_uv(p)[0]), 0.0])
            }
            Surface::Nurbs(s) => {
                let uv = s.closest_param(p);
                s.principal_curvatures(uv[0], uv[1])
            }
            Surface::Revolved { profile, .. } => {
                // The meridian, and the parallel circle of radius r seen on
                // the normal section: |n_r| / r (on the axis, a smooth pole
                // bends like the meridian).
                let t = self.project_uv(p)[1];
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
            Surface::Discrete(_) => None,
        }
    }

    /// Smallest principal radius of curvature at the footpoint of `p`
    /// (`INFINITY` where flat), the input to the sagitta size bound.
    pub fn curvature_radius(&self, p: V3) -> f64 {
        match self {
            Surface::Discrete(d) => d.curvature_radius(p),
            _ => match self.principal_curvatures(p) {
                Some([k, _]) if k > 1e-12 => 1.0 / k,
                Some(_) => f64::INFINITY,
                // the apex of a cone
                None => 1e-12,
            },
        }
    }

    /// The surface's isolated SINGULAR point, if any (the cone apex): a point
    /// where the tangent plane is undefined. Such a point can sit in a face's
    /// interior with no incident B-rep edge, so topology alone never has it.
    /// The chart of the face takes it as a point of the face's own.
    pub fn singular_point(&self) -> Option<V3> {
        match self {
            Surface::Cone { apex, .. } => Some(*apex),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn close(a: V3, b: V3) -> bool {
        sub(a, b).iter().all(|&x| x.abs() < 1e-9)
    }

    #[test]
    fn plane_roundtrip() {
        let s = Surface::from_kind(
            &SurfaceKind::Facets,
            &[
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [2.0, 3.0, 1.0],
                [0.0, 3.0, 1.0],
            ],
        );
        let p = [1.3, 2.1, 1.0];
        assert!(
            close(s.eval_uv(s.project_uv(p)), p),
            "plane eval/project roundtrip"
        );
        assert!(matches!(s, Surface::Plane { .. }));
        assert!(s.curvature_radius(p).is_infinite());
    }

    #[test]
    fn cylinder_roundtrip_and_radius() {
        let s = Surface::from_kind(
            &SurfaceKind::cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0),
            &[],
        );
        // a point exactly on the barrel
        let p = s.eval_uv([0.7, 1.5]);
        assert!((dot([p[0], p[1], 0.0], [p[0], p[1], 0.0]).sqrt() - 2.0).abs() < 1e-9);
        let uv = s.project_uv(p);
        assert!(close(s.eval_uv(uv), p), "cylinder roundtrip");
        assert_eq!(s.curvature_radius(p), 2.0);
    }

    #[test]
    fn sphere_roundtrip() {
        let s = Surface::from_kind(&SurfaceKind::sphere([1.0, 0.0, 0.0], 3.0), &[]);
        for &(t, f) in &[(0.5, 0.4), (-1.2, -0.8), (PI * 0.5, 0.0)] {
            let p = s.eval_uv([t, f]);
            assert!(
                (sub(p, [1.0, 0.0, 0.0])
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    .sqrt()
                    - 3.0)
                    .abs()
                    < 1e-9
            );
            assert!(close(s.eval_uv(s.project_uv(p)), p), "sphere roundtrip");
        }
        assert_eq!(s.curvature_radius([4.0, 0.0, 0.0]), 3.0);
    }

    /// The nearest of a dense scan of `(u, v)`, as a reference footpoint.
    fn scanned(s: &Surface, p: V3, u: (f64, f64), v: (f64, f64)) -> f64 {
        let n = 400;
        let mut best = f64::INFINITY;
        for i in 0..=n {
            for j in 0..=n {
                let q = s.eval_uv([
                    u.0 + (u.1 - u.0) * i as f64 / n as f64,
                    v.0 + (v.1 - v.0) * j as f64 / n as f64,
                ]);
                best = best.min(dot(sub(q, p), sub(q, p)).sqrt());
            }
        }
        best
    }

    #[test]
    fn cone_footpoint_is_the_nearest_point() {
        let s = Surface::from_kind(
            &SurfaceKind::cone([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.6),
            &[],
        );
        // Off the barrel, outside and inside, and behind the apex.
        for p in [[2.0, 0.3, 0.5], [0.1, 0.0, 2.0], [0.4, 0.2, -0.7]] {
            let (q, _) = s.closest(p);
            let d = dot(sub(q, p), sub(q, p)).sqrt();
            let reference = scanned(&s, p, (-PI, PI), (0.0, 4.0));
            assert!(d <= reference + 1e-9, "{p:?}: {d} vs scan {reference}");
            assert!(reference - d < 1e-3, "{p:?}: {d} vs scan {reference}");
        }
        // Across the generator the normal section bends by cos / rho.
        let p = s.eval_uv([0.3, 2.0]);
        let rho = 2.0 * 0.6;
        let k = s.principal_curvatures(p).unwrap();
        assert!((k[0] - 0.6f64.atan().cos() / rho).abs() < 1e-12 && k[1] == 0.0);
    }

    #[test]
    fn fat_torus_bends_tighter_than_its_tube_inside() {
        // R < 2r: on the inner equator the parallel circle, 1 / (R - r),
        // bends more than the tube, 1 / r.
        let s = Surface::from_kind(
            &SurfaceKind::torus([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1.5, 1.0),
            &[],
        );
        let inner = [0.5, 0.0, 0.0];
        assert!((s.curvature_radius(inner) - 0.5).abs() < 1e-12);
        let outer = [2.5, 0.0, 0.0];
        assert!((s.curvature_radius(outer) - 1.0).abs() < 1e-12);
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
        let s = Surface::from_kind(
            &SurfaceKind::Revolved {
                profile: Arc::new(profile),
                origin: c,
                axis: [0.0, 0.0, 1.0],
                x: [1.0, 0.0, 0.0],
            },
            &[],
        );
        for p in [[3.0, 0.2, 2.5], [0.4, -1.1, 2.1], [-2.0, -3.0, -1.0]] {
            let (q, n) = s.closest(p);
            let d = sub(p, c);
            let l = dot(d, d).sqrt();
            let want = add(c, scale(d, r / l));
            assert!(close(q, want), "{q:?} vs {want:?}");
            assert!(close(n, scale(d, 1.0 / l)), "outward normal {n:?}");
            let k = s.principal_curvatures(p).unwrap();
            assert!(
                (k[0] - 1.0 / r).abs() < 1e-9 && (k[1] - 1.0 / r).abs() < 1e-9,
                "{k:?}"
            );
        }
    }
}
