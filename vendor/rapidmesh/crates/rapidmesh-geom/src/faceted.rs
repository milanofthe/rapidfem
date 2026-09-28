//! Faceted shapes: tessellated triangle meshes with analytic surface
//! back-references.

use crate::nurbs::NurbsCurve;
use rapidmesh_csg::{PlanarFacet, Solid, Tri};
use std::sync::Arc;

/// A flat (planar) face carried both as its helper triangulation (the range of
/// `Faceted::tris` that tiles it) and as its first-class boundary polygon. The
/// conformal arrangement intersects the helper triangles but accumulates the
/// resulting constraints at the polygon level, so a flat face has no interior
/// vertices of its own and curves piercing it land on their own vertices.
/// Curved faces (barrels, spheres,
/// tori) carry no `FlatFacet`; their triangles stay first-class.
#[derive(Debug, Clone)]
pub struct FlatFacet {
    /// Boundary loops of the face, in this shape's coordinates.
    pub facet: PlanarFacet,
    /// Index into `Faceted::surfaces` (shared with the helper triangles).
    pub surface: u32,
    /// The helper triangles of this face: `Faceted::tris[tris]`.
    pub tris: std::ops::Range<usize>,
}

/// The analytic surface a facet was tessellated from. Flat facets need no
/// snapping; curved kinds carry the data the order-2 midside snapping stage
/// projects onto. Metadata only — exactness of the mesh never depends on it.
#[derive(Debug, Clone)]
pub enum SurfaceKind {
    /// A flat face (the triangle itself is the exact surface).
    Plane,
    /// A DISCRETE smooth patch of an imported triangle soup (an STL region
    /// between crease edges): the carrier is the patch itself, queried by
    /// closest-point projection, so the import is REMESHED against its own
    /// envelope instead of frozen facet by facet.
    Discrete(std::sync::Arc<crate::discrete::DiscreteSurface>),
    /// An infinite cylinder barrel.
    Cylinder {
        /// A point on the axis.
        center: [f64; 3],
        /// Axis direction (not necessarily unit).
        axis: [f64; 3],
        /// Barrel radius.
        radius: f64,
    },
    /// A sphere.
    Sphere {
        /// Center.
        center: [f64; 3],
        /// Radius.
        radius: f64,
    },
    /// A cone barrel (frustum side with distinct radii).
    Cone {
        /// Apex point.
        apex: [f64; 3],
        /// Axis direction from apex into the cone (not necessarily unit).
        axis: [f64; 3],
        /// Tangent of the half-opening angle.
        tan_half_angle: f64,
    },
    /// A torus.
    Torus {
        /// Center of the major circle.
        center: [f64; 3],
        /// Axis direction normal to the major circle's plane (not
        /// necessarily unit).
        axis: [f64; 3],
        /// Major radius (center to tube center).
        major_radius: f64,
        /// Minor radius (tube radius).
        minor_radius: f64,
    },
    /// A 2D profile curve linearly extruded along `axis` (a developable swept
    /// surface): the surface point is
    /// `base + cx*udir + cy*vdir + h*axis`, where `(cx, cy) = profile(t)` and
    /// `h` is the extrusion height. `udir`/`vdir`/`axis` are an orthonormal
    /// frame (profile-x, profile-y, extrusion). Covers airfoils and any swept
    /// section; the curve carries the exact curvature for the sizing bias.
    Extruded {
        /// The 2D profile curve (in `(udir, vdir)` coordinates).
        profile: Arc<NurbsCurve>,
        /// Origin of the profile plane (the `h = 0` plane).
        base: [f64; 3],
        /// Unit profile-x direction in 3D.
        udir: [f64; 3],
        /// Unit profile-y direction in 3D.
        vdir: [f64; 3],
        /// Unit extrusion direction in 3D.
        axis: [f64; 3],
    },
    /// A constant-radius tube around a polyline path (swept pipes, helical
    /// coils): the surface is `dist(p, path) = radius`. The path is the
    /// SMOOTH sweep centerline; the offset/projection oracle is analytic per
    /// segment, so the carrier is smooth in the ways that matter (no facet
    /// coplanarity, exact curvature `radius` for sizing).
    Tube {
        /// Sweep centerline with its closest-point accelerator.
        path: Arc<crate::tube::TubePath>,
        /// Tube radius.
        radius: f64,
    },
    /// A profile curve in the half-plane `(r, z)` of an axis, turned about
    /// it: `profile(t) = (r, z)` at angle `theta` is the point
    /// `origin + z axis + r (cos(theta) x + sin(theta) axis x x)`. The solid
    /// lies left of the profile, so `(dz, -dr)` points out.
    Revolved {
        /// The meridian `(r, z)`, `r >= 0`.
        profile: Arc<NurbsCurve>,
        /// A point on the axis, `z = 0`.
        origin: [f64; 3],
        /// Unit axis direction.
        axis: [f64; 3],
        /// Unit direction of `theta = 0`, square to the axis.
        x: [f64; 3],
    },
    /// A NURBS surface, the general free-form carrier. Affine maps act on its
    /// control points exactly, so it survives any transform.
    Nurbs(Arc<crate::NurbsSurface>),
}

/// An affine map from a shape's own frame to the scene: `x -> linear x +
/// offset`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub linear: [[f64; 3]; 3],
    pub offset: [f64; 3],
}

impl Frame {
    pub const IDENTITY: Frame = Frame {
        linear: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        offset: [0.0; 3],
    };

    /// This frame followed by the map `x -> linear x + offset`.
    pub fn then(&self, linear: [[f64; 3]; 3], offset: [f64; 3]) -> Frame {
        let l = &self.linear;
        Frame {
            linear: std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..3).map(|k| linear[i][k] * l[k][j]).sum())
            }),
            offset: std::array::from_fn(|i| {
                (0..3).map(|k| linear[i][k] * self.offset[k]).sum::<f64>() + offset[i]
            }),
        }
    }

    /// The point of the scene `p` in the frame (the inverse map).
    pub fn to_local(&self, p: [f64; 3]) -> [f64; 3] {
        let a = &self.linear;
        let d: [f64; 3] = std::array::from_fn(|k| p[k] - self.offset[k]);
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        if !(det.abs() > 0.0) {
            return p;
        }
        // Cramer's rule.
        let col = |j: usize| -> [f64; 3] { [a[0][j], a[1][j], a[2][j]] };
        let det3 = |c0: [f64; 3], c1: [f64; 3], c2: [f64; 3]| {
            c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) - c1[0] * (c0[1] * c2[2] - c0[2] * c2[1])
                + c2[0] * (c0[1] * c1[2] - c0[2] * c1[1])
        };
        let (c0, c1, c2) = (col(0), col(1), col(2));
        [
            det3(d, c1, c2) / det,
            det3(c0, d, c2) / det,
            det3(c0, c1, d) / det,
        ]
    }
}

/// A tessellated shape: triangles plus, per triangle, the analytic surface
/// it approximates. Used for both closed solids and open sheets; closedness
/// and orientation are invariants of the builder that produced it.
#[derive(Debug, Clone)]
pub struct Faceted {
    /// Where the shape was built: the map from its own frame to the scene
    /// (identity for a new shape, followed by every transform). Entities
    /// that nothing else tells apart are ordered by where they lie in it,
    /// which a turn or a move of the shape leaves as it is.
    pub frame: Frame,
    /// Points of its boundary the shape declares as corners (the vertices
    /// of a polygon sheet): its B-rep edges end there even where nothing
    /// else would tell (one face, one boundary).
    pub corners: Vec<[f64; 3]>,
    /// The triangles.
    pub tris: Vec<Tri>,
    /// Per-triangle index into `surfaces`.
    pub face_surface: Vec<u32>,
    /// The distinct analytic surfaces of this shape.
    pub surfaces: Vec<SurfaceKind>,
    /// Flat faces as first-class boundary polygons (each backed by a range of
    /// `tris`). Empty until a builder registers them; curved faces never do.
    pub flats: Vec<FlatFacet>,
    /// Feature segments along triangle edges that no change of surface marks:
    /// creases inside one smooth region of an import (an open crease a region
    /// wraps around). They become B-rep edges inside their face.
    pub features: Vec<[[f64; 3]; 2]>,
}

impl Faceted {
    /// Empty shape.
    pub fn new() -> Faceted {
        Faceted {
            frame: Frame::IDENTITY,
            corners: Vec::new(),
            tris: Vec::new(),
            face_surface: Vec::new(),
            surfaces: Vec::new(),
            flats: Vec::new(),
            features: Vec::new(),
        }
    }

    /// Registers a surface and returns its index.
    pub fn add_surface(&mut self, s: SurfaceKind) -> u32 {
        self.surfaces.push(s);
        (self.surfaces.len() - 1) as u32
    }

    /// Adds a triangle on the given surface.
    pub fn push_tri(&mut self, t: Tri, surface: u32) {
        self.tris.push(t);
        self.face_surface.push(surface);
    }

    /// Adds a flat polygonal face: pushes its helper triangulation `tris` (all
    /// on `surface`, and which must exactly tile the polygon) and records the
    /// boundary `facet` as a first-class planar facet for the conformal
    /// arrangement, its loops wound about the triangles' normal whichever
    /// way the caller gave them.
    pub fn push_flat(&mut self, facet: PlanarFacet, tris: &[Tri], surface: u32) {
        let mut n = [0.0; 3];
        for t in tris {
            let [a, b, c] = t.v;
            let u: [f64; 3] = std::array::from_fn(|k| b[k] - a[k]);
            let w: [f64; 3] = std::array::from_fn(|k| c[k] - a[k]);
            n[0] += u[1] * w[2] - u[2] * w[1];
            n[1] += u[2] * w[0] - u[0] * w[2];
            n[2] += u[0] * w[1] - u[1] * w[0];
        }
        let facet = facet.wound_about(n);
        let start = self.tris.len();
        for t in tris {
            self.push_tri(*t, surface);
        }
        self.flats.push(FlatFacet {
            facet,
            surface,
            tris: start..self.tris.len(),
        });
    }

    /// Appends another shape (surface and triangle indices are re-based).
    pub fn append(&mut self, other: &Faceted) {
        let surf_base = self.surfaces.len() as u32;
        let tri_base = self.tris.len();
        self.surfaces.extend(other.surfaces.iter().cloned());
        self.tris.extend(other.tris.iter().copied());
        self.face_surface
            .extend(other.face_surface.iter().map(|&s| s + surf_base));
        self.flats.extend(other.flats.iter().map(|fl| FlatFacet {
            facet: fl.facet.clone(),
            surface: fl.surface + surf_base,
            tris: (fl.tris.start + tri_base)..(fl.tris.end + tri_base),
        }));
        self.features.extend(other.features.iter().copied());
    }

    /// The bare triangle soup as a CSG solid operand. Only meaningful for
    /// shapes built as closed, outward-oriented solids.
    /// This shape with `cutter` taken away, by the exact boolean.
    pub fn minus(&self, cutter: &Faceted) -> Result<Faceted, rapidmesh_csg::ArrangeError> {
        self.boolean(cutter, rapidmesh_csg::BoolOp::Difference)
    }

    /// What this shape and `other` have in common, by the exact boolean.
    pub fn common(&self, other: &Faceted) -> Result<Faceted, rapidmesh_csg::ArrangeError> {
        self.boolean(other, rapidmesh_csg::BoolOp::Intersection)
    }

    /// The exact boolean `op` of this shape and `other`. Every face keeps
    /// its carrier: this shape's faces theirs, `other`'s faces theirs,
    /// appended after this shape's surfaces (so a role past the shape's own
    /// names a face that came from `other`). The result is given as single
    /// triangles, with this shape's frame, corners and features.
    pub fn boolean(
        &self,
        other: &Faceted,
        op: rapidmesh_csg::BoolOp,
    ) -> Result<Faceted, rapidmesh_csg::ArrangeError> {
        let cutter = other;
        let out = rapidmesh_csg::boolean(&self.to_solid(), &cutter.to_solid(), op)?;
        let pts: Vec<[f64; 3]> = out
            .vertices
            .iter()
            .map(|p| p.approx().expect("boolean vertices are valid points"))
            .collect();
        let own = self.tris.len();
        let base = self.surfaces.len() as u32;
        let mut f = Faceted {
            frame: self.frame,
            corners: self.corners.clone(),
            tris: Vec::new(),
            face_surface: Vec::new(),
            surfaces: self
                .surfaces
                .iter()
                .chain(&cutter.surfaces)
                .cloned()
                .collect(),
            flats: Vec::new(),
            features: self.features.clone(),
        };
        for (t, &src) in out.triangles.iter().zip(&out.source_facet) {
            let v = t.map(|i| pts[i]);
            let tri = Tri::new(v[0], v[1], v[2]);
            if tri.is_degenerate() {
                continue; // collapsed to a line by the rounding
            }
            let surface = if src < own {
                self.face_surface[src]
            } else {
                base + cutter.face_surface[src - own]
            };
            f.push_tri(tri, surface);
        }
        Ok(f)
    }

    pub fn to_solid(&self) -> Solid {
        Solid {
            tris: self.tris.clone(),
        }
    }

    /// Every point the shape is given by: triangle corners, flat boundary
    /// loops and feature segments.
    pub fn for_each_point(&self, mut f: impl FnMut([f64; 3])) {
        for t in &self.tris {
            t.v.iter().for_each(|&p| f(p));
        }
        for fl in &self.flats {
            fl.facet.outer.iter().for_each(|&p| f(p));
            fl.facet.holes.iter().flatten().for_each(|&p| f(p));
        }
        for s in &self.features {
            s.iter().for_each(|&p| f(p));
        }
    }

    /// The shape with every point moved by `map` (a tiny move: the surface
    /// metadata stays). Triangles whose corners come to coincide are
    /// dropped, flat faces whose loops lose a corner keep the rest of it.
    pub fn with_points(&self, map: impl Fn([f64; 3]) -> [f64; 3]) -> Faceted {
        let mut out = Faceted {
            frame: self.frame,
            corners: self.corners.iter().map(|&p| map(p)).collect(),
            tris: Vec::with_capacity(self.tris.len()),
            face_surface: Vec::with_capacity(self.tris.len()),
            surfaces: self.surfaces.clone(),
            flats: Vec::with_capacity(self.flats.len()),
            features: self.features.iter().map(|s| s.map(&map)).collect(),
        };
        // Old triangle index -> new, to carry the helper ranges of the flats.
        let mut new_index = vec![usize::MAX; self.tris.len()];
        for (i, t) in self.tris.iter().enumerate() {
            let v = t.v.map(&map);
            let tri = Tri::new(v[0], v[1], v[2]);
            if tri.is_degenerate() {
                continue; // collapsed to a line by the map
            }
            new_index[i] = out.tris.len();
            out.tris.push(tri);
            out.face_surface.push(self.face_surface[i]);
        }
        let dedup = |l: &[[f64; 3]]| -> Vec<[f64; 3]> {
            let mut o: Vec<[f64; 3]> = Vec::with_capacity(l.len());
            for &p in l {
                let q = map(p);
                if o.last() != Some(&q) {
                    o.push(q);
                }
            }
            while o.len() > 1 && o.first() == o.last() {
                o.pop();
            }
            o
        };
        for fl in &self.flats {
            let kept: Vec<usize> = fl
                .tris
                .clone()
                .map(|i| new_index[i])
                .filter(|&i| i != usize::MAX)
                .collect();
            let outer = dedup(&fl.facet.outer);
            // A flat whose helpers were dropped is out of order in the new
            // triangle list or degenerate: its triangles stay as curved ones.
            let contiguous = kept.windows(2).all(|w| w[1] == w[0] + 1);
            if kept.len() != fl.tris.len() || !contiguous || outer.len() < 3 {
                continue;
            }
            out.flats.push(FlatFacet {
                facet: PlanarFacet {
                    outer,
                    holes: fl
                        .facet
                        .holes
                        .iter()
                        .map(|h| dedup(h))
                        .filter(|h| h.len() >= 3)
                        .collect(),
                },
                surface: fl.surface,
                tris: kept[0]..kept[kept.len() - 1] + 1,
            });
        }
        out
    }

    /// Rigidly transformed copy (rotation/reflection-free linear part keeps
    /// the surface metadata valid; non-rigid linear parts would invalidate
    /// radii in `surfaces`).
    pub fn transformed(&self, linear: [[f64; 3]; 3], offset: [f64; 3]) -> Faceted {
        let map = |p: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|i| {
                linear[i][0] * p[0] + linear[i][1] * p[1] + linear[i][2] * p[2] + offset[i]
            })
        };
        let map_dir = |d: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|i| linear[i][0] * d[0] + linear[i][1] * d[1] + linear[i][2] * d[2])
        };
        Faceted {
            frame: self.frame.then(linear, offset),
            corners: self.corners.iter().map(|&p| map(p)).collect(),
            tris: self
                .tris
                .iter()
                .map(|t| Tri::new(map(t.v[0]), map(t.v[1]), map(t.v[2])))
                .collect(),
            face_surface: self.face_surface.clone(),
            features: self.features.iter().map(|f| f.map(map)).collect(),
            flats: self
                .flats
                .iter()
                .map(|fl| FlatFacet {
                    facet: fl.facet.map_points(map),
                    surface: fl.surface,
                    tris: fl.tris.clone(),
                })
                .collect(),
            surfaces: self
                .surfaces
                .iter()
                .map(|s| match s {
                    SurfaceKind::Plane => SurfaceKind::Plane,
                    SurfaceKind::Cylinder {
                        center,
                        axis,
                        radius,
                    } => SurfaceKind::Cylinder {
                        center: map(*center),
                        axis: map_dir(*axis),
                        radius: *radius,
                    },
                    SurfaceKind::Sphere { center, radius } => SurfaceKind::Sphere {
                        center: map(*center),
                        radius: *radius,
                    },
                    SurfaceKind::Cone {
                        apex,
                        axis,
                        tan_half_angle,
                    } => SurfaceKind::Cone {
                        apex: map(*apex),
                        axis: map_dir(*axis),
                        tan_half_angle: *tan_half_angle,
                    },
                    SurfaceKind::Torus {
                        center,
                        axis,
                        major_radius,
                        minor_radius,
                    } => SurfaceKind::Torus {
                        center: map(*center),
                        axis: map_dir(*axis),
                        major_radius: *major_radius,
                        minor_radius: *minor_radius,
                    },
                    SurfaceKind::Extruded {
                        profile,
                        base,
                        udir,
                        vdir,
                        axis,
                    } => {
                        // Rigid map: the 2D profile is unchanged; its frame and
                        // origin move with the shape.
                        SurfaceKind::Extruded {
                            profile: profile.clone(),
                            base: map(*base),
                            udir: map_dir(*udir),
                            vdir: map_dir(*vdir),
                            axis: map_dir(*axis),
                        }
                    }
                    SurfaceKind::Tube { path, radius } => SurfaceKind::Tube {
                        path: Arc::new(crate::tube::TubePath::new(
                            path.pts.iter().map(|&q| map(q)).collect(),
                        )),
                        radius: *radius,
                    },
                    SurfaceKind::Revolved {
                        profile,
                        origin,
                        axis,
                        x,
                    } => SurfaceKind::Revolved {
                        profile: profile.clone(),
                        origin: map(*origin),
                        axis: map_dir(*axis),
                        x: map_dir(*x),
                    },
                    SurfaceKind::Nurbs(n) => SurfaceKind::Nurbs(Arc::new(crate::NurbsSurface {
                        ctrl: n.ctrl.iter().map(|&q| map(q)).collect(),
                        ..(**n).clone()
                    })),
                    // The discrete carrier IS its point set: map the points,
                    // rebuild the accelerator (normals re-derive from winding).
                    SurfaceKind::Discrete(d) => SurfaceKind::Discrete(std::sync::Arc::new(
                        crate::discrete::DiscreteSurface::new(
                            d.points.iter().map(|&q| map(q)).collect(),
                            d.tris.clone(),
                        ),
                    )),
                })
                .collect(),
        }
    }

    /// Copy reflected across the plane through `point` with normal `normal`
    /// (Householder). Reflections invert orientation, so every triangle's
    /// winding is flipped to keep solids outward-oriented; radii are
    /// isometry-invariant, so the surface metadata stays valid.
    pub fn mirrored(&self, normal: [f64; 3], point: [f64; 3]) -> Faceted {
        let len2 = normal.iter().map(|x| x * x).sum::<f64>();
        assert!(len2 > 0.0, "mirror normal must be nonzero");
        let n: [f64; 3] = std::array::from_fn(|i| normal[i] / len2.sqrt());
        let mut linear = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let kron = if i == j { 1.0 } else { 0.0 };
                linear[i][j] = kron - 2.0 * n[i] * n[j];
            }
        }
        // Reflect about the plane point: x -> H (x - p) + p.
        let offset: [f64; 3] = std::array::from_fn(|i| {
            point[i] - (linear[i][0] * point[0] + linear[i][1] * point[1] + linear[i][2] * point[2])
        });
        let mut out = self.transformed(linear, offset);
        for t in &mut out.tris {
            t.v.swap(1, 2);
        }
        for fl in &mut out.flats {
            fl.facet = fl.facet.reversed();
        }
        out
    }

    /// Copy scaled per axis about `center`. Uniform scaling keeps the
    /// analytic surface metadata (radii scale along); NON-uniform scaling
    /// turns cylinders/spheres into quadrics this library does not model, so
    /// curved back-references DEGRADE to [`SurfaceKind::Plane`] (each facet
    /// becomes its own exact constraint; fidelity snapping is off for them).
    /// Negative factors with a negative product invert orientation and flip
    /// the winding accordingly.
    pub fn scaled(&self, factors: [f64; 3], center: [f64; 3]) -> Faceted {
        assert!(
            factors.iter().all(|&f| f != 0.0),
            "scale factors must be nonzero"
        );
        let linear = [
            [factors[0], 0.0, 0.0],
            [0.0, factors[1], 0.0],
            [0.0, 0.0, factors[2]],
        ];
        let offset: [f64; 3] = std::array::from_fn(|i| center[i] * (1.0 - factors[i]));
        let uniform = factors[0] == factors[1] && factors[1] == factors[2];
        let mut out = self.transformed(linear, offset);
        if uniform {
            let s = factors[0].abs();
            for kind in &mut out.surfaces {
                match kind {
                    SurfaceKind::Plane => {}
                    // discrete carriers scale with their (already transformed)
                    // point set; nothing else to adjust
                    SurfaceKind::Discrete(_) => {}
                    SurfaceKind::Cylinder { radius, .. } => *radius *= s,
                    SurfaceKind::Sphere { radius, .. } => *radius *= s,
                    SurfaceKind::Cone { .. } => {}
                    SurfaceKind::Torus {
                        major_radius,
                        minor_radius,
                        ..
                    } => {
                        *major_radius *= s;
                        *minor_radius *= s;
                    }
                    // `transformed` already scaled the (meant-to-be-unit) frame
                    // vectors, so the analytic extrusion no longer holds; keep
                    // the facets but drop the back-reference.
                    SurfaceKind::Extruded { .. } => *kind = SurfaceKind::Plane,
                    // uniform scale: the path scaled with the shape, only the
                    // radius follows here
                    SurfaceKind::Tube { radius, .. } => *radius *= s,
                    // the control points scaled with the shape
                    SurfaceKind::Nurbs(_) => {}
                    // `transformed` scaled the frame; the meridian follows
                    SurfaceKind::Revolved {
                        profile, axis, x, ..
                    } => {
                        *profile = Arc::new(profile.scaled(s));
                        *axis = crate::vec3::normalize(*axis);
                        *x = crate::vec3::normalize(*x);
                    }
                }
            }
        } else {
            for kind in &mut out.surfaces {
                // discrete and NURBS carriers survive any linear map (their
                // points were transformed); analytic kinds lose their closed
                // form
                if !matches!(
                    kind,
                    SurfaceKind::Plane | SurfaceKind::Discrete(_) | SurfaceKind::Nurbs(_)
                ) {
                    *kind = SurfaceKind::Plane;
                }
            }
        }
        if factors[0] * factors[1] * factors[2] < 0.0 {
            for t in &mut out.tris {
                t.v.swap(1, 2);
            }
            for fl in &mut out.flats {
                fl.facet = fl.facet.reversed();
            }
        }
        out
    }

    /// Translated copy.
    pub fn translated(&self, offset: [f64; 3]) -> Faceted {
        self.transformed([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], offset)
    }

    /// Copy rotated by `angle` radians around the axis `(origin, dir)`
    /// (Rodrigues formula, right-handed).
    pub fn rotated(&self, origin: [f64; 3], dir: [f64; 3], angle: f64) -> Faceted {
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        assert!(len > 0.0, "rotation axis must be nonzero");
        let u = [dir[0] / len, dir[1] / len, dir[2] / len];
        let (s, c) = angle.sin_cos();
        let mut rot = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let kron = if i == j { 1.0 } else { 0.0 };
                // Levi-Civita term of the cross-product matrix.
                let eps = match (i, j) {
                    (0, 1) => -u[2],
                    (1, 0) => u[2],
                    (0, 2) => u[1],
                    (2, 0) => -u[1],
                    (1, 2) => -u[0],
                    (2, 1) => u[0],
                    _ => 0.0,
                };
                rot[i][j] = c * kron + s * eps + (1.0 - c) * u[i] * u[j];
            }
        }
        // Rotate about the origin point: x -> R (x - o) + o.
        let offset = std::array::from_fn(|i| {
            origin[i] - (rot[i][0] * origin[0] + rot[i][1] * origin[1] + rot[i][2] * origin[2])
        });
        self.transformed(rot, offset)
    }
}

impl Default for Faceted {
    fn default() -> Self {
        Faceted::new()
    }
}
