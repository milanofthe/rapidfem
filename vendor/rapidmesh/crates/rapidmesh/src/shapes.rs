//! The solids and sheets a [`crate::Geometry`] is built from.
//!
//! Every shape is a plain struct with public fields: `new` takes what the
//! shape cannot do without and fills the rest with the defaults, which
//! struct update syntax overrides:
//!
//! ```
//! use rapidmesh::shapes::Cylinder;
//! let c = Cylinder { segments: 48, ..Cylinder::new(0.5, 2.0).at([1.0, 1.0, 0.0]) };
//! assert_eq!(c.segments, 48);
//! ```
//!
//! A shape is only a description; [`crate::Geometry::add_solid`] facets it,
//! with the geometry's target size where the faceting depends on it.

use crate::{Error, Result};
pub use rapidmesh_geom::ProfileEdge;
use rapidmesh_geom::{
    cylinder, cylinder_iso, extrude_polygon, extrude_spline_profile, facet_subdivisions, frustum,
    frustum_iso, helix, icosphere, import_obj_creased, import_stl_creased, loft, mesh_solid,
    naca0012_profile, pipe, revolve, sheet_disk, sheet_nurbs, sheet_polygon, sheet_rect, solid_box,
    torus, validate_closed, wedge, Faceted, NurbsSurface,
};
use std::f64::consts::TAU;
use std::path::PathBuf;

type P3 = [f64; 3];
type P2 = [f64; 2];

/// `v` scaled to unit length; an error for the zero vector.
pub(crate) fn unit(v: P3) -> Result<P3> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n == 0.0 {
        return Err(Error::Invalid("axis must be nonzero".into()));
    }
    Ok(v.map(|c| c / n))
}

fn cross(a: P3, b: P3) -> P3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Rows of a structured barrel with roughly square cells: the height over
/// the chord of `segments` round a circle of `radius`.
fn square_rows(height: f64, radius: f64, segments: usize) -> usize {
    let chord = TAU * radius / segments.max(1) as f64;
    if chord > 0.0 {
        ((height / chord).round_ties_even() as usize).max(1)
    } else {
        1
    }
}

/// Axis-aligned box: `size` along x, y, z from the lower corner `position`.
#[derive(Clone, Debug)]
pub struct Cuboid {
    pub size: P3,
    pub position: P3,
}

impl Cuboid {
    pub fn new(size: P3) -> Cuboid {
        Cuboid {
            size,
            position: [0.0; 3],
        }
    }

    pub fn at(self, position: P3) -> Cuboid {
        Cuboid { position, ..self }
    }
}

/// Cylinder from the base centre `position` along `axis`. The barrel is
/// faceted with `segments` chords and carries the analytic surface. With
/// `uniform` the barrel is a structured grid of `rows` (roughly square cells
/// when `None`) instead of full-height strips.
#[derive(Clone, Debug)]
pub struct Cylinder {
    pub radius: f64,
    pub height: f64,
    pub position: P3,
    pub axis: P3,
    pub segments: usize,
    pub uniform: bool,
    pub rows: Option<usize>,
}

impl Cylinder {
    pub fn new(radius: f64, height: f64) -> Cylinder {
        Cylinder {
            radius,
            height,
            position: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            segments: 24,
            uniform: false,
            rows: None,
        }
    }

    pub fn at(self, position: P3) -> Cylinder {
        Cylinder { position, ..self }
    }

    pub fn along(self, axis: P3) -> Cylinder {
        Cylinder { axis, ..self }
    }
}

/// Sphere centred at `position` (faceted geodesically, analytic surface).
/// The facet density follows the target size; `segments` is a floor.
#[derive(Clone, Debug)]
pub struct Sphere {
    pub radius: f64,
    pub position: P3,
    pub segments: usize,
}

impl Sphere {
    pub fn new(radius: f64) -> Sphere {
        Sphere {
            radius,
            position: [0.0; 3],
            segments: 24,
        }
    }

    pub fn at(self, position: P3) -> Sphere {
        Sphere { position, ..self }
    }
}

/// Geodesic sphere with a fixed facet level: `20 * 4^subdivisions` faces.
#[derive(Clone, Debug)]
pub struct Icosphere {
    pub radius: f64,
    pub position: P3,
    pub subdivisions: usize,
}

impl Icosphere {
    pub fn new(radius: f64) -> Icosphere {
        Icosphere {
            radius,
            position: [0.0; 3],
            subdivisions: 3,
        }
    }

    pub fn at(self, position: P3) -> Icosphere {
        Icosphere { position, ..self }
    }
}

/// A NACA 0012 airfoil (chord along +x, leading edge at `position`)
/// extruded along `span_axis` by `span`: one analytic spline skin, a flat
/// blunt trailing edge and two caps.
#[derive(Clone, Debug)]
pub struct Naca0012 {
    pub chord: f64,
    pub span: f64,
    pub position: P3,
    pub span_axis: P3,
    pub n_per_side: usize,
    pub n_seg: usize,
}

impl Naca0012 {
    pub fn new(chord: f64, span: f64) -> Naca0012 {
        Naca0012 {
            chord,
            span,
            position: [0.0; 3],
            span_axis: [0.0, 0.0, 1.0],
            n_per_side: 40,
            n_seg: 120,
        }
    }

    pub fn at(self, position: P3) -> Naca0012 {
        Naca0012 { position, ..self }
    }
}

/// Conical frustum: radius `r1` at `position`, `r2` (0 for a full cone) at
/// `position + height * axis`. `uniform` and `rows` as for [`Cylinder`].
#[derive(Clone, Debug)]
pub struct Cone {
    pub r1: f64,
    pub r2: f64,
    pub height: f64,
    pub position: P3,
    pub axis: P3,
    pub segments: usize,
    pub uniform: bool,
    pub rows: Option<usize>,
}

impl Cone {
    pub fn new(r1: f64, r2: f64, height: f64) -> Cone {
        Cone {
            r1,
            r2,
            height,
            position: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            segments: 24,
            uniform: false,
            rows: None,
        }
    }

    pub fn at(self, position: P3) -> Cone {
        Cone { position, ..self }
    }

    pub fn along(self, axis: P3) -> Cone {
        Cone { axis, ..self }
    }
}

/// Right prism: the polygon `points` (xy plane, offset by `position`) with
/// `holes`, extruded by `height` along z.
#[derive(Clone, Debug)]
pub struct Prism {
    pub points: Vec<P2>,
    pub holes: Vec<Vec<P2>>,
    pub height: f64,
    pub position: P3,
}

impl Prism {
    pub fn new(points: Vec<P2>, height: f64) -> Prism {
        Prism {
            points,
            holes: Vec::new(),
            height,
            position: [0.0; 3],
        }
    }

    pub fn at(self, position: P3) -> Prism {
        Prism { position, ..self }
    }
}

/// Torus centred at `position`, its plane normal to `axis`.
#[derive(Clone, Debug)]
pub struct Torus {
    pub major_radius: f64,
    pub minor_radius: f64,
    pub position: P3,
    pub axis: P3,
    pub segments: usize,
    pub tube_segments: usize,
}

impl Torus {
    pub fn new(major_radius: f64, minor_radius: f64) -> Torus {
        Torus {
            major_radius,
            minor_radius,
            position: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            segments: 32,
            tube_segments: 16,
        }
    }

    pub fn at(self, position: P3) -> Torus {
        Torus { position, ..self }
    }
}

/// A `dx x dy x dz` box whose top edge is shortened to `top_x` along x (0 a
/// triangular prism); the taper runs in the xz plane.
#[derive(Clone, Debug)]
pub struct Wedge {
    pub size: P3,
    pub position: P3,
    pub top_x: f64,
}

impl Wedge {
    pub fn new(size: P3) -> Wedge {
        Wedge {
            size,
            position: [0.0; 3],
            top_x: 0.0,
        }
    }

    pub fn at(self, position: P3) -> Wedge {
        Wedge { position, ..self }
    }
}

/// A round tube of `radius` swept along the open polyline `path`.
#[derive(Clone, Debug)]
pub struct Sweep {
    pub path: Vec<P3>,
    pub radius: f64,
    pub segments: usize,
}

impl Sweep {
    pub fn new(path: Vec<P3>, radius: f64) -> Sweep {
        Sweep {
            path,
            radius,
            segments: 16,
        }
    }
}

/// Helical coil around +z through `position`: helix `radius`, `pitch` per
/// turn, round wire of `wire_radius`.
#[derive(Clone, Debug)]
pub struct Helix {
    pub radius: f64,
    pub pitch: f64,
    pub turns: f64,
    pub wire_radius: f64,
    pub position: P3,
    pub points_per_turn: usize,
    pub segments: usize,
}

impl Helix {
    pub fn new(radius: f64, pitch: f64, turns: f64, wire_radius: f64) -> Helix {
        Helix {
            radius,
            pitch,
            turns,
            wire_radius,
            position: [0.0; 3],
            points_per_turn: 24,
            segments: 12,
        }
    }

    pub fn at(self, position: P3) -> Helix {
        Helix { position, ..self }
    }
}

/// Ruled loft between two planar profiles with the same vertex count,
/// corresponded by index; both star-shaped about their centroid.
#[derive(Clone, Debug)]
pub struct Loft {
    pub profile_a: Vec<P3>,
    pub profile_b: Vec<P3>,
}

/// Solid of revolution: the closed profile `points` (`(r, z)`, `r >= 0`,
/// edge `i` from point `i` to the next) turned about `axis` through
/// `position` by `angle` degrees. Every edge is a surface with its exact
/// carrier (plane, cylinder, cone, sphere, torus or a revolved spline); a
/// part turn adds a start and an end cap. The facet count follows the
/// target size; `segments` is a floor around a full turn.
#[derive(Clone, Debug)]
pub struct Revolve {
    pub points: Vec<P2>,
    pub edges: Vec<ProfileEdge>,
    pub position: P3,
    pub axis: P3,
    pub angle: f64,
    pub segments: usize,
}

impl Revolve {
    /// A full turn about z of the polygon `points`.
    pub fn new(points: Vec<P2>) -> Revolve {
        Revolve {
            edges: vec![ProfileEdge::Line; points.len()],
            points,
            position: [0.0; 3],
            axis: [0.0, 0.0, 1.0],
            angle: 360.0,
            segments: 24,
        }
    }
}

/// A closed, non-self-intersecting triangle soup taken as it is (no
/// analytic surface); the winding is made outward.
#[derive(Clone, Debug)]
pub struct Triangles {
    pub verts: Vec<P3>,
    pub tris: Vec<[u32; 3]>,
}

/// Which axis of a file points up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Up {
    Y,
    Z,
}

impl std::str::FromStr for Up {
    type Err = Error;
    fn from_str(s: &str) -> Result<Up> {
        match s.to_ascii_lowercase().as_str() {
            "y" => Ok(Up::Y),
            "z" => Ok(Up::Z),
            other => Err(Error::Invalid(format!(
                "unknown up axis {other:?} (expected \"y\" or \"z\")"
            ))),
        }
    }
}

/// A solid from an STL or OBJ file, split into smooth surfaces at creases
/// sharper than `crease_deg` and remeshed. The file must be a closed,
/// consistently oriented 2-manifold; a y-up file is turned z-up.
#[derive(Clone, Debug)]
pub struct Import {
    pub path: PathBuf,
    pub crease_deg: f64,
    pub up: Up,
}

impl Import {
    pub fn new(path: impl Into<PathBuf>) -> Import {
        Import {
            path: path.into(),
            crease_deg: 40.0,
            up: Up::Z,
        }
    }
}

/// Any solid.
#[derive(Clone, Debug)]
pub enum Shape {
    Cuboid(Cuboid),
    Cylinder(Cylinder),
    Sphere(Sphere),
    Icosphere(Icosphere),
    Naca0012(Naca0012),
    Cone(Cone),
    Prism(Prism),
    Torus(Torus),
    Wedge(Wedge),
    Sweep(Sweep),
    Helix(Helix),
    Loft(Loft),
    Revolve(Revolve),
    Triangles(Triangles),
    Import(Import),
}

macro_rules! into_shape {
    ($($t:ident),*) => {$(
        impl From<$t> for Shape {
            fn from(s: $t) -> Shape {
                Shape::$t(s)
            }
        }
    )*};
}

into_shape!(
    Cuboid, Cylinder, Sphere, Icosphere, Naca0012, Cone, Prism, Torus, Wedge, Sweep, Helix, Loft,
    Revolve, Triangles, Import
);

/// The faces of a box, in their order.
pub const BOX_ROLES: [&str; 6] = ["-z", "+z", "-y", "+y", "-x", "+x"];
/// The faces of a cylinder or cone, in their order.
pub const AXIAL_ROLES: [&str; 3] = ["side", "top", "bottom"];
/// The first faces of a prism or an extruded sheet, the walls after them.
pub const CAP_ROLES: [&str; 2] = ["bottom", "top"];

impl Shape {
    /// The names of the solid's faces by role: a box's sides, a cylinder's
    /// or cone's side and ends, a prism's ends, a revolve's profile edges
    /// (`edge0`, `edge1`, ...) and on a part turn its `start` and `end`.
    /// Other shapes name none (their faces go by index).
    pub fn role_names(&self) -> Vec<String> {
        let names = |n: &[&str]| n.iter().map(|s| s.to_string()).collect();
        match self {
            Shape::Cuboid(_) => names(&BOX_ROLES),
            Shape::Cylinder(_) | Shape::Cone(_) => names(&AXIAL_ROLES),
            Shape::Prism(_) => names(&CAP_ROLES),
            Shape::Revolve(r) => {
                let mut out: Vec<String> =
                    (0..r.points.len()).map(|i| format!("edge{i}")).collect();
                if r.angle < 360.0 {
                    out.extend(["start".to_string(), "end".to_string()]);
                }
                out
            }
            _ => Vec::new(),
        }
    }

    /// The faceted solid; `maxh` is the target size at the solid, which
    /// sets the facet density of a sphere.
    pub(crate) fn faceted(&self, maxh: Option<f64>) -> Result<Faceted> {
        let scaled = |axis: P3, h: f64| unit(axis).map(|a| a.map(|c| c * h));
        Ok(match self {
            Shape::Cuboid(b) => {
                let (p, s) = (b.position, b.size);
                solid_box(p, [p[0] + s[0], p[1] + s[1], p[2] + s[2]])
            }
            Shape::Cylinder(c) => {
                let ax = scaled(c.axis, c.height)?;
                if c.uniform {
                    let rows = c
                        .rows
                        .unwrap_or_else(|| square_rows(c.height, c.radius, c.segments));
                    cylinder_iso(c.position, ax, c.radius, c.segments, rows)
                } else {
                    cylinder(c.position, ax, c.radius, c.segments)
                }
            }
            // Faceted geodesically: isotropic and pole-free, the level from
            // the target size (`segments` only a floor).
            Shape::Sphere(s) => {
                let level = facet_subdivisions(s.radius, maxh, 1e-2, s.segments);
                icosphere(s.position, s.radius, level)
            }
            Shape::Icosphere(s) => icosphere(s.position, s.radius, s.subdivisions),
            Shape::Naca0012(a) => {
                let h = scaled(a.span_axis, a.span)?;
                let profile = naca0012_profile(a.chord, a.n_per_side);
                extrude_spline_profile(
                    profile,
                    a.n_seg,
                    a.position,
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    h,
                )
            }
            Shape::Cone(c) => {
                let ax = scaled(c.axis, c.height)?;
                if c.uniform {
                    let rows = c
                        .rows
                        .unwrap_or_else(|| square_rows(c.height, 0.5 * (c.r1 + c.r2), c.segments));
                    frustum_iso(c.position, ax, c.r1, c.r2, c.segments, rows)
                } else {
                    frustum(c.position, ax, c.r1, c.r2, c.segments)
                }
            }
            Shape::Prism(p) => extrude_polygon(
                &p.points,
                &p.holes,
                p.position,
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, p.height],
            ),
            Shape::Torus(t) => torus(
                t.position,
                unit(t.axis)?,
                t.major_radius,
                t.minor_radius,
                t.segments,
                t.tube_segments,
            ),
            Shape::Wedge(w) => wedge(w.position, w.size[0], w.size[1], w.size[2], w.top_x),
            Shape::Sweep(s) => pipe(&s.path, s.radius, s.segments),
            Shape::Helix(h) => helix(
                h.position,
                h.radius,
                h.pitch,
                h.turns,
                h.wire_radius,
                h.points_per_turn,
                h.segments,
            ),
            Shape::Loft(l) => loft(&l.profile_a, &l.profile_b),
            Shape::Revolve(r) => revolve(
                &r.points,
                &r.edges,
                r.position,
                r.axis,
                r.angle.to_radians(),
                maxh,
                1e-2,
                r.segments,
            )
            .map_err(|e| Error::Invalid(format!("revolve: {e}")))?,
            Shape::Triangles(t) => mesh_solid(&t.verts, &t.tris),
            Shape::Import(i) => {
                let p = &i.path;
                let name = p.display();
                let stl = p
                    .extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.eq_ignore_ascii_case("stl"));
                let f = if stl {
                    import_stl_creased(p, i.crease_deg)
                } else {
                    import_obj_creased(p, i.crease_deg)
                }
                .map_err(|e| Error::Invalid(format!("{name}: {e}")))?;
                let f = match i.up {
                    Up::Z => f,
                    // y up to z up: +90 degrees about x, (x, y, z) -> (x, -z, y).
                    Up::Y => f.transformed(
                        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                        [0.0; 3],
                    ),
                };
                validate_closed(&f).map_err(|e| Error::Invalid(format!("{name}: {e}")))?;
                f
            }
        })
    }
}

/// A zero-thickness face embedded in the volume mesh (a PEC trace, a port).
#[derive(Clone, Debug)]
pub enum Sheet {
    /// The parallelogram from `corner` spanned by `u` and `v`.
    Rect { corner: P3, u: P3, v: P3 },
    /// A disc of `radius` centred at `center`, normal to `axis`.
    Disc {
        radius: f64,
        center: P3,
        axis: P3,
        segments: usize,
    },
    /// The polygon `points` with `holes` in the xy plane at `position`.
    Polygon {
        points: Vec<P2>,
        holes: Vec<Vec<P2>>,
        position: P3,
    },
    /// A NURBS patch, tessellated `segments` per parameter direction and
    /// carried by the exact surface.
    Nurbs {
        surface: NurbsSurface,
        segments: [usize; 2],
    },
}

impl Sheet {
    /// A rectangle in an xy plane: `width` along x, `height` along y.
    pub fn xy(width: f64, height: f64, position: P3) -> Sheet {
        Sheet::plate(position, [width, 0.0, 0.0], [0.0, height, 0.0])
    }

    /// A rectangle in an xz plane: `width` along x, `height` along z.
    pub fn xz(width: f64, height: f64, position: P3) -> Sheet {
        Sheet::plate(position, [width, 0.0, 0.0], [0.0, 0.0, height])
    }

    /// A rectangle in a yz plane: `width` along y, `height` along z.
    pub fn yz(width: f64, height: f64, position: P3) -> Sheet {
        Sheet::plate(position, [0.0, width, 0.0], [0.0, 0.0, height])
    }

    /// The parallelogram from `corner` spanned by `u` and `v`.
    pub fn plate(corner: P3, u: P3, v: P3) -> Sheet {
        Sheet::Rect { corner, u, v }
    }

    /// A disc of `radius` at `center`, normal to `axis`, 24 segments.
    pub fn disc(radius: f64, center: P3, axis: P3) -> Sheet {
        Sheet::Disc {
            radius,
            center,
            axis,
            segments: 24,
        }
    }

    /// A polygon in the xy plane at `position`.
    pub fn polygon(points: Vec<P2>, position: P3) -> Sheet {
        Sheet::Polygon {
            points,
            holes: Vec::new(),
            position,
        }
    }

    /// A NURBS patch over the control net `ctrl` (`ctrl[i][j]`, `i` along
    /// u), with `degree` per direction, positive `weights` (all 1 when none)
    /// and clamped uniform knots when none are given. The tessellation takes
    /// four segments per knot span, at least 8 and at most 64 per direction.
    pub fn nurbs(
        ctrl: Vec<Vec<P3>>,
        degree: [usize; 2],
        weights: Option<Vec<Vec<f64>>>,
        knots: Option<[Vec<f64>; 2]>,
    ) -> Result<Sheet> {
        let nu = ctrl.len();
        let nv = ctrl.first().map_or(0, |r| r.len());
        if nu == 0 || nv == 0 || ctrl.iter().any(|r| r.len() != nv) {
            return Err(Error::Invalid(
                "a NURBS control net must be a non-empty rectangular grid".into(),
            ));
        }
        let weights = match weights {
            Some(w) if w.len() != nu || w.iter().any(|r| r.len() != nv) => {
                return Err(Error::Invalid(
                    "NURBS weights must match the control net".into(),
                ))
            }
            Some(w) => w.concat(),
            None => vec![1.0; nu * nv],
        };
        if degree[0] >= nu || degree[1] >= nv {
            return Err(Error::Invalid(format!(
                "a {nu} x {nv} control net allows degrees below ({nu}, {nv}), not {degree:?}"
            )));
        }
        let knots = knots.unwrap_or_else(|| {
            [
                NurbsSurface::clamped_knots(nu, degree[0]),
                NurbsSurface::clamped_knots(nv, degree[1]),
            ]
        });
        let segments = [0, 1].map(|d| {
            let (k, p, n) = (&knots[d], degree[d], [nu, nv][d]);
            let spans = (p..n).filter(|&s| k[s] < k[s + 1]).count();
            (4 * spans).clamp(8, 64)
        });
        let surface = NurbsSurface::try_new(degree, knots, [nu, nv], ctrl.concat(), weights)
            .map_err(|e| Error::Invalid(format!("NURBS patch: {e}")))?;
        Ok(Sheet::Nurbs { surface, segments })
    }

    pub(crate) fn faceted(&self) -> Result<Faceted> {
        Ok(match self {
            Sheet::Rect { corner, u, v } => sheet_rect(*corner, *u, *v),
            Sheet::Disc {
                radius,
                center,
                axis,
                segments,
            } => {
                let a = unit(*axis)?;
                let pick = if a[0].abs() < 0.9 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let e1 = unit(cross(a, pick))?;
                let e2 = cross(a, e1);
                sheet_disk(
                    *center,
                    e1.map(|c| radius * c),
                    e2.map(|c| radius * c),
                    *segments,
                )
            }
            Sheet::Polygon {
                points,
                holes,
                position,
            } => sheet_polygon(points, holes, *position, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            Sheet::Nurbs { surface, segments } => sheet_nurbs(surface, *segments),
        })
    }
}
