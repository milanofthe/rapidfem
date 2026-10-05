//! A STEP file as typed entities: its solids, their faces, edges and
//! vertices, and the geometry under them. Decoded once from the exchange
//! structure; whatever reads the model after that sees Rust types, and an
//! entity that does not decode is named with the reason.

use crate::geometry::{cone, extruded, placement, revolved, Reparam};
use crate::part21::{Exchange, Record, Value};
use rapidmesh_exact::vector::Frame;
use rapidmesh_exact::vector::V3;
use rapidmesh_exact::vector::{bbox, normalize, scale};
use rapidmesh_exact::vector::{mul, mul_vec, sub as vsub, transpose, Affine};
use rapidmesh_geom::{Curve, NurbsCurve, NurbsSurface, Surface};
use rustc_hash::FxHashMap;
use std::sync::Arc;

/// An entity that does not decode.
#[derive(Clone, Debug, PartialEq)]
pub struct StepError {
    pub id: u32,
    pub message: String,
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}: {}", self.id, self.message)
    }
}

impl std::error::Error for StepError {}

type R<T> = Result<T, StepError>;

fn fail<T>(id: u32, message: impl Into<String>) -> R<T> {
    Err(StepError {
        id,
        message: message.into(),
    })
}

/// An edge: from vertex `ends[0]` to `ends[1]` along `curve`, with or
/// against its parameter; and its curve in the parameters of each face's
/// surface where the file gives one.
#[derive(Clone, Debug)]
pub struct Edge {
    pub ends: [usize; 2],
    pub curve: usize,
    pub forward: bool,
    /// (surface, its parameter curve in the surface's parameters), the
    /// parameter the curve's own.
    pub pcurves: Vec<(usize, Curve<2>)>,
}

/// A bound of a face: its edges in turn, each along itself or against, or
/// a single vertex (a pole).
#[derive(Clone, Debug)]
pub enum Bound {
    Edges(Vec<(usize, bool)>),
    Vertex(usize),
}

#[derive(Clone, Debug)]
pub struct Face {
    /// The id in the file.
    pub id: u32,
    pub surface: usize,
    /// Whether the face's normal is its surface's.
    pub same_sense: bool,
    pub bounds: Vec<Bound>,
}

/// The map from the axes `from` onto the axes `to`: a point with
/// coordinates `a, b, c` in `from` goes to the point with the same in `to`.
fn between(from: &Frame, to: &Frame) -> Affine {
    let (f, t) = ([from.x, from.y, from.z], [to.x, to.y, to.z]);
    let linear = mul(transpose(t), f);
    Affine {
        linear,
        offset: vsub(to.o, mul_vec(linear, from.o)),
    }
}

#[derive(Clone, Debug)]
pub struct Solid {
    /// Its product's name where the file gives one, else the solid's.
    pub name: String,
    pub faces: Vec<usize>,
    /// Where the assembly places it.
    pub placement: Affine,
}

/// The typed model of a file.
#[derive(Debug, Default)]
pub struct Model {
    pub metres_per_unit: f64,
    pub vertices: Vec<V3>,
    pub curves: Vec<Curve<3>>,
    pub surfaces: Vec<Surface>,
    pub edges: Vec<Edge>,
    pub faces: Vec<Face>,
    pub solids: Vec<Solid>,
}

/// A record with typed access to its parameters.
struct Rec<'a> {
    id: u32,
    r: &'a Record,
}

impl<'a> Rec<'a> {
    fn get(&self, i: usize) -> R<&'a Value> {
        match self.r.args.get(i) {
            Some(v) => Ok(v),
            None => fail(self.id, format!("{} has no parameter {i}", self.r.name)),
        }
    }

    fn refr(&self, i: usize) -> R<u32> {
        match self.get(i)?.as_ref() {
            Some(r) => Ok(r),
            None => fail(
                self.id,
                format!("parameter {i} of {} is no reference", self.r.name),
            ),
        }
    }

    fn num(&self, i: usize) -> R<f64> {
        match self.get(i)?.as_f64() {
            Some(x) => Ok(x),
            None => fail(
                self.id,
                format!("parameter {i} of {} is no number", self.r.name),
            ),
        }
    }

    fn int(&self, i: usize) -> R<i64> {
        match self.get(i)?.as_int() {
            Some(x) => Ok(x),
            None => fail(
                self.id,
                format!("parameter {i} of {} is no integer", self.r.name),
            ),
        }
    }

    /// `.T.` unless it says `.F.`
    fn flag(&self, i: usize) -> bool {
        self.r.args.get(i).and_then(Value::as_bool).unwrap_or(true)
    }

    fn list(&self, i: usize) -> R<&'a [Value]> {
        match self.get(i)?.as_list() {
            Some(l) => Ok(l),
            None => fail(
                self.id,
                format!("parameter {i} of {} is no list", self.r.name),
            ),
        }
    }

    fn refs(&self, i: usize) -> R<Vec<u32>> {
        Ok(self.list(i)?.iter().filter_map(Value::as_ref).collect())
    }

    fn nums(&self, i: usize) -> R<Vec<f64>> {
        self.list(i)?
            .iter()
            .map(|v| match v.as_f64() {
                Some(x) => Ok(x),
                None => fail(
                    self.id,
                    format!("parameter {i} of {} holds no number", self.r.name),
                ),
            })
            .collect()
    }

    fn name(&self) -> String {
        self.r
            .args
            .first()
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    }
}

struct Decoder<'a> {
    x: &'a Exchange,
    m: Model,
    /// The file's unit of plane angles in radians.
    radians_per_unit: f64,
    vertex_of: FxHashMap<u32, usize>,
    curve_of: FxHashMap<u32, usize>,
    surface_of: FxHashMap<u32, usize>,
    edge_of: FxHashMap<u32, usize>,
    face_of: FxHashMap<u32, usize>,
    /// The surfaces whose normal runs against the file's (swept ones read
    /// as a torus, say): the faces on them turn.
    flipped: rustc_hash::FxHashSet<usize>,
    /// The surfaces whose parameters are not the file's (a cone), with how
    /// the file's map to them: their parameter curves are carried over.
    reparam: FxHashMap<usize, Reparam>,
}

impl<'a> Decoder<'a> {
    fn new(x: &'a Exchange) -> Decoder<'a> {
        let [metres, radians] = units(x);
        Decoder {
            x,
            m: Model {
                metres_per_unit: metres,
                ..Model::default()
            },
            radians_per_unit: radians,
            vertex_of: FxHashMap::default(),
            curve_of: FxHashMap::default(),
            surface_of: FxHashMap::default(),
            edge_of: FxHashMap::default(),
            face_of: FxHashMap::default(),
            flipped: Default::default(),
            reparam: FxHashMap::default(),
        }
    }

    /// The record `name` of instance `id`.
    fn rec(&self, id: u32, name: &str) -> R<Rec<'a>> {
        match self.x.record(id, name) {
            Some(r) => Ok(Rec { id, r }),
            None => {
                let is = self.x.instances.get(&id).map(|rs| {
                    rs.iter()
                        .map(|r| r.name.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                });
                match is {
                    Some(is) => fail(id, format!("{is} where {name} was expected")),
                    None => fail(id, "no such instance"),
                }
            }
        }
    }

    /// The simple record of `id`, whatever its name.
    fn any(&self, id: u32) -> R<Rec<'a>> {
        match self.x.instances.get(&id).map(|rs| rs.as_slice()) {
            Some([r]) => Ok(Rec { id, r }),
            Some(_) => fail(id, "a complex instance where a simple one was expected"),
            None => fail(id, "no such instance"),
        }
    }

    fn coords<const N: usize>(&self, id: u32, name: &str) -> R<[f64; N]> {
        let c = self.rec(id, name)?.nums(1)?;
        Ok(std::array::from_fn(|k| c.get(k).copied().unwrap_or(0.0)))
    }

    fn point(&self, id: u32) -> R<V3> {
        self.coords(id, "CARTESIAN_POINT")
    }

    fn direction(&self, id: u32) -> R<V3> {
        self.coords(id, "DIRECTION")
    }

    fn placement(&self, id: u32) -> R<Frame> {
        let r = self.rec(id, "AXIS2_PLACEMENT_3D")?;
        let o = self.point(r.refr(1)?)?;
        let z = match r.get(2)?.as_ref() {
            Some(d) => self.direction(d)?,
            None => [0.0, 0.0, 1.0],
        };
        let x = match r.get(3)?.as_ref() {
            Some(d) => Some(self.direction(d)?),
            None => None,
        };
        placement(o, z, x).map_or_else(|| fail(id, "a placement along no direction"), Ok)
    }

    /// B-spline data of a curve: degree, control point ids, knots, weights.
    fn spline_parts(&self, id: u32) -> R<(usize, Vec<u32>, Vec<f64>, Option<Vec<f64>>)> {
        let (degree, ctrl, knots) = match self.x.kind(id) {
            Some("B_SPLINE_CURVE_WITH_KNOTS") => {
                // name, degree, points, form, closed, self-intersect,
                // multiplicities, knots, spec
                let r = self.rec(id, "B_SPLINE_CURVE_WITH_KNOTS")?;
                (r.int(1)?, r.refs(2)?, expand(&r, 6, 7)?)
            }
            _ => {
                let b = self.rec(id, "B_SPLINE_CURVE")?;
                let k = self.rec(id, "B_SPLINE_CURVE_WITH_KNOTS")?;
                (b.int(0)?, b.refs(1)?, expand(&k, 0, 1)?)
            }
        };
        let weights = match self.x.record(id, "RATIONAL_B_SPLINE_CURVE") {
            Some(r) => Some(Rec { id, r }.nums(0)?),
            None => None,
        };
        let degree = degree.max(1) as usize;
        if degree > rapidmesh_geom::nurbs::MAX_DEGREE {
            return fail(id, format!("degree {degree} above the largest supported"));
        }
        if weights
            .as_ref()
            .is_some_and(|w| w.len() != ctrl.len() || w.iter().any(|&w| !(w > 0.0)))
        {
            return fail(id, "weights that do not fit the points".to_string());
        }
        if knots.len() != ctrl.len() + degree + 1 {
            return fail(
                id,
                format!(
                    "{} knots for {} points of degree {degree}",
                    knots.len(),
                    ctrl.len()
                ),
            );
        }
        Ok((degree, ctrl, knots, weights))
    }

    /// The 3D curve of an edge (through SURFACE_CURVE, SEAM_CURVE and
    /// TRIMMED_CURVE to their basis).
    fn curve3(&self, id: u32) -> R<Curve<3>> {
        if self.x.record(id, "B_SPLINE_CURVE").is_some()
            || self.x.kind(id) == Some("B_SPLINE_CURVE_WITH_KNOTS")
        {
            let (degree, ids, knots, weights) = self.spline_parts(id)?;
            let ctrl: Vec<V3> = ids.iter().map(|&p| self.point(p)).collect::<R<_>>()?;
            let weights = weights.unwrap_or_else(|| vec![1.0; ctrl.len()]);
            return Ok(Curve::Nurbs(Arc::new(NurbsCurve {
                degree,
                knots,
                ctrl,
                weights,
            })));
        }
        let r = self.any(id)?;
        // A conic by its centre and the vectors along its placement's axes.
        let conic = |a: f64, b: f64| -> R<(V3, V3, V3)> {
            let f = self.placement(r.refr(1)?)?;
            Ok((f.o, scale(f.x, a), scale(f.y, b)))
        };
        match r.r.name.as_str() {
            "SURFACE_CURVE" | "SEAM_CURVE" | "INTERSECTION_CURVE" | "TRIMMED_CURVE" => {
                self.curve3(r.refr(1)?)
            }
            "LINE" => {
                let p = self.point(r.refr(1)?)?;
                let v = self.rec(r.refr(2)?, "VECTOR")?;
                let d = self.direction(v.refr(1)?)?;
                Ok(Curve::Line {
                    p,
                    d: scale(normalize(d), v.num(2)?),
                })
            }
            "CIRCLE" => {
                let (c, p, q) = conic(r.num(2)?, r.num(2)?)?;
                Ok(Curve::Ellipse { c, p, q })
            }
            "ELLIPSE" => {
                let (c, p, q) = conic(r.num(2)?, r.num(3)?)?;
                Ok(Curve::Ellipse { c, p, q })
            }
            "HYPERBOLA" => {
                let (c, p, q) = conic(r.num(2)?, r.num(3)?)?;
                Ok(Curve::Hyperbola { c, p, q })
            }
            "PARABOLA" => {
                let focal = r.num(2)?;
                let (c, p, q) = conic(focal, 2.0 * focal)?;
                Ok(Curve::Parabola { c, p, q })
            }
            other => fail(id, format!("the curve {other} is not read")),
        }
    }

    /// A curve in a surface's parameters.
    fn curve2(&self, id: u32) -> R<Curve<2>> {
        if self.x.record(id, "B_SPLINE_CURVE").is_some()
            || self.x.kind(id) == Some("B_SPLINE_CURVE_WITH_KNOTS")
        {
            let (degree, ids, knots, weights) = self.spline_parts(id)?;
            let ctrl: Vec<[f64; 2]> = ids
                .iter()
                .map(|&p| self.coords(p, "CARTESIAN_POINT"))
                .collect::<R<_>>()?;
            let weights = weights.unwrap_or_else(|| vec![1.0; ctrl.len()]);
            return Ok(Curve::Nurbs(Arc::new(NurbsCurve {
                degree,
                knots,
                ctrl,
                weights,
            })));
        }
        let r = self.any(id)?;
        // A conic by its centre and the vectors along its placement's axes
        // (`y` the `x` turned a quarter).
        let conic = |a: f64, b: f64| -> R<([f64; 2], [f64; 2], [f64; 2])> {
            let pl = self.rec(r.refr(1)?, "AXIS2_PLACEMENT_2D")?;
            let o: [f64; 2] = self.coords(pl.refr(1)?, "CARTESIAN_POINT")?;
            let x: [f64; 2] = match pl.get(2)?.as_ref() {
                Some(d) => normalize(self.coords(d, "DIRECTION")?),
                None => [1.0, 0.0],
            };
            Ok((o, scale(x, a), scale([-x[1], x[0]], b)))
        };
        match r.r.name.as_str() {
            "TRIMMED_CURVE" => self.curve2(r.refr(1)?),
            "LINE" => {
                let p: [f64; 2] = self.coords(r.refr(1)?, "CARTESIAN_POINT")?;
                let v = self.rec(r.refr(2)?, "VECTOR")?;
                let d: [f64; 2] = self.coords(v.refr(1)?, "DIRECTION")?;
                Ok(Curve::Line {
                    p,
                    d: scale(normalize(d), v.num(2)?),
                })
            }
            "CIRCLE" => {
                let (c, p, q) = conic(r.num(2)?, r.num(2)?)?;
                Ok(Curve::Ellipse { c, p, q })
            }
            "ELLIPSE" => {
                let (c, p, q) = conic(r.num(2)?, r.num(3)?)?;
                Ok(Curve::Ellipse { c, p, q })
            }
            other => fail(id, format!("the parameter curve {other} is not read")),
        }
    }

    fn surface(&mut self, id: u32) -> R<usize> {
        if let Some(&s) = self.surface_of.get(&id) {
            return Ok(s);
        }
        let s = if self.x.record(id, "B_SPLINE_SURFACE").is_some()
            || self.x.kind(id) == Some("B_SPLINE_SURFACE_WITH_KNOTS")
        {
            self.spline_surface(id)?
        } else if let Some(name @ ("SURFACE_OF_REVOLUTION" | "SURFACE_OF_LINEAR_EXTRUSION")) =
            self.x.kind(id)
        {
            let r = self.any(id)?;
            let profile = self.curve3(r.refr(1)?)?;
            let swept = if name == "SURFACE_OF_REVOLUTION" {
                let a = self.rec(r.refr(2)?, "AXIS1_PLACEMENT")?;
                let o = self.point(a.refr(1)?)?;
                let axis = match a.get(2)?.as_ref() {
                    Some(d) => self.direction(d)?,
                    None => [0.0, 0.0, 1.0],
                };
                revolved(&profile, o, axis)
            } else {
                let v = self.rec(r.refr(2)?, "VECTOR")?;
                extruded(
                    &profile,
                    scale(normalize(self.direction(v.refr(1)?)?), v.num(2)?),
                )
            };
            let Some(swept) = swept else {
                return fail(id, format!("{name} of this profile is not read"));
            };
            if swept.flipped {
                self.flipped.insert(self.m.surfaces.len());
            }
            swept.surface
        } else {
            let r = self.any(id)?;
            let frame = self.placement(r.refr(1)?)?;
            match r.r.name.as_str() {
                "PLANE" => Surface::Plane(frame),
                "CYLINDRICAL_SURFACE" => Surface::Cylinder {
                    frame,
                    radius: r.num(2)?,
                },
                "CONICAL_SURFACE" => {
                    let (s, map) = cone(frame, r.num(2)?, r.num(3)? * self.radians_per_unit);
                    self.reparam.insert(self.m.surfaces.len(), map);
                    s
                }
                "SPHERICAL_SURFACE" => Surface::Sphere {
                    frame,
                    radius: r.num(2)?,
                },
                "TOROIDAL_SURFACE" => Surface::Torus {
                    frame,
                    major: r.num(2)?,
                    minor: r.num(3)?,
                },
                other => return fail(id, format!("the surface {other} is not read")),
            }
        };
        self.m.surfaces.push(s);
        let i = self.m.surfaces.len() - 1;
        self.surface_of.insert(id, i);
        Ok(i)
    }

    #[cfg(test)]
    fn spline_surface_exact(&self, id: u32) -> R<Surface> {
        self.spline_surface_with(id, false)
    }

    fn spline_surface(&self, id: u32) -> R<Surface> {
        self.spline_surface_with(id, true)
    }

    fn spline_surface_with(&self, id: u32, lower: bool) -> R<Surface> {
        let simple = self.x.kind(id) == Some("B_SPLINE_SURFACE_WITH_KNOTS");
        // The simple form: name, degrees, net, form, closed u, closed v,
        // self-intersect, then the knots' multiplicities and values.
        let (b, off) = if simple {
            (self.rec(id, "B_SPLINE_SURFACE_WITH_KNOTS")?, 1)
        } else {
            (self.rec(id, "B_SPLINE_SURFACE")?, 0)
        };
        let degree = [b.int(off)?.max(1) as usize, b.int(off + 1)?.max(1) as usize];
        let rows = b.list(off + 2)?;
        let mut ctrl = Vec::new();
        let mut nv = 0;
        for row in rows {
            let Some(row) = row.as_list() else {
                return fail(id, "a row of the control net is no list");
            };
            nv = row.len();
            for v in row {
                match v.as_ref() {
                    Some(p) => ctrl.push(self.point(p)?),
                    None => return fail(id, "a control point is no reference"),
                }
            }
        }
        let k = if simple {
            b
        } else {
            self.rec(id, "B_SPLINE_SURFACE_WITH_KNOTS")?
        };
        let at = if simple { off + 7 } else { 0 };
        let ku = expand(&k, at, at + 2)?;
        let kv = expand(&k, at + 1, at + 3)?;
        let weights: Vec<f64> = match self.x.record(id, "RATIONAL_B_SPLINE_SURFACE") {
            Some(r) => {
                let rows = Rec { id, r }.list(0)?;
                rows.iter()
                    .flat_map(|row| row.as_list().unwrap_or(&[]).iter())
                    .map(|v| v.as_f64().ok_or(()))
                    .collect::<Result<_, _>>()
                    .or_else(|_| fail(id, "a weight is no number"))?
            }
            None => vec![1.0; ctrl.len()],
        };
        match NurbsSurface::try_new(degree, [ku, kv], [rows.len(), nv], ctrl, weights) {
            Ok(s) => Ok(Surface::Nurbs(Arc::new(if lower {
                low_degree(s)
            } else {
                s
            }))),
            Err(e) => fail(id, e),
        }
    }

    fn vertex(&mut self, id: u32) -> R<usize> {
        if let Some(&v) = self.vertex_of.get(&id) {
            return Ok(v);
        }
        let p = self.point(self.rec(id, "VERTEX_POINT")?.refr(1)?)?;
        self.m.vertices.push(p);
        let i = self.m.vertices.len() - 1;
        self.vertex_of.insert(id, i);
        Ok(i)
    }

    fn edge(&mut self, id: u32) -> R<usize> {
        if let Some(&e) = self.edge_of.get(&id) {
            return Ok(e);
        }
        let r = self.rec(id, "EDGE_CURVE")?;
        let ends = [self.vertex(r.refr(1)?)?, self.vertex(r.refr(2)?)?];
        let geometry = r.refr(3)?;
        let forward = r.flag(4);
        let curve = match self.curve_of.get(&geometry) {
            Some(&c) => c,
            None => {
                let c = self.curve3(geometry)?;
                self.m.curves.push(c);
                let i = self.m.curves.len() - 1;
                self.curve_of.insert(geometry, i);
                i
            }
        };
        // The parameter curves of a SURFACE_CURVE or SEAM_CURVE (a seam has
        // two on its surface, one for each side).
        let mut pcurves = Vec::new();
        if let Ok(sc) = self.any(geometry) {
            if matches!(sc.r.name.as_str(), "SURFACE_CURVE" | "SEAM_CURVE") {
                for pc in sc.refs(2)? {
                    let Ok(p) = self.rec(pc, "PCURVE") else {
                        continue;
                    };
                    let surface = self.surface(p.refr(1)?)?;
                    let rep = self.rec(p.refr(2)?, "DEFINITIONAL_REPRESENTATION")?;
                    if let Some(&c) = rep.refs(1)?.first() {
                        let c = self.curve2(c)?;
                        let c = match self.reparam.get(&surface) {
                            Some(&(scale, shift)) => c.rescaled(scale, shift),
                            None => c,
                        };
                        pcurves.push((surface, c));
                    }
                }
            }
        }
        self.m.edges.push(Edge {
            ends,
            curve,
            forward,
            pcurves,
        });
        let i = self.m.edges.len() - 1;
        self.edge_of.insert(id, i);
        Ok(i)
    }

    fn face(&mut self, id: u32) -> R<usize> {
        if let Some(&f) = self.face_of.get(&id) {
            return Ok(f);
        }
        let r = self
            .rec(id, "ADVANCED_FACE")
            .or_else(|_| self.rec(id, "FACE_SURFACE"))?;
        let surface = self.surface(r.refr(2)?)?;
        let same_sense = r.flag(3) != self.flipped.contains(&surface);
        let mut bounds = Vec::new();
        for b in r.refs(1)? {
            let fb = self
                .rec(b, "FACE_OUTER_BOUND")
                .or_else(|_| self.rec(b, "FACE_BOUND"))?;
            let (lp, along) = (fb.refr(1)?, fb.flag(2));
            if let Ok(vl) = self.rec(lp, "VERTEX_LOOP") {
                bounds.push(Bound::Vertex(self.vertex(vl.refr(1)?)?));
                continue;
            }
            let mut edges = Vec::new();
            for oe in self.rec(lp, "EDGE_LOOP")?.refs(1)? {
                let o = self.rec(oe, "ORIENTED_EDGE")?;
                edges.push((self.edge(o.refr(3)?)?, o.flag(4)));
            }
            if !along {
                edges.reverse();
                edges.iter_mut().for_each(|e| e.1 = !e.1);
            }
            bounds.push(Bound::Edges(edges));
        }
        self.m.faces.push(Face {
            id,
            surface,
            same_sense,
            bounds,
        });
        let i = self.m.faces.len() - 1;
        self.face_of.insert(id, i);
        Ok(i)
    }

    /// A copy of face `i` facing the other way: its normal against its
    /// surface's where it was with it, its loops run backwards.
    fn reversed(&mut self, i: usize) -> usize {
        let mut f = self.m.faces[i].clone();
        f.same_sense = !f.same_sense;
        for b in &mut f.bounds {
            if let Bound::Edges(edges) = b {
                edges.reverse();
                edges.iter_mut().for_each(|e| e.1 = !e.1);
            }
        }
        self.m.faces.push(f);
        self.m.faces.len() - 1
    }

    /// Every solid: its faces, its name, where the assembly puts it.
    fn solids(&mut self) -> R<()> {
        let names = self.product_names();
        let places = self.placements();
        let mut ids = self.x.all("MANIFOLD_SOLID_BREP");
        ids.extend(self.x.all("BREP_WITH_VOIDS"));
        ids.sort_unstable();
        ids.dedup();
        for id in ids {
            let r = self
                .rec(id, "MANIFOLD_SOLID_BREP")
                .or_else(|_| self.rec(id, "BREP_WITH_VOIDS"))?;
            let mut shells = vec![r.refr(1)?];
            if r.r.name == "BREP_WITH_VOIDS" {
                shells.extend(r.refs(2)?);
            }
            let mut faces = Vec::new();
            for s in shells {
                // An oriented shell (the void of a BREP_WITH_VOIDS) names a
                // closed shell and whether it keeps its orientation.
                let (s, keep) = match self.rec(s, "ORIENTED_CLOSED_SHELL") {
                    Ok(o) => (o.refr(2)?, o.flag(3)),
                    Err(_) => (s, true),
                };
                for f in self.rec(s, "CLOSED_SHELL")?.refs(1)? {
                    let f = self.face(f)?;
                    faces.push(if keep { f } else { self.reversed(f) });
                }
            }
            let rep = self.representation_of(id);
            let name = rep
                .and_then(|r| names.get(&r).cloned())
                .filter(|n| !n.is_empty() && !n.starts_with("Open CASCADE STEP translator"))
                .unwrap_or_else(|| r.name());
            let placement = rep
                .and_then(|r| places.get(&r).copied())
                .unwrap_or(Affine::IDENTITY);
            self.m.solids.push(Solid {
                name,
                faces,
                placement,
            });
        }
        Ok(())
    }

    /// The shape representation that holds item `id`.
    fn representation_of(&self, id: u32) -> Option<u32> {
        [
            "ADVANCED_BREP_SHAPE_REPRESENTATION",
            "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
            "SHAPE_REPRESENTATION",
        ]
        .iter()
        .flat_map(|n| self.x.all(n))
        .find(|&rep| {
            self.x.instances[&rep].iter().any(|r| {
                r.args
                    .get(1)
                    .and_then(Value::as_list)
                    .is_some_and(|l| l.iter().any(|v| v.as_ref() == Some(id)))
            })
        })
    }

    /// The name of the product of each shape representation that has one,
    /// directly or through a representation relationship.
    fn product_names(&self) -> FxHashMap<u32, String> {
        let mut out: FxHashMap<u32, String> = FxHashMap::default();
        for sdr in self.x.all("SHAPE_DEFINITION_REPRESENTATION") {
            let Ok(r) = self.rec(sdr, "SHAPE_DEFINITION_REPRESENTATION") else {
                continue;
            };
            let (Ok(pds), Ok(rep)) = (r.refr(0), r.refr(1)) else {
                continue;
            };
            let name = (|| -> Option<String> {
                let pd = self
                    .x
                    .record(pds, "PRODUCT_DEFINITION_SHAPE")?
                    .args
                    .get(2)?
                    .as_ref()?;
                let pdf = self
                    .x
                    .record(pd, "PRODUCT_DEFINITION")?
                    .args
                    .get(2)?
                    .as_ref()?;
                let p = self
                    .x
                    .record(pdf, "PRODUCT_DEFINITION_FORMATION")
                    .or_else(|| {
                        self.x
                            .record(pdf, "PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE")
                    })?
                    .args
                    .get(2)?
                    .as_ref()?;
                Some(
                    self.x
                        .record(p, "PRODUCT")?
                        .args
                        .first()?
                        .as_str()?
                        .to_string(),
                )
            })();
            if let Some(n) = name {
                out.insert(rep, n);
            }
        }
        // A B-rep representation related to a named one takes its name.
        for rel in self.x.all("SHAPE_REPRESENTATION_RELATIONSHIP") {
            let Some(r) = self.x.record(rel, "REPRESENTATION_RELATIONSHIP") else {
                continue;
            };
            let (Some(a), Some(b)) = (
                r.args.get(2).and_then(Value::as_ref),
                r.args.get(3).and_then(Value::as_ref),
            ) else {
                continue;
            };
            if self
                .x
                .record(rel, "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")
                .is_some()
            {
                continue;
            }
            for (x, y) in [(a, b), (b, a)] {
                if !out.contains_key(&x) {
                    if let Some(n) = out.get(&y).cloned() {
                        out.insert(x, n);
                    }
                }
            }
        }
        out
    }

    /// Where the assembly places each shape representation: the maps of
    /// its transforming relationships up to the root, composed.
    fn placements(&self) -> FxHashMap<u32, Affine> {
        // child -> (parent, map into the parent)
        let mut up: FxHashMap<u32, (u32, Affine)> = FxHashMap::default();
        let mut same: Vec<(u32, u32)> = Vec::new();
        for rel in self.x.all("REPRESENTATION_RELATIONSHIP") {
            let Some(r) = self.x.record(rel, "REPRESENTATION_RELATIONSHIP") else {
                continue;
            };
            let (Some(child), Some(parent)) = (
                r.args.get(2).and_then(Value::as_ref),
                r.args.get(3).and_then(Value::as_ref),
            ) else {
                continue;
            };
            let map = self
                .x
                .record(rel, "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")
                .and_then(|t| t.args.first()?.as_ref())
                .and_then(|t| self.x.record(t, "ITEM_DEFINED_TRANSFORMATION"))
                .and_then(|t| {
                    let from = self.placement(t.args.get(2)?.as_ref()?).ok()?;
                    let to = self.placement(t.args.get(3)?.as_ref()?).ok()?;
                    Some(between(&from, &to))
                });
            match map {
                Some(m) => {
                    up.insert(child, (parent, m));
                }
                None => same.push((child, parent)),
            }
        }
        // A representation related without a transform sits where the other
        // does (a B-rep inside its product's shape).
        for _ in 0..2 {
            for &(a, b) in &same {
                for (x, y) in [(a, b), (b, a)] {
                    if !up.contains_key(&x) {
                        if let Some(&u) = up.get(&y) {
                            up.insert(x, u);
                        }
                    }
                }
            }
        }
        let mut out = FxHashMap::default();
        for &rep in up.keys().chain(same.iter().flat_map(|(a, b)| [a, b])) {
            let mut m = Affine::IDENTITY;
            let mut at = rep;
            for _ in 0..64 {
                match up.get(&at) {
                    Some(&(parent, step)) => {
                        m = m.then(&step);
                        at = parent;
                    }
                    None => break,
                }
            }
            out.insert(rep, m);
        }
        out
    }
}

/// The highest degree a B-spline carrier keeps.
const CARRIER_DEGREE: usize = 5;

/// How far a lower-degree carrier may lie from the surface it replaces,
/// relative to the size of its control net.
const CARRIER_FIT: f64 = 1e-6;

/// `s` where its degree is at most [`CARRIER_DEGREE`]; else a polynomial
/// B-spline within [`CARRIER_FIT`] of it over the same parameters (so the
/// file's parameter curves still hold): each direction of higher degree
/// in equal spans of that degree, as few as it takes, the others on their
/// own knots. A CAD loft comes out of degree 14 in one span round its
/// section: every evaluation costs the square of that, and the patch
/// pruning of a projection has nothing to prune.
fn low_degree(s: NurbsSurface) -> NurbsSurface {
    if s.degree.iter().all(|&p| p <= CARRIER_DEGREE) {
        return s;
    }
    let (lo, hi) = bbox(&s.ctrl);
    let size = (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f64>().sqrt();
    let (du, dv) = s.domain();
    let degree = s.degree.map(|p| p.min(CARRIER_DEGREE));
    let f = |u: f64, v: f64| s.eval(u, v);
    let mut best: Option<(NurbsSurface, f64)> = None;
    for spans in [2, 4, 8, 16, 32, 64] {
        let knots = [0, 1].map(|d| {
            if s.degree[d] <= CARRIER_DEGREE {
                s.knots[d].clone()
            } else {
                NurbsSurface::uniform_knots([du, dv][d], degree[d], spans)
            }
        });
        let (fit, err) = NurbsSurface::fit(&f, degree, knots);
        if best.as_ref().is_none_or(|b| err < b.1) {
            best = Some((fit, err));
        }
        if err <= CARRIER_FIT * size {
            break;
        }
    }
    match best {
        Some((fit, err)) if err <= 1e2 * CARRIER_FIT * size => fit,
        _ => s,
    }
}

/// Knots from the multiplicities at parameter `m` and the values at `k`.
fn expand(r: &Rec, m: usize, k: usize) -> R<Vec<f64>> {
    let mults = r.list(m)?;
    let values = r.nums(k)?;
    let mut out = Vec::new();
    for (mv, kv) in mults.iter().zip(values) {
        let Some(n) = mv.as_int() else {
            return fail(r.id, "a knot multiplicity is no integer");
        };
        out.extend(std::iter::repeat_n(kv, n.max(0) as usize));
    }
    Ok(out)
}

/// The units of `x`: of length in metres, of plane angles in radians.
/// Those its geometry is given in are the ones a context assigns to it;
/// where none does, the first of the file's, else millimetres and
/// radians.
fn units(x: &Exchange) -> [f64; 2] {
    let assigned: Vec<u32> = x
        .all("GLOBAL_UNIT_ASSIGNED_CONTEXT")
        .into_iter()
        .filter_map(|c| x.record(c, "GLOBAL_UNIT_ASSIGNED_CONTEXT"))
        .flat_map(|r| {
            r.args
                .first()
                .and_then(|v| v.as_list())
                .unwrap_or(&[])
                .to_vec()
        })
        .filter_map(|v| v.as_ref())
        .collect();
    let find = |kind: &str, fallback: f64| {
        let mut candidates: Vec<u32> = assigned
            .iter()
            .copied()
            .filter(|&u| x.record(u, kind).is_some())
            .collect();
        if candidates.is_empty() {
            candidates = x.all(kind);
        }
        candidates
            .into_iter()
            .find_map(|u| unit_value(x, u, 0))
            .unwrap_or(fallback)
    };
    [find("LENGTH_UNIT", 1e-3), find("PLANE_ANGLE_UNIT", 1.0)]
}

/// The size of unit `u` in SI units (metres, radians): an SI unit with its
/// prefix, or a unit based on another by a measure of it.
fn unit_value(x: &Exchange, u: u32, depth: usize) -> Option<f64> {
    if let Some(si) = x.record(u, "SI_UNIT") {
        return Some(
            match si.args.first().and_then(|v| v.as_enum()).unwrap_or("") {
                "KILO" => 1e3,
                "CENTI" => 1e-2,
                "MILLI" => 1e-3,
                "MICRO" => 1e-6,
                "NANO" => 1e-9,
                _ => 1.0,
            },
        );
    }
    let c = x.record(u, "CONVERSION_BASED_UNIT")?;
    let by_measure = c.args.get(1).and_then(|v| v.as_ref()).and_then(|m| {
        let r = x
            .record(m, "LENGTH_MEASURE_WITH_UNIT")
            .or_else(|| x.record(m, "PLANE_ANGLE_MEASURE_WITH_UNIT"))
            .or_else(|| x.record(m, "MEASURE_WITH_UNIT"))?;
        let value = r.args.first()?.as_f64()?;
        let base = r.args.get(1)?.as_ref()?;
        (depth < 4)
            .then(|| unit_value(x, base, depth + 1))
            .flatten()
            .map(|b| value * b)
    });
    // A degree written as a rounded factor (0.0174532925) is a degree.
    let degree = std::f64::consts::PI / 180.0;
    let by_measure = by_measure.map(|f| {
        if (f / degree - 1.0).abs() < 1e-6 {
            degree
        } else {
            f
        }
    });
    by_measure.or_else(|| {
        let name = c
            .args
            .first()
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_uppercase();
        match () {
            _ if name.contains("INCH") => Some(0.0254),
            _ if name.contains("FOOT") => Some(0.3048),
            _ if name.contains("DEGREE") => Some(std::f64::consts::PI / 180.0),
            _ => None,
        }
    })
}

/// The typed model of `x`.
pub fn decode(x: &Exchange) -> Result<Model, StepError> {
    let mut d = Decoder::new(x);
    d.solids()?;
    Ok(d.m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapidmesh_exact::vector::dist;

    #[test]
    fn a_map_between_frames_takes_one_onto_the_other() {
        let a = placement([1.0, 2.0, 3.0], [0.0, 0.0, 1.0], Some([1.0, 0.0, 0.0])).unwrap();
        let b = placement([-4.0, 0.5, 2.0], [1.0, 0.0, 0.0], Some([0.0, 1.0, 0.0])).unwrap();
        let m = between(&a, &b);
        let p = m.point([1.0, 2.0, 3.0]);
        assert!((0..3).all(|k| (p[k] - b.o[k]).abs() < 1e-12));
        // a.x maps onto b.x.
        let q = m.point([2.0, 2.0, 3.0]);
        assert!((0..3).all(|k| (q[k] - (b.o[k] + b.x[k])).abs() < 1e-12));
    }

    /// The loft of the fixtures, of degree 14 round its section, comes in
    /// as a carrier of degree 5 within the fit of the file's surface.
    #[test]
    fn a_high_degree_carrier_comes_in_lowered() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/loft.step");
        let x = crate::part21::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
        let d = Decoder::new(&x);
        let spline = |s: Surface| match s {
            Surface::Nurbs(s) => s,
            _ => panic!("no B-spline"),
        };
        let exact = spline(d.spline_surface_exact(76).unwrap());
        let lowered = spline(d.spline_surface(76).unwrap());
        assert_eq!(exact.degree, [14, 3]);
        assert_eq!(lowered.degree, [5, 3]);
        let (du, dv) = exact.domain();
        for i in 0..=40 {
            for j in 0..=10 {
                let u = du[0] + (du[1] - du[0]) * i as f64 / 40.0;
                let v = dv[0] + (dv[1] - dv[0]) * j as f64 / 10.0;
                let (p, q) = (exact.eval(u, v), lowered.eval(u, v));
                assert!(dist(p, q) < 1e-4, "{u} {v}: {}", dist(p, q));
            }
        }
    }

    /// The simple form of a B-spline surface (as KiCad writes it) reads
    /// its knots after the self-intersection flag.
    #[test]
    fn a_simple_b_spline_surface_reads() {
        let text = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n\
            #1=CARTESIAN_POINT('',(0.,0.,0.));\n#2=CARTESIAN_POINT('',(0.,1.,0.));\n\
            #3=CARTESIAN_POINT('',(2.,0.,0.));\n#4=CARTESIAN_POINT('',(2.,1.,1.));\n\
            #5=B_SPLINE_SURFACE_WITH_KNOTS('',1,1,((#1,#2),(#3,#4)),.UNSPECIFIED.,.F.,.F.,.F.,\
            (2,2),(2,2),(0.,1.21),(0.,1.),.PIECEWISE_BEZIER_KNOTS.);\nENDSEC;\n";
        let x = crate::part21::parse(text).unwrap();
        let d = Decoder::new(&x);
        let Surface::Nurbs(s) = d.spline_surface(5).unwrap() else {
            panic!("no B-spline");
        };
        assert_eq!(s.domain(), ([0.0, 1.21], [0.0, 1.0]));
        assert!(dist(s.eval(1.21, 1.0), [2.0, 1.0, 1.0]) < 1e-12);
    }

    /// A model in degrees (as Creo writes it): the unit the context
    /// assigns, based on radians by a measure, not the radian it is
    /// defined in.
    #[test]
    fn angles_come_in_the_unit_the_context_assigns() {
        let text = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n\
            #1=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#5,#2)) \
            REPRESENTATION_CONTEXT('',''));\n\
            #2=(CONVERSION_BASED_UNIT('DEGREE',#3) NAMED_UNIT(#9) PLANE_ANGLE_UNIT());\n\
            #3=PLANE_ANGLE_MEASURE_WITH_UNIT(PLANE_ANGLE_MEASURE(0.0174532925),#4);\n\
            #4=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));\n\
            #5=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));\nENDSEC;\n";
        let x = crate::part21::parse(text).unwrap();
        let [metres, radians] = units(&x);
        assert_eq!(metres, 1e-3);
        assert_eq!(radians, std::f64::consts::PI / 180.0);
    }

    #[test]
    fn a_missing_entity_is_named() {
        let text = "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n\
            #1 = MANIFOLD_SOLID_BREP('',#2);\n#2 = CLOSED_SHELL('',(#3));\n\
            #3 = ADVANCED_FACE('',(),#4,.T.);\n#4 = PLANE('',#5);\nENDSEC;\n";
        let x = crate::part21::parse(text).unwrap();
        let e = decode(&x).unwrap_err();
        assert_eq!(e.id, 5);
    }
}
