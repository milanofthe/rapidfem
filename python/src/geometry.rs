// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! `_native.Geometry`, `_native.FemMesh` and `_native.MeshStats`: the
//! rapidfem geometry on the vendored rapidmesh (`rapidfem_geom`), the
//! materials and physics placed on it, and the solver mesh it produces.
//!
//! Objects are addressed by their index, faces by their selection key
//! `(object, role, side, across)` (`FaceSel::key`), edges by
//! `(object, (role_a, role_b))` and the groups of a loaded mesh by name.
//! The Python `rapidfem.Geometry` is a thin layer over these calls.

use std::collections::BTreeSet;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rapidfem_core::geom::{norm, scale};
use rapidfem_core::model::{Debye, Drude, FaceSpec, MaterialSpec, PmlSpec, WaveKind};
use rapidfem_geom::fem_mesh::{fem_mesh, viewer_mesh, Group, MeshStats};
use rapidfem_geom::geometry::{normalized, polygon_sheet, EdgeOp, Extent, FaceSel, Geometry, ObjId, SELECT_TOL};
use rapidfem_geom::msh::{write_msh, MeshScene};
use rapidfem_geom::rfic::Scene;
use rapidfem_geom::setup::{Condition, Physics, Setup, Tagging, Target};
use rapidfem_geom::sheet_ops::SheetOp;
use rapidmesh::shapes::{Cone, Cuboid, Cylinder, Helix, Import, Sheet, Sphere, Torus, Wedge};
use rapidmesh::{EdgeCut, EdgePick, MeshOptions, Transform};

use crate::model::PyModel;

type P3 = [f64; 3];

/// A face selection key `(object, role, side, across)`, see `FaceSel::key`.
type PySel = (usize, i64, i64, i64);

/// What a Python handle names.
#[derive(FromPyObject)]
enum Key {
    /// A face selection, `(object, role, side, across)`.
    Face(usize, i64, i64, i64),
    /// An edge of a solid, `(object, (role_a, role_b))`.
    Edge(usize, (u32, u32)),
    /// A solid object.
    #[pyo3(transparent)]
    Object(usize),
    /// A group of a loaded mesh.
    #[pyo3(transparent)]
    Group(String),
}

fn err(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
}

const MESH_MODE: &str = "the geometry is in mesh mode (a .msh file was loaded): it takes \
                         materials and physics on the file's groups, not new objects";

const SOLID_SHELL: &str = "SurfaceImpedance: the targets cover the complete shell of a solid \
                           that is still meshed. On internal faces the BC is a transition sheet \
                           and the field enters the conductor core, so the loss comes out wrong. \
                           Cut the conductor out of the mesh and put the BC on the walls \
                           (two_sided=True with thickness=2V/S), or mesh it as a volume conductor.";

/// The model entry of a `rapidfem.Material` (its volume tag set later).
fn material_spec(m: &Bound<'_, PyAny>) -> PyResult<MaterialSpec> {
    let value = |name: &str| -> PyResult<f64> { m.getattr(name)?.extract() };
    let diag = |name: &str| -> PyResult<Option<[f64; 3]>> { m.getattr(name)?.extract() };
    let debye = m.getattr("debye")?;
    let drude = m.getattr("drude")?;
    let of = |o: &Bound<'_, PyAny>, name: &str| -> PyResult<f64> { o.getattr(name)?.extract() };
    Ok(MaterialSpec {
        volume_tag: 0,
        er: value("er")?,
        ur: value("ur")?,
        tand: value("tand")?,
        conductivity: value("conductivity")?,
        cond_diag: diag("cond_diag")?,
        er_diag: diag("er_diag")?,
        ur_diag: diag("ur_diag")?,
        debye: if debye.is_none() {
            None
        } else {
            Some(Debye {
                er_inf: of(&debye, "er_inf")?,
                er_static: of(&debye, "er_static")?,
                tau_s: of(&debye, "tau_s")?,
            })
        },
        drude: if drude.is_none() {
            None
        } else {
            Some(Drude {
                er_inf: of(&drude, "er_inf")?,
                plasma_freq_hz: of(&drude, "plasma_freq_hz")?,
                damping_freq_hz: of(&drude, "damping_freq_hz")?,
            })
        },
    })
}

/// The rapidfem geometry: solids and sheets on rapidmesh (or a loaded
/// mesh), the materials and physics placed on them, and their mesh.
#[pyclass(name = "Geometry", module = "rapidfem._native", unsendable)]
pub struct PyGeometry {
    inner: Geometry,
    setup: Setup,
    /// The Python material objects, by material index of `setup`.
    materials: Vec<Py<PyAny>>,
    /// A loaded mesh, the whole geometry (mesh mode).
    scene: Option<MeshScene>,
    grading: bool,
    /// The size `mesh()` takes when its call passes none.
    #[pyo3(get, set)]
    pub(crate) mesh_maxh: Option<f64>,
    /// The elements across every region `mesh()` asks for when its call
    /// passes none.
    #[pyo3(get, set)]
    pub(crate) cells_across: Option<f64>,
    mesh: Option<rapidmesh::Mesh>,
    /// The tags of the last `mesh()`, with the setup revision they are of.
    meshed: Option<(u64, Tagging)>,
}

#[pymethods]
impl PyGeometry {
    #[new]
    #[pyo3(signature = (maxh=None, grading=true))]
    fn new(maxh: Option<f64>, grading: bool) -> Self {
        PyGeometry::fresh(maxh, grading)
    }

    /// The physics objects from index `start` on as `(kind, params,
    /// targets)`: the Python class name, its constructor parameters and
    /// the target keys, for the Python objects of physics a native builder
    /// placed.
    fn physics_since<'py>(&self, py: Python<'py>, start: usize) -> PyResult<Vec<(String, Bound<'py, PyDict>, Vec<Bound<'py, PyAny>>)>> {
        let mut out = Vec::new();
        for p in self.setup.physics().iter().skip(start) {
            let d = PyDict::new(py);
            let kind = match &p.condition {
                Condition::Pec => "PEC",
                Condition::FarField => "FarFieldSurface",
                Condition::Pml(s) => {
                    d.set_item("direction", s.direction)?;
                    d.set_item("inner_face", s.inner_face)?;
                    d.set_item("thickness", s.thickness)?;
                    d.set_item("er_base", s.er_base)?;
                    d.set_item("ur_base", s.ur_base)?;
                    d.set_item("exponent", s.exponent)?;
                    d.set_item("delta_max", s.delta_max)?;
                    "PML"
                }
                Condition::Face(FaceSpec::Abc { .. }) => "ABC",
                Condition::Face(FaceSpec::Pmc { .. }) => "PMC",
                Condition::Face(FaceSpec::Lumped { z0, l, c, direction, width, height, power, .. }) => {
                    d.set_item("direction", *direction)?;
                    d.set_item("z0", *z0)?;
                    d.set_item("l", *l)?;
                    d.set_item("c", *c)?;
                    d.set_item("power", *power)?;
                    d.set_item("width", *width)?;
                    d.set_item("height", *height)?;
                    "LumpedPort"
                }
                Condition::Face(FaceSpec::SurfaceImpedance { conductivity, mur, er, thickness, two_sided, sheet, zs, .. }) => {
                    d.set_item("conductivity", *conductivity)?;
                    d.set_item("mur", *mur)?;
                    d.set_item("er", *er)?;
                    d.set_item("thickness", *thickness)?;
                    d.set_item("two_sided", *two_sided)?;
                    d.set_item("sheet", *sheet)?;
                    d.set_item("zs", *zs)?;
                    "SurfaceImpedance"
                }
                other => return Err(PyNotImplementedError::new_err(format!("no Python object for the native physics {other:?}"))),
            };
            let keys = p
                .targets
                .iter()
                .map(|t| match t {
                    Target::Face(sel) => sel.key().into_pyobject(py).map(|k| k.into_any()),
                    Target::Object(o) => Ok(o.into_pyobject(py)?.into_any()),
                    Target::Group { name, .. } => Ok(name.into_pyobject(py)?.into_any()),
                })
                .collect::<PyResult<Vec<_>>>()?;
            out.push((kind.to_string(), d, keys));
        }
        Ok(out)
    }

    // ── objects ─────────────────────────────────────────────────────────

    #[pyo3(signature = (size, position=[0.0; 3], maxh=None))]
    fn add_box(&mut self, size: P3, position: P3, maxh: Option<f64>) -> PyResult<ObjId> {
        Ok(self.geo()?.add_solid(Cuboid::new(size).at(position), maxh, false))
    }

    #[pyo3(signature = (radius, height, position=[0.0; 3], axis=[0.0, 0.0, 1.0], maxh=None))]
    fn add_cylinder(&mut self, radius: f64, height: f64, position: P3, axis: P3, maxh: Option<f64>) -> PyResult<ObjId> {
        Ok(self.geo()?.add_solid(Cylinder::new(radius, height).at(position).along(axis), maxh, false))
    }

    #[pyo3(signature = (radius, position=[0.0; 3], maxh=None))]
    fn add_sphere(&mut self, radius: f64, position: P3, maxh: Option<f64>) -> PyResult<ObjId> {
        Ok(self.geo()?.add_solid(Sphere::new(radius).at(position), maxh, false))
    }

    #[pyo3(signature = (r1, r2, height, position=[0.0; 3], axis=[0.0, 0.0, 1.0], maxh=None))]
    fn add_cone(&mut self, r1: f64, r2: f64, height: f64, position: P3, axis: P3, maxh: Option<f64>) -> PyResult<ObjId> {
        Ok(self.geo()?.add_solid(Cone::new(r1, r2, height).at(position).along(axis), maxh, false))
    }

    #[pyo3(signature = (size, top_x, position=[0.0; 3], maxh=None))]
    fn add_wedge(&mut self, size: P3, top_x: f64, position: P3, maxh: Option<f64>) -> PyResult<ObjId> {
        let mut w = Wedge::new(size).at(position);
        w.top_x = top_x;
        Ok(self.geo()?.add_solid(w, maxh, false))
    }

    #[pyo3(signature = (major_radius, minor_radius, position=[0.0; 3], axis=[0.0, 0.0, 1.0], maxh=None))]
    fn add_torus(&mut self, major_radius: f64, minor_radius: f64, position: P3, axis: P3, maxh: Option<f64>) -> PyResult<ObjId> {
        let mut t = Torus::new(major_radius, minor_radius).at(position);
        t.axis = axis;
        Ok(self.geo()?.add_solid(t, maxh, false))
    }

    /// The parallelogram from `corner` spanned by `u` and `v`.
    #[pyo3(signature = (corner, u, v, maxh=None))]
    fn add_plate(&mut self, corner: P3, u: P3, v: P3, maxh: Option<f64>) -> PyResult<ObjId> {
        Ok(self.geo()?.add_sheet(Sheet::plate(corner, u, v), maxh))
    }

    #[pyo3(signature = (radius, center, axis=[0.0, 0.0, 1.0], maxh=None))]
    fn add_disc(&mut self, radius: f64, center: P3, axis: P3, maxh: Option<f64>) -> PyResult<ObjId> {
        Ok(self.geo()?.add_sheet(Sheet::disc(radius, center, axis), maxh))
    }

    /// A planar polygon with holes, its vertices `(x, y)` (at z = 0) or
    /// `(x, y, z)`, offset by `position`, in the plane of its vertices (see
    /// `polygon_sheet`).
    #[pyo3(signature = (points, position=[0.0; 3], holes=Vec::new(), maxh=None))]
    fn add_polygon(&mut self, points: Vec<Vec<f64>>, position: P3, holes: Vec<Vec<Vec<f64>>>, maxh: Option<f64>) -> PyResult<ObjId> {
        let [x0, y0, z0] = position;
        let lift = |pts: Vec<Vec<f64>>| -> PyResult<Vec<P3>> {
            pts.iter()
                .map(|p| match p.as_slice() {
                    [x, y] => Ok([x + x0, y + y0, z0]),
                    [x, y, z] => Ok([x + x0, y + y0, z + z0]),
                    _ => Err(PyValueError::new_err("a polygon vertex is (x, y) or (x, y, z)")),
                })
                .collect()
        };
        if points.len() < 3 {
            return Err(PyValueError::new_err("polygon needs at least 3 vertices"));
        }
        let points = lift(points)?;
        let holes = holes.into_iter().map(&lift).collect::<PyResult<Vec<_>>>()?;
        let sheet = polygon_sheet(&points, &holes)
            .ok_or_else(|| PyValueError::new_err("polygon: the vertices are not in one plane"))?;
        Ok(self.geo()?.add_sheet(sheet, maxh))
    }

    /// Extrudes a sheet by `height` along `axis` into a solid, in place (see
    /// `Geometry::extrude`); `maxh`, if given, becomes the solid's size.
    #[pyo3(signature = (id, height, axis, maxh=None))]
    fn extrude(&mut self, id: ObjId, height: f64, axis: P3, maxh: Option<f64>) -> PyResult<()> {
        let n = norm(axis);
        if n == 0.0 {
            return Err(PyValueError::new_err("extrude: axis must be a nonzero vector"));
        }
        let g = self.geo()?;
        g.extrude(id, scale(axis, height / n)).map_err(|e| PyValueError::new_err(format!("extrude: {e}")))?;
        if maxh.is_some() {
            g.set_object_maxh(id, maxh);
        }
        Ok(())
    }

    /// The ruled solid between two sheet profiles; they are used up.
    #[pyo3(signature = (a, b, maxh=None))]
    fn loft(&mut self, a: ObjId, b: ObjId, maxh: Option<f64>) -> PyResult<ObjId> {
        self.geo()?.loft(a, b, maxh).map_err(|e| PyValueError::new_err(format!("loft: {e}")))
    }

    /// The solid a sheet profile sweeps turning by `angle` radians about the
    /// axis along `axis` through `point`; the profile is used up.
    #[pyo3(signature = (id, point, axis, angle, maxh=None))]
    fn revolve(&mut self, id: ObjId, point: P3, axis: P3, angle: f64, maxh: Option<f64>) -> PyResult<ObjId> {
        self.geo()?.revolve(id, point, axis, angle, maxh).map_err(|e| PyValueError::new_err(format!("revolve: {e}")))
    }

    /// A round tube along the spline through `points`, its radius that of
    /// the unmoved disc `profile`, which is used up.
    #[pyo3(signature = (profile, points, maxh=None))]
    fn sweep(&mut self, profile: ObjId, points: Vec<P3>, maxh: Option<f64>) -> PyResult<ObjId> {
        self.geo()?.sweep(profile, &points, maxh).map_err(|e| PyValueError::new_err(format!("sweep_along_path: {e}")))
    }

    /// A helical coil of round wire about +z through `position`.
    #[pyo3(signature = (radius, pitch, turns, wire_radius, position=[0.0; 3], points_per_turn=24, segments=12, maxh=None))]
    fn add_helix(
        &mut self,
        radius: f64,
        pitch: f64,
        turns: f64,
        wire_radius: f64,
        position: P3,
        points_per_turn: usize,
        segments: usize,
        maxh: Option<f64>,
    ) -> PyResult<ObjId> {
        let mut h = Helix::new(radius, pitch, turns, wire_radius);
        h.position = position;
        h.points_per_turn = points_per_turn;
        h.segments = segments;
        Ok(self.geo()?.add_solid(h, maxh, false))
    }

    /// The solids of a STEP file (one object each, in the file's unit) and
    /// the length of that unit in metres.
    #[pyo3(signature = (path, maxh=None))]
    fn add_step(&mut self, path: String, maxh: Option<f64>) -> PyResult<(Vec<ObjId>, f64)> {
        self.geo()?.add_step(std::path::Path::new(&path), maxh).map_err(PyValueError::new_err)
    }

    /// A closed STL or OBJ surface as a solid, split into smooth surfaces
    /// at creases sharper than `crease_deg`.
    #[pyo3(signature = (path, crease_deg=40.0, maxh=None))]
    fn add_import(&mut self, path: String, crease_deg: f64, maxh: Option<f64>) -> PyResult<ObjId> {
        let mut i = Import::new(path);
        i.crease_deg = crease_deg;
        Ok(self.geo()?.add_solid(i, maxh, false))
    }

    /// A copy of an object in the same place (its own region), with its
    /// size and material; returns its id.
    fn copy(&mut self, id: ObjId) -> PyResult<ObjId> {
        let new = self.geo()?.copy(id);
        let material = self.setup.material(&Target::Object(id));
        if material.is_some() {
            self.setup.set_material(Target::Object(new), material);
        }
        Ok(new)
    }

    /// Chamfers (`fillet=false`, `size` = distance) or fillets (`size` =
    /// radius) edges of a solid, picked by the roles of the two faces
    /// meeting there; `None` cuts every edge.
    #[pyo3(signature = (id, size, fillet, edges=None, void=false))]
    fn cut_edges(&mut self, id: ObjId, size: f64, fillet: bool, edges: Option<Vec<(u32, u32)>>, void: bool) -> PyResult<()> {
        let edges = match edges {
            None => vec![EdgePick::All],
            Some(e) => e.into_iter().map(|(a, b)| EdgePick::Between(a, b)).collect(),
        };
        let cut = if fillet { EdgeCut::Fillet(size) } else { EdgeCut::Chamfer(size) };
        self.geo()?.cut_edges(EdgeOp { object: id, edges, cut, void });
        Ok(())
    }

    fn translate(&mut self, id: ObjId, d: P3) -> PyResult<()> {
        self.geo()?.transform(id, Transform::Translate(d));
        Ok(())
    }

    /// Turns an object by `angle` radians about the axis along `axis`
    /// through `center` (right-handed).
    fn rotate(&mut self, id: ObjId, angle: f64, axis: P3, center: P3) -> PyResult<()> {
        self.geo()?.transform(id, Transform::Rotate { angle, axis: normalized(axis), center });
        Ok(())
    }

    /// Mirrors an object across the plane through `point` with `normal`.
    fn mirror(&mut self, id: ObjId, normal: P3, point: P3) -> PyResult<()> {
        self.geo()?.transform(id, Transform::Mirror { normal: normalized(normal), point });
        Ok(())
    }

    /// Stretches an object by `factors` along x, y and z about `center`.
    fn stretch(&mut self, id: ObjId, factors: P3, center: P3) -> PyResult<()> {
        self.geo()?.transform(id, Transform::Stretch { factors, center });
        Ok(())
    }

    /// `target` becomes what it shares with every tool; the tools are used
    /// up and lose their material.
    fn intersect(&mut self, target: ObjId, tools: Vec<ObjId>) -> PyResult<()> {
        let g = self.geo()?;
        g.check_solids(&[target]).and_then(|_| g.check_solids(&tools)).map_err(PyValueError::new_err)?;
        g.intersect(target, tools.clone());
        for t in tools {
            self.setup.set_material(Target::Object(t), None);
        }
        Ok(())
    }

    /// Fuses solid objects into the first one's material region.
    fn fuse(&mut self, ids: Vec<ObjId>) -> PyResult<()> {
        let g = self.geo()?;
        g.check_solids(&ids).map_err(PyValueError::new_err)?;
        g.fuse(ids);
        Ok(())
    }

    /// Combines the sheet `target` with the sheets `tools` in their common
    /// plane: `op` is "union", "difference" or "intersection"; the tools
    /// are used up.
    fn sheet_boolean(&mut self, op: &str, target: ObjId, tools: Vec<ObjId>) -> PyResult<()> {
        let op = match op {
            "union" => SheetOp::Union,
            "difference" => SheetOp::Difference,
            "intersection" => SheetOp::Intersection,
            _ => return Err(PyValueError::new_err(format!("unknown sheet boolean {op:?}"))),
        };
        self.geo()?.sheet_boolean(op, target, &tools).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Puts the solids of `ids` above every other one, in order (the last
    /// wins); sheets have no region to order and are left as they are.
    fn bring_to_front(&mut self, ids: Vec<ObjId>) -> PyResult<()> {
        let g = self.geo()?;
        let solids: Vec<ObjId> = ids.into_iter().filter(|&i| g.object(i).is_solid()).collect();
        if !solids.is_empty() {
            g.bring_to_front(&solids);
        }
        Ok(())
    }

    /// Turns solids into holes, their walls boundary faces.
    fn make_void(&mut self, ids: Vec<ObjId>) -> PyResult<()> {
        let g = self.geo()?;
        g.check_solids(&ids).map_err(PyValueError::new_err)?;
        g.make_void(&ids);
        Ok(())
    }

    fn remove(&mut self, id: ObjId) -> PyResult<()> {
        self.geo()?.remove(id);
        Ok(())
    }

    /// Size points: a target size `h` at each of `points`.
    fn add_size_points(&mut self, points: Vec<P3>, h: f64) -> PyResult<()> {
        let g = self.geo()?;
        for p in points {
            g.add_size_point(p, h);
        }
        Ok(())
    }

    /// The live objects, by id.
    fn objects(&self) -> Vec<ObjId> {
        (0..self.inner.objects().len()).filter(|&i| self.inner.object(i).alive).collect()
    }

    fn is_sheet(&self, id: ObjId) -> bool {
        !self.inner.object(id).is_solid()
    }

    fn is_void(&self, id: ObjId) -> bool {
        self.inner.object(id).void
    }

    /// Face selection keys bounding an object.
    fn faces_of(&self, id: ObjId) -> PyResult<Vec<PySel>> {
        Ok(self.inner.faces_of(id).map_err(err)?.iter().map(FaceSel::key).collect())
    }

    /// Edges of a solid object: `((role_a, role_b), midpoint)`.
    fn edges_of(&self, id: ObjId) -> PyResult<Vec<((u32, u32), P3)>> {
        Ok(self.inner.edges_of(id).map_err(err)?.into_iter().map(|(r, m)| ((r[0], r[1]), m)).collect())
    }

    /// See `Geometry::hollow`: the wall keys and the thickness `2V/S`, or
    /// None when no solid is named `name`.
    fn hollow(&mut self, name: &str) -> PyResult<Option<(Vec<PySel>, f64)>> {
        let walls = self.geo()?.hollow(name).map_err(|e| PyValueError::new_err(format!("_hollow: {e}")))?;
        Ok(walls.map(|(w, t)| (w.iter().map(FaceSel::key).collect(), t)))
    }

    /// See `Geometry::auto_refine`.
    #[pyo3(signature = (base_maxh, resolution=3.0, min_maxh=None))]
    fn auto_refine(&mut self, base_maxh: f64, resolution: f64, min_maxh: Option<f64>) -> PyResult<Vec<(String, f64)>> {
        self.geo()?.auto_refine(base_maxh, resolution, min_maxh).map_err(err)
    }

    // ── attributes of handles ───────────────────────────────────────────

    /// The name of an object or a face (a sheet's own selection is the
    /// sheet object).
    fn name(&self, key: Key) -> Option<String> {
        match key {
            Key::Object(o) => self.inner.object(o).name.clone(),
            Key::Face(o, r, s, a) => {
                let sel = FaceSel::from_key((o, r, s, a));
                if sel == FaceSel::sheet(o) {
                    self.inner.object(o).name.clone()
                } else {
                    self.inner.face_name(&sel).map(str::to_string)
                }
            }
            Key::Edge(..) | Key::Group(_) => None,
        }
    }

    #[pyo3(signature = (key, name=None))]
    fn set_name(&mut self, key: Key, name: Option<String>) -> PyResult<()> {
        match key {
            Key::Object(o) => self.inner.set_name(o, name),
            Key::Face(o, r, s, a) => {
                let sel = FaceSel::from_key((o, r, s, a));
                if sel == FaceSel::sheet(o) {
                    self.inner.set_name(o, name)
                } else {
                    self.inner.set_face_name(sel, name)
                }
            }
            Key::Edge(..) | Key::Group(_) => {
                return Err(PyValueError::new_err("edges and the groups of a loaded mesh take no name"))
            }
        }
        Ok(())
    }

    /// The size set on an object (its own, not its material's) or a face; a
    /// sheet seen as a face is the sheet object.
    fn maxh(&self, key: Key) -> Option<f64> {
        match key {
            Key::Object(o) | Key::Face(o, -1, _, _) => self.inner.object(o).maxh,
            Key::Face(o, r, s, a) => self.inner.face_maxh(&FaceSel::from_key((o, r, s, a))),
            Key::Edge(..) | Key::Group(_) => None,
        }
    }

    /// Target size `h` on objects and faces (a sheet seen as a face sizes
    /// the sheet; `None` clears an object's own size); edges and the groups
    /// of a loaded mesh (which is not remeshed) are skipped.
    #[pyo3(signature = (keys, h=None))]
    fn set_maxh(&mut self, keys: Vec<Key>, h: Option<f64>) {
        let mut faces = Vec::new();
        for k in keys {
            match k {
                Key::Object(o) => self.inner.set_object_maxh(o, h),
                Key::Face(o, r, _, _) if r < 0 => self.inner.set_object_maxh(o, h),
                Key::Face(o, r, s, a) => faces.push(FaceSel::from_key((o, r, s, a))),
                Key::Edge(..) | Key::Group(_) => {}
            }
        }
        if let (Some(h), false) = (h, faces.is_empty()) {
            self.inner.set_face_maxh(faces, h);
        }
    }

    /// The material object filling a solid or a volume group (None for
    /// faces and edges).
    fn material(&self, py: Python<'_>, key: Key) -> PyResult<Option<Py<PyAny>>> {
        if let Key::Edge(..) = key {
            return Ok(None);
        }
        let target = self.target(key)?;
        Ok(self.setup.material(&target).map(|m| self.materials[m].clone_ref(py)))
    }

    /// Fills solids and volume groups with `material` (None empties them),
    /// one material tag per material object; faces and edges take none. A
    /// solid without a size of its own is meshed at the material's `maxh`.
    #[pyo3(signature = (keys, material=None))]
    fn set_material(&mut self, keys: Vec<Key>, material: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let (index, maxh) = match &material {
            None => (None, None),
            Some(mat) => {
                let index = match self.materials.iter().position(|m| m.as_ptr() == mat.as_ptr()) {
                    Some(i) => i,
                    None => {
                        let kind: String = mat.getattr("__class__")?.getattr("__name__")?.extract()?;
                        self.materials.push(mat.clone().unbind());
                        self.setup.add_material(kind.to_lowercase())
                    }
                };
                (Some(index), mat.getattr("maxh")?.extract::<Option<f64>>()?)
            }
        };
        for k in keys {
            if let Key::Edge(..) = k {
                continue;
            }
            let target = self.target(k)?;
            if let Target::Object(o) = target {
                self.inner.set_material_maxh(o, maxh);
            }
            self.setup.set_material(target, index);
        }
        Ok(())
    }

    // ── selection ───────────────────────────────────────────────────────

    /// Where each handle lies: `(centroid, area, bbox)`.
    fn extents(&self, keys: Vec<Key>) -> PyResult<Vec<(P3, f64, [f64; 6])>> {
        keys.iter().map(|k| self.extent(k).map(|e| (e.centroid, e.area, e.bbox))).collect()
    }

    /// The positions of the handles whose centroid lies at the minimum (or
    /// the maximum) along `axis` ("x", "y" or "z").
    fn select_extreme(&self, keys: Vec<Key>, axis: &str, max: bool) -> PyResult<Vec<usize>> {
        let axis = match axis.to_ascii_lowercase().as_str() {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            _ => return Err(PyValueError::new_err(format!("axis must be 'x', 'y' or 'z', got '{axis}'"))),
        };
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let items = self.extents_of(&keys)?;
        let tol = SELECT_TOL * Extent::size(&self.model_box()?);
        Ok(Extent::extreme(&items, axis, max, tol))
    }

    /// The positions of the handles lying flat on the bounding box of the
    /// whole model, or with `own` on the box around the handles themselves.
    fn select_on_box(&self, keys: Vec<Key>, own: bool) -> PyResult<Vec<usize>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let items = self.extents_of(&keys)?;
        let model = self.model_box()?;
        let b = if own { Extent::union(&items) } else { model };
        Ok(Extent::on_box(&items, &b, SELECT_TOL * Extent::size(&model)))
    }

    /// The positions of the handles no physics object sits on.
    fn select_unassigned(&self, keys: Vec<Key>) -> PyResult<Vec<usize>> {
        let mut out = Vec::new();
        for (i, k) in keys.into_iter().enumerate() {
            let free = match k {
                Key::Edge(..) => true,
                k => !self.setup.is_assigned(&self.target(k)?),
            };
            if free {
                out.push(i);
            }
        }
        Ok(out)
    }

    /// The B-rep face ids the selections currently resolve to.
    fn face_ids(&self, keys: Vec<PySel>) -> PyResult<Vec<u32>> {
        let sels: Vec<FaceSel> = keys.into_iter().map(FaceSel::from_key).collect();
        Ok(self.inner.resolve(&sels).map_err(err)?.iter().map(|f| f.id).collect())
    }

    /// Bounding box `(xmin, ymin, zmin, xmax, ymax, zmax)` of the scene.
    fn bbox(&self) -> PyResult<[f64; 6]> {
        self.model_box()
    }

    /// Coarse per-face triangulation for previews: `(face id, corners,
    /// normals)`, nine values per triangle each.
    #[pyo3(signature = (target_triangles=4000))]
    fn preview(&self, target_triangles: usize) -> PyResult<Vec<(u32, Vec<f64>, Vec<f64>)>> {
        Ok(self.inner.preview(target_triangles).map_err(err)?.into_iter().map(|(id, (p, n))| (id, p, n)).collect())
    }

    // ── physics ─────────────────────────────────────────────────────────
    //
    // Each returns the index of the physics object; its tag follows from
    // the setup (`physics_tag`).

    fn add_pec(&mut self, targets: Vec<Key>) -> PyResult<usize> {
        self.add(Condition::Pec, targets)
    }

    fn add_pmc(&mut self, targets: Vec<Key>) -> PyResult<usize> {
        self.add(Condition::Face(FaceSpec::Pmc { tag: 0 }), targets)
    }

    fn add_abc(&mut self, targets: Vec<Key>) -> PyResult<usize> {
        self.add(Condition::Face(FaceSpec::Abc { tag: 0 }), targets)
    }

    fn add_far_field(&mut self, targets: Vec<Key>) -> PyResult<usize> {
        self.add(Condition::FarField, targets)
    }

    fn add_periodic(&mut self, a: Vec<Key>, b: Vec<Key>) -> PyResult<usize> {
        let mut p = Physics::new(Condition::Periodic, self.targets(a)?);
        p.pair = self.targets(b)?;
        Ok(self.setup.add(p))
    }

    #[pyo3(signature = (targets, *, width, height, mode, er, power))]
    fn add_rect_port(&mut self, targets: Vec<Key>, width: f64, height: f64, mode: [usize; 2], er: f64, power: f64) -> PyResult<usize> {
        self.add(Condition::Face(FaceSpec::Rectangular { tag: 0, width, height, mode, er, power }), targets)
    }

    #[pyo3(signature = (targets, *, scan_theta_deg, scan_phi_deg, mode_nr, power))]
    fn add_floquet_port(&mut self, targets: Vec<Key>, scan_theta_deg: f64, scan_phi_deg: f64, mode_nr: u32, power: f64) -> PyResult<usize> {
        self.add(Condition::Face(FaceSpec::Floquet { tag: 0, scan_theta_deg, scan_phi_deg, mode_nr, power }), targets)
    }

    #[pyo3(signature = (targets, *, e_field, power))]
    fn add_user_port(&mut self, targets: Vec<Key>, e_field: P3, power: f64) -> PyResult<usize> {
        self.add(Condition::Face(FaceSpec::UserDefined { tag: 0, e_field, power }), targets)
    }

    #[pyo3(signature = (targets, *, ri, ro, er, power, origin=None, z_axis=None))]
    fn add_coax_port(
        &mut self,
        targets: Vec<Key>,
        ri: f64,
        ro: f64,
        er: f64,
        power: f64,
        origin: Option<P3>,
        z_axis: Option<P3>,
    ) -> PyResult<usize> {
        self.add(Condition::Face(FaceSpec::Coax { tag: 0, ri, ro, origin, z_axis, er, power }), targets)
    }

    #[pyo3(signature = (targets, *, z0, l, direction, width, height, power, c=None))]
    fn add_lumped_port(
        &mut self,
        targets: Vec<Key>,
        z0: f64,
        l: f64,
        direction: P3,
        width: f64,
        height: f64,
        power: f64,
        c: Option<f64>,
    ) -> PyResult<usize> {
        self.add(Condition::Face(FaceSpec::Lumped { tag: 0, z0, l, c, direction, width, height, power }), targets)
    }

    #[pyo3(signature = (targets, *, r, l, direction, width, height, c=None))]
    fn add_lumped_element(
        &mut self,
        targets: Vec<Key>,
        r: f64,
        l: f64,
        direction: P3,
        width: f64,
        height: f64,
        c: Option<f64>,
    ) -> PyResult<usize> {
        self.add(Condition::Face(FaceSpec::LumpedElement { tag: 0, r, l, c, width, height, direction }), targets)
    }

    /// A numerical wave port: `mode_kind` "te" / "tm" picks the scalar
    /// solve, "auto" the vector solve at `f0` (without `f0` the scalar TE
    /// solve); `pec` are the physics objects (indices) marking internal
    /// conductors of the cross-section.
    #[pyo3(signature = (targets, *, mode_kind, mode_index, power, f0=None, pec=Vec::new()))]
    fn add_wave_port(
        &mut self,
        targets: Vec<Key>,
        mode_kind: &str,
        mode_index: usize,
        power: f64,
        f0: Option<f64>,
        pec: Vec<usize>,
    ) -> PyResult<usize> {
        let kind = WaveKind::pick(mode_kind, f0).map_err(PyValueError::new_err)?;
        let spec = FaceSpec::WaveNumerical { tag: 0, f0, mode_index, kind, pec_tags: Vec::new(), power };
        let mut p = Physics::new(Condition::Face(spec), self.targets(targets)?);
        p.pec = pec;
        Ok(self.setup.add(p))
    }

    /// A surface impedance; refused on the complete shell of a solid that
    /// is still meshed (there it would sit inside the mesh).
    #[pyo3(signature = (targets, *, conductivity, mur, er, thickness=None, two_sided=false, sheet=false, zs=None))]
    fn add_surface_impedance(
        &mut self,
        targets: Vec<Key>,
        conductivity: f64,
        mur: f64,
        er: f64,
        thickness: Option<f64>,
        two_sided: bool,
        sheet: bool,
        zs: Option<[f64; 2]>,
    ) -> PyResult<usize> {
        if sheet && two_sided {
            return Err(PyValueError::new_err("SurfaceImpedance: sheet and two_sided exclude each other"));
        }
        let targets = self.targets(targets)?;
        let faces: Vec<FaceSel> = targets
            .iter()
            .filter_map(|t| match t {
                Target::Face(s) => Some(*s),
                _ => None,
            })
            .collect();
        if self.inner.covers_solid_shell(&faces) {
            return Err(PyValueError::new_err(SOLID_SHELL));
        }
        let spec = FaceSpec::SurfaceImpedance { tag: 0, conductivity, mur, er, thickness, two_sided, sheet, zs };
        Ok(self.setup.add(Physics::new(Condition::Face(spec), targets)))
    }

    #[pyo3(signature = (targets, *, direction, inner_face, thickness, er_base, ur_base, exponent, delta_max))]
    fn add_pml(
        &mut self,
        targets: Vec<Key>,
        direction: P3,
        inner_face: f64,
        thickness: f64,
        er_base: f64,
        ur_base: f64,
        exponent: f64,
        delta_max: f64,
    ) -> PyResult<usize> {
        let spec = PmlSpec { volume_tag: 0, direction, inner_face, thickness, er_base, ur_base, exponent, delta_max };
        self.add(Condition::Pml(spec), targets)
    }

    /// The mesh tag of physics object `index` (of a periodic pair side a,
    /// side b has the next tag).
    fn physics_tag(&self, index: usize) -> PyResult<i32> {
        self.setup
            .tagging()
            .physics
            .get(index)
            .copied()
            .ok_or_else(|| PyIndexError::new_err(format!("no physics object {index}")))
    }

    /// `(tag, name, dim)` of every group of the last `mesh()` (before one,
    /// of the current setup), dim 3 for a volume group, 2 for faces.
    fn group_names(&self) -> Vec<(i32, String, u8)> {
        let current;
        let t = match &self.meshed {
            Some((_, t)) => t,
            None => {
                current = self.setup.tagging();
                &current
            }
        };
        let volumes: BTreeSet<i32> = t.volumes.iter().map(|(tag, _)| *tag).collect();
        t.names
            .iter()
            .map(|(&tag, name)| (tag, name.clone(), if volumes.contains(&tag) { 3 } else { 2 }))
            .collect()
    }

    // ── mesh ────────────────────────────────────────────────────────────

    /// Loads a pre-built `.msh` volume mesh as the whole geometry (mesh
    /// mode); returns `(name, dim)` of its named groups in file order.
    fn load_msh(&mut self, path: String) -> PyResult<Vec<(String, u8)>> {
        if self.scene.is_some() || !self.inner.objects().is_empty() {
            return Err(PyRuntimeError::new_err(
                "load(.msh): a loaded mesh is the whole geometry; load it into a fresh Geometry()",
            ));
        }
        let scene = MeshScene::load(&path).map_err(PyValueError::new_err)?;
        let groups = scene.groups.iter().map(|g| (g.name.clone(), g.dim)).collect();
        self.scene = Some(scene);
        Ok(groups)
    }

    /// The factor on every target size (see `Geometry::set_size_scale`).
    #[getter]
    fn size_scale(&self) -> f64 {
        self.inner.size_scale()
    }

    #[setter]
    fn set_size_scale(&mut self, scale: f64) -> PyResult<()> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(PyValueError::new_err(format!("size_scale must be a positive number, got {scale}")));
        }
        self.inner.set_size_scale(scale);
        Ok(())
    }

    #[getter]
    fn mesh_mode(&self) -> bool {
        self.scene.is_some()
    }

    /// Meshes the scene (a loaded mesh is taken as it is), tags every
    /// material and physics object and returns the solver mesh with its
    /// stats. `maxh` and `cells_across` default to `mesh_maxh` and
    /// `cells_across`; without grading the size jumps.
    #[pyo3(signature = (maxh=None, cells_across=None, target_elements=None))]
    fn mesh(&mut self, maxh: Option<f64>, cells_across: Option<f64>, target_elements: Option<usize>) -> PyResult<(PyFemMesh, PyMeshStats)> {
        let tagging = self.setup.tagging();
        let (fm, quality) = match &self.scene {
            Some(scene) => {
                if tagging.is_empty() {
                    return Err(PyRuntimeError::new_err(
                        "mesh(): no materials or physics are bound to the loaded mesh's groups",
                    ));
                }
                let names = |groups: &[(i32, Vec<Target>)]| -> Vec<(i32, Vec<String>)> {
                    groups
                        .iter()
                        .map(|(tag, ts)| {
                            let names = ts
                                .iter()
                                .filter_map(|t| match t {
                                    Target::Group { name, .. } => Some(name.clone()),
                                    _ => None,
                                })
                                .collect();
                            (*tag, names)
                        })
                        .collect()
                };
                let fm = scene.fem_mesh(&names(&tagging.faces), &names(&tagging.volumes)).map_err(PyValueError::new_err)?;
                (fm, (scene.mesh.quality.min_dihedral_deg, scene.mesh.quality.slivers.len()))
            }
            None => {
                let opts = MeshOptions {
                    maxh: maxh.or(self.mesh_maxh),
                    grading: if self.grading { None } else { Some(1e9) },
                    cells_across: cells_across.or(self.cells_across),
                    target_elements,
                    ..Default::default()
                };
                let m = self.inner.mesh(&opts).map_err(err)?;
                let (faces, volumes) = self.brep_groups(&tagging)?;
                let holes = self.inner.hole_regions().map_err(err)?;
                let fm = fem_mesh(&m, &faces, &volumes, &holes);
                let quality = (m.quality.min_dihedral_deg, m.quality.slivers.len());
                self.mesh = Some(m);
                (fm, quality)
            }
        };
        let stats = MeshStats::new(&fm, quality.0, quality.1, &tagging.names);
        self.meshed = Some((self.setup.revision(), tagging));
        Ok((PyFemMesh { inner: fm }, PyMeshStats { inner: stats }))
    }

    /// The model: every material and physics object under its tag, the
    /// tags of the last `mesh()` (refused when the materials or physics
    /// changed since; before a mesh, the tags it will give).
    fn model(&self, py: Python<'_>) -> PyResult<PyModel> {
        let current;
        let tagging = match &self.meshed {
            Some((revision, t)) if *revision == self.setup.revision() => t,
            Some(_) => {
                return Err(PyRuntimeError::new_err(
                    "materials or physics changed after g.mesh(), re-run g.mesh() before solving",
                ))
            }
            None => {
                current = self.setup.tagging();
                &current
            }
        };
        let specs = self.materials.iter().map(|m| material_spec(m.bind(py))).collect::<PyResult<Vec<_>>>()?;
        Ok(PyModel { inner: self.setup.model(tagging, &specs).map_err(PyValueError::new_err)? })
    }

    /// Writes the last `mesh()` (holes left out) as gmsh MSH 4.1, the face
    /// and volume groups its named physical groups; returns the volume
    /// groups that found no region of their own.
    fn save_msh(&self, path: String) -> PyResult<Vec<String>> {
        if self.scene.is_some() {
            return Err(PyRuntimeError::new_err("save_mesh: the geometry is a loaded mesh; save its file instead"));
        }
        let (Some(m), Some((_, tagging))) = (self.mesh.as_ref(), self.meshed.as_ref()) else {
            return Err(PyRuntimeError::new_err("save_mesh: call mesh() first"));
        };
        let (faces, volumes) = self.brep_groups(tagging)?;
        let named = |gs: Vec<Group>| -> Vec<(String, Vec<u32>)> {
            gs.into_iter()
                .map(|g| (tagging.names.get(&g.tag).cloned().unwrap_or(format!("group_{}", g.tag)), g.ids))
                .collect()
        };
        let holes = self.inner.hole_regions().map_err(err)?;
        let io = |e: std::io::Error| PyRuntimeError::new_err(format!("{path}: {e}"));
        let mut w = std::io::BufWriter::new(std::fs::File::create(&path).map_err(io)?);
        let dropped = write_msh(m, &named(volumes), &named(faces), &holes, &mut w).map_err(io)?;
        std::io::Write::flush(&mut w).map_err(io)?;
        Ok(dropped)
    }
}

impl PyGeometry {
    pub(crate) fn fresh(maxh: Option<f64>, grading: bool) -> Self {
        PyGeometry {
            inner: Geometry::new(maxh),
            setup: Setup::default(),
            materials: Vec::new(),
            scene: None,
            grading,
            mesh_maxh: maxh,
            cells_across: None,
            mesh: None,
            meshed: None,
        }
    }

    /// Runs a native builder on the scene; the materials it added become
    /// `rapidfem.materials` objects.
    pub(crate) fn build<R>(&mut self, py: Python<'_>, f: impl FnOnce(&mut Scene) -> Result<R, String>) -> PyResult<R> {
        let mut scene = Scene::new(&mut self.inner, &mut self.setup);
        let out = f(&mut scene).map_err(PyValueError::new_err)?;
        let added = std::mem::take(&mut scene.materials);
        let module = py.import("rapidfem.materials")?;
        for (index, m) in added {
            let kwargs = PyDict::new(py);
            kwargs.set_item("maxh", m.maxh)?;
            let object = if m.air {
                module.getattr("Air")?.call((), Some(&kwargs))?
            } else {
                kwargs.set_item("tand", m.tand)?;
                kwargs.set_item("conductivity", m.conductivity)?;
                kwargs.set_item("cond_diag", m.cond_diag)?;
                module.getattr("Dielectric")?.call((m.er,), Some(&kwargs))?
            };
            if index != self.materials.len() {
                return Err(PyRuntimeError::new_err("a native builder added materials out of order"));
            }
            self.materials.push(object.unbind());
        }
        Ok(out)
    }

    /// The scene, for a change; refused in mesh mode.
    fn geo(&mut self) -> PyResult<&mut Geometry> {
        match self.scene {
            Some(_) => Err(PyRuntimeError::new_err(MESH_MODE)),
            None => Ok(&mut self.inner),
        }
    }

    fn target(&self, key: Key) -> PyResult<Target> {
        match key {
            Key::Face(o, r, s, a) => Ok(Target::Face(FaceSel::from_key((o, r, s, a)))),
            Key::Object(o) => Ok(Target::Object(o)),
            Key::Group(name) => {
                let scene = self.scene.as_ref().ok_or_else(|| PyKeyError::new_err(format!("no loaded mesh for group '{name}'")))?;
                let index = scene
                    .groups
                    .iter()
                    .position(|g| g.name == name)
                    .ok_or_else(|| PyKeyError::new_err(format!("no group '{name}' in the loaded mesh")))?;
                Ok(Target::Group { index, dim: scene.groups[index].dim, name })
            }
            Key::Edge(o, (a, b)) => Err(PyValueError::new_err(format!("the edge ({a}, {b}) of object {o} takes no material or physics"))),
        }
    }

    fn targets(&self, keys: Vec<Key>) -> PyResult<Vec<Target>> {
        keys.into_iter().map(|k| self.target(k)).collect()
    }

    fn add(&mut self, condition: Condition, targets: Vec<Key>) -> PyResult<usize> {
        let targets = self.targets(targets)?;
        Ok(self.setup.add(Physics::new(condition, targets)))
    }

    fn extent(&self, key: &Key) -> PyResult<Extent> {
        let point = |m: P3| Extent { centroid: m, area: 0.0, bbox: [m[0], m[1], m[2], m[0], m[1], m[2]] };
        match key {
            Key::Face(o, r, s, a) => self.inner.face_extent(&FaceSel::from_key((*o, *r, *s, *a))).map_err(err),
            Key::Object(o) => self.inner.object_extent(*o).map_err(err),
            Key::Edge(o, roles) => {
                let edges = self.inner.edges_of(*o).map_err(err)?;
                let (_, mid) = edges
                    .iter()
                    .find(|(r, _)| (r[0], r[1]) == *roles)
                    .ok_or_else(|| PyValueError::new_err(format!("object {o} has no edge {roles:?}")))?;
                Ok(point(*mid))
            }
            Key::Group(name) => {
                let scene = self.scene.as_ref().ok_or_else(|| PyKeyError::new_err(format!("no loaded mesh for group '{name}'")))?;
                let b = scene.bbox(name).map_err(PyKeyError::new_err)?;
                Ok(Extent { centroid: [0, 1, 2].map(|k| (b[k] + b[k + 3]) / 2.0), area: 0.0, bbox: b })
            }
        }
    }

    fn extents_of(&self, keys: &[Key]) -> PyResult<Vec<Extent>> {
        keys.iter().map(|k| self.extent(k)).collect()
    }

    /// The bounding box of the scene, or of the loaded mesh.
    fn model_box(&self) -> PyResult<[f64; 6]> {
        match &self.scene {
            Some(s) => {
                let mut b = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                for p in &s.mesh.points {
                    for k in 0..3 {
                        b[k] = b[k].min(p[k]);
                        b[k + 3] = b[k + 3].max(p[k]);
                    }
                }
                Ok(b)
            }
            None => self.inner.bbox().map_err(err),
        }
    }

    /// The B-rep faces and regions of the face and volume groups of `t`.
    fn brep_groups(&self, t: &Tagging) -> PyResult<(Vec<Group>, Vec<Group>)> {
        let faces: Vec<(i32, Vec<FaceSel>)> = t
            .faces
            .iter()
            .map(|(tag, ts)| {
                let sels = ts
                    .iter()
                    .filter_map(|x| match x {
                        Target::Face(s) => Some(*s),
                        _ => None,
                    })
                    .collect();
                (*tag, sels)
            })
            .collect();
        let volumes: Vec<(i32, Vec<ObjId>)> = t
            .volumes
            .iter()
            .map(|(tag, ts)| {
                let objects = ts
                    .iter()
                    .filter_map(|x| match x {
                        Target::Object(o) => Some(*o),
                        _ => None,
                    })
                    .collect();
                (*tag, objects)
            })
            .collect();
        let materials: Vec<i32> = t.materials.iter().flatten().copied().collect();
        self.inner.groups(&faces, &volumes, &materials).map_err(err)
    }
}

/// Size and quality report of the last generated mesh.
///
/// Stored on `geometry.mesh_stats` by `Geometry.mesh()`. The DOF numbers
/// bound the FD solver's Nedelec space: `dofs_min` is the uniform order-1
/// count (one DOF per edge), `dofs_max` the uniform order-2 count (two per
/// edge plus two per triangular face), the default of `ProblemFD.sweep`;
/// `order="adaptive"` lands in between. The bounds gate RAM and runtime
/// before any assembly. `quality_min` is the smallest dihedral angle in
/// degrees (the sliver indicator that governs the conditioning),
/// `n_slivers` the number of tets below the sliver threshold, `groups` the
/// element count per group name.
#[pyclass(name = "MeshStats", module = "rapidfem._native")]
pub struct PyMeshStats {
    inner: MeshStats,
}

#[pymethods]
impl PyMeshStats {
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes
    }

    #[getter]
    fn n_tets(&self) -> usize {
        self.inner.n_tets
    }

    /// Unique tet faces, interior and boundary.
    #[getter]
    fn n_tris(&self) -> usize {
        self.inner.n_tris
    }

    /// Unique tet edges.
    #[getter]
    fn n_edges(&self) -> usize {
        self.inner.n_edges
    }

    /// DOFs of uniform order 1: one per edge.
    #[getter]
    fn dofs_min(&self) -> usize {
        self.inner.dofs_min
    }

    /// DOFs of uniform order 2 (the default of `ProblemFD.sweep`): two per
    /// edge plus two per triangle.
    #[getter]
    fn dofs_max(&self) -> usize {
        self.inner.dofs_max
    }

    /// Smallest dihedral angle in degrees.
    #[getter]
    fn quality_min(&self) -> f64 {
        self.inner.quality_min
    }

    #[getter]
    fn n_slivers(&self) -> usize {
        self.inner.n_slivers
    }

    /// Group name -> element count.
    #[getter]
    fn groups<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, n) in &self.inner.groups {
            d.set_item(name, n)?;
        }
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let s = &self.inner;
        format!(
            "MeshStats({} nodes, {} tets, {} edges, dofs {}..{}, min dihedral {:.1} deg, {} slivers)",
            s.n_nodes, s.n_tets, s.n_edges, s.dofs_min, s.dofs_max, s.quality_min, s.n_slivers
        )
    }
}

/// The solver mesh (nodes, tets, derived edges and faces, tag groups).
#[pyclass(name = "FemMesh", module = "rapidfem._native", skip_from_py_object)]
#[derive(Clone)]
pub struct PyFemMesh {
    pub inner: rapidfem_core::mesh::Mesh,
}

#[pymethods]
impl PyFemMesh {
    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    #[getter]
    fn n_tets(&self) -> usize {
        self.inner.n_tets()
    }

    #[getter]
    fn n_tris(&self) -> usize {
        self.inner.n_tris()
    }

    #[getter]
    fn n_edges(&self) -> usize {
        self.inner.edges.len()
    }

    /// What a viewer draws: `(nodes, tris, tri_tags, tets, tet_tags)`,
    /// flat; the triangles on the boundary or in a face group, each with
    /// its group's tag (0 for none), every tet with its volume group's tag.
    fn viewer(&self) -> (Vec<f64>, Vec<usize>, Vec<i32>, Vec<usize>, Vec<i32>) {
        let m = &self.inner;
        let v = viewer_mesh(m);
        (
            m.nodes.iter().flatten().copied().collect(),
            v.tris.iter().flatten().copied().collect(),
            v.tri_tags,
            m.tets.iter().flatten().copied().collect(),
            v.tet_tags,
        )
    }
}
