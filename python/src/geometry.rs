// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! `_native.Geometry` and `_native.FemMesh`: the rapidfem geometry on the
//! vendored rapidmesh (`rapidfem_geom`), and the solver mesh it produces.
//!
//! Objects are addressed by their index, faces by their origin
//! `(object, role)` (role -1 for a sheet). The Python `rapidfem.Geometry`
//! is a thin layer over these calls.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rapidfem_geom::fem_mesh::{fem_mesh, viewer_mesh, Group};
use rapidfem_geom::msh::{write_msh, MeshScene};
use rapidfem_geom::path::spline;
use rapidfem_geom::geometry::{Across, EdgeOp, FaceOrigin, FaceSel, Geometry, Item, ObjId};
use rapidmesh::shapes::{Cone, Cuboid, Cylinder, Helix, Import, Loft, Prism, Revolve, Sheet, Sphere, Sweep, Torus, Wedge};
use rapidmesh::{EdgeCut, EdgePick, MeshOptions, Transform};

type P3 = [f64; 3];

/// A face selection from Python: `(object, role, side, across)`, role -1
/// for a sheet, side -1 for none, across -1 for any, -2 for outside.
type PySel = (usize, i64, i64, i64);

fn sel_of((object, role, side, across): PySel) -> FaceSel {
    let origin = if role < 0 {
        FaceOrigin::Sheet { object }
    } else {
        FaceOrigin::Solid { object, role: role as u32 }
    };
    let across = match across {
        -1 => None,
        -2 => Some(Across::Outside),
        o => Some(Across::Object(o as usize)),
    };
    FaceSel { origin, side: (side >= 0).then_some(side as usize), across }
}

fn sel_tuple(s: FaceSel) -> PySel {
    let side = s.side.map_or(-1, |o| o as i64);
    let across = match s.across {
        None => -1,
        Some(Across::Outside) => -2,
        Some(Across::Object(o)) => o as i64,
    };
    match s.origin {
        FaceOrigin::Solid { object, role } => (object, role as i64, side, across),
        FaceOrigin::Sheet { object } => (object, -1, side, across),
    }
}

fn err(e: String) -> PyErr {
    PyRuntimeError::new_err(e)
}

/// The rapidfem geometry: solids and sheets on rapidmesh.
#[pyclass(name = "Geometry", module = "rapidfem._native", unsendable)]
pub struct PyGeometry {
    inner: Geometry,
    mesh: Option<rapidmesh::Mesh>,
}

#[pymethods]
impl PyGeometry {
    #[new]
    #[pyo3(signature = (maxh=None))]
    fn new(maxh: Option<f64>) -> Self {
        PyGeometry { inner: Geometry::new(maxh), mesh: None }
    }

    #[getter]
    fn maxh(&self) -> Option<f64> {
        self.inner.maxh()
    }

    #[setter]
    fn set_maxh(&mut self, h: Option<f64>) {
        self.inner.set_maxh(h);
    }

    #[pyo3(signature = (size, position=[0.0; 3], maxh=None))]
    fn add_box(&mut self, size: P3, position: P3, maxh: Option<f64>) -> ObjId {
        self.inner.add_solid(Cuboid::new(size).at(position), maxh, false)
    }

    #[pyo3(signature = (radius, height, position=[0.0; 3], axis=[0.0, 0.0, 1.0], maxh=None))]
    fn add_cylinder(&mut self, radius: f64, height: f64, position: P3, axis: P3, maxh: Option<f64>) -> ObjId {
        self.inner.add_solid(Cylinder::new(radius, height).at(position).along(axis), maxh, false)
    }

    #[pyo3(signature = (radius, position=[0.0; 3], maxh=None))]
    fn add_sphere(&mut self, radius: f64, position: P3, maxh: Option<f64>) -> ObjId {
        self.inner.add_solid(Sphere::new(radius).at(position), maxh, false)
    }

    #[pyo3(signature = (r1, r2, height, position=[0.0; 3], axis=[0.0, 0.0, 1.0], maxh=None))]
    fn add_cone(&mut self, r1: f64, r2: f64, height: f64, position: P3, axis: P3, maxh: Option<f64>) -> ObjId {
        self.inner.add_solid(Cone::new(r1, r2, height).at(position).along(axis), maxh, false)
    }

    #[pyo3(signature = (size, top_x, position=[0.0; 3], maxh=None))]
    fn add_wedge(&mut self, size: P3, top_x: f64, position: P3, maxh: Option<f64>) -> ObjId {
        let mut w = Wedge::new(size).at(position);
        w.top_x = top_x;
        self.inner.add_solid(w, maxh, false)
    }

    #[pyo3(signature = (major_radius, minor_radius, position=[0.0; 3], axis=[0.0, 0.0, 1.0], maxh=None))]
    fn add_torus(&mut self, major_radius: f64, minor_radius: f64, position: P3, axis: P3, maxh: Option<f64>) -> ObjId {
        let mut t = Torus::new(major_radius, minor_radius).at(position);
        t.axis = axis;
        self.inner.add_solid(t, maxh, false)
    }

    /// The parallelogram from `corner` spanned by `u` and `v`.
    #[pyo3(signature = (corner, u, v, maxh=None))]
    fn add_plate(&mut self, corner: P3, u: P3, v: P3, maxh: Option<f64>) -> ObjId {
        self.inner.add_sheet(Sheet::plate(corner, u, v), maxh)
    }

    #[pyo3(signature = (radius, center, axis=[0.0, 0.0, 1.0], maxh=None))]
    fn add_disc(&mut self, radius: f64, center: P3, axis: P3, maxh: Option<f64>) -> ObjId {
        self.inner.add_sheet(Sheet::disc(radius, center, axis), maxh)
    }

    /// A polygon (with holes) in the xy plane at height `z`.
    #[pyo3(signature = (points, z, holes=Vec::new(), maxh=None))]
    fn add_polygon(&mut self, points: Vec<[f64; 2]>, z: f64, holes: Vec<Vec<[f64; 2]>>, maxh: Option<f64>) -> ObjId {
        self.inner.add_sheet(Sheet::Polygon { points, holes, position: [0.0, 0.0, z] }, maxh)
    }

    /// Extrudes a sheet along `vector` into a solid, in place: the object
    /// keeps its id and the sheet becomes its bottom face (role 0, top 1,
    /// then the walls). An xy polygon or rectangle swept along z becomes a
    /// prism.
    fn extrude(&mut self, id: ObjId, vector: P3) -> PyResult<()> {
        let [vx, vy, h] = vector;
        let o = self.inner.object(id);
        let prism = match &o.item {
            _ if vx != 0.0 || vy != 0.0 || !o.transforms.is_empty() => None,
            Item::Sheet(Sheet::Polygon { points, holes, position }) => Some(Prism {
                points: points.clone(),
                holes: holes.clone(),
                height: h,
                position: *position,
            }),
            Item::Sheet(Sheet::Rect { corner, u, v }) if u[2] == 0.0 && v[2] == 0.0 => Some(Prism {
                points: vec![[0.0, 0.0], [u[0], u[1]], [u[0] + v[0], u[1] + v[1]], [v[0], v[1]]],
                holes: Vec::new(),
                height: h,
                position: *corner,
            }),
            _ => None,
        };
        match prism {
            // a negative height extrudes downwards: the same prism, shifted
            Some(p) if h < 0.0 => {
                let q = p.position;
                let p = Prism { height: -h, position: [q[0], q[1], q[2] + h], ..p };
                self.inner.replace(id, Item::Solid(p.into()));
                Ok(())
            }
            Some(p) => {
                self.inner.replace(id, Item::Solid(p.into()));
                Ok(())
            }
            None => self.inner.extrude(id, vector).map_err(PyValueError::new_err),
        }
    }

    /// Ruled loft between two planar profiles with the same vertex count.
    #[pyo3(signature = (profile_a, profile_b, maxh=None))]
    fn add_loft(&mut self, profile_a: Vec<P3>, profile_b: Vec<P3>, maxh: Option<f64>) -> ObjId {
        self.inner.add_solid(Loft { profile_a, profile_b }, maxh, false)
    }

    /// Solid of revolution of the closed profile `points` (`(r, z)` in the
    /// frame of the axis through `position` along `axis`), by `angle`
    /// degrees.
    #[pyo3(signature = (points, position, axis, angle=360.0, maxh=None))]
    fn add_revolve(&mut self, points: Vec<[f64; 2]>, position: P3, axis: P3, angle: f64, maxh: Option<f64>) -> ObjId {
        let mut r = Revolve::new(points);
        r.position = position;
        r.axis = axis;
        r.angle = angle;
        self.inner.add_solid(r, maxh, false)
    }

    /// A round tube of `radius` along the Catmull-Rom spline through
    /// `points` (`samples` points per span; a straight tube for two
    /// points), its cross-section a `segments`-gon.
    #[pyo3(signature = (points, radius, samples=8, segments=16, maxh=None))]
    fn add_sweep(&mut self, points: Vec<P3>, radius: f64, samples: usize, segments: usize, maxh: Option<f64>) -> ObjId {
        let mut s = Sweep::new(spline(&points, samples), radius);
        s.segments = segments;
        self.inner.add_solid(s, maxh, false)
    }

    /// A helical coil of round wire about +z through `position`.
    #[pyo3(signature = (radius, pitch, turns, wire_radius, position=[0.0; 3], points_per_turn=24, segments=12, maxh=None))]
    #[allow(clippy::too_many_arguments)]
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
    ) -> ObjId {
        let mut h = Helix::new(radius, pitch, turns, wire_radius);
        h.position = position;
        h.points_per_turn = points_per_turn;
        h.segments = segments;
        self.inner.add_solid(h, maxh, false)
    }

    /// `(radius, center, axis)` of an untransformed disc sheet.
    fn disc_of(&self, id: ObjId) -> Option<(f64, P3, P3)> {
        let o = self.inner.object(id);
        match &o.item {
            Item::Sheet(Sheet::Disc { radius, center, axis, .. }) if o.transforms.is_empty() => {
                Some((*radius, *center, *axis))
            }
            _ => None,
        }
    }

    /// Chamfers (`fillet=false`, `size` = distance) or fillets (`size` =
    /// radius) edges of a solid, picked by the roles of the two faces
    /// meeting there; `None` cuts every edge.
    #[pyo3(signature = (id, size, fillet, edges=None, void=false))]
    fn cut_edges(&mut self, id: ObjId, size: f64, fillet: bool, edges: Option<Vec<(u32, u32)>>, void: bool) {
        let edges = match edges {
            None => vec![EdgePick::All],
            Some(e) => e.into_iter().map(|(a, b)| EdgePick::Between(a, b)).collect(),
        };
        let cut = if fillet { EdgeCut::Fillet(size) } else { EdgeCut::Chamfer(size) };
        self.inner.cut_edges(EdgeOp { object: id, edges, cut, void });
    }

    /// Edges of a solid object: `((role_a, role_b), midpoint)`.
    fn edges_of(&self, id: ObjId) -> PyResult<Vec<((u32, u32), P3)>> {
        Ok(self.inner.edges_of(id).map_err(err)?.into_iter().map(|(r, m)| ((r[0], r[1]), m)).collect())
    }

    /// Volume of a solid object's own shape (before other solids carve it).
    fn volume(&self, id: ObjId) -> PyResult<f64> {
        self.inner.solid_volume(id).map_err(PyValueError::new_err)
    }

    fn translate(&mut self, id: ObjId, d: P3) {
        self.inner.transform(id, Transform::Translate(d));
    }

    /// Turns an object by `angle` radians about the axis along `axis`
    /// through `center` (right-handed).
    fn rotate(&mut self, id: ObjId, angle: f64, axis: P3, center: P3) {
        self.inner.transform(id, Transform::Rotate { angle, axis, center });
    }

    /// Mirrors an object across the plane through `point` with `normal`.
    fn mirror(&mut self, id: ObjId, normal: P3, point: P3) {
        self.inner.transform(id, Transform::Mirror { normal, point });
    }

    /// Stretches an object by `factors` along x, y and z about `center`.
    fn stretch(&mut self, id: ObjId, factors: P3, center: P3) {
        self.inner.transform(id, Transform::Stretch { factors, center });
    }

    /// A copy of an object in the same place (its own region); returns its id.
    fn copy(&mut self, id: ObjId) -> ObjId {
        self.inner.copy(id)
    }

    /// `target` becomes what it shares with every tool; the tools are used up.
    fn intersect(&mut self, target: ObjId, tools: Vec<ObjId>) {
        self.inner.intersect(target, tools);
    }

    /// Fuses solid objects into the first one's material region.
    fn fuse(&mut self, ids: Vec<ObjId>) {
        self.inner.fuse(ids);
    }

    fn set_face_maxh(&mut self, origins: Vec<PySel>, h: f64) {
        self.inner.set_face_maxh(origins.into_iter().map(sel_of).collect(), h);
    }

    /// A closed STL or OBJ surface as a solid, split into smooth surfaces
    /// at creases sharper than `crease_deg`.
    #[pyo3(signature = (path, crease_deg=40.0, maxh=None))]
    fn add_import(&mut self, path: String, crease_deg: f64, maxh: Option<f64>) -> ObjId {
        let mut i = Import::new(path);
        i.crease_deg = crease_deg;
        self.inner.add_solid(i, maxh, false)
    }

    /// Puts objects above every other one, in order (the last wins).
    fn bring_to_front(&mut self, ids: Vec<ObjId>) {
        self.inner.bring_to_front(&ids);
    }

    fn make_void(&mut self, ids: Vec<ObjId>) {
        self.inner.make_void(&ids);
    }

    fn remove(&mut self, id: ObjId) {
        self.inner.remove(id);
    }

    #[pyo3(signature = (id, maxh=None))]
    fn set_object_maxh(&mut self, id: ObjId, maxh: Option<f64>) {
        self.inner.set_object_maxh(id, maxh);
    }

    fn add_size_point(&mut self, p: P3, h: f64) {
        self.inner.add_size_point(p, h);
    }

    fn is_sheet(&self, id: ObjId) -> bool {
        matches!(self.inner.object(id).item, Item::Sheet(_))
    }

    fn is_void(&self, id: ObjId) -> bool {
        self.inner.object(id).void
    }

    fn n_objects(&self) -> usize {
        self.inner.objects().len()
    }

    /// Region of a solid object (None for sheets and voids).
    fn region(&self, id: ObjId) -> PyResult<Option<u32>> {
        self.inner.region(id).map_err(err)
    }

    /// Face origins `(object, role)` bounding an object.
    fn faces_of(&self, id: ObjId) -> PyResult<Vec<PySel>> {
        Ok(self.inner.faces_of(id).map_err(err)?.into_iter().map(sel_tuple).collect())
    }

    /// Every face origin of the realised model.
    fn all_faces(&self) -> PyResult<Vec<PySel>> {
        let r = self.inner.realized().map_err(err)?;
        let mut o: Vec<FaceSel> = r.faces.iter().map(|f| FaceSel { origin: f.origin, side: None, across: None }).collect();
        o.sort();
        o.dedup();
        Ok(o.into_iter().map(sel_tuple).collect())
    }

    /// Per origin: area-weighted centroid, mean normal, total area, the
    /// regions on either side (0 outside or in a void) and the bounding box
    /// `(xmin, ymin, zmin, xmax, ymax, zmax)`, over the faces it currently
    /// resolves to.
    fn face_info(&self, origins: Vec<PySel>) -> PyResult<Vec<(P3, P3, f64, Vec<u32>, [f64; 6])>> {
        let mut out = Vec::with_capacity(origins.len());
        for o in origins {
            let faces = self.inner.resolve(&[sel_of(o)]).map_err(err)?;
            let area: f64 = faces.iter().map(|f| f.area).sum();
            let mut c = [0.0; 3];
            let mut n = [0.0; 3];
            let mut regions = Vec::new();
            let mut bbox = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
            for f in &faces {
                for k in 0..3 {
                    bbox[k] = bbox[k].min(f.bbox[k]);
                    bbox[k + 3] = bbox[k + 3].max(f.bbox[k + 3]);
                }
                for k in 0..3 {
                    c[k] += f.centroid[k] * f.area / area.max(f64::MIN_POSITIVE);
                    n[k] += f.normal[k] * f.area / area.max(f64::MIN_POSITIVE);
                }
                regions.extend(f.regions);
            }
            regions.sort();
            regions.dedup();
            out.push((c, n, area, regions, bbox));
        }
        Ok(out)
    }

    /// Meshes the scene; returns `(n_points, n_tets, min_dihedral_deg,
    /// n_slivers)`.
    #[pyo3(signature = (maxh=None, grading=None, cells_across=1.0, optimize=false, target_elements=None))]
    fn mesh(
        &mut self,
        maxh: Option<f64>,
        grading: Option<f64>,
        cells_across: f64,
        optimize: bool,
        target_elements: Option<usize>,
    ) -> PyResult<(usize, usize, f64, usize)> {
        let opts = MeshOptions { maxh, grading, cells_across, optimize, target_elements, ..Default::default() };
        let m = self.inner.mesh(&opts).map_err(err)?;
        let stats = (m.points.len(), m.tets.len(), m.quality.min_dihedral_deg, m.quality.n_slivers);
        self.mesh = Some(m);
        Ok(stats)
    }

    /// Bounding box `(xmin, ymin, zmin, xmax, ymax, zmax)` of the scene.
    fn bbox(&self) -> PyResult<[f64; 6]> {
        self.inner.bbox().map_err(err)
    }

    /// The B-rep face ids the selections currently resolve to.
    fn face_ids(&self, origins: Vec<PySel>) -> PyResult<Vec<u32>> {
        let sels: Vec<FaceSel> = origins.into_iter().map(sel_of).collect();
        Ok(self.inner.resolve(&sels).map_err(err)?.iter().map(|f| f.id).collect())
    }

    /// Coarse per-face triangulation for previews: `(face id, corners,
    /// normals)`, nine values per triangle each.
    #[pyo3(signature = (target_triangles=4000))]
    fn preview(&self, target_triangles: usize) -> PyResult<Vec<(u32, Vec<f64>, Vec<f64>)>> {
        Ok(self.inner.preview(target_triangles).map_err(err)?.into_iter().map(|(id, (p, n))| (id, p, n)).collect())
    }

    /// Writes the last `mesh()` (holes left out) as gmsh MSH 4.1, the face
    /// and volume groups named by `names[tag]` its physical groups; returns
    /// the volume groups that found no region of their own.
    fn save_msh(
        &self,
        path: String,
        face_groups: Vec<(i32, Vec<PySel>)>,
        volume_groups: Vec<(i32, Vec<ObjId>)>,
        names: std::collections::HashMap<i32, String>,
    ) -> PyResult<Vec<String>> {
        let m = self.mesh.as_ref().ok_or_else(|| PyRuntimeError::new_err("mesh() first"))?;
        let (faces, volumes) = self.groups(face_groups, volume_groups)?;
        let named = |gs: Vec<Group>| -> Vec<(String, Vec<u32>)> {
            gs.into_iter().map(|g| (names.get(&g.tag).cloned().unwrap_or(format!("group_{}", g.tag)), g.ids)).collect()
        };
        let holes = self.inner.hole_regions().map_err(err)?;
        let io = |e: std::io::Error| PyRuntimeError::new_err(format!("{path}: {e}"));
        let mut w = std::io::BufWriter::new(std::fs::File::create(&path).map_err(io)?);
        let dropped = write_msh(m, &named(volumes), &named(faces), &holes, &mut w).map_err(io)?;
        std::io::Write::flush(&mut w).map_err(io)?;
        Ok(dropped)
    }

    /// The solver mesh of the last `mesh()`, with face groups `(tag,
    /// [origin])` and volume groups `(tag, [object])` for the model's tags.
    fn fem_mesh(
        &self,
        face_groups: Vec<(i32, Vec<PySel>)>,
        volume_groups: Vec<(i32, Vec<ObjId>)>,
    ) -> PyResult<PyFemMesh> {
        let m = self.mesh.as_ref().ok_or_else(|| PyRuntimeError::new_err("mesh() first"))?;
        let (faces, volumes) = self.groups(face_groups, volume_groups)?;
        let holes = self.inner.hole_regions().map_err(err)?;
        Ok(PyFemMesh { inner: fem_mesh(m, &faces, &volumes, &holes) })
    }
}

impl PyGeometry {
    fn groups(
        &self,
        face_groups: Vec<(i32, Vec<PySel>)>,
        volume_groups: Vec<(i32, Vec<ObjId>)>,
    ) -> PyResult<(Vec<Group>, Vec<Group>)> {
        let faces: Vec<(i32, Vec<FaceSel>)> =
            face_groups.into_iter().map(|(t, s)| (t, s.into_iter().map(sel_of).collect())).collect();
        self.inner.groups(&faces, &volume_groups).map_err(err)
    }
}

/// A pre-built MSH volume mesh (no remeshing) with its named physical
/// groups.
#[pyclass(name = "MeshScene", module = "rapidfem._native")]
pub struct PyMeshScene {
    inner: MeshScene,
}

#[pymethods]
impl PyMeshScene {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        Ok(PyMeshScene { inner: MeshScene::load(&path).map_err(PyValueError::new_err)? })
    }

    /// `(name, dim)` of every named group, in file order.
    fn groups(&self) -> Vec<(String, u8)> {
        self.inner.groups.iter().map(|g| (g.name.clone(), g.dim)).collect()
    }

    fn bbox(&self, name: &str) -> PyResult<[f64; 6]> {
        self.inner.bbox(name).map_err(PyKeyError::new_err)
    }

    /// `(n_points, n_tets, min_dihedral, n_slivers)`.
    fn stats(&self) -> (usize, usize, f64, usize) {
        let m = &self.inner.mesh;
        (m.points.len(), m.tets.len(), m.quality.min_dihedral_deg, m.quality.n_slivers)
    }

    /// The solver mesh, face and volume groups `(tag, [group name])`.
    fn fem_mesh(
        &self,
        face_groups: Vec<(i32, Vec<String>)>,
        volume_groups: Vec<(i32, Vec<String>)>,
    ) -> PyResult<PyFemMesh> {
        Ok(PyFemMesh { inner: self.inner.fem_mesh(&face_groups, &volume_groups).map_err(PyValueError::new_err)? })
    }
}

/// The solver mesh (nodes, tets, derived edges and faces, tag groups).
#[pyclass(name = "FemMesh", module = "rapidfem._native")]
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

    /// Tets and faces per tag, `(tag -> n_tets, tag -> n_tris)`.
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

    fn group_sizes(&self) -> (Vec<(i32, usize)>, Vec<(i32, usize)>) {
        let mut v: Vec<(i32, usize)> = self.inner.vtag_to_tet.iter().map(|(&t, s)| (t, s.len())).collect();
        let mut f: Vec<(i32, usize)> = self.inner.ftag_to_tri.iter().map(|(&t, s)| (t, s.len())).collect();
        v.sort();
        f.sort();
        (v, f)
    }
}
