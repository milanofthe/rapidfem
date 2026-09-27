// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! `_native.Geometry` and `_native.FemMesh`: the rapidfem geometry on the
//! vendored rapidmesh (`rapidfem_geom`), and the solver mesh it produces.
//!
//! Objects are addressed by their index, faces by their origin
//! `(object, role)` (role -1 for a sheet). The Python `rapidfem.Geometry`
//! is a thin layer over these calls.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rapidfem_geom::fem_mesh::{fem_mesh, Group};
use rapidfem_geom::geometry::{FaceOrigin, Geometry, Item, ObjId};
use rapidmesh::shapes::{Cone, Cuboid, Cylinder, Loft, Prism, Sheet, Sphere, Torus, Wedge};
use rapidmesh::MeshOptions;

type P3 = [f64; 3];

fn origin_of((object, role): (usize, i64)) -> FaceOrigin {
    if role < 0 {
        FaceOrigin::Sheet { object }
    } else {
        FaceOrigin::Solid { object, role: role as u32 }
    }
}

fn origin_tuple(o: FaceOrigin) -> (usize, i64) {
    match o {
        FaceOrigin::Solid { object, role } => (object, role as i64),
        FaceOrigin::Sheet { object } => (object, -1),
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

    /// Extrudes an xy-plane sheet (polygon or axis-aligned rectangle) along
    /// +z by `height` into a prism, in place: the object keeps its id.
    fn extrude(&mut self, id: ObjId, height: f64) -> PyResult<()> {
        let prism = match &self.inner.object(id).item {
            Item::Sheet(Sheet::Polygon { points, holes, position }) => Prism {
                points: points.clone(),
                holes: holes.clone(),
                height,
                position: *position,
            },
            Item::Sheet(Sheet::Rect { corner, u, v }) if u[2] == 0.0 && v[2] == 0.0 => Prism {
                points: vec![
                    [0.0, 0.0],
                    [u[0], u[1]],
                    [u[0] + v[0], u[1] + v[1]],
                    [v[0], v[1]],
                ],
                holes: Vec::new(),
                height,
                position: *corner,
            },
            _ => {
                return Err(PyValueError::new_err(
                    "extrude: only xy-plane polygons and rectangles extrude along z so far \
                     (general directions: milanofthe/rapidmesh-dev#140)",
                ))
            }
        };
        // a negative height extrudes downwards: the same prism, shifted
        let prism = if height < 0.0 {
            let p = prism.position;
            Prism { height: -height, position: [p[0], p[1], p[2] + height], ..prism }
        } else {
            prism
        };
        self.inner.replace(id, Item::Solid(prism.into()));
        Ok(())
    }

    /// Ruled loft between two planar profiles with the same vertex count.
    #[pyo3(signature = (profile_a, profile_b, maxh=None))]
    fn add_loft(&mut self, profile_a: Vec<P3>, profile_b: Vec<P3>, maxh: Option<f64>) -> ObjId {
        self.inner.add_solid(Loft { profile_a, profile_b }, maxh, false)
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
    fn faces_of(&self, id: ObjId) -> PyResult<Vec<(usize, i64)>> {
        Ok(self.inner.faces_of(id).map_err(err)?.into_iter().map(origin_tuple).collect())
    }

    /// Every face origin of the realised model.
    fn all_faces(&self) -> PyResult<Vec<(usize, i64)>> {
        let r = self.inner.realized().map_err(err)?;
        let mut o: Vec<FaceOrigin> = r.faces.iter().map(|f| f.origin).collect();
        o.sort();
        o.dedup();
        Ok(o.into_iter().map(origin_tuple).collect())
    }

    /// Per origin: area-weighted centroid, mean normal, total area and the
    /// regions on either side (0 outside or in a void), over the faces it
    /// currently resolves to.
    fn face_info(&self, origins: Vec<(usize, i64)>) -> PyResult<Vec<(P3, P3, f64, Vec<u32>)>> {
        let mut out = Vec::with_capacity(origins.len());
        for o in origins {
            let faces = self.inner.resolve(&[origin_of(o)]).map_err(err)?;
            let area: f64 = faces.iter().map(|f| f.area).sum();
            let mut c = [0.0; 3];
            let mut n = [0.0; 3];
            let mut regions = Vec::new();
            for f in &faces {
                for k in 0..3 {
                    c[k] += f.centroid[k] * f.area / area.max(f64::MIN_POSITIVE);
                    n[k] += f.normal[k] * f.area / area.max(f64::MIN_POSITIVE);
                }
                regions.extend(f.regions);
            }
            regions.sort();
            regions.dedup();
            out.push((c, n, area, regions));
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

    /// The solver mesh of the last `mesh()`, with face groups `(tag,
    /// [origin])` and volume groups `(tag, [object])` for the model's tags.
    fn fem_mesh(
        &self,
        face_groups: Vec<(i32, Vec<(usize, i64)>)>,
        volume_groups: Vec<(i32, Vec<ObjId>)>,
    ) -> PyResult<PyFemMesh> {
        let m = self.mesh.as_ref().ok_or_else(|| PyRuntimeError::new_err("mesh() first"))?;
        let mut faces = Vec::with_capacity(face_groups.len());
        for (tag, origins) in face_groups {
            let origins: Vec<FaceOrigin> = origins.into_iter().map(origin_of).collect();
            let ids = self.inner.resolve(&origins).map_err(err)?.iter().map(|f| f.id).collect();
            faces.push(Group { tag, ids });
        }
        let mut volumes = Vec::with_capacity(volume_groups.len());
        for (tag, objs) in volume_groups {
            let mut ids = Vec::new();
            for o in objs {
                if let Some(r) = self.inner.region(o).map_err(err)? {
                    ids.push(r);
                }
            }
            volumes.push(Group { tag, ids });
        }
        Ok(PyFemMesh { inner: fem_mesh(m, &faces, &volumes) })
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
    fn group_sizes(&self) -> (Vec<(i32, usize)>, Vec<(i32, usize)>) {
        let mut v: Vec<(i32, usize)> = self.inner.vtag_to_tet.iter().map(|(&t, s)| (t, s.len())).collect();
        let mut f: Vec<(i32, usize)> = self.inner.ftag_to_tri.iter().map(|(&t, s)| (t, s.len())).collect();
        v.sort();
        f.sort();
        (v, f)
    }
}
