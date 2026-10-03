// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! RFIC models: the process stack, GDS layouts and the rapidpassives FEM
//! JSON turned into solve-ready scenes.

pub mod build;
pub mod fem_json;
pub mod gds;
#[cfg(feature = "python")]
pub mod python;
pub mod region;
pub mod stack;

use rapidmesh::shapes::Sheet;

use crate::geometry::{Geometry, ObjId};
use crate::setup::{Setup, Target};
use region::{P2, Region};

/// A material the builders fill objects with: air or a dielectric with
/// loss tangent, conductivity or a diagonal conductivity, and the mesh
/// size of what it fills.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub air: bool,
    pub er: f64,
    pub tand: f64,
    pub conductivity: f64,
    pub cond_diag: Option<[f64; 3]>,
    pub maxh: Option<f64>,
}

impl Mat {
    pub fn air() -> Mat {
        Mat { air: true, er: 1.0, tand: 0.0, conductivity: 0.0, cond_diag: None, maxh: None }
    }

    pub fn dielectric(er: f64) -> Mat {
        Mat { air: false, er, ..Mat::air() }
    }
}

/// The scene a builder works on: the geometry, its setup and the
/// materials it added (by their index in the setup), which the caller
/// turns into its own material objects.
pub struct Scene<'a> {
    pub geo: &'a mut Geometry,
    pub setup: &'a mut Setup,
    pub materials: Vec<(usize, Mat)>,
}

impl<'a> Scene<'a> {
    pub fn new(geo: &'a mut Geometry, setup: &'a mut Setup) -> Scene<'a> {
        Scene { geo, setup, materials: Vec::new() }
    }

    /// Fills the objects `ids` with one new material `m`.
    pub fn fill(&mut self, ids: &[ObjId], m: Mat) {
        let index = self.setup.add_material(if m.air { "air" } else { "dielectric" }.into());
        for &id in ids {
            self.geo.set_material_maxh(id, m.maxh);
            self.setup.set_material(Target::Object(id), Some(index));
        }
        self.materials.push((index, m));
    }

    /// The xy polygon `contours` (outer, then holes) at height `z`, extruded
    /// by `height` when given (a sheet otherwise), filled with `material`
    /// and sized `maxh`.
    pub fn prism(&mut self, contours: &[Vec<P2>], z: f64, height: Option<f64>, material: Option<Mat>, maxh: Option<f64>) -> Result<ObjId, String> {
        let lift = |c: &Vec<P2>| c.iter().map(|p| [p[0], p[1], z]).collect::<Vec<[f64; 3]>>();
        let (outer, holes) = contours.split_first().ok_or("an empty polygon")?;
        let holes: Vec<Vec<[f64; 3]>> = holes.iter().map(lift).collect();
        let sheet: Sheet = crate::geometry::polygon_sheet(&lift(outer), &holes).ok_or("a polygon off its plane")?;
        let id = self.geo.add_sheet(sheet, None);
        if let Some(h) = height {
            self.geo.extrude(id, [0.0, 0.0, h])?;
        }
        if maxh.is_some() {
            self.geo.set_object_maxh(id, maxh);
        }
        if let Some(m) = material {
            self.fill(&[id], m);
        }
        Ok(id)
    }

    /// One prism per shape of `region`, all filled with one `material`.
    pub fn prisms(&mut self, region: &Region, z: f64, height: f64, material: Option<Mat>, maxh: Option<f64>) -> Result<Vec<ObjId>, String> {
        let ids = region.iter().map(|shape| self.prism(shape, z, Some(height), None, maxh)).collect::<Result<Vec<_>, _>>()?;
        if let Some(m) = material {
            self.fill(&ids, m);
        }
        Ok(ids)
    }
}
