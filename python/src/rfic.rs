// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The RFIC builders (`rapidfem_geom::rfic`) as `_native` functions: each
//! makes a fresh `Geometry` and returns it with the objects it built, by
//! index. The Python `rapidfem.rfic` wraps them.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rapidfem_geom::geometry::ObjId;
use rapidfem_geom::rfic::build::{self, Mesh, MeshSpec, ViaPort, ZBound};
use rapidfem_geom::rfic::fem_json;
use rapidfem_geom::rfic::stack::Stack;
use rapidfem_geom::rfic::{gds, python::from_py};

use crate::geometry::PyGeometry;

/// A `ViaPort` bound: a height or a layer name.
#[derive(FromPyObject)]
pub(crate) enum ZArg {
    Height(f64),
    Layer(String),
}

impl From<ZArg> for ZBound {
    fn from(b: ZArg) -> ZBound {
        match b {
            ZArg::Height(z) => ZBound::Height(z),
            ZArg::Layer(n) => ZBound::Layer(n),
        }
    }
}

/// A Python `rfic.ViaPort` as the builder's.
fn via_port(p: &Bound<'_, PyAny>) -> PyResult<ViaPort> {
    let (lo, hi): (ZArg, ZArg) = p.getattr("z")?.extract()?;
    let axis: String = p.getattr("axis")?.extract()?;
    let axis = match axis.as_str() {
        "x" => 'x',
        "y" => 'y',
        _ => return Err(PyValueError::new_err(format!("ViaPort axis must be 'x' or 'y', got {axis:?}"))),
    };
    Ok(ViaPort {
        z: (lo.into(), hi.into()),
        span: p.getattr("span")?.extract()?,
        at: p.getattr("at")?.extract()?,
        axis,
        marker: p.getattr("marker")?.extract()?,
        z0: p.getattr("z0")?.extract()?,
    })
}

/// The mesh policy: a preset name, None for "balanced", or a `MeshSpec`.
#[derive(FromPyObject)]
pub(crate) enum MeshArg {
    Preset(String),
    Spec(MeshSpec),
}

fn named<'py>(py: Python<'py>, items: &[(String, Vec<ObjId>)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (name, ids) in items {
        d.set_item(name, ids)?;
    }
    Ok(d)
}

/// Builds the RFIC model of the GDS layout `gds` in `stack`; returns the
/// geometry and a dict of what was built (`conductors`, `slabs`,
/// `air_shell`, `ports`, `footprint`, `warnings`).
#[pyfunction]
#[pyo3(signature = (gds, stack, *, top_cell, ports, margin, air, air_top, pec_floor, conductor_model, band, mesh, passivation, pass_t_side, pass_t_top, conformal_over, boundary, via_merge))]
#[allow(clippy::too_many_arguments)]
pub fn rfic_build<'py>(
    py: Python<'py>,
    gds: PathBuf,
    stack: Stack,
    top_cell: Option<String>,
    ports: Vec<Bound<'py, PyAny>>,
    margin: f64,
    air: f64,
    air_top: Option<f64>,
    pec_floor: bool,
    conductor_model: Option<BTreeMap<String, String>>,
    band: Option<(f64, f64)>,
    mesh: Option<MeshArg>,
    passivation: String,
    pass_t_side: f64,
    pass_t_top: Option<f64>,
    conformal_over: Option<String>,
    boundary: String,
    via_merge: Option<f64>,
) -> PyResult<(PyGeometry, Bound<'py, PyDict>)> {
    let opts = build::Options {
        top_cell,
        ports: ports.iter().map(via_port).collect::<PyResult<_>>()?,
        margin,
        air,
        air_top,
        pec_floor,
        conductor_model: conductor_model.unwrap_or_default(),
        band,
        mesh: match mesh {
            None => Mesh::Preset("balanced".into()),
            Some(MeshArg::Preset(p)) => Mesh::Preset(p),
            Some(MeshArg::Spec(s)) => Mesh::Spec(s),
        },
        passivation,
        pass_t_side,
        pass_t_top,
        conformal_over,
        boundary,
        via_merge,
    };
    let mut g = PyGeometry::fresh(None, true);
    let built = g.build(py, |scene| build::build(scene, &gds, &stack, &opts))?;
    g.mesh_maxh = Some(built.maxh);
    // a stack is layers far thinner than the element size: flat tets
    // through each, not elements across every one
    g.cells_across = Some(0.0);
    let d = PyDict::new(py);
    d.set_item("conductors", named(py, &built.conductors)?)?;
    d.set_item("slabs", named(py, &built.slabs)?)?;
    d.set_item("air_shell", &built.air_shell)?;
    d.set_item("ports", &built.ports)?;
    d.set_item("footprint", built.footprint)?;
    d.set_item("warnings", &built.warnings)?;
    Ok((g, d))
}

/// Every stack layer of the GDS layout as prisms (sheets with
/// `thin_conductors` for metals), named after the layer; returns the
/// geometry and the cell name.
#[pyfunction]
#[pyo3(signature = (path, stack, *, top_cell = None, bbox = None, merge = true, thin_conductors = false))]
pub fn rfic_from_gds(
    py: Python<'_>,
    path: PathBuf,
    stack: Stack,
    top_cell: Option<String>,
    bbox: Option<[f64; 4]>,
    merge: bool,
    thin_conductors: bool,
) -> PyResult<(PyGeometry, String)> {
    let layout = gds::read(&path, top_cell.as_deref()).map_err(PyValueError::new_err)?;
    let mut g = PyGeometry::fresh(None, true);
    g.build(py, |scene| build::extrude_layout(scene, &layout, &stack, bbox, merge, thin_conductors, None).map(|(layers, _)| layers))?;
    Ok((g, layout.cell))
}

/// The scene of a rapidpassives FEM JSON `doc`; returns the geometry and a
/// dict of what was built (`conductors`, `ports`, `ground_patches`,
/// `substrate`, `oxide`, `air`).
#[pyfunction]
#[pyo3(signature = (doc, *, stack, via_mode, footprint_margin, air_height_um, conductor_maxh_um, port_maxh_um, port_tab_um, port_inset_um, port_z0))]
#[allow(clippy::too_many_arguments)]
pub fn rfic_from_fem_json<'py>(
    py: Python<'py>,
    doc: Bound<'py, PyAny>,
    stack: Option<Stack>,
    via_mode: String,
    footprint_margin: f64,
    air_height_um: f64,
    conductor_maxh_um: f64,
    port_maxh_um: f64,
    port_tab_um: f64,
    port_inset_um: Option<f64>,
    port_z0: f64,
) -> PyResult<(PyGeometry, Bound<'py, PyDict>)> {
    if via_mode != "merged" && via_mode != "cells" {
        return Err(PyValueError::new_err(format!("via_mode must be 'merged' or 'cells', got {via_mode:?}")));
    }
    let doc = from_py(&doc)?;
    let opts = fem_json::Options { via_mode, footprint_margin, air_height_um, conductor_maxh_um, port_maxh_um, port_tab_um, port_inset_um, port_z0 };
    let mut g = PyGeometry::fresh(None, true);
    let built = g.build(py, |scene| fem_json::from_fem_json(scene, &doc, stack.as_ref(), &opts))?;
    g.mesh_maxh = Some(built.maxh);
    let d = PyDict::new(py);
    d.set_item("conductors", named(py, &built.conductors)?)?;
    let ports = PyDict::new(py);
    for (name, id) in &built.ports {
        ports.set_item(name, id)?;
    }
    d.set_item("ports", ports)?;
    d.set_item("ground_patches", &built.ground_patches)?;
    d.set_item("substrate", built.substrate)?;
    d.set_item("oxide", built.oxide)?;
    d.set_item("air", built.air)?;
    Ok((g, d))
}
