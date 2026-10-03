// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! The Python face of the stack types (`rapidfem.rfic.Stack`, `PdkLayer`,
//! `DielectricLayer`, `StackMaterial`, `MeshSpec`), with the `python`
//! feature. Lengths in metres.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::build::MeshSpec;
use super::stack::{DielectricLayer, PdkLayer, Stack, StackMaterial};

fn value_err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// A JSON value as the Python object `json.loads` makes of it.
pub fn to_py<'py>(py: Python<'py>, v: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    py.import("json")?.call_method1("loads", (v.to_string(),))
}

/// A Python object as JSON, by `json.dumps`.
pub fn from_py(o: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let text: String = o.py().import("json")?.call_method1("dumps", (o,))?.extract()?;
    serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymethods]
impl StackMaterial {
    #[new]
    #[pyo3(signature = (name, kind = "dielectric", er = 1.0, tand = 0.0, sigma = 0.0, color = "#888"))]
    fn py_new(name: String, kind: &str, er: f64, tand: f64, sigma: f64, color: &str) -> Self {
        StackMaterial { name, kind: kind.into(), er, tand, sigma, color: color.into() }
    }

    fn __repr__(&self) -> String {
        format!("StackMaterial({:?}, {:?}, er={}, tand={}, sigma={})", self.name, self.kind, self.er, self.tand, self.sigma)
    }
}

#[pymethods]
impl DielectricLayer {
    #[new]
    fn py_new(name: String, material: String, z: f64, thickness: f64) -> Self {
        DielectricLayer { name, material, z, thickness }
    }

    #[getter(z_top)]
    fn py_z_top(&self) -> f64 {
        self.z_top()
    }

    fn __repr__(&self) -> String {
        format!("DielectricLayer({:?}, {:?}, z={}, thickness={})", self.name, self.material, self.z, self.thickness)
    }
}

#[pymethods]
impl PdkLayer {
    #[new]
    #[pyo3(signature = (name, gds, datatype, z, thickness, color = "#888", r#type = "metal", er = 1.0, ur = 1.0, tand = 0.0, sigma = 0.0, material = None))]
    fn py_new(
        name: &str,
        gds: i32,
        datatype: i32,
        z: f64,
        thickness: f64,
        color: &str,
        r#type: &str,
        er: f64,
        ur: f64,
        tand: f64,
        sigma: f64,
        material: Option<String>,
    ) -> Self {
        PdkLayer { er, ur, tand, sigma, material, ..PdkLayer::new(name, gds, datatype, z, thickness, color, r#type) }
    }

    #[getter(z_top)]
    fn py_z_top(&self) -> f64 {
        self.z_top()
    }

    #[getter]
    fn gds_key(&self) -> (i32, i32) {
        (self.gds, self.datatype)
    }

    #[getter(is_pec)]
    fn py_is_pec(&self) -> bool {
        self.is_pec()
    }

    fn __repr__(&self) -> String {
        format!("PdkLayer({:?}, {}/{}, z={}, thickness={}, {:?})", self.name, self.gds, self.datatype, self.z, self.thickness, self.r#type)
    }
}

/// A layer of either kind, for `Stack.material_of`.
#[derive(FromPyObject)]
enum AnyLayer {
    Patterned(PdkLayer),
    Background(DielectricLayer),
}

/// A stackup source: XML markup (leading '<') or a path.
#[derive(FromPyObject)]
enum Source {
    Text(String),
    Path(PathBuf),
}

#[pymethods]
impl Stack {
    #[new]
    #[pyo3(signature = (name, layers, dielectrics = Vec::new(), materials = None))]
    fn py_new(name: String, layers: Vec<PdkLayer>, dielectrics: Vec<DielectricLayer>, materials: Option<BTreeMap<String, StackMaterial>>) -> Self {
        Stack::new(name, layers, dielectrics, materials.map(|m| m.into_values().collect()).unwrap_or_default())
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// Patterned layers, bottom to top.
    #[getter]
    fn layers(&self) -> Vec<PdkLayer> {
        self.layers.clone()
    }

    /// Background slabs, bottom to top.
    #[getter]
    fn dielectrics(&self) -> Vec<DielectricLayer> {
        self.dielectrics.clone()
    }

    /// The materials table, by name.
    #[getter]
    fn materials<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for m in &self.materials {
            d.set_item(&m.name, m.clone())?;
        }
        Ok(d)
    }

    /// A preset by PDK name: "sky130" or "sg13g2".
    #[staticmethod]
    #[pyo3(name = "from_pdk")]
    fn py_from_pdk(name: &str) -> PyResult<Stack> {
        Stack::from_pdk(name).map_err(value_err)
    }

    /// A gds2palace / ADS stackup XML, from a path or as markup.
    #[staticmethod]
    #[pyo3(name = "from_xml", signature = (source, *, name = None))]
    fn py_from_xml(source: Source, name: Option<String>) -> PyResult<Stack> {
        let (text, default) = match source {
            Source::Text(t) if t.trim_start().starts_with('<') => (t, "stack".to_string()),
            Source::Text(t) => {
                let p = PathBuf::from(t);
                (std::fs::read_to_string(&p).map_err(|e| PyValueError::new_err(format!("{}: {e}", p.display())))?, stem(&p))
            }
            Source::Path(p) => (std::fs::read_to_string(&p).map_err(|e| PyValueError::new_err(format!("{}: {e}", p.display())))?, stem(&p)),
        };
        Stack::from_xml(&text, name.as_deref().unwrap_or(&default)).map_err(value_err)
    }

    #[staticmethod]
    #[pyo3(name = "sky130")]
    fn py_sky130() -> Stack {
        Stack::sky130()
    }

    #[staticmethod]
    #[pyo3(name = "sg13g2")]
    fn py_sg13g2() -> Stack {
        Stack::sg13g2()
    }

    /// The stack of a rapidpassives `Pdk` JSON dict.
    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyAny>) -> PyResult<Stack> {
        Stack::from_json(&from_py(d)?).map_err(value_err)
    }

    /// The rapidpassives `Pdk` JSON dict (lengths in microns).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.to_json())
    }

    #[pyo3(name = "by_name")]
    fn py_by_name(&self, name: &str) -> PyResult<PdkLayer> {
        self.by_name(name).cloned().map_err(PyKeyError::new_err)
    }

    /// The layer on GDS (number, datatype), or None.
    #[pyo3(name = "by_gds", signature = (gds, datatype = 0))]
    fn py_by_gds(&self, gds: i32, datatype: i32) -> Option<PdkLayer> {
        self.by_gds(gds, datatype).cloned()
    }

    fn metals(&self) -> Vec<PdkLayer> {
        self.layers.iter().filter(|l| l.r#type == "metal").cloned().collect()
    }

    fn vias(&self) -> Vec<PdkLayer> {
        self.layers.iter().filter(|l| l.r#type == "via").cloned().collect()
    }

    /// A layer's material record (synthesised from the layer's scalars
    /// when the table lacks it).
    fn material_of(&self, layer: AnyLayer) -> StackMaterial {
        match layer {
            AnyLayer::Patterned(l) => self.layer_material(&l),
            AnyLayer::Background(d) => self.slab_material(&d),
        }
    }

    /// The background slab containing height `z`, or None.
    #[pyo3(name = "dielectric_at")]
    fn py_dielectric_at(&self, z: f64) -> Option<DielectricLayer> {
        self.dielectric_at(z).cloned()
    }

    #[getter(top_z)]
    fn py_top_z(&self) -> f64 {
        self.top_z()
    }

    /// z of the lowest patterned layer.
    #[getter(bottom_z)]
    fn py_bottom_z(&self) -> f64 {
        self.bottom_z()
    }

    fn __repr__(&self) -> String {
        format!("Stack({:?}, {} layers, {} dielectrics)", self.name, self.layers.len(), self.dielectrics.len())
    }
}

fn stem(p: &std::path::Path) -> String {
    p.file_stem().map_or_else(|| "stack".into(), |s| s.to_string_lossy().into_owned())
}

#[pymethods]
impl MeshSpec {
    #[new]
    #[pyo3(signature = (scale = 1.0, conductor = 5e-6, port = 3e-6, global_h = 40e-6, slab_default = None, slabs = BTreeMap::new(), graded = BTreeMap::new()))]
    fn py_new(
        scale: f64,
        conductor: f64,
        port: f64,
        global_h: f64,
        slab_default: Option<f64>,
        slabs: BTreeMap<String, f64>,
        graded: BTreeMap<String, Vec<(f64, f64)>>,
    ) -> Self {
        MeshSpec { scale, conductor, port, global_h, slab_default, slabs, graded }
    }

    #[pyo3(name = "h")]
    fn py_h(&self, value: f64) -> f64 {
        self.h(value)
    }

    #[pyo3(name = "slab_h")]
    fn py_slab_h(&self, name: &str) -> f64 {
        self.slab_h(name)
    }

    /// The mesh policy from the stack and the layers drawn in the layout,
    /// see the module documentation of `rfic.build`.
    #[staticmethod]
    #[pyo3(name = "derive", signature = (stack, layer_names, preset = "balanced"))]
    fn py_derive(stack: &Stack, layer_names: &Bound<'_, PyAny>, preset: &str) -> PyResult<MeshSpec> {
        let names: Vec<String> = layer_names.try_iter()?.map(|n| n?.extract()).collect::<PyResult<_>>()?;
        MeshSpec::derive(stack, &names, preset).map_err(value_err)
    }

    fn __repr__(&self) -> String {
        format!("MeshSpec(scale={}, conductor={}, port={}, global_h={})", self.scale, self.conductor, self.port, self.global_h)
    }
}

/// Registers the classes with a Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Stack>()?;
    m.add_class::<PdkLayer>()?;
    m.add_class::<DielectricLayer>()?;
    m.add_class::<StackMaterial>()?;
    m.add_class::<MeshSpec>()?;
    m.add("VIA_LATERAL_FACTOR", super::build::VIA_LATERAL_FACTOR)?;
    m.add("FEM_JSON_SCHEMA_VERSIONS", super::fem_json::SCHEMA_VERSIONS.to_vec())?;
    Ok(())
}
