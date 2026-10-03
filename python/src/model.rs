// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! `_native.Model`: the typed simulation model (`rapidfem_core::model`) of a
//! meshed geometry, every material and physics object under its mesh tag.
//! `_native.Geometry.model()` builds it; both backends are built from it.

use pyo3::prelude::*;
use rapidfem_core::model::Model;

#[pyclass(name = "Model", module = "rapidfem._native", skip_from_py_object)]
#[derive(Clone, Default)]
pub struct PyModel {
    pub inner: Model,
}

#[pymethods]
impl PyModel {
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}
