// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! `_native.Model`: the typed simulation model (`rapidfem_core::model`) built
//! from Python, one method per material, port and boundary condition. The
//! Python physics classes call these with their mesh tag; both backends are
//! built from the result.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rapidfem_core::model::{
    Debye, Drude, FaceSpec, MaterialSpec, Model, PmlSpec, WaveKind,
};

#[pyclass(name = "Model", module = "rapidfem._native", skip_from_py_object)]
#[derive(Clone, Default)]
pub struct PyModel {
    pub inner: Model,
}

#[pymethods]
impl PyModel {
    #[new]
    fn new() -> Self {
        PyModel::default()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    #[pyo3(signature = (volume_tag, *, er, ur, tand, conductivity, cond_diag=None, er_diag=None, ur_diag=None, debye=None, drude=None))]
    fn add_material(
        &mut self,
        volume_tag: i32,
        er: f64,
        ur: f64,
        tand: f64,
        conductivity: f64,
        cond_diag: Option<[f64; 3]>,
        er_diag: Option<[f64; 3]>,
        ur_diag: Option<[f64; 3]>,
        debye: Option<(f64, f64, f64)>,
        drude: Option<(f64, f64, f64)>,
    ) {
        self.inner.materials.push(MaterialSpec {
            volume_tag,
            er,
            ur,
            tand,
            conductivity,
            cond_diag,
            er_diag,
            ur_diag,
            debye: debye.map(|(er_inf, er_static, tau_s)| Debye { er_inf, er_static, tau_s }),
            drude: drude.map(|(er_inf, plasma_freq_hz, damping_freq_hz)| Drude {
                er_inf,
                plasma_freq_hz,
                damping_freq_hz,
            }),
        });
    }

    fn add_pec(&mut self, tag: i32) {
        self.inner.pec_tags.push(tag);
    }

    fn add_pmc(&mut self, tag: i32) {
        self.inner.faces.push(FaceSpec::Pmc { tag });
    }

    fn add_abc(&mut self, tag: i32) {
        self.inner.faces.push(FaceSpec::Abc { tag });
    }

    fn set_far_field(&mut self, tag: i32) -> PyResult<()> {
        if self.inner.far_field_tag.is_some() {
            return Err(PyValueError::new_err(
                "multiple FarFieldSurface objects, but only one near-field-to-far-field \
                 surface is supported. Pass every face to a single FarFieldSurface(...) \
                 call (e.g. rf.FarFieldSurface(*air.faces.hull)).",
            ));
        }
        self.inner.far_field_tag = Some(tag);
        Ok(())
    }

    fn add_periodic(&mut self, tag_a: i32, tag_b: i32) {
        self.inner.periodic.push((tag_a, tag_b));
    }

    #[pyo3(signature = (tag, *, width, height, mode, er, power))]
    fn add_rect_port(&mut self, tag: i32, width: f64, height: f64, mode: [usize; 2], er: f64, power: f64) {
        self.inner.faces.push(FaceSpec::Rectangular { tag, width, height, mode, er, power });
    }

    #[pyo3(signature = (tag, *, scan_theta_deg, scan_phi_deg, mode_nr, er, power))]
    fn add_floquet_port(
        &mut self,
        tag: i32,
        scan_theta_deg: f64,
        scan_phi_deg: f64,
        mode_nr: u32,
        er: f64,
        power: f64,
    ) {
        self.inner.faces.push(FaceSpec::Floquet { tag, scan_theta_deg, scan_phi_deg, mode_nr, er, power });
    }

    #[pyo3(signature = (tag, *, e_field, power))]
    fn add_user_port(&mut self, tag: i32, e_field: [f64; 3], power: f64) {
        self.inner.faces.push(FaceSpec::UserDefined { tag, e_field, power });
    }

    #[pyo3(signature = (tag, *, ri, ro, er, power, origin=None, z_axis=None))]
    fn add_coax_port(
        &mut self,
        tag: i32,
        ri: f64,
        ro: f64,
        er: f64,
        power: f64,
        origin: Option<[f64; 3]>,
        z_axis: Option<[f64; 3]>,
    ) {
        self.inner.faces.push(FaceSpec::Coax { tag, ri, ro, origin, z_axis, er, power });
    }

    #[pyo3(signature = (tag, *, z0, l, direction, width, height, power, c=None))]
    fn add_lumped_port(
        &mut self,
        tag: i32,
        z0: f64,
        l: f64,
        direction: [f64; 3],
        width: f64,
        height: f64,
        power: f64,
        c: Option<f64>,
    ) {
        self.inner.faces.push(FaceSpec::Lumped { tag, z0, l, c, direction, width, height, power });
    }

    #[pyo3(signature = (tag, *, r, l, direction, width, height, c=None))]
    fn add_lumped_element(
        &mut self,
        tag: i32,
        r: f64,
        l: f64,
        direction: [f64; 3],
        width: f64,
        height: f64,
        c: Option<f64>,
    ) {
        self.inner.faces.push(FaceSpec::LumpedElement { tag, r, l, c, width, height, direction });
    }

    /// `kind` is `"vector"`, `"te"` or `"tm"`.
    #[pyo3(signature = (tag, *, kind, mode_index, power, f0=None, pec_tags=Vec::new()))]
    fn add_wave_port(
        &mut self,
        tag: i32,
        kind: &str,
        mode_index: usize,
        power: f64,
        f0: Option<f64>,
        pec_tags: Vec<i32>,
    ) -> PyResult<()> {
        let kind = match kind {
            "vector" => WaveKind::Vector,
            "te" => WaveKind::Te,
            "tm" => WaveKind::Tm,
            other => {
                return Err(PyValueError::new_err(format!(
                    "wave port kind must be 'vector', 'te' or 'tm', got {other:?}"
                )))
            }
        };
        self.inner.faces.push(FaceSpec::WaveNumerical { tag, f0, mode_index, kind, pec_tags, power });
        Ok(())
    }

    #[pyo3(signature = (tag, *, conductivity, mur, er, thickness=None, two_sided=false, sheet=false, zs=None))]
    fn add_surface_impedance(
        &mut self,
        tag: i32,
        conductivity: f64,
        mur: f64,
        er: f64,
        thickness: Option<f64>,
        two_sided: bool,
        sheet: bool,
        zs: Option<[f64; 2]>,
    ) {
        self.inner.faces.push(FaceSpec::SurfaceImpedance {
            tag,
            conductivity,
            mur,
            er,
            thickness,
            two_sided,
            sheet,
            zs,
        });
    }

    #[pyo3(signature = (volume_tag, *, direction, inner_face, thickness, er_base, ur_base, exponent, delta_max))]
    fn add_pml(
        &mut self,
        volume_tag: i32,
        direction: [f64; 3],
        inner_face: f64,
        thickness: f64,
        er_base: f64,
        ur_base: f64,
        exponent: f64,
        delta_max: f64,
    ) {
        self.inner.pml.push(PmlSpec {
            volume_tag,
            direction,
            inner_face,
            thickness,
            er_base,
            ur_base,
            exponent,
            delta_max,
        });
    }
}
