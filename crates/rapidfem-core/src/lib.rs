// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! rapidfem-core, solver-agnostic substrate shared by the frequency-domain
//! and time-domain backends: mesh, quadrature, the material data model and
//! the sparse symmetric solver.

pub mod constants;
pub mod linalg;
pub mod model;
pub mod quadrature;
pub mod mesh;
pub mod quality;
pub mod materials;
pub mod port_eigen;
pub mod topology;
