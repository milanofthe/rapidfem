// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! rapidfem-fd, frequency-domain Nédélec-FEM backend.
//!
//! The solver-agnostic mesh / quadrature / material data model lives in
//! `rapidfem-core` and is re-exported here, so existing `crate::mesh`-style
//! paths inside this crate keep resolving unchanged.

pub use rapidfem_core::{constants, materials, mesh, mesh_io, quadrature};

pub mod excitation;
pub mod coefficients;
pub mod basis;
pub mod tet_assembly_r2;
pub mod tri_assembly_r2;
pub mod waveguide;
pub mod sparam;
pub mod interp;
pub mod port;
pub mod touchstone;
pub mod pardiso;
pub mod solver;
#[cfg(feature = "vtk")]
pub mod vtk_export;
pub mod error_estimator;
pub mod eigenmode;
pub mod config;
pub mod assembly;
pub mod farfield;
pub mod simulation;
