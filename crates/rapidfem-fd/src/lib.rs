// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! rapidfem-fd, frequency-domain Nédélec-FEM backend.
//!
//! The solver-agnostic mesh / quadrature / material data model lives in
//! `rapidfem-core` and is re-exported here, so existing `crate::mesh`-style
//! paths inside this crate keep resolving unchanged.

pub use rapidfem_core::{constants, linalg, materials, mesh, model, quadrature};

mod dump;
pub mod excitation;
pub mod coefficients;
pub mod dofmap;
pub mod order;
pub mod basis;
pub mod tet_assembly;
pub mod tri_assembly;
pub mod waveguide;
pub mod sparam;
pub mod network;
pub mod interp;
pub mod port;
pub mod sibc_edge;
pub mod error_estimator;
pub mod eigenmode;
pub mod assembly;
pub mod farfield;
pub mod simulation;
