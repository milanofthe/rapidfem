// SPDX-License-Identifier: AGPL-3.0-only
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors

//! rapidfem-td, time-domain DGTD backend.
//!
//! The DG spatial operator, the Krylov/ETD exponential propagator, the
//! state-space export and model-order reduction land here. See
//! issue #27 for the work-package breakdown.

pub mod absorber;
pub mod build;
pub mod constants;
pub mod dg_basis;
pub mod dispersive;
pub mod explicit;
pub mod explicit_adaptive;
pub mod geom_factors;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod mesh_gen;
pub mod propagator;
pub mod rhs;
pub mod session;
pub mod waveguide;
