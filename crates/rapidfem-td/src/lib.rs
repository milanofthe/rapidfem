// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! rapidfem-td — time-domain DGTD backend.
//!
//! The DG spatial operator, the Krylov/ETD exponential propagator, the
//! state-space export and model-order reduction land here. See
//! `docs/td-backend-plan.md` for the work-package breakdown.

pub mod absorber;
pub mod constants;
pub mod dg_basis;
pub mod dispersive;
pub mod explicit;
pub mod explicit_adaptive;
pub mod geom_factors;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod macromodel;
pub mod mesh_gen;
pub mod mor;
pub mod propagator;
pub mod rhs;
pub mod waveguide;
