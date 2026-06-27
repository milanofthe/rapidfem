// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! rapidfem-core, solver-agnostic substrate shared by the frequency-domain
//! and time-domain backends: mesh, quadrature, and the material data model.

pub mod constants;
pub mod quadrature;
pub mod mesh;
pub mod quality;
pub mod mesh_io;
pub mod materials;
pub mod port_eigen;
pub mod topology;
