// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
// Copyright (C) Robert Fennis (original EMerge source)
//
// This file is part of rapidfem and contains code ported from EMerge
// (https://github.com/FennisRobert/EMerge), originally licensed under
// GPL-2.0-or-later with the Gmsh additional permission; redistributed
// here under GPL-3.0-or-later with that permission preserved.
// See LICENSE and NOTICE for the full terms.

//! Physical constants (mirrors emerge/_emerge/const.py).

pub const C0: f64 = 299_792_458.0;                    // Speed of light (m/s)
pub const Z0: f64 = 376.73031366857;                   // Free space impedance (Ω)
pub const EPS0: f64 = 8.854187818814e-12;              // Permittivity of free space (F/m)
pub const MU0: f64 = 1.0 / (C0 * C0 * EPS0);          // Permeability of free space (H/m)
pub const PI: f64 = std::f64::consts::PI;

// --- Numerical tolerances ---

/// Magnitudes below this are treated as zero. A determinant this small
/// marks a singular matrix or degenerate element (`1/det` would produce
/// NaN/inf); a power or norm this small is numerical noise. A uniform
/// "negligible" floor across the solver.
pub const SINGULAR_EPS: f64 = 1e-30;

/// Barycentric-coordinate slack for point-in-tetrahedron containment.
pub const POINT_IN_TET_EPS: f64 = 1e-8;

/// Lanczos lucky-breakdown threshold for the eigenmode solver: the Krylov
/// subspace stops growing once the next vector's norm drops below this.
pub const LANCZOS_BREAKDOWN: f64 = 1e-12;

/// Relative Ritz-pair convergence tolerance for the eigenmode solver: a
/// Ritz value mu is accepted only when its residual bound
/// `beta_m * |y_last|` is below this fraction of `|mu|`. Unconverged Ritz
/// values otherwise surface as shift-tracking ghost modes (first seen on
/// rapidmesh cavity meshes, whose spectrum converges less uniformly than
/// gmsh's).
pub const LANCZOS_RITZ_TOL: f64 = 1e-6;

/// dB value reported for a far-field power at or below `SINGULAR_EPS`.
pub const FARFIELD_DB_FLOOR: f64 = -100.0;

/// Relative slack for the lumped-port voltage-integration axis projection.
pub const LUMPED_PORT_PROJ_EPS: f64 = 1e-9;

/// When building an orthonormal frame from a single axis, the global x hat is
/// used as the Gram-Schmidt reference unless the axis is too close to x. A
/// component magnitude above this (about 26 degrees off the axis) means x is
/// too parallel, so the global y hat is used instead, keeping the reference
/// well away from the axis and the cross product well conditioned.
pub const AXIS_REF_PARALLEL_MAX: f64 = 0.9;
