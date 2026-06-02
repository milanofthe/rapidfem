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

/// dB value reported for a far-field power at or below `SINGULAR_EPS`.
pub const FARFIELD_DB_FLOOR: f64 = -100.0;

/// Relative slack for the lumped-port voltage-integration axis projection.
pub const LUMPED_PORT_PROJ_EPS: f64 = 1e-9;

// --- Schur-complement domain decomposition (FD solver, issue #12) ---

/// Interface-DOF count above which the Schur driver switches from forming the
/// dense interface matrix S explicitly (one back-solve per interface column,
/// O(n_iface) solves) to a matrix-free interface GMRES. Below it the dense
/// path is cheaper and exact; above it the explicit S is prohibitive in both
/// time and memory.
pub const SCHUR_EXPLICIT_INTERFACE_MAX: usize = 2000;

/// Relative residual tolerance for the matrix-free interface GMRES.
pub const SCHUR_GMRES_TOL: f64 = 1e-9;

/// GMRES restart length (Krylov dimension per cycle) for the interface solve.
pub const SCHUR_GMRES_RESTART: usize = 100;

/// Maximum GMRES restart cycles for the interface solve.
pub const SCHUR_GMRES_MAX_OUTER: usize = 40;
