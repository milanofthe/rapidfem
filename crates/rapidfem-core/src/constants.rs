// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2026 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Physical constants and solver tolerances.
//!
//! The electromagnetic constants are SI / CODATA reference values; c₀ is the
//! exact SI definition and the rest follow from ε₀ = 1/(μ₀c₀²), Z₀ = μ₀c₀.
//! The numerical tolerances below are rapidfem-specific.

pub const C0: f64 = 299_792_458.0;                    // Speed of light (m/s), exact SI
pub const Z0: f64 = 376.73031366857;                   // Free-space impedance √(μ₀/ε₀) (Ω)
pub const EPS0: f64 = 8.854187818814e-12;              // Vacuum permittivity ε₀ (F/m)
pub const MU0: f64 = 1.0 / (C0 * C0 * EPS0);          // Vacuum permeability μ₀ = 1/(ε₀c₀²) (H/m)
pub const PI: f64 = std::f64::consts::PI;

// --- Numerical tolerances ---

/// Magnitudes below this are treated as zero. A determinant this small
/// marks a singular matrix or degenerate element (`1/det` would produce
/// NaN/inf); a power or norm this small is numerical noise. A uniform
/// "negligible" floor across the solver.
pub const SINGULAR_EPS: f64 = 1e-30;

/// Barycentric-coordinate slack for point-in-tetrahedron containment.
pub const POINT_IN_TET_EPS: f64 = 1e-8;

/// Normalized tet volume q = 6V / h_mean³ below which the element is treated as
/// numerically degenerate and its volume is floored so the assembly cannot emit
/// NaN/Inf. The conditioning of the canonical R2 element scales like (1/q)², so
/// q ≈ 1e-9 corresponds to cond ≳ 1/u (numerically singular); see
/// `derivations/conditioning/`.
pub const SLIVER_NORMVOL_FLOOR: f64 = 1e-9;

/// Normalized tet volume below which a mesh-load quality warning is emitted
/// (cond ≳ 1e12 — a real conditioning concern, well before the hard floor).
pub const SLIVER_NORMVOL_WARN: f64 = 1e-6;

/// Lanczos lucky-breakdown threshold for the eigenmode solver: the Krylov
/// subspace stops growing once the next vector's norm drops below this.
pub const LANCZOS_BREAKDOWN: f64 = 1e-12;

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
