// SPDX-License-Identifier: GPL-3.0-or-later
//
// Copyright (C) 2024-2025 Milan Rother and rapidfem contributors
//
// This file is part of rapidfem, distributed under GPL-3.0-or-later with
// the Gmsh additional permission. See LICENSE for the full terms.

//! Centralised numerical constants for the time-domain backend.
//!
//! Every tolerance and algorithmic threshold lives here — never inline,
//! never hardcoded at a call site. A solver constant that needs tuning is
//! tuned in one place. The numerical precision is centralised here too:
//! [`Field`] and [`Accum`] are the two scalar types the solver is written
//! against, so a precision change is a one-line edit, not a sweep.

// ── Numerical precision ───────────────────────────────────────────────────

/// Scalar type of the electromagnetic field state and the matrix-free
/// spatial operator ([`crate::rhs::MaxwellOperator`] and its `apply`).
///
/// This is the precision a future GPU / mixed-precision path may drop to
/// `f32`: the per-element curl + flux is the throughput-bound hot path and
/// the part a consumer GPU runs fastest in single precision. On the CPU it
/// is `f64` today. The operator-side data path — fields, fluxes, geometric
/// factors, material tensors — is meant to be typed `Field` end to end.
///
/// `Field` and [`Accum`] meet at the matvec boundary inside
/// [`crate::propagator::KrylovWorkspace::expmv_into`]: once they differ,
/// the matvec closure passed in owns the down/up-cast.
pub type Field = f64;

/// Scalar type of the Krylov accumulator: the Arnoldi basis, the CGS2
/// orthogonalisation, the Hessenberg matrix and the dense `expm`.
///
/// Stays `f64` — the scaling-and-squaring propagator and the
/// [`KRYLOV_TOL`]` = 1e-10` a-posteriori estimate depend on double
/// precision, so this is *not* a flip target. Typing the propagator
/// against `Accum` documents that boundary rather than enabling a change.
pub type Accum = f64;

/// Acceptable relative L2 error of the `Field`-precision (f32 / GPU) path
/// against the f64 reference. Every GPU validation gate checks against
/// this one value, so tuning the mixed-precision accuracy budget is a
/// one-line edit here.
///
/// Fixed from the `precision` example (the `examples/precision.rs` probe,
/// run once as an f64 reference and once as an f32 comparator): a single
/// f32 matvec drifts ~8e-6 from f64 (the DG flux jump `E_minus - E_plus`
/// amplifies f32 epsilon), and a 500-step f32 transient ~9e-5. `1e-3`
/// clears that with about a decade of headroom for GPU rounding-order
/// differences and longer runs, while staying far below any real-bug
/// signature (a wrong kernel diverges or gives O(1) error).
pub const GPU_REL_TOL: f64 = 1e-3;

// ── Krylov / ETD exponential propagator ───────────────────────────────────

/// Relative a-posteriori error tolerance for the Krylov exponential
/// propagator. After each Arnoldi vector the step's error estimate is
/// checked against this; the subspace stops growing once it converges, so
/// an easy step costs far fewer matvecs than the `krylov_dim` cap.
pub const KRYLOV_TOL: Accum = 1e-10;

/// Arnoldi lucky-breakdown threshold. When the next basis vector's norm
/// falls below this the Krylov subspace already spans the relevant
/// invariant subspace exactly and the iteration stops.
pub const ARNOLDI_BREAKDOWN: Accum = 1e-12;

/// Per-sub-step Krylov dimension cap for the GPU exponential propagator.
/// `exp(t*A)*v` for a requested dimension above this is computed by
/// sub-stepping (`exp(t*A) = exp((t/k)*A)^k`, exact), each sub-step a
/// small Krylov space. The device-resident Arnoldi basis is then capped
/// at `~KRYLOV_CHUNK * n_dof` instead of `~m * n_dof`, the dominant GPU
/// memory cost of the propagator, and the O(m^2) orthogonalisation work
/// drops with it.
pub const KRYLOV_CHUNK: usize = 12;

/// Minimum working-vector slice length per rayon task in the parallel CGS2
/// orthogonalisation. The chunk is sized from `n` so every core gets work
/// (see [`ARNOLDI_TASKS_PER_THREAD`]); this floor keeps a task above the
/// rayon dispatch overhead and clear of cache-line false sharing on `w`.
pub const ARNOLDI_MIN_CHUNK: usize = 256;

/// Target rayon tasks per worker thread for the CGS2 `w -= V·c` update — a
/// few-fold over-subscription so the work-stealing scheduler balances the
/// pass even at modest `n`. A fixed chunk size starved the pass on small
/// and medium meshes (only `⌈n/chunk⌉` tasks regardless of core count);
/// deriving the chunk from `n` and the thread count fixes that.
pub const ARNOLDI_TASKS_PER_THREAD: usize = 4;


// ── Dense matrix exponential (scaling-and-squaring) ───────────────────────

/// Scaling-and-squaring threshold: the matrix is halved until its
/// ∞-norm is at or below this before the Taylor core is summed.
pub const EXPM_SCALE_THRESHOLD: Accum = 0.5;

/// Taylor-series order for the dense `expm` core — the number of terms
/// summed after scaling. About 18 terms reach double precision once the
/// scaled norm is within [`EXPM_SCALE_THRESHOLD`].
pub const EXPM_TAYLOR_TERMS: usize = 18;

// ── Explicit low-storage Runge-Kutta (LSERK4) ─────────────────────────────

/// `a` coefficients of the Carpenter-Kennedy 5-stage 4th-order low-storage
/// Runge-Kutta scheme (LSERK4). The explicit alternative to the Krylov
/// exponential propagator: five matvecs per step, two state registers, and
/// — unlike the unconditionally stable exponential integrator — a CFL step
/// limit set by the spectral radius of the operator. Standard nodal-DG
/// integrator (Hesthaven & Warburton, *Nodal DG Methods*).
pub const LSERK4_A: [Field; 5] = [
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
];

/// `b` coefficients of LSERK4 — the per-stage update weights. Paired with
/// [`LSERK4_A`]; see there.
pub const LSERK4_B: [Field; 5] = [
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
];

/// LSERK4 stage count — five stages reaching fourth order.
pub const LSERK4_STAGES: usize = 5;

// ── Embedded adaptive RK: Kennedy-Carpenter-Lewis RK4(3)5[2R+]C ────────────
//
// Kennedy, Carpenter & Lewis 2000, "Low-storage, explicit Runge-Kutta
// schemes for the compressible Navier-Stokes equations" (Appl. Num. Math.
// 35, 177-219). The "2R+" form needs two state-shaped registers like
// [`LSERK4_A`] plus one extra accumulator for the embedded error vector —
// the third register goes to nothing else, only the adaptive controller. A
// step is five matvecs (same as LSERK4) but yields a fourth-order solution
// *and* a third-order embedded estimate, so the step-size controller can
// detect non-normal growth and cut the step without consulting `cfl_dt`.
//
// Coefficients are the exact rationals from the KCL 2000 paper, Table 1
// (RK4(3)5[2R+]C variant), as tabulated in NodePy's `low_storage_rk.py`
// (https://github.com/ketch/nodepy, file `low_storage_rk.py`, identifier
// `RK4(3)5[2R+]C`). Storing `B_HAT` directly — and computing the embedded
// error weight `e_i = B_HAT_i - B_i` at use — keeps the table verifiable
// against the paper at a glance.

/// `â_i = A_{i+1,i} - b_i` for KCL RK4(3)5[2R+]C — the four post-stage
/// updates that march the stage register `S2` to the next evaluation point
/// in Ketcheson's van-der-Houwen 2R form (Ketcheson 2010, JCP 229, Alg. 2).
/// Only `s - 1 = 4` values: the final stage closes the step without
/// updating `S2`. Sourced verbatim from NodePy's `a` array for
/// `RK4(3)5[2R+]C`.
pub const KCL_A: [Field; 4] = [
    970286171893.0 / 4311952581923.0,
    6584761158862.0 / 12103376702013.0,
    2251764453980.0 / 15575788980749.0,
    26877169314380.0 / 34165994151039.0,
];

/// `b_i` coefficients of KCL RK4(3)5[2R+]C — the per-stage main-solution
/// update weights (fourth-order accurate). Paired with [`KCL_A`].
pub const KCL_B: [Field; 5] = [
    1153189308089.0 / 22510343858157.0,
    1772645290293.0 / 4653164025191.0,
    -1672844663538.0 / 4480602732383.0,
    2114624349019.0 / 3568978502595.0,
    5198255086312.0 / 14908931495163.0,
];

/// `b̂_i` coefficients of KCL RK4(3)5[2R+]C — the per-stage *embedded*
/// (third-order) solution weights. The step-size controller never needs
/// `y_emb` itself, only its difference `y_emb - y_main = sum_i (b̂_i - b_i)
/// q_i`, accumulated into the embedded-error register at each stage.
pub const KCL_BHAT: [Field; 5] = [
    1016888040809.0 / 7410784769900.0,
    11231460423587.0 / 58533540763752.0,
    -1563879915014.0 / 6823010717585.0,
    606302364029.0 / 971179775848.0,
    1097981568119.0 / 3980877426909.0,
];

/// KCL RK4(3)5[2R+]C stage count — five stages, fourth-order main,
/// third-order embedded.
pub const KCL_STAGES: usize = 5;

// ── Waveguide ports ───────────────────────────────────────────────────────

/// In-plane radius below which a coaxial-port TEM profile is taken as zero.
///
/// The coax TEM field is `E_ρ ∝ 1/ρ` and diverges on the coax axis. A
/// meshed annulus never places a quadrature node on the axis, but a guard
/// keeps `e_profile` finite for any query point: a radius this small is
/// effectively on the (un-meshed) axis, off the physical port aperture, so
/// the profile is reported as zero rather than blowing up.
pub const COAX_RADIUS_FLOOR: Field = 1e-12;

// ── Matrix-free operator (apply) ──────────────────────────────────────────

/// Target rayon tasks per worker thread for the matrix-free `apply`'s
/// element loop. Elements are dealt out in this many contiguous runs per
/// thread: enough to let the work-stealing scheduler balance an uneven
/// mesh, few enough that neighbouring threads do not false-share the `dy`
/// cache lines at the run boundaries. A per-element chunk regressed past
/// roughly six cores; a contiguous run per task restores the scaling.
pub const APPLY_TASKS_PER_THREAD: usize = 4;

// ── Periodic boundary matcher ─────────────────────────────────────────────

/// Relative tolerance (fraction of the period translation magnitude) used
/// when matching a side-A boundary triangle to its side-B partner across a
/// periodic translation. Two triangles are considered partners when their
/// centroids agree (after translation) to within `PERIODIC_MATCH_REL_TOL`
/// times the period magnitude. A meshing-symmetric pair lines up to
/// machine precision; this loose-ish tolerance keeps the matcher robust to
/// gmsh's floating-point round-off without ever wandering onto a wrong
/// partner (the next nearest triangle on a structured face is one cell
/// size away, orders of magnitude beyond this).
pub const PERIODIC_MATCH_REL_TOL: Field = 1e-9;

/// Absolute fallback (in mesh length units) for the periodic-triangle
/// match tolerance when the period magnitude is zero, degenerate, but
/// guarded so the matcher's tolerance never collapses to zero. A pair this
/// close in absolute terms is effectively coincident.
pub const PERIODIC_MATCH_ABS_FLOOR: Field = 1e-12;

// ── Macromodel / model-order reduction ────────────────────────────────────

/// Block-Krylov deflation tolerance: a candidate block column whose
/// residual norm after orthogonalisation falls below this is dropped (its
/// direction is already in the subspace) and the build proceeds with
/// one column fewer. Distinct from [`KRYLOV_TOL`] (the propagator
/// a-posteriori estimate) and [`ARNOLDI_BREAKDOWN`] (the single-vector
/// exponential-propagator breakdown), because the block-Krylov
/// deflation is a different event, measured on the raw residual norm
/// of a candidate block column rather than the matvec a-posteriori
/// estimate.
pub const MACROMODEL_DEFLATION_TOL: Accum = 1e-10;

/// Default block-Krylov order `r` for the macromodel build when the
/// caller does not pick one explicitly. Tens to a few hundred is the
/// regime the impulse-Krylov macromodel lives in (see
/// `docs/td-macromodel-plan.md`); 80 is a reasonable midpoint that
/// covers a handful of resonances on a matched two-port test case.
/// Separate from [`KRYLOV_CHUNK`] (the per-sub-step exponential cap),
/// since the macromodel build is one-shot and not sub-stepped.
pub const MACROMODEL_DEFAULT_R: usize = 80;

/// Stride of the interleaved `[E, H]` block in the time-domain state
/// layout `y[(e*Np + i)*6 + c]`: components `c = 0..3` are the three
/// electric-field components at the node, components `c = 3..6` the
/// three magnetic-field components. Used by the SPRIM-style E/H mask
/// in [`crate::macromodel`] to project a state vector onto its E-part
/// or H-part for structure-preserving block-Krylov.
pub const TD_STATE_BLOCK_STRIDE: usize = 6;

/// Relative residual tolerance for the matrix-free complex GMRES that
/// solves the shifted system `(sigma I - A) x = b` inside the
/// shift-invert macromodel build
/// ([`crate::macromodel::MacroModel::build_shift_invert`]). Loose by
/// design: the shift-invert step's output is then re-orthogonalised
/// against the basis, so a GMRES residual at `~1e-6` is amply tighter
/// than the orthogonalisation drift.
pub const GMRES_SHIFT_TOL: Accum = 1.0e-6;

/// Maximum GMRES iterations per shift-invert solve before restart.
/// Larger keeps the Arnoldi basis bigger (more memory, fewer restarts);
/// smaller restarts more often and costs more matvecs at hard
/// frequencies. 60 is a healthy default for the small spectral region
/// around a single shift in a Maxwell macromodel.
pub const GMRES_SHIFT_MAX_ITER: usize = 60;
