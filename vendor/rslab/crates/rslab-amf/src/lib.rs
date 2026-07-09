//! Approximate Minimum Fill (AMF / HAMF4) fill-reducing ordering.
//!
//! Standalone clean-room implementation of the AMF quotient-graph
//! algorithm (Amestoy 1999 habilitation thesis, MUMPS HAMF4
//! variant). The metric is approximate fill rather than approximate
//! degree - the per-variable score is
//!
//! ```text
//! RMF(i) = ( deg(i) * (deg(i) - 1 + 2*degme) - WF(i) ) / (nv(i) + 1)
//! ```
//!
//! quantized into a `2N + 2` head-array with a coarse stride above
//! `N`. The inner loop, including the lazy `WF(e)` element cache,
//! the supervariable max-merge of `WF`, and the saturated/regular
//! RMF branch, lives in `rslab-ordering-core` behind the `MinFill`
//! `Metric` impl.
//!
//! The public surface conforms to the RSLAB ordering-crate
//! contract (`dev/plans/ordering-crate-contract.md`). `CscPattern`,
//! `OrderingStats`, `OrderingError`, and `CONTRACT_VERSION` are
//! re-exported from `rslab-ordering-core`.
//!
//! HAMF4 always aggressively absorbs elements, so [`AmfOptions`]
//! does not expose an `aggressive` knob; the `dense_alpha` knob
//! lives in the shared workspace and behaves identically to
//! `rslab-amd`'s.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod stats;

pub use rslab_ordering_core::{CscPattern, OrderingError, OrderingStats, CONTRACT_VERSION};
pub use stats::AmfStats;

use rslab_ordering_core::quotient_graph::{order, MinFill, WorkspaceOptions};
use std::time::Instant;

/// Tunable parameters for AMF ordering.
///
/// Defaults: `dense_alpha = 10.0`. HAMF4 always aggressively
/// absorbs absorbed elements during the inner Pass-2 loop, so
/// there is no `aggressive` knob.
#[derive(Debug, Clone)]
pub struct AmfOptions {
    /// Dense-row threshold multiplier. A variable with initial
    /// degree exceeding `min(max(16, floor(dense_alpha * sqrt(n))), n)`
    /// is deferred to the end of the ordering - the `max(16)` floor is
    /// applied before the `min(n)` cap, matching faer `amd.rs:173-179`.
    /// A negative value uses a raw threshold of `n - 2` with the same
    /// clamps; for `n >= 18` that is exactly `n - 2`, suppressing
    /// deferral for all but true hubs of degree `n - 1`.
    pub dense_alpha: f64,
}

impl Default for AmfOptions {
    fn default() -> Self {
        Self { dense_alpha: 10.0 }
    }
}

/// Compute a fill-reducing AMF ordering.
///
/// Returns a permutation `perm` (new-to-old) such that factoring
/// `P·A·Pᵀ` with `P[k] = perm[k]` produces less fill than the
/// natural ordering. The input must be the full symmetric pattern
/// (both halves present).
pub fn amf_order(pattern: &CscPattern<'_>) -> Result<Vec<i32>, OrderingError> {
    amf_order_opts(pattern, &AmfOptions::default()).map(|(perm, _)| perm)
}

/// Compute an AMF ordering and return the crate-specific diagnostic
/// counters.
///
/// See [`amf_order`] and [`AmfStats`]. Callers that also need the
/// shared [`OrderingStats`] (wall time, fill estimate) should use
/// [`amf_order_full`] instead.
pub fn amf_order_with_stats(
    pattern: &CscPattern<'_>,
) -> Result<(Vec<i32>, AmfStats), OrderingError> {
    amf_order_opts(pattern, &AmfOptions::default())
}

/// Compute an AMF ordering with explicit options.
///
/// Returns `(perm, amf_stats)`. See [`amf_order_full`] for the
/// contract-conforming three-tuple return.
pub fn amf_order_opts(
    pattern: &CscPattern<'_>,
    opts: &AmfOptions,
) -> Result<(Vec<i32>, AmfStats), OrderingError> {
    amf_order_full(pattern, opts).map(|(perm, _, amf_stats)| (perm, amf_stats))
}

/// Contract-conforming ordering producer.
///
/// Signature matches the shape every RSLAB ordering crate must
/// expose per `dev/plans/ordering-crate-contract.md`: input is a
/// full-symmetric [`CscPattern`] and options; output is a
/// three-tuple of `(perm, OrderingStats, crate-stats)`, with
/// errors in [`OrderingError`].
///
/// `OrderingStats.time_us` is the wall-clock time of this call.
/// `fill_estimate` and `flop_estimate` are left as `None` for AMF -
/// the per-crate [`AmfStats`] carries `ndiv` / `nms_lu` / `nms_ldl`
/// flop counters that may be surfaced here in a future revision
/// without bumping the contract.
pub fn amf_order_full(
    pattern: &CscPattern<'_>,
    opts: &AmfOptions,
) -> Result<(Vec<i32>, OrderingStats, AmfStats), OrderingError> {
    let t0 = Instant::now();
    let ws_opts = WorkspaceOptions {
        dense_alpha: opts.dense_alpha,
    };
    // HAMF4 always aggressively absorbs; pass `aggressive = true`
    // unconditionally. The flag only gates the AMD-specific
    // `(p3 == pn) && (elen[i] == 1)` mass-elimination branch which
    // the AMF loop reuses with the same semantics.
    let (perm, diag) = order::<MinFill>(pattern, &ws_opts, true)?;
    let amf_stats = AmfStats {
        ncmpa: diag.ncmpa,
        n_clear_flag: 0,
        n_mass_elim: diag.n_mass_elim,
        n_supervar_merge: diag.n_supervar_merge,
        n_dense_deferred: diag.ndense.max(0) as u32,
        ndiv: diag.flops.ndiv.max(0.0) as u64,
        nms_lu: diag.flops.nms_lu.max(0.0) as u64,
        nms_ldl: diag.flops.nms_ldl.max(0.0) as u64,
    };
    let ordering_stats = OrderingStats {
        time_us: t0.elapsed().as_micros() as u64,
        fill_estimate: None,
        flop_estimate: None,
    };
    Ok((perm, ordering_stats, amf_stats))
}
