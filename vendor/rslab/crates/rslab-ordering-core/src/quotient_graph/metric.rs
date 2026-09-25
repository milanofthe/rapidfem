//! Selection-metric trait abstracting AMD vs AMF differences.
//!
//! The shared quotient-graph machinery (`Workspace`, `select_pivot`,
//! `create_element`, `finalize_step`, hash-based supervariable
//! detection, mass elimination, aggressive absorption) is identical
//! across AMD and AMF. The metrics differ in:
//!
//! 1. **Initial score** seeded from each row's adjacency length.
//!    AMD: identity. AMF: identity (both = `len`).
//! 2. **Bucket array length.** AMD: `n` (degrees `0..n` indexable;
//!    `select_pivot` scans `while deg < n`). AMF: `2 * n + 2` because
//!    the quantized RMF can exceed `n`.
//! 3. **Bucket index for a score.** AMD: identity. AMF: identity for
//!    `s <= n`, then coarse stride `PAS = max(n / 8, 1)` above.
//! 4. **Pivot selection within a bucket.** AMD: head only.
//!    AMF: linear scan when the bucket is in the coarse region.
//! 5. **Score on supervariable merge.** AMD: no-op (only `nv[i]`
//!    accumulates). AMF: `score[i] = max(score[i], score[j])`.
//! 6. **Score finalisation** at the end of Pass-2: AMD's loose-degree
//!    formula `min(deg_prev, scan2_deg) + degme - nvi` clamped at
//!    `nleft - nvi`; AMF's quantized RMF (Amestoy 1999 thesis).
//!
//! The trait below covers (1)-(5). Site (6) is metric-specific in the
//! Pass-2 inner loop and is reached via the `run_elimination`
//! dispatch - each metric impl wires its own concrete loop in
//! `algo.rs` (AMD: `run_elimination`, AMF: `run_elimination_amf`).
//! The trait stays light: keeping the inner loops as parallel
//! concrete functions trades ~300 LoC duplication for zero risk to
//! AMD's bit-identity with SuiteSparse.
//!
//! Reference: Amestoy (1999) habilitation thesis; MUMPS HAMF4.

use super::algo::{run_elimination as run_elimination_amd, run_elimination_amf, StepFlops};
use super::workspace::Workspace;
use crate::OrderingError;

/// Selection metric for an AMD-family bottom-up ordering.
///
/// All methods are zero-overhead `#[inline(always)]` no-ops or
/// identity functions in the AMD case; AMF (`MinFill`) provides
/// non-trivial implementations. The trait is consumed at the
/// `run_elimination` dispatch point and at the bucket-allocation
/// dispatch in [`crate::quotient_graph::order`]; the metric-specific
/// inner-loop sites are inlined into the concrete `run_elimination_*`
/// functions in `algo.rs`.
pub trait Metric {
    /// Bucket key produced by the selection metric. AMD uses `i32`
    /// (the running degree); AMF will also use `i32` (quantized RMF).
    type Score: Copy + Ord + Default;

    /// Length of the bucket head array `Workspace::head`. AMD: `n`
    /// (indexed up to `n - 1` by `select_pivot`'s `while deg < n`).
    /// AMF: `2 * n + 2`.
    fn n_buckets(n: usize) -> usize;

    /// Initial score for a freshly-loaded variable with adjacency
    /// length `len`. AMD and AMF both seed `len`.
    fn init_score(len: i32) -> Self::Score;

    /// Bucket index for the given score. AMD: identity. AMF: identity
    /// for `s <= n`, coarse-stride above.
    fn bucket(score: Self::Score, n: usize) -> usize;

    /// Whether `idx` falls in the "coarse" bucket region - i.e.
    /// `select_pivot` must linear-scan the bucket chain to pick the
    /// minimum-score entry, rather than just taking the head. AMD
    /// always returns `false`; AMF returns `idx > n`.
    fn coarse_bucket(idx: usize, n: usize) -> bool;

    /// Update `parent`'s score on supervariable merge of `child` into
    /// `parent`. AMD: no-op. AMF: `*parent = max(*parent, child)`.
    fn merge_supervariable(parent: &mut Self::Score, child: Self::Score);

    /// Run the metric's elimination loop on a freshly initialised
    /// `Workspace`. Returns the accumulated flop counters.
    ///
    /// MinDegree dispatches to `run_elimination` (the AMD-specific
    /// loop); MinFill dispatches to `run_elimination_amf`.
    fn run_elimination(ws: &mut Workspace, aggressive: bool) -> Result<StepFlops, OrderingError>;
}

/// Minimum-degree metric - the AMD selection rule of Amestoy, Davis,
/// Duff (1996).
///
/// Score is the running degree. Bucket index is the score itself.
/// All buckets are "fine" (head-only pivot selection). Supervariable
/// merge does not update the score (AMD tracks degree only via
/// `nv[i]` and the per-iteration Pass-2 monotone cap).
#[derive(Debug, Clone, Copy, Default)]
pub struct MinDegree;

impl Metric for MinDegree {
    type Score = i32;

    #[inline(always)]
    fn n_buckets(n: usize) -> usize {
        n
    }

    #[inline(always)]
    fn init_score(len: i32) -> i32 {
        len
    }

    #[inline(always)]
    fn bucket(score: i32, _n: usize) -> usize {
        score as usize
    }

    #[inline(always)]
    fn coarse_bucket(_idx: usize, _n: usize) -> bool {
        false
    }

    #[inline(always)]
    fn merge_supervariable(_parent: &mut i32, _child: i32) {
        // AMD does not maintain a per-supervariable score; degree
        // bookkeeping flows entirely through `nv[i]` and the
        // re-insertion loop's loose-degree formula.
    }

    #[inline(always)]
    fn run_elimination(ws: &mut Workspace, aggressive: bool) -> Result<StepFlops, OrderingError> {
        run_elimination_amd(ws, aggressive)
    }
}

/// Approximate Minimum Fill metric (HAMF4) - Amestoy 1999 thesis.
///
/// AMF selects the next pivot to minimise the *fill* introduced by
/// the elimination, rather than the candidate's degree. On bipartite-
/// KKT graphs with a few "hub" rows AMF can give well over an order of
/// magnitude less `nnz_L` than AMD.
///
/// Score is a quantized `RMF = DEG*(DEG-1+2*DEGME) - WF(i)` value
/// stored in `i32`. Buckets up to and including `NORIG = n` are one
/// bucket per integer score; above `NORIG` the buckets quantize with
/// stride `PAS = max(n / 8, 1)` and the head must be linear-scanned
/// to pick the minimum-RMF entry. Supervariable absorption merges
/// the per-supervariable WF with `max`.
///
/// **Inner loop**: [`MinFill::run_elimination`] dispatches to
/// `run_elimination_amf`.
/// The lazy WF(e) cache, three-accumulator Pass-2, supervariable
/// max-merge of `wf`, saturated/regular RMF branch, and coarse-bucket
/// linear scan all live in `algo.rs`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MinFill;

impl Metric for MinFill {
    type Score = i32;

    /// AMF needs `2 * n + 2` slots: `0..=NORIG` for one-per-score
    /// fine buckets, `NORIG+1..=NBBUCK` (`NBBUCK = 2 * n`) for
    /// coarse-stride buckets, and one halo slot at `NBBUCK + 1`
    /// reserved for HAMF4's boundary (halo) variables, inert here
    /// because no halo is ever passed in.
    #[inline(always)]
    fn n_buckets(n: usize) -> usize {
        2 * n + 2
    }

    /// Seed the per-supervariable score with the row's adjacency
    /// length `len(i)`. Same seeding as AMD; the AMF metric only
    /// diverges once the elimination loop starts producing elements.
    #[inline(always)]
    fn init_score(len: i32) -> i32 {
        len
    }

    /// Quantize `score` into a bucket index.
    ///
    /// `0..=n` are fine buckets (one per integer score). Above `n`
    /// the buckets are coarse with stride `PAS = max(n / 8, 1)`,
    /// capped at `NBBUCK = 2 * n`. Negative scores (which should not
    /// occur in a well-formed AMF run, but we defend against
    /// truncation underflow on `RMF / (NVI + 1)`) clamp to bucket 0.
    ///
    /// Bucket layout follows the HAMF4 quantization described in
    /// Amestoy's 1999 habilitation thesis and used by MUMPS HAMF4.
    #[inline]
    fn bucket(score: i32, n: usize) -> usize {
        if score <= 0 {
            return 0;
        }
        let s = score as usize;
        if s <= n {
            return s;
        }
        let pas = (n / 8).max(1);
        let nbbuck = 2 * n;
        let coarse = (s - n) / pas + n;
        coarse.min(nbbuck)
    }

    /// Coarse buckets are those above `NORIG = n`. `select_pivot`
    /// must walk the bucket chain and pick the entry with the
    /// smallest *exact* score (coarse buckets only sort
    /// approximately; validated against the HAMF4 oracle).
    #[inline(always)]
    fn coarse_bucket(idx: usize, n: usize) -> bool {
        idx > n
    }

    /// On supervariable merge `j -> i`, update the surviving anchor's
    /// score with `max(WF(i), WF(j))` (the HAMF4 merge rule).
    #[inline(always)]
    fn merge_supervariable(parent: &mut i32, child: i32) {
        if child > *parent {
            *parent = child;
        }
    }

    fn run_elimination(ws: &mut Workspace, aggressive: bool) -> Result<StepFlops, OrderingError> {
        run_elimination_amf(ws, aggressive)
    }
}
