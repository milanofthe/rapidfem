//! Quotient-graph workspace and initialization, shared by AMD-family
//! bottom-up orderings.
//!
//! All arrays follow the faer / SuiteSparse naming convention and use
//! signed `i32` so we can reserve negative values for sentinels via
//! [`flip`]: `flip(x) = -2 - x`. The input pattern is also `i32`-
//! indexed (per the ordering-crate contract), so workspace ingestion
//! converts to `usize` only for Rust slice addressing.
//!
//! Builds the workspace arrays from a full-symmetric CSC pattern,
//! runs the two initialization fast paths (zero-degree
//! pre-elimination and dense-deferred bucket), and seats the
//! remaining variables into degree-indexed linked lists ready for
//! the elimination loop.
//!
//! The workspace is shared by `rslab-amd` and `rslab-amf`; the
//! AMD-vs-AMF delta lives entirely in the elimination metric.
//
// Not every field or helper is read by every elimination loop.
#![allow(dead_code)]

use super::WorkspaceOptions;
use crate::{CscPattern, OrderingError};

/// Sentinel for "no index" in the `i32` arrays.
pub const NONE: i32 = -1;

/// Sentinel encoding used by the quotient graph: `flip(x) = -2 - x`.
/// Used to mark absorbed elements (`pe[e] < 0` => `flip(parent)`) and
/// as a tag on `elen` for freshly eliminated zero-degree variables.
#[inline(always)]
pub fn flip(x: i32) -> i32 {
    -2 - x
}

/// Reset the generation counter `wflg` without visiting `iw`.
///
/// Called when `wflg` would overflow the `wbig` ceiling, or when it
/// drops below 2 (which should not happen in practice, but matches
/// SuiteSparse AMD's defensive check). All nonzero entries in `w`
/// are clamped to `1`, and `2` is returned as the fresh counter.
///
/// Reference: `amd.rs:130-143`.
#[inline]
pub fn clear_flag(wflg: i32, wbig: i32, w: &mut [i32]) -> i32 {
    if wflg < 2 || wflg >= wbig {
        for x in w.iter_mut() {
            if *x != 0 {
                *x = 1;
            }
        }
        return 2;
    }
    wflg
}

/// In-memory workspace for one AMD run.
///
/// The fields mirror faer's `amd_2` locals. Ownership of every buffer
/// is held here so the elimination loop can borrow them concurrently
/// through split borrows without reallocation.
#[derive(Debug)]
pub struct Workspace {
    pub n: usize,
    pub iwlen: usize,
    pub pfree: usize,
    pub iw: Vec<i32>,

    pub pe: Vec<i32>,
    pub len: Vec<i32>,
    pub nv: Vec<i32>,
    pub elen: Vec<i32>,
    pub degree: Vec<i32>,
    pub w: Vec<i32>,
    pub head: Vec<i32>,
    pub next: Vec<i32>,
    pub last: Vec<i32>,
    /// Per-supervariable / per-element fill-score scratch used by the
    /// AMF inner loop only. Length `n`. Ignored by the AMD path
    /// (`rslab-amd` never reads or writes it). For variable indices
    /// `i`, holds the running quantized RMF score; for element indices
    /// `e`, holds the lazily-cached `dext * (2*deg(e) - dext - 1)`
    /// surface contribution (sentinel `0` = "first touch this iter").
    ///
    /// `i64` (not `i32`): the un-quantized surface contribution has
    /// both factors `O(n)`, so it reaches ~`n^2` and overflows `i32`
    /// for `n` >~ 46k before being consumed as `f64` in the RMF score.
    /// HAMF4 computes
    /// the RMF in DBLE for the same reason. The post-quantization RMF
    /// score stored here later is bounded by `i32::MAX - 1`.
    pub wf: Vec<i64>,

    /// Generation counter for the mark array `w`.
    pub wflg: i32,
    /// Overflow ceiling for `wflg`: `i32::MAX - n`.
    pub wbig: i32,
    /// Largest element size encountered so far - used by supervariable
    /// detection to bump `wflg` safely.
    pub lemax: i32,
    /// Lower bound on the next pivot's degree. Monotone non-decreasing.
    pub mindeg: usize,
    /// Number of garbage-collection compactions so far.
    pub ncmpa: u32,
    /// Supervariables eliminated so far (pivoted OR dense-deferred).
    pub nel: usize,
    /// Dense-deferred supervariable count.
    pub ndense: i32,
    /// Variables folded into a concurrent pivot by mass elimination.
    pub n_mass_elim: u32,
    /// Supervariable merges detected during indistinguishable-variable
    /// consolidation.
    pub n_supervar_merge: u32,
}

impl Workspace {
    /// Build a workspace from a full-symmetric CSC pattern and run
    /// initialization. On return, all variables have been classified
    /// into one of three buckets:
    ///
    /// 1. **Zero-degree** (`deg == 0`) - pre-eliminated. `pe[i] = NONE`,
    ///    `elen[i] = flip(1)`, `w[i] = 0`, `nel` incremented.
    /// 2. **Dense-deferred** (`deg > dense`) - moved to the dense tail.
    ///    `pe[i] = NONE`, `nv[i] = 0`, `elen[i] = NONE`, `nel` incremented.
    /// 3. **Live** - inserted LIFO into the degree-indexed linked list
    ///    headed by `head[deg]`, threaded through `next`/`last`.
    ///
    /// `pattern` must be the full-symmetric graph (both halves). The
    /// diagonal is ignored if present.
    pub fn new(
        pattern: &CscPattern<'_>,
        opts: &WorkspaceOptions,
    ) -> Result<Workspace, OrderingError> {
        Self::new_with_n_buckets(pattern, opts, pattern.n)
    }

    /// Variant of [`Workspace::new`] that allocates `head` with the
    /// caller-supplied bucket count. Used by AMF, where the quantized
    /// fill score can exceed `n` and the head array must extend up to
    /// `NBBUCK + 1 = 2 * n + 1`. AMD always passes `pattern.n`, which
    /// makes this byte-equivalent to [`Workspace::new`].
    ///
    /// `n_buckets` must be at least `n` so the init insertion at
    /// `head[deg]` (with `deg <= dense <= n`) is in range.
    pub fn new_with_n_buckets(
        pattern: &CscPattern<'_>,
        opts: &WorkspaceOptions,
        n_buckets: usize,
    ) -> Result<Workspace, OrderingError> {
        let n = pattern.n;
        debug_assert!(n_buckets >= n, "n_buckets must cover deg in [0, n)");

        // i32 addressing requires n < i32::MAX. The algorithm also
        // stores `pfree` as i32 via `pe[i]`, so iwlen must fit.
        if n >= i32::MAX as usize {
            return Err(OrderingError::IndexOverflow);
        }

        // Count off-diagonal entries per column.
        let mut len: Vec<i32> = vec![0; n];
        let mut nzaat: usize = 0;
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            let j_i32 = j as i32;
            let start = pattern.col_ptr[j] as usize;
            let end = pattern.col_ptr[j + 1] as usize;
            let mut cnt: usize = 0;
            for &r in &pattern.row_idx[start..end] {
                if r != j_i32 {
                    cnt += 1;
                }
            }
            len[j] = cnt as i32;
            nzaat += cnt;
        }

        // iwlen = nzaat + nzaat/5 + n  (faer amd.rs:921-924).
        let iwlen = nzaat
            .checked_add(nzaat / 5)
            .and_then(|s| s.checked_add(n))
            .ok_or(OrderingError::IndexOverflow)?;
        if iwlen > i32::MAX as usize {
            return Err(OrderingError::IndexOverflow);
        }
        // iw needs at least one slot even when n==0 so `pfree` is
        // addressable; we allocate exactly iwlen.
        let mut iw: Vec<i32> = vec![0; iwlen];

        // pe[j] = start of j's adjacency list in iw; fill iw with
        // off-diagonals, in the order they appear in the CSC pattern.
        let mut pe: Vec<i32> = vec![0; n];
        let mut pfree: usize = 0;
        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            pe[j] = pfree as i32;
            let j_i32 = j as i32;
            let start = pattern.col_ptr[j] as usize;
            let end = pattern.col_ptr[j + 1] as usize;
            for &r in &pattern.row_idx[start..end] {
                if r != j_i32 {
                    iw[pfree] = r;
                    pfree += 1;
                }
            }
        }
        debug_assert_eq!(pfree, nzaat);

        // Dense threshold. alpha < 0 disables dense deferral.
        let dense = if opts.dense_alpha < 0.0 {
            n.saturating_sub(2)
        } else {
            (opts.dense_alpha * (n as f64).sqrt()) as usize
        };
        let dense = dense.max(16).min(n);

        // Fixed-value arrays.
        let mut nv: Vec<i32> = vec![1; n];
        let mut elen: Vec<i32> = vec![0; n];
        let mut w: Vec<i32> = vec![1; n];
        let degree: Vec<i32> = len.clone();
        let mut head: Vec<i32> = vec![NONE; n_buckets];
        let mut next: Vec<i32> = vec![NONE; n];
        let mut last: Vec<i32> = vec![NONE; n];
        let mut wf: Vec<i64> = vec![0; n];

        let wbig = i32::MAX - n as i32;
        let wflg = 0; // clear_flag will lift to 2 on first use.

        let mut nel: usize = 0;
        let mut ndense: i32 = 0;

        // Classify each variable.
        for i in 0..n {
            let deg = degree[i] as usize;
            if deg == 0 {
                // Zero-degree fast path - pre-eliminated.
                elen[i] = flip(1);
                nel += 1;
                pe[i] = NONE;
                w[i] = 0;
            } else if deg > dense {
                // Dense-deferred fast path.
                ndense += 1;
                nv[i] = 0;
                elen[i] = NONE;
                pe[i] = NONE;
                nel += 1;
            } else {
                // LIFO head-insert at head[deg]. The AMF metric's
                // `bucket(deg, n)` is identity for `deg <= n`, so the
                // index `deg` is the right slot for both AMD and AMF
                // at init time. Seed the AMF fill score so the AMF
                // path can compute `bucket(wf[i], n)` consistently.
                let inext = head[deg];
                if inext != NONE {
                    last[inext as usize] = i as i32;
                }
                next[i] = inext;
                head[deg] = i as i32;
                wf[i] = deg as i64;
            }
        }

        Ok(Workspace {
            n,
            iwlen,
            pfree,
            iw,
            pe,
            len,
            nv,
            elen,
            degree,
            w,
            head,
            next,
            last,
            wf,
            wflg,
            wbig,
            lemax: 0,
            mindeg: 0,
            ncmpa: 0,
            nel,
            ndense,
            n_mass_elim: 0,
            n_supervar_merge: 0,
        })
    }
}
