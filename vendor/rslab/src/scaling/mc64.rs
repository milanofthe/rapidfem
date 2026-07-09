//! MC64 wrapper: input preprocessing, Hungarian call, symmetric
//! averaging, and output guards.
//!
//! Given a sparse symmetric matrix (lower triangle only in the
//! input CSC), this module produces a symmetric scaling vector
//! `s` such that `D · A · D` (with `D = diag(s)`) has
//! magnitude-bounded off-diagonals and unit-scale diagonals.
//!
//! Algorithm (mirrors `ref/spral/src/scaling.f90::hungarian_wrapper`,
//! lines 597-801, in its non-singular branch):
//!
//!   1. Expand the lower-triangle CSC to a full symmetric pattern,
//!      carrying the original values with the transpose entries.
//!   2. Drop explicit zero entries (log of zero is -∞).
//!   3. Compute `c[k] = log |a[k]|` on the remaining entries.
//!   4. For each column j, compute `cmax[j] = max_k c[k]` and replace
//!      each `c[k]` by `cmax[j] - c[k]`. The cost graph is now
//!      non-negative and has minimum 0 in each column.
//!   5. Run `hungarian_match` on the cost graph.
//!   6. Unwind the normalization (SPRAL scaling.f90:681-682):
//!      - `rscaling[i] = u[i]` (row dual unchanged)
//!      - `cscaling[j] = v[j] - cmax[j]` (column dual minus column max)
//!   7. Symmetric average (SPRAL scaling.f90:169):
//!      `s[i] = exp((rscaling[i] + cscaling[i]) / 2)`.
//!   8. Safety guards: clamp the exponent to avoid overflow,
//!      rewrite any `s[i] == 0` or non-finite result to `1.0`.
//!   9. On partial matching, set `s[i] = 1.0` for any index whose row
//!      OR column is unmatched (the two sets can differ even on a
//!      symmetric pattern) and return
//!      `ScalingInfo::PartialSingular { n_unmatched }`.
//!
//! The partial-singular path deviates from SPRAL, which runs a
//! second Hungarian pass on the full-rank submatrix and then
//! applies a Duff-Pralet correction (scaling.f90:688-800). The
//! research note `dev/research/mc64-scaling.md` §"Structurally
//! singular matrices" specifies identity fallback for unmatched
//! rows/columns as the correct behavior for rslab, because KKT
//! matrices from IPOPT are occasionally structurally rank-deficient
//! and a hard failure would regress the current `ForceAccept`
//! pathway.

use super::hungarian::{hungarian_match, CostGraph, Matching};
use super::ScalingInfo;
use crate::error::RslabError;
use crate::sparse::csc::CscMatrix;

/// Upper bound on the argument to `exp` before overflow.
/// `ln(f64::MAX) ≈ 709.78`. We use 709.0 as a safe ceiling.
const LOG_HUGE: f64 = 709.0;

/// Cached MC64 output: the full Hungarian matching plus the
/// column-max normalization, from which the scaling vector can be
/// recovered without rerunning the expensive Hungarian kernel.
///
/// Populated by [`compute_matching`] when `LdltCompress` preprocessing
/// runs; consumed by [`scaling_from_cache`] in the numeric phase when
/// the caller's scaling strategy resolves to `Mc64Symmetric`. Moves
/// the ~70% of symbolic overhead (Hungarian + cost graph build) off
/// the critical path for matrices where both compression and MC64
/// scaling run.
#[derive(Debug, Clone)]
pub(crate) struct Mc64Cache {
    pub perm: Vec<usize>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub cmax: Vec<f64>,
    pub n_matched: usize,
}

/// Probe counter: number of times `compute_matching` ran the full
/// Hungarian pipeline. Process-global; readable from any thread.
/// Used by the value-bounded-cache investigation (see
/// `dev/research/mc64-value-bounded-cache-2026-05-17.md`) to confirm
/// whether MC64 dominates warm IPM wall in the live ipopt-rslab path,
/// not just on the pounce-dumped corpus. Set `RSLAB_MC64_TRACE=1` in
/// the environment to also stream per-call wall time to stderr.
pub static MC64_RECOMPUTE_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Run the expensive MC64 pipeline - build the cost graph and run
/// the Hungarian kernel - and return the full output. The cheap
/// scaling-vector post-processing is in [`scaling_from_cache`].
pub(crate) fn compute_matching(matrix: &CscMatrix) -> Result<Mc64Cache, RslabError> {
    let n = matrix.n;
    if n == 0 {
        return Ok(Mc64Cache {
            perm: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            cmax: Vec::new(),
            n_matched: 0,
        });
    }
    MC64_RECOMPUTE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let trace = matches!(
        std::env::var("RSLAB_MC64_TRACE").as_deref(),
        Ok("1") | Ok("on")
    );
    let t0 = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };

    let (cost_graph, cmax) = build_cost_graph(matrix)?;
    let Matching {
        perm,
        u,
        v,
        n_matched,
    } = hungarian_match(&cost_graph);

    if let Some(t0) = t0 {
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let count = MC64_RECOMPUTE_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "[rslab mc64] call #{} n={} nnz={} matching {:.1} ms",
            count,
            n,
            matrix.row_idx.len(),
            ms,
        );
    }

    Ok(Mc64Cache {
        perm,
        u,
        v,
        cmax,
        n_matched,
    })
}

pub(crate) fn compute_symmetric(matrix: &CscMatrix) -> Result<(Vec<f64>, ScalingInfo), RslabError> {
    let cache = compute_matching(matrix)?;
    Ok(scaling_from_cache(&cache))
}

/// Cheap O(n) post-processing that turns a cached MC64 matching into
/// the symmetric scaling vector. Mirrors the body of
/// [`compute_symmetric`] from step 6 onward.
pub(crate) fn scaling_from_cache(cache: &Mc64Cache) -> (Vec<f64>, ScalingInfo) {
    let n = cache.perm.len();
    if n == 0 {
        return (Vec::new(), ScalingInfo::Applied);
    }
    let Mc64Cache {
        perm,
        u,
        v,
        cmax,
        n_matched,
    } = cache;

    // Step 6-7: unwind normalization and form the symmetric average.
    //
    //   rscaling[i] = u[i]
    //   cscaling[i] = v[i] - cmax[i]
    //   s[i]        = exp((rscaling[i] + cscaling[i]) / 2)
    //               = exp((u[i] + v[i] - cmax[i]) / 2)
    //
    // Matches SPRAL scaling.f90:681-682 followed by :169.
    //
    // The matched-ROW set: `perm[j]` is the row matched to column `j`, so a
    // row appears here iff some column matched it. On a partial matching the
    // matched-row and matched-column sets can differ even for a symmetric
    // pattern (index i may have its column matched while its row is
    // unmatched), so both must be consulted below (X4).
    let mut row_matched = vec![false; n];
    for &r in perm.iter() {
        if r != usize::MAX {
            row_matched[r] = true;
        }
    }

    let mut scaling = vec![1.0_f64; n];
    for i in 0..n {
        // If the column had no usable entries at all, cmax[i] is
        // `f64::NEG_INFINITY` (see `build_cost_graph`). Any such
        // index is "empty" - the Hungarian kernel cannot match
        // that column meaningfully - so we fall back to identity
        // scaling for it. This is the structurally empty-column
        // case from the research note.
        if !cmax[i].is_finite() {
            scaling[i] = 1.0;
            continue;
        }

        // For an unmatched column (`perm[i] == MAX`) OR an unmatched row
        // (`!row_matched[i]`), fall back to identity scaling rather than
        // using the dual variables, which are meaningless on the unmatched
        // part of the graph. The row check is what fixes X4: a matched column
        // with an unmatched row has `u[i]` zeroed by `build_matching`, so the
        // symmetric average would otherwise fold a meaningless zero half-dual
        // into `s[i]`.
        if perm[i] == usize::MAX || !row_matched[i] {
            scaling[i] = 1.0;
            continue;
        }

        let mut arg = (u[i] + v[i] - cmax[i]) / 2.0;

        // Clamp to avoid overflow on `exp`. A dual variable can
        // grow to ±∞-ish magnitudes on pathological inputs; both
        // MUMPS (dana_aux.F:1797-1816) and SSIDS guard against
        // this. The clamp is symmetric so that a clamped row
        // exponentiates to a very large or very small but finite
        // value rather than `+∞` or `0`.
        if !arg.is_finite() {
            scaling[i] = 1.0;
            continue;
        }
        arg = arg.clamp(-LOG_HUGE, LOG_HUGE);

        let s = arg.exp();

        // Defensive rewrite: a zero or non-finite scaling would
        // annihilate a whole row/column and destroy symmetry.
        // Mirrors MUMPS dana_aux.F:1809-1811.
        if s == 0.0 || !s.is_finite() {
            scaling[i] = 1.0;
        } else {
            scaling[i] = s;
        }
    }

    let info = if *n_matched == n {
        ScalingInfo::Applied
    } else {
        ScalingInfo::PartialSingular {
            n_unmatched: n - *n_matched,
        }
    };

    (scaling, info)
}

/// Build the Hungarian cost graph and per-column maximum (`cmax`).
///
/// Expands the lower-triangle CSC `matrix` to a full symmetric
/// pattern, drops explicit-zero entries, takes the log of the
/// absolute value of each remaining entry, and normalizes each
/// column by subtracting its maximum so that the resulting costs
/// are non-negative.
///
/// Returns `(CostGraph, cmax)` where `cmax[j]` is the pre-
/// normalization column maximum (i.e., `max_i log|a[i,j]|`) used
/// in step 6 of `compute_symmetric` to unwind the normalization.
/// Columns with no finite (non-zero) entries have
/// `cmax[j] = f64::NEG_INFINITY`, which the caller treats as a
/// "fall back to identity" signal.
///
/// Algorithmic mirror: `ref/spral/src/scaling.f90:636-657`.
fn build_cost_graph(matrix: &CscMatrix) -> Result<(CostGraph, Vec<f64>), RslabError> {
    let n = matrix.n;

    // Two-pass expansion: first count the non-zero entries per
    // expanded column, then fill in the rows and values.
    //
    // For each stored lower-triangle entry at (row=i, col=j):
    //   * if val != 0 and i == j: contributes one entry to column j.
    //   * if val != 0 and i > j:  contributes to both column j (row i)
    //                             and column i (row j).
    //
    // Zero entries are dropped at the counting step so `log 0`
    // never appears.
    let mut col_counts = vec![0usize; n];
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let val = matrix.values[k];
            if val == 0.0 {
                continue;
            }
            col_counts[j] += 1;
            if i != j {
                col_counts[i] += 1;
            }
        }
    }

    // Prefix sum to column pointers.
    let mut col_ptr = vec![0usize; n + 1];
    for j in 0..n {
        col_ptr[j + 1] = col_ptr[j] + col_counts[j];
    }
    let nnz_full = col_ptr[n];

    let mut row_idx = vec![0usize; nnz_full];
    let mut cost = vec![0.0_f64; nnz_full];
    let mut offsets: Vec<usize> = col_ptr[..n].to_vec();

    // Second pass: place entries.
    for j in 0..n {
        for k in matrix.col_ptr[j]..matrix.col_ptr[j + 1] {
            let i = matrix.row_idx[k];
            let val = matrix.values[k];
            if val == 0.0 {
                continue;
            }
            let logabs = val.abs().ln();
            // (i, j) stays in column j.
            let p = offsets[j];
            row_idx[p] = i;
            cost[p] = logabs;
            offsets[j] += 1;
            // (j, i) transpose entry, if off-diagonal.
            if i != j {
                let q = offsets[i];
                row_idx[q] = j;
                cost[q] = logabs;
                offsets[i] += 1;
            }
        }
    }

    // Sort each column's rows ascending (Hungarian kernel does not
    // strictly require this, but a predictable order makes the
    // greedy initialization deterministic and matches SPRAL's
    // behaviour after `half_to_full`).
    for j in 0..n {
        let start = col_ptr[j];
        let end = col_ptr[j + 1];
        let mut pairs: Vec<(usize, f64)> = (start..end).map(|k| (row_idx[k], cost[k])).collect();
        pairs.sort_by_key(|&(r, _)| r);
        for (k, (r, c)) in (start..end).zip(pairs) {
            row_idx[k] = r;
            cost[k] = c;
        }
    }

    // Column-max normalization: for each column, find the maximum
    // log-absolute value and subtract it from every entry in that
    // column. Entries of an all-zero column are absent, so the
    // `cmax` for an empty column is `f64::NEG_INFINITY` and its
    // range is already empty - nothing to normalize.
    let mut cmax = vec![f64::NEG_INFINITY; n];
    for j in 0..n {
        let start = col_ptr[j];
        let end = col_ptr[j + 1];
        if start == end {
            continue;
        }
        let mut m = cost[start];
        for &c in &cost[(start + 1)..end] {
            if c > m {
                m = c;
            }
        }
        cmax[j] = m;
        for c in &mut cost[start..end] {
            *c = m - *c;
        }
    }

    let graph = CostGraph {
        n,
        col_ptr,
        row_idx,
        cost,
    };
    Ok((graph, cmax))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Diagonal SPD: expansion is a no-op, cost is all zero after
    /// normalization, Hungarian returns identity matching with
    /// `u = v = 0`, unwinding gives `scaling[i] = exp(-log A_ii / 2)
    /// = 1/sqrt(A_ii)`, and the scaled diagonal is 1.
    #[test]
    fn diagonal_matrix_produces_inverse_sqrt_scaling() {
        let csc = CscMatrix::from_triplets(3, &[0, 1, 2], &[0, 1, 2], &[2.0, 3.0, 5.0]).unwrap();
        let (s, info) = compute_symmetric(&csc).unwrap();
        assert_eq!(info, ScalingInfo::Applied);
        let expected = [
            1.0 / 2.0_f64.sqrt(),
            1.0 / 3.0_f64.sqrt(),
            1.0 / 5.0_f64.sqrt(),
        ];
        for i in 0..3 {
            assert!(
                (s[i] - expected[i]).abs() < 1e-12,
                "s[{}] = {}, expected {}",
                i,
                s[i],
                expected[i]
            );
        }
    }

    /// X4 (dev/research/repo-review-2026-06-09.md): on a partial matching an
    /// index `i` can have its COLUMN matched while its ROW is unmatched - the
    /// matched-row and matched-column sets differ even on symmetric patterns.
    /// `build_matching` zeroes `u[i]` for an unmatched row, so the symmetric
    /// average `s[i] = exp((u[i] + v[i] - cmax[i]) / 2)` folds a meaningless
    /// zero half-dual into the scaling - exactly the "duals are meaningless on
    /// the unmatched part" condition the adjacent comments warn about. The
    /// documented contract (step 9 in the module header,
    /// `dev/research/mc64-scaling.md`) is identity scaling for any index whose
    /// row OR column is unmatched.
    ///
    /// Synthetic cache for n = 2: column 0 is matched to row 1
    /// (`perm[0] = 1`), column 1 is unmatched (`perm[1] = usize::MAX`). The
    /// matched-row set is therefore {1}, so ROW 0 is unmatched and `u[0] = 0`
    /// (as `build_matching` leaves it). Both columns are non-empty (finite
    /// `cmax`). The contract requires `s[0] = 1.0` (row 0 unmatched) and
    /// `s[1] = 1.0` (column 1 unmatched). Pre-fix the code only skipped on an
    /// unmatched COLUMN, so index 0 took `exp((0 + v[0] - cmax[0]) / 2)
    /// = exp((0 + 2 - 1) / 2) = exp(0.5) ≈ 1.6487` - the witness.
    #[test]
    fn unmatched_row_with_matched_column_falls_back_to_identity() {
        let cache = Mc64Cache {
            // col 0 -> row 1; col 1 unmatched.
            perm: vec![1, usize::MAX],
            // row 0 unmatched => u[0] zeroed by build_matching; row 1's dual
            // is irrelevant to index 0's scaling.
            u: vec![0.0, 0.0],
            // col 0 matched dual; col 1 unmatched => 0.
            v: vec![2.0, 0.0],
            // both columns non-empty (finite max).
            cmax: vec![1.0, 1.0],
            n_matched: 1,
        };
        let (s, info) = scaling_from_cache(&cache);
        assert_eq!(info, ScalingInfo::PartialSingular { n_unmatched: 1 });
        assert!(
            (s[0] - 1.0).abs() < 1e-12,
            "index 0's ROW is unmatched; the contract requires identity \
             scaling, got {} (X4)",
            s[0]
        );
        assert!(
            (s[1] - 1.0).abs() < 1e-12,
            "index 1's column is unmatched; must be identity, got {}",
            s[1]
        );
    }

    /// Empty 0×0 matrix returns an empty scaling vector.
    #[test]
    fn empty_matrix_returns_empty_scaling() {
        let csc = CscMatrix {
            n: 0,
            col_ptr: vec![0],
            row_idx: vec![],
            values: vec![],
        };
        let (s, info) = compute_symmetric(&csc).unwrap();
        assert!(s.is_empty());
        assert_eq!(info, ScalingInfo::Applied);
    }
}
