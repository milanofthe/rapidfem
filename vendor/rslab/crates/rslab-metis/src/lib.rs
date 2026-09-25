//! Multilevel nested-dissection fill-reducing ordering.
//!
//! Clean-room Rust implementation of the algorithm described in
//! Karypis & Kumar, "A Fast and High Quality Multilevel Scheme for
//! Partitioning Irregular Graphs" (SIAM J. Sci. Comput., 1998), and
//! George, "Nested Dissection of a Regular Finite Element Mesh"
//! (SIAM J. Numer. Anal., 1973).
//!
//! The public surface conforms to the ordering-crate contract of
//! `rslab-ordering-core`: `CscPattern`, `OrderingStats`,
//! `OrderingError`, and `CONTRACT_VERSION` are re-exported from it.
//!
//! `metis_order_full` coarsens the graph (SHEM + 2-hop), picks the
//! best of `niparts` initial bisections scored on their post-FM cut,
//! turns the coarsest edge bisection into a node separator via min
//! vertex cover (Konig's theorem), refines that node separator at
//! every uncoarsening level, and recursively orders the two sides -
//! handing off to AMD on subgraphs no larger than
//! `nd_to_amd_switch`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

// Internal modules are `pub` (hidden) for tests and examples; not
// every helper is used by `metis_order_full`, so dead-code lint is
// suppressed at the module root.
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod coarsen;
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod fm_refine;
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod graph;
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod initial_partition;
mod node_nd;
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod rng;
mod sep_refine;
#[doc(hidden)]
#[allow(dead_code, missing_docs)]
pub mod separator;

pub use rslab_ordering_core::{CscPattern, OrderingError, OrderingStats, CONTRACT_VERSION};

/// Tunable parameters for METIS nested-dissection ordering.
///
/// Defaults mirror METIS 5.2.0's `METIS_NodeND` defaults (MUMPS uses
/// stock METIS defaults for KKT problems: `METIS_OPTION_NUMBERING = 1`,
/// all other options at library default).
#[derive(Debug, Clone)]
pub struct MetisOptions {
    /// Deterministic RNG seed. Defaults to 1. Two runs with the same
    /// seed on the same input must produce the same permutation.
    pub seed: u64,
    /// Number of initial-bisection trials at the coarsest level
    /// (METIS 5.2.0 default: 7). Each trial alternates GGP and random
    /// BFS and is scored on its post-FM cut.
    pub niparts: u32,
    /// Stop coarsening when the graph has fewer than this many
    /// vertices (METIS 5.2.0 default: 120).
    pub coarsen_floor: u32,
    /// Switch from recursive ND to AMD on uncoarsened subproblems of
    /// at most this many vertices (METIS 5.2.0 default: 200).
    pub nd_to_amd_switch: u32,
    /// Reduction-ratio threshold below which SHEM falls back to
    /// 2-hop matching (METIS 5.2.0 default: 0.85).
    pub two_hop_ratio_threshold: f64,
    /// Maximum partition imbalance factor (`ufactor` in METIS terms,
    /// encoded as a fraction here). METIS 5.2.0 uses 200, which
    /// corresponds to 1.20 load balance tolerance; expressed as the
    /// fractional deviation 0.20.
    pub max_imbalance: f64,
    /// Number of FM passes at each uncoarsening level (METIS 5.2.0
    /// default: 10).
    pub fm_passes: u32,
    /// A separator refinement pass stops after this many consecutive moves
    /// without a better separator. Limits below about 50 000 cost fill on
    /// grids (+37 % nnz(L) on a 40^3 grid at `min(5 * separator, 400)`),
    /// so the default effectively never binds.
    pub move_limit: usize,
    /// A refinement pass also stops when the separator has grown to this
    /// multiple of the best one seen.
    pub max_overshoot: f64,
    /// Subproblems with at least this many vertices order their two sides
    /// in parallel. Affects time only, not the permutation.
    pub parallel_min_vertices: usize,
    /// Levels with at least this many adjacency entries contract in
    /// parallel blocks. Affects time only, not the permutation.
    pub parallel_min_edges: usize,
}

impl Default for MetisOptions {
    fn default() -> Self {
        Self {
            seed: 1,
            niparts: 7,
            coarsen_floor: 120,
            nd_to_amd_switch: 200,
            two_hop_ratio_threshold: 0.85,
            max_imbalance: 0.20,
            fm_passes: 10,
            move_limit: 1 << 20,
            max_overshoot: 4.0,
            parallel_min_vertices: 4096,
            parallel_min_edges: 200_000,
        }
    }
}

/// Crate-specific diagnostic counters for METIS nested dissection.
///
/// Populated per call to [`metis_order_full`]. Callers that only need
/// the permutation should use [`metis_order`]; callers that need the
/// shared [`OrderingStats`] (wall time) should use
/// [`metis_order_full`].
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MetisStats {
    /// Number of coarsening levels built.
    pub n_levels: u32,
    /// Number of top-level connected components encountered.
    pub n_components: u32,
    /// Number of vertices assigned to a separator at any level.
    pub n_separator_vertices: u32,
    /// Number of FM passes executed across all levels.
    pub n_fm_passes: u32,
    /// Number of times SHEM fell through to the 2-hop matching path.
    pub n_two_hop_fallbacks: u32,
    /// Number of subgraphs handed off to the AMD leaf solver (when
    /// `nd_to_amd_switch` triggers).
    pub n_amd_leaf_calls: u32,
}

/// Compute a fill-reducing METIS nested-dissection ordering.
///
/// Thin wrapper over [`metis_order_full`] that discards the
/// diagnostic stats. Returns a permutation `perm` (new-to-old).
pub fn metis_order(pattern: &CscPattern<'_>) -> Result<Vec<i32>, OrderingError> {
    metis_order_full(pattern, &MetisOptions::default()).map(|(perm, _, _)| perm)
}

/// Contract-conforming ordering producer.
///
/// Signature matches the shape every RSLAB ordering crate must expose
/// per the `rslab-ordering-core` contract: input is a
/// full-symmetric [`CscPattern`] and options; output is a three-tuple
/// of `(perm, OrderingStats, crate-stats)`, with errors in
/// [`OrderingError`].
///
/// `OrderingStats.time_us` is the wall-clock time of this call.
/// `fill_estimate` and `flop_estimate` stay `None` - METIS does not
/// produce them at the ordering boundary; they belong to a downstream
/// symbolic analysis.
///
/// Runs the full pipeline: coarsen, initial bisection, FM, separator
/// construction and refinement, and recursive nested dissection with an
/// AMD leaf fallback for subgraphs of at most `nd_to_amd_switch`
/// vertices.
/// The two sides of every bisection are ordered in parallel on the ambient
/// rayon pool once a subproblem is large enough; child seeds are derived
/// structurally, so the permutation depends on `opts.seed` only.
pub fn metis_order_full(
    pattern: &CscPattern<'_>,
    opts: &MetisOptions,
) -> Result<(Vec<i32>, OrderingStats, MetisStats), OrderingError> {
    if pattern.col_ptr.len() != pattern.n + 1 {
        return Err(OrderingError::MalformedInput);
    }
    let t0 = rslab_ordering_core::clock::Instant::now();
    let mut stats = MetisStats::default();

    let perm = node_nd::nd_order(pattern, None, opts, &mut stats)?;

    let ordering_stats = OrderingStats {
        time_us: t0.elapsed().as_micros() as u64,
        fill_estimate: None,
        flop_estimate: None,
    };
    Ok((perm, ordering_stats, stats))
}

/// [`metis_order_full`] on a vertex-weighted graph.
///
/// `vwgt[v]` (at least 1) is the weight of vertex `v`, and the bisections
/// balance the summed weight instead of the vertex count. The use is a
/// compressed graph whose vertices stand for groups of original vertices
/// with identical adjacency: weighting each by its group size keeps the
/// separators as balanced as on the uncompressed graph.
pub fn metis_order_weighted(
    pattern: &CscPattern<'_>,
    vwgt: &[i32],
    opts: &MetisOptions,
) -> Result<(Vec<i32>, OrderingStats, MetisStats), OrderingError> {
    if pattern.col_ptr.len() != pattern.n + 1 {
        return Err(OrderingError::MalformedInput);
    }
    let t0 = rslab_ordering_core::clock::Instant::now();
    let mut stats = MetisStats::default();
    let perm = node_nd::nd_order(pattern, Some(vwgt), opts, &mut stats)?;
    let ordering_stats = OrderingStats {
        time_us: t0.elapsed().as_micros() as u64,
        fill_estimate: None,
        flop_estimate: None,
    };
    Ok((perm, ordering_stats, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trivial_pattern() -> (Vec<i32>, Vec<i32>) {
        // Diagonal n=3: col_ptr=[0,1,2,3], row_idx=[0,1,2]
        (vec![0, 1, 2, 3], vec![0, 1, 2])
    }

    #[test]
    fn diagonal_pattern_yields_permutation() {
        let (cp, ri) = trivial_pattern();
        let pat = CscPattern::new(3, &cp, &ri).unwrap();
        let (perm, ostats, _mstats) = metis_order_full(&pat, &MetisOptions::default()).expect("ok");
        assert_eq!(perm.len(), 3);
        let mut seen = [false; 3];
        for &p in &perm {
            assert!((0..3).contains(&p));
            seen[p as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
        // time_us is populated; fill/flop remain None.
        assert!(ostats.fill_estimate.is_none());
        assert!(ostats.flop_estimate.is_none());
    }

    #[test]
    fn convenience_wrapper_returns_permutation() {
        let (cp, ri) = trivial_pattern();
        let pat = CscPattern::new(3, &cp, &ri).unwrap();
        let perm = metis_order(&pat).expect("ok");
        assert_eq!(perm.len(), 3);
    }
}
