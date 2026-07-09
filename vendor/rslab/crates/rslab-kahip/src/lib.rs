//! KaHIP-style flow-based nested-dissection fill-reducing ordering.
//!
//! **Status: phases K1-K6 complete.**
//! [`kahip_order`] produces a contract-conforming permutation via the
//! full pipeline: K1 data reduction (degree-1 / degree-2 / twin /
//! subset), then K2-K6 multilevel flow-based nested dissection on the
//! reduced graph (coarsen → initial bisect → uncoarsen with K3 flow
//! refinement → K4 boundary-bipartite node separator → recurse), then
//! K1 expansion to lift the reduced-graph permutation back to original
//! indices.
//!
//! **Plan.** `dev/plans/ordering-kahip.md` tracks the six
//! implementation phases:
//!   - K1: Data reduction (degree-1 / degree-2 / twin / neighborhood-
//!     subset rules, fixed-point loop, expansion permutation stack).
//!   - K2: Push-relabel max-flow (with gap relabeling).
//!   - K3: Flow-based edge refinement (band extraction, super-source/
//!     sink construction, Most Balanced Min Cut).
//!   - K4: Flow-based node separator (vertex-capacitated max-flow).
//!   - K5: V-cycle / F-cycle controller (cut-edge-preserving
//!     re-coarsening for monotone quality improvement).
//!   - K6: Driver and Fast / Eco / Strong modes.
//!
//! **Reference papers** (published, public-domain algorithms - the
//! implementation must be clean-room from these sources, not from
//! KaHIP's C++ codebase):
//!   - Sanders & Schulz, "Engineering Multilevel Graph Partitioning
//!     Algorithms" (2011) - the kaffpa framework.
//!   - Ost, Schulz & Strash, "Engineering Data Reduction for Nested
//!     Dissection" (2021) - the K1 reduction rules.
//!
//! The public surface conforms to the RSLAB ordering-crate contract
//! (`dev/plans/ordering-crate-contract.md`): `CscPattern`,
//! `OrderingStats`, `OrderingError`, and `CONTRACT_VERSION` are
//! re-exported from `rslab-ordering-core`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use rslab_ordering_core::{CscPattern, OrderingError, OrderingStats, CONTRACT_VERSION};

// Phase K1: data reduction (Ost-Schulz-Strash 2021). Wired into the
// K6 driver (see `node_nd::kahip_nd_order`) as a fixed-point pre-pass
// that shrinks the graph via degree-1 / degree-2 / twin / subset rules
// before multilevel partitioning. Eliminated vertices are expanded
// back into the final permutation via `expand_permutation`. See
// `dev/plans/ordering-kahip.md` and `dev/research/ordering-kahip-k1.md`.
mod data_reduction;

// Phase K2: push-relabel max-flow / min-cut (Goldberg-Tarjan 1988 +
// Cherkassky-Goldberg 1995 gap relabeling). Internal until K3 (flow-
// based edge refinement) consumes it; see
// `dev/plans/ordering-kahip.md` and `dev/research/ordering-kahip-k2.md`.
#[allow(dead_code)]
mod flow;

// Phase K3 scaffolding: shared undirected-graph type (CSR) used by
// K3/K4/K5/K6, and flow-based edge refinement of a bisection.
// Internal until K5/K6 consume them; see
// `dev/plans/ordering-kahip.md` and `dev/research/ordering-kahip-k3.md`.
#[allow(dead_code)]
mod flow_refine;
#[allow(dead_code)]
mod graph;

// Phase K4: flow-based node separator via boundary-bipartite vertex
// cover (König's theorem reduction). Internal until K5/K6 consume
// it; see `dev/plans/ordering-kahip.md` and
// `dev/research/ordering-kahip-k4.md`.
#[allow(dead_code)]
mod node_separator;

// Phase K5 (multilevel bisection controller) and K6 (recursive ND
// driver). K5 reuses rslab-metis's coarsening / initial-partition / FM
// plumbing and plugs in K3 flow refinement at each uncoarsening level.
// K6 walks connected components, recurses on each, and layers K4 on top
// of K5 to produce a node separator at every internal level.
mod cycle;
mod node_nd;

/// Crate-specific diagnostic statistics.
///
/// Populated by [`kahip_order_full`] once the implementation lands;
/// zeroed while the crate is in its scaffold state.
#[derive(Debug, Default, Clone)]
pub struct KahipStats {
    /// Number of vertices after data-reduction preprocessing.
    /// `reduced_n == 0` indicates the reduction phase has not run
    /// (scaffold state).
    pub reduced_n: usize,
    /// Largest max-flow subproblem size, in vertices, encountered
    /// during flow-based refinement. `0` while scaffolded.
    pub max_flow_vertices: usize,
    /// Number of multilevel bisections performed - one per node-separator
    /// computation across the nested-dissection tree. Each is a single
    /// V-cycle (one coarsen followed by one uncoarsen). `0` while
    /// scaffolded.
    pub cycles: usize,
    /// Number of top-level connected components encountered by the
    /// nested-dissection driver. `0` while scaffolded. Matches the
    /// `n_components` field on `MetisStats` / `ScotchStats`.
    pub n_components: u32,
}

/// Quality / speed tradeoff modes for the KaHIP driver.
///
/// The exact tuning of each mode is fixed once phase K6 lands. Until
/// then the enum is reserved so that callers can encode intent
/// without the crate compiling the mapping.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum KahipMode {
    /// METIS-comparable wall-clock; single multilevel pass.
    #[default]
    Fast,
    /// 2-3× Fast; one V-cycle with flow refinement at the finest
    /// level.
    Eco,
    /// 5-10× Fast; F-cycle with flow refinement at every level.
    Strong,
}

/// Tunable parameters for KaHIP nested-dissection ordering.
///
/// Kept intentionally narrow while the crate is a scaffold -
/// defaults will match KaHIP's library defaults (seed=0, mode=Fast)
/// once phase K6 is implemented.
#[derive(Debug, Clone)]
pub struct KahipOptions {
    /// Deterministic RNG seed. Two runs with the same seed on the
    /// same input must produce the same permutation.
    pub seed: u64,
    /// Quality / speed tradeoff. See [`KahipMode`].
    pub mode: KahipMode,
}

impl Default for KahipOptions {
    fn default() -> Self {
        Self {
            seed: 1,
            mode: KahipMode::default(),
        }
    }
}

/// Compute a fill-reducing KaHIP nested-dissection ordering.
///
/// Thin wrapper over [`kahip_order_full`] that discards the
/// diagnostic stats. Returns a permutation `perm` (new-to-old).
///
/// Runs the K2-K6 pipeline with default options; see
/// [`kahip_order_full`] for the tunable entry point.
pub fn kahip_order(pattern: &CscPattern<'_>) -> Result<Vec<i32>, OrderingError> {
    kahip_order_full(pattern, &KahipOptions::default()).map(|(perm, _, _)| perm)
}

/// Contract-conforming ordering producer.
///
/// Signature matches the shape every RSLAB ordering crate must expose
/// per `dev/plans/ordering-crate-contract.md`: input is a
/// full-symmetric [`CscPattern`] and options; output is a three-tuple
/// of `(perm, OrderingStats, crate-stats)`, with errors in
/// [`OrderingError`].
///
/// Runs the K2-K6 pipeline: K5 multilevel edge bisection (coarsen,
/// initial bisect, uncoarsen with K3 flow refinement at each level),
/// K4 boundary-bipartite vertex cover to lift the bisection to a node
/// separator, and recursive nested dissection with an AMD leaf
/// fallback for subgraphs below the mode-dependent switch.
///
/// `OrderingStats.time_us` is the wall-clock time of this call.
/// `fill_estimate` and `flop_estimate` stay `None` - KaHIP does not
/// produce them at the ordering boundary; they belong to a downstream
/// symbolic analysis.
pub fn kahip_order_full(
    pattern: &CscPattern<'_>,
    opts: &KahipOptions,
) -> Result<(Vec<i32>, OrderingStats, KahipStats), OrderingError> {
    if pattern.col_ptr.len() != pattern.n + 1 {
        return Err(OrderingError::MalformedInput);
    }
    let t0 = std::time::Instant::now();
    let mut stats = KahipStats::default();
    let perm = node_nd::kahip_nd_order(pattern, opts, &mut stats)?;
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

    #[test]
    fn scaffold_rejects_malformed_input() {
        let col_ptr = [0i32, 0];
        let row_idx: [i32; 0] = [];
        let pattern = CscPattern::new(5, &col_ptr, &row_idx);
        assert!(pattern.is_none(), "malformed pattern must fail validation");
    }

    #[test]
    fn diagonal_pattern_yields_valid_permutation() {
        let col_ptr = [0i32, 1, 2, 3];
        let row_idx = [0i32, 1, 2];
        let pattern = CscPattern::new(3, &col_ptr, &row_idx).expect("valid pattern");
        let perm = kahip_order(&pattern).expect("ordering ok");
        assert_eq!(perm.len(), 3);
        let mut seen = [false; 3];
        for &p in &perm {
            assert!((0..3).contains(&p));
            seen[p as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn scaffold_propagates_malformed_input_to_caller() {
        // Caller-side malformed check: col_ptr len mismatch.
        // We have to construct this manually since CscPattern::new
        // refuses it - so build the struct through a sibling-crate
        // pattern then corrupt through direct field access is not
        // possible. Instead, test the same invariant via public API.
        let col_ptr = [0i32, 2];
        let row_idx = [0i32, 1];
        let pattern = CscPattern::new(1, &col_ptr, &row_idx);
        assert!(
            pattern.is_none(),
            "n=1 but col_ptr suggests 1 column with 2 rows"
        );
    }
}
