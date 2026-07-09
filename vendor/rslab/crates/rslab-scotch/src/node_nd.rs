//! SCOTCH-style nested-dissection driver (S5).
//!
//! Top-level algorithm (Pellegrini 1996 §3, with audit corrections
//! tracked in `dev/plans/ordering-scotch.md`):
//!
//! 1. **Optional graph compression** at the top level. If
//!    `compress_graph` succeeds, recurse on the compressed graph and
//!    expand the resulting permutation through `vertex_map`.
//! 2. **Connected-component split.** Each component is ordered
//!    independently and concatenated.
//! 3. **Per-component recursion.** Subgraphs of at most
//!    `amd_switch` vertices are handed off to AMD; otherwise a
//!    multilevel node bisection runs and the separator is numbered
//!    last.
//!
//! Differences from `rslab_metis::node_nd`:
//!
//! - **Top-level compression** (steps S1, this module).
//! - **Halo-FM refinement** during uncoarsening (S3) instead of plain
//!   boundary FM, so the candidate set widens dynamically.
//! - **Direct vertex separator** via two-sided FM (S2) replaces the
//!   König min-vertex-cover construction. Optimises separator weight
//!   directly instead of an upper bound.
//!
//! The numbering convention is "permutation" = new-position → old-id.
//! Internally we populate `iperm[original_vertex] = new_position` and
//! invert at the end.

use rslab_metis::internals::coarsen::{coarsen, CoarsenCounters};
use rslab_metis::internals::fm_refine::refine_bisection;
use rslab_metis::internals::graph::Graph;
use rslab_metis::internals::initial_partition::{
    initial_bisect_bfs, initial_bisect_ggp, PART_A, PART_B,
};
use rslab_metis::internals::rng::SplitMix;
use rslab_metis::MetisOptions;
use rslab_ordering_core::{CscPattern, OrderingError};

use crate::compress::compress_graph;
use crate::halo_fm::halo_fm_refine;
use crate::vertex_separator::compute_vertex_separator;
use crate::{ScotchOptions, ScotchStats};

/// Entry point. Produces a permutation `perm` where `perm[i]` is the
/// old vertex id placed at new position `i` (new-to-old).
pub(crate) fn scotch_nd_order(
    pattern: &CscPattern<'_>,
    opts: &ScotchOptions,
    stats: &mut ScotchStats,
) -> Result<Vec<i32>, OrderingError> {
    let graph = Graph::from_csc_pattern(pattern)?;
    let n = graph.nvtxs as usize;

    // -- Top-level graph compression.
    if opts.compress {
        if let Some(cg) = compress_graph(&graph, opts.compress_ratio) {
            stats.n_compressed_out = (n - cg.graph.nvtxs as usize) as u32;
            let mut sub_iperm: Vec<i32> = vec![-1; cg.graph.nvtxs as usize];
            let mut rng = SplitMix::new(opts.seed);
            run_top(&cg.graph, &mut sub_iperm, opts, &mut rng, stats)?;
            return expand_perm(&cg.vertex_map, &sub_iperm, n);
        }
    }

    let mut iperm: Vec<i32> = vec![-1; n];
    let mut rng = SplitMix::new(opts.seed);
    run_top(&graph, &mut iperm, opts, &mut rng, stats)?;
    invert_iperm(&iperm, n)
}

/// Walk the connected components of `graph`, dispatching each to
/// `recurse` with a contiguous numbering window.
fn run_top(
    graph: &Graph,
    iperm: &mut [i32],
    opts: &ScotchOptions,
    rng: &mut SplitMix,
    stats: &mut ScotchStats,
) -> Result<(), OrderingError> {
    let n = graph.nvtxs as usize;
    let (cc_label, ncc) = connected_components(graph);
    stats.n_components = ncc as u32;
    let mut offset: usize = 0;
    for c in 0..ncc {
        let (sub, vtx_map) = extract_by_label(graph, &cc_label, c as i32);
        let count = sub.nvtxs as usize;
        if count > 0 {
            recurse(&sub, &vtx_map, offset, iperm, opts, rng, stats)?;
        }
        offset += count;
    }
    debug_assert_eq!(offset, n);
    Ok(())
}

fn recurse(
    subgraph: &Graph,
    vtx_map: &[i32],
    offset: usize,
    iperm: &mut [i32],
    opts: &ScotchOptions,
    rng: &mut SplitMix,
    stats: &mut ScotchStats,
) -> Result<(), OrderingError> {
    let n = subgraph.nvtxs as usize;
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        iperm[vtx_map[0] as usize] = offset as i32;
        return Ok(());
    }

    // Sub-problem may be disconnected after a prior bisection.
    let (cc_label, ncc) = connected_components(subgraph);
    if ncc > 1 {
        let mut off = offset;
        for c in 0..ncc {
            let (sub, map) = extract_by_label(subgraph, &cc_label, c as i32);
            let map_to_orig: Vec<i32> = map.iter().map(|&local| vtx_map[local as usize]).collect();
            let count = sub.nvtxs as usize;
            recurse(&sub, &map_to_orig, off, iperm, opts, rng, stats)?;
            off += count;
        }
        return Ok(());
    }

    if n <= opts.amd_switch as usize {
        amd_leaf(subgraph, vtx_map, offset, iperm, stats)?;
        return Ok(());
    }

    let labels = multilevel_node_bisection(subgraph, opts, rng, stats);

    let mut a_verts: Vec<i32> = Vec::new();
    let mut b_verts: Vec<i32> = Vec::new();
    let mut s_verts: Vec<i32> = Vec::new();
    for (v, &l) in labels.iter().enumerate() {
        match l {
            PART_A => a_verts.push(v as i32),
            PART_B => b_verts.push(v as i32),
            _ => s_verts.push(v as i32),
        }
    }

    // Degenerate bisection: fall back to AMD on the whole subgraph.
    if a_verts.is_empty() || b_verts.is_empty() {
        amd_leaf(subgraph, vtx_map, offset, iperm, stats)?;
        return Ok(());
    }

    stats.n_separator_vertices += s_verts.len() as u32;
    let na = a_verts.len();
    let nb = b_verts.len();

    // Number separator last: positions [offset + na + nb, offset + n).
    for (i, &v) in s_verts.iter().enumerate() {
        let orig = vtx_map[v as usize];
        iperm[orig as usize] = (offset + na + nb + i) as i32;
    }

    let (sub_a, map_a_local) = extract_by_list(subgraph, &a_verts);
    let map_a: Vec<i32> = map_a_local
        .iter()
        .map(|&local| vtx_map[local as usize])
        .collect();
    let (sub_b, map_b_local) = extract_by_list(subgraph, &b_verts);
    let map_b: Vec<i32> = map_b_local
        .iter()
        .map(|&local| vtx_map[local as usize])
        .collect();

    recurse(&sub_a, &map_a, offset, iperm, opts, rng, stats)?;
    recurse(&sub_b, &map_b, offset + na, iperm, opts, rng, stats)?;
    Ok(())
}

/// SCOTCH multilevel node bisection: coarsen → best-of-`n_sep_trials`
/// initial bisection → uncoarsen with halo FM → direct vertex
/// separator. Returns labels in `{PART_A, PART_B, PART_SEP}`.
fn multilevel_node_bisection(
    graph: &Graph,
    opts: &ScotchOptions,
    rng: &mut SplitMix,
    stats: &mut ScotchStats,
) -> Vec<u8> {
    // Build a MetisOptions to drive the shared coarsening framework.
    // SCOTCH defaults map onto the same knobs (coarsen_floor,
    // two_hop_ratio_threshold). The seed is forwarded so the
    // coarsening matching is deterministic per ScotchOptions::seed.
    let metis_opts = MetisOptions {
        seed: opts.seed,
        niparts: opts.n_sep_trials,
        coarsen_floor: opts.coarsen_floor,
        nd_to_amd_switch: opts.amd_switch,
        // SCOTCH uses 0.85 for the same dense-fallback role as METIS.
        two_hop_ratio_threshold: 0.85,
        max_imbalance: opts.max_imbalance,
        fm_passes: opts.fm_pass_cap,
        // Inherit any new MetisOptions knobs from the upstream
        // defaults. SCOTCH-specific dense-quotient handling lives in
        // its own driver, not here.
        ..MetisOptions::default()
    };
    let mut counters = CoarsenCounters::default();
    let levels = coarsen(graph, &metis_opts, rng, &mut counters);
    stats.n_levels += levels.len() as u32;

    let coarsest: &Graph = match levels.last() {
        Some(cg) => &cg.graph,
        None => graph,
    };
    let total: i64 = coarsest.vwgt.iter().map(|&w| w as i64).sum();
    let target = total / 2;

    // Best-of-`n_sep_trials` initial bisection, scored on post-FM cut.
    let mut best_labels: Vec<u8> = vec![PART_A; coarsest.nvtxs as usize];
    let mut best_cut: i32 = i32::MAX;
    for trial in 0..opts.n_sep_trials {
        let mut trial_labels = if trial % 2 == 0 {
            initial_bisect_ggp(coarsest, rng, target)
        } else {
            initial_bisect_bfs(coarsest, rng, target)
        };
        let cut = refine_bisection(
            coarsest,
            &mut trial_labels,
            opts.max_imbalance,
            opts.fm_pass_cap,
        );
        stats.n_fm_passes += opts.fm_pass_cap;
        if cut < best_cut {
            best_cut = cut;
            best_labels = trial_labels;
        }
    }
    let mut labels = best_labels;

    // Uncoarsen with halo FM at every projected level. Halo FM
    // dynamically extends the candidate set beyond the strict
    // boundary; on level 0 (the original `graph`) this is the SCOTCH
    // refinement step that feeds the direct vertex separator below.
    for level_idx in (0..levels.len()).rev() {
        let cg = &levels[level_idx];
        let prev_graph: &Graph = if level_idx == 0 {
            graph
        } else {
            &levels[level_idx - 1].graph
        };
        let prev_n = prev_graph.nvtxs as usize;
        let mut proj: Vec<u8> = vec![PART_A; prev_n];
        for (v, p) in proj.iter_mut().enumerate().take(prev_n) {
            let c = cg.cmap[v] as usize;
            *p = labels[c];
        }
        labels = proj;
        halo_fm_refine(
            prev_graph,
            &mut labels,
            opts.max_imbalance,
            opts.fm_pass_cap,
        );
        stats.n_fm_passes += opts.fm_pass_cap;
    }

    // Direct vertex separator (SCOTCH-specific): two-sided FM
    // operating on `labels ∈ {A, B}` and producing
    // `labels ∈ {A, B, SEP}`. Optimises separator *weight* directly,
    // not an upper bound.
    let _sep_w = compute_vertex_separator(
        graph,
        &mut labels,
        opts.max_imbalance,
        opts.fm_move_cap,
        opts.fm_pass_cap,
    );

    labels
}

/// Order a leaf subgraph with AMD and write its slice of `iperm`.
///
/// Finding O15: when top-level graph compression is enabled,
/// `subgraph.vwgt` can carry supervariable weights (one vertex standing
/// for several original rows), and those weights ride down through
/// bisection into the leaves. `graph_to_csc_pattern` emits only the
/// adjacency *structure*, so the AMD call below orders on the pattern
/// alone - `rslab_amd` exposes no weighted entry point - and a
/// weight-7 supervariable is scored as a unit vertex. This can skew
/// AMD's degree-based pivot choice on heavily-compressed inputs, but
/// the leaf still emits a valid bijection over its vertices, so
/// `expand_perm` lifts it to a valid permutation of the original
/// matrix (correctness holds; only pivot quality is affected). A
/// weight-aware AMD leaf is future work; see dev/tried-and-rejected.md
/// (O15).
fn amd_leaf(
    subgraph: &Graph,
    vtx_map: &[i32],
    offset: usize,
    iperm: &mut [i32],
    stats: &mut ScotchStats,
) -> Result<(), OrderingError> {
    stats.n_amd_leaf_calls += 1;
    let n = subgraph.nvtxs as usize;
    if n == 0 {
        return Ok(());
    }
    let (col_ptr, row_idx) = graph_to_csc_pattern(subgraph);
    let pattern = CscPattern::new(n, &col_ptr, &row_idx).ok_or(OrderingError::MalformedInput)?;
    let perm_local = rslab_amd::amd_order(&pattern)?;
    debug_assert_eq!(perm_local.len(), n);
    for (new_pos, &local_id) in perm_local.iter().enumerate() {
        let orig = vtx_map[local_id as usize];
        iperm[orig as usize] = (offset + new_pos) as i32;
    }
    Ok(())
}

/// Build a full-symmetric CSC pattern from an internal `Graph`.
/// Adjacency in `Graph` is full-symmetric with the diagonal dropped;
/// we reinsert the diagonal at its sorted position so AMD's intake
/// validator accepts the pattern.
fn graph_to_csc_pattern(graph: &Graph) -> (Vec<i32>, Vec<i32>) {
    let n = graph.nvtxs as usize;
    let mut col_ptr: Vec<i32> = Vec::with_capacity(n + 1);
    let mut row_idx: Vec<i32> = Vec::with_capacity(graph.adjncy.len() + n);
    col_ptr.push(0);
    for v in 0..n {
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        let mut diag_inserted = false;
        for k in lo..hi {
            let u = graph.adjncy[k];
            if !diag_inserted && u as usize > v {
                row_idx.push(v as i32);
                diag_inserted = true;
            }
            row_idx.push(u);
        }
        if !diag_inserted {
            row_idx.push(v as i32);
        }
        col_ptr.push(row_idx.len() as i32);
    }
    (col_ptr, row_idx)
}

fn connected_components(graph: &Graph) -> (Vec<i32>, usize) {
    let n = graph.nvtxs as usize;
    let mut cc: Vec<i32> = vec![-1; n];
    let mut ncc: i32 = 0;
    let mut queue: Vec<i32> = Vec::new();
    for start in 0..n {
        if cc[start] >= 0 {
            continue;
        }
        cc[start] = ncc;
        queue.clear();
        queue.push(start as i32);
        while let Some(v) = queue.pop() {
            let vu = v as usize;
            let lo = graph.xadj[vu] as usize;
            let hi = graph.xadj[vu + 1] as usize;
            for k in lo..hi {
                let u = graph.adjncy[k];
                if cc[u as usize] < 0 {
                    cc[u as usize] = ncc;
                    queue.push(u);
                }
            }
        }
        ncc += 1;
    }
    (cc, ncc as usize)
}

fn extract_by_label(graph: &Graph, label: &[i32], c: i32) -> (Graph, Vec<i32>) {
    let n = graph.nvtxs as usize;
    let mut vtx_map: Vec<i32> = Vec::new();
    let mut local_id: Vec<i32> = vec![-1; n];
    for v in 0..n {
        if label[v] == c {
            local_id[v] = vtx_map.len() as i32;
            vtx_map.push(v as i32);
        }
    }
    (build_induced(graph, &vtx_map, &local_id), vtx_map)
}

fn extract_by_list(graph: &Graph, verts: &[i32]) -> (Graph, Vec<i32>) {
    let n = graph.nvtxs as usize;
    let mut local_id: Vec<i32> = vec![-1; n];
    for (i, &v) in verts.iter().enumerate() {
        local_id[v as usize] = i as i32;
    }
    let vtx_map = verts.to_vec();
    (build_induced(graph, &vtx_map, &local_id), vtx_map)
}

fn build_induced(graph: &Graph, vtx_map: &[i32], local_id: &[i32]) -> Graph {
    let sub_n = vtx_map.len();
    let mut xadj: Vec<i32> = Vec::with_capacity(sub_n + 1);
    let mut adjncy: Vec<i32> = Vec::new();
    let mut adjwgt: Vec<i32> = Vec::new();
    let mut vwgt: Vec<i32> = Vec::with_capacity(sub_n);
    xadj.push(0);
    for &orig in vtx_map {
        let v = orig as usize;
        vwgt.push(graph.vwgt[v]);
        let lo = graph.xadj[v] as usize;
        let hi = graph.xadj[v + 1] as usize;
        for k in lo..hi {
            let u = graph.adjncy[k] as usize;
            let lu = local_id[u];
            if lu >= 0 {
                adjncy.push(lu);
                adjwgt.push(graph.adjwgt[k]);
            }
        }
        xadj.push(adjncy.len() as i32);
    }
    Graph {
        nvtxs: sub_n as i32,
        xadj,
        adjncy,
        vwgt,
        adjwgt,
    }
}

/// Invert an iperm (`old → new`) into a perm (`new → old`), validating
/// that every new position is filled exactly once.
fn invert_iperm(iperm: &[i32], n: usize) -> Result<Vec<i32>, OrderingError> {
    let mut perm: Vec<i32> = vec![-1; n];
    for (old, &new_pos) in iperm.iter().enumerate() {
        if new_pos < 0 || (new_pos as usize) >= n {
            return Err(OrderingError::Internal(
                "scotch nd produced invalid permutation",
            ));
        }
        let np = new_pos as usize;
        if perm[np] >= 0 {
            return Err(OrderingError::Internal(
                "scotch nd produced duplicate position",
            ));
        }
        perm[np] = old as i32;
    }
    Ok(perm)
}

/// Expand a permutation produced on a compressed graph into a
/// permutation on the original vertex set.
///
/// `sub_iperm[c]` gives the new position of compressed vertex `c`.
/// Each compressed vertex represents a contiguous *block* in the
/// expanded numbering: members of a supervariable are placed
/// consecutively in the order recorded by `vertex_map[c]`. The block
/// for `c` starts at the cumulative sum of all class sizes whose
/// representatives come earlier in the ND ordering.
fn expand_perm(
    vertex_map: &[Vec<i32>],
    sub_iperm: &[i32],
    n_orig: usize,
) -> Result<Vec<i32>, OrderingError> {
    let n_sub = sub_iperm.len();
    if vertex_map.len() != n_sub {
        return Err(OrderingError::Internal(
            "scotch compression: vertex_map / sub_iperm size mismatch",
        ));
    }
    // Invert sub_iperm (compressed): sub_perm[new_sub_pos] = compressed id.
    let mut sub_perm: Vec<i32> = vec![-1; n_sub];
    for (c, &np) in sub_iperm.iter().enumerate() {
        if np < 0 || (np as usize) >= n_sub {
            return Err(OrderingError::Internal(
                "scotch compression: invalid sub-permutation",
            ));
        }
        sub_perm[np as usize] = c as i32;
    }

    // Walk new positions in order; for each compressed vertex, emit
    // its members in original-id order (vertex_map is already sorted
    // ascending by construction in compress.rs).
    let mut perm: Vec<i32> = Vec::with_capacity(n_orig);
    for &c in &sub_perm {
        if c < 0 {
            return Err(OrderingError::Internal(
                "scotch compression: incomplete sub-permutation",
            ));
        }
        for &orig in &vertex_map[c as usize] {
            perm.push(orig);
        }
    }
    if perm.len() != n_orig {
        return Err(OrderingError::Internal(
            "scotch compression: expanded length mismatch",
        ));
    }
    Ok(perm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rslab_ordering_core::CscPattern;
    use std::collections::BTreeSet;

    fn csc_from_triples(n: usize, triples: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
        let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
        for &(i, j) in triples {
            set.insert((i, j));
            set.insert((j, i));
        }
        let mut cols: Vec<Vec<i32>> = vec![Vec::new(); n];
        for &(r, c) in &set {
            cols[c].push(r as i32);
        }
        for col in &mut cols {
            col.sort();
        }
        let mut col_ptr: Vec<i32> = vec![0];
        let mut row_idx: Vec<i32> = Vec::new();
        for col in &cols {
            for &r in col {
                row_idx.push(r);
            }
            col_ptr.push(row_idx.len() as i32);
        }
        (col_ptr, row_idx)
    }

    fn grid_triples(m: usize, n: usize) -> Vec<(usize, usize)> {
        let idx = |r: usize, c: usize| r * n + c;
        let mut t = Vec::new();
        for r in 0..m {
            for c in 0..n {
                let k = idx(r, c);
                t.push((k, k));
                if r + 1 < m {
                    t.push((k, idx(r + 1, c)));
                }
                if c + 1 < n {
                    t.push((k, idx(r, c + 1)));
                }
            }
        }
        t
    }

    fn assert_permutation(perm: &[i32]) {
        let n = perm.len();
        let mut seen = vec![false; n];
        for &p in perm {
            let p = p as usize;
            assert!(p < n, "index {} out of bounds", p);
            assert!(!seen[p], "duplicate {}", p);
            seen[p] = true;
        }
    }

    #[test]
    fn diagonal_pattern_yields_permutation() {
        let cp = vec![0, 1, 2, 3];
        let ri = vec![0, 1, 2];
        let pat = CscPattern::new(3, &cp, &ri).unwrap();
        let opts = ScotchOptions::default();
        let mut stats = ScotchStats::default();
        let perm = scotch_nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 3);
        assert_permutation(&perm);
    }

    #[test]
    fn small_grid_uses_amd_leaf() {
        // 10x10 grid, 100 vertices. amd_switch=120 so the top-level
        // bottoms out in AMD (after compression is rejected - a grid
        // has very few indistinguishable vertices, ratio ≈ 1).
        let t = grid_triples(10, 10);
        let (cp, ri) = csc_from_triples(100, &t);
        let pat = CscPattern::new(100, &cp, &ri).unwrap();
        let opts = ScotchOptions::default();
        let mut stats = ScotchStats::default();
        let perm = scotch_nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 100);
        assert_permutation(&perm);
        assert!(stats.n_amd_leaf_calls >= 1);
    }

    #[test]
    fn large_grid_runs_multilevel() {
        // 16x16 = 256 vertices > amd_switch (120) so the top level
        // exercises coarsen + initial bisect + halo FM + direct vertex
        // separator at least once.
        let t = grid_triples(16, 16);
        let (cp, ri) = csc_from_triples(256, &t);
        let pat = CscPattern::new(256, &cp, &ri).unwrap();
        let opts = ScotchOptions::default();
        let mut stats = ScotchStats::default();
        let perm = scotch_nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 256);
        assert_permutation(&perm);
        assert!(
            stats.n_separator_vertices > 0,
            "expected a top-level separator on 16x16 grid"
        );
        assert!(stats.n_levels > 0, "expected coarsening to produce levels");
    }

    #[test]
    fn deterministic_across_runs() {
        let t = grid_triples(14, 14);
        let n = 196;
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let opts = ScotchOptions::default();
        let mut s1 = ScotchStats::default();
        let mut s2 = ScotchStats::default();
        let p1 = scotch_nd_order(&pat, &opts, &mut s1).unwrap();
        let p2 = scotch_nd_order(&pat, &opts, &mut s2).unwrap();
        assert_eq!(p1, p2, "scotch ND must be deterministic for fixed seed");
    }

    #[test]
    fn handles_disconnected_components() {
        // Two disjoint 8x8 grids → ncc = 2.
        let mut t = grid_triples(8, 8);
        for &(i, j) in grid_triples(8, 8).iter() {
            t.push((i + 64, j + 64));
        }
        let (cp, ri) = csc_from_triples(128, &t);
        let pat = CscPattern::new(128, &cp, &ri).unwrap();
        let opts = ScotchOptions::default();
        let mut stats = ScotchStats::default();
        let perm = scotch_nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), 128);
        assert_permutation(&perm);
        assert_eq!(stats.n_components, 2);
    }

    #[test]
    fn compression_path_produces_valid_permutation() {
        // Block-diagonal of three 4x4 grids, each with three identical
        // copies of an "extra" vertex hooked to the corner. The extras
        // are indistinguishable → compression triggers and we exercise
        // the expand_perm path.
        //
        // To keep the test self-contained we use a denser construction:
        // four cliques of 5 vertices each, sharing structure.
        // Indistinguishability needs *closed* neighborhoods to match -
        // simplest way is parallel duplicate vertices that each connect
        // to the same anchor set. Here: 1 anchor (vertex 0) with 6
        // pendant vertices (1..7), all attached only to vertex 0.
        // Closed nbhd of vertex i in 1..=6 is {0, i}, which differs per
        // i, so no compression. We need the same closed nbhd: vertices
        // 1..=6 plus 0 forming a clique works (every vertex has closed
        // nbhd = {0..=6}). Build the K_7 case explicitly.
        let n = 7;
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
            for j in (i + 1)..n {
                t.push((i, j));
            }
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let opts = ScotchOptions::default(); // compress=true, ratio=0.7
        let mut stats = ScotchStats::default();
        let perm = scotch_nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), n);
        assert_permutation(&perm);
        // K_7 collapses to a single supervariable of weight 7 →
        // n_compressed_out = 6.
        assert_eq!(stats.n_compressed_out, 6);
    }

    #[test]
    fn compression_disabled_skips_compress_path() {
        let n = 7;
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i));
            for j in (i + 1)..n {
                t.push((i, j));
            }
        }
        let (cp, ri) = csc_from_triples(n, &t);
        let pat = CscPattern::new(n, &cp, &ri).unwrap();
        let opts = ScotchOptions {
            compress: false,
            ..ScotchOptions::default()
        };
        let mut stats = ScotchStats::default();
        let perm = scotch_nd_order(&pat, &opts, &mut stats).unwrap();
        assert_eq!(perm.len(), n);
        assert_permutation(&perm);
        assert_eq!(stats.n_compressed_out, 0);
    }

    #[test]
    fn empty_graph_returns_empty_permutation() {
        let cp = vec![0];
        let ri: Vec<i32> = Vec::new();
        let pat = CscPattern::new(0, &cp, &ri).unwrap();
        let opts = ScotchOptions::default();
        let mut stats = ScotchStats::default();
        let perm = scotch_nd_order(&pat, &opts, &mut stats).unwrap();
        assert!(perm.is_empty());
    }
}
