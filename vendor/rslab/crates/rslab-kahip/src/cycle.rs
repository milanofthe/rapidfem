//! KaHIP Phase K5 - multilevel edge bisection controller.
//!
//! One coarsen → initial-bisect → uncoarsen pass. At each uncoarsening
//! level we apply a cheap FM bootstrap (reusing `rslab_metis::internals`
//! for the multilevel plumbing) and then one or more iterations of K3
//! flow-based refinement. The returned labels are in `{PART_A, PART_B}`;
//! K6 wraps this in a recursive nested-dissection driver that layers
//! K4 on top to produce a node separator.
//!
//! V1 scope per `dev/research/ordering-kahip-k5-k6.md`:
//!  - single pass (trivial V-cycle); full V/F-cycle re-coarsening per
//!    Sanders-Schulz 2011 §4.3 is deferred.
//!  - fixed mode per call (no adaptive escalation).

use rslab_metis::internals::coarsen::{coarsen, CoarsenCounters};
use rslab_metis::internals::fm_refine::refine_bisection;
use rslab_metis::internals::graph::Graph;
use rslab_metis::internals::initial_partition::{initial_bisect_bfs, initial_bisect_ggp, PART_A};
use rslab_metis::internals::rng::SplitMix;
use rslab_metis::MetisOptions;

use crate::flow_refine::flow_refine_bisection;
use crate::graph::UndirectedGraph;
use crate::{KahipMode, KahipOptions, KahipStats};

/// Multilevel edge bisection. Returns labels in `{PART_A, PART_B}`.
///
/// `opts.mode` selects flow-refinement aggressiveness at each level
/// (see `tune` below). The coarsen / FM parameters follow the METIS
/// defaults since those pieces are shared plumbing.
pub(crate) fn multilevel_bisection(
    graph: &Graph,
    opts: &KahipOptions,
    rng: &mut SplitMix,
    stats: &mut KahipStats,
) -> Vec<u8> {
    let p = tune(opts.mode);

    let metis_opts = MetisOptions {
        seed: opts.seed,
        niparts: p.n_sep_trials,
        coarsen_floor: p.coarsen_floor,
        nd_to_amd_switch: p.amd_switch,
        two_hop_ratio_threshold: 0.85,
        max_imbalance: p.max_imbalance,
        fm_passes: p.fm_pass_cap,
        // Inherit any new MetisOptions knobs (e.g. dense quotient)
        // from upstream defaults. KaHIP only uses MetisOptions to
        // drive the shared coarsening framework, which ignores the
        // dense-quotient fields.
        ..MetisOptions::default()
    };

    let mut counters = CoarsenCounters::default();
    let levels = coarsen(graph, &metis_opts, rng, &mut counters);
    // One multilevel bisection is a single V-cycle (one coarsen followed by
    // one uncoarsen); count it. Not a per-coarsening-level counter.
    stats.cycles = stats.cycles.saturating_add(1);

    let coarsest: &Graph = match levels.last() {
        Some(cg) => &cg.graph,
        None => graph,
    };
    let total: i64 = coarsest.vwgt.iter().map(|&w| w as i64).sum();
    let target = total / 2;

    // Best-of-`n_sep_trials` initial bisection scored on post-FM cut.
    let mut best_labels: Vec<u8> = vec![PART_A; coarsest.nvtxs as usize];
    let mut best_cut: i32 = i32::MAX;
    for trial in 0..p.n_sep_trials {
        let mut trial_labels = if trial % 2 == 0 {
            initial_bisect_ggp(coarsest, rng, target)
        } else {
            initial_bisect_bfs(coarsest, rng, target)
        };
        let cut = refine_bisection(coarsest, &mut trial_labels, p.max_imbalance, p.fm_pass_cap);
        if cut < best_cut {
            best_cut = cut;
            best_labels = trial_labels;
        }
    }
    let mut labels = best_labels;

    // Uncoarsen: project, FM bootstrap, then K3 flow refinement.
    for level_idx in (0..levels.len()).rev() {
        let cg = &levels[level_idx];
        let prev_graph: &Graph = if level_idx == 0 {
            graph
        } else {
            &levels[level_idx - 1].graph
        };
        let prev_n = prev_graph.nvtxs as usize;
        let mut proj: Vec<u8> = vec![PART_A; prev_n];
        for (v, p_out) in proj.iter_mut().enumerate().take(prev_n) {
            let c = cg.cmap[v] as usize;
            *p_out = labels[c];
        }
        labels = proj;

        refine_bisection(prev_graph, &mut labels, p.max_imbalance, p.fm_pass_cap);

        let is_finest = level_idx == 0;
        let do_flow = match opts.mode {
            KahipMode::Fast => is_finest,
            KahipMode::Eco | KahipMode::Strong => true,
        };
        if do_flow {
            let ug = graph_to_undirected(prev_graph);
            let iters = match opts.mode {
                KahipMode::Fast | KahipMode::Eco => 1,
                KahipMode::Strong => 2,
            };
            for _ in 0..iters {
                let band_n = ug.n;
                stats.max_flow_vertices = stats.max_flow_vertices.max(band_n);
                let changed =
                    flow_refine_bisection(&ug, &mut labels, p.bnd_distance, p.max_imbalance);
                if !changed {
                    break;
                }
            }
        }
    }

    // If no coarsening occurred (tiny graph), bisect directly.
    if levels.is_empty() {
        refine_bisection(graph, &mut labels, p.max_imbalance, p.fm_pass_cap);
    }

    labels
}

struct ModeParams {
    n_sep_trials: u32,
    coarsen_floor: u32,
    amd_switch: u32,
    max_imbalance: f64,
    fm_pass_cap: u32,
    bnd_distance: usize,
}

fn tune(mode: KahipMode) -> ModeParams {
    match mode {
        KahipMode::Fast => ModeParams {
            n_sep_trials: 3,
            coarsen_floor: 120,
            amd_switch: 200,
            max_imbalance: 0.20,
            fm_pass_cap: 5,
            bnd_distance: 2,
        },
        KahipMode::Eco => ModeParams {
            n_sep_trials: 5,
            coarsen_floor: 100,
            amd_switch: 120,
            max_imbalance: 0.20,
            fm_pass_cap: 8,
            bnd_distance: 3,
        },
        KahipMode::Strong => ModeParams {
            n_sep_trials: 7,
            coarsen_floor: 80,
            amd_switch: 80,
            max_imbalance: 0.20,
            fm_pass_cap: 10,
            bnd_distance: 4,
        },
    }
}

/// Bridge rslab-metis's `i32`-indexed [`Graph`] to the `usize`-indexed
/// [`UndirectedGraph`] that K3/K4 consume. Vertex weights are carried
/// from `Graph::vwgt`; on a coarse graph these are supervertex masses
/// `≫ 1`, and K3/K4 measure balance against them (a count-balanced cut
/// here would be weight-imbalanced). Edge weights carried through. Each
/// undirected edge is already stored twice in `Graph::adjncy`, so the
/// conversion is a direct copy.
pub(crate) fn graph_to_undirected(g: &Graph) -> UndirectedGraph {
    let n = g.nvtxs as usize;
    let mut xadj: Vec<usize> = Vec::with_capacity(n + 1);
    let mut adjncy: Vec<usize> = Vec::with_capacity(g.adjncy.len());
    let mut eweight: Vec<i64> = Vec::with_capacity(g.adjncy.len());
    let vweight: Vec<i64> = (0..n).map(|v| (g.vwgt[v] as i64).max(1)).collect();
    xadj.push(0);
    for v in 0..n {
        let lo = g.xadj[v] as usize;
        let hi = g.xadj[v + 1] as usize;
        // Neighbors may be unsorted post-coarsening; sort so the
        // UndirectedGraph invariant (sorted neighbors) holds.
        let mut row: Vec<(usize, i64)> = (lo..hi)
            .map(|k| (g.adjncy[k] as usize, g.adjwgt[k] as i64))
            .collect();
        row.sort_by_key(|&(u, _)| u);
        for (u, w) in row {
            adjncy.push(u);
            eweight.push(w.max(1));
        }
        xadj.push(adjncy.len());
    }
    UndirectedGraph {
        n,
        xadj,
        adjncy,
        eweight,
        vweight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rslab_metis::internals::initial_partition::PART_B;
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

    fn build_graph(n: usize, triples: &[(usize, usize)]) -> Graph {
        let (cp, ri) = csc_from_triples(n, triples);
        let pat = CscPattern::new(n, &cp, &ri).expect("valid pattern");
        Graph::from_csc_pattern(&pat).expect("graph")
    }

    #[test]
    fn trivial_graph_returns_valid_bisection() {
        let g = build_graph(10, &grid_triples(2, 5));
        let opts = KahipOptions::default();
        let mut rng = SplitMix::new(opts.seed);
        let mut stats = KahipStats::default();
        let labels = multilevel_bisection(&g, &opts, &mut rng, &mut stats);
        assert_eq!(labels.len(), 10);
        let c0 = labels.iter().filter(|&&l| l == PART_A).count();
        let c1 = labels.iter().filter(|&&l| l == PART_B).count();
        assert_eq!(c0 + c1, 10);
        assert!(c0 > 0 && c1 > 0, "both parts must be non-empty");
    }

    #[test]
    fn deterministic_across_repeats() {
        let g = build_graph(64, &grid_triples(8, 8));
        let opts = KahipOptions::default();
        let mut rng1 = SplitMix::new(opts.seed);
        let mut rng2 = SplitMix::new(opts.seed);
        let mut s1 = KahipStats::default();
        let mut s2 = KahipStats::default();
        let l1 = multilevel_bisection(&g, &opts, &mut rng1, &mut s1);
        let l2 = multilevel_bisection(&g, &opts, &mut rng2, &mut s2);
        assert_eq!(l1, l2);
    }

    #[test]
    fn balance_within_slack() {
        let g = build_graph(100, &grid_triples(10, 10));
        let opts = KahipOptions::default();
        let mut rng = SplitMix::new(opts.seed);
        let mut stats = KahipStats::default();
        let labels = multilevel_bisection(&g, &opts, &mut rng, &mut stats);
        let c0 = labels.iter().filter(|&&l| l == PART_A).count();
        let c1 = labels.iter().filter(|&&l| l == PART_B).count();
        let half_up = 100usize.div_ceil(2);
        let slack = ((1.0 + 0.20) * half_up as f64).floor() as usize;
        assert!(c0.max(c1) <= slack, "imbalance exceeded: {}|{}", c0, c1);
    }

    #[test]
    fn all_modes_produce_valid_bisection() {
        let g = build_graph(144, &grid_triples(12, 12));
        for mode in [KahipMode::Fast, KahipMode::Eco, KahipMode::Strong] {
            let opts = KahipOptions { seed: 7, mode };
            let mut rng = SplitMix::new(opts.seed);
            let mut stats = KahipStats::default();
            let labels = multilevel_bisection(&g, &opts, &mut rng, &mut stats);
            assert_eq!(labels.len(), 144);
            let c0 = labels.iter().filter(|&&l| l == PART_A).count();
            let c1 = labels.iter().filter(|&&l| l == PART_B).count();
            assert!(c0 > 0 && c1 > 0, "mode {:?}: empty part", mode);
        }
    }

    #[test]
    fn graph_to_undirected_preserves_edges() {
        let g = build_graph(9, &grid_triples(3, 3));
        let ug = graph_to_undirected(&g);
        assert_eq!(ug.n, 9);
        // 3x3 grid has 12 undirected edges => 24 entries in adjncy.
        assert_eq!(ug.adjncy.len(), 24);
        for v in 0..9 {
            // Neighbors sorted.
            let nbrs = ug.neighbors(v);
            for w in nbrs.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }
}
