//! Phase K1: graph data reduction (Ost-Schulz-Strash 2021).
//!
//! Applies four fill-preserving reduction rules in a fixed-point loop,
//! producing a smaller graph and an operation stack that can be replayed
//! in reverse to expand any elimination ordering on the reduced graph
//! back to the original vertex set.
//!
//! ## Rules
//! 1. **Degree-1 elimination** (with cascading) - leaves removed first.
//! 2. **Degree-2 path compression** - two sub-cases:
//!    - simplicial (endpoints adjacent): zero fill, no edge added;
//!    - non-simplicial: one fill edge `(u, w)` added.
//! 3. **Twin detection** - both open (`N(u) = N(v)`, `u ≁ v`) and
//!    closed (`N[u] = N[v]`, `u ∼ v`) twins.
//! 4. **Subset elimination** - `v` is dominated by a neighbor `u`
//!    if `N(v) \ {u} ⊆ N(u)`.
//!
//! ## Expansion
//! Eliminated vertices are anchored to a surviving vertex via
//! path-compressed union-find. Each eliminated vertex is inserted into
//! the final permutation immediately before its surviving anchor, in
//! the order they were removed.
//!
//! See `dev/research/ordering-kahip-k1.md` for the formal definitions,
//! proofs, and test-oracle construction.

use rslab_ordering_core::{CscPattern, OrderingError};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// One entry in the reduction operation stack.
///
/// Stored in application order. Expansion walks the stack in reverse
/// (newest-first) to rebuild the permutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReductionOp {
    /// Degree-1 leaf `v` removed; its sole neighbor was `owner`.
    /// Expansion: place `v` immediately before `owner`.
    Degree1 { v: usize, owner: usize },
    /// Degree-2 path interior compressed between endpoints `u` and `w`.
    /// `path` lists the interior vertices in traversal order
    /// (`u → path[0] → path[1] → ... → path[k-1] → w`).
    /// If `simplicial`, `u ∼ w` was already an edge (zero fill);
    /// otherwise the fill edge `(u, w)` was added to the reduced graph.
    /// Expansion: insert `path` before `u` in path order.
    Degree2Path {
        u: usize,
        w: usize,
        path: Vec<usize>,
        simplicial: bool,
    },
    /// Twin merge: `dup` collapsed into `rep`.
    /// `closed` distinguishes closed twins (`N[rep] = N[dup]`, adjacent)
    /// from open twins (`N(rep) = N(dup)`, non-adjacent).
    /// Expansion: place `dup` immediately before `rep`.
    Twin {
        rep: usize,
        dup: usize,
        closed: bool,
    },
    /// `v` was dominated by neighbor `owner`: `N(v) \ {owner} ⊆ N(owner)`.
    /// Expansion: place `v` immediately before `owner`.
    SubsetElim { v: usize, owner: usize },
}

/// Reduced graph + replay stack.
///
/// `n` is the number of surviving vertices; `col_ptr`/`row_idx` is the
/// reduced CSC in new-index space (same contract as
/// [`CscPattern`] but without the diagonal); `old_of_new[i]` is the
/// original vertex index of new vertex `i`.
#[derive(Debug, Clone)]
pub(crate) struct ReducedGraph {
    pub n: usize,
    pub col_ptr: Vec<i32>,
    pub row_idx: Vec<i32>,
    pub old_of_new: Vec<usize>,
    pub ops: Vec<ReductionOp>,
}

/// Internal mutable adjacency with per-vertex tombstones.
///
/// Diagonal is not stored. Adjacency is kept as a `BTreeSet<usize>`
/// for `O(log deg)` insert/remove and sorted iteration (needed for
/// canonical twin signatures).
struct MutAdj {
    alive: Vec<bool>,
    adj: Vec<BTreeSet<usize>>,
}

impl MutAdj {
    fn from_pattern(pattern: &CscPattern<'_>) -> Result<Self, OrderingError> {
        let n = pattern.n;
        let mut adj = vec![BTreeSet::<usize>::new(); n];
        for j in 0..n {
            let lo = pattern.col_ptr[j] as usize;
            let hi = pattern.col_ptr[j + 1] as usize;
            if lo > hi || hi > pattern.row_idx.len() {
                return Err(OrderingError::MalformedInput);
            }
            for &r in &pattern.row_idx[lo..hi] {
                if r < 0 || (r as usize) >= n {
                    return Err(OrderingError::MalformedInput);
                }
                let i = r as usize;
                if i == j {
                    continue; // drop diagonal
                }
                adj[j].insert(i);
                adj[i].insert(j); // enforce symmetry defensively
            }
        }
        Ok(Self {
            alive: vec![true; n],
            adj,
        })
    }

    fn degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    fn remove_vertex(&mut self, v: usize) {
        debug_assert!(self.alive[v]);
        let ns: Vec<usize> = self.adj[v].iter().copied().collect();
        for u in ns {
            self.adj[u].remove(&v);
        }
        self.adj[v].clear();
        self.alive[v] = false;
    }

    fn add_edge(&mut self, u: usize, w: usize) {
        if u == w {
            return;
        }
        self.adj[u].insert(w);
        self.adj[w].insert(u);
    }

    fn adjacent(&self, u: usize, w: usize) -> bool {
        self.adj[u].contains(&w)
    }
}

/// Which K1 rules the driver is allowed to apply.
///
/// Empirically on the RSLAB parity + large-matrix corpus, Rules 2-4
/// hurt fill dramatically on several matrices (vesuvio/vesuviou/
/// cresc132 blow up 40-50× when Rules 2-4 are enabled, even with a
/// correct expansion). Rule 1 alone is safe: pure degree-1 cascading
/// has no fill interaction with the downstream multilevel partitioner.
/// The higher rules remain implemented so that unit tests continue to
/// validate them, but the driver's default preset enables only Rule 1.
/// The reason for the Rule 2-4 regressions is an open question tracked
/// in `dev/tried-and-rejected.md` - see the K1 rollout entry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ReduceOptions {
    pub degree2_simplicial: bool,
    pub degree2_nonsimplicial: bool,
    pub twins: bool,
    pub subset: bool,
}

impl ReduceOptions {
    /// Rule 1 only - the driver's empirical safe choice.
    pub(crate) const fn conservative() -> Self {
        Self {
            degree2_simplicial: false,
            degree2_nonsimplicial: false,
            twins: false,
            subset: false,
        }
    }

    /// All four rules - used by unit tests.
    #[cfg(test)]
    pub(crate) const fn full() -> Self {
        Self {
            degree2_simplicial: true,
            degree2_nonsimplicial: true,
            twins: true,
            subset: true,
        }
    }
}

/// Apply the K1 fixed-point data reduction.
///
/// Returns `Some(ReducedGraph)` if the reduction shrank the graph
/// below `max_ratio * n_original`, else `None` (caller should proceed
/// without reduction).
///
/// `max_ratio` in `(0, 1]`: e.g. 0.7 accepts reductions of 30% or more.
/// `max_ratio >= 1.0` accepts any reduction including the empty case.
/// `opts` selects which reduction rules fire - see [`ReduceOptions`].
pub(crate) fn reduce_graph(
    pattern: &CscPattern<'_>,
    max_ratio: f64,
    opts: ReduceOptions,
) -> Result<Option<ReducedGraph>, OrderingError> {
    let n_original = pattern.n;
    if n_original == 0 {
        return Ok(Some(ReducedGraph {
            n: 0,
            col_ptr: vec![0],
            row_idx: Vec::new(),
            old_of_new: Vec::new(),
            ops: Vec::new(),
        }));
    }

    let mut g = MutAdj::from_pattern(pattern)?;
    let mut ops: Vec<ReductionOp> = Vec::new();

    loop {
        let mut progress = 0usize;
        progress += apply_degree1(&mut g, &mut ops);
        if opts.degree2_simplicial || opts.degree2_nonsimplicial {
            progress += apply_degree2(&mut g, &mut ops, opts.degree2_nonsimplicial);
        }
        if opts.twins {
            progress += apply_twins(&mut g, &mut ops);
        }
        if opts.subset {
            progress += apply_subset(&mut g, &mut ops);
        }
        if progress == 0 {
            break;
        }
    }

    let reduced_n = g.alive.iter().filter(|&&a| a).count();

    if (reduced_n as f64) > max_ratio * (n_original as f64) && reduced_n != 0 {
        return Ok(None);
    }

    // Relabel surviving vertices 0..reduced_n in original-index order.
    let mut old_of_new: Vec<usize> = Vec::with_capacity(reduced_n);
    let mut new_of_old: Vec<i32> = vec![-1; n_original];
    for (v, slot) in new_of_old.iter_mut().enumerate() {
        if g.alive[v] {
            *slot = old_of_new.len() as i32;
            old_of_new.push(v);
        }
    }

    let mut col_ptr: Vec<i32> = Vec::with_capacity(reduced_n + 1);
    col_ptr.push(0);
    let mut row_idx: Vec<i32> = Vec::new();
    for &old_v in &old_of_new {
        // adjacency is already sorted (BTreeSet iteration).
        for &u_old in &g.adj[old_v] {
            let u_new = new_of_old[u_old];
            debug_assert!(u_new >= 0, "alive neighbor must be relabeled");
            row_idx.push(u_new);
        }
        col_ptr.push(row_idx.len() as i32);
    }

    Ok(Some(ReducedGraph {
        n: reduced_n,
        col_ptr,
        row_idx,
        old_of_new,
        ops,
    }))
}

/// Rule 1: remove all degree-1 vertices, cascading.
///
/// Returns the number of vertices removed in this invocation.
fn apply_degree1(g: &mut MutAdj, ops: &mut Vec<ReductionOp>) -> usize {
    let mut removed = 0usize;
    // Worklist of vertices currently at degree 1.
    let n = g.adj.len();
    let mut work: Vec<usize> = (0..n).filter(|&v| g.alive[v] && g.degree(v) == 1).collect();
    while let Some(v) = work.pop() {
        if !g.alive[v] || g.degree(v) != 1 {
            continue;
        }
        let owner = *g.adj[v].iter().next().expect("degree-1 has one nbr");
        g.remove_vertex(v);
        ops.push(ReductionOp::Degree1 { v, owner });
        removed += 1;
        if g.alive[owner] && g.degree(owner) == 1 {
            work.push(owner);
        }
    }
    removed
}

/// Rule 2: compress maximal degree-2 chains between branch endpoints.
///
/// A "branch endpoint" is a surviving vertex whose degree is not 2
/// (could be 1 after cascading, or ≥3). We walk from each endpoint
/// along every degree-2 neighbor to discover the full path, then
/// collapse the interior.
fn apply_degree2(g: &mut MutAdj, ops: &mut Vec<ReductionOp>, allow_nonsimplicial: bool) -> usize {
    let mut removed = 0usize;
    let n = g.adj.len();
    // `skip[v]` marks vertices that belong to a degree-2 chain whose
    // branch-endpoint walk returned u == w (pure cycle attached at a
    // single branch, or wholly-enclosed cycle). We skip them for the
    // rest of this pass so subsequent seeds can find other chains.
    let mut skip = vec![false; n];
    // O17 (repo-review-2026-06-09): the seed scan below restarts from index 0
    // on every outer iteration, so a graph that is one long degree-2 chain
    // costs O(n^2) in seed-scanning alone. A non-rewinding cursor is NOT a safe
    // drop-in replacement: the simplicial collapse below (lines ~399-405) adds
    // no compensating (u, w) edge when `u ~ w` already, so removing the chain
    // interior drops each branch endpoint by one degree - a degree-3 endpoint
    // can become a fresh degree-2 seed at an index *below* the current `seed`.
    // The from-0 scan always picks the lowest-index eligible vertex, so it
    // collapses that endpoint within this same call; a cursor advanced past it
    // would instead defer the collapse to the next fixed-point round (the
    // `reduce_graph` loop), reordering the emitted `Degree2Path` ops and thus
    // changing the reconstructed permutation (`expand_permutation` replays the
    // op stack in reverse). The order-preserving O(n log n) fix is a min-index
    // worklist (a binary heap keyed by vertex index, with a lazy
    // alive/!skip/degree==2 staleness check on pop), not a cursor. Left as-is
    // for now: Rule 2 is test-only - the driver runs
    // `ReduceOptions::conservative()` (Rule 1 only), so this cost is latent.
    'outer: loop {
        // Find any unskipped degree-2 vertex (lowest index first).
        let start = (0..n).find(|&v| g.alive[v] && !skip[v] && g.degree(v) == 2);
        let Some(seed) = start else { break };

        // Walk backwards from seed to a branch endpoint u (a vertex of
        // degree != 2) or detect a pure cycle.
        let mut visited: Vec<usize> = vec![seed];
        let (u, first_step) = {
            let mut cur = seed;
            let mut prev: Option<usize> = None;
            loop {
                let mut next_opt: Option<usize> = None;
                for &nbr in g.adj[cur].iter() {
                    if Some(nbr) != prev {
                        next_opt = Some(nbr);
                        break;
                    }
                }
                let Some(next) = next_opt else {
                    break (cur, prev.unwrap_or(cur));
                };
                if g.degree(next) != 2 {
                    break (next, cur);
                }
                if next == seed {
                    // Wrapped around a pure cycle: no branch endpoint.
                    break (cur, prev.unwrap_or(cur));
                }
                visited.push(next);
                prev = Some(cur);
                cur = next;
            }
        };

        // Pure-cycle case: mark every visited chain vertex as skipped
        // and move on - Rule 3 (twins) may still collapse the cycle
        // later this pass.
        if g.degree(u) == 2 {
            for &v in &visited {
                skip[v] = true;
            }
            continue 'outer;
        }

        // Walk forward from u through the chain to collect the interior
        // path and pick up the far branch endpoint w.
        let mut path: Vec<usize> = Vec::new();
        let mut cur = first_step;
        let mut prev = u;
        loop {
            if g.degree(cur) != 2 {
                break;
            }
            path.push(cur);
            let mut next_opt: Option<usize> = None;
            for &nbr in g.adj[cur].iter() {
                if nbr != prev {
                    next_opt = Some(nbr);
                    break;
                }
            }
            let Some(next) = next_opt else {
                break;
            };
            prev = cur;
            cur = next;
        }
        let w = cur;

        // If u == w, the "chain" is a cycle-through-a-single-branch -
        // compressing would require a self-loop on u. Skip the chain.
        if u == w || path.is_empty() {
            for &v in &path {
                skip[v] = true;
            }
            skip[seed] = true;
            continue 'outer;
        }

        let simplicial = g.adjacent(u, w);
        if !simplicial && !allow_nonsimplicial {
            // Non-simplicial Rule 2 adds a fill edge (u, w) to the reduced
            // graph. Empirically this harms fill on our corpus: the
            // extra edge perturbs the multilevel partitioner's cut
            // choices. Mark the chain as skipped for the remainder
            // of this pass and move on.
            for &v in &path {
                skip[v] = true;
            }
            continue 'outer;
        }
        for &v in &path {
            g.remove_vertex(v);
            removed += 1;
        }
        if !simplicial {
            g.add_edge(u, w);
        }
        ops.push(ReductionOp::Degree2Path {
            u,
            w,
            path,
            simplicial,
        });
    }
    removed
}

/// Rule 3: detect and merge open + closed twins.
///
/// Open twin: `u ≁ v`, `N(u) = N(v)`.
/// Closed twin: `u ∼ v`, `N[u] = N[v]` where `N[x] = N(x) ∪ {x}`.
fn apply_twins(g: &mut MutAdj, ops: &mut Vec<ReductionOp>) -> usize {
    let mut removed = 0usize;
    let n = g.adj.len();

    // Closed twins first: group by closed signature, merge within group.
    // `BTreeMap` (not `HashMap`) so the group iteration order - and therefore
    // the emitted `Twin` op-stack order - is sorted by signature and identical
    // run-to-run; a `HashMap` here iterates in `RandomState`-seed order and
    // breaks the crate's determinism contract (O2, repo-review-2026-06-09.md).
    let mut closed_groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    for v in 0..n {
        if !g.alive[v] {
            continue;
        }
        let mut sig: Vec<usize> = g.adj[v].iter().copied().collect();
        // Insert v in sorted order.
        let pos = sig.partition_point(|&x| x < v);
        sig.insert(pos, v);
        closed_groups.entry(sig).or_default().push(v);
    }
    for (_sig, mut group) in closed_groups {
        if group.len() < 2 {
            continue;
        }
        group.sort();
        let rep = group[0];
        // Sanity: rep must still be alive.
        if !g.alive[rep] {
            continue;
        }
        for &dup in group.iter().skip(1) {
            if !g.alive[dup] {
                continue;
            }
            // Signatures matched → closed twins are automatically adjacent
            // (both v and u appear in the shared closed signature, and
            // are therefore neighbors of each other).
            debug_assert!(g.adjacent(rep, dup));
            g.remove_vertex(dup);
            ops.push(ReductionOp::Twin {
                rep,
                dup,
                closed: true,
            });
            removed += 1;
        }
    }

    // Open twins: group by open signature, then within group find
    // pairs that are mutually non-adjacent. `BTreeMap` for the same
    // determinism reason as `closed_groups` above (O2).
    let mut open_groups: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
    for v in 0..n {
        if !g.alive[v] {
            continue;
        }
        let sig: Vec<usize> = g.adj[v].iter().copied().collect();
        if sig.is_empty() {
            continue; // isolated vertex - not useful as a twin
        }
        open_groups.entry(sig).or_default().push(v);
    }
    for (_sig, mut group) in open_groups {
        if group.len() < 2 {
            continue;
        }
        group.sort();
        let rep = group[0];
        if !g.alive[rep] {
            continue;
        }
        for &dup in group.iter().skip(1) {
            if !g.alive[dup] {
                continue;
            }
            // Open twin condition requires they not be adjacent.
            // Since N(rep) == N(dup), adjacency would mean rep ∈ N(rep),
            // a self-loop, which we forbid. So this holds automatically
            // - but assert defensively.
            debug_assert!(!g.adjacent(rep, dup));
            g.remove_vertex(dup);
            ops.push(ReductionOp::Twin {
                rep,
                dup,
                closed: false,
            });
            removed += 1;
        }
    }

    removed
}

/// Rule 4: subset elimination via mark-array.
///
/// For each surviving `u`, mark `N(u)`. For each `v ∈ N(u)`, count
/// how many of `v`'s neighbors (other than `u`) are marked; if this
/// equals `deg(v) - 1`, then `N(v) \ {u} ⊆ N(u)` and `v` can be
/// eliminated with no extra fill.
///
/// Conservative policy for the first cut: mark `v` at most once per
/// pass, and skip vertices of degree > threshold where the cost of
/// checking would dominate. `deg_threshold` is chosen to balance
/// work against reduction yield; exhaustive application is left to
/// a future tuning phase.
fn apply_subset(g: &mut MutAdj, ops: &mut Vec<ReductionOp>) -> usize {
    let mut removed = 0usize;
    let n = g.adj.len();
    let mut mark: Vec<u32> = vec![0; n];
    let mut token: u32 = 0;

    for u in 0..n {
        if !g.alive[u] {
            continue;
        }
        if g.degree(u) == 0 {
            continue;
        }
        token = token.wrapping_add(1);
        if token == 0 {
            // token overflow: reset
            for m in mark.iter_mut() {
                *m = 0;
            }
            token = 1;
        }
        for &nbr in g.adj[u].iter() {
            mark[nbr] = token;
        }
        // Iterate over a snapshot because we may remove v during the loop.
        let neighbors: Vec<usize> = g.adj[u].iter().copied().collect();
        for v in neighbors {
            if !g.alive[v] || !g.alive[u] {
                continue;
            }
            if v == u {
                continue;
            }
            let deg_v = g.degree(v);
            if deg_v == 0 {
                continue;
            }
            // Count v's neighbors (other than u) that are marked.
            let mut count = 0usize;
            for &w in g.adj[v].iter() {
                if w == u {
                    continue;
                }
                if mark[w] == token {
                    count += 1;
                }
            }
            // N(v) \ {u} ⊆ N(u) ↔ count == deg(v) - 1.
            if count + 1 == deg_v {
                // Do not eliminate u via this rule (would be circular).
                // Also require v != u (already guarded).
                g.remove_vertex(v);
                ops.push(ReductionOp::SubsetElim { v, owner: u });
                removed += 1;
            }
        }
    }
    removed
}

/// Expand a permutation from the reduced graph back to original indices.
///
/// `reduced_perm[i]` is a new-index in `0..reduced.n` - the i-th vertex
/// to eliminate on the reduced graph. The returned permutation is in
/// original-index space and has length `n_original`.
///
/// Convention: eliminated-during-reduction vertices are inserted
/// immediately before their (path-compressed) anchor, in op-stack order.
pub(crate) fn expand_permutation(
    reduced: &ReducedGraph,
    reduced_perm: &[i32],
    n_original: usize,
) -> Result<Vec<i32>, OrderingError> {
    if reduced_perm.len() != reduced.n {
        return Err(OrderingError::MalformedInput);
    }
    if reduced.old_of_new.len() != reduced.n {
        return Err(OrderingError::MalformedInput);
    }

    // anchor[v]: the vertex that v's elimination is anchored to. For
    // surviving vertices, anchor[v] = v. For eliminated vertices, walk
    // up to a surviving vertex via path compression.
    let mut anchor: Vec<usize> = (0..n_original).collect();

    // Initialize anchors by processing the op stack in FORWARD order.
    // Each eliminated vertex gets its immediate anchor (which may be
    // another eliminated vertex; resolved at read time via compression).
    for op in &reduced.ops {
        match op {
            ReductionOp::Degree1 { v, owner } => {
                anchor[*v] = *owner;
            }
            ReductionOp::Degree2Path { u, path, .. } => {
                // All path interior vertices anchor to u.
                for &v in path {
                    anchor[v] = *u;
                }
            }
            ReductionOp::Twin { rep, dup, .. } => {
                anchor[*dup] = *rep;
            }
            ReductionOp::SubsetElim { v, owner } => {
                anchor[*v] = *owner;
            }
        }
    }

    // Path-compress every vertex to its ultimate (surviving) anchor.
    // A surviving vertex s satisfies anchor[s] == s initially (since
    // we never wrote to survivors). Verify that invariant.
    let mut survives = vec![false; n_original];
    for &old in &reduced.old_of_new {
        survives[old] = true;
    }
    for v in 0..n_original {
        if survives[v] && anchor[v] != v {
            return Err(OrderingError::Internal(
                "KaHIP K1: surviving vertex had an anchor written - expansion invariant broken",
            ));
        }
    }
    // Compression.
    for v in 0..n_original {
        let mut cur = v;
        while !survives[cur] {
            cur = anchor[cur];
        }
        anchor[v] = cur;
    }

    // Build a reverse map pos_of_old[v] = reduced-perm position of
    // surviving vertex v, for non-surviving vertices pos = i32::MAX.
    // Used below to choose the earlier endpoint for Rule 2 paths.
    let mut pos_of_old: Vec<i32> = vec![i32::MAX; n_original];
    for (new_pos, &new_idx) in reduced_perm.iter().enumerate() {
        if new_idx < 0 || (new_idx as usize) >= reduced.n {
            return Err(OrderingError::MalformedInput);
        }
        let old = reduced.old_of_new[new_idx as usize];
        pos_of_old[old] = new_pos as i32;
    }

    // Group eliminated vertices by their ultimate anchor, preserving
    // op-stack order (newer ops inserted later → appear later in the
    // group, which means they are eliminated later in the expanded
    // perm but still before the anchor). For path interiors, we want
    // the path's order preserved.
    //
    // For Rule 2 (non-simplicial) path compression, the fill-preservation
    // invariant requires the path to be eliminated before BOTH of the
    // compressed endpoints u and w, not just before u. If the reduced
    // permutation places w earlier than u, anchoring the path at u
    // alone causes the expanded graph to still contain the path when w
    // is eliminated, producing extra fill through the path-interior
    // neighbors. Fix: anchor the path to whichever of the two endpoints'
    // ultimate anchors has the lower reduced-perm position.
    //
    // Implementation: walk the op stack in forward order and append
    // eliminated vertices to their group. Each group's vertex list
    // is then the desired pre-anchor elimination sequence.
    // `HashMap` is safe here (unlike `closed_groups`/`open_groups`, O2): this
    // map is only consumed via keyed `group.remove(&old)` in the deterministic
    // `reduced_perm` order below - it is never iterated for output order.
    let mut group: HashMap<usize, Vec<usize>> = HashMap::new();
    for op in &reduced.ops {
        match op {
            ReductionOp::Degree1 { v, .. } | ReductionOp::SubsetElim { v, .. } => {
                group.entry(anchor[*v]).or_default().push(*v);
            }
            ReductionOp::Twin { dup, .. } => {
                group.entry(anchor[*dup]).or_default().push(*dup);
            }
            ReductionOp::Degree2Path { u, w, path, .. } => {
                if path.is_empty() {
                    continue;
                }
                let au = anchor[*u];
                let aw = anchor[*w];
                // Pick the anchor whose reduced-perm position is lower.
                // Ties: prefer au for determinism.
                let chosen = if pos_of_old[aw] < pos_of_old[au] {
                    aw
                } else {
                    au
                };
                let bucket = group.entry(chosen).or_default();
                for &v in path {
                    bucket.push(v);
                }
            }
        }
    }

    // Emit the final permutation: for each reduced-perm vertex, emit
    // its group (pre-anchor eliminations) followed by the anchor itself.
    // `new_idx` already validated while building `pos_of_old` above.
    let mut out: Vec<i32> = Vec::with_capacity(n_original);
    for &new_idx in reduced_perm {
        let old = reduced.old_of_new[new_idx as usize];
        if let Some(pre) = group.remove(&old) {
            for v in pre {
                out.push(v as i32);
            }
        }
        out.push(old as i32);
    }

    // Any vertices still in group (anchored to something not in
    // reduced_perm) indicate a malformed reduced_perm.
    if !group.is_empty() {
        return Err(OrderingError::Internal(
            "KaHIP K1: reduced_perm missing surviving vertices with anchored eliminations",
        ));
    }
    if out.len() != n_original {
        return Err(OrderingError::Internal(
            "KaHIP K1: expansion produced wrong vertex count",
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- pattern builders (full-symmetric CSC with no diagonal) ----

    fn make_pattern(n: usize, edges: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
        let mut cols: Vec<BTreeSet<i32>> = vec![BTreeSet::new(); n];
        for &(a, b) in edges {
            if a == b {
                continue;
            }
            cols[a].insert(b as i32);
            cols[b].insert(a as i32);
        }
        let mut col_ptr: Vec<i32> = Vec::with_capacity(n + 1);
        col_ptr.push(0);
        let mut row_idx: Vec<i32> = Vec::new();
        for col in cols {
            for r in col {
                row_idx.push(r);
            }
            col_ptr.push(row_idx.len() as i32);
        }
        (col_ptr, row_idx)
    }

    fn path_edges(n: usize) -> Vec<(usize, usize)> {
        (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect()
    }

    fn star_edges(n: usize) -> Vec<(usize, usize)> {
        (1..n).map(|i| (0, i)).collect()
    }

    fn complete_edges(n: usize) -> Vec<(usize, usize)> {
        let mut e = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                e.push((i, j));
            }
        }
        e
    }

    fn bipartite_edges(m: usize, n: usize) -> Vec<(usize, usize)> {
        // K_{m,n}: vertices 0..m on one side, m..m+n on the other.
        let mut e = Vec::new();
        for i in 0..m {
            for j in 0..n {
                e.push((i, m + j));
            }
        }
        e
    }

    fn is_permutation(p: &[i32], n: usize) -> bool {
        if p.len() != n {
            return false;
        }
        let mut seen = vec![false; n];
        for &v in p {
            if v < 0 || (v as usize) >= n || seen[v as usize] {
                return false;
            }
            seen[v as usize] = true;
        }
        true
    }

    // ---- Rule 3: twin-merge determinism (O2) ----

    /// O2 (repo-review-2026-06-09.md): twin reduction grouped vertices in
    /// `HashMap`s (`closed_groups`, `open_groups`) and iterated them
    /// directly, so the order twin groups were merged - and therefore the
    /// `ReductionOp::Twin` stack order - varied run-to-run with the
    /// per-instance `RandomState` seed, violating the crate's determinism
    /// contract. With independent twin groups, the map iteration order is
    /// the only thing deciding the op order.
    ///
    /// Reproduction: 12 disjoint triangles. Each triangle
    /// `{3i, 3i+1, 3i+2}` is a closed-twin group (all three share the
    /// closed signature), so `apply_twins` emits two `Twin` ops per
    /// triangle. The merge *result* is order-independent, but the op-stack
    /// *order* is whatever order the group map iterates. Running
    /// `apply_twins` on fresh graphs must produce byte-identical ops every
    /// time. Pre-fix (`HashMap`) the orders diverge across runs; post-fix
    /// (`BTreeMap`) they are sorted by signature and identical. Oracle:
    /// determinism - same input must give the same output.
    #[test]
    fn apply_twins_op_order_is_deterministic() {
        const K: usize = 12;
        let n = 3 * K;
        let mut edges = Vec::new();
        for i in 0..K {
            let (a, b, c) = (3 * i, 3 * i + 1, 3 * i + 2);
            edges.push((a, b));
            edges.push((b, c));
            edges.push((a, c));
        }
        let (cp, ri) = make_pattern(n, &edges);
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");

        let run = || {
            let mut g = MutAdj::from_pattern(&pat).expect("valid");
            let mut ops = Vec::new();
            apply_twins(&mut g, &mut ops);
            ops
        };

        let baseline = run();
        // Sanity: the reduction actually fired (two Twin dups per triangle).
        assert_eq!(baseline.len(), 2 * K, "each triangle merges two dups");
        // Determinism contract: every independent run is byte-identical.
        for _ in 0..32 {
            assert_eq!(run(), baseline, "twin op order must be deterministic");
        }
    }

    // ---- Rule 1: degree-1 cascading ----

    #[test]
    fn star_n10_collapses_via_full_cascade() {
        // Star on 10 vertices: hub 0, leaves 1..9 each degree 1.
        // Rule 1 cascades: leaves removed first, then as the hub's
        // degree drops it eventually becomes degree-1 itself and is
        // also eliminated. One arbitrary vertex survives as the "base
        // case" of the cascade.
        let n = 10;
        let (cp, ri) = make_pattern(n, &star_edges(n));
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let reduced = reduce_graph(&pat, 1.0, ReduceOptions::full())
            .unwrap()
            .expect("reduces");
        assert_eq!(reduced.n, 1, "exactly one vertex survives");
        assert_eq!(reduced.ops.len(), 9, "9 vertices removed");
        for op in &reduced.ops {
            assert!(matches!(op, ReductionOp::Degree1 { .. }));
        }
        // Expand identity perm on the reduced graph - must be a bijection.
        let expanded = expand_permutation(&reduced, &[0], n).unwrap();
        assert!(is_permutation(&expanded, n));
        // The surviving vertex appears last in the expanded perm.
        assert_eq!(expanded[n - 1] as usize, reduced.old_of_new[0]);
    }

    // ---- Rule 2: degree-2 path compression ----

    #[test]
    fn path_n5_collapses_to_branch_endpoints() {
        // Path: 0-1-2-3-4. Endpoints 0, 4 have degree 1; interior
        // 1, 2, 3 have degree 2. Rule 1 eats 0 and 4 first (they
        // become the only alive vertices' owners); then 1-2-3 is a
        // degree-2 chain with branch endpoints of degree... wait:
        // once 0 and 4 are removed, vertices 1 and 3 become degree 1,
        // and they cascade out via rule 1. Vertex 2 becomes degree 0.
        // So the whole path collapses to 1 isolated vertex (pivot 2).
        let n = 5;
        let (cp, ri) = make_pattern(n, &path_edges(n));
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let reduced = reduce_graph(&pat, 1.0, ReduceOptions::full())
            .unwrap()
            .expect("reduces");
        assert_eq!(reduced.n, 1, "path collapses to a single vertex");
        let expanded = expand_permutation(&reduced, &[0], n).unwrap();
        assert!(is_permutation(&expanded, n));
    }

    #[test]
    fn triangle_with_tail_collapses_via_rule1_then_closed_twins() {
        // Graph: triangle {0,1,2} + tail 2-3-4-5.
        //   Rule 1 eats 5→4→3 (cascade; 3 Degree1 ops), leaving the
        //   triangle alive with all three vertices at degree 2.
        //   Rule 2 detects the cycle (u==w) and skips it.
        //   Rule 3 (closed twins) sees all three triangle vertices
        //   share the closed signature {0,1,2}, merging them to one.
        let n = 6;
        let mut edges = vec![(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5)];
        edges.sort();
        let (cp, ri) = make_pattern(n, &edges);
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let reduced = reduce_graph(&pat, 1.0, ReduceOptions::full())
            .unwrap()
            .expect("reduces");
        assert_eq!(reduced.n, 1, "triangle collapses via closed twins");
        let d1_count = reduced
            .ops
            .iter()
            .filter(|op| matches!(op, ReductionOp::Degree1 { .. }))
            .count();
        assert_eq!(d1_count, 3, "tail cascades: 5, 4, 3");
        let closed_twin_count = reduced
            .ops
            .iter()
            .filter(|op| matches!(op, ReductionOp::Twin { closed: true, .. }))
            .count();
        assert_eq!(
            closed_twin_count, 2,
            "triangle has 3 vertices → 2 twin merges"
        );
        // Expansion is a valid bijection.
        let expanded = expand_permutation(&reduced, &[0], n).unwrap();
        assert!(is_permutation(&expanded, n));
    }

    #[test]
    fn degree2_compression_fires_on_isolated_chain() {
        // Construct a graph where Rule 2 *must* fire first:
        //   Two "bumpy" hubs that Rule 1 can't cascade through and that
        //   have distinct closed signatures (so Rule 3 doesn't shortcut).
        //   Hub A: vertex 0 with neighbors {1, 2} plus chain entry 3.
        //          Edges {0-1, 0-2, 1-2, 0-3} make {0,1,2} a triangle
        //          with 0 as a "branch hub" via edge 0-3. deg(0)=3.
        //   Chain: 3-4-5 (3 interior-ish).
        //   Hub B: vertex 5 with neighbors {6, 7} plus chain exit 4.
        //          Edges {5-6, 5-7, 6-7, 4-5}. deg(5)=3.
        // Rule 1 has no degree-1 vertex initially. Rule 2 sees vertex 4
        // with degree 2 (neighbors 3, 5) between branch endpoints 3 and
        // 5 - BUT 3 has degree 2 too (neighbors 0 and 4). So walking
        // backward from seed=4 lands on u=0 (the first deg ≥ 3 vertex).
        // Then forward through 3, 4 lands at w=5 (deg 3). Endpoints 0, 5
        // are not adjacent → non-simplicial compression with path [3, 4].
        let n = 8;
        let mut edges = vec![
            (0, 1),
            (0, 2),
            (1, 2), // left triangle
            (0, 3), // hub A → chain
            (3, 4),
            (4, 5), // chain
            (5, 6),
            (5, 7),
            (6, 7), // right triangle (5 is hub B)
        ];
        edges.sort();
        let (cp, ri) = make_pattern(n, &edges);
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let reduced = reduce_graph(&pat, 1.0, ReduceOptions::full())
            .unwrap()
            .expect("reduces");
        let path_ops: Vec<&ReductionOp> = reduced
            .ops
            .iter()
            .filter(|op| matches!(op, ReductionOp::Degree2Path { .. }))
            .collect();
        assert!(
            !path_ops.is_empty(),
            "Rule 2 path compression must fire at least once"
        );
        // One of those ops must be the hub-A to hub-B bridge with
        // distinct endpoints and non-simplicial flag.
        let bridge = path_ops
            .iter()
            .find(|op| {
                matches!(op, ReductionOp::Degree2Path { u, w, simplicial, path }
                    if *u != *w && !*simplicial && path.len() >= 2)
            })
            .expect("non-simplicial bridge compression between distinct hubs");
        if let ReductionOp::Degree2Path { u, w, .. } = bridge {
            // Hub-A and Hub-B are vertices 0 and 5 in the input.
            let endpoints = std::collections::BTreeSet::from([*u, *w]);
            let expected = std::collections::BTreeSet::from([0usize, 5usize]);
            assert_eq!(endpoints, expected, "bridge endpoints must be 0 and 5");
        }
    }

    // ---- Rule 3: twin detection ----

    #[test]
    fn k4_closed_twins_collapse_to_one() {
        // K4: every pair is a closed twin (N[u] = {all four vertices}).
        // Expected: rep=0 survives, 1,2,3 all merged as closed twins.
        let n = 4;
        let (cp, ri) = make_pattern(n, &complete_edges(n));
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let reduced = reduce_graph(&pat, 1.0, ReduceOptions::full())
            .unwrap()
            .expect("reduces");
        assert_eq!(reduced.n, 1);
        assert_eq!(reduced.old_of_new[0], 0);
        let twin_count = reduced
            .ops
            .iter()
            .filter(|op| matches!(op, ReductionOp::Twin { closed: true, .. }))
            .count();
        assert_eq!(twin_count, 3);
    }

    #[test]
    fn k_2_3_collapses_via_degree2_then_closed_twins() {
        // K_{2,3}: vertices 0,1 (size-2 side); 2,3,4 (size-3 side).
        // All of {2,3,4} have degree 2 with neighbors {0,1} - the
        // degree-2 rule processes them first. Walking backward from
        // seed=2 lands on u=0 (deg 3), then forward through 2 reaches
        // w=1 (deg 3 after it loses 2). First compression: path=[2],
        // u=0, w=1, simplicial=false (0-1 not adjacent) → edge 0-1
        // added. Second compression: seed=3, u=0, w=1, simplicial=true
        // (0-1 now exists) → remove 3. Same for 4. Then closed twins
        // merges 0 and 1 (both have closed_sig=[0,1]). Final n=1.
        let n = 5;
        let (cp, ri) = make_pattern(n, &bipartite_edges(2, 3));
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let reduced = reduce_graph(&pat, 1.0, ReduceOptions::full())
            .unwrap()
            .expect("reduces");
        assert_eq!(reduced.n, 1, "K_2_3 collapses to a single vertex");
        let path_count = reduced
            .ops
            .iter()
            .filter(|op| matches!(op, ReductionOp::Degree2Path { .. }))
            .count();
        assert_eq!(
            path_count, 2,
            "two degree-2 compressions before triangle forms"
        );
        let closed_twin_count = reduced
            .ops
            .iter()
            .filter(|op| matches!(op, ReductionOp::Twin { closed: true, .. }))
            .count();
        assert_eq!(closed_twin_count, 2, "triangle collapses via 2 twin merges");
    }

    // ---- Expansion bijection ----

    #[test]
    fn expand_identity_is_bijection_on_mixed_graph() {
        // A graph that exercises all rules:
        //   Triangle {0,1,2} (twins within)
        //   Path 2-3-4 (degree-2 chain)
        //   Leaf 5-4 (degree-1)
        let n = 6;
        let mut edges = vec![(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5)];
        edges.sort();
        let (cp, ri) = make_pattern(n, &edges);
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let reduced = reduce_graph(&pat, 1.0, ReduceOptions::full())
            .unwrap()
            .expect("reduces");
        // Identity perm on reduced graph.
        let red_perm: Vec<i32> = (0..reduced.n as i32).collect();
        let expanded = expand_permutation(&reduced, &red_perm, n).unwrap();
        assert!(is_permutation(&expanded, n), "expansion is a bijection");
    }

    #[test]
    fn isolated_vertices_survive_as_singletons() {
        // Three isolated vertices: no reductions apply (deg=0 for all).
        let n = 3;
        let (cp, ri) = make_pattern(n, &[]);
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let reduced = reduce_graph(&pat, 1.0, ReduceOptions::full())
            .unwrap()
            .expect("reduces");
        assert_eq!(reduced.n, 3);
        assert!(reduced.ops.is_empty());
        let expanded = expand_permutation(&reduced, &[0, 1, 2], n).unwrap();
        assert_eq!(expanded, vec![0, 1, 2]);
    }

    #[test]
    fn empty_graph_reduces_trivially() {
        let cp: Vec<i32> = vec![0];
        let ri: Vec<i32> = vec![];
        let pat = CscPattern::new(0, &cp, &ri).expect("valid");
        let reduced = reduce_graph(&pat, 1.0, ReduceOptions::full())
            .unwrap()
            .expect("reduces");
        assert_eq!(reduced.n, 0);
        assert!(reduced.ops.is_empty());
        let expanded = expand_permutation(&reduced, &[], 0).unwrap();
        assert!(expanded.is_empty());
    }

    // ---- max_ratio gate ----

    #[test]
    fn max_ratio_rejects_weak_reductions() {
        // Isolated vertices: zero reduction. With max_ratio < 1.0 the
        // reducer must return None (reduced == original).
        let n = 3;
        let (cp, ri) = make_pattern(n, &[]);
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let outcome = reduce_graph(&pat, 0.5, ReduceOptions::full()).unwrap();
        assert!(outcome.is_none(), "no reduction, must report None");
    }

    #[test]
    fn max_ratio_accepts_full_collapse() {
        // Star: collapses to 1 vertex (90% reduction) - must pass any
        // reasonable max_ratio.
        let n = 10;
        let (cp, ri) = make_pattern(n, &star_edges(n));
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let outcome = reduce_graph(&pat, 0.5, ReduceOptions::full()).unwrap();
        assert!(outcome.is_some(), "90% reduction must pass");
    }

    // ---- reduced CSC invariants ----

    #[test]
    fn reduced_csc_is_full_symmetric_no_diagonal() {
        // Dumbbell after path compression should emit a CSC where:
        // - row indices are sorted within each column,
        // - no diagonal entries,
        // - structurally symmetric.
        let n = 8;
        let mut edges = vec![
            (0, 1),
            (1, 2),
            (0, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (5, 7),
        ];
        edges.sort();
        let (cp, ri) = make_pattern(n, &edges);
        let pat = CscPattern::new(n, &cp, &ri).expect("valid");
        let reduced = reduce_graph(&pat, 1.0, ReduceOptions::full())
            .unwrap()
            .expect("reduces");
        let nr = reduced.n;
        assert_eq!(reduced.col_ptr.len(), nr + 1);
        // No diagonal, sorted within column.
        for j in 0..nr {
            let lo = reduced.col_ptr[j] as usize;
            let hi = reduced.col_ptr[j + 1] as usize;
            let col = &reduced.row_idx[lo..hi];
            assert!(col.windows(2).all(|w| w[0] < w[1]), "sorted & unique");
            assert!(col.iter().all(|&r| r as usize != j), "no diagonal");
        }
        // Structural symmetry.
        let has_edge = |a: usize, b: usize| -> bool {
            let lo = reduced.col_ptr[a] as usize;
            let hi = reduced.col_ptr[a + 1] as usize;
            reduced.row_idx[lo..hi].binary_search(&(b as i32)).is_ok()
        };
        for j in 0..nr {
            let lo = reduced.col_ptr[j] as usize;
            let hi = reduced.col_ptr[j + 1] as usize;
            for &r in &reduced.row_idx[lo..hi] {
                assert!(has_edge(r as usize, j), "asymmetric edge at ({}, {})", j, r);
            }
        }
    }
}
