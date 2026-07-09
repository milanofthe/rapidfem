//! Hungarian algorithm for the minimum-cost perfect bipartite
//! matching problem.
//!
//! The core kernel for MC64-style scaling. Given a sparse
//! non-negative cost matrix in CSC form, finds a perfect matching
//! (row-to-column) minimizing total cost, and returns both the
//! matching permutation and the optimal dual variables from the
//! LP dual. Those duals are what get exponentiated into the
//! row/column scalings in `mc64.rs`.
//!
//! Reference: citet:duff2001mc64 §4. Source model:
//! `ref/spral/src/scaling.f90::hungarian_match` (lines 938-1171),
//! itself a clean-room rewrite of HSL_MC80. The algorithm is the
//! standard shortest-augmenting-path variant - each augmenting
//! path is a Dijkstra-like search on the reduced-cost graph, and
//! the dual variables are updated to preserve complementary
//! slackness.
//!
//! PHASE 2.2.1 STATUS: real implementation (Step 3). Follows SPRAL's
//! `hungarian_match` / `hungarian_init_heurisitic` line-by-line, with
//! Rust-idiomatic naming and a custom index-based binary min-heap
//! that supports decrease-key (mirroring SPRAL's `q`/`d`/`l` arrays).
//! The custom heap is used instead of `std::collections::BinaryHeap`
//! because BinaryHeap lacks decrease-key, and lazy deletion would
//! require wrapping `f64` distances in a tie-breakable ordered type -
//! mirroring SPRAL's index-based heap produces cleaner code.

/// A sparse non-negative cost graph for the Hungarian algorithm.
///
/// Stored in CSC format on a square pattern (rows = cols = n).
/// All costs must be finite and non-negative; the MC64 wrapper
/// ensures this via per-column normalization of log absolute values.
/// Explicit zero entries in the pattern are allowed and represent
/// "cost 0" edges (which are the column-maximum entries after
/// normalization).
#[derive(Debug, Clone)]
pub(crate) struct CostGraph {
    pub n: usize,
    pub col_ptr: Vec<usize>,
    pub row_idx: Vec<usize>,
    pub cost: Vec<f64>,
}

/// Result of a Hungarian matching run.
#[derive(Debug, Clone)]
pub(crate) struct Matching {
    /// `perm[j]` is the row matched to column `j`. `usize::MAX`
    /// sentinel for unmatched columns (only populated in the
    /// partial-matching case).
    pub perm: Vec<usize>,
    /// Dual variable `u[i]` for row `i` (length `n`).
    pub u: Vec<f64>,
    /// Dual variable `v[j]` for column `j` (length `n`).
    pub v: Vec<f64>,
    /// Number of columns successfully matched. `n_matched == n`
    /// means a full perfect matching was found; a smaller value
    /// indicates structural singularity on the cost graph.
    pub n_matched: usize,
}

/// Sentinel for "unmatched" in the working arrays.
const NONE: usize = usize::MAX;

/// Opt-in work counters for the Hungarian kernel, used by the
/// deterministic scaling regression guard (issue #80). These count
/// *algorithmic work*, not wall-clock, so the guard is immune to CI
/// timing noise.
///
/// - `heap_init_slots`: total `pos[]` entries zeroed across the run -
///   `m` for the single `IndexHeap::new` plus `|touched|` for each
///   per-search `reset`. With the #80 fix this is `m + touched_total`
///   (linear in `n + nnz`). If the heap were reallocated per
///   unmatched column (the pre-#80 bug), every search would route a
///   fresh `new(m)` through this counter, making it `≈ searches·m`
///   (quadratic on near-tree KKTs). The structural invariant
///   `heap_init_slots == m + touched_total` is what the guard checks.
/// - `augment_searches`: number of shortest-augmenting-path searches
///   run in the main loop (one per still-unmatched column).
/// - `touched_total`: sum of `|touched|` over all searches.
/// - `phase3_inner_iters`: iterations of the length-2 augmentation
///   inner loop in `hungarian_init_heuristic`. O(nnz²) blow-up there
///   shows up as super-linear growth of this counter vs nnz.
/// - `main_loop_edge_scans`: total edges examined in the main
///   shortest-path loop (root-column scan + each popped row's matched
///   column). A near-dense coupling column makes each scan touching it
///   O(degree), so on a dense-column matrix this counter - not the
///   heap work - is the dominant cost. Counted once per column-scan
///   (by column length), so the kernel overhead is O(1) per scan.
// Work counters threaded through the live matching. Their only reader is the
// `#80` super-linear-regression guard test (`mc64_hungarian_no_quadratic_heap_
// realloc_regression`), so outside `cfg(test)` the fields are write-only - a
// narrow, documented allow rather than deleting a real performance guard.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct HungarianStats {
    pub heap_init_slots: u64,
    pub augment_searches: u64,
    pub touched_total: u64,
    pub phase3_inner_iters: u64,
    pub main_loop_edge_scans: u64,
}

/// A large finite value used as "+∞" for shortest-path distances.
/// We avoid `f64::INFINITY` in arithmetic to match SPRAL's `RINF`
/// style (it uses `huge(1.0_wp)/10`, a large finite constant). Any
/// value strictly larger than the maximum finite cost in the graph
/// suffices; `f64::MAX / 2.0` leaves plenty of headroom and never
/// overflows under the `dnew = vj + cost - u[i]` arithmetic.
const RINF: f64 = f64::MAX / 2.0;

/// Index-based binary min-heap keyed on an external `d` array,
/// mirroring SPRAL's `q` / `l` / `heap_update` / `heap_delete` /
/// `heap_pop`. The heap stores row indices `i`; the key for row `i`
/// is `d[i]`. `pos[i]` is the 1-based position of `i` in `heap`, or
/// `0` when `i` is not currently in the heap.
///
/// Decrease-key is supported via `update`, which assumes `d[i]` has
/// just been lowered and sifts `i` up. Arbitrary-position deletion
/// is supported via `delete`, which sifts either up or down as
/// needed. These together are what the shortest-path search needs.
///
/// 1-based indexing (SPRAL style) is preserved internally to make
/// parent/child arithmetic trivial (`parent = pos / 2`); externally
/// the type exposes 0-based row indices.
struct IndexHeap {
    /// `heap[1..=len]` contains row indices. `heap[0]` is a dummy.
    heap: Vec<usize>,
    /// `pos[i]` = 1-based position of `i` in `heap`, or `0` if not
    /// present. The SPRAL `l(i)` array.
    pos: Vec<usize>,
    /// Number of live entries in the heap.
    len: usize,
}

impl IndexHeap {
    fn new(m: usize, stats: &mut HungarianStats) -> Self {
        // Zeroing the `pos` array is O(m); count it so that a revert to
        // per-search allocation (issue #80) is observable as quadratic
        // `heap_init_slots` growth regardless of where `new` is called.
        stats.heap_init_slots += m as u64;
        IndexHeap {
            heap: vec![0; m + 1],
            pos: vec![0; m],
            len: 0,
        }
    }

    /// Return the heap to the empty state for reuse across augmenting
    /// searches without reallocating. `rows` must list every index whose
    /// `pos` could be nonzero - in `hungarian_match` that is exactly the
    /// `touched` set, since an index is only ever heap-inserted right
    /// after being pushed to `touched`. The `heap` backing array is left
    /// with stale entries; `len = 0` makes them unreachable. This turns
    /// the per-iteration `IndexHeap::new(m)` (O(m) alloc+zero per
    /// unmatched column, i.e. O(n·m) overall) into O(|touched|).
    fn reset(&mut self, rows: &[usize], stats: &mut HungarianStats) {
        stats.heap_init_slots += rows.len() as u64;
        for &i in rows {
            self.pos[i] = 0;
        }
        self.len = 0;
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn peek(&self) -> usize {
        self.heap[1]
    }

    fn contains(&self, i: usize) -> bool {
        self.pos[i] != 0
    }

    /// Called when `d[i]` has just been decreased (or `i` was just
    /// inserted with `pos[i] == len`). Sifts `i` upward in the heap.
    fn update(&mut self, i: usize, d: &[f64]) {
        let mut p = self.pos[i];
        if p <= 1 {
            self.heap[p] = i;
            return;
        }
        let v = d[i];
        while p > 1 {
            let parent_pos = p / 2;
            let parent_idx = self.heap[parent_pos];
            if v >= d[parent_idx] {
                break;
            }
            self.heap[p] = parent_idx;
            self.pos[parent_idx] = p;
            p = parent_pos;
        }
        self.heap[p] = i;
        self.pos[i] = p;
    }

    /// Insert row `i` (assumes `!contains(i)` and `d[i]` is set).
    fn insert(&mut self, i: usize, d: &[f64]) {
        self.len += 1;
        self.pos[i] = self.len;
        self.update(i, d);
    }

    /// Delete the entry at 1-based heap position `pos0`. Mirrors
    /// SPRAL's `heap_delete`.
    fn delete(&mut self, pos0: usize, d: &[f64]) {
        let removed = self.heap[pos0];
        self.pos[removed] = 0;
        if self.len == pos0 {
            self.len -= 1;
            return;
        }
        let idx = self.heap[self.len];
        let v = d[idx];
        self.len -= 1;
        let mut p = pos0;

        // Sift up.
        if p > 1 {
            loop {
                let parent = p / 2;
                let pk = self.heap[parent];
                if v >= d[pk] {
                    break;
                }
                self.heap[p] = pk;
                self.pos[pk] = p;
                p = parent;
                if p <= 1 {
                    break;
                }
            }
        }
        self.heap[p] = idx;
        self.pos[idx] = p;
        if p != pos0 {
            return;
        }

        // Otherwise sift down.
        loop {
            let mut child = 2 * p;
            if child > self.len {
                break;
            }
            let mut dk = d[self.heap[child]];
            if child < self.len {
                let dr = d[self.heap[child + 1]];
                if dk > dr {
                    child += 1;
                    dk = dr;
                }
            }
            if v <= dk {
                break;
            }
            let qk = self.heap[child];
            self.heap[p] = qk;
            self.pos[qk] = p;
            p = child;
        }
        self.heap[p] = idx;
        self.pos[idx] = p;
    }

    /// Pop and return the minimum-key row. Mirrors SPRAL's `heap_pop`.
    fn pop(&mut self, d: &[f64]) -> usize {
        let top = self.heap[1];
        self.delete(1, d);
        top
    }
}

/// Greedy initialization, mirroring SPRAL's `hungarian_init_heurisitic`
/// (`ref/spral/src/scaling.f90:810-929`). Two passes:
///
/// 1. Row-minimum pass: set `u[i]` to the smallest cost among edges
///    incident on row `i`, remembering the column `j` and CSC index
///    `k` that achieved it.
/// 2. Row-claim pass: for each row, if its candidate column is not
///    yet claimed and is not dense (more than `m/10` entries when
///    `m > 50`), match on that edge.
/// 3. Improve-assignment pass: for each still-unmatched column,
///    scan it for the smallest reduced cost `val - u[i]` and either
///    match directly or try a length-2 augmentation via another
///    already-matched column.
///
/// On exit, `iperm[i]` is the column matched to row `i` (or `NONE`),
/// `jperm[j]` is the CSC index of the matched entry for column `j`
/// (or `NONE`), and `u[i]` is set to `RINF` for empty rows (rows
/// with no incident edges), finite otherwise.
fn hungarian_init_heuristic(
    cost: &CostGraph,
    iperm: &mut [usize],
    jperm: &mut [usize],
    u: &mut [f64],
    stats: &mut HungarianStats,
) -> usize {
    let n = cost.n;
    let m = n;
    let mut num = 0usize;

    // Scratch used across phases. `l[i]` holds the CSC index `k` of
    // the row-minimum edge for row `i` during phase 1 (SPRAL's `l`).
    let mut l_row: Vec<usize> = vec![NONE; m];
    // `d_col[j]` is the improvement value `d(j)` in SPRAL phase 3.
    let mut d_col: Vec<f64> = vec![0.0; n];
    // `search_from[j]` is the CSC scan position reached in column
    // `j` during phase 3's length-2 augmentation attempts.
    let mut search_from: Vec<usize> = (0..n).map(|j| cost.col_ptr[j]).collect();

    // Phase 1: record smallest entry in each row.
    for ui in u.iter_mut().take(m) {
        *ui = RINF;
    }
    for j in 0..n {
        for k in cost.col_ptr[j]..cost.col_ptr[j + 1] {
            let i = cost.row_idx[k];
            if cost.cost[k] <= u[i] {
                u[i] = cost.cost[k];
                iperm[i] = j;
                l_row[i] = k;
            }
        }
    }

    // Phase 2: claim row-minimum edges where possible, skipping
    // dense columns (SPRAL's "don't assign on dense cols" guard).
    for i in 0..m {
        let j = iperm[i];
        if j == NONE {
            continue;
        }
        iperm[i] = NONE;
        if jperm[j] != NONE {
            continue;
        }
        let col_len = cost.col_ptr[j + 1] - cost.col_ptr[j];
        if col_len > m / 10 && m > 50 {
            continue;
        }
        num += 1;
        iperm[i] = j;
        jperm[j] = l_row[i];
    }
    if num == n {
        return num;
    }

    // Phase 3: scan unmatched columns for improvement augmentations.
    'improve_assign: for j in 0..n {
        if jperm[j] != NONE {
            continue;
        }
        if cost.col_ptr[j] >= cost.col_ptr[j + 1] {
            continue; // empty column
        }
        // Find smallest `val(k) - u[i]` in column j, with tie-break
        // preferring a first-unmatched row.
        let start = cost.col_ptr[j];
        let end = cost.col_ptr[j + 1];
        let mut i0 = cost.row_idx[start];
        let mut vj = cost.cost[start] - u[i0];
        let mut k0 = start;
        for k in (start + 1)..end {
            let i = cost.row_idx[k];
            let di = cost.cost[k] - u[i];
            if di > vj {
                continue;
            }
            if di == vj && di != RINF {
                // Tie-break: prefer first unmatched row.
                if iperm[i] != NONE || iperm[i0] == NONE {
                    continue;
                }
            }
            vj = di;
            i0 = i;
            k0 = k;
        }
        d_col[j] = vj;
        // If `i0` is unmatched, match it directly.
        if iperm[i0] == NONE {
            num += 1;
            jperm[j] = k0;
            iperm[i0] = j;
            search_from[j] = k0 + 1;
            continue;
        }
        // Otherwise attempt a length-2 augmentation.
        for k in k0..end {
            let i = cost.row_idx[k];
            if (cost.cost[k] - u[i]) > vj {
                continue;
            }
            let jj = iperm[i];
            if jj == NONE {
                continue;
            }
            let jj_end = cost.col_ptr[jj + 1];
            for kk in search_from[jj]..jj_end {
                stats.phase3_inner_iters += 1;
                let ii = cost.row_idx[kk];
                if iperm[ii] != NONE {
                    continue;
                }
                if (cost.cost[kk] - u[ii]) <= d_col[jj] {
                    jperm[jj] = kk;
                    iperm[ii] = jj;
                    search_from[jj] = kk + 1;
                    num += 1;
                    jperm[j] = k;
                    iperm[i] = j;
                    search_from[j] = k + 1;
                    continue 'improve_assign;
                }
            }
            search_from[jj] = jj_end;
        }
    }

    num
}

/// Solve the minimum-cost perfect bipartite matching problem via
/// the shortest-augmenting-path Hungarian algorithm.
///
/// At termination the dual variables satisfy
/// `u[i] + v[j] <= cost[i][j]` for every edge, with equality on
/// matched edges (the LP complementary-slackness conditions).
///
/// Algorithm: Duff & Koster 2001 §4, mirroring SPRAL's
/// `hungarian_match` at `ref/spral/src/scaling.f90:938-1171`.
///
/// Structure of the working state (following SPRAL line-by-line):
/// - `iperm[i]` = column matched to row `i`, or `NONE`.
/// - `jperm[j]` = CSC index `k` of the matched entry for column `j`,
///   or `NONE`. The matched row is `row_idx[jperm[j]]`.
/// - `u[i]`, `v[j]` = dual variables.
/// - `d[i]` = current shortest-path distance from the current root
///   column to row `i`, initialized to `RINF` at the start of each
///   augmenting search.
/// - `pr[j]` = parent column on the augmenting tree toward the root.
/// - `out[j]` = CSC index of the edge by which column `j` was
///   first reached (so `row_idx[out[j]]` is the row along the tree).
/// - `visited[i] = true` means row `i` has been extracted from the
///   shortest-path search and its `d[i]` is final (SPRAL's `up`
///   region of `q`, which here is represented by a boolean plus the
///   accumulated `visited_rows` list for the post-update sweep).
/// - `heap` is the open-set min-heap keyed on `d[i]`.
pub(crate) fn hungarian_match(cost: &CostGraph) -> Matching {
    hungarian_match_instrumented(cost).0
}

/// Same as [`hungarian_match`] but also returns [`HungarianStats`]
/// work counters. The work-counting `+=` calls are off the Dijkstra
/// inner loop (they live in `IndexHeap::new`/`reset` and the phase-3
/// augmentation loop), so this carries negligible overhead and
/// produces bit-identical matchings. Used by the #80 regression guard.
pub(crate) fn hungarian_match_instrumented(cost: &CostGraph) -> (Matching, HungarianStats) {
    let n = cost.n;
    let m = n; // square cost graph
    let mut stats = HungarianStats::default();

    let mut iperm: Vec<usize> = vec![NONE; m];
    let mut jperm: Vec<usize> = vec![NONE; n];
    let mut u: Vec<f64> = vec![0.0; m];
    let mut v: Vec<f64> = vec![0.0; n];

    if n == 0 {
        return (
            Matching {
                perm: Vec::new(),
                u,
                v,
                n_matched: 0,
            },
            stats,
        );
    }

    // Greedy initialization.
    let mut num = hungarian_init_heuristic(cost, &mut iperm, &mut jperm, &mut u, &mut stats);

    // Clamp any `u[i] == RINF` (set for empty rows) to 0 so that
    // the reduced-cost arithmetic below never produces `-RINF`.
    // Empty rows cannot participate in the matching anyway.
    for ui in u.iter_mut() {
        if *ui >= RINF {
            *ui = 0.0;
        }
    }

    if num == n {
        finalize_duals(cost, &iperm, &jperm, &u, &mut v);
        return (build_matching(cost, iperm, jperm, u, v, num), stats);
    }

    // Main loop: for each unmatched column, run a shortest-path
    // search rooted at that column and augment.
    let mut d: Vec<f64> = vec![RINF; m];
    let mut pr: Vec<usize> = vec![NONE; n];
    let mut out_idx: Vec<usize> = vec![NONE; n];
    // `visited[i]` = true iff row `i` has been finalized during the
    // current search (moved to SPRAL's "q(up:m)" region). These are
    // the rows whose duals get updated when the augmenting path is
    // found.
    let mut visited: Vec<bool> = vec![false; m];
    // Rows touched by this search, so the end-of-iteration reset
    // can clear only what was changed rather than re-allocating.
    let mut touched: Vec<usize> = Vec::with_capacity(m);
    let mut visited_rows: Vec<usize> = Vec::with_capacity(m);
    // Allocated once and reused across all augmenting searches; reset
    // incrementally (over `touched`) at the end of each iteration, the
    // same way `d` and `visited` are. Previously this was reallocated
    // per unmatched column - O(n·m) alloc+zeroing that dominated MC64 on
    // large near-tree KKTs (issue #80).
    let mut heap = IndexHeap::new(m, &mut stats);

    for jord in 0..n {
        if jperm[jord] != NONE {
            continue;
        }
        stats.augment_searches += 1;

        // Per-iteration working state.
        let mut csp = RINF; // cost of the shortest augmenting path
        let mut isp: usize = NONE; // CSC index of the terminal edge
        let mut jsp: usize = NONE; // column that owns that edge
        visited_rows.clear();
        touched.clear();

        // Build the shortest-path tree from column `jord`.
        let j = jord;
        pr[j] = NONE;

        // Scan the root column: each entry is either a direct
        // augmenting-path terminator (if the row is unmatched) or
        // a seed for the open set (if the row is matched).
        stats.main_loop_edge_scans += (cost.col_ptr[j + 1] - cost.col_ptr[j]) as u64;
        for k in cost.col_ptr[j]..cost.col_ptr[j + 1] {
            let i = cost.row_idx[k];
            let dnew = cost.cost[k] - u[i];
            if dnew >= csp {
                continue;
            }
            if iperm[i] == NONE {
                csp = dnew;
                isp = k;
                jsp = j;
            } else if dnew < d[i] {
                if d[i] == RINF {
                    touched.push(i);
                }
                d[i] = dnew;
                let jj = iperm[i];
                out_idx[jj] = k;
                pr[jj] = j;
                if heap.contains(i) {
                    heap.update(i, &d);
                } else {
                    heap.insert(i, &d);
                }
            }
        }

        // Main shortest-path loop.
        loop {
            if heap.is_empty() {
                break;
            }
            let top = heap.peek();
            if d[top] >= csp {
                break;
            }
            let q0 = heap.pop(&d);
            visited[q0] = true;
            visited_rows.push(q0);
            let dq0 = d[q0];

            // Scan the column matched to row q0.
            let j2 = iperm[q0];
            // SPRAL formula: vj = dq0 - val(jperm(j)) + dualu(q0).
            // This is the accumulated reduced-cost offset to apply
            // to each outgoing edge.
            let vj = dq0 - cost.cost[jperm[j2]] + u[q0];
            stats.main_loop_edge_scans += (cost.col_ptr[j2 + 1] - cost.col_ptr[j2]) as u64;
            for k in cost.col_ptr[j2]..cost.col_ptr[j2 + 1] {
                let i = cost.row_idx[k];
                if visited[i] {
                    continue;
                }
                let dnew = vj + cost.cost[k] - u[i];
                if dnew >= csp {
                    continue;
                }
                if iperm[i] == NONE {
                    // Unmatched row: candidate terminator.
                    csp = dnew;
                    isp = k;
                    jsp = j2;
                } else {
                    let di = d[i];
                    if di <= dnew {
                        continue;
                    }
                    if d[i] == RINF {
                        touched.push(i);
                    }
                    d[i] = dnew;
                    if heap.contains(i) {
                        heap.update(i, &d);
                    } else {
                        heap.insert(i, &d);
                    }
                    let jj = iperm[i];
                    out_idx[jj] = k;
                    pr[jj] = j2;
                }
            }
        }

        if csp < RINF {
            // Flip the matching along the augmenting path.
            num += 1;
            let i_term = cost.row_idx[isp];
            iperm[i_term] = jsp;
            jperm[jsp] = isp;
            let mut j_cur = jsp;
            for _ in 0..num {
                let jj = pr[j_cur];
                if jj == NONE {
                    break;
                }
                let k = out_idx[j_cur];
                let i_tree = cost.row_idx[k];
                iperm[i_tree] = jj;
                jperm[jj] = k;
                j_cur = jj;
            }
            // Update dual variables for all finalized rows.
            for &i in &visited_rows {
                u[i] += d[i] - csp;
            }
        }

        // Reset per-iteration scratch for rows touched this round.
        for &i in &touched {
            d[i] = RINF;
        }
        for &i in &visited_rows {
            visited[i] = false;
        }
        // Return the heap to empty for the next search. Every heap member
        // is in `touched`, so clearing their `pos` (plus `len = 0`) is a
        // complete reset.
        stats.touched_total += touched.len() as u64;
        heap.reset(&touched, &mut stats);
    }

    finalize_duals(cost, &iperm, &jperm, &u, &mut v);
    (build_matching(cost, iperm, jperm, u, v, num), stats)
}

/// Compute column duals from the final matching. Mirrors
/// `scaling.f90:1158-1169`: for each matched column `j` with
/// `jperm[j] = k`, the complementary-slackness equality
/// `u[row(k)] + v[j] == cost[k]` forces
/// `v[j] = cost[k] - u[row(k)]`. Unmatched columns get `v[j] = 0`.
/// Rows that never participated in any match get `u[i] = 0`.
fn finalize_duals(cost: &CostGraph, iperm: &[usize], jperm: &[usize], u: &[f64], v: &mut [f64]) {
    for (j, vj) in v.iter_mut().enumerate() {
        if jperm[j] != NONE {
            let k = jperm[j];
            let i = cost.row_idx[k];
            *vj = cost.cost[k] - u[i];
        } else {
            *vj = 0.0;
        }
    }
    // Caller updates `u` in place; here we only need to zero out
    // rows that were never matched (defensive - the main loop does
    // not touch `u[i]` for such rows, but `hungarian_init_heuristic`
    // may have left a row-minimum value in `u[i]` from phase 1).
    let _ = iperm; // unused beyond its role in the main loop
}

/// Convert the working `(iperm, jperm, u, v, num)` state into the
/// public `Matching` return type. Also applies the "zero u for
/// unmatched rows" rule from `scaling.f90:1169`.
fn build_matching(
    cost: &CostGraph,
    iperm: Vec<usize>,
    jperm: Vec<usize>,
    mut u: Vec<f64>,
    v: Vec<f64>,
    num: usize,
) -> Matching {
    let n = cost.n;
    let mut perm = vec![NONE; n];
    for (j, &k) in jperm.iter().enumerate() {
        if k != NONE {
            perm[j] = cost.row_idx[k];
        }
    }
    for (i, &col) in iperm.iter().enumerate() {
        if col == NONE {
            u[i] = 0.0;
        }
    }
    Matching {
        perm,
        u,
        v,
        n_matched: num,
    }
}

#[cfg(test)]
mod tests {
    //! Hungarian kernel unit tests.
    //!
    //! These tests exercise the `hungarian_match` function directly
    //! on small cost graphs where the answer can be hand-derived.
    //! Pre-Step 3 (the stub), tests that assert on identity-like
    //! behavior pass; tests that assert on non-trivial matchings
    //! or non-zero duals fail. This is intentional - the test file
    //! is the red→green gate for Step 3.
    //!
    //! Hand-derivation method: any minimum-cost perfect matching on
    //! a bipartite graph satisfies the LP optimality conditions
    //!   `u[i] + v[j] ≤ cost[i][j]`   for all edges,
    //!   `u[i] + v[j] == cost[i][j]`  on matched edges,
    //! so the matching plus any feasible dual that makes the total
    //! `sum(u) + sum(v)` equal to `sum(cost[matched])` is optimal.

    use super::*;

    /// Build a `CostGraph` from dense (row, col, cost) triples.
    /// Only used in tests - converts a small list of entries into
    /// the CSC format the Hungarian kernel expects.
    fn build_cost_graph(n: usize, entries: &[(usize, usize, f64)]) -> CostGraph {
        let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for &(r, c, v) in entries {
            by_col[c].push((r, v));
        }
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_idx = Vec::new();
        let mut cost = Vec::new();
        for j in 0..n {
            by_col[j].sort_by_key(|&(r, _)| r);
            for &(r, v) in &by_col[j] {
                row_idx.push(r);
                cost.push(v);
            }
            col_ptr[j + 1] = row_idx.len();
        }
        CostGraph {
            n,
            col_ptr,
            row_idx,
            cost,
        }
    }

    /// Verify the LP optimality conditions for a `Matching` on a
    /// `CostGraph`: for every edge `u[i] + v[j] ≤ cost[i][j]` with
    /// equality on matched edges.
    fn assert_matching_optimal(cost: &CostGraph, m: &Matching) {
        let n = cost.n;
        assert_eq!(m.u.len(), n);
        assert_eq!(m.v.len(), n);
        assert_eq!(m.perm.len(), n);

        let mut matched_row = vec![false; n];
        for j in 0..n {
            if m.perm[j] != usize::MAX {
                matched_row[m.perm[j]] = true;
            }
        }

        for j in 0..n {
            for k in cost.col_ptr[j]..cost.col_ptr[j + 1] {
                let i = cost.row_idx[k];
                let c = cost.cost[k];
                let reduced = m.u[i] + m.v[j];
                assert!(
                    reduced <= c + 1e-10,
                    "edge ({},{}) has cost {} but u+v={} (reduced > cost)",
                    i,
                    j,
                    c,
                    reduced
                );
                if m.perm[j] == i {
                    assert!(
                        (reduced - c).abs() < 1e-10,
                        "matched edge ({},{}) has cost {} but u+v={} (not tight)",
                        i,
                        j,
                        c,
                        reduced
                    );
                }
            }
        }
    }

    /// 3×3 identity pattern: matching is trivially identity with
    /// zero duals. The stub passes this because "identity matching
    /// with zero duals" is exactly what it returns.
    #[test]
    fn match_diagonal_3x3_identity() {
        let cost = build_cost_graph(3, &[(0, 0, 0.0), (1, 1, 0.0), (2, 2, 0.0)]);
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 3);
        assert_eq!(m.perm, vec![0, 1, 2]);
        assert_matching_optimal(&cost, &m);
    }

    /// 3×3 with a non-identity permutation pattern:
    ///   cost(0, 1) = 0
    ///   cost(1, 2) = 0
    ///   cost(2, 0) = 0
    /// The only perfect matching is 0↔1, 1↔2, 2↔0 (i.e., col 0 is
    /// matched with row 2, etc.). The stub returns identity, which
    /// is NOT a valid matching on this sparsity pattern, so this
    /// test MUST fail on the stub.
    ///
    /// Step 3 has landed; the real Hungarian kernel handles this.
    #[test]
    fn match_permutation_3x3() {
        let cost = build_cost_graph(3, &[(1, 0, 0.0), (2, 1, 0.0), (0, 2, 0.0)]);
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 3);
        // perm[j] is the row matched to column j
        assert_eq!(m.perm[0], 1, "col 0 should match row 1");
        assert_eq!(m.perm[1], 2, "col 1 should match row 2");
        assert_eq!(m.perm[2], 0, "col 2 should match row 0");
        assert_matching_optimal(&cost, &m);
    }

    /// 3×3 with a non-trivial cost matrix where the answer requires
    /// actual Hungarian logic. The costs are:
    ///    col 0: row 0 -> 3, row 1 -> 1
    ///    col 1: row 0 -> 2, row 2 -> 4
    ///    col 2: row 1 -> 5, row 2 -> 0
    /// Minimum total cost is 1 + 2 + 0 = 3 via matching
    /// 0↔1, 1↔0, 2↔2 (col 0 ↔ row 1, col 1 ↔ row 0, col 2 ↔ row 2).
    /// Alternative matching 0↔0, 1↔2, 2↔1 has cost 3 + 4 + 5 = 12.
    /// Only the first is optimal. The stub returns identity
    /// `perm = [0, 1, 2]`, which on this cost graph would be
    /// 0↔0, 1↔1, 2↔2 - but (1,1) has no entry in our graph, so
    /// the stub's matching is not even feasible.
    ///
    /// Step 3 has landed; the real Hungarian kernel handles this.
    #[test]
    fn match_hand_computed_3x3() {
        let cost = build_cost_graph(
            3,
            &[
                (0, 0, 3.0),
                (1, 0, 1.0),
                (0, 1, 2.0),
                (2, 1, 4.0),
                (1, 2, 5.0),
                (2, 2, 0.0),
            ],
        );
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 3);
        assert_eq!(m.perm[0], 1, "col 0 matches row 1 (cost 1)");
        assert_eq!(m.perm[1], 0, "col 1 matches row 0 (cost 2)");
        assert_eq!(m.perm[2], 2, "col 2 matches row 2 (cost 0)");
        assert_matching_optimal(&cost, &m);
        let total: f64 = (0..3).map(|j| m.u[m.perm[j]] + m.v[j]).sum();
        assert!(
            (total - 3.0).abs() < 1e-10,
            "total matching cost should be 3 (1+2+0), got {}",
            total
        );
    }

    /// 4×4 dense cost matrix. The optimal matching minimizes
    /// `sum cost[perm[j]][j]`. For
    ///
    ///        j=0 j=1 j=2 j=3
    ///   i=0   1   2   3   4
    ///   i=1   2   4   6   8
    ///   i=2   3   6   1   2
    ///   i=3   4   8   2   1
    ///
    /// Brute-force enumeration of all 24 permutations confirms the
    /// minimum is 6, achieved by matching col0->row1(2), col1->row0(2),
    /// col2->row2(1), col3->row3(1). The symmetric alternative
    /// col0->row0(1), col1->row1(4), col2->row2(1), col3->row3(1)
    /// totals 7 and is not optimal.
    #[test]
    fn match_dense_4x4() {
        let n = 4;
        let mat = [
            [1.0_f64, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 6.0, 1.0, 2.0],
            [4.0, 8.0, 2.0, 1.0],
        ];
        let mut entries = Vec::new();
        for (i, row) in mat.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                entries.push((i, j, v));
            }
        }
        let cost = build_cost_graph(n, &entries);
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, n);
        assert_matching_optimal(&cost, &m);
        let total: f64 = (0..n).map(|j| mat[m.perm[j]][j]).sum();
        assert!(
            (total - 6.0).abs() < 1e-10,
            "total matching cost should be 6, got {} with perm {:?}",
            total,
            m.perm
        );
    }

    /// 5×5 sparse pattern with non-trivial connectivity. The pattern
    /// forces most columns to use specific rows because alternatives
    /// are absent. Exercises the shortest-path search through a
    /// chain of augmentations (no one-shot greedy init suffices).
    #[test]
    fn match_sparse_5x5() {
        // Column 0: rows 0,1 with costs (10, 1)
        // Column 1: rows 0,2 with costs (1, 10)
        // Column 2: rows 1,3 with costs (10, 1)
        // Column 3: rows 2,4 with costs (1, 10)
        // Column 4: rows 3,4 with costs (10, 1)
        // Optimal matching: col0->row1(1), col1->row0(1), col2->row3(1),
        // col3->row2(1), col4->row4(1). Total = 5.
        let cost = build_cost_graph(
            5,
            &[
                (0, 0, 10.0),
                (1, 0, 1.0),
                (0, 1, 1.0),
                (2, 1, 10.0),
                (1, 2, 10.0),
                (3, 2, 1.0),
                (2, 3, 1.0),
                (4, 3, 10.0),
                (3, 4, 10.0),
                (4, 4, 1.0),
            ],
        );
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 5);
        assert_matching_optimal(&cost, &m);
        // Sum of matched costs should be the optimum (5.0).
        let mut total = 0.0;
        for j in 0..5 {
            for k in cost.col_ptr[j]..cost.col_ptr[j + 1] {
                if cost.row_idx[k] == m.perm[j] {
                    total += cost.cost[k];
                }
            }
        }
        assert!(
            (total - 5.0).abs() < 1e-10,
            "total matching cost should be 5, got {} with perm {:?}",
            total,
            m.perm
        );
    }

    /// Deterministic LCG (Numerical Recipes constants) for building
    /// reproducible synthetic cost graphs without an RNG dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn below(&mut self, bound: usize) -> usize {
            (self.next_u64() >> 33) as usize % bound
        }
    }

    /// Build an n×n cost graph with `deg` distinct random rows per
    /// column and random positive costs. With small constant `deg`,
    /// the greedy init leaves a constant fraction of columns unmatched,
    /// so the main augmenting loop runs Θ(n) shortest-path searches -
    /// exactly the near-tree regime where the issue #80 per-column heap
    /// reallocation was O(n·m). nnz = deg·n grows linearly in n.
    fn gen_random_sparse(n: usize, deg: usize, seed: u64) -> CostGraph {
        let mut rng = Lcg(seed);
        let mut by_col: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for col in by_col.iter_mut() {
            let mut rows: Vec<usize> = Vec::with_capacity(deg);
            while rows.len() < deg {
                let r = rng.below(n);
                if !rows.contains(&r) {
                    rows.push(r);
                }
            }
            for &r in &rows {
                let c = 1.0 + (rng.below(100) as f64);
                col.push((r, c));
            }
        }
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_idx = Vec::new();
        let mut cost = Vec::new();
        for j in 0..n {
            by_col[j].sort_by_key(|&(r, _)| r);
            for &(r, v) in &by_col[j] {
                row_idx.push(r);
                cost.push(v);
            }
            col_ptr[j + 1] = row_idx.len();
        }
        CostGraph {
            n,
            col_ptr,
            row_idx,
            cost,
        }
    }

    /// Deterministic scaling regression guard for MC64 (issue #80).
    ///
    /// Two independent O(n²) hazards in the Hungarian kernel are
    /// pinned here without any reliance on wall-clock (CI-noise-immune):
    ///
    /// 1. **Per-column heap reallocation (the #80 bug).** Before the
    ///    fix, `IndexHeap::new(m)` was called inside the per-unmatched-
    ///    column loop, costing O(m) zeroing per search → O(searches·m)
    ///    = O(n²) on near-tree KKTs. The fix allocates once and resets
    ///    incrementally over `touched`. The exact structural invariant
    ///    of the fix is `heap_init_slots == n + touched_total` (one
    ///    O(n) allocation plus Σ|touched| reset work). A revert that
    ///    re-allocates per search routes O(m) per search through the
    ///    same counter, breaking the equality at every size. This is a
    ///    stronger, threshold-free guard than a growth ratio - and
    ///    necessary, because on a hard matching the *legitimate*
    ///    `touched_total` is itself super-linear (long augmenting
    ///    paths), so total heap work is not linear to begin with.
    ///
    /// 2. **Length-2 augmentation blow-up.** The phase-3 inner loop in
    ///    `hungarian_init_heuristic` is O(nnz²) in the worst case. On
    ///    this random-sparse family it is observed linear in nnz
    ///    (~8.2× over an 8× size increase), so we assert the count
    ///    grows sub-quadratically (8× ladder, quadratic ≈ 64×).
    ///
    /// The family is `gen_random_sparse(n, deg=3)`: with small constant
    /// degree, greedy init leaves a constant fraction of columns
    /// unmatched, so the main augmenting loop runs (searches > 0) - the
    /// exact regime where the #80 realloc was the dominant cost.
    #[test]
    fn mc64_hungarian_no_quadratic_heap_realloc_regression() {
        let sizes = [1000usize, 2000, 4000, 8000];
        let mut phase3_first = 0u64;
        let mut phase3_last = 0u64;
        for (idx, &n) in sizes.iter().enumerate() {
            let cost = gen_random_sparse(n, 3, 0x1234_5678);
            let (m, stats) = hungarian_match_instrumented(&cost);

            // The main loop must actually run, or the heap-lifecycle
            // invariant below would hold vacuously.
            assert!(
                stats.augment_searches > 0,
                "n={n}: greedy matched everything; guard not exercised"
            );

            // #80 structural invariant: heap allocated exactly once
            // (O(n)) plus incremental resets totalling touched_total.
            // Re-introducing per-column `IndexHeap::new(m)` breaks this.
            assert_eq!(
                stats.heap_init_slots,
                n as u64 + stats.touched_total,
                "n={n}: heap-init work {} != n + touched_total {} \
                 (issue #80 per-column heap reallocation reintroduced?)",
                stats.heap_init_slots,
                n as u64 + stats.touched_total,
            );

            // Behavior preservation: instrumentation must not change
            // the matching; it must still be LP-optimal.
            assert_matching_optimal(&cost, &m);

            if idx == 0 {
                phase3_first = stats.phase3_inner_iters;
            }
            if idx == sizes.len() - 1 {
                phase3_last = stats.phase3_inner_iters;
            }
        }

        // Phase-3 length-2 augmentation must stay sub-quadratic across
        // the 8× size ladder (linear ≈ 8×, quadratic ≈ 64×).
        assert!(phase3_first > 0, "phase-3 augmentation never exercised");
        assert!(
            phase3_last < 16 * phase3_first,
            "phase-3 inner iterations grew {}->{} ( >16x over 8x size ) \
             - possible O(nnz^2) augmentation regression",
            phase3_first,
            phase3_last,
        );
    }

    /// Structurally singular 3×3: only two distinct rows appear in
    /// the pattern (row 0 appears in two columns, but row 2 never
    /// appears at all), so at most two columns can be matched. The
    /// algorithm should report `n_matched < n` and leave the
    /// unmatchable column with `perm[j] == usize::MAX`, without
    /// panicking or producing infeasible duals.
    #[test]
    fn match_structurally_singular_3x3() {
        // col 0: row 0, row 1
        // col 1: row 0, row 1
        // col 2: row 0          (only row 0; singular)
        let cost = build_cost_graph(
            3,
            &[
                (0, 0, 1.0),
                (1, 0, 2.0),
                (0, 1, 3.0),
                (1, 1, 4.0),
                (0, 2, 5.0),
            ],
        );
        let m = hungarian_match(&cost);
        assert_eq!(m.n_matched, 2, "only 2 of 3 columns should match");
        // Exactly one column is unmatched.
        let n_unmatched = m.perm.iter().filter(|&&p| p == usize::MAX).count();
        assert_eq!(n_unmatched, 1);
        assert_matching_optimal(&cost, &m);
    }
}
