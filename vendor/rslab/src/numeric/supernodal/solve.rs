//! Supernodal, tree-parallel triangular solves for the sparse LDL^T factor.
//!
//! A scalar CSC factor is the wrong layout for a solve: every entry costs a
//! value, a `usize` row index and a random access into the right-hand side,
//! on one core. The plan works on the factor in *supernodal panel form*
//! ([`PanelFactor`]): one dense column-major panel per supernode (unit-lower
//! diagonal block on top, the off-block rows below) with a single shared
//! list of `u32` row indices. The LDL^T drivers produce that form directly,
//! so the plan takes the panels as its storage without a copy; a CSC factor
//! (the LU path) is converted once. A forward or backward sweep then streams
//! contiguous panels, touches each row index once per panel instead of once
//! per entry, and runs the inner loops over contiguous memory.
//!
//! Parallelism comes from the supernodal elimination tree. The tree is cut
//! into a fixed number of independent leaf subtrees (`SolveSettings::leaf_subtrees`, not a
//! function of the thread count) plus the ancestors above the cut. In the
//! forward sweep the subtrees run in parallel; updates that leave a subtree
//! (into ancestor rows) go to a per-subtree accumulator and are reduced in
//! subtree order before the ancestors are processed sequentially, their
//! large panels row-parallel. In the backward sweep the ancestors go first
//! and the subtrees then run in parallel reading only final values. Every
//! sum is formed in the same order for every thread count, so the solution
//! is bit-identical from one thread to many.

// The kernels index several parallel arrays (panels, row lists, slots,
// right-hand sides) at offset positions; index loops read closer to the
// linear algebra than zipped iterators would.
#![allow(clippy::needless_range_loop)]

use rayon::prelude::*;

use crate::error::RslabError;
use crate::numeric::ldlt::LdltPivots;
use crate::numeric::supernodal::panel::{PanelFactor, PanelStorage};
use crate::scalar::{fmadd, Scalar};

const NONE: u32 = u32::MAX;

/// One leaf subtree of the cut: its supernodes in elimination order and the
/// ancestor columns its off-tree updates accumulate into.
pub(crate) struct Subtree {
    nodes: Vec<u32>,
    /// `y` indices of the accumulator slots (the columns of the ancestors
    /// above the cut, in chain order).
    path_cols: Vec<u32>,
}

/// The factor in solve layout plus the tree schedule.
pub(crate) struct SolvePlan<T> {
    n: usize,
    /// Supernode column ranges, `ns + 1`.
    sn_col: Vec<u32>,
    /// Off-block row ranges into `rows` / `ext_slot`, `ns + 1`.
    row_ptr: Vec<usize>,
    /// Off-block row indices per supernode, ascending.
    rows: Vec<u32>,
    /// Per off-block row: accumulator slot when the row lies outside the
    /// supernode's leaf subtree, `NONE` when it is written directly.
    ext_slot: Vec<u32>,
    /// The panels, one `(w + m) x w` column-major block per supernode with
    /// leading dimension `w + m` (the factor's own storage, see
    /// [`PanelFactor`]).
    val_ptr: Vec<usize>,
    vals: Vec<T>,
    /// A row list of the factor's size, the buffer the next refactorization
    /// emits its rows into (see [`take_storage`](Self::take_storage)).
    spare_rows: Vec<u32>,
    /// Reciprocal diagonal per column for a non-unit triangular factor (the
    /// `U^T` of an LU); empty for a unit-diagonal factor.
    diag_inv: Vec<T>,
    subtrees: Vec<Subtree>,
    /// The ancestors (supernodes above the cut) by depth from the roots;
    /// siblings of a level are independent.
    pub(crate) top_levels: Vec<Vec<u32>>,
    /// Accumulator path (`y` indices) of every ancestor, by `top_index`.
    top_paths: Vec<Vec<u32>>,
    /// Position of a supernode in `top` (`NONE` below the cut).
    top_index: Vec<u32>,
    /// Per ancestor level, the start of each node's accumulator in the
    /// level's block, in slots (`level.len() + 1`).
    top_acc_ptr: Vec<Vec<usize>>,
    /// The blocking of the sweeps.
    cfg: crate::SolveSettings,
}

/// Per-task scratch vectors, reused across nodes: the big nodes need
/// buffers of tens to hundreds of kilobytes, and allocating those per node
/// (page faults under the allocator lock) serialises the parallel phases.
pub(crate) struct Scratch<T> {
    t: Vec<T>,
    g: Vec<T>,
    v: Vec<T>,
    partial: Vec<T>,
    accv: Vec<T>,
    /// A leaf subtree's accumulator of its off-tree updates.
    acc: Vec<T>,
}

impl<T> Default for Scratch<T> {
    fn default() -> Self {
        Self {
            t: Vec::new(),
            g: Vec::new(),
            v: Vec::new(),
            partial: Vec::new(),
            accv: Vec::new(),
            acc: Vec::new(),
        }
    }
}

/// The sweeps' scratch across solves (part of [`crate::SolveWork`]): a
/// [`Scratch`] per leaf subtree and per node of the widest node-parallel
/// ancestor level, the apex nodes' scratch, and an ancestor level's
/// accumulator block. Sized by the first solve through a plan, reused
/// after, so a warm sweep allocates nothing.
pub(crate) struct PlanWork<T> {
    subtrees: Vec<Scratch<T>>,
    level: Vec<Scratch<T>>,
    apex: Scratch<T>,
    top_acc: Vec<T>,
    /// This sweep's decision: on the pool, or on the calling thread.
    parallel: bool,
}

impl<T> Default for PlanWork<T> {
    fn default() -> Self {
        Self {
            subtrees: Vec::new(),
            level: Vec::new(),
            apex: Scratch::default(),
            top_acc: Vec::new(),
            parallel: false,
        }
    }
}

impl<T> PlanWork<T> {
    /// At least a scratch per subtree and per node of the widest level of
    /// `plan`; never shrinks, so one work serves several plans.
    fn fit(&mut self, subtrees: usize, widest: usize) {
        if self.subtrees.len() < subtrees {
            self.subtrees.resize_with(subtrees, Scratch::default);
        }
        if self.level.len() < widest {
            self.level.resize_with(widest, Scratch::default);
        }
    }
}

/// Shared mutable view of a right-hand side for the subtree phases.
///
/// SAFETY contract: concurrent users write disjoint index sets (a subtree
/// writes only its own columns) and read only indices no other task writes
/// during the phase (its own columns, or ancestor columns that are final).
#[derive(Clone, Copy)]
struct Shared<T>(*mut T, usize);
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Sync> Sync for Shared<T> {}
impl<T> Shared<T> {
    #[inline]
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice(&self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0, self.1)
    }
}

/// The cut of the tree ([`SolvePlan::from_panels`]): split the heaviest
/// subtree until every leaf subtree holds at most `1 / leaf_subtrees` of the
/// work (or is a single supernode). `own(s)` is the work of supernode `s`,
/// its panel entries. Returns which supernodes lie above the cut and the
/// leaf subtree roots, ascending.
fn cut(
    parent: &[u32],
    children: &[Vec<u32>],
    leaf_subtrees: usize,
    own: impl Fn(usize) -> u64,
) -> (Vec<bool>, Vec<u32>) {
    let ns = parent.len();
    let mut work = vec![0u64; ns];
    for s in 0..ns {
        work[s] += own(s);
        if parent[s] != NONE {
            let w = work[s];
            work[parent[s] as usize] += w;
        }
    }
    let roots = || (0..ns).filter(|&s| parent[s] == NONE);
    let leaf_cap = roots().map(|s| work[s]).sum::<u64>() / leaf_subtrees.max(1) as u64;
    let mut heap: std::collections::BinaryHeap<(u64, std::cmp::Reverse<u32>)> = roots()
        .map(|s| (work[s], std::cmp::Reverse(s as u32)))
        .collect();
    let mut is_top = vec![false; ns];
    let mut leaf_roots: Vec<u32> = Vec::new();
    while let Some((wk, std::cmp::Reverse(s))) = heap.pop() {
        if wk <= leaf_cap || children[s as usize].is_empty() {
            leaf_roots.push(s);
        } else {
            is_top[s as usize] = true;
            for &c in &children[s as usize] {
                heap.push((work[c as usize], std::cmp::Reverse(c)));
            }
        }
    }
    leaf_roots.sort_unstable();
    (is_top, leaf_roots)
}

/// Columns of every supernode's strict ancestors, the length of its
/// accumulator path. Parents carry larger indices than their children.
fn columns_above(parent: &[u32], width: impl Fn(usize) -> usize) -> Vec<usize> {
    let mut above = vec![0usize; parent.len()];
    for s in (0..parent.len()).rev() {
        let p = parent[s];
        if p != NONE {
            above[s] = above[p as usize] + width(p as usize);
        }
    }
    above
}

/// The heap of a [`SolvePlan`] not built yet, for the memory plan: the
/// schedule around the panels, its build and a solve's work vectors.
pub(crate) struct LayoutSize {
    /// Bytes of the schedule (all but the values and the reciprocal diagonal).
    pub held: u64,
    /// Peak bytes of the build's temporaries.
    pub build: u64,
    /// Entries of `T` per right-hand side a solve allocates besides the
    /// right-hand side itself: the accumulators, the apex buffers and each
    /// thread's node scratch.
    pub solve_per_rhs: u64,
}

/// The [`LayoutSize`] of the plan [`SolvePlan::from_panels`] builds on the
/// supernode tree (`parent`, `usize::MAX` for a root) with each supernode's
/// width `w` and off-block row count `m`, solved on `threads` threads.
/// Mirrors `from_panels` on a tree-nested structure.
pub(crate) fn layout_size(
    parent: &[usize],
    w: &[usize],
    m: &[usize],
    cfg: &crate::SolveSettings,
    threads: usize,
) -> LayoutSize {
    use crate::memory::pushed;
    let ns = parent.len();
    let n: usize = w.iter().sum();
    let rows: usize = m.iter().sum();
    let parent: Vec<u32> = (0..ns)
        .map(|s| match parent[s] {
            p if p != usize::MAX && p < ns && p > s => p as u32,
            _ => NONE,
        })
        .collect();
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); ns];
    for s in 0..ns {
        if parent[s] != NONE {
            children[parent[s] as usize].push(s as u32);
        }
    }
    let (is_top, leaf_roots) = cut(&parent, &children, cfg.leaf_subtrees, |s| {
        (w[s] * (w[s] + m[s])) as u64
    });
    let above = columns_above(&parent, |s| w[s]);
    let mut size = vec![1usize; ns];
    for s in 0..ns {
        if parent[s] != NONE {
            size[parent[s] as usize] += size[s];
        }
    }
    let mut depth = vec![0usize; ns];
    let mut level_len: Vec<usize> = Vec::new();
    for s in (0..ns).rev().filter(|&s| is_top[s]) {
        let p = parent[s];
        depth[s] = if p == NONE { 0 } else { depth[p as usize] + 1 };
        if level_len.len() <= depth[s] {
            level_len.resize(depth[s] + 1, 0);
        }
        level_len[depth[s]] += 1;
    }
    let top: Vec<usize> = (0..ns).filter(|&s| is_top[s]).collect();
    // sn_col, row_ptr, val_ptr (pushed by the arena), rows, ext_slot, top_index.
    let fixed = 4 * (ns + 1) + 8 * (ns + 1) + 8 * pushed(ns + 1) + 8 * rows + 4 * ns;
    let subtrees = leaf_roots.len() * std::mem::size_of::<Subtree>()
        + leaf_roots
            .iter()
            .map(|&r| 4 * pushed(size[r as usize]) + 4 * above[r as usize])
            .sum::<usize>();
    let levels = level_len.len() * 24 + level_len.iter().map(|&l| 4 * pushed(l)).sum::<usize>();
    let paths = top.len() * 24 + top.iter().map(|&t| 4 * above[t]).sum::<usize>();
    // sn_of, the tree (parent, children), work, heap, flags and slots.
    let build = 4 * n + ns * (4 + 24 + 4 + 8 + 16 + 1 + 4 + 4 + 4);
    // Accumulators of the subtrees and of the widest ancestor level, the
    // apex node's extended vector and partial slabs, and per thread the
    // node scratch of a subtree sweep.
    let max_ld = (0..ns).map(|s| w[s] + m[s]).max().unwrap_or(0);
    let mut level_acc = vec![0usize; level_len.len()];
    for &t in &top {
        level_acc[depth[t]] += above[t];
    }
    let gmax = cfg.block.max(1).div_ceil(cfg.ancestor_chunk.max(1));
    let solve_per_rhs = leaf_roots.iter().map(|&r| above[r as usize]).sum::<usize>()
        + level_acc.iter().copied().max().unwrap_or(0)
        + (2 + gmax) * max_ld
        + threads.max(1) * 2 * max_ld;
    LayoutSize {
        held: (fixed + subtrees + levels + paths) as u64,
        build: build as u64,
        solve_per_rhs: solve_per_rhs as u64,
    }
}

impl<T: Scalar> SolvePlan<T> {
    /// Heap bytes held: the panels (the factor's values) and the schedule.
    pub(crate) fn heap_bytes(&self) -> u64 {
        use crate::memory::{nested_bytes, vec_bytes};
        vec_bytes(&self.sn_col)
            + vec_bytes(&self.row_ptr)
            + vec_bytes(&self.rows)
            + vec_bytes(&self.ext_slot)
            + vec_bytes(&self.val_ptr)
            + vec_bytes(&self.vals)
            + vec_bytes(&self.spare_rows)
            + vec_bytes(&self.diag_inv)
            + vec_bytes(&self.subtrees)
            + self
                .subtrees
                .iter()
                .map(|t| vec_bytes(&t.nodes) + vec_bytes(&t.path_cols))
                .sum::<u64>()
            + nested_bytes(&self.top_levels)
            + nested_bytes(&self.top_paths)
            + vec_bytes(&self.top_index)
            + nested_bytes(&self.top_acc_ptr)
    }

    /// Build the schedule over a factor in panel form, taking the panels as
    /// the plan's storage (no copy). `supernode_parent` is the supernode
    /// tree of the analysis (`usize::MAX` for a root); an empty or
    /// mismatched one is replaced by the parent implied by the first
    /// off-block row of every supernode.
    pub fn from_panels(
        factor: PanelFactor<T>,
        supernode_parent: &[usize],
        unit: bool,
        cfg: crate::SolveSettings,
    ) -> Self {
        let n = factor.n;
        let ns = factor.n_supernodes();
        let sn_col: Vec<u32> = factor.sn_col.clone();
        let mut sn_of = vec![0u32; n];
        for s in 0..ns {
            for c in sn_col[s] as usize..sn_col[s + 1] as usize {
                sn_of[c] = s as u32;
            }
        }
        let diag_inv: Vec<T> = if unit {
            Vec::new()
        } else {
            let mut d = Vec::with_capacity(n);
            for s in 0..ns {
                let (_, w, m) = factor.shape(s);
                let ld = w + m;
                for k in 0..w {
                    d.push(factor.panel(s)[k * ld + k].recip());
                }
            }
            d
        };
        let PanelFactor {
            rows,
            row_ptr,
            val_ptr,
            vals,
            ..
        } = factor;
        debug_assert!(
            (0..ns).all(|s| rows[row_ptr[s]..row_ptr[s + 1]]
                .windows(2)
                .all(|p| p[0] < p[1])),
            "off-block rows ascending"
        );
        let known = supernode_parent.len() == ns;

        let mut parent = vec![NONE; ns];
        let mut children: Vec<Vec<u32>> = vec![Vec::new(); ns];
        if known {
            for s in 0..ns {
                let p = supernode_parent[s];
                if p != usize::MAX && p < ns && p > s {
                    parent[s] = p as u32;
                    children[p].push(s as u32);
                }
            }
        } else {
            for s in 0..ns {
                if row_ptr[s + 1] > row_ptr[s] {
                    let p = sn_of[rows[row_ptr[s]] as usize];
                    parent[s] = p;
                    children[p as usize].push(s as u32);
                }
            }
        }
        let width = |s: usize| (sn_col[s + 1] - sn_col[s]) as usize;
        let (is_top, leaf_roots) = cut(&parent, &children, cfg.leaf_subtrees, |s| {
            let w = width(s) as u64;
            w * (w + (row_ptr[s + 1] - row_ptr[s]) as u64)
        });
        let above = columns_above(&parent, width);

        // Subtree membership and node lists (ascending = elimination order).
        let mut subtree_of = vec![NONE; ns];
        let mut subtrees: Vec<Subtree> = Vec::with_capacity(leaf_roots.len());
        for (t, &r) in leaf_roots.iter().enumerate() {
            let mut nodes = vec![r];
            let mut stack = vec![r];
            while let Some(s) = stack.pop() {
                for &c in &children[s as usize] {
                    nodes.push(c);
                    stack.push(c);
                }
            }
            nodes.sort_unstable();
            for &s in &nodes {
                subtree_of[s as usize] = t as u32;
            }
            let mut path_cols = Vec::with_capacity(above[r as usize]);
            let mut a = parent[r as usize];
            while a != NONE {
                path_cols.extend(sn_col[a as usize]..sn_col[a as usize + 1]);
                a = parent[a as usize];
            }
            subtrees.push(Subtree { nodes, path_cols });
        }
        let top: Vec<u32> = (0..ns as u32).filter(|&s| is_top[s as usize]).collect();
        // Ancestors by depth from the roots (level 0 = roots) and each
        // ancestor's accumulator path.
        let mut depth = vec![0u32; ns];
        for &t in top.iter().rev() {
            // Parents carry larger indices: descending order sets them first.
            let p = parent[t as usize];
            depth[t as usize] = if p == NONE { 0 } else { depth[p as usize] + 1 };
        }
        let max_depth = top.iter().map(|&t| depth[t as usize]).max().unwrap_or(0) as usize;
        let mut top_levels: Vec<Vec<u32>> =
            vec![Vec::new(); if top.is_empty() { 0 } else { max_depth + 1 }];
        for &t in &top {
            top_levels[depth[t as usize] as usize].push(t);
        }
        let top_paths: Vec<Vec<u32>> = top
            .iter()
            .map(|&t| {
                let mut path_cols = Vec::with_capacity(above[t as usize]);
                let mut a = parent[t as usize];
                while a != NONE {
                    path_cols.extend(sn_col[a as usize]..sn_col[a as usize + 1]);
                    a = parent[a as usize];
                }
                path_cols
            })
            .collect();

        // Accumulator slots of the off-tree rows. A row whose supernode is
        // not an ancestor of the writer (possible only for a pruned,
        // `drop_tol` factor, whose structure no longer follows an elimination
        // tree) has no slot; such a factor is solved without the subtree
        // phase (everything above the cut, sequential in the tree phases).
        let mut ext_slot = vec![NONE; rows.len()];
        let mut slot_base = vec![NONE; ns];
        let mut degenerate = false;
        for st in &subtrees {
            let Some(&root) = st.nodes.last() else {
                continue;
            };
            let mut a = parent[root as usize];
            let mut off = 0u32;
            while a != NONE {
                slot_base[a as usize] = off;
                off += sn_col[a as usize + 1] - sn_col[a as usize];
                a = parent[a as usize];
            }
            for &s in &st.nodes {
                for e in row_ptr[s as usize]..row_ptr[s as usize + 1] {
                    let r = rows[e];
                    let a = sn_of[r as usize];
                    if subtree_of[a as usize] != subtree_of[s as usize] {
                        if slot_base[a as usize] == NONE {
                            degenerate = true;
                        } else {
                            ext_slot[e] = slot_base[a as usize] + (r - sn_col[a as usize]);
                        }
                    }
                }
            }
            let mut a = parent[root as usize];
            while a != NONE {
                slot_base[a as usize] = NONE;
                a = parent[a as usize];
            }
        }

        // Ancestor rows: every off-block row lies in an ancestor above, so
        // it gets a slot on the node's own path.
        for &t in &top {
            let mut a = parent[t as usize];
            let mut off = 0u32;
            while a != NONE {
                slot_base[a as usize] = off;
                off += sn_col[a as usize + 1] - sn_col[a as usize];
                a = parent[a as usize];
            }
            for e in row_ptr[t as usize]..row_ptr[t as usize + 1] {
                let r = rows[e];
                let a = sn_of[r as usize];
                if slot_base[a as usize] == NONE {
                    degenerate = true;
                } else {
                    ext_slot[e] = slot_base[a as usize] + (r - sn_col[a as usize]);
                }
            }
            let mut a = parent[t as usize];
            while a != NONE {
                slot_base[a as usize] = NONE;
                a = parent[a as usize];
            }
        }
        let (subtrees, top, top_levels, top_paths, ext_slot) = if degenerate {
            // The structure does not follow an elimination tree (a pruned
            // factor, or an LU whose row pivoting moved rows across
            // subtrees): one sequential chain, every node an ancestor
            // without slots. Levels run from the root, so the deepest level
            // (processed first by the forward sweep) is the first column.
            if crate::logging::enabled(crate::logging::LogLevel::Debug) {
                crate::logging::debug("solve layout: structure not tree-nested, sequential sweeps");
            }
            let all: Vec<u32> = (0..ns as u32).collect();
            let levels: Vec<Vec<u32>> = all.iter().rev().map(|&s| vec![s]).collect();
            let paths = vec![Vec::new(); ns];
            (Vec::new(), all, levels, paths, vec![NONE; rows.len()])
        } else {
            (subtrees, top, top_levels, top_paths, ext_slot)
        };
        let top_index = {
            let mut ix = vec![NONE; ns];
            for (i, &t) in top.iter().enumerate() {
                ix[t as usize] = i as u32;
            }
            ix
        };
        // Per ancestor level, where each node's accumulator starts in the
        // level's block, in slots.
        let top_acc_ptr: Vec<Vec<usize>> = top_levels
            .iter()
            .map(|level| {
                std::iter::once(0)
                    .chain(level.iter().scan(0usize, |o, &s| {
                        *o += top_paths[top_index[s as usize] as usize].len();
                        Some(*o)
                    }))
                    .collect()
            })
            .collect();
        Self {
            n,
            sn_col,
            row_ptr,
            rows,
            ext_slot,
            val_ptr,
            vals,
            diag_inv,
            subtrees,
            top_levels,
            top_paths,
            top_index,
            top_acc_ptr,
            spare_rows: Vec::new(),
            cfg: crate::SolveSettings {
                block: cfg.block.max(1),
                ancestor_chunk: cfg.ancestor_chunk.max(1),
                ..cfg
            },
        }
    }

    /// The panels' buffer and a spare row list, taken out for the next
    /// factorization of the same analysis to fill; [`refill`](Self::refill)
    /// puts the result back. The plan cannot solve until then.
    pub(crate) fn take_storage(&mut self) -> PanelStorage<T> {
        PanelStorage {
            vals: std::mem::take(&mut self.vals),
            rows: std::mem::take(&mut self.spare_rows),
        }
    }

    /// Take `factor`, the next factorization of this plan's analysis, as
    /// [`from_panels`](Self::from_panels) would. When its supernodes and rows
    /// are the plan's and `cfg` is unchanged, only the panels and the
    /// reciprocal diagonal change and the schedule stays; a factorization
    /// whose pivoting or cancellation changed the rows gets a new plan.
    pub fn refill(
        &mut self,
        factor: PanelFactor<T>,
        supernode_parent: &[usize],
        unit: bool,
        cfg: crate::SolveSettings,
    ) {
        let cfg_now = crate::SolveSettings {
            block: cfg.block.max(1),
            ancestor_chunk: cfg.ancestor_chunk.max(1),
            ..cfg
        };
        let ns = factor.n_supernodes();
        let same = cfg_now == self.cfg
            && factor.n == self.n
            && unit == self.diag_inv.is_empty()
            && factor.sn_col == self.sn_col
            && factor.val_ptr == self.val_ptr
            && factor.row_ptr == self.row_ptr
            && factor.rows == self.rows;
        if !same {
            *self = Self::from_panels(factor, supernode_parent, unit, cfg);
            return;
        }
        // The new row list equals the plan's: kept as the next spare.
        let PanelFactor { vals, rows, .. } = factor;
        self.vals = vals;
        self.spare_rows = rows;
        if !unit {
            for s in 0..ns {
                let (c0, c1) = (self.sn_col[s] as usize, self.sn_col[s + 1] as usize);
                let ld = c1 - c0 + self.row_ptr[s + 1] - self.row_ptr[s];
                let panel = &self.vals[self.val_ptr[s]..];
                for k in 0..c1 - c0 {
                    self.diag_inv[c0 + k] = panel[k * ld + k].recip();
                }
            }
        }
    }

    /// Bytes of the panel storage (values plus row indices).
    pub fn bytes(&self) -> usize {
        self.vals.len() * std::mem::size_of::<T>() + self.rows.len() * 4
    }

    #[inline]
    fn node(&self, s: u32) -> (usize, usize, usize, usize, usize, &[T]) {
        let s = s as usize;
        let c0 = self.sn_col[s] as usize;
        let w = self.sn_col[s + 1] as usize - c0;
        let r0 = self.row_ptr[s];
        let m = self.row_ptr[s + 1] - r0;
        let panel = &self.vals[self.val_ptr[s]..self.val_ptr[s + 1]];
        (c0, w, r0, m, w + m, panel)
    }

    // -----------------------------------------------------------------------
    // Single right-hand side
    // -----------------------------------------------------------------------

    /// Forward sweep through one supernode: unit-lower block solve on
    /// `y[c0..c0+w]`, then the off-block update into `y` (direct rows) or
    /// `acc` (rows outside the subtree).
    fn fwd_node(&self, s: u32, y: &mut [T], acc: &mut [T], t: &mut Vec<T>) {
        let (c0, w, r0, m, ld, panel) = self.node(s);
        let unit = self.diag_inv.is_empty();
        for k in 0..w {
            if !unit {
                y[c0 + k] = y[c0 + k] * self.diag_inv[c0 + k];
            }
            let yk = y[c0 + k];
            let col = &panel[k * ld..k * ld + w];
            for i in k + 1..w {
                y[c0 + i] = y[c0 + i] - col[i] * yk;
            }
        }
        if m == 0 {
            return;
        }
        t.clear();
        t.resize(m, T::zero());
        for k in 0..w {
            let yk = y[c0 + k];
            let col = &panel[k * ld + w..(k + 1) * ld];
            for (ti, &l) in t.iter_mut().zip(col) {
                *ti = fmadd(l, yk, *ti);
            }
        }
        let rows = &self.rows[r0..r0 + m];
        let slots = &self.ext_slot[r0..r0 + m];
        for i in 0..m {
            if slots[i] == NONE {
                let r = rows[i] as usize;
                y[r] = y[r] - t[i];
            } else {
                let sl = slots[i] as usize;
                acc[sl] = acc[sl] + t[i];
            }
        }
    }

    /// Backward sweep through one supernode: `x[c0..c0+w] -= L_off^T x[rows]`
    /// then the unit-upper block solve (plain transpose, no conjugation).
    fn bwd_node(&self, s: u32, x: &mut [T], g: &mut Vec<T>) {
        let (c0, w, r0, m, ld, panel) = self.node(s);
        g.clear();
        g.extend(self.rows[r0..r0 + m].iter().map(|&r| x[r as usize]));
        let unit = self.diag_inv.is_empty();
        for k in (0..w).rev() {
            let col = &panel[k * ld..(k + 1) * ld];
            let mut acc = dot4(&col[w..], g);
            for i in k + 1..w {
                acc = fmadd(col[i], x[c0 + i], acc);
            }
            x[c0 + k] = x[c0 + k] - acc;
            if !unit {
                x[c0 + k] = x[c0 + k] * self.diag_inv[c0 + k];
            }
        }
    }

    /// Run `f` inside the rayon pool when `par`: every parallel section is
    /// cheap to start from a worker and expensive to inject from outside, so
    /// a solve enters the pool once.
    fn in_pool<R: Send>(par: bool, f: impl FnOnce() -> R + Send) -> R {
        if par && rayon::current_thread_index().is_none() && rayon::current_num_threads() > 1 {
            rayon::join(f, || ()).0
        } else {
            f()
        }
    }

    /// `f` over the items with their scratch, on the pool when `par`, in
    /// order on the calling thread otherwise: the same sums either way.
    fn each<A: Sync, S: Send>(
        par: bool,
        items: &[A],
        scratch: &mut [S],
        f: impl Fn(usize, &A, &mut S) + Sync + Send,
    ) {
        if par {
            items
                .par_iter()
                .zip(scratch.par_iter_mut())
                .enumerate()
                .for_each(|(i, (a, s))| f(i, a, s));
        } else {
            for (i, (a, s)) in items.iter().zip(scratch.iter_mut()).enumerate() {
                f(i, a, s);
            }
        }
    }

    /// `work` fitted to a sweep of `nr` right-hand sides: a scratch per
    /// subtree and per node of the widest ancestor level, and the decision
    /// to run on the pool, taken on the sweep's work (panel entries times
    /// right-hand sides up to four: a wider block gains no more from the
    /// pool than four columns do).
    fn fit<'w>(&self, work: &'w mut PlanWork<T>, nr: usize) -> &'w mut PlanWork<T> {
        let widest = self.top_levels.iter().map(Vec::len).max().unwrap_or(0);
        work.fit(self.subtrees.len(), widest);
        work.parallel = self.vals.len().saturating_mul(nr.min(4)) >= self.cfg.par_min_work;
        work
    }

    /// Forward sweep `L y = y` in place on `nr` row-major right-hand sides.
    pub fn forward(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>) {
        let work = self.fit(work, nr);
        Self::in_pool(work.parallel, || {
            if nr == 1 {
                self.forward_single(y, work);
            } else {
                self.forward_block(nr, y, work);
            }
        })
    }

    /// Backward sweep `L^T x = x` (or `U x = x` for a non-unit factor) in
    /// place on `nr` row-major right-hand sides.
    pub fn backward(&self, nr: usize, x: &mut [T], work: &mut PlanWork<T>) {
        let work = self.fit(work, nr);
        Self::in_pool(work.parallel, || {
            if nr == 1 {
                self.backward_single(x, work);
            } else {
                self.backward_block(nr, x, work);
            }
        })
    }

    /// Solve `L D L^T y = y` in place on the permuted, scaled right-hand side.
    pub fn solve_in_place(
        &self,
        f: &LdltPivots<T>,
        y: &mut [T],
        work: &mut PlanWork<T>,
    ) -> Result<(), RslabError> {
        let work = self.fit(work, 1);
        Self::in_pool(work.parallel, || self.solve_in_place_inner(f, y, work))
    }

    fn solve_in_place_inner(
        &self,
        f: &LdltPivots<T>,
        y: &mut [T],
        work: &mut PlanWork<T>,
    ) -> Result<(), RslabError> {
        debug_assert_eq!(y.len(), self.n);
        let mut phases = PhaseTrace::start();
        self.forward_single(y, work);
        phases.lap("forward");
        solve_diagonal(f, y, 1)?;
        phases.lap("diag");
        self.backward_single(y, work);
        phases.lap("backward");
        phases.finish("solve");
        Ok(())
    }

    fn forward_single(&self, y: &mut [T], work: &mut PlanWork<T>) {
        let shared = Shared(y.as_mut_ptr(), y.len());
        let mut phases = PhaseTrace::start();
        // Forward: leaf subtrees in parallel, off-tree updates accumulated.
        let par = work.parallel;
        let scratch = &mut work.subtrees[..self.subtrees.len()];
        Self::each(par, &self.subtrees, scratch, |_, st, sc| {
            sc.acc.clear();
            sc.acc.resize(st.path_cols.len(), T::zero());
            // SAFETY: see `Shared`; this subtree writes only its own
            // columns and reads only them.
            let yv = unsafe { shared.slice() };
            for &s in &st.nodes {
                self.fwd_node(s, yv, &mut sc.acc, &mut sc.t);
            }
        });
        phases.lap("fwd-subtrees");
        for (st, sc) in self.subtrees.iter().zip(&work.subtrees) {
            for (&c, &a) in st.path_cols.iter().zip(&sc.acc) {
                y[c as usize] = y[c as usize] - a;
            }
        }
        phases.lap("fwd-reduce");
        self.top_forward(1, y, work);
        phases.lap("fwd-top");
        phases.finish("forward");
    }

    fn backward_single(&self, y: &mut [T], work: &mut PlanWork<T>) {
        let mut phases = PhaseTrace::start();
        // Backward: ancestors first, then the subtrees in parallel.
        self.top_backward(1, y, work);
        phases.lap("bwd-top");
        let shared = Shared(y.as_mut_ptr(), y.len());
        let par = work.parallel;
        let scratch = &mut work.subtrees[..self.subtrees.len()];
        Self::each(par, &self.subtrees, scratch, |_, st, sc| {
            // SAFETY: see `Shared`; this subtree writes only its own
            // columns and reads its own plus ancestor columns, which
            // are final.
            let xv = unsafe { shared.slice() };
            for &s in st.nodes.iter().rev() {
                self.bwd_node(s, xv, &mut sc.g);
            }
        });
        phases.lap("bwd-subtrees");
        phases.finish("backward");
    }

    // -----------------------------------------------------------------------
    // Ancestor supernodes: level parallelism, node sections at the apex
    // -----------------------------------------------------------------------
    //
    // The supernodes above the cut form the top of the elimination tree.
    // Siblings of one depth are independent: a level with at least as many
    // nodes as threads runs node-parallel, each node swept sequentially with
    // its off-block product accumulated on the node's own ancestor path
    // (the same slot machinery as the leaf subtrees) and reduced afterwards
    // in node order. The apex levels (fewer nodes than threads: the root and
    // the top separators, which hold much of the factor) run node by node
    // with parallel sections inside the node: column-chunk products into
    // private slabs, a row-parallel reduction in fixed chunk order, and the
    // block triangles sequentially. Every sum keeps a fixed association, so
    // the result does not depend on the thread count.

    /// Work of a supernode's sweep: its panel entries.
    #[inline]
    fn work(&self, s: u32) -> usize {
        let (_, w, _, m, _, _) = self.node(s);
        w * (w + m)
    }

    fn top_forward(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>) {
        let par = work.parallel;
        let nt = rayon::current_num_threads().max(1);
        let shared = Shared(y.as_mut_ptr(), y.len());
        for (li, level) in self.top_levels.iter().enumerate().rev() {
            let node_par = !par || level.len() >= nt || nt == 1;
            // One accumulator block for the level, a disjoint slice per node.
            let ptr = &self.top_acc_ptr[li];
            work.top_acc.clear();
            work.top_acc.resize(ptr[level.len()] * nr, T::zero());
            let accs = Shared(work.top_acc.as_mut_ptr(), work.top_acc.len());
            let sweep = |i: usize, s: u32, y: &mut [T], par: bool, sc: &mut Scratch<T>| {
                // SAFETY: node `i` owns its range of the level's block.
                let acc = unsafe { &mut accs.slice()[ptr[i] * nr..ptr[i + 1] * nr] };
                if self.work(s) >= self.cfg.apex_min_work {
                    self.apex_forward(s, nr, y, acc, par, sc);
                } else if nr == 1 {
                    self.fwd_node(s, y, acc, &mut sc.t);
                } else {
                    self.fwd_node_block(s, nr, y, acc, &mut sc.t);
                }
            };
            if node_par {
                Self::each(par, level, &mut work.level[..level.len()], |i, &s, sc| {
                    // SAFETY: see `Shared`; the node writes only its own
                    // columns, its off-block rows go to its accumulator.
                    let yv = unsafe { shared.slice() };
                    sweep(i, s, yv, false, sc);
                });
            } else {
                for (i, &s) in level.iter().enumerate() {
                    sweep(i, s, y, true, &mut work.apex);
                }
            }
            for (i, &s) in level.iter().enumerate() {
                let path = &self.top_paths[self.top_index[s as usize] as usize];
                let acc = &work.top_acc[ptr[i] * nr..ptr[i + 1] * nr];
                for (k, &c) in path.iter().enumerate() {
                    sub_assign(
                        &mut y[c as usize * nr..(c as usize + 1) * nr],
                        &acc[k * nr..(k + 1) * nr],
                    );
                }
            }
        }
    }

    fn top_backward(&self, nr: usize, x: &mut [T], work: &mut PlanWork<T>) {
        let par = work.parallel;
        let nt = rayon::current_num_threads().max(1);
        let shared = Shared(x.as_mut_ptr(), x.len());
        for level in &self.top_levels {
            let node_par = !par || level.len() >= nt || nt == 1;
            let sweep = |s: u32, x: &mut [T], par: bool, sc: &mut Scratch<T>| {
                if self.work(s) >= self.cfg.apex_min_work {
                    self.apex_backward(s, nr, x, par, sc);
                } else if nr == 1 {
                    self.bwd_node(s, x, &mut sc.g);
                } else {
                    self.bwd_node_block(s, nr, x, &mut sc.g, &mut sc.accv);
                }
            };
            if node_par {
                Self::each(par, level, &mut work.level[..level.len()], |_, &s, sc| {
                    // SAFETY: see `Shared`; the node writes only its own
                    // columns and reads final ancestor columns.
                    let xv = unsafe { shared.slice() };
                    sweep(s, xv, false, sc);
                });
            } else {
                for &s in level {
                    sweep(s, x, true, &mut work.apex);
                }
            }
        }
    }

    /// Forward sweep through one large ancestor node: the extended vector
    /// `v = [y_block, t]` (`t` the negated off-block product) in column
    /// blocks of `block`; the block triangle sequential, the update of the
    /// rows below as column-chunk products into private slabs plus a
    /// reduction in fixed chunk order, both parallel when `par` (the same
    /// arithmetic either way). The off-block rows end up in `acc` (the
    /// node's ancestor-path slots).
    fn apex_forward(
        &self,
        s: u32,
        nr: usize,
        y: &mut [T],
        acc: &mut [T],
        par: bool,
        sc: &mut Scratch<T>,
    ) {
        let (c0, w, r0, m, ld, panel) = self.node(s);
        let v = &mut sc.v;
        v.clear();
        v.extend_from_slice(&y[c0 * nr..(c0 + w) * nr]);
        v.resize(ld * nr, T::zero());
        let gmax = self.cfg.block.div_ceil(self.cfg.ancestor_chunk);
        let partial = &mut sc.partial;
        for (jb, je) in col_blocks(w, self.cfg.block) {
            let dinv = self.diag_inv.get(c0..c0 + w).unwrap_or(&[]);
            tri_forward(v, panel, dinv, ld, nr, jb, je);
            if je == ld {
                break;
            }
            // Column chunk `g` of the block: `[jb + g * chunk, ...)`.
            let chunk = self.cfg.ancestor_chunk;
            let used = (je - jb).div_ceil(chunk);
            let rows = ld - je;
            let slab = rows * nr;
            partial.clear();
            partial.resize(gmax * slab, T::zero());
            let (vhead, vtail) = v.split_at_mut(je * nr);
            let vhead: &[T] = vhead;
            let product = |g: usize, out: &mut [T]| {
                for k in jb + g * chunk..(jb + (g + 1) * chunk).min(je) {
                    let vk = &vhead[k * nr..(k + 1) * nr];
                    let col = &panel[k * ld + je..(k + 1) * ld];
                    if nr == 1 {
                        // `col[i] * vk`, the block branch's operand order: the fused complex
                        // multiply-add is not symmetric in its factors (the imaginary part
                        // nests the two cross products in operand order), so the other order
                        // made a one-column solve differ from a block column under FMA.
                        for (o, &l) in out.iter_mut().zip(col) {
                            *o = fmadd(l, vk[0], *o);
                        }
                    } else {
                        for (row, &l) in out.chunks_exact_mut(nr).zip(col) {
                            axpy(row, l, vk);
                        }
                    }
                }
            };
            let rchunk = (rows / (4 * rayon::current_num_threads().max(1))).max(64) * nr;
            let reduce = |ci: usize, vr: &mut [T], partial: &[T]| {
                let o = ci * rchunk;
                for g in 0..used {
                    sub_assign(vr, &partial[g * slab + o..g * slab + o + vr.len()]);
                }
            };
            if par {
                partial[..used * slab]
                    .par_chunks_mut(slab)
                    .enumerate()
                    .for_each(|(g, out)| product(g, out));
                let partial: &[T] = partial;
                vtail
                    .par_chunks_mut(rchunk)
                    .enumerate()
                    .for_each(|(ci, vr)| reduce(ci, vr, partial));
            } else {
                for (g, out) in partial[..used * slab].chunks_mut(slab).enumerate() {
                    product(g, out);
                }
                for (ci, vr) in vtail.chunks_mut(rchunk).enumerate() {
                    reduce(ci, vr, partial);
                }
            }
        }
        y[c0 * nr..(c0 + w) * nr].copy_from_slice(&v[..w * nr]);
        let slots = &self.ext_slot[r0..r0 + m];
        for (i, vi) in v[w * nr..].chunks_exact(nr).enumerate() {
            let sl = slots[i] as usize;
            // `v` holds the negated product; `acc` collects the product.
            sub_assign(&mut acc[sl * nr..(sl + 1) * nr], vi);
        }
    }

    /// Backward sweep through one large ancestor node: column blocks last
    /// to first; the dots of a block's columns against the rows below in
    /// column chunks (parallel when `par`), the triangle sequentially.
    fn apex_backward(&self, s: u32, nr: usize, x: &mut [T], par: bool, sc: &mut Scratch<T>) {
        let (c0, w, r0, m, ld, panel) = self.node(s);
        let v = &mut sc.v;
        v.clear();
        v.extend_from_slice(&x[c0 * nr..(c0 + w) * nr]);
        for &r in &self.rows[r0..r0 + m] {
            v.extend_from_slice(&x[r as usize * nr..(r as usize + 1) * nr]);
        }
        let accv = &mut sc.accv;
        accv.clear();
        accv.resize(w * nr, T::zero());
        for (jb, je) in col_blocks(w, self.cfg.block).rev() {
            if je < ld {
                let tail: &[T] = &v[je * nr..ld * nr];
                let dots = |c: usize, outs: &mut [T]| {
                    for (kk, out) in outs.chunks_exact_mut(nr).enumerate() {
                        let k = jb + c * self.cfg.ancestor_chunk + kk;
                        let col = &panel[k * ld + je..(k + 1) * ld];
                        if nr == 1 {
                            out[0] = dot4(col, tail);
                        } else {
                            dot4_block(out, col, tail, nr);
                        }
                    }
                };
                let ab = &mut accv[jb * nr..je * nr];
                if par {
                    ab.par_chunks_mut(self.cfg.ancestor_chunk * nr)
                        .enumerate()
                        .for_each(|(c, outs)| dots(c, outs));
                } else {
                    for (c, outs) in ab.chunks_mut(self.cfg.ancestor_chunk * nr).enumerate() {
                        dots(c, outs);
                    }
                }
            }
            let dinv = (!self.diag_inv.is_empty()).then(|| &self.diag_inv[c0..c0 + w]);
            tri_backward(v, accv, panel, ld, nr, jb, je, dinv);
        }
        x[c0 * nr..(c0 + w) * nr].copy_from_slice(&v[..w * nr]);
    }

    // -----------------------------------------------------------------------
    // Blocked right-hand sides (row-major `n x nrhs`)
    // -----------------------------------------------------------------------

    fn fwd_node_block(&self, s: u32, nr: usize, y: &mut [T], acc: &mut [T], t: &mut Vec<T>) {
        let (c0, w, r0, m, ld, panel) = self.node(s);
        for k in 0..w {
            let (head, tail) = y.split_at_mut((c0 + k + 1) * nr);
            if let Some(&d) = self.diag_inv.get(c0 + k) {
                for v in &mut head[(c0 + k) * nr..] {
                    *v = *v * d;
                }
            }
            let yk = &head[(c0 + k) * nr..];
            let col = &panel[k * ld..k * ld + w];
            for i in k + 1..w {
                let l = col[i];
                let row = &mut tail[(i - k - 1) * nr..(i - k) * nr];
                axpy_neg(row, l, yk);
            }
        }
        if m == 0 {
            return;
        }
        t.clear();
        t.resize(m * nr, T::zero());
        for k in 0..w {
            let yk = &y[(c0 + k) * nr..(c0 + k + 1) * nr];
            let col = &panel[k * ld + w..(k + 1) * ld];
            for (ti, &l) in t.chunks_exact_mut(nr).zip(col) {
                axpy(ti, l, yk);
            }
        }
        let rows = &self.rows[r0..r0 + m];
        let slots = &self.ext_slot[r0..r0 + m];
        for (i, ti) in t.chunks_exact(nr).enumerate() {
            if slots[i] == NONE {
                let r = rows[i] as usize;
                sub_assign(&mut y[r * nr..(r + 1) * nr], ti);
            } else {
                let sl = slots[i] as usize;
                add_assign(&mut acc[sl * nr..(sl + 1) * nr], ti);
            }
        }
    }

    fn bwd_node_block(&self, s: u32, nr: usize, x: &mut [T], g: &mut Vec<T>, accv: &mut Vec<T>) {
        let (c0, w, r0, m, ld, panel) = self.node(s);
        g.clear();
        for &r in &self.rows[r0..r0 + m] {
            g.extend_from_slice(&x[r as usize * nr..(r as usize + 1) * nr]);
        }
        accv.clear();
        accv.resize(nr, T::zero());
        for k in (0..w).rev() {
            let col = &panel[k * ld..(k + 1) * ld];
            dot4_block(accv, &col[w..], g, nr);
            for i in k + 1..w {
                let l = col[i];
                let xi = &x[(c0 + i) * nr..(c0 + i + 1) * nr];
                axpy(accv, l, xi);
            }
            let xk = &mut x[(c0 + k) * nr..(c0 + k + 1) * nr];
            sub_assign(xk, accv);
            if !self.diag_inv.is_empty() {
                let d = self.diag_inv[c0 + k];
                xk.iter_mut().for_each(|v| *v = *v * d);
            }
        }
    }

    /// Solve `L D L^T Y = Y` in place on a row-major `n x nrhs` block.
    pub fn solve_block_in_place(
        &self,
        f: &LdltPivots<T>,
        y: &mut [T],
        nr: usize,
        work: &mut PlanWork<T>,
    ) -> Result<(), RslabError> {
        let work = self.fit(work, nr);
        Self::in_pool(work.parallel, || self.solve_block_inner(f, y, nr, work))
    }

    fn solve_block_inner(
        &self,
        f: &LdltPivots<T>,
        y: &mut [T],
        nr: usize,
        work: &mut PlanWork<T>,
    ) -> Result<(), RslabError> {
        debug_assert_eq!(y.len(), self.n * nr);
        let mut phases = PhaseTrace::start();
        self.forward_block(nr, y, work);
        phases.lap("forward");
        solve_diagonal(f, y, nr)?;
        phases.lap("diag");
        self.backward_block(nr, y, work);
        phases.lap("backward");
        phases.finish("solve-block");
        Ok(())
    }

    fn forward_block(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>) {
        let shared = Shared(y.as_mut_ptr(), y.len());
        let mut phases = PhaseTrace::start();
        let par = work.parallel;
        let scratch = &mut work.subtrees[..self.subtrees.len()];
        Self::each(par, &self.subtrees, scratch, |_, st, sc| {
            sc.acc.clear();
            sc.acc.resize(st.path_cols.len() * nr, T::zero());
            // SAFETY: see `Shared`.
            let yv = unsafe { shared.slice() };
            for &s in &st.nodes {
                self.fwd_node_block(s, nr, yv, &mut sc.acc, &mut sc.t);
            }
        });
        phases.lap("fwd-subtrees");
        for (st, sc) in self.subtrees.iter().zip(&work.subtrees) {
            for (i, &c) in st.path_cols.iter().enumerate() {
                let yr = &mut y[c as usize * nr..(c as usize + 1) * nr];
                sub_assign(yr, &sc.acc[i * nr..(i + 1) * nr]);
            }
        }
        phases.lap("fwd-reduce");
        self.top_forward(nr, y, work);
        phases.lap("fwd-top");
        phases.finish("forward-block");
    }

    fn backward_block(&self, nr: usize, y: &mut [T], work: &mut PlanWork<T>) {
        let mut phases = PhaseTrace::start();
        self.top_backward(nr, y, work);
        phases.lap("bwd-top");
        let shared = Shared(y.as_mut_ptr(), y.len());
        let par = work.parallel;
        let scratch = &mut work.subtrees[..self.subtrees.len()];
        Self::each(par, &self.subtrees, scratch, |_, st, sc| {
            // SAFETY: see `Shared`.
            let xv = unsafe { shared.slice() };
            for &s in st.nodes.iter().rev() {
                self.bwd_node_block(s, nr, xv, &mut sc.g, &mut sc.accv);
            }
        });
        phases.lap("bwd-subtrees");
        phases.finish("backward-block");
    }
}

/// `y += a * x` over equal-length slices (bounds checks hoisted, vectorizable).
#[inline(always)]
fn axpy<T: Scalar>(y: &mut [T], a: T, x: &[T]) {
    let n = y.len().min(x.len());
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = fmadd(a, xi, *yi);
    }
}

/// `out[c] = dot4(col, g[:, c])` for every column of the row-major `g` (`nr`
/// wide): four independent partial sums to hide the FMA latency, in exactly
/// the association of [`dot4`]. A column's value therefore does not depend on
/// how many right-hand sides share the sweep, which a non-flexible Krylov
/// method needs (its update applies the solve to one column, its Arnoldi
/// steps to a block; any difference breaks the Arnoldi relation).
///
/// The columns go in groups of [`DOT_COLS`] ([`dot4_cols`]), their partial
/// sums on the stack: every column is summed alike whatever its group.
#[inline(always)]
fn dot4_block<T: Scalar>(out: &mut [T], col: &[T], g: &[T], nr: usize) {
    for c0 in (0..nr).step_by(DOT_COLS) {
        let w = DOT_COLS.min(nr - c0);
        dot4_cols(&mut out[c0..c0 + w], col, g, nr, c0);
    }
}

/// Columns per group of [`dot4_block`].
const DOT_COLS: usize = 8;

/// [`dot4_block`] for the columns `c0..c0 + out.len()` of `g`, at most
/// [`DOT_COLS`] of them.
#[inline(always)]
fn dot4_cols<T: Scalar>(out: &mut [T], col: &[T], g: &[T], nr: usize, c0: usize) {
    let (m, w) = (col.len().min(g.len() / nr.max(1)), out.len());
    let mut s = [[T::zero(); DOT_COLS]; 4];
    let at = |i: usize| &g[i * nr + c0..i * nr + c0 + w];
    let mut i = 0;
    while i + 4 <= m {
        for (q, sq) in s.iter_mut().enumerate() {
            axpy(&mut sq[..w], col[i + q], at(i + q));
        }
        i += 4;
    }
    for (c, oc) in out.iter_mut().enumerate() {
        *oc = (s[0][c] + s[1][c]) + (s[2][c] + s[3][c]);
    }
    while i < m {
        axpy(out, col[i], at(i));
        i += 1;
    }
}

/// `y -= a * x`.
#[inline(always)]
fn axpy_neg<T: Scalar>(y: &mut [T], a: T, x: &[T]) {
    let n = y.len().min(x.len());
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = *yi - a * xi;
    }
}

/// `y -= x`.
#[inline(always)]
fn sub_assign<T: Scalar>(y: &mut [T], x: &[T]) {
    let n = y.len().min(x.len());
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = *yi - xi;
    }
}

/// `y += x`.
#[inline(always)]
fn add_assign<T: Scalar>(y: &mut [T], x: &[T]) {
    let n = y.len().min(x.len());
    for (yi, &xi) in y[..n].iter_mut().zip(&x[..n]) {
        *yi = *yi + xi;
    }
}

/// Per-phase wall times of one solve, emitted at the `debug` log level
/// (nothing is measured otherwise).
struct PhaseTrace {
    t0: Option<crate::clock::Instant>,
    laps: Vec<(&'static str, f64)>,
}

impl PhaseTrace {
    fn start() -> Self {
        let on = crate::logging::enabled(crate::logging::LogLevel::Debug);
        Self {
            t0: on.then(crate::clock::Instant::now),
            laps: Vec::new(),
        }
    }

    fn lap(&mut self, name: &'static str) {
        if let Some(t0) = self.t0 {
            self.laps.push((name, t0.elapsed().as_secs_f64() * 1e3));
            self.t0 = Some(crate::clock::Instant::now());
        }
    }

    fn finish(&self, what: &str) {
        if self.t0.is_some() {
            let total: f64 = self.laps.iter().map(|l| l.1).sum();
            let parts: Vec<String> = self
                .laps
                .iter()
                .map(|(n, ms)| format!("{n}={ms:.2}ms"))
                .collect();
            crate::logging::debug(&format!(
                "{what}: {total:.2}ms threads={} {}",
                rayon::current_num_threads(),
                parts.join(" ")
            ));
        }
    }
}

/// Unit-lower triangular solve of column block `[jb, je)` on `v` (row-major
/// `nr` wide), in place.
#[allow(clippy::too_many_arguments)]
fn tri_forward<T: Scalar>(
    v: &mut [T],
    panel: &[T],
    diag_inv: &[T],
    ld: usize,
    nr: usize,
    jb: usize,
    je: usize,
) {
    for k in jb..je {
        let (head, tail) = v.split_at_mut((k + 1) * nr);
        if let Some(&d) = diag_inv.get(k) {
            for x in &mut head[k * nr..] {
                *x = *x * d;
            }
        }
        let vk = &head[k * nr..];
        let col = &panel[k * ld + k + 1..k * ld + je];
        if nr == 1 {
            axpy_neg(&mut tail[..je - k - 1], vk[0], col);
        } else {
            for (row, &l) in tail[..(je - k - 1) * nr].chunks_exact_mut(nr).zip(col) {
                axpy_neg(row, l, vk);
            }
        }
    }
}

/// Unit-upper (`L^T`) solve of column block `[jb, je)` on `v`, given in
/// `accv[k]` the dots of column `k` against the rows below the block.
#[allow(clippy::too_many_arguments)]
fn tri_backward<T: Scalar>(
    v: &mut [T],
    accv: &mut [T],
    panel: &[T],
    ld: usize,
    nr: usize,
    jb: usize,
    je: usize,
    dinv: Option<&[T]>,
) {
    for k in (jb..je).rev() {
        let col = &panel[k * ld + k + 1..k * ld + je];
        let (head, tail) = v.split_at_mut((k + 1) * nr);
        let ak = &mut accv[k * nr..(k + 1) * nr];
        if nr == 1 {
            ak[0] = ak[0] + dot4(col, &tail[..je - k - 1]);
        } else {
            // as the single column: the dot first, then onto the accumulator
            for c0 in (0..nr).step_by(DOT_COLS) {
                let w = DOT_COLS.min(nr - c0);
                let mut d = [T::zero(); DOT_COLS];
                dot4_cols(&mut d[..w], col, &tail[..(je - k - 1) * nr], nr, c0);
                add_assign(&mut ak[c0..c0 + w], &d[..w]);
            }
        }
        let xk = &mut head[k * nr..];
        sub_assign(xk, ak);
        if let Some(d) = dinv {
            let d = d[k];
            xk.iter_mut().for_each(|x| *x = *x * d);
        }
    }
}

/// Column blocks `[jb, je)` of width `nb` over `w` columns.
fn col_blocks(w: usize, nb: usize) -> impl DoubleEndedIterator<Item = (usize, usize)> {
    (0..w).step_by(nb).map(move |jb| (jb, (jb + nb).min(w)))
}

/// Dot product with four independent accumulators (fixed summation order).
#[inline]
fn dot4<T: Scalar>(a: &[T], b: &[T]) -> T {
    debug_assert_eq!(a.len(), b.len());
    let mut s = [T::zero(); 4];
    let mut i = 0;
    while i + 4 <= a.len() {
        s[0] = fmadd(a[i], b[i], s[0]);
        s[1] = fmadd(a[i + 1], b[i + 1], s[1]);
        s[2] = fmadd(a[i + 2], b[i + 2], s[2]);
        s[3] = fmadd(a[i + 3], b[i + 3], s[3]);
        i += 4;
    }
    let mut acc = (s[0] + s[1]) + (s[2] + s[3]);
    while i < a.len() {
        acc = fmadd(a[i], b[i], acc);
        i += 1;
    }
    acc
}

/// `D z = y` for the block diagonal (1x1 and 2x2 pivots), on `nr` right-hand
/// sides stored row-major.
fn solve_diagonal<T: Scalar>(f: &LdltPivots<T>, y: &mut [T], nr: usize) -> Result<(), RslabError> {
    let n = f.n;
    let mut k = 0;
    while k < n {
        if f.two_by_two[k] {
            let d11 = f.d_diag[k];
            let d21 = f.d_subdiag[k];
            let d22 = f.d_diag[k + 1];
            let det = d11 * d22 - d21 * d21;
            if det == T::zero() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            let detinv = det.recip();
            let (r0, r1) = y[k * nr..(k + 2) * nr].split_at_mut(nr);
            for c in 0..nr {
                let z0 = r0[c];
                let z1 = r1[c];
                r0[c] = (d22 * z0 - d21 * z1) * detinv;
                r1[c] = (d11 * z1 - d21 * z0) * detinv;
            }
            k += 2;
        } else {
            let d = f.d_diag[k];
            if d == T::zero() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            let dinv = d.recip();
            for v in &mut y[k * nr..(k + 1) * nr] {
                *v = *v * dinv;
            }
            k += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{CscMatrix, LdltSolver, SolverSettings};

    /// A 2D grid Laplacian shifted to be indefinite (2x2 pivots appear).
    fn grid(m: usize, shift: f64) -> CscMatrix<f64> {
        let n = m * m;
        let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
        for j in 0..n {
            let (x, y) = (j % m, j / m);
            row_idx.push(j);
            values.push(4.0 - shift + 0.1 * ((j * 7919) % 13) as f64);
            if x + 1 < m {
                row_idx.push(j + 1);
                values.push(-1.0);
            }
            if y + 1 < m {
                row_idx.push(j + m);
                values.push(-1.0);
            }
            col_ptr.push(row_idx.len());
        }
        CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// A 3D grid Laplacian: wide separator supernodes exercise the blocked,
    /// parallel ancestor kernels.
    fn grid3d(m: usize) -> CscMatrix<f64> {
        let n = m * m * m;
        let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
        for j in 0..n {
            let (x, y, z) = (j % m, (j / m) % m, j / (m * m));
            row_idx.push(j);
            values.push(6.0 + 0.01 * ((j * 7919) % 13) as f64);
            for (cond, nb) in [
                (x + 1 < m, j + 1),
                (y + 1 < m, j + m),
                (z + 1 < m, j + m * m),
            ] {
                if cond {
                    row_idx.push(nb);
                    values.push(-1.0);
                }
            }
            col_ptr.push(row_idx.len());
        }
        CscMatrix {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    fn residual(a: &CscMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = a.n;
        let mut r = b.to_vec();
        for j in 0..n {
            for e in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[e];
                r[i] -= a.values[e] * x[j];
                if i != j {
                    r[j] -= a.values[e] * x[i];
                }
            }
        }
        let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        rn / bn
    }

    #[test]
    fn plan_solve_matches_scalar_kernel_and_is_thread_invariant() {
        let cases: Vec<(String, CscMatrix<f64>)> = vec![
            ("grid 7".into(), grid(7, 0.0)),
            ("grid 40".into(), grid(40, 0.0)),
            ("grid 40 indefinite".into(), grid(40, 3.7)),
            ("grid3d 26".into(), grid3d(26)),
        ];
        for (name, a) in &cases {
            let (m, shift) = (name.as_str(), 0.0);
            let n = a.n;
            let opts = SolverSettings::default()
                .with_threads(1)
                .with_ordering(crate::OrderingMethod::MetisND);
            let s = LdltSolver::factor(a, &opts).unwrap();
            let b: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
            let x1 = s.solve(&b).unwrap();
            assert!(residual(a, &x1, &b) < 1e-10, "residual m={m} shift={shift}");
            // Block solve, columns must agree with single solves.
            let nrhs = 3;
            let bb: Vec<f64> = (0..n * nrhs)
                .map(|k| ((k * 13) % 11) as f64 - 5.0)
                .collect();
            let xb = s.solve_many(&bb, nrhs).unwrap();
            for c in 0..nrhs {
                let bc: Vec<f64> = (0..n).map(|i| bb[c * n + i]).collect();
                let xc = s.solve(&bc).unwrap();
                for i in 0..n {
                    assert!((xb[c * n + i] - xc[i]).abs() <= 1e-9 * (1.0 + xc[i].abs()));
                }
            }
            // Bit-identical for every thread count.
            for threads in [1usize, 2, 5] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let xt = pool.install(|| s.solve(&b).unwrap());
                assert_eq!(xt, x1, "threads={threads}");
                let xbt = pool.install(|| s.solve_many(&bb, nrhs).unwrap());
                assert_eq!(xbt, xb, "block threads={threads}");
            }
        }
    }

    #[test]
    fn plan_cuts_the_tree_and_falls_back_on_pruned_factors() {
        let a = grid(60, 0.0);
        let s = LdltSolver::factor(&a, &SolverSettings::default().with_threads(1)).unwrap();
        assert!(s.plan.subtrees.len() > 1);
        assert!(!s.plan.top_levels.is_empty());
        let pruned = LdltSolver::factor(
            &a,
            &SolverSettings::default().with_threads(1).with_drop_tol(0.2),
        )
        .unwrap();
        let b = vec![1.0; a.n];
        let x = pruned.solve(&b).unwrap();
        assert!(x.iter().all(|v| v.is_finite()));
    }

    /// Unsymmetric convection-diffusion grid for the LU plans.
    fn convdiff(m: usize) -> crate::GeneralCsc<f64> {
        let n = m * m;
        let (mut col_ptr, mut row_idx, mut values) = (vec![0usize], Vec::new(), Vec::new());
        for j in 0..n {
            let (x, y) = (j % m, j / m);
            // Column j holds the entries A[i, j]; A[i, j] for neighbors i.
            if y > 0 {
                row_idx.push(j - m);
                values.push(-1.0);
            }
            if x > 0 {
                row_idx.push(j - 1);
                values.push(-1.0 - 0.4);
            }
            row_idx.push(j);
            values.push(4.0 + 0.05 * ((j * 7919) % 13) as f64);
            if x + 1 < m {
                row_idx.push(j + 1);
                values.push(-1.0 + 0.4);
            }
            if y + 1 < m {
                row_idx.push(j + m);
                values.push(-1.0);
            }
            col_ptr.push(row_idx.len());
        }
        crate::GeneralCsc {
            n,
            col_ptr,
            row_idx,
            values,
        }
    }

    fn residual_general(a: &crate::GeneralCsc<f64>, x: &[f64], b: &[f64]) -> f64 {
        let n = a.n;
        let mut r = b.to_vec();
        for j in 0..n {
            for e in a.col_ptr[j]..a.col_ptr[j + 1] {
                r[a.row_idx[e]] -= a.values[e] * x[j];
            }
        }
        let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        rn / bn
    }

    #[test]
    fn lu_plans_solve_and_are_thread_invariant() {
        use crate::{LuSolver, OrderingMethod};
        for m in [30usize, 60] {
            let a = convdiff(m);
            let n = a.n;
            let opts = SolverSettings::default()
                .with_threads(1)
                .with_ordering(OrderingMethod::MetisND);
            let s = LuSolver::factor(&a, &opts).unwrap();
            let b: Vec<f64> = (0..n).map(|i| ((i * 31) % 17) as f64 - 8.0).collect();
            let x1 = s.solve(&b).unwrap();
            assert!(residual_general(&a, &x1, &b) < 1e-10, "residual m={m}");
            let nrhs = 3;
            let bb: Vec<f64> = (0..n * nrhs)
                .map(|k| ((k * 13) % 11) as f64 - 5.0)
                .collect();
            let xb = s.solve_many(&bb, nrhs).unwrap();
            for c in 0..nrhs {
                let bc: Vec<f64> = (0..n).map(|i| bb[c * n + i]).collect();
                let xc = s.solve(&bc).unwrap();
                for i in 0..n {
                    assert!((xb[c * n + i] - xc[i]).abs() <= 1e-9 * (1.0 + xc[i].abs()));
                }
            }
            // Refinement runs through the plans too.
            let (xr, out) = s
                .solve_refined(&a, &b, &crate::RefinePolicy::steps(2))
                .unwrap();
            assert!(out.steps <= 2);
            assert!(residual_general(&a, &xr, &b) < 1e-12);
            for threads in [1usize, 2, 5] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let xt = pool.install(|| s.solve(&b).unwrap());
                assert_eq!(xt, x1, "threads={threads}");
                let xbt = pool.install(|| s.solve_many(&bb, nrhs).unwrap());
                assert_eq!(xbt, xb, "block threads={threads}");
            }
        }
    }
}
