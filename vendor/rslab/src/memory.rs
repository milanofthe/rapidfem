//! Heap accounting and the memory plan: the bytes an object holds, from its
//! containers' capacities, and the heap a factorization will need, predicted
//! from the analysis before any numeric work.
//!
//! The plan counts what exists exactly (the analysis's own heap) and models
//! the rest from the code that allocates it: the factor's panels and solve
//! schedule from the symbolic structure, and each kernel's scratch from the
//! same shape decisions the kernel takes (see the `*_scratch` functions next
//! to the kernels). Buffers grown by `Vec`'s amortized doubling are counted
//! at the capacity the allocator hands out ([`grown`]).

use crate::symbolic::SymbolicFactorization;
use std::fmt;

/// Heap bytes of a vector's allocation (its capacity, not its length).
pub(crate) fn vec_bytes<T>(v: &Vec<T>) -> u64 {
    (v.capacity() * std::mem::size_of::<T>()) as u64
}

/// Heap bytes of a vector of vectors: the outer allocation and every inner one.
pub(crate) fn nested_bytes<T>(v: &Vec<Vec<T>>) -> u64 {
    vec_bytes(v) + v.iter().map(vec_bytes).sum::<u64>()
}

/// Heap bytes of a scratch object: what a [`ScratchPool`] weighs before it
/// keeps one.
///
/// [`ScratchPool`]: crate::numeric::supernodal::ScratchPool
pub(crate) trait HeapBytes {
    fn heap_bytes(&self) -> u64;
}

impl<T> HeapBytes for Vec<T> {
    fn heap_bytes(&self) -> u64 {
        vec_bytes(self)
    }
}

impl<A: HeapBytes, B: HeapBytes, C: HeapBytes> HeapBytes for (A, B, C) {
    fn heap_bytes(&self) -> u64 {
        self.0.heap_bytes() + self.1.heap_bytes() + self.2.heap_bytes()
    }
}

/// The most heap one scratch object, and one worker's buffer of split
/// planes, keep between the kernels that borrow them; larger ones are
/// freed when given back. Small factorizations (Newton loops up to some
/// thousand unknowns) keep all their scratch and refactor without
/// allocating; on large ones a node that needs more allocates its own, which
/// its flops dwarf, and what the pools and threads hold on to stays bounded
/// by the worker count instead of growing to the largest nodes.
pub(crate) const SCRATCH_KEEP: u64 = 256 << 10;

/// The capacity of a `Vec` of capacity `cap` after it is resized or
/// reserved to hold `need` entries: unchanged if it fits, else the larger of
/// `need` and twice the old capacity (the standard library's amortized
/// growth).
pub(crate) fn grown(cap: usize, need: usize) -> usize {
    if need <= cap {
        cap
    } else {
        need.max(2 * cap)
    }
}

/// The capacity of a `Vec` filled by `len` single pushes from empty (or
/// from one element): powers of two from 4 on.
pub(crate) fn pushed(len: usize) -> usize {
    if len == 0 {
        0
    } else {
        len.next_power_of_two().max(4)
    }
}

/// The heap one factorization needs, in bytes, predicted from the analysis
/// before any numeric work: for a preflight check against the memory
/// available, and for scheduling several factorizations side by side.
///
/// Counts the solver's own heap: the analysis, the factor, the kernels'
/// scratch and the solve's work vectors. The caller's matrix and
/// right-hand sides, thread stacks and the allocator's cached pages come on
/// top. The model is an upper bound on what the allocator is asked for,
/// validated against a counting allocator over the benchmark corpus
/// (`benches/memory_peak.py`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct MemoryPlan {
    /// Workers the plan is for (the kernels' scratch scales with them).
    pub threads: usize,
    /// Right-hand sides per solve the plan is for.
    pub nrhs: usize,
    /// Heap the analysis holds now.
    pub analysis_bytes: u64,
    /// What the first factorization adds to the analysis (the permuted input
    /// program it keeps for every later one); zero once it has been factored.
    pub analysis_growth_bytes: u64,
    /// Heap the factor holds: its values, pivots and solve schedule.
    pub factor_bytes: u64,
    /// Heap peak of the factorization above what was live when it began,
    /// the analysis growth included.
    pub factor_peak_bytes: u64,
    /// Heap peak of one solve of `nrhs` right-hand sides, the returned
    /// solution included.
    pub solve_bytes: u64,
    /// Heap kept after the factorization for the next one: the workers'
    /// global-to-local maps and split planes, which live on in the threads
    /// of the pool the calling thread keeps, and the kernels' scratch pools
    /// a refactorized solver holds.
    pub kept_bytes: u64,
}

impl MemoryPlan {
    /// Heap held while the factor is kept for solves: the analysis (grown by
    /// the first factorization), the factor and what the workers keep.
    pub fn resident_bytes(&self) -> u64 {
        self.analysis_bytes + self.analysis_growth_bytes + self.factor_bytes + self.kept_bytes
    }

    /// Heap peak from here on: the factorization with the analysis held,
    /// then a solve with both held. The number a preflight check compares
    /// against the memory available.
    pub fn peak_bytes(&self) -> u64 {
        (self.analysis_bytes + self.factor_peak_bytes).max(self.resident_bytes() + self.solve_bytes)
    }

    /// Does the peak fit in `available_bytes`?
    pub fn fits_in(&self, available_bytes: u64) -> bool {
        self.peak_bytes() <= available_bytes
    }
}

impl fmt::Display for MemoryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mb = |b: u64| b as f64 / 1e6;
        write!(
            f,
            "peak {:.0} MB at {} threads (analysis {:.0} MB, factor {:.0} MB, \
             kept {:.0} MB, factorization +{:.0} MB, solve of {} rhs +{:.0} MB)",
            mb(self.peak_bytes()),
            self.threads,
            mb(self.analysis_bytes + self.analysis_growth_bytes),
            mb(self.factor_bytes),
            mb(self.kept_bytes),
            mb(self.factor_peak_bytes),
            self.nrhs,
            mb(self.solve_bytes),
        )
    }
}

/// Heap every factorization allocates besides its data: the diagnostics and
/// log records, the thread pool's registry and the per-call small vectors.
pub(crate) const BOOKKEEPING: u64 = 512 << 10;

/// Nodes one worker can have in flight: a worker waiting inside a parallel
/// product of its node runs other ready work meanwhile (rayon's work
/// stealing), which can be another node.
pub(crate) const NODES_PER_WORKER: usize = 2;

/// Heap the `workers` of a supernodal factorization hold whatever nodes they
/// run, and what they keep after it: per node in flight `maps`
/// global-to-local maps of `n` entries (a node stolen by a waiting worker
/// takes fresh ones), per worker the GEMM crate's packing slab, and the
/// buffers of split real planes. A buffer grows to at most twice the largest
/// product run on it, and the nodes in flight on different workers are
/// different nodes, so the buffers are bounded by the largest needs of
/// `planes` (one `(entries, workers running them)` per node) over the nodes
/// in flight. Afterwards each worker keeps one set of maps and its planes up
/// to [`SCRATCH_KEEP`]. Returns `(during, kept)`.
pub(crate) fn worker_bytes<T: crate::Scalar>(
    n: usize,
    maps: usize,
    workers: usize,
    planes: &mut [(usize, usize)],
) -> (u64, u64) {
    const GEMM_SLAB: usize = 1 << 20;
    let real = std::mem::size_of::<T>() / if T::COMPLEX { 2 } else { 1 };
    planes.sort_unstable_by_key(|&(p, _)| std::cmp::Reverse(p));
    let (mut left, mut entries) = (NODES_PER_WORKER * workers, 0);
    for &(p, copies) in planes.iter() {
        let take = copies.min(left);
        entries += take * p;
        left -= take;
        if left == 0 || p == 0 {
            break;
        }
    }
    let largest = planes.first().map_or(0, |&(p, _)| 2 * p * real) as u64;
    let during = workers * (NODES_PER_WORKER * maps * 4 * n + GEMM_SLAB) + 2 * entries * real;
    let kept = workers as u64 * ((maps * 4 * n) as u64 + largest.min(SCRATCH_KEEP));
    (during as u64, kept)
}

/// Heap `pools` scratch pools of a factorization on `workers` hold on top of
/// what their running nodes need: each object, in its pool or lent to a
/// node, carries at most [`SCRATCH_KEEP`] from earlier nodes, and no more
/// than the `largest` scratch of any node, and a pool has no more objects
/// than nodes can be in flight.
pub(crate) fn pooled_bytes(pools: usize, workers: usize, largest: u64) -> u64 {
    (pools * NODES_PER_WORKER * workers.max(1)) as u64 * largest.min(SCRATCH_KEEP)
}

/// The heaviest set of at most `workers` x [`NODES_PER_WORKER`] supernodes
/// no two of which are ancestor and descendant, by `weight`: a bound on the
/// node scratch live at once, since a node starts only after its whole
/// subtree has finished. Supernodes are in postorder (children before their
/// parent).
pub(crate) fn concurrent_peak(
    sym: &SymbolicFactorization,
    workers: usize,
    weight: impl Fn(usize) -> u64,
) -> u64 {
    // `best[s][k]`: the heaviest such set of at most `k` nodes in the subtree
    // of `s`, kept only until the parent has merged it; the lengths are capped
    // by the leaf count, which bounds the merges at O(nodes x workers).
    let cap = NODES_PER_WORKER * workers.max(1) + 1;
    let merge = |a: &[u64], b: &[u64]| -> Vec<u64> {
        let len = (a.len() + b.len() - 1).min(cap);
        let mut out = vec![0u64; len];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate().take(len - i) {
                out[i + j] = out[i + j].max(x + y);
            }
        }
        out
    };
    let nodes = &sym.supernodes;
    let mut best: Vec<Vec<u64>> = vec![Vec::new(); nodes.len()];
    let mut is_child = vec![false; nodes.len()];
    for s in 0..nodes.len() {
        let mut acc = vec![0u64];
        for &c in &nodes[s].children {
            debug_assert!(c < s, "supernodes in postorder");
            acc = merge(&acc, &std::mem::take(&mut best[c]));
            is_child[c] = true;
        }
        let own = weight(s);
        if acc.len() == 1 {
            acc.push(own);
        } else {
            acc[1..].iter_mut().for_each(|v| *v = (*v).max(own));
        }
        best[s] = acc;
    }
    let mut all = vec![0u64];
    for s in (0..nodes.len()).filter(|&s| !is_child[s]) {
        all = merge(&all, &best[s]);
    }
    all.into_iter().max().unwrap_or(0)
}
