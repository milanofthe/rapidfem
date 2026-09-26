//! The dependency-count scheduler over the assembly forest.

use super::LlSchedule;
use crate::error::RslabError;

/// The left-looking factorization of the whole assembly forest, shared
/// by the LDLT/LU twins and scheduled by dependency counts instead of fork-join:
/// every node is one task, started by whichever thread finishes its last child.
/// `factor_node` and `emit_free` carry the path-specific kernels.
///
/// Under fork-join a parent runs on the thread that forked its children, after
/// that thread's wait returns - and a rayon wait executes other ready work
/// meanwhile, often a whole unrelated subtree, so the parent started long after
/// its last child had finished. On a 338k FEM system that gap was more than
/// half of the critical chain (1.0 to 1.1 s of 1.8 s at 12 threads). Here
/// nobody waits: a finished node counts down its parent and, as the last child,
/// hands the parent on as the next task. A stolen task is one node, never a
/// subtree. Per node the work is unchanged (factor, free every updater whose
/// last consumer it was, emit itself if nothing above reads it), and what a
/// node computes does not depend on the thread running it. There is no
/// recursion either, so deep chain trees need no large worker stacks here.
pub(crate) fn ll_forest(
    sym: &crate::symbolic::SymbolicFactorization,
    sched: &LlSchedule,
    refcount: &[std::sync::atomic::AtomicUsize],
    factor_node: &(dyn Fn(usize) -> Result<(), RslabError> + Sync),
    emit_free: &(dyn Fn(usize) + Sync),
) -> Result<(), RslabError> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    const NONE: usize = usize::MAX;
    let nodes = &sym.supernodes;
    // Parents and the leaves, depth first from the roots.
    let mut parent = vec![NONE; nodes.len()];
    let mut leaves = Vec::new();
    let mut stack: Vec<usize> = forest_roots(sym);
    while let Some(s) = stack.pop() {
        if nodes[s].children.is_empty() {
            leaves.push(s);
        }
        for &c in &nodes[s].children {
            parent[c] = s;
            stack.push(c);
        }
    }
    let pending: Vec<AtomicUsize> = nodes
        .iter()
        .map(|sn| AtomicUsize::new(sn.children.len()))
        .collect();
    let failed = AtomicBool::new(false);
    let error = std::sync::Mutex::new(None);

    // Factor `s` and release what it was the last consumer of; the parent's
    // task is the caller's to start when this was its last child.
    let node = |s: usize| -> bool {
        if failed.load(Ordering::Relaxed) {
            return false;
        }
        if let Err(e) = factor_node(s) {
            failed.store(true, Ordering::Relaxed);
            if let Ok(mut slot) = error.lock() {
                slot.get_or_insert(e);
            }
            return false;
        }
        // Disjoint `k`, so the wide top-of-tree free runs in parallel.
        const FREE_PAR: usize = 64;
        let free = |k: usize| {
            if refcount[k].fetch_sub(1, Ordering::AcqRel) == 1 {
                emit_free(k);
            }
        };
        if sched.updaters(s).len() >= FREE_PAR {
            sched.updaters(s).par_iter().for_each(|&k| free(k as usize));
        } else {
            for &k in sched.updaters(s) {
                free(k as usize);
            }
        }
        if refcount[s].load(Ordering::Relaxed) == 0 {
            emit_free(s);
        }
        true
    };
    fn run<'s>(
        scope: &rayon::Scope<'s>,
        s: usize,
        node: &'s (dyn Fn(usize) -> bool + Sync),
        parent: &'s [usize],
        pending: &'s [AtomicUsize],
    ) {
        if !node(s) {
            return;
        }
        let p = parent[s];
        if p != NONE && pending[p].fetch_sub(1, Ordering::AcqRel) == 1 {
            scope.spawn(move |sc| run(sc, p, node, parent, pending));
        }
    }
    let node: &(dyn Fn(usize) -> bool + Sync) = &node;
    let (parent, pending) = (&parent[..], &pending[..]);
    rayon::scope(|sc| {
        for &leaf in &leaves {
            sc.spawn(move |sc| run(sc, leaf, node, parent, pending));
        }
    });
    match error.into_inner() {
        Ok(Some(e)) => Err(e),
        _ => Ok(()),
    }
}

/// Roots of the assembly forest: supernodes that are no node's child.
fn forest_roots(sym: &crate::symbolic::SymbolicFactorization) -> Vec<usize> {
    let nsuper = sym.supernodes.len();
    let mut is_child = vec![false; nsuper];
    for snode in &sym.supernodes {
        for &ch in &snode.children {
            is_child[ch] = true;
        }
    }
    (0..nsuper).filter(|&s| !is_child[s]).collect()
}
