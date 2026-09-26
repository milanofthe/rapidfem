//! The numeric LDL^T factorization: the permuted input, the left-looking
//! driver over the assembly forest, and the emit of each finished panel.

use super::node::ll_factor_node;
use super::pivots::LdltPivots;
use crate::numeric::supernodal::analysis::{recommend_threads_for_sym, SupernodalAnalysis};

use crate::error::RslabError;
use crate::inertia::Inertia;
use crate::numeric::settings::{
    stack_for_depth, supernode_tree_depth, SolverSettings, ZeroPivotAction,
};
use crate::numeric::supernodal::panel::{finish_panel, PanelArena, PanelFactor, PanelOut};
use crate::numeric::supernodal::{emit_refcount_offsets, Cells, Input, InputProgram, LlSchedule};
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::symbolic::SymbolicFactorization;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The numeric result of a sparse LDL^T factorization: the unit lower factor
/// `L` in supernodal panel form (the storage the solves run on, written by
/// the drivers without a copy) plus the block diagonal, the pivot
/// permutation and the numeric outcome.
pub(crate) struct LdltNumeric<T> {
    /// `L` in panel form, in elimination order.
    pub factor: PanelFactor<T>,
    /// `D`, the permutation and the outcome.
    pub pivots: LdltPivots<T>,
    /// Structural panel slots holding an exact zero (cancellation or
    /// `drop_tol`); the stored nonzeros are `factor.nnz() - n_zeros`.
    pub n_zeros: usize,
}

pub(crate) fn factor_numeric<T: Scalar>(
    symb: &SupernodalAnalysis,
    a: &CscMatrix<T>,
    scale: Option<&[f64]>,
    opts: &SolverSettings,
) -> Result<LdltNumeric<T>, RslabError> {
    a.validate()?;
    let n = symb.n;
    if a.n != n || a.row_idx.len() != symb.nnz {
        return Err(RslabError::InvalidInput(
            "factor_numeric: matrix does not match the analyzed pattern".to_string(),
        ));
    }
    let inner = match &symb.inner {
        None => {
            return Ok(LdltNumeric {
                factor: PanelFactor::empty(),
                pivots: LdltPivots {
                    n: 0,
                    d_diag: Vec::new(),
                    d_subdiag: Vec::new(),
                    two_by_two: Vec::new(),
                    perm: Vec::new(),
                    supernode_parent: Vec::new(),
                    n_perturbed: 0,
                    inertia: Inertia::new(0, 0, 0),
                },
                n_zeros: 0,
            });
        }
        Some(i) => i,
    };
    let sym = &inner.sym;
    // Worker stack sized to the assembly-tree depth so the recursive tree
    // factorization never overflows on deep chain trees (banded / 1D + low nemin).
    let stack = stack_for_depth(supernode_tree_depth(sym));

    // P^T A P (lower fold) through the cached input program: the structure is
    // frozen on the first factorization of this pattern; every later
    // (re)factorization pays one linear values pass only.
    let prog = inner.input.get_or_init(|| {
        crate::logging::timed(
            || "ldlt: input program".into(),
            || InputProgram::symmetric(&a.col_ptr, &a.row_idx, &sym.perm_inv),
        )
    });
    let weight = scale.map(|s| move |i: usize, j: usize| s[i] * s[j]);
    let vals = prog.values(
        &a.col_ptr,
        &a.row_idx,
        &a.values,
        weight.as_ref().map(|w| w as &dyn Fn(usize, usize) -> f64),
    );
    let inp = Input::new(prog, &vals);

    // Run in a scoped pool of `opts.threads` so concurrent solves don't
    // oversubscribe.
    let sched = inner.schedule();
    opts.threads.run(
        stack,
        |cap| recommend_threads_for_sym(symb, cap),
        || factor_left_looking(sym, sched, inp, opts),
    )
}

/// One factored supernode's left-looking payload: the dense panel, the
/// Bunch-Kaufman D (diagonal + sub-diagonal + 2x2 flags, pivoted order), and
/// the within-panel pivot permutation (identity on the off-diagonal rows).
pub(super) struct LdltSlot<T> {
    pub(super) d: Vec<T>,
    pub(super) dsub: Vec<T>,
    pub(super) two: Vec<bool>,
    pub(super) lperm: Vec<usize>,
}
impl<T> Default for LdltSlot<T> {
    fn default() -> Self {
        LdltSlot {
            d: Vec::new(),
            dsub: Vec::new(),
            two: Vec::new(),
            lperm: Vec::new(),
        }
    }
}
pub(super) type LlStore<T> = crate::numeric::supernodal::SlotStore<LdltSlot<T>>;

/// Compact (CSC-fragment) form of one supernode's L factor, produced the moment
/// its last consumer pulls from it so the dense panel can be freed during
/// factorization. Row indices are already final elimination positions.
pub(super) struct LlEmitLdlt<T> {
    pub(super) refcount: Vec<AtomicUsize>,
    pub(super) e_offset: Vec<usize>,
    /// The factor's buffer: every supernode factors into its own slot.
    pub(super) arena: PanelArena<T>,
    pub(super) panels: Cells<PanelOut>,
    pub(super) e_of_g: Cells<usize>,
    pub(super) perm: Cells<usize>,
    pub(super) d_diag: Cells<T>,
    pub(super) d_subdiag: Cells<T>,
    pub(super) two_by_two: Cells<bool>,
    // Inertia accumulated across supernodes (block-aware).
    pub(super) inertia_pos: AtomicUsize,
    pub(super) inertia_neg: AtomicUsize,
    pub(super) inertia_zero: AtomicUsize,
}

impl<T: Scalar> LlEmitLdlt<T> {
    fn new(sym: &SymbolicFactorization, sched: &LlSchedule) -> Self {
        let nsuper = sym.supernodes.len();
        let n = sym.n;
        let (refcount, e_offset) = emit_refcount_offsets(sym, sched);
        let arena =
            PanelArena::new((0..nsuper).map(|s| sched.rows(s).len() * sym.supernodes[s].ncol));
        LlEmitLdlt {
            refcount,
            e_offset,
            arena,
            panels: Cells::new_default(nsuper),
            e_of_g: Cells::new(n, usize::MAX),
            perm: Cells::new(n, 0),
            d_diag: Cells::new(n, T::zero()),
            d_subdiag: Cells::new(n, T::zero()),
            two_by_two: Cells::new(n, false),
            inertia_pos: AtomicUsize::new(0),
            inertia_neg: AtomicUsize::new(0),
            inertia_zero: AtomicUsize::new(0),
        }
    }
    #[inline]
    unsafe fn eg(&self, g: usize) -> usize {
        *self.e_of_g.get(g)
    }
}

/// Compact supernode `k`'s L factor and free its dense panel + D/lperm. Called the
/// instant `k`'s last consumer pulled from it. Mirrors the per-supernode body of
/// the legacy L emit (unit diagonal, skip the 2x2 `d21` coupling row).
fn ldlt_emit_and_free<T: Scalar>(
    k: usize,
    store: &LlStore<T>,
    emit: &LlEmitLdlt<T>,
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    drop_tol: Option<f64>,
) {
    let ncol = sym.supernodes[k].ncol;
    let nrow = sched.rows(k).len();
    // SAFETY: the owner of supernode `k` emits it exactly once, after its last
    // updater has read the slot (refcount zero); nobody reads it afterwards.
    let slot = unsafe { store.take(k) };
    let panel = unsafe { emit.arena.slot_mut(k) };
    let (lperm, t2) = (&slot.lperm, &slot.two);
    debug_assert_eq!(panel.len(), nrow * ncol);
    debug_assert!(
        (0..ncol)
            .all(|p| unsafe { emit.eg(sched.rows(k)[lperm[p]] as usize) } == emit.e_offset[k] + p),
        "the diagonal block is in elimination order"
    );
    let e_rows: Vec<u32> = (ncol..nrow)
        .map(|i| unsafe { emit.eg(sched.rows(k)[lperm[i]] as usize) } as u32)
        .collect();
    let out = finish_panel(panel, ncol, e_rows, Some(&t2[..ncol]), drop_tol);
    unsafe { emit.panels.set(k, out) };
    if ldlt_no_free() {
        // The debugging hold: keep an (emptied) shell in place.
        unsafe { store.set(k, LdltSlot::default()) };
    }
}

static LDLT_NO_FREE_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
#[inline]
fn ldlt_no_free() -> bool {
    *LDLT_NO_FREE_FLAG.get_or_init(|| {
        std::env::var("RLA_NO_FREE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Static-pivot floor (absolute), translated from rslab's ZeroPivotAction.
/// `PerturbToEps { abs_floor }` is taken as given (rslab convention: an
/// absolute floor, typically `eps_rel * ||A||inf`); `Fail` disables
/// perturbation. `values` are the values being factored.
fn static_pivot_floor<T: Scalar>(values: &[T], opts: &SolverSettings) -> Option<f64> {
    match opts.pivoting.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = values.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    }
}

/// Supernodal **left-looking** LDL^T with **Bunch-Kaufman 1x1/2x2 pivoting**. Each
/// supernode's dense panel is assembled from `A`, updated by every previously
/// factored descendant (`cmod`: pull the descendant's contribution columns that
/// land in this panel, applying its block-diagonal `D`), then factored in place
/// (`cdiv`: partial Bunch-Kaufman, no trailing update). Pivoting is bounded to
/// each panel's fully-summed block, so the off-diagonal rows keep their identity
/// and the descendant->ancestor `cmod` is unaffected by a panel's internal
/// permutation. There is **no contribution-block stack and no extract copy-out**
/// (the panels are the factor), so the transient is just the factor itself (the
/// PARDISO memory profile), including indefinite (zero-/tiny-diagonal) systems
/// via the 2x2 blocks.
fn factor_left_looking<T: Scalar>(
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    inp: Input<T>,
    opts: &SolverSettings,
) -> Result<LdltNumeric<T>, RslabError> {
    let n = sym.n;
    let perturb_floor = static_pivot_floor(inp.values(), opts);

    let nsuper = sym.supernodes.len();
    // Factor in parallel over the assembly forest: sibling subtrees concurrently,
    // each node after its subtree (whose panels are its only updaters). Panels are
    // written once and read only by ancestors -> no synchronization needed beyond
    // the recursion structure (see `LlStore`).
    let store = LlStore::<T>::new(nsuper);
    let emit = LlEmitLdlt::<T>::new(sym, sched);
    let n_perturbed_atomic = AtomicUsize::new(0);
    let kt = opts.kernel();
    let factor_node = |s: usize| {
        ll_factor_node(
            s,
            sym,
            inp,
            sched,
            &store,
            &emit,
            perturb_floor,
            &n_perturbed_atomic,
            kt,
        )
    };
    let emit_free = |k: usize| ldlt_emit_and_free(k, &store, &emit, sym, sched, opts.drop_tol);
    crate::numeric::supernodal::ll_forest(sym, sched, &emit.refcount, &factor_node, &emit_free)?;
    drop(store); // panels moved into the emit cells; release the shells
    let n_perturbed = n_perturbed_atomic.load(Ordering::Relaxed);
    let kept: Vec<bool> = sym.supernodes.iter().map(|sn| sn.ncol > 0).collect();
    let supernode_parent = crate::symbolic::supernode_parents(&sym.supernodes, &kept);
    let LlEmitLdlt {
        arena,
        panels,
        perm,
        d_diag,
        d_subdiag,
        two_by_two,
        inertia_pos,
        inertia_neg,
        inertia_zero,
        ..
    } = emit;
    let (factor, n_zeros) = arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(panels.get_mut(s))
    });
    let perm: Vec<usize> = (0..n).map(|e| unsafe { *perm.get(e) }).collect();
    let d_diag: Vec<T> = (0..n).map(|e| unsafe { *d_diag.get(e) }).collect();
    let d_subdiag: Vec<T> = (0..n).map(|e| unsafe { *d_subdiag.get(e) }).collect();
    let two_by_two: Vec<bool> = (0..n).map(|e| unsafe { *two_by_two.get(e) }).collect();
    let inertia = Inertia::new(
        inertia_pos.load(Ordering::Relaxed),
        inertia_neg.load(Ordering::Relaxed),
        inertia_zero.load(Ordering::Relaxed),
    );

    Ok(LdltNumeric {
        factor,
        pivots: LdltPivots {
            n,
            d_diag,
            d_subdiag,
            two_by_two,
            perm,
            supernode_parent,
            n_perturbed,
            inertia,
        },
        n_zeros,
    })
}
