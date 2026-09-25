//! The numeric LU factorization: scaling, the permuted input, the
//! left-looking driver over the assembly forest, and the emit of each
//! finished panel into `L` and `U^T`.

use super::factors::LuNumeric;
use super::node::lu_ll_factor_node;
use super::solver::LuSymbolic;
use super::structure::LuStructure;
use crate::numeric::supernodal::{Input, InputProgram};

use crate::error::RslabError;
use crate::numeric::gemm_tuning::KernelTuning;
use crate::numeric::settings::{SolverSettings, ZeroPivotAction};
use crate::numeric::supernodal::panel::{finish_panel, PanelArena, PanelFactor, PanelOut};
use crate::numeric::supernodal::{emit_refcount_offsets, Cells, LlSchedule};
use crate::scalar::Scalar;
use crate::sparse::general::GeneralCsc;
use crate::symbolic::SymbolicFactorization;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Fold the MC64 row matching back into the factors: pivot rows and the row
/// scaling are reported in `A`'s row indices, as the solves expect.
fn finish_matching<T: Scalar>(fac: &mut LuNumeric<T>, lusym: &LuSymbolic) {
    if let Some(m) = &lusym.matching {
        for e in fac.perm_row.iter_mut() {
            *e = m.row_of[*e];
        }
        fac.d_row = m.r.clone();
    }
}

/// One factored supernode's left-looking payload besides its arena slots: the
/// within-front row permutation from partial pivoting (`rperm[i]` is the
/// row-structure index physically at panel position `i`; identity on the
/// trailing rows, only read by the emit).
pub(super) type LuLlStore = crate::numeric::supernodal::SlotStore<Vec<usize>>;

pub(super) struct LlEmit<T> {
    /// Number of consumers (ancestors that pull) still to come; freed at 0.
    pub(super) refcount: Vec<AtomicUsize>,
    /// First elimination position of each supernode (symbolic prefix sum of ncol).
    pub(super) e_offset: Vec<usize>,
    /// The factors' buffers: every supernode factors `L` straight into its
    /// slot of `l_arena`; `U^T` is gathered into `u_arena` at the emit.
    pub(super) l_arena: PanelArena<T>,
    pub(super) u_arena: PanelArena<T>,
    pub(super) panels: Cells<(PanelOut, PanelOut)>,
    /// `e_of_g[g]` = elimination position of COLUMN g; `row_pos_of_g[g]` =
    /// position whose PIVOT ROW is g. Written in-node (disjoint g), read after the
    /// join barrier (and in `emit_and_free`, where the join chain makes consumer
    /// writes visible).
    pub(super) e_of_g: Cells<usize>,
    pub(super) row_pos_of_g: Cells<usize>,
    pub(super) perm: Cells<usize>,
    pub(super) perm_row: Cells<usize>,
}

impl<T: Scalar> LlEmit<T> {
    fn new(sym: &SymbolicFactorization, sched: &LlSchedule, st: &LuStructure) -> Self {
        let n = sym.n;
        let (refcount, e_offset) = emit_refcount_offsets(sym, sched);
        let ns = sym.supernodes.len();
        let ncol = |s: usize| sym.supernodes[s].ncol;
        LlEmit {
            refcount,
            e_offset,
            l_arena: PanelArena::new((0..ns).map(|s| st.rows_l(s).len() * ncol(s))),
            u_arena: PanelArena::new((0..ns).map(|s| st.cols_u(s).len() * ncol(s))),
            panels: Cells::new_default(sym.supernodes.len()),
            e_of_g: Cells::new(n, usize::MAX),
            row_pos_of_g: Cells::new(n, usize::MAX),
            perm: Cells::new(n, 0),
            perm_row: Cells::new(n, 0),
        }
    }
    #[inline]
    unsafe fn eg(&self, g: usize) -> usize {
        *self.e_of_g.get(g)
    }
    #[inline]
    unsafe fn rg(&self, g: usize) -> usize {
        *self.row_pos_of_g.get(g)
    }
}

/// Emit supernode `k` once its last updater is done: the left-looking panel
/// `lbuf` (`L` strictly below the diagonal, `U`'s diagonal block on and above
/// it) becomes the `L` panel as it is, and `U^T`'s panel, whose off-block rows
/// the factorization wrote in place (`U12`), gets its diagonal block from the
/// upper triangle. Both get their off-block rows in elimination order.
fn emit_and_free<T: Scalar>(
    k: usize,
    store: &LuLlStore,
    emit: &LlEmit<T>,
    sym: &SymbolicFactorization,
    st: &LuStructure,
    drop_tol: Option<f64>,
) {
    let snode = &sym.supernodes[k];
    let (first, ncol) = (snode.first_col, snode.ncol);
    let (rows_l, cols_u) = (st.rows_l(k), st.cols_u(k));
    let (nrow_l, nrow_u) = (rows_l.len(), cols_u.len());
    // SAFETY: the owner of supernode `k` emits it exactly once, after its last
    // updater has read the slots (refcount zero); nobody reads them afterwards.
    let rperm = unsafe { store.take(k) };
    let eoff = emit.e_offset[k];
    debug_assert!(
        (0..ncol).all(|p| unsafe { emit.eg(first + p) } == eoff + p)
            && (0..ncol).all(|i| unsafe { emit.rg(rows_l[rperm[i]] as usize) } == eoff + i),
        "the diagonal block is in elimination order"
    );
    // `U^T`'s diagonal block from the upper triangle of the `L` slot.
    {
        let lbuf: &[T] = unsafe { emit.l_arena.slot(k) };
        let ut = unsafe { emit.u_arena.slot_mut(k) };
        debug_assert_eq!(ut.len(), nrow_u * ncol);
        for p in 0..ncol {
            for i in p..ncol {
                ut[p * nrow_u + i] = lbuf[i * nrow_l + p];
            }
        }
    }
    let l_rows: Vec<u32> = (ncol..nrow_l)
        .map(|i| unsafe { emit.rg(rows_l[rperm[i]] as usize) } as u32)
        .collect();
    let u_rows: Vec<u32> = (ncol..nrow_u)
        .map(|t| unsafe { emit.eg(cols_u[t] as usize) } as u32)
        .collect();
    let l_out = finish_panel(
        unsafe { emit.l_arena.slot_mut(k) },
        ncol,
        l_rows,
        None,
        drop_tol,
    );
    let u_out = finish_panel(
        unsafe { emit.u_arena.slot_mut(k) },
        ncol,
        u_rows,
        None,
        drop_tol,
    );
    unsafe { emit.panels.set(k, (l_out, u_out)) };
}

/// Supernodal left-looking LU into an `LuNumeric`. `inp` is the equilibrated permuted matrix; `d_row`/`d_col`
/// the equilibration carried into the result.
#[allow(clippy::too_many_arguments)]
fn factor_lu_left_looking<T: Scalar>(
    sym: &SymbolicFactorization,
    sched: &LlSchedule,
    st: &LuStructure,
    inp: Input<T>,
    d_row: &[f64],
    d_col: &[f64],
    perturb_floor: Option<f64>,
    drop_tol: Option<f64>,
    kt: KernelTuning,
) -> Result<LuNumeric<T>, RslabError> {
    let n = sym.n;
    let nsuper = sym.supernodes.len();
    let store = LuLlStore::new(nsuper);
    let emit = LlEmit::<T>::new(sym, sched, st);
    let n_perturbed_atomic = AtomicUsize::new(0);
    let factor_node = |s: usize| {
        lu_ll_factor_node(
            s,
            sym,
            inp,
            sched,
            st,
            &store,
            &emit,
            perturb_floor,
            &n_perturbed_atomic,
            kt,
        )
    };
    let emit_free = |k: usize| emit_and_free(k, &store, &emit, sym, st, drop_tol);
    crate::numeric::supernodal::ll_forest(sym, sched, &emit.refcount, &factor_node, &emit_free)?;
    drop(store); // panels moved into the emit cells; release the shells
    let n_perturbed = n_perturbed_atomic.load(Ordering::Relaxed);
    let kept: Vec<bool> = sym.supernodes.iter().map(|sn| sn.ncol > 0).collect();
    let supernode_parent = crate::symbolic::supernode_parents(&sym.supernodes, &kept);
    let LlEmit {
        l_arena,
        u_arena,
        panels,
        ..
    } = emit;
    let (l, zeros_l) = l_arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).0)
    });
    let (ut, zeros_u) = u_arena.finish(n, sym.supernodes.iter().map(|sn| sn.ncol), |s| unsafe {
        std::mem::take(&mut panels.get_mut(s).1)
    });
    let perm: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm.get(e) }).collect();
    let perm_row: Vec<usize> = (0..n).map(|e| unsafe { *emit.perm_row.get(e) }).collect();

    Ok(LuNumeric {
        l,
        ut,
        perm,
        perm_row,
        d_row: d_row.to_vec(),
        d_col: d_col.to_vec(),
        supernode_parent,
        n_perturbed,
        n_zeros: zeros_l + zeros_u,
    })
}

/// PARDISO phases 2-3 for the general path: numeric LU reusing a [`LuSymbolic`].
/// `a` must share the analyzed pattern (`n`, `nnz`).
#[allow(clippy::needless_range_loop)] // CSC column loops index col_ptr + scaling
pub(crate) fn factor_general_lu_numeric<T: Scalar>(
    lusym: &LuSymbolic,
    a: &GeneralCsc<T>,
    opts: &SolverSettings,
) -> Result<LuNumeric<T>, RslabError> {
    a.validate()?;
    let n = lusym.n;
    if a.n != n || a.row_idx.len() != lusym.nnz {
        return Err(RslabError::InvalidInput(
            "factor_general_lu_numeric: matrix does not match the analyzed pattern".to_string(),
        ));
    }
    if n == 0 {
        return Ok(LuNumeric {
            l: PanelFactor::empty(),
            ut: PanelFactor::empty(),
            perm: Vec::new(),
            perm_row: Vec::new(),
            d_row: Vec::new(),
            d_col: Vec::new(),
            supernode_parent: Vec::new(),
            n_perturbed: 0,
            n_zeros: 0,
        });
    }

    let perturb_floor: Option<f64> = match opts.pivoting.on_zero_pivot {
        ZeroPivotAction::Fail => None,
        ZeroPivotAction::PerturbToEps { abs_floor } => Some(abs_floor.max(0.0)),
        ZeroPivotAction::ForceAccept => {
            let anorm = a.values.iter().map(|v| v.magnitude()).fold(0.0, f64::max);
            Some(anorm.max(1.0) * f64::EPSILON)
        }
    };

    // The assembly-tree levels are no longer needed: the driver is a
    // work-stealing tree recursion, not a level-synchronous sweep.
    let (sym, _by_level) = lusym
        .symb
        .sym_and_levels()
        .ok_or_else(|| RslabError::InvalidInput("internal: empty symbolic".to_string()))?;

    // Two-sided equilibration A_hat = D_r A D_c with d_r[i] = 1/sqrt(max_j |A_ij|),
    // d_c[j] = 1/sqrt(max_i |A_ij|). Tames the dynamic range (these MoM near-field
    // matrices span ~6 orders) so the LU factor - and any incomplete drop -
    // stays well-scaled; the solve undoes it transparently. Computed from the
    // original (unpermuted) A.
    // The matrix the pipeline factors: `A` itself, or its MC64 row-permuted
    // form `B` (row `i` of `B` is row `row_of[i]` of `A`) with the matching's
    // scalings; otherwise the max-norm equilibration.
    let (d_row, d_col): (Vec<f64>, Vec<f64>) = match &lusym.matching {
        Some(m) => ((0..n).map(|i| m.r[m.row_of[i]]).collect(), m.c.clone()),
        None => {
            let mut rmax = vec![0.0f64; n];
            let mut cmax = vec![0.0f64; n];
            for j in 0..n {
                for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                    let i = a.row_idx[k];
                    let m = a.values[k].magnitude();
                    if m > rmax[i] {
                        rmax[i] = m;
                    }
                    if m > cmax[j] {
                        cmax[j] = m;
                    }
                }
            }
            (
                rmax.iter()
                    .map(|&r| crate::scaling::inv_sqrt_scale_guarded(r))
                    .collect(),
                cmax.iter()
                    .map(|&c| crate::scaling::inv_sqrt_scale_guarded(c))
                    .collect(),
            )
        }
    };

    // `A`'s row `r` is `B`'s row `b_row[r]`, scaled by `d_row[b_row[r]]`.
    let b_row: Option<Vec<usize>> = lusym.matching.as_ref().map(|m| m.row_map());
    let prog = lusym.input.get_or_init(|| {
        crate::logging::timed(
            || "lu: input program".into(),
            || InputProgram::general(&a.col_ptr, &a.row_idx, b_row.as_deref(), sym),
        )
    });
    let weight = |r: usize, j: usize| d_row[b_row.as_ref().map_or(r, |b| b[r])] * d_col[j];
    let vals = prog.values(&a.col_ptr, &a.row_idx, &a.values, Some(&weight));
    drop(b_row);
    let inp = Input::new(prog, &vals);
    // Worker stack sized to the assembly-tree depth (overflow-safe on deep chain
    // trees), shared by both LU paths.
    let stack = crate::numeric::settings::stack_for_depth(
        crate::numeric::settings::supernode_tree_depth(sym),
    );

    // Run in a scoped pool of `opts.threads` so concurrent solves don't
    // oversubscribe.
    let mut fac = opts.threads.run(
        stack,
        |cap| crate::numeric::supernodal::analysis::recommend_threads_for_sym(&lusym.symb, cap),
        || {
            factor_lu_left_looking(
                sym,
                lusym.symb.ll_schedule().ok_or_else(|| {
                    RslabError::InvalidInput("internal: empty symbolic".to_string())
                })?,
                &lusym.structure,
                inp,
                &d_row,
                &d_col,
                perturb_floor,
                opts.drop_tol,
                opts.kernel(),
            )
        },
    )?;
    finish_matching(&mut fac, lusym);
    Ok(fac)
}
