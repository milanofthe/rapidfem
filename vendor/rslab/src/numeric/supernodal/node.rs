//! Per-node machinery shared by the LDL^T and LU kernels: the cmod plan and
//! the global-to-local row map.

use super::Li;
use crate::scalar::Scalar;

/// How a node applies its descendants' updates (`cmod`), shared by the LDL^T
/// and LU kernels. Every field is a pure function of the node and the
/// kernel thresholds, never of the thread count or of timing: a
/// timing-dependent choice broke bit-identity twice (a chain-phase-dependent
/// `tiled`, and a fork below the gate while few nodes were in flight, which
/// switched GEMMs between serial and parallel mode, not bit-identical for
/// complex scalars); see `tests/ll_thread_determinism.rs`.
pub(crate) struct CmodPlan {
    /// The updaters with work for this node, in updater order.
    pub spans: Vec<Span>,
    /// The node's update work is large enough to fork (see
    /// [`CMOD_FORK_MIN_FLOPS`], raised to the parallel-GEMM threshold).
    pub forks: bool,
    /// Width of the column slabs of the tiled mode.
    pub tile_w: usize,
    /// Tiled mode: the panel is cut into column slabs (disjoint `&mut`
    /// chunks) and each slab receives every updater's contribution in updater
    /// order with a serial GEMM. One fan-out per node instead of one per
    /// update, the slab stays cache-hot across the updaters, and a root
    /// separator running alone with hundreds of updaters still parallelizes.
    /// Not bit-identical to the sequential mode (that one routes small updates
    /// through the scalar kernel, and cutting an update at slab boundaries
    /// changes the GEMM shapes, whose per-element bits are shape-dependent),
    /// which is why the pick is a pure function of the node.
    pub tiled: bool,
}

/// One updater of a node: `l` is the landing range of its off-block `L` rows
/// (`[p0, p1)` within the node's columns), `u` the same for its off-block `U`
/// columns. The symmetric path has one structure, so `l == u` there.
#[derive(Clone, Copy)]
pub(crate) struct Span {
    pub k: usize,
    pub l: (usize, usize),
    pub u: (usize, usize),
}

impl CmodPlan {
    /// The plan of supernode `s` over its `updaters`; `lists(k)` gives an
    /// updater's full `L` row and `U` column lists (own columns first). With
    /// `count_u` the `U12` updates count as work too (the LU kernel).
    pub fn new<'a>(
        sym: &crate::symbolic::SymbolicFactorization,
        s: usize,
        updaters: &[Li],
        lists: impl Fn(usize) -> (&'a [Li], &'a [Li]),
        count_u: bool,
        par_gemm: usize,
        fork_min_flops: usize,
    ) -> Self {
        let (first, ncol) = (sym.supernodes[s].first_col, sym.supernodes[s].ncol);
        let landing = |v: &[Li]| {
            let p0 = v.partition_point(|&g| (g as usize) < first);
            (
                p0,
                p0 + v[p0..].partition_point(|&g| (g as usize) < first + ncol),
            )
        };
        let mut spans = Vec::with_capacity(updaters.len());
        let mut flops: usize = 0;
        for &k in updaters {
            let k = k as usize;
            let nck = sym.supernodes[k].ncol;
            let (lk, uk) = lists(k);
            let (ol, ou) = (&lk[nck..], &uk[nck..]);
            let (l, u) = (landing(ol), landing(ou));
            // `L` rows from the landing range down times the `U` columns landing
            // here, and (LU) the landing `L` rows times the `U` columns past here.
            let lwork = (ol.len() - l.0) * (u.1 - u.0) * nck;
            let uwork = if count_u {
                (l.1 - l.0) * (ou.len() - u.1) * nck
            } else {
                0
            };
            if lwork + uwork == 0 {
                continue;
            }
            flops += lwork + uwork;
            spans.push(Span { k, l, u });
        }
        let forks = flops >= fork_min_flops.max(par_gemm);
        let tile_w = (ncol / 16).clamp(32, 256);
        CmodPlan {
            spans,
            forks,
            tile_w,
            tiled: forks && ncol >= 2 * tile_w,
        }
    }
}

thread_local! {
    /// Per-worker global-to-local maps, held at all-`Li::MAX` between nodes;
    /// two, so a node can map its `L` rows and its `U` columns at once.
    static GLOC_SCRATCH: [std::cell::RefCell<Vec<Li>>; 2] =
        const { [std::cell::RefCell::new(Vec::new()), std::cell::RefCell::new(Vec::new())] };
}

/// Global-to-local row map of one supernode: `map[rows[li]] == li`, every other
/// global index maps to `Li::MAX`. Borrows the worker's scratch and restores
/// it on drop, so every way out of a node (errors included) leaves the
/// invariant intact for the next node on this thread.
pub(crate) struct Gloc<'a> {
    map: Vec<Li>,
    rows: &'a [Li],
    slot: usize,
}

impl<'a> Gloc<'a> {
    pub fn new(n: usize, rows: &'a [Li]) -> Self {
        Self::in_slot(0, n, rows)
    }

    /// A second map alongside [`new`](Self::new)'s, on its own scratch.
    pub fn second(n: usize, rows: &'a [Li]) -> Self {
        Self::in_slot(1, n, rows)
    }

    fn in_slot(slot: usize, n: usize, rows: &'a [Li]) -> Self {
        let mut map = GLOC_SCRATCH.with(|c| std::mem::take(&mut *c[slot].borrow_mut()));
        if map.len() < n {
            map.resize(n, Li::MAX);
        }
        for (li, &g) in rows.iter().enumerate() {
            map[g as usize] = li as Li;
        }
        Gloc { map, rows, slot }
    }
}

impl std::ops::Deref for Gloc<'_> {
    type Target = [Li];
    fn deref(&self) -> &[Li] {
        &self.map
    }
}

impl Drop for Gloc<'_> {
    fn drop(&mut self) {
        for &g in self.rows {
            self.map[g as usize] = Li::MAX;
        }
        let map = std::mem::take(&mut self.map);
        GLOC_SCRATCH.with(|c| *c[self.slot].borrow_mut() = map);
    }
}

/// Static-pivot perturbation, the complex-symmetric analogue of rslab's f64
/// `perturb_to_floor` (`dense::factor`): lift a pivot whose magnitude is below
/// `abs_floor` up to that floor, preserving phase. For `T = f64` this reduces
/// to `sign(d)*max(|d|, abs_floor)`, matching the real kernel.
#[inline]
pub(crate) fn perturb_pivot<T: Scalar>(d: T, abs_floor: f64) -> T {
    let mag = d.magnitude();
    if mag >= abs_floor {
        d
    } else if mag == 0.0 {
        T::from_real(abs_floor)
    } else {
        d * T::from_real(abs_floor / mag)
    }
}
