//! The symmetric trailing-update GEMM of the LDL^T kernel, computed only on and
//! below the diagonal.

use crate::scalar::Scalar;

/// Grow `buf` to at least `len` entries without clearing what it holds: the
/// cmod callers overwrite the prefix they read (the D-apply loops fill `vc` and
/// `vd_buf`, and `lower_tile_gemm` writes `u_buf` with `read_dst = false`), so
/// zeroing it before every update only cost bandwidth.
pub(super) fn grow_scratch<T: Scalar>(buf: &mut Vec<T>, len: usize) {
    if buf.len() < len {
        buf.resize(len, T::zero());
    }
}

/// Symmetric trailing-update GEMM computed **only on and below the tile
/// diagonal**: `TMP[:, j] = G * L21^T[:, j]` for rows `>= tile start`. The
/// consumers (the front Schur subtraction and the left-looking panel
/// subtraction) read only entries with `row >= col`, so the full `m x ncols`
/// product wastes up to half the flops (exactly half for the square front
/// Schur, approaching half for wide root panels where `ncols ~ m`). Tiling
/// the columns and starting each tile's rows at its own diagonal keeps the
/// per-element summation deterministic while cutting the waste to
/// `< schur_tile / 2` rows per tile ([`KernelSettings::schur_tile`](crate::KernelSettings::schur_tile)).
///
/// Layouts: `tmp` is `m x ncols` column-major (column stride `m`, row
/// stride 1); `lhs` is `m x k` with column stride `lhs_cs` (row stride 1);
/// `rhs` is read as `k x ncols` with strides `(rhs_cs = 1, rhs_rs)` -
/// element `(kk, j)` at `rhs[j + kk*rhs_rs]`. Each tile's GEMM goes
/// rayon-parallel at/above the `par_cdiv` flop bar.
///
/// SAFETY: the three buffers must be pairwise-disjoint allocations sized
/// for the strides passed (`tmp` >= `m*ncols`; `lhs` rows `[0, m)` x cols
/// `[0, k)` under `lhs_cs`; `rhs` valid at `j + kk*rhs_rs` for `j < ncols`,
/// `kk < k`).
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn lower_tile_gemm<T: Scalar>(
    tmp: &mut [T],
    m: usize,
    ncols: usize,
    k: usize,
    lhs: *const T,
    lhs_cs: isize,
    rhs: *const T,
    rhs_rs: isize,
    par_cdiv: usize,
    ks: &crate::KernelSettings,
) {
    debug_assert!(ncols <= m);
    debug_assert!(tmp.len() >= m * ncols);
    let mut c0 = 0usize;
    while c0 < ncols {
        let tw = ks.schur_tile.max(1).min(ncols - c0);
        let mrows = m - c0;
        let par = if (mrows as u128) * (tw as u128) * (k as u128) >= par_cdiv as u128 {
            gemm::Parallelism::Rayon(0)
        } else {
            gemm::Parallelism::None
        };
        // Dst tile = columns [c0, c0+tw) rows [c0, m) of `tmp`; lhs = rows
        // [c0, m); rhs = columns [c0, c0+tw).
        crate::dense::gemm_backend::gemm(
            mrows,
            tw,
            k,
            tmp.as_mut_ptr().add(c0 * m + c0),
            m as isize,
            1,
            false,
            lhs.add(c0),
            lhs_cs,
            1,
            rhs.add(c0),
            1,
            rhs_rs,
            T::zero(),
            T::one(),
            false,
            false,
            false,
            crate::dense::gemm_backend::GemmMode::new(par, ks),
        );
        c0 += tw;
    }
}
