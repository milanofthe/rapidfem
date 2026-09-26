//! Arithmetic shared by the Krylov solvers: the bilinear and sesquilinear
//! products, Givens rotations, the least-squares rank guard and the batched
//! Gram-Schmidt passes of the block solver.

use crate::numeric::settings::Threads;
use crate::scalar::Scalar;
use rayon::prelude::*;

/// Unconjugated bilinear inner product `x^T y = sum x_i y_i` (no complex conjugation -
/// the defining choice of the COCG geometry for `A = A^T`).
#[inline]
pub(super) fn dotu<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + xi * yi;
    }
    s
}

/// Euclidean norm `||x||_2 = sqrt(sum |x_i|^2)` (genuine modulus - used only for the stopping
/// test, never inside the Krylov recurrences).
#[inline]
pub(super) fn norm2<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.magnitude_sq()).sum::<f64>().sqrt()
}

/// Conjugated (Hermitian) inner product `<x, y> = sum conj(x_i)*y_i` - the geometry
/// GMRES orthogonalises in (distinct from COCG's unconjugated form).
#[inline]
pub(super) fn dotc<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (&xi, &yi) in x.iter().zip(y) {
        s = s + xi.conj() * yi;
    }
    s
}

/// Complex Givens rotation `(c, s)` that zeroes `g` against `f`: with the
/// rotation `[[conj(c), conj(s)], [-s, c]]`, `conj(c)*f + conj(s)*g = r` (real)
/// and `-s*f + c*g = 0`, `|c|^2+|s|^2 = 1`.
#[inline]
pub(super) fn givens<T: Scalar>(f: T, g: T) -> (T, T) {
    if g == T::zero() {
        return (T::one(), T::zero());
    }
    if f == T::zero() {
        return (T::zero(), T::one());
    }
    let r = (f.magnitude_sq() + g.magnitude_sq()).sqrt();
    let inv = T::from_real(1.0 / r);
    (f * inv, g * inv)
}

/// Largest leading dimension `d <= jdim` whose Hessenberg diagonals are all above a
/// relative breakdown threshold `eps * max_i |h[i][i]|`. The upper-triangular
/// solve for `y` back-substitutes with `1/h[i][i]`; after Givens the diagonal is
/// `sqrt(|f|^2+|g|^2)` and normally nonzero, but under exact stagnation or a rank-
/// deficient Hessenberg (hard / indefinite / singular operators) some `h[i][i]`
/// can be `0`, so an unguarded `recip()` would emit `Inf`/`NaN` into `x` and the
/// residual. Truncating the solve to this well-conditioned prefix instead yields a
/// **deterministic breakdown**: the leading Krylov block is solved, the degenerate
/// tail is dropped, and the outer true-residual check reports non-convergence. In
/// the well-conditioned case every diagonal clears the threshold and `jdim` is
/// returned unchanged, so the normal path is unaffected.
#[inline]
#[allow(clippy::needless_range_loop)]
pub(super) fn well_conditioned_dim<T: Scalar>(h: &[Vec<T>], jdim: usize) -> usize {
    if jdim == 0 {
        return 0;
    }
    let mut hmax = 0.0f64;
    for i in 0..jdim {
        hmax = hmax.max(h[i][i].magnitude());
    }
    let thresh = f64::EPSILON * hmax;
    for i in 0..jdim {
        if h[i][i].magnitude() <= thresh {
            return i;
        }
    }
    jdim
}

/// [`well_conditioned_dim`] for a **flat** row-major Hessenberg buffer:
/// the single-RHS [`gmres`] stores `H` as one `(m+1)xm` `Vec<T>` (diagonal entry
/// `i` at `h[i*stride + i]`) rather than a `Vec<Vec<T>>`, so this reads the same
/// breakdown-guarded leading dimension off the flat layout. Identical logic.
#[inline]
pub(super) fn well_conditioned_dim_flat<T: Scalar>(h: &[T], stride: usize, jdim: usize) -> usize {
    if jdim == 0 {
        return 0;
    }
    let mut hmax = 0.0f64;
    for i in 0..jdim {
        hmax = hmax.max(h[i * stride + i].magnitude());
    }
    let thresh = f64::EPSILON * hmax;
    for i in 0..jdim {
        if h[i * stride + i].magnitude() <= thresh {
            return i;
        }
    }
    jdim
}

/// The scoped rayon pool for the block-GMRES orthogonalization reductions,
/// from the preconditioner's [`Threads`] policy. `Ambient`
/// returns `None` - the reductions then run on the caller's current pool (the
/// solver-in-the-loop path, where the caller has already installed one bounded
/// rayon pool). Any concrete policy builds a
/// pool of that width **once** per solve; the four `block_project` / `block_subtract`
/// calls per step reuse it (cheap `install`), so factor and solve share one
/// concurrency budget instead of the solve fanning out over the global pool. The
/// chunk-order reduction fold is thread-count independent, so the pool never
/// perturbs the bit-identical-across-thread-counts guarantee.
pub(super) fn solve_thread_pool(policy: Threads) -> Option<rayon::ThreadPool> {
    match policy {
        Threads::Ambient => None,
        // `Fixed(k)` (the resolved factor budget): a `k`-worker pool (`0` = all
        // cores). `Auto` never reaches here - factors resolve it to `Fixed` up
        // front - but the `|cap| cap` fallback keeps this total.
        p => {
            let workers = p.resolve(|cap| cap);
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .ok()
        }
    }
}

/// Run an orthogonalization reduction `f` in the solve pool if one was built,
/// else on the current pool. Confined to closures capturing only Krylov *data*
/// (basis / panel slices) - never the operator or preconditioner - so it never
/// imposes a `Send`/`Sync` bound on the matrix-free (`FnOperator`/`FnPreconditioner`) call path.
#[inline]
pub(super) fn ortho_in_pool<R: Send>(
    pool: &Option<rayon::ThreadPool>,
    f: impl FnOnce() -> R + Send,
) -> R {
    match pool {
        Some(p) => p.install(f),
        None => f(),
    }
}

/// Column-wise projection of a panel `W` (`nxsa`) onto each column's **own**
/// Arnoldi basis: `proj[i*sa + ap] = <V_i[:,ap], W[:,ap]>` for block `i` in
/// `0..blocks` and active column `ap` in `0..sa`. The basis is blocks-major -
/// block `i` is the contiguous slice `vbas[i*sa*n .. (i+1)*sa*n]`, its column `ap`
/// at offset `+ap*n`; `W` column `ap` is `w[ap*n .. ap*n+n]`.
///
/// This is the classical (block) Gram-Schmidt projection: **all** projections are
/// taken against the same `W`, so the `blocks*sa` inner products are independent
/// and computed as one panel sweep instead of the `O(blocks*sa)` sequential,
/// latency-bound BLAS-1 reductions of modified Gram-Schmidt. The reduction is a
/// fixed row-chunk sum folded in chunk order -> deterministic regardless of the
/// thread count.
/// `scratch` is a caller-owned reduction buffer of length `>= nchunks * width`
/// (`nchunks = ceil(n / chunk)`, `width = blocks*sa`), reused across steps so the
/// hot loop allocates nothing. Each chunk writes its `width` partial sums into its
/// own slice; the slices are then folded in chunk order.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn block_project<T: Scalar>(
    vbas: &[T],
    w: &[T],
    blocks: usize,
    sa: usize,
    n: usize,
    chunk: usize,
    proj: &mut [T],
    scratch: &mut [T],
) {
    let width = blocks * sa;
    for p in proj[..width].iter_mut() {
        *p = T::zero();
    }
    if width == 0 || n == 0 {
        return;
    }
    let nchunks = n.div_ceil(chunk);
    let part = &mut scratch[..nchunks * width];
    part.par_chunks_mut(width)
        .enumerate()
        .for_each(|(ci, out)| {
            let r0 = ci * chunk;
            let r1 = (r0 + chunk).min(n);
            for i in 0..blocks {
                for ap in 0..sa {
                    let vb = (i * sa + ap) * n;
                    let wb = ap * n;
                    let mut sdot = T::zero();
                    for k in r0..r1 {
                        sdot = sdot + vbas[vb + k].conj() * w[wb + k];
                    }
                    out[i * sa + ap] = sdot;
                }
            }
        });
    // Fold partials in chunk order: the summation order is fixed by the chunk
    // layout, so the result does not depend on how many threads ran.
    for ci in 0..nchunks {
        let base = ci * width;
        for t in 0..width {
            proj[t] = proj[t] + part[base + t];
        }
    }
}

/// Subtract the projected components from the panel in place, per column:
/// `W[:,ap] -= sum_i proj[i*sa+ap] * V_i[:,ap]`, accumulated in block order (`i`
/// ascending) at every element. Parallel over fixed row-chunks within each column
/// -> the element-wise order is fixed, so the update is deterministic.
#[inline]
pub(super) fn block_subtract<T: Scalar>(
    vbas: &[T],
    w: &mut [T],
    blocks: usize,
    sa: usize,
    n: usize,
    chunk: usize,
    proj: &[T],
) {
    for ap in 0..sa {
        let wcol = &mut w[ap * n..ap * n + n];
        wcol.par_chunks_mut(chunk).enumerate().for_each(|(ci, wc)| {
            let r0 = ci * chunk;
            for i in 0..blocks {
                let hij = proj[i * sa + ap];
                if hij == T::zero() {
                    continue;
                }
                let vb = (i * sa + ap) * n + r0;
                for k in 0..wc.len() {
                    wc[k] = wc[k] - hij * vbas[vb + k];
                }
            }
        });
    }
}
