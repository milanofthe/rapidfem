//! GCRO-DR: restarted GMRES that carries a recycled subspace across solves.

use super::util::*;
use super::*;

use crate::error::RslabError;
use crate::scalar::Scalar;
use num_complex::Complex;

// ===========================================================================
// GCRO-DR: Krylov subspace recycling for sequences of related solves
// ===========================================================================
//
// **When it helps.** RSLAB targets solver-in-the-loop workloads: many solves of
// a slowly varying `A` / `M` / `b`. On *stagnating* systems (a cluster of small
// eigenvalues that restarted GMRES keeps re-discovering and discarding), plain
// FGMRES(`m`) throws away the near-invariant subspace at every restart. GCRO-DR
// (Parks, de Sturler, Mackey, Johnson, Maiti, SISC 2006) keeps `k` harmonic-Ritz
// vectors `U` approximating that subspace and (a) *deflates* it within a solve -
// carrying `U` across restarts, which cuts restart counts on a single hard solve -
// and (b) *recycles* it across solves through an opaque [`Recycle`] handle, so the
// next related system starts already knowing where the trouble is.
//
// **Scope.** Recycling is implemented for the single-RHS FGMRES [`gmres`] path
// ([`gmres_recycled`]). The block path is out of scope: its within-cycle deflation
// + basis compaction remap the per-column Arnoldi state mid-cycle, which does not
// compose cleanly with a shared recycle subspace, and warrants its own design pass.
//
// **The split (right-preconditioned / flexible).** The Krylov space is built on
// the preconditioned operator `A_tilde = A M^-1` (FGMRES stores `Z = M^-1 V`, updates
// `x += Z y`), so `U` lives in the *preconditioned* space (approximate smallest
// eigenvectors of `A_tilde`). Each cycle recomputes `P = M^-1 U` (`k` preconditioner
// solves) and `C = A P = A_tilde U` (`k` matvecs), orthonormalizes `C = Q R` (updating
// `P, U <- *R^-1` so `C = A_tilde U` and `P = M^-1 U` still hold), then runs the GCRO
// split: project the residual onto `range(C)` (`x += P C^H r`, `r -= C C^H r`)
// and build the rest of the Krylov space **orthogonal to `C`** (each Arnoldi
// vector has its `C` component removed, recorded in `B = C^H A Z`). The restart
// least-squares problem then decouples: `y` is the ordinary GMRES solution of the
// Hessenberg system and the recycle coordinate is `z_1 = -B y`, giving
// `x += P z_1 + Z y`. Recomputing `C` each solve (the default) is `k` matvecs +
// `k` M-solves and keeps the invariant exact when `A` / `M` change between solves.
//
// **Memory.** `U` and `C` are each `n*k` extra scalars (plus the same-size `P`),
// on top of the FGMRES `V`+`Z` bases. `k` defaults modest and is capped at
// `restart/2` so the deflation never starves the Arnoldi space.
//
// **Correctness is independent of `U`'s quality.** `U` only shapes the Krylov
// space to *accelerate* convergence; the outer true-residual stop test is
// authoritative, so an inaccurate / rank-deficient recycle subspace can only make
// a solve slower, never wrong. Every recycle step is accordingly guarded (rank
// dropping, singular-projection fallbacks) and degrades gracefully to plain FGMRES.

/// Sealing for [`RecycleScalar`]: only the two fields RSLAB's Krylov solvers run
/// in (`f64`, `Complex<f64>`) may carry a recycle subspace, so downstream crates
/// cannot add ill-defined implementations.
mod recycle_sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for num_complex::Complex<f64> {}
    impl Sealed for num_complex::Complex<f32> {}
}

/// Scalar fields that support GCRO-DR recycling. The harmonic-Ritz small
/// eigenproblem is always solved in `Complex<f64>` (see the `dense_eig`
/// internals); this trait provides the two field-specific
/// bridges: promoting a scalar to `Complex<f64>` for the small matrices, and
/// reconstructing the recycle vectors from complex coefficient vectors - which
/// for the **real** field must split a complex conjugate Ritz pair into its real
/// and imaginary parts (two real recycle vectors), since a real `U` cannot store
/// a complex eigenvector directly. A sealed trait (impls for the four scalar
/// fields `f64` / `f32` / `Complex<f64>` / `Complex<f32>`); not user-implementable.
pub trait RecycleScalar: Scalar + recycle_sealed::Sealed {
    /// Promote to `Complex<f64>` for the small dense harmonic-Ritz problem.
    #[doc(hidden)]
    fn to_c(self) -> Complex<f64>;
    /// Reconstruct up to `kmax` recycle vectors (column-major, length `n` each)
    /// from the search-space columns `cols` (`n x d`) and the harmonic-Ritz
    /// `(theta, g)` pairs (coefficient vectors of length `d`, sorted by ascending
    /// `|theta|`). Real fields split each complex conjugate pair into `Re g`, `Im g`.
    #[doc(hidden)]
    fn combine_ritz(
        cols: &[Self],
        n: usize,
        d: usize,
        pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
        kmax: usize,
    ) -> Vec<Self>;
}

/// Real-field recycle reconstruction (`f64` / `f32`): a real `U` cannot hold a
/// complex eigenvector, so each **complex** conjugate Ritz pair contributes its
/// `Re g` and `Im g` (two real vectors spanning the real 2-D invariant subspace),
/// while a real Ritz value contributes one. Multiplies columns only by the real
/// coefficient parts via [`Scalar::from_real`], so it is generic over the real
/// field. Emits at most `kmax` vectors.
fn combine_ritz_real<T: Scalar>(
    cols: &[T],
    n: usize,
    d: usize,
    pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
    kmax: usize,
) -> Vec<T> {
    let mut out: Vec<T> = Vec::new();
    for (theta, g) in pairs {
        if out.len() / n >= kmax {
            break;
        }
        let mag = theta.norm().max(1e-300);
        if theta.im.abs() <= 1e-8 * mag {
            let mut v = vec![T::zero(); n];
            for j in 0..d {
                let re = T::from_real(g[j].re);
                for kk in 0..n {
                    v[kk] = v[kk] + re * cols[j * n + kk];
                }
            }
            out.extend_from_slice(&v);
        } else if theta.im > 0.0 {
            let mut vr = vec![T::zero(); n];
            let mut vi = vec![T::zero(); n];
            for j in 0..d {
                let (re, im) = (T::from_real(g[j].re), T::from_real(g[j].im));
                for kk in 0..n {
                    vr[kk] = vr[kk] + re * cols[j * n + kk];
                    vi[kk] = vi[kk] + im * cols[j * n + kk];
                }
            }
            out.extend_from_slice(&vr);
            if out.len() / n < kmax {
                out.extend_from_slice(&vi);
            }
        }
    }
    out
}

/// Complex-field recycle reconstruction (`Complex<f64>` / `Complex<f32>`): the
/// harmonic-Ritz vectors are genuinely complex, so `U <- [U, V] g_i` directly, one
/// vector per pair. `mk` casts a `Complex<f64>` coefficient into the working field
/// (identity for `Complex<f64>`, a narrowing cast for `Complex<f32>`).
fn combine_ritz_complex<T: Scalar>(
    cols: &[T],
    n: usize,
    d: usize,
    pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
    kmax: usize,
    mk: impl Fn(Complex<f64>) -> T,
) -> Vec<T> {
    let mut out: Vec<T> = Vec::new();
    for (i, (_theta, g)) in pairs.iter().enumerate() {
        if i >= kmax {
            break;
        }
        let mut v = vec![T::zero(); n];
        for j in 0..d {
            let gj = mk(g[j]);
            for kk in 0..n {
                v[kk] = v[kk] + gj * cols[j * n + kk];
            }
        }
        out.extend_from_slice(&v);
    }
    out
}

impl RecycleScalar for f64 {
    fn to_c(self) -> Complex<f64> {
        Complex::new(self, 0.0)
    }
    fn combine_ritz(
        cols: &[f64],
        n: usize,
        d: usize,
        pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
        kmax: usize,
    ) -> Vec<f64> {
        combine_ritz_real(cols, n, d, pairs, kmax)
    }
}

impl RecycleScalar for f32 {
    fn to_c(self) -> Complex<f64> {
        Complex::new(self as f64, 0.0)
    }
    fn combine_ritz(
        cols: &[f32],
        n: usize,
        d: usize,
        pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
        kmax: usize,
    ) -> Vec<f32> {
        combine_ritz_real(cols, n, d, pairs, kmax)
    }
}

impl RecycleScalar for Complex<f64> {
    fn to_c(self) -> Complex<f64> {
        self
    }
    fn combine_ritz(
        cols: &[Complex<f64>],
        n: usize,
        d: usize,
        pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
        kmax: usize,
    ) -> Vec<Complex<f64>> {
        combine_ritz_complex(cols, n, d, pairs, kmax, |z| z)
    }
}

impl RecycleScalar for Complex<f32> {
    fn to_c(self) -> Complex<f64> {
        Complex::new(self.re as f64, self.im as f64)
    }
    fn combine_ritz(
        cols: &[Complex<f32>],
        n: usize,
        d: usize,
        pairs: &[(Complex<f64>, Vec<Complex<f64>>)],
        kmax: usize,
    ) -> Vec<Complex<f32>> {
        combine_ritz_complex(cols, n, d, pairs, kmax, |z| {
            Complex::new(z.re as f32, z.im as f32)
        })
    }
}

/// An opaque **recycle subspace** carried across a sequence of related solves
/// ([`gmres_recycled`]). Holds `k` harmonic-Ritz vectors `U` (in the
/// preconditioned space, i.e. approximate smallest eigenvectors of `A M^-1`) that
/// dominate GMRES stagnation, so the next related solve deflates them from the
/// start instead of re-discovering them.
///
/// **Lifetime / validity.** Create once with [`Recycle::new`] (target dimension
/// `k`) and pass `&mut` to each solve; the handle is *updated in place* at the
/// end of every solve with the freshest harmonic-Ritz vectors. It stays useful
/// across solves with the **same or slowly varying** `A` and `M`: the invariant
/// subspace it tracks drifts slowly, and `C = A M^-1 U` is recomputed from `U`
/// every solve (`k` matvecs + `k` M-solves), so a changed operator is handled
/// exactly with no stale `C`. If the dimension `n` changes the handle resets
/// itself. Reusing it on an *unrelated* system is safe (never wrong) but may not
/// help. Call [`Recycle::clear`] to forget the accumulated subspace.
///
/// **Memory:** `U` is `n*k` scalars; the transient `P = M^-1 U` and `C = A M^-1 U`
/// (another `2*n*k`) live only for the duration of a solve.
#[derive(Debug, Clone)]
pub struct Recycle<T> {
    /// Target subspace dimension (capped at `restart/2` per solve).
    k: usize,
    /// Recycle vectors, column-major `n x kc` (`kc <= k`), in preconditioned space.
    u: Vec<T>,
    /// Current number of stored recycle vectors.
    kc: usize,
    /// Dimension the stored vectors belong to (`0` when empty).
    n: usize,
}

impl<T: Scalar> Recycle<T> {
    /// The stored recycle vectors: column-major `n x kc` (`kc` columns of length `n`),
    /// in the preconditioned space — the approximate smallest-magnitude eigenvectors of
    /// `A M^-1` the last solve found. For inspection (which physical directions a solver
    /// keeps re-discovering); empty before the first solve.
    pub fn vectors(&self) -> (&[T], usize, usize) {
        (&self.u, self.n, self.kc)
    }

    /// A fresh, empty recycle handle targeting `k` harmonic-Ritz vectors. The
    /// first solve seeds `U` (there is nothing to deflate yet); subsequent solves
    /// deflate and refresh it. `k` is capped at `restart/2` inside the solve.
    pub fn new(k: usize) -> Self {
        Recycle {
            k,
            u: Vec::new(),
            kc: 0,
            n: 0,
        }
    }

    /// The target subspace dimension `k`.
    pub fn dim(&self) -> usize {
        self.k
    }

    /// The number of recycle vectors currently stored (`<= k`, `0` until the first
    /// solve populates it or after [`clear`](Self::clear)).
    pub fn active(&self) -> usize {
        self.kc
    }

    /// Forget the accumulated subspace (e.g. before an unrelated system). The
    /// target dimension `k` is retained.
    pub fn clear(&mut self) {
        self.u.clear();
        self.kc = 0;
        self.n = 0;
    }
}

/// Modified-Gram-Schmidt orthonormalization of the recycle triple `(C, P, U)`
/// (each column-major `n x kc`), maintaining the invariants `C = A P` and
/// `P = M^-1 U`: the same linear combination applied to `C`'s columns is applied
/// to `P` and `U`, so after `C <- Q` (orthonormal) the scaled `P, U` still satisfy
/// them. Rank-deficient columns (norm collapses under the projection) are
/// **dropped** and the survivors compacted forward. Returns the surviving count.
fn orthonormalize_recycle<T: Scalar>(
    c: &mut [T],
    p: &mut [T],
    u: &mut [T],
    n: usize,
    kc: usize,
) -> usize {
    let mut w = 0usize;
    for src in 0..kc {
        let mut ccol: Vec<T> = c[src * n..src * n + n].to_vec();
        let mut pcol: Vec<T> = p[src * n..src * n + n].to_vec();
        let mut ucol: Vec<T> = u[src * n..src * n + n].to_vec();
        let orig = norm2(&ccol);
        if orig == 0.0 {
            continue;
        }
        for j in 0..w {
            let r = dotc(&c[j * n..j * n + n], &ccol);
            for kk in 0..n {
                ccol[kk] = ccol[kk] - r * c[j * n + kk];
                pcol[kk] = pcol[kk] - r * p[j * n + kk];
                ucol[kk] = ucol[kk] - r * u[j * n + kk];
            }
        }
        let nrm = norm2(&ccol);
        if nrm <= 1e-12 * orig {
            continue; // rank-deficient direction: drop it
        }
        let inv = T::from_real(1.0 / nrm);
        for kk in 0..n {
            c[w * n + kk] = ccol[kk] * inv;
            p[w * n + kk] = pcol[kk] * inv;
            u[w * n + kk] = ucol[kk] * inv;
        }
        w += 1;
    }
    w
}

/// Modified-Gram-Schmidt orthonormalization of `nc` column-major `n`-vectors in
/// place, dropping rank-deficient columns (the guard for a defective harmonic-Ritz
/// reconstruction, requirement #4). Returns the surviving orthonormal count.
fn orthonormalize_columns<T: Scalar>(v: &mut [T], n: usize, nc: usize) -> usize {
    let mut w = 0usize;
    for src in 0..nc {
        let mut col: Vec<T> = v[src * n..src * n + n].to_vec();
        let orig = norm2(&col);
        if orig == 0.0 {
            continue;
        }
        for j in 0..w {
            let r = dotc(&v[j * n..j * n + n], &col);
            for kk in 0..n {
                col[kk] = col[kk] - r * v[j * n + kk];
            }
        }
        let nrm = norm2(&col);
        if nrm <= 1e-12 * orig {
            continue;
        }
        let inv = T::from_real(1.0 / nrm);
        for kk in 0..n {
            v[w * n + kk] = col[kk] * inv;
        }
        w += 1;
    }
    w
}

/// Recompute the recycle subspace `U` from this cycle's search space by GCRO-DR
/// harmonic-Ritz extraction. Forms the small `(d+1) x d` matrix
/// `G_bar = [[I_k, B], [0, H_bar]]` and the orthonormal image basis `W_hat = [C, V_{p+1}]`
/// (`d = k + p`, `p = jdim` Arnoldi steps), solves the generalized eigenproblem
/// `G_bar^H G_bar g = theta G_bar^H (W_hat^H [U, V]) g` for the `k` smallest `|theta|`, and reconstructs
/// `U <- [U, V] G_k` (orthonormalized, rank-guarded). Returns `(U, count)` or
/// `None` (keep the previous subspace) on a singular projection / empty spectrum.
#[allow(clippy::too_many_arguments)]
fn recompute_recycle<T: RecycleScalar>(
    cmat: &[T],
    u: &[T],
    vb: &[T],
    hbar: &[T],
    bmat: &[T],
    n: usize,
    ncur: usize,
    jdim: usize,
    p_arn: usize,
    k: usize,
) -> Option<(Vec<T>, usize)> {
    let d = ncur + jdim;
    if d == 0 || k == 0 {
        return None;
    }
    let rows = d + 1;
    let c0 = Complex::new(0.0, 0.0);
    let c1 = Complex::new(1.0, 0.0);
    // G_bar (rows x d), row-major, in Complex<f64>.
    let mut ghat = vec![c0; rows * d];
    for i in 0..ncur {
        ghat[i * d + i] = c1; // I_k block
    }
    for i in 0..ncur {
        for j in 0..jdim {
            ghat[i * d + (ncur + j)] = bmat[i * p_arn + j].to_c(); // B block
        }
    }
    for r in 0..(jdim + 1) {
        for j in 0..jdim {
            ghat[(ncur + r) * d + (ncur + j)] = hbar[r * p_arn + j].to_c(); // H_bar block
        }
    }
    // W_hat^H [U, V]  (rows x d): [[C^H U, 0], [V^H U, [I_p; 0]]].
    let mut whatu = vec![c0; rows * d];
    for i in 0..ncur {
        for a in 0..ncur {
            whatu[i * d + a] = dotc(&cmat[i * n..i * n + n], &u[a * n..a * n + n]).to_c();
        }
    }
    for r in 0..(jdim + 1) {
        for a in 0..ncur {
            whatu[(ncur + r) * d + a] = dotc(&vb[r * n..r * n + n], &u[a * n..a * n + n]).to_c();
        }
    }
    for j in 0..jdim {
        whatu[(ncur + j) * d + (ncur + j)] = c1;
    }
    // M1 = G_bar^H G_bar,  M2 = G_bar^H (W_hat^H [U, V]).
    let mut m1 = vec![c0; d * d];
    let mut m2 = vec![c0; d * d];
    for a in 0..d {
        for b in 0..d {
            let mut s1 = c0;
            let mut s2 = c0;
            for r in 0..rows {
                let ga = ghat[r * d + a].conj();
                s1 += ga * ghat[r * d + b];
                s2 += ga * whatu[r * d + b];
            }
            m1[a * d + b] = s1;
            m2[a * d + b] = s2;
        }
    }
    let pairs = crate::numeric::dense_eig::harmonic_ritz_smallest(&m1, &m2, d, k);
    if pairs.is_empty() {
        return None;
    }
    // Search-space columns [U | V_p]  (n x d), column-major.
    let mut space = vec![T::zero(); n * d];
    for a in 0..ncur {
        space[a * n..a * n + n].copy_from_slice(&u[a * n..a * n + n]);
    }
    for j in 0..jdim {
        space[(ncur + j) * n..(ncur + j) * n + n].copy_from_slice(&vb[j * n..j * n + n]);
    }
    let mut newu = T::combine_ritz(&space, n, d, &pairs, k);
    let cnt = newu.len() / n;
    let cnt2 = orthonormalize_columns(&mut newu, n, cnt);
    if cnt2 == 0 {
        return None;
    }
    newu.truncate(cnt2 * n);
    Some((newu, cnt2))
}

/// **GCRO-DR** - flexible right-preconditioned restarted GMRES with **Krylov
/// subspace recycling** (Parks/de Sturler et al. 2006). The recycling
/// companion to [`gmres`]: identical convergence semantics and diagnostics, plus
/// a [`Recycle`] handle that (a) deflates a `k`-dimensional near-invariant
/// subspace *across restarts within this solve* and (b) *carries it to the next
/// solve*. See the module-level "GCRO-DR" section for the algorithm and when it
/// pays (stagnating, restart-limited, or slowly-varying-sequence solves).
///
/// `recycle.dim()` = `k` (capped at `restart/2`). Pass the **same** handle to a
/// sequence of related solves; it is refreshed in place at the end of each. The
/// optional `x0` warm start composes with recycling (seed from the previous
/// solution *and* recycle its stagnation subspace). With an empty handle and
/// `k = 0` this reduces to plain [`gmres`].
#[allow(clippy::too_many_arguments)]
pub fn gmres_recycled<T, A, M>(
    op: &A,
    b: &[T],
    precond: &M,
    tol: f64,
    max_iter: usize,
    restart: usize,
    x0: Option<&[T]>,
    recycle: &mut Recycle<T>,
) -> Result<KrylovResult<T>, RslabError>
where
    T: RecycleScalar,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = op.n();
    if b.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    const REORTH_ETA: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let m = restart.max(1);
    // Cap the recycle dimension at restart/2 so the Arnoldi space (m - k) is never
    // starved. `k = 0` (or an empty handle) degrades to plain FGMRES.
    let k = recycle.k.min(m / 2);
    let bnorm = norm2(b);
    let mut x = match x0 {
        Some(g) => {
            if g.len() != n {
                return Err(RslabError::DimensionMismatch {
                    expected: n,
                    got: g.len(),
                });
            }
            g.to_vec()
        }
        None => vec![T::zero(); n],
    };
    if bnorm == 0.0 {
        x.iter_mut().for_each(|v| *v = T::zero());
        recycle.u.clear();
        recycle.kc = 0;
        recycle.n = 0;
        return Ok(KrylovResult {
            x,
            iters: 0,
            converged: true,
            final_res: 0.0,
            stop: StopReason::Converged,
        });
    }

    // Recycle subspace carried in: reset if the dimension changed.
    if recycle.n != n {
        recycle.u.clear();
        recycle.kc = 0;
    }
    let mut u: Vec<T> = recycle.u.clone();
    let mut ncur = recycle.kc.min(k);
    u.truncate(ncur * n);

    let mut w = vec![T::zero(); n];
    let mut ax = vec![T::zero(); n];
    let mut total = 0usize;
    let mut final_res;

    loop {
        // Top-of-cycle TRUE residual (authoritative stop test).
        op.apply(&x, &mut ax);
        let mut r: Vec<T> = (0..n).map(|i| b[i] - ax[i]).collect();
        let beta0 = norm2(&r);
        final_res = beta0 / bnorm;
        if final_res <= tol || total >= max_iter {
            break;
        }

        // --- Recycle refresh: C = A M^-1 U, orthonormalize, GCRO project. ---
        let mut cmat = vec![T::zero(); n * ncur];
        let mut pmat = vec![T::zero(); n * ncur];
        if ncur > 0 {
            for i in 0..ncur {
                precond.apply(&u[i * n..i * n + n], &mut pmat[i * n..i * n + n])?;
                op.apply(&pmat[i * n..i * n + n], &mut cmat[i * n..i * n + n]);
            }
            ncur = orthonormalize_recycle(&mut cmat, &mut pmat, &mut u, n, ncur);
            // Outer minimization over range(C): x += P (C^H r), r -= C (C^H r).
            for i in 0..ncur {
                let di = dotc(&cmat[i * n..i * n + n], &r);
                for kk in 0..n {
                    x[kk] = x[kk] + pmat[i * n + kk] * di;
                    r[kk] = r[kk] - cmat[i * n + kk] * di;
                }
            }
        }
        let beta = norm2(&r);
        if beta == 0.0 {
            continue; // exact in the recycle space; loop top confirms convergence
        }

        // Arnoldi dimension this cycle (orthogonal to C).
        let p_arn = m - ncur;
        let inv_beta = T::from_real(1.0 / beta);
        let mut vb = vec![T::zero(); n * (p_arn + 1)];
        let mut zb = vec![T::zero(); n * p_arn];
        for kk in 0..n {
            vb[kk] = r[kk] * inv_beta;
        }
        // Raw (unrotated) Hessenberg for harmonic Ritz; rotated copy for the LS.
        let mut hbar = vec![T::zero(); (p_arn + 1) * p_arn];
        let mut rmat = vec![T::zero(); (p_arn + 1) * p_arn];
        let mut bmat = vec![T::zero(); ncur.max(1) * p_arn]; // C-projection coeffs
        let mut cs = vec![T::zero(); p_arn];
        let mut sn = vec![T::zero(); p_arn];
        let mut g = vec![T::zero(); p_arn + 1];
        let mut y = vec![T::zero(); p_arn];
        g[0] = T::from_real(beta);
        let mut jdim = 0usize;
        for j in 0..p_arn {
            if total >= max_iter {
                break;
            }
            precond.apply(&vb[j * n..j * n + n], &mut zb[j * n..j * n + n])?;
            op.apply(&zb[j * n..j * n + n], &mut w);
            let wnorm0 = norm2(&w);
            // Project the new direction orthogonal to C (record B[:,j] = C^H w).
            for i in 0..ncur {
                let bij = dotc(&cmat[i * n..i * n + n], &w);
                bmat[i * p_arn + j] = bij;
                for kk in 0..n {
                    w[kk] = w[kk] - cmat[i * n + kk] * bij;
                }
            }
            // Modified Gram-Schmidt against V_0..V_j.
            for i in 0..=j {
                let hij = dotc(&vb[i * n..i * n + n], &w);
                hbar[i * p_arn + j] = hij;
                for kk in 0..n {
                    w[kk] = w[kk] - vb[i * n + kk] * hij;
                }
            }
            let mut hn = norm2(&w);
            // Conditional DGKS second pass (against C and V).
            if hn < REORTH_ETA * wnorm0 {
                for i in 0..ncur {
                    let s = dotc(&cmat[i * n..i * n + n], &w);
                    bmat[i * p_arn + j] = bmat[i * p_arn + j] + s;
                    for kk in 0..n {
                        w[kk] = w[kk] - cmat[i * n + kk] * s;
                    }
                }
                for i in 0..=j {
                    let s = dotc(&vb[i * n..i * n + n], &w);
                    hbar[i * p_arn + j] = hbar[i * p_arn + j] + s;
                    for kk in 0..n {
                        w[kk] = w[kk] - vb[i * n + kk] * s;
                    }
                }
                hn = norm2(&w);
            }
            hbar[(j + 1) * p_arn + j] = T::from_real(hn);
            if hn > 0.0 {
                let inv = T::from_real(1.0 / hn);
                for kk in 0..n {
                    vb[(j + 1) * n + kk] = w[kk] * inv;
                }
            } else {
                for kk in 0..n {
                    vb[(j + 1) * n + kk] = T::zero();
                }
            }
            // Copy the raw column into the rotated buffer and apply Givens for LS.
            for i in 0..=(j + 1) {
                rmat[i * p_arn + j] = hbar[i * p_arn + j];
            }
            for i in 0..j {
                let temp =
                    cs[i].conj() * rmat[i * p_arn + j] + sn[i].conj() * rmat[(i + 1) * p_arn + j];
                rmat[(i + 1) * p_arn + j] =
                    -sn[i] * rmat[i * p_arn + j] + cs[i] * rmat[(i + 1) * p_arn + j];
                rmat[i * p_arn + j] = temp;
            }
            let (c, s) = givens(rmat[j * p_arn + j], rmat[(j + 1) * p_arn + j]);
            cs[j] = c;
            sn[j] = s;
            rmat[j * p_arn + j] =
                c.conj() * rmat[j * p_arn + j] + s.conj() * rmat[(j + 1) * p_arn + j];
            rmat[(j + 1) * p_arn + j] = T::zero();
            let g_next = -s * g[j];
            g[j] = c.conj() * g[j];
            g[j + 1] = g_next;
            total += 1;
            jdim = j + 1;
            if g[j + 1].magnitude() / bnorm <= tol {
                break;
            }
        }

        // Least-squares solve on the well-conditioned leading block.
        let jd = well_conditioned_dim_flat(&rmat, p_arn, jdim);
        for i in (0..jd).rev() {
            let mut s = g[i];
            for kk in (i + 1)..jd {
                s = s - rmat[i * p_arn + kk] * y[kk];
            }
            y[i] = s * rmat[i * p_arn + i].recip();
        }
        // x += Z y   (FGMRES: preconditioned basis, no extra M-solve).
        for i in 0..jd {
            let yi = y[i];
            for kk in 0..n {
                x[kk] = x[kk] + zb[i * n + kk] * yi;
            }
        }
        // x += P z_1 with z_1 = -B y   (the recycle-coordinate update).
        for i in 0..ncur {
            let mut z1i = T::zero();
            for j in 0..jd {
                z1i = z1i - bmat[i * p_arn + j] * y[j];
            }
            for kk in 0..n {
                x[kk] = x[kk] + pmat[i * n + kk] * z1i;
            }
        }

        // --- Refresh the recycle subspace from this cycle's harmonic Ritz. ---
        if k > 0 && jdim > 0 {
            if let Some((newu, cnt)) =
                recompute_recycle(&cmat, &u, &vb, &hbar, &bmat, n, ncur, jdim, p_arn, k)
            {
                u = newu;
                ncur = cnt;
            }
        }
    }

    // Persist the freshest recycle subspace for the next related solve.
    recycle.u = u;
    recycle.kc = ncur;
    recycle.n = n;

    Ok(KrylovResult {
        x,
        iters: total,
        converged: final_res <= tol,
        final_res,
        // GMRES-family: converged (residual met, incl. happy breakdown) or the
        // iteration budget ran out. No non-converged breakdown state.
        stop: if final_res <= tol {
            StopReason::Converged
        } else {
            StopReason::MaxIter
        },
    })
}
