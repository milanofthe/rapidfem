//! Generic dense Bunch-Kaufman LDLᵀ factorization over any [`Scalar`] field.
//!
//! This is a clean, unblocked, correctness-first implementation of the
//! symmetric-indefinite factorization `Pᵀ A P = L D Lᵀ`, where `L` is unit
//! lower triangular and `D` is block diagonal with 1×1 and 2×2 blocks. It is
//! generic over `T: Scalar`, so it serves both the real (`f64`) and the
//! complex-*symmetric* (`Complex<f64>`, PARDISO `mtype 6`) paths.
//!
//! The algorithm is the lower-triangular, right-looking Bunch-Kaufman scheme of
//! LAPACK's `?sytf2` (`xSYTF2`). For the complex-symmetric case this is exactly
//! `zsytf2`: identical control flow to the real `dsytf2`, with magnitudes taken
//! as the complex modulus `|z|` and **no conjugation** anywhere (the matrix is
//! symmetric `A = Aᵀ`, not Hermitian). The pivot threshold is the classical
//! `α = (1 + √17)/8`.
//!
//! This is the shared, data-type-generic dense kernel that every multifrontal
//! front reduces to (the former f64-dedicated dense path, with its blocked SIMD
//! Schur kernel and rook rescue, has been removed in favour of this single
//! generic path). Further performance work happens here.

use crate::dense::matrix::SymmetricMatrix;
use crate::error::RslabError;
use crate::scalar::Scalar;
use rayon::prelude::*;

/// Result of a generic Bunch-Kaufman LDLᵀ factorization.
///
/// `Pᵀ A P = L D Lᵀ`. The permutation is symmetric (the same `P` acts on rows
/// and columns), so factoring preserves symmetry.
#[derive(Debug, Clone)]
pub struct LdltFactors<T> {
    pub n: usize,
    /// Unit lower triangular `L` in CSC (compressed sparse column). Column `j`
    /// is `l_row_idx[l_col_ptr[j]..l_col_ptr[j+1]]` with matching `l_values`,
    /// sorted by row, and includes the explicit unit diagonal `(j, 1)`. For a
    /// 2×2 pivot the intra-block entry `L[k+1][k]` is `0` (that coupling lives
    /// in `D`, not `L`). Storing `L` sparsely keeps the factor `O(nnz(L))`
    /// rather than `O(n²)`.
    pub l_col_ptr: Vec<usize>,
    pub l_row_idx: Vec<usize>,
    pub l_values: Vec<T>,
    /// Diagonal of the block-diagonal `D`, length `n`.
    pub d_diag: Vec<T>,
    /// Sub-diagonal of `D`, length `n`. `d_subdiag[k]` is the `(k+1, k)` entry
    /// of a 2×2 block starting at column `k`; it is `0` for 1×1 pivots and for
    /// the second column of a 2×2 block.
    pub d_subdiag: Vec<T>,
    /// `true` at the starting column of each 2×2 pivot block. The column after
    /// such a start is the block's second column; every other column is a 1×1
    /// pivot.
    pub two_by_two: Vec<bool>,
    /// Symmetric pivot permutation (forward): `perm[i] = j` means original
    /// index `j` occupies pivot position `i`.
    pub perm: Vec<usize>,
    /// Number of pivots that were statically perturbed (replaced by a floor)
    /// to avoid a singular/tiny pivot. Zero for an exact factorization;
    /// nonzero only when static-pivoting (preconditioner) mode is enabled. The
    /// factor then reconstructs `A + E` for a small `E`, which is exactly what
    /// a preconditioner wants.
    pub n_perturbed: usize,
    /// Inertia (counts of positive/negative/zero `D` pivots). Exact for a real
    /// symmetric matrix (`T = f64`/`f32`); for a complex-symmetric matrix the
    /// eigenvalues are complex and have no sign - there it is advisory only
    /// (classified by each pivot's real part).
    pub inertia: crate::inertia::Inertia,
}

/// The Bunch-Kaufman pivot threshold `α = (1 + √17)/8 ≈ 0.6404`.
#[inline]
pub(crate) fn bk_alpha() -> f64 {
    (1.0 + 17.0_f64.sqrt()) / 8.0
}

/// Swap symmetric indices `p` and `q` (`p < q`) in a lower-triangle,
/// column-major working matrix. This swaps the corresponding rows *and*
/// columns across the whole matrix - including the already-computed `L`
/// columns to the left - so the partial factorization stays consistent. The
/// crossing element `(q, p)` maps to itself and is left in place.
pub(crate) fn swap_sym_lower<T: Scalar>(a: &mut [T], n: usize, p: usize, q: usize) {
    debug_assert!(p < q && q < n);
    // Column segment strictly below q: (i, p) <-> (i, q) for i > q.
    for i in (q + 1)..n {
        a.swap(p * n + i, q * n + i);
    }
    // Middle cross strip: (i, p) <-> (q, i) for p < i < q.
    for i in (p + 1)..q {
        a.swap(p * n + i, i * n + q);
    }
    // Diagonal: (p, p) <-> (q, q).
    a.swap(p * n + p, q * n + q);
    // Left row segments: (p, j) <-> (q, j) for j < p.
    for j in 0..p {
        a.swap(j * n + p, j * n + q);
    }
}

/// Factor a symmetric matrix `A` as `Pᵀ A P = L D Lᵀ` using unblocked
/// Bunch-Kaufman pivoting.
///
/// Works for any [`Scalar`]; for `T = Complex<f64>` this is the
/// complex-symmetric (`A = Aᵀ`) factorization. Returns
/// [`RslabError::NumericallyRankDeficient`] if a structurally zero pivot (1×1
/// of value 0, or a 2×2 block with zero determinant) is encountered.
pub fn factor_ldlt<T: Scalar>(matrix: &SymmetricMatrix<T>) -> Result<LdltFactors<T>, RslabError> {
    matrix.validate()?;
    let n = matrix.n;
    let alpha = bk_alpha();

    // Working copy; only the lower triangle (i >= j) is read/written.
    let mut a = matrix.data.clone();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut d_diag = vec![T::zero(); n];
    let mut d_subdiag = vec![T::zero(); n];
    let mut two_by_two = vec![false; n];
    let mut inertia = crate::inertia::Inertia::new(0, 0, 0);

    let mut k = 0;
    while k < n {
        let absakk = a[k * n + k].magnitude();

        // colmax = largest |A[i][k]| below the diagonal, at row imax.
        let mut colmax = 0.0;
        let mut imax = k;
        for i in (k + 1)..n {
            let m = a[k * n + i].magnitude();
            if m > colmax {
                colmax = m;
                imax = i;
            }
        }

        // Decide pivot size (kstep) and which index to interchange with (kp).
        let kstep;
        let kp;
        if absakk.max(colmax) == 0.0 {
            // Structurally zero column: singular.
            return Err(RslabError::NumericallyRankDeficient);
        } else if absakk >= alpha * colmax {
            kstep = 1;
            kp = k;
        } else {
            // rowmax = largest off-diagonal magnitude in row imax.
            let mut rowmax = 0.0;
            for j in k..imax {
                let m = a[j * n + imax].magnitude(); // A[imax][j], imax > j
                if m > rowmax {
                    rowmax = m;
                }
            }
            for i in (imax + 1)..n {
                let m = a[imax * n + i].magnitude(); // A[i][imax]
                if m > rowmax {
                    rowmax = m;
                }
            }

            if absakk >= alpha * colmax * (colmax / rowmax) {
                kstep = 1;
                kp = k;
            } else if a[imax * n + imax].magnitude() >= alpha * rowmax {
                kstep = 1;
                kp = imax;
            } else {
                kstep = 2;
                kp = imax;
            }
        }

        if kstep == 1 {
            // 1×1 pivot. Interchange index k with kp if needed.
            if kp != k {
                swap_sym_lower(&mut a, n, k, kp);
                perm.swap(k, kp);
            }
            let d = a[k * n + k];
            if d == T::zero() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            d_diag[k] = d;
            let r = d.real();
            if r > 0.0 {
                inertia.positive += 1;
            } else if r < 0.0 {
                inertia.negative += 1;
            } else {
                inertia.zero += 1;
            }
            let dinv = d.recip();

            // Rank-1 trailing update using the original pivot column, then
            // overwrite the column with the multipliers L[i][k] = A[i][k]/d.
            for j in (k + 1)..n {
                let wj_dinv = a[k * n + j] * dinv; // A[j][k] / d
                if wj_dinv != T::zero() {
                    for i in j..n {
                        a[j * n + i] = a[j * n + i] - a[k * n + i] * wj_dinv;
                    }
                }
            }
            for i in (k + 1)..n {
                a[k * n + i] = a[k * n + i] * dinv;
            }
            k += 1;
        } else {
            // 2×2 pivot at (k, k+1). Interchange index k+1 with kp if needed.
            if kp != k + 1 {
                swap_sym_lower(&mut a, n, k + 1, kp);
                perm.swap(k + 1, kp);
            }
            let d11 = a[k * n + k];
            let d21 = a[k * n + (k + 1)]; // A[k+1][k]
            let d22 = a[(k + 1) * n + (k + 1)];
            let det = d11 * d22 - d21 * d21;
            if det == T::zero() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            let detinv = det.recip();
            d_diag[k] = d11;
            d_subdiag[k] = d21;
            d_diag[k + 1] = d22;
            two_by_two[k] = true;
            let det_r = det.real();
            let tr_r = (d11 + d22).real();
            if det_r < 0.0 {
                inertia.positive += 1;
                inertia.negative += 1;
            } else if det_r > 0.0 {
                if tr_r >= 0.0 {
                    inertia.positive += 2;
                } else {
                    inertia.negative += 2;
                }
            } else {
                inertia.zero += 1;
                if tr_r >= 0.0 {
                    inertia.positive += 1;
                } else {
                    inertia.negative += 1;
                }
            }

            // Multiplier columns L_i = D⁻¹ · [A[i][k], A[i][k+1]]ᵀ for i >= k+2,
            // with D⁻¹ = (1/det)·[[d22, -d21], [-d21, d11]].
            let mut l1 = vec![T::zero(); n];
            let mut l2 = vec![T::zero(); n];
            for i in (k + 2)..n {
                let wik = a[k * n + i];
                let wik1 = a[(k + 1) * n + i];
                l1[i] = (d22 * wik - d21 * wik1) * detinv;
                l2[i] = (d11 * wik1 - d21 * wik) * detinv;
            }
            // Trailing update A22[i][j] -= W1_i·l1_j + W2_i·l2_j, reading the
            // original pivot columns (still intact) before overwriting them.
            for j in (k + 2)..n {
                let l1j = l1[j];
                let l2j = l2[j];
                for i in j..n {
                    a[j * n + i] = a[j * n + i] - a[k * n + i] * l1j - a[(k + 1) * n + i] * l2j;
                }
            }
            for i in (k + 2)..n {
                a[k * n + i] = l1[i];
                a[(k + 1) * n + i] = l2[i];
            }
            k += 2;
        }
    }

    // Extract L into CSC, honoring block structure. Columns are emitted in
    // increasing order; within a column the explicit unit diagonal comes first,
    // then the strictly-lower multipliers in ascending row order.
    let one = T::one();
    let mut l_col_ptr = Vec::with_capacity(n + 1);
    l_col_ptr.push(0);
    let mut l_row_idx: Vec<usize> = Vec::new();
    let mut l_values: Vec<T> = Vec::new();
    let mut push_col = |col: usize, src_off: usize, start_row: usize| {
        l_row_idx.push(col);
        l_values.push(one);
        for i in start_row..n {
            let v = a[src_off + i];
            if v != T::zero() {
                l_row_idx.push(i);
                l_values.push(v);
            }
        }
        l_col_ptr.push(l_row_idx.len());
    };
    let mut c = 0;
    while c < n {
        if two_by_two[c] {
            // L[c+1][c] is omitted (intra-block coupling lives in D); both
            // columns' multipliers start at row c+2.
            push_col(c, c * n, c + 2);
            push_col(c + 1, (c + 1) * n, c + 2);
            c += 2;
        } else {
            push_col(c, c * n, c + 1);
            c += 1;
        }
    }

    Ok(LdltFactors {
        n,
        l_col_ptr,
        l_row_idx,
        l_values,
        d_diag,
        d_subdiag,
        two_by_two,
        perm,
        n_perturbed: 0,
        inertia,
    })
}

/// Solve `A · x = rhs` from a generic LDLᵀ factorization.
///
/// Applies the five-step sequence `x = P L⁻ᵀ D⁻¹ L⁻¹ Pᵀ rhs`.
pub fn solve_ldlt<T: Scalar>(factors: &LdltFactors<T>, rhs: &[T]) -> Result<Vec<T>, RslabError> {
    let n = factors.n;
    if rhs.len() != n {
        return Err(RslabError::DimensionMismatch {
            expected: n,
            got: rhs.len(),
        });
    }
    // y = Pᵀ · rhs : y[i] = rhs[perm[i]].
    let mut y = vec![T::zero(); n];
    for (i, yi) in y.iter_mut().enumerate() {
        *yi = rhs[factors.perm[i]];
    }

    // Forward solve L · z = y (unit lower, CSC column-oriented): once y[j] is
    // final, propagate it down its column.
    for j in 0..n {
        let zj = y[j];
        for k in factors.l_col_ptr[j]..factors.l_col_ptr[j + 1] {
            let i = factors.l_row_idx[k];
            if i != j {
                y[i] = y[i] - factors.l_values[k] * zj;
            }
        }
    }

    // D-block solve: w = D⁻¹ · z, in place in y.
    let mut k = 0;
    while k < n {
        if factors.two_by_two[k] {
            let d11 = factors.d_diag[k];
            let d21 = factors.d_subdiag[k];
            let d22 = factors.d_diag[k + 1];
            let det = d11 * d22 - d21 * d21;
            if det == T::zero() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            let detinv = det.recip();
            let z0 = y[k];
            let z1 = y[k + 1];
            y[k] = (d22 * z0 - d21 * z1) * detinv;
            y[k + 1] = (d11 * z1 - d21 * z0) * detinv;
            k += 2;
        } else {
            let d = factors.d_diag[k];
            if d == T::zero() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            y[k] = y[k] * d.recip();
            k += 1;
        }
    }

    // Backward solve Lᵀ · v = w (CSC column j = row j of Lᵀ): dot column j's
    // multipliers against the already-solved tail.
    for j in (0..n).rev() {
        let mut acc = y[j];
        for k in factors.l_col_ptr[j]..factors.l_col_ptr[j + 1] {
            let i = factors.l_row_idx[k];
            if i != j {
                acc = acc - factors.l_values[k] * y[i];
            }
        }
        y[j] = acc;
    }

    // x = P · v : x[perm[i]] = v[i].
    let mut x = vec![T::zero(); n];
    for (i, &vi) in y.iter().enumerate() {
        x[factors.perm[i]] = vi;
    }
    Ok(x)
}

/// A memory-compact form of [`LdltFactors`] with the CSC index arrays and the
/// pivot permutation stored as `u32` instead of `usize`, halving the index
/// footprint on 64-bit targets when `n < 2^31` (and `nnz(L) < 2^32`). Indices
/// are the non-value half of a sparse factor, so this shrinks the stored factor
/// by up to `4·(nnz(L) + 2n)` bytes at **no accuracy cost**: the values are moved
/// in unchanged and [`solve`](Self::solve) is bit-identical to [`solve_ldlt`]
/// (only the index *type* differs, cast back to `usize` on read). A pure memory
/// axis for the exact factor - build it from a full factor and drop the original.
pub struct CompressedLdltFactors<T> {
    /// Matrix dimension.
    pub n: usize,
    l_col_ptr: Vec<u32>,
    l_row_idx: Vec<u32>,
    l_values: Vec<T>,
    d_diag: Vec<T>,
    d_subdiag: Vec<T>,
    two_by_two: Vec<bool>,
    perm: Vec<u32>,
}

impl<T: Scalar> CompressedLdltFactors<T> {
    /// Build from full [`LdltFactors`], **consuming** them and moving the values
    /// (no duplication - the compact factor replaces the original). Returns
    /// `None` when the indices do not fit `u32` (`n >= 2^31` or `nnz(L) >= 2^32`),
    /// where 32-bit compression does not apply; the input is dropped in that case.
    pub fn from_factors(f: LdltFactors<T>) -> Option<Self> {
        let nnz = *f.l_col_ptr.last().unwrap_or(&0);
        if f.n as u64 > u32::MAX as u64 || nnz as u64 > u32::MAX as u64 {
            return None;
        }
        Some(Self {
            n: f.n,
            l_col_ptr: f.l_col_ptr.iter().map(|&x| x as u32).collect(),
            l_row_idx: f.l_row_idx.iter().map(|&x| x as u32).collect(),
            l_values: f.l_values,
            d_diag: f.d_diag,
            d_subdiag: f.d_subdiag,
            two_by_two: f.two_by_two,
            perm: f.perm.iter().map(|&x| x as u32).collect(),
        })
    }

    /// Number of stored `L` nonzeros (the fill).
    pub fn factor_nnz(&self) -> usize {
        self.l_values.len()
    }

    /// Stored index footprint in bytes (the `u32` `l_col_ptr` + `l_row_idx` +
    /// `perm` arrays). Half of the `usize` original on a 64-bit target.
    pub fn index_bytes(&self) -> usize {
        4 * (self.l_col_ptr.len() + self.l_row_idx.len() + self.perm.len())
    }

    /// Solve `A x = b`, bit-identical to [`solve_ldlt`] on the source factors.
    pub fn solve(&self, rhs: &[T]) -> Result<Vec<T>, RslabError> {
        let n = self.n;
        if rhs.len() != n {
            return Err(RslabError::DimensionMismatch {
                expected: n,
                got: rhs.len(),
            });
        }
        let mut y = vec![T::zero(); n];
        for (i, yi) in y.iter_mut().enumerate() {
            *yi = rhs[self.perm[i] as usize];
        }
        // Forward solve L z = y.
        for j in 0..n {
            let zj = y[j];
            for k in self.l_col_ptr[j] as usize..self.l_col_ptr[j + 1] as usize {
                let i = self.l_row_idx[k] as usize;
                if i != j {
                    y[i] = y[i] - self.l_values[k] * zj;
                }
            }
        }
        // D-block solve.
        let mut k = 0;
        while k < n {
            if self.two_by_two[k] {
                let d11 = self.d_diag[k];
                let d21 = self.d_subdiag[k];
                let d22 = self.d_diag[k + 1];
                let det = d11 * d22 - d21 * d21;
                if det == T::zero() {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                let detinv = det.recip();
                let (z0, z1) = (y[k], y[k + 1]);
                y[k] = (d22 * z0 - d21 * z1) * detinv;
                y[k + 1] = (d11 * z1 - d21 * z0) * detinv;
                k += 2;
            } else {
                let d = self.d_diag[k];
                if d == T::zero() {
                    return Err(RslabError::NumericallyRankDeficient);
                }
                y[k] = y[k] * d.recip();
                k += 1;
            }
        }
        // Backward solve Lᵀ v = w.
        for j in (0..n).rev() {
            let mut acc = y[j];
            for k in self.l_col_ptr[j] as usize..self.l_col_ptr[j + 1] as usize {
                let i = self.l_row_idx[k] as usize;
                if i != j {
                    acc = acc - self.l_values[k] * y[i];
                }
            }
            y[j] = acc;
        }
        let mut x = vec![T::zero(); n];
        for (i, &vi) in y.iter().enumerate() {
            x[self.perm[i] as usize] = vi;
        }
        Ok(x)
    }
}

/// Below this RHS count (and work size) the block solve runs serially - the
/// gather/scatter + thread-spawn overhead of the parallel path only amortizes for
/// genuinely wide multi-RHS solves.
const PAR_SOLVE_MIN_RHS: usize = 8;
const PAR_SOLVE_MIN_WORK: usize = 1 << 18;

/// Solve `A · X = B` for `nrhs` right-hand sides at once. `b` and the returned
/// `x` are **row-major** `n × nrhs` buffers (row `i`'s `nrhs` values contiguous,
/// i.e. `b[i*nrhs + c]` is RHS `c` at row `i`). Processing the RHS as a block
/// loads each `L`/`D` value once and applies it to all `nrhs` columns - the
/// memory-bound amortization that makes one block solve beat `nrhs` separate
/// [`solve_ldlt`] calls.
///
/// For a genuinely wide RHS the columns are split into per-thread chunks and
/// solved concurrently (each RHS is independent). The result is **bit-identical**
/// to the serial block solve: every column undergoes the exact same operations in
/// the same order regardless of chunking, so the determinism guarantee holds.
pub fn solve_ldlt_many<T: Scalar>(
    factors: &LdltFactors<T>,
    b: &[T],
    nrhs: usize,
) -> Result<Vec<T>, RslabError> {
    let n = factors.n;
    if nrhs == 0 || b.len() != n * nrhs {
        return Err(RslabError::DimensionMismatch {
            expected: n * nrhs,
            got: b.len(),
        });
    }
    let nthreads = rayon::current_num_threads().max(1);
    if nrhs < PAR_SOLVE_MIN_RHS || n * nrhs < PAR_SOLVE_MIN_WORK || nthreads < 2 {
        return solve_ldlt_block(factors, b, nrhs);
    }
    // Split the RHS columns into `nchunks` independent contiguous ranges and solve
    // each on its own worker. Each chunk is gathered into a compact row-major
    // `n × w` sub-block, solved with the serial block kernel, then scattered back.
    let nchunks = nthreads.min(nrhs);
    let chunk = nrhs.div_ceil(nchunks);
    let ranges: Vec<(usize, usize)> = (0..nchunks)
        .map(|t| (t * chunk, ((t + 1) * chunk).min(nrhs)))
        .filter(|&(a, e)| a < e)
        .collect();
    let parts: Result<Vec<(usize, usize, Vec<T>)>, RslabError> = ranges
        .par_iter()
        .map(|&(c0, c1)| {
            let w = c1 - c0;
            let mut sub = vec![T::zero(); n * w];
            for i in 0..n {
                let ib = i * nrhs;
                let sb = i * w;
                sub[sb..sb + w].copy_from_slice(&b[ib + c0..ib + c1]);
            }
            let xs = solve_ldlt_block(factors, &sub, w)?;
            Ok((c0, c1, xs))
        })
        .collect();
    let parts = parts?;
    let mut x = vec![T::zero(); n * nrhs];
    for (c0, c1, xs) in parts {
        let w = c1 - c0;
        for i in 0..n {
            let ib = i * nrhs;
            let sb = i * w;
            x[ib + c0..ib + c1].copy_from_slice(&xs[sb..sb + w]);
        }
    }
    Ok(x)
}

/// Serial block solve over `nrhs` right-hand sides (the memory-bound AXPY kernel).
/// The parallel [`solve_ldlt_many`] fans this over column chunks.
fn solve_ldlt_block<T: Scalar>(
    factors: &LdltFactors<T>,
    b: &[T],
    nrhs: usize,
) -> Result<Vec<T>, RslabError> {
    let n = factors.n;
    // Y = Pᵀ B (gather rows; each row's `nrhs` block moves as a unit).
    let mut y = vec![T::zero(); n * nrhs];
    for i in 0..n {
        let src = factors.perm[i] * nrhs;
        y[i * nrhs..i * nrhs + nrhs].copy_from_slice(&b[src..src + nrhs]);
    }

    // Reusable single-row scratch: hoisting the reused row into a **local** buffer
    // breaks `y`'s apparent self-aliasing so the `nrhs`-wide AXPY kernels operate
    // on non-aliasing contiguous slices the compiler can vectorize (and the row is
    // loaded once per column, not once per nonzero).
    let mut row = vec![T::zero(); nrhs];
    // Forward solve L Z = Y. `y[j]` (the source row) is read by every nonzero of
    // column `j` and is not written in this sweep.
    for j in 0..n {
        let jb = j * nrhs;
        row.copy_from_slice(&y[jb..jb + nrhs]);
        for k in factors.l_col_ptr[j]..factors.l_col_ptr[j + 1] {
            let i = factors.l_row_idx[k];
            if i != j {
                let lval = factors.l_values[k];
                let ib = i * nrhs;
                let tgt = &mut y[ib..ib + nrhs];
                for c in 0..nrhs {
                    tgt[c] = tgt[c] - lval * row[c];
                }
            }
        }
    }

    // D-block solve W = D⁻¹ Z, in place.
    let mut k = 0;
    while k < n {
        if factors.two_by_two[k] {
            let d11 = factors.d_diag[k];
            let d21 = factors.d_subdiag[k];
            let d22 = factors.d_diag[k + 1];
            let det = d11 * d22 - d21 * d21;
            if det == T::zero() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            let detinv = det.recip();
            let (k0, k1) = (k * nrhs, (k + 1) * nrhs);
            for c in 0..nrhs {
                let z0 = y[k0 + c];
                let z1 = y[k1 + c];
                y[k0 + c] = (d22 * z0 - d21 * z1) * detinv;
                y[k1 + c] = (d11 * z1 - d21 * z0) * detinv;
            }
            k += 2;
        } else {
            let d = factors.d_diag[k];
            if d == T::zero() {
                return Err(RslabError::NumericallyRankDeficient);
            }
            let dinv = d.recip();
            let kb = k * nrhs;
            for c in 0..nrhs {
                y[kb + c] = y[kb + c] * dinv;
            }
            k += 1;
        }
    }

    // Backward solve Lᵀ V = W. Accumulate column `j`'s update in the local buffer
    // (the sources `y[i]`, `i > j`, are already solved and not touched here), then
    // write it back.
    for j in (0..n).rev() {
        let jb = j * nrhs;
        row.copy_from_slice(&y[jb..jb + nrhs]);
        for k in factors.l_col_ptr[j]..factors.l_col_ptr[j + 1] {
            let i = factors.l_row_idx[k];
            if i != j {
                let lval = factors.l_values[k];
                let ib = i * nrhs;
                let src = &y[ib..ib + nrhs];
                for c in 0..nrhs {
                    row[c] = row[c] - lval * src[c];
                }
            }
        }
        y[jb..jb + nrhs].copy_from_slice(&row);
    }

    // X = P V (scatter rows).
    let mut x = vec![T::zero(); n * nrhs];
    for i in 0..n {
        let dst = factors.perm[i] * nrhs;
        x[dst..dst + nrhs].copy_from_slice(&y[i * nrhs..i * nrhs + nrhs]);
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    /// ‖A·x − b‖∞ for a real or complex symmetric `A` (via `symv`).
    fn residual_inf<T: Scalar>(a: &SymmetricMatrix<T>, x: &[T], b: &[T]) -> f64 {
        let mut ax = vec![T::zero(); a.n];
        a.symv(x, &mut ax);
        (0..a.n)
            .map(|i| (ax[i] - b[i]).magnitude())
            .fold(0.0, f64::max)
    }

    #[test]
    fn compressed_ldlt_is_bit_identical_and_smaller() {
        // 32-bit index compression: the compact factor solves bit-identically to
        // the full-index factor (only the index type differs) and stores its
        // indices in half the bytes.
        let m = 16;
        let n = m * m;
        let idx = |a: usize, b: usize| a * m + b;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                r.push(p);
                cc.push(p);
                v.push(6.0_f64);
                if b + 1 < m {
                    r.push(idx(a, b + 1));
                    cc.push(p);
                    v.push(-1.0);
                }
                if a + 1 < m {
                    r.push(idx(a + 1, b));
                    cc.push(p);
                    v.push(-1.0);
                }
            }
        }
        let a = crate::CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
        let b: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();

        let reference = crate::factor_sparse_ldlt(&a).unwrap();
        let x_ref = solve_ldlt(&reference, &b).unwrap();
        let full_index_bytes =
            8 * (reference.l_col_ptr.len() + reference.l_row_idx.len() + reference.perm.len());

        let compact = CompressedLdltFactors::from_factors(crate::factor_sparse_ldlt(&a).unwrap())
            .expect("n < 2^31 compresses");
        let x_cmp = compact.solve(&b).unwrap();

        assert_eq!(x_ref.len(), x_cmp.len());
        for i in 0..n {
            assert_eq!(
                x_ref[i].to_bits(),
                x_cmp[i].to_bits(),
                "compressed solve differs from full at row {i}"
            );
        }
        assert_eq!(
            compact.index_bytes() * 2,
            full_index_bytes,
            "u32 indices halve the footprint"
        );
        assert_eq!(compact.factor_nnz(), reference.l_values.len());
    }

    #[test]
    fn parallel_wide_multi_rhs_is_bit_identical_to_serial() {
        // A large-enough sparse SPD system so solve_ldlt_many takes the parallel
        // column-chunk path (n*nrhs over the threshold). Each RHS column must be
        // bit-identical to the serial single-RHS solve, and the block residual
        // small - the determinism guarantee under RHS-parallelism.
        let m = 50;
        let n = m * m; // 2500
        let idx = |a: usize, b: usize| a * m + b;
        let (mut r, mut cc, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for a in 0..m {
            for b in 0..m {
                let p = idx(a, b);
                r.push(p);
                cc.push(p);
                v.push(6.0_f64);
                if b + 1 < m {
                    r.push(idx(a, b + 1));
                    cc.push(p);
                    v.push(-1.0);
                }
                if a + 1 < m {
                    r.push(idx(a + 1, b));
                    cc.push(p);
                    v.push(-1.0);
                }
            }
        }
        let a = crate::CscMatrix::<f64>::from_triplets(n, &r, &cc, &v).unwrap();
        let factors = crate::factor_sparse_ldlt(&a).unwrap();

        let nrhs = 128; // n*nrhs = 320_000 > PAR_SOLVE_MIN_WORK, nrhs > PAR_SOLVE_MIN_RHS
        let b: Vec<f64> = (0..n * nrhs).map(|k| ((k % 13) as f64) - 6.0).collect();
        let x_par = solve_ldlt_many(&factors, &b, nrhs).unwrap();

        // Reference: solve each column on its own (nrhs = 1 → serial block, w = 1).
        for c in 0..nrhs {
            let bc: Vec<f64> = (0..n).map(|i| b[i * nrhs + c]).collect();
            let xc = solve_ldlt_many(&factors, &bc, 1).unwrap();
            for i in 0..n {
                assert_eq!(
                    x_par[i * nrhs + c].to_bits(),
                    xc[i].to_bits(),
                    "parallel column {c} row {i} differs from serial"
                );
            }
        }
    }

    // ---- f64 ----------------------------------------------------------------

    #[test]
    fn f64_indefinite_2x2_pivot() {
        // A = [[0, 1], [1, 0]] has a zero diagonal: forces a 2×2 pivot.
        let a = SymmetricMatrix::<f64>::from_lower_triangle(
            2,
            &[(0, 0, 0.0), (1, 0, 1.0), (1, 1, 0.0)],
        );
        let f = factor_ldlt(&a).unwrap();
        let b = vec![3.0, 5.0];
        let x = solve_ldlt(&f, &b).unwrap();
        // A x = [x1, x0] = b  =>  x = [5, 3].
        assert!((x[0] - 5.0).abs() < 1e-12);
        assert!((x[1] - 3.0).abs() < 1e-12);
        assert!(residual_inf(&a, &x, &b) < 1e-12);
    }

    #[test]
    fn f64_spd_solves_to_tight_residual() {
        let entries = [
            (0, 0, 4.0),
            (1, 0, 1.0),
            (1, 1, 3.0),
            (2, 0, 2.0),
            (2, 1, -1.0),
            (2, 2, 5.0),
        ];
        let a = SymmetricMatrix::<f64>::from_lower_triangle(3, &entries);
        let b = vec![1.0, 2.0, 3.0];

        let f = factor_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-12);
    }

    #[test]
    fn f64_larger_indefinite_residual() {
        // A symmetric indefinite 5×5 exercising both 1×1 and 2×2 pivots.
        let entries = [
            (0, 0, 1.0),
            (1, 0, 3.0),
            (1, 1, 2.0),
            (2, 0, 0.5),
            (2, 1, -1.0),
            (2, 2, -4.0),
            (3, 1, 2.0),
            (3, 2, 1.0),
            (3, 3, 0.0),
            (4, 0, -2.0),
            (4, 3, 3.0),
            (4, 4, 1.0),
        ];
        let a = SymmetricMatrix::<f64>::from_lower_triangle(5, &entries);
        let b = vec![1.0, -2.0, 3.0, 0.5, -1.5];
        let f = factor_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-10);
    }

    // ---- Complex symmetric (A = Aᵀ, PARDISO mtype 6) ------------------------

    #[test]
    fn complex_antidiagonal_2x2_pivot() {
        let c = |re, im| Complex::new(re, im);
        // A = [[0, 1], [1, 0]] (complex symmetric, zero diagonal -> 2×2 pivot).
        let a = SymmetricMatrix::<Complex<f64>>::from_lower_triangle(
            2,
            &[
                (0, 0, c(0.0, 0.0)),
                (1, 0, c(1.0, 0.0)),
                (1, 1, c(0.0, 0.0)),
            ],
        );
        let f = factor_ldlt(&a).unwrap();
        let b = vec![c(1.0, 1.0), c(2.0, -1.0)];
        let x = solve_ldlt(&f, &b).unwrap();
        // A x = [x1, x0] = b  =>  x = [2 - i, 1 + i].
        assert!((x[0] - c(2.0, -1.0)).norm() < 1e-12);
        assert!((x[1] - c(1.0, 1.0)).norm() < 1e-12);
        assert!(residual_inf(&a, &x, &b) < 1e-12);
    }

    #[test]
    fn complex_diagonal_pivots() {
        let c = |re, im| Complex::new(re, im);
        // Diagonally dominant complex symmetric: all 1×1 pivots.
        let a = SymmetricMatrix::<Complex<f64>>::from_lower_triangle(
            3,
            &[
                (0, 0, c(4.0, 1.0)),
                (1, 0, c(1.0, -1.0)),
                (1, 1, c(5.0, -2.0)),
                (2, 0, c(0.5, 0.0)),
                (2, 1, c(-1.0, 0.5)),
                (2, 2, c(6.0, 1.0)),
            ],
        );
        let b = vec![c(1.0, 0.0), c(0.0, 2.0), c(-1.0, 1.0)];
        let f = factor_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(residual_inf(&a, &x, &b) < 1e-11);
    }

    #[test]
    fn complex_indefinite_mixed_pivots() {
        let c = |re, im| Complex::new(re, im);
        // 5×5 complex symmetric with small/zero diagonals to force 2×2 pivots.
        let a = SymmetricMatrix::<Complex<f64>>::from_lower_triangle(
            5,
            &[
                (0, 0, c(0.0, 0.0)),
                (1, 0, c(2.0, 1.0)),
                (1, 1, c(1.0, -1.0)),
                (2, 0, c(1.0, 0.0)),
                (2, 1, c(0.0, 1.0)),
                (2, 2, c(0.0, 0.0)),
                (3, 1, c(-1.0, 2.0)),
                (3, 2, c(3.0, 0.0)),
                (3, 3, c(2.0, 1.0)),
                (4, 0, c(1.0, 1.0)),
                (4, 3, c(0.0, -1.0)),
                (4, 4, c(1.0, 0.0)),
            ],
        );
        let b = vec![
            c(1.0, 0.0),
            c(0.0, 1.0),
            c(-1.0, 1.0),
            c(2.0, 0.0),
            c(0.5, -0.5),
        ];
        let f = factor_ldlt(&a).unwrap();
        let x = solve_ldlt(&f, &b).unwrap();
        assert!(
            residual_inf(&a, &x, &b) < 1e-10,
            "residual too large: {}",
            residual_inf(&a, &x, &b)
        );
    }

    #[test]
    fn singular_column_is_rejected() {
        // A fully zero matrix is structurally singular.
        let a = SymmetricMatrix::<f64>::zeros(2);
        assert!(matches!(
            factor_ldlt(&a),
            Err(RslabError::NumericallyRankDeficient)
        ));
    }
}
