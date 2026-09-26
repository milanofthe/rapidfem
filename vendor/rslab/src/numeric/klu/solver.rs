//! The factored solver: the block substitutions and the factor exports.

use super::*;

impl<T: Scalar> crate::numeric::direct::SolveCore<T> for KluSolver<T> {
    const NAME: &'static str = "klu";

    fn dim(&self) -> usize {
        self.factors.n
    }

    fn counter(&self) -> &crate::diagnostics::SolveCounter {
        &self.solves
    }

    fn solve_raw(&self, b: &[T], nrhs: usize, transpose: bool) -> Result<Vec<T>, RslabError> {
        let n = self.factors.n;
        Ok(if transpose {
            (0..nrhs)
                .flat_map(|c| self.solve_transpose_one(&b[c * n..(c + 1) * n]))
                .collect()
        } else if nrhs == 1 {
            self.solve_one(b)
        } else {
            self.solve_block(b, nrhs)
        })
    }
}

crate::numeric::direct::direct_solver!(KluSolver);

impl<T: Scalar> KluSolver<T> {
    /// One-shot analyze + factor with the given settings. Skips the a-priori
    /// estimate and stage timing (empty [`diagnostics`](Self::diagnostics)),
    /// like [`LuSolver::factor`](crate::LuSolver::factor); use the phased
    /// [`KluSymbolic::factor`] for populated diagnostics.
    pub fn factor(a: &GeneralCsc<T>, settings: &KluSettings) -> Result<Self, RslabError> {
        KluSymbolic::analyze(a, settings)?.factor(a, settings)
    }

    /// Everything this factorization can tell about itself (see
    /// [`Diagnostics`](crate::Diagnostics)): the measured factor and refactor
    /// stages, the fill, the a-priori
    /// [`MemoryEstimate`](crate::diagnostics::MemoryEstimate) where
    /// [`KluSymbolic::estimate_memory`] computed it, and the solve-phase
    /// accumulators. A snapshot.
    pub fn diagnostics(&self) -> crate::diagnostics::Diagnostics {
        let mut d = self.diagnostics.clone();
        d.solves = self.solves.snapshot();
        d
    }

    /// Pivots lifted by a static regularization: always `0`, a vanishing
    /// pivot is a [`RslabError::SingularBasis`] at factor time instead.
    pub fn n_perturbed(&self) -> usize {
        0
    }

    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.factors.n
    }

    /// Number of BTF diagonal blocks.
    pub fn n_blocks(&self) -> usize {
        self.factors.block_ptr.len() - 1
    }

    /// Stored factor entries: L + U + diagonal + off-block.
    pub fn factor_nnz(&self) -> usize {
        self.factors.l_val.len()
            + self.factors.u_val.len()
            + self.factors.udiag.len()
            + self.factors.f_val.len()
    }

    /// The unit lower factor `L`, diagonal included. With the row scaling
    /// `R = diag(row_scale)` and the permutations `P_r` (row `k` is row
    /// `row_perm[k]`) and `P_c` (column `k` is column `col_perm[k]`), the
    /// factorization is
    ///
    /// ```text
    /// P_r R A P_c = L U + F
    /// ```
    ///
    /// with `L` and `U` block diagonal over the blocks
    /// [`block_ptr`](Self::block_ptr) of the block triangular form and `F`
    /// the entries above the diagonal blocks.
    pub fn l_matrix(&self) -> GeneralCsc<T> {
        let f = &self.factors;
        factor_csc(f.n, &f.l_colptr, &f.l_rowidx, &f.l_val, |_| Some(T::one()))
    }

    /// The upper factor `U`, pivots on the diagonal (see
    /// [`l_matrix`](Self::l_matrix)).
    pub fn u_matrix(&self) -> GeneralCsc<T> {
        let f = &self.factors;
        factor_csc(f.n, &f.u_colptr, &f.u_rowidx, &f.u_val, |j| {
            Some(f.udiag[j])
        })
    }

    /// The entries `F` above the diagonal blocks (see
    /// [`l_matrix`](Self::l_matrix)).
    pub fn f_matrix(&self) -> GeneralCsc<T> {
        let f = &self.factors;
        factor_csc(f.n, &f.f_colptr, &f.f_rowidx, &f.f_val, |_| None)
    }

    /// The row permutation: row `k` of the factored matrix is row
    /// `row_perm[k]` of `A` (see [`l_matrix`](Self::l_matrix)).
    pub fn row_perm(&self) -> &[usize] {
        &self.factors.row_perm
    }

    /// The column permutation: column `k` of the factored matrix is column
    /// `col_perm[k]` of `A` (see [`l_matrix`](Self::l_matrix)).
    pub fn col_perm(&self) -> &[usize] {
        &self.factors.col_perm
    }

    /// Boundaries of the diagonal blocks of the block triangular form: block
    /// `b` holds the rows and columns `block_ptr[b]..block_ptr[b + 1]` of
    /// the factored matrix.
    pub fn block_ptr(&self) -> &[usize] {
        &self.factors.block_ptr
    }

    /// The row scaling: row `i` of `A` is multiplied by `row_scale[i]`
    /// before the factorization (all ones without scaling, see
    /// [`l_matrix`](Self::l_matrix)).
    pub fn row_scale(&self) -> &[f64] {
        &self.factors.rs_inv
    }

    fn solve_one(&self, b: &[T]) -> Vec<T> {
        let f = &self.factors;
        let mut w = vec![T::zero(); f.n];
        for (k, &orig) in f.row_perm.iter().enumerate() {
            w[k] = b[orig] * T::from_real(f.rs_inv[orig]);
        }
        self.solve_permuted(&mut w);
        let mut xout = vec![T::zero(); f.n];
        for (k, &c) in f.col_perm.iter().enumerate() {
            xout[c] = w[k];
        }
        xout
    }

    /// Solve the transposed system `A^T x = b` with the **same** factorization.
    ///
    /// This is the plain transpose, NOT the conjugate transpose: for a complex
    /// adjoint solve `A^H x = b`, conjugate `b` before and `x` after. (This
    /// matches the convention of the usual sparse-LU transpose solves, and is
    /// what implicit-function adjoints over holomorphic residuals need.)
    ///
    /// The stored form is `A = Rs * P_r^T * M * C` with `M` the block-upper
    /// (BTF) permuted, row-scaled matrix and `M_bb = L_b U_b` per diagonal
    /// block, so `A^T x = b` is `M^T (P_r Rs x) = C b`: gather `b` through the
    /// column permutation, run the transposed block substitution (blocks
    /// forward, per block `U^T` forward then `L^T` backward, off-block `F^T`
    /// contributions from the already-solved earlier blocks), then scatter
    /// through the row permutation and undo the row scaling. Sequential and
    /// bit-deterministic, like [`solve`](Self::solve).
    fn solve_transpose_one(&self, b: &[T]) -> Vec<T> {
        let f = &self.factors;
        // w = C*b: position k of the permuted system reads b at its column.
        let mut w = vec![T::zero(); f.n];
        for (k, &c) in f.col_perm.iter().enumerate() {
            w[k] = b[c];
        }
        self.solve_permuted_transpose(&mut w);
        // x = Rs^-1 * P_r^T * w: scatter through the row permutation, then undo
        // the row scaling (Rs is diagonal, so it transposes onto the solution).
        let mut xout = vec![T::zero(); f.n];
        for (k, &orig) in f.row_perm.iter().enumerate() {
            xout[orig] = w[k] * T::from_real(f.rs_inv[orig]);
        }
        xout
    }

    /// The transposed block substitution on the permuted vector: `M^T` is block
    /// **lower** triangular (the transpose of the BTF block-upper `M`), so the
    /// blocks run forward, and within a block `M_bb^T = U_b^T L_b^T` solves as
    /// `U^T` (lower, diagonal `udiag`) forward then `L^T` (unit upper) backward.
    /// Column `j` of the stored `U`/`L`/`F` is row `j` of the transpose, so
    /// every inner loop is a gather over the existing column storage.
    fn solve_permuted_transpose(&self, w: &mut [T]) {
        // Deliberately mul+sub, NOT `fmadd`: every inner loop here is a
        // gather onto a single accumulator - a latency-bound serial chain
        // where the FMA's higher latency loses to the pipelined mul + sub
        // (see the `solve_ldlt` backward-sweep note). `fmadd` stays in the
        // scatter-form sweeps of `solve_permuted`/`solve_many`.
        let f = &self.factors;
        for b in 0..f.block_ptr.len() - 1 {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            // F^T: this block's rows read the already-solved earlier blocks.
            for j in bs..be {
                let mut acc = w[j];
                for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                    acc = acc - f.f_val[k] * w[f.f_rowidx[k] as usize];
                }
                w[j] = acc;
            }
            // U^T (lower triangular, diagonal `udiag`) forward within the block.
            for j in bs..be {
                let mut acc = w[j];
                for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                    acc = acc - f.u_val[k] * w[f.u_rowidx[k] as usize];
                }
                w[j] = acc / f.udiag[j];
            }
            // L^T (unit upper) backward within the block.
            for j in (bs..be).rev() {
                let mut acc = w[j];
                for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                    acc = acc - f.l_val[k] * w[f.l_rowidx[k] as usize];
                }
                w[j] = acc;
            }
        }
    }

    fn solve_block(&self, b: &[T], nrhs: usize) -> Vec<T> {
        let f = &self.factors;
        let n = f.n;
        // Permute + scale all columns into the row-major work block.
        let mut w = vec![T::zero(); n * nrhs];
        for (k, &orig) in f.row_perm.iter().enumerate() {
            let sv = T::from_real(f.rs_inv[orig]);
            for c in 0..nrhs {
                w[k * nrhs + c] = b[c * n + orig] * sv;
            }
        }
        // Row j's values, staged so the axpy targets never alias the source.
        let mut xj = vec![T::zero(); nrhs];
        for blk in (0..f.block_ptr.len() - 1).rev() {
            let (bs, be) = (f.block_ptr[blk], f.block_ptr[blk + 1]);
            // L (unit lower) forward within the block. Negating the factor
            // value (loop-invariant here) instead of `xj` keeps each column's
            // FMA product bitwise equal to `solve_permuted`'s
            // (`(-a)*b == a*(-b)` exactly per real FMA).
            for j in bs..be {
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                    let (lr, nlv) = (f.l_rowidx[k] as usize, T::zero() - f.l_val[k]);
                    let row = &mut w[lr * nrhs..lr * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&xj) {
                        *r = fmadd(nlv, x, *r);
                    }
                }
            }
            // U backward within the block. Per-element division (not
            // reciprocal-multiply) keeps each column bit-identical to `solve`.
            for j in (bs..be).rev() {
                let d = f.udiag[j];
                {
                    let row = &mut w[j * nrhs..j * nrhs + nrhs];
                    for r in row.iter_mut() {
                        *r = *r / d;
                    }
                }
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                    let (ur, nuv) = (f.u_rowidx[k] as usize, T::zero() - f.u_val[k]);
                    let row = &mut w[ur * nrhs..ur * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&xj) {
                        *r = fmadd(nuv, x, *r);
                    }
                }
            }
            // Off-block columns feed the rows of earlier blocks.
            for j in bs..be {
                xj.copy_from_slice(&w[j * nrhs..j * nrhs + nrhs]);
                for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                    let (fr, nfv) = (f.f_rowidx[k] as usize, T::zero() - f.f_val[k]);
                    let row = &mut w[fr * nrhs..fr * nrhs + nrhs];
                    for (r, &x) in row.iter_mut().zip(&xj) {
                        *r = fmadd(nfv, x, *r);
                    }
                }
            }
        }
        // Undo the column permutation.
        let mut xout = vec![T::zero(); n * nrhs];
        for (k, &col) in f.col_perm.iter().enumerate() {
            for c in 0..nrhs {
                xout[c * n + col] = w[k * nrhs + c];
            }
        }
        xout
    }

    /// The block forward/backward substitution on the permuted/scaled vector.
    /// The axpys run through `fmadd` with the loop-invariant operand negated
    /// once per column, `solve_many` negates the per-entry factor value
    /// instead, which is bitwise the same product (`(-a)*b == a*(-b)` holds
    /// exactly per real FMA), so the two stay bit-identical per column.
    fn solve_permuted(&self, w: &mut [T]) {
        let f = &self.factors;
        for b in (0..f.block_ptr.len() - 1).rev() {
            let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
            // L (unit lower) forward within the block. Unchecked: all row
            // indices are final positions `< n` by construction.
            for j in bs..be {
                let xj = w[j];
                if xj != T::zero() {
                    let nxj = T::zero() - xj;
                    for k in f.l_colptr[j]..f.l_colptr[j + 1] {
                        let lr = f.l_rowidx[k] as usize;
                        debug_assert!(lr < w.len());
                        unsafe {
                            *w.get_unchecked_mut(lr) = fmadd(f.l_val[k], nxj, *w.get_unchecked(lr));
                        }
                    }
                }
            }
            // U backward within the block.
            for j in (bs..be).rev() {
                let xj = w[j] / f.udiag[j];
                w[j] = xj;
                if xj != T::zero() {
                    let nxj = T::zero() - xj;
                    for k in f.u_colptr[j]..f.u_colptr[j + 1] {
                        let ur = f.u_rowidx[k] as usize;
                        debug_assert!(ur < w.len());
                        unsafe {
                            *w.get_unchecked_mut(ur) = fmadd(f.u_val[k], nxj, *w.get_unchecked(ur));
                        }
                    }
                }
            }
            // Off-block columns feed the rows of earlier blocks.
            for j in bs..be {
                let xj = w[j];
                if xj != T::zero() {
                    let nxj = T::zero() - xj;
                    for k in f.f_colptr[j]..f.f_colptr[j + 1] {
                        let fr = f.f_rowidx[k] as usize;
                        debug_assert!(fr < w.len());
                        unsafe {
                            *w.get_unchecked_mut(fr) = fmadd(f.f_val[k], nxj, *w.get_unchecked(fr));
                        }
                    }
                }
            }
        }
    }
}
