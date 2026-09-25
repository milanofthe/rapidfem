//! The numeric-only refactorization on a frozen pattern and pivot order.

use super::*;

impl<T: Scalar> KluSolver<T> {
    /// Numeric-only refactorization: replay the stored pattern and pivot
    /// sequence on new values with the **same** sparsity pattern. No symbolic
    /// work, no pivot search, the fast path for frequency sweeps and Newton
    /// steps. Fails with a pattern-mismatch error if `a`'s pattern deviates
    /// from the factored one, and with [`RslabError::SingularBasis`] if a
    /// frozen pivot becomes zero (re-`factor` with pivoting in that case).
    /// After an error the factorization is invalid; a subsequent successful
    /// `refactor` or a fresh `factor` makes it valid again.
    // The replay loops index several parallel arrays at offset positions;
    // iterator forms would obscure the offset arithmetic.
    #[allow(clippy::needless_range_loop)]
    pub fn refactor(&mut self, a: &GeneralCsc<T>) -> Result<(), RslabError> {
        a.validate()?;
        let t = crate::clock::Instant::now();
        // Cloned out of the factors so the borrow below stays exclusive; the
        // flag is read-only for us either way.
        let int_flag = self.factors.interrupt.clone();
        let int_flag = int_flag.as_deref();
        let f = &mut self.factors;
        if a.n != f.n {
            return Err(RslabError::DimensionMismatch {
                expected: f.n,
                got: a.n,
            });
        }
        if a.nnz() != f.nnz_a {
            return Err(pattern_mismatch());
        }
        let rs_inv = row_scale_inv(a, f.scaled);

        // Branch-free pattern verification against the recorded program: every
        // entry must map to the exact final position it had at factor time
        // (this subsumes the old per-column F-walk and leftover checks).
        {
            let mut acc: Ki = 0;
            for (k, &r) in a.row_idx.iter().enumerate() {
                acc |= (f.pinv[r] as Ki) ^ f.scatter_expect[k];
            }
            if acc != 0 {
                return Err(pattern_mismatch());
            }
        }

        // Per-block replay jobs over disjoint value ranges: L/U/F entries and
        // `udiag` of a block are contiguous (`colptr[bs]..colptr[be]`), so the
        // mutable arrays split cleanly and the blocks replay independently,
        // in parallel when the factor-time opt-in chose it (bit-identical:
        // per-block work is sequential and shares nothing).
        struct RJob<'s, T> {
            b: usize,
            l_v: &'s mut [T],
            u_v: &'s mut [T],
            ud: &'s mut [T],
            f_v: &'s mut [T],
        }
        let nblocks = f.block_ptr.len() - 1;
        let mut jobs: Vec<RJob<'_, T>> = Vec::with_capacity(nblocks);
        {
            let (mut lv, mut uv) = (f.l_val.as_mut_slice(), f.u_val.as_mut_slice());
            let (mut ud, mut fv) = (f.udiag.as_mut_slice(), f.f_val.as_mut_slice());
            for b in 0..nblocks {
                interrupt_check(int_flag)?;
                let (bs, be) = (f.block_ptr[b], f.block_ptr[b + 1]);
                let (a1, r1) =
                    std::mem::take(&mut lv).split_at_mut(f.l_colptr[be] - f.l_colptr[bs]);
                lv = r1;
                let (a2, r2) =
                    std::mem::take(&mut uv).split_at_mut(f.u_colptr[be] - f.u_colptr[bs]);
                uv = r2;
                let (a3, r3) = std::mem::take(&mut ud).split_at_mut(be - bs);
                ud = r3;
                let (a4, r4) =
                    std::mem::take(&mut fv).split_at_mut(f.f_colptr[be] - f.f_colptr[bs]);
                fv = r4;
                jobs.push(RJob {
                    b,
                    l_v: a1,
                    u_v: a2,
                    ud: a3,
                    f_v: a4,
                });
            }
        }
        let (block_ptr, col_perm) = (&f.block_ptr, &f.col_perm);
        let (l_colptr, l_rowidx) = (&f.l_colptr, &f.l_rowidx);
        let (u_colptr, u_rowidx) = (&f.u_colptr, &f.u_rowidx);
        let f_colptr = &f.f_colptr;
        let scatter_target = &f.scatter_target;
        let rs_inv_ref = &rs_inv;
        // Raw column-disjoint views of one block's value arrays for the
        // level-parallel replay: every column writes only its own L/U/F/diag
        // slots and reads L columns completed in earlier levels (fenced by the
        // per-level join), so the shared-mutable access is race-free.
        struct BlockPtrs<T> {
            l_v: PanelPtr<T>,
            u_v: PanelPtr<T>,
            ud: PanelPtr<T>,
            f_v: PanelPtr<T>,
        }
        impl<T> Clone for BlockPtrs<T> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<T> Copy for BlockPtrs<T> {}

        // Replay one column through the recorded program: scatter, eliminate
        // in the stored topological order (bit-identical to `factor_impl`'s
        // pass 3: same `fmadd`, same per-column order), pivot, emit L.
        // SAFETY: caller guarantees exclusive ownership of column `j`'s output
        // slots and completed dependency columns (see `BlockPtrs`); `x` is this
        // caller's scratch, all-zero on entry and left all-zero.
        let replay_col = |j: usize,
                          bs: usize,
                          bases: (usize, usize, usize),
                          p: BlockPtrs<T>,
                          x: &mut [T],
                          sync: Option<(
            &[std::sync::atomic::AtomicBool],
            &std::sync::atomic::AtomicBool,
        )>|
         -> Result<(), RslabError> {
            let (l_base, u_base, f_base) = bases;
            let c = col_perm[j];
            unsafe {
                for k in a.col_ptr[c]..a.col_ptr[c + 1] {
                    let r = a.row_idx[k];
                    let sv = a.values[k] * T::from_real(rs_inv_ref[r]);
                    let tv = scatter_target[k];
                    if tv & KI_FBIT != 0 {
                        *p.f_v.get().add((tv & !KI_FBIT) as usize - f_base) = sv;
                    } else {
                        x[tv as usize - bs] = sv;
                    }
                }
                for k in u_colptr[j]..u_colptr[j + 1] {
                    let pr = u_rowidx[k] as usize;
                    // Pipelined mode: consume L column `pr` only once its
                    // owner has published it (Acquire pairs with the owner's
                    // Release store after emitting the column).
                    if let Some((ready, abort)) = sync {
                        use std::sync::atomic::Ordering as AOrd;
                        let mut spins = 0u32;
                        while !ready[pr - bs].load(AOrd::Acquire) {
                            if abort.load(AOrd::Acquire) {
                                return Err(pattern_mismatch());
                            }
                            spins += 1;
                            if spins & 0x3FF == 0 {
                                std::thread::yield_now();
                            } else {
                                std::hint::spin_loop();
                            }
                        }
                    }
                    let xu = x[pr - bs];
                    x[pr - bs] = T::zero();
                    *p.u_v.get().add(k - u_base) = xu;
                    let nxu = T::zero() - xu;
                    for kl in l_colptr[pr]..l_colptr[pr + 1] {
                        let lr = l_rowidx[kl] as usize - bs;
                        debug_assert!(lr < x.len());
                        *x.get_unchecked_mut(lr) =
                            fmadd(nxu, *p.l_v.get().add(kl - l_base), *x.get_unchecked(lr));
                    }
                }
                let d = x[j - bs];
                x[j - bs] = T::zero();
                if d.magnitude() == 0.0 || !d.is_finite() {
                    return Err(RslabError::SingularBasis { column: c });
                }
                *p.ud.get().add(j - bs) = d;
                for k in l_colptr[j]..l_colptr[j + 1] {
                    let lr = l_rowidx[k] as usize - bs;
                    *p.l_v.get().add(k - l_base) = x[lr] / d;
                    x[lr] = T::zero();
                }
            }
            Ok(())
        };

        let pipelined = &f.pipelined;
        let replay_block = |job: RJob<'_, T>| -> Result<(), RslabError> {
            let b = job.b;
            let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
            let bases = (l_colptr[bs], u_colptr[bs], f_colptr[bs]);
            let ptrs = BlockPtrs {
                l_v: PanelPtr(job.l_v.as_mut_ptr()),
                u_v: PanelPtr(job.u_v.as_mut_ptr()),
                ud: PanelPtr(job.ud.as_mut_ptr()),
                f_v: PanelPtr(job.f_v.as_mut_ptr()),
            };
            let nthreads = rayon::current_num_threads().max(1);
            let pipe_nw = pipelined
                .iter()
                .find(|&&(pb, _)| pb == b)
                .map(|&(_, nw)| nw);
            if nthreads >= 2 && pipe_nw.is_some() {
                // NICSLU-style pipelined replay: worker `w` owns columns
                // `bs+w, bs+w+nw, ...` in order and spin-waits just-in-time on
                // each U-dependency's ready flag before consuming its L
                // column. Bit-identical (per-column arithmetic and writes are
                // untouched); dedicated OS threads, so a spinning peer can
                // never starve the owner of the column it waits for (a rayon
                // task pool could).
                use std::sync::atomic::{AtomicBool, Ordering as AOrd};
                let bn = be - bs;
                // Worker count bounded by the DAG's admissible speedup (the
                // plan's W/C ratio) and the thread budget.
                let nw = pipe_nw.unwrap_or(2).clamp(2, nthreads);
                let ready: Vec<AtomicBool> = (0..bn).map(|_| AtomicBool::new(false)).collect();
                let abort = AtomicBool::new(false);
                let errs: Vec<Result<(), RslabError>> = std::thread::scope(|sc| {
                    let handles: Vec<_> = (0..nw)
                        .map(|w| {
                            let (ready, abort) = (&ready, &abort);
                            sc.spawn(move || -> Result<(), RslabError> {
                                let mut x = vec![T::zero(); bn];
                                let mut jj = bs + w;
                                while jj < be {
                                    interrupt_check(int_flag)?;
                                    // SAFETY: worker-owned column; deps are
                                    // fenced by the ready Acquire loads inside
                                    // `replay_col`.
                                    let r = replay_col(
                                        jj,
                                        bs,
                                        bases,
                                        ptrs,
                                        &mut x,
                                        Some((ready, abort)),
                                    );
                                    ready[jj - bs].store(true, AOrd::Release);
                                    if let Err(e) = r {
                                        // Release everything this worker still
                                        // owns so the peers' spins terminate.
                                        abort.store(true, AOrd::Release);
                                        let mut k = jj + nw;
                                        while k < be {
                                            ready[k - bs].store(true, AOrd::Release);
                                            k += nw;
                                        }
                                        return Err(e);
                                    }
                                    if abort.load(AOrd::Acquire) {
                                        let mut k = jj + nw;
                                        while k < be {
                                            ready[k - bs].store(true, AOrd::Release);
                                            k += nw;
                                        }
                                        return Ok(());
                                    }
                                    jj += nw;
                                }
                                Ok(())
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap_or(Err(pattern_mismatch())))
                        .collect()
                });
                // Deterministic error selection: a real singular pivot wins
                // over the sympathetic aborts of the other workers (their
                // spins return `pattern_mismatch`), lowest column first.
                let mut first: Option<RslabError> = None;
                for r in errs {
                    if let Err(e) = r {
                        let better = match (&e, &first) {
                            (_, None) => true,
                            (
                                RslabError::SingularBasis { column: c1 },
                                Some(RslabError::SingularBasis { column: c0 }),
                            ) => c1 < c0,
                            (RslabError::SingularBasis { .. }, Some(_)) => true,
                            _ => false,
                        };
                        if better {
                            first = Some(e);
                        }
                    }
                }
                if let Some(e) = first {
                    return Err(e);
                }
                return Ok(());
            }
            let mut x = vec![T::zero(); be - bs];
            for j in bs..be {
                // SAFETY: this closure exclusively owns the whole block.
                replay_col(j, bs, bases, ptrs, &mut x, None)?;
            }
            Ok(())
        };
        if f.par_refactor && nblocks > 1 {
            use rayon::prelude::*;
            let results: Vec<Result<(), RslabError>> =
                jobs.into_par_iter().map(replay_block).collect();
            for r in results {
                r?;
            }
        } else {
            for job in jobs {
                replay_block(job)?;
            }
        }
        f.rs_inv = rs_inv;
        let entry = (std::mem::size_of::<T>() + std::mem::size_of::<Ki>()) as u64;
        let nnz = self.diagnostics.factor_nnz;
        self.diagnostics.push(
            "klu-refactor",
            t.elapsed().as_secs_f64() * 1e3,
            diagnostics_flops(&self.diagnostics),
            nnz * entry,
        );
        Ok(())
    }
}
