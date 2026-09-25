//! The analysis: the block triangular form and the per-block orderings.

use super::*;

impl KluSymbolic {
    /// Analyze the pattern of `a`: BTF (unless disabled) + per-block AMD.
    /// The values are read only by the row matching.
    ///
    /// Fails with [`RslabError::StructurallySingular`] when no complete
    /// matching exists (some set of `k` columns has entries in fewer than `k`
    /// rows), such a matrix is singular for *every* value assignment.
    pub fn analyze<T: Scalar>(
        a: &GeneralCsc<T>,
        settings: &KluSettings,
    ) -> Result<Self, RslabError> {
        a.validate()?;
        let n = a.n;
        // Same 31-bit range gate as factor time (the whole path is Ki-based;
        // checking here lets analyze and the BTF pass use narrow indices).
        if n as u64 >= KI_FBIT as u64 || a.nnz() as u64 >= KI_FBIT as u64 {
            return Err(RslabError::InvalidInput(
                "klu: dimension/nnz exceeds the 31-bit index range of this path".to_string(),
            ));
        }

        // Matching bakeoff (BTF on): deterministic maximum-matching
        // candidates (see `btf::matching_candidates`); with more than one,
        // each candidate's AMD-ordered blocks are scored by exact Cholesky
        // lnz (Gilbert-Ng-Peyton column counts on the ordered symmetrized
        // pattern) and the cheapest matching wins. Different maximum
        // matchings differ by several-x fill on the harmonic-balance class
        // (onetone/twotone), in matrix-dependent directions - measuring
        // beats guessing. The common MNA case (complete structural
        // diagonal) short-circuits to a single candidate and pays nothing.
        let weighted = if settings.btf && settings.matching {
            let cache = crate::scaling::mc64::compute_matching_general(a)?;
            (cache.n_matched == n).then_some(cache.perm)
        } else {
            None
        };
        let (pre_row_perm, col_perm, block_ptr) = if let Some(m) = weighted {
            // The weighted matching decides the transversal; the blocks are
            // its strongly connected components.
            let form = btf::btf_from_matching(n, &a.col_ptr, &a.row_idx, m);
            let of = order_blocks(a, form, false)?;
            (of.pre_row_perm, of.col_perm, of.block_ptr)
        } else if settings.btf {
            let cands = btf::matching_candidates(n, &a.col_ptr, &a.row_idx)
                .ok_or(RslabError::StructurallySingular)?;
            let score_it = cands.len() > 1;
            let mut best: Option<OrderedForm> = None;
            for m in cands {
                let form = btf::btf_from_matching(n, &a.col_ptr, &a.row_idx, m);
                let of = order_blocks(a, form, score_it)?;
                best = match best {
                    Some(b) if b.score <= of.score => Some(b),
                    _ => Some(of),
                };
            }
            let Some(of) = best else {
                // Unreachable: `matching_candidates` returns at least one
                // matching or `None` (handled above as StructurallySingular).
                return Err(RslabError::InvalidInput(
                    "klu: no matching candidate".to_string(),
                ));
            };
            (of.pre_row_perm, of.col_perm, of.block_ptr)
        } else {
            let ident: Vec<usize> = (0..n).collect();
            let bp = if n == 0 { vec![0] } else { vec![0, n] };
            let form = btf::BtfForm {
                row_perm: ident.clone(),
                col_perm: ident,
                block_ptr: bp,
            };
            let of = order_blocks(a, form, false)?;
            (of.pre_row_perm, of.col_perm, of.block_ptr)
        };

        // Freeze the analyzed pattern in the (final) pre-pivot space for the
        // a-priori estimators. Narrow inverse permutation: the gather is
        // bound by the random `pinv_pre[r]` reads.
        let mut pinv_pre = vec![0 as Ki; n];
        for (k, &r) in pre_row_perm.iter().enumerate() {
            pinv_pre[r] = k as Ki;
        }
        let mut pat_col_ptr = Vec::with_capacity(n + 1);
        let mut pat_row_idx = Vec::with_capacity(a.nnz());
        pat_col_ptr.push(0);
        for &c in &col_perm {
            for &r in &a.row_idx[a.col_ptr[c]..a.col_ptr[c + 1]] {
                pat_row_idx.push(pinv_pre[r] as usize);
            }
            pat_col_ptr.push(pat_row_idx.len());
        }

        Ok(Self {
            n,
            nnz: a.nnz(),
            pre_row_perm,
            col_perm,
            block_ptr,
            pat_col_ptr,
            pat_row_idx,
            fill: std::sync::OnceLock::new(),
        })
    }

    /// Symbolic Gilbert-Peierls pass over the stored pattern assuming
    /// diagonal pivots: exact per-path fill and flop counts, no values.
    /// Computed once and cached, the pass costs about as much as a numeric
    /// factor, so repeated `factor`/`estimate_memory` calls must not repay it.
    fn symbolic_fill(&self) -> KluFill {
        *self.fill.get_or_init(|| self.symbolic_fill_uncached())
    }

    fn symbolic_fill_uncached(&self) -> KluFill {
        let n = self.n;
        let mut stamp = vec![0usize; n];
        let mut node_stack = vec![0usize; n];
        let mut cur_stack = vec![0usize; n];
        let mut l_colptr = vec![0usize];
        let mut l_rowidx: Vec<usize> = Vec::new();
        let mut leaves: Vec<usize> = Vec::new();
        let (mut u_nnz, mut f_nnz, mut flops) = (0u64, 0u64, 0u64);

        for b in 0..self.block_ptr.len() - 1 {
            let (bs, be) = (self.block_ptr[b], self.block_ptr[b + 1]);
            for j in bs..be {
                let sj = j + 1;
                leaves.clear();
                for k in self.pat_col_ptr[j]..self.pat_col_ptr[j + 1] {
                    let pre = self.pat_row_idx[k];
                    if pre < bs {
                        f_nnz += 1;
                        continue;
                    }
                    if stamp[pre] == sj {
                        continue;
                    }
                    stamp[pre] = sj;
                    if pre >= j {
                        leaves.push(pre);
                        continue;
                    }
                    let mut d = 0usize;
                    node_stack[0] = pre;
                    cur_stack[0] = l_colptr[pre];
                    loop {
                        let u = node_stack[d];
                        let endp = l_colptr[u + 1];
                        let mut descended = false;
                        while cur_stack[d] < endp {
                            let ch = l_rowidx[cur_stack[d]];
                            cur_stack[d] += 1;
                            if stamp[ch] == sj {
                                continue;
                            }
                            stamp[ch] = sj;
                            if ch >= j {
                                leaves.push(ch);
                                continue;
                            }
                            d += 1;
                            node_stack[d] = ch;
                            cur_stack[d] = l_colptr[ch];
                            descended = true;
                            break;
                        }
                        if descended {
                            continue;
                        }
                        // u finished: one U entry, applying its L column.
                        u_nnz += 1;
                        flops += 2 * (l_colptr[u + 1] - l_colptr[u]) as u64;
                        if d == 0 {
                            break;
                        }
                        d -= 1;
                    }
                }
                for &lv in &leaves {
                    if lv != j {
                        l_rowidx.push(lv);
                    }
                }
                flops += (l_rowidx.len() - l_colptr[j]) as u64; // divisions
                l_colptr.push(l_rowidx.len());
            }
        }
        KluFill {
            l_nnz: l_rowidx.len() as u64,
            u_nnz,
            f_nnz,
            flops,
        }
    }

    /// Exact symbolic factor fill (`L` + `U` + diagonal + off-block entries)
    /// under the diagonal-pivoting assumption, the memory-backstop metric,
    /// mirroring [`LuSymbolic::symbolic_factor_nnz`](crate::LuSymbolic::symbolic_factor_nnz).
    pub fn symbolic_factor_nnz(&self) -> usize {
        let fill = self.symbolic_fill();
        (fill.l_nnz + fill.u_nnz + self.n as u64 + fill.f_nnz) as usize
    }

    /// **A-priori** memory/work estimate for factoring a matrix of scalar
    /// type `T` with this analysis, deterministic, computed from the stored
    /// pattern alone, mirroring [`LuSymbolic::estimate_memory`](crate::LuSymbolic::estimate_memory).
    ///
    /// KLU specifics: the fill is exact under diagonal pivoting (threshold
    /// pivoting can shift it slightly); `factor_flops` is the Gilbert-Peierls
    /// flop count (not the supernodal `nrow^2*ncol` proxy); the path is
    /// strictly sequential, so `critical_path_flops == factor_flops` and
    /// `max_tree_width == 1`; there are no dense panels.
    pub fn estimate_memory<T: Scalar>(&self) -> crate::diagnostics::MemoryEstimate {
        let fill = self.symbolic_fill();
        let value_bytes = std::mem::size_of::<T>();
        let entry = (value_bytes + std::mem::size_of::<usize>()) as u64;
        let factor_nnz = fill.l_nnz + fill.u_nnz + self.n as u64 + fill.f_nnz;
        let factor_bytes = factor_nnz * entry;
        let input_bytes = self.nnz as u64 * entry;
        let workspace_bytes = self.n as u64 * (value_bytes as u64 + 4 * 8);
        crate::diagnostics::MemoryEstimate {
            value_bytes,
            factor_nnz,
            factor_bytes,
            panels_all_bytes: 0,
            panel_live_peak_bytes: 0,
            transient_peak_bytes: factor_bytes + input_bytes + workspace_bytes,
            factor_flops: fill.flops,
            critical_path_flops: fill.flops,
            max_tree_width: 1,
        }
    }

    /// Matrix dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of diagonal blocks in the BTF form.
    pub fn n_blocks(&self) -> usize {
        self.block_ptr.len() - 1
    }

    /// Size of the largest diagonal block (the only part that generates fill).
    pub fn max_block_size(&self) -> usize {
        (0..self.n_blocks())
            .map(|b| self.block_ptr[b + 1] - self.block_ptr[b])
            .max()
            .unwrap_or(0)
    }

    /// Diagonal-block boundaries (`n_blocks + 1` entries).
    pub fn block_ptr(&self) -> &[usize] {
        &self.block_ptr
    }

    /// Numeric factorization of `a`, which must share the analyzed pattern.
    /// Records the measured factor stage in the solver's
    /// [`diagnostics`](KluSolver::diagnostics), and the a-priori estimate when
    /// [`estimate_memory`](Self::estimate_memory) computed it (the estimate
    /// costs about as much as the factorization, so it is never computed
    /// implicitly).
    pub fn factor<T: Scalar>(
        &self,
        a: &GeneralCsc<T>,
        settings: &KluSettings,
    ) -> Result<KluSolver<T>, RslabError> {
        // Attach the a-priori estimate only when it has already been computed
        // (an explicit `estimate_memory` call): the estimator is a pattern-only
        // Gilbert-Peierls pass costing about as much as a numeric factor, and
        // factor() must not silently pay it - the solvers never measure (or
        // estimate) implicitly.
        let estimate = self.fill.get().map(|_| self.estimate_memory::<T>());
        let t = crate::clock::Instant::now();
        let factors = factor_impl(self, a, settings)?;
        let factor_ms = t.elapsed().as_secs_f64() * 1e3;
        let nnz =
            (factors.l_val.len() + factors.u_val.len() + factors.udiag.len() + factors.f_val.len())
                as u64;
        let entry = (std::mem::size_of::<T>() + std::mem::size_of::<Ki>()) as u64;
        let mut diagnostics = crate::diagnostics::Diagnostics {
            threads: 1,
            n: a.n,
            nnz_a: a.row_idx.len() as u64,
            factor_nnz: nnz,
            estimate,
            decisions: crate::diagnostics::Decisions {
                ordering_requested: "Amd".to_string(),
                ordering_used: "Amd".to_string(),
                scaling: if settings.row_scaling {
                    "RowMaxAbs".to_string()
                } else {
                    "Identity".to_string()
                },
                method: "Klu".to_string(),
                btf_blocks: if settings.btf { self.n_blocks() } else { 0 },
                ..Default::default()
            },
            ..Default::default()
        };
        diagnostics.push(
            "klu-factor",
            factor_ms,
            diagnostics_flops(&diagnostics),
            nnz * entry,
        );
        if crate::logging::enabled(crate::logging::LogLevel::Info) {
            crate::logging::info(&format!("klu factor: {}", diagnostics.summary()));
        }
        Ok(KluSolver {
            factors,
            diagnostics,
            solves: Default::default(),
            solve_threads: crate::numeric::settings::Threads::Fixed(1),
        })
    }
}

/// One matching candidate after SCC + per-block AMD, with the exact
/// Cholesky-lnz score of its ordered symmetrized blocks (only computed in a
/// multi-candidate bakeoff).
struct OrderedForm {
    pre_row_perm: Vec<usize>,
    col_perm: Vec<usize>,
    block_ptr: Vec<usize>,
    score: u64,
}

/// Per-block AMD on the symmetrized block pattern (B + B^T, with diagonal,
/// as the supernodal paths feed rslab-amd), applied
/// symmetrically to the form's permutations. Blocks of size <= 2 have
/// nothing to reorder. With `score_it`, additionally accumulates the exact
/// Cholesky lnz of each AMD-ordered block pattern (Gilbert-Ng-Peyton column
/// counts, near-linear) as the bakeoff score - the trivial blocks are
/// identical across candidates and are skipped consistently.
fn order_blocks<T: Scalar>(
    a: &GeneralCsc<T>,
    form: btf::BtfForm,
    score_it: bool,
) -> Result<OrderedForm, RslabError> {
    let n = a.n;
    let btf::BtfForm {
        row_perm: mut pre_row_perm,
        mut col_perm,
        block_ptr,
    } = form;
    // Narrow inverse permutation: this stage is bound by the random
    // `pinv0[r]` lookups over the matrix entries; 32-bit halves the lookup
    // table's cache footprint (n < 2^31 is enforced at factor time and the
    // KLU design point is far below).
    let mut pinv0 = vec![0 as Ki; n];
    for (k, &r) in pre_row_perm.iter().enumerate() {
        pinv0[r] = k as Ki;
    }
    let mut score = 0u64;
    for b in 0..block_ptr.len() - 1 {
        let (bs, be) = (block_ptr[b], block_ptr[b + 1]);
        let bn = be - bs;
        if bn <= 2 {
            continue;
        }
        // Symmetrized block adjacency (B + B^T + diagonal), canonical form
        // (sorted, deduplicated columns). Built as: the off-diagonal
        // in-block entries B column by column (sequential writes, one random
        // `pinv0` read per entry), a per-column sort of B's short columns,
        // B^T by counting transpose (whose columns come out sorted for
        // free), then a linear three-way sorted merge per column. One
        // random counting/scatter pass over the entries instead of the two
        // of the old both-directions scatter - the random writes, not the
        // short-column sorts, dominate this stage.
        // 1) B: off-diagonal in-block entries, block-local coordinates.
        // Single pass over the matrix (the random `pinv0` reads are this
        // stage's bottleneck - no separate counting pass), sequential
        // pushes, then a short per-column sort.
        let cap: usize = (bs..be)
            .map(|j| {
                let c = col_perm[j];
                a.col_ptr[c + 1] - a.col_ptr[c]
            })
            .sum();
        // The block's off-diagonal pattern by column (`bri`, rows ascending)
        // and by row (`tri`, columns ascending) from counting passes: the
        // input columns are sorted by original row, not by block row, so the
        // transpose is taken twice rather than sorting every column.
        let mut bcol = Vec::with_capacity(bn + 1);
        bcol.push(0usize);
        let mut bri: Vec<i32> = Vec::with_capacity(cap);
        for lj in 0..bn {
            let c = col_perm[bs + lj];
            for &r in &a.row_idx[a.col_ptr[c]..a.col_ptr[c + 1]] {
                let pre = pinv0[r] as usize;
                if pre >= bs && pre < be && pre - bs != lj {
                    bri.push((pre - bs) as i32);
                }
            }
            bcol.push(bri.len());
        }
        let m = bcol[bn];
        let mut tcol = vec![0usize; bn + 1];
        for &li in &bri {
            tcol[li as usize + 1] += 1;
        }
        for j in 0..bn {
            tcol[j + 1] += tcol[j];
        }
        let mut tri = vec![0i32; m];
        {
            let mut cur = tcol[..bn].to_vec();
            for lj in 0..bn {
                for &li in &bri[bcol[lj]..bcol[lj + 1]] {
                    tri[cur[li as usize]] = lj as i32;
                    cur[li as usize] += 1;
                }
            }
        }
        {
            // Transpose back: rows ascending within every column.
            let mut cur = bcol[..bn].to_vec();
            for li in 0..bn {
                for &lj in &tri[tcol[li]..tcol[li + 1]] {
                    bri[cur[lj as usize]] = li as i32;
                    cur[lj as usize] += 1;
                }
            }
        }
        let mut colptr_i32 = Vec::with_capacity(bn + 1);
        let mut rowidx_i32 = Vec::with_capacity(2 * m + bn);
        colptr_i32.push(0i32);
        for lj in 0..bn {
            let (mut p, pe) = (bcol[lj], bcol[lj + 1]);
            let (mut q, qe) = (tcol[lj], tcol[lj + 1]);
            let d = lj as i32;
            let mut d_pending = true;
            let mut last = -1i32;
            while p < pe || q < qe || d_pending {
                let bv = if p < pe { bri[p] } else { i32::MAX };
                let tv = if q < qe { tri[q] } else { i32::MAX };
                let dv = if d_pending { d } else { i32::MAX };
                let v = bv.min(tv).min(dv);
                if v == bv {
                    p += 1;
                } else if v == tv {
                    q += 1;
                } else {
                    d_pending = false;
                }
                if v != last {
                    rowidx_i32.push(v);
                    last = v;
                }
            }
            colptr_i32.push(rowidx_i32.len() as i32);
        }
        let pat = rslab_ordering_core::CscPattern::new(bn, &colptr_i32, &rowidx_i32)
            .ok_or_else(|| RslabError::InvalidInput("klu: malformed block pattern".to_string()))?;
        let lperm = rslab_amd::amd_order(&pat)
            .map_err(|e| RslabError::InvalidInput(format!("klu: AMD ordering failed: {e:?}")))?;
        if score_it {
            // Exact Cholesky lnz of the AMD-ordered block: permute the full
            // symmetric pattern, then etree + GNP column counts. Both accept
            // a full symmetric pattern with unsorted columns (etree uses the
            // upper entries, GNP the lower).
            let mut newpos = vec![0usize; bn];
            for (k, &lp) in lperm.iter().enumerate() {
                newpos[lp as usize] = k;
            }
            let mut pcp = Vec::with_capacity(bn + 1);
            pcp.push(0usize);
            let mut pri = Vec::with_capacity(rowidx_i32.len());
            for &lp in lperm.iter() {
                let lp = lp as usize;
                for &r in &rowidx_i32[colptr_i32[lp] as usize..colptr_i32[lp + 1] as usize] {
                    pri.push(newpos[r as usize]);
                }
                pcp.push(pri.len());
            }
            let pat_p = crate::sparse::csc::CscPattern {
                n: bn,
                col_ptr: pcp,
                row_idx: pri,
            };
            let etree = crate::ordering::elimination_tree::EliminationTree::from_pattern(&pat_p);
            let cc = crate::symbolic::column_counts_gnp(&pat_p, &etree);
            score += crate::symbolic::total_factor_nnz(&cc) as u64;
        }
        // Apply the local (new-to-old) perm symmetrically to the block's
        // segment of both permutations.
        let old_rows: Vec<usize> = pre_row_perm[bs..be].to_vec();
        let old_cols: Vec<usize> = col_perm[bs..be].to_vec();
        for (i, &lp) in lperm.iter().enumerate() {
            pre_row_perm[bs + i] = old_rows[lp as usize];
            col_perm[bs + i] = old_cols[lp as usize];
        }
    }
    Ok(OrderedForm {
        pre_row_perm,
        col_perm,
        block_ptr,
        score,
    })
}
