//! Block Low-Rank (BLR) infrastructure for the multifrontal fronts.
//!
//! Frontal matrices from elliptic-PDE FEM (Helmholtz) and integral-equation MoM
//! near-fields are not low-rank themselves, but their **off-diagonal sub-blocks**
//! (which couple geometrically separated index clusters through a smooth kernel)
//! are numerically low-rank. Compressing those blocks shrinks both the
//! factorization flop count and the front/contribution memory, the two levers
//! behind the PARDISO throughput gap and the transient-memory spike. This module
//! provides the low-rank block type and a pure-Rust rank-revealing compressor;
//! the BLR-aware front factorization builds on it.
//!
//! Compression uses **fully-pivoted Adaptive Cross Approximation (ACA)** - the
//! integral-equation-community standard: a truncated, rank-revealing
//! Gaussian-elimination cross approximation that needs no SVD (which has no
//! pure-Rust complex implementation here). For a dense `m×n` block it is
//! `O(rank·m·n)`, robust, and stops adaptively at a relative Frobenius
//! tolerance.

use crate::scalar::Scalar;

/// A low-rank factorization `B ≈ U · Vᵀ` of a dense `m × n` block, with `U`
/// (`m × rank`) and `V` (`n × rank`) stored column-major. `Vᵀ` (not `Vᴴ`): no
/// conjugation, matching the unconjugated (complex-symmetric / general) algebra
/// the fronts use.
#[derive(Debug, Clone)]
pub struct LowRank<T> {
    pub m: usize,
    pub n: usize,
    pub rank: usize,
    /// `m × rank`, column-major.
    pub u: Vec<T>,
    /// `n × rank`, column-major.
    pub v: Vec<T>,
}

impl<T: Scalar> LowRank<T> {
    /// Stored entries `rank·(m + n)` vs the dense `m·n` - the compression only
    /// pays off when this is smaller.
    pub fn storage(&self) -> usize {
        self.rank * (self.m + self.n)
    }

    /// Reconstruct the dense approximation `U·Vᵀ` (column-major `m × n`). For
    /// tests / diagnostics; the factorization never densifies a compressed block.
    pub fn to_dense(&self) -> Vec<T> {
        let mut d = vec![T::zero(); self.m * self.n];
        for k in 0..self.rank {
            let uk = &self.u[k * self.m..k * self.m + self.m];
            let vk = &self.v[k * self.n..k * self.n + self.n];
            for j in 0..self.n {
                let vj = vk[j];
                if vj != T::zero() {
                    let col = &mut d[j * self.m..j * self.m + self.m];
                    for i in 0..self.m {
                        col[i] = col[i] + uk[i] * vj;
                    }
                }
            }
        }
        d
    }
}

/// Frobenius norm of a dense column-major buffer.
fn frob<T: Scalar>(a: &[T]) -> f64 {
    a.iter().map(|x| x.magnitude_sq()).sum::<f64>().sqrt()
}

/// Fully-pivoted ACA compression of a dense `m × n` block `b` (column-major) to
/// relative Frobenius tolerance `eps`. Caps the rank at `max_rank`.
///
/// Returns the low-rank factors and the achieved rank. If the block does not
/// compress below `eps` within `max_rank` cross steps, `rank == max_rank`
/// (`min(m, n)` cap) and the caller should treat it as not worth compressing.
///
/// Algorithm: repeatedly pick the largest-magnitude residual entry `(i*, j*)`,
/// peel off the rank-1 cross `R[:,j*] · R[i*,:] / R[i*,j*]` (Schur step of
/// Gaussian elimination with full pivoting), and stop when the residual
/// Frobenius norm falls to `eps · ‖B‖_F`.
pub fn compress_aca<T: Scalar>(
    b: &[T],
    m: usize,
    n: usize,
    eps: f64,
    max_rank: usize,
) -> LowRank<T> {
    let cap = max_rank.min(m).min(n);
    let bnorm = frob(b);
    let mut r = b.to_vec(); // residual, column-major m×n
    let mut u: Vec<T> = Vec::with_capacity(m * cap);
    let mut v: Vec<T> = Vec::with_capacity(n * cap);
    let mut rank = 0usize;

    // A structurally-zero block compresses to rank 0.
    if bnorm == 0.0 || cap == 0 {
        return LowRank {
            m,
            n,
            rank: 0,
            u,
            v,
        };
    }
    let tol = eps * bnorm;

    while rank < cap {
        // Full-pivot search: largest |R[i,j]|.
        let mut bi = 0usize;
        let mut bj = 0usize;
        let mut bmag = 0.0f64;
        for j in 0..n {
            let col = &r[j * m..j * m + m];
            for (i, &val) in col.iter().enumerate() {
                let mag = val.magnitude();
                if mag > bmag {
                    bmag = mag;
                    bi = i;
                    bj = j;
                }
            }
        }
        if bmag == 0.0 {
            break; // residual exactly zero - exact low-rank
        }
        let pinv = r[bj * m + bi].recip();
        // Cross factors: u = R[:, bj], v = R[bi, :]·(1/pivot).
        let uk: Vec<T> = (0..m).map(|i| r[bj * m + i]).collect();
        let vk: Vec<T> = (0..n).map(|j| r[j * m + bi] * pinv).collect();
        // Schur update R -= u ⊗ v.
        for j in 0..n {
            let vj = vk[j];
            if vj != T::zero() {
                let col = &mut r[j * m..j * m + m];
                for i in 0..m {
                    col[i] = col[i] - uk[i] * vj;
                }
            }
        }
        u.extend_from_slice(&uk);
        v.extend_from_slice(&vk);
        rank += 1;
        if frob(&r) <= tol {
            break;
        }
    }
    LowRank { m, n, rank, u, v }
}

/// One tile of a block-partitioned matrix: stored either dense (column-major,
/// `rows × cols`) or in compressed low-rank form `U·Vᵀ`. Diagonal tiles and
/// off-diagonal tiles that do not compress below the break-even rank stay
/// `Dense`; admissible (geometrically separated, smooth-kernel) off-diagonal
/// tiles become `LowRank`. The BLR factorization dispatches on this enum: dense
/// tiles take the ordinary kernels, low-rank tiles take the cheap
/// `U(VᵀU)Vᵀ`-style block arithmetic.
#[derive(Debug, Clone)]
pub enum Block<T> {
    /// Column-major `rows × cols` dense tile.
    Dense {
        rows: usize,
        cols: usize,
        data: Vec<T>,
    },
    /// Compressed tile `U·Vᵀ`.
    LowRank(LowRank<T>),
}

impl<T: Scalar> Block<T> {
    /// Stored scalar count - `rows·cols` dense, `rank·(rows+cols)` low-rank.
    pub fn storage(&self) -> usize {
        match self {
            Block::Dense { rows, cols, .. } => rows * cols,
            Block::LowRank(lr) => lr.storage(),
        }
    }

    // Test-only tile-kind probe (the BLR partition tests assert diagonal tiles
    // stay dense). Not on the production path, so `cfg(test)` keeps it out of
    // release builds rather than carrying a dead-code allow.
    #[cfg(test)]
    pub fn is_low_rank(&self) -> bool {
        matches!(self, Block::LowRank(_))
    }

    /// Dense `rows × cols` column-major reconstruction of the tile.
    pub fn to_dense(&self) -> Vec<T> {
        match self {
            Block::Dense { data, .. } => data.clone(),
            Block::LowRank(lr) => lr.to_dense(),
        }
    }
}

/// Extent `(start, len)` of block index `k` along an axis of length `full`
/// partitioned into tiles of size `b` (the trailing tile may be shorter).
fn block_extent(full: usize, b: usize, k: usize) -> (usize, usize) {
    let start = k * b;
    (start, b.min(full - start))
}

/// A block-partitioned (Block Low-Rank) representation of a dense column-major
/// `nrow × ncol` matrix: a `nbr × nbc` grid of [`Block`]s over a fixed block
/// size `b` (the trailing row/column tile may be smaller). The grid is stored
/// row-major (`blocks[ib·nbc + jb]`).
///
/// This is the data structure the BLR-aware front factorization operates on:
/// Stage 1 builds and reconstructs it; later stages factor it block-by-block
/// with low-rank tile arithmetic.
#[derive(Debug, Clone)]
pub struct BlrMatrix<T> {
    pub nrow: usize,
    pub ncol: usize,
    pub b: usize,
    pub nbr: usize,
    pub nbc: usize,
    blocks: Vec<Block<T>>,
}

impl<T: Scalar> BlrMatrix<T> {
    fn idx(&self, ib: usize, jb: usize) -> usize {
        ib * self.nbc + jb
    }

    pub fn block(&self, ib: usize, jb: usize) -> &Block<T> {
        &self.blocks[self.idx(ib, jb)]
    }

    /// Row extent `(start, len)` of block-row `ib`.
    pub fn row_extent(&self, ib: usize) -> (usize, usize) {
        block_extent(self.nrow, self.b, ib)
    }

    /// Column extent `(start, len)` of block-column `jb`.
    pub fn col_extent(&self, jb: usize) -> (usize, usize) {
        block_extent(self.ncol, self.b, jb)
    }

    /// Total stored scalars across all tiles - the BLR memory footprint.
    pub fn storage(&self) -> usize {
        self.blocks.iter().map(Block::storage).sum()
    }

    /// Footprint of the equivalent dense matrix, `nrow·ncol`.
    // Test-only: the compression tests assert BLR beats dense storage. Off the
    // production path, so `cfg(test)` keeps it out of release builds.
    #[cfg(test)]
    pub fn dense_storage(&self) -> usize {
        self.nrow * self.ncol
    }

    /// Number of off-diagonal tiles stored compressed.
    // Test-only: asserts some off-diagonal tiles actually compressed.
    #[cfg(test)]
    pub fn n_low_rank(&self) -> usize {
        self.blocks.iter().filter(|b| b.is_low_rank()).count()
    }

    /// Build a BLR partition of the dense column-major `nrow × ncol` matrix `a`
    /// at block size `b` and per-tile relative Frobenius tolerance `eps`.
    ///
    /// Diagonal tiles (`ib == jb`) are always dense. Each off-diagonal tile is
    /// compressed by ACA, capped at the **break-even rank**
    /// `⌊rows·cols/(rows+cols)⌋` (the rank above which `U·Vᵀ` stores no less than
    /// the dense tile): a tile that reaches that cap without converging is kept
    /// dense, which also bounds the ACA cost on incompressible near-diagonal
    /// tiles to `O(breakeven · rows·cols)`.
    pub fn from_dense(a: &[T], nrow: usize, ncol: usize, b: usize, eps: f64) -> BlrMatrix<T> {
        assert!(b > 0, "block size must be positive");
        let nbr = nrow.div_ceil(b);
        let nbc = ncol.div_ceil(b);
        let mut blocks = Vec::with_capacity(nbr * nbc);
        for ib in 0..nbr {
            let (r0, bm) = block_extent(nrow, b, ib);
            for jb in 0..nbc {
                let (c0, bn) = block_extent(ncol, b, jb);
                // Extract the tile column-major from `a` (col stride `nrow`).
                let mut tile = vec![T::zero(); bm * bn];
                for jj in 0..bn {
                    let src = (c0 + jj) * nrow + r0;
                    tile[jj * bm..jj * bm + bm].copy_from_slice(&a[src..src + bm]);
                }
                let blk = if ib == jb {
                    Block::Dense {
                        rows: bm,
                        cols: bn,
                        data: tile,
                    }
                } else {
                    let breakeven = (bm * bn / (bm + bn)).max(1);
                    let lr = compress_aca(&tile, bm, bn, eps, breakeven);
                    if lr.rank < breakeven && lr.storage() < bm * bn {
                        Block::LowRank(lr)
                    } else {
                        Block::Dense {
                            rows: bm,
                            cols: bn,
                            data: tile,
                        }
                    }
                };
                blocks.push(blk);
            }
        }
        BlrMatrix {
            nrow,
            ncol,
            b,
            nbr,
            nbc,
            blocks,
        }
    }

    /// Reconstruct the dense column-major `nrow × ncol` approximation by writing
    /// every tile into place. Test-only reconstruction check for the compression
    /// path; production never densifies the whole matrix (it consumes tiles via
    /// [`Block::to_dense`]). `cfg(test)` keeps it out of release builds.
    #[cfg(test)]
    pub fn to_dense(&self) -> Vec<T> {
        let mut out = vec![T::zero(); self.nrow * self.ncol];
        for ib in 0..self.nbr {
            let (r0, bm) = self.row_extent(ib);
            for jb in 0..self.nbc {
                let (c0, bn) = self.col_extent(jb);
                let tile = self.block(ib, jb).to_dense();
                for jj in 0..bn {
                    let dst = (c0 + jj) * self.nrow + r0;
                    out[dst..dst + bm].copy_from_slice(&tile[jj * bm..jj * bm + bm]);
                }
            }
        }
        out
    }
}

/// Diagnostic: partition a dense front `f` (`n × n` column-major, of which the
/// leading `ncol` columns are eliminated) into `b × b` blocks and report how
/// compressible its strictly-lower-triangle off-diagonal blocks are at several
/// Frobenius tolerances. This is the empirical BLR-benefit estimate - mean rank
/// and compressed-vs-dense storage of the off-diagonal blocks that BLR would
/// represent in low-rank form. Prints to stderr; gated by the caller.
pub fn probe_front<T: Scalar>(f: &[T], n: usize, ncol: usize, b: usize) {
    let nb = n.div_ceil(b);
    for &eps in &[1e-2f64, 1e-4, 1e-8] {
        let mut dense = 0usize;
        let mut comp = 0usize;
        let mut sumrank = 0usize;
        let mut nblk = 0usize;
        for jb in 0..nb {
            let j0 = jb * b;
            let jn = (j0 + b).min(n);
            for ib in (jb + 1)..nb {
                let i0 = ib * b;
                let im = (i0 + b).min(n);
                let (bm, bn) = (im - i0, jn - j0);
                let mut blk = vec![T::zero(); bm * bn];
                for jj in 0..bn {
                    for ii in 0..bm {
                        blk[jj * bm + ii] = f[(j0 + jj) * n + (i0 + ii)];
                    }
                }
                let lr = compress_aca(&blk, bm, bn, eps, bm.min(bn));
                dense += bm * bn;
                comp += lr.storage();
                sumrank += lr.rank;
                nblk += 1;
            }
        }
        if nblk > 0 {
            eprintln!(
                "[BLR_PROBE] n={n} ncol={ncol} b={b} eps={eps:.0e}: off-diag-blocks={nblk} \
                 mean-rank={:.1}/{b} compressed={:.0}% of dense",
                sumrank as f64 / nblk as f64,
                100.0 * comp as f64 / dense as f64,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    fn max_abs_diff<T: Scalar>(a: &[T], b: &[T]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x - *y).magnitude())
            .fold(0.0, f64::max)
    }

    #[test]
    fn aca_recovers_exact_low_rank() {
        // Build a rank-3 block U0·V0ᵀ (m=40, n=30) and check ACA recovers it at a
        // tiny tolerance with rank ≤ 3.
        let (m, n, r0) = (40usize, 30usize, 3usize);
        let u0: Vec<f64> = (0..m * r0).map(|t| (t * 7 % 11) as f64 - 5.0).collect();
        let v0: Vec<f64> = (0..n * r0).map(|t| (t * 5 % 13) as f64 - 6.0).collect();
        let mut b = vec![0.0f64; m * n];
        for k in 0..r0 {
            for j in 0..n {
                for i in 0..m {
                    b[j * m + i] += u0[k * m + i] * v0[k * n + j];
                }
            }
        }
        let lr = compress_aca(&b, m, n, 1e-12, m.min(n));
        assert!(lr.rank <= r0, "rank {} should be ≤ {r0}", lr.rank);
        assert!(max_abs_diff(&b, &lr.to_dense()) < 1e-9 * frob(&b));
        assert!(lr.storage() < m * n, "must actually compress");
    }

    #[test]
    fn aca_compresses_smooth_kernel_block() {
        // A smooth-kernel off-diagonal block (1/(2 + |i−j|) type, separated
        // clusters) - the EM near-field analogue - must compress to low rank at a
        // loose preconditioner tolerance.
        let (m, n) = (64usize, 64usize);
        let mut b = vec![Complex::new(0.0, 0.0); m * n];
        for j in 0..n {
            for i in 0..m {
                // rows and cols sit in well-separated 1-D clusters.
                let xi = i as f64;
                let xj = 200.0 + j as f64;
                let d = xj - xi;
                b[j * m + i] = Complex::new(1.0 / d, 0.5 / (d * d));
            }
        }
        let lr = compress_aca(&b, m, n, 1e-6, m.min(n));
        assert!(
            lr.rank <= 8,
            "smooth separated block should be very low rank, got {}",
            lr.rank
        );
        let err = max_abs_diff(&b, &lr.to_dense());
        assert!(err <= 1e-5 * frob(&b), "reconstruction error {err}");
    }

    /// A front-like dense matrix: strong diagonal, off-diagonal coupling through
    /// a smooth separated-cluster kernel - compressible away from the diagonal.
    fn smooth_front(n: usize) -> Vec<Complex<f64>> {
        let mut a = vec![Complex::new(0.0, 0.0); n * n];
        for j in 0..n {
            for i in 0..n {
                a[j * n + i] = if i == j {
                    Complex::new(n as f64, 1.0)
                } else {
                    // Smooth kernel in the index separation; rows/cols of a tile
                    // far from the diagonal sit in well-separated clusters.
                    let d = (i as f64 - j as f64).abs() + 1.0;
                    Complex::new(1.0 / d, 0.5 / (d * d))
                };
            }
        }
        a
    }

    #[test]
    fn blr_roundtrip_within_tolerance() {
        // BLR partition of a smooth front must reconstruct within the per-tile
        // tolerance and actually store fewer scalars than dense.
        let (n, b, eps) = (256usize, 64usize, 1e-4);
        let a = smooth_front(n);
        let blr = BlrMatrix::from_dense(&a, n, n, b, eps);
        let recon = blr.to_dense();
        let err = max_abs_diff(&a, &recon);
        // Loose bound: per-tile relative eps accumulates mildly across the grid.
        assert!(
            err <= 1e-2 * frob(&a),
            "reconstruction error {err} vs ‖A‖={}",
            frob(&a)
        );
        assert!(
            blr.storage() < blr.dense_storage(),
            "BLR storage {} should beat dense {}",
            blr.storage(),
            blr.dense_storage()
        );
        assert!(blr.n_low_rank() > 0, "some off-diag tiles should compress");
    }

    #[test]
    fn blr_diagonal_tiles_stay_dense() {
        let (n, b) = (128usize, 32usize);
        let a = smooth_front(n);
        let blr = BlrMatrix::from_dense(&a, n, n, b, 1e-6);
        for k in 0..blr.nbr.min(blr.nbc) {
            assert!(
                !blr.block(k, k).is_low_rank(),
                "diagonal tile ({k},{k}) must stay dense"
            );
        }
    }

    #[test]
    fn blr_ragged_partition_reconstructs() {
        // Block size that does not divide the dimension → trailing short tiles.
        let (m, n, b) = (100usize, 70usize, 32usize);
        let mut a = vec![0.0f64; m * n];
        for j in 0..n {
            for i in 0..m {
                a[j * m + i] = ((i * 13 + j * 7) % 17) as f64;
            }
        }
        let blr = BlrMatrix::from_dense(&a, m, n, b, 1e-12);
        assert_eq!(blr.nbr, 4);
        assert_eq!(blr.nbc, 3);
        let recon = blr.to_dense();
        assert!(max_abs_diff(&a, &recon) < 1e-9 * frob(&a).max(1.0));
    }

    /// Strongly diagonally dominant complex matrix (block-local pivoting is
    /// stable, no equilibration needed) - off-diagonal blocks are full-rank
    /// random and stay dense at tight tolerance.
    #[test]
    fn aca_full_rank_block_does_not_falsely_compress() {
        // A diagonal-dominant random-ish block has no low-rank structure: ACA at
        // tight tolerance should need (near) full rank.
        let (m, n) = (24usize, 24usize);
        let mut b = vec![0.0f64; m * n];
        for j in 0..n {
            for i in 0..m {
                b[j * m + i] = if i == j {
                    100.0
                } else {
                    (((i * 31 + j * 17) % 7) as f64) - 3.0
                };
            }
        }
        let lr = compress_aca(&b, m, n, 1e-12, m.min(n));
        assert!(lr.rank >= m - 2, "full-rank block, got rank {}", lr.rank);
    }
}
