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
//! pure-Rust complex implementation here). For a dense `mxn` block it is
//! `O(rank*m*n)`, robust, and stops adaptively at a relative Frobenius
//! tolerance.

use crate::scalar::Scalar;

/// A low-rank factorization `B ~ U * V^T` of a dense `m x n` block, with `U`
/// (`m x rank`) and `V` (`n x rank`) stored column-major. `V^T` (not `V^H`): no
/// conjugation, matching the unconjugated (complex-symmetric / general) algebra
/// the fronts use.
#[derive(Debug, Clone)]
pub struct LowRank<T: Scalar> {
    pub m: usize,
    pub n: usize,
    /// Total rank: full-precision crosses `[0, rank)` in `u`/`v` plus the
    /// low-precision tail `[rank, rank + rank_lo)` in `u_lo`/`v_lo`.
    pub rank: usize,
    /// `m x rank`, column-major.
    pub u: Vec<T>,
    /// `n x rank`, column-major.
    pub v: Vec<T>,
    /// Adaptive-precision tail (issue #19): trailing crosses whose
    /// contribution is small enough that storing them at the `T::Lo`
    /// roundoff stays below the compression tolerance - half the bytes
    /// per entry when `T::LO_SHRINKS`. Empty for plain compression.
    pub rank_lo: usize,
    /// `m x rank_lo`, column-major, low precision.
    pub u_lo: Vec<T::Lo>,
    /// `n x rank_lo`, column-major, low precision.
    pub v_lo: Vec<T::Lo>,
}

impl<T: Scalar> LowRank<T> {
    /// Stored size in full-precision-entry equivalents: full crosses count
    /// 1 each, low-precision tail entries count 1/2 (when `T::Lo` shrinks).
    /// Comparable against the dense `m*n`.
    pub fn storage(&self) -> usize {
        let lo = self.rank_lo * (self.m + self.n);
        let lo_eq = if T::LO_SHRINKS { lo.div_ceil(2) } else { lo };
        self.rank * (self.m + self.n) + lo_eq
    }

    /// Reconstruct the dense approximation `U*V^T` (column-major `m x n`). For
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
        for k in 0..self.rank_lo {
            let uk = &self.u_lo[k * self.m..k * self.m + self.m];
            let vk = &self.v_lo[k * self.n..k * self.n + self.n];
            for j in 0..self.n {
                let vj = T::promote(vk[j]);
                if vj != T::zero() {
                    let col = &mut d[j * self.m..j * self.m + self.m];
                    for i in 0..self.m {
                        col[i] = col[i] + T::promote(uk[i]) * vj;
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

/// Fully-pivoted ACA compression of a dense `m x n` block `b` (column-major) to
/// relative Frobenius tolerance `eps`. Caps the rank at `max_rank`.
///
/// Returns the low-rank factors and the achieved rank. If the block does not
/// compress below `eps` within `max_rank` cross steps, `rank == max_rank`
/// (`min(m, n)` cap) and the caller should treat it as not worth compressing.
///
/// Algorithm: repeatedly pick the largest-magnitude residual entry `(i*, j*)`,
/// peel off the rank-1 cross `R[:,j*] * R[i*,:] / R[i*,j*]` (Schur step of
/// Gaussian elimination with full pivoting), and stop when the residual
/// Frobenius norm falls to `eps * ||B||_F`.
#[cfg(test)]
pub fn compress_aca<T: Scalar>(
    b: &[T],
    m: usize,
    n: usize,
    eps: f64,
    max_rank: usize,
) -> LowRank<T> {
    compress_aca_impl(b, m, n, eps, max_rank, false)
}

/// [`compress_aca`] with the **adaptive-precision tail** (issue #19,
/// Amestoy/Buttari/Higham/Mary adapted to ACA crosses): after compression,
/// the trailing crosses whose weight `w_k = ||u_k||_2*||v_k||_2` satisfies
/// `w_k * eps_lo <= eps * ||B||_F / rank` are stored in `T::Lo` - their
/// storage-rounding noise (`w_k * eps_lo` each, `rank` of them at most)
/// stays below the compression tolerance already accepted, so the overall
/// approximation quality class is unchanged at half the bytes per tail
/// entry. The split is a prefix rule (all crosses from the first qualifying
/// index onward), matching the roughly decreasing ACA weights.
#[allow(dead_code)] // crate-internal API: exercised by tests; production enters via BlrMode/from_dense_with
pub fn compress_aca_adaptive<T: Scalar>(
    b: &[T],
    m: usize,
    n: usize,
    eps: f64,
    max_rank: usize,
) -> LowRank<T> {
    compress_aca_impl(b, m, n, eps, max_rank, true)
}

fn compress_aca_impl<T: Scalar>(
    b: &[T],
    m: usize,
    n: usize,
    eps: f64,
    max_rank: usize,
    lo_tail: bool,
) -> LowRank<T> {
    let cap = max_rank.min(m).min(n);
    let bnorm = frob(b);
    let mut r = b.to_vec(); // residual, column-major mxn
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
            rank_lo: 0,
            u_lo: Vec::new(),
            v_lo: Vec::new(),
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
        // Cross factors: u = R[:, bj], v = R[bi, :]*(1/pivot).
        let uk: Vec<T> = (0..m).map(|i| r[bj * m + i]).collect();
        let vk: Vec<T> = (0..n).map(|j| r[j * m + bi] * pinv).collect();
        // Schur update R -= u kron v.
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
    if !lo_tail || !T::LO_SHRINKS || rank == 0 {
        return LowRank {
            m,
            n,
            rank,
            u,
            v,
            rank_lo: 0,
            u_lo: Vec::new(),
            v_lo: Vec::new(),
        };
    }
    // Adaptive tail: demote the prefix-maximal trailing cross set whose
    // per-cross storage-rounding noise stays below the accepted tolerance.
    let cross_w = |k: usize| -> f64 {
        let un = frob(&u[k * m..k * m + m]);
        let vn = frob(&v[k * n..k * n + n]);
        un * vn
    };
    let budget = eps.max(f64::EPSILON) * bnorm / rank as f64;
    let mut k0 = rank;
    while k0 > 0 && cross_w(k0 - 1) * T::EPS_LO <= budget {
        k0 -= 1;
    }
    let rank_lo = rank - k0;
    let u_lo: Vec<T::Lo> = u[k0 * m..].iter().map(|&x| x.demote()).collect();
    let v_lo: Vec<T::Lo> = v[k0 * n..].iter().map(|&x| x.demote()).collect();
    u.truncate(k0 * m);
    v.truncate(k0 * n);
    LowRank {
        m,
        n,
        rank: k0,
        u,
        v,
        rank_lo,
        u_lo,
        v_lo,
    }
}

/// One tile of a block-partitioned matrix: stored either dense (column-major,
/// `rows x cols`) or in compressed low-rank form `U*V^T`. Diagonal tiles and
/// off-diagonal tiles that do not compress below the break-even rank stay
/// `Dense`; admissible (geometrically separated, smooth-kernel) off-diagonal
/// tiles become `LowRank`. The BLR factorization dispatches on this enum: dense
/// tiles take the ordinary kernels, low-rank tiles take the cheap
/// `U(V^TU)V^T`-style block arithmetic.
#[derive(Debug, Clone)]
pub enum Block<T: Scalar> {
    /// Column-major `rows x cols` dense tile. The shape fields are read only
    /// by the test-only `storage` accounting; production consumes `data` with
    /// the extent coming from the partition.
    Dense {
        #[cfg_attr(not(test), allow(dead_code))]
        rows: usize,
        #[cfg_attr(not(test), allow(dead_code))]
        cols: usize,
        data: Vec<T>,
    },
    /// Compressed tile `U*V^T`.
    LowRank(LowRank<T>),
}

impl<T: Scalar> Block<T> {
    /// Stored scalar count - `rows*cols` dense, `rank*(rows+cols)` low-rank.
    // Test-only: the compression tests assert BLR beats dense storage.
    #[cfg(test)]
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

    /// Dense `rows x cols` column-major reconstruction of the tile.
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
/// `nrow x ncol` matrix: a `nbr x nbc` grid of [`Block`]s over a fixed block
/// size `b` (the trailing row/column tile may be smaller). The grid is stored
/// row-major (`blocks[ib*nbc + jb]`).
///
/// This is the data structure the BLR-aware front factorization operates on:
/// Stage 1 builds and reconstructs it; later stages factor it block-by-block
/// with low-rank tile arithmetic.
#[derive(Debug, Clone)]
pub struct BlrMatrix<T: Scalar> {
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
    // Test-only: the compression tests assert BLR beats dense storage.
    #[cfg(test)]
    pub fn storage(&self) -> usize {
        self.blocks.iter().map(Block::storage).sum()
    }

    /// Footprint of the equivalent dense matrix, `nrow*ncol`.
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

    /// Build a BLR partition of the dense column-major `nrow x ncol` matrix `a`
    /// at block size `b` and per-tile relative Frobenius tolerance `eps`.
    ///
    /// Diagonal tiles (`ib == jb`) are always dense. Each off-diagonal tile is
    /// compressed by ACA, capped at the **break-even rank**
    /// `floor(rows*cols/(rows+cols))` (the rank above which `U*V^T` stores no less than
    /// the dense tile): a tile that reaches that cap without converging is kept
    /// dense, which also bounds the ACA cost on incompressible near-diagonal
    /// tiles to `O(breakeven * rows*cols)`.
    #[allow(dead_code)] // stable convenience wrapper; production enters via from_dense_with
    pub fn from_dense(a: &[T], nrow: usize, ncol: usize, b: usize, eps: f64) -> BlrMatrix<T> {
        Self::from_dense_with(a, nrow, ncol, b, eps, false)
    }

    /// [`from_dense`](Self::from_dense) with the adaptive-precision tail
    /// toggle: `adaptive = true` stores each compressed tile's small
    /// trailing crosses in `T::Lo` (see [`compress_aca_adaptive`]).
    pub fn from_dense_with(
        a: &[T],
        nrow: usize,
        ncol: usize,
        b: usize,
        eps: f64,
        adaptive: bool,
    ) -> BlrMatrix<T> {
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
                    let lr = compress_aca_impl(&tile, bm, bn, eps, breakeven, adaptive);
                    if lr.rank + lr.rank_lo < breakeven && lr.storage() < bm * bn {
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

    /// Reconstruct the dense column-major `nrow x ncol` approximation by writing
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
        // Build a rank-3 block U0*V0^T (m=40, n=30) and check ACA recovers it at a
        // tiny tolerance with rank <= 3.
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
        assert!(lr.rank <= r0, "rank {} should be <= {r0}", lr.rank);
        assert!(max_abs_diff(&b, &lr.to_dense()) < 1e-9 * frob(&b));
        assert!(lr.storage() < m * n, "must actually compress");
    }

    #[test]
    fn aca_compresses_smooth_kernel_block() {
        // A smooth-kernel off-diagonal block (1/(2 + |i-j|) type, separated
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
            "reconstruction error {err} vs ||A||={}",
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
        // Block size that does not divide the dimension -> trailing short tiles.
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

    /// Smooth kernel `1/(1 + |i-j|)`: fast-decaying cross weights, the
    /// adaptive tail must (a) keep the reconstruction inside the accepted
    /// tolerance class and (b) actually shrink the stored size.
    #[test]
    fn adaptive_tail_preserves_quality_and_shrinks_storage() {
        let (m, n) = (96usize, 96usize);
        let mut b = vec![0.0f64; m * n];
        for j in 0..n {
            for i in 0..m {
                b[j * m + i] = 1.0 / (1.0 + (i as f64 - (j + 200) as f64).abs());
            }
        }
        let eps = 1e-8;
        let bnorm = frob(&b);
        let plain = compress_aca(&b, m, n, eps, m.min(n));
        let adap = compress_aca_adaptive(&b, m, n, eps, m.min(n));
        assert_eq!(
            plain.rank,
            adap.rank + adap.rank_lo,
            "same total rank, split into hi+lo"
        );
        assert!(
            adap.rank_lo > 0,
            "decaying weights must demote a tail (rank {}, lo {})",
            adap.rank,
            adap.rank_lo
        );
        assert!(
            adap.storage() < plain.storage(),
            "adaptive must store less ({} vs {})",
            adap.storage(),
            plain.storage()
        );
        // Reconstruction quality stays in the accepted class: the tail's
        // rounding noise is budgeted below eps, allow a small constant.
        let d = adap.to_dense();
        let err = d
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        assert!(
            err <= 4.0 * eps * bnorm,
            "adaptive reconstruction error {err:.3e} vs budget {:.3e}",
            4.0 * eps * bnorm
        );
    }

    /// For an f32 matrix `Lo` is the identity: the adaptive path must be a
    /// no-op (no demotion, no storage change) rather than pretending a win.
    #[test]
    fn adaptive_tail_is_noop_for_single_precision() {
        let (m, n) = (32usize, 32usize);
        let mut b = vec![0.0f32; m * n];
        for j in 0..n {
            for i in 0..m {
                b[j * m + i] = 1.0 / (1.0 + (i as f64 - (j + 80) as f64).abs()) as f32;
            }
        }
        let adap = compress_aca_adaptive(&b, m, n, 1e-5, m.min(n));
        assert_eq!(adap.rank_lo, 0, "identity Lo must not split");
    }
}
