//! Structural feature extraction - the "structure analyzer".
//!
//! Distils a sparse matrix plus its symbolic analysis into a compact,
//! value-light [`StructuralFeatures`] vector. Two uses:
//!
//! * **Diagnostics**: a one-glance summary of *why* a matrix factors the way it
//!   does - fill, supernode shape, tree parallelism, where the work concentrates.
//! * **Auto-tuning input**: the feature vector is the input to a parameter
//!   predictor (heuristic rules today, a small offline-trained model later) that
//!   chooses solver knobs - method, threads, amalgamation, GEMM thresholds - per
//!   system. It is deterministic and reproducible, derived purely from the
//!   pattern and the pre-solve symbolic analysis.
//!
//! Features split into two groups by cost:
//! * **pattern features** are `O(nnz)` and need only the matrix (available before
//!   analysis): size, degree distribution, bandwidth, diagonal dominance.
//! * **symbolic features** come from the [`MultifrontalSymbolic`] analysis (fill,
//!   supernodes, assembly-tree depth/width, front-size and flop concentration).

use crate::numeric::multifrontal_ldlt::MultifrontalSymbolic;
use crate::numeric::multifrontal_lu::LuSymbolic;
use crate::numeric::sparse_solver::LdltSymbolic;
use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;
use serde::{Deserialize, Serialize};

/// The assembly-tree shape an analysis exposes, the symbolic input to feature
/// extraction. Implemented by the high-level [`LdltSymbolic`] / [`LuSymbolic`]
/// (pass the analysis you already built) and the low-level
/// [`MultifrontalSymbolic`]. For an unsymmetric matrix use [`LuSymbolic`]: its
/// shape reflects the symmetrized `A ∪ Aᵀ` pattern the LU actually factors.
pub trait SymbolicShape {
    /// Per-supernode `(ncol, nrow)` frontal dimensions.
    fn front_dims(&self) -> Vec<(usize, usize)>;
    /// Supernode count per assembly-tree level (leaves first).
    fn level_widths(&self) -> Vec<usize>;
}

impl SymbolicShape for MultifrontalSymbolic {
    fn front_dims(&self) -> Vec<(usize, usize)> {
        MultifrontalSymbolic::front_dims(self)
    }
    fn level_widths(&self) -> Vec<usize> {
        MultifrontalSymbolic::level_widths(self)
    }
}

impl SymbolicShape for LdltSymbolic {
    fn front_dims(&self) -> Vec<(usize, usize)> {
        LdltSymbolic::front_dims(self)
    }
    fn level_widths(&self) -> Vec<usize> {
        LdltSymbolic::level_widths(self)
    }
}

impl SymbolicShape for LuSymbolic {
    fn front_dims(&self) -> Vec<(usize, usize)> {
        LuSymbolic::front_dims(self)
    }
    fn level_widths(&self) -> Vec<usize> {
        LuSymbolic::level_widths(self)
    }
}

/// The data-driven single-solve thread-count policy, as a free function over the
/// three predictive features (so the factor path can apply it straight from the
/// symbolic analysis without building a full [`StructuralFeatures`]). Returns a
/// worker count in `1..=max_cores`. See [`StructuralFeatures::recommend_threads`]
/// for the derivation and validation.
pub fn recommend_threads_from(
    factor_flops: u64,
    front_nrow_max: usize,
    tree_width_max: usize,
    max_cores: usize,
) -> usize {
    let cores = max_cores.max(1);
    // Thin fronts + narrow tree: no node-parallelism (tiny fronts) and no
    // tree-parallelism (path-like) to exploit - oversubscription only hurts.
    if front_nrow_max < 512 && tree_width_max < 128 {
        return cores.min(2);
    }
    // Tiny total work: parallel scheduling overhead dominates the factorization.
    if factor_flops < 300_000_000 {
        return cores.min(4);
    }
    cores
}

/// A compact structural fingerprint of a sparse system: cheap pattern statistics
/// plus the symbolic-analysis shape. Serializable, so a sweep harness can emit
/// one record per matrix and a predictor can consume the same vector.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StructuralFeatures {
    // --- pattern features (O(nnz), pre-analysis) ---
    /// Matrix dimension `n`.
    pub n: usize,
    /// Stored nonzeros (lower triangle for symmetric input, full for general).
    pub nnz: usize,
    /// Mean stored entries per column.
    pub deg_mean: f64,
    /// Maximum stored entries in any column.
    pub deg_max: usize,
    /// Coefficient of variation of the column degree (`std/mean`) - pattern
    /// irregularity. Near 0 = uniform (stencil); large = a few dense columns
    /// (arrow / border).
    pub deg_cv: f64,
    /// Maximum half-bandwidth `max |i - j|` over stored entries (locality).
    pub bandwidth_max: usize,
    /// Mean `|i - j|` over off-diagonal stored entries, normalized by `n`
    /// (`0` = banded/local, toward `1` = long-range coupling).
    pub bandwidth_mean_rel: f64,
    /// Fraction of columns whose diagonal magnitude dominates the off-diagonal
    /// column sum (`|a_jj| >= sum_{i!=j} |a_ij|`) - a cheap conditioning proxy.
    pub diag_dominant_frac: f64,
    /// Fraction of columns carrying an explicit diagonal entry.
    pub diag_present_frac: f64,

    // --- symbolic features (post-analysis) ---
    /// Number of supernodes (fronts) in the assembly tree.
    pub n_supernodes: usize,
    /// Estimated factor fill `nnz(L)` from the symbolic front structure.
    pub fill_nnz: u64,
    /// Fill ratio `fill_nnz / nnz` - how much the factor grows over the input.
    pub fill_ratio: f64,
    /// Mean eliminated columns per supernode (amalgamation effectiveness).
    pub supernode_cols_mean: f64,
    /// Largest front height `nrow` (the dense bottleneck near the root).
    pub front_nrow_max: usize,
    /// Assembly-tree depth (level-parallel factorization length).
    pub tree_depth: usize,
    /// Peak tree width: most independent fronts at any one level (max
    /// tree-parallelism available).
    pub tree_width_max: usize,
    /// Mean tree width across levels (average independent work per level).
    pub tree_width_mean: f64,
    /// Estimated factor flops (`sum nrow^2 * ncol`).
    pub factor_flops: u64,
    /// Fraction of factor flops in the single largest front - the "top-of-tree
    /// heaviness". High means a few huge fronts dominate (node-parallelism
    /// matters); low means work is spread over many fronts (tree-parallelism
    /// matters).
    pub flop_top1_frac: f64,
    /// Fraction of factor flops in the top 1% of fronts by cost.
    pub flop_top1pct_frac: f64,
    /// Arithmetic intensity proxy `factor_flops / fill_nnz` - work per stored
    /// factor entry (BLAS-3 richness).
    pub arith_intensity: f64,
}

/// Per-column degree statistics and bandwidth over a CSC pattern, plus diagonal
/// dominance / presence from the magnitudes.
fn pattern_stats(
    n: usize,
    col_ptr: &[usize],
    row_idx: &[usize],
    mag: impl Fn(usize) -> f64,
) -> (f64, usize, f64, usize, f64, f64, f64) {
    if n == 0 {
        return (0.0, 0, 0.0, 0, 0.0, 0.0, 0.0);
    }
    let nnz = row_idx.len();
    let mut deg_sum = 0u64;
    let mut deg_sq = 0u64;
    let mut deg_max = 0usize;
    let mut bw_max = 0usize;
    let mut band_sum = 0f64;
    let mut offdiag_count = 0u64;
    let mut diag_present = 0usize;
    let mut diag_dominant = 0usize;
    for j in 0..n {
        let (s, e) = (col_ptr[j], col_ptr[j + 1]);
        let deg = e - s;
        deg_sum += deg as u64;
        deg_sq += (deg as u64) * (deg as u64);
        deg_max = deg_max.max(deg);
        let mut diag_mag = 0.0f64;
        let mut off_sum = 0.0f64;
        let mut has_diag = false;
        for (off, &i) in row_idx[s..e].iter().enumerate() {
            let m = mag(s + off);
            if i == j {
                has_diag = true;
                diag_mag = m;
            } else {
                off_sum += m;
                let band = i.abs_diff(j);
                bw_max = bw_max.max(band);
                band_sum += band as f64;
                offdiag_count += 1;
            }
        }
        if has_diag {
            diag_present += 1;
        }
        if has_diag && diag_mag >= off_sum {
            diag_dominant += 1;
        }
    }
    let nf = n as f64;
    let deg_mean = deg_sum as f64 / nf;
    let var = (deg_sq as f64 / nf) - deg_mean * deg_mean;
    let deg_cv = if deg_mean > 0.0 {
        var.max(0.0).sqrt() / deg_mean
    } else {
        0.0
    };
    let band_mean_rel = if offdiag_count > 0 {
        (band_sum / offdiag_count as f64) / nf
    } else {
        0.0
    };
    let _ = nnz;
    (
        deg_mean,
        deg_max,
        deg_cv,
        bw_max,
        band_mean_rel,
        diag_dominant as f64 / nf,
        diag_present as f64 / nf,
    )
}

/// Symbolic-shape features from the front dimensions and level widths.
fn symbolic_stats(
    nnz: usize,
    front_dims: &[(usize, usize)],
    level_widths: &[usize],
) -> (
    usize,
    u64,
    f64,
    f64,
    usize,
    usize,
    usize,
    f64,
    u64,
    f64,
    f64,
    f64,
) {
    let n_supernodes = front_dims.len();
    if n_supernodes == 0 {
        return (0, 0, 0.0, 0.0, 0, 0, 0, 0.0, 0, 0.0, 0.0, 0.0);
    }
    let mut fill = 0u64;
    let mut flops = 0u64;
    let mut col_sum = 0u64;
    let mut nrow_max = 0usize;
    let mut per_front_flops: Vec<u64> = Vec::with_capacity(n_supernodes);
    for &(ncol, nrow) in front_dims {
        let (nc, nr) = (ncol as u64, nrow as u64);
        // L panel: diagonal lower triangle + off-diagonal rows.
        fill += nc * (nc + 1) / 2 + (nr - nc) * nc;
        let f = nr * nr * nc;
        flops += f;
        per_front_flops.push(f);
        col_sum += nc;
        nrow_max = nrow_max.max(nrow);
    }
    let supernode_cols_mean = col_sum as f64 / n_supernodes as f64;
    let fill_ratio = if nnz > 0 {
        fill as f64 / nnz as f64
    } else {
        0.0
    };
    let arith_intensity = if fill > 0 {
        flops as f64 / fill as f64
    } else {
        0.0
    };

    per_front_flops.sort_unstable_by(|a, b| b.cmp(a));
    let flop_top1_frac = if flops > 0 {
        per_front_flops[0] as f64 / flops as f64
    } else {
        0.0
    };
    let top1pct = (n_supernodes / 100).max(1);
    let top_sum: u64 = per_front_flops.iter().take(top1pct).sum();
    let flop_top1pct_frac = if flops > 0 {
        top_sum as f64 / flops as f64
    } else {
        0.0
    };

    let tree_depth = level_widths.len();
    let tree_width_max = level_widths.iter().copied().max().unwrap_or(0);
    let tree_width_mean = if tree_depth > 0 {
        level_widths.iter().sum::<usize>() as f64 / tree_depth as f64
    } else {
        0.0
    };

    (
        n_supernodes,
        fill,
        fill_ratio,
        supernode_cols_mean,
        nrow_max,
        tree_depth,
        tree_width_max,
        tree_width_mean,
        flops,
        flop_top1_frac,
        flop_top1pct_frac,
        arith_intensity,
    )
}

impl StructuralFeatures {
    /// Extract features for a **symmetric** matrix (lower triangle) and its
    /// LDLᵀ analysis (an [`LdltSymbolic`], or any [`SymbolicShape`]). The analysis
    /// must be of `a`'s pattern.
    pub fn from_symmetric<T: Scalar>(a: &CscMatrix<T>, shape: &impl SymbolicShape) -> Self {
        let (deg_mean, deg_max, deg_cv, bandwidth_max, bandwidth_mean_rel, dd, dp) =
            pattern_stats(a.n, &a.col_ptr, &a.row_idx, |k| a.values[k].magnitude());
        Self::assemble(
            a.n,
            a.row_idx.len(),
            deg_mean,
            deg_max,
            deg_cv,
            bandwidth_max,
            bandwidth_mean_rel,
            dd,
            dp,
            shape,
        )
    }

    /// Extract features for a **general** (unsymmetric) matrix and its LU
    /// analysis (a [`LuSymbolic`], whose shape reflects the symmetrized pattern
    /// the LU factors). The analysis must be of `a`'s pattern.
    pub fn from_general<T: Scalar>(a: &GeneralCsc<T>, shape: &impl SymbolicShape) -> Self {
        let (deg_mean, deg_max, deg_cv, bandwidth_max, bandwidth_mean_rel, dd, dp) =
            pattern_stats(a.n, &a.col_ptr, &a.row_idx, |k| a.values[k].magnitude());
        Self::assemble(
            a.n,
            a.row_idx.len(),
            deg_mean,
            deg_max,
            deg_cv,
            bandwidth_max,
            bandwidth_mean_rel,
            dd,
            dp,
            shape,
        )
    }

    /// Recommend a worker-thread count for a **single** factorization of this
    /// matrix, from its structural fingerprint, clamped to `max_cores`.
    ///
    /// Derived from the corpus thread-scaling sweep (95 real + generated
    /// matrices, 1..12 cores): parallel scaling tracks the BLAS-3 work
    /// ([`factor_flops`](Self::factor_flops)) and the assembly-tree width, so use
    /// all cores for substantial fronts / wide trees, but stay low for **thin**
    /// (banded / path-like) or **tiny** systems, which only *regress* under
    /// oversubscription (more threads = slower). On the sweep this lands within
    /// ~10% of the per-matrix-optimal count (geomean), against ~50% for a fixed
    /// budget of 2 and a 1.85x vs 3.58x worst case against always-all-cores.
    ///
    /// This is the **single-solve** policy. For many concurrent solves sharing
    /// the machine (solver-in-the-loop), keep a small fixed budget instead so
    /// they coexist - that is why [`SolverSettings`](crate::SolverSettings)
    /// defaults to 2 rather than this.
    pub fn recommend_threads(&self, max_cores: usize) -> usize {
        recommend_threads_from(
            self.factor_flops,
            self.front_nrow_max,
            self.tree_width_max,
            max_cores,
        )
    }

    /// Auto-tuned [`SolverSettings`](crate::SolverSettings) for a matrix with these
    /// features: the knob config minimizing `weight·log(time) + (1-weight)·log(mem)`
    /// over a candidate grid, via the embedded MLP performance model. `weight = 1`
    /// is fastest, `0` is smallest peak memory;
    /// [`DEFAULT_TUNE_WEIGHT`](crate::DEFAULT_TUNE_WEIGHT) leans toward
    /// speed. The worker count is left to the [`Auto`](crate::Threads::Auto)
    /// thread predictor. See the `auto_tune` internals.
    pub fn recommend_settings(&self, weight: f64) -> crate::SolverSettings {
        crate::auto_tune::recommend_settings(self, weight)
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble(
        n: usize,
        nnz: usize,
        deg_mean: f64,
        deg_max: usize,
        deg_cv: f64,
        bandwidth_max: usize,
        bandwidth_mean_rel: f64,
        diag_dominant_frac: f64,
        diag_present_frac: f64,
        shape: &impl SymbolicShape,
    ) -> Self {
        let front_dims = shape.front_dims();
        let level_widths = shape.level_widths();
        let (
            n_supernodes,
            fill_nnz,
            fill_ratio,
            supernode_cols_mean,
            front_nrow_max,
            tree_depth,
            tree_width_max,
            tree_width_mean,
            factor_flops,
            flop_top1_frac,
            flop_top1pct_frac,
            arith_intensity,
        ) = symbolic_stats(nnz, &front_dims, &level_widths);
        StructuralFeatures {
            n,
            nnz,
            deg_mean,
            deg_max,
            deg_cv,
            bandwidth_max,
            bandwidth_mean_rel,
            diag_dominant_frac,
            diag_present_frac,
            n_supernodes,
            fill_nnz,
            fill_ratio,
            supernode_cols_mean,
            front_nrow_max,
            tree_depth,
            tree_width_max,
            tree_width_mean,
            factor_flops,
            flop_top1_frac,
            flop_top1pct_frac,
            arith_intensity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LdltSymbolic, LuSymbolic};

    /// A 2D 5-point Laplacian on an `m x m` grid (lower triangle), the canonical
    /// local stencil: low degree-CV, banded, diagonally dominant.
    fn laplacian2d(m: usize) -> CscMatrix<f64> {
        let n = m * m;
        let idx = |r: usize, c: usize| r * m + c;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..m {
            for c in 0..m {
                let p = idx(r, c);
                rows.push(p);
                cols.push(p);
                vals.push(4.0);
                for (dr, dc) in [(1usize, 0usize), (0, 1)] {
                    if r + dr < m && c + dc < m {
                        let q = idx(r + dr, c + dc);
                        let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                        rows.push(hi);
                        cols.push(lo);
                        vals.push(-1.0);
                    }
                }
            }
        }
        CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap()
    }

    #[test]
    fn features_on_laplacian_are_sane() {
        let a = laplacian2d(12);
        let sym = LdltSymbolic::analyze(&a).unwrap();
        let f = StructuralFeatures::from_symmetric(&a, &sym);
        assert_eq!(f.n, a.n);
        assert_eq!(f.nnz, a.row_idx.len());
        // 5-point stencil: every interior column has the diagonal dominating.
        assert!(f.diag_dominant_frac > 0.9, "dd {}", f.diag_dominant_frac);
        assert!(f.diag_present_frac > 0.99);
        // A regular grid has near-uniform degree -> small CV.
        assert!(f.deg_cv < 0.5, "deg_cv {}", f.deg_cv);
        // Fill grows over the input; flops/fill positive.
        assert!(f.fill_nnz as usize >= a.row_idx.len());
        assert!(f.fill_ratio >= 1.0);
        assert!(f.factor_flops > 0);
        assert!(f.arith_intensity > 0.0);
        assert!(f.tree_depth >= 1 && f.tree_width_max >= 1);
        assert!(f.flop_top1_frac > 0.0 && f.flop_top1_frac <= 1.0);
        // Sanity that the high-level symbolic agrees on n.
        assert_eq!(sym.n(), a.n);
    }

    #[test]
    fn arrow_has_high_degree_cv() {
        // An arrow matrix: a banded body plus one dense border column -> a single
        // very high-degree column, so the degree CV is large (vs the stencil).
        let n = 200;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(10.0);
        }
        // Last column couples to everything (stored lower triangle: rows > col 0).
        for i in 1..n {
            rows.push(i);
            cols.push(0);
            vals.push(0.5);
        }
        let a = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let sym = LdltSymbolic::analyze(&a).unwrap();
        let f = StructuralFeatures::from_symmetric(&a, &sym);
        let lap = {
            let l = laplacian2d(14);
            let sym = LdltSymbolic::analyze(&l).unwrap();
            StructuralFeatures::from_symmetric(&l, &sym)
        };
        assert!(
            f.deg_cv > lap.deg_cv,
            "arrow deg_cv {} should exceed stencil {}",
            f.deg_cv,
            lap.deg_cv
        );
    }

    #[test]
    fn recommend_threads_policy() {
        // A wide-tree 2D stencil with substantial work -> all cores.
        let big = laplacian2d(60);
        let sym = LdltSymbolic::analyze(&big).unwrap();
        let f = StructuralFeatures::from_symmetric(&big, &sym);
        // It has wide enough fronts/tree and enough flops to use the full budget.
        if f.factor_flops >= 300_000_000 || f.front_nrow_max >= 512 {
            assert_eq!(f.recommend_threads(12), 12);
        }
        // A path-like banded matrix: thin fronts + narrow tree -> capped low.
        let n = 4000;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            for d in 0..4 {
                if j + d < n {
                    rows.push(j + d);
                    cols.push(j);
                    vals.push(if d == 0 { 10.0 } else { -1.0 });
                }
            }
        }
        let banded = CscMatrix::from_triplets(n, &rows, &cols, &vals).unwrap();
        let sym = LdltSymbolic::analyze(&banded).unwrap();
        let fb = StructuralFeatures::from_symmetric(&banded, &sym);
        assert!(
            fb.front_nrow_max < 512 && fb.tree_width_max < 128,
            "banded is thin/narrow"
        );
        assert_eq!(fb.recommend_threads(12), 2, "thin matrix capped to 2");
        // Clamped to available cores.
        assert!(fb.recommend_threads(1) >= 1);
        assert!(f.recommend_threads(8) <= 8);
    }

    #[test]
    fn general_features_roundtrip() {
        // A small unsymmetric matrix: features compute and serialize.
        let n = 50;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for j in 0..n {
            rows.push(j);
            cols.push(j);
            vals.push(4.0);
            if j + 1 < n {
                rows.push(j + 1);
                cols.push(j);
                vals.push(-1.0);
                rows.push(j);
                cols.push(j + 1);
                vals.push(-2.0); // asymmetric
            }
        }
        let a = GeneralCsc::from_triplets(n, &rows, &cols, &vals).unwrap();
        let sym = LuSymbolic::analyze(&a).unwrap();
        let f = StructuralFeatures::from_general(&a, &sym);
        assert_eq!(f.n, n);
        // The LU shape reflects the symmetrized pattern -> at least one front.
        assert!(f.n_supernodes >= 1);
        let json = serde_json::to_string(&f).unwrap();
        let back: StructuralFeatures = serde_json::from_str(&json).unwrap();
        assert_eq!(f, back);
    }
}
