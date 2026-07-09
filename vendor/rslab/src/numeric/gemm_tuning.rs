//! Kernel-level GEMM scheduling knobs - the parallelism levers carried per-call
//! by [`SolverSettings`](crate::SolverSettings).
//!
//! The factorization kernels switch between a scalar triple loop and a SIMD
//! GEMM, and between a serial and a rayon-parallel GEMM, at fixed flop-count
//! thresholds. These decide *where* node-parallelism kicks in - in particular,
//! near the assembly-tree root, where tree-parallelism has dried up and a few
//! huge fronts must pick up idle cores (the "top-of-tree" parallelism lever).
//!
//! They live on [`SolverSettings`](crate::SolverSettings) (the single solver
//! settings interface) and are threaded by value into the dense-front /
//! left-looking kernels as a [`KernelTuning`] - per-call, no process-wide state.
//! The defaults reproduce the historically-tuned behaviour, so leaving them
//! untouched changes nothing; the auto-tuning sweep overrides them per matrix.

/// Default: below `4096` flops a contribution update is a scalar triple loop.
pub const DEFAULT_SCALAR_GATE: usize = 4096;
/// Default: at/above `1_000_000` flops a cmod-class GEMM goes rayon-parallel.
pub const DEFAULT_PAR_GEMM: usize = 1_000_000;
/// Default: at/above `8_000_000` flops a cdiv / Schur / LU-front GEMM goes
/// rayon-parallel.
pub const DEFAULT_PAR_CDIV: usize = 8_000_000;
/// Default Bunch-Kaufman / LU panel width (the blocking factor). Wider panels
/// push more work into the deferred trailing GEMM but enlarge the serial
/// within-panel factorization; narrower panels do the reverse at more GEMM calls.
pub const DEFAULT_PANEL_NB: usize = 64;
/// Default threshold partial-pivoting tolerance `u ∈ [0, 1]` for the left-looking
/// LU path (UMFPACK's `THRESH`). The diagonal pivot is kept unless it falls below
/// `u · |colmax|` in its fully-summed block. `u = 1` is full partial pivoting
/// (always take the column max); `u → 0` keeps the diagonal unless it is exactly
/// zero (least fill, least stable). The historical value `0.1` is the default, so
/// leaving it untouched reproduces the shipped factor.
pub const DEFAULT_PIVOT_U: f64 = 0.1;

/// The kernel scheduling knobs as a cheap `Copy` bundle, threaded by value into
/// the dense-front and left-looking kernels from [`SolverSettings`](crate::SolverSettings)
/// (no process-wide state - per-call). `panel_nb` is pre-clamped to at least 8 by
/// [`SolverSettings::kernel`](crate::SolverSettings::kernel).
#[derive(Debug, Clone, Copy)]
pub(crate) struct KernelTuning {
    pub scalar_gate: usize,
    pub par_gemm: usize,
    pub par_cdiv: usize,
    pub panel_nb: usize,
    pub use_gemm_schur: bool,
    /// Threshold partial-pivoting tolerance `u ∈ [0, 1]` for the left-looking LU
    /// path (see [`DEFAULT_PIVOT_U`]). Ignored by the LDLᵀ kernels (Bunch-Kaufman
    /// has its own fixed `α`) and by the multifrontal LU front (full pivoting).
    pub pivot_u: f64,
}

/// The three GEMM scheduling thresholds (flop counts), the kernel-level
/// parallelism levers - a convenience bundle for
/// [`SolverSettings::with_gemm_thresholds`](crate::SolverSettings::with_gemm_thresholds).
/// See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemmThresholds {
    /// Below this flop count (`rows * cols * k`) a contribution update runs as a
    /// scalar triple loop instead of a SIMD GEMM.
    pub scalar_gate: usize,
    /// At/above this flop count a cmod-class GEMM runs rayon-parallel.
    pub par_gemm: usize,
    /// At/above this flop count the panel-trailing / Schur / LU-front GEMM runs
    /// rayon-parallel. Lowering it makes large fronts near the tree root engage
    /// node-parallelism earlier (the top-of-tree lever).
    pub par_cdiv: usize,
}

impl Default for GemmThresholds {
    fn default() -> Self {
        Self {
            scalar_gate: DEFAULT_SCALAR_GATE,
            par_gemm: DEFAULT_PAR_GEMM,
            par_cdiv: DEFAULT_PAR_CDIV,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{CscMatrix, FactorMethod, GemmThresholds, LdltSolver, SolverSettings};
    use num_complex::Complex;

    fn helmholtz(m: usize) -> (CscMatrix<Complex<f64>>, Vec<Complex<f64>>) {
        let c = |re, im| Complex::new(re, im);
        let n = m * m;
        let idx = |r: usize, cc: usize| r * m + cc;
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for r in 0..m {
            for cc in 0..m {
                let p = idx(r, cc);
                rows.push(p);
                cols.push(p);
                vals.push(c(4.0, 0.5));
                for (dr, dc) in [(1usize, 0usize), (0, 1)] {
                    if r + dr < m && cc + dc < m {
                        let q = idx(r + dr, cc + dc);
                        let (hi, lo) = if q >= p { (q, p) } else { (p, q) };
                        rows.push(hi);
                        cols.push(lo);
                        vals.push(c(-1.0, 0.1));
                    }
                }
            }
        }
        let a = CscMatrix::<Complex<f64>>::from_triplets(n, &rows, &cols, &vals).unwrap();
        let b: Vec<Complex<f64>> = (0..n).map(|i| c(i as f64 - 50.0, 1.0)).collect();
        (a, b)
    }

    /// Moving the per-call thresholds changes only the serial/parallel and
    /// scalar/GEMM kernel selection - never the answer. Factor the same matrix
    /// under all-scalar/serial, all-GEMM/parallel, and the default, and confirm
    /// the solutions agree to working precision. No global state, so this is a
    /// pure per-call comparison (no serializing guard needed).
    #[test]
    fn thresholds_do_not_change_the_result() {
        let (a, b) = helmholtz(10);
        let solve = |s: &SolverSettings| LdltSolver::factor_with(&a, s).unwrap().solve(&b).unwrap();

        let x_def = solve(&SolverSettings::default().with_threads(0));
        let x_scalar = solve(
            &SolverSettings::default()
                .with_threads(0)
                .with_gemm_thresholds(GemmThresholds {
                    scalar_gate: usize::MAX,
                    par_gemm: usize::MAX,
                    par_cdiv: usize::MAX,
                }),
        );
        let x_par = solve(
            &SolverSettings::default()
                .with_threads(0)
                .with_gemm_thresholds(GemmThresholds {
                    scalar_gate: 0,
                    par_gemm: 0,
                    par_cdiv: 0,
                }),
        );
        for i in 0..x_def.len() {
            assert!(
                (x_def[i] - x_scalar[i]).norm() < 1e-9,
                "scalar path diverged at {i}"
            );
            assert!(
                (x_def[i] - x_par[i]).norm() < 1e-9,
                "parallel path diverged at {i}"
            );
        }
    }

    /// The panel width changes the pivot sequence (a different but valid factor),
    /// so the factor is not bit-identical across NB - but every width must still
    /// produce a correct solve, on both the left-looking and multifrontal paths.
    #[test]
    fn panel_nb_preserves_correctness() {
        let (a, b) = helmholtz(9);
        for method in [FactorMethod::LeftLooking, FactorMethod::Multifrontal] {
            for nb in [16usize, 32, 64, 100, 200] {
                let s = SolverSettings::default()
                    .with_method(method)
                    .with_panel_nb(nb);
                let x = LdltSolver::factor_with(&a, &s).unwrap().solve(&b).unwrap();
                let mut ax = vec![Complex::new(0.0, 0.0); a.n];
                a.symv(&x, &mut ax);
                let res: f64 = (0..a.n)
                    .map(|i| (ax[i] - b[i]).norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    / b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
                assert!(res < 1e-9, "NB={nb} method={method:?} residual {res:.2e}");
            }
        }
    }
}
