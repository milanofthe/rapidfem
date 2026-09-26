//! The kernel knobs as the kernels take them.

use crate::numeric::settings::KernelSettings;

/// [`KernelSettings`] plus the pivot threshold and the interrupt flag,
/// threaded by value into the dense kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KernelTuning<'a> {
    pub k: KernelSettings,
    pub pivot_threshold: f64,
    pub interrupt: Option<&'a std::sync::atomic::AtomicBool>,
}

impl KernelTuning<'_> {
    #[inline]
    pub fn interrupted(&self) -> Result<(), crate::error::RslabError> {
        match self.interrupt {
            Some(flag) if flag.load(std::sync::atomic::Ordering::Relaxed) => {
                Err(crate::error::RslabError::Interrupted)
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{CscMatrix, KernelSettings, LdltSolver, SolverSettings};
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
        let solve = |s: &SolverSettings| LdltSolver::factor(&a, s).unwrap().solve(&b).unwrap();

        let x_def = solve(&SolverSettings::default().with_threads(0));
        let x_scalar = solve(&SolverSettings {
            kernels: KernelSettings {
                scalar_gate: usize::MAX,
                par_gemm: usize::MAX,
                par_cdiv: usize::MAX,
                ..Default::default()
            },
            ..SolverSettings::default().with_threads(0)
        });
        let x_par = solve(&SolverSettings {
            kernels: KernelSettings {
                scalar_gate: 0,
                par_gemm: 0,
                par_cdiv: 0,
                ..Default::default()
            },
            ..SolverSettings::default().with_threads(0)
        });
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
    /// produce a correct solve.
    #[test]
    fn panel_nb_preserves_correctness() {
        let (a, b) = helmholtz(9);
        for nb in [16usize, 32, 64, 100, 200] {
            let mut s = SolverSettings::default();
            s.kernels.panel_nb = nb;
            let x = LdltSolver::factor(&a, &s).unwrap().solve(&b).unwrap();
            let mut ax = vec![Complex::new(0.0, 0.0); a.n];
            a.symv(&x, &mut ax);
            let res: f64 = (0..a.n)
                .map(|i| (ax[i] - b[i]).norm_sqr())
                .sum::<f64>()
                .sqrt()
                / b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            assert!(res < 1e-9, "NB={nb} residual {res:.2e}");
        }
    }
}
