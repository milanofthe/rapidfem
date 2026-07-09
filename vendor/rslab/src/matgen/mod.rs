//! Parametrized test-matrix generators for benchmarking and testing the solver
//! across the axes that matter for a sparse direct solver: **size**, **sparsity
//! structure**, **symmetry**, **conditioning**, and **density**.
//!
//! Design follows the established packages (MatrixDepot.jl, MATLAB `gallery`,
//! LAPACK `xLATMS`): parametrized *generators* for structured coverage plus an
//! optional *downloader* ([`download`], feature `matgen-download`) for real
//! SuiteSparse / Matrix Market matrices. Conditioning is steered **structurally**
//! (diagonal dominance, PDE refinement, near-resonance shift, coefficient jumps)
//! for the sparse families, and **spectrally** (prescribed eigenvalues, `xLATMS`
//! style) for the small dense [`spectral`] family where exact κ is needed.
//!
//! The real-valued structural families are generic over [`Scalar`]; the inherently
//! complex families (Helmholtz, BEM/MoM kernel) produce `Complex<f64>`. The
//! [`catalog`] is `Complex<f64>` (the solver's primary EM/MoM type) and tags each
//! entry so benchmarks can sweep e.g. "all SPD small" or "all ill-conditioned".

use num_complex::Complex;

use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;
use crate::sparse::general::GeneralCsc;

/// Build a lower-triangle [`CscMatrix`] from generator triplets. The generators
/// construct valid triplets by definition, so a failure is an internal invariant
/// violation (panic, not an error path) - avoids `unwrap`/`expect` in `src/`.
pub(crate) fn build_sym<T: Scalar>(
    n: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
) -> CscMatrix<T> {
    match CscMatrix::from_triplets(n, rows, cols, vals) {
        Ok(m) => m,
        Err(e) => panic!("matgen: internal invariant - invalid symmetric triplets: {e}"),
    }
}

/// As [`build_sym`] but for the full unsymmetric [`GeneralCsc`].
pub(crate) fn build_gen<T: Scalar>(
    n: usize,
    rows: &[usize],
    cols: &[usize],
    vals: &[T],
) -> GeneralCsc<T> {
    match GeneralCsc::from_triplets(n, rows, cols, vals) {
        Ok(m) => m,
        Err(e) => panic!("matgen: internal invariant - invalid triplets: {e}"),
    }
}

pub mod bem;
#[cfg(feature = "matgen-download")]
pub mod download;
pub mod fem;
pub mod random;
pub mod stencil;
pub mod structured;

/// Small, fast, deterministic PRNG (xorshift64*). Pure Rust, no `rand` dep - so
/// generated matrices are exactly reproducible from a seed across runs/platforms.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15 | 1)
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    /// Standard normal via Box-Muller (one of the pair).
    #[inline]
    pub fn normal(&mut self) -> f64 {
        let u1 = (self.unit()).max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// --------------------------------------------------------------------------
// Catalog: tagged, named matrices for benchmark sweeps.
// --------------------------------------------------------------------------

/// Sparsity / origin structure of a catalog matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Structure {
    Stencil2D,
    Stencil3D,
    Bem,
    Banded,
    Arrow,
    Random,
    Spectral,
}

/// Symmetry class - selects the solver path (LDLᵀ vs LU).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symmetry {
    /// Real or complex symmetric positive definite.
    Spd,
    /// Complex-symmetric (A = Aᵀ, not Hermitian) - the EM-FEM case.
    ComplexSymmetric,
    /// Symmetric indefinite (saddle/KKT).
    SymIndefinite,
    /// Unsymmetric (MoM/BEM).
    Unsymmetric,
}

/// Rough conditioning class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    Well,
    Moderate,
    Ill,
}

/// Rough nonzeros-per-row class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Density {
    Sparse,
    Medium,
    Dense,
}

/// A generated matrix in the form the solver consumes: symmetric matrices as a
/// lower-triangle [`CscMatrix`] (→ LDLᵀ), unsymmetric as a full [`GeneralCsc`]
/// (→ LU).
pub enum Generated {
    Symmetric(CscMatrix<Complex<f64>>),
    Unsymmetric(GeneralCsc<Complex<f64>>),
}

impl Generated {
    pub fn n(&self) -> usize {
        match self {
            Generated::Symmetric(a) => a.n,
            Generated::Unsymmetric(a) => a.n,
        }
    }
    pub fn nnz(&self) -> usize {
        match self {
            Generated::Symmetric(a) => a.values.len(),
            Generated::Unsymmetric(a) => a.values.len(),
        }
    }
}

/// A catalog entry: a tagged, named, on-demand matrix builder.
pub struct MatrixSpec {
    pub name: &'static str,
    pub structure: Structure,
    pub symmetry: Symmetry,
    pub cond: Cond,
    pub density: Density,
    /// Approximate dimension (the actual `n` may round to the grid size).
    pub size: usize,
    build: fn() -> Generated,
}

impl MatrixSpec {
    pub fn build(&self) -> Generated {
        (self.build)()
    }
}

/// The full catalog of named, tagged test matrices. Filter with the iterator
/// adapters, e.g. `catalog().into_iter().filter(|m| m.cond == Cond::Ill)`.
pub fn catalog() -> Vec<MatrixSpec> {
    let mut c = Vec::new();
    stencil::add_to_catalog(&mut c);
    bem::add_to_catalog(&mut c);
    structured::add_to_catalog(&mut c);
    random::add_to_catalog(&mut c);
    c
}

#[cfg(test)]
mod integration {
    //! Every family must produce matrices the solver actually factors and solves -
    //! the whole point. Small instances, exact factorization, true residual.
    use super::*;
    use crate::{LdltSymbolic, LuSymbolic, SolverSettings};

    type C = Complex<f64>;

    fn rhs(n: usize) -> Vec<C> {
        (0..n)
            .map(|i| Complex::new((i % 5) as f64 - 2.0, (i % 3) as f64 - 1.0))
            .collect()
    }

    fn ldlt_resid(a: &CscMatrix<C>) -> f64 {
        let b = rhs(a.n);
        let f = LdltSymbolic::analyze(a)
            .unwrap()
            .factor(a, &SolverSettings::default())
            .unwrap();
        let x = f.solve(&b).unwrap();
        let mut ax = vec![Complex::new(0.0, 0.0); a.n];
        a.symv(&x, &mut ax);
        let num: f64 = (0..a.n)
            .map(|i| (ax[i] - b[i]).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        num / den.max(1e-300)
    }

    fn lu_resid(a: &GeneralCsc<C>) -> f64 {
        let b = rhs(a.n);
        let f = LuSymbolic::analyze(a)
            .unwrap()
            .factor(a, &SolverSettings::default())
            .unwrap();
        let x = f.solve(&b).unwrap();
        let mut ax = vec![Complex::new(0.0, 0.0); a.n];
        a.matvec(&x, &mut ax);
        let num: f64 = (0..a.n)
            .map(|i| (ax[i] - b[i]).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let den: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        num / den.max(1e-300)
    }

    #[test]
    fn symmetric_families_factor_and_solve() {
        // SPD stencil, complex-symmetric Helmholtz, banded, indefinite KKT (2×2 BK),
        // and an exactly-ill-conditioned spectral matrix.
        let cases: Vec<(&str, CscMatrix<C>)> = vec![
            (
                "poisson3d",
                stencil::laplacian(&[8, 8, 8], &stencil::StencilOpts::default()),
            ),
            (
                "helmholtz",
                stencil::helmholtz(
                    &[8, 8, 8],
                    Complex::new(3.0, 0.1),
                    &stencil::StencilOpts::default(),
                ),
            ),
            ("banded", structured::banded(400, 6, 1.0, 1)),
            ("kkt_arrow", structured::arrow(400, 16, 1e-2, 1)),
            ("spectral_ill", random::spectral(200, 1e8, false, 1)),
            ("rand_spd", random::random_spd(500, 12, 1.0, 1)),
        ];
        for (name, a) in cases {
            let r = ldlt_resid(&a);
            assert!(r < 1e-6, "{name}: LDLᵀ residual {r:.1e} too large");
        }
    }

    #[test]
    fn unsymmetric_families_factor_and_solve() {
        let bem = bem::kernel(600, &bem::BemOpts::default());
        assert!(lu_resid(&bem) < 1e-6, "BEM LU residual too large");
        let r = random::random_unsym::<C>(500, 12, 2.0, 1);
        assert!(
            lu_resid(&r) < 1e-6,
            "random unsymmetric LU residual too large"
        );
    }

    #[test]
    fn solver_is_type_agnostic_all_four_scalars() {
        // Prove end-to-end that factor+solve works for **every** Scalar type - not
        // just the f64/Complex<f64> the other tests use. A latent f32/Complex<f32>
        // monomorphization gap (e.g. a GEMM only specialized for f64/c64) would fail
        // to compile or solve here.
        use crate::Scalar;
        fn sym_ok<T: Scalar>(tol: f64) {
            let a = structured::banded::<T>(200, 6, 1.0, 1);
            let b: Vec<T> = (0..a.n)
                .map(|i| T::from_real((i % 7) as f64 - 3.0))
                .collect();
            let f = LdltSymbolic::analyze(&a)
                .unwrap()
                .factor(&a, &SolverSettings::default())
                .unwrap();
            let x = f.solve(&b).unwrap();
            let mut ax = vec![T::zero(); a.n];
            a.symv(&x, &mut ax);
            let num: f64 = (0..a.n)
                .map(|i| (ax[i] - b[i]).magnitude().powi(2))
                .sum::<f64>()
                .sqrt();
            let den: f64 = b.iter().map(|v| v.magnitude().powi(2)).sum::<f64>().sqrt();
            assert!(
                num / den.max(1e-30) < tol,
                "sym {} residual {:.2e}",
                std::any::type_name::<T>(),
                num / den
            );
        }
        fn unsym_ok<T: Scalar>(tol: f64) {
            let a = random::random_unsym::<T>(200, 12, 3.0, 1);
            let b: Vec<T> = (0..a.n)
                .map(|i| T::from_real((i % 5) as f64 - 2.0))
                .collect();
            let f = LuSymbolic::analyze(&a)
                .unwrap()
                .factor(&a, &SolverSettings::default())
                .unwrap();
            let x = f.solve(&b).unwrap();
            let mut ax = vec![T::zero(); a.n];
            a.matvec(&x, &mut ax);
            let num: f64 = (0..a.n)
                .map(|i| (ax[i] - b[i]).magnitude().powi(2))
                .sum::<f64>()
                .sqrt();
            let den: f64 = b.iter().map(|v| v.magnitude().powi(2)).sum::<f64>().sqrt();
            assert!(
                num / den.max(1e-30) < tol,
                "unsym {} residual {:.2e}",
                std::any::type_name::<T>(),
                num / den
            );
        }
        // f64 / f32 / Complex<f64> / Complex<f32> - the four Scalar impls.
        sym_ok::<f64>(1e-10);
        sym_ok::<f32>(1e-2);
        sym_ok::<C>(1e-10);
        sym_ok::<Complex<f32>>(1e-2);
        unsym_ok::<f64>(1e-10);
        unsym_ok::<f32>(1e-2);
        unsym_ok::<C>(1e-10);
        unsym_ok::<Complex<f32>>(1e-2);

        // The memory estimator scales with the scalar size - agnostic too.
        let a32 = random::random_unsym::<f32>(150, 8, 2.0, 1);
        let e32 = LuSymbolic::analyze(&a32).unwrap().estimate_memory::<f32>();
        let ac64 = random::random_unsym::<Complex<f64>>(150, 8, 2.0, 1);
        let ec64 = LuSymbolic::analyze(&ac64)
            .unwrap()
            .estimate_memory::<Complex<f64>>();
        assert_eq!(e32.value_bytes, 4);
        assert_eq!(ec64.value_bytes, 16);
        assert!(
            ec64.transient_peak_bytes > e32.transient_peak_bytes,
            "16B estimate > 4B"
        );
    }

    #[test]
    fn diagnostics_and_estimate_wired_on_both_paths() {
        // Symmetric → LDLᵀ.
        let a = structured::banded::<C>(500, 8, 1.0, 1);
        let opts = SolverSettings::default().with_threads(3);
        let f = LdltSymbolic::analyze(&a)
            .unwrap()
            .factor(&a, &opts)
            .unwrap();
        let d = f.diagnostics();
        assert_eq!(d.threads, 3, "thread budget recorded");
        assert!(
            d.stages
                .iter()
                .any(|s| s.name == "factor" && s.wall_ms >= 0.0),
            "factor stage"
        );
        assert!(d.factor_nnz > 0);
        let est = d.estimate.expect("a-priori estimate attached");
        assert_eq!(est.value_bytes, 16);
        assert!(
            est.transient_peak_bytes > est.factor_bytes,
            "transient > factor alone"
        );

        // Unsymmetric → LU; threads=0 resolves to all cores.
        let g = bem::kernel(600, &bem::BemOpts::default());
        let o2 = SolverSettings::default().with_threads(0);
        let lf = LuSymbolic::analyze(&g).unwrap().factor(&g, &o2).unwrap();
        let ld = lf.diagnostics();
        assert!(ld.threads >= 1);
        assert!(ld.estimate.is_some());
        assert_eq!(ld.estimate.unwrap().value_bytes, 16);
    }

    #[test]
    fn catalog_is_well_formed() {
        let cat = catalog();
        assert!(cat.len() >= 15, "catalog has a useful number of entries");
        // Names unique.
        let mut names: Vec<&str> = cat.iter().map(|m| m.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), cat.len(), "catalog names are unique");
        // Every symmetry class is represented.
        for sym in [
            Symmetry::Spd,
            Symmetry::ComplexSymmetric,
            Symmetry::SymIndefinite,
            Symmetry::Unsymmetric,
        ] {
            assert!(
                cat.iter().any(|m| m.symmetry == sym),
                "missing symmetry {sym:?}"
            );
        }
    }
}
