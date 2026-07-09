//! PDE finite-difference stencil generators - the workhorse family, modelling the
//! EM-FEM structure. A Dirichlet Laplacian on a 2D/3D grid is SPD with condition
//! number ∼ h⁻² (steered by grid size); a complex Helmholtz shift makes it
//! complex-symmetric and, near a resonance, ill-conditioned; anisotropy and
//! jumping coefficients add further conditioning/structure stress.
// Diagonal/triplet loops use the index as a value (push `i`, read `diag[i]`).
#![allow(clippy::needless_range_loop)]

use num_complex::Complex;

use crate::scalar::Scalar;
use crate::sparse::csc::CscMatrix;

use super::{Cond, Density, Generated, MatrixSpec, Rng, Structure, Symmetry};

/// Knobs for the grid Laplacian.
#[derive(Clone)]
pub struct StencilOpts {
    /// Per-dimension conductivity (1.0 = isotropic). Strong anisotropy → harder.
    pub aniso: [f64; 3],
    /// Added to every diagonal (regularization / Helmholtz real shift). `> 0`
    /// improves conditioning; negative pushes toward indefinite/singular.
    pub shift: f64,
    /// Coefficient contrast for a jumping-coefficient problem: `1.0` = constant,
    /// `> 1` draws a log-uniform per-node coefficient in `[1, contrast]` (high
    /// contrast ⇒ ill-conditioned, as in heterogeneous media).
    pub jump_contrast: f64,
    /// Seed for the jumping coefficient field.
    pub seed: u64,
}

impl Default for StencilOpts {
    fn default() -> Self {
        StencilOpts {
            aniso: [1.0, 1.0, 1.0],
            shift: 0.0,
            jump_contrast: 1.0,
            seed: 1,
        }
    }
}

/// Real edge weights of the grid: lower-triangle off-diagonals and the diagonal,
/// returned as `(diag, lower_triplets)`. Shared by the generic and complex paths.
/// Dirichlet boundary: a missing stencil arm still loads the diagonal (so boundary
/// nodes are strictly diagonally dominant), giving an SPD M-matrix.
fn laplacian_weights(dims: &[usize], opts: &StencilOpts) -> (Vec<f64>, Vec<(usize, usize, f64)>) {
    let ndim = dims.len();
    let n: usize = dims.iter().product();
    let mut stride = vec![1usize; ndim];
    for d in 1..ndim {
        stride[d] = stride[d - 1] * dims[d - 1];
    }
    // Per-node coefficient field (jumping coefficient).
    let coeff: Vec<f64> = if opts.jump_contrast > 1.0 {
        let mut rng = Rng::new(opts.seed);
        let ln_c = opts.jump_contrast.ln();
        (0..n).map(|_| (rng.range(0.0, ln_c)).exp()).collect()
    } else {
        vec![1.0; n]
    };
    let coord = |i: usize, d: usize| (i / stride[d]) % dims[d];

    let mut diag = vec![opts.shift; n];
    let mut tri: Vec<(usize, usize, f64)> = Vec::with_capacity(n * ndim);
    for i in 0..n {
        for d in 0..ndim.min(3) {
            let c = coord(i, d);
            // + arm: emit the interior edge once (j > i); load both diagonals.
            if c + 1 < dims[d] {
                let j = i + stride[d];
                let w = opts.aniso[d] * (coeff[i] * coeff[j]).sqrt();
                diag[i] += w;
                diag[j] += w;
                tri.push((j, i, -w)); // lower triangle (j > i)
            } else {
                // + boundary (Dirichlet ghost): diagonal load only.
                diag[i] += opts.aniso[d] * coeff[i];
            }
            // − boundary: diagonal load only (interior − edges are some node's +).
            if c == 0 {
                diag[i] += opts.aniso[d] * coeff[i];
            }
        }
    }
    (diag, tri)
}

/// Real SPD grid Laplacian (Dirichlet), generic over the scalar type. `dims` is
/// `[nx, ny]` (2D) or `[nx, ny, nz]` (3D).
pub fn laplacian<T: Scalar>(dims: &[usize], opts: &StencilOpts) -> CscMatrix<T> {
    let n: usize = dims.iter().product();
    let (diag, tri) = laplacian_weights(dims, opts);
    let mut rows = Vec::with_capacity(n + tri.len());
    let mut cols = Vec::with_capacity(n + tri.len());
    let mut vals = Vec::with_capacity(n + tri.len());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        vals.push(T::from_real(diag[i]));
    }
    for (r, c, w) in tri {
        rows.push(r);
        cols.push(c);
        vals.push(T::from_real(w));
    }
    super::build_sym(n, &rows, &cols, &vals)
}

/// Complex-symmetric **Helmholtz** operator `Δ − k² I` on a grid (lower triangle).
/// `k2` is the (possibly complex, for lossy media) squared wavenumber: a large
/// real `k2` makes the operator indefinite; an imaginary part models loss. This is
/// the complex-symmetric EM-FEM analogue (`A = Aᵀ`, not Hermitian).
pub fn helmholtz(dims: &[usize], k2: Complex<f64>, opts: &StencilOpts) -> CscMatrix<Complex<f64>> {
    let n: usize = dims.iter().product();
    let (diag, tri) = laplacian_weights(dims, opts);
    let mut rows = Vec::with_capacity(n + tri.len());
    let mut cols = Vec::with_capacity(n + tri.len());
    let mut vals = Vec::with_capacity(n + tri.len());
    for i in 0..n {
        rows.push(i);
        cols.push(i);
        // Δ − k²I : the Laplacian diagonal minus the complex shift.
        vals.push(Complex::new(diag[i], 0.0) - k2);
    }
    for (r, c, w) in tri {
        rows.push(r);
        cols.push(c);
        vals.push(Complex::new(w, 0.0));
    }
    super::build_sym(n, &rows, &cols, &vals)
}

/// A roughly-cubic 3D grid dimension triple with `≈ target` nodes.
fn cube(target: usize) -> [usize; 3] {
    let k = (target as f64).cbrt().round().max(2.0) as usize;
    [k, k, k]
}

/// A roughly-square 2D grid dimension pair with `≈ target` nodes.
fn square(target: usize) -> [usize; 2] {
    let k = (target as f64).sqrt().round().max(2.0) as usize;
    [k, k]
}

pub(super) fn add_to_catalog(c: &mut Vec<MatrixSpec>) {
    fn cx(m: CscMatrix<Complex<f64>>) -> Generated {
        Generated::Symmetric(m)
    }
    // 2D Poisson (SPD), small/medium/large.
    c.push(MatrixSpec {
        name: "poisson2d_small",
        structure: Structure::Stencil2D,
        symmetry: Symmetry::Spd,
        cond: Cond::Moderate,
        density: Density::Sparse,
        size: 2_500,
        build: || cx(laplacian(&square(2_500), &StencilOpts::default())),
    });
    c.push(MatrixSpec {
        name: "poisson2d_large",
        structure: Structure::Stencil2D,
        symmetry: Symmetry::Spd,
        cond: Cond::Ill, // κ ∼ h⁻², large grid ⇒ ill-conditioned
        density: Density::Sparse,
        size: 160_000,
        build: || cx(laplacian(&square(160_000), &StencilOpts::default())),
    });
    // 3D Poisson (SPD) - the EM-FEM sparsity, small and large.
    c.push(MatrixSpec {
        name: "poisson3d_small",
        structure: Structure::Stencil3D,
        symmetry: Symmetry::Spd,
        cond: Cond::Moderate,
        density: Density::Sparse,
        size: 8_000,
        build: || cx(laplacian(&cube(8_000), &StencilOpts::default())),
    });
    c.push(MatrixSpec {
        name: "poisson3d_large",
        structure: Structure::Stencil3D,
        symmetry: Symmetry::Spd,
        cond: Cond::Ill,
        density: Density::Sparse,
        size: 125_000,
        build: || cx(laplacian(&cube(125_000), &StencilOpts::default())),
    });
    // Anisotropic 3D (harder ordering / conditioning).
    c.push(MatrixSpec {
        name: "aniso3d",
        structure: Structure::Stencil3D,
        symmetry: Symmetry::Spd,
        cond: Cond::Ill,
        density: Density::Sparse,
        size: 64_000,
        build: || {
            let o = StencilOpts {
                aniso: [1.0, 1.0, 1000.0],
                ..Default::default()
            };
            cx(laplacian(&cube(64_000), &o))
        },
    });
    // Jumping-coefficient 3D (high contrast ⇒ ill-conditioned heterogeneous media).
    c.push(MatrixSpec {
        name: "jump3d",
        structure: Structure::Stencil3D,
        symmetry: Symmetry::Spd,
        cond: Cond::Ill,
        density: Density::Sparse,
        size: 64_000,
        build: || {
            let o = StencilOpts {
                jump_contrast: 1e6,
                seed: 7,
                ..Default::default()
            };
            cx(laplacian(&cube(64_000), &o))
        },
    });
    // Complex-symmetric Helmholtz (EM): well below resonance, and near-resonance.
    c.push(MatrixSpec {
        name: "helmholtz3d",
        structure: Structure::Stencil3D,
        symmetry: Symmetry::ComplexSymmetric,
        cond: Cond::Moderate,
        density: Density::Sparse,
        size: 64_000,
        build: || {
            cx(helmholtz(
                &cube(64_000),
                Complex::new(0.5, 0.05),
                &StencilOpts::default(),
            ))
        },
    });
    c.push(MatrixSpec {
        name: "helmholtz3d_indef",
        structure: Structure::Stencil3D,
        symmetry: Symmetry::ComplexSymmetric,
        cond: Cond::Ill, // large real shift ⇒ strongly indefinite
        density: Density::Sparse,
        size: 64_000,
        build: || {
            cx(helmholtz(
                &cube(64_000),
                Complex::new(6.0, 0.1),
                &StencilOpts::default(),
            ))
        },
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson2d_is_spd_lower_triangle() {
        let a = laplacian::<f64>(&[4, 4], &StencilOpts::default());
        assert_eq!(a.n, 16);
        // Interior node has diagonal 4 (2 per dim); all off-diagonals negative.
        for j in 0..a.n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[k];
                assert!(i >= j, "lower triangle only");
                if i == j {
                    assert!(a.values[k] >= 4.0 - 1e-12, "Dirichlet diagonal ≥ 2·ndim");
                } else {
                    assert!(a.values[k] < 0.0, "off-diagonal is −w");
                }
            }
        }
    }

    #[test]
    fn helmholtz_diagonal_carries_complex_shift() {
        let k2 = Complex::new(2.0, 0.3);
        let a = helmholtz(&[3, 3, 3], k2, &StencilOpts::default());
        assert_eq!(a.n, 27);
        // Diagonal of node 0 (a corner): 6 stencil arms (3 dims × both signs) of
        // weight 1 ⇒ 6, minus k².
        let diag0 = a.values[a.col_ptr[0]];
        assert!((diag0 - (Complex::new(6.0, 0.0) - k2)).norm() < 1e-12);
    }

    #[test]
    fn jump_coefficient_changes_values_deterministically() {
        let o = StencilOpts {
            jump_contrast: 100.0,
            seed: 42,
            ..Default::default()
        };
        let a = laplacian::<f64>(&[8, 8, 8], &o);
        let b = laplacian::<f64>(&[8, 8, 8], &o);
        assert_eq!(a.values, b.values, "same seed ⇒ identical matrix");
        let c = laplacian::<f64>(&[8, 8, 8], &StencilOpts::default());
        assert_ne!(a.values.len(), 0);
        assert_ne!(a.values, c.values, "jumps change the weights");
    }
}
