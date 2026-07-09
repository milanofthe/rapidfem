//! BEM/MoM kernel generator - the complex **unsymmetric** family modelling the
//! method-of-moments preconditioner blocks (the `precond_matrices` workflow).
//!
//! Collocation double-layer operator on a point cloud over the unit sphere: with
//! the Helmholtz Green's function `G(r) = e^{ikr}/(4πr)`, the entry is the
//! observation-normal derivative
//! `A_ij = ∂G/∂n_i (r_ij) = G(r) (ik − 1/r) ((p_i − p_j)·n_i)/r`,
//! which depends on the *row*'s normal `n_i` and is therefore unsymmetric - a
//! genuine BEM operator, not a scaled symmetric matrix. Trimming to a near-field
//! radius makes it sparse (and tunes density); the wavenumber `k` tunes
//! conditioning (near a sphere resonance it is ill-conditioned).

use num_complex::Complex;

use super::{Cond, Density, Generated, MatrixSpec, Rng, Structure, Symmetry};
use crate::sparse::general::GeneralCsc;

type C = Complex<f64>;

/// Knobs for the BEM kernel matrix.
#[derive(Clone)]
pub struct BemOpts {
    /// Wavenumber `k` (free-space). Larger `k` ⇒ more oscillatory; near a sphere
    /// eigenfrequency ⇒ ill-conditioned (interior-resonance problem).
    pub k: f64,
    /// Near-field cutoff radius (sphere has diameter 2). `≥ 2.0` keeps every pair
    /// (dense); smaller values keep only nearby pairs (sparse) and set the density.
    pub cutoff: f64,
    /// Diagonal self/jump term (the double-layer solid-angle term ± a small loss
    /// for well-posedness).
    pub self_term: C,
    /// Surface roughness: per-point radial jitter amplitude. `0` is a perfect
    /// sphere (where the double-layer matrix degenerates to **symmetric**); `> 0`
    /// varies `|p_i|` so the operator is genuinely unsymmetric, as in MoM.
    pub rough: f64,
    /// Seed for the roughness field.
    pub seed: u64,
}

impl Default for BemOpts {
    fn default() -> Self {
        BemOpts {
            k: 2.0,
            cutoff: 0.5,
            self_term: Complex::new(0.5, 0.05),
            rough: 0.3,
            seed: 1,
        }
    }
}

/// A rough quasi-spherical point cloud: even Fibonacci **directions** (the unit
/// normals `n_i`), each pushed to radius `1 + rough·jitter` so radii vary - which
/// breaks the sphere's symmetry and makes the double-layer operator unsymmetric.
/// Returns `(points, normals)`.
fn surface_points(n: usize, rough: f64, seed: u64) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let golden = std::f64::consts::PI * (1.0 + 5.0_f64.sqrt());
    let mut rng = Rng::new(seed);
    let mut pts = Vec::with_capacity(n);
    let mut nrm = Vec::with_capacity(n);
    for i in 0..n {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let th = golden * i as f64;
        let dir = [r * th.cos(), r * th.sin(), z]; // unit direction = normal
        let rad = 1.0 + rough * (rng.unit() - 0.5);
        pts.push([dir[0] * rad, dir[1] * rad, dir[2] * rad]);
        nrm.push(dir);
    }
    (pts, nrm)
}

/// Build the collocation double-layer BEM matrix on `n` sphere points.
pub fn kernel(n: usize, opts: &BemOpts) -> GeneralCsc<C> {
    let (p, nrm) = surface_points(n, opts.rough, opts.seed);
    let i_unit = Complex::new(0.0, 1.0);
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    let cutoff2 = opts.cutoff * opts.cutoff;
    for i in 0..n {
        // Diagonal self-term.
        rows.push(i);
        cols.push(i);
        vals.push(opts.self_term);
        let ni = nrm[i]; // observation normal (unit direction)
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = [p[i][0] - p[j][0], p[i][1] - p[j][1], p[i][2] - p[j][2]];
            let r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            if r2 > cutoff2 {
                continue; // near-field trim ⇒ sparse
            }
            let r = r2.sqrt();
            let g = (i_unit * opts.k * r).exp() / (4.0 * std::f64::consts::PI * r);
            let dot = dx[0] * ni[0] + dx[1] * ni[1] + dx[2] * ni[2];
            // ∂G/∂n_i = G (ik − 1/r) (dx·n_i)/r
            let a = g * (i_unit * opts.k - Complex::new(1.0 / r, 0.0)) * Complex::new(dot / r, 0.0);
            rows.push(i);
            cols.push(j);
            vals.push(a);
        }
    }
    super::build_gen(n, &rows, &cols, &vals)
}

pub(super) fn add_to_catalog(c: &mut Vec<MatrixSpec>) {
    fn un(m: GeneralCsc<C>) -> Generated {
        Generated::Unsymmetric(m)
    }
    // Near-field (sparse) MoM block, well-conditioned.
    c.push(MatrixSpec {
        name: "mom_nearfield",
        structure: Structure::Bem,
        symmetry: Symmetry::Unsymmetric,
        cond: Cond::Moderate,
        density: Density::Medium,
        size: 8_000,
        build: || un(kernel(8_000, &BemOpts::default())),
    });
    // Larger, sparser near-field block.
    c.push(MatrixSpec {
        name: "mom_nearfield_large",
        structure: Structure::Bem,
        symmetry: Symmetry::Unsymmetric,
        cond: Cond::Moderate,
        density: Density::Sparse,
        size: 30_000,
        build: || {
            un(kernel(
                30_000,
                &BemOpts {
                    cutoff: 0.25,
                    ..Default::default()
                },
            ))
        },
    });
    // High wavenumber near a sphere resonance ⇒ ill-conditioned.
    c.push(MatrixSpec {
        name: "mom_resonant",
        structure: Structure::Bem,
        symmetry: Symmetry::Unsymmetric,
        cond: Cond::Ill,
        density: Density::Medium,
        size: 8_000,
        build: || {
            let o = BemOpts {
                k: 8.0,
                self_term: Complex::new(0.5, 1e-3),
                ..Default::default()
            };
            un(kernel(8_000, &o))
        },
    });
    // Denser block (larger cutoff ⇒ more nonzeros per row).
    c.push(MatrixSpec {
        name: "mom_dense",
        structure: Structure::Bem,
        symmetry: Symmetry::Unsymmetric,
        cond: Cond::Moderate,
        density: Density::Dense,
        size: 4_000,
        build: || {
            un(kernel(
                4_000,
                &BemOpts {
                    cutoff: 1.0,
                    ..Default::default()
                },
            ))
        },
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_is_unsymmetric_and_sparse() {
        let a = kernel(500, &BemOpts::default());
        assert_eq!(a.n, 500);
        // Near-field trim ⇒ far from dense.
        assert!(
            a.values.len() < 500 * 500 / 4,
            "near-field trim keeps it sparse"
        );
        // Check A_ij ≠ A_ji for some off-diagonal pair (double-layer asymmetry).
        let mut asym = false;
        for j in 0..a.n {
            for k in a.col_ptr[j]..a.col_ptr[j + 1] {
                let i = a.row_idx[k];
                if i == j {
                    continue;
                }
                // find A_ji
                let ji = (a.col_ptr[i]..a.col_ptr[i + 1]).find(|&t| a.row_idx[t] == j);
                if let Some(t) = ji {
                    if (a.values[k] - a.values[t]).norm() > 1e-9 {
                        asym = true;
                    }
                }
            }
        }
        assert!(asym, "double-layer kernel must be unsymmetric");
    }

    #[test]
    fn cutoff_controls_density() {
        let sparse = kernel(
            800,
            &BemOpts {
                cutoff: 0.2,
                ..Default::default()
            },
        );
        let dense = kernel(
            800,
            &BemOpts {
                cutoff: 1.0,
                ..Default::default()
            },
        );
        assert!(
            dense.values.len() > 3 * sparse.values.len(),
            "larger cutoff ⇒ denser"
        );
    }
}
